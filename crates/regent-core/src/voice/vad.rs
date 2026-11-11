//! Energy-based voice activity detection.
//!
//! [`EnergyVad`] buffers mono input into 20 ms frames, compares each frame's
//! RMS with an adaptive threshold and cuts utterances out of the stream. It
//! implements [`VoiceDetector`] so a model-based detector can replace it
//! without touching the voice service.
//!
//! Speech starts after [`START_FRAMES`] consecutive frames above the
//! threshold and ends after [`HANGOVER_FRAMES`] frames below it. The utterance
//! carries [`PRE_ROLL_FRAMES`] of history so its first syllable is not cut.
//! The noise floor is the minimum frame RMS over the last
//! [`NOISE_WINDOW_FRAMES`] idle frames; frames inside speech and the start
//! debounce never feed it.

use std::collections::VecDeque;

/// Samples in one analysis frame: 20 ms at 16 kHz.
pub const FRAME_SAMPLES: usize = 320;

/// How many idle frames the noise floor is measured over: 2 s.
pub const NOISE_WINDOW_FRAMES: usize = 100;

/// Frames at the start of a stream that never start speech: 200 ms.
pub const CALIBRATION_FRAMES: usize = 10;

/// Consecutive above-threshold frames needed to start speech: 200 ms.
pub const START_FRAMES: usize = 10;

/// Frames of audio kept before speech starts: 300 ms.
pub const PRE_ROLL_FRAMES: usize = 15;

/// Below-threshold frames that end an utterance: 700 ms.
pub const HANGOVER_FRAMES: usize = 35;

/// The longest utterance, in frames: 30 s.
pub const MAX_UTTERANCE_FRAMES: usize = 1500;

/// The noise floor is multiplied by this to get the threshold.
pub const THRESHOLD_FACTOR: f32 = 3.0;

/// The lowest threshold, so a silent room still needs some energy.
pub const MIN_THRESHOLD: f32 = 0.005;

/// The largest noise floor the threshold is derived from.
///
/// The first frames of a stream are calibration and feed the noise window
/// unconditionally. A stream that starts with loud audio (someone speaking
/// the moment the mic opens) would otherwise set a floor as high as speech
/// and go deaf for the rest of the session, so the floor is capped.
const MAX_NOISE_FLOOR: f32 = 0.02;

/// One detected utterance.
#[derive(Debug, Clone, PartialEq)]
pub struct Utterance {
    /// The mono 16 kHz samples of the utterance.
    pub samples: Vec<f32>,
    /// The RMS level over the whole utterance.
    pub rms: f32,
    /// The length of the utterance in milliseconds.
    pub duration_ms: u64,
}

/// The root-mean-square level of `samples`; 0 for empty input.
pub fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum: f32 = samples.iter().map(|sample| sample * sample).sum();
    (sum / samples.len() as f32).sqrt()
}

/// The interface voice input uses, so a model-based detector can replace the
/// energy one (spec §5.8).
pub trait VoiceDetector: Send {
    /// Feeds mono 16 kHz `samples`, appending any finished utterances to `out`.
    fn push(&mut self, samples: &[f32], out: &mut Vec<Utterance>);

    /// Drops any partial utterance and its counters.
    ///
    /// The learned noise floor is kept, so a detector is not re-calibrated
    /// after, say, the mic is switched off and on again.
    fn reset(&mut self);

    /// Whether a possible or confirmed utterance is in progress.
    fn is_active(&self) -> bool;

    /// The RMS of the last complete frame.
    fn last_level(&self) -> f32;
}

/// An energy-based [`VoiceDetector`].
#[derive(Debug, Clone)]
pub struct EnergyVad {
    /// The partly filled frame waiting for more samples.
    partial: Vec<f32>,
    /// RMS values of the last quiet frames.
    noise: VecDeque<f32>,
    /// How many whole frames have been processed since construction.
    frames_seen: u64,
    /// The last `PRE_ROLL_FRAMES` frames, oldest first.
    pre_roll: VecDeque<Vec<f32>>,
    /// Consecutive above-threshold frames while idle.
    start_count: usize,
    /// Consecutive below-threshold frames while in speech.
    silence_count: usize,
    /// The frames of the utterance in progress.
    utterance: Vec<f32>,
    /// Whether a confirmed utterance is in progress.
    in_speech: bool,
    /// The RMS of the last complete frame.
    last_level: f32,
}

impl EnergyVad {
    /// A detector with no state and an empty noise window.
    pub fn new() -> EnergyVad {
        EnergyVad::default()
    }

    /// The threshold for the current noise floor.
    fn threshold(&self) -> f32 {
        let floor = self.noise.iter().copied().reduce(f32::min);
        floor.map_or(MIN_THRESHOLD, |floor| {
            (floor.min(MAX_NOISE_FLOOR) * THRESHOLD_FACTOR).max(MIN_THRESHOLD)
        })
    }

    fn push_noise(&mut self, level: f32) {
        if self.noise.len() == NOISE_WINDOW_FRAMES {
            self.noise.pop_front();
        }
        self.noise.push_back(level);
    }

    fn push_pre_roll(&mut self, frame: &[f32]) {
        if self.pre_roll.len() == PRE_ROLL_FRAMES {
            self.pre_roll.pop_front();
        }
        self.pre_roll.push_back(frame.to_vec());
    }

    /// Ends the utterance in progress and reports it.
    fn finish(&mut self, out: &mut Vec<Utterance>) {
        let samples = std::mem::take(&mut self.utterance);
        out.push(Utterance {
            rms: rms(&samples),
            duration_ms: samples.len() as u64 * 1000 / 16_000,
            samples,
        });
        self.in_speech = false;
        self.start_count = 0;
        self.silence_count = 0;
        self.pre_roll.clear();
    }

    fn process_frame(&mut self, frame: &[f32], out: &mut Vec<Utterance>) {
        let level = rms(frame);
        self.last_level = level;
        let index = self.frames_seen;
        self.frames_seen += 1;
        let threshold = self.threshold();

        if self.in_speech {
            self.utterance.extend_from_slice(frame);
            if level > threshold {
                self.silence_count = 0;
            } else {
                self.silence_count += 1;
            }
            let frames = self.utterance.len() / FRAME_SAMPLES;
            if self.silence_count >= HANGOVER_FRAMES || frames >= MAX_UTTERANCE_FRAMES {
                self.finish(out);
            }
            return;
        }

        self.push_pre_roll(frame);

        if index < CALIBRATION_FRAMES as u64 {
            self.start_count = 0;
            self.push_noise(level);
            return;
        }

        if level > threshold {
            self.start_count += 1;
            if self.start_count >= START_FRAMES {
                let ring = std::mem::take(&mut self.pre_roll);
                self.utterance = Vec::with_capacity(ring.len() * FRAME_SAMPLES);
                for frame in ring {
                    self.utterance.extend_from_slice(&frame);
                }
                self.in_speech = true;
                self.start_count = 0;
                self.silence_count = 0;
                let frames = self.utterance.len() / FRAME_SAMPLES;
                if frames >= MAX_UTTERANCE_FRAMES {
                    self.finish(out);
                }
            }
        } else {
            self.start_count = 0;
            self.push_noise(level);
        }
    }
}

impl Default for EnergyVad {
    fn default() -> EnergyVad {
        EnergyVad {
            partial: Vec::with_capacity(FRAME_SAMPLES),
            noise: VecDeque::with_capacity(NOISE_WINDOW_FRAMES),
            frames_seen: 0,
            pre_roll: VecDeque::with_capacity(PRE_ROLL_FRAMES),
            start_count: 0,
            silence_count: 0,
            utterance: Vec::new(),
            in_speech: false,
            last_level: 0.0,
        }
    }
}

impl VoiceDetector for EnergyVad {
    fn push(&mut self, samples: &[f32], out: &mut Vec<Utterance>) {
        let mut rest = samples;
        while !rest.is_empty() {
            let room = FRAME_SAMPLES - self.partial.len();
            let take = room.min(rest.len());
            self.partial.extend_from_slice(&rest[..take]);
            rest = &rest[take..];

            if self.partial.len() == FRAME_SAMPLES {
                let frame = std::mem::take(&mut self.partial);
                self.process_frame(&frame, out);
                self.partial = frame;
                self.partial.clear();
            }
        }
    }

    fn reset(&mut self) {
        self.partial.clear();
        self.start_count = 0;
        self.silence_count = 0;
        self.utterance.clear();
        self.pre_roll.clear();
        self.in_speech = false;
        self.last_level = 0.0;
    }

    fn is_active(&self) -> bool {
        self.in_speech || self.start_count > 0
    }

    fn last_level(&self) -> f32 {
        self.last_level
    }
}

/// Deterministic signals for the voice tests, reusable by later tasks.
#[cfg(test)]
pub(crate) mod test_signals {
    use std::f32::consts::TAU;

    /// Pseudo-random noise in `[-amplitude, amplitude]` from a small LCG.
    pub fn noise(amplitude: f32, len: usize) -> Vec<f32> {
        let mut state: u32 = 0x1234_5678;
        (0..len)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let unit = state as f32 / u32::MAX as f32 * 2.0 - 1.0;
                unit * amplitude
            })
            .collect()
    }

    /// A sine tone of `amplitude` at `freq` Hz, sampled at `rate` Hz.
    pub fn tone(amplitude: f32, freq: f32, rate: u32, len: usize) -> Vec<f32> {
        (0..len)
            .map(|i| amplitude * (TAU * freq * i as f32 / rate as f32).sin())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::test_signals::{noise, tone};
    use super::*;

    const RATE: u32 = 16_000;

    fn seconds(seconds: f32) -> usize {
        (seconds * RATE as f32) as usize
    }

    fn detect(signal: &[f32]) -> Vec<Utterance> {
        let mut vad = EnergyVad::new();
        let mut out = Vec::new();
        vad.push(signal, &mut out);
        out
    }

    fn detect_chunked(signal: &[f32], chunk: usize) -> Vec<Utterance> {
        let mut vad = EnergyVad::new();
        let mut out = Vec::new();
        for part in signal.chunks(chunk) {
            vad.push(part, &mut out);
        }
        out
    }

    #[test]
    fn rms_of_an_empty_slice_is_zero() {
        assert_eq!(rms(&[]), 0.0);
    }

    #[test]
    fn a_tone_between_silences_makes_one_utterance() {
        let mut signal = noise(0.001, seconds(1.0));
        signal.extend(tone(0.3, 440.0, RATE, seconds(1.0)));
        signal.extend(noise(0.001, seconds(1.5)));

        let utterances = detect(&signal);
        assert_eq!(utterances.len(), 1);
        let ms = utterances[0].duration_ms;
        assert!((1600..=2100).contains(&ms), "duration {ms} ms");
    }

    #[test]
    fn a_short_burst_makes_no_utterance() {
        let mut signal = noise(0.001, seconds(1.0));
        signal.extend(tone(0.3, 440.0, RATE, seconds(0.1)));
        signal.extend(noise(0.001, seconds(1.0)));
        assert!(detect(&signal).is_empty());
    }

    #[test]
    fn an_endless_tone_is_capped_at_thirty_seconds() {
        let signal = tone(0.3, 440.0, RATE, seconds(35.0));
        let utterances = detect(&signal);
        assert!(!utterances.is_empty());
        assert_eq!(utterances[0].duration_ms, 30_000);
    }

    #[test]
    fn chunking_does_not_change_the_utterances() {
        let mut signal = noise(0.001, seconds(1.0));
        signal.extend(tone(0.3, 440.0, RATE, seconds(1.0)));
        signal.extend(noise(0.001, seconds(1.5)));

        let whole = detect(&signal);
        for chunk in [1, 160, 333] {
            assert_eq!(detect_chunked(&signal, chunk), whole, "chunk {chunk}");
        }
    }

    #[test]
    fn steady_loud_noise_is_calibrated_away() {
        let mut vad = EnergyVad::new();
        let mut out = Vec::new();
        vad.push(&noise(0.02, seconds(3.0)), &mut out);
        assert!(out.is_empty());

        vad.push(&tone(0.3, 440.0, RATE, seconds(1.0)), &mut out);
        assert!(vad.is_active());
        vad.push(&noise(0.02, seconds(1.0)), &mut out);
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn is_active_tracks_speech() {
        let mut vad = EnergyVad::new();
        let mut out = Vec::new();
        vad.push(&noise(0.001, seconds(0.1)), &mut out);
        assert!(!vad.is_active());

        vad.push(&tone(0.3, 440.0, RATE, seconds(1.0)), &mut out);
        assert!(vad.is_active());

        vad.push(&noise(0.001, seconds(1.0)), &mut out);
        assert!(!vad.is_active());
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn reset_drops_a_partial_utterance() {
        let mut vad = EnergyVad::new();
        let mut out = Vec::new();
        vad.push(&tone(0.3, 440.0, RATE, seconds(1.0)), &mut out);
        assert!(vad.is_active());

        vad.reset();
        assert!(!vad.is_active());

        vad.push(&noise(0.001, seconds(1.0)), &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn the_last_level_is_tracked() {
        let mut vad = EnergyVad::new();
        let mut out = Vec::new();
        assert_eq!(vad.last_level(), 0.0);

        let signal = tone(0.3, 440.0, RATE, FRAME_SAMPLES);
        vad.push(&signal, &mut out);
        assert!((vad.last_level() - rms(&signal)).abs() < 1e-6);
    }
}
