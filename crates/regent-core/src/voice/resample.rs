//! Mono mixing and streaming sample-rate conversion.
//!
//! The resampler is built for an always-on capture callback: it never
//! allocates (apart from growing `out`), its state is carried from call to
//! call, and feeding the input in any chunking gives exactly the same output
//! as one call, so there are no clicks at buffer boundaries.

/// Appends the mono mix of `interleaved` to `out`, one value per frame.
///
/// `channels` is the number of interleaved channels per frame. With 0 or 1
/// channels the input is copied as-is. A trailing partial frame is ignored.
pub fn downmix(interleaved: &[f32], channels: usize, out: &mut Vec<f32>) {
    if channels <= 1 {
        out.extend_from_slice(interleaved);
        return;
    }
    let frames = interleaved.len() / channels;
    out.reserve(frames);
    for frame in interleaved[..frames * channels].chunks_exact(channels) {
        let sum: f32 = frame.iter().sum();
        out.push(sum / channels as f32);
    }
}

/// A streaming boxcar low-pass, used before downsampling to stop aliasing.
#[derive(Debug, Clone)]
struct MovingAverage {
    buf: Vec<f32>,
    idx: usize,
    sum: f32,
}

impl MovingAverage {
    fn new(len: usize) -> MovingAverage {
        MovingAverage {
            buf: vec![0.0; len],
            idx: 0,
            sum: 0.0,
        }
    }

    /// Feeds one sample and returns the average of the last `len` samples.
    ///
    /// The history starts zero-filled, so the filter has no start-up gap and
    /// its state reproduces exactly across any chunking.
    fn step(&mut self, sample: f32) -> f32 {
        let old = self.buf[self.idx];
        self.buf[self.idx] = sample;
        self.idx = (self.idx + 1) % self.buf.len();
        self.sum += sample - old;
        self.sum / self.buf.len() as f32
    }

    fn reset(&mut self) {
        self.buf.fill(0.0);
        self.idx = 0;
        self.sum = 0.0;
    }
}

/// A streaming mono sample-rate converter.
///
/// Equal rates copy straight through. Otherwise samples are linearly
/// interpolated at exactly `out_rate / in_rate` output samples per input
/// sample. The read position is tracked with integer arithmetic, so it never
/// drifts, and every piece of state is carried across [`Resampler::process`]
/// calls. When downsampling, the input first passes a moving-average low-pass
/// of length `ceil(in_rate / out_rate)`.
#[derive(Debug, Clone)]
struct Converter {
    in_rate: u32,
    out_rate: u32,
    low_pass: Option<MovingAverage>,
    /// Global index of the next input sample to arrive.
    received: u64,
    /// Numerator (over `out_rate`) of the next output sample's position.
    pos_num: u64,
    /// The previous (low-passed) input sample.
    prev: f32,
}

impl Converter {
    fn new(in_rate: u32, out_rate: u32) -> Converter {
        let low_pass = if in_rate > out_rate {
            Some(MovingAverage::new(in_rate.div_ceil(out_rate) as usize))
        } else {
            None
        };
        Converter {
            in_rate,
            out_rate,
            low_pass,
            received: 0,
            pos_num: 0,
            prev: 0.0,
        }
    }

    fn process(&mut self, input: &[f32], out: &mut Vec<f32>) {
        for &sample in input {
            let filtered = match self.low_pass.as_mut() {
                Some(filter) => filter.step(sample),
                None => sample,
            };
            self.push(filtered, out);
        }
    }

    /// Adds one filtered sample at global index `self.received` and emits
    /// every output whose interpolation points are now available.
    fn push(&mut self, sample: f32, out: &mut Vec<f32>) {
        let current = self.received;
        let out_rate = self.out_rate as u64;
        let fraction_scale = self.out_rate as f32;
        loop {
            let base = self.pos_num / out_rate;
            let fraction = (self.pos_num % out_rate) as f32;
            if fraction == 0.0 {
                // Exactly on an input sample: only that sample is needed.
                if base > current {
                    break;
                }
                if base == current {
                    out.push(sample);
                } else if base + 1 == current {
                    out.push(self.prev);
                } else {
                    // The output is emitted the moment its last input sample
                    // arrives, so `base` can never fall further behind.
                    debug_assert!(false, "resampler position fell behind the input");
                    break;
                }
            } else {
                // Interpolate between the previous sample and this one;
                // this output needs input index `base + 1`, which is
                // `current` by construction.
                if base + 1 > current {
                    break;
                }
                debug_assert_eq!(base + 1, current, "resampler position fell behind");
                let t = fraction / fraction_scale;
                out.push(self.prev + (sample - self.prev) * t);
            }
            self.pos_num += self.in_rate as u64;
        }
        self.prev = sample;
        self.received = current + 1;
    }

    fn reset(&mut self) {
        if let Some(filter) = self.low_pass.as_mut() {
            filter.reset();
        }
        self.received = 0;
        self.pos_num = 0;
        self.prev = 0.0;
    }
}

/// A streaming mono resampler.
///
/// Create one per capture stream and feed it with [`Resampler::process`]. A
/// rate of 0 is treated as 1.
#[derive(Debug, Clone)]
pub struct Resampler {
    converter: Converter,
}

impl Resampler {
    /// A resampler from `in_rate` Hz to `out_rate` Hz, both mono.
    pub fn new(in_rate: u32, out_rate: u32) -> Resampler {
        Resampler {
            converter: Converter::new(in_rate.max(1), out_rate.max(1)),
        }
    }

    /// Appends the converted `input` to `out`.
    ///
    /// The result is independent of how the input is split across calls.
    pub fn process(&mut self, input: &[f32], out: &mut Vec<f32>) {
        self.converter.process(input, out);
    }

    /// Drops all carried state, as if the stream were starting again.
    pub fn reset(&mut self) {
        self.converter.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voice::vad::test_signals;

    fn run_whole(in_rate: u32, out_rate: u32, input: &[f32]) -> Vec<f32> {
        let mut out = Vec::new();
        Resampler::new(in_rate, out_rate).process(input, &mut out);
        out
    }

    fn run_chunked(in_rate: u32, out_rate: u32, input: &[f32], chunk: usize) -> Vec<f32> {
        let mut resampler = Resampler::new(in_rate, out_rate);
        let mut out = Vec::new();
        for part in input.chunks(chunk) {
            resampler.process(part, &mut out);
        }
        out
    }

    #[test]
    fn chunking_does_not_change_the_output() {
        let pairs = [
            (48_000, 16_000),
            (44_100, 16_000),
            (22_050, 48_000),
            (16_000, 16_000),
        ];
        for (in_rate, out_rate) in pairs {
            let input = test_signals::tone(1.0, 440.0, in_rate, in_rate as usize);
            let whole = run_whole(in_rate, out_rate, &input);
            assert!(!whole.is_empty(), "{in_rate} -> {out_rate}");
            for chunk in [1, 7, 480] {
                let chunked = run_chunked(in_rate, out_rate, &input, chunk);
                assert_eq!(
                    whole.len(),
                    chunked.len(),
                    "{in_rate} -> {out_rate}, chunk {chunk}"
                );
                for (a, b) in whole.iter().zip(&chunked) {
                    assert!(
                        (a - b).abs() < 1e-6,
                        "{in_rate} -> {out_rate}, chunk {chunk}: {a} vs {b}"
                    );
                }
            }
        }
    }

    #[test]
    fn the_output_length_matches_the_rate_ratio() {
        let pairs = [
            (48_000, 16_000),
            (44_100, 16_000),
            (22_050, 48_000),
            (16_000, 16_000),
        ];
        for (in_rate, out_rate) in pairs {
            let input = test_signals::tone(1.0, 440.0, in_rate, in_rate as usize);
            let out = run_whole(in_rate, out_rate, &input);
            let expected = input.len() as f64 * out_rate as f64 / in_rate as f64;
            let diff = (out.len() as f64 - expected).abs();
            assert!(
                diff <= 2.0,
                "{in_rate} -> {out_rate}: {} vs {expected}",
                out.len()
            );
        }
    }

    #[test]
    fn a_tone_keeps_its_peak() {
        let input = test_signals::tone(1.0, 1_000.0, 48_000, 24_000);
        let out = run_whole(48_000, 16_000, &input);
        let peak = out.iter().fold(0.0_f32, |peak, s| peak.max(s.abs()));
        assert!((0.8..=1.05).contains(&peak), "peak {peak}");
    }

    #[test]
    fn downmix_averages_each_frame() {
        let mut out = Vec::new();
        downmix(&[0.5, -0.5, 0.25, 0.75], 2, &mut out);
        assert_eq!(out.len(), 2);
        assert!((out[0] - 0.0).abs() < 1e-6);
        assert!((out[1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn downmix_ignores_a_partial_frame() {
        let mut out = Vec::new();
        downmix(&[0.5, 0.5, 1.0], 2, &mut out);
        assert_eq!(out, vec![0.5]);
    }

    #[test]
    fn downmix_copies_mono_and_zero_channels() {
        let input = [0.1, -0.2, 0.3];
        let mut out = Vec::new();
        downmix(&input, 1, &mut out);
        assert_eq!(out, input.to_vec());

        out.clear();
        downmix(&input, 0, &mut out);
        assert_eq!(out, input.to_vec());
    }

    #[test]
    fn equal_rates_pass_through() {
        let input: Vec<f32> = (0..100).map(|i| i as f32 / 100.0).collect();
        let out = run_whole(16_000, 16_000, &input);
        assert_eq!(out, input);
    }

    #[test]
    fn asking_for_a_zero_rate_does_not_panic() {
        let out = run_whole(0, 0, &[0.5; 10]);
        assert_eq!(out, vec![0.5; 10]);
    }

    #[test]
    fn reset_restarts_the_stream() {
        let input = test_signals::tone(1.0, 440.0, 48_000, 480);
        let mut resampler = Resampler::new(48_000, 16_000);
        let mut out = Vec::new();
        resampler.process(&input, &mut out);
        resampler.reset();
        let mut again = Vec::new();
        resampler.process(&input, &mut again);
        assert_eq!(out, run_whole(48_000, 16_000, &input));
        assert_eq!(again, out);
    }
}
