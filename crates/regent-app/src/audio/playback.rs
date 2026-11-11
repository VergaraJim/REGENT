//! Playing mono audio on the default output device with `cpal`.
//!
//! # Threading
//!
//! As with capture, `cpal::Stream` is not `Send` on every platform, so the
//! stream is built, started and held on a dedicated thread. The factory waits
//! for that thread to report the device's sample rate and channel count, and
//! gives up after [`DEVICE_OPEN_TIMEOUT`].
//!
//! Samples live in a queue shared with the realtime callback, which only ever
//! `try_lock`s it: when the queue is busy or empty it writes silence, so the
//! audio thread never blocks. [`AudioOut::play`] resamples to the device rate
//! and appends to the queue; [`AudioOut::stop`] clears it; [`AudioOut::is_idle`]
//! reports whether anything is left.
//!
//! # Failed devices
//!
//! When the device disappears mid-playback (headphones unplugged, for example)
//! `cpal` stops calling the realtime callback and reports the problem to a
//! separate error callback instead. That callback only marks the queue broken:
//! it never blocks. A broken queue then drains itself and reports idle, so the
//! voice service stops waiting for audio, and refuses further playback with
//! [`AUDIO_FAILED`], so the player drops the device and opens a fresh one for
//! the next reply.

#[cfg(feature = "audio")]
use std::collections::VecDeque;
#[cfg(feature = "audio")]
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(feature = "audio")]
use std::sync::{Mutex, MutexGuard};

use regent_core::voice::AudioOutFactory;

#[cfg(not(feature = "audio"))]
use super::NO_AUDIO;

#[cfg(feature = "audio")]
use regent_core::voice::AUDIO_FAILED;

/// The sample queue shared with the realtime callback, plus whether the device
/// has failed.
///
/// A device that reports an error is *broken*: its callback has stopped, so
/// nothing would ever drain the queue. [`OutputQueue::is_idle`] therefore
/// empties it and reports idle, and [`OutputQueue::push`] refuses new samples,
/// so playback never waits on a device that cannot play.
#[cfg(feature = "audio")]
#[derive(Default)]
struct OutputQueue {
    samples: Mutex<VecDeque<f32>>,
    broken: AtomicBool,
}

#[cfg(feature = "audio")]
impl OutputQueue {
    fn samples(&self) -> MutexGuard<'_, VecDeque<f32>> {
        // A panic while holding the lock must not take playback down with it.
        self.samples
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Appends mono samples, unless the device has failed.
    fn push(&self, samples: impl IntoIterator<Item = f32>) -> Result<(), String> {
        if self.is_broken() {
            return Err(AUDIO_FAILED.to_string());
        }
        self.samples().extend(samples);
        Ok(())
    }

    /// Drops every queued sample.
    fn clear(&self) {
        self.samples().clear();
    }

    /// Whether nothing is left to play; a broken device is always idle.
    fn is_idle(&self) -> bool {
        if self.is_broken() {
            self.clear();
            return true;
        }
        self.samples().is_empty()
    }

    /// Records that the device stopped working; called from the error callback.
    fn mark_broken(&self) {
        self.broken.store(true, Ordering::SeqCst);
    }

    /// Whether the device stopped working.
    fn is_broken(&self) -> bool {
        self.broken.load(Ordering::SeqCst)
    }
}

/// How a callback writes one mono sample into every channel.
#[cfg(feature = "audio")]
fn write_chunk<T: Copy>(
    data: &mut [T],
    channels: usize,
    queue: &OutputQueue,
    convert: impl Fn(f32) -> T,
) {
    let channels = channels.max(1);
    let Ok(mut queue) = queue.samples.try_lock() else {
        data.fill(convert(0.0));
        return;
    };
    for frame in data.chunks_mut(channels) {
        let value = convert(queue.pop_front().unwrap_or(0.0));
        for out in frame.iter_mut() {
            *out = value;
        }
    }
}

#[cfg(feature = "audio")]
mod imp {
    use std::sync::mpsc::{self, Receiver, SyncSender};
    use std::sync::Arc;
    use std::thread::{self, JoinHandle};

    use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
    use cpal::SampleFormat;
    use regent_core::voice::{AudioOut, Resampler};

    use crate::audio::{f32_to_i16, f32_to_u16, DEVICE_OPEN_TIMEOUT, NO_SPEAKERS};

    use super::{write_chunk, OutputQueue};
    use regent_core::voice::AUDIO_FAILED;

    /// What the output thread reports once the device is playing.
    struct DeviceInfo {
        sample_rate: u32,
    }

    /// A playing output device: the shared queue, the stream thread and the
    /// signal that stops it.
    pub struct CpalAudioOut {
        queue: Arc<OutputQueue>,
        device_rate: u32,
        stop_tx: Option<SyncSender<()>>,
        stream: Option<JoinHandle<()>>,
    }

    impl AudioOut for CpalAudioOut {
        fn play(&mut self, samples: &[f32], sample_rate: u32) -> Result<(), String> {
            let mut resampler = Resampler::new(sample_rate, self.device_rate);
            let mut converted = Vec::new();
            resampler.process(samples, &mut converted);
            self.queue.push(converted)
        }

        fn stop(&mut self) {
            self.queue.clear();
        }

        fn is_idle(&self) -> bool {
            self.queue.is_idle()
        }
    }

    impl Drop for CpalAudioOut {
        fn drop(&mut self) {
            drop(self.stop_tx.take());
            if let Some(stream) = self.stream.take() {
                let _ = stream.join();
            }
        }
    }

    /// Opens the default output device and starts the stream thread.
    pub fn open_output() -> Result<Box<dyn AudioOut>, String> {
        let queue = Arc::new(OutputQueue::default());
        let (open_tx, open_rx) = mpsc::channel();
        let (stop_tx, stop_rx) = mpsc::sync_channel(1);

        let thread_queue = Arc::clone(&queue);
        let stream = thread::spawn(move || run_stream(open_tx, stop_rx, thread_queue));

        let info = match open_rx.recv_timeout(DEVICE_OPEN_TIMEOUT) {
            Ok(Ok(info)) => info,
            Ok(Err(message)) => {
                drop(stop_tx);
                let _ = stream.join();
                return Err(message);
            }
            // Opening took too long; the thread closes the device as soon as
            // it is open. It is not joined, so a hang cannot block the caller.
            Err(_) => {
                drop(stop_tx);
                return Err(AUDIO_FAILED.to_string());
            }
        };

        Ok(Box::new(CpalAudioOut {
            queue,
            device_rate: info.sample_rate,
            stop_tx: Some(stop_tx),
            stream: Some(stream),
        }))
    }

    /// Builds, starts and holds the output stream, then waits for the stop signal.
    fn run_stream(
        open_tx: mpsc::Sender<Result<DeviceInfo, String>>,
        stop_rx: Receiver<()>,
        queue: Arc<OutputQueue>,
    ) {
        let host = cpal::default_host();
        let Some(device) = host.default_output_device() else {
            let _ = open_tx.send(Err(NO_SPEAKERS.to_string()));
            return;
        };
        let config = match device.default_output_config() {
            Ok(config) => config,
            Err(_) => {
                let _ = open_tx.send(Err(AUDIO_FAILED.to_string()));
                return;
            }
        };

        let format = config.sample_format();
        let config: cpal::StreamConfig = config.into();
        let channels = config.channels as usize;
        let sample_rate = config.sample_rate.0;

        // The error callback runs on an audio thread, so it only stores a flag.
        let errors = Arc::clone(&queue);
        let error_callback = move |_error: cpal::StreamError| errors.mark_broken();

        let built = match format {
            SampleFormat::F32 => {
                let queue = Arc::clone(&queue);
                device.build_output_stream(
                    &config,
                    move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                        write_chunk(data, channels, &queue, |sample| sample);
                    },
                    error_callback,
                    None,
                )
            }
            SampleFormat::I16 => {
                let queue = Arc::clone(&queue);
                device.build_output_stream(
                    &config,
                    move |data: &mut [i16], _: &cpal::OutputCallbackInfo| {
                        write_chunk(data, channels, &queue, f32_to_i16);
                    },
                    error_callback,
                    None,
                )
            }
            SampleFormat::U16 => {
                let queue = Arc::clone(&queue);
                device.build_output_stream(
                    &config,
                    move |data: &mut [u16], _: &cpal::OutputCallbackInfo| {
                        write_chunk(data, channels, &queue, f32_to_u16);
                    },
                    error_callback,
                    None,
                )
            }
            _ => {
                let _ = open_tx.send(Err(AUDIO_FAILED.to_string()));
                return;
            }
        };

        let stream = match built {
            Ok(stream) => stream,
            Err(_) => {
                let _ = open_tx.send(Err(AUDIO_FAILED.to_string()));
                return;
            }
        };
        if stream.play().is_err() {
            let _ = open_tx.send(Err(AUDIO_FAILED.to_string()));
            return;
        }
        if open_tx.send(Ok(DeviceInfo { sample_rate })).is_err() {
            // The factory gave up waiting; close the device again.
            return;
        }

        let _ = stop_rx.recv();
        drop(stream);
    }
}

#[cfg(feature = "audio")]
pub use imp::open_output;

/// An output factory backed by the default output device.
#[cfg(feature = "audio")]
pub fn audio_out_factory() -> AudioOutFactory {
    Box::new(imp::open_output)
}

/// An output factory for builds without audio support.
#[cfg(not(feature = "audio"))]
pub fn audio_out_factory() -> AudioOutFactory {
    Box::new(|| Err(NO_AUDIO.to_string()))
}

#[cfg(all(feature = "audio", test))]
mod tests {
    use super::*;

    #[test]
    fn pushing_samples_makes_the_queue_busy() {
        let queue = OutputQueue::default();
        assert!(queue.is_idle());

        queue.push([0.5, 0.25]).expect("a working device queues");

        assert!(!queue.is_idle());
    }

    #[test]
    fn a_broken_queue_is_idle_and_empty() {
        let queue = OutputQueue::default();
        queue.push([0.5; 8]).expect("a working device queues");

        queue.mark_broken();

        assert!(queue.is_broken());
        assert!(queue.is_idle());
        assert!(queue.samples().is_empty());
    }

    #[test]
    fn a_broken_queue_refuses_playback() {
        let queue = OutputQueue::default();
        queue.mark_broken();

        assert_eq!(queue.push([0.5]), Err(AUDIO_FAILED.to_string()));
        assert!(queue.is_idle());
    }

    #[test]
    fn clearing_drops_queued_samples() {
        let queue = OutputQueue::default();
        queue.push([0.5; 4]).expect("a working device queues");

        queue.clear();

        assert!(queue.is_idle());
        assert!(queue.samples().is_empty());
    }
}
