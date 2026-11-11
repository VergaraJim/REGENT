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

#[cfg(feature = "audio")]
use std::collections::VecDeque;
#[cfg(feature = "audio")]
use std::sync::Mutex;

use regent_core::voice::AudioOutFactory;

#[cfg(not(feature = "audio"))]
use super::NO_AUDIO;

/// How a callback writes one mono sample into every channel.
#[cfg(feature = "audio")]
fn write_chunk<T: Copy>(
    data: &mut [T],
    channels: usize,
    queue: &Mutex<VecDeque<f32>>,
    convert: impl Fn(f32) -> T,
) {
    let channels = channels.max(1);
    let Ok(mut queue) = queue.try_lock() else {
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
    use std::collections::VecDeque;
    use std::sync::mpsc::{self, Receiver, SyncSender};
    use std::sync::{Arc, Mutex};
    use std::thread::{self, JoinHandle};

    use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
    use cpal::SampleFormat;
    use regent_core::voice::{AudioOut, Resampler};

    use crate::audio::{f32_to_i16, f32_to_u16, DEVICE_OPEN_TIMEOUT, NO_SPEAKERS};

    use super::write_chunk;
    use regent_core::voice::AUDIO_FAILED;

    /// What the output thread reports once the device is playing.
    struct DeviceInfo {
        sample_rate: u32,
    }

    /// A playing output device: the shared queue, the stream thread and the
    /// signal that stops it.
    pub struct CpalAudioOut {
        queue: Arc<Mutex<VecDeque<f32>>>,
        device_rate: u32,
        stop_tx: Option<SyncSender<()>>,
        stream: Option<JoinHandle<()>>,
    }

    impl AudioOut for CpalAudioOut {
        fn play(&mut self, samples: &[f32], sample_rate: u32) -> Result<(), String> {
            let mut resampler = Resampler::new(sample_rate, self.device_rate);
            let mut converted = Vec::new();
            resampler.process(samples, &mut converted);
            if let Ok(mut queue) = self.queue.lock() {
                queue.extend(converted);
            }
            Ok(())
        }

        fn stop(&mut self) {
            if let Ok(mut queue) = self.queue.lock() {
                queue.clear();
            }
        }

        fn is_idle(&self) -> bool {
            self.queue
                .lock()
                .map(|queue| queue.is_empty())
                .unwrap_or(true)
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
        let queue = Arc::new(Mutex::new(VecDeque::new()));
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
        queue: Arc<Mutex<VecDeque<f32>>>,
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

        let error_callback = move |_error: cpal::StreamError| {};

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
