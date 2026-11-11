//! Capturing the default microphone with `cpal`.
//!
//! # Threading
//!
//! `cpal::Stream` is not `Send` on every platform, so the stream is built,
//! started and held on a thread of its own. Opening is asynchronous: the
//! factory spawns that thread, waits for it to report the device's channel
//! count and sample rate, and gives up after [`DEVICE_OPEN_TIMEOUT`].
//!
//! The realtime callback never touches the [`MicCallbacks`] and never blocks:
//! it converts the device buffer to `f32` and `try_send`s it, dropping the
//! chunk when the bounded queue is full. A second thread consumes that queue
//! at leisure, downmixes to mono, resamples to [`STT_SAMPLE_RATE`] and only
//! then calls `on_samples`.
//!
//! Dropping the returned stream closes the device: it signals the stream
//! thread, which drops the `cpal::Stream`, and then joins both threads.

use regent_core::voice::MicFactory;

#[cfg(not(feature = "audio"))]
use super::NO_AUDIO;
#[cfg(not(feature = "audio"))]
use regent_core::voice::{MicCallbacks, MicStream};

#[cfg(feature = "audio")]
mod imp {
    use std::sync::mpsc::{self, Receiver, SyncSender};
    use std::thread::{self, JoinHandle};

    use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
    use cpal::SampleFormat;
    use regent_core::voice::{downmix, MicCallbacks, MicStream, Resampler, STT_SAMPLE_RATE};

    use crate::audio::{
        i16_to_f32, mic_error_message, u16_to_f32, DEVICE_OPEN_TIMEOUT, MIC_LOST, MIC_NONE,
        MIC_TIMEOUT,
    };

    /// How many conversion buffers may wait for the consumer thread.
    const MIC_QUEUE_DEPTH: usize = 32;

    /// What the realtime callback hands to the consumer thread.
    enum MicEvent {
        /// Interleaved device samples.
        Samples(Vec<f32>),
        /// A device error, reported by cpal's error callback. The raw text is
        /// kept for diagnostics; the service always sees [`MIC_LOST`].
        Error(#[allow(dead_code)] String),
    }

    /// What the stream thread reports once the device is playing.
    struct DeviceInfo {
        channels: usize,
        sample_rate: u32,
    }

    /// An open microphone: the two worker threads and the signal to stop them.
    pub struct CpalMic {
        stop_tx: Option<SyncSender<()>>,
        stream: Option<JoinHandle<()>>,
        consumer: Option<JoinHandle<()>>,
    }

    impl MicStream for CpalMic {}

    impl Drop for CpalMic {
        fn drop(&mut self) {
            drop(self.stop_tx.take());
            if let Some(stream) = self.stream.take() {
                let _ = stream.join();
            }
            if let Some(consumer) = self.consumer.take() {
                let _ = consumer.join();
            }
        }
    }

    /// Opens the default input device and starts both threads.
    pub fn open_mic(callbacks: MicCallbacks) -> Result<Box<dyn MicStream>, String> {
        let (event_tx, event_rx) = mpsc::sync_channel(MIC_QUEUE_DEPTH);
        let (open_tx, open_rx) = mpsc::channel();
        let (stop_tx, stop_rx) = mpsc::sync_channel(1);

        let stream = thread::spawn(move || run_stream(open_tx, stop_rx, event_tx));

        let info = match open_rx.recv_timeout(DEVICE_OPEN_TIMEOUT) {
            Ok(Ok(info)) => info,
            Ok(Err(message)) => {
                drop(stop_tx);
                let _ = stream.join();
                return Err(message);
            }
            // The stream thread is still opening the device; asking it to
            // stop makes it close the device as soon as it is open. It is not
            // joined, so a device that hangs forever cannot block the caller.
            Err(_) => {
                drop(stop_tx);
                return Err(MIC_TIMEOUT.to_string());
            }
        };

        let consumer = thread::spawn(move || run_consumer(callbacks, info, event_rx));

        Ok(Box::new(CpalMic {
            stop_tx: Some(stop_tx),
            stream: Some(stream),
            consumer: Some(consumer),
        }))
    }

    /// Builds, starts and holds the input stream, then waits for the stop signal.
    fn run_stream(
        open_tx: mpsc::Sender<Result<DeviceInfo, String>>,
        stop_rx: Receiver<()>,
        events: SyncSender<MicEvent>,
    ) {
        let host = cpal::default_host();
        let Some(device) = host.default_input_device() else {
            let _ = open_tx.send(Err(MIC_NONE.to_string()));
            return;
        };
        let config = match device.default_input_config() {
            Ok(config) => config,
            Err(error) => {
                let _ = open_tx.send(Err(mic_error_message(&error.to_string()).to_string()));
                return;
            }
        };

        let format = config.sample_format();
        let config: cpal::StreamConfig = config.into();
        let channels = config.channels as usize;
        let sample_rate = config.sample_rate.0;

        let error_events = events.clone();
        let error_callback = move |error: cpal::StreamError| {
            let _ = error_events.try_send(MicEvent::Error(error.to_string()));
        };

        let built = match format {
            SampleFormat::F32 => {
                let events = events.clone();
                device.build_input_stream(
                    &config,
                    move |data: &[f32], _: &cpal::InputCallbackInfo| {
                        send_chunk(&events, data.to_vec());
                    },
                    error_callback,
                    None,
                )
            }
            SampleFormat::I16 => {
                let events = events.clone();
                device.build_input_stream(
                    &config,
                    move |data: &[i16], _: &cpal::InputCallbackInfo| {
                        send_chunk(&events, data.iter().copied().map(i16_to_f32).collect());
                    },
                    error_callback,
                    None,
                )
            }
            SampleFormat::U16 => {
                let events = events.clone();
                device.build_input_stream(
                    &config,
                    move |data: &[u16], _: &cpal::InputCallbackInfo| {
                        send_chunk(&events, data.iter().copied().map(u16_to_f32).collect());
                    },
                    error_callback,
                    None,
                )
            }
            other => {
                let _ = open_tx.send(Err(mic_error_message(&format!("{other:?}")).to_string()));
                return;
            }
        };

        let stream = match built {
            Ok(stream) => stream,
            Err(error) => {
                let _ = open_tx.send(Err(mic_error_message(&error.to_string()).to_string()));
                return;
            }
        };
        if let Err(error) = stream.play() {
            let _ = open_tx.send(Err(mic_error_message(&error.to_string()).to_string()));
            return;
        }
        if open_tx
            .send(Ok(DeviceInfo {
                channels,
                sample_rate,
            }))
            .is_err()
        {
            // The factory gave up waiting; close the device again.
            return;
        }

        let _ = stop_rx.recv();
        drop(stream);
    }

    /// Hands one converted buffer to the consumer, dropping it when full.
    fn send_chunk(events: &SyncSender<MicEvent>, chunk: Vec<f32>) {
        if chunk.is_empty() {
            return;
        }
        let _ = events.try_send(MicEvent::Samples(chunk));
    }

    /// Downmixes, resamples and delivers the device buffers.
    fn run_consumer(callbacks: MicCallbacks, info: DeviceInfo, events: Receiver<MicEvent>) {
        let MicCallbacks {
            mut on_samples,
            mut on_error,
        } = callbacks;
        let mut resampler = Resampler::new(info.sample_rate, STT_SAMPLE_RATE);
        let mut mono = Vec::new();
        let mut converted = Vec::new();

        while let Ok(event) = events.recv() {
            match event {
                MicEvent::Samples(chunk) => {
                    mono.clear();
                    downmix(&chunk, info.channels, &mut mono);
                    converted.clear();
                    resampler.process(&mono, &mut converted);
                    if !converted.is_empty() {
                        on_samples(&converted);
                    }
                }
                MicEvent::Error(_) => {
                    // Reported once: the consumer stops for good here, so no
                    // later error can reach the service.
                    on_error(MIC_LOST.to_string());
                    return;
                }
            }
        }
    }
}

#[cfg(feature = "audio")]
pub use imp::open_mic;

/// A microphone factory backed by the default input device.
#[cfg(feature = "audio")]
pub fn mic_factory() -> MicFactory {
    Box::new(imp::open_mic)
}

/// A microphone factory for builds without audio support.
#[cfg(not(feature = "audio"))]
pub fn mic_factory() -> MicFactory {
    Box::new(
        |_callbacks: MicCallbacks| -> Result<Box<dyn MicStream>, String> {
            Err(NO_AUDIO.to_string())
        },
    )
}
