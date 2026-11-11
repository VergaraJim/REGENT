//! Engine-agnostic vision: answering questions about what is on the screen.
//!
//! Everything here is pure logic: the real vision engine and the real screen
//! capture are supplied by the app through [`VisionDeps`], so the service and
//! its tests run without a model, a GPU or a display.

use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::time::Duration;

use serde::Serialize;

pub mod fake;
pub mod service;

pub use fake::{fake_capture, FakeCaptureHandle, FakeVision, FakeVisionHandle};
pub use service::VisionService;

/// The most tokens one screen answer may use.
pub const VISION_MAX_TOKENS: usize = 300;

/// What a vision engine needs to load the model.
#[derive(Debug, Clone, PartialEq)]
pub struct VisionConfig {
    /// The vision model file on disk.
    pub model_path: PathBuf,
    /// The multimodal projector file on disk.
    pub projector_path: PathBuf,
    /// How many layers to offload to the GPU.
    pub gpu_layers: u32,
    /// The longest edge, in pixels, a captured screen is downscaled to.
    pub max_image_edge: u32,
    /// How long the loaded model stays in memory after the last look.
    pub keep_alive: Duration,
}

/// RGB8 pixels, row-major; never written to disk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScreenImage {
    width: u32,
    height: u32,
    rgb: Vec<u8>,
}

impl ScreenImage {
    /// Builds an image, checking the pixel buffer matches the dimensions.
    pub fn new(width: u32, height: u32, rgb: Vec<u8>) -> Result<Self, String> {
        if width == 0 || height == 0 {
            return Err("The image size must be at least 1x1.".to_string());
        }
        let expected = (width as usize)
            .saturating_mul(height as usize)
            .saturating_mul(3);
        if rgb.len() != expected {
            return Err("The image buffer must hold width * height * 3 bytes.".to_string());
        }
        Ok(Self { width, height, rgb })
    }

    /// The image width in pixels.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// The image height in pixels.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// The row-major RGB8 pixel buffer.
    pub fn rgb(&self) -> &[u8] {
        &self.rgb
    }
}

/// A loaded vision model that answers questions about an image.
///
/// [`answer`](VisionEngine::answer) runs on the vision worker thread and sees
/// the captured screen exactly once; the image is dropped right after.
pub trait VisionEngine: Send {
    /// Answers `question` about `image`, writing at most `max_tokens` tokens.
    fn answer(
        &mut self,
        image: &ScreenImage,
        question: &str,
        max_tokens: usize,
        cancel: &AtomicBool,
    ) -> Result<String, String>;
}

/// Loads a [`VisionEngine`] from a [`VisionConfig`]; called on the worker.
pub type VisionFactory =
    Box<dyn Fn(&VisionConfig) -> Result<Box<dyn VisionEngine>, String> + Send + Sync>;

/// Captures the primary screen, downscaled so its longest edge is at most the given number of pixels.
pub type ScreenCapture = Box<dyn Fn(u32) -> Result<ScreenImage, String> + Send + Sync>;

/// Where the vision service is in its life cycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum VisionState {
    /// No complete config.
    Off,
    /// Configured but not loaded.
    Idle,
    /// The model is loading.
    Loading,
    /// The model is loaded.
    Ready,
}

/// The vision service's current state, as the frontend sees it.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct VisionStatus {
    /// The current state.
    pub state: VisionState,
    /// The plain-language error, when there is one.
    pub error: Option<String>,
}

/// Receives every vision state change.
pub type VisionSink = Box<dyn Fn(VisionStatus) + Send + Sync>;

/// Everything [`VisionService::spawn`] needs to start its worker.
pub struct VisionDeps {
    /// Builds the vision engine.
    pub factory: VisionFactory,
    /// Captures the primary screen.
    pub capture: ScreenCapture,
    /// Receives every vision state change.
    pub sink: VisionSink,
}

/// The message for a look before vision has been set up.
pub const VISION_NOT_SET_UP: &str =
    "Vision isn't set up. Choose a vision model and its projector in Settings.";

/// The message for a vision model that produced no visible answer.
pub const VISION_NO_ANSWER: &str = "The vision model gave no answer.";
