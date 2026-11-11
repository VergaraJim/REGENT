//! Captures the primary screen for the vision service.
//!
//! The pixels live only in memory: nothing here writes an image to a file, a
//! temporary file or a log. The capture runs on whatever thread calls the
//! closure (the vision worker), never on the UI or realtime threads.

use regent_core::vision::ScreenCapture;
#[cfg(feature = "screen")]
use regent_core::vision::ScreenImage;

/// The message for a machine with no monitor to capture.
pub const NO_SCREEN: &str = "No screen was found to look at.";

/// The message for a screen capture that failed on this platform.
pub fn capture_failed_message(wayland: bool, macos: bool) -> &'static str {
    if macos {
        "Regent needs Screen Recording permission. Turn it on in System Settings → Privacy & Security → Screen Recording, then restart Regent."
    } else if wayland {
        "Regent couldn't capture the screen. Allow screen sharing when your desktop asks, then try again."
    } else {
        "Regent couldn't capture the screen."
    }
}

/// Scales `width` x `height` down to fit inside `max_edge`, keeping the aspect
/// ratio. Never upscales, never returns a zero side, and `max_edge == 0` means
/// no limit.
pub fn fit_within(width: u32, height: u32, max_edge: u32) -> (u32, u32) {
    if width == 0 || height == 0 || max_edge == 0 {
        return (width.max(1), height.max(1));
    }

    let longest = width.max(height);
    if longest <= max_edge {
        return (width, height);
    }

    let scaled = |edge: u32| -> u32 {
        let value = (edge as u64 * max_edge as u64) / longest as u64;
        value.max(1) as u32
    };

    if width >= height {
        (max_edge, scaled(height))
    } else {
        (scaled(width), max_edge)
    }
}

/// Converts row-major RGBA8 pixels to RGB8 by dropping every fourth byte.
pub fn rgba_to_rgb(rgba: &[u8]) -> Vec<u8> {
    let (pixels, _) = rgba.as_chunks::<4>();
    let mut rgb = Vec::with_capacity(pixels.len() * 3);
    for pixel in pixels {
        rgb.extend_from_slice(&pixel[..3]);
    }
    rgb
}

/// Captures the primary screen, downscaled so its longest edge is at most the
/// given number of pixels.
#[cfg(feature = "screen")]
pub fn screen_capture() -> ScreenCapture {
    Box::new(|max_edge| {
        use xcap::image::imageops::{self, FilterType};
        use xcap::Monitor;

        let failed = || {
            capture_failed_message(
                std::env::var_os("WAYLAND_DISPLAY").is_some(),
                cfg!(target_os = "macos"),
            )
        };

        let monitors = Monitor::all().map_err(|error| {
            eprintln!("regent: screen capture failed: {error}");
            failed().to_string()
        })?;

        let monitor = monitors
            .iter()
            .find(|monitor| matches!(monitor.is_primary(), Ok(true)))
            .or_else(|| monitors.first())
            .ok_or_else(|| NO_SCREEN.to_string())?;

        let image = monitor.capture_image().map_err(|error| {
            eprintln!("regent: screen capture failed: {error}");
            failed().to_string()
        })?;

        let (width, height) = fit_within(image.width(), image.height(), max_edge);
        let image = if (width, height) == (image.width(), image.height()) {
            image
        } else {
            imageops::resize(&image, width, height, FilterType::Triangle)
        };

        let rgb = rgba_to_rgb(image.as_raw());
        ScreenImage::new(width, height, rgb)
    })
}

/// Screen capture is not compiled into this build.
#[cfg(not(feature = "screen"))]
pub fn screen_capture() -> ScreenCapture {
    Box::new(|_| Err("Screen capture isn't available in this build.".to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fit_within_landscape() {
        assert_eq!(fit_within(1920, 1080, 1280), (1280, 720));
    }

    #[test]
    fn fit_within_portrait() {
        assert_eq!(fit_within(1080, 1920, 1280), (720, 1280));
    }

    #[test]
    fn fit_within_already_small_is_unchanged() {
        assert_eq!(fit_within(800, 600, 1280), (800, 600));
        assert_eq!(fit_within(1280, 720, 1280), (1280, 720));
    }

    #[test]
    fn fit_within_zero_means_no_limit() {
        assert_eq!(fit_within(3840, 2160, 0), (3840, 2160));
        assert_eq!(fit_within(1, 1, 0), (1, 1));
    }

    #[test]
    fn fit_within_extreme_aspect_never_returns_zero() {
        assert_eq!(fit_within(10_000, 1, 100), (100, 1));
        assert_eq!(fit_within(1, 10_000, 100), (1, 100));
        assert_eq!(fit_within(u32::MAX, 1, 1), (1, 1));
    }

    #[test]
    fn rgba_to_rgb_drops_the_alpha_byte() {
        let rgba = [1, 2, 3, 4, 5, 6, 7, 8];
        assert_eq!(rgba_to_rgb(&rgba), vec![1, 2, 3, 5, 6, 7]);
        assert_eq!(rgba_to_rgb(&[]), Vec::<u8>::new());
        // A trailing partial pixel is ignored.
        assert_eq!(rgba_to_rgb(&[9, 8, 7]), Vec::<u8>::new());
    }

    #[test]
    fn capture_failed_message_picks_the_platform_text() {
        assert!(capture_failed_message(false, true).contains("Screen Recording permission"));
        assert!(capture_failed_message(true, false).contains("screen sharing"));
        assert_eq!(
            capture_failed_message(false, false),
            "Regent couldn't capture the screen."
        );
        // macOS wins over Wayland when both flags are set.
        assert!(capture_failed_message(true, true).contains("Screen Recording permission"));
        assert_eq!(NO_SCREEN, "No screen was found to look at.");
    }
}
