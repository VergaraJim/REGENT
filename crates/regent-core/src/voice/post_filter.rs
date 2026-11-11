//! Cleaning up what speech-to-text returns.
//!
//! STT models emit stock phrases on silence ("Thank you.", "Thanks for
//! watching", `[BLANK_AUDIO]`). Dropping them unconditionally would also throw
//! away a genuine "thank you", so they are dropped only when the utterance was
//! short or quiet (spec §5.8).

/// Below this duration an utterance counts as short.
pub const SHORT_UTTERANCE_MS: u64 = 2500;

/// Below this RMS an utterance counts as quiet.
pub const QUIET_RMS: f32 = 0.01;

/// Transcripts shorter than this many characters are dropped.
pub const MIN_TRANSCRIPT_CHARS: usize = 2;

/// The stock phrases STT models produce on silence, already normalised.
const STOCK_PHRASES: &[&str] = &[
    "thank you",
    "thank you very much",
    "thanks",
    "thanks for watching",
    "thank you for watching",
    "thanks for listening",
    "please subscribe",
    "subscribe",
    "like and subscribe",
    "you",
    "bye",
    "blank audio",
    "silence",
    "music",
    "applause",
];

/// Trims `text` and collapses every whitespace run to one space.
fn collapse_whitespace(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Lowercases `text` and keeps only letters, digits and single spaces.
fn stock_form(text: &str) -> String {
    let mut kept = String::with_capacity(text.len());
    let mut pending_space = false;
    for ch in text.to_lowercase().chars() {
        if ch.is_alphanumeric() {
            if pending_space && !kept.is_empty() {
                kept.push(' ');
            }
            pending_space = false;
            kept.push(ch);
        } else if ch.is_whitespace() {
            pending_space = true;
        }
    }
    kept
}

/// Whether `text` is a single bracketed marker such as `[BLANK_AUDIO]`,
/// `(music)` or `*applause*`.
fn is_bracketed_marker(text: &str) -> bool {
    let trimmed = text.trim();
    let bytes = trimmed.as_bytes();
    let (open, close) = match bytes.first() {
        Some(b'[') => (b'[', b']'),
        Some(b'(') => (b'(', b')'),
        Some(b'*') => (b'*', b'*'),
        _ => return false,
    };
    if bytes.len() < 2 || bytes[bytes.len() - 1] != close {
        return false;
    }
    let inner = &trimmed[1..trimmed.len() - 1];
    !inner.contains(open as char) && !inner.contains(close as char)
}

fn is_stock_phrase(text: &str) -> bool {
    let form = stock_form(text);
    STOCK_PHRASES.contains(&form.as_str())
}

/// Cleans a raw transcript and decides whether it is worth keeping.
///
/// Whitespace is trimmed and collapsed. When the utterance was short
/// (under [`SHORT_UTTERANCE_MS`]) or quiet (RMS under [`QUIET_RMS`]), stock
/// phrases and lone bracketed markers are dropped. So is anything shorter
/// than [`MIN_TRANSCRIPT_CHARS`].
pub fn clean_transcript(text: &str, duration_ms: u64, rms: f32) -> Option<String> {
    let cleaned = collapse_whitespace(text);
    if cleaned.is_empty() {
        return None;
    }
    let short_or_quiet = duration_ms < SHORT_UTTERANCE_MS || rms < QUIET_RMS;
    if short_or_quiet && (is_stock_phrase(&cleaned) || is_bracketed_marker(&cleaned)) {
        return None;
    }
    if cleaned.chars().count() < MIN_TRANSCRIPT_CHARS {
        return None;
    }
    Some(cleaned)
}

#[cfg(test)]
mod tests {
    use super::*;

    const LONG: u64 = 4000;
    const LOUD: f32 = 0.05;

    fn dropped(text: &str) -> bool {
        clean_transcript(text, 1000, 0.001).is_none()
    }

    #[test]
    fn stock_phrases_are_dropped_when_short() {
        for phrase in STOCK_PHRASES {
            assert!(dropped(phrase), "{phrase}");
        }
        for variant in [
            "Thank you!",
            "  THANK YOU  ",
            "Thanks for watching...",
            "Thanks for listening.",
            "Please, subscribe!",
            "Like and subscribe.",
        ] {
            assert!(dropped(variant), "{variant}");
        }
    }

    #[test]
    fn a_genuine_thank_you_is_kept_when_long_and_loud() {
        assert_eq!(
            clean_transcript("Thank you.", LONG, LOUD).as_deref(),
            Some("Thank you.")
        );
    }

    #[test]
    fn bracketed_markers_are_dropped_when_quiet() {
        assert_eq!(clean_transcript("[BLANK_AUDIO]", 500, 0.001), None);
        assert_eq!(clean_transcript("[ Silence ]", 500, 0.001), None);
        assert_eq!(clean_transcript("(music)", 500, 0.001), None);
        assert_eq!(clean_transcript("*applause*", 500, 0.001), None);
    }

    #[test]
    fn very_short_transcripts_are_dropped() {
        assert_eq!(clean_transcript("a", LONG, LOUD), None);
        assert_eq!(clean_transcript("Hi", LONG, LOUD).as_deref(), Some("Hi"));
        assert_eq!(clean_transcript("", LONG, LOUD), None);
        assert_eq!(clean_transcript("   ", LONG, LOUD), None);
    }

    #[test]
    fn whitespace_is_collapsed() {
        assert_eq!(
            clean_transcript("  Hello \n\t there   world  ", LONG, LOUD).as_deref(),
            Some("Hello there world")
        );
    }

    #[test]
    fn a_stock_phrase_is_kept_when_the_utterance_is_long_and_loud() {
        assert_eq!(
            clean_transcript("Thanks for watching", LONG, LOUD).as_deref(),
            Some("Thanks for watching")
        );
    }
}
