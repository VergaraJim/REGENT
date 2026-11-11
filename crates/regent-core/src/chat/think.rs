//! Hides a hybrid reasoning model's reasoning block from the reply.
//!
//! The filter is pure streaming logic: pieces go in and only the text that
//! belongs in the visible reply comes out. The chat service runs it before the
//! tool-call parser, so reasoning never reaches the UI, the store, speech or a
//! tool call.

/// The opening marker of a reasoning block.
pub const THINK_OPEN: &str = "<think>";

/// The closing marker of a reasoning block.
pub const THINK_CLOSE: &str = "</think>";

/// The state of the streaming reasoning filter.
#[derive(Debug, Clone, Default)]
pub struct ThinkFilter {
    /// A trailing suffix that could still turn out to be a marker.
    pending: String,
    /// Whether a reasoning block is being dropped.
    in_think: bool,
    /// Whether any visible text has been emitted yet.
    started: bool,
    /// Whether whitespace right after a closed block is still being dropped.
    just_closed: bool,
}

impl ThinkFilter {
    /// A filter that has seen nothing yet.
    pub fn new() -> Self {
        Self::default()
    }

    /// Feeds one streamed piece and returns the visible text it completes.
    pub fn push(&mut self, piece: &str) -> String {
        let mut text = std::mem::take(&mut self.pending);
        text.push_str(piece);
        let mut visible = String::new();

        loop {
            if self.in_think {
                let Some(end) = text.find(THINK_CLOSE) else {
                    // Drop everything but a suffix that may start the close
                    // marker once the next piece arrives.
                    let hold = partial_tag_len(&text);
                    let split = text.len() - hold;
                    self.pending = text[split..].to_string();
                    return visible;
                };
                text.drain(..end + THINK_CLOSE.len());
                self.in_think = false;
                self.just_closed = true;
                continue;
            }

            let open = text.find(THINK_OPEN);
            let close = text.find(THINK_CLOSE);
            let stray = match (open, close) {
                (Some(open), Some(close)) => close < open,
                (None, Some(_)) => true,
                _ => false,
            };
            if let (true, Some(end)) = (stray, close) {
                // A stray close marker with no open block: drop the tag only.
                self.emit(&mut visible, &text[..end]);
                text.drain(..end + THINK_CLOSE.len());
                continue;
            }
            if let Some(start) = open {
                self.emit(&mut visible, &text[..start]);
                text.drain(..start + THINK_OPEN.len());
                self.in_think = true;
                continue;
            }

            let hold = partial_tag_len(&text);
            let split = text.len() - hold;
            self.emit(&mut visible, &text[..split]);
            self.pending = text[split..].to_string();
            return visible;
        }
    }

    /// Ends the stream: an unterminated block is dropped and a held-back
    /// partial marker comes out as ordinary text.
    pub fn finish(self) -> String {
        if self.in_think {
            String::new()
        } else {
            self.pending
        }
    }

    /// Appends visible text, dropping leading whitespace at the start of the
    /// reply and right after a closed block.
    fn emit(&mut self, visible: &mut String, text: &str) {
        let text = if self.started && !self.just_closed {
            text
        } else {
            text.trim_start()
        };
        if text.is_empty() {
            return;
        }
        self.started = true;
        self.just_closed = false;
        visible.push_str(text);
    }
}

/// Runs the filter over a whole text and trims the result.
pub fn strip_think(text: &str) -> String {
    let mut filter = ThinkFilter::new();
    let mut visible = filter.push(text);
    visible.push_str(&filter.finish());
    visible.trim().to_string()
}

/// The length of the longest suffix of `text` that could start a marker.
fn partial_tag_len(text: &str) -> usize {
    let longest = THINK_OPEN.len().max(THINK_CLOSE.len()).saturating_sub(1);
    for length in (1..=longest.min(text.len())).rev() {
        if ends_with_prefix(text, THINK_OPEN, length) || ends_with_prefix(text, THINK_CLOSE, length)
        {
            return length;
        }
    }
    0
}

/// Whether `text` ends with the first `length` bytes of `tag`.
fn ends_with_prefix(text: &str, tag: &str, length: usize) -> bool {
    length <= tag.len() && text.ends_with(&tag[..length])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_text_passes_through() {
        let mut filter = ThinkFilter::new();
        assert_eq!(filter.push("Hello there"), "Hello there");
        assert_eq!(filter.finish(), "");
    }

    #[test]
    fn a_leading_block_and_its_whitespace_are_dropped() {
        let text = format!("{THINK_OPEN}plan{THINK_CLOSE}\n\nHello");
        assert_eq!(strip_think(&text), "Hello");
        let spaced = format!("   {THINK_OPEN}x{THINK_CLOSE}\n\n  Hello");
        assert_eq!(strip_think(&spaced), "Hello");

        let mut filter = ThinkFilter::new();
        let head = format!("{THINK_OPEN}plan{THINK_CLOSE}\n\n");
        assert_eq!(filter.push(&head), "");
        assert_eq!(filter.push("Hello"), "Hello");
        assert_eq!(filter.finish(), "");
    }

    #[test]
    fn markers_split_across_pieces_are_recognised() {
        let mut filter = ThinkFilter::new();
        let mut visible = String::new();
        for piece in ["<thi", "nk>x</th", "ink>Hi"] {
            visible.push_str(&filter.push(piece));
        }
        assert_eq!(visible, "Hi");
        assert_eq!(filter.finish(), "");
    }

    #[test]
    fn a_block_in_the_middle_is_dropped() {
        let text = format!("Hello {THINK_OPEN}plan{THINK_CLOSE} world");
        assert_eq!(strip_think(&text), "Hello world");

        let mut filter = ThinkFilter::new();
        let mut visible = filter.push(&text);
        visible.push_str(&filter.finish());
        assert_eq!(visible, "Hello world");
    }

    #[test]
    fn a_lone_angle_bracket_survives() {
        assert_eq!(strip_think("a < b"), "a < b");
        let mut filter = ThinkFilter::new();
        assert_eq!(filter.push("a < b"), "a < b");
        assert_eq!(filter.finish(), "");
    }

    #[test]
    fn an_unterminated_block_is_dropped_at_finish() {
        let text = format!("before {THINK_OPEN}still reasoning");
        let mut filter = ThinkFilter::new();
        assert_eq!(filter.push(&text), "before ");
        assert_eq!(filter.finish(), "");
    }

    #[test]
    fn a_held_back_partial_marker_comes_out_at_finish() {
        let mut filter = ThinkFilter::new();
        assert_eq!(filter.push("text <thi"), "text ");
        assert_eq!(filter.finish(), "<thi");
    }

    #[test]
    fn a_stray_close_marker_is_removed() {
        let stray = format!("Hi{THINK_CLOSE}there");
        let mut filter = ThinkFilter::new();
        assert_eq!(filter.push(&stray), "Hithere");
        assert_eq!(filter.finish(), "");

        let spaced = format!("Hi {THINK_CLOSE} there");
        assert_eq!(strip_think(&spaced), "Hi  there");
    }

    #[test]
    fn strip_think_cases() {
        assert_eq!(strip_think("plain text"), "plain text");
        assert_eq!(strip_think("  plain text  "), "plain text");
        let leading = format!("{THINK_OPEN}a{THINK_CLOSE}\n\nanswer");
        assert_eq!(strip_think(&leading), "answer");
        let mixed = format!("one {THINK_OPEN}two{THINK_CLOSE} three");
        assert_eq!(strip_think(&mixed), "one three");
        let open = format!("{THINK_OPEN}never closed");
        assert_eq!(strip_think(&open), "");
    }
}
