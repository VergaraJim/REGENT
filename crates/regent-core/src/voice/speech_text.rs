//! Turning a streamed reply into speakable sentences (spec §5.9).
//!
//! A language model streams tokens that still contain markdown, code fences and
//! tool markup. Text-to-speech needs clean, complete sentences as early as
//! possible: [`SentenceSplitter::push`] takes arbitrary pieces (a token, a few
//! characters) and returns the sentences that can no longer change, while
//! [`SentenceSplitter::finish`] flushes whatever is left.
//!
//! The implementation works in three stages:
//!
//! 1. a character-level state machine that drops `<tool_call>…</tool_call>`
//!    blocks across pieces and lines (and holds back a trailing `<` that could
//!    still begin a tool tag);
//! 2. line assembly that drops code fences, horizontal rules and table
//!    separators, and rewrites the rest of the line's markdown;
//! 3. sentence splitting of the cleaned line, including early emission of
//!    overly long sentences at a comma, semicolon, colon or space.
//!
//! Everything here is pure logic: no I/O, no threads, no new dependencies.

/// Sentences longer than this are cut at a natural break so speech can start.
pub const MAX_SENTENCE_CHARS: usize = 300;

const TOOL_OPEN: &str = "<tool_call>";
const TOOL_CLOSE: &str = "</tool_call>";

/// Words whose trailing dot does not end a sentence.
const ABBREVIATIONS: &[&str] = &[
    "e.g", "i.e", "mr", "mrs", "ms", "dr", "prof", "st", "vs", "approx", "fig", "no",
];

/// Removes tool-call blocks from a stream of characters.
#[derive(Default)]
struct ToolFilter {
    in_tag: bool,
    pending: String,
}

impl ToolFilter {
    /// Filters `input`, returning the characters that are outside tool blocks.
    fn filter(&mut self, input: &str) -> String {
        let mut out = String::new();
        for ch in input.chars() {
            self.pending.push(ch);
            self.settle(&mut out);
        }
        out
    }

    /// Consumes as much of `pending` as can be decided right now.
    fn settle(&mut self, out: &mut String) {
        loop {
            if self.pending.is_empty() {
                return;
            }
            let target = if self.in_tag { TOOL_CLOSE } else { TOOL_OPEN };
            if self.pending == target {
                self.in_tag = !self.in_tag;
                self.pending.clear();
                return;
            }
            if target.starts_with(self.pending.as_str()) {
                // A partial tag: hold it until the next character decides.
                return;
            }
            // Not a tag after all. Outside a block the first char is text;
            // inside a block it is noise.
            let ch = self.pending.remove(0);
            if !self.in_tag {
                out.push(ch);
            }
        }
    }

    /// Releases a held partial open tag; drops an unclosed block.
    fn finish(&mut self) -> String {
        let mut out = String::new();
        if !self.in_tag {
            out.push_str(&self.pending);
        }
        self.pending.clear();
        self.in_tag = false;
        out
    }

    fn reset(&mut self) {
        self.pending.clear();
        self.in_tag = false;
    }
}

/// Splits a streamed reply into speakable sentences.
///
/// The splitter is reusable: after [`finish`](Self::finish) it behaves like a
/// fresh one.
#[derive(Default)]
pub struct SentenceSplitter {
    tool: ToolFilter,
    /// The current, possibly incomplete, raw line (after tool filtering).
    line: String,
    /// Whether a code fence is open, in which case lines are dropped.
    in_fence: bool,
    /// How many chunks of the current line have already been returned.
    emitted: usize,
}

impl SentenceSplitter {
    pub fn new() -> Self {
        Self::default()
    }

    /// Feeds one piece of text and returns the sentences that are now final.
    pub fn push(&mut self, piece: &str) -> Vec<String> {
        let mut out = Vec::new();
        let filtered = self.tool.filter(piece);
        for ch in filtered.chars() {
            if ch == '\n' {
                self.finish_line(&mut out);
            } else {
                self.line.push(ch);
            }
        }
        self.emit_streaming(&mut out);
        out
    }

    /// Flushes the remaining text. The splitter is empty and reusable after.
    pub fn finish(&mut self) -> Vec<String> {
        let mut out = Vec::new();
        let tail = self.tool.finish();
        for ch in tail.chars() {
            if ch == '\n' {
                self.finish_line(&mut out);
            } else {
                self.line.push(ch);
            }
        }
        self.finish_line(&mut out);
        self.in_fence = false;
        self.emitted = 0;
        out
    }

    /// Drops everything pending, keeping the splitter usable.
    pub fn clear(&mut self) {
        self.tool.reset();
        self.line.clear();
        self.in_fence = false;
        self.emitted = 0;
    }

    /// Completes the current line and emits every sentence it contains.
    fn finish_line(&mut self, out: &mut Vec<String>) {
        let line = std::mem::take(&mut self.line);
        if self.in_fence {
            if is_fence_line(&line) {
                self.in_fence = false;
            }
        } else if is_fence_line(&line) {
            self.in_fence = true;
        } else if let Some(cleaned) = clean_line(&line) {
            let chunks = split_sentences(&cleaned, true);
            self.emit_chunks(chunks, out);
        }
        self.emitted = 0;
    }

    /// Emits the final chunks of the still-incomplete current line.
    fn emit_streaming(&mut self, out: &mut Vec<String>) {
        let trimmed = self.line.trim_start();
        if trimmed.is_empty() {
            return;
        }
        // The line could still turn out to be a fence, so hold it all back.
        if trimmed.starts_with('`') {
            return;
        }
        if let Some(cleaned) = clean_line(&self.line) {
            let chunks = split_sentences(&cleaned, false);
            self.emit_chunks(chunks, out);
        }
    }

    /// Emits final chunks, skipping those already returned for this line.
    fn emit_chunks(&mut self, chunks: Vec<(String, bool)>, out: &mut Vec<String>) {
        for (idx, (text, is_final)) in chunks.into_iter().enumerate().skip(self.emitted) {
            if !is_final {
                break;
            }
            self.emitted = idx + 1;
            if is_speakable(&text) {
                out.push(text);
            }
        }
    }
}

fn is_speakable(text: &str) -> bool {
    text.chars().any(|c| c.is_alphabetic() || c.is_numeric())
}

/// Trims `text` and collapses every whitespace run to one space.
fn normalize(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn is_fence_line(line: &str) -> bool {
    line.trim_start().starts_with("```")
}

fn is_horizontal_rule(line: &str) -> bool {
    let marks: Vec<char> = line.chars().filter(|c| !c.is_whitespace()).collect();
    marks.len() >= 3 && marks.iter().all(|&c| c == '-' || c == '*' || c == '_')
}

fn is_table_separator(line: &str) -> bool {
    let marks: Vec<char> = line.chars().filter(|c| !c.is_whitespace()).collect();
    !marks.is_empty()
        && marks.iter().all(|&c| c == '|' || c == '-' || c == ':')
        && marks.contains(&'|')
        && marks.contains(&'-')
}

/// Joins the cells of a markdown table row with commas.
fn table_row_to_text(line: &str) -> String {
    line.trim_matches('|')
        .split('|')
        .map(str::trim)
        .filter(|cell| !cell.is_empty())
        .collect::<Vec<_>>()
        .join(", ")
}

fn strip_heading(line: &str) -> Option<&str> {
    let hashes = line.chars().take_while(|&c| c == '#').count();
    if hashes == 0 {
        return None;
    }
    let rest = &line[hashes..];
    if hashes <= 6 && rest.starts_with(' ') {
        Some(&rest[1..])
    } else {
        None
    }
}

fn strip_numbered(line: &str) -> Option<&str> {
    let digits = line.chars().take_while(|c| c.is_ascii_digit()).count();
    if digits == 0 {
        return None;
    }
    let rest = &line[digits..];
    let mut chars = rest.chars();
    let separator = chars.next()?;
    if (separator == '.' || separator == ')') && chars.next() == Some(' ') {
        Some(&rest[2..])
    } else {
        None
    }
}

fn strip_bullet(line: &str) -> Option<&str> {
    ["- ", "* ", "+ "]
        .iter()
        .find_map(|prefix| line.strip_prefix(prefix))
}

fn strip_task_box(line: &str) -> Option<&str> {
    line.strip_prefix("[ ] ")
        .or_else(|| line.strip_prefix("[x] "))
        .or_else(|| line.strip_prefix("[X] "))
}

/// Removes line-start markdown such as headings, quotes, bullets and numbers.
fn strip_line_start(line: &str) -> String {
    let mut rest = line;
    loop {
        if let Some(next) = strip_heading(rest) {
            rest = next;
        } else if let Some(next) = rest.strip_prefix('>') {
            rest = next.trim_start();
        } else if let Some(next) = strip_bullet(rest) {
            rest = next;
        } else if let Some(next) = strip_numbered(rest) {
            rest = next;
        } else if let Some(next) = strip_task_box(rest) {
            rest = next;
        } else {
            return rest.to_string();
        }
    }
}

/// Cleans one complete line, or returns `None` when the line is dropped.
fn clean_line(line: &str) -> Option<String> {
    let trimmed = line.trim();
    if trimmed.is_empty() || is_horizontal_rule(trimmed) || is_table_separator(trimmed) {
        return None;
    }
    let text = if trimmed.starts_with('|') || trimmed.ends_with('|') {
        table_row_to_text(trimmed)
    } else {
        strip_line_start(trimmed)
    };
    let cleaned = clean_inline(&text);
    if cleaned.trim().is_empty() {
        None
    } else {
        Some(cleaned)
    }
}

fn starts_with_at(chars: &[char], start: usize, pattern: &str) -> bool {
    for (index, expected) in (start..).zip(pattern.chars()) {
        if chars.get(index) != Some(&expected) {
            return false;
        }
    }
    true
}

/// Parses `[text](url)` at `open` and returns the text and the index after it.
fn parse_link(chars: &[char], open: usize) -> Option<(String, usize)> {
    let close = open + 1 + chars[open + 1..].iter().position(|&c| c == ']')?;
    if chars.get(close + 1) != Some(&'(') {
        return None;
    }
    let paren = close + 2 + chars[close + 2..].iter().position(|&c| c == ')')?;
    let text: String = chars[open + 1..close].iter().collect();
    Some((text, paren + 1))
}

/// Parses an HTML-like tag `</?[A-Za-z][^>]*>` and returns the index after it.
fn parse_html_tag(chars: &[char], start: usize) -> Option<usize> {
    let mut index = start + 1;
    if chars.get(index) == Some(&'/') {
        index += 1;
    }
    if !chars.get(index).is_some_and(char::is_ascii_alphabetic) {
        return None;
    }
    index += 1;
    while index < chars.len() {
        if chars[index] == '>' {
            return Some(index + 1);
        }
        index += 1;
    }
    None
}

/// Removes inline markdown, HTML tags and bare URLs.
fn clean_inline(input: &str) -> String {
    let chars: Vec<char> = input.chars().collect();
    let mut out = String::new();
    let mut i = 0;
    while i < chars.len() {
        let ch = chars[i];
        if ch == '!' && chars.get(i + 1) == Some(&'[') {
            if let Some((text, next)) = parse_link(&chars, i + 1) {
                out.push_str(&clean_inline(&text));
                i = next;
                continue;
            }
        }
        if ch == '[' {
            if let Some((text, next)) = parse_link(&chars, i) {
                out.push_str(&clean_inline(&text));
                i = next;
                continue;
            }
        }
        if ch == '<' {
            if let Some(next) = parse_html_tag(&chars, i) {
                i = next;
                continue;
            }
        }
        if ch == 'h'
            && (starts_with_at(&chars, i, "http://") || starts_with_at(&chars, i, "https://"))
        {
            let mut end = i;
            while end < chars.len() && !chars[end].is_whitespace() {
                end += 1;
            }
            i = end;
            continue;
        }
        match ch {
            '`' | '*' => {
                i += 1;
                continue;
            }
            '~' if chars.get(i + 1) == Some(&'~') => {
                i += 2;
                continue;
            }
            '_' => {
                let previous = i.checked_sub(1).map(|p| chars[p]);
                let next = chars.get(i + 1).copied();
                if previous.is_some_and(char::is_alphanumeric)
                    && next.is_some_and(char::is_alphanumeric)
                {
                    out.push('_');
                }
                i += 1;
                continue;
            }
            _ => {}
        }
        out.push(ch);
        i += 1;
    }
    out
}

fn is_end_punctuation(ch: char) -> bool {
    matches!(ch, '.' | '!' | '?' | '…')
}

fn is_closing_quote(ch: char) -> bool {
    matches!(ch, '"' | '\'' | '”' | '’' | ')' | ']')
}

/// Whether the word ending right before `position` is an abbreviation or an
/// initial, so its dot does not end the sentence.
fn is_abbreviated(chars: &[char], position: usize) -> bool {
    let mut start = position;
    while start > 0 && (chars[start - 1].is_alphanumeric() || chars[start - 1] == '.') {
        start -= 1;
    }
    let word: String = chars[start..position].iter().collect();
    if word.chars().count() == 1 && word.chars().all(|c| c.is_uppercase()) {
        return true;
    }
    ABBREVIATIONS.contains(&word.to_lowercase().as_str())
}

/// Cuts `text` into pieces of at most [`MAX_SENTENCE_CHARS`] at a natural
/// break. `final_flag` is attached to the trailing remainder.
fn split_long(text: String, final_flag: bool) -> Vec<(String, bool)> {
    let mut out = Vec::new();
    let mut rest = text;
    loop {
        let chars: Vec<char> = rest.chars().collect();
        if chars.len() <= MAX_SENTENCE_CHARS {
            if !rest.is_empty() {
                out.push((rest, final_flag));
            }
            break;
        }
        let mut split = None;
        for index in (1..MAX_SENTENCE_CHARS).rev() {
            match chars[index] {
                ',' | ';' | ':' => {
                    split = Some(index + 1);
                    break;
                }
                ' ' => {
                    split = Some(index);
                    break;
                }
                _ => {}
            }
        }
        let head_len = split.unwrap_or(MAX_SENTENCE_CHARS);
        let head = normalize(&chars[..head_len].iter().collect::<String>());
        if !head.is_empty() {
            out.push((head, true));
        }
        rest = normalize(&chars[head_len..].iter().collect::<String>());
        if rest.is_empty() {
            break;
        }
    }
    out
}

/// Splits a cleaned line into chunks, marking which can already be spoken.
///
/// The last chunk is final only when `complete` is true (the line ended) or
/// when it was produced by cutting a long sentence.
fn split_sentences(cleaned: &str, complete: bool) -> Vec<(String, bool)> {
    let chars: Vec<char> = cleaned.chars().collect();
    let mut result = Vec::new();
    let mut start = 0;
    let mut i = 0;
    while i < chars.len() {
        if !is_end_punctuation(chars[i]) {
            i += 1;
            continue;
        }
        let mut end = i;
        while end < chars.len() && is_end_punctuation(chars[end]) {
            end += 1;
        }
        let run_len = end - i;
        let mut after = end;
        while after < chars.len() && is_closing_quote(chars[after]) {
            after += 1;
        }
        let followed_by_space = after < chars.len() && chars[after].is_whitespace();
        let at_end = after >= chars.len();
        let exempt = run_len == 1 && chars[i] == '.' && is_abbreviated(&chars, i);
        if !exempt && (followed_by_space || (at_end && complete)) {
            let text = normalize(&chars[start..after].iter().collect::<String>());
            if !text.is_empty() {
                result.extend(split_long(text, true));
            }
            start = after;
            i = after;
            continue;
        }
        i = end;
    }
    if start < chars.len() {
        let text = normalize(&chars[start..].iter().collect::<String>());
        if !text.is_empty() {
            result.extend(split_long(text, complete));
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn split_all(pieces: &[&str]) -> Vec<String> {
        let mut splitter = SentenceSplitter::new();
        let mut out = Vec::new();
        for piece in pieces {
            out.extend(splitter.push(piece));
        }
        out.extend(splitter.finish());
        out
    }

    fn by_chars(text: &str) -> Vec<String> {
        let mut splitter = SentenceSplitter::new();
        let mut out = Vec::new();
        for ch in text.chars() {
            out.extend(splitter.push(&ch.to_string()));
        }
        out.extend(splitter.finish());
        out
    }

    fn by_three(text: &str) -> Vec<String> {
        let chars: Vec<char> = text.chars().collect();
        let mut splitter = SentenceSplitter::new();
        let mut out = Vec::new();
        for piece in chars.chunks(3) {
            out.extend(splitter.push(&piece.iter().collect::<String>()));
        }
        out.extend(splitter.finish());
        out
    }

    /// Asserts invariance across chunkings and returns the output.
    fn invariant(text: &str) -> Vec<String> {
        let whole = split_all(&[text]);
        assert_eq!(whole, by_chars(text), "char chunking differs for {text:?}");
        assert_eq!(
            whole,
            by_three(text),
            "3-char chunking differs for {text:?}"
        );
        whole
    }

    #[test]
    fn simple_sentences() {
        assert_eq!(
            invariant("Hello there. How are you? Fine"),
            ["Hello there.", "How are you?", "Fine"]
        );
    }

    #[test]
    fn abbreviations_and_decimals_do_not_split() {
        let out = invariant("Dr. Smith arrived at 3.5 p.m. today. He sat.");
        assert_ne!(out[0], "Dr.");
        assert!(out[0].contains("3.5"), "{out:?}");
        assert_eq!(out.last().map(String::as_str), Some("He sat."));
    }

    #[test]
    fn closing_quotes_stay_with_the_sentence() {
        assert_eq!(
            invariant("He said \"hi!\" Then left."),
            ["He said \"hi!\"", "Then left."]
        );
    }

    #[test]
    fn headings_and_bullets_are_removed() {
        assert_eq!(
            invariant("# Title\n- **First** item\n- Second item\n"),
            ["Title", "First item", "Second item"]
        );
    }

    #[test]
    fn code_fences_are_dropped() {
        assert_eq!(
            invariant("Look:\n```rust\nlet x = 1;\n```\nDone."),
            ["Look:", "Done."]
        );
        assert_eq!(invariant("```"), Vec::<String>::new());
        assert!(invariant("```\nunclosed code").is_empty());
        assert_eq!(invariant("---"), Vec::<String>::new());
    }

    #[test]
    fn inline_code_keeps_its_text() {
        assert_eq!(invariant("Run `cargo test` now."), ["Run cargo test now."]);
    }

    #[test]
    fn links_and_bare_urls_are_simplified() {
        assert_eq!(
            invariant("See [the docs](https://x.y/z) or https://example.com now."),
            ["See the docs or now."]
        );
        assert_eq!(
            invariant("An ![image](https://x.y/i.png) here."),
            ["An image here."]
        );
    }

    #[test]
    fn tables_become_comma_joined_rows() {
        let table = "| Name | Value |\n| --- | --- |\n| First | 1 |\n";
        assert_eq!(invariant(table), ["Name, Value", "First, 1"]);
    }

    #[test]
    fn tool_blocks_are_dropped() {
        let text = "Let me check. <tool_call>{\"name\":\"x\"}</tool_call>Found it.";
        let out = invariant(text);
        assert_eq!(out, ["Let me check.", "Found it."]);
        assert!(out.iter().all(|s| !s.contains("tool")));

        // An unclosed block is dropped at finish.
        assert_eq!(
            invariant("Before. <tool_call>{\"name\":\"x\"}"),
            ["Before."]
        );
    }

    #[test]
    fn a_trailing_partial_tag_is_released() {
        assert_eq!(invariant("a < b"), ["a < b"]);
        assert_eq!(invariant("not a <tool tag"), ["not a <tool tag"]);
    }

    #[test]
    fn ellipses_split() {
        assert_eq!(invariant("Wait... what?"), ["Wait...", "what?"]);
        assert_eq!(invariant("Okay… sure."), ["Okay…", "sure."]);
    }

    #[test]
    fn snake_case_survives() {
        assert_eq!(invariant("Use snake_case here."), ["Use snake_case here."]);
    }

    #[test]
    fn long_sentences_are_cut_at_natural_breaks() {
        let long = "word, ".repeat(120);
        let out = invariant(&long);
        assert!(out.len() > 1, "{out:?}");
        assert!(out.iter().all(|s| s.chars().count() <= MAX_SENTENCE_CHARS));
    }

    #[test]
    fn push_only_returns_finished_sentences() {
        let mut splitter = SentenceSplitter::new();
        assert_eq!(splitter.push("Hello there. Ho"), ["Hello there."]);
        assert_eq!(splitter.finish(), ["Ho"]);

        let mut splitter = SentenceSplitter::new();
        assert!(splitter.push("Hello there.").is_empty());
        assert_eq!(splitter.finish(), ["Hello there."]);
    }

    #[test]
    fn clear_drops_pending_text() {
        let mut splitter = SentenceSplitter::new();
        assert_eq!(splitter.push("Hello there. Ho"), ["Hello there."]);
        splitter.clear();
        assert!(splitter.finish().is_empty());
        // Still usable afterwards.
        assert_eq!(splitter.push("New start."), Vec::<String>::new());
        assert_eq!(splitter.finish(), ["New start."]);
    }

    #[test]
    fn finish_returns_unterminated_final_sentence() {
        let mut splitter = SentenceSplitter::new();
        assert!(splitter.push("hello wor").is_empty());
        assert_eq!(splitter.finish(), ["hello wor"]);
    }

    #[test]
    fn finished_splitter_is_reusable() {
        let mut splitter = SentenceSplitter::new();
        assert_eq!(splitter.push("One."), Vec::<String>::new());
        assert_eq!(splitter.finish(), ["One."]);
        assert_eq!(splitter.push("Two."), Vec::<String>::new());
        assert_eq!(splitter.finish(), ["Two."]);
    }
}
