//! Splitting a message into overlapping chunks for embedding.

/// The size of one chunk, in characters: about 256 tokens at 4 characters per
/// token.
pub const CHUNK_CHARS: usize = 1024;

/// How much neighbouring chunks overlap, in characters: 20% of a chunk, so a
/// sentence cut at one boundary still appears whole in the next chunk.
pub const CHUNK_OVERLAP_CHARS: usize = 204;

/// Splits `text` into overlapping windows of at most [`CHUNK_CHARS`] chars.
///
/// The text is trimmed first; empty text gives no chunks and short text gives
/// one. Windows always start and end on character boundaries, so multi-byte
/// text is never split. Each chunk is trimmed.
pub fn chunk_text(text: &str) -> Vec<String> {
    let text = text.trim();
    if text.is_empty() {
        return Vec::new();
    }

    let chars: Vec<char> = text.chars().collect();
    if chars.len() <= CHUNK_CHARS {
        return vec![text.to_string()];
    }

    let step = CHUNK_CHARS - CHUNK_OVERLAP_CHARS;
    let mut chunks = Vec::new();
    let mut start = 0;
    loop {
        let end = (start + CHUNK_CHARS).min(chars.len());
        let chunk: String = chars[start..end].iter().collect();
        chunks.push(chunk.trim().to_string());
        if end == chars.len() {
            break;
        }
        start += step;
    }
    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_and_blank_text_gives_no_chunks() {
        assert!(chunk_text("").is_empty());
        assert!(chunk_text("   \n\t ").is_empty());
    }

    #[test]
    fn short_text_gives_one_trimmed_chunk() {
        assert_eq!(chunk_text("  hello world \n"), vec!["hello world"]);
    }

    #[test]
    fn exactly_one_chunk_of_the_limit_gives_one_chunk() {
        let text = "a".repeat(CHUNK_CHARS);
        let chunks = chunk_text(&text);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].chars().count(), CHUNK_CHARS);
    }

    #[test]
    fn long_text_is_split_into_overlapping_windows() {
        let text: String = (0..3000)
            .map(|index| char::from(b'a' + (index % 26) as u8))
            .collect();
        let chunks = chunk_text(&text);

        assert!(chunks.len() > 1);
        for chunk in &chunks {
            assert!(chunk.chars().count() <= CHUNK_CHARS);
        }
        for pair in chunks.windows(2) {
            let (previous, next) = (&pair[0], &pair[1]);
            let previous_len = previous.chars().count();
            let overlap: String = previous
                .chars()
                .skip(previous_len - CHUNK_OVERLAP_CHARS)
                .collect();
            assert!(next.starts_with(&overlap), "the windows overlap");
        }
        assert_eq!(chunks.len(), 4);
        assert_eq!(chunks.last().expect("a last chunk").chars().count(), 540);
        assert!(text.ends_with(chunks.last().expect("a last chunk")));
        assert!(text.starts_with(chunks.first().expect("a first chunk")));
    }

    #[test]
    fn multi_byte_text_never_panics() {
        let text = "é".repeat(3000);
        let chunks = chunk_text(&text);
        assert!(!chunks.is_empty());
        for chunk in &chunks {
            assert!(chunk.chars().count() <= CHUNK_CHARS);
            assert!(chunk.chars().all(|character| character == 'é'));
        }
    }
}
