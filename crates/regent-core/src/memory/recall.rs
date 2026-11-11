//! Formatting recalled messages for the model and for `search_memory`.

use crate::chat::{ChatMessage, Role};

/// How many characters of a message are shown when it is recalled.
pub const SNIPPET_CHARS: usize = 300;

/// The most characters `search_memory` returns in total: about 600 tokens.
pub const RECALL_MAX_CHARS: usize = 2400;

/// Returned when the index holds no messages at all.
pub const NOTHING_INDEXED: &str = "Nothing has been saved to memory yet.";

/// Returned when the index holds messages but none matched.
pub const NO_MATCHES: &str = "No earlier messages matched.";

/// Formats a unix timestamp as UTC `YYYY-MM-DD`.
///
/// It uses the days-from-civil inverse algorithm, so no date crate is needed.
pub fn format_date(unix_seconds: i64) -> String {
    let days = unix_seconds.div_euclid(86_400);
    let (year, month, day) = civil_from_days(days);
    format!("{year:04}-{month:02}-{day:02}")
}

/// Converts days since 1970-01-01 into a (year, month, day) civil date.
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let shifted = days + 719_468;
    let era = if shifted >= 0 {
        shifted
    } else {
        shifted - 146_096
    } / 146_097;
    let day_of_era = shifted - era * 146_097;
    let year_of_era =
        (day_of_era - day_of_era / 1460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let year = year_of_era + era * 400;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_prime = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_prime + 2) / 5 + 1;
    let month = if month_prime < 10 {
        month_prime + 3
    } else {
        month_prime - 9
    };
    let year = if month <= 2 { year + 1 } else { year };
    (year, month as u32, day as u32)
}

/// Formats recalled messages as one `[date] role: snippet` line each.
///
/// Messages keep the order given. Tool rows are skipped, whitespace is
/// collapsed to single spaces, and each snippet is cut to [`SNIPPET_CHARS`]
/// with `…` appended when it was cut. Lines are joined with newlines and the
/// result stops before a line that would pass [`RECALL_MAX_CHARS`]. When no
/// line fits the result is [`NO_MATCHES`].
pub fn format_recall(messages: &[ChatMessage]) -> String {
    let mut lines: Vec<String> = Vec::new();
    let mut total = 0;
    for message in messages {
        let role = match message.role {
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => continue,
        };
        let line = format!(
            "[{}] {}: {}",
            format_date(message.created_at),
            role,
            snippet(&message.text)
        );
        let line_len = line.chars().count();
        let needed = if lines.is_empty() {
            line_len
        } else {
            total + 1 + line_len
        };
        if needed > RECALL_MAX_CHARS {
            break;
        }
        total = needed;
        lines.push(line);
    }
    if lines.is_empty() {
        NO_MATCHES.to_string()
    } else {
        lines.join("\n")
    }
}

/// Collapses whitespace and cuts the text to [`SNIPPET_CHARS`].
fn snippet(text: &str) -> String {
    let collapsed = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut chars = collapsed.chars();
    let mut snippet: String = chars.by_ref().take(SNIPPET_CHARS).collect();
    if chars.next().is_some() {
        snippet.push('…');
    }
    snippet
}

#[cfg(test)]
mod tests {
    use super::*;

    fn message(id: i64, role: Role, text: &str, created_at: i64) -> ChatMessage {
        ChatMessage {
            id,
            role,
            text: text.to_string(),
            from_voice: false,
            created_at,
            tool_name: None,
            tool_call: None,
        }
    }

    #[test]
    fn dates_are_utc() {
        assert_eq!(format_date(0), "1970-01-01");
        assert_eq!(format_date(1_762_819_200), "2025-11-11");
        assert_eq!(format_date(951_782_400), "2000-02-29");
    }

    #[test]
    fn whitespace_is_collapsed() {
        let text = format_recall(&[message(1, Role::User, "  hello\n\tworld  ", 0)]);
        assert_eq!(text, "[1970-01-01] user: hello world");
    }

    #[test]
    fn long_messages_are_cut_with_an_ellipsis() {
        let long = "a".repeat(SNIPPET_CHARS + 50);
        let text = format_recall(&[message(1, Role::Assistant, &long, 0)]);
        let snippet = text
            .strip_prefix("[1970-01-01] assistant: ")
            .expect("the line prefix");
        assert_eq!(snippet.chars().count(), SNIPPET_CHARS + 1);
        assert!(snippet.ends_with('…'));

        let exact = "b".repeat(SNIPPET_CHARS);
        let text = format_recall(&[message(2, Role::Assistant, &exact, 0)]);
        assert!(!text.ends_with('…'));
    }

    #[test]
    fn tool_rows_are_skipped() {
        let messages = [
            message(1, Role::User, "asked", 0),
            message(2, Role::Tool, "a tool result", 0),
            message(3, Role::Assistant, "answered", 0),
        ];
        let text = format_recall(&messages);
        assert_eq!(
            text,
            "[1970-01-01] user: asked\n[1970-01-01] assistant: answered"
        );
    }

    #[test]
    fn the_total_is_capped() {
        let messages: Vec<ChatMessage> = (0..20)
            .map(|id| message(id, Role::User, &"a".repeat(SNIPPET_CHARS), 0))
            .collect();
        let text = format_recall(&messages);
        assert!(text.chars().count() <= RECALL_MAX_CHARS);
        let lines = text.lines().count();
        assert!(lines > 0);
        assert!(lines < messages.len());
    }

    #[test]
    fn an_empty_input_has_no_matches() {
        assert_eq!(format_recall(&[]), NO_MATCHES);
        assert_eq!(
            format_recall(&[message(1, Role::Tool, "only a tool", 0)]),
            NO_MATCHES
        );
    }

    #[test]
    fn messages_keep_their_order() {
        let messages = [
            message(2, Role::Assistant, "second", 86_400),
            message(1, Role::User, "first", 0),
        ];
        assert_eq!(
            format_recall(&messages),
            "[1970-01-02] assistant: second\n[1970-01-01] user: first"
        );
    }
}
