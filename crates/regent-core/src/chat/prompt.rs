//! Builds the ordered prompt for a chat turn.
//!
//! The prompt is kept prefix-stable: with the same system addition the system
//! message is byte-identical on every call, so the model's prefix cache can be
//! reused across turns.

use super::compaction::{MESSAGE_TEMPLATE_COST, SUMMARY_HEADING};
use super::message::{ChatMessage, Role};

/// Tokens kept out of the prompt budget and reserved for one reply.
pub const REPLY_RESERVE_TOKENS: usize = 512;

/// The constant assistant prompt.
///
/// It contains no dates or other runtime values, so it never changes between
/// runs.
pub const SYSTEM_PROMPT: &str = "You are Regent, a private assistant that runs \
entirely on this computer. Answer clearly and helpfully, in plain language. Keep \
replies concise unless the user asks for detail. If you are unsure, say so.";

/// Who a prompt message is from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromptRole {
    /// The fixed instructions that open every prompt.
    System,
    /// A message from the person using Regent.
    User,
    /// A reply from Regent.
    Assistant,
}

/// One message in the ordered prompt sent to the model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromptMessage {
    /// Who the message is from.
    pub role: PromptRole,
    /// The message text.
    pub text: String,
}

/// The fixed prefix of every prompt: [`SYSTEM_PROMPT`] plus the trimmed
/// `system_addition` when it is not empty.
///
/// The result depends only on the addition, so it is byte-identical across
/// turns and the model's prefix cache stays valid.
pub fn system_text(system_addition: &str) -> String {
    let mut text = String::from(SYSTEM_PROMPT);
    let addition = system_addition.trim();
    if !addition.is_empty() {
        text.push_str("\n\n");
        text.push_str(addition);
    }
    text
}

/// The system message text: `system_text` with the rolling summary appended
/// when there is one.
///
/// Shared with the compaction planner so both agree on exactly what would be
/// sent.
pub(crate) fn system_text_with_summary(system_text: &str, summary: Option<&str>) -> String {
    let mut text = system_text.to_string();
    if let Some(summary) = summary {
        let summary = summary.trim();
        if !summary.is_empty() {
            text.push_str("\n\n");
            text.push_str(SUMMARY_HEADING);
            text.push('\n');
            text.push_str(summary);
        }
    }
    text
}

/// Builds the prompt for a turn.
///
/// The first message is [`system_text`] for `system_addition`, plus the
/// trimmed `summary` under [`SUMMARY_HEADING`] when there is one. It is
/// followed by `history` in order. While the total token count (each message
/// plus [`MESSAGE_TEMPLATE_COST`]) exceeds
/// `context_tokens - REPLY_RESERVE_TOKENS`, the oldest history message is
/// dropped, but the newest history message and the system message are always
/// kept.
pub fn build_prompt(
    system_addition: &str,
    summary: Option<&str>,
    history: &[ChatMessage],
    context_tokens: usize,
    count_tokens: &dyn Fn(&str) -> usize,
) -> Vec<PromptMessage> {
    let system_message = PromptMessage {
        role: PromptRole::System,
        text: system_text_with_summary(&system_text(system_addition), summary),
    };

    let mut kept: Vec<PromptMessage> = history
        .iter()
        .map(|message| PromptMessage {
            role: match message.role {
                Role::User => PromptRole::User,
                Role::Assistant => PromptRole::Assistant,
                // Tool rows get their own replay format in the tool-call task.
                Role::Tool => PromptRole::User,
            },
            text: message.text.clone(),
        })
        .collect();

    let budget = context_tokens.saturating_sub(REPLY_RESERVE_TOKENS);
    while kept.len() > 1 {
        let total = count_tokens(&system_message.text)
            + MESSAGE_TEMPLATE_COST
            + kept
                .iter()
                .map(|m| count_tokens(&m.text) + MESSAGE_TEMPLATE_COST)
                .sum::<usize>();
        if total <= budget {
            break;
        }
        // Fallback only: compaction normally keeps the prompt under the
        // trigger; this drops the oldest turns from the prompt, never from
        // history.
        kept.remove(0);
    }

    let mut prompt = Vec::with_capacity(kept.len() + 1);
    prompt.push(system_message);
    prompt.extend(kept);
    prompt
}

#[cfg(test)]
mod tests {
    use super::*;

    fn message(id: i64, role: Role, text: &str) -> ChatMessage {
        ChatMessage {
            id,
            role,
            text: text.to_string(),
            from_voice: false,
            created_at: 0,
            tool_name: None,
            tool_call: None,
        }
    }

    fn word_count(text: &str) -> usize {
        text.split_whitespace().count()
    }

    #[test]
    fn the_prompt_starts_with_the_system_message() {
        let prompt = build_prompt("", None, &[], 4096, &word_count);
        assert_eq!(prompt.len(), 1);
        assert_eq!(prompt[0].role, PromptRole::System);
        assert_eq!(prompt[0].text, SYSTEM_PROMPT);
    }

    #[test]
    fn the_addition_is_appended_to_the_system_message() {
        let prompt = build_prompt("  Be terse.  ", None, &[], 4096, &word_count);
        assert_eq!(prompt[0].text, format!("{SYSTEM_PROMPT}\n\nBe terse."));

        // Blank additions add nothing.
        let plain = build_prompt("   \n ", None, &[], 4096, &word_count);
        assert_eq!(plain[0].text, SYSTEM_PROMPT);
    }

    #[test]
    fn the_system_text_is_identical_across_calls() {
        let first = build_prompt("Same addition", None, &[], 4096, &word_count);
        let second = build_prompt("Same addition", None, &[], 4096, &word_count);
        assert_eq!(first[0].text.as_bytes(), second[0].text.as_bytes());
    }

    #[test]
    fn history_follows_in_order() {
        let history = vec![
            message(1, Role::User, "hello"),
            message(2, Role::Assistant, "hi"),
        ];
        let prompt = build_prompt("", None, &history, 4096, &word_count);
        assert_eq!(prompt.len(), 3);
        assert_eq!(prompt[1].role, PromptRole::User);
        assert_eq!(prompt[1].text, "hello");
        assert_eq!(prompt[2].role, PromptRole::Assistant);
        assert_eq!(prompt[2].text, "hi");
    }

    #[test]
    fn trimming_drops_the_oldest_but_keeps_the_newest() {
        let history = vec![
            message(1, Role::User, "one two three four five"),
            message(2, Role::Assistant, "six seven eight nine ten"),
            message(3, Role::User, "newest message"),
        ];
        // Every word counts as one token. The budget is tiny, so only the
        // newest history message can survive beside the system message.
        let prompt = build_prompt("", None, &history, REPLY_RESERVE_TOKENS + 20, &word_count);
        assert_eq!(prompt.len(), 2);
        assert_eq!(prompt[0].role, PromptRole::System);
        assert_eq!(prompt[1].role, PromptRole::User);
        assert_eq!(prompt[1].text, "newest message");
    }

    #[test]
    fn a_single_history_message_is_never_dropped() {
        let history = vec![message(1, Role::User, "this one is far too long to fit")];
        let prompt = build_prompt("", None, &history, 0, &word_count);
        assert_eq!(prompt.len(), 2);
        assert_eq!(prompt[1].text, "this one is far too long to fit");
    }

    #[test]
    fn the_summary_is_appended_to_the_system_message() {
        let prompt = build_prompt(
            "Be terse.",
            Some("  talked about rust  "),
            &[],
            4096,
            &word_count,
        );
        assert_eq!(
            prompt[0].text,
            format!(
                "{}\n\n{}\ntalked about rust",
                system_text("Be terse."),
                SUMMARY_HEADING
            )
        );
    }

    #[test]
    fn a_blank_summary_adds_nothing() {
        let prompt = build_prompt("Be terse.", Some("   \n "), &[], 4096, &word_count);
        assert_eq!(prompt[0].text, system_text("Be terse."));
    }

    #[test]
    fn the_system_text_prefix_is_unchanged_by_a_summary() {
        let prompt = build_prompt("Be terse.", Some("some summary"), &[], 4096, &word_count);
        assert!(prompt[0].text.starts_with(&system_text("Be terse.")));
        assert_eq!(
            system_text("Be terse."),
            format!("{SYSTEM_PROMPT}\n\nBe terse.")
        );
    }
}
