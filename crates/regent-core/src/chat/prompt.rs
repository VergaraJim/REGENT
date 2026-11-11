//! Builds the ordered prompt for a chat turn.
//!
//! The prompt is kept prefix-stable: with the same system addition the system
//! message is byte-identical on every call, so the model's prefix cache can be
//! reused across turns.

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

/// Builds the prompt for a turn.
///
/// The first message is the system prompt plus the trimmed
/// `system_addition` (when it is not empty). It is followed by `history` in
/// order. While the total token count exceeds
/// `context_tokens - REPLY_RESERVE_TOKENS`, the oldest history message is
/// dropped, but the newest history message and the system message are always
/// kept.
///
/// Real compaction arrives in a later stage; this function only trims.
pub fn build_prompt(
    system_addition: &str,
    history: &[ChatMessage],
    context_tokens: usize,
    count_tokens: &dyn Fn(&str) -> usize,
) -> Vec<PromptMessage> {
    let mut system = String::from(SYSTEM_PROMPT);
    let addition = system_addition.trim();
    if !addition.is_empty() {
        system.push_str("\n\n");
        system.push_str(addition);
    }

    let system_message = PromptMessage {
        role: PromptRole::System,
        text: system,
    };

    let mut kept: Vec<PromptMessage> = history
        .iter()
        .map(|message| PromptMessage {
            role: match message.role {
                Role::User => PromptRole::User,
                Role::Assistant => PromptRole::Assistant,
            },
            text: message.text.clone(),
        })
        .collect();

    let budget = context_tokens.saturating_sub(REPLY_RESERVE_TOKENS);
    while kept.len() > 1 {
        let total = count_tokens(&system_message.text)
            + kept.iter().map(|m| count_tokens(&m.text)).sum::<usize>();
        if total <= budget {
            break;
        }
        // TODO(stage 4): replace dropping with compaction.
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
        }
    }

    fn word_count(text: &str) -> usize {
        text.split_whitespace().count()
    }

    #[test]
    fn the_prompt_starts_with_the_system_message() {
        let prompt = build_prompt("", &[], 4096, &word_count);
        assert_eq!(prompt.len(), 1);
        assert_eq!(prompt[0].role, PromptRole::System);
        assert_eq!(prompt[0].text, SYSTEM_PROMPT);
    }

    #[test]
    fn the_addition_is_appended_to_the_system_message() {
        let prompt = build_prompt("  Be terse.  ", &[], 4096, &word_count);
        assert_eq!(prompt[0].text, format!("{SYSTEM_PROMPT}\n\nBe terse."));

        // Blank additions add nothing.
        let plain = build_prompt("   \n ", &[], 4096, &word_count);
        assert_eq!(plain[0].text, SYSTEM_PROMPT);
    }

    #[test]
    fn the_system_text_is_identical_across_calls() {
        let first = build_prompt("Same addition", &[], 4096, &word_count);
        let second = build_prompt("Same addition", &[], 4096, &word_count);
        assert_eq!(first[0].text.as_bytes(), second[0].text.as_bytes());
    }

    #[test]
    fn history_follows_in_order() {
        let history = vec![
            message(1, Role::User, "hello"),
            message(2, Role::Assistant, "hi"),
        ];
        let prompt = build_prompt("", &history, 4096, &word_count);
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
        let prompt = build_prompt("", &history, REPLY_RESERVE_TOKENS + 20, &word_count);
        assert_eq!(prompt.len(), 2);
        assert_eq!(prompt[0].role, PromptRole::System);
        assert_eq!(prompt[1].role, PromptRole::User);
        assert_eq!(prompt[1].text, "newest message");
    }

    #[test]
    fn a_single_history_message_is_never_dropped() {
        let history = vec![message(1, Role::User, "this one is far too long to fit")];
        let prompt = build_prompt("", &history, 0, &word_count);
        assert_eq!(prompt.len(), 2);
        assert_eq!(prompt[1].text, "this one is far too long to fit");
    }
}
