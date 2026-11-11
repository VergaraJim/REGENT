//! Builds the ordered prompt for a chat turn.
//!
//! The prompt is kept prefix-stable: with the same system addition the system
//! message is byte-identical on every call, so the model's prefix cache can be
//! reused across turns.

use super::compaction::{MESSAGE_TEMPLATE_COST, SUMMARY_HEADING};
use super::message::{ChatMessage, Role};
use super::tools::{call_block, TOOL_RESULT_HEADING};

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
    /// The result of a tool call Regent made.
    Tool,
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

/// The whole fixed prefix of every prompt: [`system_text`] followed by the
/// tools section when it is not empty.
///
/// This is used for the prefix cache and by the compaction planner, so both
/// agree with what [`build_prompt`] sends before any summary or history.
pub fn fixed_prefix(system_addition: &str, tools_section: &str) -> String {
    let mut text = system_text(system_addition);
    if !tools_section.is_empty() {
        text.push_str("\n\n");
        text.push_str(tools_section);
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
/// The first message is [`fixed_prefix`] for `system_addition` and
/// `tools_section`, plus the trimmed `summary` under [`SUMMARY_HEADING`] when
/// there is one. It is followed by `history` in order; tool rows replay their
/// call block on the preceding assistant message and then appear as a tool
/// result. While the total token count (each message plus
/// [`MESSAGE_TEMPLATE_COST`]) exceeds
/// `context_tokens - REPLY_RESERVE_TOKENS`, the oldest history message is
/// dropped, but the newest history message and the system message are always
/// kept.
pub fn build_prompt(
    system_addition: &str,
    tools_section: &str,
    summary: Option<&str>,
    history: &[ChatMessage],
    context_tokens: usize,
    count_tokens: &dyn Fn(&str) -> usize,
) -> Vec<PromptMessage> {
    let system_message = PromptMessage {
        role: PromptRole::System,
        text: system_text_with_summary(&fixed_prefix(system_addition, tools_section), summary),
    };

    let mut kept: Vec<PromptMessage> = Vec::with_capacity(history.len());
    for message in history {
        match message.role {
            Role::User => kept.push(PromptMessage {
                role: PromptRole::User,
                text: message.text.clone(),
            }),
            Role::Assistant => kept.push(PromptMessage {
                role: PromptRole::Assistant,
                text: message.text.clone(),
            }),
            Role::Tool => {
                let block = call_block(message.tool_call.as_deref());
                match kept.last_mut() {
                    Some(previous) if previous.role == PromptRole::Assistant => {
                        if !previous.text.is_empty() {
                            previous.text.push('\n');
                        }
                        previous.text.push_str(&block);
                    }
                    _ => kept.push(PromptMessage {
                        role: PromptRole::Assistant,
                        text: block,
                    }),
                }
                kept.push(PromptMessage {
                    role: PromptRole::Tool,
                    text: format!("{TOOL_RESULT_HEADING}\n{}", message.text),
                });
            }
        }
    }

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

    fn tool_message(id: i64, text: &str, call: Option<&str>) -> ChatMessage {
        ChatMessage {
            id,
            role: Role::Tool,
            text: text.to_string(),
            from_voice: false,
            created_at: 0,
            tool_name: Some("search_memory".to_string()),
            tool_call: call.map(str::to_string),
        }
    }

    #[test]
    fn the_prompt_starts_with_the_system_message() {
        let prompt = build_prompt("", "", None, &[], 4096, &word_count);
        assert_eq!(prompt.len(), 1);
        assert_eq!(prompt[0].role, PromptRole::System);
        assert_eq!(prompt[0].text, SYSTEM_PROMPT);
    }

    #[test]
    fn the_addition_is_appended_to_the_system_message() {
        let prompt = build_prompt("  Be terse.  ", "", None, &[], 4096, &word_count);
        assert_eq!(prompt[0].text, format!("{SYSTEM_PROMPT}\n\nBe terse."));

        // Blank additions add nothing.
        let plain = build_prompt("   \n ", "", None, &[], 4096, &word_count);
        assert_eq!(plain[0].text, SYSTEM_PROMPT);
    }

    #[test]
    fn the_system_text_is_identical_across_calls() {
        let first = build_prompt("Same addition", "", None, &[], 4096, &word_count);
        let second = build_prompt("Same addition", "", None, &[], 4096, &word_count);
        assert_eq!(first[0].text.as_bytes(), second[0].text.as_bytes());
    }

    #[test]
    fn history_follows_in_order() {
        let history = vec![
            message(1, Role::User, "hello"),
            message(2, Role::Assistant, "hi"),
        ];
        let prompt = build_prompt("", "", None, &history, 4096, &word_count);
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
        let prompt = build_prompt(
            "",
            "",
            None,
            &history,
            REPLY_RESERVE_TOKENS + 20,
            &word_count,
        );
        assert_eq!(prompt.len(), 2);
        assert_eq!(prompt[0].role, PromptRole::System);
        assert_eq!(prompt[1].role, PromptRole::User);
        assert_eq!(prompt[1].text, "newest message");
    }

    #[test]
    fn a_single_history_message_is_never_dropped() {
        let history = vec![message(1, Role::User, "this one is far too long to fit")];
        let prompt = build_prompt("", "", None, &history, 0, &word_count);
        assert_eq!(prompt.len(), 2);
        assert_eq!(prompt[1].text, "this one is far too long to fit");
    }

    #[test]
    fn the_summary_is_appended_to_the_system_message() {
        let prompt = build_prompt(
            "Be terse.",
            "",
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
        let prompt = build_prompt("Be terse.", "", Some("   \n "), &[], 4096, &word_count);
        assert_eq!(prompt[0].text, system_text("Be terse."));
    }

    #[test]
    fn the_system_text_prefix_is_unchanged_by_a_summary() {
        let prompt = build_prompt(
            "Be terse.",
            "",
            Some("some summary"),
            &[],
            4096,
            &word_count,
        );
        assert!(prompt[0].text.starts_with(&system_text("Be terse.")));
        assert_eq!(
            system_text("Be terse."),
            format!("{SYSTEM_PROMPT}\n\nBe terse.")
        );
    }

    #[test]
    fn the_tools_section_appears_only_when_non_empty() {
        let plain = build_prompt("Be terse.", "", None, &[], 4096, &word_count);
        assert_eq!(plain[0].text, system_text("Be terse."));

        let with_tools = build_prompt("Be terse.", "Tools:\n- x", None, &[], 4096, &word_count);
        assert_eq!(
            with_tools[0].text,
            format!("{}\n\nTools:\n- x", system_text("Be terse."))
        );

        assert_eq!(
            fixed_prefix("Be terse.", "Tools:"),
            format!("{}\n\nTools:", system_text("Be terse."))
        );
        assert_eq!(fixed_prefix("Be terse.", ""), system_text("Be terse."));
    }

    #[test]
    fn a_tool_row_replays_its_call_on_the_previous_assistant_message() {
        let history = vec![
            message(1, Role::User, "hi"),
            message(2, Role::Assistant, "Let me check."),
            tool_message(
                3,
                "Found it in the notes.",
                Some("{\"name\": \"search_memory\"}"),
            ),
            message(4, Role::Assistant, "Found it."),
        ];
        let prompt = build_prompt("", "", None, &history, 4096, &word_count);
        assert_eq!(prompt.len(), 5);
        assert_eq!(prompt[0].role, PromptRole::System);
        assert_eq!(prompt[1].role, PromptRole::User);
        assert_eq!(prompt[1].text, "hi");
        assert_eq!(prompt[2].role, PromptRole::Assistant);
        assert_eq!(
            prompt[2].text,
            "Let me check.\n<tool_call>{\"name\": \"search_memory\"}</tool_call>"
        );
        assert_eq!(prompt[3].role, PromptRole::Tool);
        assert_eq!(prompt[3].text, "Tool result:\nFound it in the notes.");
        assert_eq!(prompt[4].role, PromptRole::Assistant);
        assert_eq!(prompt[4].text, "Found it.");
    }

    #[test]
    fn a_tool_row_without_a_preceding_assistant_synthesizes_a_block() {
        let history = vec![
            message(1, Role::User, "hi"),
            tool_message(2, "no call recorded", None),
            message(3, Role::Assistant, "done"),
        ];
        let prompt = build_prompt("", "", None, &history, 4096, &word_count);
        assert_eq!(prompt.len(), 5);
        assert_eq!(prompt[1].role, PromptRole::User);
        assert_eq!(prompt[2].role, PromptRole::Assistant);
        assert_eq!(prompt[2].text, "<tool_call>{}</tool_call>");
        assert_eq!(prompt[3].role, PromptRole::Tool);
        assert_eq!(prompt[3].text, "Tool result:\nno call recorded");
        assert_eq!(prompt[4].role, PromptRole::Assistant);
    }
}
