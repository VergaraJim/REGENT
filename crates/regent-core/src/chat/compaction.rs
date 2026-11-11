//! Pure compaction planning (spec §5.5).
//!
//! When the working prompt grows past a trigger, the oldest turns are folded
//! into a rolling summary. This module decides *what* to fold and builds the
//! summarizer prompt; it performs no I/O and never touches the history store.
//! The engine that runs the summarizer pass is wired in by a later stage.

use super::message::{ChatMessage, Role};
use super::prompt::{system_text_with_summary, PromptMessage, PromptRole};

/// The per-message template overhead in tokens.
///
/// Chat templates wrap every message in role markers and separators; this is
/// the allowance reserved for that, on top of the message text itself.
pub const MESSAGE_TEMPLATE_COST: usize = 4;

/// How many of the newest messages are always kept verbatim in the prompt.
pub const KEEP_VERBATIM: usize = 6;

/// The smallest summary cap, in tokens.
///
/// A summary shorter than this is not worth asking the model for.
pub const MIN_SUMMARY_TOKENS: usize = 32;

/// How many compaction passes one turn may run before it gives up.
pub const MAX_COMPACTION_PASSES: usize = 4;

/// The sampling temperature used for a summarizer pass.
pub const SUMMARY_TEMPERATURE: f32 = 0.2;

/// The heading that introduces the rolling summary in the system message.
pub const SUMMARY_HEADING: &str = "Summary of the conversation so far:";

/// The system prompt for a summarizer pass.
pub const SUMMARIZER_SYSTEM_PROMPT: &str = "You maintain a running summary of a \
conversation between a user and Regent, their assistant. Merge the previous \
summary and the new messages into one updated summary. Keep names, facts, \
preferences, decisions and open tasks. Drop small talk. Write plain prose. Reply \
with the summary only.";

/// The most tokens one summarizer pass may use, prompt plus output.
///
/// Reserving half the window means the temporary summarizer context needs at
/// most half a window, so it can always coexist with the working prompt.
pub fn summarizer_budget(context_tokens: usize) -> usize {
    context_tokens / 2
}

/// The most tokens an accepted summary may contain.
///
/// It is 40% of the target working prompt, but never less than
/// [`MIN_SUMMARY_TOKENS`].
pub fn summary_cap(target_tokens: usize) -> usize {
    MIN_SUMMARY_TOKENS.max(target_tokens.saturating_mul(40) / 100)
}

/// Counts the tokens of exactly what [`super::prompt::build_prompt`] would
/// send before any trimming.
///
/// That is the system message (with `summary` appended the same way) plus
/// every message, each with [`MESSAGE_TEMPLATE_COST`] on top.
pub fn working_memory_tokens(
    system_text: &str,
    summary: Option<&str>,
    messages: &[ChatMessage],
    count_tokens: &dyn Fn(&str) -> usize,
) -> usize {
    let system = system_text_with_summary(system_text, summary);
    let mut total = count_tokens(&system) + MESSAGE_TEMPLATE_COST;
    for message in messages {
        total += count_tokens(&message.text) + MESSAGE_TEMPLATE_COST;
    }
    total
}

/// What one compaction step would do.
#[derive(Debug, Clone, PartialEq)]
pub struct CompactionPlan {
    /// The summary carried over from previous compactions, trimmed.
    pub previous_summary: Option<String>,
    /// The messages to fold into the new summary, oldest first.
    pub fold: Vec<ChatMessage>,
    /// The id of the newest message the new summary covers.
    pub covers_through: i64,
    /// The token cap for the accepted summary.
    pub cap_tokens: usize,
}

/// Builds the two-message prompt for a summarizer pass.
///
/// The system message is [`SUMMARIZER_SYSTEM_PROMPT`]. The user message holds
/// the previous summary (when there is one) and every folded message, one per
/// line, labelled `User:` or `Regent:`.
pub fn build_summary_prompt(
    previous_summary: Option<&str>,
    fold: &[ChatMessage],
) -> Vec<PromptMessage> {
    let mut text = String::new();
    if let Some(previous) = previous_summary {
        let previous = previous.trim();
        if !previous.is_empty() {
            text.push_str("Previous summary:\n");
            text.push_str(previous);
            text.push_str("\n\n");
        }
    }
    text.push_str("New messages:\n");
    let lines: Vec<String> = fold
        .iter()
        .map(|message| {
            let speaker = match message.role {
                Role::User => "User",
                Role::Assistant => "Regent",
                Role::Tool => "Tool",
            };
            format!("{speaker}: {}", message.text)
        })
        .collect();
    text.push_str(&lines.join("\n"));
    text.push_str("\n\nWrite the updated summary.");

    vec![
        PromptMessage {
            role: PromptRole::System,
            text: SUMMARIZER_SYSTEM_PROMPT.to_string(),
        },
        PromptMessage {
            role: PromptRole::User,
            text,
        },
    ]
}

/// The token count of [`build_summary_prompt`], text only.
fn summary_prompt_tokens(
    previous_summary: Option<&str>,
    fold: &[ChatMessage],
    count_tokens: &dyn Fn(&str) -> usize,
) -> usize {
    build_summary_prompt(previous_summary, fold)
        .iter()
        .map(|message| count_tokens(&message.text))
        .sum()
}

/// Plans one compaction step, or returns `None` when none is needed.
///
/// A plan is made only when the working memory exceeds `threshold` and more
/// than [`KEEP_VERBATIM`] messages exist. The newest [`KEEP_VERBATIM`] messages
/// are kept verbatim; the older candidates are folded oldest first. The first
/// candidate is always folded, and further ones are added while the summarizer
/// prompt, its template overhead and the summary cap stay within `budget`.
pub fn plan_compaction(
    system_text: &str,
    summary: Option<&str>,
    messages: &[ChatMessage],
    threshold: usize,
    target: usize,
    budget: usize,
    count_tokens: &dyn Fn(&str) -> usize,
) -> Option<CompactionPlan> {
    if working_memory_tokens(system_text, summary, messages, count_tokens) <= threshold {
        return None;
    }
    if messages.len() <= KEEP_VERBATIM {
        return None;
    }

    let candidates = &messages[..messages.len() - KEEP_VERBATIM];
    let cap_tokens = summary_cap(target);

    let mut fold: Vec<ChatMessage> = Vec::new();
    for candidate in candidates {
        let mut trial = fold.clone();
        trial.push(candidate.clone());
        let cost = summary_prompt_tokens(summary, &trial, count_tokens)
            + MESSAGE_TEMPLATE_COST * 2
            + cap_tokens;
        if fold.is_empty() || cost <= budget {
            fold = trial;
        } else {
            break;
        }
    }

    Some(CompactionPlan {
        previous_summary: summary
            .map(str::trim)
            .filter(|summary| !summary.is_empty())
            .map(str::to_string),
        covers_through: fold.last()?.id,
        fold,
        cap_tokens,
    })
}

/// Accepts a summarizer reply, or returns `None` when it is unusable.
///
/// The text is trimmed; an empty result or one counting more than
/// `cap_tokens` is rejected.
pub fn accept_summary(
    text: &str,
    cap_tokens: usize,
    count_tokens: &dyn Fn(&str) -> usize,
) -> Option<String> {
    let text = text.trim();
    if text.is_empty() || count_tokens(text) > cap_tokens {
        return None;
    }
    Some(text.to_string())
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

    fn ten_messages() -> Vec<ChatMessage> {
        (1..=10)
            .map(|id| {
                let role = if id % 2 == 0 {
                    Role::Assistant
                } else {
                    Role::User
                };
                message(id, role, "one")
            })
            .collect()
    }

    #[test]
    fn summary_cap_is_forty_percent_with_a_floor() {
        assert_eq!(summary_cap(0), 32);
        assert_eq!(summary_cap(50), 32);
        assert_eq!(summary_cap(1000), 400);
    }

    #[test]
    fn working_memory_counts_every_message_and_template() {
        let messages = vec![
            message(1, Role::User, "hello world"),
            message(2, Role::Assistant, "hi"),
        ];
        // system: "sys text" is 2 words + 4 template = 6
        // "hello world": 2 + 4 = 6, "hi": 1 + 4 = 5
        assert_eq!(
            working_memory_tokens("sys text", None, &messages, &word_count),
            17
        );
    }

    #[test]
    fn no_plan_at_or_under_the_threshold() {
        let messages = ten_messages();
        let tokens = working_memory_tokens("sys", None, &messages, &word_count);
        assert!(
            plan_compaction("sys", None, &messages, tokens, 100, 10_000, &word_count).is_none()
        );
        assert!(
            plan_compaction("sys", None, &messages, tokens + 1, 100, 10_000, &word_count).is_none()
        );
    }

    #[test]
    fn no_plan_with_six_or_fewer_messages() {
        for count in 1..=KEEP_VERBATIM {
            let messages: Vec<ChatMessage> = (1..=count as i64)
                .map(|id| message(id, Role::User, "one"))
                .collect();
            assert!(
                plan_compaction("sys", None, &messages, 0, 100, 10_000, &word_count).is_none(),
                "{count} messages should not compact"
            );
        }
    }

    #[test]
    fn a_plan_keeps_the_newest_six_and_folds_the_oldest_four() {
        let messages = ten_messages();
        let plan = plan_compaction("sys", None, &messages, 10, 100, 10_000, &word_count)
            .expect("a plan is needed");
        assert_eq!(plan.fold.len(), 4);
        assert_eq!(
            plan.fold.iter().map(|m| m.id).collect::<Vec<_>>(),
            vec![1, 2, 3, 4]
        );
        assert_eq!(plan.covers_through, 4);
        assert_eq!(plan.cap_tokens, 40);
        assert_eq!(plan.previous_summary, None);
    }

    #[test]
    fn a_small_budget_folds_fewer_but_at_least_one() {
        let messages = ten_messages();
        let plan = plan_compaction("sys", None, &messages, 10, 100, 0, &word_count)
            .expect("a plan is needed");
        assert_eq!(plan.fold.len(), 1);
        assert_eq!(plan.covers_through, messages[0].id);

        // A budget that fits exactly one folded message: the first is always
        // included, the second would push the cost past the budget.
        let one_fold = summary_prompt_tokens(None, &messages[..1], &word_count)
            + MESSAGE_TEMPLATE_COST * 2
            + summary_cap(100);
        let plan = plan_compaction("sys", None, &messages, 10, 100, one_fold, &word_count)
            .expect("a plan is needed");
        assert_eq!(plan.fold.len(), 1);
    }

    #[test]
    fn the_summary_prompt_carries_the_previous_summary_and_speakers() {
        let fold = vec![
            message(1, Role::User, "hello"),
            message(2, Role::Assistant, "hi there"),
        ];

        let prompt = build_summary_prompt(Some("  old facts  "), &fold);
        assert_eq!(prompt.len(), 2);
        assert_eq!(prompt[0].role, PromptRole::System);
        assert_eq!(prompt[0].text, SUMMARIZER_SYSTEM_PROMPT);
        assert_eq!(prompt[1].role, PromptRole::User);
        assert_eq!(
            prompt[1].text,
            "Previous summary:\nold facts\n\nNew messages:\nUser: hello\nRegent: hi there\n\nWrite the updated summary."
        );

        let fresh = build_summary_prompt(None, &fold);
        assert_eq!(
            fresh[1].text,
            "New messages:\nUser: hello\nRegent: hi there\n\nWrite the updated summary."
        );
        assert!(!fresh[1].text.contains("Previous summary:"));

        let blank = build_summary_prompt(Some("   "), &fold);
        assert!(!blank[1].text.contains("Previous summary:"));
    }

    #[test]
    fn accept_summary_rejects_empty_and_over_cap_text() {
        assert_eq!(accept_summary("   ", 10, &word_count), None);
        assert_eq!(accept_summary("one two three", 2, &word_count), None);
        assert_eq!(
            accept_summary("  one two  ", 2, &word_count),
            Some("one two".to_string())
        );
    }
}
