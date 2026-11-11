//! The plain-text tool-call protocol (spec §5.7).
//!
//! Everything here is pure logic: the streaming parser that hides
//! `<tool_call>…</tool_call>` blocks from the visible reply, validation of a
//! call against the tool specs, and the constant tools section of the system
//! prompt. The chat service is wired to this by a later task.

use serde_json::{Map, Value};

/// The opening marker of a tool call.
pub const TOOL_CALL_OPEN: &str = "<tool_call>";
/// The closing marker of a tool call.
pub const TOOL_CALL_CLOSE: &str = "</tool_call>";
/// How many tool calls one turn may make.
pub const MAX_TOOL_CALLS_PER_TURN: usize = 3;
/// The heading that introduces a tool result in the prompt.
pub const TOOL_RESULT_HEADING: &str = "Tool result:";
/// The reminder sent once the turn has used all its tool calls.
pub const ANSWER_NOW: &str = "You have used all your tool calls. Answer the user now.";

/// The protocol text that opens the tools section.
const TOOLS_INTRO: &str = "You can use tools. To call one, write exactly one block like\n\
<tool_call>{\"name\": \"TOOL_NAME\", \"arguments\": {\"ARGUMENT\": \"VALUE\"}}</tool_call>\n\
and stop. The result arrives in the next message. Only call a tool when it helps you answer.\n\
Tools:";

/// The kind of value a tool parameter accepts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParamKind {
    /// Free text; a non-blank string.
    Text,
    /// A whole number within an inclusive range.
    Integer {
        /// The smallest accepted value.
        min: i64,
        /// The largest accepted value.
        max: i64,
    },
}

/// One parameter of a tool.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParamSpec {
    /// The argument name.
    pub name: &'static str,
    /// The accepted kind of value.
    pub kind: ParamKind,
    /// Whether the argument must be present.
    pub required: bool,
    /// One plain-language line describing the argument.
    pub description: &'static str,
}

/// The description of one tool.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolSpec {
    /// The tool name, as written in the call.
    pub name: &'static str,
    /// One plain-language line describing the tool.
    pub description: &'static str,
    /// The tool's parameters, in the order they are listed.
    pub params: Vec<ParamSpec>,
}

/// The spec of the `search_memory` tool.
pub fn search_memory_spec() -> ToolSpec {
    ToolSpec {
        name: "search_memory",
        description: "Search earlier conversations for facts, preferences, \
decisions or anything the user said before.",
        params: vec![
            ParamSpec {
                name: "query",
                kind: ParamKind::Text,
                required: true,
                description: "What to look for.",
            },
            ParamSpec {
                name: "limit",
                kind: ParamKind::Integer { min: 1, max: 8 },
                required: false,
                description: "How many messages to return (default 4).",
            },
        ],
    }
}

/// The constant tools section of the system prompt.
///
/// The output depends only on the specs, never on runtime values, so the
/// prompt prefix stays byte-identical across turns.
pub fn tools_section(tools: &[ToolSpec]) -> String {
    if tools.is_empty() {
        return String::new();
    }

    let mut section = String::from(TOOLS_INTRO);
    let mut sorted: Vec<&ToolSpec> = tools.iter().collect();
    sorted.sort_by_key(|tool| tool.name);
    for tool in sorted {
        section.push('\n');
        section.push_str(&tool_line(tool));
    }
    section
}

/// One `- name(args): description args-descriptions` line for a tool.
fn tool_line(tool: &ToolSpec) -> String {
    let params = tool
        .params
        .iter()
        .map(|param| {
            let marker = if param.required { "" } else { "?" };
            format!("{}{}: {}", param.name, marker, kind_text(&param.kind))
        })
        .collect::<Vec<_>>()
        .join(", ");

    let mut line = format!("- {}({}): {}", tool.name, params, tool.description);
    let descriptions = tool
        .params
        .iter()
        .map(|param| format!("{}: {}", param.name, param.description))
        .collect::<Vec<_>>();
    if !descriptions.is_empty() {
        line.push(' ');
        line.push_str(&descriptions.join("; "));
    }
    line
}

/// How a parameter kind is written in the tools section.
fn kind_text(kind: &ParamKind) -> String {
    match kind {
        ParamKind::Text => "text".to_string(),
        ParamKind::Integer { min, max } => format!("whole number {min}-{max}"),
    }
}

/// A validated tool call.
#[derive(Debug, Clone, PartialEq)]
pub struct ValidCall {
    /// The tool name.
    pub name: String,
    /// The validated arguments.
    pub arguments: Map<String, Value>,
}

/// Validates a raw tool-call body (the text between the markers).
///
/// Errors are the exact texts stored as the tool result, so the model sees a
/// plain-language explanation of what went wrong.
pub fn validate_call(raw: &str, tools: &[ToolSpec]) -> Result<ValidCall, String> {
    let value: Value = serde_json::from_str(raw)
        .map_err(|_| "The tool call was not valid JSON with a \"name\".".to_string())?;
    let Value::Object(object) = &value else {
        return Err("The tool call was not valid JSON with a \"name\".".to_string());
    };
    let Some(name) = object.get("name").and_then(Value::as_str) else {
        return Err("The tool call was not valid JSON with a \"name\".".to_string());
    };
    let Some(tool) = tools.iter().find(|tool| tool.name == name) else {
        let available = tools
            .iter()
            .map(|tool| tool.name)
            .collect::<Vec<_>>()
            .join(", ");
        return Err(format!(
            "Unknown tool \"{name}\". Available tools: {available}."
        ));
    };

    let arguments = match object.get("arguments") {
        None => Map::new(),
        Some(Value::Object(arguments)) => arguments.clone(),
        Some(_) => return Err(format!("The \"arguments\" of {name} must be an object.")),
    };

    for argument in arguments.keys() {
        if !tool.params.iter().any(|param| param.name == argument) {
            return Err(format!("Unknown argument \"{argument}\" for {name}."));
        }
    }

    for param in &tool.params {
        if param.required && !arguments.contains_key(param.name) {
            return Err(format!("Missing argument \"{}\" for {name}.", param.name));
        }
    }

    for param in &tool.params {
        let Some(value) = arguments.get(param.name) else {
            continue;
        };
        match param.kind {
            ParamKind::Text => {
                if value.as_str().is_none_or(|text| text.trim().is_empty()) {
                    return Err(format!("Argument \"{}\" must be text.", param.name));
                }
            }
            ParamKind::Integer { min, max } => {
                let in_range = value
                    .as_i64()
                    .is_some_and(|number| number >= min && number <= max);
                if !in_range {
                    return Err(format!(
                        "Argument \"{}\" must be a whole number from {min} to {max}.",
                        param.name
                    ));
                }
            }
        }
    }

    Ok(ValidCall {
        name: name.to_string(),
        arguments,
    })
}

/// The call block a tool row replays in the prompt.
pub(crate) fn call_block(tool_call: Option<&str>) -> String {
    format!(
        "{TOOL_CALL_OPEN}{}{TOOL_CALL_CLOSE}",
        tool_call.unwrap_or("{}")
    )
}

/// One piece handed back by [`ToolCallParser::push`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ParsedPiece {
    /// The text that belongs in the visible reply.
    pub visible: String,
    /// The body of a completed call, when one finishes in this piece.
    pub call: Option<String>,
}

/// The state of the streaming tool-call filter.
#[derive(Debug, Clone, Default)]
pub struct ToolCallParser {
    /// Ordinary text that could still turn out to start a call marker.
    pending: String,
    /// Whether a call block is being collected.
    in_block: bool,
    /// The text collected inside the current block.
    block: String,
    /// Whether a call has already completed; everything after it is ignored.
    after_call: bool,
}

impl ToolCallParser {
    /// A parser that has seen nothing yet.
    pub fn new() -> Self {
        Self::default()
    }

    /// Feeds one streamed piece and returns the visible text and any call.
    pub fn push(&mut self, piece: &str) -> ParsedPiece {
        if self.after_call {
            return ParsedPiece::default();
        }

        if self.in_block {
            self.block.push_str(piece);
            if let Some(end) = self.block.find(TOOL_CALL_CLOSE) {
                let call = self.block[..end].trim().to_string();
                self.block.clear();
                self.in_block = false;
                self.after_call = true;
                return ParsedPiece {
                    visible: String::new(),
                    call: Some(call),
                };
            }
            return ParsedPiece::default();
        }

        let mut text = std::mem::take(&mut self.pending);
        text.push_str(piece);
        let mut visible = String::new();
        if let Some(start) = text.find(TOOL_CALL_OPEN) {
            visible.push_str(&text[..start]);
            text.drain(..start + TOOL_CALL_OPEN.len());
            self.in_block = true;
            self.block = text;
            if let Some(end) = self.block.find(TOOL_CALL_CLOSE) {
                let call = self.block[..end].trim().to_string();
                self.block.clear();
                self.in_block = false;
                self.after_call = true;
                return ParsedPiece {
                    visible,
                    call: Some(call),
                };
            }
            return ParsedPiece {
                visible,
                call: None,
            };
        }

        let hold = partial_open_len(&text);
        let split = text.len() - hold;
        visible.push_str(&text[..split]);
        self.pending = text[split..].to_string();
        ParsedPiece {
            visible,
            call: None,
        }
    }

    /// Ends the stream, returning any held-back ordinary text.
    ///
    /// An unterminated block is dropped; ordinary text before it was already
    /// returned by [`ToolCallParser::push`].
    pub fn finish(self) -> String {
        self.pending
    }
}

/// The length of the longest suffix of `text` that could start a call marker.
fn partial_open_len(text: &str) -> usize {
    let longest = TOOL_CALL_OPEN.len() - 1;
    for length in (1..=longest.min(text.len())).rev() {
        if text.ends_with(&TOOL_CALL_OPEN[..length]) {
            return length;
        }
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn specs() -> Vec<ToolSpec> {
        vec![search_memory_spec()]
    }

    #[test]
    fn plain_text_passes_through() {
        let mut parser = ToolCallParser::new();
        assert_eq!(
            parser.push("Hello there"),
            ParsedPiece {
                visible: "Hello there".to_string(),
                call: None,
            }
        );
        assert_eq!(parser.finish(), "");
    }

    #[test]
    fn a_tag_split_across_pieces_is_recognised() {
        let mut parser = ToolCallParser::new();
        let mut visible = String::new();
        let mut call = None;
        for piece in [
            "Sure <to",
            "ol_call>{\"name\":",
            "\"x\"}</tool",
            "_call> trailing",
        ] {
            let parsed = parser.push(piece);
            visible.push_str(&parsed.visible);
            if parsed.call.is_some() {
                call = parsed.call;
            }
        }
        assert_eq!(visible, "Sure ");
        assert_eq!(call.as_deref(), Some("{\"name\":\"x\"}"));
    }

    #[test]
    fn a_held_back_angle_bracket_comes_out_as_text() {
        let mut parser = ToolCallParser::new();
        assert_eq!(parser.push("<").visible, "");
        assert_eq!(parser.push("b>").visible, "<b>");
        assert_eq!(parser.finish(), "");
    }

    #[test]
    fn an_unterminated_block_is_dropped_and_earlier_text_kept() {
        let mut parser = ToolCallParser::new();
        assert_eq!(
            parser.push("before <tool_call>{\"name\":").visible,
            "before "
        );
        assert_eq!(parser.finish(), "");
    }

    #[test]
    fn a_partial_marker_at_finish_is_returned_as_text() {
        let mut parser = ToolCallParser::new();
        assert_eq!(parser.push("text <tool_ca").visible, "text ");
        assert_eq!(parser.finish(), "<tool_ca");
    }

    #[test]
    fn tools_section_is_empty_for_no_tools() {
        assert_eq!(tools_section(&[]), "");
    }

    #[test]
    fn tools_section_is_sorted_and_constant() {
        let mut alpha = search_memory_spec();
        alpha.name = "alpha";
        let mut zeta = search_memory_spec();
        zeta.name = "zeta";
        let tools = vec![zeta, alpha];
        let section = tools_section(&tools);
        let alpha_at = section.find("- alpha(").expect("alpha line");
        let zeta_at = section.find("- zeta(").expect("zeta line");
        assert!(alpha_at < zeta_at, "{section}");
        assert_eq!(section, tools_section(&tools));
    }

    #[test]
    fn tools_section_contains_the_search_memory_line() {
        let section = tools_section(&specs());
        assert!(section.contains("- search_memory("), "{section}");
        assert!(section.contains("query: text"), "{section}");
        assert!(section.contains("limit?: whole number 1-8"), "{section}");
        assert!(
            section.contains("Search earlier conversations"),
            "{section}"
        );
        assert!(section.contains("query: What to look for."), "{section}");
    }

    #[test]
    fn validation_rejects_malformed_json() {
        let expected = "The tool call was not valid JSON with a \"name\".";
        assert_eq!(validate_call("not json", &specs()).unwrap_err(), expected);
        assert_eq!(validate_call("[]", &specs()).unwrap_err(), expected);
        assert_eq!(
            validate_call("{\"arguments\": {}}", &specs()).unwrap_err(),
            expected
        );
    }

    #[test]
    fn validation_rejects_unknown_names_and_arguments() {
        assert_eq!(
            validate_call("{\"name\": \"nope\"}", &specs()).unwrap_err(),
            "Unknown tool \"nope\". Available tools: search_memory."
        );
        assert_eq!(
            validate_call(
                "{\"name\": \"search_memory\", \"arguments\": {\"query\": \"x\", \"other\": 1}}",
                &specs()
            )
            .unwrap_err(),
            "Unknown argument \"other\" for search_memory."
        );
    }

    #[test]
    fn validation_requires_an_object_for_arguments() {
        assert_eq!(
            validate_call("{\"name\": \"search_memory\", \"arguments\": []}", &specs())
                .unwrap_err(),
            "The \"arguments\" of search_memory must be an object."
        );
    }

    #[test]
    fn validation_reports_missing_required_arguments() {
        assert_eq!(
            validate_call("{\"name\": \"search_memory\"}", &specs()).unwrap_err(),
            "Missing argument \"query\" for search_memory."
        );
    }

    #[test]
    fn validation_checks_argument_kinds() {
        assert_eq!(
            validate_call(
                "{\"name\": \"search_memory\", \"arguments\": {\"query\": \"  \"}}",
                &specs()
            )
            .unwrap_err(),
            "Argument \"query\" must be text."
        );
        assert_eq!(
            validate_call(
                "{\"name\": \"search_memory\", \"arguments\": {\"query\": 5}}",
                &specs()
            )
            .unwrap_err(),
            "Argument \"query\" must be text."
        );
        for limit in ["9", "0", "1.5", "\"4\""] {
            assert_eq!(
                validate_call(
                    &format!(
                        "{{\"name\": \"search_memory\", \"arguments\": {{\"query\": \"x\", \"limit\": {limit}}}}}"
                    ),
                    &specs()
                )
                .unwrap_err(),
                "Argument \"limit\" must be a whole number from 1 to 8."
            );
        }
    }

    #[test]
    fn validation_accepts_a_call_with_and_without_limit() {
        let call = validate_call(
            "{\"name\": \"search_memory\", \"arguments\": {\"query\": \"weather\"}}",
            &specs(),
        )
        .expect("a valid call");
        assert_eq!(call.name, "search_memory");
        assert_eq!(call.arguments.get("query"), Some(&Value::from("weather")));

        let call = validate_call(
            "{\"name\": \"search_memory\", \"arguments\": {\"query\": \"weather\", \"limit\": 3}}",
            &specs(),
        )
        .expect("a valid call");
        assert_eq!(call.arguments.get("limit"), Some(&Value::from(3)));
    }
}
