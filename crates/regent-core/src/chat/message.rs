//! One stored chat turn.

use serde::{Deserialize, Serialize};

/// Who a chat message came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    /// The person using Regent.
    User,
    /// Regent's reply.
    Assistant,
}

impl Role {
    /// The storage string for this role.
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Role::User => "user",
            Role::Assistant => "assistant",
        }
    }

    /// Parses the storage string, returning `None` for unknown values.
    pub(crate) fn from_storage(text: &str) -> Option<Role> {
        match text {
            "user" => Some(Role::User),
            "assistant" => Some(Role::Assistant),
            _ => None,
        }
    }
}

/// One message as it is stored in the chat history.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatMessage {
    /// The database row id, in append order.
    pub id: i64,
    /// Who wrote it.
    pub role: Role,
    /// The message text.
    pub text: String,
    /// Whether the message was spoken rather than typed.
    pub from_voice: bool,
    /// When it was stored, in unix seconds.
    pub created_at: i64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn role_round_trips_through_storage_strings() {
        for role in [Role::User, Role::Assistant] {
            assert_eq!(Role::from_storage(role.as_str()), Some(role));
        }
        assert_eq!(Role::from_storage("system"), None);
    }

    #[derive(Serialize, Deserialize)]
    struct Wrapper {
        role: Role,
    }

    #[test]
    fn role_serialises_snake_case() {
        let text = toml::to_string(&Wrapper {
            role: Role::Assistant,
        })
        .expect("serialise");
        assert!(text.contains("role = \"assistant\""), "{text}");
        let wrapper: Wrapper = toml::from_str("role = \"user\"").expect("deserialise");
        assert_eq!(wrapper.role, Role::User);
    }
}
