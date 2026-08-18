use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::PathBuf;

/// Persistent memory shared across sessions: durable key/value facts the model
/// chooses to remember, plus a rolling log of past sessions.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Memory {
    #[serde(default)]
    pub facts: BTreeMap<String, String>,
    #[serde(default)]
    pub sessions: Vec<SessionRecord>,
    #[serde(skip)]
    path: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionRecord {
    pub query: String,
    pub answer: String,
}

impl Memory {
    pub fn load(path: &str) -> Memory {
        let pb = PathBuf::from(path);
        let mut mem = std::fs::read_to_string(&pb)
            .ok()
            .and_then(|t| serde_json::from_str::<Memory>(&t).ok())
            .unwrap_or_default();
        mem.path = pb;
        mem
    }

    pub fn save(&self) {
        if let Ok(text) = serde_json::to_string_pretty(self) {
            let _ = std::fs::write(&self.path, text);
        }
    }

    pub fn remember(&mut self, key: &str, value: &str) {
        self.facts.insert(key.to_string(), value.to_string());
        self.save();
    }

    pub fn recall(&self, key: &str) -> Option<&String> {
        self.facts.get(key)
    }

    pub fn record_session(&mut self, query: &str, answer: &str) {
        self.sessions.push(SessionRecord {
            query: crate::client::truncate(query, 500),
            answer: crate::client::truncate(answer, 1000),
        });
        // Keep the log bounded.
        let len = self.sessions.len();
        if len > 50 {
            self.sessions.drain(0..len - 50);
        }
        self.save();
    }

    /// Short digest injected into the root system prompt.
    pub fn digest(&self, max_chars: usize) -> String {
        let mut out = String::new();
        if !self.facts.is_empty() {
            out.push_str("Persistent memory facts (use recall(key) for values):\n");
            for k in self.facts.keys() {
                out.push_str(&format!("  - {k}\n"));
            }
        }
        if !self.sessions.is_empty() {
            out.push_str("Recent sessions:\n");
            for s in self.sessions.iter().rev().take(5) {
                out.push_str(&format!(
                    "  - Q: {} | A: {}\n",
                    crate::client::truncate(&s.query, 120),
                    crate::client::truncate(&s.answer, 200)
                ));
            }
        }
        crate::client::truncate(&out, max_chars)
    }
}
