use crate::config::RlmConfig;
use anyhow::{bail, Context, Result};
use serde_json::{json, Value};
use std::time::Duration;

#[derive(Clone)]
pub struct LlmClient {
    agent: ureq::Agent,
    base_url: String,
    pub temperature: f64,
    /// When false, requests ask the chat template to skip chain-of-thought — right for
    /// leaf sub-calls where reasoning burn dwarfs the answer (measured 512 vs 35 tokens).
    pub thinking: bool,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl ChatMessage {
    pub fn new(role: &str, content: impl Into<String>) -> Self {
        ChatMessage { role: role.into(), content: content.into() }
    }
}

impl LlmClient {
    pub fn new(cfg: &RlmConfig) -> Self {
        Self::for_port(cfg, cfg.port)
    }

    /// Client against a specific port on the configured host (e.g. the worker model).
    pub fn for_port(cfg: &RlmConfig, port: u16) -> Self {
        let agent = ureq::AgentBuilder::new()
            .timeout_connect(Duration::from_secs(10))
            // Big models on partial offload can be slow; give generation plenty of room.
            .timeout_read(Duration::from_secs(3600))
            .build();
        LlmClient {
            agent,
            base_url: format!("http://{}:{}", cfg.host, port),
            temperature: cfg.temperature,
            thinking: true,
        }
    }

    pub fn without_thinking(mut self) -> Self {
        self.thinking = false;
        self
    }

    pub fn healthy(&self) -> bool {
        self.agent
            .get(&format!("{}/health", self.base_url))
            .timeout(Duration::from_secs(3))
            .call()
            .is_ok()
    }

    /// OpenAI-compatible chat completion against llama-server.
    pub fn chat(&self, messages: &[ChatMessage], max_tokens: u32) -> Result<String> {
        // Always sent explicitly so the per-request choice overrides whatever
        // server-wide default --chat-template-kwargs establishes.
        let body = json!({
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": self.temperature,
            "chat_template_kwargs": {"enable_thinking": self.thinking},
        });
        let resp: Value = self
            .agent
            .post(&format!("{}/v1/chat/completions", self.base_url))
            .send_json(body)
            .context("llama-server chat request failed")?
            .into_json()
            .context("llama-server returned non-JSON")?;
        let msg = &resp["choices"][0]["message"];
        match msg["content"].as_str() {
            // Reasoning models stream chain-of-thought into `reasoning_content`; if the token
            // budget runs out before any final `content` is emitted, say so instead of
            // silently returning an empty answer.
            Some("") => {
                let finish = resp["choices"][0]["finish_reason"].as_str().unwrap_or("?");
                bail!(
                    "model returned empty content (finish_reason={finish}); the token budget was \
                     likely consumed by reasoning — raise max_tokens/sub_max_tokens in the config"
                )
            }
            Some(s) => Ok(s.to_string()),
            None => bail!("unexpected llama-server response: {}", truncate(&resp.to_string(), 500)),
        }
    }
}

pub fn truncate(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        return s.to_string();
    }
    let cut: String = s.chars().take(max_chars).collect();
    format!("{cut}\n...[truncated {} of {} chars]", max_chars, s.chars().count())
}
