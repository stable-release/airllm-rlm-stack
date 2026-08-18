use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Smallest non-projector .gguf directly under `dir` — the sensible default when the
/// config names no model, since quantized files sort below full-precision ones.
fn smallest_gguf(dir: &Path) -> Option<PathBuf> {
    std::fs::read_dir(dir)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.extension().is_some_and(|e| e.eq_ignore_ascii_case("gguf"))
                && !p
                    .file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with("mmproj"))
        })
        .min_by_key(|p| std::fs::metadata(p).map(|m| m.len()).unwrap_or(u64::MAX))
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct RlmConfig {
    /// Path to the GGUF model file llama-server should load.
    pub model_path: String,
    /// Path to llama-server.exe (the llama.cpp runtime bundled in this repo).
    pub server_bin: String,
    pub host: String,
    pub port: u16,
    /// Model context window to request from llama-server.
    pub ctx_size: u32,
    /// GPU layers; null lets llama.cpp auto-fit to available VRAM.
    pub n_gpu_layers: Option<i32>,
    /// KV cache quantization for llama-server (empty = server default, f16).
    pub kv_cache_type: String,
    /// Flash attention (-fa on).
    pub flash_attn: bool,
    /// Extra raw llama-server arguments appended verbatim.
    pub extra_server_args: Vec<String>,
    /// Fast worker model for leaf sub-calls (llm / llm_on single-shot). None routes
    /// everything to the main model. Root-loop reasoning always stays on the main model.
    pub worker_port: Option<u16>,
    /// Worker GGUF; empty auto-discovers the smallest .gguf under models\worker\.
    pub worker_model_path: String,
    pub worker_ctx: u32,
    /// Worker GPU layers. The main model's n_gpu_layers must leave VRAM headroom for
    /// this — two auto-fits overcommit and Windows silently pages, wrecking both.
    pub worker_n_gpu_layers: Option<i32>,
    pub temperature: f64,
    /// Max tokens per root-loop model response.
    pub max_tokens: u32,
    /// Max tokens per recursive sub-call response.
    pub sub_max_tokens: u32,
    /// Max REPL iterations of the root loop before forcing a final answer.
    pub max_iterations: u32,
    /// Max recursion depth for llm_on() sub-calls.
    pub max_depth: u32,
    /// Context slices longer than this (chars) are handled by a nested RLM loop
    /// instead of a single sub-call.
    pub recurse_threshold: usize,
    /// Persistent memory file (facts + session log).
    pub memory_path: String,
    /// Seconds to wait for llama-server to load the model.
    pub startup_timeout_secs: u64,
}

impl Default for RlmConfig {
    fn default() -> Self {
        // Relative paths are resolved against the repo root at load time (see resolve_paths).
        // An empty model_path auto-discovers the smallest .gguf under models\ at load time.
        RlmConfig {
            model_path: String::new(),
            server_bin: r"runtime\llama.cpp\llama-server.exe".into(),
            host: "127.0.0.1".into(),
            port: 8090,
            // 32k: the RLM environment carries long material via peek/grep/recursion, so a
            // huge window is unnecessary and its KV cache would cost GPU layers (speed).
            ctx_size: 32768,
            // Capped (not auto-fit) to leave VRAM headroom for the worker model.
            n_gpu_layers: Some(38),
            kv_cache_type: "q8_0".into(),
            flash_attn: true,
            extra_server_args: Vec::new(),
            worker_port: Some(8092),
            worker_model_path: String::new(),
            worker_ctx: 8192,
            worker_n_gpu_layers: Some(99),
            temperature: 0.7,
            // Generous caps: reasoning models spend chain-of-thought tokens against the
            // completion budget before the final answer appears.
            max_tokens: 4096,
            sub_max_tokens: 2048,
            max_iterations: 12,
            max_depth: 2,
            recurse_threshold: 30_000,
            memory_path: r"rlm-rs\rlm_memory.json".into(),
            startup_timeout_secs: 1800,
        }
    }
}

impl RlmConfig {
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let mut cfg = if path.exists() {
            let text = std::fs::read_to_string(path)?;
            serde_json::from_str(&text)?
        } else {
            let cfg = RlmConfig::default();
            std::fs::write(path, serde_json::to_string_pretty(&cfg)?)?;
            eprintln!("[rlm] wrote default config to {}", path.display());
            cfg
        };
        // Resolve relative paths against the repo root derived from the executable's
        // location — stable regardless of where the config file itself lives.
        let repo_root = crate_dir()
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."));
        cfg.resolve_paths(&repo_root);
        if cfg.model_path.is_empty() {
            if let Some(gguf) = smallest_gguf(&repo_root.join("models")) {
                eprintln!("[rlm] auto-discovered model: {}", gguf.display());
                cfg.model_path = gguf.to_string_lossy().into_owned();
            }
        }
        if cfg.worker_port.is_some() && cfg.worker_model_path.is_empty() {
            match smallest_gguf(&repo_root.join("models").join("worker")) {
                Some(gguf) => {
                    eprintln!("[rlm] auto-discovered worker model: {}", gguf.display());
                    cfg.worker_model_path = gguf.to_string_lossy().into_owned();
                }
                None => cfg.worker_port = None, // no worker model present: single-model mode
            }
        }
        Ok(cfg)
    }

    fn resolve_paths(&mut self, base: &Path) {
        for field in [&mut self.model_path, &mut self.server_bin, &mut self.memory_path,
                      &mut self.worker_model_path] {
            if field.is_empty() {
                continue;
            }
            let p = PathBuf::from(&*field);
            if p.is_relative() {
                *field = base.join(p).to_string_lossy().into_owned();
            }
        }
    }

    pub fn base_url(&self) -> String {
        format!("http://{}:{}", self.host, self.port)
    }

    pub fn default_config_path() -> PathBuf {
        crate_dir().join("rlm.config.json")
    }
}

/// The rlm-rs crate directory, derived from the executable's location
/// (<repo>/rlm-rs/target/release/rlm.exe), so paths work from any working directory.
fn crate_dir() -> PathBuf {
    std::env::current_exe()
        .ok()
        .and_then(|exe| exe.ancestors().nth(3).map(Path::to_path_buf))
        .filter(|dir| dir.join("Cargo.toml").exists())
        .unwrap_or_else(|| PathBuf::from("rlm-rs"))
}
