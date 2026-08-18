use crate::client::LlmClient;
use crate::config::RlmConfig;
use anyhow::{bail, Result};
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// Ensure a llama-server is up on the configured port; spawn one if needed.
/// Returns true if this call spawned the server.
pub fn ensure_server(cfg: &RlmConfig, client: &LlmClient) -> Result<bool> {
    if client.healthy() {
        return Ok(false);
    }
    if !Path::new(&cfg.server_bin).exists() {
        bail!("llama-server not found at {}", cfg.server_bin);
    }
    if !Path::new(&cfg.model_path).exists() {
        bail!("model file not found at {}", cfg.model_path);
    }

    let mut cmd = Command::new(&cfg.server_bin);
    cmd.args(["-m", &cfg.model_path])
        .args(["--host", &cfg.host])
        .args(["--port", &cfg.port.to_string()])
        .args(["-c", &cfg.ctx_size.to_string()])
        .args(["--jinja"])
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    if cfg.flash_attn {
        cmd.args(["-fa", "on"]);
    }
    if !cfg.kv_cache_type.is_empty() {
        cmd.args(["--cache-type-k", &cfg.kv_cache_type, "--cache-type-v", &cfg.kv_cache_type]);
    }
    if let Some(ngl) = cfg.n_gpu_layers {
        cmd.args(["-ngl", &ngl.to_string()]);
    }
    cmd.args(&cfg.extra_server_args);
    let child = cmd.spawn()?;
    eprintln!(
        "[rlm] started llama-server (pid {}) with {} — waiting for model load...",
        child.id(),
        cfg.model_path
    );

    let deadline = Instant::now() + Duration::from_secs(cfg.startup_timeout_secs);
    while Instant::now() < deadline {
        if client.healthy() {
            eprintln!("[rlm] llama-server is ready at {}", cfg.base_url());
            return Ok(true);
        }
        std::thread::sleep(Duration::from_secs(2));
    }
    bail!(
        "llama-server did not become healthy within {}s (large models can take a while; \
         raise startup_timeout_secs in the config if needed)",
        cfg.startup_timeout_secs
    )
}
