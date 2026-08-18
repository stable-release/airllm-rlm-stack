use crate::client::LlmClient;
use crate::config::RlmConfig;
use anyhow::{bail, Result};
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

struct ServerSpec<'a> {
    label: &'a str,
    model: &'a str,
    port: u16,
    ctx: u32,
    n_gpu_layers: Option<i32>,
    extra_args: &'a [String],
}

/// Ensure the main llama-server is up on the configured port; spawn it if needed.
/// Returns true if this call spawned the server.
pub fn ensure_server(cfg: &RlmConfig, client: &LlmClient) -> Result<bool> {
    ensure(
        cfg,
        client,
        &ServerSpec {
            label: "main",
            model: &cfg.model_path,
            port: cfg.port,
            ctx: cfg.ctx_size,
            n_gpu_layers: cfg.n_gpu_layers,
            extra_args: &cfg.extra_server_args,
        },
    )
}

/// Ensure the worker llama-server (leaf sub-call model) is up, when configured.
pub fn ensure_worker(cfg: &RlmConfig, worker_client: &LlmClient) -> Result<bool> {
    let Some(port) = cfg.worker_port else { return Ok(false) };
    // A clean model alias so OpenAI-compatible clients can tell the tiers apart.
    let worker_args = ["--alias".to_string(), "local-worker".to_string()];
    ensure(
        cfg,
        worker_client,
        &ServerSpec {
            label: "worker",
            model: &cfg.worker_model_path,
            port,
            ctx: cfg.worker_ctx,
            n_gpu_layers: cfg.worker_n_gpu_layers,
            extra_args: &worker_args,
        },
    )
}

fn ensure(cfg: &RlmConfig, client: &LlmClient, spec: &ServerSpec) -> Result<bool> {
    if client.healthy() {
        return Ok(false);
    }
    if !Path::new(&cfg.server_bin).exists() {
        bail!("llama-server not found at {}", cfg.server_bin);
    }
    if !Path::new(spec.model).exists() {
        bail!("{} model file not found at {}", spec.label, spec.model);
    }

    let mut cmd = Command::new(&cfg.server_bin);
    cmd.args(["-m", spec.model])
        .args(["--host", &cfg.host])
        .args(["--port", &spec.port.to_string()])
        .args(["-c", &spec.ctx.to_string()])
        .args(["--jinja"])
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    if cfg.flash_attn {
        cmd.args(["-fa", "on"]);
    }
    if !cfg.kv_cache_type.is_empty() {
        cmd.args(["--cache-type-k", &cfg.kv_cache_type, "--cache-type-v", &cfg.kv_cache_type]);
    }
    if let Some(ngl) = spec.n_gpu_layers {
        cmd.args(["-ngl", &ngl.to_string()]);
    }
    cmd.args(spec.extra_args);

    let child = cmd.spawn()?;
    eprintln!(
        "[rlm] started {} llama-server (pid {}) with {} — waiting for model load...",
        spec.label,
        child.id(),
        spec.model
    );

    let deadline = Instant::now() + Duration::from_secs(cfg.startup_timeout_secs);
    while Instant::now() < deadline {
        if client.healthy() {
            eprintln!("[rlm] {} server is ready on port {}", spec.label, spec.port);
            return Ok(true);
        }
        std::thread::sleep(Duration::from_secs(2));
    }
    bail!(
        "{} llama-server did not become healthy within {}s (large models can take a while; \
         raise startup_timeout_secs in the config if needed)",
        spec.label,
        cfg.startup_timeout_secs
    )
}
