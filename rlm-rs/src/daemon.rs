//! Loopback-only RLM control plane, ported from the fork's `rlmd` design.
//!
//! Security posture (matching the original's v0 decisions):
//! - Binds 127.0.0.1 only — remote access goes through a tunnel the user sets up.
//! - Optional bearer auth: if the RLM_API_TOKEN env var is set, every /v1/* request
//!   must carry it. Health endpoints never expose run data.
//! - Capabilities default to deny and are intersected with server config (v0: `memory`).
//! - Clients supply context *text inline*; the daemon never reads local files on a
//!   client's behalf (no ambient local-file capability).
//! - Per-run limits can only lower the configured ceilings, never raise them.
//! - Runs execute serially through one permit (a memory-constrained host should not
//!   overlap generations); the queue is bounded and over-admission is rejected.
//! - Cancellation is cooperative at safe iteration boundaries; a dispatched
//!   generation is drained, never interrupted.
//! - The store keeps bounded previews (JSONL), not full documents.

use crate::client::LlmClient;
use crate::config::RlmConfig;
use crate::engine::{self, RunControl, RunEvent, CANCELLED_MSG};
use crate::memory::Memory;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::cell::RefCell;
use std::collections::HashMap;
use std::io::Read;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Condvar, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

const MAX_BODY_BYTES: u64 = 32 * 1024 * 1024;
const MAX_CONTEXTS: usize = 16;
const ANSWER_PREVIEW_CAP: usize = 8000;
const MAX_EVENTS_PER_RUN: usize = 1000;
const MAX_PENDING_RUNS: usize = 8;

// ---------------------------------------------------------------- wire types

#[derive(Deserialize, Default)]
struct Capabilities {
    #[serde(default)]
    memory: bool,
}

#[derive(Deserialize, Default)]
struct Limits {
    max_iterations: Option<u32>,
    max_tokens: Option<u32>,
    max_run_seconds: Option<u64>,
}

#[derive(Deserialize)]
struct RunRequest {
    prompt: String,
    #[serde(default)]
    contexts: Vec<ContextIn>,
    #[serde(default)]
    capabilities: Capabilities,
    #[serde(default)]
    limits: Limits,
}

#[derive(Deserialize)]
struct ContextIn {
    name: String,
    text: String,
}

#[derive(Serialize, Clone)]
struct Snapshot {
    run_id: String,
    status: String, // queued | running | done | failed | cancelled
    created_unix: u64,
    finished_unix: Option<u64>,
    iterations: u32,
    answer: Option<String>,
    error: Option<String>,
}

#[derive(Serialize, Clone)]
struct EventRecord {
    seq: u64,
    unix: u64,
    kind: String,
    data: serde_json::Value,
}

// ------------------------------------------------------------------- store

struct RunEntry {
    snapshot: Mutex<Snapshot>,
    events: Mutex<Vec<EventRecord>>,
    events_cv: Condvar,
    cancel: Arc<AtomicBool>,
}

impl RunEntry {
    fn push_event(&self, kind: &str, data: serde_json::Value) {
        let mut events = self.events.lock().unwrap();
        if events.len() >= MAX_EVENTS_PER_RUN {
            return;
        }
        let seq = events.len() as u64;
        events.push(EventRecord { seq, unix: now_unix(), kind: kind.into(), data });
        self.events_cv.notify_all();
    }

    fn terminal(&self) -> bool {
        matches!(self.snapshot.lock().unwrap().status.as_str(), "done" | "failed" | "cancelled")
    }
}

struct Store {
    runs: Mutex<HashMap<String, Arc<RunEntry>>>,
    id_counter: AtomicU64,
}

fn now_unix() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

impl Store {
    fn new_run_id(&self) -> String {
        let n = self.id_counter.fetch_add(1, Ordering::Relaxed);
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.subsec_nanos())
            .unwrap_or(0);
        format!("run-{}-{:06}-{n}", now_unix(), nanos % 1_000_000)
    }
}

// ------------------------------------------------------------------ daemon

pub fn serve(cfg: RlmConfig, port: u16) -> Result<()> {
    let token = std::env::var("RLM_API_TOKEN").ok().filter(|t| !t.is_empty());
    if token.is_none() {
        eprintln!("[rlmd] RLM_API_TOKEN not set — API is unauthenticated (loopback-only)");
    }

    let store = Arc::new(Store { runs: Mutex::new(HashMap::new()), id_counter: AtomicU64::new(1) });
    let (queue_tx, queue_rx) = mpsc::sync_channel::<String>(MAX_PENDING_RUNS);

    // Single-permit executor: one RLM run at a time.
    {
        let store = store.clone();
        let cfg = cfg.clone();
        std::thread::spawn(move || executor_loop(store, cfg, queue_rx));
    }

    // Loopback only, deliberately not configurable (rlmd v0 posture).
    let server = tiny_http::Server::http(("127.0.0.1", port))
        .map_err(|e| anyhow::anyhow!("bind 127.0.0.1:{port} failed: {e}"))?;
    eprintln!("[rlmd] control plane listening on http://127.0.0.1:{port} (auth: {})",
              if token.is_some() { "bearer token" } else { "none" });
    eprintln!("[rlmd] runs log: {}", if cfg.daemon_runs_path.is_empty() { "(disabled)" } else { &cfg.daemon_runs_path });

    loop {
        let request = match server.recv() {
            Ok(r) => r,
            Err(e) => {
                eprintln!("[rlmd] accept error: {e}");
                continue;
            }
        };
        let store = store.clone();
        let cfg = cfg.clone();
        let token = token.clone();
        let queue_tx = queue_tx.clone();
        std::thread::spawn(move || handle(request, store, cfg, token, queue_tx));
    }
}

fn respond_json(request: tiny_http::Request, code: u32, body: serde_json::Value) {
    let data = body.to_string();
    let response = tiny_http::Response::from_string(data)
        .with_status_code(tiny_http::StatusCode(code as u16))
        .with_header(
            tiny_http::Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap(),
        );
    let _ = request.respond(response);
}

fn handle(
    mut request: tiny_http::Request,
    store: Arc<Store>,
    cfg: RlmConfig,
    token: Option<String>,
    queue_tx: mpsc::SyncSender<String>,
) {
    let url = request.url().to_string();
    let path = url.split('?').next().unwrap_or("").to_string();
    let method = request.method().as_str().to_string();

    // Health endpoints: open, no run data.
    if method == "GET" && path == "/health" {
        return respond_json(request, 200, json!({"status": "ok", "service": "rlmd-rs"}));
    }

    // Everything under /v1 requires the bearer token when one is configured.
    if path.starts_with("/v1/") {
        if let Some(expected) = &token {
            let ok = request
                .headers()
                .iter()
                .find(|h| h.field.equiv("Authorization"))
                .map(|h| h.value.as_str() == format!("Bearer {expected}"))
                .unwrap_or(false);
            if !ok {
                return respond_json(request, 401, json!({"error": "missing or invalid bearer token"}));
            }
        }
    } else if !(method == "GET" && path == "/ready") {
        return respond_json(request, 404, json!({"error": "not found"}));
    }

    if method == "GET" && path == "/ready" {
        let main_ok = LlmClient::new(&cfg).healthy();
        let code = if main_ok { 200 } else { 503 };
        return respond_json(request, code, json!({"model_server": main_ok}));
    }

    match (method.as_str(), path.as_str()) {
        ("POST", "/v1/runs") => create_run(request, store, cfg, queue_tx),
        _ => {
            // /v1/runs/{id}[/events|/cancel]
            let parts: Vec<&str> = path.trim_start_matches('/').split('/').collect();
            match (method.as_str(), parts.as_slice()) {
                ("GET", ["v1", "runs", id]) => {
                    let entry = store.runs.lock().unwrap().get(*id).cloned();
                    match entry {
                        Some(e) => {
                            let snap = e.snapshot.lock().unwrap().clone();
                            respond_json(request, 200, serde_json::to_value(snap).unwrap())
                        }
                        None => respond_json(request, 404, json!({"error": "unknown run"})),
                    }
                }
                ("POST", ["v1", "runs", id, "cancel"]) => {
                    let entry = store.runs.lock().unwrap().get(*id).cloned();
                    match entry {
                        Some(e) => {
                            e.cancel.store(true, Ordering::Relaxed);
                            {
                                let mut snap = e.snapshot.lock().unwrap();
                                if snap.status == "queued" {
                                    snap.status = "cancelled".into();
                                    snap.finished_unix = Some(now_unix());
                                }
                            }
                            e.events_cv.notify_all();
                            let snap = e.snapshot.lock().unwrap().clone();
                            respond_json(request, 200, serde_json::to_value(snap).unwrap())
                        }
                        None => respond_json(request, 404, json!({"error": "unknown run"})),
                    }
                }
                ("GET", ["v1", "runs", id, "events"]) => {
                    let after: i64 = url
                        .split('?')
                        .nth(1)
                        .and_then(|q| {
                            q.split('&').find_map(|kv| kv.strip_prefix("after=")?.parse().ok())
                        })
                        .unwrap_or(-1);
                    let entry = store.runs.lock().unwrap().get(*id).cloned();
                    match entry {
                        Some(e) => stream_events(request, e, after),
                        None => respond_json(request, 404, json!({"error": "unknown run"})),
                    }
                }
                _ => respond_json(request, 404, json!({"error": "not found"})),
            }
        }
    }

    // (create_run and stream_events consume the request themselves)
    fn create_run(
        mut request: tiny_http::Request,
        store: Arc<Store>,
        cfg: RlmConfig,
        queue_tx: mpsc::SyncSender<String>,
    ) {
        if request.body_length().map(|l| l as u64 > MAX_BODY_BYTES).unwrap_or(false) {
            return respond_json(request, 413, json!({"error": "body too large"}));
        }
        let mut body = String::new();
        if request
            .as_reader()
            .take(MAX_BODY_BYTES + 1)
            .read_to_string(&mut body)
            .is_err()
            || body.len() as u64 > MAX_BODY_BYTES
        {
            return respond_json(request, 400, json!({"error": "unreadable or oversized body"}));
        }
        let req: RunRequest = match serde_json::from_str(&body) {
            Ok(r) => r,
            Err(e) => return respond_json(request, 400, json!({"error": format!("bad request: {e}")})),
        };
        if req.prompt.trim().is_empty() {
            return respond_json(request, 400, json!({"error": "prompt is required"}));
        }
        if req.contexts.len() > MAX_CONTEXTS {
            return respond_json(request, 413, json!({"error": "too many contexts"}));
        }
        let total_chars: usize = req.contexts.iter().map(|c| c.text.chars().count()).sum();
        if total_chars > cfg.daemon_max_context_chars {
            return respond_json(request, 413, json!({"error": "contexts exceed configured size limit"}));
        }

        // Per-run limits may only lower the configured ceilings.
        let mut run_cfg = cfg.clone();
        if let Some(v) = req.limits.max_iterations {
            run_cfg.max_iterations = v.min(cfg.max_iterations).max(1);
        }
        if let Some(v) = req.limits.max_tokens {
            run_cfg.max_tokens = v.min(cfg.max_tokens).max(64);
        }
        if let Some(v) = req.limits.max_run_seconds {
            run_cfg.max_run_seconds = if cfg.max_run_seconds == 0 { v } else { v.min(cfg.max_run_seconds) };
        }
        // Capability intersection: request AND server config.
        let allow_memory = req.capabilities.memory && cfg.daemon_allow_memory;

        let run_id = store.new_run_id();
        let entry = Arc::new(RunEntry {
            snapshot: Mutex::new(Snapshot {
                run_id: run_id.clone(),
                status: "queued".into(),
                created_unix: now_unix(),
                finished_unix: None,
                iterations: 0,
                answer: None,
                error: None,
            }),
            events: Mutex::new(Vec::new()),
            events_cv: Condvar::new(),
            cancel: Arc::new(AtomicBool::new(false)),
        });
        entry.push_event("queued", json!({"capabilities": {"memory": allow_memory}}));

        // Execution inputs are held separately from the visible snapshot/event store.
        PENDING.lock().unwrap().put(
            run_id.clone(),
            PendingRun {
                prompt: req.prompt,
                contexts: req.contexts.into_iter().map(|c| (sanitize_name(&c.name), c.text)).collect(),
                run_cfg,
                allow_memory,
            },
        );
        store.runs.lock().unwrap().insert(run_id.clone(), entry.clone());

        match queue_tx.try_send(run_id.clone()) {
            Ok(()) => {
                let snap = entry.snapshot.lock().unwrap().clone();
                respond_json(request, 202, serde_json::to_value(snap).unwrap())
            }
            Err(_) => {
                store.runs.lock().unwrap().remove(&run_id);
                PENDING.lock().unwrap().take_run(&run_id);
                respond_json(request, 429, json!({"error": "run queue is full"}))
            }
        }
    }

    fn stream_events(request: tiny_http::Request, entry: Arc<RunEntry>, after: i64) {
        let (tx, rx) = mpsc::sync_channel::<Vec<u8>>(64);
        {
            let entry = entry.clone();
            std::thread::spawn(move || {
                let mut cursor = after;
                loop {
                    let batch: Vec<EventRecord> = {
                        let mut events = entry.events.lock().unwrap();
                        loop {
                            let fresh: Vec<EventRecord> = events
                                .iter()
                                .filter(|e| (e.seq as i64) > cursor)
                                .cloned()
                                .collect();
                            if !fresh.is_empty() || entry.terminal() {
                                break fresh;
                            }
                            events = entry.events_cv.wait(events).unwrap();
                        }
                    };
                    for ev in &batch {
                        cursor = ev.seq as i64;
                        let line = format!(
                            "id: {}\nevent: {}\ndata: {}\n\n",
                            ev.seq,
                            ev.kind,
                            serde_json::to_string(ev).unwrap_or_default()
                        );
                        if tx.send(line.into_bytes()).is_err() {
                            return; // client went away
                        }
                    }
                    if batch.is_empty() && entry.terminal() {
                        return; // sender drops -> EOF closes the stream
                    }
                    if entry.terminal() {
                        // Drain any events emitted after the terminal check next loop;
                        // when none remain the branch above ends the stream.
                        continue;
                    }
                }
            });
        }
        let reader = ChannelReader { rx, buf: Vec::new(), pos: 0 };
        let response = tiny_http::Response::new(
            tiny_http::StatusCode(200),
            vec![
                tiny_http::Header::from_bytes(&b"Content-Type"[..], &b"text/event-stream"[..]).unwrap(),
                tiny_http::Header::from_bytes(&b"Cache-Control"[..], &b"no-store"[..]).unwrap(),
            ],
            reader,
            None,
            None,
        );
        let _ = request.respond(response);
    }
}

fn sanitize_name(name: &str) -> String {
    let cleaned: String = name
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.'))
        .take(64)
        .collect();
    if cleaned.is_empty() { "context".to_string() } else { cleaned }
}

struct PendingRun {
    prompt: String,
    contexts: Vec<(String, String)>,
    run_cfg: RlmConfig,
    allow_memory: bool,
}

static PENDING: Mutex<Option<HashMap<String, PendingRun>>> = Mutex::new(None);

trait PendingExt {
    fn put(&mut self, id: String, run: PendingRun);
    fn take_run(&mut self, id: &str) -> Option<PendingRun>;
}

impl PendingExt for Option<HashMap<String, PendingRun>> {
    fn put(&mut self, id: String, run: PendingRun) {
        self.get_or_insert_with(HashMap::new).insert(id, run);
    }
    fn take_run(&mut self, id: &str) -> Option<PendingRun> {
        self.as_mut()?.remove(id)
    }
}

// ---------------------------------------------------------------- executor

fn executor_loop(store: Arc<Store>, cfg: RlmConfig, queue_rx: mpsc::Receiver<String>) {
    for run_id in queue_rx.iter() {
        let Some(entry) = store.runs.lock().unwrap().get(&run_id).cloned() else { continue };
        let Some(pending) = PENDING.lock().unwrap().take_run(&run_id) else { continue };

        if entry.cancel.load(Ordering::Relaxed) {
            // Cancelled while queued; snapshot already terminal — still record it.
            persist_snapshot(&pending.run_cfg, &entry.snapshot.lock().unwrap().clone());
            continue;
        }
        {
            let mut snap = entry.snapshot.lock().unwrap();
            snap.status = "running".into();
        }
        entry.push_event("running", json!({}));

        let outcome = execute_run(&pending, &entry);

        let mut snap = entry.snapshot.lock().unwrap();
        snap.finished_unix = Some(now_unix());
        match outcome {
            Ok(answer) => {
                snap.status = "done".into();
                snap.answer = Some(crate::client::truncate(&answer, ANSWER_PREVIEW_CAP));
            }
            Err(e) if e.to_string().contains(CANCELLED_MSG) => {
                snap.status = "cancelled".into();
            }
            Err(e) => {
                snap.status = "failed".into();
                snap.error = Some(crate::client::truncate(&e.to_string(), 2000));
            }
        }
        let final_snap = snap.clone();
        drop(snap);
        entry.push_event(&final_snap.status.clone(), serde_json::to_value(&final_snap).unwrap_or_default());
        persist_snapshot(&pending.run_cfg, &final_snap);
    }
}

fn execute_run(pending: &PendingRun, entry: &Arc<RunEntry>) -> Result<String> {
    let cfg = &pending.run_cfg;
    let client = if cfg.root_thinking {
        LlmClient::new(cfg)
    } else {
        LlmClient::new(cfg).without_thinking()
    };
    if !client.healthy() {
        crate::server::ensure_server(cfg, &client)?;
    }
    let sub_client = match cfg.worker_port {
        Some(port) => {
            let worker = LlmClient::for_port(cfg, port);
            let up = worker.healthy() || crate::server::ensure_worker(cfg, &worker).is_ok();
            if up {
                if cfg.worker_thinking { worker } else { worker.without_thinking() }
            } else {
                client.clone()
            }
        }
        None => client.clone(),
    };

    let (ev_tx, ev_rx) = mpsc::channel::<RunEvent>();
    {
        let entry = entry.clone();
        std::thread::spawn(move || {
            for ev in ev_rx.iter() {
                let RunEvent::Iteration { n, of, repl_preview } = ev;
                {
                    let mut snap = entry.snapshot.lock().unwrap();
                    snap.iterations = n;
                }
                entry.push_event("iteration", json!({"n": n, "of": of, "repl_preview": repl_preview}));
            }
        });
    }

    let control = RunControl {
        cancel: entry.cancel.clone(),
        events: Some(ev_tx),
        allow_memory: pending.allow_memory,
    };
    let memory = Rc::new(RefCell::new(Memory::load(&cfg.memory_path)));
    let answer = engine::run_rlm(
        &client,
        &sub_client,
        cfg,
        pending.contexts.clone(),
        &pending.prompt,
        0,
        memory.clone(),
        &control,
    )?;
    if pending.allow_memory {
        memory.borrow_mut().record_session(&pending.prompt, &answer);
    }
    Ok(answer)
}

fn persist_snapshot(cfg: &RlmConfig, snap: &Snapshot) {
    if cfg.daemon_runs_path.is_empty() {
        return;
    }
    use std::io::Write;
    let line = match serde_json::to_string(snap) {
        Ok(l) => l,
        Err(e) => return eprintln!("[rlmd] snapshot serialize failed: {e}"),
    };
    match std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&cfg.daemon_runs_path)
    {
        Ok(mut f) => {
            if let Err(e) = writeln!(f, "{line}") {
                eprintln!("[rlmd] runs log write failed ({}): {e}", cfg.daemon_runs_path);
            }
        }
        Err(e) => eprintln!("[rlmd] runs log open failed ({}): {e}", cfg.daemon_runs_path),
    }
}

// ------------------------------------------------------------- SSE plumbing

struct ChannelReader {
    rx: mpsc::Receiver<Vec<u8>>,
    buf: Vec<u8>,
    pos: usize,
}

impl Read for ChannelReader {
    fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
        if self.pos >= self.buf.len() {
            match self.rx.recv() {
                Ok(chunk) => {
                    self.buf = chunk;
                    self.pos = 0;
                }
                Err(_) => return Ok(0), // producer done -> EOF
            }
        }
        let n = (self.buf.len() - self.pos).min(out.len());
        out[..n].copy_from_slice(&self.buf[self.pos..self.pos + n]);
        self.pos += n;
        Ok(n)
    }
}
