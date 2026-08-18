mod client;
mod config;
mod engine;
mod memory;
mod server;

use anyhow::{bail, Result};
use client::LlmClient;
use config::RlmConfig;
use memory::Memory;
use std::cell::RefCell;
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use std::rc::Rc;

const USAGE: &str = r#"rlm — Recursive Language Model harness (Rust) over a local llama.cpp server

USAGE:
  rlm run  -q "<question>" [-c <file>]...   answer a question, optionally over long context files
  rlm chat [-c <file>]...                   interactive chat with recursive context management
  rlm serve                                 just start the model server and exit

OPTIONS:
  -q, --query <text>      the question (run mode)
  -c, --context <file>    load a file as navigable context (repeatable)
      --model <path>      GGUF model path (overrides config)
      --config <path>     config file (default: rlm-rs\rlm.config.json, created if missing)
      --port <port>       llama-server port (overrides config)
      --iters <n>         max root-loop iterations
      --depth <n>         max recursion depth
      --no-server         fail instead of auto-starting llama-server
"#;

struct Cli {
    command: String,
    query: Option<String>,
    context_files: Vec<PathBuf>,
    model: Option<String>,
    config_path: Option<PathBuf>,
    port: Option<u16>,
    iters: Option<u32>,
    depth: Option<u32>,
    no_server: bool,
}

fn parse_cli() -> Result<Cli> {
    let mut args = std::env::args().skip(1).peekable();
    let mut cli = Cli {
        command: "run".to_string(),
        query: None,
        context_files: vec![],
        model: None,
        config_path: None,
        port: None,
        iters: None,
        depth: None,
        no_server: false,
    };
    if let Some(first) = args.peek() {
        if !first.starts_with('-') {
            cli.command = args.next().unwrap();
        }
    }
    while let Some(arg) = args.next() {
        let mut need = |name: &str| -> Result<String> {
            match args.next() {
                Some(v) => Ok(v),
                None => bail!("missing value for {name}"),
            }
        };
        match arg.as_str() {
            "-q" | "--query" => cli.query = Some(need(&arg)?),
            "-c" | "--context" => cli.context_files.push(PathBuf::from(need(&arg)?)),
            "--model" => cli.model = Some(need(&arg)?),
            "--config" => cli.config_path = Some(PathBuf::from(need(&arg)?)),
            "--port" => cli.port = Some(need(&arg)?.parse()?),
            "--iters" => cli.iters = Some(need(&arg)?.parse()?),
            "--depth" => cli.depth = Some(need(&arg)?.parse()?),
            "--no-server" => cli.no_server = true,
            "-h" | "--help" => {
                println!("{USAGE}");
                std::process::exit(0);
            }
            other => bail!("unknown argument: {other}\n\n{USAGE}"),
        }
    }
    Ok(cli)
}

fn load_contexts(files: &[PathBuf]) -> Result<Vec<(String, String)>> {
    let mut out = vec![];
    for f in files {
        if !f.exists() {
            bail!("context file not found: {}", f.display());
        }
        let name = f
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| f.display().to_string());
        let bytes = std::fs::read(f)?;
        let content = String::from_utf8_lossy(&bytes).to_string();
        eprintln!("[rlm] loaded context \"{name}\" ({} chars)", content.chars().count());
        out.push((name, content));
    }
    Ok(out)
}

fn main() -> Result<()> {
    let cli = parse_cli()?;

    let cfg_path = cli
        .config_path
        .clone()
        .unwrap_or_else(RlmConfig::default_config_path);
    let mut cfg = RlmConfig::load(Path::new(&cfg_path))?;
    if let Some(m) = &cli.model {
        cfg.model_path = m.clone();
    }
    if let Some(p) = cli.port {
        cfg.port = p;
    }
    if let Some(i) = cli.iters {
        cfg.max_iterations = i;
    }
    if let Some(d) = cli.depth {
        cfg.max_depth = d;
    }

    let client = LlmClient::new(&cfg);

    // Leaf sub-calls go to the fast worker model when one is configured and reachable;
    // otherwise everything runs on the main model.
    let make_sub_client = |cfg: &config::RlmConfig, auto_start: bool| -> LlmClient {
        if let Some(port) = cfg.worker_port {
            let worker = LlmClient::for_port(cfg, port);
            let up = if auto_start {
                server::ensure_worker(cfg, &worker).map(|_| true).unwrap_or_else(|e| {
                    eprintln!("[rlm] worker unavailable ({e}); sub-calls use the main model");
                    false
                })
            } else {
                worker.healthy()
            };
            if up {
                return worker;
            }
        }
        LlmClient::new(cfg)
    };

    match cli.command.as_str() {
        "serve" => {
            server::ensure_server(&cfg, &client)?;
            server::ensure_worker(&cfg, &LlmClient::for_port(&cfg, cfg.worker_port.unwrap_or(cfg.port)))?;
            println!("llama-server running at {} (OpenAI-compatible: {}/v1)", cfg.base_url(), cfg.base_url());
            Ok(())
        }
        "run" => {
            let Some(query) = cli.query.clone() else {
                bail!("run mode needs -q \"<question>\"\n\n{USAGE}");
            };
            if !cli.no_server {
                server::ensure_server(&cfg, &client)?;
            } else if !client.healthy() {
                bail!("no llama-server reachable at {} and --no-server was given", cfg.base_url());
            }
            let sub_client = make_sub_client(&cfg, !cli.no_server);
            let contexts = load_contexts(&cli.context_files)?;
            let memory = Rc::new(RefCell::new(Memory::load(&cfg.memory_path)));
            let answer = engine::run_rlm(&client, &sub_client, &cfg, contexts, &query, 0, memory.clone())?;
            memory.borrow_mut().record_session(&query, &answer);
            println!("{answer}");
            Ok(())
        }
        "chat" => {
            if !cli.no_server {
                server::ensure_server(&cfg, &client)?;
            }
            let sub_client = make_sub_client(&cfg, !cli.no_server);
            let base_contexts = load_contexts(&cli.context_files)?;
            let memory = Rc::new(RefCell::new(Memory::load(&cfg.memory_path)));
            let mut conversation = String::new();

            let stdin = std::io::stdin();
            let mut lines = stdin.lock().lines();
            loop {
                print!("you> ");
                std::io::stdout().flush()?;
                let Some(Ok(line)) = lines.next() else { break };
                let query = line.trim().to_string();
                if query.is_empty() {
                    continue;
                }
                if matches!(query.as_str(), "exit" | "quit" | "/exit" | "/quit") {
                    break;
                }
                // The running conversation is itself managed as a navigable context,
                // so long chats never overflow the model's window.
                let mut contexts = base_contexts.clone();
                if !conversation.is_empty() {
                    contexts.push(("conversation_history".to_string(), conversation.clone()));
                }
                match engine::run_rlm(&client, &sub_client, &cfg, contexts, &query, 0, memory.clone()) {
                    Ok(answer) => {
                        println!("rlm> {answer}\n");
                        conversation.push_str(&format!("USER: {query}\nASSISTANT: {answer}\n\n"));
                        memory.borrow_mut().record_session(&query, &answer);
                    }
                    Err(e) => eprintln!("[rlm] error: {e}"),
                }
            }
            Ok(())
        }
        other => bail!("unknown command: {other}\n\n{USAGE}"),
    }
}
