//! The RLM root loop: the model is given a script environment (Rhai) where the long
//! context lives as named variables it can peek/grep/chunk, and where it can issue
//! recursive sub-calls (`llm`, `llm_on`) over slices of that context — the Recursive
//! Language Model inference pattern, ported from the Python `rlms` package.

use crate::client::{truncate, ChatMessage, LlmClient};
use crate::config::RlmConfig;
use crate::memory::Memory;
use anyhow::Result;
use regex::Regex;
use rhai::{Dynamic, Engine, ImmutableString};
use std::cell::RefCell;
use std::rc::Rc;

const PEEK_CAP: usize = 16_000;
const GREP_MAX_MATCHES: usize = 80;
const GREP_LINE_CAP: usize = 240;
const SINGLE_SHOT_SLICE_CAP: usize = 80_000;
const REPL_OUTPUT_CAP: usize = 12_000;

pub struct EvalState {
    pub contexts: Vec<(String, String)>,
    pub output: String,
    pub final_answer: Option<String>,
}

fn context_slice(contexts: &[(String, String)], name: &str, start: i64, len: i64) -> Result<String, String> {
    let Some((_, content)) = contexts.iter().find(|(n, _)| n == name) else {
        let names: Vec<&str> = contexts.iter().map(|(n, _)| n.as_str()).collect();
        return Err(format!("[error] no context named '{name}'. Available: {names:?}"));
    };
    let start = start.max(0) as usize;
    let len = len.max(0) as usize;
    Ok(content.chars().skip(start).take(len).collect())
}

pub fn run_rlm(
    client: &LlmClient,
    cfg: &RlmConfig,
    contexts: Vec<(String, String)>,
    query: &str,
    depth: u32,
    memory: Rc<RefCell<Memory>>,
) -> Result<String> {
    let state = Rc::new(RefCell::new(EvalState {
        contexts,
        output: String::new(),
        final_answer: None,
    }));

    let engine = build_engine(client.clone(), cfg.clone(), depth, state.clone(), memory.clone());
    let system = system_prompt(cfg, &state.borrow(), &memory.borrow(), depth);

    let mut messages = vec![
        ChatMessage::new("system", system),
        ChatMessage::new("user", query.to_string()),
    ];

    let code_re = Regex::new(r"(?s)```(?:rhai|rust|js|javascript)?[ \t]*\r?\n(.*?)```").unwrap();

    for iter in 0..cfg.max_iterations {
        let response = client.chat(&messages, cfg.max_tokens)?;
        messages.push(ChatMessage::new("assistant", response.clone()));

        if let Some(answer) = state.borrow().final_answer.clone() {
            return Ok(answer);
        }

        let Some(cap) = code_re.captures(&response) else {
            // No code block: the model is answering directly.
            return Ok(response.trim().to_string());
        };
        let code = cap[1].to_string();

        state.borrow_mut().output.clear();
        let eval_result = engine.eval::<Dynamic>(&code);

        let mut repl_out = state.borrow().output.clone();
        match eval_result {
            Ok(v) if !v.is_unit() => repl_out.push_str(&format!("=> {v}\n")),
            Ok(_) => {}
            Err(e) => repl_out.push_str(&format!("[script error] {e}\n")),
        }
        if let Some(answer) = state.borrow().final_answer.clone() {
            return Ok(answer);
        }
        if repl_out.trim().is_empty() {
            repl_out.push_str("(no output — use print(...) to inspect values)\n");
        }
        if depth == 0 {
            eprintln!("[rlm] iteration {}/{} — REPL output {} chars", iter + 1, cfg.max_iterations, repl_out.len());
        }
        messages.push(ChatMessage::new(
            "user",
            format!("[REPL output]\n{}", truncate(&repl_out, REPL_OUTPUT_CAP)),
        ));
    }

    messages.push(ChatMessage::new(
        "user",
        "Iteration limit reached. Reply now with your final answer as plain text (no code block).",
    ));
    let final_resp = client.chat(&messages, cfg.max_tokens)?;
    Ok(final_resp.trim().to_string())
}

fn build_engine(
    client: LlmClient,
    cfg: RlmConfig,
    depth: u32,
    state: Rc<RefCell<EvalState>>,
    memory: Rc<RefCell<Memory>>,
) -> Engine {
    let mut engine = Engine::new();
    engine.set_max_operations(2_000_000);
    engine.set_max_expr_depths(64, 64);

    // ---- printing --------------------------------------------------------
    {
        let st = state.clone();
        engine.on_print(move |s| {
            let mut st = st.borrow_mut();
            st.output.push_str(s);
            st.output.push('\n');
        });
    }
    {
        let st = state.clone();
        engine.on_debug(move |s, _, _| {
            let mut st = st.borrow_mut();
            st.output.push_str(s);
            st.output.push('\n');
        });
    }

    // ---- context inspection ---------------------------------------------
    {
        let st = state.clone();
        engine.register_fn("ctx_list", move || -> String {
            st.borrow()
                .contexts
                .iter()
                .map(|(n, c)| format!("{n} ({} chars)", c.chars().count()))
                .collect::<Vec<_>>()
                .join("\n")
        });
    }
    {
        let st = state.clone();
        engine.register_fn("ctx_len", move |name: ImmutableString| -> i64 {
            st.borrow()
                .contexts
                .iter()
                .find(|(n, _)| n.as_str() == name.as_str())
                .map(|(_, c)| c.chars().count() as i64)
                .unwrap_or(-1)
        });
    }
    {
        let st = state.clone();
        engine.register_fn(
            "peek",
            move |name: ImmutableString, start: i64, len: i64| -> String {
                let len = len.min(PEEK_CAP as i64);
                match context_slice(&st.borrow().contexts, &name, start, len) {
                    Ok(s) => s,
                    Err(e) => e,
                }
            },
        );
    }
    {
        let st = state.clone();
        engine.register_fn(
            "grep",
            move |name: ImmutableString, pattern: ImmutableString| -> String {
                let re = match Regex::new(&pattern) {
                    Ok(r) => r,
                    Err(e) => return format!("[error] bad regex: {e}"),
                };
                let st = st.borrow();
                let Some((_, content)) = st.contexts.iter().find(|(n, _)| n.as_str() == name.as_str()) else {
                    return format!("[error] no context named '{name}'");
                };
                let mut out = String::new();
                let mut offset = 0usize;
                let mut hits = 0usize;
                for line in content.lines() {
                    if re.is_match(line) {
                        hits += 1;
                        if hits <= GREP_MAX_MATCHES {
                            out.push_str(&format!("@{offset}: {}\n", truncate(line.trim(), GREP_LINE_CAP)));
                        }
                    }
                    offset += line.chars().count() + 1;
                }
                if hits == 0 {
                    "(no matches)".to_string()
                } else {
                    if hits > GREP_MAX_MATCHES {
                        out.push_str(&format!("...({} more matches not shown)\n", hits - GREP_MAX_MATCHES));
                    }
                    out
                }
            },
        );
    }

    // ---- recursive sub-calls --------------------------------------------
    {
        let cl = client.clone();
        let cf = cfg.clone();
        engine.register_fn("llm", move |prompt: ImmutableString| -> String {
            let messages = [
                ChatMessage::new("system", "You are a capable assistant. Answer concisely and directly."),
                ChatMessage::new("user", prompt.to_string()),
            ];
            match cl.chat(&messages, cf.sub_max_tokens) {
                Ok(s) => s,
                Err(e) => format!("[error] sub-call failed: {e}"),
            }
        });
    }
    {
        let cl = client.clone();
        let cf = cfg.clone();
        let st = state.clone();
        let mem = memory.clone();
        engine.register_fn(
            "llm_on",
            move |prompt: ImmutableString, name: ImmutableString, start: i64, len: i64| -> String {
                let slice = match context_slice(&st.borrow().contexts, &name, start, len) {
                    Ok(s) => s,
                    Err(e) => return e,
                };
                if slice.is_empty() {
                    return "[error] empty slice".to_string();
                }
                // Large slices get a nested RLM loop of their own (true recursion);
                // small ones are answered with a single grounded sub-call.
                if depth + 1 < cf.max_depth && slice.chars().count() > cf.recurse_threshold {
                    let sub_ctx = vec![("excerpt".to_string(), slice)];
                    match run_rlm(&cl, &cf, sub_ctx, &prompt, depth + 1, mem.clone()) {
                        Ok(s) => s,
                        Err(e) => format!("[error] recursive call failed: {e}"),
                    }
                } else {
                    let slice = truncate(&slice, SINGLE_SHOT_SLICE_CAP);
                    let messages = [
                        ChatMessage::new(
                            "system",
                            format!(
                                "Answer the user's question using ONLY the following excerpt \
                                 from '{name}'. If the answer is not in the excerpt, say so.\n\
                                 --- EXCERPT ---\n{slice}\n--- END EXCERPT ---"
                            ),
                        ),
                        ChatMessage::new("user", prompt.to_string()),
                    ];
                    match cl.chat(&messages, cf.sub_max_tokens) {
                        Ok(s) => s,
                        Err(e) => format!("[error] sub-call failed: {e}"),
                    }
                }
            },
        );
    }

    // ---- persistent memory ----------------------------------------------
    {
        let mem = memory.clone();
        engine.register_fn("remember", move |key: ImmutableString, value: ImmutableString| {
            mem.borrow_mut().remember(&key, &value);
        });
    }
    {
        let mem = memory.clone();
        engine.register_fn("recall", move |key: ImmutableString| -> String {
            mem.borrow()
                .recall(&key)
                .cloned()
                .unwrap_or_else(|| format!("[error] no memory for key '{key}'"))
        });
    }
    {
        let mem = memory;
        engine.register_fn("memory_keys", move || -> String {
            let mem = mem.borrow();
            if mem.facts.is_empty() {
                "(memory is empty)".to_string()
            } else {
                mem.facts.keys().cloned().collect::<Vec<_>>().join("\n")
            }
        });
    }

    // ---- finishing -------------------------------------------------------
    {
        let st = state;
        engine.register_fn("finish", move |answer: ImmutableString| {
            st.borrow_mut().final_answer = Some(answer.to_string());
        });
    }

    engine
}

fn system_prompt(cfg: &RlmConfig, state: &EvalState, memory: &Memory, depth: u32) -> String {
    let ctx_lines = if state.contexts.is_empty() {
        "  (none loaded — you can still use llm(), memory, and your own knowledge)".to_string()
    } else {
        state
            .contexts
            .iter()
            .map(|(n, c)| format!("  - \"{n}\": {} chars", c.chars().count()))
            .collect::<Vec<_>>()
            .join("\n")
    };
    let mem_digest = memory.digest(2000);
    let role = if depth == 0 { "root" } else { "recursive sub" };

    format!(
        r#"You are a Recursive Language Model ({role} call, depth {depth}). Long context is NOT pasted into your prompt; it lives in an environment you interact with by writing code.

Loaded contexts:
{ctx_lines}

{mem_digest}
Each of your messages must contain exactly ONE fenced code block (```rhai ... ```) with a short script. After it runs you receive the printed output, then you write the next script. Available functions:
  ctx_list()                        -> list of context names and sizes
  ctx_len(name)                     -> length of a context in chars
  peek(name, start, len)            -> read a slice of a context (max {peek_cap} chars)
  grep(name, regex)                 -> matching lines with their char offsets
  llm(prompt)                       -> ask a fresh sub-model (no context attached)
  llm_on(prompt, name, start, len)  -> ask a sub-model grounded on a context slice; large slices recurse
  remember(key, value) / recall(key) / memory_keys()  -> persistent memory across sessions
  print(x)                          -> show a value in the REPL output
  finish(answer)                    -> submit your final answer and stop

Rhai syntax notes: `let x = 3;`, string concat with `+`, `if`/`for`/`while` as in Rust, no Python syntax.

Strategy: inspect sizes first, grep/peek to locate relevant regions, delegate large regions to llm_on, combine results, then call finish() with a complete, well-written answer. You have at most {iters} script turns. Never fabricate content you did not observe."#,
        peek_cap = PEEK_CAP,
        iters = cfg.max_iterations,
    )
}
