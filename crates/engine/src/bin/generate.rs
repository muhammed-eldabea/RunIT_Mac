/// `generate` — end-to-end inference benchmark + interactive generation.
///
/// Usage:
///   cargo run --release -p bare-metal-engine --bin generate -- <model.gguf> [options]
///
/// Options:
///   --tokenizer <path>   Path to tokenizer.json (enables text I/O)
///   --prompt <text>      Input prompt (default: BOS token)
///   --tokens N           Tokens to generate (default: 30)
///   --warmup N           Warmup steps excluded from aggregate stats (default: 1)
///   --temperature T      Sampling temperature 0.0=greedy (default: 0.0)
///   --top-p P            Nucleus sampling probability (default: 0.9)
///   --top-k K            Top-K sampling (default: 50)
///   --seed S             RNG seed (default: 42)
///   --tq                 Use TurboQuant KV cache (Phase 5)
///   --tq-key-bits N      TQ key bits: 3 or 4 (default: 3)
///   --tq-val-bits N      TQ value bits: 2 or 4 (default: 4)
///
/// Prints per-step stats and a final timing summary.
/// Exit code: 0 = success, non-zero = error.

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("generate requires macOS (Metal GPU)");
    std::process::exit(1);
}

#[cfg(target_os = "macos")]
fn main() -> anyhow::Result<()> {
    use bare_metal_engine::{
        chat_template::SpecialTokens,
        forward::{Executor, ForwardError},
        kv_cache::KvCache,
        loader::{load_model_opts, LoadOptions},
        sampler::{SamplerConfig, SimpleRng},
        tq_kv_cache::TqKvCache,
    };
    use bare_metal_kernels::context::MetalContext;
    use std::path::Path;
    use std::time::{Duration, Instant};

    // ── Arg parsing ───────────────────────────────────────────────────────────
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 || args[1] == "--help" || args[1] == "-h" {
        eprintln!(
            "Usage: generate <model.gguf> [--tokenizer <path>] [--prompt <text>]\n\
             \x20                           [--tokens N] [--warmup N]\n\
             \x20                           [--temperature T] [--top-p P] [--top-k K] [--seed S]\n\
             \x20                           [--tq] [--tq-key-bits N] [--tq-val-bits N]"
        );
        std::process::exit(if args.len() < 2 { 1 } else { 0 });
    }
    let model_path = Path::new(&args[1]);

    let mut num_tokens:      usize         = 30;
    let mut warmup_steps:    usize         = 1;
    let mut use_tq                          = false;
    let mut tq_key_bits:     u32           = 3;
    let mut tq_val_bits:     u32           = 4;
    let mut tokenizer_path:  Option<String> = None;
    let mut prompt_text:     Option<String> = None;
    let mut temperature:     f32           = 0.0;
    let mut top_p:           f32           = 0.9;
    let mut top_k:           usize         = 50;
    let mut seed:            u64           = 42;
    let mut rep_penalty:     f32           = 1.1;
    let mut json_summary:    Option<String> = None; // path to write JSON metrics

    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--tokens"       if i + 1 < args.len() => { num_tokens    = args[i+1].parse()?; i += 2; }
            "--warmup"       if i + 1 < args.len() => { warmup_steps  = args[i+1].parse()?; i += 2; }
            "--tq"                                  => { use_tq = true; i += 1; }
            "--tq-key-bits"  if i + 1 < args.len() => { tq_key_bits   = args[i+1].parse()?; i += 2; }
            "--tq-val-bits"  if i + 1 < args.len() => { tq_val_bits   = args[i+1].parse()?; i += 2; }
            "--tokenizer"    if i + 1 < args.len() => { tokenizer_path = Some(args[i+1].clone()); i += 2; }
            "--prompt"       if i + 1 < args.len() => { prompt_text   = Some(args[i+1].clone()); i += 2; }
            "--temperature"  if i + 1 < args.len() => { temperature   = args[i+1].parse()?; i += 2; }
            "--top-p"        if i + 1 < args.len() => { top_p         = args[i+1].parse()?; i += 2; }
            "--top-k"        if i + 1 < args.len() => { top_k         = args[i+1].parse()?; i += 2; }
            "--seed"         if i + 1 < args.len() => { seed          = args[i+1].parse()?; i += 2; }
            "--rep-penalty"  if i + 1 < args.len() => { rep_penalty   = args[i+1].parse()?; i += 2; }
            "--json-summary" if i + 1 < args.len() => { json_summary  = Some(args[i+1].clone()); i += 2; }
            other => { eprintln!("Unknown argument: {other}"); i += 1; }
        }
    }

    // ── Load tokenizer (optional) ─────────────────────────────────────────────
    let tokenizer = tokenizer_path.as_deref().map(|p| {
        bare_metal_tokenizer::Tokenizer::from_file(p)
            .expect("failed to load tokenizer")
    });

    // ── Load model ────────────────────────────────────────────────────────────
    let t_load = Instant::now();
    println!("═══════════════════════════════════════════════════════");
    println!(" Bare-Metal MX LLM — Inference");
    println!("═══════════════════════════════════════════════════════");
    println!("Model : {}", model_path.display());

    // Skip shape validation: quantised weights often have transposed or
    // packed shapes that don't match the expected [out, in] convention.
    // Real missing-weight errors surface later as ForwardError::MissingWeight.
    let model = load_model_opts(model_path, LoadOptions { validate_weights: false })?;
    let cfg   = &model.config;
    let load_ms = t_load.elapsed().as_millis();

    println!(
        "Config: {} layers  hidden={}  heads={}/kv={}  vocab={}  head_dim={}",
        cfg.num_hidden_layers, cfg.hidden_size,
        cfg.num_attention_heads, cfg.num_key_value_heads,
        cfg.vocab_size, cfg.head_dim,
    );
    println!("Load  : {load_ms} ms");

    // ── Metal context + executor ──────────────────────────────────────────────
    let ctx = MetalContext::new().map_err(|e| anyhow::anyhow!("Metal: {e}"))?;
    println!("GPU   : {}", ctx.device.name());

    let t_upload = Instant::now();
    let executor = match Executor::new(ctx, &model) {
        Ok(e) => e,
        Err(ForwardError::UnsupportedDtype { tensor, dtype }) => {
            eprintln!("Error: tensor '{tensor}' has unsupported dtype {dtype:?}");
            std::process::exit(1);
        }
        Err(e) => return Err(e.into()),
    };
    let upload_ms = t_upload.elapsed().as_millis() as u64; // capture now, before generation
    println!("Upload: {} ms", upload_ms);

    // ── Encode prompt ─────────────────────────────────────────────────────────
    let (prompt_ids, bos) = if let (Some(tok), Some(text)) = (&tokenizer, &prompt_text) {
        let mut ids = tok.encode(text, false)?;
        let bos = cfg.bos_token_id.unwrap_or(1);
        if ids.is_empty() { ids.push(bos); }
        println!("Prompt: {:?} ({} tokens)", text, ids.len());
        let first_token = ids[0];
        (ids, first_token)
    } else {
        let bos = cfg.bos_token_id.unwrap_or(1);
        (vec![bos], bos)
    };

    // ── KV cache ──────────────────────────────────────────────────────────────
    let max_seq = prompt_ids.len() + num_tokens + 8;
    let (kv_mb, cache_desc) = if use_tq {
        let kv = TqKvCache::new(&executor.ctx, cfg, tq_key_bits, tq_val_bits, 128);
        let mb = kv.gpu_memory_bytes() as f64 / 1_048_576.0;
        drop(kv);
        (mb, format!("TurboQuant (key={tq_key_bits}b val={tq_val_bits}b, buf=128)"))
    } else {
        let kv = KvCache::new(&executor.ctx, cfg, max_seq);
        let mb = kv.memory_bytes() as f64 / 1_048_576.0;
        drop(kv);
        (mb, "F16 flat".to_string())
    };
    println!("KV    : {cache_desc}  {kv_mb:.1} MB  (max_seq={max_seq})");

    // ── Sampler config ────────────────────────────────────────────────────────
    let sampler_cfg = SamplerConfig { temperature, top_k, top_p, seed, repetition_penalty: rep_penalty };

    // ── Generation ────────────────────────────────────────────────────────────
    println!("\nGenerating {num_tokens} tokens (warmup={warmup_steps})…");
    println!("─────────────────────────────────────────────────────────");
    println!(" step  | token  | logit range       | ms/tok");
    println!("─────────────────────────────────────────────────────────");

    let mut step_times: Vec<Duration> = Vec::with_capacity(num_tokens);
    let mut generated_ids: Vec<u32>   = Vec::with_capacity(num_tokens);

    let special = SpecialTokens::default();
    let mut rng  = SimpleRng::new(seed);

    if use_tq {
        let mut kv = TqKvCache::new(&executor.ctx, cfg, tq_key_bits, tq_val_bits, 128);
        run_loop_tq(&executor, &mut kv, &prompt_ids, num_tokens, warmup_steps,
                    &sampler_cfg, &special, &mut rng,
                    &mut step_times, &mut generated_ids)?;
    } else {
        let mut kv = KvCache::new(&executor.ctx, cfg, max_seq);
        run_loop_std(&executor, &mut kv, &prompt_ids, num_tokens, warmup_steps,
                     &sampler_cfg, &special, &mut rng,
                     &mut step_times, &mut generated_ids)?;
    }

    // ── Print generated text ──────────────────────────────────────────────────
    if let Some(tok) = &tokenizer {
        println!("─────────────────────────────────────────────────────────");
        let text = tok.decode(&generated_ids, true).unwrap_or_default();
        println!("\nGenerated text:\n{text}\n");
    }

    // ── Quality metrics ───────────────────────────────────────────────────────
    let eos_id     = cfg.eos_token_id.unwrap_or(u32::MAX);
    let eos_pos    = generated_ids.iter().position(|&t| t == eos_id);
    let gen_len    = generated_ids.len();
    let unique_tok = generated_ids.iter().collect::<std::collections::HashSet<_>>().len();
    let unique_ratio = if gen_len > 0 { unique_tok as f64 / gen_len as f64 } else { 0.0 };
    // Repetition: fraction of tokens that repeat the immediately preceding token
    let immediate_repeats = generated_ids.windows(2).filter(|w| w[0] == w[1]).count();
    let repeat_rate = if gen_len > 1 { immediate_repeats as f64 / (gen_len - 1) as f64 } else { 0.0 };

    println!("─────────────────────────────────────────────────────────");
    println!("Quality: {} tokens  EOS={} (pos {:?})  unique={:.0}%  repeat={:.1}%",
        gen_len,
        if eos_pos.is_some() { "hit" } else { "not hit" },
        eos_pos,
        unique_ratio * 100.0,
        repeat_rate * 100.0,
    );

    // ── Timing summary ────────────────────────────────────────────────────────
    println!("─────────────────────────────────────────────────────────");
    print_summary(
        &step_times, warmup_steps, num_tokens,
        prompt_ids.len(), load_ms as u64, upload_ms,
        kv_mb, use_tq,
        eos_pos, unique_ratio, repeat_rate,
        json_summary.as_deref(),
    );

    Ok(())
}

// ── Standard KV-cache generation loop ────────────────────────────────────────

#[cfg(target_os = "macos")]
fn run_loop_std(
    executor:     &bare_metal_engine::forward::Executor,
    kv:           &mut bare_metal_engine::kv_cache::KvCache,
    prompt_ids:   &[u32],
    num_tokens:   usize,
    warmup_steps: usize,
    sampler_cfg:  &bare_metal_engine::sampler::SamplerConfig,
    special:      &bare_metal_engine::chat_template::SpecialTokens,
    rng:          &mut bare_metal_engine::sampler::SimpleRng,
    step_times:   &mut Vec<std::time::Duration>,
    generated:    &mut Vec<u32>,
) -> anyhow::Result<()> {
    use bare_metal_engine::sampler::sample;

    // Process prompt token-by-token (no prefill — f32 decode path for precision)
    let (start_pos, mut next_token) = if prompt_ids.len() > 1 {
        let t0 = std::time::Instant::now();
        for (i, &tid) in prompt_ids[..prompt_ids.len()-1].iter().enumerate() {
            executor.forward_greedy(tid, i as u32, kv)?;
        }
        let last = *prompt_ids.last().unwrap();
        let pos = (prompt_ids.len() - 1) as u32;
        let logits = executor.forward(last, pos, kv)?;
        let elapsed = t0.elapsed();
        if warmup_steps == 0 { step_times.push(elapsed); }

        let next = sample(&logits, sampler_cfg, &[], rng);
        (prompt_ids.len() as u32, next)
    } else {
        (0_u32, prompt_ids[0])
    };

    let mut step = 0usize;
    let mut pos  = start_pos;

    while step < num_tokens {
        let t0 = std::time::Instant::now();

        // Always use forward() for f32 logits + repetition penalty support
        let logits = executor.forward(next_token, pos, kv)?;
        let elapsed = t0.elapsed();

        if step >= warmup_steps {
            step_times.push(elapsed);
        }

        let max_l = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let min_l = logits.iter().copied().fold(f32::INFINITY,     f32::min);
        let marker = if step < warmup_steps { "W" } else { " " };
        println!(
            " {:2}{} | {:6} | [{:8.2}, {:7.2}] | {:.2}",
            step, marker, next_token, min_l, max_l,
            elapsed.as_secs_f64() * 1000.0,
        );

        let sampled = sample(&logits, sampler_cfg, generated.as_slice(), rng);
        generated.push(sampled);

        pos  += 1;
        step += 1;

        if special.is_stop(sampled) { break; }
        next_token = sampled;
    }
    Ok(())
}

// ── TurboQuant KV-cache generation loop ──────────────────────────────────────

#[cfg(target_os = "macos")]
fn run_loop_tq(
    executor:     &bare_metal_engine::forward::Executor,
    kv:           &mut bare_metal_engine::tq_kv_cache::TqKvCache,
    prompt_ids:   &[u32],
    num_tokens:   usize,
    warmup_steps: usize,
    sampler_cfg:  &bare_metal_engine::sampler::SamplerConfig,
    special:      &bare_metal_engine::chat_template::SpecialTokens,
    rng:          &mut bare_metal_engine::sampler::SimpleRng,
    step_times:   &mut Vec<std::time::Duration>,
    generated:    &mut Vec<u32>,
) -> anyhow::Result<()> {
    use bare_metal_engine::sampler::sample;

    let mut pos: u32 = 0;
    let mut next_token = prompt_ids[0];

    for step in 0..num_tokens {
        let t0 = std::time::Instant::now();
        let logits = executor.forward_tq(next_token, pos, kv)?;
        let elapsed = t0.elapsed();

        if step >= warmup_steps {
            step_times.push(elapsed);
        }

        let logits_f32: Vec<f32> = logits.iter().map(|x| x.to_f32()).collect();
        let max_l = logits_f32.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let min_l = logits_f32.iter().copied().fold(f32::INFINITY,     f32::min);
        let marker = if step < warmup_steps { "W" } else { " " };
        println!(
            " {:2}{} | {:6} | [{:8.2}, {:7.2}] | {:.2}",
            step, marker, next_token, min_l, max_l,
            elapsed.as_secs_f64() * 1000.0,
        );

        let sampled = sample(&logits_f32, sampler_cfg, generated.as_slice(), rng);
        generated.push(sampled);
        pos += 1;

        if special.is_stop(sampled) { break; }
        next_token = sampled;
    }
    Ok(())
}

// ── Aggregate timing summary ──────────────────────────────────────────────────

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn print_summary(
    step_times:    &[std::time::Duration],
    warmup_steps:  usize,
    num_tokens:    usize,
    prompt_tokens: usize,
    load_ms:       u64,
    upload_ms:     u64,
    kv_mb:         f64,
    use_tq:        bool,
    eos_pos:       Option<usize>,
    unique_ratio:  f64,
    repeat_rate:   f64,
    json_path:     Option<&str>,
) {
    if step_times.is_empty() {
        println!("No benchmark steps recorded.");
        return;
    }

    let total_ms: f64 = step_times.iter().map(|d| d.as_secs_f64() * 1000.0).sum();
    let avg_ms   = total_ms / step_times.len() as f64;
    let min_ms   = step_times.iter().map(|d| d.as_secs_f64() * 1000.0).fold(f64::INFINITY, f64::min);
    let max_ms   = step_times.iter().map(|d| d.as_secs_f64() * 1000.0).fold(0.0f64, f64::max);
    let tok_sec  = 1000.0 / avg_ms;

    let mut sorted: Vec<f64> = step_times.iter().map(|d| d.as_secs_f64() * 1000.0).collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = sorted[sorted.len() / 2];
    let p95 = sorted[(sorted.len() as f64 * 0.95).min(sorted.len() as f64 - 1.0) as usize];

    let cache_type = if use_tq { "TurboQuant" } else { "F16" };

    println!();
    println!("╔══════════════════════════════════════╗");
    println!("║      TIMING SUMMARY                  ║");
    println!("╠══════════════════════════════════════╣");
    println!("║  prompt tokens    : {:5}             ║", prompt_tokens);
    println!("║  tokens generated : {:5}             ║", num_tokens);
    println!("║  warmup excluded  : {:5}             ║", warmup_steps);
    println!("║  measured steps   : {:5}             ║", step_times.len());
    println!("║  kv cache type    : {:<16}  ║", cache_type);
    println!("║  kv cache size    : {:7.1} MB        ║", kv_mb);
    println!("╠══════════════════════════════════════╣");
    println!("║  model load       : {:6} ms          ║", load_ms);
    println!("║  weight upload    : {:6} ms          ║", upload_ms);
    println!("╠══════════════════════════════════════╣");
    println!("║  throughput       : {:7.2} tok/sec   ║", tok_sec);
    println!("║  avg latency      : {:7.2} ms/tok    ║", avg_ms);
    println!("║  p50 latency      : {:7.2} ms/tok    ║", p50);
    println!("║  p95 latency      : {:7.2} ms/tok    ║", p95);
    println!("║  min latency      : {:7.2} ms/tok    ║", min_ms);
    println!("║  max latency      : {:7.2} ms/tok    ║", max_ms);
    println!("║  total decode     : {:7.2} ms        ║", total_ms);
    println!("╚══════════════════════════════════════╝");

    println!(
        "BENCHMARK_RESULT avg_ms={avg_ms:.2} p50_ms={p50:.2} \
         p95_ms={p95:.2} tok_sec={tok_sec:.2} steps={}",
        step_times.len()
    );

    // ── Optional JSON summary ─────────────────────────────────────────────────
    if let Some(path) = json_path {
        let eos_hit  = eos_pos.is_some();
        let eos_pos_val = eos_pos.map(|p| p as i64).unwrap_or(-1);
        let json = format!(
            "{{\
             \"prompt_tokens\":{prompt_tokens},\
             \"decode_tokens\":{},\
             \"warmup_tokens\":{warmup_steps},\
             \"kv_cache_type\":\"{cache_type}\",\
             \"kv_cache_mb\":{kv_mb:.2},\
             \"load_ms\":{load_ms},\
             \"upload_ms\":{upload_ms},\
             \"tok_per_sec\":{tok_sec:.2},\
             \"avg_ms\":{avg_ms:.2},\
             \"p50_ms\":{p50:.2},\
             \"p95_ms\":{p95:.2},\
             \"min_ms\":{min_ms:.2},\
             \"max_ms\":{max_ms:.2},\
             \"total_ms\":{total_ms:.2},\
             \"measured_steps\":{},\
             \"eos_hit\":{eos_hit},\
             \"eos_pos\":{eos_pos_val},\
             \"unique_token_pct\":{:.1},\
             \"repeat_rate_pct\":{:.1}\
             }}",
            step_times.len(),
            step_times.len(),
            unique_ratio * 100.0,
            repeat_rate * 100.0,
        );
        if let Err(e) = std::fs::write(path, &json) {
            eprintln!("Warning: could not write JSON summary to {path}: {e}");
        }
    }
}
