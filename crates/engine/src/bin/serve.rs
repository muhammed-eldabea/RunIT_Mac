/// `serve` — OpenAI-compatible HTTP server for bare-metal LLM inference.
///
/// Usage:
///   cargo run --release -p bare-metal-engine --bin serve -- \
///       <model.gguf> <tokenizer.json> [options]
///
/// Options:
///   --host <addr>    Listen address (default: 127.0.0.1)
///   --port <port>    Listen port    (default: 8080)
///   --model-id <id>  Model ID in API responses (default: filename stem)
///
/// Endpoints:
///   GET  /health
///   GET  /v1/models
///   POST /v1/chat/completions   (blocking + SSE streaming)

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("serve requires macOS (Metal GPU)");
    std::process::exit(1);
}

#[cfg(target_os = "macos")]
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    use std::net::SocketAddr;
    use std::path::Path;
    use std::sync::{Arc, Mutex};

    use bare_metal_engine::{
        forward::{Executor, ForwardError},
        load_model,
        server::{build_router, AppState},
    };
    use bare_metal_kernels::context::MetalContext;
    use bare_metal_tokenizer::Tokenizer;

    // ── Arg parsing ───────────────────────────────────────────────────────────
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 || args[1] == "--help" || args[1] == "-h" {
        eprintln!(
            "Usage: serve <model.gguf> <tokenizer.json> \
             [--host <addr>] [--port <port>] [--model-id <id>]"
        );
        std::process::exit(if args.len() < 3 { 1 } else { 0 });
    }

    let model_path     = Path::new(&args[1]);
    let tokenizer_path = Path::new(&args[2]);

    let mut host      = "127.0.0.1".to_string();
    let mut port: u16 = 8080;
    let mut model_id  = model_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("model")
        .to_string();

    let mut i = 3;
    while i < args.len() {
        match args[i].as_str() {
            "--host"     if i + 1 < args.len() => { host     = args[i+1].clone(); i += 2; }
            "--port"     if i + 1 < args.len() => { port     = args[i+1].parse()?; i += 2; }
            "--model-id" if i + 1 < args.len() => { model_id = args[i+1].clone(); i += 2; }
            other => { eprintln!("Unknown argument: {other}"); i += 1; }
        }
    }

    // ── Load ──────────────────────────────────────────────────────────────────
    println!("Loading model  : {}", model_path.display());
    let model = load_model(model_path)?;

    println!("Loading tokenizer: {}", tokenizer_path.display());
    let tokenizer = Tokenizer::from_file(tokenizer_path)?;

    println!("Initialising Metal context…");
    let ctx = MetalContext::new().map_err(|e| anyhow::anyhow!("Metal: {e}"))?;
    println!("GPU: {}", ctx.device.name());

    let executor = match Executor::new(ctx, &model) {
        Ok(e) => e,
        Err(ForwardError::UnsupportedDtype { tensor, dtype }) => {
            anyhow::bail!("tensor '{tensor}' has unsupported dtype {dtype:?}");
        }
        Err(e) => return Err(e.into()),
    };

    let state = Arc::new(AppState {
        model_id:  model_id.clone(),
        tokenizer,
        executor:  Mutex::new(executor),
    });

    // ── Start server ──────────────────────────────────────────────────────────
    let addr: SocketAddr = format!("{host}:{port}").parse()?;
    let router = build_router(state);

    println!("╔══════════════════════════════════════════╗");
    println!("║  Bare-Metal MX LLM — HTTP Server         ║");
    println!("╠══════════════════════════════════════════╣");
    println!("║  model    : {:<29}║", model_id);
    println!("║  address  : http://{:<21}║", addr);
    println!("╠══════════════════════════════════════════╣");
    println!("║  POST /v1/chat/completions               ║");
    println!("║  GET  /v1/models                         ║");
    println!("║  GET  /health                            ║");
    println!("╚══════════════════════════════════════════╝");

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, router).await?;

    Ok(())
}
