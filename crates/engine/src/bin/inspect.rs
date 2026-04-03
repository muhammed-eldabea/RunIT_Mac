/// inspect — load a GGUF file and print its config + tensor inventory
///
/// Usage:
///   cargo run -p bare-metal-engine --bin inspect -- path/to/model.gguf
///   cargo run -p bare-metal-engine --bin inspect -- path/to/model.gguf --no-validate

use std::path::Path;
use std::process;

use bare_metal_engine::loader::{load_model_opts, LoadOptions};

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: inspect <model.gguf> [--no-validate]");
        process::exit(1);
    }

    let path = Path::new(&args[1]);
    let validate = !args.iter().any(|a| a == "--no-validate");

    if !path.exists() {
        eprintln!("Error: file not found: {}", path.display());
        process::exit(1);
    }

    let file_size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);

    println!("═══════════════════════════════════════════════════════════════");
    println!("  GGUF Model Inspector");
    println!("═══════════════════════════════════════════════════════════════");
    println!("  File : {}", path.display());
    println!("  Size : {:.2} GB ({} bytes)", file_size as f64 / 1e9, file_size);
    println!();

    let opts = LoadOptions {
        validate_weights: validate,
    };

    match load_model_opts(path, opts) {
        Ok(model) => {
            println!("── Model Configuration ────────────────────────────────────────");
            println!("{}", model.config);
            println!();

            println!("── Tensor Inventory ({} tensors) ─────────────────────────────", model.num_tensors());
            print!("{}", model.tensor_summary());

            // Memory footprint estimate
            let total_bytes: usize = model.tensors.values().map(|t| t.data_size).sum();
            println!();
            println!("── Memory ─────────────────────────────────────────────────────");
            println!("  Weights  : {:.2} GB ({} bytes)", total_bytes as f64 / 1e9, total_bytes);
            println!(
                "  KV cache : {:.2} MB / token (F16, {} layers)",
                model.config.kv_cache_bytes_per_token() as f64 * model.config.num_hidden_layers as f64 / 1e6,
                model.config.num_hidden_layers,
            );

            // Spot-check: first tensor raw bytes via zero-copy pointer
            if let Some(tb) = model.weights().token_embedding().and_then(|_| model.tensor_buffer("token_embd.weight")) {
                println!();
                println!("── Zero-copy pointer check ────────────────────────────────────");
                println!("  token_embd.weight ptr = {:p}, size = {} bytes", tb.ptr, tb.size);
                // Safely read first 4 bytes as f16 placeholder value
                let first_bytes: [u8; 2] = unsafe { [*tb.ptr, *tb.ptr.add(1)] };
                println!("  First 2 raw bytes     = [{:#04x}, {:#04x}]", first_bytes[0], first_bytes[1]);
            }

            println!();
            println!("✓ Model loaded successfully");
        }
        Err(e) => {
            eprintln!("Error loading model: {e}");
            process::exit(1);
        }
    }
}
