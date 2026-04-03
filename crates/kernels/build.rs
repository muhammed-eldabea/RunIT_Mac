use std::path::Path;
use std::process::Command;

fn main() {
    // Metal compilation is macOS-only
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos") {
        return;
    }

    // Allow CI (or developers) to pre-compile the Metal library and pass its path
    // via the METALLIB_PATH environment variable.  build.rs sets rerun-if-env-changed
    // so cargo re-runs this script if the variable changes.
    println!("cargo:rerun-if-env-changed=METALLIB_PATH");
    if let Ok(prebuilt) = std::env::var("METALLIB_PATH") {
        if Path::new(&prebuilt).exists() {
            println!("cargo:rustc-env=METALLIB_PATH={prebuilt}");
            println!("cargo:warning=Using pre-built METALLIB_PATH={prebuilt}");
            return;
        } else {
            println!("cargo:warning=METALLIB_PATH={prebuilt} set but file not found; will try xcrun");
        }
    }

    let shader_dir = Path::new("shaders");
    println!("cargo:rerun-if-changed=shaders/");

    // Collect all .metal files
    let metal_files: Vec<_> = match std::fs::read_dir(shader_dir) {
        Ok(entries) => entries
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().map(|x| x == "metal").unwrap_or(false))
            .map(|e| e.path())
            .collect(),
        Err(_) => {
            println!("cargo:warning=No shaders/ directory found, skipping Metal compilation");
            return;
        }
    };

    if metal_files.is_empty() {
        println!("cargo:warning=No .metal files in shaders/, skipping Metal compilation");
        return;
    }

    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR not set");
    let metallib_path = Path::new(&out_dir).join("kernels.metallib");

    // Compile each .metal → .air
    let mut air_files = Vec::new();
    for metal_file in &metal_files {
        let stem = metal_file.file_stem().unwrap().to_str().unwrap();
        let air_path = Path::new(&out_dir).join(format!("{stem}.air"));

        let result = Command::new("xcrun")
            .args(["metal", "-c", "-o"])
            .arg(&air_path)
            .arg(metal_file)
            .output();

        match result {
            Ok(out) if out.status.success() => {}
            Ok(out) => {
                let stderr = String::from_utf8_lossy(&out.stderr);
                println!(
                    "cargo:warning=Metal shader compilation failed (exit {}) for {}",
                    out.status.code().unwrap_or(-1),
                    metal_file.display()
                );
                // Emit each line of the compiler error as a separate warning
                // so it surfaces in `cargo build` output.
                for line in stderr.lines() {
                    println!("cargo:warning=  {line}");
                }
                return;
            }
            Err(e) => {
                println!(
                    "cargo:warning=xcrun not found ({}): Metal shaders will not be compiled. \
                     Install Xcode and run `xcode-select --install`.",
                    e
                );
                return;
            }
        }
        air_files.push(air_path);
    }

    // Link all .air → .metallib
    let mut cmd = Command::new("xcrun");
    cmd.args(["metallib", "-o"]).arg(&metallib_path);
    for air in &air_files {
        cmd.arg(air);
    }

    match cmd.status() {
        Ok(s) if s.success() => {}
        Ok(s) => {
            println!(
                "cargo:warning=metallib linking failed (exit {}): \
                 kernels will be unavailable at runtime",
                s.code().unwrap_or(-1)
            );
            return;
        }
        Err(e) => {
            println!("cargo:warning=xcrun metallib not found ({})", e);
            return;
        }
    }

    // Expose path to Rust code via env var — only set when compilation succeeded
    println!("cargo:rustc-env=METALLIB_PATH={}", metallib_path.display());
    println!(
        "cargo:warning=Compiled {} Metal shader(s) → {}",
        metal_files.len(),
        metallib_path.display()
    );
}
