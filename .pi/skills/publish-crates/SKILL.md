---
name: publish-crates
description: Publish all RunIT crates to crates.io. Bumps version, updates per-crate READMEs with latest benchmark results, commits, tags, and publishes in dependency order.
---

# Publish Crates to crates.io

## Pre-flight Checklist

1. All shaders compile: `runit-check-shaders`
2. Release build succeeds: `runit-build --mode release`
3. Kernel tests pass: `runit-test --crate bare-metal-kernels --filter gemv`
4. Benchmark numbers are current (run generate with Q8_0 model)

## Steps

### 1. Bump Version

Update `version` in the root `Cargo.toml` workspace section.
All crates inherit via `version.workspace = true`.
Also update the `version` in workspace dependency declarations.

```bash
# Example: bump from 0.7.1 to 0.7.2
sed -i '' 's/version = "0.7.1"/version = "0.7.2"/g' Cargo.toml
```

### 2. Update Per-Crate READMEs

Update benchmark numbers in these files:
- `crates/engine/README.md` — tok/sec table, model support
- `crates/kernels/README.md` — kernel count, shader count
- `crates/gguf/README.md` — supported quantization types
- `crates/tokenizer/README.md` — vocab size if changed
- `crates/reference/README.md` — rarely changes

Key numbers to update:
- tok/sec for Qwen2.5-0.5B Q8_0
- tok/sec for Qwen3-0.6B Q8_0
- Total kernel count (grep KERNEL_NAMES in context.rs)
- Total shader file count (ls crates/kernels/shaders/*.metal | wc -l)

### 3. Update Root README

Update `README.md` with:
- Highlights box (tok/sec, speedup)
- Performance table
- Model support table
- Quantization support table
- Roadmap phase status

### 4. Commit and Tag

```bash
git add -A
git commit -m "chore: bump to v${VERSION} for crates.io publish

Updated: per-crate READMEs, benchmark numbers, version"
git push origin main
git tag -a v${VERSION} -m "v${VERSION} — ${DESCRIPTION}"
git push origin v${VERSION}
```

### 5. Publish in Dependency Order

Crates MUST be published in this order (each depends on the previous):

```bash
# 1. No internal deps
cargo publish -p bare-metal-gguf
# Wait for index update (~15s)

# 2. No internal deps  
cargo publish -p bare-metal-tokenizer

# 3. No internal deps
cargo publish -p bare-metal-reference

# 4. Depends on: bare-metal-reference
cargo publish -p bare-metal-kernels

# 5. Depends on: bare-metal-gguf, bare-metal-tokenizer, bare-metal-kernels
cargo publish -p bare-metal-engine
```

Note: `bare-metal-bench` has `publish = false` and is skipped.

### 6. Verify

Check all crates are live:
- https://crates.io/crates/bare-metal-gguf
- https://crates.io/crates/bare-metal-tokenizer
- https://crates.io/crates/bare-metal-reference
- https://crates.io/crates/bare-metal-kernels
- https://crates.io/crates/bare-metal-engine

## Common Issues

- **"verified email required"**: Visit https://crates.io/settings/profile
- **"allow-dirty"**: Commit all changes before publishing
- **"already uploaded"**: Version already exists, bump version number
- **"dependency not found"**: Wait 15-30s between publishes for index to update
