---
name: bench-perf
description: Run performance benchmarks for the inference engine. Measures tokens/sec, kernel latency, GGUF parse time, and memory usage. Use when optimizing or validating performance.
---

# Benchmark Performance

You are benchmarking a bare-metal LLM inference engine on Apple Silicon.

## Available Benchmarks

### 1. End-to-end generation benchmark
The `generate` binary outputs timing information:
```bash
cargo run --release --bin generate -- \
  --model <path-to-model.gguf> \
  --prompt "Explain quantum computing in detail" \
  --max-tokens 256 \
  2>&1
```
Look for: tokens/sec, time-to-first-token, total time.

### 2. GGUF parsing benchmark (Criterion)
```bash
cargo bench -p bare-metal-bench 2>&1
```

### 3. Kernel-level timing
Run individual kernel tests and measure dispatch time:
```bash
cargo test -p bare-metal-kernels -- --nocapture 2>&1
```

## Key Performance Metrics

| Metric | Target (7B Q4K, M1 Pro) | Where to Find |
|--------|--------------------------|---------------|
| Decode tok/s | 30-50 | generate binary output |
| Time to first token | <500ms | generate binary output |
| GGUF parse | <100ms | criterion bench |
| Memory usage | <8GB | Activity Monitor / `vm_stat` |

## Profiling Tools

### Metal System Trace
```bash
# Capture GPU trace
export MTL_CAPTURE_ENABLED=1
cargo run --release --bin generate -- --model <path> --prompt "test" --max-tokens 10 2>&1
```

### Instruments (if deeper profiling needed)
```bash
xcrun xctrace record --template "Metal System Trace" --launch -- \
  target/release/generate --model <path> --prompt "test" --max-tokens 10
```

### Memory profiling
```bash
# Check peak memory during generation
/usr/bin/time -l target/release/generate --model <path> --prompt "test" --max-tokens 50 2>&1
```

## Optimization Checklist

When investigating slow performance:
1. **Build mode**: Must be `--release` (debug is 10-50x slower)
2. **Quantization**: Q4_K_M is fastest, F16 uses most memory bandwidth
3. **KV cache**: TurboQuant compression reduces memory pressure (check `tq_kv_cache.rs`)
4. **Threadgroup sizing**: Check dispatch.rs grid calculations match hardware limits
5. **Buffer allocation**: Large allocations at startup, not per-token
6. **Synchronous dispatch**: Current design blocks per-kernel — pipelining is a future optimization

## After Benchmarking

Report:
1. Tokens/sec achieved
2. Comparison to targets above
3. Memory usage
4. Bottleneck identification (compute-bound vs memory-bound)
5. Specific optimization suggestions with file locations
