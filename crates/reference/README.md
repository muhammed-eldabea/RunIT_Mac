# bare-metal-reference

Numerical validation harness for GPU kernel correctness testing.

Part of the [RunIT Engine](https://github.com/muhammed-eldabea/RunIT_Mac) — a from-scratch LLM inference engine in Rust + Metal for Apple Silicon.

## Features

- **Layer output validation** — compare GPU kernel outputs against CPU reference
- **Tolerance-based matching** — configurable absolute tolerance for f16/f32 comparison
- **Detailed error reporting** — reports max diff, mismatch count, and element positions

## Usage

```rust
use bare_metal_reference::validate_layer_output;

let reference = vec![1.0, 2.0, 3.0];
let gpu_output = vec![1.001, 2.002, 2.998];

validate_layer_output(&reference, &gpu_output, 0, 0.01)?;
// OK — all values within 1% tolerance
```

## License

MIT
