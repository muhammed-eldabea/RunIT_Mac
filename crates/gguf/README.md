# bare-metal-gguf

Zero-copy GGUF file parser for LLM inference.

Part of the [RunIT Engine](https://github.com/muhammed-eldabea/RunIT_Mac) — a from-scratch LLM inference engine in Rust + Metal for Apple Silicon.

## Features

- **Memory-mapped parsing** — `mmap()` the GGUF file, zero heap allocation for tensor data
- **All quantization types** — Q2K, Q3K, Q4K, Q4\_0, Q5\_0, Q5K, Q6K, Q8\_0, Q8K, F16, F32, BF16
- **Metadata extraction** — architecture, hyperparameters, tokenizer vocabulary
- **Tensor buffer views** — raw pointer + size + dtype for each weight tensor

## Usage

```rust
use bare_metal_gguf::GgufFile;

let file = GgufFile::open("model.gguf")?;

// Access metadata
let arch = file.metadata.get("general.architecture");

// Iterate tensors
for (name, tensor) in &file.tensors {
    println!("{}: {:?} {:?}", name, tensor.shape, tensor.dtype);
}
```

## Supported GGUF Versions

- GGUF v3 (current standard, used by llama.cpp)

## License

MIT
