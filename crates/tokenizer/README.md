# bare-metal-tokenizer

HuggingFace BPE tokenizer wrapper for LLM inference.

Part of the [RunIT Engine](https://github.com/muhammed-eldabea/RunIT_Mac) — a from-scratch LLM inference engine in Rust + Metal for Apple Silicon.

## Features

- **HuggingFace compatible** — loads `tokenizer.json` files directly
- **BPE tokenization** — encode text → token IDs, decode token IDs → text
- **151K+ vocabulary** — supports Qwen, LLaMA, Mistral tokenizers

## Usage

```rust
use bare_metal_tokenizer::Tokenizer;

let tokenizer = Tokenizer::from_file("tokenizer.json")?;

// Encode
let ids = tokenizer.encode("Hello, world!", false)?;

// Decode
let text = tokenizer.decode(&ids, true)?;
```

## License

MIT
