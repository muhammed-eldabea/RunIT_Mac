---
name: test-server
description: Start and test the OpenAI-compatible HTTP inference server. Tests /v1/chat/completions endpoint with curl. Use when working on the API server.
---

# Test HTTP Server

You are testing an OpenAI-compatible HTTP API server for the bare-metal LLM inference engine.

## Server Architecture

- Framework: Axum 0.7 (async, Tokio runtime)
- Endpoint: `POST /v1/chat/completions`
- Format: OpenAI-compatible request/response
- Source: `crates/engine/src/server/` (mod.rs, types.rs, handlers.rs)
- Binary: `crates/engine/src/bin/serve.rs`

## Starting the Server

```bash
cargo run --release --bin serve -- --model <path-to-model.gguf> --port 8080 2>&1 &
```

Wait for "Listening on 0.0.0.0:8080" before sending requests.

## Testing Endpoints

### Basic chat completion
```bash
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is 2+2?"}
    ],
    "max_tokens": 50,
    "temperature": 0.7
  }' | python3 -m json.tool
```

### Streaming response
```bash
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 50,
    "stream": true
  }'
```

### Health check / model info
```bash
curl -s http://localhost:8080/v1/models | python3 -m json.tool
```

### Multiple messages (conversation)
```bash
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5",
    "messages": [
      {"role": "system", "content": "You are a coding assistant."},
      {"role": "user", "content": "Write a Python hello world"},
      {"role": "assistant", "content": "print(\"Hello, World!\")"},
      {"role": "user", "content": "Now make it a function"}
    ],
    "max_tokens": 100
  }' | python3 -m json.tool
```

## Validation Checklist

- [ ] Server starts without errors
- [ ] Response has correct OpenAI format (id, object, choices, usage)
- [ ] Token counts in `usage` field are accurate
- [ ] Streaming sends proper SSE events (`data: {...}\n\n`)
- [ ] Streaming ends with `data: [DONE]\n\n`
- [ ] Error responses have proper HTTP status codes
- [ ] Server handles concurrent requests (test with multiple curl in parallel)

## Stopping the Server

```bash
# Find and kill the server process
kill $(lsof -ti:8080) 2>/dev/null
```

## After Testing

Report:
1. Whether server started successfully
2. Response format correctness
3. Generation quality and speed
4. Any errors or malformed responses
5. Suggested fixes for issues found
