use criterion::{criterion_group, criterion_main, Criterion};
use std::io::Cursor;

use bare_metal_gguf::GgufParser;

fn build_test_gguf() -> Vec<u8> {
    let mut buf: Vec<u8> = Vec::new();

    buf.extend_from_slice(&0x46475547u32.to_le_bytes());
    buf.extend_from_slice(&3u32.to_le_bytes());
    buf.extend_from_slice(&1u64.to_le_bytes());
    buf.extend_from_slice(&1u64.to_le_bytes());

    let key = b"general.architecture";
    buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
    buf.extend_from_slice(key);
    buf.extend_from_slice(&8u32.to_le_bytes());
    let val = b"qwen2";
    buf.extend_from_slice(&(val.len() as u64).to_le_bytes());
    buf.extend_from_slice(val);

    let name = b"test.weight";
    buf.extend_from_slice(&(name.len() as u64).to_le_bytes());
    buf.extend_from_slice(name);
    buf.extend_from_slice(&2u32.to_le_bytes());
    buf.extend_from_slice(&4u64.to_le_bytes());
    buf.extend_from_slice(&8u64.to_le_bytes());
    buf.extend_from_slice(&0u32.to_le_bytes());
    buf.extend_from_slice(&0u64.to_le_bytes());

    let current = buf.len() as u64;
    let aligned = if current % 32 == 0 {
        current
    } else {
        current + (32 - current % 32)
    };
    buf.resize(aligned as usize, 0u8);

    for i in 0..32u32 {
        buf.extend_from_slice(&(i as f32).to_le_bytes());
    }

    buf
}

fn bench_gguf_parse(c: &mut Criterion) {
    let data = build_test_gguf();
    c.bench_function("parse_minimal_gguf", |b| {
        b.iter(|| {
            let cursor = Cursor::new(&data);
            let mut parser = GgufParser::new(cursor);
            parser.parse().unwrap()
        })
    });
}

criterion_group!(benches, bench_gguf_parse);
criterion_main!(benches);
