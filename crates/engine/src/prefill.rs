/// Prefill forward pass — Phase 8.
///
/// Processes the entire prompt in a single batched GPU pass using GEMM instead
/// of N sequential GEMV calls. Returns logits for the last token.
///
/// vs decode (`forward()`):
///   - GEMV  (single query)  → GEMM  (all prompt positions at once)
///   - single-pos RoPE       → batched RoPE covering positions 0..seq_len-1
///   - FlashAttention with q_len = seq_len (causal mask handled by the kernel)

use half::f16;
use metal::MTLResourceOptions;

use crate::forward::{Executor, ForwardError};
use crate::kv_cache::KvCache;

use bare_metal_kernels::dispatch::{
    add_bias_broadcast_f16, add_f16, flash_attention_f16, gemm_f16, gemv_f16,
    rms_norm_f16, rope_batch_inplace_f16,
    scale_accumulate_f16, silu_mul_f16,
};

impl Executor {
    /// Process `tokens` (a full prompt) in one batched pass and return the
    /// logits for the final token position. All KV positions are populated.
    ///
    /// `kv` must be empty (freshly created or reset). After this call
    /// `kv.seq_len() == tokens.len()`.
    pub fn prefill(
        &self,
        tokens: &[u32],
        kv: &mut KvCache,
    ) -> Result<Vec<f16>, ForwardError> {
        if tokens.is_empty() {
            return Err(ForwardError::InvalidTokenId(0));
        }

        let cfg  = &self.config;
        let ctx  = &self.ctx;
        let opts = MTLResourceOptions::StorageModeShared;
        let seq  = tokens.len();

        let h      = cfg.hidden_size;
        let q_dim  = cfg.num_attention_heads * cfg.head_dim;
        let kv_dim = cfg.num_key_value_heads  * cfg.head_dim;
        let ff_dim = cfg.intermediate_size;

        // ── 1. Embed all tokens → X [seq, h] ─────────────────────────────────
        let x_buf = ctx.device.new_buffer((seq * h * 2) as u64, opts);
        {
            let emb_row = h * 2;
            let dst = x_buf.contents() as *mut u8;
            let src = self.weights.tok_emb.contents() as *const u8;
            for (i, &tid) in tokens.iter().enumerate() {
                if tid >= cfg.vocab_size as u32 {
                    return Err(ForwardError::InvalidTokenId(tid));
                }
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        src.add(tid as usize * emb_row),
                        dst.add(i * emb_row),
                        emb_row,
                    );
                }
            }
        }

        // Scratch buffers for [seq, dim] activations
        let x_norm_buf   = ctx.device.new_buffer((seq * h     * 2) as u64, opts);
        let q_buf        = ctx.device.new_buffer((seq * q_dim * 2) as u64, opts);
        let k_buf        = ctx.device.new_buffer((seq * kv_dim* 2) as u64, opts);
        let v_buf        = ctx.device.new_buffer((seq * kv_dim* 2) as u64, opts);
        let attn_out_buf = ctx.device.new_buffer((seq * q_dim * 2) as u64, opts);
        let proj_buf     = ctx.device.new_buffer((seq * h     * 2) as u64, opts);
        let x_norm2_buf  = ctx.device.new_buffer((seq * h     * 2) as u64, opts);
        let gate_buf     = ctx.device.new_buffer((seq * ff_dim* 2) as u64, opts);
        let up_buf       = ctx.device.new_buffer((seq * ff_dim* 2) as u64, opts);
        let act_buf      = ctx.device.new_buffer((seq * ff_dim* 2) as u64, opts);
        let ffn_out_buf  = ctx.device.new_buffer((seq * h     * 2) as u64, opts);

        // ── 2. Transformer layers ─────────────────────────────────────────────
        for l in 0..cfg.num_hidden_layers {
            let w = &self.weights;

            // a) Attention pre-norm (batch)
            rms_norm_f16(ctx, &x_buf, &x_norm_buf, &w.attn_norm[l],
                         cfg.rms_norm_eps, h as u32, seq as u32)?;

            // b) QKV projections (GEMM — uses F16 weights for batched matmul)
            gemm_f16(ctx, &x_norm_buf, w.attn_q[l].f16_buf(), &q_buf,
                     seq as u32, q_dim as u32, h as u32)?;
            gemm_f16(ctx, &x_norm_buf, w.attn_k[l].f16_buf(), &k_buf,
                     seq as u32, kv_dim as u32, h as u32)?;
            gemm_f16(ctx, &x_norm_buf, w.attn_v[l].f16_buf(), &v_buf,
                     seq as u32, kv_dim as u32, h as u32)?;

            // c) Optional biases — single broadcast kernel per bias (Phase 4 optimisation)
            if let Some(bias) = &w.attn_q_bias[l] {
                add_bias_broadcast_f16(ctx, &q_buf, bias, seq as u32, q_dim as u32)?;
            }
            if let Some(bias) = &w.attn_k_bias[l] {
                add_bias_broadcast_f16(ctx, &k_buf, bias, seq as u32, kv_dim as u32)?;
            }
            if let Some(bias) = &w.attn_v_bias[l] {
                add_bias_broadcast_f16(ctx, &v_buf, bias, seq as u32, kv_dim as u32)?;
            }

            // d) Batched RoPE
            rope_batch_inplace_f16(ctx, &q_buf, cfg.head_dim as u32,
                                   cfg.num_attention_heads as u32,
                                   0, cfg.rope_theta, seq as u32)?;
            rope_batch_inplace_f16(ctx, &k_buf, cfg.head_dim as u32,
                                   cfg.num_key_value_heads as u32,
                                   0, cfg.rope_theta, seq as u32)?;

            // e) Populate KV cache for all seq positions.
            //    Flush GPU so RoPE results are visible, then copy directly.
            ctx.flush();
            let kh = cfg.num_key_value_heads;
            let hd = cfg.head_dim;
            let k_ptr = k_buf.contents() as *const f16;
            let v_ptr = v_buf.contents() as *const f16;
            for pos in 0..seq {
                let k_slice: Vec<f16> = unsafe {
                    (0..kh * hd).map(|i| *k_ptr.add(pos * kh * hd + i)).collect()
                };
                let v_slice: Vec<f16> = unsafe {
                    (0..kh * hd).map(|i| *v_ptr.add(pos * kh * hd + i)).collect()
                };
                kv.write_at(l, pos, &k_slice, &v_slice);
            }

            // f) FlashAttention for the full prompt (q_len = seq, causal).
            flash_attention_f16(
                ctx, &q_buf, kv.k_buf(l), kv.v_buf(l), &attn_out_buf,
                1,                  // batch
                seq as u32,         // q_len = full prompt
                seq as u32,         // kv_len = mask bound
                cfg.num_attention_heads  as u32,
                cfg.num_key_value_heads  as u32,
                cfg.head_dim as u32,
                kv.max_seq_len(),   // kv_head_stride = allocated capacity, not seq
            )?;

            // g) Output projection
            gemm_f16(ctx, &attn_out_buf, w.attn_out[l].f16_buf(), &proj_buf,
                     seq as u32, h as u32, q_dim as u32)?;

            // h) Residual
            add_f16(ctx, &x_buf, &proj_buf, &x_buf, (seq * h) as u32)?;

            // i) FFN pre-norm
            rms_norm_f16(ctx, &x_buf, &x_norm2_buf, &w.ffn_norm[l],
                         cfg.rms_norm_eps, h as u32, seq as u32)?;

            if cfg.is_moe() {
                // ── MoE FFN (prefill): process each token individually through router ──
                // Each token may route to different experts. We process sequentially
                // (correctness first; batched expert dispatch is a future optimization).
                let n_exp = cfg.num_experts;
                let top_k = cfg.num_experts_per_tok;
                let eff_dim = self.expert_ff_dim;

                let router_buf = ctx.device.new_buffer((n_exp * 2) as u64, opts);
                let eg_buf = ctx.device.new_buffer((eff_dim * 2) as u64, opts);
                let eu_buf = ctx.device.new_buffer((eff_dim * 2) as u64, opts);
                let ea_buf = ctx.device.new_buffer((eff_dim * 2) as u64, opts);
                let eo_buf = ctx.device.new_buffer((h * 2) as u64, opts);
                let token_in = ctx.device.new_buffer((h * 2) as u64, opts);
                let token_out = ctx.device.new_buffer((h * 2) as u64, opts);

                for pos in 0..seq {
                    // Extract this token's x_norm2 row
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            (x_norm2_buf.contents() as *const u8).add(pos * h * 2),
                            token_in.contents() as *mut u8, h * 2);
                    }

                    // Router
                    gemv_f16(ctx, &w.moe_router[l], &token_in, &router_buf,
                             n_exp as u32, h as u32)?;
                    ctx.flush();

                    let (expert_ids, expert_weights) = crate::forward::topk_softmax(
                        &router_buf, n_exp, top_k);

                    // Zero token_out as accumulator
                    unsafe {
                        std::ptr::write_bytes(token_out.contents() as *mut u8, 0, h * 2);
                    }

                    // GPU-side expert dispatch + accumulate (no per-expert flush)
                    for (i, &eid) in expert_ids.iter().enumerate() {
                        w.moe_gate_exps[l][eid].gemv(ctx, &token_in, &eg_buf,
                                                      eff_dim as u32, h as u32)?;
                        w.moe_up_exps[l][eid].gemv(ctx, &token_in, &eu_buf,
                                                    eff_dim as u32, h as u32)?;
                        silu_mul_f16(ctx, &eg_buf, &eu_buf, &ea_buf, eff_dim as u32)?;
                        w.moe_down_exps[l][eid].gemv(ctx, &ea_buf, &eo_buf,
                                                      h as u32, eff_dim as u32)?;
                        scale_accumulate_f16(ctx, &eo_buf, &token_out,
                                             expert_weights[i], h as u32)?;
                    }

                    // Flush once per token (not per expert), then add residual
                    ctx.flush();
                    unsafe {
                        let x_ptr = (x_buf.contents() as *mut f16).add(pos * h);
                        let t_ptr = token_out.contents() as *const f16;
                        for j in 0..h {
                            *x_ptr.add(j) = f16::from_f32((*x_ptr.add(j)).to_f32() + (*t_ptr.add(j)).to_f32());
                        }
                    }
                }
            } else {
                // ── Dense FFN (batched GEMM) ──────────────────────────────────────
                let ff_dim = self.expert_ff_dim;
                gemm_f16(ctx, &x_norm2_buf, w.ffn_gate[l].f16_buf(), &gate_buf,
                         seq as u32, ff_dim as u32, h as u32)?;
                gemm_f16(ctx, &x_norm2_buf, w.ffn_up[l].f16_buf(),   &up_buf,
                         seq as u32, ff_dim as u32, h as u32)?;
                silu_mul_f16(ctx, &gate_buf, &up_buf, &act_buf, (seq * ff_dim) as u32)?;
                gemm_f16(ctx, &act_buf, w.ffn_down[l].f16_buf(), &ffn_out_buf,
                         seq as u32, h as u32, ff_dim as u32)?;
                add_f16(ctx, &x_buf, &ffn_out_buf, &x_buf, (seq * h) as u32)?;
            }
        }

        // Set the KV cache logical sequence length to cover all prefill positions.
        kv.set_seq_len(seq);

        // ── 3. Final norm + lm_head on LAST token only ────────────────────────
        // Flush so x_buf is visible, then extract last token's hidden state.
        ctx.flush();
        let last_h_buf = ctx.device.new_buffer((h * 2) as u64, opts);
        unsafe {
            std::ptr::copy_nonoverlapping(
                (x_buf.contents() as *const u8).add((seq - 1) * h * 2),
                last_h_buf.contents() as *mut u8,
                h * 2,
            );
        }

        let x_final_buf = ctx.device.new_buffer((h * 2) as u64, opts);
        rms_norm_f16(ctx, &last_h_buf, &x_final_buf,
                     &self.weights.output_norm, cfg.rms_norm_eps, h as u32, 1)?;

        let logit_buf = ctx.device.new_buffer((cfg.vocab_size * 2) as u64, opts);
        gemv_f16(ctx, self.weights.lm_head.f16_buf(), &x_final_buf, &logit_buf,
                 cfg.vocab_size as u32, h as u32)?;

        ctx.flush();
        Ok(download_f16_buf(&logit_buf, cfg.vocab_size))
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn download_f16_buf(buf: &metal::Buffer, n: usize) -> Vec<f16> {
    let ptr = buf.contents() as *const f16;
    (0..n).map(|i| unsafe { *ptr.add(i) }).collect()
}
