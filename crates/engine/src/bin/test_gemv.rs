use half::f16;
use metal::MTLResourceOptions;
use bare_metal_kernels::context::MetalContext;
use bare_metal_kernels::dispatch::{gemv_f16, gemv_f16w_f32in};

fn main() -> anyhow::Result<()> {
    let ctx = MetalContext::new()?;
    let opts = MTLResourceOptions::StorageModeShared;

    // Test: 4×4 identity-like matrix × [1, 2, 3, 4] = [1, 2, 3, 4]
    let m: u32 = 4;
    let k: u32 = 4;

    // Weight matrix (f16): identity
    let w_data: Vec<f16> = vec![
        // Row 0: [1, 0, 0, 0]
        f16::from_f32(1.0), f16::from_f32(0.0), f16::from_f32(0.0), f16::from_f32(0.0),
        // Row 1: [0, 1, 0, 0]
        f16::from_f32(0.0), f16::from_f32(1.0), f16::from_f32(0.0), f16::from_f32(0.0),
        // Row 2: [0, 0, 1, 0]
        f16::from_f32(0.0), f16::from_f32(0.0), f16::from_f32(1.0), f16::from_f32(0.0),
        // Row 3: [0, 0, 0, 1]
        f16::from_f32(0.0), f16::from_f32(0.0), f16::from_f32(0.0), f16::from_f32(1.0),
    ];

    let w_buf = ctx.device.new_buffer((m * k * 2) as u64, opts);
    unsafe {
        std::ptr::copy_nonoverlapping(
            w_data.as_ptr() as *const u8, w_buf.contents() as *mut u8, (m * k * 2) as usize);
    }

    // Input vector (f32): [1, 2, 3, 4]
    let x_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let x_buf = ctx.device.new_buffer((k * 4) as u64, opts);
    unsafe {
        std::ptr::copy_nonoverlapping(
            x_data.as_ptr() as *const u8, x_buf.contents() as *mut u8, (k * 4) as usize);
    }

    // Output buffer (f16)
    let y_buf = ctx.device.new_buffer((m * 2) as u64, opts);

    // Run gemv_f16w_f32in: y = W @ x (identity × [1,2,3,4] = [1,2,3,4])
    gemv_f16w_f32in(&ctx, &w_buf, &x_buf, &y_buf, m, k)?;
    ctx.flush();

    let py = y_buf.contents() as *const f16;
    let result: Vec<f32> = (0..m as usize).map(|i| unsafe { (*py.add(i)).to_f32() }).collect();
    println!("Test 1 (identity × [1,2,3,4]):");
    println!("  Expected: [1.0, 2.0, 3.0, 4.0]");
    println!("  Got:      {:?}", result);
    assert!((result[0] - 1.0).abs() < 0.01, "Mismatch at index 0");
    assert!((result[1] - 2.0).abs() < 0.01, "Mismatch at index 1");
    assert!((result[2] - 3.0).abs() < 0.01, "Mismatch at index 2");
    assert!((result[3] - 4.0).abs() < 0.01, "Mismatch at index 3");
    println!("  PASS ✓");

    // Test 2: larger matrix (32×32) with known values
    let m2: u32 = 32;
    let k2: u32 = 32;
    // W = all 1s, x = 1..32
    let w2_data: Vec<f16> = vec![f16::from_f32(1.0); (m2 * k2) as usize];
    let x2_data: Vec<f32> = (1..=k2).map(|i| i as f32).collect();
    // Expected: each row dot product = sum(1..32) = 32*33/2 = 528

    let w2_buf = ctx.device.new_buffer((m2 * k2 * 2) as u64, opts);
    let x2_buf = ctx.device.new_buffer((k2 * 4) as u64, opts);
    let y2_buf = ctx.device.new_buffer((m2 * 2) as u64, opts);
    unsafe {
        std::ptr::copy_nonoverlapping(w2_data.as_ptr() as *const u8, w2_buf.contents() as *mut u8, (m2 * k2 * 2) as usize);
        std::ptr::copy_nonoverlapping(x2_data.as_ptr() as *const u8, x2_buf.contents() as *mut u8, (k2 * 4) as usize);
    }

    gemv_f16w_f32in(&ctx, &w2_buf, &x2_buf, &y2_buf, m2, k2)?;
    ctx.flush();

    let py2 = y2_buf.contents() as *const f16;
    let result2: Vec<f32> = (0..m2 as usize).map(|i| unsafe { (*py2.add(i)).to_f32() }).collect();
    println!("\nTest 2 (all-ones 32x32 × [1..32]):");
    println!("  Expected: all 528.0");
    println!("  Got[0..4]: {:?}", &result2[0..4]);
    let err = (result2[0] - 528.0).abs();
    println!("  Error: {:.4}", err);
    assert!(err < 1.0, "Large error in test 2");
    println!("  PASS ✓");

    // Test 3: realistic size (896×896) with random-ish values
    let m3: u32 = 896;
    let k3: u32 = 896;
    let w3_data: Vec<f16> = (0..m3*k3).map(|i| {
        let v = ((i % 17) as f32 - 8.0) * 0.001;
        f16::from_f32(v)
    }).collect();
    let x3_data: Vec<f32> = (0..k3).map(|i| {
        ((i % 13) as f32 - 6.0) * 0.01
    }).collect();

    let w3_buf = ctx.device.new_buffer((m3 * k3 * 2) as u64, opts);
    let x3_buf = ctx.device.new_buffer((k3 * 4) as u64, opts);
    let y3_buf = ctx.device.new_buffer((m3 * 2) as u64, opts);
    unsafe {
        std::ptr::copy_nonoverlapping(w3_data.as_ptr() as *const u8, w3_buf.contents() as *mut u8, (m3 * k3 * 2) as usize);
        std::ptr::copy_nonoverlapping(x3_data.as_ptr() as *const u8, x3_buf.contents() as *mut u8, (k3 * 4) as usize);
    }

    gemv_f16w_f32in(&ctx, &w3_buf, &x3_buf, &y3_buf, m3, k3)?;
    ctx.flush();

    // CPU reference
    let mut cpu_ref = vec![0.0f64; m3 as usize];
    for row in 0..m3 as usize {
        for col in 0..k3 as usize {
            cpu_ref[row] += w3_data[row * k3 as usize + col].to_f32() as f64 * x3_data[col] as f64;
        }
    }

    let py3 = y3_buf.contents() as *const f16;
    let result3: Vec<f32> = (0..m3 as usize).map(|i| unsafe { (*py3.add(i)).to_f32() }).collect();

    let mut max_err: f64 = 0.0;
    let mut max_err_idx = 0;
    for i in 0..m3 as usize {
        let err = (result3[i] as f64 - cpu_ref[i]).abs();
        if err > max_err {
            max_err = err;
            max_err_idx = i;
        }
    }

    println!("\nTest 3 (896x896 pseudo-random):");
    println!("  GPU[0..5]: {:?}", &result3[0..5]);
    println!("  CPU[0..5]: {:?}", &cpu_ref[0..5].iter().map(|&x| x as f32).collect::<Vec<_>>());
    println!("  Max error: {:.6} at index {}", max_err, max_err_idx);
    println!("  GPU[{}]: {}, CPU[{}]: {}", max_err_idx, result3[max_err_idx], max_err_idx, cpu_ref[max_err_idx]);
    assert!(max_err < 0.1, "Large error in 896x896 test");
    println!("  PASS ✓");

    // Test 4: Also test gemv_f16 (f16 input)
    let x3_f16: Vec<f16> = x3_data.iter().map(|&x| f16::from_f32(x)).collect();
    let x3f16_buf = ctx.device.new_buffer((k3 * 2) as u64, opts);
    let y4_buf = ctx.device.new_buffer((m3 * 2) as u64, opts);
    unsafe {
        std::ptr::copy_nonoverlapping(x3_f16.as_ptr() as *const u8, x3f16_buf.contents() as *mut u8, (k3 * 2) as usize);
    }
    gemv_f16(&ctx, &w3_buf, &x3f16_buf, &y4_buf, m3, k3)?;
    ctx.flush();

    let py4 = y4_buf.contents() as *const f16;
    let result4: Vec<f32> = (0..m3 as usize).map(|i| unsafe { (*py4.add(i)).to_f32() }).collect();

    let mut max_err4: f64 = 0.0;
    for i in 0..m3 as usize {
        let err = (result4[i] as f64 - cpu_ref[i]).abs();
        if err > max_err4 { max_err4 = err; }
    }
    println!("\nTest 4 (gemv_f16, 896x896):");
    println!("  GPU[0..5]: {:?}", &result4[0..5]);
    println!("  Max error vs CPU: {:.6}", max_err4);
    assert!(max_err4 < 0.1, "Large error in gemv_f16 test");
    println!("  PASS ✓");

    println!("\nAll GEMV kernel tests passed!");
    Ok(())
}
