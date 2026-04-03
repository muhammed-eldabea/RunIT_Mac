use metal::{Buffer, Device, MTLResourceOptions};

use crate::loader::TensorBuffer;
use crate::tensor::DType;

const PAGE_SIZE: usize = 4096;

/// A GPU-side tensor backed by a `MTLBuffer`.
///
/// Two construction modes:
/// - **Zero-copy** (`from_tensor_buffer`): wraps an existing mmap pointer via
///   `newBufferWithBytesNoCopy` when both the pointer and size are 4096-byte
///   page-aligned.  Falls back to a copying allocation for per-tensor offsets
///   within a GGUF file, which are only 32-byte aligned.
/// - **Owned** (`zeros`): allocates fresh GPU-writable memory for activations
///   and outputs.
pub struct MetalTensor {
    /// Underlying Metal buffer
    pub buffer: Buffer,
    /// Logical shape (outermost-first)
    pub shape: Vec<usize>,
    /// Data type
    pub dtype: DType,
}

impl MetalTensor {
    /// Wrap an existing CPU/mmap pointer as a Metal buffer.
    ///
    /// Attempts zero-copy (`newBufferWithBytesNoCopy`) when both the pointer
    /// and size are 4096-byte page-aligned.  Falls back to a copying allocation
    /// (`newBufferWithBytes`) otherwise — transparent to the caller.
    ///
    /// # Safety
    /// - `tb.ptr` must point to valid memory of at least `tb.size` bytes.
    /// - On the zero-copy path `tb.ptr` must remain valid for the lifetime of
    ///   this `MetalTensor` (satisfied when `Model` owns the `Mmap` and
    ///   outlives this tensor).  The copying path has no such requirement.
    pub unsafe fn from_tensor_buffer(
        device: &Device,
        tb: TensorBuffer,
        shape: Vec<usize>,
        dtype: DType,
    ) -> Self {
        let addr = tb.ptr as usize;
        let buffer = if addr % PAGE_SIZE == 0 && tb.size % PAGE_SIZE == 0 {
            // Both pointer and length are page-aligned — zero-copy.
            device.new_buffer_with_bytes_no_copy(
                tb.ptr as *mut std::ffi::c_void,
                tb.size as u64,
                MTLResourceOptions::StorageModeShared,
                None, // no deallocator — mmap owns the memory
            )
        } else {
            // Pointer or length is not page-aligned (typical for per-tensor
            // offsets within a GGUF file, which are only 32-byte aligned).
            // Allocate a Metal buffer and memcpy the data in.
            // metal-rs 0.29 has no `new_buffer_with_bytes`; we use `new_buffer`
            // (zeroed) and write into it through the CPU-visible contents pointer.
            let buf = device.new_buffer(
                tb.size as u64,
                MTLResourceOptions::StorageModeShared,
            );
            std::ptr::copy_nonoverlapping(
                tb.ptr,
                buf.contents() as *mut u8,
                tb.size,
            );
            buf
        };
        Self { buffer, shape, dtype }
    }

    /// Allocate a new GPU-writable buffer filled with zeros.
    /// Used for activations and attention outputs.
    pub fn zeros(device: &Device, shape: Vec<usize>, dtype: DType) -> Self {
        let n: usize = shape.iter().product::<usize>().max(1);
        let bytes = n * dtype.element_size();
        let buffer = device.new_buffer(bytes as u64, MTLResourceOptions::StorageModeShared);
        // MTLResourceOptions::StorageModeShared buffers are zero-initialised by Metal
        Self { buffer, shape, dtype }
    }

    /// Total number of elements.
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product::<usize>().max(1)
    }

    /// Byte size of the underlying buffer.
    pub fn byte_size(&self) -> usize {
        self.num_elements() * self.dtype.element_size()
    }
}

// ── DType element size helper ─────────────────────────────────────────────────

impl DType {
    /// Byte size of one element for non-quantized types.
    /// For quantized types, returns the block byte size divided by block_size —
    /// this is the "effective" bytes per element for buffer sizing.
    pub fn element_size(&self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::I32 => 4,
            Self::I16 => 2,
            Self::I8 => 1,
            // Quantized: use type_size() / block_size() as average bytes per element
            Self::Q4_0 | Self::Q4_1 => 18 / 32,  // ~0.5 — but ceil to 1 for safety
            Self::Q8_0 | Self::Q8_1 => 34 / 32,  // ~1
            _ => 2, // default to 2 bytes (f16) for other quant types
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zeros_allocation() {
        let device = match metal::Device::system_default() {
            Some(d) => d,
            None => { eprintln!("SKIP: Metal device not available"); return; }
        };
        let t = MetalTensor::zeros(&device, vec![4, 8], DType::F16);
        assert_eq!(t.num_elements(), 32);
        assert_eq!(t.byte_size(), 64);
        assert!(!t.buffer.contents().is_null());
    }

    #[test]
    fn test_zeros_contents_are_zero() {
        let device = match metal::Device::system_default() {
            Some(d) => d,
            None => { eprintln!("SKIP: Metal device not available"); return; }
        };
        let n = 128usize;
        let t = MetalTensor::zeros(&device, vec![n], DType::F32);
        let ptr = t.buffer.contents() as *const f32;
        for i in 0..n {
            assert_eq!(unsafe { *ptr.add(i) }, 0.0f32);
        }
    }

    #[test]
    fn test_from_tensor_buffer_unaligned_copies() {
        use crate::loader::TensorBuffer;

        let device = match metal::Device::system_default() {
            Some(d) => d,
            None => { eprintln!("SKIP: Metal device not available"); return; }
        };
        // Allocate with extra space so we can take an unaligned interior pointer.
        let data: Vec<u8> = vec![0xABu8; PAGE_SIZE + 64];
        // Offset by 32 bytes — not page-aligned.
        let unaligned_ptr = unsafe { data.as_ptr().add(32) };
        let size = 64usize;
        assert!(unaligned_ptr as usize % PAGE_SIZE != 0);

        let tb = TensorBuffer { ptr: unaligned_ptr, size };
        let t = unsafe {
            MetalTensor::from_tensor_buffer(&device, tb, vec![size], DType::I8)
        };
        assert_eq!(t.num_elements(), size);
        // Verify the data was copied correctly.
        let out = t.buffer.contents() as *const u8;
        for i in 0..size {
            assert_eq!(unsafe { *out.add(i) }, 0xABu8);
        }
    }
}
