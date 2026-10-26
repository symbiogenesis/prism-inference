//! Reviewed C ABI from the immutable Prism source lock. Native objects stay opaque.

use super::protocol::{Request, require};
use std::ffi::{c_char, c_int, c_void};
use std::ptr::{self, NonNull};

const GGML_TYPE_F32: c_int = 0;
const GGML_TYPE_PQ2_0: c_int = 142;
const GGML_TYPE_PTQ1_0: c_int = 143;

#[repr(C)]
struct InitParams {
    mem_size: usize,
    mem_buffer: *mut c_void,
    no_alloc: bool,
}

type ToFloat = unsafe extern "C" fn(*const c_void, *mut f32, i64);

#[repr(C)]
struct TypeTraits {
    name: *const c_char,
    block_size: i64,
    interleave: i64,
    type_size: usize,
    quantized: u8,
    to_float: Option<ToFloat>,
    from_float: *const c_void,
}

const _: () = assert!(size_of::<TypeTraits>() == 56);
const _: () = assert!(std::mem::offset_of!(TypeTraits, to_float) == 40);

unsafe extern "C" {
    fn ggml_init(params: InitParams) -> *mut c_void;
    fn ggml_free(context: *mut c_void);
    fn ggml_new_tensor_2d(context: *mut c_void, kind: c_int, k: i64, rows: i64) -> *mut c_void;
    fn ggml_nbytes(tensor: *const c_void) -> usize;
    fn ggml_get_data(tensor: *const c_void) -> *mut c_void;
    fn ggml_mul_mat(
        context: *mut c_void,
        weights: *mut c_void,
        activations: *mut c_void,
    ) -> *mut c_void;
    fn ggml_new_graph(context: *mut c_void) -> *mut c_void;
    fn ggml_build_forward_expand(graph: *mut c_void, tensor: *mut c_void);
    fn ggml_graph_compute_with_ctx(
        context: *mut c_void,
        graph: *mut c_void,
        threads: c_int,
    ) -> c_int;
    fn ggml_get_type_traits(kind: c_int) -> *const TypeTraits;
}

struct Context(NonNull<c_void>);

impl Context {
    fn new() -> Result<Self, &'static str> {
        // SAFETY: InitParams matches the pinned public C layout. ggml owns the
        // 128 MiB arena; no externally owned buffer or tensor storage is passed.
        let pointer = unsafe {
            ggml_init(InitParams {
                mem_size: 128 * 1024 * 1024,
                mem_buffer: ptr::null_mut(),
                no_alloc: false,
            })
        };
        NonNull::new(pointer).map(Self).ok_or("ggml_init failed")
    }

    fn tensor(&self, kind: c_int, k: u32, rows: u32) -> Result<NonNull<c_void>, &'static str> {
        // SAFETY: The live context owns the tensor and its data. Shape and type
        // have been validated against the bounded protocol before execution.
        let pointer =
            unsafe { ggml_new_tensor_2d(self.0.as_ptr(), kind, i64::from(k), i64::from(rows)) };
        NonNull::new(pointer).ok_or("ggml tensor allocation failed")
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        // SAFETY: This wrapper owns the context uniquely; all native operations
        // have finished and no tensor or graph outlives the context.
        unsafe { ggml_free(self.0.as_ptr()) };
    }
}

fn data(tensor: NonNull<c_void>) -> Result<NonNull<c_void>, &'static str> {
    // SAFETY: Only tensors owned by the live allocating Context reach here.
    NonNull::new(unsafe { ggml_get_data(tensor.as_ptr()) }).ok_or("ggml tensor data unavailable")
}

pub fn execute(item: &Request) -> Result<Vec<f32>, &'static str> {
    let context = Context::new()?;
    let kind = if item.operation == 1 {
        GGML_TYPE_PQ2_0
    } else {
        GGML_TYPE_PTQ1_0
    };
    let packed = context.tensor(kind, item.k, item.m)?;
    // SAFETY: packed is a live tensor in context's allocating arena.
    require(
        unsafe { ggml_nbytes(packed.as_ptr()) } == item.packed.len(),
        "production packed layout mismatch",
    )?;
    let packed_data = data(packed)?;
    // SAFETY: The size was checked against the native tensor allocation. The
    // arena and the immutable Rust input are disjoint and live for this copy.
    unsafe {
        ptr::copy_nonoverlapping(
            item.packed.as_ptr(),
            packed_data.as_ptr().cast::<u8>(),
            item.packed.len(),
        );
    };
    let mut result = vec![0.0; item.output_count() as usize];
    if item.operation == 1 {
        let activations = context.tensor(GGML_TYPE_F32, item.k, item.n)?;
        let activation_data = data(activations)?;
        // SAFETY: The validated K*N f32 payload matches the allocated tensor;
        // ggml's arena is aligned for f32 and disjoint from the Rust payload.
        unsafe {
            ptr::copy_nonoverlapping(
                item.activations.as_ptr(),
                activation_data.as_ptr().cast::<f32>(),
                item.activations.len(),
            );
        };
        // SAFETY: Both live tensors have the validated compatible K dimension.
        let product = NonNull::new(unsafe {
            ggml_mul_mat(context.0.as_ptr(), packed.as_ptr(), activations.as_ptr())
        })
        .ok_or("ggml tensor allocation failed")?;
        // SAFETY: The live context owns the graph metadata allocation.
        let graph = NonNull::new(unsafe { ggml_new_graph(context.0.as_ptr()) })
            .ok_or("ggml graph allocation failed")?;
        // SAFETY: The graph and product belong to the same live context.
        unsafe { ggml_build_forward_expand(graph.as_ptr(), product.as_ptr()) };
        // SAFETY: The complete compatible graph and all tensor data remain
        // allocated throughout this synchronous four-thread production call.
        require(
            unsafe { ggml_graph_compute_with_ctx(context.0.as_ptr(), graph.as_ptr(), 4) } == 0,
            "ggml graph execution failed",
        )?;
        // SAFETY: product is a live allocated output tensor.
        require(
            unsafe { ggml_nbytes(product.as_ptr()) } == result.len() * size_of::<f32>(),
            "production output layout mismatch",
        )?;
        let product_data = data(product)?;
        // SAFETY: The checked native f32 result and initialized Rust result have
        // equal sizes, valid alignment, separate storage and live allocations.
        unsafe {
            ptr::copy_nonoverlapping(
                product_data.as_ptr().cast::<f32>(),
                result.as_mut_ptr(),
                result.len(),
            );
        };
    } else {
        // SAFETY: The immutable registered PTQ1 type ID is pinned by source.lock.
        let traits = NonNull::new(unsafe { ggml_get_type_traits(GGML_TYPE_PTQ1_0) }.cast_mut())
            .ok_or("production PTQ1 decoder unavailable")?;
        // SAFETY: The pinned API returns a live static TypeTraits object with
        // the reviewed C layout and to_float offset asserted above.
        let decoder = unsafe { traits.as_ref() }
            .to_float
            .ok_or("production PTQ1 decoder unavailable")?;
        // SAFETY: ggml owns the aligned packed allocation; validated whole
        // blocks cover exactly the initialized output's 128*M float elements.
        unsafe {
            decoder(
                packed_data.as_ptr(),
                result.as_mut_ptr(),
                i64::from(item.output_count()),
            );
        };
    }
    Ok(result)
}
