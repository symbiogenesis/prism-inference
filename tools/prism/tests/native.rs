//! Real pinned ggml allocator and CPU arithmetic regression for the Rust C ABI.
//!
//! The public `ggml_init_params` and `gguf_init_params` are the only native
//! layouts reproduced here. All ggml and GGUF objects stay opaque. The Rust
//! loader extracts real tensor metadata, reads a synthetic GGUF file, creates
//! canonical tensors/views, and populates the actual backend allocation.

#![allow(
    unsafe_code,
    reason = "These tests exercise the pinned native ggml ABI through its documented safety contracts"
)]
#![allow(
    clippy::expect_used,
    reason = "Test setup failures need an explanatory assertion"
)]

use bonsai_prism::loader::{
    NativeLoader, bonsai_prism_loader_error, bonsai_prism_loader_free, bonsai_prism_loader_load,
    bonsai_prism_loader_make, bonsai_prism_loader_new, bonsai_prism_loader_shared_bytes,
    bonsai_prism_loader_shared_tensors,
};
use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::fs::{self, OpenOptions};
use std::path::PathBuf;
use std::ptr::{self, NonNull};
use std::time::{SystemTime, UNIX_EPOCH};

const GGML_TYPE_F32: c_int = 0;
const GGML_TYPE_F16: c_int = 1;
const GGML_STATUS_SUCCESS: c_int = 0;
const B_ROWS: i64 = 5120;
const B_BYTES: usize = 20_480;
const EXPECTED_SAVING: usize = 2_621_440;

#[repr(C)]
struct InitParams {
    mem_size: usize,
    mem_buffer: *mut c_void,
    no_alloc: bool,
}

#[repr(C)]
struct GgufInitParams {
    no_alloc: bool,
    ctx: *mut *mut c_void,
}

// Signatures and enum values are from the source.lock.json-pinned ggml headers.
unsafe extern "C" {
    fn ggml_init(params: InitParams) -> *mut c_void;
    fn ggml_free(context: *mut c_void);
    fn ggml_tensor_overhead() -> usize;
    fn ggml_graph_overhead_custom(size: usize, grads: bool) -> usize;
    fn ggml_new_tensor_2d(context: *mut c_void, kind: c_int, ne0: i64, ne1: i64) -> *mut c_void;
    fn ggml_view_2d(
        context: *mut c_void,
        source: *mut c_void,
        ne0: i64,
        ne1: i64,
        nb1: usize,
        offset: usize,
    ) -> *mut c_void;
    fn ggml_mul_mat(context: *mut c_void, a: *mut c_void, b: *mut c_void) -> *mut c_void;
    fn ggml_is_contiguous(tensor: *const c_void) -> bool;
    fn ggml_are_same_shape(left: *const c_void, right: *const c_void) -> bool;
    fn ggml_are_same_stride(left: *const c_void, right: *const c_void) -> bool;
    fn ggml_nbytes(tensor: *const c_void) -> usize;
    fn ggml_get_data(tensor: *const c_void) -> *mut c_void;
    fn ggml_set_name(tensor: *mut c_void, name: *const c_char) -> *mut c_void;
    fn ggml_get_name(tensor: *const c_void) -> *const c_char;
    fn ggml_new_graph_custom(context: *mut c_void, size: usize, grads: bool) -> *mut c_void;
    fn ggml_build_forward_expand(graph: *mut c_void, tensor: *mut c_void);
    fn ggml_backend_cpu_buffer_type() -> *mut c_void;
    fn ggml_backend_alloc_ctx_tensors_from_buft(
        context: *mut c_void,
        buffer_type: *mut c_void,
    ) -> *mut c_void;
    fn ggml_backend_buffer_get_size(buffer: *mut c_void) -> usize;
    fn ggml_backend_buffer_free(buffer: *mut c_void);
    fn ggml_backend_tensor_set(
        tensor: *mut c_void,
        data: *const c_void,
        offset: usize,
        size: usize,
    );
    fn ggml_backend_tensor_get(
        tensor: *const c_void,
        data: *mut c_void,
        offset: usize,
        size: usize,
    );
    fn ggml_backend_cpu_init() -> *mut c_void;
    fn ggml_backend_cpu_set_n_threads(backend: *mut c_void, threads: c_int);
    fn ggml_backend_graph_compute(backend: *mut c_void, graph: *mut c_void) -> c_int;
    fn ggml_backend_free(backend: *mut c_void);
    fn gguf_init_empty() -> *mut c_void;
    fn gguf_add_tensor(context: *mut c_void, tensor: *const c_void);
    fn gguf_get_meta_size(context: *const c_void) -> usize;
    fn gguf_get_meta_data(context: *const c_void, data: *mut c_void);
    fn gguf_init_from_file(path: *const c_char, params: GgufInitParams) -> *mut c_void;
    fn gguf_get_data_offset(context: *const c_void) -> usize;
    fn gguf_get_tensor_offset(context: *const c_void, tensor: i64) -> usize;
    fn gguf_free(context: *mut c_void);
}

struct Context(NonNull<c_void>);

impl Context {
    fn new() -> Self {
        // SAFETY: This metadata query has no pointer arguments or side effects.
        let tensor_overhead = unsafe { ggml_tensor_overhead() };
        // SAFETY: The fixed graph size is valid for the pinned engine.
        let graph_overhead = unsafe { ggml_graph_overhead_custom(64, false) };
        // SAFETY: Params matches the pinned public C layout. ggml owns the arena
        // allocation; no_alloc makes each tensor's data await backend allocation.
        let value = unsafe {
            ggml_init(InitParams {
                mem_size: 512 * tensor_overhead + graph_overhead,
                mem_buffer: ptr::null_mut(),
                no_alloc: true,
            })
        };
        Self(NonNull::new(value).expect("ggml context allocation failed"))
    }

    const fn as_ptr(&self) -> *mut c_void {
        self.0.as_ptr()
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        // SAFETY: This uniquely owned context is freed once, after its backend
        // buffers and Rust loader. No tensor pointer survives the test.
        unsafe { ggml_free(self.as_ptr()) };
    }
}

struct Buffer(NonNull<c_void>);

impl Drop for Buffer {
    fn drop(&mut self) {
        // SAFETY: The backend returned this uniquely owned live buffer.
        unsafe { ggml_backend_buffer_free(self.0.as_ptr()) };
    }
}

struct Backend(NonNull<c_void>);

impl Drop for Backend {
    fn drop(&mut self) {
        // SAFETY: This uniquely owned backend is no longer executing a graph.
        unsafe { ggml_backend_free(self.0.as_ptr()) };
    }
}

struct LoaderHandle(NonNull<NativeLoader>);

impl LoaderHandle {
    fn new(fixture: &GgufFixture) -> Self {
        // SAFETY: The UTF-8 path is a live NUL-terminated C string. The loader
        // owns its opened file, so the borrowed path need only outlive this call.
        let value = unsafe { bonsai_prism_loader_new(fixture.path_c_string().as_ptr()) };
        Self(NonNull::new(value).expect("Rust loader construction failed"))
    }

    const fn as_ptr(&self) -> *mut NativeLoader {
        self.0.as_ptr()
    }

    fn assert_error(&self) {
        // SAFETY: The live loader returns a NUL-terminated error string valid
        // until its next mutable operation. No operation occurs during access.
        let error = unsafe { bonsai_prism_loader_error(self.as_ptr()) };
        assert!(!error.is_null());
        // SAFETY: The checked nonnull pointer names the live loader's unchanged
        // NUL-terminated error storage.
        assert_ne!(unsafe { CStr::from_ptr(error) }.to_bytes(), b"");
    }
}

impl Drop for LoaderHandle {
    fn drop(&mut self) {
        // SAFETY: This uniquely owned handle was returned by loader_new. Its
        // operations have completed, and its tensor owners remain alive.
        unsafe { bonsai_prism_loader_free(self.as_ptr()) };
    }
}

struct GgufContext(NonNull<c_void>);

impl Drop for GgufContext {
    fn drop(&mut self) {
        // SAFETY: This uniquely owned metadata context came from gguf_init_*.
        unsafe { gguf_free(self.0.as_ptr()) };
    }
}

struct GgufFixture {
    metadata: GgufContext,
    path: PathBuf,
    directory: PathBuf,
}

impl GgufFixture {
    /// Writes native GGUF metadata followed by caller-provided aligned payloads.
    /// Empty payloads may intentionally omit tensor data for malformed fixtures.
    ///
    /// # Safety
    /// Each tensor must be a live named ggml tensor for the entire call.
    unsafe fn new(tensors: &[*mut c_void], payloads: &[&[u8]]) -> Self {
        assert_eq!(tensors.len(), payloads.len());
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock predates the Unix epoch")
            .as_nanos();
        let directory = std::env::temp_dir().join(format!(
            "SwiftBonsai-PrismRustNative-{}-{stamp}",
            std::process::id(),
        ));
        assert!(directory.is_absolute());
        fs::create_dir(&directory).expect("cannot create external native test lane");
        let path = directory.join("synthetic-adapter.gguf");
        let mut contents = {
            // SAFETY: Creating an empty metadata context has no pointer inputs.
            let writer = GgufContext(
                NonNull::new(unsafe { gguf_init_empty() }).expect("GGUF writer failed"),
            );
            for &tensor in tensors {
                // SAFETY: The writer is live and the caller guarantees each
                // tensor remains live while its named metadata is copied.
                unsafe { gguf_add_tensor(writer.0.as_ptr(), tensor) };
            }
            // SAFETY: The writer owns complete initialized tensor metadata.
            let size = unsafe { gguf_get_meta_size(writer.0.as_ptr()) };
            let mut metadata = vec![0_u8; size];
            // SAFETY: The writable buffer has exactly the size returned for
            // this unchanged live writer and remains live during serialization.
            unsafe { gguf_get_meta_data(writer.0.as_ptr(), metadata.as_mut_ptr().cast()) };
            metadata
        };
        for payload in payloads {
            contents.extend_from_slice(payload);
            contents.resize(contents.len().next_multiple_of(32), 0);
        }
        fs::write(&path, contents).expect("cannot write synthetic GGUF");
        let path_c_string = CString::new(path.to_str().expect("test path is not UTF-8"))
            .expect("test path contains NUL");
        // SAFETY: The path C string is live, and null ctx requests only native
        // metadata parsing. The returned context owns all parsed descriptions.
        let metadata = unsafe {
            GgufContext(
                NonNull::new(gguf_init_from_file(
                    path_c_string.as_ptr(),
                    GgufInitParams {
                        no_alloc: true,
                        ctx: ptr::null_mut(),
                    },
                ))
                .expect("native parser rejected synthetic GGUF metadata"),
            )
        };
        Self {
            metadata,
            path,
            directory,
        }
    }

    const fn metadata_ptr(&self) -> *mut c_void {
        self.metadata.0.as_ptr()
    }

    fn path_c_string(&self) -> CString {
        CString::new(self.path.to_str().expect("test path is not UTF-8"))
            .expect("test path contains NUL")
    }
}

impl Drop for GgufFixture {
    fn drop(&mut self) {
        // LoaderHandle is declared after the fixture and closes its file first.
        // Preserve a failed fixture as evidence; successful fixtures are private
        // disposable files within the external directory allocated above.
        if !std::thread::panicking() {
            assert!(self.path.starts_with(&self.directory));
            fs::remove_file(&self.path).expect("cannot remove private synthetic GGUF");
            fs::remove_dir(&self.directory).expect("cannot remove private fixture directory");
        }
    }
}

#[test]
#[allow(
    clippy::too_many_lines,
    reason = "One fixture keeps the native object lifetimes and validation together"
)]
fn native_storage_and_math() {
    let shared = Context::new();
    let copied = Context::new();
    let bytes: Vec<u8> = (0_i16..5120)
        .flat_map(|index| (f32::from(index % 17 - 8) / 16.0).to_ne_bytes())
        .collect();
    let mut aliases = Vec::new();
    let mut originals = Vec::new();

    {
        // SAFETY: The engine's CPU buffer identity query has no pointer inputs.
        let cpu = unsafe { ggml_backend_cpu_buffer_type() };
        assert!(!cpu.is_null(), "CPU buffer type unavailable");
        for index in 0..129 {
            // SAFETY: The context is live and exclusively accessed; dimensions
            // and type describe the fixed bounded fixture tensor.
            let original = unsafe { ggml_new_tensor_2d(copied.as_ptr(), GGML_TYPE_F32, 1, B_ROWS) };
            assert!(!original.is_null(), "source tensor allocation failed");
            // SAFETY: The successful allocation is a live initialized tensor.
            assert_eq!(unsafe { ggml_nbytes(original) }, B_BYTES);
            // SAFETY: The successful allocation is a live initialized tensor.
            assert!(unsafe { ggml_is_contiguous(original) });
            let source_name =
                CString::new(format!("b-{index}.lora_b")).expect("test name contains NUL");
            // SAFETY: The live tensor is exclusively accessed and ggml copies
            // the live NUL-terminated name during this call.
            unsafe { ggml_set_name(original, source_name.as_ptr()) };
            originals.push(original);
        }
        let payloads = vec![bytes.as_slice(); originals.len()];
        // SAFETY: All originals are live named tensors in the copied context.
        let fixture = unsafe { GgufFixture::new(&originals, &payloads) };
        let loader = LoaderHandle::new(&fixture);
        for (index, &original) in originals.iter().enumerate() {
            // SAFETY: The loader, descriptor and source are live and disjoint;
            // the exclusive destination context outlives the loader and views.
            let alias = unsafe {
                bonsai_prism_loader_make(
                    loader.as_ptr(),
                    fixture.metadata_ptr(),
                    shared.as_ptr(),
                    original,
                    cpu,
                )
            };
            assert!(!alias.is_null(), "view allocation failed");
            let name = CString::new(format!("b-{index}")).expect("test name contains NUL");
            // SAFETY: The returned live tensor is exclusively accessed and
            // ggml copies the live NUL-terminated name during this call.
            unsafe { ggml_set_name(alias, name.as_ptr()) };
            aliases.push(alias);
        }
        // SAFETY: The shared context is live and exclusively accessed, and the
        // fixed dimensions/type describe a bounded tensor.
        let rank = unsafe { ggml_new_tensor_2d(shared.as_ptr(), GGML_TYPE_F32, 1, 2) };
        // SAFETY: The copied context is live and exclusively accessed, and the
        // fixed dimensions/type describe a bounded tensor.
        let copy_rank = unsafe { ggml_new_tensor_2d(copied.as_ptr(), GGML_TYPE_F32, 1, 2) };
        assert!(!rank.is_null() && !copy_rank.is_null());
        // SAFETY: Both initialized operands belong to the exclusive shared
        // context and their inner dimensions agree.
        let actual = unsafe { ggml_mul_mat(shared.as_ptr(), aliases[128], rank) };
        // SAFETY: Both initialized operands belong to the exclusive copied
        // context and their inner dimensions agree.
        let expected = unsafe { ggml_mul_mat(copied.as_ptr(), originals[128], copy_rank) };
        assert!(!actual.is_null() && !expected.is_null());
        // SAFETY: The context and CPU buffer type are live; this uniquely owned
        // buffer allocates storage for its initialized tensors before use.
        let shared_allocation =
            unsafe { ggml_backend_alloc_ctx_tensors_from_buft(shared.as_ptr(), cpu) };
        let shared_buffer =
            Buffer(NonNull::new(shared_allocation).expect("shared backend allocation failed"));
        // SAFETY: The context and CPU buffer type are live; this uniquely owned
        // buffer allocates storage for its initialized tensors before use.
        let copied_allocation =
            unsafe { ggml_backend_alloc_ctx_tensors_from_buft(copied.as_ptr(), cpu) };
        let copied_buffer =
            Buffer(NonNull::new(copied_allocation).expect("copied backend allocation failed"));
        assert_eq!(
            // SAFETY: The loader is live and no mutable operation occurs during the query.
            unsafe { bonsai_prism_loader_shared_tensors(loader.as_ptr()) },
            128
        );
        assert_eq!(
            // SAFETY: The loader is live and no mutable operation occurs during the query.
            unsafe { bonsai_prism_loader_shared_bytes(loader.as_ptr()) },
            EXPECTED_SAVING
        );
        // SAFETY: The copied buffer remains live under its unique RAII owner.
        let copied_size = unsafe { ggml_backend_buffer_get_size(copied_buffer.0.as_ptr()) };
        // SAFETY: The shared buffer remains live under its unique RAII owner.
        let shared_size = unsafe { ggml_backend_buffer_get_size(shared_buffer.0.as_ptr()) };
        assert_eq!(
            copied_size.checked_sub(shared_size),
            Some(EXPECTED_SAVING),
            "the real allocator must eliminate exactly 2.5 MiB",
        );
        // SAFETY: The first alias is live with backend storage allocated above.
        let owner_data = unsafe { ggml_get_data(aliases[0]) };
        assert!(!owner_data.is_null());
        for (index, (&alias, &original)) in aliases.iter().zip(&originals).enumerate() {
            // SAFETY: The alias is live with backend storage allocated above.
            assert_eq!(unsafe { ggml_get_data(alias) }, owner_data);
            // SAFETY: The live tensor owns initialized NUL-terminated name storage.
            let name = unsafe { ggml_get_name(alias) };
            assert_eq!(
                // SAFETY: The tensor remains live and no operation mutates its name.
                unsafe { CStr::from_ptr(name) }.to_bytes(),
                format!("b-{index}").as_bytes(),
            );
            // SAFETY: The alias and original are live initialized tensors.
            assert!(unsafe { ggml_are_same_shape(alias, original) });
            // SAFETY: The alias and original are live initialized tensors.
            assert!(unsafe { ggml_are_same_stride(alias, original) });
            // SAFETY: The loader and descriptor are live; both tensors have
            // matching sizes and exclusively accessed allocated storage.
            assert!(unsafe {
                bonsai_prism_loader_load(loader.as_ptr(), fixture.metadata_ptr(), original, alias)
            });
            // SAFETY: The original has allocated writable storage of B_BYTES,
            // exactly matching the readable payload that remains live here.
            unsafe { ggml_backend_tensor_set(original, bytes.as_ptr().cast(), 0, bytes.len()) };
        }
        let mut owner_payload = vec![0_u8; B_BYTES];
        // SAFETY: The allocated owner and writable payload buffer both have
        // B_BYTES and remain live throughout the read.
        unsafe {
            ggml_backend_tensor_get(
                aliases[0],
                owner_payload.as_mut_ptr().cast(),
                0,
                owner_payload.len(),
            );
        }
        assert_eq!(
            owner_payload, bytes,
            "Rust loader populated incorrect owner bytes"
        );
        let factors: Vec<u8> = [0.375_f32, -0.625_f32]
            .into_iter()
            .flat_map(f32::to_ne_bytes)
            .collect();
        // SAFETY: The allocated rank tensor and readable factors each have two
        // f32 elements and remain live during this exclusive write.
        unsafe { ggml_backend_tensor_set(rank, factors.as_ptr().cast(), 0, factors.len()) };
        // SAFETY: The allocated copied rank tensor has the same size as the
        // readable factors and is exclusively accessed during this write.
        unsafe { ggml_backend_tensor_set(copy_rank, factors.as_ptr().cast(), 0, factors.len()) };
        // SAFETY: CPU backend construction has no borrowed pointer inputs.
        let backend =
            Backend(NonNull::new(unsafe { ggml_backend_cpu_init() }).expect("CPU init failed"));
        // SAFETY: The live uniquely owned backend is idle, and one thread is valid.
        unsafe { ggml_backend_cpu_set_n_threads(backend.0.as_ptr(), 1) };
        // SAFETY: The live exclusive shared context has arena capacity for the graph.
        let actual_graph = unsafe { ggml_new_graph_custom(shared.as_ptr(), 64, false) };
        // SAFETY: The live exclusive copied context has arena capacity for the graph.
        let expected_graph = unsafe { ggml_new_graph_custom(copied.as_ptr(), 64, false) };
        assert!(!actual_graph.is_null() && !expected_graph.is_null());
        // SAFETY: The live graph and initialized operation belong to the shared
        // context and the graph has sufficient capacity for this one operation.
        unsafe { ggml_build_forward_expand(actual_graph, actual) };
        // SAFETY: The live graph and initialized operation belong to the copied
        // context and the graph has sufficient capacity for this one operation.
        unsafe { ggml_build_forward_expand(expected_graph, expected) };
        assert_eq!(
            // SAFETY: The exclusive CPU backend and graph are live; all graph
            // tensors have allocated storage and initialized input bytes.
            unsafe { ggml_backend_graph_compute(backend.0.as_ptr(), actual_graph) },
            GGML_STATUS_SUCCESS,
        );
        assert_eq!(
            // SAFETY: The exclusive CPU backend and graph are live; all graph
            // tensors have allocated storage and initialized input bytes.
            unsafe { ggml_backend_graph_compute(backend.0.as_ptr(), expected_graph) },
            GGML_STATUS_SUCCESS,
        );
        // SAFETY: The actual output is a live initialized tensor.
        let actual_size = unsafe { ggml_nbytes(actual) };
        // SAFETY: The expected output is a live initialized tensor.
        let expected_size = unsafe { ggml_nbytes(expected) };
        assert_eq!(actual_size, expected_size);
        let mut actual_bytes = vec![0_u8; actual_size];
        let mut expected_bytes = vec![0_u8; expected_size];
        // SAFETY: The live allocated output has exactly the writable buffer's
        // size, and graph computation completed before this read.
        unsafe {
            ggml_backend_tensor_get(
                actual,
                actual_bytes.as_mut_ptr().cast(),
                0,
                actual_bytes.len(),
            );
        }
        // SAFETY: The live allocated output has exactly the writable buffer's
        // size, and graph computation completed before this read.
        unsafe {
            ggml_backend_tensor_get(
                expected,
                expected_bytes.as_mut_ptr().cast(),
                0,
                expected_bytes.len(),
            );
        }
        assert_eq!(
            actual_bytes, expected_bytes,
            "shared B changed MUL_MAT output bytes"
        );
    }
}

#[test]
#[allow(
    clippy::too_many_lines,
    reason = "One fixture documents fallback allocations and malformed native descriptors together"
)]
fn native_loader_rejects_invalid_reads_and_keeps_fallbacks() {
    let source_context = Context::new();
    let destination_context = Context::new();
    let f16_bytes: Vec<u8> = (0..5120).flat_map(|_| 0x3c00_u16.to_ne_bytes()).collect();
    let f32_bytes = vec![0_u8; B_BYTES];

    {
        // SAFETY: The CPU buffer identity query has no borrowed pointer inputs.
        let cpu = unsafe { ggml_backend_cpu_buffer_type() };
        // SAFETY: The live exclusive source context has capacity for this
        // bounded tensor with valid F16 dimensions.
        let f16 = unsafe { ggml_new_tensor_2d(source_context.as_ptr(), GGML_TYPE_F16, 1, B_ROWS) };
        // SAFETY: The live exclusive source context has capacity for this
        // bounded tensor with valid F32 dimensions.
        let wide = unsafe { ggml_new_tensor_2d(source_context.as_ptr(), GGML_TYPE_F32, 2, B_ROWS) };
        assert!(!wide.is_null());
        // SAFETY: The source tensor is live; the view dimensions, stride and
        // zero offset remain within its initialized shape and storage extent.
        let strided = unsafe { ggml_view_2d(source_context.as_ptr(), wide, 1, B_ROWS, 8, 0) };
        // SAFETY: The live exclusive source context has capacity for this
        // bounded tensor with valid F32 dimensions.
        let ordinary =
            unsafe { ggml_new_tensor_2d(source_context.as_ptr(), GGML_TYPE_F32, 1, B_ROWS) };
        assert!(!f16.is_null() && !strided.is_null() && !ordinary.is_null());
        // SAFETY: The successful view allocation is a live initialized tensor.
        assert!(!unsafe { ggml_is_contiguous(strided) });
        // SAFETY: The live tensor is exclusive and the name is a static C string
        // that ggml copies before returning.
        unsafe { ggml_set_name(f16, c"f16.lora_b".as_ptr()) };
        // SAFETY: The live view is exclusive and the name is a static C string
        // that ggml copies before returning.
        unsafe { ggml_set_name(strided, c"strided.lora_b".as_ptr()) };
        // SAFETY: The live tensor is exclusive and the name is a static C string
        // that ggml copies before returning.
        unsafe { ggml_set_name(ordinary, c"ordinary.lora_b".as_ptr()) };
        // SAFETY: The strided view is live with initialized metadata.
        let strided_bytes = vec![0_u8; unsafe { ggml_nbytes(strided) }];
        // SAFETY: Both named source tensors remain live in source_context.
        let fallback_fixture =
            unsafe { GgufFixture::new(&[f16, strided], &[&f16_bytes, &strided_bytes]) };
        let fallback_loader = LoaderHandle::new(&fallback_fixture);
        // SAFETY: All native objects are live and disjoint; destination_context
        // is exclusively accessed and outlives the loader and returned tensors.
        let first_f16 = unsafe {
            bonsai_prism_loader_make(
                fallback_loader.as_ptr(),
                fallback_fixture.metadata_ptr(),
                destination_context.as_ptr(),
                f16,
                cpu,
            )
        };
        // SAFETY: The same live objects remain disjoint and the destination
        // context remains exclusively accessed with sufficient lifetime.
        let second_f16 = unsafe {
            bonsai_prism_loader_make(
                fallback_loader.as_ptr(),
                fallback_fixture.metadata_ptr(),
                destination_context.as_ptr(),
                f16,
                cpu,
            )
        };
        // SAFETY: All native objects are live and disjoint; the valid strided
        // view remains live and the destination context is exclusively accessed.
        let first_strided = unsafe {
            bonsai_prism_loader_make(
                fallback_loader.as_ptr(),
                fallback_fixture.metadata_ptr(),
                destination_context.as_ptr(),
                strided,
                cpu,
            )
        };
        // SAFETY: The same live objects remain disjoint and the destination
        // context remains exclusively accessed with sufficient lifetime.
        let second_strided = unsafe {
            bonsai_prism_loader_make(
                fallback_loader.as_ptr(),
                fallback_fixture.metadata_ptr(),
                destination_context.as_ptr(),
                strided,
                cpu,
            )
        };
        // A non-CPU request must fall back before any missing payload lookup.
        // SAFETY: All required objects are live and disjoint, the destination
        // context is exclusive, and the API accepts null to disable sharing.
        let other_buffer = unsafe {
            bonsai_prism_loader_make(
                fallback_loader.as_ptr(),
                fallback_fixture.metadata_ptr(),
                destination_context.as_ptr(),
                ordinary,
                ptr::null_mut(),
            )
        };
        assert!(!first_f16.is_null() && !second_f16.is_null());
        assert!(!first_strided.is_null() && !second_strided.is_null());
        assert!(!other_buffer.is_null());
        // SAFETY: Both queried tensors are live with initialized metadata.
        assert!(unsafe { ggml_are_same_shape(f16, first_f16) });
        // SAFETY: Both queried tensors are live with initialized metadata.
        assert!(unsafe { ggml_are_same_stride(f16, first_f16) });
        // SAFETY: Both queried tensors are live with initialized metadata.
        assert!(unsafe { ggml_are_same_shape(strided, first_strided) });
        // SAFETY: The successfully returned tensor is live and initialized.
        assert!(unsafe { ggml_is_contiguous(first_strided) });
        // SAFETY: The live exclusive destination context has capacity for this
        // bounded F32 tensor, whose dimensions are valid.
        let truncated_destination =
            unsafe { ggml_new_tensor_2d(destination_context.as_ptr(), GGML_TYPE_F32, 1, B_ROWS) };
        assert!(!truncated_destination.is_null());
        // SAFETY: The context and CPU buffer type are live; allocating this
        // uniquely owned buffer provides writable storage before loading.
        let allocation =
            unsafe { ggml_backend_alloc_ctx_tensors_from_buft(destination_context.as_ptr(), cpu) };
        let _buffer = Buffer(NonNull::new(allocation).expect("fallback backend allocation failed"));
        // SAFETY: The first F16 tensor is live with allocated backend storage.
        let first_f16_data = unsafe { ggml_get_data(first_f16) };
        // SAFETY: The second F16 tensor is live with allocated backend storage.
        assert_ne!(first_f16_data, unsafe { ggml_get_data(second_f16) });
        // SAFETY: The first strided fallback is live with allocated storage.
        let first_strided_data = unsafe { ggml_get_data(first_strided) };
        // SAFETY: The second strided fallback is live with allocated storage.
        assert_ne!(first_strided_data, unsafe { ggml_get_data(second_strided) });
        assert_eq!(
            // SAFETY: The loader remains live and is not mutated during the query.
            unsafe { bonsai_prism_loader_shared_tensors(fallback_loader.as_ptr()) },
            0
        );
        assert_eq!(
            // SAFETY: The loader remains live and is not mutated during the query.
            unsafe { bonsai_prism_loader_shared_bytes(fallback_loader.as_ptr()) },
            0
        );
        // SAFETY: The loader, descriptor and source remain live and disjoint;
        // the destination is exclusive, allocated and has matching F16 size.
        assert!(unsafe {
            bonsai_prism_loader_load(
                fallback_loader.as_ptr(),
                fallback_fixture.metadata_ptr(),
                f16,
                first_f16,
            )
        });
        let mut loaded_f16 = vec![0_u8; f16_bytes.len()];
        // SAFETY: The live allocated F16 tensor and writable output buffer have
        // equal lengths and remain live throughout the read.
        unsafe {
            ggml_backend_tensor_get(
                first_f16,
                loaded_f16.as_mut_ptr().cast(),
                0,
                loaded_f16.len(),
            );
        }
        assert_eq!(loaded_f16, f16_bytes, "F16 fallback bytes changed");

        // Parse genuine metadata first, then truncate before the loader opens
        // its read-only handle (Windows denies a concurrent file truncation).
        // SAFETY: The named ordinary tensor remains live in source_context.
        let truncated_fixture = unsafe { GgufFixture::new(&[ordinary], &[&f32_bytes]) };
        let file = OpenOptions::new()
            .write(true)
            .open(&truncated_fixture.path)
            .expect("cannot open private truncation fixture");
        // SAFETY: The fixture owns a live native descriptor with initialized metadata.
        let truncated_len =
            unsafe { gguf_get_data_offset(truncated_fixture.metadata_ptr()) } + B_BYTES - 1;
        file.set_len(u64::try_from(truncated_len).expect("file length exceeds u64"))
            .expect("cannot truncate private fixture");
        drop(file);
        let truncated_loader = LoaderHandle::new(&truncated_fixture);
        assert!(
            // SAFETY: Native objects remain live and disjoint; only on-disk payload
            // bytes were removed, and the exclusive destination context is valid.
            unsafe {
                bonsai_prism_loader_make(
                    truncated_loader.as_ptr(),
                    truncated_fixture.metadata_ptr(),
                    destination_context.as_ptr(),
                    ordinary,
                    cpu,
                )
            }
            .is_null()
        );
        truncated_loader.assert_error();
        // SAFETY: Objects remain live and disjoint; the destination has matching
        // allocated writable storage, while malformed file bounds are checked.
        assert!(!unsafe {
            bonsai_prism_loader_load(
                truncated_loader.as_ptr(),
                truncated_fixture.metadata_ptr(),
                ordinary,
                truncated_destination,
            )
        });
        truncated_loader.assert_error();

        // Native GGUF metadata describes a large preceding tensor, so the B
        // offset lies well beyond this intentionally short file. No huge data
        // allocation or private native struct mutation is necessary.
        // SAFETY: The exclusive no_alloc context stores metadata only; the large
        // valid F32 dimensions do not allocate a corresponding payload.
        let prefix =
            unsafe { ggml_new_tensor_2d(source_context.as_ptr(), GGML_TYPE_F32, 1, 1_i64 << 32) };
        assert!(!prefix.is_null());
        // SAFETY: The live tensor is exclusive and ggml copies this static C string.
        unsafe { ggml_set_name(prefix, c"omitted-prefix".as_ptr()) };
        // SAFETY: Both named tensors remain live; the fixture contract permits
        // intentionally missing payloads to exercise malformed file rejection.
        let offset_fixture = unsafe { GgufFixture::new(&[prefix, ordinary], &[&[], &f32_bytes]) };
        assert!(
            // SAFETY: The live descriptor has exactly the two tensors added above,
            // so index one is valid and its offset metadata is initialized.
            unsafe { gguf_get_tensor_offset(offset_fixture.metadata_ptr(), 1) }
                > usize::try_from(
                    fs::metadata(&offset_fixture.path)
                        .expect("fixture disappeared")
                        .len()
                )
                .expect("file size exceeds usize")
        );
        let offset_loader = LoaderHandle::new(&offset_fixture);
        assert!(
            // SAFETY: All native objects remain live and disjoint with an exclusive
            // destination context; malformed on-disk bounds are checked by the loader.
            unsafe {
                bonsai_prism_loader_make(
                    offset_loader.as_ptr(),
                    offset_fixture.metadata_ptr(),
                    destination_context.as_ptr(),
                    ordinary,
                    cpu,
                )
            }
            .is_null()
        );
        offset_loader.assert_error();
        // SAFETY: All native objects remain live and disjoint with matching
        // allocated destination storage; only on-disk payload bounds are invalid.
        assert!(!unsafe {
            bonsai_prism_loader_load(
                offset_loader.as_ptr(),
                offset_fixture.metadata_ptr(),
                ordinary,
                truncated_destination,
            )
        });
        offset_loader.assert_error();
        assert_eq!(
            // SAFETY: The loader remains live and is not mutated during the query.
            unsafe { bonsai_prism_loader_shared_tensors(offset_loader.as_ptr()) },
            0
        );
        assert_eq!(
            // SAFETY: The loader remains live and is not mutated during the query.
            unsafe { bonsai_prism_loader_shared_bytes(offset_loader.as_ptr()) },
            0
        );
    }
}
