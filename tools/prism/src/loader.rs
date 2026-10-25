//! Native ggml tensor creation and one-handle immutable `LoRA` file loading.
//!
//! The binding targets the public C API of the hash-pinned engine. Only the
//! initial scalar layout fields of `ggml_tensor` are read directly; tensor
//! creation, views, names, byte counts, and backend writes use engine functions.

use crate::{Layout, Pool};
use std::alloc::{Layout as AllocationLayout, alloc};
use std::ffi::{CStr, c_char, c_int, c_void};
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom};
use std::ptr;

#[cfg(windows)]
use std::os::windows::fs::OpenOptionsExt;

// The pinned public ggml.h starts with these fields, in this exact order.
// No field after nb is represented or accessed. GGML_TYPE_F32 is zero.
#[repr(C)]
struct TensorPrefix {
    kind: c_int,
    buffer: *mut c_void,
    dimensions: [i64; 4],
    strides: [usize; 4],
}

unsafe extern "C" {
    fn ggml_get_name(tensor: *const c_void) -> *const c_char;
    fn ggml_nbytes(tensor: *const c_void) -> usize;
    fn ggml_is_contiguous(tensor: *const c_void) -> bool;
    fn ggml_is_view(tensor: *const c_void) -> bool;
    fn ggml_dup_tensor(context: *mut c_void, source: *const c_void) -> *mut c_void;
    fn ggml_view_tensor(context: *mut c_void, source: *mut c_void) -> *mut c_void;
    fn ggml_backend_cpu_buffer_type() -> *mut c_void;
    fn ggml_backend_tensor_set(
        tensor: *mut c_void,
        bytes: *const c_void,
        offset: usize,
        size: usize,
    );
    fn gguf_find_tensor(gguf: *const c_void, name: *const c_char) -> i64;
    fn gguf_get_n_tensors(gguf: *const c_void) -> i64;
    fn gguf_get_data_offset(gguf: *const c_void) -> usize;
    fn gguf_get_tensor_offset(gguf: *const c_void, index: i64) -> usize;
}

/// Opaque, single-threaded owner of the sharing pool and one open file handle.
///
/// Windows opens permit other readers and deny writers and deletion. On other
/// platforms the caller must keep the opened file immutable for this handle's
/// lifetime. Every read checks the current length against its initial value.
/// Tensor contexts and all returned tensors must outlive this handle.
pub struct NativeLoader {
    file: File,
    file_len: u64,
    pool: Pool,
    bytes: Vec<u8>,
    error: [u8; 256],
}

impl NativeLoader {
    fn open(path: &str) -> Option<Self> {
        let mut options = OpenOptions::new();
        options.read(true);
        #[cfg(windows)]
        // FILE_SHARE_READ: deny both writable and delete-sharing handles.
        options.share_mode(1);
        let file = options.open(path).ok()?;
        let file_len = file.metadata().ok()?.len();
        Some(Self {
            file,
            file_len,
            pool: Pool::new(),
            bytes: Vec::new(),
            error: [0; 256],
        })
    }

    fn set_error(&mut self, message: &[u8]) {
        self.error.fill(0);
        let length = message.len().min(self.error.len() - 1);
        self.error[..length].copy_from_slice(&message[..length]);
    }

    const fn clear_error(&mut self) {
        self.error[0] = 0;
    }

    // SAFETY contract: gguf/source are live pinned-engine objects throughout
    // this call. No native object is mutated while reading its metadata.
    unsafe fn read_tensor(
        &mut self,
        gguf: *const c_void,
        source: *const c_void,
    ) -> Result<(), &'static [u8]> {
        // SAFETY: The caller provides a live tensor. Its name storage remains
        // valid during the descriptor call.
        let name = unsafe { ggml_get_name(source) };
        // SAFETY: The caller provides a live tensor with initialized metadata.
        let size = unsafe { ggml_nbytes(source) };
        if name.is_null() {
            return Err(b"LoRA tensor has no name");
        }
        // SAFETY: Name and descriptor validity are established above and by the caller.
        let index = unsafe { gguf_find_tensor(gguf, name) };
        // SAFETY: The caller provides a live GGUF descriptor.
        let count = unsafe { gguf_get_n_tensors(gguf) };
        if index < 0 || index >= count {
            return Err(b"LoRA tensor is missing from its GGUF descriptor");
        }
        // SAFETY: The descriptor is live and has initialized metadata.
        let data_offset = unsafe { gguf_get_data_offset(gguf) };
        // SAFETY: The descriptor is live and index is within its tensor count.
        let tensor_offset = unsafe { gguf_get_tensor_offset(gguf, index) };
        let offset = data_offset
            .checked_add(tensor_offset)
            .and_then(|value| u64::try_from(value).ok())
            .ok_or(b"LoRA tensor file offset overflows".as_slice())?;
        let size_u64 =
            u64::try_from(size).map_err(|_| b"LoRA tensor payload size overflows".as_slice())?;
        let current_len = self
            .file
            .metadata()
            .map_err(|_| b"Cannot inspect opened LoRA file".as_slice())?
            .len();
        if current_len != self.file_len {
            return Err(b"LoRA file length changed while loading");
        }
        if offset > current_len || size_u64 > current_len - offset {
            return Err(b"LoRA tensor payload lies outside the opened file bounds");
        }
        self.bytes
            .try_reserve_exact(size.saturating_sub(self.bytes.len()))
            .map_err(|_| b"Cannot allocate LoRA tensor read buffer".as_slice())?;
        self.bytes.resize(size, 0);
        self.file
            .seek(SeekFrom::Start(offset))
            .map_err(|_| b"Cannot seek within opened LoRA file".as_slice())?;
        self.file
            .read_exact(&mut self.bytes)
            .map_err(|_| b"Cannot read complete LoRA tensor payload".as_slice())
    }

    // SAFETY contract: source is a live, aligned tensor from the pinned engine;
    // context and buffer_type are destination identities that remain alive.
    unsafe fn describe(
        context: *mut c_void,
        source: *const c_void,
        buffer_type: *mut c_void,
    ) -> Layout {
        // SAFETY: repr(C) mirrors the verified initial fields of ggml_tensor;
        // source is aligned for that tensor and no shared metadata is mutated.
        let prefix = unsafe { &*source.cast::<TensorPrefix>() };
        Layout {
            context: context.addr(),
            buffer_type: buffer_type.addr(),
            // SAFETY: Public metadata queries accept the live source tensor;
            // CPU buffer identity is supplied by the pinned engine itself.
            cpu: buffer_type == unsafe { ggml_backend_cpu_buffer_type() },
            f32: prefix.kind == 0,
            // SAFETY: source is a live tensor with initialized public metadata.
            contiguous: unsafe { ggml_is_contiguous(source) },
            dimensions: prefix.dimensions,
            strides: prefix.strides,
            // SAFETY: source is a live tensor with initialized public metadata.
            byte_len: unsafe { ggml_nbytes(source) },
        }
    }

    // SAFETY contract: all pointer and lifetime requirements of loader_make.
    unsafe fn make(
        &mut self,
        gguf: *const c_void,
        context: *mut c_void,
        source: *const c_void,
        buffer_type: *mut c_void,
    ) -> Result<*mut c_void, &'static [u8]> {
        // SAFETY: The exported boundary checked nulls; the caller guarantees
        // live engine objects with the pinned layout and exclusive context.
        let layout = unsafe { Self::describe(context, source, buffer_type) };
        if !layout.eligible() {
            // SAFETY: The destination context and source metadata remain live.
            return nonnull_tensor(unsafe { ggml_dup_tensor(context, source) });
        }
        // SAFETY: The caller guarantees the descriptor and source remain live.
        unsafe { self.read_tensor(gguf, source) }?;
        if let Some(owner) = self.pool.find(&layout, &self.bytes) {
            // SAFETY: Pool identities came from ggml_dup_tensor in this context
            // and must remain alive and immutable for the loader's lifetime.
            return nonnull_tensor(unsafe {
                ggml_view_tensor(context, ptr::with_exposed_provenance_mut(owner))
            });
        }
        // SAFETY: The context is exclusively accessed and source remains live.
        let tensor = nonnull_tensor(unsafe { ggml_dup_tensor(context, source) })?;
        // SAFETY: A successful duplicate is a live tensor in context. Its own
        // metadata can normalize source strides and must be cached instead.
        let owner_layout = unsafe { Self::describe(context, tensor, buffer_type) };
        let _ = self
            .pool
            .remember(&owner_layout, &self.bytes, tensor.expose_provenance());
        Ok(tensor)
    }

    // SAFETY contract: all pointer and allocation requirements of loader_load.
    unsafe fn load(
        &mut self,
        gguf: *const c_void,
        original: *const c_void,
        destination: *mut c_void,
    ) -> Result<(), &'static [u8]> {
        // SAFETY: destination is a live initialized tensor from the engine.
        if unsafe { ggml_is_view(destination) } {
            return Ok(());
        }
        // SAFETY: The source is a live initialized tensor.
        let original_size = unsafe { ggml_nbytes(original) };
        // SAFETY: The destination is a live initialized tensor.
        let destination_size = unsafe { ggml_nbytes(destination) };
        if original_size != destination_size {
            return Err(b"LoRA source and destination payload sizes differ");
        }
        // SAFETY: The caller guarantees source and descriptor validity.
        unsafe { self.read_tensor(gguf, original) }?;
        // SAFETY: The caller supplies an allocated writable backend tensor;
        // sizes agree and the entire readable slice stays live during copying.
        unsafe {
            ggml_backend_tensor_set(destination, self.bytes.as_ptr().cast(), 0, self.bytes.len());
        }
        Ok(())
    }
}

const fn nonnull_tensor(tensor: *mut c_void) -> Result<*mut c_void, &'static [u8]> {
    if tensor.is_null() {
        Err(b"Cannot allocate native LoRA tensor")
    } else {
        Ok(tensor)
    }
}

/// Opens one immutable UTF-8 file path, returning null on invalid input or error.
///
/// # Safety
/// A nonnull path must be a readable, terminated C string for this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn bonsai_prism_loader_new(path: *const c_char) -> *mut NativeLoader {
    if path.is_null() {
        return ptr::null_mut();
    }
    // SAFETY: The caller guarantees readable terminated C string storage.
    let Ok(path) = unsafe { CStr::from_ptr(path) }.to_str() else {
        return ptr::null_mut();
    };
    let Some(loader) = NativeLoader::open(path) else {
        return ptr::null_mut();
    };
    #[allow(
        clippy::cast_ptr_alignment,
        reason = "Allocation requests handle alignment"
    )]
    // SAFETY: The allocation requests the nonzero handle size and alignment.
    let result = unsafe { alloc(AllocationLayout::new::<NativeLoader>()) }.cast::<NativeLoader>();
    if !result.is_null() {
        // SAFETY: A successful allocation has the size and alignment for the
        // handle and is initialized exactly once before publication.
        unsafe { result.write(loader) };
    }
    result
}

/// Frees the file handle, read buffer, and cache; null is accepted.
///
/// # Safety
/// A nonnull handle must come from `bonsai_prism_loader_new`, be unfreed, and
/// have no outstanding access. Each handle may be freed only once.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn bonsai_prism_loader_free(loader: *mut NativeLoader) {
    if !loader.is_null() {
        // SAFETY: The caller transfers exclusive ownership of this initialized
        // global-allocator allocation, originally returned by loader_new.
        unsafe { drop(Box::from_raw(loader)) };
    }
}

/// Duplicates a tensor or creates a same-context immutable B-vector view.
///
/// Returns null on error; `bonsai_prism_loader_error` describes that error.
///
/// # Safety
/// A nonnull loader must be live and exclusively borrowed. The gguf descriptor,
/// context, and source must be live objects from the pinned engine; context is
/// exclusively accessed and all cached tensors must remain live and immutable.
/// Source and descriptor memory must not overlap loader storage. Null required
/// pointers are rejected; a null buffer type simply disables sharing.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn bonsai_prism_loader_make(
    loader: *mut NativeLoader,
    gguf: *const c_void,
    context: *mut c_void,
    source: *const c_void,
    buffer_type: *mut c_void,
) -> *mut c_void {
    // SAFETY: The caller guarantees a valid exclusive handle when nonnull.
    let Some(loader) = (unsafe { loader.as_mut() }) else {
        return ptr::null_mut();
    };
    loader.clear_error();
    if gguf.is_null() || context.is_null() || source.is_null() {
        loader.set_error(b"Null GGUF, context, or source passed to LoRA tensor creation");
        return ptr::null_mut();
    }
    // SAFETY: Nulls are rejected above; the caller guarantees native object
    // validity, pinned layout, and required lifetimes and exclusive access.
    match unsafe { loader.make(gguf, context, source, buffer_type) } {
        Ok(tensor) => tensor,
        Err(message) => {
            loader.set_error(message);
            ptr::null_mut()
        }
    }
}

/// Loads one tensor's bytes into its allocated backend storage.
///
/// Native views are skipped because their immutable owner is loaded separately.
/// Returns false on error; `bonsai_prism_loader_error` describes that error.
///
/// # Safety
/// The loader must be live and exclusively borrowed. Descriptor and source must
/// be live pinned-engine objects. Destination must be an allocated, writable
/// pinned-engine tensor of the same byte size, with no concurrent access. All
/// native objects must be disjoint from loader storage. Nulls are rejected.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn bonsai_prism_loader_load(
    loader: *mut NativeLoader,
    gguf: *const c_void,
    original: *const c_void,
    destination: *mut c_void,
) -> bool {
    // SAFETY: The caller guarantees a valid exclusive handle when nonnull.
    let Some(loader) = (unsafe { loader.as_mut() }) else {
        return false;
    };
    loader.clear_error();
    if gguf.is_null() || original.is_null() || destination.is_null() {
        loader.set_error(b"Null GGUF, source, or destination passed to LoRA tensor loading");
        return false;
    }
    // SAFETY: Nulls are rejected above; caller guarantees object validity and
    // an allocated exclusive destination with no overlap of loader storage.
    match unsafe { loader.load(gguf, original, destination) } {
        Ok(()) => true,
        Err(message) => {
            loader.set_error(message);
            false
        }
    }
}

/// Returns a bounded error string, or an empty string after a successful call.
///
/// The returned storage belongs to the loader and changes on its next mutation.
/// Null returns a static diagnostic. No returned pointer may be freed.
///
/// # Safety
/// A nonnull loader must be live and cannot be concurrently mutated. Returned
/// storage must not be read after mutation or destruction of that loader.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn bonsai_prism_loader_error(loader: *const NativeLoader) -> *const c_char {
    // SAFETY: The caller guarantees a live shared handle when nonnull.
    unsafe { loader.as_ref() }.map_or(c"Invalid Rust LoRA loader".as_ptr(), |value| {
        value.error.as_ptr().cast()
    })
}

/// Returns successful sharing lookups, or zero for a null loader.
///
/// # Safety
/// A nonnull loader must be live and cannot be concurrently mutated.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn bonsai_prism_loader_shared_tensors(loader: *const NativeLoader) -> usize {
    // SAFETY: The caller guarantees a live shared handle when nonnull.
    unsafe { loader.as_ref() }.map_or(0, |value| value.pool.shared_tensors())
}

/// Returns saved tensor payload bytes, or zero for a null loader.
///
/// # Safety
/// A nonnull loader must be live and cannot be concurrently mutated.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn bonsai_prism_loader_shared_bytes(loader: *const NativeLoader) -> usize {
    // SAFETY: The caller guarantees a live shared handle when nonnull.
    unsafe { loader.as_ref() }.map_or(0, |value| value.pool.shared_bytes())
}
