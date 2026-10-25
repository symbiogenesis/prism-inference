//! The released 73bfe6cc Windows ARM64 ggml ABI, loaded only in the owned child.

use std::ffi::{CStr, c_char, c_void};
use std::os::windows::ffi::OsStrExt as _;
use std::path::Path;
use std::ptr::NonNull;
use windows_sys::Win32::Foundation::{FreeLibrary, HMODULE};
use windows_sys::Win32::System::LibraryLoader::{
    GetProcAddress, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR, LOAD_LIBRARY_SEARCH_SYSTEM32, LoadLibraryExW,
};

type ToFloat = unsafe extern "C" fn(*const c_void, *mut f32, i64);
type GetTraits = unsafe extern "C" fn(i32) -> *const TypeTraits;

// Pinned ggml/include/ggml.h: the one-byte C bool and seven padding bytes
// place to_float at byte 40. Use u8 instead of bool to validate foreign data.
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

// Native block_pq2_0 and block_ptq1_0 contain a uint16_t ggml_half. Rust's
// byte arrays (including the frozen corpus) have no corresponding alignment
// guarantee, so establish it with owned copies before passing them to C.
#[repr(C, align(2))]
struct NativeBlock<const N: usize>([u8; N]);

const _: () = assert!(align_of::<NativeBlock<34>>() == align_of::<u16>());
const _: () = assert!(size_of::<NativeBlock<34>>() == 34);
const _: () = assert!(size_of::<NativeBlock<28>>() == 28);

struct Library(HMODULE);

impl Drop for Library {
    fn drop(&mut self) {
        // SAFETY: this wrapper exclusively owns the successful LoadLibraryExW
        // reference; every function call has completed before it is released.
        unsafe { FreeLibrary(self.0) };
    }
}

fn decoder(
    get_traits: GetTraits,
    type_id: i32,
    expected_name: &CStr,
    expected_bytes: usize,
) -> Result<ToFloat, String> {
    // SAFETY: the export belongs to the verified pinned ggml-base DLL, whose
    // reviewed C ABI accepts these valid registered type IDs (142 and 143).
    let pointer = unsafe { get_traits(type_id) };
    let pointer =
        NonNull::new(pointer.cast_mut()).ok_or("The pinned native type traits are missing.")?;
    // SAFETY: the pinned export returns a live, aligned static type-traits
    // object with the 56-byte layout asserted above; the DLL remains loaded.
    let traits = unsafe { pointer.as_ref() };
    if traits.name.is_null() {
        return Err("The pinned native type name is missing.".into());
    }
    // SAFETY: the reviewed static native traits name is a NUL-terminated C
    // string, owned by the still-loaded verified DLL.
    let name = unsafe { CStr::from_ptr(traits.name) };
    if name != expected_name
        || traits.block_size != 128
        || traits.interleave != 0
        || traits.type_size != expected_bytes
        || traits.quantized != 1
    {
        return Err("The pinned native codec type contract changed.".into());
    }
    traits
        .to_float
        .ok_or_else(|| "The pinned native dequantizer is missing.".into())
}

/// Runs every fixed synthetic block against the released native dequantizers.
#[expect(
    clippy::redundant_pub_crate,
    reason = "binary implementation modules intentionally expose only to their parent"
)]
pub(super) fn run(runtime: &Path) -> Result<usize, String> {
    let path = runtime.join("ggml-base.dll");
    if !path.is_file() {
        return Err("The fully verified native codec library is missing.".into());
    }
    let wide: Vec<_> = path.as_os_str().encode_wide().chain([0]).collect();
    if wide[..wide.len() - 1].contains(&0) {
        return Err("The native library path contains a NUL.".into());
    }
    // SAFETY: wide is an absolute NUL-terminated DLL path live for the call.
    // Dependency resolution is restricted to that DLL directory and System32;
    // the wrapper verified the complete runtime package before child launch.
    let handle = unsafe {
        LoadLibraryExW(
            wide.as_ptr(),
            std::ptr::null_mut(),
            LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32,
        )
    };
    if handle.is_null() {
        return Err(std::io::Error::last_os_error().to_string());
    }
    let library = Library(handle);
    // SAFETY: library owns a live DLL reference and the export name is NUL
    // terminated. The result is checked before the reviewed ABI conversion.
    let address = unsafe { GetProcAddress(library.0, c"ggml_get_type_traits".as_ptr().cast()) }
        .ok_or("The pinned ggml_get_type_traits export is missing.")?;
    // SAFETY: the exact pinned export has the reviewed C GetTraits signature;
    // both pointer types share the Windows ARM64 function-pointer ABI.
    let get_traits =
        unsafe { std::mem::transmute::<unsafe extern "system" fn() -> isize, GetTraits>(address) };
    let pq2 = decoder(get_traits, 142, c"pq2_0", 34)?;
    let ptq1 = decoder(get_traits, 143, c"ptq1_0", 28)?;
    super::vectors::check_vectors(|source, packed| {
        let source = NativeBlock(*source);
        let packed = NativeBlock(*packed);
        let mut original = [0.0; 128];
        let mut repacked = [0.0; 128];
        // SAFETY: the reviewed PQ2 decoder reads exactly one 34-byte block
        // aligned for its uint16_t scale and writes exactly 128 FP32 elements
        // into the disjoint live destination array.
        unsafe { pq2(source.0.as_ptr().cast(), original.as_mut_ptr(), 128) };
        // SAFETY: the reviewed PTQ1 decoder reads exactly one 28-byte block
        // aligned for its uint16_t scale and writes exactly 128 FP32 elements
        // into the disjoint live destination array.
        unsafe { ptq1(packed.0.as_ptr().cast(), repacked.as_mut_ptr(), 128) };
        Ok((original, repacked))
    })
}
