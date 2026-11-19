//! Safe System32 Vulkan 1.x loader, native Qualcomm Adreno device selection, and compute runtime.
//!
//! Confines all Vulkan and Win32 loader `unsafe` operations to this module, loads ONLY
//! `%SystemRoot%\System32\vulkan-1.dll`, rejects `VK_*` driver/layer overrides and D3D12/CPU
//! layered drivers, queries full Adreno compute capabilities, inserts explicit pipeline
//! barriers, and latches unrecoverable faults with uncertain-buffer retention until teardown.

use std::collections::BTreeSet;
use std::ffi::CStr;
use std::ptr::null_mut;
use std::slice;

use ash::vk;

use crate::{
    AdrenoFamily, CoopMatShape, Error, MemoryHeapBudget, VulkanComputeLimits,
    VulkanDeviceCapabilities,
};

const FENCE_TIMEOUT_NS: u64 = 5_000_000_000;
const QUALCOMM_VENDOR_ID: u32 = 0x5143;

const FORBIDDEN_VK_ENV_PREFIXES: [&str; 10] = [
    "VK_ICD_FILENAMES",
    "VK_DRIVER_FILES",
    "VK_ADD_DRIVER_FILES",
    "VK_LAYER_PATH",
    "VK_ADD_LAYER_PATH",
    "VK_INSTANCE_LAYERS",
    "VK_LOADER_DRIVERS_SELECT",
    "VK_LOADER_DRIVERS_DISABLE",
    "VK_LOADER_LAYERS_ENABLE",
    "VK_LOADER_LAYERS_DISABLE",
];

/// Opaque handle to a GPU buffer owned by [`AdrenoVulkanDevice`].
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct BufferId(usize);

/// Opaque handle to a compiled compute pipeline owned by [`AdrenoVulkanDevice`].
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct PipelineId(usize);

/// One compute dispatch stage queued into [`AdrenoVulkanDevice::submit_compute_stages`].
#[derive(Clone, Copy, Debug)]
pub struct ComputeStage<'a> {
    /// Compiled compute pipeline handle.
    pub pipeline: PipelineId,
    /// Storage buffers bound sequentially to set `0`, bindings `0..n`.
    pub buffers: &'a [BufferId],
    /// Push constant `u32` words starting at byte offset `0`.
    pub push_u32s: &'a [u32],
    /// Workgroup grid `[x, y, z]`.
    pub workgroups: [u32; 3],
}

#[cfg(windows)]
mod win_loader {
    use std::ffi::CStr;
    use std::ptr::null_mut;

    use windows_sys::Win32::Foundation::{FreeLibrary, HMODULE};
    use windows_sys::Win32::System::LibraryLoader::{
        GetProcAddress, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR, LOAD_LIBRARY_SEARCH_SYSTEM32,
        LoadLibraryExW,
    };
    use windows_sys::Win32::System::SystemInformation::GetSystemDirectoryW;

    use crate::Error;

    pub(super) struct VulkanLibrary {
        handle: HMODULE,
    }

    impl VulkanLibrary {
        pub(super) fn load_system32() -> Result<Self, Error> {
            let mut buffer = [0_u16; 300];
            let buf_cap = u32::try_from(buffer.len())
                .map_err(|_| Error::Vulkan("system directory buffer overflow".to_owned()))?;
            // SAFETY: `buffer` is writable for `buf_cap` UTF-16 code units.
            let sys_len =
                usize::try_from(unsafe { GetSystemDirectoryW(buffer.as_mut_ptr(), buf_cap) })
                    .unwrap_or(0);
            if sys_len == 0 || sys_len >= buffer.len() {
                return Err(Error::Unsupported);
            }
            let suffix: Vec<u16> = "\\vulkan-1.dll\0".encode_utf16().collect();
            if sys_len + suffix.len() > buffer.len() {
                return Err(Error::Unsupported);
            }
            buffer[sys_len..sys_len + suffix.len()].copy_from_slice(&suffix);
            let flags = LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32;
            // SAFETY: `buffer` is NUL-terminated UTF-16 pointing into System32.
            let handle = unsafe { LoadLibraryExW(buffer.as_ptr(), null_mut(), flags) };
            if handle.is_null() {
                return Err(Error::Unsupported);
            }
            Ok(Self { handle })
        }

        pub(super) fn symbol(
            &self,
            name: &CStr,
        ) -> Result<unsafe extern "system" fn() -> isize, Error> {
            let ptr = name.as_ptr().cast::<u8>();
            // SAFETY: `self.handle` is a live module handle and `name` is NUL-terminated ASCII.
            let sym = unsafe { GetProcAddress(self.handle, ptr) };
            sym.ok_or(Error::Unsupported)
        }
    }

    impl Drop for VulkanLibrary {
        fn drop(&mut self) {
            if !self.handle.is_null() {
                // SAFETY: `self.handle` came from `LoadLibraryExW` and is freed after Vulkan handles drop.
                let _ = unsafe { FreeLibrary(self.handle) };
                self.handle = null_mut();
            }
        }
    }
}

struct OwnedBuffer {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    size_bytes: u64,
    host_coherent: bool,
}

struct OwnedPipeline {
    binding_count: usize,
    push_bytes: u32,
    set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
}

/// Self-contained native Qualcomm Adreno Vulkan compute device and resource owner.
pub struct AdrenoVulkanDevice {
    capabilities: VulkanDeviceCapabilities,
    sdot_enabled: bool,
    buffers: Vec<Option<OwnedBuffer>>,
    uncertain_buffers: Vec<OwnedBuffer>,
    pipelines: Vec<OwnedPipeline>,
    faulted: bool,
    fence: vk::Fence,
    command_buffer: vk::CommandBuffer,
    descriptor_pool: vk::DescriptorPool,
    command_pool: vk::CommandPool,
    compute_queue: vk::Queue,
    mem_properties: vk::PhysicalDeviceMemoryProperties,
    device: ash::Device,
    instance: ash::Instance,
    _entry: ash::Entry,
    #[cfg(windows)]
    _library: win_loader::VulkanLibrary,
}

impl AdrenoVulkanDevice {
    /// Rejects environment overrides, loads `%SystemRoot%\System32\vulkan-1.dll`, selects the
    /// native Qualcomm Adreno integrated GPU (`0x5143`), queries capabilities, and creates a
    /// compute queue context.
    ///
    /// # Errors
    /// Returns [`Error::InvalidInput`] if forbidden `VK_*` environment overrides are set,
    /// [`Error::Unsupported`] if no native Adreno Vulkan compute adapter exists, or
    /// [`Error::Vulkan`] if Vulkan device/pool creation fails.
    pub fn open() -> Result<Self, Error> {
        check_forbidden_env()?;
        Self::open_internal()
    }

    #[cfg(not(windows))]
    fn open_internal() -> Result<Self, Error> {
        Err(Error::Unsupported)
    }

    #[cfg(windows)]
    fn open_internal() -> Result<Self, Error> {
        let library = win_loader::VulkanLibrary::load_system32()?;
        let proc_raw = library.symbol(c"vkGetInstanceProcAddr")?;
        // SAFETY: `vkGetInstanceProcAddr` matches `vk::PFN_vkGetInstanceProcAddr` ABI.
        let get_instance_proc_addr: vk::PFN_vkGetInstanceProcAddr =
            unsafe { std::mem::transmute(proc_raw) };
        // SAFETY: Null instance query for `vkCreateInstance` is valid per Vulkan spec.
        let create_fn =
            unsafe { get_instance_proc_addr(vk::Instance::null(), c"vkCreateInstance".as_ptr()) };
        if create_fn.is_none() {
            return Err(Error::Unsupported);
        }
        // SAFETY: `get_instance_proc_addr` is valid for `_library`'s lifetime and `vkCreateInstance` exists.
        let entry = unsafe {
            ash::Entry::from_static_fn(ash::StaticFn {
                get_instance_proc_addr,
            })
        };

        // SAFETY: Querying instance version with no external state.
        let instance_api = unsafe { entry.try_enumerate_instance_version() }
            .map_err(|err| Error::Vulkan(format!("vkEnumerateInstanceVersion failed: {err:?}")))?
            .unwrap_or(vk::API_VERSION_1_0);
        if instance_api < vk::API_VERSION_1_3 {
            return Err(Error::Unsupported);
        }
        let app_info = vk::ApplicationInfo::default()
            .application_name(c"bonsai-adreno-vulkan")
            .application_version(1)
            .engine_name(c"prism")
            .engine_version(1)
            .api_version(vk::API_VERSION_1_3);
        let create_info = vk::InstanceCreateInfo::default().application_info(&app_info);
        // SAFETY: `create_info` points to stack structs valid for this call.
        let instance =
            unsafe { entry.create_instance(&create_info, None) }.map_err(|_| Error::Unsupported)?;

        Self::init_from_instance(entry, instance, library)
    }

    #[cfg(windows)]
    fn init_from_instance(
        entry: ash::Entry,
        instance: ash::Instance,
        library: win_loader::VulkanLibrary,
    ) -> Result<Self, Error> {
        // SAFETY: `instance` is valid; enumerates physical devices.
        let physical_devices = match unsafe { instance.enumerate_physical_devices() } {
            Ok(list) => list,
            Err(err) => {
                // SAFETY: Destroying partially initialized instance on error path.
                unsafe { instance.destroy_instance(None) };
                return Err(Error::Vulkan(format!(
                    "enumerate_physical_devices: {err:?}"
                )));
            }
        };

        let Some((phys, queue_family, caps, _ext_set, mem_properties)) =
            select_native_adreno(&entry, &instance, &physical_devices)
        else {
            // SAFETY: No suitable Adreno adapter found; clean up instance.
            unsafe { instance.destroy_instance(None) };
            return Err(Error::Unsupported);
        };

        let (device, sdot_enabled) =
            match create_compute_device(&instance, phys, queue_family, &caps) {
                Ok(pair) => pair,
                Err(err) => {
                    // SAFETY: Clean up instance after device creation failure.
                    unsafe { instance.destroy_instance(None) };
                    return Err(err);
                }
            };

        Self::create_runtime_objects(
            entry,
            instance,
            device,
            library,
            queue_family,
            caps,
            sdot_enabled,
            &mem_properties,
        )
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "Transfers ownership of already-created Ash handles cleanly into one owner."
    )]
    #[cfg(windows)]
    fn create_runtime_objects(
        entry: ash::Entry,
        instance: ash::Instance,
        device: ash::Device,
        library: win_loader::VulkanLibrary,
        queue_family: u32,
        capabilities: VulkanDeviceCapabilities,
        sdot_enabled: bool,
        mem_properties: &vk::PhysicalDeviceMemoryProperties,
    ) -> Result<Self, Error> {
        // SAFETY: `queue_family` and queue index 0 were requested in `DeviceQueueCreateInfo`.
        let compute_queue = unsafe { device.get_device_queue(queue_family, 0) };

        let pool_info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(queue_family);
        // SAFETY: Creating command pool on live device.
        let command_pool =
            unsafe { device.create_command_pool(&pool_info, None) }.map_err(|err| {
                destroy_dev_inst_err(&device, &instance, format!("create_command_pool: {err:?}"))
            })?;

        let cmd_alloc = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        // SAFETY: Allocating 1 primary command buffer from `command_pool`.
        let cmd_bufs = match unsafe { device.allocate_command_buffers(&cmd_alloc) } {
            Ok(list) => list,
            Err(err) => {
                // SAFETY: Rollback `command_pool` on error.
                unsafe { device.destroy_command_pool(command_pool, None) };
                return Err(destroy_dev_inst_err(
                    &device,
                    &instance,
                    format!("allocate_command_buffers: {err:?}"),
                ));
            }
        };
        let command_buffer = cmd_bufs[0];

        let pool_sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 1024,
        }];
        let desc_info = vk::DescriptorPoolCreateInfo::default()
            .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET)
            .max_sets(256)
            .pool_sizes(&pool_sizes);
        // SAFETY: Creating descriptor pool on live device.
        let descriptor_pool = match unsafe { device.create_descriptor_pool(&desc_info, None) } {
            Ok(pool) => pool,
            Err(err) => {
                // SAFETY: Rollback command pool.
                unsafe { device.destroy_command_pool(command_pool, None) };
                return Err(destroy_dev_inst_err(
                    &device,
                    &instance,
                    format!("create_descriptor_pool: {err:?}"),
                ));
            }
        };

        let fence_info = vk::FenceCreateInfo::default();
        // SAFETY: Creating fence on live device.
        let fence = match unsafe { device.create_fence(&fence_info, None) } {
            Ok(f) => f,
            Err(err) => {
                // SAFETY: Rollback descriptor pool.
                unsafe { device.destroy_descriptor_pool(descriptor_pool, None) };
                // SAFETY: Rollback command pool.
                unsafe { device.destroy_command_pool(command_pool, None) };
                return Err(destroy_dev_inst_err(
                    &device,
                    &instance,
                    format!("create_fence: {err:?}"),
                ));
            }
        };

        Ok(Self {
            capabilities,
            sdot_enabled,
            buffers: Vec::new(),
            uncertain_buffers: Vec::new(),
            pipelines: Vec::new(),
            faulted: false,
            fence,
            command_buffer,
            descriptor_pool,
            command_pool,
            compute_queue,
            mem_properties: *mem_properties,
            device,
            instance,
            _entry: entry,
            _library: library,
        })
    }

    /// Returns the discovered Qualcomm Adreno capability snapshot.
    #[must_use]
    pub const fn capabilities(&self) -> &VulkanDeviceCapabilities {
        &self.capabilities
    }

    /// Returns `true` if `VK_KHR_shader_integer_dot_product` was enabled on the logical device.
    #[must_use]
    pub const fn integer_dot_product_enabled(&self) -> bool {
        self.sdot_enabled
    }

    /// Returns `true` if a timeout or device loss latched a terminal fault state.
    #[must_use]
    pub const fn is_faulted(&self) -> bool {
        self.faulted
    }

    fn ensure_healthy(&self) -> Result<(), Error> {
        if self.faulted {
            Err(Error::DeviceFault(
                "Vulkan device is in faulted state and requires teardown".to_owned(),
            ))
        } else {
            Ok(())
        }
    }

    /// Allocates a storage buffer large enough for `size_bytes` (`>= 4`).
    ///
    /// # Errors
    /// Returns [`Error::DeviceFault`] if faulted or [`Error::Vulkan`] if buffer/memory allocation fails.
    pub fn create_storage_buffer(&mut self, size_bytes: u64) -> Result<BufferId, Error> {
        self.ensure_healthy()?;
        let size = size_bytes.max(4);
        let buf_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        // SAFETY: Creating `vk::Buffer` on live device.
        let buffer = unsafe { self.device.create_buffer(&buf_info, None) }
            .map_err(|err| Error::Vulkan(format!("create_buffer: {err:?}")))?;
        // SAFETY: Querying memory requirements for `buffer`.
        let reqs = unsafe { self.device.get_buffer_memory_requirements(buffer) };
        let Some((mem_type_idx, host_coherent)) =
            find_memory_type(&self.mem_properties, reqs.memory_type_bits)
        else {
            // SAFETY: Cleaning up unused `buffer` handle.
            unsafe { self.device.destroy_buffer(buffer, None) };
            return Err(Error::ResourceLimit(
                "no host-visible Vulkan memory type satisfies buffer requirements".to_owned(),
            ));
        };
        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(reqs.size)
            .memory_type_index(mem_type_idx);
        // SAFETY: Allocating `reqs.size` bytes on selected `mem_type_idx`.
        let memory = match unsafe { self.device.allocate_memory(&alloc_info, None) } {
            Ok(mem) => mem,
            Err(err) => {
                // SAFETY: Destroy unbacked `buffer`.
                unsafe { self.device.destroy_buffer(buffer, None) };
                return Err(Error::Vulkan(format!("allocate_memory: {err:?}")));
            }
        };
        // SAFETY: Binding `memory` at offset 0 to `buffer`.
        if let Err(err) = unsafe { self.device.bind_buffer_memory(buffer, memory, 0) } {
            // SAFETY: Free `memory` on bind failure.
            unsafe { self.device.free_memory(memory, None) };
            // SAFETY: Destroy `buffer` on bind failure.
            unsafe { self.device.destroy_buffer(buffer, None) };
            return Err(Error::Vulkan(format!("bind_buffer_memory: {err:?}")));
        }

        let slot = OwnedBuffer {
            buffer,
            memory,
            size_bytes: size,
            host_coherent,
        };
        if let Some(empty_idx) = self.buffers.iter().position(Option::is_none) {
            self.buffers[empty_idx] = Some(slot);
            Ok(BufferId(empty_idx))
        } else {
            let idx = self.buffers.len();
            self.buffers.push(Some(slot));
            Ok(BufferId(idx))
        }
    }

    /// Uploads raw bytes into a host-visible storage buffer starting at byte offset `0`.
    ///
    /// # Errors
    /// Returns [`Error::InvalidInput`] if `data.len()` exceeds buffer size or [`Error::Vulkan`] on map failure.
    pub fn write_buffer_bytes(&self, id: BufferId, data: &[u8]) -> Result<(), Error> {
        self.ensure_healthy()?;
        let entry = self
            .buffers
            .get(id.0)
            .and_then(Option::as_ref)
            .ok_or_else(|| Error::InvalidInput("invalid buffer handle".to_owned()))?;
        let data_len_u64 = u64::try_from(data.len())
            .map_err(|_| Error::InvalidInput("upload byte length overflow".to_owned()))?;
        if data_len_u64 > entry.size_bytes {
            return Err(Error::InvalidInput(format!(
                "upload ({data_len_u64} B) exceeds buffer size ({} B)",
                entry.size_bytes
            )));
        }
        if data.is_empty() {
            return Ok(());
        }
        // SAFETY: Mapping non-empty host-visible `entry.memory` for `data_len_u64` bytes.
        let ptr = unsafe {
            self.device
                .map_memory(entry.memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
        }
        .map_err(|err| Error::Vulkan(format!("map_memory: {err:?}")))?;
        // SAFETY: Both `data.as_ptr()` and mapped `ptr` are valid, non-overlapping for `data.len()` bytes.
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), ptr.cast::<u8>(), data.len()) };
        if !entry.host_coherent {
            let range = [vk::MappedMemoryRange::default()
                .memory(entry.memory)
                .offset(0)
                .size(vk::WHOLE_SIZE)];
            // SAFETY: Flushing mapped memory range when heap is non-coherent.
            let _ = unsafe { self.device.flush_mapped_memory_ranges(&range) };
        }
        // SAFETY: Unmapping `entry.memory` cleanly.
        unsafe { self.device.unmap_memory(entry.memory) };
        Ok(())
    }

    /// Uploads a slice of `u32` words into a storage buffer.
    ///
    /// # Errors
    /// Propagates [`Self::write_buffer_bytes`] errors.
    pub fn write_buffer_u32(&self, id: BufferId, words: &[u32]) -> Result<(), Error> {
        let byte_len = words
            .len()
            .checked_mul(4)
            .ok_or_else(|| Error::InvalidInput("u32 byte count overflow".to_owned()))?;
        let mut bytes = Vec::with_capacity(byte_len);
        for &word in words {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
        self.write_buffer_bytes(id, &bytes)
    }

    /// Uploads a slice of `f32` scalars into a storage buffer.
    ///
    /// # Errors
    /// Propagates [`Self::write_buffer_bytes`] errors.
    pub fn write_buffer_f32(&self, id: BufferId, values: &[f32]) -> Result<(), Error> {
        let byte_len = values
            .len()
            .checked_mul(4)
            .ok_or_else(|| Error::InvalidInput("f32 byte count overflow".to_owned()))?;
        let mut bytes = Vec::with_capacity(byte_len);
        for &val in values {
            bytes.extend_from_slice(&val.to_le_bytes());
        }
        self.write_buffer_bytes(id, &bytes)
    }

    /// Downloads `dst.len()` `f32` values from a storage buffer starting at byte offset `0`.
    ///
    /// # Errors
    /// Returns [`Error::InvalidInput`] if `dst` exceeds buffer bounds or [`Error::Vulkan`] on map failure.
    pub fn read_buffer_f32(&self, id: BufferId, dst: &mut [f32]) -> Result<(), Error> {
        self.ensure_healthy()?;
        let entry = self
            .buffers
            .get(id.0)
            .and_then(Option::as_ref)
            .ok_or_else(|| Error::InvalidInput("invalid buffer handle".to_owned()))?;
        let byte_len = dst
            .len()
            .checked_mul(4)
            .ok_or_else(|| Error::InvalidInput("download byte length overflow".to_owned()))?;
        let byte_len_u64 = u64::try_from(byte_len)
            .map_err(|_| Error::InvalidInput("download byte length overflow".to_owned()))?;
        if byte_len_u64 > entry.size_bytes {
            return Err(Error::InvalidInput(format!(
                "download ({byte_len_u64} B) exceeds buffer size ({} B)",
                entry.size_bytes
            )));
        }
        if dst.is_empty() {
            return Ok(());
        }
        // SAFETY: Mapping host-visible `entry.memory` for readback.
        let ptr = unsafe {
            self.device
                .map_memory(entry.memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
        }
        .map_err(|err| Error::Vulkan(format!("map_memory: {err:?}")))?;
        if !entry.host_coherent {
            let range = [vk::MappedMemoryRange::default()
                .memory(entry.memory)
                .offset(0)
                .size(vk::WHOLE_SIZE)];
            // SAFETY: Invalidating mapped range before CPU readback when non-coherent.
            let _ = unsafe { self.device.invalidate_mapped_memory_ranges(&range) };
        }
        let mut raw = vec![0_u8; byte_len];
        // SAFETY: `ptr` is valid for at least `byte_len` bytes and does not overlap `raw`.
        unsafe { std::ptr::copy_nonoverlapping(ptr.cast::<u8>(), raw.as_mut_ptr(), byte_len) };
        // SAFETY: Unmapping `entry.memory` after copy.
        unsafe { self.device.unmap_memory(entry.memory) };

        for (out_slot, chunk) in dst.iter_mut().zip(raw.as_chunks::<4>().0) {
            *out_slot = f32::from_le_bytes(*chunk);
        }
        Ok(())
    }

    /// Releases a buffer immediately when healthy, or retains it in uncertain storage when faulted.
    pub fn free_buffer(&mut self, id: BufferId) {
        let Some(slot) = self.buffers.get_mut(id.0) else {
            return;
        };
        let Some(buf) = slot.take() else {
            return;
        };
        if self.faulted {
            self.uncertain_buffers.push(buf);
        } else {
            // SAFETY: Device is idle between submissions and healthy; safe to destroy buffer.
            unsafe { self.device.destroy_buffer(buf.buffer, None) };
            // SAFETY: Safe to free backing `DeviceMemory` after destroying bound `vk::Buffer`.
            unsafe { self.device.free_memory(buf.memory, None) };
        }
    }

    /// Compiles one SPIR-V 1.6 compute shader module into a reusable pipeline handle.
    ///
    /// # Errors
    /// Returns [`Error::InvalidInput`] on empty SPIR-V or invalid descriptor/push counts,
    /// [`Error::DeviceFault`] if faulted, or [`Error::Vulkan`] if pipeline compilation fails.
    pub fn create_compute_pipeline(
        &mut self,
        spirv_words: &[u32],
        binding_count: usize,
        push_u32_count: usize,
    ) -> Result<PipelineId, Error> {
        self.ensure_healthy()?;
        if spirv_words.len() < 5 {
            return Err(Error::InvalidInput("empty SPIR-V shader module".to_owned()));
        }
        let push_bytes = u32::try_from(push_u32_count)
            .ok()
            .and_then(|c| c.checked_mul(4))
            .ok_or_else(|| Error::InvalidInput("push constant size overflow".to_owned()))?;
        if push_bytes > self.capabilities.compute_limits.max_push_constants_size {
            return Err(Error::ResourceLimit(format!(
                "push constants ({push_bytes} B) exceed maxPushConstantsSize ({} B)",
                self.capabilities.compute_limits.max_push_constants_size
            )));
        }

        let shader_info = vk::ShaderModuleCreateInfo::default().code(spirv_words);
        // SAFETY: `spirv_words` slice is valid for `shader_info` duration.
        let shader_module = unsafe { self.device.create_shader_module(&shader_info, None) }
            .map_err(|err| Error::Vulkan(format!("create_shader_module: {err:?}")))?;

        let bindings: Vec<vk::DescriptorSetLayoutBinding<'_>> = (0..binding_count)
            .map(|idx| vk::DescriptorSetLayoutBinding {
                binding: u32::try_from(idx).unwrap_or(0),
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                p_immutable_samplers: null_mut(),
                _marker: std::marker::PhantomData,
            })
            .collect();
        let set_layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        // SAFETY: Creating descriptor set layout on live device.
        let set_layout = match unsafe {
            self.device
                .create_descriptor_set_layout(&set_layout_info, None)
        } {
            Ok(layout) => layout,
            Err(err) => {
                // SAFETY: Clean up `shader_module` on error.
                unsafe { self.device.destroy_shader_module(shader_module, None) };
                return Err(Error::Vulkan(format!(
                    "create_descriptor_set_layout: {err:?}"
                )));
            }
        };

        let push_ranges = if push_bytes > 0 {
            vec![vk::PushConstantRange {
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                offset: 0,
                size: push_bytes,
            }]
        } else {
            Vec::new()
        };
        let set_layouts = [set_layout];
        let layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&set_layouts)
            .push_constant_ranges(&push_ranges);
        // SAFETY: Creating pipeline layout on live device.
        let layout = match unsafe { self.device.create_pipeline_layout(&layout_info, None) } {
            Ok(pl) => pl,
            Err(err) => {
                // SAFETY: Clean up `set_layout`.
                unsafe { self.device.destroy_descriptor_set_layout(set_layout, None) };
                // SAFETY: Clean up `shader_module`.
                unsafe { self.device.destroy_shader_module(shader_module, None) };
                return Err(Error::Vulkan(format!("create_pipeline_layout: {err:?}")));
            }
        };

        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader_module)
            .name(c"main");
        let cp_info = [vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(layout)];
        // SAFETY: Creating compute pipeline; shader module destroyed immediately afterwards.
        let pipelines = unsafe {
            self.device
                .create_compute_pipelines(vk::PipelineCache::null(), &cp_info, None)
        };
        // SAFETY: Shader module is no longer needed once pipeline creation returns.
        unsafe { self.device.destroy_shader_module(shader_module, None) };

        let pipeline = match pipelines {
            Ok(list) => list[0],
            Err((partial, err)) => {
                for p in partial {
                    // SAFETY: Destroy any partial pipeline.
                    unsafe { self.device.destroy_pipeline(p, None) };
                }
                // SAFETY: Clean up `layout`.
                unsafe { self.device.destroy_pipeline_layout(layout, None) };
                // SAFETY: Clean up `set_layout`.
                unsafe { self.device.destroy_descriptor_set_layout(set_layout, None) };
                return Err(Error::Vulkan(format!("create_compute_pipelines: {err:?}")));
            }
        };

        let id = PipelineId(self.pipelines.len());
        self.pipelines.push(OwnedPipeline {
            binding_count,
            push_bytes,
            set_layout,
            layout,
            pipeline,
        });
        Ok(id)
    }

    /// Executes one compute pipeline dispatch synchronously with bounded fence wait (`5s`).
    ///
    /// # Errors
    /// Returns [`Error::InvalidInput`] on invalid handles/extents, [`Error::DeviceFault`] if
    /// fence wait times out or reports device loss, or [`Error::Vulkan`] on driver failure.
    pub fn submit_compute(
        &mut self,
        pipeline: PipelineId,
        buffers: &[BufferId],
        push_u32s: &[u32],
        workgroups: [u32; 3],
    ) -> Result<(), Error> {
        let stage = ComputeStage {
            pipeline,
            buffers,
            push_u32s,
            workgroups,
        };
        self.submit_compute_stages(&[stage])
    }

    fn validate_stages(&self, stages: &[ComputeStage<'_>]) -> Result<(), Error> {
        if stages.is_empty() {
            return Err(Error::InvalidInput(
                "at least one compute stage is required".to_owned(),
            ));
        }
        for stage in stages {
            let pipe = self
                .pipelines
                .get(stage.pipeline.0)
                .ok_or_else(|| Error::InvalidInput("invalid pipeline handle".to_owned()))?;
            if stage.buffers.len() != pipe.binding_count {
                return Err(Error::InvalidInput(format!(
                    "pipeline expects {} buffers, got {}",
                    pipe.binding_count,
                    stage.buffers.len()
                )));
            }
            let push_bytes = u32::try_from(stage.push_u32s.len())
                .ok()
                .and_then(|c| c.checked_mul(4))
                .ok_or_else(|| Error::InvalidInput("push constants overflow".to_owned()))?;
            if push_bytes != pipe.push_bytes {
                return Err(Error::InvalidInput(format!(
                    "pipeline expects {} push constant bytes, got {push_bytes}",
                    pipe.push_bytes
                )));
            }
            if stage.workgroups.contains(&0) {
                return Err(Error::InvalidInput(
                    "workgroup dispatch extents must be >= 1".to_owned(),
                ));
            }
            let limits = &self
                .capabilities
                .compute_limits
                .max_compute_work_group_count;
            if stage.workgroups[0] > limits[0]
                || stage.workgroups[1] > limits[1]
                || stage.workgroups[2] > limits[2]
            {
                return Err(Error::ResourceLimit(
                    "workgroup count exceeds maxComputeWorkGroupCount".to_owned(),
                ));
            }
            for &buf_id in stage.buffers {
                if self
                    .buffers
                    .get(buf_id.0)
                    .and_then(Option::as_ref)
                    .is_none()
                {
                    return Err(Error::InvalidInput(
                        "invalid buffer handle in stage".to_owned(),
                    ));
                }
            }
        }
        Ok(())
    }

    /// Records and executes one or more compute stages in sequence inside one command buffer,
    /// inserting explicit `COMPUTE_SHADER -> COMPUTE_SHADER` (`SHADER_WRITE -> SHADER_READ`)
    /// pipeline barriers between consecutive stages and waiting with a bounded `5s` fence.
    ///
    /// # Errors
    /// Returns [`Error::InvalidInput`] on malformed stages, [`Error::DeviceFault`] on timeout or
    /// `ERROR_DEVICE_LOST` (latching terminal teardown), or [`Error::Vulkan`] on other driver failures.
    pub fn submit_compute_stages(&mut self, stages: &[ComputeStage<'_>]) -> Result<(), Error> {
        self.ensure_healthy()?;
        self.validate_stages(stages)?;

        let mut allocated_sets = Vec::with_capacity(stages.len());
        for stage in stages {
            let set_layout = self.pipelines[stage.pipeline.0].set_layout;
            let set_layouts = [set_layout];
            let alloc = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(self.descriptor_pool)
                .set_layouts(&set_layouts);
            // SAFETY: Allocating 1 descriptor set from `descriptor_pool`.
            let sets = match unsafe { self.device.allocate_descriptor_sets(&alloc) } {
                Ok(s) => s,
                Err(err) => {
                    self.free_descriptor_sets(&allocated_sets);
                    return Err(Error::Vulkan(format!("allocate_descriptor_sets: {err:?}")));
                }
            };
            let set = sets[0];
            allocated_sets.push(set);

            let infos: Vec<vk::DescriptorBufferInfo> = stage
                .buffers
                .iter()
                .filter_map(|&id| {
                    self.buffers.get(id.0).and_then(Option::as_ref).map(|b| {
                        vk::DescriptorBufferInfo {
                            buffer: b.buffer,
                            offset: 0,
                            range: vk::WHOLE_SIZE,
                        }
                    })
                })
                .collect();
            let writes: Vec<vk::WriteDescriptorSet<'_>> = infos
                .iter()
                .enumerate()
                .map(|(binding, info)| vk::WriteDescriptorSet {
                    dst_set: set,
                    dst_binding: u32::try_from(binding).unwrap_or(0),
                    dst_array_element: 0,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                    p_buffer_info: info,
                    ..Default::default()
                })
                .collect();
            // SAFETY: Descriptor set `set` and `infos` remain valid across `update_descriptor_sets`.
            unsafe { self.device.update_descriptor_sets(&writes, &[]) };
        }

        let record_result = self.record_and_submit_stages(stages, &allocated_sets);
        if !self.faulted {
            self.free_descriptor_sets(&allocated_sets);
        }
        record_result
    }

    fn free_descriptor_sets(&self, sets: &[vk::DescriptorSet]) {
        if !sets.is_empty() {
            // SAFETY: Command buffer finished executing before freeing transient descriptor sets.
            let _ = unsafe { self.device.free_descriptor_sets(self.descriptor_pool, sets) };
        }
    }

    fn record_and_submit_stages(
        &mut self,
        stages: &[ComputeStage<'_>],
        sets: &[vk::DescriptorSet],
    ) -> Result<(), Error> {
        let fences = [self.fence];
        // SAFETY: Fence is unsignaled after creation or prior completion.
        if let Err(err) = unsafe { self.device.reset_fences(&fences) } {
            return Err(Error::Vulkan(format!("reset_fences: {err:?}")));
        }
        // SAFETY: Command buffer is not in use between synchronous submissions.
        if let Err(err) = unsafe {
            self.device
                .reset_command_buffer(self.command_buffer, vk::CommandBufferResetFlags::empty())
        } {
            return Err(Error::Vulkan(format!("reset_command_buffer: {err:?}")));
        }
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        // SAFETY: Recording commands into reset primary command buffer.
        if let Err(err) = unsafe {
            self.device
                .begin_command_buffer(self.command_buffer, &begin)
        } {
            return Err(Error::Vulkan(format!("begin_command_buffer: {err:?}")));
        }

        for (stage_idx, (stage, &set)) in stages.iter().zip(sets).enumerate() {
            if stage_idx > 0 {
                let barrier = [vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ)];
                // SAFETY: Inserting explicit compute-to-compute memory barrier between stages.
                unsafe {
                    self.device.cmd_pipeline_barrier(
                        self.command_buffer,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        vk::DependencyFlags::empty(),
                        &barrier,
                        &[],
                        &[],
                    );
                }
            }
            self.record_stage_dispatch(stage, set);
        }

        let host_barrier = [vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::HOST_READ)];
        // SAFETY: Synchronizing final shader writes for host readback.
        unsafe {
            self.device.cmd_pipeline_barrier(
                self.command_buffer,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &host_barrier,
                &[],
                &[],
            );
        }

        // SAFETY: Ending command buffer recording prior to queue submit.
        if let Err(err) = unsafe { self.device.end_command_buffer(self.command_buffer) } {
            return Err(Error::Vulkan(format!("end_command_buffer: {err:?}")));
        }

        self.submit_recorded_and_wait()
    }

    fn record_stage_dispatch(&self, stage: &ComputeStage<'_>, set: vk::DescriptorSet) {
        let pipe = &self.pipelines[stage.pipeline.0];
        // SAFETY: Binding compute pipeline for stage dispatch.
        unsafe {
            self.device.cmd_bind_pipeline(
                self.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                pipe.pipeline,
            );
        }
        let bind_sets = [set];
        // SAFETY: Binding updated descriptor set to pipeline layout set 0.
        unsafe {
            self.device.cmd_bind_descriptor_sets(
                self.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                pipe.layout,
                0,
                &bind_sets,
                &[],
            );
        }
        if !stage.push_u32s.is_empty() {
            let mut push_bytes = Vec::with_capacity(stage.push_u32s.len() * 4);
            for &word in stage.push_u32s {
                push_bytes.extend_from_slice(&word.to_le_bytes());
            }
            // SAFETY: Pushing validated push constants matching pipeline layout.
            unsafe {
                self.device.cmd_push_constants(
                    self.command_buffer,
                    pipe.layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    &push_bytes,
                );
            }
        }
        // SAFETY: Dispatching bounded workgroups.
        unsafe {
            self.device.cmd_dispatch(
                self.command_buffer,
                stage.workgroups[0],
                stage.workgroups[1],
                stage.workgroups[2],
            );
        }
    }

    fn submit_recorded_and_wait(&mut self) -> Result<(), Error> {
        let fences = [self.fence];
        let cmd_bufs = [self.command_buffer];
        let submits = [vk::SubmitInfo::default().command_buffers(&cmd_bufs)];
        // SAFETY: Submitting recorded command buffer to `compute_queue` with `self.fence`.
        if let Err(err) = unsafe {
            self.device
                .queue_submit(self.compute_queue, &submits, self.fence)
        } {
            if err == vk::Result::ERROR_DEVICE_LOST {
                self.faulted = true;
                return Err(Error::DeviceFault(
                    "queue_submit returned ERROR_DEVICE_LOST".to_owned(),
                ));
            }
            return Err(Error::Vulkan(format!("queue_submit: {err:?}")));
        }

        // SAFETY: Waiting up to `FENCE_TIMEOUT_NS` for submitted compute work.
        match unsafe { self.device.wait_for_fences(&fences, true, FENCE_TIMEOUT_NS) } {
            Ok(()) => Ok(()),
            Err(vk::Result::TIMEOUT) => {
                self.faulted = true;
                Err(Error::DeviceFault(
                    "GPU compute submission timed out after 5s; retaining buffers until teardown"
                        .to_owned(),
                ))
            }
            Err(vk::Result::ERROR_DEVICE_LOST) => {
                self.faulted = true;
                Err(Error::DeviceFault(
                    "GPU device lost while waiting on compute fence".to_owned(),
                ))
            }
            Err(err) => {
                self.faulted = true;
                Err(Error::DeviceFault(format!(
                    "wait_for_fences failed ({err:?}); latching terminal fault"
                )))
            }
        }
    }
}

impl Drop for AdrenoVulkanDevice {
    fn drop(&mut self) {
        // SAFETY: Draining all GPU work before releasing pools, buffers, pipelines, and device.
        let _ = unsafe { self.device.device_wait_idle() };

        for slot in &mut self.buffers {
            if let Some(buf) = slot.take() {
                // SAFETY: Device is idle; destroying owned `vk::Buffer`.
                unsafe { self.device.destroy_buffer(buf.buffer, None) };
                // SAFETY: Freeing owned `DeviceMemory`.
                unsafe { self.device.free_memory(buf.memory, None) };
            }
        }
        for buf in self.uncertain_buffers.drain(..) {
            // SAFETY: Device is idle or dropped on teardown; destroying retained uncertain buffer.
            unsafe { self.device.destroy_buffer(buf.buffer, None) };
            // SAFETY: Freeing retained uncertain memory.
            unsafe { self.device.free_memory(buf.memory, None) };
        }
        for pipe in self.pipelines.drain(..) {
            // SAFETY: Destroying owned `vk::Pipeline`.
            unsafe { self.device.destroy_pipeline(pipe.pipeline, None) };
            // SAFETY: Destroying owned `vk::PipelineLayout`.
            unsafe { self.device.destroy_pipeline_layout(pipe.layout, None) };
            // SAFETY: Destroying owned `vk::DescriptorSetLayout`.
            unsafe {
                self.device
                    .destroy_descriptor_set_layout(pipe.set_layout, None);
            }
        }
        // SAFETY: Destroying owned fence.
        unsafe { self.device.destroy_fence(self.fence, None) };
        // SAFETY: Destroying owned descriptor pool.
        unsafe {
            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);
        }
        // SAFETY: Destroying owned command pool.
        unsafe { self.device.destroy_command_pool(self.command_pool, None) };
        // SAFETY: Destroying logical Vulkan device.
        unsafe { self.device.destroy_device(None) };
        // SAFETY: Destroying Vulkan instance prior to `_library` drop.
        unsafe { self.instance.destroy_instance(None) };
    }
}

fn check_forbidden_env() -> Result<(), Error> {
    for (name, _) in std::env::vars_os() {
        if let Some(name_str) = name.to_str()
            && is_forbidden_vk_env(name_str)
        {
            return Err(Error::InvalidInput(format!(
                "forbidden Vulkan loader/layer override in environment: {name_str}"
            )));
        }
    }
    Ok(())
}

fn is_forbidden_vk_env(name: &str) -> bool {
    FORBIDDEN_VK_ENV_PREFIXES
        .iter()
        .any(|forbidden| name.eq_ignore_ascii_case(forbidden))
}

#[cfg(windows)]
fn destroy_dev_inst_err(device: &ash::Device, instance: &ash::Instance, msg: String) -> Error {
    // SAFETY: Rollback cleanup on partially initialized device.
    unsafe { device.destroy_device(None) };
    // SAFETY: Rollback cleanup on instance.
    unsafe { instance.destroy_instance(None) };
    Error::Vulkan(msg)
}

fn find_memory_type(
    props: &vk::PhysicalDeviceMemoryProperties,
    type_filter: u32,
) -> Option<(u32, bool)> {
    let preferred = vk::MemoryPropertyFlags::DEVICE_LOCAL
        | vk::MemoryPropertyFlags::HOST_VISIBLE
        | vk::MemoryPropertyFlags::HOST_COHERENT;
    let fallback = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
    let host_any = vk::MemoryPropertyFlags::HOST_VISIBLE;

    let mem_type_count = usize::try_from(props.memory_type_count).unwrap_or(0);
    for target in [preferred, fallback, host_any] {
        for (idx, mem_type) in props.memory_types.iter().enumerate().take(mem_type_count) {
            let idx_u32 = u32::try_from(idx).unwrap_or(0);
            let type_bit = 1_u32.checked_shl(idx_u32)?;
            let flags = mem_type.property_flags;
            if (type_filter & type_bit) != 0 && flags.contains(target) {
                return Some((
                    idx_u32,
                    flags.contains(vk::MemoryPropertyFlags::HOST_COHERENT),
                ));
            }
        }
    }
    None
}

fn cstr_to_string_lossy(raw: &[std::ffi::c_char]) -> Result<String, Error> {
    // SAFETY: Physical device fixed-size char arrays are NUL-terminated ASCII/UTF-8 by spec; slice bounds match `raw`.
    let bytes: &[u8] = unsafe { slice::from_raw_parts(raw.as_ptr().cast::<u8>(), raw.len()) };
    let text = CStr::from_bytes_until_nul(bytes)
        .map_err(|_| Error::Vulkan("unterminated Vulkan string".to_owned()))?
        .to_str()
        .map_err(|_| Error::Vulkan("non-UTF8 Vulkan string".to_owned()))?;
    if text.chars().any(char::is_control) {
        return Err(Error::Vulkan(
            "Vulkan string contained control characters".to_owned(),
        ));
    }
    Ok(text.to_owned())
}

#[cfg(windows)]
type SelectedAdapter = (
    vk::PhysicalDevice,
    u32,
    VulkanDeviceCapabilities,
    BTreeSet<String>,
    vk::PhysicalDeviceMemoryProperties,
);

#[cfg(windows)]
fn select_native_adreno(
    entry: &ash::Entry,
    instance: &ash::Instance,
    devices: &[vk::PhysicalDevice],
) -> Option<SelectedAdapter> {
    for &phys in devices {
        if let Some(selected) = inspect_adreno_candidate(entry, instance, phys) {
            return Some(selected);
        }
    }
    None
}

#[cfg(windows)]
fn inspect_adreno_candidate(
    entry: &ash::Entry,
    instance: &ash::Instance,
    phys: vk::PhysicalDevice,
) -> Option<SelectedAdapter> {
    // SAFETY: Querying physical device extensions on enumerated handle.
    let exts = (unsafe { instance.enumerate_device_extension_properties(phys) }).ok()?;
    let mut ext_set = BTreeSet::new();
    for ext in exts {
        if let Ok(name) = cstr_to_string_lossy(&ext.extension_name) {
            ext_set.insert(name);
        }
    }
    if ext_set.contains("VK_MSFT_layered_driver") {
        return None;
    }

    let mut subgroup = vk::PhysicalDeviceSubgroupProperties::default();
    let mut sdot_props = vk::PhysicalDeviceShaderIntegerDotProductProperties::default();
    let mut props2 = vk::PhysicalDeviceProperties2::default()
        .push_next(&mut subgroup)
        .push_next(&mut sdot_props);
    // SAFETY: Querying physical device properties2 chain.
    unsafe { instance.get_physical_device_properties2(phys, &mut props2) };
    let props = props2.properties;

    if props.vendor_id != QUALCOMM_VENDOR_ID
        || props.device_type != vk::PhysicalDeviceType::INTEGRATED_GPU
        || props.api_version < vk::API_VERSION_1_3
    {
        return None;
    }
    let device_name = cstr_to_string_lossy(&props.device_name).ok()?;
    if device_name.contains("Microsoft") || device_name.contains("Direct3D12") {
        return None;
    }

    // SAFETY: Querying queue families on selected physical device.
    let queues = unsafe { instance.get_physical_device_queue_family_properties(phys) };
    let compute_family = queues
        .iter()
        .position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE) && q.queue_count > 0)
        .and_then(|idx| u32::try_from(idx).ok())?;

    let mut f16_feats = vk::PhysicalDeviceShaderFloat16Int8Features::default();
    let mut sdot_feats = vk::PhysicalDeviceShaderIntegerDotProductFeatures::default();
    let mut feats2 = vk::PhysicalDeviceFeatures2::default()
        .push_next(&mut f16_feats)
        .push_next(&mut sdot_feats);
    // SAFETY: Querying physical device features2 chain.
    unsafe { instance.get_physical_device_features2(phys, &mut feats2) };

    let coop_shapes = if ext_set.contains("VK_KHR_cooperative_matrix") {
        query_coop_mat_shapes(entry, instance, phys)
    } else {
        Vec::new()
    };

    let (mem_properties, memory_heaps) = query_memory_heaps(instance, phys, &ext_set);
    let family = AdrenoFamily::from_device_name(&device_name);
    let mut caps = VulkanDeviceCapabilities {
        device_name,
        family,
        vendor_id: props.vendor_id,
        device_id: props.device_id,
        driver_version: props.driver_version,
        api_version: props.api_version,
        subgroup_size: subgroup.subgroup_size,
        shader_float16: f16_feats.shader_float16 == vk::TRUE,
        shader_integer_dot_product: sdot_feats.shader_integer_dot_product == vk::TRUE,
        integer_dot_4x8_packed_signed_accelerated: sdot_props
            .integer_dot_product4x8_bit_packed_signed_accelerated
            == vk::TRUE,
        cooperative_matrix_shapes: coop_shapes,
        qcom_cooperative_matrix_conversion: ext_set
            .contains("VK_QCOM_cooperative_matrix_conversion"),
        compute_limits: VulkanComputeLimits {
            max_compute_shared_memory_size: props.limits.max_compute_shared_memory_size,
            max_compute_work_group_count: props.limits.max_compute_work_group_count,
            max_compute_work_group_invocations: props.limits.max_compute_work_group_invocations,
            max_compute_work_group_size: props.limits.max_compute_work_group_size,
            max_storage_buffer_range: props.limits.max_storage_buffer_range,
            max_push_constants_size: props.limits.max_push_constants_size,
        },
        memory_heaps,
        identity_sha256: String::new(),
    };
    caps.recompute_identity_sha256().ok()?;
    Some((phys, compute_family, caps, ext_set, mem_properties))
}

#[cfg(windows)]
fn query_memory_heaps(
    instance: &ash::Instance,
    phys: vk::PhysicalDevice,
    ext_set: &BTreeSet<String>,
) -> (vk::PhysicalDeviceMemoryProperties, Vec<MemoryHeapBudget>) {
    let mut budget_props = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
    let has_budget = ext_set.contains("VK_EXT_memory_budget");
    let mut mem_props2 = vk::PhysicalDeviceMemoryProperties2::default();
    if has_budget {
        mem_props2 = mem_props2.push_next(&mut budget_props);
    }
    // SAFETY: Querying memory properties2 chain.
    unsafe { instance.get_physical_device_memory_properties2(phys, &mut mem_props2) };
    let mem_properties = mem_props2.memory_properties;

    let heap_count = usize::try_from(mem_properties.memory_heap_count).unwrap_or(0);
    let mut memory_heaps = Vec::with_capacity(heap_count);
    for (heap_index, heap) in mem_properties
        .memory_heaps
        .iter()
        .enumerate()
        .take(heap_count)
    {
        let (budget_bytes, usage_bytes) = if has_budget {
            (
                budget_props.heap_budget[heap_index],
                budget_props.heap_usage[heap_index],
            )
        } else {
            (heap.size, 0)
        };
        memory_heaps.push(MemoryHeapBudget {
            heap_index,
            size_bytes: heap.size,
            budget_bytes,
            usage_bytes,
            device_local: heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL),
        });
    }
    (mem_properties, memory_heaps)
}

#[cfg(windows)]
fn component_type_name(ty: vk::ComponentTypeKHR) -> String {
    match ty {
        vk::ComponentTypeKHR::FLOAT16 => "f16".to_owned(),
        vk::ComponentTypeKHR::FLOAT32 => "f32".to_owned(),
        vk::ComponentTypeKHR::FLOAT64 => "f64".to_owned(),
        vk::ComponentTypeKHR::SINT8 => "i8".to_owned(),
        vk::ComponentTypeKHR::SINT16 => "i16".to_owned(),
        vk::ComponentTypeKHR::SINT32 => "i32".to_owned(),
        vk::ComponentTypeKHR::UINT8 => "u8".to_owned(),
        vk::ComponentTypeKHR::UINT16 => "u16".to_owned(),
        vk::ComponentTypeKHR::UINT32 => "u32".to_owned(),
        other => format!("raw({})", other.as_raw()),
    }
}

#[cfg(windows)]
fn query_coop_mat_shapes(
    entry: &ash::Entry,
    instance: &ash::Instance,
    phys: vk::PhysicalDevice,
) -> Vec<CoopMatShape> {
    let loader = ash::khr::cooperative_matrix::Instance::new(entry, instance);
    // SAFETY: Querying supported `VK_KHR_cooperative_matrix` properties on physical device.
    let Ok(list) = (unsafe { loader.get_physical_device_cooperative_matrix_properties(phys) })
    else {
        return Vec::new();
    };
    list.into_iter()
        .take(64)
        .map(|item| CoopMatShape {
            m_size: item.m_size,
            n_size: item.n_size,
            k_size: item.k_size,
            a_type: component_type_name(item.a_type),
            b_type: component_type_name(item.b_type),
            c_type: component_type_name(item.c_type),
            result_type: component_type_name(item.result_type),
            scope: u32::try_from(item.scope.as_raw()).unwrap_or(0),
            saturating_accumulation: item.saturating_accumulation == vk::TRUE,
        })
        .collect()
}

#[cfg(windows)]
fn create_compute_device(
    instance: &ash::Instance,
    phys: vk::PhysicalDevice,
    queue_family: u32,
    caps: &VulkanDeviceCapabilities,
) -> Result<(ash::Device, bool), Error> {
    let priorities = [1.0_f32];
    let queue_info = [vk::DeviceQueueCreateInfo::default()
        .queue_family_index(queue_family)
        .queue_priorities(&priorities)];

    let enable_sdot = caps.shader_integer_dot_product;
    let mut sdot_feats = vk::PhysicalDeviceShaderIntegerDotProductFeatures::default()
        .shader_integer_dot_product(enable_sdot);
    let mut create_info = vk::DeviceCreateInfo::default().queue_create_infos(&queue_info);
    if enable_sdot {
        create_info = create_info.push_next(&mut sdot_feats);
    }

    // SAFETY: `create_info` and enabled feature structs remain live across `create_device`.
    let device = unsafe { instance.create_device(phys, &create_info, None) }
        .map_err(|err| Error::Vulkan(format!("vkCreateDevice failed: {err:?}")))?;
    Ok((device, enable_sdot))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forbidden_vk_env_names_are_rejected_case_insensitively() {
        assert!(is_forbidden_vk_env("vk_icd_filenames"));
        assert!(is_forbidden_vk_env("VK_LAYER_PATH"));
        assert!(!is_forbidden_vk_env("SYSTEMROOT"));
    }

    #[test]
    fn opens_native_qualcomm_adreno_and_queries_capabilities() -> Result<(), Error> {
        let device = AdrenoVulkanDevice::open()?;
        let caps = device.capabilities();
        assert_eq!(caps.vendor_id, QUALCOMM_VENDOR_ID);
        assert_eq!(caps.identity_sha256.len(), 64);
        assert!(!caps.device_name.contains("Direct3D12"));
        assert!(caps.compute_limits.max_compute_shared_memory_size >= 16_384);
        Ok(())
    }
}
