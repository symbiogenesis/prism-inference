//! Pure-Rust SPIR-V 1.6 compute shader emitter for Qualcomm Adreno Vulkan kernels.
//!
//! Emits validated SPIR-V 1.6 binary modules without external shader compilers (`glslang`,
//! `naga`, or `shaderc`), covering both `#39` (`PQ2_0`/`PTQ1_0` integer matrix-vector
//! projections with optional core SPIR-V 1.6 `OpSDot` integer dot product) and `#40` (fused
//! and 3-stage decomposed attention with explicit IEEE-754 binary16 probability rounding).
#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};

use bonsai_native_hash::Hasher;

use crate::{Error, ProjectionKernelPath};

const SPIRV_MAGIC: u32 = 0x0723_0203;
const SPIRV_VERSION_1_6: u32 = 0x0001_0600;
const WORKGROUP_X: u32 = 64;

const OP_EXTENSION: u16 = 10;
const OP_EXT_INST_IMPORT: u16 = 11;
const OP_EXT_INST: u16 = 12;
const OP_MEMORY_MODEL: u16 = 14;
const OP_ENTRY_POINT: u16 = 15;
const OP_EXECUTION_MODE: u16 = 16;
const OP_CAPABILITY: u16 = 17;
const OP_TYPE_VOID: u16 = 19;
const OP_TYPE_BOOL: u16 = 20;
const OP_TYPE_INT: u16 = 21;
const OP_TYPE_FLOAT: u16 = 22;
const OP_TYPE_VECTOR: u16 = 23;
const OP_TYPE_ARRAY: u16 = 28;
const OP_TYPE_RUNTIME_ARRAY: u16 = 29;
const OP_TYPE_STRUCT: u16 = 30;
const OP_TYPE_POINTER: u16 = 32;
const OP_TYPE_FUNCTION: u16 = 33;
const OP_CONSTANT: u16 = 43;
const OP_FUNCTION: u16 = 54;
const OP_FUNCTION_END: u16 = 56;
const OP_VARIABLE: u16 = 59;
const OP_LOAD: u16 = 61;
const OP_STORE: u16 = 62;
const OP_ACCESS_CHAIN: u16 = 65;
const OP_DECORATE: u16 = 71;
const OP_MEMBER_DECORATE: u16 = 72;
const OP_COMPOSITE_EXTRACT: u16 = 81;
const OP_CONVERT_F_TO_S: u16 = 110;
const OP_CONVERT_S_TO_F: u16 = 111;
const OP_CONVERT_U_TO_F: u16 = 112;
const OP_BITCAST: u16 = 124;
const OP_I_ADD: u16 = 128;
const OP_F_ADD: u16 = 129;
const OP_I_SUB: u16 = 130;
const OP_F_SUB: u16 = 131;
const OP_I_MUL: u16 = 132;
const OP_F_MUL: u16 = 133;
const OP_U_DIV: u16 = 134;
const OP_F_DIV: u16 = 136;
const OP_LOGICAL_OR: u16 = 166;
const OP_LOGICAL_AND: u16 = 167;
const OP_SELECT: u16 = 169;
const OP_I_EQUAL: u16 = 170;
const OP_I_NOT_EQUAL: u16 = 171;
const OP_U_GREATER_THAN_EQUAL: u16 = 174;
const OP_U_LESS_THAN: u16 = 176;
const OP_F_ORD_EQUAL: u16 = 180;
const OP_F_ORD_LESS_THAN: u16 = 184;
const OP_F_ORD_GREATER_THAN: u16 = 186;
const OP_F_ORD_GREATER_THAN_EQUAL: u16 = 190;
const OP_SHIFT_RIGHT_LOGICAL: u16 = 194;
const OP_SHIFT_RIGHT_ARITHMETIC: u16 = 195;
const OP_SHIFT_LEFT_LOGICAL: u16 = 196;
const OP_BITWISE_OR: u16 = 197;
const OP_BITWISE_XOR: u16 = 198;
const OP_BITWISE_AND: u16 = 199;
const OP_CONTROL_BARRIER: u16 = 224;
const OP_PHI: u16 = 245;
const OP_LOOP_MERGE: u16 = 246;
const OP_SELECTION_MERGE: u16 = 247;
const OP_LABEL: u16 = 248;
const OP_BRANCH: u16 = 249;
const OP_BRANCH_CONDITIONAL: u16 = 250;
const OP_RETURN: u16 = 253;
const OP_S_DOT: u16 = 4450;

const CAPABILITY_SHADER: u32 = 1;
const CAPABILITY_FLOAT64: u32 = 10;
const CAPABILITY_INT64: u32 = 11;
const CAPABILITY_DOT_PRODUCT_INPUT_4X8_BIT_PACKED: u32 = 6018;
const CAPABILITY_DOT_PRODUCT: u32 = 6019;

const GLSL_FABS: u32 = 4;
const GLSL_FLOOR: u32 = 8;
const GLSL_EXP: u32 = 27;
const GLSL_LOG: u32 = 28;
const GLSL_SQRT: u32 = 31;
const GLSL_SMIN: u32 = 39;
const GLSL_FMAX: u32 = 40;
const GLSL_SMAX: u32 = 42;
const GLSL_FMA: u32 = 50;
const GLSL_UNPACK_HALF_2X16: u32 = 62;

const EXEC_MODEL_GL_COMPUTE: u32 = 5;
const EXEC_MODE_LOCAL_SIZE: u32 = 17;
const ADDRESSING_LOGICAL: u32 = 0;
const MEMORY_MODEL_GLSL450: u32 = 1;

const STORAGE_INPUT: u32 = 1;
const STORAGE_WORKGROUP: u32 = 4;
const STORAGE_FUNCTION: u32 = 7;
const STORAGE_PUSH_CONSTANT: u32 = 9;
const STORAGE_STORAGE_BUFFER: u32 = 12;

const DECORATION_BLOCK: u32 = 2;
const DECORATION_ARRAY_STRIDE: u32 = 6;
const DECORATION_BUILT_IN: u32 = 11;
const DECORATION_NON_WRITABLE: u32 = 24;
const DECORATION_BINDING: u32 = 33;
const DECORATION_DESCRIPTOR_SET: u32 = 34;
const DECORATION_OFFSET: u32 = 35;
const DECORATION_NO_CONTRACTION: u32 = 42;

const BUILTIN_WORKGROUP_ID: u32 = 26;
const BUILTIN_LOCAL_INVOCATION_ID: u32 = 27;
const BUILTIN_GLOBAL_INVOCATION_ID: u32 = 28;

const SCOPE_WORKGROUP: u32 = 2;
const SEMANTICS_ACQUIRE_RELEASE_WORKGROUP: u32 = 0x108;
const PACKED_FORMAT_4X8_BIT: u32 = 0;

#[derive(Clone, Copy)]
enum ElemType {
    U32,
    F32,
}

#[derive(Clone, Copy)]
enum AccessMode {
    ReadOnly,
    ReadWrite,
}

struct ShaderBuilder {
    next_id: u32,
    capabilities: Vec<u32>,
    ext_inst_imports: Vec<u32>,
    memory_model: Vec<u32>,
    entry_points: Vec<u32>,
    execution_modes: Vec<u32>,
    annotations: Vec<u32>,
    types_consts_globals: Vec<u32>,
    fn_header: Vec<u32>,
    fn_body: Vec<u32>,

    glsl_id: u32,
    void_t: u32,
    bool_t: u32,
    u32_t: u32,
    i32_t: u32,
    f32_t: u32,
    vec2f32_t: u32,
    vec3u32_t: u32,

    ptr_input_vec3u32_t: u32,
    ptr_input_u32_t: u32,
    ptr_fn_f32_t: Option<u32>,
    ptr_fn_u32_t: Option<u32>,
    ptr_push_u32_t: Option<u32>,
    ptr_sb_u32_t: Option<u32>,
    ptr_sb_f32_t: Option<u32>,
    ptr_wg_f32_t: Option<u32>,
    ptr_wg_u32_t: Option<u32>,

    rt_u32_t: Option<u32>,
    rt_f32_t: Option<u32>,
    ro_u32_buf_t: Option<u32>,
    rw_u32_buf_t: Option<u32>,
    ro_f32_buf_t: Option<u32>,
    rw_f32_buf_t: Option<u32>,

    u32_consts: BTreeMap<u32, u32>,
    i32_consts: BTreeMap<u32, u32>,
    f32_consts: BTreeMap<u32, u32>,

    main_fn_id: u32,
    current_label: u32,
    entry_interface: Vec<u32>,
    gid_var: Option<u32>,
    lid_var: Option<u32>,
    wid_var: Option<u32>,
}

fn push_inst(out: &mut Vec<u32>, opcode: u16, operands: &[u32]) {
    let words = u32::try_from(operands.len() + 1).unwrap_or(1);
    out.push((words << 16) | u32::from(opcode));
    out.extend_from_slice(operands);
}

fn encode_string(text: &str) -> Vec<u32> {
    let bytes = text.as_bytes();
    let total = bytes.len() + 1;
    let word_count = total.div_ceil(4);
    let mut words = vec![0_u32; word_count];
    for (index, &byte) in bytes.iter().enumerate() {
        let shift = (index % 4) * 8;
        words[index / 4] |= u32::from(byte) << shift;
    }
    words
}

impl ShaderBuilder {
    fn new(use_sdot: bool, push_u32_count: usize) -> (Self, Option<u32>) {
        let mut this = Self {
            next_id: 1,
            capabilities: Vec::new(),
            ext_inst_imports: Vec::new(),
            memory_model: Vec::new(),
            entry_points: Vec::new(),
            execution_modes: Vec::new(),
            annotations: Vec::new(),
            types_consts_globals: Vec::new(),
            fn_header: Vec::new(),
            fn_body: Vec::new(),
            glsl_id: 0,
            void_t: 0,
            bool_t: 0,
            u32_t: 0,
            i32_t: 0,
            f32_t: 0,
            vec2f32_t: 0,
            vec3u32_t: 0,
            ptr_input_vec3u32_t: 0,
            ptr_input_u32_t: 0,
            ptr_fn_f32_t: None,
            ptr_fn_u32_t: None,
            ptr_push_u32_t: None,
            ptr_sb_u32_t: None,
            ptr_sb_f32_t: None,
            ptr_wg_f32_t: None,
            ptr_wg_u32_t: None,
            rt_u32_t: None,
            rt_f32_t: None,
            ro_u32_buf_t: None,
            rw_u32_buf_t: None,
            ro_f32_buf_t: None,
            rw_f32_buf_t: None,
            u32_consts: BTreeMap::new(),
            i32_consts: BTreeMap::new(),
            f32_consts: BTreeMap::new(),
            main_fn_id: 0,
            current_label: 0,
            entry_interface: Vec::new(),
            gid_var: None,
            lid_var: None,
            wid_var: None,
        };

        push_inst(&mut this.capabilities, OP_CAPABILITY, &[CAPABILITY_SHADER]);
        if use_sdot {
            push_inst(
                &mut this.capabilities,
                OP_CAPABILITY,
                &[CAPABILITY_DOT_PRODUCT],
            );
            push_inst(
                &mut this.capabilities,
                OP_CAPABILITY,
                &[CAPABILITY_DOT_PRODUCT_INPUT_4X8_BIT_PACKED],
            );
        }

        let glsl_id = this.alloc_id();
        this.glsl_id = glsl_id;
        let mut ext_import = vec![glsl_id];
        ext_import.extend(encode_string("GLSL.std.450"));
        push_inst(&mut this.ext_inst_imports, OP_EXT_INST_IMPORT, &ext_import);
        push_inst(
            &mut this.memory_model,
            OP_MEMORY_MODEL,
            &[ADDRESSING_LOGICAL, MEMORY_MODEL_GLSL450],
        );

        this.init_core_types();
        let push_var = this.init_push_constants(push_u32_count);

        let fn_t = this.alloc_id();
        push_inst(
            &mut this.types_consts_globals,
            OP_TYPE_FUNCTION,
            &[fn_t, this.void_t],
        );

        let main_id = this.alloc_id();
        this.main_fn_id = main_id;
        let entry_lbl = this.alloc_id();
        this.current_label = entry_lbl;
        push_inst(
            &mut this.fn_header,
            OP_FUNCTION,
            &[this.void_t, main_id, 0, fn_t],
        );
        push_inst(&mut this.fn_header, OP_LABEL, &[entry_lbl]);

        (this, push_var)
    }

    fn init_core_types(&mut self) {
        self.void_t = self.alloc_id();
        push_inst(&mut self.types_consts_globals, OP_TYPE_VOID, &[self.void_t]);
        self.bool_t = self.alloc_id();
        push_inst(&mut self.types_consts_globals, OP_TYPE_BOOL, &[self.bool_t]);
        self.u32_t = self.alloc_id();
        push_inst(
            &mut self.types_consts_globals,
            OP_TYPE_INT,
            &[self.u32_t, 32, 0],
        );
        self.i32_t = self.alloc_id();
        push_inst(
            &mut self.types_consts_globals,
            OP_TYPE_INT,
            &[self.i32_t, 32, 1],
        );
        self.f32_t = self.alloc_id();
        push_inst(
            &mut self.types_consts_globals,
            OP_TYPE_FLOAT,
            &[self.f32_t, 32],
        );
        self.vec2f32_t = self.alloc_id();
        push_inst(
            &mut self.types_consts_globals,
            OP_TYPE_VECTOR,
            &[self.vec2f32_t, self.f32_t, 2],
        );
        self.vec3u32_t = self.alloc_id();
        push_inst(
            &mut self.types_consts_globals,
            OP_TYPE_VECTOR,
            &[self.vec3u32_t, self.u32_t, 3],
        );
        self.ptr_input_vec3u32_t = self.alloc_id();
        push_inst(
            &mut self.types_consts_globals,
            OP_TYPE_POINTER,
            &[self.ptr_input_vec3u32_t, STORAGE_INPUT, self.vec3u32_t],
        );
        self.ptr_input_u32_t = self.alloc_id();
        push_inst(
            &mut self.types_consts_globals,
            OP_TYPE_POINTER,
            &[self.ptr_input_u32_t, STORAGE_INPUT, self.u32_t],
        );
    }

    fn init_push_constants(&mut self, push_u32_count: usize) -> Option<u32> {
        if push_u32_count == 0 {
            return None;
        }
        let struct_t = self.alloc_id();
        let mut operands = vec![struct_t];
        for _ in 0..push_u32_count {
            operands.push(self.u32_t);
        }
        push_inst(&mut self.types_consts_globals, OP_TYPE_STRUCT, &operands);
        push_inst(
            &mut self.annotations,
            OP_DECORATE,
            &[struct_t, DECORATION_BLOCK],
        );
        for idx in 0..push_u32_count {
            let idx_u32 = u32::try_from(idx).unwrap_or(0);
            push_inst(
                &mut self.annotations,
                OP_MEMBER_DECORATE,
                &[struct_t, idx_u32, DECORATION_OFFSET, idx_u32 * 4],
            );
        }
        let ptr_push_struct = self.alloc_id();
        push_inst(
            &mut self.types_consts_globals,
            OP_TYPE_POINTER,
            &[ptr_push_struct, STORAGE_PUSH_CONSTANT, struct_t],
        );
        let ptr_push_u32 = self.alloc_id();
        push_inst(
            &mut self.types_consts_globals,
            OP_TYPE_POINTER,
            &[ptr_push_u32, STORAGE_PUSH_CONSTANT, self.u32_t],
        );
        self.ptr_push_u32_t = Some(ptr_push_u32);
        let var_id = self.alloc_id();
        push_inst(
            &mut self.types_consts_globals,
            OP_VARIABLE,
            &[ptr_push_struct, var_id, STORAGE_PUSH_CONSTANT],
        );
        self.record_entry_interface(var_id);
        Some(var_id)
    }

    fn record_entry_interface(&mut self, var_id: u32) {
        if !self.entry_interface.contains(&var_id) {
            self.entry_interface.push(var_id);
        }
    }

    const fn alloc_id(&mut self) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    fn finish(mut self) -> Vec<u32> {
        let mut ep = vec![EXEC_MODEL_GL_COMPUTE, self.main_fn_id];
        ep.extend(encode_string("main"));
        ep.extend(self.entry_interface.iter().copied());
        push_inst(&mut self.entry_points, OP_ENTRY_POINT, &ep);
        push_inst(
            &mut self.execution_modes,
            OP_EXECUTION_MODE,
            &[self.main_fn_id, EXEC_MODE_LOCAL_SIZE, WORKGROUP_X, 1, 1],
        );
        push_inst(&mut self.fn_body, OP_RETURN, &[]);
        push_inst(&mut self.fn_body, OP_FUNCTION_END, &[]);

        let mut words = Vec::with_capacity(
            5 + self.capabilities.len()
                + self.ext_inst_imports.len()
                + self.memory_model.len()
                + self.entry_points.len()
                + self.execution_modes.len()
                + self.annotations.len()
                + self.types_consts_globals.len()
                + self.fn_header.len()
                + self.fn_body.len(),
        );
        words.push(SPIRV_MAGIC);
        words.push(SPIRV_VERSION_1_6);
        words.push(0);
        words.push(self.next_id);
        words.push(0);
        words.extend(self.capabilities);
        words.extend(self.ext_inst_imports);
        words.extend(self.memory_model);
        words.extend(self.entry_points);
        words.extend(self.execution_modes);
        words.extend(self.annotations);
        words.extend(self.types_consts_globals);
        words.extend(self.fn_header);
        words.extend(self.fn_body);
        words
    }

    fn const_u32(&mut self, value: u32) -> u32 {
        if let Some(&id) = self.u32_consts.get(&value) {
            return id;
        }
        let id = self.alloc_id();
        push_inst(
            &mut self.types_consts_globals,
            OP_CONSTANT,
            &[self.u32_t, id, value],
        );
        self.u32_consts.insert(value, id);
        id
    }

    fn const_i32(&mut self, value: i32) -> u32 {
        let bits = value.cast_unsigned();
        if let Some(&id) = self.i32_consts.get(&bits) {
            return id;
        }
        let id = self.alloc_id();
        push_inst(
            &mut self.types_consts_globals,
            OP_CONSTANT,
            &[self.i32_t, id, bits],
        );
        self.i32_consts.insert(bits, id);
        id
    }

    fn const_f32(&mut self, value: f32) -> u32 {
        let bits = value.to_bits();
        if let Some(&id) = self.f32_consts.get(&bits) {
            return id;
        }
        let id = self.alloc_id();
        push_inst(
            &mut self.types_consts_globals,
            OP_CONSTANT,
            &[self.f32_t, id, bits],
        );
        self.f32_consts.insert(bits, id);
        id
    }

    fn global_invocation_id(&mut self) -> u32 {
        if let Some(id) = self.gid_var {
            return id;
        }
        let var = self.alloc_id();
        push_inst(
            &mut self.types_consts_globals,
            OP_VARIABLE,
            &[self.ptr_input_vec3u32_t, var, STORAGE_INPUT],
        );
        push_inst(
            &mut self.annotations,
            OP_DECORATE,
            &[var, DECORATION_BUILT_IN, BUILTIN_GLOBAL_INVOCATION_ID],
        );
        self.record_entry_interface(var);
        self.gid_var = Some(var);
        var
    }

    fn local_invocation_id(&mut self) -> u32 {
        if let Some(id) = self.lid_var {
            return id;
        }
        let var = self.alloc_id();
        push_inst(
            &mut self.types_consts_globals,
            OP_VARIABLE,
            &[self.ptr_input_vec3u32_t, var, STORAGE_INPUT],
        );
        push_inst(
            &mut self.annotations,
            OP_DECORATE,
            &[var, DECORATION_BUILT_IN, BUILTIN_LOCAL_INVOCATION_ID],
        );
        self.record_entry_interface(var);
        self.lid_var = Some(var);
        var
    }

    fn workgroup_id(&mut self) -> u32 {
        if let Some(id) = self.wid_var {
            return id;
        }
        let var = self.alloc_id();
        push_inst(
            &mut self.types_consts_globals,
            OP_VARIABLE,
            &[self.ptr_input_vec3u32_t, var, STORAGE_INPUT],
        );
        push_inst(
            &mut self.annotations,
            OP_DECORATE,
            &[var, DECORATION_BUILT_IN, BUILTIN_WORKGROUP_ID],
        );
        self.record_entry_interface(var);
        self.wid_var = Some(var);
        var
    }

    fn load_input_component(&mut self, builtin_var: u32, comp: u32) -> u32 {
        let comp_const = self.const_u32(comp);
        let ptr = self.alloc_id();
        push_inst(
            &mut self.fn_body,
            OP_ACCESS_CHAIN,
            &[self.ptr_input_u32_t, ptr, builtin_var, comp_const],
        );
        let val = self.alloc_id();
        push_inst(&mut self.fn_body, OP_LOAD, &[self.u32_t, val, ptr]);
        val
    }

    fn runtime_array_type(&mut self, elem: ElemType) -> u32 {
        match elem {
            ElemType::U32 => {
                if let Some(id) = self.rt_u32_t {
                    return id;
                }
                let rt = self.alloc_id();
                push_inst(
                    &mut self.types_consts_globals,
                    OP_TYPE_RUNTIME_ARRAY,
                    &[rt, self.u32_t],
                );
                push_inst(
                    &mut self.annotations,
                    OP_DECORATE,
                    &[rt, DECORATION_ARRAY_STRIDE, 4],
                );
                self.rt_u32_t = Some(rt);
                rt
            }
            ElemType::F32 => {
                if let Some(id) = self.rt_f32_t {
                    return id;
                }
                let rt = self.alloc_id();
                push_inst(
                    &mut self.types_consts_globals,
                    OP_TYPE_RUNTIME_ARRAY,
                    &[rt, self.f32_t],
                );
                push_inst(
                    &mut self.annotations,
                    OP_DECORATE,
                    &[rt, DECORATION_ARRAY_STRIDE, 4],
                );
                self.rt_f32_t = Some(rt);
                rt
            }
        }
    }

    fn storage_buffer(&mut self, binding: u32, elem: ElemType, access: AccessMode) -> u32 {
        let rt_t = self.runtime_array_type(elem);
        let cache = match (elem, access) {
            (ElemType::U32, AccessMode::ReadOnly) => &mut self.ro_u32_buf_t,
            (ElemType::F32, AccessMode::ReadOnly) => &mut self.ro_f32_buf_t,
            (ElemType::F32, AccessMode::ReadWrite) => &mut self.rw_f32_buf_t,
            (ElemType::U32, AccessMode::ReadWrite) => &mut self.rw_u32_buf_t,
        };
        let struct_t = if let Some(id) = *cache {
            id
        } else {
            let s = self.make_sb_struct(rt_t, matches!(access, AccessMode::ReadOnly));
            *match (elem, access) {
                (ElemType::U32, AccessMode::ReadOnly) => &mut self.ro_u32_buf_t,
                (ElemType::F32, AccessMode::ReadOnly) => &mut self.ro_f32_buf_t,
                (ElemType::F32, AccessMode::ReadWrite) => &mut self.rw_f32_buf_t,
                (ElemType::U32, AccessMode::ReadWrite) => &mut self.rw_u32_buf_t,
            } = Some(s);
            s
        };

        let ptr_struct = self.alloc_id();
        push_inst(
            &mut self.types_consts_globals,
            OP_TYPE_POINTER,
            &[ptr_struct, STORAGE_STORAGE_BUFFER, struct_t],
        );
        let var = self.alloc_id();
        push_inst(
            &mut self.types_consts_globals,
            OP_VARIABLE,
            &[ptr_struct, var, STORAGE_STORAGE_BUFFER],
        );
        push_inst(
            &mut self.annotations,
            OP_DECORATE,
            &[var, DECORATION_DESCRIPTOR_SET, 0],
        );
        push_inst(
            &mut self.annotations,
            OP_DECORATE,
            &[var, DECORATION_BINDING, binding],
        );
        self.record_entry_interface(var);
        var
    }

    fn make_sb_struct(&mut self, rt_t: u32, read_only: bool) -> u32 {
        let struct_t = self.alloc_id();
        push_inst(
            &mut self.types_consts_globals,
            OP_TYPE_STRUCT,
            &[struct_t, rt_t],
        );
        push_inst(
            &mut self.annotations,
            OP_DECORATE,
            &[struct_t, DECORATION_BLOCK],
        );
        push_inst(
            &mut self.annotations,
            OP_MEMBER_DECORATE,
            &[struct_t, 0, DECORATION_OFFSET, 0],
        );
        if read_only {
            push_inst(
                &mut self.annotations,
                OP_MEMBER_DECORATE,
                &[struct_t, 0, DECORATION_NON_WRITABLE],
            );
        }
        struct_t
    }

    fn ptr_sb_u32(&mut self) -> u32 {
        if let Some(id) = self.ptr_sb_u32_t {
            return id;
        }
        let id = self.alloc_id();
        push_inst(
            &mut self.types_consts_globals,
            OP_TYPE_POINTER,
            &[id, STORAGE_STORAGE_BUFFER, self.u32_t],
        );
        self.ptr_sb_u32_t = Some(id);
        id
    }

    fn ptr_sb_f32(&mut self) -> u32 {
        if let Some(id) = self.ptr_sb_f32_t {
            return id;
        }
        let id = self.alloc_id();
        push_inst(
            &mut self.types_consts_globals,
            OP_TYPE_POINTER,
            &[id, STORAGE_STORAGE_BUFFER, self.f32_t],
        );
        self.ptr_sb_f32_t = Some(id);
        id
    }

    fn workgroup_f32_array(&mut self, len: u32) -> u32 {
        let len_c = self.const_u32(len);
        let arr_t = self.alloc_id();
        push_inst(
            &mut self.types_consts_globals,
            OP_TYPE_ARRAY,
            &[arr_t, self.f32_t, len_c],
        );
        let ptr_arr = self.alloc_id();
        push_inst(
            &mut self.types_consts_globals,
            OP_TYPE_POINTER,
            &[ptr_arr, STORAGE_WORKGROUP, arr_t],
        );
        let var = self.alloc_id();
        push_inst(
            &mut self.types_consts_globals,
            OP_VARIABLE,
            &[ptr_arr, var, STORAGE_WORKGROUP],
        );
        self.record_entry_interface(var);
        var
    }

    fn workgroup_u32_array(&mut self, len: u32) -> u32 {
        let len_c = self.const_u32(len);
        let arr_t = self.alloc_id();
        push_inst(
            &mut self.types_consts_globals,
            OP_TYPE_ARRAY,
            &[arr_t, self.u32_t, len_c],
        );
        let ptr_arr = self.alloc_id();
        push_inst(
            &mut self.types_consts_globals,
            OP_TYPE_POINTER,
            &[ptr_arr, STORAGE_WORKGROUP, arr_t],
        );
        let var = self.alloc_id();
        push_inst(
            &mut self.types_consts_globals,
            OP_VARIABLE,
            &[ptr_arr, var, STORAGE_WORKGROUP],
        );
        self.record_entry_interface(var);
        var
    }

    fn ptr_wg_u32(&mut self) -> u32 {
        if let Some(id) = self.ptr_wg_u32_t {
            return id;
        }
        let id = self.alloc_id();
        push_inst(
            &mut self.types_consts_globals,
            OP_TYPE_POINTER,
            &[id, STORAGE_WORKGROUP, self.u32_t],
        );
        self.ptr_wg_u32_t = Some(id);
        id
    }

    fn load_wg_u32(&mut self, arr_var: u32, idx: u32) -> u32 {
        let ptr_t = self.ptr_wg_u32();
        let ptr = self.alloc_id();
        push_inst(
            &mut self.fn_body,
            OP_ACCESS_CHAIN,
            &[ptr_t, ptr, arr_var, idx],
        );
        let val = self.alloc_id();
        push_inst(&mut self.fn_body, OP_LOAD, &[self.u32_t, val, ptr]);
        val
    }

    fn store_wg_u32(&mut self, arr_var: u32, idx: u32, val: u32) {
        let ptr_t = self.ptr_wg_u32();
        let ptr = self.alloc_id();
        push_inst(
            &mut self.fn_body,
            OP_ACCESS_CHAIN,
            &[ptr_t, ptr, arr_var, idx],
        );
        push_inst(&mut self.fn_body, OP_STORE, &[ptr, val]);
    }

    fn store_sb_u32(&mut self, buf_var: u32, idx: u32, val: u32) {
        let ptr_t = self.ptr_sb_u32();
        let zero = self.const_u32(0);
        let ptr = self.alloc_id();
        push_inst(
            &mut self.fn_body,
            OP_ACCESS_CHAIN,
            &[ptr_t, ptr, buf_var, zero, idx],
        );
        push_inst(&mut self.fn_body, OP_STORE, &[ptr, val]);
    }

    fn local_u32_var(&mut self) -> u32 {
        let ptr_t = if let Some(id) = self.ptr_fn_u32_t {
            id
        } else {
            let id = self.alloc_id();
            push_inst(
                &mut self.types_consts_globals,
                OP_TYPE_POINTER,
                &[id, STORAGE_FUNCTION, self.u32_t],
            );
            self.ptr_fn_u32_t = Some(id);
            id
        };
        let var = self.alloc_id();
        push_inst(
            &mut self.fn_header,
            OP_VARIABLE,
            &[ptr_t, var, STORAGE_FUNCTION],
        );
        var
    }

    fn load_var_u32(&mut self, var: u32) -> u32 {
        let val = self.alloc_id();
        push_inst(&mut self.fn_body, OP_LOAD, &[self.u32_t, val, var]);
        val
    }

    fn ptr_wg_f32(&mut self) -> u32 {
        if let Some(id) = self.ptr_wg_f32_t {
            return id;
        }
        let id = self.alloc_id();
        push_inst(
            &mut self.types_consts_globals,
            OP_TYPE_POINTER,
            &[id, STORAGE_WORKGROUP, self.f32_t],
        );
        self.ptr_wg_f32_t = Some(id);
        id
    }

    fn local_f32_var(&mut self) -> u32 {
        let ptr_t = if let Some(id) = self.ptr_fn_f32_t {
            id
        } else {
            let id = self.alloc_id();
            push_inst(
                &mut self.types_consts_globals,
                OP_TYPE_POINTER,
                &[id, STORAGE_FUNCTION, self.f32_t],
            );
            self.ptr_fn_f32_t = Some(id);
            id
        };
        let var = self.alloc_id();
        push_inst(
            &mut self.fn_header,
            OP_VARIABLE,
            &[ptr_t, var, STORAGE_FUNCTION],
        );
        var
    }

    fn load_var_f32(&mut self, var: u32) -> u32 {
        let val = self.alloc_id();
        push_inst(&mut self.fn_body, OP_LOAD, &[self.f32_t, val, var]);
        val
    }

    fn store_var(&mut self, var: u32, val: u32) {
        push_inst(&mut self.fn_body, OP_STORE, &[var, val]);
    }

    fn load_push(&mut self, push_var: u32, member_idx: u32) -> u32 {
        let ptr_t = self.ptr_push_u32_t.unwrap_or(0);
        let idx_c = self.const_u32(member_idx);
        let ptr = self.alloc_id();
        push_inst(
            &mut self.fn_body,
            OP_ACCESS_CHAIN,
            &[ptr_t, ptr, push_var, idx_c],
        );
        let val = self.alloc_id();
        push_inst(&mut self.fn_body, OP_LOAD, &[self.u32_t, val, ptr]);
        val
    }

    fn load_sb_u32(&mut self, buf_var: u32, idx: u32) -> u32 {
        let ptr_t = self.ptr_sb_u32();
        let zero = self.const_u32(0);
        let ptr = self.alloc_id();
        push_inst(
            &mut self.fn_body,
            OP_ACCESS_CHAIN,
            &[ptr_t, ptr, buf_var, zero, idx],
        );
        let val = self.alloc_id();
        push_inst(&mut self.fn_body, OP_LOAD, &[self.u32_t, val, ptr]);
        val
    }

    fn load_sb_f32(&mut self, buf_var: u32, idx: u32) -> u32 {
        let ptr_t = self.ptr_sb_f32();
        let zero = self.const_u32(0);
        let ptr = self.alloc_id();
        push_inst(
            &mut self.fn_body,
            OP_ACCESS_CHAIN,
            &[ptr_t, ptr, buf_var, zero, idx],
        );
        let val = self.alloc_id();
        push_inst(&mut self.fn_body, OP_LOAD, &[self.f32_t, val, ptr]);
        val
    }

    fn store_sb_f32(&mut self, buf_var: u32, idx: u32, val: u32) {
        let ptr_t = self.ptr_sb_f32();
        let zero = self.const_u32(0);
        let ptr = self.alloc_id();
        push_inst(
            &mut self.fn_body,
            OP_ACCESS_CHAIN,
            &[ptr_t, ptr, buf_var, zero, idx],
        );
        push_inst(&mut self.fn_body, OP_STORE, &[ptr, val]);
    }

    fn load_wg_f32(&mut self, arr_var: u32, idx: u32) -> u32 {
        let ptr_t = self.ptr_wg_f32();
        let ptr = self.alloc_id();
        push_inst(
            &mut self.fn_body,
            OP_ACCESS_CHAIN,
            &[ptr_t, ptr, arr_var, idx],
        );
        let val = self.alloc_id();
        push_inst(&mut self.fn_body, OP_LOAD, &[self.f32_t, val, ptr]);
        val
    }

    fn store_wg_f32(&mut self, arr_var: u32, idx: u32, val: u32) {
        let ptr_t = self.ptr_wg_f32();
        let ptr = self.alloc_id();
        push_inst(
            &mut self.fn_body,
            OP_ACCESS_CHAIN,
            &[ptr_t, ptr, arr_var, idx],
        );
        push_inst(&mut self.fn_body, OP_STORE, &[ptr, val]);
    }

    fn binop(&mut self, opcode: u16, res_t: u32, lhs: u32, rhs: u32) -> u32 {
        let id = self.alloc_id();
        push_inst(&mut self.fn_body, opcode, &[res_t, id, lhs, rhs]);
        id
    }

    fn uadd(&mut self, a: u32, b: u32) -> u32 {
        self.binop(OP_I_ADD, self.u32_t, a, b)
    }

    fn umul(&mut self, a: u32, b: u32) -> u32 {
        self.binop(OP_I_MUL, self.u32_t, a, b)
    }

    fn udiv(&mut self, a: u32, b: u32) -> u32 {
        self.binop(OP_U_DIV, self.u32_t, a, b)
    }

    fn iadd(&mut self, a: u32, b: u32) -> u32 {
        self.binop(OP_I_ADD, self.i32_t, a, b)
    }

    fn isub(&mut self, a: u32, b: u32) -> u32 {
        self.binop(OP_I_SUB, self.i32_t, a, b)
    }

    fn imul(&mut self, a: u32, b: u32) -> u32 {
        self.binop(OP_I_MUL, self.i32_t, a, b)
    }

    fn uand(&mut self, a: u32, b: u32) -> u32 {
        self.binop(OP_BITWISE_AND, self.u32_t, a, b)
    }

    fn uor(&mut self, a: u32, b: u32) -> u32 {
        self.binop(OP_BITWISE_OR, self.u32_t, a, b)
    }

    fn uxor(&mut self, a: u32, b: u32) -> u32 {
        self.binop(OP_BITWISE_XOR, self.u32_t, a, b)
    }

    fn ushl(&mut self, a: u32, b: u32) -> u32 {
        self.binop(OP_SHIFT_LEFT_LOGICAL, self.u32_t, a, b)
    }

    fn usub(&mut self, a: u32, b: u32) -> u32 {
        self.binop(OP_I_SUB, self.u32_t, a, b)
    }

    fn ushr(&mut self, a: u32, b: u32) -> u32 {
        self.binop(OP_SHIFT_RIGHT_LOGICAL, self.u32_t, a, b)
    }

    fn ishl(&mut self, a: u32, b: u32) -> u32 {
        self.binop(OP_SHIFT_LEFT_LOGICAL, self.i32_t, a, b)
    }

    fn ishr_arith(&mut self, a: u32, b: u32) -> u32 {
        self.binop(OP_SHIFT_RIGHT_ARITHMETIC, self.i32_t, a, b)
    }

    fn u_lt(&mut self, a: u32, b: u32) -> u32 {
        self.binop(OP_U_LESS_THAN, self.bool_t, a, b)
    }

    fn u_ge(&mut self, a: u32, b: u32) -> u32 {
        self.binop(OP_U_GREATER_THAN_EQUAL, self.bool_t, a, b)
    }

    fn u_eq(&mut self, a: u32, b: u32) -> u32 {
        self.binop(OP_I_EQUAL, self.bool_t, a, b)
    }

    fn f_eq(&mut self, a: u32, b: u32) -> u32 {
        self.binop(OP_F_ORD_EQUAL, self.bool_t, a, b)
    }

    fn f_lt(&mut self, a: u32, b: u32) -> u32 {
        self.binop(OP_F_ORD_LESS_THAN, self.bool_t, a, b)
    }

    fn f_gt(&mut self, a: u32, b: u32) -> u32 {
        self.binop(OP_F_ORD_GREATER_THAN, self.bool_t, a, b)
    }

    fn f_ge(&mut self, a: u32, b: u32) -> u32 {
        self.binop(OP_F_ORD_GREATER_THAN_EQUAL, self.bool_t, a, b)
    }

    fn u_ne(&mut self, a: u32, b: u32) -> u32 {
        self.binop(OP_I_NOT_EQUAL, self.bool_t, a, b)
    }

    fn logical_and(&mut self, a: u32, b: u32) -> u32 {
        self.binop(OP_LOGICAL_AND, self.bool_t, a, b)
    }

    fn logical_or(&mut self, a: u32, b: u32) -> u32 {
        self.binop(OP_LOGICAL_OR, self.bool_t, a, b)
    }

    fn select(&mut self, ty: u32, cond: u32, t: u32, f: u32) -> u32 {
        let id = self.alloc_id();
        push_inst(&mut self.fn_body, OP_SELECT, &[ty, id, cond, t, f]);
        id
    }

    fn bitcast(&mut self, dst_t: u32, src: u32) -> u32 {
        let id = self.alloc_id();
        push_inst(&mut self.fn_body, OP_BITCAST, &[dst_t, id, src]);
        id
    }

    fn i32_to_f32(&mut self, val: u32) -> u32 {
        let id = self.alloc_id();
        push_inst(&mut self.fn_body, OP_CONVERT_S_TO_F, &[self.f32_t, id, val]);
        id
    }

    fn u32_to_f32(&mut self, val: u32) -> u32 {
        let id = self.alloc_id();
        push_inst(&mut self.fn_body, OP_CONVERT_U_TO_F, &[self.f32_t, id, val]);
        id
    }

    fn f32_to_i32(&mut self, val: u32) -> u32 {
        let id = self.alloc_id();
        push_inst(&mut self.fn_body, OP_CONVERT_F_TO_S, &[self.i32_t, id, val]);
        id
    }

    fn fadd_nc(&mut self, a: u32, b: u32) -> u32 {
        let id = self.binop(OP_F_ADD, self.f32_t, a, b);
        push_inst(
            &mut self.annotations,
            OP_DECORATE,
            &[id, DECORATION_NO_CONTRACTION],
        );
        id
    }

    fn fsub_nc(&mut self, a: u32, b: u32) -> u32 {
        let id = self.binop(OP_F_SUB, self.f32_t, a, b);
        push_inst(
            &mut self.annotations,
            OP_DECORATE,
            &[id, DECORATION_NO_CONTRACTION],
        );
        id
    }

    fn fmul_nc(&mut self, a: u32, b: u32) -> u32 {
        let id = self.binop(OP_F_MUL, self.f32_t, a, b);
        push_inst(
            &mut self.annotations,
            OP_DECORATE,
            &[id, DECORATION_NO_CONTRACTION],
        );
        id
    }

    fn fdiv(&mut self, a: u32, b: u32) -> u32 {
        self.binop(OP_F_DIV, self.f32_t, a, b)
    }

    fn glsl1(&mut self, res_t: u32, glsl_op: u32, a: u32) -> u32 {
        let id = self.alloc_id();
        push_inst(
            &mut self.fn_body,
            OP_EXT_INST,
            &[res_t, id, self.glsl_id, glsl_op, a],
        );
        id
    }

    fn glsl2(&mut self, res_t: u32, glsl_op: u32, a: u32, b: u32) -> u32 {
        let id = self.alloc_id();
        push_inst(
            &mut self.fn_body,
            OP_EXT_INST,
            &[res_t, id, self.glsl_id, glsl_op, a, b],
        );
        id
    }

    fn fma(&mut self, a: u32, b: u32, c: u32) -> u32 {
        let id = self.alloc_id();
        push_inst(
            &mut self.fn_body,
            OP_EXT_INST,
            &[self.f32_t, id, self.glsl_id, GLSL_FMA, a, b, c],
        );
        push_inst(
            &mut self.annotations,
            OP_DECORATE,
            &[id, DECORATION_NO_CONTRACTION],
        );
        id
    }

    fn f_abs(&mut self, a: u32) -> u32 {
        self.glsl1(self.f32_t, GLSL_FABS, a)
    }

    fn f_floor(&mut self, a: u32) -> u32 {
        self.glsl1(self.f32_t, GLSL_FLOOR, a)
    }

    fn f_exp(&mut self, a: u32) -> u32 {
        self.glsl1(self.f32_t, GLSL_EXP, a)
    }

    fn f_log(&mut self, a: u32) -> u32 {
        self.glsl1(self.f32_t, GLSL_LOG, a)
    }

    fn f_sqrt(&mut self, a: u32) -> u32 {
        self.glsl1(self.f32_t, GLSL_SQRT, a)
    }

    fn f_max(&mut self, a: u32, b: u32) -> u32 {
        self.glsl2(self.f32_t, GLSL_FMAX, a, b)
    }

    fn s_min(&mut self, a: u32, b: u32) -> u32 {
        self.glsl2(self.i32_t, GLSL_SMIN, a, b)
    }

    fn s_max(&mut self, a: u32, b: u32) -> u32 {
        self.glsl2(self.i32_t, GLSL_SMAX, a, b)
    }

    fn unpack_half_pair(&mut self, u32_val: u32) -> (u32, u32) {
        let vec2 = self.glsl1(self.vec2f32_t, GLSL_UNPACK_HALF_2X16, u32_val);
        let lo = self.alloc_id();
        push_inst(
            &mut self.fn_body,
            OP_COMPOSITE_EXTRACT,
            &[self.f32_t, lo, vec2, 0],
        );
        let hi = self.alloc_id();
        push_inst(
            &mut self.fn_body,
            OP_COMPOSITE_EXTRACT,
            &[self.f32_t, hi, vec2, 1],
        );
        (lo, hi)
    }

    fn unpack_half_low(&mut self, u32_val: u32) -> u32 {
        self.unpack_half_pair(u32_val).0
    }

    fn round_f16_bits(&mut self, val: u32) -> u32 {
        let bits = self.bitcast(self.u32_t, val);
        let mask_abs = self.const_u32(0x7FFF_FFFF);
        let abs_bits = self.uand(bits, mask_abs);
        let abs_val = self.bitcast(self.f32_t, abs_bits);
        let c_up = self.const_f32(f32::from_bits(0x7780_0000));
        let c_down = self.const_f32(f32::from_bits(0x0880_0000));
        let t1 = self.fmul_nc(abs_val, c_up);
        let base0 = self.fmul_nc(t1, c_down);

        let doubled = self.uadd(bits, bits);
        let mask_exp = self.const_u32(0xFF00_0000);
        let min_exp = self.const_u32(0x7100_0000);
        let exp_masked = self.uand(doubled, mask_exp);
        let exp_lt = self.u_lt(exp_masked, min_exp);
        let exp_bias = self.select(self.u32_t, exp_lt, min_exp, exp_masked);
        let c1 = self.const_u32(1);
        let bias_half = self.ushr(exp_bias, c1);
        let c_add_bias = self.const_u32(0x0780_0000);
        let bias_bits = self.uadd(bias_half, c_add_bias);
        let bias_f = self.bitcast(self.f32_t, bias_bits);

        let base = self.fadd_nc(base0, bias_f);
        let rounded = self.bitcast(self.u32_t, base);
        let c13 = self.const_u32(13);
        let mask_7c00 = self.const_u32(0x7C00);
        let mask_0fff = self.const_u32(0x0FFF);
        let r_shr13 = self.ushr(rounded, c13);
        let hi_part = self.uand(r_shr13, mask_7c00);
        let lo_part = self.uand(rounded, mask_0fff);
        let mag_norm = self.uadd(hi_part, lo_part);

        let is_nan = self.u_lt(mask_exp, doubled);
        let c_nan = self.const_u32(0x7E00);
        let magnitude = self.select(self.u32_t, is_nan, c_nan, mag_norm);

        let c16 = self.const_u32(16);
        let mask_sign = self.const_u32(0x8000);
        let sign_shr = self.ushr(bits, c16);
        let sign_bit = self.uand(sign_shr, mask_sign);
        self.uor(sign_bit, magnitude)
    }

    fn round_f16(&mut self, val: u32) -> u32 {
        let half_u16 = self.round_f16_bits(val);
        self.unpack_half_low(half_u16)
    }

    fn sdot_packed(&mut self, a_u32: u32, b_u32: u32) -> u32 {
        let a_i = self.bitcast(self.i32_t, a_u32);
        let b_i = self.bitcast(self.i32_t, b_u32);
        let id = self.alloc_id();
        push_inst(
            &mut self.fn_body,
            OP_S_DOT,
            &[self.i32_t, id, a_i, b_i, PACKED_FORMAT_4X8_BIT],
        );
        id
    }

    fn workgroup_barrier(&mut self) {
        let scope = self.const_u32(SCOPE_WORKGROUP);
        let sem = self.const_u32(SEMANTICS_ACQUIRE_RELEASE_WORKGROUP);
        push_inst(&mut self.fn_body, OP_CONTROL_BARRIER, &[scope, scope, sem]);
    }

    fn emit_if<F>(&mut self, cond: u32, then_fn: F)
    where
        F: FnOnce(&mut Self),
    {
        let then_lbl = self.alloc_id();
        let merge_lbl = self.alloc_id();
        push_inst(&mut self.fn_body, OP_SELECTION_MERGE, &[merge_lbl, 0]);
        push_inst(
            &mut self.fn_body,
            OP_BRANCH_CONDITIONAL,
            &[cond, then_lbl, merge_lbl],
        );
        push_inst(&mut self.fn_body, OP_LABEL, &[then_lbl]);
        self.current_label = then_lbl;
        then_fn(self);
        push_inst(&mut self.fn_body, OP_BRANCH, &[merge_lbl]);
        push_inst(&mut self.fn_body, OP_LABEL, &[merge_lbl]);
        self.current_label = merge_lbl;
    }

    fn emit_if_else<F, G>(&mut self, cond: u32, then_fn: F, else_fn: G)
    where
        F: FnOnce(&mut Self),
        G: FnOnce(&mut Self),
    {
        let then_lbl = self.alloc_id();
        let else_lbl = self.alloc_id();
        let merge_lbl = self.alloc_id();
        push_inst(&mut self.fn_body, OP_SELECTION_MERGE, &[merge_lbl, 0]);
        push_inst(
            &mut self.fn_body,
            OP_BRANCH_CONDITIONAL,
            &[cond, then_lbl, else_lbl],
        );
        push_inst(&mut self.fn_body, OP_LABEL, &[then_lbl]);
        self.current_label = then_lbl;
        then_fn(self);
        push_inst(&mut self.fn_body, OP_BRANCH, &[merge_lbl]);
        push_inst(&mut self.fn_body, OP_LABEL, &[else_lbl]);
        self.current_label = else_lbl;
        else_fn(self);
        push_inst(&mut self.fn_body, OP_BRANCH, &[merge_lbl]);
        push_inst(&mut self.fn_body, OP_LABEL, &[merge_lbl]);
        self.current_label = merge_lbl;
    }

    fn emit_for_range<F>(&mut self, init: u32, limit: u32, step: u32, body_fn: F)
    where
        F: FnOnce(&mut Self, u32),
    {
        let preheader = self.current_label;
        let header = self.alloc_id();
        let body = self.alloc_id();
        let cont_lbl = self.alloc_id();
        let merge = self.alloc_id();
        let idx = self.alloc_id();
        let next_idx = self.alloc_id();

        push_inst(&mut self.fn_body, OP_BRANCH, &[header]);
        push_inst(&mut self.fn_body, OP_LABEL, &[header]);
        push_inst(
            &mut self.fn_body,
            OP_PHI,
            &[self.u32_t, idx, init, preheader, next_idx, cont_lbl],
        );
        let loop_cond = self.u_lt(idx, limit);
        push_inst(&mut self.fn_body, OP_LOOP_MERGE, &[merge, cont_lbl, 0]);
        push_inst(
            &mut self.fn_body,
            OP_BRANCH_CONDITIONAL,
            &[loop_cond, body, merge],
        );
        push_inst(&mut self.fn_body, OP_LABEL, &[body]);
        self.current_label = body;
        body_fn(self, idx);
        push_inst(&mut self.fn_body, OP_BRANCH, &[cont_lbl]);
        push_inst(&mut self.fn_body, OP_LABEL, &[cont_lbl]);
        push_inst(
            &mut self.fn_body,
            OP_I_ADD,
            &[self.u32_t, next_idx, idx, step],
        );
        push_inst(&mut self.fn_body, OP_BRANCH, &[header]);
        push_inst(&mut self.fn_body, OP_LABEL, &[merge]);
        self.current_label = merge;
    }

    fn emit_while<C, F>(&mut self, cond_fn: C, body_fn: F)
    where
        C: FnOnce(&mut Self) -> u32,
        F: FnOnce(&mut Self),
    {
        let header = self.alloc_id();
        let body = self.alloc_id();
        let cont_lbl = self.alloc_id();
        let merge = self.alloc_id();

        push_inst(&mut self.fn_body, OP_BRANCH, &[header]);
        push_inst(&mut self.fn_body, OP_LABEL, &[header]);
        self.current_label = header;
        let cond = cond_fn(self);
        push_inst(&mut self.fn_body, OP_LOOP_MERGE, &[merge, cont_lbl, 0]);
        push_inst(
            &mut self.fn_body,
            OP_BRANCH_CONDITIONAL,
            &[cond, body, merge],
        );
        push_inst(&mut self.fn_body, OP_LABEL, &[body]);
        self.current_label = body;
        body_fn(self);
        push_inst(&mut self.fn_body, OP_BRANCH, &[cont_lbl]);
        push_inst(&mut self.fn_body, OP_LABEL, &[cont_lbl]);
        push_inst(&mut self.fn_body, OP_BRANCH, &[header]);
        push_inst(&mut self.fn_body, OP_LABEL, &[merge]);
        self.current_label = merge;
    }
}

/// Returns `true` when the active kernel path uses `OpSDot` (`SPV_KHR_integer_dot_product`).
#[must_use]
pub const fn path_uses_sdot(path: ProjectionKernelPath) -> bool {
    matches!(path, ProjectionKernelPath::IntegerDotPacked4x8)
}

/// Computes the canonical lowercase hex SHA-256 digest of a SPIR-V word sequence.
///
/// # Errors
/// Returns [`Error::InvalidInput`] if hashing fails.
pub fn shader_digest(words: &[u32]) -> Result<String, Error> {
    let mut hasher = Hasher::new().map_err(|err| Error::InvalidInput(err.to_string()))?;
    for &word in words {
        hasher
            .update(&word.to_le_bytes())
            .map_err(|err| Error::InvalidInput(err.to_string()))?;
    }
    let digest = hasher
        .finish()
        .map_err(|err| Error::InvalidInput(err.to_string()))?;
    Ok(hex::encode(digest))
}

fn load_u32_at_hw(b: &mut ShaderBuilder, weights_buf: u32, hw_idx: u32) -> u32 {
    let one = b.const_u32(1);
    let zero = b.const_u32(0);
    let sixteen = b.const_u32(16);
    let word_idx = b.ushr(hw_idx, one);
    let next_word = b.uadd(word_idx, one);
    let w0 = b.load_sb_u32(weights_buf, word_idx);
    let w1 = b.load_sb_u32(weights_buf, next_word);
    let odd_bit = b.uand(hw_idx, one);
    let is_odd = b.u_ne(odd_bit, zero);
    let hi = b.ushr(w0, sixteen);
    let lo = b.ushl(w1, sixteen);
    let shifted = b.uor(hi, lo);
    b.select(b.u32_t, is_odd, shifted, w0)
}

fn swar_pack_4x2bit_to_4xi8(b: &mut ShaderBuilder, raw_byte: u32) -> u32 {
    let m0 = b.const_u32(0x03);
    let m1 = b.const_u32(0x0C);
    let m2 = b.const_u32(0x30);
    let m3 = b.const_u32(0xC0);
    let s1 = b.const_u32(6);
    let s2 = b.const_u32(12);
    let s3 = b.const_u32(18);
    let p0 = b.uand(raw_byte, m0);
    let t1 = b.uand(raw_byte, m1);
    let p1 = b.ushl(t1, s1);
    let t2 = b.uand(raw_byte, m2);
    let p2 = b.ushl(t2, s2);
    let t3 = b.uand(raw_byte, m3);
    let p3 = b.ushl(t3, s3);
    let p01 = b.uor(p0, p1);
    let p23 = b.uor(p2, p3);
    let lanes = b.uor(p01, p23);
    swar_bias_minus_one(b, lanes)
}

fn swar_bias_minus_one(b: &mut ShaderBuilder, packed_unsigned_4bytes: u32) -> u32 {
    let bias_7f = b.const_u32(0x7F7F_7F7F);
    let sign_80 = b.const_u32(0x8080_8080);
    let added = b.uadd(packed_unsigned_4bytes, bias_7f);
    b.uxor(added, sign_80)
}

fn unpack_2bit_byte_to_i32_weights(b: &mut ShaderBuilder, raw_byte: u32) -> [u32; 4] {
    let m = b.const_u32(3);
    let two = b.const_u32(2);
    let four = b.const_u32(4);
    let six = b.const_u32(6);
    let i_one = b.const_i32(1);

    let d0 = b.uand(raw_byte, m);
    let d1s = b.ushr(raw_byte, two);
    let d1 = b.uand(d1s, m);
    let d2s = b.ushr(raw_byte, four);
    let d2 = b.uand(d2s, m);
    let d3s = b.ushr(raw_byte, six);
    let d3 = b.uand(d3s, m);

    let d0_i = b.bitcast(b.i32_t, d0);
    let d1_i = b.bitcast(b.i32_t, d1);
    let d2_i = b.bitcast(b.i32_t, d2);
    let d3_i = b.bitcast(b.i32_t, d3);

    [
        b.isub(d0_i, i_one),
        b.isub(d1_i, i_one),
        b.isub(d2_i, i_one),
        b.isub(d3_i, i_one),
    ]
}

fn portable_dot4_from_2bit_byte(b: &mut ShaderBuilder, raw_byte: u32, act_u32: u32) -> u32 {
    let weights = unpack_2bit_byte_to_i32_weights(b, raw_byte);
    portable_dot4_i32_weights(b, weights, act_u32)
}

fn unpack_packed_trits_to_i32_weights(b: &mut ShaderBuilder, lanes: u32) -> [u32; 4] {
    let mask = b.const_u32(0x03);
    let c8 = b.const_u32(8);
    let c16 = b.const_u32(16);
    let c24 = b.const_u32(24);
    let i_one = b.const_i32(1);

    let d0 = b.uand(lanes, mask);
    let d1s = b.ushr(lanes, c8);
    let d1 = b.uand(d1s, mask);
    let d2s = b.ushr(lanes, c16);
    let d2 = b.uand(d2s, mask);
    let d3 = b.ushr(lanes, c24);

    let d0_i = b.bitcast(b.i32_t, d0);
    let d1_i = b.bitcast(b.i32_t, d1);
    let d2_i = b.bitcast(b.i32_t, d2);
    let d3_i = b.bitcast(b.i32_t, d3);

    [
        b.isub(d0_i, i_one),
        b.isub(d1_i, i_one),
        b.isub(d2_i, i_one),
        b.isub(d3_i, i_one),
    ]
}

fn portable_dot4_from_packed_trits(b: &mut ShaderBuilder, lanes: u32, act_u32: u32) -> u32 {
    let weights = unpack_packed_trits_to_i32_weights(b, lanes);
    portable_dot4_i32_weights(b, weights, act_u32)
}

fn portable_dot4_i32_weights(b: &mut ShaderBuilder, weights: [u32; 4], act_u32: u32) -> u32 {
    let ai = b.bitcast(b.i32_t, act_u32);
    let c24 = b.const_u32(24);
    let c16 = b.const_u32(16);
    let c8 = b.const_u32(8);

    let s0 = b.ishl(ai, c24);
    let a0 = b.ishr_arith(s0, c24);
    let s1 = b.ishl(ai, c16);
    let a1 = b.ishr_arith(s1, c24);
    let s2 = b.ishl(ai, c8);
    let a2 = b.ishr_arith(s2, c24);
    let a3 = b.ishr_arith(ai, c24);

    let p0 = b.imul(weights[0], a0);
    let p1 = b.imul(weights[1], a1);
    let p2 = b.imul(weights[2], a2);
    let p3 = b.imul(weights[3], a3);
    let p01 = b.iadd(p0, p1);
    let p23 = b.iadd(p2, p3);
    b.iadd(p01, p23)
}

fn pq2_block_int_dot(
    b: &mut ShaderBuilder,
    weights_buf: u32,
    hw_weight_words_base: u32,
    act_buf: u32,
    act_words_base: u32,
    use_sdot: bool,
) -> u32 {
    let c1 = b.const_u32(1);
    let c2 = b.const_u32(2);
    let c3 = b.const_u32(3);
    let c8 = b.const_u32(8);
    let c16 = b.const_u32(16);
    let c24 = b.const_u32(24);
    let mut sum = b.const_i32(0);

    for word_j in 0_u32..8 {
        let off = b.const_u32(word_j * 2);
        let hw = b.uadd(hw_weight_words_base, off);
        let pq_word = load_u32_at_hw(b, weights_buf, hw);

        let act_base_off = b.const_u32(word_j * 4);
        let act_idx0 = b.uadd(act_words_base, act_base_off);
        let act_idx1 = b.uadd(act_idx0, c1);
        let act_idx2 = b.uadd(act_idx0, c2);
        let act_idx3 = b.uadd(act_idx0, c3);

        let aw0 = b.load_sb_u32(act_buf, act_idx0);
        let aw1 = b.load_sb_u32(act_buf, act_idx1);
        let aw2 = b.load_sb_u32(act_buf, act_idx2);
        let aw3 = b.load_sb_u32(act_buf, act_idx3);

        let b0 = pq_word;
        let b1 = b.ushr(pq_word, c8);
        let b2 = b.ushr(pq_word, c16);
        let b3 = b.ushr(pq_word, c24);

        for (raw_byte, act_w) in [(b0, aw0), (b1, aw1), (b2, aw2), (b3, aw3)] {
            let dot = if use_sdot {
                let packed = swar_pack_4x2bit_to_4xi8(b, raw_byte);
                b.sdot_packed(packed, act_w)
            } else {
                portable_dot4_from_2bit_byte(b, raw_byte, act_w)
            };
            sum = b.iadd(sum, dot);
        }
    }
    sum
}

/// Emits `#39` `PQ2_0` × `Q8_K` GEMV SPIR-V 1.6 words.
///
/// `fused = false` emits `NoContraction` `fmul` + `fadd` matching CPU `dot_pq2::<false>` bit-for-bit
/// (`0 ULP`). `fused = true` emits `GLSL.std.450` `Fma` (`NoContraction`), whose Vulkan precision
/// inherits `a * b + c` and can differ by `1 ULP` from single-rounded CPU `f32::mul_add` on Adreno
/// drivers lowering to `mad.f32`, requiring a separate numerical profile gate.
///
/// Bindings: `0` read-only `weights` (`u32` words, padded +1 trailing word), `1` read-only
/// packed `Q8KBlock`s (`65` `u32` words per block: `[scale_bits, values[64]]`), `2` writable
/// `output` (`f32`). Push constants (`4 × u32`): `[rows, blocks_per_row, tokens, act_words_per_token]`.
#[must_use]
pub fn pq2_q8k_gemv(use_sdot: bool, fused: bool) -> Vec<u32> {
    let (mut b, push_var) = ShaderBuilder::new(use_sdot, 4);
    let push_var = push_var.unwrap_or(0);
    let weights_buf = b.storage_buffer(0, ElemType::U32, AccessMode::ReadOnly);
    let act_buf = b.storage_buffer(1, ElemType::U32, AccessMode::ReadOnly);
    let out_buf = b.storage_buffer(2, ElemType::F32, AccessMode::ReadWrite);

    let gid = b.global_invocation_id();
    let row = b.load_input_component(gid, 0);
    let token = b.load_input_component(gid, 1);

    let rows = b.load_push(push_var, 0);
    let blocks_per_row = b.load_push(push_var, 1);
    let tokens = b.load_push(push_var, 2);
    let act_words_per_token = b.load_push(push_var, 3);

    let row_ok = b.u_lt(row, rows);
    let tok_ok = b.u_lt(token, tokens);
    let in_bounds = b.logical_and(row_ok, tok_ok);

    b.emit_if(in_bounds, |b| {
        let acc_var = b.local_f32_var();
        let f_zero = b.const_f32(0.0);
        b.store_var(acc_var, f_zero);

        let c0 = b.const_u32(0);
        let c1 = b.const_u32(1);
        let c17 = b.const_u32(17);
        let c32 = b.const_u32(32);
        let c65 = b.const_u32(65);
        let mask16 = b.const_u32(0xFFFF);

        let row_blocks = b.umul(row, blocks_per_row);
        let tok_act_base = b.umul(token, act_words_per_token);

        b.emit_for_range(c0, blocks_per_row, c1, |b, blk| {
            let blk_total = b.uadd(row_blocks, blk);
            let hw_base = b.umul(blk_total, c17);
            let scale_raw = load_u32_at_hw(b, weights_buf, hw_base);
            let w_scale_bits = b.uand(scale_raw, mask16);
            let w_scale = b.unpack_half_low(w_scale_bits);

            let q8k_idx = b.ushr(blk, c1);
            let q8k_off = b.umul(q8k_idx, c65);
            let q8k_base = b.uadd(tok_act_base, q8k_off);
            let act_scale_u32 = b.load_sb_u32(act_buf, q8k_base);
            let act_scale = b.bitcast(b.f32_t, act_scale_u32);

            let half = b.uand(blk, c1);
            let half_off = b.umul(half, c32);
            let act_vals_start = b.uadd(q8k_base, c1);
            let act_vals_base = b.uadd(act_vals_start, half_off);

            let hw_words = b.uadd(hw_base, c1);
            let int_dot =
                pq2_block_int_dot(b, weights_buf, hw_words, act_buf, act_vals_base, use_sdot);
            let float_dot = b.i32_to_f32(int_dot);
            let d = b.fmul_nc(w_scale, act_scale);
            let prev = b.load_var_f32(acc_var);
            let next = if fused {
                b.fma(d, float_dot, prev)
            } else {
                let prod = b.fmul_nc(d, float_dot);
                b.fadd_nc(prev, prod)
            };
            b.store_var(acc_var, next);
        });

        let tok_out = b.umul(token, rows);
        let out_idx = b.uadd(tok_out, row);
        let final_acc = b.load_var_f32(acc_var);
        b.store_sb_f32(out_buf, out_idx, final_acc);
    });

    b.finish()
}

/// Emits `#39` `PQ2_0` × `Q8_0` GEMV SPIR-V 1.6 words (supports odd `128`-column multiples).
///
/// Bindings: `0` read-only `weights` (`u32`, +1 pad word), `1` read-only packed `Q8Block`s
/// (`9` `u32` words per 32-element block: `[scale_bits, values[8]]`), `2` writable `output`
/// (`f32`). Push constants (`4 × u32`): `[rows, blocks_per_row, tokens, act_words_per_token]`.
#[must_use]
pub fn pq2_q8_0_gemv(use_sdot: bool) -> Vec<u32> {
    let (mut b, push_var) = ShaderBuilder::new(use_sdot, 4);
    let push_var = push_var.unwrap_or(0);
    let weights_buf = b.storage_buffer(0, ElemType::U32, AccessMode::ReadOnly);
    let act_buf = b.storage_buffer(1, ElemType::U32, AccessMode::ReadOnly);
    let out_buf = b.storage_buffer(2, ElemType::F32, AccessMode::ReadWrite);

    let gid = b.global_invocation_id();
    let row = b.load_input_component(gid, 0);
    let token = b.load_input_component(gid, 1);

    let rows = b.load_push(push_var, 0);
    let blocks_per_row = b.load_push(push_var, 1);
    let tokens = b.load_push(push_var, 2);
    let act_words_per_token = b.load_push(push_var, 3);

    let row_ok = b.u_lt(row, rows);
    let tok_ok = b.u_lt(token, tokens);
    let in_bounds = b.logical_and(row_ok, tok_ok);

    b.emit_if(in_bounds, |b| {
        let acc_var = b.local_f32_var();
        let f_zero = b.const_f32(0.0);
        b.store_var(acc_var, f_zero);

        let c0 = b.const_u32(0);
        let c1 = b.const_u32(1);
        let c2 = b.const_u32(2);
        let c3 = b.const_u32(3);
        let c8 = b.const_u32(8);
        let c16 = b.const_u32(16);
        let c17 = b.const_u32(17);
        let c24 = b.const_u32(24);
        let c36 = b.const_u32(36);
        let mask16 = b.const_u32(0xFFFF);

        let row_blocks = b.umul(row, blocks_per_row);
        let tok_act_base = b.umul(token, act_words_per_token);

        b.emit_for_range(c0, blocks_per_row, c1, |b, blk| {
            let blk_total = b.uadd(row_blocks, blk);
            let hw_base = b.umul(blk_total, c17);
            let scale_raw = load_u32_at_hw(b, weights_buf, hw_base);
            let w_scale_bits = b.uand(scale_raw, mask16);
            let w_scale = b.unpack_half_low(w_scale_bits);
            let hw_words = b.uadd(hw_base, c1);

            let blk_act_off = b.umul(blk, c36);
            let blk_act_base = b.uadd(tok_act_base, blk_act_off);

            let mut block_sum = f_zero;
            for grp in 0_u32..4 {
                let grp_off_c = b.const_u32(grp * 9);
                let grp_base = b.uadd(blk_act_base, grp_off_c);
                let act_scale_u32 = b.load_sb_u32(act_buf, grp_base);
                let act_scale = b.bitcast(b.f32_t, act_scale_u32);

                let mut grp_int = b.const_i32(0);
                for pair_idx in 0_u32..2 {
                    let word_j = grp * 2 + pair_idx;
                    let hw_off = b.const_u32(word_j * 2);
                    let hw = b.uadd(hw_words, hw_off);
                    let pq_word = load_u32_at_hw(b, weights_buf, hw);

                    let act_w_off = b.const_u32(1 + pair_idx * 4);
                    let act_w_start = b.uadd(grp_base, act_w_off);
                    let act_w_i1 = b.uadd(act_w_start, c1);
                    let act_w_i2 = b.uadd(act_w_start, c2);
                    let act_w_i3 = b.uadd(act_w_start, c3);
                    let aw0 = b.load_sb_u32(act_buf, act_w_start);
                    let aw1 = b.load_sb_u32(act_buf, act_w_i1);
                    let aw2 = b.load_sb_u32(act_buf, act_w_i2);
                    let aw3 = b.load_sb_u32(act_buf, act_w_i3);

                    let b0 = pq_word;
                    let b1 = b.ushr(pq_word, c8);
                    let b2 = b.ushr(pq_word, c16);
                    let b3 = b.ushr(pq_word, c24);

                    for (raw_byte, act_w) in [(b0, aw0), (b1, aw1), (b2, aw2), (b3, aw3)] {
                        let dot = if use_sdot {
                            let packed = swar_pack_4x2bit_to_4xi8(b, raw_byte);
                            b.sdot_packed(packed, act_w)
                        } else {
                            portable_dot4_from_2bit_byte(b, raw_byte, act_w)
                        };
                        grp_int = b.iadd(grp_int, dot);
                    }
                }
                let grp_f32 = b.i32_to_f32(grp_int);
                let scaled = b.fmul_nc(act_scale, grp_f32);
                block_sum = b.fadd_nc(block_sum, scaled);
            }

            let prev = b.load_var_f32(acc_var);
            let contrib = b.fmul_nc(w_scale, block_sum);
            let next = b.fadd_nc(prev, contrib);
            b.store_var(acc_var, next);
        });

        let tok_out = b.umul(token, rows);
        let out_idx = b.uadd(tok_out, row);
        let final_acc = b.load_var_f32(acc_var);
        b.store_sb_f32(out_buf, out_idx, final_acc);
    });

    b.finish()
}

struct PtqDecodedBlock {
    w_scale: u32,
    lanes: [u32; 32],
}

fn decode_ptq1_block_lanes(b: &mut ShaderBuilder, weights_buf: u32, wbase: u32) -> PtqDecodedBlock {
    decode_ptq1_block_lanes_strided(b, weights_buf, wbase, 1)
}

fn decode_ptq1_block_lanes_strided(
    b: &mut ShaderBuilder,
    weights_buf: u32,
    wbase: u32,
    word_stride: u32,
) -> PtqDecodedBlock {
    let m00ff = b.const_u32(0x00FF_00FF);
    let mask_low_byte = b.const_u32(0xFF);
    let mask_hi_byte = b.const_u32(0xFF00);
    let c3 = b.const_u32(3);
    let c8 = b.const_u32(8);
    let c16 = b.const_u32(16);
    let c24 = b.const_u32(24);
    let mask_trit = b.const_u32(3);

    let mut lanes = [0_u32; 32];

    for g in 0_u32..4 {
        let g_c = b.const_u32(g * word_stride);
        let wg_idx = b.uadd(wbase, g_c);
        let wg = b.load_sb_u32(weights_buf, wg_idx);
        let mut lo = b.uand(wg, m00ff);
        let wg_s8 = b.ushr(wg, c8);
        let mut hi = b.uand(wg_s8, m00ff);

        for n in 0_u32..5 {
            lo = b.umul(lo, c3);
            hi = b.umul(hi, c3);
            let lo_s8 = b.ushr(lo, c8);
            let d0 = b.uand(lo_s8, mask_trit);
            let lo_s24 = b.ushr(lo, c24);
            let d2 = b.uand(lo_s24, mask_trit);
            let hi_s8 = b.ushr(hi, c8);
            let d1 = b.uand(hi_s8, mask_trit);
            let hi_s24 = b.ushr(hi, c24);
            let d3 = b.uand(hi_s24, mask_trit);
            lo = b.uand(lo, m00ff);
            hi = b.uand(hi, m00ff);

            let t1 = b.ushl(d1, c8);
            let t2 = b.ushl(d2, c16);
            let t3 = b.ushl(d3, c24);
            let t01 = b.uor(d0, t1);
            let t23 = b.uor(t2, t3);
            lanes[usize::try_from(4 * n + g).unwrap_or(0)] = b.uor(t01, t23);
        }
    }

    for g2 in 0_u32..2 {
        let g2_c = b.const_u32((4 + g2) * word_stride);
        let wg_idx = b.uadd(wbase, g2_c);
        let wg = b.load_sb_u32(weights_buf, wg_idx);
        let mut lo = b.uand(wg, m00ff);
        let wg_s8 = b.ushr(wg, c8);
        let mut hi = b.uand(wg_s8, m00ff);

        for n in 0_u32..5 {
            lo = b.umul(lo, c3);
            hi = b.umul(hi, c3);
            let lo_s8 = b.ushr(lo, c8);
            let d0 = b.uand(lo_s8, mask_trit);
            let lo_s24 = b.ushr(lo, c24);
            let d2 = b.uand(lo_s24, mask_trit);
            let hi_s8 = b.ushr(hi, c8);
            let d1 = b.uand(hi_s8, mask_trit);
            let hi_s24 = b.ushr(hi, c24);
            let d3 = b.uand(hi_s24, mask_trit);
            lo = b.uand(lo, m00ff);
            hi = b.uand(hi, m00ff);

            let t1 = b.ushl(d1, c8);
            let t2 = b.ushl(d2, c16);
            let t3 = b.ushl(d3, c24);
            let t01 = b.uor(d0, t1);
            let t23 = b.uor(t2, t3);
            lanes[usize::try_from(20 + 2 * n + g2).unwrap_or(0)] = b.uor(t01, t23);
        }
    }

    let c6 = b.const_u32(6 * word_stride);
    let w6_idx = b.uadd(wbase, c6);
    let w6 = b.load_sb_u32(weights_buf, w6_idx);
    let w_scale_u16 = b.ushr(w6, c16);
    let w_scale = b.unpack_half_low(w_scale_u16);

    let b24 = b.uand(w6, mask_low_byte);
    let b25_masked = b.uand(w6, mask_hi_byte);
    let b25_shifted = b.ushl(b25_masked, c8);
    let mut pair = b.uor(b24, b25_shifted);

    let mut d_steps = [(0_u32, 0_u32); 4];
    for slot in &mut d_steps {
        pair = b.umul(pair, c3);
        let pair_s8 = b.ushr(pair, c8);
        let ev = b.uand(pair_s8, mask_trit);
        let pair_s24 = b.ushr(pair, c24);
        let od = b.uand(pair_s24, mask_trit);
        pair = b.uand(pair, m00ff);
        *slot = (ev, od);
    }
    let d0_1 = b.ushl(d_steps[0].1, c8);
    let w30_01 = b.uor(d_steps[0].0, d0_1);
    let d1_0 = b.ushl(d_steps[1].0, c16);
    let d1_1 = b.ushl(d_steps[1].1, c24);
    let w30_23 = b.uor(d1_0, d1_1);
    lanes[30] = b.uor(w30_01, w30_23);

    let d2_1 = b.ushl(d_steps[2].1, c8);
    let w31_01 = b.uor(d_steps[2].0, d2_1);
    let d3_0 = b.ushl(d_steps[3].0, c16);
    let d3_1 = b.ushl(d_steps[3].1, c24);
    let w31_23 = b.uor(d3_0, d3_1);
    lanes[31] = b.uor(w31_01, w31_23);

    PtqDecodedBlock { w_scale, lanes }
}

fn dot_trit_lanes_with_act(b: &mut ShaderBuilder, lanes: u32, act_w: u32, use_sdot: bool) -> u32 {
    if use_sdot {
        let packed_w = swar_bias_minus_one(b, lanes);
        b.sdot_packed(packed_w, act_w)
    } else {
        portable_dot4_from_packed_trits(b, lanes, act_w)
    }
}

/// Emits `#39` `PTQ1_0` × `Q8_0` GEMV SPIR-V 1.6 words (`pinned_4lane` or standard).
///
/// Bindings: `0` read-only `weights` (`7` `u32` words per 128-element `PTQ1_0` block),
/// `1` read-only packed `Q8Block`s (`9` `u32` words per 32-element block), `2` writable `output`
/// (`f32`). Push constants (`4 × u32`): `[rows, blocks_per_row, tokens, act_words_per_token]`.
#[must_use]
pub fn ptq1_q8_0_gemv(use_sdot: bool, pinned_4lane: bool) -> Vec<u32> {
    let (mut b, push_var) = ShaderBuilder::new(use_sdot, 4);
    let push_var = push_var.unwrap_or(0);
    let weights_buf = b.storage_buffer(0, ElemType::U32, AccessMode::ReadOnly);
    let act_buf = b.storage_buffer(1, ElemType::U32, AccessMode::ReadOnly);
    let out_buf = b.storage_buffer(2, ElemType::F32, AccessMode::ReadWrite);

    let gid = b.global_invocation_id();
    let row = b.load_input_component(gid, 0);
    let token = b.load_input_component(gid, 1);

    let rows = b.load_push(push_var, 0);
    let blocks_per_row = b.load_push(push_var, 1);
    let tokens = b.load_push(push_var, 2);
    let act_words_per_token = b.load_push(push_var, 3);

    let row_ok = b.u_lt(row, rows);
    let tok_ok = b.u_lt(token, tokens);
    let in_bounds = b.logical_and(row_ok, tok_ok);

    b.emit_if(in_bounds, |b| {
        let f_zero = b.const_f32(0.0);
        let lane_vars = [
            b.local_f32_var(),
            b.local_f32_var(),
            b.local_f32_var(),
            b.local_f32_var(),
        ];
        for var in lane_vars {
            b.store_var(var, f_zero);
        }

        let c0 = b.const_u32(0);
        let c1 = b.const_u32(1);
        let c7 = b.const_u32(7);
        let c36 = b.const_u32(36);

        let row_blocks = b.umul(row, blocks_per_row);
        let tok_act_base = b.umul(token, act_words_per_token);

        b.emit_for_range(c0, blocks_per_row, c1, |b, blk| {
            let blk_total = b.uadd(row_blocks, blk);
            let wbase = b.umul(blk_total, c7);
            let decoded = decode_ptq1_block_lanes(b, weights_buf, wbase);

            let blk_act_off = b.umul(blk, c36);
            let blk_act_base = b.uadd(tok_act_base, blk_act_off);

            let mut block_sum = f_zero;

            for grp in 0_usize..4 {
                let grp_index = u32::try_from(grp).unwrap_or(0);
                let grp_off_c = b.const_u32(grp_index * 9);
                let grp_base = b.uadd(blk_act_base, grp_off_c);
                let act_scale_u32 = b.load_sb_u32(act_buf, grp_base);
                let act_scale = b.bitcast(b.f32_t, act_scale_u32);

                let mut lane_sums = [b.const_i32(0); 4];
                for half in 0_usize..2 {
                    for (l, lane_slot) in lane_sums.iter_mut().enumerate() {
                        let act_within_grp = u32::try_from(half * 4 + l).unwrap_or(0);
                        let act_off_c = b.const_u32(1 + act_within_grp);
                        let act_w_idx = b.uadd(grp_base, act_off_c);
                        let act_w = b.load_sb_u32(act_buf, act_w_idx);
                        let word_idx = grp * 8 + half * 4 + l;
                        let dot =
                            dot_trit_lanes_with_act(b, decoded.lanes[word_idx], act_w, use_sdot);
                        *lane_slot = b.iadd(*lane_slot, dot);
                    }
                }

                if pinned_4lane {
                    let d = b.fmul_nc(decoded.w_scale, act_scale);
                    for (&lane_var, &sum_val) in lane_vars.iter().zip(&lane_sums) {
                        let lane_f32 = b.i32_to_f32(sum_val);
                        let prev = b.load_var_f32(lane_var);
                        let next = b.fma(lane_f32, d, prev);
                        b.store_var(lane_var, next);
                    }
                } else {
                    let s01 = b.iadd(lane_sums[0], lane_sums[1]);
                    let s23 = b.iadd(lane_sums[2], lane_sums[3]);
                    let grp_int = b.iadd(s01, s23);
                    let grp_sum_f32 = b.i32_to_f32(grp_int);
                    let scaled = b.fmul_nc(act_scale, grp_sum_f32);
                    block_sum = b.fadd_nc(block_sum, scaled);
                }
            }

            if !pinned_4lane {
                let prev = b.load_var_f32(lane_vars[0]);
                let contrib = b.fmul_nc(decoded.w_scale, block_sum);
                let next = b.fadd_nc(prev, contrib);
                b.store_var(lane_vars[0], next);
            }
        });

        let final_val = if pinned_4lane {
            let l0 = b.load_var_f32(lane_vars[0]);
            let l1 = b.load_var_f32(lane_vars[1]);
            let l2 = b.load_var_f32(lane_vars[2]);
            let l3 = b.load_var_f32(lane_vars[3]);
            let s01 = b.fadd_nc(l0, l1);
            let s23 = b.fadd_nc(l2, l3);
            b.fadd_nc(s01, s23)
        } else {
            b.load_var_f32(lane_vars[0])
        };
        let tok_out = b.umul(token, rows);
        let out_idx = b.uadd(tok_out, row);
        b.store_sb_f32(out_buf, out_idx, final_val);
    });

    b.finish()
}

/// Emits `#39` `PTQ1_0` × `Q8_K` direct GEMV SPIR-V 1.6 words (supports odd `128`-col multiples).
///
/// Bindings: `0` read-only `PTQ1_0` weights (`7` `u32`/block), `1` read-only `Q8KBlock` (`65`
/// `u32`/block), `2` writable `output` (`f32`).
#[must_use]
pub fn ptq1_q8k_gemv(use_sdot: bool) -> Vec<u32> {
    let (mut b, push_var) = ShaderBuilder::new(use_sdot, 4);
    let push_var = push_var.unwrap_or(0);
    let weights_buf = b.storage_buffer(0, ElemType::U32, AccessMode::ReadOnly);
    let act_buf = b.storage_buffer(1, ElemType::U32, AccessMode::ReadOnly);
    let out_buf = b.storage_buffer(2, ElemType::F32, AccessMode::ReadWrite);

    let gid = b.global_invocation_id();
    let row = b.load_input_component(gid, 0);
    let token = b.load_input_component(gid, 1);

    let rows = b.load_push(push_var, 0);
    let blocks_per_row = b.load_push(push_var, 1);
    let tokens = b.load_push(push_var, 2);
    let act_words_per_token = b.load_push(push_var, 3);

    let row_ok = b.u_lt(row, rows);
    let tok_ok = b.u_lt(token, tokens);
    let in_bounds = b.logical_and(row_ok, tok_ok);

    b.emit_if(in_bounds, |b| {
        let acc_var = b.local_f32_var();
        let f_zero = b.const_f32(0.0);
        b.store_var(acc_var, f_zero);

        let c0 = b.const_u32(0);
        let c1 = b.const_u32(1);
        let c7 = b.const_u32(7);
        let c32 = b.const_u32(32);
        let c65 = b.const_u32(65);

        let row_blocks = b.umul(row, blocks_per_row);
        let tok_act_base = b.umul(token, act_words_per_token);

        b.emit_for_range(c0, blocks_per_row, c1, |b, blk| {
            let blk_total = b.uadd(row_blocks, blk);
            let wbase = b.umul(blk_total, c7);
            let decoded = decode_ptq1_block_lanes(b, weights_buf, wbase);

            let q8k_idx = b.ushr(blk, c1);
            let q8k_off = b.umul(q8k_idx, c65);
            let q8k_base = b.uadd(tok_act_base, q8k_off);
            let act_scale_u32 = b.load_sb_u32(act_buf, q8k_base);
            let act_scale = b.bitcast(b.f32_t, act_scale_u32);

            let half = b.uand(blk, c1);
            let half_off = b.umul(half, c32);
            let q8k_vals_start = b.uadd(q8k_base, c1);
            let act_vals_base = b.uadd(q8k_vals_start, half_off);

            let mut sum_int = b.const_i32(0);
            for (word_idx, &lanes) in decoded.lanes.iter().enumerate() {
                let off_c = b.const_u32(u32::try_from(word_idx).unwrap_or(0));
                let act_idx = b.uadd(act_vals_base, off_c);
                let act_w = b.load_sb_u32(act_buf, act_idx);
                let dot = dot_trit_lanes_with_act(b, lanes, act_w, use_sdot);
                sum_int = b.iadd(sum_int, dot);
            }

            let sum_f32 = b.i32_to_f32(sum_int);
            let d = b.fmul_nc(decoded.w_scale, act_scale);
            let prev = b.load_var_f32(acc_var);
            let prod = b.fmul_nc(d, sum_f32);
            let next = b.fadd_nc(prev, prod);
            b.store_var(acc_var, next);
        });

        let tok_out = b.umul(token, rows);
        let out_idx = b.uadd(tok_out, row);
        let final_acc = b.load_var_f32(acc_var);
        b.store_sb_f32(out_buf, out_idx, final_acc);
    });

    b.finish()
}

fn attn_valid_len(
    b: &mut ShaderBuilder,
    qi: u32,
    past_len: u32,
    total_seq: u32,
    causal: u32,
) -> u32 {
    let c0 = b.const_u32(0);
    let c1 = b.const_u32(1);
    let is_causal = b.u_ne(causal, c0);
    let past_q = b.uadd(past_len, qi);
    let past_q1 = b.uadd(past_q, c1);
    let exceeds = b.u_lt(total_seq, past_q1);
    let clamped = b.select(b.u32_t, exceeds, total_seq, past_q1);
    b.select(b.u32_t, is_causal, clamped, total_seq)
}

fn portable_dot4_u8_packed4_s8(b: &mut ShaderBuilder, u4: u32, act_s4: u32) -> u32 {
    let m_ff = b.const_u32(0xFF);
    let c8 = b.const_u32(8);
    let c16 = b.const_u32(16);
    let c24 = b.const_u32(24);
    let u0 = b.uand(u4, m_ff);
    let u1s = b.ushr(u4, c8);
    let u1 = b.uand(u1s, m_ff);
    let u2s = b.ushr(u4, c16);
    let u2 = b.uand(u2s, m_ff);
    let u3 = b.ushr(u4, c24);
    let weights = [
        b.bitcast(b.i32_t, u0),
        b.bitcast(b.i32_t, u1),
        b.bitcast(b.i32_t, u2),
        b.bitcast(b.i32_t, u3),
    ];
    portable_dot4_i32_weights(b, weights, act_s4)
}

fn coalesced_pq2_scale(b: &mut ShaderBuilder, weights_buf: u32, tile_base: u32, lane: u32) -> u32 {
    let c0 = b.const_u32(0);
    let c1 = b.const_u32(1);
    let pair_off = b.ushr(lane, c1);
    let idx = b.uadd(tile_base, pair_off);
    let raw_pair = b.load_sb_u32(weights_buf, idx);
    let (s_ev, s_od) = b.unpack_half_pair(raw_pair);
    let odd_bit = b.uand(lane, c1);
    let is_odd = b.u_ne(odd_bit, c0);
    b.select(b.f32_t, is_odd, s_od, s_ev)
}

fn coalesced_pq2_unpack_4digits(b: &mut ShaderBuilder, pq_word: u32) -> [u32; 4] {
    let m03 = b.const_u32(0x0303_0303);
    let c2 = b.const_u32(2);
    let c4 = b.const_u32(4);
    let c6 = b.const_u32(6);
    let q0 = b.uand(pq_word, m03);
    let s2 = b.ushr(pq_word, c2);
    let q1 = b.uand(s2, m03);
    let s4 = b.ushr(pq_word, c4);
    let q2 = b.uand(s4, m03);
    let s6 = b.ushr(pq_word, c6);
    let q3 = b.uand(s6, m03);
    [q0, q1, q2, q3]
}

fn dot_pq2_4digits_with_act(
    b: &mut ShaderBuilder,
    q: [u32; 4],
    act_buf: u32,
    act_words_base: u32,
    use_sdot: bool,
) -> u32 {
    let mut sum = b.const_i32(0);
    for (digit, &q_word) in q.iter().enumerate() {
        let d_off = b.const_u32(u32::try_from(digit).unwrap_or(0));
        let act_idx = b.uadd(act_words_base, d_off);
        let plane = b.load_sb_u32(act_buf, act_idx);
        let dot = if use_sdot {
            b.sdot_packed(q_word, plane)
        } else {
            portable_dot4_u8_packed4_s8(b, q_word, plane)
        };
        sum = b.iadd(sum, dot);
    }
    sum
}

fn store_maybe_residual_f32(
    b: &mut ShaderBuilder,
    out_buf: u32,
    res_buf: Option<u32>,
    out_idx: u32,
    final_acc: u32,
) {
    let out_val = res_buf.map_or(final_acc, |res_var| {
        let r_val = b.load_sb_f32(res_var, out_idx);
        b.fadd_nc(final_acc, r_val)
    });
    b.store_sb_f32(out_buf, out_idx, out_val);
}

fn clamp_four_token_act_bases(
    b: &mut ShaderBuilder,
    tok_base: u32,
    tokens: u32,
    act_words_per_token: u32,
) -> ([u32; 4], [u32; 4]) {
    let c1 = b.const_u32(1);
    let last_tok = b.binop(OP_I_SUB, b.u32_t, tokens, c1);
    let mut tok_clamped = [0_u32; 4];
    let mut tok_act_bases = [0_u32; 4];
    for k in 0_u32..4 {
        let k_c = b.const_u32(k);
        let tk = b.uadd(tok_base, k_c);
        let is_valid = b.u_lt(tk, tokens);
        let clamped = b.select(b.u32_t, is_valid, tk, last_tok);
        let ki = usize::try_from(k).unwrap_or(0);
        tok_clamped[ki] = tk;
        tok_act_bases[ki] = b.umul(clamped, act_words_per_token);
    }
    (tok_clamped, tok_act_bases)
}

/// Emits coalesced 64-row `PQ2_0` Ã— 4-plane `Q8_K` GEMV SPIR-V 1.6 words (`0%` weight overhead).
///
/// Bindings: `0` read-only coalesced `weights` (`544` `u32` words per `(row_group_64, k_block_128)`),
/// `1` read-only 4-plane `Q8_K` (`34` `u32` words per 128 elements: `[scale_bits, half_sum_i32, 32 plane words]`),
/// `2` writable `output` (`f32`), optional `3` read-only `residual` (`f32`).
/// Push constants (`4 Ã— u32`): `[rows, blocks_per_row, tokens, act_words_per_token]`.
#[must_use]
pub fn pq2_q8k_gemv_coalesced(use_sdot: bool, fused: bool, has_residual: bool) -> Vec<u32> {
    let (mut b, push_var) = ShaderBuilder::new(use_sdot, 4);
    let push_var = push_var.unwrap_or(0);
    let weights_buf = b.storage_buffer(0, ElemType::U32, AccessMode::ReadOnly);
    let act_buf = b.storage_buffer(1, ElemType::U32, AccessMode::ReadOnly);
    let out_buf = b.storage_buffer(2, ElemType::F32, AccessMode::ReadWrite);
    let res_buf = has_residual.then(|| b.storage_buffer(3, ElemType::F32, AccessMode::ReadOnly));

    let gid = b.global_invocation_id();
    let row = b.load_input_component(gid, 0);
    let token = b.load_input_component(gid, 1);

    let rows = b.load_push(push_var, 0);
    let blocks_per_row = b.load_push(push_var, 1);
    let tokens = b.load_push(push_var, 2);
    let act_words_per_token = b.load_push(push_var, 3);

    let row_ok = b.u_lt(row, rows);
    let tok_ok = b.u_lt(token, tokens);
    let in_bounds = b.logical_and(row_ok, tok_ok);

    b.emit_if(in_bounds, |b| {
        let acc_var = b.local_f32_var();
        let f_zero = b.const_f32(0.0);
        b.store_var(acc_var, f_zero);

        let c0 = b.const_u32(0);
        let c1 = b.const_u32(1);
        let c6 = b.const_u32(6);
        let c34 = b.const_u32(34);
        let c63 = b.const_u32(63);
        let c544 = b.const_u32(544);

        let rg = b.ushr(row, c6);
        let lane = b.uand(row, c63);
        let rg_tiles = b.umul(rg, blocks_per_row);
        let rg_base = b.umul(rg_tiles, c544);
        let tok_act_base = b.umul(token, act_words_per_token);

        b.emit_for_range(c0, blocks_per_row, c1, |b, blk| {
            let blk_off = b.umul(blk, c544);
            let tile_base = b.uadd(rg_base, blk_off);
            let w_scale = coalesced_pq2_scale(b, weights_buf, tile_base, lane);

            let act_off = b.umul(blk, c34);
            let act_blk_base = b.uadd(tok_act_base, act_off);
            let act_scale_u32 = b.load_sb_u32(act_buf, act_blk_base);
            let act_scale = b.bitcast(b.f32_t, act_scale_u32);
            let hsum_idx = b.uadd(act_blk_base, c1);
            let hsum_u32 = b.load_sb_u32(act_buf, hsum_idx);
            let half_sum = b.bitcast(b.i32_t, hsum_u32);

            let mut int_sum = b.const_i32(0);
            for word_j in 0_u32..8 {
                let w_off = b.const_u32(32 + word_j * 64);
                let w_lane_off = b.uadd(w_off, lane);
                let w_idx = b.uadd(tile_base, w_lane_off);
                let pq_word = b.load_sb_u32(weights_buf, w_idx);
                let q = coalesced_pq2_unpack_4digits(b, pq_word);

                let plane_off = b.const_u32(2 + word_j * 4);
                let plane_base = b.uadd(act_blk_base, plane_off);
                let part = dot_pq2_4digits_with_act(b, q, act_buf, plane_base, use_sdot);
                int_sum = b.iadd(int_sum, part);
            }

            let corrected = b.isub(int_sum, half_sum);
            let float_dot = b.i32_to_f32(corrected);
            let d = b.fmul_nc(w_scale, act_scale);
            let prev = b.load_var_f32(acc_var);
            let next = if fused {
                b.fma(d, float_dot, prev)
            } else {
                let prod = b.fmul_nc(d, float_dot);
                b.fadd_nc(prev, prod)
            };
            b.store_var(acc_var, next);
        });

        let tok_out = b.umul(token, rows);
        let out_idx = b.uadd(tok_out, row);
        let final_acc = b.load_var_f32(acc_var);
        store_maybe_residual_f32(b, out_buf, res_buf, out_idx, final_acc);
    });

    b.finish()
}

/// Emits coalesced 64-row Ã— 4-token `PQ2_0` Ã— 4-plane `Q8_K` GEMM SPIR-V 1.6 words.
///
/// Dispatch grid: `(rows.div_ceil(64), tokens.div_ceil(4), 1)`. Each weight word is loaded and
/// unpacked once per K-block and reused across up to 4 consecutive tokens cleanly.
/// Push constants (`4 Ã— u32`): `[rows, blocks_per_row, tokens, act_words_per_token]`.
#[must_use]
pub fn pq2_q8k_gemm_4tok_coalesced(use_sdot: bool, fused: bool, has_residual: bool) -> Vec<u32> {
    let (mut b, push_var) = ShaderBuilder::new(use_sdot, 4);
    let push_var = push_var.unwrap_or(0);
    let weights_buf = b.storage_buffer(0, ElemType::U32, AccessMode::ReadOnly);
    let act_buf = b.storage_buffer(1, ElemType::U32, AccessMode::ReadOnly);
    let out_buf = b.storage_buffer(2, ElemType::F32, AccessMode::ReadWrite);
    let res_buf = has_residual.then(|| b.storage_buffer(3, ElemType::F32, AccessMode::ReadOnly));

    let gid = b.global_invocation_id();
    let row = b.load_input_component(gid, 0);
    let tok_tile = b.load_input_component(gid, 1);

    let rows = b.load_push(push_var, 0);
    let blocks_per_row = b.load_push(push_var, 1);
    let tokens = b.load_push(push_var, 2);
    let act_words_per_token = b.load_push(push_var, 3);

    let c4 = b.const_u32(4);
    let tok_base = b.umul(tok_tile, c4);
    let row_ok = b.u_lt(row, rows);
    let tok_ok = b.u_lt(tok_base, tokens);
    let in_bounds = b.logical_and(row_ok, tok_ok);

    b.emit_if(in_bounds, |b| {
        let f_zero = b.const_f32(0.0);
        let acc_vars = [
            b.local_f32_var(),
            b.local_f32_var(),
            b.local_f32_var(),
            b.local_f32_var(),
        ];
        for v in acc_vars {
            b.store_var(v, f_zero);
        }

        let c0 = b.const_u32(0);
        let c1 = b.const_u32(1);
        let c6 = b.const_u32(6);
        let c34 = b.const_u32(34);
        let c63 = b.const_u32(63);
        let c544 = b.const_u32(544);

        let (tok_clamped, tok_act_bases) =
            clamp_four_token_act_bases(b, tok_base, tokens, act_words_per_token);

        let rg = b.ushr(row, c6);
        let lane = b.uand(row, c63);
        let rg_tiles = b.umul(rg, blocks_per_row);
        let rg_base = b.umul(rg_tiles, c544);

        b.emit_for_range(c0, blocks_per_row, c1, |b, blk| {
            let blk_off = b.umul(blk, c544);
            let tile_base = b.uadd(rg_base, blk_off);
            let w_scale = coalesced_pq2_scale(b, weights_buf, tile_base, lane);
            let act_off = b.umul(blk, c34);
            let act_bases = tok_act_bases.map(|base| b.uadd(base, act_off));
            let mut int_sums = [b.const_i32(0); 4];

            for word_j in 0_u32..8 {
                let w_off = b.const_u32(32 + word_j * 64);
                let w_lane_off = b.uadd(w_off, lane);
                let w_idx = b.uadd(tile_base, w_lane_off);
                let pq_word = b.load_sb_u32(weights_buf, w_idx);
                let q = coalesced_pq2_unpack_4digits(b, pq_word);
                let plane_off = b.const_u32(2 + word_j * 4);
                for (t_idx, &a_base) in act_bases.iter().enumerate() {
                    let plane_base = b.uadd(a_base, plane_off);
                    let part = dot_pq2_4digits_with_act(b, q, act_buf, plane_base, use_sdot);
                    int_sums[t_idx] = b.iadd(int_sums[t_idx], part);
                }
            }

            for ((&a_base, &int_sum), &acc_var) in act_bases.iter().zip(&int_sums).zip(&acc_vars) {
                let act_scale_u32 = b.load_sb_u32(act_buf, a_base);
                let act_scale = b.bitcast(b.f32_t, act_scale_u32);
                let hsum_idx = b.uadd(a_base, c1);
                let hsum_u32 = b.load_sb_u32(act_buf, hsum_idx);
                let half_sum = b.bitcast(b.i32_t, hsum_u32);
                let corrected = b.isub(int_sum, half_sum);
                let float_dot = b.i32_to_f32(corrected);
                let d = b.fmul_nc(w_scale, act_scale);
                let prev = b.load_var_f32(acc_var);
                let next = if fused {
                    b.fma(d, float_dot, prev)
                } else {
                    let prod = b.fmul_nc(d, float_dot);
                    b.fadd_nc(prev, prod)
                };
                b.store_var(acc_var, next);
            }
        });

        for (&tk, &acc_var) in tok_clamped.iter().zip(&acc_vars) {
            let valid = b.u_lt(tk, tokens);
            b.emit_if(valid, |b| {
                let tok_out = b.umul(tk, rows);
                let out_idx = b.uadd(tok_out, row);
                let final_acc = b.load_var_f32(acc_var);
                store_maybe_residual_f32(b, out_buf, res_buf, out_idx, final_acc);
            });
        }
    });

    b.finish()
}

/// Emits coalesced 64-row `PQ2_0` Ã— `Q8_0` GEMV SPIR-V 1.6 words (odd `128`-col support).
///
/// Bindings: `0` read-only coalesced `PQ2` weights (`544` `u32`/block-tile), `1` read-only
/// `Q8Block` (`9` `u32`/32-elem block), `2` writable `output` (`f32`).
/// Push constants (`4 Ã— u32`): `[rows, blocks_per_row, tokens, act_words_per_token]`.
#[must_use]
pub fn pq2_q8_0_gemv_coalesced(use_sdot: bool) -> Vec<u32> {
    let (mut b, push_var) = ShaderBuilder::new(use_sdot, 4);
    let push_var = push_var.unwrap_or(0);
    let weights_buf = b.storage_buffer(0, ElemType::U32, AccessMode::ReadOnly);
    let act_buf = b.storage_buffer(1, ElemType::U32, AccessMode::ReadOnly);
    let out_buf = b.storage_buffer(2, ElemType::F32, AccessMode::ReadWrite);

    let gid = b.global_invocation_id();
    let row = b.load_input_component(gid, 0);
    let token = b.load_input_component(gid, 1);

    let rows = b.load_push(push_var, 0);
    let blocks_per_row = b.load_push(push_var, 1);
    let tokens = b.load_push(push_var, 2);
    let act_words_per_token = b.load_push(push_var, 3);

    let row_ok = b.u_lt(row, rows);
    let tok_ok = b.u_lt(token, tokens);
    let in_bounds = b.logical_and(row_ok, tok_ok);

    b.emit_if(in_bounds, |b| {
        let acc_var = b.local_f32_var();
        let f_zero = b.const_f32(0.0);
        b.store_var(acc_var, f_zero);

        let c0 = b.const_u32(0);
        let c1 = b.const_u32(1);
        let c2 = b.const_u32(2);
        let c3 = b.const_u32(3);
        let c6 = b.const_u32(6);
        let c8 = b.const_u32(8);
        let c16 = b.const_u32(16);
        let c24 = b.const_u32(24);
        let c36 = b.const_u32(36);
        let c63 = b.const_u32(63);
        let c544 = b.const_u32(544);

        let rg = b.ushr(row, c6);
        let lane = b.uand(row, c63);
        let rg_tiles = b.umul(rg, blocks_per_row);
        let rg_base = b.umul(rg_tiles, c544);
        let tok_act_base = b.umul(token, act_words_per_token);

        b.emit_for_range(c0, blocks_per_row, c1, |b, blk| {
            let blk_off = b.umul(blk, c544);
            let tile_base = b.uadd(rg_base, blk_off);
            let w_scale = coalesced_pq2_scale(b, weights_buf, tile_base, lane);

            let blk_act_off = b.umul(blk, c36);
            let blk_act_base = b.uadd(tok_act_base, blk_act_off);

            let mut block_sum = f_zero;
            for grp in 0_u32..4 {
                let grp_off_c = b.const_u32(grp * 9);
                let grp_base = b.uadd(blk_act_base, grp_off_c);
                let act_scale_u32 = b.load_sb_u32(act_buf, grp_base);
                let act_scale = b.bitcast(b.f32_t, act_scale_u32);

                let mut grp_int = b.const_i32(0);
                for pair_idx in 0_u32..2 {
                    let word_j = grp * 2 + pair_idx;
                    let w_off = b.const_u32(32 + word_j * 64);
                    let w_lane_off = b.uadd(w_off, lane);
                    let w_idx = b.uadd(tile_base, w_lane_off);
                    let pq_word = b.load_sb_u32(weights_buf, w_idx);

                    let act_w_off = b.const_u32(1 + pair_idx * 4);
                    let act_w_start = b.uadd(grp_base, act_w_off);
                    let aw0 = b.load_sb_u32(act_buf, act_w_start);
                    let a_i1 = b.uadd(act_w_start, c1);
                    let aw1 = b.load_sb_u32(act_buf, a_i1);
                    let a_i2 = b.uadd(act_w_start, c2);
                    let aw2 = b.load_sb_u32(act_buf, a_i2);
                    let a_i3 = b.uadd(act_w_start, c3);
                    let aw3 = b.load_sb_u32(act_buf, a_i3);

                    let b0 = pq_word;
                    let b1 = b.ushr(pq_word, c8);
                    let b2 = b.ushr(pq_word, c16);
                    let b3 = b.ushr(pq_word, c24);

                    for (raw_byte, act_w) in [(b0, aw0), (b1, aw1), (b2, aw2), (b3, aw3)] {
                        let dot = if use_sdot {
                            let packed = swar_pack_4x2bit_to_4xi8(b, raw_byte);
                            b.sdot_packed(packed, act_w)
                        } else {
                            portable_dot4_from_2bit_byte(b, raw_byte, act_w)
                        };
                        grp_int = b.iadd(grp_int, dot);
                    }
                }
                let grp_f32 = b.i32_to_f32(grp_int);
                let scaled = b.fmul_nc(act_scale, grp_f32);
                block_sum = b.fadd_nc(block_sum, scaled);
            }

            let prev = b.load_var_f32(acc_var);
            let contrib = b.fmul_nc(w_scale, block_sum);
            let next = b.fadd_nc(prev, contrib);
            b.store_var(acc_var, next);
        });

        let tok_out = b.umul(token, rows);
        let out_idx = b.uadd(tok_out, row);
        let final_acc = b.load_var_f32(acc_var);
        b.store_sb_f32(out_buf, out_idx, final_acc);
    });

    b.finish()
}

#[expect(
    clippy::too_many_arguments,
    reason = "Maintains clean SPIR-V emission decomposition for 4-token PQ2xQ8_0 groups."
)]
fn pq2_q8_0_4tok_group_step(
    b: &mut ShaderBuilder,
    weights_buf: u32,
    tile_base: u32,
    lane: u32,
    act_buf: u32,
    blk_act_bases: [u32; 4],
    grp: u32,
    use_sdot: bool,
    block_sums: &mut [u32; 4],
) {
    let c1 = b.const_u32(1);
    let c2 = b.const_u32(2);
    let c3 = b.const_u32(3);
    let c8 = b.const_u32(8);
    let c16 = b.const_u32(16);
    let c24 = b.const_u32(24);
    let grp_off_c = b.const_u32(grp * 9);
    let grp_bases = blk_act_bases.map(|base| b.uadd(base, grp_off_c));
    let act_scales = grp_bases.map(|gb| {
        let u = b.load_sb_u32(act_buf, gb);
        b.bitcast(b.f32_t, u)
    });
    let mut grp_ints = [b.const_i32(0); 4];

    for pair_idx in 0_u32..2 {
        let word_j = grp * 2 + pair_idx;
        let w_off = b.const_u32(32 + word_j * 64);
        let w_lane_off = b.uadd(w_off, lane);
        let w_idx = b.uadd(tile_base, w_lane_off);
        let pq_word = b.load_sb_u32(weights_buf, w_idx);
        let raw_bytes = [
            pq_word,
            b.ushr(pq_word, c8),
            b.ushr(pq_word, c16),
            b.ushr(pq_word, c24),
        ];
        let act_w_off = b.const_u32(1 + pair_idx * 4);
        if use_sdot {
            let packed_bytes = raw_bytes.map(|rb| swar_pack_4x2bit_to_4xi8(b, rb));
            for (k, &gb) in grp_bases.iter().enumerate() {
                let a0 = b.uadd(gb, act_w_off);
                let a1 = b.uadd(a0, c1);
                let a2 = b.uadd(a0, c2);
                let a3 = b.uadd(a0, c3);
                for (pb, ai) in packed_bytes.iter().zip([a0, a1, a2, a3]) {
                    let aw = b.load_sb_u32(act_buf, ai);
                    let dot = b.sdot_packed(*pb, aw);
                    grp_ints[k] = b.iadd(grp_ints[k], dot);
                }
            }
        } else {
            let unpacked = raw_bytes.map(|rb| unpack_2bit_byte_to_i32_weights(b, rb));
            for (k, &gb) in grp_bases.iter().enumerate() {
                let a0 = b.uadd(gb, act_w_off);
                let a1 = b.uadd(a0, c1);
                let a2 = b.uadd(a0, c2);
                let a3 = b.uadd(a0, c3);
                for (ws, ai) in unpacked.iter().zip([a0, a1, a2, a3]) {
                    let aw = b.load_sb_u32(act_buf, ai);
                    let dot = portable_dot4_i32_weights(b, *ws, aw);
                    grp_ints[k] = b.iadd(grp_ints[k], dot);
                }
            }
        }
    }

    for (k, &act_scale) in act_scales.iter().enumerate() {
        let grp_f32 = b.i32_to_f32(grp_ints[k]);
        let scaled = b.fmul_nc(act_scale, grp_f32);
        block_sums[k] = b.fadd_nc(block_sums[k], scaled);
    }
}

/// Emits coalesced 64-row `×` 4-token `PQ2_0` `×` `Q8_0` GEMM SPIR-V 1.6 words.
///
/// Dispatch grid: `(rows.div_ceil(64), tokens.div_ceil(4), 1)`. Each weight word is loaded once
/// per K-block and reused across up to 4 consecutive tokens.
/// Push constants (`4 × u32`): `[rows, blocks_per_row, tokens, act_words_per_token]`.
#[must_use]
pub fn pq2_q8_0_gemm_4tok_coalesced(use_sdot: bool) -> Vec<u32> {
    let (mut b, push_var) = ShaderBuilder::new(use_sdot, 4);
    let push_var = push_var.unwrap_or(0);
    let weights_buf = b.storage_buffer(0, ElemType::U32, AccessMode::ReadOnly);
    let act_buf = b.storage_buffer(1, ElemType::U32, AccessMode::ReadOnly);
    let out_buf = b.storage_buffer(2, ElemType::F32, AccessMode::ReadWrite);

    let gid = b.global_invocation_id();
    let row = b.load_input_component(gid, 0);
    let tok_tile = b.load_input_component(gid, 1);

    let rows = b.load_push(push_var, 0);
    let blocks_per_row = b.load_push(push_var, 1);
    let tokens = b.load_push(push_var, 2);
    let act_words_per_token = b.load_push(push_var, 3);

    let c4 = b.const_u32(4);
    let tok_base = b.umul(tok_tile, c4);
    let row_ok = b.u_lt(row, rows);
    let tok_ok = b.u_lt(tok_base, tokens);
    let in_bounds = b.logical_and(row_ok, tok_ok);

    b.emit_if(in_bounds, |b| {
        let f_zero = b.const_f32(0.0);
        let acc_vars = [
            b.local_f32_var(),
            b.local_f32_var(),
            b.local_f32_var(),
            b.local_f32_var(),
        ];
        for v in acc_vars {
            b.store_var(v, f_zero);
        }

        let c0 = b.const_u32(0);
        let c1 = b.const_u32(1);
        let c6 = b.const_u32(6);
        let c36 = b.const_u32(36);
        let c63 = b.const_u32(63);
        let c544 = b.const_u32(544);

        let (tok_clamped, tok_act_bases) =
            clamp_four_token_act_bases(b, tok_base, tokens, act_words_per_token);
        let rg = b.ushr(row, c6);
        let lane = b.uand(row, c63);
        let rg_tiles = b.umul(rg, blocks_per_row);
        let rg_base = b.umul(rg_tiles, c544);

        b.emit_for_range(c0, blocks_per_row, c1, |b, blk| {
            let blk_off = b.umul(blk, c544);
            let tile_base = b.uadd(rg_base, blk_off);
            let w_scale = coalesced_pq2_scale(b, weights_buf, tile_base, lane);
            let blk_act_off = b.umul(blk, c36);
            let blk_act_bases = tok_act_bases.map(|base| b.uadd(base, blk_act_off));

            let mut block_sums = [f_zero; 4];
            for grp in 0_u32..4 {
                pq2_q8_0_4tok_group_step(
                    b,
                    weights_buf,
                    tile_base,
                    lane,
                    act_buf,
                    blk_act_bases,
                    grp,
                    use_sdot,
                    &mut block_sums,
                );
            }

            for (&acc_var, &block_sum) in acc_vars.iter().zip(&block_sums) {
                let prev = b.load_var_f32(acc_var);
                let contrib = b.fmul_nc(w_scale, block_sum);
                let next = b.fadd_nc(prev, contrib);
                b.store_var(acc_var, next);
            }
        });

        for (&tk, &acc_var) in tok_clamped.iter().zip(&acc_vars) {
            let valid = b.u_lt(tk, tokens);
            b.emit_if(valid, |b| {
                let tok_out = b.umul(tk, rows);
                let out_idx = b.uadd(tok_out, row);
                let final_acc = b.load_var_f32(acc_var);
                b.store_sb_f32(out_buf, out_idx, final_acc);
            });
        }
    });

    b.finish()
}

#[allow(
    clippy::too_many_arguments,
    reason = "Maintains clean SPIR-V emission decomposition for 32-element PTQ1xQ8_0 groups."
)]
fn ptq1_q8_0_group_step(
    b: &mut ShaderBuilder,
    decoded: &PtqDecodedBlock,
    act_buf: u32,
    grp_base: u32,
    grp: usize,
    use_sdot: bool,
    pinned_4lane: bool,
    lane_vars: [u32; 4],
    block_sum: &mut u32,
) {
    let act_scale_u32 = b.load_sb_u32(act_buf, grp_base);
    let act_scale = b.bitcast(b.f32_t, act_scale_u32);
    let mut lane_sums = [b.const_i32(0); 4];
    for half in 0_usize..2 {
        for (l, lane_slot) in lane_sums.iter_mut().enumerate() {
            let act_within_grp = u32::try_from(half * 4 + l).unwrap_or(0);
            let act_off_c = b.const_u32(1 + act_within_grp);
            let act_w_idx = b.uadd(grp_base, act_off_c);
            let act_w = b.load_sb_u32(act_buf, act_w_idx);
            let word_idx = grp * 8 + half * 4 + l;
            let dot = dot_trit_lanes_with_act(b, decoded.lanes[word_idx], act_w, use_sdot);
            *lane_slot = b.iadd(*lane_slot, dot);
        }
    }
    if pinned_4lane {
        let d = b.fmul_nc(decoded.w_scale, act_scale);
        for (&lane_var, &sum_val) in lane_vars.iter().zip(&lane_sums) {
            let lane_f32 = b.i32_to_f32(sum_val);
            let prev = b.load_var_f32(lane_var);
            let next = b.fma(lane_f32, d, prev);
            b.store_var(lane_var, next);
        }
    } else {
        let s01 = b.iadd(lane_sums[0], lane_sums[1]);
        let s23 = b.iadd(lane_sums[2], lane_sums[3]);
        let grp_int = b.iadd(s01, s23);
        let grp_sum_f32 = b.i32_to_f32(grp_int);
        let scaled = b.fmul_nc(act_scale, grp_sum_f32);
        *block_sum = b.fadd_nc(*block_sum, scaled);
    }
}

/// Emits coalesced 64-row `PTQ1_0` Ã— `Q8_0` GEMV SPIR-V 1.6 words (`0%` weight overhead).
///
/// Bindings: `0` read-only coalesced `PTQ1_0` (`448` `u32`/block-tile), `1` read-only `Q8Block`
/// (`9` `u32`/32-elem block), `2` writable `output` (`f32`).
/// Push constants (`4 Ã— u32`): `[rows, blocks_per_row, tokens, act_words_per_token]`.
#[must_use]
pub fn ptq1_q8_0_gemv_coalesced(use_sdot: bool, pinned_4lane: bool) -> Vec<u32> {
    let (mut b, push_var) = ShaderBuilder::new(use_sdot, 4);
    let push_var = push_var.unwrap_or(0);
    let weights_buf = b.storage_buffer(0, ElemType::U32, AccessMode::ReadOnly);
    let act_buf = b.storage_buffer(1, ElemType::U32, AccessMode::ReadOnly);
    let out_buf = b.storage_buffer(2, ElemType::F32, AccessMode::ReadWrite);

    let gid = b.global_invocation_id();
    let row = b.load_input_component(gid, 0);
    let token = b.load_input_component(gid, 1);

    let rows = b.load_push(push_var, 0);
    let blocks_per_row = b.load_push(push_var, 1);
    let tokens = b.load_push(push_var, 2);
    let act_words_per_token = b.load_push(push_var, 3);

    let row_ok = b.u_lt(row, rows);
    let tok_ok = b.u_lt(token, tokens);
    let in_bounds = b.logical_and(row_ok, tok_ok);

    b.emit_if(in_bounds, |b| {
        let f_zero = b.const_f32(0.0);
        let lane_vars = [
            b.local_f32_var(),
            b.local_f32_var(),
            b.local_f32_var(),
            b.local_f32_var(),
        ];
        for var in lane_vars {
            b.store_var(var, f_zero);
        }

        let c0 = b.const_u32(0);
        let c1 = b.const_u32(1);
        let c6 = b.const_u32(6);
        let c36 = b.const_u32(36);
        let c63 = b.const_u32(63);
        let c448 = b.const_u32(448);

        let rg = b.ushr(row, c6);
        let lane = b.uand(row, c63);
        let rg_tiles = b.umul(rg, blocks_per_row);
        let rg_base = b.umul(rg_tiles, c448);
        let tok_act_base = b.umul(token, act_words_per_token);

        b.emit_for_range(c0, blocks_per_row, c1, |b, blk| {
            let blk_off = b.umul(blk, c448);
            let tile_base = b.uadd(rg_base, blk_off);
            let wbase = b.uadd(tile_base, lane);
            let decoded = decode_ptq1_block_lanes_strided(b, weights_buf, wbase, 64);

            let blk_act_off = b.umul(blk, c36);
            let blk_act_base = b.uadd(tok_act_base, blk_act_off);
            let mut block_sum = f_zero;

            for grp in 0_usize..4 {
                let grp_index = u32::try_from(grp).unwrap_or(0);
                let grp_off_c = b.const_u32(grp_index * 9);
                let grp_base = b.uadd(blk_act_base, grp_off_c);
                ptq1_q8_0_group_step(
                    b,
                    &decoded,
                    act_buf,
                    grp_base,
                    grp,
                    use_sdot,
                    pinned_4lane,
                    lane_vars,
                    &mut block_sum,
                );
            }

            if !pinned_4lane {
                let prev = b.load_var_f32(lane_vars[0]);
                let contrib = b.fmul_nc(decoded.w_scale, block_sum);
                let next = b.fadd_nc(prev, contrib);
                b.store_var(lane_vars[0], next);
            }
        });

        let final_val = if pinned_4lane {
            let l0 = b.load_var_f32(lane_vars[0]);
            let l1 = b.load_var_f32(lane_vars[1]);
            let l2 = b.load_var_f32(lane_vars[2]);
            let l3 = b.load_var_f32(lane_vars[3]);
            let s01 = b.fadd_nc(l0, l1);
            let s23 = b.fadd_nc(l2, l3);
            b.fadd_nc(s01, s23)
        } else {
            b.load_var_f32(lane_vars[0])
        };
        let tok_out = b.umul(token, rows);
        let out_idx = b.uadd(tok_out, row);
        b.store_sb_f32(out_buf, out_idx, final_val);
    });

    b.finish()
}

#[expect(
    clippy::too_many_arguments,
    reason = "Maintains clean SPIR-V emission decomposition for 4-token 32-element PTQ1xQ8_0 groups."
)]
fn ptq1_q8_0_4tok_group_step(
    b: &mut ShaderBuilder,
    decoded: &PtqDecodedBlock,
    act_buf: u32,
    grp_bases: [u32; 4],
    grp: usize,
    use_sdot: bool,
    pinned_4lane: bool,
    lane_vars: &[[u32; 4]; 4],
    block_sums: &mut [u32; 4],
) {
    let act_scales = grp_bases.map(|gb| {
        let u = b.load_sb_u32(act_buf, gb);
        b.bitcast(b.f32_t, u)
    });
    let mut lane_sums = [[b.const_i32(0); 4]; 4];
    for half in 0_usize..2 {
        for l in 0_usize..4 {
            let act_within_grp = u32::try_from(half * 4 + l).unwrap_or(0);
            let act_off_c = b.const_u32(1 + act_within_grp);
            let word_idx = grp * 8 + half * 4 + l;
            if use_sdot {
                let packed_w = swar_bias_minus_one(b, decoded.lanes[word_idx]);
                for (k, &gb) in grp_bases.iter().enumerate() {
                    let act_w_idx = b.uadd(gb, act_off_c);
                    let act_w = b.load_sb_u32(act_buf, act_w_idx);
                    let dot = b.sdot_packed(packed_w, act_w);
                    lane_sums[k][l] = b.iadd(lane_sums[k][l], dot);
                }
            } else {
                let w_i32 = unpack_packed_trits_to_i32_weights(b, decoded.lanes[word_idx]);
                for (k, &gb) in grp_bases.iter().enumerate() {
                    let act_w_idx = b.uadd(gb, act_off_c);
                    let act_w = b.load_sb_u32(act_buf, act_w_idx);
                    let dot = portable_dot4_i32_weights(b, w_i32, act_w);
                    lane_sums[k][l] = b.iadd(lane_sums[k][l], dot);
                }
            }
        }
    }

    for k in 0_usize..4 {
        if pinned_4lane {
            let d = b.fmul_nc(decoded.w_scale, act_scales[k]);
            for (&lane_var, &sum_val) in lane_vars[k].iter().zip(&lane_sums[k]) {
                let lane_f32 = b.i32_to_f32(sum_val);
                let prev = b.load_var_f32(lane_var);
                let next = b.fma(lane_f32, d, prev);
                b.store_var(lane_var, next);
            }
        } else {
            let s01 = b.iadd(lane_sums[k][0], lane_sums[k][1]);
            let s23 = b.iadd(lane_sums[k][2], lane_sums[k][3]);
            let grp_int = b.iadd(s01, s23);
            let grp_sum_f32 = b.i32_to_f32(grp_int);
            let scaled = b.fmul_nc(act_scales[k], grp_sum_f32);
            block_sums[k] = b.fadd_nc(block_sums[k], scaled);
        }
    }
}

/// Emits coalesced 64-row `×` 4-token `PTQ1_0` `×` `Q8_0` GEMM SPIR-V 1.6 words (`pinned_4lane` or standard).
///
/// Dispatch grid: `(rows.div_ceil(64), tokens.div_ceil(4), 1)`. Decodes each `PTQ1_0` block once
/// per K-block and reuses decoded trits across up to 4 consecutive tokens.
/// Push constants (`4 × u32`): `[rows, blocks_per_row, tokens, act_words_per_token]`.
#[must_use]
#[expect(
    clippy::too_many_lines,
    reason = "Builds complete SPIR-V 1.6 compute module for 4-token coalesced PTQ1 x Q8_0 GEMM with lane-accumulator setup and tail store."
)]
pub fn ptq1_q8_0_gemm_4tok_coalesced(use_sdot: bool, pinned_4lane: bool) -> Vec<u32> {
    let (mut b, push_var) = ShaderBuilder::new(use_sdot, 4);
    let push_var = push_var.unwrap_or(0);
    let weights_buf = b.storage_buffer(0, ElemType::U32, AccessMode::ReadOnly);
    let act_buf = b.storage_buffer(1, ElemType::U32, AccessMode::ReadOnly);
    let out_buf = b.storage_buffer(2, ElemType::F32, AccessMode::ReadWrite);

    let gid = b.global_invocation_id();
    let row = b.load_input_component(gid, 0);
    let tok_tile = b.load_input_component(gid, 1);

    let rows = b.load_push(push_var, 0);
    let blocks_per_row = b.load_push(push_var, 1);
    let tokens = b.load_push(push_var, 2);
    let act_words_per_token = b.load_push(push_var, 3);

    let c4 = b.const_u32(4);
    let tok_base = b.umul(tok_tile, c4);
    let row_ok = b.u_lt(row, rows);
    let tok_ok = b.u_lt(tok_base, tokens);
    let in_bounds = b.logical_and(row_ok, tok_ok);

    b.emit_if(in_bounds, |b| {
        let f_zero = b.const_f32(0.0);
        let lane_vars = if pinned_4lane {
            std::array::from_fn(|_| {
                [
                    b.local_f32_var(),
                    b.local_f32_var(),
                    b.local_f32_var(),
                    b.local_f32_var(),
                ]
            })
        } else {
            std::array::from_fn(|_| {
                let v = b.local_f32_var();
                [v, v, v, v]
            })
        };
        for row_vars in &lane_vars {
            let count = if pinned_4lane { 4 } else { 1 };
            for &var in &row_vars[..count] {
                b.store_var(var, f_zero);
            }
        }

        let c0 = b.const_u32(0);
        let c1 = b.const_u32(1);
        let c6 = b.const_u32(6);
        let c36 = b.const_u32(36);
        let c63 = b.const_u32(63);
        let c448 = b.const_u32(448);

        let (tok_clamped, tok_act_bases) =
            clamp_four_token_act_bases(b, tok_base, tokens, act_words_per_token);
        let rg = b.ushr(row, c6);
        let lane = b.uand(row, c63);
        let rg_tiles = b.umul(rg, blocks_per_row);
        let rg_base = b.umul(rg_tiles, c448);

        b.emit_for_range(c0, blocks_per_row, c1, |b, blk| {
            let blk_off = b.umul(blk, c448);
            let tile_base = b.uadd(rg_base, blk_off);
            let wbase = b.uadd(tile_base, lane);
            let decoded = decode_ptq1_block_lanes_strided(b, weights_buf, wbase, 64);

            let blk_act_off = b.umul(blk, c36);
            let blk_act_bases = tok_act_bases.map(|base| b.uadd(base, blk_act_off));
            let mut block_sums = [f_zero; 4];

            for grp in 0_usize..4 {
                let grp_index = u32::try_from(grp).unwrap_or(0);
                let grp_off_c = b.const_u32(grp_index * 9);
                let grp_bases = blk_act_bases.map(|base| b.uadd(base, grp_off_c));
                ptq1_q8_0_4tok_group_step(
                    b,
                    &decoded,
                    act_buf,
                    grp_bases,
                    grp,
                    use_sdot,
                    pinned_4lane,
                    &lane_vars,
                    &mut block_sums,
                );
            }

            if !pinned_4lane {
                for (row_vars, &block_sum) in lane_vars.iter().zip(&block_sums) {
                    let prev = b.load_var_f32(row_vars[0]);
                    let contrib = b.fmul_nc(decoded.w_scale, block_sum);
                    let next = b.fadd_nc(prev, contrib);
                    b.store_var(row_vars[0], next);
                }
            }
        });

        for (&tk, row_vars) in tok_clamped.iter().zip(&lane_vars) {
            let valid = b.u_lt(tk, tokens);
            b.emit_if(valid, |b| {
                let final_val = if pinned_4lane {
                    let l0 = b.load_var_f32(row_vars[0]);
                    let l1 = b.load_var_f32(row_vars[1]);
                    let l2 = b.load_var_f32(row_vars[2]);
                    let l3 = b.load_var_f32(row_vars[3]);
                    let s01 = b.fadd_nc(l0, l1);
                    let s23 = b.fadd_nc(l2, l3);
                    b.fadd_nc(s01, s23)
                } else {
                    b.load_var_f32(row_vars[0])
                };
                let tok_out = b.umul(tk, rows);
                let out_idx = b.uadd(tok_out, row);
                b.store_sb_f32(out_buf, out_idx, final_val);
            });
        }
    });

    b.finish()
}

/// Emits coalesced 64-row `PTQ1_0` Ã— `Q8_K` GEMV SPIR-V 1.6 words (`0%` weight overhead).
///
/// Bindings: `0` read-only coalesced `PTQ1_0` (`448` `u32`/block-tile), `1` read-only `Q8KBlock`
/// (`65` `u32`/block), `2` writable `output` (`f32`).
/// Push constants (`4 Ã— u32`): `[rows, blocks_per_row, tokens, act_words_per_token]`.
#[must_use]
pub fn ptq1_q8k_gemv_coalesced(use_sdot: bool) -> Vec<u32> {
    let (mut b, push_var) = ShaderBuilder::new(use_sdot, 4);
    let push_var = push_var.unwrap_or(0);
    let weights_buf = b.storage_buffer(0, ElemType::U32, AccessMode::ReadOnly);
    let act_buf = b.storage_buffer(1, ElemType::U32, AccessMode::ReadOnly);
    let out_buf = b.storage_buffer(2, ElemType::F32, AccessMode::ReadWrite);

    let gid = b.global_invocation_id();
    let row = b.load_input_component(gid, 0);
    let token = b.load_input_component(gid, 1);

    let rows = b.load_push(push_var, 0);
    let blocks_per_row = b.load_push(push_var, 1);
    let tokens = b.load_push(push_var, 2);
    let act_words_per_token = b.load_push(push_var, 3);

    let row_ok = b.u_lt(row, rows);
    let tok_ok = b.u_lt(token, tokens);
    let in_bounds = b.logical_and(row_ok, tok_ok);

    b.emit_if(in_bounds, |b| {
        let acc_var = b.local_f32_var();
        let f_zero = b.const_f32(0.0);
        b.store_var(acc_var, f_zero);

        let c0 = b.const_u32(0);
        let c1 = b.const_u32(1);
        let c6 = b.const_u32(6);
        let c32 = b.const_u32(32);
        let c63 = b.const_u32(63);
        let c65 = b.const_u32(65);
        let c448 = b.const_u32(448);

        let rg = b.ushr(row, c6);
        let lane = b.uand(row, c63);
        let rg_tiles = b.umul(rg, blocks_per_row);
        let rg_base = b.umul(rg_tiles, c448);
        let tok_act_base = b.umul(token, act_words_per_token);

        b.emit_for_range(c0, blocks_per_row, c1, |b, blk| {
            let blk_off = b.umul(blk, c448);
            let tile_base = b.uadd(rg_base, blk_off);
            let wbase = b.uadd(tile_base, lane);
            let decoded = decode_ptq1_block_lanes_strided(b, weights_buf, wbase, 64);

            let q8k_idx = b.ushr(blk, c1);
            let q8k_off = b.umul(q8k_idx, c65);
            let q8k_base = b.uadd(tok_act_base, q8k_off);
            let act_scale_u32 = b.load_sb_u32(act_buf, q8k_base);
            let act_scale = b.bitcast(b.f32_t, act_scale_u32);

            let half = b.uand(blk, c1);
            let half_off = b.umul(half, c32);
            let q8k_vals_start = b.uadd(q8k_base, c1);
            let act_vals_base = b.uadd(q8k_vals_start, half_off);

            let mut sum_int = b.const_i32(0);
            for (word_idx, &lanes) in decoded.lanes.iter().enumerate() {
                let off_c = b.const_u32(u32::try_from(word_idx).unwrap_or(0));
                let act_idx = b.uadd(act_vals_base, off_c);
                let act_w = b.load_sb_u32(act_buf, act_idx);
                let dot = dot_trit_lanes_with_act(b, lanes, act_w, use_sdot);
                sum_int = b.iadd(sum_int, dot);
            }

            let sum_f32 = b.i32_to_f32(sum_int);
            let d = b.fmul_nc(decoded.w_scale, act_scale);
            let prev = b.load_var_f32(acc_var);
            let prod = b.fmul_nc(d, sum_f32);
            let next = b.fadd_nc(prev, prod);
            b.store_var(acc_var, next);
        });

        let tok_out = b.umul(token, rows);
        let out_idx = b.uadd(tok_out, row);
        let final_acc = b.load_var_f32(acc_var);
        b.store_sb_f32(out_buf, out_idx, final_acc);
    });

    b.finish()
}

fn q8k_signed_abs_max_shared(
    b: &mut ShaderBuilder,
    smem_val: u32,
    smem_idx: u32,
    lid: u32,
    local_best: u32,
    local_idx: u32,
) -> u32 {
    let c0 = b.const_u32(0);
    let best_bits = b.bitcast(b.u32_t, local_best);
    b.store_wg_u32(smem_val, lid, best_bits);
    b.store_wg_u32(smem_idx, lid, local_idx);
    b.workgroup_barrier();

    for stride in [32_u32, 16, 8, 4, 2, 1] {
        let sc = b.const_u32(stride);
        let active = b.u_lt(lid, sc);
        b.emit_if(active, |b| {
            let oth_slot = b.uadd(lid, sc);
            let cur_bits = b.load_wg_u32(smem_val, lid);
            let oth_bits = b.load_wg_u32(smem_val, oth_slot);
            let cur_idx = b.load_wg_u32(smem_idx, lid);
            let oth_idx = b.load_wg_u32(smem_idx, oth_slot);
            let cur_f = b.bitcast(b.f32_t, cur_bits);
            let oth_f = b.bitcast(b.f32_t, oth_bits);
            let abs_c = b.f_abs(cur_f);
            let abs_o = b.f_abs(oth_f);
            let abs_gt = b.f_gt(abs_o, abs_c);
            let abs_eq = b.f_eq(abs_o, abs_c);
            let idx_lt = b.u_lt(oth_idx, cur_idx);
            let eq_earlier = b.logical_and(abs_eq, idx_lt);
            let take_oth = b.logical_or(abs_gt, eq_earlier);
            let chosen_val = b.select(b.u32_t, take_oth, oth_bits, cur_bits);
            let chosen_idx = b.select(b.u32_t, take_oth, oth_idx, cur_idx);
            b.store_wg_u32(smem_val, lid, chosen_val);
            b.store_wg_u32(smem_idx, lid, chosen_idx);
        });
        b.workgroup_barrier();
    }

    let max_u32 = b.load_wg_u32(smem_val, c0);
    b.workgroup_barrier();
    b.bitcast(b.f32_t, max_u32)
}

fn veltkamp_split(b: &mut ShaderBuilder, x: u32) -> (u32, u32) {
    let split_c = b.const_f32(4097.0);
    let t = b.fmul_nc(x, split_c);
    let hi_diff = b.fsub_nc(t, x);
    let hi = b.fsub_nc(t, hi_diff);
    let lo = b.fsub_nc(x, hi);
    (hi, lo)
}

fn exact_fdiv_dekker(b: &mut ShaderBuilder, numer: u32, denom: u32) -> u32 {
    let q0 = b.fdiv(numer, denom);
    let (d_hi, d_lo) = veltkamp_split(b, denom);
    let (q_hi, q_lo) = veltkamp_split(b, q0);
    let p_sum = b.fmul_nc(denom, q0);
    let term_hh = b.fmul_nc(d_hi, q_hi);
    let diag_dhi_qlo = b.fmul_nc(d_hi, q_lo);
    let diag_dlo_qhi = b.fmul_nc(d_lo, q_hi);
    let tail_ll = b.fmul_nc(d_lo, q_lo);
    let e0 = b.fsub_nc(term_hh, p_sum);
    let e1 = b.fadd_nc(e0, diag_dhi_qlo);
    let e2 = b.fadd_nc(e1, diag_dlo_qhi);
    let err = b.fadd_nc(e2, tail_ll);
    let diff = b.fsub_nc(numer, p_sum);
    let rem = b.fsub_nc(diff, err);
    let delta = b.fdiv(rem, denom);
    b.fadd_nc(q0, delta)
}

fn q8k_scale_from_max(b: &mut ShaderBuilder, maximum: u32) -> (u32, u32, u32) {
    let abs_max = b.f_abs(maximum);
    let f_zero = b.const_f32(0.0);
    let f_one = b.const_f32(1.0);
    let neg_127 = b.const_f32(-127.0);
    let is_zero = b.f_eq(abs_max, f_zero);
    let safe_max = b.select(b.f32_t, is_zero, f_one, maximum);
    let inv_candidate = exact_fdiv_dekker(b, neg_127, safe_max);
    let iscale = b.select(b.f32_t, is_zero, f_zero, inv_candidate);
    let safe_iscale = b.select(b.f32_t, is_zero, f_one, iscale);
    let unit_scale = exact_fdiv_dekker(b, f_one, safe_iscale);
    let scale = b.select(b.f32_t, is_zero, f_zero, unit_scale);
    (is_zero, iscale, scale)
}

#[allow(
    clippy::too_many_arguments,
    reason = "Maintains clean SPIR-V emission separation for 128-element half-sum header reduction."
)]
fn reduce_and_store_q8k_header(
    b: &mut ShaderBuilder,
    smem: u32,
    lid: u32,
    word_in_sub: u32,
    qs: [u32; 4],
    out_buf: u32,
    out_blk_base: u32,
    scale: u32,
) {
    let q01 = b.iadd(qs[0], qs[1]);
    let q23 = b.iadd(qs[2], qs[3]);
    let sum4 = b.iadd(q01, q23);
    let sum4_u32 = b.bitcast(b.u32_t, sum4);
    b.store_wg_u32(smem, lid, sum4_u32);
    b.workgroup_barrier();

    for stride in [16_u32, 8, 4, 2, 1] {
        let sc = b.const_u32(stride);
        let active = b.u_lt(word_in_sub, sc);
        b.emit_if(active, |b| {
            let oth_slot = b.uadd(lid, sc);
            let cur_u = b.load_wg_u32(smem, lid);
            let oth_u = b.load_wg_u32(smem, oth_slot);
            let cur_i = b.bitcast(b.i32_t, cur_u);
            let oth_i = b.bitcast(b.i32_t, oth_u);
            let sum_i = b.iadd(cur_i, oth_i);
            let sum_u = b.bitcast(b.u32_t, sum_i);
            b.store_wg_u32(smem, lid, sum_u);
        });
        b.workgroup_barrier();
    }

    let c0 = b.const_u32(0);
    let c1 = b.const_u32(1);
    let is_sub_head = b.u_eq(word_in_sub, c0);
    b.emit_if(is_sub_head, |b| {
        let scale_bits = b.bitcast(b.u32_t, scale);
        b.store_sb_u32(out_buf, out_blk_base, scale_bits);
        let hsum_u = b.load_wg_u32(smem, lid);
        let hsum_dst = b.uadd(out_blk_base, c1);
        b.store_sb_u32(out_buf, hsum_dst, hsum_u);
    });
}

fn quantize_one_q8k_lane(
    b: &mut ShaderBuilder,
    val: u32,
    iscale: u32,
    is_zero: u32,
    pinned: bool,
) -> u32 {
    let magic = b.const_f32(12_582_912.0);
    let mask_mant = b.const_u32(0x007F_FFFF);
    let bias = b.const_i32(0x0040_0000);
    let max_code = b.const_i32(127);
    let min_code = b.const_i32(if pinned { -128 } else { -127 });
    let zero_i = b.const_i32(0);

    let rounded = if pinned {
        b.fma(iscale, val, magic)
    } else {
        let prod = b.fmul_nc(val, iscale);
        b.fadd_nc(prod, magic)
    };
    let r_bits = b.bitcast(b.u32_t, rounded);
    let mant = b.uand(r_bits, mask_mant);
    let mant_i = b.bitcast(b.i32_t, mant);
    let raw_q = b.isub(mant_i, bias);
    let q_lo = b.s_max(raw_q, min_code);
    let q_clamped = b.s_min(q_lo, max_code);
    b.select(b.i32_t, is_zero, zero_i, q_clamped)
}

fn pack_four_i8_to_u32(b: &mut ShaderBuilder, qs: [u32; 4]) -> u32 {
    let m_ff = b.const_u32(0xFF);
    let c8 = b.const_u32(8);
    let c16 = b.const_u32(16);
    let c24 = b.const_u32(24);
    let mut bytes = [0_u32; 4];
    for (i, &q) in qs.iter().enumerate() {
        let bits = b.bitcast(b.u32_t, q);
        bytes[i] = b.uand(bits, m_ff);
    }
    let s1 = b.ushl(bytes[1], c8);
    let s2 = b.ushl(bytes[2], c16);
    let s3 = b.ushl(bytes[3], c24);
    let p01 = b.uor(bytes[0], s1);
    let p23 = b.uor(s2, s3);
    b.uor(p01, p23)
}

/// Emits pure SPIR-V 1.6 `F32 -> 4-plane Q8_K` quantizer matching CPU `q8k_block` bit-for-bit.
///
/// Bindings: `0` read-only `input` (`f32`, `tokens * cols`), `1` writable `q8k_4plane` (`u32`,
/// `tokens * (cols / 128) * 34`).
/// Push constants (`2 Ã— u32`): `[cols, tokens]`. Dispatch: `(cols / 256, tokens, 1)`.
#[must_use]
pub fn quantize_f32_to_q8k_4plane(pinned: bool) -> Vec<u32> {
    let (mut b, push_var) = ShaderBuilder::new(false, 2);
    let push_var = push_var.unwrap_or(0);
    let in_buf = b.storage_buffer(0, ElemType::F32, AccessMode::ReadOnly);
    let out_buf = b.storage_buffer(1, ElemType::U32, AccessMode::ReadWrite);
    let smem = b.workgroup_u32_array(WORKGROUP_X);
    let smem_idx = b.workgroup_u32_array(WORKGROUP_X);

    let lid_vec = b.local_invocation_id();
    let wid_vec = b.workgroup_id();
    let lid = b.load_input_component(lid_vec, 0);
    let blk256 = b.load_input_component(wid_vec, 0);
    let tok = b.load_input_component(wid_vec, 1);
    let cols = b.load_push(push_var, 0);

    let c1 = b.const_u32(1);
    let c2 = b.const_u32(2);
    let c3 = b.const_u32(3);
    let c4 = b.const_u32(4);
    let c5 = b.const_u32(5);
    let c7 = b.const_u32(7);
    let c8 = b.const_u32(8);
    let c12 = b.const_u32(12);
    let c31 = b.const_u32(31);
    let c34 = b.const_u32(34);
    let sub_blk = b.ushr(lid, c5);
    let word_in_sub = b.uand(lid, c31);
    let word_j = b.ushr(word_in_sub, c2);
    let digit = b.uand(word_in_sub, c3);
    let sub_off = b.ushl(sub_blk, c7);
    let word_off = b.ushl(word_j, c4);
    let base_sw = b.uadd(sub_off, word_off);
    let elem_local = b.uadd(base_sw, digit);

    let tok_in_base = b.umul(tok, cols);
    let blk_in_off = b.const_u32(256);
    let blk_in_base = b.umul(blk256, blk_in_off);
    let in_blk_base = b.uadd(tok_in_base, blk_in_base);
    let e0 = b.uadd(in_blk_base, elem_local);
    let e1 = b.uadd(e0, c4);
    let e2 = b.uadd(e0, c8);
    let e3 = b.uadd(e0, c12);

    let v0 = b.load_sb_f32(in_buf, e0);
    let v1 = b.load_sb_f32(in_buf, e1);
    let v2 = b.load_sb_f32(in_buf, e2);
    let v3 = b.load_sb_f32(in_buf, e3);

    let mut best_val = v0;
    let mut best_idx = elem_local;
    for (candidate, cand_off) in [(v1, c4), (v2, c8), (v3, c12)] {
        let abs_b = b.f_abs(best_val);
        let abs_c = b.f_abs(candidate);
        let gt = b.f_gt(abs_c, abs_b);
        best_val = b.select(b.f32_t, gt, candidate, best_val);
        let cand_idx = b.uadd(elem_local, cand_off);
        best_idx = b.select(b.u32_t, gt, cand_idx, best_idx);
    }

    let maximum = q8k_signed_abs_max_shared(&mut b, smem, smem_idx, lid, best_val, best_idx);
    let (is_zero, iscale, scale) = q8k_scale_from_max(&mut b, maximum);

    let q0 = quantize_one_q8k_lane(&mut b, v0, iscale, is_zero, pinned);
    let q1 = quantize_one_q8k_lane(&mut b, v1, iscale, is_zero, pinned);
    let q2 = quantize_one_q8k_lane(&mut b, v2, iscale, is_zero, pinned);
    let q3 = quantize_one_q8k_lane(&mut b, v3, iscale, is_zero, pinned);

    let packed = pack_four_i8_to_u32(&mut b, [q0, q1, q2, q3]);

    let blocks128_per_tok = b.ushr(cols, c7);
    let tok_blk128_off = b.umul(tok, blocks128_per_tok);
    let blk128_hi = b.ushl(blk256, c1);
    let blk128_in_tok = b.uadd(blk128_hi, sub_blk);
    let blk128_idx = b.uadd(tok_blk128_off, blk128_in_tok);
    let out_blk_base = b.umul(blk128_idx, c34);
    let plane_dst_off = b.uadd(c2, word_in_sub);
    let plane_dst = b.uadd(out_blk_base, plane_dst_off);
    b.store_sb_u32(out_buf, plane_dst, packed);

    reduce_and_store_q8k_header(
        &mut b,
        smem,
        lid,
        word_in_sub,
        [q0, q1, q2, q3],
        out_buf,
        out_blk_base,
        scale,
    );

    b.finish()
}

/// Emits batched multi-token `F32 -> 4-plane Q8_K` quantizer SPIR-V 1.6 words (`[T, cols]`).
///
/// Bindings: `0` read-only `input` (`f32`, `tokens * cols`), `1` writable `q8k_4plane` (`u32`,
/// `tokens * (cols / 128) * 34`).
/// Push constants (`2 × u32`): `[cols, tokens]`. Dispatch: `(cols / 256, tokens, 1)`.
#[must_use]
pub fn quantize_f32_to_q8k_4plane_batched(pinned: bool) -> Vec<u32> {
    quantize_f32_to_q8k_4plane(pinned)
}

fn vector_exp_spirv(b: &mut ShaderBuilder, value: u32) -> u32 {
    let f_zero = b.const_f32(0.0);
    let magic = b.const_f32(12_582_912.0);
    let c_log2e = b.const_f32(f32::from_bits(0x3fb8_aa3b));
    let c_ln2_hi = b.const_f32(f32::from_bits(0x3f31_7200));
    let c_ln2_lo = b.const_f32(f32::from_bits(0x35bf_be8e));
    let c_p0 = b.const_f32(f32::from_bits(0x3c07_2010));
    let c_p1 = b.const_f32(f32::from_bits(0x3d2b_9f17));
    let c_p2 = b.const_f32(f32::from_bits(0x3e2a_af33));
    let c_p3 = b.const_f32(f32::from_bits(0x3eff_fedb));
    let c_p4 = b.const_f32(f32::from_bits(0x3f7f_fff6));
    let c23 = b.const_u32(23);
    let c_one_bits = b.const_u32(1.0_f32.to_bits());
    let c_126 = b.const_f32(126.0);
    let c_192 = b.const_f32(192.0);
    let c0_u32 = b.const_u32(0);
    let c_shift = b.const_u32(0x8200_0000);
    let c_7f = b.const_u32(0x7f00_0000);

    let z = b.fma(value, c_log2e, magic);
    let n = b.fsub_nc(z, magic);
    let neg_n = b.fsub_nc(f_zero, n);
    let rem1 = b.fma(neg_n, c_ln2_hi, value);
    let rem = b.fma(neg_n, c_ln2_lo, rem1);
    let z_bits = b.bitcast(b.u32_t, z);
    let exponent = b.ushl(z_bits, c23);
    let k_bits = b.uadd(exponent, c_one_bits);
    let k = b.bitcast(b.f32_t, k_bits);
    let sq = b.fmul_nc(rem, rem);
    let p_a = b.fma(c_p0, rem, c_p1);
    let p_b = b.fma(c_p2, rem, c_p3);
    let inner = b.fma(p_a, sq, p_b);
    let p_tail = b.fmul_nc(c_p4, rem);
    let poly = b.fma(inner, sq, p_tail);
    let normal = b.fma(poly, k, k);

    let abs_n = b.f_abs(n);
    let gt_126 = b.f_gt(abs_n, c_126);
    let gt_192 = b.f_gt(abs_n, c_192);
    let n_gt_zero = b.f_gt(n, f_zero);
    let shift = b.select(b.u32_t, n_gt_zero, c0_u32, c_shift);
    let s1_bits = b.uadd(shift, c_7f);
    let s1 = b.bitcast(b.f32_t, s1_bits);
    let s1_sq = b.fmul_nc(s1, s1);
    let s2_bits = b.binop(OP_I_SUB, b.u32_t, exponent, shift);
    let s2 = b.bitcast(b.f32_t, s2_bits);
    let s2_poly = b.fma(s2, poly, s2);
    let scaled = b.fmul_nc(s2_poly, s1);
    let over = b.select(b.f32_t, gt_192, s1_sq, scaled);
    b.select(b.f32_t, gt_126, over, normal)
}

/// Emits GPU `SwiGLU` activation (`u * silu(g)` / `(g * u) / (1 + exp(-g))`) SPIR-V 1.6 words.
///
/// Bindings: `0` read-only `gate` (`f32`), `1` read-only `up` (`f32`), `2` writable `out` (`f32`).
/// Push constants (`1 Ã— u32`): `[count]`. Dispatch: `(count.div_ceil(64), 1, 1)`.
#[must_use]
pub fn ffn_swiglu_f32(pinned: bool) -> Vec<u32> {
    let (mut b, push_var) = ShaderBuilder::new(false, 1);
    let push_var = push_var.unwrap_or(0);
    let gate_buf = b.storage_buffer(0, ElemType::F32, AccessMode::ReadOnly);
    let up_buf = b.storage_buffer(1, ElemType::F32, AccessMode::ReadOnly);
    let out_buf = b.storage_buffer(2, ElemType::F32, AccessMode::ReadWrite);

    let gid = b.global_invocation_id();
    let idx = b.load_input_component(gid, 0);
    let count = b.load_push(push_var, 0);

    let in_bounds = b.u_lt(idx, count);
    b.emit_if(in_bounds, |b| {
        let f_zero = b.const_f32(0.0);
        let f_one = b.const_f32(1.0);
        let g = b.load_sb_f32(gate_buf, idx);
        let u = b.load_sb_f32(up_buf, idx);
        let neg_g = b.fsub_nc(f_zero, g);
        let exp_neg_g = if pinned {
            vector_exp_spirv(b, neg_g)
        } else {
            b.glsl1(b.f32_t, GLSL_EXP, neg_g)
        };
        let denom = b.fadd_nc(f_one, exp_neg_g);
        let res = if pinned {
            let num = b.fmul_nc(g, u);
            b.fdiv(num, denom)
        } else {
            let silu = b.fdiv(g, denom);
            b.fmul_nc(u, silu)
        };
        b.store_sb_f32(out_buf, idx, res);
    });

    b.finish()
}

#[expect(
    clippy::too_many_arguments,
    reason = "Shares the normalized Sylvester Hadamard butterfly across single-token and batched workgroup grids."
)]
fn emit_hadamard_workgroup_body(
    b: &mut ShaderBuilder,
    in_buf: u32,
    out_buf: u32,
    smem: u32,
    lid: u32,
    blk_idx: u32,
    block_size: u32,
    norm: u32,
) {
    let c1 = b.const_u32(1);
    let c64 = b.const_u32(WORKGROUP_X);
    let blk_base = b.umul(blk_idx, block_size);

    b.emit_for_range(lid, block_size, c64, |b, i| {
        let src_idx = b.uadd(blk_base, i);
        let val = b.load_sb_f32(in_buf, src_idx);
        b.store_wg_f32(smem, i, val);
    });
    b.workgroup_barrier();

    let half_var = b.local_u32_var();
    b.store_var(half_var, c1);
    let pairs = b.ushr(block_size, c1);

    b.emit_while(
        |b| {
            let h = b.load_var_u32(half_var);
            b.u_lt(h, block_size)
        },
        |b| {
            let half = b.load_var_u32(half_var);
            let mask = b.binop(OP_I_SUB, b.u32_t, half, c1);
            b.emit_for_range(lid, pairs, c64, |b, p| {
                let lo = b.uand(p, mask);
                let base_p = b.binop(OP_I_SUB, b.u32_t, p, lo);
                let hi = b.ushl(base_p, c1);
                let left = b.uor(hi, lo);
                let right = b.uor(left, half);
                let l_val = b.load_wg_f32(smem, left);
                let r_val = b.load_wg_f32(smem, right);
                let sum = b.fadd_nc(l_val, r_val);
                let diff = b.fsub_nc(l_val, r_val);
                b.store_wg_f32(smem, left, sum);
                b.store_wg_f32(smem, right, diff);
            });
            b.workgroup_barrier();
            let next_half = b.ushl(half, c1);
            b.store_var(half_var, next_half);
        },
    );

    b.emit_for_range(lid, block_size, c64, |b, i| {
        let v = b.load_wg_f32(smem, i);
        let scaled = b.fmul_nc(v, norm);
        let dst_idx = b.uadd(blk_base, i);
        b.store_sb_f32(out_buf, dst_idx, scaled);
    });
}

/// Emits normalized Sylvester Hadamard transform SPIR-V 1.6 words for power-of-two blocks `2..=1024`.
///
/// Bindings: `0` read-only `input` (`f32`), `1` writable `output` (`f32`).
/// Push constants (`2 × u32`): `[block_size, norm_bits]`. Dispatch: `(total_elems / block_size, 1, 1)`.
#[must_use]
pub fn ffn_hadamard_f32() -> Vec<u32> {
    let (mut b, push_var) = ShaderBuilder::new(false, 2);
    let push_var = push_var.unwrap_or(0);
    let in_buf = b.storage_buffer(0, ElemType::F32, AccessMode::ReadOnly);
    let out_buf = b.storage_buffer(1, ElemType::F32, AccessMode::ReadWrite);
    let smem = b.workgroup_f32_array(1024);

    let lid_vec = b.local_invocation_id();
    let wid_vec = b.workgroup_id();
    let lid = b.load_input_component(lid_vec, 0);
    let blk_idx = b.load_input_component(wid_vec, 0);
    let block_size = b.load_push(push_var, 0);
    let norm_bits = b.load_push(push_var, 1);
    let norm = b.bitcast(b.f32_t, norm_bits);

    emit_hadamard_workgroup_body(
        &mut b, in_buf, out_buf, smem, lid, blk_idx, block_size, norm,
    );

    b.finish()
}

/// Emits batched multi-token Sylvester Hadamard transform SPIR-V 1.6 words (`[tokens, cols]`).
///
/// Bindings: `0` read-only `input` (`f32`), `1` writable `output` (`f32`).
/// Push constants (`4 × u32`): `[block_size, norm_bits, blocks_per_tok, tokens]`.
/// Dispatch: `(blocks_per_tok, tokens, 1)`.
#[must_use]
pub fn ffn_hadamard_batched_f32() -> Vec<u32> {
    let (mut b, push_var) = ShaderBuilder::new(false, 4);
    let push_var = push_var.unwrap_or(0);
    let in_buf = b.storage_buffer(0, ElemType::F32, AccessMode::ReadOnly);
    let out_buf = b.storage_buffer(1, ElemType::F32, AccessMode::ReadWrite);
    let smem = b.workgroup_f32_array(1024);

    let lid_vec = b.local_invocation_id();
    let wid_vec = b.workgroup_id();
    let lid = b.load_input_component(lid_vec, 0);
    let blk_in_tok = b.load_input_component(wid_vec, 0);
    let tok = b.load_input_component(wid_vec, 1);

    let block_size = b.load_push(push_var, 0);
    let norm_bits = b.load_push(push_var, 1);
    let blocks_per_tok = b.load_push(push_var, 2);
    let tokens = b.load_push(push_var, 3);
    let norm = b.bitcast(b.f32_t, norm_bits);

    let blk_ok = b.u_lt(blk_in_tok, blocks_per_tok);
    let tok_ok = b.u_lt(tok, tokens);
    let in_bounds = b.logical_and(blk_ok, tok_ok);
    b.emit_if(in_bounds, |b| {
        let tok_off = b.umul(tok, blocks_per_tok);
        let blk_idx = b.uadd(tok_off, blk_in_tok);
        emit_hadamard_workgroup_body(b, in_buf, out_buf, smem, lid, blk_idx, block_size, norm);
    });

    b.finish()
}

/// Emits rank-1 `dot(a, b)` reduction SPIR-V 1.6 words writing scalar sum into `out[0]`.
///
/// Bindings: `0` read-only `a` (`f32`), `1` read-only `b` (`f32`), `2` writable `out` (`f32`).
/// Push constants (`1 Ã— u32`): `[count]`. Dispatch: `(1, 1, 1)`.
#[must_use]
pub fn dot_f32_scalar() -> Vec<u32> {
    let (mut b, push_var) = ShaderBuilder::new(false, 1);
    let push_var = push_var.unwrap_or(0);
    let a_buf = b.storage_buffer(0, ElemType::F32, AccessMode::ReadOnly);
    let b_buf = b.storage_buffer(1, ElemType::F32, AccessMode::ReadOnly);
    let out_buf = b.storage_buffer(2, ElemType::F32, AccessMode::ReadWrite);
    let smem = b.workgroup_f32_array(WORKGROUP_X);

    let lid_vec = b.local_invocation_id();
    let lid = b.load_input_component(lid_vec, 0);
    let count = b.load_push(push_var, 0);

    let c0 = b.const_u32(0);
    let c64 = b.const_u32(WORKGROUP_X);
    let f_zero = b.const_f32(0.0);
    let acc_var = b.local_f32_var();
    b.store_var(acc_var, f_zero);

    b.emit_for_range(lid, count, c64, |b, i| {
        let av = b.load_sb_f32(a_buf, i);
        let bv = b.load_sb_f32(b_buf, i);
        let prod = b.fmul_nc(av, bv);
        let prev = b.load_var_f32(acc_var);
        let next = b.fadd_nc(prev, prod);
        b.store_var(acc_var, next);
    });

    let partial = b.load_var_f32(acc_var);
    b.store_wg_f32(smem, lid, partial);
    b.workgroup_barrier();

    for stride in [32_u32, 16, 8, 4, 2, 1] {
        let sc = b.const_u32(stride);
        let active = b.u_lt(lid, sc);
        b.emit_if(active, |b| {
            let oth_idx = b.uadd(lid, sc);
            let cur = b.load_wg_f32(smem, lid);
            let oth = b.load_wg_f32(smem, oth_idx);
            let sm = b.fadd_nc(cur, oth);
            b.store_wg_f32(smem, lid, sm);
        });
        b.workgroup_barrier();
    }

    let is_head = b.u_eq(lid, c0);
    b.emit_if(is_head, |b| {
        let res = b.load_wg_f32(smem, c0);
        b.store_sb_f32(out_buf, c0, res);
    });

    b.finish()
}

/// Emits `#40` Stage 1 (`QK^T` scaled causal scores) SPIR-V 1.6 words.
///
/// Bindings: `0` read-only `Q` (`f32`), `1` read-only `K` (`f32`), `2` writable `scores` (`f32`).
/// Push constants (`8 × u32`): `[query_len, total_seq, past_len, num_heads, num_kv_heads, head_dim, scale_bits, causal]`.
/// Dispatch: `(ceil(total_seq / 64), num_heads, query_len)`.
#[must_use]
pub fn attn_qk_scores() -> Vec<u32> {
    emit_attn_qk_scores(false)
}

/// Emits `#40` Stage 1 (`QK^T` scaled causal scores) SPIR-V 1.6 words reading packed-FP16 `K`.
///
/// Bindings: `0` read-only `Q` (`f32`), `1` read-only `K` (`u32`, `2 × f16` via `UnpackHalf2x16`),
/// `2` writable `scores` (`f32`).
/// Push constants (`8 × u32`): `[query_len, total_seq, past_len, num_heads, num_kv_heads, head_dim, scale_bits, causal]`.
/// Dispatch: `(ceil(total_seq / 64), num_heads, query_len)`.
#[must_use]
pub fn attn_qk_scores_f16kv() -> Vec<u32> {
    emit_attn_qk_scores(true)
}

fn emit_attn_qk_scores(f16_kv: bool) -> Vec<u32> {
    let (mut b, push_var) = ShaderBuilder::new(false, 8);
    let push_var = push_var.unwrap_or(0);
    let q_buf = b.storage_buffer(0, ElemType::F32, AccessMode::ReadOnly);
    let k_elem_ty = if f16_kv { ElemType::U32 } else { ElemType::F32 };
    let k_buf = b.storage_buffer(1, k_elem_ty, AccessMode::ReadOnly);
    let s_buf = b.storage_buffer(2, ElemType::F32, AccessMode::ReadWrite);

    let gid = b.global_invocation_id();
    let ki = b.load_input_component(gid, 0);
    let hi = b.load_input_component(gid, 1);
    let qi = b.load_input_component(gid, 2);

    let query_len = b.load_push(push_var, 0);
    let total_seq = b.load_push(push_var, 1);
    let past_len = b.load_push(push_var, 2);
    let num_heads = b.load_push(push_var, 3);
    let num_kv_heads = b.load_push(push_var, 4);
    let head_dim = b.load_push(push_var, 5);
    let scale_bits = b.load_push(push_var, 6);
    let causal = b.load_push(push_var, 7);
    let scale = b.bitcast(b.f32_t, scale_bits);

    let k_ok = b.u_lt(ki, total_seq);
    let h_ok = b.u_lt(hi, num_heads);
    let q_ok = b.u_lt(qi, query_len);
    let k_and_h_ok = b.logical_and(k_ok, h_ok);
    let in_bounds = b.logical_and(k_and_h_ok, q_ok);

    b.emit_if(in_bounds, |b| {
        let valid_len = attn_valid_len(b, qi, past_len, total_seq, causal);
        let is_valid = b.u_lt(ki, valid_len);
        let qh = b.umul(qi, num_heads);
        let s_row = b.uadd(qh, hi);
        let s_row_seq = b.umul(s_row, total_seq);
        let s_idx = b.uadd(s_row_seq, ki);
        let f_zero = b.const_f32(0.0);

        b.emit_if_else(
            is_valid,
            |b| {
                let gqa = b.udiv(num_heads, num_kv_heads);
                let kv_h = b.udiv(hi, gqa);
                let q_base = b.umul(s_row, head_dim);
                let k_seq_off = b.umul(ki, num_kv_heads);
                let k_row = b.uadd(k_seq_off, kv_h);
                let k_base = b.umul(k_row, head_dim);

                let scaled =
                    emit_dot_qk_row(b, q_buf, q_base, k_buf, k_base, head_dim, scale, f16_kv);
                b.store_sb_f32(s_buf, s_idx, scaled);
            },
            |b| {
                b.store_sb_f32(s_buf, s_idx, f_zero);
            },
        );
    });

    b.finish()
}

/// Emits `#40` Stage 2 (global row softmax with IEEE binary16 rounding) SPIR-V 1.6 words.
///
/// Bindings: `0` read-only `scores` (`f32`), `1` writable `probs` (`f32`).
/// Push constants (`8 × u32`): `[query_len, total_seq, past_len, num_heads, num_kv_heads, head_dim, scale_bits, causal]`.
/// Dispatch: `(1, num_heads, query_len)` with workgroup shared `[f32; 64]`.
#[must_use]
pub fn attn_softmax_f16() -> Vec<u32> {
    let (mut b, push_var) = ShaderBuilder::new(false, 8);
    let push_var = push_var.unwrap_or(0);
    let scores_buf = b.storage_buffer(0, ElemType::F32, AccessMode::ReadOnly);
    let probs_buf = b.storage_buffer(1, ElemType::F32, AccessMode::ReadWrite);
    let wg_reduce = b.workgroup_f32_array(WORKGROUP_X);

    let lid_vec = b.local_invocation_id();
    let wid_vec = b.workgroup_id();
    let lid = b.load_input_component(lid_vec, 0);
    let hi = b.load_input_component(wid_vec, 1);
    let qi = b.load_input_component(wid_vec, 2);

    let total_seq = b.load_push(push_var, 1);
    let past_len = b.load_push(push_var, 2);
    let num_heads = b.load_push(push_var, 3);
    let causal = b.load_push(push_var, 7);

    let valid_len = attn_valid_len(&mut b, qi, past_len, total_seq, causal);
    let qh = b.umul(qi, num_heads);
    let row_head = b.uadd(qh, hi);
    let row_base = b.umul(row_head, total_seq);
    let c0 = b.const_u32(0);
    let c64 = b.const_u32(WORKGROUP_X);

    let score0 = b.load_sb_f32(scores_buf, row_base);
    let max_var = b.local_f32_var();
    b.store_var(max_var, score0);
    b.emit_for_range(lid, valid_len, c64, |b, k| {
        let s_idx = b.uadd(row_base, k);
        let s = b.load_sb_f32(scores_buf, s_idx);
        let prev = b.load_var_f32(max_var);
        let next = b.glsl2(b.f32_t, GLSL_FMAX, prev, s);
        b.store_var(max_var, next);
    });
    let my_max = b.load_var_f32(max_var);
    b.store_wg_f32(wg_reduce, lid, my_max);
    b.workgroup_barrier();

    for stride in [32_u32, 16, 8, 4, 2, 1] {
        let sc = b.const_u32(stride);
        let active = b.u_lt(lid, sc);
        b.emit_if(active, |b| {
            let other_idx = b.uadd(lid, sc);
            let other = b.load_wg_f32(wg_reduce, other_idx);
            let cur = b.load_wg_f32(wg_reduce, lid);
            let mx = b.glsl2(b.f32_t, GLSL_FMAX, cur, other);
            b.store_wg_f32(wg_reduce, lid, mx);
        });
        b.workgroup_barrier();
    }
    let row_max = b.load_wg_f32(wg_reduce, c0);
    b.workgroup_barrier();

    let sum_var = b.local_f32_var();
    let f_zero = b.const_f32(0.0);
    b.store_var(sum_var, f_zero);
    b.emit_for_range(lid, valid_len, c64, |b, k| {
        let s_idx = b.uadd(row_base, k);
        let s = b.load_sb_f32(scores_buf, s_idx);
        let shifted = b.fsub_nc(s, row_max);
        let e = b.glsl1(b.f32_t, GLSL_EXP, shifted);
        b.store_sb_f32(probs_buf, s_idx, e);
        let prev = b.load_var_f32(sum_var);
        let next = b.fadd_nc(prev, e);
        b.store_var(sum_var, next);
    });
    let my_sum = b.load_var_f32(sum_var);
    b.store_wg_f32(wg_reduce, lid, my_sum);
    b.workgroup_barrier();

    for stride in [32_u32, 16, 8, 4, 2, 1] {
        let sc = b.const_u32(stride);
        let active = b.u_lt(lid, sc);
        b.emit_if(active, |b| {
            let other_idx = b.uadd(lid, sc);
            let other = b.load_wg_f32(wg_reduce, other_idx);
            let cur = b.load_wg_f32(wg_reduce, lid);
            let sm = b.fadd_nc(cur, other);
            b.store_wg_f32(wg_reduce, lid, sm);
        });
        b.workgroup_barrier();
    }
    let total_sum = b.load_wg_f32(wg_reduce, c0);
    let f_one = b.const_f32(1.0);
    let inv_sum = b.fdiv(f_one, total_sum);

    b.emit_for_range(lid, valid_len, c64, |b, k| {
        let idx = b.uadd(row_base, k);
        let e = b.load_sb_f32(probs_buf, idx);
        let raw_p = b.fmul_nc(e, inv_sum);
        let rounded_p = b.round_f16(raw_p);
        b.store_sb_f32(probs_buf, idx, rounded_p);
    });

    let zero_start = b.uadd(valid_len, lid);
    b.emit_for_range(zero_start, total_seq, c64, |b, k| {
        let z_idx = b.uadd(row_base, k);
        b.store_sb_f32(probs_buf, z_idx, f_zero);
    });

    b.finish()
}

/// Emits `#40` Stage 3 (`probs × V` accumulation) SPIR-V 1.6 words.
///
/// Bindings: `0` read-only `probs` (`f32`), `1` read-only `V` (`f32`), `2` writable `out` (`f32`).
/// Push constants (`8 × u32`): `[query_len, total_seq, past_len, num_heads, num_kv_heads, head_dim, scale_bits, causal]`.
/// Dispatch: `(ceil(head_dim / 64), num_heads, query_len)`.
#[must_use]
pub fn attn_pv_accum() -> Vec<u32> {
    emit_attn_pv_accum(false)
}

/// Emits `#40` Stage 3 (`probs × V` accumulation) SPIR-V 1.6 words reading packed-FP16 `V`.
///
/// Bindings: `0` read-only `probs` (`f32`), `1` read-only `V` (`u32`, `2 × f16` via `UnpackHalf2x16`),
/// `2` writable `out` (`f32`).
/// Push constants (`8 × u32`): `[query_len, total_seq, past_len, num_heads, num_kv_heads, head_dim, scale_bits, causal]`.
/// Dispatch: `(ceil(head_dim / 64), num_heads, query_len)`.
#[must_use]
pub fn attn_pv_accum_f16kv() -> Vec<u32> {
    emit_attn_pv_accum(true)
}

fn emit_attn_pv_accum(f16_kv: bool) -> Vec<u32> {
    let (mut b, push_var) = ShaderBuilder::new(false, 8);
    let push_var = push_var.unwrap_or(0);
    let p_buf = b.storage_buffer(0, ElemType::F32, AccessMode::ReadOnly);
    let v_elem_ty = if f16_kv { ElemType::U32 } else { ElemType::F32 };
    let v_buf = b.storage_buffer(1, v_elem_ty, AccessMode::ReadOnly);
    let o_buf = b.storage_buffer(2, ElemType::F32, AccessMode::ReadWrite);

    let gid = b.global_invocation_id();
    let d = b.load_input_component(gid, 0);
    let hi = b.load_input_component(gid, 1);
    let qi = b.load_input_component(gid, 2);

    let query_len = b.load_push(push_var, 0);
    let total_seq = b.load_push(push_var, 1);
    let past_len = b.load_push(push_var, 2);
    let num_heads = b.load_push(push_var, 3);
    let num_kv_heads = b.load_push(push_var, 4);
    let head_dim = b.load_push(push_var, 5);
    let causal = b.load_push(push_var, 7);

    let d_ok = b.u_lt(d, head_dim);
    let h_ok = b.u_lt(hi, num_heads);
    let q_ok = b.u_lt(qi, query_len);
    let d_and_h_ok = b.logical_and(d_ok, h_ok);
    let in_bounds = b.logical_and(d_and_h_ok, q_ok);

    b.emit_if(in_bounds, |b| {
        let valid_len = attn_valid_len(b, qi, past_len, total_seq, causal);
        let gqa = b.udiv(num_heads, num_kv_heads);
        let kv_h = b.udiv(hi, gqa);
        let qh = b.umul(qi, num_heads);
        let row_head = b.uadd(qh, hi);
        let p_base = b.umul(row_head, total_seq);
        let o_row = b.umul(row_head, head_dim);
        let o_idx = b.uadd(o_row, d);
        let v_kv_stride = b.umul(num_kv_heads, head_dim);
        let v_head_off = b.umul(kv_h, head_dim);
        let v_head_d = b.uadd(v_head_off, d);

        let acc_var = b.local_f32_var();
        let f_zero = b.const_f32(0.0);
        b.store_var(acc_var, f_zero);

        let c0 = b.const_u32(0);
        let c1 = b.const_u32(1);
        b.emit_for_range(c0, valid_len, c1, |b, k| {
            let p_idx = b.uadd(p_base, k);
            let p = b.load_sb_f32(p_buf, p_idx);
            let k_stride = b.umul(k, v_kv_stride);
            let v_idx = b.uadd(k_stride, v_head_d);
            let vv = emit_load_kv_f32(b, v_buf, v_idx, f16_kv);
            let prod = b.fmul_nc(p, vv);
            let prev = b.load_var_f32(acc_var);
            let next = b.fadd_nc(prev, prod);
            b.store_var(acc_var, next);
        });

        let out_v = b.load_var_f32(acc_var);
        b.store_sb_f32(o_buf, o_idx, out_v);
    });

    b.finish()
}

fn emit_wg_reduce_64(b: &mut ShaderBuilder, wg_reduce: u32, lid: u32, is_max: bool) -> u32 {
    for stride in [32_u32, 16, 8, 4, 2, 1] {
        let sc = b.const_u32(stride);
        let active = b.u_lt(lid, sc);
        b.emit_if(active, |b| {
            let other_idx = b.uadd(lid, sc);
            let other = b.load_wg_f32(wg_reduce, other_idx);
            let cur = b.load_wg_f32(wg_reduce, lid);
            let reduced = if is_max {
                b.glsl2(b.f32_t, GLSL_FMAX, cur, other)
            } else {
                b.fadd_nc(cur, other)
            };
            b.store_wg_f32(wg_reduce, lid, reduced);
        });
        b.workgroup_barrier();
    }
    let c0 = b.const_u32(0);
    b.load_wg_f32(wg_reduce, c0)
}

/// Emits `#40` single-dispatch fused attention with shared-memory binary16 probability rounding.
///
/// Requires `total_seq <= 512` (`2304` bytes of workgroup shared storage).
/// Bindings: `0` read-only `Q` (`f32`), `1` read-only `K` (`f32`), `2` read-only `V` (`f32`),
/// `3` writable `out` (`f32`).
/// Dispatch: `(1, num_heads, query_len)`.
#[must_use]
pub fn attn_fused_f16prob() -> Vec<u32> {
    emit_attn_fused_f16prob(false)
}

/// Emits `#40` single-dispatch fused attention reading packed-FP16 `K` and `V` directly via `UnpackHalf2x16`.
///
/// Requires `total_seq <= 512` (`2304` bytes of workgroup shared storage).
/// Bindings: `0` read-only `Q` (`f32`), `1` read-only `K` (`u32`), `2` read-only `V` (`u32`),
/// `3` writable `out` (`f32`).
/// Dispatch: `(1, num_heads, query_len)`.
#[must_use]
pub fn attn_fused_f16prob_f16kv() -> Vec<u32> {
    emit_attn_fused_f16prob(true)
}

fn emit_attn_fused_f16prob(f16_kv: bool) -> Vec<u32> {
    let (mut b, push_var) = ShaderBuilder::new(false, 8);
    let push_var = push_var.unwrap_or(0);
    let q_buf = b.storage_buffer(0, ElemType::F32, AccessMode::ReadOnly);
    let kv_elem_ty = if f16_kv { ElemType::U32 } else { ElemType::F32 };
    let k_buf = b.storage_buffer(1, kv_elem_ty, AccessMode::ReadOnly);
    let v_buf = b.storage_buffer(2, kv_elem_ty, AccessMode::ReadOnly);
    let o_buf = b.storage_buffer(3, ElemType::F32, AccessMode::ReadWrite);

    let wg_scores = b.workgroup_f32_array(512);
    let wg_reduce = b.workgroup_f32_array(WORKGROUP_X);

    let lid_vec = b.local_invocation_id();
    let wid_vec = b.workgroup_id();
    let lid = b.load_input_component(lid_vec, 0);
    let hi = b.load_input_component(wid_vec, 1);
    let qi = b.load_input_component(wid_vec, 2);

    let total_seq = b.load_push(push_var, 1);
    let past_len = b.load_push(push_var, 2);
    let num_heads = b.load_push(push_var, 3);
    let num_kv_heads = b.load_push(push_var, 4);
    let head_dim = b.load_push(push_var, 5);
    let scale_bits = b.load_push(push_var, 6);
    let causal = b.load_push(push_var, 7);
    let scale = b.bitcast(b.f32_t, scale_bits);

    let valid_len = attn_valid_len(&mut b, qi, past_len, total_seq, causal);
    let gqa = b.udiv(num_heads, num_kv_heads);
    let kv_h = b.udiv(hi, gqa);
    let qh = b.umul(qi, num_heads);
    let row_head = b.uadd(qh, hi);
    let q_base = b.umul(row_head, head_dim);
    let o_base = q_base;
    let kv_stride = b.umul(num_kv_heads, head_dim);
    let kv_head_off = b.umul(kv_h, head_dim);

    let c0 = b.const_u32(0);
    let c1 = b.const_u32(1);
    let c64 = b.const_u32(WORKGROUP_X);
    let f_zero = b.const_f32(0.0);

    b.emit_for_range(lid, valid_len, c64, |b, k| {
        let k_off = b.umul(k, kv_stride);
        let k_base = b.uadd(k_off, kv_head_off);
        let s = emit_dot_qk_row(b, q_buf, q_base, k_buf, k_base, head_dim, scale, f16_kv);
        b.store_wg_f32(wg_scores, k, s);
    });
    b.workgroup_barrier();

    let s0 = b.load_wg_f32(wg_scores, c0);
    let max_var = b.local_f32_var();
    b.store_var(max_var, s0);
    b.emit_for_range(lid, valid_len, c64, |b, k| {
        let s = b.load_wg_f32(wg_scores, k);
        let prev = b.load_var_f32(max_var);
        let next = b.glsl2(b.f32_t, GLSL_FMAX, prev, s);
        b.store_var(max_var, next);
    });
    let my_max = b.load_var_f32(max_var);
    b.store_wg_f32(wg_reduce, lid, my_max);
    b.workgroup_barrier();
    let row_max = emit_wg_reduce_64(&mut b, wg_reduce, lid, true);
    b.workgroup_barrier();

    let sum_var = b.local_f32_var();
    b.store_var(sum_var, f_zero);
    b.emit_for_range(lid, valid_len, c64, |b, k| {
        let s = b.load_wg_f32(wg_scores, k);
        let shifted = b.fsub_nc(s, row_max);
        let e = b.glsl1(b.f32_t, GLSL_EXP, shifted);
        b.store_wg_f32(wg_scores, k, e);
        let prev = b.load_var_f32(sum_var);
        let next = b.fadd_nc(prev, e);
        b.store_var(sum_var, next);
    });
    let my_sum = b.load_var_f32(sum_var);
    b.store_wg_f32(wg_reduce, lid, my_sum);
    b.workgroup_barrier();
    let f_one = b.const_f32(1.0);
    let denom = emit_wg_reduce_64(&mut b, wg_reduce, lid, false);
    let inv_sum = b.fdiv(f_one, denom);
    b.workgroup_barrier();

    b.emit_for_range(lid, valid_len, c64, |b, k| {
        let e = b.load_wg_f32(wg_scores, k);
        let raw_p = b.fmul_nc(e, inv_sum);
        let rounded_p = b.round_f16(raw_p);
        b.store_wg_f32(wg_scores, k, rounded_p);
    });
    b.workgroup_barrier();

    b.emit_for_range(lid, head_dim, c64, |b, d| {
        let acc_var = b.local_f32_var();
        b.store_var(acc_var, f_zero);
        let v_head_d = b.uadd(kv_head_off, d);
        b.emit_for_range(c0, valid_len, c1, |b, k| {
            let p = b.load_wg_f32(wg_scores, k);
            let k_off = b.umul(k, kv_stride);
            let v_idx = b.uadd(k_off, v_head_d);
            let vv = emit_load_kv_f32(b, v_buf, v_idx, f16_kv);
            let prod = b.fmul_nc(p, vv);
            let prev = b.load_var_f32(acc_var);
            let next = b.fadd_nc(prev, prod);
            b.store_var(acc_var, next);
        });
        let out_v = b.load_var_f32(acc_var);
        let out_idx = b.uadd(o_base, d);
        b.store_sb_f32(o_buf, out_idx, out_v);
    });

    b.finish()
}

fn emit_load_kv_f32(b: &mut ShaderBuilder, buf: u32, elem_idx: u32, f16_kv: bool) -> u32 {
    if !f16_kv {
        return b.load_sb_f32(buf, elem_idx);
    }
    let c0 = b.const_u32(0);
    let c1 = b.const_u32(1);
    let word_idx = b.ushr(elem_idx, c1);
    let lane = b.uand(elem_idx, c1);
    let is_hi = b.u_ne(lane, c0);
    let packed = b.load_sb_u32(buf, word_idx);
    let (lo, hi) = b.unpack_half_pair(packed);
    b.select(b.f32_t, is_hi, hi, lo)
}

#[expect(
    clippy::too_many_arguments,
    reason = "Shares F32 and packed-FP16 QK^T dot product emission cleanly."
)]
fn emit_dot_qk_row(
    b: &mut ShaderBuilder,
    q_buf: u32,
    q_base: u32,
    k_buf: u32,
    k_base: u32,
    head_dim: u32,
    scale: u32,
    f16_kv: bool,
) -> u32 {
    let c0 = b.const_u32(0);
    let c1 = b.const_u32(1);
    let f_zero = b.const_f32(0.0);
    let acc_var = b.local_f32_var();
    b.store_var(acc_var, f_zero);
    if f16_kv {
        let c2 = b.const_u32(2);
        let c_even_mask = b.const_u32(!1_u32);
        let k_odd_bit = b.uand(k_base, c1);
        let k_even_aligned = b.u_eq(k_odd_bit, c0);
        let even_end = b.uand(head_dim, c_even_mask);
        b.emit_if_else(
            k_even_aligned,
            |b| {
                b.emit_for_range(c0, even_end, c2, |b, d| {
                    let q_idx0 = b.uadd(q_base, d);
                    let q_idx1 = b.uadd(q_idx0, c1);
                    let k_elem0 = b.uadd(k_base, d);
                    let k_word = b.ushr(k_elem0, c1);
                    let packed = b.load_sb_u32(k_buf, k_word);
                    let (k0, k1) = b.unpack_half_pair(packed);
                    let q0 = b.load_sb_f32(q_buf, q_idx0);
                    let q1 = b.load_sb_f32(q_buf, q_idx1);
                    let prod0 = b.fmul_nc(q0, k0);
                    let acc0 = b.load_var_f32(acc_var);
                    let acc1 = b.fadd_nc(acc0, prod0);
                    let prod1 = b.fmul_nc(q1, k1);
                    let acc2 = b.fadd_nc(acc1, prod1);
                    b.store_var(acc_var, acc2);
                });
                let has_tail = b.u_lt(even_end, head_dim);
                b.emit_if(has_tail, |b| {
                    let q_idx = b.uadd(q_base, even_end);
                    let k_elem = b.uadd(k_base, even_end);
                    let qv = b.load_sb_f32(q_buf, q_idx);
                    let kv = emit_load_kv_f32(b, k_buf, k_elem, true);
                    let prod = b.fmul_nc(qv, kv);
                    let prev = b.load_var_f32(acc_var);
                    let next = b.fadd_nc(prev, prod);
                    b.store_var(acc_var, next);
                });
            },
            |b| {
                b.emit_for_range(c0, head_dim, c1, |b, d| {
                    let q_idx = b.uadd(q_base, d);
                    let k_elem = b.uadd(k_base, d);
                    let qv = b.load_sb_f32(q_buf, q_idx);
                    let kv = emit_load_kv_f32(b, k_buf, k_elem, true);
                    let prod = b.fmul_nc(qv, kv);
                    let prev = b.load_var_f32(acc_var);
                    let next = b.fadd_nc(prev, prod);
                    b.store_var(acc_var, next);
                });
            },
        );
    } else {
        b.emit_for_range(c0, head_dim, c1, |b, d| {
            let q_idx = b.uadd(q_base, d);
            let k_idx = b.uadd(k_base, d);
            let qv = b.load_sb_f32(q_buf, q_idx);
            let kv = b.load_sb_f32(k_buf, k_idx);
            let prod = b.fmul_nc(qv, kv);
            let prev = b.load_var_f32(acc_var);
            let next = b.fadd_nc(prev, prod);
            b.store_var(acc_var, next);
        });
    }
    let acc_v = b.load_var_f32(acc_var);
    b.fmul_nc(scale, acc_v)
}

#[expect(
    clippy::too_many_arguments,
    reason = "Splits shared RMSNorm emission cleanly between standalone and fused residual kernels."
)]
fn emit_rms_norm_workgroup_body(
    b: &mut ShaderBuilder,
    in_buf: u32,
    res_update: Option<(u32, u32)>,
    w_buf: u32,
    out_buf: u32,
    wg_reduce: u32,
    dim: u32,
    eps_bits: u32,
) {
    let lid_vec = b.local_invocation_id();
    let wid_vec = b.workgroup_id();
    let lid = b.load_input_component(lid_vec, 0);
    let row = b.load_input_component(wid_vec, 1);

    let c0 = b.const_u32(0);
    let c64 = b.const_u32(WORKGROUP_X);
    let f_zero = b.const_f32(0.0);
    let f_one = b.const_f32(1.0);
    let eps = b.bitcast(b.f32_t, eps_bits);

    let dim_ok = b.u_ne(dim, c0);
    b.emit_if(dim_ok, |b| {
        let base = b.umul(row, dim);
        let acc_var = b.local_f32_var();
        b.store_var(acc_var, f_zero);

        b.emit_for_range(lid, dim, c64, |b, i| {
            let idx = b.uadd(base, i);
            let val = if let Some((res_buf, delta_buf)) = res_update {
                let r = b.load_sb_f32(res_buf, idx);
                let d = b.load_sb_f32(delta_buf, idx);
                let updated = b.fadd_nc(r, d);
                b.store_sb_f32(res_buf, idx, updated);
                updated
            } else {
                b.load_sb_f32(in_buf, idx)
            };
            let sq = b.fmul_nc(val, val);
            let prev = b.load_var_f32(acc_var);
            let next = b.fadd_nc(prev, sq);
            b.store_var(acc_var, next);
        });

        let partial = b.load_var_f32(acc_var);
        b.store_wg_f32(wg_reduce, lid, partial);
        b.workgroup_barrier();
        let sum_sq = emit_wg_reduce_64(b, wg_reduce, lid, false);

        let dim_f = b.u32_to_f32(dim);
        let mean = b.fdiv(sum_sq, dim_f);
        let mean_eps = b.fadd_nc(mean, eps);
        let denom = b.f_sqrt(mean_eps);
        let scale = b.fdiv(f_one, denom);

        let read_buf = res_update.map_or(in_buf, |(res_buf, _)| res_buf);
        b.emit_for_range(lid, dim, c64, |b, i| {
            let idx = b.uadd(base, i);
            let v = b.load_sb_f32(read_buf, idx);
            let w = b.load_sb_f32(w_buf, i);
            let scaled = b.fmul_nc(v, scale);
            let normed = b.fmul_nc(scaled, w);
            b.store_sb_f32(out_buf, idx, normed);
        });
    });
}

/// Emits GPU `RMSNorm` SPIR-V 1.6 words (`out[r, i] = input[r, i] * rsqrt(mean(x^2) + eps) * weight[i]`).
///
/// Bindings: `0` read-only `input` (`f32`), `1` read-only `weight` (`f32`), `2` writable `out` (`f32`).
/// Push constants (`2 × u32`): `[dim, eps_bits]`. Dispatch: `(1, rows, 1)` (`rows = 1` or `num_heads`).
#[must_use]
pub fn rms_norm_f32() -> Vec<u32> {
    let (mut b, push_var) = ShaderBuilder::new(false, 2);
    let push_var = push_var.unwrap_or(0);
    let in_buf = b.storage_buffer(0, ElemType::F32, AccessMode::ReadOnly);
    let w_buf = b.storage_buffer(1, ElemType::F32, AccessMode::ReadOnly);
    let out_buf = b.storage_buffer(2, ElemType::F32, AccessMode::ReadWrite);
    let wg_reduce = b.workgroup_f32_array(WORKGROUP_X);

    let dim = b.load_push(push_var, 0);
    let eps_bits = b.load_push(push_var, 1);
    emit_rms_norm_workgroup_body(
        &mut b, in_buf, None, w_buf, out_buf, wg_reduce, dim, eps_bits,
    );

    b.finish()
}

/// Emits batched multi-token `RMSNorm` SPIR-V 1.6 words (`[T, dim]`).
///
/// Bindings: `0` read-only `input` (`f32`), `1` read-only `weight` (`f32`), `2` writable `out` (`f32`).
/// Push constants (`2 × u32`): `[dim, eps_bits]`. Dispatch: `(1, rows, 1)`.
#[must_use]
pub fn rms_norm_batched_f32() -> Vec<u32> {
    rms_norm_f32()
}

/// Emits fused in-place residual update + `RMSNorm` SPIR-V 1.6 words (`residual += delta; norm_out = rms_norm(residual)`).
///
/// Bindings: `0` read-write `residual_inout` (`f32`), `1` read-only `delta` (`f32`),
/// `2` read-only `weight` (`f32`), `3` writable `norm_out` (`f32`).
/// Push constants (`2 × u32`): `[dim, eps_bits]`. Dispatch: `(1, rows, 1)`.
#[must_use]
pub fn add_rms_norm_f32() -> Vec<u32> {
    let (mut b, push_var) = ShaderBuilder::new(false, 2);
    let push_var = push_var.unwrap_or(0);
    let res_buf = b.storage_buffer(0, ElemType::F32, AccessMode::ReadWrite);
    let delta_buf = b.storage_buffer(1, ElemType::F32, AccessMode::ReadOnly);
    let w_buf = b.storage_buffer(2, ElemType::F32, AccessMode::ReadOnly);
    let out_buf = b.storage_buffer(3, ElemType::F32, AccessMode::ReadWrite);
    let wg_reduce = b.workgroup_f32_array(WORKGROUP_X);

    let dim = b.load_push(push_var, 0);
    let eps_bits = b.load_push(push_var, 1);
    emit_rms_norm_workgroup_body(
        &mut b,
        res_buf,
        Some((res_buf, delta_buf)),
        w_buf,
        out_buf,
        wg_reduce,
        dim,
        eps_bits,
    );

    b.finish()
}

/// Emits batched multi-token fused residual update + `RMSNorm` SPIR-V 1.6 words (`[T, dim]`).
///
/// Bindings: `0` read-write `residual_inout` (`f32`), `1` read-only `delta` (`f32`),
/// `2` read-only `weight` (`f32`), `3` writable `norm_out` (`f32`).
/// Push constants (`2 × u32`): `[dim, eps_bits]`. Dispatch: `(1, rows, 1)`.
#[must_use]
pub fn add_rms_norm_batched_f32() -> Vec<u32> {
    add_rms_norm_f32()
}

/// Emits elementwise in-place residual addition SPIR-V 1.6 words (`residual_inout[i] += delta[i]`).
///
/// Bindings: `0` read-write `residual_inout` (`f32`), `1` read-only `delta` (`f32`).
/// Push constants (`1 × u32`): `[len]`. Dispatch: `(len.div_ceil(64), 1, 1)`.
#[must_use]
pub fn residual_add_f32() -> Vec<u32> {
    let (mut b, push_var) = ShaderBuilder::new(false, 1);
    let push_var = push_var.unwrap_or(0);
    let res_buf = b.storage_buffer(0, ElemType::F32, AccessMode::ReadWrite);
    let delta_buf = b.storage_buffer(1, ElemType::F32, AccessMode::ReadOnly);

    let gid = b.global_invocation_id();
    let idx = b.load_input_component(gid, 0);
    let len = b.load_push(push_var, 0);

    let in_bounds = b.u_lt(idx, len);
    b.emit_if(in_bounds, |b| {
        let r = b.load_sb_f32(res_buf, idx);
        let d = b.load_sb_f32(delta_buf, idx);
        let sum = b.fadd_nc(r, d);
        b.store_sb_f32(res_buf, idx, sum);
    });

    b.finish()
}

/// Emits batched 2D in-place residual addition SPIR-V 1.6 words (`[tokens, dim]`).
///
/// Bindings: `0` read-write `residual_inout` (`f32`), `1` read-only `delta` (`f32`).
/// Push constants (`2 × u32`): `[dim, tokens]`. Dispatch: `(dim.div_ceil(64), tokens, 1)`.
#[must_use]
pub fn residual_add_batched_f32() -> Vec<u32> {
    let (mut b, push_var) = ShaderBuilder::new(false, 2);
    let push_var = push_var.unwrap_or(0);
    let res_buf = b.storage_buffer(0, ElemType::F32, AccessMode::ReadWrite);
    let delta_buf = b.storage_buffer(1, ElemType::F32, AccessMode::ReadOnly);

    let gid = b.global_invocation_id();
    let col = b.load_input_component(gid, 0);
    let tok = b.load_input_component(gid, 1);
    let dim = b.load_push(push_var, 0);
    let tokens = b.load_push(push_var, 1);

    let col_ok = b.u_lt(col, dim);
    let tok_ok = b.u_lt(tok, tokens);
    let in_bounds = b.logical_and(col_ok, tok_ok);
    b.emit_if(in_bounds, |b| {
        let base = b.umul(tok, dim);
        let idx = b.uadd(base, col);
        let r = b.load_sb_f32(res_buf, idx);
        let d = b.load_sb_f32(delta_buf, idx);
        let sum = b.fadd_nc(r, d);
        b.store_sb_f32(res_buf, idx, sum);
    });

    b.finish()
}

#[expect(
    clippy::too_many_arguments,
    reason = "Shares NeoX rotary embedding arithmetic cleanly across single-token and batched RoPE kernels."
)]
fn emit_rope_element(
    b: &mut ShaderBuilder,
    vec_buf: u32,
    freqs_buf: u32,
    head_base: u32,
    freq_base: u32,
    d: u32,
    rope_dim: u32,
    round_flag: u32,
) {
    let c0 = b.const_u32(0);
    let c1 = b.const_u32(1);
    let c2 = b.const_u32(2);
    let half_rope = b.ushr(rope_dim, c1);
    let do_round = b.u_ne(round_flag, c0);
    let is_pair = b.u_lt(d, half_rope);

    b.emit_if_else(
        is_pair,
        |b| {
            let idx0 = b.uadd(head_base, d);
            let idx1 = b.uadd(idx0, half_rope);
            let first = b.load_sb_f32(vec_buf, idx0);
            let second = b.load_sb_f32(vec_buf, idx1);
            let d2 = b.umul(d, c2);
            let f_off = b.uadd(freq_base, d2);
            let f_sin_off = b.uadd(f_off, c1);
            let cos_v = b.load_sb_f32(freqs_buf, f_off);
            let sin_v = b.load_sb_f32(freqs_buf, f_sin_off);

            let fc = b.fmul_nc(first, cos_v);
            let ss = b.fmul_nc(second, sin_v);
            let r0_raw = b.fsub_nc(fc, ss);
            let fs = b.fmul_nc(first, sin_v);
            let sc = b.fmul_nc(second, cos_v);
            let r1_raw = b.fadd_nc(fs, sc);

            b.emit_if_else(
                do_round,
                |b| {
                    let r0 = b.round_f16(r0_raw);
                    let r1 = b.round_f16(r1_raw);
                    b.store_sb_f32(vec_buf, idx0, r0);
                    b.store_sb_f32(vec_buf, idx1, r1);
                },
                |b| {
                    b.store_sb_f32(vec_buf, idx0, r0_raw);
                    b.store_sb_f32(vec_buf, idx1, r1_raw);
                },
            );
        },
        |b| {
            let past_rope = b.u_ge(d, rope_dim);
            let tail_round = b.logical_and(past_rope, do_round);
            b.emit_if(tail_round, |b| {
                let idx = b.uadd(head_base, d);
                let v = b.load_sb_f32(vec_buf, idx);
                let vr = b.round_f16(v);
                b.store_sb_f32(vec_buf, idx, vr);
            });
        },
    );
}

/// Emits per-head NeoX-style rotary embedding (`RoPE`) SPIR-V 1.6 words with optional binary16 rounding.
///
/// Preserves exact Prism `(cos, sin)` pairing across `0..rope_dim/2` paired with `rope_dim/2..rope_dim`,
/// and rounds all `0..head_dim` elements per head to IEEE binary16 when `round_f16 != 0`.
///
/// Bindings: `0` read-write `vec_inout` (`f32`, `[heads, head_dim]`),
/// `1` read-only `freqs` (`f32`, interleaved `[cos0, sin0, ...]`, length `rope_dim`).
/// Push constants (`4 × u32`): `[heads, head_dim, rope_dim, round_f16]`.
/// Dispatch: `(head_dim.div_ceil(64), heads, 1)`.
#[must_use]
pub fn rope_f32() -> Vec<u32> {
    let (mut b, push_var) = ShaderBuilder::new(false, 4);
    let push_var = push_var.unwrap_or(0);
    let vec_buf = b.storage_buffer(0, ElemType::F32, AccessMode::ReadWrite);
    let freqs_buf = b.storage_buffer(1, ElemType::F32, AccessMode::ReadOnly);

    let gid = b.global_invocation_id();
    let d = b.load_input_component(gid, 0);
    let h = b.load_input_component(gid, 1);

    let heads = b.load_push(push_var, 0);
    let head_dim = b.load_push(push_var, 1);
    let rope_dim = b.load_push(push_var, 2);
    let round_flag = b.load_push(push_var, 3);

    let d_ok = b.u_lt(d, head_dim);
    let h_ok = b.u_lt(h, heads);
    let in_bounds = b.logical_and(d_ok, h_ok);

    b.emit_if(in_bounds, |b| {
        let c0 = b.const_u32(0);
        let head_base = b.umul(h, head_dim);
        emit_rope_element(
            b, vec_buf, freqs_buf, head_base, c0, d, rope_dim, round_flag,
        );
    });

    b.finish()
}

/// Emits batched multi-token NeoX-style rotary embedding (`RoPE`) SPIR-V 1.6 words.
///
/// Bindings: `0` read-write `vec_inout` (`f32`, `[tokens, heads, head_dim]`),
/// `1` read-only `freqs` (`f32`, `[tokens, rope_dim]`).
/// Push constants (`5 × u32`): `[heads, head_dim, rope_dim, round_f16, tokens]`.
/// Dispatch: `(head_dim.div_ceil(64), heads, tokens)`.
#[must_use]
pub fn rope_batched_f32() -> Vec<u32> {
    let (mut b, push_var) = ShaderBuilder::new(false, 5);
    let push_var = push_var.unwrap_or(0);
    let vec_buf = b.storage_buffer(0, ElemType::F32, AccessMode::ReadWrite);
    let freqs_buf = b.storage_buffer(1, ElemType::F32, AccessMode::ReadOnly);

    let gid = b.global_invocation_id();
    let d = b.load_input_component(gid, 0);
    let h = b.load_input_component(gid, 1);
    let tok = b.load_input_component(gid, 2);

    let heads = b.load_push(push_var, 0);
    let head_dim = b.load_push(push_var, 1);
    let rope_dim = b.load_push(push_var, 2);
    let round_flag = b.load_push(push_var, 3);
    let tokens = b.load_push(push_var, 4);

    let d_ok = b.u_lt(d, head_dim);
    let h_ok = b.u_lt(h, heads);
    let tok_ok = b.u_lt(tok, tokens);
    let head_elem_ok = b.logical_and(d_ok, h_ok);
    let in_bounds = b.logical_and(head_elem_ok, tok_ok);

    b.emit_if(in_bounds, |b| {
        let th = b.umul(tok, heads);
        let row_head = b.uadd(th, h);
        let head_base = b.umul(row_head, head_dim);
        let freq_base = b.umul(tok, rope_dim);
        emit_rope_element(
            b, vec_buf, freqs_buf, head_base, freq_base, d, rope_dim, round_flag,
        );
    });

    b.finish()
}

/// Emits elementwise attention output gating SPIR-V 1.6 words (`values_inout[i] *= sigmoid(gate[i])`).
///
/// Bindings: `0` read-write `values_inout` (`f32`), `1` read-only `gate` (`f32`).
/// Push constants (`1 × u32`): `[len]`. Dispatch: `(len.div_ceil(64), 1, 1)`.
#[must_use]
pub fn mul_sigmoid_f32() -> Vec<u32> {
    let (mut b, push_var) = ShaderBuilder::new(false, 1);
    let push_var = push_var.unwrap_or(0);
    let vals_buf = b.storage_buffer(0, ElemType::F32, AccessMode::ReadWrite);
    let gate_buf = b.storage_buffer(1, ElemType::F32, AccessMode::ReadOnly);

    let gid = b.global_invocation_id();
    let idx = b.load_input_component(gid, 0);
    let len = b.load_push(push_var, 0);

    let in_bounds = b.u_lt(idx, len);
    b.emit_if(in_bounds, |b| {
        let f_zero = b.const_f32(0.0);
        let f_one = b.const_f32(1.0);
        let v = b.load_sb_f32(vals_buf, idx);
        let g = b.load_sb_f32(gate_buf, idx);
        let neg_g = b.fsub_nc(f_zero, g);
        let e = b.f_exp(neg_g);
        let denom = b.fadd_nc(f_one, e);
        let sig = b.fdiv(f_one, denom);
        let out = b.fmul_nc(v, sig);
        b.store_sb_f32(vals_buf, idx, out);
    });

    b.finish()
}

#[expect(
    clippy::too_many_arguments,
    reason = "Shares causal conv1d + rolling history shift + SiLU across offset=0 and token-indexed variants."
)]
fn emit_conv1d_silu_body(
    b: &mut ShaderBuilder,
    qkv_buf: u32,
    state_buf: u32,
    weight_buf: u32,
    out_buf: u32,
    ch: u32,
    qkv_offset: u32,
    channels: u32,
    kernel_size: u32,
) {
    let ch_ok = b.u_lt(ch, channels);
    let c0 = b.const_u32(0);
    let k_ok = b.u_ne(kernel_size, c0);
    let in_bounds = b.logical_and(ch_ok, k_ok);

    b.emit_if(in_bounds, |b| {
        let c1 = b.const_u32(1);
        let f_zero = b.const_f32(0.0);
        let f_one = b.const_f32(1.0);

        let hist_width = b.usub(kernel_size, c1);
        let w_base = b.umul(ch, kernel_size);
        let acc_var = b.local_f32_var();
        b.store_var(acc_var, f_zero);

        let has_hist = b.u_lt(c1, kernel_size);
        b.emit_if(has_hist, |b| {
            let h_base = b.umul(ch, hist_width);
            b.emit_for_range(c0, hist_width, c1, |b, t| {
                let h_idx = b.uadd(h_base, t);
                let w_idx = b.uadd(w_base, t);
                let hv = b.load_sb_f32(state_buf, h_idx);
                let wv = b.load_sb_f32(weight_buf, w_idx);
                let prod = b.fmul_nc(hv, wv);
                let prev = b.load_var_f32(acc_var);
                let next = b.fadd_nc(prev, prod);
                b.store_var(acc_var, next);
            });
        });

        let qkv_idx = b.uadd(qkv_offset, ch);
        let cur = b.load_sb_f32(qkv_buf, qkv_idx);
        let w_cur_idx = b.uadd(w_base, hist_width);
        let w_cur = b.load_sb_f32(weight_buf, w_cur_idx);
        let cur_prod = b.fmul_nc(cur, w_cur);
        let prev_sum = b.load_var_f32(acc_var);
        let conv_val = b.fadd_nc(prev_sum, cur_prod);

        b.emit_if(has_hist, |b| {
            let h_base = b.umul(ch, hist_width);
            let shift_limit = b.usub(hist_width, c1);
            b.emit_for_range(c0, shift_limit, c1, |b, t| {
                let dst_idx = b.uadd(h_base, t);
                let src_idx = b.uadd(dst_idx, c1);
                let next_h = b.load_sb_f32(state_buf, src_idx);
                b.store_sb_f32(state_buf, dst_idx, next_h);
            });
            let last_idx = b.uadd(h_base, shift_limit);
            b.store_sb_f32(state_buf, last_idx, cur);
        });

        let neg_val = b.fsub_nc(f_zero, conv_val);
        let e = b.f_exp(neg_val);
        let denom = b.fadd_nc(f_one, e);
        let activated = b.fdiv(conv_val, denom);
        b.store_sb_f32(out_buf, ch, activated);
    });
}

/// Emits depthwise causal `conv1d` + rolling history shift + `SiLU` SPIR-V 1.6 words.
///
/// Matches Prism's channel-major `conv_state` layout (`channels * (kernel_size - 1)` floats, oldest
/// tap first) and channel-major `conv_weight` (`channels * kernel_size` floats).
///
/// Bindings: `0` read-only `qkv_in` (`f32`), `1` read-write `conv_state_inout` (`f32`),
/// `2` read-only `conv_weight` (`f32`), `3` writable `conv_out` (`f32`).
/// Push constants (`2 × u32`): `[channels, kernel_size]`. Dispatch: `(channels.div_ceil(64), 1, 1)`.
#[must_use]
pub fn conv1d_silu_f32() -> Vec<u32> {
    let (mut b, push_var) = ShaderBuilder::new(false, 2);
    let push_var = push_var.unwrap_or(0);
    let qkv_buf = b.storage_buffer(0, ElemType::F32, AccessMode::ReadOnly);
    let state_buf = b.storage_buffer(1, ElemType::F32, AccessMode::ReadWrite);
    let weight_buf = b.storage_buffer(2, ElemType::F32, AccessMode::ReadOnly);
    let out_buf = b.storage_buffer(3, ElemType::F32, AccessMode::ReadWrite);

    let gid = b.global_invocation_id();
    let ch = b.load_input_component(gid, 0);
    let channels = b.load_push(push_var, 0);
    let kernel_size = b.load_push(push_var, 1);
    let c0 = b.const_u32(0);

    emit_conv1d_silu_body(
        &mut b,
        qkv_buf,
        state_buf,
        weight_buf,
        out_buf,
        ch,
        c0,
        channels,
        kernel_size,
    );

    b.finish()
}

/// Emits offset-indexed depthwise causal `conv1d` + rolling history shift + `SiLU` SPIR-V 1.6 words.
///
/// Reads `qkv_in[qkv_offset + ch]` and writes `conv_out[ch]` (`0..channels`) so batched token
/// loops avoid unaligned storage buffer subspans.
///
/// Bindings: `0` read-only `qkv_in` (`f32`), `1` read-write `conv_state_inout` (`f32`),
/// `2` read-only `conv_weight` (`f32`), `3` writable `conv_out` (`f32`).
/// Push constants (`3 × u32`): `[channels, kernel_size, qkv_offset]`.
/// Dispatch: `(channels.div_ceil(64), 1, 1)`.
#[must_use]
pub fn conv1d_silu_offset_f32() -> Vec<u32> {
    let (mut b, push_var) = ShaderBuilder::new(false, 3);
    let push_var = push_var.unwrap_or(0);
    let qkv_buf = b.storage_buffer(0, ElemType::F32, AccessMode::ReadOnly);
    let state_buf = b.storage_buffer(1, ElemType::F32, AccessMode::ReadWrite);
    let weight_buf = b.storage_buffer(2, ElemType::F32, AccessMode::ReadOnly);
    let out_buf = b.storage_buffer(3, ElemType::F32, AccessMode::ReadWrite);

    let gid = b.global_invocation_id();
    let ch = b.load_input_component(gid, 0);
    let channels = b.load_push(push_var, 0);
    let kernel_size = b.load_push(push_var, 1);
    let qkv_offset = b.load_push(push_var, 2);

    emit_conv1d_silu_body(
        &mut b,
        qkv_buf,
        state_buf,
        weight_buf,
        out_buf,
        ch,
        qkv_offset,
        channels,
        kernel_size,
    );

    b.finish()
}

/// Emits per-head `ClampedL2` normalization SPIR-V 1.6 words (`x *= 1.0 / max(sqrt(sum(x^2)), eps)`).
///
/// Bindings: `0` read-write `vec_inout` (`f32`, `[heads, head_dim]`).
/// Push constants (`3 × u32`): `[heads, head_dim, eps_bits]`. Dispatch: `(1, heads, 1)`.
#[must_use]
pub fn l2_norm_f32() -> Vec<u32> {
    let (mut b, push_var) = ShaderBuilder::new(false, 3);
    let push_var = push_var.unwrap_or(0);
    let vec_buf = b.storage_buffer(0, ElemType::F32, AccessMode::ReadWrite);
    let wg_reduce = b.workgroup_f32_array(WORKGROUP_X);

    let lid_vec = b.local_invocation_id();
    let wid_vec = b.workgroup_id();
    let lid = b.load_input_component(lid_vec, 0);
    let h = b.load_input_component(wid_vec, 1);

    let heads = b.load_push(push_var, 0);
    let head_dim = b.load_push(push_var, 1);
    let eps_bits = b.load_push(push_var, 2);
    let eps = b.bitcast(b.f32_t, eps_bits);

    let h_ok = b.u_lt(h, heads);
    b.emit_if(h_ok, |b| {
        let c64 = b.const_u32(WORKGROUP_X);
        let f_zero = b.const_f32(0.0);
        let f_one = b.const_f32(1.0);
        let base = b.umul(h, head_dim);

        let acc_var = b.local_f32_var();
        b.store_var(acc_var, f_zero);

        b.emit_for_range(lid, head_dim, c64, |b, i| {
            let idx = b.uadd(base, i);
            let x = b.load_sb_f32(vec_buf, idx);
            let sq = b.fmul_nc(x, x);
            let prev = b.load_var_f32(acc_var);
            let next = b.fadd_nc(prev, sq);
            b.store_var(acc_var, next);
        });

        let partial = b.load_var_f32(acc_var);
        b.store_wg_f32(wg_reduce, lid, partial);
        b.workgroup_barrier();
        let sum_sq = emit_wg_reduce_64(b, wg_reduce, lid, false);

        let norm = b.f_sqrt(sum_sq);
        let clamped = b.f_max(norm, eps);
        let scale = b.fdiv(f_one, clamped);

        b.emit_for_range(lid, head_dim, c64, |b, i| {
            let idx = b.uadd(base, i);
            let x = b.load_sb_f32(vec_buf, idx);
            let scaled = b.fmul_nc(x, scale);
            b.store_sb_f32(vec_buf, idx, scaled);
        });
    });

    b.finish()
}

#[expect(
    clippy::too_many_arguments,
    reason = "Shares GDN recurrent state update arithmetic across standalone and token-indexed batched buffers."
)]
fn emit_gdn_step_body(
    b: &mut ShaderBuilder,
    state_buf: u32,
    q_buf: u32,
    k_buf: u32,
    v_buf: u32,
    alpha_buf: u32,
    beta_buf: u32,
    out_buf: u32,
    wg_reduce: u32,
    lid: u32,
    row: u32,
    vh: u32,
    v_heads: u32,
    k_heads: u32,
    head_dim: u32,
    scale: u32,
    q_base_off: u32,
    k_base_off: u32,
    v_base_off: u32,
    gate_off: u32,
    out_off: u32,
) {
    let c0 = b.const_u32(0);
    let row_ok = b.u_lt(row, head_dim);
    let vh_ok = b.u_lt(vh, v_heads);
    let k_ok = b.u_ne(k_heads, c0);
    let row_vh_ok = b.logical_and(row_ok, vh_ok);
    let in_bounds = b.logical_and(row_vh_ok, k_ok);

    b.emit_if(in_bounds, |b| {
        let c64 = b.const_u32(WORKGROUP_X);
        let f_zero = b.const_f32(0.0);

        let q_div = b.udiv(vh, k_heads);
        let q_mul = b.umul(q_div, k_heads);
        let kh = b.usub(vh, q_mul);

        let kh_dim = b.umul(kh, head_dim);
        let q_base = b.uadd(q_base_off, kh_dim);
        let k_base = b.uadd(k_base_off, kh_dim);
        let v_head_base = b.umul(vh, head_dim);
        let head_state_elems = b.umul(head_dim, head_dim);
        let state_head_base = b.umul(vh, head_state_elems);
        let state_row_off = b.umul(row, head_dim);
        let state_row_base = b.uadd(state_head_base, state_row_off);

        let gate_idx = b.uadd(gate_off, vh);
        let decay = b.load_sb_f32(alpha_buf, gate_idx);
        let beta_val = b.load_sb_f32(beta_buf, gate_idx);
        let v_rel = b.uadd(v_head_base, row);
        let v_idx = b.uadd(v_base_off, v_rel);
        let v_val = b.load_sb_f32(v_buf, v_idx);

        let k_dot_var = b.local_f32_var();
        b.store_var(k_dot_var, f_zero);
        b.emit_for_range(lid, head_dim, c64, |b, col| {
            let s_idx = b.uadd(state_row_base, col);
            let k_idx = b.uadd(k_base, col);
            let s_old = b.load_sb_f32(state_buf, s_idx);
            let s_dec = b.fmul_nc(s_old, decay);
            b.store_sb_f32(state_buf, s_idx, s_dec);
            let kv = b.load_sb_f32(k_buf, k_idx);
            let prod = b.fmul_nc(s_dec, kv);
            let prev = b.load_var_f32(k_dot_var);
            let next = b.fadd_nc(prev, prod);
            b.store_var(k_dot_var, next);
        });
        let partial_k = b.load_var_f32(k_dot_var);
        b.store_wg_f32(wg_reduce, lid, partial_k);
        b.workgroup_barrier();
        let dot_k = emit_wg_reduce_64(b, wg_reduce, lid, false);
        b.workgroup_barrier();

        let err = b.fsub_nc(v_val, dot_k);
        let delta = b.fmul_nc(err, beta_val);

        let q_dot_var = b.local_f32_var();
        b.store_var(q_dot_var, f_zero);
        b.emit_for_range(lid, head_dim, c64, |b, col| {
            let s_idx = b.uadd(state_row_base, col);
            let k_idx = b.uadd(k_base, col);
            let q_idx = b.uadd(q_base, col);
            let s_dec = b.load_sb_f32(state_buf, s_idx);
            let kv = b.load_sb_f32(k_buf, k_idx);
            let qv = b.load_sb_f32(q_buf, q_idx);
            let upd = b.fmul_nc(delta, kv);
            let s_new = b.fadd_nc(s_dec, upd);
            b.store_sb_f32(state_buf, s_idx, s_new);
            let q_prod = b.fmul_nc(s_new, qv);
            let prev = b.load_var_f32(q_dot_var);
            let next = b.fadd_nc(prev, q_prod);
            b.store_var(q_dot_var, next);
        });
        let partial_q = b.load_var_f32(q_dot_var);
        b.store_wg_f32(wg_reduce, lid, partial_q);
        b.workgroup_barrier();
        let dot_q = emit_wg_reduce_64(b, wg_reduce, lid, false);

        let is_t0 = b.u_eq(lid, c0);
        b.emit_if(is_t0, |b| {
            let out_v = b.fmul_nc(dot_q, scale);
            let out_idx = b.uadd(out_off, v_rel);
            b.store_sb_f32(out_buf, out_idx, out_v);
        });
    });
}

/// Emits parallel per-head F32 GDN recurrent step SPIR-V 1.6 words (`[num_v_heads, head_dim, head_dim]`).
///
/// Updates each row `(vh, row)` in value-head, value-row, contiguous-key-column order:
/// 1. `S[vh, row, :] *= alpha_decay[vh]` and `dot_k = dot(S[vh, row, :], k[kh, :])`
/// 2. `delta = (v[vh, row] - dot_k) * beta[vh]`
/// 3. `S[vh, row, :] += delta * k[kh, :]` and `out[vh, row] = dot(S[vh, row, :], q[kh, :]) * scale`
///
///    where `kh = vh % num_k_heads`.
///
/// Bindings: `0` read-write `state_inout` (`f32`), `1` read-only `q` (`f32`), `2` read-only `k` (`f32`),
/// `3` read-only `v` (`f32`), `4` read-only `alpha_decay` (`f32`), `5` read-only `beta` (`f32`),
/// `6` writable `out` (`f32`).
/// Push constants (`4 × u32`): `[num_v_heads, num_k_heads, head_dim, scale_bits]`.
/// Dispatch: `(head_dim, num_v_heads, 1)`.
#[must_use]
pub fn gdn_step_f32() -> Vec<u32> {
    let (mut b, push_var) = ShaderBuilder::new(false, 4);
    let push_var = push_var.unwrap_or(0);
    let state_buf = b.storage_buffer(0, ElemType::F32, AccessMode::ReadWrite);
    let q_buf = b.storage_buffer(1, ElemType::F32, AccessMode::ReadOnly);
    let k_buf = b.storage_buffer(2, ElemType::F32, AccessMode::ReadOnly);
    let v_buf = b.storage_buffer(3, ElemType::F32, AccessMode::ReadOnly);
    let alpha_buf = b.storage_buffer(4, ElemType::F32, AccessMode::ReadOnly);
    let beta_buf = b.storage_buffer(5, ElemType::F32, AccessMode::ReadOnly);
    let out_buf = b.storage_buffer(6, ElemType::F32, AccessMode::ReadWrite);
    let wg_reduce = b.workgroup_f32_array(WORKGROUP_X);

    let lid_vec = b.local_invocation_id();
    let wid_vec = b.workgroup_id();
    let lid = b.load_input_component(lid_vec, 0);
    let row = b.load_input_component(wid_vec, 0);
    let vh = b.load_input_component(wid_vec, 1);

    let v_heads = b.load_push(push_var, 0);
    let k_heads = b.load_push(push_var, 1);
    let head_dim = b.load_push(push_var, 2);
    let scale_bits = b.load_push(push_var, 3);
    let scale = b.bitcast(b.f32_t, scale_bits);
    let c0 = b.const_u32(0);

    emit_gdn_step_body(
        &mut b, state_buf, q_buf, k_buf, v_buf, alpha_buf, beta_buf, out_buf, wg_reduce, lid, row,
        vh, v_heads, k_heads, head_dim, scale, c0, c0, c0, c0, c0,
    );

    b.finish()
}

/// Emits token-indexed per-head F32 GDN recurrent step SPIR-V 1.6 words for batched prefill.
///
/// Reads `[q, k, v]` packed at offset `0` of `qkv_conv`, reads `alpha_decay`/`beta` at
/// `token_idx * num_v_heads`, updates `state_inout`, and writes `out` at
/// `token_idx * (num_v_heads * head_dim)` without requiring aligned buffer subspans.
///
/// Bindings: `0` read-write `state_inout` (`f32`), `1` read-only `qkv_conv` (`f32`),
/// `2` read-only `alpha_decay` (`f32`, `[tokens, num_v_heads]`),
/// `3` read-only `beta` (`f32`, `[tokens, num_v_heads]`),
/// `4` writable `out` (`f32`, `[tokens, num_v_heads * head_dim]`).
/// Push constants (`5 × u32`): `[num_v_heads, num_k_heads, head_dim, scale_bits, token_idx]`.
/// Dispatch: `(head_dim, num_v_heads, 1)`.
#[must_use]
pub fn gdn_step_batched_token_f32() -> Vec<u32> {
    let (mut b, push_var) = ShaderBuilder::new(false, 5);
    let push_var = push_var.unwrap_or(0);
    let state_buf = b.storage_buffer(0, ElemType::F32, AccessMode::ReadWrite);
    let qkv_buf = b.storage_buffer(1, ElemType::F32, AccessMode::ReadOnly);
    let alpha_buf = b.storage_buffer(2, ElemType::F32, AccessMode::ReadOnly);
    let beta_buf = b.storage_buffer(3, ElemType::F32, AccessMode::ReadOnly);
    let out_buf = b.storage_buffer(4, ElemType::F32, AccessMode::ReadWrite);
    let wg_reduce = b.workgroup_f32_array(WORKGROUP_X);

    let lid_vec = b.local_invocation_id();
    let wid_vec = b.workgroup_id();
    let lid = b.load_input_component(lid_vec, 0);
    let row = b.load_input_component(wid_vec, 0);
    let vh = b.load_input_component(wid_vec, 1);

    let v_heads = b.load_push(push_var, 0);
    let k_heads = b.load_push(push_var, 1);
    let head_dim = b.load_push(push_var, 2);
    let scale_bits = b.load_push(push_var, 3);
    let token_idx = b.load_push(push_var, 4);
    let scale = b.bitcast(b.f32_t, scale_bits);

    let c0 = b.const_u32(0);
    let c2 = b.const_u32(2);
    let key_width = b.umul(k_heads, head_dim);
    let two_key_width = b.umul(c2, key_width);
    let val_width = b.umul(v_heads, head_dim);
    let gate_off = b.umul(token_idx, v_heads);
    let out_off = b.umul(token_idx, val_width);

    emit_gdn_step_body(
        &mut b,
        state_buf,
        qkv_buf,
        qkv_buf,
        qkv_buf,
        alpha_buf,
        beta_buf,
        out_buf,
        wg_reduce,
        lid,
        row,
        vh,
        v_heads,
        k_heads,
        head_dim,
        scale,
        c0,
        key_width,
        two_key_width,
        gate_off,
        out_off,
    );

    b.finish()
}

fn emit_gdn_gates_body(
    b: &mut ShaderBuilder,
    alpha_buf: u32,
    beta_buf: u32,
    bias_buf: u32,
    ssm_a_buf: u32,
    gate_idx: u32,
    param_idx: u32,
) {
    let f_zero = b.const_f32(0.0);
    let f_one = b.const_f32(1.0);
    let c_20 = b.const_f32(20.0);

    let beta_raw = b.load_sb_f32(beta_buf, gate_idx);
    let neg_beta = b.fsub_nc(f_zero, beta_raw);
    let exp_neg_beta = b.f_exp(neg_beta);
    let beta_denom = b.fadd_nc(f_one, exp_neg_beta);
    let beta_sig = b.fdiv(f_one, beta_denom);
    b.store_sb_f32(beta_buf, gate_idx, beta_sig);

    let alpha = b.load_sb_f32(alpha_buf, gate_idx);
    let bias = b.load_sb_f32(bias_buf, param_idx);
    let decay_w = b.load_sb_f32(ssm_a_buf, param_idx);

    let is_zero_w = b.f_eq(decay_w, f_zero);
    b.emit_if_else(
        is_zero_w,
        |b| {
            b.store_sb_f32(alpha_buf, gate_idx, f_one);
        },
        |b| {
            let linear = b.fadd_nc(alpha, bias);
            let is_large = b.f_gt(linear, c_20);
            b.emit_if_else(
                is_large,
                |b| {
                    let prod = b.fmul_nc(linear, decay_w);
                    let d = b.f_exp(prod);
                    b.store_sb_f32(alpha_buf, gate_idx, d);
                },
                |b| {
                    let e_lin = b.f_exp(linear);
                    let one_plus = b.fadd_nc(f_one, e_lin);
                    let sp = b.f_log(one_plus);
                    let prod = b.fmul_nc(sp, decay_w);
                    let d = b.f_exp(prod);
                    b.store_sb_f32(alpha_buf, gate_idx, d);
                },
            );
        },
    );
}

/// Emits GDN scalar decay/beta gate preparation SPIR-V 1.6 words (`alpha -> decay`, `beta -> sigmoid(beta)`).
///
/// Bindings: `0` read-write `alpha_inout` (`f32`), `1` read-write `beta_inout` (`f32`),
/// `2` read-only `time_bias` (`f32`), `3` read-only `ssm_a` (`f32`).
/// Push constants (`1 × u32`): `[num_v_heads]`. Dispatch: `(num_v_heads.div_ceil(64), 1, 1)`.
#[must_use]
pub fn gdn_gates_f32() -> Vec<u32> {
    let (mut b, push_var) = ShaderBuilder::new(false, 1);
    let push_var = push_var.unwrap_or(0);
    let alpha_buf = b.storage_buffer(0, ElemType::F32, AccessMode::ReadWrite);
    let beta_buf = b.storage_buffer(1, ElemType::F32, AccessMode::ReadWrite);
    let bias_buf = b.storage_buffer(2, ElemType::F32, AccessMode::ReadOnly);
    let ssm_a_buf = b.storage_buffer(3, ElemType::F32, AccessMode::ReadOnly);

    let gid = b.global_invocation_id();
    let h = b.load_input_component(gid, 0);
    let num_v_heads = b.load_push(push_var, 0);

    let in_bounds = b.u_lt(h, num_v_heads);
    b.emit_if(in_bounds, |b| {
        emit_gdn_gates_body(b, alpha_buf, beta_buf, bias_buf, ssm_a_buf, h, h);
    });

    b.finish()
}

/// Emits batched multi-token GDN scalar decay/beta gate preparation SPIR-V 1.6 words (`[tokens, num_v_heads]`).
///
/// Bindings: `0` read-write `alpha_inout` (`f32`), `1` read-write `beta_inout` (`f32`),
/// `2` read-only `time_bias` (`f32`), `3` read-only `ssm_a` (`f32`).
/// Push constants (`2 × u32`): `[num_v_heads, tokens]`.
/// Dispatch: `(num_v_heads.div_ceil(64), tokens, 1)`.
#[must_use]
pub fn gdn_gates_batched_f32() -> Vec<u32> {
    let (mut b, push_var) = ShaderBuilder::new(false, 2);
    let push_var = push_var.unwrap_or(0);
    let alpha_buf = b.storage_buffer(0, ElemType::F32, AccessMode::ReadWrite);
    let beta_buf = b.storage_buffer(1, ElemType::F32, AccessMode::ReadWrite);
    let bias_buf = b.storage_buffer(2, ElemType::F32, AccessMode::ReadOnly);
    let ssm_a_buf = b.storage_buffer(3, ElemType::F32, AccessMode::ReadOnly);

    let gid = b.global_invocation_id();
    let h = b.load_input_component(gid, 0);
    let tok = b.load_input_component(gid, 1);
    let num_v_heads = b.load_push(push_var, 0);
    let tokens = b.load_push(push_var, 1);

    let h_ok = b.u_lt(h, num_v_heads);
    let tok_ok = b.u_lt(tok, tokens);
    let in_bounds = b.logical_and(h_ok, tok_ok);
    b.emit_if(in_bounds, |b| {
        let tok_base = b.umul(tok, num_v_heads);
        let gate_idx = b.uadd(tok_base, h);
        emit_gdn_gates_body(b, alpha_buf, beta_buf, bias_buf, ssm_a_buf, gate_idx, h);
    });

    b.finish()
}

/// Emits interleaved Q+Gate deinterleaving SPIR-V 1.6 words (`[heads, 2, head_dim] -> q, gate`).
///
/// Bindings: `0` read-only `q_gate_in` (`f32`), `1` writable `q_out` (`f32`), `2` writable `gate_out` (`f32`).
/// Push constants (`2 × u32`): `[num_heads, head_dim]`. Dispatch: `((num_heads * head_dim).div_ceil(64), 1, 1)`.
#[must_use]
pub fn deinterleave_q_gate_f32() -> Vec<u32> {
    let (mut b, push_var) = ShaderBuilder::new(false, 2);
    let push_var = push_var.unwrap_or(0);
    let in_buf = b.storage_buffer(0, ElemType::F32, AccessMode::ReadOnly);
    let q_buf = b.storage_buffer(1, ElemType::F32, AccessMode::ReadWrite);
    let gate_buf = b.storage_buffer(2, ElemType::F32, AccessMode::ReadWrite);

    let gid = b.global_invocation_id();
    let idx = b.load_input_component(gid, 0);
    let num_heads = b.load_push(push_var, 0);
    let head_dim = b.load_push(push_var, 1);

    let c0 = b.const_u32(0);
    let c2 = b.const_u32(2);
    let total = b.umul(num_heads, head_dim);
    let hd_ok = b.u_ne(head_dim, c0);
    let idx_ok = b.u_lt(idx, total);
    let in_bounds = b.logical_and(idx_ok, hd_ok);

    b.emit_if(in_bounds, |b| {
        let h = b.udiv(idx, head_dim);
        let h_dim = b.umul(h, head_dim);
        let d = b.usub(idx, h_dim);
        let two_dim = b.umul(c2, head_dim);
        let q_head_off = b.umul(h, two_dim);
        let g_head_off = b.uadd(q_head_off, head_dim);
        let q_src = b.uadd(q_head_off, d);
        let g_src = b.uadd(g_head_off, d);
        let q_val = b.load_sb_f32(in_buf, q_src);
        let g_val = b.load_sb_f32(in_buf, g_src);
        b.store_sb_f32(q_buf, idx, q_val);
        b.store_sb_f32(gate_buf, idx, g_val);
    });

    b.finish()
}

/// Emits batched multi-token Q+Gate deinterleaving SPIR-V 1.6 words (`[tokens, heads, 2, head_dim] -> q, gate`).
///
/// Bindings: `0` read-only `q_gate_in` (`f32`), `1` writable `q_out` (`f32`), `2` writable `gate_out` (`f32`).
/// Push constants (`3 × u32`): `[num_heads, head_dim, tokens]`.
/// Dispatch: `((num_heads * head_dim).div_ceil(64), tokens, 1)`.
#[must_use]
pub fn deinterleave_q_gate_batched_f32() -> Vec<u32> {
    let (mut b, push_var) = ShaderBuilder::new(false, 3);
    let push_var = push_var.unwrap_or(0);
    let in_buf = b.storage_buffer(0, ElemType::F32, AccessMode::ReadOnly);
    let q_buf = b.storage_buffer(1, ElemType::F32, AccessMode::ReadWrite);
    let gate_buf = b.storage_buffer(2, ElemType::F32, AccessMode::ReadWrite);

    let gid = b.global_invocation_id();
    let idx = b.load_input_component(gid, 0);
    let tok = b.load_input_component(gid, 1);
    let num_heads = b.load_push(push_var, 0);
    let head_dim = b.load_push(push_var, 1);
    let tokens = b.load_push(push_var, 2);

    let c0 = b.const_u32(0);
    let c2 = b.const_u32(2);
    let q_width = b.umul(num_heads, head_dim);
    let hd_ok = b.u_ne(head_dim, c0);
    let idx_ok = b.u_lt(idx, q_width);
    let tok_ok = b.u_lt(tok, tokens);
    let idx_hd_ok = b.logical_and(idx_ok, hd_ok);
    let in_bounds = b.logical_and(idx_hd_ok, tok_ok);

    b.emit_if(in_bounds, |b| {
        let h = b.udiv(idx, head_dim);
        let h_dim = b.umul(h, head_dim);
        let d = b.usub(idx, h_dim);
        let two_dim = b.umul(c2, head_dim);
        let two_q = b.umul(c2, q_width);
        let src_tok_base = b.umul(tok, two_q);
        let q_head_off = b.umul(h, two_dim);
        let q_tok_head = b.uadd(src_tok_base, q_head_off);
        let q_src = b.uadd(q_tok_head, d);
        let g_src = b.uadd(q_src, head_dim);
        let dst_tok_base = b.umul(tok, q_width);
        let dst_idx = b.uadd(dst_tok_base, idx);
        let q_val = b.load_sb_f32(in_buf, q_src);
        let g_val = b.load_sb_f32(in_buf, g_src);
        b.store_sb_f32(q_buf, dst_idx, q_val);
        b.store_sb_f32(gate_buf, dst_idx, g_val);
    });

    b.finish()
}

/// Emits in-place IEEE binary16 value rounding SPIR-V 1.6 words (`v = half_to_float(float_to_half(v))`).
///
/// Bindings: `0` read-write `vec_inout` (`f32`). Push constants (`1 × u32`): `[len]`.
#[must_use]
pub fn round_f16_f32() -> Vec<u32> {
    let (mut b, push_var) = ShaderBuilder::new(false, 1);
    let push_var = push_var.unwrap_or(0);
    let vec_buf = b.storage_buffer(0, ElemType::F32, AccessMode::ReadWrite);

    let gid = b.global_invocation_id();
    let idx = b.load_input_component(gid, 0);
    let len = b.load_push(push_var, 0);

    let in_bounds = b.u_lt(idx, len);
    b.emit_if(in_bounds, |b| {
        let v = b.load_sb_f32(vec_buf, idx);
        let r = b.round_f16(v);
        b.store_sb_f32(vec_buf, idx, r);
    });

    b.finish()
}

/// Emits GPU F32 KV cache token append SPIR-V 1.6 words (`cache[pos * kv_width + i] = cur[i]`).
///
/// Bindings: `0` read-only `cur` (`f32`), `1` writable `cache` (`f32`).
/// Push constants (`2 × u32`): `[kv_width, pos]`. Dispatch: `(kv_width.div_ceil(64), 1, 1)`.
#[must_use]
pub fn kv_append_f32() -> Vec<u32> {
    let (mut b, push_var) = ShaderBuilder::new(false, 2);
    let push_var = push_var.unwrap_or(0);
    let cur_buf = b.storage_buffer(0, ElemType::F32, AccessMode::ReadOnly);
    let cache_buf = b.storage_buffer(1, ElemType::F32, AccessMode::ReadWrite);

    let gid = b.global_invocation_id();
    let idx = b.load_input_component(gid, 0);
    let kv_width = b.load_push(push_var, 0);
    let pos = b.load_push(push_var, 1);

    let in_bounds = b.u_lt(idx, kv_width);
    b.emit_if(in_bounds, |b| {
        let base = b.umul(pos, kv_width);
        let dst = b.uadd(base, idx);
        let v = b.load_sb_f32(cur_buf, idx);
        b.store_sb_f32(cache_buf, dst, v);
    });

    b.finish()
}

/// Emits batched multi-token GPU F32 KV cache append SPIR-V 1.6 words (`[tokens, kv_width]`).
///
/// Bindings: `0` read-only `cur` (`f32`), `1` writable `cache` (`f32`).
/// Push constants (`3 × u32`): `[kv_width, start_pos, tokens]`.
/// Dispatch: `(kv_width.div_ceil(64), tokens, 1)`.
#[must_use]
pub fn kv_append_batched_f32() -> Vec<u32> {
    let (mut b, push_var) = ShaderBuilder::new(false, 3);
    let push_var = push_var.unwrap_or(0);
    let cur_buf = b.storage_buffer(0, ElemType::F32, AccessMode::ReadOnly);
    let cache_buf = b.storage_buffer(1, ElemType::F32, AccessMode::ReadWrite);

    let gid = b.global_invocation_id();
    let idx = b.load_input_component(gid, 0);
    let tok = b.load_input_component(gid, 1);
    let kv_width = b.load_push(push_var, 0);
    let start_pos = b.load_push(push_var, 1);
    let tokens = b.load_push(push_var, 2);

    let idx_ok = b.u_lt(idx, kv_width);
    let tok_ok = b.u_lt(tok, tokens);
    let in_bounds = b.logical_and(idx_ok, tok_ok);
    b.emit_if(in_bounds, |b| {
        let src_base = b.umul(tok, kv_width);
        let src = b.uadd(src_base, idx);
        let dst_pos = b.uadd(start_pos, tok);
        let dst_base = b.umul(dst_pos, kv_width);
        let dst = b.uadd(dst_base, idx);
        let v = b.load_sb_f32(cur_buf, src);
        b.store_sb_f32(cache_buf, dst, v);
    });

    b.finish()
}

fn emit_kv_append_f16_pair(
    b: &mut ShaderBuilder,
    cur_buf: u32,
    cache_buf: u32,
    src_base: u32,
    dst_pos: u32,
    half_width: u32,
    pair_idx: u32,
) {
    let c1 = b.const_u32(1);
    let c2 = b.const_u32(2);
    let c16 = b.const_u32(16);
    let pair_off = b.umul(pair_idx, c2);
    let e0 = b.uadd(src_base, pair_off);
    let e1 = b.uadd(e0, c1);
    let v0 = b.load_sb_f32(cur_buf, e0);
    let v1 = b.load_sb_f32(cur_buf, e1);
    let h0 = b.round_f16_bits(v0);
    let h1 = b.round_f16_bits(v1);
    let hi_shifted = b.ushl(h1, c16);
    let packed = b.uor(h0, hi_shifted);
    let dst_base = b.umul(dst_pos, half_width);
    let dst = b.uadd(dst_base, pair_idx);
    b.store_sb_u32(cache_buf, dst, packed);
}

/// Emits GPU packed-FP16 KV cache token append SPIR-V 1.6 words (`cache_u32[pos * (kv_width/2) + p]`).
///
/// Bindings: `0` read-only `cur` (`f32`, even `kv_width`), `1` writable `cache_u32` (`u32`).
/// Push constants (`2 × u32`): `[kv_width, pos]`. Dispatch: `((kv_width / 2).div_ceil(64), 1, 1)`.
#[must_use]
pub fn kv_append_f16() -> Vec<u32> {
    let (mut b, push_var) = ShaderBuilder::new(false, 2);
    let push_var = push_var.unwrap_or(0);
    let cur_buf = b.storage_buffer(0, ElemType::F32, AccessMode::ReadOnly);
    let cache_buf = b.storage_buffer(1, ElemType::U32, AccessMode::ReadWrite);

    let gid = b.global_invocation_id();
    let pair_idx = b.load_input_component(gid, 0);
    let kv_width = b.load_push(push_var, 0);
    let pos = b.load_push(push_var, 1);

    let c0 = b.const_u32(0);
    let c1 = b.const_u32(1);
    let half_width = b.ushr(kv_width, c1);

    let in_bounds = b.u_lt(pair_idx, half_width);
    b.emit_if(in_bounds, |b| {
        emit_kv_append_f16_pair(b, cur_buf, cache_buf, c0, pos, half_width, pair_idx);
    });

    b.finish()
}

/// Emits batched multi-token GPU packed-FP16 KV cache append SPIR-V 1.6 words (`[tokens, kv_width]`).
///
/// Bindings: `0` read-only `cur` (`f32`, even `kv_width`), `1` writable `cache_u32` (`u32`).
/// Push constants (`3 × u32`): `[kv_width, start_pos, tokens]`.
/// Dispatch: `((kv_width / 2).div_ceil(64), tokens, 1)`.
#[must_use]
pub fn kv_append_batched_f16() -> Vec<u32> {
    let (mut b, push_var) = ShaderBuilder::new(false, 3);
    let push_var = push_var.unwrap_or(0);
    let cur_buf = b.storage_buffer(0, ElemType::F32, AccessMode::ReadOnly);
    let cache_buf = b.storage_buffer(1, ElemType::U32, AccessMode::ReadWrite);

    let gid = b.global_invocation_id();
    let pair_idx = b.load_input_component(gid, 0);
    let tok = b.load_input_component(gid, 1);
    let kv_width = b.load_push(push_var, 0);
    let start_pos = b.load_push(push_var, 1);
    let tokens = b.load_push(push_var, 2);

    let c1 = b.const_u32(1);
    let half_width = b.ushr(kv_width, c1);

    let pair_ok = b.u_lt(pair_idx, half_width);
    let tok_ok = b.u_lt(tok, tokens);
    let in_bounds = b.logical_and(pair_ok, tok_ok);
    b.emit_if(in_bounds, |b| {
        let src_base = b.umul(tok, kv_width);
        let dst_pos = b.uadd(start_pos, tok);
        emit_kv_append_f16_pair(
            b, cur_buf, cache_buf, src_base, dst_pos, half_width, pair_idx,
        );
    });

    b.finish()
}

/// Emits packed-FP16 to F32 KV unpack SPIR-V 1.6 words for `#40` attention consumption.
///
/// Bindings: `0` read-only `cache_u32` (`u32`), `1` writable `out_f32` (`f32`).
/// Push constants (`1 × u32`): `[word_count]`. Dispatch: `(word_count.div_ceil(64), 1, 1)`.
#[must_use]
pub fn kv_unpack_f16_to_f32() -> Vec<u32> {
    let (mut b, push_var) = ShaderBuilder::new(false, 1);
    let push_var = push_var.unwrap_or(0);
    let cache_buf = b.storage_buffer(0, ElemType::U32, AccessMode::ReadOnly);
    let out_buf = b.storage_buffer(1, ElemType::F32, AccessMode::ReadWrite);

    let gid = b.global_invocation_id();
    let w_idx = b.load_input_component(gid, 0);
    let word_count = b.load_push(push_var, 0);

    let in_bounds = b.u_lt(w_idx, word_count);
    b.emit_if(in_bounds, |b| {
        let c1 = b.const_u32(1);
        let c2 = b.const_u32(2);
        let packed = b.load_sb_u32(cache_buf, w_idx);
        let (lo, hi) = b.unpack_half_pair(packed);
        let dst0 = b.umul(w_idx, c2);
        let dst1 = b.uadd(dst0, c1);
        b.store_sb_f32(out_buf, dst0, lo);
        b.store_sb_f32(out_buf, dst1, hi);
    });

    b.finish()
}

fn emit_quantize_q8_0_block(
    b: &mut ShaderBuilder,
    in_buf: u32,
    out_buf: u32,
    blk: u32,
    pinned: bool,
    native_scale: bool,
) {
    let c0 = b.const_u32(0);
    let c1 = b.const_u32(1);
    let c9 = b.const_u32(9);
    let c32 = b.const_u32(32);
    let f_zero = b.const_f32(0.0);
    let f_half = b.const_f32(0.5);
    let f_one = b.const_f32(1.0);
    let f_127 = b.const_f32(127.0);
    let i_zero = b.const_i32(0);
    let i_min = b.const_i32(-127);
    let i_max = b.const_i32(127);

    let in_base = b.umul(blk, c32);
    let out_base = b.umul(blk, c9);

    let max_var = b.local_f32_var();
    b.store_var(max_var, f_zero);
    b.emit_for_range(c0, c32, c1, |b, i| {
        let in_idx = b.uadd(in_base, i);
        let v = b.load_sb_f32(in_buf, in_idx);
        let av = b.f_abs(v);
        let prev = b.load_var_f32(max_var);
        let mx = b.f_max(prev, av);
        b.store_var(max_var, mx);
    });
    let maximum = b.load_var_f32(max_var);
    let d = if native_scale {
        let c_inv127 = b.const_f32(f32::from_bits(0x3c01_0204));
        b.fmul_nc(maximum, c_inv127)
    } else {
        b.fdiv(maximum, f_127)
    };
    let scale = b.round_f16(d);
    let scale_u32 = b.bitcast(b.u32_t, scale);
    b.store_sb_u32(out_buf, out_base, scale_u32);

    let zero_blk = b.f_eq(d, f_zero);
    let safe_denom = b.select(
        b.f32_t,
        zero_blk,
        f_one,
        if native_scale { maximum } else { d },
    );
    let numer = if native_scale { f_127 } else { f_one };
    let inv_cand = b.fdiv(numer, safe_denom);
    let inv_scale = b.select(b.f32_t, zero_blk, f_zero, inv_cand);

    for w_j in 0_u32..8 {
        let mut qs = [0_u32; 4];
        for lane in 0_u32..4 {
            let elem_off = b.const_u32(w_j * 4 + lane);
            let elem_idx = b.uadd(in_base, elem_off);
            let v = b.load_sb_f32(in_buf, elem_idx);
            let q = if pinned {
                quantize_one_q8k_lane(b, v, inv_scale, zero_blk, false)
            } else {
                let prod = b.fmul_nc(v, inv_scale);
                let ax = b.f_abs(prod);
                let fl = b.f_floor(ax);
                let frac = b.fsub_nc(ax, fl);
                let up = b.f_ge(frac, f_half);
                let add_bit = b.select(b.f32_t, up, f_one, f_zero);
                let mag_f = b.fadd_nc(fl, add_bit);
                let mag_i = b.f32_to_i32(mag_f);
                let neg_i = b.isub(i_zero, mag_i);
                let is_neg = b.f_lt(prod, f_zero);
                let signed_q = b.select(b.i32_t, is_neg, neg_i, mag_i);
                let q_lo = b.s_max(signed_q, i_min);
                let q_clamped = b.s_min(q_lo, i_max);
                b.select(b.i32_t, zero_blk, i_zero, q_clamped)
            };
            qs[usize::try_from(lane).unwrap_or(0)] = q;
        }
        let packed = pack_four_i8_to_u32(b, qs);
        let dst_off = b.const_u32(1 + w_j);
        let dst_idx = b.uadd(out_base, dst_off);
        b.store_sb_u32(out_buf, dst_idx, packed);
    }
}

/// Emits pure SPIR-V 1.6 `F32 -> Q8_0` (`9` `u32` words per 32 elements) quantizer.
///
/// Matches Prism's `q8_with_input` contracts:
/// - `pinned = false, native_scale = false`: `Source` (`d = max / 127`, `round` ties away from zero)
/// - `pinned = true, native_scale = false`: `Pinned` `Q8` (`d = max / 127`, `round_ties_even`)
/// - `pinned = true, native_scale = true`: `PinnedPtq1Q8` (`d = max * 0x3c010204`, `inv = 127 / max`, `round_ties_even`)
///
/// Bindings: `0` read-only `in_f32` (`f32`), `1` writable `out_q8` (`u32`, `num_blocks * 9`).
/// Push constants (`1 × u32`): `[num_blocks]`. Dispatch: `(num_blocks.div_ceil(64), 1, 1)`.
#[must_use]
pub fn quantize_f32_to_q8_0(pinned: bool, native_scale: bool) -> Vec<u32> {
    let (mut b, push_var) = ShaderBuilder::new(false, 1);
    let push_var = push_var.unwrap_or(0);
    let in_buf = b.storage_buffer(0, ElemType::F32, AccessMode::ReadOnly);
    let out_buf = b.storage_buffer(1, ElemType::U32, AccessMode::ReadWrite);

    let gid = b.global_invocation_id();
    let blk = b.load_input_component(gid, 0);
    let num_blocks = b.load_push(push_var, 0);

    let in_bounds = b.u_lt(blk, num_blocks);
    b.emit_if(in_bounds, |b| {
        emit_quantize_q8_0_block(b, in_buf, out_buf, blk, pinned, native_scale);
    });

    b.finish()
}

/// Emits batched multi-token `F32 -> Q8_0` quantizer SPIR-V 1.6 words (`[tokens, blocks_per_tok * 9]`).
///
/// Bindings: `0` read-only `in_f32` (`f32`), `1` writable `out_q8` (`u32`).
/// Push constants (`2 × u32`): `[blocks_per_tok, tokens]`.
/// Dispatch: `(blocks_per_tok.div_ceil(64), tokens, 1)`.
#[must_use]
pub fn quantize_f32_to_q8_0_batched(pinned: bool, native_scale: bool) -> Vec<u32> {
    let (mut b, push_var) = ShaderBuilder::new(false, 2);
    let push_var = push_var.unwrap_or(0);
    let in_buf = b.storage_buffer(0, ElemType::F32, AccessMode::ReadOnly);
    let out_buf = b.storage_buffer(1, ElemType::U32, AccessMode::ReadWrite);

    let gid = b.global_invocation_id();
    let blk_in_tok = b.load_input_component(gid, 0);
    let tok = b.load_input_component(gid, 1);
    let blocks_per_tok = b.load_push(push_var, 0);
    let tokens = b.load_push(push_var, 1);

    let blk_ok = b.u_lt(blk_in_tok, blocks_per_tok);
    let tok_ok = b.u_lt(tok, tokens);
    let in_bounds = b.logical_and(blk_ok, tok_ok);
    b.emit_if(in_bounds, |b| {
        let tok_base = b.umul(tok, blocks_per_tok);
        let blk = b.uadd(tok_base, blk_in_tok);
        emit_quantize_q8_0_block(b, in_buf, out_buf, blk, pinned, native_scale);
    });

    b.finish()
}

/// Emits dense `F32` matrix-vector multiplication SPIR-V 1.6 words (`[cols, rows] * [cols] -> [rows]`).
///
/// Uses one 64-thread workgroup per output row for coalesced memory loads.
/// Bindings: `0` read-only `weights` (`f32`), `1` read-only `input` (`f32`), `2` writable `output` (`f32`).
/// Push constants (`2 × u32`): `[cols, rows]`. Dispatch: `(rows, 1, 1)`.
#[must_use]
pub fn f32_matvec() -> Vec<u32> {
    let (mut b, push_var) = ShaderBuilder::new(false, 2);
    let push_var = push_var.unwrap_or(0);
    let w_buf = b.storage_buffer(0, ElemType::F32, AccessMode::ReadOnly);
    let in_buf = b.storage_buffer(1, ElemType::F32, AccessMode::ReadOnly);
    let out_buf = b.storage_buffer(2, ElemType::F32, AccessMode::ReadWrite);
    let wg_reduce = b.workgroup_f32_array(WORKGROUP_X);

    let lid_vec = b.local_invocation_id();
    let wid_vec = b.workgroup_id();
    let lid = b.load_input_component(lid_vec, 0);
    let row = b.load_input_component(wid_vec, 0);
    let cols = b.load_push(push_var, 0);
    let rows = b.load_push(push_var, 1);

    let row_ok = b.u_lt(row, rows);
    b.emit_if(row_ok, |b| {
        let c0 = b.const_u32(0);
        let c64 = b.const_u32(WORKGROUP_X);
        let f_zero = b.const_f32(0.0);
        let row_base = b.umul(row, cols);

        let acc_var = b.local_f32_var();
        b.store_var(acc_var, f_zero);
        b.emit_for_range(lid, cols, c64, |b, c| {
            let w_idx = b.uadd(row_base, c);
            let wv = b.load_sb_f32(w_buf, w_idx);
            let xv = b.load_sb_f32(in_buf, c);
            let prod = b.fmul_nc(wv, xv);
            let prev = b.load_var_f32(acc_var);
            let next = b.fadd_nc(prev, prod);
            b.store_var(acc_var, next);
        });

        let partial = b.load_var_f32(acc_var);
        b.store_wg_f32(wg_reduce, lid, partial);
        b.workgroup_barrier();
        let sum = emit_wg_reduce_64(b, wg_reduce, lid, false);

        let is_t0 = b.u_eq(lid, c0);
        b.emit_if(is_t0, |b| {
            b.store_sb_f32(out_buf, row, sum);
        });
    });

    b.finish()
}

/// Emits batched 4-token `F32` GEMM SPIR-V 1.6 words (`[cols, rows] * [tokens, cols] -> [tokens, rows]`).
///
/// Uses one 64-thread workgroup per `(row, token_tile_4)` so each weight element is read once
/// and reused across up to 4 tokens with identical reduction order to [`f32_matvec`].
/// Bindings: `0` read-only `weights` (`f32`), `1` read-only `input` (`f32`), `2` writable `output` (`f32`).
/// Push constants (`3 × u32`): `[cols, rows, tokens]`. Dispatch: `(rows, tokens.div_ceil(4), 1)`.
#[must_use]
pub fn f32_gemm_batched() -> Vec<u32> {
    let (mut b, push_var) = ShaderBuilder::new(false, 3);
    let push_var = push_var.unwrap_or(0);
    let w_buf = b.storage_buffer(0, ElemType::F32, AccessMode::ReadOnly);
    let in_buf = b.storage_buffer(1, ElemType::F32, AccessMode::ReadOnly);
    let out_buf = b.storage_buffer(2, ElemType::F32, AccessMode::ReadWrite);
    let wg_reduce = b.workgroup_f32_array(WORKGROUP_X);

    let lid_vec = b.local_invocation_id();
    let wid_vec = b.workgroup_id();
    let lid = b.load_input_component(lid_vec, 0);
    let row = b.load_input_component(wid_vec, 0);
    let tok_tile = b.load_input_component(wid_vec, 1);

    let cols = b.load_push(push_var, 0);
    let rows = b.load_push(push_var, 1);
    let tokens = b.load_push(push_var, 2);

    let c4 = b.const_u32(4);
    let tok_base = b.umul(tok_tile, c4);
    let row_ok = b.u_lt(row, rows);
    let tok_ok = b.u_lt(tok_base, tokens);
    let in_bounds = b.logical_and(row_ok, tok_ok);

    b.emit_if(in_bounds, |b| {
        let c0 = b.const_u32(0);
        let c64 = b.const_u32(WORKGROUP_X);
        let f_zero = b.const_f32(0.0);
        let row_base = b.umul(row, cols);
        let (tok_clamped, tok_in_bases) = clamp_four_token_act_bases(b, tok_base, tokens, cols);

        let acc_vars = [
            b.local_f32_var(),
            b.local_f32_var(),
            b.local_f32_var(),
            b.local_f32_var(),
        ];
        for v in acc_vars {
            b.store_var(v, f_zero);
        }

        b.emit_for_range(lid, cols, c64, |b, c| {
            let w_idx = b.uadd(row_base, c);
            let wv = b.load_sb_f32(w_buf, w_idx);
            for (&in_base, &acc_var) in tok_in_bases.iter().zip(&acc_vars) {
                let x_idx = b.uadd(in_base, c);
                let xv = b.load_sb_f32(in_buf, x_idx);
                let prod = b.fmul_nc(wv, xv);
                let prev = b.load_var_f32(acc_var);
                let next = b.fadd_nc(prev, prod);
                b.store_var(acc_var, next);
            }
        });

        for (k, (&tk, &acc_var)) in tok_clamped.iter().zip(&acc_vars).enumerate() {
            if k > 0 {
                b.workgroup_barrier();
            }
            let partial = b.load_var_f32(acc_var);
            b.store_wg_f32(wg_reduce, lid, partial);
            b.workgroup_barrier();
            let sum = emit_wg_reduce_64(b, wg_reduce, lid, false);

            let is_t0 = b.u_eq(lid, c0);
            let valid = b.u_lt(tk, tokens);
            let write_ok = b.logical_and(is_t0, valid);
            b.emit_if(write_ok, |b| {
                let tok_out = b.umul(tk, rows);
                let out_idx = b.uadd(tok_out, row);
                b.store_sb_f32(out_buf, out_idx, sum);
            });
        }
    });

    b.finish()
}

/// Emits dense `F32` `LoRA` B accumulation SPIR-V 1.6 words (`output[row] += scale * dot(B[row], ranks)`).
///
/// Bindings: `0` read-only `b_weights` (`f32`, `[out_features, rank]` row-major),
/// `1` read-only `ranks` (`f32`, `[rank]`), `2` read-write `output` (`f32`, `[out_features]`).
/// Push constants (`3 × u32`): `[out_features, rank, scale_bits]`.
/// Dispatch: `(out_features.div_ceil(64), 1, 1)`.
#[must_use]
pub fn lora_b_accum_f32() -> Vec<u32> {
    let (mut b, push_var) = ShaderBuilder::new(false, 3);
    let push_var = push_var.unwrap_or(0);
    let b_buf = b.storage_buffer(0, ElemType::F32, AccessMode::ReadOnly);
    let ranks_buf = b.storage_buffer(1, ElemType::F32, AccessMode::ReadOnly);
    let out_buf = b.storage_buffer(2, ElemType::F32, AccessMode::ReadWrite);

    let gid = b.global_invocation_id();
    let row = b.load_input_component(gid, 0);
    let out_features = b.load_push(push_var, 0);
    let rank = b.load_push(push_var, 1);
    let scale_bits = b.load_push(push_var, 2);
    let scale = b.bitcast(b.f32_t, scale_bits);

    let in_bounds = b.u_lt(row, out_features);
    b.emit_if(in_bounds, |b| {
        let c0 = b.const_u32(0);
        let c1 = b.const_u32(1);
        let f_zero = b.const_f32(0.0);
        let row_base = b.umul(row, rank);

        let acc_var = b.local_f32_var();
        b.store_var(acc_var, f_zero);
        b.emit_for_range(c0, rank, c1, |b, r| {
            let b_idx = b.uadd(row_base, r);
            let bv = b.load_sb_f32(b_buf, b_idx);
            let rv = b.load_sb_f32(ranks_buf, r);
            let prod = b.fmul_nc(bv, rv);
            let prev = b.load_var_f32(acc_var);
            let next = b.fadd_nc(prev, prod);
            b.store_var(acc_var, next);
        });

        let dot = b.load_var_f32(acc_var);
        let scaled = b.fmul_nc(dot, scale);
        let cur = b.load_sb_f32(out_buf, row);
        let updated = b.fadd_nc(cur, scaled);
        b.store_sb_f32(out_buf, row, updated);
    });

    b.finish()
}

/// Emits batched 4-token `F32` `LoRA` B accumulation SPIR-V 1.6 words (`output[t, row] += scale * dot(B[row], ranks[t])`).
///
/// Bindings: `0` read-only `b_weights` (`f32`, `[out_features, rank]` row-major),
/// `1` read-only `ranks` (`f32`, `[tokens, rank]`), `2` read-write `output` (`f32`, `[tokens, out_features]`).
/// Push constants (`4 × u32`): `[out_features, rank, tokens, scale_bits]`.
/// Dispatch: `(out_features.div_ceil(64), tokens.div_ceil(4), 1)`.
#[must_use]
pub fn lora_b_accum_batched_f32() -> Vec<u32> {
    let (mut b, push_var) = ShaderBuilder::new(false, 4);
    let push_var = push_var.unwrap_or(0);
    let b_buf = b.storage_buffer(0, ElemType::F32, AccessMode::ReadOnly);
    let ranks_buf = b.storage_buffer(1, ElemType::F32, AccessMode::ReadOnly);
    let out_buf = b.storage_buffer(2, ElemType::F32, AccessMode::ReadWrite);

    let gid = b.global_invocation_id();
    let row = b.load_input_component(gid, 0);
    let tok_tile = b.load_input_component(gid, 1);
    let out_features = b.load_push(push_var, 0);
    let rank = b.load_push(push_var, 1);
    let tokens = b.load_push(push_var, 2);
    let scale_bits = b.load_push(push_var, 3);
    let scale = b.bitcast(b.f32_t, scale_bits);

    let c4 = b.const_u32(4);
    let tok_base = b.umul(tok_tile, c4);
    let row_ok = b.u_lt(row, out_features);
    let tok_ok = b.u_lt(tok_base, tokens);
    let in_bounds = b.logical_and(row_ok, tok_ok);

    b.emit_if(in_bounds, |b| {
        let c0 = b.const_u32(0);
        let c1 = b.const_u32(1);
        let f_zero = b.const_f32(0.0);
        let row_base = b.umul(row, rank);
        let (tok_clamped, tok_rank_bases) = clamp_four_token_act_bases(b, tok_base, tokens, rank);

        let acc_vars = [
            b.local_f32_var(),
            b.local_f32_var(),
            b.local_f32_var(),
            b.local_f32_var(),
        ];
        for v in acc_vars {
            b.store_var(v, f_zero);
        }

        b.emit_for_range(c0, rank, c1, |b, r| {
            let b_idx = b.uadd(row_base, r);
            let bv = b.load_sb_f32(b_buf, b_idx);
            for (&r_base, &acc_var) in tok_rank_bases.iter().zip(&acc_vars) {
                let r_idx = b.uadd(r_base, r);
                let rv = b.load_sb_f32(ranks_buf, r_idx);
                let prod = b.fmul_nc(bv, rv);
                let prev = b.load_var_f32(acc_var);
                let next = b.fadd_nc(prev, prod);
                b.store_var(acc_var, next);
            }
        });

        for (&tk, &acc_var) in tok_clamped.iter().zip(&acc_vars) {
            let valid = b.u_lt(tk, tokens);
            b.emit_if(valid, |b| {
                let dot = b.load_var_f32(acc_var);
                let scaled = b.fmul_nc(dot, scale);
                let tok_out = b.umul(tk, out_features);
                let out_idx = b.uadd(tok_out, row);
                let cur = b.load_sb_f32(out_buf, out_idx);
                let updated = b.fadd_nc(cur, scaled);
                b.store_sb_f32(out_buf, out_idx, updated);
            });
        }
    });

    b.finish()
}

/// Emits elementwise `F32` product SPIR-V 1.6 words (`out[i] = a[i] * b[i]`).
///
/// Bindings: `0` read-only `a` (`f32`), `1` read-only `b` (`f32`), `2` writable `out` (`f32`).
/// Push constants (`1 × u32`): `[len]`. Dispatch: `(len.div_ceil(64), 1, 1)`.
#[must_use]
pub fn mul_f32_elemwise() -> Vec<u32> {
    let (mut b, push_var) = ShaderBuilder::new(false, 1);
    let push_var = push_var.unwrap_or(0);
    let a_buf = b.storage_buffer(0, ElemType::F32, AccessMode::ReadOnly);
    let b_buf = b.storage_buffer(1, ElemType::F32, AccessMode::ReadOnly);
    let out_buf = b.storage_buffer(2, ElemType::F32, AccessMode::ReadWrite);

    let gid = b.global_invocation_id();
    let idx = b.load_input_component(gid, 0);
    let len = b.load_push(push_var, 0);

    let in_bounds = b.u_lt(idx, len);
    b.emit_if(in_bounds, |b| {
        let av = b.load_sb_f32(a_buf, idx);
        let bv = b.load_sb_f32(b_buf, idx);
        let prod = b.fmul_nc(av, bv);
        b.store_sb_f32(out_buf, idx, prod);
    });

    b.finish()
}

/// Emits batched broadcast elementwise `F32` product SPIR-V 1.6 words (`out[t, i] = a[t, i] * b[i]`).
///
/// Broadcasts 1-token sign/scale vector `b` (`[len]`) across all `tokens` rows of `a` (`[tokens, len]`).
/// Bindings: `0` read-only `a` (`f32`), `1` read-only `b` (`f32`), `2` writable `out` (`f32`).
/// Push constants (`2 × u32`): `[len, tokens]`. Dispatch: `(len.div_ceil(64), tokens, 1)`.
#[must_use]
pub fn mul_f32_elemwise_batched() -> Vec<u32> {
    let (mut b, push_var) = ShaderBuilder::new(false, 2);
    let push_var = push_var.unwrap_or(0);
    let a_buf = b.storage_buffer(0, ElemType::F32, AccessMode::ReadOnly);
    let b_buf = b.storage_buffer(1, ElemType::F32, AccessMode::ReadOnly);
    let out_buf = b.storage_buffer(2, ElemType::F32, AccessMode::ReadWrite);

    let gid = b.global_invocation_id();
    let idx = b.load_input_component(gid, 0);
    let tok = b.load_input_component(gid, 1);
    let len = b.load_push(push_var, 0);
    let tokens = b.load_push(push_var, 1);

    let idx_ok = b.u_lt(idx, len);
    let tok_ok = b.u_lt(tok, tokens);
    let in_bounds = b.logical_and(idx_ok, tok_ok);
    b.emit_if(in_bounds, |b| {
        let tok_base = b.umul(tok, len);
        let full_idx = b.uadd(tok_base, idx);
        let av = b.load_sb_f32(a_buf, full_idx);
        let bv = b.load_sb_f32(b_buf, idx);
        let prod = b.fmul_nc(av, bv);
        b.store_sb_f32(out_buf, full_idx, prod);
    });

    b.finish()
}

#[expect(
    clippy::too_many_arguments,
    reason = "Shares `[repeat, key, d] -> [key, repeat, d]` permutation indexing across single-token and batched SPIR-V shaders."
)]
fn emit_permute_ssm_heads_elem(
    b: &mut ShaderBuilder,
    in_buf: u32,
    out_buf: u32,
    tok_base: u32,
    idx: u32,
    num_k_heads: u32,
    repetitions: u32,
    head_dim: u32,
) {
    let head_idx = b.udiv(idx, head_dim);
    let head_off = b.umul(head_idx, head_dim);
    let d = b.usub(idx, head_off);
    let repeat = b.udiv(head_idx, num_k_heads);
    let rep_k = b.umul(repeat, num_k_heads);
    let key = b.usub(head_idx, rep_k);
    let key_rep = b.umul(key, repetitions);
    let dst_head = b.uadd(key_rep, repeat);
    let dst_head_off = b.umul(dst_head, head_dim);
    let dst_idx = b.uadd(dst_head_off, d);
    let src_full = b.uadd(tok_base, idx);
    let dst_full = b.uadd(tok_base, dst_idx);
    let v = b.load_sb_f32(in_buf, src_full);
    b.store_sb_f32(out_buf, dst_full, v);
}

/// Emits grouped GDN SSM head permutation SPIR-V 1.6 words (`[repeat, key, d] -> [key, repeat, d]`).
///
/// Bindings: `0` read-only `in_f32` (`f32`), `1` writable `out_f32` (`f32`).
/// Push constants (`3 × u32`): `[num_k_heads, repetitions, head_dim]`.
/// Dispatch: `((num_k_heads * repetitions * head_dim).div_ceil(64), 1, 1)`.
#[must_use]
pub fn permute_ssm_heads_f32() -> Vec<u32> {
    let (mut b, push_var) = ShaderBuilder::new(false, 3);
    let push_var = push_var.unwrap_or(0);
    let in_buf = b.storage_buffer(0, ElemType::F32, AccessMode::ReadOnly);
    let out_buf = b.storage_buffer(1, ElemType::F32, AccessMode::ReadWrite);

    let gid = b.global_invocation_id();
    let idx = b.load_input_component(gid, 0);
    let num_k_heads = b.load_push(push_var, 0);
    let repetitions = b.load_push(push_var, 1);
    let head_dim = b.load_push(push_var, 2);

    let c0 = b.const_u32(0);
    let kh_rep = b.umul(num_k_heads, repetitions);
    let total = b.umul(kh_rep, head_dim);
    let kh_ok = b.u_ne(num_k_heads, c0);
    let hd_ok = b.u_ne(head_dim, c0);
    let dims_ok = b.logical_and(kh_ok, hd_ok);
    let idx_ok = b.u_lt(idx, total);
    let in_bounds = b.logical_and(idx_ok, dims_ok);

    b.emit_if(in_bounds, |b| {
        emit_permute_ssm_heads_elem(
            b,
            in_buf,
            out_buf,
            c0,
            idx,
            num_k_heads,
            repetitions,
            head_dim,
        );
    });

    b.finish()
}

/// Emits batched multi-token grouped GDN SSM head permutation SPIR-V 1.6 words (`[tokens, repeat, key, d] -> [tokens, key, repeat, d]`).
///
/// Bindings: `0` read-only `in_f32` (`f32`), `1` writable `out_f32` (`f32`).
/// Push constants (`4 × u32`): `[num_k_heads, repetitions, head_dim, tokens]`.
/// Dispatch: `((num_k_heads * repetitions * head_dim).div_ceil(64), tokens, 1)`.
#[must_use]
pub fn permute_ssm_heads_batched_f32() -> Vec<u32> {
    let (mut b, push_var) = ShaderBuilder::new(false, 4);
    let push_var = push_var.unwrap_or(0);
    let in_buf = b.storage_buffer(0, ElemType::F32, AccessMode::ReadOnly);
    let out_buf = b.storage_buffer(1, ElemType::F32, AccessMode::ReadWrite);

    let gid = b.global_invocation_id();
    let idx = b.load_input_component(gid, 0);
    let tok = b.load_input_component(gid, 1);
    let num_k_heads = b.load_push(push_var, 0);
    let repetitions = b.load_push(push_var, 1);
    let head_dim = b.load_push(push_var, 2);
    let tokens = b.load_push(push_var, 3);

    let c0 = b.const_u32(0);
    let kh_rep = b.umul(num_k_heads, repetitions);
    let total = b.umul(kh_rep, head_dim);
    let kh_ok = b.u_ne(num_k_heads, c0);
    let hd_ok = b.u_ne(head_dim, c0);
    let dims_ok = b.logical_and(kh_ok, hd_ok);
    let idx_ok = b.u_lt(idx, total);
    let tok_ok = b.u_lt(tok, tokens);
    let it_ok = b.logical_and(idx_ok, tok_ok);
    let in_bounds = b.logical_and(it_ok, dims_ok);

    b.emit_if(in_bounds, |b| {
        let tok_base = b.umul(tok, total);
        emit_permute_ssm_heads_elem(
            b,
            in_buf,
            out_buf,
            tok_base,
            idx,
            num_k_heads,
            repetitions,
            head_dim,
        );
    });

    b.finish()
}

/// Emits contiguous `F32` slice copy SPIR-V 1.6 words (`dst[i] = src[src_offset + i]`).
///
/// Useful for extracting unaligned tensor sub-slices on GPU without violating Vulkan
/// `minStorageBufferOffsetAlignment`.
///
/// Bindings: `0` read-only `src` (`f32`), `1` writable `dst` (`f32`).
/// Push constants (`2 × u32`): `[src_offset, len]`. Dispatch: `(len.div_ceil(64), 1, 1)`.
#[must_use]
pub fn copy_slice_f32() -> Vec<u32> {
    let (mut b, push_var) = ShaderBuilder::new(false, 2);
    let push_var = push_var.unwrap_or(0);
    let src_buf = b.storage_buffer(0, ElemType::F32, AccessMode::ReadOnly);
    let dst_buf = b.storage_buffer(1, ElemType::F32, AccessMode::ReadWrite);

    let gid = b.global_invocation_id();
    let idx = b.load_input_component(gid, 0);
    let src_offset = b.load_push(push_var, 0);
    let len = b.load_push(push_var, 1);

    let in_bounds = b.u_lt(idx, len);
    b.emit_if(in_bounds, |b| {
        let src_idx = b.uadd(src_offset, idx);
        let v = b.load_sb_f32(src_buf, src_idx);
        b.store_sb_f32(dst_buf, idx, v);
    });

    b.finish()
}

/// Emits contiguous `F32` slice copy SPIR-V 1.6 words with both source and destination offsets
/// (`dst[dst_offset + i] = src[src_offset + i]`).
///
/// Bindings: `0` read-only `src` (`f32`), `1` writable `dst` (`f32`).
/// Push constants (`3 × u32`): `[src_offset, dst_offset, len]`. Dispatch: `(len.div_ceil(64), 1, 1)`.
#[must_use]
pub fn copy_slice_offset_f32() -> Vec<u32> {
    let (mut b, push_var) = ShaderBuilder::new(false, 3);
    let push_var = push_var.unwrap_or(0);
    let src_buf = b.storage_buffer(0, ElemType::F32, AccessMode::ReadOnly);
    let dst_buf = b.storage_buffer(1, ElemType::F32, AccessMode::ReadWrite);

    let gid = b.global_invocation_id();
    let idx = b.load_input_component(gid, 0);
    let src_offset = b.load_push(push_var, 0);
    let dst_offset = b.load_push(push_var, 1);
    let len = b.load_push(push_var, 2);

    let in_bounds = b.u_lt(idx, len);
    b.emit_if(in_bounds, |b| {
        let src_idx = b.uadd(src_offset, idx);
        let dst_idx = b.uadd(dst_offset, idx);
        let v = b.load_sb_f32(src_buf, src_idx);
        b.store_sb_f32(dst_buf, dst_idx, v);
    });

    b.finish()
}

/// Validates an emitted SPIR-V 1.6 compute module structurally.
///
/// Checks section ordering, ID bounds, balanced labels/terminators, entry-point interface
/// completeness, and `OpSDot`/portable capability hygiene.
///
/// # Errors
/// Returns [`Error::InvalidInput`] describing the structural or capability violation.
#[expect(
    clippy::too_many_lines,
    reason = "Single-pass SPIR-V 1.6 structural validator checking headers, sections, IDs, blocks, and capabilities."
)]
pub fn validate_spirv_module(words: &[u32], expected_sdot: Option<bool>) -> Result<(), Error> {
    if words.len() < 5 {
        return Err(Error::InvalidInput(
            "SPIR-V module shorter than 5-word header".to_owned(),
        ));
    }
    if words[0] != SPIRV_MAGIC {
        return Err(Error::InvalidInput(format!(
            "invalid SPIR-V magic 0x{:08x}",
            words[0]
        )));
    }
    if words[1] != SPIRV_VERSION_1_6 {
        return Err(Error::InvalidInput(format!(
            "invalid SPIR-V version 0x{:08x}, expected 1.6 (0x{SPIRV_VERSION_1_6:08x})",
            words[1]
        )));
    }
    let bound = words[3];
    if bound <= 1 {
        return Err(Error::InvalidInput(format!(
            "invalid SPIR-V ID bound {bound}"
        )));
    }
    if words[4] != 0 {
        return Err(Error::InvalidInput(format!(
            "non-zero SPIR-V reserved schema word 0x{:08x}",
            words[4]
        )));
    }

    let check_id = |id: u32, ctx: &str| -> Result<(), Error> {
        if id == 0 || id >= bound {
            Err(Error::InvalidInput(format!(
                "SPIR-V ID {id} out of bound 1..{bound} in {ctx}"
            )))
        } else {
            Ok(())
        }
    };

    let mut offset = 5_usize;
    let mut max_section = 0_u8;
    let mut in_function = false;
    let mut in_block = false;
    let mut seen_first_block_non_var = false;
    let mut seen_merge = false;
    let mut function_depth = 0_usize;

    let mut result_ids = BTreeSet::new();
    let mut capabilities = BTreeSet::new();
    let mut entry_interface = Vec::new();
    let mut global_vars = Vec::new();
    let mut function_vars = Vec::new();
    let mut func_labels = BTreeSet::new();
    let mut func_branch_targets = Vec::new();
    let mut func_phi_preds = Vec::new();
    let mut has_memory_model = false;
    let mut has_entry_point = false;
    let mut has_exec_mode = false;
    let mut has_sdot = false;
    let mut last_opcode = 0_u16;

    while offset < words.len() {
        let header = words[offset];
        let wc = usize::try_from(header >> 16).unwrap_or(0);
        let op = u16::try_from(header & 0xFFFF).unwrap_or(0);
        if wc == 0 || offset + wc > words.len() {
            return Err(Error::InvalidInput(format!(
                "invalid SPIR-V instruction word count {wc} at offset {offset}"
            )));
        }
        let ops = &words[offset + 1..offset + wc];
        last_opcode = op;

        let section = match op {
            OP_CAPABILITY => 1,
            OP_EXTENSION => {
                return Err(Error::InvalidInput(
                    "SPIR-V OpExtension is forbidden in core 1.6 modules".to_owned(),
                ));
            }
            OP_EXT_INST_IMPORT => 3,
            OP_MEMORY_MODEL => 4,
            OP_ENTRY_POINT => 5,
            OP_EXECUTION_MODE => 6,
            OP_DECORATE | OP_MEMBER_DECORATE => 7,
            OP_TYPE_VOID..=OP_TYPE_FUNCTION | OP_CONSTANT => 8,
            OP_VARIABLE if !in_function => 8,
            _ => 9,
        };
        if section < max_section {
            return Err(Error::InvalidInput(format!(
                "SPIR-V section order violation: opcode {op} (section {section}) after section {max_section}"
            )));
        }
        max_section = section;

        let mut register_result = |id: u32, ctx: &str| -> Result<(), Error> {
            check_id(id, ctx)?;
            if !result_ids.insert(id) {
                return Err(Error::InvalidInput(format!(
                    "duplicate SPIR-V result ID {id} in {ctx}"
                )));
            }
            Ok(())
        };

        if in_function {
            if op == OP_FUNCTION_END {
                if in_block {
                    return Err(Error::InvalidInput(
                        "unterminated SPIR-V basic block before OpFunctionEnd".to_owned(),
                    ));
                }
                for tgt in &func_branch_targets {
                    if !func_labels.contains(tgt) {
                        return Err(Error::InvalidInput(format!(
                            "SPIR-V branch/merge references unknown label {tgt}"
                        )));
                    }
                }
                func_branch_targets.clear();
                for pred in &func_phi_preds {
                    if !func_labels.contains(pred) {
                        return Err(Error::InvalidInput(format!(
                            "SPIR-V OpPhi references unknown predecessor label {pred}"
                        )));
                    }
                }
                func_phi_preds.clear();
                func_labels.clear();
                in_function = false;
                function_depth += 1;
                offset += wc;
                continue;
            }
            if op == OP_LABEL {
                if in_block || seen_merge {
                    return Err(Error::InvalidInput(
                        "SPIR-V OpLabel encountered inside unterminated block".to_owned(),
                    ));
                }
                let lbl = *ops
                    .first()
                    .ok_or_else(|| Error::InvalidInput("OpLabel missing ID".to_owned()))?;
                register_result(lbl, "OpLabel")?;
                func_labels.insert(lbl);
                in_block = true;
                offset += wc;
                continue;
            }
            if !in_block {
                return Err(Error::InvalidInput(format!(
                    "SPIR-V instruction {op} outside basic block inside function"
                )));
            }
            if seen_merge && !matches!(op, OP_BRANCH_CONDITIONAL | OP_BRANCH) {
                return Err(Error::InvalidInput(format!(
                    "SPIR-V merge instruction must immediately precede branch, got {op}"
                )));
            }
            if op != OP_VARIABLE {
                seen_first_block_non_var = true;
            } else if seen_first_block_non_var {
                return Err(Error::InvalidInput(
                    "function-scoped OpVariable must appear in entry block before other ops"
                        .to_owned(),
                ));
            }
        }

        match op {
            OP_CAPABILITY => {
                let cap = *ops.first().ok_or_else(|| {
                    Error::InvalidInput("OpCapability missing operand".to_owned())
                })?;
                if cap == CAPABILITY_FLOAT64 || cap == CAPABILITY_INT64 {
                    return Err(Error::InvalidInput(format!(
                        "forbidden non-portable SPIR-V capability {cap}"
                    )));
                }
                capabilities.insert(cap);
            }
            OP_EXT_INST_IMPORT => {
                let res = *ops.first().ok_or_else(|| {
                    Error::InvalidInput("OpExtInstImport missing result".to_owned())
                })?;
                register_result(res, "OpExtInstImport")?;
            }
            OP_MEMORY_MODEL => {
                has_memory_model = true;
                if ops != [ADDRESSING_LOGICAL, MEMORY_MODEL_GLSL450] {
                    return Err(Error::InvalidInput(
                        "unexpected SPIR-V OpMemoryModel addressing/memory model".to_owned(),
                    ));
                }
            }
            OP_ENTRY_POINT => {
                has_entry_point = true;
                if ops.len() < 3 || ops[0] != EXEC_MODEL_GL_COMPUTE {
                    return Err(Error::InvalidInput(
                        "invalid OpEntryPoint execution model or operands".to_owned(),
                    ));
                }
                check_id(ops[1], "OpEntryPoint fn")?;
                let mut idx = 2_usize;
                while idx < ops.len() {
                    let word = ops[idx];
                    idx += 1;
                    if word.to_le_bytes().contains(&0) {
                        break;
                    }
                }
                for &iface_id in &ops[idx..] {
                    check_id(iface_id, "OpEntryPoint interface")?;
                    entry_interface.push(iface_id);
                }
            }
            OP_EXECUTION_MODE => {
                has_exec_mode = true;
                let fn_id = *ops
                    .first()
                    .ok_or_else(|| Error::InvalidInput("OpExecutionMode missing fn".to_owned()))?;
                check_id(fn_id, "OpExecutionMode")?;
                if ops.get(1) == Some(&EXEC_MODE_LOCAL_SIZE)
                    && (ops.len() < 5 || ops[2] == 0 || ops[3] == 0 || ops[4] == 0)
                {
                    return Err(Error::InvalidInput(
                        "invalid LocalSize execution mode extents".to_owned(),
                    ));
                }
            }
            OP_DECORATE | OP_MEMBER_DECORATE => {
                let target = *ops
                    .first()
                    .ok_or_else(|| Error::InvalidInput("OpDecorate missing target".to_owned()))?;
                check_id(target, "OpDecorate")?;
            }
            OP_TYPE_VOID | OP_TYPE_BOOL | OP_TYPE_INT | OP_TYPE_FLOAT => {
                let res = *ops
                    .first()
                    .ok_or_else(|| Error::InvalidInput("type opcode missing result".to_owned()))?;
                register_result(res, "primitive type")?;
            }
            OP_TYPE_VECTOR => {
                if ops.len() < 3 {
                    return Err(Error::InvalidInput("OpTypeVector truncated".to_owned()));
                }
                register_result(ops[0], "OpTypeVector")?;
                check_id(ops[1], "OpTypeVector element")?;
            }
            OP_TYPE_ARRAY | OP_TYPE_RUNTIME_ARRAY | OP_TYPE_STRUCT | OP_TYPE_FUNCTION => {
                let res = *ops
                    .first()
                    .ok_or_else(|| Error::InvalidInput("composite type missing id".to_owned()))?;
                register_result(res, "composite type")?;
                for &id in &ops[1..] {
                    check_id(id, "composite type operand")?;
                }
            }
            OP_TYPE_POINTER => {
                if ops.len() < 3 {
                    return Err(Error::InvalidInput("OpTypePointer truncated".to_owned()));
                }
                register_result(ops[0], "OpTypePointer")?;
                check_id(ops[2], "OpTypePointer pointee")?;
            }
            OP_CONSTANT => {
                if ops.len() < 3 {
                    return Err(Error::InvalidInput("OpConstant truncated".to_owned()));
                }
                check_id(ops[0], "OpConstant type")?;
                register_result(ops[1], "OpConstant")?;
            }
            OP_VARIABLE => {
                if ops.len() < 3 {
                    return Err(Error::InvalidInput("OpVariable truncated".to_owned()));
                }
                check_id(ops[0], "OpVariable type")?;
                register_result(ops[1], "OpVariable")?;
                if ops[2] == STORAGE_FUNCTION {
                    if !in_function {
                        return Err(Error::InvalidInput(
                            "Function-scoped OpVariable outside function".to_owned(),
                        ));
                    }
                    function_vars.push(ops[1]);
                } else {
                    if in_function {
                        return Err(Error::InvalidInput(
                            "non-Function OpVariable inside function".to_owned(),
                        ));
                    }
                    global_vars.push(ops[1]);
                }
                if let Some(&init_id) = ops.get(3) {
                    check_id(init_id, "OpVariable initializer")?;
                }
            }
            OP_FUNCTION => {
                if in_function || ops.len() < 4 {
                    return Err(Error::InvalidInput("malformed OpFunction".to_owned()));
                }
                check_id(ops[0], "OpFunction ret type")?;
                register_result(ops[1], "OpFunction")?;
                check_id(ops[3], "OpFunction type")?;
                in_function = true;
                in_block = false;
                seen_first_block_non_var = false;
                seen_merge = false;
            }
            OP_EXT_INST => {
                if ops.len() < 4 {
                    return Err(Error::InvalidInput("OpExtInst truncated".to_owned()));
                }
                check_id(ops[0], "OpExtInst type")?;
                register_result(ops[1], "OpExtInst")?;
                check_id(ops[2], "OpExtInst import")?;
                for &arg_id in &ops[4..] {
                    check_id(arg_id, "OpExtInst arg")?;
                }
            }
            OP_COMPOSITE_EXTRACT => {
                if ops.len() < 4 {
                    return Err(Error::InvalidInput(
                        "OpCompositeExtract truncated".to_owned(),
                    ));
                }
                check_id(ops[0], "OpCompositeExtract type")?;
                register_result(ops[1], "OpCompositeExtract")?;
                check_id(ops[2], "OpCompositeExtract composite")?;
            }
            OP_S_DOT => {
                if ops.len() < 5 {
                    return Err(Error::InvalidInput("OpSDot truncated".to_owned()));
                }
                has_sdot = true;
                check_id(ops[0], "OpSDot type")?;
                register_result(ops[1], "OpSDot")?;
                check_id(ops[2], "OpSDot vec1")?;
                check_id(ops[3], "OpSDot vec2")?;
            }
            OP_STORE => {
                if ops.len() < 2 {
                    return Err(Error::InvalidInput("OpStore truncated".to_owned()));
                }
                check_id(ops[0], "OpStore ptr")?;
                check_id(ops[1], "OpStore val")?;
            }
            OP_CONTROL_BARRIER => {
                if ops.len() != 3 {
                    return Err(Error::InvalidInput("OpControlBarrier arity".to_owned()));
                }
                for &id in ops {
                    check_id(id, "OpControlBarrier")?;
                }
            }
            OP_SELECTION_MERGE => {
                if ops.len() < 2 {
                    return Err(Error::InvalidInput("OpSelectionMerge truncated".to_owned()));
                }
                check_id(ops[0], "OpSelectionMerge target")?;
                func_branch_targets.push(ops[0]);
                seen_merge = true;
            }
            OP_LOOP_MERGE => {
                if ops.len() < 3 {
                    return Err(Error::InvalidInput("OpLoopMerge truncated".to_owned()));
                }
                check_id(ops[0], "OpLoopMerge merge")?;
                check_id(ops[1], "OpLoopMerge continue")?;
                func_branch_targets.push(ops[0]);
                func_branch_targets.push(ops[1]);
                seen_merge = true;
            }
            OP_BRANCH => {
                if ops.len() != 1 {
                    return Err(Error::InvalidInput("OpBranch arity".to_owned()));
                }
                check_id(ops[0], "OpBranch target")?;
                func_branch_targets.push(ops[0]);
                in_block = false;
                seen_merge = false;
            }
            OP_BRANCH_CONDITIONAL => {
                if ops.len() < 3 {
                    return Err(Error::InvalidInput(
                        "OpBranchConditional truncated".to_owned(),
                    ));
                }
                check_id(ops[0], "OpBranchConditional cond")?;
                check_id(ops[1], "OpBranchConditional true")?;
                check_id(ops[2], "OpBranchConditional false")?;
                func_branch_targets.push(ops[1]);
                func_branch_targets.push(ops[2]);
                in_block = false;
                seen_merge = false;
            }
            OP_RETURN => {
                if !ops.is_empty() {
                    return Err(Error::InvalidInput(
                        "OpReturn must have no operands".to_owned(),
                    ));
                }
                in_block = false;
                seen_merge = false;
            }
            OP_PHI => {
                if ops.len() < 4 || !ops.len().is_multiple_of(2) {
                    return Err(Error::InvalidInput("OpPhi arity invalid".to_owned()));
                }
                check_id(ops[0], "OpPhi type")?;
                register_result(ops[1], "OpPhi")?;
                for pair in ops[2..].as_chunks::<2>().0 {
                    check_id(pair[0], "OpPhi value")?;
                    check_id(pair[1], "OpPhi label")?;
                    func_phi_preds.push(pair[1]);
                }
            }
            _ => {
                if ops.len() < 2 {
                    return Err(Error::InvalidInput(format!(
                        "unexpected SPIR-V instruction opcode {op} with {} operands",
                        ops.len()
                    )));
                }
                check_id(ops[0], "result type")?;
                register_result(ops[1], "instruction result")?;
                for &id in &ops[2..] {
                    check_id(id, "instruction operand")?;
                }
            }
        }

        offset += wc;
    }

    if in_function || in_block || function_depth == 0 || last_opcode != OP_FUNCTION_END {
        return Err(Error::InvalidInput(
            "SPIR-V module has unbalanced function/block termination".to_owned(),
        ));
    }
    if !(has_memory_model && has_entry_point && has_exec_mode) {
        return Err(Error::InvalidInput(
            "SPIR-V module missing required MemoryModel, EntryPoint, or ExecutionMode".to_owned(),
        ));
    }
    if !capabilities.contains(&CAPABILITY_SHADER) {
        return Err(Error::InvalidInput(
            "SPIR-V module missing Shader capability".to_owned(),
        ));
    }

    let has_sdot_caps = capabilities.contains(&CAPABILITY_DOT_PRODUCT)
        && capabilities.contains(&CAPABILITY_DOT_PRODUCT_INPUT_4X8_BIT_PACKED);
    let any_sdot_cap = capabilities.contains(&CAPABILITY_DOT_PRODUCT)
        || capabilities.contains(&CAPABILITY_DOT_PRODUCT_INPUT_4X8_BIT_PACKED);
    if has_sdot && !has_sdot_caps {
        return Err(Error::InvalidInput(
            "SPIR-V module emits OpSDot without required DotProduct capabilities".to_owned(),
        ));
    }
    if any_sdot_cap && !has_sdot_caps {
        return Err(Error::InvalidInput(
            "SPIR-V module declares incomplete DotProduct capabilities".to_owned(),
        ));
    }
    match expected_sdot {
        Some(true) if !(has_sdot && has_sdot_caps) => {
            return Err(Error::InvalidInput(
                "expected OpSDot module but OpSDot or DotProduct capability is missing".to_owned(),
            ));
        }
        Some(false) if has_sdot || any_sdot_cap => {
            return Err(Error::InvalidInput(
                "portable SPIR-V module must not emit OpSDot or DotProduct capabilities".to_owned(),
            ));
        }
        _ => {}
    }

    let unique_iface: BTreeSet<u32> = entry_interface.iter().copied().collect();
    if unique_iface.len() != entry_interface.len() {
        return Err(Error::InvalidInput(
            "duplicate IDs in SPIR-V OpEntryPoint interface".to_owned(),
        ));
    }
    for var_id in &global_vars {
        if !unique_iface.contains(var_id) {
            return Err(Error::InvalidInput(format!(
                "missing global variable {var_id} from SPIR-V 1.6 OpEntryPoint interface"
            )));
        }
    }
    for var_id in &function_vars {
        if unique_iface.contains(var_id) {
            return Err(Error::InvalidInput(format!(
                "function-scoped variable {var_id} must not appear in OpEntryPoint interface"
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_valid_spirv_structure(words: &[u32], expected_sdot: Option<bool>) {
        assert_eq!(
            validate_spirv_module(words, expected_sdot),
            Ok(()),
            "SPIR-V validation failed"
        );
    }

    #[test]
    fn emitted_shaders_are_valid_spirv_and_deterministically_hashed() -> Result<(), Error> {
        let s1 = pq2_q8k_gemv(true, true);
        let s2 = pq2_q8k_gemv(true, true);
        let s3 = pq2_q8k_gemv(false, false);
        assert_eq!(s1, s2);
        assert_ne!(s1, s3);

        for (module, expected_sdot) in [
            (pq2_q8k_gemv(true, true), Some(true)),
            (pq2_q8k_gemv(true, false), Some(true)),
            (pq2_q8k_gemv(false, true), Some(false)),
            (pq2_q8k_gemv(false, false), Some(false)),
            (pq2_q8k_gemv_coalesced(true, true, false), Some(true)),
            (pq2_q8k_gemv_coalesced(true, false, true), Some(true)),
            (pq2_q8k_gemv_coalesced(false, true, false), Some(false)),
            (pq2_q8k_gemv_coalesced(false, false, true), Some(false)),
            (pq2_q8k_gemm_4tok_coalesced(true, true, false), Some(true)),
            (pq2_q8k_gemm_4tok_coalesced(true, false, true), Some(true)),
            (
                pq2_q8k_gemm_4tok_coalesced(false, false, false),
                Some(false),
            ),
            (pq2_q8_0_gemv(true), Some(true)),
            (pq2_q8_0_gemv(false), Some(false)),
            (pq2_q8_0_gemv_coalesced(true), Some(true)),
            (pq2_q8_0_gemv_coalesced(false), Some(false)),
            (pq2_q8_0_gemm_4tok_coalesced(true), Some(true)),
            (pq2_q8_0_gemm_4tok_coalesced(false), Some(false)),
            (ptq1_q8_0_gemv(true, true), Some(true)),
            (ptq1_q8_0_gemv(true, false), Some(true)),
            (ptq1_q8_0_gemv(false, true), Some(false)),
            (ptq1_q8_0_gemv(false, false), Some(false)),
            (ptq1_q8_0_gemv_coalesced(true, true), Some(true)),
            (ptq1_q8_0_gemv_coalesced(false, false), Some(false)),
            (ptq1_q8_0_gemm_4tok_coalesced(true, true), Some(true)),
            (ptq1_q8_0_gemm_4tok_coalesced(true, false), Some(true)),
            (ptq1_q8_0_gemm_4tok_coalesced(false, true), Some(false)),
            (ptq1_q8_0_gemm_4tok_coalesced(false, false), Some(false)),
            (ptq1_q8k_gemv(true), Some(true)),
            (ptq1_q8k_gemv(false), Some(false)),
            (ptq1_q8k_gemv_coalesced(true), Some(true)),
            (ptq1_q8k_gemv_coalesced(false), Some(false)),
        ] {
            assert_valid_spirv_structure(&module, expected_sdot);
            assert_eq!(shader_digest(&module)?.len(), 64);
        }

        // Validator rejects tampered magic, bad version, out-of-bounds IDs, and sdot mismatch.
        assert!(validate_spirv_module(&s3, Some(true)).is_err());
        assert!(validate_spirv_module(&s1, Some(false)).is_err());
        let mut bad_magic = s3.clone();
        bad_magic[0] = 0;
        assert!(validate_spirv_module(&bad_magic, None).is_err());
        let mut bad_bound = s3;
        bad_bound[3] = 2;
        assert!(validate_spirv_module(&bad_bound, None).is_err());
        Ok(())
    }

    #[test]
    fn emitted_layer_and_batched_shaders_are_valid_spirv() -> Result<(), Error> {
        for module in [
            quantize_f32_to_q8k_4plane(true),
            quantize_f32_to_q8k_4plane(false),
            quantize_f32_to_q8k_4plane_batched(true),
            quantize_f32_to_q8k_4plane_batched(false),
            quantize_f32_to_q8_0(false, false),
            quantize_f32_to_q8_0(true, false),
            quantize_f32_to_q8_0(true, true),
            quantize_f32_to_q8_0_batched(false, false),
            quantize_f32_to_q8_0_batched(true, false),
            quantize_f32_to_q8_0_batched(true, true),
            ffn_swiglu_f32(false),
            ffn_swiglu_f32(true),
            ffn_hadamard_f32(),
            ffn_hadamard_batched_f32(),
            dot_f32_scalar(),
            attn_qk_scores(),
            attn_qk_scores_f16kv(),
            attn_softmax_f16(),
            attn_pv_accum(),
            attn_pv_accum_f16kv(),
            attn_fused_f16prob(),
            attn_fused_f16prob_f16kv(),
            rms_norm_f32(),
            rms_norm_batched_f32(),
            add_rms_norm_f32(),
            add_rms_norm_batched_f32(),
            residual_add_f32(),
            residual_add_batched_f32(),
            rope_f32(),
            rope_batched_f32(),
            mul_sigmoid_f32(),
            conv1d_silu_f32(),
            conv1d_silu_offset_f32(),
            l2_norm_f32(),
            gdn_step_f32(),
            gdn_step_batched_token_f32(),
            gdn_gates_f32(),
            gdn_gates_batched_f32(),
            deinterleave_q_gate_f32(),
            deinterleave_q_gate_batched_f32(),
            round_f16_f32(),
            kv_append_f32(),
            kv_append_batched_f32(),
            kv_append_f16(),
            kv_append_batched_f16(),
            kv_unpack_f16_to_f32(),
            f32_matvec(),
            f32_gemm_batched(),
            lora_b_accum_f32(),
            lora_b_accum_batched_f32(),
            mul_f32_elemwise(),
            mul_f32_elemwise_batched(),
            permute_ssm_heads_f32(),
            permute_ssm_heads_batched_f32(),
            copy_slice_f32(),
            copy_slice_offset_f32(),
        ] {
            assert_valid_spirv_structure(&module, Some(false));
            assert_eq!(shader_digest(&module)?.len(), 64);
        }
        Ok(())
    }
}
