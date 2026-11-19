//! Pure-Rust SPIR-V 1.3 compute shader emitter for Qualcomm Adreno Vulkan kernels.
//!
//! Emits validated SPIR-V 1.3 binary modules without external shader compilers (`glslang`,
//! `naga`, or `shaderc`), covering both `#39` (`PQ2_0`/`PTQ1_0` integer matrix-vector
//! projections with optional `SPV_KHR_integer_dot_product` `OpSDot`) and `#40` (fused and
//! 3-stage decomposed attention with explicit IEEE-754 binary16 probability rounding).
#![forbid(unsafe_code)]

use std::collections::BTreeMap;

use bonsai_native_hash::Hasher;

use crate::{Error, ProjectionKernelPath};

const SPIRV_MAGIC: u32 = 0x0723_0203;
const SPIRV_VERSION_1_3: u32 = 0x0001_0300;
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
const OP_COMPOSITE_CONSTRUCT: u16 = 80;
const OP_COMPOSITE_EXTRACT: u16 = 81;
const OP_CONVERT_S_TO_F: u16 = 111;
const OP_BITCAST: u16 = 124;
const OP_I_ADD: u16 = 128;
const OP_F_ADD: u16 = 129;
const OP_I_SUB: u16 = 130;
const OP_F_SUB: u16 = 131;
const OP_I_MUL: u16 = 132;
const OP_F_MUL: u16 = 133;
const OP_U_DIV: u16 = 134;
const OP_F_DIV: u16 = 136;
const OP_LOGICAL_AND: u16 = 167;
const OP_SELECT: u16 = 169;
const OP_I_NOT_EQUAL: u16 = 171;
const OP_U_LESS_THAN: u16 = 176;
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
const CAPABILITY_DOT_PRODUCT_INPUT_4X8_BIT_PACKED: u32 = 6018;
const CAPABILITY_DOT_PRODUCT: u32 = 6019;

const GLSL_EXP: u32 = 27;
const GLSL_FMAX: u32 = 40;
const GLSL_FMA: u32 = 50;
const GLSL_PACK_HALF_2X16: u32 = 58;
const GLSL_UNPACK_HALF_2X16: u32 = 60;

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
    extensions: Vec<u32>,
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
    ptr_push_u32_t: Option<u32>,
    ptr_sb_u32_t: Option<u32>,
    ptr_sb_f32_t: Option<u32>,
    ptr_wg_f32_t: Option<u32>,

    rt_u32_t: Option<u32>,
    rt_f32_t: Option<u32>,
    ro_u32_buf_t: Option<u32>,
    ro_f32_buf_t: Option<u32>,
    rw_f32_buf_t: Option<u32>,

    u32_consts: BTreeMap<u32, u32>,
    i32_consts: BTreeMap<u32, u32>,
    f32_consts: BTreeMap<u32, u32>,

    main_fn_id: u32,
    current_label: u32,
    entry_inputs: Vec<u32>,
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
            extensions: Vec::new(),
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
            ptr_push_u32_t: None,
            ptr_sb_u32_t: None,
            ptr_sb_f32_t: None,
            ptr_wg_f32_t: None,
            rt_u32_t: None,
            rt_f32_t: None,
            ro_u32_buf_t: None,
            ro_f32_buf_t: None,
            rw_f32_buf_t: None,
            u32_consts: BTreeMap::new(),
            i32_consts: BTreeMap::new(),
            f32_consts: BTreeMap::new(),
            main_fn_id: 0,
            current_label: 0,
            entry_inputs: Vec::new(),
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
            let ext_words = encode_string("SPV_KHR_integer_dot_product");
            push_inst(&mut this.extensions, OP_EXTENSION, &ext_words);
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
        Some(var_id)
    }

    const fn alloc_id(&mut self) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    fn finish(mut self) -> Vec<u32> {
        let mut ep = vec![EXEC_MODEL_GL_COMPUTE, self.main_fn_id];
        ep.extend(encode_string("main"));
        ep.extend(self.entry_inputs.iter().copied());
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
                + self.extensions.len()
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
        words.push(SPIRV_VERSION_1_3);
        words.push(0);
        words.push(self.next_id);
        words.push(0);
        words.extend(self.capabilities);
        words.extend(self.extensions);
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
        self.entry_inputs.push(var);
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
        self.entry_inputs.push(var);
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
        self.entry_inputs.push(var);
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

    fn storage_buffer(&mut self, binding: u32, elem: ElemType, access: AccessMode) -> u32 {
        let rt_t = match elem {
            ElemType::U32 => {
                if let Some(id) = self.rt_u32_t {
                    id
                } else {
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
            }
            ElemType::F32 => {
                if let Some(id) = self.rt_f32_t {
                    id
                } else {
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
        };

        let struct_t = match (elem, access) {
            (ElemType::U32, AccessMode::ReadOnly) => {
                if let Some(id) = self.ro_u32_buf_t {
                    id
                } else {
                    let s = self.make_sb_struct(rt_t, true);
                    self.ro_u32_buf_t = Some(s);
                    s
                }
            }
            (ElemType::F32, AccessMode::ReadOnly) => {
                if let Some(id) = self.ro_f32_buf_t {
                    id
                } else {
                    let s = self.make_sb_struct(rt_t, true);
                    self.ro_f32_buf_t = Some(s);
                    s
                }
            }
            (ElemType::F32, AccessMode::ReadWrite) => {
                if let Some(id) = self.rw_f32_buf_t {
                    id
                } else {
                    let s = self.make_sb_struct(rt_t, false);
                    self.rw_f32_buf_t = Some(s);
                    s
                }
            }
            (ElemType::U32, AccessMode::ReadWrite) => self.make_sb_struct(rt_t, false),
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
        var
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

    fn u_ne(&mut self, a: u32, b: u32) -> u32 {
        self.binop(OP_I_NOT_EQUAL, self.bool_t, a, b)
    }

    fn logical_and(&mut self, a: u32, b: u32) -> u32 {
        self.binop(OP_LOGICAL_AND, self.bool_t, a, b)
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
        id
    }

    fn unpack_half_low(&mut self, u32_val: u32) -> u32 {
        let vec2 = self.glsl1(self.vec2f32_t, GLSL_UNPACK_HALF_2X16, u32_val);
        let out = self.alloc_id();
        push_inst(
            &mut self.fn_body,
            OP_COMPOSITE_EXTRACT,
            &[self.f32_t, out, vec2, 0],
        );
        out
    }

    fn round_f16(&mut self, val: u32) -> u32 {
        let zero = self.const_f32(0.0);
        let pair = self.alloc_id();
        push_inst(
            &mut self.fn_body,
            OP_COMPOSITE_CONSTRUCT,
            &[self.vec2f32_t, pair, val, zero],
        );
        let packed = self.glsl1(self.u32_t, GLSL_PACK_HALF_2X16, pair);
        self.unpack_half_low(packed)
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

fn portable_dot4_from_2bit_byte(b: &mut ShaderBuilder, raw_byte: u32, act_u32: u32) -> u32 {
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

    let w0 = b.isub(d0_i, i_one);
    let w1 = b.isub(d1_i, i_one);
    let w2 = b.isub(d2_i, i_one);
    let w3 = b.isub(d3_i, i_one);

    portable_dot4_i32_weights(b, [w0, w1, w2, w3], act_u32)
}

fn portable_dot4_from_packed_trits(b: &mut ShaderBuilder, lanes: u32, act_u32: u32) -> u32 {
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

    let w0 = b.isub(d0_i, i_one);
    let w1 = b.isub(d1_i, i_one);
    let w2 = b.isub(d2_i, i_one);
    let w3 = b.isub(d3_i, i_one);

    portable_dot4_i32_weights(b, [w0, w1, w2, w3], act_u32)
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

/// Emits `#39` `PQ2_0` × `Q8_K` GEMV SPIR-V 1.3 words.
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

/// Emits `#39` `PQ2_0` × `Q8_0` GEMV SPIR-V 1.3 words (supports odd `128`-column multiples).
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
        let g_c = b.const_u32(g);
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
        let g2_c = b.const_u32(4 + g2);
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

    let c6 = b.const_u32(6);
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

/// Emits `#39` `PTQ1_0` × `Q8_0` GEMV SPIR-V 1.3 words (`pinned_4lane` or standard).
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

/// Emits `#39` `PTQ1_0` × `Q8_K` direct GEMV SPIR-V 1.3 words (supports odd `128`-col multiples).
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

/// Emits `#40` Stage 1 (`QK^T` scaled causal scores) SPIR-V 1.3 words.
///
/// Bindings: `0` read-only `Q` (`f32`), `1` read-only `K` (`f32`), `2` writable `scores` (`f32`).
/// Push constants (`8 × u32`): `[query_len, total_seq, past_len, num_heads, num_kv_heads, head_dim, scale_bits, causal]`.
/// Dispatch: `(ceil(total_seq / 64), num_heads, query_len)`.
#[must_use]
pub fn attn_qk_scores() -> Vec<u32> {
    let (mut b, push_var) = ShaderBuilder::new(false, 8);
    let push_var = push_var.unwrap_or(0);
    let q_buf = b.storage_buffer(0, ElemType::F32, AccessMode::ReadOnly);
    let k_buf = b.storage_buffer(1, ElemType::F32, AccessMode::ReadOnly);
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

                let acc_var = b.local_f32_var();
                b.store_var(acc_var, f_zero);

                let c0 = b.const_u32(0);
                let c1 = b.const_u32(1);
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

                let dot = b.load_var_f32(acc_var);
                let scaled = b.fmul_nc(scale, dot);
                b.store_sb_f32(s_buf, s_idx, scaled);
            },
            |b| {
                b.store_sb_f32(s_buf, s_idx, f_zero);
            },
        );
    });

    b.finish()
}

/// Emits `#40` Stage 2 (global row softmax with IEEE binary16 rounding) SPIR-V 1.3 words.
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

/// Emits `#40` Stage 3 (`probs × V` accumulation) SPIR-V 1.3 words.
///
/// Bindings: `0` read-only `probs` (`f32`), `1` read-only `V` (`f32`), `2` writable `out` (`f32`).
/// Push constants (`8 × u32`): `[query_len, total_seq, past_len, num_heads, num_kv_heads, head_dim, scale_bits, causal]`.
/// Dispatch: `(ceil(head_dim / 64), num_heads, query_len)`.
#[must_use]
pub fn attn_pv_accum() -> Vec<u32> {
    let (mut b, push_var) = ShaderBuilder::new(false, 8);
    let push_var = push_var.unwrap_or(0);
    let p_buf = b.storage_buffer(0, ElemType::F32, AccessMode::ReadOnly);
    let v_buf = b.storage_buffer(1, ElemType::F32, AccessMode::ReadOnly);
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
            let vv = b.load_sb_f32(v_buf, v_idx);
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
    let (mut b, push_var) = ShaderBuilder::new(false, 8);
    let push_var = push_var.unwrap_or(0);
    let q_buf = b.storage_buffer(0, ElemType::F32, AccessMode::ReadOnly);
    let k_buf = b.storage_buffer(1, ElemType::F32, AccessMode::ReadOnly);
    let v_buf = b.storage_buffer(2, ElemType::F32, AccessMode::ReadOnly);
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
        let s = emit_dot_qk_row(b, q_buf, q_base, k_buf, k_base, head_dim, scale);
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
            let vv = b.load_sb_f32(v_buf, v_idx);
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

fn emit_dot_qk_row(
    b: &mut ShaderBuilder,
    q_buf: u32,
    q_base: u32,
    k_buf: u32,
    k_base: u32,
    head_dim: u32,
    scale: u32,
) -> u32 {
    let c0 = b.const_u32(0);
    let c1 = b.const_u32(1);
    let f_zero = b.const_f32(0.0);
    let acc_var = b.local_f32_var();
    b.store_var(acc_var, f_zero);
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
    let acc_v = b.load_var_f32(acc_var);
    b.fmul_nc(scale, acc_v)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_valid_spirv_structure(words: &[u32]) {
        assert!(words.len() >= 5);
        assert_eq!(words[0], SPIRV_MAGIC);
        assert_eq!(words[1], SPIRV_VERSION_1_3);
        let bound = words[3];
        assert!(bound > 1);
        assert_eq!(words[4], 0);

        let mut offset = 5;
        let mut opcodes = Vec::new();
        while offset < words.len() {
            let header = words[offset];
            let wc = usize::try_from(header >> 16).unwrap_or(0);
            let op = u16::try_from(header & 0xFFFF).unwrap_or(0);
            assert!(wc >= 1, "zero word count at word {offset}");
            assert!(
                offset + wc <= words.len(),
                "truncated instruction at word {offset}"
            );
            opcodes.push(op);
            offset += wc;
        }
        assert_eq!(offset, words.len());
        assert_eq!(opcodes.last().copied(), Some(OP_FUNCTION_END));
        assert!(opcodes.contains(&OP_MEMORY_MODEL));
        assert!(opcodes.contains(&OP_ENTRY_POINT));
        assert!(opcodes.contains(&OP_EXECUTION_MODE));
    }

    #[test]
    fn emitted_shaders_are_valid_spirv_and_deterministically_hashed() -> Result<(), Error> {
        let s1 = pq2_q8k_gemv(true, true);
        let s2 = pq2_q8k_gemv(true, true);
        let s3 = pq2_q8k_gemv(false, false);
        assert_eq!(s1, s2);
        assert_ne!(s1, s3);
        for module in [
            pq2_q8k_gemv(true, true),
            pq2_q8k_gemv(true, false),
            pq2_q8k_gemv(false, true),
            pq2_q8k_gemv(false, false),
            pq2_q8_0_gemv(true),
            pq2_q8_0_gemv(false),
            ptq1_q8_0_gemv(true, true),
            ptq1_q8_0_gemv(true, false),
            ptq1_q8_0_gemv(false, true),
            ptq1_q8_0_gemv(false, false),
            ptq1_q8k_gemv(true),
            ptq1_q8k_gemv(false),
            attn_qk_scores(),
            attn_softmax_f16(),
            attn_pv_accum(),
            attn_fused_f16prob(),
        ] {
            assert_valid_spirv_structure(&module);
            let digest = shader_digest(&module)?;
            assert_eq!(digest.len(), 64);
        }
        Ok(())
    }
}
