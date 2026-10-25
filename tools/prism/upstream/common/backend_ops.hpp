// Typed production GGML calls and bounded binary serialization. Workloads,
// generators, references, comparisons and benchmark schedules belong to Rust.
#pragma once
#include "action_file.hpp"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-impl.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <type_traits>
#include <vector>

namespace prism_backend_ops {
using json = nlohmann::ordered_json;
inline void require(bool value, const char * message) { if (!value) throw std::invalid_argument(message); }
inline void keys(const json & o, std::initializer_list<const char *> names) {
    require(o.is_object() && o.size() == names.size(), "backend object fields differ");
    for (const auto * name : names) require(o.contains(name), "backend object field missing");
}
inline std::uint64_t u64(const json & v) {
    require(v.is_number_integer(), "backend unsigned integer required");
    require(v.is_number_unsigned() || v.get<std::int64_t>() >= 0, "backend unsigned integer required");
    return v.get<std::uint64_t>();
}
inline std::int64_t i64(const json & v) {
    require(v.is_number_integer(), "backend integer required");
    if (v.is_number_unsigned()) require(v.get<std::uint64_t>() <= std::uint64_t(INT64_MAX), "backend signed integer overflow");
    return v.get<std::int64_t>();
}
inline std::size_t bounded(const json & v, std::size_t low, std::size_t high) {
    const auto n = u64(v); require(n >= low && n <= high, "backend integer outside bound"); return std::size_t(n);
}
inline std::uint32_t id(const json & v) { return std::uint32_t(bounded(v, 0, UINT32_MAX)); }
inline std::string text(const json & v) {
    require(v.is_string(), "backend string required"); auto s = v.get<std::string>();
    require(s.size() <= 65536 && s.find('\0') == std::string::npos, "backend string outside bound"); return s;
}
inline std::vector<std::uint8_t> unhex(const json & v) {
    const auto s = text(v); require(s.size() % 2 == 0, "backend hex length differs");
    std::vector<std::uint8_t> bytes(s.size()/2);
    const auto digit = [](char c) { if(c >= '0' && c <= '9') return c-'0'; if(c >= 'a' && c <= 'f') return c-'a'+10; throw std::invalid_argument("backend lowercase hex required"); };
    for(std::size_t i=0;i<bytes.size();++i) bytes[i]=std::uint8_t(digit(s[2*i])*16+digit(s[2*i+1])); return bytes;
}
inline std::string hex(const void * data, std::size_t bytes) {
    const auto * p = static_cast<const std::uint8_t *>(data); std::string out(bytes*2, '0');
    for(std::size_t i=0;i<bytes;++i) { out[2*i]="0123456789abcdef"[p[i]>>4]; out[2*i+1]="0123456789abcdef"[p[i]&15]; } return out;
}
inline bool handles(const std::string & kind) { return kind == "backend-graph"; }
inline void validate(const json & o) {
    if(o.contains("ActionsFile")) keys(o,{"Kind","Directory","MaxFileBytes","ActionsFile"});
    else keys(o,{"Kind","Directory","MaxFileBytes","Actions"});
    require(std::filesystem::path(text(o.at("Directory"))).is_absolute(), "backend Directory must be absolute");
    bounded(o.at("MaxFileBytes"), 1, std::size_t(4)*1024*1024*1024);
    if(o.contains("ActionsFile")) require(std::filesystem::path(text(o.at("ActionsFile"))).is_absolute(),"backend ActionsFile must be absolute");
    else {
        require(o.at("Actions").is_array() && !o.at("Actions").empty() && o.at("Actions").size() <= 65536, "backend action count outside bound");
        for(const auto & a:o.at("Actions")) require(a.is_object() && a.contains("Op") && a.at("Op").is_string(), "backend action Op required");
    }
}

struct argument {
    json value;
    std::vector<std::int64_t> integers64;
    std::vector<std::int32_t> integers32;
    std::vector<float> floats;
    std::vector<ggml_tensor *> tensors;
    std::string string;
};
struct state {
    std::filesystem::path directory;
    std::size_t max_file_bytes;
    std::map<std::uint32_t, ggml_context *> contexts;
    std::map<std::uint32_t, ggml_backend_t> backends;
    std::map<std::uint32_t, ggml_backend_buffer_t> buffers;
    std::map<std::uint32_t, ggml_tensor *> tensors;
    std::map<std::uint32_t, ggml_cgraph *> graphs;
    std::map<std::uint32_t, json> scalars;
    std::map<ggml_tensor *, std::uint32_t> tensor_ids;
    std::map<ggml_tensor *, std::set<std::uint32_t>> tensor_aliases;
    std::uint32_t automatic_id = UINT32_C(0x80000000);
    std::size_t context_bytes = 0;
    explicit state(const json & o): directory(std::filesystem::canonical(text(o.at("Directory")))), max_file_bytes(std::size_t(u64(o.at("MaxFileBytes")))) {
        require(std::filesystem::is_directory(directory), "backend Directory missing");
        // A partition reuses this process, but every geometry still owns fresh
        // graph/backend state. Runtime discovery and immutable CPU tables are
        // process resources: do not rescan backend DLL directories per case.
        static const bool initialized = [] {
            ggml_backend_load_all(); ggml_cpu_init(); ggml_time_init(); return true;
        }();
        (void) initialized;
    }
    ~state() {
        for(const auto & b:buffers) ggml_backend_buffer_free(b.second);
        for(const auto & b:backends) ggml_backend_free(b.second);
        for(const auto & c:contexts) ggml_free(c.second);
    }
    template<class T> static T lookup(const std::map<std::uint32_t,T> & values, const json & key) {
        const auto found = values.find(id(key)); require(found != values.end(), "backend unknown handle"); return found->second;
    }
    template<class T> static void bind(std::map<std::uint32_t,T> & values, const json & key, T value) {
        const auto n=id(key); require(n < UINT32_C(0x80000000) && values.size() < 32768 && values.emplace(n,value).second, "backend duplicate or excessive handle");
    }
    json tensor_id(ggml_tensor * t) {
        if(!t) return nullptr;
        auto found=tensor_ids.find(t); if(found != tensor_ids.end()) return found->second;
        require(tensors.size()<32768 && automatic_id != UINT32_MAX,"backend automatic handle bound");
        const auto n=automatic_id++; tensor_ids.emplace(t,n); tensors.emplace(n,t); tensor_aliases[t].insert(n); return n;
    }
    void bind_tensor(const json & n, ggml_tensor * t) {
        // Null is a valid raw observation for get_grad/get_tensor calls.
        bind(tensors,n,t); if(t) { tensor_ids.emplace(t,id(n)); tensor_aliases[t].insert(id(n)); }
    }
    std::filesystem::path file(const json & v) const {
        auto path=std::filesystem::path(text(v)); require(path.is_absolute(),"backend file must be absolute");
        path=std::filesystem::weakly_canonical(path);
        auto relative=path.lexically_relative(directory);
        require(!relative.empty() && !relative.is_absolute(),"backend file outside Directory");
        for(const auto & component:relative) require(component != "..", "backend file outside Directory");
        return path;
    }
    static bool exists(const std::filesystem::path & path) { return std::filesystem::exists(path); }
    std::vector<std::uint8_t> read(const json & v, std::size_t offset, std::size_t bytes) const {
        const auto p=file(v); require(std::filesystem::is_regular_file(p),"backend input must be regular file");
        const auto size=std::filesystem::file_size(p);
        require(size <= max_file_bytes && offset <= size && bytes <= size-offset,"backend file read extent differs");
        std::ifstream f(p,std::ios::binary); require(bool(f),"backend input open failed"); f.seekg(std::streamoff(offset));
        std::vector<std::uint8_t> data(bytes); if(bytes) f.read(reinterpret_cast<char *>(data.data()),std::streamsize(bytes)); require(bool(f),"backend file read failed"); return data;
    }
    std::vector<std::uint8_t> read_all(const json & v) const {
        const auto size=std::filesystem::file_size(file(v)); require(size<=max_file_bytes,"backend input file exceeds cap"); return read(v,0,std::size_t(size));
    }
    void write(const json & v, const void * data, std::size_t bytes, std::size_t offset=0, bool truncate=true) const {
        require(offset<=max_file_bytes && bytes<=max_file_bytes-offset,"backend file write exceeds cap"); const auto p=file(v);
        std::fstream f;
        if(truncate) f.open(p,std::ios::binary|std::ios::out|std::ios::trunc);
        else { if(!exists(p)) { std::ofstream create(p,std::ios::binary); require(bool(create),"backend output create failed"); } f.open(p,std::ios::binary|std::ios::in|std::ios::out); }
        require(bool(f),"backend output open failed"); f.seekp(std::streamoff(offset)); if(bytes) f.write(static_cast<const char *>(data),std::streamsize(bytes)); f.flush(); require(bool(f),"backend output write failed");
    }
    json metadata(ggml_tensor * t) {
        require(t,"backend null tensor metadata");
        json ne=json::array(),nb=json::array(),src=json::array(),aliases=json::array();
        for(int i=0;i<GGML_MAX_DIMS;++i) { ne.push_back(t->ne[i]); nb.push_back(t->nb[i]); }
        for(int i=0;i<GGML_MAX_SRC;++i) src.push_back(tensor_id(t->src[i]));
        const auto canonical=tensor_id(t);
        // Keep the original numeric handle order without scanning all bound
        // tensors for every tensor in a context snapshot.
        for(const auto alias:tensor_aliases.at(t)) aliases.push_back(alias);
        return {{"Id",canonical},{"Aliases",aliases},{"Type",int(t->type)},{"NE",ne},{"NB",nb},{"Op",int(t->op)},{"Flags",t->flags},{"Name",t->name},
            {"ViewSource",tensor_id(t->view_src)},{"ViewOffset",t->view_offs},{"Sources",src},{"OpParamsHex",hex(t->op_params,sizeof(t->op_params))},
            {"NBytes",ggml_nbytes(t)},{"NElements",ggml_nelements(t)},{"NRows",ggml_nrows(t)},{"BlockSize",ggml_blck_size(t->type)},{"TypeSize",ggml_type_size(t->type)},
            {"OpDescription",ggml_op_desc(t)},{"Contiguous",ggml_is_contiguous(t)},{"IsTransposed",ggml_is_transposed(t)}};
    }
    std::int64_t integer(const json & value, int depth=0) {
        require(depth<24,"backend argument nesting exceeds bound");
        require(value.is_object() && value.size()==1,"backend typed integer required");
        if(value.contains("I32")) { const auto n=i64(value.at("I32")); require(n>=INT32_MIN && n<=INT32_MAX,"backend I32 overflow"); return n; }
        if(value.contains("I64")) return i64(value.at("I64"));
        if(value.contains("U64")) { const auto n=u64(value.at("U64")); require(n<=std::uint64_t(INT64_MAX),"backend integer overflow"); return std::int64_t(n); }
        if(value.contains("Scalar")) return integer(lookup(scalars,value.at("Scalar")),depth+1);
        if(value.contains("TensorField")) {
            const auto & f=value.at("TensorField"); const auto name=text(f.at("Field")); auto * t=lookup(tensors,f.at("Tensor")); require(t,"backend null field tensor");
            if(name=="ne" || name=="nb") { keys(f,{"Tensor","Field","Index"}); const auto n=bounded(f.at("Index"),0,3); if(name=="ne") return t->ne[n]; require(t->nb[n]<=std::size_t(INT64_MAX),"backend stride overflow"); return std::int64_t(t->nb[n]); }
            keys(f,{"Tensor","Field"}); if(name=="type") return int(t->type); if(name=="op") return int(t->op); if(name=="flags") return t->flags;
            throw std::invalid_argument("backend unknown tensor field");
        }
        if(value.contains("Arithmetic")) {
            const auto & a=value.at("Arithmetic"); keys(a,{"Op","Arguments"}); const auto op=text(a.at("Op")); const auto & args=a.at("Arguments"); require(args.is_array() && args.size()==2,"backend binary arithmetic required");
            const auto x=integer(args[0],depth+1),y=integer(args[1],depth+1);
            if(op=="+") { require((y>=0 && x<=INT64_MAX-y)||(y<0 && x>=INT64_MIN-y),"backend addition overflow"); return x+y; }
            if(op=="-") { require((y>=0 && x>=INT64_MIN+y)||(y<0 && x<=INT64_MAX+y),"backend subtraction overflow"); return x-y; }
            if(op=="*") {
                if(x==0 || y==0) return 0;
                require(!((x==INT64_MIN && y==-1)||(y==INT64_MIN && x==-1)),"backend product overflow");
                if(x>0) require(y>0 ? x<=INT64_MAX/y : y>=INT64_MIN/x,"backend product overflow");
                else require(y>0 ? x>=INT64_MIN/y : x>=INT64_MAX/y,"backend product overflow");
                return x*y;
            }
            require((op=="/" || op=="%") && y!=0 && !(x==INT64_MIN && y==-1),"backend invalid division"); return op=="/" ? x/y : x%y;
        }
        throw std::invalid_argument("backend unknown typed integer");
    }
    template<class T> T convert(argument & a) {
        const auto & v=a.value;
        if constexpr(std::is_pointer_v<T>) {
            if(v.is_null()) return nullptr;
            using P=std::remove_cv_t<std::remove_pointer_t<T>>;
            if constexpr(std::is_same_v<P,ggml_context>) { keys(v,{"Context"}); return lookup(contexts,v.at("Context")); }
            else if constexpr(std::is_same_v<P,ggml_tensor>) { keys(v,{"Tensor"}); return lookup(tensors,v.at("Tensor")); }
            else if constexpr(std::is_same_v<P,ggml_cgraph>) { keys(v,{"Graph"}); return lookup(graphs,v.at("Graph")); }
            else if constexpr(std::is_same_v<T,ggml_backend_t>) { keys(v,{"Backend"}); return lookup(backends,v.at("Backend")); }
            else if constexpr(std::is_same_v<T,ggml_backend_buffer_t>) { keys(v,{"Buffer"}); return lookup(buffers,v.at("Buffer")); }
            else if constexpr(std::is_same_v<P,char>) { const auto value=v.contains("Scalar")?lookup(scalars,v.at("Scalar")):v; keys(value,{"String"}); if(value.at("String").is_null()) return nullptr; a.string=text(value.at("String")); return a.string.data(); }
            else if constexpr(std::is_same_v<P,std::int64_t>) { keys(v,{"I64Array"}); const auto & values=v.at("I64Array"); require(values.is_array() && values.size()<=32768,"backend I64 array exceeds bound"); for(const auto & n:values) a.integers64.push_back(n.is_object()?integer(n):i64(n)); return a.integers64.data(); }
            else if constexpr(std::is_same_v<P,std::int32_t>) { keys(v,{"I32Array"}); const auto & values=v.at("I32Array"); require(values.is_array() && values.size()<=32768,"backend I32 array exceeds bound"); for(const auto & n:values) { const auto x=n.is_object()?integer(n):i64(n); require(x>=INT32_MIN && x<=INT32_MAX,"backend I32 array overflow"); a.integers32.push_back(std::int32_t(x)); } return a.integers32.data(); }
            else if constexpr(std::is_same_v<P,float>) { keys(v,{"F32BitsArray"}); const auto & values=v.at("F32BitsArray"); require(values.is_array() && values.size()<=32768,"backend float array exceeds bound"); for(const auto & n:values) { const auto bits=std::uint32_t(bounded(n,0,UINT32_MAX)); float x; std::memcpy(&x,&bits,4); a.floats.push_back(x); } return a.floats.data(); }
            else if constexpr(std::is_same_v<P,ggml_tensor *>) { keys(v,{"TensorArray"}); const auto & values=v.at("TensorArray"); require(values.is_array() && values.size()<=32768,"backend tensor array exceeds bound"); for(const auto & n:values) a.tensors.push_back(n.is_null()?nullptr:lookup(tensors,n)); return a.tensors.data(); }
            else { static_assert(!std::is_same_v<T,T>,"unsupported production pointer argument"); }
        } else if constexpr(std::is_same_v<T,bool>) { if(v.contains("Scalar")) { const auto value=lookup(scalars,v.at("Scalar")); keys(value,{"Bool"}); return value.at("Bool").get<bool>(); } keys(v,{"Bool"}); require(v.at("Bool").is_boolean(),"backend Bool required"); return v.at("Bool").get<bool>(); }
        else if constexpr(std::is_same_v<T,float> || std::is_same_v<T,double>) {
            const auto value=v.contains("Scalar")?lookup(scalars,v.at("Scalar")):v;
            if constexpr(std::is_same_v<T,float>) { keys(value,{"F32Bits"}); const auto bits=std::uint32_t(bounded(value.at("F32Bits"),0,UINT32_MAX)); float x; std::memcpy(&x,&bits,4); return x; }
            else { keys(value,{"F64Bits"}); const auto bits=u64(value.at("F64Bits")); double x; std::memcpy(&x,&bits,8); return x; }
        } else if constexpr(std::is_enum_v<T>) { const auto n=integer(v); require(n>=INT32_MIN && n<=INT32_MAX,"backend enum overflow"); return static_cast<T>(n); }
        else if constexpr(std::is_integral_v<T>) { const auto n=integer(v); if constexpr(std::is_unsigned_v<T>) require(n>=0 && std::uint64_t(n)<=std::numeric_limits<T>::max(),"backend unsigned conversion overflow"); else require(n>=std::numeric_limits<T>::min() && n<=std::numeric_limits<T>::max(),"backend signed conversion overflow"); return static_cast<T>(n); }
        else { static_assert(!std::is_same_v<T,T>,"unsupported production argument type"); }
    }
    template<class T> void save(const json & result, T value) {
        if(result.is_null()) return;
        if constexpr(std::is_same_v<T,ggml_tensor *>) { keys(result,{"Tensor"}); bind_tensor(result.at("Tensor"),value); }
        else if constexpr(std::is_same_v<T,ggml_cgraph *>) { keys(result,{"Graph"}); require(value,"backend graph allocation failed"); bind(graphs,result.at("Graph"),value); }
        else if constexpr(std::is_same_v<T,const char *> || std::is_same_v<T,char *>) { keys(result,{"Scalar"}); bind(scalars,result.at("Scalar"),json{{"String",value?json(value):json(nullptr)}}); }
        else {
            keys(result,{"Scalar"}); json scalar;
            if constexpr(std::is_same_v<T,bool>) scalar={{"Bool",value}};
            else if constexpr(std::is_same_v<T,float>) { std::uint32_t bits; std::memcpy(&bits,&value,4); scalar={{"F32Bits",bits}}; }
            else if constexpr(std::is_same_v<T,double>) { std::uint64_t bits; std::memcpy(&bits,&value,8); scalar={{"F64Bits",bits}}; }
            else if constexpr(std::is_enum_v<T>) scalar={{"I32",int(value)}};
            else if constexpr(std::is_integral_v<T>) { if constexpr(std::is_signed_v<T>) scalar={{"I64",value}}; else scalar={{"U64",value}}; }
            else { static_assert(!std::is_same_v<T,T>,"unsupported production return type"); }
            bind(scalars,result.at("Scalar"),scalar);
        }
    }
    template<class R,class... A,std::size_t... I> void invoke(R(*fn)(A...),std::vector<argument> & args,const json & result,std::index_sequence<I...>) {
        require(args.size()==sizeof...(A),"backend call argument count differs");
        if constexpr(std::is_void_v<R>) { require(result.is_null(),"backend void call cannot bind result"); fn(convert<A>(args[I])...); }
        else { auto value=fn(convert<A>(args[I])...); save(result,value); }
    }
    template<class R,class... A> void invoke(R(*fn)(A...),const json & action) {
        const auto & values=action.at("Arguments"); require(values.is_array() && values.size()<=32,"backend call argument count exceeds bound");
        std::vector<argument> args; args.reserve(values.size()); for(const auto & v:values) { argument a; a.value=v; args.push_back(std::move(a)); }
        invoke(fn,args,action.at("Result"),std::index_sequence_for<A...>{});
    }
    void call(const json & action);
    json run(const json & action);
};

inline void state::call(const json & action) {
    keys(action,{"Op","Function","Arguments","Result"}); const auto name=text(action.at("Function"));
#define PRISM_BACKEND_CALL(fn) if(name == #fn) { invoke(&fn,action); return; }
    // Production declaration allowlist; generated only from the public headers.
    PRISM_BACKEND_CALL(ggml_acc)
    PRISM_BACKEND_CALL(ggml_add)
    PRISM_BACKEND_CALL(ggml_add_inplace)
    PRISM_BACKEND_CALL(ggml_add_id)
    PRISM_BACKEND_CALL(ggml_arange)
    PRISM_BACKEND_CALL(ggml_are_same_shape)
    PRISM_BACKEND_CALL(ggml_argmax)
    PRISM_BACKEND_CALL(ggml_argsort)
    PRISM_BACKEND_CALL(ggml_argsort_top_k)
    PRISM_BACKEND_CALL(ggml_backend_buffer_set_usage)
    PRISM_BACKEND_CALL(ggml_backend_graph_compute)
    PRISM_BACKEND_CALL(ggml_blck_size)
    PRISM_BACKEND_CALL(ggml_build_backward_expand)
    PRISM_BACKEND_CALL(ggml_build_forward_expand)
    PRISM_BACKEND_CALL(ggml_ceil)
    PRISM_BACKEND_CALL(ggml_clamp)
    PRISM_BACKEND_CALL(ggml_col2im_1d)
    PRISM_BACKEND_CALL(ggml_concat)
    PRISM_BACKEND_CALL(ggml_cont)
    PRISM_BACKEND_CALL(ggml_conv_2d_direct)
    PRISM_BACKEND_CALL(ggml_conv_2d_dw_direct)
    PRISM_BACKEND_CALL(ggml_conv_3d_direct)
    PRISM_BACKEND_CALL(ggml_conv_transpose_1d)
    PRISM_BACKEND_CALL(ggml_conv_transpose_2d_p0)
    PRISM_BACKEND_CALL(ggml_cos)
    PRISM_BACKEND_CALL(ggml_count_equal)
    PRISM_BACKEND_CALL(ggml_cpy)
    PRISM_BACKEND_CALL(ggml_cross_entropy_loss)
    PRISM_BACKEND_CALL(ggml_cross_entropy_loss_back)
    PRISM_BACKEND_CALL(ggml_cumsum)
    PRISM_BACKEND_CALL(ggml_diag)
    PRISM_BACKEND_CALL(ggml_diag_mask_inf)
    PRISM_BACKEND_CALL(ggml_div)
    PRISM_BACKEND_CALL(ggml_div_inplace)
    PRISM_BACKEND_CALL(ggml_dsv4_hc_comb)
    PRISM_BACKEND_CALL(ggml_dsv4_hc_post)
    PRISM_BACKEND_CALL(ggml_dsv4_hc_pre)
    PRISM_BACKEND_CALL(ggml_dup)
    PRISM_BACKEND_CALL(ggml_element_size)
    PRISM_BACKEND_CALL(ggml_fill)
    PRISM_BACKEND_CALL(ggml_flash_attn_ext)
    PRISM_BACKEND_CALL(ggml_flash_attn_ext_add_sinks)
    PRISM_BACKEND_CALL(ggml_flash_attn_ext_set_prec)
    PRISM_BACKEND_CALL(ggml_floor)
    PRISM_BACKEND_CALL(ggml_gated_delta_net)
    PRISM_BACKEND_CALL(ggml_gated_delta_net_replay)
    PRISM_BACKEND_CALL(ggml_gated_delta_net_rows)
    PRISM_BACKEND_CALL(ggml_gated_delta_net_set_raw_gates)
    PRISM_BACKEND_CALL(ggml_gated_linear_attn)
    PRISM_BACKEND_CALL(ggml_get_rows)
    PRISM_BACKEND_CALL(ggml_get_rows_back)
    PRISM_BACKEND_CALL(ggml_get_tensor)
    PRISM_BACKEND_CALL(ggml_glu)
    PRISM_BACKEND_CALL(ggml_glu_split)
    PRISM_BACKEND_CALL(ggml_graph_add_node)
    PRISM_BACKEND_CALL(ggml_graph_cpy)
    PRISM_BACKEND_CALL(ggml_graph_n_nodes)
    PRISM_BACKEND_CALL(ggml_graph_overhead)
    PRISM_BACKEND_CALL(ggml_graph_overhead_custom)
    PRISM_BACKEND_CALL(ggml_graph_reset)
    PRISM_BACKEND_CALL(ggml_graph_size)
    PRISM_BACKEND_CALL(ggml_group_norm)
    PRISM_BACKEND_CALL(ggml_im2col)
    PRISM_BACKEND_CALL(ggml_im2col_3d)
    PRISM_BACKEND_CALL(ggml_interpolate)
    PRISM_BACKEND_CALL(ggml_is_matrix)
    PRISM_BACKEND_CALL(ggml_is_quantized)
    PRISM_BACKEND_CALL(ggml_is_vector)
    PRISM_BACKEND_CALL(ggml_l2_norm)
    PRISM_BACKEND_CALL(ggml_leaky_relu)
    PRISM_BACKEND_CALL(ggml_lightning_indexer)
    PRISM_BACKEND_CALL(ggml_log)
    PRISM_BACKEND_CALL(ggml_mean)
    PRISM_BACKEND_CALL(ggml_mul)
    PRISM_BACKEND_CALL(ggml_mul_inplace)
    PRISM_BACKEND_CALL(ggml_mul_mat)
    PRISM_BACKEND_CALL(ggml_mul_mat_id)
    PRISM_BACKEND_CALL(ggml_mul_mat_set_hint)
    PRISM_BACKEND_CALL(ggml_nbytes)
    PRISM_BACKEND_CALL(ggml_nelements)
    PRISM_BACKEND_CALL(ggml_new_graph)
    PRISM_BACKEND_CALL(ggml_new_graph_custom)
    PRISM_BACKEND_CALL(ggml_new_tensor)
    PRISM_BACKEND_CALL(ggml_new_tensor_1d)
    PRISM_BACKEND_CALL(ggml_new_tensor_2d)
    PRISM_BACKEND_CALL(ggml_new_tensor_3d)
    PRISM_BACKEND_CALL(ggml_new_tensor_4d)
    PRISM_BACKEND_CALL(ggml_norm)
    PRISM_BACKEND_CALL(ggml_nrows)
    PRISM_BACKEND_CALL(ggml_opt_step_adamw)
    PRISM_BACKEND_CALL(ggml_opt_step_sgd)
    PRISM_BACKEND_CALL(ggml_out_prod)
    PRISM_BACKEND_CALL(ggml_pad)
    PRISM_BACKEND_CALL(ggml_pad_circular)
    PRISM_BACKEND_CALL(ggml_pad_ext)
    PRISM_BACKEND_CALL(ggml_pad_ext_circular)
    PRISM_BACKEND_CALL(ggml_pad_reflect_1d)
    PRISM_BACKEND_CALL(ggml_permute)
    PRISM_BACKEND_CALL(ggml_pool_1d)
    PRISM_BACKEND_CALL(ggml_pool_2d)
    PRISM_BACKEND_CALL(ggml_relu)
    PRISM_BACKEND_CALL(ggml_repeat)
    PRISM_BACKEND_CALL(ggml_repeat_4d)
    PRISM_BACKEND_CALL(ggml_repeat_back)
    PRISM_BACKEND_CALL(ggml_reshape_2d)
    PRISM_BACKEND_CALL(ggml_reshape_3d)
    PRISM_BACKEND_CALL(ggml_reshape_4d)
    PRISM_BACKEND_CALL(ggml_rms_norm)
    PRISM_BACKEND_CALL(ggml_rms_norm_back)
    PRISM_BACKEND_CALL(ggml_rms_norm_inplace)
    PRISM_BACKEND_CALL(ggml_roll)
    PRISM_BACKEND_CALL(ggml_rope)
    PRISM_BACKEND_CALL(ggml_rope_ext)
    PRISM_BACKEND_CALL(ggml_rope_ext_back)
    PRISM_BACKEND_CALL(ggml_rope_ext_inplace)
    PRISM_BACKEND_CALL(ggml_rope_multi)
    PRISM_BACKEND_CALL(ggml_rope_multi_back)
    PRISM_BACKEND_CALL(ggml_rope_multi_inplace)
    PRISM_BACKEND_CALL(ggml_rope_set_offset)
    PRISM_BACKEND_CALL(ggml_round)
    PRISM_BACKEND_CALL(ggml_rwkv_wkv6)
    PRISM_BACKEND_CALL(ggml_rwkv_wkv7)
    PRISM_BACKEND_CALL(ggml_scale)
    PRISM_BACKEND_CALL(ggml_scale_bias)
    PRISM_BACKEND_CALL(ggml_scale_bias_inplace)
    PRISM_BACKEND_CALL(ggml_set)
    PRISM_BACKEND_CALL(ggml_set_inplace)
    PRISM_BACKEND_CALL(ggml_set_loss)
    PRISM_BACKEND_CALL(ggml_set_name)
    PRISM_BACKEND_CALL(ggml_set_param)
    PRISM_BACKEND_CALL(ggml_set_rows)
    PRISM_BACKEND_CALL(ggml_sigmoid)
    PRISM_BACKEND_CALL(ggml_silu)
    PRISM_BACKEND_CALL(ggml_silu_back)
    PRISM_BACKEND_CALL(ggml_sin)
    PRISM_BACKEND_CALL(ggml_soft_max)
    PRISM_BACKEND_CALL(ggml_soft_max_add_sinks)
    PRISM_BACKEND_CALL(ggml_soft_max_ext)
    PRISM_BACKEND_CALL(ggml_soft_max_ext_back)
    PRISM_BACKEND_CALL(ggml_soft_max_ext_inplace)
    PRISM_BACKEND_CALL(ggml_softplus)
    PRISM_BACKEND_CALL(ggml_solve_tri)
    PRISM_BACKEND_CALL(ggml_sqr)
    PRISM_BACKEND_CALL(ggml_sqrt)
    PRISM_BACKEND_CALL(ggml_ssm_conv)
    PRISM_BACKEND_CALL(ggml_ssm_scan)
    PRISM_BACKEND_CALL(ggml_sub)
    PRISM_BACKEND_CALL(ggml_sub_inplace)
    PRISM_BACKEND_CALL(ggml_sum)
    PRISM_BACKEND_CALL(ggml_sum_rows)
    PRISM_BACKEND_CALL(ggml_swiglu_oai)
    PRISM_BACKEND_CALL(ggml_swiglu_split)
    PRISM_BACKEND_CALL(ggml_tanh)
    PRISM_BACKEND_CALL(ggml_tensor_overhead)
    PRISM_BACKEND_CALL(ggml_timestep_embedding)
    PRISM_BACKEND_CALL(ggml_top_k)
    PRISM_BACKEND_CALL(ggml_transpose)
    PRISM_BACKEND_CALL(ggml_tri)
    PRISM_BACKEND_CALL(ggml_trunc)
    PRISM_BACKEND_CALL(ggml_type_size)
    PRISM_BACKEND_CALL(ggml_type_name)
    PRISM_BACKEND_CALL(ggml_unary_op_name)
    PRISM_BACKEND_CALL(ggml_op_name)
    PRISM_BACKEND_CALL(ggml_op_desc)
    PRISM_BACKEND_CALL(ggml_get_name)
    PRISM_BACKEND_CALL(ggml_unary)
    PRISM_BACKEND_CALL(ggml_upscale)
    PRISM_BACKEND_CALL(ggml_view_1d)
    PRISM_BACKEND_CALL(ggml_view_2d)
    PRISM_BACKEND_CALL(ggml_view_3d)
    PRISM_BACKEND_CALL(ggml_view_4d)
    PRISM_BACKEND_CALL(ggml_view_tensor)
    PRISM_BACKEND_CALL(ggml_xielu)
#undef PRISM_BACKEND_CALL
    throw std::invalid_argument("backend production function is not allowlisted");
}

struct comparison_capture {
    state & owner;
    std::filesystem::path prefix;
    std::size_t maximum;
    std::size_t total=0;
    json pairs=json::array();
    std::exception_ptr error;
    static bool callback(int index,ggml_tensor * left,ggml_tensor * right,void * data) noexcept {
        auto & c=*static_cast<comparison_capture *>(data);
        if(c.error) return true;
        try {
            const auto left_bytes=ggml_nbytes(left),right_bytes=ggml_nbytes(right);
            require(left_bytes<=c.maximum-c.total,"backend compare capture exceeds bound"); c.total+=left_bytes;
            require(right_bytes<=c.maximum-c.total,"backend compare capture exceeds bound"); c.total+=right_bytes;
            const auto serial=c.pairs.size();
            const auto lp=c.prefix.string()+"-"+std::to_string(serial)+"-left.bin";
            const auto rp=c.prefix.string()+"-"+std::to_string(serial)+"-right.bin";
            std::vector<std::uint8_t> bytes(left_bytes); ggml_backend_tensor_get(left,bytes.data(),0,left_bytes); c.owner.write(lp,bytes.data(),bytes.size());
            bytes.resize(right_bytes); ggml_backend_tensor_get(right,bytes.data(),0,right_bytes); c.owner.write(rp,bytes.data(),bytes.size());
            c.pairs.push_back({{"Index",index},{"Left",c.owner.metadata(left)},{"Right",c.owner.metadata(right)},{"LeftFile",lp},{"RightFile",rp},{"Bytes",left_bytes}});
        } catch(...) { c.error=std::current_exception(); }
        // The source callback also completes the graph after comparison failures.
        return true;
    }
};

inline json state::run(const json & a) {
    const auto op=text(a.at("Op"));
    if(op=="context-init") {
        keys(a,{"Op","Id","MemoryBytes","NoAlloc"}); const auto bytes=bounded(a.at("MemoryBytes"),1,std::size_t(1)<<30);
        require(a.at("NoAlloc").is_boolean() && context_bytes<=std::size_t(1)<<30 && bytes<=(std::size_t(1)<<30)-context_bytes,"backend context memory bound");
        require(contexts.size()<16,"backend context count exceeds bound"); auto * context=ggml_init({bytes,nullptr,a.at("NoAlloc").get<bool>()}); require(context,"backend context init failed");
        try { bind(contexts,a.at("Id"),context); } catch(...) { ggml_free(context); throw; } context_bytes+=bytes; return json::object();
    }
    if(op=="backend-init") {
        keys(a,{"Op","Id","Device","Threads","UseRef"}); require(text(a.at("Device"))=="CPU" && a.at("UseRef").is_boolean(),"backend CPU device/UseRef required");
        require(backends.size()<4,"backend count exceeds bound"); const auto threads=a.at("Threads").is_null()?0:bounded(a.at("Threads"),1,256);
        auto * backend=ggml_backend_cpu_init(); require(backend,"backend CPU init failed");
        try { bind(backends,a.at("Id"),backend); } catch(...) { ggml_backend_free(backend); throw; }
        if(!a.at("Threads").is_null()) ggml_backend_cpu_set_n_threads(backend,int(threads));
        const auto setter=reinterpret_cast<void(*)(ggml_backend_t,bool)>(ggml_backend_reg_get_proc_address(ggml_backend_cpu_reg(),"ggml_backend_cpu_set_use_ref"));
        if(setter) setter(backend,a.at("UseRef").get<bool>());
        return {{"UseRefAvailable",setter!=nullptr}};
    }
    if(op=="call") { call(a); return json::object(); }
    if(op=="scalar") {
        if(a.contains("Release")) { keys(a,{"Op","Scalar","Release"}); require(a.at("Release").is_boolean(),"backend scalar Release must be boolean"); }
        else keys(a,{"Op","Scalar"});
        const auto value=lookup(scalars,a.at("Scalar")); if(a.value("Release",false)) scalars.erase(id(a.at("Scalar"))); return {{"Value",value}};
    }
    if(op=="context-snapshot") {
        keys(a,{"Op","Context"}); auto * ctx=lookup(contexts,a.at("Context")); json values=json::array();
        for(auto * t=ggml_get_first_tensor(ctx);t;t=ggml_get_next_tensor(ctx,t)) { require(values.size()<32768,"backend tensor count exceeds bound"); tensor_id(t); }
        for(auto * t=ggml_get_first_tensor(ctx);t;t=ggml_get_next_tensor(ctx,t)) values.push_back(metadata(t)); return {{"Tensors",values}};
    }
    if(op=="tensor-snapshot") {
        keys(a,{"Op","Tensor"}); auto * tensor=lookup(tensors,a.at("Tensor")); require(tensor,"backend null tensor snapshot");
        // Match a complete ordered context snapshot's automatic identities,
        // without serializing unrelated tensors for a single field lookup.
        for(const auto & context:contexts) {
            for(auto * t=ggml_get_first_tensor(context.second);t;t=ggml_get_next_tensor(context.second,t)) tensor_id(t);
            for(auto * t=ggml_get_first_tensor(context.second);t;t=ggml_get_next_tensor(context.second,t)) {
                for(auto * source:t->src) tensor_id(source);
                tensor_id(t); tensor_id(t->view_src);
            }
        }
        return {{"Tensor",metadata(tensor)}};
    }
    if(op=="graph-snapshot") {
        keys(a,{"Op","Graph"}); auto * g=lookup(graphs,a.at("Graph")); json nodes=json::array(),leaves=json::array(),gradients=json::array();
        require(g->n_nodes>=0 && g->n_nodes<=32768 && g->n_leafs>=0 && g->n_leafs<=32768,"backend graph extent exceeds bound");
        for(int i=0;i<g->n_nodes;++i) { auto * t=g->nodes[i]; nodes.push_back(tensor_id(t)); gradients.push_back({{"Tensor",tensor_id(t)},{"Gradient",tensor_id(ggml_graph_get_grad(g,t))},{"GradientAccumulator",tensor_id(ggml_graph_get_grad_acc(g,t))}}); }
        for(int i=0;i<g->n_leafs;++i) leaves.push_back(tensor_id(g->leafs[i])); return {{"Nodes",nodes},{"Leaves",leaves},{"Gradients",gradients}};
    }
    if(op=="allocate") {
        keys(a,{"Op","Context","Backend","Result"}); keys(a.at("Result"),{"Buffer"}); auto * buffer=ggml_backend_alloc_ctx_tensors(lookup(contexts,a.at("Context")),lookup(backends,a.at("Backend")));
        require(buffer,"backend tensor allocation failed"); try { bind(buffers,a.at("Result").at("Buffer"),buffer); } catch(...) { ggml_backend_buffer_free(buffer); throw; } return json::object();
    }
    if(op=="supports") { keys(a,{"Op","Backend","Tensor"}); return {{"Supported",ggml_backend_supports_op(lookup(backends,a.at("Backend")),lookup(tensors,a.at("Tensor")))}}; }
    if(op=="backend-features") {
        keys(a,{"Op","Backend"}); const auto backend=lookup(backends,a.at("Backend"));
        const auto device=ggml_backend_get_device(backend); const auto registry=ggml_backend_dev_backend_reg(device);
        const auto get=reinterpret_cast<ggml_backend_get_features_t>(ggml_backend_reg_get_proc_address(registry,"ggml_backend_get_features"));
        json features=nullptr;
        if(get) if(const auto * values=get(registry)) {
            features=json::array();
            for(const auto * f=values;f->name;++f) { require(features.size()<1024,"backend feature count exceeds bound"); features.push_back({{"Name",f->name},{"Value",f->value?json(f->value):json(nullptr)}}); }
        }
        return {{"ProcedureAvailable",get!=nullptr},{"Features",features},{"RegistryName",ggml_backend_reg_name(registry)},{"BackendName",ggml_backend_name(backend)},{"DeviceName",device?json(ggml_backend_dev_name(device)):json(nullptr)}};
    }
    if(op=="tensor-set" || op=="tensor-get") {
        if(a.contains("Truncate")) { keys(a,{"Op","Tensor","File","FileOffset","TensorOffset","Bytes","Truncate"}); require(op=="tensor-get" && a.at("Truncate").is_boolean(),"backend Truncate only applies to tensor-get"); }
        else keys(a,{"Op","Tensor","File","FileOffset","TensorOffset","Bytes"});
        auto * t=lookup(tensors,a.at("Tensor")); require(t && t->data,"backend unallocated tensor I/O");
        const auto offset=bounded(a.at("TensorOffset"),0,max_file_bytes),bytes=bounded(a.at("Bytes"),0,max_file_bytes),file_offset=bounded(a.at("FileOffset"),0,max_file_bytes);
        require(offset<=ggml_nbytes(t) && bytes<=ggml_nbytes(t)-offset,"backend tensor I/O extent differs");
        if(op=="tensor-set") { const auto data=read(a.at("File"),file_offset,bytes); ggml_backend_tensor_set(t,data.data(),offset,bytes); return json::object(); }
        std::vector<std::uint8_t> data(bytes); ggml_backend_tensor_get(t,data.data(),offset,bytes); write(a.at("File"),data.data(),bytes,file_offset,a.value("Truncate",file_offset==0)); return {{"Bytes",bytes}};
    }
    if(op=="tensor-write") {
        keys(a,{"Op","Tensor","Fields"}); auto * t=lookup(tensors,a.at("Tensor")); require(t,"backend null tensor write"); const auto & fields=a.at("Fields"); require(fields.is_object(),"backend tensor fields required");
        for(auto it=fields.begin();it!=fields.end();++it) {
            if(it.key()=="NB") { require(it.value().is_array() && it.value().size()==GGML_MAX_DIMS,"backend NB extent differs"); for(int i=0;i<GGML_MAX_DIMS;++i) t->nb[i]=bounded(it.value()[i],0,max_file_bytes); }
            else if(it.key()=="Op") t->op=static_cast<ggml_op>(bounded(it.value(),0,GGML_OP_COUNT-1));
            else if(it.key()=="Sources") { require(it.value().is_array() && it.value().size()==GGML_MAX_SRC,"backend source extent differs"); for(int i=0;i<GGML_MAX_SRC;++i) t->src[i]=it.value()[i].is_null()?nullptr:lookup(tensors,it.value()[i]); }
            else if(it.key()=="OpParamsHex") { const auto bytes=unhex(it.value()); require(bytes.size()==sizeof(t->op_params),"backend op params extent differs"); std::memcpy(t->op_params,bytes.data(),bytes.size()); }
            else throw std::invalid_argument("backend unknown mutable tensor field");
        } return json::object();
    }
    if(op=="compare") {
        keys(a,{"Op","Backend1","Backend2","Graph","Nodes","Prefix","CaptureTensors","MaxOutputBytes"});
        const auto maximum=bounded(a.at("MaxOutputBytes"),1,std::size_t(8)*1024*1024*1024);
        comparison_capture capture{*this,file(a.at("Prefix")),maximum}; std::vector<const ggml_tensor *> nodes;
        if(!a.at("Nodes").is_null()) { require(a.at("Nodes").is_array() && !a.at("Nodes").empty() && a.at("Nodes").size()<=32768,"backend compared nodes exceeds bound"); for(const auto & n:a.at("Nodes")) nodes.push_back(lookup(tensors,n)); }
        const auto compared=ggml_backend_compare_graph_backend(lookup(backends,a.at("Backend1")),lookup(backends,a.at("Backend2")),lookup(graphs,a.at("Graph")),comparison_capture::callback,&capture,nodes.empty()?nullptr:nodes.data(),nodes.size());
        if(capture.error) std::rethrow_exception(capture.error);
        const auto & ids=a.at("CaptureTensors"); require(ids.is_array() && ids.size()<=32768,"backend capture count exceeds bound"); json captures=json::array();
        for(const auto & n:ids) { auto * t=lookup(tensors,n); const auto bytes=ggml_nbytes(t); require(bytes<=maximum-capture.total,"backend tensor capture exceeds bound"); capture.total+=bytes;
            const auto path=capture.prefix.string()+"-capture-"+std::to_string(id(n))+".bin"; std::vector<std::uint8_t> data(bytes); ggml_backend_tensor_get(t,data.data(),0,bytes); write(path,data.data(),bytes); captures.push_back({{"Tensor",n},{"File",path},{"Bytes",bytes}}); }
        return {{"Compared",compared},{"Pairs",capture.pairs},{"Captures",captures}};
    }
    if(op=="type-info") {
        keys(a,{"Op","Type"}); const auto type=static_cast<ggml_type>(bounded(a.at("Type"),0,GGML_TYPE_COUNT-1));
        const auto * traits=ggml_get_type_traits(type);
        return {{"Name",traits->type_name?json(traits->type_name):json(nullptr)},{"BlockSize",traits->blck_size},{"TypeSize",traits->type_size},
            {"Quantized",traits->is_quantized},{"RequiresImportance",ggml_quantize_requires_imatrix(type)},{"HasToFloat",traits->to_float!=nullptr}};
    }
    if(op=="quantize-file") {
        keys(a,{"Op","Type","InputFile","OutputFile","Start","Rows","ElementsPerRow","ImportanceFile"});
        const auto type=static_cast<ggml_type>(bounded(a.at("Type"),0,GGML_TYPE_COUNT-1)); const auto * traits=ggml_get_type_traits(type);
        require(traits->blck_size>0 && traits->type_size>0,"backend unavailable quantization type");
        const auto start=bounded(a.at("Start"),0,max_file_bytes/sizeof(float)),rows=bounded(a.at("Rows"),1,max_file_bytes/sizeof(float)),row=bounded(a.at("ElementsPerRow"),1,max_file_bytes/sizeof(float));
        const auto source_bytes=read_all(a.at("InputFile")); require(source_bytes.size()%sizeof(float)==0,"backend F32 input extent differs");
        const auto count=source_bytes.size()/sizeof(float);
        require(start<=count && row%std::size_t(traits->blck_size)==0 && start%row==0 && rows<=(count-start)/row,"backend quantization extent differs");
        std::vector<float> source(count); std::memcpy(source.data(),source_bytes.data(),source_bytes.size());
        std::vector<float> importance;
        if(!a.at("ImportanceFile").is_null()) { const auto bytes=read_all(a.at("ImportanceFile")); require(bytes.size()%sizeof(float)==0 && bytes.size()/sizeof(float)>=row,"backend importance extent differs"); importance.resize(bytes.size()/sizeof(float)); std::memcpy(importance.data(),bytes.data(),bytes.size()); }
        require(!importance.empty() || !ggml_quantize_requires_imatrix(type),"backend quantization requires importance");
        const auto output_size=ggml_row_size(type,std::int64_t(start+rows*row)); require(output_size<=max_file_bytes,"backend quantization output exceeds bound");
        std::vector<std::uint8_t> output(output_size);
        ggml_quantize_init(type);
        const auto written=ggml_quantize_chunk(type,source.data(),output.data(),std::int64_t(start),std::int64_t(rows),std::int64_t(row),importance.empty()?nullptr:importance.data());
        const auto offset=ggml_row_size(type,std::int64_t(start)); require(offset<=output.size() && written<=output.size()-offset,"backend quantizer returned invalid extent");
        write(a.at("OutputFile"),output.data()+offset,written,offset,false); return {{"Written",written}};
    }
    if(op=="dequantize-file") {
        keys(a,{"Op","Type","Count","InputFile","OutputFile"}); const auto type=static_cast<ggml_type>(bounded(a.at("Type"),0,GGML_TYPE_COUNT-1)); const auto * traits=ggml_get_type_traits(type);
        const auto count=bounded(a.at("Count"),1,max_file_bytes/sizeof(float));
        require(traits->blck_size>0 && traits->type_size>0 && count%std::size_t(traits->blck_size)==0,"backend dequantization block extent differs");
        const auto input=read_all(a.at("InputFile")); require(input.size()==ggml_row_size(type,std::int64_t(count)),"backend dequantization input extent differs");
        std::vector<float> output(count);
        if(type==GGML_TYPE_F32) std::memcpy(output.data(),input.data(),input.size());
        else { require(traits->to_float!=nullptr,"backend production dequantizer unavailable"); traits->to_float(input.data(),output.data(),std::int64_t(count)); }
        write(a.at("OutputFile"),output.data(),output.size()*sizeof(float)); return {{"Bytes",output.size()*sizeof(float)}};
    }
    if(op=="clock") { keys(a,{"Op"}); return {{"Microseconds",ggml_time_us()}}; }
    if(op=="hardware-concurrency") { keys(a,{"Op"}); return {{"Threads",std::thread::hardware_concurrency()}}; }
    throw std::invalid_argument("backend unknown action");
}
inline json read_actions(state & s,const json & file) {
        const auto content=prism_action_file::read(s.file(file),"backend ActionsFile");
        require(!content.empty(),"backend ActionsFile extent differs");
        std::vector<std::set<std::string>> object_keys;
        const auto callback=[&](int depth,json::parse_event_t event,json & parsed) {
            require(depth<=160,"backend ActionsFile nesting exceeds limit");
            if(event==json::parse_event_t::object_start) object_keys.emplace_back();
            else if(event==json::parse_event_t::key) require(!object_keys.empty() && object_keys.back().insert(parsed.get<std::string>()).second,"backend ActionsFile duplicate key");
            else if(event==json::parse_event_t::object_end) { require(!object_keys.empty(),"backend ActionsFile object nesting differs"); object_keys.pop_back(); }
            return true;
        };
        auto supplied=json::parse(content,callback); require(supplied.is_array() && supplied.size()<=524288,"backend external action count exceeds bound"); return supplied;
}
inline json execute(const json & o) {
    validate(o); state s(o); auto supplied=o.contains("ActionsFile")?read_actions(s,o.at("ActionsFile")):o.at("Actions");
    require(!supplied.empty(),"backend initial action array must not be empty");
    json trace=json::array(); std::size_t total=0,index=0;
    for(;;) {
        if(index==supplied.size()) return {{"Trace",trace}};
        require(total++<524288,"backend cumulative action count exceeds bound");
        const auto & a=supplied[index];
        try {
            require(a.is_object() && a.contains("Op") && a.at("Op").is_string(),"backend action Op required");
            if(a.at("Op")=="checkpoint") {
                keys(a,{"Op","OutputFile","NextFile","TimeoutMilliseconds"});
                require(index+1==supplied.size(),"backend checkpoint must end its batch");
                const auto output=s.file(a.at("OutputFile")),next=s.file(a.at("NextFile"));
                require(output!=next && !state::exists(output) && !state::exists(next),"backend checkpoint files must be fresh and distinct");
                const auto timeout=bounded(a.at("TimeoutMilliseconds"),1,3600000);
                trace.push_back(json::object()); const auto encoded=json({{"Trace",trace}}).dump();
                require(encoded.size()<=16*1024*1024,"backend checkpoint response exceeds16MiB");
                const auto temporary=output.string()+".tmp"; require(!state::exists(temporary),"backend checkpoint temporary file already exists");
                s.write(temporary,encoded.data(),encoded.size()); std::filesystem::rename(temporary,output);
                const auto deadline=std::chrono::steady_clock::now()+std::chrono::milliseconds(timeout);
                while(!state::exists(next)) {
                    require(std::chrono::steady_clock::now()<deadline,"backend checkpoint continuation timed out");
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
                const auto more=read_actions(s,next.string()); trace=json::array(); supplied=more; index=0;
                continue;
            }
            trace.push_back(s.run(a));
        } catch(const std::exception & e) { throw std::runtime_error("backend action "+std::to_string(total-1)+": "+e.what()); }
        ++index;
    }
}
} // namespace prism_backend_ops
