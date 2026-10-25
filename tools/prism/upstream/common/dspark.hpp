// Bounded production model-loader observations; no fixtures or verdicts.
#pragma once
#include "llama.h"
#include "ggml-backend.h"
#include <nlohmann/json.hpp>
#include <atomic>
#include <cstdint>
#include <filesystem>
#include <initializer_list>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
namespace prism_dspark {
using json=nlohmann::ordered_json;
inline void require(bool ok,const char*message){if(!ok)throw std::runtime_error(message);}
inline void fields(const json&v,std::initializer_list<const char*>keys){require(v.is_object()&&v.size()==keys.size(),"model observation fields mismatch");for(auto*k:keys)require(v.contains(k),"model observation field missing");}
inline std::string path(const json&v){require(v.is_string(),"model observation path must be string");auto s=v.get<std::string>();require(!s.empty()&&s.size()<=32768&&s.find('\0')==std::string::npos,"model observation path extent/NUL violation");auto p=std::filesystem::u8path(s);require(p.is_absolute()&&std::filesystem::is_regular_file(p)&&std::filesystem::file_size(p)<=64*1024*1024,"model observation file must be regular/absolute/bounded64MiB");return s;}
inline bool handles(const std::string&kind){return kind=="model-load-observe";}
inline void validate(const json&op){fields(op,{"Kind","Paths","GpuLayers"});require(op.at("Kind")=="model-load-observe","unknown model observation kind");require(op.at("GpuLayers").is_number_integer()&&op.at("GpuLayers").get<std::int64_t>()>=0&&op.at("GpuLayers").get<std::int64_t>()<=999,"model observation GPU layers outside0..999");const auto&paths=op.at("Paths");require(paths.is_array()&&!paths.empty()&&paths.size()<=16,"model observation path count outside1..16");for(const auto&p:paths)path(p);}
inline std::string hex(const std::string&bytes){constexpr char digits[]="0123456789abcdef";std::string value(bytes.size()*2,'\0');for(std::size_t i=0;i<bytes.size();++i){auto b=static_cast<unsigned char>(bytes[i]);value[2*i]=digits[b>>4];value[2*i+1]=digits[b&15];}return value;}
struct backend_owner{backend_owner(){ggml_backend_load_all();llama_backend_init();}~backend_owner(){llama_backend_free();}};
struct log_owner{
    ggml_log_callback original=nullptr;void*user=nullptr;std::mutex mutex;std::string bytes;std::atomic<bool>failed{false};
    log_owner(){llama_log_get(&original,&user);llama_log_set([](ggml_log_level,const char*message,void*opaque){auto&state=*static_cast<log_owner*>(opaque);try{std::lock_guard<std::mutex>guard(state.mutex);const auto n=std::char_traits<char>::length(message);if(n>128*1024-state.bytes.size()){state.failed=true;return;}state.bytes.append(message,n);}catch(...){state.failed=true;}},this);}
    ~log_owner(){llama_log_set(original,user);}
};
inline json execute(const json&op){validate(op);backend_owner backend;json models=json::array();std::size_t total=0;for(const auto&p:op.at("Paths")){log_owner log;auto parameters=llama_model_default_params();parameters.n_gpu_layers=op.at("GpuLayers").get<int>();std::unique_ptr<llama_model,decltype(&llama_model_free)>model(llama_model_load_from_file(path(p).c_str(),parameters),llama_model_free);std::string observed;{std::lock_guard<std::mutex>guard(log.mutex);observed=log.bytes;}require(!log.failed.load(),"production model log exceeded128KiB/capture failed");total+=observed.size();require(total<=1024*1024,"production model logs exceeded1MiB");models.push_back(json{{"Loaded",model!=nullptr},{"LogHex",hex(observed)}});}return json{{"Models",models}};}
} // namespace prism_dspark
