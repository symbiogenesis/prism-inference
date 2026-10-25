// Persistent production model/context operations. All schedules/verdicts are Rust.
#pragma once
#include "llama.h"
#include "common.h"
#include "arg.h"
#include "sampling.h"
#include "chat.h"
#include "vocab_session.hpp"
#include "model_resolution.hpp"
#include "generated_model.hpp"
#include "context_graph.hpp"
#include "dspark_runtime.hpp"
#include "llguidance.hpp"
#include "backend_sampler.hpp"
#include "codec_buffers.hpp"
#include "llama-model-saver.h"
#include "ggml-cpu.h"
#include <nlohmann/json.hpp>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

namespace prism_model_runtime {
using json=nlohmann::ordered_json;
inline void require(bool condition,const char*message){if(!condition)throw std::runtime_error(message);}
inline void fields(const json&v,std::initializer_list<const char*>names){require(v.is_object()&&v.size()==names.size(),"model session fields mismatch");for(auto*n:names)require(v.contains(n),"model session field missing");}
inline std::uint64_t number(const json&v,std::uint64_t maximum){require(v.is_number_integer()&&(v.is_number_unsigned()||v.get<std::int64_t>()>=0),"model session unsigned integer required");auto n=v.get<std::uint64_t>();require(n<=maximum,"model session integer bound exceeded");return n;}
inline std::int32_t integer(const json&v,std::int32_t minimum,std::int32_t maximum){require(v.is_number_integer(),"model session signed integer required");if(v.is_number_unsigned())require(v.get<std::uint64_t>()<=static_cast<std::uint64_t>(maximum),"model session integer bound exceeded");auto n=v.get<std::int64_t>();require(n>=minimum&&n<=maximum,"model session integer bound exceeded");return static_cast<std::int32_t>(n);}
inline std::string text(const json&v,std::size_t bound=32768){require(v.is_string(),"model session string required");auto s=v.get<std::string>();require(s.size()<=bound&&s.find('\0')==std::string::npos,"model session string bound/NUL violation");return s;}
inline std::string hex(const void*p,std::size_t n){require(n<=8*1024*1024,"model state data exceeds bound");const auto*b=static_cast<const std::uint8_t*>(p);constexpr char d[]="0123456789abcdef";std::string s(n*2,'\0');for(std::size_t i=0;i<n;++i){s[2*i]=d[b[i]>>4];s[2*i+1]=d[b[i]&15];}return s;}
inline std::string path(const json&v){auto s=text(v);require(std::filesystem::u8path(s).is_absolute(),"model path must be absolute");return s;}
inline std::string state_name(const json&v){auto s=text(v,64);require(!s.empty()&&s.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-.")==std::string::npos&&s.find("..") == std::string::npos,"state file must be an owned plain filename");return s;}

struct parameters{common_params value;};
struct common_owner{std::shared_ptr<parameters> params;common_init_result_ptr value;};
struct model{llama_model*value=nullptr;std::shared_ptr<common_owner> common;std::shared_ptr<gguf_context> metadata;std::shared_ptr<model>head_source;~model(){if(value&&!common)llama_model_free(value);}};
struct context{llama_context*value=nullptr;std::shared_ptr<model> parent;std::shared_ptr<common_owner> common;std::string mean_center_path;std::shared_ptr<prism_dspark_runtime::trace>trace;std::map<llama_seq_id,std::shared_ptr<void>>backend_samplers;std::vector<llama_sampler_seq_config>sampler_configs;~context(){if(value&&!common)llama_free(value);}};
struct sampler{bool borrowed=false;llama_sampler*core=nullptr;common_sampler*common=nullptr;std::shared_ptr<model> parent;~sampler(){if(core)llama_sampler_free(core);if(common&&!borrowed)common_sampler_free(common);}};
struct batch{llama_batch value{};bool allocated=false;std::size_t capacity=0,sequences=0;std::vector<llama_token>tokens;std::vector<llama_pos>positions;~batch(){if(allocated)llama_batch_free(value);}};
struct blob{std::vector<std::uint8_t>data;std::shared_ptr<context>parent;};
struct temporary_file{FILE*value=nullptr;~temporary_file(){if(value)std::fclose(value);}};
struct checkpoint{common_prompt_checkpoint value{};};
template<class T>struct table{
    std::mutex mutex;std::map<std::uint64_t,std::pair<std::uint32_t,std::shared_ptr<T>>>values;
    std::uint64_t add(std::uint64_t id,std::uint32_t worker,std::shared_ptr<T>value){std::lock_guard<std::mutex>guard(mutex);require(values.size()<512,"model handle count exceeded");values.emplace(id,std::make_pair(worker,std::move(value)));return id;}
    std::shared_ptr<T>get(std::uint64_t id,std::uint32_t worker,bool shared=false){std::lock_guard<std::mutex>guard(mutex);auto it=values.find(id);require(it!=values.end(),"unknown model session handle");require(shared||it->second.first==worker,"handle used outside owning native worker");return it->second.second;}
    void release(std::uint64_t id,std::uint32_t worker){std::shared_ptr<T>owned;{std::lock_guard<std::mutex>guard(mutex);auto it=values.find(id);require(it!=values.end()&&it->second.first==worker,"release must use owning native worker");owned=std::move(it->second.second);values.erase(it);}owned.reset();}
};
struct lane{
    std::mutex mutex;std::condition_variable ready;std::deque<std::function<void()>>tasks;bool stopping=false;std::thread thread;
    lane():thread([this]{for(;;){std::function<void()>task;{std::unique_lock<std::mutex>guard(mutex);ready.wait(guard,[this]{return stopping||!tasks.empty();});if(tasks.empty())return;task=std::move(tasks.front());tasks.pop_front();}task();}}){}
    void push(std::function<void()>task){{std::lock_guard<std::mutex>guard(mutex);require(!stopping&&tasks.size()<32,"native worker queue exceeds bound");tasks.push_back(std::move(task));}ready.notify_one();}
    void stop(){{std::lock_guard<std::mutex>guard(mutex);stopping=true;}ready.notify_all();if(thread.joinable())thread.join();}
    ~lane(){stop();}
};
class session{
    FILE*wire;
    struct logger_state{ggml_log_callback original=nullptr;void*data=nullptr;ggml_log_level minimum=GGML_LOG_LEVEL_ERROR;bool active=false;}logger;
    table<parameters>params;table<model>models;table<context>contexts;table<sampler>samplers;table<batch>batches;table<blob>blobs;table<temporary_file>files;table<checkpoint>checkpoints;prism_dspark_runtime::state dspark_runtime;prism_llguidance::state llguidance;prism_backend_sampler::state backend_sampler;prism_codec_buffers::state codec_buffers;
    std::atomic<std::uint64_t>next_handle{1},next_callback{1};std::atomic<std::size_t>inflight{0};
    std::mutex output_mutex,callback_mutex;std::condition_variable callback_ready;std::map<std::uint64_t,json>callback_returns;std::map<std::uint64_t,bool>pending_callbacks;bool stopping=false;
    std::map<std::uint32_t,std::unique_ptr<lane>>workers;
    std::shared_ptr<parameters>param(const json&v,std::uint32_t w){return params.get(number(v,1000000),w,true);}
    std::shared_ptr<model>mod(const json&v,std::uint32_t w){return models.get(number(v,1000000),w,true);}
    std::shared_ptr<context>ctx(const json&v,std::uint32_t w){return contexts.get(number(v,1000000),w);}
    std::shared_ptr<batch>bat(const json&v,std::uint32_t w){return batches.get(number(v,1000000),w);}
    template<class T>std::uint64_t add(table<T>&t,std::uint32_t w,std::shared_ptr<T>v){return t.add(next_handle++,w,std::move(v));}
public:
    explicit session(FILE*output):wire(output){}
    void send(const json&v){auto line=v.dump();require(line.size()<=16*1024*1024,"model response frame exceeds16MiB");std::lock_guard<std::mutex>guard(output_mutex);require(std::fwrite(line.data(),1,line.size(),wire)==line.size()&&std::fputc('\n',wire)!=EOF&&std::fflush(wire)==0,"model wire write failed");}
    bool progress(std::uint64_t request,float value){const auto id=next_callback++;require(id<=8192,"model progress callback count exceeds bound");std::uint32_t bits;std::memcpy(&bits,&value,sizeof(bits));{std::lock_guard<std::mutex>guard(callback_mutex);require(pending_callbacks.emplace(id,false).second,"duplicate callback allocation");}send(json{{"CallbackId",id},{"RequestId",request},{"ProgressBits",bits}});std::unique_lock<std::mutex>guard(callback_mutex);require(callback_ready.wait_for(guard,std::chrono::seconds(120),[&]{return stopping||callback_returns.count(id);}),"Rust callback reply timed out");if(stopping)return false;const bool result=callback_returns.at(id).at("Return").get<bool>();callback_returns.erase(id);pending_callbacks.erase(id);return result;}
    json tensor(std::uint64_t request,const json&spec){const auto id=next_callback++;require(id<=8192,"model tensor callback count exceeds bound");{std::lock_guard<std::mutex>guard(callback_mutex);require(pending_callbacks.emplace(id,true).second,"duplicate tensor callback allocation");}send(json{{"CallbackId",id},{"RequestId",request},{"TensorSpec",spec}});std::unique_lock<std::mutex>guard(callback_mutex);require(callback_ready.wait_for(guard,std::chrono::seconds(120),[&]{return stopping||callback_returns.count(id);}),"Rust tensor reply timed out");require(!stopping,"model session stopped during tensor input");auto reply=callback_returns.at(id);callback_returns.erase(id);pending_callbacks.erase(id);if(reply.contains("Error"))throw std::runtime_error(text(reply.at("Error"),4096));return reply.at("Result");}
    json sampler_event(std::uint64_t request,const json&spec){const auto id=next_callback++;require(id<=8192,"model sampler callback count exceeds bound");{std::lock_guard<std::mutex>guard(callback_mutex);require(pending_callbacks.emplace(id,true).second,"duplicate tensor callback allocation");}send(json{{"CallbackId",id},{"RequestId",request},{"SamplerEvent",spec}});std::unique_lock<std::mutex>guard(callback_mutex);require(callback_ready.wait_for(guard,std::chrono::seconds(120),[&]{return stopping||callback_returns.count(id);}),"Rust sampler reply timed out");require(!stopping,"model session stopped during sampler input");auto reply=callback_returns.at(id);callback_returns.erase(id);pending_callbacks.erase(id);if(reply.contains("Error"))throw std::runtime_error(text(reply.at("Error"),4096));return reply.at("Result");}
    void callback(const json&v){require(v.is_object()&&v.contains("CallbackId"),"callback reply ID required");auto id=number(v.at("CallbackId"),8192);std::lock_guard<std::mutex>guard(callback_mutex);auto pending=pending_callbacks.find(id);require(pending!=pending_callbacks.end(),"callback reply is not pending");if(v.contains("Return")){fields(v,{"CallbackId","Return"});require(!pending->second&&v.at("Return").is_boolean(),"progress reply must be boolean");}else if(v.contains("Result")){fields(v,{"CallbackId","Result"});require(pending->second,"tensor reply sent for progress");}else{fields(v,{"CallbackId","Error"});require(pending->second,"tensor error sent for progress");text(v.at("Error"),4096);}require(callback_returns.size()<32&&callback_returns.emplace(id,v).second,"duplicate/excess callback reply");callback_ready.notify_all();}
    void shutdown(){{std::lock_guard<std::mutex>guard(callback_mutex);stopping=true;}callback_ready.notify_all();for(auto&worker:workers)worker.second->stop();}
    ~session(){shutdown();if(logger.active)llama_log_set(logger.original,logger.data);}
    void enqueue(std::uint64_t id,std::uint32_t worker,json operation){require(inflight.fetch_add(1)<32,"model request concurrency exceeds32");auto&target=workers[worker];if(!target)target=std::make_unique<lane>();target->push([this,id,worker,operation=std::move(operation)]{try{auto value=execute(id,worker,operation);if(worker==0)backend_sampler.check();send(json{{"Id",id},{"Result",value}});}catch(const std::exception&e){send(json{{"Id",id},{"Error",std::string(e.what()).substr(0,4096)}});}--inflight;});}
    json execute(std::uint64_t request,std::uint32_t worker,const json&o){
        require(o.is_object()&&o.contains("Op"),"model operation Op required");const auto op=text(o.at("Op"),64);
        if(worker==0)backend_sampler.prepare(request,[this](std::uint64_t active,const json&event){return sampler_event(active,event);});
        if(prism_codec_buffers::handles(op))return codec_buffers.execute(o,worker,[&](const json&v){return mod(v,worker);});
        if(prism_backend_sampler::handles(op))return backend_sampler.execute(o,worker,[&](const json&v){return ctx(v,worker);},[&](const json&v){return mod(v,worker);});
        if(op=="log-minimum"){fields(o,{"Op","Level"});auto level=integer(o.at("Level"),1,5);if(!logger.active){llama_log_get(&logger.original,&logger.data);logger.active=true;}logger.minimum=static_cast<ggml_log_level>(level);llama_log_set([](ggml_log_level severity,const char*message,void*opaque){auto&state=*static_cast<logger_state*>(opaque);state.original(severity>=state.minimum?severity:GGML_LOG_LEVEL_DEBUG,message,state.data);},&logger);return nullptr;}
        if(prism_llguidance::handles(op))return llguidance.execute(o,worker,[&](const json&v){return mod(v,worker);});
        if(prism_dspark_runtime::handles(op))return dspark_runtime.execute(o,worker,[this,worker](const json&v){return ctx(v,worker);},[this,worker](const json&v){return mod(v,worker);},[this,worker](const json&v){return bat(v,worker);});
        if(prism_model_resolution::handles(op))return prism_model_resolution::execute(o);
        if(prism_context_graph::handles(op)){auto owner=ctx(o.at("Context"),worker);return prism_context_graph::execute(o,owner->value);}
        if(prism_generated_model::handles(op)){std::shared_ptr<context>owned;if(op!="kv-center-write")owned=ctx(o.at("Context"),worker);return prism_generated_model::execute(o,owned?owned->value:nullptr);}
        if(prism_vocab_session::handles(op)){std::shared_ptr<context>owned;llama_context*shared=nullptr;if(op=="vocab-tokenize"||op=="vocab-detokenize"||op=="raw-vocab-detokenize"){owned=contexts.get(number(o.at("Context"),1000000),worker,true);shared=owned->value;}return prism_vocab_session::execute(o,shared);}
        if(op=="backend-init"||op=="backend-free"||op=="backend-load-all"||op=="common-init"){
            fields(o,{"Op"});if(op=="backend-init")llama_backend_init();else if(op=="backend-free")llama_backend_free();else if(op=="backend-load-all")ggml_backend_load_all();else common_init();return nullptr;
        }
        if(op=="devices"){fields(o,{"Op"});json values=json::array();require(ggml_backend_dev_count()<=16,"device count exceeds16");for(std::size_t i=0;i<ggml_backend_dev_count();++i){auto*d=ggml_backend_dev_get(i);values.push_back(json{{"Index",i},{"Type",static_cast<int>(ggml_backend_dev_type(d))},{"Name",ggml_backend_dev_name(d)},{"Description",ggml_backend_dev_description(d)},{"CpuBuffer",ggml_backend_dev_buffer_type(d)==ggml_backend_cpu_buffer_type()}});}return json{{"Devices",values}};}
        if(op=="architecture-inventory"){fields(o,{"Op"});json values=json::array();for(auto arch:llm_arch_all())values.push_back(json{{"Id",static_cast<int>(arch)},{"Name",llm_arch_name(arch)},{"ParsedId",static_cast<int>(llm_arch_from_string(llm_arch_name(arch)))},{"Saver",llama_model_saver_supports_arch(arch)},{"TensorSplit",llm_arch_supports_sm_tensor(arch)}});bool webgpu=false,hip=false;
#ifdef GGML_USE_WEBGPU
            webgpu=true;
#endif
#ifdef GGML_USE_HIP
            hip=true;
#endif
            return json{{"Architectures",values},{"WebGpu",webgpu},{"Hip",hip}};}
        if(op=="temporary-file"){fields(o,{"Op"});auto value=std::make_shared<temporary_file>();value->value=std::tmpfile();return json{{"File",value->value?json(add(files,worker,value)):json(nullptr)}};}
        if(op=="model-save-file-pointer"){fields(o,{"Op","Model","File"});auto m=mod(o.at("Model"),worker);auto f=files.get(number(o.at("File"),1000000),worker);llama_model_saver saver(m->value);saver.add_kv_from_model();saver.add_tensors_from_model();saver.save(f->value);std::rewind(f->value);return nullptr;}
        if(op=="model-load-file-pointer"){fields(o,{"Op","File","Devices","SplitMode"});auto f=files.get(number(o.at("File"),1000000),worker);auto p=llama_model_default_params();p.progress_callback=[](float,void*){return true;};p.split_mode=static_cast<llama_split_mode>(integer(o.at("SplitMode"),0,2));require(o.at("Devices").is_array()&&o.at("Devices").size()<=16,"file model devices exceed16");std::vector<ggml_backend_dev_t>devices;for(const auto&id:o.at("Devices")){auto i=number(id,15);require(i<ggml_backend_dev_count(),"file model device outside inventory");devices.push_back(ggml_backend_dev_get(i));}devices.push_back(nullptr);p.devices=devices.data();auto value=std::make_shared<model>();value->value=llama_model_load_from_file_ptr(f->value,p);return json{{"Model",value->value?json(add(models,worker,value)):json(nullptr)}};}
        if(op=="model-save"){fields(o,{"Op","Model","Name"});auto filename=state_name(o.at("Name"));require(!std::filesystem::exists(filename),"model output already exists");llama_model_save_to_file(mod(o.at("Model"),worker)->value,filename.c_str());return json{{"Exists",std::filesystem::is_regular_file(filename)},{"Bytes",std::filesystem::is_regular_file(filename)?std::filesystem::file_size(filename):0}};}
        if(op=="hardware-concurrency"){fields(o,{"Op"});return json{{"Count",std::thread::hardware_concurrency()}};}
        if(op=="numa-init"){fields(o,{"Op","Params"});llama_numa_init(param(o.at("Params"),worker)->value.numa);return nullptr;}
        if(op=="params"){
            if(o.contains("Example")){fields(o,{"Op","Initial","Arguments","Example"});require(o.at("Example")=="completion"||o.at("Example")=="export-graph-ops","unsupported parameter example");}else fields(o,{"Op","Initial","Arguments"});auto value=std::make_shared<parameters>();const auto&initial=o.at("Initial");require(initial.is_object()&&initial.size()<=8,"common initial field count exceeds bound");
            for(auto it=initial.begin();it!=initial.end();++it){if(it.key()=="Predict")value->value.n_predict=integer(it.value(),-1,2048);else if(it.key()=="Batch")value->value.n_batch=integer(it.value(),1,2048);else if(it.key()=="Context")value->value.n_ctx=integer(it.value(),0,4096);else if(it.key()=="Parallel")value->value.n_parallel=integer(it.value(),1,16);else if(it.key()=="Seed")value->value.sampling.seed=static_cast<std::uint32_t>(number(it.value(),UINT32_MAX));else if(it.key()=="Unified"){require(it.value().is_boolean(),"Unified must be boolean");value->value.kv_unified=it.value().get<bool>();}else if(it.key()=="Prompt")value->value.prompt=text(it.value());else if(it.key()=="OutputFile")value->value.out_file=state_name(it.value());else throw std::runtime_error("unknown common initial field");}
            const auto&arguments=o.at("Arguments");require(arguments.is_array()&&!arguments.empty()&&arguments.size()<=32,"common argument count exceeds bound");std::vector<std::string>strings;for(const auto&arg:arguments)strings.push_back(text(arg));const std::set<std::string>allowed={"-m","-ngl","-p","-n","-c","-ub","-np","-t","--lora","--seed","--temp"};for(std::size_t i=1;i<strings.size();){if(strings[i]=="-no-cnv"){++i;continue;}require(i+1<strings.size()&&allowed.count(strings[i]),"unsupported common argument");if(strings[i]=="-m"||strings[i]=="--lora")path(arguments[i+1]);i+=2;}std::vector<char*>argv;for(auto&s:strings)argv.push_back(s.data());bool parsed=common_params_parse(static_cast<int>(argv.size()),argv.data(),value->value,o.contains("Example")?(o.at("Example")=="completion"?LLAMA_EXAMPLE_COMPLETION:LLAMA_EXAMPLE_EXPORT_GRAPH_OPS):LLAMA_EXAMPLE_COMMON);auto id=add(params,worker,value);return json{{"Handle",id},{"Parsed",parsed}};
        }
        if(op=="params-update"){fields(o,{"Op","Params","Fields"});auto p=param(o.at("Params"),worker);const auto&values=o.at("Fields");require(values.is_object()&&values.size()<=4,"parameter update field count exceeds bound");for(auto it=values.begin();it!=values.end();++it){const auto key=it.key();if(key=="Devices"){require(it.value().is_array()&&it.value().size()<=16,"parameter device list exceeds bound");std::vector<ggml_backend_dev_t>devices;for(const auto&id:it.value()){auto index=number(id,15);require(index<ggml_backend_dev_count(),"parameter device outside inventory");devices.push_back(ggml_backend_dev_get(index));}devices.push_back(nullptr);p->value.devices=std::move(devices);}else if(key=="GpuLayers")p->value.n_gpu_layers=integer(it.value(),-1,999);else if(key=="FitParams"||key=="Warmup"){require(it.value().is_boolean(),"parameter boolean update required");if(key=="FitParams")p->value.fit_params=it.value().get<bool>();else p->value.warmup=it.value().get<bool>();}else throw std::runtime_error("unknown parameter update field");}return nullptr;}
        if(op=="common-model-sampler"){fields(o,{"Op","Model","Sequence"});auto m=mod(o.at("Model"),worker);require(m->common&&m->common->value,"model has no common initialization owner");auto value=std::make_shared<sampler>();value->parent=m;value->borrowed=true;value->common=m->common->value->sampler(integer(o.at("Sequence"),0,15));return json{{"Sampler",value->common?json(add(samplers,worker,value)):json(nullptr)}};}
        if(op=="model-user-metadata"){fields(o,{"Op","MetadataFile","GpuLayers","Devices"});auto p=llama_model_default_params();p.n_gpu_layers=integer(o.at("GpuLayers"),-1,999);std::vector<ggml_backend_dev_t>devices;if(!o.at("Devices").is_null()){require(o.at("Devices").is_array()&&o.at("Devices").size()<=16,"metadata model device count exceeds16");for(const auto&id:o.at("Devices")){auto index=number(id,15);require(index<ggml_backend_dev_count(),"metadata model device outside inventory");devices.push_back(ggml_backend_dev_get(index));}devices.push_back(nullptr);p.devices=devices.data();}auto value=std::make_shared<model>();auto loaded=prism_generated_model::initialize_metadata(o.at("MetadataFile"),p);value->value=loaded.value;value->metadata=std::move(loaded.metadata);return json{{"Model",value->value?json(add(models,worker,value)):json(nullptr)}};}
        if(op=="params-unified") {fields(o,{"Op","Params","Value"});auto p=param(o.at("Params"),worker);require(o.at("Value").is_boolean(),"unified value must be boolean");p->value.kv_unified=o.at("Value").get<bool>();return nullptr;}
        if(op=="context-params"){fields(o,{"Op","Params"});auto p=common_context_params_to_llama(param(o.at("Params"),worker)->value);return json{{"Batch",p.n_batch},{"MicroBatch",p.n_ubatch}};}
        if(op=="common-model"){
            fields(o,{"Op","Params","ModelOnly"});auto p=param(o.at("Params"),worker);require(o.at("ModelOnly").is_boolean(),"ModelOnly must be boolean");auto owner=std::make_shared<common_owner>();owner->params=p;owner->value=common_init_from_params(p->value,o.at("ModelOnly").get<bool>());json model_id=nullptr,context_id=nullptr;
            if(owner->value&&owner->value->model()){auto m=std::make_shared<model>();m->common=owner;m->value=owner->value->model();model_id=add(models,worker,m);if(owner->value->context()){auto c=std::make_shared<context>();c->parent=m;c->common=owner;c->value=owner->value->context();context_id=add(contexts,worker,c);}}
            return json{{"Model",model_id},{"Context",context_id}};
        }
        if(op=="model-user"){
            fields(o,{"Op","MetadataFile","Devices","SplitMode"});auto p=llama_model_default_params();p.split_mode=static_cast<llama_split_mode>(integer(o.at("SplitMode"),0,2));std::vector<ggml_backend_dev_t>devices;if(!o.at("Devices").is_null()){require(o.at("Devices").is_array()&&o.at("Devices").size()<=16,"user model device count exceeds16");for(const auto&value:o.at("Devices")){auto index=number(value,15);require(index<ggml_backend_dev_count(),"user model device outside inventory");devices.push_back(ggml_backend_dev_get(index));}devices.push_back(nullptr);p.devices=devices.data();}auto value=std::make_shared<model>();auto loaded=prism_generated_model::initialize(o.at("MetadataFile"),p,[this,request](const json&spec){return tensor(request,spec);});value->value=loaded.value;value->metadata=std::move(loaded.metadata);return json{{"Model",value->value?json(add(models,worker,value)):json(nullptr)}};
        }
        if(op=="model-load"){
            fields(o,{"Op","Path","Base","Params","Overrides","Progress"});auto filename=path(o.at("Path"));auto base=text(o.at("Base"),16);llama_model_params p{};if(base=="default")p=llama_model_default_params();else if(base=="common")p=common_model_params_to_llama(param(o.at("Params"),worker)->value);else require(base=="zero","unknown model parameter base");const auto&overrides=o.at("Overrides");require(overrides.is_object()&&overrides.size()<=7,"model overrides exceed bound");std::vector<ggml_backend_dev_t>devices;std::shared_ptr<model>head_source;
            for(auto it=overrides.begin();it!=overrides.end();++it){if(it.key()=="GpuLayers")p.n_gpu_layers=integer(it.value(),0,999);else if(it.key()=="HeadSource"){head_source=mod(it.value(),worker);p.dspark_head_source=head_source->value;}else if(it.key()=="VocabOnly"){require(it.value().is_boolean(),"VocabOnly must be boolean");p.vocab_only=it.value().get<bool>();}else if(it.key()=="LoadMode"){require(it.value()=="none","unsupported load mode");p.load_mode=LLAMA_LOAD_MODE_NONE;}else if(it.key()=="SplitMode")p.split_mode=static_cast<llama_split_mode>(integer(it.value(),0,2));else if(it.key()=="MainGpu")p.main_gpu=integer(it.value(),-1,15);else if(it.key()=="Devices"){require(it.value().is_array()&&it.value().size()<=16,"model device list exceeds bound");for(const auto&id:it.value()){auto i=number(id,15);require(i<ggml_backend_dev_count(),"model device index outside inventory");devices.push_back(ggml_backend_dev_get(i));}devices.push_back(nullptr);p.devices=devices.data();}else throw std::runtime_error("unknown model override");}
            struct callback_data{session*owner;std::uint64_t request;std::exception_ptr error;}callback{this,request,{}};require(o.at("Progress").is_boolean(),"Progress must be boolean");if(o.at("Progress").get<bool>()){p.progress_callback=[](float v,void*u){auto*d=static_cast<callback_data*>(u);try{return d->owner->progress(d->request,v);}catch(...){d->error=std::current_exception();return false;}};p.progress_callback_user_data=&callback;}
            auto value=std::make_shared<model>();value->head_source=std::move(head_source);value->value=llama_model_load_from_file(filename.c_str(),p);if(callback.error)std::rethrow_exception(callback.error);return json{{"Model",value->value?json(add(models,worker,value)):json(nullptr)}};
        }
        if(op=="context"){
            fields(o,{"Op","Model","Params","Overrides"});auto m=mod(o.at("Model"),worker);auto p=o.at("Params").is_null()?llama_context_default_params():common_context_params_to_llama(param(o.at("Params"),worker)->value);auto value=std::make_shared<context>();value->parent=m;const auto&overrides=o.at("Overrides");require(overrides.is_object()&&overrides.size()<=16,"context overrides exceed bound");
            for(auto it=overrides.begin();it!=overrides.end();++it){auto key=it.key();if(key=="Samplers"){require(worker==0&&it.value().is_array()&&it.value().size()<=16,"context sampler configs require worker0 and at most16 entries");for(const auto&config:it.value()){fields(config,{"Sequence","Sampler"});auto seq=integer(config.at("Sequence"),0,15);auto owned=backend_sampler.sampler(config.at("Sampler"));value->backend_samplers[seq]=owned;value->sampler_configs.push_back({seq,owned->value});}p.samplers=value->sampler_configs.data();p.n_samplers=value->sampler_configs.size();}else if(key=="TraceId"){require(worker==0,"trace configuration requires native worker0");value->trace=dspark_runtime.trace_handle(it.value());p.cb_eval=prism_dspark_runtime::trace::callback;p.cb_eval_user_data=value->trace.get();}else if(key=="NoPerf"){require(it.value().is_boolean(),"NoPerf must be boolean");p.no_perf=it.value().get<bool>();}else if(key=="OutputsMax")p.n_outputs_max=static_cast<std::uint32_t>(number(it.value(),4096));else if(key=="OutputsPerSequence")p.n_outputs_max_per_seq=static_cast<std::uint32_t>(number(it.value(),4096));else if(key=="Context")p.n_ctx=static_cast<std::uint32_t>(number(it.value(),4096));else if(key=="Sequences")p.n_seq_max=static_cast<std::uint32_t>(integer(it.value(),1,16));else if(key=="Unified"){require(it.value().is_boolean(),"Unified must be boolean");p.kv_unified=it.value().get<bool>();}else if(key=="Batch")p.n_batch=static_cast<std::uint32_t>(integer(it.value(),1,4096));else if(key=="MicroBatch")p.n_ubatch=static_cast<std::uint32_t>(integer(it.value(),1,4096));else if(key=="Threads")p.n_threads=integer(it.value(),1,256);else if(key=="ThreadsBatch")p.n_threads_batch=integer(it.value(),1,256);else if(key=="TypeK")p.type_k=static_cast<ggml_type>(integer(it.value(),0,GGML_TYPE_COUNT-1));else if(key=="FlashAttention")p.flash_attn_type=static_cast<llama_flash_attn_type>(integer(it.value(),-1,1));else if(key=="RecurrentSnapshots")p.n_rs_seq=static_cast<std::uint32_t>(number(it.value(),64));else if(key=="MeanCenterPath"){value->mean_center_path=prism_generated_model::name(it.value());p.path_kv_mean_center=value->mean_center_path.c_str();}else throw std::runtime_error("unknown context override");}
            value->value=llama_init_from_model(m->value,p);return json{{"Context",value->value?json(add(contexts,worker,value)):json(nullptr)}};
        }
        if(op=="model-chat-render"){
            fields(o,{"Op","Model","Messages","EnableThinking","AddGeneration"});
            require(o.at("Messages").is_array()&&o.at("Messages").size()<=256&&o.at("EnableThinking").is_boolean()&&o.at("AddGeneration").is_boolean(),"model chat render extent/flags invalid");
            auto m=mod(o.at("Model"),worker);auto templates=common_chat_templates_init(m->value,"");common_chat_templates_inputs inputs;
            inputs.enable_thinking=o.at("EnableThinking").get<bool>();inputs.add_generation_prompt=o.at("AddGeneration").get<bool>();
            std::size_t supplied=0;for(const auto&message:o.at("Messages")){fields(message,{"role","content"});common_chat_msg v;v.role=text(message.at("role"),128);v.content=text(message.at("content"),1024*1024);supplied+=v.role.size()+v.content.size();require(supplied<=1024*1024,"model chat messages exceed1MiB");inputs.messages.push_back(std::move(v));}
            return json{{"Prompt",common_chat_templates_apply(templates.get(),inputs).prompt}};
        }
        if(op=="vocab"){fields(o,{"Op","Model"});auto m=mod(o.at("Model"),worker);const auto*v=llama_model_get_vocab(m->value);return json{{"Count",llama_vocab_n_tokens(v)},{"Bos",llama_vocab_bos(v)},{"Type",static_cast<int>(llama_vocab_type(v))}};}
        if(op=="tokenize"){fields(o,{"Op","Context","Text","Special"});require(o.at("Special").is_boolean(),"tokenization Special must be boolean");auto c=ctx(o.at("Context"),worker);return json{{"Tokens",common_tokenize(c->value,text(o.at("Text")),o.at("Special").get<bool>())}};}
        if(op=="token-info"){if(o.contains("Special")){fields(o,{"Op","Context","Token","Special"});require(o.at("Special").is_boolean(),"token piece Special must be boolean");}else fields(o,{"Op","Context","Token"});auto c=ctx(o.at("Context"),worker);auto token=integer(o.at("Token"),0,1048576);const auto*v=llama_model_get_vocab(c->parent->value);require(token<llama_vocab_n_tokens(v),"token outside model vocabulary");auto piece=common_token_to_piece(c->value,token,o.contains("Special")?o.at("Special").get<bool>():true);return json{{"Eog",llama_vocab_is_eog(v,token)},{"PieceHex",hex(piece.data(),piece.size())}};}
        if(op=="batch-new"){fields(o,{"Op","Capacity","Sequences"});auto capacity=integer(o.at("Capacity"),1,2048),sequences=integer(o.at("Sequences"),1,16);auto value=std::make_shared<batch>();value->capacity=capacity;value->sequences=sequences;value->allocated=true;value->value=llama_batch_init(capacity,0,sequences);return json{{"Batch",add(batches,worker,value)}};}
        if(op=="batch-one"){
            fields(o,{"Op","Batch","Tokens","Positions"});auto value=o.at("Batch").is_null()?std::make_shared<batch>():bat(o.at("Batch"),worker);require(!value->allocated,"cannot replace allocated batch with borrowed one");require(o.at("Tokens").is_array()&&!o.at("Tokens").empty()&&o.at("Tokens").size()<=2048,"one-batch token count exceeds bound");value->tokens.clear();for(const auto&t:o.at("Tokens"))value->tokens.push_back(integer(t,0,1048576));value->value=llama_batch_get_one(value->tokens.data(),static_cast<int32_t>(value->tokens.size()));value->positions.clear();if(!o.at("Positions").is_null()){require(o.at("Positions").is_array()&&o.at("Positions").size()==value->tokens.size(),"one-batch position extent mismatch");for(const auto&p:o.at("Positions"))value->positions.push_back(integer(p,0,8192));value->value.pos=value->positions.data();}return json{{"Batch",o.at("Batch").is_null()?json(add(batches,worker,value)):o.at("Batch")}};
        }
        if(op=="batch-clear"){fields(o,{"Op","Batch"});auto value=bat(o.at("Batch"),worker);require(value->allocated,"clear requires allocated batch");common_batch_clear(value->value);return nullptr;}
        if(op=="batch-add"){
            fields(o,{"Op","Batch","Rows"});auto value=bat(o.at("Batch"),worker);require(value->allocated&&o.at("Rows").is_array()&&value->value.n_tokens>=0&&static_cast<std::size_t>(value->value.n_tokens)+o.at("Rows").size()<=value->capacity,"batch-add exceeds allocated capacity");for(const auto&row:o.at("Rows")){fields(row,{"Token","Position","Sequences","Logits"});require(row.at("Sequences").is_array()&&!row.at("Sequences").empty()&&row.at("Sequences").size()<=value->sequences&&row.at("Logits").is_boolean(),"invalid batch row sequences/logits");std::vector<llama_seq_id>seqs;for(const auto&s:row.at("Sequences"))seqs.push_back(integer(s,0,15));common_batch_add(value->value,integer(row.at("Token"),0,1048576),integer(row.at("Position"),0,8192),seqs,row.at("Logits").get<bool>());}return nullptr;
        }
        if(op=="batch-logit"){fields(o,{"Op","Batch","Index","Value"});auto value=bat(o.at("Batch"),worker);auto index=number(o.at("Index"),2047);require(value->allocated&&index<static_cast<std::size_t>(value->value.n_tokens)&&o.at("Value").is_boolean(),"invalid batch logit input");value->value.logits[index]=o.at("Value").get<bool>();return nullptr;}
        if(op=="decode"||op=="encode"){fields(o,{"Op","Context","Batch"});auto c=ctx(o.at("Context"),worker);auto b=bat(o.at("Batch"),worker);return json{{"Code",op=="decode"?llama_decode(c->value,b->value):llama_encode(c->value,b->value)}};}
        if(op=="sampler"){
            fields(o,{"Op","Model","Kind","Seed","Params"});auto m=mod(o.at("Model"),worker);auto value=std::make_shared<sampler>();value->parent=m;auto kind=text(o.at("Kind"),16);if(kind=="dist"){auto p=llama_sampler_chain_default_params();value->core=llama_sampler_chain_init(p);if(value->core)llama_sampler_chain_add(value->core,llama_sampler_init_dist(static_cast<std::uint32_t>(number(o.at("Seed"),UINT32_MAX))));}else{require(kind=="common","unknown sampler kind");value->common=common_sampler_init(m->value,param(o.at("Params"),worker)->value.sampling);}return json{{"Sampler",value->core||value->common?json(add(samplers,worker,value)):json(nullptr)}};
        }
        if(op=="sample"){fields(o,{"Op","Sampler","Context","Index"});auto value=samplers.get(number(o.at("Sampler"),1000000),worker);auto c=ctx(o.at("Context"),worker);auto index=integer(o.at("Index"),-1,2047);return json{{"Token",value->core?llama_sampler_sample(value->core,c->value,index):common_sampler_sample(value->common,c->value,index)}};}
        if(op=="sampler-accept"){fields(o,{"Op","Sampler","Token","Grammar"});require(o.at("Grammar").is_boolean(),"sampler Grammar must be boolean");auto value=samplers.get(number(o.at("Sampler"),1000000),worker);auto token=integer(o.at("Token"),-1,1048576);if(value->core)llama_sampler_accept(value->core,token);else common_sampler_accept(value->common,token,o.at("Grammar").get<bool>());return nullptr;}
        if(op=="synchronize"){fields(o,{"Op","Context"});llama_synchronize(ctx(o.at("Context"),worker)->value);return nullptr;}
        if(op=="state-save-file"){
            fields(o,{"Op","Context","Name","Tokens"});auto c=ctx(o.at("Context"),worker);auto filename=state_name(o.at("Name"));require(o.at("Tokens").is_array()&&o.at("Tokens").size()<=2048,"state token count exceeds bound");std::vector<llama_token>tokens;for(const auto&t:o.at("Tokens"))tokens.push_back(integer(t,0,1048576));return json{{"Saved",llama_state_save_file(c->value,filename.c_str(),tokens.data(),tokens.size())}};
        }
        if(op=="state-load-file"){
            fields(o,{"Op","Context","Name","Capacity"});auto c=ctx(o.at("Context"),worker);auto filename=state_name(o.at("Name"));auto capacity=number(o.at("Capacity"),2048);std::vector<llama_token>tokens(capacity);std::size_t count=0;bool loaded=llama_state_load_file(c->value,filename.c_str(),tokens.data(),tokens.size(),&count);require(count<=capacity,"production state token count exceeds capacity");tokens.resize(count);return json{{"Loaded",loaded},{"Count",count},{"Tokens",tokens}};
        }
        if(op=="state-get"){
            fields(o,{"Op","Context","Sequence","Extended","Flags","ReturnData"});auto c=ctx(o.at("Context"),worker);auto sequence=integer(o.at("Sequence"),0,15);require(o.at("Extended").is_boolean()&&o.at("ReturnData").is_boolean(),"state-get booleans required");auto extended=o.at("Extended").get<bool>();auto flags=static_cast<llama_state_seq_flags>(number(o.at("Flags"),2));auto size=extended?llama_state_seq_get_size_ext(c->value,sequence,flags):llama_state_seq_get_size(c->value,sequence);require(size<=8*1024*1024,"production sequence state exceeds8MiB");auto value=std::make_shared<blob>();value->parent=c;value->data.resize(size);auto copied=extended?llama_state_seq_get_data_ext(c->value,value->data.data(),size,sequence,flags):llama_state_seq_get_data(c->value,value->data.data(),size,sequence);return json{{"Blob",add(blobs,worker,value)},{"Size",size},{"Copied",copied},{"DataHex",o.at("ReturnData").get<bool>()?json(hex(value->data.data(),size)):json(nullptr)}};
        }
        if(op=="state-set"){
            fields(o,{"Op","Context","Blob","Sequence","Extended","Flags"});auto c=ctx(o.at("Context"),worker);auto value=blobs.get(number(o.at("Blob"),1000000),worker);auto sequence=integer(o.at("Sequence"),0,15);require(o.at("Extended").is_boolean(),"state-set Extended must be boolean");auto flags=static_cast<llama_state_seq_flags>(number(o.at("Flags"),2));auto copied=o.at("Extended").get<bool>()?llama_state_seq_set_data_ext(c->value,value->data.data(),value->data.size(),sequence,flags):llama_state_seq_set_data(c->value,value->data.data(),value->data.size(),sequence);return json{{"Copied",copied}};
        }
        if(op=="memory-clear"){fields(o,{"Op","Context","Data"});require(o.at("Data").is_boolean(),"memory clear Data must be boolean");llama_memory_clear(llama_get_memory(ctx(o.at("Context"),worker)->value),o.at("Data").get<bool>());return nullptr;}
        if(op=="memory-remove"){fields(o,{"Op","Context","Sequence","From","To"});auto c=ctx(o.at("Context"),worker);return json{{"Removed",llama_memory_seq_rm(llama_get_memory(c->value),integer(o.at("Sequence"),0,15),integer(o.at("From"),-1,8192),integer(o.at("To"),-1,8192))}};}
        if(op=="memory-copy"){fields(o,{"Op","Context","Source","Destination","From","To"});llama_memory_seq_cp(llama_get_memory(ctx(o.at("Context"),worker)->value),integer(o.at("Source"),0,15),integer(o.at("Destination"),0,15),integer(o.at("From"),-1,8192),integer(o.at("To"),-1,8192));return nullptr;}
        if(op=="checkpoint-update"){fields(o,{"Op","Checkpoint","Context","Sequence","Flags"});auto value=o.at("Checkpoint").is_null()?std::make_shared<checkpoint>():checkpoints.get(number(o.at("Checkpoint"),1000000),worker);auto c=ctx(o.at("Context"),worker);auto flags=static_cast<llama_state_seq_flags>(number(o.at("Flags"),2));value->value.update_tgt(c->value,integer(o.at("Sequence"),0,15),flags);require(value->value.size()<=8*1024*1024,"checkpoint exceeds8MiB");return json{{"Checkpoint",o.at("Checkpoint").is_null()?json(add(checkpoints,worker,value)):o.at("Checkpoint")},{"Bytes",value->value.size()}};}
        if(op=="checkpoint-load"){fields(o,{"Op","Checkpoint","Context","Sequence","Flags"});auto value=checkpoints.get(number(o.at("Checkpoint"),1000000),worker);value->value.load_tgt(ctx(o.at("Context"),worker)->value,integer(o.at("Sequence"),0,15),static_cast<llama_state_seq_flags>(number(o.at("Flags"),2)));return nullptr;}
        if(op=="release"){fields(o,{"Op","Kind","Handle"});auto kind=text(o.at("Kind"),16);auto id=number(o.at("Handle"),1000000);if(kind=="params")params.release(id,worker);else if(kind=="model")models.release(id,worker);else if(kind=="context")contexts.release(id,worker);else if(kind=="sampler")samplers.release(id,worker);else if(kind=="batch")batches.release(id,worker);else if(kind=="blob")blobs.release(id,worker);else if(kind=="file")files.release(id,worker);else if(kind=="checkpoint")checkpoints.release(id,worker);else throw std::runtime_error("unknown handle release kind");return nullptr;}
        throw std::runtime_error("unknown model operation");
    }
};
inline bool read_line(std::string&line){line.clear();for(char c;std::cin.get(c);){if(c=='\n')return true;require(line.size()<16*1024*1024,"model request frame exceeds16MiB");line.push_back(c);}require(line.empty(),"truncated model request frame");return false;}
struct wire_output{
    FILE*value=nullptr;
    wire_output(){require(std::fflush(stdout)==0,"cannot flush production stdout");
#ifdef _WIN32
        const int descriptor=_dup(_fileno(stdout));require(descriptor>=0,"cannot duplicate model wire stdout");require(_dup2(_fileno(stderr),_fileno(stdout))==0,"cannot route production stdout to stderr");value=_fdopen(descriptor,"wb");
#else
        const int descriptor=dup(fileno(stdout));require(descriptor>=0,"cannot duplicate model wire stdout");require(dup2(fileno(stderr),fileno(stdout))>=0,"cannot route production stdout to stderr");value=fdopen(descriptor,"wb");
#endif
        require(value!=nullptr,"cannot open dedicated model wire stream");
    }
    ~wire_output(){if(value)std::fclose(value);}
};
inline json parse(const std::string&line){std::vector<std::set<std::string>>keys;auto callback=[&](int depth,json::parse_event_t event,json&value){require(depth<=8,"model frame nesting exceeds8");if(event==json::parse_event_t::object_start)keys.emplace_back();else if(event==json::parse_event_t::key){require(!keys.empty()&&keys.back().insert(value.get<std::string>()).second,"duplicate model JSON key");}else if(event==json::parse_event_t::object_end){require(!keys.empty(),"invalid model object nesting");keys.pop_back();}return true;};return json::parse(line,callback);}
inline int serve(){try{wire_output output;session state(output.value);std::set<std::uint64_t>seen;std::string line;while(read_line(line)){auto value=parse(line);if(value.contains("Shutdown")){fields(value,{"Shutdown"});require(value.at("Shutdown")==true,"invalid shutdown marker");state.shutdown();return 0;}if(value.contains("CallbackId")){state.callback(value);continue;}fields(value,{"Id","Worker","Operation"});auto id=number(value.at("Id"),1000000);auto worker=static_cast<std::uint32_t>(number(value.at("Worker"),15));require(id>0&&seen.size()<65536&&seen.insert(id).second,"duplicate/excess model request ID");state.enqueue(id,worker,value.at("Operation"));}state.shutdown();return 0;}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
} // namespace prism_model_runtime
