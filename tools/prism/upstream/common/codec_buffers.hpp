// Persistent byte buffers and production codec primitives. Rust owns all inputs,
// warmups, iterations, scheduling, timing math, statistics and expectations.
#pragma once
#include "action_file.hpp"
#include "codec.hpp"
#include "llama-model.h"
#include <algorithm>
#include <chrono>
#include <thread>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#if defined(__x86_64__) || defined(__i386__)
#include <x86intrin.h>
#endif
namespace prism_codec_buffers {
using json=nlohmann::ordered_json;
using prism_codec::require;
using prism_codec::keys;
using prism_codec::integer;
constexpr std::size_t max_buffer=512ULL*1024*1024;
inline bool handles(const std::string&op){return op.rfind("qb-",0)==0;}
inline std::string text(const json&v){require(v.is_string(),"codec buffer text required");auto s=v.get<std::string>();require(s.size()<=32768&&s.find('\0')==std::string::npos,"codec buffer text exceeds bound");return s;}
inline std::filesystem::path path(const json&v){auto p=std::filesystem::u8path(text(v));require(p.is_absolute(),"codec buffer path must be absolute");p=std::filesystem::weakly_canonical(p);auto relative=p.lexically_relative(std::filesystem::canonical(std::filesystem::current_path()));require(!relative.empty()&&!relative.is_absolute(),"codec buffer path outside session");for(const auto&part:relative)require(part!="..","codec buffer path outside session");return p;}
struct buffer{std::vector<std::uint8_t>storage;std::uint8_t*data=nullptr;std::size_t bytes=0;buffer(std::size_t n,std::size_t alignment,std::size_t offset):storage(n+alignment*2),bytes(n){void*p=storage.data();auto size=storage.size();require(std::align(alignment,alignment,p,size)!=nullptr,"codec buffer alignment failed");data=static_cast<std::uint8_t*>(p)+offset;require(std::size_t(data-storage.data())+bytes<=storage.size(),"codec buffer aligned extent differs");}};
struct state{
 std::mutex mutex;std::map<std::uint32_t,std::shared_ptr<buffer>>buffers;std::uint32_t next=1;std::size_t allocated=0;
 std::shared_ptr<buffer>get(const json&id){std::lock_guard<std::mutex>guard(mutex);auto it=buffers.find(static_cast<std::uint32_t>(integer(id,1000000)));require(it!=buffers.end(),"unknown codec buffer handle");return it->second;}
 static std::uint8_t*region(buffer&b,const json&v,std::size_t bytes){const auto offset=integer(v,b.bytes);require(bytes<=b.bytes-offset,"codec buffer access exceeds extent");return b.data+offset;}
 static std::size_t row_bytes(ggml_type type,std::size_t n){const auto block=ggml_blck_size(type);require(block>0&&n%std::size_t(block)==0,"codec element count violates production block precondition");auto bytes=ggml_row_size(type,static_cast<std::int64_t>(n));require(bytes<=max_buffer,"codec row exceeds buffer bound");return bytes;}
 template<class Model>json execute(const json&o,std::uint32_t worker,Model model){
 const auto op=text(o.at("Op"));
 if(op=="qb-inventory"){keys(o,{"Op"});ggml_cpu_init();json types=json::array();for(int i=0;i<GGML_TYPE_COUNT;++i){auto type=static_cast<ggml_type>(i);const auto*t=ggml_get_type_traits(type);const auto*c=ggml_get_type_traits_cpu(type);types.push_back({{"Type",i},{"Name",t->type_name?json(t->type_name):json(nullptr)},{"BlockSize",t->blck_size},{"TypeSize",t->type_size},{"FromFloatReference",t->from_float_ref!=nullptr},{"FromFloatCpu",c->from_float!=nullptr},{"ToFloat",t->to_float!=nullptr},{"VecDot",c->vec_dot!=nullptr},{"VecDotType",static_cast<int>(c->vec_dot_type)}});}return{{"Types",types}};}
 if(op=="qb-time-init"){keys(o,{"Op"});ggml_time_init();return nullptr;}
 if(op=="qb-init-type"){keys(o,{"Op","Type"});ggml_quantize_init(static_cast<ggml_type>(integer(o.at("Type"),GGML_TYPE_COUNT-1)));return nullptr;}
 if(op=="qb-buffer-new"){keys(o,{"Op","Bytes","Alignment","Offset"});require(worker==0,"codec allocation requires worker zero");auto bytes=integer(o.at("Bytes"),max_buffer),alignment=integer(o.at("Alignment"),4096),offset=integer(o.at("Offset"),4096);require(alignment>0&&(alignment&(alignment-1))==0&&offset<=alignment,"invalid codec buffer alignment/offset");std::lock_guard<std::mutex>guard(mutex);require(buffers.size()<32&&next<1000000&&bytes<=2ULL*1024*1024*1024-allocated,"codec aggregate buffer bound");auto value=std::make_shared<buffer>(bytes,alignment,offset);auto id=next++;buffers[id]=std::move(value);allocated+=bytes;return{{"Buffer",id}};}
 if(op=="qb-buffer-free"){keys(o,{"Op","Buffer"});require(worker==0,"codec release requires worker zero");std::lock_guard<std::mutex>guard(mutex);auto it=buffers.find(static_cast<std::uint32_t>(integer(o.at("Buffer"),1000000)));require(it!=buffers.end(),"unknown codec release handle");allocated-=it->second->bytes;buffers.erase(it);return nullptr;}
 if(op=="qb-buffer-load"||op=="qb-buffer-store"){keys(o,{"Op","Buffer","Offset","File","FileOffset","Bytes","Truncate"});require(o.at("Truncate").is_boolean(),"codec file Truncate must be boolean");auto b=get(o.at("Buffer"));auto count=integer(o.at("Bytes"),max_buffer);auto*data=region(*b,o.at("Offset"),count);auto filename=path(o.at("File"));auto offset=integer(o.at("FileOffset"),4ULL*1024*1024*1024);if(op=="qb-buffer-load"){require(!o.at("Truncate").get<bool>()&&std::filesystem::is_regular_file(filename)&&offset<=std::filesystem::file_size(filename)&&count<=std::filesystem::file_size(filename)-offset,"codec input file extent differs");std::ifstream input(filename,std::ios::binary);input.seekg(static_cast<std::streamoff>(offset));if(count)input.read(reinterpret_cast<char*>(data),static_cast<std::streamsize>(count));require(bool(input),"codec file load failed");}else{const auto truncate=o.at("Truncate").get<bool>();require(!truncate||offset==0,"codec truncate requires file offset zero");std::fstream output(filename,std::ios::binary|std::ios::out|(truncate?std::ios::trunc:std::ios::in));output.seekp(static_cast<std::streamoff>(offset));if(count)output.write(reinterpret_cast<const char*>(data),static_cast<std::streamsize>(count));require(bool(output),"codec file store failed");}return nullptr;}
 if(op=="qb-call"){
  const auto function=text(o.at("Function"));if(function=="vec-dot")keys(o,{"Op","Function","Type","Count","Input","Rhs","Output"});else keys(o,{"Op","Function","Type","Count","Input","Output"});auto type=static_cast<ggml_type>(integer(o.at("Type"),GGML_TYPE_COUNT-1));const auto*t=ggml_get_type_traits(type);const auto*c=ggml_get_type_traits_cpu(type);auto count=integer(o.at("Count"),134217728);keys(o.at("Input"),{"Buffer","Offset"});keys(o.at("Output"),{"Buffer","Offset"});auto input=get(o.at("Input").at("Buffer")),output=get(o.at("Output").at("Buffer"));
  if(function=="from-float-cpu"||function=="from-float-reference"){auto*p=region(*input,o.at("Input").at("Offset"),count*4);auto*q=region(*output,o.at("Output").at("Offset"),row_bytes(type,count));if(function=="from-float-cpu"){require(c->from_float,"production CPU quantizer unavailable");c->from_float(reinterpret_cast<const float*>(p),q,static_cast<std::int64_t>(count));}else{require(t->from_float_ref,"production reference quantizer unavailable");t->from_float_ref(reinterpret_cast<const float*>(p),q,static_cast<std::int64_t>(count));}}
  else if(function=="to-float"){require(t->to_float,"production decoder unavailable");t->to_float(region(*input,o.at("Input").at("Offset"),row_bytes(type,count)),reinterpret_cast<float*>(region(*output,o.at("Output").at("Offset"),count*4)),static_cast<std::int64_t>(count));}
  else{require(function=="vec-dot"&&c->vec_dot,"production vector dot unavailable");keys(o.at("Rhs"),{"Buffer","Offset"});auto rhs=get(o.at("Rhs").at("Buffer"));c->vec_dot(static_cast<int>(count),reinterpret_cast<float*>(region(*output,o.at("Output").at("Offset"),4)),0,region(*input,o.at("Input").at("Offset"),row_bytes(type,count)),0,region(*rhs,o.at("Rhs").at("Offset"),row_bytes(c->vec_dot_type,count)),0,1);}return nullptr;
 }
 if(op=="qb-clock"){keys(o,{"Op","Counter"});auto counter=text(o.at("Counter"));if(counter=="microseconds")return{{"Value",ggml_time_us()}};require(counter=="cycles","unknown codec clock counter");std::int64_t value=0;
#if defined(__x86_64__) || defined(__i386__)
#ifdef __POPCNT__
 unsigned int auxiliary;value=static_cast<std::int64_t>(__rdtscp(&auxiliary));
#else
 value=static_cast<std::int64_t>(__rdtsc());
#endif
#endif
 return{{"Value",value}};}
 if(op=="qb-trace"){
  const auto read_actions=[&](const std::filesystem::path&filename){return json::parse(prism_action_file::read(filename,"codec action file"),[](int depth,json::parse_event_t,json&){require(depth<=160,"codec action JSON depth exceeds bound");return true;});};
  json actions;if(o.contains("ActionsFile")){keys(o,{"Op","ActionsFile"});actions=read_actions(path(o.at("ActionsFile")));}else{keys(o,{"Op","Actions"});actions=o.at("Actions");}
  json trace=json::array();std::size_t total=0;for(;;){require(actions.is_array()&&actions.size()<=524288-total,"codec explicit action count exceeds bound");total+=actions.size();bool continued=false;
   for(std::size_t index=0;index<actions.size();++index){const auto&action=actions[index];require(action.is_object()&&action.contains("Op"),"codec trace action required");const auto name=text(action.at("Op"));
    if(name=="qb-checkpoint"){
     keys(action,{"Op","OutputFile","NextFile","TimeoutMilliseconds"});require(index+1==actions.size(),"codec checkpoint must end the supplied action batch");auto output=path(action.at("OutputFile")),next_file=path(action.at("NextFile"));auto temporary=output;temporary+=".partial";require(output!=next_file&&!std::filesystem::exists(output)&&!std::filesystem::exists(next_file)&&!std::filesystem::exists(temporary),"codec checkpoint files must be fresh and distinct");const auto timeout=integer(action.at("TimeoutMilliseconds"),3600000);require(timeout>0,"codec checkpoint timeout empty");trace.push_back(nullptr);auto encoded=json{{"Trace",trace}}.dump();require(encoded.size()<=16*1024*1024,"codec checkpoint output exceeds16MiB");{std::ofstream stream(temporary,std::ios::binary);stream.write(encoded.data(),static_cast<std::streamsize>(encoded.size()));require(bool(stream),"codec checkpoint write failed");}std::filesystem::rename(temporary,output);auto deadline=std::chrono::steady_clock::now()+std::chrono::milliseconds(timeout);while(!std::filesystem::exists(next_file)){require(std::chrono::steady_clock::now()<deadline,"codec continuation timed out");std::this_thread::sleep_for(std::chrono::milliseconds(1));}actions=read_actions(next_file);trace=json::array();continued=true;break;
    }
    require(name=="qb-call"||name=="qb-clock"||name=="qb-init-type"||name=="qb-time-init"||name=="qb-model-copy-f32"||name.rfind("qb-buffer-",0)==0,"codec trace action outside primitive allowlist");trace.push_back(execute(action,worker,model));
   }
   if(!continued)break;
  }
  return{{"Trace",trace}};
 }
 if(op=="qb-model-tensors"){keys(o,{"Op","Model"});auto m=model(o.at("Model"));json values=json::array();const auto&tensors=llama_internal_get_tensor_map(m->value);require(tensors.size()<=65536,"codec model tensor count exceeds bound");for(const auto&entry:tensors){auto*t=entry.second;json ne=json::array(),nb=json::array();for(int i=0;i<GGML_MAX_DIMS;++i){ne.push_back(t->ne[i]);nb.push_back(t->nb[i]);}values.push_back({{"Name",entry.first},{"Type",static_cast<int>(t->type)},{"NE",ne},{"NB",nb},{"Elements",ggml_nelements(t)},{"Bytes",ggml_nbytes(t)},{"TypeSize",ggml_type_size(t->type)},{"BlockSize",ggml_blck_size(t->type)}});}return{{"Tensors",values}};}
 if(op=="qb-model-copy-f32"){keys(o,{"Op","Model","Name","ElementOffset","Count","Buffer","BufferOffset"});auto m=model(o.at("Model"));const auto name=text(o.at("Name"));const auto&tensors=llama_internal_get_tensor_map(m->value);auto it=std::find_if(tensors.begin(),tensors.end(),[&](const auto&entry){return entry.first==name;});require(it!=tensors.end(),"codec model tensor missing");auto*t=it->second;require(t->data&&(t->type==GGML_TYPE_F32||t->type==GGML_TYPE_F16)&&ggml_is_contiguous(t),"codec source tensor must be contiguous CPU float");auto offset=integer(o.at("ElementOffset"),134217728),count=integer(o.at("Count"),134217728);require(offset<=static_cast<std::size_t>(ggml_nelements(t))&&count<=static_cast<std::size_t>(ggml_nelements(t))-offset,"codec model element range exceeds tensor");auto b=get(o.at("Buffer"));auto*out=reinterpret_cast<float*>(region(*b,o.at("BufferOffset"),count*4));if(t->type==GGML_TYPE_F32)std::memcpy(out,ggml_get_data_f32(t)+offset,count*4);else for(std::size_t i=0;i<count;++i)out[i]=ggml_get_f32_1d(t,static_cast<int>(i+offset));return nullptr;}
 throw std::runtime_error("unknown codec buffer operation");
 }
};
} // namespace prism_codec_buffers
