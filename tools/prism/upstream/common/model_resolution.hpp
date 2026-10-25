// Generic production HF plan/handler observations. The HTTP fixture and all
// expected resolution/path/precedence/type decisions belong to safe Rust.
#pragma once
#include "arg.h"
#include "download.h"
#include "log.h"
#include <nlohmann/json.hpp>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <vector>

namespace prism_model_resolution {
using json=nlohmann::ordered_json;
inline void require(bool value,const char*message){if(!value)throw std::runtime_error(message);}
inline void fields(const json&v,std::initializer_list<const char*>names){require(v.is_object()&&v.size()==names.size(),"resolution fields mismatch");for(const auto*n:names)require(v.contains(n),"resolution field missing");}
inline void text(const json&v,std::size_t bound){require(v.is_string(),"resolution string required");const auto&s=v.get_ref<const std::string&>();require(s.size()<=bound&&s.find('\0')==std::string::npos,"resolution text extent/NUL violation");}
inline bool handles(const std::string&op){return op=="resolution-plan"||op=="resolution-handler"||op=="resolution-pause-log";}
inline void validate(const json&operation){
    const auto op=operation.at("Op").get<std::string>();
    if(op=="resolution-pause-log"){fields(operation,{"Op"});return;}
    if(op=="resolution-plan"){
        fields(operation,{"Op","HfRepo","HfFile","Options"});text(operation.at("HfRepo"),256);text(operation.at("HfFile"),4096);const auto&options=operation.at("Options");fields(options,{"Mmproj","Mtp","Dflash","Eagle3","Dspark"});for(const auto*name:{"Mmproj","Mtp","Dflash","Eagle3","Dspark"})require(options.at(name).is_boolean(),"resolution option must be boolean");return;
    }
    require(op=="resolution-handler","unknown resolution operation");fields(operation,{"Op","Arguments","Example","OfflineAfterInit","Apply"});require(operation.at("Example").is_number_integer()&&(operation.at("Example").is_number_unsigned()||operation.at("Example").get<std::int64_t>()>=0)&&operation.at("Example").get<std::uint64_t>()<LLAMA_EXAMPLE_COUNT,"resolution example outside enum");require(operation.at("OfflineAfterInit").is_boolean()&&operation.at("Apply").is_boolean(),"resolution handler flags must be boolean");const auto&args=operation.at("Arguments");require(args.is_array()&&!args.empty()&&args.size()<=64,"resolution argv count outside bound");std::size_t total=0;for(const auto&arg:args){text(arg,16384);total+=arg.get_ref<const std::string&>().size();require(total<=65536,"resolution argv bytes exceed64KiB");}
}
inline json file(const hf_cache::hf_file&value){return json{{"Path",value.path},{"Url",value.url},{"LocalPath",value.local_path},{"FinalPath",value.final_path},{"Oid",value.oid},{"Repo",value.repo_id}};}
inline json plan(const common_download_hf_plan&value){json files=json::array();for(const auto&item:value.model_files)files.push_back(file(item));return json{{"Primary",file(value.primary)},{"ModelFiles",files},{"Mmproj",file(value.mmproj)},{"Mtp",file(value.mtp)},{"Dflash",file(value.dflash)},{"Eagle3",file(value.eagle3)},{"Dspark",file(value.dspark)},{"Preset",file(value.preset)}};}
inline json snapshot(const common_params&value){return json{{"ModelPath",value.model.path},{"ModelHfRepo",value.model.hf_repo},{"MmprojPath",value.mmproj.path},{"DraftPath",value.speculative.draft.mparams.path},{"SpecTypes",value.speculative.types},{"Preset",value.models_preset},{"Offline",value.offline}};}
inline json execute(const json&operation){
    validate(operation);const auto op=operation.at("Op").get<std::string>();
    if(op=="resolution-pause-log"){common_log_pause(common_log_main());return nullptr;}
    if(op=="resolution-plan"){
        common_params_model model;model.hf_repo=operation.at("HfRepo").get<std::string>();model.hf_file=operation.at("HfFile").get<std::string>();common_download_opts opts;const auto&o=operation.at("Options");opts.download_mmproj=o.at("Mmproj").get<bool>();opts.download_mtp=o.at("Mtp").get<bool>();opts.download_dflash=o.at("Dflash").get<bool>();opts.download_eagle3=o.at("Eagle3").get<bool>();opts.download_dspark=o.at("Dspark").get<bool>();return plan(common_download_get_hf_plan(model,opts));
    }
    common_params params;std::vector<std::string>storage;for(const auto&value:operation.at("Arguments"))storage.push_back(value.get<std::string>());std::vector<char*>argv;for(auto&value:storage)argv.push_back(value.data());const auto example=static_cast<llama_example>(operation.at("Example").get<int>());const bool parsed=common_params_parse(static_cast<int>(argv.size()),argv.data(),params,example);if(!parsed)return json{{"Parsed",false},{"Params",snapshot(params)},{"Plan",nullptr},{"DraftPlan",nullptr}};
    auto handler=common_models_handler_init(params,example);params.offline=operation.at("OfflineAfterInit").get<bool>();if(operation.at("Apply").get<bool>())common_models_handler_apply(handler,params);return json{{"Parsed",true},{"Params",snapshot(params)},{"Plan",plan(handler.plan)},{"DraftPlan",plan(handler.plan_spec)}};
}
} // namespace prism_model_resolution
