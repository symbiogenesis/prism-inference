#pragma once
// Bounded descriptors and direct production optimizer calls. Rust owns datasets,
// graph descriptions, parameter schedules, training schedules and every verdict.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-opt.h"
#include <nlohmann/json.hpp>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace prism_optimizer {
using json = nlohmann::ordered_json;
inline void require(bool condition, const char * message) { if (!condition) throw std::runtime_error(message); }
inline void fields(const json & value, std::initializer_list<const char *> names) {
    require(value.is_object() && value.size() == names.size(), "optimizer fields mismatch");
    for (const auto * name : names) require(value.contains(name), "missing optimizer field");
}
inline std::int64_t integer(const json & value, std::int64_t low, std::int64_t high) {
    require(value.is_number_integer(), "optimizer integer required");
    if (value.is_number_unsigned()) require(value.get<std::uint64_t>() <= std::uint64_t(high), "optimizer integer exceeds bound");
    const auto number = value.get<std::int64_t>(); require(number >= low && number <= high, "optimizer integer outside bound"); return number;
}
inline int handle(const json & value) { return static_cast<int>(integer(value, 0, 4095)); }
inline bool boolean(const json & value) { require(value.is_boolean(), "optimizer boolean required"); return value.get<bool>(); }
inline float floating(const json & value) {
    const auto bits = static_cast<std::uint32_t>(integer(value, 0, UINT32_MAX)); float number; std::memcpy(&number, &bits, 4);
    require(std::isfinite(number), "optimizer input float must be finite"); return number;
}
inline std::uint32_t bits(float value) { std::uint32_t result; std::memcpy(&result, &value, 4); return result; }
inline std::uint64_t bits(double value) { std::uint64_t result; std::memcpy(&result, &value, 8); return result; }
inline std::vector<std::uint8_t> unhex(const json & value) {
    require(value.is_string(), "optimizer data must be hex"); const auto & text = value.get_ref<const std::string &>();
    require(text.size() % 2 == 0 && text.size() <= 2*1024*1024, "optimizer hex extent exceeds bound");
    const auto nibble = [](char c) { require((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'), "optimizer hex must be lowercase"); return c <= '9' ? c-'0' : c-'a'+10; };
    std::vector<std::uint8_t> result(text.size()/2);
    for (std::size_t i = 0; i < result.size(); ++i) result[i] = static_cast<std::uint8_t>((nibble(text[2*i])<<4)|nibble(text[2*i+1]));
    return result;
}
inline std::string hex(const void * data, std::size_t size) {
    require(size <= 1024*1024, "optimizer output bytes exceed bound"); const auto * bytes = static_cast<const std::uint8_t *>(data); constexpr char digits[] = "0123456789abcdef";
    std::string result(size*2, '\0'); for (std::size_t i = 0; i < size; ++i) { result[2*i] = digits[bytes[i]>>4]; result[2*i+1] = digits[bytes[i]&15]; } return result;
}
inline ggml_opt_optimizer_params parameters(const json & value) {
    fields(value, {"AdamW", "Sgd"}); const auto & adam = value.at("AdamW"), & sgd = value.at("Sgd");
    require(adam.is_array() && adam.size() == 5 && sgd.is_array() && sgd.size() == 2, "optimizer parameter extent mismatch");
    auto result = ggml_opt_get_default_optimizer_params(nullptr);
    result.adamw.alpha = floating(adam[0]); result.adamw.beta1 = floating(adam[1]); result.adamw.beta2 = floating(adam[2]); result.adamw.eps = floating(adam[3]); result.adamw.wd = floating(adam[4]);
    result.sgd.alpha = floating(sgd[0]); result.sgd.wd = floating(sgd[1]); return result;
}
inline json encode(const ggml_opt_optimizer_params & p) {
    return json{{"AdamW", {bits(p.adamw.alpha),bits(p.adamw.beta1),bits(p.adamw.beta2),bits(p.adamw.eps),bits(p.adamw.wd)}}, {"Sgd", {bits(p.sgd.alpha),bits(p.sgd.wd)}}};
}
inline bool handles(const std::string & kind) { return kind == "optimizer-info" || kind == "optimizer-trace"; }
inline void validate(const json & operation) {
    if (operation.at("Kind") == "optimizer-info") { fields(operation, {"Kind"}); return; }
    fields(operation, {"Kind", "Devices", "Scheduler", "Backend", "Threads", "Actions"});
    const auto & devices = operation.at("Devices"), & scheduler = operation.at("Scheduler"), & actions = operation.at("Actions");
    require(devices.is_array() && !devices.empty() && devices.size() <= 8, "optimizer devices exceed bound");
    for (const auto & device : devices) integer(device, 0, 63);
    integer(operation.at("Backend"), 0, static_cast<std::int64_t>(devices.size()-1)); integer(operation.at("Threads"), 0, 256);
    require(scheduler.is_array() && !scheduler.empty() && scheduler.size() <= 16, "optimizer scheduler exceeds bound");
    for (const auto & index : scheduler) integer(index, 0, static_cast<std::int64_t>(devices.size()-1));
    require(actions.is_array() && actions.size() <= 8192, "optimizer actions exceed bound");
    for (const auto & action : actions) {
        require(action.is_object() && action.contains("Op") && action.at("Op").is_string(), "optimizer action Op required");
        const auto op = action.at("Op").get<std::string>();
        if (op == "context") { fields(action, {"Op", "Handle", "Bytes"}); handle(action.at("Handle")); integer(action.at("Bytes"), 1, 8*1024*1024); }
        else if (op == "tensor") {
            fields(action, {"Op", "Handle", "Context", "Dimensions", "Parameter", "Name"}); handle(action.at("Handle")); handle(action.at("Context")); boolean(action.at("Parameter"));
            require(action.at("Name").is_string() && action.at("Name").get_ref<const std::string &>().size() <= 128 && action.at("Name").get_ref<const std::string &>().find('\0') == std::string::npos, "optimizer tensor name invalid");
            const auto & dims = action.at("Dimensions"); require(dims.is_array() && !dims.empty() && dims.size() <= 4, "optimizer dimensions invalid"); std::int64_t elements = 1;
            for (const auto & dim : dims) { elements *= integer(dim, 1, 65536); require(elements <= 262144, "optimizer tensor elements exceed bound"); }
        } else if (op == "node") {
            fields(action, {"Op", "Handle", "Context", "Operation", "Inputs", "FactorBits", "Name"}); handle(action.at("Handle")); handle(action.at("Context"));
            require(action.at("Operation").is_string(), "optimizer node operation invalid"); const auto name = action.at("Operation").get<std::string>();
            const std::size_t count = name == "add" || name == "mul" ? 2 : name == "scale" ? 1 : name == "adamw-step" ? 5 : name == "sgd-step" ? 3 : 0;
            const auto & inputs = action.at("Inputs"); require(count && inputs.is_array() && inputs.size() == count, "optimizer node input count invalid"); for (const auto & input : inputs) handle(input);
            if (name == "scale") floating(action.at("FactorBits")); else require(action.at("FactorBits").is_null(), "non-scale factor must be null");
            require(action.at("Name").is_string() && action.at("Name").get_ref<const std::string &>().size() <= 128, "optimizer node name invalid");
        } else if (op == "allocate") { fields(action, {"Op", "Handle", "Context"}); handle(action.at("Handle")); handle(action.at("Context")); }
        else if (op == "set") { fields(action, {"Op", "Tensor", "DataHex"}); handle(action.at("Tensor")); (void) unhex(action.at("DataHex")); }
        else if (op == "save" || op == "restore") { fields(action, {"Op", "Tensor", "Handle"}); handle(action.at("Tensor")); handle(action.at("Handle")); }
        else if (op == "get" || op == "supports") { fields(action, {"Op", "Tensor"}); handle(action.at("Tensor")); }
        else if (op == "dataset") {
            fields(action, {"Op", "Handle", "DatapointElements", "LabelElements", "Count", "Shard", "DataHex", "LabelsHex"}); handle(action.at("Handle"));
            const auto count = integer(action.at("Count"), 1, 4096), data = integer(action.at("DatapointElements"), 1, 64), label = integer(action.at("LabelElements"), 0, 64); integer(action.at("Shard"), 1, count);
            require(unhex(action.at("DataHex")).size() == std::size_t(count*data*4) && unhex(action.at("LabelsHex")).size() == std::size_t(count*label*4), "optimizer dataset bytes mismatch");
        } else if (op == "params") { fields(action, {"Op", "Loss"}); integer(action.at("Loss"), 0, 3); }
        else if (op == "init") {
            fields(action, {"Op", "Handle", "Context", "Inputs", "Outputs", "Loss", "Optimizer", "Period", "Parameters"}); for (const auto * key : {"Handle","Context","Inputs","Outputs"}) handle(action.at(key));
            integer(action.at("Loss"), 0, 3); integer(action.at("Optimizer"), 0, 1); integer(action.at("Period"), 1, 1000000); if (!action.at("Parameters").is_null()) (void) parameters(action.at("Parameters"));
        } else if (op == "result") { fields(action, {"Op", "Handle"}); handle(action.at("Handle")); }
        else if (op == "result-reset") { fields(action, {"Op", "Result"}); handle(action.at("Result")); }
        else if (op == "result-snapshot") { fields(action, {"Op", "Result", "LossUncertainty", "AccuracyUncertainty"}); handle(action.at("Result")); boolean(action.at("LossUncertainty")); boolean(action.at("AccuracyUncertainty")); }
        else if (op == "alloc" || op == "reset") { fields(action, {"Op", "Optimizer", "Value"}); handle(action.at("Optimizer")); boolean(action.at("Value")); }
        else if (op == "eval") { fields(action, {"Op", "Optimizer", "Result"}); handle(action.at("Optimizer")); if (!action.at("Result").is_null()) handle(action.at("Result")); }
        else if (op == "loss" || op == "gradient") {
            if (op == "loss") fields(action, {"Op", "Optimizer", "Handle"}); else { fields(action, {"Op", "Optimizer", "Handle", "Tensor"}); handle(action.at("Tensor")); }
            handle(action.at("Optimizer")); handle(action.at("Handle"));
        } else if (op == "shuffle") { fields(action, {"Op", "Optimizer", "Dataset", "Count"}); handle(action.at("Optimizer")); handle(action.at("Dataset")); integer(action.at("Count"), -1, 4096); }
        else if (op == "batch") { fields(action, {"Op", "Dataset", "Data", "Labels", "Index"}); for (const auto * key : {"Dataset","Data","Labels"}) handle(action.at(key)); integer(action.at("Index"), 0, 4095); }
        else if (op == "epoch") {
            fields(action, {"Op", "Optimizer", "Dataset", "Train", "Evaluate", "Split"}); handle(action.at("Optimizer")); handle(action.at("Dataset")); for (const auto * key : {"Train","Evaluate"}) if (!action.at(key).is_null()) handle(action.at(key)); integer(action.at("Split"), 0, 4096);
        } else if (op == "fit") {
            fields(action, {"Op", "Context", "Inputs", "Outputs", "Dataset", "Loss", "Optimizer", "Epochs", "LogicalBatch", "ValidationBits", "Silent", "Parameters"});
            for (const auto * key : {"Context","Inputs","Outputs","Dataset"}) handle(action.at(key)); integer(action.at("Loss"), 0, 3); integer(action.at("Optimizer"), 0, 1); const auto epochs = integer(action.at("Epochs"), 1, 4096); integer(action.at("LogicalBatch"), 1, 4096);
            const auto split = floating(action.at("ValidationBits")); require(split >= 0 && split < 1, "optimizer validation split invalid"); boolean(action.at("Silent"));
            const auto & values = action.at("Parameters"); if (!values.is_null()) { require(values.is_array() && values.size() == std::size_t(epochs), "optimizer fit parameter schedule mismatch"); for (const auto & value : values) (void) parameters(value); }
        } else if (op == "free") { fields(action, {"Op", "Type", "Handle"}); handle(action.at("Handle")); require(action.at("Type").is_string(), "optimizer free type required"); }
        else throw std::runtime_error("unknown optimizer action");
    }
}
struct dataset { ggml_opt_dataset_t value; std::int64_t count, data, labels, shard; };
struct optimizer { ggml_opt_context_t value = nullptr; ggml_opt_optimizer_params params{}; int context = -1; };
struct tensor { ggml_tensor * value; int context; int optimizer = -1; };
struct state {
    std::vector<ggml_backend_t> backends;
    ggml_backend_sched_t scheduler = nullptr;
    std::map<int, ggml_context *> contexts;
    std::map<int, tensor> tensors;
    std::map<int, ggml_backend_buffer_t> buffers;
    std::map<int, int> buffer_contexts;
    std::set<int> unavailable_storage;
    std::map<int, dataset> datasets;
    std::map<int, std::unique_ptr<optimizer>> optimizers;
    std::map<int, ggml_opt_result_t> results;
    std::map<int, std::vector<std::uint8_t>> saved;
    ~state() {
        for (auto & p : optimizers) ggml_opt_free(p.second->value);
        for (auto & p : results) ggml_opt_result_free(p.second);
        for (auto & p : buffers) ggml_backend_buffer_free(p.second);
        for (auto & p : contexts) ggml_free(p.second);
        for (auto & p : datasets) ggml_opt_dataset_free(p.second.value);
        if (scheduler) ggml_backend_sched_free(scheduler);
        for (auto backend : backends) ggml_backend_free(backend);
    }
    ggml_tensor * t(const json & id) { const auto value = tensors.find(handle(id)); require(value != tensors.end() && value->second.value, "unknown optimizer tensor"); return value->second.value; }
    ggml_tensor * storage(const json & id) { require(!unavailable_storage.count(tensors.at(handle(id)).context), "optimizer tensor storage lifetime ended or not allocated"); return t(id); }
    ggml_context * c(const json & id) { return contexts.at(handle(id)); }
    ggml_opt_context_t o(const json & id) { return optimizers.at(handle(id))->value; }
    ggml_opt_result_t r(const json & id) { return id.is_null() ? nullptr : results.at(handle(id)); }
    dataset & d(const json & id) { return datasets.at(handle(id)); }
};
inline thread_local const std::vector<ggml_opt_optimizer_params> * fit_parameters = nullptr;
inline thread_local bool fit_callback_failed = false;
inline ggml_opt_optimizer_params scheduled_parameters(void * userdata) noexcept {
    const auto epoch = userdata ? *static_cast<std::int64_t *>(userdata) : 0;
    if (!fit_parameters || epoch < 1 || std::size_t(epoch) > fit_parameters->size()) { fit_callback_failed = true; return ggml_opt_get_default_optimizer_params(nullptr); }
    return (*fit_parameters)[std::size_t(epoch)-1];
}
inline json execute(const json & operation) {
    ggml_backend_load_all();
    if (operation.at("Kind") == "optimizer-info") {
        json devices = json::array();
        for (std::size_t i = 0; i < ggml_backend_dev_count(); ++i) { const auto device = ggml_backend_dev_get(i); devices.push_back({{"Index", i}, {"Name", ggml_backend_dev_name(device)}, {"Type", static_cast<int>(ggml_backend_dev_type(device))}}); }
        return json{{"Devices", devices}, {"HardwareConcurrency", std::thread::hardware_concurrency()}, {"Defaults", encode(ggml_opt_get_default_optimizer_params(nullptr))}, {"TensorOverhead", ggml_tensor_overhead()}, {"GraphOverhead", ggml_graph_overhead()}, {"DefaultGraphSize", GGML_DEFAULT_GRAPH_SIZE}};
    }
    state s;
    for (const auto & id : operation.at("Devices")) {
        const auto index = std::size_t(integer(id, 0, 63)); require(index < ggml_backend_dev_count(), "optimizer device missing"); const auto device = ggml_backend_dev_get(index);
        auto backend = ggml_backend_dev_init(device, nullptr); require(backend != nullptr, "production optimizer backend init failed"); s.backends.push_back(backend);
        auto set_threads = reinterpret_cast<ggml_backend_set_n_threads_t>(ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(device), "ggml_backend_set_n_threads")); if (set_threads) set_threads(backend, static_cast<int>(integer(operation.at("Threads"), 0, 256)));
    }
    std::vector<ggml_backend_t> order; for (const auto & index : operation.at("Scheduler")) order.push_back(s.backends[std::size_t(integer(index, 0, 7))]);
    s.scheduler = ggml_backend_sched_new(order.data(), nullptr, order.size(), GGML_DEFAULT_GRAPH_SIZE, false, true); require(s.scheduler != nullptr, "production optimizer scheduler init failed");
    const auto backend = s.backends[std::size_t(integer(operation.at("Backend"), 0, 7))]; json trace = json::array();
    for (const auto & a : operation.at("Actions")) {
        const auto op = a.at("Op").get<std::string>(); json event = json::object();
        if (op == "context") { const int id = handle(a.at("Handle")); require(s.contexts.size() < 16 && !s.contexts.count(id), "optimizer context exists or count exceeds bound"); auto * c = ggml_init({std::size_t(integer(a.at("Bytes"), 1, 8*1024*1024)), nullptr, true}); require(c != nullptr, "production ggml_init failed"); s.contexts[id] = c; s.unavailable_storage.insert(id); event = {{"Created", true}}; }
        else if (op == "tensor") {
            const int id = handle(a.at("Handle")); require(s.tensors.size() < 256 && !s.tensors.count(id), "optimizer tensor exists or count exceeds bound"); std::int64_t dims[4] = {1,1,1,1}; for (std::size_t i = 0; i < a.at("Dimensions").size(); ++i) dims[i] = a.at("Dimensions")[i].get<std::int64_t>();
            auto * t = ggml_new_tensor(s.c(a.at("Context")), GGML_TYPE_F32, static_cast<int>(a.at("Dimensions").size()), dims); if (boolean(a.at("Parameter"))) ggml_set_param(t); ggml_set_name(t, a.at("Name").get<std::string>().c_str()); s.tensors[id] = {t, handle(a.at("Context"))};
        } else if (op == "node") {
            const int id = handle(a.at("Handle")); require(s.tensors.size() < 256 && !s.tensors.count(id), "optimizer node exists or count exceeds bound"); const auto name = a.at("Operation").get<std::string>(); const auto & ids = a.at("Inputs"); auto * ctx = s.c(a.at("Context")); auto * first = s.t(ids[0]); ggml_tensor * value = nullptr;
            if (name == "add" || name == "mul") { auto * second = s.t(ids[1]); require(ggml_can_repeat(second, first), "optimizer broadcast shape mismatch"); value = name == "add" ? ggml_add(ctx, first, second) : ggml_mul(ctx, first, second); }
            else if (name == "scale") value = ggml_scale(ctx, first, floating(a.at("FactorBits")));
            else if (name == "adamw-step") { require(ggml_are_same_shape(first, s.t(ids[1])) && ggml_are_same_shape(first, s.t(ids[2])) && ggml_are_same_shape(first, s.t(ids[3])) && ggml_nelements(s.t(ids[4])) == 7, "optimizer AdamW step shape mismatch"); value = ggml_opt_step_adamw(ctx, first, s.t(ids[1]), s.t(ids[2]), s.t(ids[3]), s.t(ids[4])); }
            else { require(ggml_are_same_shape(first, s.t(ids[1])) && ggml_nelements(s.t(ids[2])) == 2, "optimizer SGD step shape mismatch"); value = ggml_opt_step_sgd(ctx, first, s.t(ids[1]), s.t(ids[2])); }
            ggml_set_name(value, a.at("Name").get<std::string>().c_str()); s.tensors[id] = {value, handle(a.at("Context"))};
        } else if (op == "allocate") { const int id = handle(a.at("Handle")); require(s.buffers.size() < 16 && !s.buffers.count(id), "optimizer buffer exists or count exceeds bound"); const auto context = handle(a.at("Context")); const auto value = ggml_backend_alloc_ctx_tensors(s.c(a.at("Context")), backend); require(value != nullptr, "production optimizer tensor allocation failed"); s.buffers[id] = value; s.buffer_contexts[id] = context; s.unavailable_storage.erase(context); }
        else if (op == "set") { const auto bytes = unhex(a.at("DataHex")); auto * t = s.storage(a.at("Tensor")); require(t->data && bytes.size() == ggml_nbytes(t), "optimizer tensor set extent/storage mismatch"); ggml_backend_tensor_set(t, bytes.data(), 0, bytes.size()); }
        else if (op == "save" || op == "restore") { auto * t = s.storage(a.at("Tensor")); const auto id = handle(a.at("Handle")); require(t->data && ggml_nbytes(t) <= 1024*1024, "optimizer saved tensor extent/storage invalid"); if (op == "save") { require(s.saved.size() < 16 || s.saved.count(id), "optimizer saved tensor count exceeds bound"); auto & bytes = s.saved[id]; bytes.resize(ggml_nbytes(t)); ggml_backend_tensor_get(t, bytes.data(), 0, bytes.size()); } else { const auto & bytes = s.saved.at(id); require(bytes.size() == ggml_nbytes(t), "optimizer restore extent mismatch"); ggml_backend_tensor_set(t, bytes.data(), 0, bytes.size()); } }
        else if (op == "get") { auto * t = s.storage(a.at("Tensor")); require(t->data && ggml_nbytes(t) <= 1024*1024, "optimizer get storage/extent mismatch"); std::vector<std::uint8_t> bytes(ggml_nbytes(t)); ggml_backend_tensor_get(t, bytes.data(), 0, bytes.size()); event = {{"DataHex", hex(bytes.data(), bytes.size())}}; }
        else if (op == "supports") event = {{"Supported", ggml_backend_supports_op(backend, s.t(a.at("Tensor")))}};
        else if (op == "dataset") {
            const int id = handle(a.at("Handle")); require(s.datasets.size() < 16 && !s.datasets.count(id), "optimizer dataset exists or count exceeds bound"); const auto count = a.at("Count").get<std::int64_t>(), data = a.at("DatapointElements").get<std::int64_t>(), labels = a.at("LabelElements").get<std::int64_t>(), shard = a.at("Shard").get<std::int64_t>();
            auto value = ggml_opt_dataset_init(GGML_TYPE_F32, GGML_TYPE_F32, data, labels, count, shard); require(value != nullptr, "production optimizer dataset init failed"); s.datasets[id] = {value,count,data,labels,shard}; const auto raw = unhex(a.at("DataHex")), label = unhex(a.at("LabelsHex")); std::memcpy(ggml_get_data(ggml_opt_dataset_data(value)), raw.data(), raw.size()); if (labels) std::memcpy(ggml_get_data(ggml_opt_dataset_labels(value)), label.data(), label.size()); event = {{"Count", ggml_opt_dataset_ndata(value)}};
        } else if (op == "params") { const auto p = ggml_opt_default_params(s.scheduler, static_cast<ggml_opt_loss_type>(a.at("Loss").get<int>())); event = {{"HasCallback", p.get_opt_pars != nullptr}}; }
        else if (op == "init") {
            const int id = handle(a.at("Handle")); require(s.optimizers.size() < 16 && !s.optimizers.count(id), "optimizer handle exists or count exceeds bound"); auto record = std::make_unique<optimizer>(); record->context = handle(a.at("Context")); auto p = ggml_opt_default_params(s.scheduler, static_cast<ggml_opt_loss_type>(a.at("Loss").get<int>())); p.ctx_compute = s.c(a.at("Context")); p.inputs = s.t(a.at("Inputs")); p.outputs = s.t(a.at("Outputs")); p.opt_period = a.at("Period").get<int>(); p.optimizer = static_cast<ggml_opt_optimizer_type>(a.at("Optimizer").get<int>());
            if (!a.at("Parameters").is_null()) { record->params = parameters(a.at("Parameters")); p.get_opt_pars = ggml_opt_get_constant_optimizer_params; p.get_opt_pars_ud = &record->params; }
            record->value = ggml_opt_init(p); require(record->value != nullptr, "production optimizer init failed"); event = {{"Created", true}, {"Optimizer", static_cast<int>(ggml_opt_context_optimizer_type(record->value))}, {"HasCallback", p.get_opt_pars != nullptr}}; s.optimizers[id] = std::move(record);
        } else if (op == "result") { const int id = handle(a.at("Handle")); require(s.results.size() < 16 && !s.results.count(id), "optimizer result exists or count exceeds bound"); auto value = ggml_opt_result_init(); require(value != nullptr, "production optimizer result init failed"); s.results[id] = value; }
        else if (op == "result-reset") ggml_opt_result_reset(s.r(a.at("Result")));
        else if (op == "result-snapshot") { auto result = s.r(a.at("Result")); std::int64_t count; double loss, loss_unc, accuracy, accuracy_unc; ggml_opt_result_ndata(result, &count); const bool lu = boolean(a.at("LossUncertainty")), au = boolean(a.at("AccuracyUncertainty")); ggml_opt_result_loss(result, &loss, lu ? &loss_unc : nullptr); ggml_opt_result_accuracy(result, &accuracy, au ? &accuracy_unc : nullptr); event = {{"Count",count}, {"LossBits",bits(loss)}, {"LossUncertaintyBits",lu ? json(bits(loss_unc)) : json(nullptr)}, {"AccuracyBits",bits(accuracy)}, {"AccuracyUncertaintyBits",au ? json(bits(accuracy_unc)) : json(nullptr)}}; }
        else if (op == "alloc") { ggml_opt_alloc(s.o(a.at("Optimizer")), boolean(a.at("Value"))); s.unavailable_storage.erase(s.optimizers.at(handle(a.at("Optimizer")))->context); }
        else if (op == "reset") ggml_opt_reset(s.o(a.at("Optimizer")), boolean(a.at("Value")));
        else if (op == "eval") ggml_opt_eval(s.o(a.at("Optimizer")), s.r(a.at("Result")));
        else if (op == "loss" || op == "gradient") { const int id = handle(a.at("Handle")); const auto index = handle(a.at("Optimizer")); auto * value = op == "loss" ? ggml_opt_loss(s.o(a.at("Optimizer"))) : ggml_opt_grad_acc(s.o(a.at("Optimizer")), s.t(a.at("Tensor"))); require(value != nullptr, "production optimizer observation tensor missing"); s.tensors[id] = {value,s.optimizers.at(index)->context,index}; }
        else if (op == "shuffle") { auto & d = s.d(a.at("Dataset")); const auto count = a.at("Count").get<std::int64_t>(); require(count < 0 || count <= d.count, "optimizer shuffle count exceeds dataset"); ggml_opt_dataset_shuffle(s.o(a.at("Optimizer")), d.value, count); }
        else if (op == "batch") { auto & d = s.d(a.at("Dataset")); auto * data = s.storage(a.at("Data")), * labels = s.storage(a.at("Labels")); const auto count = ggml_nelements(data)/d.data, index = a.at("Index").get<std::int64_t>(); require(data->data && labels->data && ggml_nelements(data) % d.data == 0 && ggml_nelements(labels) == count*d.labels && count % d.shard == 0 && (index+1)*count <= d.count, "optimizer batch extent/storage mismatch"); ggml_opt_dataset_get_batch(d.value, data, labels, index); }
        else if (op == "epoch") { auto & d = s.d(a.at("Dataset")); const auto split = a.at("Split").get<std::int64_t>(); require(split <= d.count, "optimizer epoch split exceeds dataset"); ggml_opt_epoch(s.o(a.at("Optimizer")), d.value, s.r(a.at("Train")), s.r(a.at("Evaluate")), split, nullptr, nullptr); s.unavailable_storage.erase(s.optimizers.at(handle(a.at("Optimizer")))->context); }
        else if (op == "fit") {
            auto & d = s.d(a.at("Dataset")); const auto logical = a.at("LogicalBatch").get<std::int64_t>(); auto * inputs = s.t(a.at("Inputs")); require(d.count % logical == 0 && logical % inputs->ne[1] == 0, "optimizer fit batch extent mismatch"); std::vector<ggml_opt_optimizer_params> schedule; if (!a.at("Parameters").is_null()) for (const auto & value : a.at("Parameters")) schedule.push_back(parameters(value));
            struct guard { ~guard() { fit_parameters = nullptr; } } cleanup; fit_callback_failed = false; fit_parameters = schedule.empty() ? nullptr : &schedule;
            ggml_opt_fit(s.scheduler, s.c(a.at("Context")), inputs, s.t(a.at("Outputs")), d.value, static_cast<ggml_opt_loss_type>(a.at("Loss").get<int>()), static_cast<ggml_opt_optimizer_type>(a.at("Optimizer").get<int>()), schedule.empty() ? ggml_opt_get_default_optimizer_params : scheduled_parameters, a.at("Epochs").get<std::int64_t>(), logical, floating(a.at("ValidationBits")), boolean(a.at("Silent"))); s.unavailable_storage.insert(handle(a.at("Context"))); require(!fit_callback_failed, "production optimizer callback requested outside supplied schedule");
        } else if (op == "free") {
            const auto type = a.at("Type").get<std::string>(); const int id = handle(a.at("Handle"));
            if (type == "optimizer") { s.unavailable_storage.insert(s.optimizers.at(id)->context); ggml_opt_free(s.optimizers.at(id)->value); s.optimizers.erase(id); for (auto it = s.tensors.begin(); it != s.tensors.end();) if (it->second.optimizer == id) it = s.tensors.erase(it); else ++it; }
            else if (type == "result") { ggml_opt_result_free(s.results.at(id)); s.results.erase(id); }
            else if (type == "dataset") { ggml_opt_dataset_free(s.datasets.at(id).value); s.datasets.erase(id); }
            else if (type == "buffer") { require(s.optimizers.empty(), "optimizer still retains tensor buffers"); s.unavailable_storage.insert(s.buffer_contexts.at(id)); ggml_backend_buffer_free(s.buffers.at(id)); s.buffers.erase(id); s.buffer_contexts.erase(id); }
            else { require(type == "context", "unknown optimizer free type"); for (const auto & item : s.optimizers) require(item.second->context != id, "optimizer still retains context"); for (const auto & item : s.buffer_contexts) require(item.second != id, "optimizer context still has allocated buffer"); ggml_free(s.contexts.at(id)); s.contexts.erase(id); s.unavailable_storage.erase(id); for (auto it = s.tensors.begin(); it != s.tensors.end();) if (it->second.context == id) it = s.tensors.erase(it); else ++it; }
        }
        trace.push_back(std::move(event));
    }
    return json{{"Trace",trace}};
}
} // namespace prism_optimizer
