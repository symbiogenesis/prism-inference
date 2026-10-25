// Typed construction, serialization and production MTMD calls only.
// The safe Rust suite owns every input specification and expected result.
#pragma once

#include "mtmd-image.h"
#include "mtmd-internal.h"
#include <nlohmann/json.hpp>

#include <cstdint>
#include <cstddef>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace prism_mtmd {
using json = nlohmann::ordered_json;

inline void require(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

inline void require_keys(const json & object, std::initializer_list<const char *> keys) {
    require(object.is_object() && object.size() == keys.size(), "invalid MTMD object fields");
    for (const auto * key : keys) {
        require(object.contains(key), "missing or unknown MTMD object field");
    }
}

inline int positive(const json & value, int maximum) {
    require(value.is_number_integer(), "MTMD field must be a positive integer");
    if (value.is_number_unsigned()) {
        const auto number = value.get<std::uint64_t>();
        require(number > 0 && number <= static_cast<std::uint64_t>(maximum), "MTMD integer outside bound");
        return static_cast<int>(number);
    }
    const auto number = value.get<std::int64_t>();
    require(number > 0 && number <= maximum, "MTMD integer outside bound");
    return static_cast<int>(number);
}

inline bool handles(const std::string & kind) {
    return kind == "mtmd-image-should-tile" || kind == "mtmd-group-bitmaps";
}

inline void validate(const json & operation) {
    require(operation.is_object() && operation.contains("Kind") && operation.at("Kind").is_string(),
            "MTMD Kind must be a string");
    const auto & kind = operation.at("Kind").get_ref<const std::string &>();
    if (kind == "mtmd-image-should-tile") {
        require_keys(operation, {"Kind", "PatchSize", "Merge", "MinImageTokens", "MaxImageTokens", "Width", "Height"});
        positive(operation.at("PatchSize"), 64);
        positive(operation.at("Merge"), 8);
        const auto minimum = positive(operation.at("MinImageTokens"), 4096);
        const auto maximum = positive(operation.at("MaxImageTokens"), 4096);
        require(minimum <= maximum, "MTMD token bounds must be ordered");
        positive(operation.at("Width"), 4096);
        positive(operation.at("Height"), 4096);
        // The caps also keep the production int patch-area/pixel arithmetic
        // below INT32_MAX: 64^2 * 8^2 * 4096 = 1,073,741,824.
    } else if (kind == "mtmd-group-bitmaps") {
        require_keys(operation, {"Kind", "Merge", "Parts"});
        positive(operation.at("Merge"), 64);
        const auto & parts = operation.at("Parts");
        require(parts.is_array() && parts.size() <= 64, "MTMD Parts must be an array of at most 64 parts");
        for (const auto & part : parts) {
            require(part.is_object() && part.contains("Kind") && part.at("Kind").is_string(),
                    "MTMD part Kind must be a string");
            const auto & part_kind = part.at("Kind").get_ref<const std::string &>();
            if (part_kind == "text") {
                require_keys(part, {"Kind", "Text"});
                require(part.at("Text").is_string() && part.at("Text").get_ref<const std::string &>().size() <= 65536,
                        "MTMD Text must be a string of at most 65536 bytes");
            } else if (part_kind == "bitmap") {
                require_keys(part, {"Kind", "Width", "Height", "Mergeable"});
                positive(part.at("Width"), 4096);
                positive(part.at("Height"), 4096);
                require(part.at("Mergeable").is_boolean(), "MTMD Mergeable must be boolean");
            } else if (part_kind == "audio") {
                require_keys(part, {"Kind", "Samples", "Mergeable"});
                positive(part.at("Samples"), 1048576);
                require(part.at("Mergeable").is_boolean(), "MTMD Mergeable must be boolean");
            } else {
                throw std::runtime_error("unknown MTMD part Kind");
            }
        }
    } else {
        throw std::runtime_error("unknown MTMD operation Kind");
    }
}

inline json execute(const json & operation) {
    if (operation.at("Kind") == "mtmd-image-should-tile") {
        clip_hparams hparams;
        hparams.patch_size = positive(operation.at("PatchSize"), 64);
        hparams.n_merge = positive(operation.at("Merge"), 8);
        hparams.set_limit_image_tokens(positive(operation.at("MinImageTokens"), 4096),
                positive(operation.at("MaxImageTokens"), 4096));
        const clip_image_size size = {positive(operation.at("Width"), 4096), positive(operation.at("Height"), 4096)};
        return json{{"ShouldTile", mtmd_image_preprocessor_lfm2::should_tile(hparams, size)}};
    }

    std::vector<mtmd::bitmap_ptr> pool;
    std::vector<mtmd_input_part> parts;
    for (const auto & part : operation.at("Parts")) {
        const auto & kind = part.at("Kind").get_ref<const std::string &>();
        if (kind == "text") {
            parts.push_back({part.at("Text").get<std::string>(), nullptr});
        } else {
            mtmd::bitmap_ptr bitmap(kind == "audio"
                    ? mtmd_bitmap_init_from_audio(static_cast<std::size_t>(positive(part.at("Samples"), 1048576)), nullptr)
                    : mtmd_bitmap_init(static_cast<std::uint32_t>(positive(part.at("Width"), 4096)),
                            static_cast<std::uint32_t>(positive(part.at("Height"), 4096)), nullptr));
            require(bool(bitmap), "MTMD bitmap construction failed");
            mtmd_bitmap_set_mergeable(bitmap.get(), part.at("Mergeable").get<bool>());
            parts.push_back({"", bitmap.get()});
            pool.push_back(std::move(bitmap));
        }
    }
    const auto groups = mtmd_group_mergeable_bitmaps(parts, positive(operation.at("Merge"), 64));
    json sizes = json::array();
    for (const auto & group : groups) {
        sizes.push_back(group.size());
    }
    std::size_t remaining = 0;
    for (const auto & part : parts) {
        remaining += part.bitmap != nullptr ? 1 : 0;
    }
    return json{{"GroupSizes", sizes}, {"RemainingBitmapParts", remaining}};
}
} // namespace prism_mtmd
