//! Independent nearest-lattice and run-partition observations for MTMD helpers.
//!
//! No production Rust/native media API or expected-answer corpus is used here.
//! Immutable upstream MIT source identities are disclosed by `identity`.

use serde::Deserialize;
use serde_json::{Value, json};

use super::{Result, invalid};

pub(super) const ID: &str = "tests/test-mtmd-impl.cpp";

#[derive(Deserialize)]
#[serde(rename_all = "PascalCase", deny_unknown_fields)]
struct Request {
    schema_version: u32,
    operations: Vec<Operation>,
}

#[derive(Deserialize)]
#[serde(tag = "Kind", rename_all_fields = "PascalCase", deny_unknown_fields)]
enum Operation {
    #[serde(rename = "mtmd-image-should-tile")]
    Tile {
        patch_size: u32,
        merge: u32,
        min_image_tokens: u32,
        max_image_tokens: u32,
        width: u32,
        height: u32,
    },
    #[serde(rename = "mtmd-group-bitmaps")]
    Group { merge: u32, parts: Vec<Part> },
}

#[derive(Deserialize)]
#[serde(tag = "Kind", rename_all_fields = "PascalCase", deny_unknown_fields)]
enum Part {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "bitmap")]
    Image {
        width: u32,
        height: u32,
        mergeable: bool,
    },
    #[serde(rename = "audio")]
    Audio {
        samples: u32,
        // The admitted protocol carries this flag for both kinds of bitmap.
        // Upstream explicitly refuses audio merging regardless of its value.
        #[serde(rename = "Mergeable")]
        _mergeable: bool,
    },
}

impl Operation {
    fn validate(&self) -> Result<()> {
        let valid = match self {
            Self::Tile {
                patch_size,
                merge,
                min_image_tokens,
                max_image_tokens,
                width,
                height,
            } => {
                (1..=64).contains(patch_size)
                    && (1..=8).contains(merge)
                    && (1..=4096).contains(min_image_tokens)
                    && (*min_image_tokens..=4096).contains(max_image_tokens)
                    && (1..=4096).contains(width)
                    && (1..=4096).contains(height)
            }
            Self::Group { merge, parts } => {
                (1..=64).contains(merge)
                    && parts.len() <= 64
                    && parts.iter().all(|part| match part {
                        Part::Text { text } => text.len() <= 65_536,
                        Part::Image { width, height, .. } => {
                            (1..=4096).contains(width) && (1..=4096).contains(height)
                        }
                        Part::Audio { samples, .. } => (1..=1_048_576).contains(samples),
                    })
            }
        };
        if valid {
            Ok(())
        } else {
            Err(invalid("independent MTMD bounds or token order violation"))
        }
    }

    fn observe(&self) -> Value {
        match self {
            Self::Tile {
                patch_size,
                merge,
                max_image_tokens,
                width,
                height,
                ..
            } => {
                let align = patch_size * merge;
                let area = nearest_lattice(*width, align, *patch_size)
                    * nearest_lattice(*height, align, *patch_size);
                let allowance = u64::from(align).pow(2) * u64::from(*max_image_tokens) * 2;
                json!({"ShouldTile": area > allowance})
            }
            Self::Group { merge, parts } => group(parts, *merge),
        }
    }
}

// Minimize distance over the complete bounded alignment lattice. Ties prefer
// an even lattice index. This independently specifies nearbyint's default
// nearest-even result without production's quotient/remainder rounding branch.
fn nearest_lattice(dimension: u32, align: u32, minimum: u32) -> u64 {
    let coordinate = (0..=dimension.div_ceil(align))
        .map(|index| {
            let coordinate = index * align;
            (dimension.abs_diff(coordinate), index % 2, coordinate)
        })
        .min()
        .map_or(0, |(_, _, coordinate)| coordinate);
    u64::from(coordinate.max(minimum))
}

fn merge_signature(part: &Part) -> Option<(u32, u32)> {
    match part {
        Part::Image {
            width,
            height,
            mergeable: true,
        } => Some((*width, *height)),
        _ => None,
    }
}

fn group(parts: &[Part], merge: u32) -> Value {
    let mut groups = Vec::new();
    let mut removed = vec![false; parts.len()];
    let mut start = 0;
    while start < parts.len() {
        if matches!(parts[start], Part::Text { .. }) {
            start += 1;
            continue;
        }
        let mut end = start + 1;
        if merge > 1
            && let Some(signature) = merge_signature(&parts[start])
        {
            while end < parts.len() && merge_signature(&parts[end]) == Some(signature) {
                end += 1;
            }
        }
        // Partition an original equal-frame run into pairs, independent of
        // production's repeated mutation/removal of the next vector element.
        for first in (start..end).step_by(2) {
            let count = (end - first).min(2);
            groups.push(count);
            if count == 2 {
                removed[first + 1] = true;
            }
        }
        start = end;
    }
    // Observe original-vector survivors separately from the group-size result.
    let remaining = parts
        .iter()
        .enumerate()
        .filter(|(index, part)| !removed[*index] && !matches!(part, Part::Text { .. }))
        .count();
    json!({"GroupSizes": groups, "RemainingBitmapParts": remaining})
}

pub(super) fn handles(value: &Value) -> bool {
    value["Operations"].as_array().is_some_and(|operations| {
        !operations.is_empty()
            && operations.iter().all(|operation| {
                matches!(
                    operation["Kind"].as_str(),
                    Some("mtmd-image-should-tile" | "mtmd-group-bitmaps")
                )
            })
    })
}

pub(super) fn execute(value: &Value) -> Result<Value> {
    if !value["Operations"]
        .as_array()
        .is_some_and(|operations| (1..=128).contains(&operations.len()))
    {
        return Err(invalid(
            "independent MTMD requires 1 through 128 operations",
        ));
    }
    let request: Request = serde_json::from_value(value.clone())?;
    if request.schema_version != 1 {
        return Err(invalid("unsupported independent MTMD schema"));
    }
    for operation in &request.operations {
        operation.validate()?;
    }
    let results = request
        .operations
        .iter()
        .map(Operation::observe)
        .collect::<Vec<_>>();
    Ok(json!({"SchemaVersion": 1, "Results": results}))
}

pub(super) fn identity() -> Value {
    json!({"Kind": "independent-rust-mtmd-geometry", "SupportedSourceIds": [ID],
        "UpstreamCommit": super::UPSTREAM,
        "AssertionSource": {"Path": ID, "Bytes": 4854, "Sha256": "b6cffd8cbe193c590cb25b98e480f5db6615f8da3ef7e732ca85b19a2345388b"},
        "ProductionSources": [
            {"Path": "tools/mtmd/mtmd.cpp", "Bytes": 103_968, "Sha256": "9300de0676f5a3100ab67b0194e4999a57b217d12ef5c0c933069d7bf9592925"},
            {"Path": "tools/mtmd/mtmd-image.cpp", "Bytes": 66_160, "Sha256": "e972f926e842d0352c599f3d82277552b6d08783880c9c5676dd957cc6575829"},
            {"Path": "tools/mtmd/clip-model.h", "Bytes": 33_092, "Sha256": "13010c0ab79624d5d68db13572905740a06ecf05fb65833212b39a23f0aa5225"},
            {"Path": "tools/mtmd/mtmd-image.h", "Bytes": 11_105, "Sha256": "34b5d904db52fa3680c04786ad854681579fe9a711dc10f9c0b5fef003ab6596"}],
        "Arithmetic": "nearest-even-lattice-distance-and-original-run-partition-v1",
        "OriginalAssertionCount": 29, "NativeReferenceReplay": false,
        "ProductionRuntimeCalled": false, "CompleteSourceGate": false, "NativeRetirementQualified": false})
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{Engine, Target};

    fn tile() -> Value {
        json!({"Kind": "mtmd-image-should-tile", "PatchSize": 16, "Merge": 2,
            "MinImageTokens": 64, "MaxImageTokens": 256, "Width": 720, "Height": 720})
    }

    fn request(operation: Value) -> Value {
        let mut request = json!({"SchemaVersion": 1});
        request["Operations"] = Value::Array(vec![operation]);
        request
    }

    #[test]
    fn independent_mtmd_preserves_complete_source_and_both_reference_selectors() -> Result<()> {
        for target in [Target::RustReference, Target::Cpp, Target::Rust] {
            let engine = Engine::new(target, None, None, None, None, None)?;
            let report = crate::mtmd::run(&engine)?;
            assert_eq!(report["Id"], ID);
            assert_eq!(report["Status"], "pass", "{report}");
            assert!(
                report["Detail"]
                    .as_str()
                    .is_some_and(|detail| detail.contains("29 original assertions"))
            );
        }
        assert_eq!(identity()["NativeReferenceReplay"], false);
        Ok(())
    }

    #[test]
    fn independent_mtmd_nearest_even_strict_area_and_original_run_survivors() -> Result<()> {
        assert_eq!(nearest_lattice(16, 32, 16), 16);
        assert_eq!(nearest_lattice(48, 32, 16), 64);
        assert_eq!(nearest_lattice(80, 32, 16), 64);
        for (dimension, expected) in [(496, false), (528, false), (560, true)] {
            let mut operation = tile();
            operation["MinImageTokens"] = json!(1);
            operation["MaxImageTokens"] = json!(128);
            operation["Width"] = json!(dimension);
            operation["Height"] = json!(dimension);
            assert_eq!(
                execute(&request(operation))?["Results"][0]["ShouldTile"],
                expected
            );
        }
        assert_eq!(
            execute(&request(
                json!({"Kind":"mtmd-image-should-tile", "PatchSize":64,
            "Merge":8, "MinImageTokens":4096, "MaxImageTokens":4096, "Width":4096, "Height":4096})
            ))?["Results"][0]["ShouldTile"],
            false
        );
        let image = json!({"Kind":"bitmap", "Width":100, "Height":80, "Mergeable":true});
        let mut parts = vec![image.clone(); 5];
        parts.push(json!({"Kind":"text", "Text":"break\0"}));
        parts.push(image);
        parts.push(json!({"Kind":"audio", "Samples":100, "Mergeable":true}));
        let input = request(json!({"Kind":"mtmd-group-bitmaps", "Merge":64, "Parts":parts}));
        let original = input.clone();
        let output = execute(&input)?;
        assert_eq!(
            output["Results"][0],
            json!({"GroupSizes":[2,2,1,1,1], "RemainingBitmapParts":5})
        );
        assert_eq!(input, original);
        assert_eq!(execute(&input)?, output);
        assert_eq!(
            execute(&request(
                json!({"Kind":"mtmd-group-bitmaps", "Merge":1, "Parts":[]})
            ))?["Results"][0],
            json!({"GroupSizes":[], "RemainingBitmapParts":0})
        );
        Ok(())
    }

    #[test]
    fn independent_mtmd_rejects_scalar_shape_and_part_boundary_violations() {
        for (field, value) in [
            ("PatchSize", json!(0)),
            ("PatchSize", json!(65)),
            ("Merge", json!(9)),
            ("MinImageTokens", json!(0)),
            ("MinImageTokens", json!(257)),
            ("MaxImageTokens", json!(4097)),
            ("Width", json!(0)),
            ("Height", json!(4097)),
            ("Width", json!(-1)),
            ("Height", json!(1.0)),
            ("PatchSize", json!(true)),
            ("Unknown", json!(0)),
        ] {
            let mut operation = tile();
            operation[field] = value;
            assert!(execute(&request(operation)).is_err(), "{field}");
        }
        let image = json!({"Kind":"bitmap", "Width":1, "Height":1, "Mergeable":true});
        for parts in [
            json!(vec![image; 65]),
            json!([{"Kind":"bitmap", "Width":1, "Height":0, "Mergeable":true}]),
            json!([{"Kind":"audio", "Samples":1_048_577, "Mergeable":true}]),
            json!([{"Kind":"audio", "Samples":1, "Mergeable":0}]),
            json!([{"Kind":"text", "Text":"x".repeat(65_537)}]),
            json!([{"Kind":"text", "Text":"valid", "Extra":0}]),
            json!([{"Kind":"unknown"}]),
        ] {
            assert!(
                execute(&request(
                    json!({"Kind":"mtmd-group-bitmaps", "Merge":2, "Parts":parts})
                ))
                .is_err()
            );
        }
        for operations in [json!([]), json!(vec![tile(); 129])] {
            assert!(execute(&json!({"SchemaVersion":1,"Operations":operations})).is_err());
        }
        let mut invalid_version = request(tile());
        invalid_version["SchemaVersion"] = json!(2);
        assert!(execute(&invalid_version).is_err());
        assert!(execute(&request(tile())).is_ok());
    }
}
