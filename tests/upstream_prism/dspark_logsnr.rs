//! Complete eight-variant `DSpark` log-SNR production-loader source contract.
use super::dspark_inputs::Corpus;
use super::engine::Engine;
use super::{Result, invalid, result};
use serde_json::{Value, json};
const ID: &str = "tests/test-dspark-logsnr-meta.cpp";
const SOURCE_SHA: &str = "2fef2521dbc0985a3aed6bbe50de5a54b7a75dc6c3c5933e04fc8ceb4c7e3112";
#[derive(Clone, Copy)]
struct Variant {
    name: &'static str,
    conditioning: bool,
    minimum: Option<u32>,
    maximum: Option<u32>,
    load: bool,
}
const VARIANTS: [Variant; 8] = [
    Variant {
        name: "logsnr-valid.gguf",
        conditioning: true,
        minimum: Some((-9_f32).to_bits()),
        maximum: Some(6_f32.to_bits()),
        load: true,
    },
    Variant {
        name: "logsnr-omit-min.gguf",
        conditioning: true,
        minimum: None,
        maximum: Some(6_f32.to_bits()),
        load: false,
    },
    Variant {
        name: "logsnr-omit-max.gguf",
        conditioning: true,
        minimum: Some((-9_f32).to_bits()),
        maximum: None,
        load: false,
    },
    Variant {
        name: "logsnr-min-nan.gguf",
        conditioning: true,
        minimum: Some(0x7fc0_0000),
        maximum: Some(6_f32.to_bits()),
        load: false,
    },
    Variant {
        name: "logsnr-max-inf.gguf",
        conditioning: true,
        minimum: Some((-9_f32).to_bits()),
        maximum: Some(0x7f80_0000),
        load: false,
    },
    Variant {
        name: "logsnr-equal.gguf",
        conditioning: true,
        minimum: Some(6_f32.to_bits()),
        maximum: Some(6_f32.to_bits()),
        load: false,
    },
    Variant {
        name: "logsnr-reversed.gguf",
        conditioning: true,
        minimum: Some(6_f32.to_bits()),
        maximum: Some((-9_f32).to_bits()),
        load: false,
    },
    Variant {
        name: "logsnr-cond-off.gguf",
        conditioning: false,
        minimum: None,
        maximum: None,
        load: true,
    },
];
fn unhex(text: &str) -> Result<Vec<u8>> {
    if text.len() % 2 != 0 || text.len() > 256 * 1024 {
        return Err(invalid("DSpark log extent outside bound"));
    }
    text.as_bytes()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| {
            let digit = |b| match b {
                b'0'..=b'9' => Ok(b - b'0'),
                b'a'..=b'f' => Ok(b - b'a' + 10),
                _ => Err(invalid("DSpark log hex digit invalid")),
            };
            Ok(digit(pair[0])? * 16 + digit(pair[1])?)
        })
        .collect()
}
fn verdict(variant: Variant, observed: &Value) -> Result<bool> {
    let loaded = observed["Loaded"]
        .as_bool()
        .ok_or_else(|| invalid("DSpark production load result absent"))?;
    let log = unhex(
        observed["LogHex"]
            .as_str()
            .ok_or_else(|| invalid("DSpark production load log absent"))?,
    )?;
    let guard = log.windows(7).any(|s| s == b"log_snr" || s == b"log-SNR");
    Ok(if variant.load {
        loaded
    } else {
        !loaded && guard
    })
}
#[allow(
    clippy::unnecessary_wraps,
    reason = "source adapter returns an explicit measured failure result"
)]
pub(super) fn run(engine: &Engine) -> Result<Value> {
    let execute: Result<Value> = (|| {
        let directory = engine.evidence_path("dspark-logsnr-variants")?;
        let corpus = Corpus::load()?;
        let mut paths = Vec::new();
        for variant in VARIANTS {
            let path = directory.join(variant.name);
            corpus.write(
                &path,
                variant.conditioning,
                variant.minimum,
                variant.maximum,
                false,
            )?;
            paths.push(path);
        }
        let present = paths.iter().filter(|p| p.is_file()).count();
        if present != VARIANTS.len() {
            return Err(invalid("source partial DSpark variant suite"));
        }
        let request = json!({"SchemaVersion":1,"Operations":[{"Kind":"model-load-observe","Paths":paths,"GpuLayers":0}]});
        let response = engine
            .execute_common("dspark-logsnr-full", &request)?
            .ok_or_else(|| invalid("missing production DSpark model-loader API"))?;
        let observations = response["Results"][0]["Models"]
            .as_array()
            .filter(|v| v.len() == VARIANTS.len())
            .ok_or_else(|| invalid("DSpark production model observation count differs"))?;
        let mut results = Vec::new();
        let mut passed = true;
        for (variant, observed) in VARIANTS.into_iter().zip(observations) {
            let ok = verdict(variant, observed)?;
            passed &= ok;
            results.push(json!({"File":variant.name,"ExpectedLoad":variant.load,"Passed":ok,"Observation":observed}));
        }
        if !passed {
            return Err(invalid(format!(
                "DSpark log-SNR loader predicate failed: {}",
                json!(results)
            )));
        }
        Ok(
            json!({"SourceSHA256":SOURCE_SHA,"Variants":8,"ExpectedLoaded":2,"ExpectedGuardRejected":6,"SourcePredicates":14,"GeneratedDirectory":directory,"MissingSuiteSkip":"source retained as prerequisite rule; this port generates complete inputs","Results":results}),
        )
    })();
    match execute {
        Ok(v) => Ok(result(ID, true, v.to_string())),
        Err(e) => Ok(result(
            ID,
            false,
            format!("complete DSpark log-SNR source contract: {e}"),
        )),
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn focused_dslug_guard_rejects_unrelated_load_failure() -> Result<()> {
        if verdict(
            VARIANTS[1],
            &json!({"Loaded":false,"LogHex":"756e72656c61746564"}),
        )? {
            return Err(invalid("unrelated model failure accepted"));
        }
        if !verdict(
            VARIANTS[1],
            &json!({"Loaded":false,"LogHex":"6c6f675f736e72"}),
        )? {
            return Err(invalid("expected loader guard not recognized"));
        }
        let c = Corpus::load()?;
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "dspark-input-framing-{}-{stamp}.gguf",
            std::process::id()
        ));
        c.write(&path, true, VARIANTS[0].minimum, VARIANTS[0].maximum, false)?;
        let bytes = std::fs::read(&path)?;
        std::fs::remove_file(path)?;
        if !bytes.starts_with(b"GGUF\x03\0\0\0")
            || u64::from_le_bytes(bytes[8..16].try_into()?) != 30
        {
            return Err(invalid("DSpark independent GGUF input framing differs"));
        }
        Ok(())
    }
}
