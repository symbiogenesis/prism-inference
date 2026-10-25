//! Full safe Rust test-moe-weighted-reduce.cpp source on the CPU fallback.
//!
//! Rust owns all 48 geometries/layouts, original LCG inputs, manual overlapping
//! buffer placement and bit comparisons. Native code executes generic graphs.
//! Upstream MIT attribution: tools/prism/upstream/LICENSE.

use serde_json::{Value, json};

use super::engine::Engine;
use super::{Result, invalid, result};

const ID: &str = "tests/test-moe-weighted-reduce.cpp";

fn hex(values: &[f32]) -> String {
    const DIGITS: &[u8] = b"0123456789abcdef";
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .flat_map(|byte| {
            [
                char::from(DIGITS[usize::from(byte >> 4)]),
                char::from(DIGITS[usize::from(byte & 15)]),
            ]
        })
        .collect()
}

fn random_values(seed: &mut u32, count: usize) -> Result<Vec<f32>> {
    (0..count)
        .map(|_| {
            *seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            Ok(f32::from(u16::try_from((*seed >> 8) & 0xffff)?) / 32768.0 - 1.0)
        })
        .collect()
}

fn exchange(engine: &Engine, name: &str, operations: &[Value]) -> Result<Option<Vec<Value>>> {
    let Some(response) =
        engine.execute_common(name, &json!({"SchemaVersion":1,"Operations":operations}))?
    else {
        return Ok(None);
    };
    let results = response["Results"]
        .as_array()
        .ok_or_else(|| invalid("arena Results must be an array"))?;
    if response["SchemaVersion"] != 1 || results.len() != operations.len() {
        return Err(invalid("arena response shape mismatch"));
    }
    Ok(Some(results.clone()))
}

fn graph(
    n_embd: usize,
    n_exp: usize,
    n_tok: usize,
    layout: usize,
) -> Result<(Value, Vec<usize>, Vec<usize>)> {
    let mut seed = u32::try_from(n_embd * 1_000_003 + n_exp * 10_007 + n_tok * 101 + layout)?;
    let x = random_values(&mut seed, n_embd * n_exp * n_tok)?;
    let w = random_values(&mut seed, n_exp * n_tok)?;
    let mut nodes = vec![
        json!({"Op":"input","Type":"F32","Dimensions":[n_embd,n_exp,n_tok],"DataHex":hex(&x)}),
        json!({"Op":"input","Type":"F32","Dimensions":[1,n_exp,n_tok],"DataHex":hex(&w)}),
        json!({"Op":"mul","Inputs":[0,1]}),
    ];
    let mut sizes = vec![x.len() * 4, w.len() * 4, x.len() * 4];
    let mut views = Vec::new();
    for expert in 0..n_exp {
        views.push(nodes.len());
        nodes.push(json!({"Op":"view_2d","Inputs":[2],"Dimensions":[n_embd,n_tok],"RowStrideBytes":n_embd*n_exp*4,"OffsetBytes":expert*n_embd*4}));
        // ggml_nbytes includes the gaps between a strided view's rows.
        sizes.push(((n_tok - 1) * n_embd * n_exp + n_embd) * 4);
    }
    let mut adds = Vec::new();
    let mut output = views[0];
    for &view in &views[1..] {
        let prior = output;
        output = nodes.len();
        nodes.push(json!({"Op":"add","Inputs":[prior,view]}));
        adds.push(output);
        sizes.push(n_embd * n_tok * 4);
    }
    let mut forward = views.clone();
    forward.push(output);
    // Both production CPU backends retain the pinned GGML_DEFAULT_N_THREADS=4.
    Ok((
        json!({"Kind":"cpu-arena-graph","Graph":{"Kind":"tensor-graph","Threads":4,"Nodes":nodes,"Outputs":[output]},"Forward":forward,"Allocation":Value::Null}),
        sizes,
        adds,
    ))
}

fn aligned(value: usize, alignment: usize) -> Result<usize> {
    value
        .checked_add(alignment - 1)
        .map(|sum| sum / alignment * alignment)
        .ok_or_else(|| invalid("arena byte offset overflow"))
}

fn placement(
    operation: &mut Value,
    sizes: &[usize],
    adds: &[usize],
    layout: usize,
    alignment: usize,
) -> Result<()> {
    let total = sizes.iter().try_fold(4096_usize, |total, size| {
        total
            .checked_add(size + 512)
            .ok_or_else(|| invalid("arena storage bound overflow"))
    })?;
    let mut offsets = vec![Value::Null; sizes.len()];
    let mut offset = 0;
    let mut next = |tensor: usize| -> Result<usize> {
        let at = offset;
        offset = aligned(offset + sizes[tensor], alignment)?;
        Ok(at)
    };
    let x = next(0)?;
    offsets[0] = json!(x);
    offsets[2] = json!(next(2)?);
    for &add in &adds[..adds.len() - 1] {
        offsets[add] = json!(next(add)?);
    }
    let dst = *adds
        .last()
        .ok_or_else(|| invalid("MoE source requires at least two experts"))?;
    match layout {
        0 => {
            offsets[dst] = json!(next(dst)?);
            offsets[1] = json!(next(1)?);
        }
        1 => {
            let at = next(dst)?;
            if alignment + sizes[1] > sizes[dst] {
                return Err(invalid(
                    "original weights-in-destination layout does not fit",
                ));
            }
            offsets[dst] = json!(at);
            offsets[1] = json!(at + alignment);
        }
        2 => {
            if alignment + sizes[dst] > sizes[0] {
                return Err(invalid("original destination-in-input layout does not fit"));
            }
            offsets[dst] = json!(x + alignment);
            offsets[1] = json!(next(1)?);
        }
        _ => return Err(invalid("unknown MoE source layout")),
    }
    for (tensor, at) in offsets.iter().enumerate() {
        if let Some(at) = at.as_u64()
            && usize::try_from(at)?
                .checked_add(sizes[tensor])
                .is_none_or(|end| end > total)
        {
            return Err(invalid("original arena total is insufficient"));
        }
    }
    operation["Allocation"] = json!({"BufferBytes":total,"Offsets":offsets});
    Ok(())
}

fn output(value: &Value, count: usize) -> Result<Option<&str>> {
    if value["Status"].as_i64().is_none() {
        return Err(invalid("arena graph Status must be an integer"));
    }
    if value["Status"] != 0 {
        return Ok(None);
    }
    if value["Outputs"].as_array().map(Vec::len) != Some(1) {
        return Err(invalid("arena output count mismatch"));
    }
    let data = value["Outputs"][0]["DataHex"]
        .as_str()
        .ok_or_else(|| invalid("arena output must contain byte hex"))?;
    if data.len() != count * 8
        || !data
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(invalid("arena output byte extent mismatch"));
    }
    Ok(Some(data))
}

pub(super) fn run(engine: &Engine) -> Result<Value> {
    let Some(info) = exchange(
        engine,
        "moe-buffer-info",
        &[json!({"Kind":"cpu-buffer-info"})],
    )?
    else {
        return Ok(result(ID,false,"Rust production manually aliased backend graph API is unavailable; all 48 source cases remain required.".to_owned()));
    };
    let alignment = usize::try_from(
        info[0]["Alignment"]
            .as_u64()
            .ok_or_else(|| invalid("CPU buffer Alignment must be unsigned"))?,
    )?;
    if alignment == 0 || alignment > 4096 {
        return Err(invalid("CPU buffer alignment outside bound"));
    }
    let mut count = 0;
    let mut failed = Vec::new();
    for layout in 0..3 {
        for n_exp in [2, 4, 8, 15] {
            for n_embd in [64, 66] {
                for n_tok in [1, 7] {
                    let (plain, sizes, adds) = graph(n_embd, n_exp, n_tok, layout)?;
                    let mut aliased = plain.clone();
                    placement(&mut aliased, &sizes, &adds, layout, alignment)?;
                    let values = exchange(engine, "moe-reduce", &[aliased, plain])?
                        .ok_or_else(|| invalid("CPU arena backend disappeared"))?;
                    let got = output(&values[0], n_embd * n_tok)?;
                    let expected = output(&values[1], n_embd * n_tok)?
                        .ok_or_else(|| invalid("source plain CPU reference graph failed"))?;
                    count += 1;
                    if got != Some(expected) {
                        failed.push(format!(
                            "layout={layout},n_exp={n_exp},n_embd={n_embd},n_tok={n_tok}"
                        ));
                    }
                }
            }
        }
    }
    if count != 48 {
        return Err(invalid("MoE source case count drifted"));
    }
    Ok(result(
        ID,
        failed.is_empty(),
        format!(
            "All 48 source geometries/layouts executed against a plain production CPU reference with bitwise comparisons; manual W_IN_DST and X_IN_DST overlaps retained. CPU fallback only; CUDA fusion unqualified. failed={}: {}",
            failed.len(),
            failed.join("; ")
        ),
    ))
}
