//! Full quantize-perf source workload. Every execution is restricted to CI.
use super::engine::Engine;
use super::model_session::ModelSession;
use super::quant_buffers::{Driver, Type, call, integer, pointer, row_size, verify};
use super::{Result, invalid, result};
use serde_json::{Value, json};
use std::env;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
const ID: &str = "tests/test-quantize-perf.cpp";
const OPS: [&str; 5] = [
    "quantize_row_q_reference",
    "quantize_row_q",
    "dequantize_row_q",
    "quantize_row_q_dot",
    "vec_dot_q",
];
const SIZES: [usize; 4] = [4096, 65536, 655_360, 65_536_000];
#[derive(Default)]
pub(super) struct Options {
    pub(super) types: Vec<String>,
    pub(super) sizes: Vec<usize>,
    pub(super) operations: Vec<String>,
    pub(super) alignment_offset: usize,
    pub(super) iterations: Option<u32>,
}
/// C++ stoi's whitespace/sign/numeric-prefix rules, including its int32 bound.
fn stoi(text: &str) -> Result<i32> {
    let text = text.trim_start_matches(|c: char| c.is_ascii_whitespace());
    let (negative, digits) = text.strip_prefix('-').map_or_else(
        || (false, text.strip_prefix('+').unwrap_or(text)),
        |s| (true, s),
    );
    let prefix = digits.bytes().take_while(u8::is_ascii_digit).count();
    verify(prefix > 0, "quantize-perf expects stoi integer")?;
    let magnitude = digits[..prefix].parse::<i64>()?;
    i32::try_from(if negative { -magnitude } else { magnitude }).map_err(Into::into)
}
/// None denotes original --help, whose source exit status is1.
pub(super) fn parse_arguments(arguments: &[String]) -> Result<Option<Options>> {
    let mut out = Options::default();
    let mut index = 0;
    while index < arguments.len() {
        let argument = &arguments[index];
        index += 1;
        match argument.as_str() {
            "-h" | "--help" => return Ok(None),
            "-3" => out.sizes.extend_from_slice(&SIZES[..3]),
            "-4" => out.sizes.extend_from_slice(&SIZES),
            "--size" | "--op" | "--type" | "--alignment-offset" | "-i" | "--iterations" => {
                let value = arguments
                    .get(index)
                    .ok_or_else(|| invalid("missing quantize-perf option argument"))?;
                index += 1;
                match argument.as_str() {
                    "--size" => {
                        let parsed = stoi(value)?;
                        verify(parsed % 32 == 0, "quantize-perf size must be divisible32")?;
                        out.sizes
                            .push(usize::try_from(i64::from(parsed).cast_unsigned())?);
                    }
                    "--op" => {
                        verify(
                            OPS.contains(&value.as_str()),
                            "unknown quantize-perf operation",
                        )?;
                        out.operations.push(value.clone());
                    }
                    "--type" => out.types.push(value.clone()),
                    "--alignment-offset" => {
                        let offset = stoi(value)?;
                        verify(
                            (0..=64).contains(&offset),
                            "quantize-perf alignment offset outside0..64",
                        )?;
                        out.alignment_offset = usize::try_from(offset)?;
                    }
                    _ => {
                        let iterations = stoi(value)?;
                        verify(
                            (0..=100_000_000).contains(&iterations),
                            "quantize-perf iterations outside0..100000000",
                        )?;
                        out.iterations = Some(u32::try_from(iterations)?);
                    }
                }
            }
            _ => return Err(invalid(format!("unknown quantize-perf option {argument}"))),
        }
    }
    Ok(Some(out))
}
#[allow(
    clippy::as_conversions,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::suboptimal_flops,
    reason = "Source converts size_t to binary32 before cosf, multiplies in binary32, then adds double0.1 before narrowing."
)]
fn generate(path: &Path, count: usize, offset: f32) -> Result<()> {
    let mut output = BufWriter::new(File::create(path)?);
    for index in 0..count {
        let value = (f64::from(2.0_f32 * ((index as f32) + offset).cos()) + 0.1_f64) as f32;
        output.write_all(&value.to_le_bytes())?;
    }
    output.flush()?;
    Ok(())
}
fn clock(counter: &str) -> Value {
    json!({"Op":"qb-clock","Counter":counter})
}
#[derive(Default)]
struct Timing {
    minimum_us: Option<i64>,
    total_us: i64,
    minimum_cycles: Option<i64>,
    total_cycles: i64,
    iterations: u32,
}
fn timed(driver: &Driver, operation: &Value, iterations: u32) -> Result<Timing> {
    let warmups = vec![operation.clone(); 5];
    let trace = driver.trace(&warmups)?;
    verify(
        trace.iter().all(Value::is_null),
        "quantization warmup acknowledgement differs",
    )?;
    let mut timing = Timing::default();
    let mut remaining = iterations;
    while remaining > 0 {
        let amount = remaining.min(32);
        let mut actions = Vec::new();
        for _ in 0..amount {
            actions.extend([
                clock("microseconds"),
                clock("cycles"),
                operation.clone(),
                clock("cycles"),
                clock("microseconds"),
            ]);
        }
        let trace = driver.trace(&actions)?;
        for entries in trace.as_chunks::<5>().0 {
            let start_us = integer(&entries[0], "Value")?;
            let start_cycles = integer(&entries[1], "Value")?;
            verify(
                entries[2].is_null(),
                "quantization timed acknowledgement differs",
            )?;
            let end_cycles = integer(&entries[3], "Value")?;
            let end_us = integer(&entries[4], "Value")?;
            let us = end_us.wrapping_sub(start_us);
            let cycles = end_cycles.wrapping_sub(start_cycles);
            timing.minimum_us = Some(timing.minimum_us.map_or(us, |old| old.min(us)));
            timing.minimum_cycles =
                Some(timing.minimum_cycles.map_or(cycles, |old| old.min(cycles)));
            timing.total_us = timing.total_us.wrapping_add(us);
            timing.total_cycles = timing.total_cycles.wrapping_add(cycles);
            timing.iterations += 1;
        }
        remaining -= amount;
    }
    verify(
        timing.iterations == iterations,
        "quantization measured iteration extent differs",
    )?;
    Ok(timing)
}
fn number(value: f32) -> Value {
    json!({"Value":value.is_finite().then_some(value),"Bits":value.to_bits()})
}
#[allow(
    clippy::as_conversions,
    clippy::cast_precision_loss,
    clippy::suboptimal_flops,
    reason = "Reported source metrics intentionally use binary32 integer conversions and separate operations."
)]
fn metrics(size: usize, quantized: usize, timing: &Timing) -> Value {
    let iterations = timing.iterations as usize;
    let min_cycles = timing.minimum_cycles.unwrap_or(i64::MAX);
    let cycles_min = (32_i64.wrapping_mul(min_cycles) as f32) / (size as f32);
    let cycles_average =
        (32_i64.wrapping_mul(timing.total_cycles) as f32) / (size.wrapping_mul(iterations) as f32);
    let throughput = |bytes: usize| {
        ((bytes as f32) / (timing.total_us as f32)) * 1_000_000.0_f32 / (1024 * 1024 * 1024) as f32
    };
    json!({"Iterations":timing.iterations,"MinMicroseconds":timing.minimum_us.unwrap_or(i64::MAX),"TotalMicroseconds":timing.total_us,"MinCycles":min_cycles,"TotalCycles":timing.total_cycles,"MinimumCyclesPer32Values":number(cycles_min),"AverageCyclesPer32Values":number(cycles_average),"Float32GiBPerSecond":number(throughput(4_usize.wrapping_mul(size).wrapping_mul(iterations))),"QuantizedGiBPerSecond":number(throughput(quantized.wrapping_mul(iterations))),"OriginalZeroIterationSignedOverflow":timing.iterations==0})
}
fn benchmark_type(
    driver: &Driver,
    t: &Type,
    types: &[Type],
    sizes: &[usize],
    operations: &[&str],
    buffers: &[u64; 5],
    iterations: u32,
) -> Result<Vec<Value>> {
    let largest = *sizes
        .last()
        .ok_or_else(|| invalid("perf source sizes empty"))?;
    let [data1, data2, q1, q2, output] = *buffers;
    driver.null(0, json!({"Op":"qb-init-type","Type":t.id}))?;
    let mut reports = Vec::new();
    for &op in operations {
        if op == "dequantize_row_q" || op == "vec_dot_q" {
            driver.null(
                0,
                call(
                    "from-float-cpu",
                    t.id,
                    largest,
                    pointer(data1, 0),
                    pointer(q1, 0),
                ),
            )?;
            if op == "vec_dot_q" {
                driver.null(
                    0,
                    call(
                        "from-float-cpu",
                        t.id,
                        largest,
                        pointer(data2, 0),
                        pointer(q2, 0),
                    ),
                )?;
            }
        }
        for &size in sizes {
            let bytes = row_size(t, size)?;
            let operation = match op {
                "quantize_row_q_reference" => call(
                    "from-float-reference",
                    t.id,
                    size,
                    pointer(data1, 0),
                    pointer(q1, 0),
                ),
                "quantize_row_q" => call(
                    "from-float-cpu",
                    t.id,
                    size,
                    pointer(data1, 0),
                    pointer(q1, 0),
                ),
                "dequantize_row_q" => {
                    call("to-float", t.id, size, pointer(q1, 0), pointer(output, 0))
                }
                "quantize_row_q_dot" => {
                    verify(
                        types.get(usize::try_from(t.dot_operand)?).is_some(),
                        "vec-dot type missing from inventory",
                    )?;
                    call(
                        "from-float-cpu",
                        t.dot_operand,
                        size,
                        pointer(data1, 0),
                        pointer(q1, 0),
                    )
                }
                _ => {
                    let mut operation =
                        call("vec-dot", t.id, size, pointer(q1, 0), pointer(output, 0));
                    operation["Rhs"] = pointer(q2, 0);
                    operation
                }
            };
            let timing = timed(driver, &operation, iterations)?;
            reports.push(json!({"Type":t.id,"Name":t.name,"Operation":op,"Size":size,"QuantizedBytes":bytes,"Timing":metrics(size,bytes,&timing),"RhsPreparedType":(op=="vec_dot_q").then_some(t.id),"RhsProductionDotType":(op=="vec_dot_q").then_some(t.dot_operand),"SourceDotBufferTypeDiffers":op=="vec_dot_q"&&t.id!=t.dot_operand}));
        }
    }
    Ok(reports)
}
pub(super) fn run(engine: &Engine) -> Result<Value> {
    run_with_options(engine, &Options::default())
}
pub(super) fn run_with_options(engine: &Engine, options: &Options) -> Result<Value> {
    if env::var("GITHUB_ACTIONS").as_deref() != Ok("true")
        || env::var("GITHUB_REF").as_deref() != Ok("refs/heads/main")
    {
        return Err(invalid(
            "Quantization performance source runs only in GitHub Actions on main",
        ));
    }
    verify(
        options.alignment_offset <= 64,
        "perf alignment offset exceeds source limit",
    )?;
    let iterations = options.iterations.unwrap_or(10);
    verify(
        iterations <= 100_000_000,
        "perf iterations exceed source limit",
    )?;
    let mut sizes = options.sizes.clone();
    if sizes.is_empty() {
        sizes.push(SIZES[0]);
    }
    verify(
        sizes.iter().all(|size| size % 32 == 0),
        "perf sizes must be divisible32",
    )?;
    sizes.sort_unstable();
    let largest = *sizes.last().ok_or_else(|| invalid("empty perf sizes"))?;
    let capacity = largest
        .checked_mul(4)
        .and_then(|n| n.checked_add(128))
        .ok_or_else(|| invalid("perf source buffer extent overflow"))?;
    let operations = OPS
        .iter()
        .copied()
        .filter(|name| {
            options.operations.is_empty()
                || options.operations.iter().any(|selected| selected == name)
        })
        .collect::<Vec<_>>();
    verify(!operations.is_empty(), "perf operation selection empty")?;
    let Some(session) = ModelSession::open(engine, "quantize-perf")? else {
        return Ok(result(ID,false,"Rust runtime lacks the production trait/buffer/timing APIs for the full performance source".into()));
    };
    let driver = Driver::new(session.client(), session.evidence_directory());
    let mut buffers = [0_u64; 5];
    for value in &mut buffers {
        *value = driver.buffer(capacity, 64, options.alignment_offset)?;
    }
    let input1 = driver.path("synthetic-offset0");
    let input2 = driver.path("synthetic-offset1");
    generate(&input1, largest, 0.0)?;
    generate(&input2, largest, 1.0)?;
    driver.load(buffers[0], &input1, largest * 4)?;
    driver.load(buffers[1], &input2, largest * 4)?;
    let types = driver.inventory()?;
    let mut observations = Vec::new();
    let mut completed = 0;
    for t in &types {
        if !options.types.is_empty()
            && t.name
                .as_ref()
                .is_some_and(|name| !options.types.contains(name))
        {
            continue;
        }
        if t.cpu && t.decoder {
            observations.extend(benchmark_type(
                &driver,
                t,
                &types,
                &sizes,
                &operations,
                &buffers,
                iterations,
            )?);
            completed += 1;
        }
    }
    for buffer in buffers {
        driver.free(buffer)?;
    }
    session.finish()?;
    let mut output = result(
        ID,
        true,
        format!(
            "{completed} source-eligible types, {} size/operation workloads;5warmups and{iterations} measured calls each;0 original numeric assertions",
            observations.len()
        ),
    );
    output["SourceSha256"] =
        json!("09de1812b11836bd899a63dfa6b06927a2d0a012edc1e702a33fb51c95d63563");
    output["Observations"] = json!(observations);
    output["SourceAssertions"] = json!(0);
    output["AlignmentOffset"] = json!(options.alignment_offset);
    output["TimingScope"] = json!(
        "Production calls plus generic action dispatch, excluding input preparation, warmups and request parsing"
    );
    Ok(output)
}
