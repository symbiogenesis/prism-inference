//! Entire quantize-stats controller, including source scheduling and statistics.
//! The native side supplies model/codec calls and bytes, never error calculations.
use super::engine::Engine;
use super::llama_grammar::{array, exact_fields, field, uint};
use super::model_session::ModelSession;
use super::quant_buffers::{Driver, Type, call, integer, pointer, row_size, verify};
use super::{Result, invalid, result};
use serde_json::{Value, json};
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Condvar, Mutex, mpsc};
use std::thread;
use std::time::{Duration, Instant};
const ID: &str = "tests/test-quantize-stats.cpp";
const CHUNK: usize = 32 * 512;
// Coalesce transport, retaining every original CHUNK-sized production call.
// The source mutex permits a worker to claim these consecutive chunks while
// the other workers are paused. Reserve that valid schedule up front, then
// accumulate its elements in their original order before reserving more work.
const TRANSPORT_CHUNKS: usize = 16;
const BUCKETS: usize = 150;
const RANGE: f64 = 0.03;
#[derive(Default)]
#[allow(
    clippy::struct_excessive_bools,
    reason = "Original independent CLI flags are retained individually."
)]
pub(super) struct Options {
    pub(super) model: Option<PathBuf>,
    pub(super) reference: bool,
    pub(super) verbose: bool,
    pub(super) per_layer_stats: bool,
    pub(super) histogram: bool,
    pub(super) include_layers: Vec<String>,
    pub(super) exclude_layers: Vec<String>,
    pub(super) types: Vec<String>,
    pub(super) threads: i32,
}
fn atoi(text: &str) -> Result<i32> {
    let text = text.trim_start_matches(|c: char| c.is_ascii_whitespace());
    let (negative, digits) = text.strip_prefix('-').map_or_else(
        || (false, text.strip_prefix('+').unwrap_or(text)),
        |digits| (true, digits),
    );
    let n = digits.bytes().take_while(u8::is_ascii_digit).count();
    if n == 0 {
        return Ok(0);
    }
    let magnitude = digits[..n].parse::<i64>()?;
    // C atoi has undefined behavior outside int range; no invented wrapped value.
    i32::try_from(if negative { -magnitude } else { magnitude }).map_err(Into::into)
}
/// None is the original help exit (status0). Type names are validated before load.
pub(super) fn parse_arguments(arguments: &[String]) -> Result<Option<Options>> {
    let mut out = Options::default();
    let mut index = 0;
    while index < arguments.len() {
        let arg = &arguments[index];
        index += 1;
        match arg.as_str() {
            "-h" | "--help" => return Ok(None),
            "-r" | "--reference" => out.reference = true,
            "-v" => out.verbose = true,
            "-p" | "--per-layer-stats" => out.per_layer_stats = true,
            "--histogram" => out.histogram = true,
            "-m" | "--model" | "-l" | "--include-layer" | "-L" | "--exclude-layer" | "-t"
            | "--type" | "-n" | "--num-threads" => {
                let value = arguments
                    .get(index)
                    .ok_or_else(|| invalid("missing quantize-stats option argument"))?;
                index += 1;
                match arg.as_str() {
                    "-m" | "--model" => out.model = Some(value.into()),
                    "-l" | "--include-layer" => out.include_layers.push(value.clone()),
                    "-L" | "--exclude-layer" => out.exclude_layers.push(value.clone()),
                    "-t" | "--type" => out.types.push(value.clone()),
                    _ => out.threads = atoi(value)?,
                }
            }
            _ => return Err(invalid(format!("unknown quantize-stats option {arg}"))),
        }
    }
    Ok(Some(out))
}
#[derive(Clone)]
struct Layer {
    name: String,
    type_id: u32,
    ne: [usize; 4],
    nb: [usize; 4],
    elements: usize,
    type_size: usize,
    block: usize,
}
fn dimensions(value: &Value, name: &str) -> Result<[usize; 4]> {
    let values = array(value, name)?;
    verify(values.len() == 4, "source requires GGML_MAX_DIMS4")?;
    let mut out = [0; 4];
    for (destination, source) in out.iter_mut().zip(values) {
        *destination = usize::try_from(
            source
                .as_u64()
                .ok_or_else(|| invalid("invalid tensor extent"))?,
        )?;
    }
    Ok(out)
}
fn layers(driver: &Driver, model: u64) -> Result<Vec<Layer>> {
    let value = driver.call(0, json!({"Op":"qb-model-tensors","Model":model}))?;
    exact_fields(&value, &["Tensors"])?;
    array(&value, "Tensors")?
        .iter()
        .map(|t| {
            exact_fields(
                t,
                &[
                    "Name",
                    "Type",
                    "NE",
                    "NB",
                    "Elements",
                    "Bytes",
                    "TypeSize",
                    "BlockSize",
                ],
            )?;
            let layer = Layer {
                name: field(t, "Name")?
                    .as_str()
                    .ok_or_else(|| invalid("tensor name missing"))?
                    .into(),
                type_id: u32::try_from(uint(t, "Type")?)?,
                ne: dimensions(t, "NE")?,
                nb: dimensions(t, "NB")?,
                elements: usize::try_from(uint(t, "Elements")?)?,
                type_size: usize::try_from(uint(t, "TypeSize")?)?,
                block: usize::try_from(uint(t, "BlockSize")?)?,
            };
            verify(
                layer.ne.iter().all(|n| *n > 0)
                    && layer
                        .ne
                        .iter()
                        .try_fold(1_usize, |product, n| product.checked_mul(*n))
                        == Some(layer.elements),
                "model tensor element metadata differs from dimensions",
            )?;
            Ok(layer)
        })
        .collect()
}
fn contiguous(t: &Layer) -> bool {
    t.block != 0
        && t.nb[0] == t.type_size
        && t.nb[0].checked_mul(t.ne[0]).map(|n| n / t.block) == Some(t.nb[1])
        && t.nb[1].checked_mul(t.ne[1]) == Some(t.nb[2])
        && t.nb[2].checked_mul(t.ne[2]) == Some(t.nb[3])
}
/// `std::regex`'s ECMAScript matches are raw observations; Rust owns precedence
/// (exclude first, then include, then empty-include default), as in the source.
fn layer_included(options: &Options, mut search: impl FnMut(&str) -> Result<bool>) -> Result<bool> {
    for pattern in &options.exclude_layers {
        if search(pattern)? {
            return Ok(false);
        }
    }
    for pattern in &options.include_layers {
        if search(pattern)? {
            return Ok(true);
        }
    }
    Ok(options.include_layers.is_empty())
}
fn regex_search(engine: &Engine, pattern: &str, layer: &str) -> Result<bool> {
    // Compile only the next pattern reached by the source. A later invalid
    // expression must remain unobserved after an earlier match short-circuits.
    let request = json!({"SchemaVersion":1,"Operations":[{"Kind":"chat-regex","Patterns":[pattern],"Actions":[{"Pattern":0,"Function":"search","Bytes":layer.as_bytes()}]}]});
    let response = engine
        .execute_common("quantize-stats-layer-filter", &request)?
        .ok_or_else(|| invalid("ECMAScript regex API unavailable"))?;
    exact_fields(&response, &["SchemaVersion", "Results"])?;
    verify(
        uint(&response, "SchemaVersion")? == 1,
        "regex response schema differs",
    )?;
    let results = array(&response, "Results")?;
    verify(results.len() == 1, "regex result count differs")?;
    exact_fields(&results[0], &["Trace"])?;
    let trace = array(&results[0], "Trace")?;
    verify(trace.len() == 1, "regex trace extent differs")?;
    exact_fields(&trace[0], &["Matched", "MarkCount", "Captures"])?;
    super::quant_buffers::flag(&trace[0], "Matched")
}
fn selected_layers(engine: &Engine, options: &Options, layers: Vec<Layer>) -> Result<Vec<Layer>> {
    if options.include_layers.is_empty() && options.exclude_layers.is_empty() {
        return Ok(layers);
    }
    let mut included = Vec::new();
    for layer in layers {
        if layer_included(options, |pattern| {
            regex_search(engine, pattern, &layer.name)
        })? {
            included.push(layer);
        }
    }
    Ok(included)
}
#[derive(Clone)]
struct Stats {
    samples: u64,
    squared: f64,
    maximum: f64,
    histogram: [u64; BUCKETS],
}
impl Default for Stats {
    fn default() -> Self {
        Self {
            samples: 0,
            squared: 0.0,
            maximum: 0.0,
            histogram: [0; BUCKETS],
        }
    }
}
impl Stats {
    #[allow(
        clippy::as_conversions,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss,
        clippy::suboptimal_flops,
        reason = "Original histogram floors a nonnegative finite double before conversion and accumulates separate double operations."
    )]
    fn update(&mut self, input: &[u8], output: &[u8]) -> Result<()> {
        verify(
            input.len() == output.len() && input.len() % 4 == 0,
            "quantization input/output extent differs",
        )?;
        for (a, b) in input
            .as_chunks::<4>()
            .0
            .iter()
            .zip(output.as_chunks::<4>().0)
        {
            let diff = f64::from(f32::from_le_bytes(*a) - f32::from_le_bytes(*b));
            verify(
                diff.is_finite(),
                "source histogram float-to-size_t conversion is undefined for nonfinite error",
            )?;
            self.squared += diff * diff;
            self.maximum = self.maximum.max(diff.abs());
            let bucket = (diff.abs() / RANGE * (BUCKETS as f64)).floor();
            verify(
                bucket < 18_446_744_073_709_551_616.0,
                "source histogram float-to-size_t conversion exceeds64-bit range",
            )?;
            let bucket = bucket.min((BUCKETS - 1) as f64) as usize;
            self.histogram[bucket] = self.histogram[bucket].wrapping_add(1);
        }
        self.samples = self.samples.wrapping_add(u64::try_from(input.len() / 4)?);
        Ok(())
    }
    fn combine(&mut self, from: &Self) {
        self.samples = self.samples.wrapping_add(from.samples);
        self.squared += from.squared;
        if from.maximum > self.maximum {
            self.maximum = from.maximum;
        }
        for (a, b) in self.histogram.iter_mut().zip(from.histogram) {
            *a = a.wrapping_add(b);
        }
    }
    #[allow(
        clippy::as_conversions,
        clippy::cast_precision_loss,
        clippy::suboptimal_flops,
        reason = "Source accumulate and reporting explicitly convert counters to double, preserving histogram order."
    )]
    fn quantile(&self, q: f64) -> f64 {
        let sum = self
            .histogram
            .iter()
            .fold(0.0, |total, n| total + (*n as f64));
        let mut accum = 0.0;
        for (i, n) in self.histogram.iter().enumerate() {
            accum += *n as f64;
            if accum >= sum * q {
                return ((i + 1) as f64) * RANGE / (BUCKETS as f64);
            }
        }
        f64::INFINITY
    }
    #[allow(
        clippy::as_conversions,
        clippy::cast_precision_loss,
        reason = "Source RMSE and histogram interval display convert integer counts to double."
    )]
    fn report(&self, print_histogram: bool) -> Value {
        let intervals=(0..BUCKETS).map(|i|json!({"Lower":number((i as f64)*RANGE/(BUCKETS as f64)),"Upper":number(if i==BUCKETS-1{f64::INFINITY}else{((i+1)as f64)*RANGE/(BUCKETS as f64)}),"Count":self.histogram[i]})).collect::<Vec<_>>();
        json!({"Samples":self.samples,"SquaredError":number(self.squared),"RMSE":number((self.squared/(self.samples as f64)).sqrt()),"Maximum":number(self.maximum),"MedianUpperBound":number(self.quantile(0.5)),"Percentile95UpperBound":number(self.quantile(0.95)),"Histogram":self.histogram.to_vec(),"PrintedHistogram":print_histogram.then_some(intervals)})
    }
}
fn number(v: f64) -> Value {
    json!({"Value":v.is_finite().then_some(v),"Bits":v.to_bits()})
}
struct Queue {
    counter: usize,
    target: Stats,
}
impl Queue {
    fn claim(&mut self, elements: usize, local: &Stats) -> Result<Option<(usize, usize)>> {
        let offset = self.counter;
        self.counter = self
            .counter
            .checked_add(CHUNK * TRANSPORT_CHUNKS)
            .ok_or_else(|| invalid("statistics chunk counter overflow"))?;
        if offset >= elements {
            self.target.combine(local);
            Ok(None)
        } else {
            Ok(Some((
                offset,
                (elements - offset).min(CHUNK * TRANSPORT_CHUNKS),
            )))
        }
    }
}
struct Lanes {
    free: Mutex<Vec<u32>>,
    changed: Condvar,
}
struct Lane<'a> {
    number: u32,
    pool: &'a Lanes,
}
impl Drop for Lane<'_> {
    fn drop(&mut self) {
        if let Ok(mut free) = self.pool.free.lock() {
            free.push(self.number);
            self.pool.changed.notify_one();
        }
    }
}
impl Lanes {
    fn acquire(&self) -> Result<Lane<'_>> {
        let mut free = self
            .free
            .lock()
            .map_err(|_| invalid("quantization lane mutex poisoned"))?;
        loop {
            if let Some(number) = free.pop() {
                return Ok(Lane { number, pool: self });
            }
            free = self
                .changed
                .wait(free)
                .map_err(|_| invalid("quantization lane mutex poisoned"))?;
        }
    }
}
struct ChunkFiles {
    output: PathBuf,
    next: PathBuf,
    input: PathBuf,
    decoded: PathBuf,
}
fn codec_chunks(
    offset: usize,
    count: usize,
    grouped: bool,
) -> impl Iterator<Item = (usize, usize)> {
    // Source single-thread execution calls the codec once for the entire layer.
    let chunk = if grouped { CHUNK } else { count };
    (0..count)
        .step_by(chunk)
        .map(move |relative| (offset + relative, (count - relative).min(chunk)))
}
fn chunk_actions(
    work: &Work<'_>,
    offset: usize,
    count: usize,
    grouped: bool,
) -> (Vec<Value>, ChunkFiles) {
    let (driver, model, layer, t, reference, buffers) = (
        work.driver,
        work.model,
        work.layer,
        work.t,
        work.reference,
        work.buffers,
    );
    let [input, quantized, decoded] = buffers;
    let byte_offset = 4 * offset;
    let bytes = 4 * count;
    let files = ChunkFiles {
        output: driver.path("stats-checkpoint"),
        next: driver.path("stats-continuation"),
        input: driver.path("stats-input"),
        decoded: driver.path("stats-decoded"),
    };
    let timeout = if std::env::var("GITHUB_ACTIONS").as_deref() == Ok("true")
        && std::env::var("GITHUB_REF").as_deref() == Ok("refs/heads/main")
    {
        3_600_000
    } else {
        60_000
    };
    let mut actions = Vec::with_capacity(3 * count.div_ceil(CHUNK) + 3);
    for (chunk_offset, chunk_count) in codec_chunks(offset, count, grouped) {
        let chunk_bytes = 4 * chunk_offset;
        actions.push(json!({"Op":"qb-model-copy-f32","Model":model,"Name":layer.name,"ElementOffset":chunk_offset,"Count":chunk_count,"Buffer":input,"BufferOffset":chunk_bytes}));
        actions.push(call(
            if reference {
                "from-float-reference"
            } else {
                "from-float-cpu"
            },
            t.id,
            chunk_count,
            pointer(input, chunk_bytes),
            pointer(quantized, chunk_bytes),
        ));
        actions.push(call(
            "to-float",
            t.id,
            chunk_count,
            pointer(quantized, chunk_bytes),
            pointer(decoded, chunk_bytes),
        ));
    }
    actions.extend([
        json!({"Op":"qb-buffer-store","Buffer":input,"Offset":byte_offset,"File":files.input,"FileOffset":0,"Bytes":bytes,"Truncate":true}),
        json!({"Op":"qb-buffer-store","Buffer":decoded,"Offset":byte_offset,"File":files.decoded,"FileOffset":0,"Bytes":bytes,"Truncate":true}),
        json!({"Op":"qb-checkpoint","OutputFile":files.output,"NextFile":files.next,"TimeoutMilliseconds":timeout}),
    ]);
    (actions, files)
}
fn publish(path: &Path, actions: &[Value]) -> Result<()> {
    let temporary = path.with_extension("partial");
    let mut file = fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&temporary)?;
    let encoded = serde_json::to_vec(actions)?;
    verify(
        encoded.len() <= 128 * 1024 * 1024,
        "statistics action payload exceeds bound",
    )?;
    file.write_all(&encoded)?;
    drop(file);
    fs::rename(temporary, path)?;
    Ok(())
}
fn read_bounded(path: &Path, maximum: usize) -> Result<Vec<u8>> {
    verify(
        maximum <= 512 * 1024 * 1024,
        "statistics payload bound exceeds production buffer limit",
    )?;
    let mut bytes = Vec::new();
    fs::File::open(path)?
        .take(u64::try_from(maximum)? + 1)
        .read_to_end(&mut bytes)?;
    verify(
        bytes.len() <= maximum,
        "statistics native file exceeds its declared extent",
    )?;
    Ok(bytes)
}
fn consume(files: &ChunkFiles, expected: usize) -> Result<(Vec<u8>, Vec<u8>)> {
    let checkpoint: Value =
        serde_json::from_slice(&read_bounded(&files.output, 16 * 1024 * 1024)?)?;
    exact_fields(&checkpoint, &["Trace"])?;
    let trace = array(&checkpoint, "Trace")?;
    verify(
        trace.len() == 3 * expected.div_ceil(CHUNK) + 3 && trace.iter().all(Value::is_null),
        "statistics checkpoint acknowledgement differs",
    )?;
    let input = read_bounded(&files.input, 4 * expected)?;
    let decoded = read_bounded(&files.decoded, 4 * expected)?;
    verify(
        input.len() == 4 * expected && decoded.len() == 4 * expected,
        "statistics chunk payload extent differs",
    )?;
    fs::remove_file(&files.input)?;
    fs::remove_file(&files.decoded)?;
    fs::remove_file(&files.output)?;
    Ok((input, decoded))
}
fn wait(
    files: &ChunkFiles,
    receiver: &mpsc::Receiver<Result<Value>>,
    trace_ended: &mut bool,
) -> Result<()> {
    let started = Instant::now();
    while !files.output.is_file() {
        match receiver.try_recv() {
            Ok(reply) => {
                *trace_ended = true;
                let _ = reply?;
                return Err(invalid("codec trace ended before checkpoint"));
            }
            Err(mpsc::TryRecvError::Disconnected) => {
                *trace_ended = true;
                return Err(invalid("codec trace controller disconnected"));
            }
            Err(mpsc::TryRecvError::Empty) => {}
        }
        verify(
            started.elapsed() < Duration::from_secs(3600),
            "statistics checkpoint wait exceeded bound",
        )?;
        thread::sleep(Duration::from_millis(1));
    }
    Ok(())
}
fn finish_trace(
    receiver: &mpsc::Receiver<Result<Value>>,
    trace_ended: bool,
    controlled: Result<()>,
) -> Result<()> {
    if trace_ended {
        return controlled;
    }
    let reply = receiver
        .recv()
        .map_err(|_| invalid("statistics worker trace disconnected"))??;
    exact_fields(&reply, &["Trace"])?;
    verify(
        array(&reply, "Trace")?.is_empty(),
        "statistics completed trace not empty",
    )?;
    controlled
}
struct Work<'a> {
    driver: &'a Driver,
    model: u64,
    layer: &'a Layer,
    t: &'a Type,
    reference: bool,
    buffers: [u64; 3],
}
/// One source compute closure, on a retained native lane. Each bounded transport
/// group follows a valid source assignment of consecutive CHUNK-sized jobs.
fn worker(work: &Work<'_>, queue: &Mutex<Queue>, lanes: &Lanes) -> Result<()> {
    let lane = lanes.acquire()?;
    let mut local = Stats::default();
    let claim = |local: &Stats| -> Result<Option<(usize, usize)>> {
        let mut q = queue
            .lock()
            .map_err(|_| invalid("statistics source mutex poisoned"))?;
        q.claim(work.layer.elements, local)
    };
    let Some((offset, mut count)) = claim(&local)? else {
        return Ok(());
    };
    let (actions, mut files) = chunk_actions(work, offset, count, true);
    let initial = work.driver.path("stats-initial-actions");
    fs::write(&initial, serde_json::to_vec(&actions)?)?;
    let initial_for_cleanup = initial.clone();
    thread::scope(|scope| -> Result<()> {
        let (sender, receiver) = mpsc::channel();
        let lane_number = lane.number;
        scope.spawn(move || {
            let reply = work
                .driver
                .call(lane_number, json!({"Op":"qb-trace","ActionsFile":initial}));
            let _ = sender.send(reply);
        });
        let mut trace_ended = false;
        let controlled = (|| -> Result<()> {
            let mut previous_next = None;
            let mut first = true;
            loop {
                wait(&files, &receiver, &mut trace_ended)?;
                // A new checkpoint proves the preceding action file was fully
                // read. Retain failed payloads, but avoid collecting hundreds
                // of thousands of successful transport files in CI artifacts.
                if let Some(previous) = previous_next.take() {
                    fs::remove_file(previous)?;
                }
                if first {
                    fs::remove_file(&initial_for_cleanup)?;
                    first = false;
                }
                let (input, decoded) = consume(&files, count)?;
                local.update(&input, &decoded)?;
                let Some((offset, next_count)) = claim(&local)? else {
                    publish(&files.next, &[])?;
                    break;
                };
                let (actions, next_files) = chunk_actions(work, offset, next_count, true);
                publish(&files.next, &actions)?;
                previous_next = Some(files.next.clone());
                files = next_files;
                count = next_count;
            }
            Ok(())
        })();
        if controlled.is_err() && !trace_ended && !files.next.exists() {
            let _ = publish(&files.next, &[]);
        }
        let finished = finish_trace(&receiver, trace_ended, controlled);
        if finished.is_ok() {
            fs::remove_file(&files.next)?;
        }
        finished
    })
}
fn roundtrip_layer(work: &Work<'_>, target: Stats, max_thread: usize) -> Result<(Stats, usize)> {
    let chunks = work.layer.elements.div_ceil(CHUNK);
    if chunks < 2 || max_thread < 2 {
        let (mut actions, files) = chunk_actions(work, 0, work.layer.elements, false);
        actions.pop();
        let trace = work.driver.trace(&actions)?;
        verify(
            trace.iter().all(Value::is_null),
            "single-chunk acknowledgement differs",
        )?;
        let input = read_bounded(&files.input, 4 * work.layer.elements)?;
        let decoded = read_bounded(&files.decoded, 4 * work.layer.elements)?;
        verify(
            input.len() == 4 * work.layer.elements,
            "single-chunk input extent differs",
        )?;
        let mut stats = target;
        stats.update(&input, &decoded)?;
        fs::remove_file(files.input)?;
        fs::remove_file(files.decoded)?;
        return Ok((stats, 1));
    }
    let nthread = chunks.min(max_thread);
    verify(
        nthread <= 256,
        "statistics explicit thread count exceeds host resource bound256",
    )?;
    let queue = Mutex::new(Queue { counter: 0, target });
    let lanes = Lanes {
        free: Mutex::new((0..16).rev().collect()),
        changed: Condvar::new(),
    };
    thread::scope(|scope| -> Result<()> {
        let mut workers = Vec::new();
        for _ in 1..nthread {
            workers.push(scope.spawn(|| worker(work, &queue, &lanes)));
        }
        let main = worker(work, &queue, &lanes);
        let mut error = main.err();
        for worker in workers {
            match worker.join() {
                Ok(Ok(())) => {}
                Ok(Err(e)) => {
                    if error.is_none() {
                        error = Some(e);
                    }
                }
                Err(_) => {
                    if error.is_none() {
                        error = Some(invalid("statistics source worker panicked"));
                    }
                }
            }
        }
        error.map_or(Ok(()), Err)
    })?;
    Ok((
        queue
            .into_inner()
            .map_err(|_| invalid("statistics source mutex poisoned"))?
            .target,
        nthread,
    ))
}
pub(super) fn run(engine: &Engine) -> Result<Value> {
    run_with_options(engine, &Options::default())
}
#[allow(
    clippy::too_many_lines,
    reason = "Keeps source loading, eligibility, per-type loop and teardown in the original observable order."
)]
pub(super) fn run_with_options(engine: &Engine, options: &Options) -> Result<Value> {
    if std::env::var("GITHUB_ACTIONS").as_deref() != Ok("true")
        || std::env::var("GITHUB_REF").as_deref() != Ok("refs/heads/main")
    {
        return Err(invalid(
            "The complete quantization statistics source runs only in GitHub Actions on main",
        ));
    }
    let Some(session) = ModelSession::open_quantization(engine)? else {
        return Ok(result(ID,false,"Rust runtime lacks production model tensor-map and codec trait APIs for the complete statistics source".into()));
    };
    let driver = Driver::new(session.client(), session.evidence_directory());
    driver.null(0, json!({"Op":"qb-time-init"}))?;
    let types = driver.inventory()?;
    for requested in &options.types {
        verify(
            types.iter().any(|t| t.name.as_ref() == Some(requested)),
            format!("{requested} not in list of types"),
        )?;
    }
    let path = if let Some(path) = &options.model {
        fs::canonicalize(path)?
    } else {
        super::quantize_input::prepare(engine, session.evidence_directory())?
    };
    let start = integer(
        &driver.call(0, json!({"Op":"qb-clock","Counter":"microseconds"}))?,
        "Value",
    )?;
    let model=driver.handle(json!({"Op":"model-load","Path":path,"Base":"default","Params":null,"Overrides":{"LoadMode":"none"},"Progress":false}),"Model")?;
    let context = driver.handle(
        json!({"Op":"context","Model":model,"Params":null,"Overrides":{"Context":256}}),
        "Context",
    )?;
    let layers = selected_layers(engine, options, layers(&driver, model)?)?;
    let mut maximum = 0;
    let mut f16 = false;
    for layer in &layers {
        verify(
            layer.type_id <= 1,
            format!(
                "Quantization requires a float model; {} has type{}",
                layer.name, layer.type_id
            ),
        )?;
        f16 |= layer.type_id == 1;
        maximum = maximum.max(layer.elements);
    }
    let hardware = driver.call(0, json!({"Op":"hardware-concurrency"}))?;
    exact_fields(&hardware, &["Count"])?;
    let max_thread = if options.threads < 1 {
        usize::try_from(uint(&hardware, "Count")?)?
    } else {
        usize::try_from(options.threads)?
    };
    let bytes = maximum
        .checked_mul(4)
        .ok_or_else(|| invalid("statistics scratch size overflow"))?;
    let buffers = [
        driver.buffer(bytes, 16, 0)?,
        driver.buffer(bytes, 16, 0)?,
        driver.buffer(bytes, 16, 0)?,
    ];
    let mut reports = Vec::new();
    let mut assertions = 0;
    let mut total_chunks = 0_u64;
    let mut transport_groups = 0_u64;
    for t in &types {
        if !options.types.is_empty() && !t.name.as_ref().is_some_and(|n| options.types.contains(n))
        {
            continue;
        }
        if !t.cpu || !t.decoder {
            continue;
        }
        let type_name = t
            .name
            .as_deref()
            .ok_or_else(|| invalid("source-eligible quantizer name unavailable"))?;
        driver.null(0, json!({"Op":"qb-init-type","Type":t.id}))?;
        let mut global = Stats::default();
        let mut layer_reports = Vec::new();
        for layer in &layers {
            verify(
                contiguous(layer),
                format!("source contiguous assertion failed for{}", layer.name),
            )?;
            assertions += 1;
            row_size(t, layer.elements)?;
            let target = if options.per_layer_stats {
                Stats::default()
            } else {
                global.clone()
            };
            let work = Work {
                driver: &driver,
                model,
                layer,
                t,
                reference: options.reference,
                buffers,
            };
            let (stats, workers) = roundtrip_layer(&work, target, max_thread)?;
            total_chunks += u64::try_from(if workers == 1 {
                1
            } else {
                layer.elements.div_ceil(CHUNK)
            })?;
            transport_groups += u64::try_from(if workers == 1 {
                1
            } else {
                layer.elements.div_ceil(CHUNK * TRANSPORT_CHUNKS)
            })?;
            if options.per_layer_stats {
                layer_reports.push(json!({"Name":format!("{type_name}::{}",layer.name),"Workers":workers,"Statistics":stats.report(false)}));
                global.combine(&stats);
            } else {
                global = stats;
            }
        }
        reports.push(json!({"Type":t.id,"Name":t.name,"Quantizer":if options.reference{"reference"}else{"cpu"},"FromFloatReferenceAvailable":t.reference,"VecDotAvailable":t.dot,"Statistics":global.report(options.histogram),"Layers":layer_reports}));
    }
    driver.release("context", context)?;
    driver.release("model", model)?;
    let end = integer(
        &driver.call(0, json!({"Op":"qb-clock","Counter":"microseconds"}))?,
        "Value",
    )?;
    for buffer in buffers {
        driver.free(buffer)?;
    }
    session.finish()?;
    let mut output = result(
        ID,
        true,
        format!(
            "{} float layers across{} source-eligible types;{assertions} original contiguous predicates;{total_chunks} roundtrip chunks",
            layers.len(),
            reports.len()
        ),
    );
    output["SourceSha256"] =
        json!("1866b300a8966c3797740a5b4543236d92d1c7b8f2ca67829f453ffcd5b9707b");
    output["SourceAssertions"] = json!(assertions);
    output["Statistics"] = json!(reports);
    output["IncludedLayers"] = json!(layers.len());
    output["MaximumElements"] = json!(maximum);
    output["SourceContainsF16"] = json!(f16);
    output["Verbose"] = json!(options.verbose);
    output["VerboseLayers"] = json!(options.verbose.then(|| {
        layers
            .iter()
            .map(|layer| json!({"Name":layer.name,"Type":layer.type_id,"Elements":layer.elements}))
            .collect::<Vec<_>>()
    }));
    output["MaxThreads"] = json!(max_thread);
    output["TotalMicroseconds"] = json!(end.wrapping_sub(start));
    output["DefaultModelInput"] = json!(options.model.is_none().then_some(
        "Official immutable stories42M.bin float weights mapped to GGUF in safe Rust; all75 original tensors preserved, all counts divisible256"
    ));
    output["ModelPath"] = json!(path);
    output["Scheduling"] = json!(
        "Original Rust worker count, CHUNK-sized codec calls and local element/reduction order; mutex reserves up to16 consecutive chunks per transport group as a valid source schedule; up to16 persistent native production lanes, caller-directed continuation after each group error update"
    );
    output["TransportChunksPerGroup"] = json!(TRANSPORT_CHUNKS);
    output["TransportGroups"] = json!(transport_groups);
    Ok(output)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn grouped_transport_retains_source_calls_and_scalar_accumulation() -> Result<()> {
        for elements in [
            256,
            CHUNK,
            CHUNK + 256,
            TRANSPORT_CHUNKS * CHUNK,
            TRANSPORT_CHUNKS * CHUNK + 256,
        ] {
            let mut queue = Queue {
                counter: 0,
                target: Stats::default(),
            };
            let mut input = Vec::with_capacity(elements * 4);
            let mut decoded = Vec::with_capacity(elements * 4);
            let samples = [1.0_f32, 0.02, -0.04, 1.0e-12, 0.003, -0.0001, 1.0e5];
            for index in 0..elements {
                input.extend(samples[index % samples.len()].to_le_bytes());
                decoded.extend(0.0_f32.to_le_bytes());
            }
            let mut scalar = Stats::default();
            for (before, after) in input.chunks(4 * CHUNK).zip(decoded.chunks(4 * CHUNK)) {
                scalar.update(before, after)?;
            }
            let mut grouped = Stats::default();
            let mut calls = Vec::new();
            while let Some((offset, count)) = queue.claim(elements, &grouped)? {
                assert!(count <= TRANSPORT_CHUNKS * CHUNK);
                calls.extend(codec_chunks(offset, count, true));
                grouped.update(
                    &input[4 * offset..4 * (offset + count)],
                    &decoded[4 * offset..4 * (offset + count)],
                )?;
            }
            assert_eq!(calls.len(), elements.div_ceil(CHUNK));
            for (index, (offset, count)) in calls.into_iter().enumerate() {
                assert_eq!(offset, index * CHUNK);
                assert_eq!(count, (elements - offset).min(CHUNK));
            }
            assert_eq!(queue.target.report(true), scalar.report(true));
            // A requested single thread retains the source whole-layer call.
            assert_eq!(
                codec_chunks(0, elements, false).collect::<Vec<_>>(),
                vec![(0, elements)]
            );
        }
        Ok(())
    }
    #[test]
    fn worker_completion_preserves_native_error() -> Result<()> {
        let path = std::env::temp_dir().join(format!(
            "prism-stats-absent-checkpoint-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_nanos()
        ));
        let files = ChunkFiles {
            output: path.clone(),
            next: path.clone(),
            input: path.clone(),
            decoded: path,
        };
        let (sender, receiver) = mpsc::channel();
        sender.send(Err(invalid("native codec action file open failed")))?;
        drop(sender);
        let mut ended = false;
        let controlled = wait(&files, &receiver, &mut ended);
        let error = finish_trace(&receiver, ended, controlled)
            .err()
            .ok_or_else(|| invalid("native worker error was accepted"))?;
        verify(
            ended && error.to_string() == "native codec action file open failed",
            "consumed native worker error was replaced by disconnect",
        )?;
        let (sender, receiver) = mpsc::channel();
        sender.send(Ok(json!({"Trace":[]})))?;
        drop(sender);
        let error = finish_trace(&receiver, false, Err(invalid("controller payload failed")))
            .err()
            .ok_or_else(|| invalid("controller error was accepted"))?;
        verify(
            error.to_string() == "controller payload failed",
            "completed native trace replaced the controller error",
        )
    }
    #[test]
    #[allow(
        clippy::suboptimal_flops,
        reason = "Checks the original separate multiply/add accumulation without FMA."
    )]
    fn independent_error_math_and_options() -> Result<()> {
        let mut stats = Stats::default();
        let mut input = Vec::new();
        let mut output = Vec::new();
        for (a, b) in [(1.0_f32, 1.0_f32), (0.02, 0.0), (-0.04, 0.0)] {
            input.extend(a.to_le_bytes());
            output.extend(b.to_le_bytes());
        }
        stats.update(&input, &output)?;
        verify(
            stats.samples == 3
                && stats.histogram.iter().sum::<u64>() == 3
                && stats.histogram[0] == 1
                && stats.histogram[149] == 1,
            "histogram guard failed",
        )?;
        verify(
            stats.squared.to_bits()
                == (f64::from(0.02_f32).powi(2) + f64::from(0.04_f32).powi(2)).to_bits(),
            "double error accumulation differs",
        )?;
        let empty = Stats::default();
        verify(
            empty.quantile(0.95).to_bits() == 0x3f2a_36e2_eb1c_432c,
            "empty histogram source quantile differs",
        )?;
        verify(
            atoi(" -7suffix")? == -7 && atoi("text")? == 0,
            "atoi prefix semantics differ",
        )?;
        verify(
            parse_arguments(&["--verbose".into()]).is_err(),
            "source does not accept advertised --verbose",
        )?;
        verify(
            parse_arguments(&["--help".into()])?.is_none(),
            "help parsing differs",
        )?;
        let options = Options {
            exclude_layers: vec!["first-match".into(), "unreached-invalid".into()],
            ..Options::default()
        };
        let mut calls = 0;
        verify(
            !layer_included(&options, |pattern| {
                calls += 1;
                verify(pattern == "first-match", "unreachable regex was compiled")?;
                Ok(true)
            })? && calls == 1,
            "exclusion short circuit differs",
        )?;
        let options = Options {
            include_layers: vec!["first-match".into(), "unreached-invalid".into()],
            ..Options::default()
        };
        verify(
            layer_included(&options, |pattern| {
                verify(pattern == "first-match", "unreachable include was compiled")?;
                Ok(true)
            })?,
            "inclusion short circuit differs",
        )?;
        Ok(())
    }
}
