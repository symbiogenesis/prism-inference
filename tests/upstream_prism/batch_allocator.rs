//! All 26 default cases and 198 assertions from pinned test-batch-alloc.cpp.
//! Three additional observations retain the stronger owned-batch checks.
//!
//! Upstream MIT attribution: tools/prism/upstream/LICENSE. Typed batch inputs,
//! action traces and expected observations live in safe Rust; the adapter calls
//! the production `llama_batch_allocr` and returns its actual data.

use super::engine::Engine;
use super::{Result, invalid, result};
use serde_json::{Value, json};

type RuntimeInput = bonsai_prism_runtime::model::batch_allocator::Input;
type MemoryRanges = std::collections::BTreeMap<usize, (i32, i32)>;

const ID: &str = "tests/test-batch-alloc.cpp";
const SOURCE_SHA: &str = "341b3c4291cb46e412d6e750c2ab2e19131812d08ee3b236938c587e99341d79";
const ASSERTION_COUNT: usize = 198 + 3;

fn runtime_integer(value: &Value, minimum: i64, maximum: i64) -> Result<i64> {
    value
        .as_i64()
        .filter(|value| *value >= minimum && *value <= maximum)
        .ok_or_else(|| invalid("batch allocator protocol integer outside bound"))
}

fn runtime_size(value: &Value, minimum: i64, maximum: i64) -> Result<usize> {
    Ok(usize::try_from(runtime_integer(value, minimum, maximum)?)?)
}

fn runtime_array(value: &Value) -> Result<&[Value]> {
    value
        .as_array()
        .map(Vec::as_slice)
        .ok_or_else(|| invalid("batch allocator protocol array required"))
}

fn runtime_batch(batch: &bonsai_prism_runtime::model::batch_allocator::Batch) -> Value {
    json!({"Embeddings":batch.embeddings,"Tokens":batch.tokens,"Positions":batch.positions,
        "SequenceIds":batch.sequence_ids,"Logits":batch.logits})
}

fn runtime_state(
    allocator: &bonsai_prism_runtime::model::batch_allocator::BatchAllocator,
    limit: usize,
) -> Result<Value> {
    let batch = allocator
        .batch()
        .ok_or_else(|| invalid("batch allocator has no initialized input"))?;
    let positions = (0..limit)
        .map(|id| {
            let (min, max) = allocator
                .sequence_position(id)
                .ok_or_else(|| invalid("batch sequence ID outside bound"))?;
            Ok(json!({"Min":min,"Max":max}))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(
        json!({"NTokens":batch.len(),"NOutputs":batch.logits.iter().filter(|flag|**flag!=0).count(),
        "NUsed":allocator.used(),"Batch":runtime_batch(batch),"OutIds":allocator.output_ids(),"SequencePositions":positions}),
    )
}

fn runtime_ubatch(value: &bonsai_prism_runtime::model::batch_allocator::MicroBatch) -> Value {
    let data = value.data.as_ref();
    json!({"NTokens":data.map_or(0, bonsai_prism_runtime::model::batch_allocator::Batch::len),
        "NSequenceTokens":value.sequence_tokens,"NSequences":value.sequences,"NUniqueSequences":value.unique_sequence_ids.len(),
        "NPositions":value.positions,"EqualSequences":value.equal_sequences,"IsPosition2D":value.positions>=3,
        "HasTokens":data.is_some_and(|data|data.tokens.is_some()),"HasEmbeddings":data.is_some_and(|data|data.embeddings.is_some()),
        "Tokens":data.and_then(|data|data.tokens.as_ref()),"Embeddings":data.and_then(|data|data.embeddings.as_ref()),
        "Positions":data.map(|data|&data.positions),"SequenceIds":data.map(|data|&data.sequence_ids),
        "UniqueSequenceIds":data.map(|_|&value.unique_sequence_ids),"SequenceIndices":data.map(|_|&value.sequence_indices),"Output":data.map(|data|&data.logits)})
}

fn runtime_input(action: &Value, position_sections: usize) -> Result<(RuntimeInput, MemoryRanges)> {
    use bonsai_prism_runtime::model::batch_allocator::Input;
    fields(
        action,
        &[
            "Op",
            "EmbeddingWidth",
            "Embeddings",
            "Tokens",
            "Positions",
            "SequenceIds",
            "Logits",
            "SequenceLimit",
            "OutputAll",
            "MemoryRanges",
        ],
    )?;
    let width = runtime_size(&action["EmbeddingWidth"], 0, 8)?;
    let tokens = runtime_integers(&action["Tokens"])?;
    let embeddings = if action["Embeddings"].is_null() {
        None
    } else {
        Some(serde_json::from_value::<Vec<f32>>(
            action["Embeddings"].clone(),
        )?)
    };
    let count = match (&tokens, &embeddings) {
        (Some(tokens), None) if width == 0 => tokens.len(),
        (None, Some(embeddings)) if width > 0 && embeddings.len().is_multiple_of(width) => {
            embeddings.len() / width
        }
        _ => {
            return Err(invalid(
                "batch requires either tokens or complete embeddings",
            ));
        }
    };
    if count == 0 || count > 64 {
        return Err(invalid("batch protocol token count outside bound"));
    }
    let positions = runtime_integers(&action["Positions"])?;
    if positions.as_ref().is_some_and(|positions| {
        positions.len()
            != count
                * if tokens.is_some() {
                    1
                } else {
                    position_sections
                }
    }) {
        return Err(invalid("batch protocol position extent mismatch"));
    }
    let sequence_ids = if action["SequenceIds"].is_null() {
        None
    } else {
        let values = runtime_array(&action["SequenceIds"])?;
        if values.len() != count {
            return Err(invalid("batch protocol sequence extent mismatch"));
        }
        let mut ids = Vec::with_capacity(count);
        for value in values {
            let sequence = runtime_array(value)?;
            if sequence.is_empty() || sequence.len() > 8 {
                return Err(invalid("batch protocol memberships outside bound"));
            }
            ids.push(
                sequence
                    .iter()
                    .map(|value| Ok(i32::try_from(runtime_integer(value, -1, 256)?)?))
                    .collect::<Result<Vec<_>>>()?,
            );
        }
        Some(ids)
    };
    let logits = if action["Logits"].is_null() {
        None
    } else {
        let values = runtime_array(&action["Logits"])?;
        if values.len() != count {
            return Err(invalid("batch protocol output extent mismatch"));
        }
        Some(
            values
                .iter()
                .map(|value| Ok(i8::try_from(runtime_integer(value, 0, 1)?)?))
                .collect::<Result<Vec<_>>>()?,
        )
    };
    Ok((
        Input {
            tokens,
            embeddings,
            embedding_width: width,
            positions,
            sequence_ids,
            logits,
        },
        runtime_memory(action)?,
    ))
}

fn runtime_integers(value: &Value) -> Result<Option<Vec<i32>>> {
    if value.is_null() {
        return Ok(None);
    }
    Ok(Some(
        runtime_array(value)?
            .iter()
            .map(|value| Ok(i32::try_from(runtime_integer(value, -32768, 32768)?)?))
            .collect::<Result<Vec<_>>>()?,
    ))
}

fn runtime_memory(action: &Value) -> Result<MemoryRanges> {
    let mut ranges = MemoryRanges::new();
    if !action["MemoryRanges"].is_null() {
        let values = runtime_array(&action["MemoryRanges"])?;
        if values.len() > 16 {
            return Err(invalid("batch memory range bound exceeded"));
        }
        for value in values {
            fields(value, &["Sequence", "Min", "Max"])?;
            let id = runtime_size(&value["Sequence"], 0, 255)?;
            let range = (
                i32::try_from(runtime_integer(&value["Min"], -1, 32768)?)?,
                i32::try_from(runtime_integer(&value["Max"], -1, 32768)?)?,
            );
            if ranges.insert(id, range).is_some() {
                return Err(invalid("duplicate batch memory range"));
            }
        }
    }
    Ok(ranges)
}

fn runtime_operation(operation: &Value) -> Result<Value> {
    use bonsai_prism_runtime::model::batch_allocator::BatchAllocator;
    fields(operation, &["Kind", "PositionsPerEmbedding", "Actions"])?;
    let positions = runtime_size(&operation["PositionsPerEmbedding"], 1, 4)?;
    let mut allocator = BatchAllocator::new(positions)?;
    let actions = runtime_array(&operation["Actions"])?;
    if actions.is_empty() || actions.len() > 64 {
        return Err(invalid("batch protocol action count outside bound"));
    }
    let mut limit = 0;
    let mut observations = Vec::with_capacity(actions.len());
    for action in actions {
        let op = action["Op"]
            .as_str()
            .ok_or_else(|| invalid("batch protocol action Op must be text"))?;
        let event = match op {
            "init" => {
                let (input, memory) = runtime_input(action, positions)?;
                limit = runtime_size(&action["SequenceLimit"], 1, 257)?;
                let output_all = action["OutputAll"]
                    .as_bool()
                    .ok_or_else(|| invalid("batch OutputAll must be boolean"))?;
                let initialized = allocator.init(
                    input,
                    0,
                    limit,
                    output_all,
                    if action["MemoryRanges"].is_null() {
                        None
                    } else {
                        Some(&memory)
                    },
                )?;
                json!({"Initialized":initialized,"State":if initialized{runtime_state(&allocator,limit)?}else{Value::Null}})
            }
            "reserve" => {
                fields(action, &["Op", "SequenceTokens", "Sequences"])?;
                let tokens = runtime_size(&action["SequenceTokens"], 1, 16)?;
                let sequences = runtime_size(&action["Sequences"], 1, 16)?;
                if tokens * sequences > 64 {
                    return Err(invalid("batch reservation exceeds bound"));
                }
                json!({"Ubatch":runtime_ubatch(&allocator.reserve(tokens,sequences)?),"State":null})
            }
            "split-reset" => {
                fields(action, &["Op"])?;
                if allocator.batch().is_none() {
                    return Err(invalid("batch reset requires initialized input"));
                }
                allocator.split_reset();
                json!({"State":runtime_state(&allocator,limit)?})
            }
            "split-simple" | "split-seq" | "split-equal" => {
                fields(
                    action,
                    if op == "split-equal" {
                        &["Op", "Capacity", "Sequential", "KeepTail"]
                    } else {
                        &["Op", "Capacity"]
                    },
                )?;
                let capacity = runtime_size(&action["Capacity"], 1, 64)?;
                let ubatch = match op {
                    "split-simple" => allocator.split_simple(capacity)?,
                    "split-seq" => allocator.split_sequence(capacity)?,
                    _ => allocator.split_equal(
                        capacity,
                        action["Sequential"]
                            .as_bool()
                            .ok_or_else(|| invalid("batch Sequential must be boolean"))?,
                        runtime_size(&action["KeepTail"], 0, 64)?,
                    )?,
                };
                json!({"Ubatch":runtime_ubatch(&ubatch),"State":runtime_state(&allocator,limit)?})
            }
            _ => return Err(invalid("unknown batch protocol action")),
        };
        observations.push(event);
    }
    Ok(json!({"Actions":observations}))
}

pub(super) fn execute_runtime(request: &Value) -> Result<Option<Value>> {
    fields(request, &["SchemaVersion", "Operations"])?;
    if request["SchemaVersion"] != 1 {
        return Err(invalid("batch protocol version mismatch"));
    }
    let operations = runtime_array(&request["Operations"])?;
    if operations.is_empty() || operations.len() > 128 {
        return Err(invalid("batch protocol operation count outside bound"));
    }
    if operations
        .iter()
        .any(|operation| operation["Kind"] != "batch-allocator")
    {
        return Ok(None);
    }
    Ok(Some(
        json!({"SchemaVersion":1,"Results":operations.iter().map(runtime_operation).collect::<Result<Vec<_>>>()?}),
    ))
}

#[derive(Default)]
struct Batch {
    embeddings: Vec<f32>,
    positions: Vec<i32>,
    sequences: Vec<Vec<i32>>,
    logits: Vec<i32>,
    token: Option<i32>,
}
impl Batch {
    #[allow(
        clippy::suboptimal_flops,
        reason = "Preserve the source embedding fixture's arithmetic expression."
    )]
    fn add(&mut self, position: i32, ids: &[i32], output: bool) -> Result<()> {
        let i = f32::from(u16::try_from(self.sequences.len())?);
        self.embeddings.extend([100.0 * i, 100.0 * i + 1.0]);
        self.positions.push(position);
        self.sequences.push(ids.to_vec());
        self.logits.push(i32::from(output));
        Ok(())
    }
    fn sequence(&mut self, count: i32, id: i32) -> Result<()> {
        for i in 0..count {
            self.add(i, &[id], i == count - 1)?;
        }
        Ok(())
    }
    fn init(&self, options: Options) -> Value {
        let token = self.token;
        json!({"Op":"init","EmbeddingWidth":if token.is_some(){0}else{2},
            "Embeddings":if token.is_some(){Value::Null}else{json!(self.embeddings)},
            "Tokens":token.map_or(Value::Null, |token|json!([token])),
            "Positions":if options.positions{json!(self.positions)}else{Value::Null},
            "SequenceIds":if options.sequences{json!(self.sequences)}else{Value::Null},
            "Logits":if options.logits{json!(self.logits)}else{Value::Null},
            "SequenceLimit":options.limit,"OutputAll":options.output_all,
            "MemoryRanges":options.memory.map(|ranges|ranges.into_iter().map(|(id,min,max)|json!({"Sequence":id,"Min":min,"Max":max})).collect::<Vec<_>>())})
    }
}
#[allow(
    clippy::struct_excessive_bools,
    reason = "Independent source batch pointer-presence and output-all options."
)]
struct Options {
    positions: bool,
    sequences: bool,
    logits: bool,
    output_all: bool,
    limit: i32,
    memory: Option<Vec<(i32, i32, i32)>>,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            positions: true,
            sequences: true,
            logits: true,
            output_all: false,
            limit: 4,
            memory: None,
        }
    }
}
struct Assertion {
    path: String,
    expected: Value,
    length: bool,
}
struct Case {
    name: &'static str,
    positions: u32,
    actions: Vec<Value>,
    assertions: Vec<Assertion>,
}
impl Case {
    const fn new(name: &'static str) -> Self {
        Self {
            name,
            positions: 1,
            actions: Vec::new(),
            assertions: Vec::new(),
        }
    }
    fn action(&mut self, action: Value) -> usize {
        let index = self.actions.len();
        self.actions.push(action);
        index
    }
    fn equal(&mut self, action: usize, suffix: &str, expected: Value) {
        self.assertions.push(Assertion {
            path: format!("/Actions/{action}/{suffix}"),
            expected,
            length: false,
        });
    }
    fn length(&mut self, action: usize, suffix: &str, expected: usize) {
        self.assertions.push(Assertion {
            path: format!("/Actions/{action}/{suffix}"),
            expected: json!(expected),
            length: true,
        });
    }
    fn init(&mut self, batch: &Batch, options: Options, accepted: bool) -> usize {
        let action = self.action(batch.init(options));
        self.equal(action, "Initialized", json!(accepted));
        action
    }
    fn simple(&mut self, capacity: u32) -> usize {
        self.action(json!({"Op":"split-simple","Capacity":capacity}))
    }
    fn equal_split(&mut self, capacity: u32, sequential: bool, tail: u32) -> usize {
        self.action(
            json!({"Op":"split-equal","Capacity":capacity,"Sequential":sequential,"KeepTail":tail}),
        )
    }
    fn request(&self) -> Value {
        json!({"Kind":"batch-allocator","PositionsPerEmbedding":self.positions,"Actions":self.actions})
    }
}

#[allow(
    clippy::too_many_lines,
    reason = "Literal source case construction and expected assertions remain in the original order for parity review."
)]
fn cases() -> Result<Vec<Case>> {
    let mut cases = Vec::new();
    let mut c = Case::new("init/rejects_n_seq_max_too_large");
    let mut b = Batch::default();
    b.add(0, &[0], true)?;
    c.init(
        &b,
        Options {
            limit: 257,
            ..Options::default()
        },
        false,
    );
    cases.push(c);

    let mut c = Case::new("init/rejects_invalid_token");
    for token in [0, -1] {
        let b = Batch {
            token: Some(token),
            ..Batch::default()
        };
        c.init(
            &b,
            Options {
                positions: false,
                sequences: false,
                logits: false,
                limit: 1,
                ..Options::default()
            },
            false,
        );
    }
    cases.push(c);

    let mut c = Case::new("init/rejects_invalid_seq_id");
    for id in [4, -1] {
        let mut b = Batch::default();
        b.add(0, &[id], true)?;
        c.init(&b, Options::default(), false);
    }
    cases.push(c);

    let mut c = Case::new("init/autofill_defaults");
    let mut b = Batch::default();
    for _ in 0..4 {
        b.add(0, &[0], false)?;
    }
    let a = c.init(
        &b,
        Options {
            positions: false,
            sequences: false,
            logits: false,
            ..Options::default()
        },
        true,
    );
    c.equal(a, "State/NTokens", json!(4));
    for i in 0..4 {
        c.equal(a, &format!("State/Batch/Positions/{i}"), json!(i));
        c.length(a, &format!("State/Batch/SequenceIds/{i}"), 1);
        c.equal(a, &format!("State/Batch/SequenceIds/{i}/0"), json!(0));
    }
    c.equal(a, "State/NOutputs", json!(1));
    c.equal(a, "State/Batch/Logits/0", json!(0));
    c.equal(a, "State/Batch/Logits/3", json!(1));
    c.equal(a, "State/Batch/Logits", json!([0, 0, 0, 1]));
    c.equal(a, "State/SequencePositions/0/Min", json!(0));
    c.equal(a, "State/SequencePositions/0/Max", json!(3));
    c.equal(a, "State/SequencePositions/1/Min", json!(-1));
    cases.push(c);

    let mut c = Case::new("init/output_all");
    let mut b = Batch::default();
    for i in 0..4 {
        b.add(i, &[0], false)?;
    }
    let a = c.init(
        &b,
        Options {
            logits: false,
            output_all: true,
            ..Options::default()
        },
        true,
    );
    c.equal(a, "State/NOutputs", json!(4));
    cases.push(c);

    let mut c = Case::new("init/explicit_logits");
    let mut b = Batch::default();
    b.add(0, &[0], true)?;
    b.add(1, &[0], false)?;
    b.add(2, &[0], true)?;
    let a = c.init(&b, Options::default(), true);
    c.equal(a, "State/NOutputs", json!(2));
    let a = c.simple(10);
    c.equal(a, "Ubatch/NTokens", json!(3));
    for (i, value) in [1, 0, 1].into_iter().enumerate() {
        c.equal(a, &format!("Ubatch/Output/{i}"), json!(value));
    }
    c.length(a, "State/OutIds", 2);
    c.equal(a, "State/OutIds/0", json!(0));
    c.equal(a, "State/OutIds/1", json!(2));
    cases.push(c);

    let mut c = Case::new("init/pos_from_memory");
    let mut b = Batch::default();
    for _ in 0..3 {
        b.add(0, &[0], false)?;
    }
    let a = c.init(
        &b,
        Options {
            positions: false,
            logits: false,
            memory: Some(vec![(0, 0, 9)]),
            ..Options::default()
        },
        true,
    );
    c.equal(a, "State/SequencePositions/0/Min", json!(10));
    c.equal(a, "State/SequencePositions/0/Max", json!(12));
    cases.push(c);

    let mut c = Case::new("init/pos_continuity_with_memory");
    for (first, accepted) in [(10, true), (11, false), (9, false)] {
        let mut b = Batch::default();
        b.add(first, &[0], false)?;
        b.add(first + 1, &[0], true)?;
        c.init(
            &b,
            Options {
                memory: Some(vec![(0, 0, 9)]),
                ..Options::default()
            },
            accepted,
        );
    }
    cases.push(c);

    let mut c = Case::new("init/rejects_non_continuous_positions");
    let mut b = Batch::default();
    for i in [0, 1, 3] {
        b.add(i, &[0], i == 3)?;
    }
    c.init(&b, Options::default(), false);
    cases.push(c);
    let mut c = Case::new("init/rejects_decreasing_positions");
    let mut b = Batch::default();
    for (pos, id) in [4, 5, 0, 1, 6, 2, 3].into_iter().zip([0, 0, 1, 1, 0, 1, 0]) {
        b.add(pos, &[id], false)?;
    }
    c.init(
        &b,
        Options {
            logits: false,
            ..Options::default()
        },
        false,
    );
    cases.push(c);
    let mut c = Case::new("init/allows_equal_positions_in_seq");
    let mut b = Batch::default();
    b.add(0, &[0], false)?;
    b.add(0, &[0], false)?;
    b.add(1, &[0], true)?;
    c.init(&b, Options::default(), true);
    cases.push(c);
    let mut c = Case::new("init/rejects_coupled_diverged_seqs");
    let mut b = Batch::default();
    b.add(6, &[0, 1], true)?;
    c.init(
        &b,
        Options {
            memory: Some(vec![(0, 0, 5), (1, 2, 5)]),
            ..Options::default()
        },
        false,
    );
    c.init(
        &b,
        Options {
            memory: Some(vec![(0, 0, 5), (1, 0, 5)]),
            ..Options::default()
        },
        true,
    );
    cases.push(c);

    let mut c = Case::new("split/split_simple_chunks");
    let mut b = Batch::default();
    b.sequence(5, 0)?;
    c.init(&b, Options::default(), true);
    let a = c.simple(2);
    c.equal(a, "Ubatch/NTokens", json!(2));
    c.equal(a, "Ubatch/EqualSequences", json!(false));
    c.equal(a, "Ubatch/NUniqueSequences", json!(1));
    c.equal(a, "Ubatch/UniqueSequenceIds/0", json!(0));
    c.equal(a, "Ubatch/SequenceIndices/0", json!(0));
    for i in 0..2 {
        c.equal(a, &format!("Ubatch/Positions/{i}"), json!(i));
        c.length(a, &format!("Ubatch/SequenceIds/{i}"), 1);
        c.equal(a, &format!("Ubatch/SequenceIds/{i}/0"), json!(0));
        let base = f32::from(u16::try_from(i)?) * 100.0;
        c.equal(a, &format!("Ubatch/Embeddings/{}", i * 2), json!(base));
        c.equal(
            a,
            &format!("Ubatch/Embeddings/{}", i * 2 + 1),
            json!(base + 1.0),
        );
    }
    let a = c.simple(2);
    c.equal(a, "Ubatch/NTokens", json!(2));
    c.equal(a, "Ubatch/Positions/0", json!(2));
    c.equal(a, "Ubatch/Positions/1", json!(3));
    let a = c.simple(2);
    c.equal(a, "Ubatch/NTokens", json!(1));
    c.equal(a, "Ubatch/Positions/0", json!(4));
    c.equal(a, "Ubatch/Output/0", json!(1));
    c.equal(a, "State/NUsed", json!(5));
    let a = c.simple(2);
    c.equal(a, "Ubatch/NTokens", json!(0));
    c.length(a, "State/OutIds", 1);
    c.equal(a, "State/OutIds/0", json!(4));
    cases.push(c);

    let mut c = Case::new("split/split_reset_allows_resplit");
    let mut b = Batch::default();
    b.sequence(3, 0)?;
    c.init(&b, Options::default(), true);
    for _ in 0..3 {
        c.simple(1);
    }
    let a = c.simple(1);
    c.equal(a, "State/NUsed", json!(3));
    let a = c.action(json!({"Op":"split-reset"}));
    c.equal(a, "State/NUsed", json!(0));
    let a = c.simple(10);
    c.equal(a, "Ubatch/NTokens", json!(3));
    cases.push(c);

    let mut c = Case::new("split/split_equal_unequal_lengths");
    let mut b = Batch::default();
    b.sequence(4, 0)?;
    b.sequence(2, 1)?;
    c.init(&b, Options::default(), true);
    let a = c.equal_split(8, false, 0);
    c.equal(a, "Ubatch/EqualSequences", json!(true));
    c.equal(a, "Ubatch/NTokens", json!(4));
    c.equal(a, "Ubatch/NSequenceTokens", json!(2));
    c.equal(a, "Ubatch/NSequences", json!(2));
    c.equal(a, "Ubatch/NUniqueSequences", json!(2));
    for (i, id) in [0, 0, 1, 1].into_iter().enumerate() {
        c.equal(a, &format!("Ubatch/SequenceIds/{i}/0"), json!(id));
    }
    for (i, pos) in [0, 1, 0, 1].into_iter().enumerate() {
        c.equal(a, &format!("Ubatch/Positions/{i}"), json!(pos));
    }
    let a = c.equal_split(8, false, 0);
    c.equal(a, "Ubatch/NTokens", json!(2));
    c.equal(a, "Ubatch/NSequences", json!(1));
    c.equal(a, "Ubatch/Positions/0", json!(2));
    c.equal(a, "Ubatch/Positions/1", json!(3));
    let a = c.equal_split(8, false, 0);
    c.equal(a, "Ubatch/NTokens", json!(0));
    c.equal(a, "State/NUsed", json!(6));
    cases.push(c);

    let mut c = Case::new("split/split_equal_coupled");
    let mut b = Batch::default();
    b.add(0, &[0, 1], false)?;
    b.add(1, &[0, 1], true)?;
    c.init(&b, Options::default(), true);
    let a = c.equal_split(4, true, 0);
    c.equal(a, "Ubatch/NTokens", json!(0));
    let a = c.equal_split(4, false, 0);
    c.equal(a, "Ubatch/NTokens", json!(2));
    c.equal(a, "Ubatch/NSequences", json!(1));
    c.equal(a, "Ubatch/NUniqueSequences", json!(2));
    c.length(a, "Ubatch/SequenceIds/0", 2);
    c.equal(a, "Ubatch/SequenceIndices/0", json!(0));
    c.equal(a, "Ubatch/SequenceIndices/1", json!(1));
    c.equal(a, "Ubatch/UniqueSequenceIds", json!([0, 1]));
    c.equal(a, "State/OutIds", json!([1]));
    cases.push(c);

    let mut c = Case::new("split/split_seq_per_sequence");
    let mut b = Batch::default();
    for id in 0..3 {
        b.sequence(2, id)?;
    }
    c.init(&b, Options::default(), true);
    for id in 0..3 {
        let a = c.action(json!({"Op":"split-seq","Capacity":8}));
        c.equal(a, "Ubatch/NTokens", json!(2));
        c.equal(a, "Ubatch/NSequences", json!(1));
        c.equal(a, "Ubatch/SequenceIds/0/0", json!(id));
        c.equal(a, "Ubatch/UniqueSequenceIds/0", json!(id));
    }
    let a = c.action(json!({"Op":"split-seq","Capacity":8}));
    c.equal(a, "Ubatch/NTokens", json!(0));
    c.equal(a, "State/NUsed", json!(6));
    cases.push(c);

    let mut c = Case::new("split/ubatch_reserve");
    let a = c.action(json!({"Op":"reserve","SequenceTokens":3,"Sequences":2}));
    c.equal(a, "Ubatch/NTokens", json!(6));
    c.equal(a, "Ubatch/NSequenceTokens", json!(3));
    c.equal(a, "Ubatch/NSequences", json!(2));
    c.equal(a, "Ubatch/NUniqueSequences", json!(2));
    c.equal(a, "Ubatch/EqualSequences", json!(true));
    c.equal(a, "Ubatch/UniqueSequenceIds/0", json!(0));
    c.equal(a, "Ubatch/UniqueSequenceIds/1", json!(1));
    c.equal(a, "Ubatch/HasTokens", json!(true));
    c.equal(a, "Ubatch/HasEmbeddings", json!(false));
    cases.push(c);

    let mut c = Case::new("keep_tail/noop_when_seqs_complete");
    let mut b = Batch::default();
    b.sequence(2, 0)?;
    b.sequence(2, 1)?;
    c.init(&b, Options::default(), true);
    let a = c.equal_split(4, false, 2);
    c.equal(a, "Ubatch/NTokens", json!(4));
    c.equal(a, "Ubatch/NSequences", json!(2));
    c.equal(a, "Ubatch/NSequenceTokens", json!(2));
    let a = c.equal_split(4, false, 2);
    c.equal(a, "Ubatch/NTokens", json!(0));
    cases.push(c);

    let mut c = Case::new("keep_tail/defers_seq_with_short_remainder");
    let mut b = Batch::default();
    b.sequence(2, 0)?;
    b.sequence(3, 1)?;
    c.init(&b, Options::default(), true);
    let a = c.equal_split(4, true, 2);
    c.equal(a, "Ubatch/NTokens", json!(2));
    c.equal(a, "Ubatch/NSequences", json!(1));
    c.equal(a, "Ubatch/SequenceIds/0/0", json!(0));
    c.equal(a, "State/NUsed", json!(2));
    let a = c.equal_split(4, true, 2);
    c.equal(a, "Ubatch/NTokens", json!(3));
    c.equal(a, "Ubatch/NSequences", json!(1));
    c.equal(a, "Ubatch/SequenceIds/0/0", json!(1));
    for i in 0..3 {
        c.equal(a, &format!("Ubatch/Positions/{i}"), json!(i));
    }
    c.equal(a, "State/NUsed", json!(5));
    let a = c.equal_split(4, true, 2);
    c.equal(a, "Ubatch/NTokens", json!(0));
    cases.push(c);

    let mut c = Case::new("keep_tail/completes_first_seq_when_all_violate");
    let mut b = Batch::default();
    b.sequence(3, 0)?;
    b.sequence(3, 1)?;
    c.init(&b, Options::default(), true);
    let a = c.equal_split(4, false, 2);
    c.equal(a, "Ubatch/NTokens", json!(3));
    c.equal(a, "Ubatch/NSequences", json!(1));
    c.equal(a, "Ubatch/NSequenceTokens", json!(3));
    c.equal(a, "Ubatch/SequenceIds/0/0", json!(0));
    for i in 0..3 {
        c.equal(a, &format!("Ubatch/Positions/{i}"), json!(i));
    }
    c.equal(a, "State/NUsed", json!(3));
    let a = c.equal_split(4, false, 2);
    c.equal(a, "Ubatch/NTokens", json!(3));
    c.equal(a, "Ubatch/SequenceIds/0/0", json!(1));
    c.equal(a, "State/NUsed", json!(6));
    cases.push(c);

    let mut c = Case::new("keep_tail/truncates_to_preserve_tail");
    let mut b = Batch::default();
    b.sequence(5, 0)?;
    c.init(&b, Options::default(), true);
    let a = c.equal_split(4, false, 2);
    c.equal(a, "Ubatch/NTokens", json!(3));
    c.equal(a, "Ubatch/NSequences", json!(1));
    c.equal(a, "Ubatch/Positions/2", json!(2));
    c.equal(a, "State/NUsed", json!(3));
    let a = c.equal_split(4, false, 2);
    c.equal(a, "Ubatch/NTokens", json!(2));
    c.equal(a, "Ubatch/Positions/0", json!(3));
    c.equal(a, "Ubatch/Positions/1", json!(4));
    c.equal(a, "Ubatch/Output/1", json!(1));
    c.equal(a, "State/NUsed", json!(5));
    cases.push(c);

    let mut c = Case::new("keep_tail/keeps_full_ubatch_with_sufficient_remainder");
    let mut b = Batch::default();
    b.sequence(6, 0)?;
    c.init(&b, Options::default(), true);
    let a = c.equal_split(4, false, 2);
    c.equal(a, "Ubatch/NTokens", json!(4));
    let a = c.equal_split(4, false, 2);
    c.equal(a, "Ubatch/NTokens", json!(2));
    c.equal(a, "Ubatch/Positions/0", json!(4));
    c.equal(a, "Ubatch/Positions/1", json!(5));
    c.equal(a, "State/NUsed", json!(6));
    cases.push(c);

    let mut c = Case::new("keep_tail/multi_seq_prefix_kept");
    let mut b = Batch::default();
    b.sequence(3, 0)?;
    b.sequence(4, 1)?;
    c.init(
        &b,
        Options {
            limit: 6,
            ..Options::default()
        },
        true,
    );
    let a = c.equal_split(6, true, 2);
    c.equal(a, "Ubatch/NTokens", json!(3));
    c.equal(a, "Ubatch/NSequences", json!(1));
    c.equal(a, "Ubatch/SequenceIds/0/0", json!(0));
    c.equal(a, "State/NUsed", json!(3));
    let a = c.equal_split(6, true, 2);
    c.equal(a, "Ubatch/NTokens", json!(4));
    c.equal(a, "Ubatch/SequenceIds/0/0", json!(1));
    c.equal(a, "State/NUsed", json!(7));
    cases.push(c);

    let mut c = Case::new("mrope/pos_layout_and_split");
    c.positions = 4;
    let mut b = Batch::default();
    b.add(10, &[0], false)?;
    b.add(11, &[0], true)?;
    b.positions = vec![10, 11, 5, 6, 7, 8, 0, 0];
    c.init(&b, Options::default(), true);
    let a = c.simple(2);
    c.equal(a, "Ubatch/NTokens", json!(2));
    c.equal(a, "Ubatch/NPositions", json!(4));
    c.equal(a, "Ubatch/IsPosition2D", json!(true));
    for (i, pos) in [10, 11, 5, 6, 7, 8, 0, 0].into_iter().enumerate() {
        c.equal(a, &format!("Ubatch/Positions/{i}"), json!(pos));
    }
    cases.push(c);

    let mut c = Case::new("mrope/pos_jump_allowed");
    c.positions = 4;
    for (pos, accepted) in [(15, true), (9, true), (8, false)] {
        let mut b = Batch::default();
        b.add(pos, &[0], true)?;
        b.positions = vec![pos, 1, 1, 0];
        c.init(
            &b,
            Options {
                memory: Some(vec![(0, 0, 9)]),
                ..Options::default()
            },
            accepted,
        );
    }
    cases.push(c);
    if cases.len() != 26
        || cases.iter().map(|c| c.assertions.len()).sum::<usize>() != ASSERTION_COUNT
    {
        return Err(invalid("batch allocator source corpus count mismatch"));
    }
    Ok(cases)
}

fn fields(value: &Value, names: &[&str]) -> Result<()> {
    let object = value
        .as_object()
        .ok_or_else(|| invalid("batch allocator response must be an object"))?;
    if object.len() != names.len() || names.iter().any(|name| !object.contains_key(*name)) {
        return Err(invalid("batch allocator response fields mismatch"));
    }
    Ok(())
}
fn observations<'a>(response: &'a Value, cases: &[Case]) -> Result<&'a [Value]> {
    fields(response, &["SchemaVersion", "Results"])?;
    if response["SchemaVersion"] != 1 {
        return Err(invalid("batch allocator response version mismatch"));
    }
    let values = response["Results"]
        .as_array()
        .ok_or_else(|| invalid("batch allocator Results must be array"))?;
    if values.len() != cases.len() {
        return Err(invalid("batch allocator result count mismatch"));
    }
    for (value, case) in values.iter().zip(cases) {
        fields(value, &["Actions"])?;
        let actions = value["Actions"]
            .as_array()
            .ok_or_else(|| invalid("batch allocator Actions must be array"))?;
        if actions.len() != case.actions.len() {
            return Err(invalid("batch allocator action count mismatch"));
        }
    }
    Ok(values)
}
fn compare(case: &Case, value: &Value) -> Result<Vec<String>> {
    let mut failures = Vec::new();
    for assertion in &case.assertions {
        let actual = value.pointer(&assertion.path).ok_or_else(|| {
            invalid(format!(
                "batch allocator missing observation {}",
                assertion.path
            ))
        })?;
        let actual = if assertion.length {
            json!(
                actual
                    .as_array()
                    .ok_or_else(|| invalid("batch allocator length observation must be array"))?
                    .len()
            )
        } else {
            actual.clone()
        };
        if actual != assertion.expected {
            failures.push(format!(
                "{} {} expected={} actual={actual}",
                case.name, assertion.path, assertion.expected
            ));
        }
    }
    Ok(failures)
}
pub(super) fn run(engine: &Engine) -> Result<Value> {
    let cases = cases()?;
    let request =
        json!({"SchemaVersion":1,"Operations":cases.iter().map(Case::request).collect::<Vec<_>>()});
    let Some(response) = engine.execute_common("batch-allocator", &request)? else {
        return Ok(result(
            ID,
            false,
            "unsupported: production batch allocator API is missing".to_owned(),
        ));
    };
    let values = observations(&response, &cases)?;
    let mut failures = Vec::new();
    for (case, value) in cases.iter().zip(values) {
        failures.extend(compare(case, value)?);
    }
    Ok(result(
        ID,
        failures.is_empty(),
        format!(
            "source={SOURCE_SHA}; all26 cases and198 original assertions plus3 additional observations; failures={}; examples={:?}",
            failures.len(),
            failures.iter().take(12).collect::<Vec<_>>()
        ),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_missing_action_and_detects_changed_expected_observation() -> Result<()> {
        // Only construct the source input/expectation inventory here. The
        // complete allocator source contracts execute in hosted acceptance.
        let inventory = cases()?;
        assert_eq!(inventory.len(), 26);
        assert_eq!(
            inventory
                .iter()
                .map(|case| case.assertions.len())
                .sum::<usize>(),
            ASSERTION_COUNT
        );
        let mut case = Case::new("focused");
        case.equal(0, "Ubatch/Positions/0", json!(11));
        assert!(compare(&case, &json!({"Actions":[]})).is_err());
        assert_eq!(
            compare(&case, &json!({"Actions":[{"Ubatch":{"Positions":[11]}}]}))?,
            Vec::<String>::new()
        );
        assert_eq!(
            compare(&case, &json!({"Actions":[{"Ubatch":{"Positions":[12]}}]}))?.len(),
            1
        );
        Ok(())
    }
}
