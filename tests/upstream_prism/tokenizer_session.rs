//! Bounded rotation of stateless production vocabulary observations.
use super::engine::Engine;
#[path = "tokenizer_runtime.rs"]
mod tokenizer_runtime;
use super::{Result, invalid};
use serde_json::{Value, json};
use std::path::{Path, PathBuf};
use std::time::Instant;
pub(super) use tokenizer_runtime::{Client, ModelSession};

const ROW_LIMIT: usize = 65536;
const WIRE_LIMIT: usize = 16 * 1024 * 1024;
struct State {
    session: ModelSession,
    client: Client,
    model: u64,
    context: u64,
}
pub(super) struct VocabularySession<'a> {
    engine: &'a Engine,
    name: String,
    path: PathBuf,
    state: Option<State>,
    rows: usize,
    wire: usize,
    started: Instant,
    leases: Vec<Value>,
}
pub(super) struct Observation {
    pub(super) tokens: Value,
    pub(super) texts: Value,
    pub(super) encode_ns: u128,
    pub(super) decode_ns: u128,
}
fn call(client: &Client, operation: Value) -> Result<Value> {
    Ok(client.call(0, operation, None)?.value)
}
fn handle(value: &Value, key: &str) -> Result<u64> {
    value
        .get(key)
        .and_then(Value::as_u64)
        .ok_or_else(|| invalid(format!("Tokenizer production {key} handle missing")))
}
fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut text = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        text.push(char::from(DIGITS[usize::from(byte >> 4)]));
        text.push(char::from(DIGITS[usize::from(byte & 15)]));
    }
    text
}
#[derive(Default)]
struct ByteCount(usize);
impl std::io::Write for ByteCount {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0 = self
            .0
            .checked_add(bytes.len())
            .ok_or_else(|| std::io::Error::other("Tokenizer JSON byte count overflow"))?;
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}
fn json_size(value: &Value) -> Result<usize> {
    let mut count = ByteCount::default();
    serde_json::to_writer(&mut count, value)?;
    Ok(count.0)
}
fn open(engine: &Engine, name: &str, path: &Path, index: usize) -> Result<Option<State>> {
    let Some(session) = ModelSession::open(engine, &format!("tokenizer-{name}-lease-{index}"))?
    else {
        return Ok(None);
    };
    let client = session.client();
    call(&client, json!({"Op":"backend-init"}))?;
    let model = handle(
        &call(
            &client,
            json!({"Op":"model-load","Path":path,"Base":"default","Params":null,"Overrides":{"VocabOnly":true},"Progress":false}),
        )?,
        "Model",
    )?;
    let context = handle(
        &call(
            &client,
            json!({"Op":"context","Model":model,"Params":null,"Overrides":{"Context":4096}}),
        )?,
        "Context",
    )?;
    Ok(Some(State {
        session,
        client,
        model,
        context,
    }))
}
impl<'a> VocabularySession<'a> {
    pub(super) fn open(engine: &'a Engine, name: &str, path: &Path) -> Result<Option<Self>> {
        let Some(state) = open(engine, name, path, 0)? else {
            return Ok(None);
        };
        Ok(Some(Self {
            engine,
            name: name.to_owned(),
            path: path.to_owned(),
            state: Some(state),
            rows: 0,
            wire: 0,
            started: Instant::now(),
            leases: Vec::new(),
        }))
    }
    fn close(&mut self, reason: &str) -> Result<()> {
        let state = self
            .state
            .take()
            .ok_or_else(|| invalid("Tokenizer vocabulary lease is not open"))?;
        call(
            &state.client,
            json!({"Op":"release","Kind":"context","Handle":state.context}),
        )?;
        call(
            &state.client,
            json!({"Op":"release","Kind":"model","Handle":state.model}),
        )?;
        call(&state.client, json!({"Op":"backend-free"}))?;
        state.session.finish()?;
        self.leases.push(json!({"Index":self.leases.len(),"PrefetchedRows":self.rows,"EstimatedCombinedWireBytes":self.wire,"LifetimeMilliseconds":self.started.elapsed().as_millis().to_string(),"ClosedFor":reason,"Completed":true}));
        Ok(())
    }
    fn rotate(&mut self, reason: &str) -> Result<()> {
        self.close(reason)?;
        self.state = open(self.engine, &self.name, &self.path, self.leases.len())?;
        if self.state.is_none() {
            return Err(invalid(
                "Production vocabulary API disappeared during stateless rotation",
            ));
        }
        self.rows = 0;
        self.wire = 0;
        self.started = Instant::now();
        Ok(())
    }
    fn exchange(&mut self, operation: Value) -> Result<Value> {
        let size = json_size(&operation)?;
        let state = self
            .state
            .as_ref()
            .ok_or_else(|| invalid("Tokenizer vocabulary lease missing"))?;
        let response = call(&state.client, operation)?;
        self.wire = self
            .wire
            .checked_add(size)
            .and_then(|n| n.checked_add(json_size(&response).ok()?))
            .and_then(|n| n.checked_add(256))
            .ok_or_else(|| invalid("Tokenizer wire accounting overflow"))?;
        Ok(response)
    }
    pub(super) fn observe(
        &mut self,
        inputs: &[String],
        reference_tokens: &[&[u32]],
    ) -> Result<Observation> {
        if inputs.is_empty()
            || inputs.len() > 256
            || inputs.len() != reference_tokens.len()
            || inputs.iter().map(String::len).sum::<usize>() > 1024 * 1024
            || reference_tokens
                .iter()
                .map(|tokens| tokens.len() * 4)
                .sum::<usize>()
                > 1024 * 1024
        {
            return Err(invalid(
                "Tokenizer observation batch outside protocol bounds",
            ));
        }
        if self.rows + inputs.len() > ROW_LIMIT {
            self.rotate("prefetched-row-bound")?;
        } else if self.wire >= WIRE_LIMIT {
            self.rotate("combined-wire-bound")?;
        } else if self.started.elapsed().as_secs() >= 120 {
            self.rotate("lifetime-bound")?;
        }
        let context = self
            .state
            .as_ref()
            .ok_or_else(|| invalid("Tokenizer vocabulary lease missing"))?
            .context;
        let inputs = inputs
            .iter()
            .map(|input| hex(input.as_bytes()))
            .collect::<Vec<_>>();
        let start = Instant::now();
        let tokens=self.exchange(json!({"Op":"vocab-tokenize","Context":context,"InputsHex":inputs,"AddSpecial":true,"ParseSpecial":true}))?;
        let encode_ns = start.elapsed().as_nanos();
        let start = Instant::now();
        let texts=self.exchange(json!({"Op":"raw-vocab-detokenize","Context":context,"TokenLists":reference_tokens,"RemoveSpecial":false,"UnparseSpecial":true}))?;
        let decode_ns = start.elapsed().as_nanos();
        self.rows += inputs.len();
        Ok(Observation {
            tokens,
            texts,
            encode_ns,
            decode_ns,
        })
    }
    pub(super) fn finish(mut self) -> Result<Value> {
        self.close("workload-complete")?;
        Ok(
            json!({"Sessions":self.leases,"MaximumPrefetchedRowsPerSession":ROW_LIMIT,"RotationCombinedWireThreshold":WIRE_LIMIT,"RotationLifetimeSeconds":120,"SourceCorrection":"Vocabulary tokenization/detokenization is stateless; the identical immutable input is reloaded across bounded process leases while source counters, generator order and compared prefix remain unchanged"}),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn focused_tokenizer_wire_accounting_guard() -> Result<()> {
        for value in [
            json!(null),
            json!({"Text":"\0\n\r\t\"é𝕒","Numbers":[0, 17, 4_294_967_295_u64]}),
            json!({"Values":[[1,2],[],[3]],"Errors":[null,null,null]}),
        ] {
            assert_eq!(json_size(&value)?, serde_json::to_vec(&value)?.len());
        }
        assert_eq!(hex(&[0, 1, 15, 16, 127, 128, 255]), "00010f107f80ff");
        Ok(())
    }
}
