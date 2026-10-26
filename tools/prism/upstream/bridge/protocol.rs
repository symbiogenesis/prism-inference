//! Bounded little-endian request/response protocol, independent of native code.

use std::io::{Read, Write};

pub const MAX_FILE_BYTES: u64 = 128 * 1024 * 1024;
const MAX_REQUESTS: u32 = 64;
const MAX_PACKED_BYTES: u32 = 16 * 1024 * 1024;
const MAX_FLOAT_COUNT: u32 = 4 * 1024 * 1024;
const MAX_TOTAL_OUTPUT_COUNT: u32 = 16 * 1024 * 1024;

pub struct Request {
    pub operation: u32,
    pub k: u32,
    pub m: u32,
    pub n: u32,
    pub packed: Vec<u8>,
    pub activations: Vec<f32>,
}

impl Request {
    pub const fn output_count(&self) -> u32 {
        if self.operation == 1 {
            self.m * self.n
        } else {
            self.k * self.m
        }
    }
}

pub const fn require(condition: bool, message: &'static str) -> Result<(), &'static str> {
    if condition { Ok(()) } else { Err(message) }
}

fn read_u32(input: &mut impl Read) -> Result<u32, &'static str> {
    let mut bytes = [0; 4];
    input
        .read_exact(&mut bytes)
        .map_err(|_| "truncated request")?;
    Ok(u32::from_le_bytes(bytes))
}

pub fn read_requests(mut input: impl Read) -> Result<Vec<Request>, &'static str> {
    let mut magic = [0; 8];
    input
        .read_exact(&mut magic)
        .map_err(|_| "invalid request magic")?;
    require(&magic == b"BSPREQ1\n", "invalid request magic")?;
    let count = read_u32(&mut input)?;
    require(
        count > 0 && count <= MAX_REQUESTS,
        "request count outside 1..64",
    )?;
    let mut requests = Vec::with_capacity(count as usize);
    let mut total_output_count = 0;
    for _ in 0..count {
        let operation = read_u32(&mut input)?;
        let k = read_u32(&mut input)?;
        let m = read_u32(&mut input)?;
        let n = read_u32(&mut input)?;
        let packed_count = read_u32(&mut input)?;
        let activation_count = read_u32(&mut input)?;
        require(
            packed_count <= MAX_PACKED_BYTES && activation_count <= MAX_FLOAT_COUNT,
            "request payload exceeds size limit",
        )?;
        match operation {
            1 => {
                require(
                    k > 0 && k <= 8192 && k % 128 == 0 && m > 0 && m <= 4096 && n > 0 && n <= 256,
                    "invalid PQ2 matrix shape",
                )?;
                require(
                    u64::from(k) * u64::from(m) <= 8 * 1024 * 1024,
                    "PQ2 weights exceed element limit",
                )?;
                require(
                    u64::from(packed_count) == u64::from(k / 128) * u64::from(m) * 34
                        && u64::from(activation_count) == u64::from(k) * u64::from(n),
                    "PQ2 payload does not match shape",
                )?;
            }
            2 => {
                require(
                    k == 128 && m > 0 && m <= 20_000 && n == 0,
                    "invalid PTQ1 batch shape",
                )?;
                require(
                    packed_count == m * 28 && activation_count == 0,
                    "PTQ1 payload does not match shape",
                )?;
            }
            _ => return Err("unknown operation"),
        }
        let mut item = Request {
            operation,
            k,
            m,
            n,
            packed: Vec::new(),
            activations: Vec::new(),
        };
        require(
            item.output_count() <= MAX_FLOAT_COUNT,
            "response exceeds element limit",
        )?;
        total_output_count += item.output_count();
        require(
            total_output_count <= MAX_TOTAL_OUTPUT_COUNT,
            "responses exceed total element limit",
        )?;
        item.packed.resize(packed_count as usize, 0);
        item.activations.reserve(activation_count as usize);
        input
            .read_exact(&mut item.packed)
            .map_err(|_| "truncated packed payload")?;
        for _ in 0..activation_count {
            item.activations.push(f32::from_bits(read_u32(&mut input)?));
        }
        requests.push(item);
    }
    let mut trailing = [0];
    require(
        matches!(input.read(&mut trailing), Ok(0)),
        "trailing request data",
    )?;
    Ok(requests)
}

pub fn write_responses(
    mut output: impl Write,
    requests: &[Request],
    mut execute: impl FnMut(&Request) -> Result<Vec<f32>, &'static str>,
) -> Result<(), Box<dyn std::error::Error>> {
    output.write_all(b"BSPRES1\n")?;
    output.write_all(&u32::try_from(requests.len())?.to_le_bytes())?;
    for item in requests {
        let result = execute(item)?;
        output.write_all(&u32::try_from(result.len())?.to_le_bytes())?;
        for value in result {
            output.write_all(&value.to_bits().to_le_bytes())?;
        }
    }
    Ok(())
}
