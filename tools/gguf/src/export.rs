//! Bounded, exact PQ2 symbol/scale export to the pinned native PTQ1 layout.

use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::time::{Duration, Instant};

use bonsai_ternary::{pack_ptq, unpack_ptq};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::inspect::Inspection;
use crate::{CODEC_COMMIT, Result, bad};

const BATCH_BLOCKS: usize = 1927;
const COPY_BYTES: usize = 65_536;
const PAD_BYTES: usize = 8192;
const PQ2_BYTES: usize = 34;
const PTQ1_BYTES: usize = 28;
const TIMEOUT_MESSAGE: &str = "PTQ1 export exceeded its deadline.";

struct Deadline {
    started: Instant,
    timeout: Duration,
}

impl Deadline {
    fn check(&self) -> Result<()> {
        if self.started.elapsed() > self.timeout {
            return Err(bad(TIMEOUT_MESSAGE));
        }
        Ok(())
    }
}

fn overflow() -> crate::Error {
    bad("export arithmetic overflow.")
}

fn add(left: u64, right: u64) -> Result<u64> {
    left.checked_add(right).ok_or_else(overflow)
}

fn align(value: u64, alignment: u64) -> Result<u64> {
    let rounded = add(value, alignment - 1)? / alignment;
    rounded.checked_mul(alignment).ok_or_else(overflow)
}

fn report_integer(report: &Value, key: &str) -> Result<u64> {
    report[key]
        .as_u64()
        .ok_or_else(|| bad(format!("export requires a validated {key}.")))
}

fn read_exact(file: &mut File, bytes: &mut [u8]) -> Result<()> {
    file.read_exact(bytes).map_err(|error| {
        if error.kind() == std::io::ErrorKind::UnexpectedEof {
            bad("truncated export source.")
        } else {
            error.into()
        }
    })
}

fn copy_range(
    source: &mut File,
    output: &mut File,
    start: u64,
    mut bytes: u64,
    buffer: &mut [u8],
    deadline: &Deadline,
) -> Result<()> {
    source.seek(SeekFrom::Start(start))?;
    while bytes > 0 {
        deadline.check()?;
        let count = usize::try_from(bytes.min(u64::try_from(buffer.len())?))?;
        read_exact(source, &mut buffer[..count])?;
        output.write_all(&buffer[..count])?;
        bytes -= u64::try_from(count)?;
    }
    deadline.check()
}

fn pad_to(output: &mut File, position: u64, zeros: &[u8], deadline: &Deadline) -> Result<()> {
    let mut current = output.stream_position()?;
    if current > position {
        return Err(bad("export tensor extents overlap."));
    }
    while current < position {
        deadline.check()?;
        let count = usize::try_from((position - current).min(u64::try_from(zeros.len())?))?;
        output.write_all(&zeros[..count])?;
        current = add(current, u64::try_from(count)?)?;
    }
    deadline.check()
}

fn output_digest(
    output: &mut File,
    mut bytes: u64,
    buffer: &mut [u8],
    deadline: &Deadline,
) -> Result<String> {
    output.seek(SeekFrom::Start(0))?;
    let mut hash = Sha256::new();
    while bytes > 0 {
        deadline.check()?;
        let count = usize::try_from(bytes.min(u64::try_from(buffer.len())?))?;
        output.read_exact(&mut buffer[..count])?;
        hash.update(&buffer[..count]);
        bytes -= u64::try_from(count)?;
    }
    deadline.check()?;
    Ok(crate::digest_hex(hash))
}

fn new_output(path: &Path) -> Result<File> {
    let mut options = OpenOptions::new();
    options.create_new(true).read(true).write(true);
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        options.share_mode(0);
    }
    Ok(options.open(path)?)
}

/// Writes a new experimental export, retaining any partial file on failure.
/// The caller verifies the independently pinned source and codec identities.
#[allow(
    clippy::too_many_lines,
    reason = "Keep the validated streamed serialization and its receipt in one operation."
)]
#[allow(
    clippy::redundant_pub_crate,
    reason = "The export helper is an internal operation behind the validated public API"
)]
pub(super) fn export(
    file: &mut File,
    output_path: &Path,
    inspection: &Inspection,
    timeout: Duration,
) -> Result<Value> {
    if timeout < Duration::from_secs(1) || timeout > Duration::from_secs(3600) {
        return Err(bad("export timeout must be between 1 and 3600 seconds."));
    }
    let file_type_position = inspection.file_type_position.ok_or_else(|| {
        bad("PTQ1 export requires a UINT32 Prism PQ2 general.file_type (141 or legacy 142).")
    })?;
    let file_type = inspection.report["Scalars"]["general.file_type"].as_u64();
    if !matches!(file_type, Some(141 | 142)) || file_type_position < 4 {
        return Err(bad(
            "PTQ1 export requires a UINT32 Prism PQ2 general.file_type (141 or legacy 142).",
        ));
    }
    // JSON numbers do not retain a metadata value's declared GGUF integer type.
    // Check the raw tag so an INT64/UINT64 value cannot be relabeled as UINT32.
    file.seek(SeekFrom::Start(file_type_position - 4))?;
    let mut raw_file_type = [0; 8];
    read_exact(file, &mut raw_file_type)?;
    if u32::from_le_bytes(raw_file_type[..4].try_into()?) != 4
        || Some(u64::from(u32::from_le_bytes(
            raw_file_type[4..].try_into()?,
        ))) != file_type
    {
        return Err(bad(
            "PTQ1 export requires a UINT32 Prism PQ2 general.file_type (141 or legacy 142).",
        ));
    }
    if !inspection
        .tensors
        .iter()
        .any(|tensor| tensor.type_id == 142)
    {
        return Err(bad("PTQ1 export requires at least one PQ2 tensor."));
    }
    let alignment = report_integer(&inspection.report, "Alignment")?;
    if alignment == 0 || !alignment.is_power_of_two() || alignment > 1_048_576 {
        return Err(bad("invalid/over-limit export alignment."));
    }
    let data_offset = report_integer(&inspection.report, "TensorDataOffset")?;
    let source_bytes = file.metadata()?.len();
    if data_offset > source_bytes || add(file_type_position, 4)? > data_offset {
        return Err(bad("export metadata exceeds the source header."));
    }
    let mut ordered = inspection.tensors.iter().collect::<Vec<_>>();
    ordered.sort_unstable_by_key(|tensor| tensor.offset);
    let mut offsets = Vec::with_capacity(ordered.len());
    let mut extent = 0;
    let mut blocks = 0;
    let mut converted_tensors = 0_u64;
    let mut source_extent = 0;
    for tensor in &ordered {
        if add(tensor.type_position, 12)? > data_offset {
            return Err(bad("export tensor descriptor exceeds the source header."));
        }
        let source_end = add(tensor.offset, tensor.bytes)?;
        if tensor.offset < source_extent || add(data_offset, source_end)? > source_bytes {
            return Err(bad(
                "export source tensor extents are truncated or overlapping.",
            ));
        }
        source_extent = source_end;
        let offset = align(extent, alignment)?;
        offsets.push(offset);
        let bytes = if tensor.type_id == 142 {
            let tensor_blocks = tensor.elements / 128;
            if !tensor.elements.is_multiple_of(128)
                || tensor_blocks.checked_mul(34) != Some(tensor.bytes)
            {
                return Err(bad("export requires complete 34-byte PQ2 blocks."));
            }
            blocks = add(blocks, tensor_blocks)?;
            converted_tensors = add(converted_tensors, 1)?;
            tensor_blocks.checked_mul(28).ok_or_else(overflow)?
        } else {
            tensor.bytes
        };
        extent = add(offset, bytes)?;
    }
    let output_bytes = add(data_offset, align(extent, alignment)?)?;
    if output_bytes > u64::try_from(i64::MAX)? {
        return Err(bad("export exceeds supported file size."));
    }

    let deadline = Deadline {
        started: Instant::now(),
        timeout,
    };
    let mut input = vec![0; BATCH_BLOCKS * PQ2_BYTES];
    let mut packed = vec![0; BATCH_BLOCKS * PTQ1_BYTES];
    let mut copy_buffer = vec![0; COPY_BYTES];
    let zeros = [0; PAD_BYTES];
    let mut output = new_output(output_path)?;
    copy_range(
        file,
        &mut output,
        0,
        data_offset,
        &mut copy_buffer,
        &deadline,
    )?;
    output.seek(SeekFrom::Start(file_type_position))?;
    output.write_all(&143_u32.to_le_bytes())?;
    for (tensor, offset) in ordered.iter().zip(&offsets) {
        deadline.check()?;
        output.seek(SeekFrom::Start(tensor.type_position))?;
        let type_id = if tensor.type_id == 142 {
            143
        } else {
            tensor.type_id
        };
        output.write_all(&type_id.to_le_bytes())?;
        output.write_all(&offset.to_le_bytes())?;
    }
    output.seek(SeekFrom::Start(data_offset))?;
    for (tensor, offset) in ordered.iter().zip(offsets) {
        deadline.check()?;
        pad_to(&mut output, add(data_offset, offset)?, &zeros, &deadline)?;
        let source_start = add(data_offset, tensor.offset)?;
        if tensor.type_id != 142 {
            copy_range(
                file,
                &mut output,
                source_start,
                tensor.bytes,
                &mut copy_buffer,
                &deadline,
            )?;
            continue;
        }
        file.seek(SeekFrom::Start(source_start))?;
        let mut remaining = tensor.elements / 128;
        while remaining > 0 {
            deadline.check()?;
            let count = usize::try_from(remaining.min(u64::try_from(BATCH_BLOCKS)?))?;
            let source = &mut input[..count * PQ2_BYTES];
            let destination = &mut packed[..count * PTQ1_BYTES];
            read_exact(file, source)?;
            let mut exact = true;
            for (source_block, destination_block) in source
                .as_chunks::<PQ2_BYTES>()
                .0
                .iter()
                .zip(destination.as_chunks_mut::<PTQ1_BYTES>().0)
            {
                deadline.check()?;
                let encoded = pack_ptq(source_block)?;
                destination_block.copy_from_slice(&encoded);
                exact &= unpack_ptq(&encoded)? == *source_block;
            }
            if !exact {
                return Err(bad("PTQ1 export failed exact PQ2 reconstruction."));
            }
            deadline.check()?;
            output.write_all(destination)?;
            remaining -= u64::try_from(count)?;
        }
    }
    pad_to(&mut output, output_bytes, &zeros, &deadline)?;
    output.sync_all()?;
    deadline.check()?;
    let sha = output_digest(&mut output, output_bytes, &mut copy_buffer, &deadline)?;
    Ok(json!({
        "SchemaVersion": 1,
        "CodecCommit": CODEC_COMMIT,
        "SourceBytes": source_bytes,
        "OutputBytes": output_bytes,
        "OutputSha256": sha,
        "ConvertedTensorCount": converted_tensors,
        "ConvertedBlocks": blocks,
        "ExactPq2ReconstructionVerified": true,
        "NonPq2TensorBytesCopiedVerbatim": true,
        "MetadataChanges": "general.file_type UINT32 -> 143; PQ2 tensor type/offset fields; alignment padding normalized",
        "WorkingBufferBytes": input.len() + packed.len() + copy_buffer.len() + zeros.len(),
        "Qualification": "weight-symbol-and-scale parity only; PTQ1 native activation quantization, logits, quality and performance unqualified"
    }))
}
