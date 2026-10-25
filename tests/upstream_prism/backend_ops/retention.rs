// SPDX-License-Identifier: MIT
//! Bounded failed-case raw evidence; every discarded file retains its byte count and digest.

use super::super::{Result, invalid};
use super::host::encode_hex;
use serde_json::{Value as Json, json};
use sha2::{Digest, Sha256};
use std::fs::{self, File};
use std::io::Read;
use std::path::Path;

const MAX_RETAINED_BYTES: u64 = 256 * 1024 * 1024;
const MAX_RETAINED_FILE: u64 = 64 * 1024 * 1024;
const MAX_RETAINED_CASES: usize = 2;

#[derive(Default)]
pub(super) struct Budget {
    retained_bytes: u64,
    retained_cases: usize,
}

fn digest(path: &Path, bytes: u64) -> Result<String> {
    let mut input = File::open(path)?.take(
        bytes
            .checked_add(1)
            .ok_or_else(|| invalid("artifact size overflow"))?,
    );
    let mut hash = Sha256::new();
    let mut count = 0_u64;
    let mut buffer = vec![0_u8; 65_536];
    loop {
        let read = input.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        count += u64::try_from(read)?;
        hash.update(&buffer[..read]);
    }
    if count != bytes {
        return Err(invalid("backend artifact extent changed while hashing"));
    }
    Ok(encode_hex(&hash.finalize()))
}

impl Budget {
    pub(super) fn failure(&mut self, directory: &Path) -> Result<Json> {
        let owned = fs::canonicalize(directory)?;
        if !owned.is_absolute() {
            return Err(invalid("failure retention directory must be absolute"));
        }
        let mut entries = fs::read_dir(&owned)?.collect::<std::io::Result<Vec<_>>>()?;
        entries.sort_by_key(fs::DirEntry::file_name);
        let retain_case = self.retained_cases < MAX_RETAINED_CASES;
        let mut retained = 0_u64;
        let mut deleted = 0_u64;
        let mut files = Vec::new();
        for entry in entries {
            let name = entry.file_name();
            if name == "case-receipt.json"
                || name == "input-state.json"
                || name == "retention-receipt.json"
            {
                continue;
            }
            if !entry.file_type()?.is_file() {
                return Err(invalid("failure retention encountered a non-file artifact"));
            }
            let path = fs::canonicalize(entry.path())?;
            if path.parent() != Some(owned.as_path()) {
                return Err(invalid("failure retention encountered an unowned artifact"));
            }
            let bytes = entry.metadata()?.len();
            let sha256 = digest(&path, bytes)?;
            let keep = retain_case
                && bytes <= MAX_RETAINED_FILE
                && bytes <= MAX_RETAINED_BYTES - self.retained_bytes;
            if keep {
                self.retained_bytes += bytes;
                retained += bytes;
            } else {
                fs::remove_file(&path)?;
                deleted = deleted
                    .checked_add(bytes)
                    .ok_or_else(|| invalid("pruned artifact extent overflow"))?;
            }
            files.push(json!({"File":name.to_str().ok_or_else(||invalid("backend artifact name is not UTF-8"))?,"Bytes":bytes,"Sha256":sha256,"Retained":keep}));
        }
        if retain_case {
            self.retained_cases += 1;
        }
        let receipt = json!({"SchemaVersion":1,"Policy":"first two failed/error cases, at most64MiB per file and256MiB total per partition; other raw bytes are discarded after hashing","MaxRetainedBytes":MAX_RETAINED_BYTES,"MaxRetainedFileBytes":MAX_RETAINED_FILE,"MaxRetainedCases":MAX_RETAINED_CASES,"RetainedBytes":retained,"PrunedBytes":deleted,"PartitionRetainedBytes":self.retained_bytes,"Files":files});
        let bytes = serde_json::to_vec_pretty(&receipt)?;
        fs::write(owned.join("retention-receipt.json"), &bytes)?;
        Ok(
            json!({"Receipt":"retention-receipt.json","ReceiptSha256":encode_hex(&Sha256::digest(&bytes)),"RetainedBytes":retained,"PrunedBytes":deleted,"PartitionRetainedBytes":self.retained_bytes}),
        )
    }
}
