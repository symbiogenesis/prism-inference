// SPDX-License-Identifier: MIT
//! Keep small, successful case transcripts in memory instead of writing and deleting them.

use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};

const MEMORY_BYTES: usize = 256 * 1024;

pub(in super::super) struct Transcript {
    path: PathBuf,
    memory: Vec<u8>,
    spill: Option<BufWriter<File>>,
    discard_on_success: bool,
    finished: bool,
}

impl Transcript {
    pub(in super::super) fn new(path: &Path) -> Self {
        Self {
            path: path.to_owned(),
            memory: Vec::new(),
            spill: None,
            discard_on_success: false,
            finished: false,
        }
    }

    pub(in super::super) const fn discard_on_success(&mut self) {
        self.discard_on_success = true;
    }

    pub(in super::super) fn finish(&mut self, success: bool) -> io::Result<()> {
        if self.finished {
            return Ok(());
        }
        if let Some(output) = &mut self.spill {
            output.flush()?;
        }
        let spilled = self.spill.take().is_some();
        if success && self.discard_on_success {
            if spilled {
                fs::remove_file(&self.path)?;
            }
        } else if !spilled {
            let mut output = File::create(&self.path)?;
            output.write_all(&self.memory)?;
        }
        self.memory.clear();
        self.finished = true;
        Ok(())
    }
}

impl Write for Transcript {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if self.finished {
            return Err(io::Error::other("backend transcript is already finalized"));
        }
        if self.spill.is_none() && bytes.len() <= MEMORY_BYTES - self.memory.len() {
            self.memory.extend_from_slice(bytes);
        } else {
            if self.spill.is_none() {
                let mut output = BufWriter::new(File::create(&self.path)?);
                output.write_all(&self.memory)?;
                self.memory.clear();
                self.spill = Some(output);
            }
            if let Some(output) = &mut self.spill {
                output.write_all(bytes)?;
            }
        }
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        if let Some(output) = &mut self.spill {
            output.flush()?;
        }
        Ok(())
    }
}

impl Drop for Transcript {
    fn drop(&mut self) {
        // Explicit completion reports I/O errors; unwinding still attempts to
        // retain the transcript rather than silently dropping buffered bytes.
        let _ = self.finish(false);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transcripts_skip_successful_files_and_preserve_failure_and_spill_bytes() -> io::Result<()> {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(io::Error::other)?
            .as_nanos();
        let directory =
            std::env::temp_dir().join(format!("prism-transcripts-{}-{stamp}", std::process::id()));
        fs::create_dir(&directory)?;
        for (name, length, discard, success, retained) in [
            ("small-pass", 128, true, true, false),
            ("small-fail", 128, true, false, true),
            ("small-default", 128, false, true, true),
            ("spill-pass", MEMORY_BYTES + 1, true, true, false),
            ("spill-fail", MEMORY_BYTES + 1, true, false, true),
        ] {
            let path = directory.join(name);
            let bytes = vec![b'x'; length];
            let mut transcript = Transcript::new(&path);
            transcript.write_all(&bytes[..128])?;
            transcript.write_all(&bytes[128..])?;
            transcript.flush()?;
            assert_eq!(path.exists(), length > MEMORY_BYTES);
            if discard {
                transcript.discard_on_success();
            }
            transcript.finish(success)?;
            assert_eq!(path.exists(), retained);
            if retained {
                assert_eq!(fs::read(&path)?, bytes);
                fs::remove_file(path)?;
            }
        }
        let path = directory.join("unwinding");
        {
            let mut transcript = Transcript::new(&path);
            transcript.write_all(b"failed case")?;
            transcript.discard_on_success();
        }
        assert_eq!(fs::read(&path)?, b"failed case");
        fs::remove_file(path)?;
        fs::remove_dir(directory)
    }
}
