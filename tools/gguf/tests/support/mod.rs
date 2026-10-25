//! Dependency-free, exclusively owned scratch directories for synthetic tests.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

static NEXT_DIRECTORY: AtomicU64 = AtomicU64::new(0);

#[expect(
    clippy::redundant_pub_crate,
    reason = "The helper is visible only to its owning integration test, not publicly exported."
)]
pub(super) struct ScratchDirectory {
    path: PathBuf,
    closed: bool,
}

impl ScratchDirectory {
    pub(super) fn new() -> io::Result<Self> {
        let parent = std::env::temp_dir();
        if !parent.is_absolute() {
            return Err(io::Error::other("test scratch parent must be absolute"));
        }
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(io::Error::other)?
            .as_nanos();
        let path = parent.join(format!(
            "SwiftBonsai-GgufTest-{}-{nonce}-{}",
            std::process::id(),
            NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed)
        ));
        // create_dir grants ownership only if this exact directory did not
        // already exist. A collision fails rather than adopting another lane.
        fs::create_dir(&path)?;
        Ok(Self {
            path,
            closed: false,
        })
    }

    pub(super) fn path(&self) -> &Path {
        &self.path
    }

    pub(super) fn close(mut self) -> io::Result<()> {
        fs::remove_dir_all(&self.path)?;
        self.closed = true;
        Ok(())
    }
}

impl Drop for ScratchDirectory {
    fn drop(&mut self) {
        if !self.closed {
            let _ = fs::remove_dir_all(&self.path);
        }
    }
}
