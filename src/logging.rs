//! Synchronized bounded log records with explicit flush and close.

use std::io::Write;
use std::sync::{Mutex, MutexGuard};
use std::time::SystemTime;

use crate::{Result, invalid};

/// Severity attached to a production log record.
#[derive(Clone, Copy, Debug)]
pub enum Level {
    /// Routine progress.
    Info,
    /// Recoverable issue.
    Warning,
    /// Failed operation.
    Error,
    /// Diagnostic detail.
    Debug,
}

struct State<W> {
    writer: Option<W>,
    timestamps: bool,
    prefix: bool,
}

/// A writer shared by concurrent runtime workers.
///
/// Record writes and format changes serialize on one mutex; close flushes the
/// writer and rejects later operations. The sink owns any persistent storage.
pub struct Logger<W: Write + Send> {
    state: Mutex<State<W>>,
}

impl<W: Write + Send> Logger<W> {
    /// Owns a writer with timestamps and severity prefixes initially disabled.
    pub const fn new(writer: W) -> Self {
        Self {
            state: Mutex::new(State {
                writer: Some(writer),
                timestamps: false,
                prefix: false,
            }),
        }
    }

    fn state(&self) -> Result<MutexGuard<'_, State<W>>> {
        let state = self
            .state
            .lock()
            .map_err(|_| invalid("logger worker poisoned the writer"))?;
        if state.writer.is_none() {
            return Err(invalid("logger is closed"));
        }
        Ok(state)
    }

    /// Changes timestamp emission atomically with respect to record writes.
    ///
    /// # Errors
    /// Rejects a closed or poisoned writer.
    pub fn set_timestamps(&self, enabled: bool) -> Result<()> {
        self.state()?.timestamps = enabled;
        Ok(())
    }

    /// Changes severity prefix emission atomically with respect to writes.
    ///
    /// # Errors
    /// Rejects a closed or poisoned writer.
    pub fn set_prefix(&self, enabled: bool) -> Result<()> {
        self.state()?.prefix = enabled;
        Ok(())
    }

    /// Writes a complete UTF-8 record while excluding other workers.
    ///
    /// # Errors
    /// Rejects NUL, records larger than 1024 bytes, closed writers and I/O errors.
    pub fn emit(&self, level: Level, text: &str) -> Result<()> {
        if text.len() > 1024 || text.contains('\0') {
            return Err(invalid("log record exceeds bounds"));
        }
        let mut state = self.state()?;
        let timestamps = state.timestamps;
        let prefix = state.prefix;
        let writer = state
            .writer
            .as_mut()
            .ok_or_else(|| invalid("logger is closed"))?;
        if timestamps {
            let timestamp = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH)?;
            write!(
                writer,
                "[{}.{:03}] ",
                timestamp.as_secs(),
                timestamp.subsec_millis()
            )?;
        }
        if prefix {
            write!(writer, "{level:?}: ")?;
        }
        writer.write_all(text.as_bytes())?;
        drop(state);
        Ok(())
    }

    /// Flushes all completed records to the owned writer.
    ///
    /// # Errors
    /// Rejects a closed/poisoned writer and propagates flush failures.
    pub fn flush(&self) -> Result<()> {
        self.state()?
            .writer
            .as_mut()
            .ok_or_else(|| invalid("logger is closed"))?
            .flush()?;
        Ok(())
    }

    /// Flushes and destroys the writer, making all subsequent calls fail.
    ///
    /// # Errors
    /// Rejects repeated close, poisoning and flush errors. On a flush error the
    /// writer remains owned so the caller can retry or inspect its sink.
    pub fn close(&self) -> Result<()> {
        let mut state = self.state()?;
        state
            .writer
            .as_mut()
            .ok_or_else(|| invalid("logger is closed"))?
            .flush()?;
        state.writer.take();
        drop(state);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn closing_is_visible_to_all_worker_calls() -> Result<()> {
        let log = Logger::new(Vec::new());
        std::thread::scope(|scope| {
            let worker = scope.spawn(|| log.emit(Level::Info, "worker\n"));
            worker.join().map_err(|_| invalid("log worker panicked"))?
        })?;
        log.flush()?;
        log.close()?;
        assert!(log.emit(Level::Info, "late").is_err());
        assert!(log.set_prefix(true).is_err());
        assert!(log.close().is_err());
        Ok(())
    }
}
