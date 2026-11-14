//! Fixed request metadata for diagnosing prefill and generation failures.

use bonsai_prism_runtime::runtime::{GenerationPhase, GenerationTimings, PhaseTiming};
use std::{
    io::{self, Write},
    time::Duration,
};

pub(super) fn write_generation(
    writer: &mut impl Write,
    succeeded: bool,
    elapsed: Duration,
    prompt_tokens: usize,
    first_text: Option<Duration>,
    timings: Option<&GenerationTimings>,
) -> io::Result<()> {
    let result = if succeeded { "ok" } else { "failed" };
    write!(
        writer,
        "private-inference-generation result={result} elapsedMicros={} promptTokens={prompt_tokens} firstTextMicros=",
        elapsed.as_micros()
    )?;
    write_optional(writer, first_text.map(|duration| duration.as_micros()))?;
    if let Some(timings) = timings {
        write!(
            writer,
            " runtime=measured entered={} runtimeSucceeded={} runtimeMicros={} reusedTokens=",
            timings.entered,
            timings.succeeded,
            timings.total.as_micros()
        )?;
        write_optional(writer, timings.reused_tokens)?;
        write!(writer, " committedGeneratedTokens=")?;
        write_optional(writer, timings.committed_generated_tokens)?;
        for phase in GenerationPhase::ALL {
            let timing = timings.phase(phase);
            if timing.attempted_calls != 0 {
                write_phase(writer, phase, timing)?;
            }
        }
    } else {
        write!(writer, " runtime=unavailable-cpu-speculation")?;
    }
    writeln!(writer)
}

fn write_optional(
    writer: &mut impl Write,
    value: Option<impl std::fmt::Display>,
) -> io::Result<()> {
    if let Some(value) = value {
        write!(writer, "{value}")
    } else {
        write!(writer, "unknown")
    }
}

fn write_phase(
    writer: &mut impl Write,
    phase: GenerationPhase,
    timing: &PhaseTiming,
) -> io::Result<()> {
    write!(
        writer,
        " {}Micros={} {}Calls={}/{} {}Tokens={}/{} {}PartialUnknown={}",
        phase.name(),
        timing.elapsed.as_micros(),
        phase.name(),
        timing.attempted_calls,
        timing.completed_calls,
        phase.name(),
        timing.attempted_tokens,
        timing.completed_tokens_lower_bound,
        phase.name(),
        timing.partial_progress_unknown
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generation_failure_preserves_unknown_counts_and_failed_phase_progress() -> io::Result<()> {
        let mut bytes = Vec::new();
        write_generation(
            &mut bytes,
            false,
            Duration::from_millis(600),
            2048,
            None,
            Some(&GenerationTimings::default()),
        )?;
        let summary = String::from_utf8(bytes).map_err(io::Error::other)?;
        assert_eq!(
            summary,
            "private-inference-generation result=failed elapsedMicros=600000 promptTokens=2048 firstTextMicros=unknown runtime=measured entered=false runtimeSucceeded=false runtimeMicros=0 reusedTokens=unknown committedGeneratedTokens=unknown\n"
        );
        let mut bytes = Vec::new();
        let timing = PhaseTiming {
            elapsed: Duration::from_millis(599),
            attempted_calls: 2,
            completed_calls: 1,
            attempted_tokens: 128,
            completed_tokens_lower_bound: 64,
            partial_progress_unknown: true,
        };
        write_phase(&mut bytes, GenerationPhase::PrefillIngestion, &timing)?;
        assert_eq!(
            String::from_utf8(bytes).map_err(io::Error::other)?,
            " prefill-ingestionMicros=599000 prefill-ingestionCalls=2/1 prefill-ingestionTokens=128/64 prefill-ingestionPartialUnknown=true"
        );
        Ok(())
    }

    #[test]
    fn speculation_reports_unavailable_attribution_without_inventing_runtime_counts()
    -> io::Result<()> {
        let mut bytes = Vec::new();
        write_generation(
            &mut bytes,
            true,
            Duration::from_millis(5),
            100,
            Some(Duration::from_millis(2)),
            None,
        )?;
        assert_eq!(
            String::from_utf8(bytes).map_err(io::Error::other)?,
            "private-inference-generation result=ok elapsedMicros=5000 promptTokens=100 firstTextMicros=2000 runtime=unavailable-cpu-speculation\n"
        );
        Ok(())
    }
}
