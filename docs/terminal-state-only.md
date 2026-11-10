# Reset-only terminal state ingestion

The sampled `SwiftSession` service path now uses state-only ingestion for the
last emitted token when it reaches its declared output cap and
`PromptReuse::Disabled` is selected. That request will not sample again, and the
next request resets model state. Computing the final token's next-token logits
therefore has no consumer. The existing native `Decoder::ingest_block` path
retains complete KV, recurrent and convolution state while omitting output-only
work. Qwen35 can also use its existing eligible-shape final-layer state-only
narrowing; this change introduces no additional graph arithmetic.
As with state-only prompt ingestion, omitted output-only projections are not
executed or numerically validated; every computed state value remains checked.

Only this final non-stop token uses state-only ingestion. Earlier generated
tokens still compute logits for the next sample. A sampled stop token consumes
the original sampler draw, enters neither state nor output and performs no
terminal ingestion. Callback backpressure, cancellation and deadline checks
surround the state-only operation under the same complete-request decoder and
sampler rollback. A state-only failure clears logits and restores the earlier
committed snapshot; failed restoration still closes admission until reset.

The service-only runtime entry point is crate-private and requires zero retained
prefix checkpoints. Successful terminal ingestion clears logits. If an internal
caller nevertheless continues that Runtime without resetting, prefix preparation
recomputes from the prompt instead of sampling stale penultimate logits. Ordinary
public Runtime APIs retain complete terminal logits and exact-prefix reuse.
`PromptReuse::Latest` keeps the original sampled path, including valid terminal
logits. Changing the session reuse policy still resets state. The explicit greedy
suffix-speculation service path also retains its existing logits behavior.

The timing inventory appends `terminal-ingestion` after the existing eleven
phases, preserving their labels, order and indexes. Its token count describes
state-only work; `decode-evaluation` counts only generated-token operations that
actually compute logits. Conservative forward-work admission still reserves the
complete evaluation allowance, and context, memory and device-lifetime limits
are unchanged. The Swift vocabulary has 248,320 entries: omitting this one output
projection and its 993,280-byte result is a source work reduction once per
output-capped request, not a measured latency or memory benefit.

Four model-free controls in `runtime::terminal_tests` cover output limits one,
four and eight, timed and untimed execution, exact token/state/RNG parity,
next-request recomputation, EOS, callback failure, post-mutation ingestion
failure, cancellation, expired admission and retained-checkpoint rejection.
The native graph's existing state-only ingestion controls remain required for
full-attention/recurrent state and continuation parity. Owning hosted checks,
actual-model state/adapter and cancellation acceptance, effective accelerator
work/resources and matched complete-request benefit remain pending before release
or Runner repinning.
