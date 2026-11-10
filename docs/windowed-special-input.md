# Bounded special-token request preparation

`SpecialInput::split`, and therefore the composed `Tokenizer::encode`, uses
fixed reverse-trie windows instead of retaining one machine-sized match index
for every input byte. A window owns at most 4096 positions in an 8 KiB array of
16-bit special indices. The 4096-entry vocabulary bound leaves a distinct
no-match sentinel. Each window additionally scans at most the longest spelling
minus one bytes, capped at 255, so matches crossing its boundary retain the
original earliest-start and longest-at-start behavior.

When a selected spelling crosses the window boundary, the next window starts
after that complete spelling. Overlapping later matches stay unreachable.
Windows may end inside UTF-8 scalars: scanning operates on bytes, and only
complete validated UTF-8 special spellings create text-slice boundaries.
Failure-link selection, declared token IDs and borrowed text remain unchanged.

The result vector now grows fallibly as segments are emitted, with geometric
capacity capped by the existing segment limit. A long ordinary prompt with a
special-enabled vocabulary retains one borrowed text segment without reserving
one segment per input byte. The 1 MiB input, 256-byte spelling, 4096-special and
65,536-segment limits remain unchanged. Recognition still finishes before BPE,
so segmentation errors keep their original precedence over token-output limits.
No cache persists across requests, and literal input keeps its direct path.

At the maximum admitted input length, the old match array contained 8 MiB of
logical indices on a 64-bit target; the replacement scratch is 8 KiB. This is a
source allocation bound, not a measured process-memory or throughput result.
The bounded overlap adds scan work, and native stack/code-generation review and
matched request-time acceptance remain required before release or repinning.
Existing admission limits receive no discount.

Independent controls compare every offset of a maximum-length boundary-spanning
spelling with a naive UTF-8 matcher, cover overlapping/failure-link matches and
special indices through 4095, and check maximum input/segment bounds, compact
plain-input result capacity and exact borrowed text. The composed-tokenizer
control verifies token IDs, UTF-8 round trips, output-limit typing, reuse after
rejection and segmentation-before-output error precedence.
