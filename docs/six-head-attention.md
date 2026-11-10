# Six-head Swift decode locality

The exact Swift graph has 24 query heads and four KV heads. Its decode path now
partitions workers by complete six-head KV groups when the selected worker count
does not exceed the KV-head count, including the default four-worker policy.
Larger worker budgets retain the finer existing partition. Each group reads and converts
one cached key element once for all six queries and reduces each cached value
tile once for all six heads. Source-order arithmetic and other head geometries
retain their existing dispatch. Tiled prompt prefill retains its distinct path.

The prior decode path grouped four heads and handled the other two separately.
At Swift's 256-channel head width it visited the value cache twice for the
four-head 128-channel tiles and sixteen more times for the two single-head
32-channel tiles. Six-head 64-channel tiles require four visits. These are source
iteration counts, not measured traffic or latency. Score evaluation trades the
old four-key/four-query block for one-key/six-query sharing; final ARM64 codegen,
spills and matched whole-request benefit still require qualification.

The six-head temporal accumulator is 24 KiB, below the existing four-head
128-channel accumulator's 32 KiB. Reduced values reuse dead first-lane storage;
there is no additional heap scratch or expanded cache. Existing conservative
admission remains unchanged. Each query retains its sixteen temporal FMA lanes,
padding signed-zero effects, original reduction tree, globally normalized F16
probabilities and separate sigmoid gate arithmetic. Baseline NEON is optional;
the scalar implementation preserves the same bit contract and bounds.

Six-head value helpers validate the entire reduction before gated output
publication, including failures in the last head. Their output is transactional
graph scratch. Committed KV pages, recurrent/convolution state and request
rollback are unchanged; existing four-head helper publication semantics remain
unchanged. Worker completion and cancellation retain the graph's existing layer
and request boundaries, with no new asynchronous work or cache owner.

Focused controls independently compare score lanes/tails and every value
head/channel against the native scalar reduction, count one iterator visit per
cached row, reject malformed geometry and late-head failures atomically, and
compare complete six-head attention across sealed KV pages and mutable tails.
The existing original-gate worker test covers Source and pinned execution with
one, two, three, four, seven and eighteen workers. Owning hosted strict analysis, portable compilation,
production linking, actual-model state/logit/adapter/cancellation parity and
matched complete-request latency/resources remain separate acceptance gates.
