# Prism inference private server

The independently built prism-inference-server executable supplies the same strict private HTTP contract used by SwiftBonsai's guarded development launcher. The owner verifies executable and model bytes before launch and retains process/Job ownership, the fixed memory limit, authenticated readiness and listener-PID checks.

The server loads the current pinned Swift Bonsai PQ2 model before binding numeric IPv4 loopback. It preserves exact count/completion input binding, one acknowledged output event, provisional streaming, cancellation/deadline rollback and explicit reset after failed generation. This extraction does not qualify full-model parity or device performance.

The private canonical request accepts either boolean value for
`chat_template_kwargs.enable_thinking`. Both exact counting and generation use
that selected template mode. With thinking disabled, the template closes the
reasoning block before generation and all generated text is visible answer
content. Supplied assistant reasoning history remains preserved. Effort stays
`xhigh` when thinking is enabled, and other template and sampler overrides remain
rejected. Existing public Rust `prepare` methods and the JSONL worker retain their
thinking-enabled behavior; `prepare_with_thinking` selects the mode explicitly.
