# Prism inference private server

The independently built prism-inference-server executable supplies the same strict private HTTP contract used by SwiftBonsai's guarded development launcher. The owner verifies executable and model bytes before launch and retains process/Job ownership, the fixed memory limit, authenticated readiness and listener-PID checks.

The server loads the current pinned Swift Bonsai PQ2 model before binding numeric IPv4 loopback. It preserves exact count/completion input binding, one acknowledged output event, provisional streaming, cancellation/deadline rollback and explicit reset after failed generation. This extraction does not qualify full-model parity or device performance.
