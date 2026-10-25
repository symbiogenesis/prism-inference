`weighted-q4.json` contains eight focused native production `Q4_0` rows.
The real pinned C++ common bridge executed `ggml_quantize_chunk` with each
256-element input and either all-ones or varying nonnegative importance.
It used the same x64 Release bridge and source revision recorded in
`row-quantizers.json`; the file retains their SHA-256 provenance.

Fixture SHA-256:
`78cecc196fc25aff174a03c69df01bd90d4ad5357f793c41b547a3abeb866994`.
These are exact codec correctness records without timing measurements.
