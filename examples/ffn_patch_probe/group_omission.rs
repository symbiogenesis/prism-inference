//! Two-pass, one-site structured FFN capture with matched original teacher rows.

use std::collections::BTreeMap;

use super::{
    Deadline, Decoder, EXECUTION_PROFILE, FfnProbe, GROUP_OMISSION_PROFILE, Hasher,
    MAX_CAPTURE_BYTES, Qwen35, Representation, Request, Result, SURROGATE_INPUT_BASIS,
    SWIFT_ADAPTER_SHA256, Value, factor_hash, hex, json, publish, reject,
};

fn digest(domain: &[u8], bytes: &[u8]) -> Result<String> {
    let mut hash = Hasher::new()?;
    hash.update(domain)?;
    hash.update(bytes)?;
    hex(hash.finish()?)
}

fn prefix(graph: &str, tokens: &[u32]) -> Result<String> {
    let mut hash = Hasher::new()?;
    hash.update(b"prism-ffn-capture-token-prefix-v1\0")?;
    hash.update(&u64::try_from(graph.len())?.to_le_bytes())?;
    hash.update(graph.as_bytes())?;
    hash.update(&u64::try_from(tokens.len())?.to_le_bytes())?;
    for token in tokens {
        hash.update(&token.to_le_bytes())?;
    }
    hex(hash.finish()?)
}

#[expect(
    clippy::too_many_lines,
    reason = "Keep original replay, reset, changed-graph replay and publication ordering auditable."
)]
#[expect(
    clippy::redundant_pub_crate,
    reason = "The parent binary owns this capture entry point; it is not a public API."
)]
pub(super) fn capture(
    request: &Request,
    representation: Representation,
    scale: f32,
    executable_sha256: &str,
    graph: Qwen35,
    deadline: &Deadline,
) -> Result<Value> {
    let groups = request
        .omitted_groups
        .as_ref()
        .ok_or_else(|| reject("structured capture requires omittedGroups"))?;
    let mut probe = FfnProbe::new(graph, request.layer, None)?;
    let original_graph = probe.identity().to_owned();
    let mut original = BTreeMap::new();
    let mut logits = Vec::new();
    // Reuse the same immutable model and serially reset its numerical state.
    // Both complete passes share the original deadline and token admission.
    for sequence in &request.sequences {
        deadline.check()?;
        probe.reset()?;
        for (index, &token) in sequence.token_ids.iter().enumerate() {
            deadline.check()?;
            if sequence.capture_token_indices.contains(&index) {
                let observation = probe.evaluate_surrogate_into(token, &mut logits)?;
                deadline.check()?;
                original.insert((sequence.sample_id.as_str(), index), observation.target);
            } else {
                probe.evaluate_into(token, &mut logits)?;
            }
        }
    }
    deadline.check()?;
    let mut probe = probe.with_group_omission(groups.clone())?;
    let student_graph = probe.identity().to_owned();
    let full_signs = probe
        .omission_signs_sha256()
        .ok_or_else(|| reject("missing actual omission sign identity"))?
        .to_owned();
    let source = json!({
        "modelSha256":representation.model_sha256(),"runtimeSha256":executable_sha256,
        "runtimeSourceCommit":request.runtime_source_commit,
        "adapterSha256":request.adapter_path.as_ref().map(|_| SWIFT_ADAPTER_SHA256),
        "adapterScaleBits":format!("{:08x}",scale.to_bits()),
        "weightRepresentation":representation.name(),"executionProfile":EXECUTION_PROFILE,
        "graphIdentity":original_graph,
    });
    let policy = json!({"graphIdentity":student_graph,"omittedGroups":groups,
        "executionProfile":GROUP_OMISSION_PROFILE,"recovery":"none",
        "rollout":"fixed-token-ids-teacher-forced","reset":"before-each-sequence",
        "sequences":request.sequences.iter().map(|sequence|json!({
            "sampleId":sequence.sample_id,"groupId":sequence.group_id,
            "split":sequence.split,"tokenIds":sequence.token_ids,
        })).collect::<Vec<_>>()});
    let prefix_policy_sha256 = digest(
        b"prism-ffn-student-prefix-policy-v1\0",
        &serde_json::to_vec(&policy)?,
    )?;
    let mut rows = Vec::new();
    for sequence in &request.sequences {
        deadline.check()?;
        probe.reset()?;
        for (index, &token) in sequence.token_ids.iter().enumerate() {
            deadline.check()?;
            if sequence.capture_token_indices.contains(&index) {
                let observation = probe.evaluate_group_omission_into(token, &mut logits)?;
                deadline.check()?;
                let baseline = original
                    .remove(&(sequence.sample_id.as_str(), index))
                    .ok_or_else(|| reject("missing matched original teacher row"))?;
                rows.push(json!({
                    "sampleId":sequence.sample_id,"groupId":sequence.group_id,
                    "tokenIndex":index,"split":sequence.split,
                    "sharedInputSha256":factor_hash(&observation.input)?,
                    "studentPrefixSha256":prefix(&student_graph,&sequence.token_ids[..=index])?,
                    "originalPrefixSha256":prefix(&original_graph,&sequence.token_ids[..=index])?,
                    "input":observation.input,"teacherTarget":observation.teacher_target,
                    "retainedBranch":observation.retained_branch,"originalPrefixTeacherTarget":baseline,
                }));
            } else {
                probe.evaluate_into(token, &mut logits)?;
            }
        }
    }
    if !original.is_empty() {
        return Err(reject("unconsumed original teacher rows"));
    }
    let row_count = rows.len();
    let mut packet = json!({"schemaVersion":2,"source":source,
        "studentExecution":{"executionProfile":GROUP_OMISSION_PROFILE,
            "graphIdentity":student_graph,"prefixPolicySha256":prefix_policy_sha256},
        "site":{"layer":request.layer,"inputBasis":SURROGATE_INPUT_BASIS,
            "targetBasis":"teacher-minus-retained-ffn-branch","inputWidth":5120,"outputWidth":5120},
        "omission":{"groupWidth":1024,"originalHiddenWidth":17408,"omittedGroups":groups,
            "transform":"independent-contiguous-1024-signed-hadamard-v1",
            "retainedAdapterPolicy":"zero-omitted-original-hidden-before-standard-adapter",
            "signIdentity":{"fullSignsSha256":full_signs,"downSignOffset":11264,"downSignCount":17408}},
        "rows":rows,
    });
    let capture_sha256 = digest(
        b"prism-ffn-structured-capture-packet-v1\0",
        &serde_json::to_vec(&packet)?,
    )?;
    packet["captureSha256"] = json!(capture_sha256);
    let bytes = serde_json::to_vec(&packet)?;
    if bytes.len() > MAX_CAPTURE_BYTES {
        return Err(reject(
            "serialized structured FFN capture exceeds its byte bound",
        ));
    }
    deadline.check()?;
    publish(&request.capture_path, &bytes)?;
    Ok(json!({"event":"ffn-probe-captured","rows":row_count,
        "capturePath":request.capture_path,"executionProfile":GROUP_OMISSION_PROFILE,
        "originalReplayPasses":1,"studentReplayPasses":1,"recoveryApplied":false,
        "allWeightsLoaded":true,"fullTeacherFfnExecuted":true,
        "teacherForcedOnly":true,"performanceQualified":false,"processMemoryQualified":false,
        "qualification":"unqualified one-site research capture; no compensated rollout or free generation"}))
}

#[cfg(test)]
mod tests {
    use super::prefix;
    use crate::Result;

    #[test]
    fn structured_prefix_hash_binds_graph_order_length_and_token_bits() -> Result<()> {
        let baseline = prefix("teacher", &[1, 2])?;
        for (graph, tokens) in [
            ("student", vec![1, 2]),
            ("teacher", vec![2, 1]),
            ("teacher", vec![1]),
            ("teacher", vec![1, 2, 0]),
        ] {
            assert_ne!(baseline, prefix(graph, &tokens)?);
        }
        assert_eq!(baseline, prefix("teacher", &[1, 2])?);
        Ok(())
    }
}
