//! Full exporter-shaped intake, plus optional external Runner export replay.

use super::{admit, matrix_hash, sign_binding};
use crate::{Result, Value, json};

fn fixture() -> Result<Value> {
    let mut a = vec![0.0_f32; 5120];
    a[0] = 1.0;
    let a = vec![a];
    let b = vec![vec![0.001_f32; 5120]];
    let source = json!({"modelSha256":"1".repeat(64),"runtimeSha256":"2".repeat(64),
        "runtimeSourceCommit":"3".repeat(40),"adapterSha256":null,"adapterScaleBits":"00000000",
        "weightRepresentation":"PQ2_0","executionProfile":"experimental-ffn-cpu-scalar-unfused-v1",
        "graphIdentity":"teacher"});
    let site = json!({"layer":0,"inputBasis":"original-ffn-normalized-input",
        "targetBasis":"teacher-minus-retained-ffn-branch","inputWidth":5120,"outputWidth":5120});
    let omission = json!({"groupWidth":1024,"originalHiddenWidth":17408,"omittedGroups":[0,16],
        "transform":"independent-contiguous-1024-signed-hadamard-v1",
        "retainedAdapterPolicy":"zero-omitted-original-hidden-before-standard-adapter",
        "signIdentity":{"fullSignsSha256":"4".repeat(64),"downSignOffset":11264,"downSignCount":17408}});
    let screen = json!({"kind":"algebraic-development-screen-only","passed":true,
        "maximumDevelopmentRelativeSquaredError":0.1,"maximumDevelopmentMinusFitRelativeSquaredError":0.05,
        "requireNoWorseThanUncompensatedDevelopment":true,"rejectionReasons":[],
        "thresholdsAreDesignChoices":true,"answerQualityQualified":false,"changedPrefixDriftRecoveryQualified":false});
    let mut combined = screen.clone();
    combined["requireExportedFactorF64Screen"] = true.into();
    combined["requireSourceOrderF32Screen"] = true.into();
    combined["factorApplicationArithmetic"] = super::FACTOR_APPLICATION_ARITHMETIC.into();
    Ok(
        json!({"schemaVersion":2,"kind":"experimental-structured-ffn-compensation-v2",
        "enabled":false,"qualification":"unqualified-research-only","source":source,
        "studentExecution":{"executionProfile":super::GROUP_OMISSION_PROFILE,
            "graphIdentity":"omitted","prefixPolicySha256":"5".repeat(64)},
        "captureSha256":"6".repeat(64),"site":site,"omission":omission,
        "modelBoundSignSha256":sign_binding(&source,&site,&omission)?,
        "signPayloadArithmetic":"complete metadata sign_values array in original order, I32-le +/-1",
        "signBindingRecipe":"SHA256(domain bonsai-ffn-model-bound-signs-v1 NUL || modelSha256 ASCII || layer U32-le || fullSignsSha256 ASCII || downSignOffset U32-le || downSignCount U32-le)",
        "composition":"retained-ffn-branch-plus-normalized-input-low-rank-compensation",
        "teacherAndRetainedTargetsIncludeStandardAdapter":true,"applyStandardAdapterAgain":false,
        "formula":"retained_branch + sum_k b[k] * dot(a[k], normalized_original_basis_student_input)",
        "factorApplicationArithmetic":super::FACTOR_APPLICATION_ARITHMETIC,
        "fitTargetArithmetic":"F64(teacherTarget F32) - F64(retainedBranch F32)",
        "fitInputSha256":"7".repeat(64),"controls":{"rank":1,"ridge":0.001,"timeBudgetMilliseconds":5000},
        "rank":1,"factors":{"layout":"rank-major-f32-le","aSha256":matrix_hash(&a)?,"bSha256":matrix_hash(&b)?,"a":a,"b":b},
        "developmentScreen":combined,"exportedFactorF64DevelopmentScreen":screen,
        "sourceOrderF32DevelopmentScreen":screen,"sourceAuthenticated":false,
        "changedPrefixDriftRecoveryQualified":false,"runtimeImplemented":false,
        "runtimeImplementedScope":"released-serving-and-selective-loading","behaviorQualified":false}),
    )
}

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "Keep the full incoming-artifact mutation matrix beside the same frozen baseline."
)]
fn accepts_frozen_runner_artifact_and_rejects_mutations() -> Result<()> {
    let artifact = if let Some(path) = std::env::var_os("PRISM_FFN_EXPORTED_ARTIFACT") {
        serde_json::from_slice(&crate::read_bounded(
            std::path::Path::new(&path),
            super::MAX_RECOVERY_BYTES,
        )?)?
    } else {
        fixture()?
    };
    let source = &artifact["source"];
    let site = &artifact["site"];
    let omission = &artifact["omission"];
    let graph = artifact["studentExecution"]["graphIdentity"]
        .as_str()
        .ok_or_else(|| crate::reject("missing test graph"))?;
    let bytes = serde_json::to_vec(&artifact)?;
    let compensation = admit(&bytes, source, site, omission, graph)?;
    assert_eq!(compensation.identity().len(), 64);
    for (pointer, replacement) in [
        ("/source/runtimeSha256", json!("8".repeat(64))),
        ("/source/adapterScaleBits", json!("3fa66666")),
        ("/source/runtimeSourceCommit", json!("8".repeat(40))),
        ("/site/inputBasis", json!("original-ffn-hidden")),
        ("/omission/omittedGroups", json!([1, 16])),
        (
            "/omission/signIdentity/fullSignsSha256",
            json!("8".repeat(64)),
        ),
        ("/studentExecution/graphIdentity", json!("foreign")),
        ("/modelBoundSignSha256", json!("8".repeat(64))),
        (
            "/factorApplicationArithmetic",
            json!("f64-summed-correction"),
        ),
        ("/developmentScreen/passed", json!(false)),
        ("/sourceOrderF32DevelopmentScreen/passed", json!(false)),
        ("/exportedFactorF64DevelopmentScreen/passed", json!(false)),
        ("/factors/aSha256", json!("8".repeat(64))),
        ("/rank", json!(17)),
        ("/applyStandardAdapterAgain", json!(true)),
        (
            "/developmentScreen/maximumDevelopmentRelativeSquaredError",
            json!(0.9),
        ),
        (
            "/sourceOrderF32DevelopmentScreen/maximumDevelopmentMinusFitRelativeSquaredError",
            json!(0.9),
        ),
        (
            "/exportedFactorF64DevelopmentScreen/requireNoWorseThanUncompensatedDevelopment",
            json!(false),
        ),
        (
            "/sourceOrderF32DevelopmentScreen/rejectionReasons",
            json!(["contradiction"]),
        ),
    ] {
        let mut changed = artifact.clone();
        *changed
            .pointer_mut(pointer)
            .ok_or_else(|| crate::reject("missing fixture mutation field"))? = replacement;
        assert!(
            admit(
                &serde_json::to_vec(&changed)?,
                source,
                site,
                omission,
                graph
            )
            .is_err(),
            "accepted {pointer}"
        );
    }
    let mut changed = artifact.clone();
    changed["factors"]["a"][0][0] = if artifact["factors"]["a"][0][0] == 0.0 {
        1.0.into()
    } else {
        0.0.into()
    };
    assert!(
        admit(
            &serde_json::to_vec(&changed)?,
            source,
            site,
            omission,
            graph
        )
        .is_err()
    );
    let mut changed = artifact.clone();
    changed
        .as_object_mut()
        .ok_or_else(|| crate::reject("missing test object"))?
        .remove("factorApplicationArithmetic");
    assert!(
        admit(
            &serde_json::to_vec(&changed)?,
            source,
            site,
            omission,
            graph
        )
        .is_err()
    );
    changed = artifact.clone();
    changed["unexpectedField"] = true.into();
    assert!(
        admit(
            &serde_json::to_vec(&changed)?,
            source,
            site,
            omission,
            graph
        )
        .is_err()
    );
    Ok(())
}

#[test]
fn rank_sixteen_pretty_export_fits_recovery_bound_beyond_rank_one_limit() -> Result<()> {
    let mut artifact = fixture()?;
    let factors = vec![vec![0.123_456_7_f32; 5120]; 16];
    let hash = matrix_hash(&factors)?;
    artifact["rank"] = 16.into();
    artifact["controls"]["rank"] = 16.into();
    artifact["factors"]["a"] = json!(factors);
    artifact["factors"]["b"] = json!(factors);
    artifact["factors"]["aSha256"] = hash.clone().into();
    artifact["factors"]["bSha256"] = hash.into();
    let bytes = serde_json::to_vec_pretty(&artifact)?;
    assert!(u64::try_from(bytes.len())? > crate::MAX_PATCH_BYTES);
    assert!(u64::try_from(bytes.len())? <= super::MAX_RECOVERY_BYTES);
    admit(
        &bytes,
        &artifact["source"],
        &artifact["site"],
        &artifact["omission"],
        "omitted",
    )?;
    Ok(())
}
