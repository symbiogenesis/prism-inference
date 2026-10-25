//! Catalog identity and regression-gate boundary checks.

use super::*;

const FIRST: &str = "tests/test-first.cpp";
const SECOND: &str = "tests/nested/test-second.py";

fn catalog() -> Value {
    json!({
        "SchemaVersion":1, "UpstreamCommit":"a".repeat(40), "ArchiveSha256":"b".repeat(64),
        "TestCMakeSourceSha256":"c".repeat(64), "Measure":"upstream-source-files",
        "AdapterIds":[FIRST], "Cases":[
            {"Id":FIRST,"SourceSha256":"d".repeat(64)},
            {"Id":SECOND,"SourceSha256":"e".repeat(64)},
        ],
    })
}

fn results(target_status: &str, reference_status: &str) -> Value {
    let catalog = catalog();
    let result = |status: &str| {
        json!({
            "SchemaVersion":1, "UpstreamCommit":catalog["UpstreamCommit"],
            "Cases":[{"Id":FIRST,"Status":status,"Detail":"Details: \"quoted\", \\, \n, λ, 😀"}],
        })
    };
    json!({
        "Operation":"report", "Catalog":catalog,
        "TargetResults":result(target_status), "ReferenceResults":result(reference_status),
    })
}

fn gate(report: &Value) -> Result<Value> {
    let catalog = catalog();
    let digest = Catalog::parse(&catalog)?.digest()?;
    run(json!({
        "Operation":"baseline", "Catalog":catalog, "Report":report,
        "Baseline":{
            "SchemaVersion":1, "UpstreamCommit":catalog["UpstreamCommit"],
            "CatalogSha256":digest, "Measure":catalog["Measure"],
            "Total":2,"MinimumPassed":1,"RequiredPassedIds":[FIRST],
            "RequiredReferencePassedIds":[FIRST],
        },
    }))
}

#[test]
fn pinned_catalog_identity_matches_powershell_and_ignores_extensions() -> Result<()> {
    let mut catalog: Value =
        serde_json::from_str(include_str!("../../tests/fixtures/prism-catalog-v1.json"))
            .map_err(|error| error.to_string())?;
    assert_eq!(catalog["Cases"].as_array().map(Vec::len), Some(71));
    catalog["CatalogSha256"] = json!("stale digest");
    catalog["Provenance"] = json!({"Extra":"é, λ, 😀, \" and \\"});
    catalog["Cases"][0]["Extra"] = json!([true, "retained"]);
    let imported = run(json!({"Operation":"catalog","Catalog":catalog}))?;
    assert_eq!(
        imported["CatalogSha256"],
        "2a262f01c68f4137f01519188323fa4ad9edb55a5b5ec05f1b7d00df40f03efd"
    );
    assert_eq!(imported["Provenance"], catalog["Provenance"]);
    assert_eq!(imported["Cases"], catalog["Cases"]);
    Ok(())
}

#[test]
fn catalog_rejects_coerced_numbers_ambiguous_ids_and_invalid_provenance() {
    for schema in [
        json!("1"),
        json!(true),
        json!(1.0),
        json!(0),
        json!(1_000_001),
        Value::Null,
    ] {
        let mut catalog = catalog();
        catalog["SchemaVersion"] = schema;
        assert!(run(json!({"Operation":"catalog","Catalog":catalog})).is_err());
    }
    for id in [
        "tests/test-first.cpp\n",
        "tests/../test-first.cpp",
        "Tests/test-first.cpp",
        "tests/test-.cpp",
        "tests/test-first.CPP",
    ] {
        let mut catalog = catalog();
        catalog["Cases"][1]["Id"] = json!(id);
        assert!(run(json!({"Operation":"catalog","Catalog":catalog})).is_err());
    }
    let mut duplicate = catalog();
    duplicate["Cases"][1]["Id"] = json!(FIRST);
    assert!(run(json!({"Operation":"catalog","Catalog":duplicate})).is_err());
    let mut wrong_case = catalog();
    wrong_case["ArchiveSha256"] = json!("B".repeat(64));
    assert!(run(json!({"Operation":"catalog","Catalog":wrong_case})).is_err());
}

#[test]
fn source_inventory_is_exact_and_binds_cmake() {
    let mut request = json!({
        "Operation":"catalog", "Catalog":catalog(), "TestCMakeSourceSha256":"c".repeat(64),
        "SourceInventory":[{"Id":SECOND,"Sha256":"e".repeat(64)},{"Id":FIRST,"Sha256":"d".repeat(64)}],
    });
    assert!(run(request.clone()).is_ok());
    for inventory in [
        json!([{"Id":FIRST,"Sha256":"d".repeat(64)}]),
        json!([{"Id":FIRST,"Sha256":"d".repeat(64)},{"Id":FIRST,"Sha256":"d".repeat(64)}]),
        json!([{"Id":"tests/test-unknown.cpp","Sha256":"e".repeat(64)}]),
        json!([{"Id":FIRST,"Sha256":"f".repeat(64)}]),
    ] {
        let mut malformed = request.clone();
        malformed["SourceInventory"] = inventory;
        assert!(run(malformed).is_err());
    }
    request["TestCMakeSourceSha256"] = json!("0".repeat(64));
    assert!(run(request).is_err());
}

#[test]
fn sparse_reports_keep_catalog_order_and_all_red_sources() -> Result<()> {
    let report = run(results("skip", "pass"))?;
    assert_eq!(report["Cases"][0]["Id"], FIRST);
    assert_eq!(report["Cases"][1]["Id"], SECOND);
    assert_eq!(report["Cases"][1]["Status"], "unsupported");
    assert_eq!(report["Summary"]["Total"], 2);
    assert_eq!(report["Summary"]["RedIds"], json!([FIRST, SECOND]));
    assert_eq!(report["Summary"]["ReferencePassed"], true);
    for target in [true, false] {
        let field = if target {
            "TargetResults"
        } else {
            "ReferenceResults"
        };
        let mut missing = results("pass", "pass");
        missing[field]["Cases"] = json!([]);
        assert!(run(missing).is_err());
        let mut unsupported = results("pass", "pass");
        unsupported[field]["Cases"][0]["Status"] = json!("unsupported");
        assert!(run(unsupported).is_err());
        let mut duplicate = results("pass", "pass");
        let case = duplicate[field]["Cases"][0].clone();
        duplicate[field]["Cases"] = json!([case, case]);
        assert!(run(duplicate).is_err());
    }
    Ok(())
}

#[test]
fn reference_failures_reject_even_with_green_target_and_forged_summary() -> Result<()> {
    let mut report = run(results("pass", "fail"))?;
    report["Summary"] = json!({"Passed":2,"ReferencePassed":true,"ReferenceFailureIds":[]});
    let gate = gate(&report)?;
    assert_eq!(gate["Passed"], false);
    assert_eq!(gate["Summary"]["Passed"], 1);
    assert_eq!(gate["Summary"]["ReferenceFailureIds"], json!([FIRST]));
    assert_eq!(
        gate["Issues"],
        json!([
            format!("Previously passing C++ reference regressed: {FIRST} (fail)."),
            format!("C++ reference did not pass for adapted case: {FIRST}."),
        ])
    );
    Ok(())
}

#[test]
fn previously_green_identity_cannot_be_replaced_by_another_green_source() -> Result<()> {
    let mut report = run(results("fail", "pass"))?;
    report["Cases"][1]["Status"] = json!("pass");
    report["Cases"][1]["ReferenceStatus"] = json!("pass");
    let rejected = gate(&report)?;
    assert_eq!(rejected["Passed"], false);
    assert_eq!(
        rejected["Issues"],
        json!([format!("Previously green case regressed: {FIRST} (fail).")])
    );
    report["Cases"][0]["Status"] = json!("pass");
    assert_eq!(gate(&report)?["Passed"], true);
    report["Cases"][0]["ReferenceStatus"] = json!("unsupported");
    assert!(gate(&report).is_err());
    report["Cases"][0]["ReferenceStatus"] = json!("pass");
    report["Cases"][1]["SourceSha256"] = json!("f".repeat(64));
    assert!(gate(&report).is_err());
    Ok(())
}
