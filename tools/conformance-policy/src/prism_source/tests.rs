//! Focused ordinary source policy, provenance and retained-input regressions.

use std::fmt::Write as _;
use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};

use super::*;

const IDS: [&str; 6] = [
    "tests/test-future-source.cpp",
    "tests/test-tokenizer-random.py",
    BACKEND,
    "tests/test-thread-safety.cpp",
    "tests/test-quantize-stats.cpp",
    "tests/test-grammar-llguidance.cpp",
];

fn scratch() -> Result<PathBuf> {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| error.to_string())?
        .as_nanos();
    let directory = std::env::temp_dir().join(format!(
        "SwiftBonsai-PrismSource-{}-{nonce}",
        std::process::id()
    ));
    fs::create_dir(&directory).map_err(|error| error.to_string())?;
    Ok(directory)
}

fn write(path: &Path, value: &Value) -> Result<()> {
    fs::write(
        path,
        serde_json::to_vec_pretty(value).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())
}

fn fixture() -> Result<(PathBuf, Value)> {
    let root = scratch()?;
    let request = json!({
        "Operation":"merge", "InputDirectory":root,
        "Catalog":{"UpstreamCommit":"a".repeat(40),"ArchiveSha256":"b".repeat(64),"AdapterIds":IDS},
        "ProjectRevision":"c".repeat(40), "HarnessSha256":"d".repeat(64), "CommonBridgeSha256":"e".repeat(64)
    });
    for index in 0..4 {
        let directory = root.join(format!("prism-ordinary-results-{index}"));
        fs::create_dir(&directory).map_err(|error| error.to_string())?;
        write(
            &directory.join("receipt.json"),
            &json!({
                "SchemaVersion":1,"ProjectRevision":request["ProjectRevision"],
                "SourceCommit":request["Catalog"]["UpstreamCommit"],"SourceArchiveSha256":request["Catalog"]["ArchiveSha256"],
                "Status":"awaiting-backend","Selection":"All","SourcePartition":index,"SourcePartitionCount":4,
                "NativePlatform":"ARM64","Architecture":"Arm64","DeferredSource":BACKEND,
                "HarnessSha256":request["HarnessSha256"],"NativeBuilds":[{"Family":"common","ExecutableSha256":request["CommonBridgeSha256"]}],
                "CaseRuns":[{"Partition":index}],"ExtraReceipt":{"Retained":true}
            }),
        )?;
        for target in ["cpp", "rust"] {
            let cases: Vec<_> = IDS.iter().copied().filter(|id| *id != BACKEND && partition(id) == index)
                .map(|id| json!({"Id":id,"Status":if target == "cpp" {"pass"} else {"fail"},"Detail":"Synthetic outcome.","ExtraCase":[1,2]})).collect();
            write(
                &directory.join(format!("{target}-results.json")),
                &json!({"SchemaVersion":1,"UpstreamCommit":request["Catalog"]["UpstreamCommit"],"Cases":cases}),
            )?;
        }
    }
    Ok((root, request))
}

#[test]
fn partitions_preserve_order_and_include_future_sources() -> Result<()> {
    let cases: Vec<_> = IDS
        .iter()
        .enumerate()
        .rev()
        .map(|(ordinal, id)| json!({"Id":id,"Ordinal":ordinal}))
        .collect();
    for (index, expected) in [(0, vec![1]), (1, vec![3]), (2, vec![4, 5]), (3, vec![0, 2])] {
        assert_eq!(
            run(json!({"Operation":"partition","Cases":cases,"Index":index}))?,
            json!({"Ordinals":expected})
        );
    }
    assert_eq!(
        run(json!({"Operation":"partition","Cases":[],"Index":0}))?,
        json!({"Ordinals":[]})
    );
    for invalid in [json!(-1), json!(4), json!("0"), json!(0.0), Value::Null] {
        assert!(run(json!({"Operation":"partition","Cases":cases,"Index":invalid})).is_err());
    }
    for invalid in [
        json!(-1),
        json!("0"),
        json!(0.0),
        json!(2_147_483_648_u64),
        Value::Null,
    ] {
        assert!(run(json!({"Operation":"partition","Cases":[{"Id":"future","Ordinal":invalid}],"Index":3})).is_err());
    }
    for invalid in [
        json!([{"Id":"a","Ordinal":0},{"Id":"a","Ordinal":1}]),
        json!([{"Id":"a","Ordinal":0},{"Id":"b","Ordinal":0}]),
        json!([{"Id":" ","Ordinal":0}]),
        json!([["a", 0]]),
        json!([null]),
    ] {
        assert!(run(json!({"Operation":"partition","Cases":invalid,"Index":3})).is_err());
    }
    Ok(())
}

#[test]
fn fan_in_keeps_failures_order_extensions_and_exact_receipt_hashes() -> Result<()> {
    let (root, request) = fixture()?;
    // Independent partitions can use different native features and executable bytes.
    let path = root.join("prism-ordinary-results-1/receipt.json");
    let mut receipt: Value =
        serde_json::from_slice(&fs::read(&path).map_err(|error| error.to_string())?)
            .map_err(|error| error.to_string())?;
    receipt["HarnessSha256"] = json!("f".repeat(64));
    receipt["NativeBuilds"][0]["ExecutableSha256"] = json!("0".repeat(64));
    write(&path, &receipt)?;
    let merged = run(request)?;
    let expected: Vec<_> = IDS.iter().copied().filter(|id| *id != BACKEND).collect();
    for target in ["cpp", "rust"] {
        let cases = array(&merged[target], "Cases")?;
        assert_eq!(
            cases
                .iter()
                .map(|case| text(case, "Id"))
                .collect::<Result<Vec<_>>>()?,
            expected
        );
        assert!(
            cases
                .iter()
                .all(|case| case["Status"] == if target == "cpp" { "pass" } else { "fail" })
        );
        assert!(cases.iter().all(|case| case["ExtraCase"] == json!([1, 2])));
    }
    assert_eq!(merged["Receipt"]["SourcePartition"], -1);
    assert_eq!(merged["Receipt"]["ExtraReceipt"], json!({"Retained":true}));
    assert_eq!(
        merged["Receipt"]["CaseRuns"],
        json!([{"Partition":0},{"Partition":1},{"Partition":2},{"Partition":3}])
    );
    for (index, partition) in array(&merged["Receipt"], "SourcePartitions")?
        .iter()
        .enumerate()
    {
        let bytes = fs::read(root.join(format!("prism-ordinary-results-{index}/receipt.json")))
            .map_err(|error| error.to_string())?;
        let mut digest = String::new();
        for byte in Sha256::digest(bytes) {
            write!(&mut digest, "{byte:02x}").map_err(|error| error.to_string())?;
        }
        assert_eq!(partition["ReceiptSha256"], digest);
    }
    Ok(())
}

#[test]
fn fan_in_rejects_foreign_and_mistyped_provenance() -> Result<()> {
    let (root, request) = fixture()?;
    let path = root.join("prism-ordinary-results-0/receipt.json");
    let original: Value =
        serde_json::from_slice(&fs::read(&path).map_err(|error| error.to_string())?)
            .map_err(|error| error.to_string())?;
    for (field, invalid) in [
        ("SchemaVersion", json!("1")),
        ("SchemaVersion", json!(1.0)),
        ("ProjectRevision", json!("f".repeat(40))),
        ("SourcePartition", json!(1)),
        ("SourcePartition", json!("0")),
        ("SourcePartitionCount", json!("4")),
        ("SourcePartitionCount", json!(4.0)),
        ("Status", json!("building")),
        ("SourceCommit", json!("f".repeat(40))),
        ("SourceArchiveSha256", json!("f".repeat(64))),
        ("HarnessSha256", json!("f".repeat(64))),
        ("CaseRuns", json!({})),
        ("NativeBuilds", json!([])),
        (
            "NativeBuilds",
            json!([original["NativeBuilds"][0], original["NativeBuilds"][0]]),
        ),
    ] {
        let mut changed = original.clone();
        changed[field] = invalid;
        write(&path, &changed)?;
        assert!(
            run(request.clone()).is_err(),
            "Accepted invalid receipt {field}"
        );
    }
    let mut changed = original.clone();
    changed["NativeBuilds"][0]["ExecutableSha256"] = json!("f".repeat(64));
    write(&path, &changed)?;
    assert!(run(request.clone()).is_err());
    write(&path, &original)?;
    for field in ["ProjectRevision", "HarnessSha256", "CommonBridgeSha256"] {
        let mut changed = request.clone();
        changed[field] = json!("INVALID");
        assert!(run(changed).is_err());
    }
    fs::rename(&path, path.with_extension("missing")).map_err(|error| error.to_string())?;
    assert!(run(request).is_err());
    Ok(())
}

#[test]
fn fan_in_rejects_missing_duplicate_and_moved_results() -> Result<()> {
    let (root, request) = fixture()?;
    let path = root.join("prism-ordinary-results-2/rust-results.json");
    let original: Value =
        serde_json::from_slice(&fs::read(&path).map_err(|error| error.to_string())?)
            .map_err(|error| error.to_string())?;
    for changed in [
        json!({"SchemaVersion":1,"UpstreamCommit":original["UpstreamCommit"],"Cases":[original["Cases"][0]]}),
        json!({"SchemaVersion":1,"UpstreamCommit":original["UpstreamCommit"],"Cases":[original["Cases"][0],original["Cases"][0]]}),
        json!({"SchemaVersion":1,"UpstreamCommit":original["UpstreamCommit"],"Cases":[{"Id":IDS[1]},original["Cases"][1]]}),
        json!({"SchemaVersion":1,"UpstreamCommit":"f".repeat(40),"Cases":original["Cases"]}),
        json!({"SchemaVersion":"1","UpstreamCommit":original["UpstreamCommit"],"Cases":original["Cases"]}),
        json!({"SchemaVersion":1,"UpstreamCommit":original["UpstreamCommit"],"Cases":{}}),
    ] {
        write(&path, &changed)?;
        assert!(run(request.clone()).is_err());
    }
    Ok(())
}

#[test]
fn artifact_reads_are_bounded_and_reject_git_storage() -> Result<()> {
    let root = scratch()?;
    let path = root.join("input.json");
    fs::write(&path, b"{\"a\":1}").map_err(|error| error.to_string())?;
    let mut inputs = Inputs::new();
    assert!(inputs.read(&path, 6).is_err());
    assert_eq!(inputs.read(&path, 7)?.0, json!({"a":1}));
    drop(inputs);
    fs::write(&path, [0xff, 0xfe, 0xff]).map_err(|error| error.to_string())?;
    assert!(
        Inputs {
            handles: Vec::new()
        }
        .read(&path, 32)
        .is_err()
    );
    fs::create_dir(root.join(".git")).map_err(|error| error.to_string())?;
    assert!(external_path(&path).is_err());
    assert!(expanded_path(Path::new("relative.json")).is_err());
    Ok(())
}

#[cfg(windows)]
#[test]
fn artifact_handles_deny_writes_and_replacement_until_merge_finishes() -> Result<()> {
    let root = scratch()?;
    let path = root.join("input.json");
    fs::write(&path, b"{}").map_err(|error| error.to_string())?;
    let mut inputs = Inputs::new();
    inputs.read(&path, 32)?;
    assert!(OpenOptions::new().write(true).open(&path).is_err());
    assert!(fs::rename(&path, root.join("moved.json")).is_err());
    assert!(fs::rename(&root, root.with_extension("moved")).is_err());
    drop(inputs);
    fs::write(&path, b"{}").map_err(|error| error.to_string())?;
    Ok(())
}

#[test]
fn artifact_bom_encodings_preserve_valid_unicode_and_original_bytes() -> Result<()> {
    let root = scratch()?;
    let path = root.join("input.json");
    let text = "{\"Detail\":\"\u{4e2d}\u{1f600}\"}";
    let mut encodings = vec![
        text.as_bytes().to_vec(),
        [vec![0xef, 0xbb, 0xbf], text.as_bytes().to_vec()].concat(),
    ];
    for (little, bom) in [(true, vec![0xff, 0xfe]), (false, vec![0xfe, 0xff])] {
        let mut bytes = bom;
        for unit in text.encode_utf16() {
            bytes.extend(if little {
                unit.to_le_bytes()
            } else {
                unit.to_be_bytes()
            });
        }
        encodings.push(bytes);
    }
    for (little, bom) in [
        (true, vec![0xff, 0xfe, 0, 0]),
        (false, vec![0, 0, 0xfe, 0xff]),
    ] {
        let mut bytes = bom;
        for character in text.chars() {
            let unit = u32::from(character);
            bytes.extend(if little {
                unit.to_le_bytes()
            } else {
                unit.to_be_bytes()
            });
        }
        encodings.push(bytes);
    }
    for bytes in encodings {
        fs::write(&path, &bytes).map_err(|error| error.to_string())?;
        let (value, digest) = Inputs::new().read(&path, 1024)?;
        assert_eq!(value["Detail"], "\u{4e2d}\u{1f600}");
        let mut expected = String::new();
        for byte in Sha256::digest(bytes) {
            write!(&mut expected, "{byte:02x}").map_err(|error| error.to_string())?;
        }
        assert_eq!(digest, expected);
    }
    for invalid in [
        vec![0xff, 0xfe, 0x22],
        vec![0, 0, 0xfe, 0xff, 0, 0],
        vec![0xff, 0xfe, 0, 0, 0xff, 0xff, 0xff, 0xff],
    ] {
        assert!(decode_text(&invalid).is_err());
    }
    Ok(())
}
