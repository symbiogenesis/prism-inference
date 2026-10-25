//! Packaged JSON command discovery and compatibility with legacy scoring limits.
#![forbid(unsafe_code)]

use std::fs;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{Value, json};

#[test]
fn version_advertises_policy_commands_without_resolving_a_request_file()
-> Result<(), Box<dyn std::error::Error>> {
    let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let scratch = std::env::temp_dir().join(format!(
        "SwiftBonsai-CommandDiscovery-{}-{nonce}",
        std::process::id()
    ));
    fs::create_dir(&scratch)?;
    let output = Command::new(env!("CARGO_BIN_EXE_prism-conformance-policy"))
        .arg("--version")
        .current_dir(&scratch)
        .output()?;
    fs::remove_dir(scratch)?;
    assert!(output.status.success());
    assert_eq!(output.stderr, Vec::<u8>::new());
    let version: Value = serde_json::from_slice(&output.stdout)?;
    assert_eq!(version["Name"], "prism-conformance-policy");
    assert_eq!(version["ProtocolVersion"], 1);
    let commands = version["Commands"]
        .as_array()
        .ok_or("Missing command capabilities")?;
    for name in ["prism-source", "prism-baseline"] {
        assert!(commands.iter().any(|command| command == name));
    }
    Ok(())
}

#[test]
fn prism_baseline_cli_matches_frozen_powershell_reports() -> Result<(), Box<dyn std::error::Error>>
{
    let golden: Value = serde_json::from_str(include_str!(
        "../../../tests/fixtures/prism-baseline-v1.json"
    ))?;
    let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let scratch = std::env::temp_dir().join(format!(
        "SwiftBonsai-BaselineCommand-{}-{nonce}",
        std::process::id()
    ));
    fs::create_dir(&scratch)?;
    let request_path = scratch.join("request.json");
    for (request, expected) in [
        (
            json!({"Operation":"catalog", "Catalog":golden["Catalog"]}),
            &golden["Catalog"],
        ),
        (
            json!({"Operation":"report", "Catalog":golden["Catalog"],
            "TargetResults":golden["TargetResults"], "ReferenceResults":golden["ReferenceResults"]}),
            &golden["Report"],
        ),
        (
            json!({"Operation":"baseline", "Catalog":golden["Catalog"],
            "Report":golden["Report"], "Baseline":golden["Baseline"]}),
            &golden["Gate"],
        ),
    ] {
        fs::write(&request_path, serde_json::to_vec(&request)?)?;
        let output = Command::new(env!("CARGO_BIN_EXE_prism-conformance-policy"))
            .args(["prism-baseline", "--request"])
            .arg(&request_path)
            .current_dir(&scratch)
            .output()?;
        assert!(
            output.status.success(),
            "{}. Evidence: {}",
            String::from_utf8_lossy(&output.stderr),
            scratch.display()
        );
        assert_eq!(output.stderr, Vec::<u8>::new());
        assert_eq!(&serde_json::from_slice::<Value>(&output.stdout)?, expected);
    }
    fs::remove_file(request_path)?;
    fs::remove_dir(scratch)?;
    Ok(())
}
