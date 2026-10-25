#![forbid(unsafe_code)]
//! Model-free capability handshake and fail-closed option-boundary regressions.

#[cfg(feature = "cli")]
mod driver {
    use bonsai_prism_runtime::Result;
    use bonsai_prism_runtime::kernels::MatvecScratch;
    use serde_json::{Value, json};
    use std::fs;
    use std::path::PathBuf;
    use std::process::{Command, Stdio};
    use std::time::{SystemTime, UNIX_EPOCH};

    struct Lane(PathBuf);

    impl Drop for Lane {
        fn drop(&mut self) {
            let _removed = fs::remove_dir(&self.0);
        }
    }

    #[test]
    fn capabilities_driver_reports_compiled_support_without_model_or_qualification() -> Result<()> {
        let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let lane = Lane(std::env::temp_dir().join(format!(
            "Prism-CapabilityDriver-{}-{nonce}",
            std::process::id()
        )));
        fs::create_dir(&lane.0)?;
        let output = Command::new(env!("CARGO_BIN_EXE_prism-inference"))
            .arg("--capabilities")
            .current_dir(&lane.0)
            .stdin(Stdio::null())
            .output()?;
        assert!(output.status.success(), "capability command failed");
        assert_eq!(output.stderr, [0_u8; 0]);
        assert_eq!(std::str::from_utf8(&output.stdout)?.lines().count(), 1);
        assert!(output.stdout.ends_with(b"\n"));
        assert!(output.stdout.len() < 4096);
        let actual: Value = serde_json::from_slice(&output.stdout)?;
        let mut expected: Value =
            serde_json::from_str(include_str!("fixtures/runtime-capabilities-v1.json"))?;
        expected["runtime"]["version"] = json!(env!("CARGO_PKG_VERSION"));
        expected["runtime"]["target"]["architecture"] = json!(std::env::consts::ARCH);
        expected["runtime"]["target"]["os"] = json!(std::env::consts::OS);
        expected["runtime"]["target"]["environment"] = json!(if cfg!(target_env = "msvc") {
            "msvc"
        } else if cfg!(target_env = "gnu") {
            "gnu"
        } else if cfg!(target_env = "musl") {
            "musl"
        } else {
            "unknown"
        });
        expected["cpu"]["neon_build_eligible"] =
            json!(MatvecScratch::default().diagnostics().neon_build_eligible);
        assert_eq!(actual, expected);

        for arguments in [
            vec!["--capabilities", "--capabilities"],
            vec!["--capabilities", "--help"],
            vec!["--capabilities", "--model", "private-unopened-model.gguf"],
            vec!["--model", "private-unopened-model.gguf", "--capabilities"],
        ] {
            let rejected = Command::new(env!("CARGO_BIN_EXE_prism-inference"))
                .args(arguments)
                .current_dir(&lane.0)
                .stdin(Stdio::null())
                .output()?;
            assert!(!rejected.status.success());
            assert_eq!(rejected.stdout, [0_u8; 0]);
            let error = std::str::from_utf8(&rejected.stderr)?;
            assert!(error.contains("unknown or repeated option"));
            assert!(!error.contains("private-unopened-model.gguf"));
        }
        Ok(())
    }
}
