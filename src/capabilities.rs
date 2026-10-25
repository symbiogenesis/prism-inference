//! Versioned no-model inventory of compiled runtime support.
//!
//! Availability and graph contract identifiers describe source/build support.
//! They do not establish artifact, hardware, performance or daily qualification.
//! NEON eligibility comes from the production kernel dispatch diagnostics.

use serde::Serialize;

use crate::graph::{QWEN35_GRAPH_CONTRACT, SWIFT_GRAPH_CONTRACT};
use crate::kernels::MatvecScratch;

/// Version-one runtime support descriptor for discovery and compatibility checks.
///
/// Serialize this value as JSON. Consumers must reject unsupported schema
/// versions and require independent artifact/evidence/resource admission before
/// selecting a profile. Graph suffixes are not portable checkpoint identities.
#[derive(Debug, Serialize)]
pub struct RuntimeCapabilities {
    schema_version: u32,
    kind: &'static str,
    runtime: RuntimeBuild,
    cpu: CpuSupport,
    backends: AcceleratorSupport,
    supported_contracts: [&'static str; 2],
    qualification: &'static str,
}

#[derive(Debug, Serialize)]
struct RuntimeBuild {
    name: &'static str,
    version: &'static str,
    target: BuildTarget,
}

#[derive(Debug, Serialize)]
struct BuildTarget {
    architecture: &'static str,
    os: &'static str,
    environment: &'static str,
}

#[derive(Debug, Serialize)]
struct CpuSupport {
    scalar: bool,
    neon_build_eligible: bool,
}

#[derive(Debug, Serialize)]
struct AcceleratorSupport {
    vulkan: &'static str,
    hexagon: &'static str,
}

impl RuntimeCapabilities {
    /// Inventory this executable's compiled support without loading a model.
    ///
    /// No device execution, performance capture or hardware probe occurs.
    /// Build target facts describe this process, including under emulation.
    #[must_use]
    pub fn current() -> Self {
        let environment = if cfg!(target_env = "msvc") {
            "msvc"
        } else if cfg!(target_env = "gnu") {
            "gnu"
        } else if cfg!(target_env = "musl") {
            "musl"
        } else {
            "unknown"
        };
        Self {
            schema_version: 1,
            kind: "prism-runtime-capabilities",
            runtime: RuntimeBuild {
                name: "prism-inference",
                version: env!("CARGO_PKG_VERSION"),
                target: BuildTarget {
                    architecture: std::env::consts::ARCH,
                    os: std::env::consts::OS,
                    environment,
                },
            },
            cpu: CpuSupport {
                scalar: true,
                neon_build_eligible: MatvecScratch::default().diagnostics().neon_build_eligible,
            },
            backends: AcceleratorSupport {
                vulkan: "unavailable",
                hexagon: "unavailable",
            },
            supported_contracts: [QWEN35_GRAPH_CONTRACT, SWIFT_GRAPH_CONTRACT],
            qualification: "unqualified",
        }
    }
}
