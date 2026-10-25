//! Strict intake deadlines for one private HTTP request.
#![allow(
    clippy::redundant_pub_crate,
    reason = "Explicit parent scope keeps these implementation helpers private."
)]
use std::time::Duration;
pub(super) const HEADER_TIMEOUT: Duration = Duration::from_secs(5);
pub(super) const INTAKE_TIMEOUT: Duration = Duration::from_secs(30);
