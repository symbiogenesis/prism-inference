//! Total private request deadline.
#![allow(
    clippy::redundant_pub_crate,
    reason = "Explicit parent scope keeps these implementation helpers private."
)]
use std::time::Duration;
pub(super) const REQUEST_TIMEOUT: Duration = Duration::from_mins(10);
