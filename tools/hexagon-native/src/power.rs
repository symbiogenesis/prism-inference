//! Read-only owner-workload power snapshots, without policy changes.

use crate::Error;

/// One Windows power-source and battery-saver observation.
///
/// Equal endpoints do not establish uninterrupted stability or energy usage.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PowerStatus {
    /// AC connected, with the documented Windows unknown sentinel preserved.
    pub ac_connected: Option<bool>,
    /// Current Windows battery-saver flag.
    pub battery_saver: bool,
    /// Remaining battery percentage, preserving unknown rather than guessing.
    pub battery_percent: Option<u8>,
}

/// Observe current power conditions without changing any process/device policy.
///
/// # Errors
/// Rejects unsupported platforms, failed queries and undocumented native values.
#[cfg_attr(
    not(all(windows, target_arch = "aarch64")),
    allow(
        clippy::missing_const_for_fn,
        reason = "Supported native targets perform a read-only OS call."
    )
)]
pub fn power_status() -> Result<PowerStatus, Error> {
    #[cfg(all(windows, target_arch = "aarch64"))]
    {
        crate::windows::power_status()
    }
    #[cfg(not(all(windows, target_arch = "aarch64")))]
    {
        Err(Error::UnsupportedPlatform)
    }
}

#[cfg_attr(
    not(all(windows, target_arch = "aarch64")),
    allow(
        dead_code,
        reason = "Native decoder also has model-free hosted controls."
    )
)]
#[allow(
    clippy::redundant_pub_crate,
    reason = "Only the audited native sibling and model-free controls consume the decoder."
)]
pub(super) const fn decode(ac: u8, percent: u8, saver: u8) -> Result<PowerStatus, Error> {
    let ac_connected = match ac {
        0 => Some(false),
        1 => Some(true),
        255 => None,
        _ => return Err(Error::Contract),
    };
    let battery_percent = match percent {
        0..=100 => Some(percent),
        255 => None,
        _ => return Err(Error::Contract),
    };
    let battery_saver = match saver {
        0 => false,
        1 => true,
        _ => return Err(Error::Contract),
    };
    Ok(PowerStatus {
        ac_connected,
        battery_saver,
        battery_percent,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshots_preserve_unknown_and_refuse_undocumented_fields() -> Result<(), Error> {
        assert_eq!(
            decode(255, 255, 0)?,
            PowerStatus {
                ac_connected: None,
                battery_percent: None,
                battery_saver: false
            }
        );
        assert_eq!(
            decode(1, 100, 1)?,
            PowerStatus {
                ac_connected: Some(true),
                battery_percent: Some(100),
                battery_saver: true
            }
        );
        for (ac, percent, saver) in [(2, 100, 0), (1, 101, 0), (1, 100, 2)] {
            assert!(decode(ac, percent, saver).is_err());
        }
        Ok(())
    }
}
