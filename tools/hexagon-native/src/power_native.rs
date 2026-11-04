//! One synchronous, read-only Kernel32 power-status query.

use crate::{Error, PowerStatus};

#[repr(C)]
#[derive(Default)]
struct SystemPowerStatus {
    ac_line_status: u8,
    battery_flag: u8,
    battery_life_percent: u8,
    system_status_flag: u8,
    battery_life_time: u32,
    battery_full_life_time: u32,
}

const _: [(); 12] = [(); size_of::<SystemPowerStatus>()];

#[link(name = "kernel32")]
unsafe extern "system" {
    fn GetSystemPowerStatus(status: *mut SystemPowerStatus) -> i32;
}

pub(super) fn query() -> Result<PowerStatus, Error> {
    let mut status = SystemPowerStatus::default();
    // SAFETY: Kernel32 writes the documented 12-byte repr(C) structure during
    // this synchronous call and retains no pointer. This changes no policy.
    if unsafe { GetSystemPowerStatus(&raw mut status) } == 0 {
        return Err(Error::File(std::io::Error::last_os_error()));
    }
    crate::power::decode(
        status.ac_line_status,
        status.battery_life_percent,
        status.system_status_flag,
    )
}
