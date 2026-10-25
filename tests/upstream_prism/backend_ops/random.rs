// SPDX-License-Identifier: MIT
//! Source PRNGs and Windows STL distribution mappings.

#[derive(Clone, Debug)]
pub(super) struct Mt19937 {
    state: [u32; 624],
    index: usize,
    pub(super) draws: u64,
}

impl Mt19937 {
    pub(super) fn new(seed: u32) -> Self {
        let mut state = [0_u32; 624];
        state[0] = seed;
        for index in 1..624 {
            let previous = state[index - 1];
            state[index] = 1_812_433_253_u32
                .wrapping_mul(previous ^ (previous >> 30))
                .wrapping_add(index as u32);
        }
        Self {
            state,
            index: 624,
            draws: 0,
        }
    }

    pub(super) fn next(&mut self) -> u32 {
        if self.index == 624 {
            for index in 0..624 {
                let value = (self.state[index] & 0x8000_0000)
                    | (self.state[(index + 1) % 624] & 0x7fff_ffff);
                self.state[index] = self.state[(index + 397) % 624]
                    ^ (value >> 1)
                    ^ if value & 1 == 0 { 0 } else { 0x9908_b0df };
            }
            self.index = 0;
        }
        let mut value = self.state[self.index];
        self.index += 1;
        self.draws += 1;
        value ^= value >> 11;
        value ^= (value << 7) & 0x9d2c_5680;
        value ^= (value << 15) & 0xefc6_0000;
        value ^ (value >> 18)
    }

    pub(super) fn uniform_below(&mut self, exclusive: u64) -> Result<u64, String> {
        if exclusive == 0 {
            return Err("source uniform integer interval is empty".into());
        }
        // MSVC _Rng_from_urng_v2 concatenates another draw only when the interval
        // exceeds the URNG's 32-bit range, even for a 64-bit result type.
        if let Ok(range) = u32::try_from(exclusive) {
            let threshold = range.wrapping_neg() % range;
            loop {
                let product = u64::from(self.next()) * u64::from(range);
                if product as u32 >= threshold {
                    return Ok(product >> 32);
                }
            }
        }
        if exclusive == 1_u64 << 32 {
            return Ok(u64::from(self.next()));
        }
        let threshold = exclusive.wrapping_neg() % exclusive;
        loop {
            let draw = (u64::from(self.next()) << 32) | u64::from(self.next());
            let product = u128::from(draw) * u128::from(exclusive);
            if product as u64 >= threshold {
                return Ok((product >> 64) as u64);
            }
        }
    }

    pub(super) fn uniform_integer(&mut self, minimum: i64, maximum: i64) -> Result<i64, String> {
        let width = i128::from(maximum) - i128::from(minimum) + 1;
        let width =
            u64::try_from(width).map_err(|_| "source uniform integer interval exceeds u64")?;
        let draw = self.uniform_below(width)?;
        Ok((i128::from(minimum) + i128::from(draw)) as i64)
    }

    pub(super) fn uniform_real(&mut self, minimum: f32, maximum: f32) -> f32 {
        // MSVC generate_canonical<float,24> for the full-width MT19937 URNG.
        let canonical = (self.next() >> 8) as f32 * (1.0_f32 / 16_777_216.0);
        canonical * (maximum - minimum) + minimum
    }
}

#[derive(Clone, Debug)]
pub(super) struct CrtRandom {
    state: u32,
    pub(super) draws: u64,
}

impl CrtRandom {
    pub(super) const fn new(seed: u32) -> Self {
        Self {
            state: seed,
            draws: 0,
        }
    }

    pub(super) const fn next(&mut self) -> i32 {
        self.state = self.state.wrapping_mul(214_013).wrapping_add(2_531_011);
        self.draws += 1;
        ((self.state >> 16) & 0x7fff) as i32
    }

    /// Jump the same affine recurrence without evaluating preceding test cases.
    pub(super) const fn skip(&mut self, count: u64) {
        let (mut multiplier, mut addition) = (214_013_u32, 2_531_011_u32);
        let (mut accumulated_multiplier, mut accumulated_addition) = (1_u32, 0_u32);
        let mut remaining = count;
        while remaining != 0 {
            if remaining & 1 != 0 {
                accumulated_multiplier = accumulated_multiplier.wrapping_mul(multiplier);
                accumulated_addition = accumulated_addition
                    .wrapping_mul(multiplier)
                    .wrapping_add(addition);
            }
            addition = multiplier.wrapping_add(1).wrapping_mul(addition);
            multiplier = multiplier.wrapping_mul(multiplier);
            remaining >>= 1;
        }
        self.state = accumulated_multiplier
            .wrapping_mul(self.state)
            .wrapping_add(accumulated_addition);
        self.draws += count;
    }
}

#[derive(Clone, Debug)]
pub(super) struct SeedLedger {
    state: u64,
    pub(super) seeds: Vec<u32>,
}

impl SeedLedger {
    pub(super) const fn new(seed: u64) -> Self {
        Self {
            state: seed,
            seeds: Vec::new(),
        }
    }

    pub(super) fn next(&mut self) -> u32 {
        // A recorded deterministic replacement only for source random_device inputs.
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        let value = (value ^ (value >> 31)) as u32;
        self.seeds.push(value);
        value
    }
}
