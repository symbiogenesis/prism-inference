//! Safe, input-only reproduction of the Windows C++ distributions used by sources.
//!
//! MT19937 is standardized; the bounded, canonical and polar-normal choices here
//! follow the installed MSVC STL. No engine observations or expected results enter
//! these generators. The source float operations deliberately retain their order.

use super::{Result, invalid};

/// Standard 32-bit MT19937 state.
#[derive(Clone)]
pub(super) struct Random {
    state: [u32; 624],
    index: usize,
}

impl Random {
    /// Seeds the standard engine with one unsigned 32-bit integer.
    pub(super) fn new(seed: u32) -> Self {
        let mut state = [0; 624];
        state[0] = seed;
        for (index, increment) in (1..624).zip(1_u32..) {
            state[index] = 1_812_433_253_u32
                .wrapping_mul(state[index - 1] ^ (state[index - 1] >> 30))
                .wrapping_add(increment);
        }
        Self { state, index: 624 }
    }

    /// Draws the next standardized 32-bit engine value.
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
        value ^= value >> 11;
        value ^= (value << 7) & 0x9d2c_5680;
        value ^= (value << 15) & 0xefc6_0000;
        value ^ (value >> 18)
    }

    /// Reproduces MSVC's unbiased distribution on `[0, bound)`.
    pub(super) fn bounded32(&mut self, bound: u32) -> Result<u32> {
        if bound == 0 {
            return Err(invalid("Windows random bound must be positive"));
        }
        let threshold = bound.wrapping_neg() % bound;
        loop {
            let product = u64::from(self.next()) * u64::from(bound);
            let low = u32::from_le_bytes(product.to_le_bytes()[..4].try_into()?);
            if low >= threshold {
                return u32::try_from(product >> 32).map_err(Into::into);
            }
        }
    }

    #[allow(
        clippy::as_conversions,
        clippy::cast_precision_loss,
        reason = "The top24 unsigned bits fit exactly in IEEE binary32, matching MSVC generate_canonical<float,24>."
    )]
    fn canonical_f32(&mut self) -> f32 {
        ((self.next() >> 8) as f32) * (1.0 / 16_777_216.0)
    }

    /// Reproduces MSVC's two-draw, 53-bit canonical double distribution.
    #[allow(
        clippy::as_conversions,
        clippy::cast_precision_loss,
        clippy::suboptimal_flops,
        reason = "The 53-bit integer is exact in binary64; separate multiply/add preserve the source distribution's rounding."
    )]
    pub(super) fn uniform_f64(&mut self, low: f64, high: f64) -> f64 {
        let bits = u64::from(self.next() >> 11) + (u64::from(self.next()) << 21);
        ((bits as f64) * (1.0 / 9_007_199_254_740_992.0)) * (high - low) + low
    }
    /// Reproduces the source `uniform_real_distribution<float>` evaluation order.
    #[allow(
        clippy::suboptimal_flops,
        reason = "Separate subtract, multiply and add preserve the C++ distribution's binary32 roundings."
    )]
    pub(super) fn uniform_f32(&mut self, low: f32, high: f32) -> f32 {
        self.canonical_f32() * (high - low) + low
    }
}

/// Stateful MSVC polar-normal distribution, including its cached second value.
pub(super) struct Normal {
    mean: f32,
    sigma: f32,
    cached: Option<f32>,
}

impl Normal {
    /// Creates the source distribution; callers supply the source's valid sigma.
    pub(super) const fn new(mean: f32, sigma: f32) -> Self {
        Self {
            mean,
            sigma,
            cached: None,
        }
    }

    /// Draws one sample with the installed STL's rejection and scaling choices.
    #[allow(
        clippy::suboptimal_flops,
        reason = "The installed STL explicitly rounds these multiply/add steps; fused arithmetic changes source inputs."
    )]
    pub(super) fn sample(&mut self, random: &mut Random) -> f32 {
        let value = if let Some(cached) = self.cached.take() {
            cached
        } else {
            let (mut x, mut y, mut squared) = loop {
                let x = 2.0 * random.canonical_f32() - 1.0;
                let y = 2.0 * random.canonical_f32() - 1.0;
                let squared = x * x + y * y;
                if squared < 1.0 && x != 0.0 && y != 0.0 {
                    break (x, y, squared);
                }
            };
            let log = if squared > 1e-4 {
                squared.ln()
            } else {
                // Canonical draws are multiples of 2^-24, so accepted nonzero
                // x/y are normal floats and their binary exponents are -23..0.
                let exponent =
                    i32::from(((x.abs().max(y.abs()).to_bits() >> 23) & 255).to_le_bytes()[0])
                        - 127;
                let scale = 2.0_f32.powi(-exponent);
                x *= scale;
                y *= scale;
                squared = x * x + y * y;
                squared.ln() + exponent_as_f32(exponent) * (std::f32::consts::LN_2 * 2.0)
            };
            let factor = (-2.0 * log / squared).sqrt();
            self.cached = Some(factor * y);
            factor * x
        };
        value * self.sigma + self.mean
    }
}

#[allow(
    clippy::as_conversions,
    clippy::cast_precision_loss,
    reason = "The polar algorithm's bounded exponent -23..0 converts exactly to binary32."
)]
const fn exponent_as_f32(exponent: i32) -> f32 {
    exponent as f32
}

/// MSVC's 64-bit `std::hash<string>` byte hash used to seed generated inputs.
pub(super) fn hash_name64(name: &str) -> u64 {
    name.bytes()
        .fold(14_695_981_039_346_656_037, |state, byte| {
            (state ^ u64::from(byte)).wrapping_mul(1_099_511_628_211)
        })
}
