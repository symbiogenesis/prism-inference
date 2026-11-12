//! Binary32-rounded symmetric Gaussian Lloyd-Max reconstruction levels.
//!
//! These numerical constants specify the codec, not third-party source. For a
//! standard normal variable, each decision boundary is the midpoint of adjacent
//! centroids and each centroid is the conditional mean of its interval:
//! `(phi(left) - phi(right)) / (Phi(right) - Phi(left))`. The symmetric eight- and
//! sixteen-level fixed points are rounded once to binary32 below. The final
//! eight slots of the three-bit table are zero and are never encoded or read.

pub(super) const THREE: [f32; 16] = [
    -2.151_945_6,
    -1.343_909_3,
    -0.756_005_3,
    -0.245_094_17,
    0.245_094_17,
    0.756_005_3,
    1.343_909_3,
    2.151_945_6,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
];

pub(super) const FOUR: [f32; 16] = [
    -2.732_589,
    -2.069_017_2,
    -1.618_046_4,
    -1.256_231_2,
    -0.942_340_25,
    -0.656_758_96,
    -0.388_048_3,
    -0.128_395_04,
    0.128_395_04,
    0.388_048_3,
    0.656_758_96,
    0.942_340_25,
    1.256_231_2,
    1.618_046_4,
    2.069_017_2,
    2.732_589,
];
