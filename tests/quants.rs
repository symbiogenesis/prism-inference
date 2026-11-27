//! Quantization codec, row-quantizer, and upstream differential integration tests.
#![forbid(unsafe_code)]

#[path = "quants/codec.rs"]
mod codec;
#[path = "quants/fp4.rs"]
mod fp4;
#[path = "quants/iq1.rs"]
mod iq1;
#[path = "quants/iq2.rs"]
mod iq2;
#[path = "quants/iq3.rs"]
mod iq3;
#[path = "quants/iq_encoders.rs"]
mod iq_encoders;
#[path = "quants/iquants.rs"]
mod iquants;
#[path = "quants/kquants.rs"]
mod kquants;
#[path = "quants/packed_codec.rs"]
mod packed_codec;
#[path = "quants/quantization_policy.rs"]
mod quantization_policy;
#[path = "quants/row_quantizers.rs"]
mod row_quantizers;
#[path = "quants/signquants.rs"]
mod signquants;
#[path = "quants/tquants.rs"]
mod tquants;
