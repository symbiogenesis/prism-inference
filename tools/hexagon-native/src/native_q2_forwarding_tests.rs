//! Model-free behavior of the same forwarding implementation used by the owned facade.

use super::{
    Error, NativeQ2Submitter, NativeQ2ValidatedOutputs, NativeQ2WeightHandle,
    NativeQ2WeightRegistry,
};

struct RecordingSubmitter {
    registry: NativeQ2WeightRegistry,
    calls: Vec<&'static str>,
    fail: bool,
}

impl RecordingSubmitter {
    fn complete(
        &mut self,
        method: &'static str,
        output: &mut [f32],
        value: f32,
    ) -> Result<(), Error> {
        self.calls.push(method);
        if self.fail {
            output.fill(f32::NAN);
            return Err(Error::Contract);
        }
        output.fill(value);
        Ok(())
    }
}

impl NativeQ2Submitter for RecordingSubmitter {
    fn remaining_operations(&self) -> Option<u64> {
        Some(321)
    }

    fn residency_capacity_bytes(&self) -> usize {
        self.registry.remaining_bytes()
    }

    fn supports_in_place_weight_preparation(&self) -> bool {
        !self.fail
    }

    fn prepare_weights_in_place(
        &mut self,
        columns: usize,
        rows: usize,
        initialize: &mut dyn FnMut(&mut [u8]) -> Result<(), Error>,
    ) -> Result<Option<NativeQ2WeightHandle>, Error> {
        self.calls.push("prepare-in-place");
        let mut bytes = [0_u8; 1152];
        initialize(&mut bytes)?;
        self.registry.prepare(columns, rows, &bytes)
    }

    fn submit_resident_multi8(
        &mut self,
        _handles: [&NativeQ2WeightHandle; 8],
        activations: [&[f32]; 8],
        outputs: [&mut [f32]; 8],
    ) -> Result<(), Error> {
        self.calls.push("multi8");
        for (output, activation) in outputs.into_iter().zip(activations) {
            output.fill(activation[0]);
        }
        if self.fail {
            return Err(Error::Contract);
        }
        Ok(())
    }

    fn submit_resident_ffn_swiglu(
        &mut self,
        _gate_handle: &NativeQ2WeightHandle,
        _up_handle: &NativeQ2WeightHandle,
        activation: &[f32],
        output: &mut [f32],
    ) -> Result<(), Error> {
        self.complete("swiglu", output, activation[0])
    }

    fn submit_resident_ffn_swiglu_multi2(
        &mut self,
        _gate_handle: &NativeQ2WeightHandle,
        _up_handle: &NativeQ2WeightHandle,
        activations: [&[f32]; 2],
        outputs: [&mut [f32]; 2],
    ) -> Result<(), Error> {
        self.calls.push("swiglu-multi2");
        for (output, activation) in outputs.into_iter().zip(activations) {
            output.fill(activation[0]);
        }
        if self.fail {
            return Err(Error::Contract);
        }
        Ok(())
    }

    fn submit_resident_validated<'a>(
        &mut self,
        _handle: &NativeQ2WeightHandle,
        activation: &[f32],
        output: &'a mut [f32],
    ) -> Result<NativeQ2ValidatedOutputs<'a, 1>, Error> {
        self.complete("validated", output, activation[0])?;
        NativeQ2ValidatedOutputs::validate([output])
    }

    fn submit_resident_batch2_validated<'a>(
        &mut self,
        _handles: [&NativeQ2WeightHandle; 2],
        activation: &[f32],
        outputs: [&'a mut [f32]; 2],
    ) -> Result<NativeQ2ValidatedOutputs<'a, 2>, Error> {
        let [o0, o1] = outputs;
        self.complete("validated-batch2", o0, activation[0])?;
        o1.fill(activation[1]);
        NativeQ2ValidatedOutputs::validate([o0, o1])
    }

    fn submit_resident_batch3_validated<'a>(
        &mut self,
        _handles: [&NativeQ2WeightHandle; 3],
        activation: &[f32],
        outputs: [&'a mut [f32]; 3],
    ) -> Result<NativeQ2ValidatedOutputs<'a, 3>, Error> {
        let [o0, o1, o2] = outputs;
        self.complete("validated-batch3", o0, activation[0])?;
        o1.fill(activation[1]);
        o2.fill(activation[2]);
        NativeQ2ValidatedOutputs::validate([o0, o1, o2])
    }

    fn submit(
        &mut self,
        _columns: usize,
        _rows: usize,
        _weights: &[u8],
        _activation: &[f32],
        _output: &mut [f32],
    ) -> Result<(), Error> {
        Err(Error::Contract)
    }
}

struct RecordingSession(RecordingSubmitter);

macro_rules! dispatch_recording_session {
    ($session:ident, $method:ident($($argument:expr),*)) => {
        $session.0.$method($($argument),*)
    };
}

impl_native_q2_session_forwarding!(RecordingSession, dispatch_recording_session);

fn fixture() -> Result<(RecordingSession, NativeQ2WeightHandle), Error> {
    let mut registry = NativeQ2WeightRegistry::new(4096)?;
    let handle = registry
        .prepare(128, 32, &[0_u8; 1152])?
        .ok_or(Error::Contract)?;
    Ok((
        RecordingSession(RecordingSubmitter {
            registry,
            calls: Vec::new(),
            fail: false,
        }),
        handle,
    ))
}

#[test]
fn owned_session_forwards_direct_resident_initialization_once() -> Result<(), Error> {
    let (mut session, _) = fixture()?;
    assert!(session.supports_in_place_weight_preparation());
    assert_eq!(session.remaining_operations(), Some(321));
    let before = session.residency_capacity_bytes();
    let mut callbacks = 0;
    let handle = session
        .prepare_weights_in_place(128, 32, &mut |bytes| {
            callbacks += 1;
            assert_eq!(bytes.len(), 1152);
            bytes.fill(0x55);
            Ok(())
        })?
        .ok_or(Error::Contract)?;
    assert_eq!(callbacks, 1);
    assert_eq!(handle.packed_bytes(), 1152);
    assert_eq!(session.residency_capacity_bytes(), before - 1152);
    assert_eq!(session.0.calls, ["prepare-in-place"]);
    let used = session.0.registry.used_bytes();
    assert!(
        session
            .prepare_weights_in_place(128, 32, &mut |_| Err(Error::Contract))
            .is_err()
    );
    assert_eq!(session.0.registry.used_bytes(), used);
    Ok(())
}

#[test]
fn owned_session_keeps_eight_operations_in_one_native_call() -> Result<(), Error> {
    let (mut session, handle) = fixture()?;
    let activations = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0].map(|value| [value; 128]);
    let mut outputs = [[f32::NAN; 32]; 8];
    session.submit_resident_multi8(
        [&handle; 8],
        activations.each_ref().map(|row| row.as_slice()),
        outputs.each_mut().map(|row| row.as_mut_slice()),
    )?;
    assert_eq!(session.0.calls, ["multi8"]);
    for (output, activation) in outputs.iter().zip(&activations) {
        assert_eq!(*output, [activation[0]; 32]);
    }
    session.0.fail = true;
    assert!(
        session
            .submit_resident_multi8(
                [&handle; 8],
                activations.each_ref().map(|row| row.as_slice()),
                outputs.each_mut().map(|row| row.as_mut_slice())
            )
            .is_err()
    );
    assert_eq!(session.0.calls, ["multi8", "multi8"]);
    // A forwarding facade must preserve even the submitter's failure outputs;
    // it must not replay a partial packet or replace its publication policy.
    for (output, activation) in outputs.iter().zip(&activations) {
        assert_eq!(*output, [activation[0]; 32]);
    }
    Ok(())
}

#[test]
fn owned_session_preserves_fused_swiglu_calls_and_rejection() -> Result<(), Error> {
    let (mut session, handle) = fixture()?;
    let activation = [3.0; 128];
    let mut output = [f32::NAN; 32];
    session.submit_resident_ffn_swiglu(&handle, &handle, &activation, &mut output)?;
    assert_eq!(output, [3.0; 32]);
    let mut pair = [[f32::NAN; 32]; 2];
    session.submit_resident_ffn_swiglu_multi2(
        &handle,
        &handle,
        [&activation; 2],
        pair.each_mut().map(|row| row.as_mut_slice()),
    )?;
    assert_eq!(pair, [[3.0; 32]; 2]);
    assert_eq!(session.0.calls, ["swiglu", "swiglu-multi2"]);
    session.0.fail = true;
    assert!(
        session
            .submit_resident_ffn_swiglu(&handle, &handle, &activation, &mut output)
            .is_err()
    );
    assert!(output.iter().all(|value| value.is_nan()));
    pair.fill([99.0; 32]);
    assert!(
        session
            .submit_resident_ffn_swiglu_multi2(
                &handle,
                &handle,
                [&activation; 2],
                pair.each_mut().map(|row| row.as_mut_slice())
            )
            .is_err()
    );
    assert_eq!(pair, [[3.0; 32]; 2]);
    assert_eq!(
        session.0.calls,
        ["swiglu", "swiglu-multi2", "swiglu", "swiglu-multi2"]
    );
    // Forwarding does not manufacture support for the unfinished full-chain backend.
    assert!(
        session
            .submit_resident_ffn_chain([&handle; 3], &activation, None, None, &mut output)
            .is_err()
    );
    assert!(output.iter().all(|value| value.is_nan()));
    Ok(())
}

#[test]
fn owned_session_preserves_borrowed_output_evidence_and_errors() -> Result<(), Error> {
    let (mut session, handle) = fixture()?;
    let activation = [-0.0, f32::from_bits(1), 1.0];
    let mut output = [f32::NAN; 32];
    let destination = (output.as_ptr(), output.len());
    session
        .submit_resident_validated(&handle, &activation, &mut output)?
        .validate_destinations([destination])?;
    assert!(
        output
            .iter()
            .all(|value| value.to_bits() == (-0.0_f32).to_bits())
    );
    let mut pair = [[f32::NAN; 32]; 2];
    let destinations = pair.each_ref().map(|row| (row.as_ptr(), row.len()));
    session
        .submit_resident_batch2_validated(
            [&handle; 2],
            &activation,
            pair.each_mut().map(|row| row.as_mut_slice()),
        )?
        .validate_destinations(destinations)?;
    let mut triple = [[f32::NAN; 32]; 3];
    let destinations = triple.each_ref().map(|row| (row.as_ptr(), row.len()));
    session
        .submit_resident_batch3_validated(
            [&handle; 3],
            &activation,
            triple.each_mut().map(|row| row.as_mut_slice()),
        )?
        .validate_destinations(destinations)?;
    assert_eq!(
        session.0.calls,
        ["validated", "validated-batch2", "validated-batch3"]
    );
    assert!(pair[1].iter().all(|value| value.to_bits() == 1));
    assert_eq!(triple[2], [1.0; 32]);
    session.0.fail = true;
    assert!(!session.supports_in_place_weight_preparation());
    assert!(
        session
            .submit_resident_validated(&handle, &activation, &mut output)
            .is_err()
    );
    assert!(output.iter().all(|value| value.is_nan()));
    Ok(())
}
