//! Production CPU pair/triple consumers compared with sequential projections.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use bonsai_gguf::MetadataValue;

use super::*;
use crate::model::Model;

fn packed(name: &str, rows: usize) -> Tensor {
    let symbol = name.bytes().fold(rows.to_le_bytes()[0], |sum, byte| {
        sum.wrapping_mul(31).wrapping_add(byte)
    });
    let mut block = [symbol; 34];
    block[..2].copy_from_slice(&0x3555_u16.to_le_bytes());
    Tensor {
        name: name.into(),
        dimensions: vec![256, rows],
        kind: 142,
        data: Arc::from(block.repeat(2 * rows)),
    }
}

fn factor(name: String, dimensions: Vec<usize>, value: f32) -> Tensor {
    let count = dimensions.iter().product::<usize>();
    Tensor {
        name,
        dimensions,
        kind: 0,
        data: Arc::from(value.to_le_bytes().repeat(count)),
    }
}

struct Fixture {
    geometry: Geometry,
    tensors: [Tensor; 3],
    prior: Tensor,
    transforms: Transforms,
    adapter: RankOneAdapter,
}

impl Fixture {
    fn new(targeted: bool) -> Result<Self> {
        let target = "blk.0.ffn_down.weight";
        let tensors = [
            packed("blk.0.ffn_gate.weight", 13),
            packed(
                if targeted {
                    target
                } else {
                    "blk.0.ffn_up.weight"
                },
                9,
            ),
            packed("blk.0.attn_q.weight", 5),
        ];
        let prior = packed("blk.1.ffn_up.weight", 5);
        let base = Model {
            metadata: BTreeMap::from([(
                "general.architecture".into(),
                MetadataValue::from("qwen35"),
            )]),
            tensors: BTreeMap::from([(target.into(), packed(target, 9))]),
            identity: "cpu-sibling-adapter-base".into(),
        };
        let a = factor(format!("{target}.lora_a"), vec![256, 1], 0.03125);
        let b = factor(format!("{target}.lora_b"), vec![1, 9], -0.125);
        let factors = Model {
            metadata: BTreeMap::from([
                ("general.architecture".into(), MetadataValue::from("qwen35")),
                ("general.type".into(), MetadataValue::from("adapter")),
                ("adapter.type".into(), MetadataValue::from("lora")),
                ("adapter.lora_alpha".into(), MetadataValue::from(1.0_f32)),
            ]),
            tensors: BTreeMap::from([(a.name.clone(), a), (b.name.clone(), b)]),
            identity: "cpu-sibling-adapter-factors".into(),
        };
        Ok(Self {
            geometry: Geometry {
                embedding: 256,
                feed_forward: 256,
                heads: 1,
                kv_heads: 1,
                head: 256,
                rope: 2,
                rope_sections: [1, 0, 0, 0],
                rope_base: 10_000.0,
                epsilon: 1e-6,
                ssm_head: 128,
                ssm_keys: 1,
                ssm_values: 1,
                convolution: 4,
            },
            transforms: Transforms {
                block: 256,
                weights: tensors
                    .iter()
                    .chain([&prior])
                    .map(|tensor| tensor.name.clone())
                    .collect::<BTreeSet<_>>(),
                signs: BTreeMap::from([(
                    256,
                    (0..256)
                        .map(|index| if index % 3 == 0 { -1.0 } else { 1.0 })
                        .collect(),
                )]),
                ..Transforms::default()
            },
            adapter: RankOneAdapter::from_model(factors, &base)?,
            tensors,
            prior,
        })
    }

    fn with_bf16(mut self) -> Self {
        for tensor in self.tensors.iter_mut().chain([&mut self.prior]) {
            let elements = tensor.dimensions.iter().product::<usize>();
            tensor.kind = 30;
            tensor.data = Arc::from(
                [0x3f80_u16, 0xc000, 0x8000, 0x0080, 0x3eaa, 0x0001]
                    .into_iter()
                    .cycle()
                    .take(elements)
                    .flat_map(u16::to_le_bytes)
                    .collect::<Vec<_>>(),
            );
        }
        self
    }

    fn with_ptq1(mut self) -> Result<Self> {
        for (index, tensor) in self.tensors.iter_mut().chain([&mut self.prior]).enumerate() {
            let rows = tensor.dimensions[1];
            let mut block = [[0x66, 0x19, 0x44, 0x99][index]; 34];
            block[..2].copy_from_slice(&0x3555_u16.to_le_bytes());
            tensor.kind = 143;
            tensor.data = Arc::from(bonsai_ternary::pack_ptq(&block)?.repeat(2 * rows));
        }
        Ok(self)
    }

    #[expect(
        clippy::too_many_lines,
        reason = "Keep production pair/triple invocation, scoped reuse and independent sequential oracle together."
    )]
    fn run(
        &self,
        arithmetic: Arithmetic,
        inputs: &[Vec<f32>],
        count: usize,
        reuse: bool,
        grouped: bool,
    ) -> Result<Vec<Vec<Vec<u32>>>> {
        let workers = if cfg!(feature = "parallel-rows") {
            4
        } else {
            1
        };
        let mut projector = ProjectionTile::with_backend(MatvecBackend::Auto);
        projector.set_row_parallelism(RowParallelism::new(workers)?);
        let mut projection = arithmetic.scratch(MatvecBackend::Auto);
        let mut adapter_scratch = AdapterScratch::new(
            MatvecBackend::Auto,
            matches!(arithmetic, Arithmetic::PinnedSwift),
        );
        let mut transformed = vec![Vec::new(); inputs.len()];
        let mut warm = vec![Vec::new(); inputs.len()];
        let mut previous = TileReuse::default();
        if reuse {
            previous = project_tile(
                &self.transforms,
                &self.geometry,
                arithmetic,
                Some(&self.adapter),
                1.3,
                &mut adapter_scratch,
                &mut projection,
                previous,
                &self.prior,
                inputs,
                &mut warm,
                &mut transformed,
                &mut projector,
                None,
                &|| Ok(()),
            )?;
            assert!(previous.basis.is_some());
            assert!(previous.prepared.is_some());
        }
        let mut outputs: [Vec<Vec<f32>>; 3] =
            std::array::from_fn(|_| vec![Vec::new(); inputs.len()]);
        if grouped {
            let [a, b, c] = &mut outputs;
            previous = if count == 2 {
                project_tile_pair(
                    &self.transforms,
                    &self.geometry,
                    arithmetic,
                    Some(&self.adapter),
                    1.3,
                    &mut adapter_scratch,
                    &mut projection,
                    previous,
                    [&self.tensors[0], &self.tensors[1]],
                    inputs,
                    a,
                    b,
                    &mut transformed,
                    &mut projector,
                    None,
                    &|| Ok(()),
                )?
            } else {
                project_tile_triple(
                    &self.transforms,
                    &self.geometry,
                    arithmetic,
                    Some(&self.adapter),
                    1.3,
                    &mut adapter_scratch,
                    &mut projection,
                    previous,
                    self.tensors.each_ref(),
                    inputs,
                    a,
                    b,
                    c,
                    &mut transformed,
                    &mut projector,
                    None,
                    &|| Ok(()),
                )?
            };
        } else {
            for (tensor, output) in self.tensors.iter().zip(&mut outputs).take(count) {
                previous = project_tile(
                    &self.transforms,
                    &self.geometry,
                    arithmetic,
                    Some(&self.adapter),
                    1.3,
                    &mut adapter_scratch,
                    &mut projection,
                    previous,
                    tensor,
                    inputs,
                    output,
                    &mut transformed,
                    &mut projector,
                    None,
                    &|| Ok(()),
                )?;
            }
        }
        assert!(
            previous
                .basis
                .is_some_and(|tensor| std::ptr::eq(tensor, &raw const self.tensors[count - 1]))
        );
        assert!(
            previous
                .prepared
                .is_some_and(|tensor| std::ptr::eq(tensor, &raw const self.tensors[count - 1]))
        );
        // Consume the returned proof again, ensuring it owns current prepared
        // activations rather than stale data from the prior projection.
        let _ = project_tile(
            &self.transforms,
            &self.geometry,
            arithmetic,
            Some(&self.adapter),
            1.3,
            &mut adapter_scratch,
            &mut projection,
            previous,
            &self.prior,
            inputs,
            &mut warm,
            &mut transformed,
            &mut projector,
            None,
            &|| Ok(()),
        )?;
        Ok(outputs
            .into_iter()
            .take(count)
            .chain([warm])
            .map(|group| {
                group
                    .into_iter()
                    .map(|row| row.into_iter().map(f32::to_bits).collect())
                    .collect()
            })
            .collect())
    }
}

#[test]
fn production_cpu_siblings_preserve_adapter_and_prepared_reuse() -> Result<()> {
    let inputs: Vec<Vec<f32>> = (0..3)
        .map(|token| {
            (0..256)
                .map(|index| [-0.75, -0.0, 0.25, 0.5, 1.0][(index + token) % 5])
                .collect()
        })
        .collect();
    for (targeted, bf16) in [(false, false), (true, false), (false, true), (true, true)] {
        let fixture = if bf16 {
            Fixture::new(targeted)?.with_bf16()
        } else {
            Fixture::new(targeted)?
        };
        assert!(ProjectionTile::supports_siblings(
            &fixture.tensors.each_ref()
        ));
        assert!(
            fixture
                .tensors
                .iter()
                .all(|tensor| fixture.transforms.basis(tensor)
                    == fixture.transforms.basis(&fixture.prior))
        );
        assert_eq!(
            fixture
                .adapter
                .affects_linear(&fixture.tensors[1].name, 1.3),
            targeted
        );
        assert!(
            !fixture
                .adapter
                .affects_linear(&fixture.tensors[0].name, 1.3)
        );
        assert!(
            !fixture
                .adapter
                .affects_linear(&fixture.tensors[2].name, 1.3)
        );
        for arithmetic in [Arithmetic::Source, Arithmetic::PinnedSwift] {
            for count in [2, 3] {
                for reuse in [false, true] {
                    let expected = fixture.run(arithmetic, &inputs, count, reuse, false)?;
                    let actual = fixture.run(arithmetic, &inputs, count, reuse, true)?;
                    assert_eq!(actual, expected);
                }
            }
        }
    }
    Ok(())
}

#[test]
fn production_ptq1_siblings_preserve_adapter_and_prepared_reuse() -> Result<()> {
    let inputs: Vec<Vec<f32>> = (0..9)
        .map(|token| {
            (0..256)
                .map(|index| [-0.75, -0.0, 0.25, 0.5, 1.0][(index + token) % 5])
                .collect()
        })
        .collect();
    for targeted in [false, true] {
        let fixture = Fixture::new(targeted)?.with_ptq1()?;
        assert!(ProjectionTile::supports_siblings(
            &fixture.tensors.each_ref()
        ));
        assert_eq!(
            fixture
                .adapter
                .affects_linear(&fixture.tensors[1].name, 1.3),
            targeted
        );
        for arithmetic in [Arithmetic::Source, Arithmetic::PinnedSwift] {
            for count in [2, 3] {
                for reuse in [false, true] {
                    let expected = fixture.run(arithmetic, &inputs, count, reuse, false)?;
                    let actual = fixture.run(arithmetic, &inputs, count, reuse, true)?;
                    assert_eq!(actual, expected);
                }
            }
        }
    }
    Ok(())
}
