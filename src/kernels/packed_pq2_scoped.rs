//! One borrowed scale-validation scope per projection or sibling group.

use super::Activations;
use crate::{Result, invalid};

pub(in crate::kernels) struct Consumer<'a> {
    input: &'a Activations,
    dotprod: Option<bonsai_neon_dot::Pq2Prepared<'a>>,
}

impl<'a> Consumer<'a> {
    pub(in crate::kernels) fn new(input: &'a Activations) -> Result<Self> {
        let dotprod = input
            .dotprod
            .map(|detected| {
                detected
                    .prepare_scaled(input)
                    .ok_or_else(|| invalid("nonfinite tensor or activation value"))
            })
            .transpose()?;
        Ok(Self { input, dotprod })
    }

    pub(in crate::kernels) fn dot_row<const FUSED: bool>(&self, bytes: &[u8]) -> Result<f32> {
        if let Some(dotprod) = self.dotprod {
            self.check_extents([bytes])?;
            return dotprod
                .dot_scaled_rows::<1, FUSED>([bytes])
                .ok_or_else(|| invalid("nonfinite tensor or activation value"))
                .map(|output| output[0]);
        }
        super::dot_row::<FUSED>(bytes, self.input)
    }

    pub(in crate::kernels) fn dot_four_rows<const FUSED: bool>(
        &self,
        rows: [&[u8]; 4],
    ) -> Result<[f32; 4]> {
        if let Some(dotprod) = self.dotprod {
            self.check_extents(rows)?;
            return dotprod
                .dot_scaled_rows::<4, FUSED>(rows)
                .ok_or_else(|| invalid("nonfinite tensor or activation value"));
        }
        super::dot_four_rows::<FUSED>(rows, self.input)
    }

    pub(in crate::kernels) fn dot_eight_rows<const FUSED: bool>(
        &self,
        rows: [&[u8]; 8],
    ) -> Result<[f32; 8]> {
        if let Some(dotprod) = self.dotprod {
            self.check_extents(rows)?;
            return dotprod
                .dot_scaled_rows::<8, FUSED>(rows)
                .ok_or_else(|| invalid("nonfinite tensor or activation value"));
        }
        super::dot_eight_rows::<FUSED>(rows, self.input)
    }

    fn check_extents<const ROWS: usize>(&self, rows: [&[u8]; ROWS]) -> Result<()> {
        if self.input.is_empty()
            || rows.iter().any(|bytes| {
                bytes.len() / 34 != self.input.len() || !bytes.len().is_multiple_of(34)
            })
        {
            return Err(invalid("incompatible packed consumption activation extent"));
        }
        Ok(())
    }
}
