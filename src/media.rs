//! Image tiling and temporal grouping for multimodal input preparation.
//!
//! The LFM2 tiling and adjacent-frame contracts follow the MIT-licensed Prism
//! source pinned by `tools/prism/source.lock.json`. These operations describe
//! input layout; they do not imply a vision/audio model implementation.

use crate::{Result, invalid};

pub mod chunks;

/// A multimodal input part with independently owned metadata.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Part {
    /// Ordinary text, which separates temporal image groups.
    Text(String),
    /// Image dimensions and permission to merge with an adjacent frame.
    Image {
        /// Width in pixels.
        width: u32,
        /// Height in pixels.
        height: u32,
        /// Whether the frame may join an adjacent equal-sized frame.
        mergeable: bool,
    },
    /// Audio inputs never join temporal image groups.
    Audio {
        /// Sample count.
        samples: u32,
    },
}

/// Whether aligned image area exceeds LFM2's two-times token-area allowance.
///
/// Alignment rounds to nearest with ties to even, matching `nearbyint` in
/// the pinned implementation. Minimum token count is validated with maximum
/// count although only the maximum determines the tiling decision.
///
/// # Errors
/// Rejects zero, unordered or excessive dimensions and token/patch settings.
pub fn should_tile(
    patch: u32,
    merge: u32,
    min_tokens: u32,
    max_tokens: u32,
    width: u32,
    height: u32,
) -> Result<bool> {
    if !(1..=64).contains(&patch)
        || !(1..=8).contains(&merge)
        || !(1..=4096).contains(&min_tokens)
        || !(min_tokens..=4096).contains(&max_tokens)
        || !(1..=4096).contains(&width)
        || !(1..=4096).contains(&height)
    {
        return Err(invalid("invalid image tiling dimensions or token bounds"));
    }
    let align = u64::from(patch) * u64::from(merge);
    let round = |dimension: u32| {
        let value = u64::from(dimension);
        let quotient = value / align;
        let remainder = value % align;
        let up = remainder * 2 > align || (remainder * 2 == align && quotient % 2 != 0);
        u64::from(patch).max((quotient + u64::from(up)) * align)
    };
    Ok(round(width) * round(height) > align * align * u64::from(max_tokens) * 2)
}

fn compatible(left: &Part, right: &Part) -> bool {
    matches!(
        (left, right),
        (Part::Image { width: lw, height: lh, mergeable: true },
         Part::Image { width: rw, height: rh, mergeable: true })
            if lw == rw && lh == rh
    )
}

/// Groups adjacent compatible frames and removes each group's second frame.
///
/// The pinned temporal operation merges at most two frames even when `merge`
/// is larger. Text and audio preserve their positions and break image groups.
/// Returned groups own their metadata; the first frame remains in `parts`.
///
/// # Errors
/// Rejects an invalid merge setting, more than 64 parts or invalid part bounds.
pub fn group_bitmaps(parts: &mut Vec<Part>, merge: u32) -> Result<Vec<Vec<Part>>> {
    if !(1..=64).contains(&merge) || parts.len() > 64 {
        return Err(invalid("invalid temporal merge setting or part count"));
    }
    for part in parts.iter() {
        let valid = match part {
            Part::Text(text) => text.len() <= 65536,
            Part::Image { width, height, .. } => {
                (1..=4096).contains(width) && (1..=4096).contains(height)
            }
            Part::Audio { samples } => (1..=1_048_576).contains(samples),
        };
        if !valid {
            return Err(invalid("invalid multimodal input part"));
        }
    }
    let mut groups = Vec::new();
    let mut index = 0;
    while index < parts.len() {
        if !matches!(parts[index], Part::Text(_)) {
            let mut group = vec![parts[index].clone()];
            if merge > 1 && index + 1 < parts.len() && compatible(&parts[index], &parts[index + 1])
            {
                group.push(parts.remove(index + 1));
            }
            groups.push(group);
        }
        index += 1;
    }
    Ok(groups)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiling_ties_and_temporal_mutation() -> Result<()> {
        assert!(!should_tile(16, 2, 64, 256, 720, 720)?);
        assert!(should_tile(16, 2, 64, 256, 736, 736)?);
        assert!(should_tile(0, 2, 64, 256, 736, 736).is_err());
        let image = Part::Image {
            width: 100,
            height: 100,
            mergeable: true,
        };
        let mut parts = vec![
            image.clone(),
            image.clone(),
            Part::Text("gap".into()),
            image,
        ];
        let groups = group_bitmaps(&mut parts, 64)?;
        assert_eq!(groups.iter().map(Vec::len).collect::<Vec<_>>(), [2, 1]);
        assert_eq!(parts.len(), 3);
        assert!(matches!(&parts[1], Part::Text(text) if text == "gap"));
        Ok(())
    }
}
