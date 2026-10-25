//! Decimal integer range lowering from the pinned MIT-licensed Prism converter.

use crate::{Result, invalid};
use std::fmt::Write as _;

fn range(from: u8, to: u8) -> String {
    if from == to {
        format!("[{}]", char::from(from))
    } else {
        format!("[{}-{}]", char::from(from), char::from(to))
    }
}

fn digits(minimum: i32, maximum: i32) -> String {
    if minimum == 1 && maximum == 1 {
        return "[0-9]".to_owned();
    }
    if minimum == maximum {
        format!("[0-9]{{{minimum}}}")
    } else if maximum == i32::MAX {
        format!("[0-9]{{{minimum},}}")
    } else {
        format!("[0-9]{{{minimum},{maximum}}}")
    }
}

fn uniform(from: &str, to: &str) -> Result<String> {
    if from.len() != to.len()
        || !from
            .bytes()
            .chain(to.bytes())
            .all(|byte| byte.is_ascii_digit())
    {
        return Err(invalid(
            "integer grammar range requires equally sized digit strings",
        ));
    }
    let shared = from
        .bytes()
        .zip(to.bytes())
        .take_while(|(left, right)| left == right)
        .count();
    let mut output = String::new();
    if shared > 0 {
        write!(output, "\"{}\"", &from[..shared])?;
    }
    if shared < from.len() {
        if shared > 0 {
            output.push(' ');
        }
        let trailing = from.len() - shared - 1;
        let first = from.as_bytes()[shared];
        let last = to.as_bytes()[shared];
        if trailing == 0 {
            write!(output, "[{}-{}]", char::from(first), char::from(last))?;
        } else {
            let lower = &from[shared + 1..];
            let upper = &to[shared + 1..];
            let zeros = "0".repeat(trailing);
            let nines = "9".repeat(trailing);
            let mut reached = false;
            output.push('(');
            if lower == zeros {
                output.push_str(&range(
                    first,
                    last.checked_sub(1)
                        .ok_or_else(|| invalid("invalid integer digit bound"))?,
                ));
                output.push(' ');
                output.push_str(&digits(i32::try_from(trailing)?, i32::try_from(trailing)?));
            } else {
                write!(
                    output,
                    "[{}] ({})",
                    char::from(first),
                    uniform(lower, &nines)?
                )?;
                if first + 1 < last {
                    output.push_str(" | ");
                    if upper == nines {
                        output.push_str(&range(first + 1, last));
                        reached = true;
                    } else {
                        output.push_str(&range(first + 1, last - 1));
                    }
                    output.push(' ');
                    output.push_str(&digits(i32::try_from(trailing)?, i32::try_from(trailing)?));
                }
            }
            if !reached {
                output.push_str(" | ");
                output.push_str(&range(last, last));
                output.push(' ');
                output.push_str(&uniform(&zeros, upper)?);
            }
            output.push(')');
        }
    }
    Ok(output)
}

/// Preserves the upstream decimal grammar and unbounded `i64` sentinels.
#[allow(
    clippy::too_many_lines,
    reason = "The signed decimal branches retain the pinned converter's exact grammar formatting."
)]
pub(super) fn build(mut minimum: i64, maximum: i64, decimals: i32, top: bool) -> Result<String> {
    if !(1..=32).contains(&decimals) {
        return Err(invalid("integer grammar decimal budget outside 1..32"));
    }
    let has_min = minimum != i64::MIN;
    let has_max = maximum != i64::MAX;
    let less = (decimals - 1).max(1);
    if has_min && has_max {
        if minimum < 0 && maximum < 0 {
            return Ok(format!(
                "\"-\" ({})",
                build(
                    maximum
                        .checked_neg()
                        .ok_or_else(|| invalid("integer grammar negation overflow"))?,
                    minimum
                        .checked_neg()
                        .ok_or_else(|| invalid("integer grammar negation overflow"))?,
                    decimals,
                    true
                )?
            ));
        }
        let mut prefix = String::new();
        if minimum < 0 {
            prefix = format!(
                "\"-\" ({}) | ",
                build(
                    0,
                    minimum
                        .checked_neg()
                        .ok_or_else(|| invalid("integer grammar negation overflow"))?,
                    decimals,
                    true
                )?
            );
            minimum = 0;
        }
        let mut lower = minimum.to_string();
        let upper = maximum.to_string();
        for length in lower.len()..upper.len() {
            prefix.push_str(&uniform(&lower, &"9".repeat(length))?);
            prefix.push_str(" | ");
            lower = format!("1{}", "0".repeat(length));
        }
        prefix.push_str(&uniform(&lower, &upper)?);
        return Ok(prefix);
    }
    if has_min {
        if minimum < 0 {
            return Ok(format!(
                "\"-\" ({}) | [0] | [1-9] {}",
                build(
                    i64::MIN,
                    minimum
                        .checked_neg()
                        .ok_or_else(|| invalid("integer grammar negation overflow"))?,
                    decimals,
                    false
                )?,
                digits(0, decimals - 1)
            ));
        }
        if minimum == 0 {
            return Ok(if top {
                format!("[0] | [1-9] {}", digits(0, less))
            } else {
                digits(1, decimals)
            });
        }
        if minimum <= 9 {
            let digit = b'0' + u8::try_from(minimum)?;
            let first = if top { b'1' } else { b'0' };
            let prefix = if digit > first {
                format!("{} {} | ", range(first, digit - 1), digits(1, less))
            } else {
                String::new()
            };
            return Ok(format!(
                "{prefix}{} {}",
                range(digit, b'9'),
                digits(0, less)
            ));
        }
        let lower = minimum.to_string();
        let length = i32::try_from(lower.len())?;
        let digit = lower.as_bytes()[0];
        let mut output = if digit > b'1' {
            format!(
                "{} {} | ",
                range(if top { b'1' } else { b'0' }, digit - 1),
                digits(length, less)
            )
        } else {
            String::new()
        };
        write!(
            output,
            "{} ({})",
            range(digit, digit),
            build(lower[1..].parse()?, i64::MAX, less, false)?
        )?;
        if digit < b'9' {
            write!(
                output,
                " | {} {}",
                range(digit + 1, b'9'),
                digits(length - 1, less)
            )?;
        }
        return Ok(output);
    }
    if has_max {
        if maximum >= 0 {
            let prefix = if top {
                format!("\"-\" [1-9] {} | ", digits(0, less))
            } else {
                String::new()
            };
            return Ok(format!("{prefix}{}", build(0, maximum, decimals, true)?));
        }
        return Ok(format!(
            "\"-\" ({})",
            build(
                maximum
                    .checked_neg()
                    .ok_or_else(|| invalid("integer grammar negation overflow"))?,
                i64::MAX,
                decimals,
                false
            )?
        ));
    }
    Err(invalid("at least one integer grammar bound is required"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_range_lowering_retains_negative_and_digit_transitions() -> Result<()> {
        assert_eq!(build(0, i64::MAX, 16, true)?, "[0] | [1-9] [0-9]{0,15}");
        assert_eq!(
            build(25, i64::MAX, 16, true)?,
            "[1] [0-9]{2,15} | [2] ([0-4] [0-9]{1,14} | [5-9] [0-9]{0,14}) | [3-9] [0-9]{1,15}"
        );
        assert_eq!(
            build(i64::MIN, 30, 16, true)?,
            "\"-\" [1-9] [0-9]{0,15} | [0-9] | ([1-2] [0-9] | [3] \"0\")"
        );
        assert_eq!(build(-1, 1, 16, true)?, "\"-\" ([0-1]) | [0-1]");
        assert!(build(i64::MIN, i64::MAX, 16, true).is_err());
        Ok(())
    }
}
