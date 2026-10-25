// Copyright (c) 2023-2026 The ggml/llama.cpp authors
// SPDX-License-Identifier: MIT
//! Safe Rust reader for the original --test-file graph descriptor format.

use super::super::{Result, invalid};
use super::program::{Geometry, Program};
use serde_json::json;

struct Tokens<'a> {
    words: std::str::SplitAsciiWhitespace<'a>,
}
impl Tokens<'_> {
    fn signed(&mut self) -> Result<i64> {
        self.words
            .next()
            .ok_or_else(|| invalid("test-file descriptor ended inside a signed integer"))?
            .parse()
            .map_err(Into::into)
    }
    fn unsigned(&mut self) -> Result<u64> {
        let text = self
            .words
            .next()
            .ok_or_else(|| invalid("test-file descriptor ended inside an unsigned integer"))?;
        if text.starts_with('-') {
            let value = text.parse::<i128>()?;
            Ok(value as u64)
        } else {
            text.parse().map_err(Into::into)
        }
    }
    fn signed_four(&mut self) -> Result<[i64; 4]> {
        Ok([
            self.signed()?,
            self.signed()?,
            self.signed()?,
            self.signed()?,
        ])
    }
    fn unsigned_four(&mut self) -> Result<[u64; 4]> {
        Ok([
            self.unsigned()?,
            self.unsigned()?,
            self.unsigned()?,
            self.unsigned()?,
        ])
    }
}

pub(super) fn parse(program: &Program, text: &str) -> Result<Vec<Geometry>> {
    if text.len() > 128 * 1024 * 1024 {
        return Err(invalid("test-file descriptor text exceeds128MiB"));
    }
    let maximum = usize::try_from(
        program.constants["GGML_MAX_SRC"]
            .as_u64()
            .ok_or_else(|| invalid("source slot count missing"))?,
    )?;
    let params = usize::try_from(
        program.constants["GGML_MAX_OP_PARAMS"]
            .as_u64()
            .ok_or_else(|| invalid("source parameter bytes missing"))?
            / 4,
    )?;
    let mut geometries = Vec::new();
    for (number, line) in text.lines().enumerate() {
        let parsed = (|| {
            let mut words = Tokens {
                words: line.split_ascii_whitespace(),
            };
            let operation = words.unsigned()? as u32 as i32;
            let kind = words.unsigned()? as u32 as i32;
            let shape = words.signed_four()?;
            let mut parameters = vec![0_i32; params];
            let declared = words.unsigned()?;
            // The source consumes only min(count,capacity); it does not skip excess fields.
            for parameter in parameters
                .iter_mut()
                .take(usize::try_from(declared.min(params as u64))?)
            {
                *parameter = i32::try_from(words.signed()?)?;
            }
            let count = usize::try_from(words.unsigned()?.min(maximum as u64))?;
            let mut sources = Vec::with_capacity(count);
            for _ in 0..count {
                let kind = words.unsigned()? as u32 as i32;
                let shape = words.signed_four()?;
                let strides = words.unsigned_four()?;
                sources.push(json!({"Record":{"type":kind,"ne":shape,"nb":strides}}));
            }
            let name = words.words.next().unwrap_or("");
            let name = if name == "-" { "" } else { name };
            Ok(Geometry {
                class: "test_generic_op".into(),
                arguments: vec![
                    json!(operation),
                    json!(kind),
                    json!(shape),
                    json!(parameters),
                    json!(sources),
                    json!(name),
                ],
            })
        })();
        geometries.push(
            parsed.map_err(|error: Box<dyn std::error::Error + Send + Sync>| {
                invalid(format!("test-file line{}: {error}", number + 1))
            })?,
        );
    }
    Ok(geometries)
}
