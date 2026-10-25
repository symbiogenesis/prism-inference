//! Original exhaustive and seeded optional Python input schedules, in safe Rust.
use super::{Result, invalid};
use serde_json::Value;
use std::collections::BTreeSet;

#[derive(Clone, Copy)]
pub(super) enum Kind {
    Custom,
    Edges,
    Ascii,
    Apostrophe,
    Unicode,
    Vocab,
    Added,
    RandomAdded,
    RandomChars,
    RandomUnicode,
    RandomVocabChars,
    RandomVocabWords,
}
pub(super) struct Vocabulary {
    pub(super) words: Vec<String>,
    pub(super) special: Vec<String>,
    pub(super) added: Vec<String>,
    pub(super) bos: Option<String>,
    pub(super) eos: Option<String>,
    pub(super) add_bos: bool,
    pub(super) add_eos: bool,
}
pub(super) struct PythonRandom {
    mt: [u32; 624],
    index: usize,
}
impl PythonRandom {
    pub(super) fn new(seed: u32) -> Self {
        let mut value = Self {
            mt: [0; 624],
            index: 624,
        };
        value.mt[0] = 19_650_218;
        for i in 1..624 {
            value.mt[i] = 1_812_433_253_u32
                .wrapping_mul(value.mt[i - 1] ^ (value.mt[i - 1] >> 30))
                .wrapping_add(u32::try_from(i).unwrap_or(0));
        }
        let mut i = 1;
        for _ in 0..624 {
            value.mt[i] = (value.mt[i]
                ^ ((value.mt[i - 1] ^ (value.mt[i - 1] >> 30)).wrapping_mul(1_664_525)))
            .wrapping_add(seed);
            i += 1;
            if i == 624 {
                value.mt[0] = value.mt[623];
                i = 1;
            }
        }
        for _ in 0..623 {
            value.mt[i] = (value.mt[i]
                ^ ((value.mt[i - 1] ^ (value.mt[i - 1] >> 30)).wrapping_mul(1_566_083_941)))
            .wrapping_sub(u32::try_from(i).unwrap_or(0));
            i += 1;
            if i == 624 {
                value.mt[0] = value.mt[623];
                i = 1;
            }
        }
        value.mt[0] = 0x8000_0000;
        value
    }
    fn next(&mut self) -> u32 {
        if self.index == 624 {
            for i in 0..624 {
                let y = (self.mt[i] & 0x8000_0000) | (self.mt[(i + 1) % 624] & 0x7fff_ffff);
                self.mt[i] =
                    self.mt[(i + 397) % 624] ^ (y >> 1) ^ if y & 1 == 1 { 0x9908_b0df } else { 0 };
            }
            self.index = 0;
        }
        let mut y = self.mt[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^ (y >> 18)
    }
    pub(super) fn below(&mut self, n: usize) -> Result<usize> {
        let n = u32::try_from(n)
            .ok()
            .filter(|&v| v > 0)
            .ok_or_else(|| invalid("Python random bound invalid"))?;
        let bits = 32 - n.leading_zeros();
        loop {
            let value = self.next() >> (32 - bits);
            if value < n {
                return Ok(usize::try_from(value)?);
            }
        }
    }
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::as_conversions,
        clippy::suboptimal_flops,
        reason = "Python unweighted choices uses floor(random()*len), with positive bounded pool"
    )]
    pub(super) fn choices_index(&mut self, n: usize) -> Result<usize> {
        let n = u32::try_from(n)
            .ok()
            .filter(|&v| v > 0)
            .ok_or_else(|| invalid("Python choices pool invalid"))?;
        let upper = self.next() >> 5;
        let lower = self.next() >> 6;
        let value = (f64::from(upper) * 67_108_864.0 + f64::from(lower)) / 9_007_199_254_740_992.0;
        Ok((value * f64::from(n)).floor() as usize)
    }
}
fn inputs() -> Result<Value> {
    Ok(serde_json::from_str(include_str!(
        "tokenizer-generator-inputs.json"
    ))?)
}
pub(super) fn unicode_characters() -> Result<Vec<char>> {
    let data = inputs()?;
    let mut values = Vec::new();
    if data["PythonUnicodeVersion"] != "16.0.0" || data["AssignedCodepointCount"] != 145_595 {
        return Err(invalid("Python Unicode generator identity differs"));
    }
    for range in data["AssignedCodepointRanges"]
        .as_array()
        .ok_or_else(|| invalid("Unicode ranges absent"))?
    {
        let begin = u32::try_from(
            range[0]
                .as_u64()
                .ok_or_else(|| invalid("Unicode begin absent"))?,
        )?;
        let end = u32::try_from(
            range[1]
                .as_u64()
                .ok_or_else(|| invalid("Unicode end absent"))?,
        )?;
        if begin > end || end >= 0x30000 {
            return Err(invalid("Unicode range invalid"));
        }
        for point in begin..=end {
            values.push(
                char::from_u32(point)
                    .ok_or_else(|| invalid("Unicode surrogate in valid domain"))?,
            );
        }
    }
    if values.len() != 145_595 || !values.windows(2).all(|v| v[0] < v[1]) {
        return Err(invalid("Unicode domain extent/order differs"));
    }
    Ok(values)
}
fn whitespace() -> Vec<char> {
    (" ".repeat(20) + &"\n".repeat(5) + &"\r\n".repeat(5) + &"\t".repeat(5))
        .chars()
        .collect()
}
fn special_words(vocabulary: &Vocabulary) -> Vec<String> {
    vocabulary
        .special
        .iter()
        .chain(&vocabulary.added)
        .cloned()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}
pub(super) fn python_strip(text: &str) -> &str {
    text.trim_matches(|c: char| c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c))
}
#[allow(
    clippy::too_many_lines,
    reason = "preserve the original source's complete generator families and loop nesting"
)]
pub(super) fn visit(
    kind: Kind,
    vocabulary: &Vocabulary,
    iterations: usize,
    mut emit: impl FnMut(String) -> Result<bool>,
) -> Result<usize> {
    if iterations > 10_000 {
        return Err(invalid(
            "optional tokenizer random iterations exceed source maximum10000",
        ));
    }
    let mut count = 0;
    let mut supply = |text: String| -> Result<bool> {
        count += 1;
        emit(text)
    };
    match kind {
        Kind::Custom | Kind::Edges => {
            let data = inputs()?;
            let key = if matches!(kind, Kind::Custom) {
                "generator_custom_text"
            } else {
                "generator_custom_text_edge_cases"
            };
            for text in data[key]
                .as_array()
                .ok_or_else(|| invalid("custom inputs absent"))?
            {
                if !supply(
                    text.as_str()
                        .ok_or_else(|| invalid("custom input not text"))?
                        .into(),
                )? {
                    break;
                }
            }
        }
        Kind::Ascii | Kind::Apostrophe => {
            let characters = (1_u8..128)
                .map(|v| char::from(v).to_string())
                .chain(std::iter::once(String::new()))
                .collect::<Vec<_>>();
            'all: for a in &characters {
                for b in &characters {
                    for left in ["", " ", "  "] {
                        for right in ["", " ", "  "] {
                            let values = if matches!(kind, Kind::Ascii) {
                                [
                                    format!("{left}{a}{b}{right}"),
                                    format!("{left}{a}{right}{b}"),
                                    format!("{a}{left}{b}{right}"),
                                ]
                            } else {
                                [
                                    format!("{a}{left}'{right}{b}"),
                                    format!("{a}{b}{left}'{right}z"),
                                    format!("a{left}'{right}{a}{b}"),
                                ]
                            };
                            for text in values {
                                if !supply(text)? {
                                    break 'all;
                                }
                            }
                        }
                    }
                }
            }
        }
        Kind::Unicode => {
            for c in unicode_characters()? {
                if !supply(c.to_string())? {
                    break;
                }
            }
        }
        Kind::Vocab => {
            for word in &vocabulary.words {
                if !supply(word.clone())? {
                    break;
                }
            }
        }
        Kind::Added => {
            'all: for token in special_words(vocabulary) {
                for left in ["", " ", "  ", "\n", "\r\n", "\n\n", "\t", "\t\t"] {
                    for right in ["", " ", "  ", "\n", "\r\n", "\n\n", "\t", "\t\t"] {
                        for text in [
                            format!("{left}{token}{right}"),
                            format!("a{left}{token}{right}"),
                            format!("{left}{token}{right}z"),
                            format!("a{left}{token}{right}z"),
                        ] {
                            if !supply(text)? {
                                break 'all;
                            }
                        }
                    }
                }
            }
        }
        Kind::RandomAdded => {
            let mut pool = special_words(vocabulary);
            pool.extend(
                [" ", "\n", "\t", "-", "!", "one", "1", "<s>", "</s>"]
                    .into_iter()
                    .map(str::to_owned),
            );
            pool = pool
                .into_iter()
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect();
            for seed in 0..iterations {
                let mut random = PythonRandom::new(u32::try_from(seed)?);
                let mut words = (0..500)
                    .map(|_| Ok(pool[random.choices_index(pool.len())?].clone()))
                    .collect::<Result<Vec<_>>>()?;
                if words.first() == vocabulary.bos.as_ref() {
                    while words.len() > 1 && words.get(1) == vocabulary.bos.as_ref() {
                        words.remove(0);
                    }
                    if vocabulary.add_bos {
                        words.remove(0);
                    }
                }
                if words.last() == vocabulary.eos.as_ref() {
                    while words.len() > 1 && words.get(words.len() - 2) == vocabulary.eos.as_ref() {
                        words.pop();
                    }
                    if vocabulary.add_eos {
                        words.pop();
                    }
                }
                if !supply(words.concat())? {
                    break;
                }
            }
        }
        Kind::RandomChars | Kind::RandomUnicode => {
            let chars = if matches!(kind, Kind::RandomUnicode) {
                unicode_characters()?
            } else {
                "\n        ABCDEFGHIJKLMNOPQRSTUVWXYZ\n        abcdefghijklmnopqrstuvwxyz\n        ÁÉÍÓÚÀÈÌÒÙÂÊÎÔÛÄËÏÖÜ\n        áéíóúàèìòùâêîôûäëïöü\n        .-,*/-+ª!\"·$%&/()=?¿[]{}<>\\|@#~½¬~;:_\n    ".chars().collect::<BTreeSet<_>>().into_iter().collect()
            };
            let spaces = whitespace();
            let words = if matches!(kind, Kind::RandomUnicode) {
                200
            } else {
                400
            };
            for seed in 0..iterations {
                let mut random = PythonRandom::new(u32::try_from(seed)?);
                let mut text = String::new();
                for _ in 0..words {
                    for _ in 0..=random.below(7)? {
                        text.push(chars[random.choices_index(chars.len())?]);
                    }
                    text.push(spaces[random.below(spaces.len())?]);
                }
                if !supply(text)? {
                    break;
                }
            }
        }
        Kind::RandomVocabChars => {
            let chars = vocabulary
                .words
                .iter()
                .flat_map(|s| s.chars())
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect::<Vec<_>>();
            for seed in 0..iterations {
                let mut random = PythonRandom::new(u32::try_from(seed)?);
                let mut text = String::new();
                for _ in 0..1024 {
                    text.push(chars[random.choices_index(chars.len())?]);
                }
                if !supply(text)? {
                    break;
                }
            }
        }
        Kind::RandomVocabWords => {
            let pool = vocabulary
                .words
                .iter()
                .map(|s| python_strip(s))
                .collect::<Vec<_>>();
            for word in &pool {
                if !supply((*word).into())? {
                    return Ok(count);
                }
            }
            let spaces = [' ', ' ', ' ', ' ', ' ', '\n', '\r', '\t'];
            for seed in 0..iterations {
                let mut random = PythonRandom::new(u32::try_from(seed)?);
                let mut text = String::new();
                let words = random.below(101)? + 300;
                for _ in 0..words {
                    for _ in 0..=random.below(3)? {
                        text.push_str(pool[random.choices_index(pool.len())?]);
                    }
                    text.push(spaces[random.below(spaces.len())?]);
                }
                if !supply(text)? {
                    break;
                }
            }
        }
    }
    Ok(count)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn focused_python_generator_input_parity_guard() -> Result<()> {
        let vocabulary = Vocabulary {
            words: vec![
                String::new(),
                " ".into(),
                "a".into(),
                "a".into(),
                "é".into(),
            ],
            special: vec!["<s>".into(), "</s>".into(), "<unk>".into()],
            added: vec!["<s>".into(), "</s>".into(), "X".into()],
            bos: Some("<s>".into()),
            eos: Some("</s>".into()),
            add_bos: true,
            add_eos: false,
        };
        // Original Python generator input digests, never engine observations.
        for (kind, limit, expected) in [
            (
                Kind::Custom,
                2,
                "68e49bea537e81f2af3357ffba00de0635df66f63165c3e707c4b7d0b9d9fa83",
            ),
            (
                Kind::Edges,
                2,
                "25be57a044d3d47d7d994ba0bcc439c92048d0d2c8f1be01823d922f52dc7c35",
            ),
            (
                Kind::Vocab,
                2,
                "68e49bea537e81f2af3357ffba00de0635df66f63165c3e707c4b7d0b9d9fa83",
            ),
            (
                Kind::Ascii,
                2,
                "98e6a57bc85dc665d469b0f9c2d7239404b6401009d2c2a4215004f22ba91244",
            ),
            (
                Kind::Apostrophe,
                2,
                "ae9d448676b19cc5923758cdc80956ca47a703cdad3cbae937f39863a106e2f9",
            ),
            (
                Kind::Added,
                2,
                "1f461abfee72784285f2a0b2d848e7f1ebb7fa146a5e06619cfb88c88dc1fb0b",
            ),
            (
                Kind::RandomAdded,
                2,
                "962fbe962462b0cb90bd41a4e1bac98153d40a36d1a9c18fa7ca93010b9438f3",
            ),
            (
                Kind::RandomChars,
                2,
                "cb63c65b9b9c912286f983c71c0888e455d629295450c1528a4c8e84fdd9d93d",
            ),
            (
                Kind::Unicode,
                2,
                "9b086f8eb1ac4eebf86edbc62707c0c66d6a49064c78a9fa42eb68e683208cc2",
            ),
            (
                Kind::RandomUnicode,
                2,
                "5d8147c41c50f35a389078234f0689b24a507ae85b0c70a11909624c16403ac5",
            ),
            (
                Kind::RandomVocabChars,
                2,
                "e355c489280c71ad087ccd360776d9358c8923ef2ec34190eca03543a8c28b8f",
            ),
            (
                Kind::RandomVocabWords,
                7,
                "20005df160a7ffd1ac5ea3e9ef5c87d9d78531195754a4b61c302d46ff609af2",
            ),
        ] {
            let mut raw = Vec::new();
            let mut emitted = 0;
            visit(kind, &vocabulary, 2, |text| {
                raw.extend_from_slice(&u64::try_from(text.len())?.to_le_bytes());
                raw.extend_from_slice(text.as_bytes());
                emitted += 1;
                Ok(emitted < limit)
            })?;
            if emitted != limit || super::super::lora_inputs::hash_bytes(&raw) != expected {
                return Err(invalid("original Python generator input parity differs"));
            }
        }
        if unicode_characters()?.len() != 145_595 {
            return Err(invalid("default generator domain differs"));
        }
        Ok(())
    }
}
