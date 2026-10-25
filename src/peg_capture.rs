//! Owned structural captures over the production streaming PEG executor.
use super::{Arena, ParseResult, Parser};
use crate::{Result, invalid};

/// A named rule or tagged byte extent in a PEG parse tree.
#[derive(Clone, Debug)]
pub struct Capture {
    /// Rule name, empty for tag wrappers.
    pub name: String,
    /// Semantic tag, empty for ordinary named rules.
    pub tag: String,
    /// Initial byte offset.
    pub start: usize,
    /// Exclusive byte offset reached by the capture.
    pub end: usize,
    /// Whether this node represents an incomplete input prefix.
    pub incomplete: bool,
    /// Captures nested under this rule or tag in source traversal order.
    pub children: Vec<Self>,
}
/// Parse status together with its reachable rule and tag nodes.
#[derive(Clone, Debug)]
pub struct CapturedParse {
    /// Production streaming parse extent and status.
    pub result: ParseResult,
    /// Root captures in source traversal order.
    pub nodes: Vec<Capture>,
}
impl Arena {
    /// Parse arbitrary bytes and retain rule/tag extents, optionally starting within input.
    ///
    /// # Errors
    /// Returns an error for invalid offsets/flags or input, recursion and work limits.
    pub fn parse_captured(&self, bytes: &[u8], flags: u8, start: usize) -> Result<CapturedParse> {
        if bytes.len() > 1024 * 1024 || flags > 3 || start > bytes.len() {
            return Err(invalid("PEG capture input/flags/offset outside bound"));
        }
        self.capture_at(self.root, bytes, flags & 1 != 0, start, 0, &mut 0)
    }
    fn capture_at(
        &self,
        id: usize,
        bytes: &[u8],
        lenient: bool,
        start: usize,
        depth: usize,
        steps: &mut usize,
    ) -> Result<CapturedParse> {
        let result = self.parse_at(id, bytes, lenient, start, depth, steps)?;
        let mut nodes = Vec::new();
        if result.result_type == 0 {
            return Ok(CapturedParse { result, nodes });
        }
        let child = |id, start, steps: &mut usize| {
            self.capture_at(id, bytes, lenient, start, depth + 1, steps)
        };
        let parser = &self.parsers[id];
        match parser {
            Parser::Sequence { children } => {
                let mut offset = start;
                for &id in children {
                    let parsed = child(id, offset, steps)?;
                    nodes.extend(parsed.nodes);
                    offset = parsed.result.end;
                    if parsed.result.result_type != 1 {
                        break;
                    }
                }
            }
            Parser::Choice { children } => {
                for &id in children {
                    let parsed = child(id, start, steps)?;
                    if parsed.result.result_type != 0 {
                        nodes = parsed.nodes;
                        break;
                    }
                }
            }
            Parser::Repetition {
                child: id,
                max_count,
                ..
            } => {
                let mut offset = start;
                let mut count = 0;
                while (*max_count < 0 || count < *max_count) && offset < bytes.len() {
                    let parsed = child(*id, offset, steps)?;
                    if parsed.result.result_type == 0 {
                        break;
                    }
                    nodes.extend(parsed.nodes);
                    if parsed.result.result_type == 2 || parsed.result.end == offset {
                        break;
                    }
                    offset = parsed.result.end;
                    count += 1;
                }
            }
            Parser::Rule {
                child: id, name, ..
            }
            | Parser::Tag {
                child: id,
                tag: name,
            } => {
                let parsed = child(*id, start, steps)?;
                nodes.push(Capture {
                    name: if matches!(parser, Parser::Rule { .. }) {
                        name.clone()
                    } else {
                        String::new()
                    },
                    tag: if matches!(parser, Parser::Tag { .. }) {
                        name.clone()
                    } else {
                        String::new()
                    },
                    start: result.start,
                    end: result.end,
                    incomplete: result.result_type == 2,
                    children: parsed.nodes,
                });
            }
            Parser::Atomic { child: id } => {
                if result.result_type != 2 {
                    nodes = child(*id, start, steps)?.nodes;
                }
            }
            Parser::Gbnf { child: id, .. }
            | Parser::Ac { child: id, .. }
            | Parser::Schema { child: id, .. } => nodes = child(*id, start, steps)?.nodes,
            Parser::Ref { name } => {
                nodes = child(
                    *self
                        .rules
                        .get(name)
                        .ok_or_else(|| invalid("PEG rule reference missing"))?,
                    start,
                    steps,
                )?
                .nodes;
            }
            _ => {}
        }
        if nodes.len() > 65536 {
            return Err(invalid("PEG capture node bound exceeded"));
        }
        Ok(CapturedParse { result, nodes })
    }
}
