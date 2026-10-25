// SPDX-License-Identifier: MIT
//! Bounded execution of pure counted source loops over existing typed buffers.
//!
//! Compile only local scalar arithmetic, immutable captured numbers and checked
//! buffer stores. Calls, pointer accesses, external scalar writes and mutated
//! loop counters fall back before execution. Each loop has constant integer
//! bounds, and its total stores cannot exceed the destination buffer capacity.
//! The portable bit-count loop is folded with the same final local values; no
//! source class, input geometry, tensor formula or expected result is special.

use super::super::{Result, invalid};
use super::interpreter::Interpreter;
use super::number::{Kind, Number};
use super::program::{Expression, Statement};
use super::value::{Buffer, Value};
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;

enum Scalar {
    Constant(Number),
    Local(usize),
    Cast(Kind, Box<Self>),
    Unary(String, Box<Self>),
    Binary(String, Box<Self>, Box<Self>),
    Choose(Box<Self>, Box<Self>, Box<Self>),
}

fn binary(op: &str, a: Number, b: Number) -> Result<Number> {
    let boolean = match op {
        "==" => Some(a.equal(b)),
        "!=" => Some(!a.equal(b)),
        "<" => Some(a.less(b)),
        "<=" => Some(a.less(b) || a.equal(b)),
        ">" => Some(b.less(a)),
        ">=" => Some(b.less(a) || a.equal(b)),
        _ => None,
    };
    boolean.map_or_else(
        || a.binary(op, b).map_err(invalid),
        |value| Ok(Number::I32(i32::from(value))),
    )
}

impl Scalar {
    fn value(&self, locals: &[Number]) -> Result<Number> {
        match self {
            Self::Constant(value) => Ok(*value),
            Self::Local(index) => Ok(locals[*index]),
            Self::Cast(kind, value) => Ok(value.value(locals)?.cast(*kind)),
            Self::Unary(op, value) => {
                let value = value.value(locals)?;
                Ok(if op == "-" {
                    value.negate()
                } else {
                    Number::I32(i32::from(!value.truth()))
                })
            }
            Self::Binary(op, left, right) => binary(op, left.value(locals)?, right.value(locals)?),
            Self::Choose(condition, yes, no) => {
                if condition.value(locals)?.truth() {
                    yes.value(locals)
                } else {
                    no.value(locals)
                }
            }
        }
    }

    fn constant(&self) -> Option<Number> {
        match self {
            Self::Local(_) => None,
            Self::Constant(value) => Some(*value),
            Self::Cast(kind, value) => Some(value.constant()?.cast(*kind)),
            Self::Unary(op, value) => {
                let value = value.constant()?;
                Some(if op == "-" {
                    value.negate()
                } else {
                    Number::I32(i32::from(!value.truth()))
                })
            }
            Self::Binary(op, left, right) => binary(op, left.constant()?, right.constant()?).ok(),
            Self::Choose(condition, yes, no) => {
                if condition.constant()?.truth() {
                    yes.constant()
                } else {
                    no.constant()
                }
            }
        }
    }
}

enum Instruction {
    Set(usize, Scalar),
    Store(usize, Scalar, Scalar),
    CountBits {
        value: usize,
        sum: usize,
    },
    For {
        index: usize,
        start: u64,
        end: u64,
        body: Vec<Self>,
    },
}

struct Compiler<'a, 'b> {
    runtime: &'a Interpreter<'b>,
    names: BTreeMap<String, usize>,
    kinds: Vec<Kind>,
    counters: BTreeSet<usize>,
    buffers: Vec<Rc<RefCell<Buffer>>>,
    nodes: usize,
}

fn numeric_kind(kind: &str) -> Option<Kind> {
    let clean = kind.replace("constexpr", "").replace("const", "");
    [
        "int", "int32_t", "uint32_t", "int64_t", "uint64_t", "size_t", "float", "double",
    ]
    .contains(&clean.as_str())
    .then(|| Kind::from_type(&clean))
}

fn id(expression: &Expression) -> Option<&str> {
    if let Expression::Id { name } = expression {
        Some(name)
    } else {
        None
    }
}

fn one(expression: &Expression) -> bool {
    let Expression::Literal { value, number_kind } = expression else {
        return false;
    };
    super::standard::literal(value, number_kind.as_deref())
        .ok()
        .and_then(|value| value.number().ok())
        .is_some_and(|value| !matches!(value.kind(), Kind::F32 | Kind::F64) && value.as_u64() == 1)
}

impl Compiler<'_, '_> {
    fn captured(&self, name: &str) -> Option<Value> {
        // A scalar reference can alias a written buffer element. Freezing that
        // reference would change later reads even though its binding is stable.
        if self
            .runtime
            .cell(name)
            .is_some_and(|cell| matches!(cell.borrow().value, Value::Reference(_)))
        {
            return None;
        }
        self.runtime.value(name).ok()
    }

    fn scalar(&mut self, expression: &Expression) -> Option<Scalar> {
        self.nodes += 1;
        if self.nodes > 256 {
            return None;
        }
        Some(match expression {
            Expression::Literal { value, number_kind } => {
                let number = super::standard::literal(value, number_kind.as_deref())
                    .ok()?
                    .number()
                    .ok()?;
                Scalar::Constant(number)
            }
            Expression::Id { name } => {
                if let Some(index) = self.names.get(name) {
                    Scalar::Local(*index)
                } else {
                    Scalar::Constant(self.captured(name)?.number().ok()?)
                }
            }
            Expression::Cast { kind, value } => {
                Scalar::Cast(numeric_kind(kind)?, Box::new(self.scalar(value)?))
            }
            Expression::Unary { op, value } if matches!(op.as_str(), "-" | "!") => {
                Scalar::Unary(op.clone(), Box::new(self.scalar(value)?))
            }
            Expression::Binary { op, left, right }
                if matches!(
                    op.as_str(),
                    "+" | "-"
                        | "*"
                        | "/"
                        | "%"
                        | "&"
                        | "|"
                        | "^"
                        | "<<"
                        | ">>"
                        | "=="
                        | "!="
                        | "<"
                        | "<="
                        | ">"
                        | ">="
                ) =>
            {
                Scalar::Binary(
                    op.clone(),
                    Box::new(self.scalar(left)?),
                    Box::new(self.scalar(right)?),
                )
            }
            Expression::Ternary { condition, yes, no } => Scalar::Choose(
                Box::new(self.scalar(condition)?),
                Box::new(self.scalar(yes)?),
                Box::new(self.scalar(no)?),
            ),
            _ => return None,
        })
    }

    fn local(&mut self, name: &str, kind: &str) -> Option<usize> {
        if self.kinds.len() >= 32 {
            return None;
        }
        let index = self.kinds.len();
        self.kinds.push(numeric_kind(kind)?);
        self.names.insert(name.into(), index);
        Some(index)
    }

    fn writable(&self, expression: &Expression) -> Option<usize> {
        let index = *self.names.get(id(expression)?)?;
        (!self.counters.contains(&index)).then_some(index)
    }

    fn statements(&mut self, statements: &[Statement]) -> Option<Vec<Instruction>> {
        let mut output = Vec::new();
        for statement in statements {
            output.extend(self.statement(statement)?);
        }
        Some(output)
    }

    fn statement(&mut self, statement: &Statement) -> Option<Vec<Instruction>> {
        Some(match statement {
            Statement::Empty => Vec::new(),
            Statement::Declare {
                name,
                kind,
                dimensions,
                value: Some(value),
            } if dimensions.is_empty() => {
                // The initializer sees the enclosing binding before this declaration.
                let value = self.scalar(value)?;
                vec![Instruction::Set(self.local(name, kind)?, value)]
            }
            Statement::Declarations { body } => self.statements(body)?,
            Statement::Scope { body } => {
                let saved = self.names.clone();
                let body = self.statements(body)?;
                self.names = saved;
                body
            }
            Statement::Expression {
                value: Expression::Binary { op, left, right },
            } if op == "=" => {
                let value = self.scalar(right)?;
                if let Expression::Index { object, index } = left.as_ref() {
                    if self.names.contains_key(id(object)?) {
                        return None;
                    }
                    let Value::Buffer(buffer) = self.captured(id(object)?)? else {
                        return None;
                    };
                    let target = self
                        .buffers
                        .iter()
                        .position(|candidate| Rc::ptr_eq(candidate, &buffer))
                        .unwrap_or_else(|| {
                            self.buffers.push(buffer);
                            self.buffers.len() - 1
                        });
                    vec![Instruction::Store(target, self.scalar(index)?, value)]
                } else {
                    vec![Instruction::Set(self.writable(left)?, value)]
                }
            }
            Statement::For {
                initialization,
                condition,
                increment,
                body,
            } => vec![self.for_loop(initialization, condition, increment, body)?],
            Statement::While { condition, body } => vec![self.count_bits(condition, body)?],
            _ => return None,
        })
    }

    fn for_loop(
        &mut self,
        initialization: &[Statement],
        condition: &Expression,
        increment: &Expression,
        body: &[Statement],
    ) -> Option<Instruction> {
        let [
            Statement::Declare {
                name,
                kind,
                dimensions,
                value: Some(start),
            },
        ] = initialization
        else {
            return None;
        };
        if !dimensions.is_empty() {
            return None;
        }
        let kind = numeric_kind(kind)?;
        if matches!(kind, Kind::F32 | Kind::F64) {
            return None;
        }
        let Expression::Binary { op, left, right } = condition else {
            return None;
        };
        if op != "<" || id(left)? != name {
            return None;
        }
        let Expression::Unary { op, value } = increment else {
            return None;
        };
        if op != "++" || id(value)? != name {
            return None;
        }
        let start = self.scalar(start)?.constant()?.cast(kind);
        let saved = self.names.clone();
        let index = self.local(
            name,
            match kind {
                Kind::I32 => "int",
                Kind::U32 => "uint32_t",
                Kind::I64 => "int64_t",
                _ => "uint64_t",
            },
        )?;
        self.counters.insert(index);
        // The condition sees the newly declared counter, including when it
        // shadows an enclosing binding. Such a bound is not immutable.
        let end = self.scalar(right)?.constant()?;
        // Nonnegative, representable endpoints prove ++ cannot wrap and the
        // Number comparison has exactly this finite half-open integer range.
        let start = u64::try_from(start.index().ok()?).ok()?;
        let end = u64::try_from(end.index().ok()?).ok()?;
        if Number::U64(end).cast(kind).as_u64() != end {
            return None;
        }
        let body = self.statements(body)?;
        self.counters.remove(&index);
        self.names = saved;
        Some(Instruction::For {
            index,
            start,
            end,
            body,
        })
    }

    fn count_bits(&self, condition: &Expression, body: &[Statement]) -> Option<Instruction> {
        let [
            Statement::Expression {
                value:
                    Expression::Binary {
                        op: add,
                        left: sum,
                        right,
                    },
            },
            Statement::Expression {
                value:
                    Expression::Binary {
                        op: shift,
                        left: shifted,
                        right: amount,
                    },
            },
        ] = body
        else {
            return None;
        };
        let Expression::Binary {
            op: mask,
            left: source,
            right: bit,
        } = right.as_ref()
        else {
            return None;
        };
        if add != "+="
            || shift != ">>="
            || mask != "&"
            || !one(bit)
            || !one(amount)
            || id(condition)? != id(source)?
            || id(condition)? != id(shifted)?
        {
            return None;
        }
        let value = self.writable(condition)?;
        let sum = self.writable(sum)?;
        if value == sum
            || [self.kinds[value], self.kinds[sum]]
                .iter()
                .any(|kind| matches!(kind, Kind::F32 | Kind::F64))
        {
            return None;
        }
        Some(Instruction::CountBits { value, sum })
    }
}

fn writes(
    instructions: &[Instruction],
    counts: &mut [u64],
    repetitions: u64,
    visits: &mut u64,
    depth: u64,
) -> Option<u64> {
    let mut stores = false;
    let mut maximum_depth = depth;
    for instruction in instructions {
        match instruction {
            Instruction::Store(buffer, _, _) => {
                counts[*buffer] = counts[*buffer].checked_add(repetitions)?;
                stores = true;
            }
            Instruction::For {
                start, end, body, ..
            } => {
                let count = end.saturating_sub(*start);
                let repetitions = repetitions.checked_mul(count)?;
                *visits = visits.checked_add(repetitions)?;
                maximum_depth =
                    maximum_depth.max(writes(body, counts, repetitions, visits, depth + 1)?);
                stores = true;
            }
            _ => {}
        }
    }
    stores.then_some(maximum_depth)
}

fn counts_bits(instructions: &[Instruction]) -> bool {
    instructions.iter().any(|instruction| match instruction {
        Instruction::CountBits { .. } => true,
        Instruction::For { body, .. } => counts_bits(body),
        _ => false,
    })
}

fn run(
    instructions: &[Instruction],
    locals: &mut [Number],
    kinds: &[Kind],
    buffers: &[Rc<RefCell<Buffer>>],
) -> Result<()> {
    for instruction in instructions {
        match instruction {
            Instruction::Set(index, value) => {
                locals[*index] = value.value(locals)?.cast(kinds[*index]);
            }
            Instruction::Store(buffer, index, value) => {
                let value = value.value(locals)?;
                let index = index.value(locals)?.index().map_err(invalid)?;
                buffers[*buffer]
                    .borrow_mut()
                    .set(index, value)
                    .map_err(invalid)?;
            }
            Instruction::CountBits { value, sum } => {
                let bits = locals[*value];
                if matches!(bits.kind(), Kind::I32 | Kind::I64) && bits.as_i64() < 0 {
                    return Err(invalid(
                        "source bit-count right shift does not terminate for a negative value",
                    ));
                }
                let count = Number::I32(i32::try_from(bits.as_u64().count_ones())?);
                locals[*sum] = locals[*sum]
                    .binary("+", count)
                    .map_err(invalid)?
                    .cast(kinds[*sum]);
                locals[*value] = Number::I32(0).cast(kinds[*value]);
            }
            Instruction::For {
                index,
                start,
                end,
                body,
            } => {
                for value in *start..*end {
                    locals[*index] = Number::U64(value).cast(kinds[*index]);
                    run(body, locals, kinds, buffers)?;
                }
                locals[*index] = Number::U64((*start).max(*end)).cast(kinds[*index]);
            }
        }
    }
    Ok(())
}

/// Returns false without changing state unless the entire loop is recognized.
pub(super) fn execute(runtime: &Interpreter<'_>, statement: &Statement) -> Result<bool> {
    #[cfg(test)]
    if DISABLED.get() {
        return Ok(false);
    }
    if !matches!(statement, Statement::For { .. }) {
        return Ok(false);
    }
    let mut compiler = Compiler {
        runtime,
        names: BTreeMap::new(),
        kinds: Vec::new(),
        counters: BTreeSet::new(),
        buffers: Vec::new(),
        nodes: 0,
    };
    let Some(instructions) = compiler.statement(statement) else {
        return Ok(false);
    };
    let mut counts = vec![0; compiler.buffers.len()];
    let mut visits = 0;
    let Some(depth) = writes(&instructions, &mut counts, 1, &mut visits, 0) else {
        return Ok(false);
    };
    let capacity = compiler.buffers.iter().try_fold(0_u64, |capacity, buffer| {
        capacity.checked_add(buffer.borrow().len() as u64)
    });
    if !counts_bits(&instructions)
        || capacity
            .and_then(|capacity| capacity.checked_mul(depth))
            .is_none_or(|limit| visits > limit)
        || counts
            .iter()
            .zip(&compiler.buffers)
            .any(|(count, buffer)| *count > buffer.borrow().len() as u64)
    {
        return Ok(false);
    }
    let mut locals = vec![Number::I32(0); compiler.kinds.len()];
    run(
        &instructions,
        &mut locals,
        &compiler.kinds,
        &compiler.buffers,
    )?;
    Ok(true)
}

#[cfg(test)]
thread_local! { static DISABLED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) }; }

#[cfg(test)]
fn unoptimized<T>(operation: impl FnOnce() -> T) -> T {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            DISABLED.set(self.0);
        }
    }
    let _restore = Restore(DISABLED.replace(true));
    operation()
}

#[cfg(test)]
mod tests {
    use super::super::interpreter::Host;
    use super::super::program::{Method, Program};
    use super::super::value::{Element, Handle, Place, variable};
    use super::*;
    use serde_json::json;

    #[derive(Default)]
    struct MatrixHost {
        width: i64,
        bytes: Vec<u8>,
    }

    impl Host for MatrixHost {
        fn call(&mut self, name: &str, arguments: &[Value]) -> Result<Option<Value>> {
            Ok(Some(match name {
                "ggml_get_first_tensor" => Value::Handle(Handle::Tensor(1)),
                "ggml_get_next_tensor" => Value::Null,
                "ggml_nrows" => Value::Number(Number::I64(self.width)),
                "ggml_backend_tensor_set" => {
                    let Value::Pointer { value, index: 0 } = &arguments[1] else {
                        return Err(invalid("focused initializer pointer differs"));
                    };
                    let Value::Buffer(buffer) = value.as_ref() else {
                        return Err(invalid("focused initializer buffer missing"));
                    };
                    self.bytes.clone_from(&buffer.borrow().bytes);
                    Value::Null
                }
                _ => return Err(invalid(format!("unexpected focused source call {name}"))),
            }))
        }

        fn member(&mut self, _value: &Value, name: &str) -> Result<Option<Value>> {
            Ok(Some(match name {
                "name" => Value::Bytes(b"a".to_vec()),
                "ne" => Value::array(
                    [self.width, self.width, 1, 1]
                        .map(|value| Value::Number(Number::I64(value)))
                        .into(),
                ),
                _ => return Err(invalid("unexpected focused source member")),
            }))
        }

        fn write(
            &mut self,
            _value: &Value,
            _name: &str,
            _index: Option<usize>,
            _replacement: Value,
        ) -> Result<()> {
            Err(invalid("focused source must not mutate native metadata"))
        }

        fn force(&mut self, value: Value) -> Result<Value> {
            Ok(value)
        }
    }

    fn source_loop() -> Result<Statement> {
        fn find(statements: &[Statement]) -> Option<&Statement> {
            for statement in statements {
                match statement {
                    Statement::For {
                        condition: Expression::Binary { left, right, .. },
                        ..
                    } if id(left) == Some("r") && id(right) == Some("n_rows") => {
                        return Some(statement);
                    }
                    Statement::For { body, .. } | Statement::If { body, .. } => {
                        if let Some(found) = find(body) {
                            return Some(found);
                        }
                    }
                    _ => {}
                }
            }
            None
        }
        let (program, _) = super::super::program::load().map_err(invalid)?;
        let method = program
            .method("test_fwht_signed", "initialize_tensors", 1)
            .ok_or_else(|| invalid("pinned signed FWHT initializer missing"))?;
        find(&method.statements)
            .cloned()
            .ok_or_else(|| invalid("pinned source matrix loop missing"))
    }

    fn capture(runtime: &mut Interpreter<'_>, count: usize) -> Result<Rc<RefCell<Buffer>>> {
        let buffer = Rc::new(RefCell::new(
            Buffer::new(Element::F32, count).map_err(invalid)?,
        ));
        for (name, kind, value) in [
            ("n_rows", "int64_t", Value::Number(Number::I64(8))),
            ("n_cols", "int64_t", Value::Number(Number::I64(8))),
            (
                "scale",
                "float",
                Value::Number(Number::F32(1.0 / 8.0_f32.sqrt())),
            ),
            ("data", "std::vector<float>", Value::Buffer(buffer.clone())),
        ] {
            runtime.scopes[0].insert(name.into(), variable(kind.into(), value));
        }
        Ok(buffer)
    }

    #[test]
    fn counted_loop_matches_original_interpreter_and_complete_source_initializer() -> Result<()> {
        let (source, _) = super::super::program::load().map_err(invalid)?;
        let mut optimized_host = MatrixHost {
            width: 8,
            ..MatrixHost::default()
        };
        let mut original_host = MatrixHost {
            width: 8,
            ..MatrixHost::default()
        };
        let mut optimized = Interpreter::new(source, "test_fwht_signed", &mut optimized_host)?;
        optimized.call(
            "initialize_tensors",
            vec![Value::Handle(Handle::Context(1))],
        )?;
        let mut original = Interpreter::new(source, "test_fwht_signed", &mut original_host)?;
        unoptimized(|| {
            original.call(
                "initialize_tensors",
                vec![Value::Handle(Handle::Context(1))],
            )
        })?;
        assert_eq!(optimized_host.bytes, original_host.bytes);
        assert_eq!(optimized_host.bytes.len(), 8 * 8 * 4);

        let statement = source_loop()?;
        let mut host = MatrixHost::default();
        let mut runtime = Interpreter::new(source, "test_fwht_signed", &mut host)?;
        let buffer = capture(&mut runtime, 64)?;
        assert!(execute(&runtime, &statement)?);
        assert_eq!(buffer.borrow().bytes, original_host.bytes);
        Ok(())
    }

    #[test]
    fn scalar_casts_and_overflow_match_interpreter_rounding() -> Result<()> {
        let (program, _) = super::super::program::load().map_err(invalid)?;
        let mut host = MatrixHost::default();
        let mut runtime = Interpreter::new(program, "test_fwht_signed", &mut host)?;
        let expressions = [
            json!({"k":"cast","type":"float","value":{"k":"binary","op":"+","left":{"k":"cast","type":"uint32_t","value":{"k":"literal","value":4_294_967_295_u64}},"right":{"k":"literal","value":2}}}),
            json!({"k":"cast","type":"float","value":{"k":"literal","value":16_777_217}}),
            json!({"k":"binary","op":"*","left":{"k":"cast","type":"float","value":{"k":"literal","value":16_777_217}},"right":{"k":"literal","value":{"F32Bits":1_036_831_949},"numberKind":"f32"}}),
        ];
        for expression in expressions {
            let expression = serde_json::from_value(expression)?;
            let expected = runtime.eval(&expression)?.number().map_err(invalid)?;
            let mut compiler = Compiler {
                runtime: &runtime,
                names: BTreeMap::new(),
                kinds: Vec::new(),
                counters: BTreeSet::new(),
                buffers: Vec::new(),
                nodes: 0,
            };
            let scalar = compiler
                .scalar(&expression)
                .ok_or_else(|| invalid("focused scalar not recognized"))?;
            assert_eq!(scalar.value(&[])?, expected);
        }
        Ok(())
    }

    #[test]
    fn unrecognized_mutation_negative_bounds_and_excess_writes_fall_back_without_effects()
    -> Result<()> {
        let (program, _) = super::super::program::load().map_err(invalid)?;
        let mut host = MatrixHost::default();
        let mut runtime = Interpreter::new(program, "test_fwht_signed", &mut host)?;
        let buffer = capture(&mut runtime, 64)?;
        let statement = source_loop()?;
        let mut counter_mutation = statement.clone();
        let Statement::For { body, .. } = &mut counter_mutation else {
            return Err(invalid("source for missing"));
        };
        body.push(serde_json::from_value(json!({"k":"expr","value":{"k":"binary","op":"=","left":{"k":"id","name":"r"},"right":{"k":"literal","value":0}}}))?);
        let mut negative = statement.clone();
        let Statement::For { initialization, .. } = &mut negative else {
            return Err(invalid("source for missing"));
        };
        let Statement::Declare { value, .. } = &mut initialization[0] else {
            return Err(invalid("source counter missing"));
        };
        *value = Some(serde_json::from_value(json!({"k":"literal","value":-1}))?);
        let mut unknown_call = statement.clone();
        let Statement::For { body, .. } = &mut unknown_call else {
            return Err(invalid("source for missing"));
        };
        body.push(serde_json::from_value(json!({"k":"expr","value":{"k":"call","callee":{"k":"id","name":"unrecognized"},"args":[]}}))?);
        for rejected in [counter_mutation, negative, unknown_call] {
            assert!(!execute(&runtime, &rejected)?);
            assert!(buffer.borrow().bytes.iter().all(|&byte| byte == 0));
        }
        let small = capture(&mut runtime, 63)?;
        assert!(!execute(&runtime, &statement)?);
        assert!(small.borrow().bytes.iter().all(|&byte| byte == 0));
        // Zero-trip inner loops still spend work on each outer iteration.
        // Store counts alone must not admit this billion-visit source loop.
        runtime.scopes[0]
            .get("n_rows")
            .ok_or_else(|| invalid("focused rows missing"))?
            .borrow_mut()
            .value = Value::Number(Number::I64(1_000_000_000));
        runtime.scopes[0]
            .get("n_cols")
            .ok_or_else(|| invalid("focused columns missing"))?
            .borrow_mut()
            .value = Value::Number(Number::I64(0));
        assert!(!execute(&runtime, &statement)?);
        assert!(small.borrow().bytes.iter().all(|&byte| byte == 0));
        Ok(())
    }

    #[test]
    fn folded_bit_count_preserves_compound_promotions_narrowing_and_final_word() -> Result<()> {
        for word in [
            Number::I32(255),
            Number::U32(u32::MAX),
            Number::I64(i64::MAX),
            Number::U64(u64::MAX),
        ] {
            for initial in [
                Number::I32(i32::MAX),
                Number::U32(u32::MAX),
                Number::I64(i64::MAX),
                Number::U64(u64::MAX),
            ] {
                let mut expected_word = word;
                let mut expected_sum = initial;
                while expected_word.truth() {
                    expected_sum = expected_sum
                        .binary(
                            "+",
                            expected_word.binary("&", Number::I32(1)).map_err(invalid)?,
                        )
                        .map_err(invalid)?
                        .cast(initial.kind());
                    expected_word = expected_word
                        .binary(">>", Number::I32(1))
                        .map_err(invalid)?
                        .cast(word.kind());
                }
                let mut locals = [word, initial];
                run(
                    &[Instruction::CountBits { value: 0, sum: 1 }],
                    &mut locals,
                    &[word.kind(), initial.kind()],
                    &[],
                )?;
                assert_eq!(locals, [expected_word, expected_sum]);
            }
        }
        let mut negative = [Number::I64(-1), Number::I32(0)];
        assert!(
            run(
                &[Instruction::CountBits { value: 0, sum: 1 }],
                &mut negative,
                &[Kind::I64, Kind::I32],
                &[]
            )
            .is_err()
        );
        assert_eq!(negative, [Number::I64(-1), Number::I32(0)]);
        Ok(())
    }

    #[test]
    fn counter_shadowing_matches_interpreter_and_aliased_captures_do_not_freeze() -> Result<()> {
        let mut statement = source_loop()?;
        let Statement::For {
            condition: Expression::Binary { right, .. },
            ..
        } = &mut statement
        else {
            return Err(invalid("source counter condition missing"));
        };
        **right = Expression::Id { name: "r".into() };
        let mut program: Program = serde_json::from_value(json!({
            "SchemaVersion":1,"Source":"focused","SourceSha256":"",
            "Classes":[{"Name":"focused","Base":"","Fields":[],"Constructor":{"Parameters":[],"Initializers":[],"Statements":[]},"Methods":[]}],
            "Helpers":[],"Constants":{}
        }))?;
        program.helpers.push(Method {
            name: "fill".into(),
            parameters: Vec::new(),
            statements: vec![statement.clone()],
            source_start: 0,
            source_end: 0,
        });
        let mut optimized_host = MatrixHost::default();
        let mut original_host = MatrixHost::default();
        let mut optimized = Interpreter::new(&program, "focused", &mut optimized_host)?;
        let mut original = Interpreter::new(&program, "focused", &mut original_host)?;
        let optimized_buffer = capture(&mut optimized, 64)?;
        let original_buffer = capture(&mut original, 64)?;
        for runtime in [&mut optimized, &mut original] {
            runtime.fields = std::mem::take(&mut runtime.scopes[0]);
            runtime.fields.insert(
                "r".into(),
                variable("int64_t".into(), Value::Number(Number::I64(8))),
            );
        }
        assert!(!execute(&optimized, &statement)?);
        optimized.call("fill", Vec::new())?;
        unoptimized(|| original.call("fill", Vec::new()))?;
        assert_eq!(
            optimized_buffer.borrow().bytes,
            original_buffer.borrow().bytes
        );
        assert!(
            optimized_buffer
                .borrow()
                .bytes
                .iter()
                .all(|&byte| byte == 0)
        );
        assert_eq!(
            optimized.value("r")?.number().map_err(invalid)?,
            Number::I64(8)
        );

        optimized
            .fields
            .get("scale")
            .ok_or_else(|| invalid("focused scale missing"))?
            .borrow_mut()
            .value = Value::Reference(Place::Buffer(optimized_buffer.clone(), 0));
        assert!(!execute(&optimized, &source_loop()?)?);
        assert!(
            optimized_buffer
                .borrow()
                .bytes
                .iter()
                .all(|&byte| byte == 0)
        );
        for kind in ["f32", "f64", "float", "double"] {
            assert!(!one(&Expression::Literal {
                value: json!(1),
                number_kind: Some(kind.into())
            }));
        }
        assert!(one(&Expression::Literal {
            value: json!(1),
            number_kind: Some("integer".into())
        }));
        Ok(())
    }

    #[test]
    #[ignore = "focused 8192-square source initialization retains a 256 MiB matrix"]
    fn full_8192_source_initializer_completes_every_original_matrix_element() -> Result<()> {
        let (program, _) = super::super::program::load().map_err(invalid)?;
        let mut host = MatrixHost {
            width: 8192,
            ..MatrixHost::default()
        };
        let mut runtime = Interpreter::new(program, "test_fwht_signed", &mut host)?;
        runtime.call(
            "initialize_tensors",
            vec![Value::Handle(Handle::Context(1))],
        )?;
        let width = 8192_usize;
        assert_eq!(host.bytes.len(), width * width * 4);
        let scale = (1.0_f32 / 8192.0_f32.sqrt()).to_bits();
        let mut negative = 0;
        for bytes in host.bytes.as_chunks::<4>().0 {
            let bits = u32::from_le_bytes(*bytes);
            assert_eq!(bits & 0x7fff_ffff, scale);
            negative += usize::from(bits >> 31 != 0);
        }
        // A normalized Sylvester matrix has a positive first row/column and
        // balanced signs in each remaining row. Inspect all writes without
        // recomputing the optimized portable bit-count expression.
        assert_eq!(negative, (width * width - width) / 2);
        for index in 0..width {
            assert_eq!(&host.bytes[index * 4..index * 4 + 4], scale.to_le_bytes());
            let column = index * width * 4;
            assert_eq!(&host.bytes[column..column + 4], scale.to_le_bytes());
        }
        Ok(())
    }
}
