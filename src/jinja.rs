//! Bounded Jinja template execution with ordered values, usage and byte provenance.
//!
//! Implements the MIT-licensed Prism `common/jinja` contracts at
//! `b87ca7225456b06b0975b8239c66b5aecf07e37c`; attribution is retained in
//! `tools/prism/upstream/LICENSE`. Templates are parsed into reusable syntax trees;
//! runtime code has no source-test schedules, fixture access or expected outputs.
#![allow(
    clippy::similar_names,
    reason = "Retain conventional source slice start/stop/step names"
)]
#[path = "jinja_ast.rs"]
mod ast;
#[path = "jinja_builtins.rs"]
mod builtins;
#[path = "jinja_caps.rs"]
mod caps;
#[path = "jinja_hash.rs"]
mod hash;
#[path = "jinja_inventory.rs"]
mod inventory;
#[path = "jinja_value.rs"]
mod value;
use crate::{Result, invalid};
use ast::{Arg, Expr, Literal, Stmt};
pub use caps::ChatCapabilities;
pub use hash::Hasher;
use std::collections::BTreeMap;
use value::{Kind, Part, Val};
/// Names exposed by each source value category's callable builtin table.
#[must_use]
pub const fn builtin_inventory() -> &'static [(&'static str, &'static [&'static str])] {
    inventory::INVENTORY
}

/// A path into supplied template variables.
#[derive(Clone, Debug)]
pub enum PathSegment {
    /// Object member name.
    Key(String),
    /// Array element index.
    Index(usize),
}
/// One rendered byte range and its origin.
#[derive(Clone, Debug)]
pub struct StringPart {
    /// Unmodified bytes of this output range.
    pub bytes: Vec<u8>,
    /// Whether the range originated in supplied variables.
    pub is_input: bool,
}
/// Parsed reusable template program.
#[derive(Clone, Debug)]
pub struct Template {
    nodes: Vec<Stmt>,
}
/// Rendering observations, retained even if execution raises an exception.
pub struct Execution {
    parts: Vec<Part>,
    globals: Val,
    /// Runtime exception, when execution could not complete.
    pub error: Option<String>,
}
impl Execution {
    /// Concatenated rendering bytes. May contain raw non-UTF-8 input bytes.
    #[must_use]
    pub fn bytes(&self) -> Vec<u8> {
        self.parts
            .iter()
            .flat_map(|part| part.bytes.iter().copied())
            .collect()
    }
    /// Adjacent byte ranges with the same provenance are joined.
    #[must_use]
    pub fn parts(&self) -> Vec<StringPart> {
        value::merge(self.parts.clone())
            .into_iter()
            .map(|part| StringPart {
                bytes: part.bytes,
                is_input: part.input,
            })
            .collect()
    }
    fn at(&self, path: &[PathSegment]) -> Val {
        let mut value = self.globals.clone();
        for segment in path {
            value = value.get(
                &match segment {
                    PathSegment::Key(key) => Val::string(key.clone()),
                    PathSegment::Index(index) => {
                        i64::try_from(*index).map_or_else(|_| Val::undefined(), Val::int)
                    }
                },
                false,
            );
        }
        value
    }
    /// Whether execution used the supplied value.
    #[must_use]
    pub fn used(&self, path: &[PathSegment]) -> bool {
        self.at(path).0.borrow().used
    }
    /// Whether execution attempted array access or selectattr on this value.
    #[must_use]
    pub fn array_access(&self, path: &[PathSegment]) -> bool {
        self.at(path).0.borrow().array_access
    }
}
impl Template {
    /// Lex and parse one source template, retaining no caller-owned buffers.
    ///
    /// # Errors
    /// Returns an error for invalid syntax or templates exceeding resource bounds.
    pub fn parse(source: &str) -> Result<Self> {
        Ok(Self {
            nodes: ast::parse(source)?,
        })
    }
    /// Execute using an ordered JSON object, preserving strings as input ranges.
    ///
    /// # Errors
    /// Returns an error for malformed or oversized variable objects.
    pub fn execute(&self, variables_json: &str) -> Result<Execution> {
        self.execute_with_bytes(variables_json, &[])
    }
    /// Replace existing string variables with raw bytes before execution.
    ///
    /// # Errors
    /// Returns an error for invalid variables, missing overlay targets or bounds violations.
    pub fn execute_with_bytes(
        &self,
        variables_json: &str,
        overlays: &[(Vec<PathSegment>, Vec<u8>)],
    ) -> Result<Execution> {
        let globals = value::variables(variables_json)?;
        if overlays.len() > 16 {
            return Err(invalid("too many Jinja byte overlays"));
        }
        for (path, bytes) in overlays {
            if path.len() > 32 || bytes.len() > 65536 {
                return Err(invalid("Jinja byte overlay bound exceeded"));
            }
            let mut destination = globals.clone();
            for segment in path {
                destination = destination.get(
                    &match segment {
                        PathSegment::Key(key) => Val::string(key.clone()),
                        PathSegment::Index(index) => Val::int(i64::try_from(*index)?),
                    },
                    false,
                );
            }
            if !matches!(destination.kind(), Kind::String(_)) {
                return Err(invalid("Jinja byte overlay requires an existing string"));
            }
            destination.0.borrow_mut().kind = Kind::String(vec![Part {
                bytes: bytes.clone(),
                input: true,
            }]);
        }
        let scope = if let Kind::Object(values) = globals.kind() {
            values
                .into_iter()
                .map(|(key, value)| (String::from_utf8_lossy(&key.text()).into_owned(), value))
                .collect()
        } else {
            BTreeMap::new()
        };
        let mut runtime = Runtime {
            scopes: vec![scope],
            steps: 0,
            depth: 0,
            expression_depth: 0,
            output_bytes: 0,
        };
        let mut parts = Vec::new();
        let error = runtime
            .render(&self.nodes, &mut parts)
            .err()
            .map(|error| error.to_string());
        Ok(Execution {
            parts,
            globals,
            error,
        })
    }
    /// Observe the two content forms exposed by the pinned upstream probe.
    ///
    /// # Errors
    /// Returns an error if the probe variables cannot be prepared.
    pub fn content_capabilities(&self) -> Result<(bool, bool)> {
        caps::detect(|variables| {
            let execution = self.execute(variables)?;
            Ok(caps::Observation {
                rendered: execution.bytes(),
                success: execution.error.is_none(),
                content_array_access: execution.array_access(&[
                    PathSegment::Key("messages".into()),
                    PathSegment::Index(0),
                    PathSegment::Key("content".into()),
                ]),
            })
        })
    }
    /// Observe the six chat flags using the pinned native capability probes.
    ///
    /// Object/string tool arguments and parallel calls follow the source's
    /// fallback rules. Runtime exceptions are retained as probe observations;
    /// capability flags do not promise support outside the fixed probe inputs.
    ///
    /// # Errors
    /// Returns an error if the fixed probe variables cannot be prepared.
    pub fn chat_capabilities(&self) -> Result<ChatCapabilities> {
        caps::chat_capabilities(self)
    }
}
#[derive(Clone, Copy, PartialEq, Eq)]
enum Flow {
    Normal,
    Break,
    Continue,
}
struct Runtime {
    scopes: Vec<BTreeMap<String, Val>>,
    steps: usize,
    depth: usize,
    expression_depth: usize,
    output_bytes: usize,
}
impl Runtime {
    fn tick(&mut self) -> Result<()> {
        self.steps += 1;
        if self.steps > 1_000_000 {
            return Err(invalid("Jinja execution step bound exceeded"));
        }
        Ok(())
    }
    fn lookup(&self, name: &str) -> Val {
        for scope in self.scopes.iter().rev() {
            if let Some(value) = scope.get(name) {
                return value.clone();
            }
        }
        match name {
            "true" | "True" => Val::boolean(true),
            "false" | "False" => Val::boolean(false),
            "none" | "None" => Val::new(Kind::None),
            _ if builtins::is_global(name) => Val::new(Kind::Function(name.into())),
            _ => Val::undefined(),
        }
    }
    fn put(&mut self, name: String, value: Val) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(name, value);
        }
    }
    fn append(&mut self, result: &mut Vec<Part>, parts: Vec<Part>) -> Result<()> {
        self.output_bytes = self
            .output_bytes
            .checked_add(parts.iter().map(|part| part.bytes.len()).sum::<usize>())
            .ok_or_else(|| invalid("Jinja output overflow"))?;
        if self.output_bytes > 4 * 1024 * 1024 {
            return Err(invalid("Jinja output bound exceeded"));
        }
        result.extend(parts);
        Ok(())
    }
    fn assign(&mut self, target: &Expr, value: Val) -> Result<()> {
        match target {
            Expr::Var(name) => self.put(name.clone(), value),
            Expr::Tuple(targets) => {
                let values = value.elements()?;
                if targets.len() != values.len() {
                    return Err(invalid("Jinja assignment tuple size mismatch"));
                }
                for (target, value) in targets.iter().zip(values) {
                    self.assign(target, value)?;
                }
            }
            Expr::Attribute(container, key) => {
                let container = self.eval(container)?;
                container.set(Val::string(key.clone()), value)?;
            }
            Expr::Member(container, key) => {
                let container = self.eval(container)?;
                let key = self.eval(key)?;
                container.set(key, value)?;
            }
            _ => return Err(invalid("invalid Jinja assignment target")),
        }
        Ok(())
    }
    fn render(&mut self, nodes: &[Stmt], result: &mut Vec<Part>) -> Result<Flow> {
        self.depth += 1;
        if self.depth > 128 {
            return Err(invalid("Jinja execution nesting bound exceeded"));
        }
        let flow = self.render_inner(nodes, result);
        self.depth -= 1;
        flow
    }
    #[allow(
        clippy::too_many_lines,
        reason = "Keep statement variants and their scope/flow transitions in one exhaustive dispatch"
    )]
    fn render_inner(&mut self, nodes: &[Stmt], result: &mut Vec<Part>) -> Result<Flow> {
        for node in nodes {
            self.tick()?;
            match node {
                Stmt::Text(bytes) => self.append(
                    result,
                    vec![Part {
                        bytes: bytes.clone(),
                        input: false,
                    }],
                )?,
                Stmt::Output(expr) => {
                    let value = self.eval(expr)?;
                    self.append(result, value.parts())?;
                }
                Stmt::Set(target, expression, body) => {
                    let value = if let Some(expression) = expression {
                        self.eval(expression)?
                    } else {
                        let mut captured = Vec::new();
                        self.render(body, &mut captured)?;
                        Val::new(Kind::String(captured))
                    };
                    self.assign(target, value)?;
                }
                Stmt::If(branches, alternate) => {
                    let mut selected = alternate;
                    for (condition, body) in branches {
                        if self.eval(condition)?.truth() {
                            selected = body;
                            break;
                        }
                    }
                    let flow = self.render(selected, result)?;
                    if flow != Flow::Normal {
                        return Ok(flow);
                    }
                }
                Stmt::For(target, iterable, body, alternate) => {
                    let (expression, condition) =
                        if let Expr::Select(expression, condition) = iterable {
                            (&**expression, Some(&**condition))
                        } else {
                            (iterable, None)
                        };
                    let values = self.eval(expression)?.elements()?;
                    if values.len() > 65536 {
                        return Err(invalid("Jinja loop bound exceeded"));
                    }
                    let mut filtered = Vec::new();
                    self.scopes.push(BTreeMap::new());
                    for value in values {
                        self.assign(target, value.clone())?;
                        if condition.is_none()
                            || self.eval(condition.unwrap_or(expression))?.truth()
                        {
                            filtered.push(value);
                        }
                    }
                    if filtered.is_empty() {
                        self.render(alternate, result)?;
                    } else {
                        let length = filtered.len();
                        for (index, item) in filtered.into_iter().enumerate() {
                            self.assign(target, item)?;
                            let fields = vec![
                                ("index", Val::int(i64::try_from(index)? + 1)),
                                ("index0", Val::int(i64::try_from(index)?)),
                                ("revindex", Val::int(i64::try_from(length - index)?)),
                                ("revindex0", Val::int(i64::try_from(length - index - 1)?)),
                                ("first", Val::boolean(index == 0)),
                                ("last", Val::boolean(index + 1 == length)),
                                ("length", Val::int(i64::try_from(length)?)),
                            ];
                            let loop_value = Val::new(Kind::Object(
                                fields
                                    .into_iter()
                                    .map(|(key, value)| (Val::string(key), value))
                                    .collect(),
                            ));
                            loop_value.0.borrow_mut().namespace = true;
                            self.put("loop".into(), loop_value);
                            if self.render(body, result)? == Flow::Break {
                                break;
                            }
                        }
                    }
                    self.scopes.pop();
                }
                Stmt::Macro(name, arguments, body) => self.put(
                    name.clone(),
                    Val::new(Kind::Macro(arguments.clone(), body.clone())),
                ),
                Stmt::Call(arguments, call, body) => {
                    let captured = self.scopes.clone();
                    let previous = self.scopes.last_mut().and_then(|scope| {
                        scope.insert(
                            "caller".into(),
                            Val::new(Kind::Caller(arguments.clone(), body.clone(), captured)),
                        )
                    });
                    let value = self.eval(call)?;
                    if let Some(scope) = self.scopes.last_mut() {
                        if let Some(previous) = previous {
                            scope.insert("caller".into(), previous);
                        } else {
                            scope.remove("caller");
                        }
                    }
                    self.append(result, value.parts())?;
                }
                Stmt::Filter(filter, body) => {
                    let mut captured = Vec::new();
                    self.render(body, &mut captured)?;
                    let value = Val::new(Kind::String(captured));
                    let value = self.block_filter(filter, value)?;
                    self.append(result, value.parts())?;
                }
                Stmt::Break => return Ok(Flow::Break),
                Stmt::Continue => return Ok(Flow::Continue),
            }
        }
        Ok(Flow::Normal)
    }
    fn block_filter(&mut self, filter: &Expr, value: Val) -> Result<Val> {
        match filter {
            Expr::Var(name) => self.builtin(name, vec![value], BTreeMap::new()),
            Expr::Call(function, args) => {
                let Expr::Var(name) = &**function else {
                    return Err(invalid("invalid Jinja filter block"));
                };
                let (mut positional, keywords) = self.arguments(args)?;
                positional.insert(0, value);
                self.builtin(name, positional, keywords)
            }
            Expr::Filter(inner, name, args) => {
                let value = self.block_filter(inner, value)?;
                let (mut positional, keywords) = self.arguments(args)?;
                positional.insert(0, value);
                self.builtin(name, positional, keywords)
            }
            _ => Err(invalid("invalid Jinja filter block")),
        }
    }
    fn arguments(&mut self, args: &[Arg]) -> Result<(Vec<Val>, BTreeMap<String, Val>)> {
        let mut positional = Vec::new();
        let mut keywords = BTreeMap::new();
        for arg in args {
            match arg {
                Arg::Pos(expr) => positional.push(self.eval(expr)?),
                Arg::Keyword(name, expr) => {
                    if keywords.insert(name.clone(), self.eval(expr)?).is_some() {
                        return Err(invalid("duplicate Jinja keyword"));
                    }
                }
                Arg::Spread(expr) => positional.extend(self.eval(expr)?.elements()?),
            }
        }
        Ok((positional, keywords))
    }
    fn call(
        &mut self,
        function: &Val,
        positional: Vec<Val>,
        keywords: BTreeMap<String, Val>,
    ) -> Result<Val> {
        match function.kind() {
            Kind::Function(name) => self.builtin(&name, positional, keywords),
            Kind::Method(value, name) => self.method(value, &name, positional, keywords),
            Kind::Macro(parameters, body) => {
                if positional.len() > parameters.len() {
                    return Err(invalid("too many Jinja macro arguments"));
                }
                let mut values = BTreeMap::new();
                for (index, (name, default)) in parameters.iter().enumerate() {
                    let value = if let Some(value) = positional.get(index) {
                        if keywords.contains_key(name) {
                            return Err(invalid("duplicate Jinja macro argument"));
                        }
                        value.clone()
                    } else if let Some(value) = keywords.get(name) {
                        value.clone()
                    } else if let Some(default) = default {
                        self.eval(default)?
                    } else {
                        Val::undefined()
                    };
                    values.insert(name.clone(), value);
                }
                if keywords
                    .keys()
                    .any(|key| !parameters.iter().any(|(name, _)| name == key))
                {
                    return Err(invalid("unknown Jinja macro argument"));
                }
                self.scopes.push(values);
                let mut result = Vec::new();
                let rendered = self.render(&body, &mut result);
                self.scopes.pop();
                rendered?;
                Ok(Val::new(Kind::String(result)))
            }
            Kind::Caller(parameters, body, captured) => {
                let mut scope = BTreeMap::new();
                for (index, name) in parameters.iter().enumerate() {
                    scope.insert(
                        name.clone(),
                        positional
                            .get(index)
                            .cloned()
                            .unwrap_or_else(Val::undefined),
                    );
                }
                let previous = std::mem::replace(&mut self.scopes, captured);
                self.scopes.push(scope);
                let mut result = Vec::new();
                let rendered = self.render(&body, &mut result);
                self.scopes = previous;
                rendered?;
                Ok(Val::new(Kind::String(result)))
            }
            _ => Err(invalid("Jinja value is not callable")),
        }
    }
    fn eval(&mut self, expr: &Expr) -> Result<Val> {
        self.expression_depth += 1;
        if self.expression_depth > 128 {
            self.expression_depth -= 1;
            return Err(invalid("Jinja expression evaluation depth exceeded"));
        }
        let result = self.eval_inner(expr);
        self.expression_depth -= 1;
        let value = result?;
        value.check_graph()?;
        Ok(value)
    }
    #[allow(
        clippy::too_many_lines,
        reason = "Keep the AST's exhaustive expression dispatch together"
    )]
    fn eval_inner(&mut self, expr: &Expr) -> Result<Val> {
        self.tick()?;
        match expr {
            Expr::Literal(value) => Ok(match value {
                Literal::Int(value) => Val::int(*value),
                Literal::Float(value) => Val::new(Kind::Float(*value)),
                Literal::Bytes(value) => Val::bytes(value.clone(), false),
            }),
            Expr::Var(name) => Ok(self.lookup(name)),
            Expr::Blank => Ok(Val::undefined()),
            Expr::Array(values) | Expr::Tuple(values) => {
                let values = values
                    .iter()
                    .map(|value| self.eval(value))
                    .collect::<Result<_>>()?;
                Ok(Val::new(if matches!(expr, Expr::Array(_)) {
                    Kind::Array(values)
                } else {
                    Kind::Tuple(values)
                }))
            }
            Expr::Object(values) => {
                let result = Val::new(Kind::Object(Vec::new()));
                for (key, value) in values {
                    let key = self.eval(key)?;
                    let value = self.eval(value)?;
                    result.set(key, value)?;
                }
                Ok(result)
            }
            Expr::Attribute(container, name) => {
                let container = self.eval(container)?;
                if builtins::has_method(&container, name) {
                    return Ok(Val::new(Kind::Method(container, name.clone())));
                }
                Ok(container.get(&Val::string(name.clone()), true))
            }
            Expr::Member(container, key) => {
                let container = self.eval(container)?;
                if let Expr::Slice(start, stop, step) = &**key {
                    return self.slice(
                        &container,
                        start.as_deref(),
                        stop.as_deref(),
                        step.as_deref(),
                    );
                }
                let key = self.eval(key)?;
                Ok(container.get(&key, true))
            }
            Expr::Call(function, args) => {
                let function = self.eval(function)?;
                let (positional, keywords) = self.arguments(args)?;
                self.call(&function, positional, keywords)
            }
            Expr::Filter(value, name, args) => {
                let value = self.eval(value)?;
                let (mut positional, keywords) = self.arguments(args)?;
                positional.insert(0, value);
                self.builtin(name, positional, keywords)
            }
            Expr::Test(value, name, args, negative) => {
                let value = self.eval(value)?;
                let (positional, keywords) = self.arguments(args)?;
                if !keywords.is_empty() {
                    return Err(invalid("Jinja tests do not accept keyword arguments"));
                }
                Ok(Val::boolean(
                    self.test(&value, name, &positional)? != *negative,
                ))
            }
            Expr::Unary(operator, value) => {
                let value = self.eval(value)?;
                match operator.as_str() {
                    "not" => Ok(Val::boolean(!value.truth())),
                    "+" => {
                        value.number()?;
                        Ok(value)
                    }
                    "-" => {
                        if let Kind::Int(number) = value.kind() {
                            Ok(Val::int(number.checked_neg().ok_or_else(|| {
                                invalid("Jinja integer negation overflow")
                            })?))
                        } else {
                            Ok(Val::new(Kind::Float(-value.number()?)))
                        }
                    }
                    _ => Err(invalid("unknown Jinja unary operator")),
                }
            }
            Expr::Binary(operator, left, right) => {
                let left = self.eval(left)?;
                if operator == "or" {
                    return if left.truth() {
                        Ok(left)
                    } else {
                        self.eval(right)
                    };
                }
                if operator == "and" {
                    return if left.truth() {
                        self.eval(right)
                    } else {
                        Ok(left)
                    };
                }
                let right = self.eval(right)?;
                self.binary(operator, left, right)
            }
            Expr::Ternary(condition, yes, no) => {
                if self.eval(condition)?.truth() {
                    self.eval(yes)
                } else {
                    self.eval(no)
                }
            }
            Expr::Select(value, condition) => {
                if self.eval(condition)?.truth() {
                    self.eval(value)
                } else {
                    Ok(Val::undefined())
                }
            }
            Expr::Slice(..) => Err(invalid("Jinja slice outside indexing")),
        }
    }
    #[allow(
        clippy::too_many_lines,
        reason = "Keep checked operator dispatch and all operand type branches together"
    )]
    fn binary(&mut self, operator: &str, left: Val, right: Val) -> Result<Val> {
        self.tick()?;
        if matches!(operator, "+" | "-" | "*" | "%")
            && matches!(left.kind(), Kind::Int(_) | Kind::Bool(_))
            && matches!(right.kind(), Kind::Int(_) | Kind::Bool(_))
        {
            let a = left.integer()?;
            let b = right.integer()?;
            let result = match operator {
                "+" => a.checked_add(b),
                "-" => a.checked_sub(b),
                "*" => a.checked_mul(b),
                _ => a.checked_rem(b),
            }
            .ok_or_else(|| invalid("Jinja integer arithmetic overflow or division by zero"))?;
            return Ok(Val::int(result));
        }
        match operator {
            "==" => Ok(Val::boolean(value::equal(&left, &right))),
            "!=" => Ok(Val::boolean(!value::equal(&left, &right))),
            "in" | "not in" => Ok(Val::boolean(
                value::contains(&right, &left)? == (operator == "in"),
            )),
            "<" | ">" | "<=" | ">=" => {
                let comparison = value::compare(&left, &right)?;
                Ok(Val::boolean(match operator {
                    "<" => comparison.is_lt(),
                    ">" => comparison.is_gt(),
                    "<=" => !comparison.is_gt(),
                    _ => !comparison.is_lt(),
                }))
            }
            "~" => {
                let mut parts = left.parts();
                parts.extend(right.parts());
                Ok(Val::new(Kind::String(parts)))
            }
            "+" if matches!(left.kind(), Kind::String(_))
                && matches!(right.kind(), Kind::String(_)) =>
            {
                let mut parts = left.parts();
                parts.extend(right.parts());
                Ok(Val::new(Kind::String(parts)))
            }
            "+" if matches!(left.kind(), Kind::Array(_) | Kind::Tuple(_)) => {
                let tuple = matches!(left.kind(), Kind::Tuple(_));
                let mut values = left.elements()?;
                values.extend(right.elements()?);
                if values.len() > 65536 {
                    return Err(invalid("Jinja array bound exceeded"));
                }
                Ok(Val::new(if tuple {
                    Kind::Tuple(values)
                } else {
                    Kind::Array(values)
                }))
            }
            "*" if matches!(
                left.kind(),
                Kind::String(_) | Kind::Array(_) | Kind::Tuple(_)
            ) || matches!(
                right.kind(),
                Kind::String(_) | Kind::Array(_) | Kind::Tuple(_)
            ) =>
            {
                let (container, count) = if matches!(left.kind(), Kind::Int(_)) {
                    (right, left.integer()?)
                } else {
                    let count = right.integer()?;
                    (left, count)
                };
                let count =
                    usize::try_from(count.max(0)).map_err(|_| invalid("Jinja repetition bound"))?;
                if count.saturating_mul(container.len()?) > 65536 {
                    return Err(invalid("Jinja repetition bound exceeded"));
                }
                if matches!(container.kind(), Kind::String(_)) {
                    Ok(Val::new(Kind::String(
                        (0..count).flat_map(|_| container.parts()).collect(),
                    )))
                } else {
                    Ok(Val::new(Kind::Array(
                        (0..count)
                            .flat_map(|_| container.elements().unwrap_or_default())
                            .collect(),
                    )))
                }
            }
            "+" | "-" | "*" | "/" | "%" => {
                let a = left.number()?;
                let b = right.number()?;
                if matches!(operator, "/" | "%") && b == 0.0 {
                    return Err(invalid("Jinja division by zero"));
                }
                let result = match operator {
                    "+" => a + b,
                    "-" => a - b,
                    "*" => a * b,
                    "/" => a / b,
                    _ => a.rem_euclid(b),
                };
                if !result.is_finite() {
                    return Err(invalid("Jinja non-finite arithmetic"));
                }
                Ok(Val::new(Kind::Float(result)))
            }
            _ => Err(invalid("unknown Jinja binary operator")),
        }
    }
    fn slice(
        &mut self,
        container: &Val,
        start: Option<&Expr>,
        stop: Option<&Expr>,
        step: Option<&Expr>,
    ) -> Result<Val> {
        let values = container.elements()?;
        let length = i64::try_from(values.len())?;
        let step = if let Some(step) = step {
            self.eval(step)?.integer()?
        } else {
            1
        };
        if step == 0 {
            return Err(invalid("Jinja slice step cannot be zero"));
        }
        let bound = |value: i64| {
            if value < 0 {
                (length + value).clamp(
                    if step < 0 { -1 } else { 0 },
                    if step < 0 { length - 1 } else { length },
                )
            } else {
                value.clamp(
                    if step < 0 { -1 } else { 0 },
                    if step < 0 { length - 1 } else { length },
                )
            }
        };
        let mut index = if let Some(start) = start {
            bound(self.eval(start)?.integer()?)
        } else if step < 0 {
            length - 1
        } else {
            0
        };
        let stop = if let Some(stop) = stop {
            bound(self.eval(stop)?.integer()?)
        } else if step < 0 {
            -1
        } else {
            length
        };
        let mut result = Vec::new();
        while if step < 0 { index > stop } else { index < stop } {
            if let Some(value) = usize::try_from(index)
                .ok()
                .and_then(|index| values.get(index))
            {
                result.push(value.clone());
            }
            let Some(next) = index.checked_add(step) else {
                break;
            };
            index = next;
        }
        Ok(match container.kind() {
            Kind::String(_) => Val::new(Kind::String(
                result.into_iter().flat_map(|value| value.parts()).collect(),
            )),
            Kind::Tuple(_) => Val::new(Kind::Tuple(result)),
            _ => Val::new(Kind::Array(result)),
        })
    }
}
