// SPDX-License-Identifier: MIT
//! Safe execution of the immutable source-owned class/control program.

use super::super::{Result, invalid};
use super::number::Number;
use super::program::{Class, Expression, Geometry, Method, Parameter, Program, Statement};
use super::value::{Cell, Element, Lambda, Place, Scope, Value, variable};
use std::rc::Rc;

/// Production observations and source helpers supplied by the case controller.
pub(super) trait Host {
    fn call(&mut self, name: &str, arguments: &[Value]) -> Result<Option<Value>>;
    fn member(&mut self, value: &Value, name: &str) -> Result<Option<Value>>;
    fn write(
        &mut self,
        value: &Value,
        name: &str,
        index: Option<usize>,
        replacement: Value,
    ) -> Result<()>;
    fn force(&mut self, value: Value) -> Result<Value>;
}

#[derive(Debug)]
enum Flow {
    Next,
    Return(Value),
    Break,
    Continue,
}

enum Destination {
    Local(Place),
    Native {
        object: Value,
        name: String,
        index: Option<usize>,
    },
}

pub(super) struct Interpreter<'a> {
    pub(super) program: &'a Program,
    pub(super) host: &'a mut dyn Host,
    pub(super) fields: Scope,
    pub(super) scopes: Vec<Scope>,
    pub(super) class: &'a Class,
    steps: u64,
    depth: usize,
}

impl<'a> Interpreter<'a> {
    pub(super) fn new(program: &'a Program, class: &str, host: &'a mut dyn Host) -> Result<Self> {
        let class = program
            .class(class)
            .ok_or_else(|| invalid("source class is missing"))?;
        Ok(Self {
            program,
            host,
            fields: Scope::new(),
            scopes: vec![Scope::new()],
            class,
            steps: 0,
            depth: 0,
        })
    }

    fn step(&mut self) -> Result<()> {
        self.steps += 1;
        if self.steps > 2_000_000_000 {
            return Err(invalid(
                "source interpreter exceeds bounded instruction extent",
            ));
        }
        Ok(())
    }

    pub(super) fn cell(&self, name: &str) -> Option<Cell> {
        self.scopes
            .iter()
            .rev()
            .find_map(|scope| scope.get(name))
            .or_else(|| self.fields.get(name))
            .cloned()
    }

    pub(super) fn value(&self, name: &str) -> Result<Value> {
        if name == "this" {
            return Ok(Value::This);
        }
        if let Some(cell) = self.cell(name) {
            let value = cell.borrow().value.clone();
            return match value {
                Value::Reference(place) => place.get().map_err(invalid),
                value => Ok(value),
            };
        }
        if let Some(value) = self.program.constants.get(name) {
            return super::standard::literal(value, None);
        }
        Ok(Value::Function(name.into()))
    }

    pub(super) fn forced(&mut self, value: Value) -> Result<Value> {
        match value {
            Value::Reference(place) => self.forced(place.get().map_err(invalid)?),
            Value::NativeInteger { .. } => self.host.force(value),
            value => Ok(value),
        }
    }

    fn condition(&mut self, expression: &Expression) -> Result<bool> {
        let value = self.eval(expression)?;
        self.forced(value)?.truth().map_err(invalid)
    }

    pub(super) fn instantiate(&mut self, geometry: &Geometry) -> Result<()> {
        let arguments = geometry
            .arguments
            .iter()
            .map(|value| super::standard::literal(value, None))
            .collect::<Result<Vec<_>>>()?;
        self.construct(self.class, arguments)
    }

    fn bind(&mut self, parameters: &[Parameter], arguments: Vec<Value>) -> Result<()> {
        if arguments.len() > parameters.len() {
            return Err(invalid("source method argument count exceeds signature"));
        }
        let mut arguments = arguments.into_iter();
        for parameter in parameters {
            let value = if let Some(value) = arguments.next() {
                value
            } else if let Some(default) = &parameter.default {
                self.eval(default)?
            } else {
                return Err(invalid(format!(
                    "source parameter {} has no argument",
                    parameter.name
                )));
            };
            let kind = self.resolved_type(&parameter.kind)?;
            let value = value.coerce(&kind).map_err(invalid)?;
            self.scopes
                .last_mut()
                .ok_or_else(|| invalid("source parameter scope is absent"))?
                .insert(parameter.name.clone(), variable(kind, value));
        }
        Ok(())
    }

    fn construct(&mut self, class: &'a Class, arguments: Vec<Value>) -> Result<()> {
        self.scopes.push(Scope::new());
        let result = self.construct_inner(class, arguments);
        let _ = self.scopes.pop();
        result
    }

    fn construct_inner(&mut self, class: &'a Class, arguments: Vec<Value>) -> Result<()> {
        self.bind(&class.constructor.parameters, arguments)?;
        if let Some(base) = self.program.class(&class.base) {
            let initializer = class
                .constructor
                .initializers
                .iter()
                .find(|entry| entry.name == base.name);
            let mut arguments = Vec::new();
            if let Some(initializer) = initializer {
                for argument in &initializer.arguments {
                    arguments.push(self.eval(argument)?);
                }
            }
            self.construct(base, arguments)?;
        }
        self.construct_fields(class, &class.fields)?;
        match self.statements(&class.constructor.statements)? {
            Flow::Next => Ok(()),
            _ => Err(invalid("unexpected source constructor control flow")),
        }
    }

    fn construct_fields(&mut self, class: &'a Class, fields: &[Statement]) -> Result<()> {
        for field in fields {
            match field {
                Statement::Empty => continue,
                Statement::Declarations { body } => {
                    self.construct_fields(class, body)?;
                    continue;
                }
                _ => {}
            }
            let Statement::Declare {
                name,
                kind,
                dimensions,
                value,
            } = field
            else {
                return Err(invalid("source field is not a declaration"));
            };
            let initializer = class
                .constructor
                .initializers
                .iter()
                .find(|entry| entry.name == *name);
            let replacement = if let Some(initializer) = initializer {
                if initializer.arguments.len() != 1 {
                    return Err(invalid("source member initializer arity differs"));
                }
                Some(self.eval(&initializer.arguments[0])?)
            } else {
                value.as_ref().map(|value| self.eval(value)).transpose()?
            };
            let (kind, value) = self.declaration_value(kind, dimensions, replacement)?;
            self.fields.insert(name.clone(), variable(kind, value));
        }
        Ok(())
    }

    pub(super) fn resolved_type(&self, kind: &str) -> Result<String> {
        let mut resolved = kind.to_owned();
        if resolved.contains("GGML_MAX_OP_PARAMS/sizeof(int32_t)") {
            let count = self
                .program
                .constants
                .get("GGML_MAX_OP_PARAMS")
                .and_then(serde_json::Value::as_u64)
                .ok_or_else(|| invalid("source op parameter extent missing"))?
                / 4;
            resolved = resolved.replace("GGML_MAX_OP_PARAMS/sizeof(int32_t)", &count.to_string());
        }
        for name in ["GGML_MAX_SRC", "GGML_MAX_DIMS"] {
            if let Some(value) = self.program.constants.get(name) {
                resolved = resolved.replace(name, &value.to_string());
            }
        }
        Ok(resolved)
    }

    fn declaration_value(
        &mut self,
        kind: &str,
        dimensions: &[Option<Expression>],
        value: Option<Value>,
    ) -> Result<(String, Value)> {
        let mut kind = self.resolved_type(kind)?;
        let value = if dimensions.is_empty() {
            value.unwrap_or_else(|| super::standard::zero(&kind))
        } else {
            if dimensions.len() != 1 {
                return Err(invalid("unexpected multidimensional source C array"));
            }
            let count = if let Some(dimension) = &dimensions[0] {
                let count = self.eval(dimension)?;
                self.forced(count)?.number().map_err(invalid)?.as_u64() as usize
            } else {
                value
                    .as_ref()
                    .ok_or_else(|| invalid("source inferred array lacks initializer"))?
                    .len()
                    .map_err(invalid)?
            };
            kind = format!("std::array<{kind},{count}>");
            value.unwrap_or_else(|| Value::array(Vec::new()))
        };
        let clean = kind
            .replace("constexpr", "")
            .replace("const", "")
            .replace("static", "");
        let value = if (clean == "bool" || Element::parse(&clean).is_ok())
            && let Value::Array(values) = &value
        {
            match values.borrow().as_slice() {
                [] => super::standard::zero(&kind),
                [value] => value.clone(),
                _ => {
                    return Err(invalid(
                        "source scalar brace initializer has multiple values",
                    ));
                }
            }
        } else {
            value
        };
        let value = if clean == "std::string" {
            let value = self.forced(value)?;
            super::standard::call(self, "std::string", &[value])?
                .ok_or_else(|| invalid("source string constructor unavailable"))?
        } else if kind.contains('&') {
            value
        } else {
            value.copy_value()
        };
        let value = value.coerce(&kind).map_err(invalid)?;
        Ok((kind, value))
    }

    fn method(&self, name: &str, count: usize) -> Option<&'a Method> {
        self.program.method(&self.class.name, name, count)
    }

    fn arguments(
        &mut self,
        parameters: &[Parameter],
        expressions: &[Expression],
    ) -> Result<Vec<Value>> {
        let mut values = Vec::with_capacity(expressions.len());
        for (index, expression) in expressions.iter().enumerate() {
            let reference = parameters.get(index).is_some_and(|parameter| {
                parameter.kind.contains('&') && !parameter.kind.contains("const")
            });
            if reference {
                let Destination::Local(place) = self.destination(expression)? else {
                    return Err(invalid("source reference parameter is not local storage"));
                };
                values.push(Value::Reference(place));
            } else {
                let value = self.eval(expression)?;
                values.push(self.forced(value)?.copy_value());
            }
        }
        Ok(values)
    }

    fn invoke(
        &mut self,
        method: &Method,
        arguments: Vec<Value>,
        captured: Option<Scope>,
    ) -> Result<Value> {
        self.depth += 1;
        if self.depth > 128 {
            return Err(invalid("source call depth exceeds bound"));
        }
        let saved = std::mem::replace(&mut self.scopes, vec![captured.unwrap_or_default()]);
        let result = (|| {
            self.bind(&method.parameters, arguments)?;
            match self.statements(&method.statements)? {
                Flow::Return(value) => Ok(value),
                Flow::Next => Ok(Value::Null),
                _ => Err(invalid("source loop control escaped a function")),
            }
        })();
        self.scopes = saved;
        self.depth -= 1;
        result.map_err(|error| invalid(format!("{}::{}: {error}", self.class.name, method.name)))
    }

    pub(super) fn call(&mut self, name: &str, arguments: Vec<Value>) -> Result<Value> {
        if let Some(method) = self.method(name, arguments.len()) {
            return self.invoke(method, arguments, None);
        }
        if let Some(method) = self
            .program
            .helpers
            .iter()
            .find(|method| method.name == name)
        {
            return self.invoke(method, arguments, None);
        }
        if name == "build_graph" && arguments.len() == 2 {
            return self.call("build_graph", arguments[..1].to_vec());
        }
        if name == "max_err" {
            return self.call("max_nmse_err", arguments);
        }
        if name == "max_nmse_err" && arguments.len() == 1 {
            return self.call("max_nmse_err", Vec::new());
        }
        if name == "op_desc" {
            return self.call("ggml_op_desc", arguments);
        }
        if let Some(value) = super::standard::call(self, name, &arguments)? {
            return Ok(value);
        }
        self.host
            .call(name, &arguments)?
            .ok_or_else(|| invalid(format!("source call {name} has no safe implementation")))
    }

    fn call_expression(&mut self, callee: &Expression, arguments: &[Expression]) -> Result<Value> {
        if let Expression::Id { name } = callee {
            if name == "sizeof" {
                return super::standard::size_of(self, arguments);
            }
            if name.starts_with("VARS_TO_STR") || name == "VAR_TO_STR" {
                return super::standard::vars(self, arguments);
            }
            if let Some(method) = self.method(name, arguments.len()).or_else(|| {
                self.program
                    .helpers
                    .iter()
                    .find(|method| method.name == *name)
            }) {
                let values = self.arguments(&method.parameters, arguments)?;
                return self.invoke(method, values, None);
            }
        }
        if let Expression::Member { object, name } = callee {
            let object = self.eval(object)?;
            let values = self.arguments(&[], arguments)?;
            return super::standard::member_call(self, object, name, values);
        }
        let function = self.eval(callee)?;
        match function {
            Value::Function(name) => {
                let values = self.arguments(&[], arguments)?;
                self.call(&name, values)
            }
            Value::Lambda(lambda) => {
                let values = self.arguments(&lambda.parameters, arguments)?;
                let method = Method {
                    name: "<lambda>".into(),
                    parameters: lambda.parameters.clone(),
                    statements: lambda.statements.clone(),
                    source_start: 0,
                    source_end: 0,
                };
                self.invoke(&method, values, Some(lambda.scope.clone()))
            }
            value => {
                let values = self.arguments(&[], arguments)?;
                super::standard::call_value(self, value, &values)
            }
        }
    }

    fn destination(&mut self, expression: &Expression) -> Result<Destination> {
        match expression {
            Expression::Id { name } => {
                let cell = self.cell(name).ok_or_else(|| {
                    invalid(format!("source assignment to unknown variable {name}"))
                })?;
                let value = cell.borrow().value.clone();
                Ok(Destination::Local(if let Value::Reference(place) = value {
                    place
                } else {
                    Place::Variable(cell)
                }))
            }
            Expression::Index { object, index } => {
                let index = self.eval(index)?;
                let index = self.forced(index)?.number().map_err(invalid)?.as_u64() as usize;
                if let Expression::Member { object: base, name } = object.as_ref() {
                    let base = self.eval(base)?;
                    if matches!(base, Value::Handle(_)) {
                        return Ok(Destination::Native {
                            object: base,
                            name: name.clone(),
                            index: Some(index),
                        });
                    }
                }
                let object = self.eval(object)?;
                Ok(Destination::Local(object.place(index).map_err(invalid)?))
            }
            Expression::Member { object, name } => {
                let object = self.eval(object)?;
                if matches!(object, Value::This) {
                    let cell =
                        self.fields.get(name).cloned().ok_or_else(|| {
                            invalid(format!("explicit this field {name} is absent"))
                        })?;
                    return Ok(Destination::Local(Place::Variable(cell)));
                }
                Ok(Destination::Native {
                    object,
                    name: name.clone(),
                    index: None,
                })
            }
            Expression::Unary { op, value } if op == "*" => {
                let value = self.eval(value)?;
                Ok(Destination::Local(value.place(0).map_err(invalid)?))
            }
            _ => Err(invalid(
                "source expression is not an assignment destination",
            )),
        }
    }

    fn write(&mut self, destination: Destination, value: Value) -> Result<()> {
        match destination {
            Destination::Local(place) => place.set(value.copy_value()).map_err(invalid),
            Destination::Native {
                object,
                name,
                index,
            } => self.host.write(&object, &name, index, value),
        }
    }

    fn unary(&mut self, op: &str, expression: &Expression) -> Result<Value> {
        if op == "&" {
            let Destination::Local(place) = self.destination(expression)? else {
                return Err(invalid("source address requires local storage"));
            };
            return match place {
                Place::Array(values, index) => Ok(Value::Pointer {
                    value: Box::new(Value::Array(values)),
                    index,
                }),
                Place::Buffer(buffer, index) => Ok(Value::Pointer {
                    value: Box::new(Value::Buffer(buffer)),
                    index,
                }),
                Place::Variable(cell) => Ok(Value::Address(Place::Variable(cell))),
            };
        }
        let value = self.eval(expression)?;
        let value = self.forced(value)?;
        match op {
            "*" => value.index(0).map_err(invalid),
            "!" => Ok(Value::Bool(!value.truth().map_err(invalid)?)),
            "-" => Ok(Value::Number(value.number().map_err(invalid)?.negate())),
            "++" => {
                let value = self.binary_value("+", value, Value::Number(Number::I32(1)))?;
                let destination = self.destination(expression)?;
                self.write(destination, value.clone())?;
                Ok(value)
            }
            _ => Err(invalid(format!("unknown source unary operator {op}"))),
        }
    }

    fn binary(&mut self, op: &str, left: &Expression, right: &Expression) -> Result<Value> {
        if op == "&&" {
            return Ok(Value::Bool(self.condition(left)? && self.condition(right)?));
        }
        if op == "||" {
            return Ok(Value::Bool(self.condition(left)? || self.condition(right)?));
        }
        if ["=", "+=", "*=", "/=", "^=", ">>="].contains(&op) {
            let value = self.eval(right)?;
            let value = if op == "=" {
                value
            } else {
                let old = self.eval(left)?;
                self.binary_value(op.trim_end_matches('='), old, value)?
            };
            let destination = self.destination(left)?;
            self.write(destination, value.clone())?;
            return Ok(value);
        }
        let a = self.eval(left)?;
        let b = self.eval(right)?;
        self.binary_value(op, a, b)
    }

    pub(super) fn binary_value(&mut self, op: &str, a: Value, b: Value) -> Result<Value> {
        let a = self.forced(a)?;
        let b = self.forced(b)?;
        if matches!(
            (&a, &b),
            (
                Value::Handle(super::value::Handle::Tensor(_)),
                Value::Handle(super::value::Handle::Tensor(_))
            )
        ) && matches!(op, "==" | "!=")
        {
            let value = self
                .host
                .call("pointer-equal", &[a, b])?
                .ok_or_else(|| invalid("source tensor identity observation missing"))?;
            let equal = value.truth().map_err(invalid)?;
            return Ok(Value::Bool(if op == "==" { equal } else { !equal }));
        }
        if let Some(value) = super::standard::binary(op, &a, &b)? {
            return Ok(value);
        }
        let a = a.number().map_err(invalid)?;
        let b = b.number().map_err(invalid)?;
        if ["==", "!=", "<", "<=", ">", ">="].contains(&op) {
            return Ok(Value::Bool(match op {
                "==" => a.equal(b),
                "!=" => !a.equal(b),
                "<" => a.less(b),
                "<=" => a.less(b) || a.equal(b),
                ">" => b.less(a),
                ">=" => b.less(a) || a.equal(b),
                _ => false,
            }));
        }
        a.binary(op, b).map(Value::Number).map_err(invalid)
    }

    pub(super) fn eval(&mut self, expression: &Expression) -> Result<Value> {
        self.step()?;
        match expression {
            Expression::Literal { value, number_kind } => {
                super::standard::literal(value, number_kind.as_deref())
            }
            Expression::Bytes { bytes } => Ok(Value::Bytes(bytes.clone())),
            Expression::Id { name } => self.value(name),
            Expression::Array { values } => {
                let mut output = Vec::with_capacity(values.len());
                for value in values {
                    output.push(self.eval(value)?);
                }
                Ok(Value::array(output))
            }
            Expression::Unary { op, value } => self.unary(op, value),
            Expression::Binary { op, left, right } => self.binary(op, left, right),
            Expression::Member { object, name } => {
                let object = self.eval(object)?;
                if matches!(object, Value::This) {
                    let cell = self
                        .fields
                        .get(name)
                        .ok_or_else(|| invalid(format!("explicit this field {name} is absent")))?;
                    Ok(cell.borrow().value.clone())
                } else if let Some(value) = super::standard::member(&object, name)? {
                    Ok(value)
                } else {
                    self.host
                        .member(&object, name)?
                        .ok_or_else(|| invalid(format!("unknown source member {name}")))
                }
            }
            Expression::Index { object, index } => {
                let object = self.eval(object)?;
                let index = self.eval(index)?;
                let index = self.forced(index)?.number().map_err(invalid)?.as_u64() as usize;
                object.index(index).map_err(invalid)
            }
            Expression::Call { callee, args } => self.call_expression(callee, args),
            Expression::Ternary { condition, yes, no } => {
                if self.condition(condition)? {
                    self.eval(yes)
                } else {
                    self.eval(no)
                }
            }
            Expression::Cast { kind, value } => {
                let value = self.eval(value)?;
                let value = self.forced(value)?;
                super::standard::cast(value, kind)
            }
            Expression::Lambda { params, body } => {
                let mut scope = self.fields.clone();
                for local in &self.scopes {
                    scope.extend(local.clone());
                }
                Ok(Value::Lambda(Rc::new(Lambda {
                    parameters: params.clone(),
                    statements: body.clone(),
                    scope,
                })))
            }
            Expression::New { value } => self.eval(value),
        }
    }

    fn scoped(&mut self, statements: &[Statement]) -> Result<Flow> {
        self.scopes.push(Scope::new());
        let result = self.statements(statements);
        let _ = self.scopes.pop();
        result
    }

    fn statements(&mut self, statements: &[Statement]) -> Result<Flow> {
        for statement in statements {
            let flow = self.statement(statement)?;
            if !matches!(flow, Flow::Next) {
                return Ok(flow);
            }
        }
        Ok(Flow::Next)
    }

    fn loop_body(&mut self, body: &[Statement]) -> Result<Option<Flow>> {
        match self.scoped(body)? {
            Flow::Next | Flow::Continue => Ok(None),
            Flow::Break => Ok(Some(Flow::Next)),
            flow @ Flow::Return(_) => Ok(Some(flow)),
        }
    }

    #[allow(
        clippy::too_many_lines,
        reason = "Each branch directly implements one source statement kind in a single bounded control-flow dispatch."
    )]
    fn statement(&mut self, statement: &Statement) -> Result<Flow> {
        self.step()?;
        if super::numeric_loop::execute(self, statement)? {
            return Ok(Flow::Next);
        }
        match statement {
            Statement::Empty => {}
            Statement::Declare {
                name,
                kind,
                dimensions,
                value,
            } => {
                let value = value.as_ref().map(|value| self.eval(value)).transpose()?;
                let (kind, value) = self.declaration_value(kind, dimensions, value)?;
                self.scopes
                    .last_mut()
                    .ok_or_else(|| invalid("source declaration scope absent"))?
                    .insert(name.clone(), variable(kind, value));
            }
            Statement::Declarations { body } => return self.statements(body),
            Statement::Expression { value } => {
                let _ = self.eval(value)?;
            }
            Statement::Return { value } => {
                return Ok(Flow::Return(
                    value
                        .as_ref()
                        .map(|value| self.eval(value))
                        .transpose()?
                        .unwrap_or(Value::Null),
                ));
            }
            Statement::If {
                condition,
                body,
                otherwise,
            } => {
                return if self.condition(condition)? {
                    self.scoped(body)
                } else {
                    self.scoped(otherwise)
                };
            }
            Statement::Scope { body } => return self.scoped(body),
            Statement::For {
                initialization,
                condition,
                increment,
                body,
            } => {
                self.scopes.push(Scope::new());
                let result = (|| {
                    let flow = self.statements(initialization)?;
                    if !matches!(flow, Flow::Next) {
                        return Ok(flow);
                    }
                    while self.condition(condition)? {
                        if let Some(flow) = self.loop_body(body)? {
                            return Ok(flow);
                        }
                        let _ = self.eval(increment)?;
                    }
                    Ok(Flow::Next)
                })();
                let _ = self.scopes.pop();
                return result;
            }
            Statement::While { condition, body } => {
                while self.condition(condition)? {
                    if let Some(flow) = self.loop_body(body)? {
                        return Ok(flow);
                    }
                }
            }
            Statement::Do { condition, body } => loop {
                if let Some(flow) = self.loop_body(body)? {
                    return Ok(flow);
                }
                if !self.condition(condition)? {
                    break;
                }
            },
            Statement::RangeFor {
                name,
                kind,
                value,
                body,
            } => {
                let values = self.eval(value)?;
                for index in 0..values.len().map_err(invalid)? {
                    let value = if kind.contains('&') {
                        Value::Reference(values.place(index).map_err(invalid)?)
                    } else {
                        values.index(index).map_err(invalid)?.copy_value()
                    };
                    self.scopes
                        .push(Scope::from([(name.clone(), variable(kind.clone(), value))]));
                    let flow = self.statements(body);
                    let _ = self.scopes.pop();
                    match flow? {
                        Flow::Next | Flow::Continue => {}
                        Flow::Break => break,
                        flow @ Flow::Return(_) => return Ok(flow),
                    }
                }
            }
            Statement::Switch { value, cases } => {
                let value = self.eval(value)?;
                let mut start = None;
                let mut default = None;
                for (index, case) in cases.iter().enumerate() {
                    for label in &case.labels {
                        if let Some(label) = label {
                            let label = self.eval(label)?;
                            if self
                                .binary_value("==", value.clone(), label)?
                                .truth()
                                .map_err(invalid)?
                            {
                                start = Some(index);
                            }
                        } else {
                            default = Some(index);
                        }
                    }
                }
                if let Some(index) = start.or(default) {
                    self.scopes.push(Scope::new());
                    let result = (|| {
                        for case in &cases[index..] {
                            match self.statements(&case.body)? {
                                Flow::Next => {}
                                Flow::Break => return Ok(Flow::Next),
                                flow => return Ok(flow),
                            }
                        }
                        Ok(Flow::Next)
                    })();
                    let _ = self.scopes.pop();
                    return result;
                }
            }
            Statement::Break => return Ok(Flow::Break),
            Statement::Continue => return Ok(Flow::Continue),
        }
        Ok(Flow::Next)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    struct LanguageOnly;
    impl Host for LanguageOnly {
        fn call(&mut self, _name: &str, _arguments: &[Value]) -> Result<Option<Value>> {
            Err(invalid(
                "synthetic language guard must not invoke a production API",
            ))
        }
        fn member(&mut self, _value: &Value, _name: &str) -> Result<Option<Value>> {
            Err(invalid(
                "synthetic language guard must not observe production metadata",
            ))
        }
        fn write(
            &mut self,
            _value: &Value,
            _name: &str,
            _index: Option<usize>,
            _replacement: Value,
        ) -> Result<()> {
            Err(invalid(
                "synthetic language guard must not mutate production metadata",
            ))
        }
        fn force(&mut self, value: Value) -> Result<Value> {
            Ok(value)
        }
    }
    fn id(name: &str) -> Expression {
        Expression::Id { name: name.into() }
    }
    fn literal(value: i32) -> Expression {
        Expression::Literal {
            value: json!(value),
            number_kind: Some("integer".into()),
        }
    }

    #[test]
    fn source_shadowed_members_and_copied_view_shapes() -> Result<()> {
        let (program, _) = super::super::program::load().map_err(invalid)?;
        let mut host = LanguageOnly;
        let mut runtime = Interpreter::new(program, "test_soft_max", &mut host)?;
        runtime
            .fields
            .insert("mask".into(), variable("bool".into(), Value::Bool(true)));
        runtime.scopes[0].insert("mask".into(), variable("ggml_tensor*".into(), Value::Null));
        let member = Expression::Member {
            object: Box::new(id("this")),
            name: "mask".into(),
        };
        assert!(runtime.eval(&member)?.truth().map_err(invalid)?);
        assert!(!runtime.eval(&id("mask"))?.truth().map_err(invalid)?);
        let _ = runtime.eval(&Expression::Binary {
            op: "=".into(),
            left: Box::new(member),
            right: Box::new(Expression::Literal {
                value: json!(false),
                number_kind: None,
            }),
        })?;
        assert!(
            !runtime.fields["mask"]
                .borrow()
                .value
                .truth()
                .map_err(invalid)?
        );
        assert!(matches!(
            runtime.scopes[0]["mask"].borrow().value,
            Value::Null
        ));
        let shape = Value::array(vec![
            Value::Number(Number::I64(2)),
            Value::Number(Number::I64(3)),
            Value::Number(Number::I64(1)),
            Value::Number(Number::I64(1)),
        ])
        .coerce("std::array<int64_t,4>")
        .map_err(invalid)?;
        runtime.fields.insert(
            "ne_a".into(),
            variable("conststd::array<int64_t,4>".into(), shape),
        );
        let _ = runtime.statement(&Statement::Declare {
            name: "ne".into(),
            kind: "auto".into(),
            dimensions: Vec::new(),
            value: Some(id("ne_a")),
        })?;
        let _ = runtime.eval(&Expression::Binary {
            op: "*=".into(),
            left: Box::new(Expression::Index {
                object: Box::new(id("ne")),
                index: Box::new(literal(0)),
            }),
            right: Box::new(literal(3)),
        })?;
        assert_eq!(
            runtime
                .value("ne_a")?
                .index(0)
                .map_err(invalid)?
                .number()
                .map_err(invalid)?,
            Number::I64(2)
        );
        assert_eq!(
            runtime
                .value("ne")?
                .index(0)
                .map_err(invalid)?
                .number()
                .map_err(invalid)?,
            Number::I64(6)
        );
        Ok(())
    }

    #[test]
    fn grouped_and_empty_constructor_fields_preserve_member_initialization_order() -> Result<()> {
        let class: Class = serde_json::from_value(json!({
            "Name":"grouped_fields", "Base":"",
            "Fields":[
                {"k":"empty"},
                {"k":"decls","body":[
                    {"k":"decl","name":"a","type":"int","dimensions":[],"value":{"k":"literal","value":99}},
                    {"k":"decls","body":[
                        {"k":"empty"},
                        {"k":"decl","name":"b","type":"int","dimensions":[],"value":null}
                    ]}
                ]},
                {"k":"decl","name":"c","type":"int","dimensions":[],"value":{"k":"binary","op":"+","left":{"k":"id","name":"b"},"right":{"k":"literal","value":2}}},
                {"k":"empty"},
                {"k":"decl","name":"d","type":"int","dimensions":[],"value":{"k":"binary","op":"+","left":{"k":"id","name":"c"},"right":{"k":"literal","value":3}}}
            ],
            "Constructor":{
                "Parameters":[{"Name":"seed","Type":"int","Default":null}],
                "Initializers":[
                    {"Name":"b","Arguments":[{"k":"binary","op":"+","left":{"k":"id","name":"a"},"right":{"k":"literal","value":1}}]},
                    {"Name":"a","Arguments":[{"k":"id","name":"seed"}]}
                ],
                "Statements":[]
            },
            "Methods":[]
        }))?;
        let (program, _) = super::super::program::load().map_err(invalid)?;
        let mut host = LanguageOnly;
        let mut runtime = Interpreter::new(program, "test_soft_max", &mut host)?;
        runtime.construct(&class, vec![Value::Number(Number::I32(11))])?;
        for (name, expected) in [("a", 11), ("b", 12), ("c", 14), ("d", 17)] {
            assert_eq!(
                runtime.value(name)?.number().map_err(invalid)?,
                Number::I32(expected)
            );
        }
        Ok(())
    }

    #[test]
    fn scalar_brace_initializers_keep_arrays_and_data_pointers_distinct() -> Result<()> {
        let (program, geometries) = super::super::program::load().map_err(invalid)?;
        let geometry = geometries
            .eval
            .iter()
            .find(|geometry| geometry.class == "test_rope_set_rows")
            .ok_or_else(|| invalid("original RoPE set-rows geometry missing"))?;
        let mut host = LanguageOnly;
        let mut runtime = Interpreter::new(program, &geometry.class, &mut host)?;
        runtime.instantiate(geometry)?;
        assert_eq!(
            runtime.value("n_ctx")?.number().map_err(invalid)?,
            Number::I32(512)
        );
        assert_eq!(
            runtime.value("n_dims")?.number().map_err(invalid)?,
            Number::I32(128)
        );
        let section = runtime.eval(&Expression::Binary {
            op: "/".into(),
            left: Box::new(id("n_dims")),
            right: Box::new(literal(3)),
        })?;
        assert_eq!(section.number().map_err(invalid)?, Number::I32(42));
        let one = || Value::array(vec![Value::Number(Number::I32(128))]);
        let (_, scalar) = runtime.declaration_value("constint64_t", &[], Some(one()))?;
        assert_eq!(scalar.number().map_err(invalid)?, Number::I64(128));
        let (_, zero) = runtime.declaration_value("float", &[], Some(Value::array(Vec::new())))?;
        assert_eq!(zero.number().map_err(invalid)?, Number::F32(0.0));
        assert!(
            runtime
                .declaration_value(
                    "int",
                    &[],
                    Some(Value::array(vec![
                        Value::Number(Number::I32(1)),
                        Value::Number(Number::I32(2))
                    ]))
                )
                .is_err()
        );
        let (_, array) = runtime.declaration_value("std::array<int64_t,1>", &[], Some(one()))?;
        assert_eq!(array.len().map_err(invalid)?, 1);
        assert!(array.number().is_err());
        let (_, c_array) =
            runtime.declaration_value("int64_t", &[Some(literal(1))], Some(one()))?;
        assert_eq!(c_array.len().map_err(invalid)?, 1);
        assert!(c_array.number().is_err());
        let (_, vector) = runtime.declaration_value("std::vector<int>", &[], Some(one()))?;
        assert_eq!(vector.len().map_err(invalid)?, 1);
        let (_, deduced) = runtime.declaration_value("auto", &[], Some(one()))?;
        assert_eq!(deduced.len().map_err(invalid)?, 1);
        let (_, pointer) = runtime.declaration_value(
            "int64_t*",
            &[],
            Some(Value::Pointer {
                value: Box::new(c_array.clone()),
                index: 0,
            }),
        )?;
        pointer
            .place(0)
            .map_err(invalid)?
            .set(Value::Number(Number::I64(256)))
            .map_err(invalid)?;
        assert_eq!(
            c_array
                .index(0)
                .map_err(invalid)?
                .number()
                .map_err(invalid)?,
            Number::I64(256)
        );
        Ok(())
    }
    struct DeferredName {
        observed: Rc<std::cell::Cell<u32>>,
    }
    impl Host for DeferredName {
        fn call(&mut self, name: &str, arguments: &[Value]) -> Result<Option<Value>> {
            LanguageOnly.call(name, arguments)
        }
        fn member(&mut self, value: &Value, name: &str) -> Result<Option<Value>> {
            LanguageOnly.member(value, name)
        }
        fn write(
            &mut self,
            value: &Value,
            name: &str,
            index: Option<usize>,
            replacement: Value,
        ) -> Result<()> {
            LanguageOnly.write(value, name, index, replacement)
        }
        fn force(&mut self, value: Value) -> Result<Value> {
            if let Value::NativeInteger { argument, .. } = value {
                if argument != json!({"Scalar":1}) {
                    return Err(invalid("non-string initializer must remain deferred"));
                }
                self.observed.set(self.observed.get() + 1);
                Ok(Value::Pointer {
                    value: Box::new(Value::Bytes(b"--scale\0suffix".to_vec())),
                    index: 2,
                })
            } else {
                Ok(value)
            }
        }
    }

    #[test]
    fn owning_source_string_copies_deferred_c_string_without_forcing_pointers() -> Result<()> {
        let (program, _) = super::super::program::load().map_err(invalid)?;
        let observed = Rc::new(std::cell::Cell::new(0));
        let mut host = DeferredName {
            observed: Rc::clone(&observed),
        };
        let mut runtime = Interpreter::new(program, "test_dsv4_hc", &mut host)?;
        let method = runtime
            .class
            .methods
            .iter()
            .find(|method| method.name == "initialize_tensors")
            .ok_or_else(|| invalid("actual named source initializer missing"))?;
        let Some(Statement::For { body, .. }) = method.statements.first() else {
            return Err(invalid("actual named source loop missing"));
        };
        let Some(Statement::Declare { kind, .. }) = body.first() else {
            return Err(invalid("actual native-name declaration missing"));
        };
        assert_eq!(kind, "conststd::string");
        let (_, name) = runtime.declaration_value(
            kind,
            &[],
            Some(Value::NativeInteger {
                kind: super::super::number::Kind::I32,
                argument: json!({"Scalar":1}),
            }),
        )?;
        assert!(
            runtime
                .binary_value("==", name.clone(), Value::Bytes(b"scale".to_vec()))?
                .truth()
                .map_err(invalid)?
        );
        assert!(matches!(name, Value::Bytes(_)));
        assert_eq!(observed.get(), 1);
        for kind in ["constint64_t", "constchar*", "std::string*"] {
            let (_, value) = runtime.declaration_value(
                kind,
                &[],
                Some(Value::NativeInteger {
                    kind: super::super::number::Kind::I32,
                    argument: json!({"Scalar":2}),
                }),
            )?;
            assert!(matches!(value, Value::NativeInteger { .. }));
        }
        assert_eq!(observed.get(), 1);
        let buffer = Value::array(vec![Value::Number(Number::I32(65))])
            .coerce("std::array<char,1>")
            .map_err(invalid)?;
        let (_, pointer) = runtime.declaration_value(
            "constchar*",
            &[],
            Some(Value::Pointer {
                value: Box::new(buffer.clone()),
                index: 0,
            }),
        )?;
        pointer
            .place(0)
            .map_err(invalid)?
            .set(Value::Number(Number::I32(66)))
            .map_err(invalid)?;
        assert_eq!(
            buffer
                .index(0)
                .map_err(invalid)?
                .number()
                .map_err(invalid)?,
            Number::I32(66)
        );
        Ok(())
    }

    #[derive(Default)]
    struct RowIdWrites {
        seeds: usize,
        rows: Vec<(u64, Vec<u8>)>,
    }

    impl Host for RowIdWrites {
        fn call(&mut self, name: &str, arguments: &[Value]) -> Result<Option<Value>> {
            match name {
                "random-device" => {
                    self.seeds += 1;
                    Ok(Some(Value::Number(Number::U32(1234))))
                }
                "ggml_backend_tensor_set" => {
                    let [_, data, offset, bytes] = arguments else {
                        return Err(invalid("row-ID source transfer argument extent differs"));
                    };
                    self.rows.push((
                        offset.number().map_err(invalid)?.as_u64(),
                        data.bytes(usize::try_from(bytes.number().map_err(invalid)?.as_u64())?)
                            .map_err(invalid)?,
                    ));
                    Ok(Some(Value::Null))
                }
                _ => LanguageOnly.call(name, arguments),
            }
        }
        fn member(&mut self, value: &Value, name: &str) -> Result<Option<Value>> {
            LanguageOnly.member(value, name)
        }
        fn write(
            &mut self,
            value: &Value,
            name: &str,
            index: Option<usize>,
            replacement: Value,
        ) -> Result<()> {
            LanguageOnly.write(value, name, index, replacement)
        }
        fn force(&mut self, value: Value) -> Result<Value> {
            Ok(value)
        }
    }

    fn source_row_ids(
        geometry_index: usize,
        kind: i32,
        shape: [usize; 4],
        num_rows: i32,
    ) -> Result<Vec<(u64, Vec<i64>)>> {
        let (program, geometries) = super::super::program::load().map_err(invalid)?;
        let geometry = &geometries.eval[geometry_index];
        let mut host = RowIdWrites::default();
        let mut runtime = Interpreter::new(program, &geometry.class, &mut host)?;
        runtime.instantiate(geometry)?;
        let axis = usize::from(geometry.class == "test_rms_norm_mul_rope") + 1;
        assert_eq!(
            runtime
                .value("ne")?
                .index(axis)
                .map_err(invalid)?
                .number()
                .map_err(invalid)?
                .as_i64(),
            i64::from(num_rows)
        );
        let width = if kind == 26 { 4 } else { 8 };
        let tensor = super::super::standard::literal(
            &json!({"Record":{"type":kind,"ne":shape,"nb":[width,32,128,256]}}),
            None,
        )?;
        runtime.call(
            "init_set_rows_row_ids",
            vec![tensor, Value::Number(Number::I32(num_rows))],
        )?;
        assert_eq!(host.seeds, 1);
        assert_eq!(host.rows.len(), shape[1] * shape[2]);
        host.rows
            .into_iter()
            .map(|(offset, bytes)| {
                assert_eq!(bytes.len(), shape[0] * width);
                let values = bytes
                    .chunks_exact(width)
                    .map(|value| {
                        Ok(if kind == 26 {
                            i64::from(i32::from_le_bytes(value.try_into()?))
                        } else {
                            i64::from_le_bytes(value.try_into()?)
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                assert!(
                    values
                        .iter()
                        .all(|&value| (0..i64::from(num_rows)).contains(&value))
                );
                let unique: std::collections::BTreeSet<_> = values.iter().collect();
                assert_eq!(unique.len(), shape[0]);
                Ok((offset, values))
            })
            .collect()
    }

    #[test]
    fn pinned_set_rows_593_and_rms_rope_8800_initialize_i32_indices() -> Result<()> {
        // Exercise the actual helper AST used by both retained source failures,
        // including its shuffle, resize, static_cast and raw byte transfers.
        for (geometry, rows) in [(593, 3), (8800, 2)] {
            let i32_rows = source_row_ids(geometry, 26, [2, 1, 1, 1], rows)?;
            let i64_rows = source_row_ids(geometry, 27, [2, 1, 1, 1], rows)?;
            assert_eq!(i32_rows, i64_rows);
            assert_eq!(i32_rows[0].0, 0);
        }
        Ok(())
    }

    #[test]
    fn pinned_row_id_helper_preserves_padded_multiplane_offsets_and_cast_width() -> Result<()> {
        let i32_rows = source_row_ids(593, 26, [2, 2, 2, 1], 3)?;
        let i64_rows = source_row_ids(593, 27, [2, 2, 2, 1], 3)?;
        assert_eq!(i32_rows, i64_rows);
        assert_eq!(
            i32_rows
                .iter()
                .map(|(offset, _)| *offset)
                .collect::<Vec<_>>(),
            [0, 32, 128, 160]
        );
        Ok(())
    }
}
