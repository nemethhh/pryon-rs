//! Load-time planning: shape binding, constant folding, dead-node removal and the arena layout.

use super::ops::{Attrs, BuildFn, Built, Op, Value};
use super::proto::{Dim, ModelProto, TensorData, TensorProto, ValueInfo};
use super::tensor::{numel, DType, Tensor, MAX_ELEMS};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

const MAX_NODES: usize = 1 << 16;
/// Inputs or outputs per node; the run loop keeps one array of each on the stack (plus a scratch slot).
pub(crate) const MAX_IO: usize = 31;

pub(crate) enum Fail {
    Unsupported(String),
    Format(String),
}

/// A graph input or output: its arena region and, for a kept input, the initializer it starts from.
pub(crate) struct Io {
    pub name: String,
    pub dims: Vec<usize>,
    pub off: usize,
    pub len: usize,
    pub initial: Option<Vec<f32>>,
}

/// One runtime node. `ins`/`outs` are arena regions `(offset, len)`; `outs` ends with the scratch region if any.
pub(crate) struct Step {
    pub op: Box<dyn Op>,
    pub ins: Vec<(usize, usize)>,
    pub outs: Vec<(usize, usize)>,
    pub names: Vec<String>,
    pub dims: Vec<Vec<usize>>,
}

pub(crate) struct Plan {
    pub inputs: Vec<Io>,
    pub outputs: Vec<Io>,
    pub steps: Vec<Step>,
    /// Initial arena contents: zeros, kept-input initializers and the f32 constants runtime steps read.
    pub arena: Vec<f32>,
}

use Fail::{Format, Unsupported};

/// Decodes an initializer; a kept `placeholder` without data (firmware state inputs) is zeros.
fn konst(t: &TensorProto, placeholder: bool) -> Result<Tensor, Fail> {
    let dtype = DType::from_onnx(t.data_type)
        .ok_or_else(|| Unsupported(format!("tensor element type {} (initializer {})", t.data_type, t.name)))?;
    let dims = t
        .dims
        .iter()
        .map(|&d| usize::try_from(d))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| Format(format!("initializer {}: negative dimension", t.name)))?;
    let n = numel(&dims).map_err(|e| Unsupported(format!("initializer {}: {e}", t.name)))?;
    let empty = matches!(&t.data, TensorData::Empty) || matches!(&t.data, TensorData::Raw(b) if b.is_empty());
    if empty && n > 0 {
        if !placeholder {
            return Err(Format(format!("initializer {} has no data (a placeholder input must be kept)", t.name)));
        }
        return Ok(Tensor::i64(dims, vec![0; n]).cast(dtype));
    }
    Tensor::from_proto(t, dtype, dims, n).map_err(Format)
}

fn show(dims: &[Dim]) -> String {
    let d: Vec<String> = dims
        .iter()
        .map(|d| match d {
            Dim::Value(v) => v.to_string(),
            Dim::Param(p) => p.clone(),
            Dim::Unknown => "?".into(),
        })
        .collect();
    format!("[{}]", d.join(", "))
}

/// A declared shape (if any) must agree with the bound one in rank and in every static dimension.
fn check_declared(vi: &ValueInfo, dims: &[usize]) -> Result<(), Fail> {
    let ok = vi.dims.is_empty()
        || (vi.dims.len() == dims.len()
            && vi.dims.iter().zip(dims).all(|(d, &b)| !matches!(d, Dim::Value(v) if *v != b as i64)));
    if ok {
        return Ok(());
    }
    Err(Unsupported(format!("bound shape {dims:?} of input {} conflicts with declared {}", vi.name, show(&vi.dims))))
}

struct Values<'a> {
    vals: HashMap<&'a str, Value>,
    inits: HashMap<&'a str, &'a TensorProto>,
}

impl<'a> Values<'a> {
    /// A defined value; initializers are decoded on first use so unused ones cost nothing.
    fn get(&mut self, name: &'a str) -> Result<Option<Value>, Fail> {
        if let Some(v) = self.vals.get(name) {
            return Ok(Some(v.clone()));
        }
        let Some(t) = self.inits.get(name) else { return Ok(None) };
        let v = Value::Const(Arc::new(konst(t, false)?));
        self.vals.insert(name, v.clone());
        Ok(Some(v))
    }

    fn define(&mut self, name: &'a str, v: Value) -> Result<(), Fail> {
        if name.is_empty() {
            return Ok(());
        }
        if self.vals.contains_key(name) || self.inits.contains_key(name) {
            return Err(Format(format!("value {name} defined twice")));
        }
        self.vals.insert(name, v);
        Ok(())
    }
}

struct Arena {
    len: usize,
    regions: HashMap<String, (usize, usize)>,
    consts: Vec<(usize, Arc<Tensor>)>,
}

impl Arena {
    fn alloc(&mut self, n: usize) -> Result<usize, Fail> {
        let off = self.len;
        self.len = off
            .checked_add(n)
            .filter(|&t| t <= MAX_ELEMS)
            .ok_or_else(|| Unsupported(format!("graph needing more than {MAX_ELEMS} elements of buffers")))?;
        Ok(off)
    }

    /// Region of an f32 constant, placed in the arena once; other constants read as empty slices.
    fn konst(&mut self, name: &str, t: &Arc<Tensor>) -> Result<(usize, usize), Fail> {
        if t.dtype() != DType::F32 {
            return Ok((0, 0));
        }
        if let Some(&r) = self.regions.get(name) {
            return Ok(r);
        }
        let r = (self.alloc(t.as_f32().map_or(0, <[f32]>::len))?, t.as_f32().map_or(0, <[f32]>::len));
        self.regions.insert(name.to_string(), r);
        self.consts.push((r.0, t.clone()));
        Ok(r)
    }
}

pub(crate) fn plan(m: &ModelProto, bind: &[(&str, &[usize])], keep: &[&str], ops: &[(&str, BuildFn)]) -> Result<Plan, Fail> {
    if m.ir_version < 3 {
        return Err(Unsupported(format!("IR version {}", m.ir_version)));
    }
    let opset = m
        .opsets
        .iter()
        .find(|(d, _)| d.is_empty() || d == "ai.onnx")
        .map(|o| o.1)
        .ok_or_else(|| Unsupported("model without a default-domain opset".into()))?;
    let g = &m.graph;
    if g.nodes.len() > MAX_NODES {
        return Err(Unsupported(format!("graph of {} nodes (at most {MAX_NODES})", g.nodes.len())));
    }

    let mut vs = Values { vals: HashMap::new(), inits: HashMap::new() };
    for t in &g.initializers {
        if t.external {
            return Err(Unsupported(format!("external data (initializer {})", t.name)));
        }
        if vs.inits.insert(&t.name, t).is_some() {
            return Err(Format(format!("initializer {} defined twice", t.name)));
        }
    }

    // runtime inputs: declared inputs without an initializer, then kept initializers
    let mut declared: HashMap<&str, &ValueInfo> = HashMap::new();
    let mut names: Vec<&str> = Vec::new();
    for v in &g.inputs {
        // the DAVS decoders repeat some input declarations verbatim
        match declared.insert(&v.name, v) {
            Some(prev) if prev != v => return Err(Format(format!("input {} declared twice", v.name))),
            Some(_) => {}
            None if !vs.inits.contains_key(v.name.as_str()) || keep.contains(&v.name.as_str()) => names.push(&v.name),
            None => {}
        }
    }
    for &k in keep {
        if !vs.inits.contains_key(k) {
            return Err(Unsupported(format!("kept input {k}: not an initializer")));
        }
        if !names.contains(&k) {
            names.push(k);
        }
    }
    let mut inputs = Vec::new();
    for &name in &names {
        let bound = bind.iter().find(|(n, _)| *n == name).map(|b| b.1);
        let (dims, dtype, initial) = if keep.contains(&name) {
            let t = konst(vs.inits[name], true)?;
            if bound.is_some_and(|b| b != t.dims) {
                return Err(Unsupported(format!(
                    "bound shape {:?} of input {name} conflicts with its initializer {:?}",
                    bound.unwrap_or_default(),
                    t.dims
                )));
            }
            let initial = t.as_f32().map(<[f32]>::to_vec);
            (t.dims.clone(), t.dtype(), initial)
        } else {
            let b = bound.ok_or_else(|| Unsupported(format!("input {name} without a bound shape")))?;
            let et = declared[name].elem_type;
            let dt = DType::from_onnx(et).ok_or_else(|| Unsupported(format!("input {name} of element type {et}")))?;
            (b.to_vec(), dt, None)
        };
        if let Some(vi) = declared.get(name) {
            check_declared(vi, &dims)?;
        }
        numel(&dims).map_err(|e| Unsupported(format!("input {name}: {e}")))?;
        vs.vals.insert(name, Value::Runtime(dims.clone(), dtype));
        inputs.push(Io { name: name.to_string(), dims, off: 0, len: 0, initial });
    }
    if let Some((n, _)) = bind.iter().find(|(n, _)| !names.contains(n)) {
        return Err(Unsupported(format!("bound input {n}: not a runtime input of the graph")));
    }

    // nodes that reach no graph output are dropped before anything looks at them
    let mut needed: HashSet<&str> = g.outputs.iter().map(|o| o.name.as_str()).collect();
    let mut live = vec![false; g.nodes.len()];
    for (i, n) in g.nodes.iter().enumerate().rev() {
        if n.outputs.iter().any(|o| needed.contains(o.as_str())) {
            live[i] = true;
            needed.extend(n.inputs.iter().map(String::as_str));
        }
    }

    let mut pending: Vec<(usize, Box<dyn Op>, Vec<Vec<usize>>)> = Vec::new();
    for (i, n) in g.nodes.iter().enumerate().filter(|&(i, _)| live[i]) {
        let label = if n.name.is_empty() { n.outputs.first().map_or("", String::as_str) } else { &n.name };
        let op = &n.op_type;
        if !(n.domain.is_empty() || n.domain == "ai.onnx") {
            return Err(Unsupported(format!("operator {}.{op} (node {label})", n.domain)));
        }
        let build = ops
            .iter()
            .find(|(o, _)| o == op)
            .map(|o| o.1)
            .ok_or_else(|| Unsupported(format!("operator {op} (node {label})")))?;
        if n.inputs.len() > MAX_IO || n.outputs.len() > MAX_IO {
            return Err(Unsupported(format!("{op} (node {label}) with more than {MAX_IO} inputs or outputs")));
        }
        let mut ins = Vec::with_capacity(n.inputs.len());
        for x in &n.inputs {
            ins.push(if x.is_empty() {
                Value::Absent
            } else {
                vs.get(x)?.ok_or_else(|| {
                    Format(format!("node {label}: input {x} used before it is defined (nodes must be in topological order)"))
                })?
            });
        }
        let mut attrs = Attrs::new(n);
        let built = build(n, &mut attrs, &ins, opset).map_err(|e| Unsupported(format!("{op} (node {label}): {e}")))?;
        if let Some(a) = attrs.leftover() {
            return Err(Unsupported(format!("attribute {op}.{a} (node {label})")));
        }
        let count = match &built {
            Built::Folded(ts) => ts.len(),
            Built::Runtime { outs, .. } => outs.len(),
        };
        if count != n.outputs.len() {
            return Err(Unsupported(format!("{op} (node {label}) with {} outputs", n.outputs.len())));
        }
        let too_big = |e: String| Unsupported(format!("{op} (node {label}): {e}"));
        match built {
            Built::Folded(ts) => {
                for (o, t) in n.outputs.iter().zip(ts) {
                    numel(&t.dims).map_err(too_big)?;
                    vs.define(o, Value::Const(Arc::new(t)))?;
                }
            }
            Built::Runtime { op: rop, outs } => {
                let non_float = |x: &str| Unsupported(format!("non-float runtime tensor {x} at node {label}"));
                for (x, v) in n.inputs.iter().zip(&ins) {
                    if matches!(v, Value::Runtime(_, t) if *t != DType::F32) {
                        return Err(non_float(x));
                    }
                }
                let mut dims = Vec::with_capacity(outs.len());
                for (o, (d, t)) in n.outputs.iter().zip(outs) {
                    if t != DType::F32 {
                        return Err(non_float(o));
                    }
                    numel(&d).map_err(too_big)?;
                    vs.define(o, Value::Runtime(d.clone(), DType::F32))?;
                    dims.push(d);
                }
                pending.push((i, rop, dims));
            }
        }
    }

    let mut outputs = Vec::new();
    for o in &g.outputs {
        let v = vs.get(&o.name)?.ok_or_else(|| Format(format!("graph output {} is never produced", o.name)))?;
        if v.dtype() != Some(DType::F32) {
            return Err(Unsupported(format!("non-float output {}", o.name)));
        }
        outputs.push((o.name.as_str(), v));
    }

    let mut arena = Arena { len: 0, regions: HashMap::new(), consts: Vec::new() };
    for io in &mut inputs {
        io.len = numel(&io.dims).unwrap_or(0);
        io.off = arena.alloc(io.len)?;
        arena.regions.insert(io.name.clone(), (io.off, io.len));
    }
    let mut steps = Vec::with_capacity(pending.len());
    for (i, op, dims) in pending {
        let n = &g.nodes[i];
        let mut ins = Vec::with_capacity(n.inputs.len());
        for x in &n.inputs {
            ins.push(match vs.vals.get(x.as_str()) {
                Some(Value::Runtime(..)) => arena.regions[x],
                Some(Value::Const(t)) => arena.konst(x, t)?,
                _ => (0, 0),
            });
        }
        let mut outs = Vec::with_capacity(n.outputs.len() + 1);
        for (o, d) in n.outputs.iter().zip(&dims) {
            let len = numel(d).unwrap_or(0);
            let r = (arena.alloc(len)?, len);
            if !o.is_empty() {
                arena.regions.insert(o.clone(), r);
            }
            outs.push(r);
        }
        if op.scratch() > 0 {
            outs.push((arena.alloc(op.scratch())?, op.scratch()));
        }
        steps.push(Step { op, ins, outs, names: n.outputs.clone(), dims });
    }
    let mut outs = Vec::with_capacity(outputs.len());
    for (name, v) in outputs {
        let (off, len) = match &v {
            Value::Const(t) => arena.konst(name, t)?,
            _ => arena.regions[name],
        };
        outs.push(Io { name: name.to_string(), dims: v.dims().to_vec(), off, len, initial: None });
    }

    let mut buf = vec![0.0f32; arena.len];
    for io in &inputs {
        if let Some(v) = &io.initial {
            buf[io.off..io.off + io.len].copy_from_slice(v);
        }
    }
    for (off, t) in &arena.consts {
        let v = t.as_f32().unwrap_or_default();
        buf[*off..off + v.len()].copy_from_slice(v);
    }
    Ok(Plan { inputs, outputs: outs, steps, arena: buf })
}

#[cfg(test)]
mod tests {
    use super::super::ops::{arg, Attrs, Built, Op, Value, OPS};
    use super::super::proto::{parse_model, NodeProto};
    use super::super::{Graph, Session};
    use super::*;
    use pryon_testkit::onnx::{Dim as KDim, ModelBuilder};

    /// Doubles its input into scratch, then writes scratch + 1.
    struct Twice;

    impl Op for Twice {
        fn run(&self, ins: &[&[f32]], outs: &mut [&mut [f32]]) {
            let (y, s) = outs.split_at_mut(1);
            for (s, &x) in s[0].iter_mut().zip(ins[0]) {
                *s = x * 2.0;
            }
            for (y, &s) in y[0].iter_mut().zip(s[0].iter()) {
                *y = s + 1.0;
            }
        }
        fn scratch(&self) -> usize {
            3
        }
    }

    fn twice(_: &NodeProto, _: &mut Attrs, ins: &[Value], _: i64) -> Result<Built, String> {
        let x = arg(ins, 0)?;
        Ok(Built::Runtime { op: Box::new(Twice), outs: vec![(x.dims().to_vec(), DType::F32)] })
    }

    #[test]
    fn scratch_follows_outputs() {
        let bytes = ModelBuilder::new(8, 15)
            .input("x", 1, &[KDim::Value(3)])
            .output("y", 1, &[KDim::Value(3)])
            .node("Twice", &["x"], &["t"], &[])
            .node("Relu", &["t"], &["y"], &[])
            .build();
        let m = parse_model(&bytes).unwrap();
        let mut ops: Vec<(&str, BuildFn)> = OPS.to_vec();
        ops.push(("Twice", twice));
        let Ok(p) = plan(&m, &[("x", &[3])], &[], &ops) else { panic!("plan failed") };
        assert_eq!(p.steps[0].outs.len(), 2);
        assert_eq!(p.steps[0].outs[1].1, 3);
        let mut s = Session::new(Arc::new(Graph { plan: p }));
        s.input_mut(0).copy_from_slice(&[1.0, -2.0, 3.0]);
        let mut traced = Vec::new();
        s.run_traced(&mut |n, _, v| traced.push((n.to_string(), v.to_vec())));
        assert_eq!(s.output(0), [3.0, 0.0, 7.0]);
        assert_eq!(traced[0], ("t".to_string(), vec![3.0, -3.0, 7.0]));
        assert_eq!(traced.len(), 2);
    }
}
