//! Operator registry: each builder sees a node's inputs as constants or runtime shapes and either folds it or
//! returns a runtime [`Op`] holding whatever it precomputed at load.

mod elementwise;
mod shape;

use super::proto::{AttrValue, NodeProto};
use super::tensor::{numel, DType, Tensor};
use std::sync::Arc;

/// A runtime step. `ins` has one slice per node input (f32 values; empty for absent or non-f32 constant inputs),
/// `outs` one per node output, followed by the scratch buffer when [`Op::scratch`] is non-zero.
pub(crate) trait Op: Send + Sync {
    fn run(&self, ins: &[&[f32]], outs: &mut [&mut [f32]]);
    /// f32 elements of per-session working memory this op needs.
    fn scratch(&self) -> usize {
        0
    }
}

pub(crate) enum Built {
    /// One tensor per node output.
    Folded(Vec<Tensor>),
    /// Shape and type of each node output.
    Runtime { op: Box<dyn Op>, outs: Vec<(Vec<usize>, DType)> },
}

pub(crate) type BuildFn = fn(&NodeProto, &mut Attrs, &[Value], i64) -> Result<Built, String>;

/// Default-domain operators; adding one is a builder plus a line here.
pub(crate) static OPS: &[(&str, BuildFn)] = &[
    ("Add", elementwise::binary::<elementwise::Add>),
    ("Cast", shape::cast),
    ("Concat", shape::concat),
    ("Gather", shape::gather),
    ("Mul", elementwise::binary::<elementwise::Mul>),
    ("Neg", elementwise::unary::<elementwise::Neg>),
    ("Relu", elementwise::unary::<elementwise::Relu>),
    ("Reshape", shape::reshape),
    ("Shape", shape::shape),
    ("Slice", shape::slice),
    ("Squeeze", shape::squeeze),
    ("Sub", elementwise::binary::<elementwise::Sub>),
    ("Tile", shape::tile),
    ("Transpose", shape::transpose),
    ("Unsqueeze", shape::unsqueeze),
];

/// A node input as the builder sees it.
#[derive(Clone, Debug)]
pub(crate) enum Value {
    Const(Arc<Tensor>),
    Runtime(Vec<usize>, DType),
    /// An optional input left empty (`""`) or past the end of the node's input list.
    Absent,
}

impl Value {
    pub fn dims(&self) -> &[usize] {
        match self {
            Value::Const(t) => &t.dims,
            Value::Runtime(d, _) => d,
            Value::Absent => &[],
        }
    }

    pub fn dtype(&self) -> Option<DType> {
        match self {
            Value::Const(t) => Some(t.dtype()),
            Value::Runtime(_, t) => Some(*t),
            Value::Absent => None,
        }
    }

    pub fn konst(&self) -> Option<&Tensor> {
        match self {
            Value::Const(t) => Some(t),
            _ => None,
        }
    }
}

/// Input `i`, which must be present.
pub(crate) fn arg(ins: &[Value], i: usize) -> Result<&Value, String> {
    match ins.get(i) {
        Some(v @ (Value::Const(_) | Value::Runtime(..))) => Ok(v),
        _ => Err(format!("input {i} missing")),
    }
}

/// Optional input `i`, `None` when absent.
pub(crate) fn opt_arg(ins: &[Value], i: usize) -> Option<&Value> {
    ins.get(i).filter(|v| !matches!(v, Value::Absent))
}

/// The integers of a constant input (axes, shapes, indices); a runtime one cannot be folded and is refused.
pub(crate) fn const_ints(v: &Value, what: &str) -> Result<Vec<i64>, String> {
    v.konst().ok_or_else(|| format!("{what} computed at runtime"))?.to_i64s().map_err(|e| format!("{what}: {e}"))
}

/// Normalises a possibly negative axis against `rank`.
pub(crate) fn axis(a: i64, rank: usize) -> Result<usize, String> {
    let r = rank as i64;
    if (-r..r).contains(&a) {
        Ok(a.rem_euclid(r) as usize)
    } else {
        Err(format!("axis {a} out of range for rank {rank}"))
    }
}

/// Hands out a node's attributes by name; whatever no builder asked for is reported by the planner.
pub(crate) struct Attrs {
    op: String,
    list: Vec<(String, AttrValue)>,
}

impl Attrs {
    pub fn new(n: &NodeProto) -> Attrs {
        Attrs { op: n.op_type.clone(), list: n.attrs.clone() }
    }

    fn take(&mut self, name: &str) -> Option<AttrValue> {
        let i = self.list.iter().position(|(n, _)| n == name)?;
        Some(self.list.remove(i).1)
    }

    fn wrong(&self, name: &str, want: &str) -> String {
        format!("attribute {}.{name} is not {want}", self.op)
    }

    pub fn int(&mut self, name: &str) -> Result<Option<i64>, String> {
        match self.take(name) {
            None => Ok(None),
            Some(AttrValue::Int(v)) => Ok(Some(v)),
            Some(_) => Err(self.wrong(name, "an int")),
        }
    }

    pub fn ints(&mut self, name: &str) -> Result<Option<Vec<i64>>, String> {
        match self.take(name) {
            None => Ok(None),
            Some(AttrValue::Ints(v)) => Ok(Some(v)),
            Some(_) => Err(self.wrong(name, "a list of ints")),
        }
    }

    /// Consumes an attribute that has no effect on this engine.
    pub fn ignore(&mut self, name: &str) {
        self.take(name);
    }

    /// The first attribute nobody consumed.
    pub fn leftover(&self) -> Option<&str> {
        self.list.first().map(|(n, _)| n.as_str())
    }
}

/// Calls `f` with every multi-index of `dims` in row-major order, after checking the element count.
pub(crate) fn each_index(dims: &[usize], mut f: impl FnMut(&[usize])) -> Result<(), String> {
    let n = numel(dims)?;
    let mut idx = vec![0usize; dims.len()];
    for _ in 0..n {
        f(&idx);
        for k in (0..dims.len()).rev() {
            idx[k] += 1;
            if idx[k] < dims[k] {
                break;
            }
            idx[k] = 0;
        }
    }
    Ok(())
}

pub(crate) fn strides(dims: &[usize]) -> Vec<usize> {
    let mut s = vec![1; dims.len()];
    for k in (0..dims.len().saturating_sub(1)).rev() {
        s[k] = s[k + 1] * dims[k + 1];
    }
    s
}

/// Axes from the `axes` attribute or from input `i`, whichever the node uses, regardless of opset.
pub(crate) fn axes_of(a: &mut Attrs, ins: &[Value], i: usize) -> Result<Option<Vec<i64>>, String> {
    match (a.ints("axes")?, opt_arg(ins, i)) {
        (Some(_), Some(_)) => Err("axes given as attribute and input".into()),
        (Some(v), None) => Ok(Some(v)),
        (None, Some(v)) => const_ints(v, "axes").map(Some),
        (None, None) => Ok(None),
    }
}

/// Data movement given per output element as `(input, offset)` over inputs `0..n`; folds when all are constant.
pub(crate) fn copy_op(ins: &[Value], n: usize, map: Vec<(usize, usize)>, dims: Vec<usize>) -> Result<Built, String> {
    let consts: Option<Vec<&Tensor>> = ins[..n].iter().map(Value::konst).collect();
    if let Some(parts) = consts {
        return Ok(Built::Folded(vec![Tensor::gather(&parts, &map, dims)?]));
    }
    let dtype = ins[0].dtype().ok_or("input 0 missing")?;
    if ins[..n].iter().any(|v| v.dtype() != Some(dtype)) {
        return Err("inputs of different element types".into());
    }
    Ok(Built::Runtime { op: Box::new(CopyRuns::new(&map)), outs: vec![(dims, dtype)] })
}

/// The same elements under new dims (Reshape, Squeeze, Unsqueeze, Cast at runtime).
pub(crate) fn view(x: &Value, dims: Vec<usize>) -> Result<Built, String> {
    let n = numel(&dims)?;
    Ok(match x {
        Value::Const(t) => Built::Folded(vec![Tensor { dims, data: t.data.clone() }]),
        _ => {
            let dtype = x.dtype().ok_or("input missing")?;
            Built::Runtime { op: Box::new(CopyRuns(vec![(0, 0, n)])), outs: vec![(dims, dtype)] }
        }
    })
}

/// Contiguous runs `(input, offset, len)` written back to back into output 0.
struct CopyRuns(Vec<(usize, usize, usize)>);

impl CopyRuns {
    fn new(map: &[(usize, usize)]) -> CopyRuns {
        let mut runs: Vec<(usize, usize, usize)> = Vec::new();
        for &(i, o) in map {
            match runs.last_mut() {
                Some((ri, ro, rl)) if *ri == i && *ro + *rl == o => *rl += 1,
                _ => runs.push((i, o, 1)),
            }
        }
        CopyRuns(runs)
    }
}

impl Op for CopyRuns {
    fn run(&self, ins: &[&[f32]], outs: &mut [&mut [f32]]) {
        let mut p = 0;
        for &(i, o, n) in &self.0 {
            outs[0][p..p + n].copy_from_slice(&ins[i][o..o + n]);
            p += n;
        }
    }
}
