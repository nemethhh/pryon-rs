//! Float operators that combine elements: Softmax, ReduceSum/Max, MaxPool, BatchNormalization, in libpryon's order.

use super::{arg, axes_of, axis, fold_numel, opt_arg, strides, Attrs, Built, Op, Value};
use crate::num::{cpunk_softmax, fmaf, vadd, vmul, vsub, zdiv, zsqrt};
use crate::onnx::proto::NodeProto;
use crate::onnx::tensor::{numel, Data, DType, Tensor, MAX_RANK};

/// Folds when every input is a constant f32 tensor, otherwise stays a runtime op.
fn finish(op: Box<dyn Op>, ins: &[Value], dims: Vec<usize>) -> Result<Built, String> {
    if ins.iter().all(|v| matches!(v, Value::Const(_) | Value::Absent)) {
        let n = fold_numel(&dims)?;
        let held: Vec<&[f32]> = ins
            .iter()
            .map(|v| match v {
                Value::Const(t) => t.as_f32().ok_or_else(|| format!("constant of type {:?}", t.dtype())),
                _ => Ok(&[][..]),
            })
            .collect::<Result<_, _>>()?;
        let mut out = vec![0.0f32; n + op.scratch()];
        let (o, scratch) = out.split_at_mut(n);
        let mut outs = vec![o];
        if !scratch.is_empty() {
            outs.push(scratch);
        }
        op.run(&held, &mut outs);
        out.truncate(n);
        return Ok(Built::Folded(vec![Tensor { dims, data: Data::F32(out) }]));
    }
    numel(&dims)?;
    Ok(Built::Runtime { op, outs: vec![(dims, DType::F32)] })
}

fn float_arg(ins: &[Value], i: usize) -> Result<&Value, String> {
    let v = arg(ins, i)?;
    if v.dtype() != Some(DType::F32) {
        return Err(format!("input {i} of type {:?}", v.dtype()));
    }
    Ok(v)
}

/// Rows of `len` contiguous values (libpryon's Softmax, 0xcc883c, flattens at the axis).
struct Softmax {
    len: usize,
}

impl Op for Softmax {
    fn run(&self, ins: &[&[f32]], outs: &mut [&mut [f32]]) {
        for (x, y) in ins[0].chunks_exact(self.len).zip(outs[0].chunks_exact_mut(self.len)) {
            cpunk_softmax(x, y);
        }
    }
}

pub(super) fn softmax(_: &NodeProto, a: &mut Attrs, ins: &[Value], opset: i64) -> Result<Built, String> {
    let x = float_arg(ins, 0)?;
    let dims = x.dims().to_vec();
    if dims.is_empty() {
        return Err("scalar input".into());
    }
    let legacy = opset < 13;
    let ax = axis(a.int("axis")?.unwrap_or(if legacy { 1 } else { -1 }), dims.len())?;
    if !legacy && ax + 1 != dims.len() {
        // libpryon normalises over everything from the axis on; opset 13 means the axis alone
        return Err(format!("attribute axis {ax} before the last axis (opset {opset})"));
    }
    let len = numel(&dims[ax..])?;
    if len == 0 {
        return Err(format!("empty rows in {dims:?}"));
    }
    finish(Box::new(Softmax { len }), ins, dims)
}

/// libpryon reduces one axis slice by slice (0xb027cc, 0xe39d84): the first slice copied, then `step` per slice.
pub(crate) trait RedKind: Send + Sync + 'static {
    fn step(acc: f32, x: f32) -> f32;
}

pub(crate) struct Sum;
pub(crate) struct Max;

impl RedKind for Sum {
    /// CPUNK Add (NEON).
    fn step(acc: f32, x: f32) -> f32 {
        vadd(acc, x)
    }
}

impl RedKind for Max {
    /// VFP compare: the accumulator stays unless `acc < x` (a NaN in either keeps it).
    fn step(acc: f32, x: f32) -> f32 {
        if acc < x {
            x
        } else {
            acc
        }
    }
}

struct Reduce<K> {
    dims: Vec<usize>,
    /// Output stride of each input axis, 0 on reduced ones.
    ostride: Vec<usize>,
    k: std::marker::PhantomData<K>,
}

impl<K: RedKind> Op for Reduce<K> {
    fn run(&self, ins: &[&[f32]], outs: &mut [&mut [f32]]) {
        let mut idx = [0usize; MAX_RANK];
        let mut off = 0;
        // reduced axes at a nonzero index: 0 marks an output's first contribution
        let mut moved = 0usize;
        for &x in ins[0] {
            outs[0][off] = if moved == 0 { x } else { K::step(outs[0][off], x) };
            for k in (0..self.dims.len()).rev() {
                let red = self.ostride[k] == 0 && self.dims[k] > 1;
                idx[k] += 1;
                off += self.ostride[k];
                if idx[k] < self.dims[k] {
                    moved += usize::from(red && idx[k] == 1);
                    break;
                }
                off -= self.ostride[k] * self.dims[k];
                idx[k] = 0;
                moved -= usize::from(red);
            }
        }
    }
}

pub(super) fn reduce<K: RedKind>(_: &NodeProto, a: &mut Attrs, ins: &[Value], _: i64) -> Result<Built, String> {
    let x = float_arg(ins, 0)?;
    let dims = x.dims().to_vec();
    if dims.len() > MAX_RANK {
        return Err(format!("input of rank {}", dims.len()));
    }
    let keep = a.int("keepdims")?.unwrap_or(1) != 0;
    if a.int("noop_with_empty_axes")?.unwrap_or(0) != 0 {
        return Err("noop_with_empty_axes set".into());
    }
    let mut red = vec![false; dims.len()];
    match axes_of(a, ins, 1)?.filter(|v| !v.is_empty()) {
        None => red.fill(true),
        Some(v) => {
            for ax in v {
                red[axis(ax, dims.len())?] = true;
            }
        }
    }
    let out: Vec<usize> = (0..dims.len()).filter(|&k| keep || !red[k]).map(|k| if red[k] { 1 } else { dims[k] }).collect();
    let kept: Vec<usize> = dims.iter().zip(&red).map(|(&d, &r)| if r { 1 } else { d }).collect();
    let ks = strides(&kept);
    let ostride = (0..dims.len()).map(|k| if red[k] { 0 } else { ks[k] }).collect();
    if numel(&dims)? == 0 && numel(&out)? != 0 {
        return Err("reduction over an empty tensor".into());
    }
    finish(Box::new(Reduce::<K> { dims, ostride, k: Default::default() }), &ins[..1], out)
}

struct MaxPool {
    nc: usize,
    h: usize,
    w: usize,
    k: (usize, usize),
    s: (usize, usize),
    o: (usize, usize),
}

impl Op for MaxPool {
    fn run(&self, ins: &[&[f32]], outs: &mut [&mut [f32]]) {
        let (x, y) = (ins[0], &mut *outs[0]);
        for p in 0..self.nc {
            let plane = &x[p * self.h * self.w..][..self.h * self.w];
            for oh in 0..self.o.0 {
                for ow in 0..self.o.1 {
                    // 0xd016a8: from -FLT_MAX, replaced only when `v > m`
                    let mut m = f32::MIN;
                    for i in 0..self.k.0 {
                        for j in 0..self.k.1 {
                            let v = plane[(oh * self.s.0 + i) * self.w + ow * self.s.1 + j];
                            m = if v > m { v } else { m };
                        }
                    }
                    y[(p * self.o.0 + oh) * self.o.1 + ow] = m;
                }
            }
        }
    }
}

pub(super) fn max_pool(_: &NodeProto, a: &mut Attrs, ins: &[Value], _: i64) -> Result<Built, String> {
    let x = float_arg(ins, 0)?;
    let dims = x.dims().to_vec();
    let kernel = a.ints("kernel_shape")?.ok_or("attribute kernel_shape missing")?;
    let sp = kernel.len();
    if !(1..=2).contains(&sp) || dims.len() != sp + 2 {
        return Err(format!("kernel {kernel:?} on input of shape {dims:?}"));
    }
    let strides_a = a.ints("strides")?.unwrap_or_else(|| vec![1; sp]);
    let pads = a.ints("pads")?;
    let dil = a.ints("dilations")?;
    let ceil = a.int("ceil_mode")?.unwrap_or(0);
    let auto = a.string("auto_pad")?;
    let order = a.int("storage_order")?.unwrap_or(0);
    if pads.is_some_and(|p| p.iter().any(|&v| v != 0)) {
        return Err("attribute pads".into());
    }
    if dil.is_some_and(|d| d.iter().any(|&v| v != 1)) {
        return Err("attribute dilations".into());
    }
    if ceil != 0 {
        return Err("attribute ceil_mode".into());
    }
    if auto.is_some_and(|s| s != "NOTSET") {
        return Err("attribute auto_pad".into());
    }
    if order != 0 {
        return Err("attribute storage_order".into());
    }
    let pos = |v: &[i64], what: &str| -> Result<Vec<usize>, String> {
        if v.len() != sp || v.iter().any(|&e| e < 1) {
            return Err(format!("{what} {v:?}"));
        }
        Ok(v.iter().map(|&e| e as usize).collect())
    };
    let (kernel, st) = (pos(&kernel, "kernel_shape")?, pos(&strides_a, "strides")?);
    // a 1-D pool is a 2-D one with height 1
    let (kh, kw, sh, sw, h, w) = if sp == 2 {
        (kernel[0], kernel[1], st[0], st[1], dims[2], dims[3])
    } else {
        (1, kernel[0], 1, st[0], 1, dims[2])
    };
    if kh > h || kw > w {
        return Err(format!("kernel {kernel:?} larger than input {dims:?}"));
    }
    let o = ((h - kh) / sh + 1, (w - kw) / sw + 1);
    let mut out = dims[..2].to_vec();
    if sp == 2 {
        out.push(o.0);
    }
    out.push(o.1);
    let nc = numel(&dims[..2])?;
    finish(Box::new(MaxPool { nc, h, w, k: (kh, kw), s: (sh, sw), o }), &ins[..1], out)
}

struct BatchNorm {
    scale: Vec<f32>,
    shift: Vec<f32>,
    inner: usize,
}

impl Op for BatchNorm {
    fn run(&self, ins: &[&[f32]], outs: &mut [&mut [f32]]) {
        let c = self.scale.len();
        for (p, (y, x)) in outs[0].chunks_mut(self.inner).zip(ins[0].chunks(self.inner)).enumerate() {
            let (s, b) = (self.scale[p % c], self.shift[p % c]);
            for (o, &v) in y.iter_mut().zip(x) {
                *o = fmaf(v, s, b);
            }
        }
    }
}

pub(super) fn batch_norm(_: &NodeProto, a: &mut Attrs, ins: &[Value], _: i64) -> Result<Built, String> {
    let x = float_arg(ins, 0)?;
    let eps = a.float("epsilon")?.unwrap_or(1e-5);
    a.ignore("momentum");
    if a.int("training_mode")?.unwrap_or(0) != 0 {
        return Err("attribute training_mode".into());
    }
    let dims = x.dims().to_vec();
    if dims.len() < 2 {
        return Err(format!("input of rank {}", dims.len()));
    }
    let c = dims[1];
    let param = |i: usize, what: &str| -> Result<Vec<f32>, String> {
        let t = float_arg(ins, i)?.konst().ok_or_else(|| format!("{what} computed at runtime"))?;
        match t.as_f32() {
            Some(v) if t.dims == [c] => Ok(v.to_vec()),
            _ => Err(format!("{what} of shape {:?}, want [{c}]", t.dims)),
        }
    };
    let (g, b, mu, var) = (param(1, "scale")?, param(2, "B")?, param(3, "mean")?, param(4, "var")?);
    if opt_arg(ins, 5).is_some() {
        return Err("more than five inputs".into());
    }
    // 0xb3e534: CPUNK Add, Sqrt, Div, Mul, Sub fold the parameters; the kernel then runs bionic fmaf
    let scale: Vec<f32> = g.iter().zip(&var).map(|(&g, &v)| zdiv(g, zsqrt(vadd(v, eps)))).collect();
    let shift: Vec<f32> = b.iter().zip(&mu).zip(&scale).map(|((&b, &m), &s)| vsub(b, vmul(m, s))).collect();
    let inner = numel(&dims[2..])?;
    if inner == 0 || c == 0 {
        return Err(format!("empty input {dims:?}"));
    }
    finish(Box::new(BatchNorm { scale, shift, inner }), &ins[..1], dims)
}
