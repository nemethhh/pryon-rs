//! HybridGemm, HybridConv and HybridLSTM: the wake graphs' int8 operators, on [`crate::cpunk`]'s product.

use super::{arg, opt_arg, Attrs, Built, Op, Value};
use crate::cpunk::{qmatmul_row, QWeights, MAX_COLS};
use crate::num::{sigmoid, tanh};
use crate::onnx::proto::NodeProto;
use crate::onnx::tensor::{numel, Data, DType};

fn float_arg(ins: &[Value], i: usize) -> Result<&Value, String> {
    let v = arg(ins, i)?;
    if v.dtype() != Some(DType::F32) {
        return Err(format!("input {i} of type {:?}", v.dtype()));
    }
    Ok(v)
}

/// A constant int8 weight tensor: its dims and values.
fn weights(ins: &[Value], i: usize) -> Result<(&[usize], &[i8]), String> {
    let t = arg(ins, i)?.konst().ok_or_else(|| format!("input {i} (weights) computed at runtime"))?;
    match &t.data {
        Data::I8(v) => Ok((&t.dims, v)),
        _ => Err(format!("input {i} (weights) of type {:?}", t.dtype())),
    }
}

/// A float input (constant or runtime) of exactly `n` elements, read at run time (biases).
fn float_vec(ins: &[Value], i: usize, n: usize) -> Result<(), String> {
    let v = float_arg(ins, i)?;
    if numel(v.dims())? != n {
        return Err(format!("input {i} of shape {:?}, want {n} elements", v.dims()));
    }
    Ok(())
}

/// Scales, zero points and row sums at `base + 1..=base + 3`, completing the weights at `base`.
fn qweights(ins: &[Value], base: usize, rows: usize, cols: usize, w: Vec<i8>) -> Result<QWeights, String> {
    let konst = |i: usize| {
        let t = arg(ins, i)?.konst().ok_or_else(|| format!("input {i} computed at runtime"))?;
        if t.len() != rows {
            return Err(format!("input {i} of shape {:?}, want {rows} elements", t.dims));
        }
        Ok(t)
    };
    let sw = konst(base + 1)?.as_f32().ok_or_else(|| format!("input {} (scales) is not f32", base + 1))?.to_vec();
    let zw = konst(base + 2)?
        .to_i64s()?
        .into_iter()
        .map(|z| i32::try_from(z).map_err(|_| format!("zero point {z}")))
        .collect::<Result<Vec<_>, _>>()?;
    // sums (Σw per row) agree with the weights in every real graph; refuse a model where they mean something else
    let sums = konst(base + 3)?.to_i64s()?;
    QWeights::new(rows, cols, w, sw, zw, &sums)
}

fn no_more_inputs(ins: &[Value], from: usize) -> Result<(), String> {
    match (from..ins.len()).find(|&i| opt_arg(ins, i).is_some()) {
        Some(i) => Err(format!("unexpected input {i}")),
        None => Ok(()),
    }
}

struct Gemm {
    wq: QWeights,
    bias: bool,
}

impl Op for Gemm {
    fn run(&self, ins: &[&[f32]], outs: &mut [&mut [f32]]) {
        let mut q = [0i8; MAX_COLS];
        let bias = self.bias.then(|| ins[5]);
        for (x, y) in ins[0].chunks_exact(self.wq.cols).zip(outs[0].chunks_exact_mut(self.wq.rows)) {
            qmatmul_row(x, &self.wq, bias, &mut q, y);
        }
    }
}

/// `(x, w, sw, zw, sums[, bias])`: rows of `x` along its last axis times `w` (`[N, K]`, or `[K, N]` with `transB = 0`).
pub(super) fn hybrid_gemm(_: &NodeProto, a: &mut Attrs, ins: &[Value], _: i64) -> Result<Built, String> {
    let x = float_arg(ins, 0)?;
    let (wd, w) = weights(ins, 1)?;
    let trans_b = a.int("transB")?.unwrap_or(1);
    let (n, k, w) = match (trans_b, wd) {
        (1, &[n, k]) => (n, k, w.to_vec()),
        (0, &[k, n]) => (n, k, (0..n * k).map(|i| w[(i % k) * n + i / k]).collect()),
        (0 | 1, _) => return Err(format!("weights of shape {wd:?}")),
        _ => return Err(format!("attribute transB = {trans_b}")),
    };
    let xd = x.dims();
    if xd.last() != Some(&k) {
        return Err(format!("input of shape {xd:?} against weights of {k} columns"));
    }
    let wq = qweights(ins, 1, n, k, w)?;
    let bias = opt_arg(ins, 5).is_some();
    if bias {
        float_vec(ins, 5, n)?;
    }
    no_more_inputs(ins, 6)?;
    let out: Vec<usize> = xd[..xd.len() - 1].iter().copied().chain([n]).collect();
    numel(&out)?;
    Ok(Built::Runtime { op: Box::new(Gemm { wq, bias }), outs: vec![(out, DType::F32)] })
}

struct Conv {
    wq: QWeights,
    bias: bool,
    /// Input `[batch, c, h, w]`, kernel `(kh, kw)`, strides and output `(oh, ow)`.
    x: [usize; 4],
    k: (usize, usize),
    s: (usize, usize),
    o: (usize, usize),
}

impl Op for Conv {
    fn run(&self, ins: &[&[f32]], outs: &mut [&mut [f32]]) {
        let [batch, c, h, w] = self.x;
        let ((kh, kw), (oh, ow), n) = (self.k, self.o, self.wq.rows);
        let (y, scratch) = outs.split_at_mut(1);
        let (patch, row) = scratch[0].split_at_mut(self.wq.cols);
        let mut q = [0i8; MAX_COLS];
        let bias = self.bias.then(|| ins[5]);
        for b in 0..batch {
            for i in 0..oh {
                for j in 0..ow {
                    // patch order (C, kh, kw), as the weights are flattened
                    for ci in 0..c {
                        for u in 0..kh {
                            let at = ((b * c + ci) * h + i * self.s.0 + u) * w + j * self.s.1;
                            patch[(ci * kh + u) * kw..][..kw].copy_from_slice(&ins[0][at..at + kw]);
                        }
                    }
                    qmatmul_row(patch, &self.wq, bias, &mut q, row);
                    for (f, &v) in row.iter().enumerate() {
                        y[0][((b * n + f) * oh + i) * ow + j] = v;
                    }
                }
            }
        }
    }

    fn scratch(&self) -> usize {
        self.wq.cols + self.wq.rows
    }
}

/// `(x, w, sw, zw, sums, bias)`: valid NCHW convolution through im2col rows, bias added in the kernel.
pub(super) fn hybrid_conv(_: &NodeProto, a: &mut Attrs, ins: &[Value], _: i64) -> Result<Built, String> {
    let x = float_arg(ins, 0)?;
    let &[batch, c, h, w] = x.dims() else { return Err(format!("input of shape {:?}", x.dims())) };
    let pair = |v: Option<Vec<i64>>, what: &str| -> Result<(usize, usize), String> {
        match v.as_deref().unwrap_or(&[1, 1]) {
            &[p, q] if p >= 1 && q >= 1 => Ok((p as usize, q as usize)),
            o => Err(format!("attribute {what} = {o:?}")),
        }
    };
    let k = pair(Some(a.ints("kernel_shape")?.ok_or("attribute kernel_shape missing")?), "kernel_shape")?;
    let s = pair(a.ints("strides")?, "strides")?;
    if pair(a.ints("dilations")?, "dilations")? != (1, 1) {
        return Err("attribute dilations other than [1, 1]".into());
    }
    if a.int("group")?.unwrap_or(1) != 1 {
        return Err("attribute group other than 1".into());
    }
    if a.string("data_format")?.is_some_and(|f| f != "NCHW") {
        return Err("attribute data_format other than NCHW".into());
    }
    if k.0 > h || k.1 > w {
        return Err(format!("kernel {k:?} larger than input {:?}", x.dims()));
    }
    let (wd, wv) = weights(ins, 1)?;
    let cols = c * k.0 * k.1;
    let n = match wd {
        &[n, kk] if kk == cols => n,
        &[n, wc, wh, ww] if (wc, wh, ww) == (c, k.0, k.1) => n,
        _ => return Err(format!("weights of shape {wd:?} for {c} channels and kernel {k:?}")),
    };
    let wq = qweights(ins, 1, n, cols, wv.to_vec())?;
    let bias = opt_arg(ins, 5).is_some();
    if bias {
        float_vec(ins, 5, n)?;
    }
    no_more_inputs(ins, 6)?;
    let o = ((h - k.0) / s.0 + 1, (w - k.1) / s.1 + 1);
    let out = vec![batch, n, o.0, o.1];
    numel(&out)?;
    let op = Conv { wq, bias, x: [batch, c, h, w], k, s, o };
    Ok(Built::Runtime { op: Box::new(op), outs: vec![(out, DType::F32)] })
}

struct Lstm {
    w: QWeights,
    r: QWeights,
    hid: usize,
    lanes: usize,
    steps: usize,
    /// Rows of `x` per step: 1 (broadcast to every lane) or one per lane.
    xrows: usize,
    nout: usize,
}

impl Op for Lstm {
    fn run(&self, ins: &[&[f32]], outs: &mut [&mut [f32]]) {
        let (hid, lanes, g4) = (self.hid, self.lanes, 4 * self.hid);
        let (o, scratch) = outs.split_at_mut(self.nout);
        let (h, rest) = scratch[0].split_at_mut(lanes * hid);
        let (c, rest) = rest.split_at_mut(lanes * hid);
        let (gx, g) = rest.split_at_mut(self.xrows * g4);
        h.copy_from_slice(ins[11]);
        c.copy_from_slice(ins[12]);
        let (wb, rb) = ins[9].split_at(g4);
        let mut q = [0i8; MAX_COLS];
        let n_in = self.w.cols;
        for t in 0..self.steps {
            for (r, gr) in gx.chunks_exact_mut(g4).enumerate() {
                qmatmul_row(&ins[0][(t * self.xrows + r) * n_in..][..n_in], &self.w, None, &mut q, gr);
            }
            for l in 0..lanes {
                let (hl, cl) = (&mut h[l * hid..][..hid], &mut c[l * hid..][..hid]);
                qmatmul_row(hl, &self.r, None, &mut q, g);
                let gxl = &gx[if self.xrows == 1 { 0 } else { l * g4 }..][..g4];
                for j in 0..g4 {
                    g[j] = ((gxl[j] + g[j]) + wb[j]) + rb[j];
                }
                // gates i, o, f, c
                for u in 0..hid {
                    let (i, og, f) = (sigmoid(g[u]), sigmoid(g[hid + u]), sigmoid(g[2 * hid + u]));
                    let cn = f * cl[u] + i * tanh(g[3 * hid + u]);
                    cl[u] = cn;
                    hl[u] = og * tanh(cn);
                }
                o[0][(t * lanes + l) * hid..][..hid].copy_from_slice(hl);
            }
        }
        if self.nout > 1 {
            o[1].copy_from_slice(h);
        }
        if self.nout > 2 {
            o[2].copy_from_slice(c);
        }
    }

    fn scratch(&self) -> usize {
        (2 * self.lanes * self.hid) + (self.xrows + 1) * 4 * self.hid
    }
}

/// `(x, W, sw, zw, sums, R, sw_r, zw_r, sums_r, B, —, h0, c0)`: a forward LSTM over `x [T, 1 or lanes, n]`,
/// gates i, o, f, c; outputs `Y [T, 1, lanes, h]`, `Y_h` and `Y_c [1, lanes, h]`.
pub(super) fn hybrid_lstm(n: &NodeProto, a: &mut Attrs, ins: &[Value], _: i64) -> Result<Built, String> {
    let x = float_arg(ins, 0)?;
    let (wd, wv) = weights(ins, 1)?;
    let (rd, rv) = weights(ins, 5)?;
    let &[1, g4, hid] = rd else { return Err(format!("recurrence weights of shape {rd:?}")) };
    if g4 != 4 * hid || hid == 0 {
        return Err(format!("recurrence weights of shape {rd:?}"));
    }
    if let Some(hs) = a.int("hidden_size")? {
        if hs != hid as i64 {
            return Err(format!("attribute hidden_size = {hs}, recurrence weights of shape {rd:?}"));
        }
    }
    if let Some(act) = a.strings("activations")? {
        if act != ["Sigmoid", "Tanh", "Tanh"] {
            return Err(format!("attribute activations = {act:?}"));
        }
    }
    if let Some(dir) = a.string("direction")? {
        if dir != "forward" {
            return Err(format!("attribute direction = {dir}"));
        }
    }
    let &[steps, xrows, n_in] = x.dims() else { return Err(format!("input of shape {:?}", x.dims())) };
    if wd != [1, g4, n_in] {
        return Err(format!("input weights of shape {wd:?} for input {:?}", x.dims()));
    }
    let w = qweights(ins, 1, g4, n_in, wv.to_vec())?;
    let r = qweights(ins, 5, g4, hid, rv.to_vec())?;
    float_vec(ins, 9, 2 * g4)?;
    if opt_arg(ins, 10).is_some() {
        return Err("input 10 (sequence_lens)".into());
    }
    let h0 = float_arg(ins, 11)?.dims();
    let &[1, lanes, hh] = h0 else { return Err(format!("initial state of shape {h0:?}")) };
    if hh != hid || float_arg(ins, 12)?.dims() != h0 {
        return Err(format!("initial states of shape {h0:?} and {:?}", ins[12].dims()));
    }
    if xrows != 1 && xrows != lanes {
        return Err(format!("input of shape {:?} for {lanes} lanes", x.dims()));
    }
    no_more_inputs(ins, 13)?;
    let nout = n.outputs.len();
    if !(1..=3).contains(&nout) {
        return Err(format!("{nout} outputs"));
    }
    let y = vec![steps, 1, lanes, hid];
    numel(&y)?;
    let mut outs = vec![(y, DType::F32), (h0.to_vec(), DType::F32), (h0.to_vec(), DType::F32)];
    outs.truncate(nout);
    let op = Lstm { w, r, hid, lanes, steps, xrows, nout };
    numel(&[op.scratch()])?;
    Ok(Built::Runtime { op: Box::new(op), outs })
}
