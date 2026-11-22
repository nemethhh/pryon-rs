//! Shape and data-movement operators; with bound input shapes their integer side always folds at load.

use super::{arg, axes_of, axis, const_ints, copy_op, each_index, opt_arg, strides, view, Attrs, Built, Value};
use crate::onnx::proto::NodeProto;
use crate::onnx::tensor::{numel, DType, Tensor, MAX_ELEMS};

pub(super) fn shape(_: &NodeProto, a: &mut Attrs, ins: &[Value], _: i64) -> Result<Built, String> {
    a.ignore("dtype"); // tf2onnx output-type hint; Shape always yields int64
    let dims = arg(ins, 0)?.dims();
    let r = dims.len() as i64;
    let clamp = |v: i64| (if v < 0 { v.saturating_add(r) } else { v }).clamp(0, r) as usize;
    let start = clamp(a.int("start")?.unwrap_or(0));
    let end = clamp(a.int("end")?.unwrap_or(r)).max(start);
    let v: Vec<i64> = dims[start..end].iter().map(|&d| d as i64).collect();
    Ok(Built::Folded(vec![Tensor::i64(vec![v.len()], v)]))
}

pub(super) fn cast(_: &NodeProto, a: &mut Attrs, ins: &[Value], _: i64) -> Result<Built, String> {
    let to = a.int("to")?.ok_or("attribute to missing")?;
    let to = i32::try_from(to).ok().and_then(DType::from_onnx).ok_or_else(|| format!("cast to element type {to}"))?;
    match arg(ins, 0)? {
        Value::Const(t) => Ok(Built::Folded(vec![t.cast(to)])),
        x => {
            let mut b = view(x, x.dims().to_vec())?;
            if let Built::Runtime { outs, .. } = &mut b {
                outs[0].1 = to;
            }
            Ok(b)
        }
    }
}

pub(super) fn reshape(_: &NodeProto, a: &mut Attrs, ins: &[Value], _: i64) -> Result<Built, String> {
    let x = arg(ins, 0)?;
    let shape = const_ints(arg(ins, 1)?, "shape")?;
    let allowzero = a.int("allowzero")?.unwrap_or(0) != 0;
    let n = numel(x.dims())?;
    let bad = || format!("cannot reshape {:?} to {shape:?}", x.dims());
    let (mut out, mut infer) = (Vec::with_capacity(shape.len()), None);
    for (k, &s) in shape.iter().enumerate() {
        out.push(match s {
            -1 if infer.is_none() => {
                infer = Some(k);
                1
            }
            0 if !allowzero => *x.dims().get(k).ok_or_else(bad)?,
            s => usize::try_from(s).map_err(|_| bad())?,
        });
    }
    let known = numel(&out)?;
    if let Some(k) = infer {
        if known == 0 || n % known != 0 {
            return Err(bad());
        }
        out[k] = n / known;
    }
    if numel(&out)? != n {
        return Err(bad());
    }
    view(x, out)
}

pub(super) fn squeeze(_: &NodeProto, a: &mut Attrs, ins: &[Value], _: i64) -> Result<Built, String> {
    let x = arg(ins, 0)?;
    let dims = x.dims();
    let out = match axes_of(a, ins, 1)? {
        None => dims.iter().copied().filter(|&d| d != 1).collect(),
        Some(axes) => {
            let mut drop = vec![false; dims.len()];
            for &ax in &axes {
                let k = axis(ax, dims.len())?;
                if dims[k] != 1 {
                    return Err(format!("squeeze of axis {ax} with size {}", dims[k]));
                }
                drop[k] = true;
            }
            dims.iter().zip(&drop).filter(|(_, &d)| !d).map(|(&n, _)| n).collect()
        }
    };
    view(x, out)
}

pub(super) fn unsqueeze(_: &NodeProto, a: &mut Attrs, ins: &[Value], _: i64) -> Result<Built, String> {
    let x = arg(ins, 0)?;
    let axes = axes_of(a, ins, 1)?.ok_or("axes missing")?;
    let r = x.dims().len() + axes.len();
    let mut new = vec![false; r];
    for &ax in &axes {
        let k = axis(ax, r)?;
        if std::mem::replace(&mut new[k], true) {
            return Err(format!("axis {ax} repeated"));
        }
    }
    let mut it = x.dims().iter();
    let out = new.iter().map(|&n| if n { 1 } else { *it.next().unwrap_or(&1) }).collect();
    view(x, out)
}

pub(super) fn concat(_: &NodeProto, a: &mut Attrs, ins: &[Value], _: i64) -> Result<Built, String> {
    let ax = a.int("axis")?.ok_or("attribute axis missing")?;
    let parts = (0..ins.len()).map(|i| arg(ins, i)).collect::<Result<Vec<_>, _>>()?;
    let first = parts.first().ok_or("no inputs")?.dims();
    let k = axis(ax, first.len())?;
    let mut out = first.to_vec();
    out[k] = 0;
    for p in &parts {
        let d = p.dims();
        if d.len() != first.len() || (0..d.len()).any(|j| j != k && d[j] != first[j]) {
            return Err(format!("shapes {first:?} and {d:?} do not concatenate on axis {ax}"));
        }
        out[k] = out[k].checked_add(d[k]).ok_or("concatenated axis overflows")?;
    }
    numel(&out)?;
    let outer: usize = first[..k].iter().product();
    let inner: Vec<usize> = parts.iter().map(|p| p.dims()[k..].iter().product()).collect();
    let mut map = Vec::new();
    for o in 0..outer {
        for (i, &n) in inner.iter().enumerate() {
            map.extend((0..n).map(|e| (i, o * n + e)));
        }
    }
    copy_op(ins, parts.len(), map, out)
}

pub(super) fn slice(_: &NodeProto, a: &mut Attrs, ins: &[Value], _: i64) -> Result<Built, String> {
    let x = arg(ins, 0)?;
    let dims = x.dims();
    let r = dims.len();
    let ints = |i: usize, what: &str| opt_arg(ins, i).map(|v| const_ints(v, what)).transpose();
    let (starts, ends, axes, steps) = if ins.len() > 1 {
        (ints(1, "starts")?.ok_or("starts missing")?, ints(2, "ends")?.ok_or("ends missing")?, ints(3, "axes")?, ints(4, "steps")?)
    } else {
        (a.ints("starts")?.ok_or("starts missing")?, a.ints("ends")?.ok_or("ends missing")?, a.ints("axes")?, None)
    };
    let n = starts.len();
    let axes = axes.unwrap_or_else(|| (0..n as i64).collect());
    let steps = steps.unwrap_or_else(|| vec![1; n]);
    if ends.len() != n || axes.len() != n || steps.len() != n {
        return Err("starts, ends, axes and steps differ in length".into());
    }
    let (mut start, mut step, mut out, mut seen) = (vec![0i64; r], vec![1i64; r], dims.to_vec(), vec![false; r]);
    for j in 0..n {
        let k = axis(axes[j], r)?;
        if std::mem::replace(&mut seen[k], true) {
            return Err(format!("axis {} repeated", axes[j]));
        }
        let (d, st) = (dims[k] as i64, steps[j]);
        let norm = |v: i64| if v < 0 { v.saturating_add(d) } else { v };
        let (s, len) = if st == 0 {
            return Err("step 0".into());
        } else if d == 0 {
            (0, 0)
        } else if st > 0 {
            let (s, e) = (norm(starts[j]).clamp(0, d), norm(ends[j]).clamp(0, d));
            (s, if e > s { (e - s - 1) as u64 / st.unsigned_abs() + 1 } else { 0 })
        } else {
            let (s, e) = (norm(starts[j]).clamp(0, d - 1), norm(ends[j]).clamp(-1, d - 1));
            (s, if s > e { (s - e - 1) as u64 / st.unsigned_abs() + 1 } else { 0 })
        };
        (start[k], step[k], out[k]) = (s, st, len as usize);
    }
    let sx = strides(dims);
    let mut map = Vec::new();
    each_index(&out, |idx| {
        let off: i64 = (0..r).map(|k| (start[k] + idx[k] as i64 * step[k]) * sx[k] as i64).sum();
        map.push((0, off as usize));
    })?;
    copy_op(ins, 1, map, out)
}

pub(super) fn gather(_: &NodeProto, a: &mut Attrs, ins: &[Value], _: i64) -> Result<Built, String> {
    let (x, iv) = (arg(ins, 0)?, arg(ins, 1)?);
    let idx = const_ints(iv, "indices")?;
    let dims = x.dims();
    let k = axis(a.int("axis")?.unwrap_or(0), dims.len())?;
    let d = dims[k];
    let idx = idx
        .iter()
        .map(|&i| {
            let j = if i < 0 { i + d as i64 } else { i };
            usize::try_from(j).ok().filter(|&j| j < d).ok_or_else(|| format!("index {i} out of range for size {d}"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let out = [&dims[..k], iv.dims(), &dims[k + 1..]].concat();
    numel(&out)?;
    let (outer, inner): (usize, usize) = (dims[..k].iter().product(), dims[k + 1..].iter().product());
    let mut map = Vec::new();
    for o in 0..outer {
        for &i in &idx {
            map.extend((0..inner).map(|e| (0, (o * d + i) * inner + e)));
        }
    }
    copy_op(ins, 1, map, out)
}

pub(super) fn tile(_: &NodeProto, _: &mut Attrs, ins: &[Value], _: i64) -> Result<Built, String> {
    let x = arg(ins, 0)?;
    let reps = const_ints(arg(ins, 1)?, "repeats")?;
    let dims = x.dims();
    if reps.len() != dims.len() {
        return Err(format!("{} repeats for rank {}", reps.len(), dims.len()));
    }
    let out = dims
        .iter()
        .zip(&reps)
        .map(|(&d, &r)| {
            let r = usize::try_from(r).map_err(|_| format!("negative repeat {r}"))?;
            d.checked_mul(r).ok_or_else(|| format!("tile of {dims:?} by {reps:?}: more than {MAX_ELEMS} elements"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let s = strides(dims);
    let mut map = Vec::new();
    each_index(&out, |idx| map.push((0, (0..dims.len()).map(|k| idx[k] % dims[k] * s[k]).sum())))?;
    copy_op(ins, 1, map, out)
}

pub(super) fn transpose(_: &NodeProto, a: &mut Attrs, ins: &[Value], _: i64) -> Result<Built, String> {
    let x = arg(ins, 0)?;
    let dims = x.dims();
    let r = dims.len();
    let perm = a.ints("perm")?.unwrap_or_else(|| (0..r as i64).rev().collect());
    let mut seen = vec![false; r];
    let perm = perm
        .iter()
        .map(|&p| usize::try_from(p).ok().filter(|&p| p < r && !std::mem::replace(&mut seen[p], true)))
        .collect::<Option<Vec<_>>>()
        .filter(|p| p.len() == r)
        .ok_or_else(|| format!("perm {perm:?} for rank {r}"))?;
    let out: Vec<usize> = perm.iter().map(|&p| dims[p]).collect();
    let s = strides(dims);
    let mut map = Vec::new();
    each_index(&out, |idx| map.push((0, (0..r).map(|k| idx[k] * s[perm[k]]).sum())))?;
    copy_op(ins, 1, map, out)
}
