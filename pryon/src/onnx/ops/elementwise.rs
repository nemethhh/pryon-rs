//! Elementwise arithmetic with NumPy broadcasting; integer constants fold with wrapping arithmetic.
//! Float Add/Sub/Mul are libpryon's CPUNK NEON kernels (0x123461c, 0x124c79c, 0x1247910): flush to zero, default NaN.

use super::{arg, each_index, fold_numel, strides, Attrs, Built, Op, Value};
use crate::num::{vadd, vmul, vsub};
use crate::onnx::proto::NodeProto;
use crate::onnx::tensor::{numel, Data, Tensor};
use std::marker::PhantomData;

pub(crate) trait BinKind: Send + Sync + 'static {
    fn f32(a: f32, b: f32) -> f32;
    fn int(a: i64, b: i64) -> i64;
}

pub(crate) struct Add;
pub(crate) struct Sub;
pub(crate) struct Mul;

impl BinKind for Add {
    fn f32(a: f32, b: f32) -> f32 {
        vadd(a, b)
    }
    fn int(a: i64, b: i64) -> i64 {
        a.wrapping_add(b)
    }
}

impl BinKind for Sub {
    fn f32(a: f32, b: f32) -> f32 {
        vsub(a, b)
    }
    fn int(a: i64, b: i64) -> i64 {
        a.wrapping_sub(b)
    }
}

impl BinKind for Mul {
    fn f32(a: f32, b: f32) -> f32 {
        vmul(a, b)
    }
    fn int(a: i64, b: i64) -> i64 {
        a.wrapping_mul(b)
    }
}

/// Output as rows of `n` contiguous elements; within a row input a (b) advances by `sa` (`sb`), 0 or 1.
struct Bcast {
    n: usize,
    sa: usize,
    sb: usize,
    rows: Vec<(usize, usize)>,
}

fn pad(d: &[usize], r: usize) -> Vec<usize> {
    [vec![1; r - d.len()], d.to_vec()].concat()
}

fn broadcast_shape(a: &[usize], b: &[usize]) -> Result<Vec<usize>, String> {
    let r = a.len().max(b.len());
    let (pa, pb) = (pad(a, r), pad(b, r));
    let out = (0..r)
        .map(|k| match (pa[k], pb[k]) {
            (x, y) if x == y => Ok(x),
            (1, y) => Ok(y),
            (x, 1) => Ok(x),
            _ => Err(format!("shapes {a:?} and {b:?} do not broadcast")),
        })
        .collect::<Result<Vec<_>, _>>()?;
    numel(&out)?;
    Ok(out)
}

/// Plan for broadcasting `a` and `b` to `out` (from `broadcast_shape`).
fn broadcast(a: &[usize], b: &[usize], out: &[usize]) -> Result<Bcast, String> {
    let r = out.len();
    let (pa, pb) = (pad(a, r), pad(b, r));
    if numel(out)? == 0 {
        return Ok(Bcast { n: 1, sa: 0, sb: 0, rows: Vec::new() });
    }
    let st = |p: &[usize]| {
        let s = strides(p);
        (0..r).map(|k| if p[k] == 1 { 0 } else { s[k] }).collect::<Vec<_>>()
    };
    let (sa, sb) = (st(&pa), st(&pb));
    // size-1 output axes carry nothing; merge inner axes while both inputs stay linear across them
    let kept: Vec<usize> = (0..r).filter(|&k| out[k] != 1).collect();
    let (mut n, mut ua, mut ub, mut m) = (1, 0, 0, kept.len());
    if let Some(&last) = kept.last() {
        (n, ua, ub, m) = (out[last], sa[last], sb[last], m - 1);
        while m > 0 {
            let k = kept[m - 1];
            let linear = |s: &[usize], u: usize| s[k] == u * n;
            if !(linear(&sa, ua) && linear(&sb, ub)) {
                break;
            }
            n *= out[k];
            m -= 1;
        }
    }
    let outer: Vec<usize> = kept[..m].iter().map(|&k| out[k]).collect();
    let mut rows = Vec::new();
    each_index(&outer, |idx| {
        let off = |s: &[usize]| idx.iter().zip(&kept[..m]).map(|(&i, &k)| i * s[k]).sum::<usize>();
        rows.push((off(&sa), off(&sb)));
    })?;
    Ok(Bcast { n, sa: ua, sb: ub, rows })
}

impl Bcast {
    #[inline(always)]
    fn apply<T: Copy>(&self, a: &[T], b: &[T], out: &mut [T], f: impl Fn(T, T) -> T) {
        let n = self.n;
        for (o, &(oa, ob)) in out.chunks_exact_mut(n).zip(&self.rows) {
            match (self.sa, self.sb) {
                (1, 1) => {
                    for ((y, &x), &z) in o.iter_mut().zip(&a[oa..oa + n]).zip(&b[ob..ob + n]) {
                        *y = f(x, z);
                    }
                }
                (1, _) => {
                    let z = b[ob];
                    for (y, &x) in o.iter_mut().zip(&a[oa..oa + n]) {
                        *y = f(x, z);
                    }
                }
                (_, 1) => {
                    let x = a[oa];
                    for (y, &z) in o.iter_mut().zip(&b[ob..ob + n]) {
                        *y = f(x, z);
                    }
                }
                _ => o.fill(f(a[oa], b[ob])),
            }
        }
    }
}

struct Binary<K> {
    plan: Bcast,
    k: PhantomData<K>,
}

impl<K: BinKind> Op for Binary<K> {
    fn run(&self, ins: &[&[f32]], outs: &mut [&mut [f32]]) {
        self.plan.apply(ins[0], ins[1], outs[0], K::f32);
    }
}

pub(super) fn binary<K: BinKind>(_: &NodeProto, _: &mut Attrs, ins: &[Value], _: i64) -> Result<Built, String> {
    let (a, b) = (arg(ins, 0)?, arg(ins, 1)?);
    if a.dtype() != b.dtype() {
        return Err(format!("element types {:?} and {:?}", a.dtype(), b.dtype()));
    }
    let dims = broadcast_shape(a.dims(), b.dims())?;
    if let (Some(x), Some(y)) = (a.konst(), b.konst()) {
        let n = fold_numel(&dims)?;
        let plan = broadcast(x.dims.as_slice(), &y.dims, &dims)?;
        let data = match (&x.data, &y.data) {
            (Data::F32(x), Data::F32(y)) => Data::F32(fold(&plan, x, y, n, K::f32)),
            (Data::I64(x), Data::I64(y)) => Data::I64(fold(&plan, x, y, n, K::int)),
            (Data::I32(x), Data::I32(y)) => Data::I32(fold(&plan, x, y, n, |p, q| K::int(p.into(), q.into()) as i32)),
            _ => return Err(format!("constants of type {:?}", x.dtype())),
        };
        return Ok(Built::Folded(vec![Tensor { dims, data }]));
    }
    let dtype = a.dtype().ok_or("input missing")?;
    let plan = broadcast(a.dims(), b.dims(), &dims)?;
    Ok(Built::Runtime { op: Box::new(Binary::<K> { plan, k: PhantomData }), outs: vec![(dims, dtype)] })
}

fn fold<T: Copy + Default>(plan: &Bcast, a: &[T], b: &[T], n: usize, f: impl Fn(T, T) -> T) -> Vec<T> {
    let mut out = vec![T::default(); n];
    plan.apply(a, b, &mut out, f);
    out
}

pub(crate) trait UnKind: Send + Sync + 'static {
    fn f32(x: f32) -> f32;
    fn int(x: i64) -> i64;
}

pub(crate) struct Neg;
pub(crate) struct Relu;

impl UnKind for Neg {
    fn f32(x: f32) -> f32 {
        -x
    }
    fn int(x: i64) -> i64 {
        x.wrapping_neg()
    }
}

impl UnKind for Relu {
    fn f32(x: f32) -> f32 {
        if x < 0.0 {
            0.0
        } else {
            x
        }
    }
    fn int(x: i64) -> i64 {
        x.max(0)
    }
}

struct Unary<K>(PhantomData<K>);

impl<K: UnKind> Op for Unary<K> {
    fn run(&self, ins: &[&[f32]], outs: &mut [&mut [f32]]) {
        for (y, &x) in outs[0].iter_mut().zip(ins[0]) {
            *y = K::f32(x);
        }
    }
}

pub(super) fn unary<K: UnKind>(_: &NodeProto, _: &mut Attrs, ins: &[Value], _: i64) -> Result<Built, String> {
    let x = arg(ins, 0)?;
    if let Some(t) = x.konst() {
        fold_numel(&t.dims)?;
        let data = match &t.data {
            Data::F32(v) => Data::F32(v.iter().map(|&x| K::f32(x)).collect()),
            Data::I64(v) => Data::I64(v.iter().map(|&x| K::int(x)).collect()),
            Data::I32(v) => Data::I32(v.iter().map(|&x| K::int(x.into()) as i32).collect()),
            _ => return Err(format!("constant of type {:?}", t.dtype())),
        };
        return Ok(Built::Folded(vec![Tensor { dims: t.dims.clone(), data }]));
    }
    let dtype = x.dtype().ok_or("input missing")?;
    Ok(Built::Runtime { op: Box::new(Unary::<K>(PhantomData)), outs: vec![(x.dims().to_vec(), dtype)] })
}
