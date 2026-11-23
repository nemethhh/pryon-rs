pub(crate) fn dot_rows(w: &[u8], stride: usize, k: &[u8], out: &mut [i32]) {
    for (group, o) in w.chunks_exact(4 * stride).zip(out.chunks_exact_mut(4)) {
        o.fill(0);
        for (block, kc) in group.chunks_exact(32).zip(k[..stride].chunks_exact(8)) {
            for (row, acc) in block.chunks_exact(8).zip(o.iter_mut()) {
                *acc += row.iter().zip(kc).map(|(&a, &b)| (a as i32 - 128) * b as i32).sum::<i32>();
            }
        }
    }
}

pub(crate) fn sigmoid_inplace(x: &mut [f32]) {
    x.iter_mut().for_each(|v| *v = crate::num::sigmoid(*v));
}

pub(crate) fn tanh_inplace(x: &mut [f32]) {
    x.iter_mut().for_each(|v| *v = crate::num::tanh(*v));
}

pub(crate) fn dequant_rows(acc: &[i32], [rinv, off, sx]: [f32; 3], rs: &[f32], b: &[f32], a: &[f32], bias: &[f32], y: &mut [f32]) {
    for (r, yr) in y.iter_mut().enumerate() {
        let mut t = (rinv * acc[r] as f32) * rs[r];
        t += off * b[r];
        t += sx * a[r];
        *yr = bias[r] + (*yr + t);
    }
}

pub(crate) fn dot_i8_rows(w: &[i8], cols: usize, q: &[i8], out: &mut [i32]) {
    for (row, o) in w.chunks_exact(cols).zip(out.iter_mut()) {
        *o = row.iter().zip(&q[..cols]).map(|(&a, &b)| i32::from(a) * i32::from(b)).sum();
    }
}

pub(crate) fn cpunk_exp_inplace(x: &mut [f32]) {
    x.iter_mut().for_each(|v| *v = crate::num::cpunk_exp(*v));
}

pub(crate) fn fmaf_inplace(a: &[f32], b: &[f32], c: &mut [f32]) {
    for ((o, &x), &y) in c.iter_mut().zip(a).zip(b) {
        *o = crate::num::fmaf(x, y, *o);
    }
}

pub(crate) fn cpunk_minmax(x: &[f32]) -> (f32, f32) {
    let (mut lo, mut hi) = (f32::MAX, -f32::MAX);
    for &v in x {
        (lo, hi) = (crate::num::vmin(lo, v), crate::num::vmax(hi, v));
    }
    (lo, hi)
}

pub(crate) fn cpunk_quant4(x: &[f32], inv: f32, zb: i32, q: &mut [i8]) -> i32 {
    let mut sum = 0i32;
    for (o, &v) in q.iter_mut().zip(x) {
        // 1.5·2^23 + v·inv: the float's low bits are the rounded integer while |v·inv| < 2^22
        let r = (crate::num::vmla(12582912.0, inv, v).to_bits() as i32).wrapping_add(zb).clamp(0, 127);
        *o = r as i8;
        sum = sum.wrapping_add(r);
    }
    sum
}

pub(crate) fn cpunk_deq(dot: &[i32], w: [&[f32]; 3], bias: Option<&[f32]>, [c, zf, sx]: [f32; 3], y: &mut [f32]) {
    use crate::num::{vmla, vmul, vsub};
    let [zw, sums, sw] = w;
    for (r, o) in y.iter_mut().enumerate() {
        let corr = vmla(vmul(c, zw[r]), zf, sums[r]);
        let acc = vsub(dot[r] as f32, corr);
        let scale = vmul(sx, sw[r]);
        *o = match bias {
            Some(b) => vmla(vmul(b[r], 1.0), scale, acc),
            None => vmul(scale, acc),
        };
    }
}
