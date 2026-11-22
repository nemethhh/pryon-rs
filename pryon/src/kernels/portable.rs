pub(crate) fn dot_rows(w: &[i8], stride: usize, k: &[u8], out: &mut [i32]) {
    for (group, o) in w.chunks_exact(4 * stride).zip(out.chunks_exact_mut(4)) {
        o.fill(0);
        for (block, kc) in group.chunks_exact(32).zip(k[..stride].chunks_exact(8)) {
            for (row, acc) in block.chunks_exact(8).zip(o.iter_mut()) {
                *acc += row.iter().zip(kc).map(|(&a, &b)| a as i32 * b as i32).sum::<i32>();
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
        t = t + off * b[r];
        t = t + sx * a[r];
        *yr = bias[r] + (*yr + t);
    }
}
