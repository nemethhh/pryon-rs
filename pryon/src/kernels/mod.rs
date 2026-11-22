#[cfg(all(pryon_neon, target_arch = "arm"))]
mod armv7;
#[cfg(any(test, not(all(pryon_neon, target_arch = "arm"))))]
mod portable;

/// `out[r] = Σ_c W[r][c] k[c]` over `c < stride` for weights stored by `store_weight` in the `block_index` layout;
/// `out.len()` is a multiple of 4, `w.len() == out.len() * stride`, `stride` a multiple of 8, `k.len() >= stride`.
#[cfg(all(pryon_neon, target_arch = "arm"))]
pub(crate) use armv7::dot_rows;
#[cfg(not(all(pryon_neon, target_arch = "arm")))]
pub(crate) use portable::dot_rows;

/// `y[r] = bias[r] + (y[r] + ((rinv * acc[r]) * rs[r] + off * b[r] + sx * a[r]))` for `s = [rinv, off, sx]`, each
/// operation rounded in that order. NEON flushes subnormals as libpryon's NEON dequant does; the scalar twin agrees
/// whenever no intermediate is subnormal.
#[cfg(all(pryon_neon, target_arch = "arm"))]
pub(crate) use armv7::dequant_rows;
#[cfg(not(all(pryon_neon, target_arch = "arm")))]
pub(crate) use portable::dequant_rows;

/// `num::sigmoid` / `num::tanh` over a slice in place, bit for bit on finite inputs.
#[cfg(all(pryon_neon, target_arch = "arm"))]
pub(crate) use armv7::{sigmoid_inplace, tanh_inplace};
#[cfg(not(all(pryon_neon, target_arch = "arm")))]
pub(crate) use portable::{sigmoid_inplace, tanh_inplace};

/// int8 weight as stored: offset binary (w + 128).
pub(crate) fn store_weight(w: i8) -> u8 {
    w as u8 ^ 0x80
}

pub(crate) fn load_weight(w: u8) -> i8 {
    (w ^ 0x80) as i8
}

/// Position of `W[r][c]`: blocks of four rows by eight columns (row-major inside), so the kernel streams one run.
pub(crate) fn block_index(r: usize, c: usize, stride: usize) -> usize {
    (r / 4) * 4 * stride + (c / 8) * 32 + (r % 4) * 8 + c % 8
}

#[cfg(test)]
mod tests {
    use super::{block_index, load_weight, store_weight};

    #[cfg(all(pryon_neon, target_arch = "arm"))]
    fn xorshift(mut s: u32) -> impl FnMut() -> u32 {
        move || {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            s
        }
    }

    fn blocked(rows: usize, stride: usize, v: impl Fn(usize, usize) -> i8) -> Vec<u8> {
        let mut w = vec![0u8; rows * stride];
        for r in 0..rows {
            for c in 0..stride {
                w[block_index(r, c, stride)] = store_weight(v(r, c));
            }
        }
        w
    }

    #[test]
    fn weight_store_roundtrip() {
        assert_eq!((store_weight(-128), store_weight(0), store_weight(127)), (0, 128, 255));
        assert!((-128..=127).all(|w| load_weight(store_weight(w)) == w));
    }

    #[test]
    fn block_index_is_a_permutation() {
        let mut seen = vec![false; 8 * 24];
        for r in 0..8 {
            for c in 0..24 {
                assert!(!std::mem::replace(&mut seen[block_index(r, c, 24)], true));
            }
        }
    }

    #[test]
    fn dot_rows_each_row() {
        let (rows, stride) = (8, 24);
        let w = blocked(rows, stride, |r, c| ((r * 24 + c) * 37 % 256) as u8 as i8);
        let k: Vec<u8> = (0..stride).map(|i| ((i * 91) % 256) as u8).collect();
        let mut out = [0i32; 8];
        super::dot_rows(&w, stride, &k, &mut out);
        for (r, &o) in out.iter().enumerate() {
            let want: i32 = (0..stride).map(|c| load_weight(w[block_index(r, c, stride)]) as i32 * k[c] as i32).sum();
            assert_eq!(o, want, "row {r}");
        }
    }

    #[test]
    fn dot_rows_extremes() {
        let k = vec![255u8; 384];
        let mut out = [0i32; 4];
        super::dot_rows(&blocked(4, 384, |_, _| -128), 384, &k, &mut out);
        assert_eq!(out, [-12_533_760; 4]);
        super::dot_rows(&blocked(4, 384, |_, _| 127), 384, &k, &mut out);
        assert_eq!(out, [127 * 255 * 384; 4]);
        super::dot_rows(&blocked(4, 384, |_, _| 0), 384, &k, &mut out);
        assert_eq!(out, [0; 4]);
    }

    #[cfg(all(pryon_neon, target_arch = "arm"))]
    #[test]
    fn dot_rows_asm_equals_portable() {
        let mut next = xorshift(0x9e37_79b9);
        for rows in [4, 8, 12, 256, 384] {
            for stride in [8, 16, 24, 64, 96, 104, 512] {
                let w: Vec<u8> = (0..rows * stride).map(|_| next() as u8).collect();
                let k: Vec<u8> = (0..stride + 8).map(|_| next() as u8).collect();
                let (mut a, mut b) = (vec![0i32; rows], vec![1i32; rows]);
                super::dot_rows(&w, stride, &k, &mut a);
                super::portable::dot_rows(&w, stride, &k, &mut b);
                assert_eq!(a, b, "rows={rows} stride={stride}");
            }
        }
    }

    #[cfg(all(pryon_neon, target_arch = "arm"))]
    fn act_inputs() -> Vec<f32> {
        let mut next = xorshift(0x2545_f491);
        let mut v: Vec<f32> = [0.0, -0.0, 1e-40, -1e-40, 1e-20, -1e-20, 0.3465, -0.3466, 1.0, -1.0, 44.0, -44.0]
            .into_iter()
            .chain([87.0, -87.0, 87.4, -87.4, 88.5, -88.5, 100.0, -100.0, f32::MAX, f32::MIN, 1e-10, -1e-10])
            .collect();
        v.extend((-40_000..40_000).map(|i| i as f32 * 0.0025));
        v.extend((0..200_000).map(|_| f32::from_bits(next())).filter(|x| x.is_finite()));
        v.extend((0..100_000).map(|_| (next() as i32 as f32) * 2.4e-8));
        v.truncate(v.len() & !3);
        v.push(0.75);
        v
    }

    #[cfg(all(pryon_neon, target_arch = "arm"))]
    #[test]
    fn act_asm_equals_portable() {
        let x = act_inputs();
        for (name, asm, port) in [
            ("sigmoid", super::sigmoid_inplace as fn(&mut [f32]), super::portable::sigmoid_inplace as fn(&mut [f32])),
            ("tanh", super::tanh_inplace, super::portable::tanh_inplace),
        ] {
            let (mut a, mut b) = (x.clone(), x.clone());
            asm(&mut a);
            port(&mut b);
            for ((&i, &p), &q) in x.iter().zip(&a).zip(&b) {
                assert_eq!(p.to_bits(), q.to_bits(), "{name}({i:e} = {:#010x})", i.to_bits());
            }
        }
    }

    #[cfg(all(pryon_neon, target_arch = "arm"))]
    #[test]
    fn dequant_asm_equals_portable() {
        let mut next = xorshift(0x1b87_3593);
        let mut unit = move || (next() >> 8) as f32 / (1 << 23) as f32 - 1.0;
        for n in [1, 4, 7, 24, 256, 384] {
            for _ in 0..200 {
                let acc: Vec<i32> = (0..n).map(|_| (unit() * 3.2e6) as i32).collect();
                let s = [unit().abs() * 0.05 + 1e-4, unit() * 4.0, unit() * 60.0];
                let mut v = |scale: f32| (0..n).map(|_| unit() * scale).collect::<Vec<f32>>();
                let (rs, b, a, bias, y) = (v(0.02), v(300.0), v(0.5), v(2.0), v(8.0));
                let (mut ya, mut yb) = (y.clone(), y);
                super::dequant_rows(&acc, s, &rs, &b, &a, &bias, &mut ya);
                super::portable::dequant_rows(&acc, s, &rs, &b, &a, &bias, &mut yb);
                let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
                assert_eq!(bits(&ya), bits(&yb), "n={n} s={s:?}");
            }
        }
    }
}
