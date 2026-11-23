#[cfg(all(pryon_neon, target_arch = "arm"))]
mod armv7;
#[cfg_attr(all(pryon_neon, target_arch = "arm", not(test)), allow(dead_code))]
mod portable;

/// `out[r] = Σ_c w[r·cols + c]·q[c]` for every `r < out.len()`; `w` is row-major, `q.len() >= cols`, every `q[c]` in
/// `0..=127` (CPUNK's quantiser range). NEON also needs `cols` a multiple of 16 and `w.len() == out.len() * cols` with
/// `out.len()` a multiple of 4 (`cpunk` pads).
#[cfg(all(pryon_neon, target_arch = "arm"))]
pub(crate) use armv7::dot_i8_rows;
#[cfg(not(all(pryon_neon, target_arch = "arm")))]
pub(crate) use portable::dot_i8_rows;

/// `num::cpunk_exp` over a slice in place, bit for bit (NEON `vfma`, which always flushes subnormals).
#[cfg(all(pryon_neon, target_arch = "arm"))]
pub(crate) use armv7::cpunk_exp_inplace;
#[cfg(not(all(pryon_neon, target_arch = "arm")))]
pub(crate) use portable::cpunk_exp_inplace;

/// `c[i] = num::fmaf(a[i], b[i], c[i])`, unflushed as bionic's fmaf, bit for bit (NaN payloads aside) whatever the FPSCR.
#[cfg(all(pryon_neon, target_arch = "arm"))]
pub(crate) use armv7::fmaf_inplace;
#[cfg(not(all(pryon_neon, target_arch = "arm")))]
pub(crate) use portable::fmaf_inplace;

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

/// `(lo, hi)` of `x` by NEON `vmin`/`vmax` from `(f32::MAX, −f32::MAX)`: CPUNK's row range.
#[cfg(all(pryon_neon, target_arch = "arm"))]
pub(crate) use armv7::cpunk_minmax;
#[cfg(not(all(pryon_neon, target_arch = "arm")))]
pub(crate) use portable::cpunk_minmax;

/// `q[c] = clamp(bits(vmla(1.5·2²³, inv, x[c])) + zb, 0, 127)` for `c < x.len()` (a multiple of 4, `q` at least as
/// long); returns `Σ q[c]`: CPUNK's NEON row quantiser.
#[cfg(all(pryon_neon, target_arch = "arm"))]
pub(crate) use armv7::cpunk_quant4;
#[cfg(not(all(pryon_neon, target_arch = "arm")))]
pub(crate) use portable::cpunk_quant4;

/// CPUNK's NEON dequant for `r < y.len()`, with `w = [zw, sums, sw]` and `s = [c, zf, sx]`:
/// `y[r] = vmla(vmul(bias[r], 1), vmul(sx, sw[r]), vsub(dot[r], vmla(vmul(c, zw[r]), zf, sums[r])))`, or the plain
/// product without a bias.
#[cfg(all(pryon_neon, target_arch = "arm"))]
pub(crate) use armv7::cpunk_deq;
#[cfg(not(all(pryon_neon, target_arch = "arm")))]
pub(crate) use portable::cpunk_deq;

/// `num::cpunk_sigmoid_of` / `num::cpunk_tanh_of` over a slice in place (VFP division under FZ, whatever the
/// caller's FPSCR).
#[cfg(all(pryon_neon, target_arch = "arm"))]
pub(crate) use armv7::{cpunk_sigmoid_of_inplace, cpunk_tanh_of_inplace};
#[cfg(not(all(pryon_neon, target_arch = "arm")))]
pub(crate) use portable::{cpunk_sigmoid_of_inplace, cpunk_tanh_of_inplace};

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
        let mut seen = [false; 8 * 24];
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
    fn dot_i8_rows_each_row() {
        let (rows, cols) = (8, 48);
        let w: Vec<i8> = (0..rows * cols).map(|i| ((i * 37) % 256) as u8 as i8).collect();
        let q: Vec<i8> = (0..cols).map(|i| ((i * 91) % 128) as i8).collect();
        let mut out = [7i32; 8];
        super::dot_i8_rows(&w, cols, &q, &mut out);
        for (r, &o) in out.iter().enumerate() {
            let want: i32 = (0..cols).map(|c| w[r * cols + c] as i32 * q[c] as i32).sum();
            assert_eq!(o, want, "row {r}");
        }
        let mut out = [0i32; 4];
        super::dot_i8_rows(&[-128; 4 * 1440], 1440, &[127; 1440], &mut out);
        assert_eq!(out, [-128 * 127 * 1440; 4]);
    }

    #[cfg(all(pryon_neon, target_arch = "arm"))]
    #[test]
    fn dot_i8_asm_equals_portable() {
        let mut next = xorshift(0x6a09_e667);
        let rows = 16;
        for cols in (16..=1440).step_by(16) {
            let mut w: Vec<i8> = (0..rows * cols).map(|_| next() as i8).collect();
            for c in 0..cols {
                let alt = if c % 2 == 0 { -128 } else { 127 };
                (w[c], w[cols + c], w[2 * cols + c], w[3 * cols + c]) = (-128, 127, alt, 0);
            }
            // q is longer than a row: the kernel must read only cols of it
            let random: Vec<i8> = (0..cols + 16).map(|_| (next() & 0x7F) as i8).collect();
            for q in [random, vec![0; cols], vec![127; cols]] {
                let (mut a, mut b) = (vec![0i32; rows], vec![1i32; rows]);
                super::dot_i8_rows(&w, cols, &q, &mut a);
                super::portable::dot_i8_rows(&w, cols, &q, &mut b);
                assert_eq!(a, b, "cols={cols} q[0]={}", q[0]);
            }
        }
        let mut out = [0i32; 4];
        super::dot_i8_rows(&[-128; 4 * 1440], 1440, &[127; 1440], &mut out);
        assert_eq!(out, [-128 * 127 * 1440; 4]);
    }

    #[cfg(all(pryon_neon, target_arch = "arm"))]
    #[test]
    fn exp_asm_equals_portable() {
        let c = f32::from_bits;
        let mut x = act_inputs();
        // clamp bounds and their neighbours, subnormal and tiny normal inputs (NEON flushes the former)
        for b in [0x42B1_7217u32, 0xC2AE_AC4F] {
            x.extend([c(b - 1), c(b), c(b + 1)]);
        }
        x.extend([c(1), c(0x8000_0001), c(0x007F_FFFF), c(0x807F_FFFF), f32::MIN_POSITIVE, -f32::MIN_POSITIVE]);
        x.extend((0..4000u32).map(|i| c(0x0080_0000 + i * 7919 | (i & 1) << 31)));
        x.extend([f32::INFINITY, f32::NEG_INFINITY, f32::NAN]);
        // every tail length, then the whole set
        for n in [1, 2, 3, 5, 6, 7, x.len()] {
            let (mut a, mut b) = (x[..n].to_vec(), x[..n].to_vec());
            super::cpunk_exp_inplace(&mut a);
            super::portable::cpunk_exp_inplace(&mut b);
            for ((&i, &p), &q) in x.iter().zip(&a).zip(&b) {
                assert_eq!(p.to_bits(), q.to_bits(), "exp({i:e} = {:#010x})", i.to_bits());
            }
        }
    }

    /// Runs `f` with `bits` set in FPSCR, restoring it after.
    #[cfg(all(pryon_neon, target_arch = "arm"))]
    fn with_fpscr<T>(bits: u32, f: impl FnOnce() -> T) -> T {
        let old: u32;
        // SAFETY: only FPSCR changes, and it is restored below.
        unsafe {
            core::arch::asm!("vmrs {o}, fpscr", "orr {t}, {o}, {b}", "vmsr fpscr, {t}", o = out(reg) old, t = out(reg) _,
                b = in(reg) bits, options(nostack));
        }
        let r = f();
        unsafe { core::arch::asm!("vmsr fpscr, {}", in(reg) old, options(nostack)) };
        r
    }

    #[cfg(all(pryon_neon, target_arch = "arm"))]
    fn fpscr() -> u32 {
        let v: u32;
        // SAFETY: reads FPSCR only.
        unsafe { core::arch::asm!("vmrs {}, fpscr", out(reg) v, options(nomem, nostack)) };
        v
    }

    #[cfg(all(pryon_neon, target_arch = "arm"))]
    #[test]
    fn fmaf_asm_equals_portable() {
        let mut next = xorshift(0x3c6e_f372);
        let special = [0.0f32, -0.0, 1.0, -1.0, f32::MIN_POSITIVE, 1e-45, -3e-45, 1e-40, 3.4e38, -3.4e38, 0.5, 1.0000001];
        let mut abc = Vec::new();
        for &x in &special {
            for &y in &special {
                abc.extend(special.map(|z| (x, y, z)));
            }
        }
        let f = |e: u32, m: u32| f32::from_bits((m & 0x807F_FFFF) | (e.min(254) << 23));
        for i in 0..100_000u32 {
            // products from subnormal to overflow, and c cancelling most of a·b
            let ea = next() % 255;
            let eb = (254 - ea + next() % 200).saturating_sub(100);
            let (x, y) = (f(ea, next()), f(eb, next()));
            let z = match i % 3 {
                0 => f32::from_bits(next()),
                1 => -(x * y) * (1.0 + (next() % 9) as f32 * f32::EPSILON),
                _ => f((ea + eb).saturating_sub(127 + next() % 60), next()),
            };
            abc.push((x, y, z));
        }
        // the cell's own inputs: gates in (0, 1) and (−1, 1), tiny f·c0
        for _ in 0..10_000 {
            let u = |r: u32| (r >> 8) as f32 / (1 << 24) as f32;
            abc.push((u(next()), u(next()) * 2.0 - 1.0, f32::from_bits(next() & 0x80FF_FFFF)));
        }
        let a: Vec<f32> = abc.iter().map(|t| t.0).collect();
        let b: Vec<f32> = abc.iter().map(|t| t.1).collect();
        let c: Vec<f32> = abc.iter().map(|t| t.2).collect();
        let mut want = c.clone();
        super::portable::fmaf_inplace(&a, &b, &mut want);
        // the caller's FPSCR as found, then with DN, FZ and round-towards-zero set
        for bits in [0, 0x03C0_0000] {
            let mut got = c.clone();
            let after = with_fpscr(bits, || {
                super::fmaf_inplace(&a, &b, &mut got);
                fpscr()
            });
            assert_eq!(after & bits, bits, "FPSCR not restored");
            for (i, (&g, &w)) in got.iter().zip(&want).enumerate() {
                let same = g.to_bits() == w.to_bits() || (g.is_nan() && w.is_nan());
                assert!(same, "FPSCR |= {bits:#x}: fmaf({:e}, {:e}, {:e}) = {g:e}, want {w:e}", a[i], b[i], c[i]);
            }
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

    /// Floats for the CPUNK kernels: ±0, subnormals, FLT_MIN neighbours, ±inf, NaNs, extremes and random bits.
    #[cfg(all(pryon_neon, target_arch = "arm"))]
    fn cpunk_inputs(next: &mut impl FnMut() -> u32, n: usize) -> Vec<f32> {
        let c = f32::from_bits;
        let special = [0.0, -0.0, c(1), c(0x8000_0001), c(0x007F_FFFF), c(0x807F_FFFF), f32::MIN_POSITIVE,
            -f32::MIN_POSITIVE, c(0x0080_0001), 1.0, -1.0, 0.5, f32::MAX, f32::MIN, f32::INFINITY, f32::NEG_INFINITY,
            f32::NAN, c(0xFFC0_0001), c(0x7F80_0001), 4194303.5, -4194304.0, 8388608.0, 127.49, 127.5, -0.5];
        (0..n)
            .map(|i| match next() % 4 {
                0 => special[(next() as usize) % special.len()],
                1 => f32::from_bits(next()),
                2 => (next() as i32 as f32) * 1e-7,
                _ => ((next() % 400) as f32 - 140.0) * 0.37 + i as f32 * 1e-3,
            })
            .collect()
    }

    #[cfg(all(pryon_neon, target_arch = "arm"))]
    fn same(a: f32, b: f32) -> bool {
        a.to_bits() == b.to_bits()
    }

    #[cfg(all(pryon_neon, target_arch = "arm"))]
    #[test]
    fn cpunk_minmax_asm_equals_portable() {
        let mut next = xorshift(0x510e_527f);
        for n in (0..=19).chain([64, 1440, 4093]) {
            for round in 0..200 {
                let mut x = cpunk_inputs(&mut next, n);
                // rows without NaN too, or the default NaN would hide the ±0 and subnormal rules
                if round % 2 == 0 {
                    x.retain(|v| !v.is_nan());
                }
                let (a, b) = (super::cpunk_minmax(&x), super::portable::cpunk_minmax(&x));
                assert!(same(a.0, b.0) && same(a.1, b.1), "n={n} {a:?} {b:?} {x:?}");
            }
        }
    }

    #[cfg(all(pryon_neon, target_arch = "arm"))]
    #[test]
    fn cpunk_quant4_asm_equals_portable() {
        let mut next = xorshift(0x9b05_688c);
        let invs = [1.0f32, 0.0, -0.0, 3.7, 1e-30, 1e30, f32::from_bits(1), f32::INFINITY, f32::NAN, 127.0 / 3.0];
        for n in (0..=40).step_by(4).chain([1440, 4096]) {
            for round in 0..100 {
                let x = cpunk_inputs(&mut next, n);
                let inv = if round < invs.len() { invs[round] } else { f32::from_bits(next()) };
                let zb = match round % 3 {
                    0 => 0xB4C0_0000u32 as i32,
                    1 => (0xB4C0_0000u32 as i32).wrapping_add((next() % 256) as i32 - 128),
                    _ => next() as i32,
                };
                let (mut qa, mut qb) = (vec![1i8; n + 4], vec![1i8; n + 4]);
                let sa = super::cpunk_quant4(&x, inv, zb, &mut qa);
                let sb = super::portable::cpunk_quant4(&x, inv, zb, &mut qb);
                assert_eq!((sa, &qa), (sb, &qb), "n={n} inv={inv:e} zb={zb:#x}");
            }
        }
        // products just off k + 0.5: the rounded product ties where a fused one would not
        let x: Vec<f32> = (0..4096).map(|k| (2 * k + 1) as f32 / 6.0).collect();
        for inv in [3.0f32, 1.0 / 3.0, 3.0000002] {
            let zb = 0xB4C0_0000u32 as i32 - 1000;
            let (mut qa, mut qb) = (vec![0i8; 4096], vec![0i8; 4096]);
            let sa = super::cpunk_quant4(&x, inv, zb, &mut qa);
            assert_eq!((sa, &qa), (super::portable::cpunk_quant4(&x, inv, zb, &mut qb), &qb), "inv={inv:e}");
        }
    }

    #[cfg(all(pryon_neon, target_arch = "arm"))]
    #[test]
    fn cpunk_deq_asm_equals_portable() {
        let mut next = xorshift(0x1f83_d9ab);
        for n in (0..=19).chain([64, 1024]) {
            for round in 0..150 {
                let f = cpunk_inputs(&mut next, 5 * n + 3);
                let dot: Vec<i32> = (0..n)
                    .map(|_| match next() % 3 {
                        0 => next() as i32,
                        1 => (next() % 2_000_001) as i32 - 1_000_000,
                        _ => [0, 1, -1, i32::MAX, i32::MIN, 16_777_217][(next() % 6) as usize],
                    })
                    .collect();
                let w = [&f[..n], &f[n..2 * n], &f[2 * n..3 * n]];
                let s = [f[5 * n], f[5 * n + 1], f[5 * n + 2]];
                let s = if round % 2 == 0 { s.map(|v| if v.is_finite() { v } else { 0.25 }) } else { s };
                for bias in [None, Some(&f[3 * n..4 * n])] {
                    let (mut ya, mut yb) = (f[4 * n..5 * n].to_vec(), vec![0.0f32; n]);
                    super::cpunk_deq(&dot, w, bias, s, &mut ya);
                    super::portable::cpunk_deq(&dot, w, bias, s, &mut yb);
                    for r in 0..n {
                        let b = bias.is_some();
                        assert!(same(ya[r], yb[r]), "n={n} r={r} bias={b} s={s:?}: {} vs {}", ya[r], yb[r]);
                    }
                }
            }
        }
    }

    #[cfg(all(pryon_neon, target_arch = "arm"))]
    #[test]
    fn cpunk_act_of_asm_equals_portable() {
        let mut next = xorshift(0x5be0_cd19);
        let c = f32::from_bits;
        let mut e = cpunk_inputs(&mut next, 20_000);
        // exp outputs as the LSTM feeds them, and quotients at FLT_MIN (FZ judges the exact quotient)
        let mut x: Vec<f32> = (0..20_000).map(|_| ((next() % 36_000) as f32 - 18_000.0) * 0.005).collect();
        super::portable::cpunk_exp_inplace(&mut x);
        e.extend(x);
        e.extend([c(0x0080_0000), c(0x0100_0000), c(0x0080_0001), c(0x00FF_FFFF), -1.0, -2.0, c(0xBF80_0001)]);
        e.extend((0..2000u32).map(|i| c(0x0080_0000 + i * 4099)));
        // the caller's FPSCR as found, then with DN, FZ and round-towards-zero set
        for bits in [0, 0x03C0_0000] {
            for n in [1, 2, 3, 4, 5, 6, 7, 8, e.len()] {
                for tanh in [false, true] {
                    let (mut a, mut b) = (e[..n].to_vec(), e[..n].to_vec());
                    let after = with_fpscr(bits, || {
                        if tanh {
                            super::cpunk_tanh_of_inplace(&mut a)
                        } else {
                            super::cpunk_sigmoid_of_inplace(&mut a)
                        }
                        fpscr()
                    });
                    assert_eq!(after & 0x03C0_0000, bits, "FPSCR not restored");
                    if tanh {
                        super::portable::cpunk_tanh_of_inplace(&mut b)
                    } else {
                        super::portable::cpunk_sigmoid_of_inplace(&mut b)
                    }
                    for i in 0..n {
                        assert!(same(a[i], b[i]), "fpscr|={bits:#x} tanh={tanh} e={:e} ({:#010x}): {:e} vs {:e}",
                            e[i], e[i].to_bits(), a[i], b[i]);
                    }
                }
            }
        }
    }
}
