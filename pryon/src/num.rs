//! Bit-exact scalar ports of libpryon's NEON numerics (quantizer, sums, exp, reciprocal, nonlinearities).

pub(crate) fn quantize(x: &[f32], inv: f32, off: f32, k: &mut [u8]) {
    let n16 = x.len() & !15;
    for (i, (&v, o)) in x.iter().zip(k.iter_mut()).enumerate() {
        let t = inv * (v - off);
        *o = if i < n16 {
            let h = t + 0.5f32.copysign(t);
            if h > 0.0 { h.trunc().min(255.0) as u8 } else { 0 }
        } else if t > 255.0 {
            255
        } else if t >= 0.0 {
            t.round_ties_even() as u8
        } else {
            0
        };
    }
}

pub(crate) fn neon_sum(x: &[f32]) -> f32 {
    let n4 = x.len() & !3;
    let mut lanes = [0f32; 4];
    for c in x[..n4].chunks_exact(4) {
        for (l, &v) in lanes.iter_mut().zip(c) {
            *l += v;
        }
    }
    let mut t = 0f32;
    for &v in &x[n4..] {
        t += v;
    }
    for l in lanes {
        t += l;
    }
    t
}

// Port of Julien Pommier's neon_mathfun exp_ps (zlib); altered version, see THIRD_PARTY_NOTICES.md.
pub(crate) fn exp_ps(x: f32) -> f32 {
    let c = f32::from_bits;
    let x = x.min(c(0x42B0C0A5)).max(c(0xC2B0C0A5));
    let fx = x * c(0x3FB8AA3B) + 0.5;
    let tmp = fx.trunc();
    let fx = tmp - if tmp > fx { 1.0 } else { 0.0 };
    let x = x + fx * c(0xBF318000);
    let x = x + fx * c(0x395E8083);
    let mut y = c(0x3AB743CE) + x * c(0x39506967);
    for k in [0x3C088908, 0x3D2AA9C1, 0x3E2AAAAA, 0x3F000000] {
        y = c(k) + x * y;
    }
    y = x + (x * x) * y;
    y += 1.0;
    let pow2n = f32::from_bits(((fx as i32).wrapping_shl(23).wrapping_add(0x3F800000)) as u32);
    y * pow2n
}

pub(crate) fn recpe(d: f32) -> f32 {
    let bits = d.to_bits();
    let a = (256 | ((bits >> 15) & 0xFF)) * 2 + 1;
    let r = (((1u32 << 19) / a) + 1) >> 1;
    let e = 253 - ((bits >> 23) & 0xFF) as i32;
    if e <= 0 {
        return 0.0;
    }
    f32::from_bits(((e as u32) << 23) | ((r & 0xFF) << 15))
}

pub(crate) fn recps(a: f32, b: f32) -> f32 {
    2.0 - a * b
}

pub(crate) fn recip(d: f32) -> f32 {
    let r = recpe(d);
    let r = r * recps(d, r);
    recps(d, r) * r
}

pub(crate) fn sigmoid(z: f32) -> f32 {
    recip(exp_ps(0.0 - z) + 1.0)
}

pub(crate) fn tanh(z: f32) -> f32 {
    let m = 0.0 - z;
    let r = recip(exp_ps(m + m) + 1.0);
    (r + r) + -1.0
}

/// Flush to zero: a subnormal becomes a zero of its sign (VFP with FPSCR.FZ, and always NEON).
pub(crate) fn fz(x: f32) -> f32 {
    if x.is_subnormal() {
        0.0f32.copysign(x)
    } else {
        x
    }
}

/// A NEON result: flushed to zero, any NaN the default NaN.
fn dn(x: f32) -> f32 {
    if x.is_nan() {
        f32::from_bits(0x7FC0_0000)
    } else {
        fz(x)
    }
}

pub(crate) fn vadd(a: f32, b: f32) -> f32 {
    dn(fz(a) + fz(b))
}

pub(crate) fn vsub(a: f32, b: f32) -> f32 {
    dn(fz(a) - fz(b))
}

pub(crate) fn vmul(a: f32, b: f32) -> f32 {
    dn(mul_fz(fz(a), fz(b)))
}

/// A product under FZ: ARM flushes on the unrounded value, so a product just under FLT_MIN is zero even when it
/// rounds up to FLT_MIN.
fn mul_fz(a: f32, b: f32) -> f32 {
    let r = a * b;
    if r.abs() < f32::MIN_POSITIVE {
        0.0f32.copysign(r)
    } else if r.abs() == f32::MIN_POSITIVE {
        fma(a, b, -0.0, true)
    } else {
        r
    }
}

/// NEON `vmla`: `a + b·c`, the product rounded first.
pub(crate) fn vmla(a: f32, b: f32, c: f32) -> f32 {
    vadd(a, vmul(b, c))
}

/// NEON `vfma`: `a + b·c` rounded once.
pub(crate) fn vfma(a: f32, b: f32, c: f32) -> f32 {
    dn(fma(fz(b), fz(c), fz(a), true))
}

pub(crate) fn vmin(a: f32, b: f32) -> f32 {
    let (a, b) = (fz(a), fz(b));
    if a.is_nan() || b.is_nan() {
        f32::from_bits(0x7FC0_0000)
    } else if a < b || (a == b && a.is_sign_negative()) {
        a
    } else {
        b
    }
}

pub(crate) fn vmax(a: f32, b: f32) -> f32 {
    let (a, b) = (fz(a), fz(b));
    if a.is_nan() || b.is_nan() {
        f32::from_bits(0x7FC0_0000)
    } else if a > b || (a == b && a.is_sign_positive()) {
        a
    } else {
        b
    }
}

pub(crate) fn zsub(a: f32, b: f32) -> f32 {
    fz(fz(a) - fz(b))
}

pub(crate) fn zmul(a: f32, b: f32) -> f32 {
    mul_fz(fz(a), fz(b))
}

/// VFP division under FPSCR.FZ.
pub(crate) fn zdiv(a: f32, b: f32) -> f32 {
    let (a, b) = (fz(a), fz(b));
    let r = a / b;
    // FZ judges the exact quotient: at FLT_MIN, compare |a| with FLT_MIN·|b| in integers
    let tiny = r.abs() < f32::MIN_POSITIVE
        || r.abs() == f32::MIN_POSITIVE && {
            let ((ma, ea), (mb, eb)) = (parts(a), parts(b));
            let (ma, mb, d) = (u64::from(ma), u64::from(mb), ea - eb + 126);
            if d >= 0 { d < 40 && ma << d < mb } else { d <= -40 || ma < mb << -d }
        };
    if tiny { 0.0f32.copysign(r) } else { r }
}

/// VFP square root under FPSCR.FZ (correctly rounded; its result is never subnormal).
pub(crate) fn zsqrt(a: f32) -> f32 {
    fz(a).sqrt()
}

/// Round half to even (`nearbyintf` in the default mode), without libm.
pub(crate) fn rint(x: f32) -> f32 {
    const BIG: f32 = 8388608.0;
    if !(x.abs() < BIG) {
        return x;
    }
    let m = BIG.copysign(x);
    ((x + m) - m).copysign(x)
}

/// Mantissa and exponent of a finite nonzero f32: `x = ±m·2^e`.
fn parts(x: f32) -> (u32, i32) {
    let u = x.to_bits();
    let e = (u >> 23 & 0xFF) as i32;
    let m = u & 0x7F_FFFF;
    if e == 0 { (m, -149) } else { (m | 0x80_0000, e - 150) }
}

/// `a·b + c` rounded once (bionic `fmaf`), in integer arithmetic: no FMA instruction, no f64.
pub(crate) fn fmaf(a: f32, b: f32, c: f32) -> f32 {
    fma(a, b, c, false)
}

/// `fmaf`, with `ftz` flushing a result that is tiny before rounding (ARM FPSCR.FZ, NEON).
fn fma(a: f32, b: f32, c: f32, ftz: bool) -> f32 {
    if !a.is_finite() || !b.is_finite() || c.is_nan() || a == 0.0 || b == 0.0 {
        return a * b + c;
    }
    if c.is_infinite() {
        return c;
    }
    let ((ma, ea), (mb, eb)) = (parts(a), parts(b));
    let p = (u64::from(ma) * u64::from(mb)) as i64;
    let p = if (a < 0.0) != (b < 0.0) { -p } else { p };
    let ep = ea + eb;
    if c == 0.0 {
        return round_scaled(p.unsigned_abs(), p < 0, ep, ftz);
    }
    let (mc, ec) = parts(c);
    let mc = if c < 0.0 { -i64::from(mc) } else { i64::from(mc) };
    let d = ep - ec;
    // the product has 48 bits and c 24: these gaps keep the aligned sum inside an i64
    if (-38..=14).contains(&d) {
        let (sum, e) = if d >= 0 { ((p << d) + mc, ec) } else { (p + (mc << -d), ep) };
        return round_scaled(sum.unsigned_abs(), sum < 0, e, ftz);
    }
    let (p, mc) = (i128::from(p), i128::from(mc));
    // beyond these gaps the smaller term only decides rounding, as a one-unit sticky of its sign
    let (sum, e) = match d {
        0..=72 => ((p << d) + mc, ec),
        73.. => ((p << 72) + mc.signum(), ep - 72),
        -100..=-1 => (p + (mc << -d), ep),
        _ => (p.signum() + (mc << 100), ec - 100),
    };
    let m = sum.unsigned_abs();
    // keep 62 bits with a sticky low bit: still well below the 24 bits that survive rounding
    let sh = (128 - m.leading_zeros()).saturating_sub(62);
    let m64 = (m >> sh) as u64 | u64::from(m & ((1u128 << sh) - 1) != 0);
    round_scaled(m64, sum < 0, e + sh as i32, ftz)
}

/// `±m·2^e` rounded to the nearest f32, ties to even; with `ftz`, zero when below FLT_MIN before rounding.
fn round_scaled(m: u64, neg: bool, e: i32, ftz: bool) -> f32 {
    if m == 0 {
        return 0.0;
    }
    let top = 63 - m.leading_zeros() as i32 + e;
    if ftz && top < -126 {
        return if neg { -0.0 } else { 0.0 };
    }
    let lsb = (top - 23).max(-149);
    let q = if lsb <= e {
        m << (e - lsb)
    } else {
        let sh = (lsb - e) as u32;
        let (q, rem) = if sh >= 64 { (0, m) } else { (m >> sh, m & ((1u64 << sh) - 1)) };
        let half = if sh > 64 { u64::MAX } else { 1u64 << (sh - 1) };
        if rem > half || (rem == half && q & 1 == 1) { q + 1 } else { q }
    };
    let bits = if top > 127 { 0x7F80_0000 } else { ((((lsb + 149) as u32) << 23) + q as u32).min(0x7F80_0000) };
    f32::from_bits(bits | u32::from(neg) << 31)
}

/// CPUNK's NEON exp (SigmoidImpl/TanhImpl<CortexA15>, 0x1249cb0/0x124d500): Cody-Waite, degree-5 fused polynomial.
pub(crate) fn cpunk_exp(x: f32) -> f32 {
    let c = f32::from_bits;
    let x = vmax(vmin(x, c(0x42B1_7217)), c(0xC2AE_AC4F));
    let t = vmla(12582912.0, x, c(0x3FB8_AA3B));
    let n = (t.to_bits() as i32).wrapping_add(0xB4C0_0000u32 as i32);
    let nf = n as f32;
    let r = vfma(vfma(x, nf, c(0xBF31_7200)), nf, c(0xB5BF_BE8E));
    let mut p = vfma(c(0x3D2B_A65B), r, c(0x3C08_0161));
    for k in [0x3E2A_ACD3, 0x3EFF_FEC6, 0x3F7F_FFFC, 0x3F80_0000] {
        p = vfma(c(k), p, r);
    }
    f32::from_bits(((n as u32) << 23).wrapping_add(p.to_bits()))
}

/// CPUNK ReduceMaxImpl<CortexA15> (0x124ac90): NEON vmax from -FLT_MAX (lane order is immaterial: vmax is exact).
pub(crate) fn cpunk_max(x: &[f32]) -> f32 {
    x.iter().fold(f32::MIN, |m, &v| vmax(m, v))
}

/// CPUNK SoftmaxImpl<CortexA15> on one row (0x124af60; 0x124b8d0 for four rows sums in the same order).
pub(crate) fn cpunk_softmax(x: &[f32], y: &mut [f32]) {
    let m = cpunk_max(x);
    for (o, &v) in y.iter_mut().zip(x) {
        *o = cpunk_exp(vsub(v, m));
    }
    let n4 = y.len() & !3;
    let mut l = [0f32; 4];
    if n4 < y.len() {
        for (k, &e) in y[n4..].iter().enumerate() {
            l[k] = e;
        }
        l = l.map(|v| vadd(v, 0.0));
    }
    for c in y[..n4].chunks_exact(4).rev() {
        for (a, &e) in l.iter_mut().zip(c) {
            *a = vadd(*a, e);
        }
    }
    let inv = zdiv(1.0, vadd(vadd(l[0], l[2]), vadd(l[1], l[3])));
    // a one-element tail is scaled by a VFP multiply, everything else by NEON
    let vfp = y.len() - n4 == 1;
    for (i, o) in y.iter_mut().enumerate() {
        *o = if vfp && i == n4 { zmul(inv, *o) } else { vmul(inv, *o) };
    }
}

/// libpryon's CPUNK sigmoid of `x` from `e = exp(x)`: `e / (e + 1)`.
pub(crate) fn cpunk_sigmoid_of(e: f32) -> f32 {
    zdiv(e, vadd(e, 1.0))
}

/// libpryon's CPUNK tanh of `x` from `e = exp(2x)`: `(e − 1) / (e + 1)`.
pub(crate) fn cpunk_tanh_of(e: f32) -> f32 {
    zdiv(vadd(e, -1.0), vadd(e, 1.0))
}

#[cfg(test)]
pub(crate) fn cpunk_sigmoid(x: f32) -> f32 {
    cpunk_sigmoid_of(cpunk_exp(x))
}

#[cfg(test)]
pub(crate) fn cpunk_tanh(x: f32) -> f32 {
    cpunk_tanh_of(cpunk_exp(vadd(x, x)))
}

/// OpenBLAS ARMv7 sdot_vfp order: two accumulators over blocks of four, remainder into the first.
pub(crate) fn sdot_vfp(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let (mut s0, mut s1) = (0f32, 0f32);
    let mut i = 0;
    while i + 4 <= n {
        s0 += a[i] * b[i];
        s1 += a[i + 1] * b[i + 1];
        s0 += a[i + 2] * b[i + 2];
        s1 += a[i + 3] * b[i + 3];
        i += 4;
    }
    while i < n {
        s0 += a[i] * b[i];
        i += 1;
    }
    s0 + s1
}

/// OpenBLAS ARMv7 sgemv_t / sgemm 4x4 kernel order: one accumulator, inputs in order.
pub(crate) fn dot_seq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).fold(0.0, |acc, (&x, &y)| acc + x * y)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    #[test]
    fn exp_ps_bits() {
        let xs = [-100.0, -88.5, -20.0, -1.0, -1e-8, 0.0, 1e-8, 0.5, 1.0, 20.0, 88.5, 100.0];
        let want = [
            0x00000000u32, 0x00000000, 0x310DA433, 0x3EBC5AB2, 0x3F800000, 0x3F800000, 0x3F800000, 0x3FD3094C,
            0x402DF854, 0x4DE75844, 0x7F3504A4, 0x7F3504A4,
        ];
        let got: Vec<u32> = xs.iter().map(|&x| exp_ps(x).to_bits()).collect();
        assert_eq!(got, want);
    }

    const ZS: [f32; 7] = [-30.0, -5.0, -0.1, 0.0, 0.1, 5.0, 30.0];

    #[test]
    fn sigmoid_bits() {
        let want = [0x29D2B707u32, 0x3BDB4FB4, 0x3EF335ED, 0x3F000000, 0x3F066509, 0x3F7E4960, 0x3F800000];
        let got: Vec<u32> = ZS.iter().map(|&z| sigmoid(z).to_bits()).collect();
        assert_eq!(got, want);
    }

    #[test]
    fn tanh_bits() {
        let want = [0xBF800000u32, 0xBF7FFA0D, 0xBDCC1EB8, 0x00000000, 0x3DCC1EB0, 0x3F7FFA0A, 0x3F800000];
        let got: Vec<u32> = ZS.iter().map(|&z| tanh(z).to_bits()).collect();
        assert_eq!(got, want);
    }

    #[test]
    fn recip_bits() {
        let xs = [1.0, 1.5, 2.000001, 3.7, 1e6];
        let want = [0x3F800000u32, 0x3F2AAAAB, 0x3EFFFFF8, 0x3E8A60DC, 0x358637BD];
        let got: Vec<u32> = xs.iter().map(|&x| recip(x).to_bits()).collect();
        assert_eq!(got, want);
    }

    #[test]
    fn quantize_matches_ref() {
        let x: Vec<f32> = (0..37).map(|i| -1.3f32 + i as f32 * 0.1).collect();
        let (inv, off) = (f32::from_bits(0x420F5555), f32::from_bits(0xBFA49D2C));
        let mut k = [0u8; 37];
        quantize(&x, inv, off, &mut k);
        let want: [u8; 37] = [
            0, 3, 7, 10, 14, 17, 21, 25, 28, 32, 35, 39, 43, 46, 50, 53, 57, 60, 64, 68, 71, 75, 78, 82, 86, 89, 93,
            96, 100, 103, 107, 111, 114, 118, 121, 125, 128,
        ];
        assert_eq!(k, want);
    }

    #[test]
    fn quantize_half_cases() {
        // Head rounds half away from zero, tail rounds half to even.
        let x: Vec<f32> = [0.5, 1.5, 2.5, -0.5, 255.5, 300.0, -3.0, 0.49].repeat(3);
        let mut k = [0u8; 24];
        quantize(&x, 1.0, 0.0, &mut k);
        let want = [1u8, 2, 3, 0, 255, 255, 0, 0, 1, 2, 3, 0, 255, 255, 0, 0, 0, 2, 2, 0, 255, 255, 0, 0];
        assert_eq!(k, want);
    }

    #[test]
    fn dot_seq_order() {
        // A two-accumulator order (sdot_vfp) gives 2 here.
        assert_eq!(dot_seq(&[1e8, 1.0, -1e8, 1.0], &[1.0; 4]), 1.0);
        assert_eq!(dot_seq(&[1.5, -2.0, 0.25], &[2.0, 0.5, 4.0]), 3.0);
    }

    fn xorshift(mut s: u32) -> impl FnMut() -> u32 {
        move || {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            s
        }
    }

    #[test]
    fn rint_ties_even() {
        let xs = [2.5f32, 3.5, -2.5, -0.4, 0.5, 8388607.5, -8388607.5, 1e10, -0.0];
        let want = [2.0f32, 4.0, -2.0, -0.0, 0.0, 8388608.0, -8388608.0, 1e10, -0.0];
        assert_eq!(bits(&xs.map(rint)), bits(&want));
    }

    #[test]
    fn fmaf_is_fused() {
        let mut next = xorshift(0x2545_f491);
        let special = [0.0f32, -0.0, 1.0, -1.0, f32::MIN_POSITIVE, 1e-45, -3e-45, 3.4e38, -3.4e38, 0.5, 1.0000001];
        let mut cases = Vec::new();
        for &a in &special {
            for &b in &special {
                for &c in &special {
                    cases.push((a, b, c));
                }
            }
        }
        for i in 0..400_000u32 {
            // exponents spread so products land everywhere from subnormal to overflow
            let f = |e: u32, m: u32| f32::from_bits((m & 0x807F_FFFF) | (e.min(254) << 23));
            let ea = next() % 255;
            let eb = (254 - ea + next() % 200).saturating_sub(100);
            let (a, b) = (f(ea, next()), f(eb, next()));
            let (r, m) = (next(), next());
            let c = match i % 3 {
                0 => f32::from_bits(r),
                // c cancels most of a·b
                1 => -(a * b) * (1.0 + (r % 9) as f32 * f32::EPSILON),
                _ => f((ea + eb).saturating_sub(127 + r % 60), m),
            };
            cases.push((a, b, c));
        }
        // full mantissas at every exponent gap around the i64/i128 switch, both signs of c
        for gap in -110..=80 {
            for (ma, mb, mc) in [(0x7F_FFFF, 0x7F_FFFF, 0x7F_FFFF), (0x7F_FF7E, 0x0A_3B11, 0x7F_FFFC), (0, 0, 1)] {
                let (a, b) = (f32::from_bits(0x3F00_0000 | ma), f32::from_bits(0x3F00_0000 | mb));
                let c = f32::from_bits((((126 - gap).clamp(1, 254) as u32) << 23) | mc);
                cases.extend([(a, b, c), (a, b, -c), (-a, b, c)]);
            }
        }
        for (a, b, c) in cases {
            let (got, want) = (fmaf(a, b, c), a.mul_add(b, c));
            let same = got.to_bits() == want.to_bits() || (got.is_nan() && want.is_nan());
            assert!(same, "fmaf({a:e}, {b:e}, {c:e}) = {got:e}, fused {want:e}");
        }
    }

    #[test]
    fn neon_flushes_subnormals() {
        let tiny = f32::from_bits(1);
        assert_eq!(vadd(tiny, 0.0).to_bits(), 0);
        assert_eq!(vmul(-1e-20, 1e-20).to_bits(), 0x8000_0000);
        assert_eq!(vmul(f32::NAN, 1.0).to_bits(), 0x7FC0_0000);
        assert_eq!(vfma(1.0, tiny, 1e30).to_bits(), 1.0f32.to_bits());
        assert_eq!(zdiv(f32::MIN_POSITIVE, 2.0).to_bits(), 0);
    }

    #[test]
    fn fz_tininess_before_rounding() {
        // exact results in [FLT_MIN − 2^-150, FLT_MIN) round up to FLT_MIN, but ARM's FZ tests the unrounded value
        let (min, just_below_one) = (f32::MIN_POSITIVE, 1.0 - f32::EPSILON / 2.0);
        let wide = f32::from_bits(0x00FF_FFFF);
        assert_eq!(min * just_below_one, min);
        assert_eq!(vmul(wide, 0.5).to_bits(), 0);
        assert_eq!(vmul(min, just_below_one).to_bits(), 0);
        assert_eq!(vmul(-min, just_below_one).to_bits(), 0x8000_0000);
        assert_eq!(zmul(wide, 0.5).to_bits(), 0);
        assert_eq!(zdiv(wide, 2.0).to_bits(), 0);
        assert_eq!(zdiv(-wide, 2.0).to_bits(), 0x8000_0000);
        assert_eq!(vfma(0.0, min, just_below_one).to_bits(), 0);
        assert_eq!(vfma(-0.0, wide, -0.5).to_bits(), 0x8000_0000);
        // FLT_MIN itself and anything above stays
        assert_eq!(vmul(min, 1.0), min);
        assert_eq!(zdiv(f32::from_bits(0x0100_0000), 2.0), min);
        assert_eq!(vmul(f32::from_bits(0x0080_0001), 1.0), f32::from_bits(0x0080_0001));
        assert_eq!(vfma(min, 1.0, 0.0), min);
    }

    /// CPUNK's exp (SigmoidImpl/TanhImpl<CortexA15>) written with the platform's fused multiply-add.
    fn ref_exp(x: f32) -> f32 {
        let c = f32::from_bits;
        let x = x.min(c(0x42B1_7217)).max(c(0xC2AE_AC4F));
        let t = 12582912.0f32 + x * c(0x3FB8_AA3B);
        let n = (t.to_bits() as i32).wrapping_add(0xB4C0_0000u32 as i32);
        let nf = n as f32;
        let r = nf.mul_add(c(0xBF31_7200), x);
        let r = nf.mul_add(c(0xB5BF_BE8E), r);
        let mut p = r.mul_add(c(0x3C08_0161), c(0x3D2B_A65B));
        for k in [0x3E2A_ACD3, 0x3EFF_FEC6, 0x3F7F_FFFC, 0x3F80_0000] {
            p = p.mul_add(r, c(k));
        }
        f32::from_bits(((n as u32) << 23).wrapping_add(p.to_bits()))
    }

    #[test]
    fn activation_bits() {
        let xs = [-100.0f32, -87.0, -20.0, -5.0, -1.0, -0.1, -1e-6, 0.0, 1e-6, 0.1, 0.5, 1.0, 5.0, 20.0, 88.0, 100.0];
        let sig = [
            0x0080_0026u32, 0x00B3_3687, 0x310D_A432, 0x3BDB_4FB2, 0x3E89_B2B1, 0x3EF3_35ED, 0x3EFF_FFF7, 0x3F00_0000,
            0x3F00_0004, 0x3F06_6509, 0x3F1F_597E, 0x3F3B_26A8, 0x3F7E_4961, 0x3F80_0000, 0x3F80_0000, 0x3F80_0000,
        ];
        let tanh = [
            0xBF80_0000u32, 0xBF80_0000, 0xBF80_0000, 0xBF7F_FA0C, 0xBF42_F7D5, 0xBDCC_1EC4, 0xB588_0009, 0x0000_0000,
            0x3587_FFF8, 0x3DCC_1EB6, 0x3EEC_9A9F, 0x3F42_F7D6, 0x3F7F_FA0D, 0x3F80_0000, 0x3F80_0000, 0x3F80_0000,
        ];
        assert_eq!(bits(&xs.map(cpunk_sigmoid)), sig);
        assert_eq!(bits(&xs.map(cpunk_tanh)), tanh);
        let mut next = xorshift(7);
        for _ in 0..200_000 {
            let x = f32::from_bits(next() % 0x42C8_0000 | (next() & 0x8000_0000));
            let e = ref_exp(x);
            assert_eq!(cpunk_sigmoid(x).to_bits(), (e / (e + 1.0)).to_bits(), "sigmoid({x:e})");
            let e = ref_exp(x + x);
            assert_eq!(cpunk_tanh(x).to_bits(), ((e + -1.0) / (e + 1.0)).to_bits(), "tanh({x:e})");
        }
    }

    #[test]
    fn neon_sum_order() {
        let x: Vec<f32> = (0..23).map(|i| 0.1f32 * i as f32).collect();
        assert_eq!(bits(&[neon_sum(&x)]), [0x41CA6666]);
    }
}
