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

    #[test]
    fn neon_sum_order() {
        let x: Vec<f32> = (0..23).map(|i| 0.1f32 * i as f32).collect();
        assert_eq!(bits(&[neon_sum(&x)]), [0x41CA6666]);
    }
}
