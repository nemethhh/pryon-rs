use core::arch::asm;

pub(crate) fn dot_rows(w: &[u8], stride: usize, k: &[u8], out: &mut [i32]) {
    assert!(stride > 0 && stride % 8 == 0 && k.len() >= stride && out.len() % 4 == 0);
    assert!(w.len() == out.len() * stride);
    // Weights are stored +128: Σ (w + 128) k − 128 Σ k, with u8 x u8 products that fit u16.
    let corr = 128 * k[..stride].iter().map(|&v| v as i32).sum::<i32>();
    let mut wp = w.as_ptr();
    for o in out.chunks_exact_mut(4) {
        let (mut kp, mut n) = (k.as_ptr(), stride / 8);
        // SAFETY: each group reads its 4 * stride weights (w.len() == rows * stride) and stride bytes of k; one
        // 16-byte store to o[..4].
        unsafe {
            asm!(
                "vmov.i32 q0, #0",
                "vmov.i32 q1, #0",
                "vmov.i32 q2, #0",
                "vmov.i32 q3, #0",
                "vdup.32 q8, {c}",
                "2:",
                "vld1.8 {{d20-d23}}, [{w}]!",
                "vld1.8 {{d18}}, [{k}]!",
                "pld [{w}, #192]",
                "vmull.u8 q12, d20, d18",
                "vmull.u8 q13, d21, d18",
                "vmull.u8 q14, d22, d18",
                "vmull.u8 q15, d23, d18",
                "vpadal.u16 q0, q12",
                "vpadal.u16 q1, q13",
                "vpadal.u16 q2, q14",
                "vpadal.u16 q3, q15",
                "subs {n}, {n}, #1",
                "bne 2b",
                "vpadd.i32 d0, d0, d1",
                "vpadd.i32 d2, d2, d3",
                "vpadd.i32 d4, d4, d5",
                "vpadd.i32 d6, d6, d7",
                "vpadd.i32 d0, d0, d2",
                "vpadd.i32 d1, d4, d6",
                "vsub.i32 q0, q0, q8",
                "vst1.32 {{d0-d1}}, [{o}]",
                w = inout(reg) wp,
                k = inout(reg) kp,
                n = inout(reg) n,
                c = in(reg) corr,
                o = in(reg) o.as_mut_ptr(),
                out("q0") _, out("q1") _, out("q2") _, out("q3") _, out("q8") _, out("q9") _,
                out("q10") _, out("q11") _, out("q12") _, out("q13") _, out("q14") _, out("q15") _,
                options(nostack),
            );
        }
        let _ = (kp, n);
    }
}

// NEON port of Julien Pommier's neon_mathfun exp_ps (zlib); altered version, see THIRD_PARTY_NOTICES.md.
// exp_ps constants: clamp, fx rounding, 1.0, poly terms added (P1..P5), then the by-scalar pair log2e, c1, c2, p0.
static ACT_K: [u32; 13] = [
    0x42B0C0A5, 0xC2B0C0A5, 0x3F000000, 0x3F800000, 0x3AB743CE, 0x3C088908, 0x3D2AA9C1, 0x3E2AAAAA, 0x3F000000,
    0x3FB8AA3B, 0xBF318000, 0x395E8083, 0x39506967,
];

// Four lanes of `recip(exp_ps(q0) + 1)` into q3, the order and roundings of num.rs; q0..q4 scratch, q5..q15 constants.
macro_rules! exp_recip {
    () => {
        concat!(
            "vmin.f32 q0, q0, q5\n",
            "vmax.f32 q0, q0, q6\n",
            "vmul.f32 q1, q0, d14[0]\n",
            "vadd.f32 q1, q1, q8\n",
            "vcvt.s32.f32 q2, q1\n",
            "vcvt.f32.s32 q3, q2\n",
            "vcgt.f32 q4, q3, q1\n",
            "vand q4, q4, q9\n",
            "vsub.f32 q1, q3, q4\n",
            "vmul.f32 q3, q1, d14[1]\n",
            "vadd.f32 q0, q0, q3\n",
            "vmul.f32 q3, q1, d15[0]\n",
            "vadd.f32 q0, q0, q3\n",
            "vmul.f32 q2, q0, d15[1]\n",
            "vadd.f32 q2, q2, q10\n",
            "vmul.f32 q2, q0, q2\n",
            "vadd.f32 q2, q2, q11\n",
            "vmul.f32 q2, q0, q2\n",
            "vadd.f32 q2, q2, q12\n",
            "vmul.f32 q2, q0, q2\n",
            "vadd.f32 q2, q2, q13\n",
            "vmul.f32 q2, q0, q2\n",
            "vadd.f32 q2, q2, q14\n",
            "vmul.f32 q3, q0, q0\n",
            "vmul.f32 q3, q3, q2\n",
            "vadd.f32 q2, q0, q3\n",
            "vadd.f32 q2, q2, q9\n",
            "vcvt.s32.f32 q3, q1\n",
            "vshl.i32 q3, q3, #23\n",
            "vadd.i32 q3, q3, q9\n",
            "vmul.f32 q2, q2, q3\n",
            "vadd.f32 q2, q2, q9\n",
            "vrecpe.f32 q3, q2\n",
            "vrecps.f32 q4, q2, q3\n",
            "vmul.f32 q3, q3, q4\n",
            "vrecps.f32 q4, q2, q3\n",
            "vmul.f32 q3, q4, q3\n",
        )
    };
}

macro_rules! act_loop {
    ($pre:expr, $post:expr, $x:ident, $n:ident) => {
        // SAFETY: reads and writes x[..n] (n a multiple of 4) four lanes at a time; constants come from ACT_K.
        unsafe {
            asm!(
                "vld1.32 {{d10[], d11[]}}, [{k}]!",
                "vld1.32 {{d12[], d13[]}}, [{k}]!",
                "vld1.32 {{d16[], d17[]}}, [{k}]!",
                "vld1.32 {{d18[], d19[]}}, [{k}]!",
                "vld1.32 {{d20[], d21[]}}, [{k}]!",
                "vld1.32 {{d22[], d23[]}}, [{k}]!",
                "vld1.32 {{d24[], d25[]}}, [{k}]!",
                "vld1.32 {{d26[], d27[]}}, [{k}]!",
                "vld1.32 {{d28[], d29[]}}, [{k}]!",
                "vld1.32 {{d14-d15}}, [{k}]",
                "vmov.i32 q15, #0",
                "2:",
                "vld1.32 {{d0-d1}}, [{p}]",
                "vsub.f32 q0, q15, q0",
                $pre,
                exp_recip!(),
                $post,
                "vst1.32 {{d6-d7}}, [{p}]!",
                "subs {n}, {n}, #4",
                "bne 2b",
                p = inout(reg) $x.as_mut_ptr() => _,
                n = inout(reg) $n => _,
                k = inout(reg) ACT_K.as_ptr() => _,
                out("q0") _, out("q1") _, out("q2") _, out("q3") _, out("q4") _, out("q5") _, out("q6") _,
                out("q7") _, out("q8") _, out("q9") _, out("q10") _, out("q11") _, out("q12") _, out("q13") _,
                out("q14") _, out("q15") _,
                options(nostack),
            );
        }
    };
}

pub(crate) fn sigmoid_inplace(x: &mut [f32]) {
    let n = x.len() & !3;
    if n > 0 {
        act_loop!("", "", x, n);
    }
    x[n..].iter_mut().for_each(|v| *v = crate::num::sigmoid(*v));
}

pub(crate) fn tanh_inplace(x: &mut [f32]) {
    let n = x.len() & !3;
    if n > 0 {
        act_loop!("vadd.f32 q0, q0, q0", "vadd.f32 q3, q3, q3\nvsub.f32 q3, q3, q9", x, n);
    }
    x[n..].iter_mut().for_each(|v| *v = crate::num::tanh(*v));
}

pub(crate) fn dequant_rows(acc: &[i32], s: [f32; 3], rs: &[f32], b: &[f32], a: &[f32], bias: &[f32], y: &mut [f32]) {
    let n = y.len();
    assert!(acc.len() >= n && rs.len() >= n && b.len() >= n && a.len() >= n && bias.len() >= n);
    let n4 = n & !3;
    if n4 > 0 {
        let k = [s[0], s[1], s[2], 0.0];
        // SAFETY: reads n4 elements of acc, rs, b, a, bias and y, writes n4 of y (all asserted >= n).
        unsafe {
            asm!(
                "vld1.32 {{d4-d5}}, [{k}]",
                "2:",
                "vld1.32 {{d0-d1}}, [{acc}]!",
                "vld1.32 {{d2-d3}}, [{rs}]!",
                "vcvt.f32.s32 q0, q0",
                "vld1.32 {{d16-d17}}, [{b}]!",
                "vmul.f32 q0, q0, d4[0]",
                "vld1.32 {{d18-d19}}, [{a}]!",
                "vmul.f32 q8, q8, d4[1]",
                "vmul.f32 q0, q0, q1",
                "vmul.f32 q9, q9, d5[0]",
                "vld1.32 {{d20-d21}}, [{y}]",
                "vadd.f32 q0, q0, q8",
                "vld1.32 {{d22-d23}}, [{bias}]!",
                "vadd.f32 q0, q0, q9",
                "vadd.f32 q0, q10, q0",
                "vadd.f32 q0, q11, q0",
                "vst1.32 {{d0-d1}}, [{y}]!",
                "subs {n}, {n}, #4",
                "bne 2b",
                k = in(reg) k.as_ptr(),
                acc = inout(reg) acc.as_ptr() => _,
                rs = inout(reg) rs.as_ptr() => _,
                b = inout(reg) b.as_ptr() => _,
                a = inout(reg) a.as_ptr() => _,
                bias = inout(reg) bias.as_ptr() => _,
                y = inout(reg) y.as_mut_ptr() => _,
                n = inout(reg) n4 => _,
                out("q0") _, out("q1") _, out("q2") _, out("q8") _, out("q9") _, out("q10") _, out("q11") _,
                options(nostack),
            );
        }
    }
    for r in n4..n {
        let mut t = (s[0] * acc[r] as f32) * rs[r];
        t = t + s[1] * b[r];
        t = t + s[2] * a[r];
        y[r] = bias[r] + (y[r] + t);
    }
}

pub(crate) fn dot_i8_rows(w: &[i8], cols: usize, q: &[i8], out: &mut [i32]) {
    assert!(cols > 0 && cols % 16 == 0 && q.len() >= cols && out.len() % 4 == 0);
    assert!(w.len() == out.len() * cols);
    for (g, o) in w.chunks_exact(4 * cols).zip(out.chunks_exact_mut(4)) {
        // SAFETY: reads the group's four rows of cols bytes and cols bytes of q; one 16-byte store to o[..4].
        unsafe {
            asm!(
                "add {r1}, {r0}, {s}",
                "add {r2}, {r1}, {s}",
                "add {s}, {r2}, {s}",
                "vmov.i32 q12, #0",
                "vmov.i32 q13, #0",
                "vmov.i32 q14, #0",
                "vmov.i32 q15, #0",
                "2:",
                "vld1.8 {{d0-d1}}, [{q}]!",
                "vld1.8 {{d2-d3}}, [{r0}]!",
                "vld1.8 {{d4-d5}}, [{r1}]!",
                "vld1.8 {{d6-d7}}, [{r2}]!",
                "vld1.8 {{d16-d17}}, [{s}]!",
                // q in 0..=127: two products |w·q| <= 128·127 still sum within i16, so pair them before widening
                "vmull.s8 q9, d2, d0",
                "vmlal.s8 q9, d3, d1",
                "vmull.s8 q10, d4, d0",
                "vmlal.s8 q10, d5, d1",
                "vmull.s8 q11, d6, d0",
                "vmlal.s8 q11, d7, d1",
                "vmull.s8 q1, d16, d0",
                "vmlal.s8 q1, d17, d1",
                "vpadal.s16 q12, q9",
                "vpadal.s16 q13, q10",
                "vpadal.s16 q14, q11",
                "vpadal.s16 q15, q1",
                "subs {n}, {n}, #1",
                "bne 2b",
                "vpadd.i32 d24, d24, d25",
                "vpadd.i32 d26, d26, d27",
                "vpadd.i32 d28, d28, d29",
                "vpadd.i32 d30, d30, d31",
                "vpadd.i32 d0, d24, d26",
                "vpadd.i32 d1, d28, d30",
                "vst1.32 {{d0-d1}}, [{o}]",
                r0 = inout(reg) g.as_ptr() => _,
                r1 = out(reg) _,
                r2 = out(reg) _,
                s = inout(reg) cols => _,
                q = inout(reg) q.as_ptr() => _,
                n = inout(reg) cols / 16 => _,
                o = in(reg) o.as_mut_ptr(),
                out("q0") _, out("q1") _, out("q2") _, out("q3") _, out("q8") _, out("q9") _,
                out("q10") _, out("q11") _, out("q12") _, out("q13") _, out("q14") _, out("q15") _,
                options(nostack),
            );
        }
    }
}

// CPUNK exp (num::cpunk_exp): clamp, n, log2e, int bias, C1, C2, P1 held; P0 and the Horner terms reloaded each pass.
static CPUNK_EXP_K: [u32; 12] = [
    0x42B1_7217, 0xC2AE_AC4F, 0x4B40_0000, 0x3FB8_AA3B, 0xB4C0_0000, 0xBF31_7200, 0xB5BF_BE8E, 0x3C08_0161,
    0x3D2B_A65B, 0x3E2A_ACD3, 0x3EFF_FEC6, 0x3F7F_FFFC,
];

pub(crate) fn cpunk_exp_inplace(x: &mut [f32]) {
    let n4 = x.len() & !3;
    if n4 > 0 {
        cpunk_exp4(x.as_mut_ptr(), n4);
    }
    if n4 < x.len() {
        let mut t = [0f32; 4];
        let tail = &mut x[n4..];
        t[..tail.len()].copy_from_slice(tail);
        cpunk_exp4(t.as_mut_ptr(), 4);
        tail.copy_from_slice(&t[..tail.len()]);
    }
}

/// `cpunk_exp` over `n` (a nonzero multiple of 4) floats at `p`.
fn cpunk_exp4(p: *mut f32, n: usize) {
    // SAFETY: the callers pass n <= the length of the slice at p; reads and writes p[..n] four lanes at a time.
    unsafe {
        asm!(
            "vld1.32 {{d8[], d9[]}}, [{k}]!",
            "vld1.32 {{d10[], d11[]}}, [{k}]!",
            "vld1.32 {{d12[], d13[]}}, [{k}]!",
            "vld1.32 {{d14[], d15[]}}, [{k}]!",
            "vld1.32 {{d16[], d17[]}}, [{k}]!",
            "vld1.32 {{d18[], d19[]}}, [{k}]!",
            "vld1.32 {{d20[], d21[]}}, [{k}]!",
            "vld1.32 {{d22[], d23[]}}, [{k}]!",
            "2:",
            "vld1.32 {{d0-d1}}, [{p}]",
            "mov {t}, {k}",
            "vmin.f32 q0, q0, q4",
            "vmax.f32 q0, q0, q5",
            "vmov q1, q6",
            "vmla.f32 q1, q0, q7",
            "vadd.i32 q2, q1, q8",
            "vcvt.f32.s32 q3, q2",
            "vfma.f32 q0, q3, q9",
            "vfma.f32 q0, q3, q10",
            "vld1.32 {{d24[], d25[]}}, [{t}]!",
            "vfma.f32 q12, q0, q11",
            "vld1.32 {{d26[], d27[]}}, [{t}]!",
            "vfma.f32 q13, q12, q0",
            "vld1.32 {{d28[], d29[]}}, [{t}]!",
            "vfma.f32 q14, q13, q0",
            "vld1.32 {{d30[], d31[]}}, [{t}]",
            "vfma.f32 q15, q14, q0",
            "vmov.f32 q12, #1.0",
            "vfma.f32 q12, q15, q0",
            "vshl.i32 q2, q2, #23",
            "vadd.i32 q0, q2, q12",
            "vst1.32 {{d0-d1}}, [{p}]!",
            "subs {n}, {n}, #4",
            "bne 2b",
            p = inout(reg) p => _,
            n = inout(reg) n => _,
            k = inout(reg) CPUNK_EXP_K.as_ptr() => _,
            t = out(reg) _,
            out("q0") _, out("q1") _, out("q2") _, out("q3") _, out("q4") _, out("q5") _, out("q6") _,
            out("q7") _, out("q8") _, out("q9") _, out("q10") _, out("q11") _, out("q12") _, out("q13") _,
            out("q14") _, out("q15") _,
            options(nostack),
        );
    }
}

pub(crate) fn fmaf_inplace(a: &[f32], b: &[f32], c: &mut [f32]) {
    let n = c.len();
    assert!(a.len() >= n && b.len() >= n);
    if n == 0 {
        return;
    }
    // SAFETY: reads n floats of a, b and c, writes n of c; FPSCR is restored before the block ends.
    unsafe {
        asm!(
            // VFP vfma honours FPSCR: clear DN, FZ and RMode so it rounds to nearest unflushed, as bionic's fmaf
            "vmrs {old}, fpscr",
            "bic {t}, {old}, #0x03C00000",
            "vmsr fpscr, {t}",
            "2:",
            "vldmia {a}!, {{s0}}",
            "vldmia {b}!, {{s1}}",
            "vldr s2, [{c}]",
            "vfma.f32 s2, s0, s1",
            "vstmia {c}!, {{s2}}",
            "subs {n}, {n}, #1",
            "bne 2b",
            "vmsr fpscr, {old}",
            old = out(reg) _,
            t = out(reg) _,
            a = inout(reg) a.as_ptr() => _,
            b = inout(reg) b.as_ptr() => _,
            c = inout(reg) c.as_mut_ptr() => _,
            n = inout(reg) n => _,
            out("d0") _, out("d1") _,
            options(nostack),
        );
    }
}

pub(crate) fn cpunk_minmax(x: &[f32]) -> (f32, f32) {
    let n4 = x.len() & !3;
    let (mut lo, mut hi) = (f32::MAX, -f32::MAX);
    if n4 > 0 {
        let (l, h): (u32, u32);
        // SAFETY: reads n4 floats of x.
        unsafe {
            asm!(
                "vdup.32 q0, {l}",
                "vdup.32 q1, {h}",
                "2:",
                "vld1.32 {{d4-d5}}, [{x}]!",
                "vmin.f32 q0, q0, q2",
                "vmax.f32 q1, q1, q2",
                "subs {n}, {n}, #4",
                "bne 2b",
                // min and max commute, NEON's ±0 and default-NaN rules included: lane order does not matter
                "vpmin.f32 d0, d0, d1",
                "vpmin.f32 d0, d0, d0",
                "vpmax.f32 d2, d2, d3",
                "vpmax.f32 d2, d2, d2",
                "vmov.32 {l}, d0[0]",
                "vmov.32 {h}, d2[0]",
                l = inout(reg) lo.to_bits() => l,
                h = inout(reg) hi.to_bits() => h,
                x = inout(reg) x.as_ptr() => _,
                n = inout(reg) n4 => _,
                out("q0") _, out("q1") _, out("q2") _,
                options(nostack),
            );
        }
        (lo, hi) = (f32::from_bits(l), f32::from_bits(h));
    }
    for &v in &x[n4..] {
        (lo, hi) = (crate::num::vmin(lo, v), crate::num::vmax(hi, v));
    }
    (lo, hi)
}

pub(crate) fn cpunk_quant4(x: &[f32], inv: f32, zb: i32, q: &mut [i8]) -> i32 {
    let n = x.len();
    assert!(n.is_multiple_of(4) && q.len() >= n);
    if n == 0 {
        return 0;
    }
    let sum: i32;
    // SAFETY: reads n floats of x, writes n bytes of q (one 4-byte lane store per pass).
    unsafe {
        asm!(
            "vdup.32 q8, {c}",
            "vdup.32 q9, {inv}",
            "vdup.32 q10, {zb}",
            "vmov.i32 q11, #0",
            "vmov.i32 q12, #127",
            "vmov.i32 q13, #0",
            "2:",
            "vld1.32 {{d0-d1}}, [{x}]!",
            "vmov q1, q8",
            "vmla.f32 q1, q9, q0",
            "vadd.i32 q1, q1, q10",
            "vmax.s32 q1, q1, q11",
            "vmin.s32 q1, q1, q12",
            "vadd.i32 q13, q13, q1",
            "vmovn.i32 d4, q1",
            "vmovn.i16 d4, q2",
            "vst1.32 {{d4[0]}}, [{q}]!",
            "subs {n}, {n}, #4",
            "bne 2b",
            "vpadd.i32 d26, d26, d27",
            "vpadd.i32 d26, d26, d26",
            "vmov.32 {c}, d26[0]",
            c = inout(reg) 12582912.0f32.to_bits() => sum,
            inv = in(reg) inv.to_bits(),
            zb = in(reg) zb,
            x = inout(reg) x.as_ptr() => _,
            q = inout(reg) q.as_mut_ptr() => _,
            n = inout(reg) n => _,
            out("q0") _, out("q1") _, out("q2") _, out("q8") _, out("q9") _, out("q10") _, out("q11") _,
            out("q12") _, out("q13") _,
            options(nostack),
        );
    }
    sum
}

pub(crate) fn cpunk_deq(dot: &[i32], w: [&[f32]; 3], bias: Option<&[f32]>, s: [f32; 3], y: &mut [f32]) {
    let [zw, sums, sw] = w;
    let n = y.len();
    assert!(dot.len() >= n && zw.len() >= n && sums.len() >= n && sw.len() >= n);
    assert!(bias.is_none_or(|b| b.len() >= n));
    let n4 = n & !3;
    if n4 > 0 {
        // SAFETY: reads n4 values of dot, zw, sums, sw and bias (when given), writes n4 of y.
        unsafe {
            asm!(
                "vld1.32 {{d16[], d17[]}}, [{k}]!",
                "vld1.32 {{d18[], d19[]}}, [{k}]!",
                "vld1.32 {{d20[], d21[]}}, [{k}]",
                "vmov.f32 q11, #1.0",
                "2:",
                "vld1.32 {{d0-d1}}, [{zw}]!",
                "vld1.32 {{d2-d3}}, [{sums}]!",
                "vmul.f32 q0, q8, q0",
                "vmla.f32 q0, q9, q1",
                "vld1.32 {{d4-d5}}, [{dot}]!",
                "vcvt.f32.s32 q2, q2",
                "vsub.f32 q2, q2, q0",
                "vld1.32 {{d6-d7}}, [{sw}]!",
                "vmul.f32 q3, q10, q3",
                "cmp {b}, #0",
                "beq 3f",
                "vld1.32 {{d0-d1}}, [{b}]!",
                "vmul.f32 q0, q0, q11",
                "vmla.f32 q0, q3, q2",
                "b 4f",
                "3:",
                "vmul.f32 q0, q3, q2",
                "4:",
                "vst1.32 {{d0-d1}}, [{y}]!",
                "subs {n}, {n}, #4",
                "bne 2b",
                k = inout(reg) s.as_ptr() => _,
                zw = inout(reg) zw.as_ptr() => _,
                sums = inout(reg) sums.as_ptr() => _,
                sw = inout(reg) sw.as_ptr() => _,
                dot = inout(reg) dot.as_ptr() => _,
                b = inout(reg) bias.map_or(core::ptr::null(), |b| b.as_ptr()) => _,
                y = inout(reg) y.as_mut_ptr() => _,
                n = inout(reg) n4 => _,
                out("q0") _, out("q1") _, out("q2") _, out("q3") _, out("q8") _, out("q9") _, out("q10") _,
                out("q11") _,
                options(nostack),
            );
        }
    }
    if n4 < n {
        let b = bias.map(|b| &b[n4..]);
        super::portable::cpunk_deq(&dot[n4..], [&zw[n4..], &sums[n4..], &sw[n4..]], b, s, &mut y[n4..]);
    }
}

pub(crate) fn cpunk_sigmoid_of_inplace(e: &mut [f32]) {
    cpunk_act_of(e, false);
}

pub(crate) fn cpunk_tanh_of_inplace(e: &mut [f32]) {
    cpunk_act_of(e, true);
}

/// `e / (e + 1)`, or `(e − 1) / (e + 1)` for tanh: NEON adds, VFP division under FZ as libpryon's CortexA15 build.
fn cpunk_act_of(e: &mut [f32], tanh: bool) {
    let n4 = e.len() & !3;
    if n4 > 0 {
        cpunk_act_of4(e.as_mut_ptr(), n4, tanh);
    }
    if n4 < e.len() {
        let mut t = [1f32; 4];
        let tail = &mut e[n4..];
        t[..tail.len()].copy_from_slice(tail);
        cpunk_act_of4(t.as_mut_ptr(), 4, tanh);
        tail.copy_from_slice(&t[..tail.len()]);
    }
}

/// Over `n` (a nonzero multiple of 4) floats at `p`.
fn cpunk_act_of4(p: *mut f32, n: usize, tanh: bool) {
    // SAFETY: the callers pass n <= the length of the slice at p; reads and writes p[..n]; FPSCR is restored.
    unsafe {
        asm!(
            // VFP vdiv honours FPSCR: FZ on, DN and RMode off (nearest), as the division libpryon runs
            "vmrs {old}, fpscr",
            "bic {t}, {old}, #0x03C00000",
            "orr {t}, {t}, #0x01000000",
            "vmsr fpscr, {t}",
            "vmov.f32 q8, #1.0",
            "vmov.f32 q9, #-1.0",
            "2:",
            "vld1.32 {{d0-d1}}, [{p}]",
            "vadd.f32 q1, q0, q8",
            "cmp {tanh}, #0",
            "beq 3f",
            "vadd.f32 q0, q0, q9",
            "3:",
            "vdiv.f32 s0, s0, s4",
            "vdiv.f32 s1, s1, s5",
            "vdiv.f32 s2, s2, s6",
            "vdiv.f32 s3, s3, s7",
            "vst1.32 {{d0-d1}}, [{p}]!",
            "subs {n}, {n}, #4",
            "bne 2b",
            "vmsr fpscr, {old}",
            old = out(reg) _,
            t = out(reg) _,
            p = inout(reg) p => _,
            n = inout(reg) n => _,
            tanh = in(reg) u32::from(tanh),
            out("q0") _, out("q1") _, out("q8") _, out("q9") _,
            options(nostack),
        );
    }
}
