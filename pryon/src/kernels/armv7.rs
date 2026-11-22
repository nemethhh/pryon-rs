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
