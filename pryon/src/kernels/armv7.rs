use core::arch::asm;

/// `w.len()` must be a multiple of 8 and `k.len() >= w.len()`; unaligned loads are fine.
pub(crate) fn dot_i16_u8(w: &[i16], k: &[u8]) -> i32 {
    assert!(w.len() % 8 == 0 && k.len() >= w.len());
    let mut n = w.len();
    let (mut wp, mut kp) = (w.as_ptr(), k.as_ptr());
    let r: i32;
    // SAFETY: the asserts bound every load to w[..n] and k[..n]; no stores, no stack use.
    unsafe {
        asm!(
            "vmov.i32 q0, #0",
            "vmov.i32 q1, #0",
            "vmov.i32 q2, #0",
            "vmov.i32 q3, #0",
            "lsrs {t}, {n}, #4",
            "beq 3f",
            "2:",
            "vld1.16 {{d16-d19}}, [{w}]!",
            "vld1.8 {{d20-d21}}, [{k}]!",
            "vmovl.u8 q11, d20",
            "vmovl.u8 q12, d21",
            "vmlal.s16 q0, d16, d22",
            "vmlal.s16 q1, d17, d23",
            "vmlal.s16 q2, d18, d24",
            "vmlal.s16 q3, d19, d25",
            "subs {t}, {t}, #1",
            "bne 2b",
            "3:",
            "tst {n}, #8",
            "beq 4f",
            "vld1.16 {{d16-d17}}, [{w}]!",
            "vld1.8 {{d20}}, [{k}]!",
            "vmovl.u8 q11, d20",
            "vmlal.s16 q0, d16, d22",
            "vmlal.s16 q1, d17, d23",
            "4:",
            "vadd.i32 q0, q0, q1",
            "vadd.i32 q2, q2, q3",
            "vadd.i32 q0, q0, q2",
            "vadd.i32 d0, d0, d1",
            "vpadd.i32 d0, d0, d0",
            "vmov.32 {r}, d0[0]",
            w = inout(reg) wp,
            k = inout(reg) kp,
            n = inout(reg) n,
            t = out(reg) _,
            r = out(reg) r,
            out("q0") _, out("q1") _, out("q2") _, out("q3") _,
            out("q8") _, out("q9") _, out("q10") _, out("q11") _, out("q12") _,
            options(pure, readonly, nostack),
        );
    }
    let _ = (wp, kp, n);
    r
}

/// `out[r] = w[r * stride..][..stride] · k` for every row; four rows share each load of `k`.
pub(crate) fn dot_rows(w: &[i16], stride: usize, k: &[u8], out: &mut [i32]) {
    assert!(stride > 0 && stride % 8 == 0 && k.len() >= stride && w.len() == out.len() * stride);
    let quads = out.len() / 4;
    for (q, o) in out.chunks_exact_mut(4).enumerate() {
        let w0 = w[4 * q * stride..].as_ptr();
        // SAFETY: rows 4q..4q+4 lie inside w (len == rows * stride), so all four pointers stay in bounds.
        let (mut w1, mut w2, mut w3) = unsafe { (w0.add(stride), w0.add(2 * stride), w0.add(3 * stride)) };
        let (mut w0, mut kp, mut n) = (w0, k.as_ptr(), stride / 8);
        // SAFETY: each pointer reads stride elements of its row (k: stride bytes); one 16-byte store to o[..4].
        unsafe {
            asm!(
                "vmov.i32 q0, #0",
                "vmov.i32 q1, #0",
                "vmov.i32 q2, #0",
                "vmov.i32 q3, #0",
                "2:",
                "vld1.8 {{d22}}, [{k}]!",
                "vld1.16 {{d16-d17}}, [{w0}]!",
                "vld1.16 {{d18-d19}}, [{w1}]!",
                "vmovl.u8 q10, d22",
                "vld1.16 {{d24-d25}}, [{w2}]!",
                "vld1.16 {{d26-d27}}, [{w3}]!",
                "vmlal.s16 q0, d16, d20",
                "vmlal.s16 q1, d18, d20",
                "vmlal.s16 q2, d24, d20",
                "vmlal.s16 q3, d26, d20",
                "vmlal.s16 q0, d17, d21",
                "vmlal.s16 q1, d19, d21",
                "vmlal.s16 q2, d25, d21",
                "vmlal.s16 q3, d27, d21",
                "subs {n}, {n}, #1",
                "bne 2b",
                "vpadd.i32 d0, d0, d1",
                "vpadd.i32 d2, d2, d3",
                "vpadd.i32 d4, d4, d5",
                "vpadd.i32 d6, d6, d7",
                "vpadd.i32 d0, d0, d2",
                "vpadd.i32 d1, d4, d6",
                "vst1.32 {{d0-d1}}, [{o}]",
                w0 = inout(reg) w0,
                w1 = inout(reg) w1,
                w2 = inout(reg) w2,
                w3 = inout(reg) w3,
                k = inout(reg) kp,
                n = inout(reg) n,
                o = in(reg) o.as_mut_ptr(),
                out("q0") _, out("q1") _, out("q2") _, out("q3") _,
                out("q8") _, out("q9") _, out("q10") _, out("q11") _, out("q12") _, out("q13") _,
                options(nostack),
            );
        }
        let _ = (w0, w1, w2, w3, kp, n);
    }
    for (r, o) in out.iter_mut().enumerate().skip(4 * quads) {
        *o = dot_i16_u8(&w[r * stride..(r + 1) * stride], k);
    }
}
