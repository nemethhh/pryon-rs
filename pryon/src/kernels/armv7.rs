use core::arch::asm;

pub(crate) fn dot_rows(w: &[i8], stride: usize, k: &[u8], out: &mut [i32]) {
    assert!(stride > 0 && stride % 8 == 0 && k.len() >= stride && out.len() % 4 == 0);
    assert!(w.len() == out.len() * stride);
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
                "2:",
                "vld1.8 {{d16-d19}}, [{w}]!",
                "vld1.8 {{d22}}, [{k}]!",
                "pld [{w}, #192]",
                "vmovl.s8 q12, d16",
                "vmovl.s8 q13, d17",
                "vmovl.u8 q10, d22",
                "vmovl.s8 q14, d18",
                "vmovl.s8 q15, d19",
                "vmlal.s16 q0, d24, d20",
                "vmlal.s16 q1, d26, d20",
                "vmlal.s16 q2, d28, d20",
                "vmlal.s16 q3, d30, d20",
                "vmlal.s16 q0, d25, d21",
                "vmlal.s16 q1, d27, d21",
                "vmlal.s16 q2, d29, d21",
                "vmlal.s16 q3, d31, d21",
                "subs {n}, {n}, #1",
                "bne 2b",
                "vpadd.i32 d0, d0, d1",
                "vpadd.i32 d2, d2, d3",
                "vpadd.i32 d4, d4, d5",
                "vpadd.i32 d6, d6, d7",
                "vpadd.i32 d0, d0, d2",
                "vpadd.i32 d1, d4, d6",
                "vst1.32 {{d0-d1}}, [{o}]",
                w = inout(reg) wp,
                k = inout(reg) kp,
                n = inout(reg) n,
                o = in(reg) o.as_mut_ptr(),
                out("q0") _, out("q1") _, out("q2") _, out("q3") _, out("q8") _, out("q9") _,
                out("q10") _, out("q11") _, out("q12") _, out("q13") _, out("q14") _, out("q15") _,
                options(nostack),
            );
        }
        let _ = (kp, n);
    }
}
