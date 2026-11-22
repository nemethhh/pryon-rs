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
