#[cfg(all(pryon_neon, target_arch = "arm"))]
mod armv7;
#[cfg(any(test, not(all(pryon_neon, target_arch = "arm"))))]
mod portable;

/// Dot product of int8-range weights with u8 activations; `w.len()` is a multiple of 8, `k.len() >= w.len()`.
#[cfg(all(pryon_neon, target_arch = "arm"))]
pub(crate) use armv7::dot_i16_u8;
#[cfg(not(all(pryon_neon, target_arch = "arm")))]
pub(crate) use portable::dot_i16_u8;

#[cfg(test)]
mod tests {
    #[test]
    fn dot_portable() {
        let w: Vec<i16> = (0..40).map(|i| ((i * 37) % 256 - 128) as i16).collect();
        let k: Vec<u8> = (0..40).map(|i| ((i * 91) % 256) as u8).collect();
        let want: i32 = (0..40).map(|i| w[i] as i32 * k[i] as i32).sum();
        assert_eq!(super::dot_i16_u8(&w, &k), want);
        let want8: i32 = (0..8).map(|i| w[i] as i32 * k[i] as i32).sum();
        assert_eq!(super::dot_i16_u8(&w[..8], &k), want8);
    }

    #[cfg(all(pryon_neon, target_arch = "arm"))]
    #[test]
    fn dot_asm_equals_portable() {
        let mut s = 0x1234_5678u32;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            s
        };
        for n in (8..=512).step_by(8) {
            let w: Vec<i16> = (0..n).map(|_| (next() % 256) as i16 - 128).collect();
            let k: Vec<u8> = (0..n).map(|_| next() as u8).collect();
            assert_eq!(super::dot_i16_u8(&w, &k), super::portable::dot_i16_u8(&w, &k), "n={n}");
        }
        let (w, k) = (vec![-128i16; 384], vec![255u8; 384]);
        assert_eq!(super::dot_i16_u8(&w, &k), -12_533_760);
        let (w, k) = (vec![127i16; 384], vec![255u8; 384]);
        assert_eq!(super::dot_i16_u8(&w, &k), 127 * 255 * 384);
    }
}
