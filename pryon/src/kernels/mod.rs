#[cfg(all(pryon_neon, target_arch = "arm"))]
mod armv7;
#[cfg(any(test, not(all(pryon_neon, target_arch = "arm"))))]
mod portable;

/// Matrix-vector product of int8-range weights (rows of `stride`, a multiple of 8) with u8 activations.
#[cfg(all(pryon_neon, target_arch = "arm"))]
pub(crate) use armv7::dot_rows;
#[cfg(not(all(pryon_neon, target_arch = "arm")))]
pub(crate) use portable::dot_rows;
#[cfg(all(test, pryon_neon, target_arch = "arm"))]
use armv7::dot_i16_u8;
#[cfg(all(test, not(all(pryon_neon, target_arch = "arm"))))]
use portable::dot_i16_u8;

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

    #[test]
    fn dot_rows_each_row() {
        let w: Vec<i16> = (0..7 * 24).map(|i| ((i * 37) % 256 - 128) as i16).collect();
        let k: Vec<u8> = (0..24).map(|i| ((i * 91) % 256) as u8).collect();
        let mut out = [0i32; 7];
        super::dot_rows(&w, 24, &k, &mut out);
        for (r, &o) in out.iter().enumerate() {
            assert_eq!(o, (0..24).map(|i| w[r * 24 + i] as i32 * k[i] as i32).sum::<i32>(), "row {r}");
        }
    }

    #[cfg(all(pryon_neon, target_arch = "arm"))]
    #[test]
    fn dot_rows_asm_equals_portable() {
        let mut s = 0x9e37_79b9u32;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            s
        };
        for rows in 1..=13 {
            for stride in [8, 16, 24, 64, 96, 104, 512] {
                let w: Vec<i16> = (0..rows * stride).map(|_| (next() % 256) as i16 - 128).collect();
                let k: Vec<u8> = (0..stride + 8).map(|_| next() as u8).collect();
                let (mut a, mut b) = (vec![0i32; rows], vec![1i32; rows]);
                super::dot_rows(&w, stride, &k, &mut a);
                super::portable::dot_rows(&w, stride, &k, &mut b);
                assert_eq!(a, b, "rows={rows} stride={stride}");
            }
        }
        let (w, k) = (vec![-128i16; 4 * 384], vec![255u8; 384]);
        let mut out = [0i32; 4];
        super::dot_rows(&w, 384, &k, &mut out);
        assert_eq!(out, [-12_533_760; 4]);
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
