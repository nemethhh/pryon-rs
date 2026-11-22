pub(crate) fn dot_i16_u8(w: &[i16], k: &[u8]) -> i32 {
    w.iter().zip(k).map(|(&a, &b)| a as i32 * b as i32).sum()
}

#[cfg(test)]
mod tests {
    #[test]
    fn dot_portable() {
        let w: Vec<i16> = (0..37).map(|i| ((i * 37) % 256 - 128) as i16).collect();
        let k: Vec<u8> = (0..37).map(|i| ((i * 91) % 256) as u8).collect();
        let want: i32 = (0..37).map(|i| w[i] as i32 * k[i] as i32).sum();
        assert_eq!(super::dot_i16_u8(&w, &k), want);
        let want5: i32 = (0..5).map(|i| w[i] as i32 * k[i] as i32).sum();
        assert_eq!(super::dot_i16_u8(&w[..5], &k), want5);
    }
}
