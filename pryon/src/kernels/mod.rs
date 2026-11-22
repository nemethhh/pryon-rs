mod portable;

/// Dot product of int8-range weights with u8 activations.
pub(crate) fn dot_i16_u8(w: &[i16], k: &[u8]) -> i32 {
    portable::dot_i16_u8(w, k)
}
