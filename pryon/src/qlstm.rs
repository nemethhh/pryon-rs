//! Dynamic-quant int8 transformer (QAffine) and the LSTM cell built on it.

use crate::kernels::dot_i16_u8;
use crate::mlp::{QLstmParams, QMatrix};
use crate::num::{neon_sum, quantize, sigmoid, tanh};
use std::sync::Arc;

const LEVELS: f32 = 129.0;

/// `y = bias + (y + W x)` for one input row, `y` holding `y_old` on entry; `k` is scratch of at least `m.stride8`.
fn qaffine(x: &[f32], m: &QMatrix, bias: &[f32], y: &mut [f32], k: &mut [u8]) {
    let (lo, hi) = x.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), &v| (lo.min(v), hi.max(v)));
    let inv = if hi == lo { LEVELS } else { LEVELS / (hi - lo) };
    let off = lo + 1.0 / (inv + inv);
    let k = &mut k[..m.stride8];
    quantize(x, inv, off, &mut k[..x.len()]);
    k[x.len()..].fill(0);
    let sx = neon_sum(x);
    let rinv = 1.0 / inv;
    for (r, (w, yr)) in m.w.chunks_exact(m.stride8).zip(y.iter_mut()).enumerate() {
        let mut t = (rinv * dot_i16_u8(w, k) as f32) * m.rs[r];
        t = t + off * m.b[r];
        t = t + sx * m.a[r];
        *yr = bias[r] + (*yr + t);
    }
}

/// One LSTM layer: weights shared, state (h, c) and scratch owned, so clones are cheap per-detector copies.
#[derive(Clone)]
pub(crate) struct QLstm {
    p: Arc<QLstmParams>,
    h: Vec<f32>,
    c: Vec<f32>,
    z: Vec<f32>,
    k: Vec<u8>,
}

impl QLstm {
    pub fn new(p: QLstmParams) -> Self {
        let n = p.cell;
        let k = vec![0; p.wx.stride8.max(p.wh.stride8)];
        QLstm { h: vec![0.0; n], c: vec![0.0; n], z: vec![0.0; 4 * n], k, p: Arc::new(p) }
    }

    pub fn input(&self) -> usize {
        self.p.input
    }

    pub fn cell(&self) -> usize {
        self.p.cell
    }

    pub fn reset(&mut self) {
        self.h.fill(0.0);
        self.c.fill(0.0);
    }

    /// Advances one frame (`x.len() == input`) and returns the new `h`.
    pub fn step(&mut self, x: &[f32]) -> &[f32] {
        let (p, n) = (&*self.p, self.p.cell);
        let z = &mut self.z;
        z.fill(0.0);
        qaffine(x, &p.wx, &p.bx, z, &mut self.k);
        qaffine(&self.h, &p.wh, &p.bh, z, &mut self.k);
        for j in 0..n {
            let g = tanh(z[j]);
            let i = sigmoid(z[n + j]);
            let f = sigmoid(z[2 * n + j]);
            self.c[j] = f * self.c[j];
            self.c[j] = self.c[j] + i * g;
            let o = sigmoid(z[3 * n + j]);
            self.h[j] = o * tanh(self.c[j]);
        }
        &self.h
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::mlp::Mlp;
    use pryon_testkit::write_mlp;

    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    pub(crate) fn fnv(words: impl IntoIterator<Item = u32>) -> u64 {
        words
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .fold(0xcbf2_9ce4_8422_2325, |h, b| (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3))
    }

    /// Same sequence as `local/gen_task7.py`; step 2 is a constant row.
    pub(crate) fn inputs() -> Vec<[f32; 20]> {
        (0..5)
            .map(|t| {
                let mut x = [0.75f32; 20];
                if t != 2 {
                    for (i, v) in x.iter_mut().enumerate() {
                        *v = ((t * 20 + i) * 37 % 101) as f32 / 25.0 - 2.0;
                    }
                }
                x
            })
            .collect()
    }

    fn trunk() -> QLstmParams {
        Mlp::parse(write_mlp((20, 64), 9, 96, 1)).unwrap().qlstm("Hlstm0_layer_0").unwrap()
    }

    #[test]
    fn qaffine_equal_rows() {
        let p = trunk();
        let mut y = vec![0f32; 256];
        let mut k = [0u8; 24];
        qaffine(&[0.75; 20], &p.wx, &p.bx, &mut y, &mut k);
        assert!(y.iter().all(|v| v.is_finite()));
        assert_eq!(bits(&y[..4]), [0xBE906534, 0xBF6DCC64, 0x3F82C00A, 0xBF11DF08]);
        assert_eq!(fnv(bits(&y)), 5324349622486203207);
    }

    #[test]
    fn qlstm_matches_ref() {
        let mut l = QLstm::new(trunk());
        let mut all = Vec::new();
        let mut last = Vec::new();
        for x in inputs() {
            last = l.step(&x).to_vec();
            all.extend(bits(&last));
        }
        let want: [u32; 64] = [
            0xBD762DDF, 0xB96B866B, 0x26491916, 0x3F7DE1AE, 0xB9A29817, 0x399AD4A1, 0x36F5C4FB, 0xB7E34B57,
            0x3E049617, 0xBDCB0D64, 0x36D81C4F, 0xBC5DCBB9, 0xBD74ECDF, 0x31C76F75, 0xBEC84598, 0x3A94ECA0,
            0xB4374240, 0x3D13E49C, 0xB5CE2E4F, 0xB8670DB3, 0x260DE41A, 0x37C854B9, 0xBB3C5800, 0x391B76C7,
            0xBE72C473, 0xBA26A5A0, 0x3D98CB8A, 0x3A81493F, 0x39670683, 0x3D709D0B, 0x3E8CE258, 0x37F9E1AE,
            0xB173D600, 0xBF36180F, 0xB5116340, 0xBF197360, 0x3AAB8FED, 0xB3B43064, 0xBF23D636, 0x3E5219F9,
            0xBF5C658F, 0xB951D4C7, 0xBED98003, 0x3C582635, 0xAAF9DFF0, 0x3E4C1BAE, 0x300764F6, 0x38DA7FE0,
            0x3C8502B6, 0x3CC20E0A, 0xB23B13A6, 0x3F05DC32, 0xB6B5F455, 0x3BE3B5C2, 0xB7F4F025, 0xBD75306F,
            0xB65FAA2B, 0xAEC5F3EF, 0xBF0A24D5, 0xAD113844, 0x371BB320, 0x31E33474, 0x3F04B3D9, 0x36BFD846,
        ];
        assert_eq!(bits(&last), want);
        assert_eq!(fnv(all), 14823292988279750124);
    }

    #[test]
    fn qlstm_reset_clears_state() {
        let mut l = QLstm::new(trunk());
        let xs = inputs();
        let first = l.step(&xs[0]).to_vec();
        l.step(&xs[1]);
        l.reset();
        assert_eq!(bits(l.step(&xs[0])), bits(&first));
    }
}
