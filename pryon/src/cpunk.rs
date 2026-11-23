//! The CPUNK quantised matrix product behind HybridGemm, HybridConv and HybridLSTM, as libpryon runs it
//! (CortexA15 target, FPSCR.FZ set): a per-row activation quantiser, an exact int8 dot product and libpryon's f32
//! dequantisation (libpryon.so 0x123897c: quantiser 0x1236440/0x1248fc0, matmul 0x516e80, dequant 0x123a544).

use crate::kernels::{cpunk_deq, cpunk_minmax, cpunk_quant4, dot_i8_rows};
use crate::num::{rint, zdiv, zmul, zsub};

/// Longest activation row; the quantised row is a stack buffer of this many bytes.
pub(crate) const MAX_COLS: usize = 4096;

/// Rows per `dot_i8_rows` call.
const BLOCK: usize = 16;

/// int8 weights `w` (`rows × cols`, row-major) with per-row scale `sw`, and zero point and row sum as the f32 the
/// dequantiser converts them to.
pub(crate) struct QWeights {
    pub rows: usize,
    pub cols: usize,
    /// `cols` rounded up to 16: `w` holds rows of `stride`, zero-padded, and is zero-padded to a multiple of 4 rows.
    stride: usize,
    w: Vec<i8>,
    pub sw: Vec<f32>,
    zw: Vec<f32>,
    sums: Vec<f32>,
}

impl QWeights {
    /// `sums` is the model's own row-sum input, which must agree with `w`.
    pub fn new(rows: usize, cols: usize, w: Vec<i8>, sw: Vec<f32>, zw: Vec<i32>, sums: &[i64]) -> Result<QWeights, String> {
        if rows == 0 || cols == 0 || cols > MAX_COLS {
            return Err(format!("weights of {rows} × {cols} (rows of 1..={MAX_COLS} columns)"));
        }
        if w.len() != rows * cols || sw.len() != rows || zw.len() != rows || sums.len() != rows {
            return Err(format!("weights, scales, zero points or sums do not match {rows} × {cols}"));
        }
        let own: Vec<i32> = w.chunks_exact(cols).map(|r| r.iter().map(|&v| i32::from(v)).sum()).collect();
        if let Some(r) = (0..rows).find(|&r| i64::from(own[r]) != sums[r]) {
            return Err(format!("sums input is not the weight row sum (row {r}: {} vs {})", sums[r], own[r]));
        }
        let zw = zw.iter().map(|&z| z as f32).collect();
        let stride = cols.next_multiple_of(16);
        let mut wp = vec![0i8; rows.next_multiple_of(4) * stride];
        for (p, r) in wp.chunks_exact_mut(stride).zip(w.chunks_exact(cols)) {
            p[..cols].copy_from_slice(r);
        }
        Ok(QWeights { rows, cols, stride, w: wp, sw, zw, sums: own.iter().map(|&v| v as f32).collect() })
    }
}

/// A quantised activation row: `x[c] ≈ s·(q[c] − z)` with `q[c]` in `0..=127` held in the row buffer.
pub(crate) struct RowQ {
    pub s: f32,
    pub z: i32,
    pub sum_q: i32,
}

/// libpryon's scale and zero point for a row spanning `lo..=hi` (0x1236440; f32 under FZ).
pub(crate) fn row_params(lo: f32, hi: f32) -> (f32, i32) {
    const TWO31: f32 = 2147483648.0;
    let d = zsub(hi, lo);
    // qmin 0 and qmax min(⌊46340/√K⌋ − 1, 127) = 127 for every K up to MAX_COLS
    if d.abs() < f32::MIN_POSITIVE {
        return if lo == 0.0 { (1.0, 0) } else { (-lo, 1) };
    }
    let (a, b) = (zdiv(-lo, -TWO31), zdiv(d, 127.0));
    let s = if a < b { b } else { a };
    let zf = zsub(0.0, zdiv(lo, s));
    let zf = if zf > -TWO31 { zf } else { -TWO31 };
    let zf = if zf < TWO31 { zf } else { TWO31 };
    (s, rint(zf) as i32)
}

/// Quantises `x` into `q[..x.len()]` as QuantizeLinearImpl<CortexA15> (0x1248fc0) does: NEON for all but the
/// last `len % 4` values, VFP for those.
pub(crate) fn quantize_row(x: &[f32], q: &mut [i8]) -> RowQ {
    let (lo, hi) = cpunk_minmax(x);
    let (s, z) = row_params(lo, hi);
    let inv = zdiv(1.0, s);
    let head = x.len() & !3;
    let mut sum_q = cpunk_quant4(&x[..head], inv, z.wrapping_add(0xB4C0_0000u32 as i32), q);
    for (o, &v) in q[head..].iter_mut().zip(&x[head..]) {
        let r = (rint(zmul(inv, v)) as i32).wrapping_add(z).clamp(0, 127);
        *o = r as i8;
        sum_q += r;
    }
    RowQ { s, z, sum_q }
}

/// The per-row constants of FusedDequantActivationImpl<CortexA15> (0x123a544): `[c, zf, sx]` for `kernels::cpunk_deq`.
fn deq_consts(rq: &RowQ, cols: usize) -> [f32; 3] {
    let zf = rq.z as f32;
    [zsub(rq.sum_q as f32, zmul(cols as f32, zf)), zf, zmul(rq.s, 1.0)]
}

/// `y[r]` for one activation row `x`: libpryon's quantised product, with `bias[r]` fused as in its kernel.
/// `scratch` holds at least `MAX_COLS` bytes.
pub(crate) fn qmatmul_row(x: &[f32], wq: &QWeights, bias: Option<&[f32]>, scratch: &mut [i8], y: &mut [f32]) {
    let (cols, stride) = (wq.cols, wq.stride);
    let q = &mut scratch[..stride];
    let d = deq_consts(&quantize_row(&x[..cols], q), cols);
    q[cols..].fill(0);
    let mut acc = [0i32; BLOCK];
    for (b, (w, yb)) in wq.w.chunks(BLOCK * stride).zip(y[..wq.rows].chunks_mut(BLOCK)).enumerate() {
        let acc = &mut acc[..w.len() / stride];
        dot_i8_rows(w, stride, q, acc);
        let r = b * BLOCK..b * BLOCK + yb.len();
        let consts = [&wq.zw[r.clone()], &wq.sums[r.clone()], &wq.sw[r.clone()]];
        cpunk_deq(acc, consts, bias.map(|v| &v[r]), d, yb);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wvals(rows: usize, cols: usize) -> Vec<i8> {
        (0..rows * cols).map(|i| ((i * 73 + 11) % 256) as u8 as i8).collect()
    }

    fn weights(rows: usize, cols: usize, zw: i32) -> QWeights {
        let w = wvals(rows, cols);
        let sums: Vec<i64> = w.chunks(cols).map(|r| r.iter().map(|&v| i64::from(v)).sum()).collect();
        QWeights::new(rows, cols, w, vec![0.01; rows], vec![zw; rows], &sums).unwrap()
    }

    fn quantized(x: &[f32]) -> (RowQ, Vec<i8>) {
        let mut q = vec![0i8; x.len()];
        let rq = quantize_row(x, &mut q);
        (rq, q)
    }

    /// libpryon's dequant (0x123a544) written out for one output: corr and acc in f32, then the scale product.
    fn dequant_ref(rq: &RowQ, q: &[i8], w: &[i8], sw: f32, zw: i32, bias: f32) -> f32 {
        let dot: i32 = w.iter().zip(q).map(|(&a, &b)| i32::from(a) * i32::from(b)).sum();
        let sumw: i32 = w.iter().map(|&v| i32::from(v)).sum();
        let zf = rq.z as f32;
        let c = rq.sum_q as f32 - q.len() as f32 * zf;
        let acc = dot as f32 - (c * zw as f32 + zf * sumw as f32);
        bias + (rq.s * sw) * acc
    }

    #[test]
    fn small_zero_point_is_exact() {
        let wq = weights(20, 37, -41);
        let mut q = [0i8; MAX_COLS];
        for (off, spread) in [(0.0f32, 1.0f32), (0.2, 0.5), (-0.1, 2.0)] {
            let x: Vec<f32> = (0..37).map(|i| off + spread * ((i * 29 % 37) as f32 / 18.0 - 1.0)).collect();
            let (rq, qx) = quantized(&x);
            let mut y = vec![0.0; 20];
            qmatmul_row(&x, &wq, None, &mut q, &mut y);
            for (r, row) in wvals(20, 37).chunks(37).enumerate() {
                let acc: i64 = row.iter().zip(&qx).map(|(&w, &v)| (i64::from(v) - i64::from(rq.z)) * i64::from(i32::from(w) + 41)).sum();
                assert_eq!(y[r].to_bits(), (acc as f32 * (rq.s * 0.01)).to_bits(), "off {off} row {r}");
            }
        }
    }

    #[test]
    fn negative_row_large_zero_point() {
        // a wholly negative row (a silence frame) has z ≈ 8000: z·Σw passes 2^24, so libpryon's f32 dequant rounds
        let k = 35;
        let x: Vec<f32> = (0..k).map(|i| -20.0 - 0.3 * ((i * 17 % k) as f32 / (k - 1) as f32)).collect();
        let w: Vec<i8> = (0..2 * k).map(|i| if i < k { 127 } else { -128 + (i % 3) as i8 }).collect();
        let sums: Vec<i64> = w.chunks(k).map(|r| r.iter().map(|&v| i64::from(v)).sum()).collect();
        let (sw, zw) = ([0.003f32, 0.002], [-90, 77]);
        let wq = QWeights::new(2, k, w.clone(), sw.to_vec(), zw.to_vec(), &sums).unwrap();
        let (rq, qx) = quantized(&x);
        assert!((7000..9000).contains(&rq.z), "z {}", rq.z);
        let mut y = [0f32; 2];
        qmatmul_row(&x, &wq, None, &mut [0i8; MAX_COLS], &mut y);
        let mut inexact = 0;
        for r in 0..2 {
            let row = &w[r * k..][..k];
            let want = dequant_ref(&rq, &qx, row, sw[r], zw[r], 0.0);
            assert_eq!(y[r].to_bits(), want.to_bits(), "row {r}: {} vs {want}", y[r]);
            let exact: i64 = row.iter().zip(&qx).map(|(&a, &b)| (i64::from(b) - i64::from(rq.z)) * i64::from(i32::from(a) - zw[r])).sum();
            inexact += usize::from(want != exact as f32 * (rq.s * sw[r]));
        }
        assert!(inexact > 0, "the exact integer accumulator gives the same outputs");
    }

    #[test]
    fn zero_point_by_division() {
        let (lo, hi) = (f32::from_bits(0xC191_7198), f32::from_bits(0xC191_3909));
        let (s, z) = row_params(lo, hi);
        assert_eq!(s.to_bits(), ((hi - lo) / 127.0).to_bits());
        // 0 − lo/s, not −lo·(1/s), which gives 83606
        assert_eq!(z, 83607);
    }

    #[test]
    fn q_clamped_on_both_sides() {
        // at lo, rint(lo·inv) + z = −1: held at 0 in the NEON head and the VFP tail
        let (lo, hi) = (f32::from_bits(0xC203_0A3D), f32::from_bits(0xC1FE_7AE1));
        for k in [2, 4, 6] {
            let x: Vec<f32> = (0..k).map(|i| if i % 2 == 0 { lo } else { hi }).collect();
            let (rq, q) = quantized(&x);
            assert_eq!(rq.z, 4379);
            assert!(q.iter().step_by(2).all(|&v| v == 0), "k {k}: {q:?}");
            assert_eq!(rq.sum_q, q.iter().map(|&v| i32::from(v)).sum::<i32>());
        }
        // s = 1, z = rint(63.5) = 64: 63.5 quantises to 64 + 64 = 128, held at 127
        assert_eq!(quantized(&[-63.5, 63.5, 0.0, 0.0]).1, [0, 127, 64, 64]);
        assert_eq!(quantized(&[-63.5, 63.5]).1, [0, 127]);
    }

    #[test]
    fn tie_rounding_half_even() {
        // s = 1, z = 0: ties go to even in the NEON head (first 4) and the VFP tail
        assert_eq!(quantized(&[0.0, 2.5, 3.5, 127.0, 0.5, 1.5, 4.5]).1, [0, 2, 4, 127, 0, 2, 4]);
    }

    #[test]
    fn neon_rounding_beyond_2_22() {
        // a tiny range far from zero: |x·inv| > 2^22, where the head's 1.5·2^23 trick no longer rounds to integers
        let x = [1000.0f32, 1000.0001, 1000.0, 1000.0001, 1000.0, 1000.0001];
        let (rq, q) = quantized(&x);
        let inv = 1.0 / rq.s;
        for (i, (&v, &got)) in x.iter().zip(&q).enumerate() {
            let want = if i < 4 {
                let t = 12582912.0f32 + inv * v;
                (t.to_bits() as i32).wrapping_add(rq.z.wrapping_add(0xB4C0_0000u32 as i32)).clamp(0, 127)
            } else {
                ((inv * v).round_ties_even() as i32).wrapping_add(rq.z).clamp(0, 127)
            };
            assert_eq!(i32::from(got), want, "x[{i}]");
        }
        assert_ne!((q[0], q[1]), (q[4], q[5]), "head and tail agree here");
    }

    #[test]
    fn sums_must_be_row_sums() {
        let e = QWeights::new(2, 2, vec![1, 2, 3, 4], vec![1.0; 2], vec![0; 2], &[3, 6]).err().unwrap();
        assert!(e.contains("row 1: 6 vs 7"), "{e}");
    }

    #[test]
    fn constant_row_scale_minus_lo() {
        // |hi − lo| < FLT_MIN: s = −lo, z = 1 and every q' = 0, so x = s·(0 − 1); lo = 0 gives s = 1, z = 0
        assert_eq!(row_params(-2.5, -2.5), (2.5, 1));
        assert_eq!(row_params(3.0, 3.0), (-3.0, 1));
        assert_eq!(row_params(0.0, 0.0), (1.0, 0));
        let wq = weights(3, 5, 7);
        let x = [-2.5f32; 5];
        let (rq, qx) = quantized(&x);
        assert_eq!(qx, [0; 5]);
        let mut y = [0.0; 3];
        qmatmul_row(&x, &wq, Some(&[1.0, 2.0, 3.0]), &mut [0i8; MAX_COLS], &mut y);
        for (r, row) in wvals(3, 5).chunks(5).enumerate() {
            assert_eq!(y[r].to_bits(), dequant_ref(&rq, &qx, row, 0.01, 7, (r + 1) as f32).to_bits(), "row {r}");
        }
    }

    #[test]
    fn subnormal_scale_flushes_to_zero() {
        // (hi − lo)/127 and lo/2^31 are subnormal: FZ makes s = 0, z saturates and every output is a zero
        let (s, z) = row_params(1e-36, 2e-36);
        assert_eq!((s.to_bits(), z), (0, i32::MIN));
        let mut y = [1.0; 3];
        qmatmul_row(&[1e-36, 2e-36, 1.5e-36, 1e-36], &weights(3, 4, 7), None, &mut [0i8; MAX_COLS], &mut y);
        assert!(y.iter().all(|v| *v == 0.0), "{y:?}");
    }
}
