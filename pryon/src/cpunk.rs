//! The CPUNK quantised matrix product behind HybridGemm, HybridConv and HybridLSTM: a per-row activation
//! quantiser, an int8 dot product and a float epilogue, on pypryon's measured contract.

use crate::kernels::dot_i8_rows;

/// Longest activation row; the quantised row is a stack buffer of this many bytes.
pub(crate) const MAX_COLS: usize = 4096;

/// Rows per `dot_i8_rows` call.
const BLOCK: usize = 16;

/// int8 weights `w` (`rows × cols`, row-major) with per-row scale `sw`, zero point `zw` and `sums[r] = Σ_c w[r][c]`.
pub(crate) struct QWeights {
    pub rows: usize,
    pub cols: usize,
    pub w: Vec<i8>,
    pub sw: Vec<f32>,
    pub zw: Vec<i32>,
    pub sums: Vec<i32>,
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
        Ok(QWeights { rows, cols, w, sw, zw, sums: own })
    }
}

/// A quantised activation row: `x[c] ≈ s·(q[c] − z)` with `q[c]` in `0..=127` held in the row buffer.
pub(crate) struct RowQ {
    pub s: f32,
    pub z: i64,
    pub sum_q: i32,
}

/// Quantises `x` into `q[..x.len()]`; `None` when the scale is zero (a constant row), which takes [`exact_row`].
pub(crate) fn quantize_row(x: &[f32], q: &mut [i8]) -> Option<RowQ> {
    let (mut lo, mut hi) = (x[0], x[0]);
    for &v in x {
        lo = if v < lo { v } else { lo };
        hi = if v > hi { v } else { hi };
    }
    let s = (hi - lo) / 127.0;
    // hi == lo, or a range so small its scale underflows (pypryon would divide by zero there)
    if s == 0.0 {
        return None;
    }
    let inv = 1.0 / s;
    let z = (-lo * inv).round_ties_even();
    let mut sum_q = 0i32;
    for (o, &v) in q.iter_mut().zip(x) {
        *o = ((v * inv).round_ties_even() + z).min(127.0) as i8;
        sum_q += i32::from(*o);
    }
    Some(RowQ { s, z: z as i64, sum_q })
}

/// `y = f32(acc)·(s·sw[r])` for `acc = Σ_c (q[c] − z)(w[r][c] − zw[r])`, expanded around `dot = Σ_c q[c]·w[r][c]`.
pub(crate) fn epilogue(dot: i32, r: usize, rq: &RowQ, wq: &QWeights) -> f32 {
    let zw = i64::from(wq.zw[r]);
    let wsum = i64::from(wq.sums[r]) - wq.cols as i64 * zw;
    // exact in i64 for every zero point below 2^24; wraps (never panics) on absurd ones
    let acc = (i64::from(dot) - zw * i64::from(rq.sum_q)).wrapping_sub(rq.z.wrapping_mul(wsum));
    acc as f32 * (rq.s * wq.sw[r])
}

/// The constant-row path: `y[r] = (Σ_c x[c]·(w[r][c] − zw[r]))·sw[r]`, sequential f32.
pub(crate) fn exact_row(x: &[f32], wq: &QWeights, y: &mut [f32]) {
    for (r, (row, o)) in wq.w.chunks_exact(wq.cols).zip(y.iter_mut()).enumerate() {
        let zw = wq.zw[r];
        let acc = row.iter().zip(x).fold(0f32, |a, (&w, &v)| a + v * (i32::from(w) - zw) as f32);
        *o = acc * wq.sw[r];
    }
}

/// `y[r] = epilogue(Σ_c q[c]·(w[r][c] − zw[r]))` for one activation row `x`, plus `bias[r]` when given.
pub(crate) fn qmatmul_row(x: &[f32], wq: &QWeights, bias: Option<&[f32]>, scratch: &mut [i8], y: &mut [f32]) {
    let cols = wq.cols;
    let q = &mut scratch[..cols];
    match quantize_row(&x[..cols], q) {
        Some(rq) => {
            let mut acc = [0i32; BLOCK];
            for (b, (w, yb)) in wq.w.chunks(BLOCK * cols).zip(y[..wq.rows].chunks_mut(BLOCK)).enumerate() {
                let acc = &mut acc[..yb.len()];
                dot_i8_rows(w, cols, q, acc);
                for (j, (o, &d)) in yb.iter_mut().zip(acc.iter()).enumerate() {
                    *o = epilogue(d, b * BLOCK + j, &rq, wq);
                }
            }
        }
        None => exact_row(x, wq, y),
    }
    if let Some(b) = bias {
        for (o, &b) in y[..wq.rows].iter_mut().zip(b) {
            *o = *o + b;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn weights(rows: usize, cols: usize, zw: i32) -> QWeights {
        let w: Vec<i8> = (0..rows * cols).map(|i| ((i * 73 + 11) % 256) as u8 as i8).collect();
        let sums: Vec<i64> = w.chunks(cols).map(|r| r.iter().map(|&v| i64::from(v)).sum()).collect();
        QWeights::new(rows, cols, w, vec![0.01; rows], vec![zw; rows], &sums).unwrap()
    }

    #[test]
    fn expansion_equals_direct_sum() {
        let wq = weights(20, 37, -41);
        let mut q = [0i8; MAX_COLS];
        // mixed-sign, wholly negative (zero point far above 127) and wholly positive rows
        for (off, spread) in [(0.0f32, 1.0f32), (-20.0, 0.003), (5.0, 0.5), (-1e3, 1.0)] {
            let x: Vec<f32> = (0..37).map(|i| off + spread * ((i * 29 % 37) as f32 / 18.0 - 1.0)).collect();
            let rq = quantize_row(&x, &mut q).unwrap();
            assert!(q[..37].iter().all(|&v| (0..=127).contains(&v)));
            let mut y = vec![0.0; 20];
            qmatmul_row(&x, &wq, None, &mut q, &mut y);
            for (r, row) in wq.w.chunks(37).enumerate() {
                let acc: i64 = row.iter().zip(&q).map(|(&w, &v)| (i64::from(v) - rq.z) * i64::from(i32::from(w) + 41)).sum();
                assert_eq!(y[r].to_bits(), (acc as f32 * (rq.s * 0.01)).to_bits(), "off {off} row {r}");
            }
        }
    }

    #[test]
    fn sums_must_be_row_sums() {
        let e = QWeights::new(2, 2, vec![1, 2, 3, 4], vec![1.0; 2], vec![0; 2], &[3, 6]).err().unwrap();
        assert!(e.contains("row 1: 6 vs 7"), "{e}");
    }

    #[test]
    fn constant_row_takes_exact_path() {
        let wq = weights(3, 5, 7);
        let mut q = [0i8; 5];
        assert!(quantize_row(&[-2.5; 5], &mut q).is_none());
        let mut y = [0.0; 3];
        qmatmul_row(&[-2.5; 5], &wq, Some(&[1.0, 2.0, 3.0]), &mut q, &mut y);
        for (r, row) in wq.w.chunks(5).enumerate() {
            let want = row.iter().map(|&w| -2.5 * (i32::from(w) - 7) as f32).sum::<f32>() * 0.01 + (r + 1) as f32;
            assert_eq!(y[r], want);
        }
    }
}
