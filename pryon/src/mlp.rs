//! dnnrt `.mlp` reader: Kaldi-style tokens, `dynamic_quant_int_8_lstm` and `affinetransform`.

use crate::Error;

pub struct QMatrix {
    pub rows: usize,
    pub cols: usize,
    /// `cols` rounded up to a multiple of 8.
    pub stride8: usize,
    /// int8 weights widened to i16, `rows * stride8`, each row zero-padded past `cols`.
    pub w: Vec<i16>,
    /// 1/s per row.
    pub rs: Vec<f32>,
    pub a: Vec<f32>,
    pub b: Vec<f32>,
}

pub struct QLstmParams {
    pub input: usize,
    pub cell: usize,
    pub wx: QMatrix,
    pub bx: Vec<f32>,
    pub wh: QMatrix,
    pub bh: Vec<f32>,
}

pub struct AffineParams {
    pub rows: usize,
    pub cols: usize,
    pub w: Vec<f32>,
    pub b: Vec<f32>,
}

pub struct Mlp {
    bytes: Vec<u8>,
}

fn fmt_err<T>(m: impl Into<String>) -> Result<T, Error> {
    Err(Error::Format(m.into()))
}

fn find(hay: &[u8], needle: &[u8], from: usize) -> Option<usize> {
    let tail = hay.get(from..)?;
    tail.windows(needle.len()).position(|w| w == needle).map(|i| i + from)
}

struct Cur<'a> {
    b: &'a [u8],
    p: usize,
}

impl<'a> Cur<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], Error> {
        let end = self.p.checked_add(n).filter(|&e| e <= self.b.len());
        let Some(end) = end else { return fmt_err(format!("truncated at {}", self.p)) };
        let s = &self.b[self.p..end];
        self.p = end;
        Ok(s)
    }

    fn tag(&mut self, t: &[u8]) -> Result<(), Error> {
        let at = self.p;
        if self.take(t.len())? != t {
            return fmt_err(format!("expected {:?} at {at}", String::from_utf8_lossy(t)));
        }
        Ok(())
    }

    fn marked(&mut self, marker: u8, n: usize) -> Result<&'a [u8], Error> {
        let at = self.p;
        if self.take(1)?[0] != marker {
            return fmt_err(format!("expected marker {marker:#04x} at {at}"));
        }
        self.take(n)
    }

    fn i32m(&mut self) -> Result<usize, Error> {
        let v = i32::from_le_bytes(self.marked(0x04, 4)?.try_into().unwrap());
        usize::try_from(v).or_else(|_| fmt_err(format!("negative dimension {v}")))
    }

    fn u64m(&mut self) -> Result<usize, Error> {
        let v = u64::from_le_bytes(self.marked(0xf8, 8)?.try_into().unwrap());
        usize::try_from(v).or_else(|_| fmt_err("dimension too large"))
    }

    fn f32s(&mut self, n: usize) -> Result<Vec<f32>, Error> {
        let Some(len) = n.checked_mul(4) else { return fmt_err("length overflow") };
        Ok(self.take(len)?.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())
    }

    fn fv(&mut self, want: usize) -> Result<Vec<f32>, Error> {
        self.tag(b"FV ")?; // count is marker-prefixed like the other ints
        let n = self.i32m()?;
        if n != want {
            return fmt_err(format!("vector length {n}, expected {want}"));
        }
        self.f32s(n)
    }

    fn qblock(&mut self, rows: usize, cols: usize) -> Result<QMatrix, Error> {
        self.tag(b"v1 ")?;
        let s = self.fv(rows)?;
        let a = self.fv(rows)?;
        let b = self.fv(rows)?;
        let (r, c, stride) = (self.u64m()?, self.u64m()?, self.u64m()?);
        if r != rows || c != cols {
            return fmt_err(format!("matrix {r}x{c}, expected {rows}x{cols}"));
        }
        if stride < cols {
            return fmt_err(format!("stride {stride} < cols {cols}"));
        }
        let Some(total) = rows.checked_mul(stride) else { return fmt_err("length overflow") };
        let raw = self.take(total)?;
        let stride8 = cols.div_ceil(8) * 8;
        let mut w = Vec::with_capacity(rows * stride8);
        for row in raw.chunks_exact(stride) {
            w.extend(row[..cols].iter().map(|&x| x as i8 as i16));
            w.resize(w.len() + stride8 - cols, 0);
        }
        let rs = s.iter().map(|&x| 1.0f32 / x).collect();
        Ok(QMatrix { rows, cols, stride8, w, rs, a, b })
    }
}

impl Mlp {
    pub fn parse(bytes: Vec<u8>) -> Result<Mlp, Error> {
        if bytes.is_empty() {
            return fmt_err("empty model file");
        }
        Ok(Mlp { bytes })
    }

    fn locate(&self, name: &str, ty: &str) -> Result<usize, Error> {
        let key = format!("<component> {name} <{ty}> ");
        match find(&self.bytes, key.as_bytes(), 0) {
            Some(p) => Ok(p + key.len()),
            None => fmt_err(format!("component {name} <{ty}> not found")),
        }
    }

    pub fn qlstm(&self, name: &str) -> Result<QLstmParams, Error> {
        let mut c = Cur { b: &self.bytes, p: self.locate(name, "dynamic_quant_int_8_lstm")? };
        let out = c.i32m()?;
        let input = c.i32m()?;
        let cell = c.i32m()?;
        let _ = c.i32m()?;
        let rows = cell.checked_mul(4).ok_or_else(|| Error::Format("cell too large".into()))?;
        if out != cell || input == 0 || cell == 0 {
            return fmt_err(format!("{name}: bad dims out {out} in {input} cell {cell}"));
        }
        let wx = c.qblock(rows, input)?;
        let bx = c.fv(rows)?;
        let wh = c.qblock(rows, cell)?;
        let bh = c.fv(rows)?;
        Ok(QLstmParams { input, cell, wx, bx, wh, bh })
    }

    pub fn affine(&self, name: &str) -> Result<AffineParams, Error> {
        let start = self.locate(name, "affinetransform")?;
        let Some(p) = find(&self.bytes, b"FM ", start) else {
            return fmt_err(format!("{name}: FM block not found"));
        };
        let mut c = Cur { b: &self.bytes, p: p + 3 };
        let (rows, cols) = (c.i32m()?, c.i32m()?);
        let Some(n) = rows.checked_mul(cols) else { return fmt_err("length overflow") };
        let w = c.f32s(n)?;
        let b = c.fv(rows)?;
        Ok(AffineParams { rows, cols, w, b })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pryon_testkit::{write_mlp, Rng};

    #[test]
    fn roundtrip_synthetic() {
        let m = Mlp::parse(write_mlp((20, 64), 9, 96, 1)).unwrap();
        let l0 = m.qlstm("Hlstm0_layer_0").unwrap();
        assert_eq!((l0.input, l0.cell, l0.wx.rows, l0.wx.cols), (20, 64, 256, 20));
        assert_eq!((l0.wh.rows, l0.wh.cols), (256, 64));
        assert_eq!((l0.bx.len(), l0.bh.len(), l0.wx.rs.len()), (256, 256, 256));
        let l3 = m.qlstm("Hlstm3_layer_0").unwrap();
        assert_eq!((l3.input, l3.cell), (64, 96));
        let r = m.affine("Affine").unwrap();
        assert_eq!((r.rows, r.cols), (12, 9));
        let a9 = m.affine("Affine9").unwrap();
        assert_eq!((a9.rows, a9.cols), (1, 96));

        // Writer draws s, A, B (256 each) then the Wx weights.
        let mut rng = Rng(1);
        for _ in 0..3 * 256 {
            rng.next_u32();
        }
        let draws: Vec<i16> = (0..256 * 20).map(|_| (rng.next_u32() % 255) as i16 - 127).collect();
        assert_eq!(l0.wx.stride8, 24);
        assert_eq!(l0.wx.w.len(), 256 * 24);
        for (row, want) in l0.wx.w.chunks_exact(24).zip(draws.chunks_exact(20)) {
            assert_eq!(&row[..20], want);
            assert!(row[20..].iter().all(|&v| v == 0));
        }
    }

    #[test]
    fn truncated_mlp_errors() {
        let full = write_mlp((20, 64), 9, 96, 1);
        for n in (0..full.len()).step_by(997) {
            let r = Mlp::parse(full[..n].to_vec()).and_then(|m| {
                m.qlstm("Hlstm9_layer_0")?;
                m.affine("Affine")?;
                Ok(())
            });
            assert!(matches!(r, Err(Error::Format(_))), "prefix {n}");
        }
    }

    #[test]
    fn missing_component_errors() {
        let m = Mlp::parse(write_mlp((20, 64), 2, 32, 1)).unwrap();
        assert!(matches!(m.qlstm("Hlstm7_layer_0"), Err(Error::Format(_))));
        assert!(matches!(m.affine("Nope"), Err(Error::Format(_))));
    }

    #[test]
    fn corrupt_marker_errors() {
        let mut b = write_mlp((20, 64), 1, 32, 1);
        let p = find(&b, b"<dynamic_quant_int_8_lstm> ", 0).unwrap() + 27;
        b[p] = 0;
        let m = Mlp::parse(b).unwrap();
        assert!(matches!(m.qlstm("Hlstm0_layer_0"), Err(Error::Format(_))));
    }
}
