//! Load-time tensors: initializers and folded constants of any supported element type.

use super::proto::{TensorData, TensorProto};

/// Largest tensor (and arena) in elements; real graphs stay far below, untrusted files are cut off here.
pub(crate) const MAX_ELEMS: usize = 1 << 24;
pub(crate) const MAX_RANK: usize = 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DType {
    F32,
    I8,
    U8,
    I16,
    I32,
    I64,
    Bool,
}

impl DType {
    pub fn from_onnx(t: i32) -> Option<DType> {
        Some(match t {
            1 => DType::F32,
            2 => DType::U8,
            3 => DType::I8,
            5 => DType::I16,
            6 => DType::I32,
            7 => DType::I64,
            9 => DType::Bool,
            _ => return None,
        })
    }

    fn size(self) -> usize {
        match self {
            DType::I8 | DType::U8 | DType::Bool => 1,
            DType::I16 => 2,
            DType::F32 | DType::I32 => 4,
            DType::I64 => 8,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Data {
    F32(Vec<f32>),
    I8(Vec<i8>),
    U8(Vec<u8>),
    I16(Vec<i16>),
    I32(Vec<i32>),
    I64(Vec<i64>),
    Bool(Vec<bool>),
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Tensor {
    pub dims: Vec<usize>,
    pub data: Data,
}

/// Element count of `dims`, refusing ranks above `MAX_RANK` and counts above `MAX_ELEMS`.
pub(crate) fn numel(dims: &[usize]) -> Result<usize, String> {
    if dims.len() > MAX_RANK {
        return Err(format!("rank {} (at most {MAX_RANK})", dims.len()));
    }
    dims.iter()
        .try_fold(1usize, |n, &d| n.checked_mul(d))
        .filter(|&n| n <= MAX_ELEMS)
        .ok_or_else(|| format!("tensor of shape {dims:?}: more than {MAX_ELEMS} elements"))
}

/// Converts a slice of primitive numbers to `to` with Rust `as` (float to int truncates toward zero).
macro_rules! convert {
    ($v:expr, $to:expr, $zero:expr) => {
        match $to {
            DType::F32 => Data::F32($v.iter().map(|&x| x as f32).collect()),
            DType::I8 => Data::I8($v.iter().map(|&x| x as i8).collect()),
            DType::U8 => Data::U8($v.iter().map(|&x| x as u8).collect()),
            DType::I16 => Data::I16($v.iter().map(|&x| x as i16).collect()),
            DType::I32 => Data::I32($v.iter().map(|&x| x as i32).collect()),
            DType::I64 => Data::I64($v.iter().map(|&x| x as i64).collect()),
            DType::Bool => Data::Bool($v.iter().map(|&x| x != $zero).collect()),
        }
    };
}

impl Tensor {
    pub fn i64(dims: Vec<usize>, v: Vec<i64>) -> Tensor {
        Tensor { dims, data: Data::I64(v) }
    }

    pub fn dtype(&self) -> DType {
        match self.data {
            Data::F32(_) => DType::F32,
            Data::I8(_) => DType::I8,
            Data::U8(_) => DType::U8,
            Data::I16(_) => DType::I16,
            Data::I32(_) => DType::I32,
            Data::I64(_) => DType::I64,
            Data::Bool(_) => DType::Bool,
        }
    }

    pub fn as_f32(&self) -> Option<&[f32]> {
        match &self.data {
            Data::F32(v) => Some(v),
            _ => None,
        }
    }

    /// Integer contents widened to i64; a float tensor is an error.
    pub fn to_i64s(&self) -> Result<Vec<i64>, String> {
        Ok(match &self.data {
            Data::I8(v) => v.iter().map(|&x| x.into()).collect(),
            Data::U8(v) => v.iter().map(|&x| x.into()).collect(),
            Data::I16(v) => v.iter().map(|&x| x.into()).collect(),
            Data::I32(v) => v.iter().map(|&x| x.into()).collect(),
            Data::I64(v) => v.clone(),
            Data::Bool(v) => v.iter().map(|&x| x.into()).collect(),
            Data::F32(_) => return Err("expected an integer tensor, got F32".into()),
        })
    }

    pub fn cast(&self, to: DType) -> Tensor {
        let data = match &self.data {
            Data::F32(v) => convert!(v, to, 0.0),
            _ => convert!(self.to_i64s().unwrap_or_default(), to, 0),
        };
        Tensor { dims: self.dims.clone(), data }
    }

    /// Element `i` of the result is `parts[map[i].0][map[i].1]`; all parts must share one type.
    pub fn gather(parts: &[&Tensor], map: &[(usize, usize)], dims: Vec<usize>) -> Result<Tensor, String> {
        macro_rules! pick {
            ($var:ident) => {{
                let mut src = Vec::with_capacity(parts.len());
                for p in parts {
                    match &p.data {
                        Data::$var(v) => src.push(&v[..]),
                        _ => return Err("inputs of different element types".into()),
                    }
                }
                Data::$var(map.iter().map(|&(i, o)| src[i][o]).collect())
            }};
        }
        let data = match parts.first().map(|p| &p.data) {
            Some(Data::F32(_)) => pick!(F32),
            Some(Data::I8(_)) => pick!(I8),
            Some(Data::U8(_)) => pick!(U8),
            Some(Data::I16(_)) => pick!(I16),
            Some(Data::I32(_)) => pick!(I32),
            Some(Data::I64(_)) => pick!(I64),
            Some(Data::Bool(_)) => pick!(Bool),
            None => return Err("no inputs".into()),
        };
        Ok(Tensor { dims, data })
    }

    /// Decodes an initializer whose type is supported and whose shape passed `numel`.
    pub fn from_proto(t: &TensorProto, dtype: DType, dims: Vec<usize>, n: usize) -> Result<Tensor, String> {
        let bad = || format!("initializer {}: data does not match type {dtype:?} and shape {dims:?}", t.name);
        let data = match &t.data {
            TensorData::Raw(b) => {
                if b.len() != n * dtype.size() {
                    return Err(bad());
                }
                match dtype {
                    DType::F32 => Data::F32(b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()),
                    DType::I8 => Data::I8(b.iter().map(|&x| x as i8).collect()),
                    DType::U8 => Data::U8(b.clone()),
                    DType::I16 => Data::I16(b.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]])).collect()),
                    DType::I32 => Data::I32(b.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()),
                    DType::I64 => Data::I64(
                        b.chunks_exact(8).map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])).collect(),
                    ),
                    DType::Bool => Data::Bool(b.iter().map(|&x| x != 0).collect()),
                }
            }
            TensorData::F32(v) if dtype == DType::F32 && v.len() == n => Data::F32(v.clone()),
            TensorData::I64(v) if dtype == DType::I64 && v.len() == n => Data::I64(v.clone()),
            // int32_data carries every integer type narrower than 64 bits
            TensorData::I32(v) if v.len() == n && !matches!(dtype, DType::F32 | DType::I64) => convert!(v, dtype, 0),
            TensorData::Empty if n == 0 => convert!([0i64; 0], dtype, 0),
            _ => return Err(bad()),
        };
        Ok(Tensor { dims, data })
    }
}
