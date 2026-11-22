//! Minimal protobuf reader for the ONNX subset the wake engine runs.

#[derive(Debug, Clone, PartialEq)]
pub struct ModelProto {
    pub ir_version: i64,
    pub opsets: Vec<(String, i64)>,
    pub graph: GraphProto,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct GraphProto {
    pub nodes: Vec<NodeProto>,
    pub initializers: Vec<TensorProto>,
    pub inputs: Vec<ValueInfo>,
    pub outputs: Vec<ValueInfo>,
    pub value_info: Vec<ValueInfo>,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct NodeProto {
    pub name: String,
    pub op_type: String,
    pub domain: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub attrs: Vec<(String, AttrValue)>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum AttrValue {
    Int(i64),
    Float(f32),
    Str(Vec<u8>),
    Ints(Vec<i64>),
    Floats(Vec<f32>),
    Tensor(TensorProto),
    /// Any other attribute type, by its `AttributeProto.type` number.
    Other(i32),
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct TensorProto {
    pub name: String,
    pub dims: Vec<i64>,
    pub data_type: i32,
    pub data: TensorData,
    pub external: bool,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub enum TensorData {
    Raw(Vec<u8>),
    F32(Vec<f32>),
    I32(Vec<i32>),
    I64(Vec<i64>),
    F64(Vec<f64>),
    #[default]
    Empty,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct ValueInfo {
    pub name: String,
    pub elem_type: i32,
    pub dims: Vec<Dim>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Dim {
    Value(i64),
    Param(String),
    Unknown,
}

enum Val<'a> {
    Varint(u64),
    Fixed64(&'a [u8]),
    Len(&'a [u8]),
    Fixed32(&'a [u8]),
}

const WIRE: &str = "wire type mismatch";

impl<'a> Val<'a> {
    fn varint(&self) -> Result<u64, String> {
        match self {
            Val::Varint(v) => Ok(*v),
            _ => Err(WIRE.into()),
        }
    }
    fn len(&self) -> Result<&'a [u8], String> {
        match self {
            Val::Len(b) => Ok(b),
            _ => Err(WIRE.into()),
        }
    }
    fn string(&self) -> Result<String, String> {
        String::from_utf8(self.len()?.to_vec()).map_err(|_| "invalid utf-8 string".to_string())
    }
    fn f32(&self) -> Result<f32, String> {
        match self {
            Val::Fixed32(b) => Ok(f32::from_le_bytes([b[0], b[1], b[2], b[3]])),
            _ => Err(WIRE.into()),
        }
    }
}

fn varint(b: &[u8], p: &mut usize) -> Result<u64, String> {
    let mut v = 0u64;
    for i in 0..10 {
        let Some(&c) = b.get(*p) else {
            return Err("truncated varint".into());
        };
        *p += 1;
        v |= u64::from(c & 0x7f) << (7 * i);
        if c & 0x80 == 0 {
            return Ok(v);
        }
    }
    Err("varint longer than 10 bytes".into())
}

fn take<'a>(b: &'a [u8], p: &mut usize, n: usize) -> Result<&'a [u8], String> {
    let end = p.checked_add(n).filter(|&e| e <= b.len()).ok_or("length past end of buffer")?;
    let s = &b[*p..end];
    *p = end;
    Ok(s)
}

/// Calls `f(field, value)` for every field; unknown fields are the caller's to ignore.
fn fields<'a>(b: &'a [u8], mut f: impl FnMut(u32, Val<'a>) -> Result<(), String>) -> Result<(), String> {
    let mut p = 0;
    while p < b.len() {
        let tag = varint(b, &mut p)?;
        let field = u32::try_from(tag >> 3).map_err(|_| "field number too large")?;
        if field == 0 {
            return Err("field number 0".into());
        }
        let v = match tag & 7 {
            0 => Val::Varint(varint(b, &mut p)?),
            1 => Val::Fixed64(take(b, &mut p, 8)?),
            2 => {
                let n = usize::try_from(varint(b, &mut p)?).map_err(|_| "length past end of buffer")?;
                Val::Len(take(b, &mut p, n)?)
            }
            5 => Val::Fixed32(take(b, &mut p, 4)?),
            3 | 4 => return Err("protobuf groups are not supported".into()),
            w => return Err(format!("invalid wire type {w}")),
        };
        f(field, v)?;
    }
    Ok(())
}

fn push_ints(out: &mut Vec<i64>, v: &Val) -> Result<(), String> {
    match v {
        Val::Varint(x) => out.push(*x as i64),
        Val::Len(b) => {
            let mut p = 0;
            while p < b.len() {
                out.push(varint(b, &mut p)? as i64);
            }
        }
        _ => return Err(WIRE.into()),
    }
    Ok(())
}

fn push_i32s(out: &mut Vec<i32>, v: &Val) -> Result<(), String> {
    let mut t = Vec::new();
    push_ints(&mut t, v)?;
    out.extend(t.into_iter().map(|x| x as i32));
    Ok(())
}

fn push_f32s(out: &mut Vec<f32>, v: &Val) -> Result<(), String> {
    match v {
        Val::Fixed32(_) => out.push(v.f32()?),
        Val::Len(b) if b.len() % 4 == 0 => {
            out.extend(b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])));
        }
        _ => return Err("bad float field".into()),
    }
    Ok(())
}

fn push_f64s(out: &mut Vec<f64>, v: &Val) -> Result<(), String> {
    let conv = |c: &[u8]| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]);
    match v {
        Val::Fixed64(b) => out.push(conv(b)),
        Val::Len(b) if b.len() % 8 == 0 => out.extend(b.chunks_exact(8).map(conv)),
        _ => return Err("bad double field".into()),
    }
    Ok(())
}

pub fn parse_model(bytes: &[u8]) -> Result<ModelProto, String> {
    let (mut ir_version, mut opsets, mut graph) = (0, Vec::new(), None);
    fields(bytes, |f, v| {
        match f {
            1 => ir_version = v.varint()? as i64,
            8 => opsets.push(opset(v.len()?)?),
            7 => graph = Some(parse_graph(v.len()?)?),
            _ => {}
        }
        Ok(())
    })?;
    let graph = graph.ok_or("model has no graph")?;
    Ok(ModelProto { ir_version, opsets, graph })
}

fn opset(b: &[u8]) -> Result<(String, i64), String> {
    let (mut domain, mut version) = (String::new(), 0);
    fields(b, |f, v| {
        match f {
            1 => domain = v.string()?,
            2 => version = v.varint()? as i64,
            _ => {}
        }
        Ok(())
    })?;
    Ok((domain, version))
}

fn parse_graph(b: &[u8]) -> Result<GraphProto, String> {
    let mut g = GraphProto::default();
    fields(b, |f, v| {
        match f {
            1 => g.nodes.push(parse_node(v.len()?)?),
            5 => g.initializers.push(parse_tensor(v.len()?)?),
            11 => g.inputs.push(parse_value_info(v.len()?)?),
            12 => g.outputs.push(parse_value_info(v.len()?)?),
            13 => g.value_info.push(parse_value_info(v.len()?)?),
            _ => {}
        }
        Ok(())
    })?;
    Ok(g)
}

fn parse_node(b: &[u8]) -> Result<NodeProto, String> {
    let mut n = NodeProto::default();
    fields(b, |f, v| {
        match f {
            1 => n.inputs.push(v.string()?),
            2 => n.outputs.push(v.string()?),
            3 => n.name = v.string()?,
            4 => n.op_type = v.string()?,
            5 => n.attrs.push(parse_attr(v.len()?)?),
            7 => n.domain = v.string()?,
            _ => {}
        }
        Ok(())
    })?;
    Ok(n)
}

fn parse_attr(b: &[u8]) -> Result<(String, AttrValue), String> {
    let mut name = String::new();
    let mut ty = 0i32;
    let (mut fl, mut i, mut s, mut t) = (0f32, 0i64, Vec::new(), None);
    let (mut floats, mut ints) = (Vec::new(), Vec::new());
    let mut seen = 0u32; // bit per value field, to infer a missing type
    fields(b, |f, v| {
        match f {
            1 => name = v.string()?,
            2 => {
                fl = v.f32()?;
                seen |= 1 << 1;
            }
            3 => {
                i = v.varint()? as i64;
                seen |= 1 << 2;
            }
            4 => {
                s = v.len()?.to_vec();
                seen |= 1 << 3;
            }
            5 => {
                t = Some(parse_tensor(v.len()?)?);
                seen |= 1 << 4;
            }
            6 => seen |= 1 << 5,
            7 => {
                push_f32s(&mut floats, &v)?;
                seen |= 1 << 6;
            }
            8 => {
                push_ints(&mut ints, &v)?;
                seen |= 1 << 7;
            }
            20 => ty = v.varint()? as i32,
            _ => {}
        }
        Ok(())
    })?;
    if ty == 0 {
        ty = match seen.trailing_zeros() {
            1 => 1,
            2 => 2,
            3 => 3,
            4 => 4,
            5 => 5,
            6 => 6,
            7 => 7,
            _ => 0,
        };
    }
    let val = match ty {
        1 => AttrValue::Float(fl),
        2 => AttrValue::Int(i),
        3 => AttrValue::Str(s),
        4 => AttrValue::Tensor(t.ok_or_else(|| format!("attribute {name}: tensor missing"))?),
        6 => AttrValue::Floats(floats),
        7 => AttrValue::Ints(ints),
        o => AttrValue::Other(o),
    };
    Ok((name, val))
}

fn parse_tensor(b: &[u8]) -> Result<TensorProto, String> {
    let mut t = TensorProto::default();
    let (mut raw, mut f32s, mut i32s, mut i64s, mut f64s) =
        (None, Vec::new(), Vec::new(), Vec::new(), Vec::new());
    fields(b, |f, v| {
        match f {
            1 => push_ints(&mut t.dims, &v)?,
            2 => t.data_type = v.varint()? as i32,
            4 => push_f32s(&mut f32s, &v)?,
            5 => push_i32s(&mut i32s, &v)?,
            7 => push_ints(&mut i64s, &v)?,
            8 => t.name = v.string()?,
            9 => raw = Some(v.len()?.to_vec()),
            10 => push_f64s(&mut f64s, &v)?,
            13 => t.external = true,
            14 => t.external |= v.varint()? == 1,
            _ => {}
        }
        Ok(())
    })?;
    t.data = if let Some(r) = raw {
        TensorData::Raw(r)
    } else if !f32s.is_empty() {
        TensorData::F32(f32s)
    } else if !i32s.is_empty() {
        TensorData::I32(i32s)
    } else if !i64s.is_empty() {
        TensorData::I64(i64s)
    } else if !f64s.is_empty() {
        TensorData::F64(f64s)
    } else {
        TensorData::Empty
    };
    Ok(t)
}

fn parse_value_info(b: &[u8]) -> Result<ValueInfo, String> {
    let mut vi = ValueInfo::default();
    fields(b, |f, v| {
        match f {
            1 => vi.name = v.string()?,
            2 => fields(v.len()?, |f, v| {
                if f == 1 {
                    tensor_type(v.len()?, &mut vi)?;
                }
                Ok(())
            })?,
            _ => {}
        }
        Ok(())
    })?;
    Ok(vi)
}

fn tensor_type(b: &[u8], vi: &mut ValueInfo) -> Result<(), String> {
    fields(b, |f, v| {
        match f {
            1 => vi.elem_type = v.varint()? as i32,
            2 => fields(v.len()?, |f, v| {
                if f == 1 {
                    vi.dims.push(dim(v.len()?)?);
                }
                Ok(())
            })?,
            _ => {}
        }
        Ok(())
    })
}

fn dim(b: &[u8]) -> Result<Dim, String> {
    let mut d = Dim::Unknown;
    fields(b, |f, v| {
        match f {
            1 => d = Dim::Value(v.varint()? as i64),
            2 => d = Dim::Param(v.string()?),
            _ => {}
        }
        Ok(())
    })?;
    Ok(d)
}

#[cfg(test)]
mod tests {
    use super::*;
    use pryon_testkit::onnx::{Attr, Dim as KDim, ModelBuilder};

    fn sample() -> Vec<u8> {
        ModelBuilder::new(8, 15)
            .input("x", 1, &[KDim::Param("N".into()), KDim::Value(4), KDim::Unknown])
            .output("y", 1, &[KDim::Value(2)])
            .init_f32("w", &[2, 2], &[1.0, -2.5, 3.0, 0.5])
            .init_f32_data("wf", &[2], &[7.0, 8.0])
            .init_f64("wd", &[1], &[0.25])
            .init_i8("q", &[3], &[-1, 0, 127])
            .init_i32("i32", &[2], &[-5, 6])
            .init_i64("i64", &[2], &[-7, 1 << 40])
            .node(
                "Op",
                &["x", "w"],
                &["y"],
                &[
                    ("a", Attr::Int(-3)),
                    ("b", Attr::Float(1.5)),
                    ("c", Attr::Str("hi".into())),
                    ("d", Attr::Ints(vec![1, -2, 300])),
                    ("e", Attr::Floats(vec![0.5, 2.0])),
                    ("f", Attr::Tensor(vec![2], vec![1.0, 2.0])),
                    ("g", Attr::Graph),
                ],
            )
            .build()
    }

    #[test]
    fn roundtrip_builder() {
        let m = parse_model(&sample()).unwrap();
        assert_eq!((m.ir_version, m.opsets.clone()), (8, vec![(String::new(), 15)]));
        let g = &m.graph;
        assert_eq!(g.inputs[0].name, "x");
        assert_eq!(g.inputs[0].elem_type, 1);
        assert_eq!(g.inputs[0].dims, vec![Dim::Param("N".into()), Dim::Value(4), Dim::Unknown]);
        assert_eq!(g.outputs[0].dims, vec![Dim::Value(2)]);
        let raw = |v: &[f32]| TensorData::Raw(v.iter().flat_map(|x| x.to_le_bytes()).collect());
        let ini = |i: usize| (&g.initializers[i].name[..], g.initializers[i].data_type, &g.initializers[i].dims, &g.initializers[i].data);
        assert_eq!(ini(0), ("w", 1, &vec![2, 2], &raw(&[1.0, -2.5, 3.0, 0.5])));
        assert_eq!(ini(1), ("wf", 1, &vec![2], &TensorData::F32(vec![7.0, 8.0])));
        assert_eq!(ini(2), ("wd", 11, &vec![1], &TensorData::F64(vec![0.25])));
        assert_eq!(ini(3), ("q", 3, &vec![3], &TensorData::Raw(vec![0xff, 0, 127])));
        assert_eq!(ini(4), ("i32", 6, &vec![2], &TensorData::I32(vec![-5, 6])));
        assert_eq!(ini(5), ("i64", 7, &vec![2], &TensorData::I64(vec![-7, 1 << 40])));
        assert!(g.initializers.iter().all(|t| !t.external));
        let n = &g.nodes[0];
        assert_eq!((n.op_type.as_str(), n.domain.as_str()), ("Op", ""));
        assert_eq!(n.inputs, ["x", "w"]);
        assert_eq!(n.outputs, ["y"]);
        let a = |i: usize| (&n.attrs[i].0[..], &n.attrs[i].1);
        assert_eq!(a(0), ("a", &AttrValue::Int(-3)));
        assert_eq!(a(1), ("b", &AttrValue::Float(1.5)));
        assert_eq!(a(2), ("c", &AttrValue::Str(b"hi".to_vec())));
        assert_eq!(a(3), ("d", &AttrValue::Ints(vec![1, -2, 300])));
        assert_eq!(a(4), ("e", &AttrValue::Floats(vec![0.5, 2.0])));
        let AttrValue::Tensor(t) = a(5).1 else { panic!("tensor attr") };
        assert_eq!((&t.dims, &t.data), (&vec![2], &raw(&[1.0, 2.0])));
        assert_eq!(a(6), ("g", &AttrValue::Other(5)));
    }

    #[test]
    fn truncated_errors() {
        let b = sample();
        for n in (0..b.len()).step_by(13) {
            assert!(parse_model(&b[..n]).is_err(), "prefix {n} of {}", b.len());
        }
    }

    #[test]
    fn unknown_fields_skipped() {
        // field 99 as varint, 64-bit, length-delimited and 32-bit, at model and graph level
        let junk: Vec<u8> = [&[0x98, 0x06, 5][..], &[0x99, 0x06, 1, 2, 3, 4, 5, 6, 7, 8], &[0x9a, 0x06, 2, 9, 9], &[0x9d, 0x06, 1, 2, 3, 4]].concat();
        let mut g = junk.clone();
        g.extend([0x0a, 4, 0x22, 2, b'O', b'p']); // node { op_type = "Op" }
        let mut m = junk;
        m.extend([0x08, 11]);
        m.extend([0x3a, g.len() as u8]);
        m.extend(&g);
        let p = parse_model(&m).unwrap();
        assert_eq!(p.ir_version, 11);
        assert_eq!(p.graph.nodes[0].op_type, "Op");
    }

    #[test]
    fn packed_and_unpacked_ints() {
        let mut out = Vec::new();
        push_ints(&mut out, &Val::Len(&[1, 0xac, 0x02, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 1])).unwrap();
        push_ints(&mut out, &Val::Varint(7)).unwrap();
        assert_eq!(out, vec![1, 300, -1, 7]);
        // dims unpacked and packed inside a tensor
        let un = parse_tensor(&[0x08, 3, 0x08, 4]).unwrap();
        let pk = parse_tensor(&[0x0a, 2, 3, 4]).unwrap();
        assert_eq!(un.dims, vec![3, 4]);
        assert_eq!(pk.dims, un.dims);
        // unpacked float_data
        let t = parse_tensor(&[0x25, 0, 0, 0x80, 0x3f, 0x25, 0, 0, 0, 0x40]).unwrap();
        assert_eq!(t.data, TensorData::F32(vec![1.0, 2.0]));
    }

    #[test]
    fn malformed_rejected() {
        let g = |b: &[u8]| {
            let mut m = vec![0x3a, b.len() as u8];
            m.extend(b);
            parse_model(&m)
        };
        assert!(g(&[0x0b]).is_err(), "group start");
        assert!(g(&[0x0c]).is_err(), "group end");
        assert!(g(&[0x0a, 50, 1]).is_err(), "length past end");
        let mut long = vec![0x38];
        long.extend([0xff; 10]);
        long.push(1);
        assert!(parse_model(&long).is_err(), "varint over 10 bytes");
        assert!(parse_model(&[0x08, 1]).is_err(), "no graph");
        assert!(g(&[]).is_ok());
    }
}
