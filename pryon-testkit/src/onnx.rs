//! Dependency-free ONNX protobuf writer for synthetic graphs.

#[derive(Debug, Clone, PartialEq)]
pub enum Dim {
    Value(i64),
    Param(String),
    Unknown,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Attr {
    Int(i64),
    Ints(Vec<i64>),
    Float(f32),
    Floats(Vec<f32>),
    Str(String),
    /// f32 tensor: dims and raw-encoded data.
    Tensor(Vec<i64>, Vec<f32>),
    /// A subgraph (empty); readers report it as an unsupported attribute type.
    Graph,
}

fn put_varint(o: &mut Vec<u8>, mut v: u64) {
    while v >= 0x80 {
        o.push((v as u8) | 0x80);
        v >>= 7;
    }
    o.push(v as u8);
}

fn put_tag(o: &mut Vec<u8>, field: u32, wire: u8) {
    put_varint(o, (u64::from(field) << 3) | u64::from(wire));
}

fn put_int(o: &mut Vec<u8>, field: u32, v: i64) {
    put_tag(o, field, 0);
    put_varint(o, v as u64);
}

fn put_bytes(o: &mut Vec<u8>, field: u32, b: &[u8]) {
    put_tag(o, field, 2);
    put_varint(o, b.len() as u64);
    o.extend_from_slice(b);
}

fn put_packed(o: &mut Vec<u8>, field: u32, vals: impl Iterator<Item = i64>) {
    let mut b = Vec::new();
    for v in vals {
        put_varint(&mut b, v as u64);
    }
    put_bytes(o, field, &b);
}

fn tensor(name: &str, dtype: i32, dims: &[i64], payload: impl FnOnce(&mut Vec<u8>)) -> Vec<u8> {
    let mut t = Vec::new();
    put_packed(&mut t, 1, dims.iter().copied());
    put_int(&mut t, 2, i64::from(dtype));
    payload(&mut t);
    if !name.is_empty() {
        put_bytes(&mut t, 8, name.as_bytes());
    }
    t
}

fn f32_raw(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn value_info(name: &str, elem: i32, dims: &[Dim]) -> Vec<u8> {
    let mut shape = Vec::new();
    for d in dims {
        let mut b = Vec::new();
        match d {
            Dim::Value(v) => put_int(&mut b, 1, *v),
            Dim::Param(p) => put_bytes(&mut b, 2, p.as_bytes()),
            Dim::Unknown => {}
        }
        put_bytes(&mut shape, 1, &b);
    }
    let mut tt = Vec::new();
    put_int(&mut tt, 1, i64::from(elem));
    put_bytes(&mut tt, 2, &shape);
    let mut ty = Vec::new();
    put_bytes(&mut ty, 1, &tt);
    let mut vi = Vec::new();
    put_bytes(&mut vi, 1, name.as_bytes());
    put_bytes(&mut vi, 2, &ty);
    vi
}

fn attribute(name: &str, a: &Attr) -> Vec<u8> {
    let mut b = Vec::new();
    put_bytes(&mut b, 1, name.as_bytes());
    let ty = match a {
        Attr::Float(f) => {
            put_tag(&mut b, 2, 5);
            b.extend_from_slice(&f.to_le_bytes());
            1
        }
        Attr::Int(i) => {
            put_int(&mut b, 3, *i);
            2
        }
        Attr::Str(s) => {
            put_bytes(&mut b, 4, s.as_bytes());
            3
        }
        Attr::Tensor(d, v) => {
            put_bytes(&mut b, 5, &tensor("", 1, d, |t| put_bytes(t, 9, &f32_raw(v))));
            4
        }
        Attr::Graph => {
            put_bytes(&mut b, 6, &[]);
            5
        }
        Attr::Floats(v) => {
            put_bytes(&mut b, 7, &f32_raw(v));
            6
        }
        Attr::Ints(v) => {
            put_packed(&mut b, 8, v.iter().copied());
            7
        }
    };
    put_int(&mut b, 20, ty);
    b
}

pub struct ModelBuilder {
    ir: i64,
    opset: i64,
    graph: Vec<u8>,
}

impl ModelBuilder {
    pub fn new(ir: i64, opset: i64) -> Self {
        Self { ir, opset, graph: Vec::new() }
    }

    pub fn input(mut self, name: &str, elem: i32, dims: &[Dim]) -> Self {
        put_bytes(&mut self.graph, 11, &value_info(name, elem, dims));
        self
    }

    pub fn output(mut self, name: &str, elem: i32, dims: &[Dim]) -> Self {
        put_bytes(&mut self.graph, 12, &value_info(name, elem, dims));
        self
    }

    fn init(mut self, t: Vec<u8>) -> Self {
        put_bytes(&mut self.graph, 5, &t);
        self
    }

    /// Stored as `raw_data`, like the real graphs.
    pub fn init_f32(self, name: &str, dims: &[i64], data: &[f32]) -> Self {
        self.init(tensor(name, 1, dims, |t| put_bytes(t, 9, &f32_raw(data))))
    }

    /// Stored in the `float_data` field.
    pub fn init_f32_data(self, name: &str, dims: &[i64], data: &[f32]) -> Self {
        self.init(tensor(name, 1, dims, |t| put_bytes(t, 4, &f32_raw(data))))
    }

    pub fn init_f64(self, name: &str, dims: &[i64], data: &[f64]) -> Self {
        let raw: Vec<u8> = data.iter().flat_map(|x| x.to_le_bytes()).collect();
        self.init(tensor(name, 11, dims, |t| put_bytes(t, 10, &raw)))
    }

    pub fn init_i8(self, name: &str, dims: &[i64], data: &[i8]) -> Self {
        let raw: Vec<u8> = data.iter().map(|&x| x as u8).collect();
        self.init(tensor(name, 3, dims, |t| put_bytes(t, 9, &raw)))
    }

    pub fn init_i32(self, name: &str, dims: &[i64], data: &[i32]) -> Self {
        self.init(tensor(name, 6, dims, |t| put_packed(t, 5, data.iter().map(|&x| i64::from(x)))))
    }

    pub fn init_i64(self, name: &str, dims: &[i64], data: &[i64]) -> Self {
        self.init(tensor(name, 7, dims, |t| put_packed(t, 7, data.iter().copied())))
    }

    pub fn node(mut self, op: &str, ins: &[&str], outs: &[&str], attrs: &[(&str, Attr)]) -> Self {
        let mut n = Vec::new();
        for i in ins {
            put_bytes(&mut n, 1, i.as_bytes());
        }
        for o in outs {
            put_bytes(&mut n, 2, o.as_bytes());
        }
        put_bytes(&mut n, 4, op.as_bytes());
        for (name, a) in attrs {
            put_bytes(&mut n, 5, &attribute(name, a));
        }
        put_bytes(&mut self.graph, 1, &n);
        self
    }

    pub fn build(self) -> Vec<u8> {
        let mut m = Vec::new();
        put_int(&mut m, 1, self.ir);
        let mut os = Vec::new();
        put_int(&mut os, 2, self.opset);
        put_bytes(&mut m, 8, &os);
        put_bytes(&mut m, 7, &self.graph);
        m
    }
}
