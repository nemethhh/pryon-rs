//! HybridGemm/Conv/LSTM against pypryon's CPUNK rewrite run in onnxruntime (`local/hybrid_ref.py` writes `.out.f32`).

use pryon::onnx::{Graph, Session};
use pryon::Error;
use pryon_testkit::onnx::{Attr, Dim, ModelBuilder};
use pryon_testkit::{Rng, TempDir};
use std::sync::Arc;

const F: i32 = 1;
const DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/hybrid");
const CASES: [&str; 7] = [
    "gemm_k35_n96",
    "gemm_bias_transb0",
    "conv_k7x5_s1",
    "conv_stride2_bias",
    "lstm_h20_16lanes_t3",
    "lstm_h100_25lanes_t1",
    "gemm_constant_row",
];

fn d(v: &[usize]) -> Vec<Dim> {
    v.iter().map(|&x| Dim::Value(x as i64)).collect()
}

fn i64s(v: &[usize]) -> Vec<i64> {
    v.iter().map(|&x| x as i64).collect()
}

struct R(Rng);

impl R {
    fn new(name: &str) -> R {
        // seed: FNV-1a of the case name
        R(Rng(name.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3))))
    }
    /// Uniform in [-1, 1).
    fn unit(&mut self) -> f32 {
        (self.0.next_u32() >> 8) as f32 / (1 << 23) as f32 - 1.0
    }
    fn i8s(&mut self, n: usize) -> Vec<i8> {
        (0..n).map(|_| self.0.next_u32() as u8 as i8).collect()
    }
    fn range(&mut self, lo: f32, hi: f32, n: usize) -> Vec<f32> {
        (0..n).map(|_| lo + (hi - lo) * (self.unit() + 1.0) * 0.5).collect()
    }
    fn ints(&mut self, lo: i32, hi: i32, n: usize) -> Vec<i32> {
        (0..n).map(|_| lo + (self.0.next_u32() % (hi - lo + 1) as u32) as i32).collect()
    }
    /// `rows` activation rows of `k` values, each with its own offset and spread (some wholly negative or positive).
    fn rows(&mut self, rows: usize, k: usize) -> Vec<f32> {
        let mut v = Vec::with_capacity(rows * k);
        for _ in 0..rows {
            let (off, spread) = (3.0 * self.unit(), 0.01 + 2.0 * (self.unit() + 1.0));
            v.extend((0..k).map(|_| off + spread * self.unit()));
        }
        v
    }
}

/// Quantised weights `[n, k]` with scales, zero points and row sums, named `p_w`, `p_sw`, `p_zw`, `p_sums`.
fn qweights(m: ModelBuilder, r: &mut R, p: &str, lead: &[usize], n: usize, k: usize, scale: f32) -> ModelBuilder {
    let w = r.i8s(n * k);
    let sums: Vec<i32> = w.chunks(k).map(|row| row.iter().map(|&x| i32::from(x)).sum()).collect();
    let (sw, zw) = (r.range(0.2 * scale, scale, n), r.ints(-100, 100, n));
    let nd: Vec<i64> = lead.iter().map(|&x| x as i64).chain([n as i64]).collect();
    m.init_i8(&format!("{p}_w"), &[&nd[..], &[k as i64]].concat(), &w)
        .init_f32(&format!("{p}_sw"), &nd, &sw)
        .init_i32(&format!("{p}_zw"), &nd, &zw)
        .init_i32(&format!("{p}_sums"), &nd, &sums)
}

/// A fixture graph with its runtime inputs (name, shape, values) in declaration order and its output names.
struct Case {
    model: ModelBuilder,
    inputs: Vec<(&'static str, Vec<usize>, Vec<f32>)>,
    outputs: Vec<&'static str>,
}

fn gemm_case(name: &str, xd: &[usize], n: usize, transb0: bool, bias: bool, scale: f32, x: Option<Vec<f32>>) -> Case {
    let mut r = R::new(name);
    let k = *xd.last().unwrap();
    let rows = xd.iter().product::<usize>() / k;
    let x = x.unwrap_or_else(|| r.rows(rows, k));
    let yd: Vec<usize> = xd[..xd.len() - 1].iter().copied().chain([n]).collect();
    let mut m = ModelBuilder::new(6, 11).input("x", F, &d(xd)).output("y", F, &d(&yd));
    let mut attrs = Vec::new();
    if transb0 {
        let w = r.i8s(n * k);
        let wt: Vec<i8> = (0..k * n).map(|i| w[(i % n) * k + i / n]).collect();
        let sums: Vec<i32> = w.chunks(k).map(|row| row.iter().map(|&x| i32::from(x)).sum()).collect();
        let (sw, zw) = (r.range(0.2 * scale, scale, n), r.ints(-100, 100, n));
        m = m
            .init_i8("g_w", &[k as i64, n as i64], &wt)
            .init_f32("g_sw", &[n as i64], &sw)
            .init_i32("g_zw", &[n as i64], &zw)
            .init_i32("g_sums", &[n as i64], &sums);
        attrs.push(("transB", Attr::Int(0)));
    } else {
        m = qweights(m, &mut r, "g", &[], n, k, scale);
    }
    let mut ins = vec!["x", "g_w", "g_sw", "g_zw", "g_sums"];
    if bias {
        m = m.init_f32("g_b", &[1, n as i64], &r.range(-2.0, 2.0, n));
        ins.push("g_b");
    }
    let model = m.node_named("gemm", "HybridGemm", &ins, &["y"], &attrs);
    Case { model, inputs: vec![("x", xd.to_vec(), x)], outputs: vec!["y"] }
}

fn conv_case(name: &str, xd: [usize; 4], n: usize, kernel: [usize; 2], strides: [usize; 2], bias: bool) -> Case {
    let mut r = R::new(name);
    let [b, c, h, w] = xd;
    let k = c * kernel[0] * kernel[1];
    let yd = [b, n, (h - kernel[0]) / strides[0] + 1, (w - kernel[1]) / strides[1] + 1];
    let x: Vec<f32> = r.rows(b * c * h, w);
    let m =
        qweights(ModelBuilder::new(6, 11).input("x", F, &d(&xd)).output("y", F, &d(&yd)), &mut r, "c", &[], n, k, 0.01);
    let bv = if bias { r.range(-1.0, 1.0, n) } else { vec![0.0; n] };
    let model = m.init_f32("c_b", &[n as i64], &bv).node_named(
        "conv",
        "HybridConv",
        &["x", "c_w", "c_sw", "c_zw", "c_sums", "c_b"],
        &["y"],
        &[
            ("data_format", Attr::Str("NCHW".into())),
            ("dilations", Attr::Ints(vec![1, 1])),
            ("strides", Attr::Ints(i64s(&strides))),
            ("kernel_shape", Attr::Ints(i64s(&kernel))),
            ("group", Attr::Int(1)),
        ],
    );
    Case { model, inputs: vec![("x", xd.to_vec(), x)], outputs: vec!["y"] }
}

fn lstm_case(name: &str, t: usize, xl: usize, n_in: usize, hid: usize, lanes: usize, firmware_attrs: bool) -> Case {
    let mut r = R::new(name);
    let g = 4 * hid;
    let m = ModelBuilder::new(8, 15)
        .input("x", F, &d(&[t, xl, n_in]))
        .input("h0", F, &d(&[1, lanes, hid]))
        .input("c0", F, &d(&[1, lanes, hid]))
        .output("y", F, &d(&[t, 1, lanes, hid]))
        .output("yh", F, &d(&[1, lanes, hid]))
        .output("yc", F, &d(&[1, lanes, hid]));
    let m = qweights(m, &mut r, "w", &[1], g, n_in, 0.02);
    let m = qweights(m, &mut r, "r", &[1], g, hid, 0.02);
    let m = m.init_f32("b", &[1, 2 * g as i64], &r.range(-0.5, 0.5, 2 * g));
    let mut attrs = vec![("hidden_size", Attr::Int(hid as i64))];
    if firmware_attrs {
        attrs.push(("activations", Attr::Strs(vec!["Sigmoid".into(), "Tanh".into(), "Tanh".into()])));
        attrs.push(("direction", Attr::Str("forward".into())));
    }
    let model = m.node_named(
        "lstm",
        "HybridLSTM",
        &["x", "w_w", "w_sw", "w_zw", "w_sums", "r_w", "r_sw", "r_zw", "r_sums", "b", "", "h0", "c0"],
        &["y", "yh", "yc"],
        &attrs,
    );
    let x = r.range(-2.0, 2.0, t * xl * n_in);
    let (h0, c0) = (r.range(-1.0, 1.0, lanes * hid), r.range(-2.0, 2.0, lanes * hid));
    Case {
        model,
        inputs: vec![("x", vec![t, xl, n_in], x), ("h0", vec![1, lanes, hid], h0), ("c0", vec![1, lanes, hid], c0)],
        outputs: vec!["y", "yh", "yc"],
    }
}

fn case(name: &str) -> Case {
    match name {
        "gemm_k35_n96" => gemm_case(name, &[2, 6, 35], 96, false, false, 0.01, None),
        "gemm_bias_transb0" => gemm_case(name, &[5, 48], 24, true, true, 0.01, None),
        "conv_k7x5_s1" => conv_case(name, [1, 1, 12, 64], 24, [7, 5], [1, 1], false),
        "conv_stride2_bias" => conv_case(name, [1, 4, 11, 9], 8, [3, 3], [2, 2], true),
        // x carries one row per lane (DAVS tiles the embedding)
        "lstm_h20_16lanes_t3" => lstm_case(name, 3, 16, 100, 20, 16, false),
        // one x row broadcast to every lane (firmware)
        "lstm_h100_25lanes_t1" => lstm_case(name, 1, 1, 100, 100, 25, true),
        "gemm_constant_row" => {
            let k = 35;
            let mut x = Vec::new();
            for v in [-20.5f32, 0.0, 0.1, -7.25] {
                x.extend(std::iter::repeat_n(v, k));
            }
            x.extend((0..k).map(|i| i as f32 * 0.3 - 4.0));
            // small scales keep |y| small, so the bound in constant_row_path is a few ulps
            gemm_case(name, &[5, k], 16, false, false, 2e-5, Some(x))
        }
        _ => panic!("unknown case {name}"),
    }
}

fn f32s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Regenerates `<case>.onnx` and `<case>.in.f32`; then run `local/hybrid_ref.py` for `.out.f32`.
#[test]
#[ignore]
fn write_fixtures() {
    std::fs::create_dir_all(DIR).unwrap();
    for name in CASES {
        let c = case(name);
        std::fs::write(format!("{DIR}/{name}.onnx"), c.model.build()).unwrap();
        let x: Vec<f32> = c.inputs.iter().flat_map(|i| i.2.iter().copied()).collect();
        std::fs::write(format!("{DIR}/{name}.in.f32"), bytes(&x)).unwrap();
    }
}

/// Runs the committed fixture once; returns our outputs and the reference, both concatenated in output order.
fn run_case(name: &str) -> (Vec<f32>, Vec<f32>) {
    let c = case(name);
    let path = format!("{DIR}/{name}.onnx");
    assert_eq!(std::fs::read(&path).unwrap(), c.model.build(), "{name}.onnx is stale: run write_fixtures");
    let bind: Vec<(&str, &[usize])> = c.inputs.iter().map(|i| (i.0, &i.1[..])).collect();
    let g = Arc::new(Graph::load(path.as_ref(), &bind, &[]).unwrap_or_else(|e| panic!("{name}: {e}")));
    let mut s = Session::new(g.clone());
    for (n, _, v) in &c.inputs {
        s.input_mut(g.input_index(n).unwrap()).copy_from_slice(v);
    }
    s.run();
    let got: Vec<f32> = c.outputs.iter().flat_map(|n| s.output(g.output_index(n).unwrap()).to_vec()).collect();
    let want = f32s(&std::fs::read(format!("{DIR}/{name}.out.f32")).unwrap());
    assert_eq!(got.len(), want.len(), "{name} output length");
    (got, want)
}

fn assert_bits(name: &str) {
    let (got, want) = run_case(name);
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert_eq!(g.to_bits(), w.to_bits(), "{name}[{i}]: {g:e} vs {w:e}");
    }
}

fn assert_close(name: &str, tol: f32) {
    let (got, want) = run_case(name);
    let mut worst = 0f32;
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert!(g.is_finite(), "{name}[{i}] = {g}");
        let e = (g - w).abs();
        assert!(e <= tol, "{name}[{i}]: {g:e} vs {w:e}");
        worst = worst.max(e);
    }
    println!("{name}: max |Δ| {worst:e}");
}

#[test]
fn gemm_matches_ref() {
    assert_bits("gemm_k35_n96");
}

#[test]
fn gemm_bias_transb() {
    assert_bits("gemm_bias_transb0");
}

#[test]
fn conv_matches_ref() {
    assert_bits("conv_k7x5_s1");
    assert_bits("conv_stride2_bias");
}

// pypryon's LSTM and constant-row contracts approximate libpryon's: a bound here, the bits in capture/unit tests
#[test]
fn lstm_matches_ref() {
    assert_close("lstm_h20_16lanes_t3", 1e-6);
    assert_close("lstm_h100_25lanes_t1", 5e-7);
}

#[test]
fn constant_row_path() {
    assert_close("gemm_constant_row", 5e-8);
}

fn load(m: ModelBuilder, bind: &[(&str, &[usize])]) -> Result<Graph, Error> {
    let dir = TempDir::new("hybrid");
    let p = dir.join("m.onnx");
    std::fs::write(&p, m.build()).unwrap();
    Graph::load(&p, bind, &[])
}

fn unsupported(r: Result<Graph, Error>) -> String {
    match r {
        Err(e @ Error::Unsupported { .. }) => e.to_string(),
        Err(e) => panic!("expected Unsupported, got {e}"),
        Ok(_) => panic!("loaded"),
    }
}

#[test]
fn q_saturates_127() {
    // s = 1, z = rint(63.5) = 64: 63.5 quantises to 64 + 64 = 128, held at 127, so q = 63; −63.5 gives q = −64
    let m = ModelBuilder::new(6, 11)
        .input("x", F, &d(&[1, 4]))
        .output("y", F, &d(&[1, 2]))
        .init_i8("w", &[2, 4], &[1, 0, 0, 0, 0, 1, 0, 0])
        .init_f32("sw", &[2], &[1.0, 1.0])
        .init_i32("zw", &[2], &[0, 0])
        .init_i32("sums", &[2], &[1, 1])
        .node("HybridGemm", &["x", "w", "sw", "zw", "sums"], &["y"], &[]);
    let g = Arc::new(load(m, &[("x", &[1, 4])]).unwrap());
    let mut s = Session::new(g.clone());
    s.input_mut(0).copy_from_slice(&[-63.5, 63.5, 0.0, 0.0]);
    s.run();
    assert_eq!(s.output(0), [-64.0, 63.0]);
}

#[test]
fn conv_dilation_refused() {
    let m = ModelBuilder::new(6, 11)
        .input("x", F, &d(&[1, 1, 4, 4]))
        .output("y", F, &d(&[1, 1, 2, 3]))
        .init_i8("w", &[1, 2], &[1, 1])
        .init_f32("sw", &[1], &[1.0])
        .init_i32("zw", &[1], &[0])
        .init_i32("sums", &[1], &[2])
        .init_f32("b", &[1], &[0.0])
        .node_named(
            "c",
            "HybridConv",
            &["x", "w", "sw", "zw", "sums", "b"],
            &["y"],
            &[("kernel_shape", Attr::Ints(vec![2, 1])), ("dilations", Attr::Ints(vec![2, 1]))],
        );
    let e = unsupported(load(m, &[("x", &[1, 1, 4, 4])]));
    assert!(e.contains("HybridConv (node c)") && e.contains("dilations"), "{e}");
}

#[test]
fn conv_missing_bias_refused() {
    let m = ModelBuilder::new(6, 11)
        .input("x", F, &d(&[1, 1, 4, 4]))
        .output("y", F, &d(&[1, 1, 3, 4]))
        .init_i8("w", &[1, 2], &[1, 1])
        .init_f32("sw", &[1], &[1.0])
        .init_i32("zw", &[1], &[0])
        .init_i32("sums", &[1], &[2])
        .node_named(
            "c",
            "HybridConv",
            &["x", "w", "sw", "zw", "sums"],
            &["y"],
            &[("kernel_shape", Attr::Ints(vec![2, 1]))],
        );
    let e = unsupported(load(m, &[("x", &[1, 1, 4, 4])]));
    assert!(e.contains("HybridConv (node c)") && e.contains("bias"), "{e}");
}

#[test]
fn lstm_reverse_refused() {
    let m = ModelBuilder::new(8, 15)
        .input("x", F, &d(&[1, 1, 8]))
        .input("h0", F, &d(&[1, 2, 4]))
        .input("c0", F, &d(&[1, 2, 4]))
        .output("y", F, &d(&[1, 1, 2, 4]));
    let mut r = R::new("reverse");
    let m = qweights(m, &mut r, "w", &[1], 16, 8, 0.02);
    let m = qweights(m, &mut r, "r", &[1], 16, 4, 0.02).init_f32("b", &[1, 32], &[0.0; 32]).node_named(
        "l",
        "HybridLSTM",
        &["x", "w_w", "w_sw", "w_zw", "w_sums", "r_w", "r_sw", "r_zw", "r_sums", "b", "", "h0", "c0"],
        &["y"],
        &[("hidden_size", Attr::Int(4)), ("direction", Attr::Str("reverse".into()))],
    );
    let e = unsupported(load(m, &[("x", &[1, 1, 8]), ("h0", &[1, 2, 4]), ("c0", &[1, 2, 4])]));
    assert!(e.contains("HybridLSTM (node l)") && e.contains("direction"), "{e}");
}

#[test]
fn lstm_h0_mismatch_without_c0_refused() {
    let m = ModelBuilder::new(8, 15).input("x", F, &d(&[1, 1, 8])).input("h0", F, &d(&[1, 2, 3])).output(
        "y",
        F,
        &d(&[1, 1, 2, 4]),
    );
    let mut r = R::new("h0");
    let m = qweights(m, &mut r, "w", &[1], 16, 8, 0.02);
    let m = qweights(m, &mut r, "r", &[1], 16, 4, 0.02).init_f32("b", &[1, 32], &[0.0; 32]).node_named(
        "l",
        "HybridLSTM",
        &["x", "w_w", "w_sw", "w_zw", "w_sums", "r_w", "r_sw", "r_zw", "r_sums", "b", "", "h0"],
        &["y"],
        &[("hidden_size", Attr::Int(4))],
    );
    let e = unsupported(load(m, &[("x", &[1, 1, 8]), ("h0", &[1, 2, 3])]));
    assert!(e.contains("HybridLSTM (node l)") && e.contains("input 12"), "{e}");
}
