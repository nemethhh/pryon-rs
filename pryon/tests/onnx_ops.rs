use pryon::onnx::{Graph, Session};
use pryon::Error;
use pryon_testkit::onnx::{Attr, Dim, ModelBuilder};
use pryon_testkit::TempDir;
use std::sync::Arc;

const F: i32 = 1;
const I64: i32 = 7;

fn d(v: &[i64]) -> Vec<Dim> {
    v.iter().map(|&x| Dim::Value(x)).collect()
}

fn load(m: ModelBuilder, bind: &[(&str, &[usize])], keep: &[&str]) -> Result<Graph, Error> {
    let dir = TempDir::new("onnx-ops");
    let p = dir.join("m.onnx");
    std::fs::write(&p, m.build()).unwrap();
    Graph::load(&p, bind, keep)
}

fn unsupported(r: Result<Graph, Error>) -> String {
    match r {
        Err(e @ Error::Unsupported { .. }) => e.to_string(),
        Err(e) => panic!("expected Unsupported, got {e}"),
        Ok(_) => panic!("loaded"),
    }
}

/// Feeds `ins` by name, runs once and returns the named outputs plus the traced runtime value names.
fn run(g: Graph, ins: &[(&str, &[f32])], outs: &[&str]) -> (Vec<Vec<f32>>, Vec<String>) {
    let g = Arc::new(g);
    let mut s = Session::new(g.clone());
    for (n, v) in ins {
        s.input_mut(g.input_index(n).unwrap()).copy_from_slice(v);
    }
    let mut traced = Vec::new();
    s.run_traced(&mut |name, _, _| traced.push(name.to_string()));
    let mut again = Session::new(g.clone());
    for (n, v) in ins {
        again.input_mut(g.input_index(n).unwrap()).copy_from_slice(v);
    }
    again.run();
    let got: Vec<Vec<f32>> = outs.iter().map(|n| s.output(g.output_index(n).unwrap()).to_vec()).collect();
    for (n, v) in outs.iter().zip(&got) {
        assert_eq!(bits(again.output(g.output_index(n).unwrap())), bits(v), "run and run_traced differ on {n}");
    }
    (got, traced)
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn shape_of(g: &Graph, out: &str) -> Vec<usize> {
    g.output_shape(g.output_index(out).unwrap()).to_vec()
}

#[test]
fn unknown_op_named() {
    let m = ModelBuilder::new(8, 15).input("x", F, &d(&[2])).output("y", F, &d(&[2])).node_named(
        "e1",
        "Erf",
        &["x"],
        &["y"],
        &[],
    );
    let e = unsupported(load(m, &[("x", &[2])], &[]));
    assert!(e.contains("m.onnx: unsupported operator Erf (node e1)"), "{e}");
}

#[test]
fn unknown_attr_named() {
    let m = ModelBuilder::new(8, 15).input("x", F, &d(&[2])).output("y", F, &d(&[2])).node_named(
        "r",
        "Relu",
        &["x"],
        &["y"],
        &[("alpha", Attr::Float(0.1))],
    );
    let e = unsupported(load(m, &[("x", &[2])], &[]));
    assert!(e.contains("unsupported attribute Relu.alpha (node r)"), "{e}");
}

#[test]
fn external_data_refused() {
    let m = ModelBuilder::new(8, 15)
        .input("x", F, &d(&[2]))
        .output("y", F, &d(&[2]))
        .init_external("w", F, &[2])
        .node("Add", &["x", "w"], &["y"], &[]);
    let e = unsupported(load(m, &[("x", &[2])], &[]));
    assert!(e.contains("external data") && e.contains('w'), "{e}");
}

#[test]
fn int_runtime_refused() {
    let m = ModelBuilder::new(8, 15).input("x", I64, &d(&[2])).output("y", I64, &d(&[2])).node_named(
        "r",
        "Relu",
        &["x"],
        &["y"],
        &[],
    );
    let e = unsupported(load(m, &[("x", &[2])], &[]));
    assert!(e.contains("non-float runtime tensor x at node r"), "{e}");
}

#[test]
fn shape_pipeline_folds() {
    let m = ModelBuilder::new(6, 11)
        .input("x", F, &[Dim::Param("N".into()), Dim::Value(6)])
        .output("y", F, &[Dim::Unknown, Dim::Value(3), Dim::Value(2)])
        .init_i64("i0", &[1], &[0])
        .init_i64("tail", &[2], &[3, -1])
        .node("Shape", &["x"], &["s"], &[("dtype", Attr::Int(7))])
        .node("Gather", &["s", "i0"], &["n"], &[("axis", Attr::Int(0))])
        .node("Cast", &["n"], &["n32"], &[("to", Attr::Int(6))])
        .node("Cast", &["n32"], &["n64"], &[("to", Attr::Int(7))])
        .node("Concat", &["n64", "tail"], &["shape"], &[("axis", Attr::Int(0))])
        .node("Reshape", &["x", "shape"], &["y"], &[]);
    let g = load(m, &[("x", &[2, 6])], &[]).unwrap();
    assert_eq!(shape_of(&g, "y"), [2, 3, 2]);
    let x: Vec<f32> = (0..12).map(|v| v as f32).collect();
    let (out, traced) = run(g, &[("x", &x)], &["y"]);
    assert_eq!(out[0], x);
    assert_eq!(traced, ["y"]);
}

#[test]
fn elementwise_broadcast() {
    let m = ModelBuilder::new(8, 15)
        .input("a", F, &d(&[2, 3]))
        .input("p", F, &d(&[2, 1]))
        .input("q", F, &d(&[1, 3]))
        .input("c", F, &d(&[2, 1, 1]))
        .output("y", F, &d(&[2, 3]))
        .output("z", F, &d(&[2, 3]))
        .output("r", F, &d(&[2, 3]))
        .output("v", F, &d(&[2, 2, 3]))
        .init_f32("b", &[3], &[10.0, 20.0, 30.0])
        .node("Add", &["a", "b"], &["y"], &[])
        .node("Mul", &["p", "q"], &["z"], &[])
        .node("Sub", &["q", "p"], &["qp"], &[])
        .node("Neg", &["qp"], &["n"], &[])
        .node("Relu", &["n"], &["r"], &[])
        .node("Add", &["c", "a"], &["v"], &[]);
    let g = load(m, &[("a", &[2, 3]), ("p", &[2, 1]), ("q", &[1, 3]), ("c", &[2, 1, 1])], &[]).unwrap();
    assert_eq!(shape_of(&g, "v"), [2, 2, 3]);
    let ins: [(&str, &[f32]); 4] = [
        ("a", &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        ("p", &[2.0, 3.0]),
        ("q", &[1.0, 10.0, 100.0]),
        ("c", &[100.0, 200.0]),
    ];
    let (o, _) = run(g, &ins, &["y", "z", "r", "v"]);
    assert_eq!(o[0], [11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
    assert_eq!(o[1], [2.0, 20.0, 200.0, 3.0, 30.0, 300.0]);
    assert_eq!(o[2], [1.0, 0.0, 0.0, 2.0, 0.0, 0.0]);
    assert_eq!(
        o[3],
        [101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 201.0, 202.0, 203.0, 204.0, 205.0, 206.0]
    );
}

#[test]
fn slice_negative_and_steps() {
    let m = ModelBuilder::new(8, 15)
        .input("x", F, &d(&[2, 5]))
        .output("a", F, &d(&[2, 3]))
        .output("b", F, &d(&[2, 3]))
        .init_i64("s1", &[1], &[-1])
        .init_i64("e1", &[1], &[-100])
        .init_i64("ax1", &[1], &[1])
        .init_i64("st1", &[1], &[-2])
        .init_i64("s2", &[2], &[0, 1])
        .init_i64("e2", &[2], &[i64::MAX, -1])
        .init_i64("ax2", &[2], &[0, -1])
        .node("Slice", &["x", "s1", "e1", "ax1", "st1"], &["a"], &[])
        .node("Slice", &["x", "s2", "e2", "ax2"], &["b"], &[]);
    let g = load(m, &[("x", &[2, 5])], &[]).unwrap();
    assert_eq!(shape_of(&g, "a"), [2, 3]);
    assert_eq!(shape_of(&g, "b"), [2, 3]);
    let x: Vec<f32> = (0..10).map(|v| v as f32).collect();
    let (o, _) = run(g, &[("x", &x)], &["a", "b"]);
    assert_eq!(o[0], [4.0, 2.0, 0.0, 9.0, 7.0, 5.0]);
    assert_eq!(o[1], [1.0, 2.0, 3.0, 6.0, 7.0, 8.0]);
}

#[test]
fn transpose_perm() {
    let m = ModelBuilder::new(8, 15)
        .input("x", F, &d(&[2, 3, 4]))
        .input("w", F, &d(&[2, 3]))
        .output("y", F, &d(&[3, 4, 2]))
        .output("t", F, &d(&[3, 2]))
        .node("Transpose", &["x"], &["y"], &[("perm", Attr::Ints(vec![1, 2, 0]))])
        .node("Transpose", &["w"], &["t"], &[]);
    let g = load(m, &[("x", &[2, 3, 4]), ("w", &[2, 3])], &[]).unwrap();
    assert_eq!(shape_of(&g, "y"), [3, 4, 2]);
    let x: Vec<f32> = (0..24).map(|v| v as f32).collect();
    let (o, _) = run(g, &[("x", &x), ("w", &x[..6])], &["y", "t"]);
    let want = [
        0, 12, 1, 13, 2, 14, 3, 15, 4, 16, 5, 17, 6, 18, 7, 19, 8, 20, 9, 21, 10, 22, 11, 23,
    ];
    assert_eq!(o[0], want.map(|v| v as f32));
    assert_eq!(o[1], [0.0, 3.0, 1.0, 4.0, 2.0, 5.0]);
}

#[test]
fn tile_repeats() {
    let m = ModelBuilder::new(8, 15)
        .input("x", F, &d(&[2, 2]))
        .output("y", F, &d(&[4, 6]))
        .init_i64("rep", &[2], &[2, 3])
        .node("Tile", &["x", "rep"], &["y"], &[]);
    let g = load(m, &[("x", &[2, 2])], &[]).unwrap();
    assert_eq!(shape_of(&g, "y"), [4, 6]);
    let (o, _) = run(g, &[("x", &[1.0, 2.0, 3.0, 4.0])], &["y"]);
    let rows = [[1.0, 2.0, 1.0, 2.0, 1.0, 2.0], [3.0, 4.0, 3.0, 4.0, 3.0, 4.0]];
    assert_eq!(o[0], [rows[0], rows[1], rows[0], rows[1]].concat());
}

#[test]
fn squeeze_axes_attr_and_input() {
    let x = [1.0, 2.0, 3.0];
    let m11 = ModelBuilder::new(6, 11)
        .input("x", F, &d(&[1, 3, 1]))
        .output("s", F, &d(&[3]))
        .output("u", F, &d(&[1, 3]))
        .node("Squeeze", &["x"], &["s"], &[("axes", Attr::Ints(vec![0, 2]))])
        .node("Unsqueeze", &["s"], &["u"], &[("axes", Attr::Ints(vec![0]))]);
    let g = load(m11, &[("x", &[1, 3, 1])], &[]).unwrap();
    assert_eq!((shape_of(&g, "s"), shape_of(&g, "u")), (vec![3], vec![1, 3]));
    let (o, _) = run(g, &[("x", &x)], &["s", "u"]);
    assert_eq!((&o[0][..], &o[1][..]), (&x[..], &x[..]));

    let m13 = ModelBuilder::new(8, 13)
        .input("x", F, &d(&[1, 3, 1]))
        .output("s", F, &d(&[1, 3]))
        .output("u", F, &d(&[1, 1, 3, 1]))
        .output("all", F, &d(&[3]))
        .init_i64("sq", &[1], &[-1])
        .init_i64("un", &[2], &[1, -1])
        .node("Squeeze", &["x", "sq"], &["s"], &[])
        .node("Unsqueeze", &["s", "un"], &["u"], &[])
        .node("Squeeze", &["x"], &["all"], &[]);
    let g = load(m13, &[("x", &[1, 3, 1])], &[]).unwrap();
    assert_eq!(shape_of(&g, "s"), [1, 3]);
    assert_eq!(shape_of(&g, "u"), [1, 1, 3, 1]);
    assert_eq!(shape_of(&g, "all"), [3]);
    let (o, _) = run(g, &[("x", &x)], &["u"]);
    assert_eq!(o[0], x);
}

#[test]
fn kept_input_initial() {
    let m = ModelBuilder::new(8, 15)
        .input("x", F, &d(&[2]))
        .output("z", F, &d(&[2]))
        .init_f32("s", &[2], &[0.5, 1.5])
        .init_f32("w", &[2], &[10.0, 20.0])
        .node("Add", &["x", "s"], &["y"], &[])
        .node("Add", &["y", "w"], &["z"], &[]);
    let g = Arc::new(load(m, &[("x", &[2])], &["s"]).unwrap());
    let (xi, si) = (g.input_index("x").unwrap(), g.input_index("s").unwrap());
    assert_eq!(g.input_index("w"), None);
    assert_eq!(g.initial(si), Some(&[0.5, 1.5][..]));
    assert_eq!(g.initial(xi), None);
    assert_eq!(g.input_shape(si), [2]);
    let zi = g.output_index("z").unwrap();
    let mut s = Session::new(g.clone());
    s.input_mut(xi).copy_from_slice(&[1.0, 1.0]);
    s.run();
    assert_eq!(s.output(zi), [11.5, 22.5]);
    s.input_mut(si).copy_from_slice(&[0.0, 0.0]);
    s.run();
    assert_eq!(s.output(zi), [11.0, 21.0]);
}

#[test]
fn bound_shape_conflict() {
    let m = || ModelBuilder::new(8, 15).input("x", F, &d(&[1, 4])).output("y", F, &d(&[1, 4])).node("Relu", &["x"], &["y"], &[]);
    let e = unsupported(load(m(), &[("x", &[1, 5])], &[]));
    assert!(e.contains("input x") && e.contains("[1, 5]"), "{e}");
    let e = unsupported(load(m(), &[("x", &[4])], &[]));
    assert!(e.contains("input x"), "{e}");
    let e = unsupported(load(m(), &[], &[]));
    assert!(e.contains("input x"), "{e}");
    let e = unsupported(load(m(), &[("x", &[1, 4]), ("nope", &[1])], &[]));
    assert!(e.contains("nope"), "{e}");
}

#[test]
fn gather_concat_runtime() {
    let m = ModelBuilder::new(8, 15)
        .input("x", F, &d(&[2, 3]))
        .output("g", F, &d(&[2, 2]))
        .output("c", F, &d(&[2, 4]))
        .init_i64("idx", &[2], &[2, -3])
        .init_f32("k", &[2, 1], &[-1.0, -2.0])
        .node("Gather", &["x", "idx"], &["g"], &[("axis", Attr::Int(1))])
        .node("Cast", &["x"], &["xf"], &[("to", Attr::Int(1))])
        .node("Concat", &["xf", "k"], &["c"], &[("axis", Attr::Int(-1))]);
    let g = load(m, &[("x", &[2, 3])], &[]).unwrap();
    let (o, _) = run(g, &[("x", &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0])], &["g", "c"]);
    assert_eq!(o[0], [2.0, 0.0, 5.0, 3.0]);
    assert_eq!(o[1], [0.0, 1.0, 2.0, -1.0, 3.0, 4.0, 5.0, -2.0]);
}

#[test]
fn constant_output_and_dead_nodes() {
    let m = ModelBuilder::new(8, 15)
        .input("x", F, &d(&[2]))
        .output("k", F, &d(&[2]))
        .output("y", F, &d(&[2]))
        .init_f32("a", &[2], &[1.0, 2.0])
        .init_f32("b", &[2], &[3.0, 5.0])
        .node("Mul", &["a", "b"], &["k"], &[])
        .node("Relu", &["x"], &["y"], &[])
        .node("Erf", &["x"], &["dead"], &[]);
    let g = load(m, &[("x", &[2])], &[]).unwrap();
    let (o, traced) = run(g, &[("x", &[-1.0, 1.0])], &["k", "y"]);
    assert_eq!(o[0], [3.0, 10.0]);
    assert_eq!(o[1], [0.0, 1.0]);
    assert_eq!(traced, ["y"]);
}

#[test]
fn absurd_sizes_refused() {
    let relu = || ModelBuilder::new(8, 15).input("x", F, &[Dim::Unknown, Dim::Unknown]).output("y", F, &[]).node("Relu", &["x"], &["y"], &[]);
    let e = unsupported(load(relu(), &[("x", &[1 << 14, 1 << 14])], &[]));
    assert!(e.contains("elements"), "{e}");
    let e = unsupported(load(relu(), &[("x", &[usize::MAX, 2])], &[]));
    assert!(e.contains("elements"), "{e}");
    let tile = ModelBuilder::new(8, 15)
        .input("x", F, &d(&[2]))
        .output("y", F, &[])
        .init_i64("rep", &[1], &[1 << 40])
        .node("Tile", &["x", "rep"], &["y"], &[]);
    let e = unsupported(load(tile, &[("x", &[2])], &[]));
    assert!(e.contains("elements"), "{e}");
}

#[test]
fn placeholder_state_and_repeated_inputs() {
    // as in the firmware graph: a state input whose initializer has dims but no data, declared as input twice
    let m = || {
        ModelBuilder::new(6, 11)
            .input("x", F, &d(&[2]))
            .input("st", F, &[Dim::Param("B".into()), Dim::Value(2)])
            .input("st", F, &[Dim::Param("B".into()), Dim::Value(2)])
            .output("y", F, &d(&[1, 2]))
            .init_f32_data("st", &[1, 2], &[])
            .init_f32("raw", &[1, 2], &[])
            .node("Add", &["st", "x"], &["s"], &[])
            .node("Add", &["s", "raw"], &["y"], &[])
    };
    let g = load(m(), &[("x", &[2])], &["st", "raw"]).unwrap();
    assert_eq!(g.initial(g.input_index("st").unwrap()), Some(&[0.0, 0.0][..]));
    assert_eq!(g.initial(g.input_index("raw").unwrap()), Some(&[0.0, 0.0][..]));
    let (o, _) = run(g, &[("x", &[1.0, 2.0])], &["y"]);
    assert_eq!(o[0], [1.0, 2.0]);
    let e = load(m(), &[("x", &[2])], &["st"]).unwrap_err().to_string();
    assert!(e.contains("initializer raw has no data"), "{e}");
    let clash = ModelBuilder::new(8, 15)
        .input("x", F, &d(&[2]))
        .input("x", F, &d(&[3]))
        .output("y", F, &d(&[2]))
        .node("Relu", &["x"], &["y"], &[]);
    let e = load(clash, &[("x", &[2])], &[]).unwrap_err().to_string();
    assert!(e.contains("input x declared twice"), "{e}");
}

#[test]
fn fold_budget_refused() {
    // one Tile past the per-tensor fold limit, and a chain of small folds past the total
    let big = ModelBuilder::new(8, 15)
        .output("y", I64, &[])
        .init_i64("one", &[1], &[0])
        .init_i64("rep", &[1], &[1 << 20])
        .node("Tile", &["one", "rep"], &["y"], &[]);
    let e = unsupported(load(big, &[], &[]));
    assert!(e.contains("folded tensor") && e.contains("elements"), "{e}");
    let mut chain = ModelBuilder::new(8, 15).init_i64("one", &[1], &[0]).init_i64("rep", &[1], &[1 << 16]);
    let mut prev = String::new();
    for k in 0..40 {
        let t = format!("t{k}");
        chain = chain.node("Tile", &["one", "rep"], &[&t], &[]);
        if k > 0 {
            let s = format!("s{k}");
            chain = chain.node("Add", &[&prev, &t], &[&s], &[]);
            prev = s;
        } else {
            prev = t;
        }
    }
    let e = unsupported(load(chain.output(&prev, I64, &[]), &[], &[]));
    assert!(e.contains("folded constants"), "{e}");
}

#[test]
fn non_float_inputs_refused() {
    let shape_only = ModelBuilder::new(8, 15)
        .input("x", I64, &d(&[2]))
        .output("y", F, &d(&[1]))
        .node_named("sh", "Shape", &["x"], &["s"], &[])
        .node("Cast", &["s"], &["y"], &[("to", Attr::Int(1))]);
    let e = unsupported(load(shape_only, &[("x", &[2])], &[]));
    assert!(e.contains("non-float runtime tensor x at node sh"), "{e}");
    let kept = ModelBuilder::new(8, 15)
        .output("y", F, &d(&[1]))
        .init_i64("k", &[3], &[1, 2, 3])
        .node_named("sh", "Shape", &["k"], &["s"], &[])
        .node("Cast", &["s"], &["y"], &[("to", Attr::Int(1))]);
    let e = unsupported(load(kept, &[], &["k"]));
    assert!(e.contains("non-float runtime tensor k at node sh"), "{e}");
    let unused = ModelBuilder::new(8, 15)
        .input("x", F, &d(&[2]))
        .input("n", I64, &d(&[2]))
        .output("y", F, &d(&[2]))
        .node("Relu", &["x"], &["y"], &[]);
    let e = unsupported(load(unused, &[("x", &[2]), ("n", &[2])], &[]));
    assert!(e.contains("non-float runtime tensor n (graph input)"), "{e}");
}

fn close(got: &[f32], want: &[f32], tol: f32) {
    assert_eq!(got.len(), want.len());
    for (g, w) in got.iter().zip(want) {
        assert!((g - w).abs() <= tol * w.abs().max(1e-3), "{got:?} vs {want:?}");
    }
}

fn softmax_ref(x: &[f32]) -> Vec<f32> {
    let m = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f64> = x.iter().map(|&v| ((v - m) as f64).exp()).collect();
    let s: f64 = e.iter().sum();
    e.iter().map(|&v| (v / s) as f32).collect()
}

#[test]
fn softmax_opset11_flattens() {
    let x: Vec<f32> = (0..12).map(|i| (i as f32 * 0.37).sin() * 3.0).collect();
    let m = ModelBuilder::new(6, 11)
        .input("x", F, &d(&[2, 2, 3]))
        .output("y", F, &d(&[2, 2, 3]))
        .node("Softmax", &["x"], &["y"], &[("axis", Attr::Int(1))]);
    let g = load(m, &[("x", &[2, 2, 3])], &[]).unwrap();
    assert_eq!(shape_of(&g, "y"), [2, 2, 3]);
    let (o, _) = run(g, &[("x", &x)], &["y"]);
    let want = [softmax_ref(&x[..6]), softmax_ref(&x[6..])].concat();
    close(&o[0], &want, 1e-6);
    assert!((o[0][..6].iter().sum::<f32>() - 1.0).abs() < 1e-6);
}

#[test]
fn softmax_opset13_axis() {
    let x: Vec<f32> = (0..12).map(|i| (i as f32 * 0.37).cos() * 2.0).collect();
    let mk = |axis: Option<i64>| -> Vec<f32> {
        let attrs: Vec<(&str, Attr)> = axis.map(|a| ("axis", Attr::Int(a))).into_iter().collect();
        let m = ModelBuilder::new(8, 13)
            .input("x", F, &d(&[2, 2, 3]))
            .output("y", F, &d(&[2, 2, 3]))
            .node("Softmax", &["x"], &["y"], &attrs);
        run(load(m, &[("x", &[2, 2, 3])], &[]).unwrap(), &[("x", &x)], &["y"]).0.remove(0)
    };
    let last = mk(None);
    close(&last, &(0..4).flat_map(|r| softmax_ref(&x[r * 3..r * 3 + 3])).collect::<Vec<_>>(), 1e-6);
    assert_eq!(mk(Some(-1)), last);
    // an axis before the last: libpryon would normalise from the axis on, not over the axis alone
    let m = ModelBuilder::new(8, 13)
        .input("x", F, &d(&[2, 2, 3]))
        .output("y", F, &d(&[2, 2, 3]))
        .node("Softmax", &["x"], &["y"], &[("axis", Attr::Int(1))]);
    let e = unsupported(load(m, &[("x", &[2, 2, 3])], &[]));
    assert!(e.contains("axis 1"), "{e}");
}

#[test]
fn reduce_keepdims_and_axes_input() {
    let x = [1.0, 5.0, 3.0, 4.0, 2.0, 6.0];
    let m = ModelBuilder::new(6, 11)
        .input("x", F, &d(&[2, 3]))
        .output("s", F, &d(&[1, 3]))
        .output("m", F, &d(&[3]))
        .node("ReduceSum", &["x"], &["s"], &[("axes", Attr::Ints(vec![0]))])
        .node("ReduceMax", &["x"], &["m"], &[("axes", Attr::Ints(vec![-2])), ("keepdims", Attr::Int(0))]);
    let g = load(m, &[("x", &[2, 3])], &[]).unwrap();
    assert_eq!((shape_of(&g, "s"), shape_of(&g, "m")), (vec![1, 3], vec![3]));
    let (o, _) = run(g, &[("x", &x)], &["s", "m"]);
    assert_eq!((&o[0][..], &o[1][..]), (&[5.0, 7.0, 9.0][..], &[4.0, 5.0, 6.0][..]));

    let m13 = ModelBuilder::new(8, 18)
        .input("x", F, &d(&[2, 3]))
        .output("s", F, &d(&[3]))
        .init_i64("ax", &[1], &[0])
        .node("ReduceSum", &["x", "ax"], &["s"], &[("keepdims", Attr::Int(0))]);
    let g = load(m13, &[("x", &[2, 3])], &[]).unwrap();
    assert_eq!(shape_of(&g, "s"), [3]);
    assert_eq!(run(g, &[("x", &x)], &["s"]).0[0], [5.0, 7.0, 9.0]);

    let noop = ModelBuilder::new(8, 18)
        .input("x", F, &d(&[2, 3]))
        .output("s", F, &d(&[2, 3]))
        .node("ReduceSum", &["x"], &["s"], &[("noop_with_empty_axes", Attr::Int(1))]);
    let e = unsupported(load(noop, &[("x", &[2, 3])], &[]));
    assert!(e.contains("noop_with_empty_axes"), "{e}");
}

/// libpryon runs other reductions with other kernels (an all-axes NEON lane sum, a VFP chain per row): refused.
#[test]
fn reduce_other_axes_refused() {
    #[allow(clippy::type_complexity)]
    let cases: [(&str, &[usize], Vec<(&str, Attr)>); 4] = [
        ("ReduceSum", &[2, 3], vec![("axes", Attr::Ints(vec![1]))]),
        ("ReduceSum", &[2, 3], vec![]),
        ("ReduceMax", &[2, 3], vec![("axes", Attr::Ints(vec![0, 1]))]),
        ("ReduceMax", &[2, 2, 3], vec![("axes", Attr::Ints(vec![0]))]),
    ];
    for (op, dims, attrs) in cases {
        let di: Vec<i64> = dims.iter().map(|&v| v as i64).collect();
        let m = ModelBuilder::new(6, 11).input("x", F, &d(&di)).output("y", F, &[]).node(op, &["x"], &["y"], &attrs);
        let e = unsupported(load(m, &[("x", dims)], &[]));
        assert!(e.contains("only axis 0 of a 2-D input"), "{op} {dims:?}: {e}");
    }
}

#[test]
fn maxpool_kernel_stride() {
    let x: Vec<f32> = (0..16).map(|i| ((i * 7) % 16) as f32).collect();
    let m = ModelBuilder::new(6, 11)
        .input("x", F, &d(&[1, 1, 4, 4]))
        .output("y", F, &d(&[1, 1, 2, 2]))
        .node("MaxPool", &["x"], &["y"], &[("kernel_shape", Attr::Ints(vec![2, 2])), ("strides", Attr::Ints(vec![2, 2]))]);
    let g = load(m, &[("x", &[1, 1, 4, 4])], &[]).unwrap();
    assert_eq!(shape_of(&g, "y"), [1, 1, 2, 2]);
    let (o, _) = run(g, &[("x", &x)], &["y"]);
    let at = |r: usize, c: usize| x[r * 4 + c];
    let want: Vec<f32> = [(0, 0), (0, 2), (2, 0), (2, 2)]
        .iter()
        .map(|&(r, c)| at(r, c).max(at(r, c + 1)).max(at(r + 1, c)).max(at(r + 1, c + 1)))
        .collect();
    assert_eq!(o[0], want);
}

#[test]
fn maxpool_pads_refused() {
    for (name, attr) in [
        ("pads", Attr::Ints(vec![1, 1, 1, 1])),
        ("dilations", Attr::Ints(vec![2, 2])),
        ("ceil_mode", Attr::Int(1)),
        ("auto_pad", Attr::Str("SAME_UPPER".into())),
    ] {
        let m = ModelBuilder::new(6, 11)
            .input("x", F, &d(&[1, 1, 4, 4]))
            .output("y", F, &d(&[1, 1, 2, 2]))
            .node("MaxPool", &["x"], &["y"], &[("kernel_shape", Attr::Ints(vec![2, 2])), (name, attr)]);
        let e = unsupported(load(m, &[("x", &[1, 1, 4, 4])], &[]));
        assert!(e.contains(name), "{e}");
    }
}

#[test]
fn batchnorm_values() {
    let x: Vec<f32> = (0..12).map(|i| i as f32 - 3.0).collect();
    let (g, b, mu, var) = ([2.0f32, 0.5], [0.25f32, -1.0], [1.0f32, -2.0], [4.0f32, 0.25]);
    let m = ModelBuilder::new(6, 11)
        .input("x", F, &d(&[2, 2, 3]))
        .output("y", F, &d(&[2, 2, 3]))
        .init_f32("g", &[2], &g)
        .init_f32("b", &[2], &b)
        .init_f32("mu", &[2], &mu)
        .init_f32("var", &[2], &var)
        .node("BatchNormalization", &["x", "g", "b", "mu", "var"], &["y"], &[("epsilon", Attr::Float(0.0))]);
    let gr = load(m, &[("x", &[2, 2, 3])], &[]).unwrap();
    let (o, _) = run(gr, &[("x", &x)], &["y"]);
    let want: Vec<f32> = (0..12)
        .map(|i| {
            let c = (i / 3) % 2;
            let s = g[c] / var[c].sqrt();
            x[i].mul_add(s, b[c] - mu[c] * s)
        })
        .collect();
    assert_eq!(o[0], want);

    let t = ModelBuilder::new(6, 11)
        .input("x", F, &d(&[2, 2, 3]))
        .output("y", F, &d(&[2, 2, 3]))
        .init_f32("g", &[2], &g)
        .init_f32("b", &[2], &b)
        .init_f32("mu", &[2], &mu)
        .init_f32("var", &[2], &var)
        .node("BatchNormalization", &["x", "g", "b", "mu", "var"], &["y"], &[("training_mode", Attr::Int(1))]);
    let e = unsupported(load(t, &[("x", &[2, 2, 3])], &[]));
    assert!(e.contains("training_mode"), "{e}");
}

#[test]
fn float_ops_fold_constants() {
    let m = ModelBuilder::new(6, 13)
        .input("x", F, &d(&[2]))
        .output("y", F, &d(&[2]))
        .init_f32("c", &[2, 2], &[1.0, 2.0, 3.0, 4.0])
        .node("ReduceSum", &["c"], &["s"], &[("axes", Attr::Ints(vec![0])), ("keepdims", Attr::Int(0))])
        .node("Softmax", &["c"], &["sm"], &[])
        .node("ReduceMax", &["sm"], &["mx"], &[("keepdims", Attr::Int(0)), ("axes", Attr::Ints(vec![0]))])
        .node("Add", &["x", "s"], &["t"], &[])
        .node("Add", &["t", "mx"], &["y"], &[]);
    let g = load(m, &[("x", &[2])], &[]).unwrap();
    let (o, traced) = run(g, &[("x", &[10.0, 20.0])], &["y"]);
    assert_eq!(traced, ["t", "y"]);
    // column sums 4 and 6; both softmax rows are equal (shift invariance), so the column max is either row
    let p = softmax_ref(&[1.0, 2.0]);
    close(&o[0], &[14.0 + p[0], 26.0 + p[1]], 1e-6);
}

#[test]
fn extra_outputs_refused() {
    let mp = ModelBuilder::new(6, 11)
        .input("x", F, &d(&[1, 1, 4, 4]))
        .output("y", F, &d(&[1, 1, 2, 2]))
        .node_named("p", "MaxPool", &["x"], &["y", "idx"], &[("kernel_shape", Attr::Ints(vec![2, 2])), ("strides", Attr::Ints(vec![2, 2]))]);
    let e = unsupported(load(mp, &[("x", &[1, 1, 4, 4])], &[]));
    assert!(e.contains("MaxPool (node p) with 2 outputs"), "{e}");
    let bn = ModelBuilder::new(6, 11)
        .input("x", F, &d(&[1, 2]))
        .output("y", F, &d(&[1, 2]))
        .init_f32("g", &[2], &[1.0, 1.0])
        .init_f32("b", &[2], &[0.0, 0.0])
        .node_named("n", "BatchNormalization", &["x", "g", "b", "g", "g"], &["y", "m", "v"], &[]);
    let e = unsupported(load(bn, &[("x", &[1, 2])], &[]));
    assert!(e.contains("BatchNormalization (node n) with 3 outputs"), "{e}");
}

/// One-node graph over `x` of shape `dims`; returns the output bits.
fn one(op: &str, dims: &[usize], attrs: &[(&str, Attr)], x: &[f32], inits: &[(&str, &[f32])]) -> Vec<u32> {
    let di: Vec<i64> = dims.iter().map(|&v| v as i64).collect();
    let mut m = ModelBuilder::new(8, 11).input("x", F, &d(&di)).output("y", F, &[]);
    let mut ins = vec!["x"];
    for (n, v) in inits {
        m = m.init_f32(n, &[v.len() as i64], v);
        ins.push(n);
    }
    let m = m.node(op, &ins, &["y"], attrs);
    let g = load(m, &[("x", dims)], &[]).unwrap();
    bits(&run(g, &[("x", x)], &["y"]).0[0])
}

/// libpryon's Softmax (CPUNK SoftmaxImpl<CortexA15>): max by NEON vmax, CPUNK exp of x - max, the sum in four lanes
/// (tail first, then vectors last to first), one reciprocal, products. Bits from an independent f32 oracle of
/// that form; every row of the captures matches it.
#[test]
fn softmax_bits() {
    let cases: [(&[f32], &[u32]); 3] = [
        (&[0.1, 0.7, -1.3], &[0x3ea6d854, 0x3f180178, 0x3da492ec]),
        (&[2.5, -0.75, 0.3, 1.125], &[0x3f368ac1, 0x3ce27e55, 0x3da1cf51, 0x3e389d8a]),
        (
            &[0.9, -2.2, 0.35, 1.7, -0.6, 0.05, 2.4, -1.1, 0.8],
            &[0x3dcc646d, 0x3b9352c6, 0x3d6bd921, 0x3e637115, 0x3cb66ca2, 0x3d2eb87c, 0x3ee50176, 0x3c5d4abd, 0x3db8f118],
        ),
    ];
    for (x, want) in cases {
        assert_eq!(one("Softmax", &[1, x.len()], &[], x, &[]), want, "{x:?}");
    }
}

/// ReduceSum over axis 0 is libpryon's chain of CPUNK Adds: the first slice copied, then NEON adds (flush to zero).
#[test]
fn reduce_sum_order() {
    let ax = [("axes", Attr::Ints(vec![0]))];
    let tiny = f32::from_bits(0x0000_1000);
    // -0 + -0 stays -0 (a 0 seed would make it +0); a subnormal is flushed by the add, kept by the copy
    assert_eq!(one("ReduceSum", &[2, 2], &ax, &[-0.0, tiny, -0.0, 0.0], &[]), [0x8000_0000, 0]);
    assert_eq!(one("ReduceSum", &[1, 2], &ax, &[-0.0, tiny], &[]), [0x8000_0000, 0x1000]);
    // in order 0; pairwise (1 + h) + (h - 1) gives 2^-24, reversed 2^-23
    let h = f32::from_bits(0x3380_0000);
    let x = [1.0, h, h, -1.0];
    assert_ne!(((1.0f32 + h) + (h + -1.0)).to_bits(), 0);
    assert_ne!((((-1.0f32 + h) + h) + 1.0).to_bits(), 0);
    assert_eq!(one("ReduceSum", &[4, 1], &ax, &x, &[]), [0]);
}

/// ReduceMax over axis 0 starts from the first slice and keeps it unless `acc < x`: a NaN there stays.
#[test]
fn reduce_max_first_slice() {
    let ax = [("axes", Attr::Ints(vec![0]))];
    let nan = f32::NAN;
    assert_eq!(one("ReduceMax", &[2, 2], &ax, &[nan, 1.0, 2.0, nan], &[]), [nan.to_bits(), 1.0f32.to_bits()]);
}

/// libpryon's BatchNormalization (0xb3e534): s = g / sqrt(var + eps), b' = B - mean·s (CPUNK), y = fmaf(x, s, b').
#[test]
fn batchnorm_form() {
    let p = |v: f32| [v];
    let cases = [
        ((0.8938f32, 1.883f32, -0.942f32, -0.069f32, 2.833f32), 0x3e0a2a7au32),
        ((2.7994, 1.517, -0.59, 0.882, 2.087), 0x3fb6233e),
    ];
    for ((x, g, b, mu, var), want) in cases {
        let got = one(
            "BatchNormalization",
            &[1, 1, 1],
            &[("epsilon", Attr::Float(0.001))],
            &[x],
            &[("g", &p(g)), ("b", &p(b)), ("mu", &p(mu)), ("var", &p(var))],
        );
        assert_eq!(got, [want], "x {x}");
    }
}

/// MaxPool (0xd016a8, NCHW): max from -FLT_MAX by `v > m`, so a window of NaN gives -FLT_MAX.
#[test]
fn maxpool_nan_window() {
    let nan = f32::NAN;
    let ks = [("kernel_shape", Attr::Ints(vec![1, 2])), ("strides", Attr::Ints(vec![1, 2]))];
    assert_eq!(one("MaxPool", &[1, 1, 1, 4], &ks, &[nan, nan, nan, 2.0], &[]), [f32::MIN.to_bits(), 2.0f32.to_bits()]);
}

/// Add/Sub/Mul run as CPUNK NEON kernels: subnormal inputs and results flush to zero.
#[test]
fn elementwise_flush_to_zero() {
    let m = ModelBuilder::new(6, 11)
        .input("a", F, &d(&[2]))
        .input("b", F, &d(&[2]))
        .output("s", F, &d(&[2]))
        .output("p", F, &d(&[2]))
        .node("Add", &["a", "b"], &["s"], &[])
        .node("Mul", &["a", "b"], &["p"], &[]);
    let g = load(m, &[("a", &[2]), ("b", &[2])], &[]).unwrap();
    let tiny = f32::from_bits(0x0000_1000);
    let (o, _) = run(g, &[("a", &[tiny, 1e-20]), ("b", &[0.0, 1e-20])], &["s", "p"]);
    assert_eq!((bits(&o[0]), bits(&o[1])), (vec![0, (1e-20f32 + 1e-20).to_bits()], vec![0, 0]));
}
