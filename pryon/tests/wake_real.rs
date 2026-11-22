use pryon::onnx::op_census;

type Ops = &'static [(&'static str, usize)];

const FW: (&str, i64, i64, Ops) = (
    "runtime_stateless_standard.streaming.fixed.int16.onnx",
    6,
    11,
    &[
        ("Add", 4), ("BatchNormalization", 7), ("Cast", 10), ("Concat", 10), ("Gather", 5), ("HybridConv", 7),
        ("HybridGemm", 4), ("HybridLSTM", 2), ("MaxPool", 2), ("Mul", 1), ("Relu", 9), ("Reshape", 10),
        ("Shape", 10), ("Slice", 11), ("Softmax", 1), ("Squeeze", 6), ("Sub", 1), ("Tile", 1), ("Transpose", 4),
        ("Unsqueeze", 4),
    ],
);

const ENCODER: Ops = &[
    ("Add", 10), ("Concat", 11), ("HybridGemm", 12), ("Mul", 21), ("ReduceMax", 3), ("ReduceSum", 4), ("Relu", 9),
    ("Reshape", 1), ("Slice", 31), ("Squeeze", 1), ("Sub", 1),
];
const DET: Ops = &[
    ("Add", 5), ("HybridGemm", 5), ("HybridLSTM", 2), ("Relu", 2), ("Softmax", 2), ("Squeeze", 6), ("Tile", 1),
    ("Unsqueeze", 4),
];
const STOP: Ops = &[
    ("Concat", 4), ("HybridConv", 4), ("HybridGemm", 2), ("Relu", 4), ("Reshape", 1), ("Slice", 4), ("Softmax", 1),
    ("Squeeze", 2), ("Unsqueeze", 2),
];
const MSWW: Ops = &[
    ("Add", 2), ("Concat", 6), ("HybridGemm", 12), ("Mul", 6), ("ReduceSum", 6), ("Relu", 8), ("Slice", 6),
    ("Softmax", 1), ("Squeeze", 4), ("Unsqueeze", 2),
];

fn check(path: &std::path::Path, ir: i64, opset: i64, want: Ops) {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let (got_ir, opsets, ops) = op_census(&bytes).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    assert_eq!(got_ir, ir, "{} ir", path.display());
    assert_eq!(opsets, vec![(String::new(), opset)], "{} opsets", path.display());
    let want: Vec<(String, usize)> = want.iter().map(|&(o, n)| (o.to_string(), n)).collect();
    assert_eq!(ops, want, "{} ops", path.display());
}

#[test]
fn real_graphs_parse() {
    if let Ok(dir) = std::env::var("PRYON_WAKE_FW_DIR") {
        check(&std::path::Path::new(&dir).join(FW.0), FW.1, FW.2, FW.3);
    } else {
        println!("skipped: set PRYON_WAKE_FW_DIR");
    }
    let Ok(dir) = std::env::var("PRYON_WAKE_SETS_DIR") else {
        println!("skipped: set PRYON_WAKE_SETS_DIR");
        return;
    };
    let d = std::path::Path::new(&dir);
    let cases: [(&str, i64, i64, Ops); 5] = [
        ("alexa-en-US/encoder/primed_encoder.int8-streaming.onnx", 8, 15, ENCODER),
        ("alexa-en-US/alexa-wakeworddecoder/det_vrf_multi_lstm.int8.onnx", 8, 15, DET),
        ("alexa-en-US/multi-wakeword-wakeworddecoder/stop_hey_disney.streaming.fixed.int8.onnx", 10, 21, STOP),
        ("amazon-en-US/multi-wakeword-wakeworddecoder/odyssey_msww.streaming.fixed.int8.onnx", 10, 21, MSWW),
        ("echo-en-US/multi-wakeword-wakeworddecoder/echo_stop_hey_disney.streaming.fixed.int8.onnx", 10, 21, MSWW),
    ];
    for (p, ir, op, ops) in cases {
        check(&d.join(p), ir, op, ops);
    }
}
