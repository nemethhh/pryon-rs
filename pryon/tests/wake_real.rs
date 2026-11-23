use pryon::onnx::op_census;

type Ops = &'static [(&'static str, usize)];

const FW: (&str, i64, i64, Ops) = (
    "runtime_stateless_standard.streaming.fixed.int16.onnx",
    6,
    11,
    &[
        ("Add", 4),
        ("BatchNormalization", 7),
        ("Cast", 10),
        ("Concat", 10),
        ("Gather", 5),
        ("HybridConv", 7),
        ("HybridGemm", 4),
        ("HybridLSTM", 2),
        ("MaxPool", 2),
        ("Mul", 1),
        ("Relu", 9),
        ("Reshape", 10),
        ("Shape", 10),
        ("Slice", 11),
        ("Softmax", 1),
        ("Squeeze", 6),
        ("Sub", 1),
        ("Tile", 1),
        ("Transpose", 4),
        ("Unsqueeze", 4),
    ],
);

const ENCODER: Ops = &[
    ("Add", 10),
    ("Concat", 11),
    ("HybridGemm", 12),
    ("Mul", 21),
    ("ReduceMax", 3),
    ("ReduceSum", 4),
    ("Relu", 9),
    ("Reshape", 1),
    ("Slice", 31),
    ("Squeeze", 1),
    ("Sub", 1),
];
const DET: Ops = &[
    ("Add", 5),
    ("HybridGemm", 5),
    ("HybridLSTM", 2),
    ("Relu", 2),
    ("Softmax", 2),
    ("Squeeze", 6),
    ("Tile", 1),
    ("Unsqueeze", 4),
];
const STOP: Ops = &[
    ("Concat", 4),
    ("HybridConv", 4),
    ("HybridGemm", 2),
    ("Relu", 4),
    ("Reshape", 1),
    ("Slice", 4),
    ("Softmax", 1),
    ("Squeeze", 2),
    ("Unsqueeze", 2),
];
const MSWW: Ops = &[
    ("Add", 2),
    ("Concat", 6),
    ("HybridGemm", 12),
    ("Mul", 6),
    ("ReduceSum", 6),
    ("Relu", 8),
    ("Slice", 6),
    ("Softmax", 1),
    ("Squeeze", 4),
    ("Unsqueeze", 2),
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

fn f32s(path: &std::path::Path) -> Vec<f32> {
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

/// Within 1e-3 of pypryon (fed the same LFBE frames, `local/make_pyref.sh` dumps) over each clip's first steps,
/// where alignment, ring, primed-state and binding faults all show; later steps may carry an activation-quantization
/// rounding flip set off by pypryon's f64 batch-norm and LSTM-cell math, so they are held to the size of one flip.
#[test]
fn pipeline_matches_pypryon() {
    use pryon::wake::{LoadedSet, Pipeline, StepOut};
    use std::path::{Path, PathBuf};
    use std::sync::Arc;
    const TOL: f32 = 1e-3;
    const PREFIX: usize = 8;
    const FLIP: f32 = 5e-2;
    let Ok(refs) = std::env::var("PRYON_PYPRYON_REF") else {
        println!("skipped: set PRYON_PYPRYON_REF");
        return;
    };
    let mut sets: Vec<PathBuf> = Vec::new();
    match std::env::var("PRYON_WAKE_FW_DIR") {
        Ok(d) => sets.push(d.into()),
        Err(_) => println!("skipped firmware set: set PRYON_WAKE_FW_DIR"),
    }
    match std::env::var("PRYON_WAKE_SETS_DIR") {
        Ok(d) => {
            let words = ["alexa", "amazon", "computer", "echo", "ziggy"];
            sets.extend(words.map(|w| Path::new(&d).join(format!("{w}-en-US"))));
        }
        Err(_) => println!("skipped DAVS sets: set PRYON_WAKE_SETS_DIR"),
    }
    for dir in sets {
        let set = Arc::new(LoadedSet::load(&dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display())));
        let root = Path::new(&refs).join(&set.spec.checksum);
        let mut clips: Vec<PathBuf> = std::fs::read_dir(&root)
            .unwrap_or_else(|e| panic!("{}: {e} (run local/make_pyref.sh)", root.display()))
            .map(|e| e.unwrap().path())
            .collect();
        clips.sort();
        assert!(!clips.is_empty(), "{}: no dumps", root.display());
        let mut worst = vec![0f32; set.components.len()];
        for clip in &clips {
            let b = std::fs::read(clip.join("pcm.raw")).unwrap();
            let pcm: Vec<i16> = b.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]])).collect();
            let mut rows: Vec<Vec<f32>> = vec![Vec::new(); set.components.len()];
            Pipeline::new(&set).push(&pcm, &mut |s: &StepOut| {
                for (r, v) in rows.iter_mut().zip(s.values) {
                    r.extend_from_slice(v);
                }
            });
            for (k, c) in set.components.iter().enumerate() {
                let want = f32s(&clip.join(format!("{}.f32", c.key)));
                assert_eq!(rows[k].len(), want.len(), "{} {}: values (steps x {:?})", clip.display(), c.key, c.shape);
                assert!(rows[k].iter().all(|x| x.is_finite()), "{} {}", clip.display(), c.key);
                let n: usize = c.shape.iter().product();
                let per_step: Vec<f32> = rows[k]
                    .chunks(n)
                    .zip(want.chunks(n))
                    .map(|(a, b)| a.iter().zip(b).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max))
                    .collect();
                let d = per_step.iter().copied().fold(0f32, f32::max);
                let over: Vec<usize> = (0..per_step.len()).filter(|&t| per_step[t] > TOL).collect();
                let what = format!("{} {} {}", dir.display(), clip.file_name().unwrap().to_string_lossy(), c.key);
                let first = over.first().copied();
                println!(
                    "{what}: max |d| {d:.3e}, {} of {} steps over {TOL}, first {first:?}",
                    over.len(),
                    per_step.len()
                );
                assert!(first.is_none_or(|t| t >= PREFIX), "{what}: step {first:?} differs by more than {TOL}");
                assert!(d <= FLIP, "{what}: max |d| {d}");
                worst[k] = worst[k].max(d);
            }
        }
        for (c, w) in set.components.iter().zip(&worst) {
            println!("{} {}: max |d| over {} clips {w:.3e}", dir.display(), c.key, clips.len());
        }
    }
}
