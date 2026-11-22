//! Bit comparisons against libpryon's own tensors, captured per step and node by `local/capture/wake_capture.py`
//! under `PRYON_CAPTURE_DIR/<set checksum>/<clip>/` (`pcm.raw`, `<stage>/<step>/shape.json` and `<tensor>.f32`).

use pryon::onnx::Session;
use pryon::wake::{LoadedSet, Pipeline, StageInput};
use serde_json::{Map, Value};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

struct Captured {
    set: Arc<LoadedSet>,
    clips: Vec<PathBuf>,
}

/// The sets whose set variable is set and that have captures; `None` (and a printed reason) when nothing to do.
fn captured() -> Option<Vec<Captured>> {
    let Ok(root) = std::env::var("PRYON_CAPTURE_DIR") else {
        println!("skipped: set PRYON_CAPTURE_DIR");
        return None;
    };
    let mut dirs: Vec<PathBuf> = Vec::new();
    match std::env::var("PRYON_WAKE_FW_DIR") {
        Ok(d) => dirs.push(d.into()),
        Err(_) => println!("skipped firmware set: set PRYON_WAKE_FW_DIR"),
    }
    match std::env::var("PRYON_WAKE_SETS_DIR") {
        Ok(d) => dirs.push(Path::new(&d).join("alexa-en-US")),
        Err(_) => println!("skipped alexa-en-US: set PRYON_WAKE_SETS_DIR"),
    }
    if dirs.is_empty() {
        return None;
    }
    let mut out = Vec::new();
    for dir in dirs {
        let set = Arc::new(LoadedSet::load(&dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display())));
        let top = Path::new(&root).join(&set.spec.checksum);
        let mut clips: Vec<PathBuf> = std::fs::read_dir(&top)
            .unwrap_or_else(|e| panic!("{}: {e} (run local/capture/capture_all.sh)", top.display()))
            .map(|e| e.unwrap().path())
            .filter(|p| p.join("capture.json").is_file())
            .collect();
        clips.sort();
        assert!(!clips.is_empty(), "{}: no captures", top.display());
        out.push(Captured { set, clips });
    }
    Some(out)
}

struct Step {
    dir: PathBuf,
    meta: Map<String, Value>,
}

impl Step {
    /// The tensor's values; `None` when libpryon held it in a type the tracer does not save (integers).
    fn tensor(&self, name: &str) -> Option<Vec<f32>> {
        let file = self.meta.get(name)?.get("file")?.as_str()?;
        let b = std::fs::read(self.dir.join(file)).unwrap();
        Some(b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
    }
}

/// Captured steps 0, 1, … of a stage, up to the first one missing.
fn steps(clip: &Path, stage: &str) -> Vec<Step> {
    (0..)
        .map(|t| clip.join(stage).join(format!("{t:05}")))
        .map_while(|dir| {
            let text = std::fs::read_to_string(dir.join("shape.json")).ok()?;
            let Value::Object(meta) = serde_json::from_str(&text).unwrap() else { panic!("{}", dir.display()) };
            Some(Step { dir, meta })
        })
        .collect()
}

fn pcm(clip: &Path) -> Vec<i16> {
    let b = std::fs::read(clip.join("pcm.raw")).unwrap();
    b.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]])).collect()
}

fn name(clip: &Path) -> String {
    clip.file_name().unwrap().to_string_lossy().into_owned()
}

/// Distance in representable f32 values (both NaN counts as equal, NaN against a number as `u32::MAX`).
fn ulp(a: f32, b: f32) -> u32 {
    if a.is_nan() || b.is_nan() {
        return if a.is_nan() && b.is_nan() { 0 } else { u32::MAX };
    }
    let key = |x: f32| {
        let u = x.to_bits();
        if u >> 31 == 1 { !u } else { u | 1 << 31 }
    };
    key(a).abs_diff(key(b))
}

/// (values differing, max ulp, first differing index).
fn diff(a: &[f32], b: &[f32]) -> (usize, u32, Option<usize>) {
    let mut out = (0, 0, None);
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        let u = ulp(*x, *y);
        if u > 0 {
            out.0 += 1;
            out.1 = out.1.max(u);
            out.2.get_or_insert(i);
        }
    }
    out
}

/// libpryon's stage-0 input of every captured step equals the engine's LFBE frames of `pcm.raw`, bit for bit.
#[test]
fn capture_lfbe_bits() {
    let Some(caps) = captured() else { return };
    let mut bad = Vec::new();
    for c in &caps {
        let st = &c.set.spec.stages[0];
        let StageInput::Audio(input) = &st.inputs[0] else { panic!("stage 0 has no audio input") };
        let (fps, bins) = (st.frames_per_step, c.set.spec.fe.bins);
        for clip in &c.clips {
            let mut fe = pryon::lfbe::Lfbe::with_spec(c.set.spec.fe);
            let mut frames = Vec::new();
            for chunk in pcm(clip).chunks(160) {
                fe.push_flat(chunk, &mut frames);
            }
            let steps = steps(clip, &st.name);
            assert!(!steps.is_empty(), "{}: no {} steps", clip.display(), st.name);
            let mut differ = 0;
            for (t, s) in steps.iter().enumerate() {
                let got = s.tensor(input).unwrap_or_else(|| panic!("{}: step {t} has no {input}", clip.display()));
                let want = frames.get(t * fps * bins..(t + 1) * fps * bins);
                let want = want.unwrap_or_else(|| panic!("{}: step {t} beyond the engine's frames", clip.display()));
                let (n, u, at) = diff(&got, want);
                if n > 0 {
                    differ += 1;
                    if differ == 1 {
                        let at = at.unwrap();
                        let where_ = format!("frame {} bin {}", t * fps + at / bins, at % bins);
                        bad.push(format!("{} {}: step {t} {where_}: {n} values, max ulp {u}", st.name, name(clip)));
                    }
                }
            }
            println!("{} {}: {} steps, {} with differing frames", st.name, name(clip), steps.len(), differ);
        }
    }
    assert!(bad.is_empty(), "LFBE frames differ:\n{}", bad.join("\n"));
}

/// The state every stage starts from (LSTM lanes, recurrent buffers) equals libpryon's step-0 state. libpryon leaves
/// a recurrent buffer whose graph stores no value (the firmware set's data-less placeholders, with
/// `recurrent_state_initialization` false) uninitialized: heap contents that change with the install path. The
/// engine zero-fills those; they are reported, not compared.
#[test]
fn capture_initial_state() {
    let Some(caps) = captured() else { return };
    let mut bad = Vec::new();
    for c in &caps {
        let p = Pipeline::new(&c.set);
        for (s, st) in c.set.spec.stages.iter().enumerate() {
            for clip in &c.clips {
                let step0 = steps(clip, &st.name).into_iter().next();
                let step0 = step0.unwrap_or_else(|| panic!("{}: no {} step 0", clip.display(), st.name));
                for (input, _) in st.lstm.iter().chain(&st.recurrent) {
                    let ours = p.input(s, input).unwrap();
                    let got = step0.tensor(input).unwrap_or_else(|| panic!("{}: no {input}", clip.display()));
                    assert_eq!(got.len(), ours.len(), "{} {input}", st.name);
                    let (n, u, _) = diff(&got, ours);
                    let what = format!("{} {} {input}", st.name, name(clip));
                    if n == 0 {
                        println!("{what}: bit-equal ({} values)", got.len());
                    } else if !st.primed && st.recurrent.iter().any(|(i, _)| i == input) {
                        let nonfinite = got.iter().filter(|x| !x.is_finite()).count();
                        let nonzero = got.iter().filter(|x| **x != 0.0).count();
                        let len = got.len();
                        println!("{what}: libpryon uninitialized ({nonzero} of {len} nonzero, {nonfinite} non-finite)");
                    } else {
                        bad.push(format!("{what}: {n} of {} values differ, max ulp {u}", got.len()));
                    }
                }
            }
        }
    }
    assert!(bad.is_empty(), "step-0 state differs:\n{}", bad.join("\n"));
}

/// A node whose output differs while every input it reads is bit-equal (or fed, or a constant).
#[derive(Default)]
struct Root {
    op: String,
    node: usize,
    steps: usize,
    first: (String, usize),
    max_ulp: u32,
    /// Steps whose captured inputs of the node are all finite (the fw warm-up feeds NaN and heap garbage).
    finite_steps: usize,
    finite_first: Option<(String, usize)>,
    finite_max_ulp: u32,
    finite_max_abs: f32,
}

#[derive(Default)]
struct StageReport {
    steps: usize,
    tensors: usize,
    equal: usize,
    /// Tensors the engine traces but the tracer did not save (counted per step).
    unmatched: BTreeMap<String, usize>,
    /// Per diverging tensor: (op, node, max ulp).
    worst: BTreeMap<String, (String, usize, u32)>,
    roots: BTreeMap<String, Root>,
    /// Per op type: (outputs computed from bit-equal inputs, of those bit-equal).
    ops: BTreeMap<String, (usize, usize)>,
}

/// `OUT/<stage>/nodes.json`: output name -> (node index, op, inputs).
fn node_table(clip: &Path, stage: &str) -> HashMap<String, (usize, String, Vec<String>)> {
    let path = clip.join(stage).join("nodes.json");
    let text = std::fs::read_to_string(&path);
    let text = text.unwrap_or_else(|e| panic!("{}: {e} (wake_capture.py --nodes-only)", path.display()));
    let Value::Array(nodes) = serde_json::from_str(&text).unwrap() else { panic!("{}", path.display()) };
    let strs = |v: &Value| -> Vec<String> {
        v.as_array().unwrap().iter().map(|x| x.as_str().unwrap().to_string()).collect()
    };
    let mut out = HashMap::new();
    for n in &nodes {
        let idx = n["node"].as_u64().unwrap() as usize;
        for o in strs(&n["outputs"]) {
            out.insert(o, (idx, n["op"].as_str().unwrap().to_string(), strs(&n["inputs"])));
        }
    }
    out
}

/// Every stage, fed libpryon's captured inputs and state of each step, reproduces every captured node output bit
/// for bit. On failure the message is the divergence report: per stage the first diverging node in graph order
/// (output differs, inputs bit-equal) with its op type, first step and max ulp, then every such node and every
/// diverging tensor.
#[test]
fn capture_nodes_bits() {
    let Some(caps) = captured() else { return };
    let mut report = String::new();
    let mut diverged = false;
    for c in &caps {
        let set = &c.set;
        for (s, st) in set.spec.stages.iter().enumerate() {
            let mut inputs: Vec<&str> = st.lstm.iter().chain(&st.recurrent).map(|(i, _)| i.as_str()).collect();
            inputs.push(match &st.inputs[0] {
                StageInput::Audio(n) => n,
                StageInput::Artifact { input, .. } => input,
            });
            let mut r = StageReport::default();
            for clip in &c.clips {
                let nodes = node_table(clip, &st.name);
                let mut sess = Session::new(set.graphs[s].clone());
                for (t, step) in steps(clip, &st.name).iter().enumerate() {
                    for input in &inputs {
                        let i = set.graphs[s].input_index(input).unwrap();
                        let v = step.tensor(input).unwrap_or_else(|| panic!("{}: no {input}", step.dir.display()));
                        sess.input_mut(i).copy_from_slice(&v);
                    }
                    let mut differ: HashSet<String> = HashSet::new();
                    sess.run_traced(&mut |tensor, _dims, ours| {
                        let Some(got) = step.tensor(tensor) else {
                            *r.unmatched.entry(tensor.to_string()).or_default() += 1;
                            return;
                        };
                        assert_eq!(got.len(), ours.len(), "{} {tensor}: values", step.dir.display());
                        r.tensors += 1;
                        let (n, u, _) = diff(&got, ours);
                        let (node, op, ins) = nodes.get(tensor).unwrap_or_else(|| panic!("{tensor}: no node"));
                        // run_traced goes in topological order, so the inputs' verdicts are already in
                        let inputs_equal = !ins.iter().any(|x| differ.contains(x));
                        if inputs_equal {
                            let e = r.ops.entry(op.clone()).or_default();
                            e.0 += 1;
                            e.1 += usize::from(n == 0);
                        }
                        if n == 0 {
                            r.equal += 1;
                            return;
                        }
                        differ.insert(tensor.to_string());
                        let w = r.worst.entry(tensor.to_string()).or_insert((op.clone(), *node, 0));
                        w.2 = w.2.max(u);
                        if !inputs_equal {
                            return;
                        }
                        let finite =
                            ins.iter().all(|x| step.tensor(x).is_none_or(|v| v.iter().all(|y| y.is_finite())));
                        let root = r.roots.entry(tensor.to_string()).or_insert_with(|| Root {
                            op: op.clone(),
                            node: *node,
                            first: (name(clip), t),
                            ..Root::default()
                        });
                        root.steps += 1;
                        root.max_ulp = root.max_ulp.max(u);
                        if finite {
                            let d = got.iter().zip(ours).map(|(a, b)| (a - b).abs()).filter(|d| d.is_finite());
                            root.finite_steps += 1;
                            root.finite_first.get_or_insert((name(clip), t));
                            root.finite_max_ulp = root.finite_max_ulp.max(u);
                            root.finite_max_abs = d.fold(root.finite_max_abs, f32::max);
                        }
                    });
                    r.steps += 1;
                }
            }
            diverged |= !r.worst.is_empty();
            report += &stage_report(set, &st.name, c.clips.len(), &r);
        }
    }
    println!("{report}");
    assert!(!diverged, "per-node divergence from libpryon:\n{report}");
}

fn stage_report(set: &LoadedSet, stage: &str, clips: usize, r: &StageReport) -> String {
    let ulp = |u: u32| if u == u32::MAX { "NaN/number".to_string() } else { u.to_string() };
    let mut o = format!(
        "== {} ({}) stage {stage}: {} steps over {clips} clips, {} tensors compared, {} bit-equal\n",
        set.spec.dir.file_name().unwrap_or_default().to_string_lossy(),
        set.spec.checksum,
        r.steps,
        r.tensors,
        r.equal
    );
    let mut roots: Vec<(&String, &Root)> = r.roots.iter().collect();
    roots.sort_by_key(|(t, x)| (x.node, t.as_str()));
    let first = |finite: bool| roots.iter().find(|(_, x)| !finite || x.finite_steps > 0);
    match (first(true), first(false)) {
        (_, None) => o += "   no divergence\n",
        (f, Some((t, x))) => {
            if let Some((t, x)) = f {
                let (clip, step) = x.finite_first.clone().unwrap();
                o += &format!(
                    "   first diverging node: {t} ({}, node {}) at step {step} of {clip}, max ulp {} (finite inputs, \
                     max |d| {:.3e})\n",
                    x.op,
                    x.node,
                    ulp(x.finite_max_ulp),
                    x.finite_max_abs
                );
            }
            if f.is_none_or(|(ft, _)| ft != t) {
                o += &format!(
                    "   first diverging node on any input: {t} ({}, node {}) at step {} of {}, max ulp {}\n",
                    x.op, x.node, x.first.1, x.first.0, ulp(x.max_ulp)
                );
            }
            o += "   nodes diverging on bit-equal inputs: steps (finite-input steps), first step, max ulp (finite):\n";
            for (t, x) in &roots {
                let fin = x.finite_first.as_ref().map_or("-".to_string(), |(c, s)| format!("{c} step {s}"));
                o += &format!(
                    "     node {:3} {:<18} {t}: {} ({}), first {} step {} / finite {fin}, max ulp {} ({}, \
                     |d| {:.3e})\n",
                    x.node,
                    x.op,
                    x.steps,
                    x.finite_steps,
                    x.first.0,
                    x.first.1,
                    ulp(x.max_ulp),
                    ulp(x.finite_max_ulp),
                    x.finite_max_abs
                );
            }
            let mut worst: Vec<(&String, &(String, usize, u32))> = r.worst.iter().collect();
            worst.sort_by_key(|(t, w)| (w.1, t.as_str()));
            o += &format!("   diverging tensors ({}), max ulp:\n", worst.len());
            for (t, (op, node, u)) in worst {
                o += &format!("     node {node:3} {op:<18} {t}: {}\n", ulp(*u));
            }
        }
    }
    let ops: Vec<String> = r.ops.iter().map(|(op, (n, e))| format!("{op} {e}/{n}")).collect();
    o += &format!("   bit-equal outputs from bit-equal inputs, by op: {}\n", ops.join(", "));
    if !r.unmatched.is_empty() {
        let names: Vec<&str> = r.unmatched.keys().map(String::as_str).collect();
        o += &format!("   traced by the engine, not saved by the tracer: {}\n", names.join(", "));
    }
    o
}
