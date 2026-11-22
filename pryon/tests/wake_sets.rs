//! Model-set loader: firmware and DAVS layouts, INCLUDE chains, refusal of anything not understood.

use pryon::lfbe::LfbeSpec;
use pryon::onnx::{Graph, Session};
use pryon::wake::{load_set, ClassifierSpec, SetSpec, StageInput, StageSpec};
use pryon::Error;
use pryon_testkit::wakeset::{write_davs_set, write_fw_set, WakeSetOpts};
use pryon_testkit::{Rng, TempDir};
use std::path::{Path, PathBuf};
use std::sync::Arc;

fn fw(tag: &str) -> TempDir {
    let d = TempDir::new(&format!("wakeset-fw-{tag}"));
    write_fw_set(&d, &WakeSetOpts { seed: 7, force_score: None });
    d
}

fn davs(tag: &str) -> TempDir {
    let d = TempDir::new(&format!("wakeset-davs-{tag}"));
    write_davs_set(&d, &WakeSetOpts { seed: 7, force_score: None });
    d
}

fn edit(path: &Path, from: &str, to: &str) {
    let s = std::fs::read_to_string(path).unwrap();
    assert!(s.contains(from), "{} lacks {from:?}", path.display());
    std::fs::write(path, s.replacen(from, to, 1)).unwrap();
}

fn append(path: &Path, text: &str) {
    let mut s = std::fs::read_to_string(path).unwrap();
    s.push_str(text);
    std::fs::write(path, s).unwrap();
}

fn unsupported(r: Result<SetSpec, Error>) -> (PathBuf, String) {
    match r {
        Err(Error::Unsupported { file, what }) => (file, what),
        Err(e) => panic!("expected Unsupported, got {e}"),
        Ok(_) => panic!("expected Unsupported, set loaded"),
    }
}

fn message(r: Result<SetSpec, Error>) -> String {
    match r {
        Err(e) => e.to_string(),
        Ok(_) => panic!("expected an error, set loaded"),
    }
}

fn spotter<'a>(s: &'a SetSpec, name: &str, aux: bool) -> &'a pryon::wake::SpotterSpec {
    s.spotters.iter().find(|p| p.name == name && p.auxiliary == aux).unwrap_or_else(|| panic!("no spotter {name}"))
}

#[test]
fn loads_synthetic_fw() {
    let d = fw("ok");
    let s = load_set(&d).unwrap();
    assert_eq!(s.dir, d.path());
    assert_eq!((s.fe.bins, s.fe.low_hz, s.fe.high_hz), (LfbeSpec::WAKE.bins, 80.0, 7200.0));
    assert_eq!(s.stages.len(), 1);
    let st = &s.stages[0];
    assert_eq!((st.frames_per_step, st.feat_dim), (6, 64));
    assert_eq!(st.graph, d.join("synthetic.streaming.onnx"));
    assert_eq!(st.inputs, vec![StageInput::Audio("input_feats".into())]);
    assert_eq!(
        st.outputs,
        vec![("softmax_output_ww".into(), "softmax_output_ww".into()), ("fc_output_se".into(), "fc_output_se".into())]
    );
    assert_eq!(st.lstm.len(), 4);
    assert_eq!(st.lstm[0], ("input_h_ww".into(), "lstm_ww_1".into()));
    assert_eq!(st.recurrent, vec![("streaming_conv_buffer_input_7".into(), "streaming_conv_buffer_output_7".into())]);
    assert!(!st.primed);
    assert_eq!(s.spotters.len(), 2);
    let a = spotter(&s, "ALEXA", false);
    assert_eq!(a.scorer, st.name);
    assert!(matches!(a.classifier, ClassifierSpec::MovingAverageMaxPool { window: 3, trim_tail: 8 }));
    assert_eq!((a.thresholds.accept, a.thresholds.notify), (0.9, Some(0.7)));
    assert_eq!(a.thresholds.overrides.len(), 3);
    assert_eq!(a.thresholds.overrides[0].prop, "AlarmState");
    assert_eq!((a.thresholds.overrides[0].equals, a.thresholds.overrides[0].accept), (1, 0.8));
    assert_eq!(a.components[0].class, 1);
    assert_eq!(a.components[0].start_end.as_deref(), Some("fc_output_se"));
    assert_eq!(spotter(&s, "STOP", false).components[0].class, 2);
    assert_eq!(s.words, vec![(0, "<eps>".into()), (1, "ALEXA".into()), (2, "STOP".into())]);
    let m = s.machine.as_ref().unwrap();
    assert_eq!(m.states[m.initial].name, "sleep");
    let awake = m.states.iter().find(|x| x.name == "awake").unwrap();
    assert_eq!(awake.timeout, Some((175, m.initial)));
    assert_eq!(s.checksum.len(), 16);
}

#[test]
fn loads_synthetic_davs() {
    let d = davs("ok");
    let s = load_set(&d).unwrap();
    assert_eq!(s.stages.len(), 3);
    let names: Vec<&str> = s.stages.iter().map(|x| x.name.as_str()).collect();
    assert_eq!(names, ["WW_encoder", "ww_stop", "ww_alexa"]);
    let fps: Vec<usize> = s.stages.iter().map(|x| x.frames_per_step).collect();
    assert_eq!(fps, [8, 1, 1]);
    let enc = &s.stages[0];
    assert_eq!(enc.inputs, vec![StageInput::Audio("input_136".into())]);
    assert_eq!(enc.outputs, vec![("encoder_embeddings".into(), "encoder_output".into())]);
    assert!(enc.primed);
    assert_eq!(enc.feat_dim, 64);
    let stop = &s.stages[1];
    assert_eq!(
        stop.inputs,
        vec![StageInput::Artifact { input: "input_6".into(), artifact: "encoder_output".into() }]
    );
    assert_eq!(stop.feat_dim, 16);
    assert!(stop.primed && stop.lstm.is_empty());
    let alexa = &s.stages[2];
    assert_eq!(alexa.lstm.len(), 4);
    assert!(!alexa.primed && alexa.recurrent.is_empty());
    assert_eq!(s.spotters.len(), 4);
    let a = spotter(&s, "ALEXA", false);
    assert_eq!(a.scorer, "ww_alexa");
    assert_eq!(a.lock, 132);
    assert_eq!(a.thresholds.probabilistic, Some((0.5, 0.001, 50)));
    assert_eq!((a.thresholds.min_accept, a.thresholds.max_accept), (Some(0.85), Some(0.93)));
    assert!(matches!(a.classifier, ClassifierSpec::MovingAverageMaxPool { window: 1, trim_tail: 0 }));
    assert_eq!(spotter(&s, "ALEXA", true).components[0].component, "ww_vrf_scores");
    let h = spotter(&s, "HEY_DISNEY", false);
    assert!(matches!(&h.classifier, ClassifierSpec::WeightedGeometricAverage { weights } if weights == &[1.0, 1.0]));
    assert_eq!(h.smoothers, vec![0.0, 0.3]);
    let hist: Vec<Option<usize>> = h.components.iter().map(|c| c.history).collect();
    assert_eq!(hist, [Some(100), Some(1)]);
    assert!(matches!(spotter(&s, "STOP", false).classifier, ClassifierSpec::ExponentialDecay { rate } if rate == 0.6));
    let m = s.machine.as_ref().unwrap();
    assert_eq!(m.states[m.initial].name, "sleep");
    assert_eq!(m.states[m.initial].rules.len(), 2);
}

/// Binds every graph input the loader named: audio at one step, artifacts at their publisher's shape, LSTM states
/// at their declared shape; recurrent buffers are kept initializers.
fn load_stage(st: &StageSpec, fe_bins: usize) -> (Arc<Graph>, Vec<(String, Vec<usize>)>) {
    let fw = st.inputs.iter().any(|i| matches!(i, StageInput::Audio(n) if n == "input_feats"));
    let mut bind: Vec<(String, Vec<usize>)> = Vec::new();
    for i in &st.inputs {
        match i {
            StageInput::Audio(n) if fw => bind.push((n.clone(), vec![1, st.frames_per_step, fe_bins, 1])),
            StageInput::Audio(n) => bind.push((n.clone(), vec![1, st.frames_per_step, fe_bins])),
            StageInput::Artifact { input, .. } => bind.push((input.clone(), vec![1, 1, st.feat_dim])),
        }
    }
    let lanes = if fw { 25 } else { 16 };
    for (i, _) in &st.lstm {
        let hid = if i.starts_with("alignments") { 4 } else { 8 };
        bind.push((i.clone(), vec![lanes, hid]));
    }
    let b: Vec<(&str, &[usize])> = bind.iter().map(|(n, d)| (n.as_str(), &d[..])).collect();
    let keep: Vec<&str> = st.recurrent.iter().map(|(i, _)| i.as_str()).collect();
    let g = Graph::load(&st.graph, &b, &keep).unwrap_or_else(|e| panic!("{}: {e}", st.name));
    (Arc::new(g), bind)
}

fn run_set(s: &SetSpec, seed: u64) -> Vec<(String, Vec<f32>)> {
    let mut rng = Rng(seed);
    let mut outs = Vec::new();
    for st in &s.stages {
        let (g, bind) = load_stage(st, s.fe.bins);
        let mut sess = Session::new(g.clone());
        for (n, _) in &bind {
            for v in sess.input_mut(g.input_index(n).unwrap()) {
                *v = (rng.next_u32() % 2001) as f32 / 1000.0 - 1.0;
            }
        }
        sess.run();
        for (comp, _) in &st.outputs {
            let o = sess.output(g.output_index(comp).unwrap()).to_vec();
            assert!(o.iter().all(|x| x.is_finite()), "{}/{comp}", st.name);
            outs.push((format!("{}/{comp}", st.name), o));
        }
    }
    outs
}

#[test]
fn synthetic_graphs_run() {
    for d in [fw("run"), davs("run")] {
        let s = load_set(&d).unwrap();
        let outs = run_set(&s, 3);
        assert!(!outs.is_empty());
        for (k, v) in &outs {
            if k.ends_with("softmax_output_ww") || k.ends_with("_scores") || k.ends_with("posteriors") {
                let classes = if k.ends_with("posteriors") { 4 } else { 3 };
                for row in v.chunks(classes) {
                    let sum: f32 = row.iter().sum();
                    assert!((sum - 1.0).abs() < 1e-4, "{k}: row sums to {sum}");
                }
            }
        }
    }
}

#[test]
fn force_score_pins_class_one() {
    for (fwset, d) in [(true, TempDir::new("wakeset-force-fw")), (false, TempDir::new("wakeset-force-davs"))] {
        let o = WakeSetOpts { seed: 11, force_score: Some(0.99) };
        if fwset { write_fw_set(&d, &o) } else { write_davs_set(&d, &o) }
        let s = load_set(&d).unwrap();
        for seed in [1, 2] {
            for (k, v) in run_set(&s, seed) {
                let classes = match () {
                    _ if k.ends_with("softmax_output_ww") || k.ends_with("_scores") => 3,
                    _ if k.ends_with("posteriors") => 4,
                    _ => continue,
                };
                for row in v.chunks(classes) {
                    assert!((row[1] - 0.99).abs() < 1e-3, "{k}: class 1 = {}", row[1]);
                }
            }
        }
    }
}

#[test]
fn include_chain_override() {
    let d = davs("include");
    let top = d.join("pryon.config");
    let s = std::fs::read_to_string(&top).unwrap();
    // set before the front-end include: the include's later value wins
    std::fs::write(&top, format!("fe.mel_fbank.low_freq = 50\n{s}INCLUDE \"override/pryon.config\" # last\n")).unwrap();
    std::fs::create_dir_all(d.join("override")).unwrap();
    std::fs::write(d.join("override/pryon.config"), "fe.mel_fbank.high_freq = 7000\n").unwrap();
    let s = load_set(&d).unwrap();
    assert_eq!((s.fe.low_hz, s.fe.high_hz), (80.0, 7000.0));
}

#[test]
fn include_cycle_named() {
    let d = davs("cycle");
    append(&d.join("encoder/pryon.config"), "INCLUDE \"../pryon.config\"\n");
    let m = message(load_set(&d));
    assert!(m.contains("cycle") && m.contains("encoder/pryon.config"), "{m}");
}

#[test]
fn missing_include_named() {
    let d = davs("missing");
    append(&d.join("encoder/pryon.config"), "INCLUDE \"nope/pryon.config\"\n");
    let m = message(load_set(&d));
    assert!(m.contains("nope/pryon.config") && m.contains("encoder/pryon.config"), "{m}");
}

#[test]
fn unknown_field_named() {
    let d = davs("unknown");
    let kw = d.join("BDPGeneratedFiles/kw.cfg.json");
    edit(&kw, r#""lock-period": 100 }"#, r#""lock-period": 100, "color": "red" }"#);
    let (file, what) = unsupported(load_set(&d));
    assert!(file.ends_with("kw.cfg.json"), "{}", file.display());
    assert!(what.contains("keyword-orchestra.keyword-spotters[0].spotter.color"), "{what}");
}

#[test]
fn pinned_field_other_value() {
    let d = davs("pinned");
    edit(&d.join("BDPGeneratedFiles/kw.cfg.json"), r#""buffer-size": 200"#, r#""buffer-size": 100"#);
    let (file, what) = unsupported(load_set(&d));
    assert!(file.ends_with("kw.cfg.json"));
    assert!(what.contains("keyword-orchestra.buffer-size") && what.contains("100"), "{what}");
}

#[test]
fn machine_unknown_keyword() {
    let d = davs("machine");
    let (from, to) = (r#""HEY_DISNEY", "next": "awake" } ] }"#, r#""HEY_MICKEY", "next": "awake" } ] }"#);
    edit(&d.join("BDPGeneratedFiles/op.cfg.json"), from, to);
    let (file, what) = unsupported(load_set(&d));
    assert!(file.ends_with("op.cfg.json"));
    assert!(what.contains("HEY_MICKEY"), "{what}");
}

#[test]
fn fe_unsupported_key() {
    let d = fw("fe");
    edit(&d.join("pryon.config"), "\"povey\"", "\"hamming\"");
    let (file, what) = unsupported(load_set(&d));
    assert!(file.ends_with("pryon.config"));
    assert!(what.contains("fe.audio_analysis.window_type") && what.contains("hamming"), "{what}");
    let d = fw("fe-unknown");
    append(&d.join("pryon.config"), "fe.feat.splice = 3\n");
    let (_, what) = unsupported(load_set(&d));
    assert!(what.contains("fe.feat.splice"), "{what}");
}

#[test]
fn unreferenced_decoders_not_opened() {
    let d = davs("unref");
    for sub in ["dvaddecoder", "siddecoder"] {
        std::fs::remove_dir_all(d.join(sub)).unwrap();
        std::fs::create_dir_all(d.join(sub)).unwrap();
        std::fs::write(d.join(sub).join("pryon.config"), "").unwrap();
    }
    let s = load_set(&d).unwrap();
    assert!(s.stages.iter().all(|x| x.name != "dvad" && x.name != "sid"));
}

fn fnv_tree(root: &Path) -> String {
    fn walk(root: &Path, dir: &Path, out: &mut Vec<(String, PathBuf)>) {
        for e in std::fs::read_dir(dir).unwrap() {
            let p = e.unwrap().path();
            if p.is_dir() {
                walk(root, &p, out);
            } else {
                out.push((p.strip_prefix(root).unwrap().to_str().unwrap().to_string(), p));
            }
        }
    }
    let mut files = Vec::new();
    walk(root, root, &mut files);
    files.sort();
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for (rel, p) in files {
        for b in rel.bytes().chain(std::fs::read(p).unwrap()) {
            h = (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3);
        }
    }
    format!("{h:016x}")
}

#[test]
fn checksum_fallback() {
    let d = davs("checksum");
    let a = load_set(&d).unwrap().checksum;
    assert_eq!(a, fnv_tree(&d));
    append(&d.join("BDPGeneratedFiles/words.shrunk.txt"), "\n");
    assert_ne!(load_set(&d).unwrap().checksum, a);
    edit(&d.join("pryon.manifest"), r#""default_model_name""#, r#""checksum": "checksum.txt", "default_model_name""#);
    std::fs::write(d.join("checksum.txt"), "0123abcd\n").unwrap();
    assert_eq!(load_set(&d).unwrap().checksum, "0123abcd");
}

fn env_dir(var: &str) -> Option<PathBuf> {
    match std::env::var_os(var) {
        Some(v) => Some(PathBuf::from(v)),
        None => {
            println!("skipped: set {var}");
            None
        }
    }
}

#[test]
fn real_sets_load() {
    if let Some(dir) = env_dir("PRYON_WAKE_FW_DIR") {
        let s = load_set(&dir).unwrap_or_else(|e| panic!("firmware ALEXA: {e}"));
        assert_eq!(s.stages.len(), 1);
        assert_eq!(s.stages[0].frames_per_step, 6);
        assert!(!s.stages[0].primed);
        assert_eq!(s.spotters.len(), 2);
        let a = spotter(&s, "ALEXA", false);
        assert_eq!((a.thresholds.accept, a.thresholds.notify), (0.923, Some(0.69)));
        assert!(matches!(a.classifier, ClassifierSpec::MovingAverageMaxPool { window: 3, trim_tail: 8 }));
        assert_eq!(s.checksum, "74a26372ecb79fc6e5954d9f9fff29a3");
        println!("firmware ALEXA: {} spotters, checksum {}", s.spotters.len(), s.checksum);
    }
    let Some(root) = env_dir("PRYON_WAKE_SETS_DIR") else { return };
    for word in ["alexa", "amazon", "computer", "echo", "ziggy"] {
        let dir = root.join(format!("{word}-en-US"));
        let s = load_set(&dir).unwrap_or_else(|e| panic!("{word}-en-US: {e}"));
        assert_eq!(s.stages[0].frames_per_step, 8);
        assert!(s.stages[0].primed);
        assert!(s.stages[1..].iter().all(|x| x.frames_per_step == 1));
        println!("{word}-en-US: {} stages, {} spotters, checksum {}", s.stages.len(), s.spotters.len(), s.checksum);
        if word != "alexa" {
            continue;
        }
        let a = spotter(&s, "ALEXA", false);
        let t = &a.thresholds;
        assert_eq!((t.accept, t.min_accept, t.max_accept, t.notify), (0.921, Some(0.845), Some(0.929), Some(0.278)));
        assert_eq!(t.probabilistic, Some((0.527, 0.001, 50)));
        assert_eq!(a.lock, 132);
        assert_eq!(spotter(&s, "ALEXA", true).thresholds.accept, 0.898);
        let stop = &spotter(&s, "STOP", false).classifier;
        assert!(matches!(stop, ClassifierSpec::ExponentialDecay { rate } if *rate == 0.6));
        let h = spotter(&s, "HEY_DISNEY", false);
        let wga = ClassifierSpec::WeightedGeometricAverage { weights: vec![1.0, 1.0] };
        assert_eq!(h.classifier, wga);
        assert_eq!(h.smoothers, vec![0.0, 0.3]);
        let hist: Vec<Option<usize>> = h.components.iter().map(|c| c.history).collect();
        assert_eq!(hist, [Some(100), Some(1)]);
        let names: Vec<&str> = s.stages.iter().map(|x| x.name.as_str()).collect();
        assert_eq!(names, ["WW_encoder", "ww_stop", "ww_alexa"]);
    }
}

#[test]
fn de_sets_refused() {
    let Some(root) = env_dir("PRYON_WAKE_SETS_DIR") else { return };
    let expect = [
        (
            "alexa-de-DE",
            "BDPGeneratedFiles/kw.cfg.json",
            "field keyword-orchestra.keyword-spotters[0].spotter.probabilistic-accept-lock-period",
        ),
        ("amazon-de-DE", "pryon.config", "key scorer.initialize.publish_frame_count"),
    ];
    for (set, file, what) in expect {
        let dir = root.join(set);
        if !dir.join("pryon.manifest").is_file() {
            println!("skipped: {set} not extracted under PRYON_WAKE_SETS_DIR");
            continue;
        }
        let (f, w) = unsupported(load_set(&dir));
        println!("{set}: {}: unsupported {w}", f.display());
        assert_eq!(f, dir.join(file), "{set}");
        assert_eq!(w, what, "{set}");
    }
}

#[test]
fn paths_confined_to_the_set() {
    let d = davs("escape-include");
    append(&d.join("encoder/pryon.config"), "INCLUDE \"../../outside.config\"\n");
    let (file, what) = unsupported(load_set(&d));
    assert!(file.ends_with("encoder/pryon.config") && what.contains("INCLUDE") && what.contains("outside"), "{what}");

    let d = davs("escape-graph");
    let nem = d.join("alexa-wakeworddecoder/nemort_scorer_config_decoder_alexa.json");
    edit(&nem, r#""det_vrf_multi_lstm.int8.onnx""#, r#""/etc/hostname""#);
    let (file, what) = unsupported(load_set(&d));
    assert_eq!(file, nem);
    assert!(what.contains("model_filepath") && what.contains("outside the set directory"), "{what}");

    let d = davs("escape-key");
    edit(&d.join("BDPGeneratedFiles/pryon.config"), r#""words.shrunk.txt""#, r#""../../words.shrunk.txt""#);
    let (_, what) = unsupported(load_set(&d));
    assert!(what.contains("id2word.mapping_filepath") && what.contains("outside"), "{what}");

    let d = davs("symlink-read");
    let words = d.join("BDPGeneratedFiles/words.shrunk.txt");
    std::fs::rename(&words, d.join("BDPGeneratedFiles/words.real")).unwrap();
    std::os::unix::fs::symlink("words.real", &words).unwrap();
    let (_, what) = unsupported(load_set(&d));
    assert!(what.contains("id2word.mapping_filepath") && what.contains("symlink"), "{what}");

    // an unread symlink still changes nothing silently: the fallback checksum refuses it
    let d = davs("symlink-checksum");
    std::os::unix::fs::symlink("BDPGeneratedFiles/words.shrunk.txt", d.join("notes.txt")).unwrap();
    let (file, what) = unsupported(load_set(&d));
    assert!(file.ends_with("notes.txt") && what.contains("symlink"), "{what}");
}

#[test]
fn cross_file_checks_named() {
    let kw = "BDPGeneratedFiles/kw.cfg.json";
    let dec = "BDPGeneratedFiles/decoders.cfg.json";
    let ta = "encoder/tensor_artifacts.json";
    let cases: [(&str, &str, &str, &str); 9] = [
        (kw, r#""scorer-index": 1"#, r#""scorer-index": 9"#, "index 9"),
        (kw, r#""name": "ww_det_scores""#, r#""name": "ww_det""#, "ww_det is not an output of scorer ww_alexa"),
        (kw, r#""exponential-decay", "decay-rate": 0.6"#, r#""median", "decay-rate": 0.6"#, "classifier.type = median"),
        (kw, r#""probabilistic-accept-lock-period": 50, "#, "", "probabilistic-accept-* given partially"),
        (dec, r#""scorer.num_dnn_outputs": 4"#, r#""scorer.num_dnn_outputs": 5"#, "num_dnn_outputs = 5"),
        (dec, r#""scorer.initialize.frame_count": 1,
      "scorer.feat.dimensions": 16,
      "scorer.num_dnn_outputs": 4"#, r#""scorer.initialize.frame_count": 2,
      "scorer.feat.dimensions": 16,
      "scorer.num_dnn_outputs": 4"#, "default_skip_frame_count = 0 with frame_count 2"),
        (ta, r#""name": "encoder_output""#, r#""name": "encoder_out""#, "encoder_output is published but not in"),
        (ta, r#""in-memory""#, r#""disk""#, "tensor_artifacts[0].format = disk"),
        (ta, r#""tensor_artifacts_list_config""#, r#""other_config""#, "object_type_name = other_config"),
    ];
    for (i, (file, from, to, want)) in cases.iter().enumerate() {
        let d = davs(&format!("cross{i}"));
        edit(&d.join(file), from, to);
        let (_, what) = unsupported(load_set(&d));
        assert!(what.contains(want), "case {i}: {what}");
    }
}
