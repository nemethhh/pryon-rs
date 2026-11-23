//! Wake Spotter API and parity with libpryon's wake events (committed fixtures on synthetic clips, env-gated corpus).

use pryon::wake::{load_set, Detection, DetectionKind, Spotter, SpotterModel, SpotterOptions};
use pryon_testkit::compare::{
    compare as compare_events, compare_gated, oracle_cache_key, pad_clip, pushed, Diff, Ev, GateTally, Limits, Tally,
    FW_GATE_MAX_FRAMES, LEAD_MS, TAIL_MS,
};
use pryon_testkit::oracle::{deterministic_set, parse_wake, restarts, OracleCmd, OracleEnv, OracleEvent, OracleKind};
use pryon_testkit::wakeset::{write_davs_set, write_fw_set, WakeSetOpts};
use pryon_testkit::{lsb_stream, lsb_stream_with, synth, Clip, LsbFrame, TempDir};
use std::path::{Path, PathBuf};
use std::sync::Arc;

fn synth_model(fw: bool, force: Option<f32>, tag: &str) -> (TempDir, Arc<SpotterModel>) {
    let d = TempDir::new(&format!("wake-{tag}"));
    let o = WakeSetOpts { seed: 1, force_score: force };
    if fw { write_fw_set(&d, &o) } else { write_davs_set(&d, &o) }
    let m = Arc::new(SpotterModel::load(&d).unwrap());
    (d, m)
}

fn run(sp: &mut Spotter, pcm: &[i16], chunk: usize) -> Vec<Detection> {
    let mut out = Vec::new();
    for c in pcm.chunks(chunk) {
        sp.feed(c, &mut out);
    }
    out
}

fn bits(d: &[Detection]) -> Vec<(Detection, u32)> {
    d.iter().map(|x| (x.clone(), x.score.to_bits())).collect()
}

#[test]
fn wake_chunking_invariant() {
    let pcm = &synth(Clip::Noise)[..16000 * 12 + 77];
    for fw in [true, false] {
        for force in [Some(0.99), Some(0.8), None] {
            let (_d, m) = synth_model(fw, force, "chunks");
            let whole = run(&mut Spotter::new(m.clone(), SpotterOptions::default()).unwrap(), pcm, pcm.len());
            if force == Some(0.99) {
                assert!(!whole.is_empty(), "fw {fw}: no events");
            }
            // a constant score above notify never ends a single-graph near-miss run
            if force == Some(0.8) && !fw {
                assert!(whole.iter().any(|d| d.kind == DetectionKind::NearMiss), "no near miss");
            }
            for chunk in [1, 7, 160, 1280, 3333] {
                let got = run(&mut Spotter::new(m.clone(), SpotterOptions::default()).unwrap(), pcm, chunk);
                assert_eq!(bits(&got), bits(&whole), "fw {fw} force {force:?} chunk {chunk}");
            }
        }
    }
}

#[test]
fn reset_keeps_sample_clock() {
    let pcm = synth(Clip::Noise);
    for fw in [true, false] {
        let (_d, m) = synth_model(fw, Some(0.99), "reset");
        let first = run(&mut Spotter::new(m.clone(), SpotterOptions::default()).unwrap(), &pcm[..48000], 1600);
        assert!(first.iter().any(|d| d.kind == DetectionKind::Accept), "fw {fw}: no accept in 3 s");
        // 47999: the reset lands inside a front-end frame.
        for at in [48000usize, 47_999] {
            let mut sp = Spotter::new(m.clone(), SpotterOptions::default()).unwrap();
            let mut out = run(&mut sp, &pcm[..at], 1600);
            sp.reset(&mut out);
            let before = out.len();
            for c in pcm[..48000].chunks(1600) {
                sp.feed(c, &mut out);
            }
            let want: Vec<Detection> = first
                .iter()
                .map(|d| {
                    let (begin_sample, end_sample) = (d.begin_sample + at as u64, d.end_sample + at as u64);
                    Detection { begin_sample, end_sample, ..d.clone() }
                })
                .collect();
            assert_eq!(bits(&out[before..]), bits(&want), "fw {fw} reset at {at}");
        }
    }
}

#[test]
fn two_spotters_share_model() {
    fn send_sync<T: Send + Sync>() {}
    send_sync::<SpotterModel>();
    fn send<T: Send>() {}
    send::<Spotter>();
    let pcm = &synth(Clip::Tones)[..16000 * 6];
    for fw in [true, false] {
        let (_d, m) = synth_model(fw, Some(0.99), "share");
        let results: Vec<Vec<Detection>> = std::thread::scope(|s| {
            let hs: Vec<_> = (0..2)
                .map(|_| {
                    let m = m.clone();
                    s.spawn(move || run(&mut Spotter::new(m, SpotterOptions::default()).unwrap(), pcm, 1280))
                })
                .collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        assert!(!results[0].is_empty());
        assert_eq!(bits(&results[0]), bits(&results[1]), "fw {fw}");
    }
}

#[test]
fn keywords_once_each() {
    let (_d, davs) = synth_model(false, None, "kw");
    assert_eq!(davs.keywords(), ["ALEXA", "HEY_DISNEY", "STOP"]);
    let sp = Spotter::new(davs.clone(), SpotterOptions::default()).unwrap();
    assert_eq!(sp.keywords(), davs.keywords());
    let (_f, fw) = synth_model(true, None, "kw");
    assert_eq!(fw.keywords(), ["ALEXA", "STOP"]);
    assert_eq!(fw.checksum(), load_set(&_f).unwrap().checksum);
}

fn kind_of(k: DetectionKind) -> OracleKind {
    match k {
        DetectionKind::Accept => OracleKind::Accept,
        DetectionKind::AcceptAuxiliary => OracleKind::AcceptAuxiliary,
        DetectionKind::NearMiss => OracleKind::NearMiss,
    }
}

fn from_rust(keywords: &[String], d: &[Detection]) -> Vec<Ev> {
    d.iter()
        .map(|d| Ev {
            keyword: keywords[d.keyword].clone(),
            kind: kind_of(d.kind),
            score: d.score,
            begin: d.begin_sample,
            end: d.end_sample,
        })
        .collect()
}

/// Thresholds a score is compared against, per keyword.
fn thresholds(dir: &Path) -> Limits {
    let mut l = Limits::default();
    for s in load_set(dir).unwrap().spotters {
        let t = &s.thresholds;
        let ov: Vec<f32> = t.overrides.iter().map(|o| o.accept).collect();
        l.add(&s.name, t.accept, t.notify, &ov);
    }
    l
}

fn compare(oracle: &[OracleEvent], engine: &[Ev], thr: &Limits, diffs: &mut Vec<String>) -> Tally {
    let mut d: Vec<Diff> = Vec::new();
    let t = compare_events(oracle, engine, thr, true, &mut d);
    diffs.extend(d.iter().map(Diff::to_string));
    t
}

fn env_dir(var: &str) -> Option<PathBuf> {
    match std::env::var_os(var) {
        Some(d) => Some(PathBuf::from(d)),
        None => {
            println!("skipped: set {var}");
            None
        }
    }
}

/// The real set a fixture directory was made with.
fn real_set(set: &str) -> Option<PathBuf> {
    match set {
        "fw" => env_dir("PRYON_WAKE_FW_DIR"),
        "alexa" => env_dir("PRYON_WAKE_SETS_DIR").map(|d| d.join("alexa-en-US")),
        _ => unreachable!(),
    }
}

fn spot(m: &Arc<SpotterModel>, pcm: &[i16], chunk: usize) -> Vec<Ev> {
    let d = run(&mut Spotter::new(m.clone(), SpotterOptions::default()).unwrap(), pcm, chunk);
    from_rust(m.keywords(), &d)
}

fn oracle_synth(set: &str, clip: Clip) {
    let Some(dir) = real_set(set) else { return };
    let m = Arc::new(SpotterModel::load(&dir).unwrap());
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("tests/fixtures/wake/{set}/{}.txt", clip.name()));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let oracle = parse_wake(&text).unwrap();
    let mut diffs = Vec::new();
    let t = compare(&oracle, &spot(&m, &synth(clip), 1600), &thresholds(&dir), &mut diffs);
    println!("{set} {}: {}", clip.name(), t.line());
    assert!(t.clean(), "{set} {}:\n{}", clip.name(), diffs.join("\n"));
}

macro_rules! oracle_synth_tests {
    ($($name:ident: $set:literal, $clip:expr;)*) => {
        $(#[test] fn $name() { oracle_synth($set, $clip); })*
    };
}

oracle_synth_tests! {
    oracle_synth_fw_noise: "fw", Clip::Noise;
    oracle_synth_fw_tones: "fw", Clip::Tones;
    oracle_synth_fw_gap49: "fw", Clip::Gap49;
    oracle_synth_fw_sparse: "fw", Clip::Sparse;
    oracle_synth_fw_zeros: "fw", Clip::Zeros;
    oracle_synth_fw_quiet: "fw", Clip::Quiet;
    oracle_synth_fw_extremes: "fw", Clip::Extremes;
    oracle_synth_fw_quiet_tones: "fw", Clip::QuietTones;
    oracle_synth_alexa_noise: "alexa", Clip::Noise;
    oracle_synth_alexa_tones: "alexa", Clip::Tones;
    oracle_synth_alexa_gap49: "alexa", Clip::Gap49;
    oracle_synth_alexa_sparse: "alexa", Clip::Sparse;
    oracle_synth_alexa_zeros: "alexa", Clip::Zeros;
    oracle_synth_alexa_quiet: "alexa", Clip::Quiet;
    oracle_synth_alexa_extremes: "alexa", Clip::Extremes;
    oracle_synth_alexa_quiet_tones: "alexa", Clip::QuietTones;
}

fn read_clip(p: &Path) -> Vec<i16> {
    let b = std::fs::read(p).unwrap();
    b.chunks_exact(2).map(|x| i16::from_le_bytes([x[0], x[1]])).collect()
}

/// Corpus clips grouped by directory: (group, clip stem, path).
fn corpus(root: &Path) -> Vec<(String, String, PathBuf)> {
    let mut out = Vec::new();
    for g in std::fs::read_dir(root).unwrap() {
        let g = g.unwrap().path();
        if !g.is_dir() {
            continue;
        }
        for c in std::fs::read_dir(&g).unwrap() {
            let c = c.unwrap().path();
            if c.extension().is_some_and(|e| e == "raw") {
                let group = g.file_name().unwrap().to_string_lossy().into_owned();
                out.push((group, c.file_stem().unwrap().to_string_lossy().into_owned(), c));
            }
        }
    }
    out.sort();
    out
}

/// Runs `f` over `items` on every core.
fn par_map<T: Sync, R: Send>(items: &[T], f: impl Fn(&T) -> R + Sync) -> Vec<R> {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;
    let next = AtomicUsize::new(0);
    let out: Mutex<Vec<Option<R>>> = Mutex::new((0..items.len()).map(|_| None).collect());
    let n = std::thread::available_parallelism().map_or(4, |n| n.get());
    std::thread::scope(|s| {
        for _ in 0..n {
            s.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                if i >= items.len() {
                    break;
                }
                let r = f(&items[i]);
                out.lock().unwrap()[i] = Some(r);
            });
        }
    });
    out.into_inner().unwrap().into_iter().map(Option::unwrap).collect()
}

/// A `pryon-cli validate` pass: clip-sample commands and the corpus groups it ran on (`local/task12/final.sh`).
struct Pass {
    name: &'static str,
    props: &'static [(u64, &'static str, i32)],
    ends: &'static [u64],
    groups: &'static [&'static str],
}

const PASSES: &[Pass] = &[
    Pass { name: "base", props: &[], ends: &[], groups: &[] },
    Pass {
        name: "AlarmState=1",
        props: &[(0, "AlarmState", 1)],
        ends: &[],
        groups: &["alexa_real", "tts_alexa", "tts_stop", "tts_disney"],
    },
    Pass { name: "AudioPlayerState=1@8000", props: &[(8000, "AudioPlayerState", 1)], ends: &[], groups: &["alexa_real"] },
    Pass { name: "-e 8000", props: &[], ends: &[8000], groups: &["alexa_real"] },
    Pass { name: "-e 28000", props: &[], ends: &[28000], groups: &["tts_stop", "tts_disney"] },
];

/// Every validate pass vs the cache, scores as exact strings; the only allowance is libpryon's start gate on a set
/// whose state it leaves uninitialised (`compare_gated`), counted per pass.
#[test]
fn oracle_corpus_validate_cache() {
    let Some(root) = env_dir("PRYON_WAKE_CORPUS") else { return };
    let partial_ok = std::env::var_os("PRYON_CORPUS_PARTIAL").is_some();
    let sets: Vec<(&str, PathBuf)> = ["fw", "alexa"].iter().filter_map(|s| real_set(s).map(|d| (*s, d))).collect();
    let clips = corpus(&root);
    assert!(!clips.is_empty(), "{}: no <group>/<clip>.raw", root.display());
    let cache = root.parent().unwrap().join("validate-v2");
    let lead = u64::from(LEAD_MS) * 16;
    let shift = |at: u64| if at == 0 { 0 } else { at + lead };
    let mut failures = Vec::new();
    for (name, dir) in &sets {
        let m = Arc::new(SpotterModel::load(dir).unwrap());
        let thr = thresholds(dir);
        let sum = m.checksum().to_string();
        let spec = load_set(dir).unwrap();
        let gate = spec.uninitialised_state().then(|| FW_GATE_MAX_FRAMES + spec.longest_lock());
        assert_eq!(gate.is_some(), *name == "fw", "{name}: libpryon leaves only the firmware state uninitialised");
        for pass in PASSES {
            let mut oracle_cmds: Vec<OracleCmd> = Vec::new();
            let mut cmds: Vec<Cmd> = Vec::new();
            for &(at, n, v) in pass.props {
                oracle_cmds.push(OracleCmd::Property { at: shift(at), name: n.into(), value: v });
                cmds.push(Cmd::Prop(shift(at), n, v));
            }
            for &at in pass.ends {
                oracle_cmds.push(OracleCmd::SessionEnd { at: shift(at) });
                cmds.push(Cmd::End(shift(at)));
            }
            let ends: Vec<u64> = pass.ends.iter().map(|&at| shift(at)).collect();
            let chosen: Vec<_> = clips.iter().filter(|c| pass.groups.is_empty() || pass.groups.contains(&c.0.as_str())).collect();
            let results = par_map(&chosen, |(group, clip, path)| {
                let pcm = pad_clip(&read_clip(path), LEAD_MS, TAIL_MS);
                let bytes: Vec<u8> = pcm.iter().flat_map(|x| x.to_le_bytes()).collect();
                let key = oracle_cache_key(pryon_testkit::fnv1a64(&bytes), &oracle_cmds, 1.0);
                let cached = cache.join(&sum).join(format!("{key:016x}.txt"));
                let Ok(text) = std::fs::read_to_string(&cached) else { return None };
                assert_eq!(pushed(&text), Some(pcm.len()), "{}: oracle not fed the whole clip", cached.display());
                let oracle = parse_wake(&text).unwrap_or_else(|e| panic!("{}: {e}", cached.display()));
                let (rust, engine_restarts) = spot_cmds(&m, &pcm, &cmds);
                let (o_restarts, kws) = (restarts(&text), m.keywords());
                let e_restarts = &engine_restarts;
                let g = compare_gated(kws, &oracle, &rust, &o_restarts, e_restarts, &ends, gate, lead, &thr, true);
                let (mut t, mut skip, mut bad) = (Tally::default(), GateTally::default(), Vec::new());
                for k in g.per_kw.values() {
                    t.add(&k.tally);
                    skip.add(&k.gate);
                    bad.extend(k.diffs.iter().map(Diff::to_string));
                }
                if let Some((a, b)) = g.restart_mismatch {
                    bad.push(format!("restart samples differ: oracle {a:?} engine {b:?}"));
                }
                Some((format!("{group}/{clip}"), t, bad, skip))
            });
            let missing = results.iter().filter(|r| r.is_none()).count();
            assert!(missing < chosen.len(), "{name} [{}]: nothing cached", pass.name);
            if missing > 0 && !partial_ok {
                failures.push(format!("{name} [{}]: {missing} of {} clips uncached (PRYON_CORPUS_PARTIAL=1 allows)", pass.name, chosen.len()));
            }
            let (mut total, mut skip) = (Tally::default(), GateTally::default());
            for (clip, t, bad, g) in results.into_iter().flatten() {
                total.add(&t);
                skip.add(&g);
                if !bad.is_empty() {
                    failures.push(format!("{name} [{}] {clip}:\n  {}", pass.name, bad.join("\n  ")));
                }
            }
            println!(
                "{name} [{}]: compared {} events; gate allowance: {} in a gate window (one-sided or a pair with both ends \
                 there), {} engine events inside the gate, {} after a diverging restart; {missing} clips uncached | {}",
                pass.name,
                total.oracle + total.engine - skip.window,
                skip.window,
                skip.inside,
                skip.after_restart,
                total.line()
            );
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// How a probe edited a config: replace every occurrence, the first one, or cut from the first string up to the
/// second.
#[derive(Clone, Copy)]
enum Edit<'a> {
    All(&'a str, &'a str),
    Once(&'a str, &'a str),
    Cut(&'a str, &'a str),
}

/// The synthetic set's kw.cfg.json or op.cfg.json.
#[derive(Clone, Copy)]
enum Cfg {
    Kw,
    Op,
}

/// A client property or a session end before the given sample (pryon_test -P / -E).
#[derive(Clone, Copy)]
enum Cmd<'a> {
    Prop(u64, &'a str, i32),
    End(u64),
}

/// A synthetic set with string edits, as `local/probe11*.sh` / `local/probe12/fixtures.sh` wrote it.
fn probe_set(name: &str, fw: bool, force: f32, edits: &[(Cfg, Edit)]) -> (TempDir, Arc<SpotterModel>) {
    let d = TempDir::new(&format!("probe-{name}"));
    let o = WakeSetOpts { seed: 1, force_score: Some(force) };
    if fw { write_fw_set(&d, &o) } else { write_davs_set(&d, &o) }
    let base = if fw { d.join("") } else { d.join("BDPGeneratedFiles") };
    for &(cfg, e) in edits {
        let path = base.join(match cfg {
            Cfg::Kw => "kw.cfg.json",
            Cfg::Op => "op.cfg.json",
        });
        let mut text = std::fs::read_to_string(&path).unwrap();
        let (Edit::All(a, _) | Edit::Once(a, _) | Edit::Cut(a, _)) = e;
        assert!(text.contains(a), "{name}: no {a}");
        text = match e {
            Edit::All(a, b) => text.replace(a, b),
            Edit::Once(a, b) => text.replacen(a, b, 1),
            Edit::Cut(a, b) => {
                let i = text.find(a).unwrap();
                let j = i + text[i..].find(b).unwrap();
                format!("{}{}", &text[..i], &text[j..])
            }
        };
        std::fs::write(&path, text).unwrap();
    }
    let m = Arc::new(SpotterModel::load(&d).unwrap());
    (d, m)
}

/// Rust's events and restart samples, the commands applied before their samples in pryon_test's order.
fn spot_cmds(m: &Arc<SpotterModel>, pcm: &[i16], cmds: &[Cmd]) -> (Vec<Ev>, Vec<u64>) {
    let (d, restarts) = detect_cmds(m, pcm, cmds, SpotterOptions::default());
    (from_rust(m.keywords(), &d), restarts)
}

fn detect_cmds(m: &Arc<SpotterModel>, pcm: &[i16], cmds: &[Cmd], opts: SpotterOptions) -> (Vec<Detection>, Vec<u64>) {
    let at = |c: &Cmd| match *c {
        Cmd::Prop(s, ..) | Cmd::End(s) => s as usize,
    };
    let mut cmds: Vec<&Cmd> = cmds.iter().collect();
    cmds.sort_by_key(|c| at(c));
    let mut sp = Spotter::new(m.clone(), opts).unwrap();
    let (mut out, mut restarts, mut pos, mut next) = (Vec::new(), Vec::new(), 0usize, 0usize);
    loop {
        while next < cmds.len() && at(cmds[next]) <= pos {
            match *cmds[next] {
                Cmd::Prop(_, n, v) => sp.set_property(n, v),
                Cmd::End(_) => sp.reset(&mut out),
            }
            next += 1;
        }
        if pos == pcm.len() {
            break;
        }
        let end = (pos + 1600).min(cmds.get(next).map_or(pcm.len(), |c| at(c).min(pcm.len())));
        let origin = sp.origin();
        sp.feed(&pcm[pos..end], &mut out);
        if sp.origin() != origin {
            restarts.push(sp.origin());
        }
        pos = end;
    }
    (out, restarts)
}

fn fixture(dir: &str, name: &str) -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("tests/fixtures/wake/{dir}/{name}.txt"));
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// Every event of `tests/fixtures/wake/<dir>/<name>.txt` (libpryon, -x 1, two identical runs), every keyword:
/// kind, begin, end and the printed score exact; returns Rust's events.
fn check_probe(dir: &str, name: &str, fw: bool, force: f32, edits: &[(Cfg, Edit)], cmds: &[Cmd], clip: Clip) -> Vec<Ev> {
    let (d, m) = probe_set(name, fw, force, edits);
    let text = fixture(dir, name);
    let oracle = parse_wake(&text).unwrap();
    let pcm = synth(clip);
    assert_eq!(pushed(&text), Some(pcm.len()), "{dir}/{name}: the oracle was not fed the whole clip");
    let rust = spot_cmds(&m, &pcm, cmds).0;
    let mut diff = Vec::new();
    let t = compare_events(&oracle, &rust, &thresholds(&d), true, &mut diff);
    println!("{dir}/{name}: {}", t.line());
    let diffs: Vec<String> = diff.iter().map(Diff::to_string).collect();
    assert!(t.clean(), "{dir}/{name}:\n{}", diffs.join("\n"));
    rust
}

/// One synthetic set with one kw.cfg field changed against libpryon's events on `Clip::Noise`, every keyword.
fn synth_variant(name: &str, fw: bool, force: f32, edits: &[Edit]) {
    let edits: Vec<(Cfg, Edit)> = edits.iter().map(|&e| (Cfg::Kw, e)).collect();
    check_probe("synth-variants", name, fw, force, &edits, &[], Clip::Noise);
}

/// The ALEXA verifier spotter, up to the end of the spotter list.
const VERIFIER: Edit = Edit::Cut(
    concat!(
        ",\n    { \"classifier\": { \"type\": \"moving-average-max-pool\", \"window-size\": 1 },\n",
        "      \"name\": \"ALEXA\", \"scorers\""
    ),
    "\n  ] }",
);

#[test]
fn synth_variant_fw_notify_rising_only() {
    synth_variant("fw_f08", true, 0.8, &[]);
}

#[test]
fn synth_variant_fw_notify_removed() {
    synth_variant("fw_f08_nonotify", true, 0.8, &[Edit::Once(r#", "notify-threshold": 0.7"#, "")]);
}

#[test]
fn synth_variant_fw_notify_lock_period() {
    let lock = Edit::Once(r#""lock-period": 100 } },"#, r#""lock-period": 100, "notify-lock-period": 40 } },"#);
    synth_variant("fw_nlock40", true, 0.8, &[lock]);
}

#[test]
fn synth_variant_fw_notify_lock_period_0_freezes() {
    let lock = Edit::Once(r#""lock-period": 100 } },"#, r#""lock-period": 100, "notify-lock-period": 0 } },"#);
    synth_variant("fw_nlock0", true, 0.8, &[lock]);
}

#[test]
fn synth_variant_fw_trim_tail_0() {
    let c = Edit::Once(r#""trim-tail": 8, "window-size": 3"#, r#""trim-tail": 0, "window-size": 3"#);
    synth_variant("fw_trim0", true, 0.99, &[c]);
}

#[test]
fn synth_variant_fw_window_1() {
    let c = Edit::Once(r#""trim-tail": 8, "window-size": 3"#, r#""trim-tail": 8, "window-size": 1"#);
    synth_variant("fw_window1", true, 0.99, &[c]);
}

#[test]
fn synth_variant_fw_lock_50() {
    synth_variant("fw_lock50", true, 0.99, &[Edit::All(r#""lock-period": 100"#, r#""lock-period": 50"#)]);
}

#[test]
fn synth_variant_fw_lock_0_freezes() {
    synth_variant("fw_lock0", true, 0.99, &[Edit::Once(r#""lock-period": 100"#, r#""lock-period": 0"#)]);
}

#[test]
fn synth_variant_davs_probabilistic_near_miss() {
    synth_variant("davs_f08", false, 0.8, &[]);
}

#[test]
fn synth_variant_davs_probabilistic_lock_0_freezes() {
    let lock = Edit::Once(
        r#""lock-period": 132 } },"#,
        r#""lock-period": 132, "probabilistic-accept-lock-period": 0 } },"#,
    );
    synth_variant("davs_plock0", false, 0.8, &[lock]);
}

#[test]
fn synth_variant_davs_notify_rising_only() {
    let p = Edit::All(r#""probabilistic-accept-threshold": 0.5"#, r#""probabilistic-accept-threshold": 0.95"#);
    synth_variant("davs_f08_noprob", false, 0.8, &[p]);
}

#[test]
fn synth_variant_davs_notify_removed() {
    synth_variant("davs_f08_nonotify", false, 0.8, &[Edit::Once(r#""notify-threshold": 0.3, "#, "")]);
}

#[test]
fn synth_variant_davs_verifier_accepts() {
    synth_variant("davs_f091", false, 0.91, &[]);
}

#[test]
fn synth_variant_davs_verifier_removed() {
    synth_variant("davs_noverifier", false, 0.91, &[VERIFIER]);
}

#[test]
fn synth_variant_davs_lock_50() {
    synth_variant("davs_lock50", false, 0.99, &[Edit::All(r#""lock-period": 132"#, r#""lock-period": 50"#)]);
}

/// The synthetic sets at force 0.99 on every synthetic clip (Task 9's probes): ALEXA, STOP after it while awake,
/// and the firmware's decoder restart after 12 accepts (pryon_test now goes on with a new decoder).
fn synth_set(fw: bool, clip: Clip) {
    check_probe(if fw { "synth-fw" } else { "synth-davs" }, clip.name(), fw, 0.99, &[], &[], clip);
}

macro_rules! synth_set_tests {
    ($($name:ident: $fw:literal, $clip:expr;)*) => {
        $(#[test] fn $name() { synth_set($fw, $clip); })*
    };
}

synth_set_tests! {
    synth_set_fw_noise: true, Clip::Noise;
    synth_set_fw_tones: true, Clip::Tones;
    synth_set_fw_gap49: true, Clip::Gap49;
    synth_set_fw_sparse: true, Clip::Sparse;
    synth_set_fw_zeros: true, Clip::Zeros;
    synth_set_fw_quiet: true, Clip::Quiet;
    synth_set_fw_extremes: true, Clip::Extremes;
    synth_set_fw_quiet_tones: true, Clip::QuietTones;
    synth_set_davs_noise: false, Clip::Noise;
    synth_set_davs_tones: false, Clip::Tones;
    synth_set_davs_gap49: false, Clip::Gap49;
    synth_set_davs_sparse: false, Clip::Sparse;
    synth_set_davs_zeros: false, Clip::Zeros;
    synth_set_davs_quiet: false, Clip::Quiet;
    synth_set_davs_extremes: false, Clip::Extremes;
    synth_set_davs_quiet_tones: false, Clip::QuietTones;
}

/// A `local/probe12/fixtures.sh` script on `Clip::Noise`.
fn script(name: &str, fw: bool, force: f32, edits: &[(Cfg, Edit)], cmds: &[Cmd]) -> Vec<Ev> {
    check_probe("synth-scripts", name, fw, force, edits, cmds, Clip::Noise)
}

const STOP_DECAY: &str = r#""classifier": { "type": "exponential-decay", "decay-rate": 0.6 }"#;
const STOP_ACCEPT: &str = r#""classification-thresholds": { "accept-threshold": 0.6 }"#;
const ALARM: Cmd = Cmd::Prop(0, "AlarmState", 1);

fn ev_line(e: &Ev) -> (&str, OracleKind, u64) {
    (e.keyword.as_str(), e.kind, e.end / 160)
}

#[test]
fn property_mid_stream_matches_oracle() {
    // AlarmState=1 lowers ALEXA to 0.7 (score 0.8): it counts from the first 10 ms block that starts at or after
    // the samples pushed when it was set; the step is scored with its last frame's properties
    let first_accept = |name: &str, cmds: &[Cmd]| {
        let r = script(name, false, 0.8, &[], cmds);
        r.iter().find(|e| e.kind == OracleKind::Accept).map(|e| e.end / 160)
    };
    assert_eq!(first_accept("pr_x0", &[ALARM]), Some(8));
    assert_eq!(first_accept("pr_x7520", &[Cmd::Prop(7520, "AlarmState", 1)]), Some(48));
    assert_eq!(first_accept("pr_x7521", &[Cmd::Prop(7521, "AlarmState", 1)]), Some(56));
    let off = script("pr_off", false, 0.8, &[], &[ALARM, Cmd::Prop(40000, "AlarmState", 0)]);
    assert!(off.iter().any(|e| e.kind == OracleKind::NearMiss && e.end > 40000));
    // with two matching overrides the first listed wins (AlarmState 0.95 before AudioPlayerState 0.7)
    let prec = [
        (Cfg::Kw, Edit::Once(r#""accept-threshold": 0.7, "clientProperties""#, r#""accept-threshold": 0.95, "clientProperties""#)),
        (
            Cfg::Kw,
            Edit::Once(
                r#""accept-threshold": 0.86, "clientProperties": [ { "equals": 1, "name": "AudioPlayerState" }"#,
                r#""accept-threshold": 0.7, "clientProperties": [ { "equals": 1, "name": "AudioPlayerState" }"#,
            ),
        ),
        (Cfg::Kw, Edit::Once(r#""accept-threshold": 0.64"#, r#""accept-threshold": 0.95"#)),
    ];
    let r = script("pr_prec", false, 0.8, &prec, &[ALARM, Cmd::Prop(0, "AudioPlayerState", 1)]);
    assert!(r.iter().all(|e| e.kind == OracleKind::NearMiss), "{r:?}");
}

#[test]
fn unknown_property_ignored() {
    let r = script("pr_unknown", false, 0.8, &[], &[Cmd::Prop(0, "NoSuchProperty", 1)]);
    let (_d, m) = probe_set("plain", false, 0.8, &[]);
    let plain = spot(&m, &synth(Clip::Noise), 1600);
    assert_eq!(r.iter().map(ev_line).collect::<Vec<_>>(), plain.iter().map(ev_line).collect::<Vec<_>>());
}

#[test]
fn property_value_two_matches_none() {
    let r = script("pr_v2", false, 0.8, &[], &[Cmd::Prop(0, "AlarmState", 2)]);
    assert!(!r.is_empty() && r.iter().all(|e| e.kind == OracleKind::NearMiss), "{r:?}");
}

#[test]
fn session_end_matches_oracle() {
    // front end, network, spotters (pending notify reported, probabilistic dropped) and machine restart at the
    // session end; client properties, the rising-score memory and the sample clock carry on
    let r = script("se_sleep_mid", false, 0.99, &[], &[Cmd::End(30080)]);
    assert!(r.iter().any(|e| e.end == 30080 + 1280 && e.kind == OracleKind::Accept));
    script("se_twice", false, 0.99, &[], &[Cmd::End(30080), Cmd::End(30081)]);
    let awake = [
        (Cfg::Kw, Edit::Once(STOP_DECAY, r#""classifier": { "type": "exponential-decay", "decay-rate": 0.95 }"#)),
        (Cfg::Kw, Edit::Once(STOP_ACCEPT, r#""classification-thresholds": { "accept-threshold": 0.5 }"#)),
        (Cfg::Op, Edit::Once(r#""duration": 175"#, r#""duration": 100000"#)),
    ];
    let quiet = Cmd::Prop(2000, "AlarmState", 0);
    let ctl = script("se_awake_ctl", false, 0.8, &awake, &[ALARM, quiet]);
    assert!(ctl.iter().any(|e| e.keyword == "STOP"));
    let r = script("se_awake", false, 0.8, &awake, &[ALARM, quiet, Cmd::End(16080)]);
    assert!(!r.iter().any(|e| e.keyword == "STOP"), "the session end put the machine to sleep");
    let r = script("se_nm", true, 0.8, &[], &[Cmd::End(8080)]);
    assert_eq!(r.iter().map(ev_line).collect::<Vec<_>>(), [("ALEXA", OracleKind::NearMiss, 49)]);
    script("se_prob", false, 0.8, &[], &[Cmd::End(16080)]);
    script("se_prop", false, 0.8, &[], &[ALARM, Cmd::End(16080)]);
}

/// ALEXA accepts once (lock 100000) at frame 8; STOP (decay 0.97) would reach its threshold at frame 176.
fn timeout_edits<'a>(stop_accept: &'a str, duration: &'a str) -> [(Cfg, Edit<'a>); 4] {
    [
        (Cfg::Kw, Edit::Once(r#""lock-period": 132"#, r#""lock-period": 100000"#)),
        (Cfg::Kw, Edit::Once(STOP_DECAY, r#""classifier": { "type": "exponential-decay", "decay-rate": 0.97 }"#)),
        (Cfg::Kw, Edit::Once(STOP_ACCEPT, stop_accept)),
        (Cfg::Op, Edit::Once(r#""duration": 175"#, duration)),
    ]
}

#[test]
fn machine_timeout_matches_oracle() {
    let thr = r#""classification-thresholds": { "accept-threshold": 0.46 }"#;
    let r = script("to_168", false, 0.99, &timeout_edits(thr, r#""duration": 168"#), &[]);
    assert!(!r.iter().any(|e| e.keyword == "STOP"), "asleep from frame 8 + 168");
    let r = script("to_169", false, 0.99, &timeout_edits(thr, r#""duration": 169"#), &[]);
    assert!(r.iter().any(|e| e.keyword == "STOP" && e.end == 176 * 160));
    // ALEXA's awake -> awake accept at frame 128 restarts the timeout, not STOP
    let edits = [
        (Cfg::Kw, Edit::Once(r#""lock-period": 132"#, r#""lock-period": 120"#)),
        (Cfg::Kw, Edit::Once(STOP_DECAY, r#""classifier": { "type": "exponential-decay", "decay-rate": 0.97 }"#)),
        (Cfg::Kw, Edit::Once(STOP_ACCEPT, r#""classification-thresholds": { "accept-threshold": 0.49 }"#)),
    ];
    let r = script("to_selfreset", false, 0.99, &edits, &[]);
    assert!(r.iter().any(|e| e.keyword == "STOP" && e.end == 192 * 160));
}

#[test]
fn stop_while_awake_matches_oracle() {
    // firmware order (ALEXA before STOP): STOP, enabled by ALEXA's accept, runs in that frame one frame ahead
    let r = script("fw_stop", true, 0.91, &[(Cfg::Kw, Edit::Once(r#""accept-threshold": 0.5 }"#, r#""accept-threshold": 0.04 }"#))], &[]);
    assert_eq!((r[1].keyword.as_str(), r[1].begin, r[1].end), ("STOP", 0, 960));
}

const DISNEY_ACCEPT: &str = r#""accept-threshold": 0.45, "notify-threshold": 0.2"#;

/// At force 0.4 (not 0.01, where the history maximum and begin follow ulp-level network noise).
#[test]
fn hey_disney_wakes() {
    let edits = [
        (Cfg::Kw, Edit::Once(DISNEY_ACCEPT, r#""accept-threshold": 0.15, "notify-threshold": 0.2"#)),
        (Cfg::Kw, Edit::Once(STOP_ACCEPT, r#""classification-thresholds": { "accept-threshold": 0.3 }"#)),
    ];
    let r = script("hey_wakes", false, 0.4, &edits, &[]);
    let lines: Vec<_> = r.iter().map(ev_line).collect();
    let hey = lines.iter().position(|l| l.0 == "HEY_DISNEY" && l.1 == OracleKind::Accept).unwrap();
    assert!(lines[hey..].iter().any(|l| l.0 == "STOP"), "{lines:?}");
    assert!(!lines[..hey].iter().any(|l| l.0 == "STOP"), "{lines:?}");
}

#[test]
fn restart_trigger_matches_oracle() {
    // three ALEXA accepts in 180 s restart everything at end + 240 samples; STOP's count is its own
    let edits = [(
        Cfg::Kw,
        Edit::Once(
            r#""restart-trigger": { "max-event-count": 30, "evaluation-span-msec": 180000 }"#,
            r#""restart-trigger": { "max-event-count": 3, "evaluation-span-msec": 180000 }"#,
        ),
    )];
    let r = script("rs_davs_99_3", false, 0.99, &edits, &[]);
    let alexa: Vec<u64> = r.iter().filter(|e| e.keyword == "ALEXA").map(|e| e.end).collect();
    assert_eq!(alexa[..4], [1280, 23040, 44800, 44800 + 240 + 1280]);
    // near misses count too, checked on an accept: two near misses and the first accept make three
    let r = script("rs_nm_3", false, 0.8, &edits, &[Cmd::Prop(45000, "AlarmState", 1)]);
    let alexa: Vec<u64> = r.iter().filter(|e| e.keyword == "ALEXA").map(|e| e.end).collect();
    assert_eq!(alexa[..4], [22400, 44160, 46080, 46080 + 240 + 1280]);
}

#[test]
fn prob_accept_not_emitted() {
    // probability 1: libpryon accepts every probabilistic window; the engine reports the near miss it would be
    // with probability 0 (what deterministic_set gives the oracle)
    let one = [(Cfg::Kw, Edit::Once(r#""probabilistic-accept-probability": 0.001"#, r#""probabilistic-accept-probability": 1"#))];
    let zero = [(Cfg::Kw, Edit::Once(r#""probabilistic-accept-probability": 0.001"#, r#""probabilistic-accept-probability": 0"#))];
    assert!(fixture("synth-scripts", "prob1").contains("detection_type=AcceptProbabilistic"));
    let r0 = script("prob0", false, 0.8, &zero, &[]);
    let (_d, m) = probe_set("prob1", false, 0.8, &one);
    let r1 = spot(&m, &synth(Clip::Noise), 1600);
    assert_eq!(r1.iter().map(ev_line).collect::<Vec<_>>(), r0.iter().map(ev_line).collect::<Vec<_>>());
    assert!(r1.iter().filter(|e| e.keyword == "ALEXA").all(|e| e.kind == OracleKind::NearMiss));
}

const ECHO: SpotterOptions = SpotterOptions { echo_stream: true };

/// Frame kinds of the `mixed` AFE clip: clean lead-in, corrupt frames, every parsed version, clean gaps, a version
/// flipping every frame (local/afe16/dump).
fn mixed(k: usize) -> LsbFrame {
    match k {
        0..300 => LsbFrame::Clean,
        300..700 if k.is_multiple_of(3) => LsbFrame::Corrupt,
        300..700 => LsbFrame::V3_3,
        700..900 => LsbFrame::V2_4,
        900..1000 => LsbFrame::V2_0,
        1000..1100 => LsbFrame::V3_0,
        1100..1300 => LsbFrame::V0_0,
        1300..1500 => LsbFrame::Clean,
        1500..1600 if k.is_multiple_of(2) => LsbFrame::V3_3,
        1500..1600 => LsbFrame::V2_4,
        1600..2000 => LsbFrame::V1_0,
        _ => LsbFrame::V3_3,
    }
}

type AfeLine = (String, OracleKind, u64, u64, Option<(i64, i64)>);

fn afe_lines(mut v: Vec<AfeLine>) -> Vec<AfeLine> {
    v.sort_by_key(|l| (l.3, l.2, l.0.clone()));
    v
}

/// `tests/fixtures/wake/synth-afe/<name>.txt` (libpryon on a testkit LSB clip, -x 1, two identical runs): every
/// event as in `check_probe`, and each one's AFE times equal libpryon's META times. Returns the engine's lines.
fn check_afe(name: &str, fw: bool, force: f32, edits: &[(Cfg, Edit)], cmds: &[Cmd], pcm: &[i16]) -> Vec<AfeLine> {
    let (d, m) = probe_set(name, fw, force, edits);
    let text = fixture("synth-afe", name);
    let oracle = parse_wake(&text).unwrap();
    assert_eq!(pushed(&text), Some(pcm.len()), "synth-afe/{name}: the oracle was not fed the whole clip");
    let (det, restarts) = detect_cmds(&m, pcm, cmds, ECHO);
    let mut diff = Vec::new();
    let t = compare_events(&oracle, &from_rust(m.keywords(), &det), &thresholds(&d), true, &mut diff);
    let diffs: Vec<String> = diff.iter().map(Diff::to_string).collect();
    assert!(t.clean(), "synth-afe/{name}:\n{}", diffs.join("\n"));
    let kws = m.keywords();
    let got = afe_lines(
        det.iter()
            .map(|d| {
                let afe = d.afe.map(|a| (a.start, a.end));
                (kws[d.keyword].clone(), kind_of(d.kind), d.begin_sample, d.end_sample, afe)
            })
            .collect(),
    );
    let want = afe_lines(oracle.iter().map(|e| (e.keyword.clone(), e.kind, e.begin, e.end, e.afe)).collect());
    // a firmware set's network runs from uninitialised state for a while after each start: libpryon's keyword end
    // there is garbage, so only whether times exist is compared (compare_gated's window)
    let (spec, mut excused) = (load_set(&d).unwrap(), 0);
    let ends = cmds.iter().filter_map(|c| if let Cmd::End(at) = *c { Some(at) } else { None });
    let starts: Vec<u64> = std::iter::once(0).chain(restarts).chain(ends).collect();
    let gate = (FW_GATE_MAX_FRAMES + spec.longest_lock()) * 160;
    let window = |s: u64, end: u64| end > s && end < s + gate;
    let gated = |l: &AfeLine| spec.uninitialised_state() && starts.iter().any(|&s| window(s, l.3));
    let blur = |v: Vec<AfeLine>, n: &mut usize| -> Vec<AfeLine> {
        v.into_iter()
            .map(|mut l| {
                if gated(&l) && l.4.is_some() {
                    l.4 = Some((-1, -1));
                    *n += 1;
                }
                l
            })
            .collect()
    };
    let (got_cmp, want) = (blur(got.clone(), &mut excused), blur(want, &mut 0));
    assert_eq!(got_cmp, want, "synth-afe/{name}");
    println!("synth-afe/{name}: {excused} events in the firmware start window compared without times");
    println!("synth-afe/{name}: {} events, {} with AFE times", got.len(), got.iter().filter(|l| l.4.is_some()).count());
    got
}

fn lsb_noise(clock: u32) -> Vec<i16> {
    lsb_stream(synth(Clip::Noise), clock)
}

#[test]
fn afe_synth_sets_match_oracle() {
    let fw = check_afe("fw_99", true, 0.99, &[], &[], &lsb_noise(1000));
    assert!(fw.iter().all(|l| l.4.is_some()));
    let wrap = check_afe("davs_99_wrap", false, 0.99, &[], &[], &lsb_noise(60000));
    let after = wrap.iter().filter(|l| l.4.is_some_and(|(s, _)| s < 60000)).count();
    assert!(after > 0 && after < wrap.len(), "the clock passes 65535 mid-clip");
    // near misses and probabilistic windows carry times too
    let nm = check_afe("davs_f08", false, 0.8, &[], &[], &lsb_noise(1000));
    assert!(nm.iter().any(|l| l.1 == OracleKind::NearMiss && l.4.is_some()));
    // STOP starts at sample 0: libpryon's lookup returns a zeroed record there
    let stop = [(Cfg::Kw, Edit::Once(r#""accept-threshold": 0.5 }"#, r#""accept-threshold": 0.04 }"#))];
    let r = check_afe("fw_stop", true, 0.91, &stop, &[], &lsb_noise(1000));
    assert!(r.iter().any(|l| l.0 == "STOP" && l.2 == 0 && l.4.is_some_and(|(s, _)| s == 0)));
}

#[test]
fn afe_aggregator_matches_oracle() {
    let edits = [
        (Cfg::Kw, Edit::Once(DISNEY_ACCEPT, r#""accept-threshold": 0.15, "notify-threshold": 0.2"#)),
        (Cfg::Kw, Edit::Once(STOP_ACCEPT, r#""classification-thresholds": { "accept-threshold": 0.3 }"#)),
    ];
    let r = check_afe("hey_wakes", false, 0.4, &edits, &[], &lsb_noise(1000));
    assert!(r.iter().any(|l| l.0 == "HEY_DISNEY" && l.4.is_some()));
}

#[test]
fn afe_session_end_and_restart_match_oracle() {
    // the session end flushes libpryon's decoder before the pending near miss is reported
    let r = check_afe("fw_f08_se", true, 0.8, &[], &[Cmd::End(8080), Cmd::End(50001)], &lsb_noise(1000));
    assert_eq!(r.iter().map(|l| (l.1, l.3, l.4)).collect::<Vec<_>>(), [(OracleKind::NearMiss, 7840, None)]);
    let r = check_afe("davs_99_se", false, 0.99, &[], &[Cmd::End(30001)], &lsb_noise(1000));
    assert!(r.iter().any(|l| l.3 > 30001 && l.4.is_some()));
    let edits = [(
        Cfg::Kw,
        Edit::Once(
            r#""restart-trigger": { "max-event-count": 30, "evaluation-span-msec": 180000 }"#,
            r#""restart-trigger": { "max-event-count": 3, "evaluation-span-msec": 180000 }"#,
        ),
    )];
    check_afe("rs_davs_99_3", false, 0.99, &edits, &[], &lsb_noise(1000));
}

#[test]
fn afe_frame_kinds_match_oracle() {
    let pcm = lsb_stream_with(synth(Clip::Noise), 0, mixed);
    for (name, fw) in [("fw_99_mixed", true), ("davs_99_mixed", false)] {
        let r = check_afe(name, fw, 0.99, &[], &[], &pcm);
        assert!(r.iter().any(|l| l.4.is_none()) && r.iter().any(|l| l.4.is_some()), "{name}");
    }
}

#[test]
fn afe_clean_audio_matches_oracle() {
    // the low bits of this clean clip pass for a frame without a clock: libpryon reports times of 0
    let pcm = synth(Clip::QuietTones);
    for (name, fw) in [("fw_99_quiet_tones", true), ("davs_99_quiet_tones", false)] {
        let r = check_afe(name, fw, 0.99, &[], &[], &pcm);
        assert!(r.iter().any(|l| l.4 == Some((0, 0))) && r.iter().any(|l| l.4.is_none()), "{name}");
    }
}

#[test]
fn afe_off_by_default() {
    let pcm = lsb_noise(1000);
    for fw in [true, false] {
        let (_d, m) = synth_model(fw, Some(0.99), "afe-off");
        let plain = run(&mut Spotter::new(m.clone(), SpotterOptions::default()).unwrap(), &pcm, 1600);
        assert!(!plain.is_empty() && plain.iter().all(|d| d.afe.is_none()), "fw {fw}");
        let echo = run(&mut Spotter::new(m.clone(), ECHO).unwrap(), &pcm, 1600);
        assert!(echo.iter().all(|d| d.afe.is_some()), "fw {fw}");
        let strip = |d: &[Detection]| d.iter().map(|x| Detection { afe: None, ..x.clone() }).collect::<Vec<_>>();
        assert_eq!(strip(&echo), plain, "fw {fw}: the option changes nothing else");
    }
}

#[test]
fn afe_none_on_clean_audio() {
    let (_d, m) = synth_model(true, Some(0.99), "afe-clean");
    let out = run(&mut Spotter::new(m, ECHO).unwrap(), &synth(Clip::Noise), 1600);
    assert!(!out.is_empty() && out.iter().all(|d| d.afe.is_none()));
}

#[test]
fn afe_chunking_invariant() {
    let pcm = &lsb_stream_with(synth(Clip::Noise), 65000, mixed)[..16000 * 12 + 77];
    for fw in [true, false] {
        let (_d, m) = synth_model(fw, Some(0.99), "afe-chunks");
        let whole = run(&mut Spotter::new(m.clone(), ECHO).unwrap(), pcm, pcm.len());
        assert!(whole.iter().any(|d| d.afe.is_some()), "fw {fw}");
        for chunk in [1, 7, 128, 160, 1280, 3333] {
            let got = run(&mut Spotter::new(m.clone(), ECHO).unwrap(), pcm, chunk);
            assert_eq!(bits(&got), bits(&whole), "fw {fw} chunk {chunk}");
        }
    }
}

#[test]
fn afe_after_reset() {
    let pcm = lsb_noise(0);
    let (_d, m) = synth_model(false, Some(0.99), "afe-reset");
    let mut sp = Spotter::new(m, ECHO).unwrap();
    let mut out = run(&mut sp, &pcm[..48000], 1600);
    sp.reset(&mut out);
    let before = out.len();
    sp.feed(&pcm[48000..96000], &mut out);
    // the first frame after the session end decodes at 48128: later starts read their own frames again
    assert!(out.len() > before);
    for d in &out[before..] {
        let start = d.afe.map(|a| a.start);
        assert_eq!(start, Some((d.begin_sample / 128 * 8) as i64), "{d:?}");
        assert!(d.begin_sample >= 48000 + 160);
    }
}

/// `PRYON_AFE_CLIP`: an Echo `micAsr` capture (16 kHz s16le `.raw`, or a directory of them) through libpryon in real
/// time and the engine; the accepts pair up one to one by keyword, begin and end, with equal META times.
#[test]
fn afe_matches_oracle_meta() {
    let Some(clip) = env_dir("PRYON_AFE_CLIP") else { return };
    let Some(env) = OracleEnv::from_env() else {
        println!("skipped: set PRYON_ORACLE_ROOTFS and PRYON_PRYON_TEST");
        return;
    };
    let clips: Vec<PathBuf> = if clip.is_dir() {
        let mut v: Vec<PathBuf> = std::fs::read_dir(&clip).unwrap().map(|e| e.unwrap().path()).collect();
        v.retain(|p| p.extension().is_some_and(|e| e == "raw"));
        v.sort();
        v
    } else {
        vec![clip]
    };
    let sets: Vec<(&str, PathBuf)> = ["fw", "alexa"].iter().filter_map(|s| real_set(s).map(|d| (*s, d))).collect();
    if sets.is_empty() {
        return;
    }
    let work = TempDir::new("afe-oracle");
    let (mut accepts, mut timed) = (0, 0);
    for (name, dir) in &sets {
        let m = Arc::new(SpotterModel::load(dir).unwrap());
        // libpryon's firmware-set events in its start window are garbage-dependent (compare_gated): not compared
        let spec = load_set(dir).unwrap();
        let gate = spec.uninitialised_state().then(|| (FW_GATE_MAX_FRAMES + spec.longest_lock()) * 160);
        let oracle_set = deterministic_set(dir, &work.join(name)).unwrap();
        let manifest = oracle_set.join("pryon.manifest");
        let results = par_map(&clips, |path| {
            let pcm = read_clip(path);
            // real time: at -x 1000 a long clip overruns libpryon's backlog and it drops audio (run_wake refuses that)
            let timeout = std::time::Duration::from_secs(10 * pcm.len() as u64 / 16000 + 60);
            let text = env.run_wake_at(&manifest, path, &[], 1.0, Some(timeout)).unwrap();
            assert_eq!(pushed(&text), Some(pcm.len()), "{}: oracle not fed the whole clip", path.display());
            let oracle = parse_wake(&text).unwrap();
            let rust = run(&mut Spotter::new(m.clone(), ECHO).unwrap(), &pcm, 1600);
            let kws = m.keywords();
            let mut bad = Vec::new();
            let (mut n, mut t) = (0, 0);
            let outside = |o: &&OracleEvent| gate.is_none_or(|g| o.end >= g);
            let accept = |d: &&Detection| d.kind != DetectionKind::NearMiss && gate.is_none_or(|g| d.end_sample >= g);
            for d in rust.iter().filter(accept) {
                let partner = oracle.iter().any(|o| {
                    let same = (o.begin, o.end) == (d.begin_sample, d.end_sample);
                    o.keyword == kws[d.keyword] && o.kind == kind_of(d.kind) && same
                });
                if !partner {
                    let (b, e) = (d.begin_sample, d.end_sample);
                    bad.push(format!("{} {:?} {b}..{e}: engine only", kws[d.keyword], d.kind));
                }
            }
            for o in oracle.iter().filter(|o| o.kind != OracleKind::NearMiss).filter(outside) {
                n += 1;
                t += usize::from(o.afe.is_some());
                let r = rust.iter().find(|d| {
                    let same = (d.begin_sample, d.end_sample) == (o.begin, o.end);
                    kws[d.keyword] == o.keyword && kind_of(d.kind) == o.kind && same
                });
                let got = r.map(|d| d.afe.map(|a| (a.start, a.end)));
                if got != Some(o.afe) {
                    let (k, b, e) = (&o.keyword, o.begin, o.end);
                    bad.push(format!("{k} {:?} {b}..{e}: oracle {:?}, engine {got:?}", o.kind, o.afe));
                }
            }
            (path.display().to_string(), n, t, bad)
        });
        for (path, n, t, bad) in results {
            println!("{name} {path}: {n} accepts, {t} with AFE times");
            assert!(bad.is_empty(), "{name} {path}:\n{}", bad.join("\n"));
            (accepts, timed) = (accepts + n, timed + t);
        }
    }
    assert!(accepts > 0, "no accept in {} clip(s)", clips.len());
    assert!(timed > 0, "no accept with AFE times: the clip carries no front-end metadata");
}
