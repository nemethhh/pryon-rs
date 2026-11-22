//! Wake Spotter API and parity with libpryon's wake events (committed fixtures on synthetic clips, env-gated corpus).

use pryon::wake::{load_set, Detection, DetectionKind, Spotter, SpotterModel, SpotterOptions};
use pryon_testkit::compare::{compare as compare_events, pushed, Diff, Ev, Limits, Tally};
use pryon_testkit::oracle::{parse_wake, OracleEvent, OracleKind};
use pryon_testkit::wakeset::{write_davs_set, write_fw_set, WakeSetOpts};
use pryon_testkit::{synth, Clip, TempDir};
use std::collections::BTreeMap;
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
            sp.reset();
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

/// `EVENT keyword=K kind=… score=… begin=B end=E` lines (pryon-cli wake's format; local pypryon dumps).
fn from_event_lines(text: &str) -> Vec<Ev> {
    let field = |l: &str, k: &str| -> String {
        let at = l.find(k).unwrap_or_else(|| panic!("no {k} in {l}")) + k.len();
        l[at..].split(' ').next().unwrap().to_string()
    };
    text.lines()
        .filter(|l| l.starts_with("EVENT "))
        .map(|l| Ev {
            keyword: field(l, "keyword="),
            kind: match field(l, "kind=").as_str() {
                "Accept" => OracleKind::Accept,
                "AcceptAuxiliary" => OracleKind::AcceptAuxiliary,
                "NearMiss" => OracleKind::NearMiss,
                k => panic!("kind {k}"),
            },
            score: field(l, "score=").parse().unwrap(),
            begin: field(l, "begin=").parse().unwrap(),
            end: field(l, "end=").parse().unwrap(),
        })
        .collect()
}

/// Thresholds a score is compared against, per keyword.
fn thresholds(dir: &Path) -> Limits {
    let mut l = Limits::default();
    for s in load_set(dir).unwrap().spotters {
        let t = &s.thresholds;
        let ov: Vec<f32> = t.overrides.iter().map(|o| o.accept).collect();
        l.add(&s.name, t.accept, t.notify, &ov, t.probabilistic.map(|p| p.0));
    }
    l
}

fn compare(oracle: &[OracleEvent], engine: &[Ev], thr: &Limits, diffs: &mut Vec<String>) -> Tally {
    let (mut d, mut prob): (Vec<Diff>, Vec<Ev>) = (Vec::new(), Vec::new());
    let t = compare_events(oracle, engine, thr, false, &mut d, &mut prob);
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

/// Rust rules (and, with `PRYON_PYPRYON_EVENTS`, pypryon's dumped events) against the cached oracle on every corpus
/// clip. The cache is `<corpus>/../oracle/<checksum>/<group>/<clip>.txt`; a clip with a second run in
/// `<corpus>/../oracle2/` that disagrees with the first is unstable (ruling F1) and only counted.
#[test]
fn oracle_corpus_pypryon_level() {
    let Some(root) = env_dir("PRYON_WAKE_CORPUS") else { return };
    let sets: Vec<(&str, PathBuf)> = ["fw", "alexa"].iter().filter_map(|s| real_set(s).map(|d| (*s, d))).collect();
    let pyref = std::env::var_os("PRYON_PYPRYON_EVENTS").map(PathBuf::from);
    let clips = corpus(&root);
    assert!(!clips.is_empty(), "{}: no <group>/<clip>.raw", root.display());
    let base = root.parent().unwrap().to_path_buf();
    let mut failures = Vec::new();
    for (name, dir) in sets {
        let m = Arc::new(SpotterModel::load(&dir).unwrap());
        let thr = thresholds(&dir);
        let sum = m.checksum().to_string();
        let second = base.join("oracle2").join(&sum);
        let results = par_map(&clips, |(group, clip, path)| {
            let cached = base.join("oracle").join(&sum).join(group).join(format!("{clip}.txt"));
            let Ok(text) = std::fs::read_to_string(&cached) else { return None };
            let oracle = parse_wake(&text).unwrap_or_else(|e| panic!("{}: {e}", cached.display()));
            let stable = match std::fs::read_to_string(second.join(group).join(format!("{clip}.txt"))) {
                Ok(t2) => Some(parse_wake(&t2).unwrap() == oracle),
                Err(_) => None,
            };
            let mut diffs = Vec::new();
            let pcm = read_clip(path);
            let pcm = &pcm[..pushed(&text).unwrap_or(pcm.len()).min(pcm.len())];
            let rust = compare(&oracle, &spot(&m, pcm, 1600), &thr, &mut diffs);
            let py = pyref.as_ref().and_then(|p| {
                let t = std::fs::read_to_string(p.join(&sum).join(group).join(format!("{clip}.txt"))).ok()?;
                let py: Vec<Ev> = from_event_lines(&t).into_iter().filter(|e| e.end <= pcm.len() as u64).collect();
                Some(compare(&oracle, &py, &thr, &mut Vec::new()))
            });
            Some((group.clone(), clip.clone(), stable, rust, py, diffs))
        });
        let missing = results.iter().filter(|r| r.is_none()).count();
        let has_second = second.is_dir();
        println!("== {name} ({sum}): {} clips cached, {missing} without oracle output", clips.len() - missing);
        if !has_second {
            println!("   no second oracle run in {}: every clip counted as stable", second.display());
        }
        let mut groups: BTreeMap<String, [Tally; 3]> = BTreeMap::new();
        let mut unstable_clips = Vec::new();
        for (group, clip, stable, rust, py, diffs) in results.into_iter().flatten() {
            let stable = stable.unwrap_or(!has_second);
            let g = groups.entry(group.clone()).or_default();
            if !stable {
                g[2].add(&rust);
                unstable_clips.push(format!("{group}/{clip}"));
                continue;
            }
            g[0].add(&rust);
            if let Some(p) = py {
                g[1].add(&p);
            }
            if rust.accepts_paired < rust.accepts {
                failures.push(format!("{name} {group}/{clip}: oracle accept unpaired\n  {}", diffs.join("\n  ")));
            }
        }
        let mut total = [Tally::default(), Tally::default(), Tally::default()];
        for (group, g) in &groups {
            println!("{group:>14} rust     {}", g[0].line());
            if pyref.is_some() {
                println!("{group:>14} pypryon  {}", g[1].line());
            }
            if g[2].clips > 0 {
                println!("{group:>14} unstable {}", g[2].line());
            }
            for (t, x) in total.iter_mut().zip(g) {
                t.add(x);
            }
        }
        println!("{:>14} rust     {}", "all", total[0].line());
        if pyref.is_some() {
            println!("{:>14} pypryon  {}", "all", total[1].line());
            let (r, p) = (&total[0], &total[1]);
            for (what, a, b) in [
                ("matched", r.matched, p.matched),
                ("end", r.end_eq, p.end_eq),
                ("accepts paired", r.accepts_paired, p.accepts_paired),
            ] {
                if a < b {
                    failures.push(format!("{name}: rust {what} {a} < pypryon {b}"));
                }
            }
        }
        println!("{:>14} {} unstable clips: {}", "", unstable_clips.len(), unstable_clips.join(" "));
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
