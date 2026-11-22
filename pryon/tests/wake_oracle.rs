//! Wake Spotter API and parity with libpryon's wake events (committed fixtures on synthetic clips, env-gated corpus).

use pryon::wake::{load_set, Detection, DetectionKind, Spotter, SpotterModel, SpotterOptions};
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

/// One event from either engine, in the oracle's terms.
#[derive(Clone, Debug)]
struct Ev {
    keyword: String,
    kind: OracleKind,
    score: f32,
    begin: u64,
    end: u64,
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

fn from_oracle(o: &[OracleEvent]) -> Vec<Ev> {
    o.iter()
        .map(|e| Ev {
            keyword: e.keyword.clone(),
            kind: e.kind,
            score: e.score.parse().unwrap(),
            begin: e.begin,
            end: e.end,
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

/// Oracle and engine events are paired per keyword by nearest end sample within this distance.
const PAIR_WINDOW: u64 = 16000;
/// Not emitted until Task 12.
const UNSERVED: &[&str] = &["HEY_DISNEY"];
/// The oracle reports STOP only while awake (op.cfg), the rules emit it always.
const AWAKE_ONLY: &str = "STOP";

#[derive(Default, Clone)]
struct Tally {
    clips: usize,
    oracle: usize,
    engine: usize,
    dropped: usize,
    matched: usize,
    differing: usize,
    oracle_only: usize,
    engine_only: usize,
    /// Near misses among the one-sided events.
    oracle_only_nm: usize,
    engine_only_nm: usize,
    awake_only: usize,
    kind_eq: usize,
    begin_eq: usize,
    end_eq: usize,
    score_ok: usize,
    accepts: usize,
    accepts_paired: usize,
    score_d: Vec<f32>,
}

impl Tally {
    fn add(&mut self, o: &Tally) {
        let n = |a: &mut usize, b: usize| *a += b;
        n(&mut self.clips, o.clips);
        n(&mut self.oracle, o.oracle);
        n(&mut self.engine, o.engine);
        n(&mut self.dropped, o.dropped);
        n(&mut self.matched, o.matched);
        n(&mut self.differing, o.differing);
        n(&mut self.oracle_only, o.oracle_only);
        n(&mut self.engine_only, o.engine_only);
        n(&mut self.oracle_only_nm, o.oracle_only_nm);
        n(&mut self.engine_only_nm, o.engine_only_nm);
        n(&mut self.awake_only, o.awake_only);
        n(&mut self.kind_eq, o.kind_eq);
        n(&mut self.begin_eq, o.begin_eq);
        n(&mut self.end_eq, o.end_eq);
        n(&mut self.score_ok, o.score_ok);
        n(&mut self.accepts, o.accepts);
        n(&mut self.accepts_paired, o.accepts_paired);
        self.score_d.extend_from_slice(&o.score_d);
    }

    fn clean(&self) -> bool {
        self.differing == 0 && self.oracle_only == 0 && self.engine_only == 0
    }

    fn line(&self) -> String {
        let mut d = self.score_d.clone();
        d.sort_by(f32::total_cmp);
        let q = |p: f32| if d.is_empty() { 0.0 } else { d[((d.len() - 1) as f32 * p) as usize] };
        format!(
            "clips {} | oracle {} (+{} dropped) engine {} | matched {} differing {} oracle-only {} ({} NM) \
             engine-only {} ({} NM, +{} {AWAKE_ONLY} while oracle asleep) | paired: kind {} begin {} end {} \
             score {} | oracle accepts {} paired {} | score |d| median {:.2e} p90 {:.2e} max {:.2e}",
            self.clips,
            self.oracle,
            self.dropped,
            self.engine,
            self.matched,
            self.differing,
            self.oracle_only,
            self.oracle_only_nm,
            self.engine_only,
            self.engine_only_nm,
            self.awake_only,
            self.kind_eq,
            self.begin_eq,
            self.end_eq,
            self.score_ok,
            self.accepts,
            self.accepts_paired,
            q(0.5),
            q(0.9),
            q(1.0),
        )
    }
}

/// Thresholds a score is compared against, per keyword (primary and verifier accept and notify, overrides).
fn thresholds(dir: &Path) -> BTreeMap<String, Vec<f32>> {
    let mut m: BTreeMap<String, Vec<f32>> = BTreeMap::new();
    for s in load_set(dir).unwrap().spotters {
        let t = m.entry(s.name.clone()).or_default();
        t.push(s.thresholds.accept);
        t.extend(s.thresholds.notify);
        t.extend(s.thresholds.overrides.iter().map(|o| o.accept));
    }
    m
}

/// Exact 6-decimal equality when every threshold is more than 1e-2 away, else within 1e-3.
fn score_ok(oracle: &Ev, printed_threshold: Option<f32>, engine: f32, thr: &[f32]) -> bool {
    let margin = thr.iter().chain(&printed_threshold).map(|t| (oracle.score - t).abs()).fold(f32::INFINITY, f32::min);
    if margin > 1e-2 {
        format!("{:.6}", oracle.score) == format!("{:.6}", engine)
    } else {
        (oracle.score - engine).abs() <= 1e-3
    }
}

fn compare(oracle: &[OracleEvent], engine: &[Ev], thr: &BTreeMap<String, Vec<f32>>, diffs: &mut Vec<String>) -> Tally {
    let mut t = Tally { clips: 1, ..Tally::default() };
    let printed: Vec<Option<f32>> = oracle.iter().map(|e| e.threshold.parse().ok()).collect();
    let all = from_oracle(oracle);
    let served = |e: &&Ev| !UNSERVED.contains(&e.keyword.as_str());
    let kept: Vec<(usize, &Ev)> = all.iter().enumerate().filter(|(_, e)| served(e)).collect();
    t.dropped = all.len() - kept.len();
    t.oracle = kept.len();
    t.engine = engine.len();
    let mut used = vec![false; engine.len()];
    for &(i, o) in &kept {
        if o.kind != OracleKind::NearMiss {
            t.accepts += 1;
        }
        let best = engine
            .iter()
            .enumerate()
            .filter(|(j, e)| !used[*j] && e.keyword == o.keyword && e.end.abs_diff(o.end) <= PAIR_WINDOW)
            .min_by_key(|(_, e)| e.end.abs_diff(o.end));
        let Some((j, e)) = best else {
            t.oracle_only += 1;
            t.oracle_only_nm += usize::from(o.kind == OracleKind::NearMiss);
            diffs.push(format!("oracle-only {o:?}"));
            continue;
        };
        used[j] = true;
        if o.kind != OracleKind::NearMiss {
            t.accepts_paired += 1;
        }
        let none = Vec::new();
        let ok_score = score_ok(o, printed[i], e.score, thr.get(&o.keyword).unwrap_or(&none));
        t.kind_eq += usize::from(o.kind == e.kind);
        t.begin_eq += usize::from(o.begin == e.begin);
        t.end_eq += usize::from(o.end == e.end);
        t.score_ok += usize::from(ok_score);
        t.score_d.push((o.score - e.score).abs());
        if o.kind == e.kind && o.begin == e.begin && o.end == e.end && ok_score {
            t.matched += 1;
        } else {
            t.differing += 1;
            diffs.push(format!("differing oracle {o:?} engine {e:?}"));
        }
    }
    for (j, e) in engine.iter().enumerate().filter(|(j, _)| !used[*j]) {
        if e.keyword == AWAKE_ONLY {
            t.awake_only += 1;
        } else {
            t.engine_only += 1;
            t.engine_only_nm += usize::from(e.kind == OracleKind::NearMiss);
            diffs.push(format!("engine-only {e:?} (#{j})"));
        }
    }
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

/// Samples pryon_test fed (`pushed N samples`); it stops early when libpryon restarts its decoder.
fn pushed(text: &str) -> Option<usize> {
    let l = text.lines().find_map(|l| l.strip_prefix("pushed "))?;
    l.split(' ').next()?.parse().ok()
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
