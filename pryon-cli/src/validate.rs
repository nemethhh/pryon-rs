//! `validate`: libpryon (qemu) and pryon-rs on the same clips, every difference printed, validated sets recorded.

use pryon::wake::{load_set, Detection, DetectionKind, Spotter, SpotterModel, SpotterOptions};
use pryon_testkit::compare::{compare, pushed, Diff, Ev, Limits, Tally};
use pryon_testkit::oracle::{parse_wake, OracleCmd, OracleEnv, OracleEvent, OracleKind};
use pryon_testkit::fnv1a64;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

struct Args {
    rootfs: Option<PathBuf>,
    pryon_test: Option<PathBuf>,
    cache: Option<PathBuf>,
    records: PathBuf,
    exact: bool,
    speed: f64,
    jobs: usize,
    /// (sample, name, value).
    props: Vec<(u64, String, i32)>,
    model: PathBuf,
    clips: Vec<PathBuf>,
}

fn parse_args(args: &[String]) -> Result<Args, String> {
    let mut a = Args {
        rootfs: None,
        pryon_test: None,
        cache: None,
        records: Path::new(env!("CARGO_MANIFEST_DIR")).join("../validated"),
        exact: false,
        speed: 1.0,
        jobs: std::thread::available_parallelism().map_or(4, |n| n.get()),
        props: Vec::new(),
        model: PathBuf::new(),
        clips: Vec::new(),
    };
    let mut pos: Vec<String> = Vec::new();
    let mut it = args.iter();
    while let Some(o) = it.next() {
        let mut val = || it.next().cloned().ok_or_else(|| format!("{o} needs a value"));
        match o.as_str() {
            "--oracle-rootfs" => a.rootfs = Some(val()?.into()),
            "--pryon-test" => a.pryon_test = Some(val()?.into()),
            "--cache" => a.cache = Some(val()?.into()),
            "--records" => a.records = val()?.into(),
            "--exact" => a.exact = true,
            "--speed" => match val()?.parse::<f64>() {
                Ok(x) if x > 0.0 && x.is_finite() => a.speed = x,
                _ => return Err("--speed needs a positive number".into()),
            },
            "--jobs" => match val()?.parse::<usize>() {
                Ok(n) if n > 0 => a.jobs = n,
                _ => return Err("--jobs needs a positive integer".into()),
            },
            "-p" => {
                let v = val()?;
                let (assign, at) = v.split_once('@').map_or((v.as_str(), Some(0)), |(x, s)| (x, s.parse::<u64>().ok()));
                match (assign.split_once('='), at) {
                    (Some((n, x)), Some(at)) if !n.is_empty() => match x.parse::<i32>() {
                        Ok(x) => a.props.push((at, n.to_string(), x)),
                        Err(_) => return Err(format!("-p {v}: VALUE must be an integer")),
                    },
                    _ => return Err(format!("-p {v}: expected NAME=VALUE[@SAMPLE]")),
                }
            }
            s if s.starts_with('-') && s.len() > 1 => return Err(format!("unknown option {s}")),
            _ => pos.push(o.clone()),
        }
    }
    if pos.len() < 2 {
        return Err("expected MODEL and at least one CLIP".into());
    }
    a.model = pos.remove(0).into();
    for p in pos {
        collect_clips(Path::new(&p), &mut a.clips)?;
    }
    a.clips.sort();
    a.clips.dedup();
    if a.clips.is_empty() {
        return Err("no .raw or .wav clips found".into());
    }
    Ok(a)
}

fn is_clip(p: &Path) -> bool {
    p.extension().and_then(|e| e.to_str()).is_some_and(|e| e.eq_ignore_ascii_case("raw") || e.eq_ignore_ascii_case("wav"))
}

fn collect_clips(p: &Path, out: &mut Vec<PathBuf>) -> Result<(), String> {
    let md = std::fs::metadata(p).map_err(|e| format!("{}: {e}", p.display()))?;
    if md.is_dir() {
        for e in std::fs::read_dir(p).map_err(|e| format!("{}: {e}", p.display()))? {
            let q = e.map_err(|e| e.to_string())?.path();
            if q.is_dir() || is_clip(&q) {
                collect_clips(&q, out)?;
            }
        }
    } else {
        out.push(p.to_path_buf());
    }
    Ok(())
}

fn read_pcm(p: &Path) -> Result<Vec<i16>, String> {
    let b = std::fs::read(p).map_err(|e| e.to_string())?;
    let data: &[u8] = if p.extension().and_then(|e| e.to_str()).is_some_and(|e| e.eq_ignore_ascii_case("wav")) {
        wav_data(&b)?
    } else {
        &b
    };
    if data.len() % 2 != 0 {
        return Err("odd byte count, not s16le".into());
    }
    Ok(data.chunks_exact(2).map(|x| i16::from_le_bytes([x[0], x[1]])).collect())
}

/// The `data` chunk of a 16 kHz mono 16-bit PCM WAV; anything else is an error.
fn wav_data(b: &[u8]) -> Result<&[u8], String> {
    if b.len() < 12 || &b[..4] != b"RIFF" || &b[8..12] != b"WAVE" {
        return Err("not a RIFF/WAVE file".into());
    }
    let u16_at = |i: usize| u16::from_le_bytes([b[i], b[i + 1]]);
    let u32_at = |i: usize| u32::from_le_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]]) as usize;
    let mut fmt_ok = false;
    let mut at = 12;
    while at + 8 <= b.len() {
        let (id, len) = (&b[at..at + 4], u32_at(at + 4));
        let body = at + 8;
        match id {
            b"fmt " if len >= 16 && body + 16 <= b.len() => {
                let (tag, ch, rate, bits) = (u16_at(body), u16_at(body + 2), u32_at(body + 4), u16_at(body + 14));
                if (tag, ch, rate, bits) != (1, 1, 16000, 16) {
                    return Err(format!("WAV is format {tag}, {ch} ch, {rate} Hz, {bits} bit; need PCM mono 16000 Hz 16 bit"));
                }
                fmt_ok = true;
            }
            b"data" => {
                if !fmt_ok {
                    return Err("WAV data before fmt".into());
                }
                return Ok(&b[body..(body + len).min(b.len())]);
            }
            _ => {}
        }
        at = body + len + (len & 1);
    }
    Err("WAV without a data chunk".into())
}

fn raw_bytes(pcm: &[i16]) -> Vec<u8> {
    pcm.iter().flat_map(|s| s.to_le_bytes()).collect()
}

fn kind_of(k: DetectionKind) -> OracleKind {
    match k {
        DetectionKind::Accept => OracleKind::Accept,
        DetectionKind::AcceptAuxiliary => OracleKind::AcceptAuxiliary,
        DetectionKind::NearMiss => OracleKind::NearMiss,
    }
}

/// Feeds `pcm` applying each property before its sample.
fn run_rust(model: &Arc<SpotterModel>, pcm: &[i16], props: &[(u64, String, i32)]) -> Result<Vec<Ev>, String> {
    let mut sp = Spotter::new(model.clone(), SpotterOptions::default()).map_err(|e| e.to_string())?;
    let mut cmds: Vec<&(u64, String, i32)> = props.iter().collect();
    cmds.sort_by_key(|c| c.0);
    let mut out: Vec<Detection> = Vec::new();
    let (mut pos, mut next) = (0usize, 0usize);
    loop {
        while next < cmds.len() && cmds[next].0 <= pos as u64 {
            sp.set_property(&cmds[next].1, cmds[next].2);
            next += 1;
        }
        if pos == pcm.len() {
            break;
        }
        let stop = cmds.get(next).map_or(pcm.len(), |c| (c.0 as usize).min(pcm.len()));
        let end = (pos + 1600).min(stop);
        sp.feed(&pcm[pos..end], &mut out);
        pos = end;
    }
    let kw = sp.keywords();
    Ok(out
        .iter()
        .map(|d| Ev { keyword: kw[d.keyword].clone(), kind: kind_of(d.kind), score: d.score, begin: d.begin_sample, end: d.end_sample })
        .collect())
}

struct ClipOut {
    name: String,
    /// Per keyword: tally, differences, probabilistic-accept events.
    per_kw: BTreeMap<String, (Tally, Vec<Diff>, Vec<Ev>)>,
    notes: Vec<String>,
    fnv: u64,
}

struct Ctx<'a> {
    args: &'a Args,
    model: Arc<SpotterModel>,
    manifest: PathBuf,
    limits: Limits,
    checksum: String,
    env: Option<OracleEnv>,
    scratch: PathBuf,
}

fn oracle_text(c: &Ctx, path: &Path, pcm: &[i16], fnv: u64) -> Result<String, String> {
    let cmds: Vec<OracleCmd> =
        c.args.props.iter().map(|(at, name, value)| OracleCmd::Property { at: *at, name: name.clone(), value: *value }).collect();
    // Properties and a non-default speed change the output, so they are part of the key.
    let mut key = fnv;
    if !cmds.is_empty() || c.args.speed != 1.0 {
        key = fnv1a64(format!("{fnv:016x}|{:?}|{}", cmds, c.args.speed).as_bytes());
    }
    let cached = c.args.cache.as_ref().map(|d| d.join(&c.checksum).join(format!("{key:016x}.txt")));
    if let Some(Ok(t)) = cached.as_ref().map(std::fs::read_to_string) {
        return Ok(t);
    }
    let env = c.env.as_ref().ok_or("no cached oracle output and no oracle (--oracle-rootfs and --pryon-test, or PRYON_ORACLE_ROOTFS and PRYON_PRYON_TEST)")?;
    let is_raw = path.extension().and_then(|e| e.to_str()).is_some_and(|e| e.eq_ignore_ascii_case("raw"));
    let tmp_raw;
    let clip = if is_raw {
        path.to_path_buf()
    } else {
        tmp_raw = c.scratch.join(format!("{fnv:016x}.raw"));
        std::fs::write(&tmp_raw, raw_bytes(pcm)).map_err(|e| e.to_string())?;
        tmp_raw.clone()
    };
    let secs = pcm.len() as f64 / 16000.0;
    let timeout = Duration::from_secs_f64(10.0 * secs / c.args.speed + 60.0);
    let text = env.run_wake_at(&c.manifest, &clip, &cmds, c.args.speed, Some(timeout)).map_err(|e| format!("oracle: {e}"))?;
    if let Some(f) = cached {
        let dir = f.parent().unwrap();
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        let tmp = f.with_extension(format!("tmp{}", std::process::id()));
        std::fs::write(&tmp, &text).and_then(|_| std::fs::rename(&tmp, &f)).map_err(|e| format!("{}: {e}", f.display()))?;
    }
    Ok(text)
}

fn one_clip(c: &Ctx, path: &Path) -> Result<ClipOut, String> {
    let name = path.display().to_string();
    let pcm = read_pcm(path)?;
    let fnv = fnv1a64(&raw_bytes(&pcm));
    let text = oracle_text(c, path, &pcm, fnv)?;
    let oracle = parse_wake(&text).map_err(|e| format!("oracle output: {e}"))?;
    let mut notes = Vec::new();
    let fed = pushed(&text).unwrap_or(pcm.len()).min(pcm.len());
    if fed < pcm.len() {
        notes.push(format!("NOTE restart {name} oracle fed {fed} of {} samples; Rust fed the same", pcm.len()));
    }
    let engine = run_rust(&c.model, &pcm[..fed], &c.args.props)?;
    let mut keywords: Vec<String> = c.model.keywords().to_vec();
    keywords.extend(oracle.iter().map(|e| e.keyword.clone()));
    keywords.sort();
    keywords.dedup();
    let mut per_kw = BTreeMap::new();
    for k in keywords {
        let o: Vec<OracleEvent> = oracle.iter().filter(|e| e.keyword == k).cloned().collect();
        let r: Vec<Ev> = engine.iter().filter(|e| e.keyword == k).cloned().collect();
        let (mut diffs, mut prob) = (Vec::new(), Vec::new());
        let t = compare(&o, &r, &c.limits, c.args.exact, &mut diffs, &mut prob);
        per_kw.insert(k, (t, diffs, prob));
    }
    Ok(ClipOut { name, per_kw, notes, fnv })
}

/// Runs `f` over `items` on `jobs` threads, results in order.
fn par_map<T: Sync, R: Send>(jobs: usize, items: &[T], f: impl Fn(&T) -> R + Sync) -> Vec<R> {
    let next = AtomicUsize::new(0);
    let out: Mutex<Vec<Option<R>>> = Mutex::new((0..items.len()).map(|_| None).collect());
    std::thread::scope(|s| {
        for _ in 0..jobs.min(items.len()) {
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

fn today() -> String {
    let days = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs() / 86400) as i64;
    // Hinnant's civil_from_days.
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!("{y:04}-{m:02}-{d:02}")
}

fn git_head() -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map_or("unknown".into(), |o| String::from_utf8_lossy(&o.stdout).trim().to_string())
}

pub fn run(args: &[String]) -> Result<ExitCode, (bool, String)> {
    let a = parse_args(args).map_err(|e| (true, e))?;
    let model_dir = std::fs::canonicalize(&a.model).map_err(|e| (false, format!("{}: {e}", a.model.display())))?;
    let model = Arc::new(SpotterModel::load(&model_dir).map_err(|e| (false, format!("{}: {e}", a.model.display())))?);
    let mut limits = Limits::default();
    for s in load_set(&model_dir).map_err(|e| (false, e.to_string()))?.spotters {
        let t = &s.thresholds;
        let ov: Vec<f32> = t.overrides.iter().map(|o| o.accept).collect();
        limits.add(&s.name, t.accept, t.notify, &ov, t.probabilistic.map(|p| p.0));
    }
    let env = match (a.rootfs.clone().or_else(|| std::env::var_os("PRYON_ORACLE_ROOTFS").map(Into::into)), a.pryon_test.clone().or_else(|| std::env::var_os("PRYON_PRYON_TEST").map(Into::into))) {
        (Some(rootfs), Some(pryon_test)) => Some(OracleEnv { rootfs, pryon_test }),
        _ => None,
    };
    let checksum = model.checksum().to_string();
    let scratch = std::env::temp_dir().join(format!("pryon-validate-{}", std::process::id()));
    std::fs::create_dir_all(&scratch).map_err(|e| (false, e.to_string()))?;
    let ctx = Ctx { args: &a, model: model.clone(), manifest: model_dir.join("pryon.manifest"), limits, checksum: checksum.clone(), env, scratch: scratch.clone() };
    let results = par_map(a.jobs, &a.clips, |p| one_clip(&ctx, p));
    let _ = std::fs::remove_dir_all(&scratch);

    let mut errors = 0usize;
    let mut totals: BTreeMap<String, Tally> = BTreeMap::new();
    let mut fnvs: BTreeMap<String, Vec<u64>> = BTreeMap::new();
    let mut all_fnvs = Vec::new();
    let mut ok_clips = 0usize;
    for (p, r) in a.clips.iter().zip(results) {
        match r {
            Err(e) => {
                errors += 1;
                println!("ERROR {}: {e}", p.display());
            }
            Ok(c) => {
                ok_clips += 1;
                all_fnvs.push(c.fnv);
                for n in &c.notes {
                    println!("{n}");
                }
                for (k, (t, diffs, prob)) in &c.per_kw {
                    for d in diffs {
                        println!("DIFF {} {}", c.name, d);
                    }
                    for e in prob {
                        println!("PROB {} {k} oracle={e}", c.name);
                    }
                    totals.entry(k.clone()).or_default().add(t);
                    fnvs.entry(k.clone()).or_default().push(c.fnv);
                }
            }
        }
    }
    let mut differences = 0usize;
    for (k, t) in &totals {
        differences += t.differing + t.oracle_only + t.engine_only;
        println!("SUMMARY {k}: {}{}", t.line(), if t.prob > 0 { format!(" | PROB {}", t.prob) } else { String::new() });
    }
    println!("TOTAL clips {} errors {errors} differences {differences} (set {checksum})", a.clips.len());
    if errors > 0 {
        return Ok(ExitCode::from(2));
    }
    if differences > 0 {
        return Ok(ExitCode::from(1));
    }
    if a.exact {
        all_fnvs.sort_unstable();
        let digest = fnv1a64(&all_fnvs.iter().flat_map(|f| f.to_be_bytes()).collect::<Vec<u8>>());
        let (date, rev) = (today(), git_head());
        std::fs::create_dir_all(&a.records).map_err(|e| (false, format!("{}: {e}", a.records.display())))?;
        for (k, t) in totals.iter().filter(|(_, t)| t.accepts > 0) {
            let f = a.records.join(format!("{checksum}.{k}.rec"));
            let body = format!(
                "set={checksum}\nkeyword={k}\nclips={ok_clips}\nclips_digest={digest:016x}\naccepts={}\nnear_misses={}\ndate={date}\npryon_rs={rev}\n",
                t.accepts, t.near_misses
            );
            std::fs::write(&f, body).map_err(|e| (false, format!("{}: {e}", f.display())))?;
            println!("RECORD {}", f.display());
        }
    }
    Ok(ExitCode::SUCCESS)
}
