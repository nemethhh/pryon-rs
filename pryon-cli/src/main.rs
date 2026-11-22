use pryon::{SoundDetector, SoundModel, SoundWindow};
use pryon_testkit::wakeset::{write_davs_set, write_fw_set, WakeSetOpts};
use pryon_testkit::{synth, write_model_dir, Clip};
use std::io::Read;
use std::path::Path;
use std::process::{Command, ExitCode};
use std::sync::Arc;
use std::time::Instant;

mod validate;

const USAGE: &str = "usage:
  pryon-cli run -m DIR [-t a,b] [--bits] CLIP.raw
  pryon-cli synth NAME OUT.raw
  pryon-cli synth-model DIR [--threshold X]
  pryon-cli synth-wake-set fw|davs DIR [--seed N] [--force SCORE]
  pryon-cli inspect DIR
  pryon-cli wake-dump -m SET [--lfbe OUT.f32] [--out DIR] CLIP.raw
  pryon-cli wake -m SET [-p NAME=VALUE[@SAMPLE]]... [--reset-at SAMPLE]... [--chunk N] [--bits] CLIP.raw
  pryon-cli bench-wake -m SET [--chunk N] CLIP.raw
  pryon-cli bench -m DIR [--chunk N] [--speed X] CLIP.raw
  pryon-cli rusage -- CMD ARGS...
  pryon-cli validate [--oracle-rootfs DIR --pryon-test PATH] [--cache DIR] [--exact] [--records DIR]
      [--speed X] [--jobs N] [-p NAME=VALUE[@SAMPLE]]... MODEL CLIPS...";

enum Fail {
    Usage(String),
    Run(String),
}

type Res = Result<(), Fail>;

fn usage<T>(msg: impl Into<String>) -> Result<T, Fail> {
    Err(Fail::Usage(msg.into()))
}

fn rt<T, E: std::fmt::Display>(what: &str, r: Result<T, E>) -> Result<T, Fail> {
    r.map_err(|e| Fail::Run(format!("{what}: {e}")))
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match dispatch(&args) {
        Ok(code) => code,
        Err(Fail::Usage(m)) => {
            eprintln!("{m}\n{USAGE}");
            ExitCode::from(2)
        }
        Err(Fail::Run(m)) => {
            eprintln!("{m}");
            ExitCode::FAILURE
        }
    }
}

fn dispatch(args: &[String]) -> Result<ExitCode, Fail> {
    let Some((cmd, rest)) = args.split_first() else {
        return usage("missing command");
    };
    match cmd.as_str() {
        "run" => cmd_run(rest)?,
        "synth" => cmd_synth(rest)?,
        "synth-model" => cmd_synth_model(rest)?,
        "synth-wake-set" => cmd_synth_wake_set(rest)?,
        "inspect" => cmd_inspect(rest)?,
        "wake-dump" => cmd_wake_dump(rest)?,
        "wake" => cmd_wake(rest)?,
        "bench-wake" => cmd_bench_wake(rest)?,
        "bench" => cmd_bench(rest)?,
        "rusage" => return cmd_rusage(rest),
        "validate" => {
            return match validate::run(rest) {
                Err((true, m)) => Err(Fail::Usage(m)),
                Err((false, m)) => {
                    eprintln!("{m}");
                    Ok(ExitCode::from(2))
                }
                Ok(c) => Ok(c),
            };
        }
        _ => return usage(format!("unknown command {cmd}")),
    }
    Ok(ExitCode::SUCCESS)
}

#[derive(Default)]
struct Opts {
    model: Option<String>,
    types: Vec<String>,
    bits: bool,
    chunk: Option<usize>,
    threshold: Option<f32>,
    speed: Option<f64>,
    seed: Option<u64>,
    force: Option<f32>,
    lfbe: Option<String>,
    out: Option<String>,
    /// (sample, name, value).
    props: Vec<(u64, String, i32)>,
    resets: Vec<u64>,
    pos: Vec<String>,
}

fn parse(args: &[String], allowed: &[&str]) -> Result<Opts, Fail> {
    let mut o = Opts::default();
    let mut it = args.iter();
    while let Some(a) = it.next() {
        if a.starts_with('-') && a.len() > 1 && !allowed.contains(&a.as_str()) {
            return usage(format!("unknown option {a}"));
        }
        let mut val = || it.next().cloned().ok_or_else(|| Fail::Usage(format!("{a} needs a value")));
        match a.as_str() {
            "-m" => o.model = Some(val()?),
            "-t" => o.types = val()?.split(',').filter(|s| !s.is_empty()).map(String::from).collect(),
            "--bits" => o.bits = true,
            "--chunk" => match val()?.parse::<usize>() {
                Ok(n) if n > 0 => o.chunk = Some(n),
                _ => return usage("--chunk needs a positive integer"),
            },
            "--speed" => match val()?.parse::<f64>() {
                Ok(x) if x > 0.0 && x.is_finite() => o.speed = Some(x),
                _ => return usage("--speed needs a positive number"),
            },
            "--threshold" => match val()?.parse::<f32>() {
                Ok(x) => o.threshold = Some(x),
                Err(_) => return usage("--threshold needs a number"),
            },
            "--seed" => match val()?.parse::<u64>() {
                Ok(n) => o.seed = Some(n),
                Err(_) => return usage("--seed needs a non-negative integer"),
            },
            "--force" => match val()?.parse::<f32>() {
                Ok(x) if x > 0.0 && x < 1.0 => o.force = Some(x),
                _ => return usage("--force needs a score strictly between 0 and 1"),
            },
            "-p" => {
                let v = val()?;
                let (assign, at) = match v.split_once('@') {
                    Some((a, s)) => (a, s.parse::<u64>().ok()),
                    None => (v.as_str(), Some(0)),
                };
                match (assign.split_once('='), at) {
                    (Some((n, x)), Some(at)) if !n.is_empty() => match x.parse::<i32>() {
                        Ok(x) => o.props.push((at, n.to_string(), x)),
                        Err(_) => return usage(format!("-p {v}: VALUE must be an integer")),
                    },
                    _ => return usage(format!("-p {v}: expected NAME=VALUE[@SAMPLE]")),
                }
            }
            "--reset-at" => match val()?.parse::<u64>() {
                Ok(n) => o.resets.push(n),
                Err(_) => return usage("--reset-at needs a sample index"),
            },
            "--lfbe" => o.lfbe = Some(val()?),
            "--out" => o.out = Some(val()?),
            _ => o.pos.push(a.clone()),
        }
    }
    Ok(o)
}

fn read_clip(path: &str) -> Result<Vec<i16>, Fail> {
    let bytes = rt(path, std::fs::read(path))?;
    if bytes.len() % 2 != 0 {
        return Err(Fail::Run(format!("{path}: odd byte count, not s16le")));
    }
    Ok(bytes.chunks_exact(2).map(|b| i16::from_le_bytes([b[0], b[1]])).collect())
}

fn load(model: &str, types: &[String]) -> Result<SoundDetector, Fail> {
    let m = rt(model, SoundModel::load(Path::new(model)))?;
    let names: Vec<&str> = types.iter().map(String::as_str).collect();
    rt("detector", SoundDetector::new(Arc::new(m), &names))
}

fn run_setup(args: &[String], allowed: &[&str]) -> Result<(Opts, String), Fail> {
    let o = parse(args, allowed)?;
    if o.model.is_none() {
        return usage("-m DIR is required");
    }
    let [clip] = o.pos.as_slice() else {
        return usage("expected exactly one CLIP.raw");
    };
    let clip = clip.clone();
    Ok((o, clip))
}

fn print_window(names: &[String], w: &SoundWindow, bits: bool) {
    println!("window {} end={:.2}s", w.index + 1, w.end_sample as f64 / 16000.0);
    for (n, t) in names.iter().zip(&w.types) {
        let mut line = format!("  {n:<20} {:.6}", t.score);
        if t.detected {
            line.push_str("  detected");
        }
        if t.near_miss {
            line.push_str("  near miss");
        }
        if bits {
            line.push_str(&format!(" {:08x}", t.score.to_bits()));
        }
        println!("{line}");
    }
    let above: Vec<String> = w.types.iter().map(|t| t.samples_above.to_string()).collect();
    println!("  energy max={} above={}", w.max_energy, above.join(","));
}

fn cmd_run(args: &[String]) -> Res {
    let (o, clip) = run_setup(args, &["-m", "-t", "--bits"])?;
    let mut det = load(o.model.as_ref().unwrap(), &o.types)?;
    let pcm = read_clip(&clip)?;
    let mut wins = Vec::new();
    det.feed(&pcm, &mut wins);
    for w in &wins {
        print_window(det.type_names(), w, o.bits);
    }
    Ok(())
}

fn cmd_synth(args: &[String]) -> Res {
    let [name, out] = args else {
        return usage("synth needs NAME OUT.raw");
    };
    let Some(clip) = Clip::from_name(name) else {
        return usage(format!("unknown clip {name}"));
    };
    let bytes: Vec<u8> = synth(clip).iter().flat_map(|s| s.to_le_bytes()).collect();
    rt(out, std::fs::write(out, bytes))
}

fn cmd_synth_model(args: &[String]) -> Res {
    let o = parse(args, &["--threshold"])?;
    let [dir] = o.pos.as_slice() else {
        return usage("synth-model needs DIR");
    };
    rt(dir, std::fs::create_dir_all(dir))?;
    write_model_dir(Path::new(dir), 1, o.threshold.unwrap_or(0.5));
    Ok(())
}

fn cmd_synth_wake_set(args: &[String]) -> Res {
    let o = parse(args, &["--seed", "--force"])?;
    let [layout, dir] = o.pos.as_slice() else {
        return usage("synth-wake-set needs fw|davs DIR");
    };
    let opts = WakeSetOpts { seed: o.seed.unwrap_or(1), force_score: o.force };
    rt(dir, std::fs::create_dir_all(dir))?;
    match layout.as_str() {
        "fw" => write_fw_set(Path::new(dir), &opts),
        "davs" => write_davs_set(Path::new(dir), &opts),
        _ => return usage(format!("unknown layout {layout} (fw or davs)")),
    }
    Ok(())
}

fn cmd_inspect(args: &[String]) -> Res {
    let [dir] = args else {
        return usage("inspect needs DIR");
    };
    let set = rt(dir, pryon::wake::load_set(Path::new(dir)))?;
    print!("{}", rt(dir, pryon::wake::describe(&set))?);
    Ok(())
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Rust LFBE frames and per-step stage outputs (`<key>.f32`, steps x values) for reference comparisons.
fn cmd_wake_dump(args: &[String]) -> Res {
    use pryon::wake::{LoadedSet, Pipeline, StepOut};
    let (o, clip) = run_setup(args, &["-m", "--lfbe", "--out"])?;
    let model = o.model.as_ref().unwrap();
    let set = Arc::new(rt(model, LoadedSet::load(Path::new(model)))?);
    let pcm = read_clip(&clip)?;
    if let Some(path) = &o.lfbe {
        let mut fe = pryon::lfbe::Lfbe::with_spec(set.spec.fe);
        let mut frames = Vec::new();
        fe.push_flat(&pcm, &mut frames);
        rt(path, std::fs::write(path, f32_bytes(&frames)))?;
    }
    let mut rows: Vec<Vec<f32>> = vec![Vec::new(); set.components.len()];
    let mut steps = 0u64;
    let mut frames = 0u64;
    Pipeline::new(&set).push(&pcm, &mut |s: &StepOut| {
        for (r, v) in rows.iter_mut().zip(s.values) {
            r.extend_from_slice(v);
        }
        steps = s.step + 1;
        frames = s.frames;
    });
    println!("set {} checksum {}: {steps} steps, {frames} frames", model, set.spec.checksum);
    for (c, r) in set.components.iter().zip(&rows) {
        println!("  {} {:?}", c.key, c.shape);
        if let Some(dir) = &o.out {
            let path = Path::new(dir).join(format!("{}.f32", c.key));
            rt(dir, std::fs::create_dir_all(path.parent().unwrap()))?;
            rt(&path.display().to_string(), std::fs::write(&path, f32_bytes(r)))?;
        }
    }
    Ok(())
}

fn load_spotter(model: &str) -> Result<(pryon::wake::Spotter, f64), Fail> {
    use pryon::wake::{Spotter, SpotterModel, SpotterOptions};
    let t0 = Instant::now();
    let m = Arc::new(rt(model, SpotterModel::load(Path::new(model)))?);
    let sp = rt(model, Spotter::new(m, SpotterOptions::default()))?;
    Ok((sp, t0.elapsed().as_secs_f64() * 1e3))
}

fn print_detection(keywords: &[String], d: &pryon::wake::Detection, bits: bool) {
    let mut line = format!(
        "EVENT keyword={} kind={:?} score={:.6} begin={} end={}",
        keywords[d.keyword], d.kind, d.score, d.begin_sample, d.end_sample
    );
    if let Some(a) = d.afe {
        line.push_str(&format!(" afe={},{}", a.start, a.end));
    }
    if bits {
        line.push_str(&format!(" bits={:08x}", d.score.to_bits()));
    }
    println!("{line}");
}

/// Feeds the clip in `--chunk` pieces, applying properties and session ends at their samples (before that sample).
fn cmd_wake(args: &[String]) -> Res {
    let (o, clip) = run_setup(args, &["-m", "-p", "--reset-at", "--chunk", "--bits"])?;
    let (mut sp, _) = load_spotter(o.model.as_ref().unwrap())?;
    let pcm = read_clip(&clip)?;
    let chunk = o.chunk.unwrap_or(pcm.len().max(1));
    // (sample, order, command): properties before session ends at the same sample, each in command-line order
    let mut cmds: Vec<(u64, usize, Option<(&str, i32)>)> = Vec::new();
    for (i, (at, n, v)) in o.props.iter().enumerate() {
        cmds.push((*at, i, Some((n.as_str(), *v))));
    }
    for (i, at) in o.resets.iter().enumerate() {
        cmds.push((*at, o.props.len() + i, None));
    }
    cmds.sort();
    let mut out = Vec::new();
    let mut pos = 0usize;
    let mut next = cmds.iter().peekable();
    loop {
        while let Some(&&(at, _, c)) = next.peek() {
            if at > pos as u64 {
                break;
            }
            match c {
                Some((n, v)) => sp.set_property(n, v),
                None => sp.reset(),
            }
            next.next();
        }
        if pos == pcm.len() {
            break;
        }
        let stop = next.peek().map_or(pcm.len(), |c| (c.0 as usize).min(pcm.len()));
        let end = (pos + chunk).min(stop);
        sp.feed(&pcm[pos..end], &mut out);
        pos = end;
    }
    for d in &out {
        print_detection(sp.keywords(), d, o.bits);
    }
    Ok(())
}

fn self_usage() -> libc::rusage {
    // SAFETY: zeroed rusage is a valid out-buffer for getrusage.
    let mut ru: libc::rusage = unsafe { std::mem::zeroed() };
    unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut ru) };
    ru
}

fn secs(tv: libc::timeval) -> f64 {
    tv.tv_sec as f64 + tv.tv_usec as f64 / 1e6
}

fn cpu(ru: &libc::rusage) -> f64 {
    secs(ru.ru_utime) + secs(ru.ru_stime)
}

/// Fills `buf` unless the file ends first; returns the bytes read.
fn read_full(f: &mut std::fs::File, buf: &mut [u8]) -> std::io::Result<usize> {
    let mut n = 0;
    while n < buf.len() {
        match f.read(&mut buf[n..]) {
            Ok(0) => break,
            Ok(k) => n += k,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(n)
}

fn cmd_bench(args: &[String]) -> Res {
    let (o, clip) = run_setup(args, &["-m", "--chunk", "--speed"])?;
    let t0 = Instant::now();
    let mut det = load(o.model.as_ref().unwrap(), &[])?;
    let load_ms = t0.elapsed().as_secs_f64() * 1e3;
    let mut f = rt(&clip, std::fs::File::open(&clip))?;
    let chunk = o.chunk.unwrap_or(800);
    let mut bytes = vec![0u8; 2 * chunk];
    let mut pcm = Vec::with_capacity(chunk);
    let mut samples = 0u64;
    let (mut wins, mut windows) = (Vec::new(), 0usize);
    let mut worst = 0.0f64;
    let before = cpu(&self_usage());
    loop {
        let n = rt(&clip, read_full(&mut f, &mut bytes))?;
        if n % 2 != 0 {
            return Err(Fail::Run(format!("{clip}: odd byte count, not s16le")));
        }
        if n == 0 {
            break;
        }
        pcm.clear();
        pcm.extend(bytes[..n].chunks_exact(2).map(|b| i16::from_le_bytes([b[0], b[1]])));
        samples += pcm.len() as u64;
        let t = Instant::now();
        det.feed(&pcm, &mut wins);
        worst = worst.max(t.elapsed().as_secs_f64() * 1e6);
        if let Some(x) = o.speed {
            // Paced like aed_test -x, so both engines run under the same CPU frequency governor conditions.
            std::thread::sleep(std::time::Duration::from_secs_f64(pcm.len() as f64 / 16000.0 / x));
        }
        windows += wins.len();
        wins.clear();
        if n < bytes.len() {
            break;
        }
    }
    let after = self_usage();
    let cpu_s = cpu(&after) - before;
    let audio_s = samples as f64 / 16000.0;
    println!("audio_s: {audio_s:.3}");
    println!("cpu_s: {cpu_s:.3}");
    println!("cpu_pct_of_core: {:.2}", if audio_s > 0.0 { 100.0 * cpu_s / audio_s } else { 0.0 });
    println!("worst_feed_us: {worst:.0}");
    println!("max_rss_kib: {}", after.ru_maxrss);
    println!("windows: {windows}");
    println!("load_ms: {load_ms:.1}");
    Ok(())
}

fn cmd_bench_wake(args: &[String]) -> Res {
    let (o, clip) = run_setup(args, &["-m", "--chunk"])?;
    let (mut sp, load_ms) = load_spotter(o.model.as_ref().unwrap())?;
    let mut f = rt(&clip, std::fs::File::open(&clip))?;
    let chunk = o.chunk.unwrap_or(1600);
    let mut bytes = vec![0u8; 2 * chunk];
    let mut pcm = Vec::with_capacity(chunk);
    let mut samples = 0u64;
    let (mut out, mut detections) = (Vec::new(), 0usize);
    let mut worst = 0.0f64;
    let before = cpu(&self_usage());
    loop {
        let n = rt(&clip, read_full(&mut f, &mut bytes))?;
        if n % 2 != 0 {
            return Err(Fail::Run(format!("{clip}: odd byte count, not s16le")));
        }
        if n == 0 {
            break;
        }
        pcm.clear();
        pcm.extend(bytes[..n].chunks_exact(2).map(|b| i16::from_le_bytes([b[0], b[1]])));
        samples += pcm.len() as u64;
        let t = Instant::now();
        sp.feed(&pcm, &mut out);
        worst = worst.max(t.elapsed().as_secs_f64() * 1e6);
        detections += out.len();
        out.clear();
        if n < bytes.len() {
            break;
        }
    }
    let after = self_usage();
    let cpu_s = cpu(&after) - before;
    let audio_s = samples as f64 / 16000.0;
    println!("audio_s: {audio_s:.3}");
    println!("cpu_s: {cpu_s:.3}");
    println!("cpu_pct_of_core: {:.2}", if audio_s > 0.0 { 100.0 * cpu_s / audio_s } else { 0.0 });
    println!("worst_feed_us: {worst:.0}");
    println!("max_rss_kib: {}", after.ru_maxrss);
    println!("detections: {detections}");
    println!("load_ms: {load_ms:.1}");
    Ok(())
}

fn cmd_rusage(args: &[String]) -> Result<ExitCode, Fail> {
    let Some((sep, cmd)) = args.split_first() else {
        return usage("rusage needs -- CMD ARGS...");
    };
    let (Some(prog), "--") = (cmd.first(), sep.as_str()) else {
        return usage("rusage needs -- CMD ARGS...");
    };
    let child = rt(prog, Command::new(prog).args(&cmd[1..]).spawn())?;
    let pid = child.id() as libc::pid_t;
    let mut status = 0;
    // SAFETY: zeroed rusage is a valid out-buffer; the child is reaped only here, never via `child`.
    let mut ru: libc::rusage = unsafe { std::mem::zeroed() };
    let r = unsafe { libc::wait4(pid, &mut status, 0, &mut ru) };
    if r < 0 {
        return Err(Fail::Run(format!("wait4: {}", std::io::Error::last_os_error())));
    }
    let code = if libc::WIFEXITED(status) {
        libc::WEXITSTATUS(status)
    } else {
        128 + libc::WTERMSIG(status)
    };
    println!("user_s: {:.3}", secs(ru.ru_utime));
    println!("sys_s: {:.3}", secs(ru.ru_stime));
    println!("cpu_s: {:.3}", cpu(&ru));
    println!("max_rss_kib: {}", ru.ru_maxrss);
    println!("exit: {code}");
    Ok(ExitCode::from(code as u8))
}
