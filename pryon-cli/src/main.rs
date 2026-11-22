use pryon::{SoundDetector, SoundModel, SoundWindow};
use pryon_testkit::{synth, write_model_dir, Clip};
use std::path::Path;
use std::process::{Command, ExitCode};
use std::sync::Arc;
use std::time::Instant;

const USAGE: &str = "usage:
  pryon-cli run -m DIR [-t a,b] [--bits] CLIP.raw
  pryon-cli synth NAME OUT.raw
  pryon-cli synth-model DIR [--threshold X]
  pryon-cli bench -m DIR [--chunk N] CLIP.raw
  pryon-cli rusage -- CMD ARGS...";

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
        "bench" => cmd_bench(rest)?,
        "rusage" => return cmd_rusage(rest),
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
            "--threshold" => match val()?.parse::<f32>() {
                Ok(x) => o.threshold = Some(x),
                Err(_) => return usage("--threshold needs a number"),
            },
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

fn cmd_bench(args: &[String]) -> Res {
    let (o, clip) = run_setup(args, &["-m", "--chunk"])?;
    let t0 = Instant::now();
    let mut det = load(o.model.as_ref().unwrap(), &[])?;
    let load_ms = t0.elapsed().as_secs_f64() * 1e3;
    let pcm = read_clip(&clip)?;
    let chunk = o.chunk.unwrap_or(800);
    let mut wins = Vec::new();
    let mut worst = 0.0f64;
    let before = cpu(&self_usage());
    for c in pcm.chunks(chunk) {
        let t = Instant::now();
        det.feed(c, &mut wins);
        worst = worst.max(t.elapsed().as_secs_f64() * 1e6);
    }
    let after = self_usage();
    let cpu_s = cpu(&after) - before;
    let audio_s = pcm.len() as f64 / 16000.0;
    println!("audio_s: {audio_s:.3}");
    println!("cpu_s: {cpu_s:.3}");
    println!("cpu_pct_of_core: {:.2}", if audio_s > 0.0 { 100.0 * cpu_s / audio_s } else { 0.0 });
    println!("worst_feed_us: {worst:.0}");
    println!("max_rss_kib: {}", after.ru_maxrss);
    println!("windows: {}", wins.len());
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
