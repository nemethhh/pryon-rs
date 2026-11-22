use pryon_testkit::TempDir;
use std::process::Command;

const BIN: &str = env!("CARGO_BIN_EXE_pryon-cli");

fn scratch(tag: &str) -> TempDir {
    TempDir::new(&format!("cli-{tag}"))
}

fn ok(args: &[&str]) -> String {
    let out = Command::new(BIN).args(args).output().unwrap();
    assert!(out.status.success(), "{args:?}: {}", String::from_utf8_lossy(&out.stderr));
    String::from_utf8(out.stdout).unwrap()
}

fn setup(tag: &str) -> (TempDir, String, String) {
    let d = scratch(tag);
    let model = d.join("model");
    let clip = d.join("noise.raw");
    ok(&["synth-model", model.to_str().unwrap()]);
    ok(&["synth", "noise", clip.to_str().unwrap()]);
    (d, model.to_str().unwrap().into(), clip.to_str().unwrap().into())
}

#[test]
fn run_prints_windows() {
    let (_d, model, clip) = setup("run");
    let out = ok(&["run", "-m", &model, &clip]);
    for header in ["window 1 end=9.98s", "window 2 end=19.96s"] {
        let i = out.find(header).unwrap_or_else(|| panic!("missing {header}:\n{out}"));
        let rest = &out[i + header.len()..];
        let body = rest.split("window ").next().unwrap();
        let types = body.lines().filter(|l| l.starts_with("  ") && !l.starts_with("  energy")).count();
        assert_eq!(types, 12, "{header}:\n{out}");
        assert!(body.contains("  energy max="));
    }
    let bits = ok(&["run", "-m", &model, "--bits", &clip]);
    let line = bits.lines().find(|l| l.starts_with("  ") && !l.starts_with("  energy")).unwrap();
    let last = line.split_whitespace().last().unwrap();
    assert!(last.len() == 8 && u32::from_str_radix(last, 16).is_ok(), "{line}");
}

#[test]
fn bench_prints_keys() {
    let (_d, model, clip) = setup("bench");
    let out = ok(&["bench", "-m", &model, "--chunk", "1600", &clip]);
    for k in ["audio_s", "cpu_s", "cpu_pct_of_core", "worst_feed_us", "max_rss_kib", "windows", "load_ms"] {
        assert!(out.lines().any(|l| l.starts_with(&format!("{k}: "))), "{k}:\n{out}");
    }
}

fn key(out: &str, k: &str) -> f64 {
    let line = out.lines().find_map(|l| l.strip_prefix(&format!("{k}: ")));
    line.unwrap_or_else(|| panic!("{k}:\n{out}")).parse().unwrap()
}

#[test]
fn bench_streams_clip() {
    let (d, model, _) = setup("stream");
    // 12 MiB of digital silence: a whole-file buffer would show in max RSS.
    let clip = d.join("big.raw");
    std::fs::write(&clip, vec![0u8; 12 << 20]).unwrap();
    let out = ok(&["bench", "-m", &model, "--chunk", "1601", clip.to_str().unwrap()]);
    assert_eq!(key(&out, "audio_s"), (6 << 20) as f64 / 16000.0, "{out}");
    assert_eq!(key(&out, "windows"), ((6 << 20) / 159_680) as f64, "{out}");
    assert!(key(&out, "max_rss_kib") < 16384.0, "{out}");
    std::fs::write(&clip, [0u8; 3201]).unwrap();
    let out = Command::new(BIN).args(["bench", "-m", &model, clip.to_str().unwrap()]).output().unwrap();
    assert_eq!(out.status.code(), Some(1));
}

#[test]
fn bench_speed_paces_feeds() {
    let (_d, model, clip) = setup("speed");
    // 25 s of audio at 40x real time: at least half a second of wall time.
    let t = std::time::Instant::now();
    let out = ok(&["bench", "-m", &model, "--speed", "40", &clip]);
    assert!(t.elapsed().as_secs_f64() >= 0.5, "{out}");
    assert_eq!(key(&out, "windows"), 2.0, "{out}");
    let bad = Command::new(BIN).args(["bench", "-m", &model, "--speed", "0", &clip]).output().unwrap();
    assert_eq!(bad.status.code(), Some(2));
}

#[test]
fn rusage_reports_child() {
    let out = Command::new(BIN).args(["rusage", "--", "sh", "-c", "echo hi; exit 3"]).output().unwrap();
    assert_eq!(out.status.code(), Some(3));
    let s = String::from_utf8(out.stdout).unwrap();
    assert!(s.contains("hi\n") && s.contains("exit: 3") && s.contains("max_rss_kib: "), "{s}");
}

#[test]
fn usage_errors_exit_2() {
    let out = Command::new(BIN).args(["run"]).output().unwrap();
    assert_eq!(out.status.code(), Some(2));
    let d = scratch("odd");
    let clip = d.join("odd.raw");
    std::fs::write(&clip, [1u8, 2, 3]).unwrap();
    let out = Command::new(BIN).args(["run", "-m", "/nonexistent", clip.to_str().unwrap()]).output().unwrap();
    assert_eq!(out.status.code(), Some(1));
}
