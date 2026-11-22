use std::path::PathBuf;
use std::process::Command;

const BIN: &str = env!("CARGO_BIN_EXE_pryon-cli");

fn scratch(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("pryon-cli-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn ok(args: &[&str]) -> String {
    let out = Command::new(BIN).args(args).output().unwrap();
    assert!(out.status.success(), "{args:?}: {}", String::from_utf8_lossy(&out.stderr));
    String::from_utf8(out.stdout).unwrap()
}

fn setup(tag: &str) -> (PathBuf, String, String) {
    let d = scratch(tag);
    let model = d.join("model");
    let clip = d.join("noise.raw");
    ok(&["synth-model", model.to_str().unwrap()]);
    ok(&["synth", "noise", clip.to_str().unwrap()]);
    (d, model.to_str().unwrap().into(), clip.to_str().unwrap().into())
}

#[test]
fn run_prints_windows() {
    let (d, model, clip) = setup("run");
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
    let _ = std::fs::remove_dir_all(d);
}

#[test]
fn bench_prints_keys() {
    let (d, model, clip) = setup("bench");
    let out = ok(&["bench", "-m", &model, "--chunk", "1600", &clip]);
    for k in ["audio_s", "cpu_s", "cpu_pct_of_core", "worst_feed_us", "max_rss_kib", "windows", "load_ms"] {
        assert!(out.lines().any(|l| l.starts_with(&format!("{k}: "))), "{k}:\n{out}");
    }
    let _ = std::fs::remove_dir_all(d);
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
    let _ = std::fs::remove_dir_all(d);
}
