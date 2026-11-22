//! `pryon-cli validate` with hand-written oracle outputs in `--cache` (no qemu).

use pryon::wake::SpotterModel;
use pryon_testkit::compare::{oracle_cache_key, pad_clip, LEAD_MS, TAIL_MS};
use pryon_testkit::oracle::OracleCmd;
use pryon_testkit::{fnv1a64, TempDir};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const BIN: &str = env!("CARGO_BIN_EXE_pryon-cli");

/// Without the oracle environment: a cache miss must fail, not run qemu.
fn cli(args: &[&str]) -> Output {
    Command::new(BIN).args(args).env_remove("PRYON_ORACLE_ROOTFS").env_remove("PRYON_PRYON_TEST").output().unwrap()
}

struct Fixture {
    dir: TempDir,
    set: PathBuf,
    clip: PathBuf,
    cache_file: PathBuf,
}

impl Fixture {
    fn new(tag: &str) -> Fixture {
        Fixture::with_session_ends(tag, &[])
    }

    /// The cached oracle output is the engine's own run with session ends before these clip samples.
    fn with_session_ends(tag: &str, ends: &[u64]) -> Fixture {
        let dir = TempDir::new(&format!("validate-{tag}"));
        let set = dir.join("set");
        let clip = dir.join("noise.raw");
        let s = |p: &Path| p.to_str().unwrap().to_string();
        assert!(cli(&["synth-wake-set", "fw", &s(&set), "--force", "0.99"]).status.success());
        assert!(cli(&["synth", "noise", &s(&clip)]).status.success());
        // the oracle sees the clip between validate's default silences
        let pcm: Vec<i16> = std::fs::read(&clip).unwrap().chunks_exact(2).map(|b| i16::from_le_bytes([b[0], b[1]])).collect();
        let padded: Vec<u8> = pad_clip(&pcm, LEAD_MS, TAIL_MS).iter().flat_map(|x| x.to_le_bytes()).collect();
        let padded_clip = dir.join("padded.raw");
        std::fs::write(&padded_clip, &padded).unwrap();
        let lead = u64::from(LEAD_MS) * 16;
        let shifted: Vec<u64> = ends.iter().map(|&at| if at == 0 { 0 } else { at + lead }).collect();
        let mut args = vec!["wake".to_string(), "-m".into(), s(&set)];
        for at in &shifted {
            args.extend(["--reset-at".to_string(), at.to_string()]);
        }
        args.push(s(&padded_clip));
        let events = cli(&args.iter().map(String::as_str).collect::<Vec<_>>());
        assert!(events.status.success());
        let mut text = String::new();
        for l in String::from_utf8(events.stdout).unwrap().lines().filter(|l| l.starts_with("EVENT ")) {
            let f = |k: &str| l[l.find(k).unwrap() + k.len()..].split(' ').next().unwrap().to_string();
            let (kind, ty) = (f("kind="), if f("kind=") == "NearMiss" { 0 } else { 2 });
            text += &format!(
                "RESULT decoder=d keyword={} type={ty} begin={} end={} (x) meta=1\n\
                 pryon[4] INFO: DecoderID=d: recognition_id=d-0: {{KWS}} segmentation_score=-inf,classifier_score={},active_threshold=0.5,detection_type={kind}\n",
                f("keyword="),
                f("begin="),
                f("end="),
                f("score=")
            );
        }
        text += &format!("pushed {} samples, 0 callbacks\n", padded.len() / 2);
        let sum = SpotterModel::load(&set).unwrap().checksum().to_string();
        let cmds: Vec<OracleCmd> = shifted.iter().map(|&at| OracleCmd::SessionEnd { at }).collect();
        let key = oracle_cache_key(fnv1a64(&padded), &cmds, 1.0);
        let cache_file = dir.join("cache").join(&sum).join(format!("{key:016x}.txt"));
        std::fs::create_dir_all(cache_file.parent().unwrap()).unwrap();
        std::fs::write(&cache_file, text).unwrap();
        Fixture { dir, set, clip, cache_file }
    }

    fn validate(&self, extra: &[&str]) -> Output {
        let (cache, records) = (self.dir.join("cache"), self.dir.join("records"));
        let mut a = vec!["validate", "--cache", cache.to_str().unwrap(), "--records", records.to_str().unwrap()];
        a.extend_from_slice(extra);
        a.push(self.set.to_str().unwrap());
        a.push(self.clip.to_str().unwrap());
        cli(&a)
    }

    fn records(&self) -> Vec<String> {
        let mut v: Vec<String> = std::fs::read_dir(self.dir.join("records"))
            .map(|d| d.map(|e| e.unwrap().file_name().to_string_lossy().into_owned()).collect())
            .unwrap_or_default();
        v.sort();
        v
    }
}

fn text(o: &Output) -> String {
    format!("{}{}", String::from_utf8_lossy(&o.stdout), String::from_utf8_lossy(&o.stderr))
}

#[test]
fn cache_hit_clean_exit_0() {
    let f = Fixture::new("clean");
    let o = f.validate(&[]);
    assert_eq!(o.status.code(), Some(0), "{}", text(&o));
    assert!(!text(&o).contains("DIFF "), "{}", text(&o));
    assert!(text(&o).contains("ALEXA"), "{}", text(&o));
}

#[test]
fn diff_reported_exit_1() {
    let f = Fixture::new("diff");
    let t = std::fs::read_to_string(&f.cache_file).unwrap();
    let edited = t.replacen("classifier_score=0.99", "classifier_score=0.50", 1);
    assert_ne!(t, edited);
    std::fs::write(&f.cache_file, edited).unwrap();
    let o = f.validate(&[]);
    assert_eq!(o.status.code(), Some(1), "{}", text(&o));
    let diff = text(&o).lines().find(|l| l.starts_with("DIFF ")).unwrap_or_else(|| panic!("{}", text(&o))).to_string();
    assert!(diff.contains("noise.raw") && diff.contains("oracle=") && diff.contains("rust="), "{diff}");
}

#[test]
fn record_written_on_exact() {
    let f = Fixture::new("record");
    let o = f.validate(&["--exact"]);
    assert_eq!(o.status.code(), Some(0), "{}", text(&o));
    let sum = SpotterModel::load(&f.set).unwrap().checksum().to_string();
    assert_eq!(f.records(), [format!("{sum}.ALEXA.rec")], "{}", text(&o));
    let rec = std::fs::read_to_string(f.dir.join("records").join(format!("{sum}.ALEXA.rec"))).unwrap();
    let keys: Vec<&str> = rec.lines().map(|l| l.split('=').next().unwrap()).collect();
    let want =
        ["set", "keyword", "clips", "clips_digest", "lead_ms", "tail_ms", "accepts", "near_misses", "date", "pryon_rs"];
    assert_eq!(keys, want, "{rec}");
    assert!(rec.contains(&format!("lead_ms={LEAD_MS}\ntail_ms={TAIL_MS}\n")), "{rec}");
    assert!(rec.contains(&format!("set={sum}\n")) && rec.contains("keyword=ALEXA\n") && rec.contains("clips=1\n"), "{rec}");
    let digest = fnv1a64(&fnv1a64(&std::fs::read(&f.clip).unwrap()).to_be_bytes());
    assert!(rec.contains(&format!("clips_digest={digest:016x}\n")), "{rec}");
}

#[test]
fn no_record_without_exact() {
    let f = Fixture::new("noexact");
    assert_eq!(f.validate(&[]).status.code(), Some(0));
    assert!(f.records().is_empty());
}

#[test]
fn no_record_on_diff() {
    let f = Fixture::new("nodiff");
    let t = std::fs::read_to_string(&f.cache_file).unwrap();
    // one digit of the printed score: fine for the margin rule, a difference in --exact
    std::fs::write(&f.cache_file, t.replacen("classifier_score=0.99", "classifier_score=0.98", 1)).unwrap();
    let o = f.validate(&["--exact"]);
    assert_eq!(o.status.code(), Some(1), "{}", text(&o));
    assert!(f.records().is_empty());
}

#[test]
fn padding_keys_cache_and_prints_clip_samples() {
    let f = Fixture::new("padding");
    // unpadded runs are a different oracle input: no cached output, no oracle
    let o = f.validate(&["--lead-ms", "0", "--tail-ms", "0"]);
    assert_eq!(o.status.code(), Some(2), "{}", text(&o));
    assert!(text(&o).contains("no cached oracle output"), "{}", text(&o));
    // events are reported in clip samples: an end after the lead, minus the lead
    let lead = u64::from(LEAD_MS) * 16;
    let t = std::fs::read_to_string(&f.cache_file).unwrap();
    let mut lines: Vec<String> = t.lines().map(str::to_string).collect();
    let end_of = |l: &str| l.split("end=").nth(1).and_then(|x| x.split(' ').next()?.parse::<u64>().ok());
    let i = lines.iter().position(|l| l.starts_with("RESULT") && end_of(l).is_some_and(|e| e > lead)).unwrap();
    let end = end_of(&lines[i]).unwrap();
    lines[i + 1] = lines[i + 1].replace("classifier_score=0.99", "classifier_score=0.50");
    std::fs::write(&f.cache_file, lines.join("\n") + "\n").unwrap();
    let o = f.validate(&[]);
    let want = format!("end={})", end - lead);
    assert!(text(&o).lines().any(|l| l.starts_with("DIFF ") && l.contains(&want)), "{want}\n{}", text(&o));
}

#[test]
fn session_end_script_keyed_and_applied() {
    let f = Fixture::with_session_ends("session-end", &[40_000]);
    // the same script hits the cache and both engines agree
    let o = f.validate(&["-e", "40000"]);
    assert_eq!(o.status.code(), Some(0), "{}", text(&o));
    // without the session end it is another oracle input
    let o = f.validate(&[]);
    assert_eq!(o.status.code(), Some(2), "{}", text(&o));
    assert!(text(&o).contains("no cached oracle output"), "{}", text(&o));
}

#[test]
fn short_oracle_feed_is_an_error() {
    let f = Fixture::new("short-feed");
    let t = std::fs::read_to_string(&f.cache_file).unwrap();
    let pushed = t.lines().find(|l| l.starts_with("pushed ")).unwrap().to_string();
    std::fs::write(&f.cache_file, t.replace(&pushed, "pushed 16000 samples, 0 callbacks")).unwrap();
    let o = f.validate(&[]);
    assert_eq!(o.status.code(), Some(2), "{}", text(&o));
    assert!(text(&o).contains("fed 16000 of"), "{}", text(&o));
}

#[test]
fn scratch_removed_when_the_oracle_set_fails() {
    use std::os::unix::fs::PermissionsExt;
    let f = Fixture::new("scratch");
    // with a checksum file the loader reads only what the set uses; deterministic_set reads every kw.cfg.json
    let manifest = f.set.join("pryon.manifest");
    let m = std::fs::read_to_string(&manifest).unwrap().replacen('{', r#"{ "checksum": "checksum.txt","#, 1);
    std::fs::write(&manifest, m).unwrap();
    std::fs::write(f.set.join("checksum.txt"), "scratch-test").unwrap();
    let bad = f.set.join("unused/kw.cfg.json");
    std::fs::create_dir_all(bad.parent().unwrap()).unwrap();
    std::fs::write(&bad, "{}").unwrap();
    std::fs::set_permissions(&bad, std::fs::Permissions::from_mode(0o000)).unwrap();
    let tmp = f.dir.join("tmp");
    std::fs::create_dir_all(&tmp).unwrap();
    let (cache, records) = (f.dir.join("cache"), f.dir.join("records"));
    let o = Command::new(BIN)
        .args(["validate", "--cache", cache.to_str().unwrap(), "--records", records.to_str().unwrap()])
        .arg(&f.set)
        .arg(&f.clip)
        .env("TMPDIR", &tmp)
        .env_remove("PRYON_ORACLE_ROOTFS")
        .env_remove("PRYON_PRYON_TEST")
        .output()
        .unwrap();
    std::fs::set_permissions(&bad, std::fs::Permissions::from_mode(0o644)).unwrap();
    assert_eq!(o.status.code(), Some(2), "{}", text(&o));
    assert!(text(&o).contains("oracle set"), "{}", text(&o));
    assert_eq!(std::fs::read_dir(&tmp).unwrap().count(), 0, "scratch left behind");
}
