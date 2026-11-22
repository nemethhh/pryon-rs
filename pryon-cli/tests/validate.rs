//! `pryon-cli validate` with hand-written oracle outputs in `--cache` (no qemu).

use pryon::wake::SpotterModel;
use pryon_testkit::compare::{pad_clip, LEAD_MS, TAIL_MS};
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
        let events = cli(&["wake", "-m", &s(&set), &s(&padded_clip)]);
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
        let key = fnv1a64(&padded);
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
