//! libpryon parity: Rust windows against `aed_test -j` outputs (committed synthetic fixtures, optional local clips).

use pryon::{SoundDetector, SoundModel, SoundWindow};
use pryon_testkit::{synth, Clip};
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::sync::Arc;

const FW: &str = "PRYON_AED_MODEL_DIR";
const DAVS: &str = "PRYON_AED_DAVS_DIR";
const CLIPS: &str = "PRYON_CLIPS_DIR";
const OVERRIDE: &str = "PRYON_ORACLE_OVERRIDE_DIR";

struct OracleType {
    score: String,
    detected: bool,
    near_miss: bool,
    max_energy: i64,
    samples_above: i64,
}

fn env_dir(var: &str) -> Option<PathBuf> {
    match std::env::var(var) {
        Ok(d) => Some(PathBuf::from(d)),
        Err(_) => {
            println!("skipped: set {var}");
            None
        }
    }
}

fn model(dir: &Path) -> Arc<SoundModel> {
    Arc::new(SoundModel::load(dir).unwrap())
}

fn detect(m: &Arc<SoundModel>, pcm: &[i16]) -> (Vec<String>, Vec<SoundWindow>) {
    let mut det = SoundDetector::new(m.clone(), &[]).unwrap();
    let mut out = Vec::new();
    for c in pcm.chunks(800) {
        det.feed(c, &mut out);
    }
    (det.type_names().to_vec(), out)
}

fn fixture_in(over: Option<&Path>, rel: &str) -> String {
    let p = match over {
        Some(d) => d.join(rel),
        None => Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/oracle").join(rel),
    };
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()))
}

fn fixture(rel: &str) -> String {
    fixture_in(std::env::var_os(OVERRIDE).map(PathBuf::from).as_deref(), rel)
}

fn field<'a>(o: &'a Value, k: &str, what: &str) -> &'a Value {
    o.get(k).unwrap_or_else(|| panic!("{what}: no {k}"))
}

/// Per window, the oracle result of each named type, read from the window's JSON line.
fn parse_oracle(text: &str, types: &[String], what: &str) -> Vec<Vec<OracleType>> {
    let headers = text.lines().filter(|l| l.starts_with("window ")).count();
    let wins: Vec<Vec<OracleType>> = text
        .lines()
        .filter(|l| l.trim_start().starts_with('{'))
        .map(|l| {
            let v: Value = serde_json::from_str(l.trim()).unwrap();
            let aed = field(&v, "amzn1.activity.device.detection.audio.aed", what);
            types
                .iter()
                .map(|n| {
                    let t = field(aed, n, what);
                    let num = |k| field(t, k, what).as_f64().unwrap();
                    let int = |k| field(t, k, what).as_i64().unwrap();
                    let flag = |k| field(t, k, what).as_bool().unwrap();
                    OracleType {
                        score: format!("{:.6}", num("score")),
                        detected: flag("detected"),
                        near_miss: flag("nearMiss"),
                        max_energy: int("maxEnergyLevel"),
                        samples_above: int("numSamplesAboveEnergyThreshold"),
                    }
                })
                .collect()
        })
        .collect();
    assert_eq!(wins.len(), headers, "{what}: every window needs its JSON line");
    wins
}

fn compare(m: &Arc<SoundModel>, pcm: &[i16], oracle_text: &str, what: &str) {
    let (names, got) = detect(m, pcm);
    let want = parse_oracle(oracle_text, &names, what);
    assert_eq!(got.len(), want.len(), "{what}: window count");
    for (w, (g, o)) in got.iter().zip(&want).enumerate() {
        for ((n, t), ot) in names.iter().zip(&g.types).zip(o) {
            let at = format!("{what} window {} {n}", w + 1);
            assert_eq!(format!("{:.6}", t.score), ot.score, "{at}: score");
            assert_eq!(t.detected, ot.detected, "{at}: detected");
            assert_eq!(t.near_miss, ot.near_miss, "{at}: near miss");
            assert_eq!(g.max_energy as i64, ot.max_energy, "{at}: max energy");
            assert_eq!(t.samples_above as i64, ot.samples_above, "{at}: samples above");
        }
    }
}

fn oracle(var: &str, sub: &str, clip: Clip) {
    let Some(dir) = env_dir(var) else { return };
    compare(&model(&dir), &synth(clip), &fixture(&format!("{sub}/{}.txt", clip.name())), clip.name());
}

// Bits lines hold the 12 scores in score_output_index order, which is AED.json order.
fn bits(clip: Clip) {
    let Some(dir) = env_dir(DAVS) else { return };
    let (_, got) = detect(&model(&dir), &synth(clip));
    let want: Vec<String> = fixture(&format!("davs/bits/{}.txt", clip.name())).lines().map(String::from).collect();
    assert_eq!(got.len(), want.len(), "{}: window count", clip.name());
    for (w, (g, line)) in got.iter().zip(&want).enumerate() {
        let b: Vec<String> = g.types.iter().map(|t| format!("{:08x}", t.score.to_bits())).collect();
        assert_eq!(b.join(" "), *line, "{} window {}", clip.name(), w + 1);
    }
}

macro_rules! parity {
    ($clip:ident, $fw:ident, $davs:ident, $bits:ident) => {
        #[test]
        fn $fw() {
            oracle(FW, "fw", Clip::$clip);
        }
        #[test]
        fn $davs() {
            oracle(DAVS, "davs", Clip::$clip);
        }
        #[test]
        fn $bits() {
            bits(Clip::$clip);
        }
    };
}

parity!(Noise, oracle_fw_noise, oracle_davs_noise, bits_davs_noise);
parity!(Tones, oracle_fw_tones, oracle_davs_tones, bits_davs_tones);
parity!(Gap49, oracle_fw_gap49, oracle_davs_gap49, bits_davs_gap49);
parity!(Sparse, oracle_fw_sparse, oracle_davs_sparse, bits_davs_sparse);
parity!(Zeros, oracle_fw_zeros, oracle_davs_zeros, bits_davs_zeros);
parity!(Quiet, oracle_fw_quiet, oracle_davs_quiet, bits_davs_quiet);
parity!(Extremes, oracle_fw_extremes, oracle_davs_extremes, bits_davs_extremes);
parity!(QuietTones, oracle_fw_quiet_tones, oracle_davs_quiet_tones, bits_davs_quiet_tones);

#[test]
fn every_clip_has_fixtures() {
    for c in Clip::ALL {
        for rel in ["fw", "davs", "davs/bits"] {
            assert!(!fixture(&format!("{rel}/{}.txt", c.name())).is_empty(), "{rel}/{}", c.name());
        }
    }
}

fn read_raw(p: &Path) -> Vec<i16> {
    let b = std::fs::read(p).unwrap();
    b.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]])).collect()
}

/// Every `<name>.raw` in PRYON_CLIPS_DIR with `<name>_out.txt` (DAVS) / `<name>_fw_out.txt` (firmware).
#[test]
fn local_clips() {
    let Some(clips) = env_dir(CLIPS) else { return };
    let mut raws: Vec<PathBuf> = std::fs::read_dir(&clips)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "raw"))
        .collect();
    raws.sort();
    let mut checked = 0;
    let mut models = 0;
    for (var, suffix) in [(DAVS, "_out.txt"), (FW, "_fw_out.txt")] {
        let Some(dir) = env_dir(var) else { continue };
        let m = model(&dir);
        models += 1;
        for raw in &raws {
            let name = raw.file_stem().unwrap().to_str().unwrap();
            let Ok(text) = std::fs::read_to_string(clips.join(format!("{name}{suffix}"))) else { continue };
            compare(&m, &read_raw(raw), &text, &format!("{name}{suffix}"));
            checked += 1;
        }
    }
    assert!(models == 0 || checked > 0, "no clip in {} has an oracle output", clips.display());
    println!("local clips checked: {checked}");
}

#[test]
fn override_dir_used() {
    let d = pryon_testkit::TempDir::new("oracle-override");
    std::fs::create_dir_all(d.join("fw")).unwrap();
    std::fs::write(d.join("fw/noise.txt"), "override marker").unwrap();
    assert_eq!(fixture_in(Some(&d), "fw/noise.txt"), "override marker");
    assert_ne!(fixture_in(None, "fw/noise.txt"), "override marker");
}
