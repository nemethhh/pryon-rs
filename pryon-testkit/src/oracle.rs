//! Runs libpryon's `pryon_test` under qemu-arm and parses its wake events.

use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OracleKind {
    Accept,
    AcceptAuxiliary,
    NearMiss,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OracleEvent {
    pub keyword: String,
    pub kind: OracleKind,
    /// Classifier score and active threshold exactly as libpryon printed them.
    pub score: String,
    pub threshold: String,
    pub begin: u64,
    pub end: u64,
    pub afe: Option<(i64, i64)>,
}

/// The n-th RESULT pairs with the n-th KWS line carrying `classifier_score=` (recognition ids do not pair them).
pub fn parse_wake(output: &str) -> Result<Vec<OracleEvent>, String> {
    #[allow(clippy::type_complexity)]
    let mut results: Vec<(String, u32, u64, u64, Option<(i64, i64)>)> = Vec::new();
    let mut kws: Vec<(String, String, String)> = Vec::new();
    for line in output.lines() {
        if let Some(rest) = line.strip_prefix("RESULT ") {
            let keyword = field(rest, "keyword=").ok_or("RESULT without keyword")?;
            let num = |k: &str| field(rest, k).and_then(|v| v.parse::<u64>().ok()).ok_or(format!("RESULT without {k}"));
            results.push((keyword.to_string(), num("type=")? as u32, num("begin=")?, num("end=")?, None));
        } else if line.starts_with("META ") {
            if let (Some(r), Some(afe)) = (results.last_mut(), meta_afe(line)) {
                r.4 = Some(afe);
            }
        } else if line.contains("{KWS}") && line.contains("classifier_score=") {
            let get = |k: &str| field(line, k).map(str::to_string).ok_or(format!("KWS line without {k}: {line}"));
            kws.push((get("classifier_score=")?, get("active_threshold=")?, get("detection_type=")?));
        }
    }
    if results.len() != kws.len() {
        return Err(format!("{} RESULT lines but {} KWS score lines", results.len(), kws.len()));
    }
    let mut events = Vec::new();
    for (i, ((keyword, ty, begin, end, afe), (score, threshold, dtype))) in results.into_iter().zip(kws).enumerate() {
        if !score.parse::<f32>().is_ok_and(f32::is_finite) {
            return Err(format!("event {i}: classifier_score={score} is not a number"));
        }
        let kind = match (ty, dtype.as_str()) {
            (2, "Accept") => OracleKind::Accept,
            (2, "AcceptAuxiliary") => OracleKind::AcceptAuxiliary,
            (0, "NearMiss") => OracleKind::NearMiss,
            (_, "AcceptProbabilistic") => {
                return Err(format!("event {i}: a probabilistic accept; run the oracle on deterministic_set()"));
            }
            _ => return Err(format!("event {i}: RESULT type {ty} contradicts detection_type={dtype}")),
        };
        events.push(OracleEvent { keyword, kind, score, threshold, begin, end, afe });
    }
    Ok(events)
}

/// The samples pryon_test restarted its decoder at (`restart: new decoder fed from sample N`).
pub fn restarts(output: &str) -> Vec<u64> {
    let at = |l: &str| l.strip_prefix("restart: new decoder fed from sample ")?.trim().parse().ok();
    output.lines().filter_map(at).collect()
}

/// Value of `key` (e.g. `type=`) up to the next space or comma.
fn field<'a>(line: &'a str, key: &str) -> Option<&'a str> {
    let at = line.find(key)? + key.len();
    let rest = &line[at..];
    Some(&rest[..rest.find([' ', ',']).unwrap_or(rest.len())])
}

fn meta_afe(line: &str) -> Option<(i64, i64)> {
    let num = |key: &str| {
        let rest = &line[line.find(key)? + key.len()..];
        let end = rest.find(|c: char| !(c.is_ascii_digit() || c == '-')).unwrap_or(rest.len());
        rest[..end].parse::<i64>().ok()
    };
    if num("\"has_audio_lsb_metadata\":")? != 1 {
        return None;
    }
    Some((num("\"timestamp_before_ww_start\":")?, num("\"timestamp_before_ww_end\":")?))
}

const PROBABILITY: &str = "\"probabilistic-accept-probability\"";

/// `text` with every kw.cfg `probabilistic-accept-probability` value set to 0, and whether any was nonzero.
pub fn zero_probability(text: &str) -> (String, bool) {
    let (mut out, mut rest, mut changed) = (String::new(), text, false);
    while let Some(i) = rest.find(PROBABILITY) {
        let after = &rest[i + PROBABILITY.len()..];
        let colon = after.len() - after.trim_start().len();
        let Some(v) = after[colon..].strip_prefix(':') else {
            out.push_str(&rest[..i + PROBABILITY.len()]);
            rest = after;
            continue;
        };
        let lead = v.len() - v.trim_start().len();
        let num = v[lead..].find(|c: char| !(c.is_ascii_digit() || "+-.eE".contains(c))).unwrap_or(v.len() - lead);
        changed |= v[lead..lead + num].parse::<f64>() != Ok(0.0);
        out.push_str(&rest[..i + PROBABILITY.len() + colon + 1 + lead]);
        out.push('0');
        rest = &v[lead + num..];
    }
    out.push_str(rest);
    (out, changed)
}

/// `set`, or a symlink copy under `work` with probabilistic-accept-probability 0: libpryon's draw is clock-seeded.
pub fn deterministic_set(set: &Path, work: &Path) -> std::io::Result<PathBuf> {
    fn walk(src: &Path, dst: &Path, patch: &mut Vec<(PathBuf, String)>, links: &mut Vec<(PathBuf, PathBuf)>) -> std::io::Result<()> {
        for e in std::fs::read_dir(src)? {
            let e = e?;
            let (s, d) = (e.path(), dst.join(e.file_name()));
            if e.file_type()?.is_dir() || (e.file_type()?.is_symlink() && s.is_dir()) {
                walk(&s, &d, patch, links)?;
            } else if e.file_name() == "kw.cfg.json" {
                let (text, changed) = zero_probability(&std::fs::read_to_string(&s)?);
                if changed {
                    patch.push((d, text));
                } else {
                    links.push((s, d));
                }
            } else {
                links.push((s, d));
            }
        }
        Ok(())
    }
    let set = std::fs::canonicalize(set)?;
    let (mut patch, mut links) = (Vec::new(), Vec::new());
    walk(&set, work, &mut patch, &mut links)?;
    if patch.is_empty() {
        return Ok(set);
    }
    let _ = std::fs::remove_dir_all(work);
    for (d, text) in &patch {
        std::fs::create_dir_all(d.parent().unwrap())?;
        std::fs::write(d, text)?;
    }
    for (s, d) in &links {
        std::fs::create_dir_all(d.parent().unwrap())?;
        std::os::unix::fs::symlink(s, d)?;
    }
    Ok(work.to_path_buf())
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OracleCmd {
    Property { at: u64, name: String, value: i32 },
    SessionEnd { at: u64 },
}

pub struct OracleEnv {
    pub rootfs: PathBuf,
    pub pryon_test: PathBuf,
}

impl OracleEnv {
    /// `PRYON_ORACLE_ROOTFS` and `PRYON_PRYON_TEST`; `None` when either is unset.
    pub fn from_env() -> Option<OracleEnv> {
        Some(OracleEnv {
            rootfs: std::env::var_os("PRYON_ORACLE_ROOTFS")?.into(),
            pryon_test: std::env::var_os("PRYON_PRYON_TEST")?.into(),
        })
    }

    /// Runs `pryon_test` on a 16 kHz s16le `.raw`; `manifest` is a path inside the rootfs. Returns stdout then stderr.
    pub fn run_wake(&self, manifest: &Path, clip: &Path, cmds: &[OracleCmd]) -> std::io::Result<String> {
        self.run_wake_at(manifest, clip, cmds, 1000.0, None)
    }

    /// As `run_wake` at feed speed `speed` (`-x`); a run still going after `timeout` is killed and is an error.
    pub fn run_wake_at(
        &self,
        manifest: &Path,
        clip: &Path,
        cmds: &[OracleCmd],
        speed: f64,
        timeout: Option<std::time::Duration>,
    ) -> std::io::Result<String> {
        use std::io::Read;
        use std::process::{Command, Stdio};
        use std::time::Instant;
        let mut cmd = Command::new("unshare");
        // The PID namespace is for bionic's 16-bit mutex owner ids; -x 1000 keeps the backlog under libpryon's drop limit.
        cmd.args(["--user", "--map-root-user", "--pid", "--fork", "--kill-child", "qemu-arm", "-L"])
            .arg(&self.rootfs)
            .arg(&self.pryon_test)
            .arg("-m")
            .arg(manifest)
            .arg("-x")
            .arg(speed.to_string());
        for c in cmds {
            match c {
                OracleCmd::Property { at, name, value } => cmd.arg("-P").arg(format!("{at}:{name}={value}")),
                OracleCmd::SessionEnd { at } => cmd.arg("-E").arg(at.to_string()),
            };
        }
        let mut child = cmd.arg(clip).stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn()?;
        let drain = |mut r: Box<dyn Read + Send>| {
            std::thread::spawn(move || {
                let mut b = Vec::new();
                let _ = r.read_to_end(&mut b);
                b
            })
        };
        let (so, se) = (drain(Box::new(child.stdout.take().unwrap())), drain(Box::new(child.stderr.take().unwrap())));
        let t0 = Instant::now();
        let status = loop {
            match child.try_wait() {
                Ok(Some(s)) => break s,
                Ok(None) => {}
                Err(e) => {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Err(e);
                }
            }
            if timeout.is_some_and(|t| t0.elapsed() > t) {
                let _ = child.kill();
                let _ = child.wait();
                return Err(std::io::Error::new(std::io::ErrorKind::TimedOut, format!("pryon_test timed out after {:?}", timeout.unwrap())));
            }
            std::thread::sleep(std::time::Duration::from_millis(20));
        };
        let (stdout, stderr) = (so.join().unwrap_or_default(), se.join().unwrap_or_default());
        // 3 is "no callbacks".
        if !matches!(status.code(), Some(0 | 3)) {
            let err = String::from_utf8_lossy(&stderr);
            let tail: Vec<&str> = err.lines().rev().take(15).collect();
            let tail: Vec<&str> = tail.into_iter().rev().collect();
            return Err(std::io::Error::other(format!("pryon_test exited {:?}: {}", status.code(), tail.join("\n"))));
        }
        Ok(format!("{}\n{}", String::from_utf8_lossy(&stdout), String::from_utf8_lossy(&stderr)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{synth, Clip, TempDir};

    const RESULT_A: &str = "RESULT decoder=dec keyword=ALEXA type=2 begin=9600 end=27840 (0.60s..1.74s) meta=1672";
    const KWS_A: &str = "pryon[4] INFO: DecoderID=dec: recognition_id=dec-0: {KWS} segmentation_score=-inf,classifier_score=0.999451,active_threshold=0.923,detection_type=Accept";
    const RESULT_B: &str = "RESULT decoder=dec keyword=ALEXA type=2 begin=22240 end=32640 (1.39s..2.04s) meta=1576";
    const KWS_B: &str = "pryon[4] INFO: DecoderID=dec: recognition_id=dec-1: {KWS} segmentation_score=-inf,classifier_score=0.88,active_threshold=0.69,detection_type=AcceptAuxiliary";
    const RESULT_NM: &str = "RESULT decoder=dec keyword=ALEXA type=0 begin=61760 end=89920 (3.86s..5.62s) meta=1512";
    const KWS_NM: &str = "pryon[4] INFO: DecoderID=dec: recognition_id=dec-1: {KWS} segmentation_score=-inf,classifier_score=0.913608,active_threshold=0.69,detection_type=NearMiss";
    const ERR: &str = "pryon[2] ERROR: DecoderID=dec: recognition_id=dec-0: {KWS} Unexpected detection event type NearMiss is emitted from DETECTION_AUXILIARY spotter";

    #[test]
    fn parse_accept_and_auxiliary() {
        let out = format!("format: x\n{RESULT_A}\n{KWS_A}\n{RESULT_B}\n{KWS_B}\npushed 1 samples\n");
        let ev = parse_wake(&out).unwrap();
        assert_eq!(ev.len(), 2);
        assert_eq!(ev[0].kind, OracleKind::Accept);
        assert_eq!((ev[0].keyword.as_str(), ev[0].score.as_str(), ev[0].threshold.as_str()), ("ALEXA", "0.999451", "0.923"));
        assert_eq!((ev[0].begin, ev[0].end, ev[0].afe), (9600, 27840, None));
        assert_eq!(ev[1].kind, OracleKind::AcceptAuxiliary);
        assert_eq!((ev[1].score.as_str(), ev[1].threshold.as_str()), ("0.88", "0.69"));
    }

    #[test]
    fn parse_pairs_by_order_ignoring_errors() {
        // The KWS log of a near miss may carry recognition_id=dec-1 on the first RESULT.
        let out = format!("{RESULT_NM}\n{ERR}\n{KWS_NM}\n");
        let ev = parse_wake(&out).unwrap();
        assert_eq!(ev.len(), 1);
        assert_eq!(ev[0].kind, OracleKind::NearMiss);
        assert_eq!(ev[0].score, "0.913608");
    }

    #[test]
    fn parse_meta_afe() {
        let meta = |lsb: u8| {
            format!(
                "META header 24 bytes, {{\"deviceMetrics\":{{\"has_audio_lsb_metadata\":{lsb},\"timestamp_before_ww_end\":9559,\"timestamp_before_ww_start\":8823}}}}"
            )
        };
        let with = parse_wake(&format!("{RESULT_A}\n{}\n{KWS_A}\n", meta(1))).unwrap();
        assert_eq!(with[0].afe, Some((8823, 9559)));
        let without = parse_wake(&format!("{RESULT_A}\n{}\n{KWS_A}\n", meta(0))).unwrap();
        assert_eq!(without[0].afe, None);
    }

    #[test]
    fn parse_unpaired_errors() {
        assert!(parse_wake(&format!("{RESULT_A}\n")).is_err());
        assert!(parse_wake(&format!("{KWS_A}\n")).is_err());
        assert_eq!(parse_wake("pushed 5 samples, 0 callbacks\n").unwrap(), vec![]);
    }

    #[test]
    fn probabilistic_accept_rejected() {
        let kws = KWS_A.replace("detection_type=Accept", "detection_type=AcceptProbabilistic");
        let e = parse_wake(&format!("{RESULT_A}\n{kws}\n")).unwrap_err();
        assert!(e.contains("deterministic_set"), "{e}");
    }

    #[test]
    fn unparsable_score_rejected() {
        for bad in ["nan", "0.9x", ""] {
            let kws = KWS_A.replace("classifier_score=0.999451", &format!("classifier_score={bad}"));
            let e = parse_wake(&format!("{RESULT_A}\n{kws}\n")).unwrap_err();
            assert!(e.contains("not a number"), "{bad}: {e}");
        }
    }

    #[test]
    fn probability_zeroed() {
        let t = r#"{ "a": 1, "probabilistic-accept-probability": 0.001, "b": { "probabilistic-accept-probability" : 1e-3 } }"#;
        let (z, changed) = zero_probability(t);
        assert!(changed);
        assert_eq!(z, r#"{ "a": 1, "probabilistic-accept-probability": 0, "b": { "probabilistic-accept-probability" : 0 } }"#);
        assert_eq!(zero_probability(&z), (z.clone(), false));
        assert_eq!(zero_probability("{}"), ("{}".to_string(), false));
    }

    #[test]
    fn deterministic_set_links_all_but_kw_cfg() {
        let src = TempDir::new("detset-src");
        std::fs::create_dir_all(src.join("BDP")).unwrap();
        std::fs::write(src.join("pryon.manifest"), "m").unwrap();
        std::fs::write(src.join("BDP/kw.cfg.json"), r#"{"probabilistic-accept-probability": 0.001}"#).unwrap();
        let work = TempDir::new("detset-work");
        let dst = work.join("set");
        let got = deterministic_set(&src, &dst).unwrap();
        assert_eq!(got, dst);
        assert!(std::fs::symlink_metadata(dst.join("pryon.manifest")).unwrap().file_type().is_symlink());
        let kw = std::fs::read_to_string(dst.join("BDP/kw.cfg.json")).unwrap();
        assert_eq!(kw, r#"{"probabilistic-accept-probability": 0}"#);
        // nothing to zero: the set itself
        std::fs::write(src.join("BDP/kw.cfg.json"), "{}").unwrap();
        assert_eq!(deterministic_set(&src, &work.join("other")).unwrap(), std::fs::canonicalize(&*src).unwrap());
    }

    #[test]
    fn oracle_runs_firmware() {
        let Some(env) = OracleEnv::from_env() else {
            eprintln!("skipped: set PRYON_ORACLE_ROOTFS and PRYON_PRYON_TEST");
            return;
        };
        let dir = TempDir::new("oracle");
        let clip = dir.join("noise.raw");
        let pcm: Vec<u8> = synth(Clip::Noise).iter().flat_map(|s| s.to_le_bytes()).collect();
        std::fs::write(&clip, pcm).unwrap();
        let manifest = Path::new("/system/local/models/keyword/en-US/ALEXA/pryon.manifest");
        let cmds = [OracleCmd::Property { at: 4000, name: "AlarmState".into(), value: 1 }, OracleCmd::SessionEnd { at: 8000 }];
        let out = env.run_wake(manifest, &clip, &cmds).unwrap();
        assert!(out.contains("command @4000 property AlarmState -> 0"), "{out}");
        assert!(out.contains("command @8000 SessionEnd -> 0"), "{out}");
        parse_wake(&out).unwrap();
    }
}
