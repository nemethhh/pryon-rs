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
        let kind = match (ty, dtype.as_str()) {
            (2, "Accept") => OracleKind::Accept,
            (2, "AcceptAuxiliary") => OracleKind::AcceptAuxiliary,
            (0, "NearMiss") => OracleKind::NearMiss,
            _ => return Err(format!("event {i}: RESULT type {ty} contradicts detection_type={dtype}")),
        };
        events.push(OracleEvent { keyword, kind, score, threshold, begin, end, afe });
    }
    Ok(events)
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
        use std::process::Command;
        let mut cmd = Command::new("unshare");
        // The PID namespace is for bionic's 16-bit mutex owner ids; -x 1000 keeps the backlog under libpryon's drop limit.
        cmd.args(["--user", "--map-root-user", "--pid", "--fork", "--kill-child", "qemu-arm", "-L"])
            .arg(&self.rootfs)
            .arg(&self.pryon_test)
            .arg("-m")
            .arg(manifest)
            .args(["-x", "1000"]);
        for c in cmds {
            match c {
                OracleCmd::Property { at, name, value } => cmd.arg("-P").arg(format!("{at}:{name}={value}")),
                OracleCmd::SessionEnd { at } => cmd.arg("-E").arg(at.to_string()),
            };
        }
        let out = cmd.arg(clip).output()?;
        // 3 is "no callbacks".
        if !matches!(out.status.code(), Some(0 | 3)) {
            let err = String::from_utf8_lossy(&out.stderr);
            let tail: Vec<&str> = err.lines().rev().take(15).collect();
            let tail: Vec<&str> = tail.into_iter().rev().collect();
            return Err(std::io::Error::other(format!("pryon_test exited {:?}: {}", out.status.code(), tail.join("\n"))));
        }
        Ok(format!("{}\n{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr)))
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
