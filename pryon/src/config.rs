//! `pryon.config` (`key = value`) and `AED.json` (minimal JSON) -> `SoundConfig`.
use crate::json::{self, Json};
use crate::Error;
use std::path::{Path, PathBuf};

pub struct TypeConfig {
    pub name: String,
    pub output_index: usize,
    pub threshold: f32,
    pub near_miss: f32,
    pub energy_threshold: i32,
    pub energy_min_samples: u32,
}

pub struct SoundConfig {
    pub mlp_path: PathBuf,
    pub window_ms: u32,
    pub types: Vec<TypeConfig>,
}

const WINDOW_MS: u32 = 9980;

fn err(m: impl Into<String>) -> Error {
    Error::Config(m.into())
}

fn read(path: &Path) -> Result<String, Error> {
    std::fs::read_to_string(path).map_err(|source| Error::Io { path: path.to_path_buf(), source })
}

impl SoundConfig {
    pub fn load(dir: &Path) -> Result<SoundConfig, Error> {
        let cfg_path = dir.join("pryon.config");
        let (mut model, mut window, mut aed) = (None, None, None);
        for line in read(&cfg_path)?.lines() {
            let line = line.trim();
            let Some((key, val)) = line.split_once('=') else { continue };
            let (key, val) = (key.trim(), val.trim());
            if line.starts_with('#') || key.contains('#') {
                continue;
            }
            let val = val.trim_matches('"').to_string();
            match key {
                "scorer.model_filepath" => model = Some(val),
                "scorer.batch_scorer_reset_interval_msec" => window = Some(val),
                "recognizer.acoustic_event_detection.config_filepath" => aed = Some(val),
                _ => {}
            }
        }
        let model = model.ok_or_else(|| err("scorer.model_filepath missing"))?;
        let aed = aed.ok_or_else(|| err("recognizer.acoustic_event_detection.config_filepath missing"))?;
        let window_ms: u32 = window
            .ok_or_else(|| err("scorer.batch_scorer_reset_interval_msec missing"))?
            .parse()
            .map_err(|_| err("scorer.batch_scorer_reset_interval_msec is not an integer"))?;
        if window_ms != WINDOW_MS {
            return Err(err(format!("window {window_ms} ms unsupported, only {WINDOW_MS}")));
        }
        let mlp_path = dir.join(model);
        std::fs::metadata(&mlp_path).map_err(|source| Error::Io { path: mlp_path.clone(), source })?;
        let types = parse_types(&read(&dir.join(aed))?)?;
        Ok(SoundConfig { mlp_path, window_ms, types })
    }
}

fn parse_types(text: &str) -> Result<Vec<TypeConfig>, Error> {
    let root = json::parse(text).map_err(err)?;
    let events = root
        .get("configuration")
        .and_then(|c| c.get("acoustic_event"))
        .and_then(|e| match e {
            Json::Arr(a) => Some(a),
            _ => None,
        })
        .ok_or_else(|| err("configuration.acoustic_event[] missing"))?;
    if events.is_empty() {
        return Err(err("acoustic_event is empty"));
    }
    events.iter().map(parse_type).collect()
}

fn parse_type(e: &Json) -> Result<TypeConfig, Error> {
    let field = |k: &str| e.get(k).ok_or_else(|| err(format!("acoustic_event field {k} missing")));
    let num = |k: &str| match field(k)? {
        Json::Num(raw) => Ok(raw.as_str()),
        _ => Err(err(format!("{k} is not a number"))),
    };
    let bad = |k: &str| err(format!("{k} is out of range"));
    let name = match field("name")? {
        Json::Str(s) => s.clone(),
        _ => return Err(err("name is not a string")),
    };
    Ok(TypeConfig {
        name,
        output_index: num("score_output_index")?.parse().map_err(|_| bad("score_output_index"))?,
        threshold: num("score_detection_threshold")?.parse().map_err(|_| bad("score_detection_threshold"))?,
        near_miss: num("score_near_miss_threshold")?.parse().map_err(|_| bad("score_near_miss_threshold"))?,
        energy_threshold: num("energy_threshold")?.parse().map_err(|_| bad("energy_threshold"))?,
        energy_min_samples: num("energy_min_num_samples")?.parse().map_err(|_| bad("energy_min_num_samples"))?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(tag: &str) -> pryon_testkit::TempDir {
        pryon_testkit::TempDir::new(&format!("config-{tag}"))
    }

    #[test]
    fn loads_written_dir() {
        let d = tmp("ok");
        pryon_testkit::write_model_dir(&d, 1, 0.8);
        let c = SoundConfig::load(&d).unwrap();
        assert_eq!(c.types.len(), 12);
        assert_eq!(c.window_ms, 9980);
        assert_eq!(c.types[0].name, "humanPresence");
        assert_eq!(c.types[0].energy_threshold, 108);
        assert_eq!(c.types[1].energy_threshold, 82);
        assert_eq!(c.types[11].name, "runningWater");
        assert_eq!(c.types[11].output_index, 11);
        assert_eq!(c.types[1].threshold, 0.8);
        assert_eq!(c.types[1].near_miss, 0.4);
        assert_eq!(c.types[1].energy_min_samples, 1);
        assert_eq!(c.mlp_path, d.join("model_gcmvn.cpu.v1.q.mlp"));
    }

    #[test]
    fn missing_files_error() {
        for f in ["pryon.config", "AED.json", "model_gcmvn.cpu.v1.q.mlp"] {
            let d = tmp("missing");
            pryon_testkit::write_model_dir(&d, 1, 0.8);
            std::fs::remove_file(d.join(f)).unwrap();
            assert!(SoundConfig::load(&d).is_err(), "{f}");
        }
    }

    #[test]
    fn bad_json_errors() {
        let d = tmp("json");
        pryon_testkit::write_model_dir(&d, 1, 0.8);
        let j = std::fs::read(d.join("AED.json")).unwrap();
        std::fs::write(d.join("AED.json"), &j[..j.len() / 2]).unwrap();
        assert!(matches!(SoundConfig::load(&d), Err(Error::Config(_))));
    }

    #[test]
    fn variant_keys_ignored() {
        let d = tmp("variant");
        pryon_testkit::write_model_dir(&d, 1, 0.8);
        let mut c = std::fs::read_to_string(d.join("pryon.config")).unwrap();
        c.push_str("scorer.model_filepath#x = \"y\"\nscorer.batch_scorer_reset_interval_msec#v = 1\n");
        std::fs::write(d.join("pryon.config"), c).unwrap();
        let c = SoundConfig::load(&d).unwrap();
        assert_eq!(c.mlp_path, d.join("model_gcmvn.cpu.v1.q.mlp"));
        assert_eq!(c.window_ms, 9980);
    }

    #[test]
    fn other_window_rejected() {
        let d = tmp("window");
        pryon_testkit::write_model_dir(&d, 1, 0.8);
        let c = std::fs::read_to_string(d.join("pryon.config")).unwrap();
        std::fs::write(d.join("pryon.config"), c.replace("9980", "5000")).unwrap();
        assert!(matches!(SoundConfig::load(&d), Err(Error::Config(_))));
    }
}
