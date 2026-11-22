//! `pryon.config` (`key = value`) and `AED.json` (minimal JSON) -> `SoundConfig`.
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
    let root = parse_json(text)?;
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

// Numbers keep their source text so f32/int parsing happens once, at the use site.
enum Json {
    Null,
    Bool,
    Num(String),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

impl Json {
    fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Obj(o) => o.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }
}

fn parse_json(text: &str) -> Result<Json, Error> {
    let mut p = Parser { s: text.as_bytes(), i: 0, depth: 0 };
    let v = p.value()?;
    p.ws();
    if p.i != p.s.len() {
        return Err(err("trailing data after JSON value"));
    }
    Ok(v)
}

struct Parser<'a> {
    s: &'a [u8],
    i: usize,
    depth: usize,
}

impl Parser<'_> {
    fn ws(&mut self) {
        while self.s.get(self.i).is_some_and(|c| c.is_ascii_whitespace()) {
            self.i += 1;
        }
    }

    fn eat(&mut self, c: u8) -> Result<(), Error> {
        self.ws();
        if self.s.get(self.i) == Some(&c) {
            self.i += 1;
            Ok(())
        } else {
            Err(err(format!("JSON: expected '{}' at byte {}", c as char, self.i)))
        }
    }

    fn lit(&mut self, word: &str, v: Json) -> Result<Json, Error> {
        if self.s[self.i..].starts_with(word.as_bytes()) {
            self.i += word.len();
            Ok(v)
        } else {
            Err(err(format!("JSON: bad literal at byte {}", self.i)))
        }
    }

    fn value(&mut self) -> Result<Json, Error> {
        self.ws();
        match self.s.get(self.i) {
            None => Err(err("JSON: unexpected end")),
            Some(b'{') => self.nested(Self::object),
            Some(b'[') => self.nested(Self::array),
            Some(b'"') => Ok(Json::Str(self.string()?)),
            Some(b't') => self.lit("true", Json::Bool),
            Some(b'f') => self.lit("false", Json::Bool),
            Some(b'n') => self.lit("null", Json::Null),
            Some(_) => self.number(),
        }
    }

    fn nested(&mut self, f: fn(&mut Self) -> Result<Json, Error>) -> Result<Json, Error> {
        self.depth += 1;
        if self.depth > 64 {
            return Err(err("JSON: nesting too deep"));
        }
        let v = f(self);
        self.depth -= 1;
        v
    }

    fn object(&mut self) -> Result<Json, Error> {
        self.eat(b'{')?;
        let mut o = Vec::new();
        self.ws();
        if self.s.get(self.i) == Some(&b'}') {
            self.i += 1;
            return Ok(Json::Obj(o));
        }
        loop {
            self.ws();
            let k = self.string()?;
            self.eat(b':')?;
            o.push((k, self.value()?));
            self.ws();
            match self.s.get(self.i) {
                Some(b',') => self.i += 1,
                Some(b'}') => {
                    self.i += 1;
                    return Ok(Json::Obj(o));
                }
                _ => return Err(err(format!("JSON: expected ',' or '}}' at byte {}", self.i))),
            }
        }
    }

    fn array(&mut self) -> Result<Json, Error> {
        self.eat(b'[')?;
        let mut a = Vec::new();
        self.ws();
        if self.s.get(self.i) == Some(&b']') {
            self.i += 1;
            return Ok(Json::Arr(a));
        }
        loop {
            a.push(self.value()?);
            self.ws();
            match self.s.get(self.i) {
                Some(b',') => self.i += 1,
                Some(b']') => {
                    self.i += 1;
                    return Ok(Json::Arr(a));
                }
                _ => return Err(err(format!("JSON: expected ',' or ']' at byte {}", self.i))),
            }
        }
    }

    fn string(&mut self) -> Result<String, Error> {
        if self.s.get(self.i) != Some(&b'"') {
            return Err(err(format!("JSON: expected string at byte {}", self.i)));
        }
        self.i += 1;
        let mut out = Vec::new();
        loop {
            let c = *self.s.get(self.i).ok_or_else(|| err("JSON: unterminated string"))?;
            self.i += 1;
            match c {
                b'"' => return String::from_utf8(out).map_err(|_| err("JSON: invalid UTF-8")),
                b'\\' => {
                    let e = *self.s.get(self.i).ok_or_else(|| err("JSON: unterminated escape"))?;
                    self.i += 1;
                    match e {
                        b'"' | b'\\' | b'/' => out.push(e),
                        b'n' => out.push(b'\n'),
                        b't' => out.push(b'\t'),
                        b'r' => out.push(b'\r'),
                        b'b' => out.push(8),
                        b'f' => out.push(12),
                        b'u' => {
                            let hex = self.s.get(self.i..self.i + 4).ok_or_else(|| err("JSON: short \\u escape"))?;
                            let n = u32::from_str_radix(std::str::from_utf8(hex).unwrap_or("x"), 16)
                                .map_err(|_| err("JSON: bad \\u escape"))?;
                            self.i += 4;
                            let ch = char::from_u32(n).unwrap_or('\u{fffd}');
                            out.extend(ch.to_string().as_bytes());
                        }
                        _ => return Err(err("JSON: bad escape")),
                    }
                }
                _ => out.push(c),
            }
        }
    }

    fn number(&mut self) -> Result<Json, Error> {
        let start = self.i;
        while self.s.get(self.i).is_some_and(|c| matches!(c, b'0'..=b'9' | b'-' | b'+' | b'.' | b'e' | b'E')) {
            self.i += 1;
        }
        if start == self.i {
            return Err(err(format!("JSON: unexpected byte at {start}")));
        }
        let raw = std::str::from_utf8(&self.s[start..self.i]).unwrap();
        raw.parse::<f64>().map_err(|_| err(format!("JSON: bad number '{raw}'")))?;
        Ok(Json::Num(raw.to_string()))
    }
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

    #[test]
    fn json_escapes_and_unknown_fields() {
        let v = parse_json(r#"{"a": [1, -2.5e1, true, false, null, "q\"\\x"], "b": {"c": {}}}"#).unwrap();
        let Json::Obj(o) = v else { panic!() };
        let Json::Arr(a) = &o[0].1 else { panic!() };
        assert!(matches!(&a[5], Json::Str(s) if s == "q\"\\x"));
        assert!(parse_json("{} x").is_err());
        assert!(parse_json("[1,]").is_err());
    }
}
