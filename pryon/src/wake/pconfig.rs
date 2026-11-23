//! pryon.manifest and pryon.config with INCLUDE chains.

use crate::json::Doc;
use crate::Error;
use std::path::{Path, PathBuf};

const MAX_INCLUDE_DEPTH: usize = 16;

/// `path` relative to the set directory when it lies inside it (for messages).
pub(crate) fn rel(dir: &Path, path: &Path) -> String {
    path.strip_prefix(dir).unwrap_or(path).display().to_string()
}

/// `base/rel` with `.` and `name/..` folded lexically, so referenced files keep readable paths.
pub(crate) fn join(base: &Path, rel: &str) -> PathBuf {
    use std::path::Component;
    let mut out = PathBuf::new();
    for c in base.join(rel).components() {
        match c {
            Component::CurDir => {}
            Component::ParentDir if matches!(out.components().next_back(), Some(Component::Normal(_))) => {
                out.pop();
            }
            c => out.push(c),
        }
    }
    out
}

/// The directory of `file`, against which the paths it names resolve.
pub(crate) fn base(file: &Path) -> &Path {
    file.parent().unwrap_or(Path::new(""))
}

pub(crate) fn read_text(path: &Path) -> Result<String, Error> {
    std::fs::read_to_string(path).map_err(|source| Error::Io { path: path.to_path_buf(), source })
}

pub(crate) fn unsupported(file: &Path, what: impl Into<String>) -> Error {
    Error::Unsupported { file: file.to_path_buf(), what: what.into() }
}

/// `path` (folded by `join`) must lie under `dir` with no symlink below `dir`; `file` and `key` name the reference.
pub(crate) fn inside(dir: &Path, path: PathBuf, file: &Path, key: &str) -> Result<PathBuf, Error> {
    use std::path::Component;
    let bad = |why: &str| unsupported(file, format!("{key} = {}: {why}", rel(dir, &path)));
    let r = path.strip_prefix(dir).map_err(|_| bad("outside the set directory"))?;
    if r.as_os_str().is_empty() || r.components().any(|c| !matches!(c, Component::Normal(_))) {
        return Err(bad("outside the set directory"));
    }
    let mut p = dir.to_path_buf();
    for c in r.components() {
        p.push(c);
        match std::fs::symlink_metadata(&p) {
            Ok(m) if m.file_type().is_symlink() => return Err(bad("reached through a symlink (not followed)")),
            Ok(_) => {}
            // a missing file is reported by the read that follows
            Err(_) => break,
        }
    }
    Ok(path)
}

pub(crate) struct Manifest {
    pub model: String,
    pub config: PathBuf,
    pub checksum: Option<PathBuf>,
}

pub(crate) fn read_manifest(dir: &Path) -> Result<Manifest, Error> {
    let d = Doc::load(&inside(dir, dir.join("pryon.manifest"), dir, "pryon.manifest")?)?;
    let top = d.top(&["checksum", "default_model_name", "models", "object_type_name", "object_type_version"])?;
    d.pin_str(&top, "object_type_name", &["ModelSetManifest"], false)?;
    d.pin_num(&top, "object_type_version", 1.0, false)?;
    let model = d.str(&top, "default_model_name")?;
    // other entries of `models` name models that are never loaded
    let models = d.get(&top, "models")?.obj("models").ok_or_else(|| d.bad("models is not an object"))?;
    let entry = d.sub(&models, model, &["config_file_path"])?;
    let config = join(dir, d.str(&entry, "config_file_path")?);
    let checksum = d.opt_str(&top, "checksum")?.map(|c| inside(dir, join(dir, c), &d.path, "checksum")).transpose()?;
    Ok(Manifest {
        model: model.to_string(),
        config: inside(dir, config, &d.path, &entry.at("config_file_path"))?,
        checksum,
    })
}

/// One `key = value` with the file that set it; relative paths resolve against that file's directory.
#[derive(Debug, Clone)]
pub(crate) struct Setting {
    pub value: String,
    pub file: PathBuf,
}

impl Setting {
    pub fn path(&self) -> PathBuf {
        join(base(&self.file), &self.value)
    }
}

/// Merged `pryon.config`: INCLUDEs expanded in place, depth-first; a later key replaces an earlier one.
#[derive(Debug)]
pub(crate) struct PConfig {
    pub dir: PathBuf,
    pub keys: Vec<(String, Setting)>,
}

impl PConfig {
    pub fn get(&self, key: &str) -> Option<&Setting> {
        self.keys.iter().find(|(k, _)| k == key).map(|(_, s)| s)
    }

    pub fn read(dir: &Path, top: &Path) -> Result<PConfig, Error> {
        let mut c = PConfig { dir: dir.to_path_buf(), keys: Vec::new() };
        c.include(top, &mut Vec::new())?;
        Ok(c)
    }

    fn chain(&self, stack: &[(PathBuf, PathBuf)]) -> String {
        stack.iter().map(|(p, _)| rel(&self.dir, p)).collect::<Vec<_>>().join(" -> ")
    }

    /// `stack` holds (path as written, canonical path) of the files being read.
    fn include(&mut self, path: &Path, stack: &mut Vec<(PathBuf, PathBuf)>) -> Result<(), Error> {
        let canon = match std::fs::canonicalize(path) {
            Ok(c) => c,
            Err(source) if stack.is_empty() => return Err(Error::Io { path: path.to_path_buf(), source }),
            Err(e) => {
                let msg = format!("{}: {e} (INCLUDE chain: {})", rel(&self.dir, path), self.chain(stack));
                return Err(Error::Config(msg));
            }
        };
        if let Some(i) = stack.iter().position(|(_, c)| *c == canon) {
            let mut cyc = stack[i..].to_vec();
            cyc.push((path.to_path_buf(), canon));
            return Err(Error::Config(format!("INCLUDE cycle: {}", self.chain(&cyc))));
        }
        if stack.len() >= MAX_INCLUDE_DEPTH {
            return Err(unsupported(path, format!("INCLUDE nesting deeper than {MAX_INCLUDE_DEPTH}")));
        }
        let text = read_text(path)?;
        stack.push((path.to_path_buf(), canon));
        for (no, raw) in text.lines().enumerate() {
            let line = strip_comment(raw).trim();
            if line.is_empty() {
                continue;
            }
            if let Some(rest) = line.strip_prefix("INCLUDE").filter(|r| r.starts_with(char::is_whitespace)) {
                let target = inside(&self.dir, join(base(path), unquote(rest.trim())), path, "INCLUDE")?;
                self.include(&target, stack)?;
                continue;
            }
            let Some((k, v)) = line.split_once('=') else {
                return Err(Error::Config(format!("{}: line {} is not `key = value`: {raw}", path.display(), no + 1)));
            };
            let key = k.trim();
            if key.is_empty() {
                return Err(Error::Config(format!("{}: line {} has no key", path.display(), no + 1)));
            }
            let s = Setting { value: unquote(v.trim()).to_string(), file: path.to_path_buf() };
            match self.keys.iter_mut().find(|(k, _)| k == key) {
                Some(e) => e.1 = s,
                None => self.keys.push((key.to_string(), s)),
            }
        }
        stack.pop();
        Ok(())
    }
}

/// `#` starts a comment at the line start or after whitespace; keys like `enable#disable-upsampling` keep theirs.
fn strip_comment(line: &str) -> &str {
    let b = line.as_bytes();
    match (0..b.len()).find(|&i| b[i] == b'#' && (i == 0 || b[i - 1].is_ascii_whitespace())) {
        Some(i) => &line[..i],
        None => line,
    }
}

fn unquote(s: &str) -> &str {
    if s.len() >= 2 && s.starts_with('"') && s.ends_with('"') {
        &s[1..s.len() - 1]
    } else {
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn comments_need_whitespace() {
        assert_eq!(strip_comment("a#b = 1 # note"), "a#b = 1 ");
        assert_eq!(strip_comment("# all"), "");
        assert_eq!(strip_comment("x = \"#\""), "x = \"#\"");
        assert_eq!(unquote("\"\""), "");
        assert_eq!(unquote("\"a"), "\"a");
    }

    #[test]
    fn later_keys_win_and_paths_follow_their_file() {
        let d = pryon_testkit::TempDir::new("pconfig-order");
        std::fs::create_dir_all(d.join("sub")).unwrap();
        std::fs::write(d.join("top.config"), "a = 1\nINCLUDE \"sub/inc.config\"\nb = \"x y\"\n").unwrap();
        std::fs::write(d.join("sub/inc.config"), "a = 2\np = \"f.json\"\nb = 0\n").unwrap();
        let c = PConfig::read(&d, &d.join("top.config")).unwrap();
        assert_eq!(c.get("a").unwrap().value, "2");
        assert_eq!(c.get("b").unwrap().value, "x y");
        assert_eq!(c.get("p").unwrap().path(), d.join("sub/f.json"));
        let keys: Vec<&str> = c.keys.iter().map(|(k, _)| k.as_str()).collect();
        // a replaced key keeps the position it was first set at
        assert_eq!(keys, ["a", "p", "b"]);
    }

    #[test]
    fn join_folds_parent_dirs() {
        assert_eq!(join(Path::new("/s/BDP"), "../dec/x.json"), Path::new("/s/dec/x.json"));
        assert_eq!(join(Path::new("s"), "./a/../../b"), Path::new("b"));
        assert_eq!(join(Path::new(""), "../x"), Path::new("../x"));
    }

    #[test]
    fn malformed_line_named() {
        let d = pryon_testkit::TempDir::new("pconfig-bad");
        std::fs::write(d.join("pryon.config"), "a = 1\njunk\n").unwrap();
        let e = PConfig::read(&d, &d.join("pryon.config")).unwrap_err().to_string();
        assert!(e.contains("line 2"), "{e}");
    }
}
