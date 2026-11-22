//! Minimal JSON reader for model configuration files, with path-naming typed access.

use crate::Error;
use std::path::{Path, PathBuf};

/// Numbers keep their source text so f32/int parsing happens once, at the use site.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Json {
    Null,
    Bool(bool),
    Num(String),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

impl Json {
    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Obj(o) => o.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    /// This value as an object whose keys are reported under `path` (empty at the root).
    pub fn obj(&self, path: &str) -> Option<Obj<'_>> {
        match self {
            Json::Obj(fields) => Some(Obj { path: path.to_string(), fields }),
            _ => None,
        }
    }

    pub fn kind(&self) -> &'static str {
        match self {
            Json::Null => "null",
            Json::Bool(_) => "a boolean",
            Json::Num(_) => "a number",
            Json::Str(_) => "a string",
            Json::Arr(_) => "an array",
            Json::Obj(_) => "an object",
        }
    }
}

/// An object together with its JSON path, so errors can name the exact field.
#[derive(Clone)]
pub(crate) struct Obj<'a> {
    pub path: String,
    pub fields: &'a [(String, Json)],
}

impl<'a> Obj<'a> {
    pub fn get(&self, key: &str) -> Option<&'a Json> {
        self.fields.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    /// The path of `key` inside this object.
    pub fn at(&self, key: &str) -> String {
        if self.path.is_empty() { key.to_string() } else { format!("{}.{key}", self.path) }
    }

    /// Fails on the first key not in `known`, or on a key given twice, naming it with its path.
    pub fn check(&self, known: &[&str]) -> Result<(), String> {
        for (i, (k, _)) in self.fields.iter().enumerate() {
            if !known.contains(&k.as_str()) {
                return Err(format!("field {}", self.at(k)));
            }
            if self.fields[..i].iter().any(|(p, _)| p == k) {
                return Err(format!("field {} given twice", self.at(k)));
            }
        }
        Ok(())
    }
}

/// One parsed JSON file; every accessor error names the file and the JSON path.
pub(crate) struct Doc {
    pub path: PathBuf,
    pub root: Json,
}

impl Doc {
    pub fn load(path: &Path) -> Result<Doc, Error> {
        let text = std::fs::read_to_string(path).map_err(|source| Error::Io { path: path.to_path_buf(), source })?;
        let root = parse(&text).map_err(|e| Error::Config(format!("{}: {e}", path.display())))?;
        Ok(Doc { path: path.to_path_buf(), root })
    }

    pub fn bad(&self, m: impl std::fmt::Display) -> Error {
        Error::Config(format!("{}: {m}", self.path.display()))
    }

    pub fn unsup(&self, m: impl Into<String>) -> Error {
        Error::Unsupported { file: self.path.clone(), what: m.into() }
    }

    pub fn top(&self, known: &[&str]) -> Result<Obj<'_>, Error> {
        self.object(&self.root, "", known)
    }

    fn object<'a>(&self, v: &'a Json, path: &str, known: &[&str]) -> Result<Obj<'a>, Error> {
        let what = if path.is_empty() { "top level" } else { path };
        let o = v.obj(path).ok_or_else(|| self.bad(format!("{what} is {}, not an object", v.kind())))?;
        o.check(known).map_err(|w| self.unsup(w))?;
        Ok(o)
    }

    pub fn get<'a>(&self, o: &Obj<'a>, key: &str) -> Result<&'a Json, Error> {
        o.get(key).ok_or_else(|| self.bad(format!("missing {}", o.at(key))))
    }

    pub fn sub<'a>(&self, o: &Obj<'a>, key: &str, known: &[&str]) -> Result<Obj<'a>, Error> {
        self.object(self.get(o, key)?, &o.at(key), known)
    }

    pub fn opt_sub<'a>(&self, o: &Obj<'a>, key: &str, known: &[&str]) -> Result<Option<Obj<'a>>, Error> {
        o.get(key).map(|v| self.object(v, &o.at(key), known)).transpose()
    }

    pub fn opt_str<'a>(&self, o: &Obj<'a>, key: &str) -> Result<Option<&'a str>, Error> {
        match o.get(key) {
            None => Ok(None),
            Some(Json::Str(s)) => Ok(Some(s)),
            Some(v) => Err(self.bad(format!("{} is {}, not a string", o.at(key), v.kind()))),
        }
    }

    pub fn str<'a>(&self, o: &Obj<'a>, key: &str) -> Result<&'a str, Error> {
        self.opt_str(o, key)?.ok_or_else(|| self.bad(format!("missing {}", o.at(key))))
    }

    fn num<'a>(&self, o: &Obj<'a>, key: &str) -> Result<Option<&'a str>, Error> {
        match o.get(key) {
            None => Ok(None),
            Some(Json::Num(n)) => Ok(Some(n)),
            Some(v) => Err(self.bad(format!("{} is {}, not a number", o.at(key), v.kind()))),
        }
    }

    pub fn opt_f32(&self, o: &Obj, key: &str) -> Result<Option<f32>, Error> {
        let Some(n) = self.num(o, key)? else { return Ok(None) };
        let x: f32 = n.parse().map_err(|_| self.bad(format!("{} = {n} is not a number", o.at(key))))?;
        if !x.is_finite() {
            return Err(self.unsup(format!("{} = {n} (not a finite f32)", o.at(key))));
        }
        Ok(Some(x))
    }

    pub fn f32(&self, o: &Obj, key: &str) -> Result<f32, Error> {
        self.opt_f32(o, key)?.ok_or_else(|| self.bad(format!("missing {}", o.at(key))))
    }

    pub fn opt_int<T: std::str::FromStr>(&self, o: &Obj, key: &str) -> Result<Option<T>, Error> {
        let Some(n) = self.num(o, key)? else { return Ok(None) };
        n.parse().map(Some).map_err(|_| self.unsup(format!("{} = {n} (not an integer in range)", o.at(key))))
    }

    pub fn int<T: std::str::FromStr>(&self, o: &Obj, key: &str) -> Result<T, Error> {
        self.opt_int(o, key)?.ok_or_else(|| self.bad(format!("missing {}", o.at(key))))
    }

    pub fn arr<'a>(&self, o: &Obj<'a>, key: &str) -> Result<&'a [Json], Error> {
        match self.get(o, key)? {
            Json::Arr(a) => Ok(a),
            v => Err(self.bad(format!("{} is {}, not an array", o.at(key), v.kind()))),
        }
    }

    /// The objects of array `key`, each checked against `known`.
    pub fn items<'a>(&self, o: &Obj<'a>, key: &str, known: &[&str]) -> Result<Vec<Obj<'a>>, Error> {
        let a = self.arr(o, key)?;
        a.iter().enumerate().map(|(i, v)| self.object(v, &format!("{}[{i}]", o.at(key)), known)).collect()
    }

    pub fn strings(&self, o: &Obj, key: &str) -> Result<Vec<String>, Error> {
        let a = self.arr(o, key)?;
        a.iter()
            .enumerate()
            .map(|(i, v)| match v {
                Json::Str(s) => Ok(s.clone()),
                _ => Err(self.bad(format!("{}[{i}] is {}, not a string", o.at(key), v.kind()))),
            })
            .collect()
    }

    /// A string whose effect is fixed: it must hold one of `want` (or be absent when not `required`).
    pub fn pin_str(&self, o: &Obj, key: &str, want: &[&str], required: bool) -> Result<(), Error> {
        match self.opt_str(o, key)? {
            None if required => Err(self.bad(format!("missing {}", o.at(key)))),
            Some(s) if !want.contains(&s) => {
                Err(self.unsup(format!("{} = {s} (implemented: {})", o.at(key), want.join(", "))))
            }
            _ => Ok(()),
        }
    }

    pub fn pin_num(&self, o: &Obj, key: &str, want: f32, required: bool) -> Result<(), Error> {
        match self.opt_f32(o, key)? {
            None if required => Err(self.bad(format!("missing {}", o.at(key)))),
            Some(x) if x != want => Err(self.unsup(format!("{} = {x} (implemented: {want})", o.at(key)))),
            _ => Ok(()),
        }
    }

    pub fn pin_bool(&self, o: &Obj, key: &str, want: bool, required: bool) -> Result<(), Error> {
        match o.get(key) {
            None if required => Err(self.bad(format!("missing {}", o.at(key)))),
            None => Ok(()),
            Some(Json::Bool(b)) if *b == want => Ok(()),
            Some(v) => Err(self.unsup(format!("{} = {v:?} (implemented: {want})", o.at(key)))),
        }
    }
}

pub(crate) fn parse(text: &str) -> Result<Json, String> {
    let mut p = Parser { s: text.as_bytes(), i: 0, depth: 0 };
    let v = p.value()?;
    p.ws();
    if p.i != p.s.len() {
        return Err("trailing data after JSON value".into());
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

    fn eat(&mut self, c: u8) -> Result<(), String> {
        self.ws();
        if self.s.get(self.i) == Some(&c) {
            self.i += 1;
            Ok(())
        } else {
            Err(format!("JSON: expected '{}' at byte {}", c as char, self.i))
        }
    }

    fn lit(&mut self, word: &str, v: Json) -> Result<Json, String> {
        if self.s[self.i..].starts_with(word.as_bytes()) {
            self.i += word.len();
            Ok(v)
        } else {
            Err(format!("JSON: bad literal at byte {}", self.i))
        }
    }

    fn value(&mut self) -> Result<Json, String> {
        self.ws();
        match self.s.get(self.i) {
            None => Err("JSON: unexpected end".into()),
            Some(b'{') => self.nested(Self::object),
            Some(b'[') => self.nested(Self::array),
            Some(b'"') => Ok(Json::Str(self.string()?)),
            Some(b't') => self.lit("true", Json::Bool(true)),
            Some(b'f') => self.lit("false", Json::Bool(false)),
            Some(b'n') => self.lit("null", Json::Null),
            Some(_) => self.number(),
        }
    }

    fn nested(&mut self, f: fn(&mut Self) -> Result<Json, String>) -> Result<Json, String> {
        self.depth += 1;
        if self.depth > 64 {
            return Err("JSON: nesting too deep".into());
        }
        let v = f(self);
        self.depth -= 1;
        v
    }

    fn object(&mut self) -> Result<Json, String> {
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
                _ => return Err(format!("JSON: expected ',' or '}}' at byte {}", self.i)),
            }
        }
    }

    fn array(&mut self) -> Result<Json, String> {
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
                _ => return Err(format!("JSON: expected ',' or ']' at byte {}", self.i)),
            }
        }
    }

    fn string(&mut self) -> Result<String, String> {
        if self.s.get(self.i) != Some(&b'"') {
            return Err(format!("JSON: expected string at byte {}", self.i));
        }
        self.i += 1;
        let mut out = Vec::new();
        loop {
            let c = *self.s.get(self.i).ok_or("JSON: unterminated string")?;
            self.i += 1;
            match c {
                b'"' => return String::from_utf8(out).map_err(|_| "JSON: invalid UTF-8".into()),
                b'\\' => {
                    let e = *self.s.get(self.i).ok_or("JSON: unterminated escape")?;
                    self.i += 1;
                    match e {
                        b'"' | b'\\' | b'/' => out.push(e),
                        b'n' => out.push(b'\n'),
                        b't' => out.push(b'\t'),
                        b'r' => out.push(b'\r'),
                        b'b' => out.push(8),
                        b'f' => out.push(12),
                        b'u' => {
                            let hex = self.s.get(self.i..self.i + 4).ok_or("JSON: short \\u escape")?;
                            let n = u32::from_str_radix(std::str::from_utf8(hex).unwrap_or("x"), 16)
                                .map_err(|_| "JSON: bad \\u escape")?;
                            self.i += 4;
                            let ch = char::from_u32(n).unwrap_or('\u{fffd}');
                            out.extend(ch.to_string().as_bytes());
                        }
                        _ => return Err("JSON: bad escape".into()),
                    }
                }
                _ => out.push(c),
            }
        }
    }

    fn number(&mut self) -> Result<Json, String> {
        let start = self.i;
        while self.s.get(self.i).is_some_and(|c| matches!(c, b'0'..=b'9' | b'-' | b'+' | b'.' | b'e' | b'E')) {
            self.i += 1;
        }
        if start == self.i {
            return Err(format!("JSON: unexpected byte at {start}"));
        }
        let raw = std::str::from_utf8(&self.s[start..self.i]).unwrap();
        if !is_number(raw.as_bytes()) {
            return Err(format!("JSON: bad number '{raw}'"));
        }
        Ok(Json::Num(raw.to_string()))
    }
}

/// RFC 8259 number grammar: `-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?`.
fn is_number(s: &[u8]) -> bool {
    let digits = |i: usize| s[i..].iter().take_while(|c| c.is_ascii_digit()).count();
    let mut i = usize::from(s.first() == Some(&b'-'));
    match s.get(i) {
        Some(b'0') => i += 1,
        Some(b'1'..=b'9') => i += digits(i),
        _ => return false,
    }
    if s.get(i) == Some(&b'.') {
        let n = digits(i + 1);
        if n == 0 {
            return false;
        }
        i += 1 + n;
    }
    if matches!(s.get(i), Some(b'e' | b'E')) {
        i += 1;
        if matches!(s.get(i), Some(b'+' | b'-')) {
            i += 1;
        }
        let n = digits(i);
        if n == 0 {
            return false;
        }
        i += n;
    }
    i == s.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn escapes_literals_and_errors() {
        let v = parse(r#"{"a": [1, -2.5e1, true, false, null, "q\"\\x"], "b": {"c": {}}}"#).unwrap();
        let Json::Arr(a) = v.get("a").unwrap() else { panic!() };
        assert!(matches!(&a[5], Json::Str(s) if s == "q\"\\x"));
        assert_eq!(a[2], Json::Bool(true));
        assert_eq!(a[3], Json::Bool(false));
        assert!(parse("{} x").is_err());
        assert!(parse("[1,]").is_err());
    }

    #[test]
    fn number_grammar() {
        for ok in ["0", "-0", "1.5", "200", "1e3", "1E-3", "0.96875", "-12.0e+2"] {
            assert!(parse(ok).is_ok(), "{ok}");
        }
        for bad in ["01", "1.", ".5", "+1", "1e", "--1", "1-2", "-"] {
            assert!(parse(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn check_names_unknown_and_duplicate_keys() {
        let v = parse(r#"{"x": {"a": 1, "b": 2, "a": 3}}"#).unwrap();
        let x = v.obj("").unwrap().get("x").unwrap().obj("root.x").unwrap();
        assert_eq!(x.check(&["a"]).unwrap_err(), "field root.x.b");
        assert_eq!(x.check(&["a", "b"]).unwrap_err(), "field root.x.a given twice");
        assert_eq!(v.obj("").unwrap().at("x"), "x");
        assert!(v.obj("").unwrap().check(&["x"]).is_ok());
    }
}
