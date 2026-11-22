use std::{fmt, io, path::PathBuf};

/// Everything loading a model or opening a detector can fail with; nothing in the crate panics on bad model files.
#[derive(Debug)]
pub enum Error {
    /// A model file could not be read.
    Io { path: PathBuf, source: io::Error },
    /// A model file is malformed or its shapes do not match the network.
    Format(String),
    /// `pryon.config` or `AED.json` is malformed, or the requested type list is invalid (e.g. duplicate).
    Config(String),
    /// A requested type name is not in the model.
    UnknownType(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io { path, source } => write!(f, "{}: {source}", path.display()),
            Error::Format(m) => write!(f, "bad model format: {m}"),
            Error::Config(m) => write!(f, "bad config: {m}"),
            Error::UnknownType(t) => write!(f, "unknown type: {t}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}
