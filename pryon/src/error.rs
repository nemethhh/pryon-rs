use std::{fmt, io, path::PathBuf};

#[derive(Debug)]
pub enum Error {
    Io { path: PathBuf, source: io::Error },
    Format(String),
    Config(String),
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
