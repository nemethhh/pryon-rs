//! Reimplementation of libpryon: the acoustic event (Alexa Guard) detector, bit-exact (see [`SoundDetector`]), and the
//! wake word engine (see [`Spotter`]).
pub mod cmn;
pub mod config;
mod cpunk;
pub mod error;
mod json;
pub mod lfbe;
mod kernels;
mod lfbe_tables;
pub mod mlp;
mod num;
pub mod onnx;
mod qlstm;
pub mod sound;
pub mod wake;

pub use config::{SoundConfig, TypeConfig};
pub use error::Error;
pub use mlp::{AffineParams, Mlp, QLstmParams, QMatrix};
pub use sound::{SoundDetector, SoundModel, SoundWindow, TypeResult, WINDOW_FRAMES, WINDOW_SAMPLES};
pub use wake::{AfeTimes, Detection, DetectionKind, Spotter, SpotterModel, SpotterOptions};
