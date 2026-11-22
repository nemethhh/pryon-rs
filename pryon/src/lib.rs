//! Bit-exact reimplementation of libpryon's acoustic event (Alexa Guard) detector: see [`SoundDetector`].
pub mod cmn;
pub mod config;
pub mod error;
pub mod lfbe;
mod kernels;
mod lfbe_tables;
pub mod mlp;
mod num;
#[doc(hidden)]
pub mod onnx;
mod qlstm;
pub mod sound;

pub use config::{SoundConfig, TypeConfig};
pub use error::Error;
pub use mlp::{AffineParams, Mlp, QLstmParams, QMatrix};
pub use sound::{SoundDetector, SoundModel, SoundWindow, TypeResult, WINDOW_FRAMES, WINDOW_SAMPLES};
