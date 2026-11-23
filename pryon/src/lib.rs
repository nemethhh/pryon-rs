//! Reimplementation of libpryon: the acoustic event (Alexa Guard) detector, bit-exact (see [`SoundDetector`]), and the
//! wake word engine (see [`Spotter`]) for the firmware and DAVS en-US wake word sets.
//!
//! A wake word set the engine does not fully understand is refused at load with [`Error::Unsupported`] naming the
//! element; nothing runs part of a set. libpryon's probabilistic accept is not reproduced: such scores are near
//! misses. The README lists the validated sets and what is still to be checked on a device.
pub mod cmn;
pub mod config;
mod cpunk;
pub mod error;
mod json;
mod kernels;
pub mod lfbe;
mod lfbe_tables;
pub mod mlp;
mod num;
#[doc(hidden)]
pub mod onnx;
mod qlstm;
pub mod sound;
pub mod wake;

pub use config::{SoundConfig, TypeConfig};
pub use error::Error;
pub use mlp::{AffineParams, Mlp, QLstmParams, QMatrix};
pub use sound::{SoundDetector, SoundModel, SoundWindow, TypeResult, WINDOW_FRAMES, WINDOW_SAMPLES};
pub use wake::{AfeTimes, Detection, DetectionKind, Spotter, SpotterModel, SpotterOptions};
