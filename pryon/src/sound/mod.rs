//! Sound (AED) detection: model loading and the per-step network.

mod detector;
mod model;

pub use detector::{SoundDetector, SoundWindow, TypeResult, WINDOW_FRAMES, WINDOW_SAMPLES};
pub use model::SoundModel;
