//! Sound (AED) detection: model loading and the per-step network.

mod detector;
mod model;
#[cfg(all(test, not(target_os = "android")))]
pub(crate) use model::tests::allocs;

pub use detector::{SoundDetector, SoundWindow, TypeResult, WINDOW_FRAMES, WINDOW_SAMPLES};
pub use model::SoundModel;
