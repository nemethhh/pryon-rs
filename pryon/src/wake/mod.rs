//! Wake word engine: model sets in the firmware and DAVS layouts.

mod kwcfg;
mod pconfig;
mod pipeline;
mod set;

#[doc(hidden)]
pub use kwcfg::{ClassifierSpec, MachineSpec, MachineState, Override, SpotterSpec, Thresholds, WordComponent};
#[doc(hidden)]
pub use pipeline::{ComponentRef, LoadedSet, Pipeline, StepOut};
#[doc(hidden)]
pub use set::{describe, load_set, SetSpec, StageInput, StageSpec};
