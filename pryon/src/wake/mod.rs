//! Wake word engine: model sets in the firmware and DAVS layouts, streamed through [`Spotter`].

mod afe;
mod kwcfg;
mod pconfig;
mod pipeline;
mod rules;
mod set;

#[doc(hidden)]
pub use kwcfg::{
    ClassifierSpec, MachineSpec, MachineState, Override, RestartSpec, SpotterSpec, Thresholds, WordComponent,
};
#[doc(hidden)]
pub use pipeline::{ComponentRef, LoadedSet, Pipeline, StepOut};
#[doc(hidden)]
pub use set::{describe, load_set, SetSpec, StageInput, StageSpec};

use crate::Error;
use afe::AfeClock;
use rules::Orchestra;
use std::path::Path;
use std::sync::Arc;

/// A loaded wake model set, shared by any number of [`Spotter`]s through an `Arc`.
pub struct SpotterModel {
    set: Arc<LoadedSet>,
    keywords: Vec<String>,
}

impl SpotterModel {
    /// Loads the set whose `pryon.manifest` is in `dir`; a set the engine cannot fully run is refused.
    pub fn load(dir: &Path) -> Result<SpotterModel, Error> {
        let set = Arc::new(LoadedSet::load(dir)?);
        Orchestra::new(&set)?;
        let keywords = rules::keyword_names(&set);
        Ok(SpotterModel { set, keywords })
    }

    /// Served keywords in words.shrunk.txt order, each once; [`Detection::keyword`] indexes this.
    pub fn keywords(&self) -> &[String] {
        &self.keywords
    }

    /// The set's checksum (its checksum file, else a hash of its files).
    pub fn checksum(&self) -> &str {
        &self.set.spec.checksum
    }
}

#[derive(Default, Clone)]
pub struct SpotterOptions {
    /// The stream carries the Echo front end's metadata in its lowest bits (an Echo's `micAsr`): detections get
    /// [`Detection::afe`] as libpryon reports it. Off: no per-sample work.
    pub echo_stream: bool,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DetectionKind {
    Accept,
    /// Accepted by the keyword's verifier spotter.
    AcceptAuxiliary,
    NearMiss,
}

/// The Echo front end's clock (ms, 16 bit, wrapping) at a detection's keyword start and end: libpryon's
/// `timestamp_before_ww_start` / `timestamp_before_ww_end`, each the clock of the newest metadata frame decoded by the
/// detection that starts at or before that sample (0 if none, or for sample 0).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct AfeTimes {
    pub start: i64,
    /// Read at libpryon's keyword end index, which may lie after [`Detection::end_sample`].
    pub end: i64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Detection {
    /// Index into [`SpotterModel::keywords`].
    pub keyword: usize,
    pub kind: DetectionKind,
    pub score: f32,
    /// Samples since the spotter was created; [`Spotter::reset`] does not restart the count.
    pub begin_sample: u64,
    pub end_sample: u64,
    /// With [`SpotterOptions::echo_stream`], once a metadata frame has decoded since the spotter's creation, last
    /// [`Spotter::reset`] or decoder restart (libpryon's `has_audio_lsb_metadata`); otherwise `None`.
    pub afe: Option<AfeTimes>,
}

/// One 16 kHz mono stream through a [`SpotterModel`].
pub struct Spotter {
    model: Arc<SpotterModel>,
    pipeline: Pipeline,
    orchestra: Orchestra,
    afe: Option<AfeClock>,
    /// Samples fed since creation.
    fed: u64,
}

const WINDOW: u64 = crate::lfbe::FL as u64;
const HOP: u64 = crate::lfbe::HOP as u64;

impl Spotter {
    pub fn new(model: Arc<SpotterModel>, opts: SpotterOptions) -> Result<Spotter, Error> {
        let pipeline = Pipeline::new(&model.set);
        let orchestra = Orchestra::new(&model.set)?;
        let afe = opts.echo_stream.then(|| AfeClock::new(orchestra.afe_records()));
        Ok(Spotter { model, pipeline, orchestra, afe, fed: 0 })
    }

    pub fn keywords(&self) -> &[String] {
        &self.model.keywords
    }

    /// Feeds PCM of any length; appends the detections it completes. Bit-exact only under the default FPSCR on ARM.
    pub fn feed(&mut self, pcm: &[i16], out: &mut Vec<Detection>) {
        let mut rest = pcm;
        while !rest.is_empty() {
            // pieces end where frames complete, so a restart takes effect exactly after its frame
            let since = self.fed - self.orchestra.origin();
            let next = if since < WINDOW { WINDOW } else { since + HOP - (since - WINDOW) % HOP };
            let n = rest.len().min((next - since) as usize);
            if let Some(a) = &mut self.afe {
                a.push(&rest[..n]);
            }
            let (orchestra, afe) = (&mut self.orchestra, self.afe.as_ref());
            self.pipeline.push_frames(&rest[..n], &mut |f, s| orchestra.tick(f, s, afe, out));
            self.fed += n as u64;
            rest = &rest[n..];
            if self.orchestra.restart_due() {
                self.pipeline.reset();
                self.orchestra.restart(self.fed);
                if let Some(a) = &mut self.afe {
                    a.flush();
                }
            }
        }
    }

    /// Stream sample of the current frame 0 (moves on a reset or a decoder restart).
    #[doc(hidden)]
    pub fn origin(&self) -> u64 {
        self.orchestra.origin()
    }

    /// Sets a client property that kw.cfg threshold overrides test, as libpryon does: from the first 10 ms frame
    /// that starts after the samples fed so far. Unknown names are ignored.
    pub fn set_property(&mut self, name: &str, value: i32) {
        let at = self.fed - self.orchestra.origin();
        self.orchestra.set_property(name, value, at);
    }

    /// Session end: a pending near miss is reported into `out`, then the front end, network, rings, rules and
    /// operating state restart; client properties stay and the sample clock keeps counting. The Echo metadata
    /// decoder restarts first, so that near miss has no [`Detection::afe`] (libpryon flushes it before).
    pub fn reset(&mut self, out: &mut Vec<Detection>) {
        self.pipeline.reset();
        if let Some(a) = &mut self.afe {
            a.flush();
        }
        self.orchestra.reset(self.fed, self.afe.as_ref(), out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pryon_testkit::wakeset::{write_davs_set, write_fw_set, WakeSetOpts};
    use pryon_testkit::{synth, Clip, TempDir};

    fn model(fw: bool, force: f32) -> (TempDir, Arc<SpotterModel>) {
        let d = TempDir::new("spotter");
        let o = WakeSetOpts { seed: 1, force_score: Some(force) };
        if fw { write_fw_set(&d, &o) } else { write_davs_set(&d, &o) }
        let m = Arc::new(SpotterModel::load(&d).unwrap());
        (d, m)
    }

    #[cfg(not(target_os = "android"))]
    #[test]
    fn feed_allocates_nothing() {
        use crate::sound::allocs;
        for (fw, echo_stream) in [(true, false), (false, false), (true, true), (false, true)] {
            let (_d, m) = model(fw, 0.99);
            let mut sp = Spotter::new(m, SpotterOptions { echo_stream }).unwrap();
            let pcm = pryon_testkit::lsb_stream(synth(Clip::Noise), 0);
            let mut out = Vec::with_capacity(1000);
            let before = allocs();
            for c in pcm.chunks(1280) {
                sp.feed(c, &mut out);
            }
            assert_eq!(allocs(), before, "fw {fw} echo {echo_stream}: feed allocated");
            assert!(out.len() > 10, "fw {fw}: {} detections", out.len());
            assert_eq!(out.iter().all(|d| d.afe.is_some()), echo_stream);
        }
    }

    #[test]
    fn min_max_accept_do_not_bound_overrides() {
        // ALEXA: accept 0.92 within [min 0.85, max 0.93]; the AlarmState override 0.7 lies outside and still applies
        // (libpryon prints active_threshold=0.7 on this set at --force 0.8 with -p AlarmState=1)
        let (_d, m) = model(false, 0.8);
        let t = &m.set.spec.spotters.iter().find(|s| s.name == "ALEXA" && !s.auxiliary).unwrap().thresholds;
        assert_eq!((t.min_accept, t.max_accept), (Some(0.85), Some(0.93)));
        assert!(t.overrides.iter().any(|o| o.prop == "AlarmState" && o.accept < 0.85));
        let mut sp = Spotter::new(m.clone(), SpotterOptions::default()).unwrap();
        sp.set_property("AlarmState", 1);
        let mut out = Vec::new();
        sp.feed(&synth(Clip::Noise)[..16000], &mut out);
        let d = out.iter().find(|d| d.keyword == 0).unwrap();
        assert_eq!((d.kind, d.score, d.end_sample), (DetectionKind::Accept, 0.8, 1280));
    }

    #[test]
    fn property_selects_override() {
        // ALEXA accept 0.92, verifier 0.9; AlarmState=1 lowers the primary to 0.7
        let (_d, m) = model(false, 0.8);
        let pcm = &synth(Clip::Noise)[..16000 * 4];
        let run = |props: &[(&str, i32)]| {
            let mut sp = Spotter::new(m.clone(), SpotterOptions::default()).unwrap();
            for &(n, v) in props {
                sp.set_property(n, v);
            }
            let mut out = Vec::new();
            sp.feed(pcm, &mut out);
            out.iter().filter(|d| d.keyword == 0).map(|d| d.kind).collect::<Vec<_>>()
        };
        let plain = run(&[]);
        assert!(!plain.is_empty() && plain.iter().all(|&k| k == DetectionKind::NearMiss), "{plain:?}");
        assert_eq!(run(&[("NoSuchProperty", 1)]), plain);
        assert_eq!(run(&[("AlarmState", 2)]), plain);
        let alarm = run(&[("AlarmState", 1)]);
        assert!(!alarm.is_empty() && alarm.iter().all(|&k| k == DetectionKind::Accept), "{alarm:?}");
    }
}
