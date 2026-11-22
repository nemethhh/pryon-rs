//! Streaming sound detector: global LFBE frames cut into 998-frame windows, one result per window.

use super::model::{Net, NUM_SCORES};
use crate::cmn::{ChannelNorm, SilenceTracker};
use crate::lfbe::Lfbe;
use crate::{Error, SoundModel};
use std::collections::VecDeque;
use std::sync::Arc;

pub const WINDOW_FRAMES: usize = 998;
pub const WINDOW_SAMPLES: u64 = 159_680;
// The window's result is the score after the step at this frame.
const RESULT_T: usize = 996;

#[derive(Clone, Debug, PartialEq)]
pub struct TypeResult {
    pub score: f32,
    pub samples_above: u32,
    pub detected: bool,
    pub near_miss: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SoundWindow {
    pub index: u64,
    pub end_sample: u64,
    pub max_energy: i32,
    pub types: Vec<TypeResult>,
}

struct Energy {
    max: i32,
    above: Vec<u32>,
}

pub struct SoundDetector {
    model: Arc<SoundModel>,
    names: Vec<String>,
    // Indices into model.config.types, in requested order.
    sel: Vec<usize>,
    lfbe: Lfbe,
    silence: SilenceTracker,
    norm: ChannelNorm,
    net: Net,
    scores: [f32; NUM_SCORES],
    frame: u64,
    samples: u64,
    cur: Energy,
    done: VecDeque<Energy>,
    frames: Vec<[f32; 20]>,
    verdicts: Vec<bool>,
    released: Vec<(usize, [f32; 20])>,
}

impl SoundDetector {
    pub fn new(model: Arc<SoundModel>, types: &[&str]) -> Result<SoundDetector, Error> {
        let all = &model.config.types;
        let sel: Vec<usize> = if types.is_empty() {
            (0..all.len()).collect()
        } else {
            let mut sel = Vec::with_capacity(types.len());
            for name in types {
                let i = all.iter().position(|t| t.name == *name).ok_or_else(|| Error::UnknownType((*name).to_string()))?;
                if sel.contains(&i) {
                    return Err(Error::Config(format!("duplicate type: {name}")));
                }
                sel.push(i);
            }
            sel
        };
        let names = sel.iter().map(|&i| all[i].name.clone()).collect();
        let n = sel.len();
        Ok(SoundDetector {
            net: Net::new(&model),
            model,
            names,
            sel,
            lfbe: Lfbe::new(),
            silence: SilenceTracker::new(),
            norm: ChannelNorm::new(),
            scores: [0.0; NUM_SCORES],
            frame: 0,
            samples: 0,
            cur: Energy { max: 0, above: vec![0; n] },
            done: VecDeque::new(),
            frames: Vec::new(),
            verdicts: Vec::new(),
            released: Vec::new(),
        })
    }

    pub fn type_names(&self) -> &[String] {
        &self.names
    }

    pub fn reset(&mut self) {
        self.lfbe = Lfbe::new();
        self.silence = SilenceTracker::new();
        self.norm.reset();
        self.net.reset();
        self.frame = 0;
        self.samples = 0;
        self.cur.max = 0;
        self.cur.above.iter_mut().for_each(|a| *a = 0);
        self.done.clear();
    }

    pub fn feed(&mut self, pcm: &[i16], out: &mut Vec<SoundWindow>) {
        self.account_energy(pcm);
        self.frames.clear();
        self.verdicts.clear();
        self.lfbe.push(pcm, &mut self.frames);
        self.silence.push(pcm, &mut self.verdicts);
        for i in 0..self.frames.len() {
            let t = (self.frame % WINDOW_FRAMES as u64) as usize;
            let index = self.frame / WINDOW_FRAMES as u64;
            self.frame += 1;
            if t == 0 {
                self.norm.reset();
                self.net.reset();
            }
            self.released.clear();
            self.norm.push(t, &self.frames[i], self.verdicts[i], &mut self.released);
            if t == WINDOW_FRAMES - 1 {
                self.norm.finish(&mut self.released);
            }
            for r in 0..self.released.len() {
                let (rt, x) = self.released[r];
                if rt % 2 == 0 {
                    self.net.step(&x, &mut self.scores);
                    if rt == RESULT_T {
                        if let Some(w) = self.finish_window(index) {
                            out.push(w);
                        }
                    }
                }
            }
        }
    }

    fn account_energy(&mut self, mut pcm: &[i16]) {
        let types = &self.model.config.types;
        while !pcm.is_empty() {
            let pos = self.samples % WINDOW_SAMPLES;
            let take = pcm.len().min((WINDOW_SAMPLES - pos) as usize);
            for &s in &pcm[..take] {
                let a = (s as i32).abs();
                self.cur.max = self.cur.max.max(a);
                for (n, &i) in self.cur.above.iter_mut().zip(&self.sel) {
                    *n += (a >= types[i].energy_threshold) as u32;
                }
            }
            self.samples += take as u64;
            pcm = &pcm[take..];
            if self.samples % WINDOW_SAMPLES == 0 {
                let fresh = Energy { max: 0, above: vec![0; self.sel.len()] };
                self.done.push_back(std::mem::replace(&mut self.cur, fresh));
            }
        }
    }

    fn finish_window(&mut self, index: u64) -> Option<SoundWindow> {
        let e = self.done.pop_front()?;
        let cfg = &self.model.config.types;
        let types = self
            .sel
            .iter()
            .zip(&e.above)
            .map(|(&i, &samples_above)| {
                let c = &cfg[i];
                let score = self.scores[c.output_index];
                // libpryon: the energy gate blocks detection only, a quiet high score still reports near miss.
                let detected = samples_above >= c.energy_min_samples && score >= c.threshold;
                let near_miss = !detected && score >= c.near_miss;
                TypeResult { score, samples_above, detected, near_miss }
            })
            .collect();
        Some(SoundWindow { index, end_sample: (index + 1) * WINDOW_SAMPLES, max_energy: e.max, types })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pryon_testkit::{synth, write_model_dir, Clip, Rng, TYPE_NAMES};
    use std::path::PathBuf;

    fn model(tag: &str) -> Arc<SoundModel> {
        let d: PathBuf = std::env::temp_dir().join(format!("pryon-det-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        write_model_dir(&d, 1, 0.0);
        Arc::new(SoundModel::load(&d).unwrap())
    }

    fn run(m: &Arc<SoundModel>, types: &[&str], pcm: &[i16], chunk: usize) -> Vec<SoundWindow> {
        let mut det = SoundDetector::new(m.clone(), types).unwrap();
        let mut out = Vec::new();
        for c in pcm.chunks(chunk) {
            det.feed(c, &mut out);
        }
        out
    }

    fn noise(n: usize) -> Vec<i16> {
        let mut rng = Rng(12345);
        (0..n).map(|_| ((rng.next_u32() % 6001) as i32 - 3000) as i16).collect()
    }

    #[test]
    fn window_count_and_ends() {
        let m = model("count");
        let pcm = noise(3 * 159_680 + 80);
        let w = run(&m, &[], &pcm[..pcm.len() - 1], 4000);
        assert_eq!(w.iter().map(|w| w.end_sample).collect::<Vec<_>>(), [159_680, 319_360]);
        let w = run(&m, &[], &pcm, 4000);
        assert_eq!(w.iter().map(|w| w.index).collect::<Vec<_>>(), [0, 1, 2]);
        assert_eq!(w[2].end_sample, 3 * 159_680);
        assert_eq!(run(&m, &[], &synth(Clip::Noise), 4000).len(), 2);
    }

    #[test]
    fn chunking_invariant() {
        let m = model("chunk");
        let pcm = synth(Clip::Noise);
        let whole = run(&m, &[], &pcm, pcm.len());
        assert_eq!(whole.len(), 2);
        for chunk in [1, 7, 160, 3333] {
            let got = run(&m, &[], &pcm, chunk);
            assert_eq!(got, whole, "chunk {chunk}");
            for (a, b) in got.iter().zip(&whole) {
                for (x, y) in a.types.iter().zip(&b.types) {
                    assert_eq!(x.score.to_bits(), y.score.to_bits());
                }
            }
        }
    }

    fn clip_with_energy(v: &[(i16, usize)]) -> Vec<i16> {
        let mut pcm: Vec<i16> = v.iter().flat_map(|&(x, n)| std::iter::repeat(x).take(n)).collect();
        pcm.resize(159_680 + 400, 0);
        pcm
    }

    #[test]
    fn energy_counts() {
        let m = model("energy");
        let w = run(&m, &[], &clip_with_energy(&[(82, 500), (81, 300), (-108, 7)]), 5000);
        assert_eq!(w.len(), 1);
        assert_eq!(w[0].max_energy, 108);
        let above = |n: &str| w[0].types[TYPE_NAMES.iter().position(|t| *t == n).unwrap()].samples_above;
        assert_eq!(above("dogBark"), 507);
        assert_eq!(above("humanPresence"), 7);
    }

    #[test]
    fn max_energy_min_sample() {
        let m = model("min");
        let w = run(&m, &[], &clip_with_energy(&[(5, 10), (-32768, 1)]), 5000);
        assert_eq!(w[0].max_energy, 32768);
    }

    // Thresholds 0.0 / near miss 0.0: every score clears both, so only the energy gate decides.
    #[test]
    fn gate_blocks_detection() {
        let m = model("gate");
        let w = run(&m, &[], &synth(Clip::Zeros), 5000);
        assert!(!w.is_empty());
        for win in &w {
            assert!(win.types.iter().all(|t| !t.detected && t.near_miss && t.samples_above == 0));
        }
    }

    #[test]
    fn type_subset_order() {
        let m = model("subset");
        let pcm = synth(Clip::Noise);
        let full = run(&m, &[], &pcm, 4000);
        let sub = run(&m, &["dogBark", "humanPresence"], &pcm, 4000);
        for (f, s) in full.iter().zip(&sub) {
            assert_eq!(s.types, [f.types[3].clone(), f.types[0].clone()]);
            assert_eq!(s.max_energy, f.max_energy);
        }
        let det = SoundDetector::new(m, &["dogBark", "humanPresence"]).unwrap();
        assert_eq!(det.type_names(), ["dogBark", "humanPresence"]);
    }

    #[test]
    fn bad_type_lists_error() {
        let m = model("bad");
        assert!(matches!(SoundDetector::new(m.clone(), &["cat"]), Err(Error::UnknownType(n)) if n == "cat"));
        assert!(matches!(SoundDetector::new(m, &["dogBark", "dogBark"]), Err(Error::Config(_))));
    }

    #[test]
    fn unknown_type_errors() {
        let m = model("unk");
        assert!(matches!(SoundDetector::new(m, &["cat"]), Err(Error::UnknownType(_))));
    }

    #[test]
    fn reset_restarts_windows() {
        let m = model("reset");
        let pcm = synth(Clip::Noise);
        let mut det = SoundDetector::new(m, &[]).unwrap();
        let mut first = Vec::new();
        det.feed(&pcm[..200_000], &mut first);
        det.reset();
        let mut a = Vec::new();
        det.feed(&pcm, &mut a);
        let mut fresh = SoundDetector::new(model("reset2"), &[]).unwrap();
        let mut b = Vec::new();
        fresh.feed(&pcm, &mut b);
        assert_eq!(a, b);
        assert_eq!(a.len(), 2);
    }
}
