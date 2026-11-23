//! Streaming sound detector: global LFBE frames cut into 998-frame windows, one result per window.

use super::model::{Net, NUM_SCORES};
use crate::cmn::{ChannelNorm, SilenceTracker};
use crate::lfbe::Lfbe;
use crate::{Error, SoundModel};
use std::collections::VecDeque;
use std::sync::Arc;

/// Feature frames (10 ms each) in one window.
pub const WINDOW_FRAMES: usize = 998;
/// Samples in one window (9.98 s at 16 kHz).
pub const WINDOW_SAMPLES: u64 = 159_680;
// A 400-sample frame at hop 160 spans 80-sample blocks 2f..2f+4.
const BLOCK: u64 = 80;
const FRAME_BLOCKS: usize = 5;

#[derive(Clone, Debug, PartialEq)]
/// One sound type's result for one window.
pub struct TypeResult {
    /// Sigmoid score of the type's output, 0..1.
    pub score: f32,
    /// Samples of the window with `|x|` >= the type's energy threshold (-32768 never counts).
    pub samples_above: u32,
    /// `score >= threshold && samples_above >= min samples`.
    pub detected: bool,
    /// `!detected && score >= near-miss threshold` (no energy test).
    pub near_miss: bool,
}

#[derive(Clone, Debug, PartialEq)]
/// The result of one complete window of `WINDOW_SAMPLES` samples counted since `new`/`reset`.
pub struct SoundWindow {
    /// Zero-based window number.
    pub index: u64,
    /// `(index + 1) * WINDOW_SAMPLES`.
    pub end_sample: u64,
    /// Max per-frame `|x|` over the window's 998 frames, 0..32767 (a frame containing -32768 contributes nothing).
    pub max_energy: i32,
    /// One entry per requested type, in `SoundDetector::type_names` order.
    pub types: Vec<TypeResult>,
}

/// Per window: samples above each type's threshold (window samples) and max |x| (window frames).
struct Energy {
    thresholds: Vec<i32>,
    samples: u64,
    above: Vec<u32>,
    done_above: VecDeque<Vec<u32>>,
    block_max: i32,
    block_void: bool,
    // The last five block maxes, None for a block holding -32768.
    blocks: [Option<i32>; FRAME_BLOCKS],
    frames: u64,
    max: i32,
    done_max: VecDeque<i32>,
}

/// Streaming detector over 16 kHz mono s16 PCM; a trailing partial window yields nothing.
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
    energy: Energy,
    frames: Vec<[f32; 20]>,
    verdicts: Vec<bool>,
    released: Vec<(usize, [f32; 20])>,
}

impl Energy {
    fn new(thresholds: Vec<i32>) -> Energy {
        Energy {
            above: vec![0; thresholds.len()],
            thresholds,
            samples: 0,
            done_above: VecDeque::new(),
            block_max: 0,
            block_void: false,
            blocks: [None; FRAME_BLOCKS],
            frames: 0,
            max: 0,
            done_max: VecDeque::new(),
        }
    }

    fn reset(&mut self) {
        *self = Energy::new(std::mem::take(&mut self.thresholds));
    }

    fn push(&mut self, mut pcm: &[i16]) {
        // Window ends are block ends (159680 = 1996 * 80), so whole-segment passes never straddle either.
        while !pcm.is_empty() {
            let left = (BLOCK - self.samples % BLOCK) as usize;
            let (seg, rest) = pcm.split_at(left.min(pcm.len()));
            pcm = rest;
            // libpryon's |x| is i16: -32768 never counts and voids every frame max it falls in (oracle probes).
            let n_min = if seg.contains(&i16::MIN) { seg.iter().filter(|&&s| s == i16::MIN).count() } else { 0 };
            self.block_void |= n_min > 0;
            // A void block's max is never read, so -32768 (abs 32768) may enter it.
            self.block_max = self.block_max.max(seg.iter().map(|&s| (s as i32).abs()).max().unwrap_or(0));
            for (n, &t) in self.above.iter_mut().zip(&self.thresholds) {
                let all = seg.iter().filter(|&&s| (s as i32).abs() >= t).count();
                *n += (all - if t <= 32768 { n_min } else { 0 }) as u32;
            }
            self.samples += seg.len() as u64;
            if self.samples.is_multiple_of(BLOCK) {
                self.end_block();
            }
            if self.samples.is_multiple_of(WINDOW_SAMPLES) {
                let fresh = vec![0; self.above.len()];
                self.done_above.push_back(std::mem::replace(&mut self.above, fresh));
            }
        }
    }

    fn end_block(&mut self) {
        let b = self.samples / BLOCK - 1;
        self.blocks[(b % FRAME_BLOCKS as u64) as usize] = (!self.block_void).then_some(self.block_max);
        self.block_max = 0;
        self.block_void = false;
        if b + 1 >= FRAME_BLOCKS as u64 && b.is_multiple_of(2) {
            let frame_max = self.blocks.iter().try_fold(0, |m, b| b.map(|x| m.max(x))).unwrap_or(0);
            self.max = self.max.max(frame_max);
            self.frames += 1;
            if self.frames.is_multiple_of(WINDOW_FRAMES as u64) {
                self.done_max.push_back(std::mem::replace(&mut self.max, 0));
            }
        }
    }

    fn pop(&mut self) -> Option<(Vec<u32>, i32)> {
        debug_assert_eq!(self.done_above.is_empty(), self.done_max.is_empty());
        Some((self.done_above.pop_front()?, self.done_max.pop_front()?))
    }
}

impl SoundDetector {
    /// Serves the named types (empty = all, in model order); unknown or duplicate names are an `Err`.
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
        let energy = Energy::new(sel.iter().map(|&i| all[i].energy_threshold).collect());
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
            energy,
            frames: Vec::new(),
            verdicts: Vec::new(),
            released: Vec::new(),
        })
    }

    /// The served types, in the order of `SoundWindow::types`.
    pub fn type_names(&self) -> &[String] {
        &self.names
    }

    /// Restarts the stream: window counting begins at 0.
    pub fn reset(&mut self) {
        self.lfbe = Lfbe::new();
        self.silence = SilenceTracker::new();
        self.norm.reset();
        self.net.reset();
        self.frame = 0;
        self.energy.reset();
    }

    /// Appends the windows completed by `pcm`; a window comes out in the feed that supplies the 240th sample past its
    /// end, and one feed can release up to ~50 network steps at once (channel-norm warm-up or the frame-997 flush).
    pub fn feed(&mut self, pcm: &[i16], out: &mut Vec<SoundWindow>) {
        self.energy.push(pcm);
        self.frames.clear();
        self.verdicts.clear();
        self.lfbe.push(pcm, &mut self.frames);
        self.silence.push(pcm, &mut self.verdicts);
        debug_assert_eq!(self.frames.len(), self.verdicts.len());
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
                }
            }
            // Scores end at the frame-996 step; frame 997 still feeds the energy max.
            if t == WINDOW_FRAMES - 1 {
                if let Some(w) = self.finish_window(index) {
                    out.push(w);
                }
            }
        }
    }

    fn finish_window(&mut self, index: u64) -> Option<SoundWindow> {
        let (above, max_energy) = self.energy.pop()?;
        let cfg = &self.model.config.types;
        let types = self
            .sel
            .iter()
            .zip(&above)
            .map(|(&i, &samples_above)| {
                let c = &cfg[i];
                let score = self.scores[c.output_index];
                // libpryon: the energy gate blocks detection only, a quiet high score still reports near miss.
                let detected = samples_above >= c.energy_min_samples && score >= c.threshold;
                let near_miss = !detected && score >= c.near_miss;
                TypeResult { score, samples_above, detected, near_miss }
            })
            .collect();
        Some(SoundWindow { index, end_sample: (index + 1) * WINDOW_SAMPLES, max_energy, types })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pryon_testkit::{synth, write_model_dir, Clip, Rng, TempDir, TYPE_NAMES};

    fn model(tag: &str) -> Arc<SoundModel> {
        let d = TempDir::new(&format!("det-{tag}"));
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
        // A window closes with its last frame, 240 samples past its end.
        let pcm = noise(3 * 159_680 + 240);
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
        let mut pcm: Vec<i16> = v.iter().flat_map(|&(x, n)| std::iter::repeat_n(x, n)).collect();
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

    fn spikes(len: usize, v: &[(usize, i16)]) -> Vec<i16> {
        let mut pcm = vec![0i16; len];
        v.iter().for_each(|&(i, x)| pcm[i] = x);
        pcm
    }

    // Oracle probes: -32768 is never above threshold and voids the max of every 400-sample frame holding it.
    #[test]
    fn max_energy_min_sample() {
        let m = model("min");
        let w = run(&m, &[], &clip_with_energy(&[(5, 10), (-32768, 1)]), 5000);
        assert_eq!(w[0].max_energy, 0);
        assert!(w[0].types.iter().all(|t| t.samples_above == 0));
        // 1050 lies in frames 5, 6, both holding 1150; 1400 in frames 7 (holds 1150) and 8 (1280..1680, does not).
        let w = run(&m, &[], &spikes(160_080, &[(1050, 900), (1150, -32768), (1400, 300)]), 7);
        assert_eq!(w[0].max_energy, 300);
        assert!(w[0].types.iter().all(|t| t.samples_above == 2));
        let w = run(&m, &[], &spikes(160_080, &[(1050, 900), (1200, -32768)]), 33);
        assert_eq!(w[0].max_energy, 900);
    }

    // Segment-wise counting against the per-sample rule, over thresholds at and past both ends of |x|.
    #[test]
    fn energy_counts_match_per_sample_rule() {
        let thresholds = vec![-5, 0, 1, 82, 32767, 32768, 40000];
        let mut rng = Rng(77);
        let n = 2 * WINDOW_SAMPLES as usize + 123;
        let pcm: Vec<i16> = (0..n)
            .map(|_| match rng.next_u32() % 50 {
                0 => i16::MIN,
                1 => i16::MAX,
                2 => 0,
                _ => (rng.next_u32() % 65536) as u16 as i16 / 64,
            })
            .collect();
        let mut e = Energy::new(thresholds.clone());
        let mut i = 0;
        while i < n {
            let len = (rng.next_u32() % 500) as usize + 1;
            e.push(&pcm[i..(i + len).min(n)]);
            i += len;
        }
        for w in 0..2 {
            let win = &pcm[w * WINDOW_SAMPLES as usize..(w + 1) * WINDOW_SAMPLES as usize];
            let want: Vec<u32> = thresholds
                .iter()
                .map(|&t| win.iter().filter(|&&s| s != i16::MIN && (s as i32).abs() >= t).count() as u32)
                .collect();
            assert_eq!(e.done_above.pop_front().unwrap(), want, "window {w}");
        }
    }

    // The max covers the window's 998 frames, whose last two reach 240 samples into the next window.
    #[test]
    fn max_energy_spans_window_frames() {
        let m = model("edge");
        let n = WINDOW_SAMPLES as usize;
        let w = run(&m, &[], &spikes(2 * n + 400, &[(n - 30, 500), (n + 30, 600), (2 * n - 30, 400)]), 4000);
        assert_eq!(w.iter().map(|w| w.max_energy).collect::<Vec<_>>(), [600, 600]);
        let w = run(&m, &[], &spikes(2 * n + 400, &[(n - 30, 500), (n + 300, 400)]), 4000);
        assert_eq!(w.iter().map(|w| w.max_energy).collect::<Vec<_>>(), [500, 400]);
        assert!(w.iter().all(|w| w.types.iter().all(|t| t.samples_above == 1)));
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
