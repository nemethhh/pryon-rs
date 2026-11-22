//! Per-window channel norm with digital-silence handling (`aed_emu.py` `cmn_window`), released in frame order.
use std::collections::VecDeque;

const BASE: usize = 100;
const DECAY: f32 = 0.003;
const SILENT_RUN_FLUSH: usize = 49;
const FRAME: usize = 400;
const HOP: usize = 160;
const SILENT_ZEROS: usize = 200;
const HISTORY: usize = 2 * FRAME;

type Frame = [f32; 20];

pub struct ChannelNorm {
    sum: Frame,
    count: usize,
    mean: Option<Frame>,
    run: usize,
    // Unreleased frames before stats: (t, x, non-silent).
    pending: Vec<(usize, Frame, bool)>,
}

impl Default for ChannelNorm {
    fn default() -> Self {
        Self::new()
    }
}

impl ChannelNorm {
    pub fn new() -> ChannelNorm {
        ChannelNorm { sum: [0.0; 20], count: 0, mean: None, run: 0, pending: Vec::new() }
    }

    pub fn reset(&mut self) {
        self.sum = [0.0; 20];
        self.count = 0;
        self.mean = None;
        self.run = 0;
        self.pending.clear();
    }

    /// t: frame index within the window; silent: digital-silence flag. Pushes (t, normalised) in increasing t
    /// as soon as ready (may push 0, 1 or many).
    pub fn push(&mut self, t: usize, x: &Frame, silent: bool, out: &mut Vec<(usize, Frame)>) {
        if silent {
            self.run += 1;
            match &self.mean {
                Some(m) => out.push((t, sub(x, m))),
                None if self.pending.is_empty() => out.push((t, *x)),
                None => self.pending.push((t, *x, false)),
            }
            if self.mean.is_none() && self.run >= SILENT_RUN_FLUSH {
                self.flush_raw(out);
            }
            return;
        }
        self.run = 0;
        if let Some(m) = &mut self.mean {
            let oma = 1.0f32 - DECAY;
            for (mi, &xi) in m.iter_mut().zip(x) {
                *mi = (oma * *mi) + (DECAY * xi);
            }
            out.push((t, sub(x, m)));
            return;
        }
        for (s, &xi) in self.sum.iter_mut().zip(x) {
            *s += xi;
        }
        self.count += 1;
        self.pending.push((t, *x, true));
        if self.count == BASE {
            let mut m = self.sum;
            for v in &mut m {
                *v *= 0.01f32;
            }
            for (pt, px, non_silent) in self.pending.drain(..) {
                out.push((pt, if non_silent { sub(&px, &m) } else { px }));
            }
            self.mean = Some(m);
        }
    }

    /// end of window: flush queued frames raw (stats never ready).
    pub fn finish(&mut self, out: &mut Vec<(usize, Frame)>) {
        self.flush_raw(out);
    }

    fn flush_raw(&mut self, out: &mut Vec<(usize, Frame)>) {
        out.extend(self.pending.drain(..).map(|(t, x, _)| (t, x)));
    }
}

fn sub(x: &Frame, m: &Frame) -> Frame {
    let mut r = [0.0f32; 20];
    for ((ri, &xi), &mi) in r.iter_mut().zip(x).zip(m) {
        *ri = xi - mi;
    }
    r
}

pub struct SilenceTracker {
    total: u64,
    // Zero flags of the last HISTORY samples.
    zeros: VecDeque<bool>,
}

impl Default for SilenceTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl SilenceTracker {
    pub fn new() -> Self {
        SilenceTracker { total: 0, zeros: VecDeque::with_capacity(HISTORY + 1) }
    }

    pub fn push(&mut self, pcm: &[i16]) {
        self.total += pcm.len() as u64;
        // Only the last HISTORY samples can matter, so skip any older part of a long push.
        let keep = pcm.len().min(HISTORY);
        self.zeros.extend(pcm[pcm.len() - keep..].iter().map(|&s| s == 0));
        while self.zeros.len() > HISTORY {
            self.zeros.pop_front();
        }
    }

    /// frame f is silent iff it contains >= 200 contiguous zero samples inside its own 400 samples.
    /// Answerable while 160f+400 <= samples pushed and 160f >= pushed - 800; otherwise false.
    pub fn frame_silent(&self, f: u64) -> bool {
        let start = f * HOP as u64;
        let end = start + FRAME as u64;
        let oldest = self.total - self.zeros.len() as u64;
        if end > self.total || start < oldest {
            return false;
        }
        let off = (start - oldest) as usize;
        let mut run = 0;
        for &z in self.zeros.range(off..off + FRAME) {
            run = if z { run + 1 } else { 0 };
            if run >= SILENT_ZEROS {
                return true;
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fr(v: f32) -> Frame {
        [v; 20]
    }

    fn run(frames: &[(Frame, bool)]) -> Vec<(usize, Frame)> {
        let mut n = ChannelNorm::new();
        let mut out = Vec::new();
        for (t, (x, s)) in frames.iter().enumerate() {
            n.push(t, x, *s, &mut out);
        }
        out
    }

    fn assert_in_order(out: &[(usize, Frame)]) {
        for (i, (t, _)) in out.iter().enumerate() {
            assert_eq!(*t, i);
        }
    }

    #[test]
    fn stats_after_100() {
        let frames = vec![(fr(3.7), false); 100];
        let mut n = ChannelNorm::new();
        let mut out = Vec::new();
        for t in 0..99 {
            n.push(t, &frames[t].0, false, &mut out);
            assert!(out.is_empty());
        }
        n.push(99, &frames[99].0, false, &mut out);
        assert_eq!(out.len(), 100);
        let mut s = 0.0f32;
        for _ in 0..100 {
            s += 3.7f32;
        }
        let m = s * 0.01f32;
        assert_in_order(&out);
        for (_, y) in &out {
            assert_eq!(y[0], 3.7f32 - m);
        }
    }

    #[test]
    fn decay_update() {
        let mut frames = vec![(fr(2.0), false); 100];
        frames.push((fr(5.0), false));
        let out = run(&frames);
        assert_eq!(out.len(), 101);
        let mut s = 0.0f32;
        for _ in 0..100 {
            s += 2.0f32;
        }
        let m = s * 0.01f32;
        let oma = 1.0f32 - 0.003f32;
        let m2 = (oma * m) + (0.003f32 * 5.0f32);
        assert_eq!(out[100].0, 100);
        assert_eq!(out[100].1[7], 5.0f32 - m2);
    }

    #[test]
    fn silent_passes_raw_before_stats_in_order() {
        let mut frames: Vec<(Frame, bool)> = (0..10).map(|i| (fr(1.0 + i as f32), false)).collect();
        frames.push((fr(-9.0), true));
        frames.extend((0..90).map(|i| (fr(0.5 + i as f32), false)));
        // 99 non-silent frames leave everything queued; the 100th (t=100 here) releases all 101
        assert!(run(&frames[..100]).is_empty());
        let out = run(&frames);
        assert_eq!(out.len(), 101);
        assert_in_order(&out);
        assert_eq!(out[10].1, fr(-9.0));
        let mut s = 0.0f32;
        for (x, sil) in &frames {
            if !sil {
                s += x[0];
            }
        }
        let m = s * 0.01f32;
        assert_eq!(out[0].1[0], 1.0f32 - m);
        assert_eq!(out[100].1[0], (0.5f32 + 89.0) - m);
    }

    #[test]
    fn run49_flushes_raw() {
        let mut frames: Vec<(Frame, bool)> = (0..10).map(|i| (fr(1.0 + i as f32), false)).collect();
        let mut n = ChannelNorm::new();
        let mut out = Vec::new();
        for (t, (x, s)) in frames.iter().enumerate() {
            n.push(t, x, *s, &mut out);
        }
        assert!(out.is_empty());
        for t in 10..58 {
            n.push(t, &fr(0.0), true, &mut out);
        }
        assert!(out.is_empty());
        n.push(58, &fr(0.0), true, &mut out);
        assert_eq!(out.len(), 59);
        assert_in_order(&out);
        for (t, y) in &out[..10] {
            assert_eq!(*y, fr(1.0 + *t as f32));
        }
        // further silent frames are raw immediately
        out.clear();
        n.push(59, &fr(0.25), true, &mut out);
        assert_eq!(out, vec![(59, fr(0.25))]);
        out.clear();
        // 90 more non-silent frames complete the 100 non-silent sums (10 earlier + 90)
        for i in 0..90 {
            let x = fr(2.0 + i as f32);
            frames.push((x, false));
            n.push(60 + i, &x, false, &mut out);
        }
        assert_eq!(out.len(), 90);
        assert_in_order_from(&out, 60);
        let mut s = 0.0f32;
        for i in 0..10 {
            s += 1.0 + i as f32;
        }
        for i in 0..90 {
            s += 2.0 + i as f32;
        }
        let m = s * 0.01f32;
        assert_eq!(out[89].1[0], (2.0f32 + 89.0) - m);
        assert_eq!(out[0].1[0], 2.0f32 - m);
    }

    fn assert_in_order_from(out: &[(usize, Frame)], first: usize) {
        for (i, (t, _)) in out.iter().enumerate() {
            assert_eq!(*t, first + i);
        }
    }

    #[test]
    fn silent_after_stats_no_update() {
        let mut frames = vec![(fr(2.0), false); 100];
        frames.push((fr(9.0), true));
        frames.push((fr(9.0), true));
        frames.push((fr(4.0), false));
        let out = run(&frames);
        assert_eq!(out.len(), 103);
        let mut s = 0.0f32;
        for _ in 0..100 {
            s += 2.0f32;
        }
        let m = s * 0.01f32;
        assert_eq!(out[100].1[0], 9.0f32 - m);
        assert_eq!(out[101].1[0], 9.0f32 - m);
        let oma = 1.0f32 - 0.003f32;
        let m2 = (oma * m) + (0.003f32 * 4.0f32);
        assert_eq!(out[102].1[0], 4.0f32 - m2);
    }

    #[test]
    fn finish_flushes_raw() {
        let mut n = ChannelNorm::new();
        let mut out = Vec::new();
        for t in 0..5 {
            n.push(t, &fr(t as f32), false, &mut out);
        }
        n.push(5, &fr(7.0), true, &mut out);
        assert!(out.is_empty());
        n.finish(&mut out);
        assert_eq!(out.len(), 6);
        assert_in_order(&out);
        assert_eq!(out[3].1, fr(3.0));
        assert_eq!(out[5].1, fr(7.0));
        n.finish(&mut out);
        assert_eq!(out.len(), 6);
    }

    #[test]
    fn reset_clears_state() {
        let mut n = ChannelNorm::new();
        let mut out = Vec::new();
        for t in 0..100 {
            n.push(t, &fr(1.0), false, &mut out);
        }
        n.reset();
        out.clear();
        n.push(0, &fr(1.0), false, &mut out);
        assert!(out.is_empty());
    }

    fn pcm_with_zero_run(len: usize, start: usize, zeros: usize) -> Vec<i16> {
        let mut v = vec![1i16; len];
        v[start..start + zeros].fill(0);
        v
    }

    #[test]
    fn silence_tracker_boundaries() {
        let mut t = SilenceTracker::new();
        t.push(&pcm_with_zero_run(400, 100, 199));
        assert!(!t.frame_silent(0));
        let mut t = SilenceTracker::new();
        t.push(&pcm_with_zero_run(400, 100, 200));
        assert!(t.frame_silent(0));
        // not answerable before its samples have arrived
        let mut t = SilenceTracker::new();
        t.push(&vec![0i16; 399]);
        assert!(!t.frame_silent(0));
        t.push(&[0]);
        assert!(t.frame_silent(0));

        // zeros 100..460: frame 0 sees 300 (silent), frame 1 (160..560) sees 300 inside, frame 2 (320..720) sees 140
        let mut t = SilenceTracker::new();
        let mut pcm = vec![1i16; 720];
        pcm[100..460].fill(0);
        t.push(&pcm[..400]);
        assert!(t.frame_silent(0));
        t.push(&pcm[400..560]);
        assert!(t.frame_silent(1));
        t.push(&pcm[560..]);
        assert!(!t.frame_silent(2));
    }

    #[test]
    fn silence_tracker_run_started_in_previous_frame() {
        // zeros 0..300; frame 1 (160..560) only has 140 of them
        let mut pcm = vec![1i16; 560];
        pcm[..300].fill(0);
        let mut t = SilenceTracker::new();
        t.push(&pcm);
        assert!(t.frame_silent(0));
        assert!(!t.frame_silent(1));
    }

    #[test]
    fn silence_tracker_chunking_invariant() {
        let mut pcm = vec![1i16; 3000];
        pcm[500..1100].fill(0);
        let mut a = SilenceTracker::new();
        let mut b = SilenceTracker::new();
        let mut expect = Vec::new();
        let mut got = Vec::new();
        let mut f = 0u64;
        for chunk in pcm.chunks(7) {
            b.push(chunk);
            while 160 * f + 400 <= b.total {
                got.push(b.frame_silent(f));
                f += 1;
            }
        }
        a.push(&pcm[..400]);
        let mut fa = 0;
        let mut pos = 400;
        loop {
            expect.push(a.frame_silent(fa));
            fa += 1;
            if pos + 160 > pcm.len() {
                break;
            }
            a.push(&pcm[pos..pos + 160]);
            pos += 160;
        }
        assert_eq!(got, expect);
        assert!(got.iter().any(|&s| s));
    }
}
