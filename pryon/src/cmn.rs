//! Per-window channel norm with digital-silence handling (libpryon parity), released in frame order.
use std::collections::VecDeque;

const BASE: usize = 100;
const DECAY: f32 = 0.003;
const FRAME: usize = 400;
const HOP: usize = 160;
const SILENT_ZEROS: usize = 200;

type Frame = [f32; 20];

pub struct ChannelNorm {
    sum: Frame,
    count: usize,
    mean: Option<Frame>,
    // Non-silent frames before stats, unreleased: (t, x).
    pending: Vec<(usize, Frame)>,
}

impl Default for ChannelNorm {
    fn default() -> Self {
        Self::new()
    }
}

impl ChannelNorm {
    pub fn new() -> ChannelNorm {
        ChannelNorm { sum: [0.0; 20], count: 0, mean: None, pending: Vec::new() }
    }

    pub fn reset(&mut self) {
        self.sum = [0.0; 20];
        self.count = 0;
        self.mean = None;
        self.pending.clear();
    }

    /// t: frame index within the window; silent: digital-silence flag. Pushes (t, normalised) in increasing t
    /// as soon as ready (may push 0, 1 or many).
    pub fn push(&mut self, t: usize, x: &Frame, silent: bool, out: &mut Vec<(usize, Frame)>) {
        if silent {
            match &self.mean {
                Some(m) => out.push((t, sub(x, m))),
                // libpryon: a silent frame before stats releases the queue raw (the flushed frames still count toward the 100-frame stats)
                None => {
                    self.flush_raw(out);
                    out.push((t, *x));
                }
            }
            return;
        }
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
        self.pending.push((t, *x));
        if self.count == BASE {
            let mut m = self.sum;
            for v in &mut m {
                *v *= 0.01f32;
            }
            for (pt, px) in self.pending.drain(..) {
                out.push((pt, sub(&px, &m)));
            }
            self.mean = Some(m);
        }
    }

    /// end of window: flush queued frames raw (stats never ready).
    pub fn finish(&mut self, out: &mut Vec<(usize, Frame)>) {
        self.flush_raw(out);
    }

    fn flush_raw(&mut self, out: &mut Vec<(usize, Frame)>) {
        out.extend(self.pending.drain(..));
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
    // Zero flags of the last FRAME samples.
    zeros: VecDeque<bool>,
}

impl Default for SilenceTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl SilenceTracker {
    pub fn new() -> Self {
        SilenceTracker { total: 0, zeros: VecDeque::with_capacity(FRAME + 1) }
    }

    /// Appends one verdict per newly completed frame (frame f completes when sample 160f+399 has been pushed), in
    /// frame order, for any chunk size. A frame is silent iff it has >= 200 contiguous zero samples inside its own
    /// 400 samples.
    pub fn push(&mut self, pcm: &[i16], out: &mut Vec<bool>) {
        for &s in pcm {
            if self.zeros.len() == FRAME {
                self.zeros.pop_front();
            }
            self.zeros.push_back(s == 0);
            self.total += 1;
            if self.total >= FRAME as u64 && (self.total - FRAME as u64) % HOP as u64 == 0 {
                out.push(self.window_silent());
            }
        }
    }

    fn window_silent(&self) -> bool {
        let mut run = 0;
        for &z in &self.zeros {
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
    fn silent_before_stats_flushes_queue_raw() {
        let mut n = ChannelNorm::new();
        let mut out = Vec::new();
        for t in 0..10 {
            n.push(t, &fr(1.0 + t as f32), false, &mut out);
        }
        assert!(out.is_empty());
        n.push(10, &fr(-9.0), true, &mut out);
        assert_eq!(out.len(), 11);
        assert_in_order(&out);
        for (t, y) in &out[..10] {
            assert_eq!(*y, fr(1.0 + *t as f32));
        }
        assert_eq!(out[10].1, fr(-9.0));
    }

    #[test]
    fn silent_run_flushes_on_its_first_frame() {
        let mut n = ChannelNorm::new();
        let mut out = Vec::new();
        for t in 0..3 {
            n.push(t, &fr(t as f32), false, &mut out);
        }
        n.push(3, &fr(-1.0), true, &mut out);
        assert_eq!(out.len(), 4);
        // later silent frames, queue empty: raw at once
        for t in 4..6 {
            out.clear();
            n.push(t, &fr(-2.0), true, &mut out);
            assert_eq!(out, vec![(t, fr(-2.0))]);
        }
    }

    #[test]
    fn flush_keeps_stats_sum() {
        // non-silent frames flushed raw still count toward the 100-frame stats
        let mut n = ChannelNorm::new();
        let mut out = Vec::new();
        for t in 0..10 {
            n.push(t, &fr(1.0 + t as f32), false, &mut out);
        }
        n.push(10, &fr(-9.0), true, &mut out);
        out.clear();
        for i in 0..89 {
            n.push(11 + i, &fr(0.5 + i as f32), false, &mut out);
        }
        assert!(out.is_empty());
        n.push(100, &fr(0.5 + 89.0), false, &mut out);
        assert_eq!(out.len(), 90);
        assert_in_order_from(&out, 11);
        let mut s = 0.0f32;
        for i in 0..10 {
            s += 1.0 + i as f32;
        }
        for i in 0..90 {
            s += 0.5 + i as f32;
        }
        let m = s * 0.01f32;
        assert_eq!(out[0].1[0], 0.5f32 - m);
        assert_eq!(out[89].1[0], (0.5f32 + 89.0) - m);
    }

    #[test]
    fn silent_between_queued_frames_flushes_each_time() {
        let mut n = ChannelNorm::new();
        let mut out = Vec::new();
        n.push(0, &fr(1.0), false, &mut out);
        n.push(1, &fr(-1.0), true, &mut out);
        n.push(2, &fr(2.0), false, &mut out);
        assert_eq!(out, vec![(0, fr(1.0)), (1, fr(-1.0))]);
        n.push(3, &fr(-3.0), true, &mut out);
        assert_eq!(out.len(), 4);
        assert_eq!(out[2], (2, fr(2.0)));
        assert_eq!(out[3], (3, fr(-3.0)));
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
        for t in 0..6 {
            n.push(t, &fr(t as f32), false, &mut out);
        }
        assert!(out.is_empty());
        n.finish(&mut out);
        assert_eq!(out.len(), 6);
        assert_in_order(&out);
        assert_eq!(out[3].1, fr(3.0));
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

    fn verdicts(pcm: &[i16], chunk: usize) -> Vec<bool> {
        let mut t = SilenceTracker::new();
        let mut out = Vec::new();
        for c in pcm.chunks(chunk) {
            t.push(c, &mut out);
        }
        out
    }

    fn brute(pcm: &[i16]) -> Vec<bool> {
        let mut out = Vec::new();
        let mut f = 0;
        while 160 * f + 400 <= pcm.len() {
            let (mut run, mut best) = (0, 0);
            for &s in &pcm[160 * f..160 * f + 400] {
                run = if s == 0 { run + 1 } else { 0 };
                best = best.max(run);
            }
            out.push(best >= 200);
            f += 1;
        }
        out
    }

    #[test]
    fn silence_tracker_boundaries() {
        assert_eq!(verdicts(&pcm_with_zero_run(400, 100, 199), 400), vec![false]);
        assert_eq!(verdicts(&pcm_with_zero_run(400, 100, 200), 400), vec![true]);
        // nothing is reported before the 400th sample
        assert!(verdicts(&vec![0i16; 399], 399).is_empty());
        assert_eq!(verdicts(&vec![0i16; 400], 1), vec![true]);
    }

    #[test]
    fn silence_tracker_run_started_in_previous_frame() {
        // zeros 100..460: frame 0 sees 300, frame 1 (160..560) sees 300, frame 2 (320..720) only 140
        let mut pcm = vec![1i16; 720];
        pcm[100..460].fill(0);
        assert_eq!(verdicts(&pcm, 720), vec![true, true, false]);
        // zeros 0..300: frame 1 (160..560) only has 140 of them
        let mut pcm = vec![1i16; 560];
        pcm[..300].fill(0);
        assert_eq!(verdicts(&pcm, 560), vec![true, false]);
    }

    #[test]
    fn silence_tracker_chunking_invariant() {
        let mut pcm = vec![1i16; 20_000];
        pcm[500..1100].fill(0);
        pcm[3000..3199].fill(0);
        pcm[9000..9200].fill(0);
        pcm[12_000..16_000].fill(0);
        let want = brute(&pcm);
        assert!(want.iter().any(|&s| s) && want.iter().any(|&s| !s));
        for chunk in [1, 159, 401, 5000, 20_000] {
            assert_eq!(verdicts(&pcm, chunk), want, "chunk {chunk}");
        }
    }
}
