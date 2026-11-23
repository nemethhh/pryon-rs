//! libpryon's keyword score smoothers (keyword_score_smoother.cpp) over one class column of a decoder output.

use crate::wake::pipeline::ComponentRef;
use crate::Error;
use std::path::Path;

/// One class of a `[lanes, 1, classes]` or `[1, lanes, classes]` output, lane-major either way.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Column {
    pub comp: usize,
    pub lanes: usize,
    pub classes: usize,
    pub class: usize,
}

impl Column {
    pub fn new(comps: &[ComponentRef], key: &str, class: usize, file: &Path) -> Result<Column, Error> {
        let unsupported = |what: String| Error::Unsupported { file: file.to_path_buf(), what };
        let Some(comp) = comps.iter().position(|c| c.key == key) else {
            return Err(unsupported(format!("component {key} is not a stage output")));
        };
        let shape = &comps[comp].shape;
        let lanes = match shape[..] {
            [1, l, c] | [l, 1, c] if l > 0 && c > 0 => l,
            _ => {
                let what = format!("component {key} {shape:?} (implemented: [1, lanes, classes], [lanes, 1, classes])");
                return Err(unsupported(what));
            }
        };
        let classes = shape[2];
        if class >= classes {
            return Err(unsupported(format!("component {key} {shape:?}: class {class} out of range")));
        }
        Ok(Column { comp, lanes, classes, class })
    }

    pub fn get(&self, values: &[&[f32]], lane: usize) -> f32 {
        values[self.comp][lane * self.classes + self.class]
    }

    /// Lane of age `a` at step `t`: age 1 is the lane zeroed before this step.
    pub fn lane_of_age(&self, t: u64, a: usize) -> usize {
        let l = self.lanes;
        ((t % l as u64) as usize + l - (a - 1) % l) % l
    }

    /// Lane behind index `i` of libpryon's score vector, which lists the lanes oldest first.
    pub fn lane_at(&self, t: u64, i: usize) -> usize {
        self.lane_of_age(t, self.lanes - i)
    }

    /// Value at index `i` of libpryon's score vector.
    pub fn at(&self, values: &[&[f32]], t: u64, i: usize) -> f32 {
        self.get(values, self.lane_at(t, i))
    }
}

/// A served smoother with its streaming state.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Smoother {
    /// `moving-average-max-pool`: drops the `trim` last (youngest) scores.
    MaxPool { window: usize, trim: usize },
    /// `exponential-decay` over a single score.
    Decay { rate: f32, state: f32 },
}

impl Smoother {
    /// The step's score and the score-vector index the keyword start is read at.
    pub fn score(&mut self, col: &Column, values: &[&[f32]], t: u64) -> (f32, usize) {
        match self {
            Smoother::MaxPool { window, trim } => {
                let n = col.lanes - *trim;
                // std::max_element: the first maximum
                let (mut arg, mut top) = (0, col.at(values, t, 0));
                for i in 1..n {
                    let x = col.at(values, t, i);
                    if x > top {
                        (arg, top) = (i, x);
                    }
                }
                let mut best = 0.0f32;
                for i in 0..=n - *window {
                    let mut s = 0.0f32;
                    for k in 0..*window {
                        s += col.at(values, t, i + k);
                    }
                    let avg = s / *window as f32;
                    if i == 0 || avg > best {
                        best = avg;
                    }
                }
                (best, arg)
            }
            Smoother::Decay { rate, state } => {
                let x = col.get(values, 0);
                *state = x * (1.0 - *rate) + *rate * *state;
                (*state, 0)
            }
        }
    }

    pub fn reset(&mut self) {
        if let Smoother::Decay { state, .. } = self {
            *state = 0.0;
        }
    }
}

/// One entry of a word component's score history.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Entry {
    pub frame: u64,
    pub score: f32,
    /// Keyword start frame predicted on that frame.
    pub start: u64,
}

/// libpryon's ScoreMaxTracker (score_max_tracker.cpp; used by 0x9afa78): the scores of the last `cap` frames in a
/// ring of `cap` entries, with the index of their maximum.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Tracker {
    ring: Vec<Entry>,
    head: usize,
    len: usize,
    max: usize,
}

impl Tracker {
    pub fn new(cap: usize) -> Tracker {
        Tracker { ring: vec![Entry { frame: 0, score: 0.0, start: 0 }; cap], head: 0, len: 0, max: 0 }
    }

    fn at(&self, i: usize) -> &Entry {
        &self.ring[(self.head + i) % self.ring.len()]
    }

    pub fn push(&mut self, e: Entry) {
        let cap = self.ring.len();
        // compared with the maximum before anything leaves the window
        let takes_over = self.len == 0 || self.at(self.max).score <= e.score;
        let full = self.len == cap;
        let tail = (self.head + self.len) % cap;
        self.ring[tail] = e;
        if full {
            self.head = (self.head + 1) % cap;
        } else {
            self.len += 1;
        }
        let oldest = (e.frame + 1).saturating_sub(cap as u64);
        let mut gone = 0;
        while gone < self.len && self.at(gone).frame < oldest {
            gone += 1;
        }
        self.head = (self.head + gone) % cap;
        self.len -= gone;
        let shift = gone + usize::from(full);
        if takes_over {
            self.max = self.len - 1;
        } else if shift > 0 {
            if self.max >= shift {
                self.max -= shift;
            } else {
                // first maximum (strict <), oldest first
                self.max = 0;
                for i in 1..self.len {
                    if self.at(self.max).score < self.at(i).score {
                        self.max = i;
                    }
                }
            }
        }
    }

    /// The maximum entry; `None` before the first push.
    pub fn max(&self) -> Option<Entry> {
        (self.len > 0).then(|| *self.at(self.max))
    }

    pub fn cap(&self) -> usize {
        self.ring.len()
    }

    pub fn reset(&mut self) {
        (self.head, self.len, self.max) = (0, 0, 0);
    }
}

/// An exponent bionic's powf (FreeBSD msun e_powf.c, Android 7.1) treats as an exact special case.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Exp {
    Zero,
    Half,
    One,
    Two,
}

impl Exp {
    pub fn of(y: f32) -> Option<Exp> {
        match y {
            _ if y == 0.0 => Some(Exp::Zero),
            0.5 => Some(Exp::Half),
            1.0 => Some(Exp::One),
            2.0 => Some(Exp::Two),
            _ => None,
        }
    }

    /// msun's powf(x, y) in its check order.
    pub fn powf(self, x: f32) -> f32 {
        match self {
            Exp::Zero => 1.0,
            _ if x == 1.0 => 1.0,
            _ if x.is_nan() => x,
            Exp::One => x,
            Exp::Two => x * x,
            Exp::Half if x.to_bits() >> 31 == 0 => x.sqrt(),
            // -0 and -inf take the |x| path; other negatives are NaN for a non-integer exponent
            Exp::Half if x == 0.0 || x == f32::NEG_INFINITY => -x,
            Exp::Half => f32::NAN,
        }
    }
}

/// WeightedGeometricAverageAggregator exponents: per weight, then 1 / (0 + w_0 + ...); `None` unless all are exact.
pub(crate) fn aggregator(weights: &[f32]) -> Option<(Vec<Exp>, Exp)> {
    let sum = weights.iter().fold(0.0f32, |a, &w| a + w);
    let exps = weights.iter().map(|&w| Exp::of(w)).collect::<Option<Vec<_>>>()?;
    Some((exps, Exp::of(1.0 / sum)?))
}

/// One word component of a multi-word spotter (0x9afa78): its class score through an exponential decay, then the
/// maximum over its history.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Word {
    pub col: Column,
    pub se: Column,
    /// The keyword-start-end-component's end column.
    pub se_end: Column,
    pub decay: Smoother,
    pub track: Tracker,
}

/// WeightedGeometricAverageAggregator::score (0x9ab474): prod(powf(s_i, w_i)) from 1.0, then powf(prod, 1 / sum).
pub(crate) fn aggregate(scores: &[f32], exps: &[Exp], inv: Exp) -> f32 {
    let prod = scores.iter().zip(exps).fold(1.0f32, |p, (&s, e)| p * e.powf(s));
    inv.powf(prod)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn col(lanes: usize, classes: usize, class: usize) -> Column {
        Column { comp: 0, lanes, classes, class }
    }

    #[test]
    fn layouts() {
        let r = |shape: Vec<usize>| ComponentRef { key: "s/c".into(), stage: 0, output: 0, shape };
        let f = Path::new("g.onnx");
        assert_eq!(Column::new(&[r(vec![25, 1, 3])], "s/c", 1, f).unwrap(), col(25, 3, 1));
        assert_eq!(Column::new(&[r(vec![1, 16, 3])], "s/c", 2, f).unwrap(), col(16, 3, 2));
        assert_eq!(Column::new(&[r(vec![1, 1, 4])], "s/c", 3, f).unwrap(), col(1, 4, 3));
        for (shape, class) in [(vec![16, 2, 3], 1), (vec![1, 16], 1), (vec![1, 16, 3], 3), (vec![1, 0, 3], 0)] {
            match Column::new(&[r(shape.clone())], "s/c", class, f) {
                Err(Error::Unsupported { what, .. }) => assert!(what.contains("s/c"), "{shape:?}: {what}"),
                other => panic!("{shape:?}: {other:?}"),
            }
        }
        assert!(Column::new(&[r(vec![1, 16, 3])], "s/x", 1, f).is_err());
    }

    #[test]
    fn ages_oldest_first() {
        let c = col(5, 2, 1);
        // t=7: age 1 is lane 2, age 2 lane 1, age 3 lane 0, age 4 lane 4, age 5 lane 3
        let by_age: Vec<usize> = (1..=5).map(|a| c.lane_of_age(7, a)).collect();
        assert_eq!(by_age, [2, 1, 0, 4, 3]);
        let by_index: Vec<usize> = (0..5).map(|i| c.lane_at(7, i)).collect();
        assert_eq!(by_index, [3, 4, 0, 1, 2]);
    }

    /// Lane `l` holds class-1 value `vals[l]`.
    fn lanes(vals: &[f32]) -> Vec<f32> {
        vals.iter().flat_map(|&v| [0.0, v]).collect()
    }

    #[test]
    fn maxpool_ages_from_trim_tail() {
        let c = col(5, 2, 1);
        // t=7, oldest first: lanes 3, 4, 0, 1, 2 -> values 4, 5, 1, 2, 3; trim 2 keeps 4, 5, 1
        let v = lanes(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let vals: [&[f32]; 1] = [&v];
        let mut m = Smoother::MaxPool { window: 2, trim: 2 };
        assert_eq!(m.score(&c, &vals, 7), (4.5, 1));
        let mut m = Smoother::MaxPool { window: 1, trim: 0 };
        assert_eq!(m.score(&c, &vals, 7), (5.0, 1));
        let mut m = Smoother::MaxPool { window: 3, trim: 0 };
        assert_eq!(m.score(&c, &vals, 7).0, 10.0 / 3.0);
        // the youngest score, however high, is trimmed
        let v = lanes(&[1.0, 2.0, 9.0, 4.0, 5.0]);
        let vals: [&[f32]; 1] = [&v];
        let mut m = Smoother::MaxPool { window: 1, trim: 1 };
        assert_eq!(m.score(&c, &vals, 7), (5.0, 1));
    }

    #[test]
    fn maxpool_first_maximum_and_sum_order() {
        let c = col(4, 2, 1);
        // t=3: oldest first is lanes 0, 1, 2, 3
        let v = lanes(&[0.5, 0.7, 0.7, 0.2]);
        let vals: [&[f32]; 1] = [&v];
        let mut m = Smoother::MaxPool { window: 1, trim: 0 };
        assert_eq!(m.score(&c, &vals, 3), (0.7, 1), "ties go to the oldest");
        // summed oldest first: (1 + e) + e rounds back to 1, (e + e) + 1 does not
        let e = f32::EPSILON / 2.0;
        let v = lanes(&[1.0, e, e, 0.0]);
        let vals: [&[f32]; 1] = [&v];
        let mut m = Smoother::MaxPool { window: 3, trim: 1 };
        let s = m.score(&c, &vals, 3).0;
        assert_eq!(s, 1.0 / 3.0);
        assert_ne!(s, (e + e + 1.0) / 3.0);
    }

    fn e(frame: u64, score: f32) -> Entry {
        Entry { frame, score, start: frame * 10 }
    }

    #[test]
    fn aggregator_values() {
        // a 20-frame history fed every 8 frames keeps three entries: frames >= f - 19
        let mut t = Tracker::new(20);
        assert_eq!(t.max(), None);
        let seq = [(8, 0.5), (16, 0.7), (24, 0.6), (32, 0.4), (40, 0.1), (48, 0.6), (56, 0.6), (64, 0.3)];
        let want = [(8, 0.5), (16, 0.7), (16, 0.7), (16, 0.7), (24, 0.6), (48, 0.6), (56, 0.6), (56, 0.6)];
        for (&(f, s), &(wf, ws)) in seq.iter().zip(&want) {
            t.push(e(f, s));
            assert_eq!(t.max(), Some(e(wf, ws)), "after frame {f}");
        }
        // a new score equal to the maximum takes over; a rescan keeps the oldest of equal maxima
        let mut t = Tracker::new(20);
        for (f, s) in [(8, 0.6), (16, 0.6), (24, 0.2), (32, 0.5)] {
            t.push(e(f, s));
        }
        assert_eq!(t.max(), Some(e(16, 0.6)));
        t.push(e(40, 0.2));
        assert_eq!(t.max(), Some(e(32, 0.5)), "16 left the window");
        // ring capacity binds when frames are denser than the ring: entries 1, 2 are overwritten by 3
        let mut t = Tracker::new(2);
        for (f, s) in [(1, 0.9), (2, 0.4), (3, 0.3)] {
            t.push(e(f, s));
        }
        assert_eq!(t.max(), Some(e(2, 0.4)));
        t.reset();
        assert_eq!(t.max(), None);
        t.push(e(9, 0.1));
        assert_eq!(t.max(), Some(e(9, 0.1)));
        // history 1: the current score only
        let mut t = Tracker::new(1);
        t.push(e(8, 0.9));
        t.push(e(16, 0.2));
        assert_eq!(t.max(), Some(e(16, 0.2)));

        // HEY_DISNEY's two components: decay 0.0 with history, decay 0.3 without; weights 1, 1
        let c = col(1, 4, 2);
        let mut d0 = Smoother::Decay { rate: 0.0, state: 0.0 };
        let mut d1 = Smoother::Decay { rate: 0.3, state: 0.0 };
        let mut h0 = Tracker::new(100);
        let inputs = [(0.2f32, 0.1f32), (0.6, 0.5), (0.3, 0.7)];
        let mut got = Vec::new();
        for (k, &(a, b)) in inputs.iter().enumerate() {
            let v = [0.0, 0.0, a, b];
            let vals: [&[f32]; 1] = [&v];
            let s0 = d0.score(&c, &vals, k as u64).0;
            let c1 = Column { class: 3, ..c };
            let s1 = d1.score(&c1, &vals, k as u64).0;
            h0.push(Entry { frame: 8 * (k as u64 + 1), score: s0, start: 1 });
            let (exps, inv) = aggregator(&[1.0, 1.0]).unwrap();
            got.push(aggregate(&[h0.max().unwrap().score, s1], &exps, inv));
        }
        let (mut x1, mut m0) = (0.0f32, 0.0f32);
        let want: Vec<f32> = inputs
            .iter()
            .map(|&(a, b)| {
                x1 = b * (1.0 - 0.3f32) + 0.3 * x1;
                m0 = if a > m0 { a } else { m0 };
                (m0 * x1).sqrt()
            })
            .collect();
        assert_eq!(got, want);
        assert_eq!(got[2], (0.6f32 * (0.7 * 0.7 + 0.3 * (0.5 * 0.7 + 0.3 * (0.1 * 0.7)))).sqrt());
    }

    #[test]
    fn powf_special_cases() {
        let x = 0.37f32;
        let p = |y: f32, x: f32| Exp::of(y).unwrap().powf(x);
        assert_eq!((p(1.0, x), p(2.0, x), p(0.5, x), p(0.0, x)), (x, x * x, x.sqrt(), 1.0));
        assert_eq!((Exp::of(0.25), Exp::of(-1.0)), (None, None));
        // msun beyond the scores' range: -0 -> +0, -inf -> +inf, NaN in -> NaN out, other negatives -> NaN
        assert_eq!(p(0.5, -0.0).to_bits(), 0);
        assert_eq!(p(0.5, f32::NEG_INFINITY), f32::INFINITY);
        assert!(p(0.5, f32::NAN).is_nan() && p(1.0, f32::NAN).is_nan() && p(0.5, -0.5).is_nan());
        assert_eq!((p(0.0, f32::NAN), p(0.5, 1.0)), (1.0, 1.0));
        let (exps, inv) = aggregator(&[1.0, 1.0]).unwrap();
        assert_eq!(aggregate(&[0.25, 0.64], &exps, inv), 0.4);
        let (exps, inv) = aggregator(&[1.0]).unwrap();
        assert_eq!(aggregate(&[0.3], &exps, inv), 0.3);
        assert_eq!(aggregator(&[2.0, 2.0]), None, "1 / 4 is not a special case");
        assert!(aggregator(&[2.0, 0.0]).is_some());
    }

    #[test]
    fn decay_update_and_reset() {
        let c = col(1, 2, 1);
        let mut d = Smoother::Decay { rate: 0.6, state: 0.0 };
        let v = lanes(&[0.99]);
        let vals: [&[f32]; 1] = [&v];
        let s1 = d.score(&c, &vals, 0).0;
        assert_eq!(s1, 0.99 * (1.0 - 0.6f32));
        let s2 = d.score(&c, &vals, 1).0;
        assert_eq!(s2, 0.99 * (1.0 - 0.6f32) + 0.6 * s1);
        d.reset();
        assert_eq!(d.score(&c, &vals, 2).0, s1);
    }
}
