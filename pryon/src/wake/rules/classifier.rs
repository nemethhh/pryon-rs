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
