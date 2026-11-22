//! Per-step classifier scores over one class column of a decoder output's lane ring.

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

    /// Max over lane ages `trim_tail+1..=lanes` of the `window`-wide moving average taken in age order.
    pub fn moving_average_max(&self, values: &[&[f32]], t: u64, trim_tail: usize, window: usize) -> f32 {
        let n = self.lanes - trim_tail;
        let mut best = f32::NEG_INFINITY;
        for i in 0..=n - window {
            let mut s = 0.0f32;
            for k in 0..window {
                s += self.get(values, self.lane_of_age(t, trim_tail + 1 + i + k));
            }
            best = best.max(s / window as f32);
        }
        best
    }

    /// The age in `trim_tail+1..=lanes` with the largest value, youngest first on ties.
    pub fn argmax_age(&self, values: &[&[f32]], t: u64, trim_tail: usize) -> usize {
        let mut best = (f32::NEG_INFINITY, trim_tail + 1);
        for a in trim_tail + 1..=self.lanes {
            let v = self.get(values, self.lane_of_age(t, a));
            if v > best.0 {
                best = (v, a);
            }
        }
        best.1
    }

    /// The lane with the largest value, lowest lane index first on ties.
    pub fn argmax_lane(&self, values: &[&[f32]]) -> usize {
        let mut best = (f32::NEG_INFINITY, 0);
        for l in 0..self.lanes {
            let v = self.get(values, l);
            if v > best.0 {
                best = (v, l);
            }
        }
        best.1
    }

    pub fn max(&self, values: &[&[f32]]) -> f32 {
        (0..self.lanes).map(|l| self.get(values, l)).fold(f32::NEG_INFINITY, f32::max)
    }
}

/// A served classifier with its streaming state.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Classifier {
    MaxPool { window: usize, trim_tail: usize },
    Decay { rate: f32, state: f32 },
}

impl Classifier {
    pub fn reset(&mut self) {
        if let Classifier::Decay { state, .. } = self {
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
    fn ages_and_moving_average() {
        let c = col(5, 2, 1);
        // lane l holds class-1 value l+1
        let v: Vec<f32> = (0..5).flat_map(|l| [0.0, (l + 1) as f32]).collect();
        let vals: [&[f32]; 1] = [&v];
        // t=7: age 1 is lane 2, age 2 lane 1, age 3 lane 0, age 4 lane 4, age 5 lane 3
        let by_age: Vec<usize> = (1..=5).map(|a| c.lane_of_age(7, a)).collect();
        assert_eq!(by_age, [2, 1, 0, 4, 3]);
        // ages 2..=5 values 2,1,5,4 -> windows of 2: 1.5, 3, 4.5
        assert_eq!(c.moving_average_max(&vals, 7, 1, 2), 4.5);
        assert_eq!(c.moving_average_max(&vals, 7, 0, 1), 5.0);
        assert_eq!(c.argmax_age(&vals, 7, 0), 4);
        assert_eq!(c.argmax_lane(&vals), 4);
        assert_eq!(c.max(&vals), 5.0);
    }
}
