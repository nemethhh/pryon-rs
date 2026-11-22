//! Keyword orchestra: per-step decoder outputs to Accept / AcceptAuxiliary / NearMiss detections.
//!
//! Rules v0 are pypryon's (decoder.py for single-graph sets, rules.py for DAVS) driven by the kw.cfg fields; Task 11
//! replaces the fitted parts with libpryon's.

mod classifier;

use super::kwcfg::{ClassifierSpec, SpotterSpec};
use super::pipeline::{LoadedSet, StepOut};
use super::{Detection, DetectionKind};
use crate::Error;
use classifier::{Classifier, Column};
use std::path::Path;

/// Samples per LFBE frame (pinned 10 ms shift at 16 kHz).
const FRAME_SAMPLES: i64 = 160;
// fitted: pypryon
const START_CYCLE: u64 = 5;
// fitted: pypryon
const SE_START_COLUMN: usize = 1;
// fitted: pypryon
const NEAR_MISS_WINDOW_STEPS: u64 = 8;

/// One spotter's score source and thresholds.
#[derive(Debug)]
struct Rule {
    col: Column,
    classifier: Classifier,
    accept: f32,
    notify: Option<f32>,
    /// (property index, equals, accept).
    overrides: Vec<(usize, i32, f32)>,
}

impl Rule {
    fn threshold(&self, props: &[(String, i32)]) -> f32 {
        self.overrides.iter().find(|&&(p, eq, _)| props[p].1 == eq).map_or(self.accept, |o| o.2)
    }

    /// Step score and the step its window starts at (DAVS).
    fn score_davs(&mut self, v: &[&[f32]], t: u64) -> (f32, i64) {
        match &mut self.classifier {
            Classifier::Decay { rate, state } => {
                *state = *rate * *state + (1.0 - *rate) * self.col.max(v);
                (*state, t as i64)
            }
            Classifier::MaxPool { window, trim_tail } => {
                let s = self.col.moving_average_max(v, t, *trim_tail, *window);
                let lane = self.col.argmax_lane(v) as i64;
                let age = (t as i64 - lane).rem_euclid(self.col.lanes as i64);
                (s, t as i64 - age)
            }
        }
    }
}

/// Near-miss candidate: (start step, peak score, peak step, begin frame).
type Run = (u64, f32, u64, i64);

/// pypryon's rule set per layout.
// fitted: pypryon
#[derive(Debug)]
enum Flavor {
    /// Single-graph (firmware) rules; begin from the start/end component's ring.
    SingleGraph { se: Column, trim_tail: usize },
    /// DAVS rules with an optional verifier; `pending` is a near miss waiting for its emission frame.
    Davs { aux: Option<Rule>, pending: Option<(f32, i64, i64)> },
}

#[derive(Debug)]
struct Keyword {
    index: usize,
    rule: Rule,
    flavor: Flavor,
    lock: i64,
    kw_window: i64,
    /// Frames since reset; nothing accepts at or before it.
    lock_until: i64,
    run: Option<Run>,
}

pub(crate) struct Orchestra {
    keywords: Vec<Keyword>,
    props: Vec<(String, i32)>,
    /// LFBE frames per step.
    fps: i64,
}

/// Served keywords: words.shrunk.txt order, each with a primary spotter, once.
pub(crate) fn keyword_names(set: &LoadedSet) -> Vec<String> {
    let sp = &set.spec.spotters;
    let mut out: Vec<String> = Vec::new();
    for (_, w) in &set.spec.words {
        if sp.iter().any(|s| !s.auxiliary && &s.name == w) && !out.contains(w) {
            out.push(w.clone());
        }
    }
    out
}

fn unsupported(file: &Path, what: impl Into<String>) -> Error {
    Error::Unsupported { file: file.to_path_buf(), what: what.into() }
}

impl Orchestra {
    pub fn new(set: &LoadedSet) -> Result<Orchestra, Error> {
        let spec = &set.spec;
        let dir = spec.dir.as_path();
        let names = keyword_names(set);
        let mut props: Vec<(String, i32)> = Vec::new();
        for s in &spec.spotters {
            for o in &s.thresholds.overrides {
                if !props.iter().any(|(n, _)| n == &o.prop) {
                    props.push((o.prop.clone(), 0));
                }
            }
        }
        let stage = |s: &SpotterSpec| {
            spec.stages.iter().position(|st| st.name == s.scorer).ok_or_else(|| {
                unsupported(dir, format!("spotter {} reads scorer {}, which is not a stage", s.name, s.scorer))
            })
        };
        let column = |s: &SpotterSpec, comp: &str, class: usize| {
            let st = stage(s)?;
            Column::new(&set.components, &format!("{}/{comp}", s.scorer), class, &spec.stages[st].graph)
        };
        // every component any spotter names must have a layout the rules can index
        for s in &spec.spotters {
            for c in &s.components {
                column(s, &c.component, c.class)?;
                if let Some(se) = &c.start_end {
                    column(s, se, c.start_index.max(c.end_index))?;
                }
            }
        }
        let rule = |s: &SpotterSpec, single: bool| -> Result<Option<Rule>, Error> {
            let what = |w: String| unsupported(dir, format!("spotter {}: {w}", s.name));
            let classifier = match (&s.classifier, single) {
                (ClassifierSpec::WeightedGeometricAverage { .. }, false) => return Ok(None),
                (&ClassifierSpec::MovingAverageMaxPool { window, trim_tail }, true) => {
                    Classifier::MaxPool { window, trim_tail }
                }
                (&ClassifierSpec::MovingAverageMaxPool { window: 1, trim_tail: 0 }, false) => {
                    Classifier::MaxPool { window: 1, trim_tail: 0 }
                }
                (&ClassifierSpec::ExponentialDecay { rate }, false) => Classifier::Decay { rate, state: 0.0 },
                (c, _) => {
                    let layout = if single { "single-graph" } else { "DAVS" };
                    return Err(what(format!("classifier {c:?} on a {layout} set (not implemented)")));
                }
            };
            let [c] = &s.components[..] else {
                return Err(what(format!("{} word components with a single-component classifier", s.components.len())));
            };
            if !s.smoothers.is_empty() {
                return Err(what("posterior-smoothers on a single-component classifier (not implemented)".into()));
            }
            let col = column(s, &c.component, c.class)?;
            if let Classifier::MaxPool { window, trim_tail } = classifier {
                if window == 0 || trim_tail >= col.lanes || window > col.lanes - trim_tail {
                    return Err(what(format!("window-size {window}, trim-tail {trim_tail} over {} lanes", col.lanes)));
                }
            }
            let t = &s.thresholds;
            let overrides = t
                .overrides
                .iter()
                .map(|o| (props.iter().position(|(n, _)| n == &o.prop).unwrap(), o.equals, o.accept))
                .collect();
            Ok(Some(Rule { col, classifier, accept: t.accept, notify: t.notify, overrides }))
        };
        let mut keywords = Vec::new();
        for (index, name) in names.iter().enumerate() {
            let find = |aux: bool| spec.spotters.iter().filter(move |s| &s.name == name && s.auxiliary == aux);
            let (Some(p), None) = (find(false).next(), find(false).nth(1)) else {
                return Err(unsupported(dir, format!("keyword {name}: more than one primary spotter")));
            };
            let aux = find(true).next();
            if find(true).nth(1).is_some() {
                return Err(unsupported(dir, format!("keyword {name}: more than one auxiliary spotter")));
            }
            // fitted: pypryon
            let single = stage(p)? == 0;
            let Some(r) = rule(p, single)? else { continue };
            let flavor = if single {
                if aux.is_some() {
                    return Err(unsupported(dir, format!("keyword {name}: verifier on a single-graph set")));
                }
                let c = &p.components[0];
                let Some(se) = &c.start_end else {
                    return Err(unsupported(dir, format!("spotter {name}: no keyword-start-end-component")));
                };
                let se = column(p, se, SE_START_COLUMN)?;
                if se.lanes != r.col.lanes {
                    return Err(unsupported(dir, format!("spotter {name}: start/end lanes differ from score lanes")));
                }
                let Classifier::MaxPool { trim_tail, .. } = r.classifier else { unreachable!() };
                Flavor::SingleGraph { se, trim_tail }
            } else {
                let aux = match aux {
                    Some(a) => Some(rule(a, false)?.ok_or_else(|| {
                        unsupported(dir, format!("keyword {name}: verifier with an aggregating classifier"))
                    })?),
                    None => None,
                };
                Flavor::Davs { aux, pending: None }
            };
            let lock = i64::from(p.lock);
            let kw_window = i64::from(p.kw_window);
            keywords.push(Keyword { index, rule: r, flavor, lock, kw_window, lock_until: -1, run: None });
        }
        let fps = spec.stages[0].frames_per_step as i64;
        Ok(Orchestra { keywords, props, fps })
    }

    pub fn reset(&mut self) {
        for k in &mut self.keywords {
            k.lock_until = -1;
            k.run = None;
            k.rule.classifier.reset();
            if let Flavor::Davs { aux, pending } = &mut k.flavor {
                *pending = None;
                if let Some(a) = aux {
                    a.classifier.reset();
                }
            }
        }
    }

    /// Unknown names are ignored.
    pub fn set_property(&mut self, name: &str, value: i32) {
        if let Some(p) = self.props.iter_mut().find(|(n, _)| n == name) {
            p.1 = value;
        }
    }

    /// Detections completed by this step; samples count from the pipeline's reset.
    pub fn step(&mut self, s: &StepOut, out: &mut Vec<Detection>) {
        let at = At { t: s.step, end: s.frames as i64, fps: self.fps, props: &self.props };
        for k in &mut self.keywords {
            match k.flavor {
                Flavor::SingleGraph { .. } => k.step_single(s.values, &at, out),
                Flavor::Davs { .. } => k.step_davs(s.values, &at, out),
            }
        }
    }
}

/// Where a step lands: its index, the frames since reset at its end, frames per step, client properties.
#[derive(Clone, Copy)]
struct At<'a> {
    t: u64,
    end: i64,
    fps: i64,
    props: &'a [(String, i32)],
}

fn detection(keyword: usize, kind: DetectionKind, score: f32, begin: i64, end: i64) -> Detection {
    Detection {
        keyword,
        kind,
        score,
        begin_sample: (begin.max(0) * FRAME_SAMPLES) as u64,
        end_sample: (end * FRAME_SAMPLES) as u64,
        afe: None,
    }
}

impl Keyword {
    /// pypryon decoder.py `Orchestra._step`; frames throughout.
    fn step_single(&mut self, v: &[&[f32]], at: &At, out: &mut Vec<Detection>) {
        let At { t, end, fps, props } = *at;
        let Flavor::SingleGraph { se, trim_tail } = self.flavor else { unreachable!() };
        if t < START_CYCLE {
            return;
        }
        let Classifier::MaxPool { window, .. } = self.rule.classifier else { unreachable!() };
        let s = self.rule.col.moving_average_max(v, t, trim_tail, window);
        let begin = || fps * (t as i64 - se.argmax_age(v, t, trim_tail) as i64 + 1) + 1;
        if end > self.lock_until && s >= self.rule.threshold(props) {
            out.push(detection(self.index, DetectionKind::Accept, s, begin(), end));
            self.lock_until = end + self.lock;
            self.run = None;
            return;
        }
        if self.rule.notify.is_some_and(|n| s >= n) && end > self.lock_until {
            if self.run.is_none_or(|r| s > r.1) {
                self.run = Some((t, s, t, begin()));
            }
        } else if let Some((_, peak, peak_t, b)) = self.run {
            let nm_end = fps * (peak_t as i64 + 1) + self.kw_window;
            if nm_end <= end {
                out.push(detection(self.index, DetectionKind::NearMiss, peak, b, nm_end));
                self.run = None;
            }
        }
    }

    /// pypryon rules.py `DavsOrchestra._step_keyword`; frames throughout.
    fn step_davs(&mut self, v: &[&[f32]], at: &At, out: &mut Vec<Detection>) {
        let At { t, end, fps, props } = *at;
        let Flavor::Davs { aux, pending } = &mut self.flavor else { unreachable!() };
        let (score, start) = self.rule.score_davs(v, t);
        let aux_score = aux.as_mut().map(|a| a.score_davs(v, t));
        if let Some((s, b, nm_end)) = *pending {
            if end >= nm_end {
                out.push(detection(self.index, DetectionKind::NearMiss, s, b, nm_end));
                *pending = None;
            }
        }
        if end > self.lock_until {
            // fitted: pypryon (the verifier only when the primary does not accept)
            let fired = if score >= self.rule.threshold(props) {
                Some((DetectionKind::Accept, score, start))
            } else {
                match (aux.as_ref(), aux_score) {
                    (Some(a), Some((s, st))) if s >= a.threshold(props) => {
                        Some((DetectionKind::AcceptAuxiliary, s, st))
                    }
                    _ => None,
                }
            };
            if let Some((kind, s, st)) = fired {
                out.push(detection(self.index, kind, s, st * fps, end));
                self.lock_until = end + self.lock;
                self.run = None;
                *pending = None;
                return;
            }
        }
        let Some(notify) = self.rule.notify else { return };
        if let Some((run_start, peak, _, _)) = &mut self.run {
            let run_start = *run_start;
            if score > *peak {
                self.run = Some((run_start, score, t, start * fps));
            }
            if t + 1 >= run_start + NEAR_MISS_WINDOW_STEPS {
                let (_, peak, peak_t, b) = self.run.take().unwrap();
                // fitted: pypryon (emitted a lock period after the peak step's end)
                *pending = Some((peak, b, fps * (peak_t as i64 + 1) + self.lock));
            }
        } else if pending.is_none() && end > self.lock_until && score >= notify {
            self.run = Some((t, score, t, start * fps));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A DAVS keyword over one `[1, lanes, 2]` output with lane values set per step.
    fn davs_kw(lanes: usize, accept: f32, notify: Option<f32>, lock: i64) -> Keyword {
        let col = Column { comp: 0, lanes, classes: 2, class: 1 };
        let classifier = Classifier::MaxPool { window: 1, trim_tail: 0 };
        let rule = Rule { col, classifier, accept, notify, overrides: vec![] };
        let flavor = Flavor::Davs { aux: None, pending: None };
        Keyword { index: 0, rule, flavor, lock, kw_window: 100, lock_until: -1, run: None }
    }

    fn feed_davs(k: &mut Keyword, scores: &[f32], lanes: usize) -> Vec<Detection> {
        let mut out = Vec::new();
        for (t, &s) in scores.iter().enumerate() {
            let mut v = vec![0.0f32; lanes * 2];
            v[(t % lanes) * 2 + 1] = s;
            let vals: [&[f32]; 1] = [&v];
            k.step_davs(&vals, &At { t: t as u64, end: 8 * (t as i64 + 1), fps: 8, props: &[] }, &mut out);
        }
        out
    }

    #[test]
    fn davs_accept_lock_and_begin() {
        let mut k = davs_kw(4, 0.9, None, 20);
        let out = feed_davs(&mut k, &[0.1, 0.95, 0.95, 0.95, 0.1, 0.1], 4);
        // step 1 accepts (window starts at step 1); lock 20 frames from frame 16 holds through frame 36
        assert_eq!(out.len(), 1);
        assert_eq!((out[0].kind, out[0].begin_sample, out[0].end_sample), (DetectionKind::Accept, 8 * 160, 16 * 160));
        let mut k = davs_kw(4, 0.9, None, 20);
        let out = feed_davs(&mut k, &[0.1, 0.95, 0.1, 0.1, 0.1, 0.95], 4);
        assert_eq!(out.iter().map(|d| d.end_sample / 160).collect::<Vec<_>>(), [16, 48]);
    }

    #[test]
    fn davs_near_miss_window() {
        let mut k = davs_kw(16, 0.9, Some(0.3), 10);
        let mut s = vec![0.0f32; 20];
        s[2] = 0.4;
        s[5] = 0.6;
        s[10] = 0.8;
        let out = feed_davs(&mut k, &s, 16);
        let got: Vec<_> = out.iter().map(|d| (d.kind, d.score, d.begin_sample / 160, d.end_sample / 160)).collect();
        // run 1 (steps 2..=9, peak 5) is emitted at step 10, which starts run 2; end = (peak+1)*8 + lock
        let nm = DetectionKind::NearMiss;
        assert_eq!(got, [(nm, 0.6, 5 * 8, 58), (nm, 0.8, 10 * 8, 98)]);
    }

    #[test]
    fn single_graph_near_miss_after_feature_window() {
        let lanes = 25;
        let col = Column { comp: 0, lanes, classes: 2, class: 1 };
        let se = Column { comp: 1, lanes, classes: 2, class: SE_START_COLUMN };
        let classifier = Classifier::MaxPool { window: 1, trim_tail: 0 };
        let rule = Rule { col, classifier, accept: 0.9, notify: Some(0.5), overrides: vec![] };
        let flavor = Flavor::SingleGraph { se, trim_tail: 0 };
        let mut k = Keyword { index: 1, rule, flavor, lock: 100, kw_window: 30, lock_until: -1, run: None };
        let mut out = Vec::new();
        for t in 0..20u64 {
            let mut v = vec![0.0f32; lanes * 2];
            let s = match t { 3 => 0.99, 6 => 0.6, 7 => 0.7, _ => 0.0 };
            v[(t as usize % lanes) * 2 + 1] = s;
            // se start peaks on the lane of age 2
            let mut e = vec![0.0f32; lanes * 2];
            e[col.lane_of_age(t, 2) * 2 + SE_START_COLUMN] = 1.0;
            let vals: [&[f32]; 2] = [&v, &e];
            k.step_single(&vals, &At { t, end: 6 * (t as i64 + 1), fps: 6, props: &[] }, &mut out);
        }
        // step 3 is before START_CYCLE; the run peaks at step 7 and is emitted at 6*8 + 30 = 78 frames
        assert_eq!(out.len(), 1, "{out:?}");
        let d = &out[0];
        assert_eq!((d.keyword, d.kind, d.score), (1, DetectionKind::NearMiss, 0.7));
        assert_eq!((d.begin_sample, d.end_sample), ((6 * 6 + 1) * 160, 78 * 160));
    }
}
