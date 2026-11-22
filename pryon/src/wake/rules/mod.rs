//! Keyword orchestra: libpryon's keyword spotters (keyword_spotter.cpp, keyword_single_stage_spotter.cpp,
//! keyword_multi_word_spotter.cpp) driven by the kw.cfg fields and the op.cfg machine, one tick per front-end frame.

mod classifier;
mod machine;

use super::kwcfg::{ClassifierSpec, RestartSpec, SpotterSpec};
use super::pipeline::{LoadedSet, StepOut};
use super::{Detection, DetectionKind};
use crate::Error;
use classifier::{aggregate, aggregator, Column, Entry, Exp, Smoother, Tracker, Word};
use machine::Machine;
use std::collections::VecDeque;
use std::path::Path;

const FRAME_SAMPLES: u64 = crate::lfbe::HOP as u64;
/// Frame shift in milliseconds (pinned).
const FRAME_MS: u32 = 10;
/// libpryon's preroll the keyword start may reach back to (predicted_index_calculator.cpp).
const PREROLL_MS: f32 = 500.0;
/// Lowest start-alignment score libpryon accepts as valid (predicted_index_calculator.cpp).
const START_FLOOR: f32 = -0.25;
/// Keywords per set the machine's masks hold.
const MAX_KEYWORDS: usize = 64;
/// Word components of a multi-word spotter (libpryon asserts at most two).
const MAX_WORDS: usize = 2;

/// Active thresholds after libpryon's clamps (keyword_spotter_thresholds.cpp, 0x9c2468).
#[derive(Clone, Copy, Debug, PartialEq)]
struct Thr {
    accept: f32,
    prob: f32,
    notify: f32,
}

impl Thr {
    /// An absent probabilistic or notify threshold is FLT_MAX, then clamped below the accept threshold.
    fn new(accept: f32, prob: Option<f32>, notify: Option<f32>) -> Thr {
        let mut prob = prob.unwrap_or(f32::MAX);
        if prob > accept {
            prob = accept;
        }
        // the escalate threshold (not served) defaults to FLT_MAX and is clamped to `prob`, then bounds notify
        let mut notify = notify.unwrap_or(f32::MAX);
        if notify > prob {
            notify = prob;
        }
        Thr { accept, prob, notify }
    }
}

/// libpryon's spotter states (keyword_spotter.cpp) with their deadline, in the spotter's own frames.
#[derive(Clone, Copy, Debug, PartialEq)]
enum State {
    Idle,
    /// Tracking the peak since the notify threshold was crossed on a rising score (sDelayedNotification).
    Notify(u64),
    /// Above the probabilistic threshold (sDelayedProbabilisticAccept); always ends in a near miss here.
    Prob(u64),
    /// After its own accept.
    Locked(u64),
    /// Locked by its primary or verifier's accept; not scored.
    Held(u64),
}

/// Score and keyword start frame (the spotter's own count) of a candidate detection.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Cand {
    score: f32,
    start: u64,
}

/// How a spotter turns a step into a score and a keyword start.
#[derive(Debug)]
enum Scorer {
    /// KeywordSingleStageSpotter: one class column through its smoother; the start row at the smoother's index.
    Single { col: Column, se: Column, smoother: Smoother },
    /// KeywordMultiWordSpotter: word components under a weighted geometric average; the start is the first
    /// component's best entry's.
    Words { words: Vec<Word>, exps: Vec<Exp>, inv: Exp },
}

#[derive(Debug)]
struct Spot {
    keyword: usize,
    aux: bool,
    scorer: Scorer,
    thr: Thr,
    /// (property index, equals, thresholds).
    overrides: Vec<(usize, i32, Thr)>,
    lock: u64,
    notify_lock: u64,
    prob_lock: u64,
    kw_window: u32,
    /// Largest start-alignment score taken as is.
    start_cap: f32,
    state: State,
    cand: Cand,
    /// Previous step score for the rising rule; kept across resets (keyword_spotter.cpp only sets it at creation).
    prev: f32,
    /// The verifier of a primary, the primary of a verifier.
    sibling: Option<usize>,
    /// Own frame count: one ahead of the stream if enabled and then run in the same frame (0x9bf658).
    own: u64,
}

/// What a scored frame does (keyword_spotter.cpp transition, 0x9b62e0).
enum Action {
    Accept,
    Prob,
    Notify,
}

/// Keyword start frame for a step ending at frame `f` (predicted_index_calculator.cpp, 0x9dee6c).
fn start_frame(kw_window: u32, cap: f32, s: f32, f: u64) -> u64 {
    let s = if s >= START_FLOOR && cap >= s {
        s
    } else {
        let a = if s > START_FLOOR { s } else { START_FLOOR };
        if a < cap { a } else { cap }
    };
    let n = kw_window as f32;
    let p = s * n;
    let start = (p - n) as i64 + f as i64 + 1;
    if start <= 0 { 1 } else { start as u64 }
}

impl Spot {
    fn active(&self, props: &[(String, Option<i32>)]) -> Thr {
        // the first override whose property is set and equal wins (0x9c3958)
        self.overrides.iter().find(|&&(p, eq, _)| props[p].1 == Some(eq)).map_or(self.thr, |o| o.2)
    }

    /// The step's score and keyword start (own frames).
    fn measure(&mut self, step: &StepOut) -> Cand {
        let (own, values, t) = (self.own, step.values, step.step);
        let (n, cap) = (self.kw_window, self.start_cap);
        match &mut self.scorer {
            Scorer::Single { col, se, smoother } => {
                let (score, at) = smoother.score(col, values, t);
                Cand { score, start: start_frame(n, cap, se.at(values, t, at), own) }
            }
            Scorer::Words { words, exps, inv } => {
                let mut best = [0.0f32; MAX_WORDS];
                for (k, w) in words.iter_mut().enumerate() {
                    let score = w.decay.score(&w.col, values, t).0;
                    let start = start_frame(n, cap, w.se.get(values, 0), own);
                    w.track.push(Entry { frame: own, score, start });
                    best[k] = w.track.max().map_or(0.0, |e| e.score);
                }
                let k = words.len();
                let start = words[0].track.max().map_or(1, |e| e.start);
                Cand { score: aggregate(&best[..k], exps, *inv), start }
            }
        }
    }

    /// State, smoothers and histories restart; `prev` stays (keyword_spotter.cpp reset, 0x9b51b0/0x9af7bc).
    fn reset(&mut self) {
        self.state = State::Idle;
        match &mut self.scorer {
            Scorer::Single { smoother, .. } => smoother.reset(),
            Scorer::Words { words, .. } => {
                for w in words {
                    w.decay.reset();
                    w.track.reset();
                }
            }
        }
    }
}

pub(crate) struct Orchestra {
    /// kw.cfg order: libpryon runs its enabled spotters in that order every frame.
    spots: Vec<Spot>,
    /// Client properties the overrides test, with their value once set.
    props: Vec<(String, Option<i32>)>,
    /// Property changes not in effect yet: (samples since the origin when set, property, value).
    pending: VecDeque<(u64, usize, i32)>,
    machine: Machine,
    restart: RestartSpec,
    /// End samples of each keyword's reported events within the restart span.
    events: Vec<VecDeque<u64>>,
    /// Stream sample of frame 0.
    origin: u64,
    /// Frames since the origin, the current one included.
    frame: u64,
    restart_due: bool,
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
        if names.len() > MAX_KEYWORDS {
            return Err(unsupported(dir, format!("{} keywords (implemented: up to {MAX_KEYWORDS})", names.len())));
        }
        let mut props: Vec<(String, Option<i32>)> = Vec::new();
        for s in &spec.spotters {
            for o in &s.thresholds.overrides {
                if !props.iter().any(|(n, _)| n == &o.prop) {
                    props.push((o.prop.clone(), None));
                }
            }
        }
        let column = |s: &SpotterSpec, comp: &str, class: usize| {
            let st = spec.stages.iter().position(|st| st.name == s.scorer).ok_or_else(|| {
                unsupported(dir, format!("spotter {} reads scorer {}, which is not a stage", s.name, s.scorer))
            })?;
            Column::new(&set.components, &format!("{}/{comp}", s.scorer), class, &spec.stages[st].graph)
        };
        let mut spots: Vec<Spot> = Vec::new();
        for s in &spec.spotters {
            let what = |w: String| unsupported(dir, format!("spotter {}: {w}", s.name));
            let Some(keyword) = names.iter().position(|n| n == &s.name) else {
                return Err(what("auxiliary spotter of a keyword without a primary".into()));
            };
            let scorer = match &s.classifier {
                ClassifierSpec::WeightedGeometricAverage { weights } => {
                    let Some((exps, inv)) = aggregator(weights) else {
                        let w = format!("weights {weights:?} need a powf other than libm's exact cases (0, 0.5, 1, 2)");
                        return Err(what(w));
                    };
                    let mut words = Vec::new();
                    for (c, &rate) in s.components.iter().zip(&s.smoothers) {
                        let col = column(s, &c.component, c.class)?;
                        let Some(se_name) = &c.start_end else {
                            return Err(what("no keyword-start-end-component".into()));
                        };
                        let se = column(s, se_name, c.start_index)?;
                        // libpryon reads row 0 of each word's output (0x9afa78)
                        if col.lanes != 1 || se.lanes != 1 {
                            return Err(what(format!("word component over {} lanes (implemented: 1)", col.lanes)));
                        }
                        let decay = Smoother::Decay { rate, state: 0.0 };
                        let track = Tracker::new(c.history.unwrap_or(1));
                        words.push(Word { col, se, decay, track });
                    }
                    Scorer::Words { words, exps, inv }
                }
                c => {
                    let smoother = match *c {
                        ClassifierSpec::MovingAverageMaxPool { window, trim_tail } => {
                            Smoother::MaxPool { window, trim: trim_tail }
                        }
                        ClassifierSpec::ExponentialDecay { rate } => Smoother::Decay { rate, state: 0.0 },
                        ClassifierSpec::WeightedGeometricAverage { .. } => unreachable!(),
                    };
                    let [c] = &s.components[..] else {
                        let n = s.components.len();
                        return Err(what(format!("{n} word components with a single-component classifier")));
                    };
                    if !s.smoothers.is_empty() {
                        return Err(what("posterior-smoothers on a single-component classifier (not implemented)".into()));
                    }
                    let col = column(s, &c.component, c.class)?;
                    let Some(se_name) = &c.start_end else {
                        return Err(what("no keyword-start-end-component".into()));
                    };
                    let se = column(s, se_name, c.start_index)?;
                    if se.lanes != col.lanes {
                        return Err(what("start/end lanes differ from score lanes".into()));
                    }
                    match smoother {
                        Smoother::MaxPool { window, trim } if trim >= col.lanes || window > col.lanes - trim => {
                            let w = format!("window-size {window}, trim-tail {trim} over {} lanes", col.lanes);
                            return Err(what(w));
                        }
                        Smoother::Decay { .. } if col.lanes != 1 => {
                            let w = format!("exponential-decay over {} lanes (libpryon takes one score)", col.lanes);
                            return Err(what(w));
                        }
                        _ => {}
                    }
                    Scorer::Single { col, se, smoother }
                }
            };
            let t = &s.thresholds;
            let prob = t.probabilistic.map(|p| p.0);
            let overrides = t
                .overrides
                .iter()
                .map(|o| {
                    let p = props.iter().position(|(n, _)| n == &o.prop).unwrap();
                    (p, o.equals, Thr::new(o.accept, prob, t.notify))
                })
                .collect();
            let cap = PREROLL_MS / (FRAME_MS * s.kw_window) as f32;
            spots.push(Spot {
                keyword,
                aux: s.auxiliary,
                scorer,
                thr: Thr::new(t.accept, prob, t.notify),
                overrides,
                lock: s.lock.into(),
                notify_lock: s.notify_lock.into(),
                prob_lock: s.prob_lock.into(),
                kw_window: s.kw_window,
                start_cap: if cap > 1.0 { 1.0 } else { cap },
                state: State::Idle,
                cand: Cand { score: 0.0, start: 1 },
                prev: -f32::MAX,
                sibling: None,
                own: 0,
            });
        }
        for i in 0..spots.len() {
            if spots[i].aux {
                let p = spots.iter().position(|p| !p.aux && p.keyword == spots[i].keyword);
                let Some(p) = p else {
                    let n = &names[spots[i].keyword];
                    let what = format!("spotter {n}: verifier of a keyword without a served primary");
                    return Err(unsupported(dir, what));
                };
                spots[i].sibling = Some(p);
                spots[p].sibling = Some(i);
            }
        }
        let machine = match &spec.machine {
            Some(m) => Machine::new(m, &names),
            None => Machine::always(names.len()),
        };
        let restart = spec.restart;
        // a keyword reports at most one event per (shortest) lock period, so this bounds the queue
        let span_frames = u64::from(restart.span_ms) / u64::from(FRAME_MS) + 1;
        let events = (0..names.len())
            .map(|k| {
                let gap = spots.iter().filter(|s| s.keyword == k).map(|s| s.lock.min(s.notify_lock).min(s.prob_lock)).min();
                let n = span_frames / gap.unwrap_or(1).max(1) + u64::from(restart.max_events) + 1;
                VecDeque::with_capacity(n.min(4096) as usize)
            })
            .collect();
        let (pending, origin, frame, restart_due) = (VecDeque::new(), 0, 0, false);
        Ok(Orchestra { spots, props, pending, machine, restart, events, origin, frame, restart_due })
    }

    /// Stream sample of the current frame 0.
    pub fn origin(&self) -> u64 {
        self.origin
    }

    /// The restart trigger fired in the last tick: the caller restarts everything from the next sample.
    pub fn restart_due(&self) -> bool {
        self.restart_due
    }

    /// A client property set when `at` samples since the origin had been fed; it applies from the first frame
    /// whose 10 ms block starts at or after that point (probes pr_x*). Unknown names are ignored.
    pub fn set_property(&mut self, name: &str, value: i32, at: u64) {
        if let Some(p) = self.props.iter().position(|(n, _)| n == name) {
            self.pending.push_back((at, p, value));
        }
    }

    fn apply_pending(&mut self, upto: u64) {
        while let Some(&(at, p, v)) = self.pending.front() {
            if at > upto {
                break;
            }
            self.props[p].1 = Some(v);
            self.pending.pop_front();
        }
    }

    /// Session end at stream sample `origin`: a pending notify near miss is reported, every spotter restarts (its
    /// rising-score memory aside), the machine returns to its initial state and frames count from `origin`.
    pub fn reset(&mut self, origin: u64, out: &mut Vec<Detection>) {
        for i in 0..self.spots.len() {
            let sp = &self.spots[i];
            if let (true, false, State::Notify(_)) = (self.machine.allows(sp.keyword), sp.aux, sp.state) {
                out.push(self.detection(i, DetectionKind::NearMiss, sp.cand, self.frame));
                self.count(sp.keyword, self.frame);
            }
        }
        for sp in &mut self.spots {
            sp.reset();
            sp.own = 0;
        }
        self.machine.reset();
        self.restart_from(origin);
    }

    /// libpryon's defensive restart: a new decoder fed from stream sample `origin`; client properties stay.
    pub fn restart(&mut self, origin: u64) {
        for sp in &mut self.spots {
            sp.reset();
            (sp.own, sp.prev) = (0, -f32::MAX);
        }
        self.machine.reset();
        for q in &mut self.events {
            q.clear();
        }
        self.restart_due = false;
        self.restart_from(origin);
    }

    fn restart_from(&mut self, origin: u64) {
        self.apply_pending(u64::MAX);
        (self.origin, self.frame) = (origin, 0);
    }

    /// One front-end frame (`f` frames since the origin), scored when it completes a step; appends its detections.
    pub fn tick(&mut self, f: u64, step: Option<&StepOut>, out: &mut Vec<Detection>) {
        self.frame = f;
        self.apply_pending((f - 1) * FRAME_SAMPLES);
        // the machine runs before the orchestra counts the frame (0x9bd990)
        let enabled = self.machine.tick();
        self.enable(enabled, f - 1);
        for i in 0..self.spots.len() {
            if !self.machine.allows(self.spots[i].keyword) {
                continue;
            }
            self.spots[i].own += 1;
            self.deadline(i, f, out);
            if let Some(s) = step {
                self.score(i, f, s, out);
            }
            // the restart throws out of the frame: later spotters do not run
            if self.restart_due {
                return;
            }
        }
    }

    /// Spotters a transition enables restart with the orchestra's frame count (0x9bf658).
    fn enable(&mut self, keywords: u64, count: u64) {
        if keywords == 0 {
            return;
        }
        for sp in &mut self.spots {
            if keywords >> sp.keyword & 1 == 1 {
                sp.reset();
                sp.own = count;
            }
        }
    }

    fn detection(&self, i: usize, kind: DetectionKind, c: Cand, f: u64) -> Detection {
        let sp = &self.spots[i];
        let begin = c.start + f - sp.own;
        Detection {
            keyword: sp.keyword,
            kind,
            score: c.score,
            begin_sample: self.origin + begin * FRAME_SAMPLES,
            end_sample: self.origin + f * FRAME_SAMPLES,
            afe: None,
        }
    }

    /// State update at the start of the spotter's frame (keyword_spotter.cpp, 0x9b5488).
    fn deadline(&mut self, i: usize, f: u64, out: &mut Vec<Detection>) {
        let sp = &self.spots[i];
        match sp.state {
            State::Notify(d) | State::Prob(d) if d == sp.own => {
                // a verifier's near miss is dropped by libpryon's result builder ("Unexpected detection event type")
                if !sp.aux {
                    out.push(self.detection(i, DetectionKind::NearMiss, sp.cand, f));
                    self.count(sp.keyword, f);
                }
                self.spots[i].state = State::Idle;
            }
            State::Locked(d) | State::Held(d) if d == sp.own => self.spots[i].state = State::Idle,
            _ => {}
        }
    }

    /// keyword_single_stage_spotter.cpp (0x9b362c, shared by the multi-word spotter) and the action it returns
    /// (keyword_spotter.cpp, 0x9b5248).
    fn score(&mut self, i: usize, f: u64, step: &StepOut, out: &mut Vec<Detection>) {
        let thr = self.spots[i].active(&self.props);
        let sp = &mut self.spots[i];
        if let State::Held(_) = sp.state {
            return;
        }
        let cur = sp.measure(step);
        let s = cur.score;
        let action = match sp.state {
            State::Idle if thr.accept <= s => Some(Action::Accept),
            State::Idle if thr.prob <= s => Some(Action::Prob),
            State::Idle if thr.notify <= s => Some(Action::Notify),
            State::Notify(_) if thr.accept <= s => Some(Action::Accept),
            State::Notify(_) if thr.prob <= s => Some(Action::Prob),
            State::Prob(_) if thr.accept <= s => Some(Action::Accept),
            _ => None,
        };
        let rising = s - sp.prev > 0.0;
        sp.prev = s;
        let own = sp.own;
        if let State::Notify(_) = sp.state {
            if s > sp.cand.score {
                sp.cand = cur;
                sp.state = State::Notify(own + sp.notify_lock);
            }
        }
        match action {
            Some(Action::Accept) => {
                let kind = if sp.aux { DetectionKind::AcceptAuxiliary } else { DetectionKind::Accept };
                let until = own + sp.lock;
                sp.state = State::Locked(until);
                let (keyword, sibling) = (sp.keyword, sp.sibling);
                if let Some(j) = sibling {
                    self.spots[j].state = State::Held(until);
                }
                out.push(self.detection(i, kind, cur, f));
                self.accepted(keyword, f);
            }
            Some(Action::Prob) => {
                sp.cand = cur;
                sp.state = State::Prob(own + sp.prob_lock);
            }
            Some(Action::Notify) if rising => {
                sp.cand = cur;
                sp.state = State::Notify(own + sp.notify_lock);
            }
            _ => {}
        }
    }

    /// The restart span runs on stream time here, on the wall clock in libpryon (equal in real time).
    fn accepted(&mut self, keyword: usize, f: u64) {
        let enabled = self.machine.on_accept(keyword);
        self.enable(enabled, f);
        if self.count(keyword, f) >= self.restart.max_events as usize {
            self.restart_due = true;
        }
    }

    /// Records a reported event of `keyword` at frame `f`; returns the events within the span.
    fn count(&mut self, keyword: usize, f: u64) -> usize {
        let end = self.origin + f * FRAME_SAMPLES;
        let span = u64::from(self.restart.span_ms) * (FRAME_SAMPLES / u64::from(FRAME_MS));
        let q = &mut self.events[keyword];
        while q.front().is_some_and(|&t| end - t >= span) {
            q.pop_front();
        }
        q.push_back(end);
        q.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FPS: u64 = 8;
    const LANES: usize = 4;

    /// A spotter over one `[1, LANES, 2]` score output (component 0) and its `[1, LANES, 2]` start/end output
    /// (component 1, start column 0); window 1, no trim.
    fn spot(keyword: usize, aux: bool, accept: f32, prob: Option<f32>, notify: Option<f32>, lock: u64) -> Spot {
        Spot {
            keyword,
            aux,
            scorer: Scorer::Single {
                col: Column { comp: 0, lanes: LANES, classes: 2, class: 1 },
                se: Column { comp: 1, lanes: LANES, classes: 2, class: 0 },
                smoother: Smoother::MaxPool { window: 1, trim: 0 },
            },
            thr: Thr::new(accept, prob, notify),
            overrides: vec![],
            lock,
            notify_lock: lock,
            prob_lock: lock,
            kw_window: 10,
            start_cap: 1.0,
            state: State::Idle,
            cand: Cand { score: 0.0, start: 1 },
            prev: -f32::MAX,
            sibling: None,
            own: 0,
        }
    }

    /// Per step, every spotter's score (all lanes alike) and start-alignment score; ticks every frame.
    fn run(o: &mut Orchestra, steps: &[(f32, f32)], frames: u64) -> Vec<(DetectionKind, usize, f32, u64, u64)> {
        let mut out = Vec::new();
        for f in 1..=frames {
            let st = if f % FPS == 0 { Some(f / FPS - 1) } else { None };
            let (s, a) = st.and_then(|t| steps.get(t as usize).copied()).unwrap_or((0.0, 0.0));
            let v: Vec<f32> = (0..LANES).flat_map(|_| [0.0, s]).collect();
            let e: Vec<f32> = (0..LANES).flat_map(|_| [a, 1.0]).collect();
            let vals: [&[f32]; 2] = [&v, &e];
            let so = st.map(|t| StepOut { step: t, frames: f, values: &vals });
            o.tick(f, so.as_ref(), &mut out);
        }
        out.iter().map(|d| (d.kind, d.keyword, d.score, d.begin_sample / 160, d.end_sample / 160)).collect()
    }

    fn orch(spots: Vec<Spot>) -> Orchestra {
        let keywords = spots.iter().map(|s| s.keyword + 1).max().unwrap_or(0);
        let restart = RestartSpec { max_events: u32::MAX, span_ms: 1 };
        let mut o = Orchestra {
            spots,
            props: vec![],
            pending: VecDeque::new(),
            machine: Machine::always(keywords),
            restart,
            events: (0..keywords).map(|_| VecDeque::new()).collect(),
            origin: 0,
            frame: 0,
            restart_due: false,
        };
        let pairs: Vec<(usize, usize)> = (0..o.spots.len())
            .filter(|&i| o.spots[i].aux)
            .map(|i| (i, o.spots.iter().position(|p| !p.aux && p.keyword == o.spots[i].keyword).unwrap()))
            .collect();
        for (a, p) in pairs {
            o.spots[a].sibling = Some(p);
            o.spots[p].sibling = Some(a);
        }
        o
    }

    use DetectionKind::{Accept, AcceptAuxiliary, NearMiss};

    #[test]
    fn begin_from_start_end_component() {
        let mut s = spot(0, false, 0.9, None, None, 100);
        (s.kw_window, s.start_cap) = (100, 0.5);
        // start = f + 1 + trunc(100 * a - 100), a clamped to [-0.25, 500 ms / (10 ms * 100)]
        assert_eq!(start_frame(100, 0.5, 0.3, 200), 131);
        assert_eq!(start_frame(100, 0.5, 0.9, 200), 151, "above the preroll cap");
        assert_eq!(start_frame(100, 0.5, -0.7, 200), 76, "below the floor");
        assert_eq!(start_frame(100, 0.5, 0.3, 40), 1, "before the stream start");
        assert_eq!(start_frame(100, 0.5, 0.305, 200), 132, "truncated toward zero");
        let mut o = orch(vec![s]);
        let mut steps = vec![(0.1, 0.0); 12];
        steps[11] = (0.95, 0.3);
        assert_eq!(run(&mut o, &steps, 96), [(Accept, 0, 0.95, 96 + 1 - 70, 96)]);
    }

    #[test]
    fn lock_units_frames() {
        // lock 16 frames = 2 steps: the deadline frame itself may accept again
        let mut o = orch(vec![spot(0, false, 0.9, None, None, 16)]);
        let got: Vec<u64> = run(&mut o, &[(0.95, 0.0); 6], 48).iter().map(|e| e.4).collect();
        assert_eq!(got, [8, 24, 40]);
        let mut o = orch(vec![spot(0, false, 0.9, None, None, 17)]);
        let got: Vec<u64> = run(&mut o, &[(0.95, 0.0); 6], 48).iter().map(|e| e.4).collect();
        assert_eq!(got, [8, 32]);
    }

    #[test]
    fn near_miss_window_from_notify_lock_period() {
        let mut o = orch(vec![spot(0, false, 0.9, None, Some(0.5), 20)]);
        // rises over notify at step 1 (frame 16), peaks at step 3 (frame 32): emitted at 32 + 20
        let got = run(&mut o, &[(0.1, 0.0), (0.6, 0.0), (0.55, 0.0), (0.7, 0.4), (0.6, 0.0), (0.1, 0.0)], 80);
        assert_eq!(got, [(NearMiss, 0, 0.7, 32 + 1 - 6, 52)]);
    }

    #[test]
    fn near_miss_needs_a_rising_score() {
        let mut o = orch(vec![spot(0, false, 0.9, None, Some(0.5), 20)]);
        // a score already above notify when the previous run ends starts no new one until it rises
        let got: Vec<u64> = run(&mut o, &[(0.6, 0.0); 8], 64).iter().map(|e| e.4).collect();
        assert_eq!(got, [28]);
        let mut o = orch(vec![spot(0, false, 0.9, None, Some(0.5), 20)]);
        let mut steps = [(0.6, 0.0); 8];
        steps[4] = (0.61, 0.0);
        let got: Vec<u64> = run(&mut o, &steps, 64).iter().map(|e| e.4).collect();
        assert_eq!(got, [28, 60]);
    }

    #[test]
    fn near_miss_from_probabilistic_threshold() {
        let mut o = orch(vec![spot(0, false, 0.9, Some(0.5), Some(0.3), 20)]);
        // over the probabilistic threshold: the entry candidate, emitted a probabilistic lock period later
        let got = run(&mut o, &[(0.6, 0.0), (0.8, 0.0), (0.1, 0.0), (0.1, 0.0), (0.1, 0.0)], 40);
        assert_eq!(got, [(NearMiss, 0, 0.6, 1, 28)]);
    }

    #[test]
    fn accept_cancels_pending_near_miss() {
        let mut o = orch(vec![spot(0, false, 0.9, None, Some(0.5), 20)]);
        let got = run(&mut o, &[(0.6, 0.0), (0.95, 0.0), (0.1, 0.0), (0.1, 0.0), (0.1, 0.0)], 40);
        assert_eq!(got, [(Accept, 0, 0.95, 16 + 1 - 10, 16)]);
    }

    #[test]
    fn verifier_same_step_precedence() {
        // primary listed first wins a shared crossing and holds the verifier for its own lock period
        let mut o = orch(vec![spot(0, false, 0.9, None, None, 40), spot(0, true, 0.9, None, None, 16)]);
        let got: Vec<_> = run(&mut o, &[(0.95, 0.0); 8], 64).iter().map(|e| (e.0, e.4)).collect();
        assert_eq!(got, [(Accept, 8), (Accept, 48)]);
        // a verifier listed first wins and holds the primary for the verifier's lock period
        let mut o = orch(vec![spot(0, true, 0.9, None, None, 16), spot(0, false, 0.9, None, None, 40)]);
        let got: Vec<_> = run(&mut o, &[(0.95, 0.0); 8], 64).iter().map(|e| (e.0, e.4)).collect();
        assert_eq!(got, [(AcceptAuxiliary, 8), (AcceptAuxiliary, 24), (AcceptAuxiliary, 40), (AcceptAuxiliary, 56)]);
    }

    #[test]
    fn verifier_near_miss_not_emitted() {
        let mut o = orch(vec![spot(0, false, 0.9, None, None, 20), spot(0, true, 0.9, None, Some(0.5), 20)]);
        assert_eq!(run(&mut o, &[(0.6, 0.0), (0.1, 0.0), (0.1, 0.0), (0.1, 0.0)], 40), []);
    }

    #[test]
    fn notify_absent_means_no_near_miss() {
        let mut o = orch(vec![spot(0, false, 0.9, None, None, 20)]);
        assert_eq!(run(&mut o, &[(0.6, 0.0), (0.89, 0.0), (0.1, 0.0), (0.1, 0.0)], 60), []);
    }

    #[test]
    fn thresholds_clamped_to_accept() {
        assert_eq!(Thr::new(0.9, Some(0.95), Some(0.97)), Thr { accept: 0.9, prob: 0.9, notify: 0.9 });
        assert_eq!(Thr::new(0.7, Some(0.5), Some(0.3)), Thr { accept: 0.7, prob: 0.5, notify: 0.3 });
        assert_eq!(Thr::new(0.4, Some(0.5), Some(0.45)), Thr { accept: 0.4, prob: 0.4, notify: 0.4 });
    }


    #[test]
    fn deadline_between_steps() {
        let mut o = orch(vec![spot(0, false, 0.9, None, Some(0.5), 13)]);
        // near miss due at frame 8 + 13 = 21, between steps; the stream stops at frame 21
        let got = run(&mut o, &[(0.6, 0.0), (0.1, 0.0), (0.1, 0.0)], 21);
        assert_eq!(got.iter().map(|e| e.4).collect::<Vec<_>>(), [21]);
    }

    #[test]
    fn held_spotter_not_scored() {
        // a decay verifier held by its primary does not update its state
        let mut p = spot(0, false, 0.5, None, None, 16);
        let mut v = spot(0, true, 0.9, None, None, 16);
        v.scorer = Scorer::Single {
            col: Column { comp: 0, lanes: 1, classes: 2, class: 1 },
            se: Column { comp: 1, lanes: 1, classes: 2, class: 0 },
            smoother: Smoother::Decay { rate: 0.5, state: 0.0 },
        };
        p.thr = Thr::new(0.5, None, None);
        let mut o = orch(vec![p, v]);
        run(&mut o, &[(0.95, 0.0), (0.1, 0.0)], 16);
        let Scorer::Single { smoother, .. } = &o.spots[1].scorer else { unreachable!() };
        assert_eq!(*smoother, Smoother::Decay { rate: 0.5, state: 0.0 });
    }
}
