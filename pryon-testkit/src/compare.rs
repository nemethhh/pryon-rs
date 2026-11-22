//! Pairs libpryon's wake events with the engine's and tallies the differences (shared by the oracle test and `validate`).

use crate::oracle::{OracleEvent, OracleKind};
use std::collections::BTreeMap;

/// One event from either engine, in the oracle's terms.
#[derive(Clone, Debug)]
pub struct Ev {
    pub keyword: String,
    pub kind: OracleKind,
    pub score: f32,
    pub begin: u64,
    pub end: u64,
}

impl std::fmt::Display for Ev {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}(score={:.6},begin={},end={})", self.kind, self.score, self.begin, self.end)
    }
}

pub fn from_oracle(o: &[OracleEvent]) -> Vec<Ev> {
    o.iter()
        .map(|e| Ev {
            keyword: e.keyword.clone(),
            kind: e.kind,
            score: e.score.parse().unwrap_or(f32::NAN),
            begin: e.begin,
            end: e.end,
        })
        .collect()
}

/// Oracle and engine events are paired per keyword by nearest end sample within this distance.
pub const PAIR_WINDOW: u64 = 16000;
/// Not emitted until Task 12.
const UNSERVED: &[&str] = &["HEY_DISNEY"];
/// The oracle reports STOP only while awake (op.cfg), the rules emit it always; Task 12 removes this.
pub const AWAKE_ONLY: &str = "STOP";

/// Thresholds a score is compared against, and probabilistic-accept thresholds, per keyword.
#[derive(Default, Clone, Debug)]
pub struct Limits {
    thr: BTreeMap<String, Vec<f32>>,
    prob: BTreeMap<String, f32>,
}

impl Limits {
    /// One spotter's accept and notify thresholds, overrides and probabilistic threshold.
    pub fn add(&mut self, keyword: &str, accept: f32, notify: Option<f32>, overrides: &[f32], prob: Option<f32>) {
        let t = self.thr.entry(keyword.to_string()).or_default();
        t.push(accept);
        t.extend(notify);
        t.extend_from_slice(overrides);
        if let Some(p) = prob {
            let e = self.prob.entry(keyword.to_string()).or_insert(p);
            *e = e.min(p);
        }
    }
}

/// One difference between the engines; a missing side is `None`.
#[derive(Clone, Debug)]
pub struct Diff {
    pub keyword: String,
    pub oracle: Option<Ev>,
    pub engine: Option<Ev>,
}

impl std::fmt::Display for Diff {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let side = |e: &Option<Ev>| e.as_ref().map_or("-".to_string(), |e| e.to_string());
        write!(f, "{} oracle={} rust={}", self.keyword, side(&self.oracle), side(&self.engine))
    }
}

#[derive(Default, Clone)]
pub struct Tally {
    pub clips: usize,
    pub oracle: usize,
    pub engine: usize,
    pub dropped: usize,
    pub matched: usize,
    pub differing: usize,
    pub oracle_only: usize,
    pub engine_only: usize,
    /// Near misses among the one-sided events.
    pub oracle_only_nm: usize,
    pub engine_only_nm: usize,
    pub awake_only: usize,
    /// Oracle accepts below the active threshold but above the probabilistic one; counted nowhere else.
    pub prob: usize,
    pub kind_eq: usize,
    pub begin_eq: usize,
    pub end_eq: usize,
    pub score_ok: usize,
    pub accepts: usize,
    pub accepts_paired: usize,
    pub near_misses: usize,
    pub score_d: Vec<f32>,
}

impl Tally {
    pub fn add(&mut self, o: &Tally) {
        let n = |a: &mut usize, b: usize| *a += b;
        n(&mut self.clips, o.clips);
        n(&mut self.oracle, o.oracle);
        n(&mut self.engine, o.engine);
        n(&mut self.dropped, o.dropped);
        n(&mut self.matched, o.matched);
        n(&mut self.differing, o.differing);
        n(&mut self.oracle_only, o.oracle_only);
        n(&mut self.engine_only, o.engine_only);
        n(&mut self.oracle_only_nm, o.oracle_only_nm);
        n(&mut self.engine_only_nm, o.engine_only_nm);
        n(&mut self.awake_only, o.awake_only);
        n(&mut self.prob, o.prob);
        n(&mut self.kind_eq, o.kind_eq);
        n(&mut self.begin_eq, o.begin_eq);
        n(&mut self.end_eq, o.end_eq);
        n(&mut self.score_ok, o.score_ok);
        n(&mut self.accepts, o.accepts);
        n(&mut self.accepts_paired, o.accepts_paired);
        n(&mut self.near_misses, o.near_misses);
        self.score_d.extend_from_slice(&o.score_d);
    }

    pub fn clean(&self) -> bool {
        self.differing == 0 && self.oracle_only == 0 && self.engine_only == 0
    }

    pub fn line(&self) -> String {
        let mut d = self.score_d.clone();
        d.sort_by(f32::total_cmp);
        let q = |p: f32| if d.is_empty() { 0.0 } else { d[((d.len() - 1) as f32 * p) as usize] };
        format!(
            "clips {} | oracle {} (+{} dropped) engine {} | matched {} differing {} oracle-only {} ({} NM) \
             engine-only {} ({} NM, +{} {AWAKE_ONLY} while oracle asleep) | paired: kind {} begin {} end {} \
             score {} | oracle accepts {} paired {} | score |d| median {:.2e} p90 {:.2e} max {:.2e}",
            self.clips,
            self.oracle,
            self.dropped,
            self.engine,
            self.matched,
            self.differing,
            self.oracle_only,
            self.oracle_only_nm,
            self.engine_only,
            self.engine_only_nm,
            self.awake_only,
            self.kind_eq,
            self.begin_eq,
            self.end_eq,
            self.score_ok,
            self.accepts,
            self.accepts_paired,
            q(0.5),
            q(0.9),
            q(1.0),
        )
    }
}

/// Exact 6-decimal equality when every threshold is more than 1e-2 away, else within 1e-3.
fn score_ok(oracle: &Ev, printed_threshold: Option<f32>, engine: f32, thr: &[f32]) -> bool {
    let margin = thr.iter().chain(&printed_threshold).map(|t| (oracle.score - t).abs()).fold(f32::INFINITY, f32::min);
    if margin > 1e-2 {
        format!("{:.6}", oracle.score) == format!("{:.6}", engine)
    } else {
        (oracle.score - engine).abs() <= 1e-3
    }
}

/// `exact`: scores are compared as 6-decimal strings with no margin rule. `prob` receives the oracle-only events
/// that match the probabilistic-accept rule.
pub fn compare(
    oracle: &[OracleEvent],
    engine: &[Ev],
    limits: &Limits,
    exact: bool,
    diffs: &mut Vec<Diff>,
    prob: &mut Vec<Ev>,
) -> Tally {
    let mut t = Tally { clips: 1, ..Tally::default() };
    let printed: Vec<Option<f32>> = oracle.iter().map(|e| e.threshold.parse().ok()).collect();
    let all = from_oracle(oracle);
    let kept: Vec<(usize, &Ev)> = all.iter().enumerate().filter(|(_, e)| !UNSERVED.contains(&e.keyword.as_str())).collect();
    t.dropped = all.len() - kept.len();
    t.oracle = kept.len();
    t.engine = engine.len();
    let mut used = vec![false; engine.len()];
    for &(i, o) in &kept {
        t.near_misses += usize::from(o.kind == OracleKind::NearMiss);
        let best = engine
            .iter()
            .enumerate()
            .filter(|(j, e)| !used[*j] && e.keyword == o.keyword && e.end.abs_diff(o.end) <= PAIR_WINDOW)
            .min_by_key(|(_, e)| e.end.abs_diff(o.end));
        let Some((j, e)) = best else {
            let probabilistic = o.kind == OracleKind::Accept
                && limits.prob.get(&o.keyword).is_some_and(|&p| o.score >= p && printed[i].is_some_and(|a| o.score < a));
            if probabilistic {
                t.prob += 1;
                t.oracle -= 1;
                prob.push(o.clone());
                continue;
            }
            t.accepts += usize::from(o.kind != OracleKind::NearMiss);
            t.oracle_only += 1;
            t.oracle_only_nm += usize::from(o.kind == OracleKind::NearMiss);
            diffs.push(Diff { keyword: o.keyword.clone(), oracle: Some(o.clone()), engine: None });
            continue;
        };
        used[j] = true;
        if o.kind != OracleKind::NearMiss {
            t.accepts += 1;
            t.accepts_paired += 1;
        }
        let ok_score = if exact {
            format!("{:.6}", o.score) == format!("{:.6}", e.score)
        } else {
            score_ok(o, printed[i], e.score, limits.thr.get(&o.keyword).map_or(&[][..], |v| v))
        };
        t.kind_eq += usize::from(o.kind == e.kind);
        t.begin_eq += usize::from(o.begin == e.begin);
        t.end_eq += usize::from(o.end == e.end);
        t.score_ok += usize::from(ok_score);
        t.score_d.push((o.score - e.score).abs());
        if o.kind == e.kind && o.begin == e.begin && o.end == e.end && ok_score {
            t.matched += 1;
        } else {
            t.differing += 1;
            diffs.push(Diff { keyword: o.keyword.clone(), oracle: Some(o.clone()), engine: Some(e.clone()) });
        }
    }
    for e in engine.iter().enumerate().filter(|(j, _)| !used[*j]).map(|(_, e)| e) {
        if e.keyword == AWAKE_ONLY {
            t.awake_only += 1;
        } else {
            t.engine_only += 1;
            t.engine_only_nm += usize::from(e.kind == OracleKind::NearMiss);
            diffs.push(Diff { keyword: e.keyword.clone(), oracle: None, engine: Some(e.clone()) });
        }
    }
    t
}

/// Samples pryon_test fed (`pushed N samples`); it stops early when libpryon restarts its decoder.
pub fn pushed(text: &str) -> Option<usize> {
    let l = text.lines().find_map(|l| l.strip_prefix("pushed "))?;
    l.split(' ').next()?.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn oe(kind: OracleKind, score: &str, thr: &str) -> OracleEvent {
        OracleEvent { keyword: "ALEXA".into(), kind, score: score.into(), threshold: thr.into(), begin: 100, end: 2000, afe: None }
    }

    fn run(o: &[OracleEvent], limits: &Limits) -> (Tally, Vec<Diff>, Vec<Ev>) {
        let (mut d, mut p) = (Vec::new(), Vec::new());
        let t = compare(o, &[], limits, false, &mut d, &mut p);
        (t, d, p)
    }

    #[test]
    fn prob_rule_only_in_band_with_probabilistic_thresholds() {
        let mut l = Limits::default();
        l.add("ALEXA", 0.9, None, &[], None);
        let (t, d, _) = run(&[oe(OracleKind::Accept, "0.85", "0.9")], &l);
        assert_eq!((t.oracle_only, d.len(), t.prob), (1, 1, 0));
        let mut l = Limits::default();
        l.add("ALEXA", 0.9, None, &[], Some(0.8));
        let (t, d, p) = run(&[oe(OracleKind::Accept, "0.85", "0.9")], &l);
        assert_eq!((t.oracle_only, d.len(), t.prob, p.len(), t.accepts, t.oracle), (0, 0, 1, 1, 0, 0));
        let (t, _, _) = run(&[oe(OracleKind::Accept, "0.79", "0.9")], &l);
        assert_eq!(t.oracle_only, 1);
    }
}
