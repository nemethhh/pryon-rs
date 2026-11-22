//! Pairs libpryon's wake events with the engine's and tallies the differences (shared by the oracle test and `validate`).

use crate::oracle::{OracleCmd, OracleEvent, OracleKind};
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
/// Thresholds a score is compared against, per keyword.
#[derive(Default, Clone, Debug)]
pub struct Limits {
    thr: BTreeMap<String, Vec<f32>>,
}

impl Limits {
    /// One spotter's accept and notify thresholds and overrides.
    pub fn add(&mut self, keyword: &str, accept: f32, notify: Option<f32>, overrides: &[f32]) {
        let t = self.thr.entry(keyword.to_string()).or_default();
        t.push(accept);
        t.extend(notify);
        t.extend_from_slice(overrides);
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
    pub matched: usize,
    pub differing: usize,
    pub oracle_only: usize,
    pub engine_only: usize,
    /// Near misses among the one-sided events.
    pub oracle_only_nm: usize,
    pub engine_only_nm: usize,
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
        n(&mut self.matched, o.matched);
        n(&mut self.differing, o.differing);
        n(&mut self.oracle_only, o.oracle_only);
        n(&mut self.engine_only, o.engine_only);
        n(&mut self.oracle_only_nm, o.oracle_only_nm);
        n(&mut self.engine_only_nm, o.engine_only_nm);
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
            "clips {} | oracle {} engine {} | matched {} differing {} oracle-only {} ({} NM) engine-only {} ({} NM) \
             | paired: kind {} begin {} end {} score {} | oracle accepts {} paired {} | score |d| median {:.2e} \
             p90 {:.2e} max {:.2e}",
            self.clips,
            self.oracle,
            self.engine,
            self.matched,
            self.differing,
            self.oracle_only,
            self.oracle_only_nm,
            self.engine_only,
            self.engine_only_nm,
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

/// `exact`: scores are compared as 6-decimal strings with no margin rule.
pub fn compare(oracle: &[OracleEvent], engine: &[Ev], limits: &Limits, exact: bool, diffs: &mut Vec<Diff>) -> Tally {
    let mut t = Tally { clips: 1, ..Tally::default() };
    let printed: Vec<Option<f32>> = oracle.iter().map(|e| e.threshold.parse().ok()).collect();
    let all = from_oracle(oracle);
    t.oracle = all.len();
    t.engine = engine.len();
    let mut used = vec![false; engine.len()];
    for (i, o) in all.iter().enumerate() {
        t.near_misses += usize::from(o.kind == OracleKind::NearMiss);
        let best = engine
            .iter()
            .enumerate()
            .filter(|(j, e)| !used[*j] && e.keyword == o.keyword && e.end.abs_diff(o.end) <= PAIR_WINDOW)
            .min_by_key(|(_, e)| e.end.abs_diff(o.end));
        let Some((j, e)) = best else {
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
        t.engine_only += 1;
        t.engine_only_nm += usize::from(e.kind == OracleKind::NearMiss);
        diffs.push(Diff { keyword: e.keyword.clone(), oracle: None, engine: Some(e.clone()) });
    }
    t
}

/// v2: pryon_test goes on after a restart; the oracle runs `deterministic_set`.
pub const ORACLE_CACHE_VERSION: u32 = 2;

/// Cache key of one oracle run: the padded clip's FNV-1a, the commands and the feed speed.
pub fn oracle_cache_key(padded_fnv: u64, cmds: &[OracleCmd], speed: f64) -> u64 {
    crate::fnv1a64(format!("v{ORACLE_CACHE_VERSION}|{padded_fnv:016x}|{cmds:?}|{speed}").as_bytes())
}

/// Leading digital silence for both engines: the firmware oracle scores nothing in its first ~1.7 s.
pub const LEAD_MS: u32 = 2500;
/// Trailing digital silence: libpryon reports an accept ~120 ms after the keyword ends, never at the clip end.
pub const TAIL_MS: u32 = 1000;

/// `pcm` between `lead_ms` and `tail_ms` of digital silence (16 kHz).
pub fn pad_clip(pcm: &[i16], lead_ms: u32, tail_ms: u32) -> Vec<i16> {
    let (lead, tail) = (lead_ms as usize * 16, tail_ms as usize * 16);
    let mut out = vec![0i16; lead + pcm.len() + tail];
    out[lead..lead + pcm.len()].copy_from_slice(pcm);
    out
}

/// Samples pryon_test fed (`pushed N samples`).
pub fn pushed(text: &str) -> Option<usize> {
    let l = text.lines().find_map(|l| l.strip_prefix("pushed "))?;
    l.split(' ').next()?.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pad_clip_silence_both_sides() {
        assert_eq!(pad_clip(&[1, -2], 0, 0), [1, -2]);
        let p = pad_clip(&[1, -2], 1, 2);
        assert_eq!(p.len(), 16 + 2 + 32);
        assert_eq!((&p[15..19], p.iter().filter(|&&x| x != 0).count()), (&[0, 1, -2, 0][..], 2));
    }

    #[test]
    fn unmatched_events_are_differences() {
        let o = OracleEvent {
            keyword: "STOP".into(),
            kind: OracleKind::Accept,
            score: "0.85".into(),
            threshold: "0.9".into(),
            begin: 100,
            end: 2000,
            afe: None,
        };
        let e = Ev { keyword: "HEY_DISNEY".into(), kind: OracleKind::NearMiss, score: 0.3, begin: 1, end: 5 };
        let mut d = Vec::new();
        let t = compare(&[o], &[e], &Limits::default(), false, &mut d);
        assert_eq!((t.oracle_only, t.engine_only, d.len(), t.clean()), (1, 1, 2, false));
    }

    #[test]
    fn cache_key_covers_commands_and_speed() {
        let p = |at| OracleCmd::Property { at, name: "AlarmState".into(), value: 1 };
        let k = oracle_cache_key(7, &[], 1.0);
        assert_ne!(k, oracle_cache_key(8, &[], 1.0));
        assert_ne!(k, oracle_cache_key(7, &[p(0)], 1.0));
        assert_ne!(oracle_cache_key(7, &[p(0)], 1.0), oracle_cache_key(7, &[p(1)], 1.0));
        assert_ne!(k, oracle_cache_key(7, &[OracleCmd::SessionEnd { at: 0 }], 1.0));
        assert_ne!(k, oracle_cache_key(7, &[], 2.0));
    }
}
