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

/// `x` as pryon_test prints a score: C `%g` (6 significant digits, ties to even, trailing zeros dropped).
pub fn fmt_g(x: f32) -> String {
    if !x.is_finite() || x == 0.0 {
        return if x == 0.0 { "0".into() } else { format!("{x}") };
    }
    // every f32 has a finite decimal expansion; 160 digits hold it, so the rounding below sees the exact value
    let full = format!("{:.160e}", f64::from(x).abs());
    let (mant, exp) = full.split_once('e').unwrap();
    let mut exp: i32 = exp.parse().unwrap();
    let digits: Vec<u8> = mant.bytes().filter(u8::is_ascii_digit).map(|b| b - b'0').collect();
    let mut kept: Vec<u8> = digits[..6].to_vec();
    let rest = &digits[6..];
    let up = rest[0] > 5 || (rest[0] == 5 && (rest[1..].iter().any(|&d| d != 0) || kept[5] % 2 == 1));
    if up {
        let mut i = 5;
        loop {
            if kept[i] < 9 {
                kept[i] += 1;
                break;
            }
            kept[i] = 0;
            if i == 0 {
                kept.insert(0, 1);
                kept.pop();
                exp += 1;
                break;
            }
            i -= 1;
        }
    }
    let sign = if x < 0.0 { "-" } else { "" };
    let trim = |s: String| if s.contains('.') { s.trim_end_matches('0').trim_end_matches('.').to_string() } else { s };
    let ds: String = kept.iter().map(|d| char::from(b'0' + d)).collect();
    if (-4..6).contains(&exp) {
        let s = if exp >= 0 {
            let (a, b) = ds.split_at(exp as usize + 1);
            format!("{a}.{b}")
        } else {
            format!("0.{}{ds}", "0".repeat((-exp - 1) as usize))
        };
        format!("{sign}{}", trim(s))
    } else {
        let m = trim(format!("{}.{}", &ds[..1], &ds[1..]));
        format!("{sign}{m}e{}{:02}", if exp < 0 { '-' } else { '+' }, exp.abs())
    }
}

/// `exact`: the engine's score printed as pryon_test prints it must equal the oracle's string, with no margin rule.
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
            oracle[i].score == fmt_g(e.score)
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

/// libpryon starts a firmware set's conv buffers from uninitialised heap (data-less initializers), so its scores are
/// NaN until those values have left the network: 168 or 174 frames after a start, session end or decoder restart in
/// the installs measured (the garbage depends on the install path). The engine starts from zeros and does not model it.
pub const FW_GATE_MAX_FRAMES: u64 = 174;

/// Events set aside by the firmware gate allowance, per clip or summed.
#[derive(Default, Clone, Copy, Debug, PartialEq)]
pub struct GateTally {
    /// One-sided events, or equal-kind pairs with both ends, in a gate window.
    pub window: usize,
    /// Engine events ending inside the gate itself, where libpryon scores nothing.
    pub inside: usize,
    /// Events after the first restart the two engines place differently.
    pub after_restart: usize,
}

impl GateTally {
    pub fn add(&mut self, o: &GateTally) {
        self.window += o.window;
        self.inside += o.inside;
        self.after_restart += o.after_restart;
    }

    pub fn total(&self) -> usize {
        self.window + self.inside + self.after_restart
    }
}

/// One clip's comparison under the gate allowance.
pub struct Gated {
    pub tally: Tally,
    /// Differences the allowance does not cover.
    pub diffs: Vec<Diff>,
    pub gate: GateTally,
    /// The first restart the engines place differently (oracle, engine), on a set without the gate: a failure.
    pub restart_mismatch: Option<(Option<u64>, Option<u64>)>,
}

/// Compares one clip's events (all samples in fed-stream coordinates; `lead` is subtracted from the events reported).
/// `gate_frames` is `Some(FW_GATE_MAX_FRAMES + longest lock)` for a set whose state libpryon leaves uninitialised:
/// then one-sided events, or equal-kind pairs with both ends, in `[S, S + gate_frames]` after a session end or an
/// agreed restart `S` are exempt, engine events inside `[S, S + FW_GATE_MAX_FRAMES)` that libpryon lacks are dropped,
/// and events after the first restart the engines disagree on are not compared. Nothing else is excused.
#[allow(clippy::too_many_arguments)]
pub fn compare_gated(
    oracle: &[OracleEvent],
    engine: &[Ev],
    oracle_restarts: &[u64],
    engine_restarts: &[u64],
    session_ends: &[u64],
    gate_frames: Option<u64>,
    lead: u64,
    limits: &Limits,
    exact: bool,
) -> Gated {
    let mut oracle = oracle.to_vec();
    let mut engine = engine.to_vec();
    let mut gate = GateTally::default();
    let agree = oracle_restarts.iter().zip(engine_restarts).take_while(|(a, b)| a == b).count();
    let (o_next, e_next) = (oracle_restarts.get(agree).copied(), engine_restarts.get(agree).copied());
    let mut restart_mismatch = None;
    let cut = match (o_next, e_next) {
        (None, None) => u64::MAX,
        (a, b) if gate_frames.is_none() => {
            restart_mismatch = Some((a, b));
            u64::MAX
        }
        (a, b) => a.unwrap_or(u64::MAX).min(b.unwrap_or(u64::MAX)),
    };
    let n = oracle.len() + engine.len();
    oracle.retain(|e| e.end < cut);
    engine.retain(|e| e.end < cut);
    gate.after_restart = n - oracle.len() - engine.len();
    let mut starts = session_ends.to_vec();
    starts.extend_from_slice(&oracle_restarts[..agree]);
    let mut windows = Vec::new();
    if let Some(frames) = gate_frames {
        let inside = FW_GATE_MAX_FRAMES * 160;
        let n = engine.len();
        // an engine event libpryon also has (same kind and end) is still compared in full
        let twin = |e: &Ev| oracle.iter().any(|o| o.keyword == e.keyword && o.kind == e.kind && o.end == e.end);
        engine.retain(|e| twin(e) || !starts.iter().any(|&s| e.end > s && e.end < s + inside));
        gate.inside = n - engine.len();
        windows = starts.iter().map(|&s| (s.saturating_sub(lead), (s + frames * 160).saturating_sub(lead))).collect();
    }
    for e in &mut oracle {
        (e.begin, e.end) = (e.begin.saturating_sub(lead), e.end.saturating_sub(lead));
    }
    for e in &mut engine {
        (e.begin, e.end) = (e.begin.saturating_sub(lead), e.end.saturating_sub(lead));
    }
    let mut all = Vec::new();
    let tally = compare(&oracle, &engine, limits, exact, &mut all);
    let mut diffs = Vec::new();
    for d in all {
        match gate_exempt(&d, &windows) {
            Some(n) => gate.window += n,
            None => diffs.push(d),
        }
    }
    Gated { tally, diffs, gate, restart_mismatch }
}

/// Events a difference may set aside under the gate: a pair needs equal kinds and both ends in one window.
pub fn gate_exempt(d: &Diff, windows: &[(u64, u64)]) -> Option<usize> {
    let has = |&(a, b): &(u64, u64), end: u64| a <= end && end <= b;
    match (&d.oracle, &d.engine) {
        (Some(o), Some(e)) if o.kind == e.kind && windows.iter().any(|w| has(w, o.end) && has(w, e.end)) => Some(2),
        (Some(x), None) | (None, Some(x)) if windows.iter().any(|w| has(w, x.end)) => Some(1),
        _ => None,
    }
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
    fn fmt_g_is_c_percent_g() {
        let cases: [(f32, &str); 12] = [
            (0.99, "0.99"),
            (0.633601, "0.633601"),
            (1.0, "1"),
            (0.045, "0.045"),
            (0.0009765625, "0.000976562"),
            (6.103515625e-05, "6.10352e-05"),
            (123456.7, "123457"),
            (1234567.0, "1.23457e+06"),
            (0.5, "0.5"),
            (0.9999996, "1"),
            (0.00012345678, "0.000123457"),
            (9.1552734375e-05, "9.15527e-05"),
        ];
        for (x, want) in cases {
            assert_eq!(fmt_g(x), want, "{x:e}");
        }
    }

    #[test]
    fn exact_scores_are_strings() {
        let o = OracleEvent {
            keyword: "ALEXA".into(),
            kind: OracleKind::Accept,
            score: "0.045".into(),
            threshold: "0.9".into(),
            begin: 100,
            end: 2000,
            afe: None,
        };
        let e = |score| Ev { keyword: "ALEXA".into(), kind: OracleKind::Accept, score, begin: 100, end: 2000 };
        let run = |s, exact| compare(&[o.clone()], &[e(s)], &Limits::default(), exact, &mut Vec::new()).matched;
        assert_eq!(run(0.045, true), 1);
        // 0.0450001 prints as "0.0450001": a 6-decimal rule would call it equal
        assert_eq!(run(0.0450001, true), 0);
    }

    fn ev(kind: OracleKind, end: u64) -> Option<Ev> {
        Some(Ev { keyword: "ALEXA".into(), kind, score: 0.9, begin: end - 8000, end })
    }

    #[test]
    fn gate_exempt_pairs() {
        let pair = |ok, ek, oe, ee| Diff { keyword: "ALEXA".into(), oracle: ev(ok, oe), engine: ev(ek, ee) };
        let w = [(10_000, 50_000), (60_000, 90_000)];
        let (a, n) = (OracleKind::Accept, OracleKind::NearMiss);
        assert_eq!(gate_exempt(&pair(a, a, 36_880, 37_840), &w), Some(2));
        assert_eq!(gate_exempt(&pair(a, n, 36_880, 37_840), &w), None);
        assert_eq!(gate_exempt(&pair(a, a, 36_880, 37_840), &[]), None);
        assert_eq!(gate_exempt(&pair(a, a, 45_000, 65_000), &w), None);
        // a score-only difference outside every window is never excused
        assert_eq!(gate_exempt(&pair(a, a, 20_000_000, 20_000_000), &w), None);
        let one = Diff { keyword: "ALEXA".into(), oracle: None, engine: ev(n, 70_000) };
        assert_eq!(gate_exempt(&one, &w), Some(1));
        assert_eq!(gate_exempt(&one, &[]), None);
    }

    #[test]
    fn gated_comparison_counts_each_allowance() {
        let oe = |end| OracleEvent {
            keyword: "ALEXA".into(),
            kind: OracleKind::Accept,
            score: "0.9".into(),
            threshold: "0.5".into(),
            begin: end - 8000,
            end,
            afe: None,
        };
        let e = |end| ev(OracleKind::Accept, end).unwrap();
        let lim = Limits::default();
        let gate = Some(FW_GATE_MAX_FRAMES + 50);
        // session end at 100 000: the engine's 110 000 lies inside the gate, the oracle's 130 000 in the window
        let oracle = [oe(50_000), oe(130_000), oe(900_000)];
        let engine = [e(50_000), e(110_000), e(950_000)];
        let g = compare_gated(&oracle, &engine, &[800_000], &[850_000], &[100_000], gate, 0, &lim, true);
        assert_eq!(g.gate, GateTally { window: 1, inside: 1, after_restart: 2 });
        assert!(g.diffs.is_empty() && g.restart_mismatch.is_none());
        assert_eq!(g.tally.matched, 1);
        // without the gate every one of them is a difference and the restart disagreement is reported
        let g = compare_gated(&oracle, &engine, &[800_000], &[850_000], &[100_000], None, 0, &lim, true);
        assert_eq!((g.gate.total(), g.diffs.len(), g.restart_mismatch), (0, 4, Some((Some(800_000), Some(850_000)))));
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
