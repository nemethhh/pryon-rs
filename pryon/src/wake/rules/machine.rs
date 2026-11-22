//! op.cfg.json's operating state (keyword_spotter_machine.cpp): a state enables the spotters of the keywords its
//! rules name; an accept of such a keyword moves to the rule's next state, and a state's timeout counts frames.

use crate::wake::kwcfg::MachineSpec;

struct St {
    /// Keywords the state's rules name (bit = keyword index).
    mask: u64,
    /// (keyword, next state).
    rules: Vec<(usize, usize)>,
    /// (frames, next state); 0 frames never times out.
    timeout: (u32, usize),
}

pub(crate) struct Machine {
    states: Vec<St>,
    initial: usize,
    cur: usize,
    /// Frames until the current state times out; 0 = no timeout pending.
    left: u32,
}

impl Machine {
    /// No op.cfg: every keyword enabled, no transitions.
    pub fn always(keywords: usize) -> Machine {
        let mask = if keywords >= 64 { u64::MAX } else { (1u64 << keywords) - 1 };
        Machine { states: vec![St { mask, rules: Vec::new(), timeout: (0, 0) }], initial: 0, cur: 0, left: 0 }
    }

    /// `keywords` indexes the masks; the loader has checked every rule names one of them.
    pub fn new(spec: &MachineSpec, keywords: &[String]) -> Machine {
        let states = spec
            .states
            .iter()
            .map(|s| {
                let rules: Vec<(usize, usize)> = s
                    .rules
                    .iter()
                    .filter_map(|(k, next)| keywords.iter().position(|w| w == k).map(|i| (i, *next)))
                    .collect();
                let mask = rules.iter().fold(0u64, |m, &(k, _)| m | 1 << k);
                St { mask, rules, timeout: s.timeout.unwrap_or((0, spec.initial)) }
            })
            .collect();
        let mut m = Machine { states, initial: spec.initial, cur: spec.initial, left: 0 };
        m.reset();
        m
    }

    pub fn allows(&self, keyword: usize) -> bool {
        self.states[self.cur].mask >> keyword & 1 == 1
    }

    /// An accept (or auxiliary accept) of `keyword`; returns the keywords this enables.
    pub fn on_accept(&mut self, keyword: usize) -> u64 {
        // rules are looked up in order, the first one naming the keyword wins (0x9bd7f8)
        match self.states[self.cur].rules.iter().find(|&&(k, _)| k == keyword) {
            Some(&(_, next)) => self.go(next),
            None => 0,
        }
    }

    /// Start of a frame, before any spotter runs; returns the keywords a timeout enables.
    pub fn tick(&mut self) -> u64 {
        if self.left == 0 {
            return 0;
        }
        self.left -= 1;
        if self.left > 0 {
            return 0;
        }
        self.go(self.states[self.cur].timeout.1)
    }

    /// Session end: back to the initial state.
    pub fn reset(&mut self) {
        self.go(self.initial);
    }

    /// Every transition, a state's own included, restarts its timeout (0x9bd8cc).
    fn go(&mut self, next: usize) -> u64 {
        let newly = self.states[next].mask & !self.states[self.cur].mask;
        self.cur = next;
        self.left = self.states[next].timeout.0;
        newly
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wake::kwcfg::MachineState;

    const ALEXA: usize = 0;
    const STOP: usize = 1;
    const DISNEY: usize = 2;

    /// The en-US op.cfg.json: sleep -> awake on ALEXA/HEY_DISNEY, awake -> sleep on STOP or after `frames`.
    fn en_us(frames: u32) -> Machine {
        let s = |n: &str| n.to_string();
        let spec = MachineSpec {
            initial: 1,
            states: vec![
                MachineState {
                    name: s("awake"),
                    rules: vec![(s("ALEXA"), 0), (s("HEY_DISNEY"), 0), (s("STOP"), 1)],
                    timeout: Some((frames, 1)),
                },
                MachineState { name: s("sleep"), rules: vec![(s("ALEXA"), 0), (s("HEY_DISNEY"), 0)], timeout: None },
            ],
        };
        Machine::new(&spec, &[s("ALEXA"), s("STOP"), s("HEY_DISNEY")])
    }

    #[test]
    fn stop_only_when_awake() {
        let mut m = en_us(175);
        assert!(m.allows(ALEXA) && m.allows(DISNEY) && !m.allows(STOP));
        assert_eq!(m.on_accept(ALEXA), 1 << STOP, "waking enables STOP");
        assert!(m.allows(STOP) && m.allows(ALEXA));
        assert_eq!(m.on_accept(ALEXA), 0, "awake -> awake enables nothing new");
        assert_eq!(m.on_accept(STOP), 0);
        assert!(!m.allows(STOP) && m.allows(ALEXA));
        assert_eq!(m.on_accept(DISNEY), 1 << STOP, "HEY_DISNEY wakes too");
        m.reset();
        assert!(!m.allows(STOP), "session end returns to the initial state");
    }

    #[test]
    fn awake_times_out() {
        // timeout 175 frames: an accept in frame f keeps STOP through frame f + 174 (to_168/to_169 probes)
        let mut m = en_us(175);
        m.on_accept(ALEXA);
        for _ in 0..174 {
            assert_eq!(m.tick(), 0);
            assert!(m.allows(STOP));
        }
        assert_eq!(m.tick(), 0);
        assert!(!m.allows(STOP), "frame f + 175 starts asleep");
        // a self-transition restarts the count (to_selfreset probe)
        m.on_accept(ALEXA);
        for _ in 0..100 {
            m.tick();
        }
        m.on_accept(ALEXA);
        for _ in 0..174 {
            m.tick();
        }
        assert!(m.allows(STOP));
        m.tick();
        assert!(!m.allows(STOP));
        // asleep nothing counts down
        for _ in 0..1000 {
            assert_eq!(m.tick(), 0);
        }
        assert!(m.allows(ALEXA) && !m.allows(STOP));
    }

    #[test]
    fn zero_duration_never_times_out() {
        let mut m = en_us(0);
        m.on_accept(ALEXA);
        for _ in 0..10_000 {
            m.tick();
        }
        assert!(m.allows(STOP));
    }

    #[test]
    fn without_op_cfg_everything_runs() {
        let mut m = Machine::always(3);
        assert!((0..3).all(|k| m.allows(k)));
        assert_eq!(m.on_accept(STOP), 0);
        assert_eq!(m.tick(), 0);
        assert!((0..3).all(|k| m.allows(k)));
    }
}
