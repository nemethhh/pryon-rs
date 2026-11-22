//! Deterministic, integer-only synthetic clips (16 kHz mono s16).

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Clip {
    Noise,
    Tones,
    Gap49,
    Sparse,
    Zeros,
    Quiet,
    Extremes,
}

impl Clip {
    pub const ALL: [Clip; 7] = [
        Clip::Noise,
        Clip::Tones,
        Clip::Gap49,
        Clip::Sparse,
        Clip::Zeros,
        Clip::Quiet,
        Clip::Extremes,
    ];

    pub fn name(&self) -> &'static str {
        match self {
            Clip::Noise => "noise",
            Clip::Tones => "tones",
            Clip::Gap49 => "gap49",
            Clip::Sparse => "sparse",
            Clip::Zeros => "zeros",
            Clip::Quiet => "quiet",
            Clip::Extremes => "extremes",
        }
    }

    pub fn from_name(name: &str) -> Option<Clip> {
        Clip::ALL.into_iter().find(|c| c.name() == name)
    }
}

/// xorshift64*.
pub struct Rng(pub u64);

impl Rng {
    pub fn next_u32(&mut self) -> u32 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        (self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 32) as u32
    }
}

const LEN: usize = 400_000;
const SHORT_LEN: usize = 320_000;

// Never returns 0, so only intended runs are digital silence.
fn noise_sample(rng: &mut Rng, amp: i32) -> i16 {
    let v = (rng.next_u32() % (2 * amp as u32 + 1)) as i32 - amp;
    if v == 0 { 1 } else { v as i16 }
}

fn noise(seed: u64, n: usize, amp: i32) -> Vec<i16> {
    let mut rng = Rng(seed);
    (0..n).map(|_| noise_sample(&mut rng, amp)).collect()
}

pub fn synth(clip: Clip) -> Vec<i16> {
    match clip {
        Clip::Noise => noise(0x9E37_79B9_7F4A_7C15, LEN, 3000),
        Clip::Tones => {
            let mut x = noise(0x1234_5678_9ABC_DEF1, LEN, 300);
            let step = ((3100u64 << 32) / 16000) as u32;
            let mut phase = 0u32;
            for (i, s) in x.iter_mut().enumerate() {
                if (i / 8000) % 2 == 0 {
                    let a = if phase < 1 << 31 { 8000 } else { -8000 };
                    *s += a;
                    phase = phase.wrapping_add(step);
                }
            }
            x
        }
        Clip::Gap49 => {
            let mut x = noise(0x0BAD_5EED_0000_0003, LEN, 3000);
            x[4000..11760].fill(0);
            x
        }
        Clip::Sparse => {
            let mut x = vec![0i16; SHORT_LEN];
            let mut rng = Rng(0x5A5A_1234_DEAD_BEE5);
            for j in 0..SHORT_LEN / 3200 {
                for s in &mut x[3200 * j..3200 * j + 160] {
                    *s = noise_sample(&mut rng, 3000);
                }
            }
            x
        }
        Clip::Zeros => vec![0; SHORT_LEN],
        Clip::Quiet => noise(0x7777_1111_2222_3333, LEN, 60),
        Clip::Extremes => {
            let mut x = noise(0x9E37_79B9_7F4A_7C15, LEN, 3000);
            for (i, s) in x.iter_mut().enumerate() {
                if (i + 1) % 1009 == 0 {
                    *s = i16::MAX;
                }
                if (i + 1) % 997 == 0 {
                    *s = i16::MIN;
                }
            }
            x
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fnv1a(bytes: impl Iterator<Item = u8>) -> u64 {
        bytes.fold(0xcbf2_9ce4_8422_2325, |h, b| (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3))
    }

    // Frame f = samples 160f..160f+400; silent iff it holds >= 200 contiguous zeros.
    fn silent(x: &[i16], f: usize) -> bool {
        let (mut run, mut best) = (0, 0);
        for &s in &x[160 * f..160 * f + 400] {
            run = if s == 0 { run + 1 } else { 0 };
            best = best.max(run);
        }
        best >= 200
    }

    fn frames(x: &[i16]) -> usize {
        (x.len() - 400) / 160 + 1
    }

    #[test]
    fn synth_is_deterministic() {
        for c in Clip::ALL {
            assert_eq!(synth(c), synth(c));
        }
        let h = fnv1a(synth(Clip::Noise).iter().flat_map(|s| s.to_le_bytes()));
        assert_eq!(h, 18236115563368143106, "pinned FNV-1a of synth(Noise)");
    }

    #[test]
    fn lengths() {
        for c in Clip::ALL {
            let want = if matches!(c, Clip::Sparse | Clip::Zeros) { 320_000 } else { 400_000 };
            assert_eq!(synth(c).len(), want, "{}", c.name());
        }
    }

    #[test]
    fn names_round_trip() {
        for c in Clip::ALL {
            assert_eq!(Clip::from_name(c.name()), Some(c));
        }
        assert_eq!(Clip::from_name("nope"), None);
    }

    #[test]
    fn zeros_all_zero() {
        assert!(synth(Clip::Zeros).iter().all(|&s| s == 0));
    }

    #[test]
    fn quiet_below_gate() {
        let m = synth(Clip::Quiet).iter().map(|&s| (s as i32).abs()).max().unwrap();
        assert!(m <= 60 && m > 0);
    }

    #[test]
    fn extremes_hits_min() {
        let x = synth(Clip::Extremes);
        assert!(x.contains(&i16::MIN) && x.contains(&i16::MAX));
    }

    #[test]
    fn gap49_run() {
        let x = synth(Clip::Gap49);
        let s: Vec<usize> = (0..frames(&x)).filter(|&f| silent(&x, f)).collect();
        assert_eq!(s, (24..=72).collect::<Vec<_>>());
        assert_eq!(s.len(), 49);
    }

    #[test]
    fn sparse_frames() {
        let x = synth(Clip::Sparse);
        assert!(frames(&x) >= 1996);
        for w in [0..998, 998..1996] {
            let (mut non_silent, mut run, mut best) = (0, 0, 0);
            for f in w {
                if silent(&x, f) {
                    run += 1;
                    best = best.max(run);
                } else {
                    non_silent += 1;
                    run = 0;
                }
            }
            assert!(non_silent < 100, "{non_silent}");
            assert!(best < 49, "{best}");
        }
    }

    #[test]
    fn no_stray_zeros() {
        for c in [Clip::Noise, Clip::Tones, Clip::Quiet, Clip::Extremes] {
            assert!(!synth(c).contains(&0), "{}", c.name());
        }
        let g = synth(Clip::Gap49);
        let zeros: Vec<usize> = (0..g.len()).filter(|&i| g[i] == 0).collect();
        assert_eq!(zeros, (4000..11760).collect::<Vec<_>>());
    }
}
