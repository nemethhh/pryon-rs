//! `SoundModel` (config + shared weights) and `Net`, the per-detector network state.

use crate::mlp::{AffineParams, Mlp};
use crate::num::{dot_seq, sigmoid};
use crate::qlstm::QLstm;
use crate::{Error, SoundConfig};
use std::path::Path;
use std::sync::Arc;

pub(crate) const NUM_SCORES: usize = 12;
const HEADS: usize = 9;
const FEATURES: usize = 20;

struct Readout {
    heads: Vec<AffineParams>,
    route: AffineParams,
}

fn shape_err<T>(m: String) -> Result<T, Error> {
    Err(Error::Format(m))
}

/// A loaded model directory: configuration plus network weights, shared (`Arc`) by any number of detectors.
pub struct SoundModel {
    pub(crate) config: SoundConfig,
    trunk: QLstm,
    heads: Vec<QLstm>,
    readout: Arc<Readout>,
}

impl SoundModel {
    /// Reads `pryon.config`, `AED.json` and the `.mlp` from `dir`; every failure is an `Err`.
    pub fn load(dir: &Path) -> Result<SoundModel, Error> {
        let config = SoundConfig::load(dir)?;
        let path = &config.mlp_path;
        let bytes = std::fs::read(path).map_err(|source| Error::Io { path: path.clone(), source })?;
        let mlp = Mlp::parse(bytes)?;
        let trunk = QLstm::new(mlp.qlstm("Hlstm0_layer_0")?);
        if trunk.input() != FEATURES {
            return shape_err(format!("trunk input {}, expected {FEATURES}", trunk.input()));
        }
        let (mut heads, mut affines) = (Vec::with_capacity(HEADS), Vec::with_capacity(HEADS));
        for j in 1..=HEADS {
            let head = QLstm::new(mlp.qlstm(&format!("Hlstm{j}_layer_0"))?);
            if head.input() != trunk.cell() {
                return shape_err(format!("Hlstm{j} input {}, expected {}", head.input(), trunk.cell()));
            }
            let a = mlp.affine(&format!("Affine{j}"))?;
            if (a.rows, a.cols) != (1, head.cell()) {
                return shape_err(format!("Affine{j} {}x{}, expected 1x{}", a.rows, a.cols, head.cell()));
            }
            heads.push(head);
            affines.push(a);
        }
        let route = mlp.affine("Affine")?;
        if (route.rows, route.cols) != (NUM_SCORES, HEADS) {
            return shape_err(format!("Affine {}x{}, expected {NUM_SCORES}x{HEADS}", route.rows, route.cols));
        }
        if config.types.len() != route.rows {
            return shape_err(format!("{} types for {} routing outputs", config.types.len(), route.rows));
        }
        if let Some(t) = config.types.iter().find(|t| t.output_index >= route.rows) {
            return shape_err(format!("{}: output index {} >= {}", t.name, t.output_index, route.rows));
        }
        let readout = Arc::new(Readout { heads: affines, route });
        Ok(SoundModel { config, trunk, heads, readout })
    }

    /// All sound types of the model, in `AED.json` order.
    pub fn type_names(&self) -> Vec<&str> {
        self.config.types.iter().map(|t| t.name.as_str()).collect()
    }
}

pub(crate) struct Net {
    trunk: QLstm,
    heads: Vec<QLstm>,
    readout: Arc<Readout>,
    head_out: [f32; HEADS],
}

impl Net {
    pub fn new(m: &SoundModel) -> Net {
        let mut net = Net {
            trunk: m.trunk.clone(),
            heads: m.heads.clone(),
            readout: Arc::clone(&m.readout),
            head_out: [0.0; HEADS],
        };
        net.reset();
        net
    }

    pub fn reset(&mut self) {
        self.trunk.reset();
        self.heads.iter_mut().for_each(QLstm::reset);
    }

    /// One normalised frame in, `sigmoid` of every routing output out (independent of enabled types).
    pub fn step(&mut self, x: &[f32; FEATURES], scores: &mut [f32; NUM_SCORES]) {
        let Net { trunk, heads, readout, head_out } = self;
        let h = trunk.step(x);
        // Float affines are OpenBLAS sgemv/sgemm (0xAEB550 -> 0x51EEE0 / 0x51FC20): bias + in-order dot.
        for ((head, a), out) in heads.iter_mut().zip(&readout.heads).zip(head_out.iter_mut()) {
            *out = a.b[0] + dot_seq(&a.w, head.step(h));
        }
        let r = &readout.route;
        for ((s, row), b) in scores.iter_mut().zip(r.w.chunks_exact(HEADS)).zip(&r.b) {
            *s = sigmoid(b + dot_seq(row, head_out));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qlstm::tests::{fnv, inputs};
    use pryon_testkit::{write_mlp, write_model_dir, TempDir, TYPE_NAMES};

    // Android's std keeps thread_local in heap-backed OS TLS, which recurses into this allocator.
    #[cfg(not(target_os = "android"))]
    use std::{
        alloc::{GlobalAlloc, Layout, System},
        cell::Cell,
    };

    #[cfg(not(target_os = "android"))]
    struct CountingAlloc;

    #[cfg(not(target_os = "android"))]
    thread_local! {
        static ALLOCS: Cell<usize> = const { Cell::new(0) };
    }

    #[cfg(not(target_os = "android"))]
    unsafe impl GlobalAlloc for CountingAlloc {
        unsafe fn alloc(&self, l: Layout) -> *mut u8 {
            ALLOCS.with(|a| a.set(a.get() + 1));
            unsafe { System.alloc(l) }
        }
        unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
            unsafe { System.dealloc(p, l) }
        }
    }

    #[cfg(not(target_os = "android"))]
    #[global_allocator]
    static GLOBAL: CountingAlloc = CountingAlloc;

    #[cfg(not(target_os = "android"))]
    fn allocs() -> usize {
        ALLOCS.with(Cell::get)
    }

    fn model_dir(tag: &str) -> TempDir {
        let d = TempDir::new(&format!("model-{tag}"));
        write_model_dir(&d, 1, 0.8);
        d
    }

    fn run(net: &mut Net, xs: &[[f32; 20]]) -> Vec<u32> {
        let mut out = Vec::new();
        let mut s = [0f32; NUM_SCORES];
        for x in xs {
            net.step(x, &mut s);
            out.extend(s.iter().map(|v| v.to_bits()));
        }
        out
    }

    #[test]
    fn load_reads_types() {
        let m = SoundModel::load(&model_dir("types")).unwrap();
        assert_eq!(m.type_names(), TYPE_NAMES);
    }

    #[cfg(not(target_os = "android"))]
    #[test]
    fn net_step_no_alloc_shapes() {
        let m = SoundModel::load(&model_dir("alloc")).unwrap();
        let mut net = Net::new(&m);
        let xs = inputs();
        let mut s = [0f32; NUM_SCORES];
        let before = allocs();
        assert!(before > 0, "allocation counter not live");
        for x in &xs {
            net.step(x, &mut s);
            assert!(s.iter().all(|&v| v > 0.0 && v < 1.0), "{s:?}");
        }
        assert_eq!(allocs(), before, "Net::step allocated");
    }

    #[test]
    fn net_matches_ref() {
        let m = SoundModel::load(&model_dir("ref")).unwrap();
        let got = run(&mut Net::new(&m), &inputs());
        let want_last = [
            0x3E850A6Au32, 0x3E32E5A6, 0x3EF3E1C9, 0x3DA82075, 0x3F5B9132, 0x3E9DD88D, 0x3EC8429F, 0x3F36DE27,
            0x3F0E0A1C, 0x3E32E5A6, 0x3E32E5A6, 0x3EC8429F,
        ];
        assert_eq!(got[got.len() - NUM_SCORES..], want_last);
        assert_eq!(fnv(got), 309673595515850512);
    }

    #[test]
    fn reset_restores() {
        let m = SoundModel::load(&model_dir("reset")).unwrap();
        let mut net = Net::new(&m);
        let xs = inputs();
        let a = run(&mut net, &xs);
        net.reset();
        assert_eq!(run(&mut net, &xs), a);
        assert_eq!(run(&mut Net::new(&m), &xs), a, "nets share weights, not state");
    }

    #[test]
    fn load_shape_errors() {
        let d = model_dir("shapes");
        let mlp = d.join("model_gcmvn.cpu.v1.q.mlp");
        for bad in [write_mlp((16, 64), 9, 96, 1), write_mlp((20, 64), 8, 96, 1), b"junk".to_vec()] {
            std::fs::write(&mlp, bad).unwrap();
            assert!(matches!(SoundModel::load(&d), Err(Error::Format(_))));
        }
        std::fs::remove_file(&mlp).unwrap();
        assert!(matches!(SoundModel::load(&d), Err(Error::Io { .. })));
    }
}
