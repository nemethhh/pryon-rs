use pryon::lfbe::{Lfbe, BINS};

fn pcm(name: &str) -> Vec<i16> {
    let p = format!("{}/tests/fixtures/lfbe/{name}.raw", env!("CARGO_MANIFEST_DIR"));
    let b = std::fs::read(p).unwrap();
    b.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]])).collect()
}

fn reference(name: &str) -> Vec<u32> {
    let p = format!("{}/tests/fixtures/lfbe/{name}.f32", env!("CARGO_MANIFEST_DIR"));
    let b = std::fs::read(p).unwrap();
    b.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn run(x: &[i16], chunk: usize) -> Vec<[f32; BINS]> {
    let mut l = Lfbe::new();
    let mut out = Vec::new();
    for c in x.chunks(chunk) {
        l.push(c, &mut out);
    }
    out
}

#[test]
fn lfbe_matches_c_reference() {
    for name in ["noise", "extremes", "gap49"] {
        let frames = run(&pcm(name), 48000);
        let want = reference(name);
        assert_eq!(frames.len(), 298, "{name}");
        assert_eq!(want.len(), 298 * BINS);
        for (f, fr) in frames.iter().enumerate() {
            for (b, v) in fr.iter().enumerate() {
                assert_eq!(v.to_bits(), want[f * BINS + b], "{name} frame {f} bin {b}");
            }
        }
    }
}

#[test]
fn chunked_push_equal() {
    let x = pcm("noise");
    let whole = run(&x, x.len());
    for chunk in [1, 159, 401] {
        let got = run(&x, chunk);
        assert_eq!(got.len(), whole.len(), "chunk {chunk}");
        for (a, b) in got.iter().zip(&whole) {
            assert!(a.iter().zip(b).all(|(p, q)| p.to_bits() == q.to_bits()), "chunk {chunk}");
        }
    }
}

#[test]
fn lfbe_extremes_no_panic() {
    let frames = run(&vec![-32768i16; 400], 400);
    assert_eq!(frames.len(), 1);
    assert!(frames[0].iter().all(|v| v.is_finite()));
}

#[test]
fn zero_frame_is_zero() {
    let frames = run(&[0i16; 400], 400);
    assert_eq!(frames.len(), 1);
    assert!(frames[0].iter().all(|v| *v == 0.0));
}
