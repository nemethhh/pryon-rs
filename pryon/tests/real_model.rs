use pryon::Mlp;

fn load(var: &str) -> Option<Mlp> {
    let Ok(dir) = std::env::var(var) else {
        println!("skipped: set {var}");
        return None;
    };
    let bytes = std::fs::read(std::path::Path::new(&dir).join("model_gcmvn.cpu.v1.q.mlp")).unwrap();
    Some(Mlp::parse(bytes).unwrap())
}

fn check(m: &Mlp) {
    let l0 = m.qlstm("Hlstm0_layer_0").unwrap();
    assert_eq!((l0.input, l0.cell), (20, 64));
    for i in 1..=9 {
        let l = m.qlstm(&format!("Hlstm{i}_layer_0")).unwrap();
        assert_eq!((l.input, l.cell), (64, 96), "Hlstm{i}");
        let a = m.affine(&format!("Affine{i}")).unwrap();
        assert_eq!((a.rows, a.cols, a.b.len()), (1, 96, 1), "Affine{i}");
    }
    let r = m.affine("Affine").unwrap();
    assert_eq!((r.rows, r.cols), (12, 9));
    for row in r.w.chunks(9) {
        assert_eq!(row.iter().filter(|&&x| x == 1.0).count(), 1);
        assert!(row.iter().all(|&x| x == 0.0 || x == 1.0));
    }
}

#[test]
fn real_shapes() {
    for var in ["PRYON_AED_DAVS_DIR", "PRYON_AED_MODEL_DIR"] {
        if let Some(m) = load(var) {
            check(&m);
        }
    }
}
