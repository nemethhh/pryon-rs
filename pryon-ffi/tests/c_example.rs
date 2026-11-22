use std::path::PathBuf;
use std::process::Command;

// Profile dir of this test binary (target/<profile>), robust to CARGO_TARGET_DIR.
fn profile_dir() -> PathBuf {
    let exe = std::env::current_exe().unwrap();
    exe.parent().unwrap().parent().unwrap().to_path_buf()
}

#[test]
fn c_example_runs() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let prof = profile_dir();
    // Integration tests only link the rlib; build the cdylib (and keep it fresh).
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".into());
    let mut c = Command::new(cargo);
    c.args(["build", "-q", "-p", "pryon-ffi", "--lib"]).current_dir(&root);
    if prof.file_name().unwrap() == "release" {
        c.arg("--release");
    }
    assert!(c.status().unwrap().success(), "cargo build failed");
    let work = std::env::temp_dir().join(format!("pryon-c-example-{}", std::process::id()));
    let model = work.join("model");
    std::fs::create_dir_all(&model).unwrap();
    pryon_testkit::write_model_dir(&model, 7, 0.5);
    let mut pcm: Vec<i16> = Vec::new();
    let noise = pryon_testkit::synth(pryon_testkit::Clip::Noise);
    while pcm.len() < 2 * 159_680 + 240 {
        pcm.extend_from_slice(&noise);
    }
    pcm.truncate(2 * 159_680 + 240);
    let clip = work.join("noise.s16");
    std::fs::write(&clip, pcm.iter().flat_map(|s| s.to_le_bytes()).collect::<Vec<u8>>()).unwrap();

    let exe = work.join("sound");
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".into());
    let st = Command::new(cc)
        .args(["-std=c11", "-Wall", "-Werror", "-I"])
        .arg(root.join("include"))
        .arg(root.join("examples/sound.c"))
        .arg("-L")
        .arg(&prof)
        .arg("-lpryon_ffi")
        .arg(format!("-Wl,-rpath,{}", prof.display()))
        .arg("-o")
        .arg(&exe)
        .status()
        .expect("system C compiler");
    assert!(st.success(), "C compile failed");

    let out = Command::new(&exe).arg(&model).arg(&clip).output().unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "exit {:?}: {}", out.status, String::from_utf8_lossy(&out.stderr));
    let lines: Vec<&str> = stdout.lines().filter(|l| l.starts_with("window ")).collect();
    assert_eq!(lines.len(), 2, "{stdout}");
    assert!(lines[0].starts_with("window 0 end 159680 "), "{}", lines[0]);
    assert!(lines[1].starts_with("window 1 end 319360 "), "{}", lines[1]);
    let _ = std::fs::remove_dir_all(&work);
}
