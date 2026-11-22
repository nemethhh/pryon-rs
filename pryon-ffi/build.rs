fn main() {
    for k in ["HOST", "TARGET"] {
        println!("cargo:rustc-env=PRYON_FFI_{k}={}", std::env::var(k).unwrap());
    }
}
