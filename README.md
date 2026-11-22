# pryon-rs

Rust replacement for Amazon's libpryon Guard (AED) sound detector on the Echo Dot.
Local repo, not published.

## Crates

- `pryon`: the engine (no runtime dependencies).
- `pryon-ffi`: C API (staticlib/cdylib).
- `pryon-cli`: command-line tools: `run` (print windows of a clip), `synth` (write a synthetic clip), `synth-model`
  (write a synthetic model dir), `bench [--chunk N] [--speed X]` (CPU and RSS over a clip), `rusage` (run a command, report
  its CPU and RSS).
- `pryon-testkit`: deterministic synthetic clips and test models.

## Using the library

Rust (`pryon = { path = "pryon-rs/pryon" }`):

```rust
use pryon::{SoundDetector, SoundModel};
use std::{path::Path, sync::Arc};

let model = Arc::new(SoundModel::load(Path::new("/path/to/model"))?);
let mut det = SoundDetector::new(model, &[])?; // empty list: every type
let mut windows = Vec::new();
det.feed(&pcm_16khz_mono_i16, &mut windows); // any chunking; call repeatedly
for w in &windows {
    for (name, t) in det.type_names().iter().zip(&w.types) {
        if t.detected { println!("window {} {name} {:.3}", w.index, t.score); }
    }
}
```

C: include `pryon-ffi/include/pryon.h` and link the staticlib (`cargo build --release -p pryon-ffi`,
`target/release/libpryon_ffi.a`) with the native libs rustc reports for the target
(`cargo rustc -p pryon-ffi --release --crate-type staticlib -- --print native-static-libs`):

    cc app.c libpryon_ffi.a -lgcc_s -lutil -lrt -lpthread -lm -ldl -lc        # x86_64 Linux
    armv7a-linux-androideabi24-clang app.c libpryon_ffi.a -ldl -llog -lunwind -lm -lc   # Echo, NDK libunwind.a

Host builds do not flush subnormals to zero while NEON does; host parity holds on every fixture, device parity is the
device check.

## Tests

    cargo test --workspace

Tests that need Amazon models or local clips skip (`skipped: set VAR`) unless set:

- `PRYON_AED_MODEL_DIR`: firmware AED model (`echo-dot-assist/firmware/donut/rootfs/system/local/models/AED`).
- `PRYON_AED_DAVS_DIR`: DAVS AED model (`pypryon/build/aed/model`).
- `PRYON_CLIPS_DIR`: local clips; `local_clips` checks every `<name>.raw` against `<name>_out.txt` (DAVS) and
  `<name>_fw_out.txt` (firmware).
- `PRYON_ORACLE_OVERRIDE_DIR`: read the oracle outputs from this dir (same `{fw,davs}/<clip>.txt` layout) instead of the
  committed fixtures.

## Building for the Echo

armv7 needs NDK r21e (`armv7a-linux-androideabi24-clang`, plus its `libunwind.a`) and the `armv7-linux-androideabi` rustup
target. `.cargo/config.toml` carries the target flags, but `CARGO_TARGET_*_RUSTFLAGS` replaces them, so the env below
repeats them (and `--cfg pryon_neon`, which selects the NEON kernel; `target_feature = "neon"` is unstable on arm):

    NDK=/path/to/android-ndk-r21e
    export CARGO_TARGET_ARMV7_LINUX_ANDROIDEABI_LINKER=$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/armv7a-linux-androideabi24-clang
    export CARGO_TARGET_ARMV7_LINUX_ANDROIDEABI_RUSTFLAGS="-C target-feature=+neon -C target-cpu=cortex-a35 --cfg pryon_neon -C link-arg=-fuse-ld=lld -L native=$NDK/sources/cxx-stl/llvm-libc++/libs/armeabi-v7a"
    cargo build --release --target armv7-linux-androideabi -p pryon-ffi -p pryon-cli

The one remaining rustc warning (`unstable feature specified for -Ctarget-feature: neon`) is expected on stable; `+neon` is
required to assemble the asm (cortex-a35 alone does not enable it) and implies the d16-d31 registers; lld replaces gold.

Tests run on the host under qemu-arm with the Echo rootfs (the `unshare` is required: bionic hangs or crashes
under bare qemu-arm). `TMPDIR` is needed because Android's default temp dir does not exist on the host;
`ANDROID_DATA`/`ANDROID_ROOT` only silence bionic's tzdata warnings:

    ROOTFS=/path/to/echo-dot-assist/firmware/donut/rootfs
    export CARGO_TARGET_ARMV7_LINUX_ANDROIDEABI_RUNNER="unshare --user --map-root-user --pid --fork --kill-child qemu-arm -L $ROOTFS"
    export TMPDIR=/tmp ANDROID_DATA=/data ANDROID_ROOT=/system
    PRYON_AED_DAVS_DIR=... PRYON_AED_MODEL_DIR=... cargo test -p pryon --target armv7-linux-androideabi

`PRYON_CLIPS_DIR` is slow under qemu; leave it unset there.

## Oracle fixtures

`pryon/tests/fixtures/oracle/{fw,davs}/<clip>.txt` are libpryon's `aed_test -j` outputs on our own synthetic
clips (`pryon_testkit::synth`, written by `pryon-cli synth NAME CLIP.raw`), one file per clip and model:

    unshare --user --map-root-user --pid --fork --kill-child \
      qemu-arm -L ../echo-dot-assist/firmware/donut/rootfs ../pypryon/build/aed/tools/aed_test \
      -m MODEL/pryon.manifest -x 1000 -j CLIP.raw > CLIP.txt

Each must hold floor(len / 159680) windows; rerun with a lower `-x` if the library dropped audio.
`davs/bits/<clip>.txt` holds the DAVS scores of each window as 12 `%08x` words in `score_output_index` order, from
the Python reference recipe (C LFBE + numpy, kept outside the repo), checked against the oracle's printed scores.
The tests compare printed scores (`%.6f`), decisions, `maxEnergyLevel` and `numSamplesAboveEnergyThreshold`.
