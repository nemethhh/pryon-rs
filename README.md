# pryon-rs

An open, native Rust reimplementation of the acoustic event detector in Amazon's `libpryon.so`, the engine behind
Alexa Guard's sound detection on Echo devices (smoke and CO alarms, glass breaking, dog barks, a baby crying, and
eight more sound types). It produces **the same scores, bit for bit, and the same decisions** as libpryon, at roughly
**half the CPU and a quarter of the memory** on an Echo Dot.

It is a plain library with a Rust and a C API: 16 kHz mono int16 audio in, one result per ~10 s window out. No
threads, no global state, no dependencies at run time. It was written to replace libpryon in
[hassmic](https://github.com/nemethhh/echo-dot-assist/tree/rust-sound), which turns stock Echos into Home Assistant
voice satellites, but nothing in it assumes hassmic, Home Assistant or an Echo.

Not affiliated with Amazon. This repository contains no Amazon code, models, firmware or recordings: you supply the
model from a device you own.

## Status

| Check | Result |
|---|---|
| Parity with libpryon on a PC (libpryon run under qemu-arm as the oracle) | identical scores (exact float bits) and decisions on every committed test clip and on 22 local real-speech clips, both model generations |
| Parity on the device | output of the Echo Dot 3 and Echo Dot 2 byte-identical to the PC on 26/26 clip × model runs |
| CPU, 164 s clip, fed in real time like libpryon | Echo Dot 3 (Cortex-A35): 21.0 % → 9.0 % of a core (2.3× less); Echo Dot 2 (Cortex-A53): 11.1 % → 5.3 % (2.1× less) |
| Peak memory | Echo Dot 3: 16.8 → 3.9 MB; Echo Dot 2: 11.0 → 2.9 MB |
| Model load | Echo Dot 3: ~85 ms; Echo Dot 2: ~60 ms |

Guard sound detection only, for now. Wake word and whisper detection are planned; they use other model formats.

## Models

A model is a directory with `pryon.config`, `AED.json` and `model_gcmvn.cpu.v1.q.mlp`. Two generations are
known, both supported:

- the firmware's, on every Fire OS Echo checked so far: `adb pull /system/local/models/AED`
- the newer one Amazon serves to devices (retrained weights, smoke/CO thresholds 0.845 instead of 0.825); hassmic's
  `scripts/artifacts.sh` installs it to `/data/local/hassmic/aed`

The type list and every threshold come from the model's `AED.json`.

## Use

### Rust

```toml
pryon = { git = "https://github.com/nemethhh/pryon-rs" }
```

```rust
use pryon::{SoundDetector, SoundModel};
use std::{path::Path, sync::Arc};

let model = Arc::new(SoundModel::load(Path::new("/path/to/AED"))?);
let mut det = SoundDetector::new(model, &[])?; // empty list: every type in AED.json
let mut windows = Vec::new();
det.feed(&pcm, &mut windows); // any chunk size; call as audio arrives
for w in &windows {
    for (name, t) in det.type_names().iter().zip(&w.types) {
        if t.detected { println!("window {} {name} {:.3}", w.index, t.score); }
    }
}
```

`cargo doc --open -p pryon` documents every public item.

### C

`pryon-ffi/include/pryon.h` documents the API (open, feed, type names, reset, close). Build the static library and
link it with the native libraries rustc lists for the target
(`cargo rustc -p pryon-ffi --release --crate-type staticlib -- --print native-static-libs`):

    cargo build --release -p pryon-ffi
    cc app.c target/release/libpryon_ffi.a -lgcc_s -lutil -lrt -lpthread -lm -ldl -lc         # x86_64 Linux
    armv7a-linux-androideabi24-clang app.c libpryon_ffi.a -ldl -llog -lunwind -lm -lc           # Echo (NDK r21e)

`pryon-ffi/examples/sound.c` is a complete program; `cargo test` builds and runs it.

### Command line

`pryon-cli`:

- `run -m MODEL [-t type,...] [--bits] CLIP.raw`: print each window's scores and decisions, in the layout of
  libpryon's test tool, so the two diff line by line
- `bench -m MODEL [--chunk N] [--speed X] CLIP.raw`: CPU per second of audio, worst single feed, peak memory;
  `--speed 1` feeds in real time
- `rusage -- CMD ...`: run a command and report its CPU time and peak memory (to measure libpryon the same way)
- `validate [--oracle-rootfs DIR --pryon-test PATH] [--cache DIR] [--exact] [--records DIR] [--record KEYWORD]...
  [--speed X] [--jobs N] [--lead-ms MS] [--tail-ms MS] [-p NAME=VALUE[@SAMPLE]]... [-e SAMPLE]... MODEL CLIPS...`:
  run libpryon (`pryon_test` under qemu-arm; `PRYON_ORACLE_ROOTFS` and `PRYON_PRYON_TEST` stand in for the two
  flags) and the wake engine on the same `.raw` or 16 kHz mono `.wav` clips
  (files or directories) and print every difference as `DIFF <clip> <keyword> oracle=<event|-> rust=<event|->`, then
  a summary per keyword. Exit 0 clean, 1 differences, 2 usage or oracle errors. The oracle runs in real time
  (`--speed 1`, the default) because libpryon's results depend on feed timing at higher speeds, `--jobs` clips at a
  time. Both engines get each clip between `--lead-ms` (default 2500) and `--tail-ms` (default 1000) of digital
  silence: the firmware oracle scores nothing in its first ~1.7 s, and libpryon reports an accept only ~120 ms after
  the keyword ends, so not at the very end of a clip; events are printed in clip samples. That firmware gate comes
  from libpryon itself: the firmware graph's conv buffers have no stored values, and libpryon starts them from
  uninitialised heap memory (NaN, denormals, 1e12-size garbage) at the start, after every session end and after
  every decoder restart, so its scores are NaN until that garbage has left the network (168 or 174 frames in the
  installs measured; the garbage, and with it the length, follows the install path). The engine starts the buffers
  from zeros and does not reproduce the garbage. So on a set whose buffers libpryon leaves uninitialised (the
  firmware set; not the DAVS sets), and only there, validate sets aside and counts (`GATE` lines, `GATE ALLOWANCE`,
  the record's `gate_exempt`): one-sided events, or pairs of equal kind, whose ends lie within 174 frames plus the
  set's longest lock period after a session end or a restart both engines agree on; engine events inside the gate
  that libpryon lacks; and, when such an event was excused before it in the clip, the events after the first
  restart the two engines place differently (a gate-hidden accept moves the engine's restart point); without one,
  a restart placed differently is a difference. Everything else is a difference. `-p` sets a client
  property and `-e` ends the session (`Spotter::reset`, libpryon's `PryonDecoder_SessionEnd`) before the given clip
  sample, in both engines. libpryon's probabilistic accept (a random accept of a near miss, drawn from a
  clock-seeded generator) is switched off for the oracle only: it runs a copy of the set whose
  `probabilistic-accept-probability` values are 0 (symlinks plus the edited `kw.cfg.json`), which is the field's
  only effect; the engine never draws. When libpryon restarts its decoder after too many accepts, `pryon_test`
  creates a new one and goes on from the sample after the triggering frame, as the engine does. `--cache` keeps
  oracle outputs as `<set checksum>/<key>.txt`, the key covering the padded clip, the commands, the speed and the
  harness version. On a clean `--exact` run (every event's kind, begin and end equal, and the engine's score printed
  as `pryon_test` prints it, C `%g`, equal to the oracle's string) it writes `validated/<checksum>.<KEYWORD>.rec`
  for each keyword the oracle accepted at least once, also one accepted only on synthetic clips (HEY_DISNEY on the
  TTS clips); `--record KEYWORD` (a keyword the set serves) limits it to the keywords chosen: the set, the
  keyword, the clip count and digest, the padding, accept and near-miss counts, the events the gate allowance set
  aside, date and pryon-rs commit. A clip the oracle did not hear to the end is an error, so it never yields a
  record.
- `synth NAME OUT.raw`, `synth-model DIR`: synthetic test clips and a random-weight model in the real format

Clips are raw 16 kHz mono s16le.

## What a result means

- A window is 159 680 samples (998 frames of 10 ms, 9.98 s), counted from the first sample fed. Its result comes
  out of the `feed` call that delivers the 240th sample after the window's end. A trailing partial window gives
  nothing.
- Per type: `score` (0..1, from the last network step of the window), `detected` = score ≥ the type's detection
  threshold **and** enough loud samples (`samples_above` ≥ the type's minimum), `near_miss` = not detected and
  score ≥ the near-miss threshold (no loudness condition). `samples_above` counts samples with |x| ≥ the type's
  energy threshold; `max_energy` is the loudest sample in the window's frames (0..32767).
- A `feed` call can run about 50 network steps at once (when a window's normalisation statistics form, or at its
  end), so a real-time audio thread should hand audio to a worker thread instead of calling `feed` itself.

These rules were measured against libpryon's own output, including its quirks (digital silence, how −32768 is
counted).

## How it works

Audio → log mel filterbank, 20 bins, 25 ms frames every 10 ms (the same float operation order as libpryon's
Kaldi/Ne10 front end) → per-window mean normalisation with libpryon's handling of digital silence → an int8
dynamic-quantised LSTM trunk (20→64) and nine LSTM heads (64→96) on every other frame → nine float outputs routed to
the model's twelve types → sigmoid. Every float operation is rounded where libpryon rounds it: no fused
multiply-add, libpryon's exp and reciprocal approximations, its quantiser's rounding. On 32-bit ARM, NEON assembly
does the heavy parts. Each assembly kernel has a portable Rust twin and a test that the two agree, and the portable
code runs on every other target.

## Building for the Echo

The Echo's userland is 32-bit Android (API 24). You need NDK r21e (`armv7a-linux-androideabi24-clang` and its
`libunwind.a`) and the rustup target `armv7-linux-androideabi`. `.cargo/config.toml` holds the target flags. Setting
`CARGO_TARGET_*_RUSTFLAGS` replaces them, so the line below repeats them, including `--cfg pryon_neon`, which selects
the NEON kernels:

    NDK=/path/to/android-ndk-r21e
    export CARGO_TARGET_ARMV7_LINUX_ANDROIDEABI_LINKER=$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/armv7a-linux-androideabi24-clang
    export CARGO_TARGET_ARMV7_LINUX_ANDROIDEABI_RUSTFLAGS="-C target-feature=+neon -C target-cpu=cortex-a35 --cfg pryon_neon -C link-arg=-fuse-ld=lld -L native=$NDK/sources/cxx-stl/llvm-libc++/libs/armeabi-v7a"
    cargo build --release --target armv7-linux-androideabi -p pryon-ffi -p pryon-cli

rustc warns once per crate that `neon` is an unstable target feature. That warning is expected on stable Rust:
the NEON assembly needs the flag.

The armv7 tests run on a PC under qemu-arm, using an unpacked Fire OS root filesystem for the Android dynamic
linker. The `unshare` is needed because bionic hangs under bare qemu-arm:

    ROOTFS=/path/to/fire-os/rootfs
    export CARGO_TARGET_ARMV7_LINUX_ANDROIDEABI_RUNNER="unshare --user --map-root-user --pid --fork --kill-child qemu-arm -L $ROOTFS"
    export TMPDIR=/tmp ANDROID_DATA=/data ANDROID_ROOT=/system
    cargo test -p pryon --target armv7-linux-androideabi

## Tests

    cargo test --workspace

That runs everything that needs no Amazon files. Tests that need a model or local clips print `skipped: set VAR`
unless these are set:

- `PRYON_AED_MODEL_DIR`: the firmware model directory
- `PRYON_AED_DAVS_DIR`: the newer model directory
- `PRYON_CLIPS_DIR`: a directory of your own clips, each `<name>.raw` with libpryon's output beside it as
  `<name>_out.txt` (newer model) and/or `<name>_fw_out.txt` (firmware model)
- `PRYON_ORACLE_OVERRIDE_DIR`: compare against libpryon outputs in this directory (same `{fw,davs}/<clip>.txt`
  layout) instead of the committed ones, e.g. outputs produced on a device

### Oracle fixtures

`pryon/tests/fixtures/oracle/{fw,davs}/<clip>.txt` are libpryon's outputs on synthetic clips that the tests
regenerate from `pryon-testkit`, one file per clip and model. They were made with `aed_test`, the libpryon test
tool from [echo-dot-assist](https://github.com/Gamer92000/echo-dot-assist) (`src/tools/aed_test.c`), run under
qemu-arm against an Echo Dot 3 root filesystem:

    pryon-cli synth noise noise.raw
    unshare --user --map-root-user --pid --fork --kill-child \
      qemu-arm -L $ROOTFS aed_test -m MODEL/pryon.manifest -x 1000 -j noise.raw > noise.txt

Each file must have floor(samples / 159680) windows; if one has fewer, rerun with a lower `-x`, because the library
dropped audio. The tests compare the printed scores (6 decimals), the decisions, `maxEnergyLevel` and
`numSamplesAboveEnergyThreshold`. `davs/bits/<clip>.txt` adds the exact score bits of each window, from a Python
reference of the recipe that matches the printed scores.

On a PC, very small intermediate values are not flushed to zero, whereas NEON flushes them (as libpryon does on the
device). No fixture is affected. Run on the device to confirm parity there.

### Wake oracle fixtures

`pryon/tests/fixtures/wake/{fw,alexa}/<clip>.txt` are the wake events of `pryon_test` (echo-dot-assist
`src/tools/pryon_test.c`) for the firmware ALEXA set and alexa-en-US on the synthetic clips, run as above with
`-m SET/pryon.manifest -x 1000`. Their tests need `PRYON_WAKE_FW_DIR` (the firmware ALEXA directory) and
`PRYON_WAKE_SETS_DIR` (a directory holding `alexa-en-US/`).

`synth-{fw,davs}/<clip>.txt` are libpryon's events for the synthetic sets of `pryon-cli synth-wake-set fw|davs DIR
--force 0.99` on the same clips; they probe libpryon's keyword rules. They were run in real time (`-x 1`): a set
that fires on every step makes libpryon fall behind at `-x 1000`, and then the event times change from run to run.
`synth-variants/` and `synth-scripts/` are the same sets with single kw.cfg/op.cfg edits, client properties and
session ends at given samples (the probes of the keyword rules, the operating-state machine, HEY_DISNEY's
aggregator, the restart trigger); every one was run twice with identical events.
`synth-afe/` are such runs on clips carrying the Echo front end's metadata in their lowest bits (testkit
`lsb_stream`: clocks that wrap, corrupt frames, every metadata version libpryon parses, clean gaps; also the clean
`quiet_tones`, whose low bits libpryon reads as a frame without a clock); their tests compare
`SpotterOptions::echo_stream`'s `Detection::afe` with libpryon's `timestamp_before_ww_start/end`. On the firmware
set the keyword end of events in libpryon's start window comes from uninitialised network state, so only whether
times exist is compared there. Env-gated `afe_matches_oracle_meta` runs libpryon and the engine on `PRYON_AFE_CLIP`
(an Echo `micAsr` capture, or a directory of `.raw` clips) with `PRYON_ORACLE_ROOTFS`, `PRYON_PRYON_TEST` and the
real sets.
In all wake fixtures the `META` lines keep only the `kw*` fields, `activeKeywords`, `clientProperties` (which also
carries libpryon's `PryonLastReceivedSampleIndex`), the LSB-metadata flags and, in `synth-afe/`, the AFE times.

## Third-party notices

The front end and the exp approximation are reimplementations of FreeBSD msun, Ne10, Kaldi and Julien Pommier's
neon_mathfun. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [licenses/](licenses/).
