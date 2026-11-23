# pryon-rs

An open, native Rust reimplementation of two engines in Amazon's `libpryon.so`:

- the **acoustic event detector** behind Alexa Guard's sound detection on Echo devices (smoke and CO alarms, glass
  breaking, dog barks, a baby crying, and eight more sound types). It produces **the same scores, bit for bit, and
  the same decisions** as libpryon, at roughly **half the CPU and a quarter of the memory** on an Echo Dot;
- the **wake word engine** ("Alexa", "Echo", "Computer", ...): the firmware ALEXA model and the en-US wake word sets
  Amazon serves. It produces the same events and the same printed scores as libpryon on the sets validated so far
  (the firmware set outside libpryon's start gate, below), on a PC and on the Echo Dot 3 and Echo Dot 2, where it
  also needs less CPU than libpryon and no more memory.

It is a plain library with a Rust and a C API: 16 kHz mono int16 audio in, sound results per ~10 s window or wake
word detections out. No threads, no global state, no dependencies at run time. It runs the models on any target
Rust builds for, with NEON kernels on 32-bit ARM such as the Echo's own CPU.

Not affiliated with Amazon. This repository contains no Amazon code, models, firmware or recordings: you supply the
model from a device you own.

## Status

Sound detection:

| Check | Result |
|---|---|
| Parity with libpryon on a PC (libpryon run under qemu-arm as the oracle) | identical scores (exact float bits) and decisions on every committed test clip and on 22 real-speech clips, both model generations |
| Parity on the device | output of the Echo Dot 3 and Echo Dot 2 byte-identical to the PC on 26/26 clip × model runs |
| CPU, 164 s clip, fed in real time like libpryon | Echo Dot 3 (Cortex-A35): 21.0 % → 9.0 % of a core (2.3× less); Echo Dot 2 (Cortex-A53): 11.1 % → 5.3 % (2.1× less) |
| Peak memory | Echo Dot 3: 16.8 → 3.9 MB; Echo Dot 2: 11.0 → 2.9 MB |
| Model load | Echo Dot 3: ~85 ms; Echo Dot 2: ~60 ms |

Wake words (`pryon-cli validate --exact`, below, on 994 clips: 329 real "Alexa" and 411 real "Computer"
recordings, TTS "Alexa", "Hey Disney", "Stop" and "Ziggy", spoken commands, long speech, synthetic clips):

| Check | Result |
|---|---|
| alexa-en-US on a PC | every event (ALEXA, HEY_DISNEY, STOP) equal to libpryon's in kind, begin, end and printed score; also with client properties set and with session ends mid-clip |
| Firmware ALEXA on a PC | the same, outside libpryon's firmware start gate (below), which validate sets aside and counts; after a session end 0.5 s into each clip most events fall in the gate (366 of 388 set aside) |
| Network tensors | every node output of both sets bit-equal to libpryon's own tensors, captured under qemu-arm on 4 clips |
| armv7 build under qemu-arm | bit-exact against the qemu oracle as on x86_64; `pryon-cli wake --bits` output byte-identical to x86_64 (both sets, 106 s of real "Alexa" recordings) |
| Parity on the device (Echo Dot 3, Echo Dot 2) | `pryon-cli wake --bits` output byte-identical to the PC on 135/135 short clips per set (firmware ALEXA, alexa-en-US) and on 8/8 long-speech runs; libpryon on the device equals the qemu oracle, apart from its own non-deterministic start gate after a restart, which is deliberately not modelled |
| CPU and memory on the device | at or below libpryon in CPU and peak RSS on both Echos; less CPU, firmware ALEXA / alexa-en-US: Echo Dot 3 (CPU capped at 747.5 MHz) 1.37× / 1.78×, Echo Dot 2 1.92× / 2.38× |
| Echo front-end clock on a real `micAsr` capture | AFE times equal to libpryon's on 13/13 accepts per set |
| Live speech on an Echo Dot 3, both engines on the same microphone stream | 10/10 events of the same keyword and kind as libpryon; 8 sample-exact, the other 2 inside libpryon's start gate |

Whisper detection is not implemented.

## Models

### Sound detection

A model is a directory with `pryon.config`, `AED.json` and `model_gcmvn.cpu.v1.q.mlp`. Two generations are
known, both supported:

- the firmware's, on every Fire OS Echo checked so far: `adb pull /system/local/models/AED`
- the newer one Amazon serves to devices (retrained weights, smoke/CO thresholds 0.845 instead of 0.825)

The type list and every threshold come from the model's `AED.json`.

### Wake words

A wake word set is a directory with a `pryon.manifest`. The engine executes the set's ONNX graphs as they are, and
its keyword rules come from the set's own `kw.cfg.json`, `decoders.cfg.json` and `op.cfg.json` (thresholds,
client-property overrides, lock periods, smoothers, auxiliary spotters, the operating state), with libpryon's
semantics read out of the library. A new set made of known operators and rule fields needs no code, only validation.

| Set | Keywords | State |
|---|---|---|
| firmware `/system/local/models/keyword/en-US/ALEXA` | ALEXA, STOP | validated (ALEXA) |
| `alexa-en-US` | ALEXA, HEY_DISNEY, STOP | validated (ALEXA, STOP) |
| `echo-en-US` | ECHO, HEY_DISNEY, STOP | loads and runs; not validated |
| `amazon-en-US`, `computer-en-US`, `ziggy-en-US` (one set, byte-identical) | AMAZON, COMPUTER, ZIGGY, HEY_DISNEY, STOP | loads and runs; not validated |

Every keyword of a set is served, HEY_DISNEY included: a HEY_DISNEY accept wakes the set like any wake word.
`op.cfg.json` gives the set's operating state: in the en-US sets a wake word moves it from `sleep` to `awake`, STOP
is spotted only while `awake`, and `awake` returns to `sleep` on STOP, after its timeout (175 frames) or at a session
end.

A set the engine does not fully understand is refused at load with `Error::Unsupported { file, what }` naming the
operator and node, attribute, layout or `pryon.config` key, or rule field; it never panics and never runs part of a
set. That covers:

- the de-DE sets: `alexa-de-DE` (its decoder layout belongs to NTT score fusion, which is not implemented) and
  `amazon-de-DE` (the older single-graph format);
- other locales, classifiers and aggregators not in the covered sets, ONNX files with external data, operators
  outside the 23 implemented ones (`Add, BatchNormalization, Cast, Concat, Gather, HybridConv, HybridGemm,
  HybridLSTM, MaxPool, Mul, Neg, ReduceMax, ReduceSum, Relu, Reshape, Shape, Slice, Softmax, Squeeze, Sub, Tile,
  Transpose, Unsqueeze`).

Decoders no keyword reads (voice activity, speaker ID) are never opened; NTT fusion, watermark and fingerprint
configuration is not run.

libpryon's probabilistic accept is not reproduced: a wake word engine should not wake at random. In alexa-en-US's ALEXA,
libpryon turns a score above `probabilistic-accept-threshold` (and below the accept threshold) into an accept
(`AcceptProbabilistic`) with `probabilistic-accept-probability` 0.001, drawn from a clock-seeded generator. The
engine follows libpryon's rule up to the draw and always reports such a score as a near miss, as libpryon does
with the probability set to 0.

Planned: validation records for `echo-en-US` and the shared amazon/computer/ziggy set, one per keyword, plus
whatever operators or rule fields they turn out to need.

## Use

### Rust

```toml
pryon = { git = "https://github.com/nemethhh/pryon-rs" }
```

Sound detection:

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

Wake words:

```rust
use pryon::{DetectionKind, Spotter, SpotterModel, SpotterOptions};
use std::{path::Path, sync::Arc};

let model = Arc::new(SpotterModel::load(Path::new("/path/to/alexa-en-US"))?); // refuses what it cannot run
let mut sp = Spotter::new(model, SpotterOptions::default())?;
let mut found = Vec::new();
sp.feed(&pcm, &mut found); // any chunk size
for d in found.drain(..) {
    if d.kind == DetectionKind::Accept {
        println!("{} {}..{}", sp.keywords()[d.keyword], d.begin_sample, d.end_sample);
    }
}
sp.set_property("AlarmState", 1); // kw.cfg threshold overrides; unknown names are ignored
sp.reset(&mut found); // session end
```

`cargo doc --open -p pryon` documents every public item.

### C

`pryon-ffi/include/pryon.h` documents the API: `pryon_sound_*` (open, feed, type names, reset, close) and
`pryon_spotter_*` (open, feed, set_property, reset, keyword names, checksum, close). Build the static library and
link it with the native libraries rustc lists for the target
(`cargo rustc -p pryon-ffi --release --crate-type staticlib -- --print native-static-libs`):

    cargo build --release -p pryon-ffi
    cc app.c target/release/libpryon_ffi.a -lgcc_s -lutil -lrt -lpthread -lm -ldl -lc         # x86_64 Linux
    armv7a-linux-androideabi24-clang app.c libpryon_ffi.a -ldl -llog -lunwind -lm -lc           # Echo (NDK r21e)

`pryon-ffi/examples/sound.c` and `pryon-ffi/examples/spotter.c` are complete programs; `cargo test` builds and runs
them. A panic inside the library never unwinds into C: the call fails and the handle refuses further audio (release
builds unwind, so link the target's unwinder, as above).

### Command line

`pryon-cli`:

- `run -m MODEL [-t type,...] [--bits] CLIP.raw`: print each window's scores and decisions, in the layout of
  libpryon's test tool, so the two diff line by line
- `bench -m MODEL [--chunk N] [--speed X] CLIP.raw`: CPU per second of audio, worst single feed, peak memory;
  `--speed 1` feeds in real time
- `rusage -- CMD ...`: run a command and report its CPU time and peak memory (to measure libpryon the same way)
- `wake -m SET [-p NAME=VALUE[@SAMPLE]]... [--reset-at SAMPLE]... [--chunk N] [--bits] CLIP.raw`: print the wake
  engine's detections (`EVENT` lines) and decoder restarts (`RESTART` lines, checked after each chunk); `-p` sets a
  client property and `--reset-at` ends the session before the given sample
- `bench-wake -m SET [--chunk N] CLIP.raw`: as `bench`, for a wake word set
- `inspect SET`: what the loader read from a wake word set (stages, spotters, thresholds, operating states)
- `validate ... MODEL CLIPS...`: compare the wake engine with libpryon, below
- `synth NAME OUT.raw`, `synth-model DIR`, `synth-wake-set fw|davs DIR [--seed N] [--force SCORE]`: synthetic test
  clips, a random-weight sound model and random-weight wake word sets in the real formats
- `wake-dump -m SET [--lfbe OUT.f32] [--out DIR] CLIP.raw`: the front end's frames and every stage output, for
  comparisons with a reference

Clips are raw 16 kHz mono s16le.

### Validating a wake word set

    pryon-cli validate [--oracle-rootfs DIR --pryon-test PATH] [--cache DIR] [--exact] [--records DIR]
        [--record KEYWORD]... [--speed X] [--jobs N] [--lead-ms MS] [--tail-ms MS] [-p NAME=VALUE[@SAMPLE]]...
        [-e SAMPLE]... MODEL CLIPS...

runs libpryon (through `pryon_test`, a small armv7 C harness around libpryon's API that is not part of this
repository, under qemu-arm against a Fire OS root filesystem; `PRYON_ORACLE_ROOTFS` and `PRYON_PRYON_TEST` stand in
for the two flags) and the wake engine on the same `.raw` or
16 kHz mono `.wav` clips (files or directories) and prints every difference as
`DIFF <clip> <keyword> oracle=<event|-> rust=<event|->`, then a summary per keyword.
Exit 0 clean, 1 differences, 2 usage or oracle errors or a set the engine refuses or cannot read.

- **Timing.** The oracle runs in real time (`--speed 1`, the default), `--jobs` clips at a time: at higher speeds
  libpryon's results depend on feed timing. `--cache` keeps oracle outputs as `<set checksum>/<key>.txt`, the key
  covering the padded clip, the commands, the speed and the harness version.
- **Padding.** Both engines get each clip between `--lead-ms` (default 2500) and `--tail-ms` (default 1000) of
  digital silence: the lead covers libpryon's firmware start gate (below), the tail its reporting delay (an accept
  comes ~120 ms after the keyword ends, so not at the very end of a clip). Events are printed in clip samples.
- **Commands.** `-p` sets a client property and `-e` ends the session (`Spotter::reset`, libpryon's
  `PryonDecoder_SessionEnd`) before the given clip sample, in both engines. When libpryon restarts its decoder after
  too many accepts of one keyword (`restart-trigger`, default 12 within 60 s), `pryon_test` creates a new one and goes
  on from the sample after the triggering frame, as the engine does.
- **Probabilistic accept.** The oracle runs on a copy of the set whose `probabilistic-accept-probability` values are
  0 (symlinks plus the edited `kw.cfg.json`), which is the field's only effect; the engine never draws.
- **Exact.** `--exact` compares every event's kind, begin and end and the engine's score printed as `pryon_test`
  prints it (C `%g`) with the oracle's string.

**The firmware start gate.** The firmware graph's conv buffers have no stored values. libpryon starts them from
uninitialised heap memory (NaN, denormals, 1e12-size garbage) at the start, after every session end
(`PryonDecoder_SessionEnd`) and after every decoder restart, so the resulting NaNs keep libpryon silent until that
garbage has left the network: the first 168 or 174 frames (about 1.7 s) in the installs measured (the garbage, and with
it the length, follows the install path). The engine starts the buffers from zeros, the only defined state, and does not
reproduce the garbage: there is no rule in the model files to follow. So on a set whose buffers libpryon leaves
uninitialised (the firmware set; not the DAVS sets), and only there, validate sets aside the following and counts them
(`GATE` lines per clip, `GATE ALLOWANCE` in the summary, `gate_exempt` in the record); the gate at the start of a clip
falls in the leading silence instead:

- one-sided events, or pairs of equal kind, whose ends lie within 174 frames plus the set's longest lock period
  after a session end or a restart both engines place at the same sample;
- engine events within 174 frames of such a point that libpryon lacks;
- the events after the first restart the two engines place differently, but only when an event of the two kinds
  above was set aside before it in the clip (a gate-hidden accept moves the engine's restart point); without one, a
  restart placed differently is a difference.

Everything else is a difference.

**Records.** A clean `--exact` run writes `validated/<checksum>.<KEYWORD>.rec` (or under `--records DIR`) for each
keyword the oracle accepted at least once, also one accepted only on synthetic clips (HEY_DISNEY on TTS clips);
`--record KEYWORD` (a keyword the set serves) limits it to the keywords chosen. A record holds the set, the keyword,
the clip count and digest, the padding, accept and near-miss counts, the events the gate allowance set aside
(`gate_exempt`), the date and the pryon-rs commit. A clip the oracle did not hear to the end is an error, so it never
yields a record. An application can serve only the (set, keyword) pairs listed in `validated/`. The committed
records:

| Record | Clips | Accepts | Near misses | Set aside by the gate |
|---|---|---|---|---|
| `74a26372ecb79fc6e5954d9f9fff29a3.ALEXA` (firmware) | 994 | 434 | 35 | 646, all from one long-speech clip after a restart moved by the gate |
| `3e1c0db2cc5bdd846dba970750af8c15.ALEXA` (alexa-en-US) | 994 | 670 | 22 | 0 |
| `3e1c0db2cc5bdd846dba970750af8c15.STOP` (alexa-en-US) | 994 | 3 | 0 | 0 |

## What a result means

### Sound detection

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

### Wake words

- A `Detection` has the keyword (an index into `keywords()`), its kind (`Accept`, `AcceptAuxiliary` for an accept
  by the keyword's verifier spotter, `NearMiss`), the score, and `begin_sample`/`end_sample` counted from the first
  sample fed to the `Spotter`; `reset` does not restart that count.
- `set_property(name, value)` applies from the first 10 ms frame that starts after the samples fed so far.
- `reset` is libpryon's session end: a pending near miss is reported, then the front end, network, rules and
  operating state restart; client properties stay.
- libpryon restarts its decoder by itself after `restart-trigger` accepts of one keyword; the engine does the same
  (its frames start again at the sample after the triggering frame).
- With `SpotterOptions::echo_stream` (C: `PRYON_SPOTTER_ECHO_STREAM`), for an Echo's `micAsr` stream that carries
  the front end's clock in its lowest bits, each detection gets `afe`: libpryon's
  `timestamp_before_ww_start/_end`. Off by default; other platforms have no such marks.

## How it works

Sound detection: audio → log mel filterbank, 20 bins, 25 ms frames every 10 ms (the same float operation order as
libpryon's Kaldi/Ne10 front end) → per-window mean normalisation with libpryon's handling of digital silence → an
int8 dynamic-quantised LSTM trunk (20→64) and nine LSTM heads (64→96) on every other frame → nine float outputs
routed to the model's twelve types → sigmoid. Every float operation is rounded where libpryon rounds it: no fused
multiply-add, libpryon's exp and reciprocal approximations, its quantiser's rounding. On 32-bit ARM, NEON assembly
does the heavy parts. Each assembly kernel has a portable Rust twin and a test that the two agree, and the portable
code runs on every other target.

Wake words: audio → the same front end with the set's 64 bins over 80–7200 Hz → a small ONNX executor (hand-written
protobuf reader, graphs checked at load against the implemented operators and attributes, shapes resolved once,
streaming state carried between steps) running the set's encoder and decoders, or the firmware's single graph →
libpryon's score smoothers and keyword rules → detections. The `Hybrid*` operators follow libpryon's int8 product
(CPUNK: per-row activation quantisation, an exact int8 dot product, its dequantisation order), and every float
operator follows libpryon's operation order, including its flush-to-zero of subnormals, which the engine applies in
software on every target. Fused multiply-add is used only where libpryon fuses: the LSTM cell's and
BatchNormalization's `fmaf` and CPUNK's exp polynomial. The portable code computes these fused results exactly in
integer arithmetic (no FMA instruction, no `f64`, no libm).

The wake engine has its own armv7 kernels: the int8 dot product (weights padded at load to a multiple of 16 columns
and 4 rows), CPUNK's row range, row quantiser and dequantisation, CPUNK's exp, and the LSTM cell's `fmaf`. NEON
`vfma` runs the exp and always flushes subnormals as libpryon's does; VFP `vfma` runs `fmaf` with FPSCR's FZ, DN and
rounding mode cleared so it is not flushed, as bionic's `fmaf` is not.

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

That runs everything that needs no Amazon files. Tests that need a model, libpryon or your own clips print
`skipped: set VAR` unless these are set:

- `PRYON_AED_MODEL_DIR`: the firmware sound model directory
- `PRYON_AED_DAVS_DIR`: the newer sound model directory
- `PRYON_CLIPS_DIR`: a directory of your own clips, each `<name>.raw` with libpryon's output beside it as
  `<name>_out.txt` (newer model) and/or `<name>_fw_out.txt` (firmware model)
- `PRYON_ORACLE_OVERRIDE_DIR`: compare against libpryon outputs in this directory (same `{fw,davs}/<clip>.txt`
  layout) instead of the committed ones, e.g. outputs produced on a device
- `PRYON_WAKE_FW_DIR`: the firmware ALEXA directory (`/system/local/models/keyword/en-US/ALEXA`)
- `PRYON_WAKE_SETS_DIR`: a directory holding the DAVS sets as `<word>-en-US/` (and `alexa-de-DE/`, `amazon-de-DE/`
  for the refusal tests)
- `PRYON_ORACLE_ROOTFS`, `PRYON_PRYON_TEST`: a Fire OS root filesystem and an armv7 `pryon_test` built against it,
  for the tests that run libpryon under qemu-arm
- `PRYON_WAKE_CORPUS`: a directory of `<group>/<clip>.raw` clips; the corpus test compares both engines on every
  clip, with the oracle outputs of a `validate --cache` run kept in `validate-v2/` beside it
  (`PRYON_CORPUS_PARTIAL=1` accepts a cache that lacks some clips)
- `PRYON_PYPRYON_REF`: per-step stage outputs of [pypryon](https://github.com/nemethhh/pryon-wyoming), the
  intermediate reference, as `<set checksum>/...`
- `PRYON_CAPTURE_DIR`: libpryon's own tensors per step and node, captured under qemu-arm, as
  `<set checksum>/<clip>/` (`pcm.raw`, `<stage>/<step>/shape.json` and `<tensor>.f32`)
- `PRYON_AFE_CLIP`: an Echo `micAsr` capture (or a directory of `.raw` clips) for the front-end clock test, with
  `PRYON_ORACLE_ROOTFS` and `PRYON_PRYON_TEST`

### Oracle fixtures

`pryon/tests/fixtures/oracle/{fw,davs}/<clip>.txt` are libpryon's outputs on synthetic clips that the tests
regenerate from `pryon-testkit`, one file per clip and model. They were made with `aed_test`, a small armv7 C harness
around libpryon's sound API (not part of this repository), run under qemu-arm against an Echo Dot 3 root
filesystem:

    pryon-cli synth noise noise.raw
    unshare --user --map-root-user --pid --fork --kill-child \
      qemu-arm -L $ROOTFS aed_test -m MODEL/pryon.manifest -x 1 -j noise.raw > noise.txt

In real time (`-x 1`): faster, libpryon may drop audio once its backlog passes about 7 s and log `Audio backlog size
reset`. Each file must have floor(samples / 159680) windows and no such line. The tests compare the printed scores
(6 decimals), the decisions, `maxEnergyLevel` and `numSamplesAboveEnergyThreshold`. `davs/bits/<clip>.txt` adds the
exact score bits of each window, from a Python reference of the recipe that matches the printed scores.

In the sound detector on a PC, very small intermediate values are not flushed to zero, whereas NEON flushes them (as
libpryon does on the device). No fixture is affected. Run on the device to confirm parity there.

### Wake fixtures

`pryon/tests/fixtures/lfbe64/` are the 64-bin front end's frames on the first 3 s of the `lfbe/` clips, from
pypryon's `vendor/pryon-native/pryon_lfbe.c` (the C front end, with the wake word models' bins and mel range).
`hybrid/` holds testkit-built `Hybrid*` graphs with random weights (`<case>.onnx`, inputs `<case>.in.f32`, written
by the ignored test `write_fixtures`) and their outputs from pypryon's CPUNK rewrite run in onnxruntime
(`<case>.out.f32`).

`pryon/tests/fixtures/wake/{fw,alexa}/<clip>.txt` are the wake events of `pryon_test` (the harness `validate`
uses) for the firmware ALEXA set and alexa-en-US on the synthetic clips, run as above with
`-m SET/pryon.manifest -x 1` (at `-x 1000` libpryon drops most of these 20-25 s clips). Their tests need
`PRYON_WAKE_FW_DIR` and `PRYON_WAKE_SETS_DIR`.

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
times exist is compared there.
In all wake fixtures the `META` lines keep only the `kw*` fields, `activeKeywords`, `clientProperties` (which also
carries libpryon's `PryonLastReceivedSampleIndex`), the LSB-metadata flags and, in `synth-afe/`, the AFE times.
Only events on synthetic clips and sets are committed: no outputs on real recordings and no captured tensors.

## Third-party notices

The front end and the exp approximation are reimplementations of FreeBSD msun, Ne10, Kaldi and Julien Pommier's
neon_mathfun. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [licenses/](licenses/).
