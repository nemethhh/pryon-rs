# pryon-rs

Rust replacement for Amazon's libpryon Guard (AED) sound detector on the Echo Dot.
Local repo, not published.

## Crates

- `pryon`: the engine (no runtime dependencies).
- `pryon-ffi`: C API (staticlib/cdylib).
- `pryon-cli`: command-line tools (`synth`, more later).
- `pryon-testkit`: deterministic synthetic clips and test models.

## Tests

    cargo test --workspace

Tests that need Amazon models or local clips skip (`skipped: set VAR`) unless set:

- `PRYON_AED_MODEL_DIR`: firmware AED model (`echo-dot-assist/firmware/donut/rootfs/system/local/models/AED`).
- `PRYON_AED_DAVS_DIR`: DAVS AED model (`pypryon/build/aed/model`).
- `PRYON_CLIPS_DIR`: local clips; `local_clips` checks every `<name>.raw` against `<name>_out.txt` (DAVS) and
  `<name>_fw_out.txt` (firmware).

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
