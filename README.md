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
