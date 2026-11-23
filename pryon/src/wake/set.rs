//! Firmware and DAVS layouts to a [`SetSpec`]; set checksum.

use super::kwcfg::{self, ClassifierSpec, MachineSpec, Nemort, OutputType, RestartSpec, SpotterSpec};
use crate::json::Doc;
use super::pconfig::{inside, join, read_manifest, read_text, rel, unsupported, PConfig};
use crate::lfbe::LfbeSpec;
use crate::onnx;
use crate::Error;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq)]
pub enum StageInput {
    Audio(String),
    Artifact { input: String, artifact: String },
}

#[derive(Debug, Clone, PartialEq)]
pub struct StageSpec {
    pub name: String,
    pub graph: PathBuf,
    /// LFBE frames per step for an audio stage; artifact steps (always 1) for the others.
    pub frames_per_step: usize,
    pub feat_dim: usize,
    pub inputs: Vec<StageInput>,
    /// (graph output, name it is published under: its artifact name, else itself).
    pub outputs: Vec<(String, String)>,
    /// (state input, state output) pairs.
    pub lstm: Vec<(String, String)>,
    pub recurrent: Vec<(String, String)>,
    /// The recurrent inputs carry stored initial values in the graph.
    pub primed: bool,
}

#[derive(Debug, Clone)]
pub struct SetSpec {
    pub dir: PathBuf,
    pub checksum: String,
    pub fe: LfbeSpec,
    pub stages: Vec<StageSpec>,
    pub spotters: Vec<SpotterSpec>,
    pub words: Vec<(u32, String)>,
    pub machine: Option<MachineSpec>,
    pub restart: RestartSpec,
}

impl SetSpec {
    /// Some stage's recurrent buffers have no stored values: libpryon then starts them from uninitialised heap, so
    /// its scores are NaN for about 28 steps after every start (the engine starts them from zeros).
    pub fn uninitialised_state(&self) -> bool {
        self.stages.iter().any(|s| !s.recurrent.is_empty() && !s.primed)
    }

    /// The longest lock period of the set's spotters, in frames.
    pub fn longest_lock(&self) -> u64 {
        self.spotters.iter().map(|s| u64::from(s.lock)).max().unwrap_or(0)
    }
}

/// How a pryon.config key is treated.
#[derive(Clone, Copy)]
enum Key {
    /// Effect not implemented: must hold this value.
    Str(&'static str),
    Num(f32),
    /// Read by the loader.
    Used,
    /// Feeds only things the engine does not run (watermark, fingerprint, NTT fusion, speaker ID).
    Ignored,
}

use Key::{Ignored, Num, Str, Used};

const COMMON_KEYS: &[(&str, Key)] = &[
    ("fe.type", Str("LFBE")),
    ("fe.audio_analysis.dither", Num(0.0)),
    ("fe.audio_analysis.frame_length_milli", Num(25.0)),
    ("fe.audio_analysis.frame_shift_milli", Num(10.0)),
    ("fe.audio_analysis.preemphasis_coeff", Num(0.96875)),
    ("fe.audio_analysis.remove_dc_offset", Num(1.0)),
    ("fe.audio_analysis.round_to_power_of_two", Num(1.0)),
    ("fe.audio_analysis.samplerate", Num(16000.0)),
    ("fe.audio_analysis.window_type", Str("povey")),
    ("fe.cep.raw_energy", Num(1.0)),
    ("fe.cep.use_energy", Num(0.0)),
    ("fe.feat.dimensions", Used),
    ("fe.feat.stack_left", Num(0.0)),
    ("fe.feat.stack_right", Num(0.0)),
    ("fe.feat.do_mean_norm", Num(0.0)),
    ("fe.feat.do_variance_norm", Num(0.0)),
    ("fe.feat.transform_filepath", Str("")),
    ("fe.feat.affine_transform_filepath", Str("")),
    ("fe.mel_fbank.num_bins", Used),
    ("fe.mel_fbank.low_freq", Used),
    ("fe.mel_fbank.high_freq", Used),
    ("fe.multiple_fe_filepath", Ignored),
    ("search.decoder_type", Str("kaldi-key-phrase")),
    ("keyword_spotter.config_filepath", Used),
    ("keyword_spotter.op_config_filepath", Used),
    ("keyword_spotter.emit_nearmiss", Num(1.0)),
    ("id2word.mapping_filepath", Used),
    ("watermark.config_filepath", Ignored),
    ("watermark.enable", Ignored),
];

const FIRMWARE_KEYS: &[(&str, Key)] = &[
    ("scorer.model_filetype", Str("nemort-base")),
    ("scorer.nemort_config_json_filepath", Used),
    ("scorer.initialize.frame_count", Used),
    ("scorer.score_upsampling.enable", Num(1.0)),
    ("scorer.score_upsampling.default_skip_frame_count", Used),
];

const DAVS_KEYS: &[(&str, Key)] = &[
    ("scorer.single.config_filepath", Used),
    ("scorer.multi.config_filepath", Used),
    ("search.tensor_artifacts_list_file", Used),
    ("ntt.config_filepath", Ignored),
    ("uom_speakerid.config_filepath", Ignored),
    ("scd.config_filepath", Ignored),
];

fn check_keys(cfg: &PConfig, top: &Path, layout: &[(&str, Key)]) -> Result<(), Error> {
    let find = |k: &str| COMMON_KEYS.iter().chain(layout).find(|(n, _)| *n == k).map(|(_, r)| *r);
    for (k, s) in &cfg.keys {
        let shown = |want: String| unsupported(&s.file, format!("{k} = {} (implemented: {want})", s.value));
        match find(k) {
            None => return Err(unsupported(&s.file, format!("key {k}"))),
            Some(Str(w)) if s.value != w => return Err(shown(format!("\"{w}\""))),
            Some(Num(w)) if s.value.parse::<f32>().ok() != Some(w) => return Err(shown(w.to_string())),
            _ => {}
        }
    }
    for (k, r) in COMMON_KEYS.iter().chain(layout) {
        if matches!(r, Str(_) | Num(_)) && cfg.get(k).is_none() {
            return Err(Error::Config(format!("{}: missing key {k}", top.display())));
        }
    }
    Ok(())
}

fn setting<'a>(cfg: &'a PConfig, top: &Path, key: &str) -> Result<&'a super::pconfig::Setting, Error> {
    cfg.get(key).ok_or_else(|| Error::Config(format!("{}: missing key {key}", top.display())))
}

/// The file a key names, confined to the set directory.
fn path_key(cfg: &PConfig, top: &Path, key: &str) -> Result<PathBuf, Error> {
    let s = setting(cfg, top, key)?;
    inside(&cfg.dir, s.path(), &s.file, key)
}

fn int_key(cfg: &PConfig, top: &Path, key: &str) -> Result<usize, Error> {
    let s = setting(cfg, top, key)?;
    s.value.parse().map_err(|_| unsupported(&s.file, format!("{key} = {} (not an integer)", s.value)))
}

fn f32_key(cfg: &PConfig, top: &Path, key: &str) -> Result<f32, Error> {
    let s = setting(cfg, top, key)?;
    let x = s.value.parse::<f32>().ok().filter(|x| x.is_finite());
    x.ok_or_else(|| unsupported(&s.file, format!("{key} = {}", s.value)))
}

fn fe_spec(cfg: &PConfig, top: &Path) -> Result<LfbeSpec, Error> {
    let bins = int_key(cfg, top, "fe.mel_fbank.num_bins")?;
    let (low_hz, high_hz) = (f32_key(cfg, top, "fe.mel_fbank.low_freq")?, f32_key(cfg, top, "fe.mel_fbank.high_freq")?);
    let file = &setting(cfg, top, "fe.mel_fbank.num_bins")?.file;
    if !((1..=128).contains(&bins) && 0.0 <= low_hz && low_hz < high_hz && high_hz <= 8000.0) {
        return Err(unsupported(file, format!("mel bank of {bins} bins over {low_hz}..{high_hz} Hz")));
    }
    let dims = int_key(cfg, top, "fe.feat.dimensions")?;
    if dims != bins {
        let f = &setting(cfg, top, "fe.feat.dimensions")?.file;
        let what = format!("fe.feat.dimensions = {dims} with {bins} mel bins (no stacking implemented)");
        return Err(unsupported(f, what));
    }
    Ok(LfbeSpec { bins, low_hz, high_hz })
}

/// A stage's graph output as spotters may reference it.
struct Component {
    stage: String,
    name: String,
    kind: OutputType,
    classes: Option<usize>,
}

/// Tensor artifacts published so far: (name, declared last dim).
type Published = Vec<(String, Option<usize>)>;

struct Builder<'a> {
    dir: &'a Path,
    fe_bins: usize,
    stages: Vec<StageSpec>,
    published: Published,
    components: Vec<Component>,
}

fn dims_str(d: &[Option<usize>]) -> String {
    let v: Vec<String> = d.iter().map(|x| x.map_or("?".into(), |x| x.to_string())).collect();
    format!("[{}]", v.join(", "))
}

impl Builder<'_> {
    /// `outputs` is the scorer's `scorer.num_dnn_outputs` with the file that sets it.
    fn stage(
        &mut self,
        name: &str,
        n: Nemort,
        frames: usize,
        feat: usize,
        outputs: Option<(usize, &Path)>,
    ) -> Result<(), Error> {
        if self.stages.iter().any(|s| s.name == name) {
            return Err(unsupported(&n.file, format!("a second stage named {name}")));
        }
        let graph = inside(self.dir, n.graph.clone(), &n.file, "model_filepath")?;
        let g = onnx::interface(&graph)?;
        let gname = rel(self.dir, &n.graph);
        let missing =
            |what: &str, x: &str| Error::Config(format!("{}: {what} {x} is not declared by {gname}", n.file.display()));
        let input = |x: &str| g.inputs.iter().find(|p| p.name == x).ok_or_else(|| missing("input", x));
        let output = |x: &str| g.outputs.iter().find(|p| p.name == x).ok_or_else(|| missing("output", x));
        let fits = |d: Option<usize>, want: usize| d.is_none_or(|d| d == want);

        let (iname, artifact) = &n.input;
        let p = input(iname)?;
        let stage_input = match artifact {
            None => {
                let d = &p.dims;
                let ok = matches!(d.len(), 3 | 4)
                    && fits(d[0], 1)
                    && fits(d[1], frames)
                    && fits(d[2], self.fe_bins)
                    && (d.len() == 3 || fits(d[3], 1));
                if !ok || feat != self.fe_bins {
                    let what = format!(
                        "audio input {iname} declared {} with feat {feat}; the front end feeds [1, {frames}, {}(, 1)]",
                        dims_str(d),
                        self.fe_bins
                    );
                    return Err(unsupported(&n.graph, what));
                }
                StageInput::Audio(iname.clone())
            }
            Some(a) => {
                let Some((_, last)) = self.published.iter().find(|(x, _)| x == a) else {
                    let have: Vec<&str> = self.published.iter().map(|(x, _)| x.as_str()).collect();
                    let have = have.join(", ");
                    let what = format!("input {iname} reads artifact {a}, which no earlier stage publishes ({have})");
                    return Err(unsupported(&n.file, what));
                };
                if frames != 1 || !fits(*last, feat) || !fits(p.dims.last().copied().flatten(), feat) {
                    let d = dims_str(&p.dims);
                    let what = format!(
                        "artifact input {iname} declared {d} with feat {feat}, {frames} steps per step (implemented: 1)"
                    );
                    return Err(unsupported(&n.file, what));
                }
                StageInput::Artifact { input: iname.clone(), artifact: a.clone() }
            }
        };
        for (i, o) in &n.lstm {
            if input(i)?.init == Some(true) {
                return Err(unsupported(&n.graph, format!("LSTM state {i} stored in the graph (rings start at zero)")));
            }
            output(o)?;
        }
        let mut stored = Vec::new();
        for (i, o) in &n.recurrent {
            stored.push(input(i)?.init == Some(true));
            output(o)?;
        }
        let primed = !stored.is_empty() && stored.iter().all(|&s| s);
        if !primed && stored.iter().any(|&s| s) {
            return Err(unsupported(&n.graph, "recurrent inputs partly stored in the graph"));
        }
        let mut outs = Vec::new();
        for o in &n.outputs {
            let dims = &output(&o.component)?.dims;
            // lanes-first decoders behind an encoder exist only in the de-DE sets, which are out of scope
            if let (Some(_), [Some(l), Some(1), _]) = (artifact, &dims[..]) {
                if *l > 1 {
                    let what = format!(
                        "output {} {} of a decoder reading a tensor artifact (implemented: [1, lanes, classes])",
                        o.component,
                        dims_str(dims)
                    );
                    return Err(unsupported(&n.graph, what));
                }
            }
            let classes = dims.last().copied().flatten();
            if let Some(a) = &o.artifact {
                if self.published.iter().any(|(x, _)| x == a) {
                    return Err(unsupported(&n.file, format!("artifact {a} published twice")));
                }
                self.published.push((a.clone(), classes));
            }
            self.components.push(Component { stage: name.into(), name: o.component.clone(), kind: o.kind, classes });
            outs.push((o.component.clone(), o.artifact.clone().unwrap_or_else(|| o.component.clone())));
        }
        if let Some((k, file)) = outputs {
            let last = |c: &str| output(c).ok().and_then(|p| p.dims.last().copied().flatten());
            if !n.outputs.iter().any(|o| fits(last(&o.component), k)) {
                let what = format!("{name}: scorer.num_dnn_outputs = {k} matches no output of {gname}");
                return Err(unsupported(file, what));
            }
        }
        self.stages.push(StageSpec {
            name: name.into(),
            graph: n.graph.clone(),
            frames_per_step: frames,
            feat_dim: feat,
            inputs: vec![stage_input],
            outputs: outs,
            lstm: n.lstm,
            recurrent: n.recurrent,
            primed,
        });
        Ok(())
    }

    /// Every spotter reads outputs its scorer has, of the right kind, with its class indices in range.
    fn check_spotters(&self, kw: &Path, spotters: &[SpotterSpec], words: &[(u32, String)]) -> Result<(), Error> {
        for s in spotters {
            let bad = |what: String| unsupported(kw, format!("spotter {}: {what}", s.name));
            if !words.iter().any(|(_, w)| *w == s.name) {
                return Err(bad("not in the word list".into()));
            }
            let find = |c: &str, kind: OutputType, idx: &[usize]| -> Result<(), Error> {
                let comp = self.components.iter().find(|x| x.stage == s.scorer && x.name == c);
                let comp = comp.ok_or_else(|| bad(format!("{c} is not an output of scorer {}", s.scorer)))?;
                if comp.kind != kind {
                    return Err(bad(format!("{c} is declared {:?}, used as {kind:?}", comp.kind)));
                }
                if let Some(&i) = idx.iter().find(|&&i| comp.classes.is_some_and(|n| i >= n)) {
                    return Err(bad(format!("index {i} outside {c} ({} classes)", comp.classes.unwrap_or(0))));
                }
                Ok(())
            };
            for w in &s.components {
                find(&w.component, OutputType::Decoding, &[w.class])?;
                if let Some(se) = &w.start_end {
                    find(se, OutputType::Alignment, &[w.start_index, w.end_index])?;
                }
            }
        }
        Ok(())
    }
}

/// Loads the set whose `pryon.manifest` is in `dir`; refuses anything the engine does not implement.
pub fn load_set(given: &Path) -> Result<SetSpec, Error> {
    let dir = &join(given, ".");
    let m = read_manifest(dir)?;
    let top = m.config.clone();
    let cfg = PConfig::read(dir, &top)?;
    let davs = cfg.get("scorer.single.config_filepath").is_some() || cfg.get("scorer.multi.config_filepath").is_some();
    check_keys(&cfg, &top, if davs { DAVS_KEYS } else { FIRMWARE_KEYS })?;
    let fe = fe_spec(&cfg, &top)?;
    let words_set = setting(&cfg, &top, "id2word.mapping_filepath")?;
    let words = kwcfg::words(&path_key(&cfg, &top, "id2word.mapping_filepath")?)?;
    let words_name = Path::new(&words_set.value).file_name().unwrap_or_default().to_string_lossy().into_owned();
    let kw = path_key(&cfg, &top, "keyword_spotter.config_filepath")?;
    let mut b = Builder { dir, fe_bins: fe.bins, stages: Vec::new(), published: Vec::new(), components: Vec::new() };

    let (spotters, restart) = if davs {
        let (spotters, restart) = kwcfg::spotters(&kw, None, &words_name)?;
        let enc = Doc::load(&path_key(&cfg, &top, "scorer.single.config_filepath")?)?;
        let list = kwcfg::scorers(&enc)?;
        let [e] = &list[..] else {
            return Err(enc.unsup(format!("{} scorers (one encoder implemented)", list.len())));
        };
        let e = kwcfg::scorer(&enc, e)?;
        let n = kwcfg::nemort(&inside(dir, e.nemort.clone(), &enc.path, "scorer.model_filepath")?)?;
        if n.input.1.is_some() {
            return Err(unsupported(&n.file, "an encoder that reads an artifact"));
        }
        b.stage(&e.name, n, e.frames, e.feat_dim, Some((e.outputs, &enc.path)))?;
        let dec = Doc::load(&path_key(&cfg, &top, "scorer.multi.config_filepath")?)?;
        let list = kwcfg::scorers(&dec)?;
        for s in &spotters {
            if !list.iter().any(|o| dec.str(o, "name").ok() == Some(s.scorer.as_str())) {
                let listed = rel(dir, &dec.path);
                let what = format!("spotter {} uses scorer {}, not listed in {listed}", s.name, s.scorer);
                return Err(unsupported(&kw, what));
            }
        }
        for o in &list {
            let name = dec.str(o, "name")?;
            // decoders no spotter reads (DVAD, speaker ID) are never opened
            if spotters.iter().any(|s| s.scorer == name) {
                let c = kwcfg::scorer(&dec, o)?;
                let key = format!("{}.scorer.model_filepath", c.name);
                let n = kwcfg::nemort(&inside(dir, c.nemort.clone(), &dec.path, &key)?)?;
                b.stage(&c.name, n, c.frames, c.feat_dim, Some((c.outputs, &dec.path)))?;
            }
        }
        let list_file = path_key(&cfg, &top, "search.tensor_artifacts_list_file")?;
        let listed = kwcfg::tensor_artifacts(&list_file)?;
        if let Some((a, _)) = b.published.iter().find(|(a, _)| !listed.contains(a)) {
            return Err(unsupported(&list_file, format!("artifact {a} is published but not in tensor_artifacts")));
        }
        (spotters, restart)
    } else {
        let frames = int_key(&cfg, &top, "scorer.initialize.frame_count")?;
        let skip = setting(&cfg, &top, "scorer.score_upsampling.default_skip_frame_count")?;
        if frames == 0 || skip.value.parse::<usize>().ok() != Some(frames - 1) {
            let k = "scorer.score_upsampling.default_skip_frame_count";
            let what = format!("{k} = {} with frame_count {frames} (implemented: frame_count - 1)", skip.value);
            return Err(unsupported(&skip.file, what));
        }
        let n = kwcfg::nemort(&path_key(&cfg, &top, "scorer.nemort_config_json_filepath")?)?;
        if n.input.1.is_some() {
            return Err(unsupported(&n.file, "a single-graph set whose graph reads an artifact"));
        }
        b.stage(&m.model, n, frames, fe.bins, None)?;
        kwcfg::spotters(&kw, Some(&m.model), &words_name)?
    };
    b.check_spotters(&kw, &spotters, &words)?;

    let machine = match cfg.get("keyword_spotter.op_config_filepath") {
        Some(_) => {
            let keywords: Vec<&str> = spotters.iter().filter(|s| !s.auxiliary).map(|s| s.name.as_str()).collect();
            Some(kwcfg::machine(&path_key(&cfg, &top, "keyword_spotter.op_config_filepath")?, &keywords)?)
        }
        None => None,
    };
    let checksum = match &m.checksum {
        Some(p) => read_text(p)?.trim().to_string(),
        None => fnv_tree(dir)?,
    };
    Ok(SetSpec { dir: given.to_path_buf(), checksum, fe, stages: b.stages, spotters, words, machine, restart })
}

/// FNV-1a-64 over every regular file under `dir`, by sorted relative path: the path's bytes, then the file's.
/// Symlinks are refused, as the loader refuses them, so the hash covers exactly what can be read.
fn fnv_tree(dir: &Path) -> Result<String, Error> {
    fn walk(root: &Path, d: &Path, out: &mut Vec<(String, PathBuf)>) -> Result<(), Error> {
        let io = |source| Error::Io { path: d.to_path_buf(), source };
        for e in std::fs::read_dir(d).map_err(io)? {
            let p = e.map_err(io)?.path();
            let t = std::fs::symlink_metadata(&p).map_err(|source| Error::Io { path: p.clone(), source })?.file_type();
            if t.is_dir() {
                walk(root, &p, out)?;
            } else if t.is_symlink() {
                return Err(unsupported(&p, "symlink in the set (checksum covers regular files only)"));
            } else if t.is_file() {
                out.push((rel(root, &p), p));
            }
        }
        Ok(())
    }
    let mut files = Vec::new();
    walk(dir, dir, &mut files)?;
    files.sort();
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for (r, p) in files {
        let data = std::fs::read(&p).map_err(|source| Error::Io { path: p.clone(), source })?;
        for &b in r.as_bytes().iter().chain(&data) {
            h = (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3);
        }
    }
    Ok(format!("{h:016x}"))
}

/// A human-readable summary of a loaded set (`pryon-cli inspect`).
#[doc(hidden)]
pub fn describe(s: &SetSpec) -> Result<String, Error> {
    let mut o = String::new();
    let _ = writeln!(o, "set {}\nchecksum {}", s.dir.display(), s.checksum);
    let _ = writeln!(o, "front end: {} bins, {}..{} Hz", s.fe.bins, s.fe.low_hz, s.fe.high_hz);
    for st in &s.stages {
        let g = onnx::interface(&st.graph)?;
        let ops: Vec<String> = g.ops.iter().map(|(op, n)| format!("{op} {n}")).collect();
        let total: usize = g.ops.iter().map(|(_, n)| n).sum();
        let ins: Vec<String> = st
            .inputs
            .iter()
            .map(|i| match i {
                StageInput::Audio(n) => format!("audio -> {n}"),
                StageInput::Artifact { input, artifact } => format!("{artifact} -> {input}"),
            })
            .collect();
        let outs: Vec<String> =
            st.outputs.iter().map(|(c, p)| if c == p { c.clone() } else { format!("{c} as {p}") }).collect();
        let _ = writeln!(
            o,
            "stage {}: {}\n  {} per step, feat {}; in {}; out {}\n  lstm pairs {}, recurrent pairs {}{}\n  \
             {total} ops: {}",
            st.name,
            rel(&s.dir, &st.graph),
            match &st.inputs[..] {
                [StageInput::Artifact { artifact, .. }] => format!("{} {artifact} step(s)", st.frames_per_step),
                _ => format!("{} frames", st.frames_per_step),
            },
            st.feat_dim,
            ins.join(", "),
            outs.join(", "),
            st.lstm.len(),
            st.recurrent.len(),
            if st.primed { " (primed)" } else { "" },
            ops.join(", ")
        );
    }
    for sp in &s.spotters {
        let c = match &sp.classifier {
            ClassifierSpec::MovingAverageMaxPool { window, trim_tail } => {
                format!("moving-average-max-pool window {window} trim-tail {trim_tail}")
            }
            ClassifierSpec::ExponentialDecay { rate } => format!("exponential-decay {rate}"),
            ClassifierSpec::WeightedGeometricAverage { weights } => format!("weighted-geometric {weights:?}"),
        };
        let t = &sp.thresholds;
        let opt = |x: Option<f32>| x.map_or("-".into(), |x| x.to_string());
        let comps: Vec<String> = sp.components.iter().map(|w| format!("{}[{}]", w.component, w.class)).collect();
        let _ = writeln!(
            o,
            "spotter {}{} on {}: {}; {c}; accept {} notify {} min {} max {}; {} overrides; lock {}",
            sp.name,
            if sp.auxiliary { " (auxiliary)" } else { "" },
            sp.scorer,
            comps.join(" + "),
            t.accept,
            opt(t.notify),
            opt(t.min_accept),
            opt(t.max_accept),
            t.overrides.len(),
            sp.lock
        );
    }
    if let Some(m) = &s.machine {
        let _ = write!(o, "machine: initial {}", m.states[m.initial].name);
        for st in &m.states {
            let r: Vec<String> = st.rules.iter().map(|(k, n)| format!("{k}->{}", m.states[*n].name)).collect();
            let t = st.timeout.map(|(d, n)| format!("; timeout {d}->{}", m.states[n].name)).unwrap_or_default();
            let _ = write!(o, "\n  {}: {}{t}", st.name, r.join(", "));
        }
        o.push('\n');
    }
    Ok(o)
}
