//! kw.cfg.json, decoders.cfg.json, nemort configs, words.shrunk.txt and op.cfg.json. Known keys whose effect is not
//! implemented are pinned to the validated sets' values; anything else is refused naming file and JSON path.

use super::pconfig::{base, join, read_text};
use crate::json::{Doc, Json, Obj};
use crate::Error;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq)]
pub struct WordComponent {
    pub class: usize,
    pub component: String,
    pub start_end: Option<String>,
    pub start_index: usize,
    pub end_index: usize,
    pub history: Option<usize>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ClassifierSpec {
    MovingAverageMaxPool { window: usize, trim_tail: usize },
    ExponentialDecay { rate: f32 },
    WeightedGeometricAverage { weights: Vec<f32> },
}

#[derive(Debug, Clone, PartialEq)]
pub struct Override {
    pub prop: String,
    pub equals: i32,
    pub accept: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Thresholds {
    pub accept: f32,
    pub notify: Option<f32>,
    pub min_accept: Option<f32>,
    pub max_accept: Option<f32>,
    /// (threshold, probability); the probabilistic accept itself is not reproduced.
    pub probabilistic: Option<(f32, f32)>,
    pub overrides: Vec<Override>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SpotterSpec {
    pub name: String,
    pub auxiliary: bool,
    pub scorer: String,
    pub components: Vec<WordComponent>,
    pub classifier: ClassifierSpec,
    /// Exponential-decay rate per component.
    pub smoothers: Vec<f32>,
    pub thresholds: Thresholds,
    pub kw_window: u32,
    /// Frames.
    pub lock: u32,
    pub notify_lock: u32,
    pub prob_lock: u32,
    /// Routes results to libpryon's local-command callback and metrics; detections are the same.
    pub local_command: bool,
}

/// libpryon's defensive decoder restart: `max_events` accepts of one keyword within `span_ms`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RestartSpec {
    pub max_events: u32,
    pub span_ms: u32,
}

/// libpryon's values when kw.cfg has no restart-trigger (0x76f658).
pub const RESTART_DEFAULT: RestartSpec = RestartSpec { max_events: 12, span_ms: 60000 };

#[derive(Debug, Clone, PartialEq)]
pub struct MachineSpec {
    pub initial: usize,
    pub states: Vec<MachineState>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MachineState {
    pub name: String,
    /// (keyword, next state index).
    pub rules: Vec<(String, usize)>,
    /// (duration in frames, next state index).
    pub timeout: Option<(u32, usize)>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OutputType {
    Decoding,
    Alignment,
}

impl OutputType {
    fn parse(d: &Doc, o: &Obj, key: &str) -> Result<OutputType, Error> {
        match d.str(o, key)? {
            "decoding-scores" => Ok(OutputType::Decoding),
            "time-alignment-scores" => Ok(OutputType::Alignment),
            s => Err(d.unsup(format!("{} = {s} (implemented: decoding-scores, time-alignment-scores)", o.at(key)))),
        }
    }
}

pub(crate) struct NemortOutput {
    pub component: String,
    /// `tensor_name`: the artifact later stages read this output as.
    pub artifact: Option<String>,
    pub kind: OutputType,
}

/// A nemort scorer config: the graph, its single input (`None` = audio, else the artifact it reads), outputs and
/// externalised state pairs.
pub(crate) struct Nemort {
    pub file: PathBuf,
    pub graph: PathBuf,
    pub input: (String, Option<String>),
    pub outputs: Vec<NemortOutput>,
    pub lstm: Vec<(String, String)>,
    pub recurrent: Vec<(String, String)>,
}

const NEMORT_KEYS: &[&str] = &[
    "object_type_name",
    "object_type_version",
    "file_format",
    "model_filepath",
    "model_component_mapping",
    "inputs",
    "outputs",
    "lstm_components",
    "recurrent_components_input",
    "recurrent_components_output",
    "recurrent_state_initialization",
];

type Io = (Vec<(String, Option<String>)>, Vec<NemortOutput>);

/// The single-graph schema (firmware): `model_component_mapping` with `input_map` / `output_map`.
fn old_schema(d: &Doc, m: &Obj) -> Result<Io, Error> {
    d.pin_str(m, "object_type_name", &["DNNModelComponentMapping"], true)?;
    d.pin_num(m, "object_type_version", 2.0, true)?;
    let mut ins = Vec::new();
    for i in d.items(m, "input_map", &["component_name", "channel_type"])? {
        d.pin_str(&i, "channel_type", &["beamformed"], true)?;
        ins.push((d.str(&i, "component_name")?.to_string(), None));
    }
    let mut outs = Vec::new();
    for o in d.items(m, "output_map", &["component_name", "output_type"])? {
        let c = d.str(&o, "component_name")?.to_string();
        outs.push(NemortOutput { component: c, artifact: None, kind: OutputType::parse(d, &o, "output_type")? });
    }
    Ok((ins, outs))
}

/// The DAVS schema: `inputs[]` reading audio or a tensor artifact, `outputs[]` optionally published by tensor name.
fn new_schema(d: &Doc, top: &Obj) -> Result<Io, Error> {
    let mut ins = Vec::new();
    for i in d.items(top, "inputs", &["input_name", "input_source", "tensor_name", "channel_type"])? {
        d.pin_str(&i, "channel_type", &["beamformed"], false)?;
        let name = d.str(&i, "input_name")?.to_string();
        match (d.str(&i, "input_source")?, d.opt_str(&i, "tensor_name")?) {
            ("audio", None) => ins.push((name, None)),
            ("tensor_artifact", Some(t)) => ins.push((name, Some(t.to_string()))),
            ("tensor_artifact", None) => return Err(d.bad(format!("missing {}", i.at("tensor_name")))),
            (s, _) => {
                let k = i.at("input_source");
                return Err(d.unsup(format!("{k} = {s} (implemented: audio, tensor_artifact with tensor_name)")));
            }
        }
    }
    let mut outs = Vec::new();
    for o in d.items(top, "outputs", &["component_name", "output_type", "tensor_name"])? {
        let c = d.str(&o, "component_name")?.to_string();
        let artifact = d.opt_str(&o, "tensor_name")?.map(str::to_string);
        outs.push(NemortOutput { component: c, artifact, kind: OutputType::parse(d, &o, "output_type")? });
    }
    Ok((ins, outs))
}

pub(crate) fn nemort(path: &Path) -> Result<Nemort, Error> {
    let d = Doc::load(path)?;
    let top = d.top(NEMORT_KEYS)?;
    d.pin_str(&top, "object_type_name", &["nemort_scorer_config"], true)?;
    d.pin_num(&top, "object_type_version", 1.0, true)?;
    d.pin_str(&top, "file_format", &["ONNX"], true)?;
    d.pin_bool(&top, "recurrent_state_initialization", false, true)?;
    let graph = join(base(path), d.str(&top, "model_filepath")?);
    let mapping = ["object_type_name", "object_type_version", "input_map", "output_map"];
    let (inputs, outputs) = match d.opt_sub(&top, "model_component_mapping", &mapping)? {
        Some(_) if top.get("inputs").is_some() || top.get("outputs").is_some() => {
            return Err(d.unsup("model_component_mapping together with inputs/outputs"));
        }
        Some(m) => old_schema(&d, &m)?,
        None => new_schema(&d, &top)?,
    };
    let n = inputs.len();
    let Ok([input]) = <[_; 1]>::try_from(inputs) else {
        return Err(d.unsup(format!("{n} graph inputs (one implemented)")));
    };
    if outputs.is_empty() {
        return Err(d.bad("no outputs"));
    }
    let mut lstm = Vec::new();
    if top.get("lstm_components").is_some() {
        for p in d.items(&top, "lstm_components", &["input_component", "output_component"])? {
            lstm.push((d.str(&p, "input_component")?.to_string(), d.str(&p, "output_component")?.to_string()));
        }
    }
    let recurrent = match (top.get("recurrent_components_input"), top.get("recurrent_components_output")) {
        (None, None) => Vec::new(),
        (Some(_), Some(_)) => {
            let i = d.strings(&top, "recurrent_components_input")?;
            let o = d.strings(&top, "recurrent_components_output")?;
            if i.len() != o.len() {
                return Err(d.bad(format!("{} recurrent inputs but {} outputs", i.len(), o.len())));
            }
            i.into_iter().zip(o).collect()
        }
        _ => return Err(d.bad("recurrent_components_input and _output must be given together")),
    };
    Ok(Nemort { file: path.to_path_buf(), graph, input, outputs, lstm, recurrent })
}

/// One `scorers[]` entry of a multi-scorer config (encoder.cfg.json, decoders.cfg.json).
pub(crate) struct ScorerCfg {
    pub name: String,
    pub nemort: PathBuf,
    pub frames: usize,
    pub feat_dim: usize,
    pub outputs: usize,
}

const SCORER_KEYS: &[&str] = &[
    "name",
    "scorer.model_filetype",
    "scorer.model_filepath",
    "scorer.initialize.frame_count",
    "scorer.initialize.publish_frame_count",
    "scorer.feat.dimensions",
    "scorer.num_dnn_outputs",
    "scorer.score_upsampling.enable",
    "scorer.score_upsampling.default_skip_frame_count",
    "checksum",
    "config_type",
    "config_version",
];

/// The `scorers[]` objects of a multi-scorer config, keys checked, values not yet.
pub(crate) fn scorers(d: &Doc) -> Result<Vec<Obj<'_>>, Error> {
    let top = d.top(&["config_type", "config_version", "scorers"])?;
    d.pin_str(&top, "config_type", &["multi_scorer_config"], true)?;
    d.pin_num(&top, "config_version", 1.0, true)?;
    let s = d.items(&top, "scorers", SCORER_KEYS)?;
    for (i, a) in s.iter().enumerate() {
        let n = d.str(a, "name")?;
        if s[..i].iter().any(|b| b.get("name") == a.get("name")) {
            return Err(d.bad(format!("scorer {n} listed twice")));
        }
    }
    Ok(s)
}

/// Validates one scorer entry; only entries a stage uses are validated.
pub(crate) fn scorer(d: &Doc, o: &Obj) -> Result<ScorerCfg, Error> {
    d.pin_str(o, "scorer.model_filetype", &["nemort-base"], true)?;
    d.pin_str(o, "config_type", &["scorer-config"], true)?;
    d.pin_num(o, "config_version", 1.0, true)?;
    d.pin_num(o, "scorer.initialize.publish_frame_count", 89.0, false)?;
    let frames: usize = d.int(o, "scorer.initialize.frame_count")?;
    upsampling(d, o, frames)?;
    d.opt_str(o, "checksum")?;
    Ok(ScorerCfg {
        name: d.str(o, "name")?.to_string(),
        nemort: join(base(&d.path), d.str(o, "scorer.model_filepath")?),
        frames,
        feat_dim: d.int(o, "scorer.feat.dimensions")?,
        outputs: d.int(o, "scorer.num_dnn_outputs")?,
    })
}

/// Score upsampling repeats a step's scores over the frames it consumed: implemented only as exactly that.
fn upsampling(d: &Doc, o: &Obj, frames: usize) -> Result<(), Error> {
    if frames == 0 {
        return Err(d.unsup(format!("{} = 0", o.at("scorer.initialize.frame_count"))));
    }
    d.pin_num(o, "scorer.score_upsampling.enable", 1.0, true)?;
    let skip: usize = d.int(o, "scorer.score_upsampling.default_skip_frame_count")?;
    if skip + 1 != frames {
        let k = o.at("scorer.score_upsampling.default_skip_frame_count");
        return Err(d.unsup(format!("{k} = {skip} with frame_count {frames} (implemented: frame_count - 1)")));
    }
    Ok(())
}

fn component(d: &Doc, w: &Obj) -> Result<WordComponent, Error> {
    let sc = d.sub(w, "scorer-output-component", &["name", "posterior-type"])?;
    d.pin_str(&sc, "posterior-type", &["decoding-scores"], true)?;
    let se = d.opt_sub(w, "keyword-start-end-component", &["name", "posterior-type"])?;
    if let Some(se) = &se {
        d.pin_str(se, "posterior-type", &["time-alignment-scores"], true)?;
    }
    Ok(WordComponent {
        class: d.int(w, "scorer-index")?,
        component: d.str(&sc, "name")?.to_string(),
        start_end: se.map(|se| d.str(&se, "name").map(str::to_string)).transpose()?,
        start_index: d.int(w, "keyword-start-scorer-index")?,
        end_index: d.int(w, "keyword-end-scorer-index")?,
        history: d.opt_int(w, "word-score-history-size")?,
    })
}

fn classifier(d: &Doc, c: &Obj, components: usize) -> Result<ClassifierSpec, Error> {
    let single = |spec: ClassifierSpec| {
        if components == 1 {
            Ok(spec)
        } else {
            Err(d.unsup(format!("{} with {components} word components (one implemented)", c.at("type"))))
        }
    };
    let check = |known: &[&str]| c.check(known).map_err(|w| d.unsup(w));
    match d.str(c, "type")? {
        "moving-average-max-pool" => {
            check(&["type", "window-size", "trim-tail"])?;
            // libpryon's defaults (keyword_score_smoother.cpp, 0x9b0e48)
            let window: usize = d.opt_int(c, "window-size")?.unwrap_or(2);
            if window == 0 {
                return Err(d.unsup(format!("{} = 0", c.at("window-size"))));
            }
            let trim_tail = d.opt_int(c, "trim-tail")?.unwrap_or(0);
            single(ClassifierSpec::MovingAverageMaxPool { window, trim_tail })
        }
        "exponential-decay" => {
            check(&["type", "decay-rate"])?;
            single(ClassifierSpec::ExponentialDecay { rate: d.f32(c, "decay-rate")? })
        }
        "weighted-geometric-average-aggregator" => {
            check(&["type", "weights"])?;
            let mut weights = Vec::new();
            for (i, v) in d.arr(c, "weights")?.iter().enumerate() {
                let x = match v {
                    Json::Num(n) => n.parse::<f32>().ok().filter(|x| x.is_finite()),
                    _ => None,
                };
                weights.push(x.ok_or_else(|| d.bad(format!("{}[{i}] is not a number", c.at("weights"))))?);
            }
            if weights.len() != components {
                let n = weights.len();
                return Err(d.unsup(format!("{} has {n} weights for {components} word components", c.at("weights"))));
            }
            Ok(ClassifierSpec::WeightedGeometricAverage { weights })
        }
        t => Err(d.unsup(format!("{} = {t}", c.at("type")))),
    }
}

const THRESHOLD_KEYS: &[&str] = &[
    "accept-threshold",
    "notify-threshold",
    "min-accept-threshold",
    "max-accept-threshold",
    "probabilistic-accept-threshold",
    "probabilistic-accept-probability",
    "probabilistic-accept-lock-period",
    "overrides",
];

fn thresholds(d: &Doc, t: &Obj) -> Result<Thresholds, Error> {
    // libpryon reads the lock period at spotter level only; this one is ignored (keyword_spotter_config.cpp)
    d.opt_int::<u32>(t, "probabilistic-accept-lock-period")?;
    let prob = (d.opt_f32(t, "probabilistic-accept-threshold")?, d.opt_f32(t, "probabilistic-accept-probability")?);
    let probabilistic = match prob {
        (Some(a), Some(b)) => Some((a, b)),
        (None, None) => None,
        _ => return Err(d.unsup(format!("{}.probabilistic-accept-* given partially", t.path))),
    };
    let mut overrides = Vec::new();
    if t.get("overrides").is_some() {
        for o in d.items(t, "overrides", &["accept-threshold", "clientProperties"])? {
            let props = d.items(&o, "clientProperties", &["equals", "name"])?;
            let [prop] = &props[..] else {
                let n = props.len();
                return Err(d.unsup(format!("{} with {n} entries (one implemented)", o.at("clientProperties"))));
            };
            overrides.push(Override {
                prop: d.str(prop, "name")?.to_string(),
                equals: d.int(prop, "equals")?,
                accept: d.f32(&o, "accept-threshold")?,
            });
        }
    }
    Ok(Thresholds {
        accept: d.f32(t, "accept-threshold")?,
        notify: d.opt_f32(t, "notify-threshold")?,
        min_accept: d.opt_f32(t, "min-accept-threshold")?,
        max_accept: d.opt_f32(t, "max-accept-threshold")?,
        probabilistic,
        overrides,
    })
}

/// KeywordMultiWordSpotter's own checks (keyword_multi_word_spotter.cpp; factory 0x9b98c4): one smoother per word
/// component, at most two components, the last one remembering only its current score.
fn multi_word(d: &Doc, s: &Obj, components: &[WordComponent], smoothers: &[f32]) -> Result<(), Error> {
    let wca = s.at("word-component-array");
    if smoothers.len() != components.len() {
        let k = components.len();
        return Err(d.unsup(format!("{} with {} entries for {k} word components", s.at("posterior-smoothers"), smoothers.len())));
    }
    if components.len() > 2 {
        return Err(d.unsup(format!("{wca} has {} entries (libpryon takes at most 2)", components.len())));
    }
    for (i, w) in components.iter().enumerate() {
        match w.history {
            // a history of frames longer than the orchestra's frame buffer is implausible
            Some(h) if (1..=ORCHESTRA_FRAMES).contains(&h) => {}
            h => {
                let want = format!("implemented: 1 to {ORCHESTRA_FRAMES}, the keyword-orchestra buffer-size");
                return Err(d.unsup(format!("{wca}[{i}].word-score-history-size = {h:?} ({want})")));
            }
        }
    }
    if components.last().and_then(|w| w.history) != Some(1) {
        return Err(d.unsup(format!("{wca}: the last word-score-history-size must be 1 (libpryon asserts it)")));
    }
    if let Some(r) = smoothers.iter().find(|r| !(0.0..=1.0).contains(*r)) {
        return Err(d.unsup(format!("{}: decay-rate {r} outside [0, 1]", s.at("posterior-smoothers"))));
    }
    Ok(())
}

const SPOTTER_KEYS: &[&str] = &[
    "classification-thresholds",
    "word-component-array",
    "keyword-feature-window-length",
    "lock-period",
    "notify-lock-period",
    "probabilistic-accept-lock-period",
    "spotter-type",
    "posterior-smoothers",
    "is-local-command",
];

/// keyword-orchestra buffer-size, in frames.
const ORCHESTRA_FRAMES: usize = 200;

const COMPONENT_KEYS: &[&str] = &[
    "scorer-index",
    "scorer-output-component",
    "keyword-start-scorer-index",
    "keyword-end-scorer-index",
    "keyword-start-end-component",
    "word-score-history-size",
];

fn spotter(d: &Doc, e: &Obj, single: Option<&str>) -> Result<SpotterSpec, Error> {
    let s = d.sub(e, "spotter", SPOTTER_KEYS)?;
    d.pin_str(&s, "spotter-type", &["DETECTION_AUXILIARY"], false)?;
    let wca = d.items(&s, "word-component-array", COMPONENT_KEYS)?;
    let components = wca.iter().map(|w| component(d, w)).collect::<Result<Vec<_>, _>>()?;
    if components.is_empty() {
        return Err(d.bad(format!("{} is empty", s.at("word-component-array"))));
    }
    let mut smoothers = Vec::new();
    if s.get("posterior-smoothers").is_some() {
        for p in d.items(&s, "posterior-smoothers", &["type", "decay-rate"])? {
            d.pin_str(&p, "type", &["exponential-decay"], true)?;
            smoothers.push(d.f32(&p, "decay-rate")?);
        }
        if smoothers.len() != components.len() {
            let (n, k) = (smoothers.len(), components.len());
            return Err(d.unsup(format!("{} has {n} entries for {k} word components", s.at("posterior-smoothers"))));
        }
    }
    let c = d.get(e, "classifier")?;
    let c = c.obj(&e.at("classifier")).ok_or_else(|| d.bad(format!("{} is not an object", e.at("classifier"))))?;
    if c.get("type").is_some_and(|t| matches!(t, Json::Str(s) if s == "weighted-geometric-average-aggregator")) {
        multi_word(d, &s, &components, &smoothers)?;
    }
    let scorer = match (single, e.get("scorers")) {
        (Some(s), None) => s.to_string(),
        (None, Some(_)) => match &d.strings(e, "scorers")?[..] {
            [one] => one.clone(),
            v => return Err(d.unsup(format!("{} lists {} scorers (one implemented)", e.at("scorers"), v.len()))),
        },
        (Some(_), Some(_)) => return Err(d.unsup(format!("{} in a single-scorer set", e.at("scorers")))),
        (None, None) => return Err(d.bad(format!("missing {}", e.at("scorers")))),
    };
    // libpryon's defaults (keyword_spotter_config.cpp, 0x9b7460); a threshold-level lock period is not read
    let lock = d.opt_int(&s, "lock-period")?.unwrap_or(25);
    let sub_lock = |key: &str| -> Result<u32, Error> {
        let v = d.opt_int(&s, key)?.unwrap_or(lock);
        if v > lock {
            return Err(d.bad(format!("{} = {v} exceeds lock-period {lock}", s.at(key))));
        }
        Ok(v)
    };
    let (notify_lock, prob_lock) = (sub_lock("notify-lock-period")?, sub_lock("probabilistic-accept-lock-period")?);
    let kw_window = d.int(&s, "keyword-feature-window-length")?;
    // libpryon defaults it on for a spotter named STOP (0x9b7460)
    let name = d.str(e, "name")?;
    let local_command = d.opt_bool(&s, "is-local-command")?.unwrap_or(name == "STOP");
    if kw_window == 0 {
        return Err(d.unsup(format!("{} = 0", s.at("keyword-feature-window-length"))));
    }
    Ok(SpotterSpec {
        name: d.str(e, "name")?.to_string(),
        auxiliary: s.get("spotter-type").is_some(),
        scorer,
        classifier: classifier(d, &c, components.len())?,
        components,
        smoothers,
        thresholds: thresholds(d, &d.sub(&s, "classification-thresholds", THRESHOLD_KEYS)?)?,
        kw_window,
        lock,
        notify_lock,
        prob_lock,
        local_command,
    })
}

/// kw.cfg.json's spotters and restart trigger. `single` is the firmware's one scorer (its spotters name none); DAVS
/// spotters name exactly one. `words_file` is the id2word file name the word-symbol-table must agree with.
pub(crate) fn spotters(
    path: &Path,
    single: Option<&str>,
    words_file: &str,
) -> Result<(Vec<SpotterSpec>, RestartSpec), Error> {
    let d = Doc::load(path)?;
    let top = d.top(&[
        "acoustic-model",
        "keyword-orchestra",
        "keyword-orchestra-monitor",
        "restart-trigger",
        "object_type_name",
        "object_type_version",
        "word-symbol-table",
    ])?;
    d.pin_str(&top, "object_type_name", &["keyword-spotter-orchestra-config"], true)?;
    d.pin_num(&top, "object_type_version", 2.0, true)?;
    let am = d.sub(&top, "acoustic-model", &["non-speech-pdf", "speech-pdf"])?;
    d.pin_num(&am, "non-speech-pdf", 0.0, true)?;
    d.pin_num(&am, "speech-pdf", 0.0, true)?;
    let mut restart = RESTART_DEFAULT;
    if let Some(r) = d.opt_sub(&top, "restart-trigger", &["max-event-count", "evaluation-span-msec"])? {
        restart = RestartSpec { max_events: d.int(&r, "max-event-count")?, span_ms: d.int(&r, "evaluation-span-msec")? };
        if restart.max_events == 0 || restart.span_ms == 0 {
            return Err(d.unsup(format!("{} = {restart:?} (implemented: both positive)", top.at("restart-trigger"))));
        }
    }
    let wst = d.sub(&top, "word-symbol-table", &["filename"])?;
    let f = d.str(&wst, "filename")?;
    if f != words_file {
        return Err(d.unsup(format!("{} = {f}, but id2word.mapping_filepath names {words_file}", wst.at("filename"))));
    }
    let orch = d.sub(&top, "keyword-orchestra", &["buffer-size", "keyword-spotters"])?;
    d.pin_num(&orch, "buffer-size", ORCHESTRA_FRAMES as f32, true)?;
    let list = d.items(&orch, "keyword-spotters", &["name", "classifier", "scorers", "spotter"])?;
    if list.is_empty() {
        return Err(d.bad(format!("{} is empty", orch.at("keyword-spotters"))));
    }
    let out = list.iter().map(|e| spotter(&d, e, single)).collect::<Result<Vec<_>, _>>()?;
    for (i, a) in out.iter().enumerate() {
        if out[..i].iter().any(|b| b.name == a.name && b.auxiliary == a.auxiliary) {
            let kind = if a.auxiliary { "auxiliary" } else { "primary" };
            return Err(d.unsup(format!("two {kind} spotters named {}", a.name)));
        }
        if a.auxiliary && !out.iter().any(|b| b.name == a.name && !b.auxiliary) {
            return Err(d.unsup(format!("auxiliary spotter {} without a primary", a.name)));
        }
    }
    Ok((out, restart))
}

/// tensor_artifacts.json: the names of the in-memory artifacts stages may publish.
pub(crate) fn tensor_artifacts(path: &Path) -> Result<Vec<String>, Error> {
    let d = Doc::load(path)?;
    let top = d.top(&["tensor_artifacts", "object_type_name", "object_type_version"])?;
    d.pin_str(&top, "object_type_name", &["tensor_artifacts_list_config"], true)?;
    d.pin_num(&top, "object_type_version", 1.0, true)?;
    let mut names: Vec<String> = Vec::new();
    for a in d.items(&top, "tensor_artifacts", &["name", "format", "protocol"])? {
        d.pin_str(&a, "format", &["in-memory"], true)?;
        d.pin_str(&a, "protocol", &["runtime"], true)?;
        let n = d.str(&a, "name")?;
        if names.iter().any(|x| x == n) {
            return Err(d.bad(format!("artifact {n} listed twice")));
        }
        names.push(n.to_string());
    }
    Ok(names)
}

/// `word index` lines, in file order.
pub(crate) fn words(path: &Path) -> Result<Vec<(u32, String)>, Error> {
    let text = read_text(path)?;
    let bad = |m: String| Error::Config(format!("{}: {m}", path.display()));
    let mut out: Vec<(u32, String)> = Vec::new();
    for (no, line) in text.lines().enumerate() {
        let parts: Vec<&str> = line.split_whitespace().collect();
        let Some((last, word)) = parts.split_last() else { continue };
        let idx: u32 = last.parse().map_err(|_| bad(format!("line {} is not `word index`", no + 1)))?;
        if word.is_empty() {
            return Err(bad(format!("line {} is not `word index`", no + 1)));
        }
        let word = word.join(" ");
        if out.iter().any(|(i, w)| *i == idx || *w == word) {
            return Err(bad(format!("line {}: {word} {idx} repeats an index or word", no + 1)));
        }
        out.push((idx, word));
    }
    Ok(out)
}

/// op.cfg.json: states are the top-level objects besides the type fields; rules must name `keywords`.
pub(crate) fn machine(path: &Path, keywords: &[&str]) -> Result<MachineSpec, Error> {
    let d = Doc::load(path)?;
    let top = d.root.obj("").ok_or_else(|| d.bad("top level is not an object"))?;
    let meta = ["object_type_name", "object_type_version", "initial"];
    let names: Vec<&str> = top.fields.iter().map(|(k, _)| k.as_str()).filter(|k| !meta.contains(k)).collect();
    let known: Vec<&str> = meta.iter().copied().chain(names.iter().copied()).collect();
    top.check(&known).map_err(|w| d.unsup(w))?;
    d.pin_str(&top, "object_type_name", &["keyword-spotter-machine-config"], true)?;
    d.pin_num(&top, "object_type_version", 1.0, true)?;
    let index = |o: &Obj, key: &str| -> Result<usize, Error> {
        let n = d.str(o, key)?;
        names.iter().position(|s| *s == n).ok_or_else(|| d.unsup(format!("{} = {n}, not a state", o.at(key))))
    };
    let mut states = Vec::new();
    for name in &names {
        let s = d.sub(&top, name, &["rules", "timeout"])?;
        let mut rules = Vec::new();
        for r in d.items(&s, "rules", &["name", "next"])? {
            let kw = d.str(&r, "name")?;
            if !keywords.contains(&kw) {
                return Err(d.unsup(format!("{} = {kw}, not a keyword of this set", r.at("name"))));
            }
            rules.push((kw.to_string(), index(&r, "next")?));
        }
        let timeout = match d.opt_sub(&s, "timeout", &["duration", "next"])? {
            Some(t) => Some((d.int(&t, "duration")?, index(&t, "next")?)),
            None => None,
        };
        states.push(MachineState { name: name.to_string(), rules, timeout });
    }
    // libpryon enables a spotter only while the state names its keyword; one named nowhere is not read here
    if let Some(k) = keywords.iter().find(|k| !states.iter().any(|s| s.rules.iter().any(|(r, _)| r == *k))) {
        return Err(d.unsup(format!("keyword {k} is in no state's rules")));
    }
    Ok(MachineSpec { initial: index(&top, "initial")?, states })
}
