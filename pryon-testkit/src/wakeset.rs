//! Synthetic wake model sets in the firmware and DAVS layouts: our own small graphs and configs, laid out like the
//! real sets so the loader, pipeline and rules can be tested without Amazon files.

mod graphs;

pub use graphs::{conv_decoder_graph, encoder_graph, fw_graph, lstm_decoder_graph};
use std::path::Path;

#[derive(Clone, Copy, Debug, Default)]
pub struct WakeSetOpts {
    pub seed: u64,
    /// Pins every decoder's class-1 softmax score near this value whatever the input.
    pub force_score: Option<f32>,
}

/// Embedding width of the synthetic encoder.
pub const EMBED: usize = 16;
pub const DAVS_LANES: usize = 16;
pub const FW_LANES: usize = 25;

fn put(dir: &Path, rel: &str, data: impl AsRef<[u8]>) {
    let p = dir.join(rel);
    std::fs::create_dir_all(p.parent().unwrap()).unwrap();
    std::fs::write(p, data).unwrap();
}

/// FNV-1a-64 hex of `data` (stands in for the MD5 the real configs carry).
fn fnv(data: &[u8]) -> String {
    let h = data.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &b| (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3));
    format!("{h:016x}")
}

const MANIFEST: &str = r#"{
    "default_model_name": "pryon",
    "models": { "pryon": { "config_file_path": "pryon.config" } },
    "object_type_name": "ModelSetManifest",
    "object_type_version": 1
}
"#;

const FE: &str = r#"fe.type = "LFBE"

fe.audio_analysis.dither = 0.0
fe.audio_analysis.frame_length_milli = 25
fe.audio_analysis.frame_shift_milli = 10
fe.audio_analysis.preemphasis_coeff = 0.96875
fe.audio_analysis.remove_dc_offset = 1
fe.audio_analysis.round_to_power_of_two = 1
fe.audio_analysis.samplerate = 16000
fe.audio_analysis.window_type = "povey"

fe.cep.raw_energy = 1
fe.cep.use_energy = 0

fe.feat.dimensions = 64
fe.feat.stack_left = 0
fe.feat.stack_right = 0
fe.feat.transform_filepath = ""
fe.feat.do_mean_norm = 0
fe.feat.do_variance_norm = 0
fe.feat.affine_transform_filepath = ""

fe.mel_fbank.high_freq = 7200
fe.mel_fbank.low_freq = 80
fe.mel_fbank.num_bins = 64
"#;

const OP_FW: &str = r#"{
  "object_type_name": "keyword-spotter-machine-config",
  "object_type_version": 1,
  "initial": "sleep",
  "awake": {
    "rules": [ { "name": "ALEXA", "next": "awake" }, { "name": "STOP", "next": "sleep" } ],
    "timeout": { "duration": 175, "next": "sleep" }
  },
  "sleep": { "rules": [ { "name": "ALEXA", "next": "awake" } ] }
}
"#;

const OP_DAVS: &str = r#"{
  "object_type_name": "keyword-spotter-machine-config",
  "object_type_version": 1,
  "initial": "sleep",
  "awake": {
    "rules": [
      { "name": "ALEXA", "next": "awake" },
      { "name": "HEY_DISNEY", "next": "awake" },
      { "name": "STOP", "next": "sleep" }
    ],
    "timeout": { "duration": 175, "next": "sleep" }
  },
  "sleep": { "rules": [ { "name": "ALEXA", "next": "awake" }, { "name": "HEY_DISNEY", "next": "awake" } ] }
}
"#;

fn overrides(accepts: [f32; 3]) -> String {
    let names = ["AlarmState", "AudioPlayerState", "TTSPlayerState"];
    let items: Vec<String> = names
        .iter()
        .zip(accepts)
        .map(|(n, a)| {
            format!(r#"{{ "accept-threshold": {a}, "clientProperties": [ {{ "equals": 1, "name": "{n}" }} ] }}"#)
        })
        .collect();
    format!("[ {} ]", items.join(", "))
}

fn component(class: usize, scores: &str, se: &str, history: Option<usize>) -> String {
    let h = history.map(|h| format!(r#""word-score-history-size": {h}, "#)).unwrap_or_default();
    format!(
        r#"{{ "scorer-index": {class},
              "scorer-output-component": {{ "name": "{scores}", "posterior-type": "decoding-scores" }},
              {h}"keyword-start-scorer-index": 0, "keyword-end-scorer-index": 1,
              "keyword-start-end-component": {{ "name": "{se}", "posterior-type": "time-alignment-scores" }} }}"#
    )
}

fn kw_cfg(spotters: &[String], extra: &str) -> String {
    format!(
        r#"{{
  "acoustic-model": {{ "non-speech-pdf": 0, "speech-pdf": 0 }},
  "keyword-orchestra": {{ "buffer-size": 200, "keyword-spotters": [
{}
  ] }},
  "keyword-orchestra-monitor": {{ "statistics": {{
    "classification": {{ "histogram": {{ "max": 1, "min": 0, "size": 50 }} }},
    "segmentation": {{ "histogram": {{ "max": 1, "min": 0, "size": 1 }} }} }} }},{extra}
  "object_type_name": "keyword-spotter-orchestra-config",
  "object_type_version": 2,
  "word-symbol-table": {{ "filename": "words.shrunk.txt" }}
}}
"#,
        spotters.join(",\n")
    )
}

/// `lstm_components` entries.
fn pairs(p: &[(&str, &str)]) -> Vec<String> {
    p.iter().map(|(i, o)| format!(r#"{{ "input_component": "{i}", "output_component": "{o}" }}"#)).collect()
}

/// A firmware-layout set (one 6-frame graph, ALEXA and STOP) in `dir`.
pub fn write_fw_set(dir: &Path, o: &WakeSetOpts) {
    put(dir, "pryon.manifest", MANIFEST);
    let cfg = format!(
        "{FE}\nscorer.model_filetype = \"nemort-base\"
scorer.nemort_config_json_filepath = \"nemort_scorer_config.json\"
scorer.initialize.frame_count = 6\nscorer.score_upsampling.enable = 1
scorer.score_upsampling.default_skip_frame_count = 5

search.decoder_type = \"kaldi-key-phrase\"\nkeyword_spotter.config_filepath = \"kw.cfg.json\"
keyword_spotter.op_config_filepath = \"op.cfg.json\"\nkeyword_spotter.emit_nearmiss = 1
id2word.mapping_filepath = \"words.shrunk.txt\"\n"
    );
    put(dir, "pryon.config", cfg);
    put(dir, "synthetic.streaming.onnx", fw_graph(o));
    let lstm = pairs(&[
        ("input_h_ww", "lstm_ww_1"),
        ("input_c_ww", "lstm_ww_1_1"),
        ("input_h_se", "lstm_se_1"),
        ("input_c_se", "lstm_se_1_1"),
    ]);
    put(
        dir,
        "nemort_scorer_config.json",
        format!(
            r#"{{
  "object_type_name": "nemort_scorer_config",
  "object_type_version": 1,
  "file_format": "ONNX",
  "model_filepath": "synthetic.streaming.onnx",
  "model_component_mapping": {{
    "object_type_name": "DNNModelComponentMapping",
    "object_type_version": 2,
    "input_map": [ {{ "component_name": "input_feats", "channel_type": "beamformed" }} ],
    "output_map": [
      {{ "component_name": "softmax_output_ww", "output_type": "decoding-scores" }},
      {{ "component_name": "fc_output_se", "output_type": "time-alignment-scores" }}
    ]
  }},
  "lstm_components": [ {} ],
  "recurrent_components_input": [ "streaming_conv_buffer_input_7" ],
  "recurrent_components_output": [ "streaming_conv_buffer_output_7" ],
  "recurrent_state_initialization": false
}}
"#,
            lstm.join(", ")
        ),
    );
    let ma = r#""classifier": { "type": "moving-average-max-pool", "trim-tail": 8, "window-size": 3 }"#;
    let alexa = format!(
        r#"    {{ "name": "ALEXA", {ma}, "spotter": {{
      "classification-thresholds": {{ "accept-threshold": 0.9, "notify-threshold": 0.7, "overrides": {} }},
      "word-component-array": [ {} ],
      "keyword-feature-window-length": 100, "lock-period": 100 }} }}"#,
        overrides([0.8, 0.85, 0.85]),
        component(1, "softmax_output_ww", "fc_output_se", None)
    );
    let stop = format!(
        r#"    {{ "name": "STOP", {ma}, "spotter": {{
      "classification-thresholds": {{ "accept-threshold": 0.5 }},
      "word-component-array": [ {} ],
      "keyword-feature-window-length": 100, "lock-period": 100 }} }}"#,
        component(2, "softmax_output_ww", "fc_output_se", None)
    );
    put(dir, "kw.cfg.json", kw_cfg(&[alexa, stop], ""));
    put(dir, "op.cfg.json", OP_FW);
    put(dir, "words.shrunk.txt", "<eps> 0\nALEXA 1\nSTOP 2\n");
}

fn scorer_entry(name: &str, nemort: &str, frames: usize, feat: usize, outputs: usize, checksum: &str) -> String {
    format!(
        r#"    {{
      "name": "{name}",
      "scorer.model_filetype": "nemort-base",
      "scorer.model_filepath": "{nemort}",
      "scorer.initialize.frame_count": {frames},
      "scorer.feat.dimensions": {feat},
      "scorer.num_dnn_outputs": {outputs},
      "scorer.score_upsampling.enable": 1,
      "scorer.score_upsampling.default_skip_frame_count": {},
      "checksum": "{checksum}",
      "config_type": "scorer-config",
      "config_version": 1
    }}"#,
        frames - 1
    )
}

fn multi_scorer(entries: &[String]) -> String {
    format!(
        "{{\n  \"config_type\": \"multi_scorer_config\",\n  \"config_version\": 1,\n  \"scorers\": [\n{}\n  ]\n}}\n",
        entries.join(",\n")
    )
}

fn nemort_new(model: &str, inputs: &str, outputs: &str, state: &str) -> String {
    format!(
        r#"{{
  "object_type_name": "nemort_scorer_config",
  "object_type_version": 1,
  "file_format": "ONNX",
  "model_filepath": "{model}",
  "inputs": [ {inputs} ],
  "outputs": [ {outputs} ],
{state}
  "recurrent_state_initialization": false
}}
"#
    )
}

/// A DAVS-layout set (encoder, LSTM decoder for ALEXA + verifier, conv decoder for STOP and HEY_DISNEY; unopened
/// dvad/sid entries) in `dir`.
pub fn write_davs_set(dir: &Path, o: &WakeSetOpts) {
    put(dir, "pryon.manifest", MANIFEST);
    put(
        dir,
        "pryon.config",
        [
            "audiofrontendconfig",
            "dvaddecoder",
            "encoder",
            "siddecoder",
            "multi-wakeword-wakeworddecoder",
            "alexa-wakeworddecoder",
            "BDPGeneratedFiles",
        ]
        .iter()
        .map(|d| format!("INCLUDE \"{d}/pryon.config\"\n"))
        .collect::<String>(),
    );
    put(dir, "audiofrontendconfig/pryon.config", FE);
    put(
        dir,
        "encoder/pryon.config",
        "# Keyword\nsearch.decoder_type = \"kaldi-key-phrase\"\n\
         #keyword_spotter.config_filepath = \"kw.cfg.json\" commented out\n\
         keyword_spotter.emit_nearmiss = 1\n\nsearch.tensor_artifacts_list_file = \"tensor_artifacts.json\"\n\
         scorer.single.config_filepath = \"encoder.cfg.json\"\n",
    );
    put(
        dir,
        "encoder/tensor_artifacts.json",
        r#"{ "tensor_artifacts": [ { "name": "encoder_output", "format": "in-memory", "protocol": "runtime" } ],
  "object_type_name": "tensor_artifacts_list_config", "object_type_version": 1 }
"#,
    );
    let enc = encoder_graph(o);
    put(
        dir,
        "encoder/encoder.cfg.json",
        multi_scorer(&[scorer_entry("WW_encoder", "nemort_scorer_config_encoder.json", 8, 64, EMBED, &fnv(&enc))]),
    );
    put(dir, "encoder/primed_encoder.int8-streaming.onnx", &enc);
    put(
        dir,
        "encoder/nemort_scorer_config_encoder.json",
        nemort_new(
            "primed_encoder.int8-streaming.onnx",
            r#"{ "input_name": "input_136", "input_source": "audio", "channel_type": "beamformed" }"#,
            r#"{ "component_name": "encoder_embeddings", "output_type": "decoding-scores",
    "tensor_name": "encoder_output" }"#,
            r#"  "recurrent_components_input": [ "conv_buffer_input/Concat:2" ],
  "recurrent_components_output": [ "conv_buffer_output/Slice:2" ],"#,
        ),
    );

    let dec = lstm_decoder_graph(o);
    put(dir, "alexa-wakeworddecoder/pryon.config", "#scorer.multi.config_filepath_list += alexa_decoder.cfg.json\n");
    put(dir, "alexa-wakeworddecoder/det_vrf_multi_lstm.int8.onnx", &dec);
    let lstm = pairs(&[
        ("scores_h0", "scores_hn"),
        ("scores_c0", "scores_cn"),
        ("alignments_h0", "alignments_hn"),
        ("alignments_c0", "alignments_cn"),
    ]);
    put(
        dir,
        "alexa-wakeworddecoder/nemort_scorer_config_decoder_alexa.json",
        nemort_new(
            "det_vrf_multi_lstm.int8.onnx",
            r#"{ "input_name": "embedding", "input_source": "tensor_artifact", "tensor_name": "encoder_output" }"#,
            r#"{ "component_name": "ww_det_scores", "output_type": "decoding-scores" },
    { "component_name": "ww_vrf_scores", "output_type": "decoding-scores" },
    { "component_name": "alignments", "output_type": "time-alignment-scores" }"#,
            &format!("  \"lstm_components\": [ {} ],", lstm.join(", ")),
        ),
    );

    let stop = conv_decoder_graph(o);
    put(
        dir,
        "multi-wakeword-wakeworddecoder/pryon.config",
        "#scorer.multi.config_filepath_list += stop_decoder.cfg.json\n",
    );
    put(dir, "multi-wakeword-wakeworddecoder/stop_hey_disney.streaming.fixed.int8.onnx", &stop);
    put(
        dir,
        "multi-wakeword-wakeworddecoder/nemort_scorer_config_decoder_stop.json",
        nemort_new(
            "stop_hey_disney.streaming.fixed.int8.onnx",
            r#"{ "input_name": "input_6", "input_source": "tensor_artifact", "tensor_name": "encoder_output" }"#,
            r#"{ "component_name": "mww/posteriors", "output_type": "decoding-scores" },
    { "component_name": "mww/start_end", "output_type": "time-alignment-scores" }"#,
            r#"  "recurrent_components_input": [ "conv_buffer_input/Concat:3" ],
  "recurrent_components_output": [ "conv_buffer_output/Slice:3" ],"#,
        ),
    );

    // DVAD and speaker-ID decoders are listed but never opened: their graphs are not written.
    for (d, n) in [("dvaddecoder", "dvad"), ("siddecoder", "sid")] {
        put(dir, &format!("{d}/pryon.config"), format!("#scorer.multi.config_filepath_list += {n}_decoder.cfg.json\n"));
        put(
            dir,
            &format!("{d}/nemort_scorer_config_decoder_{n}.json"),
            nemort_new(
                &format!("{n}.int8-streaming.onnx"),
                r#"{ "input_name": "input_1", "input_source": "tensor_artifact", "tensor_name": "encoder_output" }"#,
                &format!(r#"{{ "component_name": "{n}/posteriors", "output_type": "decoding-scores" }}"#),
                "",
            ),
        );
    }

    put(
        dir,
        "BDPGeneratedFiles/pryon.config",
        "#Files generated by the synthetic set writer.\nid2word.mapping_filepath = \"words.shrunk.txt\"\n\
         keyword_spotter.op_config_filepath = \"op.cfg.json\"\nscorer.multi.config_filepath = \"decoders.cfg.json\"\n\
         keyword_spotter.config_filepath = \"kw.cfg.json\"\n",
    );
    put(
        dir,
        "BDPGeneratedFiles/decoders.cfg.json",
        multi_scorer(&[
            scorer_entry("dvad", "../dvaddecoder/nemort_scorer_config_decoder_dvad.json", 1, EMBED, 2, "dvad"),
            scorer_entry("sid", "../siddecoder/nemort_scorer_config_decoder_sid.json", 1, EMBED, 64, "sid"),
            scorer_entry(
                "ww_stop",
                "../multi-wakeword-wakeworddecoder/nemort_scorer_config_decoder_stop.json",
                1,
                EMBED,
                4,
                &fnv(&stop),
            ),
            scorer_entry(
                "ww_alexa",
                "../alexa-wakeworddecoder/nemort_scorer_config_decoder_alexa.json",
                1,
                EMBED,
                3,
                &fnv(&dec),
            ),
        ]),
    );
    let stop_spotter = format!(
        r#"    {{ "classifier": {{ "type": "exponential-decay", "decay-rate": 0.6 }},
      "name": "STOP", "scorers": [ "ww_stop" ],
      "spotter": {{ "classification-thresholds": {{ "accept-threshold": 0.6 }},
        "word-component-array": [ {} ],
        "keyword-feature-window-length": 100, "lock-period": 100 }} }}"#,
        component(1, "mww/posteriors", "mww/start_end", None)
    );
    let disney = format!(
        r#"    {{ "classifier": {{ "type": "weighted-geometric-average-aggregator", "weights": [ 1, 1 ] }},
      "name": "HEY_DISNEY", "scorers": [ "ww_stop" ],
      "spotter": {{
        "classification-thresholds": {{ "accept-threshold": 0.45, "notify-threshold": 0.2, "overrides": {} }},
        "word-component-array": [ {}, {} ],
        "posterior-smoothers": [ {{ "type": "exponential-decay", "decay-rate": 0.0 }},
                                 {{ "type": "exponential-decay", "decay-rate": 0.3 }} ],
        "keyword-feature-window-length": 100, "lock-period": 100 }} }}"#,
        overrides([0.45, 0.45, 0.45]),
        component(2, "mww/posteriors", "mww/start_end", Some(100)),
        component(3, "mww/posteriors", "mww/start_end", Some(1))
    );
    let alexa = format!(
        r#"    {{ "name": "ALEXA", "classifier": {{ "type": "moving-average-max-pool", "window-size": 1 }},
      "scorers": [ "ww_alexa" ],
      "spotter": {{ "classification-thresholds": {{ "accept-threshold": 0.92, "min-accept-threshold": 0.85,
          "max-accept-threshold": 0.93, "notify-threshold": 0.3, "probabilistic-accept-threshold": 0.5,
          "probabilistic-accept-probability": 0.001, "probabilistic-accept-lock-period": 50, "overrides": {} }},
        "word-component-array": [ {} ],
        "keyword-feature-window-length": 100, "lock-period": 132 }} }}"#,
        overrides([0.7, 0.86, 0.86]),
        component(1, "ww_det_scores", "alignments", None)
    );
    let verifier = format!(
        r#"    {{ "classifier": {{ "type": "moving-average-max-pool", "window-size": 1 }},
      "name": "ALEXA", "scorers": [ "ww_alexa" ],
      "spotter": {{ "spotter-type": "DETECTION_AUXILIARY",
        "classification-thresholds": {{ "accept-threshold": 0.9, "notify-threshold": 0.3, "overrides": {} }},
        "word-component-array": [ {} ],
        "keyword-feature-window-length": 100, "lock-period": 100 }} }}"#,
        overrides([0.64, 0.84, 0.84]),
        component(1, "ww_vrf_scores", "alignments", None)
    );
    let restart = "\n  \"restart-trigger\": { \"max-event-count\": 30, \"evaluation-span-msec\": 180000 },";
    put(dir, "BDPGeneratedFiles/kw.cfg.json", kw_cfg(&[stop_spotter, disney, alexa, verifier], restart));
    put(dir, "BDPGeneratedFiles/op.cfg.json", OP_DAVS);
    put(dir, "BDPGeneratedFiles/words.shrunk.txt", "<eps> 0\nALEXA 1\nHEY_DISNEY 2\nSTOP 3\n");
}
