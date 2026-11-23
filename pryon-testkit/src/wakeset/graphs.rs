//! The synthetic sets' graphs: small, seeded, built only from operators the real wake graphs use.

use super::{WakeSetOpts, DAVS_LANES, EMBED, FW_LANES};
use crate::onnx::{Attr, Dim, ModelBuilder};
use crate::Rng;

const F: i32 = 1;

struct R(Rng);

impl R {
    fn new(seed: u64, salt: u64) -> R {
        R(Rng((seed ^ salt.wrapping_mul(0x9E37_79B9_7F4A_7C15)) | 1))
    }
    /// Uniform in [-1, 1).
    fn unit(&mut self) -> f32 {
        (self.0.next_u32() >> 8) as f32 / (1 << 23) as f32 - 1.0
    }
    fn vec(&mut self, n: usize, amp: f32) -> Vec<f32> {
        (0..n).map(|_| amp * self.unit()).collect()
    }
    fn i8s(&mut self, n: usize) -> Vec<i8> {
        (0..n).map(|_| self.0.next_u32() as u8 as i8).collect()
    }
    fn ints(&mut self, n: usize, amp: i32) -> Vec<i32> {
        (0..n).map(|_| (self.0.next_u32() % (2 * amp as u32 + 1)) as i32 - amp).collect()
    }
}

fn dims(v: &[usize]) -> Vec<Dim> {
    v.iter().map(|&x| Dim::Value(x as i64)).collect()
}

/// Quantised weights `lead ++ [n, k]` named `p_w`, `p_sw`, `p_zw`, `p_sums`; `scale` bounds the per-row scales.
fn qweights(m: ModelBuilder, r: &mut R, p: &str, lead: &[i64], n: usize, k: usize, scale: f32) -> ModelBuilder {
    let w = r.i8s(n * k);
    let sums: Vec<i32> = w.chunks(k).map(|row| row.iter().map(|&x| i32::from(x)).sum()).collect();
    let sw: Vec<f32> = (0..n).map(|_| scale * (0.6 + 0.4 * r.unit())).collect();
    let zw = r.ints(n, 20);
    let nd: Vec<i64> = lead.iter().copied().chain([n as i64]).collect();
    let wd: Vec<i64> = nd.iter().copied().chain([k as i64]).collect();
    m.init_i8(&format!("{p}_w"), &wd, &w)
        .init_f32(&format!("{p}_sw"), &nd, &sw)
        .init_i32(&format!("{p}_zw"), &nd, &zw)
        .init_i32(&format!("{p}_sums"), &nd, &sums)
}

fn qnames(p: &str) -> [String; 4] {
    [format!("{p}_w"), format!("{p}_sw"), format!("{p}_zw"), format!("{p}_sums")]
}

/// A classifier head `x [rows, k] -> y [rows, n]` (HybridGemm with bias); under `force` its weights are negligible
/// and the class-1 bias puts the softmax class-1 score at `force`.
#[allow(clippy::too_many_arguments)]
fn head(m: ModelBuilder, r: &mut R, p: &str, x: &str, y: &str, n: usize, k: usize, force: Option<f32>) -> ModelBuilder {
    let (scale, bias) = match force {
        Some(s) => {
            assert!(s > 0.0 && s < 1.0, "force_score must be in (0, 1)");
            let mut b = vec![0.0; n];
            b[1] = (s * (n - 1) as f32 / (1.0 - s)).ln();
            (1e-9, b)
        }
        None => (0.02, r.vec(n, 0.5)),
    };
    let m = qweights(m, r, p, &[], n, k, scale).init_f32(&format!("{p}_b"), &[n as i64], &bias);
    let [w, sw, zw, sums] = qnames(p);
    m.node("HybridGemm", &[x, &w, &sw, &zw, &sums, &format!("{p}_b")], &[y], &[])
}

/// One HybridLSTM ring of `lanes` lanes over `x [1, rows, k]`; states `h_in/c_in [lanes, hid]` in, `h_out/c_out` out,
/// hidden sequence squeezed to `y [lanes, hid]`.
#[allow(clippy::too_many_arguments)]
fn lstm(
    m: ModelBuilder,
    r: &mut R,
    p: &str,
    x: &str,
    k: usize,
    hid: usize,
    states: [&str; 4],
    y: &str,
    opset11: bool,
) -> ModelBuilder {
    let [h_in, c_in, h_out, c_out] = states;
    let g = 4 * hid;
    let m = qweights(m, r, &format!("{p}_wi"), &[1], g, k, 0.02);
    let m = qweights(m, r, &format!("{p}_wr"), &[1], g, hid, 0.02);
    let m = m.init_f32(&format!("{p}_b"), &[1, 2 * g as i64], &r.vec(2 * g, 0.2));
    let (m, h0, c0) = (unsq(m, h_in, &format!("{p}_h0"), 0, opset11), format!("{p}_h0"), format!("{p}_c0"));
    let m = unsq(m, c_in, &c0, 0, opset11);
    let [wi, swi, zwi, sumsi] = qnames(&format!("{p}_wi"));
    let [wr, swr, zwr, sumsr] = qnames(&format!("{p}_wr"));
    let mut attrs = vec![("hidden_size", Attr::Int(hid as i64))];
    if opset11 {
        attrs.push(("activations", Attr::Strs(vec!["Sigmoid".into(), "Tanh".into(), "Tanh".into()])));
        attrs.push(("direction", Attr::Str("forward".into())));
    }
    let (ys, yh, yc) = (format!("{p}_ys"), format!("{p}_yh"), format!("{p}_yc"));
    let b = format!("{p}_b");
    let m = m.node(
        "HybridLSTM",
        &[x, &wi, &swi, &zwi, &sumsi, &wr, &swr, &zwr, &sumsr, &b, "", &h0, &c0],
        &[&ys, &yh, &yc],
        &attrs,
    );
    let m = sq(m, &yh, h_out, &[0], opset11);
    let m = sq(m, &yc, c_out, &[0], opset11);
    sq(m, &ys, y, &[0, 1], opset11)
}

type Axes = (ModelBuilder, Vec<(&'static str, Attr)>, String);

/// Axes as an attribute (opset 11) or as an int64 initializer input (opset 13+).
fn axes_arg(m: ModelBuilder, name: &str, axes: &[i64], opset11: bool) -> Axes {
    if opset11 {
        (m, vec![("axes", Attr::Ints(axes.to_vec()))], String::new())
    } else {
        let n = format!("{name}_axes");
        (m.init_i64(&n, &[axes.len() as i64], axes), vec![], n)
    }
}

fn sq(m: ModelBuilder, x: &str, y: &str, axes: &[i64], opset11: bool) -> ModelBuilder {
    let (m, attrs, a) = axes_arg(m, y, axes, opset11);
    let ins: Vec<&str> = if a.is_empty() { vec![x] } else { vec![x, &a] };
    m.node("Squeeze", &ins, &[y], &attrs)
}

fn unsq(m: ModelBuilder, x: &str, y: &str, axis: i64, opset11: bool) -> ModelBuilder {
    let (m, attrs, a) = axes_arg(m, y, &[axis], opset11);
    let ins: Vec<&str> = if a.is_empty() { vec![x] } else { vec![x, &a] };
    m.node("Unsqueeze", &ins, &[y], &attrs)
}

fn reshape(m: ModelBuilder, x: &str, y: &str, shape: &[i64]) -> ModelBuilder {
    let s = format!("{y}_shape");
    m.init_i64(&s, &[shape.len() as i64], shape).node("Reshape", &[x, &s], &[y], &[])
}

/// `y = x[start..end]` along `axis`.
fn slice(m: ModelBuilder, x: &str, y: &str, axis: i64, start: i64, end: i64) -> ModelBuilder {
    let p = y.replace(['/', ':'], "_");
    let (s, e, a) = (format!("{p}_starts"), format!("{p}_ends"), format!("{p}_axes"));
    m.init_i64(&s, &[1], &[start]).init_i64(&e, &[1], &[end]).init_i64(&a, &[1], &[axis]).node(
        "Slice",
        &[x, &s, &e, &a],
        &[y],
        &[],
    )
}

#[allow(clippy::too_many_arguments)]
fn conv(m: ModelBuilder, r: &mut R, p: &str, x: &str, y: &str, n: usize, c: usize, k: [usize; 2]) -> ModelBuilder {
    let m = qweights(m, r, p, &[], n, c * k[0] * k[1], 0.01).init_f32(&format!("{p}_b"), &[n as i64], &r.vec(n, 0.2));
    let [w, sw, zw, sums] = qnames(p);
    m.node(
        "HybridConv",
        &[x, &w, &sw, &zw, &sums, &format!("{p}_b")],
        &[y],
        &[
            ("data_format", Attr::Str("NCHW".into())),
            ("dilations", Attr::Ints(vec![1, 1])),
            ("strides", Attr::Ints(vec![1, 1])),
            ("kernel_shape", Attr::Ints(vec![k[0] as i64, k[1] as i64])),
            ("group", Attr::Int(1)),
        ],
    )
}

fn softmax(m: ModelBuilder, x: &str, y: &str) -> ModelBuilder {
    m.node("Softmax", &[x], &[y], &[("axis", Attr::Int(-1))])
}

/// Firmware-style graph: `input_feats [1, 6, 64, 1]`, an unprimed conv buffer, two 25-lane LSTM rings.
pub fn fw_graph(o: &WakeSetOpts) -> Vec<u8> {
    const HID: usize = 8;
    const CH: usize = 12;
    let mut r = R::new(o.seed, 1);
    let lanes = [FW_LANES, HID];
    let mut m = ModelBuilder::new(6, 11)
        .input("input_feats", F, &[Dim::Value(1), Dim::Param("unk__232".into()), Dim::Value(64), Dim::Value(1)])
        .input(
            "streaming_conv_buffer_input_7",
            F,
            &[Dim::Param("streaming_conv_buffer_input_7_0".into()), Dim::Value(1), Dim::Value(6), Dim::Value(64)],
        );
    for s in ["input_h_ww", "input_c_ww", "input_h_se", "input_c_se"] {
        m = m.input(s, F, &dims(&lanes));
    }
    m = m
        .output("softmax_output_ww", F, &[Dim::Value(25), Dim::Param("unk__78".into()), Dim::Value(3)])
        .output("fc_output_se", F, &[Dim::Value(25), Dim::Param("unk__86".into()), Dim::Value(2)])
        .output("streaming_conv_buffer_output_7", F, &dims(&[1, 1, 6, 64]));
    for s in ["lstm_ww_1", "lstm_ww_1_1", "lstm_se_1", "lstm_se_1_1"] {
        m = m.output(s, F, &dims(&lanes));
    }
    // placeholder initializer without data, like the firmware graph: the buffer starts at zero
    m = m.init_f32("streaming_conv_buffer_input_7", &[1, 1, 6, 64], &[]);
    m = reshape(m, "input_feats", "x4", &[1, 1, -1, 64]);
    m = m.node("Concat", &["streaming_conv_buffer_input_7", "x4"], &["cat"], &[("axis", Attr::Int(2))]);
    m = slice(m, "cat", "streaming_conv_buffer_output_7", 2, 6, 12);
    m = conv(m, &mut r, "conv", "cat", "conv_y", CH, 1, [12, 64]);
    m = m.node("Relu", &["conv_y"], &["conv_r"], &[]);
    m = reshape(m, "conv_r", "feat", &[1, 1, CH as i64]);
    m = lstm(m, &mut r, "ww", "feat", CH, HID, ["input_h_ww", "input_c_ww", "lstm_ww_1", "lstm_ww_1_1"], "ww_y", true);
    m = lstm(m, &mut r, "se", "feat", CH, HID, ["input_h_se", "input_c_se", "lstm_se_1", "lstm_se_1_1"], "se_y", true);
    m = head(m, &mut r, "ww_out", "ww_y", "ww_logits", 3, HID, o.force_score);
    m = softmax(m, "ww_logits", "ww_p");
    m = unsq(m, "ww_p", "softmax_output_ww", 1, true);
    m = head(m, &mut r, "se_out", "se_y", "se_logits", 2, HID, None);
    m = unsq(m, "se_logits", "fc_output_se", 1, true);
    m.build()
}

/// DAVS-style encoder: `input_136 [1, 8, 64]` → `encoder_embeddings [1, 1, EMBED]`, one primed conv buffer.
pub fn encoder_graph(o: &WakeSetOpts) -> Vec<u8> {
    let mut r = R::new(o.seed, 2);
    let buf = r.vec(2 * 64, 1.0);
    let mut m = ModelBuilder::new(8, 15)
        .input("input_136", F, &dims(&[1, 8, 64]))
        .input("conv_buffer_input/Concat:2", F, &dims(&[2, 64]))
        .output("encoder_embeddings", F, &dims(&[1, 1, EMBED]))
        .output("conv_buffer_output/Slice:2", F, &dims(&[2, 64]))
        .init_f32("conv_buffer_input/Concat:2", &[2, 64], &buf);
    m = sq(m, "input_136", "x2", &[0], false);
    m = m.node("Concat", &["conv_buffer_input/Concat:2", "x2"], &["cat"], &[("axis", Attr::Int(0))]);
    m = slice(m, "cat", "conv_buffer_output/Slice:2", 0, 8, 10);
    m = reshape(m, "cat", "flat", &[1, 640]);
    m = qweights(m, &mut r, "enc", &[], EMBED, 640, 0.002).init_f32("enc_b", &[EMBED as i64], &r.vec(EMBED, 0.3));
    let [w, sw, zw, sums] = qnames("enc");
    m = m.node("HybridGemm", &["flat", &w, &sw, &zw, &sums, "enc_b"], &["emb"], &[]);
    m = m.node("Relu", &["emb"], &["emb_r"], &[]);
    m = reshape(m, "emb_r", "encoder_embeddings", &[1, 1, EMBED as i64]);
    m.build()
}

/// DAVS-style LSTM decoder: `embedding [1, 1, EMBED]` tiled over two 16-lane rings; detection, verifier and
/// alignment heads `[1, 16, classes]`.
pub fn lstm_decoder_graph(o: &WakeSetOpts) -> Vec<u8> {
    const HS: usize = 8;
    const HA: usize = 4;
    let mut r = R::new(o.seed, 3);
    let l = DAVS_LANES;
    let mut m = ModelBuilder::new(8, 15).input("embedding", F, &dims(&[1, 1, EMBED]));
    for (s, h) in [("scores_h0", HS), ("scores_c0", HS), ("alignments_h0", HA), ("alignments_c0", HA)] {
        m = m.input(s, F, &dims(&[l, h]));
    }
    m = m
        .output("ww_det_scores", F, &dims(&[1, l, 3]))
        .output("ww_vrf_scores", F, &dims(&[1, l, 3]))
        .output("scores_hn", F, &dims(&[l, HS]))
        .output("scores_cn", F, &dims(&[l, HS]))
        .output("alignments", F, &dims(&[1, l, 2]))
        .output("alignments_hn", F, &dims(&[l, HA]))
        .output("alignments_cn", F, &dims(&[l, HA]));
    m = m.init_i64("tile_reps", &[3], &[1, l as i64, 1]).node("Tile", &["embedding", "tile_reps"], &["xt"], &[]);
    let st = ["scores_h0", "scores_c0", "scores_hn", "scores_cn"];
    m = lstm(m, &mut r, "sc", "xt", EMBED, HS, st, "sc_y", false);
    let al = ["alignments_h0", "alignments_c0", "alignments_hn", "alignments_cn"];
    m = lstm(m, &mut r, "al", "xt", EMBED, HA, al, "al_y", false);
    for (p, out) in [("det", "ww_det_scores"), ("vrf", "ww_vrf_scores")] {
        let (lg, pr) = (format!("{p}_logits"), format!("{p}_p"));
        m = head(m, &mut r, p, "sc_y", &lg, 3, HS, o.force_score);
        m = softmax(m, &lg, &pr);
        m = unsq(m, &pr, out, 0, false);
    }
    m = head(m, &mut r, "ali", "al_y", "ali_logits", 2, HA, None);
    m = unsq(m, "ali_logits", "alignments", 0, false);
    m.build()
}

/// DAVS-style conv decoder: `input_6 [1, 1, EMBED]`, one primed conv buffer; `mww/posteriors [1, 1, 4]` (STOP, HEY,
/// DISNEY) and `mww/start_end [1, 1, 2]`.
pub fn conv_decoder_graph(o: &WakeSetOpts) -> Vec<u8> {
    const CH: usize = 12;
    let mut r = R::new(o.seed, 4);
    let e = EMBED as i64;
    let buf = r.vec(EMBED * 3, 1.0);
    let mut m = ModelBuilder::new(10, 21)
        .input("input_6", F, &dims(&[1, 1, EMBED]))
        .input("conv_buffer_input/Concat:3", F, &dims(&[1, EMBED, 3, 1]))
        .output("mww/posteriors", F, &dims(&[1, 1, 4]))
        .output("mww/start_end", F, &dims(&[1, 1, 2]))
        .output("conv_buffer_output/Slice:3", F, &dims(&[1, EMBED, 3, 1]))
        .init_f32("conv_buffer_input/Concat:3", &[1, e, 3, 1], &buf);
    m = reshape(m, "input_6", "x4", &[1, e, 1, 1]);
    m = m.node("Concat", &["conv_buffer_input/Concat:3", "x4"], &["cat"], &[("axis", Attr::Int(2))]);
    m = slice(m, "cat", "conv_buffer_output/Slice:3", 2, 1, 4);
    m = conv(m, &mut r, "conv", "cat", "conv_y", CH, EMBED, [4, 1]);
    m = m.node("Relu", &["conv_y"], &["conv_r"], &[]);
    m = reshape(m, "conv_r", "feat", &[1, CH as i64]);
    m = head(m, &mut r, "post", "feat", "post_logits", 4, CH, o.force_score);
    m = softmax(m, "post_logits", "post_p");
    m = reshape(m, "post_p", "mww/posteriors", &[1, 1, 4]);
    m = head(m, &mut r, "se", "feat", "se_logits", 2, CH, None);
    m = reshape(m, "se_logits", "mww/start_end", &[1, 1, 2]);
    m.build()
}
