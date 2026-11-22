//! Streaming wake pipeline: LFBE front end, then every stage one step per encoder step, carrying LSTM lane rings
//! and recurrent (conv) buffers between steps.

use super::set::{load_set, SetSpec, StageInput};
use crate::lfbe::Lfbe;
use crate::onnx::{self, Graph, Session};
use crate::Error;
use std::path::Path;
use std::sync::Arc;

/// Stage outputs a set may expose; `push` hands them out from a stack array.
const MAX_COMPONENTS: usize = 32;
/// Samples per front-end push: keeps the LFBE buffer within its preallocated capacity.
const HOP: usize = 160;

/// One stage output, as the keyword rules address it.
#[derive(Debug, Clone, PartialEq)]
pub struct ComponentRef {
    /// `"<scorer>/<component>"`.
    pub key: String,
    pub stage: usize,
    pub output: usize,
    pub shape: Vec<usize>,
}

/// A set with every stage graph planned and wired.
pub struct LoadedSet {
    pub spec: SetSpec,
    pub graphs: Vec<Arc<Graph>>,
    pub components: Vec<ComponentRef>,
    wiring: Vec<Wiring>,
}

/// Graph indices one stage step reads and writes.
struct Wiring {
    /// The audio or artifact input.
    input: usize,
    /// Artifact inputs: (publishing stage, its output).
    source: Option<(usize, usize)>,
    /// (input, output, lanes).
    lstm: Vec<(usize, usize, usize)>,
    /// (input, output).
    recurrent: Vec<(usize, usize)>,
}

fn unsupported(file: &Path, what: impl Into<String>) -> Error {
    Error::Unsupported { file: file.to_path_buf(), what: what.into() }
}

impl LoadedSet {
    pub fn load(dir: &Path) -> Result<LoadedSet, Error> {
        let spec = load_set(dir)?;
        let fps = spec.stages.first().map_or(0, |s| s.frames_per_step);
        if fps == 0 {
            return Err(unsupported(dir, "a set without an audio stage of at least one frame per step"));
        }
        let mut graphs: Vec<Arc<Graph>> = Vec::new();
        let mut components = Vec::new();
        let mut wiring = Vec::new();
        for (s, st) in spec.stages.iter().enumerate() {
            let file = &st.graph;
            let iface = onnx::interface(file)?;
            let port = |name: &str| {
                iface.inputs.iter().find(|p| p.name == name).ok_or_else(|| {
                    Error::Format(format!("{}: input {name} is not declared", file.display()))
                })
            };
            let mut bind: Vec<(String, Vec<usize>)> = Vec::new();
            let mut keep: Vec<&str> = Vec::new();
            let (iname, source) = match (s, &st.inputs[..]) {
                (0, [StageInput::Audio(n)]) => {
                    let rank = port(n)?.dims.len();
                    let dims = if rank == 4 { vec![1, fps, spec.fe.bins, 1] } else { vec![1, fps, spec.fe.bins] };
                    bind.push((n.clone(), dims));
                    (n, None)
                }
                (1.., [StageInput::Artifact { input, artifact }]) => {
                    let found = spec.stages[..s].iter().enumerate().find_map(|(p, ps)| {
                        ps.outputs.iter().find(|(_, publ)| publ == artifact).map(|(c, _)| (p, c))
                    });
                    let Some((p, comp)) = found else {
                        return Err(unsupported(file, format!("input {input} reads artifact {artifact}, unpublished")));
                    };
                    let o = graphs[p].output_index(comp).ok_or_else(|| {
                        Error::Format(format!("{}: output {comp} missing", spec.stages[p].graph.display()))
                    })?;
                    bind.push((input.clone(), graphs[p].output_shape(o).to_vec()));
                    (input, Some((p, o)))
                }
                _ => {
                    let what = format!(
                        "stage {} inputs {:?} (implemented: one audio input to the first stage, one artifact input \
                         to each later stage)",
                        st.name, st.inputs
                    );
                    return Err(unsupported(file, what));
                }
            };
            for (i, _) in st.lstm.iter().chain(&st.recurrent) {
                let p = port(i)?;
                if p.init.is_some() {
                    keep.push(i);
                    continue;
                }
                let dims: Option<Vec<usize>> = p.dims.iter().copied().collect();
                match dims {
                    Some(d) if !d.is_empty() => bind.push((i.clone(), d)),
                    _ => return Err(unsupported(file, format!("state input {i} without a concrete declared shape"))),
                }
            }
            let b: Vec<(&str, &[usize])> = bind.iter().map(|(n, d)| (n.as_str(), &d[..])).collect();
            let g = Graph::load(file, &b, &keep)?;
            let idx_in = |n: &str| {
                g.input_index(n).ok_or_else(|| Error::Format(format!("{}: input {n} missing", file.display())))
            };
            let idx_out = |n: &str| {
                g.output_index(n).ok_or_else(|| Error::Format(format!("{}: output {n} missing", file.display())))
            };
            let pair = |i: &str, o: &str| -> Result<(usize, usize), Error> {
                let (ii, oi) = (idx_in(i)?, idx_out(o)?);
                if g.input_shape(ii) != g.output_shape(oi) {
                    let what = format!(
                        "state {i} {:?} fed from {o} {:?} (implemented: equal shapes)",
                        g.input_shape(ii),
                        g.output_shape(oi)
                    );
                    return Err(unsupported(file, what));
                }
                Ok((ii, oi))
            };
            let mut lstm = Vec::new();
            for (i, o) in &st.lstm {
                let (ii, oi) = pair(i, o)?;
                let &[lanes, hid] = g.input_shape(ii) else {
                    let what = format!("LSTM state {i} {:?} (implemented: [lanes, hidden])", g.input_shape(ii));
                    return Err(unsupported(file, what));
                };
                if lanes == 0 || hid == 0 {
                    return Err(unsupported(file, format!("LSTM state {i} [{lanes}, {hid}]")));
                }
                lstm.push((ii, oi, lanes));
            }
            let recurrent = st.recurrent.iter().map(|(i, o)| pair(i, o)).collect::<Result<Vec<_>, _>>()?;
            for (comp, _) in &st.outputs {
                let o = idx_out(comp)?;
                let shape = g.output_shape(o).to_vec();
                components.push(ComponentRef { key: format!("{}/{comp}", st.name), stage: s, output: o, shape });
            }
            wiring.push(Wiring { input: idx_in(iname)?, source, lstm, recurrent });
            graphs.push(Arc::new(g));
        }
        if components.len() > MAX_COMPONENTS {
            let what = format!("{} stage outputs (implemented: up to {MAX_COMPONENTS})", components.len());
            return Err(unsupported(dir, what));
        }
        Ok(LoadedSet { spec, graphs, components, wiring })
    }

    pub fn component(&self, key: &str) -> Option<usize> {
        self.components.iter().position(|c| c.key == key)
    }
}

pub struct StepOut<'a> {
    /// Decoder step index since reset.
    pub step: u64,
    /// LFBE frames consumed since reset, this step's included.
    pub frames: u64,
    /// Indexed like [`LoadedSet::components`].
    pub values: &'a [&'a [f32]],
}

/// One stream through a [`LoadedSet`]; `push` allocates nothing.
pub struct Pipeline {
    set: Arc<LoadedSet>,
    fe: Lfbe,
    /// Front-end output of one `HOP` push.
    frame: Vec<f32>,
    sessions: Vec<Session>,
    /// Frames already written into the first stage's input.
    pending: usize,
    step: u64,
    frames: u64,
}

impl Pipeline {
    pub fn new(set: &Arc<LoadedSet>) -> Pipeline {
        let sessions = set.graphs.iter().map(|g| Session::new(g.clone())).collect();
        let mut p = Pipeline {
            set: set.clone(),
            fe: Lfbe::with_spec(set.spec.fe),
            frame: Vec::with_capacity(set.spec.fe.bins * 2),
            sessions,
            pending: 0,
            step: 0,
            frames: 0,
        };
        p.reset();
        p
    }

    /// Restarts the front end, rings and recurrent buffers (stored values where the graph has them).
    pub fn reset(&mut self) {
        for ((w, sess), g) in self.set.wiring.iter().zip(&mut self.sessions).zip(&self.set.graphs) {
            for &(i, _, _) in &w.lstm {
                sess.input_mut(i).fill(0.0);
            }
            for &(i, _) in &w.recurrent {
                match g.initial(i) {
                    Some(v) => sess.input_mut(i).copy_from_slice(v),
                    None => sess.input_mut(i).fill(0.0),
                }
            }
        }
        self.fe.reset();
        self.pending = 0;
        self.step = 0;
        self.frames = 0;
    }

    /// Feeds 16 kHz mono PCM; calls `f` once per completed step.
    pub fn push(&mut self, pcm: &[i16], f: &mut dyn FnMut(&StepOut)) {
        self.push_frames(pcm, &mut |_, s| {
            if let Some(s) = s {
                f(s)
            }
        });
    }

    /// Like [`Pipeline::push`], but calls `f` for every front-end frame: (frames since reset, the step it
    /// completes).
    pub fn push_frames(&mut self, pcm: &[i16], f: &mut dyn FnMut(u64, Option<&StepOut>)) {
        let bins = self.set.spec.fe.bins;
        let (fps, input) = (self.set.spec.stages[0].frames_per_step, self.set.wiring[0].input);
        for chunk in pcm.chunks(HOP) {
            self.frame.clear();
            self.fe.push_flat(chunk, &mut self.frame);
            for k in 0..self.frame.len() / bins {
                let dst = &mut self.sessions[0].input_mut(input)[self.pending * bins..][..bins];
                dst.copy_from_slice(&self.frame[k * bins..][..bins]);
                self.pending += 1;
                self.frames += 1;
                if self.pending == fps {
                    self.pending = 0;
                    self.run_step(f);
                } else {
                    f(self.frames, None);
                }
            }
        }
    }

    fn run_step(&mut self, f: &mut dyn FnMut(u64, Option<&StepOut>)) {
        self.ring();
        for (s, w) in self.set.wiring.iter().enumerate() {
            if let Some((p, o)) = w.source {
                let (src, dst) = self.sessions.split_at_mut(s);
                dst[0].input_mut(w.input).copy_from_slice(src[p].output(o));
            }
            let sess = &mut self.sessions[s];
            sess.run();
            for &(i, o, _) in &w.lstm {
                sess.feed_back(o, i);
            }
            for &(i, o) in &w.recurrent {
                sess.feed_back(o, i);
            }
        }
        let mut values: [&[f32]; MAX_COMPONENTS] = [&[]; MAX_COMPONENTS];
        let comps = &self.set.components;
        for (v, c) in values.iter_mut().zip(comps) {
            *v = self.sessions[c.stage].output(c.output);
        }
        f(self.frames, Some(&StepOut { step: self.step, frames: self.frames, values: &values[..comps.len()] }));
        self.step += 1;
    }

    /// Zeroes this step's ring lane (all lanes at step 0) of every LSTM state.
    fn ring(&mut self) {
        let t = self.step;
        for (w, sess) in self.set.wiring.iter().zip(&mut self.sessions) {
            for &(i, _, lanes) in &w.lstm {
                let st = sess.input_mut(i);
                if t == 0 {
                    st.fill(0.0);
                } else {
                    let n = st.len() / lanes;
                    let l = (t % lanes as u64) as usize;
                    st[l * n..(l + 1) * n].fill(0.0);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pryon_testkit::wakeset::{write_davs_set, write_fw_set, WakeSetOpts, DAVS_LANES};
    use pryon_testkit::{synth, Clip, TempDir};

    fn set(fw: bool, tag: &str) -> (TempDir, Arc<LoadedSet>) {
        let d = TempDir::new(&format!("pipeline-{tag}"));
        let o = WakeSetOpts { seed: 5, force_score: None };
        if fw { write_fw_set(&d, &o) } else { write_davs_set(&d, &o) }
        let s = Arc::new(LoadedSet::load(&d).unwrap());
        (d, s)
    }

    /// Samples giving exactly `n` LFBE frames.
    fn samples(n: usize) -> usize {
        400 + (n - 1) * 160
    }

    fn run(p: &mut Pipeline, pcm: &[i16], chunk: usize) -> Vec<(u64, u64, Vec<Vec<f32>>)> {
        let mut out = Vec::new();
        for c in pcm.chunks(chunk) {
            p.push(c, &mut |s: &StepOut| {
                out.push((s.step, s.frames, s.values.iter().map(|v| v.to_vec()).collect()));
            });
        }
        out
    }

    fn lstm_inputs(set: &LoadedSet) -> Vec<(usize, usize, usize)> {
        set.wiring.iter().enumerate().flat_map(|(s, w)| w.lstm.iter().map(move |&(i, _, l)| (s, i, l))).collect()
    }

    #[test]
    fn lane_ring_zeroing() {
        let (_d, set) = set(false, "ring");
        let mut p = Pipeline::new(&set);
        let states = lstm_inputs(&set);
        assert_eq!(states.len(), 4);
        for &(s, i, lanes) in &states {
            assert_eq!(lanes, DAVS_LANES);
            p.sessions[s].input_mut(i).fill(7.0);
        }
        p.ring();
        for &(s, i, _) in &states {
            assert!(p.sessions[s].input_mut(i).iter().all(|&x| x == 0.0), "step 0: all lanes zero");
        }
        let pcm = synth(Clip::Noise);
        let mut steps = 0;
        p.push(&pcm[..samples(17 * 8)], &mut |_| steps += 1);
        assert_eq!(steps, 17);
        p.ring();
        for &(s, i, lanes) in &states {
            let st = p.sessions[s].input_mut(i);
            let w = st.len() / lanes;
            for (l, lane) in st.chunks(w).enumerate() {
                if l == 17 % lanes {
                    assert!(lane.iter().all(|&x| x == 0.0), "lane {l} zeroed before step 17");
                } else {
                    assert!(lane.iter().any(|&x| x != 0.0), "lane {l} kept before step 17");
                }
            }
        }
    }

    #[test]
    fn primed_state_restored_by_reset() {
        let (_d, set) = set(false, "primed");
        let mut p = Pipeline::new(&set);
        let primed: Vec<(usize, usize)> =
            set.wiring.iter().enumerate().flat_map(|(s, w)| w.recurrent.iter().map(move |&(i, _)| (s, i))).collect();
        assert_eq!(primed.len(), 2);
        let initial = |p: &mut Pipeline| -> Vec<Vec<f32>> {
            primed.iter().map(|&(s, i)| p.sessions[s].input_mut(i).to_vec()).collect()
        };
        let start = initial(&mut p);
        for (k, &(s, i)) in primed.iter().enumerate() {
            assert_eq!(start[k], set.graphs[s].initial(i).unwrap(), "starts from the stored value");
            assert!(start[k].iter().any(|&x| x != 0.0));
        }
        let pcm = synth(Clip::Tones);
        let first = run(&mut p, &pcm[..samples(200)], 1280);
        assert_ne!(initial(&mut p), start, "buffers moved");
        p.reset();
        assert_eq!(initial(&mut p), start, "reset restores the stored values");
        let again = run(&mut p, &pcm[..samples(200)], 1280);
        assert_eq!(first, again);
        let fresh = run(&mut Pipeline::new(&set), &pcm[..samples(200)], 1280);
        assert_eq!(first, fresh);
    }

    #[test]
    fn pipeline_chunk_invariant() {
        for fw in [true, false] {
            let (_d, set) = set(fw, "chunks");
            let pcm = &synth(Clip::Noise)[..samples(300) + 77];
            let whole = run(&mut Pipeline::new(&set), pcm, pcm.len());
            assert!(!whole.is_empty());
            for chunk in [1, 7, 1280] {
                assert_eq!(run(&mut Pipeline::new(&set), pcm, chunk), whole, "fw {fw} chunk {chunk}");
            }
        }
    }

    #[cfg(not(target_os = "android"))]
    #[test]
    fn push_allocates_nothing() {
        use crate::sound::allocs;
        for fw in [true, false] {
            let (_d, set) = set(fw, "alloc");
            let mut p = Pipeline::new(&set);
            let pcm = synth(Clip::Noise);
            let mut steps = 0;
            let before = allocs();
            for c in pcm[..samples(400)].chunks(1280).chain([&pcm[samples(400)..samples(800)]]) {
                p.push(c, &mut |_| steps += 1);
            }
            assert_eq!(allocs(), before, "fw {fw}: push allocated");
            assert!(steps > 90);
        }
    }

    #[test]
    fn push_frames_ticks_every_frame() {
        let (_d, set) = set(true, "ticks");
        let pcm = &synth(Clip::Noise)[..samples(20)];
        let mut ticks = Vec::new();
        let mut p = Pipeline::new(&set);
        for c in pcm.chunks(333) {
            p.push_frames(c, &mut |f, s| ticks.push((f, s.map(|s| (s.step, s.frames)))));
        }
        let want: Vec<(u64, Option<(u64, u64)>)> =
            (1..=20).map(|f| (f, (f % 6 == 0).then(|| (f / 6 - 1, f)))).collect();
        assert_eq!(ticks, want);
    }

    #[test]
    fn steps_per_frames() {
        for (fw, fps, want) in [(true, 6u64, 100usize), (false, 8, 75)] {
            let (_d, set) = set(fw, "steps");
            let pcm = &synth(Clip::Noise)[..samples(600)];
            let out = run(&mut Pipeline::new(&set), pcm, 333);
            assert_eq!(out.len(), want, "fw {fw}");
            for (k, (step, frames, values)) in out.iter().enumerate() {
                assert_eq!((*step, *frames), (k as u64, (k as u64 + 1) * fps));
                assert_eq!(values.len(), set.components.len());
            }
        }
    }
}
