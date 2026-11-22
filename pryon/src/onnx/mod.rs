//! A streaming ONNX executor for the wake word graphs: shapes are bound and every integer/shape computation is
//! folded at load, leaving a fixed list of f32 steps over one preallocated arena per [`Session`].

mod ops;
mod plan;
pub(crate) mod proto;
mod tensor;

use crate::Error;
use plan::{Fail, Plan, MAX_IO};
use std::fmt;
use std::path::Path;
use std::sync::Arc;

/// A loaded, planned graph; share it between sessions with an `Arc`.
pub struct Graph {
    plan: Plan,
}

impl Graph {
    /// `bind` gives concrete shapes for every runtime input; `keep` names initializer-backed inputs that stay
    /// feedable (streaming state), starting from their initializer value.
    pub fn load(path: &Path, bind: &[(&str, &[usize])], keep: &[&str]) -> Result<Graph, Error> {
        let bytes = std::fs::read(path).map_err(|source| Error::Io { path: path.to_path_buf(), source })?;
        let m = proto::parse_model(&bytes).map_err(|e| Error::Format(format!("{}: {e}", path.display())))?;
        let plan = plan::plan(&m, bind, keep, ops::OPS).map_err(|f| match f {
            Fail::Unsupported(what) => Error::Unsupported { file: path.to_path_buf(), what },
            Fail::Format(e) => Error::Format(format!("{}: {e}", path.display())),
        })?;
        Ok(Graph { plan })
    }

    pub fn input_index(&self, name: &str) -> Option<usize> {
        self.plan.inputs.iter().position(|io| io.name == name)
    }

    pub fn output_index(&self, name: &str) -> Option<usize> {
        self.plan.outputs.iter().position(|io| io.name == name)
    }

    pub fn input_shape(&self, i: usize) -> &[usize] {
        &self.plan.inputs[i].dims
    }

    pub fn output_shape(&self, i: usize) -> &[usize] {
        &self.plan.outputs[i].dims
    }

    /// The stored ("primed") value of a kept input; `None` for ordinary inputs.
    pub fn initial(&self, i: usize) -> Option<&[f32]> {
        self.plan.inputs[i].initial.as_deref()
    }
}

impl fmt::Debug for Graph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let names = |v: &[plan::Io]| v.iter().map(|io| io.name.clone()).collect::<Vec<_>>();
        f.debug_struct("Graph")
            .field("inputs", &names(&self.plan.inputs))
            .field("outputs", &names(&self.plan.outputs))
            .field("steps", &self.plan.steps.len())
            .finish()
    }
}

/// Per-stream buffers for one graph; `run` executes one step and allocates nothing.
pub struct Session {
    graph: Arc<Graph>,
    arena: Vec<f32>,
}

impl Session {
    /// Kept inputs start at their initializer value, everything else at zero.
    pub fn new(g: Arc<Graph>) -> Session {
        let arena = g.plan.arena.clone();
        Session { graph: g, arena }
    }

    pub fn input_mut(&mut self, i: usize) -> &mut [f32] {
        let io = &self.graph.plan.inputs[i];
        &mut self.arena[io.off..io.off + io.len]
    }

    pub fn run(&mut self) {
        self.exec(None);
    }

    pub fn output(&self, i: usize) -> &[f32] {
        let io = &self.graph.plan.outputs[i];
        &self.arena[io.off..io.off + io.len]
    }

    /// Copies output `o` into input `i` (streaming state carried to the next step).
    pub fn feed_back(&mut self, o: usize, i: usize) {
        let (src, dst) = (&self.graph.plan.outputs[o], &self.graph.plan.inputs[i]);
        assert_eq!(src.len, dst.len, "feed_back between different sizes");
        self.arena.copy_within(src.off..src.off + src.len, dst.off);
    }

    /// `run`, calling `f(name, shape, data)` for every runtime value right after it is computed.
    #[doc(hidden)]
    pub fn run_traced(&mut self, f: &mut dyn FnMut(&str, &[usize], &[f32])) {
        self.exec(Some(f));
    }

    fn exec(&mut self, mut trace: Option<&mut dyn FnMut(&str, &[usize], &[f32])>) {
        let base = self.arena.as_mut_ptr();
        for s in &self.graph.plan.steps {
            let mut ins: [&[f32]; MAX_IO] = [&[]; MAX_IO];
            let mut outs: [&mut [f32]; MAX_IO + 1] = Default::default();
            // SAFETY: the planner places every region inside the arena, and a step's output regions are its own,
            // never shared with another region it receives, so these slices do not alias mutably.
            unsafe {
                for (k, &(o, n)) in s.ins.iter().enumerate() {
                    ins[k] = std::slice::from_raw_parts(base.add(o), n);
                }
                for (k, &(o, n)) in s.outs.iter().enumerate() {
                    outs[k] = std::slice::from_raw_parts_mut(base.add(o), n);
                }
            }
            s.op.run(&ins[..s.ins.len()], &mut outs[..s.outs.len()]);
            if let Some(f) = trace.as_mut() {
                for ((name, dims), data) in s.names.iter().zip(&s.dims).zip(&outs) {
                    if !name.is_empty() {
                        f(name, dims, data);
                    }
                }
            }
        }
    }
}

/// A graph's declared interface, read without planning it (the set loader checks names against it).
pub(crate) struct Interface {
    pub inputs: Vec<Port>,
    pub outputs: Vec<Port>,
    /// Sorted `op_type` counts.
    pub ops: Vec<(String, usize)>,
}

pub(crate) struct Port {
    pub name: String,
    /// Declared dims; `None` for a symbolic or unknown one, empty when no shape is declared.
    pub dims: Vec<Option<usize>>,
    /// Inputs only: an initializer of that name exists, and whether it carries data.
    pub init: Option<bool>,
}

pub(crate) fn interface(path: &Path) -> Result<Interface, Error> {
    let bytes = std::fs::read(path).map_err(|source| Error::Io { path: path.to_path_buf(), source })?;
    let m = proto::parse_model(&bytes).map_err(|e| Error::Format(format!("{}: {e}", path.display())))?;
    let g = &m.graph;
    let dims = |v: &proto::ValueInfo| -> Vec<Option<usize>> {
        v.dims
            .iter()
            .map(|d| match d {
                proto::Dim::Value(x) => usize::try_from(*x).ok(),
                _ => None,
            })
            .collect()
    };
    let init = |name: &str| {
        g.initializers.iter().find(|t| t.name == name).map(|t| match &t.data {
            proto::TensorData::Empty => false,
            proto::TensorData::Raw(b) => !b.is_empty(),
            proto::TensorData::F32(v) => !v.is_empty(),
            proto::TensorData::I32(v) => !v.is_empty(),
            proto::TensorData::I64(v) => !v.is_empty(),
            proto::TensorData::F64(v) => !v.is_empty(),
        })
    };
    let inputs = g.inputs.iter().map(|v| Port { name: v.name.clone(), dims: dims(v), init: init(&v.name) }).collect();
    let outputs = g.outputs.iter().map(|v| Port { name: v.name.clone(), dims: dims(v), init: None }).collect();
    let mut ops: Vec<(String, usize)> = Vec::new();
    for n in &g.nodes {
        match ops.iter_mut().find(|(o, _)| *o == n.op_type) {
            Some(e) => e.1 += 1,
            None => ops.push((n.op_type.clone(), 1)),
        }
    }
    ops.sort();
    Ok(Interface { inputs, outputs, ops })
}

/// IR version, opsets and sorted `op_type` counts of a model; test support for real-graph checks.
#[doc(hidden)]
#[allow(clippy::type_complexity)]
pub fn op_census(bytes: &[u8]) -> Result<(i64, Vec<(String, i64)>, Vec<(String, usize)>), String> {
    let m = proto::parse_model(bytes)?;
    let mut ops: Vec<(String, usize)> = Vec::new();
    for n in &m.graph.nodes {
        match ops.iter_mut().find(|(o, _)| *o == n.op_type) {
            Some(e) => e.1 += 1,
            None => ops.push((n.op_type.clone(), 1)),
        }
    }
    ops.sort();
    Ok((m.ir_version, m.opsets, ops))
}
