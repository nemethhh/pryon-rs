pub(crate) mod proto;

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
