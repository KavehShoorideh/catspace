//! PLAN 78.21 THE GRAPH NET'S FORWARD PASS IN RUST: the same function as the torch `GraphNet` of the training kit (graph_prior.py), from an exported weights file.
//!
//!   embeddings   term -> term table; fact -> rel + kind + present tables; move -> rel table; claim -> rel table + prob linear
//!   layer        HeteroConv(aggr="sum") of one SAGEConv(dim, dim) per edge type (PyG default: out = lin_l(mean of the source neighbours) + lin_r(x_dst), lin_l carries the
//!                bias, an empty neighbourhood's mean is 0), summed over the edge types into each destination type; then x = x + relu(LayerNorm(out)) per node type
//!   head         Linear(dim, dim), ReLU, Linear(dim, 1) on the move nodes: one score per move
//!   speed        every edge type is present in every graph (empty ones included), so each destination row starts at sum(lin_l bias) + (sum of the lin_r weights) x_dst,
//!                precomputed per destination type; lin_l is applied only to the rows that have a neighbour (an empty mean contributes the bias alone, already counted).
//!                The matrix products are plain loops (row-major, vectorised along the output row: no dependency).
//!   weights file JSON (the house convention of analyst_core's threat Policy): {format, dim, layers, max_arity, max_parts, vocab {term, rel} (UNK = id 0 implicit),
//!                emb {term, rel, kind, present, prob_w, prob_b} (flat row-major), conv: [ {"<src>__<rel>__<dst>": {lin_l_w, lin_l_b, lin_r_w}} per layer ],
//!                norm: [ {"<type>": {w, b}} per layer ], head {w0, b0, w1, head_bias}}; a torch Linear weight is [out, in] row-major.
use crate::graph::{edge_names, Graph, GraphSpec, MAX_ARITY};
use serde_json::Value;

const TYPES: [&str; 4] = ["term", "fact", "move", "claim"];
const EPS: f32 = 1e-5;
pub const KINDS: usize = 3;

fn ty(name: &str) -> Result<usize, String> { TYPES.iter().position(|t| *t == name).ok_or_else(|| format!("unknown node type {name}")) }

struct Conv { src: usize, dst: usize, edge: String, rev: bool, w_l: Vec<f32> }

struct Layer { convs: Vec<Conv>, r_sum: Vec<Vec<f32>>, b_sum: Vec<Vec<f32>>, norm_w: Vec<Vec<f32>>, norm_b: Vec<Vec<f32>> }

pub struct GraphNet {
    pub dim: usize,
    pub max_arity: usize,
    pub max_parts: usize,
    pub term_vocab: Vec<String>,
    pub rel_vocab: Vec<String>,
    term: Vec<f32>,
    rel: Vec<f32>,
    kind: Vec<f32>,
    present: Vec<f32>,
    layers: Vec<Layer>,
    w0: Vec<f32>,
    b0: Vec<f32>,
    w1: Vec<f32>,
    head_bias: f32,
}

fn floats(v: &Value, what: &str, len: usize) -> Result<Vec<f32>, String> {
    let a = v.as_array().ok_or_else(|| format!("{what}: a list of numbers"))?;
    if a.len() != len { return Err(format!("{what}: {} numbers, expected {len}", a.len())); }
    a.iter().map(|x| x.as_f64().map(|f| f as f32).ok_or_else(|| format!("{what}: not a number"))).collect()
}

fn get<'a>(v: &'a Value, k: &str) -> Result<&'a Value, String> { v.get(k).ok_or_else(|| format!("the weights file has no `{k}`")) }

/// W [out][in] row-major -> its transpose [in][out], so that `x . Wt` is a plain row-major product.
fn transposed(w: &[f32], n: usize) -> Vec<f32> {
    let mut t = vec![0.0; n * n];
    for o in 0..n { for i in 0..n { t[i * n + o] = w[o * n + i]; } }
    t
}

/// c (m x n) = beta * c + a (m x k) . b (k x n), all row-major: the inner loop runs along a row of `b`, which the compiler vectorises.
fn gemm(c: &mut [f32], a: &[f32], m: usize, k: usize, b: &[f32], n: usize, beta: f32) {
    debug_assert!(a.len() == m * k && b.len() == k * n && c.len() == m * n);
    for i in 0..m {
        let row = &mut c[i * n..(i + 1) * n];
        if beta != 1.0 { for v in row.iter_mut() { *v *= beta; } }
        for p in 0..k {
            let av = a[i * k + p];
            for (v, w) in row.iter_mut().zip(&b[p * n..(p + 1) * n]) { *v += av * w; }
        }
    }
}

impl GraphNet {
    pub fn load(path: &str) -> Result<GraphNet, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
        GraphNet::from_json(&text)
    }

    pub fn from_json(text: &str) -> Result<GraphNet, String> {
        let v: Value = serde_json::from_str(text).map_err(|e| format!("weights file: {e}"))?;
        if get(&v, "format")?.as_str() != Some("graph_net") { return Err("the weights file is not a graph_net".into()); }
        let num = |k: &str| get(&v, k)?.as_u64().map(|x| x as usize).ok_or_else(|| format!("`{k}` is a number"));
        let (dim, nlayers, max_arity, max_parts) = (num("dim")?, num("layers")?, num("max_arity")?, num("max_parts")?);
        if max_arity != MAX_ARITY { return Err(format!("the weights file reads {max_arity} argument positions, this build {MAX_ARITY}")); }
        let words = |k: &str| -> Result<Vec<String>, String> {
            get(get(&v, "vocab")?, k)?.as_array().ok_or("vocab is lists")?.iter().map(|x| x.as_str().map(String::from).ok_or("vocab holds text".to_string())).collect()
        };
        let (term_vocab, rel_vocab) = (words("term")?, words("rel")?);
        let emb = get(&v, "emb")?;
        let table = |k: &str, rows: usize| floats(get(emb, k)?, &format!("emb.{k}"), rows * dim);
        let mut layers = vec![];
        let (convs, norms) = (get(&v, "conv")?.as_array().ok_or("conv is a list")?, get(&v, "norm")?.as_array().ok_or("norm is a list")?);
        if convs.len() != nlayers || norms.len() != nlayers { return Err("conv / norm: one entry per layer".into()); }
        for (lc, ln) in convs.iter().zip(norms) {
            let mut cv = vec![];
            let mut r_sum = vec![vec![0.0f32; dim * dim]; 4];
            let mut b_sum = vec![vec![0.0f32; dim]; 4];
            let mut have = [false; 4];
            for (key, w) in lc.as_object().ok_or("a layer's conv is a map")? {
                let parts: Vec<&str> = key.split("__").collect();
                if parts.len() != 3 { return Err(format!("edge type key {key}")); }
                let (src, dst) = (ty(parts[0])?, ty(parts[2])?);
                let (rev, edge) = match parts[1].strip_prefix("rev_") { Some(e) => (true, e.to_string()), None => (false, parts[1].to_string()) };
                let (wl, bl, wr) = (floats(get(w, "lin_l_w")?, key, dim * dim)?, floats(get(w, "lin_l_b")?, key, dim)?, floats(get(w, "lin_r_w")?, key, dim * dim)?);
                for (s, x) in r_sum[dst].iter_mut().zip(&wr) { *s += x; }
                for (s, x) in b_sum[dst].iter_mut().zip(&bl) { *s += x; }
                have[dst] = true;
                cv.push(Conv { src, dst, edge, rev, w_l: transposed(&wl, dim) });
            }
            let r_t: Vec<Vec<f32>> = r_sum.iter().map(|w| transposed(w, dim)).collect();
            let (mut nw, mut nb) = (vec![vec![1.0; dim]; 4], vec![vec![0.0; dim]; 4]);
            for (t, name) in TYPES.iter().enumerate() {
                if let Some(n) = ln.get(*name) { nw[t] = floats(get(n, "w")?, name, dim)?; nb[t] = floats(get(n, "b")?, name, dim)?; }
                else if have[t] { return Err(format!("norm: no LayerNorm for node type {name}")); }
            }
            // a destination type no edge type reaches keeps x: mark by an empty b_sum
            for t in 0..4 { if !have[t] { b_sum[t].clear(); } }
            layers.push(Layer { convs: cv, r_sum: r_t, b_sum, norm_w: nw, norm_b: nb });
        }
        let head = get(&v, "head")?;
        let head_bias = floats(get(head, "head_bias")?, "head.head_bias", 1)?[0];
        Ok(GraphNet {
            dim, max_arity, max_parts, term: table("term", term_vocab.len() + 1)?, rel: table("rel", rel_vocab.len() + 1)?, kind: table("kind", KINDS)?, present: table("present", 2)?,
            term_vocab, rel_vocab, layers, w0: transposed(&floats(get(head, "w0")?, "head.w0", dim * dim)?, dim), b0: floats(get(head, "b0")?, "head.b0", dim)?,
            w1: floats(get(head, "w1")?, "head.w1", dim)?, head_bias,
        })
    }

    /// An error unless the weights were trained over the very vocabularies (and part count) the spec derives from its description.
    pub fn validate(&self, spec: &GraphSpec) -> Result<(), String> {
        if self.term_vocab != spec.term_vocab() { return Err(format!("the weights' term vocabulary ({} tokens) differs from the description's ({} tokens): trained for another description or graph vocabulary", self.term_vocab.len(), spec.term_vocab().len())); }
        if self.rel_vocab != spec.rel_vocab() { return Err("the weights' relation vocabulary differs from the description's".into()); }
        if self.max_parts != spec.max_parts() { return Err(format!("the weights have {} part edge types, the description {}", self.max_parts, spec.max_parts())); }
        Ok(())
    }

    /// One score per move node of `g`, in the graph's move order.
    pub fn forward(&self, g: &Graph) -> Vec<f32> {
        let d = self.dim;
        let n = [g.terms.len(), g.facts.len(), g.moves.len(), 0usize];
        let mut x: Vec<Vec<f32>> = n.iter().map(|&k| vec![0.0; k * d]).collect();
        for (i, &t) in g.term_ids.iter().enumerate() { x[0][i * d..(i + 1) * d].copy_from_slice(&self.term[t as usize * d..(t as usize + 1) * d]); }
        for (i, f) in g.facts.iter().enumerate() {
            let (r, k, p) = (f[0] as usize, f[1] as usize, f[2] as usize);
            for j in 0..d { x[1][i * d + j] = self.rel[r * d + j] + self.kind[k * d + j] + self.present[p * d + j]; }
        }
        let mv = 1usize; // "move" is the first relation of the vocabulary (id 1)
        for i in 0..n[2] { x[2][i * d..(i + 1) * d].copy_from_slice(&self.rel[mv * d..(mv + 1) * d]); }
        for layer in &self.layers {
            let mut out: Vec<Vec<f32>> = (0..4).map(|_| vec![]).collect();
            for t in 0..4 {
                if layer.b_sum[t].is_empty() || n[t] == 0 { continue; }
                let mut o = vec![0.0f32; n[t] * d];
                for r in 0..n[t] { o[r * d..(r + 1) * d].copy_from_slice(&layer.b_sum[t]); }
                gemm(&mut o, &x[t], n[t], d, &layer.r_sum[t], d, 1.0);
                out[t] = o;
            }
            for c in &layer.convs {
                let Some((es, ed)) = g.edge(&c.edge) else { continue };
                let (from, to) = if c.rev { (ed, es) } else { (es, ed) };
                if from.is_empty() || n[c.dst] == 0 { continue; }
                let mut agg = vec![0.0f32; n[c.dst] * d];
                let mut cnt = vec![0u32; n[c.dst]];
                for (&s, &t) in from.iter().zip(to) {
                    let (s, t) = (s as usize, t as usize);
                    for j in 0..d { agg[t * d + j] += x[c.src][s * d + j]; }
                    cnt[t] += 1;
                }
                let rows: Vec<usize> = (0..n[c.dst]).filter(|&r| cnt[r] > 0).collect();
                let mut mean = vec![0.0f32; rows.len() * d];
                for (k, &r) in rows.iter().enumerate() { let inv = 1.0 / cnt[r] as f32; for j in 0..d { mean[k * d + j] = agg[r * d + j] * inv; } }
                let mut prod = vec![0.0f32; rows.len() * d];
                gemm(&mut prod, &mean, rows.len(), d, &c.w_l, d, 0.0);
                for (k, &r) in rows.iter().enumerate() { for j in 0..d { out[c.dst][r * d + j] += prod[k * d + j]; } }
            }
            for t in 0..4 {
                if out[t].is_empty() { continue; }
                for r in 0..n[t] {
                    let o = &mut out[t][r * d..(r + 1) * d];
                    let mean = o.iter().sum::<f32>() / d as f32;
                    let var = o.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / d as f32;
                    let inv = 1.0 / (var + EPS).sqrt();
                    for j in 0..d {
                        let y = (o[j] - mean) * inv * layer.norm_w[t][j] + layer.norm_b[t][j];
                        x[t][r * d + j] += y.max(0.0);
                    }
                }
            }
        }
        let m = n[2];
        let mut h = vec![0.0f32; m * d];
        for r in 0..m { h[r * d..(r + 1) * d].copy_from_slice(&self.b0); }
        gemm(&mut h, &x[2], m, d, &self.w0, d, 1.0);
        (0..m).map(|r| self.head_bias + (0..d).map(|j| h[r * d + j].max(0.0) * self.w1[j]).sum::<f32>()).collect()
    }

    /// The forward edge names the net reads (for a consistency validate against `graph::edge_names`).
    pub fn edge_names(&self) -> Vec<String> { edge_names(self.max_parts) }
}
