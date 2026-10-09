//! # maia-chess
//!
//! [Maia](https://github.com/CSSLab/maia-chess) (McIlroy-Young, Sen, Kleinberg & Anderson, 2020) predicts the move a
//! human of a given rating would play. The networks are Leela Chess Zero (lc0) networks; this crate runs them in pure
//! Rust: the position is encoded in lc0's classical 112 input planes, the network is run with
//! [tract](https://github.com/sonos/tract) (ONNX; convert the official weights with `lc0 leela2onnx`, see
//! `scripts/convert.sh`), and the policy is read through lc0's own 1,858-entry move table.
//!
//! Validated against lc0 itself (`tests/bridge.rs`): same top move on 203 of 204 census positions at Maia-1100 and
//! Maia-1900; the residual differences are first-move openings given without history (pass the history to remove them).
//!
//! ```no_run
//! use maia_chess::Maia;
//! let maia = Maia::load("maia-1500.onnx").unwrap();
//! let pos: shakmaty::Chess = shakmaty::fen::Fen::from_ascii(b"rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")
//!     .unwrap().into_position(shakmaty::CastlingMode::Standard).unwrap();
//! for (uci, p) in maia.policy(&pos, 1.0).unwrap().iter().take(3) { println!("{uci} {p:.3}"); }
//! ```
//!
//! License: GPL-3.0-only, like Maia and lc0 (the move table comes from lc0).

mod encode;
mod table;

pub use encode::{encode, encode_history, PLANES};
pub use table::{policy_index, move_index};

use shakmaty::{Chess, Position};
use tract_onnx::prelude::*;

type Plan = std::sync::Arc<TypedRunnableModel>;

/// A loaded Maia (or any lc0 classical-input) network.
pub struct Maia {
    plan: Plan,
}

#[derive(Debug, Clone, Copy, PartialEq)]
/// The value head: probabilities of a win, draw and loss for the side to move.
pub struct Wdl { pub win: f32, pub draw: f32, pub loss: f32 }

impl Maia {
    /// Load an ONNX network converted from lc0 weights (`lc0 leela2onnx`): input `/input/planes` [N, 112, 8, 8],
    /// outputs `/output/policy` [N, 1858] and `/output/wdl` [N, 3].
    pub fn load(path: impl AsRef<std::path::Path>) -> Result<Maia, String> {
        let plan = tract_onnx::onnx().model_for_path(path).map_err(|e| e.to_string())?
            .with_input_fact(0, f32::fact([1, 112, 8, 8]).into()).map_err(|e| e.to_string())?
            .into_optimized().map_err(|e| e.to_string())?
            .into_runnable().map_err(|e| e.to_string())?;
        Ok(Maia { plan })
    }

    fn run(&self, planes: Vec<f32>) -> Result<(Vec<f32>, Wdl), String> {
        let input: Tensor = tract_ndarray::Array4::from_shape_vec((1, 112, 8, 8), planes).map_err(|e| e.to_string())?.into();
        let out = self.plan.run(tvec!(input.into())).map_err(|e| e.to_string())?;
        let pol = out[0].to_plain_array_view::<f32>().map_err(|e| e.to_string())?.iter().copied().collect::<Vec<f32>>();
        let w = out[1].to_plain_array_view::<f32>().map_err(|e| e.to_string())?.iter().copied().collect::<Vec<f32>>();
        let m = w.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let e: Vec<f32> = w.iter().map(|x| (x - m).exp()).collect();
        let z: f32 = e.iter().sum();
        Ok((pol, Wdl { win: e[0] / z, draw: e[1] / z, loss: e[2] / z }))
    }

    /// The policy over the legal moves of `pos` (softmax of the logits at temperature `temp`; Maia's own distribution
    /// is `temp = 1.0`, lc0's default display is 1.359), most likely first: [(uci, probability)]. Without history
    /// (lc0's rules for a bare position: copies of the position, none at the standard start, the pre-double-step
    /// position when an en-passant square is set).
    pub fn policy(&self, pos: &Chess, temp: f32) -> Result<Vec<(String, f32)>, String> {
        self.policy_planes(pos, encode(pos), temp)
    }

    /// The policy with the real game history (oldest first, the current position last; up to the last 8 are used).
    pub fn policy_history(&self, history: &[Chess], temp: f32) -> Result<Vec<(String, f32)>, String> {
        let pos = history.last().ok_or("empty history")?;
        self.policy_planes(pos, encode_history(history), temp)
    }

    fn policy_planes(&self, pos: &Chess, planes: Vec<f32>, temp: f32) -> Result<Vec<(String, f32)>, String> {
        let (logits, _) = self.run(planes)?;
        let mut rows = Vec::new();
        for m in pos.legal_moves() {
            let i = move_index(&m, pos.turn()).ok_or_else(|| format!("no policy index for {m:?}"))?;
            rows.push((m.to_uci(shakmaty::CastlingMode::Standard).to_string(), logits[i] / temp));
        }
        let mx = rows.iter().map(|r| r.1).fold(f32::NEG_INFINITY, f32::max);
        let z: f32 = rows.iter().map(|r| (r.1 - mx).exp()).sum();
        let mut out: Vec<(String, f32)> = rows.into_iter().map(|(u, l)| (u, (l - mx).exp() / z)).collect();
        out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        Ok(out)
    }

    /// The value head (win / draw / loss for the side to move).
    pub fn wdl(&self, pos: &Chess) -> Result<Wdl, String> { Ok(self.run(encode(pos))?.1) }
}

#[cfg(feature = "python")]
mod python;
