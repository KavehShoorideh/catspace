//! MAIA IN RUST (Kaveh 2026-10-09: "everything that goes into a frame or tactical calculation must be done in Rust --
//! get the Maia weights down into Rust"): the Maia networks (lc0 weights, converted by `lc0 leela2onnx`) run with tract
//! (pure-Rust ONNX), the position encoded in lc0's classical 112 planes, the policy read through lc0's own index table
//! (src/lc0_policy.txt, extracted from lc0's --verbose-move-stats output and checked complete: 1858 unique moves).
use shakmaty::{Chess, Color, Position, Role, Square, CastlingSide, Move};
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use tract_onnx::prelude::*;

type Plan = std::sync::Arc<TypedRunnableModel>;

static MODELS: OnceLock<Mutex<HashMap<String, Plan>>> = OnceLock::new();
static POLICY: OnceLock<HashMap<String, usize>> = OnceLock::new();

fn policy_index() -> &'static HashMap<String, usize> {
    POLICY.get_or_init(|| include_str!("lc0_policy.txt").lines().enumerate().map(|(i, u)| (u.to_string(), i)).collect())
}

fn model(path: &str) -> Result<Plan, String> {
    let lock = MODELS.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(p) = lock.lock().unwrap().get(path) { return Ok(p.clone()); }
    let plan = tract_onnx::onnx().model_for_path(path).map_err(|e| e.to_string())?
        .with_input_fact(0, f32::fact([1, 112, 8, 8]).into()).map_err(|e| e.to_string())?
        .into_optimized().map_err(|e| e.to_string())?
        .into_runnable().map_err(|e| e.to_string())?;
    lock.lock().unwrap().insert(path.to_string(), plan.clone());
    Ok(plan)
}

/// lc0 classical input (112 planes): 8 history slots of 13 planes (ours P N B R Q K, theirs P N B R Q K, repetition),
/// filled with the current position (lc0's "fen_only" history fill); then our O-O-O, our O-O, their O-O-O, their
/// O-O, black to move, the rule-50 count, zeros, ones. The board is seen from the side to move (mirrored for Black).
fn encode(pos: &Chess) -> Vec<f32> {
    let mut x = vec![0f32; 112 * 64];
    let us = pos.turn();
    let flip = |sq: Square| -> usize { let i = sq as usize; if us == Color::Black { i ^ 56 } else { i } };
    let roles = [Role::Pawn, Role::Knight, Role::Bishop, Role::Rook, Role::Queen, Role::King];
    let b = pos.board();
    // lc0's history fill, measured against lc0 (bridge 2026-10-09): no fill at the standard starting position (a game
    // with no past); with an en-passant square, the earlier slots hold the position BEFORE the double step (the pawn
    // back), which is how the network sees that en passant is possible; otherwise copies of the current position
    let start = pos.board() == Chess::default().board() && us == Color::White;
    let mut prev = b.clone();
    if let Some(ep) = pos.ep_square(shakmaty::EnPassantMode::Always) {
        let (now_sq, back_sq) = if us == Color::White {
            (Square::from_coords(ep.file(), shakmaty::Rank::Fifth), Square::from_coords(ep.file(), shakmaty::Rank::Seventh))
        } else {
            (Square::from_coords(ep.file(), shakmaty::Rank::Fourth), Square::from_coords(ep.file(), shakmaty::Rank::Second))
        };
        if let Some(p) = prev.remove_piece_at(now_sq) { prev.set_piece_at(back_sq, p); }
    }
    let slots = if start || std::env::var("MAIA_HISTORY_FILL").map(|v| v == "0").unwrap_or(false) { 1 } else { 8 };
    for slot in 0..slots {
        let bb = if slot == 0 { b } else { &prev };
        for (k, role) in roles.iter().enumerate() {
            for sq in bb.by_piece(role.of(us)) { x[(slot * 13 + k) * 64 + flip(sq)] = 1.0; }
            for sq in bb.by_piece(role.of(!us)) { x[(slot * 13 + 6 + k) * 64 + flip(sq)] = 1.0; }
        }
    }
    let castles = pos.castles();
    let fill = |x: &mut Vec<f32>, plane: usize, v: f32| { for i in 0..64 { x[plane * 64 + i] = v; } };
    if castles.has(us, CastlingSide::QueenSide) { fill(&mut x, 104, 1.0); }
    if castles.has(us, CastlingSide::KingSide) { fill(&mut x, 105, 1.0); }
    if castles.has(!us, CastlingSide::QueenSide) { fill(&mut x, 106, 1.0); }
    if castles.has(!us, CastlingSide::KingSide) { fill(&mut x, 107, 1.0); }
    if us == Color::Black { fill(&mut x, 108, 1.0); }
    if std::env::var("MAIA_NO_RULE50").map(|v| v == "1").unwrap_or(false) { } else { fill(&mut x, 109, pos.halfmoves() as f32); }
    fill(&mut x, 111, 1.0);
    x
}

fn nn_key(m: &Move, us: Color) -> String {
    let mirror = |sq: Square| -> Square { if us == Color::Black { Square::new((sq as u32) ^ 56) } else { sq } };
    let (from, to) = match m {
        Move::Castle { king, rook } => (*king, *rook),                // lc0: castling is the king taking its own rook (e1h1)
        _ => (m.from().unwrap(), m.to()),
    };
    let mut s = format!("{}{}", mirror(from), mirror(to));
    if let Some(p) = m.promotion() {
        if p != Role::Knight { s.push(p.char()); }                      // lc0: a plain move to the last rank = a knight
    }
    s
}

/// The policy over the legal moves (softmax of the network's logits at `temp`): [(uci, probability)].
pub fn policy(pos: &Chess, model_path: &str, temp: f32) -> Result<Vec<(String, f32)>, String> {
    let plan = model(model_path)?;
    let x = encode(pos);
    let input: Tensor = tract_ndarray::Array4::from_shape_vec((1, 112, 8, 8), x).map_err(|e| e.to_string())?.into();
    let out = plan.run(tvec!(input.into())).map_err(|e| e.to_string())?;
    let logits = out[0].to_plain_array_view::<f32>().map_err(|e| e.to_string())?;
    let idx = policy_index();
    let us = pos.turn();
    let mut rows: Vec<(String, f32)> = Vec::new();
    for m in pos.legal_moves() {
        let k = nn_key(&m, us);
        let Some(&i) = idx.get(&k) else { return Err(format!("no policy index for {}", k)) };
        rows.push((m.to_uci(shakmaty::CastlingMode::Standard).to_string(), logits[[0, i]] / temp));
    }
    let mx = rows.iter().map(|r| r.1).fold(f32::NEG_INFINITY, f32::max);
    let z: f32 = rows.iter().map(|r| (r.1 - mx).exp()).sum();
    Ok(rows.into_iter().map(|(u, l)| (u, (l - mx).exp() / z)).collect())
}
