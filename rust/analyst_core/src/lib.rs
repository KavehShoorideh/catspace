//! The analyst's hot path in Rust (Kaveh 2026-09-24: "do some sort of these calculations in Rust"): a chess
//! position with the frame's pieces as ids, the rules-level relations, the actions of a piece, the frame-local
//! outcome (one reply or the k-ply episode with a pass), and literal masks over many rows at once. Move generation
//! comes from shakmaty (import, don't reinvent). Nothing here knows what a lemma is.
use pyo3::prelude::*;
mod rules;
mod maia;
mod waiting;
mod paths;
mod plan;
pub mod lemma;
mod threat;
mod ledger;
mod pns;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::collections::HashMap;
static NODES: AtomicU64 = AtomicU64::new(0);
use shakmaty::{fen::Fen, CastlingMode, Chess, Color, FromSetup, Move, Position, Role, Square, Piece};

fn value(role: Role) -> f64 {
    match role { Role::Pawn => 1.0, Role::Knight => 3.0, Role::Bishop => 3.0, Role::Rook => 5.0, Role::Queen => 9.0, Role::King => 100.0 }
}

fn parse(fen: &str) -> Option<Chess> {
    let f: Fen = fen.parse().ok()?;
    f.into_position(CastlingMode::Standard).ok()
}

fn sq(i: i32) -> Option<Square> { if (0..64).contains(&i) { Some(Square::new(i as u32)) } else { None } }

/// material of `side` counting only the frame's pieces (ids), captured ids are -1
fn frame_material(pos: &Chess, ids: &[i32], side: Color) -> f64 {
    let b = pos.board();
    let mut tot = 0.0;
    let mut seen = std::collections::HashSet::new();
    for &i in ids {
        if i < 0 || !seen.insert(i) { continue; }
        if let Some(p) = b.piece_at(Square::new(i as u32)) {
            tot += if p.color == side { value(p.role) } else { -value(p.role) };
        }
    }
    tot
}

fn ids_after(ids: &[i32], m: &Move, pos: &Chess) -> Vec<i32> {
    let from = m.from().map(|s| s as i32).unwrap_or(-1);
    let to = m.to() as i32;
    let mut captured: Vec<i32> = vec![];
    if m.is_capture() {
        captured.push(if m.is_en_passant() { Square::from_coords(m.to().file(), m.from().unwrap().rank()) as i32 } else { to });
    }
    let mut moved = std::collections::HashMap::new();
    moved.insert(from, to);
    if let Move::Castle { king, rook } = m {
        let (rf, rt) = if rook.file() > king.file() { (rook, Square::from_coords(shakmaty::File::F, king.rank())) } else { (rook, Square::from_coords(shakmaty::File::D, king.rank())) };
        let kt = if rook.file() > king.file() { Square::from_coords(shakmaty::File::G, king.rank()) } else { Square::from_coords(shakmaty::File::C, king.rank()) };
        moved.insert(*king as i32, kt as i32);
        moved.insert(*rf as i32, rt as i32);
    }
    let _ = pos;
    ids.iter().map(|&s| if let Some(&t) = moved.get(&s) { t } else if captured.contains(&s) { -1 } else { s }).collect()
}

/// the frame episode: `depth` plies, pass allowed (the line goes on), captures/checks by anyone, quiet moves by
/// frame pieces; material of `side` at the leaves, mate +-1000
fn episode(pos: &Chess, ids: &[i32], side: Color, depth: u32, mut alpha: f64, mut beta: f64) -> f64 {
    NODES.fetch_add(1, Ordering::Relaxed);
    let legals = pos.legal_moves();
    let in_check = pos.is_check();
    let mate = if in_check && legals.is_empty() { if pos.turn() == side { -1000.0 } else { 1000.0 } } else { 0.0 };
    let stand = frame_material(pos, ids, side) + mate;
    if depth == 0 || legals.is_empty() { return stand; }
    let us = pos.turn() == side;
    let frame_sq: std::collections::HashSet<i32> = ids.iter().cloned().filter(|&x| x >= 0).collect();
    let mut children: Vec<(f64, Chess, Vec<i32>)> = vec![];
    if !in_check {
        if let Some(passed) = pass(pos) { children.push((1e9, passed, ids.to_vec())); }   // the pass first: a bound before any line, as the Python episode orders it
    }
    for m in legals.iter() {
        let quiet_frame = depth > 1 && m.from().map(|f| frame_sq.contains(&(f as i32))).unwrap_or(false);
        let forcing = in_check || m.is_capture() || m.is_promotion();
        if !forcing && !quiet_frame {
            // a quiet move by an outside piece counts only when it gives check: detect on the child
            let mut p2 = pos.clone(); p2.play_unchecked(m);
            if !p2.is_check() { continue; }
            children.push((0.5, p2, ids_after(ids, m, pos)));
            continue;
        }
        let mut p2 = pos.clone(); p2.play_unchecked(m);
        children.push((cap_value(m), p2, ids_after(ids, m, pos)));
    }
    children.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    let mut best: Option<f64> = None;
    for (_, child, cids) in children {
        let v = episode(&child, &cids, side, depth - 1, alpha, beta);
        best = Some(match best { None => v, Some(b) => if us { b.max(v) } else { b.min(v) } });
        if us { alpha = alpha.max(best.unwrap()); } else { beta = beta.min(best.unwrap()); }
        if alpha >= beta { break; }
    }
    best.unwrap_or(stand)
}

fn cap_value(m: &Move) -> f64 { m.capture().map(value).unwrap_or(0.0) + if m.is_promotion() { 8.0 } else { 0.0 } }

fn gives_check(pos: &Chess, m: &Move) -> bool { let mut p = pos.clone(); p.play_unchecked(m); p.is_check() }

fn pass(pos: &Chess) -> Option<Chess> { pos.clone().swap_turn().ok() }

// ---------------------------------------------------------------- relations (rules-level)
fn attacks_sq(pos: &Chess, from: Square, to: Square) -> bool {
    let b = pos.board();
    b.piece_at(from).map(|p| b.attacks_from(from).contains(to) && p.role != Role::King || (p.role == Role::King && b.attacks_from(from).contains(to))).unwrap_or(false)
}

#[pyfunction]
/// relation values for a batch: rows are (fen, ids, move_uci or ""), literals are (name, slots); returns one bitmask
/// (as a Vec<u8> of 0/1 per row) per literal. Names: same_side attacks defends aligned between worth_ge attacked
/// defended slides | gives_check captures captures_worth_ge stays_aligned lands_attacked
fn literal_masks(rows: Vec<(String, Vec<i32>, String)>, literals: Vec<(String, Vec<usize>)>) -> PyResult<Vec<Vec<u8>>> {
    let mut out = vec![vec![0u8; rows.len()]; literals.len()];
    for (ri, (fen, ids, mv)) in rows.iter().enumerate() {
        let pos = match parse(fen) { Some(p) => p, None => continue };
        let m: Option<Move> = if mv.is_empty() { None } else { shakmaty::uci::UciMove::from_ascii(mv.as_bytes()).ok().and_then(|u| u.to_move(&pos).ok()) };
        for (li, (name, slots)) in literals.iter().enumerate() {
            let args: Vec<i32> = slots.iter().map(|&s| ids.get(s).cloned().unwrap_or(-1)).collect();
            if args.iter().any(|&a| a < 0) { continue; }
            let v = relation(&pos, name, &args, m.as_ref());
            if v { out[li][ri] = 1; }
        }
    }
    Ok(out)
}

fn relation(pos: &Chess, name: &str, a: &[i32], m: Option<&Move>) -> bool {
    let b = pos.board();
    let piece = |i: i32| -> Option<Piece> { sq(i).and_then(|s| b.piece_at(s)) };
    match name {
        "same_side" => matches!((piece(a[0]), piece(a[1])), (Some(x), Some(y)) if x.color == y.color),
        "attacks" => match (sq(a[0]), sq(a[1]), piece(a[0])) { (Some(f), Some(t), Some(_)) => b.attacks_from(f).contains(t), _ => false },   // geometric, any colour (as the Python relation)
        "defends" => match (sq(a[0]), sq(a[1]), piece(a[0]), piece(a[1])) { (Some(f), Some(t), Some(x), Some(y)) => x.color == y.color && b.attacks_from(f).contains(t), _ => false },
        "aligned" => match (sq(a[0]), sq(a[1])) { (Some(f), Some(t)) => f != t && (f.file() == t.file() || f.rank() == t.rank() || (f.file() as i32 - t.file() as i32).abs() == (f.rank() as i32 - t.rank() as i32).abs()), _ => false },
        "between" => match (sq(a[0]), sq(a[1]), sq(a[2])) { (Some(x), Some(f), Some(t)) => shakmaty::attacks::between(f, t).contains(x) && (shakmaty::attacks::between(f, t) & b.occupied()).count() == 1, _ => false },
        "worth_ge" => match (piece(a[0]), piece(a[1])) { (Some(x), Some(y)) => value(x.role) >= value(y.role), _ => false },
        "attacked" => match (sq(a[0]), piece(a[0])) { (Some(s), Some(p)) => b.attacks_to(s, !p.color, b.occupied()).any(), _ => false },
        "defended" => match (sq(a[0]), piece(a[0])) { (Some(s), Some(p)) => b.attacks_to(s, p.color, b.occupied()).any(), _ => false },
        "slides" => piece(a[0]).map(|p| matches!(p.role, Role::Bishop | Role::Rook | Role::Queen)).unwrap_or(false),
        "gives_check" => m.map(|mv| gives_check(pos, mv)).unwrap_or(false),
        "captures" => m.map(|mv| mv.is_capture()).unwrap_or(false),
        "captures_worth_ge" => match (m, piece(a[0])) { (Some(mv), Some(p)) => mv.capture().map(|r| value(r) >= value(p.role)).unwrap_or(false), _ => false },
        "lands_attacked" => m.map(|mv| { let mut p2 = pos.clone(); p2.play_unchecked(mv); p2.board().attacks_to(mv.to(), p2.turn(), p2.board().occupied()).any() }).unwrap_or(false),
        "stays_aligned" => match (m, sq(a[0])) { (Some(mv), Some(t)) => { let f = mv.to(); f != t && (f.file() == t.file() || f.rank() == t.rank() || (f.file() as i32 - t.file() as i32).abs() == (f.rank() as i32 - t.rank() as i32).abs()) && (shakmaty::attacks::between(f, t) & { let mut p2 = pos.clone(); p2.play_unchecked(mv); p2.board().occupied() }).is_empty() }, _ => false },
        _ => false,
    }
}

#[pyfunction]
/// the frame-local outcome of every action of piece `actor` in `fen`: (uci, delta) with delta = episode value after
/// the action minus the baseline (the null-move episode when relative, else the material before); depth 0 = the
/// frame's size
fn outcomes(fen: String, ids: Vec<i32>, actor: i32, depth: u32, relative: bool) -> PyResult<Vec<(String, f64)>> {
    let pos = match parse(&fen) { Some(p) => p, None => return Ok(vec![]) };
    let side = match sq(actor).and_then(|s| pos.board().piece_at(s)) { Some(p) => p.color, None => return Ok(vec![]) };
    let d = if depth == 0 { ids.len() as u32 } else { depth };
    // the actor's side to move
    let pos = if pos.turn() == side { pos } else { match pass(&pos) { Some(p) => p, None => return Ok(vec![]) } };
    let base = if relative {
        if pos.is_check() { return Ok(vec![]); }
        match pass(&pos) { Some(p) => episode(&p, &ids, side, d, -1e9, 1e9), None => return Ok(vec![]) }
    } else { frame_material(&pos, &ids, side) };
    let mut out = vec![];
    for m in pos.legal_moves() {
        if m.from().map(|f| f as i32) != Some(actor) { continue; }
        let mut p2 = pos.clone(); p2.play_unchecked(&m);
        let ids2 = ids_after(&ids, &m, &pos);
        let v = episode(&p2, &ids2, side, d, -1e9, 1e9);
        out.push((shakmaty::uci::UciMove::from_standard(&m).to_string(), v - base));
    }
    Ok(out)
}

#[pyfunction]
fn nodes() -> u64 { NODES.swap(0, Ordering::Relaxed) }

// ---------------------------------------------------------------------------------------------------------------------
// THE KERNEL'S LOCAL CHECK, mirrored (Ideas/Fork.lean `localCheck`, proved sound by `localCheck_sound`): after move m,
// EVERY legal reply is answered by one of our ANSWERS (a capture, a move of a piece the opponent attacks, or a king
// move) after which, whatever they reply, our material is up by k at the leaves (their mate counts as reached, their
// stalemate as not). The Lean kernel is the auditor: the Python wrapper verifies this mirror against the binary on a
// sample (Kaveh 2026-10-08: "some things need to be sent out to the Rust layer"). Piece values as the kernel's
// `pieceValue` (king 0).

fn kernel_value(role: Role) -> i64 {
    match role { Role::Pawn => 1, Role::Knight => 3, Role::Bishop => 3, Role::Rook => 5, Role::Queen => 9, Role::King => 0 }
}

pub(crate) fn kernel_material(pos: &Chess, side: Color) -> i64 {
    let b = pos.board();
    let mut tot = 0;
    for sq in b.occupied() {
        if let Some(p) = b.piece_at(sq) {
            tot += if p.color == side { kernel_value(p.role) } else { -kernel_value(p.role) };
        }
    }
    tot
}

fn after(pos: &Chess, m: &Move) -> Chess { let mut p = pos.clone(); p.play_unchecked(m); p }

/// `forcedAtLeaves c goal q 1` with the opponent to move at q: every reply reaches the goal (no reply: mated = reached).
fn holds_after_every_reply(q: &Chess, side: Color, target: i64) -> bool {
    NODES.fetch_add(1, Ordering::Relaxed);
    let ms = q.legal_moves();
    if ms.is_empty() { return q.is_check(); }
    ms.iter().all(|r| kernel_material(&after(q, r), side) >= target)
}

fn answers(q: &Chess) -> Vec<Move> {
    let them = !q.turn();
    let occ = q.board().occupied();
    q.legal_moves().into_iter().filter(|m2| {
        m2.is_capture()
            || m2.role() == Role::King
            || m2.from().map(|f| q.board().attacks_to(f, them, occ).any()).unwrap_or(false)
    }).collect()
}

fn local_check_pos(p: &Chess, m: &Move, k: i64) -> bool {
    let c = p.turn();
    let target = kernel_material(p, c) + k;
    let q = after(p, m);
    let replies = q.legal_moves();
    if replies.is_empty() { return q.is_check(); }
    replies.iter().all(|r| {
        let q2 = after(&q, r);
        answers(&q2).iter().any(|m2| holds_after_every_reply(&after(&q2, m2), c, target))
    })
}

// THE FRAME-SIZED CERTIFICATE (Kaveh 2026-10-08: "instead of a fixed ply depth ... reevaluate after one piece is gone,
// the frame is smaller ... if the size of the frame is six, you need at least six move depths"): the same three plies as
// localCheck, but the leaf is not raw material: the capture frames that remain are RESOLVED -- a captures-only search
// where either side may stop (`settle`); every capture removes a piece, so it ends when the frames are exhausted, at a
// depth set by their size, not a fixed horizon. Claim: we can force material >= start + k at the moment the capture
// sequence ends. A Lean theorem for this form is still to be written (status: certified in Rust).

/// The side to move's best net material gain from captures (and queen promotions) alone, either side free to stop --
/// EXCEPT in check: a side in check cannot stand pat, it must evade (bug 2026-10-08, found reading puzzle 04XTY: the
/// leaf stood pat under check and could certify a false claim). `us`: the mover is the claim's protagonist. The search
/// is bounded by `depth`; at the cutoff a side in check gets the conservative value -- the worst for us, the best for
/// them -- so the bound stays sound.
pub(crate) fn settle_d(pos: &Chess, mut alpha: i64, beta: i64, us: bool, depth: u32) -> i64 {
    NODES.fetch_add(1, Ordering::Relaxed);
    let bud = BUDGET.load(Ordering::Relaxed);                       // the leaf search is charged to the same budget
    if bud == 0 { return if us { -1000 } else { 1000 }; }          // spent: conservative for the claim
    if bud != u64::MAX { BUDGET.store(bud - 1, Ordering::Relaxed); }
    let in_check = pos.is_check();
    if in_check {
        let ms = pos.legal_moves();
        if ms.is_empty() { return -1000; }                         // mated
        // evasions are followed at most four plies deep inside the leaf (2026-10-08: evasions to depth 12 exploded)
        if depth == 0 || depth + 4 < SETTLE_DEPTH { return if us { -1000 } else { 1000 }; }   // conservative
        let mut best = -10000;
        for m in ms {
            let g = m.capture().map(kernel_value).unwrap_or(0) + if m.is_promotion() { 8 } else { 0 };
            let q = after(pos, &m);
            let v = g - settle_d(&q, g - beta, g - alpha.max(best), !us, depth - 1);
            if v > best { best = v; if best >= beta { return best; } }
        }
        return best;
    }
    if 0 >= beta { return 0; }                                     // standing pat already refutes the window
    if 0 > alpha { alpha = 0; }
    if depth == 0 { return alpha; }
    let mut caps: Vec<Move> = pos.legal_moves().into_iter()
        .filter(|m| m.is_capture() || m.promotion() == Some(Role::Queen)).collect();   // promotions count
    caps.sort_by_key(|m| -(m.capture().map(kernel_value).unwrap_or(0) + if m.is_promotion() { 8 } else { 0 }));
    for m in caps {
        let g = m.capture().map(kernel_value).unwrap_or(0) + if m.is_promotion() { 8 } else { 0 };
        let q = after(pos, &m);
        let v = if q.is_checkmate() { 1000 } else { g - settle_d(&q, g - beta, g - alpha, !us, depth - 1) };
        if v > alpha {
            alpha = v;
            if alpha >= beta { return alpha; }
        }
    }
    alpha
}

pub(crate) const SETTLE_DEPTH: u32 = 12;
const PURSUIT_PLIES: u32 = 4;                                       // our answer, their reply, our answer, their reply

fn holds_after_every_reply_settled(q: &Chess, side: Color, target: i64) -> bool {
    NODES.fetch_add(1, Ordering::Relaxed);
    let ms = q.legal_moves();
    if ms.is_empty() { return q.is_check(); }
    ms.iter().all(|r| {
        let q4 = after(q, r);
        if q4.is_checkmate() { return false; }                       // they mated us
        let need = target - kernel_material(&q4, side);
        if need <= 0 && !q4.is_check() { return true; }
        settle_d(&q4, need - 1, need, true, SETTLE_DEPTH) >= need  // a null window: only "at least the target" is asked
    })
}

fn local_check_frames_pos(p: &Chess, m: &Move, k: i64) -> bool {
    let c = p.turn();
    let target = kernel_material(p, c) + k;
    let q = after(p, m);
    let replies = q.legal_moves();
    if replies.is_empty() { return q.is_check(); }
    replies.iter().all(|r| {
        let q2 = after(&q, r);
        answers(&q2).iter().any(|m2| holds_after_every_reply_settled(&after(&q2, m2), c, target))
    })
}

// THE PURSUIT CERTIFICATE (2026-10-08, from the trapped-piece failures: a target escapes, is attacked again, is trapped
// -- reasoning chains that follow a piece): `c` plays m, then for `n` more plies at our nodes some ANSWER, at theirs
// every legal reply; the goal -- our material + the resolved captures >= target -- is read at every one of our nodes,
// so a chain may finish early. Our answers: captures, checks, king moves, moves of a piece they attack, and moves that
// ATTACK A FOCUS PIECE (the claim's targets, followed by identity as they move). Answers are legal moves, so the
// certificate stays sound (the localCheck_sound argument); it needs its own Lean theorem.

fn goal_now(q: &Chess, side: Color, target: i64) -> bool {
    let need = target - kernel_material(q, side);
    if need <= 0 && !q.is_check() { return true; }
    settle_d(q, need - 1, need, true, SETTLE_DEPTH) >= need
}

/// m attacks a focus piece, OR covers a square a focus piece could flee to (2026-10-08, puzzle 07bjx: ...Nd6 never
/// attacks the trapped knight, it covers its escapes -- cornering inside the chain, for every piece).
fn attacks_any(q: &Chess, m: &Move, focus: &[Square]) -> bool {
    let after_m = after(q, m);
    let to = m.to();
    let b = after_m.board();
    let Some(p) = b.piece_at(to) else { return false };
    let a = shakmaty::attacks::attacks(to, p, b.occupied());
    focus.iter().any(|f| {
        if a.contains(*f) { return true; }
        match b.piece_at(*f) {
            Some(fp) => (shakmaty::attacks::attacks(*f, fp, b.occupied()) & a & !b.by_color(fp.color)).any(),
            None => false,
        }
    })
}

fn pursue(q: &Chess, side: Color, target: i64, focus: &[Square], plies: u32) -> bool {
    NODES.fetch_add(1, Ordering::Relaxed);
    let b = BUDGET.load(Ordering::Relaxed);
    if b == 0 { return false; }                                     // the budget is spent: not certified (sound)
    if b != u64::MAX { BUDGET.store(b - 1, Ordering::Relaxed); }
    if q.turn() == side {
        if goal_now(q, side, target) { return true; }
        if plies == 0 { return false; }
        let them = !side;
        let occ = q.board().occupied();
        let ms: Vec<Move> = q.legal_moves().into_iter().filter(|m2| {
            m2.is_capture() || m2.role() == Role::King
                || m2.from().map(|f| q.board().attacks_to(f, them, occ).any()).unwrap_or(false)
                || after(q, m2).is_check()
                || attacks_any(q, m2, focus)
                // CHAINS OF LEMMAS (2026-10-08, puzzle 076jI: Bxb5 Rxe7 c4 traps ANOTHER piece): a move the lemma
                // registry proposes -- one that corners or traps any of their pieces -- is an answer too; only while
                // two or more plies remain (the rules run per move)
                || (plies >= 2 && { let q2 = after(q, m2); !q2.is_check() && corners_any(&q2, side, None) })
        }).collect();
        // THE ORDER (Kaveh 2026-10-08: reason around the cornering -- cover its escapes): captures of a focus piece first,
        // then the moves leaving it the fewest safe escapes (bitboards), then the rest
        let mut ordered: Vec<(i64, Move)> = ms.into_iter().map(|m2| {
            let q2 = after(q, &m2);
            let key = if focus.contains(&m2.to()) { -1 } else {
                focus.iter().filter(|f| q2.board().piece_at(**f).map_or(false, |p| p.color != side))
                    .map(|f| safe_escapes(&q2, side, *f)).min().unwrap_or(50)
            };
            (key, m2)
        }).collect();
        ordered.sort_by_key(|(k, _)| *k);
        ordered.iter().any(|(_, m2)| {
            let q2 = after(q, m2);
            let f2: Vec<Square> = focus.iter().filter(|f| q2.board().piece_at(**f).map_or(false, |p| p.color != side)).copied().collect();
            pursue(&q2, side, target, &f2, plies - 1)
        })
    } else {
        let ms = q.legal_moves();
        if ms.is_empty() { return q.is_check(); }
        if plies == 0 { return goal_now_their_turn(q, side, target); }
        ms.iter().all(|r| {
            let q2 = after(q, r);
            // follow a focus piece that moved
            let f2: Vec<Square> = focus.iter().map(|f| if r.from() == Some(*f) { r.to() } else { *f }).collect();
            pursue(&q2, side, target, &f2, plies - 1)
        })
    }
}

fn goal_now_their_turn(q: &Chess, side: Color, target: i64) -> bool {
    q.legal_moves().iter().all(|r| { let q4 = after(q, r); !q4.is_checkmate() && goal_now(&q4, side, target) })
}

fn pursuit_check_pos(p: &Chess, m: &Move, k: i64, focus: &[Square], plies: u32) -> bool {
    let c = p.turn();
    let target = kernel_material(p, c) + k;
    let q = after(p, m);
    if q.is_checkmate() { return true; }
    // their piece on a focus square moved? (m is ours; focus is theirs: unchanged)
    pursue(&q, c, target, focus, plies)
}

// THE CORNERING SEARCH, by escapes not by plies (Kaveh 2026-10-08: "we kick the queen, the queen has three squares to
// go -- how do we cover all three? ... the bishop must go there to cover those two squares, then ..."): the target's
// SAFE ESCAPES (the squares it can move to where we do not win it) are the state. At our nodes the only moves tried are
// those that WIN it now (a capture of it), or strictly SHRINK its escape set, or attack it (kick it) -- ordered by how
// many escapes they leave; the chain ends when it is won or when we cannot shrink the set (no progress). At their
// nodes every legal reply (sound); the target is followed by identity when it moves. A safety cap on depth only.

/// The target's escape squares, read with bitboards (no rule run): a square e it can step to is UNSAFE when one of our
/// pieces cheaper than it attacks e, or we attack e and none of theirs does (the target itself off the board).
pub(crate) fn escape_set(q: &Chess, us: Color, t: Square) -> (shakmaty::Bitboard, shakmaty::Bitboard) {
    let b = q.board();
    let Some(tp) = b.piece_at(t) else { return (shakmaty::Bitboard::EMPTY, shakmaty::Bitboard::EMPTY) };
    let them = tp.color;
    let occ_wo = b.occupied().without(t);
    let tv = kernel_value(tp.role).max(1);
    let reach = shakmaty::attacks::attacks(t, tp, b.occupied()) & !b.by_color(them);
    let mut safe = shakmaty::Bitboard::EMPTY;
    for e in reach {
        let ours = b.attacks_to(e, us, occ_wo);
        let theirs = b.attacks_to(e, them, occ_wo);
        let cheaper = ours.into_iter().any(|a| b.piece_at(a).map_or(false, |p| kernel_value(p.role) < tv && p.role != Role::King) );
        let unsafe_e = cheaper || (ours.any() && !theirs.any());
        if !unsafe_e { safe.add(e); }
    }
    (safe, reach)
}

fn safe_escapes(q: &Chess, us: Color, t: Square) -> i64 { escape_set(q, us, t).0.count() as i64 }

/// Does some piece of theirs (worth 3 or more) have no safe escape and stand where we attack it or attack all its
/// escapes -- the cornering question answered with bitboards (2026-10-08: running the Ascent rules for every candidate
/// at every node made the chains too slow).
fn corners_any(q: &Chess, us: Color, z: Option<shakmaty::Bitboard>) -> bool {
    let b = q.board();
    let them = !us;
    for sq in b.by_color(them) & z.unwrap_or(shakmaty::Bitboard::FULL) {
        let Some(p) = b.piece_at(sq) else { continue };
        if p.role == Role::King || kernel_value(p.role) < 3 { continue; }
        let (safe, reach) = escape_set(q, us, sq);
        if !safe.is_empty() || reach.is_empty() { continue; }
        if b.attacks_to(sq, us, b.occupied()).any() { return true; }
    }
    false
}

fn corner(q: &Chess, side: Color, target: i64, t: Square, cap: u32) -> bool {
    NODES.fetch_add(1, Ordering::Relaxed);
    let b = BUDGET.load(Ordering::Relaxed);
    if b == 0 { return false; }
    if b != u64::MAX { BUDGET.store(b - 1, Ordering::Relaxed); }
    if q.turn() == side {
        if goal_now(q, side, target) { return true; }
        if cap == 0 { return false; }
        let them = !side;
        if q.board().piece_at(t).map_or(true, |p| p.color != them) { return false; }   // the target is gone
        let (esc, _) = escape_set(q, side, t);
        let e0 = esc.count() as i64;
        let attacked = q.board().attacks_to(t, side, q.board().occupied()).any();
        let goal_sq = esc.with(t);                                             // what our move must touch
        let mut cands: Vec<(i64, Move)> = Vec::new();
        for m2 in q.legal_moves() {
            if m2.to() == t { cands.push((-1, m2)); continue; }                     // win it now
            // CANDIDATES FROM THE ESCAPE SET (Kaveh: "the bishop must go there to cover those two squares"): the moved
            // piece, from its new square, must attack an escape square or the target
            let Some(p) = q.board().piece_at(m2.from().unwrap_or(m2.to())) else { continue };
            let occ2 = q.board().occupied().without(m2.from().unwrap_or(m2.to())).with(m2.to());
            let touches = !(shakmaty::attacks::attacks(m2.to(), p, occ2) & goal_sq).is_empty();
            let q2 = after(q, &m2);
            if touches {
                let e1 = safe_escapes(&q2, side, t);
                let kicks = !attacked && q2.board().attacks_to(t, side, q2.board().occupied()).any();
                if e1 < e0 || (kicks && e1 <= e0) { cands.push((e1, m2)); continue; }   // progress: shrink, or kick
            }
            // ...and the counterplay answers (2026-10-08, puzzle 07bjx: White's knight hits OUR knight; saving it touches
            // no escape square): captures, checks, king moves, moves of a piece of ours they attack -- after the
            // escape-set moves in the ordering
            let attacked_from = m2.from().map(|f| q.board().attacks_to(f, them, q.board().occupied()).any()).unwrap_or(false);
            if m2.is_capture() || m2.role() == Role::King || attacked_from || q2.is_check() { cands.push((100, m2)); }
        }
        cands.sort_by_key(|(e, _)| *e);
        cands.into_iter().any(|(_, m2)| corner(&after(q, &m2), side, target, t, cap - 1))
    } else {
        let ms = q.legal_moves();
        if ms.is_empty() { return q.is_check(); }
        if cap == 0 { return goal_now_their_turn(q, side, target); }
        ms.iter().all(|r| {
            let t2 = if r.from() == Some(t) { r.to() } else { t };
            corner(&after(q, r), side, target, t2, cap - 1)
        })
    }
}

fn corner_check_pos(p: &Chess, m: &Move, k: i64, t: Square) -> bool {
    let c = p.turn();
    let target = kernel_material(p, c) + k;
    let q = after(p, m);
    if q.is_checkmate() { return true; }
    BUDGET.store(300_000, Ordering::Relaxed);
    let ok = corner(&q, c, target, t, 10);
    BUDGET.store(u64::MAX, Ordering::Relaxed);
    ok
}

/// corner_check(fen, uci, k, target square) -> Option<bool>
#[pyfunction]
fn corner_check(fen: String, uci: String, k: i64, target: String) -> PyResult<Option<bool>> {
    let Some(pos) = parse(&fen) else { return Ok(None) };
    let Ok(u) = uci.parse::<shakmaty::uci::UciMove>() else { return Ok(None) };
    let Ok(m) = u.to_move(&pos) else { return Ok(None) };
    let Ok(t) = target.parse::<Square>() else { return Ok(None) };
    Ok(Some(corner_check_pos(&pos, &m, k, t)))
}

// THE NODE BUDGET, one per thread (2026-10-09, waiting frames: the condition search runs certificates in parallel on
// every core, and a shared counter would let one search spend another's budget). Same load/store calls as the atomic.
thread_local! { static BUDGET_TL: std::cell::Cell<u64> = const { std::cell::Cell::new(u64::MAX) }; }
struct Budget;
impl Budget {
    fn load(&self, _: Ordering) -> u64 { BUDGET_TL.with(|c| c.get()) }
    fn store(&self, v: u64, _: Ordering) { BUDGET_TL.with(|c| c.set(v)) }
}
static BUDGET: Budget = Budget;
static FAST: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);   // inside a tree: no deep tier
// EVERY opponent reply searched inside certificates (the frame-independence pruning OFF): the audit of 2026-10-09 found
// the pruning certified 10.9% false claims (40 of 366) and the exact search costs 14 ms a claim -- unsound and unneeded.
static EXACT: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

/// set_exact(flag): search every opponent reply inside certificates (the frame-independence lemma off) -- the audit.
#[pyfunction]
fn set_exact(flag: bool) -> PyResult<()> { EXACT.store(flag, Ordering::Relaxed); Ok(()) }

// FRAME PROOFS (Kaveh 2026-10-11: "we should have evaluated any potential capture of my opponent on the board already,
// once ... they become multiple frames ... when we're checking opponent replies, check the moves that are available to
// my opponent, or specific frames that have a synergy with this one ... we don't want to research anything"). A proof
// reads its FRAME (the zone: the target, its escapes, the pieces bearing on them, the lines between, and BOTH KINGS --
// an attack on the king is a frame like any other, differing in value only) and the INDEX of both sides' other frames
// (the static exchange on every piece, carried through the search incrementally as the forcing search does), nothing
// else: a reply outside the frame that is not one of their live frames is a pass; the leaf is the material plus the
// composed standing read from the index (Lean Ideas.ExchangeLaw.exchangeLawComposed) instead of a capture search over
// the whole board. What the proof read is RECORDED (`DEP`: every square it searched a move from or to, every square
// whose exchange it recomputed, the zones) so the memory can reuse it when nothing it depends on changed (Lean
// Ideas.FrameLemma.frame_unchanged for the static part); a proof that judged a stalemate read the whole board (GLOBAL).
// Soundness is "modulo the pass": their preparatory moves outside the frame are not searched -- measured against the
// exact proofs by the frame audit (plan.rs) before this is switched on.
pub static FRAME_PROOFS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// set_frame_proofs(flag): proofs read their frame and the index, not the board (see FRAME_PROOFS).
#[pyfunction]
fn set_frame_proofs(flag: bool) -> PyResult<()> { FRAME_PROOFS.store(flag, Ordering::Relaxed); Ok(()) }
thread_local! {
    static FRAME_OFF_TL: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };     // the audit's exact re-run
    static DEP: std::cell::Cell<shakmaty::Bitboard> = const { std::cell::Cell::new(shakmaty::Bitboard::EMPTY) };
    static GLOBAL: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    // the breakdown of DEP: the frames' cores, the exchanges that changed, the leaves' reads (the review)
    static DEP_PARTS: std::cell::Cell<[shakmaty::Bitboard; 3]> = const { std::cell::Cell::new([shakmaty::Bitboard::EMPTY; 3]) };
    // the squares whose one-step movers a lemma verdict read (its arrivals); empty: derive from the dependency
    static DEP_ONE_STEP: std::cell::Cell<shakmaty::Bitboard> = const { std::cell::Cell::new(shakmaty::Bitboard::EMPTY) };
}
thread_local! { static CERT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) }; }
/// Run `f` as a CERTIFICATE CHECK: at their nodes only the stored reply is searched (Lean Ideas.Refutation.refuted_sound:
/// fixing their replies can only make the claim easier, so a claim that still fails fails outright).
pub(crate) fn with_cert<T>(f: impl FnOnce() -> T) -> T {
    let was = CERT.with(|c| c.replace(true));
    let out = f();
    CERT.with(|c| c.set(was));
    out
}
pub(crate) fn dep_one_step(b: shakmaty::Bitboard) { DEP_ONE_STEP.with(|d| d.set(d.get() | b)); }
pub(crate) fn take_one_step() -> shakmaty::Bitboard { DEP_ONE_STEP.with(|d| d.replace(shakmaty::Bitboard::EMPTY)) }
fn dep_part(i: usize, b: shakmaty::Bitboard) { DEP_PARTS.with(|d| { let mut v = d.get(); v[i] |= b; d.set(v); }); dep_add(b); }
/// The breakdown of the last proof's dependency: [core, changed exchanges, leaf reads] as square counts.
pub(crate) fn take_dep_parts() -> [u32; 3] { DEP_PARTS.with(|d| { let v = d.replace([shakmaty::Bitboard::EMPTY; 3]); [v[0].count() as u32, v[1].count() as u32, v[2].count() as u32] }) }
pub(crate) fn frame_on() -> bool { FRAME_PROOFS.load(Ordering::Relaxed) && !FRAME_OFF_TL.with(|c| c.get()) }
/// Run `f` with frame proofs off on this thread (the audit's exact answer to the same question).
pub(crate) fn with_frame_off<T>(f: impl FnOnce() -> T) -> T {
    let was = FRAME_OFF_TL.with(|c| c.replace(true));
    let out = f();
    FRAME_OFF_TL.with(|c| c.set(was));
    out
}
fn dep_add(b: shakmaty::Bitboard) { DEP.with(|d| d.set(d.get() | b)); }
fn dep_global() { GLOBAL.with(|g| g.set(true)); }
/// What the proofs since the last call read: (the squares, whether one read the whole board).
pub(crate) fn take_dep() -> (shakmaty::Bitboard, bool) {
    (DEP.with(|d| d.replace(shakmaty::Bitboard::EMPTY)), GLOBAL.with(|g| g.replace(false)))
}
/// The index a frame proof reads: the static exchange for `side` on each of their pieces (`us`) and for them on each of
/// ours (`them`), as the forcing search carries them (threat.rs FState).
#[derive(Clone, Copy)]
pub(crate) struct Idx { us: rules::ThrArr, them: rules::ThrArr }
pub(crate) fn idx_root(q: &Chess, side: Color) -> Idx { Idx { us: rules::threats_arr(q, side), them: rules::threats_arr(q, !side) } }
/// The index after `m` (played from the position before `q2`), the recomputed squares recorded as read.
fn idx_after(ix: &Idx, q2: &Chess, m: &Move, side: Color) -> Idx {
    let out = Idx { us: rules::threats_arr_inc(&ix.us, q2, m, side), them: rules::threats_arr_inc(&ix.them, q2, m, !side) };
    // the squares whose exchange CHANGED are what the leaf's reading of the index depends on (a square recomputed to the
    // same value influenced nothing); the moves themselves are covered by the attacks onto the frame (plan.rs inputs_differ)
    let mut changed = shakmaty::Bitboard::EMPTY;
    for t in 0..64usize { if out.us[t] != ix.us[t] || out.them[t] != ix.them[t] { changed.add(Square::new(t as u32)); } }
    dep_part(1, changed);
    out
}
fn kings(q: &Chess) -> shakmaty::Bitboard { q.board().kings() }
/// The line a stored proof of `m` plays: our answers and their refutations from the hints (the proof's path starts at
/// the move: prove_pos), "pass" for the outside reply; up to 12 plies.
pub(crate) fn proof_line(p: &Chess, m: &Move, hints: &Hints) -> Vec<String> {
    let mut line = vec![m.to_uci(CastlingMode::Standard).to_string()];
    let mut q = after(p, m);
    let mut path = child_path(0, move_code(m));
    while let Some(code) = hints.get(&path).copied() {
        if code == PASS_CODE {
            line.push("pass".to_string());
            let Ok(p2) = q.clone().swap_turn() else { break };
            q = p2;
        } else {
            let Some(mv) = q.legal_moves().into_iter().find(|x| move_code(x) == code) else { break };
            line.push(mv.to_uci(CastlingMode::Standard).to_string());
            q = after(&q, &mv);
        }
        path = child_path(path, code);
        if line.len() > 12 { break; }
    }
    line
}

/// frame_line(fen, uci, focus, plies, budget) -> (certified value or None, the line the value search decided on: our
/// answers and their refutations from the recorded proof, "pass" for the outside reply) -- the review hook.
#[pyfunction]
fn frame_line(fen: String, uci: String, focus: Vec<String>, plies: u32, budget: u64) -> PyResult<(Option<i64>, Vec<String>)> {
    let Some(pos) = parse(&fen) else { return Ok((None, vec![])) };
    let Ok(u) = uci.parse::<shakmaty::uci::UciMove>() else { return Ok((None, vec![])) };
    let Ok(m) = u.to_move(&pos) else { return Ok((None, vec![])) };
    let f: Vec<Square> = focus.iter().filter_map(|s| s.parse().ok()).collect();
    let (v, rec) = with_hints(None, || certified_value(&pos, &m, &f, plies, budget));
    let mut line = vec![];
    let mut q = after(&pos, &m);
    let mut path = 0u64;                                              // the value search records from the root
    while let Some(code) = rec.get(&path).copied() {
        if code == PASS_CODE {
            line.push("pass".to_string());
            let Ok(p2) = q.clone().swap_turn() else { break };
            q = p2;
        } else {
            let Some(mv) = q.legal_moves().into_iter().find(|x| move_code(x) == code) else { break };
            line.push(mv.to_uci(CastlingMode::Standard).to_string());
            q = after(&q, &mv);
        }
        path = child_path(path, code);
        if line.len() > 16 { break; }
    }
    Ok((v, line))
}

/// frame_leaf(fen) -> (material, composed standing, best capture by the exchange law, their best now) for the side to
/// move, as a frame proof's leaf reads them from the index -- the review hook.
#[pyfunction]
fn frame_leaf(fen: String) -> PyResult<(i64, i64, i64, i64)> {
    let Some(q) = parse(&fen) else { return Ok((0, 0, 0, 0)) };
    let side = q.turn();
    let ix = idx_root(&q, side);
    let best = ix.us.iter().copied().max().unwrap_or(0).max(0) as i64;
    let crude = ix.them.iter().copied().max().unwrap_or(0).max(0) as i64;
    Ok((kernel_material(&q, side), composed_from_index(&q, &ix, side), best, crude))
}
/// THE PROMOTION a side to move can make now, by the exchange law on the promotion square: a queen (8 more than the
/// pawn) plus what it takes, less what they win back on that square; the best over its promotions, never negative.
/// (The leaf's index carries material only; the promotion is the other threat the capture search sees -- 2026-10-11:
/// Kf5 "nets 1" with f7 and f8=Q to come.) The pawns and the squares read are recorded.
fn promo_gain(q: &Chess, by: Color) -> (i64, Option<Move>) {
    if q.turn() != by { return (0, None); }
    let mut best = (0i64, None);
    for m in q.legal_moves() {
        if m.promotion() != Some(Role::Queen) { continue; }
        dep_part(2, shakmaty::Bitboard::from_square(m.to()));                 // (the pawn is a push-source of `to`)
        let q2 = after(q, &m);
        let back = rules::see(q2.board(), m.to(), !by).max(0);
        let g = 8 + m.capture().map(kernel_value).unwrap_or(0) - back;
        if g > best.0 { best = (g, Some(m)); }
    }
    best
}
/// THE COMPOSED STANDING from the index, for the side to move: the best capture the exchange law gives it (or its best
/// promotion), less the intermezzo price -- their best capture or promotion elsewhere after it (crude: their best now;
/// exact: played out, as threat.rs Forcing::judge does); never below standing pat. A lower bound on what a capture
/// search would settle (Lean Ideas.ExchangeLaw.exchangeLawComposed).
pub(crate) fn composed_from_index(q: &Chess, ix: &Idx, side: Color) -> i64 {
    let cap_best = ix.us.iter().copied().max().unwrap_or(0).max(0) as i64;
    let (pro_best, pro_move) = promo_gain(q, side);
    let best = cap_best.max(pro_best);
    if best == 0 { return 0; }
    let crude_cap = ix.them.iter().copied().max().unwrap_or(0).max(0) as i64;
    let crude_pro = q.clone().swap_turn().map(|p| promo_gain(&p, !side).0).unwrap_or(0);
    if crude_cap.max(crude_pro) == 0 { return best; }
    // the move that stands: the cheapest capture on the best square, or the promotion
    let mv = if pro_best > cap_best { pro_move } else {
        let t = (0..64).max_by_key(|x| (ix.us[*x], *x)).map(|x| Square::new(x as u32)).unwrap();
        q.legal_moves().into_iter().filter(|m| m.to() == t).min_by_key(|m| kernel_value(m.role()))
    };
    let Some(mv) = mv else { return 0 };
    let t = mv.to();
    dep_part(2, shakmaty::Bitboard::from_square(t));
    let q2 = after(q, &mv);
    // (a move that gives CHECK leaves them no capture elsewhere but an answer to the check -- the tempo)
    let legal_caps: shakmaty::Bitboard = if q2.is_check() { q2.legal_moves().iter().filter(|m| m.is_capture()).map(|m| m.to()).collect() } else { shakmaty::Bitboard::FULL };
    let them2 = rules::threats_arr_inc(&ix.them, &q2, &mv, !side);
    let px = (0..64usize).filter(|x| *x != t as usize && legal_caps.contains(Square::new(*x as u32))).max_by_key(|x| (them2[*x], *x));
    let price_cap = px.map(|x| them2[x] as i64).unwrap_or(0).max(0);
    if let Some(x) = px { if them2[x] > 0 { dep_part(2, shakmaty::Bitboard::from_square(Square::new(x as u32))); } }
    let price = price_cap.max(promo_gain(&q2, !side).0);
    (best - price).max(0)
}


/// The pursuit at `plies`, then two plies deeper only if that failed -- under a node budget each time; an exhausted
/// budget answers "not certified", which stays sound (2026-10-08: four trapped-piece puzzles certify at six plies, not
/// four; six everywhere cost up to two seconds a claim).
fn deepening_pursuit(p: &Chess, m: &Move, k: i64, focus: &[Square], plies: u32) -> bool {
    for (d, cap) in [(plies, 50_000u64), (plies + 2, 150_000u64)] {
        BUDGET.store(cap, Ordering::Relaxed);
        let ok = pursuit_check_pos(p, m, k, focus, d);
        let spent = BUDGET.load(Ordering::Relaxed) == 0;
        BUDGET.store(u64::MAX, Ordering::Relaxed);
        if ok && !spent { return true; }
    }
    false
}

/// pursuit_check(fen, uci, k, focus squares, plies) -> Option<bool>
#[pyfunction]
fn pursuit_check(fen: String, uci: String, k: i64, focus: Vec<String>, plies: u32) -> PyResult<Option<bool>> {
    let Some(pos) = parse(&fen) else { return Ok(None) };
    let Ok(u) = uci.parse::<shakmaty::uci::UciMove>() else { return Ok(None) };
    let Ok(m) = u.to_move(&pos) else { return Ok(None) };
    let f: Vec<Square> = focus.iter().filter_map(|s| s.parse::<Square>().ok()).collect();
    Ok(Some(pursuit_check_pos(&pos, &m, k, &f, plies)))
}

// THE CERTIFIED VALUE, one search per claim (2026-10-08: trying k = 9, 6, 5, ... one yes/no certificate at a time made
// threat_claims 1.7 s median, 56 s worst): alpha-beta over the same tree as the pursuit -- our side restricted to the
// frame's answers (ordered by the target's escapes left), their side EVERY legal reply -- returning the largest
// material the protagonist can force at the moment the capture sequence ends (the leaf resolved by `settle_d`). Sound
// as a lower bound for the same reason as localCheck_sound: our moves are a subset of the legal ones. A spent budget
// returns the floor (-1000): nothing certified.

fn leaf_value(q: &Chess, side: Color, alpha: i64, beta: i64, ix: Option<&Idx>) -> i64 {
    // the protagonist to move at q: its material plus the resolved captures, searched inside the caller's window
    let mat = kernel_material(q, side);
    let Some(ix) = ix else { return mat + settle_d(q, alpha - mat, beta - mat, true, SETTLE_DEPTH) };
    // THE FRAME LEAF: the material plus the composed standing read from the index; in check, each evasion followed one
    // ply with their standing read from the index after it (the evasions and what they recompute are recorded)
    if q.is_check() {
        dep_add(q.checkers());
        let ms = q.legal_moves();
        if ms.is_empty() { return mat - 1000; }
        let mut best = -10000;
        for e in ms {
            let g = e.capture().map(kernel_value).unwrap_or(0) + if e.is_promotion() { 8 } else { 0 };
            let q2 = after(q, &e);
            let ix2 = idx_after(ix, &q2, &e, side);
            let theirs = Idx { us: ix2.them, them: ix2.us };
            let v = g - composed_from_index(&q2, &theirs, !side);
            if v > best { best = v; }
        }
        return mat + best;
    }
    mat + composed_from_index(q, ix, side)
}

// STORED PROOFS (Kaveh 2026-10-09: "keep track of the frames ... only look at whatever the last move destroyed or
// created, we don't have to search again every time"): while a search runs with HINT_OUT set, every node records the
// move that decided it -- at our node the answer that reached the goal (the cutoff), at theirs the reply that refuted
// it -- keyed by the PATH of moves from the frame's root. A later search on a changed board, with those tables in
// HINT_IN, tries the stored move first at every node. Order never changes an alpha-beta result inside the same window,
// so a re-check gives the same answer as a fresh search (up to the node budget) at the cost of checking the proof.
pub type Hints = HashMap<u64, u32>;
thread_local! {
    static HINT_IN: std::cell::RefCell<Option<std::sync::Arc<Hints>>> = const { std::cell::RefCell::new(None) };
    static HINT_OUT: std::cell::RefCell<Option<Hints>> = const { std::cell::RefCell::new(None) };
}
pub fn move_code(m: &Move) -> u32 {
    let from = m.from().map_or(64u32, |f| f as u32);
    let promo = m.promotion().map_or(0u32, |r| r as u32);
    from | ((m.to() as u32) << 7) | (promo << 14)
}
fn child_path(path: u64, code: u32) -> u64 {
    let mut z = path ^ (code as u64).wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}
const PASS_CODE: u32 = 0x1F_FFFF;
fn hint_at(path: u64) -> Option<u32> { HINT_IN.with(|h| h.borrow().as_ref().and_then(|t| t.get(&path).copied())) }
fn record(path: u64, code: u32) { HINT_OUT.with(|h| if let Some(t) = h.borrow_mut().as_mut() { t.insert(path, code); }) }
/// Run `f` with `hints` as the stored proof and return its result with the proof it recorded.
pub fn with_hints<T>(hints: Option<std::sync::Arc<Hints>>, f: impl FnOnce() -> T) -> (T, Hints) {
    struct Reset;                                                       // the thread's state is cleared even on a panic
    impl Drop for Reset {
        fn drop(&mut self) { HINT_IN.with(|h| *h.borrow_mut() = None); BUDGET.store(u64::MAX, Ordering::Relaxed); }
    }
    let _reset = Reset;
    HINT_IN.with(|h| *h.borrow_mut() = hints);
    HINT_OUT.with(|h| *h.borrow_mut() = Some(Hints::new()));
    let out = f();
    let rec = HINT_OUT.with(|h| h.borrow_mut().take()).unwrap_or_default();
    (out, rec)
}

/// Where a reply leaves a focus square: the piece that stood on `f` moved with `r` (a castling king lands on its own
/// destination, not the rook's square shakmaty's `to()` names -- review 2026-10-09).
fn focus_after(r: &Move, f: Square) -> Square {
    match r {
        Move::Castle { king, rook } if *king == f => {
            let file = if rook.file() > king.file() { shakmaty::File::G } else { shakmaty::File::C };
            Square::from_coords(file, king.rank())
        }
        _ => if r.from() == Some(f) { r.to() } else { f },
    }
}

fn pursue_val(q: &Chess, side: Color, focus: &[Square], plies: u32, alpha: i64, beta: i64) -> i64 {
    let ix = if frame_on() { Some(idx_root(q, side)) } else { None };
    pursue_val_p(q, side, focus, plies, alpha, beta, 0, ix.as_ref())
}
/// `ix`: the index of both sides' frames carried through the search -- a FRAME PROOF (see FRAME_PROOFS); None: the
/// board-reading search.
fn pursue_val_p(q: &Chess, side: Color, focus: &[Square], plies: u32, mut alpha: i64, mut beta: i64, path: u64, ix: Option<&Idx>) -> i64 {
    NODES.fetch_add(1, Ordering::Relaxed);
    let b = BUDGET.load(Ordering::Relaxed);
    if b == 0 { return -1000; }
    if b != u64::MAX { BUDGET.store(b - 1, Ordering::Relaxed); }
    if q.turn() == side {
        // the protagonist stalemated: a draw, no gain (a stalemate is read off the whole board)
        if !q.is_check() && q.legal_moves().is_empty() { if ix.is_some() { dep_global(); } return -1000; }
        let stand = leaf_value(q, side, alpha, beta, ix);
        if plies == 0 || stand >= beta { return stand; }
        let mut best = stand;
        if best > alpha { alpha = best; }
        let them = !side;
        let occ = q.board().occupied();
        // OUR MOVES, IN THE FRAME (frame proofs): a capture outside the zone only when the index gives it a gain (a live
        // frame of ours); a move of an attacked piece outside the zone only when the index says it is attacked (a live
        // frame of theirs); the cornering test on frame pieces only. The zone and every move searched are recorded.
        let z = ix.map(|_| frame_zone(q, side, focus) | kings(q));
        if ix.is_some() { dep_part(0, frame_core(q, focus) | kings(q)); }
        let mut ordered: Vec<(i64, Move)> = q.legal_moves().into_iter().filter(|m2| {
            (m2.is_capture() && match (ix, z) { (Some(ix), Some(z)) => z.contains(m2.to()) || ix.us[m2.to() as usize] > 0, _ => true })
                || m2.role() == Role::King
                || m2.from().map(|f| match (ix, z) {
                    (Some(ix), Some(z)) => (z.contains(f) && q.board().attacks_to(f, them, occ).any()) || ix.them[f as usize] > 0,
                    _ => q.board().attacks_to(f, them, occ).any() }).unwrap_or(false)
                || after(q, m2).is_check()
                || attacks_any(q, m2, focus)
                || (plies >= 2 && { let q2 = after(q, m2); !q2.is_check() && corners_any(&q2, side, z) })
        }).map(|m2| {
            let q2 = after(q, &m2);
            let key = if focus.contains(&m2.to()) { -1 } else {
                focus.iter().filter(|f| q2.board().piece_at(**f).map_or(false, |p| p.color != side))
                    .map(|f| safe_escapes(&q2, side, *f)).min().unwrap_or(50)
            };
            (key, m2)
        }).collect();
        ordered.sort_by_key(|(k, _)| *k);
        if let Some(h) = hint_at(path) {                                // the stored answer first
            if let Some(i) = ordered.iter().position(|(_, m2)| move_code(m2) == h) { let x = ordered.remove(i); ordered.insert(0, x); }
        }
        let mut decided: Option<u32> = None;
        for (_, m2) in ordered {
            let q2 = after(q, &m2);
            let f2: Vec<Square> = focus.iter().filter(|f| q2.board().piece_at(**f).map_or(false, |p| p.color != side)).copied().collect();
            let code = move_code(&m2);
            let ix2 = ix.map(|x| idx_after(x, &q2, &m2, side));
            let v = pursue_val_p(&q2, side, &f2, plies - 1, alpha, beta, child_path(path, code), ix2.as_ref());
            if v > best { best = v; decided = Some(code); if best > alpha { alpha = best; } if alpha >= beta { break; } }
        }
        if let Some(c) = decided { record(path, c); }
        best
    } else {
        let ms = q.legal_moves();
        // no legal reply: mate -- or STALEMATE, a draw, which no claim to win something survives (bug 2026-10-10, found
        // from Kaveh's stalemate lemma: it was scored as keeping the material, i.e. as a win)
        if ms.is_empty() {
            if ix.is_some() { if q.is_check() { dep_add(q.checkers()); } else { dep_global(); } }
            return if q.is_check() { 1000 } else { -1000 };
        }
        // THEIR REPLIES, KEPT IN THE FRAME (Kaveh 2026-10-08: "use lemmas for the opponent reply too, and keep it in the
        // frame"): with a focus, only the replies that touch the frame are searched -- the target's moves, captures,
        // checks, and moves from, to or attacking a frame square (the target, its escapes, our pieces bearing on them,
        // the lines between); every other reply is ONE "pass" (the FRAME-INDEPENDENCE lemma, Lean
        // Ideas.FrameLemma.frame_unchanged, 2026-10-11: a move that touches no square bearing on the frame -- on a line
        // through a frame square, a knight's move from one, or the square itself -- leaves the frame's pieces and every
        // attack onto them as they were). In check every evasion is searched. The board-reading search (EXACT, the
        // default since the 2026-10-09 audit found this pass certifying 10.9% false claims) searches every reply and
        // reads the whole board at its leaf. The FRAME PROOF keeps the pass, puts both kings in the zone, takes the
        // captures outside it from the index (their live frames) and the leaf from the index too; its soundness is
        // "modulo the pass" -- their preparatory moves outside the frame -- and is measured by the frame audit.
        let frame = ix.is_some();
        let mut zone = if focus.is_empty() || q.is_check() || (EXACT.load(Ordering::Relaxed) && !frame) { None }
                   else { Some(frame_zone(q, side, focus) | if frame { kings(q) } else { shakmaty::Bitboard::EMPTY }) };
        if frame { if zone.is_some() { dep_part(0, frame_core(q, focus) | kings(q)); } else { dep_part(0, q.checkers() | kings(q)); } }
        let mut worst = 2000;
        let mut passed_once = false;
        let mut ms: Vec<Move> = ms.into_iter().collect();
        if let Some(h) = hint_at(path) {                                // the stored refutation first
            if let Some(i) = ms.iter().position(|r| move_code(r) == h) {
                let x = ms.remove(i);
                // THE CERTIFICATE CHECK: their reply fixed to the stored refutation, nothing else searched at this node
                if CERT.with(|c| c.get()) { ms.clear(); zone = None; }
                ms.insert(0, x);
            }
        }
        let mut decided: Option<u32> = None;
        for r in ms.iter() {
            if let Some(z) = zone {
                let q2p = after(q, r);
                // a capture is a live frame of theirs when the index gives it a gain (frame proofs); every capture otherwise
                let live = match ix { Some(ix) => r.is_capture() && ix.them[r.to() as usize] > 0, None => r.is_capture() };
                let touches = live || q2p.is_check() || (frame && r.is_promotion())
                    || r.from().map_or(false, |f| z.contains(f)) || z.contains(r.to())
                    || q2p.board().piece_at(r.to()).map_or(false, |p| (shakmaty::attacks::attacks(r.to(), p, q2p.board().occupied()) & z).any());
                if !touches {
                    if passed_once { continue; }                       // all the outside moves are one pass
                    passed_once = true;
                    let Ok(pass) = q.clone().swap_turn() else { continue };
                    let v = if plies == 0 { leaf_value(&pass, side, alpha, beta, ix) } else { pursue_val_p(&pass, side, focus, plies - 1, alpha, beta, child_path(path, PASS_CODE), ix) };
                    if v < worst { worst = v; decided = Some(PASS_CODE); if worst < beta { beta = worst; } if alpha >= beta { break; } }
                    continue;
                }
            }
            let q2 = after(q, r);
            let code = move_code(r);
            let ix2 = ix.map(|x| idx_after(x, &q2, r, side));
            let v = if q2.is_checkmate() { if frame { dep_add(q2.checkers()); } -1000 } else if plies == 0 { leaf_value(&q2, side, alpha, beta, ix2.as_ref()) } else {
                let f2: Vec<Square> = focus.iter().map(|f| focus_after(r, *f)).collect();
                pursue_val_p(&q2, side, &f2, plies - 1, alpha, beta, child_path(path, code), ix2.as_ref())
            };
            if v < worst { worst = v; decided = Some(code); if worst < beta { beta = worst; } if alpha >= beta { break; } }
        }
        if let Some(c) = decided { record(path, c); }
        worst
    }
}

/// The frame's CORE: each target and the squares it can move to (what a proof about it reads directly; whoever bears on
/// them is read through the attacks onto them -- plan.rs inputs_differ).
fn frame_core(q: &Chess, focus: &[Square]) -> shakmaty::Bitboard {
    let b = q.board();
    let mut z = shakmaty::Bitboard::EMPTY;
    for &f in focus {
        z.add(f);
        if let Some(p) = b.piece_at(f) { z |= shakmaty::attacks::attacks(f, p, b.occupied()); }
    }
    z
}

/// The frame's squares around the focus pieces: each target, the squares it can move to, our pieces bearing on those,
/// and the squares between our sliders and them.
fn frame_zone(q: &Chess, us: Color, focus: &[Square]) -> shakmaty::Bitboard {
    let b = q.board();
    let occ = b.occupied();
    let mut z = shakmaty::Bitboard::EMPTY;
    for &f in focus {
        let Some(p) = b.piece_at(f) else { continue };
        let reach = shakmaty::attacks::attacks(f, p, occ);
        z |= reach; z.add(f);
        for t in reach.with(f) {
            for a in b.attacks_to(t, us, occ) {
                z.add(a);
                z |= shakmaty::attacks::between(a, t);
            }
        }
    }
    z
}

/// PROVE: does m certify a gain of at least `goal` (relative to the material now)? A null-window search at the goal --
/// at our nodes the first answer reaching it ends the node, at theirs the first refutation -- so the recorded moves
/// are exactly a proof or a refutation. None: the budget ran out (nothing certified).
pub fn prove_pos(p: &Chess, m: &Move, focus: &[Square], plies: u32, budget: u64, goal: i64) -> Option<bool> {
    let c = p.turn();
    let target = kernel_material(p, c) + goal;
    let q = after(p, m);
    if q.is_checkmate() { return Some(true); }
    BUDGET.store(budget, Ordering::Relaxed);
    let ix = if frame_on() { Some(idx_root(&q, c)) } else { None };
    let v = pursue_val_p(&q, c, focus, plies, target - 1, target, child_path(0, move_code(m)), ix.as_ref());
    let left = BUDGET.load(Ordering::Relaxed);
    let spent = left == 0;
    BUDGET.store(u64::MAX, Ordering::Relaxed);
    let used = budget - left;
    let k = if spent { 2 } else if v >= target { 0 } else { 1 };
    PROVE_STATS[k * 2].fetch_add(1, Ordering::Relaxed);
    PROVE_STATS[k * 2 + 1].fetch_add(used, Ordering::Relaxed);
    PROVE_NODES.with(|c| c.set(c.get() + used));
    if spent { None } else { Some(v >= target) }
}

thread_local! { pub static PROVE_NODES: std::cell::Cell<u64> = const { std::cell::Cell::new(0) }; }

/// [proved, refuted, budget spent] x [count, nodes] since the last call (where the proof work goes)
static PROVE_STATS: [AtomicU64; 6] = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];
#[pyfunction]
fn prove_stats() -> Vec<u64> { PROVE_STATS.iter().map(|a| a.swap(0, Ordering::Relaxed)).collect() }

/// The certified gain of m (relative to the material now) with the given focus, plies and node budget; None if the
/// budget ran out (nothing certified).
pub(crate) fn certified_value(p: &Chess, m: &Move, focus: &[Square], plies: u32, budget: u64) -> Option<i64> {
    let c = p.turn();
    let base = kernel_material(p, c);
    let q = after(p, m);
    if q.is_checkmate() { return Some(1000); }
    BUDGET.store(budget, Ordering::Relaxed);
    let v = pursue_val(&q, c, focus, plies, -2000, 2000);
    let spent = BUDGET.load(Ordering::Relaxed) == 0;
    BUDGET.store(u64::MAX, Ordering::Relaxed);
    let out = if spent { None } else { Some(v - base) };
    // THE FRAME AUDIT on values: the same value by the exact search; [3] equal, [4] the frame value LARGER (an
    // overclaim), [5] smaller (counted only when both finished)
    if plan::FRAME_AUDIT.load(Ordering::Relaxed) && frame_on() {
        if let Some(fv) = out {
            if let Some(xv) = with_frame_off(|| certified_value(p, m, focus, plies, budget)) {
                plan::frame_audit_count(if fv == xv { 3 } else if fv > xv { 4 } else { 5 });
            }
        }
    }
    out
}

// TARGET-FIRST REASONING, the primitives (Kaveh 2026-10-08: "start from the highest-value goal -- can I do anything
// against the king? no -- but I can against the queen"; "the queen is so oddly placed, almost trapped ... a prior to
// recognize that"; "either create a bigger threat at the same time, or keep chasing it until it cannot escape").

/// target_profile(fen, square) -> (value, safe escapes, our attackers, their defenders, on the edge): the side to move
/// is "us"; the piece on the square is theirs.
#[pyfunction]
fn target_profile(fen: String, sq: String) -> PyResult<Option<(i64, i64, i64, i64, bool)>> {
    let Some(pos) = parse(&fen) else { return Ok(None) };
    let Ok(t) = sq.parse::<Square>() else { return Ok(None) };
    let us = pos.turn();
    let b = pos.board();
    let Some(p) = b.piece_at(t) else { return Ok(None) };
    if p.color == us { return Ok(None); }
    let occ = b.occupied();
    let edge = matches!(t.file(), shakmaty::File::A | shakmaty::File::H) || matches!(t.rank(), shakmaty::Rank::First | shakmaty::Rank::Eighth);
    let v = if p.role == Role::King { 100 } else { kernel_value(p.role) };
    Ok(Some((v, safe_escapes(&pos, us, t), b.attacks_to(t, us, occ).count() as i64,
             b.attacks_to(t, p.color, occ.without(t)).count() as i64, edge)))
}

/// target_moves(fen, square) -> [(uci, safe escapes left, attacks it, gives check)]: OUR moves that matter to the target
/// -- they take it, or strictly increase our coverage of its frame (the target and the squares it can reach), counted
/// on the whole board after the move, so a CLEARANCE (Bb7 opening the rank for the rook) counts as well as a direct
/// attack. Ordered: captures of it, then the fewest safe escapes left.
#[pyfunction]
fn target_moves(fen: String, sq: String) -> PyResult<Vec<(String, i64, bool, bool)>> {
    let Some(pos) = parse(&fen) else { return Ok(vec![]) };
    let Ok(t) = sq.parse::<Square>() else { return Ok(vec![]) };
    let us = pos.turn();
    let Some(tp) = pos.board().piece_at(t) else { return Ok(vec![]) };
    let frame = |q: &Chess| -> u32 {
        let b = q.board();
        let reach = shakmaty::attacks::attacks(t, tp, b.occupied()).with(t);
        reach.into_iter().filter(|s| b.attacks_to(*s, us, b.occupied()).any()).count() as u32
    };
    let before = frame(&pos);
    let att = |q: &Chess| q.board().attacks_to(t, us, q.board().occupied()).count();
    let att0 = att(&pos);
    let esc0 = safe_escapes(&pos, us, t);
    let mut out = Vec::new();
    for m in pos.legal_moves() {
        let q = after(&pos, &m);
        let takes = m.to() == t;
        if !takes && q.board().piece_at(t).map_or(true, |p| p.color == us) { continue; }
        // a move matters if it ADDS AN ATTACKER on the target, TAKES AWAY a safe escape, or widens our coverage of its
        // frame (bug 2026-10-08: coverage alone missed 44 of 52 misses -- a bishop stepping from h6 to g5 to hit the
        // queen gives up the squares it covered from h6, so the count did not rise)
        if !takes && att(&q) <= att0 && safe_escapes(&q, us, t) >= esc0 && frame(&q) <= before { continue; }
        let attacks = takes || q.board().attacks_to(t, us, q.board().occupied()).any();
        let e = if takes { -1 } else { safe_escapes(&q, us, t) };
        out.push((m.to_uci(CastlingMode::Standard).to_string(), e, attacks, q.is_check()));
    }
    out.sort_by_key(|x| x.1);
    Ok(out)
}

// ONE POOL OF FACTS FOR BOTH SIDES (Kaveh 2026-10-09: "one common pool of facts and reasons and lemmas that apply both
// to us and to them -- not two parallel independent runs; there are interactions between the reasons we find for us
// and the reasons we find for them"). The pool holds, for each side, its CLAIMS (what each of its moves creates,
// certified or matched -- ours on the real position, theirs on the null-move position: their reasons were we to pass),
// each with its FRAME (the squares it uses: its move's squares and its targets); and the INTERACTIONS between our
// claims and theirs, derived from shared frames:
//   answers     our move touches their claim's frame -- it moves or takes a piece of it, lands on or blocks one of its
//               squares, or makes their claimed move illegal (a capture of the attacker, a bigger threat with check)
//   shares      a square in both frames: a piece that is our target and their attacker, or theirs and ours (overload)

fn claim_frame(pos: &Chess, m: &Move, targets: &[String]) -> shakmaty::Bitboard {
    let mut z = shakmaty::Bitboard::EMPTY;
    if let Some(f) = m.from() { z.add(f); }
    z.add(m.to());
    for t in targets.iter().filter_map(|s| s.parse::<Square>().ok()) {
        z.add(t);
        z |= shakmaty::attacks::between(m.to(), t);
    }
    let _ = pos;
    z
}

/// fact_pool(fen) -> (our claims, their claims, interactions): claims (index, uci, name, k, certified, frame squares);
/// interactions (our index, their index, kind).
#[pyfunction]
fn fact_pool(fen: String) -> PyResult<(Vec<(usize, String, String, i64, bool, Vec<String>)>, Vec<(usize, String, String, i64, bool, Vec<String>)>, Vec<(usize, usize, String)>)> {
    let Some(pos) = parse(&fen) else { return Ok((vec![], vec![], vec![])) };
    let ours_raw = threat_claims(fen.clone())?;
    let passed = pos.clone().swap_turn().ok();
    let theirs_raw = match &passed {
        Some(p) => threat_claims(Fen::from_position(p.clone(), shakmaty::EnPassantMode::Legal).to_string())?,
        None => vec![],
    };
    let mk = |pp: &Chess, rows: Vec<(String, String, i64, bool, Vec<String>, bool)>| -> Vec<(usize, String, String, i64, bool, Vec<String>, shakmaty::Bitboard)> {
        rows.into_iter().enumerate().filter_map(|(i, (u, n, k, c, t, fr))| {
            let m = u.parse::<shakmaty::uci::UciMove>().ok()?.to_move(pp).ok()?;
            let z = claim_frame(pp, &m, &t);
            Some((i, u, n, k, c || fr, z.into_iter().map(|s| s.to_string()).collect(), z))
        }).collect()
    };
    let ours = mk(&pos, ours_raw);
    let theirs = match &passed { Some(p) => mk(p, theirs_raw), None => vec![] };
    let mut inter = Vec::new();
    for (i, u, _, _, _, _, zo) in ours.iter() {
        let Ok(m) = u.parse::<shakmaty::uci::UciMove>().and_then(|x| Ok(x)) else { continue };
        let Ok(mm) = m.to_move(&pos) else { continue };
        let after_m = after(&pos, &mm);
        for (j, ut, _, _, _, _, zt) in theirs.iter() {
            let touches = mm.from().map_or(false, |f| zt.contains(f)) || zt.contains(mm.to());
            let illegal = ut.parse::<shakmaty::uci::UciMove>().ok().and_then(|x| x.to_move(&after_m).ok()).is_none();
            if touches || illegal || after_m.is_check() { inter.push((*i, *j, "answers".to_string())); }
            if (*zo & *zt).any() { inter.push((*i, *j, "shares".to_string())); }
        }
    }
    let strip = |v: Vec<(usize, String, String, i64, bool, Vec<String>, shakmaty::Bitboard)>| v.into_iter().map(|(a, b, c, d, e, f, _)| (a, b, c, d, e, f)).collect();
    Ok((strip(ours), strip(theirs), inter))
}

/// maia_policy(fen, model_path, temp) -> [(uci, probability)] -- Maia's move probabilities, in Rust (src/maia.rs).
#[pyfunction]
fn maia_policy(fen: String, model_path: String, temp: f32) -> PyResult<Vec<(String, f32)>> {
    let Some(pos) = parse(&fen) else { return Ok(vec![]) };
    maia::policy(&pos, &model_path, temp).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))
}

/// frame_outs(fen, our uci, maia model path, temp, top, clock) -> [(reply uci, probability, outcome)]: THE OUTS OF A
/// FRAME (Kaveh 2026-10-09): after our move, each reply of the opponent with Maia's probability for it (the model is
/// the opponent's rating band) and its OUTCOME for us -- the material the reply takes, net of the largest gain we can
/// then CERTIFY (threat_claims' settled certificates; a mate counts 50, being mated -50). `clock` is the opponent's
/// remaining time, accepted and not used yet (the slot for a time-conditioned policy).
#[pyfunction]
#[pyo3(signature = (fen, uci, model_path, temp=1.0, top=12, clock=None))]
fn frame_outs(fen: String, uci: String, model_path: String, temp: f32, top: usize, clock: Option<f32>) -> PyResult<Vec<(String, f32, i64)>> {
    let _ = clock;
    let Some(pos) = parse(&fen) else { return Ok(vec![]) };
    let Ok(u) = uci.parse::<shakmaty::uci::UciMove>() else { return Ok(vec![]) };
    let Ok(m) = u.to_move(&pos) else { return Ok(vec![]) };
    let us = pos.turn();
    let q = after(&pos, &m);
    if q.is_game_over() { return Ok(vec![]); }
    let mut pol = maia::policy(&q, &model_path, temp).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
    pol.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let mut out = Vec::new();
    for (ru, p) in pol.into_iter().take(top) {
        let Ok(r) = ru.parse::<shakmaty::uci::UciMove>().and_then(|x| Ok(x)) else { continue };
        let Ok(rm) = r.to_move(&q) else { continue };
        let taken = rm.capture().map(kernel_value).unwrap_or(0);
        let q2 = after(&q, &rm);
        let outcome = if q2.is_checkmate() { -50 } else {
            let f = Fen::from_position(q2.clone(), shakmaty::EnPassantMode::Legal).to_string();
            let best = threat_claims(f)?.into_iter().filter(|c| c.5).map(|c| c.2.min(50)).max().unwrap_or(0);
            best - taken
        };
        let _ = us;
        out.push((ru, p, outcome));
    }
    Ok(out)
}

// THE FRAME AS A TREE OF BRANCHES (Kaveh 2026-10-09: "it's not about plies but about branches and the conditioning:
// every time we reach a branching point, assign a probability to each branch and evaluate the lemmas and the frame
// conditioned on that branch -- if they push the pawn in front of the king there is no more mate, no more forcing, and
// the rook just runs away"). After our move, each reply of theirs is a BRANCH with Maia's probability (the replies
// covering `mass` of Maia's policy) or, for a perfect opponent (`model_path` empty: everyone above 1900), the single
// branch worst for us. On each branch the certified lemmas of the new position decide: no certified claim -> the
// frame has dissolved, the branch ends with the material as it stands; a mate claim -> MATE (150); another claim -> we
// play its move and the next branching follows. Outcome per path: material change (ours - theirs) or 150 for mate.

const MATE_V: f32 = 150.0;

static CLAIMS: OnceLock<Mutex<HashMap<String, Option<(String, i64, String)>>>> = OnceLock::new();

/// The best certified claim at q, CACHED by position (2026-10-09: one tree spent 31 s, nearly all in repeated claim
/// computations -- the same positions recur across branches and across rating bands).
fn best_claim(q: &Chess) -> PyResult<Option<(String, i64, String)>> {
    let f = Fen::from_position(q.clone(), shakmaty::EnPassantMode::Legal).to_string();
    let key = format!("{}|{}", f, FAST.load(Ordering::Relaxed));        // the tiers differ by mode
    let cache = CLAIMS.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(hit) = cache.lock().unwrap().get(&key) { return Ok(hit.clone()); }
    let rows = threat_claims(f)?;
    let best = rows.into_iter().filter(|c| c.5 && c.2 >= 1).max_by_key(|c| c.2).map(|c| (c.0, c.2, c.1));
    cache.lock().unwrap().insert(key, best.clone());
    Ok(best)
}

fn branches(q: &Chess, model_path: &str, mass: f32) -> Result<Vec<(Move, f32)>, String> {
    let pol = maia::policy(q, model_path, 1.0)?;
    let mut sorted = pol; sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let mut out = Vec::new(); let mut acc = 0.0;
    for (u, p) in sorted {
        let Ok(m) = u.parse::<shakmaty::uci::UciMove>().and_then(|x| Ok(x)) else { continue };
        let Ok(m) = m.to_move(q) else { continue };
        out.push((m, p)); acc += p;
        if acc >= mass { break; }
    }
    let z: f32 = out.iter().map(|x| x.1).sum();
    Ok(out.into_iter().map(|(m, p)| (m, p / z)).collect())
}

/// Their branching point at q (they to move), material so far `mat` (ours - theirs, since our first move), `path` so
/// far; returns [(probability, outcome, path)].
fn their_node(q: &Chess, us: Color, model_path: &str, mass: f32, mat: f32, path: Vec<String>, depth: u32) -> PyResult<Vec<(f32, f32, Vec<String>)>> {
    let ms = q.legal_moves();
    if ms.is_empty() { return Ok(vec![(1.0, if q.is_check() { MATE_V } else { 0.0 }, path)]); }   // stalemate: a draw
    let perfect = model_path.starts_with("perfect:");
    let mut rows: Vec<(f32, f32, Vec<String>)> = Vec::new();
    let cand: Vec<(Move, f32)> = if perfect {
        // "perfect:<maia-1900 path>": the strongest Maia's plausible replies plus every forcing reply (capture or
        // check), the worst for us taken (2026-10-09: all legal replies at every node took 34-154 s a tree)
        let base = branches(q, &model_path["perfect:".len()..], mass).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
        let mut set: Vec<Move> = base.into_iter().map(|(m, _)| m).collect();
        for m in ms.iter() { if (m.is_capture() || after(q, m).is_check()) && !set.contains(m) { set.push(m.clone()); } }
        set.into_iter().map(|m| (m, 1.0)).collect()
    } else { branches(q, model_path, mass).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))? };
    let mut per_branch: Vec<(f32, Vec<(f32, f32, Vec<String>)>)> = Vec::new();
    for (r, p) in cand {
        let taken = r.capture().map(kernel_value).unwrap_or(0) as f32;
        let q2 = after(q, &r);
        let mut p2 = path.clone(); p2.push(r.to_uci(CastlingMode::Standard).to_string());
        let sub = if q2.is_checkmate() { vec![(1.0, -MATE_V, p2)] } else {
            our_node(&q2, us, model_path, mass, mat - taken, p2, depth)?
        };
        per_branch.push((p, sub));
    }
    if perfect {
        // the perfect opponent takes the branch whose expected outcome is worst for us
        let ev = |sub: &Vec<(f32, f32, Vec<String>)>| sub.iter().map(|x| x.0 * x.1).sum::<f32>();
        let worst = per_branch.into_iter().min_by(|a, b| ev(&a.1).partial_cmp(&ev(&b.1)).unwrap()).unwrap();
        return Ok(worst.1);
    }
    for (p, sub) in per_branch { for (pp, v, pa) in sub { rows.push((p * pp, v, pa)); } }
    Ok(rows)
}

/// Our turn at q: the certified lemmas decide -- no claim: the frame has dissolved; a mate claim: MATE; otherwise we
/// play the claim's move and branch again (until `depth` runs out: then the claim's certified gain stands).
fn our_node(q: &Chess, us: Color, model_path: &str, mass: f32, mat: f32, path: Vec<String>, depth: u32) -> PyResult<Vec<(f32, f32, Vec<String>)>> {
    if q.is_game_over() { return Ok(vec![(1.0, if q.is_checkmate() { -MATE_V } else { mat }, path)]); }
    // no lemma applies: the frame has dissolved -- but the remaining CAPTURES are resolved (either side free to stop),
    // not left as raw material (bug 2026-10-09: ...Rxd5 counted -3 with Nxd5 winning the rook back)
    BUDGET.store(200_000, Ordering::Relaxed);
    let cash = settle_d(q, -2000, 2000, true, SETTLE_DEPTH) as f32;    // what the captures on the board are worth now
    BUDGET.store(u64::MAX, Ordering::Relaxed);
    let dissolved = |path: Vec<String>| -> Vec<(f32, f32, Vec<String>)> { vec![(1.0, mat + cash, path)] };
    // CASH IN FIRST (bug 2026-10-09, the Ke2 line: after Nf5 (fork +3) Nxf5 the recapture creates no new threat, so it
    // is never a claim -- the tree chose a new +1 claim and never took the +3 it was promised): a claim is followed
    // only when its certified gain beats resolving the captures already on the board
    let Some((u, k, name)) = best_claim(q)? else { return Ok(dissolved(path)) };
    if (k as f32) <= cash && k < 50 { let mut p = path; p.push(format!("(captures resolved {cash:+})")); return Ok(dissolved(p)); }
    if k >= 50 { let mut p = path; p.push(format!("{u} ({name}: mate)")); return Ok(vec![(1.0, MATE_V, p)]); }
    let Ok(m) = u.parse::<shakmaty::uci::UciMove>().and_then(|x| Ok(x)) else { return Ok(dissolved(path)) };
    let Ok(m) = m.to_move(q) else { return Ok(dissolved(path)) };
    let gained = m.capture().map(kernel_value).unwrap_or(0) as f32;
    let mut p = path; p.push(format!("{} ({name} +{k})", m.to_uci(CastlingMode::Standard)));
    if depth == 0 { return Ok(vec![(1.0, mat + k as f32, p)]); }
    let q2 = after(q, &m);
    their_node(&q2, us, model_path, mass, mat + gained, p, depth - 1)
}

/// frame_tree(fen, our uci, maia model path ("" = a perfect opponent), mass, depth) -> [(probability, outcome, path)]
#[pyfunction]
#[pyo3(signature = (fen, uci, model_path, mass=0.95, depth=3))]
fn frame_tree(fen: String, uci: String, model_path: String, mass: f32, depth: u32) -> PyResult<Vec<(f32, f32, Vec<String>)>> {
    let Some(pos) = parse(&fen) else { return Ok(vec![]) };
    let Ok(u) = uci.parse::<shakmaty::uci::UciMove>() else { return Ok(vec![]) };
    let Ok(m) = u.to_move(&pos) else { return Ok(vec![]) };
    let us = pos.turn();
    let gained = m.capture().map(kernel_value).unwrap_or(0) as f32;
    let q = after(&pos, &m);
    if q.is_checkmate() { return Ok(vec![(1.0, MATE_V, vec![uci])]); }
    FAST.store(true, Ordering::Relaxed);                              // inside the tree: the cheap tiers only
    let r = their_node(&q, us, &model_path, mass, gained, vec![uci], depth);
    FAST.store(false, Ordering::Relaxed);
    r
}

/// best_reply(fen, uci, focus, plies, budget) -> [(their reply, the value it holds us to)], best for them first: WHY a
/// claim is not certified -- the refutation, to be named (Kaveh 2026-10-08: "Ba6 would not have worked because the
/// rook would be hanging").
#[pyfunction]
fn best_reply(fen: String, uci: String, focus: Vec<String>, plies: u32, budget: u64) -> PyResult<Vec<(String, i64)>> {
    let Some(pos) = parse(&fen) else { return Ok(vec![]) };
    let Ok(u) = uci.parse::<shakmaty::uci::UciMove>() else { return Ok(vec![]) };
    let Ok(m) = u.to_move(&pos) else { return Ok(vec![]) };
    let f: Vec<Square> = focus.iter().filter_map(|s| s.parse::<Square>().ok()).collect();
    let c = pos.turn();
    let base = kernel_material(&pos, c);
    let q = after(&pos, &m);
    let mut out = Vec::new();
    for r in q.legal_moves() {
        let q2 = after(&q, &r);
        BUDGET.store(budget, Ordering::Relaxed);
        let v = if q2.is_checkmate() { -1000 } else {
            let f2: Vec<Square> = f.iter().map(|x| focus_after(&r, *x)).collect();
            pursue_val(&q2, c, &f2, plies.saturating_sub(1), -2000, 2000)
        };
        BUDGET.store(u64::MAX, Ordering::Relaxed);
        out.push((r.to_uci(CastlingMode::Standard).to_string(), v - base));
    }
    out.sort_by_key(|(_, v)| *v);
    out.truncate(3);
    Ok(out)
}

/// certified_gain(fen, uci, focus, plies, budget) -> Option<i64>
#[pyfunction]
fn certified_gain(fen: String, uci: String, focus: Vec<String>, plies: u32, budget: u64) -> PyResult<Option<i64>> {
    let Some(pos) = parse(&fen) else { return Ok(None) };
    let Ok(u) = uci.parse::<shakmaty::uci::UciMove>() else { return Ok(None) };
    let Ok(m) = u.to_move(&pos) else { return Ok(None) };
    let f: Vec<Square> = focus.iter().filter_map(|s| s.parse::<Square>().ok()).collect();
    Ok(certified_value(&pos, &m, &f, plies, budget))
}

/// local_check_frames(fen, uci, k) -> Option<bool>: the frame-sized certificate.
#[pyfunction]
fn local_check_frames(fen: String, uci: String, k: i64) -> PyResult<Option<bool>> {
    let Some(pos) = parse(&fen) else { return Ok(None) };
    let Ok(u) = uci.parse::<shakmaty::uci::UciMove>() else { return Ok(None) };
    let Ok(m) = u.to_move(&pos) else { return Ok(None) };
    Ok(Some(local_check_frames_pos(&pos, &m, k)))
}

/// local_check(fen, uci, k) -> Option<bool> (None: bad FEN or an illegal move)
#[pyfunction]
fn local_check(fen: String, uci: String, k: i64) -> PyResult<Option<bool>> {
    let Some(pos) = parse(&fen) else { return Ok(None) };
    let Ok(u) = uci.parse::<shakmaty::uci::UciMove>() else { return Ok(None) };
    let Ok(m) = u.to_move(&pos) else { return Ok(None) };
    Ok(Some(local_check_pos(&pos, &m, k)))
}

fn mate_in_one(pos: &Chess) -> bool {
    pos.legal_moves().iter().any(|m| { let p = after(pos, m); p.is_checkmate() })
}

/// threat_claims(fen) -> [(uci, name, k, certified, targets)] for every legal move of the side to move: what the move
/// CREATED by the rule layer (rules.rs), named (fork / fork with check / cornered piece / mate threat / promotion) and
/// decided by the kernel's certificate (`local_check_pos`, mirrored from Fork.lean, bridge-tested against Lean).
#[pyfunction]
fn threat_claims(fen: String) -> PyResult<Vec<(String, String, i64, bool, Vec<String>, bool)>> {
    let Some(pos) = parse(&fen) else { return Ok(vec![]) };
    let us = pos.turn();
    let before = rules::threats_of(&pos, us);
    let mate_before = match pos.clone().swap_turn() { Ok(_) => mate_in_one(&pos), Err(_) => false };
    let corner_before: std::collections::HashSet<u8> = match pos.clone().swap_turn() {
        Ok(p) => rules::cornered(&p, us).into_iter().map(|(t, _)| t).collect(),
        Err(_) => Default::default(),
    };
    let trapped_before: std::collections::HashSet<u8> = match pos.clone().swap_turn() {
        Ok(p) => rules::trapped_unattacked(&p, us).into_iter().map(|(t, _)| t).collect(),
        Err(_) => Default::default(),
    };
    let mut out = Vec::new();
    // the pursuit's depth grows as the board empties (2026-10-08, puzzle 07bjx: a king hunting a trapped knight for
    // seven plies): with ten pieces or fewer the tree is small enough for eight plies
    let plies = if pos.board().occupied().count() <= 10 { 8 } else { PURSUIT_PLIES };
    for m in pos.legal_moves() {
        let q = after(&pos, &m);
        if q.is_game_over() { continue; }
        let now = rules::threats_of(&q, us);
        let mut new: Vec<(u8, i64)> = now.iter().filter(|(t, g)| before.get(t).map_or(true, |g0| **g > *g0)).map(|(t, g)| (*t, *g)).collect();
        new.sort_by(|a, b| b.1.cmp(&a.1));
        let uci = m.to_uci(CastlingMode::Standard).to_string();
        let names: Vec<String> = new.iter().map(|(t, _)| rules::square_name(*t)).collect();
        // THE FORCING CHAIN (2026-10-08, puzzle 06xqS: Qe2+ Be7 Bb4 traps the queen): a check with no claim of its own,
        // pursued against their most valuable pieces -- every reply to the check leaves us a winning pursuit
        // ...and a CAPTURE opens chains the same way (2026-10-08, puzzles 0GUFM / 0ohjJ: Nxd5 Qxd5 Nd4 uncovers the bishop
        // on the cornered queen; Rxh4 Qxh4 Bg5 traps it): forcing moves -- checks and captures -- are pursued
        // (a capture that already wins by the rules' static gain is valued by the frame; chains matter for sacrifices
        // and even trades -- 2026-10-08: chains after every capture made the claims too slow)
        let winning_capture = m.is_capture() && rules::threats_of(&pos, us).get(&(m.to() as u8)).map_or(false, |g| *g > 0);
        if q.is_check() || (m.is_capture() && !winning_capture) {
            let mut big: Vec<(Square, i64)> = q.board().by_color(!us).into_iter()
                .filter_map(|sq| q.board().piece_at(sq).map(|p| (sq, kernel_value(p.role)))).filter(|(_, v)| *v >= 5).collect();
            big.sort_by(|a, b| b.1.cmp(&a.1));
            if let Some(&(t, v)) = big.first() {
                let focus = [t];
                let _ = v;
                let fast = FAST.load(Ordering::Relaxed);
                let mut got = certified_value(&pos, &m, &focus, 2, 10_000).filter(|x| *x >= 1);
                if got.is_none() { got = certified_value(&pos, &m, &focus, plies, if fast { 50_000 } else { 200_000 }).filter(|x| *x >= 1); }
                if got.is_none() && !fast { got = certified_value(&pos, &m, &focus, plies + 2, 1_500_000).filter(|x| *x >= 1); }
                if let Some(x) = got {
                    out.push((uci.clone(), "forcing chain".to_string(), x.min(1000), false, vec![t.to_string()], true));
                }
            }
        }
        let mut claim = |name: &str, k: i64, targets: Vec<String>| {
            // ONE SEARCH PER CLAIM: the certified value (the largest gain that holds), first at the frame's own depth
            // (two plies with no focus, `plies` with one), then two plies deeper if nothing was certified; the raw
            // three-ply kernel certificate (Lean-proved) only at the claimed k, for the 'C' flag.
            let focus: Vec<Square> = targets.iter().filter_map(|s| s.parse::<Square>().ok()).collect();
            let d = if focus.is_empty() { 2 } else { plies };
            // TIERED (2026-10-09: 95% of the nodes went to claims that never certify): a 2-ply pre-check first (55
            // of 60 certified claims pass it), then the frame's own depth, then two plies deeper -- the deep tier only
            // at the root (FAST off); inside a tree the pre-check and the middle tier
            let fast = FAST.load(Ordering::Relaxed);
            let mut got = certified_value(&pos, &m, &focus, 2, 10_000).filter(|v| *v >= 1);
            if got.is_none() { got = certified_value(&pos, &m, &focus, d, if fast { 50_000 } else { 200_000 }).filter(|v| *v >= 1); }
            if got.is_none() && !focus.is_empty() && !fast {
                got = certified_value(&pos, &m, &focus, d + 2, 1_500_000).filter(|v| *v >= 1);
            }
            let cert = local_check_pos(&pos, &m, k);
            match got {
                Some(v) => out.push((uci.clone(), name.to_string(), v.min(1000), cert, targets, true)),
                None => out.push((uci.clone(), name.to_string(), k, cert, targets, false)),
            }
        };
        let king_sq = q.board().king_of(!us).map(|k| k as u8);
        for (t, g) in rules::cornered(&q, us) {                         // THE CORNERING FRAME, one rule for every piece:
            if Some(t) == king_sq {                                     // the king cornered is a check with no flight --
                claim("cornered king", g, vec![rules::square_name(t)]); // mate if nothing blocks or takes (certificate)
            } else if !corner_before.contains(&t) {                     // a piece newly with no escape
                claim("cornered piece", g, vec![rules::square_name(t)]);
            }
        }
        if q.is_check() && !rules::cornered(&q, us).iter().any(|(t, _)| Some(*t) == king_sq)
            && local_check_frames_pos(&pos, &m, 50) {                   // a check that still forces mate (the king has a
            claim("cornered king", 50, vec![]);                         // flight now, but every line mates): cornered later
        }
        if !q.is_check() {
            for (t, v) in rules::trapped_unattacked(&q, us) {           // no safe move yet, not yet attacked: the trap
                // only when THIS move changed something about the target (2026-10-09: the same trapped knight was
                // claimed on 26 of 28 moves, each burning the full budget): newly trapped, or the move touches its frame
                let tsq = Square::new(t as u32);
                let touches = m.to() == tsq || attacks_any(&pos, &m, &[tsq]) || !trapped_before.contains(&t);
                if touches && !corner_before.contains(&t) { claim("trapped piece", v, vec![rules::square_name(t)]); }
            }
        }
        if new.len() >= 2 {
            claim("fork", new[1].1, names.clone());
        } else if new.len() == 1 {
            claim(if q.is_check() { "fork with check" } else { "single threat" }, new[0].1, names.clone());
        }
        if !mate_before && !q.is_check() {
            if let Ok(passed) = q.clone().swap_turn() {
                if mate_in_one(&passed) { claim("cornered king (threat)", 50, vec![]); }   // cornering the king next move
            }
        }
        if m.role() == Role::Pawn && m.to().rank() == (if us == Color::White { shakmaty::Rank::Seventh } else { shakmaty::Rank::Second }) {
            claim("promotion", 8, vec![m.to().to_string()]);
        }
    }
    Ok(out)
}

/// local_check_many([(fen, uci, k)]) -> [Option<bool>] (one call for a batch)
#[pyfunction]
fn local_check_many(items: Vec<(String, String, i64)>) -> PyResult<Vec<Option<bool>>> {
    Ok(items.into_iter().map(|(fen, uci, k)| {
        let pos = parse(&fen)?;
        let m = uci.parse::<shakmaty::uci::UciMove>().ok()?.to_move(&pos).ok()?;
        Some(local_check_pos(&pos, &m, k))
    }).collect())
}

#[pymodule]
fn analyst_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(literal_masks, m)?)?;
    m.add_function(wrap_pyfunction!(outcomes, m)?)?;
    m.add_function(wrap_pyfunction!(nodes, m)?)?;
    m.add_function(wrap_pyfunction!(local_check, m)?)?;
    m.add_function(wrap_pyfunction!(local_check_many, m)?)?;
    m.add_function(wrap_pyfunction!(threat_claims, m)?)?;
    m.add_function(wrap_pyfunction!(local_check_frames, m)?)?;
    m.add_function(wrap_pyfunction!(pursuit_check, m)?)?;
    m.add_function(wrap_pyfunction!(corner_check, m)?)?;
    m.add_function(wrap_pyfunction!(certified_gain, m)?)?;
    m.add_function(wrap_pyfunction!(best_reply, m)?)?;
    m.add_function(wrap_pyfunction!(target_profile, m)?)?;
    m.add_function(wrap_pyfunction!(target_moves, m)?)?;
    m.add_function(wrap_pyfunction!(fact_pool, m)?)?;
    m.add_function(wrap_pyfunction!(maia_policy, m)?)?;
    m.add_function(wrap_pyfunction!(frame_outs, m)?)?;
    m.add_function(wrap_pyfunction!(frame_tree, m)?)?;
    m.add_function(wrap_pyfunction!(set_exact, m)?)?;
    m.add_function(wrap_pyfunction!(set_frame_proofs, m)?)?;
    m.add_function(wrap_pyfunction!(frame_leaf, m)?)?;
    m.add_function(wrap_pyfunction!(frame_line, m)?)?;
    m.add_function(wrap_pyfunction!(plan::set_zone_reuse, m)?)?;
    m.add_function(wrap_pyfunction!(plan::set_cert_reuse, m)?)?;
    m.add_function(wrap_pyfunction!(plan::set_frame_audit, m)?)?;
    m.add_function(wrap_pyfunction!(plan::frame_audit_stats, m)?)?;
    m.add_function(wrap_pyfunction!(plan::zone_stats, m)?)?;
    m.add_function(wrap_pyfunction!(lemma::set_lemma, m)?)?;
    m.add_function(wrap_pyfunction!(lemma::set_lemma_audit, m)?)?;
    m.add_function(wrap_pyfunction!(lemma::lemma_stats, m)?)?;
    m.add_function(wrap_pyfunction!(lemma::threat_lemma_check, m)?)?;
    m.add_function(wrap_pyfunction!(lemma::set_mine, m)?)?;
    m.add_function(wrap_pyfunction!(lemma::set_chase, m)?)?;
    m.add_function(wrap_pyfunction!(lemma::chase_why, m)?)?;
    m.add_function(wrap_pyfunction!(lemma::chase_check, m)?)?;
    m.add_function(wrap_pyfunction!(lemma::mine_cases, m)?)?;
    m.add_function(wrap_pyfunction!(waiting::waiting_frames, m)?)?;
    m.add_function(wrap_pyfunction!(waiting::arming_moves, m)?)?;
    m.add_class::<waiting::FramePool>()?;
    m.add_function(wrap_pyfunction!(waiting::set_deep_conditions, m)?)?;
    m.add_function(wrap_pyfunction!(prove_stats, m)?)?;
    m.add_function(wrap_pyfunction!(waiting::audit, m)?)?;
    m.add_function(wrap_pyfunction!(waiting::set_unified, m)?)?;
    m.add_function(wrap_pyfunction!(waiting::set_gate, m)?)?;
    m.add_function(wrap_pyfunction!(waiting::clear_gate, m)?)?;
    m.add_function(wrap_pyfunction!(waiting::gate_feature_names, m)?)?;
    m.add_function(wrap_pyfunction!(waiting::log_frames, m)?)?;
    m.add_function(wrap_pyfunction!(waiting::frame_log, m)?)?;
    m.add_function(wrap_pyfunction!(waiting::frame_trace, m)?)?;
    m.add_function(wrap_pyfunction!(paths::path_audit, m)?)?;
    m.add_function(wrap_pyfunction!(paths::path_audit_counts, m)?)?;
    m.add_function(wrap_pyfunction!(paths::attack_paths_of, m)?)?;
    m.add_function(wrap_pyfunction!(plan::plan_trace, m)?)?;
    m.add_function(wrap_pyfunction!(plan::threats_after, m)?)?;
    m.add_function(wrap_pyfunction!(threat::threat_check, m)?)?;
    m.add_function(wrap_pyfunction!(threat::checks_audit, m)?)?;
    m.add_function(wrap_pyfunction!(threat::forcing_stats, m)?)?;
    m.add_function(wrap_pyfunction!(rules::threats_inc_audit, m)?)?;
    m.add_function(wrap_pyfunction!(rules::set_inc_off, m)?)?;
    m.add_function(wrap_pyfunction!(threat::forcing_moves_audit, m)?)?;
    m.add_function(wrap_pyfunction!(threat::forcing_value_of, m)?)?;
    m.add_function(wrap_pyfunction!(threat::set_composed, m)?)?;
    m.add_function(wrap_pyfunction!(plan::set_strict_stop, m)?)?;
    m.add_class::<plan::PlanState>()?;
    m.add_function(wrap_pyfunction!(plan::set_reuse_fails, m)?)?;
    m.add_function(wrap_pyfunction!(plan::line_pair_check, m)?)?;
    m.add_function(wrap_pyfunction!(plan::overload_check, m)?)?;
    m.add_function(wrap_pyfunction!(plan::set_overload, m)?)?;
    m.add_function(wrap_pyfunction!(plan::set_nucleus, m)?)?;
    m.add_function(wrap_pyfunction!(plan::set_pool, m)?)?;
    m.add_function(wrap_pyfunction!(plan::set_frame_seeds, m)?)?;
    m.add_function(wrap_pyfunction!(plan::set_after_check, m)?)?;
    m.add_function(wrap_pyfunction!(plan::set_avoid, m)?)?;
    m.add_function(wrap_pyfunction!(plan::set_change_gate, m)?)?;
    m.add_function(wrap_pyfunction!(plan::set_tie_break, m)?)?;
    m.add_function(wrap_pyfunction!(plan::net_of, m)?)?;
    m.add_function(wrap_pyfunction!(plan::set_see_threats, m)?)?;
    m.add_function(wrap_pyfunction!(plan::set_harass, m)?)?;
    m.add_function(wrap_pyfunction!(waiting::set_one_move_waits, m)?)?;
    Ok(())
}
