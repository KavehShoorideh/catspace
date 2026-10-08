//! The analyst's hot path in Rust (Kaveh 2026-09-24: "do some sort of these calculations in Rust"): a chess
//! position with the frame's pieces as ids, the rules-level relations, the actions of a piece, the frame-local
//! outcome (one reply or the k-ply episode with a pass), and literal masks over many rows at once. Move generation
//! comes from shakmaty (import, don't reinvent). Nothing here knows what a lemma is.
use pyo3::prelude::*;
mod rules;
use std::sync::atomic::{AtomicU64, Ordering};
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

fn kernel_material(pos: &Chess, side: Color) -> i64 {
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
fn threat_claims(fen: String) -> PyResult<Vec<(String, String, i64, bool, Vec<String>)>> {
    let Some(pos) = parse(&fen) else { return Ok(vec![]) };
    let us = pos.turn();
    let before = rules::threats_of(&pos, us);
    let mate_before = match pos.clone().swap_turn() { Ok(_) => mate_in_one(&pos), Err(_) => false };
    let mut out = Vec::new();
    for m in pos.legal_moves() {
        let q = after(&pos, &m);
        if q.is_game_over() { continue; }
        let now = rules::threats_of(&q, us);
        let mut new: Vec<(u8, i64)> = now.iter().filter(|(t, g)| before.get(t).map_or(true, |g0| **g > *g0)).map(|(t, g)| (*t, *g)).collect();
        new.sort_by(|a, b| b.1.cmp(&a.1));
        let uci = m.to_uci(CastlingMode::Standard).to_string();
        let names: Vec<String> = new.iter().map(|(t, _)| rules::square_name(*t)).collect();
        let mut claim = |name: &str, k: i64, targets: Vec<String>| {
            let cert = local_check_pos(&pos, &m, k);
            out.push((uci.clone(), name.to_string(), k, cert, targets));
        };
        if new.len() >= 2 {
            claim("fork", new[1].1, names.clone());
        } else if new.len() == 1 {
            claim(if q.is_check() { "fork with check" } else { "cornered piece" }, new[0].1, names.clone());
        }
        if !mate_before && !q.is_check() {
            if let Ok(passed) = q.clone().swap_turn() {
                if mate_in_one(&passed) { claim("mate threat", 50, vec![]); }
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
    Ok(())
}
