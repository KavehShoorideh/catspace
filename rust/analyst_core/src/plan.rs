//! TARGET-FIRST PLANNING ON PATHS (Kaveh 2026-10-09, the castling position: "does the new rook attack anything? No ...
//! if I pass, all I'm going to lose is a pawn ... what attack can I cook up? ... if I castle and then I double, and then
//! I hop, I could set up a frame -- how do we see this frame without searching for it? And then how do we recognize
//! that every other frame is probably not going to give us anything? There is no more valuable piece to win"; "explain
//! it in a reasonable amount of calculations").
//!
//!  1 THEIR THREAT    on the null move: what we lose if we pass (their resolved captures) -- the TEMPO BUDGET: a plan
//!                    that needs a preparing move is allowed only if passing costs at most a pawn
//!  2 TARGETS         their pieces by value, the king first (mate); before each target, the STOP RULE: if what we
//!                    already have certified is worth at least the target, nothing below it can beat it
//!  3 SEEDS           only our moves on a PATH onto the target (paths.rs: arrivals and discoveries) -- each proved once
//!  4 ARMING          a seed that fails leaves the target escapes: our one-move paths onto those escapes or onto the
//!                    target (an arrival, or a battery loaded behind the seed's own piece) are the candidate preparing
//!                    moves -- each pair (preparing move, seed) proved once on the null move after it
//! Every proof is counted; the trace is the explanation.
use super::*;
use shakmaty::Bitboard;
use crate::paths::attack_paths;
use crate::waiting::{escapes, prove_n, touched, MATE_GOAL};
use std::cell::RefCell;

// THE FRAME MEMORY (Kaveh 2026-10-09: "all of these frames ... have two or three outs, and each out is a value -- so we
// should have this value memorized, so that every time we see this frame we know the outcome, unless something; then we
// check the conditions -- that's the only part where we pay extra"). Every proof the planner asks -- a QUESTION: the seed,
// the focus, the goal, the side to move (where in the analysis it was asked does not matter; the board comparison
// decides) -- is remembered with the board it was asked on, its answer, its stored proof
// or refutation, and its SUPPORT (the squares the answer rests on: the squares its proof or refutation moved through,
// the focus and its neighbourhood, the seed's squares). Asked again on a later board:
//   same board                    -> the answer, no search
//   it HELD                       -> REPLAY the stored proof (cheap; sound -- a re-check, every reply searched)
//   it FAILED                     -> re-prove, the stored refutation tried first (reusing a "fails" whose support was
//                                    untouched is an option, OFF: audited unsafe -- see `mprove`)
#[derive(Clone)]
struct Entry { board: Vec<u8>, extra: String, holds: bool, hints: std::sync::Arc<Hints>, support: Bitboard }

#[derive(Default, Clone, Copy)]
pub struct Stats { pub fresh: u32, pub replay: u32, pub reuse_exact: u32, pub reuse_fail: u32, pub reprove: u32,
                   pub nodes: u64, pub lone: u32, pub frozen: u32, pub escape_skip: u32, pub cut_by_history: u32 }

#[derive(Default)]
pub struct Memo { entries: HashMap<String, Vec<Entry>>, pub history: HashMap<String, f32>, pub stats: Stats }
const SNAPSHOTS: usize = 8;
pub static REUSE_FAILS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
#[pyfunction]
pub fn set_reuse_fails(flag: bool) -> PyResult<()> { REUSE_FAILS.store(flag, Ordering::Relaxed); Ok(()) }                                               // boards kept per question (the oldest dropped)

thread_local! {
    static MEMO: RefCell<Option<Memo>> = const { RefCell::new(None) };
    static TAG: RefCell<Vec<String>> = const { RefCell::new(Vec::new()) };
    static NO_STOP: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };   // threats_after: every target, no stop rule
}

fn snapshot(p: &Chess) -> (Vec<u8>, String) {
    let b = p.board();
    let v = (0..64u32).map(|i| b.piece_at(Square::new(i)).map_or(0u8, |pc| 1 + pc.role as u8 + if pc.color == Color::White { 0 } else { 8 })).collect();
    let fen = shakmaty::fen::Fen::from_position(p.clone(), shakmaty::EnPassantMode::Legal).to_string();
    let parts: Vec<&str> = fen.split(' ').collect();
    (v, format!("{} {} {}", parts.get(1).unwrap_or(&""), parts.get(2).unwrap_or(&""), parts.get(3).unwrap_or(&"")))
}

fn with_tag<T>(tag: String, f: impl FnOnce() -> T) -> T {
    TAG.with(|t| t.borrow_mut().push(tag));
    let out = f();
    TAG.with(|t| { t.borrow_mut().pop(); });
    out
}

fn tally(f: impl FnOnce(&mut Stats)) { MEMO.with(|m| if let Some(m) = m.borrow_mut().as_mut() { f(&mut m.stats) }); }

/// A proof question through the memory (or straight to the search when no memory is installed): (holds, nodes searched).
fn mprove(p: &Chess, seed: &str, focus: &[Square], goal: i64) -> (bool, u64) {
    let installed = MEMO.with(|m| m.borrow().is_some());
    if !installed { let (ok, _, n, _) = prove_n(p, seed, focus, goal, None); return (ok, n); }
    // the QUESTION alone is the key -- where it was asked does not matter, the board comparison decides (2026-10-09: keyed
    // by where it was asked, White's null-move analysis of Black's threat never served Black's own analysis a move later)
    let key = format!("{}|{}|{}|{}|{}", seed, focus.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(","), goal,
                      if p.turn() == Color::White { "w" } else { "b" }, crate::waiting::DEEP_CONDITIONS.load(Ordering::Relaxed));
    let (board, extra) = snapshot(p);
    let prior = MEMO.with(|m| m.borrow().as_ref().and_then(|m| m.entries.get(&key).and_then(|v| {
        v.iter().filter(|e| e.extra == extra)
            .min_by_key(|e| (0..64usize).filter(|i| e.board[*i] != board[*i]).count()).cloned()
            .or_else(|| v.last().cloned())
    })));
    let mut hints = None;
    if let Some(e) = prior {
        if e.extra == extra {
            let diff: Bitboard = (0..64u32).filter(|i| e.board[*i as usize] != board[*i as usize]).map(Square::new).collect();
            if diff.is_empty() { tally(|s| s.reuse_exact += 1); return (e.holds, 0); }
            // (OFF by default -- the audit of 2026-10-09: reusing a "fails" whose support no move touched missed two
            // immediate mates and misread their threat in several positions on 60 games: a refutation depends on more
            // than the squares it moved through, e.g. a piece guarding the landing square from afar)
            if REUSE_FAILS.load(Ordering::Relaxed) && !e.holds && (diff & e.support).is_empty() { tally(|s| s.reuse_fail += 1); return (false, 0); }
            if e.holds { tally(|s| s.replay += 1); } else { tally(|s| s.reprove += 1); }
        } else {
            tally(|s| s.reprove += 1);
        }
        hints = Some(e.hints.clone());
    } else {
        tally(|s| s.fresh += 1);
    }
    let (ok, rec, n, _) = prove_n(p, seed, focus, goal, hints);
    let mut support = touched(&rec);
    for f in focus { support.add(*f); support |= shakmaty::attacks::king_attacks(*f); }
    if let Some(m) = seed.parse::<shakmaty::uci::UciMove>().ok().and_then(|u| u.to_move(p).ok()) {
        if let Some(fr) = m.from() { support.add(fr); }
        support.add(m.to());
    }
    MEMO.with(|m| if let Some(m) = m.borrow_mut().as_mut() {
        m.stats.nodes += n;
        let v = m.entries.entry(key).or_default();
        v.retain(|e| e.board != board || e.extra != extra);
        v.push(Entry { board, extra, holds: ok, hints: std::sync::Arc::new(rec), support });
        if v.len() > SNAPSHOTS { v.remove(0); }
    });
    (ok, n)
}

pub struct Plan { pub trace: Vec<String>, pub best: Option<(String, String, i64, String)>, pub proofs: u32, pub nodes: u64, pub target: Option<Square>,
                 pub immediate: Option<(String, String, i64, String)>, pub wins: Vec<(String, Square, i64)> }

fn val(r: Role) -> i64 { if r == Role::King { 1000 } else { kernel_value(r) } }

fn san(pos: &Chess, uci: &str) -> String {
    uci.parse::<shakmaty::uci::UciMove>().ok().and_then(|u| u.to_move(pos).ok())
        .map(|m| shakmaty::san::San::from_move(pos, &m).to_string()).unwrap_or_else(|| uci.to_string())
}

/// WHAT OUR MOVE EXPOSES (the vacated-square clause of the proved path lemma, Ideas.PathLemma.path_lemma, read for THEIR
/// attacks): our move m newly exposes a piece of ours that one of their sliders now sees through the square m vacated, or
/// leaves the moved piece itself hanging -> [(square, their static gain on it)] with a gain (Kaveh 2026-10-09: "the
/// running away of the bishop exposes the knight").
fn exposes(pos: &Chess, m: &Move) -> Vec<(Square, i64)> {
    let us = pos.turn();
    let them = !us;
    let q = after(pos, m);
    let gains = rules::threats_of(&q, them);
    let b0 = pos.board();
    let occ0 = b0.occupied();
    let vac: Vec<Square> = match m { Move::Castle { king, rook } => vec![*king, *rook], _ => m.from().into_iter().collect() };
    let mut out = Vec::new();
    for x in q.board().by_color(us) {
        let Some(&g) = gains.get(&(x as u8)) else { continue };
        if g <= 0 { continue; }
        let moved_here = x == m.to() || matches!(m, Move::Castle { .. });
        let newly = q.board().attacks_to(x, them, q.board().occupied()).into_iter().any(|a| {
            !b0.attacks_to(x, them, occ0).contains(a) && vac.iter().any(|v| shakmaty::attacks::between(a, x).contains(*v))
        });
        if newly || (moved_here && x == m.to()) { out.push((x, g)); }
    }
    out
}

/// THE LINE PAIR (Kaveh 2026-10-09): their slider `sl` attacks our front piece `f` and our piece `r` stands behind it on
/// the line. Can either protect the other from a new square? (Read off the legal moves -- for the knight and the bishop
/// the proved colour lemmas say why not.) -> an explanation when neither can: one of them is lost unless the front piece
/// is protected where it stands, the slider is taken or blocked, or a bigger threat comes first.
fn line_pair(pos: &Chess, sl: Square, f: Square) -> Option<String> {
    let b = pos.board();
    let us = pos.turn();
    let occ = b.occupied();
    let beyond = shakmaty::attacks::ray(sl, f) & !shakmaty::attacks::between(sl, f) & !Bitboard::from_square(sl) & !Bitboard::from_square(f);
    let r = beyond.into_iter().filter(|x| occ.contains(*x)).min_by_key(|x| shakmaty::attacks::between(f, *x).count())?;
    if b.piece_at(r)?.color != us { return None; }
    let protects = |from: Square, to: Square| pos.legal_moves().iter().any(|m| m.from() == Some(from) && {
        let q = after(pos, m);
        q.board().attacks_to(to, us, q.board().occupied()).contains(m.to())
    });
    if protects(f, r) || protects(r, f) { return None; }
    let (pf, pr) = (b.piece_at(f)?.role, b.piece_at(r)?.role);
    let orth = (f.file() as i32 - r.file() as i32).abs() + (f.rank() as i32 - r.rank() as i32).abs() == 1;
    let why = |p: Role, other: Square| -> String {
        match p {
            Role::Bishop if orth => format!(" (a bishop keeps its colour; {other} is the other colour)"),
            Role::Knight if orth => format!(" (a knight that hops never covers an orthogonal neighbour of where it stood)"),
            _ => String::new(),
        }
    };
    Some(format!("line pair: our {:?} on {f} and {:?} on {r} under their {:?} on {sl} -- the {:?} cannot protect {r} from a new square{}, nor the {:?} protect {f}{}: moving the front piece exposes the back one; one is lost unless {f} is protected where it stands, the {:?} is taken or blocked, or a bigger threat comes first",
        pf, pr, b.piece_at(sl).map(|x| x.role)?, pf, why(pf, r), pr, why(pr, f), b.piece_at(sl).map(|x| x.role)?).to_lowercase())
}

/// THE BACK PIECE of a line under their slider `sl` through our front piece `f` (the first piece beyond f), if ours.
fn back_piece(pos: &Chess, sl: Square, f: Square) -> Option<Square> {
    let b = pos.board();
    let occ = b.occupied();
    let beyond = shakmaty::attacks::ray(sl, f) & !shakmaty::attacks::between(sl, f) & !Bitboard::from_square(sl) & !Bitboard::from_square(f);
    let r = beyond.into_iter().filter(|x| occ.contains(*x) && !shakmaty::attacks::between(sl, *x).contains(*x))
        .min_by_key(|x| shakmaty::attacks::between(f, *x).count())?;
    if b.piece_at(r)?.color != pos.turn() { return None; }
    Some(r)
}

/// ALPHA-BETA IN THE FRAME (Kaveh 2026-10-09: "we should always have the search as a fallback ... an alpha-beta mini
/// search"; "alpha beta in frame"): our first moves are the frame's -- moves of its pieces, moves onto the attacking
/// line or against the attacker, captures, checks -- each valued by the value search focused on the frame's pieces (our
/// later moves the frame's answers, every reply of theirs searched) -> [(move, value relative to now)], best first.
fn frame_fallback(pos: &Chess, frame: &[Square], line: Bitboard, budget: u64) -> Vec<(String, i64)> {
    let mut out: Vec<(String, i64)> = Vec::new();
    for m in pos.legal_moves() {
        let in_frame = m.from().map_or(false, |f| frame.contains(&f)) || frame.contains(&m.to()) || line.contains(m.to())
            || m.is_capture() || after(pos, &m).is_check();
        if !in_frame { continue; }
        if let Some(v) = certified_value(pos, &m, frame, 3, budget) {
            out.push((m.to_uci(CastlingMode::Standard).to_string(), v));
        }
    }
    out.sort_by(|a, b| b.1.cmp(&a.1));
    out
}

/// THE LINE-PAIR LEMMA (extracted 2026-10-09 from the fallback search on Kaveh's d6/d7 position, then verified on the
/// census -- line_pair_check): under their slider, with neither of our two pieces able to cover the other after moving
/// (the colour lemmas) and no defence found, the outcome is VALUE ARITHMETIC -- for each move of the front or the back
/// piece: what it captures, minus the most they can then take (the piece left behind on the line, or the mover where it
/// lands); the best of these is the cheapest way to lose. Piece directions and kinds enter only through the
/// preconditions. -> (move, predicted net, explanation)
fn line_pair_lemma(pos: &Chess, sl: Square, f: Square, r: Square) -> Option<(String, i64, String)> {
    let them = !pos.turn();
    let us = pos.turn();
    let mut best: Option<(String, i64, String)> = None;
    // THE LEMMA APPLIES ONLY WHEN EVERY MOVE OF THE PAIR IS QUIET (verified 2026-10-09: the exchange law priced the
    // pair's moves as the search did in 84%; every miss was a check, a new threat of ours, or a check left to them --
    // a fork or a mate the captures do not see): no check, no new attack of ours, no check available to them
    let before = rules::threats_of(pos, us);
    for m in pos.legal_moves() {
        let Some(from) = m.from() else { continue };
        if from != f && from != r { continue; }
        // THE EXCHANGE LAW prices it (2026-10-09: "capture minus their first capture" agreed with the search on 50% of
        // the pair's own moves -- the rest of the exchange matters): the material the move takes, then their whole
        // capture sequence resolved (settle: either side may stop)
        let q = after(pos, &m);
        let new_threat = rules::threats_of(&q, us).iter().any(|(x, g)| *g > 0 && before.get(x).map_or(true, |g0| *g > *g0) && *x != m.to() as u8);
        let their_check = q.board().king_of(us).map_or(false, |k| attack_paths(&q, them, Bitboard::from_square(k)).iter().any(|p| p.kind == "direct" || p.kind == "discovery"));
        if q.is_check() || new_threat || their_check { return None; }             // not quiet: the lemma does not apply
        let cap = kernel_material(&q, pos.turn()) - kernel_material(pos, pos.turn());
        BUDGET.store(200_000, Ordering::Relaxed);
        let theirs = settle_d(&q, -2000, 2000, false, SETTLE_DEPTH).max(0);
        BUDGET.store(u64::MAX, Ordering::Relaxed);
        let net = cap - theirs;
        let _ = (them, r);
        let why = format!("{} takes {cap}, then their captures resolve to {theirs}", m.to_uci(CastlingMode::Standard));
        if best.as_ref().map_or(true, |b| net > b.1) { best = Some((m.to_uci(CastlingMode::Standard).to_string(), net, why)); }
    }
    let _ = sl;
    best
}

/// line_pair_check(fen) -> for each line pair of the side to move under an attacking slider of theirs: (slider, front,
/// back, the lemma's (move, net), the frame fallback's top three (move, value)) -- the lemma's verification.
#[pyfunction]
pub fn line_pair_check(fen: String) -> PyResult<Vec<(String, String, String, Option<(String, i64, String)>, Vec<(String, i64)>)>> {
    let Some(pos) = parse(&fen) else { return Ok(vec![]) };
    let us = pos.turn();
    let them = !us;
    let b = pos.board();
    let mut out = Vec::new();
    for f in b.by_color(us) {
        if b.piece_at(f).map_or(true, |p| p.role == Role::King) { continue; }
        for sl in b.attacks_to(f, them, b.occupied()) {
            if !b.piece_at(sl).map_or(false, |p| matches!(p.role, Role::Rook | Role::Bishop | Role::Queen)) { continue; }
            let Some(r) = back_piece(&pos, sl, f) else { continue };
            if b.piece_at(r).map_or(true, |p| p.role == Role::King) { continue; }
            if line_pair(&pos, sl, f).is_none() { continue; }                       // the no-cover precondition
            // the lemma's other preconditions (2026-10-09: verified without them it agreed with the search on 54%): the
            // FRONT piece is winnable where it stands, and the BACK piece is winnable once the front one is gone
            let Some(pp) = pass(&pos) else { continue };
            if rules::threats_of(&pp, them).get(&(f as u8)).copied().unwrap_or(0) <= 0 { continue; }
            let mut setup = pp.clone().into_setup(shakmaty::EnPassantMode::Legal);
            setup.board.discard_piece_at(f); setup.ep_square = None;
            let Some(lifted) = Chess::from_setup(setup, CastlingMode::Standard).or_else(|e| e.ignore_invalid_castling_rights()).ok() else { continue };
            if rules::threats_of(&lifted, them).get(&(r as u8)).copied().unwrap_or(0) <= 0 { continue; }
            let lemma = line_pair_lemma(&pos, sl, f, r);
            let mut line = shakmaty::attacks::between(sl, r); line.add(sl);
            let fb = frame_fallback(&pos, &[f, r, sl], line, 50_000);
            // the narrower claim: the lemma PRICES THE PAIR'S OWN MOVES -- compare with the search over those moves only
            let pair: Vec<(String, i64)> = fb.iter().filter(|(u, _)| u.starts_with(&f.to_string()) || u.starts_with(&r.to_string())).cloned().collect();
            let mut top: Vec<(String, i64)> = fb.into_iter().take(3).collect();
            if let Some(pb) = pair.first() { top.push((format!("pair:{}", pb.0), pb.1)); }
            out.push((sl.to_string(), f.to_string(), r.to_string(), lemma, top));
        }
    }
    Ok(out)
}

/// The knight-distance table (exact, by breadth-first search), computed once.
fn knight_dist() -> &'static [[u8; 64]; 64] {
    static T: OnceLock<[[u8; 64]; 64]> = OnceLock::new();
    T.get_or_init(|| {
        let mut t = [[255u8; 64]; 64];
        for s in 0..64usize {
            t[s][s] = 0;
            let mut q = std::collections::VecDeque::from([s]);
            while let Some(u) = q.pop_front() {
                for v in shakmaty::attacks::knight_attacks(Square::new(u as u32)) {
                    let v = v as usize;
                    if t[s][v] == 255 { t[s][v] = t[s][u] + 1; q.push_back(v); }
                }
            }
        }
        t
    })
}

/// THE UNSTOPPABLE PASSER (Kaveh 2026-10-10: "cheaply find endgame solutions instead of a deep search or a table
/// lookup"): the rule of the square generalized to the minor pieces, read off exact geometry, with the side `side`'s
/// OPPONENT to move in `q`. A pawn of `side` queens if (all conservative -- a doubt means "stoppable"): the opponent has
/// no rook or queen; every square of its path to the last rank is empty and no enemy pawn stands on its file or the
/// next ones ahead of it (passed); the pawn is not attacked now; the opponent's king cannot reach the queening square in
/// time (Chebyshev distance > moves needed: the rule of the square, the defender to move); no enemy knight can block a
/// path square or attack it before the pawn passes it (the knight-distance table); no enemy bishop stands on the colour
/// of any path square still ahead (the proved lemma bishop_keeps_colour: a bishop never stops a pawn on the other
/// colour). -> the fastest such pawn: (square, moves to queen).
fn unstoppable(q: &Chess, side: Color) -> Option<(Square, u32)> { unstoppable_k(q, side, true) }

/// `with_king`: whether the defending king counts as a stopper -- in a RACE it does not (2026-10-10 audit: Black's king
/// "could catch" White's b-pawn, but not while Black spends every move pushing the h-pawn)
fn unstoppable_k(q: &Chess, side: Color, with_king: bool) -> Option<(Square, u32)> {
    let b = q.board();
    let opp = !side;
    if (b.by_color(opp) & (b.by_role(Role::Rook) | b.by_role(Role::Queen))).any() { return None; }
    let kd = knight_dist();
    let occ = b.occupied();
    let fwd: i32 = if side == Color::White { 1 } else { -1 };
    let last = if side == Color::White { 7 } else { 0 };
    let mut best: Option<(Square, u32)> = None;
    for p in b.by_color(side) & b.by_role(Role::Pawn) {
        let (f, r) = (p.file() as i32, p.rank() as i32);
        let path: Vec<Square> = (1..=((last - r) * fwd)).map(|k| Square::from_coords(shakmaty::File::new(f as u32), shakmaty::Rank::new((r + k * fwd) as u32))).collect();
        if path.is_empty() || path.iter().any(|x| occ.contains(*x)) { continue; }
        let passed = !(b.by_color(opp) & b.by_role(Role::Pawn)).into_iter().any(|e| (e.file() as i32 - f).abs() <= 1 && (e.rank() as i32 - r) * fwd > 0);
        if !passed { continue; }
        // attacked now: stopped -- except, in a race, by the defending king (busy with its own pawn)
        let att = b.attacks_to(p, opp, occ) & if with_king { Bitboard::FULL } else { !b.by_role(Role::King) };
        if att.any() { continue; }
        let start = if side == Color::White { 1 } else { 6 };
        let need = path.len() as u32 - if r == start && path.len() >= 2 { 1 } else { 0 };
        let z = *path.last().unwrap();
        if let Some(k) = b.king_of(opp).filter(|_| with_king) {
            let d = (k.file() as i32 - z.file() as i32).abs().max((k.rank() as i32 - z.rank() as i32).abs()) as u32;
            if d <= need + 1 { continue; }                              // (audit: the defender to move steps INTO the square: Kc5-b6 catches a6)
        }
        // the pawn reaches path[i] after its (i+1)-th move (the double step: path[1] after its first); the opponent has
        // moved once more by then
        let arrive = |i: usize| -> u32 { if r == start && path.len() >= 2 { if i == 0 { 1 } else { i as u32 } } else { i as u32 + 1 } };
        let mut stopped = false;
        for n in b.by_color(opp) & b.by_role(Role::Knight) {
            for (i, x) in path.iter().enumerate() {
                let t = arrive(i);
                let reach_block = kd[n as usize][*x as usize] as u32;
                let reach_attack = shakmaty::attacks::knight_attacks(*x).into_iter().map(|a| kd[n as usize][a as usize] as u32).min().unwrap_or(255);
                if reach_block <= t || reach_attack <= t { stopped = true; break; }
            }
            if stopped { break; }
        }
        if stopped { continue; }
        for bi in b.by_color(opp) & b.by_role(Role::Bishop) {
            let bc = (bi.file() as u32 + bi.rank() as u32) % 2;
            if path.iter().any(|x| (x.file() as u32 + x.rank() as u32) % 2 == bc) { stopped = true; break; }
        }
        if stopped { continue; }
        if best.map_or(true, |(_, n)| need < n) { best = Some((p, need)); }
    }
    best
}

/// THE SKEWER, the x-ray through the king: our check whose line runs on past their king to the piece `t` behind it
/// (the first piece beyond the king) -- when the king steps aside, `t` is attacked.
fn skewers(pos: &Chess, m: &Move, t: Square) -> bool {
    let them = !pos.turn();
    let q = after(pos, m);
    if !q.is_check() { return false; }
    let Some(k) = q.board().king_of(them) else { return false };
    let c = m.to();
    let Some(pc) = q.board().piece_at(c) else { return false };
    if !matches!(pc.role, Role::Rook | Role::Bishop | Role::Queen) { return false; }
    let ray = shakmaty::attacks::ray(c, k);
    ray.contains(t) && shakmaty::attacks::between(c, t).contains(k)
        && (shakmaty::attacks::between(k, t) & q.board().occupied()).is_empty()
        && shakmaty::attacks::attacks(c, pc, q.board().occupied().without(k)).contains(t)
}

/// OVERLOAD (Kaveh 2026-10-10: "when a piece is needed in one frame and in another frame there is an opportunity cost
/// ... that's where overloading comes from: it's doing two jobs in two different frames ... a rook can protect the whole
/// back rank, a bishop cannot"). A JOB of `side`'s piece d: another piece x of `side` is safe now and would be won (the
/// static count) if d were lifted -- d is its critical guard. A piece with two jobs is OVERLOADED unless one square
/// serves both: after the attacker takes on one job and d recaptures there, d must still bear on the other (for both
/// orders). The loss is value arithmetic: the attacker picks the order, the defender picks whether to recapture.
/// -> [(d, first job taken, second job, attacker's capture move, loss, explanation)], the attacker = !side to move...
/// computed for `side`'s pieces under `atk`'s attacks, whoever is to move.
fn overloads(pos: &Chess, side: Color) -> Vec<(Square, Square, Square, i64, String)> {
    let atk = !side;
    let b = pos.board();
    let occ = b.occupied();
    let lifted_board = |d: Square| -> Option<Chess> {
        let mut st = pos.clone().into_setup(shakmaty::EnPassantMode::Legal);
        st.board.discard_piece_at(d); st.ep_square = None;
        Chess::from_setup(st, CastlingMode::Standard).or_else(|e| e.ignore_invalid_castling_rights())
            .or_else(|e| e.ignore_impossible_check()).ok()
    };
    let now = rules::threats_of(pos, atk);
    let mut jobs: HashMap<Square, Vec<Square>> = HashMap::new();
    for d in b.by_color(side) {
        let Some(lp) = lifted_board(d) else { continue };
        let without = rules::threats_of(&lp, atk);
        for x in b.by_color(side) {
            if x == d || b.piece_at(x).map_or(true, |p| p.role == Role::King) { continue; }
            if !shakmaty::attacks::attacks(d, b.piece_at(d).unwrap(), occ).contains(x) { continue; }   // d guards x
            if now.get(&(x as u8)).copied().unwrap_or(0) > 0 { continue; }                      // already lost: no job
            if without.get(&(x as u8)).copied().unwrap_or(0) > 0 { jobs.entry(d).or_default().push(x); }
        }
    }
    let mut out = Vec::new();
    for (d, xs) in jobs {
        if xs.len() < 2 { continue; }
        let dp = b.piece_at(d).unwrap();
        for (i, &x1) in xs.iter().enumerate() {
            for &x2 in xs.iter().skip(i + 1) {
                // their cheapest capture on each job, and whether d still bears on the other job from there
                let cheapest = |x: Square| b.attacks_to(x, atk, occ).into_iter().min_by_key(|a| b.piece_at(*a).map_or(99, |p| kernel_value(p.role)));
                let still = |xa: Square, xb: Square, a: Square| -> bool {
                    let occ2 = occ.without(d).without(a).with(xa);
                    shakmaty::attacks::attacks(xa, dp, occ2).contains(xb)
                };
                let mut worst: Option<(Square, Square, i64)> = None;
                for (xa, xb) in [(x1, x2), (x2, x1)] {
                    let Some(a) = cheapest(xa) else { continue };
                    if still(xa, xb, a) { continue; }
                    let (va, vxa, vxb) = (kernel_value(b.piece_at(a).unwrap().role), kernel_value(b.piece_at(xa).unwrap().role), kernel_value(b.piece_at(xb).unwrap().role));
                    let loss = vxa.min(vxa + vxb - va);                    // we recapture (lose both, win the attacker) or not
                    if loss > 0 && worst.map_or(true, |w| loss > w.2) { worst = Some((xa, xb, loss)); }
                }
                if let Some((xa, xb, loss)) = worst {
                    let why = format!("the {:?} on {d} guards both {xa} and {xb}; after a capture on {xa} and its recapture there it no longer bears on {xb}{}",
                        dp.role, if dp.role == Role::Bishop && (xa.file() as u32 + xa.rank() as u32) % 2 != (xb.file() as u32 + xb.rank() as u32) % 2 { " (a bishop keeps its colour)" } else { "" }).to_lowercase();
                    out.push((d, xa, xb, loss, why));
                }
            }
        }
    }
    out
}

pub static OVERLOAD: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);
#[pyfunction]
pub fn set_overload(flag: bool) -> PyResult<()> { OVERLOAD.store(flag, Ordering::Relaxed); Ok(()) }

/// overload_check(fen) -> the overloaded pieces of BOTH sides: (side, piece, first job, second job, loss, explanation)
#[pyfunction]
pub fn overload_check(fen: String) -> PyResult<Vec<(String, String, String, String, i64, String)>> {
    let Some(pos) = parse(&fen) else { return Ok(vec![]) };
    let mut v = Vec::new();
    for side in [Color::White, Color::Black] {
        for (d, a, x, l, w) in overloads(&pos, side) {
            v.push((if side == Color::White { "w" } else { "b" }.to_string(), d.to_string(), a.to_string(), x.to_string(), l, w));
        }
    }
    Ok(v)
}

pub fn plan(pos: &Chess) -> Plan { plan_inner(pos, false) }

/// `immediate_only`: the immediate phase alone (targets, seeds, proofs) -- what the OPPONENT's threat is computed with,
/// on the null move.
pub fn plan_inner(pos: &Chess, immediate_only: bool) -> Plan {
    let us = pos.turn();
    let them = !us;
    let b = pos.board();
    let mut trace = Vec::new();
    let mut proofs = 0u32;
    let mut nodes = 0u64;
    let mut best: Option<(String, String, i64, String)> = None;    // (preparing move or "", seed, gain, how)
    let mut best_target: Option<Square> = None;
    let mut wins: Vec<(String, Square, i64)> = Vec::new();               // every immediate win found (for their threats)
    // 1 THEIR THREAT: their resolved captures if we pass, AND their immediate wins found by the same reasoning on the
    // null move (2026-10-09, after O-O-O in the castling line: Black's planner said "nothing" -- the threat counted only
    // White's captures, not White's mate)
    let mut their: Option<(String, Square, i64)> = None;
    let mut their_wins: Vec<(String, Square, i64)> = Vec::new();
    let threat = if immediate_only { 0 } else {
        let cap = match pass(pos) {
            Some(p) => { BUDGET.store(200_000, Ordering::Relaxed); let v = settle_d(&p, -2000, 2000, true, SETTLE_DEPTH).max(0); BUDGET.store(u64::MAX, Ordering::Relaxed); v }
            None => 0,
        };
        if let Some(p) = pass(pos) {
            let tp = with_tag("null".to_string(), || plan_inner(&p, true));
            proofs += tp.proofs; nodes += tp.nodes;
            their_wins = tp.wins.clone();
            if let (Some(bb), Some(tt)) = (tp.best, tp.target) {
                trace.push(format!("their best immediate win if we pass: {} ({} proofs)", bb.3, tp.proofs));
                their = Some((bb.1, tt, bb.2));
            }
        }
        cap.max(their.as_ref().map_or(0, |x| x.2))
    };
    let tempo = threat <= 1;
    // PINNED TO WHAT (Kaveh 2026-10-09: "the knight on d7 is pinned because the queen can give a mate on d8 if the knight
    // hops"): for each pin of theirs against our king zone (paths.rs), ONE analysis: lift our blocker and run their
    // immediate reasoning; if they then win more than they threaten now, the blocker is pinned to that loss, and its
    // moves that leave the line are refused without proof -- for a knight every move, by the proved lemma
    // Ideas.GeoLemmas.knight_leaves_every_line (taking the pinning piece, and a check, are still proved)
    let mut frozen: HashMap<Square, (Square, Square, i64, String)> = HashMap::new();   // blocker -> (slider, target, loss, why)
    if !immediate_only {
        if let Some(k) = b.king_of(us) {
            let zone = shakmaty::attacks::king_attacks(k).with(k);
            for p in attack_paths(pos, them, zone) {
                if p.kind != "pin" { continue; }
                let Some(bl) = p.via else { continue };
                if frozen.contains_key(&bl) { continue; }
                let mut setup = pos.clone().into_setup(shakmaty::EnPassantMode::Legal);
                setup.board.discard_piece_at(bl);
                setup.turn = them;
                setup.ep_square = None;                                  // (review: a kept ep square made the setup invalid)
                let lifted = Chess::from_setup(setup, CastlingMode::Standard).or_else(|e| e.ignore_invalid_castling_rights()).ok();
                let Some(lifted) = lifted else { continue };
                let lp = with_tag(format!("lift:{bl}"), || plan_inner(&lifted, true));
                proofs += lp.proofs; nodes += lp.nodes;
                if let Some(lb) = lp.best {
                    if lb.2 > threat.max(1) {
                        let role = b.piece_at(bl).map(|x| x.role);
                        trace.push(format!("pin: our {:?} on {bl} stands between their {:?} on {} and {}; lifted, they have {} -- {}",
                            role.unwrap_or(Role::Pawn), b.piece_at(p.piece).map(|x| x.role).unwrap_or(Role::Pawn), p.piece, p.target, lb.3,
                            if role == Some(Role::Knight) { "every knight move opens the line (knight_leaves_every_line): all frozen".to_string() } else { "moves off the line frozen".to_string() }));
                        frozen.insert(bl, (p.piece, p.target, lb.2, format!("{}", lb.3)));
                    }
                }
            }
        }
    }
    let refused = |m: &Move| -> Option<String> {
        let from = m.from()?;
        let (sl, tgt, loss, why) = frozen.get(&from)?;
        if m.to() == *sl { return None; }                                // taking the pinning piece
        if after(pos, m).is_check() { return None; }                       // a check is still proved
        if m.capture().map_or(false, |r| kernel_value(r) >= *loss) { return None; }   // a capture worth the loss: proved
        // still blocking: the move stays between the slider and the pinned-to square (review: the line to the KING was
        // used even when the pin points at another square, d8)
        let on_line = b.piece_at(from).map_or(false, |x| x.role != Role::Knight)
            && shakmaty::attacks::between(*sl, *tgt).contains(m.to());
        if on_line { return None; }
        Some(format!("pinned: it would allow {why} (worth {loss})"))
    };
    if !immediate_only {
        trace.push(format!("their threat if we pass: worth {threat}; a preparing move is {}", if tempo { "affordable" } else { "NOT affordable" }));
    }
    // our checking moves: a check gives an already-attacked piece no time to escape (Kaveh: "a pin and also a cornering"
    // -- several frames from one move; 2026-10-09 misses: Qxf7+ winning the d1 bishop, f3+ the h3 rook, Qxf6+ the g2 bishop)
    let our_checks: Vec<String> = match b.king_of(them) {
        Some(k) => {
            let mut v: Vec<String> = Vec::new();
            for p in attack_paths(pos, us, Bitboard::from_square(k)) {
                match p.kind {
                    "direct" => if let Some(m) = p.mv { if !v.contains(&m) { v.push(m); } },
                    "discovery" => {
                        let line = shakmaty::attacks::between(p.target, p.piece);
                        for m in pos.legal_moves() {
                            if m.from() == p.via && !line.contains(m.to()) { let u = m.to_uci(CastlingMode::Standard).to_string(); if !v.contains(&u) { v.push(u); } }
                        }
                    }
                    _ => {}
                }
            }
            v
        }
        None => vec![],
    };
    // 2 TARGETS by value
    let mut armed: Vec<(i64, Square, Vec<String>)> = Vec::new();        // failing seeds kept for the arming phase
    let mut targets: Vec<(i64, Square)> = b.by_color(them).into_iter().filter_map(|sq| b.piece_at(sq).map(|p| (val(p.role), sq))).filter(|(v, _)| *v >= 3).collect();
    targets.sort_by(|a, b| b.0.cmp(&a.0));
    for (v, t) in targets {
        // the STOP RULE counts only IMMEDIATE wins (2026-10-09: an armed threat -- they get a move to parry it -- stopped
        // the search before Rf5 won a bishop, Rd4 a knight)
        let have = best.as_ref().filter(|x| x.0.is_empty()).map_or(0, |x| x.2);
        if have >= v.min(MATE_GOAL) && !NO_STOP.with(|c| c.get()) {
            trace.push(format!("stop: {} on {t} is worth {}, not more than the {} already won", b.piece_at(t).unwrap().role.char(), v.min(MATE_GOAL), have));
            break;
        }
        let king = v == 1000;
        let goal = if king { MATE_GOAL } else { v };
        let name = format!("{:?} on {t}", b.piece_at(t).unwrap().role).to_lowercase();
        // 3 SEEDS: our moves on a path onto the target
        let paths = attack_paths(pos, us, Bitboard::from_square(t));
        let mut seeds: Vec<String> = Vec::new();
        for p in &paths {
            match p.kind {
                "direct" => if let Some(m) = &p.mv { if !seeds.contains(m) { seeds.push(m.clone()); } },
                "discovery" => {
                    let line = shakmaty::attacks::between(p.target, p.piece);
                    for m in pos.legal_moves() {
                        if m.from() == p.via && !line.contains(m.to()) {
                            let u = m.to_uci(CastlingMode::Standard).to_string();
                            if !seeds.contains(&u) { seeds.push(u); }
                        }
                    }
                }
                "open" => {
                    for m in pos.legal_moves() {                          // already attacked: the capture
                        if m.to() == t { let u = m.to_uci(CastlingMode::Standard).to_string(); if !seeds.contains(&u) { seeds.push(u); } }
                    }
                    if !king { for c in &our_checks { if !seeds.contains(c) { seeds.push(c.clone()); } } }   // ...or a check first
                }
                _ => {}
            }
        }
        // the SKEWER: a check whose line runs on through their king to this piece (2026-10-10, endgame puzzles: the most
        // common missed solution in rook endgames was a check)
        if !king {
            for c in &our_checks {
                if seeds.contains(c) { continue; }
                if let Some(cm) = c.parse::<shakmaty::uci::UciMove>().ok().and_then(|u| u.to_move(pos).ok()) {
                    if skewers(pos, &cm, t) { trace.push(format!("    {}: a skewer through the king onto the {name}", san(pos, c))); seeds.push(c.clone()); }
                }
            }
        }
        if seeds.is_empty() {
            trace.push(format!("{name}: no path onto it this move -- nothing to prove (the path lemma)"));
            continue;
        }
        trace.push(format!("{name} (goal {goal}): {} moves on a path onto it: {}", seeds.len(), seeds.iter().map(|u| san(pos, u)).collect::<Vec<_>>().join(", ")));
        let mut failed: Vec<String> = Vec::new();
        let before = rules::threats_of(pos, us);
        // HISTORY ORDER (Kaveh: "order the lemma check by the most recently used"): the moves that won most recently
        // first; once one wins the target outright, the rest of its moves are not proved
        let hist = MEMO.with(|m| m.borrow().as_ref().map(|m| m.history.clone()).unwrap_or_default());
        let side = if us == Color::White { "w" } else { "b" };
        let hk = |u: &String| format!("{side}:{u}");                      // (review: both sides' moves shared one history)
        seeds.sort_by(|a, b| hist.get(&hk(b)).unwrap_or(&0.0).partial_cmp(hist.get(&hk(a)).unwrap_or(&0.0)).unwrap());
        let mut won_outright = false;
        for s in &seeds {
            // (the cut only for our own moves: the opponent's threat collects EVERY win -- 2026-10-09: two mating threats,
            // each planner found a different one first and defended against only it)
            if won_outright && !immediate_only { tally(|st| st.cut_by_history += 1); continue; }
            if let Some(sm) = s.parse::<shakmaty::uci::UciMove>().ok().and_then(|u| u.to_move(pos).ok()) {
                if let Some(why) = refused(&sm) {
                    trace.push(format!("    {}: refused without proof -- {why}", san(pos, s)));
                    tally(|st| st.frozen += 1);
                    continue;
                }
            }
            // THE LONE-ATTACK LEMMA (2026-10-09, the castling position went from 4 proofs to 10 on attacks the piece just
            // steps away from): a move that attacks a piece with a safe escape wins nothing unless it also gives check or
            // attacks a second piece (escapes count pins). Not for the king: a check is the frame itself.
            if !king {
                if let Some(sm) = s.parse::<shakmaty::uci::UciMove>().ok().and_then(|u| u.to_move(pos).ok()) {
                    let q = after(pos, &sm);
                    let now = rules::threats_of(&q, us);
                    let new = now.iter().filter(|(x, g)| **g > 0 && before.get(x).map_or(true, |g0| **g > *g0)).count();
                    let capture = sm.to() == t;
                    // ...and the king counts as a second target: a move that opens a new path onto their king (a check
                    // available next move) is a double threat (2026-10-09: Rf5 wins the c5 bishop because Black must
                    // answer the threat to the h5 king first)
                    let king_threat = b.king_of(them).map_or(false, |k| {
                        let n0 = attack_paths(pos, us, Bitboard::from_square(k)).iter().filter(|p| p.kind == "direct" || p.kind == "discovery").count();
                        let n1 = pass(&q).map_or(0, |p| attack_paths(&p, us, Bitboard::from_square(k)).iter().filter(|p| p.kind == "direct" || p.kind == "discovery").count());
                        n1 > n0
                    });
                    // (a capture elsewhere is a gain in hand: fxe5 attacking the f6 rook is a double threat)
                    if !capture && !sm.is_capture() && !q.is_check() && !king_threat && new < 2 && escapes(&q, us, t).any() {
                        trace.push(format!("    {}: a lone attack on a piece that can step away, no check, no second target, no new path onto the king -- no proof (the lone-attack lemma)", san(pos, s)));
                        tally(|st| st.lone += 1);
                        failed.push(s.clone());
                        continue;
                    }
                }
            }
            // THE GOAL OF A SEED is what it can win, not the target's full value (2026-10-09: e5 against a pinned rook
            // gains 4 -- the pawn may fall -- and a fork wins the second piece): the static gain on the target after the
            // move (the exchange's first term), for two or more new targets the second best, for a capture the gain now
            let g = if king { goal } else if let Some(sm) = s.parse::<shakmaty::uci::UciMove>().ok().and_then(|u| u.to_move(pos).ok()) {
                let q = after(pos, &sm);
                let now = rules::threats_of(&q, us);
                let mut new: Vec<i64> = now.iter().filter(|(x, gg)| **gg > 0 && before.get(x).map_or(true, |g0| **gg > *g0)).map(|(_, gg)| *gg).collect();
                new.sort_by(|a, b| b.cmp(a));
                if sm.to() == t { before.get(&(t as u8)).copied().unwrap_or(goal).max(1) }
                else if new.len() >= 2 { new[1].max(1) }
                else { now.get(&(t as u8)).copied().filter(|x| *x > 0).unwrap_or(goal).max(1) }
            } else { goal };
            let (ok, n) = with_tag("seed".to_string(), || mprove(pos, s, &[t], g));
            proofs += 1; nodes += n;
            trace.push(format!("    prove {} (gain {g}): {} ({n} nodes)", san(pos, s), if ok { "HOLDS" } else { "fails" }));
            if ok {
                wins.push((s.clone(), t, g));
                MEMO.with(|m| if let Some(m) = m.borrow_mut().as_mut() { *m.history.entry(hk(s)).or_insert(0.0) += 1.0; });
                if king { won_outright = true; }                         // (review: a fork's second target may be worth more than this target -- only a mate cannot be beaten)
            }
            if ok { if best.as_ref().map_or(true, |x| g > x.2) { best = Some((String::new(), s.clone(), g, format!("{} wins {g} against the {name}", san(pos, s)))); best_target = Some(t); } }
            else { failed.push(s.clone()); }
        }
        armed.push((goal, t, failed));
    }
    // OVERLOAD: their overloaded pieces -- take on the first job, the guard is drawn off the second; one proof each,
    // focused on both jobs (pawns included: Kaveh's bishop guarding two pawns)
    for (d, xa, xb, loss, why) in if OVERLOAD.load(Ordering::Relaxed) { overloads(pos, them) } else { vec![] } {
        if best.as_ref().map_or(false, |x| x.0.is_empty() && x.2 >= loss) { continue; }
        let Some(cap) = pos.legal_moves().into_iter().filter(|m| m.to() == xa)
            .min_by_key(|m| kernel_value(m.role())) else { continue };
        let u = cap.to_uci(CastlingMode::Standard).to_string();
        let (ok, n) = mprove(pos, &u, &[xa, xb], loss); proofs += 1; nodes += n;
        trace.push(format!("overload: {why} -- {} wins {loss}? {} ({n} nodes)", san(pos, &u), if ok { "HOLDS" } else { "fails" }));
        if ok && best.as_ref().map_or(true, |x| loss > x.2) {
            let role = format!("{:?}", b.piece_at(d).map(|p| p.role).unwrap_or(Role::Pawn)).to_lowercase();
            best = Some((String::new(), u.clone(), loss, format!("{} exploits the overloaded {role} on {d} (wins {loss})", san(pos, &u))));
            best_target = Some(xa);
            wins.push((u, xa, loss));
        }
    }
    // THE PROMOTION LEMMA: a move after which we have an unstoppable passer and they have none faster (no search)
    if best.as_ref().map_or(true, |x| x.2 < 8) {
        let mut cand: Option<(String, u32, Square)> = None;
        // (the Stockfish audit of 2026-10-10, 29 of 210 claims not winning, each an "unless": the move must not lose
        // material by itself; they must have NO unstoppable passer of their own; their immediate tactics after the move
        // must be worth less than a queen)
        for m in pos.legal_moves() {
            let q = after(pos, &m);
            // promoting now counts as queening in 0 (2026-10-10 audit: with f8=Q on the board it chose a4)
            // (the lemma's domain holds for it too: the opponent has no rook or queen -- audit: c8=Q against a queen and a mate)
            let no_heavy = (b.by_color(them) & (b.by_role(Role::Rook) | b.by_role(Role::Queen))).is_empty();
            // ...and ONE PROOF that the promotion wins a rook's worth against every reply (audit: the new queen taken or
            // forked -- h8=Q met by the f7 knight)
            let now = if m.promotion() == Some(Role::Queen) && no_heavy {
                let u = m.to_uci(CastlingMode::Standard).to_string();
                let (ok, n) = mprove(pos, &u, &[m.to()], 5); proofs += 1; nodes += n;
                if ok { Some((m.to(), 0u32)) } else { continue }
            } else { None };
            let Some((p, n)) = now.or_else(|| unstoppable(&q, us)) else { continue };
            // THE STALEMATE GUARD, in the order the proved lemmas give (Ideas.StalemateLemmas): a check never stalemates
            // (free); otherwise a legal move of theirs rules it out; a move that stalemates them is a draw, not a win
            if !q.is_check() && q.legal_moves().is_empty() { continue; }
            if unstoppable_k(&q, them, false).is_some() { continue; }       // a race (our king is busy): decided by the rest of the material
            BUDGET.store(200_000, Ordering::Relaxed);
            let lost = settle_d(&q, -2000, 2000, false, SETTLE_DEPTH).max(0) - (kernel_material(&q, us) - kernel_material(pos, us)).max(0);
            BUDGET.store(u64::MAX, Ordering::Relaxed);
            if lost > 0 { continue; }                                      // the move gives material away
            if cand.as_ref().map_or(true, |c| n < c.1) { cand = Some((m.to_uci(CastlingMode::Standard).to_string(), n, p)); }
        }
        // their immediate tactics after the chosen move (one analysis)
        if let Some((mv, _, _)) = cand.clone() {
            if let Some(m) = mv.parse::<shakmaty::uci::UciMove>().ok().and_then(|u| u.to_move(pos).ok()) {
                let q = after(pos, &m);
                if !immediate_only {
                    let tp = plan_inner(&q, true);
                    proofs += tp.proofs; nodes += tp.nodes;
                    if tp.best.as_ref().map_or(false, |x| x.2 >= 8) {
                        trace.push(format!("the promotion lemma's {} refused: they then have {}", san(pos, &mv), tp.best.unwrap().3));
                        cand = None;
                    }
                }
            }
        }
        if let Some((mv, n, p)) = cand {
            trace.push(format!("the promotion lemma: after {} the pawn on {p} queens in {n} (the rule of the square; no knight in reach, no bishop of its path colour)", san(pos, &mv)));
            best = Some((String::new(), mv.clone(), 8, format!("{} makes an unstoppable passer (queens in {n})", san(pos, &mv))));
            best_target = Some(p);
            wins.push((mv, p, 8));
        }
    }
    if immediate_only { let im = best.clone(); return Plan { trace, best, proofs, nodes, target: best_target, immediate: im, wins }; }
    // 3b DEFENCE: their immediate win is bigger than ours -- the candidate defences are read off their winning move's
    // geometry (the path lemma): take the attacking piece, move the target, guard the square it lands on, step onto a
    // line of theirs into the target's zone, free an escape next to the target, or check; one proof each: does their
    // move still win after ours?
    let ours_now = best.as_ref().map_or(0, |x| x.2);
    let immediate = best.clone();
    if let Some((tseed, tt, tg)) = their.clone() {
        // DOES OUR ATTACK ALREADY ANSWER THEIRS? (Kaveh 2026-10-09; the census: defending first lost 6% of the immediate
        // wins) -- one proof: their winning move after our best one
        // A CERTIFIED WIN ALREADY ANSWERS A MATERIAL THREAT: its proof searched every reply of theirs, their captures
        // included, and resolved the captures left at each leaf -- so their capture threat is inside its gain. Only a
        // MATE threat lies outside it (the leaf counts captures, not mates) and gets the check below. (2026-10-09:
        // re-proving their material threat after our win made the planner defend in 71 positions; Stockfish preferred
        // the win in 51.)
        let answered = match &immediate {
            Some(_) if tg < MATE_GOAL => true,
            Some(x) => {
                let q = x.1.parse::<shakmaty::uci::UciMove>().ok().and_then(|u| u.to_move(pos).ok()).map(|m| after(pos, &m));
                match q {
                    Some(q) if q.is_checkmate() => true,
                    Some(q) => {
                        // every winning move of theirs worth as much must be stopped (2026-10-09 audit), and our move must
                        // expose nothing worth as much
                        let threats: Vec<(String, Square, i64)> = if their_wins.is_empty() { vec![(tseed.clone(), tt, tg)] }
                            else { their_wins.iter().filter(|w| w.2 >= tg).cloned().collect() };
                        let bm = x.1.parse::<shakmaty::uci::UciMove>().ok().and_then(|u| u.to_move(pos).ok());
                        // (the square our capturing piece lands on is left out: its recapture is already in our gain --
                        // 2026-10-09: "Nxd1 wins 2" was refused because the knight on d1 could be taken back)
                        let mut stops = bm.as_ref().map_or(true, |bm| exposes(pos, bm).iter().all(|(x, g)| *g < tg || *x == bm.to()));
                        if stops {
                            for (ws, wt, wg) in &threats {
                                let legal = ws.parse::<shakmaty::uci::UciMove>().ok().and_then(|u| u.to_move(&q).ok()).is_some();
                                // (review: their move illegal only because our move gives CHECK is not answered -- the
                                // threat returns after they answer the check)
                                if !legal { if q.is_check() { stops = false; break; } continue; }
                                let t2 = match &bm { Some(bm) if bm.from() == Some(*wt) => bm.to(), _ => *wt };
                                let (ok, n) = with_tag(format!("ans:{}", x.1), || mprove(&q, ws, &[t2], *wg)); proofs += 1; nodes += n;
                                if ok { stops = false; break; }
                            }
                        }
                        stops
                    }
                    None => false,
                }
            }
            None => false,
        };
        if answered { trace.push(format!("our {} also stops their threat", best.as_ref().unwrap().3)); }
        // TAKE OR DEFEND (2026-10-09, checked against Stockfish: of 99 positions where the planner defended instead of
        // taking an immediate win, the win was better in 81): if our win does not stop their threat, both happen -- we
        // defend only when what we win is worth less than what they win (net negative)
        let trade_ok = immediate.is_some() && ours_now >= tg;
        if !answered && trade_ok && immediate.is_some() {
            trace.push(format!("our {} is worth at least their threat ({tg}): take it", best.as_ref().unwrap().3));
        }
        if tg >= 2 && tg > ours_now && !answered && !trade_ok {
            let p0 = pass(pos).unwrap();
            let sm = tseed.parse::<shakmaty::uci::UciMove>().ok().and_then(|u| u.to_move(&p0).ok());
            if let Some(sm) = sm {
                let from = sm.from();
                let land = sm.to();
                let mut lines = Bitboard::EMPTY;                       // their lines into the target's zone
                let zone = shakmaty::attacks::king_attacks(tt).with(tt);
                for p in attack_paths(pos, them, zone) {
                    if matches!(p.kind, "open" | "discovery" | "battery" | "direct") { lines |= shakmaty::attacks::between(p.piece, p.target); if let Some(r) = p.rear { lines.add(r); } }
                }
                let mut cands: Vec<Move> = Vec::new();
                for m in pos.legal_moves() {
                    if refused(&m).is_some() { continue; }
                    let q = after(pos, &m);
                    let hit = Some(m.to()) == from || m.from() == Some(tt) || q.board().attacks_to(land, us, q.board().occupied()).any()
                        && !b.attacks_to(land, us, b.occupied()).any()
                        || lines.contains(m.to()) || m.from().map_or(false, |f| shakmaty::attacks::king_attacks(tt).contains(f)) || q.is_check();
                    if hit { cands.push(m); }
                }
                trace.push(format!("their {} wins {tg}: {} candidate defences on its geometry", san(&p0, &tseed), cands.len()));
                if let Some(slider) = sm.from() { if let Some(lp) = line_pair(pos, slider, tt) { trace.push(format!("    {lp}")); } }
                for m in cands {
                    let q = after(pos, &m);
                    let mu = m.to_uci(CastlingMode::Standard).to_string();
                    // a defence must stop EVERY winning move of theirs worth as much -- and must not EXPOSE something worth
                    // as much (the vacated-square clause: the bishop leaving d6 opens the rook's line onto the d7 knight);
                    // an exposure is confirmed with one proof of their capture
                    let mut still = false;
                    for (x, g) in exposes(pos, &m) {
                        if g < tg { continue; }
                        let cap = q.legal_moves().into_iter().find(|c| c.to() == x).map(|c| c.to_uci(CastlingMode::Standard).to_string());
                        if let Some(cap) = cap {
                            let (ok, n) = with_tag(format!("exp:{mu}"), || mprove(&q, &cap, &[x], g.min(tg))); proofs += 1; nodes += n;
                            if ok { trace.push(format!("    {}: exposes {x} (their {} wins it)", san(pos, &mu), san(&q, &cap))); still = true; break; }
                        }
                    }
                    let threats: Vec<(String, Square, i64)> = if their_wins.is_empty() { vec![(tseed.clone(), tt, tg)] }
                        else { their_wins.iter().filter(|w| w.2 >= tg).cloned().collect() };
                    for (ws, wt, wg) in &threats {
                        if still { break; }
                        let t2 = if m.from() == Some(*wt) { m.to() } else { *wt };
                        let legal = ws.parse::<shakmaty::uci::UciMove>().ok().and_then(|u| u.to_move(&q).ok()).is_some();
                        if !legal { continue; }
                        let (ok, n) = with_tag(format!("def:{mu}"), || mprove(&q, ws, &[t2], *wg)); proofs += 1; nodes += n;
                        if ok { still = true; break; }
                    }
                    if !still {
                        trace.push(format!("    {} stops it", san(pos, &mu)));
                        if best.as_ref().map_or(true, |x| x.0 != "defend") {
                            best = Some(("defend".to_string(), mu.clone(), -tg, format!("{} stops their {}", san(pos, &mu), san(&p0, &tseed))));
                        }
                    }
                }
                if best.as_ref().map_or(true, |x| x.0 != "defend") {
                    // NO DEFENCE: the cheapest way to lose -- the line-pair lemma when the frame is a line pair (no search),
                    // else alpha-beta in the frame (the fallback)
                    // THE SEARCH DECIDES; the line-pair lemma's prediction is recorded beside it as EXPERIENCE until it is
                    // verified (2026-10-09, line_pair_verify: 21 of 25 where it applies -- not yet a rule that skips search)
                    let lp = sm.from().and_then(|sl| back_piece(pos, sl, tt).filter(|_| line_pair(pos, sl, tt).is_some())
                        .and_then(|r| line_pair_lemma(pos, sl, tt, r)));
                    if let Some((mv, net, why)) = &lp {
                        trace.push(format!("no defence stops their {}: the line-pair lemma predicts {} nets {net} ({why}) -- checked by the search below", san(&p0, &tseed), san(pos, mv)));
                    }
                    {
                        let mut line = Bitboard::EMPTY;
                        if let Some(f0) = sm.from() { line = shakmaty::attacks::between(f0, tt); line.add(f0); }
                        let fb = frame_fallback(pos, &[tt], line, 50_000);
                        if let Some((mv, v)) = fb.first().cloned() {
                            trace.push(format!("no defence stops their {}: alpha-beta in the frame ({} first moves) -- {} keeps {v}{}", san(&p0, &tseed), fb.len(), san(pos, &mv),
                                match &lp { Some((_, net, _)) if *net == v => " (the lemma agreed)".to_string(), Some((_, net, _)) => format!(" (the lemma predicted {net})"), None => String::new() }));
                            best = Some(("cheapest".to_string(), mv.clone(), v, format!("{} is the cheapest way to lose (frame search, {v})", san(pos, &mv))));
                        }
                    }
                }
                return Plan { trace, best, proofs, nodes, target: best_target, immediate, wins };
            }
        }
    }
    // 4 ARMING, only when no immediate win was found and a preparing move is affordable -- by value, stopping at the first
    // armed threat: the escapes each failing seed leaves, and our one-move paths onto them; a pair is proved only if the
    // preparing move, then the seed, leaves the target at most one escape (Kaveh: "how do we cover all three?")
    if best.is_none() && tempo {
      for (goal, t, failed) in armed {
        if best.is_some() { break; }
        let name = format!("{:?} on {t}", b.piece_at(t).unwrap().role).to_lowercase();
        for s in &failed {
            let Ok(u) = s.parse::<shakmaty::uci::UciMove>() else { continue };
            let Ok(sm) = u.to_move(pos) else { continue };
            let q = after(pos, &sm);
            let esc = escapes(&q, us, if sm.from() == Some(t) { sm.to() } else { t });
            let zone = esc.with(t);
            let mut arms: Vec<String> = Vec::new();
            for p in attack_paths(pos, us, zone) {
                let ok_kind = match p.kind {
                    "battery" => p.via == sm.from(),                       // loaded behind the seed's own piece
                    "direct" => p.piece != sm.from().unwrap_or(t),         // another piece covering an escape
                    _ => false,
                };
                if ok_kind { if let Some(m) = p.mv { if m != *s && !arms.contains(&m) { arms.push(m); } } }
            }
            if arms.is_empty() { continue; }
            trace.push(format!("    {} leaves escapes {:?}: preparing moves on a path onto them: {}", san(pos, s),
                esc.into_iter().map(|e| e.to_string()).collect::<Vec<_>>(), arms.iter().map(|u| san(pos, u)).collect::<Vec<_>>().join(", ")));
            for a in &arms {
                let Ok(au) = a.parse::<shakmaty::uci::UciMove>() else { continue };
                let Ok(am) = au.to_move(pos) else { continue };
                if refused(&am).is_some() { continue; }
                let q = after(pos, &am);
                if q.is_check() { continue; }
                let Some(p) = pass(&q) else { continue };
                let Some(sm2) = s.parse::<shakmaty::uci::UciMove>().ok().and_then(|u| u.to_move(&p).ok()) else { continue };
                let q2 = after(&p, &sm2);
                let t2 = if sm2.from() == Some(t) { sm2.to() } else { t };
                // two or more escapes left open: no proof (one may still be chased -- Ne7+ Kb8 Rxd8#)
                if escapes(&q2, us, t2).count() >= 2 { tally(|st| st.escape_skip += 1); continue; }
                let (ok, n) = with_tag(format!("arm:{a}"), || mprove(&p, s, &[t], goal));
                proofs += 1; nodes += n;
                trace.push(format!("        prove {} then {} (they pass): {} ({n} nodes)", san(pos, a), san(&p, s), if ok { "HOLDS -- an armed threat" } else { "fails" }));
                if ok && best.as_ref().map_or(true, |x| goal > x.2) {
                    best = Some((a.clone(), s.clone(), goal, format!("{} arms {} against the {name}", san(pos, a), san(&p, s))));
                }
            }
        }
      }
    }
    Plan { trace, best, proofs, nodes, target: best_target, immediate, wins }
}

/// threats_after(fen, uci) -> every immediate win (seed, target square, gain) the planner proves after the move if the
/// opponent passes -- the immediate phase (seeds, the lone-attack lemma, skewers, overloads, passers, mates) with the
/// stop rule off, so a second target after a first win is still looked at; and the ms it took. (Kaveh 2026-10-10 on the
/// lone-attack lemma's surprises: "maybe it's not the lone attack lemma that's lacking, but bringing in the other
/// lemmas" -- does the reasoning we have already see a second threat?)
#[pyfunction]
pub fn threats_after(fen: String, uci: String) -> PyResult<(Vec<(String, String, i64)>, f64)> {
    let Some(pos) = parse(&fen) else { return Ok((vec![], 0.0)) };
    let Some(m) = uci.parse::<shakmaty::uci::UciMove>().ok().and_then(|u| u.to_move(&pos).ok()) else { return Ok((vec![], 0.0)) };
    let t = std::time::Instant::now();
    let Some(p) = pass(&after(&pos, &m)) else { return Ok((vec![], 0.0)) };
    NO_STOP.with(|c| c.set(true));
    let tp = plan_inner(&p, true);
    NO_STOP.with(|c| c.set(false));
    Ok((tp.wins.iter().map(|(s, t, g)| (s.clone(), t.to_string(), *g)).collect(), t.elapsed().as_secs_f64() * 1000.0))
}

/// plan_trace(fen) -> (trace lines, the decision (preparing move / "defend" / "", seed, gain, summary), proofs, nodes, ms,
/// the best immediate win found (whatever was decided))
#[pyfunction]
pub fn plan_trace(fen: String) -> PyResult<(Vec<String>, Option<(String, String, i64, String)>, u32, u64, f64, Option<(String, String, i64, String)>)> {
    let Some(pos) = parse(&fen) else { return Ok((vec![], None, 0, 0, 0.0, None)) };
    let t = std::time::Instant::now();
    let p = plan(&pos);
    Ok((p.trace, p.best, p.proofs, p.nodes, t.elapsed().as_secs_f64() * 1000.0, p.immediate))
}


/// THE PLAN STATE OF A GAME: the frame memory and the move history, kept across moves. `observe(fen)` plans the side to
/// move with the memory installed -> (trace, decision, proofs asked, nodes searched, ms, best immediate win, and this
/// move's (fresh searches, replays, exact reuses, reused failures, re-proofs, lone-attack skips, frozen moves, escape
/// skips, moves cut by the history order)).
#[pyclass(unsendable)]
pub struct PlanState { memo: Option<Memo> }

type StatRow = (u32, u32, u32, u32, u32, u32, u32, u32, u32);

#[pymethods]
impl PlanState {
    #[new]
    fn new() -> Self { PlanState { memo: Some(Memo::default()) } }

    fn observe(&mut self, fen: String) -> PyResult<(Vec<String>, Option<(String, String, i64, String)>, u32, u64, f64, Option<(String, String, i64, String)>, StatRow)> {
        let Some(pos) = parse(&fen) else { return Ok((vec![], None, 0, 0, 0.0, None, (0, 0, 0, 0, 0, 0, 0, 0, 0))) };
        let mut memo = self.memo.take().unwrap_or_default();
        memo.stats = Stats::default();
        for v in memo.history.values_mut() { *v *= 0.8; }                    // recent wins weigh more
        MEMO.with(|m| *m.borrow_mut() = Some(memo));
        let t = std::time::Instant::now();
        let p = plan(&pos);
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        let memo = MEMO.with(|m| m.borrow_mut().take()).unwrap_or_default();
        let s = memo.stats;
        self.memo = Some(memo);
        Ok((p.trace, p.best, p.proofs, s.nodes, ms, p.immediate,
            (s.fresh, s.replay, s.reuse_exact, s.reuse_fail, s.reprove, s.lone, s.frozen, s.escape_skip, s.cut_by_history)))
    }

    /// Entries in the memory.
    fn size(&self) -> usize { self.memo.as_ref().map_or(0, |m| m.entries.values().map(|v| v.len()).sum()) }
}
