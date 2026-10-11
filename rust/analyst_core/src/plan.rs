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
use crate::threat::{any_new_threat, jobs_with, new_threats, threats, Threat};
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
struct Entry { board: Vec<u8>, extra: String, holds: bool, hints: std::sync::Arc<Hints>, support: Bitboard,
               // a FRAME PROOF's dependency (lib.rs FRAME_PROOFS): the squares it read, whether it read the whole
               // board, whether its budget ran out (unknown, never reused), the index it read at the root
               dep: Bitboard, global: bool, spent: bool, idx: Option<Box<(rules::ThrArr, rules::ThrArr)>> }

#[derive(Default, Clone, Copy)]
pub struct Stats { pub fresh: u32, pub replay: u32, pub reuse_exact: u32, pub reuse_fail: u32, pub reprove: u32,
                   pub nodes: u64, pub lone: u32, pub frozen: u32, pub escape_skip: u32, pub cut_by_history: u32,
                   pub reuse_zone: u32 }

#[derive(Default)]
pub struct Memo { entries: HashMap<String, Vec<Entry>>, pub history: HashMap<String, f32>, pub stats: Stats,
                  ledger: Option<crate::ledger::Ledger>, events: Vec<crate::ledger::Event>,
                  pool: Vec<crate::waiting::Frame>, pool_ply: u32, solution: Option<Solution> }

/// A PROVEN MATE being played out (Kaveh 2026-10-10: "anytime we find a path to mate, we should play that ... you've found
/// it, you've proven it, so you can just use it. There's no more search needed"): the proof's stored moves (every reply
/// of theirs was searched, our answer to each recorded, keyed by the path of moves), where we are in it, and the board
/// after our last move from it.
#[derive(Clone)]
pub struct Solution { owner: Color, hints: std::sync::Arc<Hints>, path: u64, after_ours: Chess }
const SNAPSHOTS: usize = 8;
pub static REUSE_FAILS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
// ZONE REUSE (Kaveh 2026-10-11: "we don't want to research anything"): a frame proof is reused, held or failed, when
// no square that bears on what it read changed (Lean Ideas.FrameLemma.frame_unchanged) and the index it read at the
// root is the same; OFF until the equality audit (plan_memo_check) passes.
pub static ZONE_REUSE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
#[pyfunction]
pub fn set_zone_reuse(flag: bool) -> PyResult<()> { ZONE_REUSE.store(flag, Ordering::Relaxed); Ok(()) }
// THE FRAME AUDIT: every frame proof answered again by the exact search; [agree, false claims (frame holds, exact
// fails), missed (frame fails, exact holds)] since the last read.
pub static FRAME_AUDIT: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static FRAME_AUDIT_N: [std::sync::atomic::AtomicU64; 6] = [const { std::sync::atomic::AtomicU64::new(0) }; 6];
pub(crate) fn frame_audit_count(i: usize) { FRAME_AUDIT_N[i].fetch_add(1, Ordering::Relaxed); }
#[pyfunction]
pub fn set_frame_audit(flag: bool) -> PyResult<()> { FRAME_AUDIT.store(flag, Ordering::Relaxed); Ok(()) }
#[pyfunction]
pub fn frame_audit_stats() -> Vec<u64> { FRAME_AUDIT_N.iter().map(|a| a.swap(0, Ordering::Relaxed)).collect() }
/// The squares that bear on any square of `d` (Lean Ideas.FrameLemma.reachB: the square, a line through it, a
/// knight's move from it): a change off this set cannot change what was read on `d`.
fn bearing(d: Bitboard) -> Bitboard {
    let mut out = Bitboard::EMPTY;
    for t in d { out |= shakmaty::attacks::queen_attacks(t, Bitboard::EMPTY) | shakmaty::attacks::knight_attacks(t) | shakmaty::attacks::king_attacks(t); out.add(t); }
    out
}
#[pyfunction]
pub fn set_reuse_fails(flag: bool) -> PyResult<()> { REUSE_FAILS.store(flag, Ordering::Relaxed); Ok(()) }                                               // boards kept per question (the oldest dropped)

thread_local! {
    static MEMO: RefCell<Option<Memo>> = const { RefCell::new(None) };
    static TAG: RefCell<Vec<String>> = const { RefCell::new(Vec::new()) };
    static NO_STOP: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };   // threats_after: every target, no stop rule
    // their reasoning after a move, once per top-level plan (2026-10-10: the same move was valued up to four times --
    // the wins' ranking, the defences, the best net, the audit)
    static NET_CACHE: RefCell<HashMap<String, (i64, Option<String>, u32, u64)>> = RefCell::new(HashMap::new());
    // the moves whose win came from the FRAME fallback ("certainly nets v in its frame"), not a full proof
    static FRAMED: RefCell<std::collections::HashSet<String>> = RefCell::new(std::collections::HashSet::new());
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

fn question_key(p: &Chess, seed: &str, focus: &[Square], goal: i64) -> String {
    format!("{}|{}|{}|{}|{}", seed, focus.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(","), goal,
            if p.turn() == Color::White { "w" } else { "b" }, crate::waiting::DEEP_CONDITIONS.load(Ordering::Relaxed))
}

/// The stored proof (hints) of a question asked on exactly this board, if the memory holds one that held.
fn stored_proof(p: &Chess, seed: &str, focus: &[Square], goal: i64) -> Option<std::sync::Arc<Hints>> {
    let key = question_key(p, seed, focus, goal);
    let (board, extra) = snapshot(p);
    MEMO.with(|m| m.borrow().as_ref().and_then(|m| m.entries.get(&key).and_then(|v| v.iter().rev()
        .find(|e| e.board == board && e.extra == extra && e.holds).map(|e| e.hints.clone()))))
}

/// `prove_n`, and under the FRAME AUDIT the same question answered by the exact search, the two compared.
fn prove_audited(p: &Chess, seed: &str, focus: &[Square], goal: i64, hints: Option<std::sync::Arc<Hints>>) -> (bool, Hints, u64, bool) {
    let out = prove_n(p, seed, focus, goal, hints);
    if FRAME_AUDIT.load(Ordering::Relaxed) && crate::frame_on() {
        let (ok_x, _, _, _) = crate::with_frame_off(|| prove_n(p, seed, focus, goal, None));
        FRAME_AUDIT_N[if out.0 == ok_x { 0 } else if out.0 { 1 } else { 2 }].fetch_add(1, Ordering::Relaxed);
    }
    out
}

/// A proof question through the memory (or straight to the search when no memory is installed): (holds, nodes searched).
fn mprove(p: &Chess, seed: &str, focus: &[Square], goal: i64) -> (bool, u64) {
    let installed = MEMO.with(|m| m.borrow().is_some());
    if !installed { let (ok, _, n, _) = prove_audited(p, seed, focus, goal, None); return (ok, n); }
    // the QUESTION alone is the key -- where it was asked does not matter, the board comparison decides (2026-10-09: keyed
    // by where it was asked, White's null-move analysis of Black's threat never served Black's own analysis a move later)
    let key = question_key(p, seed, focus, goal);
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
            if ZONE_REUSE.load(Ordering::Relaxed) && crate::frame_on() && !e.global && !e.spent && (diff & bearing(e.dep)).is_empty() {
                if let Some(ix) = &e.idx {
                    if ix.0 == rules::threats_arr(p, p.turn()) && ix.1 == rules::threats_arr(p, !p.turn()) { tally(|s| s.reuse_zone += 1); return (e.holds, 0); }
                }
            }
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
    let _ = crate::take_dep();
    let (ok, rec, n, spent) = prove_audited(p, seed, focus, goal, hints);
    let (dep, global) = crate::take_dep();
    let idx = if crate::frame_on() { Some(Box::new((rules::threats_arr(p, p.turn()), rules::threats_arr(p, !p.turn())))) } else { None };
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
        v.push(Entry { board, extra, holds: ok, hints: std::sync::Arc::new(rec), support, dep, global, spent, idx });
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
    // the generic threat (Kaveh 2026-10-10: losing the king is a threat like losing a piece): a mate in one our move
    // newly allows exposes the king
    if let Some(k) = q.board().king_of(us) {
        let mate_now = |p: &Chess| threats(p, them).iter().any(|t| t.mate);
        if mate_now(&q) && !mate_now(pos) { out.push((k, MATE_GOAL)); }
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
    let before = threats(pos, us);
    for m in pos.legal_moves() {
        let Some(from) = m.from() else { continue };
        if from != f && from != r { continue; }
        // THE EXCHANGE LAW prices it (2026-10-09: "capture minus their first capture" agreed with the search on 50% of
        // the pair's own moves -- the rest of the exchange matters): the material the move takes, then their whole
        // capture sequence resolved (settle: either side may stop)
        let q = after(pos, &m);
        let now_t = threats(&q, us);
        let new_threat = new_threats(&now_t, &before).iter().any(|x| x.target != m.to());
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
/// back rank, a bishop cannot"). Made generic the same day ("protecting against the threat should be generic and not
/// specifically a piece"): a JOB of `side`'s piece d is any threat of the attacker that appears when d is lifted
/// (threat.rs: a piece it guards, a line it blocks, a MATE it covers). A piece with two jobs is OVERLOADED when the
/// attacker takes on the first job's square, d recaptures there, and the second job's threat then stands -- checked by
/// playing the two moves, not by geometry. The loss is value arithmetic: the attacker picks the order, the defender
/// picks whether to recapture (never into a mate). -> [(d, first job taken, second job's target, loss, explanation)].
fn overloads(pos: &Chess, side: Color) -> Vec<(Square, Square, Square, i64, String)> {
    let atk = !side;
    let b = pos.board();
    let Some(ap) = (if pos.turn() == atk { Some(pos.clone()) } else { pass(pos) }) else { return vec![] };
    let now = threats(pos, atk);
    let desc = |j: &Threat| if j.mate { format!("the mate {}", j.how.as_deref().unwrap_or("on its king")) } else { format!("{}", j.target) };
    let mut out = Vec::new();
    for d in b.by_color(side) {                                   // (a Bitboard iterates in square order: deterministic)
        let Some(dp) = b.piece_at(d) else { continue };
        if dp.role == Role::King { continue; }
        let js = jobs_with(pos, d, &now);
        if js.len() < 2 { continue; }
        let mut worst: Option<(Square, Square, i64, String)> = None;
        for j1 in js.iter().filter(|j| !j.mate) {
            let xa = j1.target;
            let Some(a) = ap.legal_moves().into_iter().filter(|m| m.to() == xa).min_by_key(|m| kernel_value(m.role())) else { continue };
            let q1 = after(&ap, &a);
            let Some(rec) = q1.legal_moves().into_iter().find(|m| m.from() == Some(d) && m.to() == xa) else { continue };
            let q2 = after(&q1, &rec);
            let t2 = threats(&q2, atk);
            for j2 in js.iter().filter(|j| j.target != xa) {
                let stands = t2.iter().find(|t| t.mate == j2.mate && (j2.mate || t.target == j2.target));
                let Some(st) = stands else { continue };
                let (va, vxa) = (kernel_value(a.role()), b.piece_at(xa).map_or(0, |p| kernel_value(p.role)));
                let loss = if j2.mate { vxa } else { vxa.min(vxa - va + st.value) };
                if loss > 0 && worst.as_ref().map_or(true, |w| loss > w.2) {
                    let why = format!("the {:?} on {d} has two jobs, {} and {}; after a capture on {xa} and its recapture there, {} stands{}",
                        dp.role, desc(j1), desc(j2), desc(j2),
                        if dp.role == Role::Bishop && !j2.mate && (xa.file() as u32 + xa.rank() as u32) % 2 != (j2.target.file() as u32 + j2.target.rank() as u32) % 2 { " (a bishop keeps its colour)" } else { "" }).to_lowercase();
                    worst = Some((xa, j2.target, loss, why));
                }
            }
        }
        if let Some((xa, xb, loss, why)) = worst { out.push((d, xa, xb, loss, why)); }
    }
    out
}

pub static OVERLOAD: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);
/// THE WAITING-FRAMES POOL IN THE PLANNER (Kaveh 2026-10-10: wire the existing pool -- waiting.rs, 2026-10-09 -- rather
/// than rebuild it): off by default until measured (it costs ~25-35 ms a position against the planner's ~14).
pub static POOL: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
#[pyfunction]
pub fn set_pool(flag: bool) -> PyResult<()> { POOL.store(flag, Ordering::Relaxed); Ok(()) }
/// The pool's SEEDS without the pool (measured 2026-10-10, pool_check.py, 100 games: the pool's active frames claiming 2+
/// were as good as the planner's own wins -- median 0 cp from Stockfish's best, 8% losing 150+ -- those claiming 1 pawn
/// were not, 11%; its armed decisions lost 150+ in 35%; and the pool cost 6x): waiting::seeds (forks, cornering, single
/// threats) with a goal of 2+, each proved like a seed of our own.
pub static FRAME_SEEDS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);
#[pyfunction]
pub fn set_frame_seeds(flag: bool) -> PyResult<()> { FRAME_SEEDS.store(flag, Ordering::Relaxed); Ok(()) }
/// THE BLUNDER CHECK, direct (the pool's "hands" warning flagged decisions losing 150+ in 52% against 17% unwarned --
/// the same question asked straight): after the decision, their immediate wins; reported in the trace.
/// (an AUDIT since the planner chooses every answer by its net -- 2026-10-10: it then found 2 "certain" blunders in 7,876
/// positions, neither confirmed by Stockfish; off by default)
pub static AFTER_CHECK: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
#[pyfunction]
pub fn set_after_check(flag: bool) -> PyResult<()> { AFTER_CHECK.store(flag, Ordering::Relaxed); Ok(()) }
/// GRADING THE WARNING (Kaveh 2026-10-10: "is it a probable blunder or a certain blunder? If probable, we want to
/// ascertain certainty; if certain, we want to not do it"): the warning is a certified win of theirs after our move; it
/// is a CERTAIN blunder only if another move keeps them lower (net of what each move wins us) -- then that move is
/// played; UNAVOIDABLE if every move lets them as much; PROBABLE if the alternatives could not settle it.
pub static AVOID: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);
#[pyfunction]
pub fn set_avoid(flag: bool) -> PyResult<()> { AVOID.store(flag, Ordering::Relaxed); Ok(()) }
const AVOID_MARGIN: i64 = 2;
/// (off by default: decision_quality.py, 904 decisions -- with the gate 18.3% lose 150+ cp to Stockfish's best, without 15.8%;
/// it saves ~20% cpu. No blunders first.)
pub static CHANGE_GATE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
#[pyfunction]
pub fn set_change_gate(flag: bool) -> PyResult<()> { CHANGE_GATE.store(flag, Ordering::Relaxed); Ok(()) }
const AVOID_TOP: usize = 5;
#[pyfunction]
pub fn set_see_threats(flag: bool) -> PyResult<()> { crate::rules::SEE_THREATS.store(flag, Ordering::Relaxed); Ok(()) }
pub static STRICT_STOP: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);
#[pyfunction]
pub fn set_strict_stop(flag: bool) -> PyResult<()> { STRICT_STOP.store(flag, Ordering::Relaxed); Ok(()) }
pub static NUCLEUS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);
#[pyfunction]
pub fn set_nucleus(flag: bool) -> PyResult<()> { NUCLEUS.store(flag, Ordering::Relaxed); Ok(()) }
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

fn tb_of(tp: Plan) -> Option<(String, String, i64, String)> { tp.best }

/// What a move nets right now: what we certainly win with it (a proved immediate win of the plan, else the material it
/// captures) minus their best certified immediate win after it (their reasoning once).
/// THE TIE-BREAK between moves of equal net: PIECE ACTIVITY (Kaveh 2026-10-10: "whenever there is a tie, we should always
/// prioritize piece activity as a tiebreaker for now, until a strategic planning layer steps in"): the moved piece after
/// the move -- the squares it attacks that no cheaper enemy piece attacks, plus twice the enemy pieces it attacks; a pawn
/// move counts 0. (Measured on the 14 tie cases from three walked games: it prefers Stockfish's move in 11; a count of
/// threats ours-minus-theirs did 2 of 10 and moved the losses around -- that version is gone.)
pub static TIE_BREAK: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);
#[pyfunction]
pub fn set_tie_break(flag: bool) -> PyResult<()> { TIE_BREAK.store(flag, Ordering::Relaxed); Ok(()) }

pub static HARASS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);
#[pyfunction]
pub fn set_harass(flag: bool) -> PyResult<()> { HARASS.store(flag, Ordering::Relaxed); Ok(()) }

fn tie_score(pos: &Chess, m: &Move) -> i64 {
    if !TIE_BREAK.load(Ordering::Relaxed) { return 0; }
    let us = pos.turn();
    let q = after(pos, m);
    // material taken NOW first (2026-10-10, game 10000001 ply 70: Qh4 and Rxf6 both net 4; activity chose the quiet queen
    // move, Stockfish takes the rook) -- then the moved piece's activity
    let taken = (kernel_material(&q, us) - kernel_material(pos, us)).max(0);
    let qb = q.board();
    let s = m.to();
    let Some(p) = qb.piece_at(s) else { return 1000 * taken };
    if p.role == Role::Pawn { return 1000 * taken; }
    let vp = kernel_value(p.role);
    let mut score = 0i64;
    for t in shakmaty::attacks::attacks(s, p, qb.occupied()) {
        if qb.piece_at(t).map_or(false, |x| x.color != us) { score += 2; continue; }
        let cheaper = (qb.attacks_to(t, !us, qb.occupied()) & qb.by_color(!us)).into_iter()
            .any(|a| qb.piece_at(a).map_or(false, |x| x.role == Role::King || kernel_value(x.role) < vp));
        if !cheaper { score += 1; }
    }
    // THEIR TEMPO against it (Kaveh's "ability to act", their side -- 2026-10-10, game 10000000 ply 27: Qd5 centralised,
    // c4 / Ra6 then hunted the queen down; on 15 tie cases this keeps Stockfish's choice 12 times, 11 without): each move
    // of theirs after which a cheaper piece of theirs attacks the moved piece costs 2
    if !q.is_check() && HARASS.load(Ordering::Relaxed) {
        for r in q.legal_moves() {
            if r.to() == s { continue; }
            let q2 = after(&q, &r);
            if q2.board().piece_at(s) != Some(p) { continue; }
            let hits = shakmaty::attacks::attacks(r.to(), q2.board().piece_at(r.to()).unwrap_or(p), q2.board().occupied()).contains(s);
            let cheaper = q2.board().piece_at(r.to()).map_or(false, |x| kernel_value(x.role) < vp);
            if hits && cheaper { score -= 2; }
        }
    }
    1000 * taken + score
}

/// THE NET OF A MOVE (Kaveh 2026-10-10: "with a logic-based play there should be no blunders ... whatever the blunder
/// guard is doing should be part of the main planner's job"): the one value every decision is chosen by --
///   a quiet move with a proved win: its proved gain (the proof searched every reply)
///   any other move: the material it takes now minus their best proved immediate win after it (their reasoning once),
///                   and never less than a proved gain of its own (a capture's proof already counts the recapture --
///                   2026-10-10: Nxh1 "wins 2" was flagged because Rxh1 wins 3; 5 - 3 = 2)
fn net_after(pos: &Chess, m: &Move, wins: &[(String, Square, i64)], proofs: &mut u32, nodes: &mut u64) -> (i64, Option<String>) {
    let u = m.to_uci(CastlingMode::Standard).to_string();
    let q = after(pos, m);
    if q.is_checkmate() { return (MATE_GOAL, None); }
    let proved = wins.iter().filter(|w| w.0 == u).map(|w| w.2).max();
    let taken = kernel_material(&q, pos.turn()) - kernel_material(pos, pos.turn());
    // (a quiet move with a FULL proof: its value stands -- the proof searched every reply; a FRAME value goes through the
    // board below: 2026-10-11, Qc2 "wins 2" against the b1 rook in its frame, Rc1 then traps the queen)
    let framed_quiet = proved.is_some() && taken <= 0 && FRAMED.with(|f| f.borrow().contains(&u));
    if let Some(g) = proved { if (taken <= 0 && !framed_quiet) || g >= MATE_GOAL { return (g, None); } }
    // THE CHANGE FLAG (Kaveh 2026-10-10: "see how this change changes the state of another piece -- attacked twice and
    // defended twice, stable ... we remove a defender ... that change should flag the blunder"): their static threats
    // after the move against before it (the static exchange on every piece, promotion, trapped, mate by checks). No new
    // one -> the move's net is read statically (what it takes minus their biggest threat left); a new one -> their
    // reasoning certifies it. (change_flag_check.py: the flag catches 84% of the new wins their reasoning finds.)
    let them = !pos.turn();
    let after_t = threats(&q, them);
    let before_t = threats(pos, them);
    // (the flag includes their FORCING threats -- a fork next move, material by checks: the static count alone let 16% of
    // their new wins through and decisions losing 150+ rose from 15.4% to 18.6%)
    if CHANGE_GATE.load(Ordering::Relaxed) && crate::threat::any_new_threat(&q, them, &before_t) == Some(false) {
        let left = after_t.iter().map(|t| t.value).max().unwrap_or(0);
        return ((taken - left).max(proved.unwrap_or(i64::MIN)), None);
    }
    let key = shakmaty::fen::Fen::from_position(q.clone(), shakmaty::EnPassantMode::Legal).to_string();
    let cached = NET_CACHE.with(|c| c.borrow().get(&key).cloned());
    let (theirs, their_why) = match cached {
        Some((v, w, _, _)) => (v, w),
        None => {
            let tp = with_tag("net".to_string(), || plan_inner(&q, true));
            *proofs += tp.proofs; *nodes += tp.nodes;
            let mut v = tp.best.as_ref().map_or(0, |b| b.2);
            let mut w = tp.best.map(|b| b.3);
            // (their reasoning targets pieces worth 3+; our PAWNS they win by the static exchange count too -- 2026-10-10,
            // game 10000007 ply 56: "Na1 stops their Rxb2 (nets 0)", then Rexf2)
            let us_c = !q.turn();
            // (and any piece of ours by the static exchange -- 2026-10-10, game 10000000 ply 10: Bxb8 looked like +1 because
            // Rxb8 was certified at 2 in its frame; the exchange says 3)
            // (A CHECK IS A TEMPO -- Kaveh 2026-10-10: in check, only a capture that answers the check counts; 2026-10-11:
            // Rd2+ "netted -6" for their capture of our f6 queen, which they cannot make while in check)
            let legal_caps: Bitboard = if q.is_check() { q.legal_moves().iter().filter(|m| m.is_capture()).map(|m| m.to()).collect() } else { Bitboard::FULL };
            if let Some(pt) = threats(&q, q.turn()).into_iter().filter(|t| !t.mate && legal_caps.contains(t.target) && q.board().piece_at(t.target).map_or(false, |p| p.color == us_c)).max_by_key(|t| (t.value, t.target)) {
                if pt.value > v { v = pt.value; w = Some(format!("the {} on {} by exchange (worth {})", q.board().piece_at(pt.target).map(|p| format!("{:?}", p.role).to_lowercase()).unwrap_or_default(), pt.target, pt.value)); }
            }
            NET_CACHE.with(|c| c.borrow_mut().insert(key, (v, w.clone(), tp.proofs, tp.nodes)));
            (v, w)
        }
    };
    // TWO KINDS OF PROOF (2026-10-10): a FULL proof that HOLDS searched every reply -- a lower bound at its goal: the net
    // is the larger of it and the board's count (hxg8=Q+ proved "4" takes a rook and queens, 13; Nxe5 proved "3 against
    // the bishop on g4" -- the recapture on e5 uncovers Be2xg4, which the board's one-move count cannot see: 75 such
    // wins were lost when every capture was read by the board alone). The FRAME fallback's value ("certainly nets v in
    // its frame") is blind outside its frame -- fxe4 "nets 3", Qh7+ then wins outside it: the smaller of the two.
    let board = taken - theirs;
    let framed = FRAMED.with(|f| f.borrow().contains(&u));
    // (a quiet frame value: what it wins in its frame minus their best win after the move -- their counterplay comes first)
    let net = match proved { Some(g) if g >= MATE_GOAL => g, Some(g) if framed_quiet => g - theirs, Some(g) if framed => g.min(board), Some(g) => g.max(board), None => board };
    (net, their_why)
}

/// The warning graded: every legal move's net (`net_after`) against the decision's (`net_m`); a move at least AVOID_MARGIN
/// better makes the decision a CERTAIN blunder and replaces it.
fn grade_and_avoid(pos: &Chess, p: &mut Plan, mv: &str, net_m: i64) {
    let wins = p.wins.clone();
    let (mut pr, mut nd) = (0u32, 0u64);
    let best = best_by_net(pos, &wins, mv, net_m, AVOID_MARGIN, &mut pr, &mut nd);
    p.proofs += pr; p.nodes += nd;
    match best {
        Some((net, u, why)) if net >= net_m + AVOID_MARGIN => {
            p.trace.push(format!("    certain blunder: {} nets {net_m}; {} nets {net}{} -- {} instead ({pr} proofs)", san(pos, mv), san(pos, &u),
                why.map(|w| format!(" (their best then: {w})")).unwrap_or_default(), san(pos, &u)));
            let kind = if net > 0 && wins.iter().any(|w| w.0 == u) { String::new() } else { "defend".to_string() };
            p.best = Some((kind, u.clone(), net, format!("{} avoids the blunder {} (nets {net} against {net_m})", san(pos, &u), san(pos, mv))));
        }
        Some((net, _, _)) => p.trace.push(format!("    unavoidable: no move nets more than {net} against {}'s {net_m} -- kept ({pr} proofs)", san(pos, mv))),
        None => p.trace.push("    probable: no alternative to check".to_string()),
    }
}

/// THE MOVE WITH THE BEST NET other than `skip` -- a cheap gate first (Kaveh 2026-10-09: "a cheap gate followed by a more
/// thorough gate"): every move scored by what it takes now minus their biggest static threat after it (material,
/// promotion, trapped, mate by checks); their reasoning (`net_after`) on the AVOID_TOP best, and on the rest while none of
/// those beats `floor` by AVOID_MARGIN (probable -> certain). Used by the planner's defence and by the guard.
fn best_by_net(pos: &Chess, wins: &[(String, Square, i64)], skip: &str, floor: i64, margin: i64, pr: &mut u32, nd: &mut u64) -> Option<(i64, String, Option<String>)> {
    let mv = skip;
    let net_m = floor;
    let mut best: Option<(i64, String, Option<String>)> = None;
    // a cheap gate first (Kaveh 2026-10-09: "a cheap gate followed by a more thorough gate"): every move scored by the
    // static threats -- what it wins now minus their biggest threat after it (material, promotion, trapped, mate by
    // checks); their reasoning is run on the AVOID_TOP best only
    let them = !pos.turn();
    let mut cands: Vec<(i64, Move)> = pos.legal_moves().into_iter().filter(|m| m.to_uci(CastlingMode::Standard).to_string() != mv).map(|m| {
        let u = m.to_uci(CastlingMode::Standard).to_string();
        let q = after(pos, &m);
        let ours = wins.iter().filter(|w| w.0 == u).map(|w| w.2).max().unwrap_or_else(|| (kernel_material(&q, pos.turn()) - kernel_material(pos, pos.turn())).max(0));
        let theirs = if q.is_checkmate() { -MATE_GOAL } else { threats(&q, them).iter().map(|t| t.value).max().unwrap_or(0) };
        (ours - theirs, m)
    }).collect();
    cands.sort_by(|a, b| b.0.cmp(&a.0));
    // PROBABLE -> CERTAIN: when the gate's best do not settle it, the rest are checked before "unavoidable" is said
    // (2026-10-10: with the top 5 only, 34% of the decisions kept as unavoidable lost 150+ cp to Stockfish's best)
    for (i, (_, m)) in cands.into_iter().enumerate() {
        if i >= AVOID_TOP && best.as_ref().map_or(false, |b| b.0 >= net_m + margin) { break; }
        let u = m.to_uci(CastlingMode::Standard).to_string();
        let (net, why) = net_after(pos, &m, wins, pr, nd);
        let better = best.as_ref().map_or(true, |b| net > b.0 || (net == b.0 && b.1.parse::<shakmaty::uci::UciMove>().ok()
            .and_then(|x| x.to_move(pos).ok()).map_or(true, |bm| tie_score(pos, &m) > tie_score(pos, &bm))));
        if better { best = Some((net, u, why)); }
    }
    best
}

pub fn plan(pos: &Chess) -> Plan {
    NET_CACHE.with(|c| c.borrow_mut().clear());
    FRAMED.with(|f| f.borrow_mut().clear());
    let mut p = plan_inner(pos, false);
    if AFTER_CHECK.load(Ordering::Relaxed) {
        if let Some(d) = &p.best {
            let mv = if d.0.is_empty() || d.0 == "defend" || d.0 == "cheapest" { d.1.clone() } else { d.0.clone() };
            if let Some(m) = mv.parse::<shakmaty::uci::UciMove>().ok().and_then(|u| u.to_move(pos).ok()) {
                let q = after(pos, &m);
                if !q.is_checkmate() {
                    let (mut pr, mut nd) = (0u32, 0u64);
                    let wins = p.wins.clone();
                    let (net, why) = net_after(pos, &m, &wins, &mut pr, &mut nd);
                    p.proofs += pr; p.nodes += nd;
                    if net < 0 {
                        p.trace.push(format!("after: {} nets {net}{}", san(pos, &mv), why.map(|w| format!(" (they then have {w})")).unwrap_or_default()));
                        if AVOID.load(Ordering::Relaxed) { grade_and_avoid(pos, &mut p, &mv, net); }
                    }
                }
            }
        }
    }
    // THE BLUNDER CHECK from the pool: does the decision hand them one of their waiting tactics?
    if POOL.load(Ordering::Relaxed) {
        let frames = MEMO.with(|m| m.borrow().as_ref().map(|m| m.pool.clone()).unwrap_or_default());
        if let Some(d) = &p.best {
            let mv = if d.0.is_empty() || d.0 == "defend" || d.0 == "cheapest" { &d.1 } else { &d.0 };
            if let Some(m) = mv.parse::<shakmaty::uci::UciMove>().ok().and_then(|u| u.to_move(pos).ok()) {
                for (u, name, v) in crate::waiting::hands_for(pos, &frames, &m) {
                    let q = after(pos, &m);
                    p.trace.push(format!("hands: {} hands them {name} {} (worth {v}) -- the waiting-frames pool", san(pos, mv), san(&q, &u)));
                }
            }
        }
    }
    p
}

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
    // A MATE FIRST (Kaveh 2026-10-10: once a mate is found and proven, no more time is spent): a forced mate by checks --
    // every reply searched -- ends the analysis before their threat is even computed
    if !immediate_only {
        if let Some(m) = crate::threat::mate_by_checks(pos, crate::threat::MATE_CHECKS) {
            let u = m.to_uci(CastlingMode::Standard).to_string();
            trace.push(format!("mate: {} mates by checks (proven: every reply searched) -- no further analysis", san(pos, &u)));
            let d = Some((String::new(), u.clone(), MATE_GOAL, format!("{} mates by checks", san(pos, &u))));
            return Plan { trace, best: d.clone(), proofs, nodes, target: b.king_of(them), immediate: d, wins: vec![(u, b.king_of(them).unwrap_or(m.to()), MATE_GOAL)] };
        }
    }
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
    // THE NUCLEUS (Kaveh 2026-10-10: "the pin is suddenly removed ... the knight becomes available to hop ... pony, where
    // can you hop? ... it cannot do anything, so now my queen is free ... I reason from the point of view of my pieces"):
    // the ledger's events since the last position (ledger.rs; a game's PlanState), searched first -- their freed pieces
    // asked what they can do, ours what two threats they can make at once
    let events = if immediate_only || !NUCLEUS.load(Ordering::Relaxed) { vec![] } else { MEMO.with(|m| m.borrow().as_ref().map(|m| m.events.clone()).unwrap_or_default()) };
    if !events.is_empty() {
        trace.push(format!("what changed: {}", events.iter().map(|e| e.text.clone()).collect::<Vec<_>>().join("; ")));
        let mut asked: Vec<Square> = Vec::new();
        for e in events.iter().filter(|e| e.freed) {
            if asked.contains(&e.sq) { continue; }
            asked.push(e.sq);
            if b.piece_at(e.sq) != Some(e.piece) { continue; }
            if e.piece.color == them {
                let can = crate::ledger::threats_by_piece(pos, e.sq);
                trace.push(if can.is_empty() { format!("    their {:?} on {}: what can it do? nothing -- no move of it makes a threat", e.piece.role, e.sq).to_lowercase() }
                           else { format!("    their {:?} on {}: it threatens {}", e.piece.role, e.sq, can.iter().take(3).map(|(u, t, v)| format!("{} ({v} on {t})", san(&pass(pos).unwrap_or(pos.clone()), u))).collect::<Vec<_>>().join(", ")).to_lowercase() });
                continue;
            }
            for (m, t, v, mate) in crate::ledger::double_threats_of(pos, e.sq).into_iter().take(3) {
                let u = m.to_uci(CastlingMode::Standard).to_string();
                let mut focus = vec![t];
                if mate { if let Some(k) = b.king_of(them) { focus.push(k); } }
                let (ok, n) = with_tag("nucleus".to_string(), || mprove(pos, &u, &focus, v));
                proofs += 1; nodes += n;
                trace.push(format!("    our freed {:?} on {}: {} makes two threats{} -- wins {v} on {t}? {} ({n} nodes)", e.piece.role, e.sq, san(pos, &u),
                    if mate { " (one a mate)" } else { "" }, if ok { "HOLDS" } else { "fails" }).to_lowercase().replacen("holds", "HOLDS", 1));
                if ok {
                    wins.push((u.clone(), t, v));
                    if best.as_ref().map_or(true, |x| v > x.2) { best = Some((String::new(), u.clone(), v, format!("{} makes two threats with the freed {:?} (wins {v})", san(pos, &u), e.piece.role).to_lowercase())); best_target = Some(t); }
                }
            }
        }
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
    // MATERIAL BY CHECKS (2026-10-10, game 10000000 ply 76: Ba5+ Kc8 Rg8+ Kb7 Rb1+ Kc6 Rxa8 -- tried only as a mate):
    // a chain of up to 3 checks, every reply searched, ending with 2+ won (threat.rs gain_by_checks, the cornered king
    // from mate to material), confirmed by the value search before it counts -- for THEIR reasoning too (ply 77: Nc6
    // walked into Rg7+ Ne7 Rxe7+)
    {
        // (the forcing search's value -- checks, captures and quiet threats to the end of the forcing, no count -- in place
        // of material by checks within 3)
        if let Some((m, g)) = crate::threat::forcing_value(pos, crate::threat::FBUDGET.load(Ordering::Relaxed)) {
            if g >= 2 && g < MATE_GOAL {
                let u = m.to_uci(CastlingMode::Standard).to_string();
                // (the forcing search's proof is the certificate -- every reply searched, every leaf read by the exchange
                // law; a second confirmation by the frame-limited value search ran out of budget on 7-ply lines and threw
                // proved wins away -- 2026-10-11, Re7+ wins 5 after Bxc3 went unseen)
                trace.push(format!("forced: {} wins {g} by force (checks, captures, threats; every reply searched)", san(pos, &u)));
                let t = b.king_of(them).unwrap_or(m.to());
                wins.push((u.clone(), t, g));
                if best.as_ref().map_or(true, |x| g > x.2) { best = Some((String::new(), u.clone(), g, format!("{} wins {g} by force", san(pos, &u)))); best_target = Some(t); }
            }
        }
    }
    // THE POOL'S ACTIVE FRAMES (ours, on this board): tactics the frames certified -- the same proof as a seed's
    let pool: Vec<crate::waiting::Frame> = if immediate_only || !POOL.load(Ordering::Relaxed) { vec![] }
        else { MEMO.with(|m| m.borrow().as_ref().map(|m| m.pool.clone()).unwrap_or_default()) };
    for f in pool.iter().filter(|f| f.owner == us && f.status == crate::waiting::Status::Active && !f.stale) {
        if f.seed.mv.from().map_or(true, |fr| b.piece_at(fr).map_or(true, |pc| pc.color != us)) { continue; }
        if wins.iter().any(|w| w.0 == f.seed.uci) { continue; }
        let Some(t) = f.seed.focus.first().copied() else { continue };
        let v = f.seed.goal;
        trace.push(format!("frame: {} {} holds (the waiting-frames pool) -- wins {v}", f.seed.name, san(pos, &f.seed.uci)));
        wins.push((f.seed.uci.clone(), t, v));
        if best.as_ref().map_or(true, |x| v > x.2) { best = Some((String::new(), f.seed.uci.clone(), v, format!("{} wins {v} ({}, from the frame pool)", san(pos, &f.seed.uci), f.seed.name))); best_target = Some(t); }
    }
    // THE POOL'S SEEDS (waiting::seeds, goal 2+), proved like our own -- where the target-first seeds did not already win
    if !immediate_only && FRAME_SEEDS.load(Ordering::Relaxed) {
        for sd in crate::waiting::seeds(pos) {
            // (checks are the king target's seeds already: proving each again for mate doubled the cost)
            if sd.goal < 2 || sd.name == "cornering king" || wins.iter().any(|w| w.0 == sd.uci) { continue; }
            if best.as_ref().map_or(false, |x| x.0.is_empty() && x.2 >= sd.goal) { continue; }
            if refused(&sd.mv).is_some() { continue; }
            let (ok, n) = with_tag("frame".to_string(), || mprove(pos, &sd.uci, &sd.focus, sd.goal));
            proofs += 1; nodes += n;
            if ok {
                let Some(t) = sd.focus.first().copied() else { continue };
                trace.push(format!("frame seed: {} {} wins {} ({n} nodes)", sd.name, san(pos, &sd.uci), sd.goal));
                wins.push((sd.uci.clone(), t, sd.goal));
                if best.as_ref().map_or(true, |x| sd.goal > x.2) { best = Some((String::new(), sd.uci.clone(), sd.goal, format!("{} wins {} ({})", san(pos, &sd.uci), sd.goal, sd.name))); best_target = Some(t); }
            }
        }
    }
    // 2 TARGETS by value
    let mut armed: Vec<(i64, Square, Vec<String>)> = Vec::new();        // failing seeds kept for the arming phase
    let mut targets: Vec<(i64, Square)> = b.by_color(them).into_iter().filter_map(|sq| b.piece_at(sq).map(|p| (val(p.role), sq))).filter(|(v, _)| *v >= 3).collect();
    targets.sort_by(|a, b| b.0.cmp(&a.0));
    for (v, t) in targets {
        // the STOP RULE counts only IMMEDIATE wins (2026-10-09: an armed threat -- they get a move to parry it -- stopped
        // the search before Rf5 won a bishop, Rd4 a knight)
        let have = best.as_ref().filter(|x| x.0.is_empty()).map_or(0, |x| x.2);
        // (a target's bound includes what its capture itself makes: a pawn that takes it while promoting -- 2026-10-10:
        // "r on g8 is worth 5, not more than the 5 already won" cut hxg8=Q+, which takes the rook and queens, 12)
        let promo = pos.legal_moves().iter().any(|m| m.to() == t && m.is_promotion());
        let v_bound = if promo { v + 8 } else { v };
        // (a STRICT surplus stops: an equal target may hold a plain capture -- 2026-10-11, "Bh5 wins 3 by force" cut the
        // bishop target where cxd4 simply took it; Qg4 "wins 3" cut Kxg2)
        let surplus = if STRICT_STOP.load(Ordering::Relaxed) { have > v_bound.min(MATE_GOAL) } else { have >= v_bound.min(MATE_GOAL) };
        if surplus && !NO_STOP.with(|c| c.get()) {
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
        // (a threat we cannot execute now is not in hand: in check, nothing is -- 2026-10-10, Kd1 out of check with e8=Q
        // waiting)
        let before_t = if pos.is_check() { vec![] } else { threats(pos, us) };
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
            // THE SINGLE-THREAT LEMMA (the lone-attack lemma of 2026-10-09 made generic, Kaveh 2026-10-10: "a threat should
            // have a value ... losing a piece, or becoming mate"): a move that attacks this piece wins nothing if the piece
            // has a safe escape after which we have NO new threat at all -- material or mate (threat.rs). That one test
            // replaces the hand-listed exceptions (a second target, a new path onto the king) and adds the JOB: the
            // escape gives up what the piece parried (the lemma reviewer's surprises: Rc1's queen guarding Rc8#, Rg7's
            // queen shielding its king). A capture (a gain in hand) and a check are proved. Not for the king.
            if !king {
                if let Some(sm) = s.parse::<shakmaty::uci::UciMove>().ok().and_then(|u| u.to_move(pos).ok()) {
                    let q = after(pos, &sm);
                    let capture = sm.to() == t;
                    // (a promotion is a gain in hand like a capture -- 2026-10-10: c8=Q attacking the e6 bishop was refused)
                    if !capture && !sm.is_capture() && sm.promotion().is_none() && !q.is_check() {
                        let esc = escapes(&q, us, t);
                        let quiet = q.legal_moves().into_iter().filter(|e| e.from() == Some(t) && esc.contains(e.to()))
                            .find(|e| any_new_threat(&after(&q, e), us, &before_t) == Some(false));   // certainly none: unknown proves
                        if let Some(e) = quiet {
                            trace.push(format!("    {}: one threat, and the {name} steps away to {} leaving no threat of ours, nothing forced -- no proof (the single-threat lemma)", san(pos, s), e.to()));
                            tally(|st| st.lone += 1);
                            failed.push(s.clone());
                            continue;
                        } else if esc.any() {
                            trace.push(format!("    {}: the {name} can step away, but every escape leaves a threat of ours (a second target, a job, a mate) -- proved", san(pos, s)));
                        }
                    }
                }
            }
            // THE GOAL OF A SEED is what it can win, not the target's full value (2026-10-09: e5 against a pinned rook
            // gains 4 -- the pawn may fall -- and a fork wins the second piece): the static gain on the target after the
            // move (the exchange's first term), for two or more new targets the second best, for a capture the gain now
            // (the next lower threat value is tried once if the goal fails: with a mate threat among several, the defender
            // parries the biggest and the second-best is a guess -- 2026-10-10: Rd8 proved at 9 failed, it wins 3)
            let mut lower: Option<i64> = None;
            let g = if king { goal } else if let Some(sm) = s.parse::<shakmaty::uci::UciMove>().ok().and_then(|u| u.to_move(pos).ok()) {
                let q = after(pos, &sm);
                let now_t = threats(&q, us);
                // (generic threats: a mate threat beside the attack makes the attack the second-best threat -- it wins)
                let mut new: Vec<i64> = new_threats(&now_t, &before_t).iter().map(|x| x.value).collect();
                new.sort_by(|a, b| b.cmp(a));
                if sm.to() == t { before.get(&(t as u8)).copied().unwrap_or(goal).max(1) }
                else if new.len() >= 2 { lower = new.iter().skip(2).copied().find(|x| *x < new[1] && *x > 0); new[1].max(1) }
                else { now_t.iter().find(|x| x.target == t && !x.mate).map(|x| x.value).filter(|x| *x > 0).unwrap_or(goal).max(1) }
            } else { goal };
            let (mut ok, n) = with_tag("seed".to_string(), || mprove(pos, s, &[t], g));
            proofs += 1; nodes += n;
            trace.push(format!("    prove {} (gain {g}): {} ({n} nodes)", san(pos, s), if ok { "HOLDS" } else { "fails" }));
            let mut g = g;
            // A CAPTURE'S VALUE, not only yes/no at one goal (2026-10-10, the decision review: Rxa2 taking our queen "fails"
            // at 9 because Rxe1 wins a rook back -- it still wins 4; Bxd4 fails at 9, Bxd8 trades queens -- it still wins 6;
            // the invisible wins made Qxa2+ and Bxc7 look safe): what the capture certainly nets in its frame
            if !ok && !king {
                // (and a CHECK: Rd7+ Kh8 Rxc7 "fails" at 3 because Black wins h2 back -- it nets 2; and ANY seed: Rf5 against
                // the e5 queen fails at 7 and at 5, it wins 4 -- game 10000001 ply 66, where Qxf6+ then won the tie)
                if let Some(sm) = s.parse::<shakmaty::uci::UciMove>().ok().and_then(|u| u.to_move(pos).ok()) {
                    if let Some(v) = certified_value(pos, &sm, &[t, sm.to()], 2, 20_000) {
                        if v >= 1 {
                            trace.push(format!("    {} certainly nets {v} (its value in the frame)", san(pos, s)));
                            ok = true; g = v.min(g);
                            FRAMED.with(|f| f.borrow_mut().insert(s.clone()));
                        }
                    }
                }
            }
            if let (false, Some(g2)) = (ok, lower) {
                let (ok2, n2) = with_tag("seed".to_string(), || mprove(pos, s, &[t], g2));
                proofs += 1; nodes += n2;
                trace.push(format!("    prove {} (gain {g2}, the next threat down): {} ({n2} nodes)", san(pos, s), if ok2 { "HOLDS" } else { "fails" }));
                if ok2 { ok = true; g = g2; }
            }
            if ok {
                wins.push((s.clone(), t, g));
                MEMO.with(|m| if let Some(m) = m.borrow_mut().as_mut() { *m.history.entry(hk(s)).or_insert(0.0) += 1.0; });
                if king { won_outright = true; }                         // (review: a fork's second target may be worth more than this target -- only a mate cannot be beaten)
            }
            if ok { if best.as_ref().map_or(true, |x| g > x.2) { best = Some((String::new(), s.clone(), g, format!("{} wins {g} against the {name}", san(pos, s)))); best_target = Some(t); } }
            else { failed.push(s.clone()); }
            // (a proven MATE ends the analysis -- nothing else can beat it)
            if ok && king && !immediate_only {
                trace.push(format!("mate: {} is a proven mate -- no further analysis", san(pos, s)));
                let d = Some((String::new(), s.clone(), MATE_GOAL, format!("{} mates (proven)", san(pos, s))));
                return Plan { trace, best: d.clone(), proofs, nodes, target: Some(t), immediate: d, wins };
            }
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
    // THE IMMEDIATE WINS RANKED BY THEIR NET (2026-10-10, the decision review: a win taken in its frame while the
    // opponent wins more elsewhere -- Raxa7 nets 1, Nf4 forks the queen; Rxd3 takes a knight, Ra1+ mates): the best net
    // among the proved wins; a win whose net is not positive is no win
    if best.as_ref().map_or(false, |x| x.0.is_empty()) {
        let mut cand: Vec<(String, Square, i64)> = Vec::new();
        for w in &wins { if !cand.iter().any(|c| c.0 == w.0) { cand.push(w.clone()); } }
        // SIBLINGS: a win that captures on a square brings every other capture on that square (2026-10-10, game
        // 10000000 ply 28: dxe5 "wins 3" aimed at the queen, then the stop rule cut the e5 knight's target -- Bxe5, the
        // recapture Stockfish played, was never looked at)
        let caps: Vec<Square> = cand.iter().filter_map(|c| c.0.parse::<shakmaty::uci::UciMove>().ok().and_then(|u| u.to_move(pos).ok()))
            .filter(|m| m.is_capture()).map(|m| m.to()).collect();
        for m in pos.legal_moves() {
            if m.is_capture() && caps.contains(&m.to()) {
                let u = m.to_uci(CastlingMode::Standard).to_string();
                if !cand.iter().any(|c| c.0 == u) { let g = cand.iter().map(|c| c.2).max().unwrap_or(1); cand.push((u, m.to(), g)); }
            }
        }
        cand.sort_by(|a, b| b.2.cmp(&a.2));
        let mut top: Option<(i64, String, Square, Option<String>)> = None;
        for (u, t, _) in cand.into_iter().take(8) {
            let Some(m) = u.parse::<shakmaty::uci::UciMove>().ok().and_then(|x| x.to_move(pos).ok()) else { continue };
            let (net, why) = net_after(pos, &m, &wins, &mut proofs, &mut nodes);
            let better = top.as_ref().map_or(true, |x| net > x.0 || (net == x.0 && x.1.parse::<shakmaty::uci::UciMove>().ok()
                .and_then(|y| y.to_move(pos).ok()).map_or(true, |xm| tie_score(pos, &m) > tie_score(pos, &xm))));
            if better { top = Some((net, u, t, why)); }
        }
        match top {
            Some((net, u, t, _)) if net > 0 => {
                if best.as_ref().map_or(true, |x| x.1 != u || x.2 != net) {
                    trace.push(format!("by net: {} nets {net}", san(pos, &u)));
                    best = Some((String::new(), u.clone(), net, format!("{} wins {net} (by net)", san(pos, &u))));
                    best_target = Some(t);
                }
            }
            Some((net, u, _, why)) => {
                trace.push(format!("by net: no win nets more than 0 ({} nets {net}{}) -- not a win", san(pos, &u), why.map(|w| format!("; they then have {w}")).unwrap_or_default()));
                best = None; best_target = None;
            }
            None => {}
        }
    }
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
                        // (defences RANKED by their net -- 2026-10-10: the first defence found was played; 337 of the
                        // planner's 613 warned decisions were a defence that dropped something else: Nb4 stops Nxc6, cxb4)
                        let (net, why) = net_after(pos, &m, &wins, &mut proofs, &mut nodes);
                        trace.push(format!("    {} stops it (nets {net}{})", san(pos, &mu), why.map(|w| format!("; they then have {w}")).unwrap_or_default()));
                        let better = match &best {
                            Some(x) if x.0 == "defend" => net > x.2 || (net == x.2 && {
                                let cur = x.1.parse::<shakmaty::uci::UciMove>().ok().and_then(|u| u.to_move(pos).ok());
                                cur.map_or(true, |c| tie_score(pos, &m) > tie_score(pos, &c))
                            }),
                            _ => true,
                        };
                        if better {
                            best = Some(("defend".to_string(), mu.clone(), net, format!("{} stops their {} (nets {net})", san(pos, &mu), san(&p0, &tseed))));
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
                        let mut fb = frame_fallback(pos, &[tt], line, 50_000);
                        // (the frame search's best five re-valued by their net, which sees beyond the frame -- 2026-10-10:
                        // Qxf5 "keeps -4" in the frame, gxf5 then wins the queen)
                        let mut scored: Vec<(String, i64)> = Vec::new();
                        for (u, _) in fb.iter().take(5) {
                            if let Some(m) = u.parse::<shakmaty::uci::UciMove>().ok().and_then(|x| x.to_move(pos).ok()) {
                                let (net, _) = net_after(pos, &m, &wins, &mut proofs, &mut nodes);
                                scored.push((u.clone(), net));
                            }
                        }
                        scored.sort_by(|a, b| b.1.cmp(&a.1));
                        if let Some(top) = scored.first().cloned() { fb.retain(|x| x.0 != top.0); fb.insert(0, top); }
                        if let Some((mv, v)) = fb.first().cloned() {
                            trace.push(format!("no defence stops their {}: alpha-beta in the frame ({} first moves) -- {} keeps {v}{}", san(&p0, &tseed), fb.len(), san(pos, &mv),
                                match &lp { Some((_, net, _)) if *net == v => " (the lemma agreed)".to_string(), Some((_, net, _)) => format!(" (the lemma predicted {net})"), None => String::new() }));
                            best = Some(("cheapest".to_string(), mv.clone(), v, format!("{} is the cheapest way to lose (frame search, {v})", san(pos, &mv))));
                        }
                    }
                }
                // THE ANSWER IS THE MOVE WITH THE BEST NET (Kaveh 2026-10-10: the guard's job is the planner's): after the
                // geometry's defences, every other move by the same net -- giving the piece back cheaply or a counter-capture
                // can net more than any defence that stops the threat
                let (cur_mv, cur_net) = best.as_ref().map(|x| (x.1.clone(), x.2)).unwrap_or((String::new(), -tg));
                // (only when the answer found loses something: a move netting more than 0 would be an immediate win the
                // planner already proved -- 2026-10-10: scanning always tripled the time)
                // (the planner's own choice: any better net wins -- 2026-10-10, game 10000001 ply 37: Bh7 nets 0 against the
                // frame search's -1, and the guard's margin of 2 kept the losing move)
                if cur_net < 0 { if let Some((net, u, why)) = best_by_net(pos, &wins, &cur_mv, cur_net, 1, &mut proofs, &mut nodes) {
                    if net > cur_net || best.is_none() {
                        trace.push(format!("    the best net: {} nets {net}{} -- better than {} ({cur_net})", san(pos, &u),
                            why.map(|w| format!(" (they then have {w})")).unwrap_or_default(), if cur_mv.is_empty() { "nothing".to_string() } else { san(pos, &cur_mv) }));
                        best = Some(("defend".to_string(), u.clone(), net, format!("{} answers their {} (nets {net})", san(pos, &u), san(&p0, &tseed))));
                    }
                } }
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
                    // (the preparing move must not lose material by itself -- 2026-10-10: Bh6 arming Qxg7, Qxh6)
                    let (net, why) = net_after(pos, &am, &wins, &mut proofs, &mut nodes);
                    if net < 0 { trace.push(format!("        {} would net {net}{} -- not armed", san(pos, a), why.map(|w| format!(" (they then have {w})")).unwrap_or_default())); continue; }
                    best = Some((a.clone(), s.clone(), goal, format!("{} arms {} against the {name}", san(pos, a), san(&p, s))));
                }
            }
        }
      }
    }
    // THE POOL'S ARMS (the reverse lookup): a preparing move that meets a dormant frame's missing condition, certified on
    // the null move after it -- when nothing better was found and a preparing move is affordable
    if best.is_none() && tempo && !pool.is_empty() {
        let (arms, _, _) = crate::waiting::arming_core(pos, &pool);
        if let Some(a) = arms.iter().max_by_key(|a| a.4) {
            let q = a.0.parse::<shakmaty::uci::UciMove>().ok().and_then(|u| u.to_move(pos).ok()).map(|m| after(pos, &m));
            let seed_san = q.as_ref().and_then(|q| pass(q)).map(|p| san(&p, &a.1)).unwrap_or(a.1.clone());
            trace.push(format!("pool arm: {} meets {} -- then {} ({}) wins {} on the null move", san(pos, &a.0),
                a.5.iter().map(|x| format!("{} {} {}", x.0, x.1, x.2)).collect::<Vec<_>>().join(", "), seed_san, a.2, a.4));
            best = Some((a.0.clone(), a.1.clone(), a.4, format!("{} arms {} ({}, from the frame pool)", san(pos, &a.0), seed_san, a.2)));
        }
    }
    Plan { trace, best, proofs, nodes, target: best_target, immediate, wins }
}

/// net_of(fen, uci) -> (the planner's net for that move, their best proved win after it) -- for digging: what did we think
/// of the move Stockfish played?
#[pyfunction]
pub fn net_of(fen: String, uci: String) -> PyResult<(i64, Option<String>)> {
    let Some(pos) = parse(&fen) else { return Ok((0, None)) };
    let Some(m) = uci.parse::<shakmaty::uci::UciMove>().ok().and_then(|u| u.to_move(&pos).ok()) else { return Ok((0, None)) };
    NET_CACHE.with(|c| c.borrow_mut().clear());
    FRAMED.with(|f| f.borrow_mut().clear());
    let p = plan_inner(&pos, false);
    let (mut pr, mut nd) = (0u32, 0u64);
    Ok(net_after(&pos, &m, &p.wins, &mut pr, &mut nd))
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

type StatRow = (u32, u32, u32, u32, u32, u32, u32, u32, u32, u32);

#[pymethods]
impl PlanState {
    #[new]
    fn new() -> Self { PlanState { memo: Some(Memo::default()) } }

    fn observe(&mut self, fen: String) -> PyResult<(Vec<String>, Option<(String, String, i64, String)>, u32, u64, f64, Option<(String, String, i64, String)>, StatRow)> {
        let Some(pos) = parse(&fen) else { return Ok((vec![], None, 0, 0, 0.0, None, (0, 0, 0, 0, 0, 0, 0, 0, 0, 0))) };
        let mut memo = self.memo.take().unwrap_or_default();
        memo.stats = Stats::default();
        for v in memo.history.values_mut() { *v *= 0.8; }                    // recent wins weigh more
        // the ledger of this position and what changed since the last one (the nucleus)
        if NUCLEUS.load(Ordering::Relaxed) {
            let lg = crate::ledger::ledger(&pos);
            memo.events = memo.ledger.as_ref().map(|prev| crate::ledger::diff(prev, &lg)).unwrap_or_default();
            memo.ledger = Some(lg);
        } else { memo.events.clear(); }
        if POOL.load(Ordering::Relaxed) {
            let (next, _, _, _) = crate::waiting::pool_step(&memo.pool, memo.pool_ply, &pos, true);
            memo.pool = next;
            memo.pool_ply += 1;
        }
        // PLAYING OUT A PROVEN MATE: their reply read off the board, our answer read off the stored proof -- no search
        let mut played: Option<Plan> = None;
        if let Some(sol) = memo.solution.clone() {
            if pos.turn() == sol.owner {
                let reply = sol.after_ours.legal_moves().into_iter().find(|r| { let x = after(&sol.after_ours, r); x.board() == pos.board() && x.turn() == pos.turn() });
                let next = reply.and_then(|r| {
                    let path = child_path(sol.path, move_code(&r));
                    sol.hints.get(&path).copied().and_then(|code| pos.legal_moves().into_iter().find(|m| move_code(m) == code)).map(|m| (path, m))
                });
                match next {
                    Some((path, m)) => {
                        let u = m.to_uci(CastlingMode::Standard).to_string();
                        let d = Some((String::new(), u.clone(), MATE_GOAL, format!("{} -- the proven mate, played out (no search)", san(&pos, &u))));
                        let q = after(&pos, &m);
                        memo.solution = if q.is_checkmate() { None } else { Some(Solution { owner: sol.owner, hints: sol.hints.clone(), path: child_path(path, move_code(&m)), after_ours: q }) };
                        played = Some(Plan { trace: vec![format!("mate: playing out the proven solution -- {} (stored answer to their reply)", san(&pos, &u))],
                                             best: d.clone(), proofs: 0, nodes: 0, target: pos.board().king_of(!pos.turn()), immediate: d, wins: vec![] });
                    }
                    None => { memo.solution = None; }
                }
            }
        }
        MEMO.with(|m| *m.borrow_mut() = Some(memo));
        let t = std::time::Instant::now();
        let p = match played { Some(p) => p, None => plan(&pos) };
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        // a NEW proven mate (through the memory's stored proof) becomes the solution to play out
        if p.proofs > 0 {
            if let Some((k, u, v, _)) = &p.best {
                if k.is_empty() && *v >= MATE_GOAL {
                    if let (Some(m), Some(ks)) = (u.parse::<shakmaty::uci::UciMove>().ok().and_then(|x| x.to_move(&pos).ok()), pos.board().king_of(!pos.turn())) {
                        if let Some(h) = stored_proof(&pos, u, &[ks], MATE_GOAL) {
                            let sol = Solution { owner: pos.turn(), hints: h, path: child_path(0, move_code(&m)), after_ours: after(&pos, &m) };
                            MEMO.with(|mm| if let Some(mm) = mm.borrow_mut().as_mut() { mm.solution = Some(sol); });
                        }
                    }
                }
            }
        }
        let memo = MEMO.with(|m| m.borrow_mut().take()).unwrap_or_default();
        let s = memo.stats;
        self.memo = Some(memo);
        Ok((p.trace, p.best, p.proofs, s.nodes, ms, p.immediate,
            (s.fresh, s.replay, s.reuse_exact, s.reuse_fail, s.reprove, s.lone, s.frozen, s.escape_skip, s.cut_by_history, s.reuse_zone)))
    }

    /// Entries in the memory.
    fn size(&self) -> usize { self.memo.as_ref().map_or(0, |m| m.entries.values().map(|v| v.len()).sum()) }
}
