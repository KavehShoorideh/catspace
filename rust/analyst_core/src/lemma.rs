//! THE ONE-MOVE THREAT LEMMA, IN STOPPER FORM (Kaveh 2026-10-11: "prove the ways in which it can be stopped ... that
//! becomes the information Rust uses when deciding what squares are necessary to check for the continual usefulness
//! of a lemma"; Lean Ideas.ThreatLemma.threat_stoppers). Our move attacks their piece on `s`: next move we take it for
//! the exchange law's value, WHATEVER THEY REPLY -- unless a stopper exists. Each stopper is a rule over named squares,
//! and those squares are the verdict's dependency (recorded through lib.rs dep_part / DEP_ONE_STEP; compared by
//! plan.rs inputs_differ when the verdict is asked again on a changed board):
//!   escape       the piece on `s` has a safe square                      reads s's reach, the attacks onto it
//!   defence      a reply adds an attacker of theirs onto `s`, or removes one of ours (a defender arrives, a
//!                discovery, our attacker captured or blocked)            reads the one-step movers onto bearing(s)
//!   check        a reply gives check (the tempo), or our capture is pinned  reads the one-step movers onto bearing(k)
//!   intermezzo   their best capture now, taken raw (they move first; we take `s` instead of answering), is not
//!                outweighed by the exchange on `s`                        reads the index
//!   promotion    a pawn of theirs can promote                             reads their seventh-rank pawns
//! No stopper: the verdict HOLDS at `value` = see(s) - intermezzo. Any stopper: UNDECIDED -- the search decides, as
//! before. The lemma is generic: it names values and squares, never a piece.
use super::*;
use shakmaty::Bitboard;

pub struct Verdict { pub value: i64, pub stopper: Option<&'static str>, pub dep: Bitboard, pub one_step: Bitboard }

/// The squares bearing on `t` on this board (a queen's lines to the first blocker, a knight's move, next to it).
fn bearing_now(t: Square, occ: Bitboard) -> Bitboard {
    shakmaty::attacks::queen_attacks(t, occ) | shakmaty::attacks::knight_attacks(t) | shakmaty::attacks::king_attacks(t)
}

/// Our move `m` from `p`; `s` the piece of theirs it attacks. The lemma's verdict on "we take `s` next move".
pub fn threat_lemma(p: &Chess, m: &Move, s: Square) -> Verdict {
    let us = p.turn();
    let them = !us;
    let q = after(p, m);
    let b = q.board();
    let occ = b.occupied();
    let mut dep = Bitboard::from_square(s);
    let mut one_step = Bitboard::EMPTY;
    let undecided = |why, dep, one_step| Verdict { value: 0, stopper: Some(why), dep, one_step };
    let Some(k) = b.king_of(us) else { return undecided("noKing", dep, one_step) };
    dep.add(k);
    let Some(tp) = b.piece_at(s) else { return undecided("target", dep, one_step) };
    if tp.color != them || tp.role == Role::King { return undecided("target", dep, one_step); }
    let ours = b.attacks_to(s, us, occ);
    if ours.is_empty() { return undecided("noAttack", dep, one_step); }
    dep |= ours;
    one_step |= bearing_now(s, occ) | bearing_now(k, occ);
    // the exchange on s, for us capturing first (Lean seeXB)
    let see = rules::see(b, s, us);
    if see <= 0 { return undecided("exchange", dep, one_step); }
    // THE ESCAPES (the fork form, Kaveh: "every escape leaves a threat of ours"): each move of the piece on s is priced
    // by our best COMPOSED gain after it (lib.rs composed_from_index: our best capture or promotion by the exchange law,
    // less their intermezzo price -- the piece itself where it lands, or the second target it left), the squares read
    // recorded by that leaf; the worst escape bounds the value. An escape with check is a stopper (the tempo).
    let (_, reach) = escape_set(&q, us, s);
    dep |= reach;
    let mut worst_escape = i64::MAX;
    for r in q.legal_moves() {
        if r.from() != Some(s) { continue; }
        let q2 = after(&q, &r);
        if q2.is_check() { return undecided("check", dep, one_step); }
        let v = composed_from_index(&q2, &idx_root(&q2, us), us) - r.capture().map(kernel_value).unwrap_or(0);
        if v < worst_escape { worst_escape = v; }
    }
    if worst_escape <= 0 { return undecided("escape", dep, one_step); }
    // CHECK: a reply that gives check, or our capture pinned (the capture must be legal were it our move)
    if !crate::threat::checking_moves(&q).is_empty() { return undecided("check", dep, one_step); }
    match q.clone().swap_turn() {
        Ok(ours_turn) => if !ours_turn.legal_moves().iter().any(|x| x.to() == s) { return undecided("pin", dep, one_step); },
        Err(_) => return undecided("pin", dep, one_step),
    }
    // PROMOTION: a pawn of theirs on its seventh rank (whatever it could do, the search decides)
    let seventh = if them == Color::White { Bitboard::from_rank(shakmaty::Rank::Seventh) } else { Bitboard::from_rank(shakmaty::Rank::Second) };
    let pawns7 = b.pawns() & b.by_color(them) & seventh;
    dep |= pawns7;
    if pawns7.any() { return undecided("promotion", dep, one_step); }
    // DEFENCE: a reply after which their attackers of s grew or ours shrank (the piece on s itself moving is the escape,
    // read above; its moves are skipped here)
    let theirs0 = b.attacks_to(s, them, occ);
    for r in q.legal_moves() {
        if r.from() == Some(s) { continue; }
        let q2 = after(&q, &r);
        let b2 = q2.board();
        let occ2 = b2.occupied();
        if b2.piece_at(s).map_or(true, |x| x.color != them) { continue; }   // (s captured by them: not their piece any more; a reply onto s is a capture of ours? no -- s holds their piece)
        let theirs2 = b2.attacks_to(s, them, occ2);
        let ours2 = b2.attacks_to(s, us, occ2);
        if (theirs2 & !theirs0).any() || (ours & !ours2).any() { return undecided("defence", dep, one_step); }
    }
    // INTERMEZZO: they move first -- their best capture, raw (we take s instead of answering it)
    let raw = q.legal_moves().iter().filter(|r| r.from() != Some(s)).filter_map(|r| r.capture().map(kernel_value)).max().unwrap_or(0);
    let value = (see - raw).min(worst_escape);
    if value <= 0 { return undecided("intermezzo", dep, one_step); }
    Verdict { value, stopper: None, dep, one_step }
}

/// [holds, undecided (by stopper, in STOPPERS' order) ..., audit: agree, false claim, missed] since the last read.
pub const STOPPERS: [&str; 11] = ["noKing", "target", "noAttack", "exchange", "escape", "check", "pin", "promotion", "defence", "intermezzo", "value<goal"];
pub static LEMMA_N: [std::sync::atomic::AtomicU64; 15] = [const { std::sync::atomic::AtomicU64::new(0) }; 15];
pub fn count(i: usize) { LEMMA_N[i].fetch_add(1, Ordering::Relaxed); }
pub static LEMMA: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
pub static LEMMA_AUDIT: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// set_lemma(flag): the one-move threat lemma decides a 2-ply question before the search when no stopper exists.
#[pyfunction]
pub fn set_lemma(flag: bool) -> PyResult<()> { LEMMA.store(flag, Ordering::Relaxed); Ok(()) }
/// set_lemma_audit(flag): every verdict the lemma gives is answered by the 2-ply search too and compared.
#[pyfunction]
pub fn set_lemma_audit(flag: bool) -> PyResult<()> { LEMMA_AUDIT.store(flag, Ordering::Relaxed); Ok(()) }
#[pyfunction]
pub fn lemma_stats() -> Vec<u64> { LEMMA_N.iter().map(|a| a.swap(0, Ordering::Relaxed)).collect() }

/// threat_lemma_check(fen, uci, square) -> (value, stopper or None, dependency squares, one-step squares) -- the review hook.
#[pyfunction]
pub fn threat_lemma_check(fen: String, uci: String, square: String) -> PyResult<(i64, Option<String>, Vec<String>, Vec<String>)> {
    let Some(p) = parse(&fen) else { return Ok((0, Some("fen".into()), vec![], vec![])) };
    let Ok(u) = uci.parse::<shakmaty::uci::UciMove>() else { return Ok((0, Some("uci".into()), vec![], vec![])) };
    let Ok(m) = u.to_move(&p) else { return Ok((0, Some("uci".into()), vec![], vec![])) };
    let Ok(s) = square.parse::<Square>() else { return Ok((0, Some("square".into()), vec![], vec![])) };
    let v = threat_lemma(&p, &m, s);
    Ok((v.value, v.stopper.map(|x| x.to_string()), v.dep.into_iter().map(|x| x.to_string()).collect(), v.one_step.into_iter().map(|x| x.to_string()).collect()))
}
