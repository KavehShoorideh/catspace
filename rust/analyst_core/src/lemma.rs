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

/// THE NO-CHASE LEMMA (Kaveh 2026-10-11: "if the opponent can protect the piece, chasing it becomes pointless ...
/// chasing a knight or a bishop never makes sense unless you're chasing it with a pawn"; its stoppers MINED from the
/// questions the search held: on 20 census games every held one banked material, was forcing, made a second threat,
/// chased with a pawn, or was a zugzwang). Our move `m` asked to win `goal` from their piece on `s` FAILS -- no search
/// needed -- when the target has a HARBOUR: a real reply (never a pass: zugzwang) moving it, without capture or check,
/// to a square `h` such that, after it,
///   (a) what we can collect by captures (the search's own leaf: material + the capture settle) plus what `m` banked is
///       below the goal,
///   (b) no line of checks of ours wins what is missing (threat.rs gain_by_checks, every reply searched),
///   (c) no move of ours makes the exchange on `h` positive, nor any other static threat that, with `m`'s banked
///       material, reaches the goal (a second threat, a pile-up on a pinned piece),
/// and `m` itself gives no check, promotes nothing, banks less than the goal, is not a pawn move (the pawn chase), and
/// the target is not the king (a mate question). -> 2 (both tiers fail), 0 ask the search.
/// Why the lemma did not decide, the deepest stage any harbour reached: [m forcing or a pawn move, banked >= goal,
/// the king or no target, no quiet move of the target, (a) collectable, (b) a check of ours, (c) a renewed attack on h,
/// (c) a second threat, decided].
thread_local! { static HARBOUR: std::cell::Cell<Option<Square>> = const { std::cell::Cell::new(None) }; }
/// chase_check(fen, uci, square, goal) -> (the lemma's verdict, the harbour it found) -- the review hook.
#[pyfunction]
pub fn chase_check(fen: String, uci: String, square: String, goal: i64) -> PyResult<(u8, Option<String>)> {
    let Some(p) = parse(&fen) else { return Ok((0, None)) };
    let Some(m) = uci.parse::<shakmaty::uci::UciMove>().ok().and_then(|u| u.to_move(&p).ok()) else { return Ok((0, None)) };
    let Ok(s) = square.parse::<Square>() else { return Ok((0, None)) };
    HARBOUR.with(|c| c.set(None));
    let v = chase_lemma(&p, &m, s, goal);
    Ok((v, HARBOUR.with(|c| c.get()).map(|x| x.to_string())))
}
pub static CHASE_UNKNOWN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static CHASE_WHY: [std::sync::atomic::AtomicU64; 9] = [const { std::sync::atomic::AtomicU64::new(0) }; 9];
#[pyfunction]
pub fn chase_why() -> Vec<u64> { let mut v: Vec<u64> = CHASE_WHY.iter().map(|a| a.swap(0, Ordering::Relaxed)).collect(); v.push(CHASE_UNKNOWN.swap(0, Ordering::Relaxed)); v }
/// Can their piece on `t` be saved by moving (they to move in `q`)? A legal move of it to a square where our exchange is
/// not positive -- read from its LEGAL moves (a blocked pawn has none; lib.rs escape_set reads a piece's attack squares,
/// which for a pawn are its captures, not its moves: 2026-10-11, the blocked b4 pawn looked saveable).
fn can_step_away(q: &Chess, t: Square, us: Color) -> bool {
    q.legal_moves().iter().any(|r| r.from() == Some(t) && {
        let q2 = after(q, r);
        rules::see(q2.board(), r.to(), us) <= 0
    })
}

pub fn chase_lemma(p: &Chess, m: &Move, s: Square, goal: i64) -> u8 {
    let mut deepest = 3usize;
    let v = chase_inner(p, m, s, goal, &mut deepest);
    CHASE_WHY[if v == 2 { 8 } else { deepest }].fetch_add(1, Ordering::Relaxed);
    v
}
fn chase_inner(p: &Chess, m: &Move, s: Square, goal: i64, deepest: &mut usize) -> u8 {
    let us = p.turn();
    let q = after(p, m);
    if q.is_check() || m.is_promotion() || m.role() == Role::Pawn { *deepest = 0; return 0; }
    let banked = m.capture().map(kernel_value).unwrap_or(0);
    if banked >= goal { *deepest = 1; return 0; }
    let Some(tp) = q.board().piece_at(s) else { *deepest = 2; return 0 };
    if tp.color == us || tp.role == Role::King { *deepest = 2; return 0; }
    let base = kernel_material(p, us);
    for r in q.legal_moves() {
        if r.from() != Some(s) || r.is_capture() { continue; }
        let h = r.to();
        let q2 = after(&q, &r);
        if q2.is_check() { continue; }
        // the conditions are a conjunction, so they are tested CHEAPEST FIRST (the verdict is the same in any order):
        // (c) by the static exchange, (a), (b), then (c) with the generic threat
        let mat2 = kernel_material(&q2, us) - base;
        let ours2: Vec<Move> = q2.legal_moves().into_iter().collect();
        // (c), cheap: no move of ours renews the attack on h, nor makes material threats that win the goal against one
        // reply (the fork law: the second best -- the best only when its piece cannot be saved by moving). A rejection
        // here implies the generic test's rejection: the generic list contains these threats.
        let mut stage = 0usize;
        let arr2 = rules::threats_arr(&q2, us);                         // carried incrementally to each reply (exact)
        let cheap_ok = ours2.iter().all(|m2| {
            let q3 = after(&q2, m2);
            let got = m2.capture().map(kernel_value).unwrap_or(0);
            let arr = rules::threats_arr_inc(&arr2, &q3, m2, us);
            if q3.board().piece_at(h).map_or(false, |x| x.color != us) && arr[h as usize] > 0 { stage = 6; return false; }
            let (mut b1, mut b2) = ((0i64, 64usize), 0i64);
            for x in 0..64usize { let v = arr[x] as i64; if v > b1.0 { b2 = b1.0; b1 = (v, x); } else if v > b2 { b2 = v; } }
            if mat2 + got + b2 >= goal { stage = 7; return false; }
            if b1.0 > 0 && mat2 + got + b1.0 >= goal && !can_step_away(&q3, Square::new(b1.1 as u32), us) { stage = 7; return false; }
            true
        });
        if !cheap_ok { *deepest = (*deepest).max(stage); continue; }
        // (a) the search's leaf: material + the capture settle, us to move
        let leaf = kernel_material(&q2, us) + settle_d(&q2, -2000, 2000, true, SETTLE_DEPTH);
        if leaf - base >= goal { *deepest = (*deepest).max(4); continue; }
        // (b) no line of CHECKS of ours wins what is missing (a check is a tempo only if the line it starts wins; the
        // checks-only search, every reply searched, two checks -- our moves left inside the 4-ply tier after the harbour;
        // captures are (a), quiet second threats (c)); a mate by checks counts as winning everything
        if crate::threat::gain_by_checks(&q2, crate::threat::GAIN_CHECKS).map_or(false, |(_, v)| mat2 + v >= goal) { *deepest = (*deepest).max(5); continue; }
        // (c), generic: the same with every kind of threat (threat.rs threats: material, promotion, a trapped piece, mate
        // by checks -- a mate threat is a threat like any other, of the game's value)
        let safe = ours2.iter().all(|m2| {
            let q3 = after(&q2, m2);
            let got = m2.capture().map(kernel_value).unwrap_or(0);
            let ts = crate::threat::threats(&q3, us);
            let second = ts.get(1).map_or(0, |t| t.value);
            let best = ts.first().map_or(0, |t| if t.mate || !can_step_away(&q3, t.target, us) { t.value } else { 0 });
            if mat2 + got + second.max(best) >= goal { stage = 7; return false; }
            true
        });
        if safe { HARBOUR.with(|c| c.set(Some(h))); return 2; }
        *deepest = (*deepest).max(stage);
    }
    0
}
pub static CHASE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// set_chase(flag): the no-chase lemma skips the tiers it decides (see chase_lemma).
#[pyfunction]
pub fn set_chase(flag: bool) -> PyResult<()> { CHASE.store(flag, Ordering::Relaxed); Ok(()) }

pub static MINE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
pub type Case = (String, String, String, i64, String, bool, u8, u64, Vec<String>, u8);
thread_local! { static CASES: std::cell::RefCell<Vec<Case>> = const { std::cell::RefCell::new(Vec::new()) }; }
#[allow(clippy::too_many_arguments)]
pub fn mine_push(p: &Chess, uci: &str, s: Square, goal: i64, stopper: &str, holds: bool, tier: u8, nodes: u64, line: Vec<String>, chase: u8) {
    let fen = shakmaty::fen::Fen::from_position(p.clone(), shakmaty::EnPassantMode::Legal).to_string();
    CASES.with(|c| c.borrow_mut().push((fen, uci.to_string(), s.to_string(), goal, stopper.to_string(), holds, tier, nodes, line, chase)));
}
/// set_mine(flag): record every one-target proof question (see waiting.rs prove_t).
#[pyfunction]
pub fn set_mine(flag: bool) -> PyResult<()> { MINE.store(flag, Ordering::Relaxed); Ok(()) }
/// mine_cases() -> the recorded questions since the last call: (fen, uci, target, goal, stopper, holds, tier, nodes, line).
#[pyfunction]
pub fn mine_cases() -> Vec<Case> { CASES.with(|c| std::mem::take(&mut *c.borrow_mut())) }

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
