//! THE KING-AND-QUEEN MATE IN ACTION (Kaveh 2026-10-12: "put the proved rules into Rust ... the king and queen tactic").
//!
//! PROVED (Lean `Ideas.QueenMate.kq_mate_on_edge`, from the library's checkmate): a king and a queen mate a lone king only
//! on the edge. So the lone king must be driven there. THE REGION -- the squares it can walk to without stepping onto a
//! square we attack (our pieces held still; `region`) -- is measured BY THE THEOREM: its squares OFF THE EDGE must go
//! (only on the edge can the king be mated), then the region shrinks, our king comes up, and the planner's mate by checks
//! (its pre-pass) finishes. (A first version minimized the region alone: 31 of 40 play-outs mated, 9 repeated -- the
//! king boxed on e7 / f8, two squares, one off the edge, and no shrinking left without stalemate.) Each move of ours:
//!   NEVER  a stalemate (Lean `Ideas.StalemateLemmas`: the move must leave them a move unless it mates), a queen left to
//!          the lone king, a claimable draw -- after our move or after any reply (the game context: a third occurrence,
//!          the fifty-move clock)
//!   FIRST  THE NET: a region of at most two squares -- a forced mate within three moves, searched in full (its measure:
//!          the moves to mate); else the lowest MEASURE (`KQ_MEASURE`, by rank: the region's squares off the edge; our
//!          king's distance to the FARTHEST front-zone target the proved shapes give -- Lean edge_shape / corner_shape --
//!          which SUPERSEDES the region's size; the region's size) AGAINST EVERY REPLY; then the measure right after our
//!          move; then a position the game has not seen.
//!   ELSE   THE TEMPO MOVE: the first move after which every reply leaves us a move whose measure falls -- within two
//!          moves, else three (KQ_TEMPO).
//! Every move carries its CERTIFICATE: the hypothesis of the method theorem (Lean `Ideas.Forced.forced_by_measure`: a
//! frame whose every position forces the goal or a lower measure within k moves forces the goal) checked on the board.
//! OBS 1,000 play-outs: every move certified (125 tempo moves), 6.3 moves to mate, at most 12. That EVERY position of the
//! frame has a certificate is checked, not proved -- the rung lemmas over patterns are the next proof.
use super::*;
use shakmaty::Bitboard;

/// The side to move has exactly a king and a queen, the other side only its king.
pub fn is_kq_vs_k(pos: &Chess) -> bool {
    let b = pos.board();
    let us = b.by_color(pos.turn());
    let them = b.by_color(!pos.turn());
    them.count() == 1 && us.count() == 2 && (us & b.queens()).count() == 1
}

/// THE REGION of the lone king of colour `lone` in `q`: the squares it reaches by king steps over squares the other side
/// does not attack (attacks read with the lone king lifted -- it cannot hide behind itself), its own square included.
pub fn region(q: &Chess, lone: Color) -> u32 { region_bb(q, lone).count() as u32 }

/// The region as squares.
pub fn region_bb(q: &Chess, lone: Color) -> Bitboard {
    let b = q.board();
    let Some(k) = b.king_of(lone) else { return Bitboard::EMPTY };
    let occ = b.occupied().without(k);
    let safe = |s: Square| !b.attacks_to(s, !lone, occ).any();
    let mut seen = Bitboard::from_square(k);
    let mut frontier = vec![k];
    while let Some(s) = frontier.pop() {
        for t in shakmaty::attacks::king_attacks(s) & !seen {
            if b.by_color(lone).contains(t) || !safe(t) { continue; }
            seen.add(t);
            frontier.push(t);
        }
    }
    seen
}

/// The squares of the board's edge.
fn edge() -> Bitboard { Bitboard::from(shakmaty::Rank::First) | Bitboard::from(shakmaty::Rank::Eighth) | Bitboard::from(shakmaty::File::A) | Bitboard::from(shakmaty::File::H) }

fn kdist(a: Square, b: Square) -> u32 { a.distance(b) }

/// PROGRESS IN EVERY FRAME (Kaveh 2026-10-12: "we need a notion of progress in any frame of any lemma ... anti-progress on
/// one lemma, progress on another"): a lemma's MEASURE is a named component; a frame's measure is its components in RANK
/// order, compared lexicographically -- a step may worsen a lower component only while a higher one strictly improves.
/// A component that may be worsened by a higher one says so: the higher one SUPERSEDES it (Kaveh: "acknowledge in this
/// lemma that you are violating this other lemma and index on it"), under a condition.
pub struct Component {
    pub id: &'static str,
    pub name: &'static str,
    /// the proved statement it serves (a Lean name), if any
    pub lean: Option<&'static str>,
    /// (the component it may worsen, when)
    pub supersedes: Option<(&'static str, &'static str)>,
    /// the measure, read with the attacker to move (`lone` the lone king's colour)
    pub value: fn(&Chess, Color) -> u32,
}

fn off_edge(q: &Chess, lone: Color) -> u32 { (region_bb(q, lone) & !edge()).count() as u32 }

/// THE FRONT ZONE of a region (Lean `Ideas.QueenMate.edge_shape`, `corner_shape`: in every king-and-queen mate our king
/// stands two squares from theirs, on the inside -- two lines in front within a line on an edge, at distance two in a
/// corner): the squares our king must reach, read off the region's edge squares. The defender's king moving inside its
/// box does not move the zone.
pub fn front_zone(region: Bitboard) -> Bitboard {
    let mut z = Bitboard::EMPTY;
    for r in region & edge() {
        let (f, k) = (r.file() as i32, r.rank() as i32);
        for s in Bitboard::FULL {
            let (sf, sk) = (s.file() as i32, s.rank() as i32);
            let corner = (f == 0 || f == 7) && (k == 0 || k == 7);
            let ok = if corner { r.distance(s) == 2 }
                else if k == 0 { sk == 2 && (sf - f).abs() <= 1 }
                else if k == 7 { sk == 5 && (sf - f).abs() <= 1 }
                else if f == 0 { sf == 2 && (sk - k).abs() <= 1 }
                else { sf == 5 && (sk - k).abs() <= 1 };
            if ok { z.add(s); }
        }
    }
    z
}

/// OUR KING'S WAY TO THE FRONT ZONE of the lone king's region (before the region reaches the edge: 8 plus the kings'
/// distance -- the edge component ranks above it then anyway). THE FARTHEST TARGET (APPROACH_MAX): the largest, over the
/// region's edge squares, of our king's distance to that square's zone -- the mate may come on any of them, so our king
/// must be near all; shrinking the region can only lower it (the nearest target moved away when the queen cut the region
/// along the edge: 144 uncertified moves of 150 games, 2026-10-12).
fn approach(q: &Chess, lone: Color) -> u32 {
    let b = q.board();
    let (Some(a), Some(k)) = (b.king_of(!lone), b.king_of(lone)) else { return 0 };
    if !APPROACH_ZONE.load(Ordering::Relaxed) { return kdist(a, k); }
    let r = region_bb(q, lone);
    if APPROACH_MAX.load(Ordering::Relaxed) {
        return (r & edge()).into_iter().map(|e| front_zone(Bitboard::from_square(e)).into_iter().map(|s| kdist(a, s)).min().unwrap_or(8))
            .max().unwrap_or(8 + kdist(a, k));
    }
    let z = front_zone(r);
    z.into_iter().map(|s| kdist(a, s)).min().unwrap_or(8 + kdist(a, k))
}
pub static APPROACH_MAX: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);
#[pyfunction]
pub fn set_approach_max(flag: bool) -> PyResult<()> { APPROACH_MAX.store(flag, Ordering::Relaxed); Ok(()) }

/// THE NET (the last rung): a forced mate in at most `n` moves of ours against every reply, every move of ours tried (a
/// small full-width proof -- the lone king has at most eight replies). -> the first move and the number of moves.
pub fn net_mate(pos: &Chess, n: u32) -> Option<(Move, u32)> {
    fn mates(p: &Chess, n: u32) -> Option<Move> {
        for m in p.legal_moves() {
            let q = after(p, &m);
            if q.is_checkmate() { return Some(m); }
            if n <= 1 { continue; }
            let rs = q.legal_moves();
            if rs.is_empty() { continue; }                                             // stalemate
            if crate::game::claimable_draw(&q).is_some() { continue; }
            if rs.iter().all(|r| { let x = after(&q, r); crate::game::claimable_draw(&x).is_none() && mates(&x, n - 1).is_some() }) { return Some(m); }
        }
        None
    }
    (1..=n).find_map(|k| mates(pos, k).map(|m| (m, k)))
}
pub const NET_REGION: u32 = 2;
pub const NET_MOVES: u32 = 3;

/// The approach measured to the front zone (on) or to their king (off -- the first version: the defender undoes it).
pub static APPROACH_ZONE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);
#[pyfunction]
pub fn set_approach_zone(flag: bool) -> PyResult<()> { APPROACH_ZONE.store(flag, Ordering::Relaxed); Ok(()) }
fn region_size(q: &Chess, lone: Color) -> u32 { region(q, lone) }

/// THE KING-AND-QUEEN FRAME'S MEASURE, by rank.
pub const KQ_MEASURE: [Component; 3] = [
    Component { id: "edge", name: "region squares off the edge", lean: Some("Ideas.QueenMate.kq_mate_on_edge"), supersedes: None, value: off_edge },
    Component { id: "approach", name: "our king's distance to the front zone", lean: Some("Ideas.QueenMate.edge_shape"),
                supersedes: Some(("region", "the region may grow while our king comes up -- the way round to the mating square")), value: approach },
    Component { id: "region", name: "the region's size", lean: None, supersedes: None, value: region_size },
];

pub fn measure(cs: &[Component], q: &Chess, lone: Color) -> Vec<u32> { cs.iter().map(|c| (c.value)(q, lone)).collect() }

/// The worst the measure can be after our move `q` (the lone king to move): over every reply. A mated king: all zero.
fn worst_after(cs: &[Component], q: &Chess, lone: Color) -> Vec<u32> {
    q.legal_moves().iter().map(|r| measure(cs, &after(q, r), lone)).max().unwrap_or_else(|| vec![0; cs.len()])
}

/// The explanation of a step: which component made progress, which it superseded.
pub fn explain(cs: &[Component], before: &[u32], worst: &[u32]) -> String {
    let Some(i) = (0..cs.len()).find(|i| worst[*i] != before[*i]) else { return "no component moves".to_string() };
    let mut out = format!("{} {} -> {}", cs[i].name, before[i], worst[i]);
    if worst[i] > before[i] { return format!("no progress against every reply ({out})"); }
    if let Some(l) = cs[i].lean { out.push_str(&format!(" (toward {l})")); }
    for j in i + 1..cs.len() {
        if worst[j] > before[j] {
            let by = cs[..=i].iter().rev().find(|c| c.supersedes.map_or(false, |(x, _)| x == cs[j].id));
            out.push_str(&format!("; {} {} -> {}{}", cs[j].name, before[j], worst[j],
                by.map(|c| format!(" (superseded by {}: {})", c.id, c.supersedes.unwrap().1)).unwrap_or_default()));
        }
    }
    out
}

/// The confining move for king and queen against a king (the side to move has them): a mate if one is on the board, else
/// the move whose measure is lowest AGAINST EVERY REPLY (then the lowest right after it, then a position not yet seen).
/// -> (move, the region after it, the explanation).
pub fn kq_move(pos: &Chess) -> Option<(Move, u32, String)> {
    if !is_kq_vs_k(pos) { return None; }
    let us = pos.turn();
    let before = measure(&KQ_MEASURE, pos, !us);
    // THE NET: a region of at most two squares -- the mate is forced within a few moves (its measure: the moves to mate)
    if before[2] <= NET_REGION {
        if let Some((m, k)) = net_mate(pos, NET_MOVES) { return Some((m, before[2], format!("the net: mate in {k} against every reply"))); }
    }
    let mut cands: Vec<((Vec<u32>, Vec<u32>, bool), Move, u32)> = Vec::new();
    for m in pos.legal_moves() {
        let q = after(pos, &m);
        if q.is_checkmate() { return Some((m, 0, "mate".to_string())); }
        if q.legal_moves().is_empty() { continue; }                                   // stalemate
        if crate::game::claimable_draw(&q).is_some() { continue; }                     // a draw they may claim
        if q.legal_moves().iter().any(|r| crate::game::claimable_draw(&after(&q, r)).is_some()) { continue; }   // or reach
        let b = q.board();
        let (Some(ours), Some(theirs)) = (b.king_of(us), b.king_of(!us)) else { continue };
        let Some(qs) = (b.queens() & b.by_color(us)).first() else { continue };
        let hit = shakmaty::attacks::king_attacks(theirs).contains(qs);
        let guarded = shakmaty::attacks::king_attacks(ours).contains(qs);
        if hit && !guarded { continue; }                                               // the queen left to the king
        let now = measure(&KQ_MEASURE, &q, !us);
        let key = if KQ_EVERY_REPLY.load(Ordering::Relaxed) { (worst_after(&KQ_MEASURE, &q, !us), now, crate::game::seen(&q) > 0) }
                  else { (now.clone(), now, crate::game::seen(&q) > 0) };
        cands.push((key, m, region(&q, !us)));
    }
    cands.sort_by(|a, b| a.0.cmp(&b.0));
    let (_, m0, r0) = cands.first().cloned()?;
    let q0 = after(pos, &m0);
    let w0 = worst_after(&KQ_MEASURE, &q0, !us);
    // THE CERTIFICATE: progress against every reply (one move) -- else THE TEMPO (Kaveh: "my tempo isn't right, so let me
    // shuffle around keeping the king here"): the first move, by the order, after which EVERY reply leaves us a move whose
    // measure is below where we started against every reply -- progress within two moves, found by search and named
    if w0 < before || !KQ_TEMPO.load(Ordering::Relaxed) { return Some((m0, r0, explain(&KQ_MEASURE, &before, &w0))); }
    for k in 2..=KQ_TEMPO_MOVES {
        for (_, m, r) in &cands {
            let q = after(pos, m);
            if progress_within(&q, &before, us, k) {
                return Some((m.clone(), *r, format!("a tempo move: progress within {k} moves against every reply ({})", explain(&KQ_MEASURE, &before, &worst_after(&KQ_MEASURE, &q, !us)))));
            }
        }
    }
    Some((m0, r0, explain(&KQ_MEASURE, &before, &w0)))
}
/// How far ahead a tempo move's progress may come (moves of ours, this one included).
pub const KQ_TEMPO_MOVES: u32 = 3;

/// A move of ours safe for the frame: no stalemate, no claimable draw now or after a reply, the queen not left to the king.
fn safe_step(q: &Chess, us: Color) -> bool {
    if q.is_checkmate() { return true; }
    if q.legal_moves().is_empty() || crate::game::claimable_draw(q).is_some() { return false; }
    if q.legal_moves().iter().any(|r| crate::game::claimable_draw(&after(q, r)).is_some()) { return false; }
    let b = q.board();
    let (Some(ours), Some(theirs), Some(qs)) = (b.king_of(us), b.king_of(!us), (b.queens() & b.by_color(us)).first()) else { return false };
    !(shakmaty::attacks::king_attacks(theirs).contains(qs) && !shakmaty::attacks::king_attacks(ours).contains(qs))
}

/// After our move (`q`, they to move): for every reply, a safe move of ours after which the measure, against every reply,
/// is below `before` (or a mate) -- within `k` moves of ours counting the one that led to `q`.
fn progress_within(q: &Chess, before: &[u32], us: Color, k: u32) -> bool {
    if k <= 1 { return q.is_checkmate() || worst_after(&KQ_MEASURE, q, !us).as_slice() < before; }
    q.legal_moves().iter().all(|r| {
        let x = after(q, r);
        x.legal_moves().iter().any(|m2| {
            let y = after(&x, m2);
            y.is_checkmate() || (safe_step(&y, us) && progress_within(&y, before, us, k - 1))
        })
    })
}

/// The tempo rung on (a move whose progress comes within two moves), or off.
pub static KQ_TEMPO: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);
#[pyfunction]
pub fn set_kq_tempo(flag: bool) -> PyResult<()> { KQ_TEMPO.store(flag, Ordering::Relaxed); Ok(()) }

/// CHOOSE by the measure over every reply (on), or right after our move (off, the default). Over every reply the approach
/// is undone by the defender -- their king steps between two squares of its box, our king on the wrong side never gets
/// "nearer" (OBS 998 / 1,000 mated, 11.1 moves; after our move 1,000 / 1,000, 9.2, 2026-10-12): our king's distance to
/// theirs is no ranking function. The every-reply measure is reported as each move's certificate instead.
pub static KQ_EVERY_REPLY: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);
#[pyfunction]
pub fn set_kq_every_reply(flag: bool) -> PyResult<()> { KQ_EVERY_REPLY.store(flag, Ordering::Relaxed); Ok(()) }
