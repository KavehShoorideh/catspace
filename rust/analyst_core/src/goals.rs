//! THE GOAL FIELD (Kaveh 2026-10-13: "what are the futures from here that I can imagine that are good for me? And what
//! are the futures that are bad for me ... I'm going to try to make those futures come true and then my opponent is going
//! to want to stop those futures"; "we should probably have like a basic move, like a default move. That's just the most
//! natural move ... the highest value immediate move of the opponent"; "go" -- the plan's stages 1-2, depth 3).
//!
//!   THE NATURAL OPPONENT  default_reply: a mate if they have one; else the move leaving them best off after our best
//!                         immediate capture, priced by exchange (it takes, recaptures, saves an attacked piece); else a
//!                         pass when no move beats passing (in check: the least-bad answer)
//!   OUR GOALS             goals: a search over OUR purposeful moves (checks, captures, pawn moves, a move after which the
//!                         piece newly attacks a piece of theirs or a square next to their king, a king step towards their
//!                         king), their natural reply after each, up to `depth` moves of ours; every goal reached recorded
//!                         once at its cheapest -- a mate, material won (net, after their reply), entry into a material
//!                         we know how to win -- with its line. Their goals: the same from their side (pass first).
//!   NOT A PROOF           an imagined future under a natural opponent; whether their best defence stops it is the
//!                         next stage's question (the contest), and only a certified stop will prune.

use super::*;
use crate::waiting::MATE_GOAL;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, AtomicU64};
use shakmaty::Bitboard;

/// How many moves of ours the goal field imagines (3 by default; 5 to be measured).
pub static GOAL_DEPTH: AtomicU32 = AtomicU32::new(3);
/// The safety net on the imagined moves of ours (logged when hit).
pub static GOAL_NODES: AtomicU64 = AtomicU64::new(20_000);

/// What `side` stands on in `q` once the side to move has taken its best immediate capture by exchange: the material
/// balance, less that capture when the other side is to move.
fn standing(q: &Chess, side: Color) -> i64 {
    let bal = kernel_material(q, side);
    if q.turn() == side { return bal; }
    let mover = q.turn();
    let best = (q.board().by_color(side) & !q.board().kings()).into_iter()
        .map(|s| rules::see(q.board(), s, mover)).max().unwrap_or(0).max(0);
    bal - best
}

/// The natural opponent's answer.
#[derive(Clone, Debug)]
pub enum Reply { Play(Move), Pass, Over }

pub fn default_reply(pos: &Chess) -> Reply {
    let me = pos.turn();
    let moves = pos.legal_moves();
    if moves.is_empty() { return Reply::Over; }
    if let Some(m) = moves.iter().find(|m| after(pos, m).is_checkmate()) { return Reply::Play(m.clone()); }
    // the best by what it leaves them after our best capture; ties broken by the move's name (deterministic)
    let mut best: Option<(i64, String, Move)> = None;
    for m in moves.iter() {
        let s = standing(&after(pos, m), me);
        let u = m.to_uci(CastlingMode::Standard).to_string();
        if best.as_ref().map_or(true, |(bs, bu, _)| s > *bs || (s == *bs && u < *bu)) { best = Some((s, u, m.clone())); }
    }
    let (bs, _, bm) = best.unwrap();
    if !pos.is_check() {
        if let Some(p) = pass(pos) { if standing(&p, me) >= bs { return Reply::Pass; } }
    }
    Reply::Play(bm)
}

/// default_reply_of(fen) -> the natural opponent's move (uci), "pass", or None (no legal move).
#[pyfunction]
pub fn default_reply_of(fen: String) -> Option<String> {
    let pos = parse(&fen)?;
    match default_reply(&pos) { Reply::Play(m) => Some(m.to_uci(CastlingMode::Standard).to_string()), Reply::Pass => Some("pass".into()), Reply::Over => None }
}

#[derive(Clone, Debug)]
pub struct Goal { pub kind: &'static str, pub what: String, pub value: i64, pub cost: u32, pub line: Vec<String> }

/// How many of its own moves the piece on `from` needs (the others standing still, through empty squares) before it
/// attacks a square of `targets` -- 0 if it already does; None beyond `cap` (the piece_reach walk, 2026-09-29, in Rust).
fn moves_to_attack(b: &shakmaty::Board, from: Square, targets: Bitboard, cap: u32) -> Option<u32> {
    let p = b.piece_at(from)?;
    let occ = b.occupied() & !Bitboard::from_square(from);
    let attacks_from = |s: Square| if p.role == Role::Pawn { shakmaty::attacks::pawn_attacks(p.color, s) } else { shakmaty::attacks::attacks(s, p, occ) };
    let mut seen = Bitboard::from_square(from);
    let mut front = vec![from];
    for d in 0..=cap {
        if front.iter().any(|s| !(attacks_from(*s) & targets).is_empty()) { return Some(d); }
        if d == cap { break; }
        let mut next = Vec::new();
        for s in front {
            let steps = if p.role == Role::Pawn { Bitboard::EMPTY } else { shakmaty::attacks::attacks(s, p, occ) & !b.occupied() };
            for t in steps & !seen { seen.add(t); next.push(t); }
        }
        if next.is_empty() { break; }
        front = next;
    }
    None
}

/// Is our move `m` purposeful -- does it serve some target? A check, a capture, a pawn move, or a move after which the
/// moved piece is FEWER of its own moves from attacking a target (a piece of theirs, a square next to their king) --
/// "brings a piece closer", the quiet approach a mating net needs (2026-10-13: "newly attacks" alone imagined 3 moves
/// in a K+B+N v K position and found nothing).
fn purposeful(pos: &Chess, m: &Move, them: Color) -> bool {
    if m.is_capture() || m.is_promotion() || m.role() == Role::Pawn { return true; }
    let q = after(pos, m);
    if q.is_check() { return true; }
    let Some(tk) = pos.board().king_of(them) else { return false };
    let Some(from) = m.from() else { return false };
    let targets = (pos.board().by_color(them) & !pos.board().kings()) | shakmaty::attacks::king_attacks(tk);
    let before = moves_to_attack(pos.board(), from, targets, 4).unwrap_or(5);
    let now = moves_to_attack(q.board(), m.to(), targets, 4).unwrap_or(5);
    now < before || (now == 0 && {
        let p = q.board().piece_at(m.to());
        p.map_or(false, |p| (shakmaty::attacks::attacks(m.to(), p, q.board().occupied()) & targets).count()
            > (shakmaty::attacks::attacks(from, p, pos.board().occupied()) & targets).count())
    })
}

/// Their pieces gone and ours gone between `a` and `b`, in words.
fn change(a: &Chess, b: &Chess, us: Color) -> String {
    let name = |r: Role| match r { Role::Queen => "queen", Role::Rook => "rook", Role::Bishop => "bishop", Role::Knight => "knight", Role::Pawn => "pawn", Role::King => "king" };
    let mut out = Vec::new();
    for (side, word) in [(!us, "their"), (us, "our")] {
        for r in [Role::Queen, Role::Rook, Role::Bishop, Role::Knight, Role::Pawn] {
            let n = (a.board().by_color(side) & a.board().by_role(r)).count() as i64 - (b.board().by_color(side) & b.board().by_role(r)).count() as i64;
            if n > 0 { out.push(format!("{word} {}{}", name(r), if n > 1 { format!(" x{n}") } else { String::new() })); }
            if n < 0 && side == us && r == Role::Queen { out.push("our promotion".into()); }
        }
    }
    if out.is_empty() { "nothing".into() } else { out.join(", ") }
}

struct Walk<'a> { us: Color, root: &'a Chess, root_bal: i64, root_key: String, nodes: u64, cap: u64, found: HashMap<(&'static str, String), Goal> }

impl Walk<'_> {
    fn record(&mut self, g: Goal) {
        let k = (g.kind, g.what.clone());
        let better = self.found.get(&k).map_or(true, |o| g.cost < o.cost || (g.cost == o.cost && g.value > o.value));
        if better { self.found.insert(k, g); }
    }
    fn go(&mut self, pos: &Chess, left: u32, cost: u32, line: &mut Vec<String>, high: i64) {
        if left == 0 { return; }
        let them = !self.us;
        for m in pos.legal_moves() {
            if self.nodes >= self.cap { return; }
            if !purposeful(pos, &m, them) { continue; }
            self.nodes += 1;
            let q = after(pos, &m);
            line.push(crate::plan::san(pos, &m.to_uci(CastlingMode::Standard).to_string()));
            if q.is_checkmate() {
                self.record(Goal { kind: "mate", what: "their king".into(), value: MATE_GOAL, cost: cost + 1, line: line.clone() });
                line.pop(); continue;
            }
            let (r, said) = match default_reply(&q) {
                Reply::Over => { line.pop(); continue; }                       // a stalemate: no goal down this line
                Reply::Pass => (pass(&q).unwrap_or_else(|| q.clone()), "(pass)".to_string()),
                Reply::Play(rm) => { let s = crate::plan::san(&q, &rm.to_uci(CastlingMode::Standard).to_string()); (after(&q, &rm), s) }
            };
            line.push(said);
            if r.is_checkmate() { line.pop(); line.pop(); continue; }        // their natural reply mates us: a bad line
            let bal = kernel_material(&r, self.us) - self.root_bal;
            // a NEW high of material won along the line (losing some of it later is not another goal)
            if bal > high.max(0) {
                self.record(Goal { kind: "material", what: change(self.root, &r, self.us), value: bal, cost: cost + 1, line: line.clone() });
            }
            let key = crate::vocab::material_key(r.board(), self.us);
            if key != self.root_key && bal >= 0 && crate::vocab::is_known(&key) {
                self.record(Goal { kind: "known", what: key, value: bal, cost: cost + 1, line: line.clone() });
            }
            if r.turn() == self.us { self.go(&r, left - 1, cost + 1, line, high.max(bal)); }
            line.pop(); line.pop();
        }
    }
}

/// The goals of the side to move in `pos`, imagined up to `depth` of its moves against the natural opponent: best first
/// (the value, then the cost).
pub fn goals(pos: &Chess, depth: u32) -> (Vec<Goal>, u64) {
    let us = pos.turn();
    let mut w = Walk { us, root: pos, root_bal: kernel_material(pos, us), root_key: crate::vocab::material_key(pos.board(), us),
                       nodes: 0, cap: GOAL_NODES.load(Ordering::Relaxed), found: HashMap::new() };
    // DEEPENING STEP BY STEP -- depth 1, 2, ... -- so the shallow goals are found before the cap (2026-10-13: searched
    // depth-first at once, depth 5 spent the cap deep and found none of the two goals depth 3 found)
    for d in 1..=depth {
        if w.nodes >= w.cap { break; }
        w.go(pos, d, 0, &mut Vec::new(), 0);
    }
    let mut v: Vec<Goal> = w.found.into_values().collect();
    v.sort_by(|a, b| (-a.value, a.cost, &a.what).cmp(&(-b.value, b.cost, &b.what)));
    (v, w.nodes)
}

/// goals_of(fen, depth, side) -> ([(kind, what, value, cost, line)], imagined moves of ours) -- side "us" (the side to
/// move) or "them" (the other side, as if we passed; none when we are in check).
#[pyfunction]
pub fn goals_of(fen: String, depth: u32, side: String) -> (Vec<(String, String, i64, u32, String)>, u64) {
    let Some(pos) = parse(&fen) else { return (vec![], 0) };
    let p = if side == "them" { match pass(&pos) { Some(p) => p, None => return (vec![], 0) } } else { pos };
    let (g, n) = goals(&p, depth);
    (g.into_iter().map(|g| (g.kind.to_string(), g.what, g.value, g.cost, g.line.join(" "))).collect(), n)
}

/// set_goal_depth(d): how many moves of ours the goal field imagines.
#[pyfunction]
pub fn set_goal_depth(d: u32) -> PyResult<()> { GOAL_DEPTH.store(d, Ordering::Relaxed); Ok(()) }
