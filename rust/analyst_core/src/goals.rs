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
pub struct Goal { pub kind: &'static str, pub what: String, pub value: i64, pub cost: u32, pub line: Vec<String>, pub path: Vec<Square> }

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
    fn go(&mut self, pos: &Chess, left: u32, cost: u32, line: &mut Vec<String>, path: &mut Vec<Square>, high: i64) {
        if left == 0 { return; }
        let them = !self.us;
        for m in pos.legal_moves() {
            if self.nodes >= self.cap { return; }
            if !purposeful(pos, &m, them) { continue; }
            self.nodes += 1;
            let q = after(pos, &m);
            line.push(crate::plan::san(pos, &m.to_uci(CastlingMode::Standard).to_string()));
            path.push(m.to());
            if q.is_checkmate() {
                self.record(Goal { kind: "mate", what: "their king".into(), value: MATE_GOAL, cost: cost + 1, line: line.clone(), path: path.clone() });
                line.pop(); path.pop(); continue;
            }
            let (r, said) = match default_reply(&q) {
                Reply::Over => { line.pop(); path.pop(); continue; }           // a stalemate: no goal down this line
                Reply::Pass => (pass(&q).unwrap_or_else(|| q.clone()), "(pass)".to_string()),
                Reply::Play(rm) => { let s = crate::plan::san(&q, &rm.to_uci(CastlingMode::Standard).to_string()); (after(&q, &rm), s) }
            };
            line.push(said);
            if r.is_checkmate() { line.pop(); line.pop(); path.pop(); continue; }   // their natural reply mates us: a bad line
            let bal = kernel_material(&r, self.us) - self.root_bal;
            // a NEW high of material won along the line (losing some of it later is not another goal)
            if bal > high.max(0) {
                self.record(Goal { kind: "material", what: change(self.root, &r, self.us), value: bal, cost: cost + 1, line: line.clone(), path: path.clone() });
            }
            let key = crate::vocab::material_key(r.board(), self.us);
            if key != self.root_key && bal >= 0 && crate::vocab::is_known(&key) {
                self.record(Goal { kind: "known", what: key, value: bal, cost: cost + 1, line: line.clone(), path: path.clone() });
            }
            if r.turn() == self.us { self.go(&r, left - 1, cost + 1, line, path, high.max(bal)); }
            line.pop(); line.pop(); path.pop();
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
        w.go(pos, d, 0, &mut Vec::new(), &mut Vec::new(), 0);
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


// ---------------------------------------------------------------------------------------------------------------------
// STAGE 4, THE CONTEST (the plan: "for each good future ... which of their pieces can reach a square of our trajectory in
// time ... prune only on a certified interception, and search inside the zones of the rest"). An imagined goal against
// their BEST defence: a proof search whose moves of ours stay in the goal's ZONE (purposeful moves, moves onto the path
// or the interceptors' squares, moves attacking an interceptor) and whose replies are ALL of theirs, within the goal's
// cost and one move more. Forced -> the goal holds, its line; not found in the zone -> the same bound with EVERY move of
// ours: not found there either -> CERTIFIED STOPPED within the bound (the only verdict that prunes); out of budget ->
// UNKNOWN (kept). The interceptors -- their pieces that can reach or attack a square of our path within the goal's time
// -- are the explanation of a contest.

/// The proof search's budget per question (expansions).
pub static CONTEST_BUDGET: AtomicU64 = AtomicU64::new(30_000);

#[derive(Clone)]
pub struct CState { pos: Chess, mv: Option<Move>, ours: u32 }

struct Contest { us: Color, kind: &'static str, value: i64, key: String, root_bal: i64, depth: u32, full: bool, zone: Bitboard }

impl Contest {
    /// The goal holds (our turn, after their reply).
    fn reached(&self, pos: &Chess) -> bool {
        let gain = kernel_material(pos, self.us) - self.root_bal;
        match self.kind {
            "material" => gain >= self.value,
            "known" => gain >= self.value && crate::vocab::material_key(pos.board(), self.us) == self.key,
            _ => false,
        }
    }
    fn in_zone(&self, pos: &Chess, m: &Move) -> bool {
        if self.zone.contains(m.to()) || purposeful(pos, m, !self.us) { return true; }
        let q = after(pos, m);
        q.board().piece_at(m.to()).map_or(false, |p| !(shakmaty::attacks::attacks(m.to(), p, q.board().occupied()) & self.zone).is_empty())
    }
}

impl crate::pns::Problem for Contest {
    type State = CState;
    fn judge(&self, s: &CState, or: bool, root: bool) -> Option<bool> {
        if !root && crate::game::claimable_draw(&s.pos).is_some() { return Some(false); }
        if or {
            if s.pos.legal_moves().is_empty() { return Some(false); }        // we are mated or stalemated
            if !root && self.reached(&s.pos) { return Some(true); }
            if s.ours >= self.depth { return Some(false); }
            None
        } else if s.pos.legal_moves().is_empty() { Some(s.pos.is_check()) } // they are mated (better than any goal) or stalemated
        else { None }
    }
    fn expand(&self, s: &CState, or: bool) -> Vec<(CState, f64)> {
        let us_next = if or { s.ours + 1 } else { s.ours };
        s.pos.legal_moves().into_iter().filter(|m| !or || self.full || self.in_zone(&s.pos, m))
            .map(|m| (CState { pos: after(&s.pos, &m), mv: Some(m), ours: us_next }, 1.0)).collect()
    }
    fn and_size(&self, s: &CState) -> f64 { s.pos.legal_moves().len().max(1) as f64 }
    fn key(&self, s: &CState) -> u64 { crate::threat::zkey(&s.pos) }
}

#[derive(Clone, Debug)]
pub struct Verdict { pub status: &'static str, pub line: Vec<String>, pub interceptors: Vec<String>, pub first: Option<Move> }

/// Their pieces that can stand on or attack a square of our path within `time` of their moves.
fn interceptors(pos: &Chess, us: Color, path: Bitboard, time: u32) -> (Vec<String>, Bitboard) {
    let b = pos.board();
    let mut words = Vec::new();
    let mut squares = Bitboard::EMPTY;
    for s in b.by_color(!us) {
        let Some(p) = b.piece_at(s) else { continue };
        if let Some(n) = moves_to_attack(b, s, path, time) {
            squares.add(s);
            let name = match p.role { Role::Queen => "queen", Role::Rook => "rook", Role::Bishop => "bishop", Role::Knight => "knight", Role::Pawn => "pawn", Role::King => "king" };
            words.push(format!("their {name} on {s} reaches the path in {n}"));
        }
    }
    (words, squares)
}

pub fn contest(pos: &Chess, g: &Goal) -> Verdict {
    let us = pos.turn();
    let path: Bitboard = g.path.iter().copied().collect();
    let (words, squares) = interceptors(pos, us, path, g.cost);
    let mut zone = path | squares;
    if let Some(tk) = pos.board().king_of(!us) { if g.kind == "mate" { zone |= shakmaty::attacks::king_attacks(tk); } }
    let budget = CONTEST_BUDGET.load(Ordering::Relaxed);
    let mk = |full: bool, depth: u32| Contest { us, kind: g.kind, value: g.value, key: g.what.clone(), root_bal: kernel_material(pos, us), depth, full, zone };
    let root = CState { pos: pos.clone(), mv: None, ours: 0 };
    let line_of = |l: &Vec<CState>| -> Vec<String> {
        l.windows(2).filter_map(|w| w[1].mv.as_ref().map(|m| crate::plan::san(&w[0].pos, &m.to_uci(CastlingMode::Standard).to_string()))).collect()
    };
    let forced = |lines: &Vec<Vec<CState>>| Verdict { status: "forced", line: lines.first().map(line_of).unwrap_or_default(),
                                                   first: lines.first().and_then(|l| l.get(1)).and_then(|s| s.mv.clone()), interceptors: words.clone() };
    // THE SHORTEST FORCED LINE FIRST -- 1 move of ours, 2, ... the goal's cost plus one (2026-10-13, the field's first
    // measurement: proving "within the cost plus one" and playing any proof's first move, it promised "mate in 3" and
    // re-promised it two moves later, the bishop shuffling -- 3 of 64 promises kept on time in K+R+R v K+R)
    let mut unknown = false;
    for depth in 1..=g.cost + 1 {
        let (r, lines) = crate::pns::search_proof_lines(&mk(false, depth), root.clone(), budget, 1);
        match r { Some(true) => return forced(&lines), None => unknown = true, Some(false) => {} }
    }
    if unknown { return Verdict { status: "unknown", line: vec![], first: None, interceptors: words }; }
    let (r, lines) = crate::pns::search_proof_lines(&mk(true, g.cost + 1), root, budget, 1);
    match r {
        Some(true) => forced(&lines),
        Some(false) => Verdict { status: "stopped", line: vec![], first: None, interceptors: words },
        None => Verdict { status: "unknown", line: vec![], first: None, interceptors: words },
    }
}

/// From the other side's words to ours ("their" <-> "our").
fn flip(s: &str) -> String { s.replace("their ", "\u{0}").replace("our ", "their ").replace('\u{0}', "our ") }

/// field_of(fen, depth) -> (our goals, their goals as if we passed), each [(kind, what, value, cost, imagined line,
/// verdict, the line against their best defence, interceptors)] -- in our words, worst of theirs first.
#[pyfunction]
pub fn field_of(fen: String, depth: u32) -> (Vec<(String, String, i64, u32, String, String, String, Vec<String>)>, Vec<(String, String, i64, u32, String, String, String, Vec<String>)>) {
    let Some(pos) = parse(&fen) else { return (vec![], vec![]) };
    let row = |p: &Chess, g: Goal, theirs: bool| {
        let v = contest(p, &g);
        let w = |s: &str| if theirs { flip(s) } else { s.to_string() };
        (g.kind.to_string(), w(&g.what), g.value, g.cost, g.line.join(" "), v.status.to_string(), v.line.join(" "),
         v.interceptors.iter().map(|x| w(x)).collect())
    };
    let ours = goals(&pos, depth).0.into_iter().map(|g| row(&pos, g, false)).collect();
    let theirs = match pass(&pos) { Some(p) => goals(&p, depth).0.into_iter().map(|g| row(&p, g, true)).collect(), None => vec![] };
    (ours, theirs)
}


// ---------------------------------------------------------------------------------------------------------------------
// STAGE 5, SERVE (the plan: "the planner pursues our best live goal or denies their worst, re-planning every move"). When
// the planner's own search proved no gain: our best goal FORCED against every defence (up to four contested, the most
// valuable first) -- its proof's first move, explained -- unless, as if we passed, they have a FORCED goal at least as
// valuable and no slower: then the field steps aside and the planner's defence answers it (the threat machinery already
// does denial; not duplicated). Nothing forced: the frames and the mate search decide, as before. Behind a switch (off
// until measured).
pub static GOAL_FIELD: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
#[pyfunction]
pub fn set_goal_field(flag: bool) -> PyResult<()> { GOAL_FIELD.store(flag, Ordering::Relaxed); Ok(()) }

pub fn field_move(pos: &Chess) -> Option<(Move, i64, String)> {
    let depth = GOAL_DEPTH.load(Ordering::Relaxed);
    let (ours, _) = goals(pos, depth);
    let mut best = None;
    for g in ours.into_iter().filter(|g| g.value > 0).take(4) {
        let v = contest(pos, &g);
        if v.status == "forced" { best = Some((g, v)); break; }
    }
    let (g, v) = best?;
    if let Some(p) = pass(pos) {
        for t in goals(&p, depth).0.into_iter().filter(|t| t.value >= g.value && t.cost <= g.cost).take(3) {
            if contest(&p, &t).status == "forced" { return None; }
        }
    }
    let m = v.first.clone()?;
    let what = if g.kind == "mate" { "mate".to_string() } else if g.kind == "known" { format!("entry into {}", g.what) } else { format!("winning {}", g.what) };
    let n = (v.line.len() as u32 + 1) / 2;                                 // our moves in the proved line
    Some((m, g.value.min(MATE_GOAL), format!("the goal field: {what} in {n} (+{}), forced against every defence: {}", g.value, v.line.join(" "))))
}
