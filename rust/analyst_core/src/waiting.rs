//! WAITING FRAMES (Kaveh 2026-10-09: "keep the frames and the facts in the frames, don't discard them ... a live index
//! of who's waiting on what kind of condition ... a language to communicate what condition"; "if there were another
//! attacker I'd win -- how likely can I bring one?").
//!
//! A FRAME is a seed move with a focus and a goal: a check against the king (goal: mate), a new threat or fork, a
//! cornered or trapped piece (goal: the rule layer's gain). Its certificate is the value search (`certified_value`:
//! our moves restricted to the frame's answers, EVERY reply of theirs searched).
//!   active    the certificate holds now
//!   dormant   it fails, but ONE change to the board inside the frame makes it hold -- the MISSING CONDITION
//!   far       it fails and no single change fixes it
//! THE MISSING CONDITION, found generically (no extractor per tactic family): the smallest perturbation of the board
//! inside the frame under which the certificate passes -- add one of our pieces on a square bearing on the frame,
//! remove one of theirs (a defender, a blocker), vacate one of ours (a blocker). Each condition is an ATOM over one
//! square; a frame's condition is the OR of the atoms that work. The certificates run in parallel on every core.
//! THE POOL keeps frames across moves and reports what changed (new, dormant -> active, active -> broken, dead). By
//! default every frame is recomputed every move (`exact`): reuse of frames whose footprint no move touched (occupancy
//! and attack changes) was audited on two games at 1.0% stale statuses, six missed tactics among them, for 14% of the
//! work saved (2026-10-09) -- kept as an option, off.
//! THE REVERSE LOOKUP: which of our moves makes a dormant frame of ours hold on the null move (it ARMS a threat), and
//! which makes a dormant frame of theirs hold (it HANDS them a tactic).
use super::*;
use rayon::prelude::*;
use shakmaty::{Bitboard, EnPassantMode};

pub const MATE_GOAL: i64 = 500;
/// WAITING CONDITIONS ONE MOVE AWAY (2026-10-09, measured: 85% of the frame work was ~1,000 completed refutations a
/// position -- "this hypothetical change would NOT make the frame hold" -- which a stored refutation cannot shorten):
/// an added piece of ours must be able to reach its square in one move. Conditions further away are the reachability
/// model's to estimate, not the search's to enumerate.
pub static ONE_MOVE_WAITS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);
#[pyfunction]
pub fn set_one_move_waits(flag: bool) -> PyResult<()> { ONE_MOVE_WAITS.store(flag, Ordering::Relaxed); Ok(()) }
/// The condition and breaker searches use the 2-ply pre-check only (false) or the pre-check then the frame's depth
/// (true). Measured before choosing a default (2026-10-09).
pub static DEEP_CONDITIONS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

#[pyfunction]
pub fn set_deep_conditions(flag: bool) -> PyResult<()> { DEEP_CONDITIONS.store(flag, Ordering::Relaxed); Ok(()) }

/// THE PROOF QUESTION of a frame: does the seed certify its goal on `p` (its owner to move)? The 2-ply pre-check, then
/// the frame's depth (unless DEEP_CONDITIONS is off), each a null-window search at the goal run with the stored proof
/// `hints` tried first -> (holds, the proof or refutation it recorded).
pub fn prove(p: &Chess, uci: &str, focus: &[Square], goal: i64, hints: Option<std::sync::Arc<Hints>>) -> (bool, Hints) {
    let Ok(u) = uci.parse::<shakmaty::uci::UciMove>() else { return (false, Hints::new()) };
    let Ok(m) = u.to_move(p) else { return (false, Hints::new()) };
    let deep = DEEP_CONDITIONS.load(Ordering::Relaxed);
    let mut last = Hints::new();
    for (d, b) in [(2u32, 10_000u64), (PURSUIT_PLIES, 50_000u64)] {
        if d > 2 && !deep { break; }
        let (r, rec) = with_hints(hints.clone(), || prove_pos(p, &m, focus, d, b, goal));
        last = rec;
        if r == Some(true) { return (true, last); }
    }
    (false, last)
}

/// A frame's stored proofs: one for the seed on the board (key None) and one for each candidate change (key Some).
pub type Cache = HashMap<Option<Atom>, std::sync::Arc<Hints>>;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Kind { AddOurs, AddTheirs, RemoveOurs, RemoveTheirs }

/// One condition over one square, relative to the frame's owner: our / their `role` stands on `sq` (Add), or no
/// longer stands there (Remove). A dormant frame WAITS on atoms (add ours, remove theirs, remove ours); an active
/// frame is INVALIDATED by atoms (Kaveh 2026-10-09: "each frame also needs to know what is invalidated, so that it can
/// re-fire if something happens") -- its BREAKERS: remove ours, add theirs, remove theirs (a flight square freed).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Atom { pub kind: Kind, pub role: Role, pub sq: Square }

impl Atom {
    pub fn holds(&self, pos: &Chess, owner: Color) -> bool {
        let p = pos.board().piece_at(self.sq);
        match self.kind {
            Kind::AddOurs => p == Some(Piece { color: owner, role: self.role }),
            Kind::AddTheirs => p == Some(Piece { color: !owner, role: self.role }),
            Kind::RemoveOurs => p != Some(Piece { color: owner, role: self.role }),
            Kind::RemoveTheirs => p != Some(Piece { color: !owner, role: self.role }),
        }
    }
    pub fn row(&self) -> (String, String, String) {
        let k = match self.kind { Kind::AddOurs => "add ours", Kind::AddTheirs => "add theirs", Kind::RemoveOurs => "remove ours", Kind::RemoveTheirs => "remove theirs" };
        (k.to_string(), self.role.upper_char().to_string(), self.sq.to_string())
    }
}

#[derive(Clone)]
pub struct Seed { pub mv: Move, pub uci: String, pub name: &'static str, pub focus: Vec<Square>, pub goal: i64 }

impl Seed {
    fn key(&self) -> String { format!("{}|{}", self.uci, self.focus.iter().map(|s| s.to_string()).collect::<Vec<_>>().join(",")) }
}

/// The frames the side to move could open: every check (focus: their king, goal: mate), every move making new threats
/// (fork: the second-best gain; single threat: its gain), every move newly cornering or trapping a piece. Rule-layer
/// matching only, no certificate.
pub fn seeds(pos: &Chess) -> Vec<Seed> {
    let us = pos.turn();
    let before = rules::threats_of(pos, us);
    let passed = pass(pos);
    let corner_before: std::collections::HashSet<u8> = passed.as_ref().map(|p| rules::cornered(p, us).into_iter().map(|(t, _)| t).collect()).unwrap_or_default();
    let trapped_before: std::collections::HashSet<u8> = passed.as_ref().map(|p| rules::trapped_unattacked(p, us).into_iter().map(|(t, _)| t).collect()).unwrap_or_default();
    let mut out: Vec<Seed> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for m in pos.legal_moves() {
        let q = after(pos, &m);
        if q.is_game_over() { continue; }
        let uci = m.to_uci(CastlingMode::Standard).to_string();
        let king = q.board().king_of(!us);
        let mut push = |name: &'static str, focus: Vec<Square>, goal: i64| {
            let s = Seed { mv: m.clone(), uci: uci.clone(), name, focus, goal };
            if seen.insert(s.key()) { out.push(s); }
        };
        if q.is_check() { if let Some(k) = king { push("check", vec![k], MATE_GOAL); } }
        let now = rules::threats_of(&q, us);
        let mut new: Vec<(u8, i64)> = now.iter().filter(|(t, g)| before.get(t).map_or(true, |g0| **g > *g0)).map(|(t, g)| (*t, *g)).collect();
        new.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        let sqs: Vec<Square> = new.iter().map(|(t, _)| Square::new(*t as u32)).collect();
        if new.len() >= 2 { push("fork", sqs.clone(), new[1].1); }
        else if new.len() == 1 && new[0].1 > 0 { push("single threat", sqs.clone(), new[0].1); }
        for (t, g) in rules::cornered(&q, us) {
            if Some(Square::new(t as u32)) != king && !corner_before.contains(&t) && g > 0 {
                push("cornered piece", vec![Square::new(t as u32)], g);
            }
        }
        if !q.is_check() {
            for (t, v) in rules::trapped_unattacked(&q, us) {
                if !trapped_before.contains(&t) && !corner_before.contains(&t) && v > 0 {
                    push("trapped piece", vec![Square::new(t as u32)], v);
                }
            }
        }
    }
    out
}

/// The certified value of the seed on `p` (its owner to move), or -1000: the 2-ply pre-check, then the frame's depth.
pub fn certify(p: &Chess, uci: &str, focus: &[Square], goal: i64) -> i64 {
    let Ok(u) = uci.parse::<shakmaty::uci::UciMove>() else { return -1000 };
    let Ok(m) = u.to_move(p) else { return -1000 };
    let a = certified_value(p, &m, focus, 2, 10_000).unwrap_or(-1000);
    if a >= goal { return a; }
    let b = certified_value(p, &m, focus, PURSUIT_PLIES, 50_000).unwrap_or(-1000);
    a.max(b)
}

fn perturb(pos: &Chess, a: &Atom, owner: Color) -> Option<Chess> {
    let mut s = pos.clone().into_setup(EnPassantMode::Legal);
    let here = s.board.piece_at(a.sq);
    match a.kind {
        Kind::AddOurs | Kind::AddTheirs => {
            if here.is_some() { return None; }
            let c = if a.kind == Kind::AddOurs { owner } else { !owner };
            s.board.set_piece_at(a.sq, Piece { color: c, role: a.role });
        }
        Kind::RemoveTheirs => { if here != Some(Piece { color: !owner, role: a.role }) { return None; } s.board.discard_piece_at(a.sq); }
        Kind::RemoveOurs => { if here != Some(Piece { color: owner, role: a.role }) { return None; } s.board.discard_piece_at(a.sq); }
    }
    Chess::from_setup(s, CastlingMode::Standard)
        .or_else(|e| e.ignore_invalid_castling_rights())
        .or_else(|e| e.ignore_too_much_material())
        .ok()
}

/// The frame's zone after the seed move: the focus, its reach, our pieces bearing on it and the lines between.
fn zone(pos: &Chess, s: &Seed) -> Bitboard {
    let q = after(pos, &s.mv);
    let mut z = frame_zone(&q, pos.turn(), &s.focus);
    for f in &s.focus { z.add(*f); }
    z
}

/// CAN A PIECE OF `c`'s `role` ARRIVE ON `sq`? (2026-10-09, measured on 40 census positions: 72 breakers per active
/// frame, mostly "their queen appears on some square", and conditions asking for a piece type we do not own -- dead
/// weight that also swelled every footprint until any move re-evaluated the frame.) A waiting condition (`one_move`
/// false) needs a piece of that role on the board -- a pawn on the square's file or an adjacent one, behind it; a
/// breaker (`one_move` true) is a change THEIR NEXT MOVE can make: a piece of that role that moves to `sq` (a pawn
/// pushing or capturing there).
fn can_arrive(pos: &Chess, c: Color, role: Role, sq: Square, one_move: bool) -> bool {
    let b = pos.board();
    let mine = b.by_color(c) & b.by_role(role);
    if mine.is_empty() { return false; }
    let fwd: i32 = if c == Color::White { 1 } else { -1 };
    if role == Role::Pawn {
        let (f, r) = (sq.file() as i32, sq.rank() as i32);
        return mine.into_iter().any(|p| {
            let (pf, pr) = (p.file() as i32, p.rank() as i32);
            let ahead = (r - pr) * fwd;
            if one_move {
                (pf == f && (ahead == 1 || (ahead == 2 && pr == if c == Color::White { 1 } else { 6 })))
                    || ((pf - f).abs() == 1 && ahead == 1)
            } else {
                (pf - f).abs() <= 1 && ahead >= 1
            }
        });
    }
    if !one_move { return true; }
    let occ = b.occupied();
    mine.into_iter().any(|p| shakmaty::attacks::attacks(p, Piece { color: c, role }, occ).contains(sq))
}

/// The candidate atoms of a frame: pieces added on empty squares bearing on (or standing in) the zone -- ours when
/// looking for what is missing, theirs when looking for what breaks it -- their non-king pieces removed in or bearing
/// on it, our non-king pieces removed from it.
fn candidates(pos: &Chess, s: &Seed, z: Bitboard, breakers: bool) -> Vec<Atom> {
    let us = pos.turn();
    let b = pos.board();
    let occ = after(pos, &s.mv).board().occupied();
    let mut out = Vec::new();
    for i in 0..64u32 {
        let sq = Square::new(i);
        if sq == s.mv.to() || Some(sq) == s.mv.from() || s.focus.contains(&sq) { continue; }
        match b.piece_at(sq) {
            None => {
                let (c, kind) = if breakers { (!us, Kind::AddTheirs) } else { (us, Kind::AddOurs) };
                let near = breakers || ONE_MOVE_WAITS.load(Ordering::Relaxed);
                for role in [Role::Queen, Role::Rook, Role::Bishop, Role::Knight, Role::Pawn] {
                    if !can_arrive(pos, c, role, sq, near) { continue; }
                    let reach = shakmaty::attacks::attacks(sq, Piece { color: c, role }, occ);
                    if z.contains(sq) || (reach & z).any() { out.push(Atom { kind, role, sq }); }
                }
            }
            Some(p) if p.role == Role::King => {}
            Some(p) if p.color != us => {
                let reach = shakmaty::attacks::attacks(sq, p, occ);
                if z.contains(sq) || (reach & z).any() { out.push(Atom { kind: Kind::RemoveTheirs, role: p.role, sq }); }
            }
            Some(p) => { if z.contains(sq) { out.push(Atom { kind: Kind::RemoveOurs, role: p.role, sq }); } }
        }
    }
    out
}

type Found = (Vec<(Atom, i64)>, Vec<(Option<Atom>, Hints)>);

/// The single changes under which the seed's certificate reaches its goal: [(atom, the gain it earns)], best first,
/// with every candidate's recorded proof or refutation (for the next move's re-check).
/// THE CONTROL (bug 2026-10-09, the castling position: "add our pawn on e7" made four unrelated frames hold -- the pawn
/// promotes on d8 by itself): the seed must earn its goal BEYOND what the change alone wins by captures and promotions
/// (the resolved captures with us to move, before the seed): the proof asks for goal + that, and a mate goal is refused
/// when the change alone wins a piece's worth.
pub fn conditions(pos: &Chess, s: &Seed, cache: &Cache) -> Found {
    let us = pos.turn();
    let z = zone(pos, s);
    let rows: Vec<(Atom, bool, i64, Hints)> = candidates(pos, s, z, false).par_iter().filter_map(|a| {
        let p2 = perturb(pos, a, us)?;
        BUDGET.store(50_000, Ordering::Relaxed);
        let alone = settle_d(&p2, -2000, 2000, true, SETTLE_DEPTH).max(0);
        BUDGET.store(u64::MAX, Ordering::Relaxed);
        let mate = s.goal >= MATE_GOAL;
        if mate && alone >= 9 { return None; }
        let goal = if mate { s.goal } else { s.goal + alone };
        let (ok, rec) = prove(&p2, &s.uci, &s.focus, goal, cache.get(&Some(*a)).cloned());
        Some((*a, ok, if mate { 1000 } else { goal - alone }, rec))
    }).collect();
    let mut found: Vec<(Atom, i64)> = rows.iter().filter(|r| r.1).map(|r| (r.0, r.2)).collect();
    found.sort_by(|a, b| b.1.cmp(&a.1).then((a.0.sq as u8).cmp(&(b.0.sq as u8))));
    (found, rows.into_iter().map(|r| (Some(r.0), r.3)).collect())
}

/// What invalidates an active frame: the single changes (their next move can make) under which its certificate no
/// longer reaches the goal -- [(atom, goal - 1: "below the goal")], with every candidate's recorded proof.
pub fn breakers(pos: &Chess, s: &Seed, cache: &Cache) -> Found {
    let us = pos.turn();
    let z = zone(pos, s);
    let rows: Vec<(Atom, bool, Hints)> = candidates(pos, s, z, true).par_iter().filter_map(|a| {
        let p2 = perturb(pos, a, us)?;
        let (ok, rec) = prove(&p2, &s.uci, &s.focus, s.goal, cache.get(&Some(*a)).cloned());
        Some((*a, ok, rec))
    }).collect();
    let mut found: Vec<(Atom, i64)> = rows.iter().filter(|r| !r.1).map(|r| (r.0, s.goal - 1)).collect();
    found.sort_by_key(|a| a.0.sq as u8);
    (found, rows.into_iter().map(|r| (Some(r.0), r.2)).collect())
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Status { Active, Dormant, Far }

impl Status {
    fn name(&self) -> &'static str { match self { Status::Active => "active", Status::Dormant => "dormant", Status::Far => "far" } }
}

#[derive(Clone)]
pub struct Frame {
    pub owner: Color, pub seed: Seed, pub value: i64, pub status: Status,
    pub alts: Vec<(Atom, i64)>,      // dormant: what it waits on (any one of them makes it hold)
    pub breaks: Vec<(Atom, i64)>,    // active: what invalidates it (any one of them makes it fail)
    pub footprint: Bitboard, pub born: u32,
    pub cache: Cache,                // the stored proofs, re-checked on the next move
}

/// Evaluate one seed on `p` (its owner to move): active with its breakers if it certifies, else dormant with its
/// missing conditions (or far). With `cache` (the frame's stored proofs from the last move) every proof question is
/// re-checked with its stored proof tried first -- the same answers as a fresh search, at the cost of checking. The
/// value is a lower bound: the goal when active, -1000 otherwise. The footprint holds the zone, the seed's squares and
/// every atom's square.
pub fn evaluate(p: &Chess, s: &Seed, born: u32, cache: &Cache) -> Frame {
    let (holds, rec) = prove(p, &s.uci, &s.focus, s.goal, cache.get(&None).cloned());
    let mut fp = zone(p, s);
    for f in &s.focus { fp.add(*f); }
    if let Some(f) = s.mv.from() { fp.add(f); }
    fp.add(s.mv.to());
    let (status, alts, breaks, recs) = if holds {
        let (b, r) = breakers(p, s, cache);
        (Status::Active, vec![], b, r)
    } else {
        let (a, r) = conditions(p, s, cache);
        (if a.is_empty() { Status::Far } else { Status::Dormant }, a, vec![], r)
    };
    for (a, _) in alts.iter().chain(breaks.iter()) { fp.add(a.sq); }
    let mut new_cache: Cache = recs.into_iter().map(|(k, h)| (k, std::sync::Arc::new(h))).collect();
    new_cache.insert(None, std::sync::Arc::new(rec));
    Frame { owner: p.turn(), seed: s.clone(), value: if holds { s.goal } else { -1000 }, status, alts, breaks,
            footprint: fp, born, cache: new_cache }
}

/// Both sides' frames: the side to move's on the position, the other side's on the null move (none when in check).
pub fn frames_both(pos: &Chess, born: u32) -> Vec<Frame> {
    let empty = Cache::new();
    let mut out: Vec<Frame> = seeds(pos).par_iter().map(|s| evaluate(pos, s, born, &empty)).collect();
    if let Some(p) = pass(pos) {
        out.extend(seeds(&p).par_iter().map(|s| evaluate(&p, s, born, &empty)).collect::<Vec<_>>());
    }
    out
}

type FrameRow = (String, String, String, Vec<String>, i64, i64, String, Vec<(String, String, String, i64)>, Vec<(String, String, String, i64)>, u32);

fn row(f: &Frame) -> FrameRow {
    (if f.owner == Color::White { "w" } else { "b" }.to_string(), f.seed.uci.clone(), f.seed.name.to_string(),
     f.seed.focus.iter().map(|s| s.to_string()).collect(), f.seed.goal, f.value, f.status.name().to_string(),
     f.alts.iter().map(|(a, v)| { let (k, r, s) = a.row(); (k, r, s, *v) }).collect(),
     f.breaks.iter().map(|(a, v)| { let (k, r, s) = a.row(); (k, r, s, *v) }).collect(), f.born)
}

/// waiting_frames(fen) -> [(owner, seed uci, name, focus, goal, certified value, status, waits on [(kind, role,
/// square, value)], broken by [(kind, role, square, value left)], born)] for both sides.
#[pyfunction]
pub fn waiting_frames(fen: String) -> PyResult<Vec<FrameRow>> {
    let Some(pos) = parse(&fen) else { return Ok(vec![]) };
    Ok(frames_both(&pos, 0).iter().map(row).collect())
}

type ArmRow = (String, String, String, Vec<String>, i64, Vec<(String, String, String)>);

/// The reverse lookup on `pos` (side to move = us): for each legal move M, the dormant frames of ours that M's
/// changes satisfy AND that then certify on the null move (M arms a threat), and the dormant frames of theirs that M
/// satisfies and that then certify with them to move (M hands them a tactic). Checking moves are not armed here (no
/// null move after a check). And DEFUSES: the active frames of theirs (threats on the null move) that M breaks -- one
/// of its breakers holds after M and the certificate, with them to move, falls below the goal.
pub fn arming_core(pos: &Chess, frames: &[Frame]) -> (Vec<ArmRow>, Vec<ArmRow>, Vec<ArmRow>) {
    let us = pos.turn();
    let moves: Vec<Move> = pos.legal_moves().into_iter().collect();
    let res: Vec<(Vec<ArmRow>, Vec<ArmRow>, Vec<ArmRow>)> = moves.par_iter().map(|mm| {
        let q = after(pos, mm);
        let muci = mm.to_uci(CastlingMode::Standard).to_string();
        let mut arms = Vec::new();
        let mut hands = Vec::new();
        let mut defuses = Vec::new();
        for f in frames.iter().filter(|f| f.status == Status::Active && f.owner != us) {
            let hit: Vec<(String, String, String)> = f.breaks.iter()
                .filter(|(a, _)| a.holds(&q, f.owner) && !a.holds(pos, f.owner)).map(|(a, _)| a.row()).collect();
            if hit.is_empty() { continue; }
            let v = certify(&q, &f.seed.uci, &f.seed.focus, f.seed.goal);
            if v < f.seed.goal { defuses.push((muci.clone(), f.seed.uci.clone(), f.seed.name.to_string(), f.seed.focus.iter().map(|s| s.to_string()).collect(), v, hit)); }
        }
        for f in frames.iter().filter(|f| f.status == Status::Dormant) {
            let met: Vec<(String, String, String)> = f.alts.iter()
                .filter(|(a, _)| a.holds(&q, f.owner) && !a.holds(pos, f.owner)).map(|(a, _)| a.row()).collect();
            if met.is_empty() { continue; }
            if f.owner == us {
                if q.is_check() { continue; }
                let Some(p) = pass(&q) else { continue };
                let v = certify(&p, &f.seed.uci, &f.seed.focus, f.seed.goal);
                if v >= f.seed.goal { arms.push((muci.clone(), f.seed.uci.clone(), f.seed.name.to_string(), f.seed.focus.iter().map(|s| s.to_string()).collect(), v, met)); }
            } else {
                let v = certify(&q, &f.seed.uci, &f.seed.focus, f.seed.goal);
                if v >= f.seed.goal { hands.push((muci.clone(), f.seed.uci.clone(), f.seed.name.to_string(), f.seed.focus.iter().map(|s| s.to_string()).collect(), v, met)); }
            }
        }
        (arms, hands, defuses)
    }).collect();
    let mut a = Vec::new(); let mut h = Vec::new(); let mut d = Vec::new();
    for (x, y, z) in res { a.extend(x); h.extend(y); d.extend(z); }
    (a, h, d)
}

/// arming_moves(fen) -> (arms, hands, defuses): [(our move, seed uci, name, focus, certified value, [atoms met])].
#[pyfunction]
pub fn arming_moves(fen: String) -> PyResult<(Vec<ArmRow>, Vec<ArmRow>, Vec<ArmRow>)> {
    let Some(pos) = parse(&fen) else { return Ok((vec![], vec![], vec![])) };
    let frames = frames_both(&pos, 0);
    Ok(arming_core(&pos, &frames))
}

/// THE LIVE POOL: frames kept across the moves of a game, both sides. `observe(fen)`: the seeds the move CREATED are
/// evaluated from scratch; the seeds it DESTROYED are dropped; every other frame is RE-CHECKED from its stored proofs
/// (`proofs`, default; off = every frame from scratch, for the audit). (Reuse by footprint without re-checking was
/// audited at 1.0% stale statuses for 14% saved and removed.) Returns the events: (owner, seed uci, name, focus, goal, value, status
/// before ("new" if unseen), status after ("dead" if dropped), born ply).
#[pyclass(unsendable)]
pub struct FramePool {
    frames: Vec<Frame>,
    last: Option<Chess>,
    ply: u32,
    proofs: bool,
    #[pyo3(get)] rechecked: u64,
    #[pyo3(get)] created: u64,
    #[pyo3(get)] nodes: u64,
}

type Event = (String, String, String, Vec<String>, i64, i64, String, String, u32);

fn event(f: &Frame, before: &str, after_: &str) -> Event {
    (if f.owner == Color::White { "w" } else { "b" }.to_string(), f.seed.uci.clone(), f.seed.name.to_string(),
     f.seed.focus.iter().map(|s| s.to_string()).collect(), f.seed.goal, f.value, before.to_string(), after_.to_string(), f.born)
}

#[pymethods]
impl FramePool {
    #[new]
    #[pyo3(signature = (proofs=true))]
    fn new(proofs: bool) -> Self { FramePool { frames: vec![], last: None, ply: 0, proofs, rechecked: 0, created: 0, nodes: 0 } }

    fn observe(&mut self, fen: String) -> PyResult<Vec<Event>> {
        let Some(pos) = parse(&fen) else { return Ok(vec![]) };
        let n0 = NODES.load(Ordering::Relaxed);
        let ply = self.ply;
        let mut events = Vec::new();
        let mut next: Vec<Frame> = Vec::new();
        let sides: Vec<Chess> = std::iter::once(pos.clone()).chain(pass(&pos)).collect();
        let mut alive_owners = Vec::new();
        for p in &sides {
            let owner = p.turn();
            alive_owners.push(owner);
            let ss = seeds(p);
            let old: HashMap<String, &Frame> = self.frames.iter().filter(|f| f.owner == owner).map(|f| (f.seed.key(), f)).collect();
            let proofs = self.proofs;
            let empty = Cache::new();
            let done: Vec<(Frame, Option<Status>, bool)> = ss.par_iter().map(|s| {
                match old.get(&s.key()) {
                    Some(f) => (evaluate(p, s, f.born, if proofs { &f.cache } else { &empty }), Some(f.status), true),
                    None => (evaluate(p, s, ply, &empty), None, false),
                }
            }).collect();
            let keys: std::collections::HashSet<String> = ss.iter().map(|s| s.key()).collect();
            for (f, before, reused) in done {
                if reused { self.rechecked += 1 } else { self.created += 1 }
                match before {
                    None => events.push(event(&f, "new", f.status.name())),
                    Some(b) if b != f.status => events.push(event(&f, b.name(), f.status.name())),
                    _ => {}
                }
                next.push(f);
            }
            for f in self.frames.iter().filter(|f| f.owner == owner && !keys.contains(&f.seed.key())) {
                events.push(event(f, f.status.name(), "dead"));
            }
        }
        // a side that could not be evaluated (the side to move is in check: no null move for the other) keeps its frames
        for f in self.frames.iter().filter(|f| !alive_owners.contains(&f.owner)) { next.push(f.clone()); }
        self.frames = next;
        self.nodes += NODES.load(Ordering::Relaxed).wrapping_sub(n0);
        self.last = Some(pos);
        self.ply += 1;
        Ok(events)
    }

    fn frames(&self) -> Vec<FrameRow> { self.frames.iter().map(row).collect() }

    /// The reverse lookup on the last observed position, from the pool's frames.
    fn arming(&self) -> (Vec<ArmRow>, Vec<ArmRow>, Vec<ArmRow>) {
        match &self.last { Some(p) => arming_core(p, &self.frames), None => (vec![], vec![], vec![]) }
    }
}
