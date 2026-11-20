//! PLAN 79.07 THE PROPOSER IN RUST, ASKING BY STAKE WITHIN A BUDGET (SYSTEM_SPEC 7 "the proposer, what a goal is, the budget of questions", 4 reasoning by questions, 8 step 3, 11 Rust in the hot paths; piece: the wake). The Rust form of
//! catspace/incubator/analyst/discovery/subgoal_goals.py (`goals`, `build_goal`, `goal_unit_goal`, `fix_set_of`, `line_story`), future_goals.py (`goals_of_list`) and the questions of the owner's own reading of a position (the lines onto the goal unit, who
//! bears and who holds, remove a holder or add a bearer, a move that makes two threats at once). The Python stays as the mirror. Game-free: a unit, a kind, a place, a move is a number or a name the game gave; nothing of any game is named here.
//! No oracle, no external model, no worth assumed (the only worths are the caller's measured `Record`). A goal ORDERS what is read and says what is read in full; it never adds to the judge's value (SYSTEM_SPEC 7, rule 7).
//!
//! INTERFACE (written before the bodies; who calls: the wake's loop, 79.08, and the unit tests; nothing plays through it yet).
//!   `propose(st, g, s, rec, threats, params) -> Proposal`    takes the position, the measured record, the threat list's parameters and `GoalParams` (the budget of questions, the sides); returns the concrete list of both sides (the rows against both sides' units, read in
//!       the order of their stake within the budget), the goals of each side (`Goal`: facts, causes, fixes, the fix set), the goals of two threats, the moves that make two threats at once, what was asked and what was not, and the time of each part.
//!   `goals(st, g, s, list, side, goal) -> Vec<Goal>`          the goals of one side from a concrete list: one per threat of that side that does not work yet and the goal unit's; ordered by the fewest unmet facts, then the larger stake.
//!   `two_threat_goals(list, side, mover, rec, kinds) -> TwoThreats`   two threats of one side that no one reply meets (a pair of confirmed rows on different units whose kept replies have nothing in common).
//!   `shared_fix_moves(goals) -> Vec<SharedMove>`             one legal move that is a fix under two goals (two targets): the source of the goals of two threats, no separate pattern.
//!   `answers_against(st, g, s, attacker, target, goal) -> Vec<Answer>`  THE SAME MACHINERY FROM THE OTHER SIDE: the legal moves of the target's owner that make one of the attacker's facts unmet, each named by the fact it un-meets and nothing else (whether it works is the line read's verdict).
//!   `named_moves(g, s, goals, doubles) -> HashMap<u64, Vec<String>>`  the moves by position that the fixes name, for both sides, in the form `Search::named` takes (the other side's at the position after the game's pass): read first and unreduced, never added to a value.
//! A FIX is built from LEGAL MOVES and the graph, never from an edited position: `Fix.moves` are the names of legal moves of the side that wants the goal (after the game's pass, for the side not to move).
//! THE BUDGET (SYSTEM_SPEC 4, 7): a question is one unit asked. Units are asked in the order of their stake (the goal unit has no row to read: its goal is built from the graph and the forcing read first); within `budget` questions every unit is asked; past it a unit is asked only while the last
//! question changed a fact (a line was read for the unit before it). A unit not asked is listed in `Proposal::unasked` and has no goal: a gap, never a refutation. One pass over the quiet moves serves every unit (`threats::every_unit_ask`): a line is read once however many units it serves.
use crate::threat_verdict::{Record, VerdictKind};
use crate::threats::{concrete_threats, every_unit_ask, graph_rows, kind_worths, Ask, ConcreteList, ThreatParams, ThreatRow};
use crate::wake::{new_bearer_moves, UnitRef};
use cs_game::placed::{Kind, Place, Placed};
use cs_game::{Game, Role};
use cs_tactics::sheet::base_graph::{PositionPart, Static};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::time::Instant;

/// The default budget of questions: the units asked before progress alone decides (a named parameter, not a tuned one).
pub const BUDGET: usize = 8;
/// The depth of the recursion of subgoals (a stopper's stopper): a named bound, with the budget as the other (the subgoals are at most the budget).
pub const DEPTH: u8 = 2;

/// The state of a fact (subgoal_goals.py): met, unmet, open (no line was read and the graph does not say), unread (the line was cut).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status { Met, Unmet, Open, Unread }
impl Status { pub fn word(self) -> &'static str { match self { Status::Met => "met", Status::Unmet => "unmet", Status::Open => "open", Status::Unread => "unread" } } fn bad(self) -> bool { matches!(self, Status::Unmet | Status::Unread) } }

/// What stops a fact: the type (the Python's words), all the units that stop it, the units that are lost by it, and the basis of the claim (`line`: a line that was read says so; `graph`: a relation of the position).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Cause { pub kind: &'static str, pub stoppers: Vec<UnitRef>, pub victim: Option<UnitRef>, pub victims: Vec<UnitRef>, pub basis: &'static str }

/// The three kinds of fix: take a stopper out of play (taken, or no longer bearing on the place it stops), bring a unit to bear (a legal move after which more units bear on the place), or play the quiet first move the threat needs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FixKind { Remove, Bring, FirstMove }

/// A fix: what it deals with and the LEGAL MOVES (names) of the side that wants the goal that do it. `against`: for a removal, each stopper and the place it must stop bearing on (None: it must be gone). `to`: for a bringing, the place a unit must come to bear on. `steps`: by how many moves of
/// the graph (1: a legal move now brings it; None: no legal move does).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Fix { pub kind: FixKind, pub against: Vec<(UnitRef, Option<Place>)>, pub to: Option<Place>, pub steps: Option<u8>, pub moves: Vec<String>, pub words: &'static str }

/// One fact of a goal (bears, may_take, unguarded, cornered, kept: the same for every target, the goal unit's included; and what the rules add for the goal kind: covered, forced_end) with its status, the cause when unmet, the fixes and the units it is about.
#[derive(Debug, Clone, PartialEq)]
pub struct Fact { pub name: &'static str, pub status: Status, pub cause: Option<Cause>, pub fixes: Vec<Fix>, pub a: Option<UnitRef>, pub place: Option<Place> }

/// An element of the fix set: each stopper or brought unit with the facts still unmet if only it is dealt with, ordered by the fewest left. `removable`: false for a unit of the goal kind (the rules do not let it be taken out).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FixElem { pub unit: Option<UnitRef>, pub kind: FixKind, pub removable: bool, pub resolves: Vec<&'static str>, pub left: usize, pub left_stoppers: usize, pub moves: Vec<String>, sort: String }

/// A step where nothing of ours bears: the unit brought to bear or the line freed, and by how many moves of the graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Step { pub step: &'static str, pub moves: Option<u8> }

/// A place the goal unit may move to and whether a unit of the other side covers it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Escape { pub place: Place, pub covered: bool, pub by: Vec<UnitRef>, pub occupant: Option<UnitRef> }

/// A goal of one side: a threat of its that does not work yet, as the facts that would make it work.
#[derive(Debug, Clone)]
pub struct Goal {
    pub side: Role, pub target: UnitRef, pub source: String, pub facts: Vec<Fact>, pub unmet: Vec<&'static str>, pub fixes: Vec<Fix>, pub key: String, pub stake: Option<f64>, pub line: Vec<String>, pub two_moves: bool, pub lead: Option<String>,
    pub verdict: Option<VerdictKind>, pub net: Option<i64>, pub stoppers: Vec<UnitRef>, pub fix_set: Vec<FixElem>, pub steps: Vec<Step>, pub n_unmet: usize, pub every: bool, pub own_moves: Option<u8>,
    /// the goal unit's goal (its facts are the lines onto it, the places it may move to, the forced end)
    pub goal_unit: bool, pub escapes: Vec<Escape>,
    /// THE RECURSION (79.07 amendment): a stopper of the other side of an unmet fact is itself a target with a goal of its own (the subgoal). `parent`: the target of the goal it would unblock, `via`: the fact of that goal it stops, `depth`: 0 for a goal of a unit asked.
    pub parent: Option<UnitRef>, pub via: Option<&'static str>, pub depth: u8,
}

// ------------------------------------------------------------------------------------------------------------------------------ the position as the proposer reads it

/// The game's words, read once.
struct Words { roles: Vec<String>, kinds: Vec<String>, places: Vec<String> }

/// One position's graph as the Python's `PositionFacts`: the relations bear, may_move and shut, and the units by the order of their id text (`unit:role:kind:place`, as the Python sorts them).
struct View { part: PositionPart, bear: HashSet<(usize, Place)>, may: HashSet<(usize, Place)>, shut: HashMap<(usize, Place), Vec<usize>>, ids: Vec<String>, mover: Option<Role> }



fn view<G: Game + Placed>(st: &Static, g: &G, s: &G::State, w: &Words) -> View {
    let part = st.position(g, s);
    let bear = part.bears.iter().map(|r| (r.unit, r.place)).collect();
    let may = part.moves.iter().map(|r| (r.unit, r.place)).collect();
    let mut shut = HashMap::new();
    for r in &part.shuts { shut.insert((r.unit, r.place), r.between.clone()); }
    let ids = part.units.iter().map(|u| format!("unit:{}:{}:{}", w.roles[u.owner], w.kinds[u.kind as usize], w.places[u.at as usize])).collect();
    View { mover: part.mover, part, bear, may, shut, ids }
}

impl View {
    fn n(&self) -> usize { self.part.units.len() }
    fn u(&self, i: usize) -> UnitRef { let u = &self.part.units[i]; UnitRef { owner: u.owner, kind: u.kind, at: u.at } }
    fn at(&self, p: Place) -> Option<usize> { match self.part.occupant[p as usize] { 0 => None, k => Some(k as usize - 1) } }
    fn owner(&self, i: usize) -> Role { self.part.units[i].owner }
    fn bears(&self, i: usize, p: Place) -> bool { self.bear.contains(&(i, p)) }
    /// The units of `owner` that bear on `p`, in the order of their ids.
    fn bearers(&self, p: Place, owner: Role) -> Vec<usize> { let mut v: Vec<usize> = (0..self.n()).filter(|&i| self.owner(i) == owner && self.bears(i, p)).collect(); v.sort_by(|a, b| self.ids[*a].cmp(&self.ids[*b])); v }
    /// The units named, each once, in the order first met (the Python's `_units_of`).
    fn units_of(&self, v: &[Option<usize>]) -> Vec<UnitRef> { let mut seen = HashSet::new(); v.iter().flatten().filter(|i| seen.insert(**i)).map(|i| self.u(*i)).collect() }
    fn of_idx(&self, v: &[usize]) -> Vec<UnitRef> { self.units_of(&v.iter().map(|i| Some(*i)).collect::<Vec<_>>()) }
    /// The lines of the side `taker` onto `p` that some units shut, the fewest between first (the Python's `sorted(((len(b), u, b) ...))`).
    fn shut_onto(&self, p: Place, taker: Role) -> Vec<(usize, Vec<usize>)> {
        let mut v: Vec<(usize, Vec<usize>)> = self.shut.iter().filter(|((u, q), b)| *q == p && self.owner(*u) == taker && !b.is_empty()).map(|((u, _), b)| (*u, b.clone())).collect();
        v.sort_by(|a, b| (a.1.len(), &self.ids[a.0], a.1.iter().map(|x| &self.ids[*x]).collect::<Vec<_>>()).cmp(&(b.1.len(), &self.ids[b.0], b.1.iter().map(|x| &self.ids[*x]).collect::<Vec<_>>())));
        v
    }
    fn goal_units(&self, owner: Role, goal: Option<Kind>) -> Vec<usize> { let mut v: Vec<usize> = (0..self.n()).filter(|&i| self.owner(i) == owner && Some(self.part.units[i].kind) == goal).collect(); v.sort_by(|a, b| self.ids[*a].cmp(&self.ids[*b])); v }
}

/// The position as `side` sees it to move: the position itself, or the one after the game's pass (None where the game gives none).
fn view_state<G: Game + Placed>(g: &G, s: &G::State, side: Role) -> Option<G::State> { if g.mover(s) == Some(side) { Some(s.clone()) } else { g.pass(s) } }

// ------------------------------------------------------------------------------------------------------------------------------ a line replayed (subgoal_goals.Facts.replay)

/// One ply of a line: who moved (an index into the units of the position the line starts at, carried through the line), from where to where, what was taken and whether the game ended.
struct Ply { side: Role, uid: Option<usize>, took: Option<Took>, ended: bool }
struct Took { uid: Option<usize>, at: Place }

/// The plies of the line `names` from `s`, read from the difference of two boards (no move is parsed): a unit keeps the id it had at the start. An empty list when a name is not a legal move.
fn replay<G: Game + Placed>(g: &G, s: &G::State, names: &[String]) -> Vec<Ply> {
    let mut cur = s.clone();
    let mut cells = g.cells(&cur);
    cells.sort();
    let mut who: HashMap<Place, usize> = cells.iter().enumerate().map(|(i, c)| (c.2, i)).collect();
    let mut before: HashSet<(Role, Kind, Place)> = cells.iter().copied().collect();
    let mut out = vec![];
    for nm in names {
        let Some(mover) = g.mover(&cur) else { return vec![] };
        let Some(m) = g.find_move(&cur, nm) else { return vec![] };
        let nxt = g.apply(&cur, &m);
        let after: HashSet<(Role, Kind, Place)> = g.cells(&nxt).into_iter().collect();
        let mut gone: Vec<_> = before.difference(&after).copied().collect();
        let mut came: Vec<_> = after.difference(&before).copied().collect();
        gone.sort(); came.sort();
        let m_out: Vec<_> = gone.iter().filter(|x| x.0 == mover).copied().collect();
        let mut m_in: Vec<_> = came.iter().filter(|x| x.0 == mover).copied().collect();
        let taken: Vec<_> = gone.iter().filter(|x| x.0 != mover).copied().collect();
        let mut uid = None;
        let mut best: Option<((bool, Place, Place), Place)> = None;
        for a in &m_out { for b in &m_in { let sc = (a.1 != b.1, a.2, b.2); if best.as_ref().map_or(true, |x| sc < x.0) { best = Some((sc, a.2)); } } }
        if let Some((_, from)) = best { uid = who.get(&from).copied(); }
        let took = taken.first().map(|x| Took { uid: who.get(&x.2).copied(), at: x.2 });
        let mut moved: Vec<(Place, Place)> = vec![];
        for a in &m_out {
            let mut cands: Vec<_> = m_in.iter().filter(|b| b.1 == a.1).copied().collect();
            if cands.is_empty() { cands = m_in.clone(); }
            if let Some(b) = cands.iter().min().copied() { moved.push((a.2, b.2)); m_in.retain(|y| *y != b); }
        }
        let mut nw: HashMap<Place, usize> = who.iter().filter(|(p, _)| !m_out.iter().any(|a| a.2 == **p) && !taken.iter().any(|x| x.2 == **p)).map(|(p, u)| (*p, *u)).collect();
        for (src, dst) in moved { if let Some(u) = who.get(&src) { nw.insert(dst, *u); } }
        who = nw;
        out.push(Ply { side: mover, uid, took, ended: g.outcome(&nxt).is_some() });
        cur = nxt; before = after;
    }
    out
}

/// A loss of the taker's side in a line, by the unit of the other side that took.
struct Loss { kind: &'static str, stopper: usize, victim: Option<usize>, at: Place }
struct Story { taker_unit: Option<usize>, losses: Vec<Loss>, ends: bool }

/// What the other side did in the line that stopped it (subgoal_goals.line_story): the first unit of the taker's that took on the target's place, and every loss of the taker's with its named cause (guard, only_guard, uncovered_line, loose, arrives).
fn line_story(f: &View, taker: Role, t: Place, plies: &[Ply]) -> Story {
    let taker_unit = plies.iter().find(|p| p.took.as_ref().is_some_and(|k| k.at == t) && p.side == taker).and_then(|p| p.uid);
    let mut losses = vec![];
    for (i, p) in plies.iter().enumerate() {
        let Some(k) = &p.took else { continue };
        if p.side == taker { continue; }
        let Some(x) = p.uid else { continue };
        let (v_uid, v_place) = (k.uid, k.at);
        let kind = if v_place == t { if f.bears(x, t) { "guard" } else { "arrives" } }
            else if v_uid.is_some_and(|v| f.part.units[v].at == v_place) {
                if taker_unit.is_some() && f.shut.get(&(x, v_place)).is_some_and(|b| b.len() == 1 && Some(b[0]) == taker_unit) { "uncovered_line" }
                else if f.bears(x, v_place) { if f.bearers(v_place, taker).iter().any(|g| Some(*g) != taker_unit) { "loose" } else { "only_guard" } }
                else { "arrives" }
            } else { "arrives" };
        if v_place != t && plies[i + 1..].iter().any(|q| q.side == taker && q.took.as_ref().is_some_and(|k2| k2.at == v_place)) { continue; }   // taken and taken back at once: a trade, not a loss
        losses.push(Loss { kind, stopper: x, victim: v_uid, at: v_place });
    }
    Story { taker_unit, losses, ends: plies.last().is_some_and(|p| p.ended) }
}

// ------------------------------------------------------------------------------------------------------------------------------ the fixes, as legal moves

/// What the builder of one side's goals reads: the game, the position as the side sees it to move, its graph, and the memo of the moves that deal with a stopper or bring a bearer.
struct Bld<'a, G: Game + Placed> {
    st: &'a Static, g: &'a G, w: &'a Words, goal: Option<Kind>, taker: Role, owner: Role, s_t: G::State, v: View,
    afters: std::cell::OnceCell<Vec<(String, G::State, Vec<(Role, Kind, Place)>)>>, bring_memo: std::cell::RefCell<HashMap<Place, Vec<String>>>,
    /// the legal moves of the side as (from, to, takes): read once, for `takes_legally`
    legal: std::cell::OnceCell<Vec<(Place, Place, bool)>>,
}

impl<'a, G: Game + Placed> Bld<'a, G> {
    /// PLAN 79.08 repair 1: does a LEGAL move of the side take on `to` from `from`? (The graph's may-move relation does not read whether the mover leaves its own goal unit borne on or uncovers a line onto it; the game's moves do.)
    fn takes_legally(&self, from: Place, to: Place) -> bool {
        self.legal.get_or_init(|| self.g.moves(&self.s_t).iter().map(|m| { let e = self.g.move_ends(&self.s_t, m); (e.0, e.1, !self.g.captured(&self.s_t, m).is_empty()) }).collect()).iter().any(|x| x.0 == from && x.1 == to && x.2)
    }
    fn afters(&self) -> &Vec<(String, G::State, Vec<(Role, Kind, Place)>)> {
        self.afters.get_or_init(|| self.g.moves(&self.s_t).iter().map(|m| { let a = self.g.apply(&self.s_t, m); let c = self.g.cells(&a); (self.g.move_name(&self.s_t, m), a, c) }).collect())
    }
    /// The legal moves of the side after which one of `against` is out of play: gone, or no longer bearing on the place it must stop bearing on.
    fn moves_removing(&self, against: &[(UnitRef, Option<Place>)]) -> Vec<String> {
        self.afters().iter().filter(|(_, after, cells)| against.iter().any(|(u, pl)| {
            if !cells.iter().any(|c| *c == (u.owner, u.kind, u.at)) { return true; }
            pl.is_some_and(|p| !self.g.bearing_from(after, u.owner, u.kind, u.at).contains(&p))
        })).map(|(n, _, _)| n.clone()).collect()
    }
    /// The legal moves after which more units of the side bear on `place` (`wake::new_bearer_moves`: the bearer a move brings, a discovered line included).
    fn moves_bringing(&self, place: Place) -> Vec<String> {
        if let Some(v) = self.bring_memo.borrow().get(&place) { return v.clone(); }
        let v: Vec<String> = new_bearer_moves(self.st, self.g, &self.s_t, &[place]).into_iter().map(|(m, _)| m).collect();
        self.bring_memo.borrow_mut().insert(place, v.clone());
        v
    }
    fn remove_fix(&self, units: &[UnitRef], place: Option<Place>, words: &'static str) -> Fix {
        let against: Vec<(UnitRef, Option<Place>)> = units.iter().map(|u| (*u, place)).collect();
        let moves = self.moves_removing(&against);
        Fix { kind: FixKind::Remove, against, to: None, steps: None, moves, words }
    }
    /// A unit of ours brought to bear on `place`: only by a legal move (None where no legal move does).
    fn bring_fix(&self, place: Place) -> Option<Fix> {
        let moves = self.moves_bringing(place);
        if moves.is_empty() { None } else { Some(Fix { kind: FixKind::Bring, against: vec![], to: Some(place), steps: Some(1), moves, words: "a unit of ours comes to bear by a legal move (the far end of where it comes from is a variable)" }) }
    }
    /// The units between are moved off the line, taken or drawn away: the legal moves after which more of ours bear on the place.
    fn free_fix(&self, between: &[usize], place: Place) -> Fix {
        let against: Vec<(UnitRef, Option<Place>)> = between.iter().map(|b| (self.v.u(*b), None)).collect();
        let moves = self.moves_bringing(place);
        Fix { kind: FixKind::Remove, against, to: Some(place), steps: if moves.is_empty() { None } else { Some(1) }, moves, words: "the units between are moved off the line, taken or drawn away (the far end a variable)" }
    }
}

fn cause(kind: &'static str, stoppers: Vec<UnitRef>, victim: Option<UnitRef>, basis: &'static str, victims: Option<Vec<UnitRef>>) -> Cause { Cause { kind, stoppers, victim, victims: victims.unwrap_or_else(|| victim.into_iter().collect()), basis } }
fn fact(name: &'static str, status: Status, a: Option<UnitRef>, cause: Option<Cause>, fixes: Vec<Fix>) -> Fact { Fact { name, status, cause, fixes, a, place: None } }

/// THE FIX AS A SET (subgoal_goals.fix_set_of): every unit that stops a fact and every unit to bring, each with the facts that stay unmet if only it is dealt with, ordered by the fewest left.
fn fix_set_of(facts: &[Fact], goal: Option<Kind>, w: &Words) -> Vec<FixElem> {
    let unmet: Vec<&Fact> = facts.iter().filter(|x| x.status.bad()).collect();
    let sets: Vec<HashSet<UnitRef>> = unmet.iter().map(|x| x.cause.as_ref().map(|c| c.stoppers.iter().copied().collect()).unwrap_or_default()).collect();
    let mut elems: Vec<FixElem> = vec![];
    let mut seen: HashSet<UnitRef> = HashSet::new();
    for x in &unmet {
        let Some(c) = &x.cause else { continue };
        for u in &c.stoppers {
            if !seen.insert(*u) { continue; }
            let left: Vec<usize> = sets.iter().enumerate().filter(|(_, st)| st.is_empty() || st.iter().any(|y| y != u)).map(|(i, _)| i).collect();
            let tot: usize = sets.iter().map(|st| st.iter().filter(|y| *y != u).count()).sum();
            let resolves: Vec<&'static str> = unmet.iter().enumerate().filter(|(i, _)| !left.contains(i)).map(|(_, f)| f.name).collect();
            let moves = x.fixes.iter().filter(|f| f.kind == FixKind::Remove).flat_map(|f| f.moves.clone()).collect();
            elems.push(FixElem { unit: Some(*u), kind: FixKind::Remove, removable: Some(u.kind) != goal, resolves, left: left.len(), left_stoppers: tot, moves, sort: format!("('remove', '{}')", w.places[u.at as usize]) });
        }
    }
    for x in unmet.iter().filter(|x| x.fixes.iter().any(|f| f.kind == FixKind::Bring)) {
        let left = unmet.iter().filter(|y| !std::ptr::eq(**y, *x)).count();
        let moves = x.fixes.iter().filter(|f| f.kind == FixKind::Bring).flat_map(|f| f.moves.clone()).collect();
        elems.push(FixElem { unit: None, kind: FixKind::Bring, removable: true, resolves: vec![x.name], left, left_stoppers: sets.iter().map(|s| s.len()).sum(), moves, sort: "('relocate')".to_string() });
    }
    elems.sort_by(|a, b| (a.left, a.left_stoppers, !a.removable, &a.sort).cmp(&(b.left, b.left_stoppers, !b.removable, &b.sort)));
    elems
}

/// Layer one of "reachable" (subgoal_goals.layer_one): the number of our moves that make the unmet facts true with the other side passing, or None.
fn layer_one(facts: &[Fact], steps: &[Step]) -> Option<u8> {
    let unmet: Vec<&Fact> = facts.iter().filter(|x| x.status.bad()).collect();
    if unmet.is_empty() { return Some(0); }
    if unmet.iter().any(|f| !matches!(f.name, "bears" | "covered") || !f.cause.as_ref().is_some_and(|c| matches!(c.kind, "no_bearer" | "blocked" | "needs_quiet_move"))) { return None; }
    if steps.len() < unmet.len() || steps.iter().any(|s| s.moves.is_none()) { return None; }
    let tot: u32 = steps.iter().map(|s| s.moves.unwrap() as u32).sum();
    if tot <= 2 { Some(tot as u8) } else { None }
}

/// The goal from its facts (subgoal_goals._finish): the unmet names, the key (the chances are tabled by it), the stoppers, the fix set.
fn finish<G: Game + Placed>(b: &Bld<G>, side: Role, target: UnitRef, source: &str, row: Option<&ThreatRow>, facts: Vec<Fact>, fixes: Vec<Fix>, steps: Vec<Step>, line: Vec<String>, two: bool, goal_unit: bool) -> Goal {
    let unmet: Vec<&'static str> = facts.iter().filter(|x| x.status.bad()).map(|x| x.name).collect();
    let mut keys: Vec<String> = facts.iter().filter(|x| x.status.bad()).filter_map(|x| x.cause.as_ref().map(|c| format!("{}:{}{}", x.name, c.kind, if c.stoppers.len() > 1 { format!("*{}", c.stoppers.len().min(3)) } else { String::new() }))).collect();
    keys.sort();
    let mut stoppers: Vec<UnitRef> = vec![];
    for x in facts.iter().filter(|x| x.status.bad()) { if let Some(c) = &x.cause { for u in &c.stoppers { if !stoppers.contains(u) { stoppers.push(*u); } } } }
    let fix_set = fix_set_of(&facts, b.goal, b.w);
    let own_moves = layer_one(&facts, &steps);
    let n_unmet = unmet.len();
    Goal { side, target, source: source.to_string(), unmet, fixes, key: keys.join("|"), stake: row.and_then(|r| r.stake), lead: if two { line.first().cloned() } else { None }, line, two_moves: two, verdict: row.map(|r| r.verdict), net: row.and_then(|r| r.net), stoppers, fix_set, steps, n_unmet,
        every: row.is_some_and(|r| r.every), own_moves, goal_unit, escapes: vec![], parent: None, via: None, depth: 0, facts }
}

/// The source of a goal from its row (subgoal_goals._source_of): None for a threat that already works.
fn source_of(r: &ThreatRow) -> Option<String> {
    if let Some(w) = r.why { return Some(w.to_string()); }
    match r.verdict { VerdictKind::Exchange => Some("exchange".into()), VerdictKind::Unread => Some("unread".into()), VerdictKind::Confirmed if r.two_moves => Some("two_moves".into()), _ => None }
}

// ------------------------------------------------------------------------------------------------------------------------------ the goals

/// ONE GOAL FROM ONE ROW (subgoal_goals.build_goal): the four facts of a threat of ours, read from the LINE where a line was read and from the graph where none was (`basis` says which). Units are bound in the position as the taker sees it to move.
fn build_goal<G: Game + Placed>(b: &Bld<G>, row: &ThreatRow, source: &str) -> Option<Goal> {
    let f = &b.v;
    let (taker, owner) = (b.taker, b.owner);
    let t = row.unit?.at;
    let tgt = f.at(t)?;
    if f.owner(tgt) != owner { return None; }
    let target = f.u(tgt);
    let two = row.two_moves;
    let line: Vec<String> = row.line.clone();
    let plies = if !line.is_empty() && source != "unread" { replay(b.g, &b.s_t, &line) } else { vec![] };
    let story = if !plies.is_empty() { line_story(f, taker, t, &plies) } else { Story { taker_unit: None, losses: vec![], ends: false } };
    let (mut facts, mut fixes, mut steps): (Vec<Fact>, Vec<Fix>, Vec<Step>) = (vec![], vec![], vec![]);
    // ---- bears(A, T)
    let bearers = f.bearers(t, taker);
    let a_uid;
    if !bearers.is_empty() {
        a_uid = if story.taker_unit.is_some_and(|u| bearers.contains(&u)) { story.taker_unit } else { Some(bearers[0]) };
        facts.push(fact("bears", Status::Met, a_uid.map(|i| f.u(i)), None, vec![]));
    } else {
        a_uid = story.taker_unit;
        let shut = f.shut_onto(t, taker);
        if two {
            let mover_uid = plies.first().filter(|p| p.side == taker).and_then(|p| p.uid);
            let fx = Fix { kind: FixKind::FirstMove, against: mover_uid.map(|u| vec![(f.u(u), None)]).unwrap_or_default(), to: None, steps: Some(1), moves: line.first().cloned().into_iter().collect(), words: "the quiet first move that brings a unit to bear" };
            let add = b.bring_fix(t);                                    // the ordinary second fix: any legal move that brings a bearer onto the target
            facts.push(fact("bears", Status::Unmet, a_uid.map(|i| f.u(i)), Some(cause("needs_quiet_move", vec![], None, "line", None)), std::iter::once(fx.clone()).chain(add.clone()).collect()));
            fixes.push(fx);
            if let Some(a) = add { fixes.push(a); }
            if plies.first().is_some_and(|p| p.side == taker) { steps.push(Step { step: "bring", moves: Some(1) }); }
        } else if let Some((u, btw)) = shut.first() {
            let fx = b.free_fix(btw, t);
            facts.push(fact("bears", Status::Unmet, Some(f.u(*u)), Some(cause("blocked", f.of_idx(btw), None, "graph", None)), vec![fx.clone()]));
            fixes.push(fx);
            steps.push(Step { step: "free the line", moves: Some(1) });
        } else {
            let fx = b.bring_fix(t);
            facts.push(fact("bears", Status::Unmet, None, Some(cause("no_bearer", vec![], None, "graph", None)), fx.iter().cloned().collect()));
            if let Some(fx) = fx { steps.push(Step { step: "bring", moves: fx.steps }); fixes.push(fx); }
        }
    }
    // ---- may_take(A, T)
    if !bearers.is_empty() {
        if f.mover != Some(taker) { facts.push(fact("may_take", Status::Open, a_uid.map(|i| f.u(i)), None, vec![])); }
        else if let Some(l) = bearers.iter().find(|u| b.takes_legally(f.part.units[**u].at, t)) { facts.push(fact("may_take", Status::Met, Some(f.u(*l)), None, vec![])); }          // (PLAN 79.08 repair 1: the game's own legal moves, so a tied bearer is not free; the graph's `may` relation does not read the safety of the goal unit)
        else {
            let mut a0 = bearers[0];
            let (mut tied, mut victim): (Vec<UnitRef>, Option<usize>) = (vec![], None);
            // PLAN 79.08 repair 1: the lines the other side's units have through a bearer are the graph's shut relation read with THEM to move (the graph holds the mover's lines): the position after the game's pass
            let pass_view = b.g.pass(&b.s_t).map(|q| view(b.st, b.g, &q, b.w));
            if let Some(pv) = &pass_view {
                for u in &bearers {
                    let bearer = f.u(*u);
                    let mut items: Vec<(&(usize, Place), &Vec<usize>)> = pv.shut.iter().collect();
                    items.sort_by(|x, y| (&pv.ids[x.0 .0], x.0 .1).cmp(&(&pv.ids[y.0 .0], y.0 .1)));
                    for ((x, p), btw) in items {
                        if pv.owner(*x) == owner && btw.len() == 1 && pv.u(btw[0]) == bearer && pv.at(*p).is_some_and(|k| pv.owner(k) == taker) {
                            let xr = pv.u(*x);
                            if !tied.contains(&xr) { tied.push(xr); }
                            if victim.is_none() { victim = f.at(*p); a0 = *u; }
                        }
                    }
                }
            }
            let (c, fx) = if !tied.is_empty() {
                let sts = tied.clone();
                (cause("tied", sts.clone(), victim.map(|v| f.u(v)), "graph", None), Some(b.remove_fix(&sts, None, "the unit of theirs that holds the bearer is taken, blocked or drawn away (the far end a variable)")))
            } else {
                let mine = f.goal_units(taker, b.goal);
                let mut att: Vec<usize> = mine.iter().flat_map(|u| f.bearers(f.part.units[*u].at, owner)).collect();
                att.sort_by(|x, y| f.ids[*x].cmp(&f.ids[*y])); att.dedup();
                let sts = f.of_idx(&att);
                let victim = mine.first().map(|u| f.u(*u));
                let fx = if att.is_empty() { None } else { Some(b.remove_fix(&sts, mine.first().map(|u| f.part.units[*u].at), "the unit of theirs that attacks our goal-kind unit is taken, blocked or drawn away (the far end a variable)")) };
                if att.is_empty() { (cause("no_legal_take", vec![], None, "graph", None), None) } else { (cause("goal_unit_attacked", sts, victim, "graph", None), fx) }
            };
            facts.push(fact("may_take", Status::Unmet, Some(f.u(a0)), Some(c), fx.iter().cloned().collect()));
            if let Some(fx) = fx { fixes.push(fx); }
        }
    } else { facts.push(fact("may_take", Status::Open, None, None, vec![])); }
    // ---- unguarded(T): by the line where one was read (the unit of theirs that took back at the place); by the graph where none was (the units that bear on the place)
    let guards = f.bearers(t, owner);
    let first_guard = story.losses.iter().find(|l| matches!(l.kind, "guard" | "arrives") && l.at == t);
    if !plies.is_empty() {
        match first_guard {
            None => facts.push(fact("unguarded", Status::Met, None, None, vec![])),
            Some(fg) => {
                let r = fg.stopper;
                let mut all = vec![Some(r)]; all.extend(guards.iter().filter(|g| **g != r).map(|g| Some(*g)));
                let sts = f.units_of(&all);
                let fx = b.remove_fix(&[f.u(r)], Some(t), "the unit that takes back is taken, pinned, blocked or drawn away (the far end a variable)");
                let add = b.bring_fix(t);
                facts.push(fact("unguarded", Status::Unmet, Some(f.u(r)), Some(cause("guard", sts, None, "line", None)), std::iter::once(fx.clone()).chain(add.clone()).collect()));
                fixes.push(fx);
                if let Some(a) = add { fixes.push(a); }
            }
        }
    } else if !guards.is_empty() {
        let fx = b.remove_fix(&[f.u(guards[0])], Some(t), "the unit that bears on the place is taken, pinned, blocked or drawn away (the far end a variable)");
        let add = b.bring_fix(t);
        facts.push(fact("unguarded", Status::Unmet, Some(f.u(guards[0])), Some(cause("guard", f.of_idx(&guards), None, "graph", None)), std::iter::once(fx.clone()).chain(add.clone()).collect()));
        fixes.push(fx);
        if let Some(a) = add { fixes.push(a); }
    } else { facts.push(fact("unguarded", Status::Met, None, None, vec![])); }
    // ---- cornered(T): after a quiet first move of ours, the other side's first reply is the target itself moving away
    if (two || source == "two_move_not_ahead") && plies.len() >= 2 {
        if let Some(rep1) = plies[1..].iter().find(|p| p.side == owner) {
            if rep1.uid == Some(tgt) && rep1.took.is_none() { facts.push(fact("cornered", Status::Unmet, None, Some(cause("escapes", vec![], None, "line", None)), vec![])); }
        }
    }
    // ---- kept(A)
    let elsewhere: Vec<&Loss> = story.losses.iter().filter(|l| l.at != t).collect();
    let taker_a = story.taker_unit.map(|i| f.u(i));
    if source == "unread" { facts.push(fact("kept", Status::Unread, None, Some(cause("out_of_budget", vec![], None, "line", None)), vec![])); }
    else if plies.is_empty() { facts.push(fact("kept", Status::Open, None, None, vec![])); }
    else if elsewhere.is_empty() { facts.push(fact("kept", Status::Met, taker_a, None, vec![])); }
    else {
        let l = elsewhere[0];
        let v = l.victim.map(|x| f.u(x));
        let sts = f.units_of(&elsewhere.iter().map(|y| Some(y.stopper)).collect::<Vec<_>>());
        let vics = f.units_of(&elsewhere.iter().map(|y| y.victim).collect::<Vec<_>>());
        let fx = b.remove_fix(&[f.u(l.stopper)], l.victim.map(|x| f.part.units[x].at), "the unit of theirs that takes is taken, pinned, blocked or drawn away (the far end a variable)");
        let mut ft = fact("kept", Status::Unmet, taker_a, Some(cause(l.kind, sts, v, "line", Some(vics))), vec![fx.clone()]);
        ft.place = Some(l.at);
        facts.push(ft);
        fixes.push(fx);
    }
    if source != "unread" && !plies.is_empty() && !facts.iter().any(|x| x.status == Status::Unmet && x.name != "bears") && row.verdict != VerdictKind::Confirmed {
        // the line did not end ahead but no named fact of the line is unmet: the line itself ends not ahead for the taker; name the units of theirs that take in it (the target itself taking is the goal, not something to take out)
        let took_by: Vec<Option<usize>> = plies.iter().filter(|p| p.side == owner && p.took.is_some() && p.uid.is_some() && p.uid != Some(tgt)).map(|p| p.uid).collect();
        let sts = f.units_of(&took_by);
        if story.ends && sts.is_empty() { facts.push(fact("kept", Status::Unmet, None, Some(cause("ends_game", vec![], None, "line", None)), vec![])); }
        else {
            let fx = if sts.is_empty() { None } else { Some(b.remove_fix(&sts, None, "the units of theirs that take in the line are taken, pinned, blocked or drawn away (the far end a variable)")) };
            facts.push(fact("kept", Status::Unmet, taker_a, Some(cause("not_ahead", sts, None, "line", None)), fx.iter().cloned().collect()));
            if let Some(fx) = fx { fixes.push(fx); }
        }
    }
    if source != "unread" && !facts.iter().any(|x| x.status == Status::Unmet) && row.verdict != VerdictKind::Confirmed {
        facts.push(fact("kept", Status::Unmet, None, Some(cause(if story.ends { "ends_game" } else { "arrives" }, vec![], None, "line", None)), vec![]));
    }
    Some(finish(b, taker, target, source, Some(row), facts, fixes, steps, line, two, false))
}

/// THE GOAL UNIT AS A TARGET (subgoal_goals.goal_unit_goal): it cannot be taken, so the claim is the forced end of the game the forcing read looks for (a goal row of the owner's: then the goal works and is no goal). Its facts are the SAME as any target's (a unit of ours bears on it: `bears`, with the same causes and fixes) plus only what the rules add
/// for the goal kind: every place it may move to is covered, and the forced end itself (unmet when the forcing read found none, unread when its fuse cut it). Nothing here is special to a line onto it: what stands in the way is a set of stoppers, and each stopper of the other side is a target with a goal of its own (`goals`, the recursion).
fn goal_unit_goal<G: Game + Placed>(b: &Bld<G>, list: &ConcreteList, s: &G::State) -> Option<Goal> {
    let (taker, owner) = (b.taker, b.owner);
    if list.goal.iter().any(|r| r.owner == owner) { return None; }
    let so = view_state(b.g, s, owner)?;
    let fo = view(b.st, b.g, &so, b.w);
    let f = &b.v;
    let uid = *fo.goal_units(owner, b.goal).first()?;
    let target = fo.u(uid);
    let t = target.at;
    f.at(t)?;
    let (mut facts, mut fixes, mut steps, mut escapes): (Vec<Fact>, Vec<Fix>, Vec<Step>, Vec<Escape>) = (vec![], vec![], vec![], vec![]);
    let bears = f.bearers(t, taker);
    if let Some(a) = bears.first() { facts.push(fact("bears", Status::Met, Some(f.u(*a)), None, vec![])); }
    else if let Some((u, btw)) = f.shut_onto(t, taker).first() {
        let fx = b.free_fix(btw, t);
        facts.push(fact("bears", Status::Unmet, Some(f.u(*u)), Some(cause("blocked", f.of_idx(btw), None, "graph", None)), vec![fx.clone()]));
        fixes.push(fx);
    } else {
        let fx = b.bring_fix(t);
        facts.push(fact("bears", Status::Unmet, None, Some(cause("no_bearer", vec![], None, "graph", None)), fx.iter().cloned().collect()));
        if let Some(fx) = fx { steps.push(Step { step: "bring", moves: fx.steps }); fixes.push(fx); }
    }
    let mut places: Vec<Place> = fo.may.iter().filter(|(u, _)| *u == uid).map(|(_, p)| *p).collect();
    places.sort();
    for p in places {
        let cov = f.bearers(p, taker);
        let occupant = f.at(p).map(|k| f.u(k));
        if !cov.is_empty() {
            escapes.push(Escape { place: p, covered: true, by: f.of_idx(&cov), occupant });
            let mut ft = fact("covered", Status::Met, Some(f.u(cov[0])), None, vec![]); ft.place = Some(p);
            facts.push(ft);
            continue;
        }
        let (mut ft, fx) = if let Some((u, btw)) = f.shut_onto(p, taker).first() {
            let fx = b.free_fix(btw, p);
            (fact("covered", Status::Unmet, Some(f.u(*u)), Some(cause("blocked", f.of_idx(btw), None, "graph", None)), vec![fx.clone()]), Some(fx))
        } else {
            let fx = b.bring_fix(p);
            if let Some(x) = &fx { steps.push(Step { step: "bring", moves: x.steps }); }
            (fact("covered", Status::Unmet, None, Some(cause("no_bearer", vec![], None, "graph", None)), fx.iter().cloned().collect()), fx)
        };
        ft.place = Some(p);
        facts.push(ft);
        if let Some(fx) = fx { fixes.push(fx); }
        escapes.push(Escape { place: p, covered: false, by: vec![], occupant });
    }
    let gap = list.dropped_counts.keys().any(|k| k.contains("out of budget"));
    facts.push(fact("forced_end", if gap { Status::Unread } else { Status::Unmet }, None, Some(cause(if gap { "out_of_budget" } else { "no_forced_end" }, vec![], None, "line", None)), vec![]));
    let mut gl = finish(b, taker, target, "goal_unit", None, facts, fixes, steps, vec![], false, true);
    gl.escapes = escapes;
    Some(gl)
}

/// THE GOALS OF ONE SIDE from a concrete list (subgoal_goals.goals): one per threat of `side` that does not work yet (the rows of the list, `every_unit`'s included) and the goal unit's; ordered by the fewest unmet facts, then the larger stake (a goal with no stake first), then the place.
/// `goal` is the goal kind (no row, no removal of it). `s` is the position the list was read at; the side not to move is read after the game's pass (no goals where the game gives none).
pub fn goals<G: Game + Placed>(st: &Static, g: &G, s: &G::State, list: &ConcreteList, side: Role, goal: Option<Kind>, depth: u8, cap: usize) -> Vec<Goal> {
    let w = Words { roles: g.roles(), kinds: g.kind_names(), places: g.place_names() };
    let Some(s_t) = view_state(g, s, side) else { return vec![] };
    let v = view(st, g, &s_t, &w);
    let b = Bld { st, g, w: &w, goal, taker: side, owner: g.other(side), s_t, v, afters: Default::default(), bring_memo: Default::default(), legal: Default::default() };
    let mut out = vec![];
    for r in list.mover.iter().chain(&list.other).chain(&list.exchange).chain(&list.dropped) {
        if r.goal || r.owner == side { continue; }
        let Some(src) = source_of(r) else { continue };
        if let Some(gl) = build_goal(&b, r, &src) { out.push(gl); }
    }
    if let Some(gu) = goal_unit_goal(&b, list, s) { out.push(gu); }
    out.sort_by(|a, c| {
        let k = |x: &Goal| (x.n_unmet, -x.stake.unwrap_or(9.0), w.places[x.target.at as usize].clone());
        let (x, y) = (k(a), k(c));
        x.0.cmp(&y.0).then(x.1.partial_cmp(&y.1).unwrap_or(std::cmp::Ordering::Equal)).then(x.2.cmp(&y.2))
    });
    // THE RECURSION: every stopper of the other side named by an unmet fact is a target with a goal of its own, the same facts; at most `cap` of them and `depth` levels deep, each with its parent
    let mut seen: HashSet<UnitRef> = out.iter().map(|x| x.target).collect();
    // a stopper that already has a goal of its own (it was asked on its own) is adopted: its parent is the goal it blocks (never a cycle: not an ancestor of the goal it would be a child of)
    for _ in 0..out.len() {
        let mut changed = false;
        for i in 0..out.len() {
            let stops: Vec<(UnitRef, &'static str)> = out[i].facts.iter().filter(|x| x.status.bad()).flat_map(|x| x.cause.iter().flat_map(move |c| c.stoppers.iter().map(move |u| (*u, x.name)))).collect();
            for (u, via) in stops {
                let Some(j) = out.iter().position(|x| x.target == u) else { continue };
                if j == i || out[j].parent.is_some() || out[j].goal_unit { continue; }
                let (mut anc, mut cyc) = (out[i].parent, out[i].target == u);
                while let Some(a) = anc { if a == u { cyc = true; break; } anc = out.iter().find(|x| x.target == a).and_then(|x| x.parent); }
                if cyc { continue; }
                let (par, d) = (out[i].target, out[i].depth + 1);
                out[j].parent = Some(par); out[j].via = Some(via); out[j].depth = d;
                changed = true;
            }
        }
        if !changed { break; }
    }
    let mut at = 0;
    let mut made = 0;
    while at < out.len() && made < cap {
        let (par, lvl) = (out[at].target, out[at].depth);
        let stops: Vec<(UnitRef, &'static str)> = out[at].facts.iter().filter(|x| x.status.bad()).flat_map(|x| x.cause.iter().flat_map(move |c| c.stoppers.iter().map(move |u| (*u, x.name)))).collect();
        at += 1;
        if lvl >= depth { continue; }
        for (u, via) in stops {
            if made >= cap { break; }
            if u.owner != b.owner || Some(u.kind) == goal || !seen.insert(u) { continue; }
            let mut row = ThreatRow::new(b.owner, Some(u), "stopper", vec![], None);
            row.verdict = VerdictKind::Confirmed;
            if let Some(mut sg) = build_goal(&b, &row, "stopper") { sg.parent = Some(par); sg.via = Some(via); sg.depth = lvl + 1; out.push(sg); made += 1; }
        }
    }
    out
}

/// The threats of `side` that already work (a confirmed row with no quiet first move): not goals; kept apart and counted.
pub fn works(list: &ConcreteList, side: Role) -> usize {
    list.mover.iter().chain(&list.other).chain(&list.exchange).chain(&list.dropped).filter(|r| !r.goal && r.owner != side && source_of(r).is_none()).count()
}

// ------------------------------------------------------------------------------------------------------------------------------ goals of two threats (future_goals.py)

/// Two threats of one side on two different units that no one reply meets. `kind`: "open" when one of the two has no reply at all (a unit simply won, with another threat beside it), "fork" when each has replies and no reply is in both.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TwoThreat { pub first: Option<(Place, Place)>, pub threats: Vec<(UnitRef, Option<i64>, usize)>, pub kind: &'static str, pub stake: i64, pub shape: Vec<String>, pub n_pairs: usize }

/// The goals of two threats of one side: the pairs best first, the count of rows that count as threats, the rows cut or needing a quiet move first, whether the forcing read finds a forced end, the threats in two moves (counted apart).
#[derive(Debug, Clone, Default)]
pub struct TwoThreats { pub goals: Vec<TwoThreat>, pub threats: usize, pub unread: usize, pub forced: bool, pub deferred: usize }

/// THE GOALS OF TWO THREATS (future_goals.goals_of_list). The threats of `side` are the confirmed rows against the OTHER side's units (the rows of the list that side's units do not own), no quiet first move; the replies that meet a threat are its kept responses; a pair on two different
/// units whose reply sets have nothing in common is a goal. `kinds` are the game's words for the kinds (the shape is ordered by the record's worth, the larger first).
pub fn two_threat_goals(list: &ConcreteList, side: Role, mover: Role, rec: &Record, kinds: &[String]) -> TwoThreats {
    let against = if side == mover { &list.other } else { &list.mover };
    let (mut rows, mut unread, mut deferred): (Vec<&ThreatRow>, usize, usize) = (vec![], 0, 0);
    for r in against.iter().filter(|r| !r.goal) {
        if r.two_moves || r.deferred { deferred += 1; } else if r.verdict == VerdictKind::Confirmed { rows.push(r); } else { unread += 1; }
    }
    let forced = against.iter().any(|r| r.goal);
    let replies = |r: &ThreatRow| -> HashSet<(Place, Place)> { r.responses.iter().map(|x| x.first).collect() };
    let kind_of = |r: &ThreatRow| r.unit.map_or(String::new(), |u| kinds[u.kind as usize].clone());
    let mut pairs: Vec<TwoThreat> = vec![];
    for i in 0..rows.len() {
        for j in i + 1..rows.len() {
            let (a, b) = (rows[i], rows[j]);
            let (ua, ub) = (a.unit.expect("a unit row"), b.unit.expect("a unit row"));
            if ua.at == ub.at { continue; }
            let (ra, rb) = (replies(a), replies(b));
            if ra.intersection(&rb).next().is_some() { continue; }
            let (net_a, net_b) = (a.net.unwrap_or(0), b.net.unwrap_or(0));
            let big = if net_a >= net_b { a } else { b };
            let mut shape = vec![kind_of(a), kind_of(b)];
            shape.sort_by_key(|k| (-rec.worth.get(k).copied().unwrap_or(0), k.clone()));
            pairs.push(TwoThreat { first: big.first, threats: vec![(ua, a.net, ra.len()), (ub, b.net, rb.len())], kind: if ra.is_empty() || rb.is_empty() { "open" } else { "fork" }, stake: net_a.min(net_b), shape, n_pairs: 0 });
        }
    }
    pairs.sort_by(|x, y| (-x.stake, x.kind).cmp(&(-y.stake, y.kind)));
    let n = pairs.len();
    for p in pairs.iter_mut() { p.n_pairs = n; }
    TwoThreats { goals: pairs, threats: rows.len(), unread, forced, deferred }
}

// ------------------------------------------------------------------------------------------------------------------------------ one move under two goals

/// One legal move that is a fix under two or more goals of one side (two different targets), with each target and the fact the move addresses there. It is the source of the goals of two threats: a move that adds a bearer on one target and whose leaving uncovers a line onto another is found by the same
/// fixes, with no pattern of its own.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SharedMove { pub side: Role, pub mv: String, pub under: Vec<(UnitRef, &'static str)> }

/// THE MOVES THAT ARE A FIX UNDER TWO GOALS, reported once each with every target: the most targets first, then by name.
pub fn shared_fix_moves(goals: &[Goal]) -> Vec<SharedMove> {
    let mut m: BTreeMap<(Role, String), Vec<(UnitRef, &'static str)>> = BTreeMap::new();
    for gl in goals {
        for f in &gl.facts {
            for fx in &f.fixes { for mv in &fx.moves { let e = m.entry((gl.side, mv.clone())).or_default(); if !e.contains(&(gl.target, f.name)) { e.push((gl.target, f.name)); } } }
        }
    }
    let mut out: Vec<SharedMove> = m.into_iter().filter(|(_, u)| u.iter().map(|x| x.0).collect::<HashSet<_>>().len() >= 2).map(|((side, mv), under)| SharedMove { side, mv, under }).collect();
    out.sort_by(|a, b| (std::cmp::Reverse(a.under.len()), &a.mv).cmp(&(std::cmp::Reverse(b.under.len()), &b.mv)));
    out
}

// ------------------------------------------------------------------------------------------------------------------------------ the same machinery read from the other side: the defence

/// A candidate answer to a goal of THEIRS against a unit of ours: a legal move of ours, and the facts of their goal it makes unmet (`bears`: no unit of theirs bears on the target any more, by taking the bearer, putting a unit between or moving the target away; `unguarded`: a unit of ours that holds the place was
/// added; `may_take`: the bearer is tied, its taking would uncover a line onto a unit of theirs or leave their goal unit borne on). The answer has no name beyond the fact. Whether it works is the line read's verdict.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Answer { pub mv: String, pub facts: Vec<&'static str> }

fn goal_unit_borne_on<G: Game + Placed>(g: &G, s: &G::State, attacker: Role, defender: Role, goal: Option<Kind>) -> bool {
    g.cells(s).into_iter().filter(|c| c.0 == attacker && Some(c.1) == goal).any(|c| bearer_counts_of(g, s, defender, c.2) > 0)
}
fn bearer_counts_of<G: Game + Placed>(g: &G, s: &G::State, owner: Role, place: Place) -> usize { g.cells(s).into_iter().filter(|c| c.0 == owner && g.bearing_from(s, c.0, c.1, c.2).contains(&place)).count() }
fn tied_bearer(part: &PositionPart, attacker: Role, defender: Role, t: Place) -> bool {
    let bearers: Vec<usize> = part.bears.iter().filter(|r| r.place == t && part.units[r.unit].owner == attacker).map(|r| r.unit).collect();
    part.shuts.iter().any(|r| part.units[r.unit].owner == defender && r.between.len() == 1 && bearers.contains(&r.between[0]) && part.occupant[r.place as usize] != 0 && part.units[part.occupant[r.place as usize] as usize - 1].owner == attacker)
}

/// THE ANSWERS to the goal of `attacker` against `target` (a unit of the other side), generated from the legal moves of the target's owner by the fact each un-meets. Only the facts that are met now are addressed (a fact already unmet needs no answer).
pub fn answers_against<G: Game + Placed>(st: &Static, g: &G, s: &G::State, attacker: Role, target: UnitRef, goal: Option<Kind>) -> Vec<Answer> {
    let d = target.owner;
    let Some(sd) = view_state(g, s, d) else { return vec![] };
    let Some(pd) = g.pass(&sd) else { return vec![] };
    let t = target.at;
    if bearer_counts_of(g, &pd, attacker, t) == 0 { return vec![]; }
    let hold0 = bearer_counts_of(g, &pd, d, t);
    let tied0 = tied_bearer(&st.position(g, &sd), attacker, d, t);                // the shut lines the graph holds are the mover's: read with the defender to move
    let att0 = goal_unit_borne_on(g, &pd, attacker, d, goal);
    let mut out = vec![];
    for m in g.moves(&sd) {
        let after = g.apply(&sd, &m);
        let e = g.move_ends(&sd, &m);
        let t2 = if e.0 == t { e.1 } else { t };
        let mut facts: Vec<&'static str> = vec![];
        if bearer_counts_of(g, &after, attacker, t2) == 0 { facts.push("bears"); }
        if t2 == t && bearer_counts_of(g, &after, d, t) > hold0 { facts.push("unguarded"); }
        if g.outcome(&after).is_none() && ((!tied0 && g.pass(&after).is_some_and(|dp| tied_bearer(&st.position(g, &dp), attacker, d, t2))) || (!att0 && goal_unit_borne_on(g, &after, attacker, d, goal))) { facts.push("may_take"); }
        if !facts.is_empty() { out.push(Answer { mv: g.move_name(&sd, &m), facts }); }
    }
    out
}

// ------------------------------------------------------------------------------------------------------------------------------ what a goal gives the search

/// THE MOVES BY POSITION that the fixes name, for both sides, in the form `Search::named` takes: the position's `Game::key` to the names of the legal moves to read first and unreduced there. The mover's goals name moves at the position itself; the other side's at the position after the game's pass
/// (the moves it would make if we did nothing: a move we must read in full). A move is listed once, the goal unit's goal first, the answers to their goals after the fixes' moves. Never added to a value: a name orders and exempts a move from the reduction, nothing more. Not wired into the wake's loop here.
pub fn named_moves<G: Game + Placed>(g: &G, s: &G::State, goals: &[Goal], defence: &[(UnitRef, Vec<Answer>)]) -> HashMap<u64, Vec<String>> {
    let mover = g.mover(s);
    let mut out: HashMap<u64, Vec<String>> = HashMap::new();
    let mut put = |side: Role, mv: &str| {
        let key = if Some(side) == mover { Some(g.key(s)) } else { g.pass(s).map(|p| g.key(&p)) };
        if let Some(k) = key { let e = out.entry(k).or_default(); if !e.iter().any(|x| x == mv) { e.push(mv.to_string()); } }
    };
    let mut order: Vec<&Goal> = goals.iter().collect();
    order.sort_by_key(|x| !x.goal_unit);
    for gl in order { for fx in &gl.fixes { for m in &fx.moves { put(gl.side, m); } } }
    for (target, ans) in defence { for a in ans { put(target.owner, &a.mv); } }
    out
}

// ------------------------------------------------------------------------------------------------------------------------------ the proposer

/// The parameters of `propose`: the budget of questions and the sides whose goals are wanted (None: both).
#[derive(Debug, Clone)]
pub struct GoalParams { pub budget: usize, pub sides: Option<Vec<Role>>, pub depth: u8,
    /// PLAN 79.08 B (the carry): rows of the last decision whose read set did not change; put into the list as they were, not read again
    pub held: Vec<crate::carry::CarriedRow> }
impl Default for GoalParams { fn default() -> GoalParams { GoalParams { budget: BUDGET, sides: None, depth: DEPTH, held: vec![] } } }

/// What the proposer returns for a position: the list of both sides, the goals of each side (in the order of their side's goals), the goals of two threats, the moves that make two threats at once, the units asked and not asked, the questions, the lines read, and the time of each part.
#[derive(Debug, Clone, Default)]
pub struct Proposal {
    pub list: ConcreteList, pub goals: Vec<Goal>, pub two: Vec<(Role, TwoThreats)>, pub shared: Vec<SharedMove>, pub defence: Vec<(UnitRef, Vec<Answer>)>, pub asked: Vec<UnitRef>, pub unasked: Vec<UnitRef>,
    /// the questions asked: one unit asked, the contested units of the list (the first, in the order of their stake) and the every-unit reads
    pub questions: usize, pub line_reads: usize, pub seconds: Vec<(&'static str, f64)>,
}

/// THE PROPOSER (SYSTEM_SPEC 7, 4): the concrete list of both sides read in the order of stake within the budget, then the goals of each side, the goals of two threats, the moves that make two threats at once. The goal unit's goal needs no question of a unit: it comes from the graph and
/// the forcing read of the list, first. With `threats.rows` given (the frame read) the contested rows are the caller's and not budgeted. Called by the wake (79.08) and the tests.
pub fn propose<G: Game + Placed>(st: &Static, g: &G, s: &G::State, rec: &Record, threats: &ThreatParams, params: &GoalParams) -> Proposal {
    let mut out = Proposal::default();
    let Some(mover) = g.mover(s) else { return out };
    let sides: Vec<Role> = params.sides.clone().unwrap_or_else(|| vec![mover, g.other(mover)]);
    let kinds = g.kind_names();
    // the contested units (a unit of the other side bears on) in the order of their stake: the first `budget` are read by the list, the rest are not asked
    let t0 = Instant::now();
    let mut rest: Vec<UnitRef> = vec![];
    let mut first_units: Vec<UnitRef> = vec![];
    let mut rows = threats.rows.clone();
    if rows.is_none() {
        let mut c = graph_rows(st, g, s, threats.goal);
        let by_stake = c.iter().all(|r| threats.stake.is_some_and(|f| f(&kinds[r.unit.kind as usize], r.unit.owner == mover).is_some()));
        let worth = kind_worths(g, s, rec, &c.iter().map(|r| r.unit).collect::<Vec<_>>());
        let key = |r: &crate::threats::FrameRow| -> f64 { if by_stake { threats.stake.and_then(|f| f(&kinds[r.unit.kind as usize], r.unit.owner == mover)).unwrap_or(0.0) } else { worth.get(&r.unit.kind).copied().unwrap_or(0.0) } };
        c.sort_by(|a, b| key(b).partial_cmp(&key(a)).unwrap_or(std::cmp::Ordering::Equal).then(a.unit.at.cmp(&b.unit.at)));
        let keep = params.budget.min(c.len());
        rest = c.split_off(keep).into_iter().map(|r| r.unit).collect();
        first_units = c.iter().map(|r| r.unit).collect();
        c.sort_by_key(|r| (r.unit.owner, r.unit.at));
        rows = Some(c);
    }
    let tp = ThreatParams { depth: threats.depth, fuse: threats.fuse, sweep_depth: threats.sweep_depth, sweep_fuse: threats.sweep_fuse, end_plies: threats.end_plies, end_fuse: threats.end_fuse, goal: threats.goal, rows, every_sides: Some(vec![]), stake: threats.stake, share_table: threats.share_table };
    let mut list = concrete_threats(st, g, s, rec, &tp);
    crate::carry::merge_held(&mut list, &params.held);
    out.seconds.push(("the concrete list of the contested units and the forced end", t0.elapsed().as_secs_f64()));
    // progress after the contested reads: the last unit asked (the least stake) ended with a line read
    let progress = first_units.last().map_or(true, |u| list.mover.iter().chain(&list.other).chain(&list.exchange).chain(&list.dropped).any(|r| r.unit == Some(*u) && r.net.is_some()));
    let t0 = Instant::now();
    let ev = every_unit_ask(st, g, s, &mut list, &sides, rec, &tp, &Ask { budget: Some(params.budget), used: first_units.len(), progress });
    out.seconds.push(("every unit of the other side, by stake within the budget", t0.elapsed().as_secs_f64()));
    list.unasked.extend(rest.iter().copied());
    out.questions = first_units.len() + ev.questions;
    out.line_reads = ev.line_reads;
    out.asked = first_units.iter().copied().chain(ev.asked.iter().map(|a| a.0)).collect();
    out.unasked = list.unasked.clone();
    let t0 = Instant::now();
    for &side in &sides { out.goals.extend(goals(st, g, s, &list, side, tp.goal, params.depth, params.budget)); }
    out.seconds.push(("the goals", t0.elapsed().as_secs_f64()));
    let t0 = Instant::now();
    for &side in &sides { out.two.push((side, two_threat_goals(&list, side, mover, rec, &kinds))); }
    out.seconds.push(("the goals of two threats", t0.elapsed().as_secs_f64()));
    let t0 = Instant::now();
    out.shared = shared_fix_moves(&out.goals);
    out.seconds.push(("the moves that are a fix under two goals", t0.elapsed().as_secs_f64()));
    let t0 = Instant::now();
    for r in list.mover.iter().chain(&list.other).filter(|r| !r.goal).take(params.budget) {
        if let Some(u) = r.unit { let atk = g.other(u.owner); if sides.contains(&atk) { out.defence.push((u, answers_against(st, g, s, atk, u, tp.goal))); } }
    }
    out.seconds.push(("the answers to their goals", t0.elapsed().as_secs_f64()));
    out.list = list;
    out
}

/// The roles' groups of goals by side, for the callers that read one side (a view of `Proposal::goals`).
pub fn goals_of_side(p: &Proposal, side: Role) -> Vec<&Goal> { p.goals.iter().filter(|x| x.side == side).collect() }

#[allow(dead_code)]
fn _unused(_: BTreeMap<u8, u8>) {}
