//! SEARCH INSIDE A FRAME and the two exchange frames (the Python sheet's `frame_search.FrameSearch`, `_RealExchange`, `_StaticExchange`, `_line_of`, `_solve`), game-free.
//! A frame is a small game: states, the moves of the side to move, what a move does, and an exit (the side to move may always stop). Values are the side to move's (negamax):
//! value(state) = max(exit(state), max over moves of gain - value(next)). The search keeps what it derived (a record per state: the bounds of the fail-soft window), stops at a node
//! budget (a node the budget stopped is marked OPEN: the records resting on it never answer again, as in the Python form with no lemmas and no prior) and counts its nodes.
use super::reading::Reading;
use super::{Place, Unit};
use cs_game::placed::Placed;
use cs_game::Game;
use std::collections::HashMap;
use std::hash::Hash;

pub trait Frame {
    type State: Clone;
    type Move: Clone;
    type Key: Eq + Hash + Clone;
    fn moves(&self, s: &Self::State) -> Vec<Self::Move>;
    fn gain(&self, s: &Self::State, m: &Self::Move) -> f64;
    fn next(&self, s: &Self::State, m: &Self::Move) -> Self::State;
    fn exit(&self, s: &Self::State) -> f64;
    fn key(&self, s: &Self::State) -> Self::Key;
    /// A move named for a line.
    fn label(&self, s: &Self::State, m: &Self::Move) -> String;
}

pub struct FrameSearch<'f, F: Frame> {
    frame: &'f F,
    budget: Option<u64>,
    pub nodes: u64,
    pub stopped: u64,
    table: HashMap<F::Key, Vec<(f64, f64, bool)>>,
}

impl<'f, F: Frame> FrameSearch<'f, F> {
    pub fn new(frame: &'f F, budget: Option<u64>) -> Self { FrameSearch { frame, budget, nodes: 0, stopped: 0, table: HashMap::new() } }

    /// -> (value for the side to move, whether it rests on a node the budget left open).
    fn search(&mut self, state: &F::State, alpha: f64, beta: f64) -> (f64, bool) {
        let key = self.frame.key(state);
        if let Some(recs) = self.table.get(&key) {
            for &(lo, hi, open) in recs {
                if !open && (lo == hi || lo >= beta || hi <= alpha) { return (if lo == hi || lo >= beta { lo } else { hi }, false); }
            }
        }
        if let Some(b) = self.budget {
            if self.nodes >= b { self.stopped += 1; return (self.frame.exit(state), true); }
        }
        self.nodes += 1;
        let a0 = alpha;
        let mut alpha = alpha;
        let mut open = false;
        let mut best = self.frame.exit(state);
        if best < beta {
            alpha = alpha.max(best);
            for m in self.frame.moves(state) {
                let g = self.frame.gain(state, &m);
                let (cv, cop) = self.search(&self.frame.next(state, &m), g - beta, g - alpha);
                let v = g - cv;
                open |= cop;
                best = best.max(v);
                alpha = alpha.max(v);
                if alpha >= beta { break; }
            }
        }
        let rlo = if best > a0 { best } else { f64::NEG_INFINITY };
        let rhi = if best < beta { best } else { f64::INFINITY };
        self.table.entry(key).or_default().push((rlo, rhi, open));
        (best, open)
    }

    pub fn value(&mut self, state: &F::State) -> f64 { self.search(state, f64::NEG_INFINITY, f64::INFINITY).0 }

    /// The line the best taking leads to (the moves of both sides in turn), empty when no taking is worth more than stopping or none exists.
    pub fn line_of(&mut self, state: &F::State) -> Vec<String> {
        let mut out = vec![];
        let moves = self.frame.moves(state);
        if moves.is_empty() { return out; }
        let root: Vec<f64> = moves.iter().map(|m| self.frame.gain(state, m) - self.value(&self.frame.next(state, m))).collect();
        let mut bi = 0;
        for (i, v) in root.iter().enumerate() { if *v > root[bi] { bi = i; } }
        out.push(self.frame.label(state, &moves[bi]));
        let mut state = self.frame.next(state, &moves[bi]);
        for _ in 0..20 {
            let v = self.value(&state);
            let mut ties: Vec<F::Move> = vec![];
            for m in self.frame.moves(&state) {
                if self.frame.gain(&state, &m) - self.value(&self.frame.next(&state, &m)) == v { ties.push(m); }
            }
            if ties.is_empty() || self.frame.exit(&state) == v { break; }
            let m = ties.swap_remove(0);
            out.push(self.frame.label(&state, &m));
            state = self.frame.next(&state, &m);
        }
        out
    }
}

/// What one frame solved to: the starter's value (it may stop: never below 0), its line, where it was read from, whether the budget left it open, the nodes used.
#[derive(Debug, Clone)]
pub struct Solved { pub value: f64, pub line: Vec<String>, pub source: &'static str, pub open: bool, pub nodes: u64 }

pub fn solve<F: Frame>(frame: &F, state: &F::State, budget: u64, source: &'static str) -> Solved {
    let mut fs = FrameSearch::new(frame, Some(budget));
    let value = fs.value(state);
    let line = if value >= 0.0 { fs.line_of(state) } else { vec![] };
    Solved { value, line, source, open: fs.stopped > 0, nodes: fs.nodes }
}

// ---------------------------------------------------------------------------------------------------------------------------------------------- the real exchange

/// The exchange frame over real positions (the side to move being the side that may take): its moves are the legal moves that take the unit on one place, the cheapest taker
/// first; the gain of a taking is the worth of the unit taken; the exit is 0 (the side to move may stop).
pub struct Real<'a, 'g, G: Game + Placed> { pub r: &'a Reading<'g, G>, pub place: Place }

impl<G: Game + Placed> Frame for Real<'_, '_, G> {
    type State = G::State;
    type Move = G::Move;
    type Key = u64;
    fn moves(&self, s: &G::State) -> Vec<G::Move> {
        let g = self.r.g;
        if g.kind_at(s, self.place).is_none() { return vec![]; }
        let mut found: Vec<(f64, String, G::Move)> = vec![];
        for m in g.moves(s) {
            let (from, to) = g.move_ends(s, &m);
            if to != self.place { continue; }
            let kind = g.kind_at(s, from).map_or(0.0, |k| self.r.worth_of(k));
            found.push((kind, g.move_name(s, &m), m));
        }
        found.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then_with(|| a.1.cmp(&b.1)));
        found.into_iter().map(|x| x.2).collect()
    }
    fn gain(&self, s: &G::State, _m: &G::Move) -> f64 { self.r.g.kind_at(s, self.place).map_or(0.0, |k| self.r.worth_of(k)) }
    fn next(&self, s: &G::State, m: &G::Move) -> G::State { self.r.g.apply(s, m) }
    fn exit(&self, _s: &G::State) -> f64 { 0.0 }
    fn key(&self, s: &G::State) -> u64 { self.r.g.key(s) }
    fn label(&self, s: &G::State, m: &G::Move) -> String { self.r.g.move_name(s, m) }
}

pub fn solve_real<G: Game + Placed>(r: &Reading<G>, place: Place, state: &G::State, budget: u64, source: &'static str) -> Solved {
    solve(&Real { r, place }, state, budget, source)
}

// ---------------------------------------------------------------------------------------------------------------------------------------------- the static exchange

/// A striker: (its worth, whether it may take only when the other side has none left, a label).
#[derive(Debug, Clone, PartialEq)]
pub struct Striker { pub worth: f64, pub only_last: bool, pub unit: Unit }

/// A state: (strikers of the side to move, strikers of the other side, worth of the unit standing on the place).
pub type StaticState = (Vec<Striker>, Vec<Striker>, f64);

/// The same frame read from the board as it stands: the standard swap with the option to stop. Lines behind the strikers are not read.
pub struct Static<'n> { pub names: &'n super::Names }

impl Frame for Static<'_> {
    type State = StaticState;
    type Move = usize;
    type Key = (Vec<(u64, bool, Unit)>, Vec<(u64, bool, Unit)>, u64);
    fn moves(&self, s: &StaticState) -> Vec<usize> {
        for (i, st) in s.0.iter().enumerate() { if !st.only_last || s.1.is_empty() { return vec![i]; } }
        vec![]
    }
    fn gain(&self, s: &StaticState, _m: &usize) -> f64 { s.2 }
    fn next(&self, s: &StaticState, m: &usize) -> StaticState {
        let mut mine = s.0.clone();
        let taker = mine.remove(*m);
        (s.1.clone(), mine, taker.worth)
    }
    fn exit(&self, _s: &StaticState) -> f64 { 0.0 }
    fn key(&self, s: &StaticState) -> Self::Key {
        let f = |v: &Vec<Striker>| v.iter().map(|x| (x.worth.to_bits(), x.only_last, x.unit)).collect::<Vec<_>>();
        (f(&s.0), f(&s.1), s.2.to_bits())
    }
    fn label(&self, s: &StaticState, m: &usize) -> String { self.names.words(&s.0[*m].unit) }
}

/// The strikers of `units` as the static frame reads them: sorted by (worth, goal kind, words) as the Python sheet sorts its items.
pub fn static_state<G: Game + Placed>(r: &Reading<G>, strikers: &[Unit], guards: &[Unit], occupant: f64) -> StaticState {
    let goal = r.goal_kind();
    let items = |us: &[Unit]| -> Vec<Striker> {
        let mut v: Vec<(f64, bool, String, Unit)> = us.iter().map(|u| (r.worth_of(u.kind), Some(u.kind) == goal, r.m.names.words(u), *u)).collect();
        v.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then_with(|| a.1.cmp(&b.1)).then_with(|| a.2.cmp(&b.2)));
        v.into_iter().map(|(worth, only_last, _, unit)| Striker { worth, only_last, unit }).collect()
    };
    (items(strikers), items(guards), occupant)
}

pub fn solve_static<G: Game + Placed>(r: &Reading<G>, state: &StaticState, budget: u64) -> Solved {
    solve(&Static { names: &r.m.names }, state, budget, "static")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sheet::{Names, Side};

    fn striker(worth: f64, at: Place) -> Striker { Striker { worth, only_last: false, unit: Unit { side: Side::Us, kind: 0, at } } }

    fn names() -> Names { Names::new(vec!["a".into(), "b".into(), "c".into()], vec!["k".into()], vec!["x".into(), "y".into()]) }

    /// The swap on a place holding a unit worth 5: our strikers worth 2 and 9, their guard worth 3.
    #[test]
    fn the_swap_with_the_option_to_stop_is_solved_exactly() {
        let n = names();
        let frame = Static { names: &n };
        let state: StaticState = (vec![striker(2.0, 0), striker(9.0, 1)], vec![striker(3.0, 2)], 5.0);
        let s = solve(&frame, &state, 1000, "static");
        assert_eq!(s.value, 5.0);                                                 // 5 for the first taking; taking back (2) would cost them the guard to our 9 (3), so they stop
        assert!(!s.open && s.nodes > 0);
        assert_eq!(s.line.len(), 1);
        let none: StaticState = (vec![], vec![striker(3.0, 2)], 5.0);
        assert_eq!(solve(&frame, &none, 1000, "static").value, 0.0);
    }

    #[test]
    fn a_budget_leaves_the_search_open() {
        let n = names();
        let frame = Static { names: &n };
        let state: StaticState = (vec![striker(2.0, 0), striker(9.0, 1)], vec![striker(3.0, 2)], 5.0);
        assert!(solve(&frame, &state, 1, "static").open);
    }
}
