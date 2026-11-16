//! PLAN 78.54 WHAT BECOMES OF EACH UNIT: the tracker, the base rates and the fate of each unit variable of a pattern. Game-free: no unit, place, line or kind of any game is named; a kind is an index the
//! graph gives, a fate class is a number.
//!
//! THE TRACKER. A position is the sorted list of its units (owner, kind, place), the order of the layer-1 graph's unit nodes. Between two consecutive positions the DIFFERENCE of the lists says what
//! became of each unit of the first:
//!   the same (owner, kind, place) stands in the second           the unit stays (`Alive`)
//!   it is gone and a unit of the same owner and kind has appeared   the unit moved there (`Alive`); where several units of one owner and kind are gone and appeared together they are paired in
//!                                                                 list order, and the plies where that happened are COUNTED (`ambiguous`): the pairing is as it falls, not guessed better
//!   it is gone and a unit of ANOTHER owner stands on its place now   it was taken, by the kind that stands there (`Taken(kind)`)
//!   it is gone otherwise (nobody on its place; an owner and kind that is not matched, which is a change of kind)  `Other`
//! A unit that moves and returns is the same unit throughout. Two units of one owner and kind that swap places in one step are read as staying (the same triples stand): a limit.
//!
//! THE FATE OF A UNIT AT A PLY, AT A HORIZON h (plies later): follow it step by step. `Taken` or `Other` at any step ends the chain with that class. A chain that is alive after h steps is STAYED when
//! the unit stands on the place it started on (whatever it did between), else MOVED. A horizon that runs past the end of the game has no fate (255). Classes (`ncls = 3 + number of kinds`):
//! 0 stayed, 1 moved, 2 other, 3 + k taken by a unit of kind k.
//!
//! THE BASE RATE of a class is its share among the units of one kind and one side-to-move state (the unit's owner is the side to move at the row, or not), over the rows given.
//!
//! THE FATE OF A PATTERN'S UNIT VARIABLES (`cells`): for each pattern of a wanted set, each of its unit variables and each horizon, over the rows given: the number of distinct units bound to the
//! variable in a row, summed over rows; their fates by class; the sum of their base rates by class (the number expected if each were a unit of its kind and side-to-move state in general); and, per
//! GAME, the residual (taken less the base rate of being taken) summed over the game's units, folded into (sum of the games' residuals, sum of their squares, number of games). The games, not the units,
//! are the independent draws: a unit stays in many rows, so a standard error from the games' residuals (their root sum of squares over the total count) does not pretend the rows are independent.
use super::base_graph::PositionPart;
use super::patterns::{Board, FxHashMap};
use super::{Kind, Place, Side};
use cs_game::Role;

pub type Cell = (Role, Kind, Place);
pub const NO_FATE: u8 = 255;
/// The most unit variables a pattern of three relation nodes can hold (three units each).
pub const MAXV: usize = 9;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Step { Alive(u16), Taken(Kind), Other }

/// What became of each unit of `prev` in `next` (both sorted), and how many units were paired among several of one owner and kind.
pub fn step_between(prev: &[Cell], next: &[Cell]) -> (Vec<Step>, usize) {
    let mut out = vec![Step::Other; prev.len()];
    let mut used = vec![false; next.len()];
    let mut gone: Vec<usize> = vec![];
    for (i, c) in prev.iter().enumerate() {
        match next.binary_search(c) { Ok(j) => { out[i] = Step::Alive(j as u16); used[j] = true; } Err(_) => gone.push(i) }
    }
    let appeared: Vec<usize> = (0..next.len()).filter(|j| !used[*j]).collect();
    let mut ambiguous = 0;
    let mut taken_pair = vec![false; next.len()];
    for &i in &gone {
        let (o, k, at) = prev[i];
        let same = |j: &usize| next[*j].0 == o && next[*j].1 == k;
        if gone.iter().filter(|g| prev[**g].0 == o && prev[**g].1 == k).count() > 1 || appeared.iter().filter(|j| same(j)).count() > 1 { ambiguous += 1; }
        if let Some(&j) = appeared.iter().find(|j| same(j) && !taken_pair[**j]) { taken_pair[j] = true; out[i] = Step::Alive(j as u16); continue; }
        out[i] = match next.iter().find(|c| c.2 == at && c.0 != o) { Some(c) => Step::Taken(c.1), None => Step::Other };
    }
    (out, ambiguous)
}

/// The steps of a whole game: `steps[t][i]` for the unit i of ply t, t < the last ply.
pub struct Track { pub steps: Vec<Vec<Step>>, pub ambiguous: usize }

pub fn track(plies: &[Vec<Cell>]) -> Track {
    let mut steps = vec![];
    let mut ambiguous = 0;
    for t in 0..plies.len().saturating_sub(1) {
        let (s, a) = step_between(&plies[t], &plies[t + 1]);
        steps.push(s);
        ambiguous += a;
    }
    Track { steps, ambiguous }
}

/// The class of the fate of unit `i` of ply `t` after `h` plies; `NO_FATE` where the game ends first.
pub fn fate_code(tr: &Track, plies: &[Vec<Cell>], t: usize, i: usize, h: usize) -> u8 {
    if t + h >= plies.len() { return NO_FATE; }
    let mut cur = i;
    for s in t..t + h {
        match tr.steps[s][cur] { Step::Alive(j) => cur = j as usize, Step::Taken(k) => return 3 + k, Step::Other => return 2 }
    }
    if plies[t + h][cur].2 == plies[t][i].2 { 0 } else { 1 }
}

/// PLAN 78.55 STAGE 0. WHAT A UNIT TOOK AND WHEN, BEFORE ITS OWN END. `fate_code` reads only how a unit ends; a unit that takes and is then taken (an exchange) reads there as TAKEN, the same as a unit lost for
/// nothing. The history follows the same chain and also records, in order, the kind of every unit the unit took (a step in which it moves onto a place where a unit of the other owner stood and was taken in
/// that step). A taking that leaves the victim's place is not seen (the victim's place is not the mover's destination): it reads as the victim's OTHER, as in the tracker. The CLASS of a history, by number:
///   0 stayed, 1 moved, 2 other   (none of them took anything)
///   3 took and is alive           (the kinds it took are in `took`)
///   4 + k           taken by a unit of kind k, having taken nothing: LOST FOR NOTHING
///   4 + nk + k      taken by a unit of kind k after having taken: AN EXCHANGE
/// A horizon past the end of the game has no history (`NO_FATE`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct History { pub took: Vec<Kind>, pub end: End }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum End { Stayed, Moved, Taken(Kind), Other, NoFate }

/// The kind this unit (index `i` of `prev`, now `j` of `next`) took in one step, if it moved onto a place that a unit of the other owner left by being taken.
fn took_in_step(prev: &[Cell], next: &[Cell], i: usize, j: usize) -> Option<Kind> {
    let (o, _, from) = prev[i];
    let to = next[j].2;
    if to == from { return None; }
    prev.iter().find(|c| c.2 == to && c.0 != o).map(|c| c.1)
}

pub fn fate_history(tr: &Track, plies: &[Vec<Cell>], t: usize, i: usize, h: usize) -> History {
    if t + h >= plies.len() { return History { took: vec![], end: End::NoFate }; }
    let (mut cur, mut took) = (i, vec![]);
    for s in t..t + h {
        match tr.steps[s][cur] {
            Step::Alive(j) => { if let Some(k) = took_in_step(&plies[s], &plies[s + 1], cur, j as usize) { took.push(k); } cur = j as usize; }
            Step::Taken(k) => return History { took, end: End::Taken(k) },
            Step::Other => return History { took, end: End::Other },
        }
    }
    History { took, end: if plies[t + h][cur].2 == plies[t][i].2 { End::Stayed } else { End::Moved } }
}

/// The class number of a history (see above); `nk` the number of kinds. `NO_FATE` for a horizon past the end.
pub fn history_class(h: &History, nk: usize) -> u8 {
    let took = !h.took.is_empty();
    match h.end {
        End::NoFate => NO_FATE,
        End::Other => 2,
        End::Stayed | End::Moved if took => 3,
        End::Stayed => 0,
        End::Moved => 1,
        End::Taken(k) => (4 + if took { nk } else { 0 } + k as usize) as u8,
    }
}

/// The base counts of one thread's games: [(group, horizon, kind, side-to-move state)] x (ncls classes then the total with a fate).
pub struct BaseSpec { pub nk: usize, pub ncls: usize, pub nh: usize }

impl BaseSpec {
    pub fn new(nk: usize, nh: usize) -> BaseSpec { BaseSpec { nk, ncls: 3 + nk, nh } }
    pub fn at(&self, group: usize, hi: usize, kind: usize, tm: usize) -> usize { (((group * self.nh + hi) * self.nk + kind) * 2 + tm) * (self.ncls + 1) }
}

/// Add the units of one game's wanted rows to the base counts (`counts` laid out by `BaseSpec::at`); `tomove[t][i]` and `kind` come from the rows' positions.
pub fn add_base(spec: &BaseSpec, group: usize, horizons: &[usize], tr: &Track, plies: &[Vec<Cell>], movers: &[Option<Role>], rows: &[usize], counts: &mut [u64]) {
    for &t in rows {
        let Some(m) = movers.get(t).copied().flatten() else { continue };
        for (i, c) in plies[t].iter().enumerate() {
            for (hi, &h) in horizons.iter().enumerate() {
                let code = fate_code(tr, plies, t, i, h);
                if code == NO_FATE { continue; }
                let base = spec.at(group, hi, c.1 as usize, (c.0 == m) as usize);
                counts[base + code as usize] += 1;
                counts[base + spec.ncls] += 1;
            }
        }
    }
}

/// The units each variable of every wanted pattern binds on one board, as (pattern index, variable, unit), each distinct once.
pub fn hits_board(b: &Board, max_size: usize, levels: &[u8], want: &FxHashMap<u64, u32>) -> Vec<(u32, u8, u16)> {
    let mut out: Vec<(u32, u8, u16)> = vec![];
    for size in 1..=max_size {
        b.connected_sets(size, &mut |its| {
            for key in b.keys(its, levels) {
                if key.sign == 0 { continue; }
                if let Some(&c) = want.get(&key.hash) { for (v, &u) in key.units.iter().enumerate() { out.push((c, v as u8, u)); } }
            }
        });
    }
    out.sort_unstable();
    out.dedup();
    out
}

/// One row to be read: the ply, the board of the position and each unit's (kind, whether its owner is the side to move).
pub struct RowBoard { pub ply: usize, pub board: Board, pub units: Vec<(Kind, bool)> }

impl RowBoard {
    pub fn of(ply: usize, board: Board, p: &PositionPart) -> RowBoard { RowBoard { ply, board, units: p.units.iter().map(|u| (u.kind, u.side == Side::Us)).collect() } }
}

/// The accumulator of the cells of one thread: cell -> [n, games, sum of residuals, sum of squared residuals, counts by class, expected by class].
pub type Cells = FxHashMap<u32, Vec<f64>>;

pub fn cell_index(cand: u32, var: u8, hi: usize, nh: usize) -> u32 { ((cand as usize * MAXV + var as usize) * nh + hi) as u32 }

/// Read one game's rows into `acc`. `rates[((hi * nk + kind) * 2 + tm) * ncls + class]` the base rates.
#[allow(clippy::too_many_arguments)]
pub fn add_cells(spec: &BaseSpec, horizons: &[usize], rates: &[f64], want: &FxHashMap<u64, u32>, max_size: usize, levels: &[u8], tr: &Track, plies: &[Vec<Cell>], rows: &[RowBoard], acc: &mut Cells) {
    let nc = spec.ncls;
    let mut game: FxHashMap<u32, Vec<f64>> = FxHashMap::default();
    for r in rows {
        for (c, v, u) in hits_board(&r.board, max_size, levels, want) {
            let (kind, tm) = r.units[u as usize];
            for (hi, &h) in horizons.iter().enumerate() {
                let code = fate_code(tr, plies, r.ply, u as usize, h);
                if code == NO_FATE { continue; }
                let rate = &rates[((hi * spec.nk + kind as usize) * 2 + tm as usize) * nc..][..nc];
                let e = game.entry(cell_index(c, v, hi, spec.nh)).or_insert_with(|| vec![0.0; 2 + 2 * nc]);
                let taken = (code >= 3) as u8 as f64;
                let expect: f64 = rate[3..].iter().sum();
                e[0] += 1.0;
                e[1] += taken - expect;
                e[2 + code as usize] += 1.0;
                for (k, x) in rate.iter().enumerate() { e[2 + nc + k] += x; }
            }
        }
    }
    for (cell, e) in game {
        let a = acc.entry(cell).or_insert_with(|| vec![0.0; 4 + 2 * nc]);
        a[0] += e[0];
        a[1] += 1.0;
        a[2] += e[1];
        a[3] += e[1] * e[1];
        for k in 0..2 * nc { a[4 + k] += e[2 + k]; }
    }
}

/// Merge another thread's cells.
pub fn merge_cells(into: &mut Cells, other: Cells) {
    for (c, v) in other {
        match into.get_mut(&c) { Some(a) => { for (x, y) in a.iter_mut().zip(v.iter()) { *x += y; } } None => { into.insert(c, v); } }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // owners 0 and 1; kinds 0, 1, 2; places are plain numbers
    fn p(cells: &[Cell]) -> Vec<Cell> { let mut v = cells.to_vec(); v.sort(); v }

    #[test]
    fn a_unit_that_stays_moves_or_is_taken() {
        let a = p(&[(0, 1, 5), (0, 2, 7), (1, 1, 9)]);
        // owner 0's kind 1 moves 5 -> 9 and takes the unit of owner 1 there
        let b = p(&[(0, 1, 9), (0, 2, 7)]);
        let (s, amb) = step_between(&a, &b);
        assert_eq!(amb, 0);
        let at = |c: Cell| a.iter().position(|x| *x == c).unwrap();
        assert_eq!(s[at((0, 2, 7))], Step::Alive(b.iter().position(|x| *x == (0, 2, 7)).unwrap() as u16));
        assert_eq!(s[at((0, 1, 5))], Step::Alive(b.iter().position(|x| *x == (0, 1, 9)).unwrap() as u16));
        assert_eq!(s[at((1, 1, 9))], Step::Taken(1));
    }

    #[test]
    fn a_change_of_kind_is_other_and_a_vanishing_without_a_taker_is_other() {
        let a = p(&[(0, 0, 3), (1, 0, 4)]);
        let b = p(&[(0, 2, 8), (1, 0, 4)]);                      // owner 0's kind 0 is gone and a unit of kind 2 stands elsewhere
        let (s, _) = step_between(&a, &b);
        assert_eq!(s[a.iter().position(|x| *x == (0, 0, 3)).unwrap()], Step::Other);
        let c = p(&[(0, 0, 3)]);                                  // the unit of owner 1 is gone, nobody on its place
        let (s2, _) = step_between(&a, &c);
        assert_eq!(s2[a.iter().position(|x| *x == (1, 0, 4)).unwrap()], Step::Other);
    }

    #[test]
    fn several_of_one_owner_and_kind_gone_and_appeared_are_paired_in_order_and_counted() {
        let a = p(&[(0, 1, 1), (0, 1, 2)]);
        let b = p(&[(0, 1, 5), (0, 1, 6)]);
        let (s, amb) = step_between(&a, &b);
        assert_eq!(amb, 2);
        assert_eq!(s, vec![Step::Alive(0), Step::Alive(1)]);
    }

    #[test]
    fn a_unit_that_took_before_it_was_taken_is_an_exchange_not_a_loss_for_nothing() {
        // owner 0's kind 1 on 5 takes owner 1's kind 0 on 8, then owner 1's kind 2 on 9 takes it back on 8; owner 0's kind 3 on 2 is taken for nothing by owner 1's kind 2 later
        let plies = vec![p(&[(0, 1, 5), (1, 0, 8), (1, 2, 9), (0, 3, 2)]), p(&[(0, 1, 8), (1, 2, 9), (0, 3, 2)]), p(&[(0, 1, 8), (1, 2, 9), (0, 3, 2)]), p(&[(1, 2, 8), (0, 3, 2)]), p(&[(1, 2, 2)])];
        let tr = track(&plies);
        let a = plies[0].iter().position(|x| *x == (0, 1, 5)).unwrap();
        let b = plies[0].iter().position(|x| *x == (0, 3, 2)).unwrap();
        let hist_one = fate_history(&tr, &plies, 0, a, 1);
        assert_eq!(hist_one.took, vec![0]);
        assert_eq!(hist_one.end, End::Moved);
        assert_eq!(history_class(&hist_one, 4), 3);
        let hist_three = fate_history(&tr, &plies, 0, a, 3);
        assert_eq!(hist_three.took, vec![0]);
        assert_eq!(hist_three.end, End::Taken(2));
        assert_eq!(history_class(&hist_three, 4), 4 + 4 + 2);            // an exchange
        let hb = fate_history(&tr, &plies, 0, b, 4);
        assert!(hb.took.is_empty());
        assert_eq!(hb.end, End::Taken(2));
        assert_eq!(history_class(&hb, 4), 4 + 2);                // lost for nothing
        assert_eq!(fate_history(&tr, &plies, 0, a, 9).end, End::NoFate);
    }

    #[test]
    fn the_fate_at_a_horizon_follows_the_chain_and_returning_is_staying() {
        // unit (0,1,5) goes 5 -> 6 -> 5; a unit of the other owner is taken on the third step
        let plies = vec![p(&[(0, 1, 5), (1, 0, 8)]), p(&[(0, 1, 6), (1, 0, 8)]), p(&[(0, 1, 5), (1, 0, 8)]), p(&[(0, 1, 8)])];
        let tr = track(&plies);
        let i = plies[0].iter().position(|x| *x == (0, 1, 5)).unwrap();
        let j = plies[0].iter().position(|x| *x == (1, 0, 8)).unwrap();
        assert_eq!(fate_code(&tr, &plies, 0, i, 1), 1);          // moved
        assert_eq!(fate_code(&tr, &plies, 0, i, 2), 0);          // back where it was: stayed
        assert_eq!(fate_code(&tr, &plies, 0, j, 2), 0);
        assert_eq!(fate_code(&tr, &plies, 0, j, 3), 3 + 1);      // taken by a unit of kind 1 on the third step
        assert_eq!(fate_code(&tr, &plies, 0, i, 4), NO_FATE);    // past the end
    }
}
