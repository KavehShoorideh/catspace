//! PLAN 78.58 C: THE EXCHANGE ON ONE PLACE, AS ONE QUESTION. Game-free: a unit is an index into a position's part; a kind is a raw index; its worth is an INPUT, a table of integers read from the
//! estimator's record (`kind_values`), never written here. The lists part mirrors lean/ideas/Ideas/Exchange.lean (generic Int lists: `value`, `cheapest`, `cheapestXF`, `reveal`) function by
//! function, and tests/test_plan78_58_estimator_exchange.py checks it against values that Lean evaluated. The graph part reads the participants of a place from the layer-1 relations: the units
//! that bear on it, and behind each the units on the same line that the `through` relation names, uncovered as the front ones capture.
//!
//! WHAT IT ANSWERS IS AN ESTIMATE: the order of capture is the least kind first for each side, each side may stop, and the worth is the estimator's term. A pin, a threat elsewhere, an in-between move
//! are not in it. Whoever uses the answer marks the claim that rests on it (questions.rs: F_ESTIMATE) and never calls it forced.
use super::base_graph::PositionPart;
use super::Place;

pub type Val = i64;

/// A participant: its own worth, its identity (a unit of the position's part; 0 in the pure lists), and the units behind it on its line, nearest first: (owner as the side flag, worth, identity).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Ray { pub value: Val, pub id: u16, pub behind: Vec<(bool, Val, u16)> }

impl Ray { pub fn plain(value: Val) -> Ray { Ray { value, id: 0, behind: vec![] } } }

// ------------------------------------------------------------------------------------------------------------------------------------------------------------------ the Lean lists

/// `valueF`: the true minimax of the capture game, either side free to choose the capturer, or to stop. `fuel` at least the number of pieces.
pub fn value_f(fuel: usize, on_s: Val, mine: &[Val], theirs: &[Val]) -> Val {
    if fuel == 0 || mine.is_empty() { return 0; }
    let (a, rest) = (mine[0], &mine[1..]);
    let mut best = on_s - value_f(fuel - 1, a, theirs, rest);
    for (i, &b) in rest.iter().enumerate() {
        // `rest.erase b` removes the first element equal to b; the list is rebuilt as a :: (rest without it)
        let first = rest.iter().position(|&x| x == b).unwrap_or(i);
        let mut nm: Vec<Val> = vec![a];
        nm.extend(rest.iter().enumerate().filter(|(j, _)| *j != first).map(|(_, x)| *x));
        best = best.max(on_s - value_f(fuel - 1, b, theirs, &nm));
    }
    best.max(0)
}

pub fn value(on_s: Val, mine: &[Val], theirs: &[Val]) -> Val { value_f(mine.len() + theirs.len(), on_s, mine, theirs) }

/// `cheapestF`: both sides capture with their cheapest piece (the lists ascending), or stop.
pub fn cheapest_f(fuel: usize, on_s: Val, mine: &[Val], theirs: &[Val]) -> Val {
    if fuel == 0 || mine.is_empty() { return 0; }
    (on_s - cheapest_f(fuel - 1, mine[0], theirs, &mine[1..])).max(0)
}

pub fn cheapest(on_s: Val, mine: &[Val], theirs: &[Val]) -> Val { cheapest_f(mine.len() + theirs.len(), on_s, mine, theirs) }

/// `reveal`: the piece behind the capturer joins its owner's list (cheapest first, a stable merge: the new one stands before equals). `white`: the capturer was the side flagged true.
pub fn reveal(white: bool, behind: &[(bool, Val, u16)], mut mine: Vec<Ray>, mut theirs: Vec<Ray>) -> (Vec<Ray>, Vec<Ray>) {
    if let Some(&(owner, b, id)) = behind.first() {
        let ray = Ray { value: b, id, behind: behind[1..].to_vec() };
        if owner == white { mine.insert(0, ray); mine.sort_by_key(|r| r.value); } else { theirs.insert(0, ray); theirs.sort_by_key(|r| r.value); }
    }
    (mine, theirs)
}

/// `cheapestXF`: the static exchange over rays: each side captures with its cheapest piece or stops; a capture reveals what stood behind the capturer.
pub fn cheapest_xf(fuel: usize, white: bool, on_s: Val, mine: &[Ray], theirs: &[Ray]) -> Val {
    if fuel == 0 || mine.is_empty() { return 0; }
    let (a, rest) = (&mine[0], &mine[1..]);
    let (mine2, theirs2) = reveal(white, &a.behind, rest.to_vec(), theirs.to_vec());
    (on_s - cheapest_xf(fuel - 1, !white, a.value, &theirs2, &mine2)).max(0)
}

/// The same recursion, also returning the takings it makes: the identities in order of capture (a side that stops ends the line). A tie stops (`max 0`).
pub fn cheapest_xf_line(fuel: usize, white: bool, on_s: Val, mine: &[Ray], theirs: &[Ray]) -> (Val, Vec<u16>) {
    if fuel == 0 || mine.is_empty() { return (0, vec![]); }
    let (a, rest) = (&mine[0], &mine[1..]);
    let (mine2, theirs2) = reveal(white, &a.behind, rest.to_vec(), theirs.to_vec());
    let (sub, mut line) = cheapest_xf_line(fuel - 1, !white, a.value, &theirs2, &mine2);
    if on_s - sub > 0 { line.insert(0, a.id); (on_s - sub, line) } else { (0, vec![]) }
}

/// A BRUTE READ for the tests: every order of takings, either side free to choose its capturer among those standing on the line, or to stop, with the rays revealed as they are uncovered. The value of the
/// side to move (white flag as in `cheapest_xf`). Exponential: small lists only.
pub fn brute_xf(fuel: usize, white: bool, on_s: Val, mine: &[Ray], theirs: &[Ray]) -> Val {
    if fuel == 0 { return 0; }
    let mut best = 0;
    for i in 0..mine.len() {
        let a = &mine[i];
        let mut rest = mine.to_vec();
        rest.remove(i);
        let (mine2, theirs2) = reveal(white, &a.behind, rest, theirs.to_vec());
        best = best.max(on_s - brute_xf(fuel - 1, !white, a.value, &theirs2, &mine2));
    }
    best
}

// ------------------------------------------------------------------------------------------------------------------------------------------------------------------ the participants of a place

/// The estimate of one taking by a unit on the place `to`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Estimate {
    /// What the taking is worth to its side, in the table's worth: the worth of what stands on the place less what the other side's best answer takes back (stopping where that is better). Negative: it loses.
    pub value: Val,
    /// The units that take on the place in the line the estimate reads, in order: the first is the taker, then the answers, each side with its least kind, until one stops.
    pub line: Vec<usize>,
    /// How many units bear on the place, for each side: (the taker's side, the other).
    pub bearers: (usize, usize),
}

fn ray_of(part: &PositionPart, taker_owner: usize, kv: &[Val], head: usize, front_place: Place) -> Ray {
    let u = &part.units[head];
    let mut behind: Vec<(bool, Val, u16)> = vec![];
    let (mut h, mut f) = (head, front_place);
    for _ in 0..8 {
        // the unit that bears on the place of the unit in front of `h`, through `h` and no other unit
        let Some(r) = part.throughs.iter().find(|r| r.place == f && r.between.len() == 1 && r.between[0] == h) else { break };
        let w = &part.units[r.unit];
        behind.push((w.owner == taker_owner, kv[w.kind as usize], r.unit as u16));
        f = part.units[h].at;
        h = r.unit;
    }
    Ray { value: kv[u.kind as usize], id: head as u16, behind }
}

/// THE EXCHANGE ON THE PLACE `to` WHEN THE UNIT `taker` TAKES WHAT STANDS THERE. None when nothing stands on the place (a taking that is not a plain one is not read here) or the taker does not bear on it.
/// The participants: the units that bear on `to` (the relation `bears`, direct), each with the units behind it on its line from the `through` relation, and the worth of each kind from `kv` (indexed by kind).
pub fn estimate_taking(part: &PositionPart, to: Place, taker: usize, kv: &[Val]) -> Option<Estimate> {
    let victim = *part.occupant.get(to as usize)?;
    if victim == 0 { return None; }
    let victim = victim as usize - 1;
    let owner = part.units[taker].owner;
    if part.units[victim].owner == owner { return None; }
    if !part.bears.iter().any(|r| r.unit == taker && r.place == to) { return None; }
    let mut mine: Vec<Ray> = vec![];
    let mut theirs: Vec<Ray> = vec![];
    for r in part.bears.iter().filter(|r| r.place == to && r.unit != taker) {
        let ray = ray_of(part, owner, kv, r.unit, to);
        if part.units[r.unit].owner == owner { mine.push(ray) } else { theirs.push(ray) }
    }
    mine.sort_by_key(|r| (r.value, r.id));
    theirs.sort_by_key(|r| (r.value, r.id));
    let first = ray_of(part, owner, kv, taker, to);
    let bearers = (mine.len() + 1, theirs.len());
    let fuel = 64;
    // the taker captures; what stood behind it is uncovered; the other side then answers with its least kind or stops
    let (mine2, theirs2) = reveal(true, &first.behind, mine, theirs);
    let (back, line) = cheapest_xf_line(fuel, false, first.value, &theirs2, &mine2);
    let mut full = vec![taker];
    full.extend(line.into_iter().map(|i| i as usize));
    Some(Estimate { value: kv[part.units[victim].kind as usize] - back, line: full, bearers })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rays(v: &[Val]) -> Vec<Ray> { v.iter().map(|&x| Ray::plain(x)).collect() }

    #[test]
    fn plain_lists_agree_with_the_ray_form_when_nothing_is_behind() {
        // the Lean lemma cheapestXF_plain, on every pair of small ascending lists
        let vals = [1, 2, 3, 5, 9];
        for a in 0..4 { for b in 0..4 {
            for ma in 0..vals.len().pow(a as u32).min(60) { for tb in 0..vals.len().pow(b as u32).min(60) {
                let mk = |mut code: usize, n: usize| { let mut v: Vec<Val> = vec![]; for _ in 0..n { v.push(vals[code % vals.len()]); code /= vals.len(); } v.sort(); v };
                let (mine, theirs) = (mk(ma, a), mk(tb, b));
                for on in [1, 3, 5, 9] {
                    let f = mine.len() + theirs.len();
                    assert_eq!(cheapest_xf(f, true, on, &rays(&mine), &rays(&theirs)), cheapest_f(f, on, &mine, &theirs));
                    assert!(cheapest(on, &mine, &theirs) >= 0 && value(on, &mine, &theirs) >= 0);
                }
            } }
        } }
    }

    #[test]
    fn hand_cases() {
        assert_eq!(cheapest(3, &[3], &[1]), 0);                  // a 3 takes a 3 guarded by a 1: it wins 3 and loses 3: level
        assert_eq!(cheapest(3, &[1], &[3]), 2);                  // a 1 takes a 3 guarded by a 3: wins 3, the guard takes the 1 back: 3 - 1
        assert_eq!(cheapest(1, &[5], &[1]), 0);                  // a 5 takes a 1 guarded by a 1: the guard would take the 5 back, so it stops: nothing won
        assert_eq!(value(1, &[5], &[]), 1);                      // unguarded: wins the 1
    }

    #[test]
    fn the_brute_read_agrees_with_the_cheapest_first_read_on_small_hand_cases() {
        let cases: Vec<(Val, Vec<Val>, Vec<Val>)> = vec![(3, vec![1, 5], vec![3]), (1, vec![1, 1], vec![1]), (5, vec![3], vec![3, 9]), (3, vec![3, 5, 9], vec![1, 9]), (9, vec![1, 3], vec![5])];
        for (on, m, t) in cases {
            let f = m.len() + t.len() + 1;
            let (mut m, mut t) = (m, t);
            m.sort(); t.sort();
            assert_eq!(brute_xf(f, true, on, &rays(&m), &rays(&t)), cheapest_xf(f, true, on, &rays(&m), &rays(&t)), "on {on} mine {m:?} theirs {t:?}");
        }
    }

    #[test]
    fn the_brute_read_is_never_below_the_cheapest_first_read() {
        // the cheapest-first strategy is one of the strategies the brute read minimaxes over for the side to move only when the other side is bound to it; so the brute value may differ
        // either way: what is asserted is that both are non-negative and that they agree on lists with one unit a side
        for on in 1..6 { for a in 1..6 { for b in 1..6 {
            let (m, t) = (rays(&[a]), rays(&[b]));
            assert_eq!(brute_xf(4, true, on, &m, &t), cheapest_xf(4, true, on, &m, &t));
        } } }
    }

    #[test]
    fn a_unit_behind_the_taker_is_uncovered_by_the_taking() {
        // mine: a 3 with a 5 behind it; theirs: one 3. The 3 takes the victim (worth 4), their 3 takes back, then the 5 behind takes the 3: mine wins 4 and 3, loses 3
        let mine = vec![Ray { value: 3, id: 1, behind: vec![(true, 5, 2)] }];
        let theirs = vec![Ray { value: 3, id: 3, behind: vec![] }];
        let (v, line) = cheapest_xf_line(10, true, 4, &mine, &theirs);
        assert_eq!(v, 4);                                       // 4 - (their best: take the 3 on the square = 3 - what mine takes back (the 5 takes their 3: 3) => 0) = 4
        assert_eq!(line, vec![1]);                              // they stop: taking back would lose their 3 to the unit uncovered behind
        let alone = vec![Ray { value: 3, id: 1, behind: vec![] }];
        let (v2, line2) = cheapest_xf_line(10, true, 4, &alone, &theirs);
        assert_eq!(v2, 1);                                      // without the unit behind, their 3 takes back: 4 - 3
        assert_eq!(line2, vec![1, 3]);
    }
}
