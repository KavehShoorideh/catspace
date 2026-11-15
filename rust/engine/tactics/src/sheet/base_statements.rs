//! PLAN 78.52 E: THE STATEMENTS OVER THE GRAPH. The 26 statements of plan 78.49 (`subgoal::STATEMENTS`) re-expressed as counts over the layer-1 graph (`base_graph`): the bear and may_move
//! relation nodes, the units and their places, the worth of each kind and the goal's kind (supplied from outside, as the sheet's model holds them). They give the SAME NUMBERS as
//! `subgoal::statements` on any position (tested on random positions of two variants). Then the statements the graph can state that the trait alone could not: units by edge distance,
//! pairs of own units bearing on each other along one open line, the open length of a line. Each statement is one line of plain text and one number per side (the side to move, `Us`, and
//! the other, `Them`); the open-length statements are about the board and are one number. Nothing is fitted here; no game is named.
use super::base_graph::{PositionPart, Static};
use super::{Kind, Place, Side};

/// The 26 statements of plan 78.49, the same names and texts, in the same order.
pub const FIRST: usize = 26;

/// (name, plain text) of the statements of a game: the 26 of 78.49, then the new sided ones (the per-direction ones named from the rules' own direction words), in the order of the numbers.
pub fn names(st: &Static) -> Vec<(String, String)> {
    let mut v: Vec<(String, String)> = super::subgoal::STATEMENTS.iter().map(|(a, b)| (a.to_string(), b.to_string())).collect();
    v.push(("units_on_edge".into(), "its units standing on an edge of some direction".into()));
    v.push(("edge_distance_nearest".into(), "its units' distance to the nearest edge (the least over the directions), summed".into()));
    for d in &st.directions { v.push((format!("edge_distance_{d}"), format!("its units' distance to the edge under direction {d}, summed"))); }
    v.push(("goal_edge_nearest".into(), "its goal unit's distance to the nearest edge (0 when it has no goal unit)".into()));
    v.push(("open_pairs".into(), "pairs of its own units that bear on each other along one line with no unit standing between".into()));
    v
}

/// (name, plain text) of the statements about the board, one number: the open length of the lines.
pub const BOARD: [(&str, &str); 3] = [
    ("open_lines", "lines on which no unit stands"),
    ("open_length_longest", "the longest run of places with no unit standing on them, over every line"),
    ("open_length_total", "the longest run of places with no unit standing on them on each line, summed over the lines"),
];

/// The statements of both sides of a position, `[us, them]` (each `names(st).len()` long), and the board statements. `worth` is by kind; `goal_kind` the goal's kind of the game.
pub fn statements(st: &Static, p: &PositionPart, worth: &[f64], goal_kind: Option<Kind>) -> ([Vec<f64>; 2], Vec<f64>) {
    let np = st.names.places.len();
    let nu = p.units.len();
    let w = |k: Kind| worth[k as usize];
    let struck = [p.struck_by(np, Side::Us), p.struck_by(np, Side::Them)];
    // tables over the graph's relation nodes
    let mut bears_of: Vec<Vec<usize>> = vec![vec![]; nu];       // per unit: the relation records it bears by
    let mut bearers_of: Vec<Vec<usize>> = vec![vec![]; np];     // per place: the units that bear on it, ascending
    for (ri, r) in p.bears.iter().enumerate() { bears_of[r.unit].push(ri); bearers_of[r.place as usize].push(r.unit); }
    let mut reach_of: Vec<Vec<Place>> = vec![vec![]; nu];
    for r in &p.moves { reach_of[r.unit].push(r.place); }
    let unit_at = |pl: Place| -> Option<usize> { match p.occupant[pl as usize] { 0 => None, i => Some(i as usize - 1) } };
    let goal_of = |side: Side| -> Option<usize> {
        let k = goal_kind?;
        let mut f = p.units.iter().enumerate().filter(|(_, u)| u.side == side && u.kind == k).map(|(i, _)| i);
        let first = f.next()?;
        if f.next().is_some() { None } else { Some(first) }
    };
    let goals = [goal_of(Side::Us), goal_of(Side::Them)];
    let lowest = p.units.iter().enumerate().filter(|(i, _)| !goals.contains(&Some(*i))).map(|(_, u)| w(u.kind)).fold(f64::INFINITY, f64::min);
    let bears_on = |u: usize, pl: Place| bearers_of[pl as usize].contains(&u);
    let nd = st.directions.len();
    let side = |s: Side| -> Vec<f64> {
        let o = s.other();
        let mine: Vec<usize> = (0..nu).filter(|&i| p.units[i].side == s).collect();
        let goal = goals[s.ix()];
        let their_goal = goals[o.ix()];
        let bearers_on = |pl: Place, side: Side, not: Option<usize>| -> Vec<usize> { bearers_of[pl as usize].iter().copied().filter(|&i| p.units[i].side == side && Some(i) != not).collect() };
        let (mut moves, mut movers, mut material, mut attackers) = (0.0, 0.0, 0.0, 0.0);
        let (mut attacks, mut cheap_attacks, mut bearing_total, mut widest) = (0.0, 0.0, 0.0, 0.0f64);
        let (mut guarded, mut unguarded, mut mutual, mut cheapest) = (0.0, 0.0, 0.0, 0.0);
        let mut open_pairs = 0.0;
        for &i in &mine {
            let u = p.units[i];
            let n_reach = reach_of[i].len() as f64;
            moves += n_reach;
            if n_reach > 0.0 { movers += 1.0; }
            material += w(u.kind);
            bearing_total += bears_of[i].len() as f64;
            widest = widest.max(bears_of[i].len() as f64);
            let mut hits = false;
            for &ri in &bears_of[i] {
                let r = &p.bears[ri];
                if let Some(j) = unit_at(r.place) {
                    let v = p.units[j];
                    if v.side == o { hits = true; attacks += 1.0; if w(u.kind) < w(v.kind) { cheap_attacks += 1.0; } }
                    else if j > i && bears_on(j, u.at) {
                        mutual += 1.0;
                        // along one line with nothing between, from both sides
                        let back = bears_of[j].iter().map(|&x| &p.bears[x]).find(|x| x.place == u.at);
                        if !r.along.is_empty() && r.between.is_empty() && back.is_some_and(|x| !x.along.is_empty() && x.between.is_empty()) { open_pairs += 1.0; }
                    }
                }
            }
            if hits { attackers += 1.0; }
            if bearers_on(u.at, s, Some(i)).is_empty() { unguarded += 1.0 } else { guarded += 1.0 }
            if Some(i) != goal && w(u.kind) == lowest { cheapest += 1.0; }
        }
        let (mut attack_worth, mut loose, mut loose_worth, mut overloaded, mut favourable) = (0.0, 0.0, 0.0, 0.0, 0.0);
        for j in (0..nu).filter(|&j| p.units[j].side == o) {
            let v = p.units[j];
            let by = bearers_on(v.at, s, None);
            if by.is_empty() { continue; }
            attack_worth += w(v.kind);
            let guards = bearers_on(v.at, o, Some(j));
            if guards.is_empty() { loose += 1.0; loose_worth += w(v.kind); }
            if by.len() > guards.len() { overloaded += 1.0; }
            let cheapest_by = by.iter().map(|&i| w(p.units[i].kind)).fold(f64::INFINITY, f64::min);
            if w(v.kind) > cheapest_by { favourable += 1.0; }
        }
        let (mut goal_steps, mut goal_free, mut goal_cover) = (0.0, 0.0, 0.0);
        if let Some(g) = goal {
            let reach = &reach_of[g];
            goal_steps = reach.len() as f64;
            goal_free = reach.iter().filter(|x| !struck[o.ix()].contains(**x)).count() as f64;
            goal_cover = mine.iter().filter(|&&i| Some(i) != goal && reach.iter().any(|x| bears_on(i, *x))).count() as f64;
        }
        let goal_struck = their_goal.map_or(0.0, |g| if struck[s.ix()].contains(p.units[g].at) { 1.0 } else { 0.0 });
        let ring_struck = their_goal.map_or(0.0, |g| reach_of[g].iter().filter(|x| struck[s.ix()].contains(**x)).count() as f64);
        let controlled = struck[s.ix()].len() as f64;
        let empty = struck[s.ix()].iter().filter(|x| unit_at(*x).is_none()).count() as f64;
        let only = struck[s.ix()].iter().filter(|x| !struck[o.ix()].contains(*x)).count() as f64;
        let mut v = vec![moves, movers, mine.len() as f64, material, attackers, attacks, attack_worth, cheap_attacks, loose, loose_worth, overloaded, favourable, guarded, unguarded, mutual,
                         goal_steps, goal_free, goal_cover, goal_struck, ring_struck, controlled, empty, only, bearing_total, widest, cheapest];
        // the new ones
        v.push(mine.iter().filter(|&&i| st.nearest_edge(p.units[i].at) == 0).count() as f64);
        v.push(mine.iter().map(|&i| st.nearest_edge(p.units[i].at) as f64).sum());
        for d in 0..nd { v.push(mine.iter().map(|&i| st.edge_distance[d][p.units[i].at as usize] as f64).sum()); }
        v.push(goal.map_or(0.0, |g| st.nearest_edge(p.units[g].at) as f64));
        v.push(open_pairs);
        v
    };
    let runs: Vec<usize> = (0..st.lines.len()).map(|l| p.open_length(st, l)).collect();
    let board = vec![(0..st.lines.len()).filter(|&l| st.lines[l].places.iter().all(|x| p.occupant[*x as usize] == 0)).count() as f64,
                     runs.iter().copied().max().unwrap_or(0) as f64, runs.iter().sum::<usize>() as f64];
    ([side(Side::Us), side(Side::Them)], board)
}
