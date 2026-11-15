//! PLAN 78.49 B: STATEMENTS ABOUT A POSITION, game-free. Each statement is one line of text and yields ONE NUMBER PER SIDE (the side to move, `Us`, and the other, `Them`), counted from what
//! the `Placed` trait gives through the light `Reading`: the units, their kinds and places, what each bears on, where each may move, the goal unit of each side and the worth of each kind.
//! Nothing here knows a game, and nothing here is read by a player: the statements are CANDIDATE SUBGOALS, measured by `subgoal_fit_measure.py` against a confirmed claim arriving.
//!
//! Left out because the trait does not give them: which units have moved yet (no history is asked of a game) and a unit's distance to the far edge (a place is a number and a name, no
//! geometry). The statements are asked of the side to move and of the other side alike, so the same position read from the other side gives the two lists swapped (tested).
use super::reading::Reading;
use super::{Place, PlaceSet, Side, Unit};
use cs_game::placed::Placed;
use cs_game::Game;

/// (name, plain text): the statements in the order of the numbers `statements` returns.
pub const STATEMENTS: [(&str, &str); 26] = [
    ("moves", "places its units may move to, summed over its units"),
    ("movers", "its units that may move at all"),
    ("units", "its units"),
    ("material", "the worth of its units, summed"),
    ("attackers", "its units that bear on a unit of the other side"),
    ("attacks", "pairs of one of its units and one of the other side's that it bears on"),
    ("attack_worth", "the worth of the other side's units it bears on, each counted once"),
    ("cheap_attacks", "pairs where the unit that bears is worth less than the unit borne on"),
    ("loose_targets", "the other side's units it bears on that no unit of the other side bears on"),
    ("loose_worth", "the worth of those loose units"),
    ("overloaded", "the other side's units it bears on with more of its units than the other side bears on that place"),
    ("favourable_targets", "the other side's units it bears on that are worth more than the cheapest of its units bearing on them"),
    ("guarded", "its own units that another of its own units bears on"),
    ("unguarded", "its own units that none of its own units bears on"),
    ("mutual_pairs", "pairs of its own units that bear on each other"),
    ("goal_steps", "places its goal unit may move to"),
    ("goal_free_steps", "of them, the places no unit of the other side bears on"),
    ("goal_cover", "its own units that bear on a place its goal unit may move to"),
    ("goal_struck", "1 when it bears on the place of the other side's goal unit"),
    ("ring_struck", "places the other side's goal unit may move to that it bears on"),
    ("controlled", "distinct places it bears on"),
    ("controlled_empty", "of them, the places no unit stands on"),
    ("only_controlled", "places it bears on that the other side does not"),
    ("bearing_total", "places borne on, summed over its units (a place borne on twice counts twice)"),
    ("widest_bearing", "the most places any one of its units bears on"),
    ("cheapest_units", "its units, other than the goal unit, of the least worth of any unit on the board other than a goal unit"),
];

pub const N: usize = STATEMENTS.len();

/// The statement numbers of both sides of a position: `[us, them]`, each `N` long.
pub fn statements<G: Game + Placed>(r: &Reading<G>) -> [Vec<f64>; 2] {
    let np = r.np();
    let struck = [r.struck_by(Side::Us), r.struck_by(Side::Them)];
    let goals = [r.goal_unit(Side::Us), r.goal_unit(Side::Them)];
    let lowest = r.units.iter().filter(|u| !goals.contains(&Some(**u))).map(|u| r.worth_of(u.kind)).fold(f64::INFINITY, f64::min);
    let side = |s: Side| -> Vec<f64> {
        let o = s.other();
        let mine: Vec<(usize, &Unit)> = r.units.iter().enumerate().filter(|(_, u)| u.side == s).collect();
        let goal = r.goal_unit(s);
        let their_goal = r.goal_unit(o);
        let bearers_on = |place: Place, side: Side, not: Option<usize>| -> Vec<usize> { r.bearers_of(place).iter().copied().filter(|&i| r.units[i].side == side && Some(i) != not).collect() };
        let (mut moves, mut movers, mut material, mut attackers) = (0.0, 0.0, 0.0, 0.0);
        let (mut attacks, mut cheap_attacks, mut bearing_total, mut widest) = (0.0, 0.0, 0.0, 0.0f64);
        let (mut guarded, mut unguarded, mut mutual, mut cheapest) = (0.0, 0.0, 0.0, 0.0);
        for (i, u) in &mine {
            let reach = r.one_from(u.kind, s, u.at);
            let n_reach = reach.len() as f64;
            moves += n_reach;
            if n_reach > 0.0 { movers += 1.0; }
            material += r.worth_of(u.kind);
            let b = r.strikes_of(u);
            bearing_total += b.len() as f64;
            widest = widest.max(b.len() as f64);
            let mut hits = false;
            for p in b.iter() {
                if let Some(j) = r.unit_at(p) {
                    let v = &r.units[j];
                    if v.side == o { hits = true; attacks += 1.0; if r.worth_of(u.kind) < r.worth_of(v.kind) { cheap_attacks += 1.0; } }
                    else if j > *i && r.strikes(v, u.at) { mutual += 1.0; }
                }
            }
            if hits { attackers += 1.0; }
            if bearers_on(u.at, s, Some(*i)).is_empty() { unguarded += 1.0 } else { guarded += 1.0 }
            if Some(**u) != goal && r.worth_of(u.kind) == lowest { cheapest += 1.0; }
        }
        // the other side's units this side bears on
        let (mut attack_worth, mut loose, mut loose_worth, mut overloaded, mut favourable) = (0.0, 0.0, 0.0, 0.0, 0.0);
        for (j, v) in r.units.iter().enumerate().filter(|(_, v)| v.side == o) {
            let by = bearers_on(v.at, s, None);
            if by.is_empty() { continue; }
            attack_worth += r.worth_of(v.kind);
            let guards = bearers_on(v.at, o, Some(j));
            if guards.is_empty() { loose += 1.0; loose_worth += r.worth_of(v.kind); }
            if by.len() > guards.len() { overloaded += 1.0; }
            let cheapest_by = by.iter().map(|&i| r.worth_of(r.units[i].kind)).fold(f64::INFINITY, f64::min);
            if r.worth_of(v.kind) > cheapest_by { favourable += 1.0; }
        }
        let (mut goal_steps, mut goal_free, mut goal_cover) = (0.0, 0.0, 0.0);
        if let Some(g) = goal {
            let reach = r.one_from(g.kind, s, g.at);
            goal_steps = reach.len() as f64;
            goal_free = reach.iter().filter(|p| !struck[o.ix()].contains(*p)).count() as f64;
            goal_cover = mine.iter().filter(|(_, u)| Some(**u) != goal && { let b = r.strikes_of(u); reach.iter().any(|p| b.contains(p)) }).count() as f64;
        }
        let goal_struck = their_goal.map_or(0.0, |g| if struck[s.ix()].contains(g.at) { 1.0 } else { 0.0 });
        let ring_struck = their_goal.map_or(0.0, |g| r.one_from(g.kind, o, g.at).iter().filter(|p| struck[s.ix()].contains(*p)).count() as f64);
        let controlled = struck[s.ix()].len() as f64;
        let empty = struck[s.ix()].iter().filter(|p| r.unit_at(*p).is_none()).count() as f64;
        let only = { let mut x: PlaceSet = PlaceSet::new(np); for p in struck[s.ix()].iter() { if !struck[o.ix()].contains(p) { x.insert(p); } } x.len() as f64 };
        vec![moves, movers, mine.len() as f64, material, attackers, attacks, attack_worth, cheap_attacks, loose, loose_worth, overloaded, favourable, guarded, unguarded, mutual,
             goal_steps, goal_free, goal_cover, goal_struck, ring_struck, controlled, empty, only, bearing_total, widest, cheapest]
    };
    [side(Side::Us), side(Side::Them)]
}

/// For each legal move of the side to move: (the move's name, the statements of the position after it as `statements` reads it there, the side to move of the child being `Us` of its list).
/// A move to a position the reading refuses (the game is over) is left out of the list; each entry carries the move's index in the game's own list of legal moves.
pub fn children<G: Game + Placed>(r: &Reading<G>, model: &super::Model) -> Vec<(usize, String, [Vec<f64>; 2])> {
    let mut out = vec![];
    for (ix, m) in r.g.moves(&r.s).iter().enumerate() {
        let child = r.g.apply(&r.s, m);
        if let Ok(c) = Reading::new(r.g, child, model) { out.push((ix, r.g.move_name(&r.s, m), statements(&c))); }
    }
    out
}
