//! PLAN 79.10 A, THEIR GOALS AFTER EACH OF OUR MOVES (SYSTEM_SPEC 8 the rule for choosing a move: "evaluate the threats of the opponent"; 7 one way of asking for every goal, the stake; piece: the wake). Their goals were read once, on the position after a pass; but OUR move is what changes their facts: a holder of ours leaves, a line of theirs opens, a thing is put in its way.
//! So for each root move of ours the goals of the other side are read on the position AFTER it (from the graph alone: the goals the proposer builds with no line read, which costs about a thousandth of a second a position), their chain of the highest stake is found, and the fixes of its nearest goal (the goals after it, rule.rs `names_of`) are named
//! for the search AT THAT CHILD POSITION (`Search::named`, by position). The comparison with their goals before our move (read the same way, on the position after a pass) is a FACT ABOUT OUR MOVE for the log: a thing of ours it took out of the way of their chain, a thing it put in the way. No value is added for it (rule 7): a name orders and exempts a move from the reduction,
//! the fact is only said. Game-free: a unit is a number, a move a name.
//!
//! INTERFACE (written before the bodies; who calls: `wake::decide`, behind `Pieces::after_move`, and the unit tests).
//!   `their_goals(st, g, s, rec, goal, side) -> TheirGoals`      the goals of `side` on `s` from the graph alone (no row read), their chains in order of stake, the stoppers of each chain.
//!   `their_goals_after(st, g, s, rec, goal, order, count, stop) -> AfterRead`   for each root move of ours in `order` (indices of the legal moves; the search's own order), until `stop`: the names at the child position and the fact about the move; the moves not reached are listed.
use crate::goals::{self, goals, Goal};
use crate::rule::{self, Stakes};
use crate::threat_verdict::Record;
use crate::threats::ConcreteList;
use crate::wake::UnitRef;
use cs_game::placed::{Kind, Place, Placed};
use cs_game::{Game, Role};

/// One chain of a side's goals as the comparison reads it: the target at its top, its stake, the things in the way of its nearest goal, and every stopper named in the chain.
#[derive(Debug, Clone, PartialEq)]
pub struct ChainView { pub top: UnitRef, pub stake: f64, pub in_the_way: usize, pub stoppers: Vec<UnitRef> }

/// The goals of one side on one position, read from the graph alone, with their chains in order of stake.
#[derive(Debug, Clone, Default)]
pub struct TheirGoals { pub goals: Vec<Goal>, pub chains: Vec<rule::ChainFix>, pub views: Vec<ChainView> }

/// THE GOALS OF `side` ON `s` FROM THE GRAPH ALONE (an empty list: no line is read; the goal unit's goal and the recursion of the stoppers come from the relations of the position). `s` is the position as it stands: for the side not to move the goals are read after the game's pass.
pub fn their_goals<G: Game + Placed>(st: &cs_tactics::sheet::base_graph::Static, g: &G, s: &G::State, rec: &Record, goal: Option<Kind>, side: Role) -> TheirGoals {
    let gs = goals(st, g, s, &ConcreteList::default(), side, goal, goals::DEPTH, goals::BUDGET);
    let stakes = Stakes::read(g, s, rec, &gs, goal);
    let chains = rule::named_fixes(&gs, side, &stakes);
    let views = chains.iter().map(|c| {
        let mut stoppers: Vec<UnitRef> = vec![];
        for (i, x) in gs.iter().enumerate().filter(|(i, x)| x.side == side && !x.unmet.is_empty() && rule::chain_top(&gs, *i) == c.top) { let _ = i; for u in &x.stoppers { if !stoppers.contains(u) { stoppers.push(*u); } } }
        stoppers.sort();
        ChainView { top: gs[c.top].target, stake: c.stake, in_the_way: rule::things_in_the_way(&gs[c.goal]), stoppers }
    }).collect();
    TheirGoals { goals: gs, chains, views }
}

/// THE FACT ABOUT ONE MOVE OF OURS (written to the log, adds nothing to a value): their chain of the highest stake after the move, the things in the way of its nearest goal before (None: no such chain stood before) and after, the stoppers (units of ours) that were in its way before and are not now (`took_out`),
/// and those that are now and were not (`put_in`). A unit the move itself moved is followed to where it went, so moving a stopper along is not a taking out and a putting in.
#[derive(Debug, Clone, PartialEq)]
pub struct MoveFact { pub top: UnitRef, pub in_the_way_before: Option<usize>, pub in_the_way_after: usize, pub took_out: Vec<UnitRef>, pub put_in: Vec<UnitRef> }

/// What the other side's goals say after one move of ours: the move, the position's key (`Game::key`), the names of their fixes to be read first at that position, the fact about the move (None: no chain of theirs stands after it).
#[derive(Debug, Clone, PartialEq)]
pub struct TheirAfter { pub mv: String, pub key: u64, pub names: Vec<String>, pub fact: Option<MoveFact> }

/// All the moves read: one record each; the moves the clock did not reach (a gap, never a refutation); the seconds.
#[derive(Debug, Clone, Default)]
pub struct AfterRead { pub rows: Vec<TheirAfter>, pub not_asked: Vec<String>, pub seconds: f64 }

/// The comparison of their chains before our move with their chains after it (the first, the one of the highest stake, after). `moved`: where the move took a unit from and to.
pub fn fact_of(before: &[ChainView], after: &[ChainView], moved: Option<(Place, Place)>) -> Option<MoveFact> {
    let a = after.first()?;
    let b = before.iter().find(|b| b.top.owner == a.top.owner && b.top.kind == a.top.kind);
    let follow = |u: &UnitRef| -> UnitRef { match moved { Some((from, to)) if u.at == from => UnitRef { at: to, ..*u }, _ => *u } };
    let bs: Vec<UnitRef> = b.map(|b| b.stoppers.iter().map(follow).collect()).unwrap_or_default();
    let took_out: Vec<UnitRef> = bs.iter().filter(|u| !a.stoppers.contains(u)).copied().collect();
    let put_in: Vec<UnitRef> = a.stoppers.iter().filter(|u| b.is_some() && !bs.contains(u)).copied().collect();
    Some(MoveFact { top: a.top, in_the_way_before: b.map(|b| b.in_the_way), in_the_way_after: a.in_the_way, took_out, put_in })
}

/// THEIR GOALS AFTER EACH OF OUR MOVES (see the module doc). `order`: indices of the legal moves of `s`, the search's own order; `count`: how many moves are named at a child; `stop`: the clock of the step (None: every move). A child where the game has ended or the other side is not to move gets no names.
#[allow(clippy::too_many_arguments)]
pub fn their_goals_after<G: Game + Placed>(st: &cs_tactics::sheet::base_graph::Static, g: &G, s: &G::State, rec: &Record, goal: Option<Kind>, order: &[usize], count: usize, stop: Option<std::time::Instant>) -> AfterRead {
    let t0 = std::time::Instant::now();
    let mut out = AfterRead::default();
    let Some(me) = g.mover(s) else { return out };
    let other = g.other(me);
    let moves = g.moves(s);
    let before = their_goals(st, g, s, rec, goal, other).views;
    for &i in order {
        let m = &moves[i];
        let name = g.move_name(s, m);
        if stop.is_some_and(|t| std::time::Instant::now() >= t) { out.not_asked.push(name); continue; }
        let child = g.apply(s, m);
        if g.outcome(&child).is_some() || g.mover(&child) != Some(other) { continue; }
        let tg = their_goals(st, g, &child, rec, goal, other);
        let e = g.move_ends(s, m);
        out.rows.push(TheirAfter { mv: name, key: g.key(&child), names: rule::names_of(&tg.chains, count), fact: fact_of(&before, &tg.views, Some((e.0, e.1))) });
    }
    out.seconds = t0.elapsed().as_secs_f64();
    out
}
