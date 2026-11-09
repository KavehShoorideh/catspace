//! PLAN 78.12: the controller-over-operators `decide_params` equals the decision it replaced. `oracle` below is the 78.11 `decide_with`, its body unchanged but for the
//! two depth caps made arguments (so a run ends by depth, never by the clock: the comparison is deterministic). The same moves, the same proved flag, the same trace
//! lines; the new decision's measured nodes also count the tie reading and the density prior, which the old figure left out.
use cs_decision::{decide_params, Params, NODE_FUSE, PROOF_SHARE, TIE_SHARE};
use cs_game::kernel::KernelGame;
use cs_game::rng::Rng;
use cs_game::Game;
use cs_tactics::ledger::{Count, Judge, Worth};
use cs_tactics::{self as tactics, Answer, Budget};
use std::time::Instant;

struct Old { mv: String, proved: bool, trace: Vec<String>, nodes: u64 }

#[allow(clippy::too_many_arguments)]
fn oracle<G: Game, J: Judge>(g: &G, s: &G::State, seconds: f64, judge: &J, prior: bool, history: &[u64], prior_depth: i32, ledger_share: f64, prior_defects: u32,
                             ordering: bool, prune: bool, extend: bool, max_depth: u32, ledger_depth: i32) -> Option<Old> {
    let me = g.mover(s)?;
    let t0 = Instant::now();
    let moves = g.moves(s);
    if moves.is_empty() { return None; }
    let mut trace = vec![format!("decide: {} to move, {} legal moves, {:.2} s", g.roles()[me], moves.len(), seconds)];
    let mut b = Budget::new(seconds * PROOF_SHARE, NODE_FUSE);
    match tactics::force(g, s, me, max_depth, &mut b) {
        Answer::Proved { first, depth } => {
            trace.push(format!("tactics forcing: PROVED -- {} forces the goal within {depth} move(s) against every reply ({} nodes)", g.move_name(s, &first), b.nodes));
            return Some(Old { mv: g.move_name(s, &first), proved: true, trace, nodes: b.nodes });
        }
        Answer::None { depth } => trace.push(format!("tactics forcing: no forced goal within {depth} move(s) ({} nodes)", b.nodes)),
        Answer::Gap { depth } => trace.push(format!("tactics forcing: GAP at depth {depth} -- the clock cut the reading ({} nodes)", b.nodes)),
    }
    let left = (seconds * if prior { ledger_share } else { 1.0 } - t0.elapsed().as_secs_f64()).max(0.01);
    let mut ls = tactics::ledger::Search::new(g, judge, me, left);
    ls.history = history.iter().copied().collect();
    ls.ordering = ordering;
    ls.prune = prune;
    ls.extend = extend;
    let ln = ls.run(s, ledger_depth);
    let nodes = b.nodes + ln.nodes;
    let Some(i) = ln.first_index else {
        trace.push("ledger search: no line read within the clock -- the first legal move".into());
        return Some(Old { mv: g.move_name(s, &moves[0]), proved: false, trace, nodes });
    };
    let mut i = i;
    let mut prior_line = None;
    if prior && ln.value.abs() < tactics::ledger::WIN - 1000 {
        let rest = (seconds - t0.elapsed().as_secs_f64()).max(0.0);
        ls.resume(rest * TIE_SHARE);
        let ties = ls.ties(s, ln.depth, ln.value);
        if ties.len() > 1 {
            let until = t0 + std::time::Duration::from_secs_f64(seconds);
            let fuse = cs_planning::density::FUSE * (1u64 << (2 * (prior_depth - cs_planning::density::DEPTH).max(0)));
            if let Some(pr) = cs_planning::density::prior(g, s, me, &ties, judge, until, prior_depth, fuse, prior_defects) {
                let (j, (adv0, adv1), read) = (pr.index, pr.advantage, pr.read);
                prior_line = Some(format!("planning density: {} of {} equal moves read; {} -- our tactics less theirs after it: {adv0:+} at 0 defects, {adv1:+} at 1 [PRIOR]",
                                          read, ties.len(), g.move_name(s, &g.moves(s)[j])));
                i = j;
            }
        }
    }
    let mv = g.move_name(s, &g.moves(s)[i]);
    let who = |r: cs_game::Role| if r == me { "we" } else { "they" };
    let lost: Vec<String> = ln.lost.iter().map(|(o, k)| format!("{} {k}", who(*o))).collect();
    trace.push(format!("ledger search [{}]: {} at depth {} ({} nodes): the line {} -- {}; value {}", judge.name(), g.move_name(s, &g.moves(s)[ln.first_index.unwrap_or(0)]),
                       ln.depth, ln.nodes, ln.line.join(" "), if lost.is_empty() { "nobody loses a unit".to_string() } else { lost.join(", ") }, ln.value));
    if let Some(p) = prior_line { trace.push(p); }
    Some(Old { mv, proved: false, trace, nodes })
}

/// Positions: a few set up by hand (a forced mate in two, a won queen, a promotion race) and seeded random games stopped at several depths.
fn positions() -> Vec<(KernelGame, <KernelGame as Game>::State, Vec<u64>)> {
    let mut out = vec![];
    for fen in ["6k1/5ppp/8/8/8/8/5PPP/R3K3 w - - 0 1", "4k3/8/2n5/8/Q7/8/8/4K3 w - - 0 1", "8/P6k/8/8/8/8/p6K/8 w - - 0 1",
                "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 0 1"] {
        let g = KernelGame::from_fen("chess", fen).unwrap();
        let s = g.initial();
        out.push((g, s, vec![]));
    }
    for seed in 1..=10u64 {
        let g = KernelGame::new("chess", 518).unwrap();
        let mut s = g.initial();
        let mut rng = Rng::new(seed);
        let mut keys = vec![];
        for _ in 0..(8 + 3 * seed as usize) {
            let ms = g.moves(&s);
            if ms.is_empty() || g.outcome(&s).is_some() { break; }
            keys.push(g.key(&s));
            s = g.apply(&s, &ms[rng.below(ms.len())]);
        }
        if g.outcome(&s).is_none() { out.push((g, s, keys)); }
    }
    out
}

fn same<J: Judge>(judge: &J, prior: bool, record: bool) -> (usize, usize, usize) {
    let (mut compared, mut proved, mut dens) = (0, 0, 0);
    for (g, s, keys) in positions() {
        let hist: Vec<u64> = if record { keys } else { vec![] };
        let p = Params { prior, history: hist.clone(), max_depth: 3, ledger_depth: 4, ..Params::default() };
        let new = decide_params(&g, &s, 600.0, judge, &p).expect("a decision");
        let old = oracle(&g, &s, 600.0, judge, prior, &hist, p.prior_depth, p.ledger_share, p.prior_defects, true, true, true, 3, 4).expect("a decision");
        assert_eq!(new.mv.as_ref().map(|m| g.move_name(&s, m)), Some(old.mv.clone()), "the move at {}", g.fen(&s));
        assert_eq!(new.proved, old.proved, "proved at {}", g.fen(&s));
        assert_eq!(new.trace, old.trace, "the trace at {}", g.fen(&s));
        assert!(new.cost.nodes >= old.nodes, "measured nodes {} < old {} at {}", new.cost.nodes, old.nodes, g.fen(&s));
        compared += 1;
        proved += new.proved as usize;
        dens += new.trace.iter().any(|l| l.starts_with("planning density")) as usize;
    }
    (compared, proved, dens)
}

#[test]
fn the_fixed_controller_plays_what_the_old_decision_played_without_the_prior() {
    let (n, proved, dens) = same(&Count, false, true);
    println!("OBS 78.12 equivalence, prior off, count judge: {n} positions identical, {proved} proved, {dens} with the density prior");
    assert!(n >= 10 && proved >= 1 && dens == 0);
}

#[test]
fn the_fixed_controller_plays_what_the_old_decision_played_with_the_prior() {
    let (n, proved, dens) = same(&Count, true, true);
    println!("OBS 78.12 equivalence, prior on, count judge: {n} positions identical, {proved} proved, {dens} with the density prior");
    assert!(n >= 10 && proved >= 1 && dens >= 1);
}

#[test]
fn the_fixed_controller_plays_what_the_old_decision_played_with_a_valuation_and_no_record() {
    let w = Worth { name: "test".into(), table: [("q".to_string(), 9), ("r".to_string(), 5), ("b".to_string(), 3), ("n".to_string(), 3), ("p".to_string(), 1)].into_iter().collect(), ours_first: true };
    let (n, proved, dens) = same(&w, true, false);
    println!("OBS 78.12 equivalence, prior on, valuation judge, no record: {n} positions identical, {proved} proved, {dens} with the density prior");
    assert!(n >= 10 && proved >= 1 && dens >= 1);
}
