//! PLAN 78.92: the audit of the search's own choice (SYSTEM_SPEC 5, the tactical reasoner; SYSTEM_SPEC 12, the ledger search and the row "the search's equal reading and the line reading's verdict"). Game-free: the
//! game is read through the `Game` trait alone.
//!
//! WHY. 78.85 named the cause of the equal reading's fault: the late-move reduction and the null move are not exact. Both are in the deepening itself, so the move the search plays and its value can carry the same error.
//! This module only MEASURES that; it changes no decision and has no switch of its own.
//!
//! INTERFACE. `choice_audit(g, judge, s, history, run_nodes, max_depth, big, extra)` -> `Option<ChoiceAudit>`. Takes a position, the keys of the positions already met (the rules' draw), a node budget for the deepening
//! as it plays (both prunings on, no clock, so the result repeats), the deepest depth, a node fuse for each reference reading, the number of plies the deeper reading adds and optionally the name of a second root move to be valued on the same tables, and `repair` (the verification of the choice applied first: the move read is the one it leaves). Returns the move the deepening plays (index
//! in the game's list of legal moves, name), the depth D it finished and its value, and for each READING the best value, the best move and the value of the played move at depth D and at depth D + `extra`, each on a fresh table deepened one ply at a time as the search deepens (the nodes are the deepening's so far),
//! with the nodes it cost and whether the fuse cut it (a cut reading has no numbers). The readings differ in one thing each:
//!   plain   the late-move reduction and the null move both off: alpha-beta and the table alone (THE REFERENCE)
//!   no_null the null move off, the reduction on        no_late the reduction off, the null move on        as_played both on (the search itself, fresh table, full window)
//! plus the VERIFICATION a repair would do: the played move alone asked "at least the value it was played at, at depth D?" by a null window on a fresh table, both prunings off (`verify_probe`, its nodes and whether it holds).
//! Called by: rust/engine/py/src/audit_py.rs (the measure's entry) and the tests of this crate.
use crate::ledger::{ChoiceState, Judge, Search};
use cs_game::Game;

/// One reading at one depth: `None` fields where the fuse cut it. `other` is the value of a second root move the caller names (plan 78.92 (c): a move a stored search put first), read on the same table.
#[derive(Debug, Clone)]
pub struct Reading { pub name: &'static str, pub depth: i32, pub best: Option<i64>, pub best_index: Option<usize>, pub best_name: Option<String>, pub played: Option<i64>, pub other: Option<i64>, pub nodes: u64 }

#[derive(Debug, Clone)]
pub struct ChoiceAudit {
    pub depth: i32,
    pub value: i64,
    pub played: usize,
    pub played_name: String,
    pub run_nodes: u64,
    pub readings: Vec<Reading>,
    /// the played move asked again without the two prunings (null window at `value`, depth D): its probe, the nodes, and whether the fuse cut it
    pub verify_probe: Option<i64>,
    pub verify_nodes: u64,
    /// with `repair`: what the verification of the choice did (`ledger::ChoiceState`) and what it cost; `played` is then the move it leaves
    pub repair_state: ChoiceState,
    pub repair_nodes: u64,
}

fn fresh<'a, G: Game, J: Judge>(g: &'a G, judge: &'a J, me: cs_game::Role, history: &[u64], fuse: u64, null_move: bool, late_reduce: bool) -> Search<'a, G, J> {
    let mut x = Search::with_fuse(g, judge, me, fuse);
    x.history = history.iter().copied().collect();
    x.null_move = null_move;
    x.late_reduce = late_reduce;
    x
}

#[allow(clippy::too_many_arguments)]
pub fn choice_audit<G: Game, J: Judge>(g: &G, judge: &J, s: &G::State, history: &[u64], run_nodes: u64, max_depth: i32, big: u64, extra: i32, other: Option<&str>, repair: bool) -> Option<ChoiceAudit> {
    let me = g.mover(s)?;
    let mut a = fresh(g, judge, me, history, run_nodes, true, true);
    let ln = a.run(s, max_depth);
    let mut played = ln.first_index?;
    if ln.depth == 0 { return None; }
    let (d, v) = (ln.depth, ln.value);
    // the switch of the verification of the choice (`Search::verified_choice`): the move that is read below is the one it leaves, and what it cost is kept
    let (mut repair_state, mut repair_nodes) = (ChoiceState::Off, 0);
    if repair {
        a.verify_choice = true;
        a.rebudget(big);
        let n0 = a.nodes;
        let verdict = a.verified_choice(s, d, v, played);
        repair_state = verdict.state;
        repair_nodes = a.nodes - n0;
        played = verdict.index;
    }
    let ms = g.moves(s);
    let played_name = ms.get(played).map(|m| g.move_name(s, m))?;
    let other_i = other.and_then(|n| ms.iter().position(|m| g.move_name(s, m) == n));
    let mut readings = vec![];
    for (name, null_move, late_reduce) in [("plain", false, false), ("no_null", false, true), ("no_late", true, false), ("as_played", true, true)] {
        // one deepening per variant on one fresh table, as `run` deepens (the shallower readings order the deeper ones): the readings at D and at D + extra are taken on the way, their nodes are the deepening's so far
        let mut x = fresh(g, judge, me, history, big, null_move, late_reduce);
        let mut cut = false;
        for depth in 1..=d + extra {
            let rb = if cut { None } else { x.root_best(s, depth) };
            if rb.is_none() { cut = true; }
            if depth != d && depth != d + extra { continue; }
            let (mut best_v, mut best_i, mut played_v, mut other_v) = (None, None, None, None);
            if let Some((v, i)) = rb {
                best_v = Some(v);
                best_i = i;
                played_v = x.move_value(s, played, depth);                                     // (the table is warm: the played move's own value costs little more)
                other_v = match other_i { Some(k) => x.move_value(s, k, depth), None => None };
                if played_v.is_none() { cut = true; best_v = None; best_i = None; other_v = None; }
            }
            readings.push(Reading { name, depth, best: best_v, best_index: best_i, best_name: best_i.and_then(|i| ms.get(i)).map(|m| g.move_name(s, m)), played: played_v, other: other_v, nodes: x.nodes });
        }
    }
    let mut vx = fresh(g, judge, me, history, big, false, false);
    let probe = vx.probes(s, d, v, Some(&[played])).into_iter().next().map(|(_, p)| p);
    Some(ChoiceAudit { depth: d, value: v, played, played_name, run_nodes: ln.nodes, readings, verify_probe: probe, verify_nodes: vx.nodes, repair_state, repair_nodes })
}
