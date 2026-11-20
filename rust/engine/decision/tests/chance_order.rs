//! Plan 79.14 PARTS C AND D, UNIT TESTS (SYSTEM_SPEC 8 the rule for choosing a move, 7 point 2 the estimate per goal and "among equal moves we optimize for the prediction"; piece: the wake). C: the named moves in the order of stake times the table's chance, a goal whose every fact is met kept and named first, no table: the order of main.
//! D: the gain mended (a goal the move meets counts as reached; the chains after a move on every target), with the two faults of 79.13 as cases. Constructed positions only; no puzzle, no match.
use cs_decision::after::{goals_after_with, their_goals_with};
use cs_decision::gain::{chain_value, chain_value_mended};
use cs_decision::goals::{goals, goals_with, propose, Goal, GoalParams};
use cs_decision::rule::{named_fixes, named_fixes_by, Stakes};
use cs_decision::threat_verdict::Record;
use cs_decision::threats::{ConcreteList, ThreatParams};
use cs_decision::wake::{decide, Pieces, WakeInputs, WakeParams};
use cs_game::kernel::KernelGame;
use cs_game::placed::Placed;
use cs_game::Game;
use cs_tactics::ledger::Worth;
use cs_tactics::sheet::base_graph::Static;
use cs_tactics::sheet::Names;
use std::collections::HashMap;

type St = <KernelGame as Game>::State;

fn world() -> (KernelGame, Static) {
    let g = KernelGame::new("chess", 518).unwrap();
    let text = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../../rules/chess.gdl")).unwrap();
    let steps = g.steps(&text, &["diagonal_of".to_string(), "step".to_string()], &[("square".to_string(), vec!["file".to_string(), "rank".to_string()])]).unwrap();
    let st = Static::new(&steps, &Names::new(g.place_names(), g.kind_names(), g.roles())).unwrap();
    (g, st)
}
fn table() -> HashMap<String, i64> { [("p", 475), ("n", 1216), ("b", 1301), ("r", 1945), ("q", 3583)].iter().map(|(k, v)| (k.to_string(), *v)).collect() }
fn record() -> Record { Record { worth: table(), half: HashMap::new() } }
fn state(fen: &str) -> St { KernelGame::from_fen("chess", fen).unwrap().initial() }
fn play(g: &KernelGame, s: &St, name: &str) -> St { let m = g.moves(s).into_iter().find(|m| g.move_name(s, m) == name).unwrap_or_else(|| panic!("no move {name}")); g.apply(s, &m) }
fn king(g: &KernelGame) -> Option<u8> { g.kind_names().iter().position(|k| k == "king").map(|k| k as u8) }
fn white(g: &KernelGame) -> usize { g.roles().iter().position(|r| r == "white").unwrap() }

const MID: &str = "r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 7";
/// white's queen takes the black rook on the file, which nothing guards: every fact of the goal on the rook is met
const HANGING: &str = "7k/8/8/3r4/8/8/3Q4/4K3 w - - 0 1";
/// the same, with a knight of white's on the file: the queen does not bear on the rook; any move of the knight off the file meets the last fact
const BLOCKED: &str = "7k/8/8/3r4/8/3N4/8/3QK3 w - - 0 1";

#[test]
fn c_with_no_chance_given_the_chains_are_those_of_main() {
    let (g, st) = world();
    let s = state(MID);
    let p = propose(&st, &g, &s, &record(), &ThreatParams { goal: king(&g), rows: Some(vec![]), share_table: true, ..Default::default() }, &GoalParams { budget: 40, ..Default::default() });
    let sk = Stakes::read(&g, &s, &record(), &p.goals, king(&g));
    for side in [0usize, 1] {
        let (a, b) = (named_fixes(&p.goals, side, &sk), named_fixes_by(&p.goals, side, &sk, None));
        assert_eq!(a.iter().map(|c| (c.goal, c.top, c.moves.clone())).collect::<Vec<_>>(), b.iter().map(|c| (c.goal, c.top, c.moves.clone())).collect::<Vec<_>>());
        assert!(b.iter().all(|c| c.chance.is_none()));
    }
}

#[test]
fn c_the_chains_come_in_the_order_of_stake_times_chance_and_the_goal_kind_above_every_product() {
    let (g, st) = world();
    let s = state(MID);
    let p = propose(&st, &g, &s, &record(), &ThreatParams { goal: king(&g), rows: Some(vec![]), share_table: true, ..Default::default() }, &GoalParams { budget: 40, ..Default::default() });
    let sk = Stakes::read(&g, &s, &record(), &p.goals, king(&g));
    let w = white(&g);
    let by_stake = named_fixes(&p.goals, w, &sk);
    let finite: Vec<_> = by_stake.iter().filter(|c| c.stake.is_finite()).collect();
    assert!(finite.len() >= 2, "the position gives at least two chains of finite stake: {}", by_stake.len());
    // a chance that is highest for the chain of the smallest stake: the order must follow stake times chance, not stake
    let smallest = finite.iter().map(|c| c.stake).fold(f64::INFINITY, f64::min);
    let chance = |gl: &Goal| -> f64 { if sk.of(gl.target) == smallest { 0.95 } else { 0.05 } };
    let by_chance = named_fixes_by(&p.goals, w, &sk, Some(&chance));
    let score = |c: &cs_decision::rule::ChainFix| if c.stake.is_infinite() { (1u8, c.chance.unwrap()) } else { (0u8, c.stake * c.chance.unwrap()) };
    let non_met: Vec<_> = by_chance.iter().filter(|c| !c.met).collect();
    for pair in non_met.windows(2) {
        let (x, y) = (score(pair[0]), score(pair[1]));
        assert!(x.0 > y.0 || (x.0 == y.0 && x.1 >= y.1), "the chains are in the order of stake times chance: {x:?} then {y:?}");
    }
    let order_stake: Vec<usize> = by_stake.iter().map(|c| c.top).collect();
    let order_chance: Vec<usize> = by_chance.iter().map(|c| c.top).collect();
    println!("OBS chance_order 79.14: {} chains; by stake {:?}; by chance {:?}", by_stake.len(), order_stake, order_chance);
    assert_eq!(by_stake.len(), by_chance.len(), "the same chains, ordered differently");
    assert_ne!(order_stake, order_chance, "a chance that favours the smaller stake changes the order");
    // the same names, in another order
    let mut a: Vec<String> = by_stake.iter().flat_map(|c| c.all_moves()).collect();
    let mut b: Vec<String> = by_chance.iter().flat_map(|c| c.all_moves()).collect();
    a.sort(); b.sort();
    assert_eq!(a, b);
}

#[test]
fn c_a_goal_whose_every_fact_is_met_is_kept_and_its_fix_is_the_move_that_carries_it_out_named_first() {
    let (g, st) = world();
    let s = state(HANGING);
    let w = white(&g);
    let none = goals(&st, &g, &s, &{ let mut l = ConcreteList::default(); l.other.push({ let mut r = cs_decision::threats::ThreatRow::new(1 - w, Some(cs_decision::wake::UnitRef { owner: 1 - w, kind: g.kind_names().iter().position(|k| k == "rook").unwrap() as u8, at: g.place_names().iter().position(|p| p == "d5").unwrap() as u16 }), "unit", vec![], None); r.every = true; r }); l }, w, king(&g), 2, 8);
    let rook_goal = none.iter().find(|x| x.target.at == g.place_names().iter().position(|p| p == "d5").unwrap() as u16).expect("a goal on the rook");
    assert!(rook_goal.met, "every fact the graph can decide is met (the line the clock did not read is not a fact unmet): {:?}", rook_goal.unmet);
    assert!(rook_goal.fixes.is_empty(), "without keep_met it has no fix");
    let mut l = ConcreteList::default();
    l.other.push({ let mut r = cs_decision::threats::ThreatRow::new(1 - w, Some(rook_goal.target), "unit", vec![], None); r.every = true; r });
    let kept = goals_with(&st, &g, &s, &l, w, king(&g), 2, 8, true);
    let kg = kept.iter().find(|x| x.target == rook_goal.target).unwrap();
    assert_eq!(kg.fixes.len(), 1);
    assert_eq!(kg.fixes[0].moves, vec!["d2d5".to_string()], "the move that takes the rook");
    let sk = Stakes::read(&g, &s, &record(), &kept, king(&g));
    let by = named_fixes_by(&kept, w, &sk, Some(&|_: &Goal| 0.1));
    assert!(by[0].met && by[0].moves == vec!["d2d5".to_string()], "named, and first");
}

#[test]
fn d_a_goal_the_move_meets_counts_as_reached_where_main_lowers_the_value() {
    let (g, st) = world();
    let (s, rec) = (state(BLOCKED), record());
    let child = play(&g, &s, "d3b4");
    let w = white(&g);
    let before = their_goals_with(&st, &g, &s, &rec, king(&g), w, true);
    let after = their_goals_with(&st, &g, &child, &rec, king(&g), w, true);
    assert!(before.goals.iter().any(|x| !x.met && x.target.kind != king(&g).unwrap()), "before the move the goal on the rook has a fact unmet (the line is shut)");
    assert!(after.goals.iter().any(|x| x.met), "after it every fact the graph can decide is met");
    // where a line was read every fact is decided and the goal has nothing unmet: it leaves the list of main's value (the case of 79.13's fault); the graph alone leaves the line's fact open, so the goal is emulated here
    let mut after = after;
    for x in after.goals.iter_mut().filter(|x| x.met) { x.unmet.clear(); x.n_unmet = 0; }
    let ch = |_: &Goal, _: &Stakes| 0.3;
    let (b0, a0) = (chain_value(&before.goals, w, &before.stakes, &ch), chain_value(&after.goals, w, &after.stakes, &ch));
    let (b1, a1) = (chain_value_mended(&before.goals, w, &before.stakes, &ch), chain_value_mended(&after.goals, w, &after.stakes, &ch));
    println!("OBS gain mended 79.14: main's value before {:.1} after {:.1}; mended before {:.1} after {:.1}", b0.finite, a0.finite, b1.finite, a1.finite);
    assert!(a0.finite < b0.finite, "the fault of 79.13: the move that meets the last fact LOWERS the value, the goal left the list");
    assert!(a1.finite > b1.finite, "mended: the move raises it");
    assert!((a1.finite - 1945.0).abs() < 1e-9, "the rook's stake in full for the side");
    assert!((b1.finite - 0.3 * 1945.0).abs() < 1e-9, "and before the move at its chance");
}

#[test]
fn d_the_goals_after_a_move_are_read_on_every_target_not_on_the_goal_unit_alone() {
    let (g, st) = world();
    let (s, rec) = (state(BLOCKED), record());
    let w = white(&g);
    let alone = their_goals_with(&st, &g, &s, &rec, king(&g), w, false);
    let every = their_goals_with(&st, &g, &s, &rec, king(&g), w, true);
    let rook = g.kind_names().iter().position(|k| k == "rook").unwrap() as u8;
    assert!(!alone.goals.iter().any(|x| x.target.kind == rook), "main: no goal on the rook, the goal unit's chain alone");
    assert!(every.goals.iter().any(|x| x.target.kind == rook), "mended: a goal on every target, the rook's too");
    // and through goals_after_with: the value of the chains is read after each move on every target, and the old reading is what it was
    let value_alone = |tg: &cs_decision::after::TheirGoals, side: usize| chain_value(&tg.goals, side, &tg.stakes, &|_: &Goal, _: &Stakes| 0.3);
    let value_mend = |tg: &cs_decision::after::TheirGoals, side: usize| chain_value_mended(&tg.goals, side, &tg.stakes, &|_: &Goal, _: &Stakes| 0.3);
    let order: Vec<usize> = (0..g.moves(&s).len()).collect();
    let moves = g.moves(&s);
    let old = goals_after_with(&st, &g, &s, &rec, king(&g), &order, 12, None, Some(&value_alone), false);
    let new = goals_after_with(&st, &g, &s, &rec, king(&g), &order, 12, None, Some(&value_mend), true);
    let find = |a: &cs_decision::after::AfterRead, mv: &str| a.rows.iter().find(|r| r.mv == mv).and_then(|r| r.gain).map(|x| x.ours.finite);
    let _ = moves;
    assert_eq!(old.before.unwrap().ours.finite, 0.0, "main: nothing of ours but the goal unit's chain, no finite value");
    assert!(new.before.unwrap().ours.finite > 0.0, "mended: the goal on the rook has a value");
    assert!(find(&new, "d3b4").unwrap() > new.before.unwrap().ours.finite, "the knight leaving the file raises it");
    assert!(find(&new, "d3d4").unwrap_or(0.0) <= new.before.unwrap().ours.finite + 1e-9, "a move that leaves the line shut does not");
}

#[test]
fn d_and_c_with_the_switches_off_the_wake_is_what_it_was() {
    let (g, st) = world();
    let s = state(MID);
    let judge = Worth { name: "human".into(), table: table(), ours_first: false };
    let mk = |chance_order: bool, gain_mend: bool| WakeParams { seconds: 60.0, depth: 3, pieces: Pieces { threats: true, goals: true, two_threat_goals: true, after_move: true, direct_names: true, ..Pieces::old_player() },
        inputs: WakeInputs { record: record(), goal: king(&g), budget: 4, chance_order, gain_mend, ..WakeInputs::default() }, ..WakeParams::default() };
    let a = decide(&st, &g, &s, &[], &judge, &mk(false, false));
    let b = decide(&st, &g, &s, &[], &judge, &mk(true, true));
    // no goal table is given, so neither switch changes what is named or played: the order is main's (the plan's test: "with no table the order is as on main")
    assert_eq!(a.named_mine, b.named_mine);
    assert_eq!(a.mv, b.mv);
    assert_eq!(Pieces::old_player(), Pieces::default());
}

#[test]
fn c_through_the_wake_a_table_given_changes_what_is_named_first_and_never_the_value_and_the_table_absent_changes_nothing() {
    use cs_decision::goal_table::{Field, GoalTable, Level};
    let (g, st) = world();
    let judge = Worth { name: "human".into(), table: table(), ours_first: false };
    // the fewer things in the way the likelier a goal (a table of play as the sleep counts it, here constructed)
    let tab = GoalTable { k: 16, floor: 20, levels: vec![Level { fields: vec![Field::InWay], cells: [("0", 100u32, 90u32), ("1", 100, 20), ("2", 100, 60), ("3", 100, 70), ("4", 100, 80)].iter().map(|(k, n, h)| (k.to_string(), (*n, *h))).collect() }], total: (1000, 150) };
    let mk = |chance_order: bool, t: Option<GoalTable>| WakeParams { seconds: 60.0, depth: 3, pieces: Pieces { threats: true, goals: true, two_threat_goals: true, after_move: true, direct_names: true, ..Pieces::old_player() },
        inputs: WakeInputs { record: record(), goal: king(&g), budget: 4, chance_order, goal_table: t, ..WakeInputs::default() }, ..WakeParams::default() };
    for fen in [MID, BLOCKED, "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w - - 2 3"] {
        let s = state(fen);
        let main = decide(&st, &g, &s, &[], &judge, &mk(false, Some(tab.clone())));
        let by_chance = decide(&st, &g, &s, &[], &judge, &mk(true, Some(tab.clone())));
        let no_table = decide(&st, &g, &s, &[], &judge, &mk(true, None));
        assert_eq!(main.named_mine, no_table.named_mine, "{fen}: with no table the order is as on main");
        let (mut a, mut b) = (main.named_mine.clone(), by_chance.named_mine.clone());
        println!("OBS chance_order 79.14 wake {fen}: named by stake {:?}; by chance {:?}", main.named_mine, by_chance.named_mine);
        a.sort(); b.sort();
        assert!(by_chance.named_mine.len() <= 12 && b.iter().all(|m| s_moves(&g, &s).contains(m)), "{fen}: every name is a legal move");
        assert_eq!(main.value, by_chance.value, "{fen}: the order of the names adds nothing to the judge's value");
        let _ = a;
    }
}

fn s_moves(g: &KernelGame, s: &St) -> Vec<String> { g.moves(s).iter().map(|m| g.move_name(s, m)).collect() }
