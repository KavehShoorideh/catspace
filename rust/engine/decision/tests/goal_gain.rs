//! Plan 79.13 PART C (SYSTEM_SPEC 7 "Calculation and estimate, decided together" point 2 the estimate per goal, "among equal moves we optimize for the prediction ... whichever gets us closer to our goals"; 8 the rule for choosing a move and the wall; piece: the estimate per goal): UNIT TESTS first
//! (the key of a goal, the rank of a stake, the table's back-off and floor, the value of a side's chains, their reading after each move of ours), then the INTEGRATION in the wake on constructed positions (a table given changes no value and no move in mode Fixed; the gain inputs are in the rows; the gain rule picks
//! only among the moves held equal, among those that answer the standing threat where any does). No puzzle, no stored game, no match.
use cs_decision::after::{goals_after, their_goals, their_goals_after};
use cs_decision::chooser::{apply, inputs_of, ChooserSpec, GainFacts, MoveFacts, Mode, INPUTS};
use cs_decision::gain::{chain_value, ChainValue};
use cs_decision::goal_table::{rank_of, Field, GoalFacts, GoalTable, Level};
use cs_decision::goals::goals;
use cs_decision::rule::{chain_top, Stakes};
use cs_decision::threat_verdict::{Record, VerdictKind};
use cs_decision::threats::{ConcreteList, ThreatRow};
use cs_decision::wake::{decide, Pieces, UnitRef, WakeInputs, WakeParams};
use cs_game::kernel::KernelGame;
use cs_game::placed::{Kind, Placed};
use cs_game::Game;
use cs_tactics::ledger::Worth;
use cs_tactics::sheet::base_graph::Static;
use cs_tactics::sheet::Names;
use std::collections::HashMap;

fn world() -> (KernelGame, Static) {
    let g = KernelGame::new("chess", 518).unwrap();
    let text = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../../rules/chess.gdl")).unwrap();
    let steps = g.steps(&text, &["diagonal_of".to_string(), "step".to_string()], &[("square".to_string(), vec!["file".to_string(), "rank".to_string()])]).unwrap();
    let mut st = Static::new(&steps, &Names::new(g.place_names(), g.kind_names(), g.roles())).unwrap();
    for fen in ["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1", "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w - - 2 3", "r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 7", "6k1/8/8/8/2n5/8/4B3/4K3 w - - 0 1"] {
        let s = KernelGame::from_fen("chess", fen).unwrap().initial();
        let p = st.position(&g, &s);
        st.learn(&p);
    }
    (g, st)
}
fn table() -> HashMap<String, i64> { [("p", 475), ("n", 1216), ("b", 1301), ("r", 1945), ("q", 3583)].iter().map(|(k, v)| (k.to_string(), *v)).collect() }
fn record() -> Record { Record { worth: table(), half: HashMap::new() } }
fn judge() -> Worth { Worth { name: "human".into(), table: table(), ours_first: false } }
fn kind(g: &KernelGame, name: &str) -> Kind { g.kind_names().iter().position(|k| k == name).unwrap() as Kind }
fn role(g: &KernelGame, name: &str) -> usize { g.roles().iter().position(|r| r == name).unwrap() }
fn at(g: &KernelGame, name: &str) -> u16 { g.place_names().iter().position(|p| p == name).unwrap() as u16 }
fn state(fen: &str) -> <KernelGame as Game>::State { KernelGame::from_fen("chess", fen).unwrap().initial() }
fn unit(g: &KernelGame, r: &str, k: &str, p: &str) -> UnitRef { UnitRef { owner: role(g, r), kind: kind(g, k), at: at(g, p) } }
const POSITIONS: [&str; 4] = ["r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 7", "k7/4n3/2p5/8/8/5B2/8/4R2K w - - 0 1", "4k3/8/8/3b4/8/8/4N3/R3K3 w - - 0 1", "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w - - 2 3"];

fn knight_goals() -> (KernelGame, Static, <KernelGame as Game>::State, Vec<cs_decision::goals::Goal>) {
    let (g, st) = world();
    let s = state(POSITIONS[1]);
    let mut l = ConcreteList::default();
    let mut r = ThreatRow::new(role(&g, "black"), Some(unit(&g, "black", "knight", "e7")), "contested", vec![], None);
    r.verdict = VerdictKind::Exchange;
    l.other.push(r);
    let gs = goals(&st, &g, &s, &l, role(&g, "white"), Some(kind(&g, "king")), 1, 8);
    (g, st, s, gs)
}

// ------------------------------------------------------------------------------------------------------------------------------ C: the key, the rank, the table

#[test]
fn c_the_rank_of_a_stake_is_the_count_of_distinct_worths_under_it_and_the_goal_kind_is_above_all() {
    let w = [475, 1216, 1301, 1945, 3583, 1945];
    assert_eq!(rank_of(0.0, &w), 0);
    assert_eq!(rank_of(475.0, &w), 0);
    assert_eq!(rank_of(1216.0, &w), 1);
    assert_eq!(rank_of(1945.0, &w), 3);
    assert_eq!(rank_of(3583.0, &w), 4);
    assert_eq!(rank_of(f64::INFINITY, &w), 5, "the goal kind's stake is above every worth");
}

#[test]
fn c_the_key_of_a_goal_is_its_facts_as_text_in_the_order_of_the_fields_and_names_no_object() {
    let (_g, _st, _s, gs) = knight_goals();
    let gl = gs.iter().find(|x| !x.goal_unit).expect("a goal that is not the goal unit's");
    let f = GoalFacts::of(gl, true, 3);
    let mut want: Vec<&str> = gl.unmet.clone();
    want.sort();
    want.dedup();
    assert_eq!(f.unmet, want.join("+"));
    assert_eq!(f.source, gl.source);
    assert_eq!(f.depth, gl.depth);
    let all = [Field::Source, Field::Unmet, Field::InWay, Field::Depth, Field::Rank, Field::Fix, Field::Mine];
    assert_eq!(f.key(&all), format!("{}|{}|{}|{}|3|{}|1", gl.source, want.join("+"), cs_decision::rule::things_in_the_way(gl), gl.depth, u8::from(gl.fixes.iter().any(|x| !x.moves.is_empty()))));
    assert_eq!(f.key(&[Field::Rank, Field::Mine]), "3|1");
    assert_eq!(GoalFacts::of(gl, false, 3).key(&[Field::Mine]), "0");
    assert_eq!(Field::named("in_way"), Some(Field::InWay));
    assert!(Field::named("a_name_that_is_none").is_none());
}

fn level(fields: &[Field], cells: &[(&str, u32, u32)]) -> Level { Level { fields: fields.to_vec(), cells: cells.iter().map(|(k, n, h)| (k.to_string(), (*n, *h))).collect() } }

#[test]
fn c_the_table_reads_the_first_level_with_a_cell_of_the_floor_then_the_next_then_the_whole_and_smooths() {
    let (_g, _st, _s, gs) = knight_goals();
    let gl = gs.iter().find(|x| !x.goal_unit).unwrap();
    let f = GoalFacts::of(gl, true, 3);
    let full = [Field::Source, Field::Unmet, Field::InWay, Field::Depth, Field::Rank, Field::Fix, Field::Mine];
    let mut t = GoalTable { k: 60, floor: 20, levels: vec![level(&full, &[]), level(&[Field::Rank, Field::Mine], &[]), level(&[Field::Mine], &[])], total: (100, 30) };
    let e = t.chance(&f);
    assert_eq!((e.source, e.n), ("total", 100));
    assert!((e.chance - 31.0 / 102.0).abs() < 1e-12, "(reached + 1) / (n + 2) of the whole");
    t.levels[2].cells.insert("1".into(), (50, 10));
    let e = t.chance(&f);
    assert_eq!((e.source, e.n), ("level2", 50));
    assert!((e.chance - 11.0 / 52.0).abs() < 1e-12);
    t.levels[1].cells.insert("3|1".into(), (19, 19));
    assert_eq!(t.chance(&f).source, "level2", "a cell with fewer than the floor does not speak");
    t.levels[1].cells.insert("3|1".into(), (20, 5));
    let e = t.chance(&f);
    assert_eq!((e.source, e.n, e.key.as_str()), ("level1", 20, "3|1"));
    assert!((e.chance - 6.0 / 22.0).abs() < 1e-12);
    t.levels[0].cells.insert(f.key(&full), (200, 100));
    assert_eq!(t.chance(&f).source, "level0", "the full key speaks first");
}

// ------------------------------------------------------------------------------------------------------------------------------ C: the value of a side's chains

#[test]
fn c_the_value_of_a_sides_chains_is_stake_times_chance_at_each_top_and_the_goal_kind_is_summed_apart() {
    let (g, st, s, gs) = knight_goals();
    let white = role(&g, "white");
    let sk = Stakes::read(&g, &s, &record(), &gs, Some(kind(&g, "king")));
    let v = chain_value(&gs, white, &sk, &|_, _| 0.5);
    let tops: std::collections::BTreeSet<usize> = gs.iter().enumerate().filter(|(_, x)| x.side == white && !x.unmet.is_empty()).map(|(i, _)| chain_top(&gs, i)).collect();
    let (mut fin, mut goal) = (0.0, 0.0);
    for t in &tops { let k = sk.of(gs[*t].target); if k.is_finite() { fin += k * 0.5; } else { goal += 0.5; } }
    assert!(!tops.is_empty(), "white has a goal on the knight");
    assert_eq!(v, ChainValue { finite: fin, goal });
    assert!(v.finite > 0.0 || v.goal > 0.0);
    let two = chain_value(&gs, white, &sk, &|_, _| 1.0);
    assert!((two.finite - 2.0 * v.finite).abs() < 1e-9 && (two.goal - 2.0 * v.goal).abs() < 1e-9, "linear in the chance");
    let none = chain_value(&gs, role(&g, "black"), &sk, &|_, _| 1.0);
    assert_eq!(none, ChainValue::default(), "no goal of the other side is in this list");
    assert_eq!(ChainValue { finite: 5.0, goal: 1.0 }.less(&ChainValue { finite: 2.0, goal: 0.5 }), ChainValue { finite: 3.0, goal: 0.5 });
    let _ = st;
}

// ------------------------------------------------------------------------------------------------------------------------------ C: their goals and ours after each move

#[test]
fn c_ours_are_read_after_each_move_only_where_a_value_is_given_and_what_was_read_of_theirs_is_unchanged() {
    let (g, st) = world();
    for fen in POSITIONS {
        let s = state(fen);
        let order: Vec<usize> = (0..g.moves(&s).len()).collect();
        let k = Some(kind(&g, "king"));
        let plain = their_goals_after(&st, &g, &s, &record(), k, &order, 12, None);
        assert!(plain.before.is_none() && plain.rows.iter().all(|r| r.gain.is_none()) && plain.ours_seconds == 0.0, "{fen}: nothing of ours is read with no value");
        let value = |tg: &cs_decision::after::TheirGoals, side: usize| chain_value(&tg.goals, side, &tg.stakes, &|_, _| 0.25);
        let with = goals_after(&st, &g, &s, &record(), k, &order, 12, None, Some(&value));
        assert_eq!(with.rows.len(), plain.rows.len(), "{fen}");
        for (a, b) in with.rows.iter().zip(&plain.rows) { assert_eq!((&a.mv, a.key, &a.names, &a.fact), (&b.mv, b.key, &b.names, &b.fact), "{fen}: theirs are read as before"); assert!(a.gain.is_some()); }
        let before = with.before.expect("the value before");
        let me = g.mover(&s).unwrap();
        let ours = their_goals(&st, &g, &s, &record(), k, me);
        assert_eq!(before.ours, value(&ours, me), "{fen}: before: ours read at the position as it stands");
        assert_eq!(before.theirs, value(&their_goals(&st, &g, &s, &record(), k, g.other(me)), g.other(me)));
        assert!(with.ours_seconds > 0.0);
    }
}

// ------------------------------------------------------------------------------------------------------------------------------ C: the chooser's new inputs and the gain rule

fn at_in(name: &str) -> usize { INPUTS.iter().position(|k| *k == name).unwrap() }

#[test]
fn c_the_gain_inputs_are_four_named_finite_numbers_nought_where_no_gain_was_read() {
    let r = inputs_of(&MoveFacts { gain: Some(GainFacts { ours: 3.0, theirs: -1.5, ours_goal: 0.25, theirs_goal: 0.0 }), ..MoveFacts::default() });
    assert_eq!((r[at_in("our_gain")], r[at_in("their_gain")], r[at_in("our_goal_gain")], r[at_in("their_goal_gain")]), (3.0, -1.5, 0.25, 0.0));
    assert_eq!(r.len(), INPUTS.len());
    assert!(inputs_of(&MoveFacts::default()).iter().all(|x| *x == 0.0));
    let r = inputs_of(&MoveFacts { gain: Some(GainFacts { ours: f64::INFINITY, ..GainFacts::default() }), ..MoveFacts::default() });
    assert!(r.iter().all(|x| x.is_finite()), "an input is always finite");
}

fn row(answers: bool, o: f64, t: f64, og: f64, tg: f64) -> Vec<f64> { inputs_of(&MoveFacts { answers_threat: answers, gain: Some(GainFacts { ours: o, theirs: t, ours_goal: og, theirs_goal: tg }), ..MoveFacts::default() }) }

#[test]
fn c_the_gain_rule_takes_the_largest_gain_less_theirs_the_goal_kind_first_and_answers_to_a_threat_above_it() {
    let spec = ChooserSpec::new(Mode::Gain, &[], 0).unwrap();
    // finite gains only: 5 - 1 = 4 beats 3 - 0 = 3 and 9 - 8 = 1
    let rows = vec![row(false, 3.0, 0.0, 0.0, 0.0), row(false, 5.0, 1.0, 0.0, 0.0), row(false, 9.0, 8.0, 0.0, 0.0)];
    assert_eq!(apply(&spec, &rows, Some(0)).index, 1);
    // the goal kind's chains come before every finite gain
    let rows = vec![row(false, 100.0, 0.0, 0.0, 0.0), row(false, 0.0, 0.0, 0.1, 0.0)];
    assert_eq!(apply(&spec, &rows, Some(0)).index, 1);
    // an answer to the standing threat is kept above the gain: only the answerers compete
    let rows = vec![row(false, 100.0, 0.0, 0.0, 0.0), row(true, 1.0, 0.0, 0.0, 0.0), row(true, 2.0, 0.0, 0.0, 0.0)];
    assert_eq!(apply(&spec, &rows, Some(0)).index, 2);
    // a tie goes to the rule's pick when it is among the tied, else the first of them
    let rows = vec![row(false, 1.0, 0.0, 0.0, 0.0), row(false, 2.0, 0.0, 0.0, 0.0), row(false, 2.0, 0.0, 0.0, 0.0)];
    assert_eq!(apply(&spec, &rows, Some(2)).index, 2);
    assert_eq!(apply(&spec, &rows, Some(0)).index, 1);
    // one candidate, or none: the one
    assert_eq!(apply(&spec, &rows[..1], None).index, 0);
    assert_eq!(apply(&spec, &[], None).index, 0);
}

// ------------------------------------------------------------------------------------------------------------------------------ C: the whole

fn chance_table() -> GoalTable {
    // the things in the way: the fewer, the likelier; the whole's rate for the rest
    GoalTable { k: 60, floor: 20, levels: vec![level(&[Field::InWay], &[("0", 100, 90), ("1", 100, 60), ("2", 100, 30), ("3", 100, 15), ("4", 100, 8)])], total: (1000, 150) }
}

fn whole(g: &KernelGame, goal_table: Option<GoalTable>, chooser: Option<ChooserSpec>) -> WakeParams {
    WakeParams { seconds: 60.0, depth: 4,
        pieces: Pieces { carried: false, frames: false, threats: true, goals: true, two_threat_goals: true, graph_reads: false, reasons_pick: false, heads: false, after_move: true, direct_names: true },
        inputs: WakeInputs { record: record(), goal: Some(kind(g, "king")), budget: 4, chooser, goal_table, ..WakeInputs::default() }, ..WakeParams::default() }
}

#[test]
fn c_a_table_given_changes_no_value_and_in_mode_fixed_no_move_and_the_log_says_where_the_chance_came_from() {
    let (g, st) = world();
    for fen in POSITIONS {
        let s = state(fen);
        let j = judge();
        let none = decide(&st, &g, &s, &[], &j, &whole(&g, None, Some(ChooserSpec::default())));
        let with = decide(&st, &g, &s, &[], &j, &whole(&g, Some(chance_table()), Some(ChooserSpec::default())));
        assert_eq!((none.mv.clone(), none.value, none.depth, none.equal.clone()), (with.mv.clone(), with.value, with.depth, with.equal.clone()), "{fen}: the table is an input, never a value");
        assert!(none.log.goals.iter().all(|x| x.chance.is_none()), "{fen}: no table, no chance");
        assert!(with.log.goals.iter().all(|x| x.chance.as_ref().is_some_and(|c| c.2.starts_with("level") || c.2 == "total")), "{fen}: every goal carries the new chance");
        if let Some(c) = &with.log.chooser { assert!(c.candidates.iter().all(|x| x.1.len() == INPUTS.len())); }
        // without a table the new inputs are nought
        if let Some(c) = &none.log.chooser { assert!(c.candidates.iter().all(|x| x.1[at_in("our_gain")..].iter().all(|v| *v == 0.0))); }
        // the old table is used where no new table is given
        let old = cs_decision::estimate::ChanceTable { floor: 20, total: (100, 25), ..Default::default() };
        let mut p = whole(&g, None, None);
        p.inputs.chances = Some(old);
        let d = decide(&st, &g, &s, &[], &j, &p);
        assert!(d.log.goals.iter().all(|x| x.chance.as_ref().is_some_and(|c| c.2 == "total")), "{fen}: the old table speaks where no new one is given");
    }
}

#[test]
fn c_the_gain_is_in_the_rows_of_the_candidates_and_the_gain_rule_plays_only_among_the_moves_held_equal_with_the_value_untouched() {
    let (g, st) = world();
    let mut read = 0;
    for fen in POSITIONS {
        let s = state(fen);
        let j = judge();
        let base = decide(&st, &g, &s, &[], &j, &whole(&g, Some(chance_table()), None));
        let spec = ChooserSpec::new(Mode::Gain, &[], 0).unwrap();
        let d = decide(&st, &g, &s, &[], &j, &whole(&g, Some(chance_table()), Some(spec)));
        assert_eq!(d.value, base.value, "{fen}: the search's value does not depend on the gain rule");
        if let Some(c) = d.log.chooser.as_ref().filter(|c| c.candidates.len() >= 2) {
            let mv = d.mv.clone().unwrap();
            assert!(d.equal.contains(&mv), "{fen}: {mv} is not held equal in {:?}", d.equal);
            assert_eq!(c.played.as_ref(), Some(&mv));
            let a = at_in("answers_threat");
            let pool: Vec<_> = if c.candidates.iter().any(|x| x.1[a] > 0.5) { c.candidates.iter().filter(|x| x.1[a] > 0.5).collect() } else { c.candidates.iter().collect() };
            let key = |x: &(String, Vec<f64>, f64)| (x.1[at_in("our_goal_gain")] - x.1[at_in("their_goal_gain")], x.1[at_in("our_gain")] - x.1[at_in("their_gain")]);
            let chosen = c.candidates.iter().find(|x| x.0 == mv).unwrap();
            assert!(pool.iter().any(|x| x.0 == mv), "{fen}: the move played is among the answers where there are any");
            for x in &pool { let (k, kc) = (key(x), key(chosen)); assert!(k.0 < kc.0 + 1e-9 && (k.0 < kc.0 - 1e-9 || k.1 <= kc.1 + 1e-9), "{fen}: {} has a larger gain than {mv}", x.0); }
            if c.candidates.iter().any(|x| x.1[at_in("our_gain")..].iter().any(|v| *v != 0.0)) { read += 1; }
            assert!(d.log.why.starts_with("the gain"));
        }
    }
    assert!(read >= 1, "the gain was read and was not nought in at least one position with two moves held equal");
}
