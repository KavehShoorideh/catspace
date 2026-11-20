//! Plan 79.12 PARTS A and B (SYSTEM_SPEC 8 the rule for choosing a move and the wall, 7 "Calculation and estimate, decided together" points 4 to 6; piece: the wake's last layer): UNIT TESTS first (the pure layer, the reason that names what separated the move from the search's first, the estimates of the equal moves
//! scored whatever the clock says), then the INTEGRATION of the layer in the wake on constructed positions (Fixed is the wake as it is; Explore and Weights choose only among the moves held equal; the old player is untouched). No puzzle, no stored game, no match.
use cs_decision::chooser::{apply, inputs_of, AfterFacts, ChooserSpec, FixFacts, MoveFacts, Mode, INPUTS};
use cs_decision::pick::{scores_must, Dir, Reason, Reasons, Side};
use cs_decision::rule::{why_against_first, why_of, Chosen, Decided, FirstFacts, Kept, WhyIn};
use cs_decision::threat_verdict::{Record, VerdictKind};
use cs_decision::wake::{decide, Pieces, WakeInputs, WakeParams};
use cs_game::kernel::KernelGame;
use cs_game::placed::{Kind, Placed};
use cs_game::Game;
use cs_tactics::ledger::{Search, Worth};
use cs_tactics::sheet::base_graph::Static;
use cs_tactics::sheet::patterns::Board;
use cs_tactics::sheet::Names;
use std::collections::HashMap;

// ------------------------------------------------------------------------------------------------------------------------------ B: the pure layer

fn facts() -> MoveFacts { MoveFacts { is_first: true, first_held: true, answers_threat: true, threat_stake: 475.0, fix: Some(FixFacts::new(1945.0, 2, 3, Some(0.4))), reasons_score: Some(0.25), after: Some(AfterFacts { before: Some(4), after: 2, took_out: 1, put_in: 0 }), taking: Some(1216.0), gain: None }
}
fn at(name: &str) -> usize { INPUTS.iter().position(|k| *k == name).unwrap() }

#[test]
fn b_a_row_is_every_named_input_finite_in_the_order_of_the_names() {
    let r = inputs_of(&facts());
    assert_eq!(r.len(), INPUTS.len());
    assert!(r.iter().all(|x| x.is_finite()));
    assert_eq!(r[at("is_first")], 1.0);
    assert_eq!(r[at("first_held")], 1.0);
    assert_eq!(r[at("answers_threat_stake")], 475.0);
    assert_eq!((r[at("fix_stake")], r[at("fix_in_way")], r[at("fix_top_in_way")], r[at("fix_chance")]), (1945.0, 2.0, 3.0, 0.4));
    assert_eq!((r[at("their_in_way_after")], r[at("their_in_way_change")], r[at("their_took_out")], r[at("their_put_in")]), (2.0, -2.0, 1.0, 0.0));
    assert_eq!((r[at("is_taking")], r[at("taking_stake")]), (1.0, 1216.0));
    // a stake that is the goal kind is a flag, never an infinite number
    let g = MoveFacts { threat_stake: f64::INFINITY, fix: Some(FixFacts::new(f64::INFINITY, 1, 1, None)), taking: Some(f64::INFINITY), ..facts() };
    let r = inputs_of(&g);
    assert!(r.iter().all(|x| x.is_finite()));
    assert_eq!((r[at("answers_end_threat")], r[at("answers_threat_stake")], r[at("fix_goal_kind")], r[at("fix_stake")], r[at("taking_goal_kind")], r[at("taking_stake")]), (1.0, 0.0, 1.0, 0.0, 1.0, 0.0));
    // what is not there is nought
    let r = inputs_of(&MoveFacts::default());
    assert!(r.iter().all(|x| *x == 0.0));
    // no chain stood before: the change is nought, not the whole count
    let r = inputs_of(&MoveFacts { after: Some(AfterFacts { before: None, after: 3, took_out: 0, put_in: 1 }), ..MoveFacts::default() });
    assert_eq!((r[at("their_in_way_after")], r[at("their_in_way_change")]), (3.0, 0.0));
}

#[test]
fn b_a_weight_is_by_name_and_a_name_that_is_no_input_is_an_error() {
    assert!(ChooserSpec::new(Mode::Weights, &[("no_such_input".to_string(), 1.0)], 0).is_err());
    let s = ChooserSpec::new(Mode::Weights, &[("reasons_score".to_string(), 2.0), ("is_taking".to_string(), -1.0)], 0).unwrap();
    let r = inputs_of(&facts());
    assert!((s.sum(&r) - (2.0 * 0.25 - 1.0)).abs() < 1e-12);
}

fn rows() -> Vec<Vec<f64>> {
    [0.1, 0.9, 0.5].iter().enumerate().map(|(k, x)| inputs_of(&MoveFacts { is_first: k == 0, first_held: true, reasons_score: Some(*x), ..MoveFacts::default() })).collect()
}

#[test]
fn b_fixed_returns_the_rules_pick_untouched_with_or_without_weights() {
    let none = ChooserSpec::default();
    assert_eq!(none.mode, Mode::Fixed);
    for fixed in [Some(0), Some(2), None] {
        let a = apply(&none, &rows(), fixed);
        assert_eq!(a.index, fixed.unwrap_or(0));
        assert_eq!(a.propensity, 1.0);
        assert!(a.sums.iter().all(|x| *x == 0.0));
    }
    let w = ChooserSpec::new(Mode::Fixed, &[("reasons_score".to_string(), 1.0)], 7).unwrap();
    let a = apply(&w, &rows(), Some(0));
    assert_eq!(a.index, 0, "weights given, mode Fixed: the rule still decides");
    assert!((a.sums[1] - 0.9).abs() < 1e-12, "the sums are still returned for the log");
}

#[test]
fn b_weights_play_the_largest_sum_and_a_tie_goes_to_the_rules_pick() {
    let w = ChooserSpec::new(Mode::Weights, &[("reasons_score".to_string(), 1.0)], 0).unwrap();
    assert_eq!(apply(&w, &rows(), Some(0)).index, 1);
    let neg = ChooserSpec::new(Mode::Weights, &[("reasons_score".to_string(), -1.0)], 0).unwrap();
    assert_eq!(apply(&neg, &rows(), Some(2)).index, 0);
    // all sums equal: the rule's pick, else the first
    let zero = ChooserSpec::new(Mode::Weights, &[], 0).unwrap();
    assert_eq!(apply(&zero, &rows(), Some(2)).index, 2);
    assert_eq!(apply(&zero, &rows(), None).index, 0);
    // fewer than two candidates: nothing to choose
    assert_eq!(apply(&w, &rows()[..1], Some(0)).index, 0);
}

#[test]
fn b_explore_draws_uniformly_from_the_seed_and_logs_the_propensity() {
    let n = 4;
    let rs: Vec<Vec<f64>> = (0..n).map(|_| inputs_of(&MoveFacts::default())).collect();
    let mut seen = vec![0usize; n];
    for seed in 0..4000u64 {
        let a = apply(&ChooserSpec { mode: Mode::Explore, weights: vec![0.0; INPUTS.len()], seed }, &rs, Some(0));
        assert!((a.propensity - 0.25).abs() < 1e-12);
        seen[a.index] += 1;
        assert_eq!(a.index, apply(&ChooserSpec { mode: Mode::Explore, weights: vec![0.0; INPUTS.len()], seed }, &rs, Some(3)).index, "a seed gives one draw whatever the rule's pick");
    }
    for c in &seen { assert!(*c > 800 && *c < 1200, "uniform within sampling: {seen:?}"); }
}

// ------------------------------------------------------------------------------------------------------------------------------ A: the reason names what separated the two

fn chosen(answers: bool, fixes: bool, estimate: bool) -> Chosen { Chosen { mv: "m".into(), answers_their_threat: answers, fixes_our_goal: fixes, why: String::new(), left_after_threat: 1, left_after_goal: if estimate { 2 } else { 1 }, by_estimate: estimate } }
fn first(held: Option<bool>, answers: bool, fixes: bool, own_gain: bool) -> FirstFacts { FirstFacts { name: "f".into(), held, lowered_by: None, answers, fixes, own_gain } }
fn kept(net: i64) -> Kept { Kept { name: "m".into(), verdict: VerdictKind::Confirmed, net, line: vec!["m".into()], ended: "the settled end".into() } }

#[test]
fn a_the_equal_reading_put_the_first_lower_is_said_first_and_by_how_much() {
    let c = chosen(true, true, false);
    let w = WhyIn { chosen: &c, value: 0, depth: 6, own: None, standing: Some("x".into()), answers: true, serves: vec!["goal on y (bears)".into()] };
    let (d, t) = why_against_first(&w, &FirstFacts { lowered_by: Some(1216), ..first(Some(false), false, false, false) });
    assert_eq!(d, Decided::FirstLowered);
    assert!(t.contains("the equal reading put the search's first move lower") && t.contains("1216 below the best"), "{t}");
    let (d, t) = why_against_first(&w, &first(Some(false), true, true, false));
    assert_eq!(d, Decided::FirstLowered);
    assert!(t.contains("below the best at that reading") && !t.contains("answers"), "{t}");
}

#[test]
fn a_only_the_move_played_answers_or_fixes_or_wins_else_the_estimate_is_named() {
    let k = kept(300);
    let c = chosen(true, true, false);
    let own = WhyIn { chosen: &c, value: 0, depth: 6, own: Some(&k), standing: Some("x".into()), answers: true, serves: vec!["goal on y (bears)".into()] };
    let (d, t) = why_against_first(&own, &first(Some(true), false, false, false));
    assert_eq!(d, Decided::OwnGain);
    assert!(t.starts_with("only its own line wins") && t.contains("net 300"), "{t}");
    // the first's line wins too: the own line does not separate them; the answer does
    let (d, t) = why_against_first(&own, &first(Some(true), false, false, true));
    assert_eq!(d, Decided::AnswersThreat);
    assert!(t.starts_with("only it answers the threat of theirs on x"), "{t}");
    // both answer: the fix separates, if only the played one fixes
    let ans = WhyIn { own: None, ..own.clone() };
    let (d, t) = why_against_first(&ans, &first(Some(true), true, false, false));
    assert_eq!(d, Decided::FixesGoal);
    assert!(t.starts_with("only it fixes the goal on y"), "{t}");
    // both answer and both fix: the estimate chose, and it says so and says what they share
    let c2 = chosen(true, true, true);
    let est = WhyIn { chosen: &c2, own: None, ..own.clone() };
    let (d, t) = why_against_first(&est, &first(Some(true), true, true, false));
    assert_eq!(d, Decided::Estimate);
    assert!(t.contains("nothing separates it from the search's first move but the estimates") && t.contains("both answer the threat") && t.contains("an estimate, never a value"), "{t}");
    // no estimate either: said as such
    let c3 = chosen(false, false, false);
    let (d, t) = why_against_first(&WhyIn { chosen: &c3, own: None, standing: None, answers: false, serves: vec![], ..own.clone() }, &first(Some(true), false, false, false));
    assert_eq!(d, Decided::SearchOnly);
    assert!(t.starts_with("nothing separates it"), "{t}");
}

#[test]
fn a_a_forced_end_is_said_as_before_and_the_move_that_is_the_first_keeps_the_old_reason() {
    let c = chosen(true, true, false);
    let w = WhyIn { chosen: &c, value: cs_tactics::ledger::WIN - 3, depth: 5, own: None, standing: None, answers: false, serves: vec![] };
    assert_eq!(why_against_first(&w, &first(Some(false), false, false, false)), why_of(&w));
}

// ------------------------------------------------------------------------------------------------------------------------------ A: the estimates of the moves held equal are scored first, whatever the clock

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
fn judge() -> Worth { Worth { name: "human".into(), table: table(), ours_first: false } }
fn kind(g: &KernelGame, name: &str) -> Kind { g.kind_names().iter().position(|k| k == name).unwrap() as Kind }
fn state(fen: &str) -> <KernelGame as Game>::State { KernelGame::from_fen("chess", fen).unwrap().initial() }
fn keys_of(st: &Static, g: &KernelGame, s: &<KernelGame as Game>::State) -> Vec<u64> {
    let part = st.position(g, s);
    let b = Board::new(st, &part, Some(kind(g, "king")));
    let mut keys: Vec<u64> = vec![];
    for k in 1..=2 { b.connected_sets(k, &mut |its| { for key in b.keys(its, &[0, 1, 2, 3]) { if !keys.contains(&key.hash) { keys.push(key.hash); } } }); }
    keys
}
fn reasons_of(st: &Static, g: &KernelGame, s: &<KernelGame as Game>::State) -> Reasons {
    let keys = keys_of(st, g, s);
    let mut reasons = vec![];
    for c in 0..keys.len() { for side in [Side::Mover, Side::Other] { for dir in [Dir::Up, Dir::Down] { reasons.push(Reason { column: c, side, dir, weight: 0.01 * (1 + c % 5) as f64 * if dir == Dir::Up { 1.0 } else { -0.5 } }); } } }
    Reasons { keys, reasons, max_size: 2, levels: vec![0, 1, 2, 3], tie_digits: 9 }
}
const POSITIONS: [&str; 4] = ["r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 7", "k7/4n3/2p5/8/8/5B2/8/4R2K w - - 0 1", "4k3/8/8/3b4/8/8/4N3/R3K3 w - - 0 1", "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w - - 2 3"];

#[test]
fn a_the_first_moves_of_the_order_given_are_scored_after_the_clock_has_ended_and_the_rest_are_not() {
    let (g, st) = world();
    let s = state(POSITIONS[0]);
    let r = reasons_of(&st, &g, &s);
    let names: Vec<String> = g.moves(&s).iter().map(|m| g.move_name(&s, m)).collect();
    let past = std::time::Instant::now() - std::time::Duration::from_secs(1);
    assert_eq!(scores_must(&st, &g, &s, Some(kind(&g, "king")), &r, &names, Some(past), 0).len(), 0, "the clock ended, nothing is owed: nothing is scored");
    let got = scores_must(&st, &g, &s, Some(kind(&g, "king")), &r, &names, Some(past), 3);
    assert_eq!(got.len(), 3);
    assert_eq!(got.iter().map(|x| x.mv.clone()).collect::<Vec<_>>(), names[..3].to_vec(), "the moves owed are the first of the order, in order");
    let all = scores_must(&st, &g, &s, Some(kind(&g, "king")), &r, &names, None, 0);
    assert_eq!(all.len(), names.len());
    assert_eq!(&all[..3], &got[..], "the same score whether the clock allowed it or the move was owed");
}

// ------------------------------------------------------------------------------------------------------------------------------ B: the layer in the wake

fn whole(g: &KernelGame, st: &Static, s: &<KernelGame as Game>::State, chooser: Option<ChooserSpec>) -> WakeParams {
    WakeParams { seconds: 60.0, depth: 4,
        pieces: Pieces { carried: false, frames: false, threats: true, goals: true, two_threat_goals: true, graph_reads: false, reasons_pick: true, heads: false, after_move: true, direct_names: true },
        inputs: WakeInputs { record: Record { worth: table(), half: HashMap::new() }, goal: Some(kind(g, "king")), reasons: Some(reasons_of(st, g, s)), budget: 4, chooser, ..WakeInputs::default() }, ..WakeParams::default() }
}

#[test]
fn b_with_no_chooser_and_with_mode_fixed_the_wake_plays_the_same_move_and_the_old_player_is_untouched() {
    let (g, st) = world();
    for fen in POSITIONS {
        let s = state(fen);
        let j = judge();
        let a = decide(&st, &g, &s, &[], &j, &whole(&g, &st, &s, None));
        let b = decide(&st, &g, &s, &[], &j, &whole(&g, &st, &s, Some(ChooserSpec::default())));
        let w = decide(&st, &g, &s, &[], &j, &whole(&g, &st, &s, Some(ChooserSpec::new(Mode::Fixed, &[("reasons_score".to_string(), 5.0)], 3).unwrap())));
        assert_eq!((a.mv.clone(), a.value, a.depth, a.equal.clone()), (b.mv.clone(), b.value, b.depth, b.equal.clone()), "{fen}");
        assert_eq!(a.mv, w.mv, "{fen}: weights given, mode Fixed: the rule decides");
        assert!(a.log.chooser.is_none());
        if a.equal.len() >= 2 {
            let c = b.log.chooser.as_ref().expect("a row for every candidate");
            assert!(!c.applied && c.played == a.mv && c.fixed_pick == a.mv);
            assert!(c.candidates.iter().all(|x| x.1.len() == INPUTS.len()));
        }
        // the old player is the search alone, whatever chooser is given
        let me = g.mover(&s).unwrap();
        let mut plain = Search::with_fuse(&g, &j, me, 50_000_000);
        let want = plain.run(&s, 5);
        let p = WakeParams { seconds: 60.0, depth: 5, pieces: Pieces::old_player(), inputs: whole(&g, &st, &s, Some(ChooserSpec { mode: Mode::Explore, ..Default::default() })).inputs, ..WakeParams::default() };
        let got = decide(&st, &g, &s, &[], &j, &p);
        assert_eq!((got.mv, got.value, got.depth), (want.first, want.value, want.depth));
        assert!(got.log.chooser.is_none());
    }
}

#[test]
fn b_explore_and_weights_choose_only_among_the_moves_the_search_holds_equal_and_the_value_is_untouched() {
    let (g, st) = world();
    let mut varied = 0;
    for fen in POSITIONS {
        let s = state(fen);
        let j = judge();
        let base = decide(&st, &g, &s, &[], &j, &whole(&g, &st, &s, None));
        let mut played = std::collections::HashSet::new();
        for seed in 0..24u64 {
            let d = decide(&st, &g, &s, &[], &j, &whole(&g, &st, &s, Some(ChooserSpec { mode: Mode::Explore, weights: vec![0.0; INPUTS.len()], seed })));
            let mv = d.mv.clone().unwrap();
            assert_eq!(d.value, base.value, "the search's value does not depend on the last layer");
            assert!(d.equal.contains(&mv), "{fen}: {mv} is not among the moves held equal {:?}", d.equal);
            if base.log.first_held == Some(false) && base.equal.len() > 1 { assert_ne!(Some(&mv), base.equal.first(), "a first move the equal reading dropped is no candidate"); }
            if let Some(c) = &d.log.chooser { assert!(c.applied && (c.propensity - 1.0 / c.candidates.len() as f64).abs() < 1e-12 && d.log.why.starts_with("explore")); }
            played.insert(mv);
        }
        if base.equal.len() >= 2 { assert!(played.len() >= 2, "{fen}: explore never varied the move among {:?}", base.equal); varied += 1; }
        // weights: the largest weighted sum of the logged rows plays
        let wt = ChooserSpec::new(Mode::Weights, &[("reasons_score".to_string(), 1.0), ("is_first".to_string(), -0.001)], 0).unwrap();
        let d = decide(&st, &g, &s, &[], &j, &whole(&g, &st, &s, Some(wt)));
        if let Some(c) = d.log.chooser.as_ref().filter(|c| c.candidates.len() >= 2) {
            let best = c.candidates.iter().cloned().fold(None, |a: Option<(String, Vec<f64>, f64)>, x| match a { Some(y) if y.2 >= x.2 => Some(y), _ => Some(x) }).unwrap();
            assert!(c.candidates.iter().all(|x| x.2 <= best.2 + 1e-12));
            assert_eq!(c.played.as_ref(), Some(&best.0).or(c.fixed_pick.as_ref()).filter(|_| true), "{fen}");
            assert!(d.equal.contains(&c.played.clone().unwrap()));
            assert!(d.log.why.starts_with("the chooser"));
        }
    }
    assert!(varied >= 1, "at least one constructed position has two moves held equal");
}

#[test]
fn a_the_log_carries_the_wall_of_each_step_and_the_first_kept_at_the_equal_reading() {
    let (g, st) = world();
    let s = state(POSITIONS[0]);
    let d = decide(&st, &g, &s, &[], &judge(), &whole(&g, &st, &s, None));
    assert!(d.log.step_seconds.iter().any(|x| x.0 == 5) && d.log.step_seconds.iter().any(|x| x.0 == 3) && d.log.step_seconds.iter().any(|x| x.0 == 0));
    assert!(d.log.step_seconds.iter().all(|x| x.2 >= 0.0));
    let j = d.to_json();
    assert!(j["log"]["step_seconds"].as_array().is_some_and(|a| !a.is_empty()));
    assert!(j["log"].get("first_held").is_some() && j["log"].get("chooser").is_some());
}
