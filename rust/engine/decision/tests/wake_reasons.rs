//! Plan 79.09 PART C (THE REASONS, SYSTEM_SPEC 8 "the wake gives reasons", 4 the four ways a line ends; piece: the wake): UNIT TESTS of the reason printed for the move: the claim that decided it. A line that wins by its own reading says so and names the goal it serves; "answers a threat" only where a threat of theirs stands and the
//! move's own line is no gain; the end of the game forced is said as such; no claim without the reason it ended; a goal is the subgoal of the moment only if a named move serves it. The pure part (`why_of`) on built inputs, the rest on the wake over constructed positions. No puzzle, no stored game, no match.
use cs_decision::rule::{choose, why_of, Chosen, Decided, Kept, WhyIn};
use cs_decision::threat_verdict::{Record, VerdictKind};
use cs_decision::wake::{decide, Pieces, WakeInputs, WakeParams};
use cs_game::kernel::KernelGame;
use cs_game::placed::{Kind, Placed};
use cs_game::Game;
use cs_tactics::ledger::{Worth, WIN};
use cs_tactics::sheet::base_graph::Static;
use cs_tactics::sheet::Names;
use std::collections::HashMap;

fn world() -> (KernelGame, Static) {
    let g = KernelGame::new("chess", 518).unwrap();
    let text = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../../rules/chess.gdl")).unwrap();
    let steps = g.steps(&text, &["diagonal_of".to_string(), "step".to_string()], &[("square".to_string(), vec!["file".to_string(), "rank".to_string()])]).unwrap();
    let mut st = Static::new(&steps, &Names::new(g.place_names(), g.kind_names(), g.roles())).unwrap();
    for fen in ["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1", "k7/4n3/2p5/8/8/5B2/8/4R2K w - - 0 1", "6k1/5ppp/8/8/8/8/8/R3K3 w - - 0 1"] {
        let s = KernelGame::from_fen("chess", fen).unwrap().initial();
        let p = st.position(&g, &s);
        st.learn(&p);
    }
    (g, st)
}
fn table() -> HashMap<String, i64> { [("p", 475), ("n", 1216), ("b", 1301), ("r", 1945), ("q", 3583)].iter().map(|(k, v)| (k.to_string(), *v)).collect() }
fn kind(g: &KernelGame, name: &str) -> Kind { g.kind_names().iter().position(|k| k == name).unwrap() as Kind }
fn state(fen: &str) -> <KernelGame as Game>::State { KernelGame::from_fen("chess", fen).unwrap().initial() }

fn base(c: &Chosen) -> WhyIn<'_> { WhyIn { chosen: c, value: 0, depth: 6, own: None, standing: None, answers: false, serves: vec![] } }
fn chosen(answers: bool, fixes: bool, estimate: bool) -> Chosen { Chosen { mv: "m".into(), answers_their_threat: answers, fixes_our_goal: fixes, why: String::new(), left_after_threat: 1, left_after_goal: if estimate { 2 } else { 1 }, by_estimate: estimate } }
fn kept(verdict: VerdictKind, net: i64) -> Kept { Kept { name: "m".into(), verdict, net, line: vec!["m".into()], ended: "the settled end: no capture is left to either side after 4 plies".into() } }

#[test]
fn c_the_end_of_the_game_forced_is_said_as_such_and_is_never_called_an_answer() {
    let c = chosen(true, true, false);
    let w = WhyIn { chosen: &c, value: WIN - 3, depth: 5, own: None, standing: Some("the end of the game forced against us".into()), answers: true, serves: vec!["goal on x (bears)".into()] };
    let (d, t) = why_of(&w);
    assert_eq!(d, Decided::ForcedEnd);
    assert!(t.contains("the end of the game is forced in its line") && t.contains("within 5 plies") && t.contains("it serves the goal on x"), "{t}");
    assert!(!t.contains("answers"), "{t}");
    let (d, t) = why_of(&WhyIn { value: -(WIN - 3), ..w.clone() });
    assert_eq!(d, Decided::ForcedEnd);
    assert!(t.contains("forced against us"), "{t}");
}

#[test]
fn c_a_line_that_wins_by_its_own_reading_says_so_and_names_the_goal_it_serves_even_when_it_is_also_in_a_list_of_answers() {
    let c = chosen(true, true, false);
    let k = kept(VerdictKind::Confirmed, 420);
    let w = WhyIn { chosen: &c, value: 100, depth: 6, own: Some(&k), standing: Some("white pawn on a2".into()), answers: true, serves: vec!["goal on black pawn on b2 (bears)".into(), "goal on y (bears)".into()] };
    let (d, t) = why_of(&w);
    assert_eq!(d, Decided::OwnGain);
    assert!(t.contains("wins by its own reading") && t.contains("net 420") && t.contains("it serves the goal on black pawn on b2"), "{t}");
    assert!(!t.contains("answers"), "{t}");
}

#[test]
fn c_answers_a_threat_only_where_one_stands_and_the_own_line_is_no_gain() {
    let c = chosen(true, true, false);
    let k = kept(VerdictKind::Exchange, 0);
    let standing = WhyIn { chosen: &c, value: 0, depth: 6, own: Some(&k), standing: Some("white pawn on a2".into()), answers: true, serves: vec!["goal on z (bears)".into()] };
    let (d, t) = why_of(&standing);
    assert_eq!(d, Decided::AnswersThreat);
    assert!(t.contains("answers the threat of theirs on white pawn on a2"), "{t}");
    // no threat of theirs stands: the move is not an answer, whatever lists it is in
    let (d, t) = why_of(&WhyIn { standing: None, ..standing.clone() });
    assert_eq!(d, Decided::FixesGoal);
    assert!(!t.contains("answers the threat") && t.contains("a fix of the goal on z"), "{t}");
    // a threat stands but the move is not among its answers
    let (d, _) = why_of(&WhyIn { answers: false, ..standing.clone() });
    assert_eq!(d, Decided::FixesGoal);
}

#[test]
fn c_the_estimates_and_the_search_alone_are_said_to_be_what_they_are() {
    let (c, none) = (chosen(false, false, true), chosen(false, false, false));
    let (d, t) = why_of(&base(&c));
    assert_eq!(d, Decided::Estimate);
    assert!(t.contains("an estimate, never a value"), "{t}");
    let (d, t) = why_of(&base(&none));
    assert_eq!(d, Decided::SearchOnly);
    assert!(t.contains("search's own first move"), "{t}");
    // the ladder itself is unchanged: an answer is taken before a fix
    let eq: Vec<String> = ["a", "b"].iter().map(|s| s.to_string()).collect();
    let r = choose(&eq, &["b".to_string()], &["a".to_string()], &|_: &[String]| None).unwrap();
    assert!(r.answers_their_threat && r.mv == "b");
}

fn whole(g: &KernelGame, seconds: f64) -> WakeParams {
    WakeParams { seconds, depth: 64, pieces: Pieces { carried: false, frames: false, threats: true, goals: true, two_threat_goals: true, graph_reads: false, reasons_pick: false, heads: false },
        inputs: WakeInputs { record: Record { worth: table(), half: HashMap::new() }, goal: Some(kind(g, "king")), pre_share: Some(0.5), ..WakeInputs::default() }, ..WakeParams::default() }
}

#[test]
fn c_the_wake_gives_every_claim_the_reason_it_ended_and_a_forced_end_is_labelled_as_one() {
    let (g, st) = world();
    let j = Worth { name: "human".into(), table: table(), ours_first: false };
    for fen in ["6k1/5ppp/8/8/8/8/8/R3K3 w - - 0 1", "k7/4n3/2p5/8/8/5B2/8/4R2K w - - 0 1", "4k3/8/8/3b4/8/8/4N3/R3K3 w - - 0 1", "r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 7"] {
        let s = state(fen);
        let d = decide(&st, &g, &s, &[], &j, &whole(&g, 1.0));
        assert!(d.mv.is_some());
        assert!(!d.chain.is_empty() && d.chain[0].line.first() == d.mv.as_ref(), "the first claim is the search's line of the move played: {:?} {:?}", d.chain[0].line, d.mv);
        for c in &d.chain { assert!(!c.ended.trim().is_empty(), "a claim with no reason it ended: {c:?}"); }
        assert!(!d.log.why.is_empty());
        if d.value.abs() >= WIN - 1000 {
            assert!(d.chain[0].by.contains("the end of the game forced") && d.chain[0].ended.contains("the game itself ends"), "{:?}", d.chain[0]);
            assert!(d.log.why.contains("end of the game"), "{}", d.log.why);
            assert!(!d.log.why.starts_with("answers"), "{}", d.log.why);
        }
        // "answers" is said only where a threat of theirs stands
        if d.log.why.starts_with("answers the threat") { assert!(d.log.their_threat.is_some(), "{}", d.log.why); }
        // a move whose own line wins is not called an answer
        if let Some(m) = d.log.moves.iter().find(|m| Some(&m.mv) == d.mv.as_ref()) {
            if m.own_read.as_ref().is_some_and(|r| r.0 == "Confirmed" && r.1 > 0) { assert!(!d.log.why.starts_with("answers"), "{}", d.log.why); }
        }
    }
}

#[test]
fn c_a_goal_is_the_subgoal_of_the_moment_only_if_a_named_move_serves_it() {
    let (g, st) = world();
    let j = Worth { name: "human".into(), table: table(), ours_first: false };
    for fen in ["k7/4n3/2p5/8/8/5B2/8/4R2K w - - 0 1", "4k3/8/8/3b4/8/8/4N3/R3K3 w - - 0 1", "r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 7"] {
        let s = state(fen);
        let d = decide(&st, &g, &s, &[], &j, &whole(&g, 1.0));
        let named: Vec<&String> = d.named_mine.iter().collect();
        for x in d.log.goals.iter().filter(|x| x.chosen) {
            let fixes: Vec<&String> = x.fixes.iter().flat_map(|f| f.1.iter()).collect();
            assert!(fixes.iter().any(|m| named.contains(m)) || x.fixes.iter().any(|f| f.1.len() >= 8), "the goal on {} is printed as the subgoal and no named move serves it ({named:?})", x.target);
        }
        assert_eq!(d.log.chosen_goal.is_some(), d.log.goals.iter().any(|x| x.chosen));
        if d.log.chosen_goal.is_none() && !d.log.goals.is_empty() && d.log.goals.iter().any(|x| x.side == "white" && !x.fixes.is_empty()) { assert!(d.named_mine.is_empty() || d.log.notes.iter().any(|n| n.contains("no goal of ours")) || d.log.chosen_goal.is_some()); }
    }
}
