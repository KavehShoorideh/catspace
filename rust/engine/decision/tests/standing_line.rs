//! Plan 79.15 PART C (SYSTEM_SPEC 8 the rule for choosing a move: "evaluate the threats of the opponent" first, the largest threat answered before a goal of ours is fixed; piece: the wake): UNIT TESTS of `goals::standing_line` and of the reason the wake gives, from
//! decisions 3, 11 and 17 of 79.14 (positions as the engine logged them): a line of theirs stands on the mover's goal unit; the move played answers it and the reason says so, before any fix of a goal of ours is named. No puzzle, no match.
use cs_decision::goals::standing_line;
use cs_decision::threat_verdict::Record;
use cs_decision::memory::{Parts, WakeMemory};
use cs_decision::wake::{decide, Pieces, WakeInputs, WakeParams};
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
    (g.clone(), Static::new(&steps, &Names::new(g.place_names(), g.kind_names(), g.roles())).unwrap())
}
fn table() -> HashMap<String, i64> { [("p", 475), ("n", 1216), ("b", 1301), ("r", 1945), ("q", 3583)].iter().map(|(k, v)| (k.to_string(), *v)).collect() }
fn kind(g: &KernelGame, name: &str) -> Kind { g.kind_names().iter().position(|k| k == name).unwrap() as Kind }
fn state(fen: &str) -> <KernelGame as Game>::State { KernelGame::from_fen("chess", fen).unwrap().initial() }
fn whole(g: &KernelGame, seconds: f64) -> WakeParams {
    WakeParams { seconds, depth: 64, pieces: Pieces { carried: false, frames: false, threats: true, goals: true, two_threat_goals: true, graph_reads: false, reasons_pick: false, heads: false, after_move: false, direct_names: false },
        inputs: WakeInputs { memory: WakeMemory::given(Parts { record: Some(Record { worth: table(), half: HashMap::new() }), ..Default::default() }, None), goal: Some(kind(g, "king")), pre_share: Some(0.5), ..WakeInputs::default() }, ..WakeParams::default() }
}

/// (decision of 79.14, the position, the unit that stands borne on, the bearer, a move that answers it)
const CASES: [(u32, &str, &str, &str, &str); 3] = [
    (3, "rn2k1nr/1Q1b2pp/p1p5/p3pp2/P6q/B1PPPP2/6PP/R3KBNR w KQkq - 2 15", "e1", "h4", "g2g3"),
    (11, "6nr/3R3p/pkp5/8/4PQ2/2p1P1P1/4N2P/1q2KB1R w K - 2 30", "e1", "b1", "d7d1"),
    (17, "1n1qkbnr/2p1pppp/8/3p1b2/Q7/2PPPP2/1B4PP/1N2KBNR b Kk - 0 10", "e8", "a4", "f5d7"),
];

#[test]
fn c1_a_line_on_the_goal_unit_is_found_with_its_bearer_and_its_answers() {
    let (g, st) = world();
    for (n, fen, at, bearer, answer) in CASES {
        let s = state(fen);
        let l = standing_line(&st, &g, &s, Some(kind(&g, "king"))).unwrap_or_else(|| panic!("decision {n}: no line found on the goal unit"));
        let names = g.place_names();
        assert_eq!(names[l.unit.at as usize], at, "decision {n}");
        assert_eq!(l.bearers.iter().map(|b| names[b.at as usize].clone()).collect::<Vec<_>>(), vec![bearer.to_string()], "decision {n}");
        assert!(l.answers.contains(&answer.to_string()), "decision {n}: {answer} is not among the answers {:?}", l.answers);
        // every answer is a legal move
        let legal: Vec<String> = g.moves(&s).iter().map(|m| g.move_name(&s, m)).collect();
        assert!(l.answers.iter().all(|a| legal.contains(a)), "decision {n}");
    }
}

#[test]
fn c2_no_line_no_standing_threat() {
    let (g, st) = world();
    for fen in ["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "6k1/5ppp/8/8/8/8/8/R3K3 w - - 0 1"] {
        assert!(standing_line(&st, &g, &state(fen), Some(kind(&g, "king"))).is_none());
    }
    // with no goal kind given there is no goal unit
    assert!(standing_line(&st, &g, &state(CASES[0].1), None).is_none());
}

#[test]
fn c3_the_reason_says_the_move_answers_the_line_before_any_fix_of_a_goal_of_ours() {
    let (g, st) = world();
    let j = Worth { name: "human".into(), table: table(), ours_first: false };
    for (n, fen, _at, _bearer, _answer) in CASES {
        let s = state(fen);
        let d = decide(&st, &g, &s, &[], &j, &whole(&g, 1.0));
        let l = standing_line(&st, &g, &s, Some(kind(&g, "king"))).unwrap();
        let mv = d.mv.clone().unwrap();
        assert!(l.answers.contains(&mv), "decision {n}: the move {mv} does not answer the line");
        assert!(d.log.their_threat.as_deref().is_some_and(|t| t.contains("a line of theirs stands on it")), "decision {n}: {:?}", d.log.their_threat);
        // the reason: the end of the game forced, or an own line that wins, may come first; otherwise it is the answer, never a fix of a goal of ours
        assert!(!d.log.why.starts_with("a fix of the"), "decision {n}: {}", d.log.why);
        let own_wins = d.log.moves.iter().find(|m| Some(&m.mv) == d.mv.as_ref()).and_then(|m| m.own_read.as_ref()).is_some_and(|r| r.0 == "Confirmed" && r.1 > 0);
        if d.value.abs() < cs_tactics::ledger::WIN - 1000 && !own_wins {
            assert!(d.log.why.starts_with("answers the threat of theirs on") && d.log.why.contains("a line of theirs stands on it"), "decision {n}: {}", d.log.why);
            assert!(d.chain.iter().any(|c| c.by.contains("a line of theirs stands on our goal unit")), "decision {n}: {:?}", d.chain);
        }
        // every move listed as an answer is an answer of the line
        for m in d.log.moves.iter().filter(|m| m.answers.iter().any(|a| a.contains("standing on our goal unit"))) { assert!(l.answers.contains(&m.mv), "decision {n}"); }
        println!("decision {n}: {} | {}", mv, d.log.why);
    }
}

#[test]
fn c4_a_position_with_no_line_on_the_goal_unit_is_decided_as_before() {
    let (g, st) = world();
    let j = Worth { name: "human".into(), table: table(), ours_first: false };
    let s = state("r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 7");
    let d = decide(&st, &g, &s, &[], &j, &whole(&g, 1.0));
    assert!(d.log.their_threat.as_deref().is_none_or(|t| !t.contains("a line of theirs stands on it")));
    assert!(d.log.moves.iter().all(|m| m.answers.iter().all(|a| !a.contains("standing on our goal unit"))));
}
