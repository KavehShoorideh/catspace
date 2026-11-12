//! Plan 78.39 stage 1: the scenario report (cs_tactics::scenario) on the chess description composed with the scenario concepts, and the goal's target read from the description.
use cs_game::gdl_game::FactView;
use cs_game::goal_target::{goal_target, GoalTarget};
use cs_game::kernel::KernelGame;
use cs_game::{Game, Vocabulary};
use cs_tactics::scenario::{explain, resolve_role, scenario, Params, RelSpec, Relations, Report};
use std::path::PathBuf;

fn rules(rel: &str) -> String { std::fs::read_to_string(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../rules").join(rel)).unwrap() }

/// The description as the vocabulary file names it: the base and the concepts, composed (additive deltas: the texts joined).
fn text() -> String { ["chess.gdl", "concepts/chess.gdl", "concepts/chess_units.gdl", "concepts/chess_scenario.gdl"].iter().map(|f| rules(f)).collect::<Vec<_>>().join("\n") }

fn spec(r: &str) -> RelSpec { RelSpec { relation: r.into(), kind: 0, owner: 1, from: vec![2, 3], to: vec![4, 5] } }
fn rels() -> Relations { Relations { acts_on: spec("strike"), can_go: vec![spec("go"), spec("go_within2"), spec("go_within3")] } }
fn view() -> FactView {
    FactView::new(&text(), Vocabulary { derived: ["strike", "go", "go_within2", "go_within3"].iter().map(|x| x.to_string()).collect(), delta: vec!["cell".into()], units: None }).unwrap()
}
fn target() -> GoalTarget { goal_target(&text()).unwrap() }

fn read(fen: &str, role: &str, k: u32) -> Report {
    let g = KernelGame::from_fen("chess", fen).unwrap();
    let s = g.initial();
    let me = resolve_role(&g, &s, role).unwrap();
    scenario(&g, &s, &view(), &target(), &rels(), me, &Params { k, ..Params::default() }).unwrap()
}
fn at(l: &str) -> Vec<String> { vec![l[..1].to_string(), l[1..].to_string()] }

// black Ka8 in check by Rh8; the exits a7 and b7 are open; white has Re1 (a rook on rank 7 would cover both) and Kh1
const CORNER: &str = "k6R/8/8/8/8/8/8/4R2K b - - 0 1";

#[test]
fn the_target_is_read_from_the_description_and_a_description_without_one_is_unreadable() {
    let t = target();
    assert_eq!((t.kind.as_str(), t.fluent.as_str(), t.arity), ("king", "cell", 4));
    assert_eq!((t.kind_arg, t.owner_arg, t.place_args.clone()), (3, 2, vec![0, 1]));
    println!("OBS target {:?}", t);
    let ttt = "(role x) (role o) (init (cell 1 1 b)) (init (control x))
        (<= (row ?m ?p) (true (cell ?m 1 ?p)) (true (cell ?m 2 ?p)) (true (cell ?m 3 ?p)))
        (<= (line ?p) (row ?m ?p)) (<= terminal (line x)) (<= (goal x 100) (line x)) (<= (goal o 0) (line x))
        (<= (legal ?p (mark ?m ?n)) (true (cell ?m ?n b)) (true (control ?p))) (<= (next (control o)) (true (control x)))";
    let e = goal_target(ttt).unwrap_err();
    assert!(e.starts_with("unreadable"), "{e}");
    assert!(goal_target("(role x) (init (a)) (<= terminal (a)) (<= (next (a)) (true (a)))").unwrap_err().starts_with("unreadable"));
}

#[test]
fn a_cornered_target_with_two_open_exits_and_one_unit_that_covers_both_from_one_place() {
    let r = read(CORNER, "white", 1);
    println!("{}", explain(&r).join("\n"));
    assert_eq!(r.exits_from, "moves");
    assert_eq!(r.holes.iter().map(|h| h.at.clone()).collect::<Vec<_>>(), vec![at("a7"), at("b7")]);
    assert!(r.covered.iter().any(|c| c.own && c.by.len() == 1 && c.by[0].kind == "rook"));
    assert_eq!(r.scenarios.len(), 1);
    let s = &r.scenarios[0];
    assert_eq!((s.placements.len(), s.total_moves), (1, 1));
    assert_eq!((s.placements[0].unit.kind.as_str(), s.placements[0].unit.at.clone(), s.placements[0].to.clone()), ("rook", at("e1"), at("e7")));
    assert_eq!(r.safe, Some(false));
    assert!(r.contest.is_none());
}

#[test]
fn two_holes_no_single_place_covers_and_one_unit_to_cover_them_means_no_scenario_and_safe() {
    // the same corner, white's second unit a knight: it reaches a place that acts on a7 and one that acts on b7, none on both; the rook on h8 may not leave
    let r = read("k6R/8/8/8/8/2N5/8/7K b - - 0 1", "white", 2);
    println!("{}", explain(&r).join("\n"));
    assert_eq!(r.holes.len(), 2);
    assert!(r.supply.iter().all(|s| !s.is_empty()), "each hole has a supplier alone");
    assert!(r.scenarios.is_empty());
    assert_eq!(r.safe, Some(true));
}

#[test]
fn a_hole_on_a_place_class_the_only_supplier_never_acts_on_has_no_supply() {
    // a bishop on the light squares can act on b7 (light) and never on a7 (dark)
    let r = read("k6R/8/8/8/8/8/2B5/7K b - - 0 1", "white", 2);
    println!("{}", explain(&r).join("\n"));
    let a7 = r.holes.iter().position(|h| h.at == at("a7")).unwrap();
    let b7 = r.holes.iter().position(|h| h.at == at("b7")).unwrap();
    assert!(r.supply[a7].iter().all(|x| r.units[x.unit].kind != "bishop"), "{:?}", r.supply[a7]);
    assert!(r.supply[b7].iter().any(|x| r.units[x.unit].kind == "bishop"));
    assert_eq!(r.safe, Some(true));
}

#[test]
fn the_goal_already_holds_means_no_holes() {
    let r = read("R5k1/5ppp/8/8/8/8/8/4K3 b - - 0 1", "white", 2);
    assert!(r.goal_holds && r.holes.is_empty());
    assert_eq!(r.safe, Some(false));
    let b = read("R5k1/5ppp/8/8/8/8/8/4K3 b - - 0 1", "black", 2);
    assert!(!b.goal_holds && b.holes.is_empty());
}

#[test]
fn the_same_report_from_the_other_side() {
    // white's rook and king against black's lone king in the corner: each side's report names the other side's king as its target
    let fen = "k7/8/8/8/8/8/8/4R2K w - - 0 1";
    let w = read(fen, "white", 2);
    let b = read(fen, "black", 2);
    println!("{}\n{}", explain(&w).join("\n"), explain(&b).join("\n"));
    let (tw, tb) = (w.target.as_ref().unwrap(), b.target.as_ref().unwrap());
    assert_eq!((tw.owner.as_str(), tw.at.clone()), ("black", at("a8")));
    assert_eq!((tb.owner.as_str(), tb.at.clone()), ("white", at("h1")));
    assert_eq!((w.exits_from.as_str(), b.exits_from.as_str()), ("moves", "moves"));        // (white to move: its pass; black's report: the state itself)
    assert!(!w.holes.is_empty() && w.safe.is_some());
    assert_eq!(b.safe, Some(true), "no unit of black reaches the king's corner: that king is safe from this side");
    // a side in check cannot pass: the exits are then read from the zone, and the report says so
    let z = read(CORNER, "black", 2);
    println!("{}", explain(&z).join("\n"));
    assert_eq!(z.exits_from, "zone");
    assert!(z.notes.iter().any(|n| n.contains("no pass")));
    assert_eq!(z.safe, Some(true));
    // and the report names the roles by their words
    let g = KernelGame::from_fen("chess", CORNER).unwrap();
    let s = g.initial();
    assert_eq!(resolve_role(&g, &s, "mover").unwrap(), 1);
    assert_eq!(resolve_role(&g, &s, "other").unwrap(), 0);
    assert!(resolve_role(&g, &s, "green").is_err());
}

#[test]
fn a_report_is_deterministic() {
    let a = read(CORNER, "white", 2);
    let b = read(CORNER, "white", 2);
    assert_eq!(explain(&a), explain(&b));
    assert_eq!(a.scenarios, b.scenarios);
    assert_eq!(a.nodes, b.nodes);
}

#[test]
fn the_layer_files_name_no_object_of_the_game() {
    let words = ["king", "queen", "rook", "bishop", "knight", "pawn", "chess", "check", "mate"];
    for f in ["tactics/src/scenario.rs", "game/src/goal_target.rs"] {
        let text = std::fs::read_to_string(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..").join(f)).unwrap().to_lowercase();
        let b = text.as_bytes();
        for w in words {
            let mut from = 0;
            while let Some(i) = text[from..].find(w) {
                let (s, e) = (from + i, from + i + w.len());
                let word_start = s == 0 || !b[s - 1].is_ascii_alphabetic();
                let word_end = e >= b.len() || !b[e].is_ascii_alphabetic();
                assert!(!(word_start && word_end), "{f} names {w}");
                from = e;
            }
        }
    }
    // the vocabulary file carries the relations
    let v = rules("learned/chess/vocabulary.scenario.json");
    for n in ["\"strike\"", "\"go\"", "\"go_within2\"", "\"go_within3\"", "concepts/chess_scenario.gdl"] { assert!(v.contains(n), "{n}"); }
}

#[test]
fn obs_timing_of_ten_positions() {
    let fens = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
        "r4rk1/pp3ppp/2p2n2/q2p4/3P4/2N1PN2/PP3PPP/R2QK2R w KQ - 0 12",
        "2r3k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1",
        "8/8/8/4k3/8/8/4K3/4Q3 w - - 0 1",
        "7k/5K2/8/8/8/8/8/R7 w - - 0 1",
        CORNER,
        "6k1/5p1p/6p1/8/8/5N2/5PPP/6K1 b - - 0 1",
        "r1b1k2r/ppppqppp/2n2n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQK2R w KQkq - 0 7",
        "4k3/8/8/8/8/8/PPPPPPPP/4K3 w - - 0 1",
    ];
    let t0 = std::time::Instant::now();
    let v = view();
    let tg = target();
    let load = t0.elapsed().as_secs_f64();
    let mut times = vec![];
    for f in fens {
        let g = KernelGame::from_fen("chess", f).unwrap();
        let s = g.initial();
        let me = g.mover(&s).unwrap();
        let t = std::time::Instant::now();
        let r = scenario(&g, &s, &v, &tg, &rels(), me, &Params::default()).unwrap();
        if f.starts_with("7k/5K2") { println!("{}", explain(&r).join("\n")); }
        times.push((t.elapsed().as_secs_f64(), r.holes.len(), r.scenarios.len(), r.nodes));
    }
    let mean = times.iter().map(|x| x.0).sum::<f64>() / times.len() as f64;
    let max = times.iter().map(|x| x.0).fold(0.0, f64::max);
    println!("OBS scenario positions=10 load_s={load:.3} mean_ms={:.1} max_ms={:.1} holes={:?} scenarios={:?} nodes={:?}", mean * 1000.0, max * 1000.0,
             times.iter().map(|x| x.1).collect::<Vec<_>>(), times.iter().map(|x| x.2).collect::<Vec<_>>(), times.iter().map(|x| x.3).collect::<Vec<_>>());
}
