//! PLAN 79.32 PART F (SYSTEM_SPEC 11 "The pure search is mathematics and lives in layer 0", 13 item 50; piece: the tactical reasoner as a case of the base): the SPEED of layer 0's walk as layer 4 uses it, on fixed questions: the first ten positions of each game of the goals fixture, every goal of a position in one list, one thread,
//! a budget of 50 000 nodes, the plain tree and the narrowed tree, each asked three times. It prints the nodes, the positions made, the rows matched and the nodes a second (OBS lines, directions only: a clock is a measure, never a law); the counts are asserted equal from one asking to the next.
//! It uses layer 1 (statements), layer 2 (the game through the text door) and this layer's entry. This file names no object of any game; the fixture does.
use cs_game::kernel::KernelGame;
use cs_game::rules::{Ruled, StateText};
use cs_game::Game;
use cs_rep::history::Seen;
use cs_rep::lemma::{Lit, LitJson};
use cs_rep::statement::{Holder, Says, Standing, Statement};
use cs_tactics::ledger::Count;
use cs_tactics::meansends::{ask_with, Goal, Question};
use cs_tactics::reach::{NARROWED, PLAIN};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

type R = Ruled<KernelGame>;

struct GameFx { name: String, decl: String, positions: Vec<(String, Vec<Statement>)> }

fn fixture(name: &str) -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests").join("fixtures").join(name);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

fn read_goals() -> Vec<GameFx> {
    let mut out: Vec<GameFx> = vec![];
    for line in fixture("goals.fixture").lines() {
        let f: Vec<&str> = line.split('\t').collect();
        match f[0] {
            "note" => {}
            "game" => out.push(GameFx { name: f[1].into(), decl: f[2].into(), positions: vec![] }),
            "pos" => {
                let v: cs_rep::serde_json::Value = cs_rep::serde_json::from_str(f[3]).unwrap();
                let goals = v.as_array().unwrap().iter().enumerate().map(|(i, g)| {
                    let pattern: Vec<Lit> = g["pattern"].as_array().unwrap().iter().map(|l| Lit::from_json(l).unwrap()).collect();
                    Statement { id: format!("{}#{i}", f[1]), pattern, says: Says::Holds, standing: Standing::Fact, holder: Holder::Ours, also: vec![] }
                }).collect();
                out.last_mut().unwrap().positions.push((f[2].into(), goals));
            }
            other => panic!("unknown tag {other}"),
        }
    }
    out
}

fn open(fx: &GameFx) -> R {
    let rules = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../rules");
    Ruled::open(KernelGame::new(&fx.name, 518).unwrap(), &fx.name, &rules, &fx.decl).unwrap()
}

#[test]
fn the_speed_of_the_walk_on_fixed_questions_is_printed() {
    let games = read_goals();
    let budget = 50_000;
    for (name, nar) in [("plain", PLAIN), ("narrowed", NARROWED)] {
        let mut counts: Vec<(u64, u64, u64)> = vec![];
        for _ in 0..3 {
            let (mut nodes, mut makes, mut matches, mut questions) = (0u64, 0u64, 0u64, 0usize);
            let t0 = Instant::now();
            for fx in &games {
                let g = open(fx);
                let st = g.board().unwrap();
                for (text, statements) in fx.positions.iter().take(10) {
                    let s = g.read_state_text(text).unwrap();
                    let goals: Vec<Goal> = statements.iter().map(|x| Goal::Holds(x.clone())).collect();
                    let q = Question { game: &g, board: &st, state: &s, judge: &Count, side: g.mover(&s).expect("a position the game goes on from"), goals: &goals, nodes: budget, seconds: None, max_depth: 5, history: Arc::new(Seen::new()) };
                    let r = ask_with(&q, nar);
                    nodes += r.nodes; makes += r.makes; matches += r.matches; questions += 1;
                }
            }
            let secs = t0.elapsed().as_secs_f64();
            counts.push((nodes, makes, matches));
            println!("OBS 79.32 speed {name} at {budget} nodes, {questions} questions, one thread: nodes {nodes}, positions made {makes}, rows matched {matches}, {secs:.3} s, {:.0} nodes a second", nodes as f64 / secs.max(1e-9));
        }
        assert!(counts.iter().all(|c| *c == counts[0]), "the counts are the same at every asking");
    }
}
