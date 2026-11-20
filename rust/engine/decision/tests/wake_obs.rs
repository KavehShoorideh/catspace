//! Plan 79.08 A2 and D: the OBS lines of time. The time of threats plus goals on the fixed constructed position of 79.06 with and without the shared table (A2), printed from a test as the plan asks. One thread, one run: directions.
use cs_decision::goals::{propose, GoalParams};
use cs_decision::threat_verdict::Record;
use cs_decision::threats::ThreatParams;
use cs_game::kernel::KernelGame;
use cs_game::placed::{Kind, Placed};
use cs_game::Game;
use cs_tactics::sheet::base_graph::Static;
use cs_tactics::sheet::Names;
use std::collections::HashMap;

fn world() -> (KernelGame, Static) {
    let g = KernelGame::new("chess", 518).unwrap();
    let text = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../../rules/chess.gdl")).unwrap();
    let steps = g.steps(&text, &["diagonal_of".to_string(), "step".to_string()], &[("square".to_string(), vec!["file".to_string(), "rank".to_string()])]).unwrap();
    let st = Static::new(&steps, &Names::new(g.place_names(), g.kind_names(), g.roles())).unwrap();
    (g, st)
}
fn record() -> Record { Record { worth: [("p", 475), ("n", 1216), ("b", 1301), ("r", 1945), ("q", 3583)].iter().map(|(k, v)| (k.to_string(), *v)).collect(), half: HashMap::new() } }
fn kind(g: &KernelGame, name: &str) -> Kind { g.kind_names().iter().position(|k| k == name).unwrap() as Kind }

#[test]
fn a2_obs_threats_plus_goals_after_the_shared_table() {
    let (g, st) = world();
    let s = KernelGame::from_fen("chess", "r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 7").unwrap().initial();
    for share in [false, true] {
        for b in [2usize, cs_decision::goals::BUDGET] {
            let tp = ThreatParams { goal: Some(kind(&g, "king")), share_table: share, ..Default::default() };
            let t0 = std::time::Instant::now();
            let p = propose(&st, &g, &s, &record(), &tp, &GoalParams { budget: b, ..Default::default() });
            println!("OBS threats plus goals 79.08: fixed constructed position of 79.06, one thread, budget {b}, shared table {share}: whole {:.3} s; questions asked {}; goals {}; nodes of the list {}", t0.elapsed().as_secs_f64(), p.questions, p.goals.len(), p.list.cost.iter().map(|c| c.nodes).sum::<u64>());
            assert!(!p.goals.is_empty());
        }
    }
}
