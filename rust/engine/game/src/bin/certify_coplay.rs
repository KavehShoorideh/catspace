//! certify_coplay <variant> <gdl file> <games> <seed> [<base gdl file>]: the chess kernel of `variant` against the GDL text (a delta composed on the base when one
//! is given: cs_gdl::compose), by co-play (cs_game::certify).
use cs_game::{certify, gdl_game::GdlGame, kernel};
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (variant, path, games, seed) = (&a[1], &a[2], a[3].parse().unwrap(), a[4].parse().unwrap());
    // plan 79.35: an optional start (a position text of the kernel's): the kernel begins there and the description is given the same start by `rules::restart_delta`; with no start the kernel begins at its own (the 518th of its numbered starts)
    let k = match a.get(6) { Some(fen) => kernel::KernelGame::from_fen(variant, fen).expect("start"), None => kernel::KernelGame::new(variant, 518).expect("variant") };
    let text = std::fs::read_to_string(path).expect("gdl file");
    let text = match a.get(5).filter(|b| *b != "-") { Some(base) => cs_gdl::compose(&std::fs::read_to_string(base).expect("base gdl file"), &text).expect("compose"), None => text };
    let facts = k.state_facts(&cs_game::Game::initial(&k));
    let text = if a.get(6).is_some() && !facts.is_empty() { let d = cs_game::rules::restart_delta(&text, &facts).expect("restart"); cs_gdl::compose(&text, &d).expect("compose the start") } else { text };
    let g = GdlGame::load(variant, &text).expect("gdl");
    let t = std::time::Instant::now();
    let r = certify::coplay(&k, &g, &|n| cs_game::outside::spec_term(n), games, seed);
    println!("OBS coplay {variant} vs {path}: games {}, plies {}, states {}, disagreements {}, {:.1} s", r.games, r.plies, r.states, r.disagreements.len(), t.elapsed().as_secs_f64());
    let mut ends = std::collections::BTreeMap::new();
    for t in &r.terminals { *ends.entry(t.clone()).or_insert(0) += 1; }
    println!("   game ends (the goal per role): {ends:?}");
    for d in r.disagreements.iter().take(5) {
        // the line played to the disagreement is long: its first words and, after the line, what differed
        match d.find("]: ") { Some(i) => println!("   FINDING {} ... {}", &d[..d.len().min(60)], &d[i + 3..d.len().min(i + 3 + 700)]), None => println!("   FINDING {}", &d[..d.len().min(400)]) }
    }
}
