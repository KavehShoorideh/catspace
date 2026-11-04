//! certify_coplay <variant> <gdl file> <games> <seed> [<base gdl file>]: the chess kernel of `variant` against the GDL text (a delta composed on the base when one
//! is given: cs_gdl::compose), by co-play (cs_game::certify).
use cs_game::{certify, gdl_game::GdlGame, kernel};
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (variant, path, games, seed) = (&a[1], &a[2], a[3].parse().unwrap(), a[4].parse().unwrap());
    let k = kernel::KernelGame::new(variant, 518).expect("variant");
    let text = std::fs::read_to_string(path).expect("gdl file");
    let text = match a.get(5) { Some(base) => cs_gdl::compose(&std::fs::read_to_string(base).expect("base gdl file"), &text).expect("compose"), None => text };
    let g = GdlGame::load(variant, &text).expect("gdl");
    let t = std::time::Instant::now();
    let r = certify::coplay(&k, &g, &|n| kernel::uci_to_spec_term(n), games, seed);
    println!("OBS coplay {variant} vs {path}: games {}, plies {}, states {}, disagreements {}, {:.1} s", r.games, r.plies, r.states, r.disagreements.len(), t.elapsed().as_secs_f64());
    let mut ends = std::collections::BTreeMap::new();
    for t in &r.terminals { *ends.entry(t.clone()).or_insert(0) += 1; }
    println!("   game ends (the goal per role): {ends:?}");
    for d in r.disagreements.iter().take(5) { println!("   FINDING {}", &d[..d.len().min(400)]); }
}
