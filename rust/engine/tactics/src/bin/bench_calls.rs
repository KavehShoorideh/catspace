//! The per-position cost of the Game trait's calls the ledger search makes (moves, outcome, apply, captured, changed, key) and of its judge -- where a node's time goes.
use cs_game::{kernel::KernelGame, Game};
use cs_tactics::ledger::{Judge, Worth};
use std::time::Instant;
fn main() {
    let g = KernelGame::from_fen("chess", "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 4 5").unwrap();
    let s = g.initial();
    let ms = g.moves(&s);
    let w = Worth { name: "m".into(), table: [("p", 1), ("n", 3), ("b", 3), ("r", 5), ("q", 9), ("k", 0)].iter().map(|(k, v)| (k.to_string(), *v)).collect(), ours_first: false };
    let n = 200_000;
    let t = Instant::now(); for _ in 0..n { std::hint::black_box(g.moves(&s)); } let a = t.elapsed().as_secs_f64() / n as f64;
    let t = Instant::now(); for _ in 0..n { std::hint::black_box(g.outcome(&s)); } let b = t.elapsed().as_secs_f64() / n as f64;
    let t = Instant::now(); for _ in 0..n { std::hint::black_box(g.apply(&s, &ms[0])); } let c = t.elapsed().as_secs_f64() / n as f64;
    let t = Instant::now(); for _ in 0..n { for m in &ms { std::hint::black_box(g.captured(&s, m)); } } let d = t.elapsed().as_secs_f64() / n as f64;
    let t = Instant::now(); for _ in 0..n { for m in &ms { std::hint::black_box(g.changed(&s, m)); } } let e = t.elapsed().as_secs_f64() / n as f64;
    let t = Instant::now(); for _ in 0..n { std::hint::black_box(g.key(&s)); } let f = t.elapsed().as_secs_f64() / n as f64;
    let t = Instant::now(); for _ in 0..n { for m in &ms { for (o, k) in g.captured(&s, m) { std::hint::black_box(w.delta(0, o, &k)); } } } let h = t.elapsed().as_secs_f64() / n as f64;
    println!("OBS per-position costs ({} legal moves): moves {:.2} us, outcome {:.2} us, apply {:.2} us, captured (all moves) {:.2} us, changed (all moves) {:.2} us, key {:.2} us, captured+judge (all moves) {:.2} us",
             ms.len(), a * 1e6, b * 1e6, c * 1e6, d * 1e6, e * 1e6, f * 1e6, h * 1e6);
}
