//! PLAN 79.33 PART B (LAYER 0'S OWN FOLD; SYSTEM_SPEC 11 "What plan 79.32 left of one algorithm", 13 item 51): the answers of the readings that weigh numbers (`guarantee`, `value`, `force_chance`) on made trees, written to a file behind an env variable so that the answers before and after the fold became a case of the general walk can be
//! compared to the last number (scripts/whole_79_33_compare.py). The speed of the same readings on a fixed set of trees is printed as an OBS line, never asserted.
use cs_base::choice::{force_chance, guarantee, party, value, Chooser, Order, Tree};
use cs_base::maker::{random_tree, Maker, Shape};
use cs_base::number::Num;
use std::fmt::Write as _;
use std::time::Instant;

fn trees() -> Vec<(u64, Tree<i64>)> {
    let mut out = vec![];
    for (si, shape) in [Shape { tables: false, chance: false }, Shape { tables: true, chance: false }, Shape { tables: false, chance: true }, Shape { tables: true, chance: true }].into_iter().enumerate() {
        for seed in 0..400u64 {
            let mut m = Maker::new(seed * 7 + si as u64 + 1);
            out.push((seed * 10 + si as u64, random_tree(&mut m, 7, shape)));
        }
    }
    out
}

fn read_all(t: &Tree<i64>, line: &mut String) {
    let order = Order::by(|r: &i64| Num::whole(*r));
    let (pa, pb) = (party(&[Chooser::A]), party(&[Chooser::B]));
    let _ = write!(line, "ga={:?} gb={:?} v={:?} fa={:?} fb={:?}", guarantee(t, &pa, &order), guarantee(t, &pb, &order), value(t, &order), force_chance(t, Chooser::A, &|r| *r >= 0), force_chance(t, Chooser::B, &|r| *r < 3));
}

/// A tree whose weighted sums overflow: the first overflow is said, and said alike by every build.
fn overflowing() -> Vec<Tree<i64>> {
    let big = |v: i64| Tree::Result(v);
    let half = || Num::frac(1, 2).unwrap();
    vec![
        Tree::chance(vec![(big(i64::MAX / 2), half()), (big(i64::MAX / 2), half())]).unwrap(),
        Tree::turn(Chooser::A, vec![Tree::chance(vec![(big(i64::MAX), half()), (big(i64::MAX), half())]).unwrap(), big(1)]).unwrap(),
    ]
}

#[test]
fn the_values_of_made_trees_are_written_for_the_comparison_and_their_speed_is_printed() {
    let set = trees();
    let mut text = String::new();
    for (id, t) in &set {
        let mut line = String::new();
        read_all(t, &mut line);
        let _ = writeln!(text, "{id}\t{line}");
    }
    for (i, t) in overflowing().iter().enumerate() {
        let mut line = String::new();
        read_all(t, &mut line);
        let _ = writeln!(text, "overflow{i}\t{line}");
    }
    if let Ok(path) = std::env::var("LAYER0_VALUES") { std::fs::write(path, &text).unwrap(); }
    // the speed: the same readings over the same trees, repeated
    let order = Order::by(|r: &i64| Num::whole(*r));
    let pa = party(&[Chooser::A]);
    let t0 = Instant::now();
    let mut sum = 0i64;
    for _ in 0..300 { for (_, t) in &set { if let Ok(v) = guarantee(t, &pa, &order) { sum = sum.wrapping_add(v.as_f64() as i64); } } }
    let secs = t0.elapsed().as_secs_f64();
    let nodes: usize = set.iter().map(|(_, t)| t.size()).sum::<usize>() * 300;
    println!("OBS 79.33 cs_base value, {} trees, 300 rounds of `guarantee`: {nodes} nodes in {secs:.3} s, {:.0} nodes a second (check {sum})", set.len(), nodes as f64 / secs.max(1e-9));
}
