//! The pieces of the stronger rank source, each alone: the leaf valuations, the greedy and weighted playout policies, the gain prior, and a search whose leaf
//! valuation tells apart what a draw leaf cannot. The toy game is units only: each side has units of kinds "a" (worth 3) and "b" (worth 1); a move removes one enemy
//! unit of a kind or passes; a side with no units has lost.
use cs_game::{Game, Role};
use cs_operator::{Budget, Goal, Operator};
use cs_planning::mcts::{MctsOp, Rollout, Uniform};
use cs_planning::rollout::{gain, Balance, Draw, GainPrior, Greedy, Leaf, Prior, Weighted};
use cs_tactics::ledger::{Count, Judge, Worth};
use std::collections::HashMap;

#[derive(Clone, Copy)]
struct Toy { start: [u8; 4], passive_second: bool }

/// (a0, b0, a1, b1, mover)
type St = [u8; 5];
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
enum Mv { TakeA, TakeB, Pass }

impl Game for Toy {
    type State = St;
    type Move = Mv;
    fn name(&self) -> String { "toy".into() }
    fn roles(&self) -> Vec<String> { vec!["p".into(), "q".into()] }
    fn initial(&self) -> St { [self.start[0], self.start[1], self.start[2], self.start[3], 0] }
    fn mover(&self, s: &St) -> Option<Role> { if self.outcome(s).is_some() { None } else { Some(s[4] as usize) } }
    fn moves(&self, s: &St) -> Vec<Mv> {
        if self.outcome(s).is_some() { return vec![]; }
        let me = s[4] as usize;
        if me == 1 && self.passive_second { return vec![Mv::Pass]; }
        let en = 1 - me;
        let mut v = vec![];
        if s[2 * en] > 0 { v.push(Mv::TakeA); }
        if s[2 * en + 1] > 0 { v.push(Mv::TakeB); }
        v.push(Mv::Pass);
        v
    }
    fn apply(&self, s: &St, m: &Mv) -> St {
        let me = s[4] as usize;
        let en = 1 - me;
        let mut t = *s;
        match m { Mv::TakeA => t[2 * en] -= 1, Mv::TakeB => t[2 * en + 1] -= 1, Mv::Pass => {} }
        t[4] = en as u8;
        t
    }
    fn outcome(&self, s: &St) -> Option<Vec<i64>> {
        if s[0] + s[1] == 0 { Some(vec![0, 100]) } else if s[2] + s[3] == 0 { Some(vec![100, 0]) } else { None }
    }
    fn move_name(&self, _s: &St, m: &Mv) -> String { format!("{m:?}") }
    fn key(&self, s: &St) -> u64 { s.iter().fold(7u64, |h, x| h * 31 + *x as u64) }
    fn units(&self, s: &St) -> Vec<(Role, String)> {
        let mut v = vec![];
        for r in 0..2 {
            for _ in 0..s[2 * r] { v.push((r, "a".to_string())); }
            for _ in 0..s[2 * r + 1] { v.push((r, "b".to_string())); }
        }
        v
    }
}

fn worths() -> Worth { Worth { name: "toy".into(), table: HashMap::from([("a".to_string(), 3), ("b".to_string(), 1)]), ours_first: true } }
fn toy() -> Toy { Toy { start: [3, 3, 3, 3], passive_second: false } }

#[test]
fn judges_report_the_plain_worth() {
    assert_eq!(Count.worth("a"), 1);
    assert_eq!(worths().worth("a"), 3);
    assert_eq!(worths().worth("zzz"), 0);
}

#[test]
fn draw_is_fifty_for_every_role() {
    let g = toy();
    assert_eq!(Draw.value(&g, &g.initial()), vec![50.0, 50.0]);
}

#[test]
fn balance_zero_is_exactly_fifty_and_is_symmetric() {
    let g = toy();
    let b = Balance::new(Count, 3.0);
    assert_eq!(b.value(&g, &g.initial()), vec![50.0, 50.0]);
    // one more unit for p, and the mirror image for q: the two roles' values sum to 100 and swap
    let up = [4, 3, 3, 3, 0];
    let down = [3, 3, 4, 3, 0];
    let (u, d) = (b.value(&g, &up), b.value(&g, &down));
    assert!(u[0] > 50.0 && u[1] < 50.0);
    assert!((u[0] + u[1] - 100.0).abs() < 1e-9);
    assert!((u[0] - d[1]).abs() < 1e-9 && (u[1] - d[0]).abs() < 1e-9);
}

#[test]
fn balance_scale_and_worth_set_the_squash() {
    let g = toy();
    // three units up with a count judge and scale 3: 100 / (1 + e^-1)
    let s = [6, 3, 3, 3, 0];
    let v = Balance::new(Count, 3.0).value(&g, &s)[0];
    assert!((v - 100.0 / (1.0 + (-1.0f64).exp())).abs() < 1e-9, "{v}");
    // a unit of kind "a" is worth three of kind "b" for the worth judge
    let a_up = Balance::new(worths(), 3.0).value(&g, &[4, 3, 3, 3, 0])[0];
    let b_up = Balance::new(worths(), 3.0).value(&g, &[3, 4, 3, 3, 0])[0];
    assert!(a_up > b_up && b_up > 50.0);
}

#[test]
fn names_say_what_is_in_use() {
    assert_eq!(Uniform::new(1).name(), "uniform");
    assert_eq!(Draw.name(), "draw");
    assert!(Balance::new(Count, 3.0).name().starts_with("balance("));
    assert_eq!(Greedy::new(Count, 0.1, 1).name(), "greedy");
    let w = Weighted::new(GainPrior { judge: Count, temperature: 1.0 }, 0.05, 1);
    assert!(w.name().starts_with("weighted(gain("));
}

#[test]
fn gain_reads_captures() {
    let g = toy();
    let s = g.initial();
    assert_eq!(gain(&g, &worths(), &s, &Mv::TakeA), 3);
    assert_eq!(gain(&g, &worths(), &s, &Mv::TakeB), 1);
    assert_eq!(gain(&g, &worths(), &s, &Mv::Pass), 0);
}

#[test]
fn greedy_takes_the_best_gain_and_eps_one_is_uniform() {
    let g = toy();
    let s = g.initial();
    let ms = g.moves(&s);
    let mut greedy = Greedy::new(worths(), 0.0, 5);
    for _ in 0..200 { assert_eq!(ms[greedy.pick(&g, &s, &ms)], Mv::TakeA); }
    let mut uni = Greedy::new(worths(), 1.0, 5);
    let mut counts = [0usize; 3];
    for _ in 0..3000 { counts[uni.pick(&g, &s, &ms)] += 1; }
    for c in counts { assert!((c as f64 / 3000.0 - 1.0 / 3.0).abs() < 0.04, "{counts:?}"); }
    // clones restart from the seed
    let mut a = Greedy::new(worths(), 0.5, 9);
    let mut b = a.clone();
    let xs: Vec<usize> = (0..50).map(|_| a.pick(&g, &s, &ms)).collect();
    let ys: Vec<usize> = (0..50).map(|_| b.pick(&g, &s, &ms)).collect();
    assert_eq!(xs, ys);
}

#[test]
fn greedy_is_uniform_when_nothing_gains() {
    // a judge that values nothing: no move gains, so the pick is uniform
    let zero = Worth { name: "zero".into(), table: HashMap::new(), ours_first: false };
    let g = toy();
    let s = g.initial();
    let ms = g.moves(&s);
    let mut p = Greedy::new(zero, 0.0, 3);
    let mut counts = [0usize; 3];
    for _ in 0..3000 { counts[p.pick(&g, &s, &ms)] += 1; }
    for c in counts { assert!((c as f64 / 3000.0 - 1.0 / 3.0).abs() < 0.04, "{counts:?}"); }
}

#[derive(Clone)]
struct Fixed(Vec<f64>);
impl Prior for Fixed {
    fn weights<G: Game>(&self, _g: &G, _s: &G::State, moves: &[G::Move]) -> Vec<f64> { self.0[..moves.len()].to_vec() }
    fn name(&self) -> String { "fixed".into() }
}

#[test]
fn weighted_frequencies_track_the_weights() {
    let g = toy();
    let s = g.initial();
    let ms = g.moves(&s);
    let mut w = Weighted::new(Fixed(vec![1.0, 3.0, 0.0]), 0.5, 11);
    let mut counts = [0usize; 3];
    let n = 2000;
    for _ in 0..n { counts[w.pick(&g, &s, &ms)] += 1; }
    let eff = [1.5, 3.5, 0.5];
    for i in 0..3 { assert!((counts[i] as f64 / n as f64 - eff[i] / 5.5).abs() < 0.04, "{counts:?}"); }
    assert!(counts[2] > 0, "the floor keeps every move possible");
}

#[test]
fn gain_prior_prefers_the_gain_and_matches_greedy_at_low_temperature() {
    let g = toy();
    let s = g.initial();
    let ms = g.moves(&s);
    let w = GainPrior { judge: worths(), temperature: 1.0 }.weights(&g, &s, &ms);
    assert_eq!(w.len(), 3);
    assert!(w[0] > w[1] && w[1] > w[2] && w[2] > 0.0);
    let mut p = Weighted::new(GainPrior { judge: worths(), temperature: 0.05 }, 1e-9, 2);
    for _ in 0..100 { assert_eq!(ms[p.pick(&g, &s, &ms)], Mv::TakeA); }
}

fn root_values(o: &cs_operator::Outcome<Mv>) -> Vec<(i64, i64)> {
    o.facts.iter().filter(|f| f.relation == "visits").map(|f| (f.args[1].parse().unwrap(), f.args[2].parse().unwrap())).collect()
}

#[test]
fn a_balance_leaf_prefers_the_capture_where_a_draw_leaf_cannot_tell() {
    // the second side only passes; a playout is cut at once (max_playout 0), so what ranks the root moves is the leaf
    let g = Toy { start: [3, 3, 3, 3], passive_second: true };
    let s = g.initial();
    let goal = Goal::Estimate { role: 0 };
    let draw = MctsOp::with_policy(1.41, 0, Uniform::new(1), Draw).run(&g, &s, &goal, Budget::new(30.0, 3000));
    for (_, v) in root_values(&draw) { assert_eq!(v, 5000, "a draw leaf scores every root move alike"); }
    let bal = MctsOp::with_policy(1.41, 0, Uniform::new(1), Balance::new(worths(), 3.0)).run(&g, &s, &goal, Budget::new(30.0, 3000));
    assert_eq!(bal.mv, Some(Mv::TakeA), "{:?}", root_values(&bal));
    assert!(bal.evidence.iter().any(|e| e.contains("rollout uniform") && e.contains("leaf balance(")), "{:?}", bal.evidence);
    assert!(draw.evidence.iter().any(|e| e.contains("leaf draw")));
}

#[test]
fn the_default_operator_is_uniform_with_a_draw_leaf() {
    let g = toy();
    let o = MctsOp::new(1.41, 20, 3).run(&g, &g.initial(), &Goal::Estimate { role: 0 }, Budget::new(5.0, 2000));
    assert!(o.evidence.iter().any(|e| e.contains("rollout uniform, leaf draw")));
    let o2 = MctsOp::with_rollout(1.41, 20, Greedy::new(Count, 0.1, 3)).run(&g, &g.initial(), &Goal::Estimate { role: 0 }, Budget::new(5.0, 2000));
    assert!(o2.evidence.iter().any(|e| e.contains("rollout greedy, leaf draw")));
}
