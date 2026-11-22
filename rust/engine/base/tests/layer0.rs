//! PLAN 79.30 (SYSTEM_SPEC 11 "Layer 0 holds six things"): the tests that cross modules and the two fixtures of facts. The Rust here names no game and no thing of a world; the fixture texts do (one game of turns on a finite set of points, one game of many agents at once in a plane).
//! (a) the formula against the matcher, (b) the graph's distance against a path found, (c) the value of a tree of turns against the forcing reading, (d) a tree with chance against the expectation worked by hand in `chance`,
//! (e) every one of the six modules is used by both fixtures.
use cs_base::chance::Spread;
use cs_base::choice::{can_force, force_chance, value, Chooser, Order, Tree};
use cs_base::logic::{answers, bindings, Binding, Domain, Fact, FactIndex, Formula, Lit};
use cs_base::maker::{random_tree, Maker, Shape};
use cs_base::number::Num;
use cs_base::sets::{Kinds, Set};
use cs_base::structure::{Graph, Metric, Plane, Point, Region, Sequence};
use std::collections::BTreeMap;
use std::path::PathBuf;

const FIXTURES: [&str; 2] = ["turns.fixture", "plane.fixture"];

struct Fixture { name: String, facts: Vec<Fact>, idx: FactIndex }

fn load(name: &str) -> Fixture {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests").join("fixtures").join(name);
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let facts: Vec<Fact> = text.lines().filter_map(|l| {
        let f: Vec<&str> = l.split('\t').collect();
        if f[0] == "fact" { Some(Fact { relation: f[1].to_string(), args: f[2..].iter().map(|s| s.to_string()).collect() }) } else { None }
    }).collect();
    let idx = FactIndex::new(&facts);
    Fixture { name: name.to_string(), facts, idx }
}

fn num(s: &str) -> Num { match s.parse::<i64>() { Ok(i) => Num::whole(i), Err(_) => Num::real(s.parse::<f64>().unwrap()).unwrap() } }

impl Fixture {
    fn graph(&self, both: bool) -> Graph<String, String> {
        let mut g = Graph::new();
        for r in self.idx.rows("link") { if both { g.add_both(r[0].clone(), r[1].clone(), r[2].clone()); } else { g.add_link(r[0].clone(), r[1].clone(), r[2].clone()); } }
        g
    }
    fn kinds(&self) -> Kinds<String, String> { let mut k = Kinds::new(); for r in self.idx.rows("is_a") { k.add(r[1].clone(), r[0].clone()); } k }
    fn points(&self) -> BTreeMap<String, Point> { self.idx.rows("coord").iter().map(|r| (r[0].clone(), Point::new(num(&r[1]), num(&r[2])))).collect() }
    fn sequence(&self, name: &str) -> Sequence<Num> {
        let mut rows: Vec<(usize, Num)> = self.idx.rows("seq").iter().filter(|r| r[0] == name).map(|r| (r[1].parse().unwrap(), num(&r[2]))).collect();
        rows.sort_by_key(|r| r.0);
        Sequence::new(rows.into_iter().map(|r| r.1).collect())
    }
    fn sequences(&self) -> Vec<String> { Set::from_items(self.idx.rows("seq").iter().map(|r| r[0].clone())).iter().cloned().collect() }
    fn regions(&self) -> Vec<(String, Region)> {
        self.idx.rows("region").iter().map(|r| {
            let n: Vec<Num> = r[2..].iter().map(|s| num(s)).collect();
            let pt = |i: usize| Point::new(n[i], n[i + 1]);
            let shape = match r[1].as_str() {
                "box" => Region::Box { lo: pt(0), hi: pt(2) },
                "disc" => Region::Disc { centre: pt(0), radius: n[2] },
                "poly" => Region::Polygon((0..n.len() / 2).map(|i| pt(2 * i)).collect()),
                "half" => Region::HalfPlane { a: n[0], b: n[1], c: n[2] },
                other => panic!("a shape the fixture does not use: {other}"),
            };
            (r[0].clone(), shape)
        }).collect()
    }
    fn spread(&self, act: &str) -> Spread<String> {
        Spread::new(self.idx.rows("outcome").iter().filter(|r| r[0] == act).map(|r| (r[1].clone(), r[2].parse::<f64>().unwrap())).collect()).unwrap()
    }
    fn acts(&self) -> Vec<String> { Set::from_items(self.idx.rows("outcome").iter().map(|r| r[0].clone())).iter().cloned().collect() }
    fn tree(&self, node: &str) -> Tree<i64> {
        if let Some(r) = self.idx.rows("result").iter().find(|r| r[0] == node) { return Tree::Result(r[1].parse().unwrap()); }
        if let Some(r) = self.idx.rows("turn").iter().find(|r| r[0] == node) {
            let by = if r[1] == "A" { Chooser::A } else { Chooser::B };
            return Tree::turn(by, r[2..].iter().map(|c| self.tree(c)).collect()).unwrap();
        }
        if let Some(r) = self.idx.rows("table").iter().find(|r| r[0] == node) {
            return Tree::table(r[1].parse().unwrap(), r[2].parse().unwrap(), r[3..].iter().map(|c| self.tree(c)).collect()).unwrap();
        }
        if let Some(r) = self.idx.rows("chance_node").iter().find(|r| r[0] == node) {
            return Tree::chance(r[1..].chunks(2).map(|c| (self.tree(&c[0]), c[1].parse().unwrap())).collect()).unwrap();
        }
        panic!("no node {node} in {}", self.name)
    }
    /// The inner nodes no other node leads to.
    fn roots(&self) -> Vec<String> {
        let children: Set<String> = self.idx.rows("turn").iter().flat_map(|r| r[2..].to_vec()).chain(self.idx.rows("table").iter().flat_map(|r| r[3..].to_vec()))
            .chain(self.idx.rows("chance_node").iter().flat_map(|r| r[1..].chunks(2).map(|c| c[0].clone()).collect::<Vec<_>>())).collect();
        let inner = ["turn", "table", "chance_node"].iter().flat_map(|rel| self.idx.rows(rel).iter().map(|r| r[0].clone()).collect::<Vec<_>>()).collect::<Vec<_>>();
        inner.into_iter().filter(|n| !children.contains(n)).collect()
    }
}

fn order() -> Order<i64> { Order::by(|r: &i64| Num::whole(*r)) }

/// What each module does with a fixture: a count of the results it gave (every one must be above nought).
fn uses(fx: &Fixture) -> Vec<(&'static str, usize)> {
    let mut out = vec![];
    // logic: the formula, over the facts: the points with at least two links out, by the labels the facts hold
    let g = fx.graph(false);
    let labels: Set<String> = fx.idx.rows("link").iter().map(|r| r[2].clone()).collect();
    let nodes: Set<String> = g.nodes().into_iter().collect();
    let two_links = Formula::And(vec![
        Formula::Lit(Lit::pos("coord", &["?p", "?x", "?y"])),
        Formula::AtLeast { count: 2, var: "q".into(), over: Domain::Set(nodes.clone()), body: Box::new(Formula::Some { var: "l".into(), over: Domain::Set(labels), body: Box::new(Formula::Lit(Lit::pos("link", &["?p", "?q", "?l"]))) }) },
    ]);
    let (hits, _) = answers(&two_links, &Binding::new(), &fx.idx, 1000, &mut 0).unwrap();
    out.push(("logic", hits.len()));
    // sets: how many things of each kind, and the things of two kinds
    let kinds = fx.kinds();
    let kind_names: Vec<String> = kinds.kinds().iter().cloned().collect();
    let both: usize = kind_names.iter().flat_map(|a| kind_names.iter().map(move |b| (a, b))).filter(|(a, b)| a < b).map(|(a, b)| kinds.members(a).meet(&kinds.members(b)).len()).sum();
    out.push(("sets", both + kind_names.iter().map(|k| kinds.count(k)).sum::<usize>()));
    // number: the differences of every sequence, summed and compared
    let diffs: usize = fx.sequences().iter().map(|s| fx.sequence(s).differences().unwrap().iter().filter(|d| **d != Num::whole(0)).count()).sum();
    out.push(("number", diffs));
    // structure: steps between linked nodes (the graph), the plane distance between points, and which region holds which point
    let pts = fx.points();
    let reach = nodes.iter().flat_map(|a| nodes.iter().map(move |b| (a, b))).filter(|(a, b)| a != b && g.distance(a, b).is_some()).count();
    let plane = pts.values().flat_map(|a| pts.values().map(move |b| (a, b))).filter(|(a, b)| a != b && Plane.distance(a, b).is_some()).count();
    let inside: usize = fx.regions().iter().map(|(_, r)| pts.values().filter(|p| r.inside(p).unwrap()).count()).sum();
    out.push(("structure", reach.min(1) * (reach + plane + inside)));
    // chance: the expectation of each uncertain act, scoring the outcome by its position in the list (a number is read off each outcome)
    let chance: usize = fx.acts().iter().map(|a| { let s = fx.spread(a); let e = s.expectation(&|o| Num::whole(o.len() as i64)).unwrap(); assert!(e > Num::whole(0)); s.outcomes().len() }).sum();
    out.push(("chance", chance));
    // choice: both readings of every tree of the fixture
    let mut trees = 0;
    for r in fx.roots() {
        let t = fx.tree(&r);
        let b = value(&t, &order()).unwrap();
        assert!(b.lower <= b.upper);
        let _ = can_force(&t, Chooser::A, &|x| *x >= 1);
        let _ = force_chance(&t, Chooser::A, &|x| *x >= 1).unwrap();
        trees += 1;
    }
    out.push(("choice", trees));
    out
}

#[test]
fn every_one_of_the_six_modules_is_used_by_both_fixtures() {
    let mut seen: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
    for name in FIXTURES {
        let fx = load(name);
        let used = uses(&fx);
        let line: Vec<String> = used.iter().map(|(m, n)| format!("{m} {n}")).collect();
        println!("OBS 79.30 fixture {name}: {} facts; results per module: {}", fx.facts.len(), line.join(", "));
        for (m, n) in used { assert!(n > 0, "{name}: the module {m} gave nothing"); seen.entry(m).or_default().push(n); }
    }
    assert_eq!(seen.keys().copied().collect::<Vec<_>>(), vec!["chance", "choice", "logic", "number", "sets", "structure"]);
    assert!(seen.values().all(|v| v.len() == 2), "both fixtures, each module");
}

/// (a) The formula against the matcher, on the facts of both fixtures: a body of literals gives the bindings the matcher gives, as a set.
#[test]
fn the_formula_gives_the_matcher_bindings_on_the_facts_of_both_fixtures() {
    for name in FIXTURES {
        let fx = load(name);
        let bodies = vec![
            vec![Lit::pos("link", &["?a", "?b", "?l"]), Lit::pos("coord", &["?a", "?x", "?y"])],
            vec![Lit::pos("link", &["?a", "?b", "?l"]), Lit::pos("link", &["?b", "?c", "?m"]), Lit::differ("?a", "?c")],
            vec![Lit::pos("is_a", &["?t", "?k"]), Lit::pos("is_a", &["?t", "?j"]), Lit::differ("?k", "?j")],
            vec![Lit::pos("link", &["?a", "?b", "?l"]), Lit::pos("link", &["?b", "?a", "?l"]).negated()],
        ];
        let mut total = 0;
        for body in bodies {
            let (mut want, _) = bindings(&body, &Binding::new(), &fx.idx, 100_000, &mut 0);
            want.sort();
            want.dedup();
            let (mut got, cut) = answers(&Formula::And(body.iter().cloned().map(Formula::Lit).collect()), &Binding::new(), &fx.idx, 100_000, &mut 0).unwrap();
            got.sort();
            assert!(!cut);
            assert_eq!(got, want, "{name}: {body:?}");
            total += got.len();
        }
        assert!(total > 0);
        println!("OBS 79.30 formula against matcher on {name}: four bodies, {total} bindings, equal as sets");
    }
}

/// (b) The graph's distance against a path found, on the graphs of both fixtures and on made graphs.
#[test]
fn the_graphs_distance_is_the_length_of_the_path_found() {
    let mut tried = 0;
    for (name, both) in [("turns.fixture", true), ("plane.fixture", false)] {
        let g = load(name).graph(both);
        for a in g.nodes() { for b in g.nodes() {
            match g.path(&a, &b) {
                Some(p) => { assert_eq!(g.steps(&a, &b), Some(p.len() as u32 - 1)); assert_eq!(g.distance(&a, &b), Some(Num::whole(p.len() as i64 - 1))); assert!(p.windows(2).all(|w| g.has_link(&w[0], &w[1]))); tried += 1; }
                None => assert!(g.distance(&a, &b).is_none()),
            }
        } }
    }
    let mut m = Maker::new(71);
    for _ in 0..200 {
        let n = 2 + m.range(0, 7);
        let mut g: Graph<i64, u8> = Graph::new();
        for i in 0..n { g.add_node(i); }
        for _ in 0..m.below(3 * n as u64) { g.add_link(m.range(0, n - 1), m.range(0, n - 1), 0); }
        for a in 0..n { for b in 0..n { match g.path(&a, &b) { Some(p) => { assert_eq!(g.steps(&a, &b), Some(p.len() as u32 - 1)); tried += 1; } None => assert!(g.steps(&a, &b).is_none()) } } }
    }
    println!("OBS 79.30 graph: {tried} pairs with a path, each distance equal to the length of the path found");
}

fn has_chance(t: &Tree<i64>) -> bool {
    match t { Tree::Result(_) => false, Tree::Turn { options, .. } => options.iter().any(has_chance), Tree::Table { cells, .. } => cells.iter().any(has_chance), Tree::Chance(_) => true }
}

/// (c) The value of a tree of turns against the forcing reading, on the trees of the fixtures and on made trees (with the pure choices of a table too).
#[test]
fn the_value_of_a_tree_agrees_with_the_forcing_reading_on_the_fixtures_and_on_made_trees() {
    let mut count = 0;
    for name in FIXTURES {
        let fx = load(name);
        for r in fx.roots() {
            let t = fx.tree(&r);
            let b = value(&t, &order()).unwrap();
            assert!(b.lower <= b.upper);
            if !has_chance(&t) { for k in -1..=1 { assert_eq!(b.lower >= Num::whole(k), can_force(&t, Chooser::A, &|x| *x >= k), "{name} {r} {k}"); } }
            count += 1;
        }
    }
    let mut m = Maker::new(72);
    for _ in 0..2000 {
        let tb = m.chance();
        let t = random_tree(&mut m, 5, Shape { tables: tb, chance: false });
        let v = value(&t, &order()).unwrap().lower;
        for k in -9..=9 { assert_eq!(v >= Num::whole(k), can_force(&t, Chooser::A, &|x| *x >= k)); }
        count += 1;
    }
    println!("OBS 79.30 value against forcing: {count} trees, the value reaches k exactly when the forcing reading says A can force at least k");
}

/// The expectation of a tree of chances and turns worked by hand, apart from `choice`: a recursion written here over the `chance` module's expectation.
fn by_hand(t: &Tree<i64>) -> Num {
    match t {
        Tree::Result(r) => Num::whole(*r),
        Tree::Turn { by, options } => { let v: Vec<Num> = options.iter().map(by_hand).collect(); if *by == Chooser::A { v.into_iter().max().unwrap() } else { v.into_iter().min().unwrap() } }
        Tree::Chance(s) => s.map(&|o| by_hand(o)).expectation(&|n| *n).unwrap(),
        Tree::Table { .. } => unreachable!("no table here"),
    }
}

/// (d) A tree with chance against the expectation computed by hand in `chance`.
#[test]
fn a_tree_with_chance_has_the_value_the_expectation_worked_by_hand_gives() {
    let mut m = Maker::new(73);
    let mut with_chance = 0;
    for _ in 0..2000 {
        let t = random_tree(&mut m, 5, Shape { tables: false, chance: true });
        let b = value(&t, &order()).unwrap();
        assert_eq!(b.lower, by_hand(&t));
        assert_eq!(b.upper, b.lower);
        if matches!(t, Tree::Chance(_)) || t.size() > 3 { with_chance += 1; }
    }
    assert!(with_chance > 100);
    // the fixtures' own chance nodes, by hand: a half, a quarter and a quarter of the results 1, 0 and -1
    let by_hand_value = Spread::new(vec![(1i64, 0.5), (0, 0.25), (-1, 0.25)]).unwrap().expectation(&|r| Num::whole(*r)).unwrap();
    assert_eq!(by_hand_value, Num::real(0.25).unwrap());
    let t = load("turns.fixture").tree("n3");
    assert_eq!(value(&t, &order()).unwrap().lower, by_hand_value);
    // and the fixture's act as the spread of the same fixture, outcome by outcome
    let fx = load("turns.fixture");
    let reply = fx.spread("reply");
    let score = |o: &String| Num::whole(fx.idx.rows("result").iter().find(|r| r[0] == format!("r_{o}")).map(|r| r[1].parse::<i64>().unwrap()).unwrap());
    assert_eq!(reply.expectation(&score).unwrap(), by_hand_value);
    println!("OBS 79.30 chance: {with_chance} made trees with chance, each value equal to the hand expectation; the fixtures' chance node reads {by_hand_value}");
}

/// The simultaneous choice of the second fixture, by hand: the lower and the upper value of a table of chances, pure choices only.
#[test]
fn a_choice_at_once_has_a_lower_value_below_its_upper_and_the_fixture_says_by_how_much() {
    let fx = load("plane.fixture");
    let t = fx.tree("kick");
    let b = value(&t, &order()).unwrap();
    assert_eq!((b.lower, b.upper), (Num::real(0.25).unwrap(), Num::real(0.75).unwrap()), "best row against the worst column is a quarter; the best column against the best row holds it to three quarters");
    assert!(!can_force(&t, Chooser::A, &|x| *x >= 1) && !can_force(&t, Chooser::B, &|x| *x < 1));
    assert_eq!(force_chance(&t, Chooser::A, &|x| *x >= 1).unwrap(), Num::real(0.25).unwrap());
    let run = fx.tree("run_up");
    assert_eq!(value(&run, &order()).unwrap().lower, Num::real(0.25).unwrap());
}

/// Sets against logic: the members of a kind, counted as sets, are the bindings the matcher gives for the same question.
#[test]
fn a_kind_as_a_set_has_as_many_members_as_the_matcher_finds_for_it() {
    for name in FIXTURES {
        let fx = load(name);
        let kinds = fx.kinds();
        for k in kinds.kinds().iter() {
            let (found, _) = bindings(&[Lit::pos("is_a", &["?t", k])], &Binding::new(), &fx.idx, 1000, &mut 0);
            let set: Set<String> = found.iter().map(|b| b["t"].clone()).collect();
            assert_eq!(set, kinds.members(k));
            assert_eq!(kinds.count(k), set.len());
        }
    }
}

/// Number and sequence against structure: the plane's exact quadrance between neighbours in the fixture's sequence of positions equals the sum of the products of the differences read through `number`.
#[test]
fn the_plane_distance_on_the_fixtures_points_is_the_distance_by_the_differences() {
    for name in FIXTURES {
        let fx = load(name);
        let pts: Vec<Point> = fx.points().values().cloned().collect();
        for a in &pts { for b in &pts {
            let q = Plane::quadrance(a, b).unwrap();
            let by_parts = Sequence::new(vec![a.x, b.x]).differences().unwrap()[0];
            let by_parts_y = Sequence::new(vec![a.y, b.y]).differences().unwrap()[0];
            assert_eq!(q, by_parts.mul(&by_parts).unwrap().add(&by_parts_y.mul(&by_parts_y).unwrap()).unwrap());
            assert_eq!(Plane.distance(a, b).unwrap() == Num::whole(0), a == b);
        } }
    }
}
