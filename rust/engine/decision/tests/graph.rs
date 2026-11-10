//! Plan 78.21: the typed fact graph on a tiny made-up game (two sorts, an object, a static relation, a state relation, a derived one) and the forward pass of a net
//! small enough to compute by hand.
use cs_decision::graph::{Graph, GraphSpec};
use cs_decision::graph_net::GraphNet;
use cs_decision::graph_prior::{softmax_weights, GraphPrior};
use cs_game::gdl_game::GdlGame;
use cs_game::Game;
use std::rc::Rc;

/// Two sorts (col, row) fused into the object `square`, marks, two roles, a scale (num); `owner` static, `cell` / `control` / `steps` state, `mine` derived.
const TINY: &str = "
(sort col 1 2) (sort row a b) (sort mark p q e) (sort side x o) (sort num (range 0 3)) (sort score 0 50 100) (sort move (put col row) noop)
(signature role side) (signature col col) (signature row row) (signature square col row) (signature owner side mark) (signature succ num num)
(signature cell col row mark) (signature control side) (signature steps num) (signature mine side col row)
(signature legal side move) (signature does side move) (signature goal side score) (signature terminal)
(role x) (role o)
(col 1) (col 2) (row a) (row b)
(<= (square ?c ?r) (col ?c) (row ?r))
(owner x p) (owner o q)
(succ 0 1) (succ 1 2) (succ 2 3)
(init (cell 1 a p)) (init (cell 2 b q)) (init (cell 2 a e)) (init (cell 1 b e)) (init (control x)) (init (steps 0))
(<= (legal ?s (put ?c ?r)) (true (control ?s)) (true (cell ?c ?r e)))
(<= (legal x noop) (true (control o)))
(<= (legal o noop) (true (control x)))
(<= (next (cell ?c ?r ?m)) (does ?s (put ?c ?r)) (owner ?s ?m))
(<= (next (cell ?c ?r ?m)) (true (cell ?c ?r ?m)) (not (touched ?c ?r)))
(<= (touched ?c ?r) (does ?s (put ?c ?r)))
(<= (next (control x)) (true (control o)))
(<= (next (control o)) (true (control x)))
(<= (next (steps ?m)) (true (steps ?n)) (succ ?n ?m))
(<= (mine ?s ?c ?r) (true (cell ?c ?r ?m)) (owner ?s ?m))
(<= terminal (true (steps 3)))
(goal x 50) (goal o 50)
";

const VOC: &str = r#"{"state": ["cell", "control", "steps"], "derived": ["mine"], "static": ["owner", "succ"], "delta": ["cell"],
                      "objects": {"square": ["col", "row"]}, "scales": ["num"], "units": null}"#;

fn spec() -> GraphSpec { GraphSpec::new(TINY, VOC).unwrap() }

fn game() -> GdlGame { GdlGame::load("tiny", TINY).unwrap() }

fn term_index(g: &Graph, tok: &str) -> usize { g.terms.iter().position(|t| t == tok).unwrap_or_else(|| panic!("no term {tok}: {:?}", g.terms)) }

#[test]
fn vocabularies_are_constant_and_typed() {
    let sp = spec();
    let tv = sp.term_vocab();
    assert!(tv.windows(2).all(|w| w[0] < w[1]), "sorted and unique");
    for t in ["col:1", "row:a", "square:1a", "square:2b", "mark:e", "side:us", "side:them", "side:x", "side:o", "num:0", "num:3"] { assert!(tv.contains(&t.to_string()), "{t}"); }
    assert!(!tv.contains(&"num:4".to_string()));
    // the numeral 1 is a column and a clock value: different tokens
    assert!(tv.contains(&"col:1".to_string()) && tv.contains(&"num:1".to_string()));
    assert_eq!(sp.rel_vocab()[..2], ["move".to_string(), "claim".to_string()]);
    assert_eq!(sp.rel_vocab()[2..], ["cell", "control", "mine", "owner", "steps", "succ"].map(String::from));
    assert_eq!(sp.max_parts(), 2);
}

#[test]
fn constants_are_nodes_in_every_position_and_statics_are_identical() {
    let (sp, g) = (spec(), game());
    let s0 = g.initial();
    let g0 = sp.graph(&g, &s0).unwrap();
    let m = g.moves(&s0)[0];
    let s1 = g.apply(&s0, &m);
    let g1 = sp.graph(&g, &s1).unwrap();
    // all squares are nodes (fused), with part edges to their file and rank
    for sq in ["square:1a", "square:1b", "square:2a", "square:2b"] { term_index(&g0, sq); }
    let (ps, pd) = g0.edge("part0").unwrap();
    assert_eq!(ps.len(), 4);
    let i = term_index(&g0, "square:2b");
    let k = ps.iter().position(|&x| x as usize == i).unwrap();
    assert_eq!(g0.terms[pd[k] as usize], "col:2");
    let (ps, pd) = g0.edge("part1").unwrap();
    assert_eq!(g0.terms[pd[ps.iter().position(|&x| x as usize == i).unwrap()] as usize], "row:b");
    // the constant nodes and the static facts are the same prefix in two positions
    let (nt, nf) = (g0.n_static_terms, g0.n_static_facts);
    assert_eq!((nt, nf), (g1.n_static_terms, g1.n_static_facts));
    assert_eq!(g0.terms[..nt], g1.terms[..nt]);
    assert_eq!(g0.facts[..nf], g1.facts[..nf]);
    assert!(g0.facts[..nf].iter().all(|f| f[1] == 2 && f[2] == 1));
    assert_eq!(nf, 2 + 3);                                     // owner (2 rows), succ (3 rows)
}

#[test]
fn an_unmentioned_constant_is_an_edgeless_node_and_scale_members_appear_where_they_occur() {
    let g = game();
    // without the derived relation nothing mentions `them` in the first position: the node is there, with no fact edge
    let sp = GraphSpec::new(TINY, &VOC.replace(r#""derived": ["mine"]"#, r#""derived": []"#)).unwrap();
    let g0 = sp.graph(&g, &g.initial()).unwrap();
    let them = term_index(&g0, "side:them");
    for i in 0..6 {
        let (_, d) = g0.edge(&format!("arg{i}")).unwrap();
        assert!(!d.contains(&(them as u32)), "arg{i}");
    }
    // the clock value 0 occurs (steps 0 and the static succ); a scale member nothing mentions would not be a node: here every num is in succ
    term_index(&g0, "num:0");
    // the position's own facts come after the static prefix: 4 cells, control
    let dynamic = &g0.facts[g0.n_static_facts..];
    let rel = |name: &str| 1 + sp.rel_vocab().iter().position(|r| r == name).unwrap() as u32;
    assert_eq!(dynamic.iter().filter(|f| f[0] == rel("cell") && f[1] == 0 && f[2] == 1).count(), 4);
    assert_eq!(dynamic.iter().filter(|f| f[0] == rel("control")).count(), 1);
    // a cell fact: arg0 is the fused square, arg1 the mark (typed)
    let (src, dst) = g0.edge("arg0").unwrap();
    let fi = g0.n_static_facts;                                // the first dynamic fact
    let sq = dst[src.iter().position(|&f| f as usize == fi).unwrap()] as usize;
    assert!(g0.terms[sq].starts_with("square:"));
}

#[test]
fn a_move_removes_and_adds_cell_rows_in_the_movers_perspective() {
    let (sp, g) = (spec(), game());
    let s0 = g.initial();
    let g0 = sp.graph(&g, &s0).unwrap();
    assert_eq!(g0.moves.len(), 2);
    let names: Vec<String> = g.moves(&s0).iter().map(|m| g.move_name(&s0, m)).collect();
    assert_eq!(g0.moves, names);
    let (ms, mf) = g0.edge("removes").unwrap();
    assert_eq!(ms.len(), 2);
    let (as_, af) = g0.edge("adds").unwrap();
    assert_eq!(as_.len(), 2);
    // the added rows are not in the position (present = 0), the removed ones are
    assert!(af.iter().all(|&f| g0.facts[f as usize][2] == 0));
    assert!(mf.iter().all(|&f| g0.facts[f as usize][2] == 1));
    // x moves: it adds a cell holding p (its mark)
    let (src, dst) = g0.edge("arg1").unwrap();
    let f = af[0];
    let mark = dst[src.iter().position(|&x| x == f).unwrap()] as usize;
    assert_eq!(g0.terms[mark], "mark:p");
}

#[test]
fn a_term_outside_its_sort_is_an_error() {
    let bad = TINY.replace("(init (cell 1 a p))", "(init (cell 1 a z))");
    let g = GdlGame::load("bad", &bad).unwrap();
    assert!(GraphSpec::new(&bad, VOC).unwrap().graph(&g, &g.initial()).is_err());
    assert!(GraphSpec::new(TINY, r#"{"state": ["nosuch"]}"#).is_err());
}

fn net_json(dim: usize, vocab: (Vec<String>, Vec<String>), max_parts: usize, conv: &str, norm: &str, emb: [&[f32]; 4], head: &str) -> String {
    let (tv, rv) = vocab;
    let l = |v: &[f32]| format!("{v:?}");
    format!(r#"{{"format": "graph_net", "dim": {dim}, "layers": 1, "max_arity": 6, "max_parts": {max_parts}, "vocab": {{"term": {tv:?}, "rel": {rv:?}}},
        "emb": {{"term": {}, "rel": {}, "kind": {}, "present": {}, "prob_w": [0,0], "prob_b": [0,0]}}, "conv": [{conv}], "norm": [{norm}], "head": {head}}}"#,
        l(emb[0]), l(emb[1]), l(emb[2]), l(emb[3]))
}

/// A one-layer, dim-2 net on a hand-built graph: a term (id 1), a fact (relation r = id 3, kind 0, present 1) and a move.
///   x0: term [1,0]; fact rel[3] + kind[0] + present[1] = [0,1] + [0,0] + [1,0] = [1,1]; move rel[1] = [1,0]
///   convs (lin_r = 0, biases = 0):  fact->term (arg0) = I: term out = mean of fact x = [1,1];  term->fact (rev_arg0) = 2 I: fact out = 2 [1,0] = [2,0];
///   move->fact (removes) = I: fact out += [1,0] -> [3,0];  fact->move (rev_removes) = I: move out = fact x = [1,1]
///   LayerNorm (w = 1): term [1,1] -> 0 (var 0), b = 0 -> relu 0, term stays [1,0]; fact [3,0] -> (+-1.5) / sqrt(2.25 + 1e-5) = [0.9999978, -0.9999978], b = 0 -> relu [0.9999978, 0];
///   move [1,1] -> 0, b = [1,-1] -> relu [1,0], move x = [1,0] + [1,0] = [2,0]
///   head: h = relu(W0 x + b0), W0 = [[1,1],[0,0]], b0 = [-1,0]: h = [1, 0]; score = 3*1 + 5*0 + 0.25 = 3.25
#[test]
fn forward_by_hand() {
    let id = "[1,0,0,1]";
    let two = "[2,0,0,2]";
    let zero = "[0,0,0,0]";
    let cv = |l: &str| format!(r#"{{"lin_l_w": {l}, "lin_l_b": [0,0], "lin_r_w": {zero}}}"#);
    let conv = format!(r#"{{"fact__arg0__term": {}, "term__rev_arg0__fact": {}, "move__removes__fact": {}, "fact__rev_removes__move": {}}}"#, cv(id), cv(two), cv(id), cv(id));
    let norm = r#"{"term": {"w": [1,1], "b": [0,0]}, "fact": {"w": [1,1], "b": [0,0]}, "move": {"w": [1,1], "b": [1,-1]}}"#;
    let term = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0];                  // UNK, term:1 (id 1)... ids 0..2: [UNK, "t:a", "t:b"]
    let rel = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];         // UNK, move [1,0], claim, r [0,1]
    let kind = [0.0; 6];
    let present = [0.0, 0.0, 1.0, 0.0];                         // present 1 -> [1,0]
    let json = net_json(2, (vec!["t:a".into(), "t:b".into()], vec!["move".into(), "claim".into(), "r".into()]), 0, &conv, norm, [&term, &rel, &kind, &present],
                        r#"{"w0": [1,1,0,0], "b0": [-1,0], "w1": [3,5], "head_bias": [0.25]}"#);
    let net = GraphNet::from_json(&json).unwrap();
    let mut gr = Graph::default();
    gr.terms = vec!["t:a".into()];
    gr.term_ids = vec![1];
    gr.facts = vec![[3, 0, 1]];
    gr.moves = vec!["m".into()];
    gr.edges = vec![("arg0".into(), vec![0], vec![0]), ("removes".into(), vec![0], vec![0])];
    let s = net.forward(&gr);
    assert_eq!(s.len(), 1);
    assert!((s[0] - 3.25).abs() < 1e-5, "{}", s[0]);
}

#[test]
fn the_prior_scores_every_legal_move_and_refuses_foreign_weights() {
    let (sp, g) = (Rc::new(spec()), game());
    let (tv, rv) = (sp.term_vocab().to_vec(), sp.rel_vocab().to_vec());
    let dim = 4usize;
    let mut seed = 7u32;
    let mut rnd = |n: usize| -> Vec<f32> { (0..n).map(|_| { seed = seed.wrapping_mul(1664525).wrapping_add(1013904223); ((seed >> 8) as f32 / 16777216.0 - 0.5) * 0.6 }).collect() };
    let (t, r, k, p) = (rnd((tv.len() + 1) * dim), rnd((rv.len() + 1) * dim), rnd(3 * dim), rnd(2 * dim));
    let mut convs = vec![];
    let mut names: Vec<(String, String, String)> = vec![];
    for i in 0..6 { names.push(("fact".into(), format!("arg{i}"), "term".into())); }
    for i in 0..sp.max_parts() { names.push(("term".into(), format!("part{i}"), "term".into())); }
    names.extend([("move".into(), "removes".into(), "fact".into()), ("move".into(), "adds".into(), "fact".into()), ("claim".into(), "binds".into(), "fact".into())]);
    let fwd = names.clone();
    names.extend(fwd.iter().map(|(s, r, d)| (d.clone(), format!("rev_{r}"), s.clone())));
    for (s, rel, d) in &names {
        convs.push(format!(r#""{s}__{rel}__{d}": {{"lin_l_w": {:?}, "lin_l_b": {:?}, "lin_r_w": {:?}}}"#, rnd(dim * dim), rnd(dim), rnd(dim * dim)));
    }
    let conv = format!("{{{}}}", convs.join(","));
    let norm = ["term", "fact", "move", "claim"].iter().map(|t| format!(r#""{t}": {{"w": {:?}, "b": {:?}}}"#, vec![1.0f32; dim], rnd(dim))).collect::<Vec<_>>().join(",");
    let head = format!(r#"{{"w0": {:?}, "b0": {:?}, "w1": {:?}, "head_bias": [0.1]}}"#, rnd(dim * dim), rnd(dim), rnd(dim));
    let json = net_json(dim, (tv.clone(), rv.clone()), sp.max_parts(), &conv, &format!("{{{norm}}}"), [&t, &r, &k, &p], &head);
    let net = Rc::new(GraphNet::from_json(&json).unwrap());
    let prior = GraphPrior::new(sp.clone(), net).unwrap();
    let s0 = g.initial();
    let (names, scores) = prior.scores(&g, &s0).unwrap();
    assert_eq!(names.len(), 2);
    assert_eq!(scores.len(), 2);
    assert!(scores.iter().all(|x| x.is_finite()));
    let again = prior.scores(&g, &s0).unwrap();
    assert_eq!(again.1, scores, "deterministic");
    use cs_planning::rollout::Prior;
    let w = prior.weights(&g, &s0, &g.moves(&s0));
    assert!((w.iter().sum::<f64>() - 2.0).abs() < 1e-9);
    assert!((softmax_weights(&[0.0, 0.0])[0] - 1.0).abs() < 1e-12);
    assert_eq!(prior.timing().0, 3);
    // weights trained over another vocabulary are refused
    let other = GraphSpec::new(TINY, r#"{"state": ["cell"], "derived": [], "static": [], "delta": ["cell"], "objects": {"square": ["col", "row"]}, "scales": []}"#).unwrap();
    assert!(GraphPrior::new(Rc::new(other), Rc::new(GraphNet::from_json(&json).unwrap())).is_err());
}
