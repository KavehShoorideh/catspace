//! PLAN 79.57 PART A (SYSTEM_SPEC 11, blocks "A divergence found and named" and "The support filter"; LAYER 2, piece: rules, on layer 1's kept graph; ledger rows 58, 44, 40). THIS TEXT IS THE INTERFACE, written before the body.
//! THE GATE OF PART A: THE DESCRIPTION WRITES THE GRAPH. In every game that has a placed module, along a small seeded walk with make and undo, the graph kept (by `GraphKeeper`, layer 1's live graph, written through layer 2's door) from the translated module's answers
//! (`described::Described`) equals the graph made afresh from the same answers (`Static::position`), and equals the graph kept from the bridge (the kernel's answers), node for node and link for link (`PositionPart::definitions`, the equality the keeper's own tests use).
//! TAKES  the seven games' placed modules (generated_placed/<game>.rs, every pass, read only here) and the kernel of each; a walk is a few seeded games of a few tens of plies (the moves sorted by name, one drawn by the seeded generator).
//! RETURNS OBS lines: games, states, nodes and links compared, relation nodes compared, units asked again and units in all (the keeper's own count, which the hand rule gives), and the differences; the test fails on a difference.
//! WHO CALLS IT: `cargo test --release --offline -j 4 -p cs_game --test translated_graph`.
//! This file names the games on purpose (a test file may); the layer crates do not.
mod described;
mod translated_common;
mod variants_common;
use cs_game::graph::Static;
use cs_game::keeper::GraphKeeper;
use cs_game::rng::Rng;
use cs_game::rules::Ruled;
use cs_game::Game;
use described::Described;
use translated_common::*;
use variants_common::{kernel, rules_root};

include!("generated_placed/chess.rs");
include!("generated_placed/atomic.rs");
include!("generated_placed/3check.rs");
include!("generated_placed/kingofthehill.rs");
include!("generated_placed/antichess.rs");
include!("generated_placed/chess960.rs");
include!("generated_placed/horde.rs");
impl_tr!(placed_chess, passes);
impl_tr!(placed_atomic, passes);
impl_tr!(placed_threecheck, passes);
impl_tr!(placed_kingofthehill, passes);
impl_tr!(placed_antichess, passes);
impl_tr!(placed_chess960, passes);
impl_tr!(placed_horde, passes);

pub const SEED: u64 = 5757;
pub const GAMES: usize = 3;
pub const PLIES: usize = 30;

pub fn static_of(game: &str) -> Static {
    let ruled = Ruled::open_variant(kernel(game), game, &rules_root(), "learned/chess/graph_vocabulary.json", None, None).unwrap();
    ruled.board().unwrap()
}

fn gate<T: Tr>(game: &str) {
    let st = static_of(game);
    let bridge = kernel(game);
    let d = Described::new(kernel(game), T::make());
    let mut rng = Rng::new(SEED);
    let (mut states, mut nodes, mut links, mut rels, mut asked, mut all, mut diff_fresh, mut diff_bridge, mut undone) = (0usize, 0usize, 0usize, 0usize, 0usize, 0usize, 0usize, 0usize, 0usize);
    let mut first: Vec<String> = vec![];
    for gi in 0..GAMES {
        let mut s = d.initial();
        let mut kd = GraphKeeper::new(&st, &d, &s);
        let mut kb = GraphKeeper::new(&st, &bridge, &s);
        let mut stack = vec![st.position(&d, &s).definitions(st.board())];
        for ply in 0..PLIES {
            let mv = d.moves(&s);
            if mv.is_empty() || d.outcome(&s).is_some() { break; }
            let mut sorted: Vec<(String, usize)> = mv.iter().enumerate().map(|(i, m)| (d.move_name(&s, m), i)).collect();
            sorted.sort();
            let (name, i) = sorted[rng.below(sorted.len())].clone();
            s = d.apply(&s, &mv[i]);
            kd.make(&st, &d, &s);
            kb.make(&st, &bridge, &s);
            let (a, n) = kd.asked();
            asked += a;
            all += n;
            let kept = kd.part(&st);
            let fresh = st.position(&d, &s);
            let (dk, df, db) = (kept.definitions(st.board()), fresh.definitions(st.board()), kb.part(&st).definitions(st.board()));
            states += 1;
            nodes += dk.0.len();
            links += dk.1.len();
            rels += kept.relation_count();
            if dk != df { diff_fresh += 1; if first.len() < 4 { first.push(format!("{game} game {gi} ply {ply} after {name}: kept from the description differs from fresh from the description ({} nodes against {})", dk.0.len(), df.0.len())); } }
            if dk != db { diff_bridge += 1; if first.len() < 4 { first.push(format!("{game} game {gi} ply {ply} after {name}: kept from the description differs from kept from the bridge: only in the description {:?}, only in the bridge {:?}", dk.0.iter().filter(|x| !db.0.contains(x)).take(4).collect::<Vec<_>>(), db.0.iter().filter(|x| !dk.0.contains(x)).take(4).collect::<Vec<_>>())); } }
            stack.push(df);
        }
        // taken back: at each depth the kept graph is the graph made afresh at that depth
        while stack.len() > 1 {
            stack.pop();
            assert!(kd.undo() && kb.undo());
            let back = kd.part(&st).definitions(st.board());
            undone += 1;
            if &back != stack.last().unwrap() { diff_fresh += 1; if first.len() < 4 { first.push(format!("{game} game {gi}: after an undo at depth {} the kept graph differs from the graph made afresh", stack.len() - 1)); } }
            if back != kb.part(&st).definitions(st.board()) { diff_bridge += 1; }
        }
    }
    let (b, r, w) = d.asked();
    println!("OBS 79.57 A {game}: {GAMES} games of at most {PLIES} plies (seed {SEED}), {states} states, {nodes} nodes and {links} links compared, {rels} relation nodes, {undone} undone depths; the keeper asked {asked} units again of {all} (hand rule); the module was asked at {w} states ({b} bearing questions, {r} reach questions); kept-from-description against fresh-from-description differs at {diff_fresh}, against kept-from-bridge differs at {diff_bridge}");
    for f in &first { println!("FINDING 79.57 A {f}"); }
    assert!(states > 0);
    assert_eq!((diff_fresh, diff_bridge), (0, 0), "{first:?}");
}

#[test] fn chess_the_graph_kept_from_the_description_is_the_fresh_graph_and_the_bridges() { gate::<placed_chess::Game>("chess"); }
#[test] fn atomic_the_graph_kept_from_the_description_is_the_fresh_graph_and_the_bridges() { gate::<placed_atomic::Game>("atomic"); }
#[test] fn threecheck_the_graph_kept_from_the_description_is_the_fresh_graph_and_the_bridges() { gate::<placed_threecheck::Game>("3check"); }
#[test] fn kingofthehill_the_graph_kept_from_the_description_is_the_fresh_graph_and_the_bridges() { gate::<placed_kingofthehill::Game>("kingofthehill"); }
#[test] fn antichess_the_graph_kept_from_the_description_is_the_fresh_graph_and_the_bridges() { gate::<placed_antichess::Game>("antichess"); }
#[test] fn chess960_the_graph_kept_from_the_description_is_the_fresh_graph_and_the_bridges() { gate::<placed_chess960::Game>("chess960"); }
#[test] fn horde_the_graph_kept_from_the_description_is_the_fresh_graph_and_the_bridges() { gate::<placed_horde::Game>("horde"); }
