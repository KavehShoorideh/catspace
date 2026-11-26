//! PLAN 79.57 PARTS B AND C (SYSTEM_SPEC 11, blocks "A divergence found and named" and "The support filter"; LAYER 2, piece: rules, on layer 1's kept graph; ledger rows 58, 57, 44, 40). THIS TEXT IS THE INTERFACE, written before the body.
//! The modules here are the placed modules of the games made WITH pass (27) "reads" (generated_placed/<game>_reads.rs): the answer of a unit's bearing or reach comes with the facts of the state the walk read (`Game::ask_reads`).
//!   B. THE KEPT GRAPH UNDER THE SUPPORT RULE. TAKES the module of each game, its kernel, and a seeded walk with make and undo (a few games of a few tens of plies). `Described` (tests/described) answers the keeper's questions from the module and hands the places the reads bound as each answer's support;
//!      `GraphKeeper::new_supported` and `make_supported` (src/keeper.rs) ask a unit again only when it is new or a changed place is in its support. GATE: at every state of the walk and after every undo the graph kept under the support rule equals the graph made afresh from the module, and equals the graph kept under the hand rule;
//!      the per-unit answers equal the bulk answers; the fluents the reads name are printed (a fluent other than the one that holds the units, or the fixed ones of a side's own question, would make a support `all`). A PLANTED FAULT: one place struck out of every support by hand is caught by the equality with the fresh graph.
//!   C. THE MEASURE (`#[ignore]`, by hand: `CS_FACTS_GAMES=<file> cargo test --release --offline -j 4 -p cs_game --test translated_reads -- --ignored --nocapture facts_moved`): along the games of a file (the games file of the per-ply measure; one game a line, the moves by the outside library's names; no player is read) the relation rows (bear, may_move) on the graph a state;
//!      under each rule the rows kept and the rows derived again a move; the rows that are new after the move (what the move disturbed). OBS lines only. And `facts`: `CS_FACTS_FEN=<position text> cargo test ... -- --ignored --nocapture facts` prints the graph of a state made from the description, units with what each bears on and may move to.
//! RETURNS OBS lines; the gates fail on a difference. This file names the games on purpose (a test file may); the layer crates do not.
mod described;
mod translated_common;
mod variants_common;
use cs_game::graph::Static;
use cs_game::graph_facts::FactReader;
use cs_game::keeper::{GraphKeeper, Support, Supported};
use cs_game::kernel::KernelGame;
use cs_game::placed::Placed;
use cs_game::relations::UnitRelations;
use cs_game::rng::Rng;
use cs_game::rules::{Ruled, StateText};
use cs_game::{Game, Role};
use described::Described;
use std::collections::BTreeSet;
use translated_common::*;
use variants_common::{kernel, rules_root};

include!("generated_placed/chess_reads.rs");
include!("generated_placed/atomic_reads.rs");
include!("generated_placed/3check_reads.rs");
include!("generated_placed/kingofthehill_reads.rs");
include!("generated_placed/antichess_reads.rs");
include!("generated_placed/chess960_reads.rs");
include!("generated_placed/horde_reads.rs");
impl_tr!(placed_chess_reads, reads);
impl_tr!(placed_atomic_reads, reads);
impl_tr!(placed_threecheck_reads, reads);
impl_tr!(placed_kingofthehill_reads, reads);
impl_tr!(placed_antichess_reads, reads);
impl_tr!(placed_chess960_reads, reads);
impl_tr!(placed_horde_reads, reads);

pub const SEED: u64 = 5757;
pub const GAMES: usize = 3;
pub const PLIES: usize = 30;

fn static_of(game: &str) -> Static {
    let ruled = Ruled::open_variant(kernel(game), game, &rules_root(), "learned/chess/graph_vocabulary.json", None, None).unwrap();
    ruled.board().unwrap()
}

type Row = ((usize, u8, u16), u8, u16);
/// Every bear and may_move row of a kept graph, as (unit key, 0 bear or 1 may_move, place).
fn rows_of(k: &GraphKeeper) -> BTreeSet<Row> {
    let mut v = BTreeSet::new();
    for u in k.live().units() {
        for &p in u.bears() { v.insert((u.key(), 0, p)); }
        for &p in u.reach() { v.insert((u.key(), 1, p)); }
    }
    v
}

/// What a walk adds up: per rule the units asked again and in all, the rows derived again and in all; and the rows that are new after a move.
#[derive(Default, Debug, Clone)]
struct Tally {
    states: usize, moves: usize,
    hand_units: (usize, usize), hand_rows: (usize, usize),
    sup_units: (usize, usize), sup_rows: (usize, usize),
    rows_on_graph: usize, rows_new: usize, rows_gone: usize,
    diff_fresh: usize, diff_hand: usize, diff_unit: usize, undone: usize, all_supports: usize, supports: usize, support_places: usize,
    first: Vec<String>,
}

/// Where the games come from: seeded random walks, or the moves of a file's games (the outside library's names).
enum Source { Seeded(usize, usize), Lines(Vec<Vec<String>>, usize) }

fn games_of<T: Tr + 'static>(game: &str, src: &Source, d: &Described<T>) -> Vec<Vec<<KernelGame as Game>::State>> {
    // the states of each game from its start, each following the one before
    let mut out = vec![];
    match src {
        Source::Seeded(n, plies) => {
            let mut rng = Rng::new(SEED);
            for _ in 0..*n {
                let mut s = d.initial();
                let mut states = vec![s.clone()];
                for _ in 0..*plies {
                    let mv = d.moves(&s);
                    if mv.is_empty() || d.outcome(&s).is_some() { break; }
                    let mut sorted: Vec<(String, usize)> = mv.iter().enumerate().map(|(i, m)| (d.move_name(&s, m), i)).collect();
                    sorted.sort();
                    let (_, i) = sorted[rng.below(sorted.len())].clone();
                    s = d.apply(&s, &mv[i]);
                    states.push(s.clone());
                }
                out.push(states);
            }
        }
        Source::Lines(lines, max) => {
            let _ = game;
            for l in lines {
                let mut s = d.initial();
                let mut states = vec![s.clone()];
                for nm in l.iter().take(*max) {
                    if d.outcome(&s).is_some() { break; }
                    let Some(m) = d.kernel.find_move(&s, nm) else { break };
                    s = d.apply(&s, &m);
                    states.push(s.clone());
                }
                out.push(states);
            }
        }
    }
    out
}

/// THE GATE AND THE MEASURE: every game's states are walked by three keepers: the hand rule on the bridge's answers (the reference the graph is held to), the hand rule on the description's answers, and the support rule on the description's answers with reads.
/// `strike`: one place struck out of every support by hand after every tempo (the planted fault).
fn run<T: Tr + 'static>(game: &str, src: &Source, strike: bool) -> (Tally, std::collections::BTreeMap<String, usize>) {
    let st = static_of(game);
    let bridge = kernel(game);
    let dh = Described::new(kernel(game), T::make());        // the hand rule, the description's answers (bulk)
    let ds = Described::new(kernel(game), T::make());        // the support rule, the description's answers by unit, with reads
    let games = games_of(game, src, &ds);
    let mut t = Tally::default();
    for (gi, states) in games.iter().enumerate() {
        let s0 = &states[0];
        let mut kb = GraphKeeper::new(&st, &bridge, s0);
        let mut kh = GraphKeeper::new(&st, &dh, s0);
        let mut ks = GraphKeeper::new_supported(&st, &ds, s0);
        let strike_all = |k: &mut GraphKeeper| { for v in k.supports_mut().values_mut() { if !v.all && !v.places.is_empty() { let m = v.places.len() / 2; v.places.remove(m); } } };
        if strike { strike_all(&mut ks); }
        let mut stack = vec![st.position(&dh, s0).definitions(st.board())];
        let mut prev = rows_of(&ks);
        for (ply, s) in states.iter().enumerate().skip(1) {
            kb.make(&st, &bridge, s);
            kh.make(&st, &dh, s);
            ks.make_supported(&st, &ds, s);
            if strike { strike_all(&mut ks); }
            t.states += 1;
            t.moves += 1;
            let (a, n) = kh.asked();
            t.hand_units.0 += a; t.hand_units.1 += n;
            let (a, n) = ks.asked();
            t.sup_units.0 += a; t.sup_units.1 += n;
            let (r, n) = kh.rows_again();
            t.hand_rows.0 += r; t.hand_rows.1 += n;
            let (r, n) = ks.rows_again();
            t.sup_rows.0 += r; t.sup_rows.1 += n;
            let now = rows_of(&ks);
            t.rows_on_graph += now.len();
            t.rows_new += now.difference(&prev).count();
            t.rows_gone += prev.difference(&now).count();
            prev = now;
            let fresh = st.position(&dh, s).definitions(st.board());
            let (dk_s, dk_h, dk_b) = (ks.part(&st).definitions(st.board()), kh.part(&st).definitions(st.board()), kb.part(&st).definitions(st.board()));
            if dk_s != fresh { t.diff_fresh += 1; if t.first.len() < 4 { t.first.push(format!("{game} game {gi} ply {ply}: the graph kept under the support rule differs from the fresh graph ({} nodes against {})", dk_s.0.len(), fresh.0.len())); } }
            if dk_s != dk_h || dk_s != dk_b { t.diff_hand += 1; }
            if ks.live().units().len() != kh.live().units().len() { t.diff_hand += 1; }
            for v in ks.live().units().iter().map(|u| ks.support_of(&u.key())) { if let Some(v) = v { t.supports += 1; if v.all { t.all_supports += 1; } t.support_places += v.places.len(); } }
            stack.push(fresh);
        }
        while stack.len() > 1 {
            stack.pop();
            assert!(ks.undo() && kh.undo() && kb.undo());
            t.undone += 1;
            if &ks.part(&st).definitions(st.board()) != stack.last().unwrap() { t.diff_fresh += 1; if t.first.len() < 4 { t.first.push(format!("{game} game {gi}: after an undo at depth {} the graph kept under the support rule differs from the fresh graph", stack.len() - 1)); } }
        }
    }
    // the answers by unit against the bulk answers, at every state of the walk
    for states in &games {
        for s in states {
            let rows = ds.rows(s);
            for &(o, k, p) in &rows.cells {
                let (b, sb) = ds.bearing_with_support(s, o, k, p);
                let (r, sr) = ds.reach_with_support(s, o, k, p);
                if Some(&b) != rows.bears.get(&(o, k, p)) && !(b.is_empty() && !rows.bears.contains_key(&(o, k, p))) { t.diff_unit += 1; }
                if Some(&r) != rows.reaches.get(&(o, k, p)) && !(r.is_empty() && !rows.reaches.contains_key(&(o, k, p))) { t.diff_unit += 1; }
                assert!(sb.all || sb.places.binary_search(&p).is_ok(), "the unit's own place is in the support of its bearing");
                assert!(sr.all || sr.places.binary_search(&p).is_ok(), "the unit's own place is in the support of its reach");
            }
        }
    }
    let seen = ds.fluents_read.borrow().clone();
    (t, seen)
}

fn gate<T: Tr + 'static>(game: &str) {
    let (t, seen) = run::<T>(game, &Source::Seeded(GAMES, PLIES), false);
    let (f, _) = run::<T>(game, &Source::Seeded(GAMES, PLIES), true);
    println!("OBS 79.57 B {game}: {GAMES} games of at most {PLIES} plies (seed {SEED}), {} states, {} undone depths; under the support rule the kept graph differs from the fresh graph at {} states, from the hand rule's at {}; the answers by unit differ from the bulk answers {} times; supports: {} kept, {} with no bound (all), {:.1} places on average; fluents read by the walks: {:?}",
        t.states, t.undone, t.diff_fresh, t.diff_hand, t.diff_unit, t.supports, t.all_supports, t.support_places as f64 / t.supports.max(1) as f64, seen);
    println!("OBS 79.57 B {game}: units asked again of units held a move, hand rule {} of {}, support rule {} of {}; rows derived again of rows on the graph, hand rule {} of {}, support rule {} of {}",
        t.hand_units.0, t.hand_units.1, t.sup_units.0, t.sup_units.1, t.hand_rows.0, t.hand_rows.1, t.sup_rows.0, t.sup_rows.1);
    println!("OBS 79.57 B {game} planted fault (one place struck out of every support): the kept graph differs from the fresh graph at {} of {} checks (every state of the walk and every depth taken back)", f.diff_fresh, f.states + f.undone);
    for x in &t.first { println!("FINDING 79.57 B {x}"); }
    assert!(t.states > 0);
    assert_eq!((t.diff_fresh, t.diff_hand, t.diff_unit), (0, 0, 0), "{:?}", t.first);
    assert!(f.diff_fresh > 0, "{game}: a support with a place struck out by hand was not caught");
}

#[test] fn chess_the_graph_kept_under_the_support_rule_is_the_fresh_graph() { gate::<placed_chess_reads::Game>("chess"); }
#[test] fn atomic_the_graph_kept_under_the_support_rule_is_the_fresh_graph() { gate::<placed_atomic_reads::Game>("atomic"); }
#[test] fn threecheck_the_graph_kept_under_the_support_rule_is_the_fresh_graph() { gate::<placed_threecheck_reads::Game>("3check"); }
#[test] fn kingofthehill_the_graph_kept_under_the_support_rule_is_the_fresh_graph() { gate::<placed_kingofthehill_reads::Game>("kingofthehill"); }
#[test] fn antichess_the_graph_kept_under_the_support_rule_is_the_fresh_graph() { gate::<placed_antichess_reads::Game>("antichess"); }
#[test] fn chess960_the_graph_kept_under_the_support_rule_is_the_fresh_graph() { gate::<placed_chess960_reads::Game>("chess960"); }
#[test] fn horde_the_graph_kept_under_the_support_rule_is_the_fresh_graph() { gate::<placed_horde_reads::Game>("horde"); }

fn env_usize(k: &str, d: usize) -> usize { std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d) }

/// PART C, the measure of how many facts a move disturbs, along the games of a file (the first game of the first game's module).
#[test]
#[ignore = "a measure, by hand: CS_FACTS_GAMES names the games file; walks of a few tens of plies"]
fn facts_moved() {
    let path = std::env::var("CS_FACTS_GAMES").expect("CS_FACTS_GAMES names the file of games");
    let n = env_usize("CS_FACTS_N", 30);
    let max = env_usize("CS_FACTS_MAX", 40);
    let lines: Vec<Vec<String>> = std::fs::read_to_string(&path).unwrap().lines().take(n).map(|l| l.split_whitespace().map(|w| w.to_string()).collect()).collect();
    let t0 = std::time::Instant::now();
    let (t, seen) = run::<placed_chess_reads::Game>("chess", &Source::Lines(lines.clone(), max), false);
    let per = |x: usize| x as f64 / t.moves.max(1) as f64;
    println!("OBS 79.57 C games: {} games of at most {max} plies from the games file, {} moves made ({:.1} s); fluents read: {:?}", lines.len(), t.moves, t0.elapsed().as_secs_f64(), seen);
    println!("OBS 79.57 C rows on the graph a state (bear and may_move): {:.1}; units in play a state: {:.1}", per(t.rows_on_graph), per(t.hand_units.1));
    println!("OBS 79.57 C hand rule, a move: units asked again {:.2} of {:.1}; rows derived again {:.1}, rows kept {:.1}", per(t.hand_units.0), per(t.hand_units.1), per(t.hand_rows.0), per(t.hand_rows.1 - t.hand_rows.0));
    println!("OBS 79.57 C support rule, a move: units asked again {:.2} of {:.1}; rows derived again {:.1}, rows kept {:.1}", per(t.sup_units.0), per(t.sup_units.1), per(t.sup_rows.0), per(t.sup_rows.1 - t.sup_rows.0));
    println!("OBS 79.57 C rows new after a move {:.1} (rows that went {:.1}): what the move disturbed; the rows kept stand and equal the fresh graph at every state (differences from fresh {}, from the hand rule {}, undone depths {})", per(t.rows_new), per(t.rows_gone), t.diff_fresh, t.diff_hand, t.undone);
    println!("OBS 79.57 C share of rows predicted to stand (kept and equal to the fresh graph): hand rule {:.3}, support rule {:.3}; rows that stand in fact (not new): {:.3}", 1.0 - t.hand_rows.0 as f64 / t.hand_rows.1.max(1) as f64, 1.0 - t.sup_rows.0 as f64 / t.sup_rows.1.max(1) as f64, 1.0 - t.rows_new as f64 / t.rows_on_graph.max(1) as f64);
    println!("OBS 79.57 C supports: {} kept, {} with no bound, {:.1} places on average", t.supports, t.all_supports, t.support_places as f64 / t.supports.max(1) as f64);
    assert_eq!((t.diff_fresh, t.diff_hand), (0, 0));
}

/// `facts`: the graph of one state made from the description, units with what each bears on and may move to (CS_FACTS_FEN: the position text, default the start; CS_FACTS_DUMP=1 also prints layer 1's dump of the graph).
#[test]
#[ignore = "a command, by hand: CS_FACTS_FEN=<position> cargo test --release --offline -j 4 -p cs_game --test translated_reads -- --ignored --nocapture facts"]
fn facts() {
    let st = static_of("chess");
    let ruled = Ruled::open_variant(kernel("chess"), "chess", &rules_root(), "learned/chess/graph_vocabulary.json", None, None).unwrap();
    let d = Described::new(kernel("chess"), <placed_chess_reads::Game as Tr>::make());
    // the state: a position text (CS_FACTS_FEN), or the first game of a games file after CS_FACTS_PLY moves (CS_FACTS_GAMES), or the start
    let s: <KernelGame as Game>::State = match (std::env::var("CS_FACTS_FEN"), std::env::var("CS_FACTS_GAMES")) {
        (Ok(t), _) => ruled.read_state_text(&t).expect("a position text"),
        (_, Ok(p)) => {
            let line = std::fs::read_to_string(p).unwrap().lines().next().unwrap().to_string();
            let mut s = d.initial();
            for nm in line.split_whitespace().take(env_usize("CS_FACTS_PLY", 12)) { let m = d.kernel.find_move(&s, nm).expect("a move of the game"); s = d.apply(&s, &m); }
            s
        }
        _ => d.initial(),
    };
    println!("STATE {}", d.kernel.fen(&s));
    let keeper = GraphKeeper::new_supported(&st, &d, &s);
    let rows = FactReader::new(st.board(), None).facts(keeper.live());
    println!("FACTS from the description (the compiled placed module of the first game), one state");
    for f in rows.iter().filter(|f| f.relation == "tempo") { println!("tempo {}", f.args.join(" ")); }
    let units: Vec<&cs_game::Fact> = rows.iter().filter(|f| f.relation == "unit").collect();
    for u in &units {
        let to = |rel: &str| -> Vec<&str> { rows.iter().filter(|f| f.relation == rel && f.args[..3] == u.args[..3]).map(|f| f.args[3].as_str()).collect() };
        println!("{} {} on {}: bears on [{}]; may move to [{}]", u.args[0], u.args[1], u.args[2], to("bear").join(" "), to("may_move").join(" "));
    }
    let relation_rows = rows.iter().filter(|f| f.relation == "bear" || f.relation == "may_move").count();
    println!("{} units, {} bear and may_move rows", units.len(), relation_rows);
    if std::env::var("CS_FACTS_DUMP").is_ok() { println!("{}", keeper.part(&st).dump(st.board())); }
    let sup = |u: &cs_game::Fact| -> String { let k = d.kernel.kind_names().iter().position(|x| x == &u.args[1]).unwrap() as u8; let o = d.kernel.roles().iter().position(|x| x == &u.args[0]).unwrap(); let p = d.kernel.place_names().iter().position(|x| x == &u.args[2]).unwrap() as u16; match keeper.support_of(&(o, k, p)) { Some(Support { all: true, .. }) => "all".to_string(), Some(v) => format!("{} places", v.places.len()), None => "none".to_string() } };
    if let Some(u) = units.first() { println!("support of the first unit: {}", sup(u)); }
    let _: Option<Role> = None;
}
