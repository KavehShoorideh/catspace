//! PLAN 79.57 PARTS B AND C (SYSTEM_SPEC 11, blocks "A divergence found and named" and "The support filter"; LAYER 2, piece: rules, on layer 1's kept graph; ledger rows 58, 57, 44, 40). THIS TEXT IS THE INTERFACE, written before the body.
//! The modules here are the placed modules of the games made WITH pass (27) "reads" (generated_placed/<game>_reads.rs): the answer of a unit's bearing or reach comes with the facts of the state the walk read (`Game::ask_reads`).
//!   B. THE KEPT GRAPH UNDER THE SUPPORT RULE. TAKES the module of each game, its kernel, and a seeded walk with make and undo (a few games of a few tens of plies). `Described` (tests/described) answers the keeper's questions from the module and hands the places the reads bound as each answer's support;
//!      `GraphKeeper::new_supported` and `make_supported` (src/keeper.rs) ask a unit again only when it is new or a changed place is in its support. GATE: at every state of the walk and after every undo the graph kept under the support rule equals the graph made afresh from the module, and equals the graph kept under the hand rule;
//!      the per-unit answers equal the bulk answers; the fluents the reads name are printed (a fluent other than the one that holds the units, or the fixed ones of a side's own question, would make a support `all`). A PLANTED FAULT: one place struck out of every support by hand is caught by the equality with the fresh graph.
//!   C. THE MEASURE (`#[ignore]`, by hand: `CS_FACTS_GAMES=<file> cargo test --release --offline -j 4 -p cs_game --test translated_reads -- --ignored --nocapture facts_moved`): along the games of a file (the games file of the per-ply measure; one game a line, the moves by the outside library's names; no player is read) the relation rows (bear, may_move) on the graph a state;
//!      under each rule the rows kept and the rows derived again a move; the rows that are new after the move (what the move disturbed). OBS lines only. And `facts`: `CS_FACTS_FEN=<position text> cargo test ... -- --ignored --nocapture facts` prints the graph of a state made from the description, units with what each bears on and may move to.
//! PLAN 79.58 PART A (MEASURE ONLY; SYSTEM_SPEC 11, blocks "A divergence found and named" and "The support filter"; LAYER 2, piece: rules; ledger rows 58, 57, 44). WRITTEN BEFORE THE BODY. Three ignored measures, by hand, printed by scripts/facts_moved_print.py (`classes`, `clock`, `other`); OBS lines only.
//!   `classes` (A1): along the games of a file, every unit the support rule asked again after a move is put in one class: (a) the unit is new by key (the unit that moved), (b) a place of its support changed and its answer changed, (c) a place of its support changed and its answer is THE SAME. For each class the rows derived again, how many of them are rows new on the graph and how many stood;
//!      for (c) and for (b) what a FINER SUPPORT would have saved: a support of rows (each read of the cell fluent kept with what stood there: the read is compared by the cells it matches before and after the move, `Described::reads_raw`, `cell_pattern`, `matching`) in place of places, and a support BY ROW of the answer (the reads of the one row, asked with all
//!      six places bound) in place of by unit. A unit whose every read matches the same cells before and after MUST have the same answer (the lemma): a unit that breaks it is counted and fails the run. TAKES CS_FACTS_GAMES (the file), CS_FACTS_N, CS_FACTS_MAX, CS_FACTS_JOBS (threads, default all cores).
//!   `clock` (A2): on a quiet machine, three walks of the same games, the load printed; nanoseconds for (i) one unit's question with its reads recorded (the module's `ask_reads`, key of four places, state already loaded), (ii) the same question from the bulk piece (the module's `ask_t` of the whole state, the share of a unit; and with the key bound), (iii) a whole `make` of the keeper: hand rule asking the bridge, hand rule asking the description (bulk), support rule asking the description.
//!   `other` (A3): the count of C (`facts_moved`) for each of the six other games on seeded walks (CS_FACTS_N games of at most CS_FACTS_MAX plies, seed SEED), one line a game.
//! PLAN 79.58 PART B (THE ALLOWED MOVES AS FACTS WITH SUPPORTS, THE LAG OF TWO; SYSTEM_SPEC 11 last blocks; ledger rows 58, 57, 44). WRITTEN BEFORE THE BODY. The fourth relation of layer 2 (`relations::ALLOWED`, slot 4 of layer 1's door), kept by `keeper::AllowedKeeper` (see its header), asked of the description through `Described` (`Allowing`: the module's `placed_allowed` with its reads).
//!   GATES (`allowed_gate`, every game, seeded walks of GAMES games of at most PLIES plies, make then undo): at every state and after every undo (1) the rows the keeper holds (the lag of two) equal the rows of a fresh question of every unit of the side to act, (2) those equal the ends of the moves the outside library gives (the kernel's `move_ends`, the unit that stands there),
//!      (3) the graph kept (the hand-asked rows of bear and may_move, with the allowed rows written through the door) equals the graph made afresh; and a PLANTED FAULT (one read struck out of every allowed support after every tempo) is caught by (1).
//!   B4 `allowed` (ignored; scripts/facts_moved_print.py `allowed`): over the games of a file and the seeded walks of the other games, the allowed rows a state, the units and rows asked again and kept a tempo with the lag of two, the share predicted to stand (kept) and the share that stands in fact (rows also held two tempi before), split by whether the goal unit of the side to act (the unit of the kind named `king` in this file) is borne on by the other side.
//!   B5 `states` (ignored): twenty such states printed in full: the position text, the bearing on the goal unit and the places it rested on, the units asked again, the allowed rows; and `facts` prints the allowed rows and, for each reach row that is not allowed, the reads of its refusal (`placed_allowed` with all five places bound).
//! RETURNS OBS lines; the gates fail on a difference. This file names the games on purpose (a test file may); the layer crates do not.
mod described;
mod translated_common;
mod variants_common;
use cs_game::graph::Static;
use cs_game::graph_facts::FactReader;
use cs_game::keeper::{AllowedKeeper, Allowing, GraphKeeper, Support, Supported};
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

// ------------------------------------------------------------------------------------------------------------------------------ PLAN 79.58 PART A

/// What A1 adds up over a walk. Rows are bear and may_move rows of the units asked again by the support rule (they add up to `rows_again` of the keeper).
#[derive(Default, Debug, Clone)]
struct Classes {
    moves: usize, kept_units: usize, kept_rows: usize,
    /// (a) new by key
    a_units: usize, a_rows: usize,
    /// (b) touched, answer changed; (c) touched, answer the same: units, rows derived again, rows of them that are new on the graph, rows that stood
    b_units: usize, b_rows: usize, b_new: usize, b_stood: usize,
    c_units: usize, c_rows: usize, c_new: usize, c_stood: usize,
    /// finer support by rows of the cell fluent: units of (c) it would have kept, their rows; units of (b) it would also have kept (must be 0)
    c_fine_units: usize, c_fine_rows: usize, b_fine_wrong: usize,
    /// units whose reads cannot be compared (a fluent other than the cell's, or a term not known)
    opaque_units: usize,
    /// of the units still asked again after the finer unit support: the rows that stood (in the old and in the new answer) and those of them whose own six-place reads are untouched (kept by a support by row), by class
    b_left_stood: usize, b_row_saved: usize, c_left_stood: usize, c_row_saved: usize,
    b_left_units: usize, c_left_units: usize, b_left_rows: usize, c_left_rows: usize,
    row_asks: usize,
    /// units of (c) all of whose changed places in the support are places where one unit stood before and another after (a taking)
    c_replaced: usize,
    /// a row of the old answer that went although its own reads are untouched (must be 0: the lemma)
    row_wrong: usize,
}

impl Classes {
    fn add(&mut self, o: &Classes) {
        macro_rules! f { ($($x:ident),*) => { $( self.$x += o.$x; )* } }
        f!(moves, kept_units, kept_rows, a_units, a_rows, b_units, b_rows, b_new, b_stood, c_units, c_rows, c_new, c_stood, c_fine_units, c_fine_rows, b_fine_wrong, opaque_units, b_left_stood, b_row_saved, c_left_stood, c_row_saved, b_left_units, c_left_units, b_left_rows, c_left_rows, row_asks, row_wrong, c_replaced);
    }
}

/// Whether every read of a list matches the same cells in the two states (None: a read that cannot be compared).
fn reads_same<T: Tr>(d: &Described<T>, reach: bool, reads: &[(String, Vec<u32>)], before: &<KernelGame as Game>::State, after: &<KernelGame as Game>::State) -> Option<bool> {
    let mut same = true;
    for (name, pat) in reads {
        if reach && (name == "control" || name == "ep_square") { continue; }      // fixed in a side's own question
        if name != "cell" { return None; }
        let p = d.cell_pattern(pat)?;
        if d.matching(before, &p) != d.matching(after, &p) { same = false; }
    }
    Some(same)
}

/// A1 over a slice of games: the states of each game from its start, the support rule's keeper walked, every unit asked again put in its class.
fn classify<T: Tr + 'static>(game: &str, games: &[Vec<<KernelGame as Game>::State>]) -> Classes {
    let st = static_of(game);
    let d = Described::new(kernel(game), T::make());
    let mut c = Classes::default();
    for states in games {
        let mut k = GraphKeeper::new_supported(&st, &d, &states[0]);
        for pair in states.windows(2) {
            let (sp, sn) = (&pair[0], &pair[1]);
            let prev: std::collections::BTreeMap<(usize, u8, u16), (Vec<u16>, Vec<u16>)> = k.live().units().iter().map(|u| (u.key(), (u.bears().clone(), u.reach().clone()))).collect();
            let sup: std::collections::BTreeMap<(usize, u8, u16), Support> = k.live().units().iter().map(|u| (u.key(), k.support_of(&u.key()).cloned().unwrap())).collect();
            k.make_supported(&st, &d, sn);
            let now: Vec<(usize, u8, u16)> = k.live().units().iter().map(|u| u.key()).collect();
            let mut changed: Vec<u16> = prev.keys().filter(|x| now.binary_search(x).is_err()).map(|x| x.2).collect();
            changed.extend(now.iter().filter(|x| !prev.contains_key(x)).map(|x| x.2));
            let (gone, came): (Vec<u16>, Vec<u16>) = (prev.keys().filter(|x| now.binary_search(x).is_err()).map(|x| x.2).collect(), now.iter().filter(|x| !prev.contains_key(x)).map(|x| x.2).collect());
            let replaced: Vec<u16> = gone.iter().filter(|p| came.contains(p)).copied().collect();
            c.moves += 1;
            let mut derived = 0usize;
            for u in k.live().units() {
                let key = u.key();
                let rows = u.bears().len() + u.reach().len();
                let Some(old) = prev.get(&key) else { c.a_units += 1; c.a_rows += rows; derived += rows; continue };
                if !sup[&key].meets(&changed) { c.kept_units += 1; c.kept_rows += rows; continue; }
                derived += rows;
                let same = old.0 == *u.bears() && old.1 == *u.reach();
                let stood = old.0.iter().filter(|p| u.bears().contains(p)).count() + old.1.iter().filter(|p| u.reach().contains(p)).count();
                // the finer support by rows of the cell fluent: the reads of the answer at the old state, compared by the cells they match in both
                let b4 = d.key4(key.0, key.1, key.2);
                let rb = d.reads_raw(sp, "placed_bears", &b4, None);
                let rr = d.reads_raw(sp, "placed_reaches", &b4, Some(key.0));
                let fine = match (rb, rr) {
                    (Some((_, a)), Some((_, b))) => match (reads_same(&d, false, &a, sp, sn), reads_same(&d, true, &b, sp, sn)) { (Some(x), Some(y)) => Some(x && y), _ => None },
                    _ => None,
                };
                if fine.is_none() { c.opaque_units += 1; }
                if same && sup[&key].places.iter().filter(|p| changed.contains(p)).all(|p| replaced.contains(p)) { c.c_replaced += 1; }
                if same { c.c_units += 1; c.c_rows += rows; c.c_new += rows - stood; c.c_stood += stood; } else { c.b_units += 1; c.b_rows += rows; c.b_new += rows - stood; c.b_stood += stood; }
                if fine == Some(true) {
                    if same { c.c_fine_units += 1; c.c_fine_rows += rows; } else { c.b_fine_wrong += 1; }
                    continue;
                }
                // left after the finer unit support: the rows that stood, each with its own reads (all six places bound)
                if same { c.c_left_units += 1; c.c_left_rows += rows; } else { c.b_left_units += 1; c.b_left_rows += rows; }
                for (rel, a, b, which) in [("placed_bears", &old.0, u.bears(), None), ("placed_reaches", &old.1, u.reach(), Some(key.0))] {
                    for &to in a.iter().filter(|p| b.contains(p)) {
                        let k6 = d.key6(key.0, key.1, key.2, to);
                        c.row_asks += 1;
                        let ok = d.reads_raw(sp, rel, &k6, which).and_then(|(rows6, reads)| { assert_eq!(rows6.len(), 1, "a row that stood is in the answer of its own question"); reads_same(&d, which.is_some(), &reads, sp, sn) }) == Some(true);
                        if same { c.c_left_stood += 1; if ok { c.c_row_saved += 1; } } else { c.b_left_stood += 1; if ok { c.b_row_saved += 1; } }
                    }
                    // a row of the old answer that went must not have untouched reads
                    for &to in a.iter().filter(|p| !b.contains(p)) {
                        let k6 = d.key6(key.0, key.1, key.2, to);
                        if d.reads_raw(sp, rel, &k6, which).and_then(|(_, reads)| reads_same(&d, which.is_some(), &reads, sp, sn)) == Some(true) { c.row_wrong += 1; }
                    }
                }
            }
            assert_eq!(derived, k.rows_again().0, "the classes add up to the rows the keeper derived again");
        }
    }
    c
}

fn lines_of(path: &str, n: usize) -> Vec<Vec<String>> { std::fs::read_to_string(path).unwrap().lines().take(n).map(|l| l.split_whitespace().map(|w| w.to_string()).collect()).collect() }

/// The games of a file as states, from their start (`max` plies at most).
fn states_of_lines(game: &str, lines: &[Vec<String>], max: usize) -> Vec<Vec<<KernelGame as Game>::State>> {
    let k = kernel(game);
    let mut out = vec![];
    for l in lines {
        let mut s = k.initial();
        let mut v = vec![s.clone()];
        for nm in l.iter().take(max) {
            if k.outcome(&s).is_some() { break; }
            let Some(m) = k.find_move(&s, nm) else { break };
            s = k.apply(&s, &m);
            v.push(s.clone());
        }
        out.push(v);
    }
    out
}

#[test]
#[ignore = "a measure, by hand: CS_FACTS_GAMES names the games file (plan 79.58 part A1)"]
fn classes() {
    let path = std::env::var("CS_FACTS_GAMES").expect("CS_FACTS_GAMES names the file of games");
    let n = env_usize("CS_FACTS_N", 300);
    let max = env_usize("CS_FACTS_MAX", 40);
    let jobs = env_usize("CS_FACTS_JOBS", std::thread::available_parallelism().map(|x| x.get()).unwrap_or(4));
    let lines = lines_of(&path, n);
    let games = states_of_lines("chess", &lines, max);
    let t0 = std::time::Instant::now();
    let chunk = (games.len() + jobs - 1) / jobs;
    let total = std::thread::scope(|sc| {
        let hs: Vec<_> = games.chunks(chunk.max(1)).map(|g| sc.spawn(move || classify::<placed_chess_reads::Game>("chess", g))).collect();
        let mut t = Classes::default();
        for h in hs { t.add(&h.join().unwrap()); }
        t
    });
    let m = total.moves.max(1) as f64;
    let per = |x: usize| x as f64 / m;
    println!("OBS 79.58 A1 {} games of at most {max} plies, {} moves, {} threads, {:.1} s", games.len(), total.moves, jobs, t0.elapsed().as_secs_f64());
    let derived = total.a_rows + total.b_rows + total.c_rows;
    println!("OBS 79.58 A1 rows derived again a move {:.2} (kept {:.2}); units asked again {:.2} (kept {:.2})", per(derived), per(total.kept_rows), per(total.a_units + total.b_units + total.c_units), per(total.kept_units));
    println!("OBS 79.58 A1 class a (the unit is new by key: the unit that moved): units {:.2} a move, rows derived {:.2} (all new)", per(total.a_units), per(total.a_rows));
    println!("OBS 79.58 A1 class b (support touched, answer changed): units {:.2} a move, rows derived {:.2}, of them new on the graph {:.2}, rows that stood {:.2}", per(total.b_units), per(total.b_rows), per(total.b_new), per(total.b_stood));
    println!("OBS 79.58 A1 class c (support touched, answer THE SAME): units {:.2} a move, rows derived {:.2}, of them new on the graph {:.2}, rows that stood {:.2}", per(total.c_units), per(total.c_rows), per(total.c_new), per(total.c_stood));
    println!("OBS 79.58 A1 of class c, units all of whose changed places in the support are places where one unit stood before and another after (a taking, the same place read as occupied both times): {:.2} a move of {:.2}", per(total.c_replaced), per(total.c_units));
    println!("OBS 79.58 A1 finer support by rows of the cell fluent (a read kept with the cells it matched): of class c it would have kept {:.2} units a move ({:.2} rows) of {:.2}; of class b units it would have kept {} (the lemma says 0); units whose reads cannot be compared {}", per(total.c_fine_units), per(total.c_fine_rows), per(total.c_units), total.b_fine_wrong, total.opaque_units);
    println!("OBS 79.58 A1 support by row of the answer, on the units still asked again after that: class c {:.2} units a move with {:.2} rows that stood, of which {:.2} have untouched reads of their own (kept by a row support); class b {:.2} units with {:.2} rows that stood, of which {:.2} untouched; rows of the old answer that went but had untouched reads {} (the lemma says 0); row questions asked {}",
        per(total.c_left_units), per(total.c_left_stood), per(total.c_row_saved), per(total.b_left_units), per(total.b_left_stood), per(total.b_row_saved), total.row_wrong, total.row_asks);
    println!("OBS 79.58 A1 NOT counted: a row new in the answer of a unit has no row of its own to keep, so a support by row cannot spare the question of a touched unit, only the rederiving of the rows that stood (an absent row would need a support of its own)");
    assert_eq!((total.b_fine_wrong, total.row_wrong), (0, 0));
}

/// One line of the count of C for a game, from the tally of a run.
fn count_line(game: &str, t: &Tally, secs: f64) {
    let per = |x: usize| x as f64 / t.moves.max(1) as f64;
    println!("OBS 79.58 A3 {game}: {} moves ({:.1} s); rows on the graph a state {:.1}, units in play {:.1}; hand rule asks {:.2} units again, derives {:.1} rows, keeps {:.1}; support rule asks {:.2} units again, derives {:.1} rows, keeps {:.1}; rows new after a move {:.1}, gone {:.1}; share predicted to stand: hand {:.3}, support {:.3}; share that stands in fact {:.3}; differences from fresh {}, from hand {}",
        t.moves, secs, per(t.rows_on_graph), per(t.hand_units.1), per(t.hand_units.0), per(t.hand_rows.0), per(t.hand_rows.1 - t.hand_rows.0), per(t.sup_units.0), per(t.sup_rows.0), per(t.sup_rows.1 - t.sup_rows.0), per(t.rows_new), per(t.rows_gone),
        1.0 - t.hand_rows.0 as f64 / t.hand_rows.1.max(1) as f64, 1.0 - t.sup_rows.0 as f64 / t.sup_rows.1.max(1) as f64, 1.0 - t.rows_new as f64 / t.rows_on_graph.max(1) as f64, t.diff_fresh, t.diff_hand);
}

fn other_game<T: Tr + 'static>(game: &str, n: usize, max: usize) {
    let t0 = std::time::Instant::now();
    let (t, _) = run::<T>(game, &Source::Seeded(n, max), false);
    count_line(game, &t, t0.elapsed().as_secs_f64());
    assert_eq!((t.diff_fresh, t.diff_hand, t.diff_unit), (0, 0, 0), "{:?}", t.first);
}

#[test]
#[ignore = "a measure, by hand (plan 79.58 part A3): the count of C for the six other games on seeded walks"]
fn other() {
    let (n, max) = (env_usize("CS_FACTS_N", 30), env_usize("CS_FACTS_MAX", 40));
    println!("OBS 79.58 A3 seeded walks (seed {SEED}): {n} games of at most {max} plies a game");
    other_game::<placed_atomic_reads::Game>("atomic", n, max);
    other_game::<placed_antichess_reads::Game>("antichess", n, max);
    other_game::<placed_horde_reads::Game>("horde", n, max);
    other_game::<placed_chess960_reads::Game>("chess960", n, max);
    other_game::<placed_threecheck_reads::Game>("3check", n, max);
    other_game::<placed_kingofthehill_reads::Game>("kingofthehill", n, max);
}

/// PLAN 79.58 B3: a digest of layer 1's graph (the nodes, the links, the dump, node by node) at every state of the seeded walks of the seven games, for the proof that raising the count of slots of the door by one leaves a graph with no row of the new relation unchanged byte for byte.
/// Run before and after the change: the digests must be the same.
#[test]
#[ignore = "a digest, by hand (plan 79.58 part B3): run before and after raising the count of slots"]
fn digest() {
    use std::hash::{Hash, Hasher};
    for game in ["chess", "atomic", "3check", "kingofthehill", "antichess", "chess960", "horde"] {
        let st = static_of(game);
        let k = kernel(game);
        let mut h = std::collections::hash_map::DefaultHasher::new();
        let (mut states, mut nodes, mut links, mut bytes) = (0usize, 0usize, 0usize, 0usize);
        let mut rng = Rng::new(SEED);
        for _ in 0..GAMES {
            let mut s = k.initial();
            for _ in 0..=PLIES {
                let part = st.position(&k, &s);
                let defs = part.definitions(st.board());
                let dump = part.dump(st.board());
                format!("{:?}", defs).hash(&mut h);
                dump.hash(&mut h);
                states += 1; nodes += defs.0.len(); links += defs.1.len(); bytes += dump.len();
                let mv = k.moves(&s);
                if mv.is_empty() || k.outcome(&s).is_some() { break; }
                let mut sorted: Vec<(String, usize)> = mv.iter().enumerate().map(|(i, m)| (k.move_name(&s, m), i)).collect();
                sorted.sort();
                let (_, i) = sorted[rng.below(sorted.len())].clone();
                s = k.apply(&s, &mv[i]);
            }
        }
        println!("OBS 79.58 B3 digest {game}: {states} graphs, {nodes} nodes, {links} links, {bytes} bytes of dump, digest {:016x}", h.finish());
    }
}

// ------------------------------------------------------------------------------------------------------------------------------ PLAN 79.58 PART B

type UKey = (usize, u8, u16);
type AllowedRows = std::collections::BTreeMap<UKey, Vec<u16>>;

/// The allowed ends of every unit of the side to act, each unit asked afresh.
fn allowed_fresh<T: Tr>(d: &Described<T>, s: &<KernelGame as Game>::State) -> AllowedRows {
    let m = d.mover(s).expect("a side to act");
    d.cells(s).into_iter().filter(|c| c.0 == m).map(|(o, k, p)| ((o, k, p), d.allowed_with_support(s, o, k, p).0)).collect()
}

/// Whether every read of an allowed row's question has the same facts in the two states: the cell fluent by the units it matches, the fluent of whose turn it is fixed, every other fluent by its stamp (None: a read that cannot be compared).
fn reads_same_allowed<T: Tr>(d: &Described<T>, reads: &[(String, Vec<u32>)], before: &<KernelGame as Game>::State, after: &<KernelGame as Game>::State) -> Option<bool> {
    let mut same = true;
    for (name, pat) in reads {
        if name == "control" { continue; }
        if name == "cell" {
            let p = d.cell_pattern(pat)?;
            if d.matching(before, &p) != d.matching(after, &p) { same = false; }
        } else if d.whole_stamp_of(before, 0, name) != d.whole_stamp_of(after, 0, name) { same = false; }
    }
    Some(same)
}

/// The same from the outside library's moves: the ends of each allowed move, put on the unit that stands at its start.
fn allowed_by_moves<T: Tr>(d: &Described<T>, s: &<KernelGame as Game>::State) -> AllowedRows {
    let m = d.mover(s).expect("a side to act");
    let mut out: AllowedRows = d.cells(s).into_iter().filter(|c| c.0 == m).map(|c| (c, vec![])).collect();
    for mv in d.moves(s) {
        let (from, to) = d.kernel.move_ends(s, &mv);
        let k = d.kernel.kind_at(s, from).expect("a unit stands at the start of a move");
        out.get_mut(&(m, k, from)).expect("the unit of a move is of the side to act").push(to);
    }
    for v in out.values_mut() { v.sort_unstable(); v.dedup(); }
    out
}

/// Whether the goal unit of the side to act (the unit of the kind named `king`) is borne on by the other side, at the state as it stands (the bulk rows of the description).
fn goal_borne_on<T: Tr>(d: &Described<T>, s: &<KernelGame as Game>::State) -> Option<(UKey, Vec<UKey>)> {
    let m = d.mover(s)?;
    let goal = d.kernel.kind_names().iter().position(|k| k == "king")? as u8;
    let rows = d.rows(s);
    let g = rows.cells.iter().find(|c| c.0 == m && c.1 == goal)?;
    let by: Vec<UKey> = rows.bears.iter().filter(|(k, v)| k.0 != m && v.contains(&g.2)).map(|(k, _)| *k).collect();
    if by.is_empty() { None } else { Some((*g, by)) }
}

#[derive(Default, Clone, Debug)]
struct Split { tempi: usize, rows_on_graph: usize, units: usize, units_asked: usize, units_asked_places: usize, rows_all: usize, rows_asked: usize, compared: usize, stood: usize, row_asks: usize, row_saved: usize, row_wrong: usize, row_opaque: usize, row_stood: usize }

#[derive(Default, Clone, Debug)]
struct AllowedTally {
    states: usize, undone: usize, diff_kept_fresh: usize, diff_fresh_moves: usize, diff_graph: usize,
    first: Vec<String>,
    /// [0] the goal unit of the side to act is not borne on, [1] it is; the first two states of a game (no record of the side) are in `starts`
    split: [Split; 2],
    starts: Split,
    /// what the questions read: the fluents named by the allowed answers (name -> reads)
    support_all: usize, supports: usize, support_places: usize, wholes: usize,
}

impl AllowedTally {
    fn add(&mut self, o: &AllowedTally) {
        self.states += o.states; self.undone += o.undone; self.diff_kept_fresh += o.diff_kept_fresh; self.diff_fresh_moves += o.diff_fresh_moves; self.diff_graph += o.diff_graph;
        for x in &o.first { if self.first.len() < 4 { self.first.push(x.clone()); } }
        for (a, b) in self.split.iter_mut().zip(o.split.iter()).chain(std::iter::once((&mut self.starts, &o.starts))) { a.tempi += b.tempi; a.rows_on_graph += b.rows_on_graph; a.units += b.units; a.units_asked += b.units_asked; a.units_asked_places += b.units_asked_places; a.rows_all += b.rows_all; a.rows_asked += b.rows_asked; a.compared += b.compared; a.stood += b.stood; a.row_asks += b.row_asks; a.row_saved += b.row_saved; a.row_wrong += b.row_wrong; a.row_opaque += b.row_opaque; a.row_stood += b.row_stood; }
        self.support_all += o.support_all; self.supports += o.supports; self.support_places += o.support_places; self.wholes += o.wholes;
    }
}

/// THE GATES AND THE MEASURE of the allowed rows along the given games (the states of each, one after the other). `strike`: one read struck out of every allowed support after every tempo (the planted fault).
fn allowed_run<T: Tr + 'static>(game: &str, games: &[Vec<<KernelGame as Game>::State>], strike: bool) -> (AllowedTally, std::collections::BTreeMap<String, usize>) {
    let st = static_of(game);
    let dk = Described::new(kernel(game), T::make());       // the keeper's questions (by unit, with reads)
    let df = Described::new(kernel(game), T::make());       // the fresh questions
    let dg = Described::new(kernel(game), T::make());       // the graph's relations (bulk, as the description gives them)
    let dr = Described::new(kernel(game), T::make());       // the questions of one row (the measure of a support by row)
    let row_every = env_usize("CS_ALLOWED_ROW_EVERY", 8);     // the support by row is measured at every n-th ply (a question of one row is dear)
    let mut t = AllowedTally::default();
    let strike_all = |k: &mut AllowedKeeper| { if let Some(m) = k.supports_mut() { for v in m.values_mut() { if !v.all && !v.reads.is_empty() { let i = v.reads.len() / 2; v.reads.remove(i); } } } };
    for (gi, states) in games.iter().enumerate() {
        let s0 = &states[0];
        let mut ak = AllowedKeeper::new(&dk, s0);
        let mut gk = GraphKeeper::new(&st, &dg, s0);
        if strike { strike_all(&mut ak); }
        let part_of = |gk: &GraphKeeper, ak: &AllowedKeeper| { let mut p = gk.part(&st); ak.write(&st, &mut p); p.definitions(st.board()) };
        let fresh_part = |s: &<KernelGame as Game>::State| { let mut p = st.position(&dg, s); AllowedKeeper::new(&df, s).write(&st, &mut p); p.definitions(st.board()) };
        let mut history: Vec<AllowedRows> = vec![ak.rows()];
        let mut rows_stack: Vec<AllowedRows> = vec![ak.rows()];
        let mut defs_stack = vec![part_of(&gk, &ak)];
        let check = |t: &mut AllowedTally, ak: &AllowedKeeper, s: &<KernelGame as Game>::State, what: String| {
            let kept = ak.rows();
            let fresh = allowed_fresh(&df, s);
            if kept != fresh { t.diff_kept_fresh += 1; if t.first.len() < 4 { t.first.push(format!("{game} game {gi} {what}: the rows kept under the lag of two differ from the fresh rows")); } }
            if fresh != allowed_by_moves(&df, s) { t.diff_fresh_moves += 1; if t.first.len() < 4 { t.first.push(format!("{game} game {gi} {what}: the fresh rows differ from the ends of the outside library's moves")); } }
        };
        check(&mut t, &ak, s0, "ply 0".into());
        for (ply, s) in states.iter().enumerate().skip(1) {
            gk.make(&st, &dg, s);
            ak.make(&dk, s);
            if strike { strike_all(&mut ak); }
            t.states += 1;
            if df.mover(s).is_some() {
                check(&mut t, &ak, s, format!("ply {ply}"));
                let kept_part = part_of(&gk, &ak);
                if kept_part != fresh_part(s) { t.diff_graph += 1; if t.first.len() < 4 { t.first.push(format!("{game} game {gi} ply {ply}: the graph with the allowed rows written differs from the fresh graph")); } }
                defs_stack.push(kept_part);
                // the measure
                let rows_now = ak.rows();
                let (a, n) = ak.asked();
                let (ra, rall) = ak.rows_again();
                let sp = if ply < 2 { &mut t.starts } else if goal_borne_on(&df, s).is_some() { &mut t.split[1] } else { &mut t.split[0] };
                sp.tempi += 1; sp.rows_on_graph += rall; sp.units += n; sp.units_asked += a; sp.units_asked_places += ak.asked_by_places(); sp.rows_all += rall; sp.rows_asked += ra;
                if ply >= 2 {
                    let before = &history[ply - 2];
                    for (k, v) in &rows_now { sp.compared += v.len(); if let Some(o) = before.get(k) { sp.stood += v.iter().filter(|p| o.contains(p)).count(); } }
                    // what a support BY ROW of the answer would keep: each row that stood, asked with all five places bound at the state two tempi before, its reads compared by content with this state
                    let old_state = &states[ply - 2];
                    let sampled = ply % row_every == 0;
                    for (k, o) in before.iter().filter(|_| sampled) {
                        let now = rows_now.get(k);
                        for &to in o {
                            let stands = now.map_or(false, |v| v.contains(&to));
                            sp.row_asks += 1;
                            if stands { sp.row_stood += 1; }
                            let verdict = dr.reads_raw(old_state, "placed_allowed", &dr.key5(k.0, k.2, to), None).and_then(|(_, reads)| reads_same_allowed(&dr, &reads, old_state, s));
                            if verdict.is_none() { sp.row_opaque += 1; }
                            let same = verdict == Some(true);
                            if same && stands { sp.row_saved += 1; }
                            if same && !stands { sp.row_wrong += 1; }
                        }
                    }
                }
                for k in rows_now.keys() { if let Some(x) = ak.support_of(k) { t.supports += 1; if x.all { t.support_all += 1; } t.support_places += x.places.len(); t.wholes += x.wholes.len(); } }
                history.push(rows_now.clone());
                rows_stack.push(rows_now);
            } else { history.push(AllowedRows::new()); rows_stack.push(AllowedRows::new()); defs_stack.push(part_of(&gk, &ak)); }
        }
        while rows_stack.len() > 1 {
            rows_stack.pop(); defs_stack.pop();
            assert!(ak.undo() && gk.undo());
            t.undone += 1;
            if &ak.rows() != rows_stack.last().unwrap() { t.diff_kept_fresh += 1; if t.first.len() < 4 { t.first.push(format!("{game} game {gi}: after an undo at depth {} the rows differ from those held then", rows_stack.len() - 1)); } }
            if &part_of(&gk, &ak) != defs_stack.last().unwrap() { t.diff_graph += 1; if t.first.len() < 4 { t.first.push(format!("{game} game {gi}: after an undo at depth {} the graph differs from the one held then", rows_stack.len() - 1)); } }
        }
    }
    let seen = dk.fluents_read.borrow().clone();
    (t, seen)
}

fn allowed_gate<T: Tr + 'static>(game: &str) {
    let k = kernel(game);
    let games = seeded_states(&k, GAMES, PLIES);
    let (t, seen) = allowed_run::<T>(game, &games, false);
    let (f, _) = allowed_run::<T>(game, &games, true);
    println!("OBS 79.58 B {game}: {GAMES} games of at most {PLIES} plies (seed {SEED}), {} states, {} undone depths; the rows kept under the lag of two differ from the fresh rows {} times, the fresh rows from the outside library's moves {} times, the graph with the allowed rows written through the door from the fresh graph {} times; supports: {} kept, {} with no bound (all), {:.1} places and {:.2} fluents read as a whole on average; fluents read: {:?}",
        t.states, t.undone, t.diff_kept_fresh, t.diff_fresh_moves, t.diff_graph, t.supports, t.support_all, t.support_places as f64 / t.supports.max(1) as f64, t.wholes as f64 / t.supports.max(1) as f64, seen);
    println!("OBS 79.58 B {game} planted fault (one read struck out of every allowed support after every tempo): the rows kept differ from the fresh rows at {} of {} checks, the graph at {}", f.diff_kept_fresh, f.states + f.undone, f.diff_graph);
    for x in &t.first { println!("FINDING 79.58 B {x}"); }
    assert!(t.states > 0);
    assert_eq!((t.diff_kept_fresh, t.diff_fresh_moves, t.diff_graph), (0, 0, 0), "{:?}", t.first);
    assert!(f.diff_kept_fresh + f.diff_graph > 0, "{game}: a support with a place struck out by hand was not caught");
}

/// The states of seeded walks of a game (the walk of `games_of`).
fn seeded_states(k: &KernelGame, n: usize, plies: usize) -> Vec<Vec<<KernelGame as Game>::State>> {
    let mut rng = Rng::new(SEED);
    let mut out = vec![];
    for _ in 0..n {
        let mut s = k.initial();
        let mut states = vec![s.clone()];
        for _ in 0..plies {
            let mv = k.moves(&s);
            if mv.is_empty() || k.outcome(&s).is_some() { break; }
            let mut sorted: Vec<(String, usize)> = mv.iter().enumerate().map(|(i, m)| (k.move_name(&s, m), i)).collect();
            sorted.sort();
            let (_, i) = sorted[rng.below(sorted.len())].clone();
            s = k.apply(&s, &mv[i]);
            states.push(s.clone());
        }
        out.push(states);
    }
    out
}

#[test] fn chess_the_allowed_rows_kept_under_the_lag_of_two_are_the_fresh_rows() { allowed_gate::<placed_chess_reads::Game>("chess"); }
#[test] fn atomic_the_allowed_rows_kept_under_the_lag_of_two_are_the_fresh_rows() { allowed_gate::<placed_atomic_reads::Game>("atomic"); }
#[test] fn threecheck_the_allowed_rows_kept_under_the_lag_of_two_are_the_fresh_rows() { allowed_gate::<placed_threecheck_reads::Game>("3check"); }
#[test] fn kingofthehill_the_allowed_rows_kept_under_the_lag_of_two_are_the_fresh_rows() { allowed_gate::<placed_kingofthehill_reads::Game>("kingofthehill"); }
#[test] fn antichess_the_allowed_rows_kept_under_the_lag_of_two_are_the_fresh_rows() { allowed_gate::<placed_antichess_reads::Game>("antichess"); }
#[test] fn chess960_the_allowed_rows_kept_under_the_lag_of_two_are_the_fresh_rows() { allowed_gate::<placed_chess960_reads::Game>("chess960"); }
#[test] fn horde_the_allowed_rows_kept_under_the_lag_of_two_are_the_fresh_rows() { allowed_gate::<placed_horde_reads::Game>("horde"); }

fn split_lines(label: &str, t: &AllowedTally, secs: f64) {
    let names = ["goal unit not borne on", "goal unit borne on"];
    for (i, sp) in t.split.iter().enumerate() {
        let n = sp.tempi.max(1) as f64;
        println!("OBS 79.58 B4 {label}, {} ({} tempi{}): allowed rows a state {:.1}; units of the side {:.1}, asked again {:.2} (the place rule alone would ask {:.2}); rows asked again {:.1}, kept {:.1}; share of rows predicted to stand (kept) {:.3}; share that stands in fact (rows also held two tempi before) {:.3}",
            names[i], sp.tempi, if i == 0 { format!(", {:.1} s", secs) } else { String::new() }, sp.rows_on_graph as f64 / n, sp.units as f64 / n, sp.units_asked as f64 / n, sp.units_asked_places as f64 / n, sp.rows_asked as f64 / n, (sp.rows_all - sp.rows_asked) as f64 / n,
            1.0 - sp.rows_asked as f64 / sp.rows_all.max(1) as f64, sp.stood as f64 / sp.compared.max(1) as f64);
    }
    for (i, sp) in t.split.iter().enumerate() {
        println!("OBS 79.58 B4 {label}, {}: a support BY ROW of the answer (the reads of one row, all five places bound, at the state two tempi before, measured at every CS_ALLOWED_ROW_EVERY-th ply) would keep {} of the {} rows of the old answer asked about; of the rows of those that stood in fact ({}) {:.3}; rows that went although their own reads were untouched {} (the lemma says 0); rows whose reads could not be compared {}; it cannot spare the question of a unit about a new row", names[i], sp.row_saved, sp.row_asks, sp.row_stood, sp.row_saved as f64 / sp.row_stood.max(1) as f64, sp.row_wrong, sp.row_opaque);
    }
    let s = &t.starts;
    println!("OBS 79.58 B4 {label}, the first two tempi of each game (no record of the side yet, every unit asked): {} tempi, units asked {} of {}", s.tempi, s.units_asked, s.units);
}

fn allowed_threads<T: Tr + 'static>(game: &str, games: &[Vec<<KernelGame as Game>::State>], jobs: usize) -> AllowedTally {
    let chunk = ((games.len() + jobs - 1) / jobs).max(1);
    std::thread::scope(|sc| {
        let hs: Vec<_> = games.chunks(chunk).map(|g| sc.spawn(move || allowed_run::<T>(game, g, false).0)).collect();
        let mut t = AllowedTally::default();
        for h in hs { t.add(&h.join().unwrap()); }
        t
    })
}

#[test]
#[ignore = "a measure, by hand (plan 79.58 part B4): CS_FACTS_GAMES names the games file; the six other games on seeded walks"]
fn allowed() {
    let path = std::env::var("CS_FACTS_GAMES").expect("CS_FACTS_GAMES names the file of games");
    let (n, max) = (env_usize("CS_FACTS_N", 300), env_usize("CS_FACTS_MAX", 40));
    let jobs = env_usize("CS_FACTS_JOBS", std::thread::available_parallelism().map(|x| x.get()).unwrap_or(4));
    let games = states_of_lines("chess", &lines_of(&path, n), max);
    let t0 = std::time::Instant::now();
    let t = allowed_threads::<placed_chess_reads::Game>("chess", &games, jobs);
    println!("OBS 79.58 B4 chess, {} games of the file of at most {max} plies, {} states, {} threads: differences kept against fresh {}, fresh against the library's moves {}, graph {}", games.len(), t.states, jobs, t.diff_kept_fresh, t.diff_fresh_moves, t.diff_graph);
    split_lines("chess (human games)", &t, t0.elapsed().as_secs_f64());
    assert_eq!((t.diff_kept_fresh, t.diff_fresh_moves, t.diff_graph), (0, 0, 0), "{:?}", t.first);
    assert_eq!(t.split[0].row_wrong + t.split[1].row_wrong, 0, "a row that went had untouched reads: the lemma fails");
    let (ns, plies) = (env_usize("CS_ALLOWED_SEEDED_N", 30), max);
    macro_rules! other { ($m:ident, $name:expr) => {{ let t0 = std::time::Instant::now(); let k = kernel($name); let g = seeded_states(&k, ns, plies); let t = allowed_threads::<$m::Game>($name, &g, jobs); split_lines(&format!("{} (seeded walks, seed {SEED}, {ns} games of at most {plies} plies)", $name), &t, t0.elapsed().as_secs_f64()); assert_eq!((t.diff_kept_fresh, t.diff_fresh_moves, t.diff_graph), (0, 0, 0), "{:?}", t.first); }} }
    other!(placed_atomic_reads, "atomic");
    other!(placed_antichess_reads, "antichess");
    other!(placed_horde_reads, "horde");
    other!(placed_chess960_reads, "chess960");
    other!(placed_threecheck_reads, "3check");
    other!(placed_kingofthehill_reads, "kingofthehill");
}

/// A place by its name, a unit by its kind and place.
fn pname<T: Tr>(d: &Described<T>, p: u16) -> String { d.kernel.place_names()[p as usize].clone() }
fn uname<T: Tr>(d: &Described<T>, k: &UKey) -> String { format!("{} {} on {}", d.kernel.roles()[k.0], d.kernel.kind_names()[k.1 as usize], pname(d, k.2)) }

/// B5: twenty states of the human games in which the goal unit of the side to act is borne on, printed in full.
#[test]
#[ignore = "a command, by hand (plan 79.58 part B5): CS_FACTS_GAMES names the games file"]
fn states() {
    let path = std::env::var("CS_FACTS_GAMES").expect("CS_FACTS_GAMES names the file of games");
    let (n, max, want) = (env_usize("CS_FACTS_N", 300), env_usize("CS_FACTS_MAX", 40), env_usize("CS_FACTS_STATES", 20));
    let games = states_of_lines("chess", &lines_of(&path, n), max);
    let st = static_of("chess");
    let dk = Described::new(kernel("chess"), <placed_chess_reads::Game as Tr>::make());
    let dg = Described::new(kernel("chess"), <placed_chess_reads::Game as Tr>::make());
    let (mut shown, mut touch_rows, mut survivors, mut refused, mut asked_meet, mut asked_all, mut kept_meet, mut kept_all) = (0, 0, 0, 0, 0, 0, 0, 0);
    for (gi, g) in games.iter().enumerate() {
        let mut ak = AllowedKeeper::new(&dk, &g[0]);
        let mut gk = GraphKeeper::new_supported(&st, &dg, &g[0]);
        let mut done_in_game = false;
        for (ply, s) in g.iter().enumerate().skip(1) {
            let before = ak.rows();
            let prev_support: std::collections::BTreeMap<UKey, cs_game::keeper::AllowedSupport> = before.keys().filter_map(|k| ak.support_of(k).map(|x| (*k, x.clone()))).collect();
            let _ = prev_support;
            ak.make(&dk, s);
            gk.make_supported(&st, &dg, s);
            if ply < 2 || dg.mover(s).is_none() { continue; }
            let Some((goal, by)) = goal_borne_on(&dg, s) else { continue };
            // the places the bearing on the goal unit rested on: the support of each bearing unit's question
            let mut bsup: Vec<u16> = vec![];
            let mut bearing_lines = vec![];
            for b in &by {
                let (_, sp) = dg.bearing_with_support(s, b.0, b.1, b.2);
                bsup.extend(sp.places.iter().copied());
                bearing_lines.push(format!("{} bears on {} resting on [{}]", uname(&dg, b), pname(&dg, goal.2), sp.places.iter().map(|p| pname(&dg, *p)).collect::<Vec<_>>().join(" ")));
            }
            bsup.sort_unstable(); bsup.dedup();
            let rows = ak.rows();
            let (asked_n, units_n) = ak.asked();
            // a unit asked again: the lag of two asks it when a place of its support changed between the state two tempi before and this one; the keeper does not name them, so they are those whose rows were asked: recompute from the supports
            let again: Vec<UKey> = ak.asked_units();
            let _ = asked_n;
            let touches = |k: &UKey, ends: &Vec<u16>| *k == goal || ends.iter().any(|p| bsup.contains(p)) || bsup.contains(&k.2);
            for (k, ends) in &rows {
                if ends.is_empty() { refused += 1; continue; }
                survivors += 1;
                if touches(k, ends) { touch_rows += 1; }
            }
            for k in rows.keys() {
                let meets = ak.support_of(k).map_or(true, |x| x.all || bsup.iter().any(|p| x.places.binary_search(p).is_ok()));
                if again.contains(k) { asked_all += 1; if meets { asked_meet += 1; } } else { kept_all += 1; if meets { kept_meet += 1; } }
            }
            if shown < want && !done_in_game {
                done_in_game = true;
                shown += 1;
                println!("STATE {} of game {gi} after {ply} plies; {} to act; {} of {} units asked again", dg.kernel.fen(s), dg.kernel.roles()[dg.mover(s).unwrap()], asked_n, units_n);
                println!("  goal unit: {}", uname(&dg, &goal));
                for l in &bearing_lines { println!("  {l}"); }
                println!("  support of the bearing (all places its questions read): [{}]", bsup.iter().map(|p| pname(&dg, *p)).collect::<Vec<_>>().join(" "));
                println!("  units asked again of the side to act: [{}]", again.iter().map(|k| uname(&dg, k)).collect::<Vec<_>>().join("; "));
                for (k, ends) in &rows { println!("  allowed {}: [{}]{}", uname(&dg, k), ends.iter().map(|p| pname(&dg, *p)).collect::<Vec<_>>().join(" "), if ends.is_empty() { "" } else if touches(k, ends) { " (touches the bearing)" } else { " (does not touch the bearing)" }); }
            }
        }
    }
    println!("OBS 79.58 B5 over {} games: in the states where the goal unit of the side to act is borne on, units with an allowed row {survivors}, of which touch the bearing (the goal unit itself, a unit standing on a place the bearing rested on, or a unit with an end on such a place) {touch_rows}; units with no allowed row {refused}; units asked again {asked_all}, of which the allowed support meets the bearing's {asked_meet}; units kept {kept_all}, of which the allowed support meets the bearing's {kept_meet}; states printed {shown}", games.len());
}

fn load_line() -> String { std::process::Command::new("uptime").output().map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string()).unwrap_or_default() }

/// A2: the clock. Three walks of the same games (a fresh adapter for each walk); the time of each call by `Instant`, summed.
#[test]
#[ignore = "a clock, by hand, on a quiet machine (plan 79.58 part A2): CS_FACTS_GAMES names the games file"]
fn clock() {
    use std::time::Instant;
    let path = std::env::var("CS_FACTS_GAMES").expect("CS_FACTS_GAMES names the file of games");
    let n = env_usize("CS_FACTS_N", 30);
    let max = env_usize("CS_FACTS_MAX", 40);
    let lines = lines_of(&path, n);
    let games = states_of_lines("chess", &lines, max);
    let st = static_of("chess");
    let bridge = kernel("chess");
    println!("OBS 79.58 A2 load before: {}", load_line());
    for rep in 0..3 {
        let dh = Described::new(kernel("chess"), <placed_chess_reads::Game as Tr>::make());
        let ds = Described::new(kernel("chess"), <placed_chess_reads::Game as Tr>::make());
        let dq = Described::new(kernel("chess"), <placed_chess_reads::Game as Tr>::make());
        let (mut ns_b, mut ns_h, mut ns_s, mut moves) = (0u128, 0u128, 0u128, 0u128);
        // (i) a unit's question with reads, (ii) the bulk piece's: counted over the units of every state of the walk
        let (mut q_reads, mut q_bulk, mut q_bound, mut units, mut states, mut ns_load) = (0u128, 0u128, 0u128, 0u128, 0u128, 0u128);
        let mut t = <placed_chess_reads::Game as Tr>::make();
        for g in &games {
            let mut kb = GraphKeeper::new(&st, &bridge, &g[0]);
            let mut kh = GraphKeeper::new(&st, &dh, &g[0]);
            let mut ks = GraphKeeper::new_supported(&st, &ds, &g[0]);
            for s in g.iter().skip(1) {
                let t0 = Instant::now(); kb.make(&st, &bridge, s); ns_b += t0.elapsed().as_nanos();
                let t0 = Instant::now(); kh.make(&st, &dh, s); ns_h += t0.elapsed().as_nanos();
                let t0 = Instant::now(); ks.make_supported(&st, &ds, s); ns_s += t0.elapsed().as_nanos();
                moves += 1;
            }
            // the questions, at every state of the game (the module's states made once a state, outside the clock of a question)
            for s in g.iter() {
                let base = described::pairs_of(&bridge.state_facts(s));
                let t0 = Instant::now();
                let s0 = t.state_of_facts_t(&base).unwrap();
                let sides: Vec<_> = bridge.roles().iter().map(|nm| t.state_of_facts_t(&described::with_control(&base, nm)).unwrap()).collect();
                ns_load += t0.elapsed().as_nanos();
                let cells = dq.rows(s).cells.clone();
                states += 1;
                let qb: Vec<(String, Vec<u32>)> = vec![("placed_bears".to_string(), vec![described::FREE; 6])];
                let qr: Vec<(String, Vec<u32>)> = vec![("placed_reaches".to_string(), vec![described::FREE; 6])];
                let t0 = Instant::now();
                let _ = t.ask_t(&s0, &[], &qb);
                for st1 in &sides { let _ = t.ask_t(st1, &[], &qr); }
                q_bulk += t0.elapsed().as_nanos();
                for &(o, k, p) in &cells {
                    let k4 = dq.key4(o, k, p);
                    units += 1;
                    let t0 = Instant::now();
                    let _ = t.reads_t(&s0, "placed_bears", &k4);
                    let _ = t.reads_t(&sides[o], "placed_reaches", &k4);
                    q_reads += t0.elapsed().as_nanos();
                    let mut pb = vec![described::FREE; 6]; pb[..4].copy_from_slice(&k4);
                    let t0 = Instant::now();
                    let _ = t.ask_t(&s0, &[], &[("placed_bears".to_string(), pb.clone())]);
                    let _ = t.ask_t(&sides[o], &[], &[("placed_reaches".to_string(), pb)]);
                    q_bound += t0.elapsed().as_nanos();
                }
            }
        }
        let (m, u, st_n) = (moves as f64, units as f64, states as f64);
        println!("OBS 79.58 A2 walk {} ({} moves, {} states, {} units): load {}", rep + 1, moves, states, units, load_line());
        println!("OBS 79.58 A2 walk {} a unit's two questions (bearing and reach) with the reads recorded: {:.0} ns a unit", rep + 1, q_reads as f64 / u);
        println!("OBS 79.58 A2 walk {} the same two questions from the bulk piece (a whole state asked, the bearing once and each side's reach), the share of a unit: {:.0} ns a unit ({:.0} ns a state of {:.1} units); the bulk piece asked with the key bound: {:.0} ns a unit; reads over bulk share {:.1}", rep + 1, q_bulk as f64 / u, q_bulk as f64 / st_n, u / st_n, q_bound as f64 / u, q_reads as f64 / q_bulk as f64);
        println!("OBS 79.58 A2 walk {} loading a state into the module (the state and each side's): {:.0} ns a state", rep + 1, ns_load as f64 / st_n);
        println!("OBS 79.58 A2 walk {} a whole make of the keeper: hand rule asking the bridge {:.0} ns; hand rule asking the description (bulk, adapter included) {:.0} ns; support rule asking the description (by unit with reads, adapter included) {:.0} ns", rep + 1, ns_b as f64 / m, ns_h as f64 / m, ns_s as f64 / m);
        let (a, b) = (ds.ns.get(), dh.ns.get());
        println!("OBS 79.58 A2 walk {} where the time of a make went, a move: support rule: bulk rows of a state {:.0} ns (the keeper's list of units is read from it), loading a state into the module {:.0} ns, the units' questions with reads {:.0} ns (units asked again {:.2}); hand rule on the description: bulk rows of a state {:.0} ns, loading {:.0} ns",
            rep + 1, a.0 as f64 / m, a.1 as f64 / m, a.2 as f64 / m, ds.asked().0 as f64 / m, b.0 as f64 / m, b.1 as f64 / m);
    }
    println!("OBS 79.58 A2 load after: {}", load_line());
}

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
    // PLAN 79.58 B4: the allowed rows of the side to act, and for each reach row that is not allowed the places its refusal rested on
    if let Some(m) = d.mover(&s) {
        let ak = AllowedKeeper::new(&d, &s);
        let allowed = ak.rows();
        println!("allowed moves of {} ({} rows):", d.kernel.roles()[m], allowed.values().map(|v| v.len()).sum::<usize>());
        for (k, ends) in &allowed { println!("allowed {}: [{}]", uname(&d, k), ends.iter().map(|p| pname(&d, *p)).collect::<Vec<_>>().join(" ")); }
        for u in keeper.live().units().iter().filter(|u| u.owner == m) {
            let ends = allowed.get(&u.key()).cloned().unwrap_or_default();
            for &to in u.reach().iter().filter(|p| !ends.contains(p)) {
                let why = d.reads_raw(&s, "placed_allowed", &d.key5(m, u.at, to), None).map(|(_, reads)| describe_reads(&d, &reads)).unwrap_or_else(|| "no reads".to_string());
                println!("refused {} to {}: rested on {}", uname(&d, &u.key()), pname(&d, to), why);
            }
        }
    }
}

/// The reads of a question as words: a place read by its name, a read that leaves the place free by the side and kind it asks for, any other fluent by its name.
fn describe_reads<T: Tr>(d: &Described<T>, reads: &[(String, Vec<u32>)]) -> String {
    let mut out: Vec<String> = vec![];
    for (name, pat) in reads {
        if name == "control" { continue; }
        let w = if name == "cell" {
            match d.cell_pattern(pat) {
                Some([Some(f), Some(r), c, k]) => format!("{}{}", pname(d, f * 8 + r), c.map_or(String::new(), |c| format!(" ({} units only)", d.kernel.roles()[c as usize])) + &k.map_or(String::new(), |k| format!(" ({} only)", d.kernel.kind_names()[k as usize]))),
                Some([_, _, c, k]) => format!("wherever {} {} stands", c.map_or("any".to_string(), |c| d.kernel.roles()[c as usize].clone()), k.map_or("unit".to_string(), |k| d.kernel.kind_names()[k as usize].clone())),
                None => "an unknown cell".to_string(),
            }
        } else { format!("the fluent {name} as a whole") };
        if !out.contains(&w) { out.push(w); }
    }
    format!("[{}]", out.join("; "))
}
