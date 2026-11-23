//! PLAN 79.34 PART D: THE UNIT TESTS OF LAYER 1, WHAT IS SEEN IN ONE STATE WITH NO KNOWLEDGE OF PLAY (SYSTEM_SPEC 11 "The cut of layers 1 and 2 is carried into the code"; ledger rows 37, 36; piece: representation).
//! Layer 1 says: the kinds there are and what they are called; the board, its places and the plain links between them (next to, on a line, on an edge); the TWO CLASSES of places, alternating along the lines of steps; any state it is shown (which unit
//! stands where, how many of each kind stand in it), held and written back identical; and the history as a sequence of such states. It does not say how a unit moves, whose turn it comes to be, or which state a game starts from: the start is layer 2's
//! (the rules' crate), so no test here reads a starting arrangement from layer 1. Only the game's DESCRIPTION and layer 1 are used: this crate depends on no crate that holds the rules (tests/test_plan79_20.py and tests/test_plan79_34.py scan for it).
//! The test code and its strings name no object of a game (tests/test_plan78.py scans for it); the game of chess is named in the comments and in the data file tests/fixtures/seen_chess.expect, where what is expected is written out by hand.
mod common;
use common::fixtures;
use cs_rep::graph::{Board, FactDecl};
use cs_rep::history::History;
use cs_rep::names::Names;
use cs_rep::statement::Change;
use cs_rep::steps::read_steps;
use cs_rep::{Fact, UnitDecl};
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

/// What is expected of the game's board and of a made state, written by hand in tests/fixtures/seen_chess.expect (tab-separated lines: a tag, then its fields).
struct Expect { kinds: Vec<String>, roles: Vec<String>, places: usize, first_place: String, last_place: String, lines: usize, links: Vec<usize>, edges: Vec<usize>, classes: (usize, usize), state: Vec<Fact>, mover: String, counts: Vec<(String, String, usize)> }

fn expect() -> Expect {
    let text = std::fs::read_to_string(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests").join("fixtures").join("seen_chess.expect")).unwrap();
    let mut e = Expect { kinds: vec![], roles: vec![], places: 0, first_place: String::new(), last_place: String::new(), lines: 0, links: vec![], edges: vec![], classes: (0, 0), state: vec![], mover: String::new(), counts: vec![] };
    let words = |s: &str| -> Vec<String> { s.split(' ').filter(|w| !w.is_empty()).map(|w| w.to_string()).collect() };
    let nums = |s: &str| -> Vec<usize> { words(s).iter().map(|w| w.parse().unwrap()).collect() };
    for line in text.lines().filter(|l| !l.is_empty() && !l.starts_with('#')) {
        let f: Vec<&str> = line.split('\t').collect();
        match f[0] {
            "kinds" => e.kinds = words(f[1]),
            "roles" => e.roles = words(f[1]),
            "places" => { e.places = f[1].parse().unwrap(); e.first_place = f[2].into(); e.last_place = f[3].into(); }
            "lines" => e.lines = f[1].parse().unwrap(),
            "links" => e.links = nums(f[1]),
            "edges" => e.edges = nums(f[1]),
            "classes" => e.classes = (f[1].parse().unwrap(), f[2].parse().unwrap()),
            "state" => { e.state = words(f[1]).iter().map(|t| { let mut p = t.split(','); Fact { relation: p.next().unwrap().into(), args: p.map(|a| a.to_string()).collect() } }).collect(); e.mover = f[2].into(); }
            "count" => e.counts.push((f[1].into(), f[2].into(), f[3].parse().unwrap())),
            other => panic!("unknown tag {other}"),
        }
    }
    e
}

/// the board of the first fixture: the game's board as layer 1 reads it from the description (the fixtures were made from it once)
fn board() -> Board { fixtures().remove(0).board }

// ------------------------------------------------------------------------------------------------------------------------------ the kinds and the sides: what there is and what it is called

#[test]
fn the_kinds_there_are_and_the_sides_have_the_names_the_game_gives_them() {
    let e = expect();
    for fx in fixtures() {
        let b = &fx.board;
        assert_eq!(b.names.kinds, e.kinds, "{}: the kinds, by their names", fx.file);
        assert_eq!(b.names.roles, e.roles, "{}: the sides, by their names", fx.file);
        for (i, k) in e.kinds.iter().enumerate() {
            assert_eq!(b.names.kind(k).unwrap() as usize, i, "a kind is its index in the list of names");
            assert_eq!(b.nodes[b.kind_node[i] as usize].id, format!("kind:{k}"), "each kind is a node with its own definition for an id");
        }
        for (i, r) in e.roles.iter().enumerate() { assert_eq!(b.nodes[b.side_node[i] as usize].id, format!("side:{r}")); }
        assert!(b.names.kind("no_such_kind").is_err() && b.names.role("no_such_side").is_err() && b.names.place("zz9").is_err(), "a name the game does not give is refused");
    }
    println!("OBS 79.34 D kinds: {} kinds and {} sides by name on every fixture board; an unknown name is refused", e.kinds.len(), e.roles.len());
}

// ------------------------------------------------------------------------------------------------------------------------------ the board: places and the plain links

#[test]
fn the_board_has_its_places_and_the_plain_links_next_on_a_line_and_on_an_edge() {
    let e = expect();
    let b = board();
    assert_eq!(b.names.places.len(), e.places);
    assert_eq!((b.names.places.first().unwrap().as_str(), b.names.places.last().unwrap().as_str()), (e.first_place.as_str(), e.last_place.as_str()));
    let mut distinct: Vec<&String> = b.names.places.iter().collect();
    distinct.sort();
    distinct.dedup();
    assert_eq!(distinct.len(), e.places, "each place has a name of its own");
    // next: how many places have a next place under each direction, sorted, against the hand-written list
    let mut links: Vec<usize> = (0..b.directions.len()).map(|d| b.next[d].iter().filter(|x| x.is_some()).count()).collect();
    links.sort();
    assert_eq!(links, e.links, "the links of each direction");
    // edge: a place is on the edge under a direction when no place comes next under it
    let mut edges: Vec<usize> = (0..b.directions.len()).map(|d| (0..e.places).filter(|&p| b.edge_distance[d][p] == 0).count()).collect();
    edges.sort();
    assert_eq!(edges, e.edges, "the places on the edge of each direction");
    for d in 0..b.directions.len() {
        for p in 0..e.places {
            assert_eq!(b.edge_distance[d][p] == 0, b.next[d][p].is_none(), "on the edge exactly where nothing comes next");
            if let Some(q) = b.next[d][p] { assert_eq!(b.edge_distance[d][q as usize] + 1, b.edge_distance[d][p], "one step nearer the edge"); }
        }
    }
    // a line: its places in order, each next to the one after it under the line's direction or its reverse; every place lies on at least one
    assert_eq!(b.lines.len(), e.lines);
    for (li, l) in b.lines.iter().enumerate() {
        assert!(l.places.len() >= 2);
        for w in l.places.windows(2) {
            let d = l.direction;
            assert!(b.next[d][w[0] as usize] == Some(w[1]) || b.reverse[d].is_some_and(|r| b.next[r][w[0] as usize] == Some(w[1])), "line {li}: neighbours on a line are next to one another");
        }
        for (i, p) in l.places.iter().enumerate() { assert_eq!(b.index_on(li as u32, *p), i as i32); }
    }
    assert!((0..e.places).all(|p| !b.place_lines[p].is_empty()), "no place lies on no line");
    // between: the places strictly between two on a line, in order
    let l0 = &b.lines[0];
    let (a, z) = (l0.places[0], *l0.places.last().unwrap());
    assert_eq!(b.strictly_between(0, a, z), l0.places[1..l0.places.len() - 1].to_vec());
    assert_eq!(b.strictly_between(0, z, a), l0.places[1..l0.places.len() - 1].to_vec(), "between does not depend on the order asked");
    assert!(b.strictly_between(0, a, a).is_empty());
    println!("OBS 79.34 D board: {} places, {} lines, links by direction {:?}, places on an edge by direction {:?}", e.places, b.lines.len(), links, edges);
}

// ------------------------------------------------------------------------------------------------------------------------------ the two classes of places

#[test]
fn the_places_fall_into_two_classes_that_alternate_along_the_lines_of_steps_derived_from_the_links_alone() {
    let e = expect();
    for fx in fixtures() {
        let b = &fx.board;
        let c = b.classes().expect("the board's links give two classes");
        let sizes = (c.class.iter().filter(|x| **x == 0).count(), c.class.iter().filter(|x| **x == 1).count());
        assert_eq!(sizes, e.classes, "{}: the size of each class", fx.file);
        let mut agree_steps = 0;
        let mut differ_steps = 0;
        for d in 0..b.directions.len() {
            let nlinks = b.next[d].iter().filter(|x| x.is_some()).count();
            for p in 0..b.names.places.len() {
                if let Some(q) = b.next[d][p] {
                    let differ = c.class[p] != c.class[q as usize];
                    assert_eq!(differ, c.flips[d], "{}: every step of one direction does the same to the class", fx.file);
                    if differ { differ_steps += 1 } else { agree_steps += 1 }
                }
            }
            // in the board of the fixture the directions that keep to a line of the board's rows and columns have the most steps (56 of 64 places have a next place); the others (49) keep the class
            assert_eq!(c.flips[d], nlinks == *e.links.iter().max().unwrap(), "{}: neighbours along a rank or a file differ, along a diagonal agree", fx.file);
        }
        assert!(differ_steps > 0 && agree_steps > 0);
        assert_eq!(c.flips.iter().filter(|f| **f).count(), 4, "four directions change the class");
    }
    println!("OBS 79.34 D classes: on the board of every fixture 32 places in each class; steps along a rank or a file change the class, steps along a diagonal keep it");
}

/// A board of four columns and three rows with four headings and no diagonal: an invented description, nothing like the board of the fixtures.
const SMALL: &str = "
(sort col x y z w) (sort row 1 2 3) (sort heading up down left right)
(signature col col) (signature row row) (signature col_succ col col) (signature row_succ row row) (signature step heading col row col row)
(col x) (col y) (col z) (col w) (row 1) (row 2) (row 3)
(col_succ x y) (col_succ y z) (col_succ z w) (row_succ 1 2) (row_succ 2 3)
(<= (step up ?c ?ra ?c ?rb) (col ?c) (row_succ ?ra ?rb))
(<= (step down ?c ?ra ?c ?rb) (col ?c) (row_succ ?rb ?ra))
(<= (step right ?ca ?r ?cb ?r) (row ?r) (col_succ ?ca ?cb))
(<= (step left ?ca ?r ?cb ?r) (row ?r) (col_succ ?cb ?ca))
";

fn small() -> Board {
    let places: Vec<String> = ["x", "y", "z", "w"].iter().flat_map(|c| ["1", "2", "3"].iter().map(move |r| format!("{c}{r}"))).collect();
    let steps = read_steps(SMALL, &["step".to_string()], &[("cell".to_string(), vec!["col".to_string(), "row".to_string()])], &places).unwrap();
    Board::new(&steps, &Names::new(places, vec!["piece".into(), "stone".into()], vec!["a".into(), "b".into()])).unwrap()
        .with_decl(FactDecl { units: UnitDecl { relation: "tok".into(), owner: 0, kind: Some(1) }, tempo: Some("turn".into()) })
}

#[test]
fn the_classes_of_other_boards_are_what_their_links_give() {
    // a board of four columns and three rows, no diagonal: two classes of six, every step changes the class
    let b = small();
    let c = b.classes().unwrap();
    assert_eq!((c.class.iter().filter(|x| **x == 0).count(), c.class.iter().filter(|x| **x == 1).count()), (6, 6));
    assert!(c.flips.iter().all(|f| *f), "every direction of that board changes the class");
    for d in 0..b.directions.len() { for p in 0..12 { if let Some(q) = b.next[d][p] { assert_ne!(c.class[p], c.class[q as usize]); } } }
    // a single line of five places: the classes alternate along it, three and two
    let places: Vec<String> = (1..=5).map(|i| format!("p{i}")).collect();
    let rows: Vec<(u16, u16, u16)> = (0..4).flat_map(|i| [(1u16, i as u16, i as u16 + 1), (0u16, i as u16 + 1, i as u16)]).collect();
    let steps = cs_rep::steps::Steps { directions: vec!["back".into(), "on".into()], rows, ..Default::default() };
    let b = Board::new(&steps, &Names::new(places, vec![], vec![])).unwrap();
    let c = b.classes().unwrap();
    assert_eq!(c.class, vec![0, 1, 0, 1, 0]);
    // the classes are the same on every build of one board, and written the same on a board read back from its own text
    assert_eq!(small().classes(), small().classes());
    let again = Board::from_json(&small().to_json()).unwrap();
    assert_eq!(again.classes(), small().classes());
    println!("OBS 79.34 D classes elsewhere: a 4 by 3 board without a diagonal gives 6 and 6 with every step changing the class; a line of five gives 3 and 2");
}

// ------------------------------------------------------------------------------------------------------------------------------ a state shown to layer 1: held, counted, written back

#[test]
fn a_state_shown_to_layer_1_is_held_counted_by_kind_and_written_back_identical() {
    let e = expect();
    let b = board();
    // a made state of a few units (no arrangement of a game's start is read from layer 1: the state is given)
    let part = b.position_from_facts(&e.state).unwrap();
    let d = b.decl.clone().unwrap();
    assert_eq!(part.units.len(), e.state.iter().filter(|f| f.relation == d.units.relation).count(), "the facts that name a unit give a unit each; the other fact (the side to act) is not a unit");
    for (owner, kind, want) in &e.counts {
        assert_eq!(part.count(b.names.role(owner).unwrap(), b.names.kind(kind).unwrap()), *want, "{owner} {kind}");
    }
    assert_eq!(b.names.roles[part.tempo().unwrap()], e.mover);
    // where each unit stands: the occupant of every place is the unit that stands there, and no other place holds one
    for (i, u) in part.units.iter().enumerate() { assert_eq!(part.occupant[u.at as usize] as usize, i + 1); }
    assert_eq!(part.occupant.iter().filter(|x| **x != 0).count(), part.units.len());
    // written back as text and read again: the same state
    let (mover, text) = b.state_text(&part);
    let back = b.part_from_text(mover.as_deref(), &text).unwrap();
    assert_eq!(back.definitions(&b), part.definitions(&b));
    assert_eq!(b.state_text(&back), (mover, text));
    // every state of every fixture game: the count of each kind in layer 1's reading equals a plain count of the facts that name it
    let mut states = 0;
    for fx in fixtures() {
        for pos in fx.positions().step_by(7) {
            let part = fx.board.position_from_facts(&pos.facts).unwrap();
            let mut plain: BTreeMap<(String, String), usize> = BTreeMap::new();
            for f in pos.facts.iter().filter(|f| f.relation == d.units.relation) { *plain.entry((f.args[d.units.owner].clone(), f.args[d.units.kind.unwrap()].clone())).or_default() += 1; }
            for (ri, r) in fx.board.names.roles.iter().enumerate() {
                for (ki, k) in fx.board.names.kinds.iter().enumerate() {
                    assert_eq!(part.count(ri, ki as u8), plain.get(&(r.clone(), k.clone())).copied().unwrap_or(0), "{}: {r} {k}", fx.file);
                }
            }
            let (m, t) = fx.board.state_text(&part);
            assert_eq!(fx.board.part_from_text(m.as_deref(), &t).unwrap().definitions(&fx.board), part.definitions(&fx.board));
            states += 1;
        }
    }
    println!("OBS 79.34 D state: a made state and {states} states of the fixture games are held, counted by kind as a plain count of their facts says, and written back identical");
}

// ------------------------------------------------------------------------------------------------------------------------------ the history as a sequence of states

#[test]
fn the_history_is_a_sequence_of_states_each_a_change_from_the_one_before() {
    let fx = fixtures().remove(0);
    let game = &fx.games[0];
    let key = |i: usize| 1000 + i as u64;
    let mut h = History::new(key(0));
    for (i, pos) in game.iter().enumerate().skip(1) {
        let (added, removed) = pos.since.clone().unwrap();
        h.push(Change { added, removed, ..Default::default() }, key(i));
    }
    assert_eq!(h.len(), game.len() - 1);
    assert_eq!(h.keys().len(), game.len());
    for (i, pos) in game.iter().enumerate() {
        let facts: BTreeSet<Fact> = h.facts_at(&game[0].facts, i).unwrap().into_iter().collect();
        assert_eq!(facts, pos.facts.iter().cloned().collect::<BTreeSet<_>>(), "the state at {i} is the first state with every change up to it applied");
        // each state of the sequence is a state layer 1 can hold
        fx.board.position_from_facts(&pos.facts).unwrap();
    }
    // taking the last back returns it, and the history is one state shorter
    let (ch, k) = h.undo().unwrap();
    assert_eq!(k, key(game.len() - 1));
    assert_eq!(ch.added, game.last().unwrap().since.as_ref().unwrap().0);
    assert_eq!(h.len(), game.len() - 2);
    // written and read back identical
    assert_eq!(History::from_json(&h.to_json()).unwrap().keys(), h.keys());
    println!("OBS 79.34 D history: a game of {} states held as the first state and {} changes; each state recovered, the last taken back, written and read back identical", game.len(), game.len() - 1);
}

// ------------------------------------------------------------------------------------------------------------------------------ no dependency on the rules

#[test]
fn the_crate_builds_with_no_dependency_on_the_rules_crate() {
    let toml = std::fs::read_to_string(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml")).unwrap();
    let deps: String = toml.split("[dependencies]").nth(1).unwrap_or("").to_string() + toml.split("[dev-dependencies]").nth(1).unwrap_or("");
    assert!(!deps.contains("cs_game") && !deps.contains("cs_tactics") && !deps.contains("cs_decision") && !deps.contains("shakmaty"), "no crate that holds the rules or a game's kernel, not even as a dev-dependency");
    // (that this file compiles and runs is the proof: it is built from the crate's own dependencies)
    println!("OBS 79.34 D no rules: Cargo.toml names no crate of the rules, not as a dependency and not as a dev-dependency");
}
