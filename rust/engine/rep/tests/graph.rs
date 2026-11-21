//! PLAN 79.21 STAGE B: THE TESTS OF LAYER 1'S GRAPH, CALLING NOTHING OF THE RULES (SYSTEM_SPEC 11 "one graph is the representation", "a layer's test uses only that layer and what is below it"; piece: representation).
//! The board is read from the fixture (a game's board as layer 1 writes it) and the positions are SETS OF FACTS read from the same files; nothing here asks the rules anything. One more board, with another shape and other names, is read from an invented description to show
//! that nothing of one game is written in the layer. tests/test_plan79_20.py scans this file for the names that would say otherwise.
mod common;
use common::{fixtures, Rng};
use cs_rep::graph::{Board, FactDecl, NodeType};
use cs_rep::names::{Names, Side};
use cs_rep::steps::read_steps;
use cs_rep::{Fact, UnitDecl};
use std::collections::BTreeSet;

/// the units of a set of facts as text, (owner, kind, place), by the board's declaration: the test's own plain reading, to be compared with the graph's
fn units_by_text(b: &Board, facts: &[Fact]) -> BTreeSet<(String, String, String)> {
    let d = b.decl.as_ref().unwrap();
    facts.iter().filter(|f| f.relation == d.units.relation).map(|f| {
        let place: String = f.args.iter().enumerate().filter(|(i, _)| *i != d.units.owner && Some(*i) != d.units.kind).map(|(_, a)| a.as_str()).collect();
        (f.args[d.units.owner].clone(), d.units.kind.map(|k| f.args[k].clone()).unwrap_or_default(), place)
    }).collect()
}

#[test]
fn the_board_is_written_and_read_back_identical() {
    for fx in fixtures() {
        let b = &fx.board;
        let again = Board::from_json(&b.to_json()).unwrap();
        assert_eq!(again.to_json(), b.to_json(), "{}: written and read back", fx.file);
        assert_eq!((again.nodes.clone(), again.links.clone(), again.lines.clone()), (b.nodes.clone(), b.links.clone(), b.lines.clone()));
        assert!(b.lines.len() > 4 && b.directions.len() > 2 && b.names.places.len() > 8 && b.decl.is_some());
        println!("OBS 79.21 B-1 {}: board of {} places, {} directions, {} lines, {} static nodes, {} static links, written and read back identical", fx.file, b.names.places.len(), b.directions.len(), b.lines.len(), b.nodes.len(), b.links.len());
    }
}

#[test]
fn the_graph_of_a_position_is_built_from_the_board_and_a_set_of_facts_alone() {
    for fx in fixtures() {
        let b = &fx.board;
        let d = b.decl.clone().unwrap();
        let (mut n, mut units_total, mut other_relations) = (0, 0, BTreeSet::new());
        let mut rng = Rng::new(2121);
        for pos in fx.positions() {
            n += 1;
            let part = b.position_from_facts(&pos.facts).unwrap();
            // read back: the graph's units, as text, are exactly the facts' units; the tempo is the fact's
            let back: BTreeSet<(String, String, String)> = part.units.iter().map(|u| (b.names.roles[u.owner].clone(), b.names.kinds[u.kind as usize].clone(), b.names.places[u.at as usize].clone())).collect();
            assert_eq!(back, units_by_text(b, &pos.facts), "{}: units read back", fx.file);
            assert_eq!(part.units.len(), back.len(), "no unit twice");
            let tempo_fact: Vec<&Fact> = pos.facts.iter().filter(|f| Some(&f.relation) == d.tempo.as_ref()).collect();
            assert_eq!(part.tempo().map(|r| b.names.roles[r].clone()), tempo_fact.first().map(|f| f.args[0].clone()), "{}: tempo read back", fx.file);
            // a unit is on its place, ours are the tempo's side, the occupant table says who stands where, ids are definitions
            for (i, u) in part.units.iter().enumerate() {
                assert_eq!(part.occupant[u.at as usize] as usize, i + 1);
                assert_eq!(u.side == Side::Us, Some(u.owner) == part.tempo() || part.tempo().is_none());
                assert_eq!(part.nodes[(u.node - b.nodes.len() as u32) as usize].id, format!("unit:{}:{}:{}", b.names.roles[u.owner], b.names.kinds[u.kind as usize], b.names.places[u.at as usize]));
            }
            assert_eq!(part.occupant.iter().filter(|o| **o != 0).count(), part.units.len());
            assert!(part.relation_count() == 0, "the bare graph holds no relation of the rules");
            units_total += part.units.len();
            for f in &pos.facts { if f.relation != d.units.relation && Some(&f.relation) != d.tempo.as_ref() { other_relations.insert(f.relation.clone()); } }
            // from any order of the facts, or a fact said twice, the same graph
            if n % 7 == 0 {
                let mut mixed = pos.facts.clone();
                for i in (1..mixed.len()).rev() { mixed.swap(i, rng.below(i + 1)); }
                mixed.extend(pos.facts.iter().take(3).cloned());
                let again = b.position_from_facts(&mixed).unwrap();
                assert_eq!(again.definitions(b), part.definitions(b), "{}: the same graph from any order", fx.file);
            }
        }
        println!("OBS 79.21 B-2 {}: graphs built from facts alone for {n} positions, {units_total} units read back equal to the facts' units, the tempo read back; facts of other relations (tags, not the graph's) {:?}", fx.file, other_relations);
        assert!(n >= 300 && units_total > 5000);
    }
}

#[test]
fn a_fact_that_names_nothing_the_board_knows_is_refused() {
    let fx = &fixtures()[0];
    let b = &fx.board;
    let d = b.decl.as_ref().unwrap();
    let mut facts = fx.games[0][0].facts.clone();
    let i = facts.iter().position(|f| f.relation == d.units.relation).unwrap();
    facts[i].args[d.units.owner] = "nobody".to_string();
    assert!(b.position_from_facts(&facts).is_err());
    let mut bare = b.clone();
    bare.decl = None;
    assert!(bare.position_from_facts(&fx.games[0][0].facts).is_err());
}

#[test]
fn the_door_writes_a_relation_with_its_links_and_the_units_between() {
    let fx = &fixtures()[0];
    let b = &fx.board;
    let ns = b.nodes.len() as u32;
    let (mut written, mut with_between) = (0, 0);
    for pos in fx.games[0].iter().step_by(9) {
        let mut part = b.position_from_facts(&pos.facts).unwrap();
        for ui in 0..part.units.len() {
            let at = part.units[ui].at;
            for &(l, _) in &b.place_lines[at as usize] {
                for &q in &b.lines[l as usize].places {
                    if q == at { continue; }
                    b.write(&mut part, NodeType::Bear, ui, q);
                    let r = part.bears.last().unwrap();
                    let expect: Vec<usize> = b.strictly_between(l, at, q).iter().filter(|x| part.occupant[**x as usize] != 0).map(|x| part.occupant[*x as usize] as usize - 1).collect();
                    assert!(expect.iter().all(|e| r.between.contains(e)), "every unit strictly between is linked");
                    let node = &part.nodes[(r.node - ns) as usize];
                    assert_eq!(node.id, format!("bear({},place:{})", part.nodes[(part.units[ui].node - ns) as usize].id, b.names.places[q as usize]));
                    let links: Vec<_> = part.links.iter().filter(|k| k.a == r.node).collect();
                    assert!(links.iter().any(|k| k.kind.word() == "by" && k.b == part.units[ui].node) && links.iter().any(|k| k.kind.word() == "at_place" && k.b == b.place_node[q as usize]));
                    assert_eq!(links.iter().any(|k| k.kind.word() == "at_unit"), part.occupant[q as usize] != 0);
                    written += 1;
                    with_between += !r.between.is_empty() as u32;
                }
            }
        }
    }
    println!("OBS 79.21 B-3 the door: {written} relations written onto bare graphs, each with its node id, its links and the units between ({with_between} with some)");
    assert!(written > 1000 && with_between > 100);
}

/// A board of four columns and three rows, with another set of words for everything and no diagonal: an invented description, nothing like the board of the fixtures.
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

#[test]
fn a_second_game_with_another_board_loads_from_its_description_and_has_its_graph() {
    let places: Vec<String> = ["x", "y", "z", "w"].iter().flat_map(|c| ["1", "2", "3"].iter().map(move |r| format!("{c}{r}"))).collect();
    let steps = read_steps(SMALL, &["step".to_string()], &[("cell".to_string(), vec!["col".to_string(), "row".to_string()])], &places).unwrap();
    let b = Board::new(&steps, &Names::new(places, vec!["piece".into(), "stone".into()], vec!["a".into(), "b".into()])).unwrap()
        .with_decl(FactDecl { units: UnitDecl { relation: "tok".into(), owner: 0, kind: Some(1) }, tempo: Some("turn".into()) });
    assert_eq!(b.directions, vec!["down", "left", "right", "up"]);
    assert_eq!(b.lines.len(), 3 + 4);                                          // three rows of four, four columns of three
    let (right, left) = (b.directions.iter().position(|x| x == "right").unwrap(), b.directions.iter().position(|x| x == "left").unwrap());
    let y2 = b.names.place("y2").unwrap() as usize;
    assert_eq!((b.edge_distance[right][y2], b.edge_distance[left][y2]), (2, 1));
    // the board written and read back identical, and a position of facts of this other game
    assert_eq!(Board::from_json(&b.to_json()).unwrap().to_json(), b.to_json());
    let facts = vec![Fact::new("tok", &["a", "piece", "x", "1"]), Fact::new("tok", &["b", "stone", "z", "3"]), Fact::new("turn", &["b"]), Fact::new("count", &["4"])];
    let part = b.position_from_facts(&facts).unwrap();
    assert_eq!((part.units.len(), part.tempo()), (2, Some(1)));
    assert_eq!(part.nodes.iter().map(|n| n.id.clone()).collect::<Vec<_>>(), vec!["unit:a:piece:x1".to_string(), "unit:b:stone:z3".to_string()]);
    assert!(part.units[1].side == Side::Us && part.units[0].side == Side::Them);
    // a step relation that is not a chain is refused
    let bad = cs_rep::steps::Steps { directions: vec!["d".into()], rows: vec![(0, 0, 1), (0, 0, 2)], ..Default::default() };
    assert!(Board::new(&bad, &Names::new(vec!["p".into(), "q".into(), "r".into()], vec![], vec![])).is_err());
    println!("OBS 79.21 B-4 second board: 4 by 3 places, {} lines, {} directions, read from an invented description, graph of a position of facts with other words", b.lines.len(), b.directions.len());
}
