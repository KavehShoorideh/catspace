//! PLAN 79.24 PART B: THE SEARCH AND THE LAYER SHARE ONE COPY (SYSTEM_SPEC 11 "layer 3 holds two implementations"; piece: value and measure of progress). This test is in the means-ends crate because it names crates above layer 3: the ledger's judge (a table of worths read at a
//! line's captures) and the measure's reading re-exported from cs_tactics::measure are checked against layer 3's own functions (cs_value). Constructed positions only.
use cs_game::kernel::KernelGame;
use cs_game::placed::Placed;
use cs_game::Game;
use cs_tactics::ledger::{Judge, Worth};
use cs_tactics::measure::Geometry;
use cs_tactics::sheet::base_graph::Static;
use cs_tactics::sheet::Names;
use cs_value::worth::material;
use std::collections::HashMap;

type St = <KernelGame as Game>::State;

fn state(fen: &str) -> St { KernelGame::from_fen("chess", fen).unwrap().initial() }

#[test]
fn a_capture_changes_the_layers_material_count_by_what_the_ledgers_judge_adds() {
    let g = KernelGame::new("chess", 518).unwrap();
    let table: HashMap<String, i64> = ["p", "n", "b", "r", "q", "k"].iter().enumerate().map(|(i, k)| (k.to_string(), 3 + 7 * i as i64)).collect(); // the kind words the game reports in its units and changes
    let judge = Worth { name: "a table of the layer's".into(), table: table.clone(), ours_first: false };
    let mut taken = 0;
    for fen in ["4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1", "r3k3/8/8/8/8/8/8/R3K2R w - - 0 1", "4k3/8/2b5/8/8/5N2/8/4K3 w - - 0 1", "4k3/3q4/8/8/8/8/3R4/4K3 w - - 0 1"] {
        let s = state(fen);
        let me = g.mover(&s).unwrap();
        for m in g.moves(&s) {
            let after = g.apply(&s, &m);
            let (before, now) = (material(&g, &s, me, &table), material(&g, &after, me, &table));
            let gone = g.captured(&s, &m);
            let expect: i64 = gone.iter().map(|(owner, kind)| judge.delta(me, *owner, kind)).sum();
            assert_eq!(now - before, expect, "the material count moves by what the judge adds, for {fen}");
            taken += gone.len();
        }
    }
    assert!(taken >= 3, "captures were read");
}

/// PLAN 79.26 PART C: a unit changing kind is counted by the judge the wake plays with, and by the same amount as layer 3's count of material moves (the difference of the two kinds' worths from the record; nothing assumed). The plain count cannot: with every unit worth the same, the difference is nought by what it is.
#[test]
fn a_unit_changing_kind_changes_the_layers_material_count_by_what_the_ledgers_judge_adds() {
    let g = KernelGame::new("chess", 518).unwrap();
    let table: HashMap<String, i64> = ["p", "n", "b", "r", "q", "k"].iter().enumerate().map(|(i, k)| (k.to_string(), 3 + 7 * i as i64)).collect(); // the kind words the game reports in its units and changes
    let judge = Worth { name: "a table of the layer's".into(), table: table.clone(), ours_first: false };
    let (mut changed, mut nonzero) = (0, 0);
    for fen in ["4k3/P7/8/8/8/8/8/4K3 w - - 0 1", "1r2k3/P7/8/8/8/8/8/4K3 w - - 0 1", "4k3/8/8/8/8/8/p7/4K3 b - - 0 1", "n3k3/1P6/8/8/8/8/8/4K3 w - - 0 1"] {
        let s = state(fen);
        let me = g.mover(&s).unwrap();
        for m in g.moves(&s) {
            let after = g.apply(&s, &m);
            let (before, now) = (material(&g, &s, me, &table), material(&g, &after, me, &table));
            let took: i64 = g.captured(&s, &m).iter().map(|(owner, kind)| judge.delta(me, *owner, kind)).sum();
            let turned: i64 = g.changed(&s, &m).iter().map(|(a, b)| judge.change(me, me, a, b)).sum();
            assert_eq!(now - before, took + turned, "the material count moves by what the judge adds for the takings and the changes of kind, for {fen}");
            if !g.changed(&s, &m).is_empty() { changed += 1; if turned != 0 { nonzero += 1; } }
        }
    }
    assert!(changed >= 6 && nonzero >= 4, "moves that change a unit's kind were read: {changed}, counted non-zero {nonzero}");
    // the plain count counts it as nothing, by what it is
    assert_eq!(cs_tactics::ledger::Count.change(0, 0, "p", "q"), 0);
    println!("OBS 79.26 one copy: {changed} moves that change a unit's kind, the material count moves by what the wake's judge adds in all of them ({nonzero} by a non-zero amount); the plain count adds nothing for a change of kind");
}

#[test]
fn the_measures_reading_of_the_old_path_is_the_layers_one() {
    let g = KernelGame::new("chess", 518).unwrap();
    let text = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../../rules/chess.gdl")).unwrap();
    let steps = g.steps(&text, &["diagonal_of".to_string(), "step".to_string()], &[("square".to_string(), vec!["file".to_string(), "rank".to_string()])]).unwrap();
    let st = Static::new(&steps, &Names::new(g.place_names(), g.kind_names(), g.roles())).unwrap();
    let geo = Geometry::new(&st);
    let s = state("7k/8/8/8/8/8/R7/K7 w - - 0 1");
    let term = cs_tactics::measure::Term::parse("[tuple region(1,1) corner(1,1)]").unwrap();
    let old_path = cs_tactics::measure::value(&term, &cs_tactics::measure::Reader::new(&g, &geo, &s, 0));
    let layer = cs_value::progress::value(&term, &cs_value::progress::Reader::new(&g, &geo, &s, 0));
    assert_eq!(old_path, layer);
    let through_the_interface = cs_value::Assess::<KernelGame>::assess(&cs_value::progress::Progress::given(term.clone(), 3, &geo, cs_rep::statement::Standing::Conjecture("a test".into())), &cs_value::Question { game: &g, state: &s, side: 0 }).unwrap();
    assert_eq!(through_the_interface.raw, Some(layer));
}
