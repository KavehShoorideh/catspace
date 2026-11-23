//! PLAN 79.36 PART C, LAYER 2: THE COUNT OF GRAPH BUILDS (SYSTEM_SPEC 11, last block; piece: rules; ledger row 43). The counter is one switch for the whole process, so this file holds ONE test and no other test shares its process.
use cs_game::graph::builds;
use cs_game::kernel::KernelGame;
use cs_game::rules::Ruled;
use cs_game::Game;
use std::path::PathBuf;

#[test]
fn the_builds_of_a_graph_are_counted_by_the_place_that_asked_from_start_to_take_and_not_before_or_after() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../rules");
    let g = Ruled::open(KernelGame::new("chess", 0).unwrap(), "chess", &root, "learned/chess/graph_vocabulary.json").unwrap();
    let st = g.board().unwrap();
    let s = g.initial();
    let after = g.apply(&s, &g.find_move(&s, "e2e4").unwrap());
    // off: nothing is counted
    assert!(!builds::on());
    let _ = st.position(&g, &s);
    assert!(builds::take().is_empty());
    // on: a build of the root's position and two builds of another are told apart by the key given at the start
    builds::start(g.key(&s));
    assert!(builds::on());
    let _ = st.position(&g, &s);
    let _ = st.position(&g, &after);
    let _ = st.position(&g, &after);
    let counted = builds::take();
    assert!(!builds::on(), "taking the count stops it");
    let root_builds: usize = counted.iter().filter(|(is_root, _, _)| *is_root).map(|(_, _, n)| *n).sum();
    let other_builds: usize = counted.iter().filter(|(is_root, _, _)| !*is_root).map(|(_, _, n)| *n).sum();
    assert_eq!((root_builds, other_builds), (1, 2));
    assert!(counted.iter().all(|(_, at, _)| at.starts_with("units_builds.rs:")), "the place that asked is this file and a line: {counted:?}");
    // a count begun again drops the earlier one
    builds::start(0);
    let _ = st.position(&g, &s);
    builds::start(0);
    assert!(builds::take().is_empty());
}
