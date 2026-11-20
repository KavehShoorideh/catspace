//! PLAN 79.17 PART B, a second unit test (SYSTEM_SPEC 7; piece: the measure): the region atom is the same whichever side is to move, and it reads the other side's bearing, not its moves (a unit of the other side that stands loose is a leak in the region).
use cs_game::kernel::KernelGame;
use cs_game::Game;
use cs_game::placed::Placed;
use cs_tactics::measure::{Atom, Geometry, Group, Reader};
use cs_tactics::sheet::base_graph::Static;
use cs_tactics::sheet::Names;

#[test]
fn the_region_does_not_depend_on_whose_turn_it_is() {
    let g = KernelGame::new("chess", 518).unwrap();
    let text = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../../rules/chess.gdl")).unwrap();
    let steps = g.steps(&text, &["diagonal_of".to_string(), "step".to_string()], &[("square".to_string(), vec!["file".to_string(), "rank".to_string()])]).unwrap();
    let st = Static::new(&steps, &Names::new(g.place_names(), g.kind_names(), g.roles())).unwrap();
    let geo = Geometry::new(&st);
    let kk = g.kind_names().iter().position(|k| k == "king").unwrap() as u8;
    let value = |fen: &str| { let s = KernelGame::from_fen("chess", fen).unwrap().initial(); Reader::new(&g, &geo, &s, 0).atom(&Atom::Region(Group { theirs: true, kind: kk })) };
    assert_eq!(value("7k/R7/8/8/8/8/8/K7 b - - 0 1"), value("7k/R7/8/8/8/8/8/K7 w - - 0 1"));
    assert_eq!(value("7k/R7/8/8/8/8/8/K7 b - - 0 1"), 48);
}
