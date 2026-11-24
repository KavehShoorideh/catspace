//! PLAN 79.50 PART C (SYSTEM_SPEC 11 "The description is the source of the rules; fast code is only a certified copy"; piece: rules; ledger row 44): how often the kernel's `reach_from` now has a place the one-step answer lacked,
//! counted over the fixture positions of the layer tests and the states one move after them. A test file may name games and their units. Nothing here pins a number: it prints OBS lines and holds only that every place
//! added (same file, two rows on) is the end of a move the rules allow in a position where the unit is not held by a pin.
use cs_game::kernel::KernelGame;
use cs_game::placed::Placed;
use cs_game::Game;
use std::path::PathBuf;

#[test]
fn the_places_the_new_reach_adds_over_the_fixture_positions() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let mut files: Vec<_> = std::fs::read_dir(&dir).unwrap().map(|e| e.unwrap().path()).filter(|p| p.extension().is_some_and(|x| x == "fixture")).collect();
    files.sort();
    let (mut all_states, mut all_gain, mut all_extra) = (0usize, 0usize, 0usize);
    for f in files {
        let text = std::fs::read_to_string(&f).unwrap();
        let game = text.lines().next().unwrap().split('\t').nth(1).unwrap().to_string();
        let mut fens: Vec<String> = text.lines().filter(|l| l.starts_with("perft\t") || l.starts_with("pos\t")).filter_map(|l| l.split('\t').nth(2).map(|x| x.to_string())).collect();
        fens.sort();
        fens.dedup();
        let (mut states, mut gain, mut extra, mut allowed_extra) = (0usize, 0usize, 0usize, 0usize);
        for fen in &fens {
            let Ok(k) = KernelGame::from_fen(&game, fen) else { continue };
            let start = k.initial();
            let mut seen = vec![start.clone()];
            for m in k.moves(&start) { seen.push(k.apply(&start, &m)); }
            let names = k.place_names();
            let kinds = k.kind_names();
            for s in &seen {
                if k.outcome(s).is_some() { continue; }
                let Some(mover) = k.mover(s) else { continue };
                states += 1;
                let mut here = 0;
                for (owner, kind, at) in k.cells(s) {
                    if owner != mover || kinds[kind as usize] != "pawn" { continue; }
                    let a = names[at as usize].as_bytes();
                    for to in k.reach_from(s, owner, kind, at) {
                        let t = names[to as usize].as_bytes();
                        if t[0] == a[0] && (t[1] as i32 - a[1] as i32).abs() == 2 {
                            here += 1;
                            if k.find_step(s, at, to).is_some() { allowed_extra += 1; }
                        }
                    }
                }
                if here > 0 { gain += 1; extra += here; }
            }
        }
        println!("OBS 79.50 C {game}: fixture positions {}, states read (each and the states one move after) {states}, states where the new reach has a place the one-step answer lacked {gain}, such places {extra}, of them the end of a move the rules allow {allowed_extra}", fens.len());
        all_states += states; all_gain += gain; all_extra += extra;
    }
    println!("OBS 79.50 C all games: states {all_states}, states with a place added {all_gain}, places added {all_extra}");
    assert!(all_gain > 0, "the fixtures hold units on their first row");
}
