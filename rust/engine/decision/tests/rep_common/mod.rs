//! PLAN 79.21: what the tests of layer 2 over the fixtures of layer 1 share: the fixture files (the board and the record of moves of each generated game), the replay of a record by the rules, and the board read from the description.
#![allow(dead_code)]
use cs_game::kernel::KernelGame;
use cs_game::placed::Placed;
use cs_game::Game;
use cs_rep::graph::Board;
use cs_rep::names::Names;
use std::path::PathBuf;

pub fn rules(rel: &str) -> String { std::fs::read_to_string(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../rules").join(rel)).unwrap() }
pub fn text_of(variant: &str) -> String { match variant { "chess" => rules("chess.gdl"), v => cs_gdl::compose(&rules("chess.gdl"), &rules(&format!("variants/{v}.gdl"))).unwrap() } }

/// One fixture file: the board and the record of moves of each game.
pub struct Fx { pub name: String, pub board: Board, pub games: Vec<Vec<String>> }

pub fn fixtures() -> Vec<Fx> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../rep/tests/fixtures");
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir).unwrap().map(|e| e.unwrap().path()).filter(|p| p.extension().is_some_and(|x| x == "fixture")).collect();
    files.sort();
    files.iter().map(|path| {
        let (mut name, mut board, mut games) = (String::new(), None, vec![]);
        for line in std::fs::read_to_string(path).unwrap().lines() {
            let (tag, rest) = line.split_once('\t').unwrap_or((line, ""));
            match tag {
                "key" => name = serde_json::from_str::<serde_json::Value>(rest).unwrap()["name"].as_str().unwrap().to_string(),
                "board" => board = Some(Board::from_json(&serde_json::from_str(rest).unwrap()).unwrap()),
                "moves" => games.push(rest.split(' ').filter(|m| !m.is_empty()).map(|m| m.to_string()).collect()),
                _ => {}
            }
        }
        Fx { name, board: board.unwrap(), games }
    }).collect()
}

/// The positions of one game of a fixture, replayed from its record of moves (the rules play them: this is layer 2).
pub fn replay(k: &KernelGame, moves: &[String]) -> Vec<<KernelGame as Game>::State> {
    let mut s = k.initial();
    let mut out = vec![s.clone()];
    for m in moves { let mv = k.parse(&s, m).expect("a move of the record is legal"); s = k.apply(&s, &mv); out.push(s.clone()); }
    out
}

pub fn board_from_rules(k: &KernelGame, name: &str, like: &Board) -> Board {
    let steps = k.steps(&text_of(name), &["diagonal_of".to_string(), "step".to_string()], &[("square".to_string(), vec!["file".to_string(), "rank".to_string()])]).unwrap();
    let b = Board::new(&steps, &Names::new(k.place_names(), k.kind_names(), k.roles())).unwrap();
    match &like.decl { Some(d) => b.with_decl(d.clone()), None => b }
}

