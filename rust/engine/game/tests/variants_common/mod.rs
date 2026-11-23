//! PLAN 79.35: what the two test files of the four games share: the fixtures of the games that differ from the standard one, read as opaque text (the format is described in
//! catspace/incubator/analyst/discovery/layer2_fixtures_print.py), the rules' folder and the kernel of a game. A test file may name the games.
#![allow(dead_code)]
use cs_game::kernel::KernelGame;
use cs_game::Fact;
use std::collections::BTreeSet;
use std::path::PathBuf;

pub const GAMES: [&str; 4] = ["chess960", "horde", "atomic", "antichess"];

pub struct Pos { pub label: String, pub text: String, pub end: Option<(String, Vec<i64>)>, pub moves: Vec<String>, pub facts: BTreeSet<Fact>, pub side: String }
pub struct Step { pub parent: String, pub name: String, pub end: Option<(String, Vec<i64>)>, pub added: Vec<Fact>, pub removed: Vec<Fact> }
pub struct Fx {
    pub name: String, pub ends: Vec<String>, pub differs: String, pub perft: Vec<(String, String, Vec<u64>)>, pub pos: Vec<Pos>, pub steps: Vec<Step>,
    pub starts: Vec<(usize, String)>, pub scharnagl: Vec<(usize, String)>, pub lines: Vec<(String, Vec<(String, String)>)>,
}

pub fn facts(s: &str) -> Vec<Fact> {
    s.split(' ').filter(|t| !t.is_empty() && *t != "-").map(|t| { let mut p = t.split(','); Fact { relation: p.next().unwrap().to_string(), args: p.map(|a| a.to_string()).collect() } }).collect()
}

fn end_of(s: &str) -> Option<(String, Vec<i64>)> {
    if s == "-" { return None; }
    let (label, values) = s.split_once('/').unwrap();
    Some((label.to_string(), values.split(',').map(|v| v.parse().unwrap()).collect()))
}

pub fn fixture(game: &str) -> Fx {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests").join("fixtures").join(format!("{game}.fixture"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let mut fx = Fx { name: String::new(), ends: vec![], differs: String::new(), perft: vec![], pos: vec![], steps: vec![], starts: vec![], scharnagl: vec![], lines: vec![] };
    for line in text.lines() {
        let f: Vec<&str> = line.split('\t').collect();
        let words = |s: &str| -> Vec<String> { s.split(' ').filter(|w| !w.is_empty() && *w != "-").map(|w| w.to_string()).collect() };
        match f[0] {
            "game" => fx.name = f[1].to_string(),
            "ends" => fx.ends = words(f[1]),
            "differs" => fx.differs = f[1].to_string(),
            "perft" => fx.perft.push((f[1].to_string(), f[2].to_string(), f[3].split(',').map(|n| n.parse().unwrap()).collect())),
            "pos" => fx.pos.push(Pos { label: f[1].into(), text: f[2].into(), end: end_of(f[4]), moves: words(f[5]), facts: facts(f[6]).into_iter().collect(), side: f[8].into() }),
            "step" => fx.steps.push(Step { parent: f[1].into(), name: f[2].into(), end: end_of(f[3]), added: facts(f[4]), removed: facts(f[5]) }),
            "start" => fx.starts.push((f[1].parse().unwrap(), f[2].into())),
            "scharnagl" => fx.scharnagl.push((f[1].parse().unwrap(), f[2].into())),
            "line" => fx.lines.push((f[1].into(), f[2].split(';').filter(|w| !w.is_empty()).map(|w| { let (a, b) = w.split_once('|').unwrap(); (a.to_string(), b.to_string()) }).collect())),
            "record" => {}
            other => panic!("unknown tag {other}"),
        }
    }
    assert_eq!(fx.name, game);
    fx
}

pub fn rules_root() -> PathBuf { PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../rules") }

/// The game as the kernel plays it (the standard start, or the start numbered 518 of the game with a start by number).
pub fn kernel(game: &str) -> KernelGame { KernelGame::new(game, 518).unwrap() }
