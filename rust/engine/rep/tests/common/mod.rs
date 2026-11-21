//! PLAN 79.20 PART B: what the tests of layer 1 share. The positions are read from the fixture files (tests/fixtures/*.fixture, made once by
//! rust/engine/decision/examples/make_layer1_fixtures.rs): each position is a SET OF FACTS, and nothing here asks the rules of the game anything.
#![allow(dead_code)]
use cs_rep::Fact;
use cs_rep::statement::decode_facts;
use cs_rep::graph::Board;
use cs_rep::store::GameKey;
use std::collections::BTreeSet;
use std::path::PathBuf;

/// A seeded generator (SplitMix64) of this layer's own tests: no dependency.
pub struct Rng(u64);
impl Rng {
    pub fn new(seed: u64) -> Rng { Rng(seed) }
    pub fn below(&mut self, n: usize) -> usize {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) % n as u64) as usize
    }
}

/// One position: its facts (sorted, no repeats) and what was added and removed since the one before it (None for the first of a game), as the maker's plain set difference wrote it.
pub struct Pos { pub facts: Vec<Fact>, pub since: Option<(Vec<Fact>, Vec<Fact>)> }

/// One game's file: the key of the game and its generated games, each a line of positions.
pub struct Fixture { pub file: String, pub key: GameKey, pub board: Board, pub games: Vec<Vec<Pos>>, pub checks: usize }

impl Fixture {
    pub fn positions(&self) -> impl Iterator<Item = &Pos> { self.games.iter().flatten() }
    /// every pair (before, after) of consecutive positions, with what changed between them
    pub fn pairs(&self) -> impl Iterator<Item = (&Pos, &Pos)> { self.games.iter().flat_map(|g| g.windows(2).map(|w| (&w[0], &w[1]))) }
}

fn facts_of(set: &BTreeSet<Fact>) -> Vec<Fact> { set.iter().cloned().collect() }

/// Every fixture file of the directory, in the order of their names. A position after the first is the one before it with the delta applied (plain set operations); every `recap` line must equal what the deltas built.
pub fn fixtures() -> Vec<Fixture> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests").join("fixtures");
    let mut names: Vec<PathBuf> = std::fs::read_dir(&dir).unwrap().map(|e| e.unwrap().path()).filter(|p| p.extension().is_some_and(|x| x == "fixture")).collect();
    names.sort();
    assert!(names.len() >= 2, "two games' fixtures are expected in {}", dir.display());
    names.iter().map(|path| {
        let text = std::fs::read_to_string(path).unwrap();
        let (mut key, mut board, mut games, mut cur, mut checks): (Option<GameKey>, Option<Board>, Vec<Vec<Pos>>, Option<BTreeSet<Fact>>, usize) = (None, None, vec![], None, 0);
        for line in text.lines() {
            let (tag, rest) = line.split_once('\t').unwrap_or((line, ""));
            match tag {
                "key" => key = Some(GameKey::from_json(&serde_json::from_str(rest).unwrap()).unwrap()),
                "board" => board = Some(Board::from_json(&serde_json::from_str(rest).unwrap()).unwrap()),
                "game" => games.push(vec![]),
                "moves" => {}                                                   // the record of the moves: data for the layer above, unread here
                "full" => { let s: BTreeSet<Fact> = decode_facts(rest).unwrap().into_iter().collect(); games.last_mut().unwrap().push(Pos { facts: facts_of(&s), since: None }); cur = Some(s); }
                "delta" => {
                    let v: serde_json::Value = serde_json::from_str(rest).unwrap();
                    let (added, removed) = (decode_facts(&v["added"].to_string()).unwrap(), decode_facts(&v["removed"].to_string()).unwrap());
                    let mut s = cur.take().expect("a delta follows a position");
                    for f in &removed { assert!(s.remove(f), "a removed fact was there"); }
                    for f in &added { assert!(s.insert(f.clone()), "an added fact was not there"); }
                    games.last_mut().unwrap().push(Pos { facts: facts_of(&s), since: Some((added, removed)) });
                    cur = Some(s);
                }
                "recap" => { assert_eq!(decode_facts(rest).unwrap(), facts_of(cur.as_ref().unwrap()), "a checkpoint equals what the deltas built"); checks += 1; }
                other => panic!("unknown tag {other}"),
            }
        }
        Fixture { file: path.file_name().unwrap().to_string_lossy().to_string(), key: key.expect("a key"), board: board.expect("a board"), games, checks }
    }).collect()
}
