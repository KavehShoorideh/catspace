//! PLAN 79.20 PART A: THE MAKER OF THE FIXTURES OF LAYER 1 (SYSTEM_SPEC 11 "a layer's test uses only that layer and what is below it"; piece: representation).
//! THIS IS NOT A TEST. It is the script that makes the data the tests of cs_rep read: it plays seeded random games of two games (the standard game and one variant of the repo) through the game's kernel
//! and writes, for each, one file `rust/engine/rep/tests/fixtures/<game>.fixture` of positions as SETS OF FACTS. It is the only place that uses the rules; the tests of layer 1 read the files and call nothing of the rules.
//! Run once:  cd rust/engine && cargo run --release --offline -p cs_decision --example make_layer1_fixtures -- <out dir>
//! File format (text, one line each, tab separated; the tag first):
//!   key      <json {name, rules}>          the game's name and the content address of its rules (what a partition is keyed by)
//!   board    <json>                        the game's board as layer 1 reads it (places, kinds, roles, steps by direction, and where the facts hold units and the tempo): `Board::to_json`, read from the game's description by the rules' binding (plan 79.21)
//!   game     <n>                           a generated game starts
//!   moves    <names...>                    the record of the moves of that game (data for the layer above; layer 1 ignores it)
//!   full     <facts>                       the first position of a game, its facts in the canonical text (`encode_facts`)
//!   delta    <json {added, removed}>       the next position: what was added and removed, each a list of facts in the canonical form, taken by a plain set difference here
//!   recap    <facts>                       after every tenth delta: the position in full, to be equal to the one the deltas built
//! The next position of a game is the last one with the delta applied.
use cs_game::kernel::KernelGame;
use cs_game::placed::Placed;
use cs_rep::graph::{Board, FactDecl};
use cs_rep::names::Names;
use cs_rep::UnitDecl;
use cs_game::rng::Rng;
use cs_game::{Fact, Game};
use cs_gdl::compose;
use cs_memory::statement::encode_facts;
use cs_memory::store::GameKey;
use std::collections::BTreeSet;
use std::path::PathBuf;

const GAMES: [(&str, &str); 2] = [("chess", ""), ("kingofthehill", "variants/kingofthehill.gdl")];
const PLAYED: usize = 10;
const PLIES: usize = 40;
const CHECKPOINT: usize = 10;

fn rules(rel: &str) -> String { std::fs::read_to_string(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../rules").join(rel)).unwrap() }

fn main() {
    let out = PathBuf::from(std::env::args().nth(1).expect("the output directory"));
    std::fs::create_dir_all(&out).unwrap();
    for (name, variant) in GAMES {
        let text = if variant.is_empty() { rules("chess.gdl") } else { compose(&rules("chess.gdl"), &rules(variant)).unwrap() };
        let key = GameKey::of(name, &text);
        let k = KernelGame::new(name, 518).unwrap();
        let mut rng = Rng::new(2020_1120 + name.bytes().fold(0u64, |h, b| h.wrapping_mul(131).wrapping_add(b as u64)));
        let steps = k.steps(&text, &["diagonal_of".to_string(), "step".to_string()], &[("square".to_string(), vec!["file".to_string(), "rank".to_string()])]).unwrap();
        let board = Board::new(&steps, &Names::new(k.place_names(), k.kind_names(), k.roles())).unwrap().with_decl(FactDecl { units: UnitDecl { relation: "cell".into(), owner: 2, kind: Some(3) }, tempo: Some("control".into()) });
        let mut file = format!("key\t{}\nboard\t{}\n", key.to_json(), board.to_json());
        let mut positions = 0;
        for g in 0..PLAYED {
            let (mut s, mut moves) = (k.initial(), vec![]);
            let mut line = vec![];                                         // the facts of each position of the game
            for _ in 0..PLIES {
                line.push(k.base_facts(&s).into_iter().collect::<BTreeSet<Fact>>());
                let legal = k.moves(&s);
                if legal.is_empty() || k.outcome(&s).is_some() { break; }
                let m = legal[rng.below(legal.len())].clone();
                moves.push(k.move_name(&s, &m));
                s = k.apply(&s, &m);
            }
            file += &format!("game\t{g}\nmoves\t{}\n", moves.join(" "));
            for (i, facts) in line.iter().enumerate() {
                let v: Vec<Fact> = facts.iter().cloned().collect();
                if i == 0 { file += &format!("full\t{}\n", encode_facts(&v)); }
                else {
                    let (added, removed): (Vec<Fact>, Vec<Fact>) = (facts.difference(&line[i - 1]).cloned().collect(), line[i - 1].difference(facts).cloned().collect());
                    file += &format!("delta\t{{\"added\":{},\"removed\":{}}}\n", encode_facts(&added), encode_facts(&removed));
                    if i % CHECKPOINT == 0 { file += &format!("recap\t{}\n", encode_facts(&v)); }
                }
                positions += 1;
            }
        }
        std::fs::write(out.join(format!("{name}.fixture")), &file).unwrap();
        println!("{name}: {PLAYED} games, {positions} positions, {} bytes", file.len());
    }
}
