//! PLAN 79.22 PARTS A AND B: TESTS OF LAYER 2 FOR THE GAME'S KEPT POSITION (SYSTEM_SPEC 11 "layer 1 holds the history", "one graph is the representation"; piece: representation and its use by the wake; `cs_decision::keep`).
//! Over the whole games of both fixture games, a game at hand's memory is brought to the position of each decision (`keep::bring`) and the move the decision chose is played in it (`keep::advance`):
//!   (A) the root graph the wake is lent equals a fresh build at EVERY decision, whether the memory is one tempo behind (our move played in it), two tempos behind (the caller never played ours in it), or has to be built afresh (the caller played another move);
//!   (B) the tags the memory holds equal the facts the rules give (the facts that are neither a unit nor the tempo) at every decision, one statement for each; the history holds the keys given with the change known for the tempos it saw; a repeat is read off the history's keys; and a game
//!       STARTED IN THE MIDDLE states its tags as plain facts and decides the same as a game that was kept from its start.
mod rep_common;
use cs_decision::keep::{advance, bring, root_of, Brought};
use cs_decision::memory::{Parts, WakeMemory};
use cs_decision::threat_verdict::Record;
use cs_decision::wake::{decide_seen, Pieces, WakeInputs, WakeParams};
use cs_game::kernel::KernelGame;
use cs_game::{Fact, Game};
use cs_rep::graph::PositionPart;
use cs_rep::store::{GameKey, GameMemory};
use cs_tactics::ledger::Worth;
use cs_tactics::sheet::base_graph::Static;
use rep_common::{fixtures, replay, text_of};
use std::collections::{BTreeSet, HashMap};

fn same(a: &PositionPart, b: &PositionPart) -> bool {
    a.mover == b.mover && a.nodes == b.nodes && a.links == b.links && a.units == b.units && a.bears == b.bears && a.moves == b.moves && a.shuts == b.shuts && a.throughs == b.throughs && a.occupant == b.occupant
}

type S = <KernelGame as Game>::State;

fn tags_held(m: &GameMemory) -> BTreeSet<String> { m.statements().filter(|s| s.id.starts_with("tag:")).map(|s| s.id.clone()).collect() }

fn rules_tags(k: &KernelGame, decl: &cs_rep::graph::FactDecl, s: &S) -> BTreeSet<String> {
    k.base_facts(s).into_iter().filter(|f: &Fact| f.relation != decl.units.relation && decl.tempo.as_deref() != Some(f.relation.as_str())).map(|f| format!("tag:{f}")).collect()
}

/// a move of the position that is not `not` (the first in the rules' order), to play another game on
fn other_move(k: &KernelGame, s: &S, not: &S) -> S {
    for m in k.moves(s) { let a = k.apply(s, &m); if k.key(&a) != k.key(not) { return a; } }
    k.apply(s, &k.moves(s)[0])
}

#[derive(Default)]
struct Tally { decisions: usize, same_p: usize, fresh: usize, one: usize, two: usize, tags_checked: usize, repeats_checked: usize, known: usize }

#[test]
fn layer2_the_kept_root_graph_and_tags_equal_a_fresh_build_and_the_rules_at_every_decision_of_whole_games() {
    for fx in fixtures() {
        let k = KernelGame::new(&fx.name, 518).unwrap();
        let decl = fx.board.decl.clone().unwrap();
        let mut st = Static::from_board(fx.board.clone());
        let first = replay(&k, &fx.games[0]);
        let p = st.position(&k, &first[30.min(first.len() - 1)]);
        st.learn(&p);
        let key = GameKey::of(&fx.name, &text_of(&fx.name));
        for mode in ["our move played in the memory", "the caller never played ours", "the caller played another move", "every ply a decision, nothing played in the memory", "every ply a decision, the move played in the memory"] {
            let mut t = Tally::default();
            for moves in fx.games.iter().take(6) {
                let states = replay(&k, moves);
                let keys: Vec<u64> = states.iter().map(|s| k.key(s)).collect();
                let mut mem = GameMemory::new(key.clone());
                let step = if mode.starts_with("every ply") { 1 } else { 2 };          // the decisions of one seat (every other ply), or of both seats sharing the memory (every ply)
                let mut i = 0;
                while i < states.len() {
                    let how = bring(&st, &k, &decl, &mut mem, &keys[..i], &states[i]);
                    t.decisions += 1;
                    match how { Brought::Fresh => t.fresh += 1, Brought::Same => t.same_p += 1, Brought::Continued { tempos: 1 } => t.one += 1, Brought::Continued { .. } => t.two += 1 }
                    // (A) the root graph the wake is lent
                    let root = root_of(&st, &k, &states[i], &mem);
                    assert!(same(root.part(), &st.position(&k, &states[i])), "{} [{mode}]: ply {i}: the kept root graph differs from a fresh build", fx.name);
                    // (B) the tags: one statement for each fact the rules give that is neither a unit nor the tempo
                    assert_eq!(tags_held(&mem), rules_tags(&k, &decl, &states[i]), "{} [{mode}]: ply {i}: the tags held are the rules' facts", fx.name);
                    t.tags_checked += 1;
                    // the history holds the keys given, and a repeat is read off them
                    let h = mem.history().expect("a history");
                    assert_eq!(h.keys(), &keys[..=i], "{} [{mode}]: ply {i}: the history's keys", fx.name);
                    let brute = keys[..i].iter().filter(|x| **x == keys[i]).count();
                    assert_eq!((h.repeats(), h.seen().count(keys[i])), (brute, brute));
                    t.repeats_checked += 1;
                    if how == (Brought::Continued { tempos: 1 }) || how == Brought::Same { assert!(h.change_known(i - 1) || i == 0); t.known += 1; }
                    // the decision plays its move in the memory (the move played in the game, or another one, or none)
                    if i + 1 < states.len() {
                        if mode.starts_with("our move") || mode.ends_with("the move played in the memory") { advance(&st, &k, &decl, &mut mem, &states[i + 1]); }
                        else if mode.starts_with("the caller played another") { let o = other_move(&k, &states[i], &states[i + 1]); advance(&st, &k, &decl, &mut mem, &o); }
                    }
                    i += step;
                }
            }
            println!("OBS 79.22 A/B {} [{mode}]: {} decisions in whole games, the root graph the wake is lent equals a fresh build in all, the tags held equal the rules' facts in {}, the history's keys and the repeat read off them equal in {}; brought: {} afresh, {} by one tempo, {} by several, {} asked again at the same position", fx.name, t.decisions, t.tags_checked, t.repeats_checked, t.fresh, t.one, t.two, t.same_p);
            match mode {
                "our move played in the memory" => assert!(t.one > 100 && t.fresh == 6 && t.two == 0, "{} {mode}: the other side's move alone, one tempo each time", fx.name),
                "the caller never played ours" => assert!(t.two > 100 && t.fresh == 6 && t.one == 0, "{} {mode}: two tempos at once", fx.name),
                "the caller played another move" => assert!(t.fresh > 100 && t.one == 0, "{} {mode}: the history does not continue: afresh", fx.name),
                "every ply a decision, nothing played in the memory" => assert!(t.one > 200 && t.fresh == 6, "{} {mode}: one tempo each time", fx.name),
                _ => assert!(t.same_p > 200 && t.fresh == 6, "{} {mode}: the memory is already there", fx.name),
            }
        }
    }
}

fn params(k: &KernelGame, keep: Option<cs_rep::graph::FactDecl>, memory: GameMemory) -> WakeParams {
    let table: HashMap<String, i64> = [("p", 475), ("n", 1216), ("b", 1301), ("r", 1945), ("q", 3583)].iter().map(|(a, b)| (a.to_string(), *b)).collect();
    let goal = cs_game::placed::Placed::kind_names(k).iter().position(|x| x == "king").map(|i| i as u8);
    let wm = WakeMemory::of(GameKey::of("chess", "t"), Parts { record: Some(Record { worth: table, half: HashMap::new() }), ..Default::default() }, Some(memory));
    WakeParams { seconds: 600.0, depth: 3, pieces: Pieces { threats: true, goals: true, direct_names: true, ..Pieces::old_player() }, places: vec![], inputs: WakeInputs { memory: wm, goal, budget: 4, keep, ..WakeInputs::default() } }
}

#[test]
fn layer2_a_game_started_in_the_middle_states_its_tags_as_facts_and_decides_the_same() {
    let k = KernelGame::new("chess", 518).unwrap();
    let fx = fixtures().into_iter().find(|f| f.name == "chess").unwrap();
    let decl = fx.board.decl.clone().unwrap();
    let mut st = Static::from_board(fx.board.clone());
    let states = replay(&k, &fx.games[0]);
    let p = st.position(&k, &states[30.min(states.len() - 1)]);
    st.learn(&p);
    let keys: Vec<u64> = states.iter().map(|s| k.key(s)).collect();
    let judge = Worth { name: "record".into(), table: [("p", 475), ("n", 1216), ("b", 1301), ("r", 1945), ("q", 3583)].iter().map(|(a, b)| (a.to_string(), *b)).collect(), ours_first: false };
    let (mut compared, mut stated) = (0, 0);
    for start in [20usize, 41, 66] {
        let start = start.min(states.len() - 2);
        // the game kept from its start, decided at each ply from `start - 6`, our move played in the memory each time
        let mut kept = GameMemory::new(GameKey::of("chess", "t"));
        for i in (start - 6..start).step_by(2) { bring(&st, &k, &decl, &mut kept, &keys[..i], &states[i]); advance(&st, &k, &decl, &mut kept, &states[i + 1]); }
        assert_eq!(bring(&st, &k, &decl, &mut kept, &keys[..start], &states[start]), Brought::Continued { tempos: 1 });
        // the game started in the middle: the same position with the keys before it and no memory
        let mut mid = GameMemory::new(GameKey::of("chess", "t"));
        assert_eq!(bring(&st, &k, &decl, &mut mid, &keys[..start], &states[start]), Brought::Fresh);
        assert_eq!(tags_held(&mid), rules_tags(&k, &decl, &states[start]), "started in the middle: the tags are stated as the rules' facts");
        assert_eq!(tags_held(&mid), tags_held(&kept), "started in the middle and kept from the start hold the same tags");
        assert!(mid.history().unwrap().keys() == &keys[..=start] && !mid.history().unwrap().change_known(start - 1), "started in the middle: the keys are held, the changes into them are not known");
        stated += mid.statements().filter(|s| s.id.starts_with("tag:")).count();
        // the decisions are the same: kept, started in the middle, and with nothing kept at all (the wake's own record from the keys)
        let a = decide_seen(&st, &k, &states[start], kept.history().unwrap().seen(), &judge, &params(&k, Some(decl.clone()), kept.clone()));
        let b = decide_seen(&st, &k, &states[start], mid.history().unwrap().seen(), &judge, &params(&k, Some(decl.clone()), mid.clone()));
        let c = cs_decision::wake::decide(&st, &k, &states[start], &keys[..start], &judge, &params(&k, None, GameMemory::new(GameKey::of("chess", "t"))));
        for (n, d) in [("started in the middle", &b), ("nothing kept", &c)] {
            assert_eq!((&a.mv, a.value, a.depth, a.nodes, &a.named_mine, &a.equal), (&d.mv, d.value, d.depth, d.nodes, &d.named_mine, &d.equal), "the decision kept from the start equals the one {n}");
        }
        // and the decision played its move in the memory: the position at hand is one tempo on
        let h = a.game.history().unwrap();
        assert_eq!(h.len(), kept.history().unwrap().len() + 1, "the decision's own move is in the history");
        compared += 1;
    }
    println!("OBS 79.22 B game started in the middle: {compared} positions, each stated its {} tags as facts (history keys held, changes not known), held the tags of the game kept from its start, and decided the same move, value, depth, nodes and moves named as the kept game and as a decision with no memory kept", stated / compared.max(1));
}
