//! PLAN 79.21 STAGE C: TESTS OF LAYER 2 FOR THE TAGS (SYSTEM_SPEC 11 "layer 1 holds the history; the rights are layer 2's tags"; piece: representation, the rules above it). On every position of both fixture games the tags kept step by step in make and undo equal the facts the rules give and the tags derived from the whole
//! history from the start; a repeat is a tag read off the history's keys.
mod rep_common;
use cs_game::kernel::KernelGame;
use cs_game::{Fact, Game};
use rep_common::{fixtures, replay, text_of};


fn tag_ids(m: &cs_rep::store::GameMemory) -> std::collections::BTreeSet<String> { m.statements().map(|s| s.id.clone()).collect() }

#[test]
fn layer2_the_tags_kept_step_by_step_equal_the_facts_the_rules_give_and_the_tags_derived_from_the_whole_history() {
    use cs_game::tags::TagKeeper;
    use cs_rep::store::{GameKey, GameMemory};
    use std::collections::BTreeSet;
    for fx in fixtures() {
        let k = KernelGame::new(&fx.name, 518).unwrap();
        let decl = fx.board.decl.clone().unwrap();
        let is_tag = |f: &Fact| f.relation != decl.units.relation && decl.tempo.as_deref() != Some(f.relation.as_str());
        let (mut plies, mut dropped, mut brought, mut undone) = (0usize, 0usize, 0usize, 0usize);
        for moves in &fx.games {
            let states = replay(&k, moves);
            let start = k.base_facts(&states[0]);
            let mut kp = TagKeeper::new(&k, &states[0], &decl, GameMemory::new(GameKey::of(&fx.name, &text_of(&fx.name))));
            let mut seen: Vec<(BTreeSet<Fact>, BTreeSet<String>)> = vec![(kp.tags(), tag_ids(kp.memory()))];
            for i in 1..states.len() {
                kp.make(&k, &states[i]);
                let rules: BTreeSet<Fact> = k.base_facts(&states[i]).into_iter().filter(|f| is_tag(f)).collect();
                assert_eq!(kp.tags(), rules, "{}: ply {i}: the tags kept step by step are the facts the rules give", fx.name);
                let from_history: BTreeSet<Fact> = kp.history().facts_at(&start, i).unwrap().into_iter().filter(|f| is_tag(f)).collect();
                assert_eq!(from_history, rules, "{}: ply {i}: the tags derived from the whole history", fx.name);
                assert_eq!(tag_ids(kp.memory()), rules.iter().map(|f| format!("tag:{f}")).collect::<BTreeSet<_>>(), "{}: ply {i}: one statement for each tag in the memory", fx.name);
                assert_eq!(kp.history().current_key(), k.key(&states[i]));
                let before = &seen[i - 1].0;
                dropped += before.difference(&rules).count();
                brought += rules.difference(before).count();
                seen.push((rules, tag_ids(kp.memory())));
                plies += 1;
            }
            for i in (1..states.len()).rev() {
                assert!(kp.undo());
                assert_eq!((kp.tags(), tag_ids(kp.memory())), seen[i - 1], "{}: undo to ply {}", fx.name, i - 1);
                undone += 1;
            }
            assert!(!kp.undo());
        }
        println!("OBS 79.21 C-4 layer 2 {}: tags kept step by step equal the facts the rules give and the tags derived from the whole history at {plies} plies and {undone} undos; tags that fell {dropped}, tags brought {brought}; one statement in the game's memory for each", fx.name);
        assert!(plies > 300 && dropped > 50);
    }
}

#[test]
fn layer2_a_repeat_is_a_tag_read_off_the_histories_keys() {
    use cs_game::tags::TagKeeper;
    use cs_rep::store::{GameKey, GameMemory};
    let k = KernelGame::new("chess", 518).unwrap();
    let decl = fixtures().into_iter().find(|f| f.name == "chess").unwrap().board.decl.unwrap();
    let mut s = k.initial();
    let mut kp = TagKeeper::new(&k, &s, &decl, GameMemory::new(GameKey::of("chess", &text_of("chess"))));
    assert_eq!(kp.repeat_tag().to_string(), "(repeats 0)");
    let mut keys = vec![k.key(&s)];
    let mut said = vec![];
    for m in ["g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1", "f6g8"] {
        let mv = k.parse(&s, m).unwrap();
        s = k.apply(&s, &mv);
        kp.make(&k, &s);
        let brute = keys.iter().filter(|x| **x == k.key(&s)).count();
        assert_eq!(kp.repeat_tag().to_string(), format!("(repeats {brute})"));
        keys.push(k.key(&s));
        said.push(brute);
    }
    assert!(said.iter().any(|n| *n > 0), "the units went out and back: some position is met again");
    let mut undone = 0;
    while kp.undo() { undone += 1; }
    assert_eq!((undone, kp.repeat_tag().to_string()), (8, "(repeats 0)".to_string()));
    println!("OBS 79.21 C-5 layer 2 a repeat as a tag: eight tempos with the units going out and back, the tag counted {said:?} as the keys met again, as a plain count of the keys says; all undone");
}
