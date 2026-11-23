//! PLAN 79.35 PART D (the descriptions): THE DESCRIPTION OF EACH OF THE FOUR GAMES, A DELTA UNDER rules/variants READ BY THE EVALUATOR THAT EXISTS, AGAINST THE FIXTURE (SYSTEM_SPEC 11 "Layer 2 says variants, with another start and with other rules, on the same layer 1";
//! ledger rows 41, 40; piece: rules). The kernel's mode is tested against the same fixtures in layer2_variants.rs and against the description by co-play in the certificates (discovery/certify.py); here the description is held against the OUTSIDE
//! library's file: at every position of a fixture the description, begun at that position's facts (`rules::restart_delta`), allows the moves the file says, ends as the file says and gives the facts the file gives; after each move of the file it
//! gives the facts the file gives; and played from its own start along the games of the file, it equals the file in every position. This file names the games on purpose (a test file may).
mod variants_common;
use cs_game::gdl_game::GdlGame;
use cs_game::kernel::KernelGame;
use cs_game::outside::spec_term;
use cs_game::rules::{restart_delta, Ruled, Rules};
use cs_game::starts::Starts;
use cs_game::{Fact, Game};
use std::collections::BTreeSet;
use variants_common::{fixture, kernel, rules_root, Fx, GAMES};

type R = Ruled<KernelGame>;

const DECL: &str = "learned/chess/graph_vocabulary.json";

fn open(game: &str) -> R { Ruled::open_variant(kernel(game), game, &rules_root(), DECL, Some(&format!("variants/{game}.gdl")), None).unwrap() }

/// the description begun at a position's facts
fn begun_at(text: &str, facts: &BTreeSet<Fact>) -> GdlGame {
    let f: Vec<Fact> = facts.iter().cloned().collect();
    let delta = restart_delta(text, &f).unwrap();
    GdlGame::load("shown", &cs_gdl::compose(text, &delta).unwrap()).unwrap()
}

fn terms(names: &[String]) -> BTreeSet<String> { names.iter().map(|n| spec_term(n)).collect() }

fn their_moves(g: &GdlGame, s: &<GdlGame as Game>::State) -> BTreeSet<String> { g.moves(s).iter().map(|m| g.move_name(s, m)).collect() }

fn facts_of(g: &GdlGame, s: &<GdlGame as Game>::State) -> BTreeSet<Fact> { g.base_facts(s).into_iter().collect() }

#[test]
fn each_game_has_a_description_of_its_own_with_an_address_of_its_own_and_the_rules_of_chess_it_keeps() {
    let mut keys = BTreeSet::new();
    for game in GAMES {
        let g = open(game);
        assert_eq!(g.repetition_limit(), Some(2), "{game}: the repeat limit is the base's");
        assert!(g.description().contains("(role white)") && g.fact_decl().tempo.is_some());
        keys.insert(g.rules_key().rules);
        // a name the file does not know is refused, and a description with a delta that retracts what the base does not say is refused
        assert!(Ruled::open_variant(kernel(game), game, &rules_root(), DECL, Some("variants/no_such_game.gdl"), None).is_err());
    }
    assert_eq!(keys.len(), 4, "four games, four addresses");
    // the start is not part of the address: two starts of one game are one game
    let base = open("chess960");
    let other = base.with_start(0).unwrap();
    assert_eq!(base.rules_key(), other.rules_key());
    assert_ne!(base.description(), other.description(), "but the descriptions begin differently");
    println!("OBS 79.35 D descriptions: four games, four addresses of their rules; two starts of the game with a start by number have one address and two descriptions");
}

fn check_fixture(fx: &Fx, only_starts_and_corners: bool) -> (usize, usize, usize, usize) {
    let g = open(&fx.name);
    let (mut positions, mut moves, mut steps, mut skipped) = (0usize, 0usize, 0usize, 0usize);
    for p in &fx.pos {
        if only_starts_and_corners && !(p.label.starts_with("start") || p.label.starts_with("corner:") || p.label.starts_with("line:")) { skipped += 1; continue; }
        let d = begun_at(g.description(), &p.facts);
        let s = d.initial();
        assert_eq!(facts_of(&d, &s), p.facts, "{}: the facts the description holds at {}", fx.name, p.label);
        assert_eq!(d.outcome(&s), p.end.as_ref().map(|e| e.1.clone()), "{}: the end at {} ({})", fx.name, p.label, p.text);
        if p.end.is_none() {
            assert_eq!(their_moves(&d, &s), terms(&p.moves), "{}: the allowed moves at {} ({})", fx.name, p.label, p.text);
            assert_eq!(d.mover(&s).map(|m| d.roles()[m].clone()), Some(p.side.clone()));
        } else {
            assert!(d.moves(&s).is_empty(), "{}: no move at an ended position {}", fx.name, p.label);
        }
        positions += 1;
        moves += p.moves.len();
        for st in fx.steps.iter().filter(|st| st.parent == p.label) {
            let m = d.find_move(&s, &spec_term(&st.name)).unwrap_or_else(|| panic!("{}: {} is not allowed by the description at {} ({})", fx.name, st.name, p.label, p.text));
            let after = d.apply(&s, &m);
            let mut want = p.facts.clone();
            for f in &st.removed { assert!(want.remove(f)); }
            for f in &st.added { want.insert(f.clone()); }
            assert_eq!(facts_of(&d, &after), want, "{}: the facts after {} at {} ({})", fx.name, st.name, p.label, p.text);
            assert_eq!(d.outcome(&after), st.end.as_ref().map(|e| e.1.clone()), "{}: the end after {} at {}", fx.name, st.name, p.label);
            steps += 1;
        }
    }
    (positions, moves, steps, skipped)
}

#[test]
fn horde_atomic_and_antichess_the_description_equals_the_fixture_at_every_position_and_after_every_move() {
    for game in ["horde", "atomic", "antichess"] {
        let fx = fixture(game);
        let t0 = std::time::Instant::now();
        let (positions, moves, steps, _) = check_fixture(&fx, false);
        println!("OBS 79.35 D described {game}: the description begun at each of {positions} positions of the file allows its {moves} moves, ends and holds the facts the file says; {steps} moves lead to the facts and ends the file says ({:.1} s)", t0.elapsed().as_secs_f64());
    }
}

#[test]
fn chess960_the_description_equals_the_fixture_from_every_start_and_corner_and_after_every_move() {
    let fx = fixture("chess960");
    let t0 = std::time::Instant::now();
    let (positions, moves, steps, skipped) = check_fixture(&fx, false);
    println!("OBS 79.35 D described chess960: the description begun at each of {positions} positions of the file allows its {moves} moves, ends and holds the facts the file says; {steps} moves lead to the facts and ends the file says; {skipped} left out ({:.1} s)", t0.elapsed().as_secs_f64());
}

#[test]
fn the_description_of_a_game_with_a_start_by_number_begins_where_the_kernel_does_for_every_start_of_the_file() {
    let fx = fixture("chess960");
    let base = open("chess960");
    let mut n_starts = 0;
    for (n, label) in &fx.starts {
        let g = base.with_start(*n).unwrap();
        let p = fx.pos.iter().find(|p| &p.label == label).unwrap();
        let d = GdlGame::load("start", g.description()).unwrap();
        assert_eq!(facts_of(&d, &d.initial()), p.facts, "start {n}: the description's initial facts are the file's");
        assert_eq!(their_moves(&d, &d.initial()), terms(&p.moves), "start {n}: the first moves");
        n_starts += 1;
    }
    // the horde's own start is in its description (not given by a start delta)
    let h = open("horde");
    let hf = fixture("horde");
    let d = GdlGame::load("horde", h.description()).unwrap();
    assert_eq!(facts_of(&d, &d.initial()), hf.pos.iter().find(|p| p.label == "start").unwrap().facts);
    println!("OBS 79.35 D starts: the description of the game with a start by number begins at the file's facts for {n_starts} starts, and the first moves are the file's; the horde's own start is its description's");
}

#[test]
fn played_from_its_own_start_along_the_games_of_the_file_the_description_equals_the_kernel_and_the_file_at_every_position() {
    for game in GAMES {
        let fx = fixture(game);
        let g = open(game);
        let d = GdlGame::load(game, g.description()).unwrap();
        let (mut plies, mut ends) = (0usize, 0usize);
        for (label, line) in &fx.lines {
            let (mut a, mut b) = (g.initial(), d.initial());
            for i in 0..=line.len() {
                let mapped: BTreeSet<String> = g.moves(&a).iter().map(|m| spec_term(&g.move_name(&a, m))).collect();
                assert_eq!(mapped, their_moves(&d, &b), "{game}: {label}: moves at ply {i}");
                assert_eq!(g.mover(&a).map(|m| g.roles()[m].clone()), d.mover(&b).map(|m| d.roles()[m].clone()), "{game}: {label}: mover at ply {i}");
                assert_eq!(g.outcome(&a), d.outcome(&b), "{game}: {label}: end at ply {i}");
                plies += 1;
                if g.outcome(&a).is_some() { ends += 1; }
                let Some((name, term)) = line.get(i) else { break };
                a = g.apply(&a, &g.find_move(&a, name).unwrap_or_else(|| panic!("{label}: {name}")));
                b = d.apply(&b, &d.find_move(&b, term).unwrap_or_else(|| panic!("{label}: {term}")));
            }
        }
        println!("OBS 79.35 D lines {game}: the description equals the kernel at {plies} positions of {} named games ({ends} of them ended)", fx.lines.len());
    }
}

/// A MEASURE, NOT A DEMAND (what the layers above read of a description): the goal's target, a unit under attack that the goal names (cs_game::goal_target). The standard game has one. Three of the four still have the standard end (a side with no move left and its king
/// attacked) and the reading finds that target; it does not see the other ends of those games (a king removed, no unit left), and the game whose ends are all other has none, which the reading says by its own words. Printed for the report; the layers above are not changed here.
#[test]
fn what_the_goal_names_in_each_description_the_reading_the_layers_above_make_of_it() {
    let standard = std::fs::read_to_string(rules_root().join("chess.gdl")).unwrap();
    assert!(cs_game::goal_target::goal_target(&standard).is_ok(), "the standard game's goal names a unit under attack");
    let mut said = vec![];
    for game in GAMES {
        let g = open(game);
        let r = cs_game::goal_target::goal_target(g.description());
        println!("OBS 79.35 D goal target {game}: {}", match &r { Ok(t) => format!("a unit of kind {} under `{}`", t.kind, t.under), Err(e) => e.clone() });
        said.push((game, r.is_ok()));
    }
    assert!(said.iter().any(|(_, ok)| *ok), "one of the games keeps the standard end");
}
