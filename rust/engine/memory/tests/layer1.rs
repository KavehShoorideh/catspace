//! PLAN 79.20 PART B: THE TESTS OF LAYER 1, CALLING NOTHING OF THE RULES (SYSTEM_SPEC 11 "the layers", "a layer's test uses only that layer and what is below it"; piece: representation).
//! The four tests of plan 79.19 part A in substance, now on POSITIONS AS SETS OF FACTS read from fixture files (tests/fixtures, two games, made once by the maker rust/engine/decision/examples/make_layer1_fixtures.rs, which is not a test):
//! (1) a position as a set of facts is written and read back exactly, and the change between two positions is what the maker's plain set difference said;
//! (2) every statement of every standing and holder is written, read back and found identical (patterns taken from the fixtures' facts);
//! (3) a statement falls when, and only when, something it read changed (with a negative control that must be seen to fail);
//! (4) a pattern with placeholders binds on a position exactly where a slow plain reference says it does;
//! (5) a statement written under one game is not seen by a reader of the other.
//! No move is made, no legal move asked, no outcome read, no text form of the game used, no description loaded; tests/test_plan79_20.py scans this file for the names that would say otherwise.
mod common;
use common::{fixtures, Rng};
use cs_game::Fact;
use cs_memory::lemma::{Arg, Lit};
use cs_memory::matcher::{bindings_fused, Binding, FactIndex};
use cs_memory::statement::{decode_facts, encode_facts, unifies, Change, Holder, Says, Standing, Statement};
use cs_memory::store::{GameKey, LongTerm};
use std::collections::BTreeSet;

// ------------------------------------------------------------------------------------------------------------------------------ (1) a position as a set of facts is written and read back exactly

#[test]
fn a_position_as_a_set_of_facts_is_written_and_read_back_exactly_and_a_change_is_what_the_plain_difference_says() {
    for fx in fixtures() {
        let (mut n, mut shuffled_same, mut changes, mut empty_changes) = (0, 0, 0, 0);
        let mut rng = Rng::new(7919);
        for pos in fx.positions() {
            n += 1;
            let text = encode_facts(&pos.facts);
            assert_eq!(decode_facts(&text).unwrap(), pos.facts, "{}: written and read back", fx.file);
            // the text does not depend on the order the facts come in, nor on a fact said twice
            let mut mixed = pos.facts.clone();
            for i in (1..mixed.len()).rev() { mixed.swap(i, rng.below(i + 1)); }
            mixed.extend(pos.facts.iter().take(5).cloned());
            assert_eq!(encode_facts(&mixed), text, "{}: canonical", fx.file);
            shuffled_same += 1;
        }
        // the change between two positions, and what it does when applied
        for (before, after) in fx.pairs() {
            let ch = Change::between(&before.facts, &after.facts);
            let (added, removed) = after.since.as_ref().expect("a pair's second position has its change");
            assert_eq!((&ch.added, &ch.removed), (added, removed), "{}: the change is what the plain set difference of the maker said", fx.file);
            let mut set: BTreeSet<Fact> = before.facts.iter().cloned().collect();
            for f in &ch.removed { set.remove(f); }
            for f in &ch.added { set.insert(f.clone()); }
            assert_eq!(set.into_iter().collect::<Vec<_>>(), after.facts, "{}: the change applied to the first gives the second", fx.file);
            changes += 1;
            empty_changes += ch.is_empty() as u32;
        }
        println!("OBS 79.20 B1 {}: positions {n}, each written and read back exactly and the same from any order of its facts ({shuffled_same}); checkpoints equal to what the deltas built {}; changes between consecutive positions {changes} (empty ones {empty_changes}), each equal to the maker's difference and applied to the first giving the second", fx.file, fx.checks);
        assert!(n >= 300 && fx.checks > 10);
    }
}

// ------------------------------------------------------------------------------------------------------------------------------ patterns from a position's facts

/// A pattern generalised from facts: each constant becomes a variable with probability one half, the same variable wherever the same constant occurs; the second literal joins on a shared constant where one exists; a third is negated and uses only variables or constants already bound.
fn pattern_from(facts: &[Fact], rng: &mut Rng) -> Vec<Lit> {
    if facts.is_empty() { return vec![]; }
    let mut var_of: Vec<(String, Option<String>)> = vec![];
    let lit_of = |f: &Fact, neg: bool, only_bound: bool, var_of: &mut Vec<(String, Option<String>)>, rng: &mut Rng| -> Lit {
        let args = f.args.iter().map(|a| {
            if let Some((_, v)) = var_of.iter().find(|(c, _)| c == a) { return match v { Some(v) => Arg::Var(v.clone()), None => Arg::Const(a.clone()) }; }
            if only_bound { return Arg::Const(a.clone()); }
            let v = if rng.below(2) == 0 { Some(format!("v{}", var_of.len())) } else { None };
            var_of.push((a.clone(), v.clone()));
            match v { Some(v) => Arg::Var(v), None => Arg::Const(a.clone()) }
        }).collect();
        Lit { neg, relation: f.relation.clone(), args }
    };
    let first = &facts[rng.below(facts.len())];
    let mut out = vec![lit_of(first, false, false, &mut var_of, rng)];
    if rng.below(3) > 0 {
        let joined: Vec<&Fact> = facts.iter().filter(|f| *f != first && f.args.iter().any(|a| first.args.contains(a))).collect();
        let second = if joined.is_empty() { &facts[rng.below(facts.len())] } else { joined[rng.below(joined.len())] };
        out.push(lit_of(second, false, false, &mut var_of, rng));
    }
    if rng.below(3) == 0 {
        let third = &facts[rng.below(facts.len())];
        let mut l = lit_of(third, true, true, &mut var_of, rng);
        // a negated literal whose variables are all bound is safe; a constant-only one is a plain absence
        l.args = l.args.into_iter().map(|a| match a { Arg::Var(v) if !out.iter().any(|p| !p.neg && p.vars().contains(&v)) => Arg::Const(v), x => x }).collect();
        out.push(l);
    }
    out
}

fn statement_of(id: String, pattern: Vec<Lit>) -> Statement { Statement { id, pattern, says: Says::Holds, standing: Standing::Fact, holder: Holder::Ours, also: vec![] } }

// ------------------------------------------------------------------------------------------------------------------------------ (2) every statement round-trips

#[test]
fn every_statement_of_every_standing_and_holder_is_written_and_read_back_identical() {
    let mut rng = Rng::new(79_19);
    let standings = [Standing::Fact, Standing::Proved("theorem_name".into()), Standing::Conjecture("games 1 to 40".into()), Standing::Belief { probability: 0.3333333333333333, count: 12 }];
    let mut n = 0;
    let fxs = fixtures();
    for fx in &fxs {
        for (i, pos) in fx.positions().enumerate().step_by(7) {
            let pat = pattern_from(&pos.facts, &mut rng);
            if pat.iter().all(|l| l.neg) { continue; }
            let sayings = [Says::Holds, Says::Number(rng.below(1000) as f64 / 7.0), Says::Lower { term: format!("[tuple t{}]", rng.below(9)), within: rng.below(5) as u32 }, Says::Proposes(cs_memory::lemma::Response { removed: vec![pat[0].clone()], added: vec![Lit::pos("moved", &["?m"])] })];
            for st in &standings { for h in [Holder::Ours, Holder::OursAboutOther] { for sa in &sayings {
                let s = Statement { id: format!("{}-{n}", fx.key.name), pattern: pat.clone(), says: sa.clone(), standing: st.clone(), holder: h, also: vec![format!("position:{i}")] };
                if s.validate().is_err() { continue; }
                let text = s.to_json().to_string();
                assert_eq!(Statement::from_json(&serde_json::from_str(&text).unwrap()).unwrap(), s);
                n += 1;
            } } }
        }
    }
    println!("OBS 79.20 B2: {n} statements (every standing, both holders, every saying, patterns from the facts of {} games' fixtures) written, read back and identical", fxs.len());
    assert!(n > 1000);
}

// ------------------------------------------------------------------------------------------------------------------------------ (3) a statement falls when, and only when, something it read changed

/// The bindings of a pattern on a position's facts, as a set (no limit: a count that does not fit is a test fault).
fn bound(pat: &[Lit], facts: &[Fact]) -> BTreeSet<Binding> {
    let mut steps = 0;
    let (b, cut) = bindings_fused(pat, &Binding::new(), &FactIndex::new(facts), 1_000_000, &mut steps, u64::MAX);
    assert!(!cut);
    b.into_iter().collect()
}

#[test]
fn a_statement_falls_when_and_only_when_something_it_read_changed() {
    for fx in fixtures() {
        let mut rng = Rng::new(31_337);
        let (mut checked, mut fell, mut fell_same, mut stood, mut stood_changed, mut negative_control_broken, mut unread) = (0u64, 0u64, 0u64, 0u64, 0u64, 0u64, 0u64);
        for (before, after) in fx.pairs() {
            let ch = Change::between(&before.facts, &after.facts);
            let pool: Vec<Fact> = before.facts.iter().chain(after.facts.iter()).cloned().collect();
            for i in 0..20 {
                let pat = pattern_from(&pool, &mut rng);
                if pat.iter().all(|l| l.neg) { continue; }
                let s = statement_of(format!("p{i}"), pat.clone());
                let (b0, b1) = (bound(&pat, &before.facts), bound(&pat, &after.facts));
                let falls = s.falls_by(&ch);
                checked += 1;
                // ONLY WHEN: a fall names a changed fact that unifies with a literal of the pattern
                if falls { assert!(ch.added.iter().chain(ch.removed.iter()).any(|f| pat.iter().any(|l| unifies(l, f)))); fell += 1; if b0 == b1 { fell_same += 1; } }
                // WHEN: a statement that did not fall has the bindings it had
                if !falls { assert_eq!(b0, b1, "{}: a statement stood though its bindings changed: {pat:?}", fx.file); stood += 1; }
                if b0 != b1 { stood_changed += 1; assert!(falls); }
                // the negative control: a reader that ignores the negated literals must be seen to fail on some pair
                let positives: Vec<Lit> = pat.iter().filter(|l| !l.neg).cloned().collect();
                let blind = statement_of("blind".into(), positives);
                if !blind.falls_by(&ch) && b0 != b1 { negative_control_broken += 1; }
                // a statement about a relation no fact has never falls
                let ghost = statement_of("ghost".into(), vec![Lit::pos("no_such_relation", &["?x"])]);
                if !ghost.falls_by(&ch) { unread += 1; } else { panic!("a statement of a relation nothing carries fell"); }
            }
        }
        println!("OBS 79.20 B3 {}: {checked} statement-changes checked; fell {fell} (of which {fell_same} bound the same before and after: the reading is conservative), stood {stood} with the same bindings every time; bindings changed in {stood_changed}, every one a fall; negative control (a reader blind to negated literals) wrongly stood {negative_control_broken} times; the unread relation stood {unread} times", fx.file);
        assert!(checked > 5000 && fell > 100 && stood > 100);
        assert!(negative_control_broken > 0, "the control must fail somewhere, or the test could not tell");
    }
}

// ------------------------------------------------------------------------------------------------------------------------------ (4) a pattern binds exactly where a slow plain reference says

/// The plain reference: the literals in the order given, every row of every fact tried, no index, no reordering; then the negations as failures.
fn reference(pat: &[Lit], facts: &[Fact]) -> BTreeSet<Binding> {
    fn go(i: usize, pos: &[&Lit], neg: &[&Lit], b: Binding, facts: &[Fact], out: &mut BTreeSet<Binding>) {
        if i == pos.len() {
            let violated = neg.iter().any(|n| facts.iter().any(|f| f.relation == n.relation && f.args.len() == n.args.len() && n.args.iter().zip(&f.args).all(|(a, x)| match a { Arg::Const(c) => c == x, Arg::Var(v) => b.get(v).map_or(true, |bv| bv == x) })));
            if !violated { out.insert(b); }
            return;
        }
        for f in facts.iter().filter(|f| f.relation == pos[i].relation && f.args.len() == pos[i].args.len()) {
            let mut nb = b.clone();
            let mut ok = true;
            for (a, x) in pos[i].args.iter().zip(&f.args) {
                match a { Arg::Const(c) => if c != x { ok = false; break; }, Arg::Var(v) => match nb.get(v) { Some(bv) => if bv != x { ok = false; break; }, None => { nb.insert(v.clone(), x.clone()); } } }
            }
            if ok { go(i + 1, pos, neg, nb, facts, out); }
        }
    }
    let (pos, neg): (Vec<&Lit>, Vec<&Lit>) = (pat.iter().filter(|l| !l.neg).collect(), pat.iter().filter(|l| l.neg).collect());
    let mut out = BTreeSet::new();
    go(0, &pos, &neg, Binding::new(), facts, &mut out);
    out
}

#[test]
fn a_pattern_binds_exactly_where_a_slow_plain_reference_says() {
    for fx in fixtures() {
        let mut rng = Rng::new(4_242);
        let (mut checked, mut nonempty, mut joined, mut negated) = (0, 0, 0, 0);
        for pos in fx.positions().step_by(2) {
            for _ in 0..8 {
                let pat = pattern_from(&pos.facts, &mut rng);
                if pat.iter().all(|l| l.neg) { continue; }
                let (fast, slow) = (bound(&pat, &pos.facts), reference(&pat, &pos.facts));
                assert_eq!(fast, slow, "{}: {pat:?}", fx.file);
                checked += 1;
                nonempty += !slow.is_empty() as u32;
                joined += (pat.iter().filter(|l| !l.neg).count() > 1) as u32;
                negated += pat.iter().any(|l| l.neg) as u32;
            }
        }
        println!("OBS 79.20 B4 {}: {checked} patterns on positions agree with the plain reference (bound somewhere {nonempty}, with a join {joined}, with a negation {negated})", fx.file);
        assert!(checked > 1000 && nonempty > 100 && joined > 100 && negated > 50);
    }
}

// ------------------------------------------------------------------------------------------------------------------------------ (5) one game's statement is not seen by a reader of the other

#[test]
fn a_statement_written_under_one_game_is_not_seen_by_a_reader_of_the_other() {
    let fxs = fixtures();
    let keys: Vec<GameKey> = fxs.iter().take(2).map(|f| f.key.clone()).collect();
    assert_ne!(keys[0], keys[1]);
    assert_eq!(keys[0].name == keys[1].name, false);
    assert_ne!(keys[0].rules, keys[1].rules, "the content address of the rules tells the games apart");
    // a pattern of the first fixture's first fact, its arguments as placeholders
    let f = &fxs[0].games[0][0].facts[0];
    let vars: Vec<String> = (0..f.args.len()).map(|i| format!("?a{i}")).collect();
    let pat = vec![Lit::pos(&f.relation, &vars.iter().map(|v| v.as_str()).collect::<Vec<_>>())];
    let mut store = LongTerm::new();
    let s = statement_of("only-in-the-first".into(), pat.clone());
    store.sleep().add(&keys[0], s.clone()).unwrap();
    let first = store.reader(&keys[0]).unwrap();
    assert_eq!(first.statement("only-in-the-first"), Some(&s));
    assert!(store.reader(&keys[1]).is_none(), "a game the store was told nothing of has no partition");
    store.sleep().add(&keys[1], statement_of("only-in-the-second".into(), pat)).unwrap();
    let second = store.reader(&keys[1]).unwrap();
    assert!(second.statement("only-in-the-first").is_none() && first.statement("only-in-the-second").is_none());
    // the same name with other rules is another game
    let same_name_other_rules = GameKey { name: keys[0].name.clone(), rules: keys[1].rules.clone() };
    assert!(store.reader(&same_name_other_rules).is_none());
    println!("OBS 79.20 B5: a statement written under {} is read by its reader and not by the reader of {}; the same name with the other game's rules is a third, empty partition", keys[0].name, keys[1].name);
}
