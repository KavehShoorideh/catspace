//! PLAN 79.36 PART C, LAYER 1, THE SMALL VALUE TYPES: names, sides, sets of places, claims, lemmas, statements, the content address, the snapshot (SYSTEM_SPEC 11, last block; piece: representation; ledger row 43).
//! Every test states what one item answers on a small case written by hand, and at its edges. The expected values are written here, not computed by the code under test.
//! No object of a game is named (tests/test_plan78.py scans for it).
use cs_rep::claim::{Claim, Class, Rule};
use cs_rep::lemma::{Arg, Lemma, Lit, LitJson, Response};
use cs_rep::matcher::FactIndex;
use cs_rep::names::{Names, PlaceSet, Side, Unit};
use cs_rep::serde_json::{json, Value};
use cs_rep::sha;
use cs_rep::snapshot::{self, Snapshot};
use cs_rep::statement::{Change, Holder, Read, Says, Standing, Statement};
use cs_rep::Fact;

fn names() -> Names { Names::new(vec!["n0".into(), "n1".into(), "n2".into()], vec!["k0".into(), "k1".into()], vec!["r0".into(), "r1".into()]) }

// ------------------------------------------------------------------------------------------------------------------------------ names.rs

#[test]
fn the_other_side_of_each_side_is_the_other_and_twice_is_itself() {
    assert_eq!(Side::other(Side::Us), Side::Them);
    assert_eq!(Side::other(Side::Them), Side::Us);
    assert_eq!(Side::other(Side::other(Side::Us)), Side::Us);
}

#[test]
fn a_side_has_a_word_and_an_index_zero_for_us_and_one_for_them() {
    assert_eq!((Side::word(Side::Us), Side::word(Side::Them)), ("us", "them"));
    assert_eq!((Side::ix(Side::Us), Side::ix(Side::Them)), (0, 1));
}

#[test]
fn a_kind_is_found_by_its_text_and_an_unknown_kind_is_an_error_naming_it() {
    let n = names();
    assert_eq!(Names::kind(&n, "k0"), Ok(0));
    assert_eq!(Names::kind(&n, "k1"), Ok(1));
    assert_eq!(Names::kind(&n, "k2"), Err("no such kind: k2".to_string()));
    assert_eq!(Names::kind(&n, ""), Err("no such kind: ".to_string()));
}

#[test]
fn a_place_and_a_role_are_found_by_text_and_unknown_ones_are_errors() {
    let n = names();
    assert_eq!(Names::place(&n, "n2"), Ok(2));
    assert_eq!(Names::place(&n, "n3"), Err("no such place: n3".to_string()));
    assert_eq!(Names::role(&n, "r1"), Ok(1));
    assert_eq!(Names::role(&n, "r2"), Err("no such role: r2".to_string()));
}

#[test]
fn the_text_of_a_place_and_of_a_kind_is_its_name_by_number() {
    let n = names();
    assert_eq!(Names::place_text(&n, 1), "n1");
    assert_eq!(Names::kind_text(&n, 1), "k1");
    assert_eq!(Names::place_text(&n, 0), "n0");
}

#[test]
#[should_panic(expected = "index out of bounds")]
fn the_text_of_a_place_that_is_not_a_number_of_the_game_is_refused_by_a_panic() {
    let n = names();
    let _ = Names::place_text(&n, 3);
}

#[test]
fn a_unit_in_words_says_whose_it_is_its_kind_and_its_place() {
    let n = names();
    assert_eq!(Names::words(&n, &Unit { side: Side::Us, kind: 1, at: 2 }), "my k1 on n2");
    assert_eq!(Names::words(&n, &Unit { side: Side::Them, kind: 0, at: 0 }), "their k0 on n0");
}

#[test]
fn units_are_ordered_our_side_first_then_by_kind_then_by_place() {
    let a = Unit { side: Side::Us, kind: 1, at: 0 };
    let b = Unit { side: Side::Us, kind: 1, at: 2 };
    let c = Unit { side: Side::Them, kind: 0, at: 0 };
    assert!(a < b && b < c);
}

#[test]
fn an_empty_set_of_places_holds_nothing_and_has_no_length() {
    let s = PlaceSet::new(70);
    assert!(PlaceSet::is_empty(&s));
    assert_eq!(PlaceSet::len(&s), 0);
    assert!(!PlaceSet::contains(&s, 0));
    assert_eq!(PlaceSet::iter(&s).count(), 0);
    assert!(PlaceSet::is_empty(&PlaceSet::new(0)));
}

#[test]
fn a_set_of_places_holds_what_was_inserted_across_the_word_boundary() {
    let mut s = PlaceSet::new(130);
    for p in [0u16, 63, 64, 129] { PlaceSet::insert(&mut s, p); }
    assert_eq!(PlaceSet::len(&s), 4);
    assert!(PlaceSet::contains(&s, 63) && PlaceSet::contains(&s, 64) && PlaceSet::contains(&s, 129));
    assert!(!PlaceSet::contains(&s, 1) && !PlaceSet::contains(&s, 65));
    assert!(!PlaceSet::contains(&s, 500), "a place past the end is not in");
    PlaceSet::insert(&mut s, 63);
    assert_eq!(PlaceSet::len(&s), 4, "inserting twice is once");
    assert!(!PlaceSet::is_empty(&s));
}

#[test]
fn a_set_of_places_is_read_in_the_order_of_the_numbers_whatever_the_order_it_was_made_in() {
    let s = PlaceSet::from_places(100, [70u16, 3, 64, 3, 99]);
    assert_eq!(PlaceSet::iter(&s).collect::<Vec<_>>(), vec![3, 64, 70, 99]);
    assert_eq!(PlaceSet::len(&s), 4, "a place said twice is one");
}

#[test]
fn the_union_of_two_sets_of_places_has_the_places_of_either_and_leaves_the_other_alone() {
    let mut a = PlaceSet::from_places(70, [1u16, 65]);
    let b = PlaceSet::from_places(70, [2u16, 65]);
    PlaceSet::union_with(&mut a, &b);
    assert_eq!(PlaceSet::iter(&a).collect::<Vec<_>>(), vec![1, 2, 65]);
    assert_eq!(PlaceSet::iter(&b).collect::<Vec<_>>(), vec![2, 65]);
}

#[test]
fn two_sets_of_places_intersect_when_they_share_a_place_and_not_when_they_are_apart() {
    let a = PlaceSet::from_places(70, [1u16, 65]);
    assert!(PlaceSet::intersects(&a, &PlaceSet::from_places(70, [65u16])));
    assert!(!PlaceSet::intersects(&a, &PlaceSet::from_places(70, [2u16, 64])));
    assert!(!PlaceSet::intersects(&a, &PlaceSet::new(70)), "nothing shares with the empty set");
}

// ------------------------------------------------------------------------------------------------------------------------------ claim.rs

fn claim_json() -> Value { json!({"name": "c", "body": [{"relation": "r", "args": ["?x", "k0"], "neg": false}, {"relation": "q", "args": ["?x"], "neg": false}], "claim": {"class": "draw"}, "stats": {}, "meta": {}}) }

#[test]
fn a_class_has_a_word_and_is_read_back_from_it_and_any_other_word_is_none() {
    assert_eq!([Class::word(&Class::Win), Class::word(&Class::Draw), Class::word(&Class::Loss)], ["win", "draw", "loss"]);
    assert_eq!(Class::parse("win"), Some(Class::Win));
    assert_eq!(Class::parse("draw"), Some(Class::Draw));
    assert_eq!(Class::parse("loss"), Some(Class::Loss));
    assert_eq!(Class::parse("Win"), None);
    assert_eq!(Class::parse(""), None);
}

#[test]
fn a_claim_reads_the_relations_of_its_body_once_each_and_in_order() {
    let c = Claim::from_json(&claim_json()).unwrap();
    let want: std::collections::BTreeSet<String> = ["q".to_string(), "r".to_string()].into_iter().collect();
    assert_eq!(Claim::body_relations(&c), want);
}

#[test]
fn a_claim_with_a_body_that_binds_nothing_positive_does_not_validate_and_a_good_one_does() {
    let mut c = Claim::from_json(&claim_json()).unwrap();
    assert_eq!(Claim::validate(&c), Ok(()));
    c.body = vec![Lit::pos("r", &["?x"]).negated()];
    assert!(Claim::validate(&c).unwrap_err().contains("positive literal"));
    c.body.clear();
    assert!(Claim::validate(&c).is_err());
}

#[test]
fn a_claim_is_written_with_its_class_word_and_its_name() {
    let c = Claim::from_json(&claim_json()).unwrap();
    let j = Claim::to_json(&c);
    assert_eq!(j["name"], "c");
    assert_eq!(j["claim"]["class"], "draw");
    assert_eq!(j["body"].as_array().unwrap().len(), 2);
}

#[test]
fn a_rule_is_named_by_the_lemma_or_the_claim_it_holds() {
    let claim = Rule::from_json(&claim_json()).unwrap();
    assert_eq!(Rule::name(&claim), "c");
    let lemma_json = json!({"name": "l", "body": [{"relation": "r", "args": ["?x"], "neg": false}], "response": {"removed": [{"relation": "r", "args": ["?x"], "neg": false}], "added": []}, "stats": {}, "meta": {}});
    let lemma = Rule::from_json(&lemma_json).unwrap();
    assert_eq!(Rule::name(&lemma), "l");
    assert!(matches!(lemma, Rule::Response(_)) && matches!(claim, Rule::Claim(_)));
}

// ------------------------------------------------------------------------------------------------------------------------------ lemma.rs

#[test]
fn a_literal_is_written_to_json_and_read_back_by_the_trait_and_what_is_not_a_literal_is_refused() {
    let l = Lit::pos("r", &["?x", "c"]).negated();
    let j = <Lit as LitJson>::to_json(&l);
    assert_eq!(j, json!({"relation": "r", "args": ["?x", "c"], "neg": true}));
    assert_eq!(<Lit as LitJson>::from_json(&j), Ok(l));
    assert_eq!(<Lit as LitJson>::from_json(&json!({"relation": "r", "args": []})).unwrap().neg, false);
    assert_eq!(<Lit as LitJson>::from_json(&json!({"args": []})), Err("a literal has a relation".to_string()));
    assert_eq!(<Lit as LitJson>::from_json(&json!({"relation": "r"})), Err("a literal has args".to_string()));
    assert_eq!(<Lit as LitJson>::from_json(&json!({"relation": "r", "args": [3]})), Err("an argument is text".to_string()));
}

fn lemma() -> Lemma {
    Lemma { name: "l".into(), body: vec![Lit::pos("a", &["?x"]), Lit::pos("b", &["?x", "?y"]), Lit::pos("a", &["?y"])],
            response: Response { removed: vec![Lit::pos("a", &["?x"])], added: vec![] }, stats: json!({}), meta: json!({}), suggest: false }
}

#[test]
fn a_lemma_reads_the_relations_of_its_body_without_repeats() {
    let want: std::collections::BTreeSet<String> = ["a".to_string(), "b".to_string()].into_iter().collect();
    assert_eq!(Lemma::body_relations(&lemma()), want);
    let mut none = lemma();
    none.body.clear();
    assert!(Lemma::body_relations(&none).is_empty());
}

// ------------------------------------------------------------------------------------------------------------------------------ statement.rs

fn stmt(pattern: Vec<Lit>, also: &[&str]) -> Statement {
    Statement { id: "s".into(), pattern, says: Says::Holds, standing: Standing::Fact, holder: Holder::Ours, also: also.iter().map(|s| s.to_string()).collect() }
}

#[test]
fn a_standing_has_a_word_a_json_form_and_is_read_back_from_it() {
    let all = [(Standing::Fact, "fact"), (Standing::Proved("t".into()), "proved"), (Standing::Conjecture("g".into()), "conjecture"), (Standing::Belief { probability: 0.25, count: 4 }, "belief")];
    for (s, w) in all {
        assert_eq!(Standing::word(&s), w);
        let j = Standing::to_json(&s);
        assert_eq!(j["kind"], w);
        assert_eq!(Standing::from_json(&j), Ok(s));
    }
}

#[test]
fn a_standing_without_what_it_needs_or_of_no_known_kind_is_refused_with_the_reason() {
    assert_eq!(Standing::from_json(&json!({"kind": "proved"})), Err("a standing has theorem".to_string()));
    assert_eq!(Standing::from_json(&json!({"kind": "belief", "probability": 0.5})), Err("a belief has a count".to_string()));
    assert_eq!(Standing::from_json(&json!({"kind": "guess"})), Err("no standing guess".to_string()));
    assert_eq!(Standing::from_json(&json!({})), Err("a standing has a kind".to_string()));
}

#[test]
fn a_holder_has_a_word_and_a_word_names_its_holder_and_nothing_else_does() {
    assert_eq!(Holder::word(&Holder::Ours), "ours");
    assert_eq!(Holder::word(&Holder::OursAboutOther), "ours_about_other");
    assert_eq!(Holder::parse("ours"), Some(Holder::Ours));
    assert_eq!(Holder::parse("ours_about_other"), Some(Holder::OursAboutOther));
    assert_eq!(Holder::parse("theirs"), None);
}

#[test]
fn a_saying_is_read_back_from_its_json_and_one_that_lacks_a_part_is_refused() {
    for s in [Says::Holds, Says::Number(1.5), Says::Lower { term: "t".into(), within: 2 }, Says::Proposes(Response { removed: vec![Lit::pos("a", &["?x"])], added: vec![Lit::pos("b", &["?x"])] })] {
        assert_eq!(Says::from_json(&Says::to_json(&s)), Ok(s));
    }
    assert_eq!(Says::from_json(&json!({"kind": "number"})), Err("a number is a number".to_string()));
    assert_eq!(Says::from_json(&json!({"kind": "lower", "term": "t"})), Err("lower has within".to_string()));
    assert_eq!(Says::from_json(&json!({"kind": "shout"})), Err("no saying shout".to_string()));
    assert_eq!(Says::from_json(&json!({"kind": "proposes", "removed": []})), Err("proposes has added".to_string()));
}

#[test]
fn what_a_statement_read_is_each_literal_of_its_pattern_then_each_key_of_its_also() {
    let s = stmt(vec![Lit::pos("a", &["?x"]), Lit::pos("b", &["?x"])], &["k1"]);
    let r = Statement::reads(&s);
    assert_eq!(r.len(), 3);
    assert_eq!(r[0], Read::Facts(Lit::pos("a", &["?x"])));
    assert_eq!(r[1], Read::Facts(Lit::pos("b", &["?x"])));
    assert_eq!(r[2], Read::Key("k1".to_string()));
    assert!(Statement::reads(&stmt(vec![], &[])).is_empty());
}

#[test]
fn the_bindings_of_a_statement_are_the_rows_its_pattern_matches_and_an_empty_pattern_holds_with_the_empty_binding() {
    let idx = FactIndex::new(&[Fact::new("a", &["u"]), Fact::new("a", &["v"]), Fact::new("b", &["u"])]);
    let s = stmt(vec![Lit::pos("a", &["?x"]), Lit::pos("b", &["?x"])], &[]);
    let mut steps = 0u64;
    let got = Statement::bindings(&s, &idx, 10, &mut steps);
    assert_eq!(got.len(), 1);
    assert_eq!(got[0].get("x").map(|v| v.as_str()), Some("u"));
    assert!(steps > 0, "the join counts its steps");
    let all = Statement::bindings(&stmt(vec![Lit::pos("a", &["?x"])], &[]), &idx, 10, &mut steps);
    assert_eq!(all.len(), 2);
    let none = Statement::bindings(&stmt(vec![Lit::pos("c", &["?x"])], &[]), &idx, 10, &mut steps);
    assert!(none.is_empty());
    let empty = Statement::bindings(&stmt(vec![], &[]), &idx, 10, &mut steps);
    assert_eq!(empty.len(), 1);
    assert!(empty[0].is_empty());
}

#[test]
fn the_limit_of_bindings_cuts_the_rows_returned() {
    let idx = FactIndex::new(&[Fact::new("a", &["u"]), Fact::new("a", &["v"]), Fact::new("a", &["w"])]);
    let mut steps = 0u64;
    let got = Statement::bindings(&stmt(vec![Lit::pos("a", &["?x"])], &[]), &idx, 2, &mut steps);
    assert_eq!(got.len(), 2);
}

#[test]
fn a_change_between_two_lists_is_what_was_added_and_what_was_removed_and_a_change_of_nothing_is_empty() {
    let a = [Fact::new("p", &["1"]), Fact::new("p", &["2"])];
    let b = [Fact::new("p", &["2"]), Fact::new("p", &["3"])];
    let c = Change::between(&a, &b);
    assert_eq!(c.added, vec![Fact::new("p", &["3"])]);
    assert_eq!(c.removed, vec![Fact::new("p", &["1"])]);
    assert!(!Change::is_empty(&c));
    assert!(Change::is_empty(&Change::between(&a, &a)));
    assert!(!Change::is_empty(&Change { keys: vec!["k".into()], ..Default::default() }));
}

#[test]
fn a_lemma_made_a_statement_keeps_its_name_its_body_and_the_standing_given() {
    let s = Statement::from_lemma(&lemma(), Standing::Conjecture("sample".into()));
    assert_eq!(s.id, "l");
    assert_eq!(s.pattern, lemma().body);
    assert_eq!(s.standing, Standing::Conjecture("sample".into()));
    assert_eq!(Statement::validate(&s), Ok(()));
}

#[test]
fn a_statement_without_an_id_or_with_a_proposal_that_says_nothing_is_refused() {
    let mut s = stmt(vec![Lit::pos("a", &["?x"])], &[]);
    s.id.clear();
    assert_eq!(Statement::validate(&s), Err("a statement needs an id".to_string()));
    let mut p = stmt(vec![Lit::pos("a", &["?x"])], &[]);
    p.says = Says::Proposes(Response::default());
    assert_eq!(Statement::validate(&p), Err("s: a proposal says nothing".to_string()));
    p.says = Says::Proposes(Response { removed: vec![Lit::pos("a", &["?x"]).negated()], added: vec![] });
    assert_eq!(Statement::validate(&p), Err("s: a proposal is positive literals".to_string()));
}

#[test]
fn a_statement_is_written_with_its_parts_and_read_back_and_a_missing_part_is_refused_with_its_name() {
    let s = stmt(vec![Lit::pos("a", &["?x", "c"])], &["k"]);
    let j = Statement::to_json(&s);
    assert_eq!((j["id"].as_str(), j["holder"].as_str(), j["also"][0].as_str()), (Some("s"), Some("ours"), Some("k")));
    assert_eq!(Statement::from_json(&j), Ok(s));
    let mut broken = j.clone();
    broken.as_object_mut().unwrap().remove("holder");
    assert_eq!(Statement::from_json(&broken), Err("s: a holder is ours or ours_about_other".to_string()));
    let mut no_id = j;
    no_id.as_object_mut().unwrap().remove("id");
    assert_eq!(Statement::from_json(&no_id), Err("a statement has an id".to_string()));
}

#[test]
fn an_argument_text_with_a_question_mark_is_a_placeholder_and_without_it_a_constant() {
    assert_eq!(Arg::parse("?x"), Arg::Var("x".into()));
    assert_eq!(Arg::parse("x"), Arg::Const("x".into()));
}

// ------------------------------------------------------------------------------------------------------------------------------ sha.rs and snapshot.rs

#[test]
fn the_digest_of_nothing_and_of_a_short_text_are_the_standard_values_as_bytes() {
    let d = sha::sha256(b"abc");
    assert_eq!(d.len(), 32);
    assert_eq!(&d[..4], &[0xba, 0x78, 0x16, 0xbf]);
    assert_eq!(&d[28..], &[0xf2, 0x00, 0x15, 0xad]);
    assert_eq!(&sha::sha256(b"")[..4], &[0xe3, 0xb0, 0xc4, 0x42]);
}

#[test]
fn the_digest_changes_with_one_byte_and_does_not_with_the_same_bytes() {
    assert_ne!(sha::sha256(b"abc"), sha::sha256(b"abd"));
    assert_eq!(sha::sha256(b"abc"), sha::sha256(b"abc"));
    assert_ne!(sha::sha256(&[0u8; 55]), sha::sha256(&[0u8; 56]), "the padding edge: 55 and 56 bytes");
    assert_ne!(sha::sha256(&[0u8; 63]), sha::sha256(&[0u8; 64]));
}

#[test]
fn the_vocabulary_of_a_snapshot_is_read_with_lists_that_may_be_left_out_and_units_that_may_have_no_kind() {
    let v = snapshot::vocabulary_from_json(&json!({"derived": ["a"], "delta": ["b", "c"], "units": {"relation": "u", "owner": 0, "kind": 1}})).unwrap();
    assert_eq!((v.derived, v.delta), (vec!["a".to_string()], vec!["b".to_string(), "c".to_string()]));
    let u = v.units.unwrap();
    assert_eq!((u.relation.as_str(), u.owner, u.kind), ("u", 0, Some(1)));
    let bare = snapshot::vocabulary_from_json(&json!({})).unwrap();
    assert!(bare.derived.is_empty() && bare.delta.is_empty() && bare.units.is_none());
    let no_kind = snapshot::vocabulary_from_json(&json!({"units": {"relation": "u", "owner": 2}})).unwrap();
    assert_eq!(no_kind.units.unwrap().kind, None);
}

#[test]
fn a_vocabulary_with_a_list_that_is_not_text_or_a_unit_without_its_relation_is_refused() {
    assert_eq!(snapshot::vocabulary_from_json(&json!({"derived": "a"})).unwrap_err(), "vocabulary.derived is a list");
    assert_eq!(snapshot::vocabulary_from_json(&json!({"delta": [1]})).unwrap_err(), "vocabulary.delta holds text");
    assert_eq!(snapshot::vocabulary_from_json(&json!({"units": {"owner": 0}})).unwrap_err(), "units.relation");
}

#[test]
fn the_format_of_a_snapshot_is_one_and_a_snapshot_of_another_format_is_refused() {
    assert_eq!(snapshot::FORMAT, 1);
    let text = r#"{"format":2,"game":"g","parent":null,"vocabulary":{},"lemmas":[]}"#;
    assert!(Snapshot::parse(text.as_bytes()).unwrap_err().contains("snapshot format is 1"));
    let ok = r#"{"format":1,"game":"g","parent":"abc","vocabulary":{},"lemmas":[]}"#;
    let s = Snapshot::parse(ok.as_bytes()).unwrap();
    assert_eq!((s.game.as_str(), s.parent.as_deref(), s.lemmas.len(), s.claims.len(), s.teacher_used), ("g", Some("abc"), 0, 0, false));
}
