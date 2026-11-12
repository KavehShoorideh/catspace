//! PLAN 78.36: the claim rule and the search toward claims inside the decision. On tic-tac-toe in GDL (a game the layers know nothing about): the claim reader, the claim step
//! that proves and plays, the refuted move the ledger reads last, the switch; and on the chess description: a snapshot of response rules only decides exactly as it did.
use cs_decision::{decide_with_lemmas, decide_with_store, Params};
use cs_game::gdl_game::{FactView, GdlGame};
use cs_game::kernel::KernelGame;
use cs_game::{Game, Vocabulary};
use cs_memory::claim::{Claims, Class};
use cs_memory::op::LemmaOp;
use cs_memory::snapshot::Snapshot;
use cs_operator::Operator;
use cs_tactics::ledger::Count;
use std::path::PathBuf;

const TTT: &str = "
(role x) (role o)
(init (cell 1 1 b)) (init (cell 1 2 b)) (init (cell 1 3 b))
(init (cell 2 1 b)) (init (cell 2 2 b)) (init (cell 2 3 b))
(init (cell 3 1 b)) (init (cell 3 2 b)) (init (cell 3 3 b))
(init (control x))
(<= (legal ?p (mark ?m ?n)) (true (cell ?m ?n b)) (true (control ?p)))
(<= (legal x noop) (true (control o)))
(<= (legal o noop) (true (control x)))
(<= (next (cell ?m ?n x)) (does x (mark ?m ?n)))
(<= (next (cell ?m ?n o)) (does o (mark ?m ?n)))
(<= (next (cell ?m ?n ?w)) (true (cell ?m ?n ?w)) (distinct ?w b))
(<= (next (cell ?m ?n b)) (true (cell ?m ?n b)) (does ?w (mark ?j ?k)) (distinct ?m ?j))
(<= (next (cell ?m ?n b)) (true (cell ?m ?n b)) (does ?w (mark ?j ?k)) (distinct ?n ?k))
(<= (next (control x)) (true (control o)))
(<= (next (control o)) (true (control x)))
(<= (row ?m ?p) (true (cell ?m 1 ?p)) (true (cell ?m 2 ?p)) (true (cell ?m 3 ?p)))
(<= (column ?n ?p) (true (cell 1 ?n ?p)) (true (cell 2 ?n ?p)) (true (cell 3 ?n ?p)))
(<= (diagonal ?p) (true (cell 1 1 ?p)) (true (cell 2 2 ?p)) (true (cell 3 3 ?p)))
(<= (diagonal ?p) (true (cell 1 3 ?p)) (true (cell 2 2 ?p)) (true (cell 3 1 ?p)))
(<= (line ?p) (row ?m ?p))
(<= (line ?p) (column ?n ?p))
(<= (line ?p) (diagonal ?p))
(<= open (true (cell ?m ?n b)))
(<= terminal (line x))
(<= terminal (line o))
(<= terminal (not open))
(<= (goal x 100) (line x))
(<= (goal x 50) (not (line x)) (not (line o)))
(<= (goal x 0) (line o))
(<= (goal o 100) (line o))
(<= (goal o 50) (not (line x)) (not (line o)))
(<= (goal o 0) (line x))
";

fn snapshot(rules: &str) -> Snapshot {
    let text = format!(r#"{{"format":1,"game":"ttt","parent":null,"vocabulary":{{"derived":[],"delta":["cell"],"units":null}},"judge":null,"teacher_used":false,"lemmas":[{rules}]}}"#);
    Snapshot::parse(text.as_bytes()).unwrap()
}
fn view() -> FactView { FactView::new(TTT, Vocabulary { derived: vec![], delta: vec!["cell".into()], units: None }).unwrap() }
fn lit(cell: &str, who: &str) -> String { let p: Vec<&str> = cell.split(',').collect(); format!(r#"{{"relation":"cell","args":["{}","{}","{who}"],"neg":false}}"#, p[0], p[1]) }
fn claim(name: &str, body: &[(&str, &str)], class: &str) -> String {
    format!(r#"{{"name":"{name}","body":[{}],"claim":{{"class":"{class}"}},"stats":{{}},"meta":{{}}}}"#, body.iter().map(|(c, w)| lit(c, w)).collect::<Vec<_>>().join(","))
}
/// "mark the corner (1 1) when it is free": a response rule.
const MARK_CORNER: &str = r#"{"name":"mark-corner","body":[{"relation":"cell","args":["1","1","b"],"neg":false}],
  "response":{"removed":[{"relation":"cell","args":["1","1","b"],"neg":false}],"added":[{"relation":"cell","args":["1","1","us"],"neg":false}]},"stats":{},"meta":{}}"#;

fn after(g: &GdlGame, names: &[&str]) -> <GdlGame as Game>::State {
    let mut s = g.initial();
    for n in names { let m = g.find_move(&s, n).unwrap_or_else(|| panic!("no move {n}")); s = g.apply(&s, &m); }
    s
}
fn steps<M>(d: &cs_operator::Decision<M>) -> Vec<String> { d.steps.iter().map(|s| s.operator.clone()).collect() }
fn name<G: Game>(g: &G, s: &G::State, d: &cs_operator::Decision<G::Move>) -> String { g.move_name(s, d.mv.as_ref().unwrap()) }

#[test]
fn the_reader_binds_in_the_movers_perspective_and_counts() {
    let g = GdlGame::load("ttt", TTT).unwrap();
    let snap = snapshot(&claim("them-centre", &[("2,2", "them")], "loss"));
    let v = view();
    let c = Claims::new(&snap, &v);
    assert_eq!(c.rules().count(), 1);
    // x took the centre: for o to move, them (x) holds it
    let s = after(&g, &["(mark 2 2)"]);
    assert_eq!(c.class_at(&g, &s), Some((Class::Loss, "them-centre".to_string())));
    // the start: nothing binds; and x to move holding the centre itself: them (o) does not -> nothing
    assert_eq!(c.class_at(&g, &g.initial()), None);
    assert_eq!(c.class_at(&g, &after(&g, &["(mark 2 2)", "(mark 1 1)"])), None);
    // from the other seat: o took the centre, x to move -> them holds it -> the claim speaks for x
    assert_eq!(c.class_at(&g, &after(&g, &["(mark 1 1)", "(mark 2 2)"])), Some((Class::Loss, "them-centre".to_string())));
    let st = c.stats();
    assert_eq!((st.evaluations, st.spoke, st.conflicts), (4, 2, 0));
    assert!(st.seconds > 0.0);
    // a terminal state gives no answer
    let end = after(&g, &["(mark 1 1)", "(mark 2 1)", "(mark 1 2)", "(mark 2 2)", "(mark 1 3)"]);
    assert_eq!(c.class_at(&g, &end), None);
}

#[test]
fn claims_that_disagree_make_no_answer_and_count_a_conflict() {
    let g = GdlGame::load("ttt", TTT).unwrap();
    let snap = snapshot(&format!("{},{},{}", claim("a-loss", &[("2,2", "them")], "loss"), claim("a-win", &[("2,2", "them"), ("1,1", "b")], "win"), claim("a-loss-too", &[("2,2", "them"), ("1,2", "b")], "loss")));
    let v = view();
    let c = Claims::new(&snap, &v);
    let s = after(&g, &["(mark 2 2)"]);
    assert_eq!(c.class_at(&g, &s), None, "a loss and a win at once: the rules disagree");
    assert_eq!(c.stats().conflicts, 1);
    // agreeing rules speak, under the first one's name
    let snap2 = snapshot(&format!("{},{}", claim("a-loss", &[("2,2", "them")], "loss"), claim("a-loss-too", &[("2,2", "them"), ("1,2", "b")], "loss")));
    let c2 = Claims::new(&snap2, &v);
    assert_eq!(c2.class_at(&g, &s), Some((Class::Loss, "a-loss".to_string())));
    // the ablation: a rule left out does not speak
    let mut c3 = Claims::new(&snap2, &v);
    c3.exclude.insert("a-loss".into());
    c3.exclude.insert("a-loss-too".into());
    assert!(c3.is_empty());
}

/// The ledger step's goal, as the trace names it.
fn ledger_goal<M>(d: &cs_operator::Decision<M>) -> String { d.steps.iter().find(|st| st.operator == "ledger").unwrap().goal.describe() }

/// A claimed win is a CLASS: the claim step does not play it; the ledger reads it first (every move that is not a win last) and chooses.
#[test]
fn a_claimed_win_is_not_played_at_once_but_read_first_and_the_ledger_chooses() {
    let g = GdlGame::load("ttt", TTT).unwrap();
    // "if they hold the centre, I lose": x taking the centre leaves o (to move) lost -- a claimed win; any other move lets o take the centre, leaving x lost
    let snap = snapshot(&claim("them-centre", &[("2,2", "them")], "loss"));
    let v = view();
    let c = Claims::new(&snap, &v);
    let s = g.initial();
    let d = decide_with_store(&g, &s, 1.0, &Count, &Params::default(), None, Some(&c)).unwrap();
    assert_eq!(steps(&d), vec!["forcing", "claim", "ledger"], "the claim step does not decide: the ledger does");
    assert!(!d.proved, "a claimed win is not a proof");
    let cs = d.steps.iter().find(|st| st.operator == "claim").unwrap();
    assert_eq!(cs.outcome.status, cs_operator::Status::Read);
    assert!(cs.outcome.mv.is_none());
    assert_eq!(name(&g, &s, &d), "(mark 2 2)", "equal by the ledger's judge: the first read, the claimed win");
    assert_eq!(ledger_goal(&d), "ledger(0,prefer 1,avoid 8)", "the W move first, the other eight last");
    let line = d.trace.iter().find(|l| l.starts_with("claim toward: case any W")).expect("the claim step says which case applied");
    assert!(line.contains("W 1 (game 0), D 0, U 8, L 0") && line.contains("them-centre"), "{line}");
    // the switch off: the step does not appear
    let off = Params { claims: false, ..Params::default() };
    let d0 = decide_with_store(&g, &s, 1.0, &Count, &off, None, Some(&c)).unwrap();
    assert!(!steps(&d0).contains(&"claim".to_string()));
    assert!(!d0.trace.iter().any(|l| l.contains("claim")));
}

/// The game's own win is proved and played at once; a claimed one is only a class.
#[test]
fn the_games_own_win_is_proved_by_the_operator_and_a_claimed_win_is_not() {
    use cs_decision::claim_op::ClaimOp;
    let g = GdlGame::load("ttt", TTT).unwrap();
    let v = view();
    let snap = snapshot(&claim("them-centre", &[("2,2", "them")], "loss"));
    let c = Claims::new(&snap, &v);
    let op = ClaimOp { claims: &c, history: vec![] };
    // x has two in a row with the third open: completing it is the game's own win
    let s = after(&g, &["(mark 1 1)", "(mark 2 1)", "(mark 1 2)", "(mark 2 2)"]);
    let o = op.run(&g, &s, &cs_operator::Goal::Toward { role: 0 }, cs_operator::Budget::new(5.0, 10_000_000));
    assert_eq!(o.status, cs_operator::Status::Proved);
    assert_eq!(g.move_name(&s, o.mv.as_ref().unwrap()), "(mark 1 3)");
    // from the start the only win is a claimed one: not proved, no move
    let o2 = op.run(&g, &g.initial(), &cs_operator::Goal::Toward { role: 0 }, cs_operator::Budget::new(5.0, 10_000_000));
    assert_eq!(o2.status, cs_operator::Status::Read);
    assert!(o2.mv.is_none());
    assert_eq!(o2.candidates.len(), 1, "the W moves are the candidates");
}

/// Among the W moves, one that leads to a position the record already holds comes after one that does not.
#[test]
fn among_the_won_moves_a_new_position_comes_before_one_the_record_holds() {
    let g = GdlGame::load("ttt", TTT).unwrap();
    let v = view();
    let s = g.initial();
    let snap = snapshot(&format!("{},{}", claim("them-centre", &[("2,2", "them")], "loss"), claim("them-corner", &[("1,1", "them")], "loss")));
    let claims = Claims::new(&snap, &v);
    let d = decide_with_store(&g, &s, 1.0, &Count, &Params::default(), None, Some(&claims)).unwrap();
    assert_eq!(name(&g, &s, &d), "(mark 1 1)", "two won moves of equal value: the first");
    let seen = g.key(&g.apply(&s, &g.find_move(&s, "(mark 1 1)").unwrap()));
    let d2 = decide_with_store(&g, &s, 1.0, &Count, &Params { history: vec![seen], ..Params::default() }, None, Some(&claims)).unwrap();
    assert_eq!(name(&g, &s, &d2), "(mark 2 2)", "the corner leads to a position already in the record: the centre is read first");
}

/// Nothing speaks within two plies: the step stops and says so, and the ledger's order is the one it always had.
#[test]
fn a_silent_claim_step_stops_at_two_plies_and_changes_nothing() {
    let g = GdlGame::load("ttt", TTT).unwrap();
    let v = view();
    let s = g.initial();
    let snap = snapshot(&format!("{MARK_CORNER},{}", claim("nowhere", &[("9,9", "us")], "win")));
    let op = LemmaOp::new(&snap, &v);
    let claims = Claims::new(&snap, &v);
    let d = decide_with_store(&g, &s, 1.0, &Count, &Params::default(), Some(&op as &dyn Operator<GdlGame>), Some(&claims)).unwrap();
    let cs = d.steps.iter().find(|st| st.operator == "claim").unwrap();
    assert_eq!(cs.outcome.depth, Some(cs_tactics::toward::SILENT_DEPTH as i64));
    assert!(d.trace.iter().any(|l| l.contains("no claim speaks within 2 plies: stopped")), "{:?}", d.trace);
    assert_eq!(ledger_goal(&d), "ledger(0,prefer 1)", "as without the claim step");
    assert_eq!(name(&g, &s, &d), "(mark 1 1)");
}

/// The rule that names the centre, as a SUGGESTION.
const SUGGEST_CENTRE: &str = r#"{"name":"suggest-centre","suggest":true,"body":[{"relation":"cell","args":["2","2","b"],"neg":false}],
  "response":{"removed":[{"relation":"cell","args":["2","2","b"],"neg":false}],"added":[{"relation":"cell","args":["2","2","us"],"neg":false}]},"stats":{},"meta":{}}"#;

#[test]
fn a_suggestion_is_reported_apart_and_never_a_candidate() {
    let g = GdlGame::load("ttt", TTT).unwrap();
    let v = view();
    let snap = snapshot(SUGGEST_CENTRE);
    assert!(snap.lemmas[0].suggest);
    let op = LemmaOp::new(&snap, &v);
    let s = g.initial();
    let o = op.run(&g, &s, &cs_operator::Goal::Respond { role: 0 }, cs_operator::Budget::new(5.0, 1_000_000));
    assert!(o.candidates.is_empty());
    let centre = g.moves(&s).iter().position(|m| g.move_name(&s, m) == "(mark 2 2)").unwrap();
    assert!(o.facts.iter().any(|f| f.relation == "suggested" && f.args == vec!["suggest-centre".to_string(), centre.to_string()]));
    assert!(o.facts.iter().any(|f| f.relation == "fired" && f.args[0] == "suggest-centre"));
    // the playout prior leaves it out too
    use cs_planning::rollout::Prior;
    let w = cs_decision::LemmaPrior::new(&snap, &v, 5.0).weights(&g, &s, &g.moves(&s));
    assert!(w.iter().all(|x| *x == 1.0));
}

#[test]
fn a_suggestion_is_used_only_where_the_claim_step_classes_its_move_a_win() {
    let g = GdlGame::load("ttt", TTT).unwrap();
    let v = view();
    let s = g.initial();
    let params = Params::default();
    let both_won = format!("{},{}", claim("them-centre", &[("2,2", "them")], "loss"), claim("them-corner", &[("1,1", "them")], "loss"));
    // (a) no claim step (no claims in the snapshot): ignored; the ledger's first move is played
    let plain = snapshot(SUGGEST_CENTRE);
    let (op, claims) = (LemmaOp::new(&plain, &v), Claims::new(&plain, &v));
    let d = decide_with_store(&g, &s, 1.0, &Count, &params, Some(&op as &dyn Operator<GdlGame>), Some(&claims)).unwrap();
    assert!(!steps(&d).contains(&"claim".to_string()));
    assert_eq!(ledger_goal(&d), "ledger(0)");
    assert_eq!(name(&g, &s, &d), "(mark 1 1)");
    // (b) its move is W (with another W move before it in the order): it is read first and played
    let won = snapshot(&format!("{SUGGEST_CENTRE},{both_won}"));
    let (op, claims) = (LemmaOp::new(&won, &v), Claims::new(&won, &v));
    let d = decide_with_store(&g, &s, 1.0, &Count, &params, Some(&op as &dyn Operator<GdlGame>), Some(&claims)).unwrap();
    assert_eq!(ledger_goal(&d), "ledger(0,prefer 2,avoid 7)");
    assert_eq!(name(&g, &s, &d), "(mark 2 2)", "the suggested W move before the other W move");
    // (c) its move is L (only the corner is a won move): ignored
    let lost = snapshot(&format!("{SUGGEST_CENTRE},{}", claim("them-corner", &[("1,1", "them")], "loss")));
    let (op, claims) = (LemmaOp::new(&lost, &v), Claims::new(&lost, &v));
    let d = decide_with_store(&g, &s, 1.0, &Count, &params, Some(&op as &dyn Operator<GdlGame>), Some(&claims)).unwrap();
    assert_eq!(ledger_goal(&d), "ledger(0,prefer 1,avoid 8)", "only the won corner is read first");
    assert_eq!(name(&g, &s, &d), "(mark 1 1)");
    // (e) the switch off: ignored
    let off = Params { claims: false, ..Params::default() };
    let (op, claims) = (LemmaOp::new(&won, &v), Claims::new(&won, &v));
    let d = decide_with_store(&g, &s, 1.0, &Count, &off, Some(&op as &dyn Operator<GdlGame>), Some(&claims)).unwrap();
    assert_eq!(ledger_goal(&d), "ledger(0)");
}

#[test]
fn a_refuted_response_candidate_is_not_played_when_an_equal_move_is_not_refuted() {
    let g = GdlGame::load("ttt", TTT).unwrap();
    let v = view();
    let s = g.initial();
    let lemma_only = snapshot(MARK_CORNER);
    let op = LemmaOp::new(&lemma_only, &v);
    let params = Params::default();
    // without a claim the rule's move is read first and, every move being equal by the judge, played
    let d0 = decide_with_lemmas(&g, &s, 1.0, &Count, &params, Some(&op as &dyn Operator<GdlGame>)).unwrap();
    assert_eq!(name(&g, &s, &d0), "(mark 1 1)");
    // a claim that says: the side to move who holds (1 1) loses. After x marks (1 1) and o answers, x is to move holding it: every reply loses -> (mark 1 1) is refuted
    let both = snapshot(&format!("{MARK_CORNER},{}", claim("holds-corner-loses", &[("1,1", "us")], "loss")));
    let op2 = LemmaOp::new(&both, &v);
    let claims = Claims::new(&both, &v);
    assert_eq!(both.lemmas.len(), 1);
    assert_eq!(both.claims.len(), 1);
    let d = decide_with_store(&g, &s, 1.0, &Count, &params, Some(&op2 as &dyn Operator<GdlGame>), Some(&claims)).unwrap();
    assert_eq!(steps(&d)[..3], ["lemma", "forcing", "claim"]);
    let claim_step = d.steps.iter().find(|st| st.operator == "claim").unwrap();
    assert_eq!(cs_decision::claim_op::refuted_of(&claim_step.outcome), vec![0], "(mark 1 1) is the one move refuted");
    assert!(claim_step.outcome.candidates.is_empty(), "no root move reached a claimed win");
    let ledger = d.steps.iter().find(|st| st.operator == "ledger").unwrap();
    assert_eq!(ledger.goal.describe(), "ledger(0,avoid 1)", "no W, some U: the rule's candidate is refuted, so dropped from the order, and the refuted move is read last");
    assert_ne!(name(&g, &s, &d), "(mark 1 1)", "an equal move that no claim refutes is played instead");
    assert!(d.trace.iter().any(|l| l.contains("refuted: (mark 1 1)")), "{:?}", d.trace);
    assert!(d.trace.iter().any(|l| l.starts_with("ledger search: 1 root move(s) the claim search puts last")));
    // the value is the ledger's own: the same value as without the claims (an order, never a verdict)
    let v0 = d0.steps.iter().find(|st| st.operator == "ledger").unwrap().outcome.value;
    assert_eq!(ledger.outcome.value, v0);
}

#[test]
fn when_every_root_move_is_refuted_nothing_changes() {
    let g = GdlGame::load("ttt", TTT).unwrap();
    let v = view();
    let s = g.initial();
    // "whoever is to move holding any cell, loses": every x move is met by any reply -> x then holds a cell and is to move: all nine refuted
    let any = snapshot(&format!("{MARK_CORNER},{}", [("1,1"), ("1,2"), ("1,3"), ("2,1"), ("2,2"), ("2,3"), ("3,1"), ("3,2"), ("3,3")].iter().map(|c| claim(&format!("holds-{c}"), &[(c, "us")], "loss")).collect::<Vec<_>>().join(",")));
    let op = LemmaOp::new(&any, &v);
    let claims = Claims::new(&any, &v);
    let d = decide_with_store(&g, &s, 1.0, &Count, &Params::default(), Some(&op as &dyn Operator<GdlGame>), Some(&claims)).unwrap();
    let ledger = d.steps.iter().find(|st| st.operator == "ledger").unwrap();
    assert_eq!(ledger.goal.describe(), "ledger(0,prefer 1)", "all refuted: the order is what it was");
    assert_eq!(name(&g, &s, &d), "(mark 1 1)");
}

#[test]
fn without_claims_the_decision_and_its_trace_are_what_they_were() {
    let v = view();
    let g = GdlGame::load("ttt", TTT).unwrap();
    let with_lemma = snapshot(MARK_CORNER);
    let none = snapshot("");
    for names in [vec![], vec!["(mark 2 2)"], vec!["(mark 1 1)", "(mark 2 2)"], vec!["(mark 1 1)", "(mark 2 1)", "(mark 1 2)"]] {
        let s = after(&g, &names);
        for snap in [&with_lemma, &none] {
            let op = LemmaOp::new(snap, &v);
            let lem: Option<&dyn Operator<GdlGame>> = if snap.lemmas.is_empty() { None } else { Some(&op) };
            let claims = Claims::new(snap, &v);
            let old = decide_with_lemmas(&g, &s, 0.4, &Count, &Params::default(), lem).unwrap();
            let new = decide_with_store(&g, &s, 0.4, &Count, &Params::default(), lem, Some(&claims)).unwrap();
            assert_eq!(steps(&old), steps(&new));
            assert_eq!(name(&g, &s, &old), name(&g, &s, &new));
            assert_eq!(old.trace.len(), new.trace.len());
            let strip = |t: &[String]| t.iter().map(|l| l.split(" nodes").next().unwrap().to_string()).collect::<Vec<_>>();       // (the node counts and clocks are measurements)
            assert!(!new.trace.iter().any(|l| l.contains("claim")), "{:?}", new.trace);
            assert_eq!(strip(&old.trace).len(), strip(&new.trace).len());
        }
    }
}

fn chess_spec() -> String { std::fs::read_to_string(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../rules/chess.gdl")).unwrap() }

#[test]
fn a_chess_snapshot_of_response_rules_decides_as_it_did() {
    const SNAP: &str = r#"{"format":1,"game":"chess","parent":null,
     "vocabulary":{"derived":["attack"],"delta":["cell"],"units":null},"judge":null,"teacher_used":false,"lemmas":[
     {"name":"take-the-attacker",
      "body":[{"relation":"cell","args":["?x","?y","us","?k"],"neg":false},{"relation":"attack","args":["them","?a","?b","?x","?y"],"neg":false},{"relation":"cell","args":["?a","?b","them","?j"],"neg":false}],
      "response":{"removed":[{"relation":"cell","args":["?a","?b","them","?j"],"neg":false}],"added":[{"relation":"cell","args":["?a","?b","us","?m"],"neg":false}]},
      "stats":{},"meta":{}}]}"#;
    let snap = Snapshot::parse(SNAP.as_bytes()).unwrap();
    let view = FactView::new(&chess_spec(), Vocabulary { derived: vec!["attack".into()], delta: vec!["cell".into()], units: None }).unwrap();
    let op = LemmaOp::new(&snap, &view);
    let claims = Claims::new(&snap, &view);
    for fen in ["4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1", "4k3/8/8/3p4/4P3/8/8/4K3 b - - 0 1", "8/8/4k3/8/8/4K3/4P3/8 w - - 0 1"] {
        let g = KernelGame::from_fen("chess", fen).unwrap();
        let s = g.initial();
        let lem: Option<&dyn Operator<KernelGame>> = Some(&op);
        let old = decide_with_lemmas(&g, &s, 0.5, &Count, &Params::default(), lem).unwrap();
        let new = decide_with_store(&g, &s, 0.5, &Count, &Params::default(), lem, Some(&claims)).unwrap();
        assert_eq!(steps(&old), steps(&new), "{fen}");
        assert_eq!(name(&g, &s, &old), name(&g, &s, &new), "{fen}");
        assert_eq!(old.trace.len(), new.trace.len(), "{fen}");
        assert!(!new.trace.iter().any(|l| l.contains("claim")));
    }
}

#[test]
fn a_chess_claim_gives_the_claim_step_and_the_switch_removes_it() {
    // "the side to move whose opponent has no pawn and who has one wins" is a made-up claim: the test is about the mechanism, not about chess
    const SNAP: &str = r#"{"format":1,"game":"chess","parent":null,
     "vocabulary":{"derived":["attack"],"delta":["cell"],"units":null},"judge":null,"teacher_used":false,"lemmas":[
     {"name":"has-a-pawn","body":[{"relation":"cell","args":["?x","?y","us","pawn"],"neg":false}],"claim":{"class":"draw"},"stats":{},"meta":{}}]}"#;
    let snap = Snapshot::parse(SNAP.as_bytes()).unwrap();
    assert_eq!((snap.lemmas.len(), snap.claims.len()), (0, 1));
    let view = FactView::new(&chess_spec(), Vocabulary { derived: vec!["attack".into()], delta: vec!["cell".into()], units: None }).unwrap();
    let claims = Claims::new(&snap, &view);
    let g = KernelGame::from_fen("chess", "4k3/8/8/8/8/4K3/4P3/8 w - - 0 1").unwrap();
    let s = g.initial();
    let d = decide_with_store(&g, &s, 0.5, &Count, &Params::default(), None, Some(&claims)).unwrap();
    assert_eq!(steps(&d)[..2], ["forcing", "claim"]);
    let line = d.trace.iter().find(|l| l.starts_with("claim toward: depth")).unwrap();
    assert!(line.contains("claim evaluations"), "{line}");
    let off = decide_with_store(&g, &s, 0.5, &Count, &Params { claims: false, ..Params::default() }, None, Some(&claims)).unwrap();
    assert!(!steps(&off).contains(&"claim".to_string()));
}


// ------------------------------------------------------------------------------------------------------------------------------- the cases, with the claim step scripted

use cs_operator::{Budget, Fact as OFact, Goal, Outcome, Status};

/// An operator that answers one goal with a crafted outcome: stands for the lemma step (`Respond`) or the claim step (`Toward`) so that the controller's reading of each case is tested alone.
struct Scripted { name: &'static str, toward: bool, facts: Vec<OFact>, candidates: Vec<usize> }
impl Operator<GdlGame> for Scripted {
    fn name(&self) -> String { self.name.into() }
    fn accepts(&self, goal: &Goal) -> bool { if self.toward { matches!(goal, Goal::Toward { .. }) } else { matches!(goal, Goal::Respond { .. }) } }
    fn run(&self, _: &GdlGame, _: &<GdlGame as Game>::State, _: &Goal, _: Budget) -> Outcome<<GdlGame as Game>::Move> {
        let mut o = Outcome::new(Status::Read);
        o.facts = self.facts.clone();
        o.candidates = self.candidates.clone();
        o
    }
}

/// The first move (by index) in the starting position's legal moves is "(mark 1 1)", then 1 2, 1 3, 2 1, 2 2 (index 4) ...
/// `classes`: (move index, W / D / U / L, win by the game alone, value); `cands`: the response rules' candidates; `suggested`: the suggestion rules' moves.
fn scripted(classes: &[(usize, char, bool, i32)], cands: &[usize], suggested: &[usize], history: Vec<u64>, claims_on: bool) -> (cs_operator::Decision<<GdlGame as Game>::Move>, GdlGame) {
    let g = GdlGame::load("ttt", TTT).unwrap();
    let s = g.initial();
    let mut cf: Vec<OFact> = vec![];
    for i in 0..9 {
        let (c, game, val) = classes.iter().find(|x| x.0 == i).map(|x| (x.1, x.2, x.3)).unwrap_or(('U', false, 1000));
        cf.push(OFact { relation: "root_class".into(), args: vec![i.to_string(), c.to_string(), if c == 'W' { if game { "game".into() } else { "claim".into() } } else { String::new() }, val.to_string()] });
    }
    let claim_op = Scripted { name: "claim", toward: true, facts: cf, candidates: vec![] };
    let lemma_op = Scripted { name: "lemma", toward: false, facts: suggested.iter().map(|i| OFact { relation: "suggested".into(), args: vec!["r".into(), i.to_string()] }).collect(), candidates: cands.to_vec() };
    let forcing = cs_tactics::operators::ForcingOp;
    let ledger = cs_tactics::operators::LedgerOp::new(&Count);
    let ops: Vec<&dyn Operator<GdlGame>> = vec![&forcing, &ledger, &lemma_op, &claim_op];
    let params = Params { history, ..Params::default() };
    let d = cs_operator::drive(&g, &s, 1.0, &ops, &cs_decision::Fixed { params: &params, lemmas: true, claims: claims_on }, vec![]);
    (d, g)
}
fn played(d: &cs_operator::Decision<<GdlGame as Game>::Move>, g: &GdlGame) -> String { g.move_name(&g.initial(), d.mv.as_ref().unwrap()) }
fn goal_of(d: &cs_operator::Decision<<GdlGame as Game>::Move>) -> String { d.steps.iter().find(|s| s.operator == "ledger").unwrap().goal.describe() }
const W: i32 = 3400;

#[test]
fn no_win_some_unknown_the_draw_has_no_preference_over_the_unknown() {
    // the centre (4) is a draw, the rest unknown; the rule names the corner (0), unknown: it stays first, nothing is avoided
    let (d, g) = scripted(&[(4, 'D', false, 2000)], &[0], &[], vec![], true);
    assert_eq!(goal_of(&d), "ledger(0,prefer 1)");
    assert_eq!(played(&d, &g), "(mark 1 1)");
    // a refuted move is avoided and a rule candidate that is refuted is dropped from the order
    let (d, g) = scripted(&[(4, 'D', false, 2000), (0, 'L', false, 5)], &[0, 1], &[], vec![], true);
    assert_eq!(goal_of(&d), "ledger(0,prefer 1,avoid 1)");
    assert_eq!(played(&d, &g), "(mark 1 2)");
}

#[test]
fn when_every_move_is_a_draw_or_refuted_the_draws_come_first() {
    let all_but = |draws: &[usize]| -> Vec<(usize, char, bool, i32)> { (0..9).map(|i| (i, if draws.contains(&i) { 'D' } else { 'L' }, false, if draws.contains(&i) { 2000 } else { 5 })).collect() };
    // draws 3 and 4; the rule names 0 (refuted): the draws lead, in index order; the refuted are last
    let (d, g) = scripted(&all_but(&[3, 4]), &[0], &[], vec![], true);
    assert_eq!(goal_of(&d), "ledger(0,prefer 2,avoid 7)");
    assert_eq!(played(&d, &g), "(mark 2 1)");
    // a rule candidate that is a draw is first
    let (d, g) = scripted(&all_but(&[3, 4]), &[4], &[], vec![], true);
    assert_eq!(played(&d, &g), "(mark 2 2)");
    // a suggestion that is a draw comes after the rule candidates and before the other draws
    let (d, g) = scripted(&all_but(&[3, 4]), &[], &[4], vec![], true);
    assert_eq!(played(&d, &g), "(mark 2 2)");
    // every move refuted: nothing changes
    let (d, g) = scripted(&all_but(&[]), &[2], &[], vec![], true);
    assert_eq!(goal_of(&d), "ledger(0,prefer 1)");
    assert_eq!(played(&d, &g), "(mark 1 3)");
}

#[test]
fn among_won_moves_the_rule_candidate_leads_then_a_new_position_then_the_nearest_win() {
    let g0 = GdlGame::load("ttt", TTT).unwrap();
    let s = g0.initial();
    let key_after = |i: usize| g0.key(&g0.apply(&s, &g0.moves(&s)[i]));
    let won = [(0, 'W', false, W), (2, 'W', false, W + 50), (4, 'W', false, W)];
    // nearest win first among the new positions
    let (d, g) = scripted(&won, &[], &[], vec![], true);
    assert_eq!(played(&d, &g), "(mark 1 3)");
    assert_eq!(goal_of(&d), "ledger(0,prefer 3,avoid 6)", "every move that is not a win is avoided");
    // the nearest win leads to a position the record holds: the next one, a new position, comes first
    let (d, g) = scripted(&won, &[], &[], vec![key_after(2)], true);
    assert_eq!(played(&d, &g), "(mark 1 1)");
    let (d, g) = scripted(&won, &[], &[], vec![key_after(2), key_after(0)], true);
    assert_eq!(played(&d, &g), "(mark 2 2)");
    // a response rule's candidate that is a win leads whatever the novelty and the distance
    let (d, g) = scripted(&won, &[4], &[], vec![], true);
    assert_eq!(played(&d, &g), "(mark 2 2)");
    // a rule candidate that is not a win is not read first
    let (d, g) = scripted(&won, &[1], &[], vec![], true);
    assert_eq!(played(&d, &g), "(mark 1 3)");
}

#[test]
fn a_suggestion_acts_only_on_a_won_move_and_with_a_claim_step() {
    let won = [(0, 'W', false, W), (4, 'W', false, W)];
    let (d, g) = scripted(&won, &[], &[4], vec![], true);
    assert_eq!(played(&d, &g), "(mark 2 2)", "a suggested W move is read before the other W move");
    // the same suggestion with no claim step: ignored
    let (d, g) = scripted(&won, &[], &[4], vec![], false);
    assert!(!steps(&d).contains(&"claim".to_string()));
    assert_eq!(goal_of(&d), "ledger(0)");
    assert_eq!(played(&d, &g), "(mark 1 1)");
    // suggested move is unknown or refuted: ignored
    let (d, g) = scripted(&[(0, 'W', false, W), (4, 'L', false, 5)], &[], &[4], vec![], true);
    assert_eq!(played(&d, &g), "(mark 1 1)");
    let (d, g) = scripted(&[(3, 'D', false, 2000)], &[], &[4], vec![], true);
    assert_eq!(goal_of(&d), "ledger(0)", "a suggested unknown move changes nothing");
    assert_eq!(played(&d, &g), "(mark 1 1)");
}
