//! PLAN 79.44 PART D: A LEMMA LOADS AS A STATEMENT (SYSTEM_SPEC 11 "One lemma machinery, used twice"; piece: the long-term memory's statements; ledger row 49).
//! Python (catspace/incubator/analyst/discovery/rule_lemma.py `statement_of`) writes each lemma of a folder as the JSON of a `Statement`; this test reads those files with the crate's own
//! reader. It holds no lemma and no game: the directory is named by the environment variable CS_LEMMA_STATEMENTS (tests/test_plan79_44.py sets it and runs this test; with none set the
//! test reads nothing and passes). For every file: it reads as a `Statement`, validates, writes back identical, and its standing is the one the file says (Proved(theorem) or Conjecture).
use cs_rep::statement::{Holder, Says, Standing, Statement};
use serde_json::Value;

#[test]
fn every_statement_a_lemma_was_written_as_reads_validates_and_round_trips() {
    let Ok(dir) = std::env::var("CS_LEMMA_STATEMENTS") else { return };
    let mut n = 0;
    let (mut proved, mut conjecture) = (0, 0);
    let mut names: Vec<_> = std::fs::read_dir(&dir).expect("the directory is there").map(|e| e.unwrap().path()).filter(|p| p.extension().map_or(false, |x| x == "json")).collect();
    names.sort();
    for path in names {
        let text = std::fs::read_to_string(&path).unwrap();
        let v: Value = serde_json::from_str(&text).unwrap();
        let s = Statement::from_json(&v).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        s.validate().unwrap();
        assert_eq!(s.to_json(), v, "{}: written back identical", path.display());
        assert_eq!(Statement::from_json(&s.to_json()).unwrap(), s);
        assert_eq!(s.holder, Holder::Ours);
        assert_eq!(s.says, Says::Holds);
        match &s.standing {
            Standing::Proved(t) => { assert!(!t.is_empty()); proved += 1; }
            Standing::Conjecture(x) => { assert!(!x.is_empty()); conjecture += 1; }
            other => panic!("{}: a lemma's standing is Proved or Conjecture, not {other:?}", path.display()),
        }
        n += 1;
    }
    println!("OBS 79.44 D: {n} lemma statements read by cs_rep: {proved} Proved, {conjecture} Conjecture");
}
