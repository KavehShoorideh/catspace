//! A SNAPSHOT (plan 78.14): an immutable, content-addressed state of the store for one game -- its fact vocabulary, the judge it plays by, the lemmas it has learned
//! and the snapshot it came from. The file is named by its id, the first 16 hex digits of the SHA-256 of its bytes, and a file that does not hash to its name is
//! refused: a batch that says it read snapshot X read exactly X. Sleep publishes the next snapshot; wake only reads. (The valuation stays the legacy store's entry
//! and is copied in as the judge; nothing here replaces it.)
//!
//! JSON: {"format": 1, "game", "parent": id or null, "vocabulary": {"derived": [...], "delta": [...], "units": null | {"relation", "owner", "kind"}},
//!        "judge": null | {"name", "worth": {kind: n}, "ours_first"}, "teacher_used": bool, "lemmas": [lemma...]}
use crate::claim::{Claim, Rule};
use crate::lemma::Lemma;
use crate::sha::content_id;
use cs_game::{UnitDecl, Vocabulary};
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::Path;

pub const FORMAT: u64 = 1;

#[derive(Debug, Clone, PartialEq)]
pub struct Judge { pub name: String, pub worth: BTreeMap<String, i64>, pub ours_first: bool }

#[derive(Debug, Clone)]
pub struct Snapshot {
    pub id: String,
    pub game: String,
    pub parent: Option<String>,
    pub vocabulary: Vocabulary,
    pub judge: Option<Judge>,
    /// The RESPONSE rules (the lemma operator and the priors read these only).
    pub lemmas: Vec<Lemma>,
    /// Plan 78.36: the CLAIM rules (a position's class for the side to move; read by the claim search only). They stand in the same JSON list as the lemmas, so a snapshot with
    /// no claim is byte for byte what it was.
    pub claims: Vec<Claim>,
    /// Whether a teacher's labels went into any lemma (the results can be reported with and without it).
    pub teacher_used: bool,
}

pub fn vocabulary_from_json(v: &Value) -> Result<Vocabulary, String> {
    let list = |key: &str| -> Result<Vec<String>, String> {
        match v.get(key) {
            None | Some(Value::Null) => Ok(vec![]),
            Some(a) => a.as_array().ok_or_else(|| format!("vocabulary.{key} is a list"))?.iter().map(|x| x.as_str().map(|s| s.to_string()).ok_or_else(|| format!("vocabulary.{key} holds text"))).collect(),
        }
    };
    let units = match v.get("units") {
        None | Some(Value::Null) => None,
        Some(u) => Some(UnitDecl {
            relation: u.get("relation").and_then(|x| x.as_str()).ok_or("units.relation")?.to_string(),
            owner: u.get("owner").and_then(|x| x.as_u64()).ok_or("units.owner")? as usize,
            kind: match u.get("kind") { None | Some(Value::Null) => None, Some(k) => Some(k.as_u64().ok_or("units.kind")? as usize) },
        }),
    };
    Ok(Vocabulary { derived: list("derived")?, delta: list("delta")?, units })
}

impl Snapshot {
    /// Read a snapshot from its bytes; the id is the bytes' address.
    pub fn parse(bytes: &[u8]) -> Result<Snapshot, String> {
        let id = content_id(bytes);
        let v: Value = serde_json::from_slice(bytes).map_err(|e| format!("not JSON: {e}"))?;
        if v.get("format").and_then(|x| x.as_u64()) != Some(FORMAT) { return Err(format!("the snapshot format is {FORMAT}")); }
        let game = v.get("game").and_then(|x| x.as_str()).ok_or("a snapshot names its game")?.to_string();
        let parent = match v.get("parent") { None | Some(Value::Null) => None, Some(p) => Some(p.as_str().ok_or("parent is an id")?.to_string()) };
        let vocabulary = vocabulary_from_json(v.get("vocabulary").ok_or("a snapshot has a vocabulary")?)?;
        let judge = match v.get("judge") {
            None | Some(Value::Null) => None,
            Some(j) => Some(Judge {
                name: j.get("name").and_then(|x| x.as_str()).unwrap_or("").to_string(),
                worth: j.get("worth").and_then(|x| x.as_object()).ok_or("judge.worth")?.iter().map(|(k, x)| x.as_i64().map(|n| (k.clone(), n)).ok_or("judge.worth holds numbers")).collect::<Result<_, _>>()?,
                ours_first: j.get("ours_first").and_then(|x| x.as_bool()).unwrap_or(false),
            }),
        };
        let rules = v.get("lemmas").and_then(|x| x.as_array()).ok_or("a snapshot has lemmas")?.iter().map(Rule::from_json).collect::<Result<Vec<_>, _>>()?;
        let mut names = std::collections::BTreeSet::new();
        let (mut lemmas, mut claims) = (vec![], vec![]);
        for r in rules {
            match r { Rule::Response(l) => lemmas.push(l), Rule::Claim(c) => claims.push(c) }
        }
        for c in &claims {
            if !names.insert(c.name.clone()) { return Err(format!("two rules are named {}", c.name)); }
        }
        for l in &lemmas {
            if !names.insert(l.name.clone()) { return Err(format!("two lemmas are named {}", l.name)); }
            for r in l.response.removed.iter().chain(l.response.added.iter()) {
                if !vocabulary.delta.contains(&r.relation) { return Err(format!("{}: the response speaks of {}, which is no delta relation of {game}", l.name, r.relation)); }
            }
            for b in &l.body {
                if vocabulary.derived.contains(&b.relation) || vocabulary.delta.contains(&b.relation) { continue; }
                // a body may also read a state relation outside the delta list (control, castling_right ...): the fact view knows them, the snapshot cannot
            }
        }
        Ok(Snapshot { id, game, parent, vocabulary, judge, lemmas, claims, teacher_used: v.get("teacher_used").and_then(|x| x.as_bool()).unwrap_or(false) })
    }

    /// Read `<id>.json`: the file must hash to its name.
    pub fn load(path: &Path) -> Result<Snapshot, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let s = Snapshot::parse(&bytes).map_err(|e| format!("{}: {e}", path.display()))?;
        let stem = path.file_stem().and_then(|x| x.to_str()).unwrap_or("");
        if stem != s.id { return Err(format!("{}: the file hashes to {} -- a snapshot is named by its content", path.display(), s.id)); }
        Ok(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const TEXT: &str = r#"{"format":1,"game":"toy","parent":null,"vocabulary":{"derived":["threat"],"delta":["piece"],"units":null},"judge":{"name":"count","worth":{"u":1},"ours_first":true},"teacher_used":false,"lemmas":[
      {"name":"step-away","body":[{"relation":"threat","args":["?o","?x","?y"],"neg":false}],"response":{"removed":[{"relation":"piece","args":["?y","?s"],"neg":false}],"added":[{"relation":"piece","args":["?z","?s"],"neg":false}]},"stats":{},"meta":{}}]}"#;
    #[test]
    fn a_snapshot_is_its_content() {
        let s = Snapshot::parse(TEXT.as_bytes()).unwrap();
        assert_eq!(s.id, content_id(TEXT.as_bytes()));
        assert_eq!(s.lemmas.len(), 1);
        assert_eq!(s.vocabulary.delta, vec!["piece".to_string()]);
        assert!(s.judge.unwrap().ours_first);
        let changed = TEXT.replace("step-away", "step-aside");
        assert_ne!(Snapshot::parse(changed.as_bytes()).unwrap().id, s.id);
    }
    #[test]
    fn a_snapshot_without_claims_keeps_its_address() {
        // the id is the address of the bytes: pinned (the SHA-256 of TEXT, computed outside), and a snapshot of response rules only holds no claim
        let s = Snapshot::parse(TEXT.as_bytes()).unwrap();
        assert_eq!(s.id, "3e874a092878bd16");
        assert!(s.claims.is_empty());
        assert_eq!(s.lemmas.len(), 1);
    }
    #[test]
    fn a_suggestion_is_content_and_a_snapshot_without_one_is_unchanged() {
        let sug = TEXT.replace(r#""name":"step-away","body""#, r#""name":"step-away","suggest":true,"body""#);
        let s = Snapshot::parse(sug.as_bytes()).unwrap();
        assert!(s.lemmas[0].suggest);
        assert_ne!(s.id, Snapshot::parse(TEXT.as_bytes()).unwrap().id);
        assert!(!Snapshot::parse(TEXT.as_bytes()).unwrap().lemmas[0].suggest);
        assert_eq!(Snapshot::parse(TEXT.as_bytes()).unwrap().id, "3e874a092878bd16");
        let on_claim = r#","suggest":true,"claim":{"class":"loss"}"#;
        let bad = TEXT.trim_end_matches("]}").to_string() + r#",{"name":"c","body":[{"relation":"threat","args":["?o","?x","?y"],"neg":false}]"# + on_claim + r#","stats":{},"meta":{}}]}"#;
        assert!(Snapshot::parse(bad.as_bytes()).unwrap_err().contains("suggest"));
    }
    #[test]
    fn claims_stand_in_the_lemma_list_and_are_kept_apart() {
        let claim = r#",{"name":"threatened-is-lost","body":[{"relation":"threat","args":["?o","?x","?y"],"neg":false}],"claim":{"class":"loss"},"stats":{},"meta":{}}]}"#;
        let text = TEXT.trim_end_matches("]}").to_string() + claim;
        let s = Snapshot::parse(text.as_bytes()).unwrap();
        assert_eq!((s.lemmas.len(), s.claims.len()), (1, 1));
        assert_eq!(s.claims[0].class, crate::claim::Class::Loss);
        assert_ne!(s.id, Snapshot::parse(TEXT.as_bytes()).unwrap().id, "a claim is content: the address changes");
        // two rules of one name, one of each kind: refused
        let twin = text.replace("threatened-is-lost", "step-away");
        assert!(Snapshot::parse(twin.as_bytes()).unwrap_err().contains("named"));
        // a rule that is both: refused
        let both = text.replace(r#""claim":{"class":"loss"}"#, r#""claim":{"class":"loss"},"response":{"removed":[{"relation":"piece","args":["?y","?s"],"neg":false}],"added":[]}"#);
        assert!(Snapshot::parse(both.as_bytes()).unwrap_err().contains("not both"));
    }
    #[test]
    fn a_response_outside_the_delta_relations_is_refused() {
        let bad = TEXT.replace(r#""removed":[{"relation":"piece""#, r#""removed":[{"relation":"threat""#);
        assert!(Snapshot::parse(bad.as_bytes()).unwrap_err().contains("delta"));
    }
    #[test]
    fn a_file_must_hash_to_its_name() {
        let dir = std::env::temp_dir().join(format!("cs_memory_snap_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let id = content_id(TEXT.as_bytes());
        std::fs::write(dir.join(format!("{id}.json")), TEXT).unwrap();
        assert!(Snapshot::load(&dir.join(format!("{id}.json"))).is_ok());
        std::fs::write(dir.join("0123456789abcdef.json"), TEXT).unwrap();
        assert!(Snapshot::load(&dir.join("0123456789abcdef.json")).unwrap_err().contains("named by its content"));
        std::fs::remove_dir_all(&dir).ok();
    }
}
