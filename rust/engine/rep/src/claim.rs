//! PLAN 79.21 STAGE A: THE CLAIM RULE AS DATA (SYSTEM_SPEC 11, the layers; piece: representation). A claim is a statement about a POSITION: "IF body THEN the position's class for the side to move is win / draw / loss", the body written in the
//! mover's perspective (`us` / `them`) as a lemma's is. The data types (`Class`, `Claim`, `Rule`) and their JSON live here, below the rules, because a snapshot holds them; the READER that binds claims to a game's facts (`Claims`) stays in cs_memory.
//! JSON: {"name", "body": [lit...], "claim": {"class": "win" | "draw" | "loss"}, "stats": {...}, "meta": {...}}. A rule carries EITHER a `response` or a `claim`, never both.
use crate::lemma::{validate_body, Lemma, Lit};
use serde_json::{json, Value};
use std::collections::BTreeSet;

/// The class of a position for the side to move.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Class { Loss, Draw, Win }

impl Class {
    pub fn word(&self) -> &'static str { match self { Class::Loss => "loss", Class::Draw => "draw", Class::Win => "win" } }
    pub fn parse(s: &str) -> Option<Class> { match s { "win" => Some(Class::Win), "draw" => Some(Class::Draw), "loss" => Some(Class::Loss), _ => None } }
    /// The same class seen by the other side.
    pub fn flip(&self) -> Class { match self { Class::Win => Class::Loss, Class::Loss => Class::Win, Class::Draw => Class::Draw } }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Claim {
    pub name: String,
    pub body: Vec<Lit>,
    pub class: Class,
    pub stats: Value,
    pub meta: Value,
}

impl Claim {
    pub fn to_json(&self) -> Value {
        json!({"name": self.name, "body": self.body.iter().map(Lit::to_json).collect::<Vec<_>>(), "claim": {"class": self.class.word()}, "stats": self.stats, "meta": self.meta})
    }

    pub fn from_json(v: &Value) -> Result<Claim, String> {
        let name = v.get("name").and_then(|x| x.as_str()).ok_or("a rule has a name")?.to_string();
        if v.get("suggest").is_some() { return Err(format!("{name}: suggest belongs to a response rule, not a claim")); }
        if v.get("response").is_some() { return Err(format!("{name}: a rule carries a response or a claim, not both")); }
        let c = v.get("claim").ok_or_else(|| format!("{name}: no claim"))?;
        let class = c.get("class").and_then(|x| x.as_str()).and_then(Class::parse).ok_or_else(|| format!("{name}: a claim's class is win, draw or loss"))?;
        let body = v.get("body").and_then(|x| x.as_array()).ok_or_else(|| format!("{name}: no body"))?.iter().map(Lit::from_json).collect::<Result<Vec<_>, _>>()?;
        let k = Claim { name, body, class, stats: v.get("stats").cloned().unwrap_or(Value::Null), meta: v.get("meta").cloned().unwrap_or(Value::Null) };
        k.validate()?;
        Ok(k)
    }

    pub fn validate(&self) -> Result<(), String> { validate_body(&self.name, &self.body) }

    /// The relations the body reads.
    pub fn body_relations(&self) -> BTreeSet<String> { self.body.iter().map(|l| l.relation.clone()).collect() }
}

/// A rule of either kind, as the JSON says.
#[derive(Debug, Clone, PartialEq)]
pub enum Rule { Response(Lemma), Claim(Claim) }

impl Rule {
    /// Read a rule: a `claim` makes a claim, a `response` a lemma; both or neither is refused.
    pub fn from_json(v: &Value) -> Result<Rule, String> {
        let name = v.get("name").and_then(|x| x.as_str()).unwrap_or("").to_string();
        match (v.get("response").is_some(), v.get("claim").is_some()) {
            (true, true) => Err(format!("{name}: a rule carries a response or a claim, not both")),
            (false, false) => Err(format!("{name}: a rule carries a response or a claim; this one has neither")),
            (true, false) => Lemma::from_json(v).map(Rule::Response),
            (false, true) => Claim::from_json(v).map(Rule::Claim),
        }
    }
    pub fn name(&self) -> &str { match self { Rule::Response(l) => &l.name, Rule::Claim(c) => &c.name } }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn win_rule() -> Value {
        json!({"name": "x-race", "body": [{"relation": "cell", "args": ["?f", "?r", "us", "kind_x"], "neg": false}], "claim": {"class": "win"}, "stats": {}, "meta": {}})
    }

    #[test]
    fn a_claim_round_trips_and_a_rule_is_one_kind() {
        let c = Claim::from_json(&win_rule()).unwrap();
        assert_eq!(c.class, Class::Win);
        assert_eq!(Claim::from_json(&c.to_json()).unwrap(), c);
        assert_eq!(c.to_json()["claim"]["class"], "win");
        assert!(matches!(Rule::from_json(&win_rule()).unwrap(), Rule::Claim(_)));
        // both a response and a claim: refused; neither: refused; a class that is none of the three: refused
        let mut both = win_rule();
        both["response"] = json!({"removed": [{"relation": "cell", "args": ["?f", "?r", "us", "kind_x"], "neg": false}], "added": []});
        assert!(Rule::from_json(&both).unwrap_err().contains("not both"));
        assert!(Claim::from_json(&both).unwrap_err().contains("not both"));
        let mut neither = win_rule();
        neither.as_object_mut().unwrap().remove("claim");
        assert!(Rule::from_json(&neither).unwrap_err().contains("neither"));
        let mut odd = win_rule();
        odd["claim"] = json!({"class": "maybe"});
        assert!(Rule::from_json(&odd).unwrap_err().contains("win, draw or loss"));
        // a body with no positive literal, or an unsafe negation, is refused as for a lemma
        let mut empty = win_rule();
        empty["body"] = json!([]);
        assert!(Rule::from_json(&empty).unwrap_err().contains("positive literal"));
        let mut unsafe_neg = win_rule();
        unsafe_neg["body"] = json!([{"relation": "cell", "args": ["?f", "?r", "us", "kind_x"], "neg": false}, {"relation": "cell", "args": ["?zz", "?r", "them", "kind_x"], "neg": true}]);
        assert!(Rule::from_json(&unsafe_neg).unwrap_err().contains("binds"));
        // a response rule is read as a lemma, as ever
        let l = json!({"name": "x", "body": [{"relation": "cell", "args": ["?f", "?r", "us", "kind_x"], "neg": false}],
                       "response": {"removed": [{"relation": "cell", "args": ["?f", "?r", "us", "kind_x"], "neg": false}], "added": []}, "stats": {}, "meta": {}});
        assert!(matches!(Rule::from_json(&l).unwrap(), Rule::Response(_)));
    }

    #[test]
    fn classes_flip() {
        assert_eq!(Class::Win.flip(), Class::Loss);
        assert_eq!(Class::Draw.flip(), Class::Draw);
        assert!(Class::Loss < Class::Draw && Class::Draw < Class::Win);
    }
}
