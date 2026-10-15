//! THE EVENT LOOP'S KERNEL (Kaveh 2026-10-15, docs/ANALYST_PLAN.md section 40; the blackboard / Soar lineage). Nothing here knows chess: the words
//! carry the chess. This file holds the bookkeeping only --
//!   EVENT      a record (kind, ply, subject, payload): push(move) / found(finding) / reason(finding, holder) / impasse(question). `subject` is a premise key
//!              (the canonical key of a finding, "KIND|side|pieces|squares"); `args` and `stake` ride on it, inherited down a chain.
//!   CONDITION  a premise-key pattern ("SAFE|*") on an event kind that OPENS a question kind. A question opens only from an event whose condition matches,
//!              never by a sweep.
//!   AGENDA     the open questions, ordered by `policy` (fixed: highest stake, then cheapest; swappable later). A question opens only if its stake beats the
//!              best value already known for its side (the alpha-beta: a settled stake closes the chain), once per canonical key, and while its chain's budget
//!              (set by the chain's stake) and the decision's budget last.
//!   CHAIN      the records (question, answer, reason, stake, cost) in the order they were answered -- the trace, as data.
//! The side that acts is a number (0 / 1); a reason on one side's finding opens the OTHER side's question (`flip`), the same code for both.
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum EvKind { Push, Found, Reason, Impasse }
impl EvKind {
    pub fn name(&self) -> &'static str { match self { EvKind::Push => "push", EvKind::Found => "found", EvKind::Reason => "reason", EvKind::Impasse => "impasse" } }
}

#[derive(Clone, Debug)]
pub struct Event { pub id: usize, pub kind: EvKind, pub ply: u32, pub subject: String, pub payload: String, pub actor: u8, pub stake: i64, pub args: Vec<String>, pub parent: Option<usize> }

#[derive(Clone, Debug)]
pub struct Condition { pub name: String, pub on: EvKind, pub pattern: String, pub opens: String, pub flip: bool, pub exempt: bool, pub fresh: bool }
impl Condition {
    /// `exempt`: a lookup that is not a contest for the side's best value (a question answered from a table, cost 0) opens whatever its stake.
    pub fn new(name: &str, on: EvKind, pattern: &str, opens: &str, flip: bool, exempt: bool) -> Condition {
        Condition { name: name.into(), on, pattern: pattern.into(), opens: opens.into(), flip, exempt, fresh: false }
    }
    /// `fresh`: the question starts a chain of its own (its budget is set by its own stake), though it is shown under the question that posted it.
    pub fn fresh(mut self) -> Condition { self.fresh = true; self }
    /// A premise-key pattern: '|'-separated segments, '*' matches one segment, a trailing '*' the rest.
    pub fn matches(&self, ev: &Event) -> bool {
        if ev.kind != self.on { return false; }
        let (p, s): (Vec<&str>, Vec<&str>) = (self.pattern.split('|').collect(), ev.subject.split('|').collect());
        for (i, seg) in p.iter().enumerate() {
            if *seg == "*" && i == p.len() - 1 { return true; }
            match s.get(i) { Some(x) if *seg == "*" || seg == x => {}, _ => return false }
        }
        p.len() == s.len()
    }
}

#[derive(Clone, Debug)]
pub struct Question { pub id: usize, pub kind: String, pub key: String, pub actor: u8, pub args: Vec<String>, pub payload: String, pub stake: i64, pub cost: i64, pub from_event: usize,
                      pub parent: Option<usize>, pub depth: u32, pub root: usize, pub exempt: bool }

/// One answered question: the line of the trace.
#[derive(Clone, Debug)]
pub struct ChainRec { pub q: usize, pub parent: Option<usize>, pub depth: u32, pub kind: String, pub key: String, pub actor: u8, pub args: Vec<String>, pub text: String, pub answer: String,
                      pub reason: String, pub stake: i64, pub cost: i64, pub value: Option<(i64, i64)> }

pub struct Agenda {
    pub events: Vec<Event>,
    pub conditions: Vec<Condition>,
    pub open: Vec<Question>,
    pub chains: Vec<ChainRec>,
    pub best: [i64; 2],                           // the best value known for each side
    pub refused: Vec<String>,                     // questions that matched a condition and were not opened, with why
    asked: HashSet<String>,
    pub next_q: usize,
    pub budget: u32,                              // the decision's total
    pub spent: u32,
    root_left: HashMap<usize, u32>,               // the budget left in each chain (keyed by its root question)
    qinfo: HashMap<usize, (u32, usize)>,          // question -> (depth, root)
}

/// The budget of a chain from its stake: a stake of 1 gets a short chain.
pub fn chain_budget(stake: i64) -> u32 { (3 + 2 * stake.clamp(1, 9)) as u32 }

/// THE POLICY (fixed for now, swappable): highest stake, then the cheapest (a lookup before a chain), then the first opened. -> the index in `open`.
pub fn policy(open: &[Question]) -> Option<usize> {
    open.iter().enumerate().min_by_key(|(_, q)| (std::cmp::Reverse(q.stake), q.cost, q.id)).map(|(i, _)| i)
}

impl Agenda {
    pub fn new(conditions: Vec<Condition>, budget: u32) -> Agenda {
        Agenda { events: vec![], conditions, open: vec![], chains: vec![], best: [0, 0], refused: vec![], asked: HashSet::new(), next_q: 0, budget, spent: 0, root_left: HashMap::new(), qinfo: HashMap::new() }
    }
    /// Post an event; every condition that matches proposes its question, which opens if it passes the gates. -> the ids of the questions opened.
    /// `key_of` names the question canonically (kind, side, args) so the same question is asked once.
    pub fn post(&mut self, mut ev: Event, parent_q: Option<usize>, cost_of: &dyn Fn(&str) -> i64) -> Vec<usize> {
        ev.id = self.events.len();
        self.events.push(ev.clone());
        let mut opened = vec![];
        let conds: Vec<Condition> = self.conditions.iter().filter(|c| c.matches(&ev)).cloned().collect();
        for c in conds {
            let actor = if c.flip { 1 - ev.actor } else { ev.actor };
            let key = format!("{}|{}|{}", c.opens, actor, ev.args.join(","));
            if self.asked.contains(&key) { continue; }
            if !c.exempt && ev.stake <= self.best[actor as usize] {
                self.refused.push(format!("{key}: stake {} does not beat the best known value {} for that side", ev.stake, self.best[actor as usize]));
                continue;
            }
            let id = self.next_q;
            let (depth, root) = match parent_q { Some(p) if !c.fresh => { let (d, r) = self.qinfo.get(&p).copied().unwrap_or((0, p)); (d + 1, r) },
                                                 Some(p) => { let (d, _) = self.qinfo.get(&p).copied().unwrap_or((0, p)); (d + 1, id) }, None => (0, id) };
            if root == id { self.root_left.insert(id, chain_budget(ev.stake)); }
            let left = self.root_left.get(&root).copied().unwrap_or(0);
            if !c.exempt && (left == 0 || self.spent + self.open.len() as u32 >= self.budget) {
                self.refused.push(format!("{key}: the chain's budget (or the decision's) is spent"));
                continue;
            }
            self.next_q += 1;
            self.qinfo.insert(id, (depth, root));
            self.asked.insert(key.clone());
            self.open.push(Question { id, kind: c.opens.clone(), key, actor, args: ev.args.clone(), payload: ev.payload.clone(), stake: ev.stake, cost: cost_of(&c.opens), from_event: ev.id, parent: parent_q, depth, root, exempt: c.exempt });
            opened.push(id);
        }
        opened
    }
    /// Take the next question by the policy.
    pub fn take(&mut self) -> Option<Question> {
        let i = policy(&self.open)?;
        let q = self.open.remove(i);
        self.spent += 1;
        if !q.exempt { if let Some(l) = self.root_left.get_mut(&q.root) { *l = l.saturating_sub(1); } }
        Some(q)
    }
    /// Close a question into the chain.
    pub fn settle(&mut self, q: &Question, text: String, answer: String, reason: String, cost: i64, value: Option<(i64, i64)>) {
        self.chains.push(ChainRec { q: q.id, parent: q.parent, depth: q.depth, kind: q.kind.clone(), key: q.key.clone(), actor: q.actor, args: q.args.clone(), text, answer, reason, stake: q.stake, cost, value });
    }
    /// A value found for a side (a conditional one counts: the armed tactic): later questions of that side must beat it.
    pub fn raise(&mut self, side: u8, v: i64) { if v > self.best[side as usize] { self.best[side as usize] = v; } }
    pub fn to_json(&self) -> Value {
        json!({"events": self.events.iter().map(|e| json!({"id": e.id, "kind": e.kind.name(), "ply": e.ply, "subject": e.subject, "payload": e.payload, "actor": e.actor, "stake": e.stake, "parent": e.parent})).collect::<Vec<_>>(),
               "chains": self.chains.iter().map(|c| json!({"q": c.q, "parent": c.parent, "depth": c.depth, "kind": c.kind, "key": c.key, "actor": c.actor, "text": c.text, "answer": c.answer, "reason": c.reason, "stake": c.stake, "cost": c.cost, "value": c.value})).collect::<Vec<_>>(),
               "open": self.open.len(), "spent": self.spent, "refused": self.refused, "best": self.best})
    }
}
