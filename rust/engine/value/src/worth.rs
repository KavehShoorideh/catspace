//! PLAN 79.24 PART B: THE VALUE IN LAYER 3 (SYSTEM_SPEC 5 value, nothing assumed; 6 the estimator; 11 "layer 3 holds two implementations"; piece: the value). The value says how good a position is for a side FROM RESULTS ALONE: a table fitted on the results of games
//! (human games and generated games), never on a number of an outside engine, with no worth assumed for any unit. This module holds the table's interface and the reading of it; the FIT is not here (it is sleep's, in the learning crate and the maker scripts).
//! Game-free: a unit is a kind word the game reports (`Game::units`), read only by name; the module names none.
//!
//! INTERFACE (written before the bodies).
//!   `KindModel`        a trained table of the kind's counts: `kinds` (the names it reads), `intercept` and `coef` per class (win, draw, loss, of the SIDE TO MOVE), `count` (the positions it was fitted on), `source` (what it was fitted on, in words).
//!                      `chances(diff)` takes the count of each kind of the side to move less the other side's, in the order of `kinds`, and returns the side to move's (win, draw, loss) by the multinomial logistic the table was fitted as. `Kept`: written to a
//!                      partition by sleep (`kind_model`), read back by `decoders()`; its standing is a conjecture on its sample.
//!   `CountModel`       the table of the PLAIN COUNT of material the value was fitted beside: every unit counts one, whatever its kind; the same form with one coefficient. `Kept` as `count_model`.
//!   `Value`            `Value::from_memory(reader, entry)` reads a `KindModel` from the long-term memory of one game and is an `Assess`: the answer's rank is the side asked's expected score (win plus half the draw), its chance the side's (win, draw, loss), `sure` the
//!                      table's standing, `read` the kinds' differences it read. A position the game has ended has no value (the result is the result): `Unanswered`.
//!   `Count`            the same over a `CountModel`.
//!   `material(g, s, me, worth)`   our units less theirs by a table of integer worths by kind word (the count the search's `keep` approach reads); `worth` is the record of worths of the long-term memory, never typed in a layer.
//! Called by: the layer's test; `cs_planning::measure` (the material count: one copy); the wake and the binding when a table of the long-term memory is to be read as a value.
use crate::{Answer, Assess, Chance, Question, Unanswered};
use cs_game::{Game, Role};
use cs_rep::serde_json::{json, Value as Json};
use cs_rep::statement::Standing;
use cs_rep::store::{Decoders, Kept, Reader};
use std::any::Any;
use std::collections::HashMap;
use std::sync::Arc;

/// the name of the entry of the long-term memory a value table is kept under
pub const KIND_MODEL: &str = "kind_model";
/// the name of the entry of the plain count's table
pub const COUNT_MODEL: &str = "count_model";

/// Our units less theirs by a table of integer worths by kind word (a kind the table does not name counts 0): the count of material.
pub fn material<G: Game>(g: &G, s: &G::State, me: Role, worth: &HashMap<String, i64>) -> i64 {
    g.units(s).iter().map(|(owner, kind)| { let w = *worth.get(kind).unwrap_or(&0); if *owner == me { w } else { -w } }).sum()
}

fn softmax3(z: [f64; 3]) -> [f64; 3] {
    let m = z.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let e = [(z[0] - m).exp(), (z[1] - m).exp(), (z[2] - m).exp()];
    let s = e[0] + e[1] + e[2];
    [e[0] / s, e[1] / s, e[2] / s]
}

fn num(v: &Json, k: &str) -> Result<f64, String> { v.get(k).and_then(|x| x.as_f64()).ok_or_else(|| format!("{k} is a number")) }
fn nums(v: &Json, k: &str) -> Result<Vec<f64>, String> { v.get(k).and_then(|x| x.as_array()).ok_or_else(|| format!("{k} is a list"))?.iter().map(|x| x.as_f64().ok_or_else(|| format!("{k} holds numbers"))).collect() }
fn text(v: &Json, k: &str) -> Result<String, String> { v.get(k).and_then(|x| x.as_str()).map(|s| s.to_string()).ok_or_else(|| format!("{k} is a text")) }
fn three(v: Vec<f64>, k: &str) -> Result<[f64; 3], String> { <[f64; 3]>::try_from(v).map_err(|_| format!("{k} has three entries")) }

// ---------------------------------------------------------------------------------------------------------------------------------------- the table of the kinds

/// A trained table of the kinds' counts, fitted on results (see the module doc).
#[derive(Debug, Clone, PartialEq)]
pub struct KindModel {
    pub kinds: Vec<String>,
    /// per class (win, draw, loss of the side to move)
    pub intercept: [f64; 3],
    /// per class, one coefficient per kind in the order of `kinds`
    pub coef: [Vec<f64>; 3],
    pub count: u64,
    pub source: String,
}

impl KindModel {
    /// The side to move's chances from the counts of each kind (its own less the other's), in the order of `kinds`.
    pub fn chances(&self, diff: &[i64]) -> Chance {
        let z = |c: usize| self.intercept[c] + self.coef[c].iter().zip(diff).map(|(w, d)| w * *d as f64).sum::<f64>();
        let p = softmax3([z(0), z(1), z(2)]);
        Chance { win: p[0], draw: p[1], loss: p[2] }
    }
    /// The count of each kind of `mover` less the other side's, in the order of `kinds`, read from the game's units by name.
    pub fn diff<G: Game>(&self, g: &G, s: &G::State, mover: Role) -> Vec<i64> {
        let mut d = vec![0i64; self.kinds.len()];
        for (owner, kind) in g.units(s) {
            if let Some(i) = self.kinds.iter().position(|k| *k == kind) { d[i] += if owner == mover { 1 } else { -1 }; }
        }
        d
    }
}

impl Kept for KindModel {
    fn kind(&self) -> &'static str { KIND_MODEL }
    fn standing(&self) -> Standing { Standing::Conjecture(format!("fitted on results only: {} positions; {}", self.count, self.source)) }
    fn describe(&self) -> (&'static str, &'static str) { ("the count of each kind of the side to move less the other side's", "the side to move's chances of win, draw and loss") }
    fn to_json(&self) -> Json { json!({"kinds": self.kinds, "intercept": self.intercept, "coef": self.coef, "count": self.count, "source": self.source}) }
    fn as_any(&self) -> &dyn Any { self }
}

fn decode_kind_model(v: &Json) -> Result<Arc<dyn Kept>, String> {
    let kinds: Vec<String> = v.get("kinds").and_then(|x| x.as_array()).ok_or("kinds")?.iter().map(|k| k.as_str().map(|s| s.to_string()).ok_or("a kind is a text")).collect::<Result<_, _>>()?;
    let rows = v.get("coef").and_then(|x| x.as_array()).ok_or("coef")?;
    if rows.len() != 3 { return Err("coef has a row per class".into()); }
    let row = |i: usize| -> Result<Vec<f64>, String> { rows[i].as_array().ok_or("a row of coef")?.iter().map(|x| x.as_f64().ok_or_else(|| "a coefficient is a number".to_string())).collect() };
    let coef = [row(0)?, row(1)?, row(2)?];
    if coef.iter().any(|r| r.len() != kinds.len()) { return Err("a row of coef has a coefficient per kind".into()); }
    Ok(Arc::new(KindModel { kinds, intercept: three(nums(v, "intercept")?, "intercept")?, coef, count: v.get("count").and_then(|x| x.as_u64()).ok_or("count")?, source: text(v, "source")? }))
}

/// The table of the PLAIN COUNT of material: every unit counts one, the same form with one coefficient per class.
#[derive(Debug, Clone, PartialEq)]
pub struct CountModel { pub intercept: [f64; 3], pub coef: [f64; 3], pub count: u64, pub source: String }

impl CountModel {
    /// The side to move's chances from the count of its units less the other side's.
    pub fn chances(&self, diff: i64) -> Chance {
        let p = softmax3([0, 1, 2].map(|c| self.intercept[c] + self.coef[c] * diff as f64));
        Chance { win: p[0], draw: p[1], loss: p[2] }
    }
    pub fn diff<G: Game>(&self, g: &G, s: &G::State, mover: Role) -> i64 { g.units(s).iter().map(|(o, _)| if *o == mover { 1 } else { -1 }).sum() }
}

impl Kept for CountModel {
    fn kind(&self) -> &'static str { COUNT_MODEL }
    fn standing(&self) -> Standing { Standing::Conjecture(format!("the plain count of material, fitted on results only: {} positions; {}", self.count, self.source)) }
    fn describe(&self) -> (&'static str, &'static str) { ("the count of the side to move's units less the other side's", "the side to move's chances of win, draw and loss") }
    fn to_json(&self) -> Json { json!({"intercept": self.intercept, "coef": self.coef, "count": self.count, "source": self.source}) }
    fn as_any(&self) -> &dyn Any { self }
}

fn decode_count_model(v: &Json) -> Result<Arc<dyn Kept>, String> {
    Ok(Arc::new(CountModel { intercept: three(nums(v, "intercept")?, "intercept")?, coef: three(nums(v, "coef")?, "coef")?, count: v.get("count").and_then(|x| x.as_u64()).ok_or("count")?, source: text(v, "source")? }))
}

/// The decoders of the two kinds of table this layer keeps.
pub fn decoders() -> Decoders { Decoders::new().with(KIND_MODEL, decode_kind_model).with(COUNT_MODEL, decode_count_model) }

// ---------------------------------------------------------------------------------------------------------------------------------------- the implementations

fn answer_of(mover_chance: Chance, mover: Role, side: Role, sure: Standing, read: Vec<String>) -> Answer {
    let chance = if side == mover { mover_chance } else { mover_chance.flip() };
    Answer { rank: vec![chance.score()], raw: None, chance: Some(chance), sure: Some(sure), read }
}

/// THE VALUE: the table of the kinds, fitted on results only, as an implementation of layer 3's one question.
#[derive(Debug, Clone)]
pub struct Value { pub model: KindModel }

impl Value {
    /// Read the table `entry` (by default `KIND_MODEL`) from the long-term memory of one game.
    pub fn from_memory(memory: &Reader, entry: &str) -> Result<Value, Unanswered> {
        let m = memory.get::<KindModel>(entry).ok_or_else(|| Unanswered(format!("the long-term memory of {} holds no table {entry} of the kinds", memory.key().name)))?;
        Ok(Value { model: m.clone() })
    }
}

impl<G: Game> Assess<G> for Value {
    fn name(&self) -> String { format!("value of the kinds ({} positions)", self.model.count) }
    fn assess(&self, q: &Question<G>) -> Result<Answer, Unanswered> {
        let Some(mover) = q.game.mover(q.state) else { return Err(Unanswered("the game has ended here: its result is the result, not a value".into())) };
        let d = self.model.diff(q.game, q.state, mover);
        let read = self.model.kinds.iter().zip(&d).filter(|(_, x)| **x != 0).map(|(k, x)| format!("{k}:{x:+}")).collect();
        Ok(answer_of(self.model.chances(&d), mover, q.side, Kept::standing(&self.model), read))
    }
}

/// THE PLAIN COUNT of material, the baseline the value is held to.
#[derive(Debug, Clone)]
pub struct Count { pub model: CountModel }

impl Count {
    pub fn from_memory(memory: &Reader, entry: &str) -> Result<Count, Unanswered> {
        let m = memory.get::<CountModel>(entry).ok_or_else(|| Unanswered(format!("the long-term memory of {} holds no table {entry} of the plain count", memory.key().name)))?;
        Ok(Count { model: m.clone() })
    }
}

impl<G: Game> Assess<G> for Count {
    fn name(&self) -> String { format!("plain count of material ({} positions)", self.model.count) }
    fn assess(&self, q: &Question<G>) -> Result<Answer, Unanswered> {
        let Some(mover) = q.game.mover(q.state) else { return Err(Unanswered("the game has ended here: its result is the result, not a value".into())) };
        let d = self.model.diff(q.game, q.state, mover);
        Ok(answer_of(self.model.chances(d), mover, q.side, Kept::standing(&self.model), vec![format!("count:{d:+}")]))
    }
}
