//! THE CARD CORPUS (plan 78.14): the positions a lemma is verified on. A CARD holds a position's facts and, for every legal move, the move's effect on the facts
//! and its exact value (read by a search of stated depth) with the units each side loses along its best line. A lemma is evaluated over cards by pure matching, no
//! search: "if the body holds, a move of the response's kind is among the best" is read off the table. The exact searching happens once, when the cards are made.
use crate::lemma::Lemma;
use crate::matcher::{bindings, bindings_fused, Binding, FactIndex};
use cs_game::Fact;
use serde_json::{json, Value};

#[derive(Debug, Clone, PartialEq)]
pub struct CardMove {
    pub name: String,
    pub removed: Vec<Fact>,
    pub added: Vec<Fact>,
    /// The move's value for the side to move, by the judge, exact at the card's depth.
    pub value: i64,
    /// The units the mover loses and wins along the move's best line.
    pub lost: u32,
    pub gained: u32,
    /// The effect (removed, added; in the mover's perspective) of the first move along the best line that takes one of the mover's units: where the loss happens
    /// and by what. Empty when the line takes none.
    pub loss_removed: Vec<Fact>,
    pub loss_added: Vec<Fact>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Card {
    pub id: String,
    /// The card's source (a game record): cards of one group are not independent of each other.
    pub group: String,
    pub facts: Vec<Fact>,
    pub moves: Vec<CardMove>,
    pub depth: i32,
    /// Whether the clock or fuse cut the reading (the values are then lower bounds of the work, not exact): such a card is not used to refute.
    pub cut: bool,
}

fn facts_json(fs: &[Fact]) -> Value { Value::Array(fs.iter().map(|f| { let mut v = vec![Value::String(f.relation.clone())]; v.extend(f.args.iter().map(|a| Value::String(a.clone()))); Value::Array(v) }).collect()) }
fn facts_from(v: &Value) -> Result<Vec<Fact>, String> {
    v.as_array().ok_or("facts are a list")?.iter().map(|f| {
        let a = f.as_array().ok_or("a fact is a list")?;
        let rel = a.first().and_then(|x| x.as_str()).ok_or("a fact starts with its relation")?;
        Ok(Fact { relation: rel.to_string(), args: a[1..].iter().map(|x| x.as_str().map(|s| s.to_string()).ok_or("a fact holds text")).collect::<Result<_, _>>()? })
    }).collect()
}

impl Card {
    pub fn to_json(&self) -> Value {
        json!({"id": self.id, "group": self.group, "depth": self.depth, "cut": self.cut, "facts": facts_json(&self.facts),
               "moves": self.moves.iter().map(|m| json!({"name": m.name, "removed": facts_json(&m.removed), "added": facts_json(&m.added), "value": m.value, "lost": m.lost, "gained": m.gained,
                                                      "loss_removed": facts_json(&m.loss_removed), "loss_added": facts_json(&m.loss_added)})).collect::<Vec<_>>()})
    }
    pub fn from_json(v: &Value) -> Result<Card, String> {
        let moves = v.get("moves").and_then(|x| x.as_array()).ok_or("a card has moves")?.iter().map(|m| Ok(CardMove {
            name: m.get("name").and_then(|x| x.as_str()).ok_or("a move has a name")?.to_string(),
            removed: facts_from(m.get("removed").ok_or("removed")?)?, added: facts_from(m.get("added").ok_or("added")?)?,
            value: m.get("value").and_then(|x| x.as_i64()).ok_or("value")?, lost: m.get("lost").and_then(|x| x.as_u64()).unwrap_or(0) as u32, gained: m.get("gained").and_then(|x| x.as_u64()).unwrap_or(0) as u32,
            loss_removed: match m.get("loss_removed") { Some(f) => facts_from(f)?, None => vec![] },
            loss_added: match m.get("loss_added") { Some(f) => facts_from(f)?, None => vec![] },
        })).collect::<Result<Vec<_>, String>>()?;
        Ok(Card { id: v.get("id").and_then(|x| x.as_str()).unwrap_or("").to_string(), group: v.get("group").and_then(|x| x.as_str()).unwrap_or("").to_string(),
                  facts: facts_from(v.get("facts").ok_or("facts")?)?, moves, depth: v.get("depth").and_then(|x| x.as_i64()).unwrap_or(0) as i32, cut: v.get("cut").and_then(|x| x.as_bool()).unwrap_or(false) })
    }
}

/// A corpus of cards with the indexes the matcher reads.
pub struct CardSet {
    pub cards: Vec<Card>,
    facts: Vec<FactIndex>,
    removed: Vec<Vec<FactIndex>>,
    added: Vec<Vec<FactIndex>>,
}

/// The rows one card's matching may examine before it is cut (a named fuse; a cut evaluation says so in `truncated` and is never admitted).
pub const STEP_FUSE: u64 = 5_000;

/// What a lemma did on a corpus.
#[derive(Debug, Clone, Default)]
pub struct Eval {
    /// Cards the body matched, and those where the claim held (an optimal move of the response's kind exists) / failed.
    pub fires: usize,
    pub holds: usize,
    pub refuted: usize,
    /// Of the firing cards, those whose reading was cut (not counted as refutations nor holds: unknown).
    pub unknown: usize,
    /// Distinct groups among the firing cards, and among those where it held.
    pub groups: usize,
    pub groups_held: usize,
    /// holds / (holds + refuted), the rate a random choice of as many moves would reach, and the difference.
    pub precision: f64,
    pub expected: f64,
    pub excess: f64,
    /// The mean share of the legal moves the response matches (near 1: the claim says little).
    pub selectivity: f64,
    /// A card's matching was cut (by the bindings limit or the step fuse): the evaluation stopped there and is partial.
    pub truncated: bool,
    /// The evaluation stopped early because more cards refuted the lemma than `max_refuted` allows: the counts are those of the cards read so far.
    pub stopped: bool,
    /// The rows the matcher examined.
    pub steps: u64,
    /// Per firing card: (card index, holds, moves matched). Unknown cards are not listed.
    pub cases: Vec<(usize, bool, usize)>,
    /// With `all_named` (plan 78.31): firing cards where every legal move is optimal (the rule cannot be wrong there): not counted as firings, holds or refutations.
    pub trivial: usize,
}

impl CardSet {
    pub fn new(cards: Vec<Card>) -> CardSet {
        let facts = cards.iter().map(|c| FactIndex::new(&c.facts)).collect();
        let removed = cards.iter().map(|c| c.moves.iter().map(|m| FactIndex::new(&m.removed)).collect()).collect();
        let added = cards.iter().map(|c| c.moves.iter().map(|m| FactIndex::new(&m.added)).collect()).collect();
        CardSet { cards, facts, removed, added }
    }
    pub fn len(&self) -> usize { self.cards.len() }
    pub fn is_empty(&self) -> bool { self.cards.is_empty() }

    /// Evaluate `lemma` on the cards at `indices` (all when None).
    pub fn eval(&self, lemma: &Lemma, indices: Option<&[usize]>, max_bindings: usize) -> Eval { self.eval_bounded(lemma, indices, max_bindings, None) }

    /// `eval` that stops reading as soon as a matching is cut, or as soon as more cards than `max_refuted` have refuted the lemma (a gate that allows none stops at the
    /// first): the verdict "not admissible" is then certain and the rest of the corpus need not be read.
    pub fn eval_bounded(&self, lemma: &Lemma, indices: Option<&[usize]>, max_bindings: usize, max_refuted: Option<usize>) -> Eval { self.eval_with(lemma, indices, max_bindings, max_refuted, false) }

    /// `eval_bounded` with the quantifier chosen. `all_named` false: a firing HOLDS when some move the response names is optimal (as ever). `all_named` true (plan 78.31): it HOLDS only when EVERY move
    /// it names is optimal, is REFUTED when one is not (or none is named), and a card where every legal move is optimal is TRIVIAL (counted in `trivial`, not as a firing); the chance level is then
    /// that of all `k` moves drawn at random being optimal.
    pub fn eval_with(&self, lemma: &Lemma, indices: Option<&[usize]>, max_bindings: usize, max_refuted: Option<usize>, all_named: bool) -> Eval {
        let all: Vec<usize> = (0..self.cards.len()).collect();
        let mut ev = Eval::default();
        let mut groups = std::collections::BTreeSet::new();
        let mut groups_held = std::collections::BTreeSet::new();
        let (mut exp_sum, mut sel_sum, mut judged) = (0.0f64, 0.0f64, 0usize);
        for &ci in indices.unwrap_or(&all) {
            let card = &self.cards[ci];
            if card.moves.is_empty() { continue; }
            let (bs, cut) = bindings_fused(&lemma.body, &Binding::new(), &self.facts[ci], max_bindings, &mut ev.steps, STEP_FUSE);
            ev.truncated |= cut;
            if cut { break; }
            if bs.is_empty() { continue; }
            if all_named && !card.cut {
                let b = card.moves.iter().map(|m| m.value).max().expect("a card has moves");
                if card.moves.iter().all(|m| m.value == b) { ev.trivial += 1; continue; }
            }
            ev.fires += 1;
            groups.insert(card.group.clone());
            if card.cut { ev.unknown += 1; continue; }
            let best = card.moves.iter().map(|m| m.value).max().expect("a card has moves");
            let optimal = card.moves.iter().filter(|m| m.value == best).count();
            let mut matched = vec![false; card.moves.len()];
            for b in &bs {
                for (mi, _) in card.moves.iter().enumerate() {
                    if matched[mi] { continue; }
                    if response_matches(lemma, b, &self.removed[ci][mi], &self.added[ci][mi], &mut ev.steps) { matched[mi] = true; }
                }
            }
            let k = matched.iter().filter(|x| **x).count();
            let holds = if all_named { k > 0 && card.moves.iter().zip(&matched).all(|(m, hit)| !*hit || m.value == best) } else { card.moves.iter().zip(&matched).any(|(m, hit)| *hit && m.value == best) };
            if holds { ev.holds += 1; groups_held.insert(card.group.clone()); } else { ev.refuted += 1; }
            if max_refuted.is_some_and(|m| ev.refuted > m) { ev.stopped = true; judged += 1; ev.cases.push((ci, holds, matched.iter().filter(|x| **x).count())); exp_sum += 0.0; break; }
            ev.cases.push((ci, holds, k));
            exp_sum += if all_named { chance_all(card.moves.len(), optimal, k) } else { chance(card.moves.len(), optimal, k) };
            sel_sum += k as f64 / card.moves.len() as f64;
            judged += 1;
        }
        ev.groups = groups.len();
        ev.groups_held = groups_held.len();
        if judged > 0 {
            ev.precision = ev.holds as f64 / judged as f64;
            ev.expected = exp_sum / judged as f64;
            ev.excess = ev.precision - ev.expected;
            ev.selectivity = sel_sum / judged as f64;
        }
        ev
    }
}

/// The chance that `k` moves drawn at random from `n`, of which `o` are optimal, include an optimal one.
fn chance(n: usize, o: usize, k: usize) -> f64 {
    if k == 0 { return 0.0; }
    if n < o + k { return 1.0; }
    let mut miss = 1.0f64;
    for i in 0..k { miss *= (n - o - i) as f64 / (n - i) as f64; }
    1.0 - miss
}

/// The chance that all `k` moves drawn at random from `n`, of which `o` are optimal, are optimal.
fn chance_all(n: usize, o: usize, k: usize) -> f64 {
    if k == 0 { return 0.0; }
    if k > o { return 0.0; }
    let mut p = 1.0f64;
    for i in 0..k { p *= (o - i) as f64 / (n - i) as f64; }
    p
}

/// Does the move (its removed and added facts, indexed) satisfy the lemma's response under the body's binding? Every removed literal matches a removed fact and
/// every added literal an added fact, with the variables consistent (the response's free variables take any value).
pub fn response_matches(lemma: &Lemma, b: &Binding, removed: &FactIndex, added: &FactIndex, steps: &mut u64) -> bool {
    let (rs, _) = bindings(&lemma.response.removed, b, removed, 64, steps);
    if lemma.response.added.is_empty() { return !rs.is_empty() || lemma.response.removed.is_empty(); }
    for rb in rs {
        let (ab, _) = bindings(&lemma.response.added, &rb, added, 1, steps);
        if !ab.is_empty() { return true; }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lemma::{Lit, Response};

    fn mv(name: &str, rem: &[Fact], add: &[Fact], value: i64) -> CardMove { CardMove { name: name.into(), removed: rem.to_vec(), added: add.to_vec(), value, lost: 0, gained: 0, loss_removed: vec![], loss_added: vec![] } }
    fn f(rel: &str, a: &[&str]) -> Fact { Fact::new(rel, a) }

    /// A planted rule: when an enemy threatens a piece (`threat`), moving that piece away is among the best; elsewhere it is not.
    fn corpus() -> Vec<Card> {
        let mut cards = vec![];
        for i in 0..12 {
            let threatened = i % 2 == 0;
            let mut facts = vec![f("piece", &["1", "red"]), f("piece", &["3", "blue"])];
            if threatened { facts.push(f("threat", &["blue", "3", "1"])); }
            let away = mv("away", &[f("piece", &["1", "red"])], &[f("piece", &["0", "red"])], if threatened { 0 } else { -1000 });
            let stay = mv("other", &[f("piece", &["3", "blue"])], &[f("piece", &["4", "blue"])], if threatened { -1000 } else { 0 });
            cards.push(Card { id: format!("c{i}"), group: format!("g{}", i / 2), facts, moves: vec![away, stay], depth: 2, cut: false });
        }
        cards
    }
    fn lemma(neg_free: bool) -> Lemma {
        let _ = neg_free;
        Lemma { name: "away".into(), body: vec![Lit::pos("piece", &["?x", "?s"]), Lit::pos("threat", &["?o", "?y", "?x"])],
                response: Response { removed: vec![Lit::pos("piece", &["?x", "?s"])], added: vec![Lit::pos("piece", &["?z", "?s"])] }, stats: Value::Null, meta: Value::Null }
    }

    #[test]
    fn the_planted_rule_holds_where_it_fires_and_beats_chance() {
        let set = CardSet::new(corpus());
        let ev = set.eval(&lemma(true), None, 100);
        assert_eq!(ev.fires, 6);
        assert_eq!((ev.holds, ev.refuted), (6, 0));
        assert_eq!(ev.groups, 6);
        assert!((ev.precision - 1.0).abs() < 1e-9);
        assert!((ev.expected - 0.5).abs() < 1e-9, "one of two moves is optimal: a random single move is right half the time ({})", ev.expected);
        assert!(ev.excess > 0.4);
        assert!((ev.selectivity - 0.5).abs() < 1e-9);
    }

    #[test]
    fn a_lemma_that_says_the_wrong_response_is_refuted() {
        let set = CardSet::new(corpus());
        let mut wrong = lemma(true);
        wrong.response = Response { removed: vec![Lit::pos("piece", &["?y", "?o"])], added: vec![Lit::pos("piece", &["?w", "?o"])] };     // "the enemy moves": never optimal here
        wrong.body = vec![Lit::pos("threat", &["?o", "?y", "?x"])];
        let ev = set.eval(&wrong, None, 100);
        assert_eq!((ev.fires, ev.holds, ev.refuted), (6, 0, 6));
        assert!(ev.excess < 0.0 + 1e-9);
    }

    #[test]
    fn a_cut_card_is_unknown_not_a_refutation() {
        let mut cards = corpus();
        cards[0].cut = true;
        let ev = CardSet::new(cards).eval(&lemma(true), None, 100);
        assert_eq!((ev.fires, ev.unknown, ev.holds, ev.refuted), (6, 1, 5, 0));
    }

    #[test]
    fn a_body_that_matches_everywhere_is_trivial_by_its_excess() {
        // "if a piece exists, a move of that piece away is among the best": true on half the cards by the table; chance is the same
        let set = CardSet::new(corpus());
        let mut broad = lemma(true);
        broad.body = vec![Lit::pos("piece", &["?x", "?s"])];
        let ev = set.eval(&broad, None, 100);
        assert_eq!(ev.fires, 12);
        assert!(ev.excess < 0.2, "a body that says nothing earns no excess over chance ({})", ev.excess);
    }

    #[test]
    fn cards_round_trip_through_json() {
        for c in corpus() { assert_eq!(Card::from_json(&c.to_json()).unwrap(), c); }
    }
}
