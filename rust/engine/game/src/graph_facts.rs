//! PLAN 79.27 PART B (SYSTEM_SPEC 11 "the layers", "Layer 4 takes a goal as an input"; ledger rows 36, 14; piece: rules; plan 79.34: moved up from layer 1, since the fact form of the relations that follow from how things move is layer 2's). THE FACTS OF THE GRAPH AS A PATTERN READS THEM, taken from the live graph as it stands (no graph is made anew): the rows a statement's
//! pattern (layer 1's `Statement`, bound by layer 0's matcher) is matched over when the position is the graph that `LiveGraph` keeps step by step.
//!
//! INTERFACE.
//!   `FactReader::new(board, Some(pattern))`   a reader that makes only the rows some literal of the pattern unifies with (the others cannot take part in any join, so the bindings are the same as over every row); `None`: every row.
//!   `FactReader::facts(&self, &LiveGraph)`    the rows of the graph as it now is, sorted and distinct.
//!   Called by: layer 4's search to a statement (cs_tactics::reach) and this file's test.
//! THE RELATIONS (the words of the relations the rules make are `relations.rs`'s; those of what is seen are layer 1's `cs_rep::fact`; nothing names a game):
//!   `unit(owner kind place)`            a unit stands there
//!   `bear(owner kind place target)`     the unit bears on the target place (a place held by its own side counts: it guards it)
//!   `may_move(owner kind place to)`     the unit may move to the place in one move
//!   `tempo(owner)`                      the side to move
//!   `on_line(place line)`               the place lies on the line (the line's text is its graph node's id); static
//!   `on_edge(place direction)`          the place is on the edge under the direction; static
use crate::relations::{UnitRelations, BEAR as BEAR_REL, MAY_MOVE as MAY_MOVE_REL};
use cs_rep::fact::{Fact, ON_EDGE, ON_LINE, UNIT};
use cs_rep::graph::Board;
use cs_rep::lemma::{Arg, Lit};
use cs_rep::live::LiveGraph;
use std::collections::BTreeSet;

pub const BEAR: &str = BEAR_REL.word;
pub const MAY_MOVE: &str = MAY_MOVE_REL.word;
pub const TEMPO: &str = "tempo";

/// Whether a row (relation, args) unifies with some literal as the pattern's rows are chosen: the same relation and arity and equal constants (a variable used twice is left to the join).
fn fits(lits: &[Lit], relation: &str, args: &[&str]) -> bool {
    lits.iter().any(|l| l.relation == relation && l.args.len() == args.len() && l.args.iter().zip(args).all(|(a, v)| match a { Arg::Const(c) => c == v, Arg::Var(_) => true }))
}

pub struct FactReader { keep: Option<Vec<Lit>>, roles: Vec<String>, kinds: Vec<String>, places: Vec<String>, statics: Vec<Fact> }

impl FactReader {
    pub fn new(board: &Board, pattern: Option<&[Lit]>) -> FactReader {
        let keep = pattern.map(|p| p.to_vec());
        let ok = |rel: &str, args: &[&str]| keep.as_ref().is_none_or(|k| fits(k, rel, args));
        let mut statics = vec![];
        for (li, l) in board.lines.iter().enumerate() {
            let line = board.nodes[board.line_node[li] as usize].id.as_str();
            for p in &l.places { let pn = board.names.places[*p as usize].as_str(); if ok(ON_LINE, &[pn, line]) { statics.push(Fact { relation: ON_LINE.into(), args: vec![pn.into(), line.into()] }); } }
        }
        for (d, dn) in board.directions.iter().enumerate() {
            for (p, pn) in board.names.places.iter().enumerate() { if board.edge_distance[d][p] == 0 && ok(ON_EDGE, &[pn.as_str(), dn.as_str()]) { statics.push(Fact { relation: ON_EDGE.into(), args: vec![pn.clone(), dn.clone()] }); } }
        }
        FactReader { keep, roles: board.names.roles.clone(), kinds: board.names.kinds.clone(), places: board.names.places.clone(), statics }
    }

    fn wants(&self, relation: &str, args: &[&str]) -> bool { self.keep.as_ref().is_none_or(|k| fits(k, relation, args)) }

    /// The rows as they come (not sorted; a row may stand twice): what a join reads.
    pub fn rows(&self, live: &LiveGraph) -> Vec<Fact> {
        let mut out: Vec<Fact> = self.statics.clone();
        if let Some(t) = live.tempo() { let r = self.roles[t].as_str(); if self.wants(TEMPO, &[r]) { out.push(Fact { relation: TEMPO.into(), args: vec![r.into()] }); } }
        for u in live.units() {
            let (o, k, p) = (self.roles[u.owner].as_str(), self.kinds[u.kind as usize].as_str(), self.places[u.at as usize].as_str());
            if self.wants(UNIT, &[o, k, p]) { out.push(Fact { relation: UNIT.into(), args: vec![o.into(), k.into(), p.into()] }); }
            for (rel, list) in [(BEAR, u.bears()), (MAY_MOVE, u.reach())] {
                for &q in list {
                    let qn = self.places[q as usize].as_str();
                    if self.wants(rel, &[o, k, p, qn]) { out.push(Fact { relation: rel.into(), args: vec![o.into(), k.into(), p.into(), qn.into()] }); }
                }
            }
        }
        out
    }

    /// The rows sorted and distinct (the canonical form the memories compare by).
    pub fn facts(&self, live: &LiveGraph) -> Vec<Fact> {
        let set: BTreeSet<Fact> = self.rows(live).into_iter().collect();
        set.into_iter().collect()
    }
}
