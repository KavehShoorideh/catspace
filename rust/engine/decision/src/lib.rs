//! PLAN 78.2 / 78.5 / 78.12 THE DECISION LAYER: one move within the clock, from the `Game` trait alone. Since 78.12 it is a CONTROLLER over OPERATORS (cs_operator), and
//! the order it has always played by is the first controller, `Fixed`: (1) a PROVED goal -- tactics `forcing` for us (the first move of a forced line of the game's own
//! outcome is played); (2) THE LEDGER SEARCH (tactics `ledger`, plan 78.5): the move whose line loses least of ours and wins most of theirs, by the judge -- the variable:
//! no values by default (a unit lost is a unit lost, ours first), a valuation when one is plugged in; the minimax reads the game's outcome too; (3) the DENSITY PRIOR
//! (plan 78.7, planning) among the moves the ledger holds equal, when asked. The clock is split: a share for the proof, the rest for the ledger search (named shares, not
//! tuned numbers). Every step is an `Operator` call that says its goal, its evidence and its measured cost; `drive` records them. A trained router or a mixture of experts
//! is another `Controller`, and plugs in here without touching the operators.
pub mod batch;
pub mod cards;
pub mod graph;
pub mod graph_net;
pub mod graph_prior;
pub mod lemma_prior;

pub use lemma_prior::LemmaPrior;
pub use graph_prior::GraphPrior;

use cs_game::Game;
use cs_operator::{drive, Action, Context, Controller, Goal, Mode, Operator, Status};
use cs_planning::operators::DensityOp;
use cs_tactics::ledger::{Judge, WIN};
use cs_tactics::operators::{ForcingOp, LedgerOp, ReasonOp};

pub use cs_operator::Decision;

pub const PROOF_SHARE: f64 = 0.2;          // the share of the clock the proof of the game's own outcome may take (the ledger search reads the outcome too)
pub const MAX_DEPTH: u32 = 8;              // the deepest forced line asked (a named fuse: the clock is the real bound)
pub const DANGER_DEPTH: u32 = 2;           // their forced goal looked for after each candidate, within this many of their moves
pub const NODE_FUSE: u64 = 50_000_000;

pub struct Choice<M> {
    pub mv: M,
    pub proved: bool,
    pub trace: Vec<String>,
    pub nodes: u64,
}

pub const LEDGER_DEPTH: i32 = 64;
pub const LEDGER_SHARE: f64 = 0.6;
pub const LEMMA_SHARE: f64 = 0.05;         // of the clock: what the lemma operator may take (it binds a few lemmas to one position)
pub const LEMMA_FUSE: u64 = 1_000_000;
pub const REASON_SHARE: f64 = 0.1;         // of the clock: what the reasoning chain may take (plan 78.25: threats, frame, answers; an order and an explanation, never a verdict)
pub const TIE_SHARE: f64 = 0.3;            // of what is left: the share the tie reading may take (the rest is the prior's census)

/// Everything the fixed controller and its operators are asked, with the defaults the engine plays by.
#[derive(Debug, Clone)]
pub struct Params {
    /// The density prior on: ties of the ledger are broken by tactic density.
    pub prior: bool,
    /// Read the ledger's ties (the moves as good as the best) WITHOUT the prior: the ledger keeps `ledger_share` of the clock and the ties the rest; the candidates
    /// are reported, the move is the ledger's. The engine's own measure of how coarse its judge is (the tie rate), and a seam for another tie-breaker.
    pub ties: bool,
    /// Plan 78.25: run the reasoning chain (operator "reason") after the proof did not decide, and let the ledger read its candidates after the lemmas' (an order; the
    /// ledger's value decides). Off: the decision and its trace are exactly what they were.
    pub reason: bool,
    /// The keys of the positions the game's record holds before this one (a position met again is the rules' draw).
    pub history: Vec<u64>,
    /// The census plies of the prior's readings (its node fuse grows with them).
    pub prior_depth: i32,
    /// With the prior on: the share of the clock the ledger search may take.
    pub ledger_share: f64,
    pub prior_defects: u32,
    pub ordering: bool,
    pub prune: bool,
    pub extend: bool,
    /// The deepest forced line asked, and the deepest ledger reading (named caps; the clock is the real bound).
    pub max_depth: u32,
    pub ledger_depth: i32,
}

impl Default for Params {
    fn default() -> Params {
        Params { prior: false, ties: false, reason: false, history: vec![], prior_depth: cs_planning::density::DEPTH, ledger_share: LEDGER_SHARE, prior_defects: 1, ordering: true, prune: true, extend: true,
                 max_depth: MAX_DEPTH, ledger_depth: LEDGER_DEPTH }
    }
}

/// THE FIXED CONTROLLER: proof, then ledger, then (if asked) the density prior over the ledger's ties. Its choice at each step reads only the steps so far.
pub struct Fixed<'p> { pub params: &'p Params, pub lemmas: bool }

impl<'p> Fixed<'p> {
    /// The ledger step: the lemmas' candidates are its root order, then the reasoning's (no repeats) -- an order, never a verdict.
    fn ledger<G: Game>(&self, ctx: &Context<G>, me: cs_game::Role) -> Action<G::Move> {
        let p = self.params;
        let read_ties = p.prior || p.ties;
        let left = (ctx.seconds * if read_ties { p.ledger_share } else { 1.0 } - ctx.elapsed()).max(0.01);
        let mut prefer = ctx.steps.iter().find(|s| s.operator == "lemma").map(|s| s.outcome.candidates.clone()).unwrap_or_default();
        for i in ctx.steps.iter().find(|s| s.operator == "reason").map(|s| s.outcome.candidates.clone()).unwrap_or_default() { if !prefer.contains(&i) { prefer.push(i); } }
        Action::Run { operator: "ledger".into(), goal: Goal::Ledger { role: me, ties: if read_ties { Some(TIE_SHARE) } else { None }, prefer }, seconds: left,
                      max_nodes: cs_tactics::ledger::NODE_FUSE, max_depth: Some(p.ledger_depth as u32) }
    }
}

impl<'p, G: Game> Controller<G> for Fixed<'p> {
    fn choose(&self, ctx: &Context<G>) -> Action<G::Move> {
        let p = self.params;
        let Some(me) = ctx.g.mover(ctx.s) else { return Action::Stop };
        let proof = || Action::Run { operator: "forcing".into(), goal: Goal::Win { role: me, mode: Mode::Forced }, seconds: ctx.seconds * PROOF_SHARE, max_nodes: NODE_FUSE, max_depth: Some(p.max_depth) };
        let Some(last) = ctx.steps.last() else {
            // (the learned lemmas first, when a snapshot's operator is present: a prior for the ledger's root order, read cheaply)
            return if self.lemmas { Action::Run { operator: "lemma".into(), goal: Goal::Respond { role: me }, seconds: ctx.seconds * LEMMA_SHARE, max_nodes: LEMMA_FUSE, max_depth: None } } else { proof() };
        };
        match last.operator.as_str() {
            "lemma" => proof(),
            "forcing" => match (last.outcome.status, &last.outcome.mv) {
                (Status::Proved, Some(first)) => Action::Play { mv: first.clone(), proved: true },
                _ if p.reason => Action::Run { operator: "reason".into(), goal: Goal::Reason { role: me }, seconds: ctx.seconds * REASON_SHARE, max_nodes: cs_tactics::reason::REASON_FUSE, max_depth: None },
                _ => self.ledger(ctx, me),
            },
            "reason" => self.ledger(ctx, me),
            "ledger" => {
                let Some(mv) = last.outcome.mv.clone() else {
                    return match ctx.g.moves(ctx.s).into_iter().next() { Some(m) => Action::Play { mv: m, proved: false }, None => Action::Stop };
                };
                let decided = last.outcome.value.map_or(true, |v| v.abs() >= WIN - 1000);
                if p.prior && !decided && last.outcome.candidates.len() > 1 {
                    let fuse = cs_planning::density::FUSE * (1u64 << (2 * (p.prior_depth - cs_planning::density::DEPTH).max(0)));
                    return Action::Run { operator: "density".into(), goal: Goal::Prefer { role: me, among: last.outcome.candidates.clone(), defects: p.prior_defects },
                                         seconds: ctx.remaining(), max_nodes: fuse, max_depth: Some(p.prior_depth as u32) };
                }
                Action::Play { mv, proved: false }
            }
            "density" => {
                let ledger_mv = ctx.steps.iter().rev().find(|s| s.operator == "ledger").and_then(|s| s.outcome.mv.clone());
                match last.outcome.mv.clone().or(ledger_mv) { Some(mv) => Action::Play { mv, proved: false }, None => Action::Stop }
            }
            _ => Action::Stop,
        }
    }
}

/// The move for the side to move at `s` within `seconds` (None at a terminal state). `judge`: the ledger's judge (the variable: `Count` -- no values -- or `Worth`).
pub fn decide<G: Game, J: Judge>(g: &G, s: &G::State, seconds: f64, judge: &J, prior: bool) -> Option<Choice<G::Move>> {
    decide_in(g, s, seconds, judge, prior, &[], cs_planning::density::DEPTH)
}

/// `decide` within an INSTANCE (plan 78.8): `history` -- the keys of the positions the game's record holds before `s` (a position met again is the rules' draw).
pub fn decide_in<G: Game, J: Judge>(g: &G, s: &G::State, seconds: f64, judge: &J, prior: bool, history: &[u64], prior_depth: i32) -> Option<Choice<G::Move>> {
    decide_params(g, s, seconds, judge, &Params { prior, history: history.to_vec(), prior_depth, ..Params::default() }).map(into_choice)
}

/// `decide_in` with the ledger's share of the clock when the prior is on (`ledger_share`; the prior reads ties with the rest).
#[allow(clippy::too_many_arguments)]
pub fn decide_with<G: Game, J: Judge>(g: &G, s: &G::State, seconds: f64, judge: &J, prior: bool, history: &[u64], prior_depth: i32,
                                      ledger_share: f64, prior_defects: u32, ordering: bool, prune: bool, extend: bool) -> Option<Choice<G::Move>> {
    decide_params(g, s, seconds, judge, &Params { prior, history: history.to_vec(), prior_depth, ledger_share, prior_defects, ordering, prune, extend, ..Params::default() }).map(into_choice)
}

fn into_choice<M>(d: Decision<M>) -> Choice<M> {
    Choice { mv: d.mv.expect("a decision with a move"), proved: d.proved, trace: d.trace, nodes: d.cost.nodes }
}

/// THE DECISION with every step kept: the fixed controller driving the operators. None at a terminal state or when there is no move to play.
pub fn decide_params<G: Game, J: Judge>(g: &G, s: &G::State, seconds: f64, judge: &J, params: &Params) -> Option<Decision<G::Move>> {
    decide_with_lemmas(g, s, seconds, judge, params, None)
}

/// `decide_params` with a lemma operator (a snapshot's lemmas bound to the position): the moves they propose are read first by the ledger search -- an order,
/// never a verdict, so the value found is the value found without them (within the horizon the search reaches).
pub fn decide_with_lemmas<G: Game, J: Judge>(g: &G, s: &G::State, seconds: f64, judge: &J, params: &Params, lemma: Option<&dyn Operator<G>>) -> Option<Decision<G::Move>> {
    let me = g.mover(s)?;
    let n = g.moves(s).len();
    if n == 0 { return None; }
    let trace = vec![format!("decide: {} to move, {} legal moves, {:.2} s", g.roles()[me], n, seconds)];
    let forcing = ForcingOp;
    let ledger = LedgerOp { judge, history: params.history.clone(), ordering: params.ordering, prune: params.prune, extend: params.extend };
    let density = DensityOp { judge };
    let reason = ReasonOp { judge };
    let mut ops: Vec<&dyn Operator<G>> = vec![&forcing, &ledger, &density, &reason];
    if let Some(l) = lemma { ops.push(l); }
    let d = drive(g, s, seconds, &ops, &Fixed { params, lemmas: lemma.is_some() }, trace);
    if d.mv.is_some() { Some(d) } else { None }
}
