//! PLAN 78.2 / 78.5 / 78.12 THE DECISION LAYER: one move within the clock, from the `Game` trait alone. Since 78.12 it is a CONTROLLER over OPERATORS (cs_operator), and
//! the order it has always played by is the first controller, `Fixed`: (1) a PROVED goal -- tactics `forcing` for us (the first move of a forced line of the game's own
//! outcome is played); (2) THE LEDGER SEARCH (tactics `ledger`, plan 78.5): the move whose line loses least of ours and wins most of theirs, by the judge -- the variable:
//! no values by default (a unit lost is a unit lost, ours first), a valuation when one is plugged in; the minimax reads the game's outcome too; (3) the DENSITY PRIOR
//! (plan 78.7, planning) among the moves the ledger holds equal, when asked. The clock is split: a share for the proof, the rest for the ledger search (named shares, not
//! tuned numbers). Every step is an `Operator` call that says its goal, its evidence and its measured cost; `drive` records them. A trained router or a mixture of experts
//! is another `Controller`, and plugs in here without touching the operators.
pub mod batch;
pub mod cards;
pub mod claim_op;
pub mod graph;
pub mod graph_net;
pub mod graph_prior;
pub mod lemma_prior;

pub use lemma_prior::{LemmaPrior, RulesFirst};
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
pub const WDL_SHARE: f64 = 0.15;           // of the clock: the first estimate of win / draw / loss (plan 78.26: playouts pooled with the results table)
pub const WDL_EPS: f64 = 0.1;              // the playouts' policy plays a uniform move this often
pub const WDL_SEED: u64 = 78;              // the playouts' seed (a decision is reproducible on its own)
pub const CLAIM_SHARE: f64 = 0.3;           // of the clock: what the search toward claims may take (plan 78.36; a named share, not tuned)
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
    /// Plan 78.26: the probability of win / draw / loss decides the PATH. After the proof did not decide, the operator "wdl" estimates the position; a sharp (or unknown)
    /// reading keeps reasoning (the reasoning chain, then the ledger); a quiet one reads the ledger's ties and lets the probabilities pick among the moves it leaves
    /// equal. Off: the decision and its trace are exactly what they were.
    pub wdl: bool,
    /// Plan 78.26: the results table the estimate pools with its playouts (shared across positions; None: playouts only).
    pub wdl_table: Option<std::sync::Arc<cs_planning::wdl::Table>>,
    /// Plan 78.36: run the search toward claims (operator "claim") when the snapshot holds claim rules: after the proof did not decide and before the other steps. A forced line to a
    /// claimed win is played; otherwise the ledger reads the moves of the best claimed class first and the moves a claim refutes last. Off, or with no claim in the snapshot: the
    /// decision and its trace are exactly what they were.
    pub claims: bool,
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
    /// PLAN 78.85: the equal reading asks each candidate again without the search's pruning (`Search::verify_ties`). THE DEFAULT IS SET IN `Params::default` below.
    pub verify_ties: bool,
    /// PLAN 78.92: the chosen move is asked again without the two prunings (`Search::verified_choice`). THE DEFAULT (false) IS SET IN `Params::default` below.
    pub verify_choice: bool,
    /// PLAN 78.93: where the verification of the choice takes its time (`cs_tactics::ledger::ChoiceMode`); Slot (the default, 78.92's form) in `Params::default`. Only read when `verify_choice` is on.
    pub choice_mode: cs_tactics::ledger::ChoiceMode,
    /// PLAN 79.02: the equal reading asks a move its null window held once more without the two prunings, in the deepening's order (`cs_tactics::ledger::Search::reask`). THE DEFAULT (false, the equal reading as it was) IS SET IN `Params::default` below.
    pub reask: bool,
    /// PLAN 79.03: no late-move reduction at the root (`cs_tactics::ledger::Search::root_full`). THE DEFAULT (false, the search as it was) IS SET IN `Params::default` below.
    pub root_full: bool,
    /// PLAN 78.92: the two forward prunings of the ledger search apart (`Search::null_move`, `Search::late_reduce`); both read `prune` too. THE DEFAULT (true, the search as it is) IS SET IN `Params::default` below.
    pub null_move: bool,
    pub late_reduce: bool,
    /// The deepest forced line asked, and the deepest ledger reading (named caps; the clock is the real bound).
    pub max_depth: u32,
    pub ledger_depth: i32,
    /// Plan 78.51: a caller's order of the root moves for the ledger search (None: the search as it was).
    pub root_order: Option<cs_tactics::ledger::RootOrder>,
}

impl Default for Params {
    fn default() -> Params {
        Params { prior: false, ties: false, reason: false, wdl: false, wdl_table: None, claims: true, history: vec![], prior_depth: cs_planning::density::DEPTH, ledger_share: LEDGER_SHARE, prior_defects: 1, ordering: true, prune: true, extend: true, verify_ties: false, verify_choice: false, choice_mode: cs_tactics::ledger::ChoiceMode::Slot, reask: false, root_full: false, null_move: true, late_reduce: true,
                 max_depth: MAX_DEPTH, ledger_depth: LEDGER_DEPTH, root_order: None }
    }
}

/// What the first win / draw / loss estimate says of the position.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WdlMode { Sharp, Quiet, Unknown }

/// The mode an estimate's outcome reads as (an estimate with no sharpness fact, or one that did not run, is unknown).
pub fn wdl_mode<M>(o: &cs_operator::Outcome<M>) -> WdlMode {
    match o.facts.iter().find(|f| f.relation == "sharpness").and_then(|f| f.args.get(1)).map(|w| w.as_str()) {
        Some("sharp") => WdlMode::Sharp,
        Some("quiet") => WdlMode::Quiet,
        _ => WdlMode::Unknown,
    }
}

/// The trace line that states the mode and why, from the first estimate's facts.
fn wdl_mode_line<M>(o: &cs_operator::Outcome<M>) -> String {
    use cs_planning::wdl::{MIN_MASS, SHARP};
    let fact = |name: &str| o.facts.iter().find(|f| f.relation == name);
    let sharpness: f64 = fact("sharpness").and_then(|f| f.args.first()).and_then(|v| v.parse().ok()).unwrap_or(0.0);
    let mass: f64 = fact("wdl").and_then(|f| f.args.get(3)).and_then(|v| v.parse().ok()).unwrap_or(0.0);
    match wdl_mode(o) {
        WdlMode::Sharp => format!("wdl: sharp (decisive mass {sharpness:.2} >= {SHARP:.2}): keep reasoning"),
        WdlMode::Quiet => format!("wdl: quiet (decisive mass {sharpness:.2} < {SHARP:.2}): the probabilities pick among the moves the reading leaves equal"),
        WdlMode::Unknown => format!("wdl: unknown (mass {mass:.1} < {MIN_MASS:.1}): keep reasoning"),
    }
}

/// THE FIXED CONTROLLER: proof, then ledger, then (if asked) the density prior over the ledger's ties. Its choice at each step reads only the steps so far.
pub struct Fixed<'p> {
    pub params: &'p Params,
    pub lemmas: bool,
    /// A claim operator is present (plan 78.36): the snapshot holds claim rules and the switch is on.
    pub claims: bool,
}

impl<'p> Fixed<'p> {
    /// PLAN 78.36 THE CLAIM STEP'S READING OF THE ROOT: (the ledger's root order, the moves it reads LAST), or None where it changes nothing (no claim step, no classes, every root
    /// move refuted). The classes of the root moves are W (a win by claim or game), D, U (unknown) and L (refuted). The response rules' candidates (a suggestion rule's moves are kept
    /// apart: `suggested` facts) and the reasoning's are read as the case says:
    ///   any W            [response candidates that are W] + [suggested moves that are W] + [the other W moves: those leading to a position NOT already in the game's record first,
    ///                    then the nearest win]; every move that is not W is avoided (and is not played while a W move has an equal ledger value);
    ///   no W, some U     [response candidates minus L] + [reasoning's minus L]; L avoided; a draw has no preference over an unknown;
    ///   every move D or L [response candidates that are D] + [suggested that are D] + [the other D moves]; L avoided;
    ///   every move L     nothing changes.
    fn claim_plan<G: Game>(&self, ctx: &Context<G>) -> Option<(Vec<usize>, Vec<usize>)> {
        let step = ctx.steps.iter().find(|s| s.operator == "claim")?;
        let cls = crate::claim_op::root_classes_of(&step.outcome);
        if cls.is_empty() || cls.iter().all(|c| c.class == 'L') { return None; }
        let lemma = ctx.steps.iter().find(|s| s.operator == "lemma");
        let cands: Vec<usize> = lemma.map(|s| s.outcome.candidates.clone()).unwrap_or_default();
        let suggested: Vec<usize> = lemma.map(|s| s.outcome.facts.iter().filter(|f| f.relation == "suggested").filter_map(|f| f.args.get(1).and_then(|a| a.parse().ok())).collect()).unwrap_or_default();
        let reasoned: Vec<usize> = ctx.steps.iter().find(|s| s.operator == "reason").map(|s| s.outcome.candidates.clone()).unwrap_or_default();
        let of = |ch: char| -> Vec<&crate::claim_op::RootClass> { cls.iter().filter(|c| c.class == ch).collect() };
        let is = |ch: char, i: usize| cls.iter().any(|c| c.index == i && c.class == ch);
        let mut prefer: Vec<usize> = vec![];
        let push = |v: &mut Vec<usize>, i: usize| { if !v.contains(&i) { v.push(i); } };
        if !of('W').is_empty() {
            for i in &cands { if is('W', *i) { push(&mut prefer, *i); } }
            for i in &suggested { if is('W', *i) { push(&mut prefer, *i); } }
            let history: std::collections::HashSet<u64> = self.params.history.iter().copied().collect();
            let moves = ctx.g.moves(ctx.s);
            let mut rest: Vec<(bool, i32, usize)> = of('W').iter().map(|c| (history.contains(&ctx.g.key(&ctx.g.apply(ctx.s, &moves[c.index]))), c.value, c.index)).collect();
            rest.sort_by_key(|(seen, val, i)| (*seen, -*val, *i));
            for (_, _, i) in rest { push(&mut prefer, i); }
            let avoid = cls.iter().filter(|c| c.class != 'W').map(|c| c.index).collect();
            Some((prefer, avoid))
        } else if !of('U').is_empty() {
            for i in cands.iter().chain(reasoned.iter()) { if !is('L', *i) { push(&mut prefer, *i); } }
            Some((prefer, of('L').iter().map(|c| c.index).collect()))
        } else {
            for i in &cands { if is('D', *i) { push(&mut prefer, *i); } }
            for i in &suggested { if is('D', *i) { push(&mut prefer, *i); } }
            for c in of('D') { push(&mut prefer, c.index); }
            Some((prefer, of('L').iter().map(|c| c.index).collect()))
        }
    }

    /// The moves the claim step has the ledger read last (none without a claim step).
    fn refuted<G: Game>(&self, ctx: &Context<G>) -> Vec<usize> { self.claim_plan(ctx).map(|p| p.1).unwrap_or_default() }

    /// `among` without the avoided moves (kept whole when that would leave none): the prior or the estimate picks among the moves no claim refutes.
    fn unrefuted<G: Game>(&self, ctx: &Context<G>, among: &[usize]) -> Vec<usize> {
        let r = self.refuted(ctx);
        let kept: Vec<usize> = among.iter().copied().filter(|i| !r.contains(i)).collect();
        if kept.is_empty() { among.to_vec() } else { kept }
    }

    /// What follows a proof that did not decide (and, with claims, the claim search that did not either): the estimate, the reasoning or the ledger.
    fn after_proof<G: Game>(&self, ctx: &Context<G>, me: cs_game::Role) -> Action<G::Move> {
        let p = self.params;
        if p.wdl { Action::Run { operator: "wdl".into(), goal: Goal::Estimate { role: me }, seconds: ctx.seconds * WDL_SHARE, max_nodes: NODE_FUSE, max_depth: None } }
        else if p.reason { Action::Run { operator: "reason".into(), goal: Goal::Reason { role: me }, seconds: ctx.seconds * REASON_SHARE, max_nodes: cs_tactics::reason::REASON_FUSE, max_depth: None } }
        else { self.ledger(ctx, me, false) }
    }

    /// The ledger step: the lemmas' candidates are its root order, then the reasoning's (no repeats) -- an order, never a verdict.
    fn ledger<G: Game>(&self, ctx: &Context<G>, me: cs_game::Role, force_ties: bool) -> Action<G::Move> {
        let p = self.params;
        let read_ties = p.prior || p.ties || force_ties;
        let left = (ctx.seconds * if read_ties { p.ledger_share } else { 1.0 } - ctx.elapsed()).max(0.01);
        // PLAN 78.36: the claim step's reading of the root (see `claim_plan`); without one the order is the lemmas', then the reasoning's, as it always was.
        let (prefer, refuted) = match self.claim_plan(ctx) {
            Some(p) => p,
            None => {
                let mut prefer: Vec<usize> = ctx.steps.iter().find(|s| s.operator == "lemma").map(|s| s.outcome.candidates.clone()).unwrap_or_default();
                for i in ctx.steps.iter().find(|s| s.operator == "reason").map(|s| s.outcome.candidates.clone()).unwrap_or_default() { if !prefer.contains(&i) { prefer.push(i); } }
                (prefer, vec![])
            }
        };
        Action::Run { operator: "ledger".into(), goal: Goal::Ledger { role: me, ties: if read_ties { Some(TIE_SHARE) } else { None }, prefer, avoid: refuted }, seconds: left,
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
                // (plan 78.36: with claim rules in the snapshot, the search toward claims runs here, before the other steps)
                _ if self.claims => Action::Run { operator: "claim".into(), goal: Goal::Toward { role: me }, seconds: ctx.seconds * CLAIM_SHARE, max_nodes: NODE_FUSE, max_depth: None },
                _ => self.after_proof(ctx, me),
            },
            "claim" => match (last.outcome.status, &last.outcome.mv) {
                (Status::Proved, Some(first)) => Action::Play { mv: first.clone(), proved: true },          // (the game's own win only: a claimed win is a class the ledger reads first)
                _ => self.after_proof(ctx, me),
            },
            "wdl" => match last.goal {
                Goal::Prefer { .. } => {
                    let ledger_mv = ctx.steps.iter().rev().find(|s| s.operator == "ledger").and_then(|s| s.outcome.mv.clone());
                    match last.outcome.mv.clone().or(ledger_mv) { Some(mv) => Action::Play { mv, proved: false }, None => Action::Stop }
                }
                _ => match wdl_mode(&last.outcome) {
                    // sharp or unknown: keep reasoning (the wdl candidates are not the ledger's prior here)
                    WdlMode::Quiet => self.ledger(ctx, me, true),
                    _ => Action::Run { operator: "reason".into(), goal: Goal::Reason { role: me }, seconds: ctx.seconds * REASON_SHARE, max_nodes: cs_tactics::reason::REASON_FUSE, max_depth: None },
                },
            },
            "reason" => self.ledger(ctx, me, false),
            "ledger" => {
                let Some(mv) = last.outcome.mv.clone() else {
                    return match ctx.g.moves(ctx.s).into_iter().next() { Some(m) => Action::Play { mv: m, proved: false }, None => Action::Stop };
                };
                let decided = last.outcome.value.map_or(true, |v| v.abs() >= WIN - 1000);
                let quiet = p.wdl && ctx.steps.iter().find(|s| s.operator == "wdl").map_or(false, |s| wdl_mode(&s.outcome) == WdlMode::Quiet);
                if quiet && !decided && last.outcome.candidates.len() > 1 {
                    return Action::Run { operator: "wdl".into(), goal: Goal::Prefer { role: me, among: self.unrefuted(ctx, &last.outcome.candidates), defects: 0 }, seconds: ctx.remaining(), max_nodes: NODE_FUSE, max_depth: None };
                }
                if p.prior && !decided && last.outcome.candidates.len() > 1 {
                    let fuse = cs_planning::density::FUSE * (1u64 << (2 * (p.prior_depth - cs_planning::density::DEPTH).max(0)));
                    return Action::Run { operator: "density".into(), goal: Goal::Prefer { role: me, among: self.unrefuted(ctx, &last.outcome.candidates), defects: p.prior_defects },
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
    decide_with_store(g, s, seconds, judge, params, lemma, None)
}

/// `decide_with_lemmas` with the snapshot's CLAIMS (plan 78.36): when `claims` holds at least one rule and `params.claims` is on, the search toward claims runs after the proof did not
/// decide. With None, with no claim rule or with the switch off, this is `decide_with_lemmas` exactly: no extra step, the same trace.
pub fn decide_with_store<G: Game, J: Judge>(g: &G, s: &G::State, seconds: f64, judge: &J, params: &Params, lemma: Option<&dyn Operator<G>>,
                                            claims: Option<&cs_memory::claim::Claims>) -> Option<Decision<G::Move>> {
    let me = g.mover(s)?;
    let n = g.moves(s).len();
    if n == 0 { return None; }
    let trace = vec![format!("decide: {} to move, {} legal moves, {:.2} s", g.roles()[me], n, seconds)];
    let forcing = ForcingOp;
    let ledger = LedgerOp { judge, history: params.history.clone(), ordering: params.ordering, prune: params.prune, extend: params.extend, verify_ties: params.verify_ties, verify_choice: params.verify_choice, choice_mode: params.choice_mode, reask: params.reask, root_full: params.root_full, null_move: params.null_move, late_reduce: params.late_reduce, root_order: params.root_order.clone() };
    let density = DensityOp { judge };
    let reason = ReasonOp { judge };
    let mut ops: Vec<&dyn Operator<G>> = vec![&forcing, &ledger, &density, &reason];
    // (plan 78.26: the win / draw / loss operator: the decision's judge plays the careful playouts (a move's gain less what it leaves to be taken) and values a cut one,
    // at a scale in units of the smallest worth)
    let mut wdl_op = cs_planning::wdl::WdlOp::new(cs_planning::mcts::EXPLORATION, cs_planning::wdl::WDL_PLAYOUT, cs_planning::rollout::Careful::new(judge, WDL_EPS, WDL_SEED),
        cs_planning::rollout::Balance::new(judge, cs_planning::wdl::leaf_scale(g, s, judge)), params.wdl_table.clone());
    wdl_op.leaf_note = format!("scale {} x the smallest positive worth at the root, so the squash does not depend on the judge's units", cs_planning::wdl::WDL_SCALE);
    if params.wdl { ops.push(&wdl_op); }
    if let Some(l) = lemma { ops.push(l); }
    let claim_op = claims.filter(|c| params.claims && !c.is_empty()).map(|c| claim_op::ClaimOp { claims: c, history: params.history.clone() });
    if let Some(c) = &claim_op { ops.push(c); }
    let mut d = drive(g, s, seconds, &ops, &Fixed { params, lemmas: lemma.is_some(), claims: claim_op.is_some() }, trace);
    if params.wdl {
        // the mode line sits right after the first estimate's own lines
        if let Some(k) = d.steps.iter().position(|st| st.operator == "wdl" && matches!(st.goal, Goal::Estimate { .. })) {
            let at = 1 + d.steps[..=k].iter().map(|st| st.outcome.evidence.len()).sum::<usize>();
            let line = wdl_mode_line(&d.steps[k].outcome);
            d.trace.insert(at.min(d.trace.len()), line);
        }
    }
    if d.mv.is_some() { Some(d) } else { None }
}
