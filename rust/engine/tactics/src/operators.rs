//! PLAN 78.12 THE TACTICS LAYER'S OPERATORS: `forcing`, `ledger` and `census` behind the shared interface (cs_operator). Each wraps the function the engine has always
//! called, unchanged; what is new is that the call states its goal and mode, returns its evidence, and reports the cost it measured. Game-free.
use crate::census::census;
use crate::ledger::{Judge, Search, WIN};
use crate::reason::{explain, reason_within, REASON_DEPTH, REASON_FUSE};
use crate::{force_traced, Answer};
use cs_game::Game;
use cs_operator::{Budget, Fact, Goal, Mode, Operator, Outcome, Status};

/// The deepest forced line asked when the budget names none (a named fuse: the clock is the real bound).
pub const FORCING_DEPTH: u32 = 8;
/// The deepest ledger reading asked when the budget names none.
pub const LEDGER_DEPTH: i32 = 64;
/// The census's plies per reading when the budget names none.
pub const CENSUS_DEPTH: i32 = 2;

/// `forcing`: the terminal goal by the every-reply clause, deepened while the budget lasts. Reads the FORCED mode only: a goal asked cooperatively or against a model
/// is `Unsupported`, never answered as if it were forced.
pub struct ForcingOp;

impl<G: Game> Operator<G> for ForcingOp {
    fn name(&self) -> String { "forcing".into() }
    fn accepts(&self, goal: &Goal) -> bool { matches!(goal, Goal::Win { mode: Mode::Forced, .. }) }
    fn run(&self, g: &G, s: &G::State, goal: &Goal, mut b: Budget) -> Outcome<G::Move> {
        let Goal::Win { role, mode: Mode::Forced } = goal else { return Outcome::unsupported("forcing", goal) };
        let t0 = std::time::Instant::now();
        let (ans, reads) = force_traced(g, s, *role, b.max_depth.unwrap_or(FORCING_DEPTH), &mut b);
        let mut o = match ans {
            Answer::Proved { first, depth } => {
                let mut o = Outcome::new(Status::Proved);
                o.evidence.push(format!("tactics forcing: PROVED -- {} forces the goal within {depth} move(s) against every reply ({} nodes)", g.move_name(s, &first), b.nodes));
                o.mv = Some(first);
                o.depth = Some(depth as i64);
                o
            }
            Answer::None { depth } => {
                let mut o = Outcome::new(Status::NoLine);
                o.evidence.push(format!("tactics forcing: no forced goal within {depth} move(s) ({} nodes)", b.nodes));
                o.depth = Some(depth as i64);
                o
            }
            Answer::Gap { depth } => {
                let mut o = Outcome::new(Status::Gap);
                o.evidence.push(format!("tactics forcing: GAP at depth {depth} -- the clock cut the reading ({} nodes)", b.nodes));
                o.depth = Some(depth as i64);
                o
            }
        };
        for r in &reads { o.part(format!("depth {}", r.depth), r.nodes, r.seconds); }
        o.cost.nodes = b.nodes;
        o.cost.seconds = t0.elapsed().as_secs_f64();
        o
    }
}

/// `ledger`: the move whose line loses least of ours and wins most of theirs, by the judge (the variable). With `ties` asked it goes on to read the moves as good as
/// the best, with that share of what is left of the decision's clock (the table of the run warms it) -- unless the line already decides the game.
pub struct LedgerOp<'a, J: Judge> {
    pub judge: &'a J,
    /// The keys of the positions the game's record holds before this one (instance knowledge: a position met again is the rules' draw).
    pub history: Vec<u64>,
    pub ordering: bool,
    pub prune: bool,
    pub extend: bool,
    /// Plan 78.85: the second reading of the equal reading (`Search::verify_ties`).
    pub verify_ties: bool,
    /// Plan 78.92: the chosen move asked again without the two prunings (`Search::verified_choice`); false is the search as it is.
    pub verify_choice: bool,
    /// Plan 78.93: where the verification takes its time (`ChoiceMode`); Slot is 78.92's form and the default.
    pub choice_mode: crate::ledger::ChoiceMode,
    /// Plan 79.02: the equal reading asks a held move again without the two prunings, in the deepening's order (`Search::reask`); false is the equal reading as it was.
    pub reask: bool,
    /// Plan 78.92: the two prunings apart (`Search::null_move`, `Search::late_reduce`); true is the search as it is.
    pub null_move: bool,
    pub late_reduce: bool,
    /// Plan 78.51: a caller's order of the root moves; None (the default) is the search as it was.
    pub root_order: Option<crate::ledger::RootOrder>,
}

impl<'a, J: Judge> LedgerOp<'a, J> {
    pub fn new(judge: &'a J) -> Self { LedgerOp { judge, history: vec![], ordering: true, prune: true, extend: true, verify_ties: false, verify_choice: false, choice_mode: crate::ledger::ChoiceMode::Slot, reask: false, null_move: true, late_reduce: true, root_order: None } }
}

impl<'a, G: Game, J: Judge> Operator<G> for LedgerOp<'a, J> {
    fn name(&self) -> String { "ledger".into() }
    fn accepts(&self, goal: &Goal) -> bool { matches!(goal, Goal::Ledger { .. }) }
    fn run(&self, g: &G, s: &G::State, goal: &Goal, b: Budget) -> Outcome<G::Move> {
        let Goal::Ledger { role, ties, prefer, avoid } = goal else { return Outcome::unsupported("ledger", goal) };
        let t0 = std::time::Instant::now();
        let main_seconds = b.seconds_left();
        let mut ls = Search::new(g, self.judge, *role, self.choice_mode.deepening_seconds(self.verify_choice, main_seconds));
        ls.set_fuse(b.max_nodes);
        ls.history = self.history.iter().copied().collect();
        ls.ordering = self.ordering;
        ls.prune = self.prune;
        ls.extend = self.extend;
        ls.verify_ties = self.verify_ties;
        ls.verify_choice = self.verify_choice;
        ls.choice_mode = self.choice_mode;
        ls.reask = self.reask;
        ls.null_move = self.null_move;
        ls.late_reduce = self.late_reduce;
        ls.root_first = prefer.clone();
        ls.root_last = avoid.clone();
        if let Some(ro) = &self.root_order { ls.set_root_order(&ro.order, ro.strict); }
        let ln = ls.run(s, b.max_depth.map_or(LEDGER_DEPTH, |d| d as i32));
        let moves = g.moves(s);
        let Some(i) = ln.first_index else {
            let mut o = Outcome::new(Status::Gap);
            o.evidence.push("ledger search: no line read within the clock -- the first legal move".into());
            for r in &ls.reads { o.part(format!("depth {}{}", r.depth, if r.complete { "" } else { " (cut)" }), r.nodes, r.seconds); }
            o.cost.nodes = ln.nodes;
            o.cost.seconds = t0.elapsed().as_secs_f64();
            return o;
        };
        let mut o = Outcome::new(Status::Read);
        let mut i = i;
        let mut replaced = false;
        if self.verify_choice {
            ls.resume(self.choice_mode.verification_seconds(main_seconds, t0.elapsed().as_secs_f64(), b.rest_of_decision(), ties.unwrap_or(1.0)));
            let (n0, t1) = (ls.nodes, std::time::Instant::now());
            let v = ls.verified_choice(s, ln.depth, ln.value, i);
            o.part(format!("verify_choice ({})", v.state.label()), ls.nodes - n0, t1.elapsed().as_secs_f64());
            replaced = v.index != i;
            i = v.index;
        }
        o.mv = Some(moves[i].clone());
        o.value = Some(ln.value);
        o.depth = Some(ln.depth as i64);
        o.line = if replaced { vec![g.move_name(s, &moves[i])] } else { ln.line.clone() };
        o.events = if replaced { vec![] } else { ln.lost.clone() };
        for r in &ls.reads { o.part(format!("depth {}{}", r.depth, if r.complete { "" } else { " (cut)" }), r.nodes, r.seconds); }
        let who = |r: cs_game::Role| if r == *role { "we" } else { "they" };
        let lost: Vec<String> = ln.lost.iter().map(|(owner, k)| format!("{} {k}", who(*owner))).collect();
        o.evidence.push(format!("ledger search [{}]: {} at depth {} ({} nodes): the line {} -- {}; value {}", self.judge.name(), g.move_name(s, &moves[i]), ln.depth, ln.nodes,
                                ln.line.join(" "), if lost.is_empty() { "nobody loses a unit".to_string() } else { lost.join(", ") }, ln.value));
        if !avoid.is_empty() { o.evidence.push(format!("ledger search: {} root move(s) the claim search puts last (refuted, or not a win where a win is claimed) are read last (a move of equal value that is not among them is kept)", avoid.len())); }
        let mut nodes = ln.nodes;
        if let Some(share) = ties {
            if ln.value.abs() < WIN - 1000 {
                ls.resume(b.rest_of_decision() * share);
                let n0 = ls.nodes;
                let t1 = std::time::Instant::now();
                ls.reask_skip = Some(i);
                o.candidates = ls.ties(s, ln.depth, ln.value);
                if !avoid.is_empty() { o.candidates.sort_by_key(|i| avoid.contains(i)); }          // (plan 78.36: the avoided moves are named last)
                nodes += ls.nodes - n0;
                o.part("ties", ls.nodes - n0, t1.elapsed().as_secs_f64());
            }
        }
        o.cost.nodes = nodes;
        o.cost.seconds = t0.elapsed().as_secs_f64();
        o
    }
}

/// `census`: the side to move's tactics down to `max_defects` defects, each a fact (`tactic`: defects, targets, line). The threat ranker's reading.
pub struct CensusOp<'a, J: Judge> { pub judge: &'a J }

impl<'a, G: Game, J: Judge> Operator<G> for CensusOp<'a, J> {
    fn name(&self) -> String { "census".into() }
    fn accepts(&self, goal: &Goal) -> bool { matches!(goal, Goal::Survey { .. }) }
    fn run(&self, g: &G, s: &G::State, goal: &Goal, b: Budget) -> Outcome<G::Move> {
        let Goal::Survey { max_defects } = goal else { return Outcome::unsupported("census", goal) };
        let t0 = std::time::Instant::now();
        let r = census(g, s, self.judge, *max_defects, b.max_depth.map_or(CENSUS_DEPTH, |d| d as i32), b.max_nodes);
        let mut o = Outcome::new(if r.cut == 0 { Status::Read } else { Status::Gap });
        o.value = Some(r.tactics.len() as i64);
        o.evidence.push(format!("census: {} tactics, {} readings, {} cut, {} nodes", r.tactics.len(), r.readings, r.cut, r.nodes));
        o.facts = r.tactics.iter().map(|t| Fact { relation: "tactic".into(), args: vec![t.defects.to_string(), t.targets.join(","), t.line.join(" ")] }).collect();
        for l in &r.levels { o.part(format!("defects {}", l.defects), l.nodes, l.seconds); }
        o.cost.nodes = r.nodes;
        o.cost.seconds = t0.elapsed().as_secs_f64();
        o
    }
}

/// `reason`: the reasoning chain of the side to move (threats by stake, the frame, parry or counter; with no threat, our own tactics). Status Read, or Gap when the
/// budget cut a reading; `mv` the best candidate and `value` its net (or, with no threat, its tactic's value); `candidates` the order proposed; `evidence` the
/// explanation; facts `threat`(stake, targets, line), `frame`(facts), `answer`(kind, move, net), `irrelevant`(facts, moves). An ORDER and an explanation, never a verdict.
pub struct ReasonOp<'a, J: Judge> { pub judge: &'a J }

impl<'a, G: Game, J: Judge> Operator<G> for ReasonOp<'a, J> {
    fn name(&self) -> String { "reason".into() }
    fn accepts(&self, goal: &Goal) -> bool { matches!(goal, Goal::Reason { .. }) }
    fn run(&self, g: &G, s: &G::State, goal: &Goal, b: Budget) -> Outcome<G::Move> {
        let Goal::Reason { .. } = goal else { return Outcome::unsupported("reason", goal) };
        let t0 = std::time::Instant::now();
        let fuse = b.max_nodes.min(REASON_FUSE);
        let r = reason_within(g, s, self.judge, b.max_depth.map_or(REASON_DEPTH, |d| d as i32), fuse, Some(b.seconds_left()));
        let mut o = Outcome::new(if r.cut == 0 && r.unread.is_empty() { Status::Read } else { Status::Gap });
        let moves = g.moves(s);
        o.candidates = r.candidates.clone();
        if let Some(i) = r.candidates.first() {
            o.mv = moves.get(*i).cloned();
            o.value = r.answers.iter().find(|a| a.index == *i).map(|a| a.net);
        }
        o.evidence = explain(&r);
        let join = |v: &[Fact]| v.iter().map(|f| f.to_string()).collect::<Vec<_>>().join(" ");
        for t in &r.threats {
            o.facts.push(Fact { relation: "threat".into(), args: vec![t.stake.to_string(), t.targets.join(","), t.line.join(" ")] });
            o.facts.push(Fact { relation: "frame".into(), args: vec![t.stake.to_string(), join(&t.frame)] });
        }
        for a in &r.answers { o.facts.push(Fact { relation: "answer".into(), args: vec![a.kind.word().into(), a.name.clone(), a.net.to_string()] }); }
        for u in &r.irrelevant { o.facts.push(Fact { relation: "irrelevant".into(), args: vec![join(&u.facts), u.moves.iter().map(|i| r.names[*i].clone()).collect::<Vec<_>>().join(" ")] }); }
        o.part("threats", r.threat_nodes, r.threat_seconds);
        o.part("answers", r.answer_nodes, r.answer_seconds);
        if r.ours_nodes > 0 || r.ours_seconds > 0.0 { o.part("ours", r.ours_nodes, r.ours_seconds); }
        o.cost.nodes = r.nodes;
        o.cost.seconds = t0.elapsed().as_secs_f64();
        o
    }
}
