//! PLAN 78.12 THE TACTICS LAYER'S OPERATORS: `forcing`, `ledger` and `census` behind the shared interface (cs_operator). Each wraps the function the engine has always
//! called, unchanged; what is new is that the call states its goal and mode, returns its evidence, and reports the cost it measured. Game-free.
use crate::census::census;
use crate::ledger::{Judge, Search, WIN};
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
}

impl<'a, J: Judge> LedgerOp<'a, J> {
    pub fn new(judge: &'a J) -> Self { LedgerOp { judge, history: vec![], ordering: true, prune: true, extend: true } }
}

impl<'a, G: Game, J: Judge> Operator<G> for LedgerOp<'a, J> {
    fn name(&self) -> String { "ledger".into() }
    fn accepts(&self, goal: &Goal) -> bool { matches!(goal, Goal::Ledger { .. }) }
    fn run(&self, g: &G, s: &G::State, goal: &Goal, b: Budget) -> Outcome<G::Move> {
        let Goal::Ledger { role, ties, prefer } = goal else { return Outcome::unsupported("ledger", goal) };
        let t0 = std::time::Instant::now();
        let mut ls = Search::new(g, self.judge, *role, b.seconds_left());
        ls.set_fuse(b.max_nodes);
        ls.history = self.history.iter().copied().collect();
        ls.ordering = self.ordering;
        ls.prune = self.prune;
        ls.extend = self.extend;
        ls.root_first = prefer.clone();
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
        o.mv = Some(moves[i].clone());
        o.value = Some(ln.value);
        o.depth = Some(ln.depth as i64);
        o.line = ln.line.clone();
        o.events = ln.lost.clone();
        for r in &ls.reads { o.part(format!("depth {}{}", r.depth, if r.complete { "" } else { " (cut)" }), r.nodes, r.seconds); }
        let who = |r: cs_game::Role| if r == *role { "we" } else { "they" };
        let lost: Vec<String> = ln.lost.iter().map(|(owner, k)| format!("{} {k}", who(*owner))).collect();
        o.evidence.push(format!("ledger search [{}]: {} at depth {} ({} nodes): the line {} -- {}; value {}", self.judge.name(), g.move_name(s, &moves[i]), ln.depth, ln.nodes,
                                ln.line.join(" "), if lost.is_empty() { "nobody loses a unit".to_string() } else { lost.join(", ") }, ln.value));
        let mut nodes = ln.nodes;
        if let Some(share) = ties {
            if ln.value.abs() < WIN - 1000 {
                ls.resume(b.rest_of_decision() * share);
                let n0 = ls.nodes;
                let t1 = std::time::Instant::now();
                o.candidates = ls.ties(s, ln.depth, ln.value);
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
