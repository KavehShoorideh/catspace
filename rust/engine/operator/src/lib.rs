//! PLAN 78.12 THE SHARED INTERFACE (docs/ANALYST_PLAN.md 78.12; the architecture document's Operator / Controller): every tool a decision uses sits behind ONE interface,
//! every decision-maker behind ONE other, goals are explicit arguments, and what a tool returns says what it found, on what evidence, at what MEASURED cost. Game-free:
//! it reads only the `Game` trait. Nothing here decides anything; the layer crates implement `Operator` (tactics: forcing, ledger, census; planning: keep, density) and
//! the decision crate holds the first `Controller` (the fixed order the engine has played by since 78.5).
//!
//! * `Goal`     a condition or objective over states, with the MODE it is asked in (forced: whatever the other side plays; cooperative: if it helps; modeled: against an
//!              opponent model). Mixing modes produces false lemmas, so a tool answers a mode it does not read with `Unsupported`, never with a guess.
//! * `Budget`   what a call may spend: a deadline, a node fuse, a deepest reading asked, and the deadline of the whole decision it belongs to.
//! * `Outcome`  a status, the move or value found, the evidence (the trace lines), the MEASURED cost (nodes and seconds) and any new facts.
//! * `Operator` `run(game, state, goal, budget) -> Outcome`.
//! * `Controller` `choose(context) -> Action`: which operator, which goal, how much budget -- or the move to play. A mixture of experts is a controller holding others.
//! * `drive`    the loop: ask the controller, run the operator, record the step, until the controller plays.
use cs_game::{Game, Role};
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------------------------------------------------------------------------------- budget

/// The budget ran out: what was being read is a GAP, never a refutation.
#[derive(Debug, Clone, Copy)]
pub struct Cut;

/// What one call may spend: a deadline, a node fuse (a named cap, checked as nodes are spent) and the deepest reading asked.
#[derive(Debug, Clone)]
pub struct Budget {
    pub deadline: Instant,
    pub max_nodes: u64,
    pub nodes: u64,
    /// The deepest reading asked (a named cap; the clock is the real bound). None: the operator's own default.
    pub max_depth: Option<u32>,
    /// The deadline of the WHOLE decision this call belongs to: an operator that spends a share of "what is left" reads it.
    pub outer: Instant,
}

impl Budget {
    pub fn new(seconds: f64, max_nodes: u64) -> Budget {
        let d = Instant::now() + Duration::from_secs_f64(seconds.max(0.0));
        Budget { deadline: d, max_nodes, nodes: 0, max_depth: None, outer: d }
    }
    pub fn within(mut self, outer: Instant) -> Budget { self.outer = outer; if self.deadline > outer { self.deadline = outer; } self }
    pub fn depth(mut self, d: u32) -> Budget { self.max_depth = Some(d); self }
    /// Spend one node; Err when the fuse or the clock says stop (the clock is read every 256 nodes).
    pub fn spend(&mut self) -> Result<(), Cut> {
        self.nodes += 1;
        if self.nodes >= self.max_nodes || (self.nodes & 255 == 0 && Instant::now() >= self.deadline) { return Err(Cut); }
        Ok(())
    }
    pub fn left(&self) -> bool { self.nodes < self.max_nodes && Instant::now() < self.deadline }
    pub fn seconds_left(&self) -> f64 { self.deadline.saturating_duration_since(Instant::now()).as_secs_f64() }
    /// Seconds until the whole decision's deadline.
    pub fn rest_of_decision(&self) -> f64 { self.outer.saturating_duration_since(Instant::now()).as_secs_f64() }
}

// ---------------------------------------------------------------------------------------------------------------------------------------------------- goals

/// The mode a goal is asked in. Stored lemmas are `Forced` only.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode { Forced, Cooperative, Modeled }

/// A condition or objective over states. Game-free: it names roles and move indices, never a unit or a square.
#[derive(Debug, Clone, PartialEq)]
pub enum Goal {
    /// The game's own outcome for `role` beats every other role's (the root goal: read from the game's terminal rules).
    Win { role: Role, mode: Mode },
    /// The line that loses least of `role`'s units and wins most of theirs, by the operator's judge. `ties`: when set, also read the moves as good as the best one,
    /// with that share of what is left of the decision's clock.
    Ledger { role: Role, ties: Option<f64>, prefer: Vec<usize> },
    /// The moves (indices into the legal moves) a learned lemma proposes for `role` in this position: a prior, never a verdict.
    Respond { role: Role },
    /// The move whose worst reply leaves `role`'s measure highest.
    Keep { role: Role },
    /// Among the moves `among` (indices into the game's legal moves), the one after which `role`'s tactics most outnumber theirs, read down to `defects` defects.
    Prefer { role: Role, among: Vec<usize>, defects: u32 },
    /// The tactics of the side to move, down to `max_defects` defects: reported as facts.
    Survey { max_defects: u32 },
    /// The expected outcome for `role` by sampled playouts: an estimate, never a proof.
    Estimate { role: Role },
    /// An outside source's ordering of the legal moves for `role`, with its scores (scale unknown): a label to learn from, so only a training-kit operator reads it.
    Advise { role: Role },
}

impl Goal {
    pub fn describe(&self) -> String {
        match self {
            Goal::Win { role, mode } => format!("win({role},{mode:?})"),
            Goal::Ledger { role, ties, prefer } => format!("ledger({role}{}{})", if ties.is_some() { ",ties" } else { "" }, if prefer.is_empty() { String::new() } else { format!(",prefer {}", prefer.len()) }),
            Goal::Respond { role } => format!("respond({role})"),
            Goal::Keep { role } => format!("keep({role})"),
            Goal::Prefer { role, among, defects } => format!("prefer({role},{} moves,{defects} defects)", among.len()),
            Goal::Survey { max_defects } => format!("survey({max_defects})"),
            Goal::Estimate { role } => format!("estimate({role})"),
            Goal::Advise { role } => format!("advise({role})"),
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------------------------- outcomes

/// What a call found.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    /// The goal is reached, exactly, within the horizon: `mv` starts the line.
    Proved,
    /// No such line within the horizon: every line was read (exact at that depth).
    NoLine,
    /// The budget cut the reading before it could answer: unknown, never a refutation.
    Gap,
    /// An objective was read within the horizon: `value` (and `mv`) are exact at `depth`.
    Read,
    /// The operator does not read this goal or this mode.
    Unsupported,
}

/// What a call cost, measured (never estimated).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Cost { pub nodes: u64, pub seconds: f64 }

pub use cs_game::Fact;

/// One named part of a call's measured cost (a deepening step, a phase, a candidate read): the parts of an outcome sum to no more than its total.
#[derive(Debug, Clone, PartialEq)]
pub struct Part { pub name: String, pub cost: Cost }

#[derive(Debug, Clone)]
pub struct Outcome<M> {
    pub status: Status,
    pub mv: Option<M>,
    pub value: Option<i64>,
    pub depth: Option<i64>,
    /// Indices into the game's legal moves the operator found equally good (empty when it was not asked or found none).
    pub candidates: Vec<usize>,
    /// The trace lines, in the words the harness prints.
    pub evidence: Vec<String>,
    /// The principal line (move names) and the ledger events along it as (role, event): "lose <kind>", "promote <from> to <to>".
    pub line: Vec<String>,
    pub events: Vec<(Role, String)>,
    pub cost: Cost,
    /// The cost in parts, as measured: what each deepening step, phase or candidate spent.
    pub parts: Vec<Part>,
    /// New facts. Only the survey emits them so far.
    pub facts: Vec<Fact>,
}

impl<M> Outcome<M> {
    pub fn new(status: Status) -> Outcome<M> {
        Outcome { status, mv: None, value: None, depth: None, candidates: vec![], evidence: vec![], line: vec![], events: vec![], cost: Cost::default(), parts: vec![], facts: vec![] }
    }
    /// Record one part of the cost.
    pub fn part(&mut self, name: impl Into<String>, nodes: u64, seconds: f64) { self.parts.push(Part { name: name.into(), cost: Cost { nodes, seconds } }); }
    pub fn unsupported(by: &str, goal: &Goal) -> Outcome<M> {
        let mut o = Outcome::new(Status::Unsupported);
        o.evidence.push(format!("{by}: does not read the goal {}", goal.describe()));
        o
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------------------------- the two interfaces

/// THE TWO TOOL KITS. At inference (playing, analysis) the engine may use only what it holds -- the store and search over the rules. At training (sleep) it may also use
/// outside sources (a teacher, past games, deep search). The kit is a property of each operator, so a mixture of experts keeps one boundary, not a special case.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kit { Inference, Training }

/// A tool: reads a position toward a goal within a budget, and says what it found at what measured cost.
pub trait Operator<G: Game> {
    fn name(&self) -> String;
    /// The kit this operator belongs to. Inference unless it says otherwise: a Training operator is never run by an inference drive.
    fn kit(&self) -> Kit { Kit::Inference }
    /// Does this operator read this goal (in this mode)?
    fn accepts(&self, goal: &Goal) -> bool;
    fn run(&self, g: &G, s: &G::State, goal: &Goal, budget: Budget) -> Outcome<G::Move>;
}

/// One call made and what came back.
#[derive(Debug, Clone)]
pub struct Step<M> {
    pub operator: String,
    pub goal: Goal,
    /// The kit of the operator that answered (Inference when no operator was found: nothing ran).
    pub kit: Kit,
    pub outcome: Outcome<M>,
}

/// What a controller sees when it chooses.
pub struct Context<'a, G: Game> {
    pub g: &'a G,
    pub s: &'a G::State,
    pub started: Instant,
    pub deadline: Instant,
    /// The decision's whole clock, in seconds.
    pub seconds: f64,
    /// Every step so far, in order.
    pub steps: &'a [Step<G::Move>],
}

impl<'a, G: Game> Context<'a, G> {
    pub fn elapsed(&self) -> f64 { self.started.elapsed().as_secs_f64() }
    pub fn remaining(&self) -> f64 { self.deadline.saturating_duration_since(Instant::now()).as_secs_f64() }
}

/// A decision-maker's choice.
#[derive(Debug, Clone)]
pub enum Action<M> {
    /// Run `operator` toward `goal` with this budget.
    Run { operator: String, goal: Goal, seconds: f64, max_nodes: u64, max_depth: Option<u32> },
    /// Play this move; `proved`: a goal was proved for it.
    Play { mv: M, proved: bool },
    /// Nothing to play.
    Stop,
}

/// Which operator, which goal, how much budget -- or the move.
pub trait Controller<G: Game> {
    fn choose(&self, ctx: &Context<G>) -> Action<G::Move>;
}

/// The decision a drive ended in.
#[derive(Debug, Clone)]
pub struct Decision<M> {
    pub mv: Option<M>,
    pub proved: bool,
    pub trace: Vec<String>,
    pub steps: Vec<Step<M>>,
    /// The sum of the steps' measured costs, and the decision's own wall time.
    pub cost: Cost,
}

/// A fuse on the number of calls one decision may make (a controller that never plays is stopped here).
pub const MAX_STEPS: usize = 32;

/// THE LOOP under the inference kit (what play and analysis use): see `drive_in`.
pub fn drive<G: Game>(g: &G, s: &G::State, seconds: f64, ops: &[&dyn Operator<G>], ctl: &dyn Controller<G>, trace: Vec<String>) -> Decision<G::Move> {
    drive_in(Kit::Inference, g, s, seconds, ops, ctl, trace)
}

/// THE LOOP: ask the controller; run the operator it names (an operator that is missing or does not accept the goal answers `Unsupported`, so the controller sees it
/// and moves on); record the step and its evidence; until the controller plays or stops. `kit` is the kit the drive runs under: an operator of the Training kit named under an Inference drive is NOT run (the step is
/// `Unsupported`, with a line saying so); a Training drive runs both kits. `trace` starts with the lines the caller already holds.
pub fn drive_in<G: Game>(kit: Kit, g: &G, s: &G::State, seconds: f64, ops: &[&dyn Operator<G>], ctl: &dyn Controller<G>, mut trace: Vec<String>) -> Decision<G::Move> {
    let started = Instant::now();
    let deadline = started + Duration::from_secs_f64(seconds.max(0.0));
    let mut steps: Vec<Step<G::Move>> = vec![];
    let mut nodes = 0u64;
    let (mv, proved) = loop {
        if steps.len() >= MAX_STEPS { break (None, false); }
        let action = ctl.choose(&Context { g, s, started, deadline, seconds, steps: &steps });
        match action {
            Action::Play { mv, proved } => break (Some(mv), proved),
            Action::Stop => break (None, false),
            Action::Run { operator, goal, seconds: sec, max_nodes, max_depth } => {
                let mut budget = Budget::new(sec, max_nodes).within(deadline);
                budget.max_depth = max_depth;
                let t0 = Instant::now();
                let mut step_kit = Kit::Inference;
                let mut outcome = match ops.iter().find(|o| o.name() == operator && o.accepts(&goal)) {
                    Some(op) if op.kit() == Kit::Training && kit == Kit::Inference => {
                        step_kit = Kit::Training;
                        let mut o = Outcome::new(Status::Unsupported);
                        o.evidence.push(format!("{operator}: belongs to the training kit; an inference drive does not run it"));
                        o
                    }
                    Some(op) => { step_kit = op.kit(); op.run(g, s, &goal, budget) }
                    None => Outcome::unsupported(&operator, &goal),
                };
                if outcome.cost.seconds == 0.0 { outcome.cost.seconds = t0.elapsed().as_secs_f64(); }
                nodes += outcome.cost.nodes;
                trace.extend(outcome.evidence.iter().cloned());
                steps.push(Step { operator, goal, kit: step_kit, outcome });
            }
        }
    };
    Decision { mv, proved, trace, steps, cost: Cost { nodes, seconds: started.elapsed().as_secs_f64() } }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A counting game: the state is a number, a move adds 1 or 2, the game ends at 3 or more.
    struct Count;
    impl Game for Count {
        type State = u32;
        type Move = u32;
        fn name(&self) -> String { "count".into() }
        fn roles(&self) -> Vec<String> { vec!["a".into(), "b".into()] }
        fn initial(&self) -> u32 { 0 }
        fn mover(&self, s: &u32) -> Option<Role> { if *s >= 3 { None } else { Some(0) } }
        fn moves(&self, s: &u32) -> Vec<u32> { if *s >= 3 { vec![] } else { vec![1, 2] } }
        fn apply(&self, s: &u32, m: &u32) -> u32 { s + m }
        fn outcome(&self, s: &u32) -> Option<Vec<i64>> { if *s >= 3 { Some(vec![100, 0]) } else { None } }
        fn move_name(&self, _s: &u32, m: &u32) -> String { m.to_string() }
        fn key(&self, s: &u32) -> u64 { *s as u64 }
    }

    struct Plain;
    impl Operator<Count> for Plain {
        fn name(&self) -> String { "plain".into() }
        fn accepts(&self, _: &Goal) -> bool { true }
        fn run(&self, _: &Count, _: &u32, _: &Goal, _: Budget) -> Outcome<u32> { Outcome::new(Status::Read) }
    }
    struct Trained;
    impl Operator<Count> for Trained {
        fn name(&self) -> String { "trained".into() }
        fn kit(&self) -> Kit { Kit::Training }
        fn accepts(&self, _: &Goal) -> bool { true }
        fn run(&self, _: &Count, _: &u32, _: &Goal, _: Budget) -> Outcome<u32> { Outcome::new(Status::Read) }
    }
    /// Asks each named operator once, then plays.
    struct Ask(Vec<&'static str>);
    impl Controller<Count> for Ask {
        fn choose(&self, ctx: &Context<Count>) -> Action<u32> {
            match self.0.get(ctx.steps.len()) {
                Some(n) => Action::Run { operator: n.to_string(), goal: Goal::Advise { role: 0 }, seconds: 1.0, max_nodes: 10, max_depth: None },
                None => Action::Play { mv: 1, proved: false },
            }
        }
    }

    #[test]
    fn default_kit_is_inference() {
        assert_eq!(Operator::<Count>::kit(&Plain), Kit::Inference);
        assert_eq!(Operator::<Count>::kit(&Trained), Kit::Training);
    }

    #[test]
    fn an_inference_drive_refuses_a_training_operator() {
        let ops: [&dyn Operator<Count>; 2] = [&Plain, &Trained];
        let d = drive(&Count, &0, 1.0, &ops, &Ask(vec!["trained", "plain"]), vec![]);
        assert_eq!(d.steps[0].outcome.status, Status::Unsupported);
        assert_eq!(d.steps[0].kit, Kit::Training);
        assert!(d.steps[0].outcome.evidence[0].contains("training kit"));
        assert!(d.trace.iter().any(|l| l.contains("training kit")));
        assert_eq!(d.steps[1].outcome.status, Status::Read);
        assert_eq!(d.steps[1].kit, Kit::Inference);
    }

    #[test]
    fn a_training_drive_runs_both_kits() {
        let ops: [&dyn Operator<Count>; 2] = [&Plain, &Trained];
        let d = drive_in(Kit::Training, &Count, &0, 1.0, &ops, &Ask(vec!["trained", "plain"]), vec![]);
        assert!(d.steps.iter().all(|s| s.outcome.status == Status::Read));
        assert_eq!(d.steps[0].kit, Kit::Training);
    }
}
