//! THE TRAINING KIT'S FIRST OPERATOR (plan 78.17). An `Advisor` is an outside source that VALUES a position: it orders the legal moves and scores them, on a scale
//! nobody here knows. `TeacherOp` puts one behind the shared `Operator` interface under `Kit::Training`, so an inference drive never runs it; what it returns
//! (candidates and `advice` facts) is DATA for the miners to learn from, never a decision. Game-free: it reads only the `Game` trait. The real adapters (a
//! subprocess engine) live outside this crate; `ScriptedAdvisor` is the stand-in for tests.
use cs_game::{Fact, Game};
use cs_operator::{Budget, Goal, Kit, Operator, Outcome, Status};
use std::cell::RefCell;
use std::collections::HashMap;
use std::time::Instant;

/// An outside source of move values. Several can coexist: each is one source among others.
pub trait Advisor {
    /// The source's name, for the provenance of what it labelled.
    fn name(&self) -> String;
    /// Always true for an advisor; a source that may decide is not an advisor.
    fn training_only(&self) -> bool { true }
    /// The legal moves of `s` as (index into `g.moves(s)`, score), best first; None when the source has no opinion on this position.
    fn rank<G: Game>(&mut self, g: &G, s: &G::State) -> Option<Vec<(usize, i64)>>;
}

/// An advisor that reads its answers from a table: state key -> ranking. For tests.
#[derive(Default)]
pub struct ScriptedAdvisor { pub name: String, pub table: HashMap<u64, Vec<(usize, i64)>> }

impl ScriptedAdvisor {
    pub fn new(name: &str) -> ScriptedAdvisor { ScriptedAdvisor { name: name.to_string(), table: HashMap::new() } }
    pub fn with(mut self, key: u64, ranking: Vec<(usize, i64)>) -> ScriptedAdvisor { self.table.insert(key, ranking); self }
}

impl Advisor for ScriptedAdvisor {
    fn name(&self) -> String { self.name.clone() }
    fn rank<G: Game>(&mut self, g: &G, s: &G::State) -> Option<Vec<(usize, i64)>> { self.table.get(&g.key(s)).cloned() }
}

/// The operator around an advisor. `run` takes `&self`, so the advisor sits in a cell (it may keep a process or a cache).
pub struct TeacherOp<A: Advisor> { advisor: RefCell<A> }

impl<A: Advisor> TeacherOp<A> {
    pub fn new(advisor: A) -> TeacherOp<A> {
        assert!(advisor.training_only(), "an advisor that may decide is not an advisor");
        TeacherOp { advisor: RefCell::new(advisor) }
    }
}

impl<G: Game, A: Advisor> Operator<G> for TeacherOp<A> {
    fn name(&self) -> String { format!("advice:{}", self.advisor.borrow().name()) }
    fn kit(&self) -> Kit { Kit::Training }
    fn accepts(&self, goal: &Goal) -> bool { matches!(goal, Goal::Advise { .. }) }
    fn run(&self, g: &G, s: &G::State, goal: &Goal, _budget: Budget) -> Outcome<G::Move> {
        if !Operator::<G>::accepts(self, goal) { return Outcome::unsupported(&Operator::<G>::name(self), goal); }
        let t0 = Instant::now();
        let mut adv = self.advisor.borrow_mut();
        let source = adv.name();
        let n = g.moves(s).len();
        let ranking = adv.rank(g, s).map(|r| r.into_iter().filter(|(i, _)| *i < n).collect::<Vec<_>>()).filter(|r| !r.is_empty());
        let Some(ranking) = ranking else {
            let mut o = Outcome::new(Status::NoLine);
            o.evidence.push(format!("advice {source}: no opinion on this position"));
            o.cost.seconds = t0.elapsed().as_secs_f64();
            return o;
        };
        let mut o = Outcome::new(Status::Read);
        o.candidates = ranking.iter().map(|(i, _)| *i).collect();
        o.value = Some(ranking[0].1);
        o.facts = ranking.iter().map(|(i, sc)| Fact::new("advice", &[&source, &i.to_string(), &sc.to_string()])).collect();
        o.evidence.push(format!("advice {source}: ranked {} moves, best index {} at {}", ranking.len(), ranking[0].0, ranking[0].1));
        o.cost.nodes = ranking.len() as u64;
        o.cost.seconds = t0.elapsed().as_secs_f64();
        o.part("advice", o.cost.nodes, o.cost.seconds);
        o
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cs_game::Role;

    /// A counting game: the state is a number, a move adds 1, 2 or 3, the game ends at 5 or more.
    struct Count;
    impl Game for Count {
        type State = u32;
        type Move = u32;
        fn name(&self) -> String { "count".into() }
        fn roles(&self) -> Vec<String> { vec!["a".into(), "b".into()] }
        fn initial(&self) -> u32 { 0 }
        fn mover(&self, s: &u32) -> Option<Role> { if *s >= 5 { None } else { Some(0) } }
        fn moves(&self, s: &u32) -> Vec<u32> { if *s >= 5 { vec![] } else { vec![1, 2, 3] } }
        fn apply(&self, s: &u32, m: &u32) -> u32 { s + m }
        fn outcome(&self, s: &u32) -> Option<Vec<i64>> { if *s >= 5 { Some(vec![100, 0]) } else { None } }
        fn move_name(&self, _s: &u32, m: &u32) -> String { m.to_string() }
        fn key(&self, s: &u32) -> u64 { *s as u64 }
    }

    fn op() -> TeacherOp<ScriptedAdvisor> { TeacherOp::new(ScriptedAdvisor::new("script").with(0, vec![(2, 40), (0, 10), (1, -5)])) }

    #[test]
    fn returns_the_scripted_order_as_candidates_and_facts_with_cost() {
        let o = op().run(&Count, &0, &Goal::Advise { role: 0 }, Budget::new(1.0, 100));
        assert_eq!(o.status, Status::Read);
        assert_eq!(o.candidates, vec![2, 0, 1]);
        assert_eq!(o.value, Some(40));
        assert_eq!(o.facts.len(), 3);
        assert_eq!(o.facts[0], Fact::new("advice", &["script", "2", "40"]));
        assert_eq!(o.facts[2], Fact::new("advice", &["script", "1", "-5"]));
        assert_eq!(o.cost.nodes, 3);
        assert!(o.cost.seconds >= 0.0);
        assert_eq!(o.parts.len(), 1);
    }

    #[test]
    fn is_a_training_operator_that_reads_only_advise() {
        let o = op();
        assert_eq!(Operator::<Count>::kit(&o), Kit::Training);
        assert!(Operator::<Count>::accepts(&o, &Goal::Advise { role: 0 }));
        assert!(!Operator::<Count>::accepts(&o, &Goal::Survey { max_defects: 1 }));
        let r = o.run(&Count, &0, &Goal::Survey { max_defects: 1 }, Budget::new(1.0, 100));
        assert_eq!(r.status, Status::Unsupported);
    }

    #[test]
    fn no_opinion_is_no_line_and_bad_indices_are_dropped() {
        let o = op();
        let r = o.run(&Count, &1, &Goal::Advise { role: 0 }, Budget::new(1.0, 100));
        assert_eq!(r.status, Status::NoLine);
        assert!(r.facts.is_empty());
        let o2 = TeacherOp::new(ScriptedAdvisor::new("s").with(0, vec![(9, 1), (1, 0)]));
        assert_eq!(o2.run(&Count, &0, &Goal::Advise { role: 0 }, Budget::new(1.0, 100)).candidates, vec![1]);
    }

    #[test]
    fn an_inference_drive_never_runs_it() {
        use cs_operator::{drive, drive_in, Action, Context, Controller};
        struct Ask;
        impl Controller<Count> for Ask {
            fn choose(&self, ctx: &Context<Count>) -> Action<u32> {
                if ctx.steps.is_empty() { Action::Run { operator: "advice:script".into(), goal: Goal::Advise { role: 0 }, seconds: 1.0, max_nodes: 100, max_depth: None } }
                else { Action::Play { mv: 1, proved: false } }
            }
        }
        let o = op();
        let ops: [&dyn Operator<Count>; 1] = [&o];
        let d = drive(&Count, &0, 1.0, &ops, &Ask, vec![]);
        assert_eq!(d.steps[0].outcome.status, Status::Unsupported);
        assert!(d.steps[0].outcome.facts.is_empty());
        let d = drive_in(Kit::Training, &Count, &0, 1.0, &ops, &Ask, vec![]);
        assert_eq!(d.steps[0].outcome.status, Status::Read);
    }
}
