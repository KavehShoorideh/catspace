//! PLAN 78.12: the shared interface itself -- what an operator refuses, what a call reports, what the driver does with a controller that asks for what is not there.
use cs_game::kernel::KernelGame;
use cs_game::Game;
use cs_operator::{drive, Action, Budget, Context, Controller, Goal, Mode, Operator, Status, MAX_STEPS};
use cs_tactics::ledger::Count;
use cs_tactics::operators::{CensusOp, ForcingOp, LedgerOp};

fn at(fen: &str) -> (KernelGame, <KernelGame as Game>::State) {
    let g = KernelGame::from_fen("chess", fen).unwrap();
    let s = g.initial();
    (g, s)
}

const MATE_IN_ONE: &str = "6k1/5ppp/8/8/8/8/5PPP/R3K3 w - - 0 1";

#[test]
fn forcing_reads_the_forced_mode_only() {
    let (g, s) = at(MATE_IN_ONE);
    let op = ForcingOp;
    let forced = Goal::Win { role: 0, mode: Mode::Forced };
    assert!(Operator::<KernelGame>::accepts(&op, &forced));
    for mode in [Mode::Cooperative, Mode::Modeled] {
        let goal = Goal::Win { role: 0, mode };
        assert!(!Operator::<KernelGame>::accepts(&op, &goal));
        let o = op.run(&g, &s, &goal, Budget::new(1.0, 1_000_000));
        assert_eq!(o.status, Status::Unsupported);
        assert!(o.mv.is_none());
    }
    let o = op.run(&g, &s, &forced, Budget::new(5.0, 1_000_000).depth(2));
    assert_eq!(o.status, Status::Proved);
    assert!(o.mv.is_some() && o.cost.nodes > 0 && o.cost.seconds >= 0.0 && !o.evidence.is_empty());
}

#[test]
fn a_depth_cap_ends_the_ledger_reading_by_depth_and_reports_it() {
    let (g, s) = at("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2");
    let j = Count;
    let op = LedgerOp::new(&j);
    let o = op.run(&g, &s, &Goal::Ledger { role: 0, ties: None, prefer: vec![] }, Budget::new(60.0, 50_000_000).depth(3));
    assert_eq!(o.status, Status::Read);
    assert_eq!(o.depth, Some(3));
    assert!(o.cost.nodes > 0 && o.value.is_some() && o.mv.is_some() && o.candidates.is_empty());
    let o2 = op.run(&g, &s, &Goal::Ledger { role: 0, ties: Some(0.3), prefer: vec![] }, Budget::new(60.0, 50_000_000).depth(3));
    assert!(!o2.candidates.is_empty() && o2.cost.nodes >= o.cost.nodes);
}

#[test]
fn the_survey_reports_its_tactics_as_facts() {
    let (g, s) = at("4k3/8/2n5/8/Q7/8/8/4K3 w - - 0 1");
    let j = Count;
    let op = CensusOp { judge: &j };
    let o = op.run(&g, &s, &Goal::Survey { max_defects: 1 }, Budget::new(30.0, 200_000).depth(3));
    assert!(matches!(o.status, Status::Read | Status::Gap));
    assert_eq!(o.facts.len() as i64, o.value.unwrap());
    assert!(!o.facts.is_empty(), "the queen attacks the knight: a tactic");
    assert!(o.facts.iter().all(|f| f.relation == "tactic" && f.args.len() == 3));
}

/// A controller that asks for an operator nobody holds, then for a goal the operator refuses, then stops.
struct Lost;
impl Controller<KernelGame> for Lost {
    fn choose(&self, ctx: &Context<KernelGame>) -> Action<<KernelGame as Game>::Move> {
        match ctx.steps.len() {
            0 => Action::Run { operator: "nobody".into(), goal: Goal::Survey { max_defects: 0 }, seconds: 1.0, max_nodes: 1000, max_depth: None },
            1 => Action::Run { operator: "forcing".into(), goal: Goal::Win { role: 0, mode: Mode::Modeled }, seconds: 1.0, max_nodes: 1000, max_depth: None },
            _ => Action::Stop,
        }
    }
}

#[test]
fn the_driver_answers_a_missing_operator_or_a_refused_goal_with_unsupported_and_goes_on() {
    let (g, s) = at(MATE_IN_ONE);
    let f = ForcingOp;
    let ops: [&dyn Operator<KernelGame>; 1] = [&f];
    let d = drive(&g, &s, 1.0, &ops, &Lost, vec![]);
    assert!(d.mv.is_none());
    assert_eq!(d.steps.len(), 2);
    assert!(d.steps.iter().all(|st| st.outcome.status == Status::Unsupported));
    assert_eq!(d.trace.len(), 2);
}

/// A controller that never plays.
struct Never;
impl Controller<KernelGame> for Never {
    fn choose(&self, _ctx: &Context<KernelGame>) -> Action<<KernelGame as Game>::Move> {
        Action::Run { operator: "forcing".into(), goal: Goal::Win { role: 0, mode: Mode::Forced }, seconds: 1.0, max_nodes: 1000, max_depth: Some(1) }
    }
}

#[test]
fn a_controller_that_never_plays_is_stopped_by_the_step_fuse() {
    let (g, s) = at(MATE_IN_ONE);
    let f = ForcingOp;
    let ops: [&dyn Operator<KernelGame>; 1] = [&f];
    let d = drive(&g, &s, 5.0, &ops, &Never, vec![]);
    assert_eq!(d.steps.len(), MAX_STEPS);
    assert!(d.mv.is_none());
}

#[test]
fn the_batch_keeps_item_order_and_turns_a_panic_into_an_error_row() {
    use cs_decision::batch::{error_row, solve};
    let items: Vec<usize> = (0..20).collect();
    let out = solve(&items, 4, "snap", |i, it| {
        if *it == 7 { panic!("one position fails"); }
        error_row(i, "snap", "ok", "-")
    });
    assert_eq!(out.len(), 20);
    for (i, (s, v)) in out.iter().enumerate() {
        assert_eq!(s.index, i);
        assert_eq!(v["index"].as_u64(), Some(i as u64));
    }
    assert_eq!(out[7].0.error.as_deref(), Some("the worker panicked"));
    assert_eq!(out[8].0.error.as_deref(), Some("-"));
}

fn parts_nodes<M>(o: &cs_operator::Outcome<M>) -> u64 { o.parts.iter().map(|p| p.cost.nodes).sum() }

#[test]
fn every_operator_reports_its_cost_in_parts_that_add_up_to_the_total() {
    let j = Count;
    // forcing: one part per deepening step
    let (g, s) = at(MATE_IN_ONE);
    let o = ForcingOp.run(&g, &s, &Goal::Win { role: 0, mode: Mode::Forced }, Budget::new(5.0, 1_000_000).depth(3));
    assert!(!o.parts.is_empty() && parts_nodes(&o) == o.cost.nodes, "forcing {:?} total {}", o.parts, o.cost.nodes);
    // ledger: one part per depth, and the ties
    let (g, s) = at("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2");
    let op = LedgerOp::new(&j);
    let o = op.run(&g, &s, &Goal::Ledger { role: 0, ties: Some(0.3), prefer: vec![] }, Budget::new(60.0, 50_000_000).depth(3));
    assert_eq!(o.parts.iter().filter(|p| p.name.starts_with("depth")).count(), 3);
    assert!(o.parts.iter().any(|p| p.name == "ties"));
    assert_eq!(parts_nodes(&o), o.cost.nodes, "ledger {:?}", o.parts);
    assert!(!o.line.is_empty());
    // census: one part per defect level
    let (g, s) = at("4k3/8/2n5/8/Q7/8/8/4K3 w - - 0 1");
    let o = CensusOp { judge: &j }.run(&g, &s, &Goal::Survey { max_defects: 1 }, Budget::new(30.0, 200_000).depth(3));
    assert_eq!(o.parts.len(), 2);
    assert_eq!(parts_nodes(&o), o.cost.nodes);
}

#[test]
fn the_existing_operators_belong_to_the_inference_kit() {
    use cs_operator::Kit;
    assert_eq!(Operator::<KernelGame>::kit(&ForcingOp), Kit::Inference);
}
