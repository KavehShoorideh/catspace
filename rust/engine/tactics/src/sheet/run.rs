//! PLAN 78.43 STAGE C, STEP TWO: THE WHOLE SHEET, game-free (the Python sheet's `run`, `step`, `_record`, `finish_levels`, `settle_flagged`, `first_order`, `quiet_question`, `priority`,
//! `chance_priority`, `stake_at_risk`, `verdict_kind`, `question_key`, `Chances`, `read_position`) on the light reading, the pool of waiting ways left out.
//!
//! The schedule is the anytime one: the questions in the first order, each asked a level at a time. THREE CLOCKS order the steps: `Present` (every question's level 0 first, then stake x
//! urgency over the expected cost of its next step), `Chance` (from the start, by the chance of its kind of verdict x the stake at risk over the cost) and `Plain` (the first order, each
//! question taken to its full depth in turn). The expected cost of a step is the mean of the costs seen at its level (or the level below times a growth, or a prior). THE COST of a step is,
//! by default, its measured time (`Costing::Measured`: the order then depends on the machine); `Costing::Fixed` gives every step of a level the same stated cost, which makes the order a
//! pure function of the position -- the form the two implementations are compared in. The budgets are `max_steps` and `budget_cost` (the summed cost of the steps taken); no wall-clock budget.
use super::levels::{self, Level0, Level1, WeakPlace};
use super::reading::Reading;
use super::settle::{Detail, Finding, Other, Settler, Q};
use super::walk::V;
use super::{Params, Side, Status, Unit};
use cs_game::placed::Placed;
use cs_game::Game;
use std::collections::HashMap;
use std::rc::Rc;
use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Clock { Present, Chance, Plain }

impl Clock {
    pub fn parse(s: &str) -> Result<Clock, String> {
        match s { "present" => Ok(Clock::Present), "chance" => Ok(Clock::Chance), "plain" => Ok(Clock::Plain), _ => Err(format!("no such clock: {s}")) }
    }
}

/// How the cost of a step is taken: measured (its wall time) or a stated number of seconds per level (the last one for every deeper level).
#[derive(Debug, Clone)]
pub enum Costing { Measured, Fixed(Vec<f64>) }

/// The named numbers of the schedule (the Python `Params`' own: urgency, costs, the quiet question, the risk, the chance horizon).
#[derive(Debug, Clone)]
pub struct Schedule {
    pub clock: Clock,
    pub urgency_start: f64,
    pub urgency_down: f64,
    pub urgency_up: f64,
    pub waiting_urgency: f64,
    pub quiet_stake_share: f64,
    pub cost_prior: [f64; 3],
    pub cost_growth: f64,
    pub risk: f64,
    pub chance_horizon: u32,
    pub costing: Costing,
    pub max_steps: Option<usize>,
    pub budget_cost: Option<f64>,
}

impl Default for Schedule {
    fn default() -> Schedule {
        Schedule { clock: Clock::Present, urgency_start: 1.0, urgency_down: 0.5, urgency_up: 2.0, waiting_urgency: 0.5, quiet_stake_share: 0.5, cost_prior: [0.002, 0.01, 0.01], cost_growth: 3.0,
                   risk: 1.0, chance_horizon: 4, costing: Costing::Measured, max_steps: None, budget_cost: None }
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------------------- the chance table

/// A table of counts key -> {horizon: (taken, total)} and the chance read from it: CHANCE = (taken + a) / (total + a + b), with a = SMOOTH x p and b = SMOOTH x (1 - p), p the chance of the
/// next coarser key (its prefix) computed the same way; at the root p is (taken + 1) / (total + 2). A key never seen has total 0 and so the chance of its coarser key.
#[derive(Debug, Clone, Default)]
pub struct Chances { agg: HashMap<Vec<String>, HashMap<u32, (i64, i64)>>, smooth: f64 }

pub const SMOOTH: f64 = 2.0;
pub const KEY_LEN: usize = 4;

impl Chances {
    pub fn new(cells: Vec<(Vec<String>, Vec<(u32, (i64, i64))>)>) -> Chances {
        let mut agg: HashMap<Vec<String>, HashMap<u32, (i64, i64)>> = HashMap::new();
        for (key, c) in &cells {
            for n in 0..=key.len() {
                let slot = agg.entry(key[..n].to_vec()).or_default();
                for (h, (taken, total)) in c {
                    let t = slot.entry(*h).or_insert((0, 0));
                    t.0 += taken;
                    t.1 += total;
                }
            }
        }
        Chances { agg, smooth: SMOOTH }
    }

    fn counts(&self, key: &[String], horizon: u32) -> (i64, i64) { self.agg.get(key).and_then(|m| m.get(&horizon)).copied().unwrap_or((0, 0)) }

    pub fn chance(&self, key: &[String], horizon: u32) -> f64 {
        let key = &key[..key.len().min(KEY_LEN)];
        let mut p = 0.0;
        for n in 0..=key.len() {
            let (taken, total) = self.counts(&key[..n], horizon);
            p = if n == 0 { (taken as f64 + 1.0) / (total as f64 + 2.0) } else { (taken as f64 + self.smooth * p) / (total as f64 + self.smooth) };
        }
        p
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------------------- questions and what a step found

#[derive(Debug, Clone)]
pub enum Found {
    /// the quiet question: the unit that does least, how many places it bears on and can go to
    Quiet { unit: Unit, strikes: usize, places: usize },
    L0(Level0),
    /// level 1, with the goal unit's weak places
    L1(Rc<Level1>, Option<Vec<WeakPlace>>),
    Deep(Finding),
}

#[derive(Debug, Clone)]
pub struct Fnd { pub level: usize, pub status: Status, pub new: bool, pub value: Option<f64>, pub cost: f64, pub found: Found }

impl Fnd {
    pub fn mode(&self) -> &'static str {
        match &self.found { Found::Deep(f) => match f.detail { Detail::Frame(_) => "frame", Detail::Search(_) => "search", Detail::Census(_) => "census" }, _ => "" }
    }
}

#[derive(Clone)]
pub struct Qn {
    pub target: Option<Unit>,
    pub asker: Side,
    pub stake: f64,
    pub rank: usize,
    pub goal: bool,
    pub quiet: bool,
    /// the next level to ask
    pub level: usize,
    pub open: bool,
    pub status: Status,
    pub urgency: f64,
    pub spent: f64,
    pub live_by_level1: bool,
    pub closed_at: Option<usize>,
    pub best_value: Option<f64>,
    pub findings: Vec<Fnd>,
    /// the bounds of the question's node in the Python sheet's graph, once its first step is made: (low, high)
    pub node: Option<(f64, f64)>,
    pub waiting: bool,
    pub q: Option<Q>,
}

impl Qn {
    fn new(target: Option<Unit>, asker: Side, stake: f64, rank: usize, goal: bool, quiet: bool, urgency: f64) -> Qn {
        Qn { target, asker, stake, rank, goal, quiet, level: 0, open: true, status: Status::Idle, urgency, spent: 0.0, live_by_level1: false, closed_at: None, best_value: None, findings: vec![],
             node: None, waiting: false, q: None }
    }

    /// The side that owns the target.
    pub fn owner(&self) -> Side { self.asker.other() }

    /// (low, high) of the question's stake: its node's bounds as they now stand, else 0 to the stake it was given.
    pub fn stake_bounds(&self) -> (f64, f64) {
        match self.node { Some((low, high)) => (low.min(high), high), None => (0.0, self.stake) }
    }

    /// THE TOLERANCE FOR RISK: a unit of the side to move is at risk of being lost: `risk` times the high bound plus (1 - risk) times the midpoint of the bounds; a unit of the other side is a gain
    /// to be had: the midpoint.
    pub fn stake_at_risk(&self, risk: f64) -> f64 {
        let (low, high) = self.stake_bounds();
        let mid = 0.5 * (low + high);
        if self.owner() == Side::Us { risk * high + (1.0 - risk) * mid } else { mid }
    }

    fn deep_with_results(&self, last: bool) -> Option<&Finding> {
        let has = |f: &&Fnd| f.level >= 2 && matches!(&f.found, Found::Deep(d) if match &d.detail { Detail::Frame(x) => !x.results.is_empty(), Detail::Search(x) => !x.results.is_empty(), Detail::Census(_) => false });
        let pick = if last { self.findings.iter().rev().find(has) } else { self.findings.iter().find(has) };
        match pick { Some(Fnd { found: Found::Deep(d), .. }) => Some(d), _ => None }
    }

    fn level1(&self) -> Option<&Rc<Level1>> { self.findings.iter().find(|f| f.level == 1).and_then(|f| match &f.found { Found::L1(l, _) => Some(l), _ => None }) }

    /// The kind of verdict the question has reached by its deepest finding: `lost`, `confirmed`, `waiting`, `unsettled`, `refuted` at level 2 and on; before that `live0`, `closed1`..`closed3`,
    /// `live1`, `idle1`, `idle0`, and `unread` when nothing has been asked.
    pub fn verdict_kind(&self) -> String {
        if self.findings.is_empty() { return "unread".into(); }
        let deep = self.findings.iter().rev().find(|f| f.level >= 2);
        if let Some(Fnd { status, found: Found::Deep(d), .. }) = deep {
            match &d.detail {
                Detail::Frame(x) if !x.results.is_empty() => {
                    if x.results[0].way.is_none() && x.results[0].status == V::Confirmed { return "lost".into(); }
                    let st: Vec<V> = x.results.iter().filter(|e| e.way.is_some()).map(|e| e.status).collect();
                    for (k, w) in [(V::Confirmed, "confirmed"), (V::Waiting, "waiting"), (V::Unsettled, "unsettled")] { if st.contains(&k) { return w.into(); } }
                    return "refuted".into();
                }
                Detail::Search(x) if !x.results.is_empty() => {
                    return if x.results.iter().any(|e| e.status == "confirmed") { "confirmed".into() } else { "refuted".into() };
                }
                _ => return match status { Status::Live => "confirmed", Status::Closed => "refuted", Status::Idle => "waiting" }.into(),
            }
        }
        let lv0 = self.findings.iter().find(|f| f.level == 0);
        let lv1 = self.findings.iter().find(|f| f.level == 1);
        if lv0.is_some_and(|f| f.status == Status::Live) { return "live0".into(); }
        if let Some(f) = lv1 {
            if f.status == Status::Closed {
                let at = match &f.found { Found::L1(l, _) => l.failed_at.unwrap_or(0), _ => 0 };
                return format!("closed{at}");
            }
            return if f.status == Status::Live { "live1".into() } else { "idle1".into() };
        }
        "idle0".into()
    }

    /// The number of moves the way needs: 0 (lost where it stands), 1, 2, 3+ (the fewest among the ways that decided the verdict, else among those level 1 listed), `-` when none is listed.
    fn moves_word(&self, kind: &str) -> String {
        if kind == "lost" { return "0".into(); }
        let mut ways: Vec<u32> = vec![];
        if kind == "confirmed" || kind == "waiting" {
            if let Some(d) = self.deep_with_results(true) {
                match &d.detail {
                    Detail::Frame(x) => ways = x.results.iter().filter(|e| e.way.is_some() && e.status.word() == kind).map(|e| e.way.unwrap().d).collect(),
                    Detail::Search(x) => ways = x.ways.iter().zip(x.results.iter()).filter(|(_, e)| e.status == kind).map(|(w, _)| w.d).collect(),
                    _ => {}
                }
            }
        }
        if ways.is_empty() { if let Some(l1) = self.level1() { ways = l1.ways.iter().map(|w| w.d).collect(); } }
        match ways.iter().min() { None => "-".into(), Some(n) if *n < 3 => n.to_string(), Some(_) => "3+".into() }
    }

    /// `clear` (the best-ordered way's route has nothing in the way), `own` (an obstacle of the asker's own side only) or `theirs` (an obstacle of the other side, or one nothing can clear);
    /// `-` before level 1 or with no way listed.
    fn route_word(&self) -> String {
        let Some(l1) = self.level1() else { return "-".into() };
        if l1.ways.is_empty() { return "-".into(); }
        let rank = |w: &levels::Way| l1.rank.iter().find(|(x, _)| x == w).map_or(0, |x| x.1);
        let first = l1.ways.iter().min_by_key(|w| (w.d, rank(w), w.unit.kind, w.unit.at, w.arrival)).unwrap();
        let obs = l1.routes.iter().find(|(w, _)| w == first).and_then(|(_, rs)| rs.first()).map(|r| r.obstacles.as_slice()).unwrap_or(&[]);
        if obs.is_empty() { return "clear".into(); }
        if obs.iter().any(|o| o.unit.is_some_and(|u| u.side != self.asker) || o.dead) { "theirs".into() } else { "own".into() }
    }

    /// The KEY of the question's record: (whose unit: mine = the side to move / theirs, the verdict kind, the moves the way needs, the route's group).
    pub fn question_key(&self) -> [String; 4] {
        let kind = self.verdict_kind();
        let moves = self.moves_word(&kind);
        [(if self.owner() == Side::Us { "mine" } else { "theirs" }).to_string(), kind, moves, self.route_word()]
    }
}

/// The verdict kinds that say danger: live at level 0, lost where it stands, confirmed after an arrival.
pub const DANGER: [&str; 3] = ["live0", "lost", "confirmed"];

// ---------------------------------------------------------------------------------------------------------------------------------------------- the expected cost of a step

/// The expected cost of a question's next step: the mean seen at that level, else the level below times a growth, else a prior.
struct Costs { sum: HashMap<usize, f64>, n: HashMap<usize, usize> }

impl Costs {
    fn new() -> Costs { Costs { sum: HashMap::new(), n: HashMap::new() } }
    fn add(&mut self, level: usize, cost: f64) {
        *self.sum.entry(level).or_insert(0.0) += cost;
        *self.n.entry(level).or_insert(0) += 1;
    }
    fn expect(&self, s: &Schedule, level: usize) -> f64 {
        if let Some(n) = self.n.get(&level).filter(|n| **n > 0) { return (self.sum[&level] / *n as f64).max(1e-6); }
        if level >= 1 {
            if let Some(n) = self.n.get(&(level - 1)).filter(|n| **n > 0) { return (self.sum[&(level - 1)] / *n as f64 * s.cost_growth).max(1e-6); }
        }
        let pr = &s.cost_prior;
        pr[level.min(pr.len() - 1)] * s.cost_growth.powi((level as i64 - pr.len() as i64 + 1).max(0) as i32)
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------------------- the schedule

#[derive(Debug, Clone)]
pub struct StepRec { pub question: usize, pub level: usize, pub status: Status, pub mode: &'static str, pub cost: f64 }

/// The sheet of one position: the questions in the order of the sheet (the quiet one last, when offered), the steps in the order taken, and what the schedule needed.
pub struct SheetRun { pub qs: Vec<Qn>, pub steps: Vec<StepRec>, pub seconds: f64 }

pub struct Runner<'a, 'g, G: Game + Placed> {
    r: &'a Reading<'g, G>,
    p: &'a Params,
    s: &'a Schedule,
    table: Option<&'a Chances>,
    pub settler: Settler<'a, 'g, G>,
    pub qs: Vec<Qn>,
    pub steps: Vec<StepRec>,
}

impl<'a, 'g, G: Game + Placed> Runner<'a, 'g, G> {
    pub fn new(r: &'a Reading<'g, G>, p: &'a Params, s: &'a Schedule, table: Option<&'a Chances>) -> Self {
        Runner { r, p, s, table, settler: Settler::new(r, p), qs: vec![], steps: vec![] }
    }

    fn quiet_question(&self, rank: usize, lowest: f64) -> Option<Qn> {
        if !self.r.units.iter().any(|u| u.side == Side::Us) { return None; }
        Some(Qn::new(None, Side::Us, (lowest * self.s.quiet_stake_share).max(1e-6), rank, false, true, self.s.urgency_start))
    }

    /// The unit of the side to move that does least: it bears on the fewest places and can go to the fewest, the cheapest first; with the numbers.
    fn level_quiet(&self) -> (Status, Found) {
        let r = self.r;
        let mut rows: Vec<(usize, f64, Unit)> = vec![];
        for u in r.units.iter().filter(|u| u.side == Side::Us) { rows.push((r.strikes_of(u).len() + r.one_from(u.kind, u.side, u.at).len(), -r.worth_of(u.kind), *u)); }
        rows.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.partial_cmp(&b.1).unwrap()).then_with(|| a.2.kind.cmp(&b.2.kind)).then_with(|| a.2.at.cmp(&b.2.at)));
        let least = rows[0].2;
        (Status::Closed, Found::Quiet { unit: least, strikes: r.strikes_of(&least).len(), places: r.one_from(least.kind, least.side, least.at).len() })
    }

    fn cost_of(&self, level: usize, t0: Instant) -> f64 {
        match &self.s.costing { Costing::Measured => t0.elapsed().as_secs_f64(), Costing::Fixed(v) => v[level.min(v.len() - 1)] }
    }

    /// The summary of one question that the time obstacle of the others reads.
    fn other_of(q: &Qn) -> Other {
        let ways = match q.deep_with_results(false) {
            Some(d) => match &d.detail {
                Detail::Frame(x) => x.results.iter().filter_map(|e| e.way.map(|w| (w.d, e.status == V::Confirmed || e.status == V::Waiting))).collect(),
                Detail::Search(x) => x.ways.iter().zip(x.results.iter()).map(|(w, e)| (w.d, e.status == "confirmed")).collect(),
                Detail::Census(_) => vec![],
            },
            None => vec![],
        };
        Other { asker: q.asker, stake: q.stake, quiet: q.quiet, target: q.target, ways }
    }

    /// Deepen question `qi` by one level: what it found, what it cost, and whether it is now closed, idle or live.
    fn step(&mut self, qi: usize) -> usize {
        let (r, p) = (self.r, self.p);
        let level = self.qs[qi].level;
        let (quiet, goal, asker, stake) = { let q = &self.qs[qi]; (q.quiet, q.goal, q.asker, q.stake) };
        let t0 = Instant::now();
        let (status, found, value): (Status, Found, Option<f64>) = if quiet {
            let (s, f) = self.level_quiet();
            (s, f, None)
        } else {
            let t = self.qs[qi].target.unwrap();
            if level == 0 {
                let l0 = levels::level0(r, &t, asker, goal);
                (l0.status, Found::L0(l0), None)
            } else if level == 1 {
                let l1 = levels::level1(r, &t, asker, goal, p);
                let weak = if goal { Some(levels::weak_places(r, &t, asker, p)) } else { None };
                let l1 = Rc::new(l1);
                self.qs[qi].q = Some(Q { t, asker, stake, goal, blocked: 0, l1: l1.clone() });
                (l1.status, Found::L1(l1, weak), None)
            } else {
                let q = self.qs[qi].q.as_mut().expect("a deeper level is asked after level 1");
                let f = self.settler.deep(q, level);
                let value = match &f.detail {
                    Detail::Frame(x) => x.value,
                    Detail::Search(x) => x.value,
                    Detail::Census(x) => x.value.map(|v| v as f64),
                };
                if let Detail::Frame(x) = &f.detail { self.qs[qi].waiting = x.waiting > 0; }
                (f.status, Found::Deep(f), value)
            }
        };
        let cost = self.cost_of(level, t0);
        let q = &mut self.qs[qi];
        let new = status == Status::Live && (value.is_none() || q.best_value.is_none() || value.unwrap() > q.best_value.unwrap());
        // the question's node in the claims graph: made at its first step, its bounds tightened by a level 2 finding
        if !quiet {
            let (mut low, mut high) = q.node.unwrap_or((0.0, stake));
            if level >= 2 && status == Status::Live && value.is_some() {
                let v = value.unwrap();
                low = low.max(v);
                high = high.min(v);
            } else if level >= 2 && status == Status::Closed {
                low = low.max(0.0);
                high = high.min(0.0);
            }
            if low > high { low = high; }
            q.node = Some((low, high));
        }
        q.findings.push(Fnd { level, status, new, value, cost, found });
        q.spent += cost;
        q.level += 1;
        q.status = status;
        if let Some(v) = value { if q.best_value.is_none_or(|b| v > b) { q.best_value = Some(v); } }
        if level <= 1 && status == Status::Live { q.live_by_level1 = true; }
        if status == Status::Closed { q.open = false; q.closed_at = Some(level); }
        qi
    }

    /// Step the question and record the step (the cost in `costs`, the urgency, the deepest level).
    fn take(&mut self, qi: usize, costs: &mut Costs) {
        self.step(qi);
        let s = self.s;
        let q = &mut self.qs[qi];
        let f = q.findings.last().unwrap();
        let (level, status, new, cost, mode) = (f.level, f.status, f.new, f.cost, f.mode());
        self.steps.push(StepRec { question: qi, level, status, mode, cost });
        costs.add(if q.quiet { 0 } else { level }, cost);
        if status == Status::Live && new { q.urgency *= s.urgency_up; }
        else if status != Status::Closed {
            q.urgency *= s.urgency_down;
            if status == Status::Idle && q.waiting && level >= 2 { q.urgency *= s.waiting_urgency; }
        }
        if q.open && !q.quiet && q.level > self.p.max_level { q.open = false; q.closed_at = Some(level); }
        let o = Self::other_of(&self.qs[qi]);
        let mut others = self.settler.others.borrow_mut();
        if qi < others.len() { others[qi] = o; }
    }

    fn priority(&self, q: &Qn, costs: &Costs) -> f64 { q.stake * q.urgency / costs.expect(self.s, q.level) }

    fn chance_priority(&self, q: &Qn, costs: &Costs) -> f64 {
        let chance = match self.table { None => 1.0, Some(t) => t.chance(&q.question_key(), self.s.chance_horizon) };
        chance * q.stake_at_risk(self.s.risk) / costs.expect(self.s, q.level)
    }

    fn sync_others(&self) {
        let mut o = self.settler.others.borrow_mut();
        while o.len() < self.qs.len() { let q = &self.qs[o.len()]; o.push(Self::other_of(q)); }
    }

    /// THE ANYTIME SCHEDULE over the position's questions (`only`: the units to ask about, the rest are not on the sheet; their ranks are kept).
    pub fn run(&mut self, only: Option<&[Unit]>) {
        let (r, p, s) = (self.r, self.p, self.s);
        let order = r.first_order(p);
        let mut questions: Vec<Qn> = order.iter().map(|(u, stake, goal, rank)| Qn::new(Some(*u), u.side.other(), *stake, *rank, *goal, false, s.urgency_start)).collect();
        if let Some(o) = only { questions.retain(|q| o.contains(&q.target.unwrap())); }
        let n_questions = questions.len();
        let lowest = questions.iter().map(|q| q.stake).fold(None, |m: Option<f64>, x| Some(m.map_or(x, |y| y.min(x)))).unwrap_or(p.min_stake);
        self.qs = questions;
        self.sync_others();
        let max_steps = if s.max_steps.is_none() && s.budget_cost.is_none() { Some(40) } else { s.max_steps };
        let mut costs = Costs::new();
        let mut spent = 0.0f64;
        let over = |steps: usize, spent: f64| max_steps.is_some_and(|m| steps >= m) || s.budget_cost.is_some_and(|b| spent >= b);
        let mut quiet_made = false;
        if s.clock == Clock::Present {
            for qi in 0..n_questions {
                if over(self.steps.len(), spent) { break; }
                self.take(qi, &mut costs);
                spent += self.steps.last().unwrap().cost;
            }
        }
        while !over(self.steps.len(), spent) {
            if !quiet_made && n_questions > 0 && self.qs[..n_questions].iter().all(|q| q.level > 0 && (!q.open || q.status == Status::Idle)) {
                if let Some(qq) = self.quiet_question(n_questions + 1, lowest) { self.qs.push(qq); quiet_made = true; self.sync_others(); }
            }
            let pending: Vec<usize> = (0..self.qs.len()).filter(|i| self.qs[*i].open).collect();
            if pending.is_empty() { break; }
            let pick = match s.clock {
                Clock::Chance => pending.iter().copied().max_by(|a, b| {
                    let (x, y) = (&self.qs[*a], &self.qs[*b]);
                    self.chance_priority(x, &costs).partial_cmp(&self.chance_priority(y, &costs)).unwrap().then_with(|| y.rank.cmp(&x.rank))
                }),
                Clock::Plain => pending.iter().copied().min_by_key(|i| self.qs[*i].rank),
                Clock::Present => pending.iter().copied().max_by(|a, b| {
                    let (x, y) = (&self.qs[*a], &self.qs[*b]);
                    self.priority(x, &costs).partial_cmp(&self.priority(y, &costs)).unwrap().then_with(|| y.rank.cmp(&x.rank))
                }),
            }.unwrap();
            self.take(pick, &mut costs);
            spent += self.steps.last().unwrap().cost;
        }
    }

    /// Step every open question that has not yet been asked up to level `upto`; then, when every question is closed or idle and the quiet question has not been offered, offer it and step it.
    pub fn finish_levels(&mut self, upto: usize) {
        let mut costs = Costs::new();
        for qi in 0..self.qs.len() {
            while !self.qs[qi].quiet && self.qs[qi].open && self.qs[qi].level <= upto { self.take(qi, &mut costs); }
        }
        let plain: Vec<usize> = (0..self.qs.len()).filter(|i| !self.qs[*i].quiet).collect();
        if !plain.is_empty() && !self.qs.iter().any(|q| q.quiet) && plain.iter().all(|i| !self.qs[*i].open || self.qs[*i].status == Status::Idle) {
            let lowest = plain.iter().map(|i| self.qs[*i].stake).fold(f64::INFINITY, f64::min);
            if let Some(qq) = self.quiet_question(plain.len() + 1, lowest) {
                self.qs.push(qq);
                self.sync_others();
                let qi = self.qs.len() - 1;
                self.take(qi, &mut costs);
            }
        }
    }

    /// Step to level 2 every non-goal unit above the lowest worth whose question is open after level 1.
    pub fn settle_flagged(&mut self) {
        let mut costs = Costs::new();
        for qi in 0..self.qs.len() {
            let q = &self.qs[qi];
            if !q.quiet && !q.goal && q.open && q.level == 2 && q.live_by_level1 && self.settler.weighty(&q.target.unwrap()) { self.take(qi, &mut costs); }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(a: &str, b: &str, c: &str, d: &str) -> Vec<String> { vec![a.into(), b.into(), c.into(), d.into()] }

    /// One cell seen 3 times of 4 at horizon 4: the root is (3 + 1) / (4 + 2), each longer prefix leans on the one above it with SMOOTH pseudo-observations.
    #[test]
    fn the_chance_leans_on_its_coarser_key() {
        let t = Chances::new(vec![(key("mine", "lost", "0", "clear"), vec![(4, (3, 4))])]);
        let root = 4.0 / 6.0;
        let want = {
            let mut p = root;
            for _ in 0..4 { p = (3.0 + SMOOTH * p) / (4.0 + SMOOTH); }
            p
        };
        assert!((t.chance(&key("mine", "lost", "0", "clear"), 4) - want).abs() < 1e-12);
        let unseen = t.chance(&key("theirs", "refuted", "1", "own"), 4);
        assert!((unseen - root).abs() < 1e-12);                  // a key never seen has the chance of the coarsest key it shares: here only the root
    }

    #[test]
    fn a_fixed_cost_model_makes_the_expected_cost_a_function_of_the_level() {
        let s = Schedule { costing: Costing::Fixed(vec![0.1, 0.2]), ..Schedule::default() };
        let mut c = Costs::new();
        assert_eq!(c.expect(&s, 0), 0.002);                      // before any step: the prior
        c.add(0, 0.1);
        c.add(0, 0.3);
        assert!((c.expect(&s, 0) - 0.2).abs() < 1e-12);          // the mean seen at the level
        assert!((c.expect(&s, 1) - 0.6).abs() < 1e-12);          // the level below times the growth
    }
}
