//! PLAN 78.26 WIN / DRAW / LOSS FROM ANY POSITION (Kaveh: "We need a probability of win, draw, loss from any given position ... query the database for positions that are
//! like this, or run a Monte Carlo simulation from that point on. Once we know how sharp the position is: if the chance of draw is low, win or loss is high, we should
//! continue reasoning. If not, we should rely on the probabilities to pick one path."). Game-free: it reads the `Game` trait and the judge's worths only.
//!   `Wdl`      win / draw / loss MASS (counts, possibly fractional) from one role's view, with its probabilities, score and SHARPNESS (1 minus the draw probability);
//!   `split`    one outcome (a terminal's goal value, or a cut playout's leaf value) as mass, the score kept exactly (no threshold);
//!   `Table`    a results table keyed by an ABSTRACTION of the position (the exact key, then the units standing): rows are counts the caller fills;
//!   `WdlOp`    the operator "wdl": `Goal::Estimate` (the position's WDL: playouts pooled with a table hit) and `Goal::Prefer` (the probabilities pick among moves).
//! What it answers is an ESTIMATE, never a proof. The controller reads the sharpness (`sharp`) to choose between reasoning on and taking the probabilities.
use crate::mcts::{search, Rollout, Search, BATCH, DRAW};
use crate::rollout::Leaf;
use cs_game::{Fact, Game, Role};
use cs_operator::{Budget, Goal, Kit, Operator, Outcome, Status};
use cs_tactics::ledger::Judge;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

/// A position whose decisive mass (win plus loss over the total) is at or above this is sharp: the draw is not the likely end.
pub const SHARP: f64 = 0.5;
/// Below this much mass (playouts plus table) the estimate cannot say: the sharpness is unknown.
pub const MIN_MASS: f64 = 30.0;
/// A table row needs this much mass to count as a hit.
pub const TABLE_MIN: f64 = 20.0;
/// The most mass a table hit may bring when it is pooled with playouts (a big row must not drown the position's own playouts).
pub const TABLE_CAP: f64 = 200.0;
/// The playout cap in plies before the leaf valuation (short: against the training oracle a longer playout tracked the result no better and bought fewer of them).
pub const WDL_PLAYOUT: usize = 8;
/// The leaf's squash scale, in units of the smallest positive worth standing at the root.
pub const WDL_SCALE: f64 = 3.0;
/// How many of the best moves the evidence lists.
pub const TOP_MOVES: usize = 3;
/// The reported value is the score times this (the scale `MctsOp` reports its mean outcome in).
pub const SCORE_SCALE: f64 = 10_000.0;

/// Win / draw / loss mass from one role's view.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Wdl { pub win: f64, pub draw: f64, pub loss: f64 }

impl Wdl {
    pub fn new(win: f64, draw: f64, loss: f64) -> Wdl { Wdl { win, draw, loss } }
    /// The total mass.
    pub fn n(&self) -> f64 { self.win + self.draw + self.loss }
    /// (win, draw, loss) as probabilities; None when there is no mass.
    pub fn p(&self) -> Option<(f64, f64, f64)> { let n = self.n(); if n > 0.0 { Some((self.win / n, self.draw / n, self.loss / n)) } else { None } }
    /// The expected score: a win counts 1, a draw one half.
    pub fn score(&self) -> f64 { let n = self.n(); if n > 0.0 { (self.win + self.draw / 2.0) / n } else { 0.5 } }
    /// 1 minus the draw probability: how decisive the position is.
    pub fn sharpness(&self) -> f64 { let n = self.n(); if n > 0.0 { (self.win + self.loss) / n } else { 0.0 } }
    /// The sum of two masses.
    pub fn add(&self, o: &Wdl) -> Wdl { Wdl { win: self.win + o.win, draw: self.draw + o.draw, loss: self.loss + o.loss } }
    /// Add one outcome's mass (see `split`).
    pub fn add_mass(&mut self, m: [f64; 3]) { self.win += m[0]; self.draw += m[1]; self.loss += m[2]; }
    /// The same proportions at a total mass of `to_mass` (unchanged when there is no mass).
    pub fn scaled(&self, to_mass: f64) -> Wdl { let n = self.n(); if n > 0.0 { let k = to_mass / n; Wdl { win: self.win * k, draw: self.draw * k, loss: self.loss * k } } else { *self } }
    /// The other role's view in a game of two roles.
    pub fn flip(&self) -> Wdl { Wdl { win: self.loss, draw: self.draw, loss: self.win } }
}

/// One outcome (a value 0..100 for a role) as [win, draw, loss] mass of total 1, the score kept exactly: above one half the win is 2v - 1 and the draw 2 - 2v;
/// below it the loss is 1 - 2v and the draw 2v; one half is a draw. (v is the value over 100, clamped to 0..1.)
pub fn split(value: f64) -> [f64; 3] {
    let v = (value / 100.0).clamp(0.0, 1.0);
    if v > 0.5 { [2.0 * v - 1.0, 2.0 - 2.0 * v, 0.0] } else if v < 0.5 { [0.0, 2.0 * v, 1.0 - 2.0 * v] } else { [0.0, 1.0, 0.0] }
}

/// Sharp (true), quiet (false), or None: unknown, for too little mass.
pub fn sharp(w: &Wdl) -> Option<bool> { if w.n() < MIN_MASS { None } else { Some(w.sharpness() >= SHARP) } }

/// The word for a sharpness reading.
pub fn word(s: Option<bool>) -> &'static str { match s { Some(true) => "sharp", Some(false) => "quiet", None => "unknown" } }

/// A results table: counts keyed by an abstraction of the position. Rows are the MOVER's view (mover wins, draws, mover losses). A plain map the caller fills.
#[derive(Debug, Clone, Default)]
pub struct Table { pub name: String, pub rows: HashMap<String, [f64; 3]> }

impl Table {
    /// The keys of a position, most specific first: `exact:<key in hex>`, then `units:<each role's unit kinds sorted and joined by ',', the roles joined by '|'>:<the
    /// mover's role name>` (a game that reports no units gives the exact key only).
    pub fn keys<G: Game>(g: &G, s: &G::State) -> Vec<String> {
        let mut keys = vec![format!("exact:{:x}", g.key(s))];
        let units = g.units(s);
        if !units.is_empty() {
            let roles = g.roles();
            let per: Vec<String> = (0..roles.len()).map(|r| {
                let mut kinds: Vec<String> = units.iter().filter(|(o, _)| *o == r).map(|(_, k)| k.clone()).collect();
                kinds.sort();
                kinds.join(",")
            }).collect();
            if let Some(m) = g.mover(s) { keys.push(format!("units:{}:{}", per.join("|"), roles[m])); }
        }
        keys
    }

    /// The first key of the position whose row holds `TABLE_MIN` mass, with its counts seen from `role`.
    pub fn lookup<G: Game>(&self, g: &G, s: &G::State, role: Role) -> Option<(String, Wdl)> {
        let mover = g.mover(s)?;
        for k in Table::keys(g, s) {
            if let Some(r) = self.rows.get(&k) {
                let w = Wdl::new(r[0], r[1], r[2]);
                if w.n() >= TABLE_MIN { return Some((k, if role == mover { w } else { w.flip() })); }
            }
        }
        None
    }
}

/// The leaf scale the operator's callers use: `WDL_SCALE` times the smallest positive worth among the unit kinds standing at `s` (1 when there is none), so the squash
/// does not depend on the judge's units.
pub fn leaf_scale<G: Game, J: Judge>(g: &G, s: &G::State, judge: &J) -> f64 {
    let smallest = g.units(s).iter().map(|(_, k)| judge.worth(k)).filter(|w| *w > 0).min().unwrap_or(1);
    WDL_SCALE * smallest as f64
}

/// One position's reading: the pooled mass, what each source brought, and the search behind it.
struct Reading { wdl: Wdl, playouts: Wdl, table: Option<(String, Wdl, Wdl)>, search: Option<Search> }

impl Reading {
    fn sources(&self) -> &'static str {
        match (self.playouts.n() > 0.0, self.table.is_some()) { (true, true) => "playouts+table", (true, false) => "playouts", (false, true) => "table", _ => "none" }
    }
}

/// THE OPERATOR "wdl" (see the module docstring). `table`: an optional shared results table; `leaf_note` is a sentence the evidence repeats about how the leaf was
/// scaled (the caller who built the leaf knows).
pub struct WdlOp<R: Rollout + Clone, L: Leaf + Clone> {
    pub exploration: f64,
    pub max_playout: usize,
    pub rollout: R,
    pub leaf: L,
    pub table: Option<Arc<Table>>,
    pub leaf_note: String,
}

impl<R: Rollout + Clone, L: Leaf + Clone> WdlOp<R, L> {
    pub fn new(exploration: f64, max_playout: usize, rollout: R, leaf: L, table: Option<Arc<Table>>) -> WdlOp<R, L> {
        WdlOp { exploration, max_playout, rollout, leaf, table, leaf_note: String::new() }
    }

    /// The reading of `s` for `role` within `budget`: playouts pooled with the table's hit (its mass scaled down to at most `TABLE_CAP`). A state with no move to read
    /// (terminal, or stuck) is one sure outcome at `MIN_MASS`, no search.
    fn read<G: Game>(&self, g: &G, s: &G::State, role: Role, budget: &mut Budget) -> Reading {
        let fixed = |o: Vec<f64>| -> Reading {
            let mut w = Wdl::default();
            w.add_mass(split(o.get(role).copied().unwrap_or(DRAW)));
            let w = w.scaled(MIN_MASS);
            Reading { wdl: w, playouts: w, table: None, search: None }
        };
        if let Some(o) = g.outcome(s) { return fixed(o.iter().map(|x| *x as f64).collect()); }
        if g.moves(s).is_empty() { return fixed(self.leaf.value(g, s)); }
        let mut rollout = self.rollout.clone();
        let r = search(g, s, budget, self.exploration, self.max_playout, &mut rollout, &self.leaf);
        let playouts = r.root_wdl.get(role).copied().unwrap_or_default();
        let hit = self.table.as_ref().and_then(|t| t.lookup(g, s, role));
        let mut wdl = playouts;
        let table = hit.map(|(k, w)| {
            let used = w.scaled(w.n().min(TABLE_CAP));
            wdl = wdl.add(&used);
            (k, w, used)
        });
        Reading { wdl, playouts, table, search: Some(r) }
    }

    fn wdl_fact(w: &Wdl, sources: &str) -> Fact {
        let (a, b, c) = w.p().unwrap_or((0.0, 0.0, 0.0));
        Fact { relation: "wdl".into(), args: vec![format!("{a:.4}"), format!("{b:.4}"), format!("{c:.4}"), format!("{:.1}", w.n()), sources.into()] }
    }
    fn move_fact(name: &str, w: &Wdl) -> Fact {
        let (a, b, c) = w.p().unwrap_or((0.0, 0.0, 0.0));
        Fact { relation: "move_wdl".into(), args: vec![name.into(), format!("{a:.4}"), format!("{b:.4}"), format!("{c:.4}"), format!("{:.1}", w.n())] }
    }
    /// The facts every reading carries: the pooled `wdl`, its `sharpness`, the `playouts` (playouts, cut) and, on a table hit, `table_hit` (key, then the row's counts).
    fn reading_facts(rd: &Reading, out: &mut Vec<Fact>) {
        out.push(WdlOp::<R, L>::wdl_fact(&rd.wdl, rd.sources()));
        out.push(Fact { relation: "sharpness".into(), args: vec![format!("{:.4}", rd.wdl.sharpness()), word(sharp(&rd.wdl)).into()] });
        let (p, c) = rd.search.as_ref().map_or((0, 0), |r| (r.playouts, r.cut));
        out.push(Fact { relation: "playouts".into(), args: vec![p.to_string(), c.to_string()] });
        if let Some((k, w, _)) = &rd.table { out.push(Fact { relation: "table_hit".into(), args: vec![k.clone(), format!("{}", w.win), format!("{}", w.draw), format!("{}", w.loss)] }); }
    }
    fn probs(w: &Wdl) -> String { let (a, b, c) = w.p().unwrap_or((0.0, 0.0, 0.0)); format!("win {a:.4} draw {b:.4} loss {c:.4}") }
    fn names(&self) -> String { format!("rollout {}, leaf {}{}", self.rollout.name(), self.leaf.name(), if self.leaf_note.is_empty() { String::new() } else { format!(" ({})", self.leaf_note) }) }
}

/// The score as the outcome's integer value.
fn scaled(score: f64) -> i64 { (score * SCORE_SCALE).round() as i64 }

impl<G: Game, R: Rollout + Clone, L: Leaf + Clone> Operator<G> for WdlOp<R, L> {
    fn name(&self) -> String { "wdl".into() }
    /// Inference: the table and the playouts are things the engine holds.
    fn kit(&self) -> Kit { Kit::Inference }
    fn accepts(&self, goal: &Goal) -> bool { matches!(goal, Goal::Estimate { .. } | Goal::Prefer { .. }) }
    fn run(&self, g: &G, s: &G::State, goal: &Goal, mut budget: Budget) -> Outcome<G::Move> {
        let t0 = Instant::now();
        match goal {
            Goal::Estimate { role } => {
                let role = *role;
                let rd = self.read(g, s, role, &mut budget);
                let moves = g.moves(s);
                let mut out = Outcome::new(Status::Read);
                let n_play = rd.search.as_ref().map_or(0, |r| r.playouts);
                if rd.wdl.n() <= 0.0 { out.status = Status::Gap; out.evidence.push("wdl: the budget ended before one playout finished, and no table row holds this position".into()); }
                out.value = Some(scaled(rd.wdl.score()));
                let sh = sharp(&rd.wdl);
                WdlOp::<R, L>::reading_facts(&rd, &mut out.facts);
                let mut ranked: Vec<(usize, Wdl)> = vec![];
                if let Some(r) = &rd.search {
                    let mut order: Vec<usize> = (0..r.root.len()).collect();
                    order.sort_by(|a, b| r.root[*b].0.cmp(&r.root[*a].0).then(a.cmp(b)));
                    if let Some(&top) = order.first() { if n_play > 0 { out.mv = moves.get(top).cloned(); } }
                    if n_play > 0 { out.candidates = order; }
                    out.depth = Some(r.depth);
                    for (i, w) in r.move_wdl.iter().enumerate() {
                        let w = w.get(role).copied().unwrap_or_default();
                        out.facts.push(WdlOp::<R, L>::move_fact(&g.move_name(s, &moves[i]), &w));
                        ranked.push((i, w));
                    }
                    for (i, (n, sec)) in r.batches.iter().enumerate() { out.part(format!("playouts {}..{}", i as u64 * BATCH, ((i as u64 + 1) * BATCH).min(r.playouts.max(1))), *n, *sec); }
                    out.cost.nodes = r.nodes;
                }
                out.cost.seconds = t0.elapsed().as_secs_f64();
                out.evidence.push(format!("wdl: {} from {:.1} mass for role {role} ({})", WdlOp::<R, L>::probs(&rd.wdl), rd.wdl.n(), rd.sources()));
                let mut src = format!("wdl: sources: playouts {:.1} mass", rd.playouts.n());
                if let Some(r) = &rd.search { src += &format!(" ({} playouts, {} cut at {} plies, depth {}; {})", r.playouts, r.cut, self.max_playout, r.depth, self.names()); }
                match &rd.table {
                    Some((k, w, used)) => src += &format!("; table {:.1} mass from {:.1} (key {k}, {})", used.n(), w.n(), WdlOp::<R, L>::probs(&w.scaled(w.n()))),
                    None => src += if self.table.is_some() { "; table: no row of this position holds enough mass" } else { "; table: none given" },
                }
                out.evidence.push(src);
                out.evidence.push(format!("wdl: sharpness {:.4} (win + loss over mass; sharp at or above {SHARP}, unknown below {MIN_MASS} mass): {}", rd.wdl.sharpness(), word(sh)));
                ranked.retain(|(_, w)| w.n() > 0.0);
                ranked.sort_by(|a, b| b.1.score().partial_cmp(&a.1.score()).unwrap_or(std::cmp::Ordering::Equal).then(a.0.cmp(&b.0)));
                for (i, w) in ranked.iter().take(TOP_MOVES) {
                    out.evidence.push(format!("wdl: move {} score {:.4}: {} ({:.0} playouts)", g.move_name(s, &moves[*i]), w.score(), WdlOp::<R, L>::probs(w), w.n()));
                }
                out
            }
            Goal::Prefer { role, among, .. } => {
                let role = *role;
                let moves = g.moves(s);
                let among: Vec<usize> = among.iter().copied().filter(|i| *i < moves.len()).collect();
                if among.is_empty() { return Outcome::new(Status::Gap); }
                let mut out = Outcome::new(Status::Read);
                let k = among.len();
                let mut reads: Vec<(usize, Reading)> = vec![];
                let mut nodes = 0u64;
                for (j, &i) in among.iter().enumerate() {
                    let t1 = Instant::now();
                    let left = (k - j) as f64;
                    let secs = budget.seconds_left() / left;
                    let fuse = budget.max_nodes.saturating_sub(nodes) / (k - j) as u64;
                    let mut b = Budget::new(secs, fuse.max(1)).within(budget.outer);
                    let q = g.apply(s, &moves[i]);
                    let rd = self.read(g, &q, role, &mut b);
                    nodes += b.nodes;
                    out.part(format!("read {}", g.move_name(s, &moves[i])), b.nodes, t1.elapsed().as_secs_f64());
                    reads.push((i, rd));
                }
                // rank by score, an unread move last, ties in the order of `among`
                let mut order: Vec<usize> = (0..reads.len()).collect();
                let key = |r: &Reading| if r.wdl.n() > 0.0 { r.wdl.score() } else { -1.0 };
                order.sort_by(|a, b| key(&reads[*b].1).partial_cmp(&key(&reads[*a].1)).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(b)));
                let best = order[0];
                out.mv = Some(moves[reads[best].0].clone());
                out.value = Some(scaled(reads[best].1.wdl.score()));
                out.candidates = order.iter().map(|o| reads[*o].0).collect();
                out.cost.nodes = nodes;
                out.cost.seconds = t0.elapsed().as_secs_f64();
                WdlOp::<R, L>::reading_facts(&reads[best].1, &mut out.facts);
                for (i, rd) in &reads {
                    let name = g.move_name(s, &moves[*i]);
                    out.facts.push(WdlOp::<R, L>::move_fact(&name, &rd.wdl));
                }
                out.evidence.push(format!("wdl: the probabilities pick among {k} moves, the budget split evenly over them ({})", self.names()));
                for o in &order {
                    let (i, rd) = &reads[*o];
                    out.evidence.push(format!("wdl: move {} score {:.4}: {} from {:.1} mass ({}{})", g.move_name(s, &moves[*i]), rd.wdl.score(), WdlOp::<R, L>::probs(&rd.wdl), rd.wdl.n(), rd.sources(),
                        rd.table.as_ref().map_or(String::new(), |(k, _, _)| format!(", table key {k}"))));
                }
                out
            }
            _ => Outcome::unsupported("wdl", goal),
        }
    }
}
