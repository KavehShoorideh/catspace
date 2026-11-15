//! PLAN 78.46 THE FORCING READ, game-free: a move of the side to move, the claims it makes, EVERY reply, and the claims that stand after each reply.
//!
//! A THREAT MOVE of the side to move is a legal move that (a) itself takes a unit, or (b) ends the game in its favour, or (c) after which -- the board read as if the same side moved
//! again (`Game::pass` on the child) -- a unit of the other side stands lost to an exchange on its place (the exchange frame of the sheet, from the real position) that did not stand
//! before the move, or the goal is forced in one move; where the game gives no pass for the child (the other side is under the goal's sign) the move is a threat by that alone, and
//! what it makes stand is read on the child's replies directly.
//!
//! THE READ of a threat move: the child is the other side's turn, and EVERY legal reply is read, none left out. For a reply the line's value (for the mover, a lower bound of what it
//! is sure to have won less lost) is: what the move took, less what the reply took, plus what stands after the reply -- the best exchange on a place of the other side's, the real
//! frame from the position after the reply, counted only as far as it exceeds what stood on that place before the move (the claim is the move's own: what stood already is not credited
//! to it) -- or the forced win of the goal in one (the engine's `wins_in`). Where that value is not above zero the mover's own threat moves in the position after the reply are read
//! one level deeper (the same read: every reply), while the budget lasts. The value of the move is the least over the replies; the goal is above every worth (`BIG`).
//!
//! CONFIRMED: the value is above zero for every reply. NO CLAIM: some reply leaves a value not above zero and nothing further to read (no threat move of ours after it). NOT SETTLED: neither,
//! because the budget stopped the reading (a reply line still open at the deepest level read, or the fuse). A line's value never counts a reading that was cut, and a cut line is never a
//! refutation: a cut at an OPTION of the mover (a deeper threat move not read) can only leave the lower bound lower, so it matters only to a line whose value is not already above zero; a cut at a
//! REPLY (one not read) is a cut of the whole move. THE FUSE: positions read (a position whose claims are asked counts one; one `wins_in` call has a node budget of its own), `ForcingParams::fuse`; the
//! deepening is by levels (0 = nothing deeper, 1, 2, ...) over the moves not yet settled, so a move settled at a shallow level is never read deeper and no depth is a tuned count.
//!
//! WHAT THE OTHER SIDE KEEPS (soundness). A claim carried out leaves the other side its move, and a line that only holds leaves it its move as well: the value of a reply line is reduced by
//! `their_gain` -- the other side's best single exchange or its goal forced in one on the position it then has (with us to move, on the position after our pass: the other side's threats as they stand if we did
//! nothing -- the bound is pessimistic about what we can parry and that is the sound side; a parry read as the least over our moves was tried and was less sound, 13 of 844 against 5 of 520), or its moves that put our goal under the sign followed by the best of what it then takes (`their_sign_gain`). THE NAMED LIMIT of this: the other side's gain
//! is read as ONE step; a gain that needs a second step of its own after a taking (a taking that also strikes a second unit, a move in between before a recapture, a quiet fork) is not seen,
//! and the exchange frame itself reads the takings on one place only. A claim the ledger search bears out is a measure of this (forcing_soundness_measure.py), never assumed.
//!
//! No object of any game is named here: the game gives units, moves, takings, the pass and its outcome through `Game` and `Placed`, the worth by the sheet's table.
use super::frame::solve_real;
use super::reading::Reading;
use super::walk::BIG;
use super::{Kind, Place, Side, Unit};
use crate::ledger::{Judge, Search, Worth, WIN};
use crate::{wins_in, won, Cut};
use cs_game::placed::Placed;
use cs_game::{Game, Role};
use cs_operator::Budget;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

type R<T> = Result<T, Cut>;

/// The fuse of the forcing read, as default: positions read over all the levels of one position.
pub const FUSE_POSITIONS: u64 = 20_000;

/// PLAN 78.47 E. The ledger search that has the last word on a CONFIRMED verdict: a node fuse and a depth cap, both named. A search the fuse cut, or a depth over the cap, is no test (the verdict
/// stands, marked unchecked); it is never counted as bearing the claim out.
pub const CHECK_NODES: u64 = 5_000_000;
pub const CHECK_DEPTH_CAP: i32 = 14;

#[derive(Debug, Clone)]
pub struct ForcingParams {
    /// Positions read over the whole read (a named fuse).
    pub fuse: u64,
    /// Nodes one `wins_in` call may spend.
    pub goal_nodes: u64,
    /// Nodes one exchange frame may expand (the sheet's own).
    pub frame_budget: u64,
    /// PLAN 78.47 B. A reply that touches nothing the read looked at is answered from the stored result of an earlier reply of the same move (no new position read). The old path (every reply read in
    /// full) stays reachable through `false`: the comparison of the two is the acceptance of that stage.
    pub shortcut: bool,
}

impl Default for ForcingParams { fn default() -> Self { ForcingParams { fuse: FUSE_POSITIONS, goal_nodes: 4000, frame_budget: 20_000, shortcut: true } } }

/// A claim as plain data: what it is on (a unit by its letter and place, or the goal), the kind of claim (takes | exchange | goal | sign | ends), its worth, the line it is carried out by.
#[derive(Debug, Clone)]
pub struct ClaimOut { pub target: String, pub kind: &'static str, pub value: f64, pub line: Vec<String> }

/// One reply of the other side to a threat move and what stands after it.
#[derive(Debug, Clone)]
pub struct ReplyOut {
    pub reply: String,
    /// What the reply took of ours: (the kind's letter, the place), and its worth.
    pub took: Vec<(String, String)>,
    pub took_worth: f64,
    /// The claim that stands after the reply (the best), None where none does.
    pub stands: Option<ClaimOut>,
    /// Where nothing stood: the mover's next threat move and its value, and the line.
    pub deeper: Option<(String, f64)>,
    pub line: Vec<String>,
    pub value: f64,
    /// The line's value was not above zero and something was left unread (a threat move of ours after it, not read to its end).
    pub open: bool,
    /// The reply ends the game: "won" (for the mover) | "lost" | "drawn".
    pub ends: Option<&'static str>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict { Confirmed, NotBorneOut, NotSettled, NoClaim, Unread }

impl Verdict { pub fn word(self) -> &'static str { match self { Verdict::Confirmed => "CONFIRMED", Verdict::NotBorneOut => "NOT BORNE OUT", Verdict::NotSettled => "NOT SETTLED", Verdict::NoClaim => "NO CLAIM", Verdict::Unread => "UNREAD" } } }

/// PLAN 78.47 E. What the ledger search said of a threat move the read confirmed: "holds" | "contradicted" (the verdict is then NOT BORNE OUT) | "unchecked" (the search was cut or the depth is over the
/// cap); the value the read claims from the position after the move, the search's value, the depth asked and read, the search's line.
#[derive(Debug, Clone)]
pub struct SearchOut { pub status: &'static str, pub claimed: f64, pub value: i64, pub depth: i32, pub depth_read: i32, pub cut: bool, pub line: Vec<String>, pub nodes: u64 }

#[derive(Debug, Clone)]
pub struct ThreatOut {
    pub mv: String,
    pub verdict: Verdict,
    /// The least over the replies of the line's value (what the mover won with the move and along the line, less what it lost); `BIG` and above: the goal.
    pub value: f64,
    /// What the move itself took: (letter, place) and the worth.
    pub took: Vec<(String, String)>,
    pub took_worth: f64,
    pub claims: Vec<ClaimOut>,
    pub replies: Vec<ReplyOut>,
    /// The deepening level at which the verdict was reached (0: nothing deeper read).
    pub level: u32,
    /// The longest line (plies, the move not counted) any reply line of the answer has: the depth a test by search must cover.
    pub plies: usize,
    pub searched: Option<SearchOut>,
}

#[derive(Debug, Clone)]
pub struct ForcingRead {
    /// Every threat move found: confirmed first, then not settled, no claim (with the reply that closed it), unread; by value.
    pub threats: Vec<ThreatOut>,
    pub legal: usize,
    /// Threat moves found / confirmed / not settled / no claim / not read at all (the fuse).
    pub found: usize,
    pub confirmed: usize,
    /// Threat moves the read confirmed and the ledger search contradicted (verdict NOT BORNE OUT).
    pub not_borne_out: usize,
    pub unsettled: usize,
    pub no_claim: usize,
    pub unread: usize,
    /// Positions read (the nodes), the fuse, whether the fuse stopped the reading, the deepest level read.
    pub nodes: u64,
    pub fuse: u64,
    pub cut: bool,
    pub level: u32,
    pub cost: Cost,
}

/// PLAN 78.47 A. WHERE THE READ SPENDS: per kind of work, the positions read (the fuse's own count), the exchange-frame nodes, the nodes of the goal read, the calls answered from memory and
/// the seconds. A kind is the reason a position was asked for; `deeper` is the same work inside a threat move of ours read below a reply (level above zero).
pub const WORK: [&str; 7] = ["root", "child of our move (its claims)", "after a reply (what stands)", "the other side's gain", "the other side's sign moves", "is there a threat move left", "the other side's gain, answered from the memory of positions"];
#[derive(Debug, Clone, Default)]
pub struct Cost { pub positions: [u64; 14], pub frame_nodes: [u64; 14], pub goal_nodes: [u64; 14], pub memory: [u64; 14], pub seconds: [f64; 14],
                  /// replies read: by what the reply is (took something | did not), and how many of each were answered unchanged
                  pub replies: [u64; 2], pub replies_unchanged: [u64; 2],
                  /// replies that touch nothing the read looked at (whether or not a stored result was there to answer them), and the reasons a reply touched something
                  pub untouched: u64, pub why: [u64; 8] }

/// What the units of the side NOT to move stand to lose to the side to move (whoever it is) from a position: the exchange claims (exact ones only: a frame left open is no claim), and the goal forced in one.
struct Stand {
    claims: Vec<(Unit, f64, Vec<String>)>,
    by_unit: HashMap<(Kind, Place), f64>,
    goal: Option<String>,
}

struct Threat<M> { mv: M, claims: Vec<ClaimOut>, priority: f64, wins: bool }

#[derive(Clone)]
struct ReplyRead<M> { rm: M, took_worth: f64, ends: Option<&'static str>, stands: Option<ClaimOut>, deeper: Option<(String, f64)>, line_tail: Vec<String>, value: f64, open: bool }

struct MoveRead<M> { verdict: Verdict, value: f64, replies: Vec<ReplyRead<M>>, level: u32 }

pub struct Forcing<'a, 'g, G: Game + Placed> {
    r: &'a Reading<'g, G>,
    g: &'g G,
    fp: ForcingParams,
    judge: Worth,
    us: Role,
    them: Role,
    nodes: u64,
    fuse: u64,
    stands: HashMap<u64, Rc<Stand>>,
    threats: HashMap<u64, Rc<Vec<Threat<G::Move>>>>,
    has: HashMap<u64, bool>,
    reads: HashMap<(u64, u64, u32), Rc<MoveRead<G::Move>>>,
    /// PLAN 78.47 C: what the other side keeps at a position, read once however many orders of moves reach it.
    gains: HashMap<u64, f64>,
    /// PLAN 78.47 D: the replies that held (a line value not above zero) when a move was last read, to be read first one level deeper.
    holders: HashMap<(u64, u64), Vec<G::Move>>,
    cost: Cost,
    /// the kind of work the next position read is charged to (an index of `WORK`, plus 7 when it is read inside a deeper threat move)
    ctx: usize,
    deeper: usize,
    killer: HashMap<(u64, u64), G::Move>,
}

impl<'a, 'g, G: Game + Placed> Forcing<'a, 'g, G> {
    pub fn new(r: &'a Reading<'g, G>, fp: ForcingParams) -> Self {
        let judge = Worth { name: String::new(), table: r.m.ledger_worth.clone(), ours_first: false };
        let (us, them) = (r.mover, r.g.other(r.mover));
        let fuse = fp.fuse;
        Forcing { r, g: r.g, fp, judge, us, them, nodes: 0, fuse, stands: HashMap::new(), threats: HashMap::new(), has: HashMap::new(), reads: HashMap::new(), gains: HashMap::new(), holders: HashMap::new(), cost: Cost::default(), ctx: 0, deeper: 0, killer: HashMap::new() }
    }

    fn tick(&mut self) -> R<()> {
        self.nodes += 1;
        if self.nodes > self.fuse { Err(Cut) } else { Ok(()) }
    }

    /// What the move adds for us along the ledger's own judge (a unit of the other side taken, a unit of the mover's changing kind), `by_us` whether we are the mover.
    fn gain(&self, s: &G::State, m: &G::Move, by_us: bool) -> f64 {
        let g = self.g;
        let took: i64 = g.captured(s, m).iter().map(|(o, k)| self.judge.delta(self.us, *o, k)).sum();
        let mover = if by_us { self.us } else { self.them };
        (took + g.changed(s, m).iter().map(|(a, b)| self.judge.change(self.us, mover, a, b)).sum::<i64>()) as f64
    }

    fn letter(&self, k: Kind) -> &str { &self.r.m.letters[k as usize] }

    fn unit_text(&self, u: &Unit) -> String { format!("{}@{}", self.letter(u.kind), self.r.m.names.place_text(u.at)) }

    /// `stand`, the position read charged to the kind of work `work` (an index of `WORK`).
    fn stand_as(&mut self, x: &G::State, work: usize) -> R<Rc<Stand>> {
        self.ctx = work + self.deeper;
        self.stand(x)
    }

    /// The claims a position gives the side to move (whoever it is): see `Stand`. A position counts one node.
    fn stand(&mut self, x: &G::State) -> R<Rc<Stand>> {
        let key = self.g.key(x);
        if let Some(s) = self.stands.get(&key) { self.cost.memory[self.ctx] += 1; return Ok(s.clone()); }
        self.tick()?;
        let t0 = std::time::Instant::now();
        let ctx = self.ctx;
        self.cost.positions[ctx] += 1;
        let (g, r) = (self.g, self.r);
        let mover = g.mover(x);
        let moves = g.moves(x);
        let mut dest = vec![false; r.np()];
        for m in &moves { dest[g.move_ends(x, m).1 as usize] = true; }
        let goal_kind = r.goal_kind();
        let mut claims = vec![];
        let mut by_unit = HashMap::new();
        for (o, k, at) in g.cells(x) {
            if Some(o) == mover || Some(k) == goal_kind || !dest[at as usize] { continue; }
            let s = solve_real(r, at, x, self.fp.frame_budget, "real");
            self.cost.frame_nodes[ctx] += s.nodes;
            if s.open { continue; }                                                  // an open frame is no claim (a lower bound is not read from it)
            let side = if o == self.us { Side::Us } else { Side::Them };
            if s.value > 0.0 { claims.push((Unit { side, kind: k, at }, s.value, s.line.clone())); by_unit.insert((k, at), s.value); }
        }
        let mut goal = None;
        if let (Some(_), Some(mv)) = (goal_kind, mover) {
            let mut b = Budget::new(3600.0, self.fp.goal_nodes);
            if let Ok(Some(m)) = wins_in(g, x, mv, 1, &mut b) { goal = Some(g.move_name(x, &m)); }
            self.cost.goal_nodes[ctx] += b.nodes;
        }
        self.cost.seconds[ctx] += t0.elapsed().as_secs_f64();
        let st = Rc::new(Stand { claims, by_unit, goal });
        self.stands.insert(key, st.clone());
        Ok(st)
    }

    /// WHAT THE OTHER SIDE GAINS BY ITS NEXT MOVE at `e`, as the read can see it (never below zero; the goal above every worth): with the other side to move, its best exchange or its goal
    /// forced in one; with us to move, the same on the position after OUR PASS (the other side's threats as they stand if we did nothing); where the game gives no pass (we are under the goal's
    /// sign) the least over our own moves of the same on the child. A line's end is only worth what is left after this: a claim carried out leaves the other side its move.
    fn their_gain(&mut self, e: &G::State) -> R<f64> {
        let key = self.g.key(e);
        if let Some(v) = self.gains.get(&key) { self.cost.memory[6 + self.deeper] += 1; return Ok(*v); }
        let v = self.their_gain_read(e)?;
        self.gains.insert(key, v);
        Ok(v)
    }

    fn their_gain_read(&mut self, e: &G::State) -> R<f64> {
        let g = self.g;
        if let Some(v) = g.outcome(e) { return Ok(if won(&v, self.them) { BIG } else { 0.0 }); }
        let mover = g.mover(e);
        let at = if mover == Some(self.them) { Some(e.clone()) } else { g.pass(e) };
        match at {
            Some(x) => {
                if g.outcome(&x).is_some() { return Ok(0.0); }
                let sd = self.stand_as(&x, 3)?;
                if sd.goal.is_some() { return Ok(BIG); }
                let direct = sd.claims.iter().map(|c| c.1).fold(0.0, f64::max);
                Ok(self.their_sign_gain(&x, direct)?)
            }
            None => {
                let mut least = f64::INFINITY;
                for m in g.moves(e) {
                    let c = g.apply(e, &m);
                    least = least.min(self.their_gain(&c)?);
                }
                Ok(if least.is_finite() { least } else { BIG })
            }
        }
    }

    /// THE OTHER SIDE'S MOVES THAT PUT OUR GOAL UNDER THE SIGN (we must answer them: few replies), each with every reply of ours and what the other side then takes by one exchange or by the goal forced
    /// in one: the most it is sure of (the least over our replies, our own takings counted for us). `x` has the other side to move. The same read as ours at level 0, for the moves that leave us no
    /// free move; a threat of theirs that needs a quiet preparing move is not read here (a named limit).
    /// `floor` is what the other side already has by a direct exchange: the answer is the greater of it and the sign moves' gain, so a sign move is only read as far as it can raise the answer above `floor`.
    fn their_sign_gain(&mut self, x: &G::State, floor: f64) -> R<f64> {
        let g = self.g;
        let mut best = floor.max(0.0);
        for m in g.moves(x) {
            let c = g.apply(x, &m);
            if let Some(v) = g.outcome(&c) { if won(&v, self.them) { return Ok(BIG); } continue; }
            if g.pass(&c).is_some() { continue; }                           // not a move that leaves us no free move
            let theirs = -self.gain(x, &m, false);
            let mut least = f64::INFINITY;
            // THE ORDER (78.47 D): our replies by what each takes, the most first: the one that leaves the other side least ends the loop soonest (the least over the replies does not depend on the order)
            let mut reps: Vec<(f64, G::Move)> = g.moves(&c).into_iter().map(|r| (self.gain(&c, &r, true), r)).collect();
            reps.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            for (mine, rep) in reps {
                let d = g.apply(&c, &rep);
                let after = match g.outcome(&d) {
                    Some(v) => if won(&v, self.them) { BIG } else if won(&v, self.us) { -BIG } else { 0.0 },
                    None => { let sd = self.stand_as(&d, 4)?; if sd.goal.is_some() { BIG } else { sd.claims.iter().map(|c| c.1).fold(0.0, f64::max) } }
                };
                least = least.min(theirs - mine + after);
                if least <= best { break; }
            }
            if least.is_finite() && least > best { best = least; }
        }
        Ok(best)
    }

    /// The position at the end of a claim's line (the moves played by their names).
    fn play_line(&self, x: &G::State, line: &[String]) -> G::State {
        let mut cur = x.clone();
        for name in line {
            match self.g.find_move(&cur, name) { Some(m) => cur = self.g.apply(&cur, &m), None => break }
        }
        cur
    }

    /// Is `m` a threat move at `x` (we are to move; `base` is what stood at `x`), and what claims it makes.
    fn threat_of(&mut self, x: &G::State, m: &G::Move, base: &Stand, work: usize) -> R<Option<Threat<G::Move>>> {
        let g = self.g;
        let c = g.apply(x, m);
        if let Some(v) = g.outcome(&c) {
            if won(&v, self.us) {
                return Ok(Some(Threat { mv: m.clone(), claims: vec![ClaimOut { target: "goal".into(), kind: "ends", value: BIG, line: vec![g.move_name(x, m)] }], priority: 2.0 * BIG, wins: true }));
            }
            return Ok(None);
        }
        let mut claims: Vec<ClaimOut> = vec![];
        let mut priority = 0.0f64;
        let took = self.gain(x, m, true);
        for (o, k, at) in g.taken_at(x, m) {
            if o == self.us { continue; }
            claims.push(ClaimOut { target: format!("{}@{}", k, at), kind: "takes", value: took, line: vec![] });
        }
        if !claims.is_empty() { priority = took.max(0.0); }
        match g.pass(&c) {
            Some(p2) => {
                let sd = self.stand_as(&p2, work)?;
                for (u, v, line) in &sd.claims {
                    let before = base.by_unit.get(&(u.kind, u.at)).copied().unwrap_or(0.0);
                    if *v > before {
                        claims.push(ClaimOut { target: self.unit_text(u), kind: "exchange", value: *v - before, line: line.clone() });
                        priority = priority.max(took.max(0.0) + *v - before);
                    }
                }
                if let Some(first) = &sd.goal {
                    claims.push(ClaimOut { target: "goal".into(), kind: "goal", value: BIG, line: vec![first.clone()] });
                    priority = priority.max(BIG);
                }
            }
            None => { claims.push(ClaimOut { target: "goal".into(), kind: "sign", value: 0.0, line: vec![] }); priority = priority.max(0.5); }
        }
        if claims.is_empty() { Ok(None) } else { Ok(Some(Threat { mv: m.clone(), claims, priority, wins: false })) }
    }

    /// The threat moves of `x` (we to move), best first (an order for the budget, never a verdict).
    fn threats_of(&mut self, x: &G::State, sx: &Stand) -> R<Rc<Vec<Threat<G::Move>>>> {
        let key = self.g.key(x);
        if let Some(t) = self.threats.get(&key) { return Ok(t.clone()); }
        let mut out: Vec<(Threat<G::Move>, String)> = vec![];
        for m in self.g.moves(x) {
            if let Some(t) = self.threat_of(x, &m, sx, 1)? { let n = self.g.move_name(x, &m); out.push((t, n)); }
        }
        out.sort_by(|a, b| b.0.priority.partial_cmp(&a.0.priority).unwrap().then_with(|| a.1.cmp(&b.1)));
        let rc = Rc::new(out.into_iter().map(|x| x.0).collect::<Vec<_>>());
        self.threats.insert(key, rc.clone());
        Ok(rc)
    }

    /// Does `x` (we to move) hold any threat move: the takings first, then the rest (stops at the first found).
    fn has_threat(&mut self, x: &G::State, sx: &Stand) -> R<bool> {
        let key = self.g.key(x);
        if let Some(h) = self.has.get(&key) { return Ok(*h); }
        let g = self.g;
        let moves = g.moves(x);
        let mut found = moves.iter().any(|m| !g.captured(x, m).is_empty());
        if !found {
            for m in &moves { if self.threat_of(x, m, sx, 5)?.is_some() { found = true; break; } }
        }
        self.has.insert(key, found);
        Ok(found)
    }

    /// One reply `rm` of the other side in `c` (the child of our move which took `took_m`): the line's value at `level` more levels of deepening.
    fn reply(&mut self, c: &G::State, rm: &G::Move, took_m: f64, base: &Stand, level: u32) -> R<ReplyRead<G::Move>> {
        let g = self.g;
        let line_base = took_m + self.gain(c, rm, false);
        let took_worth = -self.gain(c, rm, false);
        self.cost.replies[if g.captured(c, rm).is_empty() { 1 } else { 0 }] += 1;
        let dd = g.apply(c, rm);
        let mut rr = ReplyRead { rm: rm.clone(), took_worth, ends: None, stands: None, deeper: None, line_tail: vec![], value: line_base, open: false };
        if let Some(v) = g.outcome(&dd) {
            if won(&v, self.us) { rr.ends = Some("won"); rr.value = BIG; } else if won(&v, self.them) { rr.ends = Some("lost"); rr.value = -BIG; } else { rr.ends = Some("drawn"); }
            return Ok(rr);
        }
        let sd = self.stand_as(&dd, 2)?;
        if let Some(first) = &sd.goal {
            rr.value = BIG;
            rr.stands = Some(ClaimOut { target: "goal".into(), kind: "goal", value: BIG, line: vec![first.clone()] });
            rr.line_tail = vec![first.clone()];
            return Ok(rr);
        }
        let (o_from, o_to) = g.move_ends(c, rm);
        // what we are sure of from here: holding (nothing carried out) leaves the other side what it threatens; a claim carried out leaves it what it threatens at the end of the line
        let hold = self.their_gain(&dd)?;
        let mut best_rel = -hold;
        for cl in &sd.claims {
            let origin = if cl.0.at == o_to { o_from } else { cl.0.at };
            let before = base.by_unit.get(&(cl.0.kind, origin)).copied().unwrap_or(0.0);
            let credit = cl.1 - before;
            if credit <= 0.0 { continue; }
            let end = self.play_line(&dd, &cl.2);
            let val = credit - self.their_gain(&end)?;
            if val > best_rel {
                best_rel = val;
                rr.stands = Some(ClaimOut { target: self.unit_text(&cl.0), kind: "exchange", value: credit, line: cl.2.clone() });
                rr.line_tail = cl.2.clone();
            }
        }
        rr.value = line_base + best_rel;
        if rr.value > 0.0 { return Ok(rr); }
        if level == 0 {
            rr.open = self.has_threat(&dd, &sd)?;
            return Ok(rr);
        }
        // where nothing standing is enough: our own next threat moves in the position after the reply, one level deeper
        let ths = self.threats_of(&dd, &sd)?;
        let mut open = false;
        for th in ths.iter() {
            let was = self.deeper;
            self.deeper = 7;
            let mr = self.move_read(&dd, &th.mv, &sd, level - 1);
            self.deeper = was;
            let mr = mr?;
            match mr.verdict {
                Verdict::Confirmed => {
                    if line_base + mr.value > rr.value {
                        rr.value = line_base + mr.value;
                        let name = g.move_name(&dd, &th.mv);
                        rr.deeper = Some((name.clone(), mr.value));
                        rr.line_tail = vec![name];
                    }
                    if rr.value > 0.0 { break; }
                }
                Verdict::NotSettled => open = true,
                _ => {}
            }
        }
        rr.open = rr.value <= 0.0 && open;
        Ok(rr)
    }

    /// The units of a position with the places each bears on (the board as it stands), by place.
    fn unit_sets(&self, s: &G::State) -> Vec<(Role, Kind, Place, Vec<Place>)> {
        let g = self.g;
        let mut v: Vec<(Role, Kind, Place, Vec<Place>)> = g.cells(s).into_iter().map(|(o, k, at)| (o, k, at, g.bearing_from(s, o, k, at))).collect();
        v.sort_by_key(|u| u.2);
        v
    }

    /// PLAN 78.47 B. Does the reply `rm` (c -> dd) touch NOTHING the read of this move looked at? Yes only where all of these hold, each read from the game's own units and the places they bear on:
    /// it takes nothing, ends no game, and only moves one unit from its place to another (no other unit changes, none changes its kind); the unit is not the goal's; no unit of the board other than
    /// the mover bears on a different set of places than before (so no line of either side is opened or blocked, no guard of a place is lost or gained); neither the place it leaves nor the one it
    /// enters is borne on by a unit of ours (it is no new target and opens no line of ours); the places it bears on, before and after, hold no unit at all (it guards nothing, strikes nothing of ours, is
    /// no threat of its own and no part of an exchange). Such a reply leaves every claim and every standing of the other side exactly as the read found them.
    /// -> 0 when it touches nothing, else the reason it does (1 it takes something or ends the game, 2 it is not a plain move of one unit that no exchange involves, 3 another unit's board changes,
    /// 4 a line of some unit is opened or blocked, 5 it lands on or leaves a place of ours, 6 it bears on a hot place, 7 after one more move of its own it would).
    fn untouched(&self, c: &G::State, cs: &[(Role, Kind, Place, Vec<Place>)], inv: &HashSet<Place>, hot: &HashSet<Place>, rm: &G::Move, dd: &G::State) -> usize {
        let g = self.g;
        if !g.captured(c, rm).is_empty() || g.outcome(dd).is_some() { return 1; }
        let (f, t) = g.move_ends(c, rm);
        let Some(rep) = cs.iter().find(|u| u.2 == f) else { return 2 };
        if Some(rep.1) == self.r.goal_kind() || inv.contains(&f) { return 2; }
        let ds = self.unit_sets(dd);
        if ds.len() != cs.len() { return 3; }
        let mut moved: Option<&(Role, Kind, Place, Vec<Place>)> = None;
        for u in ds.iter() {
            if u.2 == t {
                if u.0 != rep.0 || u.1 != rep.1 { return 3; }
                moved = Some(u);
                continue;
            }
            match cs.iter().find(|x| x.2 == u.2) {
                Some(x) if x.0 == u.0 && x.1 == u.1 => {
                    // a unit outside the read whose lines change may do so only among places the read is not about and where nothing of ours stands
                    if x.3 != u.3 && (inv.contains(&u.2) || x.3.iter().chain(u.3.iter()).any(|p| x.3.contains(p) != u.3.contains(p) && (inv.contains(p) || cs.iter().any(|w| w.2 == *p && w.0 == self.us)))) { return 4; }
                }
                _ => return 3,
            }
        }
        let Some(m) = moved else { return 3 };
        if cs.iter().any(|x| x.0 == self.us && (x.3.contains(&f) || x.3.contains(&t))) { return 5; }
        // what it bears on, before and after, and what it could bear on after ONE more move of its own (its next move may be a threat, a guard of a landing place, a blocking of a line): none of it a hot place
        if rep.3.iter().any(|p| hot.contains(p)) || m.3.iter().any(|p| hot.contains(p)) { return 6; }
        let reach = g.reach_from(dd, m.0, m.1, t);
        if reach.iter().any(|q| hot.contains(q) || g.bearing_from(dd, m.0, m.1, *q).iter().any(|p| hot.contains(p))) { return 7; }
        0
    }

    /// PLAN 78.47 B. For the move whose child is `c`: (the units of `c` with the places each bears on, the places the read is about (`involved`), the HOT places): the places of the units of ours and of the
    /// read's, the places any move of ours ends on in the position where we move again (where a unit of ours lands: a taking, a line opened to the goal, a guard needed), and the places the goal unit
    /// of either side may step to. None where the game gives no pass for the child (what stands is then read on the replies themselves): nothing is answered from a stored result there.
    fn hot_places(&self, c: &G::State) -> Option<(Vec<(Role, Kind, Place, Vec<Place>)>, HashSet<Place>, HashSet<Place>)> {
        let g = self.g;
        let p2 = g.pass(c)?;
        let cs = self.unit_sets(c);
        let inv = self.involved(&cs);
        let mut hot: HashSet<Place> = inv.clone();
        for u in cs.iter().filter(|u| u.0 == self.us) { hot.insert(u.2); }
        for m in g.moves(&p2) { hot.insert(g.move_ends(&p2, &m).1); }
        if let Some(k) = self.r.goal_kind() {
            for u in cs.iter().filter(|u| u.1 == k) { hot.extend(g.reach_from(c, u.0, u.1, u.2)); }
        }
        Some((cs, inv, hot))
    }

    /// PLAN 78.47 B. THE PLACES THE READ OF A MOVE IS ABOUT, from the board after our move and the places each unit bears on: for each unit either side bears on (the target of an exchange) and each
    /// side's goal unit, the unit itself and the units that bear on its place (strikers and defenders); then every unit of the SAME side as one of those that bears on its place (a unit behind a striker joins
    /// after a taking, a protector), again and again; then, for each goal unit, the units of the other side that bear on the place of any of its own side's units in the set (a unit that pins one).
    /// A unit outside this set takes no part in any exchange, claim or goal read from here.
    fn involved(&self, cs: &[(Role, Kind, Place, Vec<Place>)]) -> HashSet<Place> {
        let goal = self.r.goal_kind();
        let mut set: HashSet<Place> = HashSet::new();
        for u in cs.iter().filter(|u| Some(u.1) == goal || cs.iter().any(|w| w.0 != u.0 && w.3.contains(&u.2))) {
            set.insert(u.2);
            for w in cs.iter().filter(|w| w.3.contains(&u.2)) { set.insert(w.2); }
        }
        loop {
            let mut add: Vec<Place> = vec![];
            for s in cs.iter().filter(|s| set.contains(&s.2)) {
                for z in cs.iter().filter(|z| z.0 == s.0 && z.3.contains(&s.2) && !set.contains(&z.2)) { add.push(z.2); }
            }
            if add.is_empty() { break; }
            set.extend(add);
        }
        for g in cs.iter().filter(|u| Some(u.1) == goal) {
            let own: Vec<Place> = cs.iter().filter(|s| s.0 == g.0 && set.contains(&s.2)).map(|s| s.2).collect();
            for sp in own {
                for z in cs.iter().filter(|z| z.0 != g.0 && z.3.contains(&sp)) { set.insert(z.2); }
            }
        }
        set
    }

    /// The read of our move `m` at `x` (`base` = what stood at `x`), every reply, `level` levels of deepening below it. NoClaim stops at the first reply that closes it.
    fn move_read(&mut self, x: &G::State, m: &G::Move, base: &Stand, level: u32) -> R<Rc<MoveRead<G::Move>>> {
        let g = self.g;
        let c = g.apply(x, m);
        let ck = (g.key(x), g.key(&c), level);
        if let Some(r) = self.reads.get(&ck) { return Ok(r.clone()); }
        let took_m = self.gain(x, m, true);
        let mut replies: Vec<(G::Move, f64)> = g.moves(&c).into_iter().map(|rm| { let t = self.gain(&c, &rm, false); (rm, t) }).collect();
        // THE ORDER (78.47 D): the replies that held when this move was last read first (the holding reply first), the one that closed it before all of them, then by what the reply takes
        let kill = self.killer.get(&(ck.0, ck.1)).cloned();
        let held: Vec<G::Move> = self.holders.get(&(ck.0, ck.1)).cloned().unwrap_or_default();
        replies.sort_by(|a, b| (Some(&a.0) != kill.as_ref()).cmp(&(Some(&b.0) != kill.as_ref()))
            .then_with(|| (!held.contains(&a.0)).cmp(&!held.contains(&b.0))).then_with(|| a.1.partial_cmp(&b.1).unwrap()));
        let mut reads: Vec<ReplyRead<G::Move>> = vec![];
        let (mut least, mut closed_fail, mut any_open) = (f64::INFINITY, false, false);
        let c_sets = if self.fp.shortcut { self.hot_places(&c).map(|(cs, inv, hot)| (cs, inv, hot)) } else { None };
        let mut proto: Option<ReplyRead<G::Move>> = None;
        for (rm, _) in replies {
            let mut touches = true;                                            // does the reply touch anything the read looked at? (78.47 B)
            let mut rr_opt: Option<ReplyRead<G::Move>> = None;
            if let Some((cs, inv, hot)) = &c_sets {
                let dd = g.apply(&c, &rm);
                let why = self.untouched(&c, cs, inv, hot, &rm, &dd);
                touches = why != 0;
                self.cost.why[why] += 1;
                if !touches { self.cost.untouched += 1; }
                if !touches {
                    if let Some(pr) = &proto {
                        let mut rr = pr.clone();
                        rr.rm = rm.clone();
                        self.cost.replies[if g.captured(&c, &rm).is_empty() { 1 } else { 0 }] += 1;
                        self.cost.replies_unchanged[1] += 1;
                        rr_opt = Some(rr);
                    }
                }
            }
            let rr = match rr_opt { Some(rr) => rr, None => {
                let rr = self.reply(&c, &rm, took_m, base, level)?;
                // the first reply of the move that touches nothing is the stored result the others are answered from: only where its reading went no deeper than the claims that stand
                // (a read that went into our next threat moves is of that position alone)
                if !touches && proto.is_none() && rr.deeper.is_none() && (rr.value > 0.0 || level == 0) && !(rr.value <= 0.0 && !rr.open) { proto = Some(rr.clone()); }
                rr
            } };
            least = least.min(rr.value);
            if rr.value <= 0.0 {
                if rr.open { any_open = true; } else { closed_fail = true; self.killer.insert((ck.0, ck.1), rm.clone()); reads.push(rr); break; }
            }
            reads.push(rr);
        }
        let verdict = if closed_fail { Verdict::NoClaim } else if least > 0.0 { Verdict::Confirmed } else if any_open { Verdict::NotSettled } else { Verdict::NoClaim };
        self.holders.insert((ck.0, ck.1), reads.iter().filter(|r| r.value <= 0.0).map(|r| r.rm.clone()).collect());
        let mr = Rc::new(MoveRead { verdict, value: if reads.is_empty() { took_m } else { least }, replies: reads, level });
        self.reads.insert(ck, mr.clone());
        Ok(mr)
    }

    /// THE FORCING READ of the position: every threat move, deepened level by level over the moves not yet settled, within the fuse.
    pub fn read(&mut self) -> ForcingRead {
        let (g, r) = (self.g, self.r);
        let x = r.s.clone();
        let moves = g.moves(&x);
        let mut out = ForcingRead { threats: vec![], legal: moves.len(), found: 0, confirmed: 0, not_borne_out: 0, unsettled: 0, no_claim: 0, unread: 0, nodes: 0, fuse: self.fuse, cut: false, level: 0, cost: Cost::default() };
        let Ok(base) = self.stand_as(&x, 0) else { out.cut = true; out.nodes = self.nodes; out.cost = self.cost.clone(); return out; };
        let mut cut = false;
        let mut found: Vec<(Threat<G::Move>, String)> = vec![];
        for m in &moves {
            match self.threat_of(&x, m, &base, 1) {
                Ok(Some(t)) => { let n = g.move_name(&x, m); found.push((t, n)); }
                Ok(None) => {}
                Err(_) => { cut = true; break; }
            }
        }
        found.sort_by(|a, b| b.0.priority.partial_cmp(&a.0.priority).unwrap().then_with(|| a.1.cmp(&b.1)));
        let mut state: Vec<(Verdict, Option<Rc<MoveRead<G::Move>>>)> = vec![(Verdict::Unread, None); found.len()];
        let mut level = 0u32;
        'deepen: loop {
            let mut pending = false;
            let before = self.nodes;
            // THE ORDER (78.47 D): the move with the fewest holding replies first (the proof-number idea in its plainest form); equal counts keep the order of the priorities
            let mut order: Vec<usize> = (0..found.len()).collect();
            let holding = |st: &(Verdict, Option<Rc<MoveRead<G::Move>>>)| st.1.as_ref().map_or(0, |m| m.replies.iter().filter(|r| r.value <= 0.0).count());
            order.sort_by_key(|&i| holding(&state[i]));
            for i in order {
                let t = &found[i].0;
                if !matches!(state[i].0, Verdict::Unread | Verdict::NotSettled) { continue; }
                if t.wins {                       // the move ends the game in our favour: nothing to read
                    state[i] = (Verdict::Confirmed, None);
                    continue;
                }
                match self.move_read(&x, &t.mv, &base, level) {
                    Ok(mr) => { state[i] = (mr.verdict, Some(mr.clone())); if mr.verdict == Verdict::NotSettled { pending = true; } }
                    Err(_) => { cut = true; break 'deepen; }
                }
            }
            out.level = level;
            if !pending || cut { break; }
            if self.nodes == before && level > 0 { break; }                              // a level that read no new position: deeper reads nothing more
            level += 1;
        }
        out.cut = cut;
        out.found = found.len();
        for (i, (t, name)) in found.iter().enumerate() {
            let (mut verdict, mr) = (state[i].0, state[i].1.clone());
            let took_w = self.gain(&x, &t.mv, true);
            let took: Vec<(String, String)> = g.taken_at(&x, &t.mv).into_iter().filter(|(o, _, _)| *o != self.us).map(|(_, k, at)| (k, at)).collect();
            if t.wins {
                out.confirmed += 1;
                out.threats.push(ThreatOut { mv: name.clone(), verdict, value: BIG, took, took_worth: took_w, claims: t.claims.clone(), replies: vec![], level: 0, plies: 0, searched: None });
                continue;
            }
            let Some(mr) = mr else {
                match verdict { Verdict::NotSettled => out.unsettled += 1, Verdict::NoClaim => out.no_claim += 1, _ => out.unread += 1 }
                out.threats.push(ThreatOut { mv: name.clone(), verdict, value: 0.0, took, took_worth: took_w, claims: t.claims.clone(), replies: vec![], level: 0, plies: 0, searched: None });
                continue;
            };
            let c = g.apply(&x, &t.mv);
            let mut replies = vec![];
            let mut plies = 0;
            for rr in &mr.replies {
                let rname = g.move_name(&c, &rr.rm);
                let took_r: Vec<(String, String)> = g.taken_at(&c, &rr.rm).into_iter().filter(|(o, _, _)| *o == self.us).map(|(_, k, at)| (k, at)).collect();
                let mut line = vec![rname.clone()];
                line.extend(rr.line_tail.iter().cloned());
                plies = plies.max(line.len());
                replies.push(ReplyOut { reply: rname, took: took_r, took_worth: rr.took_worth, stands: rr.stands.clone(), deeper: rr.deeper.clone(), line, value: rr.value, open: rr.open, ends: rr.ends });
            }
            // THE SEARCH AS THE LAST WORD (78.47 E): a confirmed move stays confirmed only if the ledger search from the position after it, to the depth that covers the read's longest line
            // (the reply first, two plies per level of deepening, one more), does not contradict what the read claims from there; a cut search, or a depth over the cap, is no test
            let mut searched = None;
            if verdict == Verdict::Confirmed {
                let depth = (plies + 2 * mr.level as usize + 1) as i32;
                let claimed = if mr.value >= BIG / 2.0 { mr.value } else { mr.value - took_w };
                let ck = if depth > CHECK_DEPTH_CAP {
                    SearchOut { status: "unchecked", claimed, value: 0, depth, depth_read: 0, cut: false, line: vec![], nodes: 0 }
                } else {
                    let (value, depth_read, cutd, line, nodes) = ledger_check(r, &x, &t.mv, depth, CHECK_NODES);
                    let status = if cutd { "unchecked" } else if (mr.value >= BIG / 2.0 && value >= WIN - 1000) || (mr.value < BIG / 2.0 && value as f64 >= claimed - 1e-9) { "holds" } else { "contradicted" };
                    SearchOut { status, claimed, value, depth, depth_read, cut: cutd, line, nodes }
                };
                if ck.status == "contradicted" { verdict = Verdict::NotBorneOut; }
                searched = Some(ck);
            }
            match verdict { Verdict::Confirmed => out.confirmed += 1, Verdict::NotBorneOut => out.not_borne_out += 1, Verdict::NotSettled => out.unsettled += 1, Verdict::NoClaim => out.no_claim += 1, Verdict::Unread => out.unread += 1 }
            out.threats.push(ThreatOut { mv: name.clone(), verdict, value: mr.value, took, took_worth: took_w, claims: t.claims.clone(), replies, level: mr.level, plies, searched });
        }
        let rank = |v: Verdict| match v { Verdict::Confirmed => 0u8, Verdict::NotBorneOut => 1, Verdict::NotSettled => 2, Verdict::NoClaim => 3, Verdict::Unread => 4 };
        out.threats.sort_by(|a, b| rank(a.verdict).cmp(&rank(b.verdict)).then_with(|| b.value.partial_cmp(&a.value).unwrap()).then_with(|| a.mv.cmp(&b.mv)));
        out.nodes = self.nodes;
        out.cost = self.cost.clone();
        out
    }
}

/// THE TEST BY SEARCH of a move of the side to move at `s`: the ledger search (judge: the sheet's worth table, no pruning heuristics) from the position after the move, deepened to `depth`
/// plies under a node fuse -> (the value for the mover of the future from there, the depth read in full, whether the fuse cut the reading before `depth`, the search's line, nodes).
pub fn ledger_check<G: Game + Placed>(r: &Reading<G>, s: &G::State, mv: &G::Move, depth: i32, nodes: u64) -> (i64, i32, bool, Vec<String>, u64) {
    let g = r.g;
    let judge = Worth { name: String::new(), table: r.m.ledger_worth.clone(), ours_first: false };
    let c = g.apply(s, mv);
    let mut se = Search::with_fuse(g, &judge, r.mover, nodes);
    se.prune = false;
    let line = se.run(&c, depth);
    let cut = se.cut() || (line.depth < depth && line.value.abs() < WIN - 1000);          // (a game decided stops the deepening: that is no cut)
    (line.value, line.depth, cut, line.line, line.nodes)
}

/// The ledger's own number for a won game (what a goal claim is checked against).
pub fn win_value() -> i64 { WIN }
