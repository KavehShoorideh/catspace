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
use std::collections::HashMap;
use std::rc::Rc;

type R<T> = Result<T, Cut>;

/// The fuse of the forcing read, as default: positions read over all the levels of one position.
pub const FUSE_POSITIONS: u64 = 20_000;

#[derive(Debug, Clone)]
pub struct ForcingParams {
    /// Positions read over the whole read (a named fuse).
    pub fuse: u64,
    /// Nodes one `wins_in` call may spend.
    pub goal_nodes: u64,
    /// Nodes one exchange frame may expand (the sheet's own).
    pub frame_budget: u64,
}

impl Default for ForcingParams { fn default() -> Self { ForcingParams { fuse: FUSE_POSITIONS, goal_nodes: 4000, frame_budget: 20_000 } } }

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
pub enum Verdict { Confirmed, NotSettled, NoClaim, Unread }

impl Verdict { pub fn word(self) -> &'static str { match self { Verdict::Confirmed => "CONFIRMED", Verdict::NotSettled => "NOT SETTLED", Verdict::NoClaim => "NO CLAIM", Verdict::Unread => "UNREAD" } } }

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
}

#[derive(Debug, Clone)]
pub struct ForcingRead {
    /// Every threat move found: confirmed first, then not settled, no claim (with the reply that closed it), unread; by value.
    pub threats: Vec<ThreatOut>,
    pub legal: usize,
    /// Threat moves found / confirmed / not settled / no claim / not read at all (the fuse).
    pub found: usize,
    pub confirmed: usize,
    pub unsettled: usize,
    pub no_claim: usize,
    pub unread: usize,
    /// Positions read (the nodes), the fuse, whether the fuse stopped the reading, the deepest level read.
    pub nodes: u64,
    pub fuse: u64,
    pub cut: bool,
    pub level: u32,
}

/// What the units of the side NOT to move stand to lose to the side to move (whoever it is) from a position: the exchange claims (exact ones only: a frame left open is no claim), and the goal forced in one.
struct Stand {
    claims: Vec<(Unit, f64, Vec<String>)>,
    by_unit: HashMap<(Kind, Place), f64>,
    goal: Option<String>,
}

struct Threat<M> { mv: M, claims: Vec<ClaimOut>, priority: f64, wins: bool }

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
    killer: HashMap<(u64, u64), G::Move>,
}

impl<'a, 'g, G: Game + Placed> Forcing<'a, 'g, G> {
    pub fn new(r: &'a Reading<'g, G>, fp: ForcingParams) -> Self {
        let judge = Worth { name: String::new(), table: r.m.ledger_worth.clone(), ours_first: false };
        let (us, them) = (r.mover, r.g.other(r.mover));
        let fuse = fp.fuse;
        Forcing { r, g: r.g, fp, judge, us, them, nodes: 0, fuse, stands: HashMap::new(), threats: HashMap::new(), has: HashMap::new(), reads: HashMap::new(), killer: HashMap::new() }
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

    /// The claims a position gives the side to move (whoever it is): see `Stand`. A position counts one node.
    fn stand(&mut self, x: &G::State) -> R<Rc<Stand>> {
        let key = self.g.key(x);
        if let Some(s) = self.stands.get(&key) { return Ok(s.clone()); }
        self.tick()?;
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
            if s.open { continue; }                                                  // an open frame is no claim (a lower bound is not read from it)
            let side = if o == self.us { Side::Us } else { Side::Them };
            if s.value > 0.0 { claims.push((Unit { side, kind: k, at }, s.value, s.line.clone())); by_unit.insert((k, at), s.value); }
        }
        let mut goal = None;
        if let (Some(_), Some(mv)) = (goal_kind, mover) {
            let mut b = Budget::new(3600.0, self.fp.goal_nodes);
            if let Ok(Some(m)) = wins_in(g, x, mv, 1, &mut b) { goal = Some(g.move_name(x, &m)); }
        }
        let st = Rc::new(Stand { claims, by_unit, goal });
        self.stands.insert(key, st.clone());
        Ok(st)
    }

    /// WHAT THE OTHER SIDE GAINS BY ITS NEXT MOVE at `e`, as the read can see it (never below zero; the goal above every worth): with the other side to move, its best exchange or its goal
    /// forced in one; with us to move, the same on the position after OUR PASS (the other side's threats as they stand if we did nothing); where the game gives no pass (we are under the goal's
    /// sign) the least over our own moves of the same on the child. A line's end is only worth what is left after this: a claim carried out leaves the other side its move.
    fn their_gain(&mut self, e: &G::State) -> R<f64> {
        let g = self.g;
        if let Some(v) = g.outcome(e) { return Ok(if won(&v, self.them) { BIG } else { 0.0 }); }
        let mover = g.mover(e);
        let at = if mover == Some(self.them) { Some(e.clone()) } else { g.pass(e) };
        match at {
            Some(x) => {
                if g.outcome(&x).is_some() { return Ok(0.0); }
                let sd = self.stand(&x)?;
                if sd.goal.is_some() { return Ok(BIG); }
                let direct = sd.claims.iter().map(|c| c.1).fold(0.0, f64::max);
                Ok(direct.max(self.their_sign_gain(&x)?))
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
    fn their_sign_gain(&mut self, x: &G::State) -> R<f64> {
        let g = self.g;
        let mut best = 0.0f64;
        for m in g.moves(x) {
            let c = g.apply(x, &m);
            if let Some(v) = g.outcome(&c) { if won(&v, self.them) { return Ok(BIG); } continue; }
            if g.pass(&c).is_some() { continue; }                           // not a move that leaves us no free move
            let theirs = -self.gain(x, &m, false);
            let mut least = f64::INFINITY;
            for rep in g.moves(&c) {
                let d = g.apply(&c, &rep);
                let mine = self.gain(&c, &rep, true);
                let after = match g.outcome(&d) {
                    Some(v) => if won(&v, self.them) { BIG } else if won(&v, self.us) { -BIG } else { 0.0 },
                    None => { let sd = self.stand(&d)?; if sd.goal.is_some() { BIG } else { sd.claims.iter().map(|c| c.1).fold(0.0, f64::max) } }
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
    fn threat_of(&mut self, x: &G::State, m: &G::Move, base: &Stand) -> R<Option<Threat<G::Move>>> {
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
                let sd = self.stand(&p2)?;
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
            if let Some(t) = self.threat_of(x, &m, sx)? { let n = self.g.move_name(x, &m); out.push((t, n)); }
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
            for m in &moves { if self.threat_of(x, m, sx)?.is_some() { found = true; break; } }
        }
        self.has.insert(key, found);
        Ok(found)
    }

    /// One reply `rm` of the other side in `c` (the child of our move which took `took_m`): the line's value at `level` more levels of deepening.
    fn reply(&mut self, c: &G::State, rm: &G::Move, took_m: f64, base: &Stand, level: u32) -> R<ReplyRead<G::Move>> {
        let g = self.g;
        let line_base = took_m + self.gain(c, rm, false);
        let took_worth = -self.gain(c, rm, false);
        let dd = g.apply(c, rm);
        let mut rr = ReplyRead { rm: rm.clone(), took_worth, ends: None, stands: None, deeper: None, line_tail: vec![], value: line_base, open: false };
        if let Some(v) = g.outcome(&dd) {
            if won(&v, self.us) { rr.ends = Some("won"); rr.value = BIG; } else if won(&v, self.them) { rr.ends = Some("lost"); rr.value = -BIG; } else { rr.ends = Some("drawn"); }
            return Ok(rr);
        }
        let sd = self.stand(&dd)?;
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
            let mr = self.move_read(&dd, &th.mv, &sd, level - 1)?;
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

    /// The read of our move `m` at `x` (`base` = what stood at `x`), every reply, `level` levels of deepening below it. NoClaim stops at the first reply that closes it.
    fn move_read(&mut self, x: &G::State, m: &G::Move, base: &Stand, level: u32) -> R<Rc<MoveRead<G::Move>>> {
        let g = self.g;
        let c = g.apply(x, m);
        let ck = (g.key(x), g.key(&c), level);
        if let Some(r) = self.reads.get(&ck) { return Ok(r.clone()); }
        let took_m = self.gain(x, m, true);
        let mut replies: Vec<(G::Move, f64)> = g.moves(&c).into_iter().map(|rm| { let t = self.gain(&c, &rm, false); (rm, t) }).collect();
        let kill = self.killer.get(&(ck.0, ck.1)).cloned();
        replies.sort_by(|a, b| (Some(&a.0) != kill.as_ref()).cmp(&(Some(&b.0) != kill.as_ref())).then_with(|| a.1.partial_cmp(&b.1).unwrap()));
        let mut reads: Vec<ReplyRead<G::Move>> = vec![];
        let (mut least, mut closed_fail, mut any_open) = (f64::INFINITY, false, false);
        for (rm, _) in replies {
            let rr = self.reply(&c, &rm, took_m, base, level)?;
            least = least.min(rr.value);
            if rr.value <= 0.0 {
                if rr.open { any_open = true; } else { closed_fail = true; self.killer.insert((ck.0, ck.1), rm.clone()); reads.push(rr); break; }
            }
            reads.push(rr);
        }
        let verdict = if closed_fail { Verdict::NoClaim } else if least > 0.0 { Verdict::Confirmed } else if any_open { Verdict::NotSettled } else { Verdict::NoClaim };
        let mr = Rc::new(MoveRead { verdict, value: if reads.is_empty() { took_m } else { least }, replies: reads, level });
        self.reads.insert(ck, mr.clone());
        Ok(mr)
    }

    /// THE FORCING READ of the position: every threat move, deepened level by level over the moves not yet settled, within the fuse.
    pub fn read(&mut self) -> ForcingRead {
        let (g, r) = (self.g, self.r);
        let x = r.s.clone();
        let moves = g.moves(&x);
        let mut out = ForcingRead { threats: vec![], legal: moves.len(), found: 0, confirmed: 0, unsettled: 0, no_claim: 0, unread: 0, nodes: 0, fuse: self.fuse, cut: false, level: 0 };
        let Ok(base) = self.stand(&x) else { out.cut = true; out.nodes = self.nodes; return out; };
        let mut cut = false;
        let mut found: Vec<(Threat<G::Move>, String)> = vec![];
        for m in &moves {
            match self.threat_of(&x, m, &base) {
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
            for (i, (t, _)) in found.iter().enumerate() {
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
            let (verdict, mr) = (state[i].0, state[i].1.clone());
            match verdict {
                Verdict::Confirmed => out.confirmed += 1,
                Verdict::NotSettled => out.unsettled += 1,
                Verdict::NoClaim => out.no_claim += 1,
                Verdict::Unread => out.unread += 1,
            }
            let took_w = self.gain(&x, &t.mv, true);
            let took: Vec<(String, String)> = g.taken_at(&x, &t.mv).into_iter().filter(|(o, _, _)| *o != self.us).map(|(_, k, at)| (k, at)).collect();
            if t.wins {
                out.threats.push(ThreatOut { mv: name.clone(), verdict, value: BIG, took, took_worth: took_w, claims: t.claims.clone(), replies: vec![], level: 0, plies: 0 });
                continue;
            }
            let Some(mr) = mr else {
                out.threats.push(ThreatOut { mv: name.clone(), verdict, value: 0.0, took, took_worth: took_w, claims: t.claims.clone(), replies: vec![], level: 0, plies: 0 });
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
            out.threats.push(ThreatOut { mv: name.clone(), verdict, value: mr.value, took, took_worth: took_w, claims: t.claims.clone(), replies, level: mr.level, plies });
        }
        let rank = |v: Verdict| match v { Verdict::Confirmed => 0u8, Verdict::NotSettled => 1, Verdict::NoClaim => 2, Verdict::Unread => 3 };
        out.threats.sort_by(|a, b| rank(a.verdict).cmp(&rank(b.verdict)).then_with(|| b.value.partial_cmp(&a.value).unwrap()).then_with(|| a.mv.cmp(&b.mv)));
        out.nodes = self.nodes;
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
