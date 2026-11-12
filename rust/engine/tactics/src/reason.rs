//! PLAN 78.25 THE REASON OPERATOR: the reasoning chain the owner described -- first, do not lose the game; see what is about to be lost and why; set aside every unit
//! that has nothing to do with it; and when a threat cannot be parried, make a greater one ("a greater threat than the unit under attack"). Game-free: it reads the
//! `Game` trait and the ledger search only; nothing here names a unit or a place.
//!
//! The chain, each step by the judge and the rules:
//!  1. THREATS against the side to move `me`: the other side's tactics with no defect on the state after a pass (what they win if we do nothing) -- the census's own
//!     reading, sorted by STAKE (the ledger value of the line by the judge; the game's own goal is the top stake). If the game gives no pass the side MUST ANSWER: no
//!     threat is listed and every legal move is read as an answer.
//!  2. The FRAME of a threat: the base facts its line removes or adds (facts, not places), the shared state (what every legal move changes) left out.
//!  3. ANSWERS: the exact net of each move of ours (the ledger value for `me`, one ply deeper than the threats were read, so that their threat fits inside the reading
//!     after our move). A move answers the top threat when its net beats what passing is worth, `-stake`. Each answer is then asked ONE question: after it, can they
//!     still carry the threat out (is the threat's first move still legal, and does its line still take the targets)? No: a PARRY (the threat is gone). Yes, and
//!     carrying it out no longer gains them (our net if they do is at least nothing): a COUNTER ("a greater threat": the threat stands and is outbid). Yes, and it
//!     still costs us, only less than passing: the move LESSENS the threat. A move that does not answer LEAVES the threat.
//!  4. The UNIT of a move: the facts of `me`'s own it removes. A unit is RELEVANT when one of its moves answers or one of its facts is in the frame, else IRRELEVANT.
//!  5. With NO threat against us: our own tactics (works now, then the free moves that create one), as the candidates.
//! The reasoning is an ORDER and an explanation, never a verdict: the ledger's value decides. When the budget cuts a reading the status is a GAP: the moves not read
//! are reported as unread, never as irrelevant.
use crate::census::{census_within, Tactic};
use crate::ledger::{Judge, Search, WIN};
use cs_game::{Fact, Game, Role};
use std::collections::BTreeSet;

/// The plies of one threat reading (the answers are read one deeper), when the budget names none.
pub const REASON_DEPTH: i32 = 2;
/// The node fuse of one reading of the threats and of our own tactics.
pub const REASON_FUSE: u64 = 200_000;
/// The answers' reading (every legal move at once, exact) may spend this many fuses.
pub const ANSWER_FUSE_FACTOR: u64 = 8;
/// A value this close to the game's own outcome IS the outcome (the ledger's convention).
const GOAL_BAND: i64 = 1000;
/// The facts and moves an explanation line names before it says "and n more".
const SHOWN: usize = 6;

/// What a move does to the threat.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// After the move the threat cannot be carried out: its first move is gone, or its line no longer takes the targets.
    Parry,
    /// The threat can still be carried out, but doing so loses them more than it wins: our net if they carry it out is above 0 (a greater threat).
    Counter,
    /// The threat can still be carried out and still costs us, only less than passing would.
    /// the threat can still be carried out and then wins them nothing: what they take is taken back or matched (our net if they carry it out is 0)
    Cover,
    Lessen,
    /// The side had no pass (it must answer): the move is read as an answer, ranked by its net.
    Reply,
}

impl Kind {
    pub fn word(&self) -> &'static str { match self { Kind::Parry => "parry", Kind::Counter => "counter", Kind::Cover => "cover", Kind::Lessen => "lessen", Kind::Reply => "reply" } }
}

/// A threat against us: what they win if we pass.
#[derive(Debug, Clone)]
pub struct Threat {
    /// The ledger value of its line by the judge (their side).
    pub stake: i64,
    /// The stake is the game's own outcome.
    pub goal: bool,
    /// Its targets, named as the census names them ("<kind>@<place>", or "goal").
    pub targets: Vec<String>,
    /// Its line from the state after the pass (their move first).
    pub line: Vec<String>,
    /// The facts its line removes or adds.
    pub frame: Vec<Fact>,
}

/// A move of ours that answers the top threat (or, with no pass, any legal move).
#[derive(Debug, Clone)]
pub struct Answer {
    pub index: usize,
    pub name: String,
    pub kind: Kind,
    /// The exact ledger value for us of the move, at the answers' depth.
    pub net: i64,
    /// The move's best line (names, our move first).
    pub line: Vec<String>,
    /// The move's effect meets the threat's frame (a tie-breaker of the order).
    pub in_frame: bool,
    /// When the threat can still be carried out after the move: our net if they do, and that line (their move first). None: it cannot (or the side had no pass).
    pub if_executed: Option<i64>,
    pub executed_line: Vec<String>,
}

/// A unit of ours (the facts a move of ours removes) and the moves that move it.
#[derive(Debug, Clone)]
pub struct Unit { pub facts: Vec<Fact>, pub moves: Vec<usize> }

#[derive(Debug, Clone, Default)]
pub struct Report {
    /// The side has no pass: it must answer.
    pub must_answer: bool,
    /// Sorted by stake, highest first.
    pub threats: Vec<Threat>,
    /// The answers of the top threat (or every move when the side must answer), by net, highest first.
    pub answers: Vec<Answer>,
    /// The units no move of which answers or touches the threat (every move of the unit was read).
    pub irrelevant: Vec<Unit>,
    /// The moves left unread because the budget cut the reading (never reported as irrelevant).
    pub unread: Vec<usize>,
    /// How many moves do not answer the top threat (read, and net no better than passing).
    pub leaving: usize,
    /// With no threat: our own tactics (defect 0 first, then the free moves that create one).
    pub ours: Vec<Tactic>,
    /// Indices into the legal moves, in the order the reasoning proposes.
    pub candidates: Vec<usize>,
    /// The legal moves' names (index = the move's index), so that a report reads alone.
    pub names: Vec<String>,
    pub readings: usize,
    pub cut: usize,
    pub nodes: u64,
    /// The cost in parts, as measured: the threats' reading (and our own tactics'), and the answers' reading.
    pub threat_nodes: u64,
    pub answer_nodes: u64,
    pub threat_seconds: f64,
    pub answer_seconds: f64,
    /// our own census (run when there is no threat): its nodes and seconds
    pub ours_nodes: u64,
    pub ours_seconds: f64,
}

/// What a threat or tactic wins, as the explanation names it (a gain that falls past the line's end has no name).
fn what(goal: bool, targets: &[String]) -> String {
    if goal { "the game".to_string() } else if targets.is_empty() { "a gain past the end of the line".to_string() } else { targets.join(" ") }
}

/// A value as the explanation prints it.
fn show(v: i64) -> String { if v >= WIN - GOAL_BAND { "goal".into() } else if v <= -(WIN - GOAL_BAND) { "-goal".into() } else { format!("{v:+}") } }
fn show_stake(v: i64) -> String { if v >= WIN - GOAL_BAND { "goal".into() } else { v.to_string() } }

/// The facts a move removes and adds (between the base facts of the two states).
pub fn effect<G: Game>(g: &G, s: &G::State, m: &G::Move) -> (BTreeSet<Fact>, BTreeSet<Fact>) {
    let before: BTreeSet<Fact> = g.base_facts(s).into_iter().collect();
    let after: BTreeSet<Fact> = g.base_facts(&g.apply(s, m)).into_iter().collect();
    (before.difference(&after).cloned().collect(), after.difference(&before).cloned().collect())
}

/// The facts of a threat's line: replay it from `s`, union what each move removes and adds.
fn frame_of<G: Game>(g: &G, s: &G::State, line: &[String]) -> BTreeSet<Fact> {
    let mut out = BTreeSet::new();
    let mut cur = s.clone();
    for name in line {
        let Some(m) = g.find_move(&cur, name) else { break };
        let (r, a) = effect(g, &cur, &m);
        out.extend(r);
        out.extend(a);
        cur = g.apply(&cur, &m);
    }
    out
}

/// The KINDS of our units a line has the other side take. A target (the census names it "<kind>@<place>", or by its bare kind when the gain falls at the horizon) is
/// compared by its kind: a unit that left its place and is taken elsewhere is still taken.
fn taken_of_ours<G: Game>(g: &G, s: &G::State, line: &[String], me: Role) -> Vec<String> {
    let mut out = vec![];
    let mut cur = s.clone();
    for name in line {
        let Some(m) = g.find_move(&cur, name) else { break };
        for (o, k, _at) in g.taken_at(&cur, &m) { if o == me { out.push(k); } }
        cur = g.apply(&cur, &m);
    }
    out
}

/// A threat's line up to its last ledger event (a unit taken, a unit changing kind): the quiet moves the reading appended after it are no part of the threat, and
/// would put facts in its frame that have nothing to do with it. A line with no event (the game's own goal, a gain past the horizon) is kept whole.
fn to_last_event<G: Game>(g: &G, s: &G::State, line: &[String]) -> Vec<String> {
    let mut cur = s.clone();
    let mut last: Option<usize> = None;
    for (i, name) in line.iter().enumerate() {
        let Some(m) = g.find_move(&cur, name) else { break };
        if !g.captured(&cur, &m).is_empty() || !g.changed(&cur, &m).is_empty() { last = Some(i); }
        cur = g.apply(&cur, &m);
    }
    match last { Some(i) => line[..=i].to_vec(), None => line.to_vec() }
}

/// The indices of the moves named by the first move of each tactic's line (or of its first free move), in the tactics' order, no repeats.
fn first_moves<G: Game>(g: &G, s: &G::State, ts: &[&Tactic]) -> Vec<usize> {
    let ms = g.moves(s);
    let mut out: Vec<usize> = vec![];
    for t in ts {
        let name = t.free.first().or(t.line.first());
        let Some(name) = name else { continue };
        if let Some(i) = ms.iter().position(|m| g.move_name(s, m) == *name) { if !out.contains(&i) { out.push(i); } }
    }
    out
}

/// THE REASONING of the side to move at `s`: threats against it by stake, their frames, the moves that answer, the units that do not matter -- `depth` plies per threat
/// reading (the answers one deeper), `fuse` nodes per reading.
pub fn reason<G: Game, J: Judge>(g: &G, s: &G::State, judge: &J, depth: i32, fuse: u64) -> Report { reason_within(g, s, judge, depth, fuse, None) }

/// `reason` under a clock: with `seconds` given, ONE deadline holds for the whole chain (the threats' census, the answers' reading and the readings after each
/// answer, our own census when there is no threat). Whatever the clock cut is counted in `cut`: the chain is then a GAP, never a guess.
pub fn reason_within<G: Game, J: Judge>(g: &G, s: &G::State, judge: &J, depth: i32, fuse: u64, seconds: Option<f64>) -> Report {
    let mut rep = Report::default();
    let t0 = std::time::Instant::now();
    let left = |t0: std::time::Instant| seconds.map(|x| (x - t0.elapsed().as_secs_f64()).max(0.0));
    let Some(me) = g.mover(s) else { return rep };
    let moves = g.moves(s);
    rep.names = moves.iter().map(|m| g.move_name(s, m)).collect();
    let passed = g.pass(s);
    rep.must_answer = passed.is_none();

    // 1. the threats: the other side's tactics on the passed state, no defect
    if let Some(p) = &passed {
        let c = census_within(g, p, judge, 0, depth, fuse, left(t0));
        rep.readings += c.readings;
        rep.cut += c.cut;
        rep.nodes += c.nodes;
        rep.threat_nodes += c.nodes;
        let mut ts: Vec<&Tactic> = c.tactics.iter().filter(|t| t.defects == 0).collect();
        ts.sort_by_key(|t| -t.value);                                       // (stable: the census's own order breaks ties)
        for t in ts {
            // (a gain that falls at the reading's horizon has no place on the line: it is named by the kinds the ledger says are lost)
            let targets = if t.targets.is_empty() { t.wins.clone() } else { t.targets.clone() };
            let goal = t.value >= WIN - GOAL_BAND;
            let line = if goal { t.line.clone() } else { to_last_event(g, p, &t.line) };
            rep.threats.push(Threat { stake: t.value, goal, targets, line,
                                      frame: Vec::new() });
        }
    }

    // the facts of the position: who owns one is read off the roles' names (the game's own); the shared state (every legal move changes it) is no unit
    let roles = g.roles();
    let base = g.base_facts(s);
    let owner_of = |f: &Fact| roles.iter().position(|r| f.args.iter().any(|a| a == r));
    let owned_game = base.iter().any(|f| owner_of(f).is_some());
    let effects: Vec<(BTreeSet<Fact>, BTreeSet<Fact>)> = moves.iter().map(|m| effect(g, s, m)).collect();
    let mut common: BTreeSet<Fact> = BTreeSet::new();
    if effects.len() > 1 {
        let all = |e: &(BTreeSet<Fact>, BTreeSet<Fact>)| e.0.union(&e.1).cloned().collect::<BTreeSet<Fact>>();
        common = all(&effects[0]);
        for e in &effects[1..] { common = common.intersection(&all(e)).cloned().collect(); }
    }
    // (a fact that carries a role's name is some side's; a game whose facts carry none is read whole: the unit is then every fact the move removes)
    let kept = |f: &Fact| !common.contains(f) && (!owned_game || owner_of(f).is_some());
    let mine = |f: &Fact| !common.contains(f) && if owned_game { owner_of(f) == Some(me) } else { true };
    for t in rep.threats.iter_mut() {
        if let Some(p) = &passed { t.frame = frame_of(g, p, &t.line).into_iter().filter(|f| kept(f)).collect(); }
    }

    // 2. the answers: the exact net of every move of ours, one ply deeper than the threats
    rep.threat_seconds = t0.elapsed().as_secs_f64();
    let t1 = std::time::Instant::now();
    let ans_depth = depth + 1;
    let mut se = Search::with_fuse(g, judge, me, fuse.saturating_mul(ANSWER_FUSE_FACTOR));
    if let Some(sec) = left(t0) { se.resume(sec); }
    // (with no threat and a pass to make there is nothing to answer: no reading)
    let vals = if rep.threats.is_empty() && !rep.must_answer { vec![] } else { se.root_values(s, ans_depth) };
    if !vals.is_empty() || se.nodes > 0 { rep.readings += 1; rep.nodes += se.nodes; rep.answer_nodes = se.nodes; rep.answer_seconds = t1.elapsed().as_secs_f64(); if se.cut() { rep.cut += 1; } }
    let read: BTreeSet<usize> = vals.iter().map(|(i, _)| *i).collect();
    rep.unread = if rep.threats.is_empty() && !rep.must_answer { vec![] } else { (0..moves.len()).filter(|i| !read.contains(i)).collect() };

    let top = rep.threats.first().cloned();
    let threshold = top.as_ref().map(|t| if t.goal { -(WIN - GOAL_BAND) } else { -t.stake });
    let frame: BTreeSet<Fact> = top.as_ref().map(|t| t.frame.iter().cloned().collect()).unwrap_or_default();
    let mut answered: BTreeSet<usize> = BTreeSet::new();
    let mut cut_after = false;
    for (i, net) in &vals {
        let mut if_executed: Option<i64> = None;
        let mut executed_line: Vec<String> = vec![];
        let kind = match (&top, threshold) {
            (Some(t), Some(th)) => {
                if *net <= th { continue; }                                  // leaves the threat
                // can they still carry the threat out after this move? its first move, if still legal, read exactly; the targets looked for on its line
                let mut kind = Kind::Parry;
                let q = g.apply(s, &moves[*i]);
                let theirs = g.moves(&q);
                let first = t.line.first().and_then(|name| theirs.iter().position(|m| g.move_name(&q, m) == *name));
                if let (false, Some(j)) = (t.goal, first) {
                    let own = se.gain(s, &moves[*i]);
                    let after = se.root_values(&q, ans_depth - 1);
                    if se.cut() { cut_after = true; }
                    if let Some((_, v)) = after.iter().find(|(k, _)| *k == j) {
                        let (line, _) = se.line_from(&q, j, ans_depth as usize + 5);
                        let taken = taken_of_ours(g, &q, &line, me);
                        if t.targets.iter().any(|x| taken.iter().any(|k| *k == x.split('@').next().unwrap_or(""))) {
                            let exec = own + *v;
                            kind = if exec > 0 { Kind::Counter } else if exec == 0 { Kind::Cover } else { Kind::Lessen };
                            if_executed = Some(exec);
                            executed_line = line;
                        }
                    }
                }
                kind
            }
            _ => if rep.must_answer { Kind::Reply } else { continue },
        };
        let (line, _) = se.line_from(s, *i, ans_depth as usize + 6);
        let e = &effects[*i];
        let in_frame = e.0.iter().chain(e.1.iter()).any(|f| frame.contains(f));
        answered.insert(*i);
        rep.answers.push(Answer { index: *i, name: g.move_name(s, &moves[*i]), kind, net: *net, line, in_frame, if_executed, executed_line });
    }
    if cut_after { rep.cut += 1; }
    rep.nodes = rep.nodes - rep.answer_nodes + se.nodes;
    rep.answer_nodes = se.nodes;
    rep.answer_seconds = t1.elapsed().as_secs_f64();
    rep.leaving = if top.is_some() { vals.len() - answered.len() } else { 0 };
    // the order: net first, then parries, counters, covers, the moves that only lessen, then the moves whose effect meets the frame, then the move's index
    let rank = |k: Kind| match k { Kind::Parry | Kind::Reply => 0, Kind::Counter => 1, Kind::Cover => 2, Kind::Lessen => 3 };
    rep.answers.sort_by_key(|a| (-a.net, rank(a.kind), !a.in_frame, a.index));

    // 3. the units of ours and which of them the threat touches
    if top.is_some() {
        let mut units: Vec<(Vec<Fact>, Vec<usize>)> = vec![];
        for (i, e) in effects.iter().enumerate() {
            let fs: Vec<Fact> = e.0.iter().filter(|f| mine(f)).cloned().collect();
            if fs.is_empty() { continue; }
            match units.iter_mut().find(|u| u.0 == fs) { Some(u) => u.1.push(i), None => units.push((fs, vec![i])) }
        }
        for (fs, ms) in units {
            let relevant = ms.iter().any(|i| answered.contains(i)) || fs.iter().any(|f| frame.contains(f));
            let unread = ms.iter().any(|i| rep.unread.contains(i));
            if !relevant && !unread { rep.irrelevant.push(Unit { facts: fs, moves: ms }); }
        }
    }

    // 4. the candidates: the answers in order; with no threat (and a pass to make), our own tactics
    if top.is_some() || rep.must_answer {
        rep.candidates = rep.answers.iter().map(|a| a.index).collect();
    } else {
        let t2 = std::time::Instant::now();
        let c = census_within(g, s, judge, 1, depth, fuse, left(t0));
        rep.readings += c.readings;
        rep.cut += c.cut;
        rep.nodes += c.nodes;
        rep.ours_nodes = c.nodes;
        rep.ours_seconds = t2.elapsed().as_secs_f64();
        let mut now: Vec<&Tactic> = c.tactics.iter().filter(|t| t.defects == 0).collect();
        let mut soon: Vec<&Tactic> = c.tactics.iter().filter(|t| t.defects == 1).collect();
        now.sort_by_key(|t| -t.value);
        soon.sort_by_key(|t| -t.value);
        let mut cands = first_moves(g, s, &now);
        for i in first_moves(g, s, &soon) { if !cands.contains(&i) { cands.push(i); } }
        rep.candidates = cands;
        rep.ours = c.tactics;
    }
    rep
}

fn list(items: &[String]) -> String {
    if items.len() <= SHOWN { items.join(", ") } else { format!("{}, and {} more", items[..SHOWN].join(", "), items.len() - SHOWN) }
}

/// THE CHAIN as ordered plain lines: the threat, its frame, the parries, the counters, what is irrelevant, the candidates.
pub fn explain(r: &Report) -> Vec<String> {
    let total = r.names.len();
    let name = |i: usize| r.names.get(i).cloned().unwrap_or_default();
    let mut out = vec![];
    if r.must_answer {
        out.push(format!("reason: the side to move has no pass (it must answer): every legal move is read as an answer, {} of {} read", r.answers.len(), total));
    } else if r.threats.is_empty() {
        out.push("reason: no threat against us (they win nothing if we pass)".to_string());
    }
    if let Some(t) = r.threats.first() {
        let named = what(t.goal, &t.targets);
        out.push(format!("reason: if we pass they win {named} by {} (stake {})", t.line.join(" "), show_stake(t.stake)));
        if r.threats.len() > 1 {
            let rest: Vec<String> = r.threats[1..].iter().map(|x| format!("{} (stake {})", what(x.goal, &x.targets), show_stake(x.stake))).collect();
            out.push(format!("reason: other threats: {}", list(&rest)));
        }
        let fr: Vec<String> = t.frame.iter().map(|f| f.to_string()).collect();
        out.push(if fr.is_empty() { "reason: the threat's frame: no base fact (the game gives none)".to_string() } else { format!("reason: the threat's frame: {}", list(&fr)) });
        let parries: Vec<String> = r.answers.iter().filter(|a| a.kind == Kind::Parry).map(|a| format!("{} (net {})", a.name, show(a.net))).collect();
        let stands = |a: &Answer| format!("{} (net {}; if they carry it out: {}, net {})", a.name, show(a.net), a.executed_line.join(" "), a.if_executed.map(show).unwrap_or_default());
        let counters: Vec<String> = r.answers.iter().filter(|a| a.kind == Kind::Counter).map(stands).collect();
        let covers: Vec<String> = r.answers.iter().filter(|a| a.kind == Kind::Cover).map(stands).collect();
        let lessens: Vec<String> = r.answers.iter().filter(|a| a.kind == Kind::Lessen).map(stands).collect();
        out.push(if parries.is_empty() { "reason: parries: none".to_string() } else { format!("reason: parries: {}", list(&parries)) });
        if !counters.is_empty() { out.push(format!("reason: counters (the threat stands and is outbid): {}", list(&counters))); }
        if !covers.is_empty() { out.push(format!("reason: covers (the threat stands and wins nothing): {}", list(&covers))); }
        if !lessens.is_empty() { out.push(format!("reason: moves that lessen it (the threat stands and costs less): {}", list(&lessens))); }
        out.push(format!("reason: {} move(s) leave the threat (their net is no better than passing)", r.leaving));
        if !r.irrelevant.is_empty() {
            let ms: usize = r.irrelevant.iter().map(|u| u.moves.len()).sum();
            let fs: Vec<String> = r.irrelevant.iter().flat_map(|u| u.facts.iter().map(|f| f.to_string())).collect();
            out.push(format!("reason: irrelevant to the threat: {} units, {ms} moves ({})", r.irrelevant.len(), list(&fs)));
        }
    } else if r.must_answer {
        let rs: Vec<String> = r.answers.iter().map(|a| format!("{} (net {})", a.name, show(a.net))).collect();
        out.push(format!("reason: answers by net: {}", list(&rs)));
    } else {
        let now: Vec<String> = r.ours.iter().filter(|t| t.defects == 0).map(|t| format!("{} wins {} (value {})", t.line.first().cloned().unwrap_or_default(), what(t.value >= WIN - GOAL_BAND, &t.targets), show_stake(t.value))).collect();
        let soon: Vec<String> = r.ours.iter().filter(|t| t.defects == 1).map(|t| format!("{} creates {} by {} (value {})", t.free.join(" "), what(t.value >= WIN - GOAL_BAND, &t.targets), t.line.first().cloned().unwrap_or_default(), show_stake(t.value))).collect();
        out.push(if now.is_empty() { "reason: our tactics that work now: none".to_string() } else { format!("reason: our tactics that work now: {}", list(&now)) });
        out.push(if soon.is_empty() { "reason: threats we can create: none".to_string() } else { format!("reason: threats we can create: {}", list(&soon)) });
    }
    if !r.unread.is_empty() || r.cut > 0 {
        out.push(format!("reason: GAP -- the budget cut {} reading(s): {} of {total} moves unread (not irrelevant: unknown), and a threat may be missing", r.cut, r.unread.len()));
    }
    let cands: Vec<String> = r.candidates.iter().map(|i| name(*i)).collect();
    out.push(if cands.is_empty() { "reason: candidates none".to_string() } else { format!("reason: candidates {}", list(&cands)) });
    out
}
