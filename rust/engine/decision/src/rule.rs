//! PLAN 79.08 C, THE RULE FOR CHOOSING A MOVE (SYSTEM_SPEC 8 "the rule for choosing a move", 7 "Calculation and estimate", 13 items 26 and 27; piece: the wake). The FIXED rule that decides until a learned chooser exists (the chooser comes after the whole wake has logged games). The parts that are pure functions
//! of the goals, the list and the search's equal set live here so that each is tested alone; the loop (`wake::decide`) calls them in the order of section 8. Game-free: a unit is a number, a move a name.
//!
//! THE RULE (his words, section 8): evaluate the threats of the other side; where there is no threat advance toward the subgoal; where there is a threat defend and advance toward the subgoal; a big threat of theirs makes refuting it the subgoal; among the answers those that also serve our subgoal first.
//!   (a) THEIR threats that work now and are above the spread (a CONFIRMED row: its taking ends ahead by more than the record's resolution, or the end of the game) against OUR units: our ANSWERS are named: the responses the line read kept, and the moves the facts of their goal give (`answers_against`).
//!   (b) PLAN 79.09 A: for EACH side the fixes named are taken CHAIN BY CHAIN IN ORDER OF STAKE (`named_fixes`): a chain is a goal and the subgoals below it; its stake is that of the target at its top (the record's worth of the kind, the goal kind above all); from each chain the fixes of its NEAREST goal (the fewest things in its way,
//!       the stopper before the goal it stops). A far goal of high stake is never read as a line, it gives the reading its nearest stopper as the subgoal; the chain of the goal unit comes first only because its stake is the largest; the nearest goal of a small stake comes after, if the budget of line reads lasts.
//!   (c) among the moves the search holds equal: one that answers their largest standing threat, then one that is a fix of our chosen goal, then the estimates (the reasons' pick, else the search's own first move). The estimates only choose among what the reading holds equal; nothing here adds to a value (rule 7).
//!   (d) a named move is one whose LINE WAS READ and not refuted, within a budget of line reads (a cut is a gap, never a refutation): `keep_read`.
//!
//! INTERFACE (written before the bodies).
//!   `things_in_the_way(goal) -> usize`            the distinct stoppers of its unmet facts, plus one for each unmet fact with no stopper named (something is missing).
//!   `Stakes { worth, goal }`, `Stakes::read(g, s, rec, goals, goal)`   the worth of each kind of the targets, from the record, and the goal kind (the largest).
//!   `stake_of(goals, i, &Stakes) -> f64`   the stake of a goal: that of the target at the top of its chain (`chain_top`).
//!   `named_fixes(goals, side, &Stakes) -> Vec<ChainFix>`   the fixes of the nearest goal of each chain of `side`, the chains in order of stake, and (79.10 B) of the later goals of the chain in the order of their nearness (`later`, `all_moves`); `names_of(chains, count)` the names in rounds; `fixes_among(chains, equal)` the ladder's fixes: those of the first chain with an equal move.
//!   `their_threats(list, mover) -> Vec<&ThreatRow>`   the rows against the mover that work now, the largest net first (the end of the game first).
//!   `fix_moves(goal) -> Vec<String>`              the moves of a goal's fix set, then of its other fixes, each once, in that order.
//!   `keep_read(g, s, rec, mine, theirs, budget, p, share) -> KeptNames`   reads the line of each named move (ours at the position, theirs after the game's pass), in order, within `budget` reads a side; keeps those not refuted.
//!   `choose(equal, answers, fixes, picked) -> Chosen`   the rule (c).
use crate::goals::{Goal, Status};
use crate::threat_verdict::{verdict_of, Record, VerdictKind};
use crate::threats::{describe, ThreatRow};
use crate::wake::UnitRef;
use cs_game::placed::{Kind, Placed};
use cs_game::{Game, Role};
use cs_tactics::ledger::{Search, Worth};
use cs_tactics::lines::{self as L, LineParams};
use std::collections::HashSet;

/// The default number of line reads a side may spend on naming moves (a named fuse of the question budget, not tuned): a question is one move's line read.
pub const NAMED_BUDGET: usize = 12;

/// How many things stand in the way of a goal: the distinct stoppers of its unmet facts, and one for each unmet fact that names none (a unit is missing, or the end is not found).
pub fn things_in_the_way(goal: &Goal) -> usize {
    let mut stoppers: HashSet<UnitRef> = HashSet::new();
    let mut bare = 0;
    for f in goal.facts.iter().filter(|f| matches!(f.status, Status::Unmet | Status::Unread)) {
        match &f.cause { Some(c) if !c.stoppers.is_empty() => { for u in &c.stoppers { stoppers.insert(*u); } } _ => bare += 1 }
    }
    stoppers.len() + bare
}

/// The moves of a goal's fix set, then of its other fixes, each once, in that order.
pub fn fix_moves(goal: &Goal) -> Vec<String> {
    let mut out: Vec<String> = vec![];
    let mut put = |m: &String| { if !out.contains(m) { out.push(m.clone()); } };
    for e in &goal.fix_set { for m in &e.moves { put(m); } }
    for f in &goal.fixes { for m in &f.moves { put(m); } }
    out
}

/// THE WORTHS OF THE KINDS the stake is read from (SYSTEM_SPEC 7, "the stake orders goals"): the record's worth of each kind (the record is keyed by the words of the game's ledger, so the word is read from what taking the unit off the board removes: `threats::kind_worths`), a number the record measured; no worth is assumed, a kind the record does not
/// hold is worth 0; `goal` is the goal kind, which the record does not hold: it is the end of the game and the largest.
#[derive(Debug, Clone, Default)]
pub struct Stakes { pub worth: std::collections::HashMap<Kind, f64>, pub goal: Option<Kind>,
    /// PLAN 79.16 D: the largest worth the record measured (0 where it holds none): the least the goal kind's stake can be, "above every worth", used only where a stake is multiplied by a chance (`rank_key`)
    pub ceiling: f64 }

impl Stakes {
    /// The worths of the kinds of the targets of `goals` in the position `s` (the targets stand there).
    pub fn read<G: Game + Placed>(g: &G, s: &G::State, rec: &Record, goals: &[Goal], goal: Option<Kind>) -> Stakes {
        let units: Vec<UnitRef> = goals.iter().map(|x| x.target).collect();
        Stakes { worth: crate::threats::kind_worths(g, s, rec, &units), goal, ceiling: rec.worth.values().copied().max().unwrap_or(0) as f64 }
    }
    /// The stake of a target: the goal kind is above every worth.
    pub fn of(&self, target: UnitRef) -> f64 { if Some(target.kind) == self.goal { f64::INFINITY } else { self.worth.get(&target.kind).copied().unwrap_or(0.0) } }
}

fn target_stake(target: UnitRef, sk: &Stakes) -> f64 { sk.of(target) }

/// PLAN 79.16 D (SYSTEM_SPEC 8 the rule for choosing a move, 7 point 2 the estimate per goal; piece: the wake): THE ONE ORDER OF FIXES BY STAKE TIMES CHANCE, for the names (`named_fixes_by`), the fixes among the moves held equal (`fixes_among` works down the chains in this order) and the reason a move is given (`serves_ranked`).
/// `stake` the stake of the chain's top (the goal kind: infinite), `chance` the chance of its goal from the table of play (None: nought), `ceiling` the largest worth of the record. The goal kind's stake is above every worth and no number is assumed for it: here it counts as the largest worth, the least it can be, so that a goal
/// on the goal unit outranks a likelier goal on a smaller unit only when its chance is at least as large, and a fix of a goal whose chance is near nought no longer outranks a fix that is likely. Nothing here is a value: it orders what is read and what is said.
pub fn rank_key(stake: f64, chance: Option<f64>, ceiling: f64) -> f64 { (if stake.is_infinite() { ceiling } else { stake }) * chance.unwrap_or(0.0) }

/// PLAN 79.16 D: the order of two chains `a` before `b` by `rank_key` (the greater first), a goal on the goal kind before a finite goal of an equal key.
pub fn rank_cmp(a: (f64, f64), b: (f64, f64), ceiling: f64) -> std::cmp::Ordering {
    let (ka, kb) = (rank_key(a.0, Some(a.1), ceiling), rank_key(b.0, Some(b.1), ceiling));
    kb.partial_cmp(&ka).unwrap_or(std::cmp::Ordering::Equal).then(b.0.is_infinite().cmp(&a.0.is_infinite()))
}

/// The index of the goal at the top of the chain of goal `i` (the goal whose subgoal it is, and so on up): the goal of the same side whose target is the parent, the one nearest the root of the recursion. A goal with no parent is its own top.
pub fn chain_top(goals: &[Goal], i: usize) -> usize {
    let mut at = i;
    for _ in 0..goals.len() {
        let Some(par) = goals[at].parent else { break };
        match goals.iter().enumerate().filter(|(j, y)| y.target == par && y.side == goals[at].side && *j != at).min_by_key(|(_, y)| y.depth) { Some((j, _)) => at = j, None => break }
    }
    at
}

/// THE STAKE OF A GOAL (PLAN 79.09 A): the stake of the target of the goal at the top of its chain; a subgoal carries the stake of the goal it serves, so the chain of the goal unit stands above every chain of a smaller target.
pub fn stake_of(goals: &[Goal], i: usize, sk: &Stakes) -> f64 { target_stake(goals[chain_top(goals, i)].target, sk) }

/// The fixes of one chain: the goal whose fixes come first (the nearest of its chain), the goal at the top of the chain, the stake of the chain, the moves of the nearest goal, and (PLAN 79.10 B) the goals after it in the order of their nearness, each with the moves of its fixes that no nearer goal of the chain already holds.
#[derive(Debug, Clone)]
pub struct ChainFix { pub goal: usize, pub top: usize, pub stake: f64, pub moves: Vec<String>, pub later: Vec<(usize, Vec<String>)>,
    /// PLAN 79.14 C: the chance of the nearest goal where the order was by chance (None: by stake alone, as before), and whether that goal has every fact met
    pub chance: Option<f64>, pub met: bool }

impl ChainFix {
    /// Every move the chain names, the nearest goal's first.
    pub fn all_moves(&self) -> Vec<String> { let mut v = self.moves.clone(); for (_, ms) in &self.later { for m in ms { if !v.contains(m) { v.push(m.clone()); } } } v }
}

/// THE FIXES NAMED FOR A SIDE, CHAIN BY CHAIN IN ORDER OF STAKE (PLAN 79.09 A; section 13 item 26 as written, both halves; PLAN 79.10 B): the chains (a goal and the subgoals below it) of `side` are taken in order of the stake of their top, the greater first; from each chain the fixes of its NEAREST goal, the one with the fewest things in its way, the deeper
/// (the stopper) breaking a tie, then the place; AND the goals of the chain after it in the same order, which are asked when the nearest goal's fixes do not hold ("can I remove one of the defenders? if not, can I add one more attacker?": the next goal of the chain is the next question). A goal with no move among its fixes is passed over to the next. Only goals with an unmet fact count.
/// A far goal gives the reading only its nearest stopper; the chain of the largest stake is always asked first; a small chain comes after. Chains of equal stake: the nearer goal first, then the place.
pub fn named_fixes(goals: &[Goal], side: Role, sk: &Stakes) -> Vec<ChainFix> { named_fixes_by(goals, side, sk, None) }

/// PLAN 79.14 C: THE CHAINS IN THE ORDER OF STAKE TIMES CHANCE. `chance`: the chance of a goal, from the table of play (None: `named_fixes` as before, the order of stake alone). With it the chains are taken in the order of the stake of their top times the chance of the nearest goal of the chain (a goal kind, whose stake is above every worth, comes
/// before every finite product, the greater chance first among them), a goal whose every fact is met is not left out (it is the nearest of all: nothing is in its way) and its chain is named FIRST, whatever its product; equal products: the nearer goal, then the place, as before. Nothing here is a value: it orders what is read.
pub fn named_fixes_by(goals: &[Goal], side: Role, sk: &Stakes, chance: Option<&dyn Fn(&Goal) -> f64>) -> Vec<ChainFix> {
    let mut by_top: std::collections::BTreeMap<usize, Vec<usize>> = std::collections::BTreeMap::new();
    for (i, gl) in goals.iter().enumerate() {
        if gl.side != side || (gl.unmet.is_empty() && !(chance.is_some() && gl.met)) || fix_moves(gl).is_empty() { continue; }
        by_top.entry(chain_top(goals, i)).or_default().push(i);
    }
    let mut out: Vec<(ChainFix, (usize, u16))> = vec![];
    for (top, mut members) in by_top {
        members.sort_by_key(|&i| (!(chance.is_some() && goals[i].met), things_in_the_way(&goals[i]), std::cmp::Reverse(goals[i].depth), goals[i].target.at));
        let near = members[0];
        let stake = target_stake(goals[top].target, sk);
        let mut seen: Vec<String> = fix_moves(&goals[near]);
        let mut later: Vec<(usize, Vec<String>)> = vec![];
        for &i in &members[1..] {
            let ms: Vec<String> = fix_moves(&goals[i]).into_iter().filter(|m| !seen.contains(m)).collect();
            if ms.is_empty() { continue; }
            seen.extend(ms.iter().cloned());
            later.push((i, ms));
        }
        let c = chance.map(|f| f(&goals[near]));
        out.push((ChainFix { goal: near, top, stake, moves: fix_moves(&goals[near]), later, chance: c, met: goals[near].met }, (things_in_the_way(&goals[near]), goals[near].target.at)));
    }
    if chance.is_some() {
        // met first; then stake times chance by the one function `rank_cmp` (PLAN 79.16 D; the goal kind counts as the largest worth); then as before
        let ceiling = sk.ceiling;
        out.sort_by(|a, b| b.0.met.cmp(&a.0.met).then_with(|| rank_cmp((a.0.stake, a.0.chance.unwrap_or(0.0)), (b.0.stake, b.0.chance.unwrap_or(0.0)), ceiling)).then(a.1.cmp(&b.1)));
    } else {
        out.sort_by(|a, b| b.0.stake.partial_cmp(&a.0.stake).unwrap_or(std::cmp::Ordering::Equal).then(a.1.cmp(&b.1)));
    }
    out.into_iter().map(|x| x.0).collect()
}

/// THE NAMES OF A POSITION from its chains (PLAN 79.10 A, B): in ROUNDS: the first round takes the fixes of each chain's nearest goal, the chains in order of stake; the second the fixes of each chain's next goal, and so on; each move once, at most `count`. A chain whose nearest goal has many fixes does not keep
/// the next goal of the chain out of the names while the count lasts, and a chain of small stake is named after those of a greater stake in every round. A count, not a reading: nothing is read here, and the search reads what is named, in full.
pub fn names_of(chains: &[ChainFix], count: usize) -> Vec<String> {
    let mut out: Vec<String> = vec![];
    let rounds = chains.iter().map(|c| 1 + c.later.len()).max().unwrap_or(0);
    for r in 0..rounds {
        for c in chains {
            let ms: &[String] = if r == 0 { &c.moves } else { c.later.get(r - 1).map_or(&[], |x| &x.1) };
            for m in ms { if !out.contains(m) { out.push(m.clone()); } }
        }
        if out.len() >= count { break; }
    }
    out.truncate(count);
    out
}

/// The moves among those the search holds equal that the FIRST chain with any of them names (the ladder of rule (c) goes down the chains in order of stake; a chain names the moves of all its goals, the nearest first): the moves and the index of the chain; empty and None where no chain names an equal move.
pub fn fixes_among(chains: &[ChainFix], equal: &[String]) -> (Vec<String>, Option<usize>) {
    for (k, c) in chains.iter().enumerate() {
        let hit: Vec<String> = c.all_moves().into_iter().filter(|m| equal.contains(m)).collect();
        if !hit.is_empty() { return (hit, Some(k)); }
    }
    (vec![], None)
}

/// THEIR THREATS that work now (a confirmed row against the mover's units, the end of the game forced against the mover included), the end of the game first, then the larger net.
pub fn their_threats(list: &crate::threats::ConcreteList) -> Vec<&ThreatRow> {
    let mut v: Vec<&ThreatRow> = list.mover.iter().filter(|r| r.verdict == VerdictKind::Confirmed && !r.two_moves && !r.deferred).collect();
    v.sort_by(|a, b| (!a.goal, -a.net.unwrap_or(0)).cmp(&(!b.goal, -b.net.unwrap_or(0))));
    v
}

/// One named move whose line was read: the verdict of its line for the side that plays it, the net, the line and how it ended.
#[derive(Debug, Clone)]
pub struct Kept { pub name: String, pub verdict: VerdictKind, pub net: i64, pub line: Vec<String>, pub ended: String }

/// The named moves after their lines were read: ours and theirs that stand, the ones refuted (name, whether theirs), the reads made, the nodes and seconds.
#[derive(Debug, Clone, Default)]
pub struct KeptNames { pub mine: Vec<Kept>, pub theirs: Vec<Kept>, pub refuted: Vec<(String, bool, String)>, pub unlisted: Vec<(String, bool)>, pub read: usize, pub nodes: u64, pub seconds: f64 }

/// A NAMED MOVE IS ONE WHOSE LINE WAS READ AND NOT REFUTED, within a budget of reads a side (the first `budget` of the order given; the rest are `unlisted`: not read, a gap). Ours are read at the position by the mover; theirs after the game's pass by the other side (none where the game gives no pass). A line the fuse
/// cut is not refuted (a gap, never a kill). With `share` the reads of one side go through one search and one table.
pub fn keep_read<G: Game + Placed>(g: &G, s: &G::State, rec: &Record, mine: &[String], theirs: &[String], budget: usize, p: LineParams, share: bool) -> KeptNames { keep_read_by(g, s, rec, mine, theirs, budget, p, share, None) }

/// PLAN 79.09 B: `keep_read` with the clock of the step (None: the budget of reads only). The moves are read in the order given (the order of stake); past the clock a move is not read and is listed in `unlisted` (a gap, never a refutation), and a line the clock cuts is not refuted.
#[allow(clippy::too_many_arguments)]
pub fn keep_read_by<G: Game + Placed>(g: &G, s: &G::State, rec: &Record, mine: &[String], theirs: &[String], budget: usize, p: LineParams, share: bool, stop: Option<std::time::Instant>) -> KeptNames {
    let t0 = std::time::Instant::now();
    let judge = Worth { name: "record".into(), table: rec.worth.clone(), ours_first: false };
    let mut out = KeptNames::default();
    let Some(mover) = g.mover(s) else { return out };
    let passed = g.pass(s);
    for (names, who) in [(mine, false), (theirs, true)] {
        let (pos, me) = if who { match &passed { Some(q) => (q, g.other(mover)), None => { for n in names { out.unlisted.push((n.clone(), true)); } continue; } } } else { (s, mover) };
        let mut shared = Search::lean_with_fuse(g, &judge, me, 1);
        if let Some(t) = stop { shared.stop_at(t); }
        let mut seen: HashSet<&String> = HashSet::new();
        let mut n = 0;
        for name in names {
            if !seen.insert(name) { continue; }
            let Some(m) = g.find_move(pos, name) else { continue };
            if n >= budget || stop.is_some_and(|t| std::time::Instant::now() >= t) { out.unlisted.push((name.clone(), who)); continue; }
            n += 1;
            let r = if share || stop.is_some() { L::read_line_in(&mut shared, pos, &m, me, p) } else { L::read_line(g, &judge, pos, &m, me, p) };
            out.read += 1; out.nodes += r.nodes;
            let v = verdict_of(&r, rec);
            let (_, ended) = describe(&v);
            if v.kind == VerdictKind::Refuted { out.refuted.push((name.clone(), who, ended)); continue; }
            let k = Kept { name: name.clone(), verdict: v.kind, net: v.net, line: r.names.clone(), ended };
            if who { out.theirs.push(k) } else { out.mine.push(k) }
        }
    }
    out.seconds = t0.elapsed().as_secs_f64();
    out
}

/// What rule (c) chose and why, in words and as the three ladders it went down.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Chosen { pub mv: String, pub answers_their_threat: bool, pub fixes_our_goal: bool, pub why: String, pub left_after_threat: usize, pub left_after_goal: usize, pub by_estimate: bool }

/// RULE (c): among the moves the search holds equal (`equal`, the search's own first move first): those that answer their largest standing threat (`answers`) if any; then of those the ones that are a fix of our chosen goal (`fixes`) if any; then `picked` (the estimates' pick among what is left,
/// None when no estimate decides) or the search's own first move. `equal` empty gives None.
pub fn choose(equal: &[String], answers: &[String], fixes: &[String], picked: &dyn Fn(&[String]) -> Option<String>) -> Option<Chosen> {
    if equal.is_empty() { return None; }
    let a: Vec<String> = equal.iter().filter(|m| answers.contains(m)).cloned().collect();
    let (set1, by_threat) = if a.is_empty() { (equal.to_vec(), false) } else { (a, true) };
    let f: Vec<String> = set1.iter().filter(|m| fixes.contains(m)).cloned().collect();
    let (set2, by_goal) = if f.is_empty() { (set1.clone(), false) } else { (f, true) };
    let (mv, by_estimate) = if set2.len() == 1 { (set2[0].clone(), false) } else { match picked(&set2) { Some(m) => (m, true), None => (set2[0].clone(), false) } };
    let why = format!("{}{}{}", if by_threat { "answers their largest standing threat; " } else { "no equal move answers a standing threat of theirs; " },
        if by_goal { "a fix of our chosen goal; " } else { "no equal move of these is a fix of our chosen goal; " },
        if set2.len() == 1 { "the only one left" } else if by_estimate { "the estimates pick among the rest" } else { "the search's own first move among the rest" });
    Some(Chosen { mv, answers_their_threat: by_threat, fixes_our_goal: by_goal, why, left_after_threat: set1.len(), left_after_goal: set2.len(), by_estimate })
}

// ------------------------------------------------------------------------------------------------------------------------------ the reason printed for the move (PLAN 79.09 C)

/// What decided the move, in the order the reasons are tried (PLAN 79.09 C; SYSTEM_SPEC 8 "the wake gives reasons", 4 the four ways a line ends): the end of the game forced in its line; its own line wins by its own reading; it answers a threat of theirs that stands and its own line is no gain; it is a fix of a goal; the estimates picked it
/// among the moves held equal; nothing named decides it (the search's own move).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Decided { ForcedEnd, OwnGain, AnswersThreat, FixesGoal, Estimate, SearchOnly,
    /// PLAN 79.12 A: the equal reading put the search's first move lower; PLAN 79.12 B: the last layer chose (Weights) or drew (Explore)
    FirstLowered, Chooser }

/// What the reason is made from: the rule's choice, the search's value for the move and the plies it read, the move's own line as read for naming (None: not named or not read), the largest threat of theirs that stands in words (None: none stands; the end of the game forced against us is one), whether the move is among the
/// answers to it, and the goals the move is a fix of (in words, the goal of the largest stake first).
#[derive(Debug, Clone)]
pub struct WhyIn<'a> { pub chosen: &'a Chosen, pub value: i64, pub depth: i32, pub own: Option<&'a Kept>, pub standing: Option<String>, pub answers: bool, pub serves: Vec<String> }

/// THE CLAIM THAT DECIDED THE MOVE, in words: a line that wins says so and names the goal it serves; "answers a threat" only where a threat of theirs stands and the move's own line is no gain; the end of the game forced is said as such; the estimates are said to be estimates. Never a reason with no claim behind it.
pub fn why_of(w: &WhyIn) -> (Decided, String) {
    let serves = w.serves.first().map(|x| format!("; it serves the {x}")).unwrap_or_default();
    if w.value.abs() >= cs_tactics::ledger::WIN - 1000 {
        return if w.value > 0 { (Decided::ForcedEnd, format!("the end of the game is forced in its line: the search finds it within {} plies", w.depth)) }
               else { (Decided::ForcedEnd, format!("the end of the game is forced against us whatever is played (the search reads it within {} plies); this move holds the longest line", w.depth)) };
    }
    if let Some(k) = w.own.filter(|k| k.verdict == VerdictKind::Confirmed && k.net > 0) { return (Decided::OwnGain, format!("its own line wins by its own reading: net {} ({}){serves}", k.net, k.ended)); }
    // what is known of the move's own line, said as it is: not read, cut by the budget (unread), or read and no gain
    let own = match w.own { None => "its own line was not read".to_string(), Some(k) if k.verdict == VerdictKind::Unread => "its own line was cut by the budget (unread)".to_string(), Some(k) => format!("its own line is no gain (net {})", k.net) };
    if let (Some(t), true) = (&w.standing, w.answers) { return (Decided::AnswersThreat, format!("answers the threat of theirs on {t}, which stands; {own}")); }
    if w.chosen.fixes_our_goal && !w.serves.is_empty() { return (Decided::FixesGoal, format!("a fix of the {}; no threat of theirs that stands is answered by it, and {own}", w.serves[0])); }
    if w.chosen.by_estimate { return (Decided::Estimate, "no threat of theirs and no goal decides it: the reasons' score picks among the moves the search holds equal (an estimate, never a value)".to_string()); }
    (Decided::SearchOnly, "no threat of theirs and no goal decides it: the search's own first move among those it holds equal".to_string())
}

/// PLAN 79.12 A (SYSTEM_SPEC 8 "the wake gives reasons"; piece: the wake): WHAT THE SEARCH'S FIRST MOVE WAS, to say what SEPARATED the move played from it: its name, whether the equal reading kept it (None: the reading did not run), by how much the best stood above its null-window reading (None: not known), whether it answers the largest standing threat, whether it is a fix of the goal named, whether its own line wins by its own reading.
#[derive(Debug, Clone, Default)]
pub struct FirstFacts { pub name: String, pub held: Option<bool>, pub lowered_by: Option<i64>, pub answers: bool, pub fixes: bool, pub own_gain: bool }

/// THE REASON WHERE THE MOVE PLAYED IS NOT THE SEARCH'S FIRST (PLAN 79.12 A): the reason names what separated the two, in this order: the end of the game forced (as `why_of`); the equal reading put the first lower (and by how much); only the move played has a line that wins by its own reading; only it answers a standing threat; only it fixes the
/// goal named; else only the estimates are left (said as such, and where both answer or both fix, said that too). Never a class of the whole equal set.
pub fn why_against_first(w: &WhyIn, f: &FirstFacts) -> (Decided, String) {
    if w.value.abs() >= cs_tactics::ledger::WIN - 1000 { return why_of(w); }
    let serves = w.serves.first().map(|x| format!("; it serves the {x}")).unwrap_or_default();
    if f.held == Some(false) {
        let by = match f.lowered_by { Some(d) if d > 0 => format!(", {d} below the best at that reading"), _ => ", below the best at that reading".to_string() };
        return (Decided::FirstLowered, format!("the equal reading put the search's first move lower{by}; this move is held equal to the best"));
    }
    if let Some(k) = w.own.filter(|k| k.verdict == VerdictKind::Confirmed && k.net > 0) {
        if !f.own_gain { return (Decided::OwnGain, format!("only its own line wins by its own reading: net {} ({}); the search's first move has none{serves}", k.net, k.ended)); }
    }
    if let (Some(t), true) = (&w.standing, w.answers) {
        if !f.answers { return (Decided::AnswersThreat, format!("only it answers the threat of theirs on {t}, which stands; the search's first move does not")); }
    }
    if w.chosen.fixes_our_goal && !w.serves.is_empty() && !f.fixes {
        return (Decided::FixesGoal, format!("only it fixes the {}; the search's first move does not", w.serves[0]));
    }
    let both = if w.standing.is_some() && w.answers && f.answers { "both answer the threat of theirs that stands; " } else if w.chosen.fixes_our_goal && f.fixes { "both fix the goal named; " } else { "" };
    if w.chosen.by_estimate { return (Decided::Estimate, format!("nothing separates it from the search's first move but the estimates: {both}the reasons' score picks it (an estimate, never a value)")); }
    (Decided::SearchOnly, format!("nothing separates it from the search's first move: {both}no threat, goal or estimate decides"))
}
