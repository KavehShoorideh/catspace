//! PLAN 78.5 THE LEDGER SEARCH (Kaveh 2026-11-04: "I don't want us to have value for the material yet ... consider material lost as an outcome of a tactic: bishop
//! lost, queen lost ... later we will consider what is the value of the thing that's lost ... for now it's just a piece lost. And we want to avoid piece losses").
//!
//! A line's OUTCOME is its LEDGER: the units each side lost along it (owner, kind -- read through `Game::captured`, so the layer names no unit), or the game's own
//! outcome where the line ends the game. A JUDGE turns a ledger into a number; the judge is the VARIABLE: `Count` (no values: every unit the same, and a unit of ours
//! lost outweighs every unit of theirs won -- losses are avoided first) is the default; `Worth` (a valuation plugged in: a table from kind names to worths, read from
//! the store when we have one) is the other. Both are additive over captures, so the search is a minimax of the judge's sum: alpha-beta, deepened while the clock
//! lasts, the line's captures resolved at its horizon (each side may stop or go on capturing), a transposition table on the future part. A line ending the game
//! scores above (or below) every ledger. Within its horizon the answer is exact; beyond it, unread.
use cs_game::{Game, Role};
use rustc_hash::FxHashMap;
use std::collections::HashMap;
use std::time::Instant;

/// The JUDGE of a ledger: what one captured unit adds for `me` (positive: theirs lost; negative: ours lost), and what one of a side's own units changing kind adds
/// (a promotion: worth of the new kind less the old).
pub trait Judge {
    fn delta(&self, me: Role, owner: Role, kind: &str) -> i64;
    fn change(&self, me: Role, owner: Role, from: &str, to: &str) -> i64;
    fn name(&self) -> String;
    /// The plain worth of one unit of `kind` (no loss weighting: what `delta` carries for a unit of theirs).
    fn worth(&self, kind: &str) -> i64;
}

/// A borrowed judge judges as the judge does (so an operator can hold a judge the decision only lends it).
impl<J: Judge + ?Sized> Judge for &J {
    fn delta(&self, me: Role, owner: Role, kind: &str) -> i64 { (**self).delta(me, owner, kind) }
    fn change(&self, me: Role, owner: Role, from: &str, to: &str) -> i64 { (**self).change(me, owner, from, to) }
    fn name(&self) -> String { (**self).name() }
    fn worth(&self, kind: &str) -> i64 { (**self).worth(kind) }
}

/// The built-in fallback when no valuation record is given: each unit the same (+1 theirs lost, -LOSS ours lost); a change of kind is no gain.
#[derive(Clone, Copy)]
pub struct Count;
pub const LOSS: i64 = 1000;
impl Judge for Count {
    fn delta(&self, me: Role, owner: Role, _kind: &str) -> i64 { if owner == me { -LOSS } else { 1 } }
    fn change(&self, _me: Role, _owner: Role, _from: &str, _to: &str) -> i64 { 0 }
    fn worth(&self, _kind: &str) -> i64 { 1 }
    fn name(&self) -> String { "count (no record: a unit lost is a unit lost; ours first)".into() }
}

/// PLAN 78.61 THE JUDGE THAT COUNTS NOTHING: no capture adds anything, for either side, and a change of kind adds nothing. Every ledger is then worth the same, so the search's value
/// differs only where a line ends the game (`WIN` above or below every ledger): it looks for a forced end and otherwise holds all moves that do not lose by force equal.
#[derive(Clone, Copy)]
pub struct Nothing;
impl Judge for Nothing {
    fn delta(&self, _me: Role, _owner: Role, _kind: &str) -> i64 { 0 }
    fn change(&self, _me: Role, _owner: Role, _from: &str, _to: &str) -> i64 { 0 }
    fn worth(&self, _kind: &str) -> i64 { 0 }
    fn name(&self) -> String { "nothing (no unit counts: only the end of the game)".into() }
}

/// PLAN 78.77: a kind-change event of a search's line ("promote <from> to <to>", "-" for a unit that arrived from nothing) as (from kind, to kind); None for any other event.
pub fn change_event_kinds(text: &str) -> Option<(String, String)> {
    let rest = text.strip_prefix("promote ")?;
    let (a, b) = rest.split_once(" to ")?;
    Some((if a == "-" { String::new() } else { a.to_string() }, b.to_string()))
}

/// PLAN 78.77 THE ABLATION OF THE KIND-CHANGE EVENT: a judge that reads everything as `J` does except a unit changing kind, which adds nothing (`change` is 0 for both sides). The search and the line read keep
/// the same moves, captures and order; only the event "a unit changed kind" (`Game::changed`, scored by `Judge::change`) is taken out of the ledger, so its effect can be measured. Game-free.
#[derive(Clone)]
pub struct NoChange<J: Judge>(pub J);
impl<J: Judge> Judge for NoChange<J> {
    fn delta(&self, me: Role, owner: Role, kind: &str) -> i64 { self.0.delta(me, owner, kind) }
    fn change(&self, _me: Role, _owner: Role, _from: &str, _to: &str) -> i64 { 0 }
    fn worth(&self, kind: &str) -> i64 { self.0.worth(kind) }
    fn name(&self) -> String { format!("{} (a change of kind adds nothing)", self.0.name()) }
}

/// A VALUATION RECORD plugged in (the store's entry in use: its worths by kind name, `ours_first` -- our losses weighted by LOSS: avoided before any gain).
#[derive(Clone)]
pub struct Worth { pub name: String, pub table: HashMap<String, i64>, pub ours_first: bool }
impl Worth {
    fn w(&self, kind: &str) -> i64 { *self.table.get(kind).unwrap_or(&0) }
}
impl Judge for Worth {
    fn delta(&self, me: Role, owner: Role, kind: &str) -> i64 {
        let w = self.w(kind);
        if owner == me { -(w * if self.ours_first { LOSS } else { 1 }) } else { w }
    }
    fn change(&self, me: Role, owner: Role, from: &str, to: &str) -> i64 {
        let d = self.w(to) - if from.is_empty() { 0 } else { self.w(from) };
        if owner == me { d } else { -d }
    }
    fn worth(&self, kind: &str) -> i64 { self.w(kind) }
    fn name(&self) -> String { format!("{} {:?}{}", self.name, self.table, if self.ours_first { ", ours first" } else { "" }) }
}

pub const WIN: i64 = 1 << 40;
const NULL_MIN_DEPTH: i32 = 3;          // the null move at nodes this deep or deeper
const NULL_R: i32 = 2;                  // ... read this much shallower
const ZUGZWANG_UNITS: usize = 3;        // ... only while the mover keeps more units than this
const LMR_MIN_DEPTH: i32 = 3;           // late quiet moves reduced at nodes this deep or deeper
const LMR_AFTER: usize = 3;             // ... after this many moves read in full
const FEW_REPLIES: usize = 2;           // a node with this few legal moves is a forced answer: the line is not shortened there
pub const NODE_FUSE: u64 = 200_000_000;
pub const QUIET_FUSE: u32 = 16;           // the deepest capture run resolved at a horizon (a named fuse: captures end sooner)

/// A caller's ORDER of the root moves (plan 78.51): indices of the root's legal moves, first read first; `strict` replaces the root's own order, else it breaks ties inside it (see `Search::root_pos`).
#[derive(Clone, Debug)]
pub struct RootOrder { pub order: Vec<usize>, pub strict: bool }

#[derive(Clone, Copy)]
enum Bound { Exact, Lower, Upper }

struct Tt { depth: i32, value: i64, bound: Bound, best: usize }

pub struct Search<'a, G: Game, J: Judge> {
    pub g: &'a G,
    pub judge: &'a J,
    pub me: Role,
    pub nodes: u64,
    fuse: u64,
    deadline: Instant,
    tt: FxHashMap<u64, Tt>,
    stopped: bool,
    /// INSTANCE KNOWLEDGE (plan 78.8): the keys of the positions the game's record already holds -- reached again, the rules' draw (repetition) is the outcome
    /// PLAN 79.22 PART B: the keys of the positions the game has been in (a repeat is the rules' draw), as layer 1 keeps them (`cs_rep::history::Seen`), borrowed by reference count: the search never copies the record.
    pub history: std::sync::Arc<cs_rep::history::Seen>,
    /// PLAN 79.22 PART B: the line being walked, kept by the tempos made and taken back (layer 1's `Path`)
    path: cs_rep::history::Path,
    /// MOVE ORDERING (plan 78.9, game-free): the quiet moves that cut a sibling off at the same ply (killers) and those that cut off anywhere (history, by the
    /// square of the depth) are read early; `ordering` off reads the old order (the comparison)
    pub ordering: bool,
    /// PRUNING (plan 78.10, game-free): the pass read shallow -- if passing already holds the bound, the node is cut (null move; never where the game has no
    /// pass or the mover has few units: a zugzwang guard); late quiet moves read two plies shallower first (re-read in full when they look better)
    pub prune: bool,
    /// PLAN 78.85 (diagnosis): the two parts of `prune` apart; both read `prune` too, so `prune` off is both off
    pub null_move: bool,
    pub late_reduce: bool,
    /// PLAN 78.85: the second reading of `ties` (`verify`); false is the equal reading exactly as it was (one reading, the search's own pruning, the warm table). THE DEFAULT OF THE ENGINE IS SET HERE (true), in `cs_decision::Params::default` and in the binding's signature.
    pub verify_ties: bool,
    /// PLAN 78.92: the move the deepening chooses is asked again without the two prunings (`verified_choice`); false (the default everywhere) is the search as it is.
    pub verify_choice: bool,
    /// PLAN 78.93: where the verification of the choice takes its time (`ChoiceMode`); only read when `verify_choice` is on. Slot (the default) is 78.92's form.
    pub choice_mode: ChoiceMode,
    /// PLAN 79.02: the equal reading reads the root moves in the deepening's own order and asks every move its null window held once more without the two prunings, in that order, while the slot lasts (`ties_reasked`); a move the unpruned
    /// answer puts under the best leaves the set, a move not reached before the clock stopped stays (held as it was). false (the default everywhere) is the equal reading as it was.
    pub reask: bool,
    /// PLAN 79.02: the move the search plays (the caller sets it after `run` and the verification of the choice): its value is the deepening's own, so `ties_reasked` does not ask it again.
    pub reask_skip: Option<usize>,
    /// PLAN 79.02: what the last `ties_reasked` did.
    pub reasked: ReaskCount,
    /// PLAN 79.03: no late-move reduction AT THE ROOT. A late quiet root move is then read at the full depth with the window like every other root move, so a root move whose win lies deeper than the reduced reading is not dropped as a fail-low
    /// (the reduction below the root and the null move are untouched). false (the default everywhere) is the search as it was.
    pub root_full: bool,
    /// PLAN 79.11 A: NAMED AND UN-NAMED ROOT MOVES ARE COMPARED BY THE SAME READING. A named move is read in full, an un-named late one reduced, and a reduced reading that fails low is taken as it is, so the named move could come out above a rival that is as good or better. With this on, when the root's best is a
    /// named move, every un-named root move that was only read reduced is read again in full (the window of the best) before the depth is called finished. Only where something is named at the root; false (the default everywhere) is the search as it was.
    pub fair_names: bool,
    /// PLAN 79.11 A: what the rereading of `fair_names` did (counted over the whole run): the rivals read again, those whose full reading came out at or above the best (the reduced reading had them under it), those above it.
    pub fair_reread: FairCount,
    /// THE FEW-REPLY EXTENSION (plan 78.11): a forced answer does not shorten the line
    pub extend: bool,
    killers: Vec<[Option<G::Move>; 2]>,
    hist: FxHashMap<(Role, G::Move), i64>,
    /// ROOT PRIOR (plan 78.14): indices of the root's legal moves a learned lemma proposes; read first (after the table's best), in this order. An order, never a
    /// verdict: an exact search returns the same value whatever the order.
    pub root_first: Vec<usize>,
    /// ROOT LAST (plan 78.36): indices of the root's legal moves a claim search refutes; read after every other root move, so a move of equal value that is not refuted comes first
    /// (the search keeps the first of equal values). A move of strictly better value is still kept: an order, never a verdict.
    pub root_last: Vec<usize>,
    /// ROOT ORDER (plan 78.51, optional; absent = the search exactly as it was): `root_pos[i]` is the place of the root move numbered `i` in a caller's order (`set_root_order`; a move the
    /// order does not name has `usize::MAX`). With `root_strict` the order REPLACES the root's reading order (the previous best is not read first, `root_first` and `root_last` are not
    /// read), so the first move of the best value in the caller's order is the one kept (`v > best`). Without it the order only breaks ties inside the groups the root's own order
    /// treats alike (same previous-best / `root_last` / `root_first` rank, same gain, same killer or history standing), the rest as before. An order, never a verdict.
    root_pos: Vec<usize>,
    root_strict: bool,
    /// PLAN 79.05 (SYSTEM_SPEC 8, the wake): MOVES NAMED BY POSITION. `named[key]` are the names (`Game::move_name`) of moves a caller wants read FIRST and UNREDUCED at the position whose `Game::key` is `key`, at the root and below it. An order and an exemption from the late-move
    /// reduction, never a verdict and never an addition to the value: an exact search returns the same value whatever the order. Empty (the default everywhere) is the search exactly as it was.
    pub named: FxHashMap<u64, Vec<String>>,
    /// PLAN 79.08 (the wake, repair 3): MOVES NAMED BY PLY. `named_ply[p]` are the names of moves read FIRST and UNREDUCED at every node of ply `p` (0: the root, 1: the reply of the other side), wherever the move is legal: the other side's moves are named for the ply after ours, by the move's name,
    /// and not by a position the search never visits. The same order-only meaning as `named`; empty (the default) is the search as it was.
    pub named_ply: FxHashMap<i32, Vec<String>>,
    /// The deepening steps of the runs so far, as measured (the granular cost of the search).
    pub reads: Vec<DepthRead>,
    /// PLAN 79.16 A (SYSTEM_SPEC 8 the rule for choosing a move and the budget; piece: the wake): THE SEARCH DOES NOT STOP ON A FAILED FIRST MOVE. Seconds the deepening may read on, after the clock stopped it, when the move it stood on at depth d failed low at depth d+1 (its d+1 reading finished, below the value of d):
    /// the d+1 reading is taken up again (the table kept) and stops as soon as one other root move holds the value of d (read in full), or when every move has been read (all failed: the best of d+1 stands), or when these seconds end (the answer is then depth d's, said so). 0.0 (the default everywhere, and the old player's path never sets it) is `run` exactly as it was.
    pub fail_extend: f64,
    /// PLAN 79.16 A: what the last `run` did on a failed first move (None: the first move did not fail low, or the extension was not asked).
    pub failed_first: Option<FailedFirst>,
    /// PLAN 79.16 C (SYSTEM_SPEC 11, everything in the hot path fast; piece: the wake): THE MOVER OF A NODE IS ASKED OF THE GAME ONCE. A game's `mover` may itself read the position to its end (the chess kernel decides whether the game is over before it names the side to move), and a node asked it up to four times (the node itself, the null move,
    /// the ordering of its moves, the cutoff) and a capture run again for the ordering. With `lean` the node asks once and passes the answer on; the capture run sorts only the moves that capture. The answers are the same (a test on logged positions: values, first moves and nodes equal); false (the default everywhere: the old player's path never sets it) is the search as it was.
    pub lean: bool,
    root_goal: Option<(i64, usize)>,
    root_found: Option<usize>,
    root_done: Vec<(usize, i64)>,
}

/// PLAN 79.16 A: the extension of the deepening on a failed first move. `depth` the depth read again (d+1), `first_value` the stood-on move's value at that depth, `goal` the value of the depth before, `outcome` one of "first_held" (the first move's null-window answer at that depth holds `goal`: the answer stays), "other_held" (a move other than the first holds `goal`: it is the answer), "all_failed" (every move was read, the best of the depth is the answer),
/// "cut" (the seconds ended, the answer is the depth before's), `found` the root move that held, `seconds` and `nodes` what the extension cost.
#[derive(Debug, Clone, PartialEq)]
pub struct FailedFirst { pub depth: i32, pub first_value: Option<i64>, pub goal: i64, pub outcome: &'static str, pub found: Option<usize>, pub seconds: f64, pub nodes: u64 }

/// PLAN 78.92: what `verified_choice` found: the move to play and why. Off: not asked; Held: the chosen move holds its value without the two prunings; Replaced: it did not and another candidate did (`index` is that one);
/// Failed: it did not and no other candidate held, the choice stands; Cut: the clock or the fuse stopped the reading, the choice stands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChoiceState { Off, Held, Replaced, Failed, Cut }

/// PLAN 78.93: where the verification of the choice (`verified_choice`) takes its time from. Slot (78.92, the default): after the deepening, out of the rest of the decision's clock, so the equal reading's slot is
/// what is left. Main: out of the main search's own clock: the deepening is stopped when `CHOICE_MAIN_SHARE` of that clock is left, the verification reads in that part and in whatever the deepening left unused, and the
/// equal reading keeps its whole slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChoiceMode { Slot, Main }

/// The part of the main search's clock kept back for the verification in `ChoiceMode::Main`. A named reading of 78.92's cost, not a tuned number: the verification of the chosen move cost a median 1.15 times the nodes of the search
/// that chose it, so about half of the clock is what it asks for.
pub const CHOICE_MAIN_SHARE: f64 = 0.5;

impl ChoiceMode {
    /// The clock the deepening itself gets out of `seconds` of the main search's clock (all of it unless the verification is on and takes its time from there).
    pub fn deepening_seconds(self, verify_choice: bool, seconds: f64) -> f64 { if verify_choice && self == ChoiceMode::Main { seconds * (1.0 - CHOICE_MAIN_SHARE) } else { seconds } }
    /// The clock the verification gets: Slot: `rest * share` as before (the equal reading's slot); Main: what is left of the main search's whole clock `seconds` after `used` seconds.
    pub fn verification_seconds(self, seconds: f64, used: f64, rest: f64, share: f64) -> f64 { if self == ChoiceMode::Main { (seconds - used).max(0.0).min(rest) } else { rest * share } }
    pub fn parse(s: &str) -> Option<ChoiceMode> { match s { "slot" => Some(ChoiceMode::Slot), "main" => Some(ChoiceMode::Main), _ => None } }
}

/// PLAN 79.02: what `ties_reasked` did at one root: `asked` the moves the first reading answered before the clock stopped it, `held` of them the moves whose pruned null-window answer held the best, `reasked` the held moves asked again without
/// the two prunings and answered, `out` of them the moves the unpruned answer put under the best, `unreached` the held moves not asked again before the clock stopped, `first_cut` whether the clock stopped the first reading.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ReaskCount { pub asked: usize, pub held: usize, pub reasked: usize, pub out: usize, pub unreached: usize, pub first_cut: bool }

/// PLAN 79.11 A: the count of the rereading of `Search::fair_names`: `reread` rivals read again in full, `understated` of them at or above the best named move's value though their reduced reading was under it, `above` of those strictly above it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FairCount { pub reread: usize, pub understated: usize, pub above: usize }

#[derive(Debug, Clone, Copy)]
pub struct ChoiceVerdict { pub index: usize, pub state: ChoiceState }

impl ChoiceState {
    pub fn label(self) -> &'static str { match self { ChoiceState::Off => "off", ChoiceState::Held => "held", ChoiceState::Replaced => "replaced", ChoiceState::Failed => "failed, kept", ChoiceState::Cut => "cut" } }
}

/// One deepening step of `run`, as measured: the depth, the nodes it spent, its seconds, and whether it finished (the last one may have been cut by the clock).
#[derive(Debug, Clone, Copy)]
pub struct DepthRead { pub depth: i32, pub nodes: u64, pub seconds: f64, pub complete: bool }

/// The answer: the move, its value, the line (move names) and its ledger, the depth read in full, the nodes.
#[derive(Debug, Clone)]
pub struct Line { pub first: Option<String>, pub value: i64, pub line: Vec<String>, pub lost: Vec<(Role, String)>, pub depth: i32, pub nodes: u64, pub first_index: Option<usize> }

impl<'a, G: Game, J: Judge> Search<'a, G, J> {
    pub fn new(g: &'a G, judge: &'a J, me: Role, seconds: f64) -> Self {
        Search { g, judge, me, nodes: 0, fuse: NODE_FUSE, deadline: Instant::now() + std::time::Duration::from_secs_f64(seconds.max(0.0)), tt: FxHashMap::default(), stopped: false, history: Default::default(), path: Default::default(), ordering: true, prune: true, null_move: true, late_reduce: true, verify_ties: false, verify_choice: false, choice_mode: ChoiceMode::Slot, reask: false, reask_skip: None, reasked: ReaskCount::default(), root_full: false, fair_names: false, fair_reread: FairCount::default(), extend: true, killers: vec![], hist: FxHashMap::default(), root_first: vec![], root_last: vec![], root_pos: vec![], root_strict: false, named: Default::default(), named_ply: Default::default(), reads: vec![], fail_extend: 0.0, failed_first: None, lean: false, root_goal: None, root_found: None, root_done: vec![] }
    }

    /// A search bounded by a node fuse alone (no clock): the census reads every position with the same budget.
    pub fn with_fuse(g: &'a G, judge: &'a J, me: Role, nodes: u64) -> Self {
        Search { g, judge, me, nodes: 0, fuse: nodes, deadline: Instant::now() + std::time::Duration::from_secs(1 << 30), tt: FxHashMap::default(), stopped: false, history: Default::default(), path: Default::default(), ordering: true, prune: true, null_move: true, late_reduce: true, verify_ties: false, verify_choice: false, choice_mode: ChoiceMode::Slot, reask: false, reask_skip: None, reasked: ReaskCount::default(), root_full: false, fair_names: false, fair_reread: FairCount::default(), extend: true,
                 killers: vec![], hist: FxHashMap::default(), root_first: vec![], root_last: vec![], root_pos: vec![], root_strict: false, named: Default::default(), named_ply: Default::default(), reads: vec![], fail_extend: 0.0, failed_first: None, lean: false, root_goal: None, root_found: None, root_done: vec![] }
    }

    /// PLAN 79.16 C: `with_fuse` for the steps before the search of the wake (the lines of the threat list, the questions): the mover of a node is asked of the game once (`lean`). The old player's reads (the census, the audits) keep `with_fuse`.
    pub fn lean_with_fuse(g: &'a G, judge: &'a J, me: Role, nodes: u64) -> Self { let mut x = Self::with_fuse(g, judge, me, nodes); x.lean = true; x }

    /// Plan 78.51: read the root moves in `order` (indices of the root's legal moves, first read first; each at most once). `strict`: see `root_pos`. An empty order is no order.
    pub fn set_root_order(&mut self, order: &[usize], strict: bool) {
        self.root_pos.clear();
        self.root_strict = strict && !order.is_empty();
        if order.is_empty() { return; }
        self.root_pos = vec![usize::MAX; order.iter().copied().max().map_or(0, |m| m + 1)];
        for (k, i) in order.iter().enumerate() { if self.root_pos[*i] == usize::MAX { self.root_pos[*i] = k; } }
    }

    fn root_place(&self, i: usize) -> usize { self.root_pos.get(i).copied().unwrap_or(usize::MAX) }

    /// The node fuse (a named cap; the interface's budget sets it).
    pub fn set_fuse(&mut self, nodes: u64) { self.fuse = nodes; }

    /// Did the fuse or the clock cut the last reading (its values are then lower bounds of the work, not exact)?
    pub fn cut(&self) -> bool { self.stopped }

    /// EVERY FIRST MOVE'S VALUE at `depth` (each read with the full window: exact within the horizon) -> [(the move's index in `moves`, its value)], best first.
    pub fn root_values(&mut self, s: &G::State, depth: i32) -> Vec<(usize, i64)> {
        let maxing = self.g.mover(s) == Some(self.me);
        let mut out = vec![];
        for (i, m, d) in self.ordered(s, maxing, None) {
            let q = self.g.apply(s, &m);
            let v = d + self.value(&q, depth - 1, 1, i64::MIN / 2, i64::MAX / 2);
            if self.stopped { break; }
            out.push((i, v));
        }
        out.sort_by_key(|(i, v)| (-*v, *i));
        out
    }

    /// A fresh node budget for the next reading, the table kept (the census reads many neighbouring positions with one search: their sub-positions are shared).
    pub fn rebudget(&mut self, nodes: u64) { self.fuse = self.nodes + nodes; self.stopped = false; }

    /// PLAN 79.09 B (SYSTEM_SPEC 7, the budget; piece: the wake): the clock of a reading that is one of the steps BEFORE the search of the wake: it stops at `at` whatever the node fuse (the lines it cuts are UNREAD, a gap, never a refutation). The clock is looked at every 1024 nodes.
    pub fn stop_at(&mut self, at: Instant) { self.deadline = at; }

    /// Go on reading after the clock stopped the search: a fresh clock of `seconds` (the table kept).
    pub fn resume(&mut self, seconds: f64) {
        self.stopped = false;
        self.deadline = Instant::now() + std::time::Duration::from_secs_f64(seconds.max(0.0));
    }

    /// THE FIRST MOVES AS GOOD AS `best` at `depth` (each asked "at least best?" by a null window; the table of the run warms it) -> their indices in `moves`.
    ///
    /// PLAN 78.85: TWO READINGS. The first is as it was (the search's own pruning, the table of the run warm): cheap, and it names the candidates. The search prunes by two heuristics that are not exact:
    /// the late-move reduction (a late quiet reply is read two plies shallower and the shallow answer is kept when it does not look better for the side to move) and the null move (the side to move passes, read
    /// shallow, and a pass that already holds the bound cuts the node). Asked whether a move holds at least the best, a reduced reading can miss the reply of the other side that wins a unit deeper, and a bound
    /// proved on such a reading is stored in the table as a full one. So each candidate is asked again by `verify`: a fresh table with both heuristics off (alpha-beta and the table alone; the search itself
    /// keeps them); the moves that hold are returned. A move the first reading did not name is not asked again (a first reading that is too cautious is a smaller set, not a false equal).
    /// Where the clock or the fuse stops either reading, only the moves verified so far are returned.
    pub fn ties(&mut self, s: &G::State, depth: i32, best: i64) -> Vec<usize> {
        if self.reask && !self.verify_ties { return self.ties_reasked(s, depth, best); }          // (plan 79.02, off by default)
        if !self.verify_ties {                                          // (the equal reading as it was before plan 78.85)
            return self.probes(s, depth, best, None).into_iter().filter(|(_, p)| *p >= best).map(|(i, _)| i).collect();
        }
        let first: Vec<usize> = self.probes(s, depth, best, None).into_iter().filter(|(_, p)| *p >= best).map(|(i, _)| i).collect();
        if self.stopped || first.is_empty() { return if self.stopped { vec![] } else { first }; }
        self.verify(s, depth, best, &first)
    }

    /// PLAN 79.02 THE EQUAL READING THAT ASKS A HELD MOVE AGAIN (SYSTEM_SPEC 4; the debt of plan 78.84). Takes the root, the depth D the deepening finished and its value `best`; the clock or the fuse must have been given back (`resume`). Returns the
    /// indices of the root moves held equal. (1) The root moves are asked "at least best at D?" by a null window on the warm table, as `ties` asks, but in the DEEPENING'S OWN ORDER of the root moves (`deepening_order`: the table's best first, then the
    /// order the last iteration read them in) and not in the fixed reading order, so that a clock that stops the reading drops the moves the deepening found least likely. (2) Each move that held is asked once more on a fresh table with the late-move
    /// reduction and the null move off (the table of the run and the two switches are put back after), in the same order, skipping `reask_skip` (the move the search plays: its value is the deepening's own). A move the unpruned answer puts under `best` leaves
    /// the set; a move held by the unpruned answer stays; a move not asked again before the clock stopped stays (held by the pruned answer as before; counted in `reasked.unreached`). A first reading the clock stopped returns the moves it had held, none asked again.
    /// A move not asked in the first reading is not in the set (it never was). The set is therefore never larger than `ties` returns on the same order. Called by: `ties` when `reask` is on (LedgerOp, the binding's equal-reading operator).
    pub fn ties_reasked(&mut self, s: &G::State, depth: i32, best: i64) -> Vec<usize> {
        self.reasked = ReaskCount::default();
        let order = self.deepening_order(s);
        let first = self.probes_in(s, depth, best, None, Some(&order));
        let mut held: Vec<usize> = first.iter().filter(|(_, p)| *p >= best).map(|(i, _)| *i).collect();
        self.reasked.asked = first.len();
        self.reasked.held = held.len();
        if self.stopped { self.reasked.first_cut = true; self.reasked.unreached = held.iter().filter(|i| Some(**i) != self.reask_skip).count(); return held; }
        let todo: Vec<usize> = held.iter().copied().filter(|i| Some(*i) != self.reask_skip).collect();
        let (warm, late, nul) = (std::mem::take(&mut self.tt), self.late_reduce, self.null_move);
        self.late_reduce = false;
        self.null_move = false;
        let mut out: Vec<usize> = vec![];
        for (k, c) in todo.iter().enumerate() {
            let ans = self.probes_in(s, depth, best, Some(&[*c]), None);
            if self.stopped { self.reasked.unreached = todo.len() - k; break; }
            self.reasked.reasked += 1;
            if !ans.iter().any(|(_, p)| *p >= best) { out.push(*c); }
        }
        self.late_reduce = late;
        self.null_move = nul;
        self.tt = warm;
        self.reasked.out = out.len();
        held.retain(|i| !out.contains(i));
        held
    }

    /// PLAN 79.02: the root moves in the order the deepening reads them (the table's best first, then `ordered_at` at the root with its killers and history).
    pub fn deepening_order(&self, s: &G::State) -> Vec<usize> {
        let maxing = self.g.mover(s) == Some(self.me);
        let hint = self.tt.get(&self.g.key(s)).map(|e| e.best);
        self.ordered_at(s, maxing, hint, 0).into_iter().map(|(i, _, _)| i).collect()
    }

    /// PLAN 78.92 THE VERIFICATION OF THE CHOICE (SYSTEM_SPEC 5; the debt of the equal reading's row, here for the move itself). Takes the root, the depth D the deepening finished, its value and the move it chose (an index
    /// in the root's legal moves); the clock or the fuse must have been given back (`resume`). The chosen move alone is asked "at least that value at depth D?" on a fresh table with the late-move reduction and the null move off
    /// (`verify`). Held: the choice stands. Failed: the search's own reading (the table of the run, warm) names the other moves whose null-window answer is at least the value, in its reading order, and each is asked the same way
    /// until one holds; that one replaces the choice. None holds: the choice stands, said so. A reading the clock or the fuse stops leaves the choice as it is, said so. Off (the switch, or a value that ends the game) asks nothing.
    /// Called by: LedgerOp (tactics/operators.rs) and the equal-reading operator of the binding (py/lib.rs), once, after `run`.
    pub fn verified_choice(&mut self, s: &G::State, depth: i32, value: i64, chosen: usize) -> ChoiceVerdict {
        if !self.verify_choice || value.abs() >= WIN - 1000 { return ChoiceVerdict { index: chosen, state: ChoiceState::Off }; }
        let held = !self.verify(s, depth, value, &[chosen]).is_empty();
        if held { return ChoiceVerdict { index: chosen, state: ChoiceState::Held }; }
        if self.stopped { return ChoiceVerdict { index: chosen, state: ChoiceState::Cut }; }
        let others: Vec<usize> = self.probes(s, depth, value, None).into_iter().filter(|(i, p)| *p >= value && *i != chosen).map(|(i, _)| i).collect();
        if self.stopped { return ChoiceVerdict { index: chosen, state: ChoiceState::Cut }; }
        for c in others {
            if !self.verify(s, depth, value, &[c]).is_empty() { return ChoiceVerdict { index: c, state: ChoiceState::Replaced }; }
            if self.stopped { return ChoiceVerdict { index: chosen, state: ChoiceState::Cut }; }
        }
        ChoiceVerdict { index: chosen, state: ChoiceState::Failed }
    }

    /// The moves of `of` (indices of the root's legal moves) that hold at least `best` at `depth` when asked on a fresh table with the late-move reduction and the null move off (the table of the run and the two
    /// switches are put back after).
    pub fn verify(&mut self, s: &G::State, depth: i32, best: i64, of: &[usize]) -> Vec<usize> {
        let (warm, late, nul) = (std::mem::take(&mut self.tt), self.late_reduce, self.null_move);
        self.late_reduce = false;
        self.null_move = false;
        let out = self.probes(s, depth, best, Some(of)).into_iter().filter(|(_, p)| *p >= best).map(|(i, _)| i).collect();
        self.late_reduce = late;
        self.null_move = nul;
        self.tt = warm;
        out
    }

    /// PLAN 78.85 (the instrument of the equal reading's audit): each first move's null-window answer at `depth` against `best`, the raw fail-soft number kept (`only`: just those moves)
    /// -> [(index, probe)]; the move is tied iff its probe is `best` or more. Stops where the clock or the fuse stops.
    pub fn probes(&mut self, s: &G::State, depth: i32, best: i64, only: Option<&[usize]>) -> Vec<(usize, i64)> { self.probes_in(s, depth, best, only, None) }

    /// PLAN 79.02: `probes` with the root moves read in `order` (indices of the root's legal moves; None: the fixed reading order `ordered`).
    pub fn probes_in(&mut self, s: &G::State, depth: i32, best: i64, only: Option<&[usize]>, order: Option<&[usize]>) -> Vec<(usize, i64)> {
        let maxing = self.g.mover(s) == Some(self.me);
        let mut out = vec![];
        let items: Vec<(usize, G::Move, i64)> = match order {
            None => self.ordered(s, maxing, None),
            Some(o) => { let ms = self.g.moves(s); o.iter().filter_map(|&i| ms.get(i).map(|m| (i, m.clone(), self.gain(s, m)))).collect() }
        };
        for (i, m, d) in items {
            if only.is_some_and(|o| !o.contains(&i)) { continue; }
            let q = self.g.apply(s, &m);
            let probe = d + self.value(&q, depth - 1, 1, best - 1 - d, best - d);
            if self.stopped { break; }
            out.push((i, probe));
        }
        out
    }

    /// PLAN 78.92 (the instrument of the choice audit): the value of `s` at `depth` with the full window and the root move the table keeps for it, or None where the clock or the fuse stopped the reading.
    pub fn root_best(&mut self, s: &G::State, depth: i32) -> Option<(i64, Option<usize>)> {
        let v = self.value(s, depth, 0, i64::MIN / 2, i64::MAX / 2);
        if self.stopped { return None; }
        Some((v, self.tt.get(&self.g.key(s)).map(|e| e.best)))
    }

    /// PLAN 78.92: the value of the root move numbered `index` at `depth` (its gain plus the future read with the full window, as `root_values` reads each move), or None where the reading was stopped.
    pub fn move_value(&mut self, s: &G::State, index: usize, depth: i32) -> Option<i64> {
        let m = self.g.moves(s).into_iter().nth(index)?;
        let d = self.gain(s, &m);
        let q = self.g.apply(s, &m);
        let v = d + self.value(&q, depth - 1, 1, i64::MIN / 2, i64::MAX / 2);
        if self.stopped { None } else { Some(v) }
    }

    /// PLAN 78.85 (the instrument of the trace): the value of the future of `s` at `depth` plies read as a node at ply `ply` of the main search with the window (`alpha`, `beta`).
    pub fn value_at(&mut self, s: &G::State, depth: i32, ply: i32, alpha: i64, beta: i64) -> i64 { self.value(s, depth, ply, alpha, beta) }

    /// THE FIRST MOVES THAT GAIN at `depth` -- each first asked only "above nothing?" (a null window: cheap), the gaining ones then read exact -> [(index, value)],
    /// best first. The census's reading: every move that gains, none that does not.
    pub fn gainers(&mut self, s: &G::State, depth: i32) -> Vec<(usize, i64)> {
        let maxing = self.g.mover(s) == Some(self.me);
        let mut out = vec![];
        for (i, m, d) in self.ordered(s, maxing, None) {
            let q = self.g.apply(s, &m);
            let probe = d + self.value(&q, depth - 1, 1, -d, -d + 1);         // (fails high iff the move's value is at least 1)
            if self.stopped { break; }
            if probe >= 1 {
                let v = d + self.value(&q, depth - 1, 1, i64::MIN / 2, i64::MAX / 2);
                if self.stopped { break; }
                if v >= 1 { out.push((i, v)); }
            }
        }
        out.sort_by_key(|(i, v)| (-*v, *i));
        out
    }

    /// The line that starts with the move numbered `first` on `s`, then the table's best moves, with its ledger events.
    pub fn line_from(&self, s: &G::State, first: usize, n: usize) -> (Vec<String>, Vec<(Role, String)>) {
        let ms = self.g.moves(s);
        let Some(m) = ms.get(first) else { return (vec![], vec![]) };
        let mover = self.g.mover(s).unwrap_or(self.me);
        let mut ev: Vec<(Role, String)> = self.g.captured(s, m).into_iter().map(|(o, k)| (o, format!("lose {k}"))).collect();
        ev.extend(self.g.changed(s, m).into_iter().map(|(a, b)| (mover, format!("promote {} to {b}", if a.is_empty() { "-" } else { a.as_str() }))));
        let q = self.g.apply(s, m);
        let (mut names, rest) = self.line_of(&q, n.saturating_sub(1));
        names.insert(0, self.g.move_name(s, m));
        ev.extend(rest);
        (names, ev)
    }

    fn terminal(&self, v: &[i64], ply: i32) -> i64 {
        let best_other = v.iter().enumerate().filter(|(i, _)| *i != self.me).map(|(_, x)| *x).max().unwrap_or(0);
        if v[self.me] > best_other { WIN - ply as i64 } else if v[self.me] < best_other { -WIN + ply as i64 } else { 0 }
    }

    /// What one move adds to the ledger for `me`, by the judge (what it takes, what changes kind).
    pub fn gain(&self, s: &G::State, m: &G::Move) -> i64 {
        let took: i64 = self.g.captured(s, m).iter().map(|(o, k)| self.judge.delta(self.me, *o, k)).sum();
        let mover = self.g.mover(s).unwrap_or(self.me);
        took + self.g.changed(s, m).iter().map(|(a, b)| self.judge.change(self.me, mover, a, b)).sum::<i64>()
    }

    fn tick(&mut self) -> bool {
        self.nodes += 1;
        if self.nodes >= self.fuse || (self.nodes & 1023 == 0 && Instant::now() >= self.deadline) { self.stopped = true; }
        self.stopped
    }

    /// The moves in their reading order: the previous best first, then the captures (by what they take for the side to move), then the rest; with each move's gain.
    fn ordered(&self, s: &G::State, maxing: bool, hint: Option<usize>) -> Vec<(usize, G::Move, i64)> {
        let mut v: Vec<(usize, G::Move, i64)> = self.g.moves(s).into_iter().enumerate().map(|(i, m)| { let d = self.gain(s, &m); (i, m, d) }).collect();
        v.sort_by_key(|(i, _, d)| (Some(*i) != hint, if maxing { -*d } else { *d }, *i));
        v
    }

    /// PLAN 79.16 C: `ordered` of the moves that gain only, in the order `ordered` reads them (the moves that do not gain are not sorted: the capture run skips them).
    fn ordered_captures(&self, s: &G::State, maxing: bool) -> Vec<(usize, G::Move, i64)> {
        let mut v: Vec<(usize, G::Move, i64)> = self.g.moves(s).into_iter().enumerate().filter_map(|(i, m)| { let d = self.gain(s, &m); if d == 0 { None } else { Some((i, m, d)) } }).collect();
        v.sort_by_key(|(i, _, d)| (false, if maxing { -*d } else { *d }, *i));
        v
    }

    /// `ordered` at a node of the main search (ply `ply`): the previous best first, then the moves that gain (best gain first), then the killers of this ply, then
    /// the quiet moves by their history.
    fn ordered_at(&self, s: &G::State, maxing: bool, hint: Option<usize>, ply: i32) -> Vec<(usize, G::Move, i64)> { self.ordered_at_by(s, maxing, hint, ply, None) }

    /// `ordered_at` where the caller already knows the mover of `s` (`known`: used with `lean`, else the game is asked as before).
    fn ordered_at_by(&self, s: &G::State, maxing: bool, hint: Option<usize>, ply: i32, known: Option<Option<Role>>) -> Vec<(usize, G::Move, i64)> {
        if !self.ordering {
            let mut v = self.ordered(s, maxing, hint);
            if ply == 0 && !self.root_last.is_empty() { v.sort_by_key(|(i, _, _)| self.root_last.contains(i)); }          // (stable: the rest keeps its order)
            return v;
        }
        let mover = known.unwrap_or_else(|| self.g.mover(s)).unwrap_or(self.me);
        let k = self.killers.get(ply as usize);
        let ro = ply == 0 && !self.root_pos.is_empty();         // (plan 78.51: a caller's root order is given)
        let mut v: Vec<(usize, G::Move, i64, i64)> = self.g.moves(s).into_iter().enumerate().map(|(i, m)| {
            let d = self.gain(s, &m);
            let rank = if ro && self.root_strict { self.root_place(i).min(1 << 60) as i64 } else if Some(i) == hint { i64::MIN } else if ply == 0 && self.root_last.contains(&i) { i64::MAX / 2 } else if ply == 0 && self.root_first.contains(&i) { i64::MIN / 2 + self.root_first.iter().position(|x| *x == i).unwrap() as i64 } else if d != 0 { -(1 << 40) + if maxing { -d } else { d } }
                       else if k.is_some_and(|k| k.iter().any(|x| x.as_ref() == Some(&m))) { -(1 << 30) }
                       else { -*self.hist.get(&(mover, m.clone())).unwrap_or(&0) };
            (i, m, d, rank)
        }).collect();
        if ro && !self.root_strict { v.sort_by_key(|(i, _, _, r)| (*r, self.root_place(*i), *i)); } else { v.sort_by_key(|(i, _, _, r)| (*r, *i)); }
        v.into_iter().map(|(i, m, d, _)| (i, m, d)).collect()
    }

    fn cutoff(&mut self, s: &G::State, m: &G::Move, d: i64, ply: i32, depth: i32, known: Option<Option<Role>>) {
        if !self.ordering || d != 0 { return; }
        let p = ply as usize;
        if self.killers.len() <= p { self.killers.resize(p + 1, [None, None]); }
        if self.killers[p][0].as_ref() != Some(m) { self.killers[p][1] = self.killers[p][0].take(); self.killers[p][0] = Some(m.clone()); }
        let mover = known.unwrap_or_else(|| self.g.mover(s)).unwrap_or(self.me);
        *self.hist.entry((mover, m.clone())).or_insert(0) += (depth as i64) * (depth as i64);
    }

    /// The capture run at a horizon: the side to move may stand (the ledger as it is) or go on capturing.
    fn quiet(&mut self, s: &G::State, ply: i32, mut alpha: i64, mut beta: i64, depth: u32) -> i64 {
        if let Some(v) = self.g.outcome(s) { return self.terminal(&v, ply); }
        if self.tick() || depth >= QUIET_FUSE { return 0; }
        let maxing = self.g.mover(s) == Some(self.me);
        let mut best = 0i64;                                   // standing: no further capture
        if maxing { if best >= beta { return best; } alpha = alpha.max(best); } else { if best <= alpha { return best; } beta = beta.min(best); }
        let moves: Vec<(usize, G::Move, i64)> = if self.lean { self.ordered_captures(s, maxing) } else { self.ordered(s, maxing, None) };
        for (_i, m, d) in moves {
            if d == 0 { continue; }                            // (captures only)
            let q = self.g.apply(s, &m);
            let v = d + self.quiet(&q, ply + 1, alpha - d, beta - d, depth + 1);
            if self.stopped { return best; }
            if maxing { if v > best { best = v; } alpha = alpha.max(v); } else { if v < best { best = v; } beta = beta.min(v); }
            if alpha >= beta { break; }
        }
        best
    }

    /// Minimax of the judge's sum over the FUTURE of `s` (what is lost from here on), `depth` plies, alpha-beta.
    fn value(&mut self, s: &G::State, depth: i32, ply: i32, mut alpha: i64, mut beta: i64) -> i64 {
        if let Some(v) = self.g.outcome(s) { return self.terminal(&v, ply); }
        if ply == 0 && self.fail_extend > 0.0 { self.root_done.clear(); self.root_found = None; }
        let key = self.g.key(s);
        if ply > 0 && (self.history.contains(&key) || self.path.contains(&key)) { return 0; }      // (a position met again: the rules' draw -- nothing more is lost or won)
        if depth <= 0 { return self.quiet(s, ply, alpha, beta, 0); }
        if self.tick() { return 0; }
        let (a0, b0) = (alpha, beta);
        let mut hint = None;
        if let Some(e) = self.tt.get(&key) {
            hint = Some(e.best);
            if e.depth >= depth {
                match e.bound { Bound::Exact => return e.value, Bound::Lower => alpha = alpha.max(e.value), Bound::Upper => beta = beta.min(e.value) }
                if alpha >= beta { return e.value; }
            }
        }
        let node_mover = self.g.mover(s);
        let known = if self.lean { Some(node_mover) } else { None };
        let maxing = node_mover == Some(self.me);
        if self.prune && self.null_move && ply > 0 && depth >= NULL_MIN_DEPTH && beta < WIN - 1000 && alpha > -WIN + 1000 {
            let mover = known.unwrap_or_else(|| self.g.mover(s));
            let mine = self.g.units(s).iter().filter(|(o, _)| Some(*o) == mover).count();
            if mine > ZUGZWANG_UNITS {
                if let Some(p) = self.g.pass(s) {
                    let v = self.value(&p, depth - 1 - NULL_R, ply + 1, alpha, beta);
                    if self.stopped { return 0; }
                    if (maxing && v >= beta) || (!maxing && v <= alpha) { return v; }
                }
            }
        }
        let mut best = if maxing { i64::MIN } else { i64::MAX };
        let mut best_i = 0;
        self.path.push(key);
        let mut order = self.ordered_at_by(s, maxing, hint, ply, known);
        // PLAN 79.05: the moves a caller named at this position are read first (stable among themselves) and are not read shallow first; nothing is done when none is named (the default)
        let mut pins: Vec<bool> = vec![];
        let by_ply = self.named_ply.get(&ply);
        if self.named.contains_key(&key) || by_ply.is_some() {
            let (a, b) = (self.named.get(&key), by_ply);
            pins = order.iter().map(|(_, m, _)| { let nm = self.g.move_name(s, m); a.is_some_and(|ns| ns.iter().any(|x| *x == nm)) || b.is_some_and(|ns| ns.iter().any(|x| *x == nm)) }).collect();
            if pins.iter().any(|p| *p) {
                let mut z: Vec<_> = order.into_iter().zip(pins).collect();
                z.sort_by_key(|(_, p)| !*p);
                pins = z.iter().map(|(_, p)| *p).collect();
                order = z.into_iter().map(|(o, _)| o).collect();
            } else { pins = vec![]; }
        }
        // THE FEW-REPLY EXTENSION (plan 78.11, game-free): a node with at most FEW_REPLIES legal moves (a forced answer) does not shorten the line
        let ext = if self.extend && order.len() <= FEW_REPLIES && ply < 2 * depth + 8 { 1 } else { 0 };
        let depth = depth + ext;
        let mut reduced_only: Vec<(usize, G::Move, i64)> = vec![];
        let mut best_pinned = false;
        let mut cut_off = false;
        let mut early = false;
        for (n_read, (i, m, d)) in order.into_iter().enumerate() {
            let q = self.g.apply(s, &m);
            let late = self.prune && self.late_reduce && depth >= LMR_MIN_DEPTH && n_read >= LMR_AFTER && d == 0 && Some(i) != hint && !(ply == 0 && self.root_strict) && !(ply == 0 && self.root_full) && !pins.get(n_read).copied().unwrap_or(false);        // (plan 78.51: under a strict root order no root move is read shallow first, so the first of the best value IS the first one that holds it at this depth)
            let mut v = if late { d + self.value(&q, depth - 3, ply + 1, alpha - d, beta - d) } else { 0 };
            let mut read_in_full = !late;
            if !late || (maxing && v > alpha) || (!maxing && v < beta) {
                v = d + self.value(&q, depth - 1, ply + 1, alpha - d, beta - d);
                read_in_full = true;
            }
            if self.stopped { self.path.pop(); return if best == i64::MIN || best == i64::MAX { 0 } else { best }; }
            if !read_in_full && ply == 0 && self.fair_names { reduced_only.push((i, m.clone(), d)); }
            if (maxing && v > best) || (!maxing && v < best) { best = v; best_i = i; best_pinned = pins.get(n_read).copied().unwrap_or(false); }
            if maxing { alpha = alpha.max(v); } else { beta = beta.min(v); }
            if ply == 0 && self.fail_extend > 0.0 {
                self.root_done.push((i, v));
                // PLAN 79.16 A: in the reading taken up after a failed first move, one other move read in full that holds the value of the depth before ends it
                if let Some((goal, skip)) = self.root_goal { if i != skip && read_in_full && ((maxing && v >= goal) || (!maxing && v <= goal)) { self.root_found = Some(i); best = v; best_i = i; early = true; break; } }
            }
            if alpha >= beta { self.cutoff(s, &m, d, ply, depth, known); cut_off = true; break; }
        }
        // PLAN 79.11 A: the best is a named move read in full: the rivals read only reduced are read again in full, so that the comparison is by one reading
        if ply == 0 && self.fair_names && best_pinned && !cut_off && !early {
            for (i, m, d) in reduced_only {
                let q = self.g.apply(s, &m);
                let v = d + self.value(&q, depth - 1, ply + 1, alpha - d, beta - d);
                if self.stopped { self.path.pop(); return best; }
                self.fair_reread.reread += 1;
                if (maxing && v >= best) || (!maxing && v <= best) { self.fair_reread.understated += 1; }
                if (maxing && v > best) || (!maxing && v < best) { best = v; best_i = i; self.fair_reread.above += 1; }
                if maxing { alpha = alpha.max(v); } else { beta = beta.min(v); }
            }
        }
        self.path.pop();
        let bound = if early { if maxing { Bound::Lower } else { Bound::Upper } } else if best <= a0 { Bound::Upper } else if best >= b0 { Bound::Lower } else { Bound::Exact };
        self.tt.insert(key, Tt { depth, value: best, bound, best: best_i });
        best
    }

    /// The principal line from `s` (by the table's best moves) with its ledger.
    fn line_of(&self, s: &G::State, n: usize) -> (Vec<String>, Vec<(Role, String)>) {
        // (the ledger's events: "lose <kind>" for a unit taken, "promote <from> to <to>" for a change of kind)
        let (mut names, mut lost) = (vec![], vec![]);
        let mut cur = s.clone();
        for _ in 0..n {
            if self.g.outcome(&cur).is_some() { break; }
            let Some(e) = self.tt.get(&self.g.key(&cur)) else { break };
            let ms = self.g.moves(&cur);
            let Some(m) = ms.get(e.best) else { break };
            names.push(self.g.move_name(&cur, m));
            lost.extend(self.g.captured(&cur, m).into_iter().map(|(o, k)| (o, format!("lose {k}"))));
            let mover = self.g.mover(&cur).unwrap_or(self.me);
            lost.extend(self.g.changed(&cur, m).into_iter().map(|(a, b)| (mover, format!("promote {} to {b}", if a.is_empty() { "-" } else { a.as_str() }))));
            cur = self.g.apply(&cur, m);
        }
        (names, lost)
    }

    /// PLAN 79.16 A (see `fail_extend`). Called by `run` where the clock stopped depth `d`; `out` is the answer of depth d-1. THE FIRST MOVE IS HEARD: where the move `out` stands on was read at depth `d` to its end and came out below `out.value`, or its reading at depth `d` was not finished (the root reads it with the full window,
    /// which is dear), it is asked "at least the value of depth d-1?" by a null window at depth `d` (the table kept) within `fail_extend` seconds. Held: the answer stays depth d-1's (it was heard). Failed low: the reading of `d` is taken up again within what is left of the seconds and stops as soon as one other root move
    /// holds the value (read in full): that move answers at depth `d`; or at the end of the root (all failed): the best move of depth `d` answers. The seconds ending leave the answer of depth d-1 (`cut`).
    fn read_on_failed_first(&mut self, s: &G::State, d: i32, out: &mut Line) {
        let (Some(fi), true) = (out.first_index, out.value.abs() < WIN - 1000) else { return };
        let maxing = self.g.mover(s) == Some(self.me);
        let goal = out.value;
        let known = self.root_done.iter().find(|x| x.0 == fi).map(|x| x.1);
        let fails = |v: i64| if maxing { v < goal } else { v > goal };
        if known.is_some_and(|v| !fails(v)) { return; }                  // the first move finished depth d and holds: nothing to hear
        let (n0, t0) = (self.nodes, Instant::now());
        let ext = self.fail_extend;
        let mut first_value = known;
        if known.is_none() {
            self.resume(ext);
            let pr = self.probes(s, d, goal, Some(&[fi]));
            let v = pr.first().map(|x| x.1);
            if self.stopped || v.is_none() { self.finish_extension(d, goal, v, "cut", None, n0, t0); self.stopped = false; return; }
            first_value = v;
            if !fails(v.unwrap_or(goal)) { self.finish_extension(d, goal, v, "first_held", None, n0, t0); self.stopped = false; return; }
        }
        let left = (ext - t0.elapsed().as_secs_f64()).max(0.0);
        self.root_goal = Some((goal, fi));
        self.resume(left);
        let v = self.value(s, d, 0, i64::MIN / 2, i64::MAX / 2);
        self.root_goal = None;
        let found = self.root_found;
        let outcome = if self.stopped { "cut" } else if found.is_some() { "other_held" } else { "all_failed" };
        self.finish_extension(d, goal, first_value, outcome, found, n0, t0);
        if self.stopped { self.stopped = false; return; }          // (the answer stays depth d-1's; the clock is given back)
        let (line, lost) = self.line_of(s, d as usize + 8);
        let fi2 = self.tt.get(&self.g.key(s)).map(|e| e.best);
        *out = Line { first: line.first().cloned(), value: v, line, lost, depth: d, nodes: self.nodes, first_index: fi2 };
    }

    fn finish_extension(&mut self, d: i32, goal: i64, first_value: Option<i64>, outcome: &'static str, found: Option<usize>, n0: u64, t0: Instant) {
        let seconds = t0.elapsed().as_secs_f64();
        self.reads.push(DepthRead { depth: d, nodes: self.nodes - n0, seconds, complete: outcome != "cut" });
        self.failed_first = Some(FailedFirst { depth: d, first_value, goal, outcome, found, seconds, nodes: self.nodes - n0 });
    }

    /// Iterative deepening to `max_depth` while the clock lasts -> the deepest line read in full.
    pub fn run(&mut self, s: &G::State, max_depth: i32) -> Line {
        let mut out = Line { first: None, value: 0, line: vec![], lost: vec![], depth: 0, nodes: 0, first_index: None };
        for d in 1..=max_depth {
            let (n0, t0) = (self.nodes, Instant::now());
            let v = self.value(s, d, 0, i64::MIN / 2, i64::MAX / 2);
            self.reads.push(DepthRead { depth: d, nodes: self.nodes - n0, seconds: t0.elapsed().as_secs_f64(), complete: !self.stopped });
            if self.stopped { if self.fail_extend > 0.0 { self.read_on_failed_first(s, d, &mut out); } break; }
            let (line, lost) = self.line_of(s, d as usize + 8);
            let fi = self.tt.get(&self.g.key(s)).map(|e| e.best);
            out = Line { first: line.first().cloned(), value: v, line, lost, depth: d, nodes: self.nodes, first_index: fi };
            if v.abs() >= WIN - 1000 { break; }                // the game's own outcome is read: no deeper line changes it
        }
        out.nodes = self.nodes;
        out
    }
}
