//! PLAN 79.32 PART A: THE PURE SEARCH IS LAYER 0'S (SYSTEM_SPEC 11 "The pure search is mathematics and lives in layer 0; what steers it does not", 13 item 50; part served: the base, with the tactical reasoner as its first case).
//!
//! WHAT THIS MODULE IS. The reading of a tree of choices that is never built: THERE IS A CHOICE SUCH THAT FOR EVERY COUNTER-CHOICE, by recursion, to a depth, with what was already answered kept by the sameness of nodes, for MORE THAN ONE PROPERTY read in one walk and in two readings together.
//! It is an algorithm over a NODE THAT GIVES ITS OPTIONS WHEN ASKED (`Space`): the walk steps into an option and back out of it on the spot, so a live position is walked without a copy, and nothing of the tree exists but the line being walked.
//! It names no game and no thing of a world. WHAT STEERS A WALK is not here but is handed in (`Steer`): the order of options, whether the walk may go on (a budget of nodes, a clock), and whether answers are kept. WHAT THE PROPERTIES ARE, which options a node has and who picks at it, are the node's.
//!
//! THE READING (what the walk computes; the parts of a node are the trait's). Properties are numbered 0 to 63 and asked as a set (a `u64` mask). Two readings are asked in ONE walk, each over its own set of properties (`Ask { strict, loose }`):
//!   STRICT: a property is MET at a node where `Look::held` says it holds. A node where the choosing is OWN (`Space::own_turn`) meets a property within n steps when SOME option does within n - 1; any other node when EVERY option does, and it has an option; an ended node (`Look::ended`) has no option and
//!     meets only what it holds itself. A property a node holds is met in nought steps (the node itself counts). The answer for a property is the steps of a line found (`Out::strict[i]`), or `N::NONE` for none within the depth. The steps given are an upper bound where the answer came from the kept answers.
//!   LOOSE (the second reading, a superset of the first: whatever is met strictly is met loosely): a property is met loosely where it is met strictly, or at a node that an OWN option made (`by_own`), after at least one step, where the property is bound (`Look::bound`) and SOME option of the node leaves it unbound
//!     (`Space::breaks`). So a property that the asker can force to bind and that the other side can always undo is met loosely and not strictly.
//!   A node met again on the line being walked (`Space::CYCLES`) or in the record the node keeps (`Space::met_before`) ends a line (in the record, an answer; on the line, an answer that rests on the line and is never kept). A node at depth nought is read for what it holds and goes no deeper (`Walk::horizon`
//!   says some line ended there with a strict property open). The walk stops, with `Err(Cut)`, where `Steer::spend` says: a cut is a gap, never an answer.
//!
//! INTERFACE (written before the body; who calls what).
//!   `Whole` (the numbers of the hot path: depth and steps are bare whole numbers; `u8` for a deep search, `u16` for a tree of unknown depth; the exact `Num` is not a `Whole` and is used only where a reading asks for exact values: `TreeSpace::value`).
//!   `Cut` (the walk was stopped by what steers it). `Ask`, `Out<N>` (the masks asked and the steps answered). `Look` (what a node says of itself).
//!   `Space` (the node; implemented by a world, or by an explicit tree: `TreeSpace`): `key`, `look`, `own_turn`, `met_before`, `options`, `breaks`, `enter`, `leave`, and `CYCLES`.
//!     STEPPING: `enter(&option)` makes the node the option leads to the current one, `leave()` takes it back; the walk calls them in pairs and never holds a node of its own. `options` fills a vector the walk hands in (it is the one allocation of a node that is expanded).
//!     LAZY LIVE: a node may be made live on first need (`look` or `options`) and taken back by `leave`: what a node costs to make is the node's, counted by it, and the walk does not know it.
//!   `Steer<S>` (what steers; implemented by the layer that asks): `spend` (called once for every node entered, before anything is read: the budget, the clock and any count), `order` (the options of an expanded node, the first open property given, put in the order to try; an order is never a verdict), `keeps`
//!     (whether answers are kept by key across the walk).
//!   `Walk<S, T, N>`: `new(space, steer)`, `go(depth, ask, later, by_own, &mut out) -> Result<(), Cut>` (the reading at the node the space is at), `line_push(key)`, `line_pop()` (a caller that walks a line by hand keeps the line's keys), `horizon`, `taint` (read and
//!     reset by the caller between deepenings), and the space and the steer as public fields (the caller keeps what it counts in them).
//!   THE CASES: layer 4's position and steering (the module `reach` of the tactical layer); the explicit tree (`TreeSpace`, for the forcing reading of `choice::can_force_set`, and `TreeSpace::value` for the readings that weigh numbers: `guarantee`, `value`, `force_chance`).
//! LAWS: lean/ideas/Ideas/Layer0/Choice.lean (what the forcing reading is: coalition_exactly_one; what the kept answers rely on: stated there).
pub use super::fold::Node;
use std::collections::HashMap;
use std::fmt::Debug;

/// The most properties one walk reads together (the width of a mask).
pub const MOST: usize = 64;

/// A whole number of the hot path: a depth, a count of steps. Bare machine numbers satisfy it, so the walk uses them; `NONE` is the largest and means "no answer".
pub trait Whole: Copy + Ord + Debug {
    const NONE: Self;
    const ZERO: Self;
    fn up(self) -> Self;
    fn down(self) -> Self;
}

macro_rules! whole {
    ($($t:ty),*) => { $( impl Whole for $t {
        const NONE: $t = <$t>::MAX;
        const ZERO: $t = 0;
        #[inline(always)] fn up(self) -> $t { self + 1 }
        #[inline(always)] fn down(self) -> $t { self.saturating_sub(1) }   // from nought it stays at nought: it never wraps to NONE (mended 2026-11-23, found by the tests of plan 79.36)
    } )* };
}
whole!(u8, u16, u32, u64, usize);

/// The walk was stopped by what steers it (a budget, a clock): the reading is a gap, never an answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Cut;

/// What is asked of a node: the properties of the strict reading and of the loose reading (masks over 0..MOST).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Ask { pub strict: u64, pub loose: u64 }

/// What a walk answers, by property: the steps of a line that meets it, or `N::NONE`.
#[derive(Clone, Copy)]
pub struct Out<N: Whole> { pub strict: [N; MOST], pub loose: [N; MOST] }

impl<N: Whole> Out<N> {
    /// No answer for any property.
    #[inline(always)]
    pub fn none() -> Out<N> { Out { strict: [N::NONE; MOST], loose: [N::NONE; MOST] } }
}

/// What a node says of itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Look {
    /// the properties that hold of the node as such
    pub bound: u64,
    /// the properties that count as held at this node (a part of `bound`)
    pub held: u64,
    /// no option follows: the node ends a line
    pub ended: bool,
}

/// A node that gives its options when asked (the stepping is `Node`'s, which the walk of values shares).
pub trait Space: Node {
    /// Whether a node can be met again on its own line (then the line's keys are kept and a repeat ends a line).
    const CYCLES: bool;
    /// The node's key: two nodes with one key are the same node for the answers kept.
    fn key(&self) -> u64;
    /// What the node says of itself (it may make the node live on first need).
    fn look(&mut self) -> Look;
    /// Whether the node was met before the walk began (in a record the node keeps), which ends a line.
    fn met_before(&self, key: u64) -> bool;
    /// Of the properties in `cand` (all bound here), those that some option of the node leaves unbound.
    fn breaks(&mut self, cand: u64, opts: &[Self::Opt]) -> u64;
}

/// What steers a walk. The node's own order is never a verdict: a walk answers the same under any order wherever it is not cut.
pub trait Steer<S: Space> {
    /// Called once for every node entered, before anything else is read; `asked` says which readings the node is asked for. `Err(Cut)` stops the walk.
    fn spend(&mut self, asked: Ask) -> Result<(), Cut>;
    /// Put the options of the node `space` is at in the order to try; `first` is the first open property (the one that steers).
    fn order(&mut self, space: &S, opts: &mut Vec<S::Opt>, first: usize);
    /// Whether answers are kept by key across the walk.
    fn keeps(&self) -> bool;
}

/// The kept answer of one property at one node: the fewest steps found, and the deepest depth it was read in full and not met.
#[derive(Clone, Copy)]
struct Entry<N: Whole> { reach: N, fail: Option<N> }

/// The index of a property's kept answer: the strict reading by property; the loose reading by property and by whether the node was made by an own option (the loose leaf depends on it).
#[inline(always)]
fn slot(i: usize, loose: bool, by_own: bool) -> u16 { i as u16 + if !loose { 0 } else if by_own { 128 } else { 64 } }

/// The set bits of a mask, lowest first.
#[inline(always)]
fn bits(mut m: u64) -> impl Iterator<Item = usize> {
    std::iter::from_fn(move || if m == 0 { None } else { let i = m.trailing_zeros() as usize; m &= m - 1; Some(i) })
}

/// THE WALK: the one algorithm. It owns the line's keys, the kept answers and the two marks of a deepening (`horizon`, `taint`).
pub struct Walk<S: Space, T: Steer<S>, N: Whole> {
    pub space: S,
    pub steer: T,
    /// some line met the depth limit with a strict property still open (reset by the caller)
    pub horizon: bool,
    /// an answer met a node of its own line: bit 0 for the strict reading, bit 1 for the loose (reset by the caller; an answer that rests on the line is never kept)
    pub taint: u8,
    line: Vec<u64>,
    kept: HashMap<(u64, u16), Entry<N>>,
}

impl<S: Space, T: Steer<S>, N: Whole> Walk<S, T, N> {
    pub fn new(space: S, steer: T) -> Self { Walk { space, steer, horizon: false, taint: 0, line: Vec::new(), kept: HashMap::new() } }

    /// Keep a key on the line (a caller that walks a line by hand), and take it off.
    pub fn line_push(&mut self, key: u64) { self.line.push(key); }
    pub fn line_pop(&mut self) { self.line.pop(); }

    /// The reading at the node the space is at. `depth` steps left; `later`: the node is below the one the caller asked about (a node at the start is not tested for a repeat); `by_own`: the node was made by an option of the asker.
    /// For each property asked, `out` gets the steps of a line that meets it, or is left as it was (the caller hands it in as `Out::none()`).
    pub fn go(&mut self, depth: N, ask: Ask, later: bool, by_own: bool, out: &mut Out<N>) -> Result<(), Cut> {
        self.steer.spend(ask)?;
        let key = self.space.key();
        let look = self.space.look();
        let hit = look.held;
        for i in bits(hit & ask.strict) { out.strict[i] = N::ZERO; }
        for i in bits(hit & ask.loose) { out.loose[i] = N::ZERO; }
        let mut left_s = ask.strict & !hit;
        let mut left_l = ask.loose & !hit;
        if left_s | left_l == 0 || look.ended { return Ok(()); }
        if later {
            if self.space.met_before(key) { return Ok(()); }
            if S::CYCLES && self.line.contains(&key) { self.taint |= (left_s != 0) as u8 | (((left_l != 0) as u8) << 1); return Ok(()); }
        }
        if depth == N::ZERO { if left_s != 0 { self.horizon = true; } return Ok(()); }
        let keeps = self.steer.keeps();
        if keeps {
            for i in bits(left_s | left_l) {
                let b = 1u64 << i;
                if left_s | left_l == 0 { break; }
                if left_s & b == 0 && left_l & b == 0 { continue; }
                // the strict answer's kept entry: met within the depth settles both answers; not met within the depth settles the strict one
                if let Some(e) = self.kept.get(&(key, slot(i, false, false))).copied() {
                    if e.reach != N::NONE && e.reach <= depth {
                        if left_s & b != 0 { out.strict[i] = e.reach; }
                        if left_l & b != 0 { out.loose[i] = e.reach; }
                        left_s &= !b; left_l &= !b;
                        continue;
                    }
                    if e.fail.is_some_and(|f| f >= depth) { left_s &= !b; }
                }
                // the loose answer's kept entry: met within the depth settles it; not met within the depth settles both
                if left_l & b != 0 {
                    if let Some(e) = self.kept.get(&(key, slot(i, true, by_own))).copied() {
                        if e.reach != N::NONE && e.reach <= depth { out.loose[i] = e.reach; left_l &= !b; }
                        else if e.fail.is_some_and(|f| f >= depth) { left_l &= !b; left_s &= !b; }
                    }
                }
            }
            if left_s | left_l == 0 { return Ok(()); }
        }
        let taint_before = self.taint;
        self.taint = 0;
        let mut opts: Vec<S::Opt> = Vec::new();
        self.space.options(&mut opts);
        self.steer.order(&self.space, &mut opts, (left_s | left_l).trailing_zeros() as usize);
        // the loose leaf: the property is bound here after an own option and some option of this node leaves it unbound
        if left_l != 0 && by_own && later {
            let cand = left_l & look.bound;
            if cand != 0 {
                let unbound = self.space.breaks(cand, &opts);
                for i in bits(unbound) { out.loose[i] = N::ZERO; }
                left_l &= !unbound;
                if left_s | left_l == 0 { self.taint |= taint_before; return Ok(()); }
            }
        }
        let (asked_s, asked_l) = (left_s, left_l);
        let maxing = self.space.own_turn();
        if S::CYCLES { self.line.push(key); }
        let mut result: Result<(), Cut> = Ok(());
        if maxing {
            let (mut todo_s, mut todo_l) = (left_s, left_l);
            for o in &opts {
                let mut sub = Out::<N>::none();
                self.space.enter(o);
                let r = self.go(depth.down(), Ask { strict: todo_s, loose: todo_l }, true, true, &mut sub);
                self.space.leave();
                if let Err(c) = r { result = Err(c); break; }
                for i in bits(todo_s | todo_l) {
                    let b = 1u64 << i;
                    if todo_s & b != 0 && sub.strict[i] != N::NONE { out.strict[i] = sub.strict[i].up(); todo_s &= !b; if todo_l & b != 0 && sub.loose[i] == N::NONE { sub.loose[i] = sub.strict[i]; } }
                    if todo_l & b != 0 && sub.loose[i] != N::NONE { out.loose[i] = sub.loose[i].up(); todo_l &= !b; }
                }
                if todo_s | todo_l == 0 { break; }
            }
            left_s = todo_s; left_l = todo_l;
        } else {
            let (mut alive_s, mut alive_l) = (left_s, left_l);
            let (mut worst_s, mut worst_l) = ([N::ZERO; MOST], [N::ZERO; MOST]);
            if opts.is_empty() { alive_s = 0; alive_l = 0; }
            for o in &opts {
                let mut sub = Out::<N>::none();
                self.space.enter(o);
                let r = self.go(depth.down(), Ask { strict: alive_s, loose: alive_l }, true, false, &mut sub);
                self.space.leave();
                if let Err(c) = r { result = Err(c); break; }
                for i in bits(alive_s | alive_l) {
                    let b = 1u64 << i;
                    if alive_s & b != 0 { if sub.strict[i] == N::NONE { alive_s &= !b; } else { worst_s[i] = worst_s[i].max(sub.strict[i].up()); } }
                    if alive_l & b != 0 { if sub.loose[i] == N::NONE { alive_l &= !b; alive_s &= !b; } else { worst_l[i] = worst_l[i].max(sub.loose[i].up()); } }
                }
                if alive_s | alive_l == 0 { break; }
            }
            if result.is_ok() {
                for i in bits(alive_s | alive_l) {
                    let b = 1u64 << i;
                    if alive_s & b != 0 { out.strict[i] = worst_s[i]; }
                    if alive_l & b != 0 { out.loose[i] = worst_l[i]; }
                }
                left_s &= !alive_s; left_l &= !alive_l;
            }
        }
        if S::CYCLES { self.line.pop(); }
        if result.is_ok() && keeps {
            for i in bits(asked_s | asked_l) {
                let b = 1u64 << i;
                if asked_s & b != 0 { self.keep(key, slot(i, false, false), out.strict[i], left_s & b != 0 && self.taint & 1 == 0, depth); }
                if asked_l & b != 0 { self.keep(key, slot(i, true, by_own), out.loose[i], left_l & b != 0 && self.taint & 2 == 0, depth); }
            }
        }
        self.taint |= taint_before;
        result
    }

    /// Keep an answer: the fewest steps where it was met, or the depth it was read in full without being met (not where it rests on the line).
    #[inline(always)]
    fn keep(&mut self, key: u64, ix: u16, got: N, failed: bool, depth: N) {
        if got != N::NONE {
            self.kept.entry((key, ix)).and_modify(|e| e.reach = e.reach.min(got)).or_insert(Entry { reach: got, fail: None });
        } else if failed {
            self.kept.entry((key, ix)).and_modify(|e| e.fail = e.fail.max(Some(depth))).or_insert(Entry { reach: N::NONE, fail: Some(depth) });
        }
    }
}
