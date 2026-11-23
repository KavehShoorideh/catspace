//! PLAN 79.33 PART A: ONE GENERAL ANSWER FOR LAYER 0'S WALK (SYSTEM_SPEC 11 "What plan 79.32 left of one algorithm", 13 item 51; part served: the base, with the tactical reasoner as cases).
//!
//! WHAT THIS MODULE IS. The reading of a tree of choices that is never built, where what a node answers is a VALUE of a kind the caller names: at a node where the asker picks the BEST of the values of its options, at any other node the WORST, at a node of chance the EXPECTATION, all under a WINDOW that cuts what cannot matter, with what was already answered kept by the sameness of nodes as an exact value or as a bound.
//! Within-how-many-steps (`walk::Walk`) is one kind of answer; an exact number is another (`TreeSpace::value`); a class and a ply as one whole number is another; a number of gain and loss along a line is another.
//! It names no game and no thing of a world. It is the recursion; what the values are, what a node is worth where the walk stops, what an option gains, which answers are kept and what the budget is, are handed in (`Kind`). Static dispatch only: no boxed function, no trait object.
//!
//! INTERFACE (written before the body; who calls what).
//!   `Node` (the stepping, shared with `walk::Space`, which extends it): `own_turn` (the asker picks here: the best; else the worst), `chance` (the node is a choice by chance: the expectation; provided, false), `options` (fill a vector the walk hands in: the one allocation of an expanded node), `enter(&option)` (the node the option leads to becomes the current one), `leave()` (take it back; the walk calls them in pairs).
//!   `Bounds<V>` (the window: `lo` is what the asker already has, `hi` what the rest already hold it to; a node whose answer cannot lie inside is cut). `Pre<V, T>` (the answer of a node without opening it, or the token of a node to be opened).
//!   `Kind<S: Node>` (what the caller hands in; implemented by the layer that asks):
//!     `Val` (the value: a bare machine number for the hot path, or the exact `Num`), `Err` (what stops a walk: a budget, an overflow), `Edge` (what an option carries: its gain, or the weight of its chance), `Token` (what a node opened keeps until it is closed: a key, a mark), `WINDOWED` (whether windows cut), `CHANCE` (whether expectations are paid for).
//!     `before(space, left, ply, &mut window) -> Result<Pre, Err>`: called once for every node entered, first: the budget and the clock, the leaf (what a node is worth where the walk stops), the answers kept (they may narrow the window or answer), repeats; `left` the steps left, `ply` the steps taken since the root asked. `Pre::Go(token)` opens the node.
//!     `horizon(space, token, ply) -> Result<Val, Err>`: the value of a node opened with no step left.
//!     `base(space, &token) -> Option<Val>`: the value of standing (the asker may stop here), kept as the best so far (provided: none).
//!     `order(space, &mut options)`: the order to try (provided: as given; an order is never a verdict).
//!     `edge(space, &option) -> Option<Edge>`: what the option carries, read before stepping into it; `None` leaves the option out (provided by none: required).
//!     `child(edge, &window) -> Bounds` (the window the option's node is read in; provided: the same), `lift(edge, value) -> Result<Val, Err>` (the value of the option from its node's value; provided: the same), `mix(acc, edge, value)` (the expectation so far, a weighted sum: only where `CHANCE`).
//!     `improved(space, ply, &option)`: the option just became strictly the best so far (a caller that keeps the line of best play keeps it here; provided: nothing). `open(&token)` / `close()`: the node's line is entered and left around its options (a caller that keeps the keys of its line; provided: nothing).
//!     `finish(space, token, got, &window_at_entry) -> Result<Val, Err>`: the value of an opened node from the best of its options (`None`: no option), where the answer is kept, with the window the node was read in.
//!   `Fold<S, K>`: `new(space, kind)`, `go(left, ply, window) -> Result<Val, Err>` (the value at the node the space is at), and the space and the kind as public fields (the caller keeps what it counts in them).
//! THE ORDER OF A NODE, as the cases rely on it: `before`; (`horizon` when `left` is nought); `base`; `options`, `order`; for each option `edge`, `child`, step in, the walk of the node, step out, `lift`, the join (`improved` when strictly better), the window moves, a cut ends the loop; `finish`. A chance node: `edge`, step in, the walk, step out, `mix`; no window.
//! CASES: the explicit tree (`TreeSpace::value`, in this plan); layer 4's search toward claims, the tail of a line, the search inside a frame (the tactical layer, in this plan).
//! LAWS: lean/ideas/Ideas/Layer0/Choice.lean (the windowed value is the true value when that lies inside the window, and a bound on the right side when it does not).
/// A node that gives its options when asked and is stepped into and back out of on the spot. A tree is never built: nothing of it exists but the line being walked.
pub trait Node {
    /// An option: what is needed to step into it.
    type Opt;
    /// Whether the asker picks at this node (the best option) or the rest do (the worst).
    fn own_turn(&self) -> bool;
    /// Whether the node is a choice by chance (the expectation of its options, weighed by their edges).
    fn chance(&self) -> bool { false }
    /// Fill `out` (empty) with the node's options in the node's own order (it makes the node live if it is not).
    fn options(&mut self, out: &mut Vec<Self::Opt>);
    /// Step into an option: the node it leads to becomes the current one.
    fn enter(&mut self, o: &Self::Opt);
    /// Step back out: the node entered last is left and the one before is current again.
    fn leave(&mut self);
}

/// The window of a number's reading: the asker already has `lo`, the rest already hold it to `hi`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bounds<V> { pub lo: V, pub hi: V }

/// What `Kind::before` says of a node: its value without opening it, or the token of the node to be opened.
pub enum Pre<V, T> { Done(V), Go(T) }

/// What a caller hands in to say what is asked of the walk and what steers it.
pub trait Kind<S: Node> {
    type Val: Copy + PartialOrd;
    type Err;
    type Edge: Copy;
    type Token;
    /// Whether windows cut (an exact number read in full has no window).
    const WINDOWED: bool;
    /// Whether the expectation is paid for (a kind without chance never calls `mix`).
    const CHANCE: bool;
    fn before(&mut self, space: &S, left: u32, ply: u32, win: &mut Bounds<Self::Val>) -> Result<Pre<Self::Val, Self::Token>, Self::Err>;
    fn horizon(&mut self, space: &S, token: Self::Token, ply: u32) -> Result<Self::Val, Self::Err>;
    fn base(&mut self, _space: &S, _token: &Self::Token) -> Option<Self::Val> { None }
    fn order(&mut self, _space: &S, _opts: &mut Vec<S::Opt>) {}
    fn edge(&mut self, space: &S, o: &S::Opt) -> Option<Self::Edge>;
    fn child(&self, _e: Self::Edge, win: &Bounds<Self::Val>) -> Bounds<Self::Val> { *win }
    fn lift(&self, _e: Self::Edge, v: Self::Val) -> Result<Self::Val, Self::Err> { Ok(v) }
    fn mix(&mut self, _acc: Option<Self::Val>, _e: Self::Edge, _v: Self::Val) -> Result<Self::Val, Self::Err> { unreachable!("a kind without chance is never asked to mix") }
    fn improved(&mut self, _space: &S, _ply: u32, _o: &S::Opt) {}
    fn open(&mut self, _token: &Self::Token) {}
    fn close(&mut self) {}
    fn finish(&mut self, space: &S, token: Self::Token, got: Option<Self::Val>, entry: &Bounds<Self::Val>) -> Result<Self::Val, Self::Err>;
}

/// THE WALK OF VALUES: the one recursion. It owns nothing of the tree and nothing of the answers: the node and the kind are the caller's.
pub struct Fold<S: Node, K: Kind<S>> {
    pub space: S,
    pub kind: K,
    /// the vectors of options of nodes already left, kept for their room: no node allocates once the walk is as deep as it will be
    spare: Vec<Vec<S::Opt>>,
}

impl<S: Node, K: Kind<S>> Fold<S, K> {
    pub fn new(space: S, kind: K) -> Self { Fold { space, kind, spare: Vec::new() } }

    /// The value at the node the space is at: `left` steps may still be taken, `ply` steps have been taken since the caller asked, `win` is the window. Best at an own node, worst at any other, the expectation at a chance node.
    pub fn go(&mut self, left: u32, ply: u32, mut win: Bounds<K::Val>) -> Result<K::Val, K::Err> { self.walk(left, ply, &mut win) }

    fn walk(&mut self, left: u32, ply: u32, win: &mut Bounds<K::Val>) -> Result<K::Val, K::Err> {
        let token = match self.kind.before(&self.space, left, ply, win)? { Pre::Done(v) => return Ok(v), Pre::Go(t) => t };
        if left == 0 { return self.kind.horizon(&self.space, token, ply); }
        let entry: Option<Bounds<K::Val>> = if K::WINDOWED { Some(*win) } else { None };
        let own = self.space.own_turn();
        let chance = K::CHANCE && self.space.chance();
        let mut got: Option<K::Val> = if chance { None } else { self.kind.base(&self.space, &token) };
        let mut skip = false;
        if K::WINDOWED {
            if let Some(b) = got { Self::improve(win, own, b); skip = win.lo >= win.hi; }
        }
        let mut opts: Vec<S::Opt> = if skip { Vec::new() } else { self.spare.pop().unwrap_or_default() };
        if !skip {
            self.space.options(&mut opts);
            self.kind.order(&self.space, &mut opts);
            self.kind.open(&token);
            if chance {
                let mut acc: Option<K::Val> = None;
                for o in &opts {
                    let Some(e) = self.kind.edge(&self.space, o) else { continue };
                    self.space.enter(o);
                    let r = self.walk(left - 1, ply + 1, win);
                    self.space.leave();
                    let v = match r { Ok(v) => v, Err(c) => { self.kind.close(); return Err(c); } };
                    acc = match self.kind.mix(acc, e, v) { Ok(a) => Some(a), Err(c) => { self.kind.close(); return Err(c); } };
                }
                got = acc;
            } else {
                for o in &opts {
                    let Some(e) = self.kind.edge(&self.space, o) else { continue };
                    self.space.enter(o);
                    let r = if K::WINDOWED { let mut cw = self.kind.child(e, win); self.walk(left - 1, ply + 1, &mut cw) } else { self.walk(left - 1, ply + 1, win) };
                    self.space.leave();
                    let v = match r { Ok(v) => v, Err(c) => { self.kind.close(); return Err(c); } };
                    let v = match self.kind.lift(e, v) { Ok(v) => v, Err(c) => { self.kind.close(); return Err(c); } };
                    let better = match got { None => true, Some(b) => if own { v > b } else { v < b } };
                    if better { got = Some(v); self.kind.improved(&self.space, ply, o); }
                    if K::WINDOWED {
                        Self::improve(win, own, v);
                        if win.lo >= win.hi { break; }
                    }
                }
            }
            self.kind.close();
            opts.clear();
            self.spare.push(opts);
        }
        match &entry { Some(e) => self.kind.finish(&self.space, token, got, e), None => self.kind.finish(&self.space, token, got, win) }
    }

    /// The window after a value is found: what the asker has can only rise at an own node, what the rest hold it to can only fall at any other.
    #[inline(always)]
    fn improve(win: &mut Bounds<K::Val>, own: bool, v: K::Val) {
        if own { if v > win.lo { win.lo = v; } } else if v < win.hi { win.hi = v; }
    }
}
