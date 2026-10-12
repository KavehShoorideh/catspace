//! PROOF-NUMBER SEARCH, generic (Allis 1994; Kaveh 2026-10-10: "take all of this proofing stuff into Rust" -- the Python
//! goal search (goal_search.py, 2026-09-06) had it, Rust did not). An AND/OR tree: at an OR node one child must be
//! proven, at an AND node all of them; the MOST-PROVING node is always expanded next (smallest proof number down the OR
//! nodes, smallest disproof number down the AND nodes). A problem plugs in through `Problem`: how to expand a node, how
//! to judge a node on sight, and the PRIOR on a child's proof number ("a prior to decide how far to follow each line --
//! unity for now; a reinforcement-learning agent later": the prior orders the work, the proof still decides). The search
//! ends when the root is proven or disproven; the node budget is only a safety net: hitting it is UNKNOWN.
use std::collections::HashMap;

pub const INF: f64 = f64::INFINITY;

pub trait Problem {
    type State: Clone;
    /// A node judged on sight (before expanding): Some(true) proven, Some(false) disproven, None open.
    fn judge(&self, s: &Self::State, or: bool, root: bool) -> Option<bool>;
    /// The children of an open node: (state, the prior weight on its proof number).
    fn expand(&self, s: &Self::State, or: bool) -> Vec<(Self::State, f64)>;
    /// The initial proof number of a fresh child of an OR node (an AND node to be): e.g. how many replies it leaves.
    fn and_size(&self, s: &Self::State) -> f64;
    /// A key for repetition (a repeated position on the path is disproven: a cycle wins nothing).
    fn key(&self, s: &Self::State) -> u64;
    /// THE STOP POLICY (Kaveh 2026-10-11: "a policy on when to stop, preferably RL"): may this child of an OR node (a
    /// line of ours, `depth` plies from the root) be followed? false: the line is dropped here -- a stop can only make a
    /// proof harder, never certify one. Default: always.
    fn allow(&self, _s: &Self::State, _depth: u32) -> bool { true }
    /// The state's features for the policy's log (None: not logged).
    fn features(&self, _s: &Self::State, _depth: u32) -> Option<Vec<f32>> { None }
}

/// THE POLICY LOG: every line of ours the search created, its features, whether it ended proven, whether the root did.
pub static LOG: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// (features, the line ended proven, the root was proven, the expansions spent in the line's subtree -- its cost)
thread_local! { pub static LOGV: std::cell::RefCell<Vec<(Vec<f32>, bool, bool, u32)>> = const { std::cell::RefCell::new(Vec::new()) }; }
pub fn take_log() -> Vec<(Vec<f32>, bool, bool, u32)> { LOGV.with(|v| std::mem::take(&mut *v.borrow_mut())) }
/// [lines created, lines stopped by the policy] since the last read.
pub static POLICY_N: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

struct Node<S> { s: S, or: bool, parent: Option<usize>, kids: Vec<usize>, pn: f64, dn: f64, open: bool, key: u64, depth: u32, feat: Option<Vec<f32>> }

/// [searches, proven, disproven, unknown, expansions] since the last read.
pub static STATS: [std::sync::atomic::AtomicU64; 5] = [const { std::sync::atomic::AtomicU64::new(0) }; 5];

/// Prove or disprove the root (an OR node). -> Some(true) proven, Some(false) disproven, None the budget ran out.
pub fn search<P: Problem>(p: &P, root: P::State, budget: u64) -> Option<bool> { search_with_proof(p, root, budget).0 }

/// `search`, and when proven the root's proving child (the first move of the proof).
pub fn search_with_proof<P: Problem>(p: &P, root: P::State, budget: u64) -> (Option<bool>, Option<P::State>) {
    let (r, first, _) = run(p, root, budget);
    (r, first)
}

/// THE PROOF AS LINES (for digging and for the explanation): the search, then the proof tree walked -- at a node of ours
/// the first proven child, at a node of theirs every child -- each line as the states along it, at most `max` lines.
pub fn search_proof_lines<P: Problem>(p: &P, root: P::State, budget: u64, max: usize) -> (Option<bool>, Vec<Vec<P::State>>) {
    let (r, _, t) = run(p, root, budget);
    let mut out: Vec<Vec<P::State>> = Vec::new();
    if r != Some(true) { return (r, out); }
    fn walk<S: Clone>(t: &[Node<S>], i: usize, path: &mut Vec<S>, out: &mut Vec<Vec<S>>, max: usize) {
        if out.len() >= max { return; }
        path.push(t[i].s.clone());
        let kids: Vec<usize> = if t[i].or { t[i].kids.iter().copied().filter(|k| t[*k].pn == 0.0).take(1).collect() } else { t[i].kids.clone() };
        if kids.is_empty() { out.push(path.clone()); } else { for k in kids { walk(t, k, path, out, max); } }
        path.pop();
    }
    let mut path = Vec::new();
    walk(&t, 0, &mut path, &mut out, max);
    (r, out)
}

fn run<P: Problem>(p: &P, root: P::State, budget: u64) -> (Option<bool>, Option<P::State>, Vec<Node<P::State>>) {
    use std::sync::atomic::Ordering::Relaxed;
    let rk = p.key(&root);
    let logging = LOG.load(Relaxed);
    let mut t: Vec<Node<P::State>> = vec![Node { s: root, or: true, parent: None, kids: vec![], pn: 1.0, dn: 1.0, open: true, key: rk, depth: 0, feat: None }];
    let mut expanded = 0u64;
    let fin = |r: Option<bool>, e: u64| {
        STATS[0].fetch_add(1, Relaxed);
        STATS[match r { Some(true) => 1, Some(false) => 2, None => 3 }].fetch_add(1, Relaxed);
        STATS[4].fetch_add(e, Relaxed);
        r
    };
    let _ = HashMap::<u64, ()>::new();
    while t[0].pn > 0.0 && t[0].dn > 0.0 {
        if expanded >= budget { if logging { log_tree(&mut t, false); } return (fin(None, expanded), None, t); }
        let mut i = 0usize;
        while !t[i].open {
            let pick_pn = t[i].or;
            i = *t[i].kids.iter().min_by(|a, b| {
                let (x, y) = if pick_pn { (t[**a].pn, t[**b].pn) } else { (t[**a].dn, t[**b].dn) };
                x.partial_cmp(&y).unwrap()
            }).unwrap();
        }
        expanded += 1;
        t[i].open = false;
        let or = t[i].or;
        // judged on sight, or a repetition on the path
        let repeated = { let mut a = t[i].parent; let mut r = false; while let Some(j) = a { if t[j].key == t[i].key && t[j].or == or { r = true; break; } a = t[j].parent; } r };
        match (repeated, p.judge(&t[i].s, or, i == 0)) {
            (true, _) => { t[i].pn = INF; t[i].dn = 0.0; }
            (_, Some(true)) => { t[i].pn = 0.0; t[i].dn = INF; }
            (_, Some(false)) => { t[i].pn = INF; t[i].dn = 0.0; }
            (false, None) => {
                let kids = p.expand(&t[i].s, or);
                if kids.is_empty() {
                    // an OR node with no move: disproven; an AND node with no reply: proven (judge() handles mates)
                    if or { t[i].pn = INF; t[i].dn = 0.0; } else { t[i].pn = 0.0; t[i].dn = INF; }
                }
                let depth = t[i].depth + 1;
                for (s, prior) in kids {
                    let child_or = !or;
                    // the stop policy, on our lines only (the children of an OR node)
                    if or { POLICY_N[0].fetch_add(1, Relaxed); if !p.allow(&s, depth) { POLICY_N[1].fetch_add(1, Relaxed); continue; } }
                    let feat = if logging && or { p.features(&s, depth) } else { None };
                    let k = p.key(&s);
                    // a child judged on sight costs no expansion (a reply that ignores our threat is proven when made)
                    let (pn, dn, open) = match p.judge(&s, child_or, false) {
                        Some(true) => (0.0, INF, false),
                        Some(false) => (INF, 0.0, false),
                        None => if child_or { (prior, 1.0, true) } else { (p.and_size(&s) * prior, 1.0, true) },
                    };
                    t.push(Node { s, or: child_or, parent: Some(i), kids: vec![], pn, dn, open, key: k, depth, feat });
                    let c = t.len() - 1;
                    t[i].kids.push(c);
                }
                // every line stopped by the policy: an OR node with no child left is disproven (an AND node cannot lose
                // children -- the policy acts on our lines only)
                if t[i].kids.is_empty() && or { t[i].pn = INF; t[i].dn = 0.0; }
            }
        }
        let mut a = Some(i);
        while let Some(j) = a {
            if !t[j].kids.is_empty() {
                let (pn, dn) = if t[j].or {
                    (t[j].kids.iter().map(|k| t[*k].pn).fold(INF, f64::min), t[j].kids.iter().map(|k| t[*k].dn).sum::<f64>())
                } else {
                    (t[j].kids.iter().map(|k| t[*k].pn).sum::<f64>(), t[j].kids.iter().map(|k| t[*k].dn).fold(INF, f64::min))
                };
                t[j].pn = pn; t[j].dn = dn;
            }
            a = t[j].parent;
        }
    }
    let r = Some(t[0].pn == 0.0);
    let first = if t[0].pn == 0.0 { t[0].kids.iter().find(|k| t[**k].pn == 0.0).map(|k| t[*k].s.clone()) } else { None };
    if logging { log_tree(&mut t, r == Some(true)); }
    (fin(r, expanded), first, t)
}

fn log_tree<S>(t: &mut [Node<S>], root_proven: bool) {
    // the cost of a line: the expanded nodes in its subtree (each expanded node adds one to every ancestor)
    let mut cost = vec![0u32; t.len()];
    for i in 0..t.len() { if !t[i].open { let mut a = t[i].parent; while let Some(j) = a { cost[j] += 1; a = t[j].parent; } } }
    LOGV.with(|v| { let mut v = v.borrow_mut(); for (i, n) in t.iter_mut().enumerate() { if let Some(f) = n.feat.take() { v.push((f, n.pn == 0.0, root_proven, cost[i])); } } });
}
