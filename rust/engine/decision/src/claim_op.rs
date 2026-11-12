//! PLAN 78.36 THE CLAIM OPERATOR: the search toward claims (cs_tactics::toward) with the claims of a snapshot (cs_memory::claim) as its leaf oracle. It answers
//! `Goal::Toward`. What it returns is read as EVIDENCE by the controller: a root move that reaches a claimed (or the game's own) win is PROVED and played; otherwise the moves
//! of the best class (win or draw) are the order the ledger reads first, and the moves a claim refutes (class loss) are listed as `refuted` facts so that the ledger reads them last.
//! The claims are the store's, admitted by the training side's drill; nothing of how they were found is here.
use cs_game::{Fact, Game};
use cs_memory::claim::{Claims, Class};
use cs_operator::{Budget, Goal, Operator, Outcome, Status};
use cs_tactics::toward::{self, LeafOracle, Verdict};
use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap, HashSet};

/// The deepest reading asked when the budget names none (a named cap; the clock and the node fuse are the real bound).
pub const CLAIM_DEPTH: u32 = 16;

/// The claims as a leaf oracle: the class of a position for its side to move, kept by the position's key (a position reached again in the deepening costs nothing), with how often
/// each rule spoke.
struct ClaimOracle<'a> {
    claims: &'a Claims<'a>,
    memo: RefCell<HashMap<u64, Option<(Class, String)>>>,
    spoke: RefCell<BTreeMap<String, u64>>,
}

impl<'a, G: Game> LeafOracle<G> for ClaimOracle<'a> {
    fn verdict(&self, g: &G, s: &G::State) -> Option<Verdict> {
        let key = g.key(s);
        let cached = self.memo.borrow().get(&key).cloned();
        let said = match cached { Some(x) => x, None => { let x = self.claims.class_at(g, s); self.memo.borrow_mut().insert(key, x.clone()); x } };
        let (class, name) = said?;
        *self.spoke.borrow_mut().entry(name).or_insert(0) += 1;
        Some(match class { Class::Win => Verdict::Win, Class::Draw => Verdict::Draw, Class::Loss => Verdict::Loss })
    }
}

/// The case the controller reads the classes by (plan 78.36): "any W" (a win by claim or game: the W moves first, every other move avoided), "some U" (no W: only the refuted moves are
/// avoided; a draw has no preference over an unknown), "all D or L" (draws first, refuted avoided), "all L" (nothing changes).
pub fn case_of(w: usize, d: usize, u: usize, l: usize) -> &'static str {
    if w > 0 { "any W" } else if u > 0 { "no W, some U" } else if d > 0 { "every move D or L" } else if l > 0 { "every move L: nothing changes" } else { "no moves" }
}

/// One root move of a `claim` step: its index, its class word (W, D, U, L), whether a win rests on the game alone, and its value (higher is better; nearer win).
#[derive(Debug, Clone, PartialEq)]
pub struct RootClass { pub index: usize, pub class: char, pub game: bool, pub value: i32 }

/// The classes of the root moves a `claim` step reported, read off its facts (empty without them).
pub fn root_classes_of<M>(o: &Outcome<M>) -> Vec<RootClass> {
    o.facts.iter().filter(|f| f.relation == "root_class" && f.args.len() == 4).filter_map(|f| Some(RootClass {
        index: f.args[0].parse().ok()?, class: f.args[1].chars().next()?, game: f.args[2] == "game", value: f.args[3].parse().ok()? })).collect()
}

/// `claim`: the search toward claims. `history`: the keys of the positions the game's record holds (a position met again is the rules' draw).
pub struct ClaimOp<'a> {
    pub claims: &'a Claims<'a>,
    pub history: Vec<u64>,
}

/// The root moves a `claim` step refuted (indices into the legal moves), read off its facts.
pub fn refuted_of<M>(o: &Outcome<M>) -> Vec<usize> {
    o.facts.iter().filter(|f| f.relation == "refuted").filter_map(|f| f.args.first().and_then(|a| a.parse().ok())).collect()
}

impl<'a, G: Game> Operator<G> for ClaimOp<'a> {
    fn name(&self) -> String { "claim".into() }
    fn accepts(&self, goal: &Goal) -> bool { matches!(goal, Goal::Toward { .. }) }
    fn run(&self, g: &G, s: &G::State, goal: &Goal, mut b: Budget) -> Outcome<G::Move> {
        let Goal::Toward { role } = goal else { return Outcome::unsupported("claim", goal) };
        let t0 = std::time::Instant::now();
        let before = self.claims.stats();
        let oracle = ClaimOracle { claims: self.claims, memo: RefCell::new(HashMap::new()), spoke: RefCell::new(BTreeMap::new()) };
        let hist: HashSet<u64> = self.history.iter().copied().collect();
        let r = toward::toward(g, s, *role, &oracle, &hist, b.max_depth.unwrap_or(CLAIM_DEPTH), &mut b);
        let after = self.claims.stats();
        let moves = g.moves(s);
        let name = |i: usize| g.move_name(s, &moves[i]);
        let wins = r.of_class(toward::Class::Win);
        let game_wins = r.game_wins();
        let best = r.best_class();
        // PLAN 78.36 (the revision): only the GAME's own win is proved and played at once; a claimed win is a class (W), ordered first by the controller, never played at once
        let mut o = Outcome::new(if !game_wins.is_empty() { Status::Proved } else if r.depth == 0 { Status::Gap } else { Status::Read });
        o.depth = Some(r.depth as i64);
        if let Some(i) = game_wins.first() { o.mv = Some(moves[*i].clone()); }
        o.candidates = wins.clone();
        let refuted = r.of_class(toward::Class::Loss);
        for (i, class, val) in &r.root {
            let tag = match class { toward::Class::Win => "W", toward::Class::Draw => "D", toward::Class::Unknown => "U", toward::Class::Loss => "L" };
            let source = if *class == toward::Class::Win { if toward::is_game_win(*val) { "game" } else { "claim" } } else { "" };
            o.facts.push(Fact { relation: "root_class".into(), args: vec![i.to_string(), tag.into(), source.into(), val.to_string()] });
        }
        for i in &refuted { o.facts.push(Fact { relation: "refuted".into(), args: vec![i.to_string(), name(*i)] }); }
        let spoke = oracle.spoke.borrow().clone();
        let mut by_count: Vec<(&String, &u64)> = spoke.iter().collect();
        by_count.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
        for (rule, n) in &by_count { o.facts.push(Fact { relation: "spoke".into(), args: vec![(*rule).clone(), n.to_string()] }); }
        for rd in &r.reads { o.part(format!("depth {}{}", rd.depth, if rd.complete { "" } else { " (cut)" }), rd.nodes, rd.seconds); }
        let secs = t0.elapsed().as_secs_f64();
        o.cost.nodes = r.nodes;
        o.cost.seconds = secs;
        let how = if r.silent { "stopped: nothing spoke" } else if r.exact { "every line read to a result" } else if r.cut { "cut by the budget" } else { "to the depth asked" };
        o.evidence.push(format!("claim toward: depth {} ({how}), {} nodes, {} claim evaluations ({} real, {} spoke, {} conflicts), {:.3} s ({:.4} s in the claims)",
                                r.depth, r.nodes, r.evals, after.evaluations - before.evaluations, after.spoke - before.spoke, after.conflicts - before.conflicts, secs, after.seconds - before.seconds));
        let rules = if by_count.is_empty() { "no claim spoke".to_string() } else { format!("spoke: {}", by_count.iter().take(4).map(|(n, c)| format!("{n} x{c}")).collect::<Vec<_>>().join(", ")) };
        let shown: Vec<String> = refuted.iter().take(6).map(|i| name(*i)).collect();
        let refuted_text = if refuted.is_empty() { String::new() } else { format!("; refuted: {}{}", shown.join(" "), if refuted.len() > shown.len() { " ..." } else { "" }) };
        let (nw, nd, nu, nl) = (r.count(toward::Class::Win), r.count(toward::Class::Draw), r.count(toward::Class::Unknown), r.count(toward::Class::Loss));
        let case = case_of(nw, nd, nu, nl);
        let silent = if r.silent { format!("no claim speaks within {} plies: stopped; ", toward::SILENT_DEPTH) } else { String::new() };
        if let Some(i) = game_wins.first() {
            o.evidence.push(format!("claim toward: PROVED -- {} reaches the game's own win against every reply; case {case}: root moves W {nw} (game {}), D {nd}, U {nu}, L {nl}; {rules}{refuted_text}", name(*i), game_wins.len()));
        } else {
            o.evidence.push(format!("claim toward: {silent}case {case}: root moves W {nw} (game {}), D {nd}, U {nu}, L {nl}; {rules}{refuted_text}", game_wins.len()));
        }
        let _ = best;
        o
    }
}
