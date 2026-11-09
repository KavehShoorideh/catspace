//! plan 78: THE END-TO-END SYSTEM's Python entry -- the harness calls in once per move (or once per certification) and reads the result; no per-node crossing.
use cs_game::{certify, gdl_game::GdlGame, kernel, Game};
use pyo3::prelude::*;
use std::collections::HashMap;

fn state_of(variant: &str, fen: Option<&str>, start960: u32, moves: &[String]) -> PyResult<(kernel::KernelGame, shakmaty_state::S)> {
    let (g, s, _h) = record_of(variant, fen, start960, moves)?;
    Ok((g, s))
}

/// The game, the state after `moves`, and the INSTANCE's record: the keys of every position before it (plan 78.8).
fn record_of(variant: &str, fen: Option<&str>, start960: u32, moves: &[String]) -> PyResult<(kernel::KernelGame, shakmaty_state::S, Vec<u64>)> {
    let g = match fen { Some(f) => kernel::KernelGame::from_fen(variant, f), None => kernel::KernelGame::new(variant, start960) }
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    let mut s = g.initial();
    let mut keys = vec![];
    for u in moves {
        let m = g.parse(&s, u).ok_or_else(|| pyo3::exceptions::PyValueError::new_err(format!("illegal move {u}")))?;
        keys.push(g.key(&s));
        s = g.apply(&s, &m);
    }
    Ok((g, s, keys))
}

mod shakmaty_state { pub type S = <cs_game::kernel::KernelGame as cs_game::Game>::State; }

/// best_move(variant, fen=None, moves=[], seconds=1.0, worth={}, start960=518) -> (move or None, proved, [trace], nodes, fen after the moves): THE END-TO-END
/// DECISION (plan 78.5: tactics `forcing`, then the ledger search) for the side to move after `moves` from the start (or from `fen`). The ledger's JUDGE is the
/// store's valuation record in use, passed in: `worth` (kind name -> worth), `ours_first`, `valuation` (its name, for the trace); an empty table: the built-in Count.
#[pyfunction]
#[pyo3(signature = (variant, fen=None, moves=vec![], seconds=1.0, worth=HashMap::new(), start960=518, ours_first=false, valuation=String::new(), prior=false, record=true, prior_depth=2, ledger_share=0.6, prior_defects=1, ordering=true, prune=true, extend=true))]
#[allow(clippy::too_many_arguments)]
fn best_move(variant: &str, fen: Option<&str>, moves: Vec<String>, seconds: f64, worth: HashMap<String, i64>, start960: u32, ours_first: bool, valuation: String,
             prior: bool, record: bool, prior_depth: i32, ledger_share: f64, prior_defects: u32, ordering: bool, prune: bool, extend: bool)
             -> PyResult<(Option<String>, bool, Vec<String>, u64, String)> {
    let (g, s, keys) = record_of(variant, fen, start960, &moves)?;
    let hist: Vec<u64> = if record { keys } else { vec![] };          // (record=False: the game's record not read -- the comparison)
    let f = g.fen(&s);
    let choice = if worth.is_empty() { cs_decision::decide_with(&g, &s, seconds, &cs_tactics::ledger::Count, prior, &hist, prior_depth, ledger_share, prior_defects, ordering, prune, extend) }
                 else { cs_decision::decide_with(&g, &s, seconds, &cs_tactics::ledger::Worth { name: valuation, table: worth.clone(), ours_first }, prior, &hist, prior_depth, ledger_share, prior_defects, ordering, prune, extend) };
    Ok(match choice {
        Some(c) => (Some(g.move_name(&s, &c.mv)), c.proved, c.trace, c.nodes, f),
        None => (None, false, vec!["the game is over".into()], 0, f),
    })
}

/// state(variant, fen=None, moves=[], start960=518) -> (fen, the mover's name or None, [legal moves], the outcome per role or None): the harness's view of a state.
#[pyfunction]
#[pyo3(signature = (variant, fen=None, moves=vec![], start960=518))]
fn state(variant: &str, fen: Option<&str>, moves: Vec<String>, start960: u32) -> PyResult<(String, Option<String>, Vec<String>, Option<Vec<i64>>, u64)> {
    let (g, s) = state_of(variant, fen, start960, &moves)?;
    let ms = g.moves(&s).iter().map(|m| g.move_name(&s, m)).collect();
    Ok((g.fen(&s), g.mover(&s).map(|r| g.roles()[r].clone()), ms, g.outcome(&s), g.key(&s)))
}

/// The variants the kernels play (lichess's names).
#[pyfunction]
fn variants() -> Vec<String> { kernel::VARIANTS.iter().map(|s| s.to_string()).collect() }

/// certify_coplay(variant, gdl_text, games, seed, base=None) -> (games, plies, states, [disagreements]): the chess kernel of `variant` against the GDL text (a
/// delta composed on `base` when given), by co-play.
#[pyfunction]
#[pyo3(signature = (variant, gdl_text, games, seed, base=None))]
fn certify_coplay(variant: &str, gdl_text: &str, games: usize, seed: u64, base: Option<&str>) -> PyResult<(usize, usize, usize, Vec<String>)> {
    let k = kernel::KernelGame::new(variant, 518).map_err(pyo3::exceptions::PyValueError::new_err)?;
    let text = match base { Some(b) => cs_gdl::compose(b, gdl_text).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.0))?, None => gdl_text.to_string() };
    let g = GdlGame::load(variant, &text).map_err(pyo3::exceptions::PyValueError::new_err)?;
    let r = certify::coplay(&k, &g, &|n| kernel::uci_to_spec_term(n), games, seed);
    Ok((r.games, r.plies, r.states, r.disagreements))
}

/// watch(variant, engine, games, random_plies=6, movetime_ms=20, seed=78, threads=11, options={}) -> [(start, [moves], outcome)]: plan 78.4 -- instances by
/// WATCHING an outside UCI engine play itself (seeded random openings; Chess960: each game its own start, seeded); the game judged by the kernel.
#[pyfunction]
#[pyo3(signature = (variant, engine, games, random_plies=6, movetime_ms=20, seed=78, threads=11, options=HashMap::new()))]
fn watch(variant: &str, engine: &str, games: usize, random_plies: usize, movetime_ms: u64, seed: u64, threads: usize, options: HashMap<String, String>)
         -> PyResult<Vec<(String, Vec<String>, Vec<i64>)>> {
    let mut opts: Vec<(String, String)> = options.into_iter().collect();
    if variant == "chess960" { opts.push(("UCI_Chess960".into(), "true".into())); } else if variant != "chess" { opts.push(("UCI_Variant".into(), variant.into())); }
    kernel::KernelGame::new(variant, 518).map_err(pyo3::exceptions::PyValueError::new_err)?;
    let v = variant.to_string();
    struct T(cs_game::outside::UciTeacher);
    impl cs_learn::watch::Teacher for T {
        fn name(&self) -> String { "uci".into() }
        fn start(&mut self, start: &str) { self.0.begin(start) }
        fn best(&mut self, moves: &[String]) -> Option<String> { self.0.answer(moves) }
    }
    let recs = cs_learn::watch::watch(|i| {
        if v == "chess960" {
            let n = (cs_game::rng::Rng::new(seed ^ (0x960 + i as u64)).next_u64() % 960) as u32;
            let f = kernel::chess960_fen(n);
            (kernel::KernelGame::from_fen("chess960", &f).unwrap(), format!("fen {f}"), f)
        } else {
            (kernel::KernelGame::new(&v, 518).unwrap(), "startpos".to_string(), String::new())
        }
    }, || cs_game::outside::UciTeacher::new(engine, &opts, movetime_ms).map(T), games, random_plies, seed, threads);
    Ok(recs.into_iter().map(|r| (r.start, r.moves, r.outcome)).collect())
}

/// mine_valuation(variant, records, from_ply=10, fuse=64) -> {kinds, constant, beta, band, logloss_best, se_best, worth, price, logloss, excess, se, tried,
/// positions, games, groups}: plan 78.4 -- THE VALUATION MINER on the records [(start, [moves], outcome)] (cs_learn::valuation).
#[pyfunction]
#[pyo3(signature = (variant, records, from_ply=10, fuse=64))]
fn mine_valuation(py: Python<'_>, variant: &str, records: Vec<(String, Vec<String>, Vec<i64>)>, from_ply: usize, fuse: u32) -> PyResult<PyObject> {
    let recs: Vec<cs_learn::Record> = records.into_iter().map(|(start, moves, outcome)| cs_learn::Record { start, moves, outcome }).collect();
    let v = variant.to_string();
    let c = cs_learn::valuation::corpus(|r: &cs_learn::Record| {
        if r.start.is_empty() { kernel::KernelGame::new(&v, 518).ok() } else { kernel::KernelGame::from_fen(&v, &r.start).ok() }
    }, &recs, from_ply);
    let f = cs_learn::valuation::search(&c, fuse);
    let d = pyo3::types::PyDict::new(py);
    d.set_item("kinds", f.kinds)?; d.set_item("constant", f.constant)?; d.set_item("beta", f.beta)?; d.set_item("band", f.band)?;
    d.set_item("logloss_best", f.logloss_best)?; d.set_item("se_best", f.se_best)?; d.set_item("worth", f.worth)?; d.set_item("price", f.price)?;
    d.set_item("logloss", f.logloss)?; d.set_item("excess", f.excess)?; d.set_item("se", f.se)?; d.set_item("tried", f.tried)?;
    d.set_item("positions", f.positions)?; d.set_item("games", f.games)?; d.set_item("groups", f.groups)?;
    Ok(d.into())
}

/// census(variant, fen=None, moves=[], max_defects=1, depth=3, fuse=200000, side="mover", worth={}, ours_first=false, valuation="") -> (tactics, stats):
/// plan 78.6 THE CENSUS OF TACTICS BY THEIR DEFECTS (cs_tactics::census) of the side to move ("mover") or of the other side ("other": read after a pass) --
/// tactics: [{defects, free, line, wins, ledger, value}]; stats: {readings, cut, nodes, frontier}. The judge: the store's record, as best_move takes it.
#[pyfunction]
#[pyo3(signature = (variant, fen=None, moves=vec![], max_defects=1, depth=3, fuse=200000, side="mover", worth=HashMap::new(), ours_first=false, valuation=String::new()))]
#[allow(clippy::too_many_arguments)]
fn census(py: Python<'_>, variant: &str, fen: Option<&str>, moves: Vec<String>, max_defects: u32, depth: i32, fuse: u64, side: &str, worth: HashMap<String, i64>,
          ours_first: bool, valuation: String) -> PyResult<(Vec<PyObject>, PyObject)> {
    let (g, s0) = state_of(variant, fen, 518, &moves)?;
    let s = if side == "other" { match g.pass(&s0) { Some(q) => q, None => return Ok((vec![], pyo3::types::PyDict::new(py).into())) } } else { s0 };
    let rep = if worth.is_empty() { cs_tactics::census::census(&g, &s, &cs_tactics::ledger::Count, max_defects, depth, fuse) }
              else { cs_tactics::census::census(&g, &s, &cs_tactics::ledger::Worth { name: valuation, table: worth, ours_first }, max_defects, depth, fuse) };
    let mut out = vec![];
    for t in rep.tactics {
        let d = pyo3::types::PyDict::new(py);
        d.set_item("defects", t.defects)?; d.set_item("free", t.free)?; d.set_item("line", t.line)?; d.set_item("wins", t.wins)?; d.set_item("targets", t.targets)?;
        d.set_item("ledger", t.ledger)?; d.set_item("value", t.value)?;
        out.push(d.into());
    }
    let st = pyo3::types::PyDict::new(py);
    st.set_item("readings", rep.readings)?; st.set_item("cut", rep.cut)?; st.set_item("nodes", rep.nodes)?; st.set_item("frontier", rep.frontier)?;
    Ok((out, st.into()))
}

type BatchItem = (String, Option<String>, Vec<String>);
type BatchRow = (Option<String>, bool, u64, f64, Option<String>);

/// The snapshot a batch reads and the descriptions its lemmas are bound through (per variant: the text and its content id).
struct Store { snapshot: cs_memory::snapshot::Snapshot, specs: HashMap<String, (String, String)> }

thread_local! {
    /// One fact view per (variant, description, snapshot) on each worker: compiling a description is costly, reading a position through it is not.
    static VIEWS: std::cell::RefCell<HashMap<String, std::rc::Rc<cs_game::gdl_game::FactView>>> = std::cell::RefCell::new(HashMap::new());
}

fn view_for(store: &Store, variant: &str) -> Option<std::rc::Rc<cs_game::gdl_game::FactView>> {
    let (text, id) = store.specs.get(variant)?;
    let key = format!("{variant}|{id}|{}", store.snapshot.id);
    VIEWS.with(|v| {
        if let Some(x) = v.borrow().get(&key) { return Some(x.clone()); }
        let view = std::rc::Rc::new(cs_game::gdl_game::FactView::new(text, store.snapshot.vocabulary.clone()).ok()?);
        v.borrow_mut().insert(key, view.clone());
        Some(view)
    })
}

/// One position of a batch: the game and state after the item's moves (its record read unless `record` is off), then the fixed controller's decision -- with the
/// snapshot's lemma operator when the batch has a store and the variant a description; the row for the log and the summary for the caller. A position that does
/// not parse, or is over, is an error row; the batch goes on.
fn batch_one<J: cs_tactics::ledger::Judge>(i: usize, it: &BatchItem, label: &str, judge: &J, seconds: f64, snapshot: &str, params: &cs_decision::Params, record: bool,
                                           store: Option<&Store>, ablate: &std::collections::BTreeSet<String>) -> (cs_decision::batch::Summary, serde_json::Value) {
    use cs_decision::batch;
    let label = if label.is_empty() { format!("{} {} {}", it.0, it.1.as_deref().unwrap_or("startpos"), it.2.join(" ")) } else { label.to_string() };
    let (g, s, keys) = match record_of(&it.0, it.1.as_deref(), 518, &it.2) {
        Ok(x) => x,
        Err(e) => return batch::error_row(i, snapshot, &label, &format!("{e}")),
    };
    let mut p = params.clone();
    p.history = if record { keys } else { vec![] };
    let view = store.and_then(|st| if st.snapshot.lemmas.is_empty() { None } else { view_for(st, &it.0) });
    let decision = match (store, &view) {
        (Some(st), Some(v)) => {
            let mut op = cs_memory::op::LemmaOp::new(&st.snapshot, v);
            op.exclude = ablate.clone();
            cs_decision::decide_with_lemmas(&g, &s, seconds, judge, &p, Some(&op as &dyn cs_operator::Operator<kernel::KernelGame>))
        }
        _ => cs_decision::decide_params(&g, &s, seconds, judge, &p),
    };
    match decision {
        Some(d) => batch::row(i, snapshot, &label, &g, &s, &d),
        None => batch::error_row(i, snapshot, &label, "the game is over"),
    }
}

fn load_store(path: &str, specs: HashMap<String, String>) -> PyResult<Store> {
    let snapshot = cs_memory::snapshot::Snapshot::load(std::path::Path::new(path)).map_err(pyo3::exceptions::PyValueError::new_err)?;
    let specs = specs.into_iter().map(|(k, v)| { let id = cs_memory::sha::content_id(v.as_bytes()); (k, (v, id)) }).collect();
    Ok(Store { snapshot, specs })
}

/// solve_batch(positions, seconds=1.0, worth={}, ours_first=False, valuation="", snapshot="", trace_path=None, threads=0, prior=False, record=True, prior_depth=2,
/// ledger_share=0.6, prior_defects=1, ordering=True, prune=True, extend=True, store=None, specs={}, labels=None, ablate=[])
/// -> [(move or None, proved, nodes, seconds, error or None)]: plan 78.12 / 78.14 THE BATCH SOLVER -- `positions` is [(variant, fen or None, moves)], all decided in ONE
/// call on `threads` workers (0: one per core) with the Python lock released, one summary per position in order. `seconds` is each position's own clock (positions run
/// side by side: keep `threads` within the cores, or the clocks measure the contention). `store` is a snapshot file (<id>.json, which must hash to its name): its id
/// tags the log, its judge is used when `worth` is empty, and its lemmas are bound through `specs` ({variant: description text}) by the lemma operator, whose proposals
/// the ledger search reads first (`ablate`: lemma names left out). `labels` names the positions in the log. With `trace_path` every decision's steps are appended to
/// that JSON Lines log.
#[pyfunction]
#[pyo3(signature = (positions, seconds=1.0, worth=HashMap::new(), ours_first=false, valuation=String::new(), snapshot=String::new(), trace_path=None, threads=0, prior=false, record=true, prior_depth=2, ledger_share=0.6, prior_defects=1, ordering=true, prune=true, extend=true, store=None, specs=HashMap::new(), labels=None, ablate=vec![], ties=false))]
#[allow(clippy::too_many_arguments)]
fn solve_batch(py: Python<'_>, positions: Vec<BatchItem>, seconds: f64, worth: HashMap<String, i64>, ours_first: bool, valuation: String, snapshot: String,
               trace_path: Option<String>, threads: usize, prior: bool, record: bool, prior_depth: i32, ledger_share: f64, prior_defects: u32, ordering: bool, prune: bool,
               extend: bool, store: Option<String>, specs: HashMap<String, String>, labels: Option<Vec<String>>, ablate: Vec<String>, ties: bool) -> PyResult<Vec<BatchRow>> {
    let params = cs_decision::Params { prior, ties, prior_depth, ledger_share, prior_defects, ordering, prune, extend, ..cs_decision::Params::default() };
    let store = match store { Some(p) => Some(load_store(&p, specs)?), None => None };
    let snapshot = store.as_ref().map(|s| s.snapshot.id.clone()).unwrap_or(snapshot);
    if let Some(l) = &labels { if l.len() != positions.len() { return Err(pyo3::exceptions::PyValueError::new_err("labels: one per position")); } }
    let ablate: std::collections::BTreeSet<String> = ablate.into_iter().collect();
    // the snapshot's judge stands in when the caller gives none
    let judge_worth: Option<(HashMap<String, i64>, bool, String)> = if worth.is_empty() {
        store.as_ref().and_then(|s| s.snapshot.judge.as_ref()).map(|j| (j.worth.iter().map(|(k, v)| (k.clone(), *v)).collect(), j.ours_first, j.name.clone()))
    } else { Some((worth.clone(), ours_first, valuation.clone())) };
    let lab = |i: usize| labels.as_ref().map(|l| l[i].clone()).unwrap_or_default();
    let done = py.allow_threads(|| match &judge_worth {
        None => cs_decision::batch::solve(&positions, threads, &snapshot, |i, it| batch_one(i, it, &lab(i), &cs_tactics::ledger::Count, seconds, &snapshot, &params, record, store.as_ref(), &ablate)),
        Some((w, of, name)) => {
            let j = cs_tactics::ledger::Worth { name: name.clone(), table: w.clone(), ours_first: *of };
            cs_decision::batch::solve(&positions, threads, &snapshot, |i, it| batch_one(i, it, &lab(i), &j, seconds, &snapshot, &params, record, store.as_ref(), &ablate))
        }
    });
    if let Some(path) = trace_path {
        let rows: Vec<serde_json::Value> = done.iter().map(|(_, v)| v.clone()).collect();
        cs_decision::batch::append(std::path::Path::new(&path), &rows).map_err(|e| pyo3::exceptions::PyOSError::new_err(format!("{path}: {e}")))?;
    }
    Ok(done.into_iter().map(|(s, _)| (s.mv, s.proved, s.nodes, s.seconds, s.error)).collect())
}

fn vocab_of(json: &str) -> PyResult<cs_game::Vocabulary> {
    let v: serde_json::Value = serde_json::from_str(json).map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("vocabulary: {e}")))?;
    cs_memory::snapshot::vocabulary_from_json(&v).map_err(pyo3::exceptions::PyValueError::new_err)
}

/// facts(variant, description, vocabulary, fen=None, moves=[], perspective=True) -> [[relation, arg ...]]: the FACT LAYER of a position -- the kernel's base facts
/// (certified against the description by co-play) and the derived relations the vocabulary (JSON) names, evaluated through the description; in the mover's
/// perspective (`us` / `them`) by default.
#[pyfunction]
#[pyo3(signature = (variant, spec, vocabulary, fen=None, moves=vec![], perspective=true))]
fn facts(variant: &str, spec: &str, vocabulary: &str, fen: Option<&str>, moves: Vec<String>, perspective: bool) -> PyResult<Vec<Vec<String>>> {
    let (g, s) = state_of(variant, fen, 518, &moves)?;
    let view = cs_game::gdl_game::FactView::new(spec, vocab_of(vocabulary)?).map_err(pyo3::exceptions::PyValueError::new_err)?;
    let base = g.base_facts(&s);
    if base.is_empty() { return Err(pyo3::exceptions::PyValueError::new_err(format!("{variant} gives no facts (no description is certified for it)"))); }
    let mut fs = view.facts(&base).map_err(pyo3::exceptions::PyValueError::new_err)?;
    if perspective { if let Some(me) = g.mover(&s) { fs = cs_memory::perspective(&fs, &g.roles(), me); } }
    Ok(fs.into_iter().map(|f| { let mut v = vec![f.relation]; v.extend(f.args); v }).collect())
}

/// The product of two priors' weights (the gain prior and the lemma prior together).
#[derive(Clone)]
struct ProductPrior<A: cs_planning::rollout::Prior + Clone, B: cs_planning::rollout::Prior + Clone>(A, B);
impl<A: cs_planning::rollout::Prior + Clone, B: cs_planning::rollout::Prior + Clone> cs_planning::rollout::Prior for ProductPrior<A, B> {
    fn weights<G: Game>(&self, g: &G, s: &G::State, moves: &[G::Move]) -> Vec<f64> {
        self.0.weights(g, s, moves).into_iter().zip(self.1.weights(g, s, moves)).map(|(a, b)| a * b).collect()
    }
    fn name(&self) -> String { format!("{} x {}", self.0.name(), self.1.name()) }
}

/// The rollout and leaf names the operator's evidence line states ("... rollout <name>, leaf <name>, value for role ...").
fn playout_names(evidence: &[String]) -> (String, String) {
    for e in evidence {
        if let (Some(a), Some(b), Some(c)) = (e.find(", rollout "), e.find(", leaf "), e.find(", value for role")) {
            if a < b && b < c { return (e[a + 10..b].to_string(), e[b + 7..c].to_string()); }
        }
    }
    (String::new(), String::new())
}

/// What `estimate` was asked of the playouts, as read from its keywords.
struct PlayoutSpec<'a> { rollout: &'a str, leaf: &'a str, eps: f64, scale: f64, temperature: f64, floor: f64, prior: bool, boost: f64, seed: u64, exploration: f64, max_playout: usize }

fn estimate_leaf<G: Game, R: cs_planning::mcts::Rollout + Clone, J: cs_tactics::ledger::Judge + Clone>(g: &G, s: &G::State, me: usize, p: &PlayoutSpec, rollout: R, judge: J,
                                                                                                       budget: cs_operator::Budget) -> Result<cs_operator::Outcome<G::Move>, String> {
    use cs_operator::Operator;
    let goal = cs_operator::Goal::Estimate { role: me };
    match p.leaf {
        "draw" => Ok(cs_planning::mcts::MctsOp::with_policy(p.exploration, p.max_playout, rollout, cs_planning::rollout::Draw).run(g, s, &goal, budget)),
        "balance" => Ok(cs_planning::mcts::MctsOp::with_policy(p.exploration, p.max_playout, rollout, cs_planning::rollout::Balance::new(judge, p.scale)).run(g, s, &goal, budget)),
        other => Err(format!("leaf: \"{other}\" is not draw or balance")),
    }
}

fn estimate_with<G: Game, J: cs_tactics::ledger::Judge + Clone>(g: &G, s: &G::State, me: usize, p: &PlayoutSpec, judge: J, lemmas: Option<cs_decision::LemmaPrior>,
                                                                budget: cs_operator::Budget) -> Result<cs_operator::Outcome<G::Move>, String> {
    use cs_planning::rollout::{GainPrior, Greedy, Weighted};
    match (p.rollout, p.prior, lemmas) {
        ("uniform", false, _) => estimate_leaf(g, s, me, p, cs_planning::mcts::Uniform::new(p.seed), judge, budget),
        ("greedy", false, _) => estimate_leaf(g, s, me, p, Greedy::new(judge.clone(), p.eps, p.seed), judge, budget),
        ("gain", false, _) => estimate_leaf(g, s, me, p, Weighted::new(GainPrior { judge: judge.clone(), temperature: p.temperature }, p.floor, p.seed), judge, budget),
        ("lemma", _, Some(lp)) | ("uniform", true, Some(lp)) => estimate_leaf(g, s, me, p, Weighted::new(lp, p.floor, p.seed), judge, budget),
        ("gain", true, Some(lp)) => estimate_leaf(g, s, me, p, Weighted::new(ProductPrior(GainPrior { judge: judge.clone(), temperature: p.temperature }, lp), p.floor, p.seed), judge, budget),
        ("greedy", true, _) => Err("prior: a lemma prior combines with rollout \"uniform\" or \"gain\" (or use rollout \"lemma\"), not \"greedy\"".into()),
        ("lemma", _, None) | (_, true, None) => Err("the lemma prior needs store and spec (the snapshot path and the variant's description text)".into()),
        (other, _, _) => Err(format!("rollout: \"{other}\" is not uniform, greedy, gain or lemma")),
    }
}

/// estimate(variant, fen, moves, seconds, max_nodes, seed, exploration, max_playout, rollout="uniform", leaf="draw", eps=0.1, scale=3.0, temperature=1.0, store=None,
/// prior=False, floor=0.05, spec=None, boost=4.0) -> {move, value, candidates, visits, playouts, cut, depth, nodes, seconds, rollout, leaf}: the
/// Monte Carlo tree search operator's ESTIMATE of the position for the side to move (sampled playouts: an estimate, never a proof). `candidates` are the legal moves in
/// visit order (most visited first), `visits` one row per root move (name, visits, mean outcome in hundredths of an outcome unit), `value` the root's mean for the
/// mover in the same units. One source of a ranking among others (the teacher is another); game-free behind the Operator interface.
/// PLAYOUTS: `rollout` is "uniform" (random moves), "greedy" (with probability `eps` random, else the move of best immediate gain), "gain" (sampled by exp(gain /
/// `temperature`) plus `floor`) or "lemma" (sampled by the snapshot's lemma prior: `boost` on the moves a bound lemma names, 1 on the rest, plus `floor`). `leaf` is what
/// a playout cut at `max_playout` scores: "draw" (50 for every role) or "balance" (the units' balance by the judge, squashed by `scale` worth units). The judge is the
/// `store` snapshot's judge record (its worths) or, without one, a count (every unit worth 1). `prior=True` (with `store` and `spec`, the variant's description text)
/// multiplies the weights of "uniform" or "gain" by the lemma prior's.
#[pyfunction]
#[pyo3(signature = (variant, fen=None, moves=vec![], seconds=0.2, max_nodes=10_000_000, seed=1, exploration=cs_planning::mcts::EXPLORATION, max_playout=cs_planning::mcts::MAX_PLAYOUT,
                    rollout="uniform", leaf="draw", eps=0.1, scale=3.0, temperature=1.0, store=None, prior=false, floor=0.05, spec=None, boost=4.0))]
#[allow(clippy::too_many_arguments)]
fn estimate(py: Python<'_>, variant: &str, fen: Option<&str>, moves: Vec<String>, seconds: f64, max_nodes: u64, seed: u64, exploration: f64, max_playout: usize, rollout: &str,
            leaf: &str, eps: f64, scale: f64, temperature: f64, store: Option<&str>, prior: bool, floor: f64, spec: Option<&str>, boost: f64) -> PyResult<PyObject> {
    let (g, s) = state_of(variant, fen, 518, &moves)?;
    let Some(me) = g.mover(&s) else { return Err(pyo3::exceptions::PyValueError::new_err("the game is over")) };
    let p = PlayoutSpec { rollout, leaf, eps, scale, temperature, floor, prior, boost, seed, exploration, max_playout };
    let snapshot = match store { Some(path) => Some(cs_memory::snapshot::Snapshot::load(std::path::Path::new(path)).map_err(pyo3::exceptions::PyValueError::new_err)?), None => None };
    let o = py.allow_threads(|| -> Result<_, String> {
        let budget = cs_operator::Budget::new(seconds, max_nodes);
        // (the fact view is built here: it is not sent between threads)
        let view = match (&snapshot, spec) { (Some(sn), Some(text)) if p.prior || p.rollout == "lemma" => Some(cs_game::gdl_game::FactView::new(text, sn.vocabulary.clone())?), _ => None };
        let lemmas = match (&snapshot, &view) { (Some(sn), Some(v)) => Some(cs_decision::LemmaPrior::new(sn, v, p.boost)), _ => None };
        match snapshot.as_ref().and_then(|sn| sn.judge.as_ref()) {
            Some(j) => estimate_with(&g, &s, me, &p, cs_tactics::ledger::Worth { name: j.name.clone(), table: j.worth.iter().map(|(k, v)| (k.clone(), *v)).collect(), ours_first: j.ours_first }, lemmas, budget),
            None => estimate_with(&g, &s, me, &p, cs_tactics::ledger::Count, lemmas, budget),
        }
    }).map_err(pyo3::exceptions::PyValueError::new_err)?;
    let legal = g.moves(&s);
    let name = |i: usize| g.move_name(&s, &legal[i]);
    let d = pyo3::types::PyDict::new(py);
    d.set_item("status", format!("{:?}", o.status))?;
    let (rollout_name, leaf_name) = playout_names(&o.evidence);
    d.set_item("rollout", rollout_name)?;
    d.set_item("leaf", leaf_name)?;
    d.set_item("move", o.mv.as_ref().map(|m| g.move_name(&s, m)))?;
    d.set_item("value", o.value)?;
    d.set_item("depth", o.depth)?;
    d.set_item("candidates", o.candidates.iter().map(|&i| name(i)).collect::<Vec<_>>())?;
    let mut visits = vec![];
    let (mut playouts, mut cut) = (0i64, 0i64);
    for f in &o.facts {
        match f.relation.as_str() {
            "visits" => if let (Ok(i), Ok(v), Ok(m)) = (f.args[0].parse::<usize>(), f.args[1].parse::<i64>(), f.args[2].parse::<i64>()) { visits.push((name(i), v, m)); },
            "playouts" => { playouts = f.args[0].parse().unwrap_or(0); cut = f.args[1].parse().unwrap_or(0); }
            _ => {}
        }
    }
    d.set_item("visits", visits)?;
    d.set_item("playouts", playouts)?;
    d.set_item("cut", cut)?;
    d.set_item("nodes", o.cost.nodes)?;
    d.set_item("seconds", o.cost.seconds)?;
    Ok(d.into())
}

/// A corpus of cards (positions with their facts and the exact value of every legal move) lemmas are verified on.
#[pyclass]
struct CardSet { inner: std::sync::Arc<cs_memory::cards::CardSet> }

fn lemma_of(text: &str) -> PyResult<cs_memory::lemma::Lemma> {
    let v: serde_json::Value = serde_json::from_str(text).map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("lemma: {e}")))?;
    cs_memory::lemma::Lemma::from_json(&v).map_err(pyo3::exceptions::PyValueError::new_err)
}

fn eval_dict(py: Python<'_>, ev: &cs_memory::cards::Eval) -> PyResult<PyObject> {
    let d = pyo3::types::PyDict::new(py);
    d.set_item("fires", ev.fires)?; d.set_item("holds", ev.holds)?; d.set_item("refuted", ev.refuted)?; d.set_item("unknown", ev.unknown)?;
    d.set_item("groups", ev.groups)?; d.set_item("groups_held", ev.groups_held)?; d.set_item("precision", ev.precision)?; d.set_item("expected", ev.expected)?;
    d.set_item("excess", ev.excess)?; d.set_item("selectivity", ev.selectivity)?; d.set_item("truncated", ev.truncated)?; d.set_item("stopped", ev.stopped)?; d.set_item("steps", ev.steps)?;
    d.set_item("cases", ev.cases.clone())?;
    Ok(d.into())
}

impl CardSet {
    fn check_indices(&self, indices: &Option<Vec<usize>>) -> PyResult<()> {
        if let Some(ix) = indices { if ix.iter().any(|i| *i >= self.inner.len()) { return Err(pyo3::exceptions::PyIndexError::new_err("a card index is out of range")); } }
        Ok(())
    }
}

#[pymethods]
impl CardSet {
    fn __len__(&self) -> usize { self.inner.len() }
    /// The card at `i` as JSON text.
    fn card(&self, i: usize) -> PyResult<String> {
        self.inner.cards.get(i).map(|c| c.to_json().to_string()).ok_or_else(|| pyo3::exceptions::PyIndexError::new_err("no such card"))
    }
    /// eval(lemma_json, indices=None, max_bindings=64, max_refuted=None) -> {fires, holds, refuted, unknown, groups, groups_held, precision, expected, excess, selectivity, truncated, stopped, steps, cases}; with `max_refuted` it stops at the first card past that many refutations (partial counts, `stopped`).
    #[pyo3(signature = (lemma, indices=None, max_bindings=64, max_refuted=None))]
    fn eval(&self, py: Python<'_>, lemma: &str, indices: Option<Vec<usize>>, max_bindings: usize, max_refuted: Option<usize>) -> PyResult<PyObject> {
        let l = lemma_of(lemma)?;
        self.check_indices(&indices)?;
        let set = self.inner.clone();
        let ev = py.allow_threads(|| set.eval_bounded(&l, indices.as_deref(), max_bindings, max_refuted));
        eval_dict(py, &ev)
    }
    /// eval_many([lemma_json], indices=None, max_bindings=64) -> [the eval dict of each]: the lemmas read side by side on the cores, in one call (the climb reads every
    /// candidate step at once). A lemma that is not well formed is an error for the whole call.
    #[pyo3(signature = (lemmas, indices=None, max_bindings=64, max_refuted=None))]
    fn eval_many(&self, py: Python<'_>, lemmas: Vec<String>, indices: Option<Vec<usize>>, max_bindings: usize, max_refuted: Option<usize>) -> PyResult<Vec<PyObject>> {
        let ls = lemmas.iter().map(|l| lemma_of(l)).collect::<PyResult<Vec<_>>>()?;
        self.check_indices(&indices)?;
        let set = self.inner.clone();
        let evs: Vec<cs_memory::cards::Eval> = py.allow_threads(|| {
            use rayon::prelude::*;
            ls.par_iter().map(|l| set.eval_bounded(l, indices.as_deref(), max_bindings, max_refuted)).collect()
        });
        evs.iter().map(|e| eval_dict(py, e)).collect()
    }
    /// Append the cards to a JSON Lines file.
    fn save(&self, path: &str) -> PyResult<()> {
        let rows: Vec<serde_json::Value> = self.inner.cards.iter().map(|c| c.to_json()).collect();
        cs_decision::batch::append(std::path::Path::new(path), &rows).map_err(|e| pyo3::exceptions::PyOSError::new_err(format!("{path}: {e}")))
    }
    #[staticmethod]
    fn load(path: &str) -> PyResult<CardSet> {
        let text = std::fs::read_to_string(path).map_err(|e| pyo3::exceptions::PyOSError::new_err(format!("{path}: {e}")))?;
        let cards = text.lines().filter(|l| !l.trim().is_empty()).map(|l| {
            let v: serde_json::Value = serde_json::from_str(l).map_err(|e| e.to_string())?;
            cs_memory::cards::Card::from_json(&v)
        }).collect::<Result<Vec<_>, String>>().map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(CardSet { inner: std::sync::Arc::new(cs_memory::cards::CardSet::new(cards)) })
    }
    /// The cards of several sets, in order, as one.
    #[staticmethod]
    fn merge(sets: Vec<PyRef<CardSet>>) -> CardSet {
        let cards = sets.iter().flat_map(|s| s.inner.cards.iter().cloned()).collect();
        CardSet { inner: std::sync::Arc::new(cs_memory::cards::CardSet::new(cards)) }
    }
}

/// build_cards(items, specs, vocabulary, depth=3, fuse=50_000_000, threads=0, worth={}, ours_first=False) -> CardSet: the cards of the positions [(variant, fen or None,
/// moves, id, group)] -- facts in the mover's perspective, every legal move's effect and exact value at `depth` (the plain minimax of the judge, no pruning), the nodes
/// of one card's reading capped by `fuse` (a cut card says so and is never used to refute). `specs` is {variant: description text}.
#[pyfunction]
#[pyo3(signature = (items, specs, vocabulary, depth=3, fuse=50_000_000, threads=0, worth=HashMap::new(), ours_first=false))]
#[allow(clippy::too_many_arguments)]
fn build_cards(py: Python<'_>, items: Vec<(String, Option<String>, Vec<String>, String, String)>, specs: HashMap<String, String>, vocabulary: &str, depth: i32, fuse: u64,
               threads: usize, worth: HashMap<String, i64>, ours_first: bool) -> PyResult<CardSet> {
    let vocab = vocab_of(vocabulary)?;
    let judged = !worth.is_empty();
    let w = cs_tactics::ledger::Worth { name: "cards".into(), table: worth, ours_first };
    let one = |it: &(String, Option<String>, Vec<String>, String, String)| -> Result<Option<cs_memory::cards::Card>, String> {
        let spec = specs.get(&it.0).ok_or_else(|| format!("no description for {}", it.0))?;
        let (g, s) = state_of(&it.0, it.1.as_deref(), 518, &it.2).map_err(|e| e.to_string())?;
        thread_local! { static V: std::cell::RefCell<HashMap<String, std::rc::Rc<cs_game::gdl_game::FactView>>> = std::cell::RefCell::new(HashMap::new()); }
        let key = format!("{}|{}|{:?}", it.0, cs_memory::sha::content_id(spec.as_bytes()), vocab);
        let view = V.with(|m| -> Result<_, String> {
            if let Some(x) = m.borrow().get(&key) { return Ok(x.clone()); }
            let x = std::rc::Rc::new(cs_game::gdl_game::FactView::new(spec, vocab.clone())?);
            m.borrow_mut().insert(key, x.clone());
            Ok(x)
        })?;
        if judged { cs_decision::cards::card_of(&g, &view, &w, &s, depth, fuse, &it.3, &it.4) } else { cs_decision::cards::card_of(&g, &view, &cs_tactics::ledger::Count, &s, depth, fuse, &it.3, &it.4) }
    };
    let results: Vec<Result<Option<cs_memory::cards::Card>, String>> = py.allow_threads(|| {
        use rayon::prelude::*;
        let run = || items.par_iter().map(|it| std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| one(it))).unwrap_or_else(|_| Err("a worker panicked".into()))).collect::<Vec<_>>();
        match rayon::ThreadPoolBuilder::new().num_threads(threads).build() { Ok(pool) => pool.install(run), Err(_) => run() }
    });
    let mut cards = vec![];
    for r in results { if let Some(c) = r.map_err(pyo3::exceptions::PyValueError::new_err)? { cards.push(c); } }
    Ok(CardSet { inner: std::sync::Arc::new(cs_memory::cards::CardSet::new(cards)) })
}

/// snapshot_info(path) -> {id, game, parent, lemmas, teacher_used}: a snapshot read the way the engine reads it (it must hash to its name).
#[pyfunction]
fn snapshot_info(py: Python<'_>, path: &str) -> PyResult<PyObject> {
    let s = cs_memory::snapshot::Snapshot::load(std::path::Path::new(path)).map_err(pyo3::exceptions::PyValueError::new_err)?;
    let d = pyo3::types::PyDict::new(py);
    d.set_item("id", s.id)?; d.set_item("game", s.game)?; d.set_item("parent", s.parent)?;
    d.set_item("lemmas", s.lemmas.iter().map(|l| l.name.clone()).collect::<Vec<_>>())?; d.set_item("teacher_used", s.teacher_used)?;
    Ok(d.into())
}

/// compose(base, delta) -> the description with the delta applied (retractions and additions), as the certificate reads a variant's text.
#[pyfunction]
fn compose(base: &str, delta: &str) -> PyResult<String> { cs_gdl::compose(base, delta).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.0)) }

/// content_id(bytes) -> the first 16 hex digits of the SHA-256: a snapshot's id.
#[pyfunction]
fn content_id(data: &[u8]) -> String { cs_memory::sha::content_id(data) }

/// check_lemma(lemma_json) -> None: raises ValueError if the lemma is not well formed (the engine's own reading of it).
#[pyfunction]
fn check_lemma(lemma: &str) -> PyResult<()> {
    let v: serde_json::Value = serde_json::from_str(lemma).map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("lemma: {e}")))?;
    cs_memory::lemma::Lemma::from_json(&v).map(|_| ()).map_err(pyo3::exceptions::PyValueError::new_err)
}

#[pymodule]
fn cs_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(variants, m)?)?;
    m.add_function(wrap_pyfunction!(certify_coplay, m)?)?;
    m.add_function(wrap_pyfunction!(best_move, m)?)?;
    m.add_function(wrap_pyfunction!(solve_batch, m)?)?;
    m.add_function(wrap_pyfunction!(facts, m)?)?;
    m.add_function(wrap_pyfunction!(build_cards, m)?)?;
    m.add_function(wrap_pyfunction!(estimate, m)?)?;
    m.add_function(wrap_pyfunction!(snapshot_info, m)?)?;
    m.add_function(wrap_pyfunction!(content_id, m)?)?;
    m.add_function(wrap_pyfunction!(compose, m)?)?;
    m.add_function(wrap_pyfunction!(check_lemma, m)?)?;
    m.add_class::<CardSet>()?;
    m.add_function(wrap_pyfunction!(state, m)?)?;
    m.add_function(wrap_pyfunction!(watch, m)?)?;
    m.add_function(wrap_pyfunction!(census, m)?)?;
    m.add_function(wrap_pyfunction!(mine_valuation, m)?)?;
    Ok(())
}
