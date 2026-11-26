//! PLAN 79.59 (THE LEMMA MINER OF LAYER 2; SYSTEM_SPEC 11, block "The lemma miner of layer 2"; LAYER 2, piece: rules; ledger rows 59, 58, 57, 49). THIS TEXT IS THE INTERFACE, written before the body.
//! A test file: it may name the games and the relations of the descriptions; the layer crates and the translator do not.
//!
//! PART A, EXPLANATIONS. The modules in generated_mined/<game>_explain.rs are the placed modules of the seven games made with pass (28) "explain": the answer of a question comes with every read of a fluent of the state it made, each tagged with the number
//! of the clause of the description and the place of the literal (in the order written) that made the read; the module holds the text of each clause and of each literal (`EXPL_CLAUSES`, `EXPL_LITS`). A recorded answer is then a small proof: the clauses used and what each literal read.
//!   `explain` (ignored; scripts/lemma_explain_print.py `explain`): TAKES the games of a file (the first game) or seeded walks (the other games) and, at every state, every unit of the side to act; asks the allowed moves of the unit (`placed_allowed`, side and place as the key) explained;
//!      RETURNS OBS lines: for each game, the reads of the answers grouped by (fluent, which places a read binds, clause, literal): how many reads, in how many of the questions, which kind of read (every unit, every unit of a side and kind, one place, a whole fluent); and the text of the clause and of the literal
//!      by the program. What the support of 79.58 holds (the line a player names, a read of every unit, a read of every unit of the goal kind, two whole fluents) is thereby traced to the clauses that make it. No clause or literal is named here: both come from the module's tables.
//!   TAKES CS_FACTS_GAMES (the games file, for the first game), CS_FACTS_N (games, default 30), CS_FACTS_MAX (plies, default 40), CS_FACTS_JOBS (threads).
//! WHO CALLS: scripts/lemma_explain_print.py. NOT on the engine's path.
mod mined_common;
mod translated_common;
mod variants_common;
use std::collections::{BTreeMap, BTreeSet};
use translated_common::*;
use variants_common::kernel;

include!("generated_mined/chess_explain.rs");
include!("generated_mined/atomic_explain.rs");
include!("generated_mined/3check_explain.rs");
include!("generated_mined/kingofthehill_explain.rs");
include!("generated_mined/antichess_explain.rs");
include!("generated_mined/chess960_explain.rs");
include!("generated_mined/horde_explain.rs");
impl_tr!(placed_chess_explain, explain);
impl_tr!(placed_atomic_explain, explain);
impl_tr!(placed_threecheck_explain, explain);
impl_tr!(placed_kingofthehill_explain, explain);
impl_tr!(placed_antichess_explain, explain);
impl_tr!(placed_chess960_explain, explain);
impl_tr!(placed_horde_explain, explain);

use cs_game::Game;
use mined_common::*;

/// One kind of read in the aggregate: the fluent, which of its places the read binds (a bit each, bound = 1), the clause and the literal that made it.
type ReadKey = (String, u32, u16, u16);

#[derive(Default, Clone, Debug)]
struct Group { reads: u64, questions: u64, chain: Vec<(u16, u16)>, leaf: (u16, u16) }

#[derive(Default, Clone, Debug)]
struct Explained {
    states: u64, questions: u64, reads: u64, raw: u64, rows: u64,
    groups: BTreeMap<ReadKey, Group>,
    /// the texts of the clauses and of their literals, from the module's tables
    clauses: Vec<String>, lits: Vec<Vec<String>>, funcs: Vec<(u16, String, Vec<u16>, bool)>,
}

impl Explained {
    fn add(&mut self, o: &Explained) {
        self.states += o.states; self.questions += o.questions; self.reads += o.reads; self.raw += o.raw; self.rows += o.rows;
        for (k, g) in &o.groups { let e = self.groups.entry(k.clone()).or_default(); e.reads += g.reads; e.questions += g.questions; if e.chain.is_empty() { e.chain = g.chain.clone(); e.leaf = g.leaf; } }
        if self.clauses.is_empty() { self.clauses = o.clauses.clone(); self.lits = o.lits.clone(); self.funcs = o.funcs.clone(); }
    }
}

/// At every state of every game, every unit of the side to act, asked for its allowed moves explained.
fn explain_game<T: Tr + 'static>(game: &str, games: &[Vec<State>]) -> Explained {
    let k = kernel(game);
    let mut t = T::make();
    let (clauses, lits, funcs) = t.explain_tables_t();
    let names = t.reads_fluents_t();
    let mut out = Explained { clauses: clauses.iter().map(|x| x.to_string()).collect(), lits: lits.iter().map(|l| l.iter().map(|x| x.to_string()).collect()).collect(), funcs: funcs.iter().map(|(c, m, o, w)| (*c, m.to_string(), o.clone(), *w)).collect(), ..Default::default() };
    for g in games {
        for s in g {
            let Some(mover) = k.mover(s) else { continue };
            let side = k.roles()[mover].clone();
            let facts = pairs(&k.state_facts(s));
            let ms = t.state_of_facts_t(&facts).expect("the module reads the state");
            out.states += 1;
            for f in facts.iter().filter(|(r, a)| r == "cell" && a[2] == side) {
                let key: Vec<u32> = [&side, &f.1[0], &f.1[1]].iter().map(|x| t.sym_t(x).expect("a symbol of the description")).collect();
                let Some((rows, reads)) = t.explain_t(&ms, "placed_allowed", &key) else { continue };
                out.questions += 1;
                out.rows += rows.len() as u64;
                // the distinct reads of the question: the same pattern read again by the same clause and literal is one read of the state
                let mut seen: BTreeSet<(ReadKey, Vec<u32>)> = BTreeSet::new();
                let mut kinds: BTreeSet<ReadKey> = BTreeSet::new();
                for (fl, pat, fi, st, ch) in &reads {
                    let mask: u32 = pat.iter().enumerate().map(|(j, p)| if *p != FREE { 1u32 << j } else { 0 }).sum();
                    let (cl, order, _, _) = (out.funcs[*fi as usize].0, &out.funcs[*fi as usize].2, 0, 0);
                    let lt = order[*st as usize];
                    let key = (names[*fl as usize].to_string(), mask, cl, lt);
                    out.raw += 1;
                    if seen.insert((key.clone(), pat.clone())) {
                        out.reads += 1;
                        let e = out.groups.entry(key.clone()).or_default();
                        e.reads += 1;
                        if e.chain.is_empty() {
                            // the first chain of calls this kind of read was made under, each frame as (clause, literal)
                            let chain: Vec<(u16, u16)> = t.explain_chain_t(*ch);
                            let g = out.groups.get_mut(&key).unwrap();
                            g.chain = chain;
                            g.leaf = (*fi, *st);
                        }
                    }
                    kinds.insert(key);
                }
                for key in kinds { out.groups.get_mut(&key).unwrap().questions += 1; }
            }
        }
    }
    out
}

fn threaded<T: Tr + 'static>(game: &str, games: &[Vec<State>], jobs: usize) -> Explained {
    let chunk = (games.len() + jobs - 1) / jobs.max(1);
    std::thread::scope(|sc| {
        let hs: Vec<_> = games.chunks(chunk.max(1)).map(|g| sc.spawn(move || explain_game::<T>(game, g))).collect();
        let mut t = Explained::default();
        for h in hs { t.add(&h.join().unwrap()); }
        t
    })
}

/// What kind of read it is, in words, from the fluent's number of places and the bits of the places it binds. The fluent's own places are not named here (the module's tables hold only its name and its arity through the pattern).
fn mask_text(mask: u32, arity: usize) -> String { (0..arity).map(|j| if mask >> j & 1 == 1 { "b" } else { "*" }).collect::<Vec<_>>().join("") }

/// How wide a read makes a support when the adapter turns it into places: the fluent that holds the units is read at places (a read that binds the file and the rank is one place, one of them a line of eight, neither every place); any other fluent is a whole fluent.
fn width_text(fluent: &str, mask: u32, cell_fluent: &str) -> &'static str {
    if fluent != cell_fluent { return "a whole fluent (not the one that holds the units)"; }
    match (mask & 1 == 1, mask & 2 == 2) { (true, true) => "one place", (false, false) => "EVERY place", _ => "a line of places" }
}

fn print_game(game: &str, e: &Explained, arity: &dyn Fn(&str) -> usize, cell_fluent: &str) {
    println!("OBS 79.59 A {game}: {} states, {} questions (the allowed moves of one unit), {} rows, {} distinct reads recorded ({:.1} a question; {} counting every repeat, {:.1} a question)", e.states, e.questions, e.rows, e.reads, e.reads as f64 / e.questions.max(1) as f64, e.raw, e.raw as f64 / e.questions.max(1) as f64);
    let mut v: Vec<(&ReadKey, &Group)> = e.groups.iter().collect();
    v.sort_by(|a, b| b.1.reads.cmp(&a.1.reads).then(a.0.cmp(b.0)));
    let mut told: BTreeSet<(u16, u16)> = BTreeSet::new();
    for (k, g) in &v {
        let (fl, mask, cl, lt) = k;
        let n = arity(fl);
        let places = mask_text(*mask, n);
        let wide = width_text(fl, *mask, cell_fluent);
        println!("OBS 79.59 A {game}: read of {fl}[{places}] by clause {cl} literal {lt}: {:.2} distinct reads a question, in {:.3} of the questions; support width: {wide}", g.reads as f64 / e.questions.max(1) as f64, g.questions as f64 / e.questions.max(1) as f64);
        if told.insert((*cl, *lt)) {
            println!("EXPL {game}: clause {cl}: {}", e.clauses[*cl as usize]);
            println!("EXPL {game}: literal {lt} of clause {cl}: {}", e.lits[*cl as usize][*lt as usize]);
        }
    }
    println!("OBS 79.59 A {game}: THE READS THAT MAKE A SUPPORT WIDER THAN A LINE OF PLACES (every place of the board, or a whole fluent), each with the clause and the literal of the description that makes it:");
    for (k, g) in &v {
        let (fl, mask, cl, lt) = k;
        let wide = width_text(fl, *mask, cell_fluent);
        if wide.starts_with("EVERY") || wide.starts_with("a whole") {
            println!("OBS 79.59 A {game}: WIDE {fl}[{}] ({wide}) in {:.3} of the questions, made by clause {cl} literal {lt}", mask_text(*mask, arity(fl)), g.questions as f64 / e.questions.max(1) as f64);
            println!("EXPL {game}: WIDE clause {cl}: {}  LITERAL {lt}: {}", e.clauses[*cl as usize], e.lits[*cl as usize][*lt as usize]);
            // the frames, outermost first, each as the function-clause that made the call and the position of the literal in its join: the text of the clause, the head places bound, the order of the join, the position (read by scripts/lemma_miner_print.py `explain`)
            let frame = |fi: u16, st: u16| -> String { let (c, m, o, w) = &e.funcs[fi as usize]; format!("{} ||| {} ||| {} ||| {} ||| {}", e.clauses[*c as usize], m, o.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(","), st, w) };
            for (f2, s2) in &g.chain { let (c2, _m, o2, _w) = &e.funcs[*f2 as usize]; let l2 = o2[*s2 as usize]; println!("EXPL {game}:   called by clause {c2} literal {l2}: {}  /  {}", e.clauses[*c2 as usize], e.lits[*c2 as usize][l2 as usize]); println!("EXPLF {game} CALL {}", frame(*f2, *s2)); }
            println!("EXPLF {game} READ {}", frame(g.leaf.0, g.leaf.1));
        }
    }
}

#[test]
#[ignore = "a measure, by hand (plan 79.59 part A): CS_FACTS_GAMES names the games file; scripts/lemma_explain_print.py explain"]
fn explain() {
    let path = std::env::var("CS_FACTS_GAMES").expect("CS_FACTS_GAMES names the file of games");
    let (n, max) = (env_usize("CS_FACTS_N", 30), env_usize("CS_FACTS_MAX", 40));
    let jobs = env_usize("CS_FACTS_JOBS", std::thread::available_parallelism().map(|x| x.get()).unwrap_or(4));
    let t0 = std::time::Instant::now();
    let chess = states_of_lines("chess", &lines_of(&path, n), max);
    // the number of places of a fluent as the module's reads say it: the three common fluents are asked by their arity from the first module's table
    let arity_of = |names: &[&'static str], ar: &[usize]| { let m: BTreeMap<String, usize> = names.iter().map(|x| x.to_string()).zip(ar.iter().copied()).collect(); move |f: &str| m[f] };
    macro_rules! game { ($m:ident, $name:expr, $states:expr) => {{
        let e = threaded::<$m::Game>($name, $states, jobs);
        let a = arity_of(&$m::READS_FLUENTS, &$m::READS_ARITY);
        print_game($name, &e, &a, "cell");
    }}}
    game!(placed_chess_explain, "chess", &chess);
    for (name, f) in [("atomic", 0), ("3check", 1), ("kingofthehill", 2), ("antichess", 3), ("chess960", 4), ("horde", 5)] {
        let k = kernel(name);
        let g = seeded_states(&k, n, max);
        match f {
            0 => game!(placed_atomic_explain, "atomic", &g),
            1 => game!(placed_threecheck_explain, "3check", &g),
            2 => game!(placed_kingofthehill_explain, "kingofthehill", &g),
            3 => game!(placed_antichess_explain, "antichess", &g),
            4 => game!(placed_chess960_explain, "chess960", &g),
            _ => game!(placed_horde_explain, "horde", &g),
        }
    }
    println!("OBS 79.59 A all seven games: {:.1} s, {} threads", t0.elapsed().as_secs_f64(), jobs);
}

/// PART F (a count only, no build): the rows of EVERY relation a description defines, at the states of the segments: for each relation the mean and the largest number of rows a state. `COUNT <game> <relation> <mean rows> <largest> <states>` lines; the shapes of the relations (which have a unit and a place, two places, what the rest are) are read from the signatures by scripts/lemma_miner_print.py `count`.
#[derive(Default)]
struct Counted { states: u64, sum: BTreeMap<String, u64>, max: BTreeMap<String, u64> }

fn count_game<T: Tr + 'static>(game: &str, segs: &[Vec<State>]) -> Counted {
    let k = kernel(game);
    let mut t = T::make();
    let mut out = Counted::default();
    for g in segs {
        for s in g {
            let ms = t.state_of_facts_t(&pairs(&k.state_facts(s))).expect("the module reads the state");
            out.states += 1;
            for (name, rows) in t.all_t(&ms, &[]) {
                *out.sum.entry(name.to_string()).or_insert(0) += rows.len() as u64;
                let e = out.max.entry(name.to_string()).or_insert(0);
                *e = (*e).max(rows.len() as u64);
            }
        }
    }
    out
}

#[test]
#[ignore = "a count, by hand (plan 79.59 part F): CS_FACTS_GAMES names the games file; scripts/lemma_miner_print.py count"]
fn relations_count() {
    let path = std::env::var("CS_FACTS_GAMES").expect("CS_FACTS_GAMES names the file of games");
    let (n, max) = (env_usize("CS_FACTS_N", 30), env_usize("CS_FACTS_MAX", 40));
    macro_rules! one { ($m:ident, $name:expr, $segs:expr) => {{
        let c = count_game::<$m::Game>($name, $segs);
        for (r, s) in &c.sum { println!("COUNT {} {} {:.3} {} {}", $name, r, *s as f64 / c.states.max(1) as f64, c.max[r], c.states); }
    }}}
    one!(placed_chess_explain, "chess", &states_of_lines("chess", &lines_of(&path, n), max));
    one!(placed_atomic_explain, "atomic", &seeded_states(&kernel("atomic"), n, max));
    one!(placed_threecheck_explain, "3check", &seeded_states(&kernel("3check"), n, max));
    one!(placed_kingofthehill_explain, "kingofthehill", &seeded_states(&kernel("kingofthehill"), n, max));
    one!(placed_antichess_explain, "antichess", &seeded_states(&kernel("antichess"), n, max));
    one!(placed_chess960_explain, "chess960", &seeded_states(&kernel("chess960"), n, max));
    one!(placed_horde_explain, "horde", &seeded_states(&kernel("horde"), n, max));
}
