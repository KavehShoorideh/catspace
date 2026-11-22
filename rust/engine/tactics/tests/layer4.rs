//! PLAN 79.25 PART B: THE TEST OF BEING DONE OF LAYER 4, MEANS-ENDS (SYSTEM_SPEC 11 "a layer's test uses only that layer and what is below it"; ledger row 36; piece: tactical reasoner).
//! It uses layer 1 (cs_rep), layer 2 (cs_game), layer 3 through the judge (cs_value's count is what the ledger judge changes by, see layer3_one_copy.rs) and this layer's one entry (cs_tactics::meansends::search), no crate above, and no outside program at test time.
//! THE REFERENCES ARE DATA, read as opaque text: tests/fixtures/puzzle.fixture (positions that are not quiet by a stated rule, each with the solution and the outside engine's stored verdict) and tests/fixtures/quiet.fixture (positions where nothing is forced), made once by
//! catspace/incubator/analyst/discovery/layer4_fixtures_print.py. This file names no object of any game: the game's name, its declaration, its positions and the names of its moves come from the files. EVERY RUN IS BOUNDED BY NODES, NEVER BY A CLOCK: the same answers on any machine.
//! THE SIX THINGS, each printing OBS lines: (1) the positions are not quiet, by the rule the fixture states, and are read; (2) THE SOLUTION IS SEEN at three budgets of nodes (the first move, the whole line) by theme and by the solution's length, kept as a RATCHET (never lower than the
//! floors below, which are what the search saw when this test was written, not a claim about its strength); (3) NO FALSE PROOF against the stored verdict, with every disagreement printed, and forced ends of n moves found at the budget that should reach them; (4) QUIET POSITIONS ARE NOT THIS LAYER'S:
//! the share where it wrongly claims a forced gain; (5) the SAME ANSWER TWICE (and from one thread or many); (6) the RATE, nodes a second, three runs, medians.
//! A WEAKNESS (a theme not seen, a false proof) is printed with its positions and is NOT mended here.
use cs_game::kernel::KernelGame;
use cs_game::rules::{Ruled, StateText};
use cs_game::Game;
use cs_rep::history::Seen;
use cs_tactics::ledger::Count;
use cs_tactics::meansends::{search, Ask, Ending, Found};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

type R = Ruled<KernelGame>;

/// The three budgets, in nodes.
const BUDGETS: [u64; 3] = [5_000, 50_000, 500_000];
const DEPTH: i32 = 64;

/// THE RATCHET: the counts of the first move seen and of the whole line seen at each budget, over every position, by the solution's length (1, 2, 3, 4 or more) and by theme (the themes with at least THEME_MIN positions). Never lower than these.
const FLOOR_ALL: ([usize; 3], [usize; 3]) = ([316, 338, 365], [169, 220, 236]);
const FLOOR_LENGTH: [([usize; 3], [usize; 3]); 4] = [([47, 48, 47], [47, 48, 47]), ([128, 133, 147], [73, 88, 94]), ([94, 105, 116], [41, 60, 65]), ([47, 52, 55], [8, 24, 30])];
/// The most false proofs and the most quiet positions wrongly claimed at each budget that the test lets pass (what the search did when this test was written: none).
const FALSE_PROOFS_MOST: [usize; 3] = [0; 3];
const QUIET_WRONG_MOST: [usize; 3] = [0; 3];
/// The forced ends of at most SHORT_END moves proved at each budget (of the positions the fixture holds): the ratchet of "the budget that should reach them".
const SHORT_PROVED_FLOOR: [usize; 3] = [124, 134, 134];
/// The forced ends of at most this many moves of ours are the ones "that the budget should reach".
const SHORT_END: u32 = 3;
const THEME_MIN: usize = 25;

// ------------------------------------------------------------------------------------------------------------------------------ the fixtures, read as opaque text

struct Puzzle { id: String, themes: Vec<String>, ours: usize, reference: String, margin: i64, text: String, solution: Vec<String> }
struct QuietPos { id: String, cp: i64, margin: i64, text: String }
struct Fx { name: String, decl: String, rules: Vec<String>, bytes: usize }

fn fixture(name: &str) -> (String, usize) {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests").join("fixtures").join(name);
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let n = text.len();
    (text, n)
}

fn puzzles() -> (Fx, Vec<Puzzle>) {
    let (text, bytes) = fixture("puzzle.fixture");
    let mut fx = Fx { name: String::new(), decl: String::new(), rules: vec![], bytes };
    let mut v = vec![];
    for line in text.lines() {
        let f: Vec<&str> = line.split('\t').collect();
        match f[0] {
            "game" => { fx.name = f[1].into(); fx.decl = f[2].into(); }
            "rule" => fx.rules.push(f[1].into()),
            "puzzle" => v.push(Puzzle { id: f[1].into(), themes: f[2].split(' ').map(|t| t.to_string()).collect(), ours: f[3].parse().unwrap(), reference: f[4].into(), margin: f[5].parse().unwrap(), text: f[6].into(),
                                         solution: f[7].split(' ').map(|m| m.to_string()).collect() }),
            other => panic!("unknown tag {other}"),
        }
    }
    (fx, v)
}

fn quiets() -> (Fx, Vec<QuietPos>) {
    let (text, bytes) = fixture("quiet.fixture");
    let mut fx = Fx { name: String::new(), decl: String::new(), rules: vec![], bytes };
    let mut v = vec![];
    for line in text.lines() {
        let f: Vec<&str> = line.split('\t').collect();
        match f[0] {
            "game" => { fx.name = f[1].into(); fx.decl = f[2].into(); }
            "rule" => fx.rules.push(f[1].into()),
            "quiet" => v.push(QuietPos { id: f[1].into(), cp: f[2].parse().unwrap(), margin: f[3].parse().unwrap(), text: f[4].into() }),
            other => panic!("unknown tag {other}"),
        }
    }
    (fx, v)
}

fn rules_root() -> PathBuf { PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../rules") }

/// The game of a fixture, opened from its own declaration (layer 2's door).
fn open(fx: &Fx) -> R { Ruled::open(KernelGame::new(&fx.name, 518).unwrap(), &fx.name, &rules_root(), &fx.decl).unwrap() }

/// What the search is asked at one position, from the side to move.
fn ask(g: &R, text: &str, nodes: u64, history: Arc<Seen>) -> Found {
    let s = g.read_state_text(text).unwrap();
    let me = g.mover(&s).expect("a position the game goes on from");
    search(&Ask { game: g, state: &s, judge: &Count, side: me, nodes, seconds: None, max_depth: DEPTH, history })
}

/// Every index of 0..n asked of `f` on all the cores, each thread with its own copy of the game; the answers in index order (they do not depend on the number of threads).
fn par<T: Send>(fx: &Fx, n: usize, f: &(dyn Fn(&R, usize) -> T + Sync)) -> Vec<T> {
    let threads = std::thread::available_parallelism().map(|x| x.get()).unwrap_or(1).min(16);
    let next = AtomicUsize::new(0);
    let out: Mutex<Vec<(usize, T)>> = Mutex::new(vec![]);
    std::thread::scope(|sc| {
        for _ in 0..threads {
            sc.spawn(|| {
                let g = open(fx);
                loop {
                    let i = next.fetch_add(1, Ordering::SeqCst);
                    if i >= n { break; }
                    let r = f(&g, i);
                    out.lock().unwrap().push((i, r));
                }
            });
        }
    });
    let mut v = out.into_inner().unwrap();
    v.sort_by_key(|x| x.0);
    v.into_iter().map(|x| x.1).collect()
}

fn none() -> Arc<Seen> { Arc::new(Seen::new()) }

/// The search's answers at every budget over every puzzle, once (shared by the tests that read them).
fn answers() -> &'static (Fx, Vec<Puzzle>, Vec<Vec<Found>>) {
    static CELL: std::sync::OnceLock<(Fx, Vec<Puzzle>, Vec<Vec<Found>>)> = std::sync::OnceLock::new();
    CELL.get_or_init(|| {
        let (fx, ps) = puzzles();
        let t0 = Instant::now();
        let all: Vec<Vec<Found>> = BUDGETS.iter().map(|&b| par(&fx, ps.len(), &|g, i| ask(g, &ps[i].text, b, none()))).collect();
        println!("OBS 79.25 answers: {} positions at budgets {:?} nodes read on all the cores in {:.1} s", ps.len(), BUDGETS, t0.elapsed().as_secs_f64());
        (fx, ps, all)
    })
}

/// The ratchet per theme, from the fixture ratchet.fixture (theme -> (first move counts, whole line counts) at the three budgets).
fn theme_floors() -> BTreeMap<String, ([usize; 3], [usize; 3])> {
    let (text, _) = fixture("ratchet.fixture");
    let three = |s: &str| -> [usize; 3] { let v: Vec<usize> = s.split(',').map(|x| x.parse().unwrap()).collect(); [v[0], v[1], v[2]] };
    text.lines().filter_map(|l| { let f: Vec<&str> = l.split('\t').collect(); if f[0] == "floor" { Some((f[1].to_string(), (three(f[2]), three(f[3])))) } else { None } }).collect()
}

fn ratio(a: usize, b: usize) -> String { format!("{a}/{b}") }

// ------------------------------------------------------------------------------------------------------------------------------ (1) the positions are not quiet, by the stated rule

#[test]
fn the_positions_are_the_fixtures_and_every_solution_move_is_legal_in_its_place() {
    let (fx, ps) = puzzles();
    let g = open(&fx);
    assert!(ps.len() >= 300, "a few hundred positions at least: {}", ps.len());
    assert!(fx.rules.iter().any(|r| r.starts_with("R1")) && fx.rules.iter().any(|r| r.starts_with("R2")), "the rule of 'not quiet' is stated in the fixture");
    println!("OBS 79.25 fixture puzzle.fixture: {} bytes, {} positions; the rule: {}", fx.bytes, ps.len(), fx.rules.join(" | "));
    let mut unmatched = vec![];
    let (mut by_len, mut by_theme): (BTreeMap<usize, usize>, BTreeMap<String, usize>) = (BTreeMap::new(), BTreeMap::new());
    for p in &ps {
        let mut s = g.read_state_text(&p.text).unwrap();
        assert_eq!(p.ours, (p.solution.len() + 1) / 2, "{}", p.id);
        assert!(p.margin >= 150, "{}: the stored margin of the best over the second is the rule's", p.id);
        for (k, name) in p.solution.iter().enumerate() {
            let found = g.moves(&s).into_iter().find(|m| &g.move_name(&s, m) == name);
            match found {
                Some(m) => { s = g.apply(&s, &m); }
                None => { unmatched.push((p.id.clone(), k, name.clone())); break; }
            }
        }
        *by_len.entry(p.ours.min(4)).or_default() += 1;
        for t in &p.themes { *by_theme.entry(t.clone()).or_default() += 1; }
    }
    println!("OBS 79.25 fixture: solutions whose moves the game does not name the same way (unmatched): {} of {} {:?}", unmatched.len(), ps.len(), unmatched);
    assert!(unmatched.len() * 50 <= ps.len(), "at most one in fifty solutions may name a move the game names differently");
    println!("OBS 79.25 fixture: positions by our moves in the solution (4 is four or more): {:?}", by_len);
    println!("OBS 79.25 fixture: positions by theme (the themes with at least {THEME_MIN}): {:?}", by_theme.iter().filter(|(_, n)| **n >= THEME_MIN).collect::<Vec<_>>());
    let forced = ps.iter().filter(|p| p.reference.starts_with('m') && !p.reference.starts_with("m-")).count();
    println!("OBS 79.25 fixture: reference verdicts: a forced end for the side to move {forced}, a score {}", ps.len() - forced);
    assert!(forced >= 50, "forced ends are in the fixture");
}

// ------------------------------------------------------------------------------------------------------------------------------ (2) the solution is seen

fn seen_counts(ps: &[Puzzle], all: &[Vec<Found>], pick: &dyn Fn(&Puzzle) -> bool) -> (usize, [usize; 3], [usize; 3]) {
    let mut n = 0;
    let (mut first, mut whole) = ([0; 3], [0; 3]);
    for (i, p) in ps.iter().enumerate() {
        if !pick(p) { continue; }
        n += 1;
        for b in 0..3 {
            let f = &all[b][i];
            if f.first_name.as_deref() == Some(p.solution[0].as_str()) { first[b] += 1; }
            if f.line.len() >= p.solution.len() && f.line[..p.solution.len()] == p.solution[..] { whole[b] += 1; }
        }
    }
    (n, first, whole)
}

fn say(label: &str, n: usize, first: [usize; 3], whole: [usize; 3]) {
    println!("OBS 79.25 seen {label}: positions {n}; first move equals the solution's at {:?} nodes: {} {} {}; the whole line at the same budgets: {} {} {}",
             BUDGETS, ratio(first[0], n), ratio(first[1], n), ratio(first[2], n), ratio(whole[0], n), ratio(whole[1], n), ratio(whole[2], n));
}

#[test]
fn the_solution_is_seen_at_three_budgets_by_theme_and_length_and_never_below_the_ratchet() {
    let (fx, ps, all) = answers();
    let _ = fx;
    let (n, first, whole) = seen_counts(ps, all, &|_| true);
    say("all", n, first, whole);
    for b in 0..3 {
        assert!(first[b] >= FLOOR_ALL.0[b] && whole[b] >= FLOOR_ALL.1[b], "the ratchet for all positions at budget {}: {:?} {:?}", BUDGETS[b], first, whole);
    }
    for (k, floor) in FLOOR_LENGTH.iter().enumerate() {
        let (n, f, w) = seen_counts(ps, all, &|p| p.ours.min(4) == k + 1);
        say(&format!("by length {}{}", k + 1, if k == 3 { " or more" } else { "" }), n, f, w);
        for b in 0..3 { assert!(f[b] >= floor.0[b] && w[b] >= floor.1[b], "the ratchet for length {} at budget {}", k + 1, BUDGETS[b]); }
    }
    let mut themes: BTreeMap<String, usize> = BTreeMap::new();
    for p in ps { for t in &p.themes { *themes.entry(t.clone()).or_default() += 1; } }
    let floors = theme_floors();
    let mut seen_floor = 0;
    for (t, c) in &themes {
        if *c < THEME_MIN { continue; }
        let (n, f, w) = seen_counts(ps, all, &|p| p.themes.contains(t));
        say(&format!("by theme {t}"), n, f, w);
        if let Some((ff, fw)) = floors.get(t.as_str()) {
            seen_floor += 1;
            for b in 0..3 { assert!(f[b] >= ff[b] && w[b] >= fw[b], "the ratchet for theme {t} at budget {}: {:?} {:?} against {:?} {:?}", BUDGETS[b], f, w, ff, fw); }
        }
    }
    assert!(seen_floor == floors.len(), "every theme with a floor is among those printed");
    // the weaknesses, printed with their positions: the themes whose first move is seen in under two thirds of their positions at the top budget
    for (t, c) in &themes {
        if *c < THEME_MIN { continue; }
        let (n, f, _) = seen_counts(ps, all, &|p| p.themes.contains(t));
        if f[2] * 3 < n * 2 {
            let missed: Vec<String> = ps.iter().enumerate().filter(|(i, p)| p.themes.contains(t) && all[2][*i].first_name.as_deref() != Some(p.solution[0].as_str())).take(3)
                .map(|(i, p)| format!("{} [{}] solution {} found {:?} value {} depth {} nodes {}", p.id, p.text, p.solution.join(" "), all[2][i].first_name, all[2][i].value, all[2][i].depth, all[2][i].nodes)).collect();
            println!("OBS 79.25 WEAKNESS theme {t}: the first move is seen in {} of {} at the top budget; examples: {}", f[2], n, missed.join(" || "));
        }
    }
    // a solution of one move not seen at the top budget, with its position
    for (i, p) in ps.iter().enumerate().filter(|(i, p)| p.ours == 1 && all[2][*i].first_name.as_deref() != Some(p.solution[0].as_str())).take(4) {
        println!("OBS 79.25 WEAKNESS a one-move solution not seen at {} nodes: {} [{}] solution {} found {:?} value {} depth {} ending {:?}", BUDGETS[2], p.id, p.text, p.solution.join(" "), all[2][i].first_name, all[2][i].value, all[2][i].depth, all[2][i].ending);
    }
}

// ------------------------------------------------------------------------------------------------------------------------------ (3) no false proof

fn verdict_agrees(f: &Found, reference: &str) -> bool {
    match f.ending {
        Ending::Estimated => true,
        Ending::Won => reference.starts_with('m') && !reference.starts_with("m-"),
        Ending::Lost => reference.starts_with("m-"),
    }
}

#[test]
fn no_end_is_proved_that_the_stored_reference_does_not_read_and_short_forced_ends_are_found() {
    let (_, ps, all) = answers();
    let mut false_proofs = [0usize; 3];
    let mut proofs = [0usize; 3];
    let mut shown = 0;
    for b in 0..3 {
        for (i, p) in ps.iter().enumerate() {
            let f = &all[b][i];
            if f.ending != Ending::Estimated { proofs[b] += 1; }
            if !verdict_agrees(f, &p.reference) {
                false_proofs[b] += 1;
                if shown < 12 { shown += 1; println!("OBS 79.25 FALSE PROOF at {} nodes: {} [{}] the search says {:?} by {} (depth {}), the reference reads {}; solution {}", BUDGETS[b], p.id, p.text, f.ending, f.line.join(" "), f.depth, p.reference, p.solution.join(" ")); }
            }
        }
    }
    println!("OBS 79.25 proofs: ends proved at {:?} nodes: {:?}; against the stored verdict, disagreements: {:?}", BUDGETS, proofs, false_proofs);
    for b in 0..3 { assert!(false_proofs[b] <= FALSE_PROOFS_MOST[b], "a new false proof at budget {}: {:?}", BUDGETS[b], false_proofs); }
    // the forced ends of at most SHORT_END moves of ours: the share the search proves, per budget (the budget that should reach them)
    let short: Vec<usize> = (0..ps.len()).filter(|&i| { let r = &ps[i].reference; r.starts_with('m') && !r.starts_with("m-") && r[1..].parse::<u32>().map(|n| n <= SHORT_END).unwrap_or(false) }).collect();
    let mut proved = [0usize; 3];
    let mut longer = [0usize; 3];
    for b in 0..3 {
        for &i in &short {
            let f = &all[b][i];
            if f.ending == Ending::Won {
                proved[b] += 1;
                let n: usize = ps[i].reference[1..].parse().unwrap();
                if f.line.len() > 2 * n - 1 { longer[b] += 1; }
            }
        }
    }
    println!("OBS 79.25 proofs: forced ends of at most {SHORT_END} moves of ours ({} positions) proved at {:?} nodes: {} {} {}; of those proved, a line longer than the reference's: {:?}", short.len(), BUDGETS, ratio(proved[0], short.len()), ratio(proved[1], short.len()), ratio(proved[2], short.len()), longer);
    assert!(short.len() >= 20);
    for b in 0..3 { assert!(proved[b] >= SHORT_PROVED_FLOOR[b], "forced ends of at most {SHORT_END} moves proved at budget {}: {:?} (floor {:?})", BUDGETS[b], proved, SHORT_PROVED_FLOOR); }
    for (i, &p) in short.iter().enumerate() {
        if i >= 6 || all[2][p].ending == Ending::Won { continue; }
        println!("OBS 79.25 WEAKNESS a forced end of {} moves not proved at {} nodes: {} [{}] found {:?} value {} depth {}", ps[p].reference, BUDGETS[2], ps[p].id, ps[p].text, all[2][p].first_name, all[2][p].value, all[2][p].depth);
    }
}

// ------------------------------------------------------------------------------------------------------------------------------ (4) quiet positions are not this layer's

#[test]
fn in_quiet_positions_the_layer_is_not_asked_for_a_forced_gain_and_the_share_that_it_wrongly_claims_one_is_printed() {
    let (fx, qs) = quiets();
    println!("OBS 79.25 fixture quiet.fixture: {} bytes, {} positions; the rule: {}", fx.bytes, qs.len(), fx.rules.join(" | "));
    assert!(qs.len() >= 40 && qs.iter().all(|q| q.cp.abs() <= 100 && q.margin < 60), "the stored reference of every quiet position says nothing is forced");
    let mut wrong = [0usize; 3];
    let mut shown = 0;
    for (b, &budget) in BUDGETS.iter().enumerate() {
        let got = par(&fx, qs.len(), &|g, i| ask(g, &qs[i].text, budget, none()));
        for (i, f) in got.iter().enumerate() {
            // a forced gain claimed: the game's own end proved, or a unit won with none of ours lost (the count's value above nought)
            if f.ending != Ending::Estimated || f.value > 0 {
                wrong[b] += 1;
                if shown < 6 { shown += 1; println!("OBS 79.25 WEAKNESS a forced gain claimed in a quiet position at {} nodes: {} [{}] {:?} value {} line {} (reference score {} cp)", budget, qs[i].id, qs[i].text, f.ending, f.value, f.line.join(" "), qs[i].cp); }
            }
        }
    }
    println!("OBS 79.25 quiet: of {} quiet positions the layer wrongly claims a forced gain at {:?} nodes: {} {} {}", qs.len(), BUDGETS, ratio(wrong[0], qs.len()), ratio(wrong[1], qs.len()), ratio(wrong[2], qs.len()));
    for b in 0..3 { assert!(wrong[b] <= QUIET_WRONG_MOST[b], "more quiet positions wrongly claimed at budget {}: {:?}", BUDGETS[b], wrong); }
}

// ------------------------------------------------------------------------------------------------------------------------------ (5) the same answer twice

#[test]
fn the_same_position_budget_and_history_give_the_same_move_line_and_node_count_on_one_thread_or_many() {
    let (fx, ps, all) = answers();
    let g = open(fx);
    let step = ps.len() / 60;
    let mut compared = 0;
    for i in (0..ps.len()).step_by(step.max(1)) {
        let again = ask(&g, &ps[i].text, BUDGETS[1], none());
        assert_eq!(again, all[1][i], "{}: one thread, the other run's answer from many threads", ps[i].id);
        let (a, b) = (ask(&g, &ps[i].text, BUDGETS[0], none()), ask(&g, &ps[i].text, BUDGETS[0], none()));
        assert_eq!(a, b, "{}: twice in a row", ps[i].id);
        compared += 1;
    }
    // with a history: the keys of the positions of the line the game went by (the record), twice
    let s = g.read_state_text(&ps[0].text).unwrap();
    let keys = vec![g.key(&s), 7, 11];
    let (rec_a, rec_b) = (Arc::new(keys.iter().copied().collect::<Seen>()), Arc::new(keys.iter().copied().collect::<Seen>()));
    assert_eq!(ask(&g, &ps[0].text, BUDGETS[1], rec_a), ask(&g, &ps[0].text, BUDGETS[1], rec_b));
    println!("OBS 79.25 same answer: {compared} positions asked twice at two budgets (and once more from many threads), and one with a history given: every move, line, value, depth and node count equal");
}

// ------------------------------------------------------------------------------------------------------------------------------ (6) the rate

#[test]
fn the_rate_nodes_a_second_on_the_fixture_three_runs_medians() {
    let (fx, ps) = puzzles();
    let g = open(&fx);
    let take: Vec<usize> = (0..ps.len()).step_by((ps.len() / 100).max(1)).collect();
    let mut rates = vec![];
    for run in 0..3 {
        let (mut nodes, t0) = (0u64, Instant::now());
        for &i in &take { nodes += ask(&g, &ps[i].text, BUDGETS[1], none()).nodes; }
        let secs = t0.elapsed().as_secs_f64();
        let rate = nodes as f64 / secs;
        println!("OBS 79.25 rate run {run}: {} positions at {} nodes, {} nodes in {:.2} s: {:.0} nodes a second", take.len(), BUDGETS[1], nodes, secs, rate);
        rates.push(rate);
    }
    rates.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!("OBS 79.25 rate: median of three runs {:.0} nodes a second (one thread, the search of the layer's one entry, a count judge)", rates[1]);
    assert!(rates[1] > 0.0);
}
