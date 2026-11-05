//! INSTANCES BY WATCHING (plan 78.4): an outside TEACHER (any source of moves: cs_game::outside, a person, a record) played against itself, the game judged by the `Game` trait (its legal moves, its outcome;
//! threefold repetition by the game's state keys -- the record's rule). Each game opens with seeded random legal moves (a named count: the openings differ), then the
//! engine plays both sides at a fixed move time. One thread per game slot, each with its own engine process. The engine is a teacher outside the system: what it
//! plays is DATA for the miners, never a rule or a value inside the engine.
use crate::Record;
use cs_game::rng::Rng;
use cs_game::Game;
use std::collections::HashMap;

/// A TEACHER: an outside source of moves to watch (an engine, a person, a record). `start` opens a game from its descriptor; `best` answers the move after the
/// moves played so far (None: it has no move -- the game ends unfinished).
pub trait Teacher {
    fn start(&mut self, start: &str);
    fn best(&mut self, moves: &[String]) -> Option<String>;
}

/// One game watched: `random_plies` seeded random legal moves, then the engine for both sides; ended by the game's outcome or a threefold repetition (a draw for
/// every role: 50 each). -> the record; an engine move the game does not hold ends the game unfinished (outcome empty).
pub fn one<G: Game, T: Teacher + ?Sized>(g: &G, eng: &mut T, label: &str, random_plies: usize, rng: &mut Rng) -> Record {
    let mut s = g.initial();
    let mut moves: Vec<String> = vec![];
    let mut seen: HashMap<u64, u32> = HashMap::new();
    loop {
        if let Some(o) = g.outcome(&s) { return Record { start: label.to_string(), moves, outcome: o }; }
        let k = g.key(&s);
        let n = seen.entry(k).or_insert(0);
        *n += 1;
        if *n >= 3 { return Record { start: label.to_string(), moves, outcome: vec![50; g.roles().len()] }; }
        let name = if moves.len() < random_plies {
            let mut names: Vec<String> = g.moves(&s).iter().map(|m| g.move_name(&s, m)).collect();
            names.sort();
            names[rng.below(names.len())].clone()
        } else {
            match eng.best(&moves) { Some(x) => x, None => return Record { start: label.to_string(), moves, outcome: vec![] } }
        };
        let Some(m) = g.find_move(&s, &name) else { return Record { start: label.to_string(), moves, outcome: vec![] } };
        s = g.apply(&s, &m);
        moves.push(name);
    }
}

/// `games` games watched on `threads` threads (each its own teacher, from `teacher()`; game i seeded `seed + i`). `make(i)` builds game i: (the game, the teacher's
/// start, the record's start descriptor).
pub fn watch<G: Game, T: Teacher, F: Fn(usize) -> (G, String, String) + Sync, M: Fn() -> Option<T> + Sync>(make: F, teacher: M, games: usize, random_plies: usize,
                                                                                                       seed: u64, threads: usize) -> Vec<Record> {
    let next = std::sync::atomic::AtomicUsize::new(0);
    let out = std::sync::Mutex::new(vec![None; games]);
    std::thread::scope(|sc| {
        for _ in 0..threads.max(1) {
            sc.spawn(|| {
                let Some(mut eng) = teacher() else { return };
                loop {
                    let i = next.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    if i >= games { break; }
                    let (g, start, label) = make(i);
                    eng.start(&start);
                    let mut rng = Rng::new(seed + i as u64);
                    let r = one(&g, &mut eng, &label, random_plies, &mut rng);
                    out.lock().unwrap()[i] = Some(r);
                }
            });
        }
    });
    out.into_inner().unwrap().into_iter().flatten().collect()
}
