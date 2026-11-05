//! PLAN 78.6 THE CENSUS OF TACTICS BY THEIR DEFECTS (Kaveh 2026-11-04: "are we able to identify all the tactics in the position? ... even tactics that may not work
//! right now ... one defect, two defects ... maybe three"). Game-free: it reads the `Game` trait and the ledger search only.
//!
//! A TACTIC of the side to move is a forced line whose ledger the judge scores above nothing (with no values: they lose a unit, we lose none; or the game's own
//! goal). Its DEFECTS are counted in TEMPO: the free moves of ours (the other side passing: `Game::pass`) it needs before it works -- 0: it works now; 1: one move
//! short (a threat: our move, their pass, the gain -- the one definition of forcing's pass clause); 2, 3: two, three moves short. A free move is a quiet one (it takes
//! nothing; a move after which they cannot pass -- a check -- is no free move). Each tactic is counted at its least number of defects: one found at k is not counted
//! again at k + 1. A tactic is named by its first forcing move and what it wins. Each reading is the ledger search at a fixed depth under a node fuse (every position
//! the same budget); a cut reading is reported, never counted as nothing.
use crate::ledger::{Judge, Search, WIN};
use cs_game::Game;
use std::collections::HashSet;

#[derive(Debug, Clone)]
pub struct Tactic {
    pub defects: u32,
    pub free: Vec<String>,             // the free moves it needs (empty: it works now)
    pub line: Vec<String>,             // the forced line from where the free moves leave us
    pub wins: Vec<String>,             // what the line takes of theirs (kind names), or "goal" (the game's own outcome)
    pub ledger: Vec<String>,           // every event of the line ("they lose n", "we lose p", "we promote p to q")
    pub value: i64,
}

#[derive(Debug, Default, Clone)]
pub struct Report { pub tactics: Vec<Tactic>, pub readings: usize, pub cut: usize, pub nodes: u64, pub frontier: Vec<usize> }

/// The census of the side to move at `s`: its tactics with at most `max_defects` defects, each reading `depth` plies under `fuse` nodes.
pub fn census<G: Game, J: Judge>(g: &G, s: &G::State, judge: &J, max_defects: u32, depth: i32, fuse: u64) -> Report {
    let mut rep = Report::default();
    let Some(me) = g.mover(s) else { return rep };
    let mut seen: HashSet<String> = HashSet::new();
    let mut frontier: Vec<(G::State, Vec<String>)> = vec![(s.clone(), vec![])];
    for k in 0..=max_defects {
        rep.frontier.push(frontier.len());
        for (st, free) in &frontier {
            let mut se = Search::with_fuse(g, judge, me, fuse);
            let vals = se.gainers(st, depth);
            rep.readings += 1;
            rep.nodes += se.nodes;
            if se.cut() { rep.cut += 1; }
            for (i, v) in vals {
                if v <= 0 { break; }                                  // (best first: the rest gain nothing)
                let (line, ev) = se.line_from(st, i, depth as usize + 6);
                let wins: Vec<String> = if v >= WIN - 1000 { vec!["goal".into()] } else {
                    let mut w: Vec<String> = ev.iter().filter(|(o, e)| *o != me && e.starts_with("lose ")).map(|(_, e)| e[5..].to_string()).collect();
                    w.sort();
                    w
                };
                let key = format!("{}|{}", line.first().cloned().unwrap_or_default(), wins.join(","));
                if !seen.insert(key) { continue; }
                let who = |o: cs_game::Role| if o == me { "we" } else { "they" };
                rep.tactics.push(Tactic { defects: k, free: free.clone(), line, wins, ledger: ev.iter().map(|(o, e)| format!("{} {e}", who(*o))).collect(), value: v });
            }
        }
        if k == max_defects { break; }
        let mut next = vec![];
        for (st, free) in &frontier {
            for m in g.moves(st) {
                if !g.captured(st, &m).is_empty() { continue; }       // a free move takes nothing
                let q = g.apply(st, &m);
                let Some(q2) = g.pass(&q) else { continue };          // (they cannot pass: no free move)
                if g.mover(&q2) != Some(me) { continue; }
                let mut f = free.clone();
                f.push(g.move_name(st, &m));
                next.push((q2, f));
            }
        }
        frontier = next;
    }
    rep
}
