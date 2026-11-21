//! PLAN 79.21 STAGE B: TESTS OF LAYER 2 FOR THE GRAPH (SYSTEM_SPEC 11 "one graph is the representation; the rules are the next layer"; piece: representation, the rules above it).
//! (1) The board in the fixture of layer 1 is the board the rules' description gives (a changed description makes the fixture stale and this fails).
//! (2) On EVERY position of both fixture games the graph made the new way (the units and the tempo from facts or numbers by layer 1, the relations written through the door by layer 2) equals the graph made the OLD way node by node and edge by edge: the old way is kept here as a
//!     reference, the code of plan 78.52 as it was before this plan (a copy, so that a change of either shows). Also through the facts the rules give: the same.
//! (3) What layer 2 hands over is the rules' reading only: the graph from a set of facts plus the rules' relations is the graph from the rules' cells plus the same relations.
use cs_game::kernel::KernelGame;
use cs_game::placed::{NoStep, Placed};
use cs_game::{Fact, Game};
use cs_rep::graph::{Board, LinkRec, LinkType, NodeRec, NodeType, PositionPart, RelRec, UnitRec};
use cs_rep::names::{Names, Side};
use cs_tactics::sheet::base_graph::Static;
use std::path::PathBuf;

mod rep_common;
use rep_common::{board_from_rules, fixtures, replay, text_of};

// ------------------------------------------------------------------------------------------------------------------------------ the old way: plan 78.52's code as it was, kept as the reference

fn old_relation(st: &Board, part: &mut PositionPart, kind: NodeType, ui: usize, to: u16, ns: u32) -> RelRec {
    let u = part.units[ui];
    let id = format!("{}({},place:{})", kind.word(), part.nodes[(u.node - ns) as usize].id, st.names.places[to as usize]);
    let node = ns + part.nodes.len() as u32;
    part.nodes.push(NodeRec { kind, id });
    let mut push = |kind: LinkType, b: u32| part.links.push(LinkRec { kind, a: node, b, via: -1, value: -1 });
    push(LinkType::By, u.node);
    push(LinkType::AtPlace, st.place_node[to as usize]);
    let target = part.occupant[to as usize];
    if target != 0 { push(LinkType::AtUnit, part.units[target as usize - 1].node); }
    let along = st.common_lines(u.at, to);
    let mut between = vec![];
    for &l in &along {
        push(LinkType::Along, st.line_node[l as usize]);
        for q in st.strictly_between(l, u.at, to) {
            let o = part.occupant[q as usize];
            if o != 0 { between.push(o as usize - 1); push(LinkType::Between, part.units[o as usize - 1].node); }
        }
    }
    RelRec { unit: ui, place: to, along, between, node }
}

fn old_position<G: Game + Placed>(st: &Board, g: &G, s: &G::State) -> PositionPart {
    let np = st.names.places.len();
    let ns = st.nodes.len() as u32;
    let mover = g.mover(s);
    let mut cells = g.cells(s);
    cells.sort_by_key(|(o, k, p)| (*o, *k, *p));
    let mut nodes: Vec<NodeRec> = vec![];
    let mut links: Vec<LinkRec> = vec![];
    let mut units: Vec<UnitRec> = Vec::with_capacity(cells.len());
    let mut occupant = vec![0u16; np];
    for (owner, kind, at) in cells {
        let side = match mover { Some(m) if m != owner => Side::Them, _ => Side::Us };
        let id = format!("unit:{}:{}:{}", st.names.roles[owner], st.names.kinds[kind as usize], st.names.places[at as usize]);
        let node = ns + nodes.len() as u32;
        nodes.push(NodeRec { kind: NodeType::Unit, id });
        links.push(LinkRec { kind: LinkType::UnitOn, a: node, b: st.place_node[at as usize], via: -1, value: -1 });
        links.push(LinkRec { kind: LinkType::UnitKind, a: node, b: st.kind_node[kind as usize], via: -1, value: -1 });
        links.push(LinkRec { kind: LinkType::UnitSide, a: node, b: st.side_node[owner], via: -1, value: -1 });
        occupant[at as usize] = units.len() as u16 + 1;
        units.push(UnitRec { owner, side, kind, at, node });
    }
    let mut part = PositionPart { mover, nodes, links, units, bears: vec![], moves: vec![], shuts: vec![], throughs: vec![], occupant };
    for ui in 0..part.units.len() {
        let u = part.units[ui];
        for p in g.bearing_from(s, u.owner, u.kind, u.at) { let r = old_relation(st, &mut part, NodeType::Bear, ui, p, ns); part.bears.push(r); }
        for p in g.reach_from(s, u.owner, u.kind, u.at) { let r = old_relation(st, &mut part, NodeType::MayMove, ui, p, ns); part.moves.push(r); }
    }
    if let Some(m) = mover {
        for ui in 0..part.units.len() {
            let u = part.units[ui];
            if u.owner != m { continue; }
            for &(l, i) in &st.place_lines[u.at as usize] {
                let ends: Vec<i32> = part.bears.iter().filter(|r| r.unit == ui && r.along.contains(&l)).map(|r| st.index_on(l, r.place) - i as i32).collect();
                for sign in [1i32, -1] {
                    let Some(far) = ends.iter().filter(|d| d.signum() == sign).map(|d| d.abs()).max() else { continue };
                    let line = &st.lines[l as usize];
                    let mut k = i as i32 + sign * (far + 1);
                    while k >= 0 && (k as usize) < line.places.len() {
                        let q = line.places[k as usize];
                        if let (NoStep::LineShut, _) = g.why_not(s, u.at, q) { let r = old_relation(st, &mut part, NodeType::Shut, ui, q, ns); part.shuts.push(r); }
                        k += sign;
                    }
                }
            }
        }
    }
    if !st.cont.is_empty() {
        let mut found: Vec<(usize, u16)> = vec![];
        for r in &part.bears {
            let u = part.units[r.unit];
            let b = part.occupant[r.place as usize];
            if b == 0 { continue; }
            for &l in &r.along {
                let (i, j) = (st.index_on(l, u.at), st.index_on(l, r.place));
                if i < 0 || j < 0 || j == i || !st.continues(u.kind, l, j > i) { continue; }
                let sign = if j > i { 1 } else { -1 };
                let line = &st.lines[l as usize];
                let mut k = j + sign;
                while k >= 0 && (k as usize) < line.places.len() {
                    let q = line.places[k as usize];
                    if part.occupant[q as usize] != 0 {
                        if st.strictly_between(l, u.at, q).iter().filter(|x| part.occupant[**x as usize] != 0).count() == 1 { found.push((r.unit, q)); }
                        break;
                    }
                    k += sign;
                }
            }
        }
        found.sort_unstable();
        found.dedup();
        for (ui, q) in found { let r = old_relation(st, &mut part, NodeType::Through, ui, q, ns); part.throughs.push(r); }
    }
    part
}

/// Two graphs equal in every field: nodes, links, units, every relation list, who stands where.
fn same(a: &PositionPart, b: &PositionPart) -> bool {
    a.mover == b.mover && a.nodes == b.nodes && a.links == b.links && a.units == b.units && a.bears == b.bears && a.moves == b.moves && a.shuts == b.shuts && a.throughs == b.throughs && a.occupant == b.occupant
}

#[test]
fn layer2_the_board_of_the_fixture_is_the_board_the_description_gives() {
    for fx in fixtures() {
        let k = KernelGame::new(&fx.name, 518).unwrap();
        let from_rules = board_from_rules(&k, &fx.name, &fx.board);
        assert_eq!(from_rules.to_json(), fx.board.to_json(), "{}: the board of the fixture is the one the description gives now", fx.name);
        println!("OBS 79.21 B-5 layer 2 {}: the board of the fixture equals the board read from the description now ({} places, {} lines)", fx.name, from_rules.names.places.len(), from_rules.lines.len());
    }
}

#[test]
fn layer2_the_graph_made_the_new_way_equals_the_graph_made_the_old_way_on_every_fixture_position() {
    for fx in fixtures() {
        let k = KernelGame::new(&fx.name, 518).unwrap();
        let mut st = Static::from_board(fx.board.clone());
        let (mut n, mut nodes, mut links, mut through_seen, mut via_facts) = (0, 0usize, 0usize, 0usize, 0);
        // the profile of the kinds is learned from the first positions of the first game, then every position is compared with the profile in place (the lines through exist from then on)
        let mut learned = false;
        for (gi, moves) in fx.games.iter().enumerate() {
            for (ply, s) in replay(&k, moves).iter().enumerate() {
                if !learned && gi == 0 && ply == 30.min(moves.len()) { let p = st.position(&k, s); st.learn(&p); learned = true; }
                let new = st.position(&k, s);
                let old = old_position(&st, &k, s);
                assert!(same(&new, &old), "{}: game {gi} ply {ply}: the graph made the new way differs from the old way", fx.name);
                // through the facts the rules give: the units and the tempo from facts alone, the relations written by the rules
                let mut by_facts = st.board().position_from_facts(&k.base_facts(s)).unwrap();
                st.write_rules(&k, s, &mut by_facts);
                assert!(same(&new, &by_facts), "{}: game {gi} ply {ply}: through facts", fx.name);
                via_facts += 1;
                n += 1;
                nodes += new.nodes.len();
                links += new.links.len();
                through_seen += new.throughs.len();
            }
        }
        println!("OBS 79.21 B-6 layer 2 {}: {n} positions, the graph made the new way equals the old way node by node and edge by edge in every field ({nodes} position nodes, {links} position links, {through_seen} lines through; profile learned: {learned}); through the facts the rules give: equal in {via_facts}", fx.name);
        assert!(n >= 300 && learned);
        let _: Vec<Fact> = vec![];
    }
}

#[test]
fn layer2_the_graph_kept_step_by_step_equals_a_fresh_build_at_every_ply_and_on_undo() {
    use cs_tactics::sheet::keeper::GraphKeeper;
    use std::time::Instant;
    for fx in fixtures() {
        let k = KernelGame::new(&fx.name, 518).unwrap();
        let mut st = Static::from_board(fx.board.clone());
        let first = replay(&k, &fx.games[0]);
        let p = st.position(&k, &first[30.min(first.len() - 1)]);
        st.learn(&p);
        let (mut plies, mut asked, mut total, mut undone) = (0usize, 0usize, 0usize, 0usize);
        let (mut t_make, mut t_part, mut t_fresh) = (0f64, 0f64, 0f64);
        for moves in &fx.games {
            let states = replay(&k, moves);
            let fresh: Vec<PositionPart> = states.iter().map(|s| st.position(&k, s)).collect();
            let mut kp = GraphKeeper::new(&st, &k, &states[0]);
            assert!(same(&kp.part(&st), &fresh[0]));
            for i in 1..states.len() {
                let t0 = Instant::now();
                kp.make(&st, &k, &states[i]);
                t_make += t0.elapsed().as_secs_f64();
                let t1 = Instant::now();
                let part = kp.part(&st);
                t_part += t1.elapsed().as_secs_f64();
                assert!(same(&part, &fresh[i]), "{}: ply {i}: the graph kept step by step differs from a fresh build", fx.name);
                let (a, t) = kp.asked();
                asked += a; total += t; plies += 1;
            }
            // take every tempo back: the graph of the position before, exactly, and no rule asked
            for i in (1..states.len()).rev() {
                assert!(kp.undo());
                assert!(same(&kp.part(&st), &fresh[i - 1]), "{}: undo to ply {}", fx.name, i - 1);
                undone += 1;
            }
            assert!(!kp.undo());
            // and forward again from the start with a few makes after a partial undo
            let top = states.len().min(6);
            for i in 1..top { kp.make(&st, &k, &states[i]); }
            kp.undo(); kp.undo();
            assert!(same(&kp.part(&st), &fresh[top - 3]));
            let t2 = Instant::now();
            for s in &states { std::hint::black_box(st.position(&k, s)); }
            t_fresh += t2.elapsed().as_secs_f64();
        }
        println!("OBS 79.21 B-7 layer 2 {}: the graph kept step by step equals a fresh build at {plies} plies and on {undone} undos; units asked of the rules again {asked} of {total} ({:.0} percent); seconds: make {:.4}, make then graph {:.4}, fresh builds of the same positions {:.4}", fx.name, 100.0 * asked as f64 / total as f64, t_make, t_make + t_part, t_fresh);
    }
}

#[test]
fn layer2_a_question_asked_of_the_kept_graph_is_answered_as_on_the_graph_made_from_the_position() {
    use cs_decision::goals::{tied_bearer, tied_bearer_live};
    use cs_tactics::sheet::keeper::GraphKeeper;
    for fx in fixtures() {
        let k = KernelGame::new(&fx.name, 518).unwrap();
        let st = Static::from_board(fx.board.clone());
        let (mut asked, mut yes) = (0u32, 0u32);
        for moves in &fx.games {
            let states = replay(&k, moves);
            let mut kp = GraphKeeper::new(&st, &k, &states[0]);
            for (i, s) in states.iter().enumerate() {
                if i > 0 { kp.make(&st, &k, s); }
                let part = st.position(&k, s);
                for att in 0..2usize {
                    for t in 0..st.names.places.len() as u16 {
                        let (a, b) = (tied_bearer(&part, att, 1 - att, t), tied_bearer_live(&st, kp.live(), att, 1 - att, t));
                        assert_eq!(a, b, "{}: ply {i} attacker {att} place {t}", fx.name);
                        asked += 1;
                        yes += a as u32;
                    }
                }
            }
        }
        println!("OBS 79.21 B-8 layer 2 {}: {asked} questions 'is a bearer tied' asked of the kept graph and of the graph made from the position: equal in all ({yes} yes)", fx.name);
        assert!(yes > 0);
    }
}

/// THE PROFILE of the one place the wake built a graph for each candidate move (`answers_against`): for every legal move of the side to move in the fixture positions, the answer to "is a bearer tied" in the position after it, the old way (a graph built from the state after the move, then asked) and the new way (the kept graph
/// changed by the move, asked, changed back). Printed: microseconds a candidate each way, how many units the rules were asked about again, and the equal answers.
#[test]
fn layer2_profile_a_candidate_move_asked_of_the_kept_graph_against_a_graph_built_for_it() {
    use cs_decision::goals::{tied_bearer, tied_bearer_live};
    use cs_tactics::sheet::keeper::GraphKeeper;
    use std::time::Instant;
    for fx in fixtures() {
        let k = KernelGame::new(&fx.name, 518).unwrap();
        let st = Static::from_board(fx.board.clone());
        let (mut n, mut equal, mut t_old, mut t_new, mut asked, mut total) = (0u32, 0u32, 0f64, 0f64, 0usize, 0usize);
        for moves in &fx.games {
            for s in replay(&k, moves).iter().step_by(4) {
                let Some(me) = k.mover(s) else { continue };
                let mut kp = GraphKeeper::new(&st, &k, s);
                for m in k.moves(s) {
                    let after = k.apply(s, &m);
                    let Some(dp) = k.pass(&after) else { continue };
                    let t0 = Instant::now();
                    let a = tied_bearer(&st.position(&k, &dp), me, 1 - me, 0);
                    t_old += t0.elapsed().as_secs_f64();
                    let t1 = Instant::now();
                    kp.make(&st, &k, &dp);
                    let b = tied_bearer_live(&st, kp.live(), me, 1 - me, 0);
                    let (x, y) = kp.asked();
                    kp.undo();
                    t_new += t1.elapsed().as_secs_f64();
                    asked += x; total += y; n += 1; equal += (a == b) as u32;
                }
            }
        }
        println!("OBS 79.21 B-9 profile {}: {n} candidate moves; a graph built for each: {:.1} microseconds a candidate; the kept graph changed by the move, asked and changed back: {:.1} microseconds; units asked of the rules again {:.0} percent; answers equal in {equal} of {n}", fx.name, 1e6 * t_old / n as f64, 1e6 * t_new / n as f64, 100.0 * asked as f64 / total as f64);
        assert_eq!(equal, n);
    }
}

