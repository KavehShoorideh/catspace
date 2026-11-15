//! PLAN 78.53 THE GRAPH, LAYER 2: patterns over the layer-1 graph, matched and counted. Game-free: no unit, place, line or kind of any game is named; every word arrives as data.
//!
//! THE LANGUAGE. A pattern is a small connected set of RELATION NODES of the graph (a size is the number of them), written over variables: a UNIT variable stands for a unit of the position, a
//! PLACE variable for an EMPTY place, a LINE variable for a line. The atoms are the relation nodes of layer 1, with their plain links:
//!   bear(u, t, l)      a unit bears on a target (a unit, or an empty place), along a line where the bearing follows one
//!   move(u, t, l)      a unit may move to a target
//!   shut(u, t, l, b)   a unit runs along a line and unit b, the one unit between, shuts it before the target (the first-blocker form of layer 1, for BOTH sides)
//! The line is absent where the relation follows no line (a leap). A relation that holds two lines takes its first. Two atoms that name the same object share the variable: an embedding of a
//! set of relation nodes in a position gives the pattern, the sharing is the pattern's structure. No kind, place or line is named. Unit attributes, by LEVEL (a coarser level forgets):
//!   level 0  the side of each unit, "mine" or "theirs" (written as the bit 1 = theirs), relative to the side the pattern is counted for
//!   level 1  + whether the unit is the goal unit of its side
//!   level 2  level 1 + whether the unit stands on an edge (the nearest edge distance is 0): the one place absoluteness enters, as a small class
//!   level 3  level 1 + the KIND of each unit, the raw symbol the rules give (its index among the graph's kind nodes), never a name: the kind-bound children of the level 1 pattern of the same atoms
//! No worth enters a pattern.
//!
//! CANONICAL FORM. The encoding of an embedding renames variables by first appearance over the atoms in an order and takes the least encoding over the orders of the atoms, then (the profile
//! appended) over swapping mine and theirs. The same pattern met in two orders of growth is one key. A swapped pattern is the same concept read from the other side, its sign flips: the value of a
//! concept at a position is (count for the side to move) less (count for the other), and an embedding that is its own swap counts nothing.
//!
//! COUNTING. Embeddings are the connected sets of relation nodes (ESU enumeration over the nodes' sharing of an object). `freq` counts the support of each key over rows, growing a size from the
//! kept patterns of the size below (a pattern is a candidate only when one of its connected parts of one relation node less is kept: growth by size); `count` gives, per row, the counts of chosen keys
//! for each side.
use super::base_graph::{PositionPart, Static};
use super::Side;
pub use rustc_hash::{FxHashMap, FxHashSet};
use rustc_hash::FxHasher;
use std::hash::Hasher;

pub const MAX_SIZE: usize = 4;
pub const LEVELS: u8 = 4;
const NONE: u8 = 255;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Obj { Unit(u16), Place(u16) }

/// One atom of a position: a relation node (with one chosen unit between, for a shut line).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Item { pub ty: u8, pub actor: u16, pub target: Obj, pub line: i32, pub between: i32 }

/// A position as the matcher reads it: its atoms, who shares what, and the attributes of the units.
pub struct Board {
    pub items: Vec<Item>,
    pub nbrs: Vec<Vec<u32>>,
    pub mine: Vec<bool>,
    pub goal: Vec<bool>,
    pub kind: Vec<u8>,
    pub edge: Vec<bool>,
}

impl Board {
    /// The board of a position of the graph. `goal_kind` the kind of the goal unit (a side's goal unit is its only unit of that kind).
    pub fn new(st: &Static, p: &PositionPart, goal_kind: Option<u8>) -> Board {
        let nu = p.units.len();
        let np = st.names.places.len();
        let mut items = vec![];
        let target_of = |pl: u16| -> Obj { match p.occupant[pl as usize] { 0 => Obj::Place(pl), i => Obj::Unit(i - 1) } };
        for r in &p.bears { items.push(Item { ty: 0, actor: r.unit as u16, target: target_of(r.place), line: r.along.first().map_or(-1, |l| *l as i32), between: -1 }); }
        for r in &p.moves { items.push(Item { ty: 1, actor: r.unit as u16, target: target_of(r.place), line: r.along.first().map_or(-1, |l| *l as i32), between: -1 }); }
        for r in &p.shuts {
            if r.between.len() != 1 || r.along.is_empty() { continue; }
            items.push(Item { ty: 2, actor: r.unit as u16, target: target_of(r.place), line: r.along[0] as i32, between: r.between[0] as i32 });
        }
        let goal_of = |side: Side| -> Option<usize> {
            let k = goal_kind?;
            let mut f = p.units.iter().enumerate().filter(|(_, u)| u.side == side && u.kind == k).map(|(i, _)| i);
            let first = f.next()?;
            if f.next().is_some() { None } else { Some(first) }
        };
        let goals = [goal_of(Side::Us), goal_of(Side::Them)];
        let mine = p.units.iter().map(|u| u.side == Side::Us).collect();
        let goal = (0..nu).map(|i| goals.contains(&Some(i))).collect();
        let w = p.units.iter().map(|u| u.kind).collect();
        let edge = p.units.iter().map(|u| st.nearest_edge(u.at) == 0).collect();
        Board::from_items(items, nu, np, st.lines.len(), mine, goal, w, edge)
    }

    /// A board from its atoms and the attributes of its units (`nu` units, `np` places, `nl` lines).
    #[allow(clippy::too_many_arguments)]
    pub fn from_items(items: Vec<Item>, nu: usize, np: usize, nl: usize, mine: Vec<bool>, goal: Vec<bool>, kind: Vec<u8>, edge: Vec<bool>) -> Board {
        let mut by_obj: Vec<Vec<u32>> = vec![vec![]; nu + np + nl];
        for (i, it) in items.iter().enumerate() {
            for o in obj_ids(it, nu, np) { if o >= 0 { by_obj[o as usize].push(i as u32); } }
        }
        let mut nbrs: Vec<Vec<u32>> = vec![vec![]; items.len()];
        for list in &by_obj {
            for &a in list { for &b in list { if a != b { nbrs[a as usize].push(b); } } }
        }
        for n in nbrs.iter_mut() { n.sort_unstable(); n.dedup(); }
        Board { items, nbrs, mine, goal, kind, edge }
    }

    fn adjacent(&self, a: usize, b: usize) -> bool { self.nbrs[a].binary_search(&(b as u32)).is_ok() }

    /// Every connected set of `k` atoms (ESU: each set once), by the atoms' indices, in ascending order of the smallest.
    pub fn connected_sets(&self, k: usize, visit: &mut impl FnMut(&[usize])) {
        let n = self.items.len();
        let mut sub: Vec<usize> = Vec::with_capacity(k);
        for v in 0..n {
            sub.clear();
            sub.push(v);
            let ext: Vec<usize> = self.nbrs[v].iter().map(|x| *x as usize).filter(|&u| u > v).collect();
            self.extend(&mut sub, ext, v, k, visit);
        }
    }

    fn extend(&self, sub: &mut Vec<usize>, mut ext: Vec<usize>, v: usize, k: usize, visit: &mut impl FnMut(&[usize])) {
        if sub.len() == k { visit(sub); return; }
        while let Some(w) = ext.pop() {
            let mut next = ext.clone();
            for &u in &self.nbrs[w] {
                let u = u as usize;
                if u > v && !sub.contains(&u) && !next.contains(&u) && !sub.iter().any(|&s| self.adjacent(s, u)) { next.push(u); }
            }
            sub.push(w);
            self.extend(sub, next, v, k, visit);
            sub.pop();
        }
    }

    /// The connected sets of `its` of one atom less (by position in `its`).
    fn parts_of(&self, its: &[usize]) -> Vec<Vec<usize>> {
        let k = its.len();
        let mut out = vec![];
        for drop in 0..k {
            let rest: Vec<usize> = (0..k).filter(|&i| i != drop).map(|i| its[i]).collect();
            // connected? flood from the first
            let mut seen = vec![false; rest.len()];
            let mut stack = vec![0usize];
            seen[0] = true;
            while let Some(a) = stack.pop() {
                for b in 0..rest.len() { if !seen[b] && self.adjacent(rest[a], rest[b]) { seen[b] = true; stack.push(b); } }
            }
            if seen.iter().all(|x| *x) { out.push(rest); }
        }
        out
    }
}

fn obj_ids(it: &Item, nu: usize, np: usize) -> [i64; 4] {
    let t = match it.target { Obj::Unit(u) => u as i64, Obj::Place(p) => (nu + p as usize) as i64 };
    [it.actor as i64, t, if it.line >= 0 { (nu + np) as i64 + it.line as i64 } else { -1 }, if it.between >= 0 { it.between as i64 } else { -1 }]
}

/// The key of an embedding at one level: the hash of its canonical encoding, the encoding itself, and the sign (+1 the embedding is the concept read for the side to move, -1 for the other
/// side, 0 an embedding that is its own swap).
#[derive(Clone)]
pub struct Key { pub hash: u64, pub sign: i8, pub bytes: Vec<u8> }

const PERMS: [&[&[usize]]; 5] = [
    &[],
    &[&[0]],
    &[&[0, 1], &[1, 0]],
    &[&[0, 1, 2], &[0, 2, 1], &[1, 0, 2], &[1, 2, 0], &[2, 0, 1], &[2, 1, 0]],
    &[&[0, 1, 2, 3], &[0, 1, 3, 2], &[0, 2, 1, 3], &[0, 2, 3, 1], &[0, 3, 1, 2], &[0, 3, 2, 1], &[1, 0, 2, 3], &[1, 0, 3, 2], &[1, 2, 0, 3], &[1, 2, 3, 0], &[1, 3, 0, 2], &[1, 3, 2, 0],
      &[2, 0, 1, 3], &[2, 0, 3, 1], &[2, 1, 0, 3], &[2, 1, 3, 0], &[2, 3, 0, 1], &[2, 3, 1, 0], &[3, 0, 1, 2], &[3, 0, 2, 1], &[3, 1, 0, 2], &[3, 1, 2, 0], &[3, 2, 0, 1], &[3, 2, 1, 0]],
];

/// One candidate structure encoding: the encoding and the units in order of their variable.
struct Cand { enc: Vec<u8>, units: Vec<u16> }

fn var_of(list: &mut Vec<i64>, id: i64) -> u8 {
    match list.iter().position(|x| *x == id) { Some(i) => i as u8, None => { list.push(id); (list.len() - 1) as u8 } }
}

impl Board {
    fn structure(&self, its: &[usize], perm: &[usize]) -> Cand {
        let (mut uv, mut pv, mut lv): (Vec<i64>, Vec<i64>, Vec<i64>) = (vec![], vec![], vec![]);
        let mut enc = Vec::with_capacity(6 * perm.len() + 2);
        for &ix in perm {
            let it = &self.items[its[ix]];
            enc.push(it.ty);
            let a = var_of(&mut uv, it.actor as i64);
            enc.push(a);
            match it.target {
                Obj::Unit(u) => { let t = var_of(&mut uv, u as i64); enc.push(0); enc.push(t); }
                Obj::Place(p) => { let t = var_of(&mut pv, p as i64); enc.push(1); enc.push(t); }
            }
            enc.push(if it.line >= 0 { var_of(&mut lv, it.line as i64) } else { NONE });
            if it.between >= 0 { let b = var_of(&mut uv, it.between as i64); enc.push(b); } else { enc.push(NONE); }
        }
        Cand { enc, units: uv.iter().map(|x| *x as u16).collect() }
    }

    fn profile(&self, c: &Cand, level: u8, swap: bool) -> Vec<u8> {
        let mut out = Vec::with_capacity(3 * c.units.len() + 1);
        for &u in &c.units { out.push((self.mine[u as usize] == swap) as u8); }          // 1 = theirs: the least encoding then writes the pattern with its first units mine
        if level >= 1 { for &u in &c.units { out.push(self.goal[u as usize] as u8); } }
        if level == 2 { for &u in &c.units { out.push(self.edge[u as usize] as u8); } }
        if level == 3 { for &u in &c.units { out.push(self.kind[u as usize]); } }
        out
    }

    /// The keys of the embedding `its` at each of `levels`.
    pub fn keys(&self, its: &[usize], levels: &[u8]) -> Vec<Key> {
        let k = its.len();
        let mut best: Vec<Cand> = vec![];
        for perm in PERMS[k] {
            let c = self.structure(its, perm);
            match best.first() {
                None => best.push(c),
                Some(b) if c.enc < b.enc => { best.clear(); best.push(c); }
                Some(b) if c.enc == b.enc => best.push(c),
                _ => {}
            }
        }
        levels.iter().map(|&level| {
            let mut win: Option<Vec<u8>> = None;
            let (mut plain, mut swapped) = (false, false);
            for c in &best {
                for swap in [false, true] {
                    let mut full = vec![level, k as u8];
                    full.extend_from_slice(&c.enc);
                    full.extend(self.profile(c, level, swap));
                    match &win {
                        Some(w) if full > *w => {}
                        Some(w) if full == *w => { if swap { swapped = true } else { plain = true } }
                        _ => { win = Some(full); plain = !swap; swapped = swap; }
                    }
                }
            }
            let bytes = win.unwrap();
            let mut h = FxHasher::default();
            h.write(&bytes);
            Key { hash: h.finish(), sign: if plain && swapped { 0 } else if plain { 1 } else { -1 }, bytes }
        }).collect()
    }

    fn hash_at(&self, its: &[usize], level: u8) -> (u64, i8) { let k = self.keys(its, &[level]).pop().unwrap(); (k.hash, k.sign) }
}

/// What `freq` learns of a key.
pub struct Seen { pub support: u32, pub total: u64, pub bytes: Vec<u8>, pub parts: Vec<u64>, pub free_parent: u64, last_row: u32 }

/// The keys of the embeddings of `size` atoms of one board, with their parts' keys; `allowed` (the kept keys of the size below) makes a pattern a candidate only when one of its connected
/// parts is in it.
pub fn freq_board(b: &Board, size: usize, levels: &[u8], allowed: Option<&FxHashSet<u64>>, row: u32, into: &mut FxHashMap<u64, Seen>) {
    b.connected_sets(size, &mut |its| {
        let own = b.keys(its, levels);
        let parts = if size > 1 { b.parts_of(its) } else { vec![] };
        for (li, key) in own.iter().enumerate() {
            if key.sign == 0 { continue; }
            let level = levels[li];
            let mut part_keys: Vec<u64> = vec![];
            if size > 1 { for p in &parts { part_keys.push(b.hash_at(p, level).0); } }
            if let Some(a) = allowed { if !part_keys.iter().any(|h| a.contains(h)) { continue; } }
            let e = into.entry(key.hash).or_insert_with(|| { let mut pk = part_keys.clone(); pk.sort_unstable(); pk.dedup(); Seen { support: 0, total: 0, bytes: key.bytes.clone(), parts: pk, free_parent: if level == 3 { b.hash_at(its, 1).0 } else { 0 }, last_row: u32::MAX } });
            e.total += 1;
            if e.last_row != row { e.support += 1; e.last_row = row; }
        }
    });
}

/// The counts of the wanted keys on one board: key -> (count for the side to move, count for the other).
pub fn count_board(b: &Board, max_size: usize, levels: &[u8], want: &FxHashMap<u64, usize>, out: &mut FxHashMap<usize, (u32, u32)>) {
    for size in 1..=max_size {
        b.connected_sets(size, &mut |its| {
            for key in b.keys(its, levels) {
                if let Some(&ci) = want.get(&key.hash) {
                    let e = out.entry(ci).or_insert((0, 0));
                    if key.sign >= 0 { e.0 += 1; }
                    if key.sign <= 0 { e.1 += 1; }
                }
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn item(ty: u8, actor: u16, target: Obj, line: i32, between: i32) -> Item { Item { ty, actor, target, line, between } }

    /// Three units (0 mine, 1 and 2 theirs), two places, two lines; atoms: 0 bears on 1 along line 0, 0 bears on 2 along line 0, 1 may move to place 0.
    fn toy(swapped: bool) -> Board {
        let items = vec![item(0, 0, Obj::Unit(1), 0, -1), item(0, 0, Obj::Unit(2), 0, -1), item(1, 1, Obj::Place(0), -1, -1)];
        let (m, g) = if swapped { (vec![false, true, true], vec![false; 3]) } else { (vec![true, false, false], vec![false; 3]) };
        Board::from_items(items, 3, 2, 2, m, g, vec![0, 1, 1], vec![false; 3])
    }

    #[test]
    fn the_same_set_of_atoms_met_in_any_order_has_one_key() {
        let b = toy(false);
        let a = b.keys(&[0, 1, 2], &[0, 1, 2, 3]);
        for perm in [[2, 0, 1], [1, 2, 0], [2, 1, 0]] {
            let c = b.keys(&perm, &[0, 1, 2, 3]);
            for (x, y) in a.iter().zip(c.iter()) { assert_eq!((x.hash, x.sign), (y.hash, y.sign)); }
        }
    }

    #[test]
    fn a_pattern_read_from_the_other_side_is_the_same_key_with_the_sign_flipped() {
        let (p, q) = (toy(false), toy(true));
        let (a, b) = (p.keys(&[0, 1, 2], &[0]), q.keys(&[0, 1, 2], &[0]));
        assert_eq!(a[0].hash, b[0].hash);
        assert_eq!(a[0].sign, -b[0].sign);
        assert_ne!(a[0].sign, 0);
    }

    #[test]
    fn atoms_sharing_a_line_differ_from_atoms_on_two_lines_and_the_levels_forget() {
        let one = toy(false);
        let mut items = one.items.clone();
        items[1].line = 1;
        let two = Board::from_items(items, 3, 2, 2, one.mine.clone(), one.goal.clone(), one.kind.clone(), one.edge.clone());
        assert_ne!(one.keys(&[0, 1], &[0])[0].hash, two.keys(&[0, 1], &[0])[0].hash);
        // the two targets differ in kind: the kind-bound level reads it, the kind-free levels do not
        let mut kd = one.kind.clone();
        kd[2] = 9;
        let other = Board::from_items(one.items.clone(), 3, 2, 2, one.mine.clone(), one.goal.clone(), kd, one.edge.clone());
        assert_eq!(one.keys(&[0, 1], &[1])[0].hash, other.keys(&[0, 1], &[1])[0].hash);
        assert_ne!(one.keys(&[0, 1], &[3])[0].hash, other.keys(&[0, 1], &[3])[0].hash);
    }

    #[test]
    fn connected_sets_are_each_listed_once_and_only_when_connected() {
        let b = toy(false);
        let mut seen: Vec<Vec<usize>> = vec![];
        b.connected_sets(2, &mut |s| { let mut v = s.to_vec(); v.sort(); seen.push(v); });
        seen.sort();
        assert_eq!(seen, vec![vec![0, 1], vec![0, 2]]);          // atoms 0 and 1 share a unit and a line; atoms 0 and 2 share unit 1; atoms 1 and 2 share nothing
        let mut tri = 0;
        b.connected_sets(3, &mut |_| tri += 1);
        assert_eq!(tri, 1);
    }
}
