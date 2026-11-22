//! STRUCTURE, FORM ONE: A GRAPH (plan 79.30; SYSTEM_SPEC 11 "Layer 0 holds six things" item 4). Nodes, links with a label, neighbours, a path, distance in steps; and the pure core lifted from cs_rep's graph, the chains of a labelled successor map.
//! Interface: see the module doc of `structure`.
use super::Metric;
use crate::number::Num;
use std::collections::{BTreeMap, BTreeSet, VecDeque};

/// A graph of nodes and labelled links, each link one way (`add_both` makes one each way).
#[derive(Debug, Clone, Default)]
pub struct Graph<N: Ord + Clone, L: Ord + Clone> { out: BTreeMap<N, BTreeSet<(N, L)>> }

impl<N: Ord + Clone, L: Ord + Clone> Graph<N, L> {
    pub fn new() -> Graph<N, L> { Graph { out: BTreeMap::new() } }
    pub fn add_node(&mut self, n: N) { self.out.entry(n).or_default(); }
    /// A link from one node to another, with a label; the nodes are added if new.
    pub fn add_link(&mut self, from: N, to: N, label: L) {
        self.add_node(to.clone());
        self.out.entry(from).or_default().insert((to, label));
    }
    /// A link each way, with the same label.
    pub fn add_both(&mut self, a: N, b: N, label: L) { self.add_link(a.clone(), b.clone(), label.clone()); self.add_link(b, a, label); }
    pub fn nodes(&self) -> Vec<N> { self.out.keys().cloned().collect() }
    pub fn links(&self) -> Vec<(N, N, L)> { self.out.iter().flat_map(|(a, v)| v.iter().map(move |(b, l)| (a.clone(), b.clone(), l.clone()))).collect() }
    /// The nodes a link leads to from the node, with the labels, in order.
    pub fn neighbours(&self, n: &N) -> Vec<(N, L)> { self.out.get(n).map(|v| v.iter().cloned().collect()).unwrap_or_default() }
    pub fn has_link(&self, a: &N, b: &N) -> bool { self.out.get(a).is_some_and(|v| v.iter().any(|(m, _)| m == b)) }

    /// A shortest path from one node to another, both ends in it: the first in the order of the nodes among the shortest. A path from a node to itself is the node alone. None when there is none or a node is unknown.
    pub fn path(&self, from: &N, to: &N) -> Option<Vec<N>> {
        if !self.out.contains_key(from) || !self.out.contains_key(to) { return None; }
        let mut before: BTreeMap<N, N> = BTreeMap::new();
        let mut seen: BTreeSet<N> = BTreeSet::from([from.clone()]);
        let mut queue: VecDeque<N> = VecDeque::from([from.clone()]);
        while let Some(at) = queue.pop_front() {
            if &at == to { break; }
            let mut next: Vec<N> = self.out[&at].iter().map(|(m, _)| m.clone()).collect();
            next.dedup();
            for m in next { if seen.insert(m.clone()) { before.insert(m.clone(), at.clone()); queue.push_back(m); } }
        }
        if !seen.contains(to) { return None; }
        let mut path = vec![to.clone()];
        while let Some(b) = before.get(path.last().expect("a path has an end")) { path.push(b.clone()); }
        path.reverse();
        Some(path)
    }

    /// The distance in steps (the links of a shortest path); None when there is no path.
    pub fn steps(&self, from: &N, to: &N) -> Option<u32> { self.path(from, to).map(|p| p.len() as u32 - 1) }
}

impl<N: Ord + Clone, L: Ord + Clone> Metric for Graph<N, L> {
    type Point = N;
    fn distance(&self, a: &N, b: &N) -> Option<Num> { self.steps(a, b).map(|s| Num::whole(s as i64)) }
}

/// A chain of two nodes or more under one label, in order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Chain { pub label: usize, pub nodes: Vec<u32> }

/// What `chains` finds in a labelled successor map.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Chains {
    /// For each label, the label whose successor map is the reverse of its own (the first such), if any.
    pub reverse: Vec<Option<usize>>,
    /// The maximal chains: for the first label of each pair of reverses (or a label with none), each chain that starts where nothing leads in.
    pub lines: Vec<Chain>,
    /// For each label and node, the steps until the chain under the label ends (nought: the node is at the end).
    pub distance_to_end: Vec<Vec<u16>>,
}

/// Why a labelled successor map is not made of chains.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChainError { TwoAfter(usize), TwoBefore(usize), Closed(usize) }

/// THE LIFTED CORE of cs_rep's graph. `next[label][node]` is the successor of the node under the label (None: the end). Every label must be a chain: no node with two successors or two predecessors (a node listed twice with two different
/// successors is `TwoAfter`; this form holds one successor per node, so only `TwoBefore` can show: two nodes leading to one), and no closed loop.
pub fn chains(next: &[Vec<Option<u32>>]) -> Result<Chains, ChainError> {
    let nd = next.len();
    let np = next.first().map_or(0, |v| v.len());
    let mut prev: Vec<Vec<Option<u32>>> = vec![vec![None; np]; nd];
    for d in 0..nd {
        for p in 0..np {
            if let Some(q) = next[d][p] {
                if prev[d][q as usize].replace(p as u32).is_some_and(|x| x != p as u32) { return Err(ChainError::TwoBefore(d)); }
            }
        }
    }
    let reverse: Vec<Option<usize>> = (0..nd).map(|d| (0..nd).find(|&e| next[e] == prev[d] && prev[d].iter().any(|x| x.is_some()))).collect();
    let mut lines: Vec<Chain> = vec![];
    for d in 0..nd {
        if reverse[d].is_some_and(|e| e < d) { continue; }
        let mut seen = vec![false; np];
        for p in 0..np {
            if next[d][p].is_none() || prev[d][p].is_some() { continue; }
            let mut nodes = vec![p as u32];
            seen[p] = true;
            let mut at = p;
            while let Some(q) = next[d][at] { nodes.push(q); seen[q as usize] = true; at = q as usize; }
            lines.push(Chain { label: d, nodes });
        }
        if (0..np).any(|p| next[d][p].is_some() && !seen[p]) { return Err(ChainError::Closed(d)); }
    }
    let distance_to_end: Vec<Vec<u16>> = (0..nd).map(|d| (0..np).map(|p| {
        let (mut n, mut at) = (0u16, p);
        while let Some(q) = next[d][at] { n += 1; at = q as usize; if n as usize > np { break; } }
        n
    }).collect()).collect();
    Ok(Chains { reverse, lines, distance_to_end })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::maker::Maker;

    /// A graph of `n` nodes with links made by the seeded maker, both ways.
    fn make(m: &mut Maker, n: i64) -> Graph<i64, u8> {
        let mut g = Graph::new();
        for i in 0..n { g.add_node(i); }
        for _ in 0..m.below(2 * n as u64) { g.add_both(m.range(0, n - 1), m.range(0, n - 1), m.below(3) as u8); }
        g
    }

    #[test]
    fn neighbours_links_and_a_path_of_one_node() {
        let mut g: Graph<&str, u8> = Graph::new();
        g.add_both("a", "b", 1);
        g.add_link("b", "c", 2);
        g.add_node("z");
        assert_eq!(g.neighbours(&"b"), vec![("a", 1), ("c", 2)]);
        assert!(g.has_link(&"b", &"c") && !g.has_link(&"c", &"b"));
        assert_eq!(g.path(&"a", &"c"), Some(vec!["a", "b", "c"]));
        assert_eq!(g.path(&"c", &"a"), None, "links go one way");
        assert_eq!(g.path(&"a", &"a"), Some(vec!["a"]));
        assert_eq!(g.path(&"a", &"z"), None);
        assert_eq!(g.path(&"a", &"nowhere"), None);
        assert_eq!(g.nodes().len(), 4);
        assert_eq!(g.links().len(), 3);
    }

    /// Mirrors Ideas.Layer0.Structure.steps_triangle, steps_zero_iff.
    #[test]
    fn distance_in_steps_obeys_the_triangle_law_and_is_nought_only_from_a_node_to_itself() {
        let mut m = Maker::new(51);
        for _ in 0..300 {
            let n = 2 + m.range(0, 8);
            let g = make(&mut m, n);
            for a in 0..n { for b in 0..n {
                assert_eq!(g.steps(&a, &b) == Some(0), a == b);
                for c in 0..n {
                    if let (Some(x), Some(y)) = (g.steps(&a, &b), g.steps(&b, &c)) { assert!(g.steps(&a, &c).unwrap() <= x + y, "triangle"); }
                }
            } }
        }
    }

    #[test]
    fn a_path_found_has_the_distance_in_steps_and_every_step_is_a_link() {
        let mut m = Maker::new(52);
        for _ in 0..300 {
            let n = 2 + m.range(0, 8);
            let g = make(&mut m, n);
            for a in 0..n { for b in 0..n {
                match g.path(&a, &b) {
                    Some(p) => {
                        assert_eq!((p[0], *p.last().unwrap()), (a, b));
                        assert_eq!(g.steps(&a, &b), Some(p.len() as u32 - 1));
                        assert!(p.windows(2).all(|w| g.has_link(&w[0], &w[1])));
                    }
                    None => assert!(g.distance(&a, &b).is_none()),
                }
            } }
        }
    }

    #[test]
    fn chains_find_the_reverse_the_lines_and_the_steps_to_the_end() {
        // five nodes in a row, label 0 runs forward and label 1 backward
        let fwd: Vec<Option<u32>> = vec![Some(1), Some(2), Some(3), Some(4), None];
        let back: Vec<Option<u32>> = vec![None, Some(0), Some(1), Some(2), Some(3)];
        let c = chains(&[fwd.clone(), back]).unwrap();
        assert_eq!(c.reverse, vec![Some(1), Some(0)]);
        assert_eq!(c.lines, vec![Chain { label: 0, nodes: vec![0, 1, 2, 3, 4] }], "one line for the pair");
        assert_eq!(c.distance_to_end[0], vec![4, 3, 2, 1, 0]);
        assert_eq!(c.distance_to_end[1], vec![0, 1, 2, 3, 4]);
        // two nodes leading to one is no chain; a loop is no chain
        assert_eq!(chains(&[vec![Some(2), Some(2), None]]), Err(ChainError::TwoBefore(0)));
        assert_eq!(chains(&[vec![Some(1), Some(0)]]), Err(ChainError::Closed(0)));
        assert_eq!(chains(&[fwd]).unwrap().reverse, vec![None]);
    }
}
