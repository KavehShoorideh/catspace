//! LAYER ONE: the medium of movement with every unit ignored (the Python sheet's `Medium`). Read by the caller from the game's description and handed in as three tables:
//! the hops (kind, side, from, to, take-only), the places passed over by a hop, and the (kind, side) pairs that have take-only hops. Sides are the game's own role names.
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::rc::Rc;

/// A table keyed by three words, looked up by borrowed text (no key is built to ask).
struct Nest<V>(HashMap<String, HashMap<String, HashMap<String, V>>>);

impl<V> Nest<V> {
    fn new() -> Nest<V> { Nest(HashMap::new()) }
    fn get(&self, a: &str, b: &str, c: &str) -> Option<&V> { self.0.get(a)?.get(b)?.get(c) }
    fn slot(&mut self, a: &str, b: &str) -> &mut HashMap<String, V> { self.0.entry(a.to_string()).or_default().entry(b.to_string()).or_default() }
}

pub struct Medium {
    hop: Nest<BTreeMap<String, bool>>,
    over_of: Nest<HashMap<String, Vec<String>>>,
    takes: HashSet<(String, String)>,
    into: Nest<Vec<(String, bool)>>,
    dist: RefCell<Nest<Rc<HashMap<String, u32>>>>,
    empty: BTreeMap<String, bool>,
}

impl Medium {
    pub fn new(hops: Vec<(String, String, String, String, bool)>, overs: Vec<(String, String, String, String, String)>, takes: Vec<(String, String)>) -> Medium {
        let mut hop: Nest<BTreeMap<String, bool>> = Nest::new();
        let mut into: Nest<Vec<(String, bool)>> = Nest::new();
        for (k, s, f, t, take) in hops {
            hop.slot(&k, &s).entry(f.clone()).or_default().insert(t.clone(), take);
            into.slot(&k, &s).entry(t).or_default().push((f, take));
        }
        let mut over_of: Nest<HashMap<String, Vec<String>>> = Nest::new();
        for (k, s, f, t, o) in overs { over_of.slot(&k, &s).entry(f).or_default().entry(t).or_default().push(o); }
        Medium { hop, over_of, takes: takes.into_iter().collect(), into, dist: RefCell::new(Nest::new()), empty: BTreeMap::new() }
    }

    /// The kind has take-only hops on this side.
    pub fn takes(&self, kind: &str, side: &str) -> bool { self.takes.contains(&(kind.to_string(), side.to_string())) }

    /// The places one move reaches, each with whether the hop is take-only.
    pub fn hops(&self, kind: &str, side: &str, from: &str) -> &BTreeMap<String, bool> { self.hop.get(kind, side, from).unwrap_or(&self.empty) }

    /// The places passed over by one hop along a line, sorted.
    pub fn over(&self, kind: &str, side: &str, from: &str, to: &str) -> Vec<String> {
        let mut v = self.over_of.get(kind, side, from).and_then(|m| m.get(to)).cloned().unwrap_or_default();
        v.sort();
        v
    }

    /// The least number of hops from `from` to every place it reaches (breadth first).
    pub fn distances(&self, kind: &str, side: &str, from: &str) -> Rc<HashMap<String, u32>> {
        if let Some(d) = self.dist.borrow().get(kind, side, from) { return d.clone(); }
        let mut dist: HashMap<String, u32> = HashMap::new();
        dist.insert(from.to_string(), 0);
        let mut layer = vec![from.to_string()];
        while !layer.is_empty() {
            let mut next = vec![];
            for p in &layer {
                for to in self.hops(kind, side, p).keys() {
                    if !dist.contains_key(to) { dist.insert(to.clone(), dist[p] + 1); next.push(to.clone()); }
                }
            }
            layer = next;
        }
        let d = Rc::new(dist);
        self.dist.borrow_mut().slot(kind, side).insert(from.to_string(), d.clone());
        d
    }

    pub fn distance(&self, kind: &str, side: &str, from: &str, to: &str) -> Option<u32> { self.distances(kind, side, from).get(to).copied() }

    /// Every shortest route (the places landed on, in order) from `from` to `to`, at most `cap`.
    pub fn routes(&self, kind: &str, side: &str, from: &str, to: &str, cap: usize) -> Vec<Vec<String>> {
        let mut out: Vec<Vec<String>> = vec![];
        let Some(n) = self.distance(kind, side, from, to) else { return out };
        if n == 0 { return out; }
        let mut path: Vec<String> = vec![];
        self.walk(kind, side, from, to, n, &mut path, &mut out, cap);
        out
    }

    #[allow(clippy::too_many_arguments)]
    fn walk(&self, kind: &str, side: &str, p: &str, to: &str, left: u32, path: &mut Vec<String>, out: &mut Vec<Vec<String>>, cap: usize) {
        if out.len() >= cap { return; }
        if left == 0 {
            if p == to { out.push(path.clone()); }
            return;
        }
        for y in self.hops(kind, side, p).keys() {
            if self.distance(kind, side, y, to) == Some(left - 1) {
                path.push(y.clone());
                self.walk(kind, side, y, to, left - 1, path, out, cap);
                path.pop();
            }
        }
    }

    /// The places from which a unit of that kind bears on `x` (by the take-only hops where the kind has them, else by its hops).
    pub fn strike_places(&self, kind: &str, side: &str, x: &str) -> BTreeSet<String> {
        let take = self.takes(kind, side);
        self.into.get(kind, side, x).map(|v| v.iter().filter(|(_, t)| if take { *t } else { true }).map(|(p, _)| p.clone()).collect()).unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A line of places "p0".."p3" where a unit "step" goes one place along, and a unit "leap" goes one or two (take-only for the two), passing over the place between.
    fn line() -> Medium {
        let s = |x: &str| x.to_string();
        let mut hops = vec![];
        let mut overs = vec![];
        for i in 0..3 { hops.push((s("step"), s("a"), format!("p{i}"), format!("p{}", i + 1), false)); }
        for i in 0..4 {
            if i + 1 < 4 { hops.push((s("leap"), s("a"), format!("p{i}"), format!("p{}", i + 1), false)); }
            if i + 2 < 4 {
                hops.push((s("leap"), s("a"), format!("p{i}"), format!("p{}", i + 2), true));
                overs.push((s("leap"), s("a"), format!("p{i}"), format!("p{}", i + 2), format!("p{}", i + 1)));
            }
        }
        Medium::new(hops, overs, vec![(s("leap"), s("a"))])
    }

    #[test]
    fn distances_are_hops_and_routes_are_the_shortest() {
        let m = line();
        assert_eq!(m.distance("step", "a", "p0", "p3"), Some(3));
        assert_eq!(m.distance("step", "a", "p3", "p0"), None);
        assert_eq!(m.distance("leap", "a", "p0", "p3"), Some(2));
        assert_eq!(m.routes("leap", "a", "p0", "p3", 200), vec![vec!["p1".to_string(), "p3".to_string()], vec!["p2".to_string(), "p3".to_string()]]);
        assert_eq!(m.routes("leap", "a", "p0", "p3", 1).len(), 1);
        assert!(m.routes("step", "a", "p0", "p0", 200).is_empty());
    }

    #[test]
    fn a_kind_with_take_only_hops_strikes_by_those_alone() {
        let m = line();
        assert!(m.takes("leap", "a") && !m.takes("step", "a"));
        assert_eq!(m.strike_places("step", "a", "p2").into_iter().collect::<Vec<_>>(), vec!["p1".to_string()]);
        assert_eq!(m.strike_places("leap", "a", "p2").into_iter().collect::<Vec<_>>(), vec!["p0".to_string()]);
        assert_eq!(m.over("leap", "a", "p0", "p2"), vec!["p1".to_string()]);
        assert!(m.over("step", "a", "p0", "p1").is_empty());
    }
}
