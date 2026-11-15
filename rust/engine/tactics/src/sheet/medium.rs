//! LAYER ONE: the medium of movement with every unit ignored (the Python sheet's `Medium`). Read by the caller from the game's description and handed in as three tables:
//! the hops (kind, side, from, to, take-only), the places passed over by a hop, and the (kind, side) pairs that have take-only hops. Sides are the game's own role names.
//! Everything is held by number (a place, a kind, a role): the hops of a (kind, role) from a place are one slice, the distances of a (kind, role) from a place are one array
//! computed on first use and kept for every position, so is the list of shortest routes between two places.
use super::{FxMap, Kind, Names, Place};
use std::cell::RefCell;
use std::rc::Rc;

pub struct Medium {
    np: usize,
    nr: usize,
    hop: Vec<Vec<(Place, bool)>>,
    over_of: FxMap<(u32, Place, Place), Vec<Place>>,
    takes: Vec<bool>,
    into: Vec<Vec<(Place, bool)>>,
    dist: RefCell<Vec<Option<Rc<Vec<u8>>>>>,
    routes: RefCell<FxMap<(u32, Place, Place, usize), Rc<Vec<Vec<Place>>>>>,
    strike: RefCell<FxMap<(u32, Place), Rc<Vec<Place>>>>,
}

const FAR: u8 = u8::MAX;

impl Medium {
    pub fn new(names: &Names, hops: &[(String, String, String, String, bool)], overs: &[(String, String, String, String, String)], takes: &[(String, String)]) -> Result<Medium, String> {
        let (np, nk, nr) = (names.places.len(), names.kinds.len(), names.roles.len());
        let mut m = Medium { np, nr, hop: vec![vec![]; nk * nr * np], over_of: FxMap::default(), takes: vec![false; nk * nr], into: vec![vec![]; nk * nr * np],
                             dist: RefCell::new(vec![None; nk * nr * np]), routes: RefCell::new(FxMap::default()), strike: RefCell::new(FxMap::default()) };
        for (k, s, f, t, take) in hops {
            let (kr, f, t) = (m.kr(names.kind(k)?, names.role(s)?), names.place(f)?, names.place(t)?);
            let (hi, ii) = (kr * np + f as usize, kr * np + t as usize);
            m.hop[hi].push((t, *take));
            m.into[ii].push((f, *take));
        }
        for row in m.hop.iter_mut() { row.sort_unstable(); row.dedup_by_key(|x| x.0); }
        for row in m.into.iter_mut() { row.sort_unstable(); }
        for (k, s, f, t, o) in overs {
            let (kr, f, t, o) = (m.kr(names.kind(k)?, names.role(s)?), names.place(f)?, names.place(t)?, names.place(o)?);
            m.over_of.entry((kr as u32, f, t)).or_default().push(o);
        }
        for v in m.over_of.values_mut() { v.sort_unstable(); }
        for (k, s) in takes { let kr = m.kr(names.kind(k)?, names.role(s)?); m.takes[kr] = true; }
        Ok(m)
    }

    fn kr(&self, kind: Kind, role: usize) -> usize { kind as usize * self.nr + role }

    /// The kind has take-only hops on this side.
    pub fn takes(&self, kind: Kind, role: usize) -> bool { self.takes[self.kr(kind, role)] }

    /// The places one move reaches, each with whether the hop is take-only, ascending.
    pub fn hops(&self, kind: Kind, role: usize, from: Place) -> &[(Place, bool)] { &self.hop[self.kr(kind, role) * self.np + from as usize] }

    /// Whether the hop from `from` to `to` is take-only (None: no such hop).
    pub fn hop_take(&self, kind: Kind, role: usize, from: Place, to: Place) -> Option<bool> {
        let row = self.hops(kind, role, from);
        row.binary_search_by_key(&to, |x| x.0).ok().map(|i| row[i].1)
    }

    /// The places passed over by one hop along a line, ascending.
    pub fn over(&self, kind: Kind, role: usize, from: Place, to: Place) -> &[Place] {
        self.over_of.get(&(self.kr(kind, role) as u32, from, to)).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// The least number of hops from `from` to every place it reaches (breadth first); `FAR` for a place it does not reach.
    fn distances(&self, kind: Kind, role: usize, from: Place) -> Rc<Vec<u8>> {
        let slot = self.kr(kind, role) * self.np + from as usize;
        if let Some(d) = &self.dist.borrow()[slot] { return d.clone(); }
        let mut dist = vec![FAR; self.np];
        dist[from as usize] = 0;
        let mut layer = vec![from];
        while !layer.is_empty() {
            let mut next = vec![];
            for p in &layer {
                for (to, _) in self.hops(kind, role, *p) {
                    if dist[*to as usize] == FAR { dist[*to as usize] = dist[*p as usize] + 1; next.push(*to); }
                }
            }
            layer = next;
        }
        let d = Rc::new(dist);
        self.dist.borrow_mut()[slot] = Some(d.clone());
        d
    }

    pub fn distance(&self, kind: Kind, role: usize, from: Place, to: Place) -> Option<u32> {
        let d = self.distances(kind, role, from)[to as usize];
        if d == FAR { None } else { Some(d as u32) }
    }

    /// Every shortest route (the places landed on, in order) from `from` to `to`, at most `cap`; kept for the next position.
    pub fn routes(&self, kind: Kind, role: usize, from: Place, to: Place, cap: usize) -> Rc<Vec<Vec<Place>>> {
        let key = (self.kr(kind, role) as u32, from, to, cap);
        if let Some(r) = self.routes.borrow().get(&key) { return r.clone(); }
        let mut out: Vec<Vec<Place>> = vec![];
        if let Some(n) = self.distance(kind, role, from, to) {
            if n > 0 {
                let mut path: Vec<Place> = vec![];
                self.walk(kind, role, from, to, n, &mut path, &mut out, cap);
            }
        }
        let out = Rc::new(out);
        self.routes.borrow_mut().insert(key, out.clone());
        out
    }

    #[allow(clippy::too_many_arguments)]
    fn walk(&self, kind: Kind, role: usize, p: Place, to: Place, left: u32, path: &mut Vec<Place>, out: &mut Vec<Vec<Place>>, cap: usize) {
        if out.len() >= cap { return; }
        if left == 0 {
            if p == to { out.push(path.clone()); }
            return;
        }
        for (y, _) in self.hops(kind, role, p) {
            if self.distance(kind, role, *y, to) == Some(left - 1) {
                path.push(*y);
                self.walk(kind, role, *y, to, left - 1, path, out, cap);
                path.pop();
            }
        }
    }

    /// The places from which a unit of that kind bears on `x` (by the take-only hops where the kind has them, else by its hops), ascending.
    pub fn strike_places(&self, kind: Kind, role: usize, x: Place) -> Rc<Vec<Place>> {
        let key = (self.kr(kind, role) as u32, x);
        if let Some(r) = self.strike.borrow().get(&key) { return r.clone(); }
        let take = self.takes(kind, role);
        let v: Vec<Place> = self.into[self.kr(kind, role) * self.np + x as usize].iter().filter(|(_, t)| if take { *t } else { true }).map(|(p, _)| *p).collect();
        let v = Rc::new(v);
        self.strike.borrow_mut().insert(key, v.clone());
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names() -> Names {
        Names::new((0..4).map(|i| format!("p{i}")).collect(), vec!["leap".to_string(), "step".to_string()], vec!["a".to_string(), "b".to_string()])
    }

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
        Medium::new(&names(), &hops, &overs, &[(s("leap"), s("a"))]).unwrap()
    }

    #[test]
    fn distances_are_hops_and_routes_are_the_shortest() {
        let m = line();
        let (leap, step) = (0u8, 1u8);
        assert_eq!(m.distance(step, 0, 0, 3), Some(3));
        assert_eq!(m.distance(step, 0, 3, 0), None);
        assert_eq!(m.distance(leap, 0, 0, 3), Some(2));
        assert_eq!(*m.routes(leap, 0, 0, 3, 200), vec![vec![1, 3], vec![2, 3]]);
        assert_eq!(m.routes(leap, 0, 0, 3, 1).len(), 1);
        assert!(m.routes(step, 0, 0, 0, 200).is_empty());
    }

    #[test]
    fn a_kind_with_take_only_hops_strikes_by_those_alone() {
        let m = line();
        let (leap, step) = (0u8, 1u8);
        assert!(m.takes(leap, 0) && !m.takes(step, 0));
        assert_eq!(*m.strike_places(step, 0, 2), vec![1]);
        assert_eq!(*m.strike_places(leap, 0, 2), vec![0]);
        assert_eq!(m.over(leap, 0, 0, 2), &[1]);
        assert!(m.over(step, 0, 0, 1).is_empty());
        assert_eq!(m.hop_take(leap, 0, 0, 2), Some(true));
        assert_eq!(m.hop_take(leap, 0, 0, 3), None);
    }

    #[test]
    fn place_sets_run_in_the_order_of_the_numbers() {
        let mut s = PlaceSet::new(130);
        for p in [129u16, 3, 64, 0] { s.insert(p); }
        assert_eq!(s.iter().collect::<Vec<_>>(), vec![0, 3, 64, 129]);
        assert!(s.contains(64) && !s.contains(65) && s.len() == 4);
    }
}

#[cfg(test)]
use super::PlaceSet;
