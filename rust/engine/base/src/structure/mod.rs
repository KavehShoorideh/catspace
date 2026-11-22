//! LAYER 0, MODULE 4: STRUCTURE (plan 79.30; SYSTEM_SPEC 11 "Layer 0 holds six things" item 4). Three pure forms that layer 1 points at, and one interface of distance that two of them give.
//!
//! INTERFACE (callers: layer 1 points its objects at these forms; cs_rep reads its lines through `graph::chains`).
//!   `Metric` (this module): `type Point`; `distance(&self, &Point, &Point) -> Option<Num>` (None: no distance, for example no path). Law: distance is nought exactly from a point to itself, symmetric where the form is, and obeys the triangle law
//!     (lean/ideas/Ideas/Layer0/Structure.lean). Given by `Graph` (whole, in steps) and `Plane` (real).
//!   graph.rs, A GRAPH: `Graph<N: Ord + Clone, L: Ord + Clone>`: `new()`, `add_node(N)`, `add_link(from, to, label)` (one direction; adds the nodes), `add_both(a, b, label)` (a link each way), `nodes() -> Vec<N>`, `links() -> Vec<(N, N, L)>`,
//!     `neighbours(&N) -> Vec<(N, L)>` (in order), `has_link(&N, &N) -> bool`, `path(&N, &N) -> Option<Vec<N>>` (a shortest path, the first in order among shortest ones, both ends in it), `steps(&N, &N) -> Option<u32>` (distance in steps).
//!     THE LIFTED CORE of cs_rep's graph (read cs_rep/graph.rs): `chains(next: &[Vec<Option<u32>>]) -> Result<Chains, ChainError>` takes, for each label, the successor of each node (the same node set for every label), and returns the label that reverses each label,
//!     the maximal chains of two nodes or more (one per label pair), and for each label and node the steps until the chain ends. `ChainError::{TwoAfter(label), TwoBefore(label), Closed(label)}` says which label is not a chain and how.
//!   sequence.rs, A SEQUENCE: `Sequence<T>`: `new(Vec<T>)`, `len()`, `get(usize) -> Option<&T>`, `before(i, j) -> bool`, `after(i, j) -> bool`, `position_of(&T) -> Option<usize>` (first), `items() -> &[T]`, `pairs() -> Vec<(&T, &T)>` (each item with the next),
//!     and for `Sequence<Num>`: `differences() -> Result<Vec<Num>, NumError>` (each item less the one before it).
//!   geometry.rs, GEOMETRY IN THE PLANE: `Point { x: Num, y: Num }`; `Plane` (the plane's `Metric`, distance the straight line, real); `Plane::quadrance(&Point, &Point) -> Result<Num, NumError>` (exact on whole coordinates);
//!     `Direction { dx: Num, dy: Num }`, `direction(from, to) -> Option<Direction>` (None when they are one point), `Direction::same_heading(&Direction) -> Result<bool, NumError>`, `Direction::angle() -> f64`;
//!     `Region::{HalfPlane { a, b, c }, Box { lo, hi }, Disc { centre, radius }, Polygon(Vec<Point>), Union(Vec<Region>), Meet(Vec<Region>)}`, `Region::union(self, Region)`, `Region::meet(self, Region)`;
//!     `Region::inside(&self, &Point) -> Result<bool, NumError>` (the edge is inside: a half plane is `a x + b y + c >= 0`); `between(a, b, p) -> Result<bool, NumError>` (p lies on the straight run from a to b, ends included).
//!   (`ChainError::TwoAfter` is kept for a caller that builds the successor map from rows and finds a node with two successors; `chains` itself takes one successor per node and so can only say `TwoBefore` and `Closed`.)
use crate::number::Num;

pub mod geometry;
pub mod graph;
pub mod sequence;

pub use geometry::{between, direction, Direction, Plane, Point, Region};
pub use graph::{chains, Chain, ChainError, Chains, Graph};
pub use sequence::Sequence;

/// ONE INTERFACE OF DISTANCE that the graph (in steps) and the plane (straight line) both give.
pub trait Metric {
    type Point;
    /// The distance from one point to another; None when there is none (no path, or the number overflowed).
    fn distance(&self, a: &Self::Point, b: &Self::Point) -> Option<Num>;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One body of tests for any `Metric`: nought only from a point to itself, and the triangle law, over the points given (the same function runs on the graph and on the plane).
    fn laws<M: Metric>(m: &M, points: &[M::Point], same: &dyn Fn(&M::Point, &M::Point) -> bool) {
        let slack = Num::real(1e-9).unwrap();
        for a in points { for b in points {
            if let Some(d) = m.distance(a, b) { assert_eq!(d == Num::whole(0), same(a, b)); assert!(d >= Num::whole(0)); }
            for c in points {
                if let (Some(x), Some(y), Some(z)) = (m.distance(a, b), m.distance(b, c), m.distance(a, c)) { assert!(z <= x.add(&y).unwrap().add(&slack).unwrap()); }
            }
        } }
    }

    #[test]
    fn one_interface_of_distance_is_given_by_the_graph_and_by_the_plane() {
        let mut g: Graph<u8, u8> = Graph::new();
        for (a, b) in [(0, 1), (1, 2), (2, 3), (3, 4), (1, 4)] { g.add_both(a, b, 0); }
        g.add_node(9);
        laws(&g, &[0, 1, 2, 3, 4, 9], &|a, b| a == b);
        assert_eq!(g.distance(&0, &3), Some(Num::whole(3)));
        assert_eq!(g.distance(&0, &9), None);
        let pts: Vec<Point> = (-2..=2).flat_map(|x| (-2..=2).map(move |y| Point::whole(x, y))).collect();
        laws(&Plane, &pts, &|a, b| a == b);
        // the plane's distance is a whole number where the root is exact and the graph's is always whole: one type, one order
        assert!(Plane.distance(&Point::whole(0, 0), &Point::whole(1, 1)).unwrap() > g.distance(&0, &1).unwrap());
    }
}
