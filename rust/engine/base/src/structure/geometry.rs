//! STRUCTURE, FORM THREE: GEOMETRY IN THE PLANE (plan 79.30; SYSTEM_SPEC 11 "Layer 0 holds six things" item 4). A point, distance, direction, a region with "inside" and "between".
//! Interface: see the module doc of `structure`. Every comparison is done in `Num`, so on whole coordinates inside, between and quadrance are exact; only the straight-line distance goes through a root.
use super::Metric;
use crate::number::{Num, NumError};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Point { pub x: Num, pub y: Num }

impl Point {
    pub fn new(x: Num, y: Num) -> Point { Point { x, y } }
    /// A point of whole coordinates.
    pub fn whole(x: i64, y: i64) -> Point { Point { x: Num::whole(x), y: Num::whole(y) } }
}

/// The plane: its distance is the straight line.
#[derive(Debug, Clone, Copy, Default)]
pub struct Plane;

impl Plane {
    /// The distance times itself: exact on whole coordinates, and the one the comparisons below use.
    pub fn quadrance(a: &Point, b: &Point) -> Result<Num, NumError> {
        let (dx, dy) = (a.x.sub(&b.x)?, a.y.sub(&b.y)?);
        dx.mul(&dx)?.add(&dy.mul(&dy)?)
    }
}

impl Metric for Plane {
    type Point = Point;
    fn distance(&self, a: &Point, b: &Point) -> Option<Num> { Plane::quadrance(a, b).and_then(|q| q.sqrt()).ok() }
}

/// A heading: the run from one point to another (not scaled).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Direction { pub dx: Num, pub dy: Num }

/// The heading from one point to another; None when they are one point.
pub fn direction(from: &Point, to: &Point) -> Option<Direction> {
    let (dx, dy) = (to.x.sub(&from.x).ok()?, to.y.sub(&from.y).ok()?);
    if dx == Num::whole(0) && dy == Num::whole(0) { None } else { Some(Direction { dx, dy }) }
}

impl Direction {
    /// Whether two headings point the same way (they run along one line, and not opposite ways).
    pub fn same_heading(&self, o: &Direction) -> Result<bool, NumError> {
        let cross = self.dx.mul(&o.dy)?.sub(&self.dy.mul(&o.dx)?)?;
        let dot = self.dx.mul(&o.dx)?.add(&self.dy.mul(&o.dy)?)?;
        Ok(cross == Num::whole(0) && dot > Num::whole(0))
    }
    /// The angle from the x axis, in radians, between -pi and pi.
    pub fn angle(&self) -> f64 { self.dy.as_f64().atan2(self.dx.as_f64()) }
}

/// Whether the point lies on the straight run from `a` to `b`, the ends included.
pub fn between(a: &Point, b: &Point, p: &Point) -> Result<bool, NumError> {
    let cross = b.x.sub(&a.x)?.mul(&p.y.sub(&a.y)?)?.sub(&b.y.sub(&a.y)?.mul(&p.x.sub(&a.x)?)?)?;
    if cross != Num::whole(0) { return Ok(false); }
    let within = |lo: &Num, hi: &Num, v: &Num| (*lo).min(*hi) <= *v && *v <= (*lo).max(*hi);
    Ok(within(&a.x, &b.x, &p.x) && within(&a.y, &b.y, &p.y))
}

/// A region of the plane. The edge is inside.
#[derive(Debug, Clone, PartialEq)]
pub enum Region {
    /// The points where `a x + b y + c >= 0`.
    HalfPlane { a: Num, b: Num, c: Num },
    /// The points between two corners, edges in (the corners in any order).
    Box { lo: Point, hi: Point },
    /// The points no farther than the radius from the centre.
    Disc { centre: Point, radius: Num },
    /// The points inside a many-sided figure, given by its corners in order.
    Polygon(Vec<Point>),
    /// The points of any of the regions.
    Union(Vec<Region>),
    /// The points of every one of the regions.
    Meet(Vec<Region>),
}

impl Region {
    pub fn union(self, o: Region) -> Region { Region::Union(vec![self, o]) }
    pub fn meet(self, o: Region) -> Region { Region::Meet(vec![self, o]) }

    pub fn inside(&self, p: &Point) -> Result<bool, NumError> {
        match self {
            Region::HalfPlane { a, b, c } => Ok(a.mul(&p.x)?.add(&b.mul(&p.y)?)?.add(c)? >= Num::whole(0)),
            Region::Box { lo, hi } => Ok(lo.x.min(hi.x) <= p.x && p.x <= lo.x.max(hi.x) && lo.y.min(hi.y) <= p.y && p.y <= lo.y.max(hi.y)),
            Region::Disc { centre, radius } => {
                if *radius < Num::whole(0) { return Ok(false); }
                Ok(Plane::quadrance(centre, p)? <= radius.mul(radius)?)
            }
            Region::Polygon(corners) => polygon_inside(corners, p),
            Region::Union(rs) => { for r in rs { if r.inside(p)? { return Ok(true); } } Ok(false) }
            Region::Meet(rs) => { for r in rs { if !r.inside(p)? { return Ok(false); } } Ok(true) }
        }
    }
}

/// The edge is inside; otherwise a ray to the right crosses the sides an odd number of times.
fn polygon_inside(corners: &[Point], p: &Point) -> Result<bool, NumError> {
    let n = corners.len();
    if n == 0 { return Ok(false); }
    for i in 0..n { if between(&corners[i], &corners[(i + 1) % n], p)? { return Ok(true); } }
    let mut inside = false;
    for i in 0..n {
        let (a, b) = (&corners[i], &corners[(i + 1) % n]);
        if (a.y > p.y) != (b.y > p.y) {
            let dy = b.y.sub(&a.y)?;
            let (lhs, rhs) = (p.x.sub(&a.x)?.mul(&dy)?, b.x.sub(&a.x)?.mul(&p.y.sub(&a.y)?)?);
            // the point is left of the side's crossing of its height when (p.x - a.x) < (b.x - a.x)(p.y - a.y) / dy
            if (dy > Num::whole(0) && lhs < rhs) || (dy < Num::whole(0) && lhs > rhs) { inside = !inside; }
        }
    }
    Ok(inside)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::maker::Maker;

    fn pt(m: &mut Maker) -> Point { Point::whole(m.range(-12, 12), m.range(-12, 12)) }

    #[test]
    fn the_plane_distance_is_exact_where_it_can_be() {
        let d = Plane.distance(&Point::whole(0, 0), &Point::whole(3, 4)).unwrap();
        assert_eq!(d.as_whole(), Some(5));
        assert_eq!(Plane::quadrance(&Point::whole(1, 1), &Point::whole(2, 3)).unwrap(), Num::whole(5));
        let half = Point::new(Num::real(0.5).unwrap(), Num::whole(0));
        assert_eq!(Plane.distance(&Point::whole(0, 0), &half).unwrap(), Num::real(0.5).unwrap());
        assert!(Plane.distance(&Point::whole(i64::MAX, 0), &Point::whole(i64::MIN, 0)).is_none(), "an overflow is no distance");
    }

    /// Mirrors Ideas.Layer0.Structure.plane_triangle, plane_dist_eq_zero.
    #[test]
    fn plane_distance_obeys_the_triangle_law_and_is_nought_only_from_a_point_to_itself() {
        let mut m = Maker::new(54);
        for _ in 0..3000 {
            let (a, b, c) = (pt(&mut m), pt(&mut m), pt(&mut m));
            let d = |x: &Point, y: &Point| Plane.distance(x, y).unwrap().as_f64();
            assert!(d(&a, &c) <= d(&a, &b) + d(&b, &c) + 1e-9, "triangle (a real root: a rounding margin)");
            assert_eq!(d(&a, &b) == 0.0, a == b);
            assert_eq!(d(&a, &b), d(&b, &a));
            // the exact form, no root: sqrt(q(a,c)) <= sqrt(q(a,b)) + sqrt(q(b,c)) with both sides multiplied out, whole numbers
            let q = |x: &Point, y: &Point| Plane::quadrance(x, y).unwrap();
            let (ab, bc, ac) = (q(&a, &b).as_whole().unwrap(), q(&b, &c).as_whole().unwrap(), q(&a, &c).as_whole().unwrap());
            let lhs = ac - ab - bc;                       // needs 2 sqrt(ab bc) >= lhs
            assert!(lhs <= 0 || lhs * lhs <= 4 * ab * bc);
        }
    }

    #[test]
    fn regions_say_inside_with_the_edge_in() {
        let half = Region::HalfPlane { a: Num::whole(1), b: Num::whole(0), c: Num::whole(-2) };             // x >= 2
        assert!(half.inside(&Point::whole(2, 9)).unwrap() && half.inside(&Point::whole(3, 0)).unwrap() && !half.inside(&Point::whole(1, 0)).unwrap());
        let bx = Region::Box { lo: Point::whole(4, 4), hi: Point::whole(0, 0) };                             // corners in any order
        assert!(bx.inside(&Point::whole(4, 0)).unwrap() && bx.inside(&Point::whole(2, 2)).unwrap() && !bx.inside(&Point::whole(5, 2)).unwrap());
        let disc = Region::Disc { centre: Point::whole(0, 0), radius: Num::whole(5) };
        assert!(disc.inside(&Point::whole(3, 4)).unwrap() && !disc.inside(&Point::whole(4, 4)).unwrap());
        let poly = Region::Polygon(vec![Point::whole(0, 0), Point::whole(6, 0), Point::whole(6, 6), Point::whole(3, 2), Point::whole(0, 6)]);   // a notch from the top
        assert!(poly.inside(&Point::whole(3, 1)).unwrap() && poly.inside(&Point::whole(6, 3)).unwrap() && poly.inside(&Point::whole(0, 0)).unwrap());
        assert!(!poly.inside(&Point::whole(3, 4)).unwrap(), "in the notch");
        assert!(!poly.inside(&Point::whole(7, 3)).unwrap() && !poly.inside(&Point::whole(-1, 1)).unwrap());
        assert!(!Region::Polygon(vec![]).inside(&Point::whole(0, 0)).unwrap());
    }

    /// Union and meet of two regions are the "or" and the "and" of "inside" (Ideas.Layer0.Structure.inside_union, inside_meet are the same statement over any predicate).
    #[test]
    fn the_union_and_the_meet_of_two_regions_are_or_and_and_of_inside() {
        let mut m = Maker::new(55);
        for _ in 0..500 {
            let r1 = Region::Disc { centre: pt(&mut m), radius: Num::whole(m.range(0, 8)) };
            let r2 = Region::Box { lo: pt(&mut m), hi: pt(&mut m) };
            let (u, t) = (r1.clone().union(r2.clone()), r1.clone().meet(r2.clone()));
            for _ in 0..20 {
                let p = pt(&mut m);
                let (i1, i2) = (r1.inside(&p).unwrap(), r2.inside(&p).unwrap());
                assert_eq!(u.inside(&p).unwrap(), i1 || i2);
                assert_eq!(t.inside(&p).unwrap(), i1 && i2);
            }
        }
    }

    #[test]
    fn a_polygon_that_is_a_box_agrees_with_the_box() {
        let mut m = Maker::new(56);
        for _ in 0..300 {
            let (a, b) = (pt(&mut m), pt(&mut m));
            let (lo, hi) = (Point::new(a.x.min(b.x), a.y.min(b.y)), Point::new(a.x.max(b.x), a.y.max(b.y)));
            if lo.x == hi.x || lo.y == hi.y { continue; }
            let poly = Region::Polygon(vec![lo, Point::new(hi.x, lo.y), hi, Point::new(lo.x, hi.y)]);
            let bx = Region::Box { lo, hi };
            for _ in 0..20 { let p = pt(&mut m); assert_eq!(poly.inside(&p).unwrap(), bx.inside(&p).unwrap(), "{lo:?} {hi:?} {p:?}"); }
        }
    }

    #[test]
    fn between_and_direction() {
        let (a, b) = (Point::whole(0, 0), Point::whole(4, 2));
        assert!(between(&a, &b, &Point::whole(2, 1)).unwrap() && between(&a, &b, &a).unwrap() && between(&a, &b, &b).unwrap());
        assert!(!between(&a, &b, &Point::whole(6, 3)).unwrap() && !between(&a, &b, &Point::whole(2, 2)).unwrap());
        let (h_ab, h_ac, h_ba) = (direction(&a, &b).unwrap(), direction(&a, &Point::whole(2, 1)).unwrap(), direction(&b, &a).unwrap());
        assert!(h_ab.same_heading(&h_ac).unwrap() && !h_ab.same_heading(&h_ba).unwrap());
        assert!(direction(&a, &a).is_none());
        assert!((h_ab.angle() - (0.5f64).atan()).abs() < 1e-12);
    }
}
