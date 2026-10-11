//! THE PRIMITIVE CLOCK (Kaveh 2026-10-11: "profile the time, see what's taking long"): calls and inclusive time per
//! named primitive on this thread -- the inner planner (their immediate reasoning on the null move or after a candidate),
//! the generic threat, the mate search, the forcing search, the proofs, the capture settle, the attack paths. Nested
//! calls count inside their callers (inclusive), so the numbers are a map of where the time goes, not a partition.
//! `prim_times()` reads and clears.
use std::cell::RefCell;
use std::time::Instant;

thread_local! { static PRIM: RefCell<Vec<(&'static str, u64, f64)>> = const { RefCell::new(Vec::new()) }; }

pub fn timed<T>(name: &'static str, f: impl FnOnce() -> T) -> T {
    let t0 = Instant::now();
    let out = f();
    let ms = t0.elapsed().as_secs_f64() * 1000.0;
    PRIM.with(|v| { let mut v = v.borrow_mut(); if let Some(e) = v.iter_mut().find(|e| e.0 == name) { e.1 += 1; e.2 += ms; } else { v.push((name, 1, ms)); } });
    out
}

/// prim_times() -> [(primitive, calls, ms)] since the last call.
#[pyo3::pyfunction]
pub fn prim_times() -> Vec<(String, u64, f64)> { PRIM.with(|v| std::mem::take(&mut *v.borrow_mut())).into_iter().map(|(n, c, t)| (n.to_string(), c, t)).collect() }
