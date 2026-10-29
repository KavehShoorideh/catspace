//! PLAN 76.5 THE VALUATION AS ONE DEFINITION (docs/ANALYST_PLAN.md 76.5): the worth of each unit kind, the unit of a gain and the royal unit -- the lemma
//! store's VALUATION entry the GOAL entries count in (lemma_store.py `goal_kinds().valuation`: `material`, HAND, unless a run points them at another entry) --
//! held ONCE per process here and read by every Rust site that weighs a unit: rules.rs (the static exchange, the threats), vocab.rs (worth, maxworth, the
//! materials, the balance), words.rs, regress.rs, gdl_chess.rs (the 'exchange from' law), pos.rs (the principal lines), lib.rs, plan.rs, threat.rs,
//! waiting.rs, reasoning.rs. NO TABLE IS WRITTEN HERE: `set_valuation(worths, unit, royal)` is called from Python with the store's entry
//! (lemma_store.install_valuation), and THE STORE IS LOADED ON FIRST USE -- a worth read before that call imports the lemma store and calls its
//! `install_valuation()`; a read never sees a default, and a store that does not load (or does not set the table) raises (a Rust panic: PanicException in
//! Python). The quantities the Rust sites wrote as literals are read off the table: the pawn's worth (an en-passant victim, the unit a promotion replaces),
//! the promotion's stake (the most a promotion may place, less the pawn), "a unit worth more than a pawn" (the old `>= 3` / `< 3`), the least such worth
//! (a threat worth a piece), the unit of a gain (the old `.max(1)`), and the royal's place in an order of worths (last as a capturer: a key, not a number).
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use shakmaty::Role;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, AtomicU8, Ordering};

static SET: AtomicBool = AtomicBool::new(false);
static LOADING: AtomicBool = AtomicBool::new(false);
static WORTH: [AtomicI64; 6] = [const { AtomicI64::new(0) }; 6];         // by the kernel's kind, pawn .. king (shakmaty's Role as u8, less one)
static UNIT: AtomicI64 = AtomicI64::new(0);
static ROYAL: AtomicU8 = AtomicU8::new(0);
static INSTALLS: AtomicU64 = AtomicU64::new(0);                           // how many times a table was set (the test reads it)
const LETTERS: [char; 6] = ['p', 'n', 'b', 'r', 'q', 'k'];               // the kernel's unit letters, in Role order (the store's keys)

#[inline]
fn ensure() { if !SET.load(Ordering::Acquire) { load_from_store(); } }

/// THE STORE ON FIRST USE: `lemma_store.install_valuation()` (which calls `set_valuation` with the entry the GOAL entries count in).
#[cold]
fn load_from_store() {
    if LOADING.swap(true, Ordering::AcqRel) {
        panic!("the valuation: a worth was read while the lemma store's valuation was being installed (install_valuation must read no Rust worth)");
    }
    let r = Python::with_gil(|py| -> PyResult<()> {
        py.import("catspace.incubator.analyst.discovery.lemma_store")?.call_method0("install_valuation")?;
        Ok(())
    });
    LOADING.store(false, Ordering::Release);
    if let Err(e) = r { panic!("the valuation could not be loaded from the lemma store (lemma_store.install_valuation): {e}"); }
    if !SET.load(Ordering::Acquire) { panic!("lemma_store.install_valuation returned without setting the valuation (analyst_core.set_valuation)"); }
}

/// The worth of a unit kind (the store's; the royal's as the store says: 0 in both entries -- its loss is the terminal, not a count).
#[inline]
pub(crate) fn worth(r: Role) -> i64 { ensure(); WORTH[r as usize - 1].load(Ordering::Relaxed) }
/// The worth by python-chess's piece type (pawn 1 .. king 6: the pyboard mirror's codes).
#[inline]
pub(crate) fn worth_pt(pt: u8) -> i64 { if (1..=6).contains(&pt) { ensure(); WORTH[pt as usize - 1].load(Ordering::Relaxed) } else { 0 } }
/// The unit of a gain (the least gain the store counts: the GOAL entries' need).
#[inline]
pub(crate) fn unit() -> i64 { ensure(); UNIT.load(Ordering::Relaxed) }
/// The royal unit's kind.
#[inline]
pub(crate) fn royal() -> Role { ensure(); Role::ALL[ROYAL.load(Ordering::Relaxed) as usize] }
/// The pawn's worth (the unit that promotes, the kernel's pawn: an en-passant victim, the unit a promotion replaces).
#[inline]
pub(crate) fn pawn() -> i64 { worth(Role::Pawn) }
/// THE PROMOTION'S STAKE: the most a promotion may place (the kernel's promotion kinds: queen, rook, bishop, knight -- the valuation picks among them, as
/// Game.placing_most does) less the pawn it replaces -- the promotion GOAL entry's stake {unit q, less p} over this table (8 for the hand worths).
pub(crate) fn promotion_stake() -> i64 {
    [Role::Queen, Role::Rook, Role::Bishop, Role::Knight].into_iter().map(worth).max().unwrap_or(0) - pawn()
}
/// "A unit worth more than a pawn" (what the literal `>= 3` meant on a worth: a piece, not a pawn).
#[inline]
pub(crate) fn more_than_pawn(w: i64) -> bool { w > pawn() }
/// The least worth of a unit worth more than a pawn (the royal aside): what the literal 3 meant on a GAIN -- a threat worth a piece (3 for the hand worths:
/// the knight and the bishop).
pub(crate) fn piece_floor() -> i64 {
    let r = royal();
    Role::ALL.into_iter().filter(|x| *x != r).map(worth).filter(|w| more_than_pawn(*w)).min().unwrap_or_else(|| pawn() + unit())
}
/// A unit's place in an ORDER of worths (a capturer's, a target's): by worth, the royal above every unit (it captures last -- Lean
/// Ideas.ExchangeLaw.valueAt: "the king counts 100 ... as a capturer it is dearer than anything") -- a key, not a number.
#[inline]
pub(crate) fn order_key(r: Role) -> (bool, i64) { (r == royal(), worth(r)) }

/// set_valuation(worths, unit, royal): THE VALUATION of this process -- `worths` maps each of the kernel's unit letters (p n b r q k, either case) to a whole
/// number >= 0, `unit` the least gain (>= 1), `royal` the royal unit's letter. Called from Python with the lemma store's entry (lemma_store.install_valuation);
/// a table different from the one set before empties the core's worth-dependent caches (the word cache, the passer memos, the spec's regression cache).
/// ValueError on a table that does not name the six kinds, a negative worth, a unit below 1 or an unknown royal.
#[pyfunction]
pub fn set_valuation(worths: HashMap<String, i64>, unit: i64, royal: String) -> PyResult<()> {
    let mut w = [0i64; 6];
    let mut seen = [false; 6];
    for (k, v) in &worths {
        let c = k.to_lowercase();
        let Some(i) = LETTERS.iter().position(|l| c.len() == 1 && c.starts_with(*l)) else {
            return Err(PyValueError::new_err(format!("set_valuation: {k:?} is no unit letter of the kernel (p n b r q k)")));
        };
        if *v < 0 { return Err(PyValueError::new_err(format!("set_valuation: the worth of {k:?} is negative ({v})"))); }
        if seen[i] { return Err(PyValueError::new_err(format!("set_valuation: {k:?} given twice"))); }
        w[i] = *v;
        seen[i] = true;
    }
    if !seen.iter().all(|s| *s) { return Err(PyValueError::new_err("set_valuation: the table must give every unit kind of the kernel (p n b r q k) a worth")); }
    if unit < 1 { return Err(PyValueError::new_err(format!("set_valuation: the unit of a gain must be >= 1 (got {unit})"))); }
    let rl = royal.to_lowercase();
    let Some(ri) = LETTERS.iter().position(|l| rl.len() == 1 && rl.starts_with(*l)) else {
        return Err(PyValueError::new_err(format!("set_valuation: the royal {royal:?} is no unit letter of the kernel")));
    };
    let was = SET.load(Ordering::Acquire);
    let changed = !was || (0..6).any(|i| WORTH[i].load(Ordering::Relaxed) != w[i]) || UNIT.load(Ordering::Relaxed) != unit
        || ROYAL.load(Ordering::Relaxed) as usize != ri;
    for i in 0..6 { WORTH[i].store(w[i], Ordering::Relaxed); }
    UNIT.store(unit, Ordering::Relaxed);
    ROYAL.store(ri as u8, Ordering::Relaxed);
    SET.store(true, Ordering::Release);
    INSTALLS.fetch_add(1, Ordering::Relaxed);
    if was && changed {
        crate::vocab::clear_word_cache();
        let _ = crate::words::passer_cache_clear();
        let _ = crate::words::reach_cache_clear();
        crate::regress::clear_caches();
        crate::gdl_chess::cache_clear_if_loaded();
    }
    Ok(())
}

/// valuation() -> (worths by letter, unit, royal letter): the table the core reads (read through it: a call before `set_valuation` loads the store).
#[pyfunction]
pub fn valuation() -> PyResult<(HashMap<String, i64>, i64, String)> {
    let w: HashMap<String, i64> = Role::ALL.into_iter().map(|r| (LETTERS[r as usize - 1].to_string(), worth(r))).collect();
    Ok((w, unit(), LETTERS[royal() as usize - 1].to_string()))
}

/// valuation_state() -> (whether a table is set, how many times one was set): read WITHOUT loading the store (the test of "the store on first use").
#[pyfunction]
pub fn valuation_state() -> (bool, u64) { (SET.load(Ordering::Acquire), INSTALLS.load(Ordering::Relaxed)) }
