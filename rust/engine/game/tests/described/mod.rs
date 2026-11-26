//! PLAN 79.57 PART A (SYSTEM_SPEC 11, blocks "A divergence found and named" and "The support filter"; LAYER 2, piece: rules, on layer 1's kept graph; ledger rows 58, 57, 44, 40). THIS TEXT IS THE INTERFACE, written before the body.
//! A test helper, not part of the crate's sources (the compiled placed modules are not on the engine's path and stay off it).
//!
//! `Described<T>`: THE DESCRIPTION ANSWERS WHAT THE KEEPER ASKS. It gives `GraphKeeper` and `Static::position` the interface they take (`Game` and `Placed`) with the units, the places each unit bears on and the places each unit may move to answered by a translated
//! placed module `T` (a module of rules/placed/, made by the translator, read through the test trait `Tr`), and every other question (the moves, the key, why a step is not made, the shut lines) by the kernel, which is only the game's own rules text and notation here.
//!   TAKES    `Described::new(kernel, t)`: the kernel of the game (for the state, the moves, the notation, and the questions the placed module does not hold: `why_not` for the shut lines) and the module made of the game's description.
//!   RETURNS  `Game` (delegated to the kernel, so the states are the kernel's); `Placed` with `cells`, `bearing_from`, `reach_from` from the module's rows.
//!   HOW A STATE IS ASKED. A question about a state is one call of the module on the state's facts for the relations of the placed questions (what stands where; what every unit bears on; and, for each side, what every unit of that side may move to when that side is to move and no place is passed over: the
//!   state of the questions the placed corpus uses). The answer rows of one state are kept under the state's key (a state asked twice is read once); a question of the keeper for one unit is a look-up in them. `asked()` counts the questions of units.
//!   HOW PLACES AND KINDS ARE NUMBERED (the adapter's only knowledge of the module): a place is `file * 8 + rank` (the files and ranks of the description's sorts in the order of their texts, which is the kernel's order of `place_names`), a kind is its index in the kernel's `kind_names`, a side its index in the kernel's `roles`.
//!   WHO CALLS IT: tests/translated_graph.rs (the gate of part A and of part B's rule), tests/translated_reads.rs, scripts' measure tests.
#![allow(dead_code)]
use crate::translated_common::Tr;
use cs_game::kernel::KernelGame;
use cs_game::placed::{NoStep, Placed};
use cs_game::{Fact, Game, Role};
use cs_rep::names::{Kind, Place};
use std::cell::{Cell, RefCell};
use std::collections::{BTreeSet, HashMap};
use std::rc::Rc;

pub type Pairs = Vec<(String, Vec<String>)>;
pub const FREE: u32 = u32::MAX;

/// Term numbers of the module as the numbers of the answers: a file, a rank, a kind, a side.
pub struct Dec { file: HashMap<u32, u16>, rank: HashMap<u32, u16>, kind: HashMap<u32, u16>, side: HashMap<u32, u16> }

impl Dec {
    pub fn new<T: Tr>(t: &T, kinds: &[String], roles: &[String]) -> Dec {
        let mut d = Dec { file: HashMap::new(), rank: HashMap::new(), kind: HashMap::new(), side: HashMap::new() };
        for (i, c) in "abcdefgh".chars().enumerate() { if let Some(n) = t.sym_t(&c.to_string()) { d.file.insert(n, i as u16); } }
        for (i, c) in "12345678".chars().enumerate() { if let Some(n) = t.sym_t(&c.to_string()) { d.rank.insert(n, i as u16); } }
        for (i, k) in kinds.iter().enumerate() { if let Some(n) = t.sym_t(k) { d.kind.insert(n, i as u16); } }
        for (i, r) in roles.iter().enumerate() { if let Some(n) = t.sym_t(r) { d.side.insert(n, i as u16); } }
        d
    }
    pub fn place(&self, f: u32, r: u32) -> Option<u16> { Some(self.file.get(&f)? * 8 + self.rank.get(&r)?) }
    pub fn kind(&self, k: u32) -> Option<u16> { self.kind.get(&k).copied() }
    pub fn side(&self, c: u32) -> Option<u16> { self.side.get(&c).copied() }
    pub fn file_of(&self, f: u32) -> Option<u16> { self.file.get(&f).copied() }
    pub fn rank_of(&self, r: u32) -> Option<u16> { self.rank.get(&r).copied() }
}

pub fn pairs_of(facts: &[Fact]) -> Pairs { facts.iter().map(|x| (x.relation.clone(), x.args.clone())).collect() }

/// The state of the question of a side's units: the side to move, no passed-over place.
pub fn with_control(base: &Pairs, side: &str) -> Pairs {
    let mut v: Pairs = base.iter().filter(|(r, _)| r != "control" && r != "ep_square").cloned().collect();
    v.push(("control".into(), vec![side.to_string()]));
    v
}

pub fn arity(rel: &str) -> usize { match rel { "placed_cell" => 4, "placed_bears" | "placed_reaches" => 6, "placed_strikes" => 3, _ => 6 } }

pub fn ask<T: Tr>(t: &mut T, s: &T::S, rels: &[&str]) -> Vec<Vec<Vec<u32>>> {
    let q: Vec<(String, Vec<u32>)> = rels.iter().map(|n| (n.to_string(), vec![FREE; arity(n)])).collect();
    t.ask_t(s, &[], &q).into_iter().map(|(rows, _)| rows).collect()
}

/// What the module says of one state: what stands where, and for each unit what it bears on and where it may move.
#[derive(Debug, Default, Clone)]
pub struct StateRows {
    pub cells: Vec<(Role, Kind, Place)>,
    pub bears: HashMap<(Role, Kind, Place), Vec<Place>>,
    pub reaches: HashMap<(Role, Kind, Place), Vec<Place>>,
}

pub struct Described<T: Tr> {
    pub kernel: KernelGame,
    t: RefCell<T>,
    dec: Dec,
    cache: RefCell<HashMap<u64, Rc<StateRows>>>,
    /// (questions of a unit's bearing, questions of a unit's reach, states the module was asked at)
    asked: Cell<(u64, u64, u64)>,
}

impl<T: Tr> Described<T> {
    pub fn new(kernel: KernelGame, t: T) -> Described<T> {
        let dec = Dec::new(&t, &kernel.kind_names(), &kernel.roles());
        Described { kernel, t: RefCell::new(t), dec, cache: RefCell::new(HashMap::new()), asked: Cell::new((0, 0, 0)) }
    }
    pub fn asked(&self) -> (u64, u64, u64) { self.asked.get() }
    pub fn reset(&self) { self.asked.set((0, 0, 0)); self.cache.borrow_mut().clear(); }

    /// The module's rows of one state (made once per state key).
    pub fn rows(&self, s: &<KernelGame as Game>::State) -> Rc<StateRows> {
        let key = self.kernel.key(s);
        if let Some(r) = self.cache.borrow().get(&key) { return r.clone(); }
        let base = pairs_of(&self.kernel.state_facts(s));
        let roles = self.kernel.roles();
        let mut t = self.t.borrow_mut();
        let s0 = t.state_of_facts_t(&base).expect("the module reads the state");
        let got = ask(&mut *t, &s0, &["placed_cell", "placed_bears"]);
        let mut out = StateRows::default();
        for x in &got[0] {
            if let (Some(p), Some(o), Some(k)) = (self.dec.place(x[0], x[1]), self.dec.side(x[2]), self.dec.kind(x[3])) { out.cells.push((o as Role, k as Kind, p)); }
        }
        out.cells.sort();
        for x in &got[1] {
            if let (Some(k), Some(o), Some(p), Some(q)) = (self.dec.kind(x[0]), self.dec.side(x[1]), self.dec.place(x[2], x[3]), self.dec.place(x[4], x[5])) { out.bears.entry((o as Role, k as Kind, p)).or_default().push(q); }
        }
        for (oi, name) in roles.iter().enumerate() {
            let s1 = t.state_of_facts_t(&with_control(&base, name)).expect("the module reads the state of a side's questions");
            let rows = ask(&mut *t, &s1, &["placed_reaches"]);
            for x in &rows[0] {
                if let (Some(k), Some(o), Some(p), Some(q)) = (self.dec.kind(x[0]), self.dec.side(x[1]), self.dec.place(x[2], x[3]), self.dec.place(x[4], x[5])) { if o as usize == oi { out.reaches.entry((o as Role, k as Kind, p)).or_default().push(q); } }
            }
        }
        for v in out.bears.values_mut().chain(out.reaches.values_mut()) { v.sort_unstable(); v.dedup(); }
        let a = self.asked.get();
        self.asked.set((a.0, a.1, a.2 + 1));
        let r = Rc::new(out);
        self.cache.borrow_mut().insert(key, r.clone());
        r
    }
}

impl<T: Tr> Game for Described<T> {
    type State = <KernelGame as Game>::State;
    type Move = <KernelGame as Game>::Move;
    fn name(&self) -> String { self.kernel.name() }
    fn roles(&self) -> Vec<String> { self.kernel.roles() }
    fn initial(&self) -> Self::State { self.kernel.initial() }
    fn mover(&self, s: &Self::State) -> Option<Role> { self.kernel.mover(s) }
    fn turn(&self, s: &Self::State) -> Role { self.kernel.turn(s) }
    fn moves(&self, s: &Self::State) -> Vec<Self::Move> { self.kernel.moves(s) }
    fn apply(&self, s: &Self::State, m: &Self::Move) -> Self::State { self.kernel.apply(s, m) }
    fn outcome(&self, s: &Self::State) -> Option<Vec<i64>> { self.kernel.outcome(s) }
    fn move_name(&self, s: &Self::State, m: &Self::Move) -> String { self.kernel.move_name(s, m) }
    fn key(&self, s: &Self::State) -> u64 { self.kernel.key(s) }
    fn units(&self, s: &Self::State) -> Vec<(Role, String)> { self.kernel.units(s) }
}

impl<T: Tr> Placed for Described<T> {
    fn place_names(&self) -> Vec<String> { self.kernel.place_names() }
    fn kind_names(&self) -> Vec<String> { self.kernel.kind_names() }
    fn cells(&self, s: &Self::State) -> Vec<(Role, Kind, Place)> { self.rows(s).cells.clone() }
    fn bearing_from(&self, s: &Self::State, owner: Role, kind: Kind, at: Place) -> Vec<Place> {
        let a = self.asked.get();
        self.asked.set((a.0 + 1, a.1, a.2));
        self.rows(s).bears.get(&(owner, kind, at)).cloned().unwrap_or_default()
    }
    fn reach_from(&self, s: &Self::State, owner: Role, kind: Kind, at: Place) -> Vec<Place> {
        let a = self.asked.get();
        self.asked.set((a.0, a.1 + 1, a.2));
        self.rows(s).reaches.get(&(owner, kind, at)).cloned().unwrap_or_default()
    }
    fn move_ends(&self, s: &Self::State, m: &Self::Move) -> (Place, Place) { self.kernel.move_ends(s, m) }
    fn kind_at(&self, s: &Self::State, at: Place) -> Option<Kind> { self.kernel.kind_at(s, at) }
    fn find_step(&self, s: &Self::State, from: Place, to: Place) -> Option<Self::Move> { self.kernel.find_step(s, from, to) }
    fn why_not(&self, s: &Self::State, from: Place, to: Place) -> (NoStep, Vec<Place>) { self.kernel.why_not(s, from, to) }
    fn move_info(&self, s: &Self::State, m: &Self::Move) -> (Kind, Option<Kind>) { self.kernel.move_info(s, m) }
    fn strikes_place(&self, s: &Self::State, owner: Role, at: Place) -> bool { self.kernel.strikes_place(s, owner, at) }
    fn without(&self, s: &Self::State, at: Place) -> Option<Self::State> { self.kernel.without(s, at) }
    fn relocate(&self, s: &Self::State, from: Place, to: Place) -> Option<Self::State> { self.kernel.relocate(s, from, to) }
}
