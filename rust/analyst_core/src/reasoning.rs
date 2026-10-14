//! THE REASONING LEDGER (Kaveh 2026-10-14, docs/ANALYST_PLAN.md section 39): "analyze the safety of each piece ... each question has
//! to be answered and each question affects some decisions ... track evidence ... those proofs become new things upon which we reason
//! ... the moment we get rid of the pin, immediately we should know that the pawn can move ... roll them forward in time".
//! (The module is `reasoning`; `ledger.rs` is the 2026-10-10 JOB ledger of threat.rs, which this one sits beside.)
//!
//! FACTS are the vocabulary's words, evaluated by the existing evaluator (vocab.rs) WITH their read set (vocab::reads). QUESTIONS are
//! typed, about named pieces, from the critical-question menu of the claim they challenge. FINDINGS are the answered questions: a claim,
//! a value interval in material (pawn units) for the side that owns the piece, a status (PROVEN / REFUTED / CONDITIONAL / UNKNOWN, and
//! ESTIMATED for a closed cycle or a rule bound), the squares read and the findings cited, keyed by a canonical form (type, side that
//! acts, the pieces as sorted (kind, square), the extra squares). RETRACTION: by read set (a move that touches a square a finding read)
//! and by cascade (a finding citing a retracted one); STALE with the ply, never deleted. LISTENERS: premise key -> the findings
//! waiting on it. The exact solver (goals.rs) only CONFIRMS the final claim.
//!
//! WHOSE MOVE: a question is stamped with the side that acts (the side NOT owning the piece for "is X safe": as if it moved now); the
//! facts are static (no side to move in `see`, `after`, `without`, `attacks`), so the same code reads both sides and a finding carries
//! over a ply of either side. Squares in facts are named from the side to move (vocab.rs); `nm` does that.
use super::*;
use shakmaty::Bitboard;
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};

#[derive(Clone, Copy, PartialEq, Debug)]
enum St { Proven, Refuted, Conditional, Unknown, Estimated }
impl St {
    fn name(&self) -> &'static str { match self { St::Proven => "PROVEN", St::Refuted => "REFUTED", St::Conditional => "CONDITIONAL", St::Unknown => "UNKNOWN", St::Estimated => "ESTIMATED" } }
}

#[derive(Clone)]
struct Finding {
    id: usize, key: String, hash: u64, kind: &'static str, claim: String, lo: i64, hi: i64, st: St, reads: Bitboard, cites: Vec<usize>,
    premise: Option<String>, ply: u32, stale: Option<u32>, circular: bool, note: String, facts: Vec<(String, i64)>,
}

type Pc = (Role, Square);

fn val(r: Role) -> i64 { if r == Role::King { 100 } else { crate::rules::val(r) } }
fn letter(r: Role) -> char { match r { Role::Pawn => 'P', Role::Knight => 'N', Role::Bishop => 'B', Role::Rook => 'R', Role::Queen => 'Q', Role::King => 'K' } }
fn pname(p: Pc) -> String { if p.0 == Role::Pawn { p.1.to_string() } else { format!("{}{}", letter(p.0), p.1) } }
fn hash_of(s: &str) -> u64 { use std::hash::{Hash, Hasher}; let mut h = std::collections::hash_map::DefaultHasher::new(); s.hash(&mut h); h.finish() }
fn side(c: Color) -> &'static str { if c == Color::White { "w" } else { "b" } }

#[pyclass(unsendable, name = "Ledger")]
pub struct ReasoningLedger {
    pos: Chess,
    line: Vec<Chess>,
    ply: u32,
    fs: Vec<Finding>,
    live: HashMap<u64, usize>,
    listeners: HashMap<u64, Vec<u64>>,
    fact_cache: HashMap<String, (i64, Bitboard)>,
    budget: u32,
    used: u32,
    carried: u32,
    reasked: u32,
    facts_n: u32,
    solver_calls: u32,
    stack: Vec<u64>,
    skipped: Vec<String>,
    agenda: Vec<String>,
    plies: Vec<Value>,
    confirmations: Vec<Value>,
    tt: HashMap<(u64, i32), ((i64, i64), Vec<String>)>,
    nodes: u32,
    node_cap_hit: bool,
    budget_hit: bool,
    deep: bool,
    cap_now: u32,
    cuts: u32,
}

impl ReasoningLedger {
    fn nm(&self, s: Square) -> String { (if self.pos.turn() == Color::White { s } else { s.flip_vertical() }).to_string() }

    /// A fact: a vocabulary term's value with the squares it read. `gone`: the term read on the board without the piece on that square.
    fn fact(&mut self, term: &str) -> (i64, Bitboard) {
        if let Some(x) = self.fact_cache.get(term) { return *x; }
        self.facts_n += 1;
        crate::game::install_line(self.line.clone());
        let r = match crate::vocab::parse_term(term) {
            Some(t) => { let lone = !self.pos.turn(); (crate::vocab::num(&t, &self.pos, lone) as i64, crate::vocab::reads(&t, &self.pos, lone)) }
            None => (0, Bitboard::FULL),
        };
        self.fact_cache.insert(term.to_string(), r);
        r
    }
    fn w(&self, gone: Option<Square>, t: &str) -> String { match gone { Some(g) => format!("(without (at {}) {})", self.nm(g), t), None => t.to_string() } }

    fn board_c(&self, gone: Option<Square>) -> Chess {
        match gone {
            Some(g) => { let mut s = self.pos.clone().into_setup(shakmaty::EnPassantMode::Legal); s.board.discard_piece_at(g); crate::vocab::relaxed(s).unwrap_or_else(|| self.pos.clone()) }
            None => self.pos.clone(),
        }
    }
    fn lookup(&mut self, hash: u64) -> Option<usize> {
        if let Some(&i) = self.live.get(&hash) { if self.fs[i].stale.is_none() { self.carried += 1; return Some(i); } }
        None
    }
    fn add(&mut self, mut f: Finding) -> usize {
        f.id = self.fs.len();
        f.ply = self.ply;
        self.live.insert(f.hash, f.id);
        self.reasked += 1;
        let id = f.id;
        self.fs.push(f);
        id
    }
    fn skeleton(&self, kind: &'static str, key: String, claim: String) -> Finding {
        Finding { id: 0, hash: hash_of(&key), key, kind, claim, lo: 0, hi: 0, st: St::Proven, reads: Bitboard::EMPTY, cites: vec![], premise: None, ply: 0,
                  stale: None, circular: false, note: String::new(), facts: vec![] }
    }
    fn ckey(kind: &str, actor: Color, pcs: &[Pc], extra: &[Square]) -> String {
        let mut v: Vec<Pc> = pcs.to_vec();
        v.sort_by_key(|(r, s)| (std::cmp::Reverse(val(*r)), *s as u32));
        format!("{kind}|{}|{}|{}", side(actor), v.iter().map(|p| pname(*p)).collect::<Vec<_>>().join(","), extra.iter().map(|s| s.to_string()).collect::<Vec<_>>().join(","))
    }
    /// The question is over budget: an ESTIMATED finding with the rule bound, the skipped question recorded.
    fn over_budget(&mut self, kind: &'static str, key: String, claim: String, hi: i64) -> usize {
        let mut f = self.skeleton(kind, key.clone(), claim);
        f.st = St::Unknown; f.lo = 0; f.hi = hi; f.note = format!("over budget ({}): skipped {}", self.budget, key);
        self.skipped.push(key);
        self.add(f)
    }
    fn spend(&mut self) -> bool { self.used += 1; self.used <= self.budget }

    /// Can the piece on `r` legally take on `y` after the piece on `a` has taken there (the board `b`, `owner` to move)?
    fn legal_move(b: &Chess, owner: Color, from: Square, to: Square) -> bool {
        let mut s = b.clone().into_setup(shakmaty::EnPassantMode::Legal);
        s.turn = owner;
        match crate::vocab::relaxed(s) { Some(p) => p.legal_moves().iter().any(|m| m.from() == Some(from) && m.to() == to), None => false }
    }
    fn board_after_capture(b: &Chess, a: Square, y: Square) -> Option<Chess> {
        let mut s = b.clone().into_setup(shakmaty::EnPassantMode::Legal);
        let pc = s.board.piece_at(a)?;
        s.board.discard_piece_at(a);
        s.board.set_piece_at(y, pc);
        s.turn = !pc.color;
        crate::vocab::relaxed(s)
    }
    fn pieces_on(b: &shakmaty::Board, c: Color) -> Vec<Pc> {
        b.by_color(c).into_iter().filter_map(|s| b.piece_at(s).filter(|p| p.role != Role::King).map(|p| (p.role, s))).collect()
    }
    fn pieces_of(&self, c: Color) -> Vec<Pc> {
        let b = self.pos.board();
        b.by_color(c).into_iter().filter_map(|s| b.piece_at(s).filter(|p| p.role != Role::King).map(|p| (p.role, s))).collect()
    }

    // ---- JOB(R, Z): R guards Z and Z would be worse off without R ------------------------------------------------------------------
    fn ask_job(&mut self, r: Pc, z: Pc, gone: Option<Square>) -> usize {
        let owner = self.pos.board().color_at(r.1).unwrap_or(Color::White);
        let key = Self::ckey("JOB", !owner, &[r, z], gone.as_slice());
        if let Some(i) = self.lookup(hash_of(&key)) { return i; }
        let term = self.w(gone, &format!("(diff (max (without (at {r}) (see {z})) (see {z})) (see {z}))", r = self.nm(r.1), z = self.nm(z.1)));
        let mut f = self.skeleton("JOB", key.clone(), String::new());
        if !self.spend() { return self.over_budget("JOB", key, format!("{} has a job on {}?", pname(r), pname(z)), val(z.0)); }
        let (j, rd) = self.fact(&term);
        f.reads = rd; f.lo = j; f.hi = j; f.facts.push((term, j));
        f.claim = if j > 0 { format!("{} has another job: without it {} would lose {} more", pname(r), pname(z), j) } else { format!("{} has no job on {}", pname(r), pname(z)) };
        self.add(f)
    }

    // ---- RECAP(R, y, A, Z): after A takes y and R retakes, what does Z lose that R was guarding ---------------------------------------
    fn ask_recapture(&mut self, r: Pc, y: Pc, a: Pc, z: Pc, gone: Option<Square>, job: usize) -> usize {
        let owner = self.pos.board().color_at(y.1).unwrap_or(Color::White);
        let key = Self::ckey("RECAP", !owner, &[r, y, a, z], gone.as_slice());
        if let Some(i) = self.lookup(hash_of(&key)) { return i; }
        if !self.spend() { return self.over_budget("RECAP", key, format!("after {} takes {} and {} retakes, is {} still guarded?", pname(a), pname(y), pname(r), pname(z)), val(z.0)); }
        let (zn, yn, an, rn) = (self.nm(z.1), self.nm(y.1), self.nm(a.1), self.nm(r.1));
        let base_t = self.w(gone, &format!("(see {zn})"));
        let aft_t = self.w(gone, &format!("(after {an} {yn} (after {rn} {yn} (see {zn})))"));
        let (b0, r0) = self.fact(&base_t);
        let (a1, r1) = self.fact(&aft_t);
        let loss = (a1 - b0).max(0);
        let mut f = self.skeleton("RECAP", key, String::new());
        f.reads = r0 | r1; f.lo = loss; f.hi = loss; f.cites = vec![job]; f.facts = vec![(base_t, b0), (aft_t, a1)];
        f.claim = if loss > 0 {
            format!("after {} takes {} and {} recaptures, {} no longer sees {}: {} is then worth {} to them", pname(a), pname(y), pname(r), pname(r), pname(z), pname(z), a1)
        } else { format!("after {} takes {} and {} recaptures, {} stays guarded", pname(a), pname(y), pname(r), pname(z)) };
        self.add(f)
    }

    // ---- SAFE(Y [| X gone]): what the exchange on Y really costs its owner -----------------------------------------------------------
    fn ask_safe(&mut self, y: Pc, gone: Option<Square>) -> usize {
        let owner = self.pos.board().color_at(y.1).unwrap_or(Color::White);
        let key = Self::ckey("SAFE", !owner, &[y], gone.as_slice());
        let h = hash_of(&key);
        if let Some(i) = self.lookup(h) { return i; }
        let gl = gone.map(|g| format!(" (with {} gone)", g)).unwrap_or_default();
        if self.stack.contains(&h) {                                    // a cycle: closed as ESTIMATED with the rule bound, marked CIRCULAR
            let mut f = self.skeleton("SAFE", key, format!("{} assumed safe{} (circular)", pname(y), gl));
            f.st = St::Estimated; f.circular = true; f.hi = val(y.0);
            return self.add(f);
        }
        if !self.spend() { return self.over_budget("SAFE", key, format!("is {} safe{}?", pname(y), gl), val(y.0)); }
        self.stack.push(h);
        let b = self.board_c(gone);
        let occ = b.board().occupied();
        let yn = self.nm(y.1);
        let mut f = self.skeleton("SAFE", key, String::new());
        let (v0, rd0) = self.fact(&self.w(gone, &format!("(see {yn})")));
        f.reads = rd0; f.facts.push((self.w(gone, &format!("(see {yn})")), v0));
        // the attackers that may legally capture
        let atk: Vec<Pc> = (b.board().attacks_to(y.1, !owner, occ) & b.board().by_color(!owner)).into_iter()
            .filter_map(|s| b.board().piece_at(s).map(|p| (p.role, s)))
            .filter(|(_, s)| Self::legal_move(&b, !owner, *s, y.1)).collect();
        if atk.is_empty() {
            f.claim = format!("{}{} is not attacked", pname(y), gl);
            self.stack.pop();
            return self.add(f);
        }
        let a = *atk.iter().min_by_key(|(r, s)| (val(*r), *s as u32)).unwrap();
        // the recapturers: each must be able to retake legally (not pinned, not blocked by check)
        let after_cap = Self::board_after_capture(&b, a.1, y.1);
        let defs: Vec<Pc> = (b.board().attacks_to(y.1, owner, occ) & b.board().by_color(owner)).into_iter()
            .filter_map(|s| b.board().piece_at(s).map(|p| (p.role, s)))
            .filter(|(_, s)| after_cap.as_ref().map_or(false, |p| Self::legal_move(p, owner, *s, y.1))).collect();
        if let Some(k) = b.board().king_of(owner) { f.reads |= crate::vocab::exchange_reads(b.board(), k) | crate::vocab::exchange_reads(b.board(), a.1); }
        // the capture with check? (no in-between moves for them)
        let mut check = false;
        if let Some(k) = b.board().king_of(owner) {
            let t = self.w(gone, &format!("(after {} {yn} (count (and (attacks {yn}) (at {}))))", self.nm(a.1), self.nm(k)));
            let (c, rc) = self.fact(&t);
            check = c > 0; f.reads |= rc; f.facts.push((t, c));
        }
        let inb = if check { 0 } else {
            let bb = b.board();
            bb.by_color(!owner).into_iter().filter(|s| *s != a.1).filter_map(|s| bb.piece_at(s).map(|p| (p.role, s)))
                .filter(|(r, s)| *r != Role::King && (bb.attacks_to(*s, owner, occ) & bb.by_color(owner)).any()).map(|(r, _)| val(r)).max().unwrap_or(0)
        };
        let mut best_r: Option<(Pc, i64)> = None;
        let mut cites = vec![];
        let mut cost = if defs.is_empty() { v0 } else { val(y.0) };
        let mut jobbed = vec![];
        for r in &defs {
            let mut c_r = 0;
            for z in self.pieces_of(owner) {
                if z == *r || z.1 == y.1 || Some(z.1) == gone { continue; }
                if !b.board().attacks_from(r.1).contains(z.1) { continue; }       // R does not guard Z
                if !attacked(b.board(), z.1, !owner) { continue; }                // Z is not attacked: no job to do
                let j = self.ask_job(*r, z, gone);
                cites.push(j);
                if self.fs[j].lo > 0 {
                    let rc = self.ask_recapture(*r, y, a, z, gone, j);
                    cites.push(rc);
                    c_r = c_r.max(self.fs[rc].lo);
                    if self.fs[rc].lo > 0 { jobbed.push(format!("{} guards {} (worth {})", pname(*r), pname(z), self.fs[rc].lo)); }
                }
            }
            let cand = v0.max(val(y.0) - val(a.0) + c_r);
            if best_r.map_or(true, |(_, c)| cand < c) { best_r = Some((*r, cand)); }
        }
        if let Some((_, c)) = best_r { cost = cost.min(c); }
        let hi = cost;
        let lo = (cost - inb).max(0);
        f.lo = lo; f.hi = hi; f.cites = cites;
        for c in f.cites.clone() { f.reads |= self.fs[c].reads; }
        f.claim = if v0 > 0 {
            format!("{}{} loses {} by the exchange ({} takes first)", pname(y), gl, v0, pname(a))
        } else if defs.is_empty() {
            format!("{}{} is attacked by {} and has no legal recapture: loses {}", pname(y), gl, pname(a), cost)
        } else if cost > 0 {
            format!("{}{} is effectively unguarded: {} takes; {} can retake only by leaving a job ({}) -> loses {}{}", pname(y), gl, pname(a),
                    defs.iter().map(|d| pname(*d)).collect::<Vec<_>>().join("/"), jobbed.join("; "), cost, if check { "; the capture gives check, no in-between move" } else { "" })
        } else {
            format!("{}{} is safe: {} takes, {} retakes (trade)", pname(y), gl, pname(a), defs.iter().map(|d| pname(*d)).collect::<Vec<_>>().join("/"))
        };
        f.note = format!("attacker {} defenders [{}] check={} in-between<={}", pname(a), defs.iter().map(|d| pname(*d)).collect::<Vec<_>>().join(","), check, inb);
        self.stack.pop();
        self.add(f)
    }

    // ---- BLOCKS(X, A, Y) and PIN(X): if X leaves, A hits Y; what do X's moves cost -------------------------------------------------------
    fn ask_blocks(&mut self, x: Pc, a: Pc, y: Pc) -> usize {
        let owner = self.pos.board().color_at(x.1).unwrap_or(Color::White);
        let key = Self::ckey("BLOCKS", !owner, &[x, a, y], &[]);
        if let Some(i) = self.lookup(hash_of(&key)) { return i; }
        if !self.spend() { return self.over_budget("BLOCKS", key, format!("{} blocks {} from {}?", pname(x), pname(a), pname(y)), val(y.0)); }
        let (xn, an, yn) = (self.nm(x.1), self.nm(a.1), self.nm(y.1));
        let t = format!("(without (at {xn}) (count (and (attacks {an}) (at {yn}))))");
        let (g, rg) = self.fact(&t);
        let mut f = self.skeleton("BLOCKS", key, String::new());
        f.reads = rg | Bitboard::from(x.1); f.facts.push((t, g));
        if g == 0 {
            f.st = St::Refuted; f.claim = format!("{} does not shield {} from {}", pname(x), pname(y), pname(a));
            return self.add(f);
        }
        let s = self.ask_safe(y, Some(x.1));
        f.cites = vec![s]; f.reads |= self.fs[s].reads;
        f.lo = self.fs[s].lo; f.hi = self.fs[s].hi;
        f.st = St::Conditional;
        f.premise = Some(format!("{} leaves {}", pname(x), x.1));
        f.claim = format!("if {} leaves, {} hits {}: {}", pname(x), pname(a), pname(y), self.fs[s].claim);
        let id = self.add(f);
        let ph = hash_of(&format!("LEAVES|{}", x.1));
        self.listeners.entry(ph).or_default().push(self.fs[id].hash);
        id
    }

    fn ask_pin(&mut self, x: Pc, a: Pc, y: Pc, blocks: usize) -> usize {
        let owner = self.pos.board().color_at(x.1).unwrap_or(Color::White);
        let key = Self::ckey("PIN", !owner, &[x, a, y], &[]);
        if let Some(i) = self.lookup(hash_of(&key)) { return i; }
        if !self.spend() { return self.over_budget("PIN", key, format!("is {} pinned to {}?", pname(x), pname(y)), val(y.0)); }
        let bl = self.fs[blocks].clone();
        let mut f = self.skeleton("PIN", key, String::new());
        f.cites = vec![blocks]; f.reads = bl.reads;
        // the moves of X (the owner to move)
        let mut s = self.pos.clone().into_setup(shakmaty::EnPassantMode::Legal);
        s.turn = owner;
        let po = crate::vocab::relaxed(s).unwrap_or_else(|| self.pos.clone());
        let moves: Vec<Move> = po.legal_moves().into_iter().filter(|m| m.from() == Some(x.1)).collect();
        if moves.is_empty() || bl.st == St::Refuted {
            f.st = if bl.st == St::Refuted { St::Refuted } else { St::Proven };
            f.claim = format!("{} has no move that costs: {}", pname(x), if bl.st == St::Refuted { "it shields nothing" } else { "it cannot move" });
            return self.add(f);
        }
        let (mut lo_all, mut hi_all) = (i64::MAX, i64::MIN);
        let mut best = String::new();
        let mut per = vec![];
        for m in &moves {
            let to = m.to();
            let gain_hi = m.capture().map_or(0, val);
            let tn = self.nm(to);
            let t = format!("(after {} {tn} (see {tn}))", self.nm(x.1));
            let (loss, rl) = self.fact(&t);
            f.reads |= rl; f.facts.push((t, loss));
            let gain_lo = gain_hi - loss;
            let tempo = { let mut q = po.clone(); q.play_unchecked(m); q.is_check() };
            let mut lo = bl.lo - gain_hi;
            let hi = bl.hi - gain_lo;
            if tempo { lo = lo.min(0); }
            per.push(json!({"move": m.to_uci(CastlingMode::Standard).to_string(), "gain": [gain_lo, gain_hi], "check": tempo, "cost": [lo, hi]}));
            if lo < lo_all { lo_all = lo; best = m.to_uci(CastlingMode::Standard).to_string(); }
            hi_all = hi_all.max(hi);
        }
        f.lo = lo_all; f.hi = hi_all;
        f.st = if bl.st == St::Unknown { St::Unknown } else { St::Proven };
        f.note = serde_json::to_string(&per).unwrap_or_default();
        f.claim = if lo_all > 0 {
            format!("{} is pinned to {}: moving it costs >= {} (the cheapest move is {})", pname(x), pname(y), lo_all, best)
        } else if hi_all <= 0 {
            format!("{} is free to move: no move of it costs anything (<= {})", pname(x), hi_all)
        } else {
            format!("{} may move: it costs between {} and {}", pname(x), lo_all.max(0), hi_all)
        };
        let id = self.add(f);
        // the solver CONFIRMS the final claim, never finds it
        if lo_all > 0 && bl.lo > 0 { self.confirm(id, &po, bl.lo, &moves); }
        id
    }

    fn confirm(&mut self, id: usize, po: &Chess, value: i64, moves: &[Move]) {
        self.solver_calls += 1;
        let fen = shakmaty::fen::Fen::from_position(po.clone(), shakmaty::EnPassantMode::Legal).to_string();
        let res = crate::goals::defence_moves(fen, value, 3, 2_000_000);
        let mut ok = true;
        let mut rows = vec![];
        for m in moves {
            let u = m.to_uci(CastlingMode::Standard).to_string();
            let st = res.iter().find(|r| r.0 == u).map(|r| r.1.clone()).unwrap_or_else(|| "missing".into());
            let line = res.iter().find(|r| r.0 == u).map(|r| r.3.join(" ")).unwrap_or_default();
            ok &= st == "forced";
            rows.push(json!({"move": u, "their_gain_of": value, "status": st, "line": line}));
        }
        self.confirmations.push(json!({"finding": id, "value": value, "confirmed": ok, "moves": rows}));
        let f = &mut self.fs[id];
        if ok { f.note.push_str(&format!(" | solver: every move of it leaves their gain of >= {} forced", value)); }
        else { f.st = St::Unknown; f.note.push_str(" | solver: NOT confirmed"); }
    }

    // ---- the entry pass, the agenda ---------------------------------------------------------------------------------------------------
    fn run_pass(&mut self) {
        self.used = 0;
        self.agenda.clear();
        self.skipped.clear();
        let b = self.pos.board().clone();
        let mut tasks: Vec<(i64, String, Option<(Pc, Pc, Pc)>, Option<Pc>)> = vec![];
        for c in [Color::White, Color::Black] {
            for p in self.pieces_of(c) {
                tasks.push((val(p.0), format!("SAFE {}", pname(p)), None, Some(p)));
            }
        }
        // X blocks A -> Y: a slider, one piece of the owner's between it and another of the owner's
        for c in [Color::White, Color::Black] {
            for a in self.pieces_of(!c).into_iter().filter(|p| matches!(p.0, Role::Queen | Role::Rook | Role::Bishop)) {
                for y in self.pieces_of(c) {
                    let btw = shakmaty::attacks::between(a.1, y.1);
                    let on_line = match a.0 { Role::Rook => a.1.file() == y.1.file() || a.1.rank() == y.1.rank(),
                        Role::Bishop => shakmaty::attacks::bishop_attacks(a.1, Bitboard::EMPTY).contains(y.1),
                        _ => shakmaty::attacks::queen_attacks(a.1, Bitboard::EMPTY).contains(y.1) };
                    if !on_line || btw.is_empty() { continue; }
                    let between = btw & b.occupied();
                    if between.count() != 1 { continue; }
                    let xs = between.first().unwrap();
                    if b.color_at(xs) != Some(c) { continue; }
                    let Some(xp) = b.piece_at(xs) else { continue };
                    if xp.role == Role::King { continue; }
                    tasks.push((val(y.0), format!("BLOCKS {} -> {} through {}", pname(a), pname(y), pname((xp.role, xs))), Some(((xp.role, xs), a, y)), None));
                }
            }
        }
        tasks.sort_by_key(|t| (std::cmp::Reverse(t.0), t.1.clone()));       // the agenda: by stake (the value of the piece the question is about)
        for t in &tasks { self.agenda.push(format!("{} (stake {})", t.1, t.0)); }
        for (_, _, blk, safe) in tasks {
            if let Some(p) = safe { self.ask_safe(p, None); }
            if let Some((x, a, y)) = blk { let bi = self.ask_blocks(x, a, y); self.ask_pin(x, a, y, bi); }
        }
    }

    fn retract(&mut self, touched: Bitboard) -> usize { self.retract_ids(touched).len() }

    fn retract_ids(&mut self, touched: Bitboard) -> Vec<usize> {
        let mut n = vec![];
        let mut changed = true;
        while changed {
            changed = false;
            for i in 0..self.fs.len() {
                if self.fs[i].stale.is_some() { continue; }
                let hit = (self.fs[i].reads & touched).any() || self.fs[i].cites.iter().any(|&c| self.fs[c].stale.is_some());
                if hit { self.fs[i].stale = Some(self.ply); n.push(i); changed = true; }
            }
        }
        n
    }

    fn counts(&self) -> Value {
        let mut m: HashMap<&str, u32> = HashMap::new();
        for f in self.fs.iter().filter(|f| f.stale.is_none()) { *m.entry(f.st.name()).or_insert(0) += 1; }
        json!(m)
    }

    fn finding_json(&self, f: &Finding) -> Value {
        json!({"id": f.id, "key": f.key, "hash": format!("{:016x}", f.hash), "type": f.kind, "claim": f.claim, "status": f.st.name(), "lo": f.lo, "hi": f.hi,
               "reads": f.reads.into_iter().map(|s| s.to_string()).collect::<Vec<_>>(), "cites": f.cites, "premise": f.premise, "ply": f.ply,
               "stale": f.stale, "circular": f.circular, "note": f.note, "facts": f.facts})
    }

    fn report(&self) -> Value {
        let live: Vec<Value> = self.fs.iter().filter(|f| f.stale.is_none()).map(|f| self.finding_json(f)).collect();
        let stale: Vec<Value> = self.fs.iter().filter(|f| f.stale.is_some()).map(|f| self.finding_json(f)).collect();
        let pocket: Vec<Value> = self.fs.iter().filter(|f| f.stale.is_none() && f.st == St::Conditional && f.hi > 0)
            .map(|f| json!({"claim": f.claim, "premise": f.premise, "value": [f.lo, f.hi]})).collect();
        json!({"fen": shakmaty::fen::Fen::from_position(self.pos.clone(), shakmaty::EnPassantMode::Legal).to_string(), "ply": self.ply,
               "counts": self.counts(), "questions_asked": self.used.min(self.budget + self.skipped.len() as u32), "budget": self.budget,
               "skipped": self.skipped, "facts_evaluated": self.facts_n, "solver_calls": self.solver_calls, "agenda": self.agenda,
               "findings": live, "stale": stale, "back_pocket": pocket, "confirmations": self.confirmations, "plies": self.plies,
               "listeners": self.listeners.iter().map(|(k, v)| (format!("{:016x}", k), v.len())).collect::<HashMap<_, _>>()})
    }
}

#[pymethods]
impl ReasoningLedger {
    #[new]
    #[pyo3(signature = (fen, budget=200))]
    fn new(fen: String, budget: u32) -> PyResult<Self> {
        let pos = crate::parse(&fen).ok_or_else(|| pyo3::exceptions::PyValueError::new_err("bad fen"))?;
        Ok(ReasoningLedger { line: vec![pos.clone()], pos, ply: 0, fs: vec![], live: HashMap::new(), listeners: HashMap::new(), fact_cache: HashMap::new(),
            budget, used: 0, carried: 0, reasked: 0, facts_n: 0, solver_calls: 0, stack: vec![], skipped: vec![], agenda: vec![], plies: vec![], confirmations: vec![], tt: HashMap::new(), nodes: 0, node_cap_hit: false, budget_hit: false, deep: false, cap_now: 0, cuts: 0 })
    }

    /// Ask every question of the entry pass (SAFE of every piece, BLOCKS of every shielded piece with its PIN) and return the report.
    fn run(&mut self, py: Python<'_>) -> PyResult<PyObject> {
        self.carried = 0; self.reasked = 0;
        self.run_pass();
        let j = py.import("json")?;
        Ok(j.call_method1("loads", (self.report().to_string(),))?.unbind())
    }

    /// Roll forward one move (uci): retract the findings whose read squares the move touched (and what cites them), re-run the entry pass
    /// (what carried over is a lookup, the rest is asked again). -> the ply's record.
    fn push(&mut self, py: Python<'_>, uci: String) -> PyResult<PyObject> {
        let um: shakmaty::uci::UciMove = uci.parse().map_err(|_| pyo3::exceptions::PyValueError::new_err("bad uci"))?;
        let m = um.to_move(&self.pos).map_err(|_| pyo3::exceptions::PyValueError::new_err("illegal move"))?;
        let mut touched = Bitboard::from(m.to());
        if let Some(f) = m.from() { touched |= Bitboard::from(f); }
        if m.is_en_passant() { if let Some(f) = m.from() { touched |= Bitboard::from(Square::from_coords(m.to().file(), f.rank())); } }
        if let Move::Castle { king, rook } = m { touched |= Bitboard::from(king) | Bitboard::from(rook); }
        let from = m.from();
        self.pos.play_unchecked(&m);
        self.line.push(self.pos.clone());
        self.ply += 1;
        self.fact_cache.clear();
        let retracted = self.retract(touched);
        // listeners: a premise that just came true ("X leaves") wakes the findings waiting on it
        let woke = from.map(|f| self.listeners.get(&hash_of(&format!("LEAVES|{}", f))).map_or(0, |v| v.len())).unwrap_or(0);
        let live_before = self.fs.iter().filter(|f| f.stale.is_none()).count();
        self.carried = 0; self.reasked = 0;
        self.run_pass();
        let pins: Vec<Value> = self.fs.iter().filter(|f| f.stale.is_none() && f.kind == "PIN")
            .map(|f| json!({"claim": f.claim, "lo": f.lo, "hi": f.hi, "status": f.st.name()})).collect();
        let rec = json!({"ply": self.ply, "move": uci, "touched": touched.into_iter().map(|s| s.to_string()).collect::<Vec<_>>(), "retracted": retracted,
                         "carried_over": live_before, "lookups": self.carried, "re_asked": self.reasked, "live": self.fs.iter().filter(|f| f.stale.is_none()).count(),
                         "premise_listeners_woken": woke, "pins": pins, "counts": self.counts()});
        self.plies.push(rec.clone());
        let j = py.import("json")?;
        Ok(j.call_method1("loads", (rec.to_string(),))?.unbind())
    }

    /// Decide the move for the side to move (candidates, intervals, B*, budget, confirmation).
    fn decide(&mut self, py: Python<'_>) -> PyResult<PyObject> { self.decide_impl(py) }

    fn report_dict(&self, py: Python<'_>) -> PyResult<PyObject> {
        let j = py.import("json")?;
        Ok(j.call_method1("loads", (self.report().to_string(),))?.unbind())
    }
}

/// ledger(fen) -> the report of the entry pass on the position (standalone analysis: ANALYST_PLAN 39.5 item 13).
#[pyfunction]
#[pyo3(signature = (fen, budget=200))]
pub fn ledger(py: Python<'_>, fen: String, budget: u32) -> PyResult<PyObject> {
    let mut l = ReasoningLedger::new(fen, budget)?;
    l.run(py)
}

// =====================================================================================================================================
// PHASE 3: THE DECISION LAYER (ANALYST_PLAN 39.2, 39.5 items 5-10). Candidates, an interval each from the ledger of the position after it
// (push, read, pop: the roll forward's retraction by read set shares what the move did not touch), the stake of a question, the B* stop,
// the budget, the solver's confirmation.
const MATE: i64 = 1000;
const SEARCH_DEPTH: i32 = 8;               // plies of the forcing line, at most (the node cap and the budget come first)
const NODE_CAP: u32 = 400000;
const NODE_PER_CAND: u32 = 20000;            // nodes one candidate's forcing lines may use before the others get theirs                // nodes (positions rolled forward into) per decision

struct Snap { pos: Chess, line_len: usize, ply: u32, fs_len: usize, live: HashMap<u64, usize>, staled: Vec<usize> }

#[derive(Clone)]
struct Cand { uci: String, mv: Move, why: Vec<&'static str>, rule: (i64, i64), iv: (i64, i64), refined: bool, level: u8, cites: Vec<usize>, note: String }

fn gain_of(m: &Move) -> i64 { m.capture().map_or(0, val) + m.promotion().map_or(0, |r| val(r) - 1) }
fn touched_of(m: &Move) -> Bitboard {
    let mut t = Bitboard::from(m.to());
    if let Some(f) = m.from() { t |= Bitboard::from(f); }
    if m.is_en_passant() { if let Some(f) = m.from() { t |= Bitboard::from(Square::from_coords(m.to().file(), f.rank())); } }
    if let Move::Castle { king, rook } = m { t |= Bitboard::from(*king) | Bitboard::from(*rook); }
    t
}
fn attacked(b: &shakmaty::Board, s: Square, by: Color) -> bool { (b.attacks_to(s, by, b.occupied()) & b.by_color(by)).any() }

impl ReasoningLedger {
    fn enter(&mut self, m: &Move) -> Snap {
        let mut snap = Snap { pos: self.pos.clone(), line_len: self.line.len(), ply: self.ply, fs_len: self.fs.len(), live: self.live.clone(), staled: vec![] };
        let t = touched_of(m);
        self.pos.play_unchecked(m);
        self.line.push(self.pos.clone());
        self.ply += 1;
        self.fact_cache.clear();
        snap.staled = self.retract_ids(t);
        snap
    }
    fn leave(&mut self, s: Snap) {
        self.fs.truncate(s.fs_len);
        self.live = s.live;
        for id in s.staled { self.fs[id].stale = None; }
        self.pos = s.pos; self.line.truncate(s.line_len); self.ply = s.ply;
        self.fact_cache.clear();
    }

    /// 39.5 item 7: the moves the findings suggest plus every check and capture.
    fn candidates(&self, legal: &[Move]) -> Vec<Cand> {
        let me = self.pos.turn();
        let b = self.pos.board();
        let unsafe_: Vec<(Square, Vec<Square>)> = self.pieces_of(me).into_iter().filter(|(_, s)| attacked(b, *s, !me))
            .map(|(_, s)| (s, (b.attacks_to(s, !me, b.occupied()) & b.by_color(!me)).into_iter().collect())).collect();
        let mut out = vec![];
        let in_check = self.pos.is_check();
        // lines of ours through one of our pieces to one of theirs: moving the piece discovers an attack (the vocabulary's `discovery`)
        let mut discovered: HashMap<Square, Vec<Square>> = HashMap::new();
        for sl in self.pieces_of(me).into_iter().filter(|p| matches!(p.0, Role::Queen | Role::Rook | Role::Bishop)) {
            for y in self.pieces_of(!me).into_iter().map(|p| p.1).chain(b.king_of(!me)) {
                let on_line = match sl.0 { Role::Rook => sl.1.file() == y.file() || sl.1.rank() == y.rank(),
                    Role::Bishop => shakmaty::attacks::bishop_attacks(sl.1, Bitboard::EMPTY).contains(y),
                    _ => shakmaty::attacks::queen_attacks(sl.1, Bitboard::EMPTY).contains(y) };
                if !on_line { continue; }
                let btw = shakmaty::attacks::between(sl.1, y) & b.occupied();
                if btw.count() == 1 { let x = btw.first().unwrap(); if b.color_at(x) == Some(me) { discovered.entry(x).or_default().push(y); } }
            }
        }
        for m in legal {
            let mut why: Vec<&'static str> = vec![];
            if in_check { why.push("reply to check"); }
            if let Some(f) = m.from() { if discovered.contains_key(&f) && !discovered[&f].contains(&m.to()) { why.push("discovery"); } }
            let mut q = self.pos.clone(); q.play_unchecked(m);
            if m.is_capture() { why.push("capture"); }
            if m.promotion().is_some() { why.push("promotion"); }
            if q.is_check() { why.push("check"); }
            for (z, atk) in &unsafe_ {
                if m.from() == Some(*z) { why.push("moves the unsafe piece"); }
                else if m.capture().is_none() && atk.iter().any(|a| shakmaty::attacks::between(*a, *z).contains(m.to())) { why.push("blocks the attack"); }
            }
            if m.capture().is_none() && m.role() != Role::King {
                let qb = q.board();
                let landing_cheap = !(qb.attacks_to(m.to(), !me, qb.occupied()) & qb.by_color(!me)).into_iter().any(|a| qb.piece_at(a).map_or(false, |p| val(p.role) < val(m.role())));
                if landing_cheap && qb.attacks_from(m.to()) .into_iter().any(|t| qb.piece_at(t).map_or(false, |p| p.color != me && p.role != Role::King && val(p.role) > 0)) { why.push("kick"); }
            }
            if m.capture().is_none() && m.role() == Role::Pawn {
                let adv = if me == Color::White { m.to().rank() as i32 } else { 7 - m.to().rank() as i32 };
                if adv >= 5 { why.push("advanced pawn"); }
            }
            if why.is_empty() && m.role() != Role::King && m.role() != Role::Pawn {
                // a piece with no job (it guards nothing that is attacked, shields nothing) moving to a square nobody attacks
                let from = m.from().unwrap();
                let guards_attacked = unsafe_.iter().any(|(z, _)| b.attacks_from(from).contains(*z));
                let qb = q.board();
                if !guards_attacked && !attacked(qb, m.to(), !me) { why.push("jobless piece"); }
            }
            if !why.is_empty() { out.push(Cand { uci: m.to_uci(CastlingMode::Standard).to_string(), mv: m.clone(), why, rule: (0, 0), iv: (0, 0), refined: false, level: 0, cites: vec![], note: String::new() }); }
        }
        if out.is_empty() {
            out = legal.iter().map(|m| Cand { uci: m.to_uci(CastlingMode::Standard).to_string(), mv: m.clone(), why: vec!["no suggestion: every move"], rule: (0, 0), iv: (0, 0), refined: false, level: 0, cites: vec![], note: String::new() }).collect();
        }
        out
    }

    /// The rule bounds (39.2): a piece left en prise costs at most its value; a check costs at most what a reply can take; mate +-infinity.
    fn rule_interval(&self, m: &Move) -> (i64, i64) {
        let mut q = self.pos.clone(); q.play_unchecked(m);
        if q.is_checkmate() { return (MATE, MATE); }
        if q.is_stalemate() || q.is_insufficient_material() { return (0, 0); }
        let me = self.pos.turn();
        let b = q.board();
        let cap = gain_of(m);
        let cost_hi = if q.is_check() { q.legal_moves().iter().map(gain_of).max().unwrap_or(0) }
            else { Self::pieces_on(b, me).iter().filter(|(_, s)| attacked(b, *s, !me)).map(|(r, _)| val(*r)).max().unwrap_or(0) };
        let threat_hi = Self::pieces_on(b, !me).iter().filter(|(_, s)| attacked(b, *s, me)).map(|(r, _)| val(*r)).max().unwrap_or(0);
        (cap - cost_hi, cap + threat_hi)
    }

    /// THE STAND-PAT value for the side to move (not in check): it can fix its worst problem, the rest is lost. The problems are its pieces the
    /// other side attacks, each with the SAFE finding's cost interval (the effective exchange: overloaded guards, jobs) while questions remain,
    /// and the exchange law's `see` with the piece's value as the upper bound after (a valid bound, no question asked). -> (lo, hi) for the mover.
    fn stand_pat(&mut self) -> (i64, i64) {
        let t = self.pos.turn();
        let mine: Vec<Pc> = self.pieces_of(t).into_iter().filter(|(_, s)| attacked(self.pos.board(), *s, !t)).collect();
        let mut costs: Vec<(i64, i64)> = vec![];
        for p in mine {
            if self.deep && self.used < self.budget {
                let id = self.ask_safe(p, None);
                costs.push((self.fs[id].lo, self.fs[id].hi));
            } else {
                if self.deep { self.budget_hit = true; }
                let (v, _) = self.fact(&format!("(see {})", self.nm(p.1)));
                costs.push((v, val(p.0).max(v)));
            }
        }
        costs.sort_by_key(|c| std::cmp::Reverse(c.0));
        let max_hi = costs.iter().map(|c| c.1).max().unwrap_or(0);
        let second_lo = if costs.len() >= 2 { costs[1].0 } else { 0 };
        (-max_hi, -second_lo)
    }

    /// KEEP ASKING UNTIL THE COST IS KNOWN: the value for the side to move, along the FORCING lines only (a check, a capture, a promotion --
    /// in check every reply), the ledger rolled forward through each (enter / leave: retraction by read set), the stand-pat where nothing is
    /// forcing. Bounded by the node cap and the depth cap, not by a fixed depth of the line. -> ((lo, hi), the principal line)
    fn search(&mut self, d: i32) -> ((i64, i64), Vec<String>) {
        self.nodes += 1;
        let legal: Vec<Move> = self.pos.legal_moves().into_iter().collect();
        if legal.is_empty() { return (if self.pos.is_check() { (-MATE, -MATE) } else { (0, 0) }, vec![]); }
        if self.pos.is_insufficient_material() { return ((0, 0), vec![]); }
        let key = (crate::threat::zkey(&self.pos), d + if self.deep { 1000 } else { 0 });
        if let Some(v) = self.tt.get(&key) { return v.clone(); }
        let in_check = self.pos.is_check();
        let sp = if in_check { (i64::MIN / 4, i64::MIN / 4) } else { self.stand_pat() };
        if d <= 0 || self.nodes > self.cap_now {
            if self.nodes > self.cap_now && d > 0 { self.node_cap_hit = true; self.cuts += 1; }
            let r = if in_check { (self.stand_pat_in_check(), vec![]) } else { (sp, vec![]) };
            return r;
        }
        let mut cands: Vec<Move> = if in_check { legal } else {
            legal.into_iter().filter(|m| m.is_capture() || m.promotion().is_some() || { let mut q = self.pos.clone(); q.play_unchecked(m); q.is_check() }).collect() };
        cands.sort_by_key(|m| (std::cmp::Reverse(gain_of(m)), std::cmp::Reverse(val(m.role()) < 0)));
        let (mut lo, mut hi) = sp;
        let mut best: (i64, Vec<String>) = (i64::MIN, vec![]);
        let cuts0 = self.cuts;
        for m in cands {
            let g = gain_of(&m);
            let uci = m.to_uci(CastlingMode::Standard).to_string();
            let snap = self.enter(&m);
            let (ch, cpv) = self.search(d - 1);
            self.leave(snap);
            let (vlo, vhi) = ((g - ch.1).max(-MATE).min(MATE * 2), (g - ch.0).min(MATE * 2));
            lo = lo.max(vlo); hi = hi.max(vhi);
            if vlo + vhi > best.0 { best = (vlo + vhi, std::iter::once(uci).chain(cpv).collect()); }
        }
        let r = ((lo, hi.max(lo)), best.1);
        if self.cuts == cuts0 { self.tt.insert(key, r.clone()); }       // (a value cut short by the node cap is not remembered as exact)
        r
    }
    /// In check at the horizon: the replies are not explored; the mover is assumed to lose what a check costs at most (nothing).
    fn stand_pat_in_check(&self) -> (i64, i64) { (-2, 0) }

    /// The interval of a candidate: its gain minus the opponent's value of the position after it, found by rolling the ledger forward.
    fn refine(&mut self, c: &mut Cand) {
        self.deep = c.level >= 1;
        self.cap_now = (self.nodes + NODE_PER_CAND).min(NODE_CAP);
        let m = c.mv.clone();
        let g = gain_of(&m);
        let snap = self.enter(&m);
        // ITERATIVE DEEPENING: each depth completes or is thrown away (a search cut short by the node cap is never used)
        let (mut ch, mut pv) = self.search(1);
        for dd in 2..SEARCH_DEPTH {
            let cuts0 = self.cuts;
            let (c2, p2) = self.search(dd);
            if self.cuts != cuts0 { break; }
            (ch, pv) = (c2, p2);
        }
        self.leave(snap);
        c.iv = ((g - ch.1).min(MATE), (g - ch.0).min(MATE));
        c.iv.1 = c.iv.1.max(c.iv.0);
        c.refined = true; c.level += 1;
        c.note = format!("line {} {}", c.uci, pv.join(" "));
        c.cites = vec![];
    }

    fn explain(&mut self, c: &Cand) -> Vec<Value> {
        // the chain of the chosen candidate: replay its line, then read the SAFE findings of the position at its end (what is still hanging there)
        let saved = (self.used, self.carried, self.reasked, self.facts_n, self.nodes);
        self.used = 0; self.deep = true;
        let line: Vec<String> = c.note.strip_prefix("line ").unwrap_or("").split_whitespace().map(|x| x.to_string()).collect();
        let mut snaps = vec![];
        for u in &line {
            let Ok(um) = u.parse::<shakmaty::uci::UciMove>() else { break };
            let Ok(m) = um.to_move(&self.pos) else { break };
            snaps.push(self.enter(&m));
        }
        let t = self.pos.turn();
        let mut ids = vec![];
        for color in [t, !t] {
            let ps: Vec<Pc> = self.pieces_of(color).into_iter().filter(|(_, s)| attacked(self.pos.board(), *s, !color)).collect();
            for p in ps { ids.push(self.ask_safe(p, None)); }
        }
        let mut out = vec![json!({"id": 0, "type": "LINE", "claim": format!("the line the ledger rolled forward: {}", line.join(" ")), "status": "PROVEN", "lo": 0, "hi": 0})];
        let mut seen = HashSet::new();
        let mut stack = ids;
        while let Some(i) = stack.pop() {
            if !seen.insert(i) { continue; }
            out.push(self.finding_json(&self.fs[i]));
            stack.extend(self.fs[i].cites.clone());
        }
        while let Some(sn) = snaps.pop() { self.leave(sn); }
        (self.used, self.carried, self.reasked, self.facts_n, self.nodes) = saved;
        out
    }
}

impl ReasoningLedger {
    /// DECIDE the move for the side to move (ANALYST_PLAN 39.2): candidates, rule intervals, B* with stake, the budget, the solver's confirmation.
    fn decide_impl(&mut self, py: Python<'_>) -> PyResult<PyObject> {
        let t0 = std::time::Instant::now();
        self.used = 0; self.carried = 0; self.reasked = 0; self.facts_n = 0; self.skipped.clear(); self.confirmations.clear();
        self.tt.clear(); self.nodes = 0; self.cap_now = 0; self.node_cap_hit = false; self.budget_hit = false;
        let legal: Vec<Move> = self.pos.legal_moves().into_iter().collect();
        if legal.is_empty() {
            let out = json!({"move": null, "status": "PROVEN", "reason": if self.pos.is_check() { "checkmate: no legal move" } else { "stalemate: no legal move" }, "interval": [0, 0], "why": [], "questions": 0, "budget": self.budget, "nodes": 0, "facts": 0, "candidates": 0, "legal": 0, "refined": 0, "zero_stake": 0, "unanswered": [], "confirm": null, "table": [], "all": [], "chain": [], "seconds": 0.0});
            let j = py.import("json")?;
            return Ok(j.call_method1("loads", (out.to_string(),))?.unbind());
        }
        let mut cands = self.candidates(&legal);
        let n_cand = cands.len();
        // the moves nothing suggested still bound the decision: a PROVEN choice must beat their rule upper bounds too (they are refined only if their
        // interval overlaps the best's, like any candidate)
        let have: HashSet<String> = cands.iter().map(|c| c.uci.clone()).collect();
        for m in &legal {
            let u = m.to_uci(CastlingMode::Standard).to_string();
            if !have.contains(&u) { cands.push(Cand { uci: u, mv: m.clone(), why: vec!["unsuggested"], rule: (0, 0), iv: (0, 0), refined: false, level: 0, cites: vec![], note: String::new() }); }
        }
        for c in cands.iter_mut() { c.rule = self.rule_interval(&c.mv); c.iv = c.rule; }
        let mut status = "PROVEN"; let mut reason = String::from("B*: the best lower bound >= every other upper bound");
        let order = |cs: &Vec<Cand>| -> usize { (0..cs.len()).max_by_key(|&i| (cs[i].iv.0, cs[i].iv.1, cs[i].why.iter().any(|w| matches!(*w, "capture" | "check" | "promotion")), cs[i].mv.role() != Role::King, std::cmp::Reverse(i))).unwrap() };
        loop {
            let bi = order(&cands);
            let best_lo = cands[bi].iv.0;
            let others_hi = (0..cands.len()).filter(|&i| i != bi).map(|i| cands[i].iv.1).max().unwrap_or(i64::MIN);
            if best_lo >= others_hi { break; }
            // STAKE: only a candidate whose interval still overlaps the best's lower bound is worth a question; the wider, the more
            let target = (0..cands.len()).filter(|&i| cands[i].level < 2 && (cands[i].iv.1 > best_lo || i == bi))
                .max_by_key(|&i| (cands[i].iv.1 - cands[i].iv.0.max(best_lo), cands[i].iv.1, std::cmp::Reverse(i)));
            let Some(t) = target else { status = "ESTIMATED"; reason = "every overlapping candidate is as refined as the ledger gets: decided on the midpoints".into(); break };
            if self.used >= self.budget && self.budget_hit { status = "ESTIMATED"; reason = format!("budget of {} questions spent: decided on the midpoints", self.budget); break; }
            if self.nodes > NODE_CAP { status = "ESTIMATED"; reason = format!("node cap of {} spent: decided on the midpoints", NODE_CAP); break; }
            let mut c = cands[t].clone();
            self.refine(&mut c);
            cands[t] = c;
        }
        let bi = if status == "PROVEN" { order(&cands) } else {
            (0..cands.len()).max_by_key(|&i| (cands[i].iv.0 + cands[i].iv.1, cands[i].why.iter().any(|w| matches!(*w, "capture" | "check" | "promotion")), cands[i].mv.role() != Role::King, std::cmp::Reverse(i))).unwrap()
        };
        let chosen = cands[bi].clone();
        let unanswered: Vec<String> = if status == "ESTIMATED" {
            cands.iter().filter(|c| !c.refined && c.iv.1 > chosen.iv.0).map(|c| format!("{} {:?} rule [{},{}]: not refined", c.uci, c.why, c.rule.0, c.rule.1)).collect()
        } else { vec![] };
        let zero_stake = cands.iter().filter(|c| !c.refined && c.iv.1 < chosen.iv.0).count();
        let questions = self.used;
        let facts = self.facts_n;
        let chain = if chosen.refined { self.explain(&chosen) } else { vec![] };
        // CONFIRM: when the claim is a win of material (or a mate) the exact solver checks it; a refuted claim is a SURPRISE
        let mut confirm = json!(null);
        if chosen.iv.0 > 0 {
            let value = if chosen.iv.0 >= MATE / 2 { 100 } else { chosen.iv.0.min(20) };
            crate::goals::solver_nodes_reset();
            let fen = shakmaty::fen::Fen::from_position(self.pos.clone(), shakmaty::EnPassantMode::Legal).to_string();
            let res = crate::goals::forced_moves(fen, value, 4, 2_000_000);
            let mine = res.iter().find(|r| r.0 == chosen.uci);
            let ok = mine.map_or(false, |r| r.1 == "forced");
            let nodes = crate::goals::solver_nodes();
            confirm = json!({"value": value, "result": if ok { "CONFIRMED" } else { "REFUTED" }, "solver_status": mine.map(|r| r.1.clone()), "nodes": nodes,
                             "line": mine.map(|r| r.3.join(" ")), "surprise": if ok { Value::Null } else {
                                 json!({"claim": format!("{} wins >= {}", chosen.uci, chosen.iv.0), "rule": chosen.rule, "suspects": chosen.cites.len(), "note": chosen.note}) }});
        }
        let tab: Vec<Value> = { let mut v: Vec<&Cand> = cands.iter().collect(); v.sort_by_key(|c| std::cmp::Reverse((c.iv.0, c.iv.1)));
            v.into_iter().take(8).map(|c| json!({"move": c.uci, "why": c.why, "rule": [c.rule.0, c.rule.1], "interval": [c.iv.0, c.iv.1], "refined": c.refined, "note": c.note})).collect() };
        let out = json!({"move": chosen.uci, "status": status, "reason": reason, "interval": [chosen.iv.0, chosen.iv.1], "why": chosen.why, "questions": questions, "budget": self.budget, "nodes": self.nodes, "budget_hit": self.budget_hit, "node_cap_hit": self.node_cap_hit,
                         "facts": facts, "candidates": n_cand, "legal": legal.len(), "refined": cands.iter().filter(|c| c.refined).count(), "zero_stake": zero_stake,
                         "unanswered": unanswered, "confirm": confirm, "table": tab, "all": cands.iter().map(|c| json!([c.uci, c.why, [c.rule.0, c.rule.1], [c.iv.0, c.iv.1], c.refined])).collect::<Vec<_>>(), "chain": chain, "seconds": t0.elapsed().as_secs_f64()});
        let j = py.import("json")?;
        Ok(j.call_method1("loads", (out.to_string(),))?.unbind())
    }
}
