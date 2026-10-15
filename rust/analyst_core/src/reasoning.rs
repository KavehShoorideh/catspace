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
use crate::vocab;
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
    sq: Vec<Square>,          // the squares of the pieces the finding is about (SAFE: piece, attacker; RECAP: guard, captured, attacker, charge)
}

type Pc = (Role, Square);

#[derive(Clone)]
struct Threat { value: i64, victim: Color, target: Pc, attacker: Pc, mate: bool, finding: Option<usize>,
    kick: Option<(Square, Square, String, String)>, why: String, detail: String, thread: Option<(u32, String)> }      // a LEARNED threat: the move (from, to, uci, san), the words that rose, what follows

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
    threats_mode: bool,
    lean_mode: bool,
    pushes: u32,
    tr: std::cell::RefCell<Vec<String>>,
    tracing: bool,
    root_ply: u32,
    sep_flag: bool,
    suppressed: std::cell::Cell<u32>,
    lean_cap: u32,
    frame: Vec<(vocab::Term, i64, bool, String, String)>,      // the learned threat frame: (term, sign, is a change, gloss, text)
    threats_learned: bool,
    looks: u32,
    slog: Vec<String>,
    resolving: u32,
    resolved_n: u32,
    cashing: bool,            // the move about to be sized cashes a live thread (an overloaded guard): a named exception to the exchange gate
    dframe: Vec<(vocab::Term, i64, bool)>,       // the learned DEFENCE frame: (term, sign, is a change)
    answers_hand: bool,
    shallow_sizing: bool,
    prefilter_root: bool,
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
                  stale: None, circular: false, note: String::new(), facts: vec![], sq: vec![] }
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

    /// Can the piece on `from` legally take on `to` (an enemy piece there) on this BOARD, `owner` to move: a pseudo-legal capture that leaves its own king
    /// unattacked. Board only -- no Position is built, so a king left in check by the other side (a check just given) cannot make it vanish.
    fn legal_capture(board: &shakmaty::Board, owner: Color, from: Square, to: Square) -> bool {
        let Some(p) = board.piece_at(from) else { return false };
        if p.color != owner || board.by_color(owner).contains(to) { return false; }
        let reach = if p.role == Role::Pawn { shakmaty::attacks::pawn_attacks(owner, from) & board.by_color(!owner) } else { board.attacks_from(from) };
        if !reach.contains(to) { return false; }
        let mut b2 = board.clone();
        b2.discard_piece_at(from);
        b2.set_piece_at(to, p);
        match b2.king_of(owner) { Some(k) => !(b2.attacks_to(k, !owner, b2.occupied()) & b2.by_color(!owner)).any(), None => true }
    }
    fn board_after_capture(b: &Chess, a: Square, y: Square) -> Option<shakmaty::Board> {
        let mut bd = b.board().clone();
        let pc = bd.piece_at(a)?;
        bd.discard_piece_at(a);
        bd.set_piece_at(y, pc);
        Some(bd)
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
        f.sq = vec![r.1, z.1];
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
        f.sq = vec![r.1, y.1, a.1, z.1];
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
        f.sq = vec![y.1];
        let (v0, rd0) = self.fact(&self.w(gone, &format!("(see {yn})")));
        f.reads = rd0; f.facts.push((self.w(gone, &format!("(see {yn})")), v0));
        // the attackers that may legally capture
        let atk: Vec<Pc> = (b.board().attacks_to(y.1, !owner, occ) & b.board().by_color(!owner)).into_iter()
            .filter_map(|s| b.board().piece_at(s).map(|p| (p.role, s)))
            .filter(|(_, s)| Self::legal_capture(b.board(), !owner, *s, y.1)).collect();
        if atk.is_empty() {
            f.claim = format!("{}{} is not attacked", pname(y), gl);
            self.stack.pop();
            return self.add(f);
        }
        let a = *atk.iter().min_by_key(|(r, s)| (val(*r), *s as u32)).unwrap();
        f.sq.push(a.1);
        // the recapturers: each must be able to retake legally (not pinned, not blocked by check)
        let after_cap = Self::board_after_capture(&b, a.1, y.1);
        let defs: Vec<Pc> = (b.board().attacks_to(y.1, owner, occ) & b.board().by_color(owner)).into_iter()
            .filter_map(|s| b.board().piece_at(s).map(|p| (p.role, s)))
            .filter(|(_, s)| after_cap.as_ref().map_or(false, |p| Self::legal_capture(p, owner, *s, y.1))).collect();
        if let Some(k) = b.board().king_of(owner) { f.reads |= crate::vocab::exchange_reads(b.board(), k) | crate::vocab::exchange_reads(b.board(), a.1); }
        // the capture with check? (no in-between moves for them)
        let mut check = false;
        if let Some(k) = b.board().king_of(owner) {
            let t = self.w(gone, &format!("(after {} {yn} (count (and (attacks {yn}) (at {}))))", self.nm(a.1), self.nm(k)));
            let (c, rc) = self.fact(&t);
            check = c > 0; f.reads |= rc; f.facts.push((t, c));
        }
        // an IN-BETWEEN capture (made instead of retaking) is worth what ITS exchange wins, not the face value of the piece: a guarded piece taken in between is a trade
        let inb = if check { 0 } else {
            let cands: Vec<Square> = { let bb = b.board(); bb.by_color(!owner).into_iter().filter(|s| *s != a.1)
                .filter(|s| bb.piece_at(*s).map_or(false, |p| p.role != Role::King) && (bb.attacks_to(*s, owner, occ) & bb.by_color(owner)).any()).collect() };
            let mut m = 0;
            for s in cands {
                let t = self.w(gone, &format!("(see {})", self.nm(s)));
                let (v, rv) = self.fact(&t);
                f.reads |= rv; m = m.max(v);
            }
            m
        };
        let mut best_r: Option<(Pc, i64)> = None;
        let mut cites = vec![];
        // INSIDE A WHAT-IF ("with X gone") the recapturers' other jobs are asked only when the plain exchange holds the piece (v0 <= 0): a piece that already loses
        // by the exchange there is lost, the what-if is answered, stop (Kaveh on a pin: "if the knight moves, the queen dies ... It should not go any deeper"). On the
        // real board they are always asked: an overloaded recapturer is a threat of its own (f4 guarding both g5 and Ne5)
        let ask_jobs = v0 <= 0 || gone.is_none();
        let mut cost = if defs.is_empty() || !ask_jobs { v0 } else { val(y.0) };
        let mut jobbed = vec![];
        for r in defs.iter().filter(|_| ask_jobs) {
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
        let mut lows: Vec<i64> = vec![];
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
            if lo < lo_all { lo_all = lo; best = crate::plan::san(&po, &m.to_uci(CastlingMode::Standard).to_string()); }
            lows.push(lo);
            hi_all = hi_all.max(hi);
        }
        let ties = lows.iter().filter(|l| **l == lo_all).count();
        f.lo = lo_all; f.hi = hi_all;
        f.st = if bl.st == St::Unknown { St::Unknown } else { St::Proven };
        f.note = serde_json::to_string(&per).unwrap_or_default();
        f.claim = if lo_all > 0 {
            // every move costs the same: the pin is the whole story, no move is named (a tie names nothing)
            if ties > 1 { format!("{} is pinned to {}: every move of it costs >= {} ({} is then taken)", pname(x), pname(y), lo_all, pname(y)) }
            else { format!("{} is pinned to {}: moving it costs >= {} (the cheapest move is {})", pname(x), pname(y), lo_all, best) }
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
            budget, used: 0, carried: 0, reasked: 0, facts_n: 0, solver_calls: 0, stack: vec![], skipped: vec![], agenda: vec![], plies: vec![], confirmations: vec![], tt: HashMap::new(), nodes: 0, node_cap_hit: false, budget_hit: false, deep: false, cap_now: 0, cuts: 0, threats_mode: false, lean_mode: false, pushes: 0, tr: std::cell::RefCell::new(vec![]), tracing: false, root_ply: 0, sep_flag: false, suppressed: std::cell::Cell::new(0), lean_cap: LEAN_CAP, frame: vec![], threats_learned: false, looks: 0, slog: vec![], resolving: 0, resolved_n: 0, cashing: false, dframe: vec![], answers_hand: false, shallow_sizing: false, prefilter_root: true })
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
    #[pyo3(signature = (forcing="quiescence"))]
    fn decide(&mut self, py: Python<'_>, forcing: &str) -> PyResult<PyObject> { self.threats_mode = forcing == "threats" || forcing == "static"; self.lean_mode = forcing == "static"; self.decide_impl(py) }

    /// THE TRACE (Kaveh 2026-10-14: "I want to see it working"): the reasoning on this one position, step by step, in plain chess words, in the order it
    /// happens -- the threats to the side to move, the latent ones (a piece that shields another), the answers or the attack from the highest-value
    /// target, every position played out with the question that needed it, the bounds, the decision. -> a dict with the lines and the result.
    /// THE SIZER, alone (the held-out check): every move of the other side on pass(P) sized by the ledger -> [(uci, san, size, the detail)], biggest first.
    #[pyo3(signature = (all_sizes=true, against_last_mover=false))]
    fn threat_report(&mut self, py: Python<'_>, all_sizes: bool, against_last_mover: bool) -> PyResult<PyObject> {
        self.lean_mode = true; self.threats_mode = true; self.deep = true; self.threats_learned = true; self.root_ply = self.ply;
        self.used = 0; self.pushes = 0; self.looks = 0; self.facts_n = 0; self.slog.clear();
        // against the side that just MOVED (a game walked through with push): its opponent is to move, so the threats are that side's moves as they stand, no pass
        let saved = self.pos.clone();
        if against_last_mover { if let Some(b) = Self::passed(&saved) { self.pos = b; self.fact_cache.clear(); } }
        let ts = self.learned_threats_n(all_sizes);
        self.pos = saved; self.fact_cache.clear();
        let out: Vec<Value> = ts.iter().map(|t| json!({"uci": t.kick.as_ref().map(|k| k.2.clone()), "san": t.kick.as_ref().map(|k| k.3.clone()), "size": t.value,
            "target": pname(t.target), "detail": t.detail, "mate": t.mate, "thread": t.thread.as_ref().map(|x| format!("ply {}: {}", x.0, x.1))})).collect();
        let res = json!({"threats": out, "positions": self.pushes, "looks": self.looks, "questions": self.used, "facts": self.facts_n, "log": self.slog});
        let j = py.import("json")?;
        Ok(j.call_method1("loads", (res.to_string(),))?.unbind())
    }

    /// The defence frame's order of the moves of the side to move: [(uci, san)], best first (the position's own S).
    fn defence_choice(&mut self, py: Python<'_>) -> PyResult<PyObject> {
        let legal: Vec<Move> = self.pos.legal_moves().into_iter().collect();
        let ranked = self.defence_rank(&legal);
        let out: Vec<Value> = ranked.iter().take(8).map(|(m, t)| json!({"uci": m.to_uci(CastlingMode::Standard).to_string(), "san": self.san_of(m), "tuple": t})).collect();
        let j = py.import("json")?;
        Ok(j.call_method1("loads", (json!(out).to_string(),))?.unbind())
    }

    /// Size only the moves that take, check or newly attack, at the root too (a cheaper sizer; the check compares).
    fn set_prefilter_root(&mut self, on: bool) { self.prefilter_root = on; }

    /// Load the learned DEFENCE frame: [(term, sign, is_change)].
    fn set_defence_frame(&mut self, terms: Vec<(String, i64, bool)>) -> PyResult<()> {
        let mut fr = vec![];
        for (t, sign, delta) in terms {
            let p = vocab::parse_term(&t).ok_or_else(|| pyo3::exceptions::PyValueError::new_err(format!("bad term {t}")))?;
            fr.push((p, sign, delta));
        }
        self.dframe = fr;
        Ok(())
    }

    /// Load the learned threat frame: [(term, sign, is_change, gloss)].
    fn set_frame(&mut self, terms: Vec<(String, i64, bool, String)>) -> PyResult<()> {
        let mut fr = vec![];
        for (t, sign, delta, gloss) in terms {
            let p = vocab::parse_term(&t).ok_or_else(|| pyo3::exceptions::PyValueError::new_err(format!("bad term {t}")))?;
            let text = format!("{}{}", if delta { "the change in " } else { "" }, gloss);
            fr.push((p, sign, delta, gloss, text));
        }
        self.frame = fr;
        Ok(())
    }

    #[pyo3(signature = (depth=6, threats="defined", answers="frame"))]
    fn trace(&mut self, py: Python<'_>, depth: i32, threats: &str, answers: &str) -> PyResult<PyObject> {
        self.answers_hand = answers == "hand";
        self.threats_learned = threats == "defined";
        self.looks = 0;
        self.tracing = true; self.lean_mode = true; self.threats_mode = true; self.deep = true; self.lean_cap = 150;
        self.suppressed.set(0);
        self.tr.borrow_mut().clear();
        self.used = 0; self.pushes = 0; self.facts_n = 0; self.tt.clear(); self.root_ply = self.ply; self.budget_hit = false; self.skipped.clear();
        let t = self.pos.turn();
        let who = if t == Color::White { "White" } else { "Black" };
        self.say(format!("POSITION {} -- {who} to move.", shakmaty::fen::Fen::from_position(self.pos.clone(), shakmaty::EnPassantMode::Legal)));
        // LATENT THREATS: a piece of ours that shields another from a slider -- the premise "it leaves" is open (listeners)
        let b = self.pos.board().clone();
        let mut any = false;
        for a in self.pieces_of(!t).into_iter().filter(|p| matches!(p.0, Role::Queen | Role::Rook | Role::Bishop)) {
            for y in self.pieces_of(t) {
                let btw = shakmaty::attacks::between(a.1, y.1);
                let on_line = match a.0 { Role::Rook => a.1.file() == y.1.file() || a.1.rank() == y.1.rank(),
                    Role::Bishop => shakmaty::attacks::bishop_attacks(a.1, Bitboard::EMPTY).contains(y.1),
                    _ => shakmaty::attacks::queen_attacks(a.1, Bitboard::EMPTY).contains(y.1) };
                if !on_line || btw.is_empty() { continue; }
                let between = btw & b.occupied();
                if between.count() != 1 { continue; }
                let xs = between.first().unwrap();
                if b.color_at(xs) != Some(t) { continue; }
                let Some(xp) = b.piece_at(xs) else { continue };
                if xp.role == Role::King { continue; }
                let x = (xp.role, xs);
                let bi = self.ask_blocks(x, a, y);
                if self.fs[bi].st == St::Refuted { continue; }
                let pi = self.ask_pin(x, a, y, bi);
                if !any { self.say(format!("LATENT THREATS to {who} (a piece that shields another; the threat appears if it leaves):")); any = true; }
                self.say(format!("  - {}", self.fs[bi].claim));
                let mut kinds: Vec<&'static str> = vec![];
                let mut stack = self.fs[bi].cites.clone();
                while let Some(c) = stack.pop() { let k = self.fs[c].kind; if !kinds.contains(&k) { kinds.push(k); } stack.extend(self.fs[c].cites.iter().copied()); }
                kinds.sort_by_key(|k| match *k { "SAFE" => 0, "JOB" => 1, "RECAP" => 2, _ => 3 });
                self.say(format!("    {}  [from the finding chain: BLOCKS#{}{}]", self.fs[pi].claim, bi, if kinds.is_empty() { String::new() } else { format!(" <- {}", kinds.join(", ")) }));
            }
        }
        if !any { self.say(format!("LATENT THREATS to {who}: no piece shields another along a line.")); }
        // LINES READ THE OTHER WAY: our slider, our shield, their piece -- moving the shield uncovers an attack
        let mut opp = false;
        for sl in self.pieces_of(t).into_iter().filter(|p| matches!(p.0, Role::Queen | Role::Rook | Role::Bishop)) {
            for y in self.pieces_of(!t) {
                let btw = shakmaty::attacks::between(sl.1, y.1);
                let on_line = match sl.0 { Role::Rook => sl.1.file() == y.1.file() || sl.1.rank() == y.1.rank(),
                    Role::Bishop => shakmaty::attacks::bishop_attacks(sl.1, Bitboard::EMPTY).contains(y.1),
                    _ => shakmaty::attacks::queen_attacks(sl.1, Bitboard::EMPTY).contains(y.1) };
                if !on_line || btw.is_empty() { continue; }
                let between = btw & b.occupied();
                if between.count() != 1 { continue; }
                let xs = between.first().unwrap();
                if b.color_at(xs) != Some(t) { continue; }
                let Some(xp) = b.piece_at(xs) else { continue };
                if xp.role == Role::King { continue; }
                let (xn, yn) = (self.nm(xs), self.nm(y.1));
                let (w, _) = self.fact(&format!("(without (at {xn}) (see {yn}))"));
                if w <= 0 { continue; }                    // a line that uncovers nothing worth anything is not shown
                let grd = b.attacks_to(y.1, !t, b.occupied()) & b.by_color(!t);
                if !opp { self.say(format!("LATENT OPPORTUNITIES for {who} (our slider, our shield, their piece; the attack appears if the shield leaves):")); opp = true; }
                self.say(format!("  - if {} leaves, {} hits {} (guarded by [{}]): uncovering it wins {} by the exchange", pname((xp.role, xs)), pname(sl), pname(y), self.names(grd), w));
            }
        }
        if !opp { self.say(format!("LATENT OPPORTUNITIES for {who}: no line uncovers anything worth winning.")); }
        let (iv, pv) = self.lean(depth);
        let mut status = if self.sep_flag { "SEPARATED (static bounds)".to_string() } else { "ESTIMATED".to_string() };
        let first = pv.first().map(|x| x.split('|').next().unwrap_or("").to_string()).unwrap_or_default();
        let san = if first.is_empty() { String::new() } else { crate::plan::san(&self.pos, &first) };
        // PROVEN only when the exact solver confirms the claim (a win of material or a mate); the static bounds alone are not a proof
        if iv.0 > 0 && !first.is_empty() {
            let value = if iv.0 >= MATE / 2 { 100 } else { iv.0.min(20) };
            let fen = shakmaty::fen::Fen::from_position(self.pos.clone(), shakmaty::EnPassantMode::Legal).to_string();
            crate::goals::solver_nodes_reset();
            let res = crate::goals::forced_moves(fen, value, 4, 1_000_000);
            let ok = res.iter().any(|r| r.0 == first && r.1 == "forced");
            self.say(format!("SOLVER (confirms, never finds): is {san} a forced gain of {value} within 4? {} -- {} solver nodes.", if ok { "yes" } else { "not proved" }, crate::goals::solver_nodes()));
            if ok { status = "PROVEN (solver-confirmed)".to_string(); }
            else if self.sep_flag { status = "SEPARATED (static bounds; the solver did not confirm)".to_string(); }
        }
        self.say(format!("DECISION: {san} ({first}), worth {}..{} to {who}: {status}. Positions examined by the ledger: {}; static looks by the threat frame: {}; questions asked: {}.", iv.0, iv.1, self.pushes, self.looks, self.used));
        if self.suppressed.get() > 0 { self.say(format!("({} deeper steps were taken but are not printed; the positions are counted above.)", self.suppressed.get())); }
        let lines = self.tr.borrow().clone();
        self.tracing = false; self.lean_cap = LEAN_CAP;
        let out = json!({"trace": lines, "move": first, "san": san, "interval": [iv.0, iv.1], "status": status, "positions": self.pushes, "questions": self.used,
                         "facts": self.facts_n, "looks": self.looks, "learned": self.threats_learned, "line": pv});
        let j = py.import("json")?;
        Ok(j.call_method1("loads", (out.to_string(),))?.unbind())
    }

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
const TRACE_SHOW: usize = 3;               // plies of the trace printed in full; deeper steps are counted, not shown
const LEAN_CAP: u32 = 1500;                // positions one decision may roll forward into in the static-first mode
const ANSWER_BREADTH: usize = 4;            // the answers to a threat tried, best (most gain, cheapest mover) first
const MAX_RESOLVE: u32 = 6;                // straddling sizes played out per sizing
const RESOLVE_NODES: u32 = 300;            // positions one straddling size may use to be played out
const NODE_PER_CAND: u32 = 20000;            // nodes one candidate's forcing lines may use before the others get theirs                // nodes (positions rolled forward into) per decision

struct Snap { pos: Chess, line_len: usize, ply: u32, fs_len: usize, live: HashMap<u64, usize>, staled: Vec<usize> }

#[derive(Clone)]
struct Cand { uci: String, mv: Move, why: Vec<&'static str>, rule: (i64, i64), iv: (i64, i64), refined: bool, level: u8, cites: Vec<usize>, note: String }

/// A finding asked under a what-if (its key carries the square taken off the board: SAFE, JOB, RECAP "with X gone").
fn whatif(f: &Finding) -> bool { matches!(f.kind, "SAFE" | "JOB" | "RECAP") && f.key.rsplit('|').next().map_or(false, |g| !g.is_empty()) }
fn gain_of(m: &Move) -> i64 { m.capture().map_or(0, val) + m.promotion().map_or(0, |r| val(r) - 1) }
fn touched_of(m: &Move) -> Bitboard {
    let mut t = Bitboard::from(m.to());
    if let Some(f) = m.from() { t |= Bitboard::from(f); }
    if m.is_en_passant() { if let Some(f) = m.from() { t |= Bitboard::from(Square::from_coords(m.to().file(), f.rank())); } }
    if let Move::Castle { king, rook } = m { t |= Bitboard::from(*king) | Bitboard::from(*rook); }
    t
}
/// The pressure on each non-king piece of `victim`: (square, how many attackers, the cheapest attacker's value). A move raises the pressure when a piece is attacked that was
/// not, or by more, or by a cheaper piece -- a pawn kick on a rook already attacked by the queen is a threat though nothing is newly attacked.
fn pressure(b: &shakmaty::Board, victim: Color) -> HashMap<Square, (usize, i64)> {
    let mut out = HashMap::new();
    for s in b.by_color(victim) {
        let Some(p) = b.piece_at(s) else { continue };
        if p.role == Role::King { continue; }
        let atk = b.attacks_to(s, !victim, b.occupied()) & b.by_color(!victim);
        let mv = atk.into_iter().filter_map(|x| b.piece_at(x)).map(|x| val(x.role)).min().unwrap_or(99);
        out.insert(s, (atk.count(), mv));
    }
    out
}
fn raises_pressure(before: &HashMap<Square, (usize, i64)>, after: &HashMap<Square, (usize, i64)>) -> bool {
    after.iter().any(|(sq, (n, mv))| { let (bn, bmv) = before.get(sq).copied().unwrap_or((0, 99)); *n > bn || *mv < bmv })
}
fn attacked(b: &shakmaty::Board, s: Square, by: Color) -> bool { (b.attacks_to(s, by, b.occupied()) & b.by_color(by)).any() }

impl ReasoningLedger {
    fn enter(&mut self, m: &Move) -> Snap {
        let mut snap = Snap { pos: self.pos.clone(), line_len: self.line.len(), ply: self.ply, fs_len: self.fs.len(), live: self.live.clone(), staled: vec![] };
        let t = touched_of(m);
        self.pushes += 1;                                   // every position rolled forward into is counted
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


    // ---- PHASE 3b: THE THREAT LIST DRIVES THE LINE (Kaveh 2026-10-14: "start with the highest-value threat") -------------------------------
    fn mate_in_one(b: &Chess) -> bool { b.legal_moves().iter().any(|m| { let mut q = b.clone(); q.play_unchecked(m); q.is_checkmate() }) }
    fn passed(b: &Chess) -> Option<Chess> {
        let mut st = b.clone().into_setup(shakmaty::EnPassantMode::Legal);
        st.turn = !b.turn();
        crate::vocab::relaxed(st)
    }

    /// THE THREAT SOURCE, pluggable (Kaveh 2026-10-14: the threat is to be learned, not written): the ranked threats to the mover and by the mover. Today it
    /// is `threat_list` (the attacked pieces by the effective exchange, the mate in one); a learned one replaces this function.
    fn ranked_threats(&mut self) -> Vec<Threat> {
        if self.threats_learned && !self.pos.is_check() {
            let mut out = self.learned_threats();
            out.extend(self.threat_list().into_iter().filter(|th| th.mate));          // a mate in one is a rule of the game, not learned
            out.sort_by_key(|th| (std::cmp::Reverse(th.value), th.victim == self.pos.turn()));
            return out;
        }
        self.threat_list()
    }

    // ---- THE LEARNED THREAT (Kaveh 2026-10-14: "I want it to learn what a threat is"): the PASS TEST. A threat against the mover S is a move that the other
    // side O could make if S passed (pass(P) = the same board, O to move), ranked by the frame learned from watching Stockfish (watch.py threat_frame)
    // against the null move (S passed, nothing happened). The words are read on the board after the move and on P itself, both with S to move.
    fn eval_frame(&self, pos: &Chess) -> Vec<i64> {
        let lone = !pos.turn();
        self.frame.iter().map(|f| vocab::num(&f.0, pos, lone) as i64).collect()
    }
    fn frame_tuple(&self, after: &[i64], base: &[i64]) -> Vec<i64> {
        self.frame.iter().enumerate().map(|(k, f)| f.1 * if f.2 { after[k] - base[k] } else { after[k] }).collect()
    }
    /// The moves of the other side on pass(base) that the frame puts above the null, biggest first (base: S to move, not in check). Every move is a static look.
    fn learned_rank(&mut self, base: &Chess) -> Vec<(Move, Vec<i64>)> {
        let Some(pb) = Self::passed(base) else { return vec![] };
        let raw_base = self.eval_frame(base);
        let null = self.frame_tuple(&raw_base, &raw_base);
        let mut out = vec![];
        for m in pb.legal_moves() {
            let mut q = pb.clone(); q.play_unchecked(&m);
            self.looks += 1;
            let raw = self.eval_frame(&q);
            let t = self.frame_tuple(&raw, &raw_base);
            if t > null { out.push((m, t)); }
        }
        out.sort_by(|a, b| b.1.cmp(&a.1));
        out
    }
    /// What the frame reads in the move: each word, its value on P (null) and after the move.
    fn frame_words(&self, base: &Chess, pb: &Chess, m: &Move) -> String {
        let raw_base = self.eval_frame(base);
        let mut q = pb.clone(); q.play_unchecked(m);
        let raw = self.eval_frame(&q);
        self.frame.iter().enumerate().map(|(k, f)| format!("{}: {} -> {}", f.4, raw_base[k], raw[k])).collect::<Vec<_>>().join("; ")
    }
    /// The size of a kick, by the ledger's own questions: the move is played on the board (O to move), S's stand-pat there (its problems, the exchange with the
    /// jobs, trapped or not) read -- S's loss, at least. None when it comes to nothing.
    fn size_kick(&mut self, m: &Move, words: String) -> Option<Threat> {
        let o = self.pos.turn();
        let s = !o;
        let (from, to) = (m.from()?, m.to());
        let uci = m.to_uci(CastlingMode::Standard).to_string();
        let san = self.san_of(m);
        let g = gain_of(m);
        // the reason of a plain capture: what it takes, who attacked it (before), who guarded it
        let cap_note = m.capture().map(|r| {
            let b = self.pos.board();
            let guards = b.attacks_to(to, s, b.occupied()) & b.by_color(s);
            let atks = b.attacks_to(to, o, b.occupied()) & b.by_color(o);
            format!("takes {} on {}; attacked by [{}], guarded by [{}]", pname((r, to)), to, self.names(atks), self.names(guards))
        });
        // A CAPTURE IS SIZED BY THE EXCHANGE ON ITS SQUARE FIRST (the exchange law; Kaveh: "the pawn is protected ... that's not something they can immediately
        // win"): what it nets with the recaptures, the recapturers' other jobs charged (an overloaded guard). If that does not pay, it is no threat -- nothing is
        // played out -- unless a named exception holds after it: with the capturer taken back, a piece of S still loses more than before (a discovery, a removed
        // defender), a check by another piece (a discovered check), or a live overload thread it cashes (a frame that names this capture)
        // The same for a move that does not capture but lands where its piece is simply taken (Nc7+ into the queen's guard): the attack it makes is
        // answered by taking it, unless a named exception holds
        let cashes = std::mem::take(&mut self.cashing);
        let gated = !m.is_en_passant() && !m.is_promotion() && !m.is_castle() && m.role() != Role::King && !cashes;
        let mut exch: Option<i64> = None;
        let mut before: Vec<(Square, i64)> = vec![];
        if gated {
            let tn = self.nm(to);
            let (loss, _) = self.fact(&format!("(after {} {tn} (see {tn}))", self.nm(from)));
            exch = match m.capture() {
                Some(cr) => {
                    let plain = val(cr) - loss;
                    let sid = self.ask_safe((cr, to), None);
                    let charged = if self.fs[sid].sq.get(1) == Some(&from) { self.fs[sid].hi } else { i64::MIN };
                    Some(plain.max(charged))
                }
                None if loss > 0 => Some(-loss),
                None => None,
            };
            if exch.map_or(false, |e| e <= 0) {
                for p in self.pieces_of(s).into_iter().filter(|p| p.0 != Role::King && p.1 != to) {
                    let (v, _) = self.fact(&format!("(see {})", self.nm(p.1)));
                    before.push((p.1, v));
                }
            }
        }
        let snap = self.enter(m);
        let mate = self.pos.is_checkmate();
        // (2) after the move the other side has a mate in one whatever S replies (every reply read, one look each) -> MATE
        let mut mating = mate;
        if !mate {
            let replies: Vec<Move> = self.pos.legal_moves().into_iter().collect();
            if !replies.is_empty() {
                let mut all_mated = true;
                for r in &replies {
                    let mut q = self.pos.clone(); q.play_unchecked(r);
                    self.looks += 1;
                    if !Self::mate_in_one(&q) { all_mated = false; break; }
                }
                mating = all_mated;
            }
        }
        if let Some(e) = exch {
            if !mating && e <= 0 {
                let tn = self.nm(to);
                let disc_check = (self.pos.checkers() & !Bitboard::from(to)).any();
                let mut newly = false;
                for (sq, v0) in &before {
                    let (v1, _) = self.fact(&format!("(without (at {tn}) (see {}))", self.nm(*sq)));
                    if v1 > (*v0).max(0) { newly = true; break; }
                }
                if !disc_check && !newly {
                    self.slog.push(if m.is_capture() { format!("{san}: the exchange on {to} nets {e} for them and nothing else falls -> not a threat") }
                                   else { format!("{san}: the piece on {to} is simply taken (it loses {} by the exchange) and nothing else falls -> not a threat", -e) });
                    self.leave(snap);
                    return None;
                }
            }
        }
        let mut sp = if mate { (-MATE, -MATE) } else if mating { (-MATE, -MATE) } else if self.pos.is_check() { self.check_value() } else { self.stand_pat() };
        // A SIZE THAT STRADDLES THE THRESHOLD (lo < 1 <= hi) is not read at its hopeful end: the line is played out (forcing lines, as B* does) until the bounds separate
        let mut resolved = String::new();
        if !mating && g - sp.1 < 1 && g - sp.0 >= 1 && self.resolving == 0 && self.resolved_n < MAX_RESOLVE {
            self.resolving += 1; self.resolved_n += 1;
            let saved = (self.threats_mode, self.cap_now);
            let was_deep = self.deep;
            self.deep = false;                       // the line is played out on the exchange law and the rules, no questions (the budget is not drained)
            self.threats_mode = false;
            self.cap_now = self.nodes + RESOLVE_NODES;
            let (v, _) = self.search(6);
            self.threats_mode = saved.0; self.cap_now = saved.1; self.deep = was_deep;
            self.resolving -= 1;
            sp = v;
            resolved = if g - sp.1 < 1 && g - sp.0 >= 1 { format!(" [undecided after playing the line out: size {}..{}]", (g - sp.1).max(0), g - sp.0) } else { " [resolved by playing the line out]".to_string() };
        }
        let mate = mating;
        let mut target: Option<Pc> = m.capture().map(|r| (r, to));
        let mut attacker: Pc = (m.role(), to);
        let mut detail = String::new();
        if target.is_none() && !mate {
            let ps: Vec<Pc> = self.pieces_of(s).into_iter().filter(|(_, q)| attacked(self.pos.board(), *q, o)).collect();
            let mut best: Option<(i64, Pc)> = None;
            for p in ps {
                let id = self.ask_safe(p, None);
                let h = self.fs[id].hi;
                if best.map_or(true, |(bh, _)| h > bh) { best = Some((h, p)); }
            }
            target = best.map(|b| b.1);
        }
        if let Some(tg) = target {
            let b = self.pos.board().clone();
            let atk = (b.attacks_to(tg.1, o, b.occupied()) & b.by_color(o)).into_iter().filter_map(|q| b.piece_at(q).map(|x| (x.role, q))).min_by_key(|(r, q)| (val(*r), *q as u32));
            if let Some(a) = atk { attacker = a; }
            if !mate && m.capture().is_none() {
                let mv = self.moves_for(s);
                let fx = self.fixers(tg, &mv);
                let grd = b.attacks_to(tg.1, s, b.occupied()) & b.by_color(s);
                let atks = b.attacks_to(tg.1, o, b.occupied()) & b.by_color(o);
                detail = format!("{} attacked by [{}], guarded by [{}]; {}", pname(tg), self.names(atks), self.names(grd),
                    if fx.is_empty() { "TRAPPED: no move saves it (no safe square, no safe capture of the attacker, no safe block or guard)".to_string() }
                    else { format!("saved by: {}", fx.iter().take(4).map(|x| self.san_of(x)).collect::<Vec<_>>().join(", ")) });
            }
        }
        if detail.is_empty() {
            detail = if mate { "a mate (no reply of S stops it)".to_string() } else if let Some(c) = &cap_note { format!("{c}; S's static value afterwards {}..{}", sp.0, sp.1) } else { format!("S's static value afterwards {}..{}", sp.0, sp.1) };
        } else if let Some(c) = &cap_note { detail = format!("{c}; {detail}"); }
        detail.push_str(&resolved);
        self.leave(snap);
        // a size that cannot be excluded is not read at the hopeful end: undecided after the line is played out, it counts as a threat of size 1 (at least)
        let value = if mate || -sp.1 >= MATE / 2 { MATE } else if g - sp.1 < 1 && g - sp.0 >= 1 { 1 } else { g - sp.1 };
        self.slog.push(format!("{san}: gain {g}, S's static value there {}..{} -> size {}", sp.0, sp.1, value));
        if value <= 0 { return None; }
        let tgt = target.unwrap_or((Role::King, self.pos.board().king_of(s).unwrap_or(Square::A1)));
        Some(Threat { value, victim: s, target: tgt, attacker, mate: value >= MATE, finding: None, kick: Some((from, to, uci, san)), why: words, detail, thread: None })
    }
    fn size_kick_from(&mut self, pb: &Chess, m: &Move, words: String) -> Option<Threat> {
        let saved = self.pos.clone();
        self.pos = pb.clone(); self.fact_cache.clear();
        let r = self.size_kick(m, words);
        self.pos = saved; self.fact_cache.clear();
        r
    }
    /// THE SIZING (reasoned, the ledger's own questions) of the moves the frame orders. At the ROOT every move of the other side is sized; deeper, the frame's
    /// top 5 plus all its captures and checks. A threat is a move after which S loses material with its best defence, or is mated (Kaveh: "Threats are
    /// material loss. Or mate."); the size is the loss. `all`: return every sized threat, not the top 3 the ledger works on.
    fn learned_threats_n(&mut self, all: bool) -> Vec<Threat> {
        let base = self.pos.clone();
        if base.is_check() { return vec![]; }
        let Some(pb) = Self::passed(&base) else { return vec![] };
        self.resolved_n = if self.shallow_sizing { MAX_RESOLVE } else { 0 };
        let root = self.ply == self.root_ply && !self.shallow_sizing && !self.prefilter_root;
        let moves: Vec<Move> = if root {
            self.looks += pb.legal_moves().len() as u32;
            pb.legal_moves().into_iter().collect()
        } else {
            // a DEFINITIONAL pre-filter (no learned order): the move takes something, gives check, or leaves some piece of ours attacked that was not before
            // (a discovery included); every other move cannot cost S material by itself
            let o = pb.turn();
            let s_ = !o;
            let was = pressure(base.board(), s_);
            let mut v = vec![];
            for m in pb.legal_moves() {
                self.looks += 1;
                let mut q = pb.clone(); q.play_unchecked(&m);
                if m.is_capture() || q.is_check() || raises_pressure(&was, &pressure(q.board(), s_)) { v.push(m); }
            }
            v
        };
        // THE LIVE THREADS: findings still standing that say a piece of S can be won (SAFE, lo >= 1) or that a guard is overloaded (RECAP > 0); the move that cashes
        // each in is sized first (and kept in the list even where the pre-filter would drop it)
        let o_ = pb.turn();
        let mut cash: HashMap<String, (u32, String)> = HashMap::new();
        // a finding asked inside a what-if ("with f3 gone") stays inside it: it is never a live thread on the real board
        // an overloaded guard (RECAP) is a thread only when the exchange it belongs to is lost by it -- no other piece can take back instead (h6 guarded by
        // Ng4 AND g7: g7 retakes, Ng4 keeps its job)
        let lost_by: HashSet<usize> = self.fs.iter().filter(|g| g.kind == "SAFE" && g.stale.is_none() && g.hi >= 1).flat_map(|g| g.cites.iter().copied()).collect();
        for f in self.fs.iter().filter(|f| f.stale.is_none() && f.lo >= 1 && !whatif(f) && (f.kind != "RECAP" || lost_by.contains(&f.id))) {
            let hit: Option<(Square, Square)> = match f.kind {
                "SAFE" if f.sq.len() == 2 && base.board().color_at(f.sq[0]) == Some(!o_) => Some((f.sq[1], f.sq[0])),
                "RECAP" if f.sq.len() == 4 && base.board().color_at(f.sq[1]) == Some(!o_) => Some((f.sq[2], f.sq[1])),
                _ => None,
            };
            if let Some((from, to)) = hit {
                for m in pb.legal_moves() {
                    // the capture the finding names, or -- an overloaded guard -- the capture of the OTHER thing it guarded
                    let charge = f.kind == "RECAP" && m.to() == f.sq[3] && m.is_capture();
                    if (m.from() == Some(from) && m.to() == to) || charge { cash.entry(m.to_uci(CastlingMode::Standard).to_string()).or_insert((f.ply, f.claim.clone())); }
                }
            }
        }
        let mut moves = moves;
        for m in pb.legal_moves() {
            let k = m.to_uci(CastlingMode::Standard).to_string();
            if cash.contains_key(&k) && !moves.iter().any(|x| x.to_uci(CastlingMode::Standard).to_string() == k) { moves.push(m); }
        }
        moves.sort_by_key(|m| !cash.contains_key(&m.to_uci(CastlingMode::Standard).to_string()));
        let mut out = vec![];
        for m in moves {
            self.cashing = cash.contains_key(&m.to_uci(CastlingMode::Standard).to_string());
            if let Some(mut th) = self.size_kick_from(&pb, &m, String::new()) {
                if let Some(c) = cash.get(&m.to_uci(CastlingMode::Standard).to_string()) { th.thread = Some(c.clone()); }
                out.push(th);
            }
        }
        // ONE THREAT PER THING THAT WOULD BE LOST (a target piece, or mate), shown with the move that cashes it in: a quiet move that only leaves a standing threat standing
        // is not a threat of its own
        let mut best: Vec<Threat> = vec![];
        let mut counts: Vec<usize> = vec![];
        for th in out {
            let key = (th.mate, if th.mate { Square::A1 } else { th.target.1 });
            let takes = |t: &Threat| t.kick.as_ref().map_or(false, |k| k.1 == t.target.1 && !t.mate);
            if let Some(i) = best.iter().position(|b| (b.mate, if b.mate { Square::A1 } else { b.target.1 }) == key) {
                counts[i] += 1;
                let better = (takes(&th), th.value) > (takes(&best[i]), best[i].value);
                if better { best[i] = th; }
            } else { best.push(th); counts.push(0); }
        }
        for (th, c) in best.iter_mut().zip(counts) { if c > 0 { th.detail.push_str(&format!(" (+{c} other move(s) only leave it standing)")); } }
        let mut out = best;
        out.sort_by_key(|th| std::cmp::Reverse(th.value));
        if !all { out.truncate(3); }
        out
    }
    fn learned_threats(&mut self) -> Vec<Threat> { self.learned_threats_n(false) }
    /// THE LEARNED DEFENCE (watch.py defence_frame): S's moves ordered by the frame, read on the TRUE board after the move with us = S; ties -- the move that also threatens
    /// them (a check, a capture, an attack on a piece) first.
    fn defence_rank(&mut self, legal: &[Move]) -> Vec<(Move, Vec<i64>)> {
        let s_ = self.pos.turn();
        let lone = !s_;
        let base: Vec<i64> = self.dframe.iter().map(|f| vocab::num(&f.0, &self.pos, lone) as i64).collect();
        let mut out: Vec<(Move, Vec<i64>, bool)> = vec![];
        for m in legal {
            let mut q = self.pos.clone(); q.play_unchecked(m);
            self.looks += 1;
            let raw: Vec<i64> = self.dframe.iter().map(|f| vocab::num(&f.0, &q, lone) as i64).collect();
            let t: Vec<i64> = self.dframe.iter().enumerate().map(|(k, f)| f.1 * if f.2 { raw[k] - base[k] } else { raw[k] }).collect();
            let qb = q.board();
            let both = q.is_check() || m.is_capture() || qb.attacks_from(m.to()).into_iter().any(|x| qb.piece_at(x).map_or(false, |p| p.color != s_ && p.role != Role::King && val(p.role) > 0));
            out.push((m.clone(), t, both));
        }
        out.sort_by(|a, b| b.1.cmp(&a.1).then(b.2.cmp(&a.2)));
        out.into_iter().map(|x| (x.0, x.1)).collect()
    }
    /// After our answer `a` (O to move there), does the kick still work, and for how much? One push for the answer, one for the kick.
    fn reask(&mut self, a: &Move, kick_uci: &str) -> i64 {
        let snap = self.enter(a);
        let r = match self.pos.legal_moves().into_iter().find(|k| k.to_uci(CastlingMode::Standard).to_string() == kick_uci) {
            Some(k) => self.size_kick(&k, String::new()).map_or(0, |t| t.value),
            None => 0,
        };
        self.leave(snap);
        r
    }
    /// Branch (b) with the learned threat: our moves ordered by the biggest threat each CREATES -- the frame on pass(P+m) -- checks (a position the frame
    /// cannot read: the checked side may not pass) after those. -> (move, the follow-up it threatens)
    /// THE HIGHER FRAME THAT NEEDS A GUARD GONE (Kaveh 2026-10-14: "if we need to deflect somebody or get rid of some attacker, then we should consider moves
    /// that don't win but deflect them" -- once the lower frames are exhausted). A piece of theirs that we attack and that holds now (its exchange loses
    /// nothing) is held by a guard with a JOB: without the guard it would lose `lo`. That frame proposes the moves that get the guard off its post: take
    /// the guard, attack it, or offer it a capture that pulls it off (after it takes there, the piece it held loses more). Each frame is one group, highest
    /// value first; the moves are weighed like any others.
    fn deflections(&mut self, legal: &[Move]) -> Vec<(String, Vec<Move>)> {
        let us = self.pos.turn();
        let b = self.pos.board().clone();
        let mut frames: Vec<(i64, Pc, Pc)> = vec![];                         // (what the piece would lose without the guard, guard, piece)
        for z in self.pieces_of(!us).into_iter().filter(|z| z.0 != Role::King && attacked(&b, z.1, us)) {
            let (v0, _) = self.fact(&format!("(see {})", self.nm(z.1)));
            if v0 > 0 { continue; }                                            // already lost: a lower frame (the capture) has it
            let guards: Vec<Pc> = (b.attacks_to(z.1, !us, b.occupied()) & b.by_color(!us)).into_iter()
                .filter_map(|q| b.piece_at(q).map(|p| (p.role, q))).filter(|g| g.0 != Role::King).collect();
            for r in guards {
                let j = self.ask_job(r, z, None);
                if self.fs[j].lo > 0 { frames.push((self.fs[j].lo, r, z)); }
            }
        }
        frames.sort_by(|a, b| b.0.cmp(&a.0));
        let mut out = vec![];
        for (lo, r, z) in frames.into_iter().take(3) {
            let mut mv: Vec<(i64, Move)> = vec![];
            for m in legal {
                let to = m.to();
                if to == z.1 { continue; }                                     // taking the held piece itself is the lower frame (its exchange), not a deflection
                let takes = to == r.1 && m.is_capture();
                let mut q = self.pos.clone(); q.play_unchecked(m);
                let qb = q.board();
                let pokes = !takes && qb.attacks_from(to).contains(r.1) && qb.piece_at(to).map_or(false, |p| p.color == us) && !b.attacks_from(to).contains(r.1)
                    && !(b.attacks_to(r.1, us, b.occupied()) & b.by_color(us)).contains(m.from().unwrap_or(to));
                let lures = !takes && Self::legal_capture(qb, !us, r.1, to) && {
                    let snap = self.enter(m);
                    let (v1, _) = self.fact(&format!("(after {} {} (see {}))", self.nm(r.1), self.nm(to), self.nm(z.1)));
                    self.leave(snap);
                    v1 > 0
                };
                if takes || pokes || lures { mv.push((if takes { 2 } else if lures { 1 } else { 0 }, m.clone())); }
            }
            if mv.is_empty() { continue; }
            mv.sort_by(|a, b| b.0.cmp(&a.0).then(gain_of(&b.1).cmp(&gain_of(&a.1))));
            out.push((format!("b: deflect {} (it guards {}: without it {} loses {})", pname(r), pname(z), pname(z), lo), mv.into_iter().map(|x| x.1).collect()));
        }
        out
    }
    fn created_moves(&mut self, legal: &[Move]) -> Vec<(Move, String)> {
        // Branch (b): our moves ordered by what each is WORTH -- the gain now plus the biggest threat it leaves them that they cannot answer: the static value the ledger
        // computes after one push (gain - their stand-pat with the fixers; in check, their replies). Executing an open threat (a capture that wins material) is therefore
        // ranked by what it wins, and Qxa8+ = 5 now + the rook on b5 still standing. The threat each CREATES (sized at the root) breaks ties. A capture that wins material
        // by the definition is never dropped by the breadth cap.
        let t = self.pos.turn();
        let was = pressure(self.pos.board(), !t);
        let mut cands: Vec<(i64, Move)> = vec![];
        for m in legal {
            let mut r = self.pos.clone(); r.play_unchecked(m);
            self.looks += 1;
            if m.is_capture() || r.is_check() || raises_pressure(&was, &pressure(r.board(), !t)) { cands.push((gain_of(m) * 100, m.clone())); }
        }
        cands.sort_by(|a, b| b.0.cmp(&a.0));
        // THE POKE (Kaveh: "the first step in creating a threat is poking something ... what can we attack in a way that we cannot be attacked ourselves?"):
        // the highest-value piece of theirs a move newly attacks, counted only when the poking piece is not simply taken where it lands (a check pokes the king)
        let mut poked: Vec<(i64, Move)> = vec![];
        let mut taken: HashSet<String> = HashSet::new();          // pokes that are simply taken: they create nothing (no poke, no threat left)
        for (_, m) in &cands {
            let mut r = self.pos.clone(); r.play_unchecked(m);
            let tn = self.nm(m.to());
            let (loss, _) = self.fact(&format!("(after {} {tn} (see {tn}))", self.nm(m.from().unwrap_or(m.to()))));
            let now = pressure(r.board(), !t);
            let mut pk = if r.is_check() { val(Role::King) } else { 0 };
            for (sq, (n, mv)) in &now {
                let (bn, bmv) = was.get(sq).copied().unwrap_or((0, 99));
                if *n > bn || *mv < bmv { pk = pk.max(r.board().piece_at(*sq).map_or(0, |p| val(p.role))); }
            }
            // a poke that is simply taken creates nothing (Bxf3 "pokes" Qd1 and is taken back). A move that does not win but deflects a guard is not a poke:
            // it comes from the higher frame that needs the guard gone (deflections), weighed only once the pokes are exhausted
            if loss > 0 { pk = 0; taken.insert(m.to_uci(CastlingMode::Standard).to_string()); }
            poked.push((pk, m.clone()));
        }
        // the safest biggest pokes are valued first (the gain breaks ties)
        poked.sort_by(|a, b| b.0.cmp(&a.0).then(gain_of(&b.1).cmp(&gain_of(&a.1))));
        // the worth of each (one push each, at most 14): what it wins for sure (lo: they can take nothing back), then the poke, then the hopeful end
        let mut worth: Vec<(i64, i64, i64, Move, String)> = vec![];       // (lo, hi, poke, move, note)
        for (pk, m) in poked.into_iter().take(14) {
            let g = gain_of(&m);
            let snap = self.enter(&m);
            let v = if self.pos.is_checkmate() { (-MATE, -MATE) } else if self.pos.legal_moves().is_empty() { (0, 0) } else if self.pos.is_check() { self.check_value() } else { self.stand_pat() };
            self.leave(snap);
            let note = if pk > 0 && pk < val(Role::King) { format!("worth {}..{} now, pokes a {}-point piece", (g - v.1).max(-MATE), g - v.0, pk) } else { format!("worth {}..{} now", (g - v.1).max(-MATE), g - v.0) };
            worth.push(((g - v.1).max(-MATE), (g - v.0).min(2 * MATE), pk, m, note));
        }
        worth.sort_by(|a, b| b.0.cmp(&a.0).then(b.2.cmp(&a.2)).then(b.1.cmp(&a.1)));
        let mut out: Vec<(Move, String)> = vec![];
        let top: Vec<(i64, i64, i64, Move, String)> = worth.iter().take(ANSWER_BREADTH).cloned().collect();
        if self.ply == self.root_ply {
            // ties at the top are broken by the threat each creates (sized, shallow)
            let mut keyed: Vec<(i64, i64, i64, Move, String)> = vec![];
            for (lo, hi, _, m, note) in &top {
                let mut r = self.pos.clone(); r.play_unchecked(m);
                let mut created = 0;
                let mut cnote = note.clone();
                if !r.is_check() && !r.legal_moves().is_empty() && !taken.contains(&m.to_uci(CastlingMode::Standard).to_string()) {
                    let saved = std::mem::replace(&mut self.pos, r);
                    self.fact_cache.clear();
                    self.shallow_sizing = true;
                    let ts = self.learned_threats_n(false);
                    self.shallow_sizing = false;
                    self.pos = saved; self.fact_cache.clear();
                    if let Some(th) = ts.first() { created = th.value; cnote = format!("{note}; leaves {} (worth {})", th.kick.as_ref().map(|x| x.3.clone()).unwrap_or_default(), th.value); }
                }
                keyed.push((*lo, created.min(99), *hi, m.clone(), cnote));
            }
            // what it wins for sure first (nothing of ours can be taken), then the threat it leaves them, then the hopeful end
            keyed.sort_by(|a, b| b.0.cmp(&a.0).then(b.1.cmp(&a.1)).then(b.2.cmp(&a.2)));
            out = keyed.into_iter().map(|x| (x.3, x.4)).collect();
        } else {
            out = top.iter().map(|x| (x.3.clone(), x.4.clone())).collect();
        }
        // a capture that wins material by the definition is never dropped by the breadth cap
        for (lo, _, _, m, note) in worth.iter().skip(ANSWER_BREADTH) {
            if m.is_capture() && *lo >= 1 && !out.iter().any(|x| x.0.to_uci(CastlingMode::Standard) == m.to_uci(CastlingMode::Standard)) { out.push((m.clone(), note.clone())); }
        }
        out
    }

    /// Every piece the other side attacks, with what it wins by the effective exchange (the SAFE finding while questions remain, else the exchange
    /// law's see), and the mate threats (a mate in one for either side, as if the other passed): sorted by value, highest first.
    fn threat_list(&mut self) -> Vec<Threat> {
        let t = self.pos.turn();
        let mut out: Vec<Threat> = vec![];
        for color in [t, !t] {
            let ps: Vec<Pc> = self.pieces_of(color).into_iter().filter(|(_, s)| attacked(self.pos.board(), *s, !color)).collect();
            for p in ps {
                let b = self.pos.board();
                let att = (b.attacks_to(p.1, !color, b.occupied()) & b.by_color(!color)).into_iter()
                    .filter_map(|s| b.piece_at(s).map(|q| (q.role, s))).min_by_key(|(r, s)| (val(*r), *s as u32));
                let Some(att) = att else { continue };
                let mut fid = None;
                let value = if self.deep && self.used < self.budget {
                    let id = self.ask_safe(p, None); fid = Some(id); self.fs[id].lo
                } else {
                    if self.deep { self.budget_hit = true; }
                    let (v, _) = self.fact(&format!("(see {})", self.nm(p.1))); v
                };
                if value > 0 { out.push(Threat { value, victim: color, target: p, attacker: att, mate: false, finding: fid, kick: None, why: String::new(), detail: String::new(), thread: None }); }
            }
        }
        // mate threats count as infinite: ours (a mate in one for the mover) and theirs (a mate in one if the mover passed)
        if Self::mate_in_one(&self.pos) { out.push(Threat { value: MATE, victim: !t, target: (Role::King, self.pos.board().king_of(!t).unwrap_or(Square::A1)), attacker: (Role::Queen, Square::A1), mate: true, finding: None, kick: None, why: String::new(), detail: String::new(), thread: None }); }
        if !self.pos.is_check() {
            if let Some(q) = Self::passed(&self.pos) { if Self::mate_in_one(&q) {
                out.push(Threat { value: MATE, victim: t, target: (Role::King, self.pos.board().king_of(t).unwrap_or(Square::A1)), attacker: (Role::Queen, Square::A1), mate: true, finding: None, kick: None, why: String::new(), detail: String::new(), thread: None }); } }
        }
        out.sort_by_key(|th| (std::cmp::Reverse(th.value), th.victim == t));
        out
    }

    /// The answers to the highest-value threat (39.7 repairs from the finding's read set): against the mover's piece -- move it, guard it, block the
    /// line, capture the attacker, or counter with a check / a capture at least as big; against the other side's piece -- execute it (take the
    /// target), or raise it (add an attacker, take a guard). A mate threat: the moves after which it is gone / the mating moves.
    /// The legal moves of `owner` in the position (the mover's, or -- the other side passed -- the other's).
    fn moves_for(&self, owner: Color) -> Vec<Move> {
        if self.pos.turn() == owner { return self.pos.legal_moves().into_iter().collect(); }
        let mut st = self.pos.clone().into_setup(shakmaty::EnPassantMode::Legal);
        st.turn = owner;
        match crate::vocab::relaxed(st) { Some(p) if p.turn() == owner => p.legal_moves().into_iter().collect(), _ => vec![] }
    }
    /// What the exchange takes back from the piece that goes from `f` to `t`, there (`see` read on the board after the move): 0 = it is safe where it lands.
    fn landing_loss(&mut self, f: Square, t: Square) -> i64 {
        let (fnm, tnm) = (self.nm(f), self.nm(t));
        self.fact(&format!("(after {fnm} {tnm} (see {tnm}))")).0
    }
    /// SAFE ON THE NEW SQUARE (39.7): the moves among `moves` that really fix the problem on piece `p` -- it moves to a square where it is safe, or the
    /// attacker is taken, or the line is blocked, or a guard is added -- each only if the mover is itself safe where it lands (or the capture pays for the
    /// loss) and the exchange on the piece's square (its new square if it moved) is then nothing. No fixer: the piece is TRAPPED, its problem is whole.
    fn fixers(&mut self, p: Pc, moves: &[Move]) -> Vec<Move> {
        let owner = self.pos.board().color_at(p.1).unwrap_or(Color::White);
        let b = self.pos.board().clone();
        let att = (b.attacks_to(p.1, !owner, b.occupied()) & b.by_color(!owner)).into_iter()
            .filter_map(|s| b.piece_at(s).map(|q| (q.role, s))).min_by_key(|(r, s)| (val(*r), *s as u32)).map(|x| x.1);
        let between = att.map(|a| shakmaty::attacks::between(a, p.1)).unwrap_or(Bitboard::EMPTY);
        let mut out = vec![];
        for m in moves {
            let Some(f) = m.from() else { continue };
            let t = m.to();
            let mut relevant = f == p.1 || Some(t) == att || between.contains(t);
            if !relevant && f != p.1 {
                let mut q = self.pos.clone(); q.play_unchecked(m);
                relevant = q.board().attacks_from(t).contains(p.1) && q.board().piece_at(t).map_or(false, |x| x.color == owner);
            }
            if !relevant { continue; }
            let loss = self.landing_loss(f, t);
            if loss > 0 && !m.capture().map_or(false, |r| val(r) >= loss) { continue; }      // it would only be lost there
            // the piece itself moves: what it loses where it lands, net of what it takes there (a trade saves it: Rxd1+ Kxd1)
            let after = if f == p.1 { (loss - m.capture().map_or(0, val)).max(0) } else { let (fnm, tnm, pn) = (self.nm(f), self.nm(t), self.nm(p.1)); self.fact(&format!("(after {fnm} {tnm} (see {pn}))")).0 };
            if after == 0 { out.push(m.clone()); }
        }
        out
    }

    fn answers(&mut self, top: &Threat, legal: &[Move]) -> Vec<Move> {
        let t = self.pos.turn();
        let mut out = vec![];
        if top.mate {
            for m in legal {
                let mut q = self.pos.clone(); q.play_unchecked(m);
                if top.victim == t { if !Self::mate_in_one(&q) { out.push(m.clone()); } } else if q.is_checkmate() { out.push(m.clone()); }
            }
            return out;
        }
        if let (Some((_, _, kuci, _)), true) = (top.kick.clone(), top.victim == t) {
            if !self.answers_hand && !self.dframe.is_empty() {
                // the answers = S's moves ordered by the LEARNED defence frame; the top few are sized by the definition (re-ask the threat after each: does the loss go away?)
                let ranked = self.defence_rank(legal);
                let mut scored: Vec<(i64, usize, Move)> = vec![];
                for (i, (m, _)) in ranked.into_iter().take(6).enumerate() {
                    let after = self.reask(&m, &kuci);
                    scored.push((after, i, m));
                }
                scored.sort_by_key(|x| (x.0, x.1));
                return scored.into_iter().map(|x| x.2).collect();
            }
        }
        if let (Some((kf, kt, kuci, _)), true) = (top.kick.clone(), top.victim == t) {
            // a LEARNED threat (the other side's next move): pre-empt (move the target), prevent (touch the kicker's squares), make it not work -- each answer
            // checked by re-asking THAT threat after it
            let tgt = top.target.1;
            let mut cands: Vec<Move> = vec![];
            for m in legal {
                let mut q = self.pos.clone(); q.play_unchecked(m);
                let qb = q.board();
                let touches = m.from() == Some(tgt) || m.to() == kf || m.to() == kt || qb.attacks_from(m.to()).contains(kt) && qb.piece_at(m.to()).map_or(false, |p| p.color == t)
                    || (qb.attacks_from(m.to()).contains(tgt) && qb.piece_at(m.to()).map_or(false, |p| p.color == t) && m.from() != Some(tgt))
                    || q.is_check() || m.capture().map_or(false, |r| val(r) >= top.value);
                if touches { cands.push(m.clone()); }
            }
            let mut scored: Vec<(i64, bool, i64, Move)> = vec![];
            for m in cands {
                let after = self.reask(&m, &kuci);
                if after < top.value {
                    let mut q = self.pos.clone(); q.play_unchecked(&m);
                    let qb = q.board();
                    let both = q.is_check() || m.is_capture() || qb.attacks_from(m.to()).into_iter().any(|x| qb.piece_at(x).map_or(false, |p| p.color != t && p.role != Role::King && val(p.role) > 0));
                    scored.push((after, both, gain_of(&m), m));
                }
            }
            scored.sort_by(|a, b| a.0.cmp(&b.0).then(b.1.cmp(&a.1)).then(b.2.cmp(&a.2)));
            return scored.into_iter().map(|x| x.3).collect();
        }
        let (target, att) = (top.target.1, top.attacker.1);
        let b = self.pos.board().clone();
        let guards_of_target: Bitboard = b.attacks_to(target, !top.victim, b.occupied()) & b.by_color(!top.victim);
        if top.victim == t {
            // against us: the safe fixes first (move the piece to a safe square, take the attacker, block, guard), then the counters with tempo
            let fixes = self.fixers(top.target, legal);
            let ukeys: HashSet<String> = fixes.iter().map(|m| m.to_uci(CastlingMode::Standard).to_string()).collect();
            let mut counters = vec![];
            for m in legal {
                if ukeys.contains(&m.to_uci(CastlingMode::Standard).to_string()) { continue; }
                let mut q = self.pos.clone(); q.play_unchecked(m);
                if q.is_check() || m.capture().map_or(false, |r| val(r) >= top.value) { counters.push(m.clone()); }      // a counter with tempo
            }
            // safe fixes first, then the counters; inside each: those that ALSO threaten them first (a check, a capture, an attack on a piece), then by gain
            let t0 = t;
            let both = |this: &Self, m: &Move| { let mut q = this.pos.clone(); q.play_unchecked(m); let qb = q.board();
                q.is_check() || m.is_capture() || qb.attacks_from(m.to()).into_iter().any(|x| qb.piece_at(x).map_or(false, |p| p.color != t0 && p.role != Role::King && val(p.role) > 0)) };
            let mut fx = out; fx.extend(Vec::<Move>::new());
            let mut fixes2: Vec<Move> = Vec::new();
            for m in legal { if ukeys.contains(&m.to_uci(CastlingMode::Standard).to_string()) { fixes2.push(m.clone()); } }
            fixes2.sort_by_key(|m| (std::cmp::Reverse(both(self, m)), std::cmp::Reverse(gain_of(m))));
            counters.sort_by_key(|m| (std::cmp::Reverse(both(self, m)), std::cmp::Reverse(gain_of(m))));
            fx.extend(fixes2); fx.extend(counters);
            return fx;
        }
        for m in legal {
            let mut q = self.pos.clone(); q.play_unchecked(m);
            let qb = q.board();
            // execute it (take the target), raise it (add an attacker), take a guard
            let keep = m.to() == target || (qb.piece_at(m.to()).map_or(false, |p| p.color == t) && qb.attacks_from(m.to()).contains(target) && m.from() != Some(target)) || (guards_of_target.contains(m.to()) && m.capture().is_some());
            let _ = att;
            if keep { out.push(m.clone()); }
        }
        out
    }

    /// BRANCH (b), no threat to us: create a threat to them, starting from the highest-value target -- their king (checks), then queen, rooks,
    /// minor pieces, pawns; for a target: execute (take it), add an attacker, take a guard, attack with tempo. The first target with any such
    /// move supplies the candidates (the next is tried only when none is found). -> (the moves, "b: <target>")
    fn attacks(&self, legal: &[Move], only: Option<Pc>) -> Vec<(Pc, Vec<Move>)> {
        let t = self.pos.turn();
        let b = self.pos.board();
        let mut targets: Vec<Pc> = self.pieces_of(!t);
        if let Some(k) = b.king_of(!t) { targets.push((Role::King, k)); }
        targets.sort_by_key(|p| (std::cmp::Reverse(val(p.0)), p.1 as u32));
        if let Some(o) = only { targets = vec![o]; }
        let mut groups = vec![];
        // LINES READ BOTH WAYS: one of our pieces shielding another of ours' line to one of theirs -- moving the shield off the line uncovers the attack
        let mut shields: HashMap<Square, Vec<(Square, Square)>> = HashMap::new();          // shield square -> [(their piece, our slider)]
        for sl in self.pieces_of(t).into_iter().filter(|p| matches!(p.0, Role::Queen | Role::Rook | Role::Bishop)) {
            for y in self.pieces_of(!t).into_iter().map(|p| p.1).chain(b.king_of(!t)) {
                let on_line = match sl.0 { Role::Rook => sl.1.file() == y.file() || sl.1.rank() == y.rank(),
                    Role::Bishop => shakmaty::attacks::bishop_attacks(sl.1, Bitboard::EMPTY).contains(y),
                    _ => shakmaty::attacks::queen_attacks(sl.1, Bitboard::EMPTY).contains(y) };
                if !on_line { continue; }
                let btw = shakmaty::attacks::between(sl.1, y) & b.occupied();
                if btw.count() == 1 { let x = btw.first().unwrap(); if b.color_at(x) == Some(t) { shields.entry(x).or_default().push((y, sl.1)); } }
            }
        }
        for tg in targets {
            let guards: Bitboard = b.attacks_to(tg.1, !t, b.occupied()) & b.by_color(!t);
            let mut out = vec![];
            for m in legal {
                let mut q = self.pos.clone(); q.play_unchecked(m);
                let qb = q.board();
                let check = q.is_check();
                let hit = if tg.0 == Role::King { check } else {
                    m.to() == tg.1                                              // execute: take it
                    || (m.capture().is_none() && qb.piece_at(m.to()).map_or(false, |p| p.color == t) && qb.attacks_from(m.to()).contains(tg.1))   // add an attacker
                    || (guards.contains(m.to()) && m.capture().is_some())       // remove a guard
                    || (check && qb.attacks_from(m.to()).contains(tg.1))        // attack it with tempo
                    || m.from().map_or(false, |f| shields.get(&f).map_or(false, |v| v.iter().any(|(y, sl)| *y == tg.1 && !shakmaty::attacks::between(*sl, *y).contains(m.to()) && m.to() != *sl)))   // uncover an attack
                } || (tg.0 == Role::King && m.from().map_or(false, |f| shields.get(&f).map_or(false, |v| v.iter().any(|(y, sl)| *y == tg.1 && !shakmaty::attacks::between(*sl, *y).contains(m.to())))));
                if hit { out.push(m.clone()); }
            }
            groups.push((tg, out));
        }
        groups
    }

    /// THE STAND-PAT value for the side to move (not in check): it can fix its worst problem, the rest is lost. The problems are its pieces the
    /// other side attacks, each with the SAFE finding's cost interval (the effective exchange: overloaded guards, jobs) while questions remain,
    /// and the exchange law's `see` with the piece's value as the upper bound after (a valid bound, no question asked). -> (lo, hi) for the mover.
    fn stand_pat(&mut self) -> (i64, i64) {
        let t = self.pos.turn();
        let mine: Vec<Pc> = self.pieces_of(t).into_iter().filter(|(_, s)| attacked(self.pos.board(), *s, !t)).collect();
        let mut costs: Vec<(i64, i64, bool)> = vec![];                // (lo, hi, fixable by a move that is safe where it lands)
        let moves = if self.lean_mode && !mine.is_empty() { self.moves_for(t) } else { vec![] };
        for p in mine {
            let (lo, hi) = if self.deep && self.used < self.budget {
                let id = self.ask_safe(p, None);
                (self.fs[id].lo, self.fs[id].hi)
            } else {
                if self.deep { self.budget_hit = true; }
                let (v, _) = self.fact(&format!("(see {})", self.nm(p.1)));
                (v, val(p.0).max(v))
            };
            let fixable = if self.lean_mode && hi > 0 { !self.fixers(p, &moves).is_empty() } else { true };
            costs.push((lo, hi, fixable));
        }
        // the mover fixes ONE problem; the rest is lost. An unfixable (trapped) piece is lost in full.
        let (pess, opt) = if self.lean_mode {
            let big_hi = costs.iter().filter(|c| c.2).enumerate().max_by_key(|(_, c)| c.1).map(|(i, _)| i);
            let big_lo = costs.iter().filter(|c| c.2).enumerate().max_by_key(|(_, c)| c.0).map(|(i, _)| i);
            let fx: Vec<&(i64, i64, bool)> = costs.iter().filter(|c| c.2).collect();
            let pess = costs.iter().filter(|c| !c.2).map(|c| c.1).chain(fx.iter().enumerate().filter(|(i, _)| Some(*i) != big_hi).map(|(_, c)| c.1)).max().unwrap_or(0);
            let opt = costs.iter().filter(|c| !c.2).map(|c| c.0).chain(fx.iter().enumerate().filter(|(i, _)| Some(*i) != big_lo).map(|(_, c)| c.0)).max().unwrap_or(0);
            (pess, opt)
        } else {
            costs.sort_by_key(|c| std::cmp::Reverse(c.0));
            (costs.iter().map(|c| c.1).max().unwrap_or(0), if costs.len() >= 2 { costs[1].0 } else { 0 })
        };
        if self.lean_mode {
            // the mover may instead EXECUTE an open threat of its own (the other side's piece it wins), then pay for its worst problem
            let theirs: Vec<Pc> = self.pieces_of(!t).into_iter().filter(|(_, s)| attacked(self.pos.board(), *s, t)).collect();
            let (mut v_lo, mut v_hi) = (0i64, 0i64);
            for p in theirs {
                let (lo_p, hi_p) = if self.deep && self.used < self.budget { let id = self.ask_safe(p, None); (self.fs[id].lo, self.fs[id].hi) }
                    else { let (v, _) = self.fact(&format!("(see {})", self.nm(p.1))); (v, val(p.0).max(v)) };
                v_lo = v_lo.max(lo_p); v_hi = v_hi.max(hi_p);
            }
            // (no piece of theirs to win = no such option: standing pat is then only the problems)
            return ((-pess).max(if v_hi > 0 { v_lo - pess } else { i64::MIN }), if v_hi > 0 { (-opt).max(v_hi) } else { -opt });
        }
        (-pess, -opt)
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
        let mut label = String::from("reply to check");
        let mut cands: Vec<Move> = if in_check { legal } else if self.threats_mode {
            let tl = self.ranked_threats();
            let t = self.pos.turn();
            if let Some(top) = tl.iter().find(|th| th.victim == t && th.value > 0).cloned() {
                // (a) the highest-value threat to us: its answers, those that also threaten them first
                label = format!("a: {} {}", if top.mate { "mate threat".to_string() } else { format!("threat {} on {}", top.value, pname(top.target)) }, "");
                let mut ans = self.answers(&top, &legal);
                ans.sort_by_key(|m| { let mut q = self.pos.clone(); q.play_unchecked(m); std::cmp::Reverse(q.is_check() || m.is_capture() || { let qb = q.board(); qb.attacks_from(m.to()).into_iter().any(|x| qb.piece_at(x).map_or(false, |p| p.color != t && val(p.role) > 0)) }) });
                ans
            } else {
                // (b) none: create a threat to them, highest-value target first
                // an existing threat of ours is executed or raised first (the highest-value one); else a threat is created, king first
                let mine = tl.iter().find(|th| th.victim != t && th.value > 0 && !th.mate).map(|th| th.target);
                let mut gs = match mine { Some(tg) => self.attacks(&legal, Some(tg)), None => vec![] };
                if gs.is_empty() { gs = self.attacks(&legal, None); }
                match gs.into_iter().find(|g| !g.1.is_empty()) { Some((tg, c)) => { label = format!("b: {}", pname(tg)); c } None => vec![] }
            }
        } else {
            legal.into_iter().filter(|m| m.is_capture() || m.promotion().is_some() || { let mut q = self.pos.clone(); q.play_unchecked(m); q.is_check() }).collect() };
        if !(self.threats_mode && !in_check) { cands.sort_by_key(|m| (std::cmp::Reverse(gain_of(m)), val(m.role()))); }
        else { let keep_order = cands.clone(); let mut idx: Vec<usize> = (0..keep_order.len()).collect(); idx.sort_by_key(|&i| (std::cmp::Reverse(gain_of(&keep_order[i])), i)); cands = idx.into_iter().map(|i| keep_order[i].clone()).collect(); }
        if self.threats_mode && !in_check && cands.len() > ANSWER_BREADTH { cands.truncate(ANSWER_BREADTH); }
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
            if vlo + vhi > best.0 { best = (vlo + vhi, std::iter::once(format!("{uci}|{label}")).chain(cpv).collect()); }
        }
        let r = ((lo, hi.max(lo)), best.1);
        if self.cuts == cuts0 { self.tt.insert(key, r.clone()); }       // (a value cut short by the node cap is not remembered as exact)
        r
    }
    /// In check at the horizon: the replies are not explored; the mover is assumed to lose what a check costs at most (nothing).
    fn stand_pat_in_check(&self) -> (i64, i64) { (-2, 0) }

    /// The value, for the side in check, of the position as it stands (no move played): its replies, statically -- what a reply takes, less what the
    /// exchange on the square it lands on takes back (`see` after it). A reply that takes the checker is the usual one.
    fn check_value(&mut self) -> (i64, i64) {
        let replies: Vec<Move> = self.pos.legal_moves().into_iter().collect();
        let (mut lo, mut hi) = (i64::MIN, i64::MIN);
        for r in &replies {
            let (tn, fnm) = (self.nm(r.to()), self.nm(r.from().unwrap()));
            let (loss, _) = self.fact(&format!("(after {fnm} {tn} (see {tn}))"));
            let cap = gain_of(r);
            hi = hi.max(cap); lo = lo.max(cap - loss);
        }
        if replies.is_empty() { return (0, 0); }
        let (mut lo, mut hi) = (lo.max(-MATE), hi.max(lo).max(0));
        if self.lean_mode {
            // the check takes the tempo: a threat against the side in check that no reply addresses (moves the piece, takes the attacker, blocks) is lost
            let t = self.pos.turn();
            let tl = self.ranked_threats();
            let (mut t_max, mut t_unaddr) = (0i64, 0i64);
            for th in tl.iter().filter(|th| th.victim == t && th.value > 0 && !th.mate).cloned().collect::<Vec<_>>() {
                t_max = t_max.max(th.value);
                // addressed only by a reply that is itself safe where it lands (the rook that "escapes" into a capture is not saved)
                if self.fixers(th.target, &replies).is_empty() { t_unaddr = t_unaddr.max(th.value); }
            }
            lo -= t_max; hi -= t_unaddr;
        }
        (lo, hi.max(lo))
    }


    fn say(&self, s: String) {
        if self.tracing {
            let depth = self.ply.saturating_sub(self.root_ply) as usize;
            if depth > TRACE_SHOW { self.suppressed.set(self.suppressed.get() + 1); return; }
            self.tr.borrow_mut().push(format!("{}{s}", "  ".repeat(depth)));
        }
    }
    fn san_of(&self, m: &Move) -> String { crate::plan::san(&self.pos, &m.to_uci(CastlingMode::Standard).to_string()) }
    fn names(&self, bb: Bitboard) -> String {
        let b = self.pos.board();
        let v: Vec<String> = bb.into_iter().filter_map(|s| b.piece_at(s).map(|p| pname((p.role, s)))).collect();
        if v.is_empty() { "nothing".into() } else { v.join(", ") }
    }
    /// A threat in plain words: the piece, who attacks it, who guards it, the effective exchange with the jobs charged -> what it loses.
    fn describe_threat(&self, th: &Threat) -> String {
        if let (Some((_, _, _, san)), Some((pl, cl))) = (&th.kick, &th.thread) {
            return format!("THREAD carried from ply {}: {} -> cashed by {} ({}) would lose {}", pl, cl, san, th.detail, th.value);
        }
        if let Some((_, _, _, san)) = &th.kick {
            return format!("next move {}:{} {} -> {} would lose {}", san, if th.why.is_empty() { String::new() } else { format!(" the frame reads [{}];", th.why) }, if th.detail.is_empty() { pname(th.target) } else { th.detail.clone() }, if th.victim == self.pos.turn() { "we" } else { "they" }, th.value);
        }
        if th.mate { return if th.victim == self.pos.turn() { "a mate in one against us if we passed".into() } else { "a mate in one for us".into() }; }
        let b = self.pos.board();
        let atk = b.attacks_to(th.target.1, !th.victim, b.occupied()) & b.by_color(!th.victim);
        let grd = b.attacks_to(th.target.1, th.victim, b.occupied()) & b.by_color(th.victim);
        let how = th.finding.map(|i| self.fs[i].claim.clone()).unwrap_or_else(|| format!("the exchange law: loses {}", th.value));
        format!("{} attacked by [{}], guarded by [{}]; effective exchange: {} -> {} {}", pname(th.target), self.names(atk), self.names(grd), how, if th.victim == self.pos.turn() { "we lose" } else { "they lose" }, th.value)
    }
    fn answer_kind(&self, top: &Threat, m: &Move) -> &'static str {
        let mut q = self.pos.clone(); q.play_unchecked(m);
        let between = shakmaty::attacks::between(top.attacker.1, top.target.1);
        if top.mate { return if top.victim == self.pos.turn() { "removes the mate threat" } else { "mates" }; }
        if top.victim == self.pos.turn() {
            if m.from() == Some(top.target.1) { "moves the piece" } else if m.to() == top.attacker.1 { "takes the attacker" }
            else if between.contains(m.to()) { "blocks the line" } else if q.is_check() { "counters with a check (tempo)" }
            else if m.capture().map_or(false, |r| val(r) >= top.value) { "counters with a capture at least as big" } else { "guards it" }
        } else if m.to() == top.target.1 { "executes the capture" } else { "raises the threat" }
    }

    /// STATIC FIRST, B* AT EVERY STEP (Kaveh 2026-10-14: "if it's taking so long, we're doing something wrong"). The value of the side to move: its
    /// stand-pat is static (the threat list, the effective exchange -- no move played). The answers (branch a / b) are each valued by ONE position
    /// (the move pushed, the static stand-pat of what follows, popped); then only the best is looked into further, and the second only while its
    /// interval overlaps the best's lower bound -- a line stops as soon as the bounds separate.
    fn lean(&mut self, d: i32) -> ((i64, i64), Vec<String>) {
        let legal: Vec<Move> = self.pos.legal_moves().into_iter().collect();
        let t = self.pos.turn();
        let who = if t == Color::White { "White" } else { "Black" };
        if legal.is_empty() { self.say(if self.pos.is_check() { format!("{who} is checkmated.") } else { format!("{who} has no move: stalemate.") }); return (if self.pos.is_check() { (-MATE, -MATE) } else { (0, 0) }, vec![]); }
        if self.pos.is_insufficient_material() { return ((0, 0), vec![]); }
        let key = (crate::threat::zkey(&self.pos), d + if self.deep { 1000 } else { 0 } + 5000);
        if let Some(v) = self.tt.get(&key) { self.say("(this position was already examined)".into()); return v.clone(); }
        let in_check = self.pos.is_check();
        let sp = if in_check { (i64::MIN / 4, i64::MIN / 4) } else { self.stand_pat() };
        if d <= 0 || self.pushes > self.lean_cap { self.say("depth limit.".into()); return (if in_check { self.stand_pat_in_check() } else { sp }, vec![]); }
        let mut label = String::from("reply to check");
        let mut top_us: Option<Threat> = None;
        // the groups of moves to try, in order: (a) / in check one group; (b) one group per target, the next only if this one brings nothing
        let mut groups: Vec<(String, Vec<Move>)> = vec![];
        if in_check {
            self.say(format!("{who} is in check: every legal reply is an answer."));
            groups.push(("reply to check".into(), legal));
        } else {
            let tl = self.ranked_threats();
            let ours: Vec<&Threat> = tl.iter().filter(|th| th.victim == t && th.value > 0).collect();
            if ours.is_empty() { self.say(format!("THREATS TO {who}: none above the margin.")); }
            else {
                self.say(format!("THREATS TO {who}, highest first:"));
                let ours_c: Vec<Threat> = ours.iter().map(|th| (*th).clone()).collect();
                let mv = self.moves_for(t);
                for th in &ours_c {
                    let tail = if th.mate || th.kick.is_some() { String::new() } else {
                        let fx = self.fixers(th.target, &mv);
                        if fx.is_empty() { " -- TRAPPED: no move saves it (no safe square, no safe capture of the attacker, no safe block or guard)".to_string() }
                        else { format!(" -- saved by: {}", fx.iter().take(4).map(|m| self.san_of(m)).collect::<Vec<_>>().join(", ")) } };
                    self.say(format!("  - {}{}", self.describe_threat(th), tail));
                }
            }
            if let Some(top) = tl.iter().find(|th| th.victim == t && th.value > 0).cloned() {
                label = format!("a: {}", if top.mate { "mate threat".to_string() } else { format!("threat {} on {}", top.value, pname(top.target)) });
                self.say(format!("BRANCH (a): answer the biggest threat, {}.", if top.mate { "a mate".to_string() } else { format!("{} ({})", pname(top.target), top.value) }));
                top_us = Some(top.clone());
                groups.push((label.clone(), self.answers(&top, &legal)));
            } else if self.threats_learned {
                // (b) with the learned threat: nothing threatens us; our moves ordered by the biggest threat each CREATES (the frame on pass(P+m))
                self.say(format!("BRANCH (b): nothing threatens {who}; our moves are ordered by the biggest threat each creates (the threat DEFINED -- material lost or mate -- and sized by the ledger on the position after it)."));
                let made = self.created_moves(&legal);
                let notes: Vec<String> = made.iter().map(|(m, f)| format!("{} threatens {}", self.san_of(m), f)).collect();
                self.say(format!("  creates: {}", if notes.is_empty() { "no move creates a threat above the null".to_string() } else { notes.join("; ") }));
                groups.push(("b: creates a threat".to_string(), made.into_iter().map(|x| x.0).collect()));
                // the higher frames, reached only when nothing above beats standing pat: a guard to deflect
                for g in self.deflections(&legal) { groups.push(g); }
            } else {
                let mine = tl.iter().find(|th| th.victim != t && th.value > 0 && !th.mate).map(|th| th.target);
                if let Some(m) = &mine {
                    self.say(format!("BRANCH (b): nothing threatens {who}; an open threat of ours on {} is executed or raised first.", pname(*m)));
                    for (tg, mv) in self.attacks(&legal, Some(*m)) { groups.push((format!("b: {} (an open threat of ours)", pname(tg)), mv)); }
                }
                else { self.say(format!("BRANCH (b): nothing threatens {who}; attacking, from the highest-value target down.")); }
                for (tg, mv) in self.attacks(&legal, None) { if !groups.iter().any(|g| g.0.starts_with(&format!("b: {}", pname(tg)))) { groups.push((format!("b: {}", pname(tg)), mv)); } }
            }
        }
        if groups.iter().all(|g| g.1.is_empty()) { self.say("no answer / no attacking move found: the position is quiet; its static value stands.".into()); self.sep_flag = true; return (sp, vec![]); }
        struct C { m: Move, g: i64, iv: (i64, i64), pv: Vec<String>, open: bool, san: String, label: String }
        let mut cs: Vec<C> = vec![];
        let mut valued: HashSet<String> = HashSet::new();
        let sp_lo = if in_check { i64::MIN / 4 } else { sp.0 };
        for (glabel, mut cands) in groups {
            if cands.is_empty() { self.say(format!("{glabel}: no way to execute, raise or create a threat against it -- abandoned, the next target.")); continue; }
            if top_us.is_none() && !glabel.starts_with("b: creates") { cands.sort_by_key(|m| (std::cmp::Reverse(gain_of(m)), val(m.role()))); }
            // a move already valued (under another target) is not pushed again
            cands.retain(|m| !valued.contains(&m.to_uci(CastlingMode::Standard).to_string()));
            if cands.is_empty() { self.say(format!("{glabel}: its moves were already weighed under another target.")); continue; }
            if !in_check && cands.len() > ANSWER_BREADTH && !glabel.starts_with("b: creates") { cands.truncate(ANSWER_BREADTH); }
            self.say(format!("{} -- {} move(s) to weigh: {}", glabel, cands.len(), cands.iter().map(|m| self.san_of(m)).collect::<Vec<_>>().join(", ")));
            let mut group_best = i64::MIN;
            for m in cands {
                valued.insert(m.to_uci(CastlingMode::Standard).to_string());
                let g = gain_of(&m);
                let san = self.san_of(&m);
                let kind = top_us.as_ref().map(|th| self.answer_kind(th, &m)).unwrap_or("");
                let snap = self.enter(&m);
                self.say(format!("play {san}{} [question: what do they hold afterwards? -- position {} examined]", if kind.is_empty() { String::new() } else { format!(" ({kind})") }, self.pushes));
                let ch = if self.pos.is_checkmate() { (-MATE, -MATE) } else if self.pos.legal_moves().len() == 0 { (0, 0) }
                         else if self.pos.is_check() { self.check_value() } else { self.stand_pat() };
                self.say(format!("   statically, the side to move there is worth {}..{}; so {san} is worth {}..{} to us (gain {g} now)", ch.0.max(-MATE), ch.1.min(MATE), (g - ch.1).max(-MATE), (g - ch.0).min(2 * MATE)));
                self.leave(snap);
                let mut c = C { iv: ((g - ch.1).max(-MATE), (g - ch.0).min(2 * MATE)), m, g, pv: vec![], open: true, san, label: glabel.clone() };
                c.iv.1 = c.iv.1.max(c.iv.0);
                group_best = group_best.max(c.iv.0);
                cs.push(c);
            }
            if in_check || top_us.is_some() { break; }
            if group_best > sp_lo { self.say(format!("{}: a threat that improves on standing pat ({} > {}) -- no need to look at lesser targets.", glabel, group_best, sp_lo)); break; }
            self.say(format!("{}: nothing here beats standing pat ({} <= {}) -- abandoned, the next target.", glabel, group_best, sp_lo));
        }
        let label = cs.iter().max_by_key(|c| (c.iv.0, c.iv.1)).map(|c| c.label.clone()).unwrap_or(label);
        let (mut lo, mut hi) = if in_check { (i64::MIN / 4, i64::MIN / 4) } else { sp };
        let mut separated = false;
        loop {
            let bi = (0..cs.len()).max_by_key(|&i| (cs[i].iv.0, cs[i].iv.1)).unwrap();
            let best_lo = cs[bi].iv.0.max(lo);
            let others_hi = (0..cs.len()).filter(|&i| i != bi).map(|i| cs[i].iv.1).max().unwrap_or(i64::MIN).max(if in_check { i64::MIN } else { hi });
            if best_lo >= others_hi { separated = true; self.say(format!("B*: {} worth {}..{} and nothing else can exceed {} -- the bounds separate, stop.", cs[bi].san, cs[bi].iv.0, cs[bi].iv.1, if others_hi == i64::MIN { "it".to_string() } else { others_hi.to_string() })); break; }
            let target = (0..cs.len()).filter(|&i| cs[i].open && (i == bi || cs[i].iv.1 > best_lo)).max_by_key(|&i| (cs[i].iv.1 - cs[i].iv.0.max(best_lo), cs[i].iv.1));
            let Some(ti) = target else { self.say("B*: nothing left to look into; the intervals overlap.".into()); break };
            if self.pushes > self.lean_cap { self.say("position budget spent.".into()); break; }
            cs[ti].open = false;
            let m = cs[ti].m.clone();
            self.say(format!("B*: {} {} -- looking further into {}.", if ti == bi { "the best answer is still open" } else { "its interval still overlaps the best's lower bound" }, format!("[{}..{}]", cs[ti].iv.0, cs[ti].iv.1), cs[ti].san));
            let snap = self.enter(&m);
            self.say(format!("play {} to see their answer [question: is the line stable? -- position {} examined]", cs[ti].san, self.pushes));
            let (ch, pv) = self.lean(d - 1);
            self.leave(snap);
            let g = cs[ti].g;
            cs[ti].iv = ((g - ch.1).max(-MATE), (g - ch.0).min(2 * MATE));
            cs[ti].iv.1 = cs[ti].iv.1.max(cs[ti].iv.0);
            cs[ti].pv = pv;
            self.say(format!("{} now worth {}..{}.", cs[ti].san, cs[ti].iv.0, cs[ti].iv.1));
        }
        self.sep_flag = separated;
        let mut best: (i64, Vec<String>) = (i64::MIN, vec![]);
        for c in &cs {
            lo = lo.max(c.iv.0); hi = hi.max(c.iv.1);
            if c.iv.0 + c.iv.1 > best.0 { best = (c.iv.0 + c.iv.1, std::iter::once(format!("{}|{}", c.m.to_uci(CastlingMode::Standard), c.label)).chain(c.pv.clone()).collect()); }
        }
        let r = ((lo, hi.max(lo)), best.1);
        let _ = &label;
        self.say(format!("{who}: value {}..{}.", r.0 .0, r.0 .1));
        self.tt.insert(key, r.clone());
        r
    }

    /// The interval of a candidate: its gain minus the opponent's value of the position after it, found by rolling the ledger forward.
    fn refine(&mut self, c: &mut Cand) {
        if self.lean_mode {
            let m = c.mv.clone();
            let g = gain_of(&m);
            self.deep = c.level >= 1;
            let snap = self.enter(&m);
            let (ch, pv) = if c.level == 0 {
                (if self.pos.is_checkmate() { (-MATE, -MATE) } else if self.pos.is_check() { self.check_value() } else { self.stand_pat() }, vec![])
            } else { self.lean(SEARCH_DEPTH - 1) };
            self.leave(snap);
            c.iv = ((g - ch.1).max(-MATE).min(MATE), (g - ch.0).min(MATE));
            c.iv.1 = c.iv.1.max(c.iv.0);
            c.refined = true; c.level += 1;
            c.note = format!("line {}", std::iter::once(format!("{}|candidate", c.uci)).chain(pv.into_iter()).collect::<Vec<_>>().join("; "));
            return;
        }
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
        c.note = format!("line {}", std::iter::once(format!("{}|candidate", c.uci)).chain(pv.into_iter()).collect::<Vec<_>>().join("; "));
        c.cites = vec![];
    }

    fn explain(&mut self, c: &Cand) -> Vec<Value> {
        // the chain of the chosen candidate: replay its line, then read the SAFE findings of the position at its end (what is still hanging there)
        let saved = (self.used, self.carried, self.reasked, self.facts_n, self.nodes);
        let saved_pushes = self.pushes;
        self.used = 0; self.deep = true;
        let line_full: Vec<String> = c.note.strip_prefix("line ").unwrap_or("").split("; ").map(|x| x.to_string()).filter(|x| !x.is_empty()).collect();
        let line: Vec<String> = line_full.iter().map(|x| x.split('|').next().unwrap_or("").to_string()).collect();
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
        let mut out = vec![json!({"id": 0, "type": "LINE", "claim": format!("the line the ledger rolled forward: {}", line_full.iter().map(|x| x.replace('|', " [") + "]").collect::<Vec<_>>().join("  ")), "status": "PROVEN", "lo": 0, "hi": 0})];
        let mut seen = HashSet::new();
        let mut stack = ids;
        while let Some(i) = stack.pop() {
            if !seen.insert(i) { continue; }
            out.push(self.finding_json(&self.fs[i]));
            stack.extend(self.fs[i].cites.clone());
        }
        while let Some(sn) = snaps.pop() { self.leave(sn); }
        (self.used, self.carried, self.reasked, self.facts_n, self.nodes) = saved;
        self.pushes = saved_pushes;
        out
    }
}

impl ReasoningLedger {
    /// DECIDE the move for the side to move (ANALYST_PLAN 39.2): candidates, rule intervals, B* with stake, the budget, the solver's confirmation.
    fn decide_impl(&mut self, py: Python<'_>) -> PyResult<PyObject> {
        let t0 = std::time::Instant::now();
        self.used = 0; self.carried = 0; self.reasked = 0; self.facts_n = 0; self.skipped.clear(); self.confirmations.clear();
        self.tt.clear(); self.nodes = 0; self.pushes = 0; self.cap_now = 0; self.root_ply = self.ply; self.node_cap_hit = false; self.budget_hit = false;
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
        let out = json!({"move": chosen.uci, "status": status, "reason": reason, "interval": [chosen.iv.0, chosen.iv.1], "why": chosen.why, "questions": questions, "budget": self.budget, "nodes": self.nodes, "positions": self.pushes, "budget_hit": self.budget_hit, "node_cap_hit": self.node_cap_hit,
                         "facts": facts, "candidates": n_cand, "legal": legal.len(), "refined": cands.iter().filter(|c| c.refined).count(), "zero_stake": zero_stake,
                         "unanswered": unanswered, "confirm": confirm, "table": tab, "all": cands.iter().map(|c| json!([c.uci, c.why, [c.rule.0, c.rule.1], [c.iv.0, c.iv.1], c.refined])).collect::<Vec<_>>(), "chain": chain, "seconds": t0.elapsed().as_secs_f64()});
        let j = py.import("json")?;
        Ok(j.call_method1("loads", (out.to_string(),))?.unbind())
    }
}
