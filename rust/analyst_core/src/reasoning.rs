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

    fn retract(&mut self, touched: Bitboard) -> usize {
        let mut n = 0;
        let mut changed = true;
        while changed {
            changed = false;
            for i in 0..self.fs.len() {
                if self.fs[i].stale.is_some() { continue; }
                let hit = (self.fs[i].reads & touched).any() || self.fs[i].cites.iter().any(|&c| self.fs[c].stale.is_some());
                if hit { self.fs[i].stale = Some(self.ply); n += 1; changed = true; }
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
            budget, used: 0, carried: 0, reasked: 0, facts_n: 0, solver_calls: 0, stack: vec![], skipped: vec![], agenda: vec![], plies: vec![], confirmations: vec![] })
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
