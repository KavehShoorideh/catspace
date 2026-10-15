//! THE EVENT LOOP'S QUESTIONS (Kaveh 2026-10-15, docs/ANALYST_PLAN.md section 40). The bookkeeping (events, conditions, the agenda, the policy, chains as
//! records) is `crate::agenda`; nothing there knows chess. Here are the questions the loop can ask -- each one the ledger's existing words:
//!   SURVEY   what a side attacks                       -> found ATTACKED(piece)         -> SAFE(piece)
//!   SAFE     the exchange on a piece (ask_safe)         -> a HELD piece posts reason(guard) for the side that owns it -> DEFLECT(guard) for the other
//!   POKES    the moves of a side that attack something with a safe landing -> found POKE(move) -> POKE(move)
//!   POKE     the move played: what it takes, what it newly attacks -> found ATTACKED(target, after the move) -> SAFE(target) ... -> ASSESS(move)
//!   ASSESS   what the other side can still do about its problems (stand-pat, the fixers they share) and what the move opens for it (LATENT) -> UNCOVER
//!   DEFLECT  can the holder be got off its post: MOBILITY and REACH (vocab.rs), then the exchange on the holder's square with each retake
//!   UNCOVER  a shield that moves opens a line onto a piece of ours
//!   TRADE    an equal exchange: the transition table's reading (data/derived/transitions.json, ANALYST_PLAN 33)
//! A question asked "after a move" carries the moves in its args ("@uci", replayed from the root, the other side passing in between): the findings asked
//! there are retracted on leaving, the answer stays in the chain.
use super::*;
use crate::agenda::{Condition, EvKind, Event, Question};

#[derive(Clone)]
pub struct Cand2 { pub actor: u8, pub uci: String, pub san: String, pub lo: i64, pub hi: i64, pub constraint: i64, pub risk: i64, pub why: String }

fn colour(a: u8) -> Color { if a == 0 { Color::White } else { Color::Black } }
fn actor_of(c: Color) -> u8 { if c == Color::White { 0 } else { 1 } }
fn who(c: Color) -> &'static str { if c == Color::White { "White" } else { "Black" } }
fn cost_of(kind: &str) -> i64 { match kind { "SURVEY" | "TRADE" => 0, "SAFE" => 2, "POKES" => 3, "POKE" => 4, "DEFLECT" => 5, "ASSESS" | "UNCOVER" => 6, _ => 9 } }

/// The conditions of the loop: premise-key patterns over events. DATA (a proposer, phase 4, adds to this list).
fn conditions() -> Vec<Condition> {
    vec![
        Condition::new("the position arrives: what each side attacks", EvKind::Push, "PUSH|*", "SURVEY", false, true),
        Condition::new("the position arrives: the moves that attack something", EvKind::Push, "PUSH|*", "POKES", false, false),
        Condition::new("a piece is attacked: is it safe", EvKind::Found, "ATTACKED|*", "SAFE", false, false),
        Condition::new("an equal exchange: the transition table", EvKind::Found, "TRADE|*", "TRADE", false, true),
        Condition::new("a finding is held by something: can it be got off its post (the other side's question)", EvKind::Reason, "HELD|*", "DEFLECT", true, false),
        Condition::new("a move attacks something: play it", EvKind::Found, "POKE|*", "POKE", false, false).fresh(),
        Condition::new("every target of a move is answered: assess it", EvKind::Found, "ASSESS|*", "ASSESS", false, false),
        Condition::new("a shield stands on a line onto a piece: can it move", EvKind::Found, "LATENT|*", "UNCOVER", false, false),
    ]
}

impl ReasoningLedger {
    // ---- plumbing -------------------------------------------------------------------------------------------------------------------------
    fn split_args(args: &[String]) -> (Vec<String>, Vec<String>) {
        let ctx: Vec<String> = args.iter().filter(|a| a.starts_with('@')).map(|a| a[1..].to_string()).collect();
        (args.iter().filter(|a| !a.starts_with('@')).cloned().collect(), ctx)
    }
    fn ctx_args(ctx: &[String]) -> Vec<String> { ctx.iter().map(|u| format!("@{u}")).collect() }
    fn pc_at(&self, s: &str) -> Option<Pc> {
        let sq: Square = s.get(s.len().saturating_sub(2)..)?.parse().ok()?;
        self.pos.board().piece_at(sq).map(|p| (p.role, sq))
    }
    fn kind_word(c: Color, r: Role) -> String { format!("{} {}", who(c), match r { Role::Pawn => "pawn", Role::Knight => "knight", Role::Bishop => "bishop", Role::Rook => "rook", Role::Queen => "queen", Role::King => "king" }) }

    /// Run `f` on the board after the moves of `ctx` (replayed from here; a move by the side not to move is preceded by the other side passing). The ledger is
    /// restored afterwards: what was asked in there is retracted (enter / leave), the answers live in the chain.
    fn with_ctx<R>(&mut self, ctx: &[String], f: impl FnOnce(&mut Self) -> R) -> Option<R> {
        let saved = (self.pos.clone(), self.line.len());
        let mut snaps = vec![];
        for u in ctx {
            let Ok(um) = u.parse::<shakmaty::uci::UciMove>() else { break };
            let Some(from) = (match &um { shakmaty::uci::UciMove::Normal { from, .. } => Some(*from), _ => None }) else { break };
            let Some(pc) = self.pos.board().piece_at(from) else { break };
            if self.pos.turn() != pc.color {
                match Self::passed(&self.pos) { Some(p) => { self.pos = p; self.fact_cache.clear(); } None => break }
            }
            let Ok(m) = um.to_move(&self.pos) else { break };
            snaps.push(self.enter(&m));
        }
        let ok = snaps.len() == ctx.len();
        let r = if ok { Some(f(self)) } else { None };
        while let Some(s) = snaps.pop() { self.leave(s); }
        self.pos = saved.0; self.line.truncate(saved.1); self.fact_cache.clear();
        r
    }

    fn post_ev(&mut self, kind: EvKind, subject: String, payload: String, actor: u8, stake: i64, args: Vec<String>) -> usize {
        let ev = Event { id: 0, kind, ply: self.ply, subject, payload, actor, stake, args, parent: self.cur_q };
        self.ag.post(ev, self.cur_q, &cost_of).len()
    }

    /// A reason: a finding of `owner`'s is HELD by something. The event opens the OTHER side's question (agenda.rs: flip).
    fn reason(&mut self, owner: Color, subject_piece: &str, holder: &str, stake: i64, mut args: Vec<String>, payload: String) {
        if !self.ag_on || stake <= 0 { return; }
        args.extend(Self::ctx_args(&self.ctx_now()));
        self.post_ev(EvKind::Reason, format!("HELD|{}|{}|{}", side(owner), subject_piece, holder), payload, actor_of(owner), stake, args);
    }
    /// The moves played since the root (the context a question was asked in): kept while a question runs.
    fn ctx_now(&self) -> Vec<String> { self.cur_ctx.clone() }

    // The hooks of the older questions: a held line (BLOCKS) and a tempo (PIN) post their holder too, for both sides.
    pub(super) fn reason_blocker(&mut self, id: usize, x: Pc, a: Pc, y: Pc) {
        if !self.ag_on { return; }
        let owner = self.pos.board().color_at(x.1).unwrap_or(Color::White);
        let st = self.fs[id].hi;
        self.reason(owner, &pname(y), &pname(x), st, vec![pname(x), pname(y)], format!("{} shields {} from {}", pname(x), pname(y), pname(a)));
    }
    pub(super) fn reason_tempo(&mut self, x: Pc, y: Pc, uci: &str) {
        if !self.ag_on { return; }
        let owner = self.pos.board().color_at(x.1).unwrap_or(Color::White);
        self.reason(owner, &pname(y), "tempo", 1, vec!["tempo".into(), pname(y)], format!("{uci} gives check: the tempo holds {}", pname(y)));
    }

    // ---- the driver -----------------------------------------------------------------------------------------------------------------------
    pub(super) fn run_chain(&mut self, questions: u32) -> Vec<String> {
        self.ag = crate::agenda::Agenda::new(conditions(), questions);
        self.ag_on = true;
        self.cur_q = None; self.cur_ctx = vec![]; self.pending.clear(); self.cands.clear();
        self.used = 0; self.budget = self.budget.max(5000); self.skipped.clear();
        let t = self.pos.turn();
        for c in [t, !t] {
            self.post_ev(EvKind::Push, format!("PUSH|{}|root", side(c)), String::new(), actor_of(c), 9, vec![]);
        }
        while let Some(q) = self.ag.take() {
            if self.deadline.map_or(false, |d| std::time::Instant::now() > d) { self.cut = true; break; }
            self.cur_q = Some(q.id);
            let (_, ctx) = Self::split_args(&q.args);
            self.cur_ctx = ctx;
            // THE LEMMAS (ANALYST_PLAN 40.3): a promoted shape answers its question in one step, the chain below it is not run -- unless the stake is above the knob or the
            // question is a held-out check, which is run and compared at the end
            let premise = if self.lem.on { self.premise_of(&q) } else { None };
            let mut by_lemma: Option<(String, String, String, Option<(i64, i64)>)> = None;
            if let Some(pk) = &premise {
                if let Some(l) = self.lem.promoted(pk) {
                    let ho = self.lem.heldout(pk);
                    if q.stake <= self.lem.stake_knob && !ho {
                        self.lem.hits += 1;
                        *self.lem.hit_by.entry(l.id.clone()).or_insert(0) += 1;
                        if let (Some((lo, _)), "SAFE") = (l.value, q.kind.as_str()) { if lo >= 1 && self.cur_ctx.is_empty() { self.ag.raise(q.actor, lo); } }
                        by_lemma = Some((format!("{} [by lemma]", Self::q_label(&q)), format!("by lemma {}: {} ({} positions)", l.id, l.outcome, l.count), String::new(), l.value));
                    } else {
                        self.lem.pending.insert(q.id, (pk.clone(), l.outcome.clone()));
                    }
                }
                if by_lemma.is_none() { self.lem.premise_of_q.insert(q.id, pk.clone()); }
            }
            let (text, answer, reason, value) = if let Some(x) = by_lemma { x } else { match q.kind.as_str() {
                "SURVEY" => self.q_survey(&q),
                "SAFE" => self.q_safe(&q),
                "POKES" => self.q_pokes(&q),
                "POKE" => self.q_poke(&q),
                "ASSESS" => self.q_assess(&q),
                "DEFLECT" => self.q_deflect(&q),
                "UNCOVER" => self.q_uncover(&q),
                "TRADE" => self.q_trade(&q),
                k => (k.to_string(), "unknown question kind".into(), String::new(), None),
            } };
            self.ag.settle(&q, text, answer, reason, cost_of(&q.kind), value);
            if let Some(p) = q.parent { if let Some(n) = self.pending.get_mut(&p) {
                *n = n.saturating_sub(1);
                if *n == 0 { self.pending.remove(&p); self.after_children(p); }
            } }
            self.cur_q = None;
        }
        self.ag_on = false;
        self.cur_ctx = vec![];
        self.lem.q_total += self.ag.spent as u64;
        self.lemma_observe();
        self.chain_lines()
    }

    /// The last target of a POKE is answered: the move is ready to be assessed.
    fn after_children(&mut self, poke_q: usize) {
        let Some(rec) = self.ag.chains.iter().find(|r| r.q == poke_q).cloned() else { return };
        self.cur_q = Some(poke_q);
        let (prim, ctx) = Self::split_args(&rec.args);
        let mut args = vec![prim[0].clone()];
        args.extend(Self::ctx_args(&ctx));
        self.post_ev(EvKind::Found, format!("ASSESS|{}|{}", side(colour(rec.actor)), prim[0]), rec.answer.clone(), rec.actor, rec.stake, args);
    }

    // ---- the printed chain ---------------------------------------------------------------------------------------------------------------
    fn chain_lines(&self) -> Vec<String> {
        let mut out = vec!["THE CHAIN (the event loop: events -> conditions -> an agenda by stake -> a policy picks the next question; [stake] is the value the question could change)".to_string()];
        // children after their parent: the order of answering, indented by the depth
        let ch = &self.ag.chains;
        fn walk(ch: &[crate::agenda::ChainRec], parent: Option<usize>, out: &mut Vec<String>) {
            for r in ch.iter().filter(|r| r.parent == parent) {
                let pad = "  ".repeat(r.depth as usize);
                let val = r.value.map(|(l, h)| if l == h { format!("  value {l}") } else { format!("  value {l}..{h}") }).unwrap_or_default();
                out.push(format!("{pad}[{}] {}: {} -> {}{}{}", r.stake, who(colour(r.actor)), r.text, r.answer, if r.reason.is_empty() { String::new() } else { format!("  BECAUSE {}", r.reason) }, val));
                walk(ch, Some(r.q), out);
            }
        }
        walk(ch, None, &mut out);
        if !self.cands.is_empty() {
            out.push("CANDIDATES (moves that attack something with a safe landing, assessed):".to_string());
            for c in &self.cands {
                let verdict = if c.constraint >= 1 && c.constraint >= c.lo { "REJECTED".to_string() } else if c.lo >= 1 { "WINS".to_string() } else if c.risk >= 1 { format!("RISKY (the shield can move at a cost of up to {} to {})", c.risk, who(colour(c.actor))) } else { "KEPT".to_string() };
                out.push(format!("  {} {}: gains {}..{}, sure cost of what it opens {} -> {}{}", who(colour(c.actor)), c.san, c.lo, c.hi, c.constraint, verdict, if c.why.is_empty() { String::new() } else { format!(" [{}]", c.why) }));
            }
        }
        out.push(format!("EVENTS: {} posted, {} questions asked, {} refused by the gates (stake below the best known value, or the budget): best known value White {} Black {}",
                         self.ag.events.len(), self.ag.spent, self.ag.refused.len(), self.ag.best[0], self.ag.best[1]));
        out
    }

    // ---- SURVEY ---------------------------------------------------------------------------------------------------------------------------
    fn q_survey(&mut self, q: &Question) -> (String, String, String, Option<(i64, i64)>) {
        let c = colour(q.actor);
        let (_, ctx) = Self::split_args(&q.args);
        let found: Vec<Pc> = self.with_ctx(&ctx, |s| { let b = s.pos.board().clone(); let mut v: Vec<Pc> = s.pieces_of(!c).into_iter().filter(|p| attacked(&b, p.1, c)).collect(); v.sort_by_key(|p| (std::cmp::Reverse(val(p.0)), p.1 as u32)); v }).unwrap_or_default();
        for p in &found {
            let mut a = vec![pname(*p)];
            a.extend(Self::ctx_args(&ctx));
            self.post_ev(EvKind::Found, format!("ATTACKED|{}|{}", side(!c), pname(*p)), String::new(), q.actor, val(p.0), a);
        }
        (format!("what does {} attack", who(c)), if found.is_empty() { "nothing".into() } else { found.iter().map(|p| format!("{} ({})", pname(*p), val(p.0))).collect::<Vec<_>>().join(", ") }, String::new(), None)
    }

    // ---- SAFE -----------------------------------------------------------------------------------------------------------------------------
    fn q_safe(&mut self, q: &Question) -> (String, String, String, Option<(i64, i64)>) {
        let c = colour(q.actor);
        let (prim, ctx) = Self::split_args(&q.args);
        let r = self.with_ctx(&ctx, |s| {
            let Some(p) = s.pc_at(&prim[0]) else { return None };
            let owner = !c;
            let id = s.ask_safe(p, None);
            let f = s.fs[id].clone();
            let mut holders = vec![];
            for h in &f.holders {
                let j = s.ask_job(*h, p, None);
                holders.push((*h, s.fs[j].lo));
            }
            let trade = f.sq.get(1).and_then(|a| s.pos.board().piece_at(*a)).map(|a| (a.role, f.sq[1])).filter(|a| val(a.0) == val(p.0) && !f.holders.is_empty() && f.hi == 0);
            Some((p, owner, f, holders, trade))
        }).flatten();
        let Some((p, owner, f, holders, trade)) = r else { return (format!("is {} safe", prim[0]), "the piece is not there".into(), String::new(), None) };
        self.qsq.insert(q.id, f.sq.clone());
        if f.lo >= 1 { if ctx.is_empty() { self.ag.raise(q.actor, f.lo); } self.post_ev(EvKind::Found, format!("WIN|{}|{}", side(owner), pname(p)), f.claim.clone(), q.actor, f.lo, vec![pname(p)]); }
        let mut reason = String::new();
        for (h, job) in &holders {
            if *job > 0 {
                let mut a = vec![pname(*h), pname(p)];
                a.extend(Self::ctx_args(&ctx));
                reason = format!("{}{} holds it (without it the exchange on {} costs {} more)", if reason.is_empty() { String::new() } else { format!("{}; ", reason) }, pname(*h), pname(p), job);
                self.post_ev(EvKind::Reason, format!("HELD|{}|{}|{}", side(owner), pname(p), pname(*h)), f.claim.clone(), actor_of(owner), *job, a);
            }
        }
        if let Some(tr) = trade {
            if p.0 == Role::Queen {
                let mut a = vec![pname(tr), pname(p)];
                a.extend(Self::ctx_args(&ctx));
                self.post_ev(EvKind::Found, format!("TRADE|{}|{},{}", side(owner), pname(tr), pname(p)), String::new(), q.actor, val(p.0), a);
            }
        }
        (format!("is {} safe", pname(p)), f.claim.clone(), reason, Some((f.lo, f.hi)))
    }

    // ---- TRADE ----------------------------------------------------------------------------------------------------------------------------
    fn q_trade(&mut self, q: &Question) -> (String, String, String, Option<(i64, i64)>) {
        let text = "what is the trade of queens worth (transition table)".to_string();
        let Some(tab) = self.trans.clone() else {
            // TODO(Kaveh): without data/derived/transitions.json the stake of a trade is unknown; no rule is invented.
            return (text, "no transition table loaded (data/derived/transitions.json): the stake of the trade is unknown".into(), String::new(), None);
        };
        let (us, bal, bin) = self.trade_cell();
        let (mut num, mut den, mut games) = (0.0f64, 0.0f64, 0i64);
        // THE POPULATION FIRST (the Lichess bands pooled): the machinery is generic; a player's own cells are a product reading, read only when the population has no row
        let mut source = "the Lichess bands pooled";
        for pass in 0..2 {
            let pre = if pass == 0 { "lichess" } else { "kaveh" };
            if let Some(t) = tab.get("table").and_then(|t| t.as_object()) {
                for (cell, v) in t {
                    if !cell.starts_with(pre) || !cell.ends_with(&format!("| {bin}")) { continue; }
                    if let Some(e) = v.get("queens_on") {
                        let n = e["n_fell"].as_i64().unwrap_or(0) + e["n_kept"].as_i64().unwrap_or(0);
                        num += e["diff"].as_f64().unwrap_or(0.0) * n as f64; den += n as f64; games += n;
                    }
                }
            }
            if den > 0.0 { break; }
            source = "the Lichess bands, pooled";
        }
        if den == 0.0 { return (text, format!("the table has no queens_on reading for the cell '{bin}'"), String::new(), None); }
        let d = num / den;
        let _ = q;
        (text, format!("{} is {} in material ({bin}); in the table the mover's result changes by {:+.3} when the queens come off within 20 plies ({}, {} rows); a score difference, not pawns: no cp-to-pawn mapping for it exists in the code, so it is the stake in its own unit (TODO Kaveh)", who(us), bal, d, source, games),
         if d < 0.0 { format!("trading queens costs the side that is {bin}") } else { format!("trading queens does not cost the side that is {bin}") }, None)
    }

    // ---- POKES ----------------------------------------------------------------------------------------------------------------------------
    /// The moves of the actor that attack something new with a safe landing (the exchange on the landing square costs no more than the move takes).
    fn q_pokes(&mut self, q: &Question) -> (String, String, String, Option<(i64, i64)>) {
        let c = colour(q.actor);
        let (_, ctx) = Self::split_args(&q.args);
        let found: Vec<(i64, String, String, String)> = self.with_ctx(&ctx, |s| {
            let pb = if s.pos.turn() == c { s.pos.clone() } else { match Self::passed(&s.pos) { Some(p) => p, None => return vec![] } };
            let was = pressure(pb.board(), !c);
            let mut out = vec![];
            for m in pb.legal_moves() {
                if m.is_castle() { continue; }
                let g = gain_of(&m);
                let mut r = pb.clone(); r.play_unchecked(&m);
                let loss = crate::rules::see(r.board(), m.to(), !c);
                if g - loss < 0 { continue; }
                let now = pressure(r.board(), !c);
                let mut tgt = 0;
                for (sq, (n, mv)) in &now {
                    let (bn, bmv) = was.get(sq).copied().unwrap_or((0, 99));
                    if *n > bn || *mv < bmv { tgt = tgt.max(r.board().piece_at(*sq).map_or(0, |p| val(p.role))); }
                }
                let stake = (g + tgt).max(if r.is_check() { 1 } else { 0 });
                if stake < 1 { continue; }
                let uci = m.to_uci(CastlingMode::Standard).to_string();
                out.push((stake, crate::plan::san(&pb, &uci), uci, format!("gain {g}, newly attacks a {tgt}-point piece")));
            }
            out.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
            out.truncate(12);
            out
        }).unwrap_or_default();
        let mut n = 0;
        for (stake, san, uci, why) in &found {
            let mut a = vec![uci.clone()];
            a.extend(Self::ctx_args(&ctx));
            n += self.post_ev(EvKind::Found, format!("POKE|{}|{}", side(c), uci), format!("{san}|{why}"), q.actor, *stake, a);
        }
        (format!("which moves of {} attack something with a safe landing", who(c)),
         if found.is_empty() { "none".into() } else { format!("{} ({} opened as questions)", found.iter().map(|f| format!("{} ({})", f.1, f.0)).collect::<Vec<_>>().join(", "), n) }, String::new(), None)
    }

    // ---- POKE -----------------------------------------------------------------------------------------------------------------------------
    fn q_poke(&mut self, q: &Question) -> (String, String, String, Option<(i64, i64)>) {
        let c = colour(q.actor);
        let (prim, ctx) = Self::split_args(&q.args);
        let uci = prim[0].clone();
        let san = q.payload.split('|').next().unwrap_or("").to_string();
        let mut full = ctx.clone();
        let pre_ctx = ctx.clone();
        full.push(uci.clone());
        let info = self.with_ctx(&pre_ctx, |s| {
            let pb = if s.pos.turn() == c { s.pos.clone() } else { Self::passed(&s.pos)? };
            let um: shakmaty::uci::UciMove = uci.parse().ok()?;
            let m = um.to_move(&pb).ok()?;
            let was = pressure(pb.board(), !c);
            Some((was, gain_of(&m), m.capture(), m.role(), m.to()))
        }).flatten();
        let Some((was, g, cap, role, to)) = info else { return (format!("play {san}"), "illegal here".into(), String::new(), None) };
        let ctx_a = Self::ctx_args(&ctx);
        let res = self.with_ctx(&full, |s| {
            let me = (role, to);
            let lands = if s.pos.board().piece_at(to).is_some() { let id = s.ask_safe(me, None); s.fs[id].claim.clone() } else { String::new() };
            let now = pressure(s.pos.board(), !c);
            let mut t: Vec<Pc> = vec![];
            for (sq, (n, mv)) in &now {
                let (bn, bmv) = was.get(sq).copied().unwrap_or((0, 99));
                if *n > bn || *mv < bmv { if let Some(p) = s.pos.board().piece_at(*sq) { if p.role != Role::King { t.push((p.role, *sq)); } } }
            }
            t.sort_by_key(|p| (std::cmp::Reverse(val(p.0)), p.1 as u32));
            t.truncate(3);
            (lands, t, s.pos.is_check())
        });
        let Some((lands, targets, check)) = res else { return (format!("play {san}"), "illegal here".into(), String::new(), None) };
        let mut n = 0;
        for p in &targets {
            let mut a = vec![pname(*p)];
            a.extend(Self::ctx_args(&full));
            n += self.post_ev(EvKind::Found, format!("ATTACKED|{}|{}", side(!c), pname(*p)), String::new(), q.actor, val(p.0), a);
        }
        let _ = ctx_a;
        if n > 0 { self.pending.insert(q.id, n as u32); }
        let takes = cap.map(|r| format!("takes the {} (+{}); ", Self::kind_word(!c, r).to_lowercase(), val(r))).unwrap_or_default();
        let answer = format!("{takes}{}{}{}", lands, if check { "; gives check" } else { "" },
                             if targets.is_empty() { "; attacks nothing new".to_string() } else { format!("; attacks {}", targets.iter().map(|p| format!("{} ({})", pname(*p), val(p.0))).collect::<Vec<_>>().join(", ")) });
        if n == 0 { self.pending.remove(&q.id); self.after_children_later(q); }
        let _ = g;
        (format!("{san}: play it"), answer, String::new(), None)
    }
    /// A POKE with no target to ask about is assessed straight away (posted after the question settles).
    fn after_children_later(&mut self, q: &Question) {
        let (prim, ctx) = Self::split_args(&q.args);
        let mut args = vec![prim[0].clone()];
        args.extend(Self::ctx_args(&ctx));
        self.post_ev(EvKind::Found, format!("ASSESS|{}|{}", side(colour(q.actor)), prim[0]), q.payload.clone(), q.actor, q.stake, args);
    }

    /// THE EXACT SOLVER CONFIRMS (never finds): the largest gain v (1..=upto) that `uci` forces within three moves against every defence, or 0. Monotone in v.
    fn probe_forced(&mut self, ctx: &[String], c: Color, uci: &str, upto: i64) -> (i64, u64) {
        let r = self.with_ctx(ctx, |s| {
            let pb = if s.pos.turn() == c { s.pos.clone() } else { Self::passed(&s.pos)? };
            let fen = shakmaty::fen::Fen::from_position(pb, shakmaty::EnPassantMode::Legal).to_string();
            crate::goals::solver_nodes_reset();
            let (mut best, mut v) = (0, 1);
            while v <= upto {
                let res = crate::goals::forced_moves(fen.clone(), v, 3, 300_000);
                if res.iter().any(|r| r.0 == uci && r.1 == "forced") { best = v; v += 1; } else { break; }
            }
            Some((best, crate::goals::solver_nodes()))
        }).flatten();
        self.solver_calls += 1;
        r.unwrap_or((0, 0))
    }

    // ---- ASSESS ---------------------------------------------------------------------------------------------------------------------------
    fn q_assess(&mut self, q: &Question) -> (String, String, String, Option<(i64, i64)>) {
        let c = colour(q.actor);
        let (prim, ctx) = Self::split_args(&q.args);
        let uci = prim[0].clone();
        let san = self.ag.chains.iter().find(|r| r.kind == "POKE" && r.args.first() == Some(&uci) && Self::split_args(&r.args).1 == ctx).and_then(|r| r.text.split(':').next().map(|x| x.to_string())).unwrap_or(uci.clone());
        let mut full = ctx.clone();
        full.push(uci.clone());
        let g = self.with_ctx(&ctx, |s| { let pb = if s.pos.turn() == c { s.pos.clone() } else { Self::passed(&s.pos)? }; let m = uci.parse::<shakmaty::uci::UciMove>().ok()?.to_move(&pb).ok()?; Some(gain_of(&m)) }).flatten().unwrap_or(0);
        let saved = (self.lean_mode, self.deep);
        self.lean_mode = true; self.deep = true;
        let to: Square = uci.get(2..4).and_then(|x| x.parse().ok()).unwrap_or(Square::A1);
        let r = self.with_ctx(&full, |s| {
            let opp = !c;
            if s.pos.legal_moves().is_empty() { return (if s.pos.is_check() { (-MATE, -MATE) } else { (0, 0) }, vec![], String::new(), vec![]); }
            let sp = if s.pos.is_check() { s.check_value() } else { s.stand_pat() };
            // the opponent's problems, the two biggest, and the moves of the opponent that fix both
            let mut probs: Vec<(i64, Pc)> = vec![];
            let b = s.pos.board().clone();
            for p in s.pieces_of(opp).into_iter().filter(|p| attacked(&b, p.1, c)) {
                let id = s.ask_safe(p, None);
                if s.fs[id].hi > 0 { probs.push((s.fs[id].hi, p)); }
            }
            probs.sort_by_key(|x| (std::cmp::Reverse(x.0), x.1 .1 as u32));
            let mut shared = String::new();
            // the pair to ask about: the problems the moved piece itself makes (a fork), else the two biggest
            let mover_att = s.pos.board().attacks_from(to);
            let mut pair: Vec<(i64, Pc)> = probs.iter().filter(|p| mover_att.contains(p.1 .1)).cloned().collect();
            if pair.len() < 2 { pair = probs.clone(); }
            if pair.len() >= 2 {
                let mv = s.moves_for(opp);
                let f1: Vec<String> = s.fixers(pair[0].1, &mv).iter().map(|m| m.to_uci(CastlingMode::Standard).to_string()).collect();
                let f2: Vec<String> = s.fixers(pair[1].1, &mv).iter().map(|m| m.to_uci(CastlingMode::Standard).to_string()).collect();
                let both: Vec<String> = f1.iter().filter(|u| f2.contains(u)).cloned().collect();
                shared = if both.is_empty() { format!("no single move of {} saves both {} and {}", who(opp), pname(pair[0].1), pname(pair[1].1)) }
                    else { format!("{} saves both {} and {}", both.iter().take(3).map(|u| crate::plan::san(&s.pos, u)).collect::<Vec<_>>().join("/"), pname(pair[0].1), pname(pair[1].1)) };
            }
            // what the move opens for the opponent: its slider, its shield, our piece (the line appears if the shield leaves)
            let mut latent = vec![];
            for sl in s.pieces_of(opp).into_iter().filter(|p| matches!(p.0, Role::Queen | Role::Rook | Role::Bishop)) {
                for y in s.pieces_of(c).into_iter().chain(b.king_of(c).map(|k| (Role::King, k))) {
                    if y.0 == Role::King { continue; }
                    let btw = shakmaty::attacks::between(sl.1, y.1);
                    let on_line = match sl.0 { Role::Rook => sl.1.file() == y.1.file() || sl.1.rank() == y.1.rank(),
                        Role::Bishop => shakmaty::attacks::bishop_attacks(sl.1, Bitboard::EMPTY).contains(y.1),
                        _ => shakmaty::attacks::queen_attacks(sl.1, Bitboard::EMPTY).contains(y.1) };
                    if !on_line || btw.is_empty() { continue; }
                    let between = btw & b.occupied();
                    if between.count() != 1 { continue; }
                    let xs = between.first().unwrap();
                    if b.color_at(xs) != Some(opp) { continue; }
                    let Some(xp) = b.piece_at(xs) else { continue };
                    if xp.role == Role::King { continue; }
                    let (w, _) = s.fact(&format!("(without (at {}) (see {}))", s.nm(xs), s.nm(y.1)));
                    if w > 0 { latent.push((w, (xp.role, xs), sl, y)); }
                }
            }
            (sp, probs, shared, latent)
        });
        self.lean_mode = saved.0; self.deep = saved.1;
        let Some((sp, probs, shared, latent)) = r else { return (format!("assess {san}"), "illegal here".into(), String::new(), None) };
        let (lo, hi) = ((g - sp.1).max(-MATE), (g - sp.0).min(2 * MATE));
        // the exact solver confirms the static interval where material is at stake (a stake of 3 or more): the gain the move FORCES within three moves
        let mut solver_txt = String::new();
        let mut lo = lo;
        let mut hi = hi;
        if q.stake >= 3 {
            let (forced, nodes) = self.probe_forced(&ctx, c, &uci, 12);
            if forced > 0 {
                solver_txt = format!("; SOLVER: forces a gain of {forced} within 3 moves ({nodes} nodes){}", if forced > hi { format!(" -- above the static upper bound {hi}: a SURPRISE (the static reading missed it)") } else if forced < lo { format!(" -- below the static lower bound {lo}") } else { String::new() });
                lo = lo.max(forced); hi = hi.max(forced);
            } else {
                solver_txt = format!("; SOLVER: forces no gain within 3 moves ({nodes} nodes)");
            }
        }
        if lo >= 1 { self.ag.raise(q.actor, lo); }
        let mut why = String::new();
        for (w, x, sl, y) in &latent {
            let mut a = vec![pname(*x), pname(*sl), pname(*y)];
            a.extend(Self::ctx_args(&full));
            why = format!("{}the {} opens {} onto {} when {} moves ({})", if why.is_empty() { String::new() } else { format!("{why}; ") }, if sl.1.file() == y.1.file() { format!("{}-file", sl.1.file()) } else { "line".to_string() }, pname(*sl), pname(*y), pname(*x), w);
            self.post_ev(EvKind::Found, format!("LATENT|{}|{}|{}", side(!c), pname(*x), pname(*y)), san.clone(), actor_of(!c), *w, a);
        }
        self.cands.push(Cand2 { actor: q.actor, uci: uci.clone(), san: san.clone(), lo, hi, constraint: 0, risk: 0, why: why.clone() });
        let probtxt = if probs.is_empty() { format!("{} has no problem", who(!c)) } else { format!("{}'s problems: {}", who(!c), probs.iter().take(3).map(|(h, p)| format!("{} (loses {})", pname(*p), h)).collect::<Vec<_>>().join(", ")) };
        (format!("assess {san}"), format!("gains {g} now; {probtxt}{}{} -> static worth {}..{}{solver_txt}", if shared.is_empty() { String::new() } else { format!("; {shared}") }, if probs.len() >= 2 { " (the mover fixes one problem, the rest is lost)" } else { "" }, (g - sp.1).max(-MATE), (g - sp.0).min(2 * MATE)),
         why, Some((lo, hi)))
    }

    // ---- UNCOVER --------------------------------------------------------------------------------------------------------------------------
    fn q_uncover(&mut self, q: &Question) -> (String, String, String, Option<(i64, i64)>) {
        let opp = colour(q.actor);
        let (prim, ctx) = Self::split_args(&q.args);
        let text = format!("can {} move the shield {} (the line from {} to {})", who(opp), prim[0], prim[1], prim[2]);
        let saved = (self.lean_mode, self.deep);
        self.lean_mode = true; self.deep = true;
        let r = self.with_ctx(&ctx, |s| {
            let Some(x) = s.pc_at(&prim[0]) else { return None };
            let mut best: Option<(i64, i64, String)> = None;          // (sure loss, possible loss, san) for the other side
            for m in s.moves_for(opp).into_iter().filter(|m| m.from() == Some(x.1)) {
                let san = s.san_of(&m);
                let snap = s.enter(&m);
                let sp = if s.pos.is_check() { s.check_value() } else { s.stand_pat() };
                s.leave(snap);
                let (sure, maybe) = ((-sp.1).max(0), (-sp.0).max(0));
                if best.as_ref().map_or(true, |b| (sure, maybe) > (b.0, b.1)) { best = Some((sure, maybe, san)); }
            }
            best
        }).flatten();
        self.lean_mode = saved.0; self.deep = saved.1;
        let Some((sure, maybe, san)) = r else { return (text, "the shield cannot move".into(), String::new(), None) };
        // the constraint on the candidate whose position this is (the last move of the context)
        if let Some(u) = ctx.last() {
            for c in self.cands.iter_mut().filter(|c| &c.uci == u) { c.constraint = c.constraint.max(sure); c.risk = c.risk.max(maybe); }
        }
        self.ag.raise(q.actor, sure);
        (text, if sure > 0 { format!("{} (best: {}) and then the other side loses at least {}", prim[0], san, sure) } else if maybe > 0 { format!("{san} may cost up to {maybe}, at least nothing is sure") } else { format!("no move of {} costs the other side anything (best {san})", prim[0]) },
         String::new(), Some((sure, maybe)))
    }

    // ---- DEFLECT --------------------------------------------------------------------------------------------------------------------------
    fn q_deflect(&mut self, q: &Question) -> (String, String, String, Option<(i64, i64)>) {
        let a = colour(q.actor);
        let (prim, ctx) = Self::split_args(&q.args);
        let text = format!("can {} be got off its post (it holds {})", prim[0], prim.get(1).cloned().unwrap_or_default());
        let saved = (self.lean_mode, self.deep);
        self.lean_mode = true; self.deep = true;
        // the move that was held (a poke of the actor's own, last in the context) is played AFTER the deflection: "if d, then the poke wins"
        let (base, held): (Vec<String>, Option<String>) = match ctx.last() {
            Some(last) if ctx.len() >= 1 => {
                let pre = ctx[..ctx.len() - 1].to_vec();
                let from: Option<Square> = last.get(..2).and_then(|x| x.parse().ok());
                let mover = from.and_then(|f| self.with_ctx(&pre, |s| s.pos.board().color_at(f))).flatten();
                if mover == Some(a) { (pre, Some(last.clone())) } else { (ctx.clone(), None) }
            }
            _ => (ctx.clone(), None),
        };
        let ctx = base;
        // 1. MOBILITY and REACH of the holder
        let r1 = self.with_ctx(&ctx, |s| {
            let h = s.pc_at(&prim[0])?;
            let t = prim.get(1).and_then(|x| s.pc_at(x));
            let mob = crate::vocab::mobility(&s.pos, h.1);
            let reach = crate::vocab::reach_route(&s.pos, a, h.1, 2);
            let mob_txt = format!("{} {} safe square(s) ({})", pname(h), mob.count(), mob.into_iter().map(|x| x.to_string()).collect::<Vec<_>>().join(","));
            Some((h, t, mob_txt, reach.map(|r| r.iter().map(|(f, t)| format!("{f}{t}")).collect::<Vec<_>>()), s.pos.clone()))
        }).flatten();
        let Some((h, t, mob_txt, reach, _pos)) = r1 else {
            self.lean_mode = saved.0; self.deep = saved.1;
            return (text, "not a piece (a tempo, a line): nothing to deflect".into(), String::new(), None);
        };
        let route = match reach {
            Err(bl) => {
                self.lean_mode = saved.0; self.deep = saved.1;
                let sqs = bl.into_iter().map(|x| x.to_string()).collect::<Vec<_>>().join(" ");
                self.reason(!a, &prim[0], &format!("blocked:[{sqs}]"), q.stake, vec![format!("blocked:[{sqs}]"), prim[0].clone()], format!("{} cannot reach {} (its landing squares are lost by the exchange)", who(a), prim[0]));
                return (text, format!("MOBILITY {}; REACH: unreachable for {} within 2 moves (landing squares lost by the exchange: [{}]) -- closed", mob_txt, who(a), sqs.replace(' ', ",")), String::new(), None);
            }
            Ok(r) => r,
        };
        // 2. the candidates: moves of the actor after which a piece of the actor attacks the holder, the landing safe; each valued by the exchange on the holder's
        // square with the best retake (the target T is what the holder held)
        let stake = q.stake;
        let cands = self.with_ctx(&ctx, |s| {
            let b = s.pos.board().clone();
            let was_att = attacked(&b, h.1, a);
            let mut out: Vec<(String, String)> = vec![];
            for m in s.moves_for(a) {
                if m.is_castle() || Some(m.to()) == t.map(|x| x.1) { continue; }
                let mut r = s.pos.clone();
                if r.turn() != a { r = match Self::passed(&r) { Some(p) => p, None => continue } }
                let mut after = r.clone(); after.play_unchecked(&m);
                let hits = (after.board().attacks_to(h.1, a, after.board().occupied()) & after.board().by_color(a)).any();
                if !hits || (was_att && m.to() != h.1) { continue; }
                let loss = crate::rules::see(after.board(), m.to(), !a);
                if gain_of(&m) - loss < 0 && m.to() != h.1 { continue; }
                let u = m.to_uci(CastlingMode::Standard).to_string();
                let san = crate::plan::san(&r, &u);
                out.push((u, san));
            }
            out.truncate(8);
            out
        }).unwrap_or_default();
        {
            let mut sqs = vec![h.1];
            if let Some(x) = t { sqs.push(x.1); }
            for (u, _) in &cands { for k in [0usize, 2] { if let Some(s) = u.get(k..k + 2).and_then(|x| x.parse::<Square>().ok()) { sqs.push(s); } } }
            self.qsq.insert(q.id, sqs);
        }
        let mut results: Vec<(i64, String, String, String)> = vec![];       // (net, san, uci, how)
        for (uci, san) in &cands {
            let mut full = ctx.clone();
            full.push(uci.clone());
            if let Some(hm) = &held { full.push(hm.clone()); }
            let tsq = t.map(|x| x.1);
            let r = self.with_ctx(&full, |s| {
                // the attacker A of the holder (cheapest) after the move; the retakers of the holder's square once A has taken it
                let b = s.pos.board().clone();
                let hold = s.pos.board().piece_at(h.1)?;
                let at = (b.attacks_to(h.1, a, b.occupied()) & b.by_color(a)).into_iter().filter_map(|x| b.piece_at(x).map(|p| (p.role, x))).min_by_key(|(r, x)| (val(*r), *x as u32))?;
                let cap_b = Self::board_after_capture(&s.pos, at.1, h.1)?;
                let owner = !a;
                let rets: Vec<Pc> = (cap_b.attacks_to(h.1, owner, cap_b.occupied()) & cap_b.by_color(owner)).into_iter().filter_map(|x| cap_b.piece_at(x).map(|p| (p.role, x)))
                    .filter(|(_, x)| Self::legal_capture(&cap_b, owner, *x, h.1)).collect();
                let base = val(hold.role);
                let (hn, an) = (s.nm(h.1), s.nm(at.1));
                let mut worst: Option<(i64, String)> = None;
                if rets.is_empty() {
                    let v = tsq.map_or(0, |t| s.fact(&format!("(after {an} {hn} (see {}))", s.nm(t))).0);
                    worst = Some((base + v, format!("{} takes {} and nothing retakes", pname(at), pname(h))));
                }
                for r in &rets {
                    let v = tsq.map_or(0, |t| s.fact(&format!("(after {an} {hn} (after {} {hn} (see {})))", s.nm(r.1), s.nm(t))).0);
                    let net = base - val(at.0) + v;
                    if worst.as_ref().map_or(true, |w| net < w.0) {
                        worst = Some((net, format!("{} takes {}, {} retakes, {} is then worth {} to the taker", pname(at), pname(h), pname(*r), t.map(pname).unwrap_or_default(), v)));
                    }
                }
                worst
            }).flatten();
            if let Some((net, how)) = r { results.push((net, san.clone(), uci.clone(), format!("{how}; net {net}"))); }
        }
        self.lean_mode = saved.0; self.deep = saved.1;
        results.sort_by(|x, y| y.0.cmp(&x.0));
        let _ = (stake, &route);
        let tn = t.map(pname).unwrap_or_default();
        let ctx_txt = if ctx.is_empty() { String::new() } else { String::new() };
        if let Some(best) = results.first().filter(|r| r.0 >= 1).cloned() {
            // THE EXACT SOLVER CONFIRMS, NEVER FINDS: after the deflection (the actor to move again) is a gain of that size forced within three moves, the held move among
            // the moves that force it? A check in the held move must be answered first, which the static reading above does not see.
            let mut full = ctx.clone();
            full.push(best.2.clone());
            let want = held.clone();
            let conf = self.with_ctx(&full, |s| {
                let pos = if s.pos.turn() == a { s.pos.clone() } else { Self::passed(&s.pos)? };
                let fen = shakmaty::fen::Fen::from_position(pos, shakmaty::EnPassantMode::Legal).to_string();
                crate::goals::solver_nodes_reset();
                let res = crate::goals::forced_moves(fen, best.0.min(20), 3, 400_000);
                let nodes = crate::goals::solver_nodes();
                let ok = res.iter().any(|r| r.1 == "forced" && want.as_ref().map_or(true, |w| &r.0 == w));
                let line = res.iter().find(|r| r.1 == "forced" && want.as_ref().map_or(true, |w| &r.0 == w)).map(|r| r.3.join(" ")).unwrap_or_default();
                Some((ok, nodes, line))
            }).flatten();
            self.solver_calls += 1;
            let (ok, nodes, line) = conf.unwrap_or((false, 0, String::new()));
            let held_san = held.as_ref().map(|u| self.ag.chains.iter().find(|r| r.kind == "POKE" && r.args.first() == Some(u)).and_then(|r| r.text.split(':').next().map(|x| x.to_string())).unwrap_or(u.clone()));
            let also: Vec<String> = results.iter().skip(1).filter(|r| r.0 == best.0).map(|r| r.1.clone()).collect();
            let how = format!("if {}{}{} then {} (the exchange on {}) and {} falls: {}", best.1, if also.is_empty() { String::new() } else { format!(" (or {})", also.join(", ")) }, held_san.as_ref().map(|u| format!(" and then {u}")).unwrap_or_default(), best.3, pname(h), tn, best.0);
            if ok {
                let mut a2 = vec![best.2.clone()]; a2.extend(Self::ctx_args(&ctx));
                self.post_ev(EvKind::Found, format!("COND|{}|{}|{}", side(a), tn, best.1), best.3.clone(), q.actor, best.0, a2);
                self.ag.raise(q.actor, best.0);
                return (text, format!("MOBILITY {mob_txt}; REACH ok; CONDITIONAL two-move threat, SOLVER-CONFIRMED ({nodes} nodes: {line}): {how}"), format!("{} is lost once {} is deflected by {}", tn, pname(h), best.1), Some((best.0, best.0)));
            }
            return (text, format!("MOBILITY {mob_txt}; REACH ok; static reading: {how}; the exact solver does NOT confirm a forced gain of {} within 3 moves ({nodes} nodes) -- ESTIMATED, not counted", best.0),
                    String::new(), Some((0, best.0)));
        }
        let tried = results.iter().take(3).map(|r| format!("{} ({})", r.1, r.3)).collect::<Vec<_>>().join("; ");
        (text, format!("MOBILITY {mob_txt}; REACH: {}; no deflection pays{}", if route.is_empty() { "attacked already".to_string() } else { format!("route {}", route.join(" ")) }, if tried.is_empty() { String::new() } else { format!(" -- tried {tried}") }), String::new(), Some((0, 0)))
    }
}

// =====================================================================================================================================
// PHASE 2 (ANALYST_PLAN 40.4): the decision read off the chain. Findings against us are CONSTRAINTS on our move (each with its stake), findings for us are
// OPPORTUNITIES; the candidates are the moves that cash an opportunity or touch a constraint; each candidate is CHECKED by pushing it and running the loop on
// the position after it (only the touched findings are asked again); the value of a candidate is what it gains less what the other side can still do.
struct Item { text: String, stake: i64, lo: i64, hi: i64, squares: Vec<Square>, target: Option<Square>, deflect: bool, moves: Vec<String> }

impl ReasoningLedger {
    /// (the side to move at the root, its material balance, the table's material bin)
    fn trade_cell(&self) -> (Color, i64, &'static str) {
        let root = self.line.first().cloned().unwrap_or_else(|| self.pos.clone());
        let us = root.turn();
        let b = root.board();
        let bal: i64 = [Color::White, Color::Black].iter().map(|c| Role::ALL.iter().filter(|r| **r != Role::King).map(|r| ((b.by_color(*c) & b.by_role(*r)).count() as i64) * val(*r) * if *c == us { 1 } else { -1 }).sum::<i64>()).sum();
        let bin = if bal <= -5 { "behind <= -5" } else if bal <= -2 { "behind -4..-2" } else if bal <= 1 { "even" } else if bal <= 4 { "ahead 2..4" } else { "ahead >= 5" };
        (us, bal, bin)
    }
    /// The table's score difference for the mover when the queens come off: the Lichess bands pooled (n-weighted) first, the player's own cells only when the population has none. None without a table.
    fn trade_score(&self) -> Option<f64> {
        let tab = self.trans.as_ref()?;
        let (_, _, bin) = self.trade_cell();
        for pre in ["lichess", "kaveh"] {
            let (mut num, mut den) = (0.0, 0.0);
            for (cell, v) in tab.get("table")?.as_object()? {
                if !cell.starts_with(pre) || !cell.ends_with(&format!("| {bin}")) { continue; }
                if let Some(e) = v.get("queens_on") {
                    let n = (e["n_fell"].as_i64().unwrap_or(0) + e["n_kept"].as_i64().unwrap_or(0)) as f64;
                    num += e["diff"].as_f64().unwrap_or(0.0) * n; den += n;
                }
            }
            if den > 0.0 { return Some(num / den); }
        }
        None
    }

    fn sq_of(name: &str) -> Option<Square> { name.get(name.len().saturating_sub(2)..)?.parse().ok() }

    pub(super) fn decide_by_chain(&mut self, bonus: &HashMap<String, f64>, questions: u32, max_cand: usize) -> Value {
        let root = self.pos.clone();
        let us = root.turn();
        let (ua, oa) = (actor_of(us), actor_of(!us));
        let _ = self.run_chain(questions);
        let recs = self.ag.chains.clone();
        let root_cands = self.cands.clone();
        let (pokes_n, best_root) = (recs.iter().filter(|r| r.kind == "POKE" && r.actor == ua).count(), self.ag.best);
        let qsq = self.qsq.clone();
        // ---- constraints (findings against us) and opportunities (findings for us)
        let mut cons: Vec<Item> = vec![];
        let mut opps: Vec<Item> = vec![];
        for r in &recs {
            let sq: Vec<Square> = qsq.get(&r.q).cloned().unwrap_or_default();
            let (lo, hi) = r.value.unwrap_or((0, 0));
            match (r.kind.as_str(), r.actor == oa) {
                ("SAFE", true) if !r.args.iter().any(|a| a.starts_with('@')) => {
                    let p = r.args.first().cloned().unwrap_or_default();
                    let st = Self::sq_of(&p).and_then(|s| root.board().piece_at(s)).map_or(0, |x| val(x.role));
                    cons.push(Item { text: format!("{p} is attacked: {}", r.answer), stake: if hi >= 1 { hi } else { st }, lo, hi, squares: sq, target: Self::sq_of(&p), deflect: false, moves: vec![] });
                }
                ("DEFLECT", true) if hi >= 1 => {
                    let pr: Vec<&String> = r.args.iter().filter(|a| !a.starts_with('@')).collect();
                    cons.push(Item { text: format!("{} must not be deflectable: {} falls ({}{})", pr[0], pr.get(1).map(|x| x.as_str()).unwrap_or("?"), hi, if lo >= hi { ", solver-confirmed" } else { ", estimated, not confirmed" }), stake: hi, lo, hi, squares: sq, target: pr.get(1).and_then(|x| Self::sq_of(x)), deflect: true, moves: vec![] });
                }
                ("ASSESS", false) if hi >= 1 => {
                    let u = r.args.first().cloned().unwrap_or_default();
                    opps.push(Item { text: format!("{}: {}", r.text.trim_start_matches("assess "), r.answer.split(" -> ").last().unwrap_or("")), stake: hi, lo, hi, squares: vec![], target: None, deflect: false, moves: vec![u] });
                }
                ("SAFE", false) if lo >= 1 && !r.args.iter().any(|a| a.starts_with('@')) => {
                    let p = r.args.first().cloned().unwrap_or_default();
                    let ms: Vec<String> = Self::sq_of(&p).map(|s| root.legal_moves().iter().filter(|m| m.to() == s && m.is_capture()).map(|m| m.to_uci(CastlingMode::Standard).to_string()).collect()).unwrap_or_default();
                    opps.push(Item { text: format!("{p} can be won: {}", r.answer), stake: lo, lo, hi, squares: sq, target: Self::sq_of(&p), deflect: false, moves: ms });
                }
                _ => {}
            }
        }
        cons.sort_by(|a, b| b.stake.cmp(&a.stake));
        { let mut seen: Vec<(Option<Square>, bool)> = vec![]; cons.retain(|c| { let k = (c.target, c.deflect); if seen.contains(&k) { false } else { seen.push(k); true } }); }
        opps.sort_by(|a, b| b.stake.cmp(&a.stake));
        // ---- candidates
        let legal: Vec<Move> = root.legal_moves().into_iter().collect();
        let mut score: HashMap<String, i64> = HashMap::new();
        let mut tags: HashMap<String, Vec<&'static str>> = HashMap::new();
        for o in &opps { for u in &o.moves { *score.entry(u.clone()).or_insert(0) += o.stake; tags.entry(u.clone()).or_default().push("opportunity"); } }
        for rc in root_cands.iter().filter(|c| c.actor == ua) { score.entry(rc.uci.clone()).or_insert(0); tags.entry(rc.uci.clone()).or_default().push("poke"); }
        let all_sq: Vec<(Square, i64)> = cons.iter().flat_map(|c| c.squares.iter().map(move |s| (*s, c.stake))).collect();
        // in check every legal move is an evasion (the rules of the game, not learned): each is a candidate, else the decision is empty
        if root.is_check() { for m in &legal { let u = m.to_uci(CastlingMode::Standard).to_string(); *score.entry(u.clone()).or_insert(0) += 1; tags.entry(u).or_default().push("evasion"); } }
        for m in &legal {
            let u = m.to_uci(CastlingMode::Standard).to_string();
            let mut q = root.clone(); q.play_unchecked(m);
            let att = q.board().attacks_from(m.to());
            let mut w = 0;
            for (s, st) in &all_sq { if m.from() == Some(*s) || m.to() == *s || att.contains(*s) { w += st; } }
            if w > 0 { tags.entry(u.clone()).or_default().push("touch"); }
            if q.is_check() { w += 5; tags.entry(u.clone()).or_default().push("check"); }
            if q.is_checkmate() { w += 1000; tags.entry(u.clone()).or_default().push("mate"); }              // a mate in one is a rule of the game, not learned
            if m.is_capture() { tags.entry(u.clone()).or_default().push("capture"); }
            if w > 0 { *score.entry(u).or_insert(0) += w; }
        }
        // THE PROPOSER'S ADJUSTMENTS (ANALYST_PLAN 40.5): a learned bonus per tag (the base priority of a source of candidates), and candidates a proposed condition adds
        if !self.prop_bonus.is_empty() {
            for (u, ts) in &tags { let b: i64 = ts.iter().map(|t| self.prop_bonus.get(*t).copied().unwrap_or(0)).sum(); if b != 0 { *score.entry(u.clone()).or_insert(0) += b; } }
        }
        for u in &self.prop_extra { if legal.iter().any(|m| m.to_uci(CastlingMode::Standard).to_string() == *u) { *score.entry(u.clone()).or_insert(0) += 100_000; tags.entry(u.clone()).or_default().push("proposed"); } }
        let quiet = cons.iter().all(|c| c.hi == 0 && c.lo == 0 && !c.deflect) && opps.is_empty() && best_root[ua as usize] <= 0;
        let nothing_threatened = cons.iter().all(|c| c.hi <= 0) ;
        let mut cand_ucis: Vec<(String, i64)> = score.into_iter().collect();
        cand_ucis.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        let default_mode = nothing_threatened && opps.is_empty() && root_cands.iter().all(|c| c.actor != ua || c.lo <= 0);
        if default_mode {
            // the opening default: the tie-break rule's best moves first, else every legal move
            let mut all: Vec<(String, i64)> = legal.iter().map(|m| { let u = m.to_uci(CastlingMode::Standard).to_string(); let b = bonus.get(&u).copied().unwrap_or(0.0); (u, (b * 10.0) as i64) }).collect();
            let top = all.iter().map(|x| x.1).max().unwrap_or(0);
            all.retain(|x| x.1 == top);
            all.sort_by(|a, b| a.0.cmp(&b.0));
            cand_ucis = all;
        }
        let generated: Vec<String> = cand_ucis.iter().map(|x| x.0.clone()).collect();
        cand_ucis.truncate(max_cand);
        let tscore = self.trade_score();
        let mut rows: Vec<Value> = vec![];
        struct Row { depth: u32, chain_q: u32, uci: String, san: String, lo: i64, hi: i64, tsc: i64, opened: i64, against: i64, risk: i64, meets: Vec<String>, proven: bool, trades: bool, proof_plies: u32, counter: i64, counter_plies: u32, gain: i64, hi0: i64 }
        let mut results: Vec<Row> = vec![];
        let saved_cands = self.cands.clone();
        for (u, _) in &cand_ucis {
            if self.deadline.map_or(false, |d| std::time::Instant::now() > d) && !results.is_empty() { self.cut = true; break; }
            let Ok(um) = u.parse::<shakmaty::uci::UciMove>() else { continue };
            let Ok(m) = um.to_move(&root) else { continue };
            let san = crate::plan::san(&root, u);
            let g = gain_of(&m);
            let trades = m.capture() == Some(Role::Queen);
            let (from, to) = (m.from(), m.to());
            let snap = self.enter(&m);
            // the loop on the position after the move, with the budget of the stake
            let budget = (24 + 4 * cand_ucis.iter().find(|x| &x.0 == u).map_or(0, |x| x.1).min(9)).min(questions as i64) as u32;
            let _ = self.run_chain(budget);
            let after = self.ag.chains.clone();
            let opp = !us;
            let opp_sure = self.ag.best[oa as usize].max(0);
            // what the other side can surely or statically get: SAFE at the root of the position, the assessed pokes, the CONFIRMED deflections and uncoverings; what the exact
            // solver did not confirm (a deflection, a shield that moves) is the candidate's RISK, kept apart (a tie-break), never subtracted
            let at_root = |r: &crate::agenda::ChainRec| !r.args.iter().any(|a| a.starts_with('@'));
            let opp_maybe = after.iter().filter(|r| r.actor == oa).map(|r| match r.kind.as_str() {
                "SAFE" if at_root(r) => r.value.map_or(0, |v| v.1),
                "ASSESS" => r.value.map_or(0, |v| v.1),
                "DEFLECT" | "UNCOVER" => r.value.map_or(0, |v| v.0),
                _ => 0 }).max().unwrap_or(0).max(opp_sure);
            let risk = after.iter().filter(|r| r.actor == oa && matches!(r.kind.as_str(), "DEFLECT" | "UNCOVER")).map(|r| r.value.map_or(0, |v| v.1)).max().unwrap_or(0);
            let opened = after.iter().filter(|r| r.kind == "POKE" && r.actor == ua).count() as i64;
            // the static stand-pat of the side that must answer: it fixes one problem, the rest is lost
            self.lean_mode = true; self.deep = true;
            let sp = if self.pos.is_checkmate() { (-MATE, -MATE) } else if self.pos.legal_moves().is_empty() { (0, 0) } else if self.pos.is_check() { self.check_value() } else { self.stand_pat() };
            let mut lo = g - sp.1.max(opp_maybe);
            let mut hi = g - sp.0.max(opp_sure);
            if self.pos.is_checkmate() { lo = MATE; hi = MATE; }
            hi = hi.max(lo);
            // the root chain's assessment of this very move (the exact solver confirmed a forced gain) is a sound lower bound
            let mut proven = false;
            if let Some(rc) = root_cands.iter().find(|c| &c.uci == u && c.actor == ua) { if rc.lo >= 1 && rc.lo > lo { lo = rc.lo; proven = true; hi = hi.max(rc.hi); } }
            // which constraints it meets
            let mut meets = vec![];
            for c in &cons {
                if c.stake <= 0 { continue; }
                let t2 = c.target.map(|t| if from == Some(t) { to } else { t });
                let met = if c.deflect {
                    !after.iter().any(|r| r.actor == oa && r.value.map_or(false, |v| v.1 >= 1) && matches!(r.kind.as_str(), "DEFLECT" | "ASSESS" | "UNCOVER") && t2.map_or(false, |t| r.text.contains(&t.to_string()) || r.answer.contains(&t.to_string())))
                } else {
                    t2.map_or(false, |t| from == Some(c.target.unwrap_or(t)) || !attacked(self.pos.board(), t, opp) || c.squares.get(1) == Some(&to))
                };
                if met { let name = c.text.split(':').next().unwrap_or(""); let name = name.split(" is attacked").next().unwrap_or(name); let nm = if c.deflect { name.to_string() } else { format!("{name} attacked") }; if !meets.iter().any(|x: &String| x.starts_with(&nm)) { meets.push(format!("{nm} ({})", c.stake)); } }
            }
            self.leave(snap);
            let tsc = if trades { (tscore.unwrap_or(0.0) * 1000.0) as i64 } else { 0 };
            results.push(Row { depth: after.iter().map(|r| r.depth + 1).max().unwrap_or(0), chain_q: self.ag.spent, uci: u.clone(), san, lo, hi, tsc, opened, against: opp_maybe, risk, meets, proven, trades, proof_plies: 0, counter: 0, counter_plies: 0, gain: g, hi0: hi });
        }
        self.cands = saved_cands;
        self.ag_on = false;
        // THE EXACT SOLVER CONFIRMS (never finds) over the candidates at the root: the gain each forces within three moves, all at once (one call per size)
        {
            let fen = shakmaty::fen::Fen::from_position(root.clone(), shakmaty::EnPassantMode::Legal).to_string();
            crate::goals::solver_nodes_reset();
            let mut v = 1;
            while v <= 20 {
                let res = crate::goals::forced_moves(fen.clone(), v, 3, 300_000);
                let mut any = false;
                for r in results.iter_mut() {
                    if let Some(x) = res.iter().find(|x| x.0 == r.uci && x.1 == "forced") { any = true; if v > r.lo.max(0) || !r.proven { r.lo = r.lo.max(v); r.hi = r.hi.max(v); r.proven = true; r.proof_plies = 2 * x.2 - 1; } }
                }
                if !any { break; }
                v += 1;
            }
            // a mate is a gain no material reaches
            let res = crate::goals::forced_moves(fen.clone(), 100, 3, 300_000);
            for r in results.iter_mut() { if res.iter().any(|x| x.0 == r.uci && x.1 == "forced") { r.lo = MATE; r.hi = MATE; r.proven = true; r.proof_plies = 99; } }
            self.solver_calls += 1;
            // THE TURN-AROUND: after each candidate, what the OTHER side can force within three moves is a sure loss for us, unless our own proof already reaches further
            // (a proof over more plies has seen their counter and what follows it; 0CUWR: Nc5 read 1..3 while Nxa3 forces 2 back over more plies than the proof)
            for r in results.iter_mut() {
                if r.lo >= MATE { continue; }
                let Ok(um) = r.uci.parse::<shakmaty::uci::UciMove>() else { continue };
                let Ok(m) = um.to_move(&root) else { continue };
                let q = crate::after(&root, &m);
                if q.legal_moves().is_empty() { continue; }
                let qfen = shakmaty::fen::Fen::from_position(q.clone(), shakmaty::EnPassantMode::Legal).to_string();
                let (mut counter, mut plies, mut v) = (0i64, 0u32, 1i64);
                while v <= 20 {
                    let res = crate::goals::forced_moves(qfen.clone(), v, 3, 300_000);
                    self.solver_calls += 1;
                    match res.iter().filter(|x| x.1 == "forced").map(|x| x.2).min() { Some(n) => { counter = v; plies = 2 * n; v += 1; } None => break }
                }
                if counter == 0 { if crate::goals::forced_moves(qfen.clone(), 100, 3, 300_000).iter().any(|x| x.1 == "forced") { counter = MATE; plies = 99; } }
                if counter > 0 && plies > r.proof_plies {
                    let cap = r.gain - counter;
                    r.counter = counter; r.counter_plies = plies;
                    r.lo = r.lo.min(cap); r.hi = r.hi.min(cap.max(r.lo)); r.proven = false;
                }
            }
            // THE HORIZON IS NOT A NUMBER (Kaveh: "keep going until we see something bigger, then turn around"): the candidates still in the running (their hopeful end reaches the best
            // sure gain) are checked one move deeper, under the node budget; a budget hit leaves them as they were (0CUWR: Kxc2 is forced only at the fourth move)
            if !default_mode && self.deadline.map_or(true, |d| std::time::Instant::now() < d) {
                let best_lo = results.iter().map(|r| r.lo).max().unwrap_or(0);
                let live: Vec<String> = results.iter().filter(|r| r.hi.max(r.hi0) >= best_lo && r.lo < MATE).map(|r| r.uci.clone()).collect();   // the hopeful end BEFORE the counter: a counter is only as deep as three moves
                if !live.is_empty() {
                    let mut v = best_lo.max(1);                                       // a forced gain is monotone in the size: only sizes that reach the best known matter
                    while v <= 20 {
                        let res = crate::goals::forced_moves(fen.clone(), v, 4, 1_000_000);
                        let mut any = false;
                        for r in results.iter_mut().filter(|r| live.contains(&r.uci)) {
                            if let Some(x) = res.iter().find(|x| x.0 == r.uci && x.1 == "forced") { any = true; let plies = 2 * x.2 - 1; if r.counter > 0 && plies < r.counter_plies { continue; }   /* a proof over fewer plies than their counter has not seen it */ if r.counter > 0 { r.counter = 0; r.hi = r.hi.max(r.gain); } if v > r.lo || !r.proven { r.lo = r.lo.max(v); r.hi = r.hi.max(v); r.proven = true; r.proof_plies = r.proof_plies.max(plies); } }
                        }
                        self.solver_calls += 1;
                        if !any { break; }
                        v += 1;
                    }
                }
            }
        }
        // ranking: what it surely gets, then the hopeful end, then (equal) the table's reading of a queen trade, then the unconfirmed risk; in the default mode the opening principles first, then the
        // opportunities it opens at the smallest stake against us
        results.sort_by(|a, b| if default_mode {
            let (ba, bb) = (bonus.get(&a.uci).copied().unwrap_or(0.0), bonus.get(&b.uci).copied().unwrap_or(0.0));
            bb.partial_cmp(&ba).unwrap().then(b.lo.cmp(&a.lo)).then(b.opened.cmp(&a.opened)).then(a.against.cmp(&b.against)).then(a.uci.cmp(&b.uci))
        } else {
            b.lo.cmp(&a.lo).then(b.hi.cmp(&a.hi)).then(b.tsc.cmp(&a.tsc)).then(a.risk.cmp(&b.risk)).then(b.meets.len().cmp(&a.meets.len())).then(a.uci.cmp(&b.uci))
        });
        let mut lines: Vec<String> = vec![format!("DECISION BY THE CHAIN for {} (ANALYST_PLAN 40.4: constraints on the move, opportunities, candidates each checked by running the loop after it)", who(us))];
        lines.push(format!("CONSTRAINTS on {}'s move ({}; the biggest stakes first):", who(us), cons.len()));
        for c in cons.iter().take(8) { lines.push(format!("  [{}] {}", c.stake, c.text)); }
        lines.push(format!("OPPORTUNITIES for {} ({}):", who(us), opps.len()));
        for o in opps.iter().take(8) { lines.push(format!("  [{}] {}", o.stake, o.text)); }
        if default_mode { lines.push(format!("Nothing is threatened and nothing is to be won: the opening default -- the tie-break principles' best moves, then the move that opens the most opportunities ({} pokes seen at the root) at the smallest stake against us.", pokes_n)); }
        lines.push(format!("CANDIDATES ({} checked){}:", results.len(), tscore.map_or(String::new(), |t| format!("; a queen capture carries the table's {:+.3}", t))));
        for r in &results {
            lines.push(format!("  {}: worth {}..{}{} (what the other side can still get: up to {}; unconfirmed risk {}; opens {} pokes for us){}{}", r.san, r.lo, r.hi, if r.trades { " [trades queens]" } else { "" }, r.against, r.risk, r.opened,
                               if r.meets.is_empty() { String::new() } else { format!("; meets {}", r.meets.join(", ")) },
                               if r.counter > 0 { format!("; TURN-AROUND: the other side then forces {} back ({} plies, further than our proof's {})", if r.counter >= MATE { "mate".to_string() } else { r.counter.to_string() }, r.counter_plies, r.proof_plies) } else { String::new() }));
        }
        let best = results.first();
        let (mv, san, lo, hi, status) = match best { Some(b) => (b.uci.clone(), b.san.clone(), b.lo, b.hi, if self.cut { "ESTIMATED (cut by the time limit)" } else if b.proven { "PROVEN (solver-confirmed forced gain)" } else { "ESTIMATED" }), None => (String::new(), String::new(), 0, 0, "NONE") };
        if let Some(b) = best {
            lines.push(format!("DECISION: {} worth {}..{} to {}: {}. Why: {}", b.san, b.lo, b.hi, who(us), status,
                               if b.meets.is_empty() { if default_mode { "the opening default".to_string() } else { "it meets no listed constraint and has the best interval".to_string() } } else { format!("it meets {}", b.meets.join(", ")) }));
        }
        for r in &results { rows.push(json!({"uci": r.uci, "san": r.san, "lo": r.lo, "hi": r.hi, "meets": r.meets, "against": r.against, "opened": r.opened, "depth": r.depth, "chain_q": r.chain_q, "counter": r.counter, "proof_plies": r.proof_plies})); }
        json!({"move": mv, "san": san, "interval": [lo, hi], "status": status, "lines": lines, "generated": generated, "cut": self.cut, "tags": tags.iter().map(|(k, v)| (k.clone(), v.clone())).collect::<HashMap<_, _>>(), "legal": legal.len(), "questions": self.lem.q_total, "lemma": self.lem.report(), "facts": self.facts_n, "candidates": rows, "default": default_mode, "quiet": quiet,
               "constraints": cons.iter().map(|c| json!({"text": c.text, "stake": c.stake})).collect::<Vec<_>>(), "opportunities": opps.iter().map(|c| json!({"text": c.text, "stake": c.stake})).collect::<Vec<_>>()})
    }
}


// =====================================================================================================================================
// PHASE 3 (ANALYST_PLAN 40.3): LEMMAS BY CHUNKING. A settled chain's SHAPE -- the relations among the pieces its question touched (attacks, which between pieces of one side is
// guard-of; the kind and the side of each piece, the side relative to the actor), squares abstracted, so a mirror or a translation of the position gives the same shape -- is the
// PREMISE; the OUTCOME is the question kinds of the chain below it, in order, and its value. A premise that settled the same way on every chain of that shape, enough times (a knob,
// never below 5), is PROMOTED: the question is answered by it in one step. A promoted lemma a chain contradicts is demoted and the contradiction kept (a surprise). The status of a
// lemma is CONJECTURE: there is no Lean proof of any of them. The relations are the vocabulary's own (`attacks` and the piece kind and side; no new word).
#[derive(Clone, Debug)]
pub struct Promoted { pub id: String, pub outcome: String, pub count: i64, pub value: Option<(i64, i64)> }

#[derive(Default)]
pub struct LemmaState {
    pub on: bool, pub learn: bool, pub stake_knob: i64, heldout_every: u32,
    promoted: HashMap<String, Promoted>,
    pub pending: HashMap<usize, (String, String)>,
    pub premise_of_q: HashMap<usize, String>,
    pub obs: HashMap<String, HashMap<String, (i64, Option<(i64, i64)>, Vec<String>)>>,
    pub hits: i64, pub hit_by: HashMap<String, i64>, pub confirms: i64, pub contradictions: Vec<Value>, pub q_total: u64, seen: u64,
}
impl LemmaState {
    pub fn load(&mut self, text: &str, stake_knob: i64, heldout_every: u32, learn: bool) -> Result<(), String> {
        self.on = true; self.learn = learn; self.stake_knob = stake_knob; self.heldout_every = heldout_every.max(1);
        self.promoted.clear();
        if text.trim().is_empty() { return Ok(()); }
        let v: Value = serde_json::from_str(text).map_err(|e| e.to_string())?;
        if let Some(ls) = v.get("lemmas").and_then(|x| x.as_object()) {
            for (k, l) in ls {
                if l["promoted"].as_bool() != Some(true) || l["demoted"].as_bool() == Some(true) { continue; }
                let o = l["outcome"].as_str().unwrap_or("").to_string();
                let val = l["value"].as_array().and_then(|a| Some((a.get(0)?.as_i64()?, a.get(1)?.as_i64()?)));
                self.promoted.insert(k.clone(), Promoted { id: l["id"].as_str().unwrap_or(k).to_string(), outcome: o, count: l["count"].as_i64().unwrap_or(0), value: val });
            }
        }
        Ok(())
    }
    pub fn promoted(&self, k: &str) -> Option<Promoted> { self.promoted.get(k).cloned() }
    pub fn heldout(&mut self, k: &str) -> bool { self.seen += 1; (hash_of(k).wrapping_add(self.seen)) % self.heldout_every as u64 == 0 }
    pub fn report(&self) -> Value {
        let obs: serde_json::Map<String, Value> = self.obs.iter().map(|(p, os)| (p.clone(), Value::Object(os.iter().map(|(o, (n, v, ev))| (o.clone(), json!({"count": n, "value": v, "evidence": ev}))).collect()))).collect();
        json!({"observations": obs, "hits": self.hits, "hit_by": self.hit_by, "confirms": self.confirms, "contradictions": self.contradictions, "questions": self.q_total, "promoted_loaded": self.promoted.len()})
    }
}

impl ReasoningLedger {
    fn q_label(q: &Question) -> String {
        let (prim, _) = Self::split_args(&q.args);
        format!("{} {}", q.kind, prim.join(","))
    }
    /// The premise of a question of a kind with a shape: a canonical graph of the pieces it touches (None for the kinds without one).
    fn premise_of(&mut self, q: &Question) -> Option<String> {
        if !matches!(q.kind.as_str(), "SAFE" | "DEFLECT" | "POKE") { return None; }
        let (prim, ctx) = Self::split_args(&q.args);
        self.premise_for(&q.kind, &if q.kind == "DEFLECT" { prim.join(",") } else { prim.first().cloned().unwrap_or_default() }, q.actor, &ctx)
    }
    pub(super) fn premise_for(&mut self, kind: &str, arg: &str, actor: u8, ctx: &[String]) -> Option<String> {
        let a = colour(actor);
        let kind = kind.to_string(); let arg = arg.to_string();
        self.with_ctx(ctx, move |s| -> Option<String> {
            let b = s.pos.board().clone();
            let mut nodes: Vec<Square> = vec![];
            let mut extra = String::new();
            let add = |nodes: &mut Vec<Square>, x: Square| { if !nodes.contains(&x) && nodes.len() < 14 { nodes.push(x); } };
            let around = |nodes: &mut Vec<Square>, b: &shakmaty::Board, y: Square| {
                for c in [Color::White, Color::Black] {
                    for x in (b.attacks_to(y, c, b.occupied()) & b.by_color(c)).into_iter() { if b.piece_at(x).map_or(false, |p| p.role != Role::King) || true { add(nodes, x); } }
                }
            };
            match kind.as_str() {
                "SAFE" => {
                    let p = s.pc_at(&arg)?;
                    add(&mut nodes, p.1);
                    around(&mut nodes, &b, p.1);
                    // the jobs: what each defender also guards that the enemy attacks
                    let owner = b.color_at(p.1)?;
                    let defs: Vec<Square> = (b.attacks_to(p.1, owner, b.occupied()) & b.by_color(owner)).into_iter().collect();
                    for d in defs { for z in b.attacks_from(d) { if b.color_at(z) == Some(owner) && attacked(&b, z, !owner) && z != p.1 { add(&mut nodes, z); around(&mut nodes, &b, z); } } }
                }
                "DEFLECT" => {
                    let mut it = arg.split(',');
                    let h = s.pc_at(it.next()?)?; let t = it.next().and_then(|x| s.pc_at(x));
                    add(&mut nodes, h.1); around(&mut nodes, &b, h.1);
                    if let Some(t) = t { add(&mut nodes, t.1); around(&mut nodes, &b, t.1); }
                    extra = format!("|held:{}", ctx.last().is_some() as u8);
                }
                "POKE" => {
                    let pb = if s.pos.turn() == a { s.pos.clone() } else { Self::passed(&s.pos)? };
                    let m = arg.parse::<shakmaty::uci::UciMove>().ok()?.to_move(&pb).ok()?;
                    let was = pressure(pb.board(), !a);
                    let mut r = pb.clone(); r.play_unchecked(&m);
                    let rb = r.board().clone();
                    add(&mut nodes, m.to());
                    let now = pressure(&rb, !a);
                    let mut ts: Vec<Square> = vec![];
                    for (sq, (n, mv)) in &now { let (bn, bmv) = was.get(sq).copied().unwrap_or((0, 99)); if (*n > bn || *mv < bmv) && rb.piece_at(*sq).map_or(false, |p| p.role != Role::King) { ts.push(*sq); } }
                    ts.sort_by_key(|x| (std::cmp::Reverse(rb.piece_at(*x).map_or(0, |p| val(p.role))), *x as u32));
                    for x in ts.into_iter().take(3) { add(&mut nodes, x); around(&mut nodes, &rb, x); }
                    extra = format!("|takes:{}|check:{}", m.capture().map_or('-', letter), r.is_check() as u8);
                    return Some(format!("{kind}|{}{extra}", Self::graph(&rb, &nodes, a)));
                }
                _ => return None,
            }
            Some(format!("{kind}|{}{extra}", Self::graph(&b, &nodes, a)))
        }).flatten()
    }
    /// The canonical graph of the pieces on `nodes`: node = side relative to the actor + piece kind, edge = "attacks" (guard-of between pieces of one side). Squares are not in
    /// it, so a mirror or a translation of the position gives the same string; nodes are named by their label and their neighbours' labels, sorted.
    fn graph(b: &shakmaty::Board, nodes: &[Square], a: Color) -> String {
        let lab = |x: Square| b.piece_at(x).map_or("?".to_string(), |p| format!("{}{}", if p.color == a { 'o' } else { 't' }, letter(p.role)));
        let n = nodes.len();
        let att = |i: usize, j: usize| i != j && b.piece_at(nodes[i]).map_or(false, |_| b.attacks_from(nodes[i]).contains(nodes[j]));
        let mut key: Vec<String> = (0..n).map(|i| {
            let mut out: Vec<String> = (0..n).filter(|&j| att(i, j)).map(|j| lab(nodes[j])).collect();
            let mut inn: Vec<String> = (0..n).filter(|&j| att(j, i)).map(|j| lab(nodes[j])).collect();
            out.sort(); inn.sort();
            format!("{}>[{}]<[{}]", lab(nodes[i]), out.join(","), inn.join(","))
        }).collect();
        // one refinement round: a node is also named by the keys of its neighbours
        let k1 = key.clone();
        for i in 0..n {
            let mut nb: Vec<String> = (0..n).filter(|&j| att(i, j) || att(j, i)).map(|j| k1[j].clone()).collect();
            nb.sort();
            key[i] = format!("{}~{}", k1[i], hash_of(&nb.join(";")) % 100000);
        }
        let mut edges: Vec<String> = vec![];
        for i in 0..n { for j in 0..n { if att(i, j) { edges.push(format!("{}->{}", key[i], key[j])); } } }
        edges.sort();
        let mut ks = key.clone(); ks.sort();
        // readable: the pieces (side relative to the actor + kind) and who attacks whom, by label; the hash makes it canonical beyond the labels
        let mut names: Vec<String> = (0..n).map(|i| lab(nodes[i])).collect();
        names.sort();
        let mut rel: Vec<String> = (0..n).flat_map(|i| (0..n).filter(move |&j| i != j).map(move |j| (i, j))).filter(|&(i, j)| att(i, j)).map(|(i, j)| format!("{}>{}", lab(nodes[i]), lab(nodes[j]))).collect();
        rel.sort();
        format!("[{}] [{}] #{}{}", names.join(" "), rel.join(" "), hash_of(&ks.join(";")) % 100000, hash_of(&edges.join(";")) % 100000)
    }

    /// At the end of a run: the outcome of every question that had a premise (the kinds of the chain below it, in order, and its value); the held-out checks compared.
    fn lemma_observe(&mut self) {
        if !self.lem.on { return; }
        let premises: Vec<(usize, String)> = self.lem.premise_of_q.drain().collect();
        let chains = self.ag.chains.clone();
        let fen = shakmaty::fen::Fen::from_position(self.line.first().cloned().unwrap_or_else(|| self.pos.clone()), shakmaty::EnPassantMode::Legal).to_string();
        for (qid, pk) in premises {
            let Some(rec) = chains.iter().find(|r| r.q == qid) else { continue };
            // the kinds below it, in the order they were asked
            let mut below: Vec<usize> = vec![];
            let mut frontier = vec![qid];
            while let Some(x) = frontier.pop() { for r in chains.iter().filter(|r| r.parent == Some(x)) { below.push(r.q); frontier.push(r.q); } }
            below.sort();
            let kinds: Vec<&str> = below.iter().filter_map(|id| chains.iter().find(|r| r.q == *id)).map(|r| r.kind.as_str()).collect();
            let v = rec.value.unwrap_or((0, 0));
            let sig = format!("{} => {}..{}", if kinds.is_empty() { "alone".to_string() } else { kinds.join(">") }, v.0, v.1);
            if let Some((_, expected)) = self.lem.pending.remove(&qid) {
                if expected == sig { self.lem.confirms += 1; }
                else {
                    self.lem.promoted.remove(&pk);
                    self.lem.contradictions.push(json!({"premise": pk, "expected": expected, "got": sig, "fen": fen}));
                }
            }
            if self.lem.learn {
                let e = self.lem.obs.entry(pk).or_default().entry(sig).or_insert((0, Some(v), vec![]));
                e.0 += 1;
                if e.2.len() < 3 && !e.2.contains(&fen) { e.2.push(fen.clone()); }
            }
        }
        // held-out checks whose question was never settled (budget) are dropped
        self.lem.pending.clear();
    }
}
