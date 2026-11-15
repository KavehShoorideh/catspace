//! PLAN 78.43 STAGE C, STEP ONE: THE QUESTION OF THE MOVE AND THE LOSS OF A UNIT, game-free (the Python sheet's `move_question`, `move_order`, `move_sheet`, `unit_loss`, `charged_ply`).
//!
//! THE QUESTION OF THE MOVE: the side to move plays a move; is something of its own lost for it? Three answers, asked in this order: (a) what the move LEAVES (the cost of leaving
//! for the moving unit: units newly lost where they stand, or the goal unit newly under the sign, because the unit is no longer where it was); (b) where it LANDS (on the position after
//! the move the other side is to move: is the moved unit lost where it now stands, by the exchange frame; when the move took something, the balance); (c) what the other side THEN
//! HAS (the mover's other units on places whose content or count of bearing units changed and that the other side now takes, not read by (a) and not lost before; the goal unit's sign).
//! The verdict is DATA: SAFE, LOSES (the unit, the amount as bounds, the line of real moves the frame found) or NOT SETTLED, with the words of each answer made here from the numbers.
//! The claims graph the Python sheet files the answers on is not ported (stage D).
//!
//! THE LOSS OF A UNIT, followed through a game's real moves (`unit_loss`, `charged_ply`): the balance of worth over the run of takings on one place, as plain data over the events of
//! a game (one per move: the unit that moves, where it ends, the unit it takes and where, the kind it has once the move is made).
use super::reading::Reading;
use super::frame::Solved;
use super::settle::Settler;
use super::{Model, Place, PlaceSet, Side, Unit};
use cs_game::placed::Placed;
use cs_game::Game;
use std::collections::HashMap;

pub const SAFE: &str = "SAFE";
pub const LOSES: &str = "LOSES";
pub const NOT_SETTLED: &str = "NOT SETTLED";

/// Python's `format(x, "g")` (and `"+g"` with `plus`): six significant digits, trailing zeros dropped, an exponent form for very small or large numbers.
pub fn fmt_g(x: f64, plus: bool) -> String {
    let sign = if x.is_sign_negative() { "-" } else if plus { "+" } else { "" };
    let a = x.abs();
    if a == 0.0 { return format!("{sign}0"); }
    let sci = format!("{:.5e}", a);
    let (mant, exp) = sci.split_once('e').unwrap();
    let exp: i32 = exp.parse().unwrap();
    if !(-4..6).contains(&exp) {
        let m = if mant.contains('.') { mant.trim_end_matches('0').trim_end_matches('.') } else { mant };
        return format!("{sign}{m}e{}{:02}", if exp < 0 { '-' } else { '+' }, exp.abs());
    }
    let fixed = format!("{:.*}", (5 - exp).max(0) as usize, a);
    let fixed = if fixed.contains('.') { fixed.trim_end_matches('0').trim_end_matches('.').to_string() } else { fixed };
    format!("{sign}{fixed}")
}

fn plural(n: usize) -> &'static str { if n != 1 { "s" } else { "" } }

/// One threat found by one of the three answers.
#[derive(Debug, Clone)]
pub struct Threat { pub unit: Unit, pub place: Place, pub part: char, pub kind: &'static str, pub value: f64, pub line: Vec<String>, pub open: bool, pub words: String }

/// An answer the frames left open (or a goal unit newly under the sign without a loss).
#[derive(Debug, Clone)]
pub struct Unsettled { pub unit: Unit, pub part: char, pub words: String, pub stake: f64 }

/// The unit the verdict names: a threat (its kind and value kept) or the first thing left open.
#[derive(Debug, Clone)]
pub struct Named { pub unit: Unit, pub place: Place, pub part: char, pub words: String, pub kind: Option<&'static str>, pub value: Option<f64> }

#[derive(Debug, Clone)]
pub struct MoveAnswer {
    pub mv: String,
    pub unit: Unit,
    pub dest: Place,
    pub took: f64,
    pub credit: f64,
    pub took_units: Vec<Unit>,
    pub answers: [String; 3],
    pub threats: Vec<Threat>,
    pub unsettled: Vec<Unsettled>,
    pub verdict: &'static str,
    pub named: Option<Named>,
    pub low: f64,
    pub high: f64,
    pub line: Vec<String>,
    pub because: String,
}

/// What a game's units on a place are, as the Python codec's `changed_places` reads it: the places whose content or whose count of bearing units (either side) differs between two
/// positions.
pub fn changed_places<G: Game + Placed>(g: &G, a: &G::State, b: &G::State, np: usize) -> PlaceSet {
    let census = |s: &G::State| {
        let mut content: Vec<Option<(usize, u8)>> = vec![None; np];
        let mut bearing = vec![[0u32; 2]; np];
        for (o, k, at) in g.cells(s) {
            content[at as usize] = Some((o, k));
            for p in g.bearing_from(s, o, k, at) { bearing[p as usize][o] += 1; }
        }
        (content, bearing)
    };
    let (ca, ba) = census(a);
    let (cb, bb) = census(b);
    PlaceSet::from_places(np, (0..np).filter(|i| ca[*i] != cb[*i] || ba[*i] != bb[*i]).map(|i| i as Place))
}

fn names_of<G: Game + Placed>(r: &Reading<G>, us: &[Unit]) -> String {
    if us.is_empty() { "none".to_string() } else { us.iter().map(|u| r.m.names.words(u)).collect::<Vec<_>>().join(", ") }
}

/// The units of the other side that the move took: the other side's units before the move that are not the child's units of the side to move (a multiset difference).
fn taken_by<G: Game + Placed>(r: &Reading<G>, child_r: &Reading<G>) -> Vec<Unit> {
    let mut have: HashMap<(u8, Place), usize> = HashMap::new();
    for u in &child_r.units { if u.side == Side::Us { *have.entry((u.kind, u.at)).or_insert(0) += 1; } }
    let mut out = vec![];
    for u in &r.units {
        if u.side != Side::Them { continue; }
        match have.get_mut(&(u.kind, u.at)) { Some(n) if *n > 0 => *n -= 1, _ => out.push(*u) }
    }
    out
}

fn source_words(source: &str) -> &'static str {
    match source {
        "route" => "the whole route followed, every answer, real moves",
        "real" => "real moves",
        "given" => "the way's first move played, every answer, real moves",
        _ => "read from the board as it stands, lines behind not read",
    }
}

impl<'a, 'g, G: Game + Placed> Settler<'a, 'g, G> {
    /// Does some legal move of the side to move in `s` land on the place of the unit `t` while a unit stands there.
    fn bears_on_in(&self, s: &G::State, t: &Unit) -> bool {
        let g = self.r.g;
        g.kind_at(s, t.at).is_some() && g.moves(s).iter().any(|m| g.move_ends(s, m).1 == t.at)
    }

    /// THE QUESTION OF THE MOVE `mv` of the side to move. Err when `mv` is not a move of a unit of the side to move.
    pub fn move_question(&self, mv: &G::Move) -> Result<MoveAnswer, String> {
        let (r, g, p) = (self.r, self.r.g, self.p);
        let n = &r.m.names;
        let (origin, dest) = g.move_ends(&r.s, mv);
        let u = *r.units.iter().find(|x| x.side == Side::Us && x.at == origin).ok_or("not a move of a unit of the side to move")?;
        let mut out = MoveAnswer { mv: g.move_name(&r.s, mv), unit: u, dest, took: 0.0, credit: 0.0, took_units: vec![], answers: [String::new(), String::new(), String::new()], threats: vec![], unsettled: vec![],
                                   verdict: SAFE, named: None, low: 0.0, high: 0.0, line: vec![], because: String::new() };
        let child = g.apply(&r.s, mv);
        if g.outcome(&child).is_some() {
            out.because = "the move ends the game; nothing is left to lose".into();
            out.answers = ["not read: the game is over".into(), "not read: the game is over".into(), "not read: the game is over".into()];
            return Ok(out);
        }
        let weight = r.weight(&u);
        let mut child_r = match Reading::new(g, child.clone(), r.m) {
            Ok(c) => c,
            Err(_) => { out.verdict = NOT_SETTLED; out.because = "the position after the move cannot be read".into(); return Ok(out); }
        };
        child_r.params_k = p.reach_k;
        let child_s = Settler::new(&child_r, p);
        let mine_now: HashMap<Place, Unit> = child_r.units.iter().filter(|x| x.side == Side::Them).map(|x| (x.at, *x)).collect();
        let took_units = taken_by(r, &child_r);
        let took: f64 = took_units.iter().map(|x| r.worth_of(x.kind)).sum();
        let credit = match mine_now.get(&dest) { Some(l) if l.kind != u.kind => r.worth_of(l.kind) - r.worth_of(u.kind), _ => 0.0 };
        out.took = took; out.took_units = took_units.clone(); out.credit = credit;
        let gain = took + credit;
        let words = |x: &Unit| n.words(x);
        let (mut threats, mut unsettled): (Vec<Threat>, Vec<Unsettled>) = (vec![], vec![]);

        // a. what it leaves
        let leave = self.cost_of_leaving(Side::Us, &u, Some(mv), &r.s);
        let cands: Vec<Unit> = if leave.read { self.leaving_candidates(Side::Us, &u, Some(dest)).0 } else { vec![] };
        for d in &leave.duties {
            let v = d.unit;
            if d.kind == "newly lost" {
                threats.push(Threat { unit: v, place: v.at, part: 'a', kind: "left unguarded", value: d.value, line: vec![], open: false,
                                      words: format!("{} is lost where it stands once {} has gone", words(&v), words(&u)) });
            } else {
                unsettled.push(Unsettled { unit: v, part: 'a', words: format!("{} comes newly under the sign once {} has gone", words(&v), words(&u)), stake: d.stake });
            }
        }
        out.answers[0] = if leave.read {
            let all: Vec<&str> = threats.iter().map(|t| t.words.as_str()).chain(unsettled.iter().map(|x| x.words.as_str())).collect();
            if all.is_empty() { format!("leaving its place newly loses nothing ({} unit{} it held were read)", cands.len(), plural(cands.len())) } else { all.join("; ") }
        } else { "not read".into() };

        // b. where it lands
        let land = mine_now.get(&dest).copied();
        let dest_text = n.place_text(dest);
        match land {
            None => out.answers[1] = "not read: the moved unit is not where the move ends".into(),
            Some(l) if !child_s.weighty(&l) => {
                out.answers[1] = format!("my {} on {} is worth the least of all, not asked", n.kind_text(l.kind), dest_text)
                    + &if took_units.is_empty() { String::new() } else { format!("; it took {} worth {}", names_of(r, &took_units), fmt_g(took, false)) };
            }
            Some(l) => {
                let ex: std::rc::Rc<Solved> = child_s.exchange_stands(&l, Side::Us);
                let mine = format!("my {} on {}", n.kind_text(l.kind), dest_text);
                let mut wh = if ex.value > 0.0 { format!("the other side's exchange on {} is worth {} to it", mine, fmt_g(ex.value, false)) } else { format!("{} is not lost where it lands", mine) };
                if credit != 0.0 { wh += &format!("; it arrives as a {}, worth {} more than it was", n.kind_text(l.kind), fmt_g(credit, true)); }
                if !took_units.is_empty() { wh += &format!("; the move took {} worth {}, so the balance is {}", names_of(r, &took_units), fmt_g(took, false), fmt_g(gain - ex.value, true)); }
                else if credit != 0.0 { wh += &format!(", so the balance is {}", fmt_g(gain - ex.value, true)); }
                out.answers[1] = format!("{} ({})", wh, source_words(ex.source));
                if ex.value > 0.0 {
                    threats.push(Threat { unit: u, place: origin, part: 'b', kind: "lands", value: ex.value, line: ex.line.clone(), open: ex.open, words: wh });
                } else if ex.open {
                    unsettled.push(Unsettled { unit: u, part: 'b', words: "the exchange where it lands was not read to its end (node cap)".into(), stake: weight });
                }
            }
        }

        // c. what the other side then has
        let changed = changed_places(g, &r.s, &child, r.np());
        let (mut read_units, mut newly): (Vec<Unit>, Vec<String>) = (vec![], vec![]);
        let goal = r.goal_unit(Side::Us);
        for v in &r.units {
            if v.side != Side::Us || *v == u || cands.contains(v) || !self.weighty(v) || Some(*v) == goal || !changed.contains(v.at) { continue; }
            let Some(vc) = mine_now.get(&v.at).copied() else { continue };
            if !self.bears_on_in(&child, &vc) { continue; }
            read_units.push(*v);
            if self.exchange_stands(v, Side::Them).value > 0.0 { continue; }
            let ex = child_s.exchange_stands(&vc, Side::Us);
            if ex.value > 0.0 {
                let w = format!("{} is lost where it stands after the move: worth {} to the other side", words(v), fmt_g(ex.value, false));
                threats.push(Threat { unit: *v, place: v.at, part: 'c', kind: "newly struck", value: ex.value, line: ex.line.clone(), open: ex.open, words: w.clone() });
                newly.push(w);
            } else if ex.open {
                unsettled.push(Unsettled { unit: *v, part: 'c', words: format!("the exchange on {} was not read to its end (node cap)", words(v)), stake: r.weight(v) });
            }
        }
        if let Some(gu) = goal {
            let near = changed.contains(gu.at) || r.exits_of(&gu).iter().any(|e| changed.contains(e));
            if gu != u && !unsettled.iter().any(|x| x.unit == gu) && near && self.goal_newly_under_sign(Side::Us, &child) {
                unsettled.push(Unsettled { unit: gu, part: 'c', words: format!("{} comes newly under the sign", words(&gu)), stake: r.weight(&gu) });
            }
        }
        let cwords: Vec<&str> = newly.iter().map(|s| s.as_str()).chain(unsettled.iter().filter(|x| x.part == 'c').map(|x| x.words.as_str())).collect();
        out.answers[2] = if cwords.is_empty() { format!("{} unit{} on changed places read; none newly lost ({} places changed)", read_units.len(), plural(read_units.len()), changed.len()) } else { cwords.join("; ") };

        // the verdict
        let mut sure: Vec<&Threat> = threats.iter().filter(|t| t.value - gain > 0.0).collect();
        sure.sort_by(|a, b| b.value.partial_cmp(&a.value).unwrap().then_with(|| a.part.cmp(&b.part)));
        if let Some(worst) = sure.first() {
            let lo = worst.value - gain;
            let hi = lo.max(threats.iter().map(|t| t.value).sum::<f64>() - gain);
            let mut because = worst.words.clone();
            if !took_units.is_empty() { because += &format!("; the move took {} worth {}", names_of(r, &took_units), fmt_g(took, false)); }
            if credit != 0.0 { because += &format!("{}it arrives worth {}", if took_units.is_empty() { "; " } else { " and " }, fmt_g(credit, true)); }
            if gain != 0.0 { because += &format!(", which leaves {}", fmt_g(lo, false)); }
            out.verdict = LOSES;
            out.named = Some(Named { unit: worst.unit, place: worst.place, part: worst.part, words: worst.words.clone(), kind: Some(worst.kind), value: Some(worst.value) });
            out.low = lo; out.high = hi; out.line = worst.line.clone(); out.because = because;
        } else if let Some(first) = unsettled.first() {
            out.verdict = NOT_SETTLED;
            out.named = Some(Named { unit: first.unit, place: first.unit.at, part: first.part, words: first.words.clone(), kind: None, value: None });
            out.low = 0.0;
            out.high = unsettled.iter().map(|x| x.stake).fold(f64::NEG_INFINITY, f64::max);
            out.because = first.words.clone();
        } else {
            out.because = format!("nothing of its own is lost: {}", out.answers.join("; "));
        }
        out.threats = threats;
        out.unsettled = unsettled;
        Ok(out)
    }

    /// The moves in the order they are asked: by the stake of the moving unit times whether its landing place is struck by the other side in the present reading (struck places first, the
    /// heavier unit first), then the rest, the heavier unit first; ties in the game's own order.
    pub fn move_order(&self, moves: &[G::Move]) -> Vec<usize> {
        let r = self.r;
        let struck = r.struck_by(Side::Them);
        let stake_of = |m: &G::Move| { let (o, _) = r.g.move_ends(&r.s, m); r.units.iter().find(|x| x.side == Side::Us && x.at == o).map_or(0.0, |x| r.weight(x)) };
        let mut keyed: Vec<(f64, f64, usize)> = moves.iter().enumerate().map(|(i, m)| {
            let stake = stake_of(m);
            let (_, d) = r.g.move_ends(&r.s, m);
            (-(stake * if struck.contains(d) { 1.0 } else { 0.0 }), -stake, i)
        }).collect();
        keyed.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then_with(|| a.1.partial_cmp(&b.1).unwrap()).then_with(|| a.2.cmp(&b.2)));
        keyed.into_iter().map(|x| x.2).collect()
    }

    /// The question of the move for each legal move of the position, in `move_order`, at most `max_moves` of them (the rest are `unread`).
    pub fn move_sheet(&self, max_moves: Option<usize>) -> MoveSheet {
        let moves = self.r.g.moves(&self.r.s);
        let order = self.move_order(&moves);
        let mut results = vec![];
        for (k, &i) in order.iter().enumerate() {
            if max_moves.is_some_and(|m| k >= m) { break; }
            if let Ok(a) = self.move_question(&moves[i]) { results.push(a); }
        }
        let names: Vec<String> = order.iter().map(|&i| self.r.g.move_name(&self.r.s, &moves[i])).collect();
        let unread = names[results.len().min(names.len())..].to_vec();
        MoveSheet { results, unread, order: names }
    }
}

#[derive(Debug, Clone)]
pub struct MoveSheet { pub results: Vec<MoveAnswer>, pub unread: Vec<String>, pub order: Vec<String> }

/// The model's worth of a kind by the game's word (a kind the table does not name is worth nothing).
pub fn worth_by_word(m: &Model, kind: &str) -> f64 { m.names.kind(kind).map_or(0.0, |k| m.worth[k as usize]) }

// ---------------------------------------------------------------------------------------------------------------------------------------------- the loss of a unit

/// One move of a game as the loss reads it: the unit that moves (id, side, kind), where it ends, the unit it takes (id, side, kind) and where it stood, the kind the mover has once the
/// move is made.
#[derive(Debug, Clone)]
pub struct Event {
    pub mover: (i64, String, String),
    pub dest: String,
    pub taken: Option<(i64, String, String)>,
    pub taken_at: Option<String>,
    pub arrives: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LossStatus { Lost, Even, Gained }

impl LossStatus {
    pub fn word(self) -> &'static str { match self { LossStatus::Lost => "lost", LossStatus::Even => "even", LossStatus::Gained => "gained" } }
}

#[derive(Debug, Clone)]
pub struct Loss { pub status: LossStatus, pub balance: f64, pub place: String, pub ply: usize, pub run: Vec<usize>, pub arrival: Option<usize> }

fn changes_kind(e: &Event) -> bool { e.arrives.as_ref().is_some_and(|a| *a != e.mover.2) }

/// THE LOSS OF A UNIT followed through a game's real moves: the unit `unit_id` is looked for among the takings of the plies `start` .. `start + horizon - 1`; the RUN is the consecutive
/// takings on the place it was taken on (the forward end is not cut by the horizon) plus the unit's own arriving move (a taking, or an arrival as another kind); the BALANCE is the
/// worth taken from the other side less the worth taken from the unit's own side, over the run, less what the change of kind gained. LOST above 0, EVEN at 0, GAINED below.
pub fn unit_loss(events: &[Event], unit_id: i64, start: usize, horizon: usize, worth: &dyn Fn(&str) -> f64) -> Option<Loss> {
    let end = (start + horizon).min(events.len());
    let hit = (start..end).find(|&i| events[i].taken.as_ref().is_some_and(|t| t.0 == unit_id))?;
    let (side, place) = (events[hit].taken.as_ref().unwrap().1.clone(), events[hit].taken_at.clone());
    let mut run = vec![hit];
    let mut j = hit;
    while j > start && events[j - 1].taken.is_some() && events[j - 1].taken_at == place { run.insert(0, j - 1); j -= 1; }
    let mut k = hit + 1;
    while k < events.len() && events[k].taken.is_some() && events[k].taken_at == place { run.push(k); k += 1; }
    let mut arrival = None;
    for j in (start..hit).rev() {
        if events[j].mover.0 == unit_id {
            if Some(&events[j].dest) == place.as_ref() && (events[j].taken.is_some() || changes_kind(&events[j])) {
                arrival = Some(j);
                if !run.contains(&j) { run.push(j); run.sort(); }
            }
            break;
        }
    }
    let mut balance = 0.0;
    for &i in &run {
        if let Some(t) = &events[i].taken { balance += (if t.1 == side { 1.0 } else { -1.0 }) * worth(&t.2); }
    }
    if let Some(a) = arrival {
        if changes_kind(&events[a]) { balance -= worth(events[a].arrives.as_ref().unwrap()) - worth(&events[a].mover.2); }
    }
    let status = if balance > 0.0 { LossStatus::Lost } else if balance == 0.0 { LossStatus::Even } else { LossStatus::Gained };
    Some(Loss { status, balance, place: place.unwrap_or_default(), ply: hit, run, arrival })
}

/// THE MOVE A LOSS IS CHARGED TO, for a unit that was LOST: the first taking of its run is the first ply of the run at which a unit of the unit's own side is taken; the charged ply is the
/// last ply before it at which the unit's own side moved. None when its side made no move before that taking.
pub fn charged_ply(events: &[Event], loss: &Loss) -> Option<usize> {
    let side = &events[loss.ply].taken.as_ref().unwrap().1;
    let first = *loss.run.iter().filter(|&&i| events[i].taken.as_ref().is_some_and(|t| t.1 == *side)).min()?;
    (0..first).filter(|&i| events[i].mover.1 == *side).max()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_general_format_is_python_s() {
        assert_eq!(fmt_g(3.0, false), "3");
        assert_eq!(fmt_g(0.5, false), "0.5");
        assert_eq!(fmt_g(1e9, false), "1e+09");
        assert_eq!(fmt_g(123456.0, false), "123456");
        assert_eq!(fmt_g(1234567.0, false), "1.23457e+06");
        assert_eq!(fmt_g(2.0, true), "+2");
        assert_eq!(fmt_g(-2.0, true), "-2");
        assert_eq!(fmt_g(0.0, true), "+0");
        assert_eq!(fmt_g(0.0001, false), "0.0001");
        assert_eq!(fmt_g(0.00001, false), "1e-05");
    }

    fn ev(id: i64, side: &str, kind: &str, dest: &str, taken: Option<(i64, &str, &str)>, at: Option<&str>) -> Event {
        Event { mover: (id, side.into(), kind.into()), dest: dest.into(), taken: taken.map(|t| (t.0, t.1.into(), t.2.into())), taken_at: at.map(|s| s.into()), arrives: Some(kind.into()) }
    }

    #[test]
    fn an_exchange_on_one_place_is_a_balance_of_worth() {
        let worth = |k: &str| match k { "a" => 1.0, "b" => 3.0, _ => 0.0 };
        // ply 0: the side w takes the side b's unit 7 (an a) on x with its unit 2 (a b); ply 1: the side b takes it back on x
        let evs = vec![ev(2, "w", "b", "x", Some((7, "b", "a")), Some("x")), ev(8, "b", "a", "x", Some((2, "w", "b")), Some("x"))];
        let l = unit_loss(&evs, 2, 0, 4, &worth).unwrap();
        assert_eq!((l.status, l.balance, l.run.clone()), (LossStatus::Lost, 2.0, vec![0, 1]));      // it took 1, it lost 3: its side is down 2
        assert!(unit_loss(&evs, 99, 0, 4, &worth).is_none());
        assert_eq!(charged_ply(&evs, &l), Some(0));                                               // the first taking of white's unit is ply 1; white's last move before it is ply 0
    }
}
