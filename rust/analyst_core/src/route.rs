//! PLAN 54 STAGE A: THE SCHEMA VERIFICATION IN RUST. `verify_route` reads, on ONE route x -> y, the binder's schemas (mate_in_one, mate_in_two, the delayed
//! families at depth one, and at depth two on the second moves the caller names) with exactly the semantics of `binder._read` / `_read2` /
//! `puzzle_apply.best_bind`: the legality legs first, the tail read once, `T > B`, then the alive test with C = T - ALIVE_NEED. The terms are built here
//! (a port of selfmine's `delayed_tail`, `tail_body`, `legs_of`, `with_legality`, `mate_atoms`, `mate2_tail`, `render`) and read by the evaluator; no Python
//! work per candidate, no stdout capture. The one thing left to Python is the SAN in a note (shakmaty's SAN is not python-chess's on the null-move board):
//! a note that names a move carries `\u{1}x1y1[,x2y2]\u{2}` and `binder` fills it in with `puzzle_apply._san_pairs` (rare: only the alive-test failures).
use super::*;
use crate::vocab::{is_num, num, parse_term, set};
use pyo3::exceptions::PyValueError;
use std::collections::BTreeMap;

#[derive(Clone)]
enum Sx { A(String), L(Vec<Sx>) }

fn a(s: &str) -> Sx { Sx::A(s.to_string()) }
fn f(h: &str, args: Vec<Sx>) -> Sx { let mut v = vec![a(h)]; v.extend(args); Sx::L(v) }

fn is_square(s: &str) -> bool {
    let b = s.as_bytes();
    b.len() == 2 && (b'a'..=b'h').contains(&b[0]) && (b'1'..=b'8').contains(&b[1])
}
/// An absolute square named from the mover's side (black reads the ranks upside down).
fn view(sq: &str, mover: &str) -> String {
    if mover == "w" { sq.to_string() } else { format!("{}{}", &sq[..1], 9 - (sq.as_bytes()[1] - b'0')) }
}
/// The term as text; with a mover, its squares named from that side (selfmine.render + base_vocabulary.sexpr).
fn text(x: &Sx, mover: Option<&str>) -> String {
    match x {
        Sx::A(s) => match mover { Some(m) if is_square(s) => view(s, m), _ => s.clone() },
        Sx::L(v) => {
            let mut out = String::from("(");
            for (i, e) in v.iter().enumerate() {
                if i > 0 { out.push(' '); }
                if i == 0 { if let Sx::A(h) = e { out.push_str(h); continue; } }      // render leaves the head alone
                out.push_str(&text(e, mover));
            }
            out.push(')');
            out
        }
    }
}

// ------------------------------------------------------------------------------------------------- the words (selfmine)
fn legal(fr: &str, to: &str) -> Sx { f("count", vec![f("and", vec![f("moves", vec![a(fr)]), f("at", vec![a(to)])])]) }
fn ours_at(fr: &str) -> Sx { f("count", vec![f("and", vec![a("ours"), f("at", vec![a(fr)])])]) }
fn no_pawn_at(fr: &str, mover: &str) -> Sx {
    f("count", vec![f("and", vec![a("ourpawns"), f("and", vec![f("at", vec![a(fr)]), f("rank", vec![a(&view("a7", mover))])])])])
}
fn is_no_pawn(g: &Sx) -> bool {
    let mut g = g;
    loop {
        match g {
            Sx::L(v) if matches!(&v[0], Sx::A(h) if h == "after" || h == "least" || h == "most") => g = v.last().unwrap(),
            _ => break,
        }
    }
    if let Sx::L(v) = g {
        if matches!(&v[0], Sx::A(h) if h == "count") && v.len() > 1 {
            if let Sx::L(w) = &v[1] { return w.len() > 1 && matches!(&w[0], Sx::A(h) if h == "and") && matches!(&w[1], Sx::A(h) if h == "ourpawns"); }
        }
    }
    false
}

enum Wrap { After(Sx, Sx), Quant }
fn rebuild(wraps: &[Wrap], mut x: Sx, ge: bool) -> Sx {
    for w in wraps.iter().rev() {
        x = match w {
            Wrap::After(fr, to) => f("after", vec![fr.clone(), to.clone(), x]),
            Wrap::Quant => f(if ge { "least" } else { "most" }, vec![x]),
        };
    }
    x
}
/// selfmine.legs_of: the legality terms of every `after` a term reads, each at its own depth and under the same quantifiers.
fn legs_of(t: &Sx, wraps: &mut Vec<Wrap>, mover: &str, out: &mut Vec<Sx>) {
    let Sx::L(v) = t else { return };
    let head = if let Sx::A(h) = &v[0] { h.as_str() } else { "" };
    if head == "after" && v.len() >= 4 {
        let (fr, to) = (text(&v[1], None), text(&v[2], None));
        out.push(rebuild(wraps, legal(&fr, &to), true));
        out.push(rebuild(wraps, ours_at(&fr), true));
        out.push(rebuild(wraps, no_pawn_at(&fr, mover), false));
        wraps.push(Wrap::After(v[1].clone(), v[2].clone()));
        legs_of(&v[3], wraps, mover, out);
        wraps.pop();
    } else if (head == "least" || head == "most") && v.len() >= 2 {
        wraps.push(Wrap::Quant);
        legs_of(&v[1], wraps, mover, out);
        wraps.pop();
    } else {
        for x in &v[1..] { legs_of(x, wraps, mover, out); }
    }
}

fn wrap_tree(ours: &[(String, String)], body: Sx) -> Sx {
    let mut body = body;
    for (j, (fr, to)) in ours.iter().rev().enumerate() {
        body = if j == 0 { f("after", vec![a(fr), a(to), body]) } else { f("after", vec![a(fr), a(to), f("least", vec![body])]) };
    }
    body
}

fn alive_need(family: &str) -> Option<i64> {
    Some(match family { "delayed_root" => 4, "delayed_root_max" => 13, "delayed_balance" => 4, "delayed_balance_deep" => 4, _ => return None })
}

fn tail_guard(mover: &str, hit: bool) -> Sx {
    let ours_hit = f("count", vec![f("and", vec![a("ours"), a("hit")])]);
    let promo = f("count", vec![f("and", vec![a("theirpawns"), f("rank", vec![a(&view("a2", mover))])])]);
    let ep = f("count", vec![f("and", vec![a("theirpawns"), f("rank", vec![a(&view("a4", mover))])])]);
    let no_reply = f("eq", vec![f("least", vec![a("1")]), a("0")]);
    if hit {
        f("eq", vec![f("sum", vec![f("sum", vec![f("sum", vec![ours_hit, promo]), ep]), no_reply]), a("0")])
    } else {
        f("eq", vec![f("sum", vec![f("sum", vec![promo, ep]), no_reply]), a("0")])
    }
}

fn tail_body(mover: &str, family: &str, c: &str) -> Sx {
    let alive = f("sum", vec![a("material_us"), a(c)]);
    let bal = |inner: Sx| inner;
    match family {
        "delayed_balance" => f("least", vec![f("most", vec![f("least", vec![f("min", vec![a("balance"), alive])])])]),
        "delayed_balance_deep" => f("least", vec![f("most", vec![f("least", vec![f("most", vec![f("least", vec![f("min", vec![a("balance"), alive])])])])])]),
        "delayed_root_max" => {
            let max_attacked = f("maxworth", vec![f("and", vec![a("ours"), a("hit")])]);
            f("least", vec![f("most", vec![f("min", vec![f("if", vec![tail_guard(mover, false), f("diff", vec![a("balance"), max_attacked]), a("0")]), alive])])])
        }
        _ => bal(f("least", vec![f("most", vec![f("min", vec![f("if", vec![tail_guard(mover, true), a("balance"), a("0")]), alive])])])),
    }
}
fn delayed_tail(pairs: &[(String, String)], mover: &str, family: &str, c: &str) -> Sx { wrap_tree(pairs, tail_body(mover, family, c)) }

// ------------------------------------------------------------------------------------------------- reading
struct Atom { term: Sx, ge: bool, v: i64 }

struct Reader { pos: Chess, lone: Color, mover: String, memo: HashMap<String, i64> }
impl Reader {
    fn new(fen: &str, mover: &str) -> PyResult<Reader> {
        let pos = parse(fen).ok_or_else(|| PyValueError::new_err(format!("verify_route: cannot read the fen {fen}")))?;
        let lone = !pos.turn();
        Ok(Reader { pos, lone, mover: mover.to_string(), memo: HashMap::new() })
    }
    fn val(&mut self, t: &Sx) -> PyResult<i64> {
        let s = text(t, Some(&self.mover));
        if let Some(v) = self.memo.get(&s) { return Ok(*v); }
        let term = parse_term(&s).ok_or_else(|| PyValueError::new_err(format!("the evaluator does not read {s}")))?;
        let v = if is_num(&term) { num(&term, &self.pos, self.lone) as i64 } else { set(&term, &self.pos, self.lone).count() as i64 };
        self.memo.insert(s, v);
        Ok(v)
    }
    fn holds(&mut self, t: &Sx, ge: bool, v: i64) -> PyResult<bool> { let x = self.val(t)?; Ok(if ge { x >= v } else { x <= v }) }
    /// The legality legs of the terms (unique, sorted by their text as with_legality does), each as (term, ge, v).
    fn legs(&self, terms: &[&Sx]) -> Vec<(Sx, bool, i64)> {
        let mut by: BTreeMap<String, Sx> = BTreeMap::new();
        for t in terms {
            let mut out = vec![];
            legs_of(t, &mut vec![], &self.mover, &mut out);
            for g in out { by.entry(text(&g, None)).or_insert(g); }
        }
        by.into_values().map(|g| if is_no_pawn(&g) { (g, false, 0) } else { (g, true, 1) }).collect()
    }
    fn legs_hold(&mut self, terms: &[&Sx]) -> PyResult<bool> {
        for (g, ge, v) in self.legs(terms) { if !self.holds(&g, ge, v)? { return Ok(false); } }
        Ok(true)
    }
    /// selfmine.premise_truth on this position: the atoms and their legs all hold.
    fn premise_truth(&mut self, atoms: &[Atom]) -> PyResult<bool> {
        let terms: Vec<&Sx> = atoms.iter().map(|x| &x.term).collect();
        if !self.legs_hold(&terms)? { return Ok(false); }
        for x in atoms { if !self.holds(&x.term, x.ge, x.v)? { return Ok(false); } }
        Ok(true)
    }
}

fn san_token(pairs: &[(String, String)]) -> String {
    format!("\u{1}{}\u{2}", pairs.iter().map(|(x, y)| format!("{x}{y}")).collect::<Vec<_>>().join(","))
}

/// puzzle_apply.best_bind on `cands` (each `depth` pairs): -> (the best candidate's index, B, T, the printed lines) -- best is the first with the largest T.
fn best_bind(rd: &mut Reader, depth: usize, family: &str, cands: &[Vec<(String, String)>]) -> PyResult<(Option<(usize, i64, i64)>, Vec<String>)> {
    let b = rd.val(&a("balance"))?;
    let need = alive_need(family).ok_or_else(|| PyValueError::new_err(format!("unknown family {family}")))?;
    let mut best: Option<(usize, i64, i64)> = None;
    let mut notes = vec![];
    for (i, pairs) in cands.iter().enumerate() {
        if pairs.len() != depth { continue; }
        let tail = delayed_tail(pairs, &rd.mover.clone(), family, "1000");
        if !rd.legs_hold(&[&tail])? { continue; }
        let t = rd.val(&tail)?;
        if t > b {
            if t < need {
                notes.push(format!("bind {family}_d{depth}: {} reads {t}, below the alive material {need}: the schema cannot apply", san_token(pairs)));
                continue;
            }
            let alive_tail = delayed_tail(pairs, &rd.mover.clone(), family, &(t - need).to_string());
            let alive = [Atom { term: a("balance"), ge: false, v: b }, Atom { term: alive_tail, ge: true, v: t }];
            if !rd.premise_truth(&alive)? {
                notes.push(format!("bind {family}_d{depth}: {} reads {t} but our material is below {need} at a leaf: the schema cannot apply", san_token(pairs)));
                continue;
            }
        }
        if best.map_or(true, |bb| t > bb.2) { best = Some((i, b, t)); }
    }
    Ok((best, notes))
}

/// verify_route(fen, x, y, mover, schema_names, second_moves) -> [(name, value or None, note, second move or None)] per schema asked, in order; the reading stops
/// after a mate schema that holds (as binder.verify_tried stops). A depth-two name (`<family>_d2`) reads the route with each of `second_moves`.
#[pyfunction]
pub fn verify_route(fen: &str, x: &str, y: &str, mover: &str, schema_names: Vec<String>, second_moves: Vec<(String, String)>)
    -> PyResult<Vec<(String, Option<i64>, String, Option<(String, String)>)>> {
    let mut rd = Reader::new(fen, mover)?;
    let pair = (x.to_string(), y.to_string());
    let mut out = vec![];
    for name in schema_names {
        let (val, note, sec): (Option<i64>, String, Option<(String, String)>);
        if name == "mate_in_one" || name == "mate_in_two" {
            let prem = if name == "mate_in_one" {
                vec![Atom { term: f("after", vec![a(x), a(y), a("movecount")]), ge: false, v: 0 }, Atom { term: f("after", vec![a(x), a(y), a("check")]), ge: true, v: 1 }]
            } else {
                vec![Atom { term: f("after", vec![a(x), a(y), f("least", vec![f("most", vec![f("min", vec![a("check"), f("eq", vec![a("movecount"), a("0")])])])])]), ge: true, v: 1 }]
            };
            let ok = rd.premise_truth(&prem)?;
            val = if ok { Some(100) } else { None };
            note = String::new();
            sec = None;
            let hit = ok;
            out.push((name, val, note, sec));
            if hit { break; }
            continue;
        } else if let Some(fam) = name.strip_suffix("_d1") {
            let (best, notes) = best_bind(&mut rd, 1, fam, &[vec![pair.clone()]])?;
            match best {
                None => { val = None; note = if notes.is_empty() { "the legs or the tail do not hold".to_string() } else { notes.join(" ") }; }
                Some((_, b, t)) if t > b => { val = Some(t - b); note = String::new(); }
                Some((_, b, t)) => { val = None; note = format!("reads {t} against {b} now: no gain"); }
            }
            sec = None;
        } else if let Some(fam) = name.strip_suffix("_d2") {
            if second_moves.is_empty() {
                out.push((name, None, "not a check (or a trap), or no capture of ours is legal after every reply".to_string(), None));
                continue;
            }
            let cands: Vec<Vec<(String, String)>> = second_moves.iter().map(|s| vec![pair.clone(), s.clone()]).collect();
            let (best, notes) = best_bind(&mut rd, 2, fam, &cands)?;
            match best {
                None => {
                    val = None; sec = None;
                    note = if notes.is_empty() { format!("the legs do not hold for any of {} capture(s) after {}{}", second_moves.len(), x, y) } else { notes.join(" ") };
                }
                Some((i, b, t)) if t > b => { val = Some(t - b); note = String::new(); sec = Some(second_moves[i].clone()); }
                Some((i, b, t)) => { val = None; note = format!("reads {t} against {b} now: no gain"); sec = Some(second_moves[i].clone()); }
            }
        } else {
            return Err(PyValueError::new_err(format!("verify_route: unknown schema {name}")));
        }
        out.push((name, val, note, sec));
    }
    Ok(out)
}
