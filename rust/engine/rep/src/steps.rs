//! PLAN 79.21 STAGE B THE BOARD, READ ONCE FROM THE GAME'S OWN DESCRIPTION (SYSTEM_SPEC 11 "one graph is the representation", reading (a): the board differs by game, so layer 1 reads the board's part from the game's description and hard-codes none; piece: representation).
//! Moved down from cs_game's `placed` (which re-exports it): reading the STATIC relations of a description (which place is next to which, by direction) needs only the description's parser (cs_gdl), no rule of play: no state is read, no move generated.
//! The bound names come from outside (the game's graph vocabulary): `static_relations` (the relations of the description that do not read the state) and `objects` (a name and the component sorts that make a place).
use crate::names::Place;
use cs_gdl::{Program, SortMember, T, FREE, Row};

/// The rules' step relation as numbers: the directions (the description's own words, sorted by text) and every row (direction, from place, to place), sorted.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Steps { pub directions: Vec<String>, pub rows: Vec<(u16, Place, Place)>, pub relation: String, pub object: String }

fn render(p: &Program, t: T) -> String {
    match p.terms.compound(t) {
        Some((f, args)) => format!("({} {})", p.terms.name(f), args.iter().map(|a| render(p, *a)).collect::<Vec<_>>().join(" ")),
        None => p.terms.name(t).to_string(),
    }
}

fn args_of(row: &Row) -> Vec<T> { row.iter().copied().take_while(|t| *t != FREE).collect() }

/// Read the places and the steps by direction from a game description, with no name of any game written here. A place is the product of the component sorts of an object; the STEP RELATION is the one static relation whose signature is a direction sort followed by two
/// places (the object's components twice); it must be exactly one. The description's places are matched to `place_names` by the components joined, and the two sets must be equal, else an error: the engine's numbering and the description's board are the same board or this refuses.
pub fn read_steps(description: &str, static_relations: &[String], objects: &[(String, Vec<String>)], place_names: &[String]) -> Result<Steps, String> {
    let prog = Program::load(description).map_err(|e| e.0)?;
    let mut found: Vec<(String, String, Vec<String>)> = vec![];
    for r in static_relations {
        let Some(sig) = prog.signature(r) else { continue };
        for (name, comps) in objects {
            let n = comps.len();
            if sig.len() == 1 + 2 * n && sig[1..1 + n] == comps[..] && sig[1 + n..] == comps[..] && !comps.contains(&sig[0]) { found.push((r.clone(), name.clone(), comps.clone())); }
        }
    }
    let (relation, object, comps) = match found.len() { 1 => found.remove(0), 0 => return Err("no static relation is a step relation (a direction sort, then two places)".into()),
        n => return Err(format!("{n} static relations look like a step relation: {:?}", found.iter().map(|f| f.0.clone()).collect::<Vec<_>>())) };
    let members = |sort: &String| -> Result<Vec<String>, String> {
        Ok(prog.sorts.get(sort).ok_or_else(|| format!("the description has no sort {sort}"))?.iter().filter_map(|m| if let SortMember::Const(c) = m { Some(c.clone()) } else { None }).collect())
    };
    let mut product: Vec<String> = vec![String::new()];
    for c in &comps { let ms = members(c)?; product = product.iter().flat_map(|p| ms.iter().map(move |m| format!("{p}{m}"))).collect(); }
    let (mut a, mut b) = (product.clone(), place_names.to_vec());
    a.sort(); b.sort();
    if a != b { return Err(format!("the description's places ({}) and the game's places ({}) are not the same set", a.len(), b.len())); }
    let index: std::collections::HashMap<&str, Place> = place_names.iter().enumerate().map(|(i, s)| (s.as_str(), i as Place)).collect();
    let rid = prog.rel_id(&relation).ok_or("the step relation is not in the description")?;
    let rel = prog.statics.get(&rid).ok_or_else(|| format!("{relation} is not a static relation"))?;
    let n = comps.len();
    let mut raw: Vec<(String, Place, Place)> = vec![];
    for row in rel.order.iter() {
        let args: Vec<String> = args_of(row).iter().map(|t| render(&prog, *t)).collect();
        if args.len() != 1 + 2 * n { return Err("a step row has the wrong width".into()); }
        let (f, t) = (args[1..1 + n].concat(), args[1 + n..].concat());
        raw.push((args[0].clone(), *index.get(f.as_str()).ok_or("a step starts on no place")?, *index.get(t.as_str()).ok_or("a step ends on no place")?));
    }
    let mut directions: Vec<String> = raw.iter().map(|r| r.0.clone()).collect();
    directions.sort(); directions.dedup();
    let mut rows: Vec<(u16, Place, Place)> = raw.into_iter().map(|(d, f, t)| (directions.iter().position(|x| *x == d).unwrap() as u16, f, t)).collect();
    rows.sort(); rows.dedup();
    Ok(Steps { directions, rows, relation, object })
}
