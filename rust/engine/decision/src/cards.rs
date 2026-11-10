//! PLAN 78.14 THE CARD BUILDER: a position into a card -- its facts (the fact layer, in the mover's perspective) and, for every legal move, the move's effect on the
//! facts and its exact value by a search of stated depth, with the units each side loses along its best line. The search is the plain minimax of the judge (alpha-beta
//! with the capture run at the horizon, ordered by the table; NO null-move pruning or reductions, which are not exact): a card's values are exact at its depth, or
//! the card says its reading was cut. Game-free.
use cs_game::gdl_game::FactView;
use cs_game::Game;
use cs_memory::cards::{Card, CardMove};
use cs_memory::op::Effects;
use cs_memory::perspective;
use cs_tactics::ledger::{Judge, Search};

/// The card of the position `s`: None where the game is over or gives no facts. `fuse` caps the nodes of the whole reading.
pub fn card_of<G: Game, J: Judge>(g: &G, view: &FactView, judge: &J, s: &G::State, depth: i32, fuse: u64, id: &str, group: &str) -> Result<Option<Card>, String> {
    let Some(me) = g.mover(s) else { return Ok(None) };
    let base = g.base_facts(s);
    if base.is_empty() { return Ok(None); }
    let roles = g.roles();
    let facts = perspective(&view.facts(&base)?, &roles, me);
    let moves = g.moves(s);
    let effects = Effects::at(view, &view.vocabulary().delta, &base)?;
    if moves.is_empty() { return Ok(None); }
    let mut se = Search::with_fuse(g, judge, me, fuse);
    se.prune = false;
    se.extend = false;
    let vals = se.root_values(s, depth);
    let cut = se.cut() || vals.len() != moves.len();
    let mut cm: Vec<Option<CardMove>> = (0..moves.len()).map(|_| None).collect();
    for (i, v) in &vals {
        let (_line, ev) = se.line_from(s, *i, depth as usize + 6);
        let (lost, gained) = ev.iter().fold((0u32, 0u32), |(l, w), (o, e)| if !e.starts_with("lose ") { (l, w) } else if *o == me { (l + 1, w) } else { (l, w + 1) });
        let (r, a) = effects.of(&g.base_facts(&g.apply(s, &moves[*i])))?;
        // where the loss happens: the effect of the first move of the best line that takes one of ours
        let (mut lr, mut la) = (vec![], vec![]);
        let mut cur = s.clone();
        for name in &_line {
            let Some(m) = g.find_move(&cur, name) else { break };
            let next = g.apply(&cur, &m);
            if g.captured(&cur, &m).iter().any(|(o, _)| *o == me) {
                let (x, y) = Effects::at(view, &view.vocabulary().delta, &g.base_facts(&cur))?.of(&g.base_facts(&next))?;
                lr = perspective(&x, &roles, me);
                la = perspective(&y, &roles, me);
                break;
            }
            cur = next;
        }
        cm[*i] = Some(CardMove { name: g.move_name(s, &moves[*i]), removed: perspective(&r, &roles, me), added: perspective(&a, &roles, me), value: *v, lost, gained,
                                 loss_removed: lr, loss_added: la });
    }
    let moves: Vec<CardMove> = cm.into_iter().flatten().collect();
    if moves.is_empty() { return Ok(None); }
    Ok(Some(Card { id: id.to_string(), group: group.to_string(), facts, moves, depth, cut }))
}
