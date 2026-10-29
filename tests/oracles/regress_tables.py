"""plan 76.4a (c) THE HAND TABLES AS TEST ORACLES: the bodies regress.py read its defects and fixes from until 76.4a (the audit's families D1a, D1b -- the mate and the
material defect kinds -- and the fix filter `_still` with its positive condition `_goal_done`), moved here verbatim. The engine reads discovery/regression.py now
(the spec's clause bodies, named by the store's DEFECT_FIX entries); the tests compare the regression against these (tests/test_plan76_4a.py, tests/test_regress_rust.py)
and the Rust twins of plan 54 (analyst_core's mate_defects_pos, material_defects_pos, still_routes_pos -- still in the crate, read by nothing in the engine) against
them. The two ways they differ from the regression, read in 76.4a (b): the cover / guard test misses a square covered along a line through their king (a table
incompleteness), and the material filter counts the capture of the target itself as a fix (it reads the exchange on our own unit there: a bug of the table)."""
from __future__ import annotations

import chess

import analyst_core as _A
from catspace.incubator.analyst.discovery import graph as G
from catspace.incubator.analyst.discovery import regress as R


def mate_defects_on(cf: chess.Board, me: bool) -> list:
    """THE HAND MATE TABLE (D1a): the defects of an already-built future `cf` (their move): [] = they are mated."""
    ksq = cf.king(not me)
    if ksq is None:
        return [R.Defect("no check", None, [], "their king is not on the board")]
    checkers = list(cf.checkers())
    if not checkers:
        return [R.Defect("no check", ksq, [], f"nothing of ours attacks {G.name(ksq)}")]
    out = []
    legal = list(cf.legal_moves)
    for c in checkers:
        for m in legal:
            if m.to_square == c:
                pc = cf.piece_at(m.from_square)
                out.append(R.Defect("checker takeable", c, [m.from_square], f"{pc.symbol().upper()} on {G.name(m.from_square)} can take the checker on {G.name(c)}"))
    for m in legal:
        if m.from_square == ksq and m.to_square not in checkers:
            out.append(R.Defect("flight", m.to_square, [ksq], f"the king can go to {G.name(m.to_square)}"))
    if len(checkers) == 1:
        c = checkers[0]
        between = set(chess.SquareSet(chess.between(c, ksq)))
        by_sq: dict = {}
        for m in legal:
            if m.from_square != ksq and m.to_square in between:
                by_sq.setdefault(m.to_square, []).append(m.from_square)
        for sq, frm in sorted(by_sq.items()):
            out.append(R.Defect("interposition", sq, frm, f"a piece can step between on {G.name(sq)}"))
    return out


def mate_defects(board: chess.Board, us, placement: list) -> list:
    """The hand mate table on the question's counterfactual (the placements made, their move)."""
    me = G.colour_of(us)
    return mate_defects_on(R.future_board(board, placement, not me), me)


def material_defects(board: chess.Board, us, target_sq: int) -> list:
    """THE HAND MATERIAL TABLE (D1b): [] when the exchange on the target already wins for us; 'not attacked'; 'defended'."""
    me = G.colour_of(us)
    if G.see_for(board, target_sq, me) > 0:
        return []
    if not G.attackers_of(board, target_sq, me):
        return [R.Defect("not attacked", target_sq, [], f"nothing of ours attacks {G.name(target_sq)}")]
    d = G.defenders_of(board, target_sq)
    return [R.Defect("defended", target_sq, d, f"attacked, guarded {len(d)}x, the exchange gives nothing")]


def goal_done(q, cf: chess.Board, me: bool) -> bool:
    """52.11 (a): the hand's POSITIVE condition of a fix's goal on the counterfactual after the prep move (cover: the square attacked by us -- with their king on its
    square; guard: the checker's square attacked by another piece of ours; remove: the blocker gone)."""
    p = q.pattern
    if q.goal == "cover":
        return cf.is_attacked_by(me, p["target"])
    if q.goal == "guard":
        return any(a != p.get("frm") and a != p["target"] for a in cf.attackers(me, p["target"]))
    if q.goal == "remove" and p.get("blocker") is not None:
        pc = cf.piece_at(p["blocker"])
        return pc is None or pc.color != me
    return True


def defects_before(q, board: chess.Board, me: bool, root) -> set:
    return {d.key()[:2] for d in mate_defects(board, me, R.placement_of(root))} - {q.defect_key[:2]}


def still(q, board: chess.Board, us, m: chess.Move, root, before: "set | None" = None) -> bool:
    """THE HAND FIX FILTER (`_still` until 76.4a): is the fix `q` NOT done by m?"""
    c = board.copy(stack=False)
    c.push(m)
    me = G.colour_of(us)
    if root.goal == "mate":
        pl = R.placement_of(root)
        after = {d.key()[:2] for d in mate_defects(c, me, pl)}
        if before is None:
            before = defects_before(q, board, me, root)
        if q.defect_key[:2] in after or any(k[0] == "no check" for k in after) or not after <= before:
            return True
        return not goal_done(q, R.future_board(c, pl, not me), me)
    if root.goal == "win":
        if c.is_check():
            return not G.attackers_of(c, root.pattern["target"], me)
        return bool(material_defects(G.with_turn(c, me), me, root.pattern["target"]))
    return True


def rust_mate_defects(board: chess.Board, us, placement: list) -> list:
    """The plan 54 Rust twin of the hand mate table (analyst_core.mate_defects_pos), as the engine called it; ValueError where shakmaty holds no such position."""
    me = G.colour_of(us)
    pl = [(p.symbol(), at, frm) for p, at, frm in placement]
    h = G.pos_of(board)
    return [R.Defect(k, sq, list(by), why) for k, sq, by, why in (_A.mate_defects_pos(h, me, pl) if h is not None else _A.mate_defects(board.fen(), me, pl))]


def rust_material_defects(board: chess.Board, us, target_sq: int) -> list:
    """The Rust twin of the hand material table (analyst_core.material_defects_pos)."""
    return [R.Defect(k, sq, list(by), why) for k, sq, by, why in _A.material_defects_pos(G.handle(board), G.colour_of(us), target_sq)]
