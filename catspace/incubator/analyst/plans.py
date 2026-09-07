"""plans.py -- the logical planner (Kaveh 2026-09-06): produce a suggestion by a CHAIN OF REASONING over the concept
objects, not by a value. Backward chaining over typed goals with the operators as rewrite rules:

    win(t)     the ledger says t falls           -> capture it                                   (leaf)
               the ledger says t is held         -> flip conditions:
                   add an attacker worth <= v     -> bring(p, t) for every own piece that can reach   (reach)
                   remove the defender d          -> win(d), then win(t)                            (recursion)
    defend(t)  (the opponent's side of the same coin) escape t to a safe square | add a defender by reach |
               win(the attacker)

Every rewrite is one rule over one object; every leaf is a move; the chain from the goal to the move is the
explanation. Branching in this space is two or three per step against thirty-odd legal moves per ply -- the
"search in a different domain" -- and each plan is then checked twice: the opponent's COUNTER-GOALS on the board
after our first hop (can they defend, escape or refute in fewer moves than we need: the race), and the engine
(reach.verify_plan). Nothing here reads the objective; everything reads the ledgers, the flips and reach, which
read the rules. Lineage: means-ends analysis (GPS, STRIPS, Wilkins' PARADISE), with a verifier."""
from __future__ import annotations

from dataclasses import dataclass, field

import chess

from . import rules as R
from .defense import defense
from .facts import value
from .reach import reach, flip_conditions, Path, verify_plan, Target, ledger_phrase, counts


@dataclass(frozen=True)
class Goal:
    kind: str                       # "win" | "defend" | "bring" | "remove" | "escape"
    square: int
    extra: float = 0.0              # add-attacker value cap, etc.

    def text(self, board: chess.Board) -> str:
        p = board.piece_at(self.square)
        name = f"{chess.piece_name(p.piece_type) if p else 'piece'} on {chess.square_name(self.square)}"
        return {"win": f"win the {name}", "defend": f"hold the {name}", "remove": f"remove the {name}",
                "escape": f"save the {name}"}.get(self.kind, f"{self.kind} {name}")


@dataclass
class Step:
    move: chess.Move
    san: str
    why: str


@dataclass
class Plan:
    side: chess.Color
    goal: Goal
    steps: list[Step]                          # our moves, in order (the opponent's replies not included)
    chain: list[str]                           # the derivation, one line per rewrite
    answers: list["Plan"] = field(default_factory=list)   # the opponent's counter-plans after our first step
    race: str = "unverified"                   # "wins" | "met" | "unverified"
    verified: object = None                    # reach.Verified from the engine, when asked
    cost: float = 0.0                          # opportunity cost of the first move: what the moving piece held elsewhere (subgames.holdings)
    breaks: list[str] = field(default_factory=list)   # those holdings in words

    @property
    def hops(self) -> int:
        return len(self.steps)


# ---------------------------------------------------------------- derivation
def _cheapest_capture(board: chess.Board, target: int, by: chess.Color) -> chess.Move | None:
    b = board.copy(stack=False); b.turn = by
    for a in sorted(b.attackers(by, target), key=lambda q: value(board, q)):
        for m in (chess.Move(a, target), chess.Move(a, target, promotion=chess.QUEEN)):
            if m in b.legal_moves:
                return m
    return None


def _path_steps(board: chess.Board, side: chess.Color, path: Path, why: str) -> list[Step] | None:
    """Turn a reach path into moves on a scratch board (our moves only)."""
    b = board.copy(stack=False); b.turn = side
    cur = path.piece; steps = []
    for to in path.squares:
        mv = next((m for m in b.legal_moves if m.from_square == cur and m.to_square == to), None)
        if mv is None:
            return None
        steps.append(Step(mv, b.san(mv), why)); b.push(mv); b.turn = side; cur = to
    return steps


def derive(board: chess.Board, side: chess.Color, goal: Goal, memo=None, depth: int = 2, max_hops: int = 3) -> list[Plan]:
    """Plans for `side` that achieve `goal`, cheapest first. depth bounds the recursion through removals."""
    plans: list[Plan] = []
    b = board.copy(stack=False); b.turn = side
    if goal.kind == "win":
        t = goal.square; tp = board.piece_at(t)
        if tp is None or tp.color == side:
            return []
        led = defense(board, t, memo=memo)
        name = goal.text(board)
        if led is not None and led.falls:
            cap = _cheapest_capture(board, t, side)
            if cap:
                b2 = board.copy(stack=False); b2.turn = side
                plans.append(Plan(side, goal, [Step(cap, b2.san(cap), "it falls now")],
                                  [f"{name}: it falls now ({ledger_phrase(board, led)})"]))
            return plans
        ledger = ledger_phrase(board, led)
        for f in flip_conditions(board, t, side, memo=memo):
            if f.kind == "add_attacker":
                attacking_now = set(led.attackers) if led else set()
                for psq, pp in board.piece_map().items():
                    if pp.color != side or R.is_royal(board, psq) or psq in attacking_now or value(board, psq) > f.max_value:
                        continue
                    path = reach(board, psq, t, max_hops, memo=memo)
                    if path is None or path.hops == 0:
                        continue
                    steps = _path_steps(board, side, path, f"brings the {chess.piece_name(pp.piece_type)} to bear on {chess.square_name(t)}")
                    if steps is None:
                        continue
                    chain = [f"{name}: it is held ({ledger})",
                             f"one more attacker worth {f.max_value:g} or less wins it",
                             f"the {chess.piece_name(pp.piece_type)} on {chess.square_name(psq)} reaches {chess.square_name(path.squares[-1])} in {path.hops} ({' '.join(s.san for s in steps)})"]
                    plans.append(Plan(side, goal, steps, chain))
            elif f.kind == "remove_defender" and depth > 0:
                for sub in derive(board, side, Goal("win", f.square), memo, depth - 1, max_hops):
                    chain = [f"{name}: it is held ({ledger})", f"its defender, the {chess.piece_name(board.piece_at(f.square).piece_type)} on {chess.square_name(f.square)}, falls first"] + ["  " + c for c in sub.chain]
                    plans.append(Plan(side, goal, sub.steps + [Step(chess.Move.null(), f"then x{chess.square_name(t)}", "then it falls")], chain))
    elif goal.kind == "defend":
        t = goal.square; tp = board.piece_at(t)
        if tp is None or tp.color != side:
            return []
        led = defense(board, t, memo=memo)
        if led is None or not led.falls:
            return []                                                  # nothing to do: it holds
        name = goal.text(board)
        # 1. escape to a square where it does not fall
        for m in b.legal_moves:
            if m.from_square != t:
                continue
            nb = b.copy(stack=False); nb.push(m)
            d2 = defense(nb, m.to_square, memo=memo)
            if d2 is None or not d2.falls:
                plans.append(Plan(side, goal, [Step(m, b.san(m), "moves it to safety")], [f"{name}: it falls ({counts(led)})", f"it steps to {chess.square_name(m.to_square)}, where it holds"]))
        # 2. add a defender by reach (a piece that comes to attack its own square = defends it)
        for psq, pp in board.piece_map().items():
            if pp.color != side or psq == t or R.is_royal(board, psq) or psq in led.defenders:
                continue
            path = reach(board, psq, t, max_hops, memo=memo)
            if path is None or path.hops == 0:
                continue
            nb = board.copy(stack=False); nb.turn = side
            steps = _path_steps(board, side, path, f"adds a defender to {chess.square_name(t)}")
            if steps is None:
                continue
            # does the added defender actually hold it? check on the board after the path
            for s in steps:
                nb.push(s.move); nb.turn = side
            d2 = defense(nb, t, memo=memo)
            if d2 is not None and d2.falls:
                continue
            plans.append(Plan(side, goal, steps, [f"{name}: it falls ({counts(led)})", f"the {chess.piece_name(pp.piece_type)} on {chess.square_name(psq)} comes to defend it in {path.hops} ({' '.join(s.san for s in steps)})"]))
        # 3. win the attacker
        if depth > 0 and led.effective_attackers:
            for a in led.effective_attackers:                 # an attacker that cannot take is no threat to remove
                for sub in derive(board, side, Goal("win", a), memo, depth - 1, max_hops):
                    plans.append(Plan(side, goal, sub.steps, [f"{name}: it falls", f"the attacker on {chess.square_name(a)} is removed"] + ["  " + c for c in sub.chain]))
    plans.sort(key=lambda p: (p.hops, sum(value(board, s.move.from_square) for s in p.steps if s.move != chess.Move.null())))
    return plans


# ---------------------------------------------------------------- counter-goals and the race
def answer(board: chess.Board, plan: Plan, memo=None) -> Plan:
    """The opponent's best counter to `plan`, judged on the board after our FIRST step: can they hold or save the
    target, or remove our joiner, in fewer moves than we still need? Sets plan.answers and plan.race."""
    if not plan.steps or plan.steps[0].move == chess.Move.null():
        plan.race = "unverified"; return plan
    b = board.copy(stack=False); b.turn = plan.side; b.push(plan.steps[0].move)
    them = not plan.side
    remaining = max(1, plan.hops - 1) + (1 if plan.goal.kind == "win" else 0)     # our hops left, plus the capture itself
    target = plan.goal.square
    cands: list[Plan] = []
    if plan.goal.kind == "win" and b.piece_at(target) is not None and b.piece_at(target).color == them:
        cands += derive(b, them, Goal("defend", target), memo, depth=1)
        # or remove the piece we are bringing (our joiner's current square)
        joiner_sq = plan.steps[0].move.to_square
        cands += derive(b, them, Goal("win", joiner_sq), memo, depth=1)
    cands.sort(key=lambda p: p.hops)
    plan.answers = cands[:3]
    fastest = cands[0].hops if cands else None
    if fastest is None:
        plan.race = "wins"
    elif fastest < remaining:
        plan.race = "met"
    else:
        plan.race = "wins"
    return plan


def plans_for(board: chess.Board, side: chess.Color, memo=None, top: int = 5, engine=None, depth: int | None = None) -> list[Plan]:
    """First the DEFENSIVE plans for our own pieces that fall now (a plan that ignores a hanging queen is no plan),
    then the winning plans against every enemy piece, answered by counter-goals, cheapest and most valuable first;
    with an engine, multi-step plans are also verified by reach.verify_plan (null-move hops, then the engine's reply)."""
    urgent = []
    for sq, p in sorted(board.piece_map().items(), key=lambda kv: -value(board, kv[0])):
        if p.color != side or R.is_royal(board, sq):
            continue
        led = defense(board, sq, memo=memo)
        if led is not None and led.falls and value(board, sq) >= 3:
            for pl in derive(board, side, Goal("defend", sq), memo)[:2]:
                pl.race = "urgent"; urgent.append(pl)
    out = []
    for sq, p in sorted(board.piece_map().items(), key=lambda kv: -value(board, kv[0])):
        if p.color == side or R.is_royal(board, sq):
            continue
        for pl in derive(board, side, Goal("win", sq), memo)[:2]:
            answer(board, pl, memo)
            out.append(pl)
    # the opportunity cost of each plan's first move: the lemmas the moving piece holds elsewhere (subgames)
    from .subgames import holdings, cost_of_move, holding_text
    held = holdings(board, memo)
    for pl in urgent + out:
        if pl.steps and pl.steps[0].move != chess.Move.null():
            broken = cost_of_move(board, pl.steps[0].move, held, memo)
            pl.cost = sum(h.loss for h in broken)
            pl.breaks = [f"the {chess.piece_name(board.piece_at(h.piece).piece_type)} on {chess.square_name(h.piece)} {holding_text(board, h)}" for h in broken]
    out.sort(key=lambda p: (p.cost >= 100, 0 if p.race == "wins" else 1, -(value(board, p.goal.square) - p.cost), p.hops))   # not playable last
    out = urgent[:2] + out[:top]
    if engine is not None:
        for pl in out:
            if pl.goal.kind == "win" and pl.hops >= 2 and pl.steps and pl.steps[0].move != chess.Move.null():
                t = Target(pl.goal.square, board.piece_at(pl.goal.square).symbol(), "joinable", None, joiner=Path(pl.steps[0].move.from_square, tuple(s.move.to_square for s in pl.steps if s.move != chess.Move.null())))
                try:
                    pl.verified = verify_plan(board, engine, t, depth=depth)
                except Exception:
                    pl.verified = None
    return out


# ---------------------------------------------------------------- prose
def plan_text(board: chess.Board, plan: Plan) -> str:
    moves = " ".join(s.san for s in plan.steps)
    lead = "First, " if plan.race == "urgent" else ""
    head = f"{lead}{plan.goal.text(board)} in {plan.hops}: {moves}"
    chain = "; ".join(c.strip() for c in plan.chain[1:]) if len(plan.chain) > 1 else (plan.chain[0].split(": ", 1)[-1] if plan.chain else "")
    ans = ""
    if plan.answers:
        a = plan.answers[0]
        ans = f" Their answer in {a.hops}: {' '.join(s.san for s in a.steps)} ({a.chain[-1].strip()})."
    race = {"wins": " The race is ours.", "met": " They are in time.", "unverified": "", "urgent": ""}[plan.race]
    if plan.breaks:
        race = " But " + "; ".join(plan.breaks) + (": not playable." if plan.cost >= 100 else f": net {value(board, plan.goal.square) - plan.cost:+g}.")
    ver = ""
    if plan.verified is not None:
        v = plan.verified
        ver = " Engine: " + ("works if unopposed" if v.unopposed else "does not work even unopposed") + (f"; their best reply {v.reply}: {v.outcome}" if v.reply else "") + "."
    return (head[0].upper() + head[1:] + (f" -- {chain}." if chain else ".") + ans + race + ver).replace("..", ".")


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--fen", required=True); ap.add_argument("--engine", action="store_true"); ap.add_argument("--side", default="")
    a = ap.parse_args(argv)
    b = chess.Board(a.fen); side = b.turn if not a.side else (a.side.lower().startswith("w"))
    eng = None
    if a.engine:
        from .engine import Engine
        eng = Engine(depth=14)
    for pl in plans_for(b, side, engine=eng, depth=14):
        print(plan_text(b, pl))
    if eng:
        eng.close()


if __name__ == "__main__":
    main()
