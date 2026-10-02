"""THE WRITTEN EXPLANATION AND ITS CHECKER (Kaveh 2026-10-02: name each instance; "more facts first"; then no
preference between automating and more -- the recommended path: an LLM writes from the facts, a checker verifies).

- `render(facts)`: the fact sheet (explain.facts) as compact text -- everything the writer may use, nothing else.
- `PROMPT`: the fixed instruction for the writer (Claude via the API, or Claude in session): three parts -- what the
  human saw, why it fails, what the better move does -- written only from the facts.
- `check(text, facts)`: FAITHFULNESS -- every chess move (SAN), square and number the text mentions is looked up: a
  move must be in one of the fact sheet's lines (or be the human or better move); a square must be on the board or in a
  line; a percentage must be a Maia probability in the facts (rounded); an evaluation ("+6.8", "-2.1") must be one of
  the facts' evaluations in pawns (rounded). What is not found is reported -- an unverified claim, not necessarily a
  wrong one (a checker false alarm is a finding too).

    python -m catspace.incubator.analyst.explain_text check-samples      # the hand-written v2 samples
"""
from __future__ import annotations

import argparse
import json
import re

import chess

from catspace.io import paths

PROMPT = """You explain a chess mistake to a club player, as a coach. You get a FACT SHEET computed by Stockfish,
python-chess and a human-move model (Maia). Write three short paragraphs, in this order:
1. What the human saw: the move the player chose, how natural it was (Maia's probability), and what it grabs,
   attacks, defends against or sets up -- from the facts.
2. Why it fails: the opponent's answer, move by move from the facts' line, and the idea behind it in plain words
   (use the tactic family's name only if it fits the line).
3. What to play instead: the better move, what it threatens or prevents, and its line from the facts.
Rules: use ONLY moves, squares, pieces and numbers that appear in the fact sheet. Do not invent variations. Write
moves in standard algebraic notation exactly as given. Evaluations are in pawns for the side to move. At most 150
words."""

SAN_RX = re.compile(r"\b(O-O-O|O-O|[KQRBN][a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?|[a-h]x[a-h][1-8](?:=[QRBN])?[+#]?|[a-h][1-8](?:=[QRBN])?[+#]?)")
SQ_RX = re.compile(r"\b(?:on|to|from|at|the) ([a-h][1-8])\b")
PCT_RX = re.compile(r"(\d+(?:\.\d+)?)%")
EVAL_RX = re.compile(r"([+−-]\d+\.\d)")


def _lines(f: dict) -> list:
    out = [f.get("opponent_line") or []]
    ht = f.get("human_thinking") or {}
    for t in (ht.get("threat"), f.get("opponent_threat_before")):
        if t:
            out.append(t["line"])
    bt = f.get("better") or {}
    out.append(bt.get("line") or [])
    a = bt.get("achieves") or {}
    for t in (a.get("threat"), a.get("opponent_threat_after")):
        if t:
            out.append(t["line"])
    return out


def _ev(cp: float) -> str:
    """An evaluation in words for the writer: pawns, or mate (Stockfish's mate scores are +-10000)."""
    if abs(cp) >= 5000:
        return "mates" if cp > 0 else "gets mated"
    return f"{cp / 100:+.1f}"


def render(f: dict) -> str:
    """The fact sheet as compact text for the writer."""
    fl = lambda line: " ".join(m["move"] + (" {" + "; ".join(m["facts"]) + "}" if m["facts"] else "") for m in (line or [])[:8])
    th = lambda t: "none" if not t else f"{fl(t['line'])} (material {t['wins_material']:+d}{', mate' if t['mates'] else ''}; evaluation {_ev(t['eval'])})"
    ht = f.get("human_thinking") or {}; bt = f.get("better") or {}; a = bt.get("achieves") or {}
    mm = ht.get("maia") or {}
    rows = [f"Position (FEN): {f['fen']} -- {f['side']} to move.",
            f"Player's move: {f['human']}. Evaluation for {f['side']}: best {_ev(f['eval_best'])}; after {f['human']} {_ev(f['eval_after_human'])}.",
            f"How natural (Maia-1500): {f['human']} {mm.get('p', 0) * 100:.0f}% (rank {mm.get('rank')}); top moves: "
            + ", ".join(f"{m} {p * 100:.0f}%" for m, p in mm.get("top", [])),
            f"{f['human']} grabs: {ht.get('grabs') or 'nothing'}; attacks: {', '.join(ht.get('attacks') or []) or 'nothing new'}.",
            f"What {f['human']} would threaten next if allowed: {th(ht.get('threat'))}.",
            f"The opponent's threat before the move: {th(f.get('opponent_threat_before'))}.",
            f"The opponent's answer to {f['human']}: {fl(f.get('opponent_line'))}. Tactic family: {f.get('family') or 'none'}."]
    if bt:
        rows += [f"Better move: {bt['move']} (Maia {bt.get('maia', {}).get('p', 0) * 100:.0f}%, rank {bt.get('maia', {}).get('rank')}). Line: {fl(bt.get('line'))}.",
                 f"{bt['move']}'s own threat: {th(a.get('threat'))}. The opponent's best after {bt['move']}: {th(a.get('opponent_threat_after'))}.",
                 f"Frames it advances: {'; '.join(a.get('frames') or []) or 'none'}. Activity: {a.get('activity')}."]
    return "\n".join(rows)


def check(text: str, f: dict) -> list:
    """The claims in `text` the facts do not back: (kind, what)."""
    b = chess.Board(f["fen"])
    text = re.sub(r"`[^`]*`", " ", text)                             # a FEN in backticks is not a claim (2026-10-02 false alarm: 'K3b1')
    moves = {f["human"].rstrip("+#!?"), (f.get("better") or {}).get("move", "").rstrip("+#!?")}
    moves |= {m.rstrip("+#") for m, _ in ((f.get("human_thinking") or {}).get("maia") or {}).get("top", [])}   # Maia's top moves
    squares = {chess.square_name(s) for s in chess.SquareSet(b.occupied)}
    for line in _lines(f):
        for m in line:
            moves.add(m["move"].rstrip("+#"))
            for x in re.findall(r"[a-h][1-8]", m["move"]):
                squares.add(x)
            for fact in m["facts"]:
                squares |= set(re.findall(r"\b[a-h][1-8]\b", fact))
    for fr in ((f.get("better") or {}).get("achieves") or {}).get("frames") or []:
        squares |= set(re.findall(r"\b[a-h][1-8]\b", fr))
    act = (((f.get("better") or {}).get("achieves") or {}).get("activity") or {})
    squares |= set(act.get("reaches_enemy_king_squares") or [])
    pcts = set()
    mm = (f.get("human_thinking") or {}).get("maia") or {}
    for p in [mm.get("p", 0)] + [p for _, p in mm.get("top", [])] + [((f.get("better") or {}).get("maia") or {}).get("p", 0)]:
        pcts |= {round(p * 100), int(p * 100)}
    evals = {f["eval_best"], f["eval_after_human"]}
    for t in [(f.get("human_thinking") or {}).get("threat"), f.get("opponent_threat_before"),
              ((f.get("better") or {}).get("achieves") or {}).get("threat"), ((f.get("better") or {}).get("achieves") or {}).get("opponent_threat_after")]:
        if t:
            evals |= {t["eval"], -t["eval"]}
    evals = {round(e / 100, 1) for e in evals if abs(e) < 5000} | {round(-e / 100, 1) for e in evals if abs(e) < 5000}
    bad = []
    for mv in SAN_RX.findall(text):
        mv_ = mv.rstrip("+#")
        if re.fullmatch(r"[a-h][1-8]", mv_):
            continue                                               # a bare square is checked as a square below
        if mv_ not in moves:
            bad.append(("move", mv))
    for sq in SQ_RX.findall(text):
        if sq not in squares:
            bad.append(("square", sq))
    for pc in PCT_RX.findall(text):
        if round(float(pc)) not in pcts:
            bad.append(("percent", pc + "%"))
    for ev in EVAL_RX.findall(text):
        v = float(ev.replace("−", "-"))
        if not any(abs(v - e) <= 0.1 for e in evals):
            bad.append(("evaluation", ev))
    return bad


MODEL = "claude-opus-5-5"                                           # the current default Claude model (claude-api skill, 2026-10-02)


def write_with_claude(f: dict, client=None) -> str | None:
    """The writer: one fact sheet -> one explanation by Claude (Messages API). Thinking stays on (adaptive is the
    default on this model; effort set explicitly); server-side refusal fallbacks enabled (the recommended default). The
    fixed instruction is a few hundred tokens -- under the minimum cacheable prefix, so no prompt caching. Needs
    Anthropic credentials (ANTHROPIC_API_KEY or an `ant auth login` profile). -> the text, or None on a refusal."""
    import anthropic
    client = client or anthropic.Anthropic()
    resp = client.beta.messages.create(
        model=MODEL, max_tokens=16000, system=PROMPT,
        output_config={"effort": "medium"},
        betas=["server-side-fallback-2026-07-01"], fallbacks="default",
        messages=[{"role": "user", "content": render(f)}])
    if resp.stop_reason == "refusal":
        return None
    return "".join(blk.text for blk in resp.content if blk.type == "text").strip()


def write_batch(n: int, offset: int, dry_run: bool = False, log=print) -> None:
    """Fact sheets for n fresh puzzles (the first n with a named tactic), each written by Claude and checked; the
    texts, their unverified claims and the totals saved to data/derived/explain_written_{n}_{offset}.json."""
    import random
    from .explain import facts, GENERIC, _ENG, _MAIA
    from .frame_puzzles import fresh_rows
    from . import tactic_reasons as T
    rows = fresh_rows(4 * n, 997, offset); random.seed(2); random.shuffle(rows)
    out = []; client = None
    try:
        for row in rows:
            b = chess.Board(row["FEN"]); h = chess.Move.from_uci(row["Moves"].split()[0])
            f = facts(b, h)
            if not f.get("family"):
                continue
            f["pid"] = row["PuzzleId"]
            if dry_run:
                log(f"--- {f['pid']} ---\n{render(f)}\n")
                text = None
            else:
                if client is None:
                    import anthropic
                    client = anthropic.Anthropic()
                text = write_with_claude(f, client)
            bad = check(text, f) if text else None
            out.append(dict(pid=f["pid"], facts=f, text=text, unverified=bad))
            if len(out) >= n:
                break
    finally:
        for e in list(_ENG.values()) + list(_MAIA.values()) + list(T._ENG.values()):
            e.quit()
    json.dump(out, open(paths.derived(f"explain_written_{n}_{offset}.json"), "w"), indent=1)
    if not dry_run:
        texts = [r for r in out if r["text"]]
        mentions = sum(len(SAN_RX.findall(r["text"])) + len(SQ_RX.findall(r["text"])) + len(PCT_RX.findall(r["text"])) + len(EVAL_RX.findall(r["text"])) for r in texts)
        log(f"OBS {len(texts)} of {len(out)} written (refused {len(out) - len(texts)}); {mentions} checkable mentions, unverified "
            f"{sum(len(r['unverified']) for r in texts)}; texts with no unverified claim {sum(1 for r in texts if not r['unverified'])}")


def _samples() -> dict:
    """The hand-written v2 samples (docs/EXPLAIN_SAMPLES.md) by puzzle id."""
    txt = open(paths.repo_root() / "docs" / "EXPLAIN_SAMPLES.md").read() if hasattr(paths, "repo_root") else open("docs/EXPLAIN_SAMPLES.md").read()
    out = {}
    for block in txt.split("\n**")[1:]:
        pid = block.split("**", 1)[0].strip()
        if re.fullmatch(r"[0-9A-Za-z]{5}", pid):
            out[pid] = block.split("---")[0]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check-samples")
    w = sub.add_parser("write"); w.add_argument("--n", type=int, default=50); w.add_argument("--offset", type=int, default=100)
    w.add_argument("--dry-run", action="store_true", help="print what would be sent; call nothing")
    a = ap.parse_args(argv)
    if a.cmd == "write":
        return write_batch(a.n, a.offset, a.dry_run)
    facts = {f["pid"]: f for f in json.load(open(paths.derived("explain_facts_samples_v2.json")))}
    n_claims = 0; n_bad = 0
    for pid, text in _samples().items():
        f = facts.get(pid)
        if f is None:
            continue
        bad = check(text, f)
        t2 = re.sub(r"`[^`]*`", " ", text)
        claims = len(SAN_RX.findall(t2)) + len(SQ_RX.findall(t2)) + len(PCT_RX.findall(t2)) + len(EVAL_RX.findall(t2))
        n_claims += claims; n_bad += len(bad)
        print(f"{pid}: {claims} checkable mentions, unverified {len(bad)}: {bad}")
    print(f"OBS the hand-written samples: {n_claims} checkable mentions (moves, squares, percentages, evaluations), unverified {n_bad}")


if __name__ == "__main__":
    main()
