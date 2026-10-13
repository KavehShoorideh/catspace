"""THE CHALLENGER (Kaveh 2026-10-12: "my idea of bringing a rook into this and interposing is a challenge that I bring up as
a challenger, but you cannot bring it up against yourself ... you need your reasoner to reason both sides"; the design
after two literature rounds -- JOURNAL 2026-10-12 19:55 -- oracle-free, so it carries to domains where the ending
playbook is partial).

  A FRAME is a ranking certificate (its measure falls); a COUNTER-FRAME is its dual, a TRAP: a region R where whatever we
  do, the defender has a reply keeping us in R with our measure no lower (Lean Ideas.Forced.hold_blocks).

  1 MINE    from the frame's REFUTED claims in a round's games (any falsifier made them: the tablebase, our own search, a
            simulator): R = {term >= theta} for a term of the vocabulary, chosen as a PREDICTOR of the refutations and
            kept when, on held-out games (split by game), the claims fail MORE inside R than outside it (a one-sided
            Fisher exact test, Bonferroni over the candidates tested -- 2026-10-12: a fixed "base + 0.10" rose with the
            frame's own failure rate and barred a trap holding on 52/60 where the frame failed 73% everywhere)
  2 CHECK   the local trap check on boards of R (vocab.rs trap_check: one move against every reply, no replay of any
            oracle). Passing it, R is a COUNTER-FRAME (the defender holds us there); failing it, R is a BLIND SPOT (we
            had a move, the frame chose wrong -- 2026-10-12: "the kings close" predicted the refutations at 0.94 against
            0.50 and held on 9/60); nothing kept at all -> a VOCABULARY GAP (logged with its boards: where new
            predicates come from)
  3 ANSWER  a STAGE for R, its measure mined by the pipeline on R's refuted boards and promoted by the verifier's rules:
            for a counter-frame its goal is to LEAVE R; for a blind spot, to play well inside R (the game's own ends:
            held, won, progress) -- each against the material's own frame inside R
  4 ARCHIVE every counter-frame, blind spot and gap kept with its status (survived the local check on N boards / predictor
            only / answered by a stage) -- a never-forget archive

  python -m catspace.incubator.analyst.discovery.challenger --run level2b --round 2
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time

import numpy as np

from catspace.io import paths

ARCHIVE = "counter_frames.json"
FAILED = ("no shrink", "other outcome", "lost ground")


def data(res: list, material: str) -> tuple:
    """(fen, refuted?, game) for every frame move of ours in the material while ahead."""
    out = []
    for g in res:
        for r in g["rows"]:
            if r["material"] == material and r["lead"] > 0 and r["claim"]["kind"] == "frame" and r.get("verdict"):
                out.append((r["fen"], r["verdict"] in FAILED, g["seed"]))
    return out


def data_phase(res: list, material: str) -> list:
    """With NO frame for the material (Kaveh 2026-10-12, "1": let the challenger attack the empty frame) the player's only
    claim is the goal within two moves -- refuted almost always (bygame round 2: 1,711 of 1,727) and so no evidence. The
    empty frame's claim is the PHASE's: we win material in it (their material falls, or mate) -- a drawn or lost phase
    refutes it. -> (fen, refuted?, game) for every position of ours in the material while ahead."""
    out = []
    for g in res:
        rows = g["rows"]
        idx = [i for i, r in enumerate(rows) if r["material"] == material]
        if not idx:
            continue
        last = idx[-1]
        won = (last + 1 < len(rows) and rows[last + 1]["theirs"] < rows[last]["theirs"]) or (last == len(rows) - 1 and g["out"] == "win")
        out += [(rows[i]["fen"], not won, g["seed"]) for i in idx if rows[i]["lead"] > 0]
    return out


def by_game_p(inside, y, games, held) -> float:
    """The phase's claim is one label a GAME (every position of a game shares it): on the held-out games, does a refuted
    (drawn) game spend more of its positions inside R than a game won? -- one-sided Mann-Whitney on the games' shares."""
    from scipy.stats import mannwhitneyu
    share, ref = {}, {}
    for r, yy, g, h in zip(inside, y, games, held):
        if h:
            share.setdefault(g, []).append(bool(r)); ref[g] = bool(yy)
    a = [np.mean(share[g]) for g in share if ref[g]]
    b = [np.mean(share[g]) for g in share if not ref[g]]
    if len(a) < 2 or len(b) < 2:
        return 1.0
    return float(mannwhitneyu(a, b, alternative="greater").pvalue)


def best_threshold(v: np.ndarray, y: np.ndarray) -> tuple:
    """theta maximising (precision - base) * recall for R = {v >= theta} predicting y -> (theta, precision, recall, score)."""
    base = y.mean() if len(y) else 0.0
    best = (None, 0.0, 0.0, -1.0)
    for th in np.unique(v):
        r = v >= th
        if r.sum() < 5:
            continue
        prec, rec = y[r].mean(), (y & r).sum() / max(y.sum(), 1)
        sc = (prec - base) * rec
        if sc > best[3]:
            best = (int(th), float(prec), float(rec), float(sc))
    return best


ALPHA = 0.05                        # the family-wise level over the candidates tested (Bonferroni)


def excess(y: np.ndarray, r: np.ndarray) -> float:
    """One-sided Fisher exact p-value: do the claims (y: refuted?) fail MORE inside R (r) than outside it? -- the bar
    does not rise with the frame's own failure rate."""
    from scipy.stats import fisher_exact
    a, b, c, d = int((y & r).sum()), int((~y & r).sum()), int((y & ~r).sum()), int((~y & ~r).sum())
    if a + b == 0 or c + d == 0:
        return 1.0
    return float(fisher_exact([[a, b], [c, d]], alternative="greater")[1])


def split(cands: list, alpha: float = ALPHA) -> tuple:
    """The candidates kept (held-out excess significant at alpha / their number, >= 10 boards checked) -> (traps, blind
    spots): a trap if the local check holds on >= 0.8 of its refuted boards, else a blind spot."""
    kept = [c for c in cands if c["p"] < alpha / max(len(cands), 1) and c["checked"] >= 10]
    return [c for c in kept if c["trap"] >= 0.8], [c for c in kept if c["trap"] < 0.8]


def pick(traps: list, blinds: list) -> tuple:
    """-> (the trap, the blind spot) answered: the trap holding most often (then the lift); the blind spot predicting
    best among those NOT the trap's own region -- one whose refuted boards are mostly the trap's is the same phenomenon
    on the other side of the 0.8 line (2026-10-12: "their rook far from its king" came out as a trap at 0.92 and as a
    blind spot at 0.77)."""
    trap = max(traps, key=lambda c: (c["trap"], c["lift"]), default=None)
    own = set(trap["boards"]) if trap else set()
    blinds = [c for c in blinds if len(own & set(c["boards"])) <= 0.5 * max(len(set(c["boards"])), 1)]
    return trap, max(blinds, key=lambda c: (c["lift"], c["recall"]), default=None)


TOP = 20                            # the regions tested on the held-out games (Bonferroni over them)


def too_little(y, train, games, phase: bool, top: int = TOP) -> str | None:
    """TOO LITTLE DATA, apart from a vocabulary gap (Kaveh 2026-10-12: "make the gap honest"): even a PERFECT predictor
    -- R exactly the refuted held-out claims (or, by game, exactly the drawn games' positions) -- would not pass the
    held-out test at ALPHA / top, or fewer than 10 refuted held-out boards are left for the local trap check. -> why, or
    None (there is enough to look for a region)."""
    held = ~train
    n_ref = int((y & held).sum())
    if n_ref < 10:
        return f"{n_ref} refuted held-out boards (the trap check needs 10)"
    p_min = by_game_p(y, y, games, held) if phase else excess(y[held], y[held])
    if p_min >= ALPHA / top:
        return f"even a perfect predictor reaches only p {p_min:.1e} on the held-out {'games' if phase else 'claims'} (the bar {ALPHA / top:.1e})"
    return None


def mine(res: list, frames: dict, material: str, top: int = TOP, check_n: int = 60, seed: int = 0) -> dict:
    import catspace.incubator.analyst  # noqa: F401
    import analyst_core as A
    from . import base_vocabulary as V
    t0 = time.time()
    phase = not frames.get(material)                                  # no frame: the phase's claim (data_phase)
    rows = data_phase(res, material) if phase else data(res, material)
    if len(rows) < 50:
        return dict(material=material, status="too little data", n=len(rows))
    fens = [r[0] for r in rows]
    y = np.array([r[1] for r in rows])
    train = np.array([r[2] % 2 == 0 for r in rows])                    # split by game: the even seeds learn, the odd test
    why = too_little(y, train, [r[2] for r in rows], phase, top)
    if why:
        print(f"OBS challenger, {material}: TOO LITTLE DATA -- {why}")
        return dict(material=material, status="too little data", n=len(rows), why=why)
    terms = [V.sexpr(t) for t in V.enumerate_numbers()]
    seen, cands = set(), []
    for t in terms:
        v = np.array(A.vocab_values(fens, t))
        key = tuple(v[train][:200])
        if v.std() == 0 or key in seen:
            continue
        seen.add(key)
        th, prec, rec, sc = best_threshold(v[train], y[train])
        if th is not None:
            cands.append((sc, t, th, v))
    cands.sort(key=lambda c: -c[0])
    base_rate = float(y[~train].mean())
    what = (f"no frame: {len(rows)} positions of {len({r[2] for r in rows})} games, the phase's claim 'we win material here' refuted in "
            f"{len({r[2] for r in rows if r[1]})} games" if phase else f"{len(rows)} frame claims ({100 * y.mean():.0f}% refuted)")
    print(f"OBS challenger, {material}: {what}, {len(seen)} distinct terms tried")
    out = []
    frame = frames.get(material, [])
    for sc, t, th, v in cands[:top]:
        yt, rt = y[~train], (v >= th)[~train]
        prec = float(yt[rt].mean()) if rt.sum() else 0.0
        outside = float(yt[~rt].mean()) if (~rt).sum() else 0.0
        rec = float((yt & rt).sum() / max(yt.sum(), 1))
        p = by_game_p(v >= th, y, [r[2] for r in rows], ~train) if phase else excess(yt, rt)
        inside = [f for f, ok in zip(fens, (v >= th) & ~train & y) if ok]  # refuted test boards inside R
        inside_games = [r[2] for r, ok in zip(rows, (v >= th) & ~train & y) if ok]
        sample = random.Random(seed).sample(inside, min(check_n, len(inside)))
        chk = [c for c in A.trap_check(sample, frame, t, th) if c is not None]
        trap = sum(chk) / max(len(chk), 1)
        out.append(dict(term=t, theta=th, gloss=V.gloss(V.parse(t)), precision=prec, outside=outside, p=p, recall=rec,
                        lift=prec / max(base_rate, 1e-9), trap=trap, checked=len(chk), boards=inside, games=inside_games))
        if len(out) > 5 and not (p < ALPHA / top and len(chk) >= 10):
            continue                                                    # (the first five printed; past them, only those kept)
        print(f"OBS   R = {{ {V.gloss(V.parse(t))} >= {th} }}: held-out, refuted {prec:.2f} inside R against {outside:.2f} outside "
              f"(base {base_rate:.2f}; one-sided {'Mann-Whitney over games' if phase else 'Fisher'} p {p:.1e}), recall {rec:.2f}; "
              f"the local trap check holds on {sum(chk)}/{len(chk)} of its refuted boards")
    traps, blinds = split(out)
    print(f"OBS   ({time.time() - t0:.0f} s; {len(out)} regions tested, kept at p < {ALPHA}/{len(out)}: {len(traps)} trap(s), {len(blinds)} blind spot(s))")
    strip = lambda c: {k: v for k, v in c.items() if k not in ("boards", "games")}
    trap, blind = pick(traps, blinds)
    status = " + ".join(s for s, c in (("counter-frame", trap), ("blind spot", blind)) if c) or "vocabulary gap"
    return dict(material=material, status=status, frame=frame, trap=trap, blind=blind, candidates=[strip(c) for c in out],
                **({} if trap or blind else dict(gap_boards=[f for f, ok in zip(fens, y & ~train) if ok][:200])))


def answer(cf: dict, material: str, kind: str, run: str, rnd: int, frames: dict, defender: str = "tablebase",
           n_boards: int = 50, budget: int = 60) -> dict:
    """The attacker's lemma-incorporation as a STAGE (the red team: "a new frame whose goal is 'leave R'"): a situational
    frame "<material>:if:<term>>=<theta>", active while R holds, its measure MINED by the pipeline like any other on R's
    refuted boards. kind "trap" (a counter-frame): its verifier games succeed on leaving R (or the material goal) without
    losing ground; kind "blind" (a blind spot): on the game's own ends -- held, won, progress. Against the material's own
    frame inside R (no stage)."""
    from .pipeline import run as pipeline_run
    import chess
    def fresh(f):
        b = chess.Board(f); b.halfmove_clock = 0
        return b.fen()
    from .verifier import split, confirm
    from .verifier import spread
    picked, pg = spread(cf["boards"], cf["games"], 100)
    choose, check, check_games = split([fresh(f) for f in picked], n_boards, pg, seed=f"{run}|{material}|{rnd}|{kind}")
    boards = choose + check                                     # choosing and confirming on different games, choosing first
    fkey = f"{material}:if:{cf['term']}>={cf['theta']}"
    hole = dict(name=f"stage_{run}_{material}_r{rnd}_{kind}", material=material, frame_key=fkey, frames=frames, piece="*",
                base=[], k=2, stuck=boards, sampled=len(boards), leave=[cf["term"], cf["theta"]] if kind == "trap" else None)
    json.dump(hole, open(paths.derived(f"hole_{hole['name']}.json"), "w"))
    out = pipeline_run(hole["name"], budget, hole=hole, defender=defender, n_boards=len(choose))
    from .pipeline import best_of
    base, best = out[0], best_of(out[1:])
    c = confirm(hole, best, check, defender, check_games)     # on games not used to choose it, one vote a game
    goal = "left R or won" if kind == "trap" else "won"
    show = lambda r: f"{goal} {r['mated']}/{r['n']}, lost ground {r['lost']}, lemma-states {r.get('states', 0) / max(r['n'], 1):.1f} a board"
    print(f"OBS   the attacker's answer -- a STAGE for the {'counter-frame' if kind == 'trap' else 'blind spot'} R ({fkey}): the best mined "
          f"measure {best and best['term']}; on {c['n']} boards not used to choose it {c['cand'] and show(c['cand'])}, without the stage "
          f"{c['base'] and show(c['base'])}; better in {c['w']} of its {c['games']} games, worse in {c['l']} -- e-value {c['e']:.3g} -> "
          f"{'PROMOTED' if c['promoted'] else 'not promoted'} (at e >= 20)")
    return dict(kind=kind, frame_key=fkey, term=best and best["term"], schema=best and best["schema"], base=c["base"] and show(c["base"]),
                answer=c["cand"] and show(c["cand"]), w=c["w"], l=c["l"], e=c["e"], promoted=bool(c["promoted"]))


def archive(entry: dict) -> None:
    p = paths.derived(ARCHIVE)
    arc = json.load(open(p)) if os.path.exists(p) else []
    arc.append({k: v for k, v in entry.items() if k not in ("boards",)})
    json.dump(arc, open(p, "w"), indent=1)


def report(material: str, cf: dict) -> None:
    for kind, c in (("trap", cf.get("trap")), ("blind", cf.get("blind"))):
        if c and kind == "trap":
            print(f"OBS COUNTER-FRAME for {material}: when {c['gloss']} >= {c['theta']}, the defender holds {cf['frame']} (local trap "
                  f"check {c['trap']:.2f} on {c['checked']} boards; held-out refuted {c['precision']:.2f} inside R, {c['outside']:.2f} outside)")
        elif c:
            print(f"OBS BLIND SPOT for {material}: when {c['gloss']} >= {c['theta']}, the frame {cf['frame']} chooses wrong (held-out "
                  f"refuted {c['precision']:.2f} inside R, {c['outside']:.2f} outside; the local trap check only {c['trap']:.2f} -- we had a move)")
    if not (cf.get("trap") or cf.get("blind")):
        print(f"OBS {cf['status'].upper()} for {material}" + (f" -- {len(cf.get('gap_boards', []))} refuted boards kept for new predicates"
                                                             if cf["status"] == "vocabulary gap" else ""))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--material", default="KRRvKR")
    ap.add_argument("--no-answer", action="store_true")
    a = ap.parse_args(argv)
    stem = f"surprise_loop_KRRvKR_{a.run}"
    history = json.load(open(paths.derived(f"{stem}.json")))
    frames = next(h["frames"] for h in history if h["round"] == a.round)
    res = json.load(open(paths.experiment(f"{stem}_r{a.round}.json")))
    import catspace.incubator.analyst  # noqa: F401
    import analyst_core as A
    for k, v in frames.items():
        A.set_material_frame(k, v)
    cf = mine(res, frames, a.material)
    report(a.material, cf)
    entry = dict(run=a.run, round=a.round, t=time.strftime("%Y-%m-%d %H:%M"), **strip_boards(cf))
    if not a.no_answer:
        entry["answers"] = [answer(cf[kind], a.material, kind, a.run, a.round, frames) for kind in ("trap", "blind") if cf.get(kind)]
    archive(entry)


def strip_boards(cf: dict) -> dict:
    return {k: ({kk: vv for kk, vv in v.items() if kk not in ("boards", "games")} if isinstance(v, dict) else v)
            for k, v in cf.items() if k != "gap_boards"}


if __name__ == "__main__":
    main()


def round_(res: list, played: dict, frames: dict, run: str, rnd: int, register: bool = True) -> dict:
    """The challenger inside the loop, after a round's mining: per material with a frame of ours, a trap and a blind spot
    mined from the claims of the frame the round was PLAYED with and checked against that frame (the data's own frame --
    2026-10-12: checked against the newly mined one it reported gaps); each answered by a stage mined with the NEW frames
    (the next round's); a promoted stage joins them. -> frames."""
    new = dict(frames)
    from .surprise_loop import MATING_SPECS
    ahead = {r["material"] for g in res for r in g["rows"] if r["lead"] > 0}
    for material in sorted(({k for k in played if ":" not in k and played[k]} | ahead) - set(MATING_SPECS)):
        cf = mine(res, played, material)
        report(material, cf)
        entry = dict(run=run, round=rnd, t=time.strftime("%Y-%m-%d %H:%M"), **strip_boards(cf), answers=[])
        for kind in ("trap", "blind"):
            c = cf.get(kind)
            if not c:
                continue
            ans = answer(c, material, kind, run, rnd, new)
            entry["answers"].append(ans)
            if ans["promoted"]:
                new[ans["frame_key"]] = [ans["term"]]
                if register:
                    from . import registry as R
                    R.add_generated(f"stage_{material.lower()}_{run}_r{rnd}_{kind}", ans["term"], "measure", f"stage_{run}_{material}_r{rnd}_{kind}",
                                    {"verifier": f"{ans['answer']} against {ans['base']} without the stage (confirmation boards; e-value {ans['e']:.3g})",
                                     ("trap" if kind == "trap" else "blind spot"): f"{c['gloss']} >= {c['theta']}",
                                     "status": "a conjecture: the stage's claim is not proved"}, used_by=f"the stage {ans['frame_key']}")
                    R.export()
        archive(entry)
    return new
