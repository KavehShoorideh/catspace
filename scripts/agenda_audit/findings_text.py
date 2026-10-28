FINDINGS = r'''## 5. Findings

### Items hard to classify, and how they were classed

1. **Value against value.** STOP (a) `self.best[0] > nstake` (agenda:1583), "chase only above the best" `self.best[0] >= q.stake` (4095), the memo served by need (2726, 2729) compare two machinery quantities: LOGIC. But every such quantity is in material units, so they are generic only once the valuation is read through one entry (75.2 (a)). Where a side is the ladder's own definition (`s0 >= 1`, `balance + value >= line`, `need = line - balance`) or a chess constant (100, 8, a worth, the unit 1): CHESS FACT.
2. **Kernel reads that carry a chess decision.** `if not self.board.is_check()` in `_decide_by_smallest_loss` (2582: "their silence is zero only in check"); no fork roots, no arrangements, no pass while in check (681, 790, 914, 1000): RULE. The decision rides on the read and has no atom of its own, so D4's and D21's row counts understate them.
3. **One test, two roles.** `q.goal == "mate"` picks the square a trace line names in `_inherit` (1719) and `_serve` (1775): NAME. The same test routes the agenda in `_ask` (2128-2155) and `_chase` (4086): CHESS FACT.
4. **Lemma names tested by prefix decide behaviour.** `claim.lemma.startswith(("chase", "delayed", "mate_in"))` decides portability (1815); `startswith("chase")` (1821, 1890); `bind.schema.startswith("delayed_") and endswith("_d1")` gates the leaf read (2397); `"balance" in str(claim.lemma)` (2434); `"repetition" in x` (4307). Classed CHESS FACT (D26, D16, A8): these are properties of store entries read off their names, and belong in entry fields.
5. **The machinery's own reason texts, parsed back.** `res["gap"].startswith("chain stop")` / `("out of time")` (3215, 3315, 4256-4259); `str(q.reason).startswith("stopped:")` and a trace line rewritten by its prefix (3974-3981); `why.startswith("the threat (")` (3753): LOGIC. `_refuted`'s regex on a kill text (1331-1332) is CHESS FACT D13 (a loss read off "(-n)", "they mate us", "stalemate", "no parry"). Plan 71.5's typed reasons would remove all of them.
6. **FORCED_WORDS (105) has three roles:** its definition, the follow list below the root (4173-4174) and the judge's forcedness (3474). Classed B6, and B7 at the judge.
7. **Tier numbers inside calls.** `forcing_why(..., mate2=k >= 2, delayed=... k >= 1)` and `_t(FLAG, 3)`: HEURISTIC (the tier composition) where compared, LOGIC inside `_t`. The 52.23 flags have been constant True since 64.2, so `_t(...)` always holds.
8. **The budget** `our units x 8 + our captures` (393-394). The 8 is "the squares round their king" (52.2), so it is the fuse's SHAPE: CHESS FACT B4. FUSE = 10 and the x 2^tier are HEURISTIC values.
9. **Switches whose comment names a chess rule** (SECOND_CHECK_STOP 87, CLOCK_PAWN 78, EVADED_RESET 88, CLOSE_AFTER_CAPTURE 95) are classed LOGIC like every switch. The rule they gate is classed where it acts.
10. **`_local_class` (2013-2036)** mixes the frame's three questions with goal kinds. Classed LOGIC (the frame), except its goal-kind tests.
11. **The literal 1 as the unit of material** cannot be told from a count, an index or a step by a mechanical pass. The sites read by hand (section 2) may not be all of them.

### Rules re-implemented in Python instead of read from the kernel

1. **Castling's rook** (agenda:569-573): H1 / A1 to F1 / D1, +56 for Black, to place the rook as the checker of a castling mate root. The kernel already applies castling (`Game.apply`).
2. **Position identity and repetition.** `_pos_key` (133-136) is the first four FEN fields. The chase's cycle (`_inner` 2732) and `_cycle_claim` (3708-3723) treat ONE repeat of the line or of the game as a draw. Lean `Ideas.Forcing.repetition_level` justifies the value, but the kernel's rule is the threefold / claimable draw. `Game.terminal` (game.py:89-96) has no repetition and no fifty-move terminal at all. The chase clock's "irreversible event" (agenda:3994-4018) is the fifty-move rule's events written by hand.
3. **The exchange's swap list** (`_principal_captures`, 3755-3793): cheapest capturer first and the optimal stop, written in Python. It sits beside the Rust `Pos.exchange` / `see`, which mirror Lean `Ideas.ExchangeLaw`, so there are two implementations.
4. **Check and capture read off the notation.** `_parry_kind` (3500-3502) looks for '+', '#' or 'x' in the SAN to set the judge's rates. `_mate_line_pieces` (4560-4589) replays a claim's line by parsing SAN tokens.
5. **Promotion built by hand.** `chess.Move(sq, path[0], promotion=chess.QUEEN if len(path) == 1 ...)` (3989, 4700, 4811): the push promotes on the last step, and always to a queen (underpromotion is never asked). `_direct_routes` excludes promotion captures (2366); the mate roots place the promoted piece (567).
6. **Attack geometry on an empty board** (`_aligned` -> `regress._aligned`, used by `_scan_reach` 738), and the king's step written as `chess.BB_KING_ATTACKS` (1934, 4584) beside `Game.adjacent`.
7. **Mate in one by python-chess** (`G.mates` 1941, `_mates` 2654-2668), beside the Rust `Pos.mate_in_one` that the chase uses (2232, 3396).
8. **Material in the interfaces.** `Game.measure` (game.py:98-100) and `BaseWords.worth` (words.py:44) supply material as an axiom and as a base word, while plan 75 calls material a derived valuation. The stake of mate is defined three times (game.py:23 `MATE_VALUE`, binder.py:24 `MATE_VALUE`, regress.py:35 `STAKE_MATE`), and agenda.py reads two of them (`BD.MATE_VALUE`, `R.STAKE_MATE`).

### Constants with no stated source (no plan item, measurement or reason beside them)

- `0.05` s as "no time left" (agenda:1063, 1141, 1146, 1154, 1179, 1285).
- `1 << 12`, the price limit of the unfinished line's tie-break (1387).
- `pool / 4` and `0.5` s, the slice before any rate is known (3617).
- `12`, the longest capture line in the swap list (3775).
- `JUDGE_MIN_AGENDAS = 8` (104): "until this many cost units have been timed", with no measurement.
- `ents[:1]` (638): one technique per class, the first by name.
- `[:3]` lemmas in a chase claim's name (4310). This truncates the DECISION line's lemma string, so it changes what the store counts.
- ponder.py: `3600.0` s as the default slot (113); `1 << 30` as "no count limit" (63, 133).
- Print caps (`[:4]`, `[:5]`, `[:6]`, `[:8]`, `[:12]`, `40`, `[:60]`, `[:80]`, `[:300]`): these shape text only.

These have a stated source:

- FUSE 10 (52.13 (2b)).
- DEPTH_FUSE 30 (the Python stack).
- CHASE_CLOCK 4 (Kaveh's starting value, "tune it later").
- JUDGE_PRICE_CAP 120 (64.3: pricing took 7.2 of 15 s on 07Z0P).
- TIER_COUNT 4 (plan 59).
- RULE2_FROM's 2 / 1 / 1 (52.17 (2)).
- The halves (52.23 (11), 69.1, 72.9 (b), and ponder's hedge of 2026-10-22).
- The budget's 8 (52.2).
- The `- 64` offset of the balance word (stated at 999).

### Names used as knowledge

1. **Lemma, schema and forcing-word names.** Item 4 above covers the lemma names. The forcing words are also read off their printed text:
   - the judge and the follow list test `r.why.removeprefix("fix: ").startswith(FORCED_WORDS ...)` (3474, 4174);
   - the route sort tests `== "check"` / `startswith("capture")` (4179);
   - the counters test `startswith("delayed threat" / "no safe move")` (4129-4131);
   - the walk of checks tests `why != "check"` (3907).

   A FORCING entry is identified by its name string, not by its kind.
2. **Notation.**
   - SAN symbols (3502).
   - uci strings recognised by their letters (1451, 1486).
   - A uci slice compared with a square name (3814).
   - From / to squares cut out of a uci for the shape key (539).
   - SAN tokens replayed (4566-4571).
3. **Piece symbols compared to decide:** `b.piece_at(frm).symbol() != p.get("sym")` (the landing table's guard, 2197); `l.sym == pc.symbol().lower()` (1507, text only).
4. **A goal kind used as a placeholder:** `ponder.resume_gap` files every resumed route under a question of kind 'win' (ponder:41), whatever goal cut it.
5. **`worth` as a base word.** words.py's BaseWords supplies `worth` (the unit values) as a base word: a valuation passed off as a name.

### Method notes

- **The snapshots are of this code.** They come from the 73.5b acceptance branch, merged as main ba587d27. agenda.py has not changed since 635990e9, which that branch contains, and ponder.py changed earlier.
- **The count budget hides the status, the pool and the judge.** At the count budget the root runs `_ask_once`: no status line, no pool, no judge, no resume. A5 (the ladder) therefore shows only through the unfinished-line decision, and B7 / B8 (the judge, still forcing) show 0. A timed run would count them.
- **The kinds of the proposed entries** are the six this audit was asked for: GOAL, STOP, FORCING, CLOSING, HEURISTIC, VALUATION. GOAL and VALUATION are new to the store (plan 75.2 (a)). The store's own DEFECT_FIX kind is not used for a proposal: a defect that is not in the store is proposed as part of its goal. Their 'reads' are the four base words of 73.1.
- **The number for 75.2 to drive to zero** is the CHESS FACT row count of section 2. These scripts can recount it if they are kept: they are not committed (the spec asked for one document).
'''
