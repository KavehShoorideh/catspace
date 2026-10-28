# AGENDA_AUDIT.md -- plan 75.1, the agenda audit

Read-only classification of `catspace/incubator/analyst/discovery/agenda.py` (the agenda, the chase, the tiers driver `_ask_tiers`, the pool driver `_ask_pool`, the judge `judge_next` / `_decides` / `_judge_*` / `_competitor` -- they live there; no other module defines them) and `ponder.py`, as of main 65024fbf (agenda.py last changed by 635990e9, ponder.py earlier). No engine file, test or other doc was changed; no gate, no snapshot, no timed run. Every number below is printed by the audit's own scripts (OBS lines, section 2). The extractor and classifier are not committed (scratch scripts); the classification rules they apply are stated in section 1.

## 1. The principle and the five classes

THE PRINCIPLE (plan 75, Kaveh 2026-10-27). The goal is the OUTCOME: don't lose, then win. We supply NAMES -- attack, move, slide, capture, check, mate, stalemate, draw, win, lose, the unit kinds, rank / file, tempo -- each bound to a definition of the rule kernel, never to a derived fact. Everything else -- words, valuations (material!), techniques, stops -- is DERIVED and lives in the lemma store, not in agenda.py.

THE FIVE CLASSES, as applied (each item exactly one; the first that fits):

* **LOGIC** -- goal regression, AND-OR over replies, frames and read sets, the heap and its stake order, memoisation and the fact store, the budgets' MECHANICS (a count or a clock against what is left), the tier / pool plumbing, the switches (a flag's test; what it gates is classed where it acts), trace plumbing (a branch or a print cap that only shapes a line). A comparison of two machinery quantities (a value with a stake, a price with the time left).
* **RULE** -- a fact of the rules read through the kernel: legal moves, check, mate, stalemate, a capture, castling, promotion, attack, a square's occupant and owner, the side to move -- even where it gates a chess decision. Rules re-implemented in Python are RULE rows (listed in the findings).
* **NAME** -- a supplied name used as a label: the colours, the square set, square / uci notation, the side letters, a goal or defect named in a trace line.
* **CHESS FACT** -- derived knowledge as code: a goal-kind test or literal that routes the agenda; a stake (mate 100, promotion 8, a unit's worth); the status ladder's thresholds, line and need; the gain unit (value >= 1); a piece-kind or defect-kind test that decides; the forcing words; a stop's shape (the clock, tempo arithmetic, the second check, the fuse, the walk of checks); a tie-break keyed by chess content (worth, the shape key, checks first). Each row names its FAMILY (section 4) and its store entry, existing or proposed (name, kind, reads).
* **HEURISTIC** -- a tuned number or rule with no proof obligation: budget and fuse values, time margins and shares, the tier composition, the policy's 1/n bar, generic tie-breaks (the uci, the price).

## 2. OBS lines

```
OBS audit atoms agenda.py: 2490 (CONST 49, BRANCH 2132, COMPARE 27, ORDER 37, LITERAL 180, KIND 47, read by hand 18)
OBS audit atoms ponder.py: 48 (CONST 0, BRANCH 38, COMPARE 1, ORDER 1, LITERAL 4, KIND 1, read by hand 3)
OBS audit rows total 1719: LOGIC 1123, RULE 104, NAME 23, CHESS FACT 428, HEURISTIC 41
OBS audit rows agenda.py 1682: LOGIC 1097, RULE 102, NAME 21, CHESS FACT 423, HEURISTIC 39
OBS audit rows ponder.py 37: LOGIC 26, RULE 2, NAME 2, CHESS FACT 5, HEURISTIC 2
OBS audit CHESS FACT rows 428: with an existing store entry 171, without (an entry proposed) 257; families 65 (existing 38, proposed 27)
OBS audit HEURISTIC rows 41
OBS audit CHESS FACT rows that only shape trace text 5 (counted: the knowledge is still written in agenda.py)
OBS audit store names checked: 82 citations of 98 entries; not in the store 0
OBS audit guard snapshots read: tier 0 cases 109, tier 2 cases 109 (ten + wide, the count budget, PYTHONHASHSEED=0; main ba587d27's agenda.py = this commit's)
```

The per-family counts (section 4) are the `OBS audit fact <id>` lines of the count script, printed beside these.

## 3. The table

WHAT AN ITEM IS (mechanical, from the AST): a module or class CONSTANT; each operand of an `and` / `or` / `not` in an `if` / `elif` / `while` / `assert` / conditional expression / comprehension filter, and in a short-circuit outside a test (a BRANCH atom); a COMPARISON outside a test; an ORDER (`sorted` / `min` / `max` / `.sort` with a key, `heappush`); a LITERAL (a number other than 0 and 1, a `chess.*` or another module's UPPERCASE constant); a KIND literal (the goal / claim / defect kind a question, claim or defect is built with). Docstrings and f-string text are not items. The literal 1 as the unit of material, a few formulas outside any test and two label dicts were read by hand ('read by hand' in section 2). ONE ROW = the atoms of one line, one class and one CHESS FACT family: a line that mixes classes gives one row per class.

Columns: file:line | the item's text (its atoms joined by ' ; ', trimmed; 'in:' gives a literal's statement) | class ('text only': a CHESS FACT that only shapes a trace line) | store entry ([family] existing name, or PROPOSED name, kind, reads) | decisions depending ('cases t0/t2' = cases of 109 at tier 0 / tier 2 passing through the family, 'DECISION' = DECISION lines of 218 citing it -- the method heads section 4 -- or the row count where nothing is printed). The last two columns are for CHESS FACT rows only.

### agenda.py

| file:line | text | class | store entry | decisions depending |
|---|---|---|---|---|
| agenda.py:67 | `GAME = GM.CHESS` | LOGIC | -- | -- |
| agenda.py:68 | `W = WD.CHESS_WORDS` | LOGIC | -- | -- |
| agenda.py:71 | `DEPTH_FUSE = 30` | HEURISTIC | -- | -- |
| agenda.py:72 | `RULE2_FROM = {"mate": 2, "win": 1, "promote": 1}` | CHESS FACT | [B5] PROPOSED: "stop: a check that fixes no defect of its goal (the walk of checks)" (STOP; reads attack, movement, tempo) | cases 0/0, DECISION 0 |
| agenda.py:72 | `RULE2_FROM values 2 / 1 / 1 (the depths)` | HEURISTIC | -- | -- |
| agenda.py:73 | `FUSE = 10` | HEURISTIC | -- | -- |
| agenda.py:74 | `NEED_AWARE_CLOSE = True` | LOGIC | -- | -- |
| agenda.py:75 | `EVASION_CHASE = True` | LOGIC | -- | -- |
| agenda.py:76 | `NEED_PROPAGATION = True` | LOGIC | -- | -- |
| agenda.py:77 | `GOAL_JUDGE = True` | LOGIC | -- | -- |
| agenda.py:78 | `CLOCK_PAWN = True` | LOGIC | -- | -- |
| agenda.py:79 | `SURVEY = True` | LOGIC | -- | -- |
| agenda.py:80 | `SKELETON_GOALS = os.environ.get("CATSPACE_SKELETON", "1") != "0"` | LOGIC | -- | -- |
| agenda.py:81 | `PASSER_ROOTS = True` | LOGIC | -- | -- |
| agenda.py:82 | `PARRY_PASS = True` | LOGIC | -- | -- |
| agenda.py:83 | `THEIR_GAPS = True` | LOGIC | -- | -- |
| agenda.py:84 | `STUCK_LEAF = True` | LOGIC | -- | -- |
| agenda.py:85 | `LOCAL_INNER = True` | LOGIC | -- | -- |
| agenda.py:86 | `CHASE_SAVES = True` | LOGIC | -- | -- |
| agenda.py:87 | `SECOND_CHECK_STOP = True` | LOGIC | -- | -- |
| agenda.py:88 | `EVADED_RESET = False` | LOGIC | -- | -- |
| agenda.py:89 | `CLOCK_IRREVERSIBLE = True` | LOGIC | -- | -- |
| agenda.py:90 | `CHASE_CLOCK = 4` | HEURISTIC | -- | -- |
| agenda.py:92 | `DEEP_WIN_CHASE = True` | LOGIC | -- | -- |
| agenda.py:93 | `PRIOR = False` | LOGIC | -- | -- |
| agenda.py:95 | `CLOSE_AFTER_CAPTURE = True` | LOGIC | -- | -- |
| agenda.py:96 | `RAISE_TO_WIN = True` | LOGIC | -- | -- |
| agenda.py:97 | `PARRIES = True` | LOGIC | -- | -- |
| agenda.py:98 | `DELAYED_THREAT = True` | LOGIC | -- | -- |
| agenda.py:99 | `NO_SAFE_MOVE = True` | LOGIC | -- | -- |
| agenda.py:100 | `ESTIMATE = True` | LOGIC | -- | -- |
| agenda.py:101 | `COST_JUDGE = True` | LOGIC | -- | -- |
| agenda.py:102 | `CYCLE = True` | LOGIC | -- | -- |
| agenda.py:103 | `JUDGE_PRICE_CAP = 120` | HEURISTIC | -- | -- |
| agenda.py:104 | `JUDGE_MIN_AGENDAS = 8` | HEURISTIC | -- | -- |
| agenda.py:105 | `FORCED_WORDS = ("check", "capture", "threat of mate")` | CHESS FACT | [B6] PROPOSED: "stop: the chase's follow list (forced words below the root)" (STOP; reads attack, movement, tempo) (beside existing: check; capture) | no printed signature: 10 rows |
| agenda.py:106 | `TIERS_UNDER_TIME = True` | LOGIC | -- | -- |
| agenda.py:108 | `TIERS = False` | LOGIC | -- | -- |
| agenda.py:109 | `TIER_COUNT = 4` | HEURISTIC | -- | -- |
| agenda.py:110 | `COMPETE = True` | LOGIC | -- | -- |
| agenda.py:111 | `UNPROVED = True` | LOGIC | -- | -- |
| agenda.py:112 | `TECHNIQUE = True` | LOGIC | -- | -- |
| agenda.py:113 | `DEEP_PASS = True` | LOGIC | -- | -- |
| agenda.py:114 | `STOP_AFFORD = True` | LOGIC | -- | -- |
| agenda.py:115 | `RESUME = True` | LOGIC | -- | -- |
| agenda.py:116 | `POOL = True` | LOGIC | -- | -- |
| agenda.py:117 | `DEFER = True` | LOGIC | -- | -- |
| agenda.py:118 | `CARRY = True` | LOGIC | -- | -- |
| agenda.py:119 | `NO_FUSE = 1 << 40` | LOGIC | -- | -- |
| agenda.py:125 | `competitor is None ; price <= competitor` | LOGIC | -- | -- |
| agenda.py:136 | `4  in: return " ".join(fen.split()[:4])` | RULE | -- | -- |
| agenda.py:162 | `(new[0], new[1]) != (old[0], old[1])` | LOGIC | -- | -- |
| agenda.py:163 | `(new[0], new[1]) > (old[0], old[1])` | LOGIC | -- | -- |
| agenda.py:164 | `ag._claim_key(new[2]) > owner._claim_key(old[2])` | LOGIC | -- | -- |
| agenda.py:170 | `facts is not None` | LOGIC | -- | -- |
| agenda.py:321 | `self.deadline is None` | LOGIC | -- | -- |
| agenda.py:323 | `time.monotonic() >= self.deadline` | LOGIC | -- | -- |
| agenda.py:325 | `self.hand is None` | LOGIC | -- | -- |
| agenda.py:326 | `self.final is not None ; self.final - self.deadline > 1e-6` | LOGIC | -- | -- |
| agenda.py:327 | `sliced` | LOGIC | -- | -- |
| agenda.py:328 | `self.hand ; self.abort` | LOGIC | -- | -- |
| agenda.py:335 | `self.hand is not None ; self.abort ; frame > 0 ; frame == self.hand` | LOGIC | -- | -- |
| agenda.py:343 | `LOCAL_INNER ; goal is not None ; frame is not None ; diff is not None ; chase_depth ; their is None ; ctx is not None` | LOGIC | -- | -- |
| agenda.py:347 | `chase_depth == 0 ; ctx is None ; carry is None` | LOGIC | -- | -- |
| agenda.py:357 | `inherit ; {}` | LOGIC | -- | -- |
| agenda.py:366 | `ctx is not None ; side_depth ; carry is not None` | LOGIC | -- | -- |
| agenda.py:367 | `self.ctx is not None ; chase_log is not None` | LOGIC | -- | -- |
| agenda.py:369 | `ctx is None ; self.ctx is not None` | LOGIC | -- | -- |
| agenda.py:370 | `memory ; []` | LOGIC | -- | -- |
| agenda.py:378 | `pos is not None ; G.BRIDGE_CHECK` | LOGIC | -- | -- |
| agenda.py:381 | `self.me` | NAME | -- | -- |
| agenda.py:393 | `self.board.piece_at(s) ; self.board.piece_at(s).color == self.me` | RULE | -- | -- |
| agenda.py:393 | `chess.SQUARES  in: self.budget = sum(1 for s in chess.SQUARES if self.board.piece_at(s) and self.board.piece_at(s).co...` | NAME | -- | -- |
| agenda.py:393 | `8  in: self.budget = sum(1 for s in chess.SQUARES if self.board.piece_at(s) and self.board.piece_at(s).color == self....` | CHESS FACT | [B4] existing: budget; chase fuse | cases 85/84, DECISION 0 |
| agenda.py:394 | `self.board.is_capture(m)` | RULE | -- | -- |
| agenda.py:396 | `ctx is None ; self.ctx is not None` | LOGIC | -- | -- |
| agenda.py:399 | `seconds is not None ; TIERS ; TIERS_UNDER_TIME` | LOGIC | -- | -- |
| agenda.py:401 | `seconds is not None ; POOL` | LOGIC | -- | -- |
| agenda.py:404 | `seconds is not None` | LOGIC | -- | -- |
| agenda.py:407 | `side_depth ; their is None` | LOGIC | -- | -- |
| agenda.py:417 | `R.STAKE_PROMOTE  in: self.promote_stake = R.STAKE_PROMOTE` | CHESS FACT | [A3] PROPOSED: "stake: promotion (queen less pawn = 8)" (VALUATION; reads material, movement) | cases 35/35, DECISION 0 |
| agenda.py:424 | `self._root ; self.ctx is not None` | LOGIC | -- | -- |
| agenda.py:435 | `self.ctx is not None` | LOGIC | -- | -- |
| agenda.py:437 | `self.ctx is not None` | LOGIC | -- | -- |
| agenda.py:438 | `self.ctx is not None ; self.ctx.deadline is not None ; time.monotonic() >= self.ctx.deadline` | LOGIC | -- | -- |
| agenda.py:441 | `SURVEY ; self._root` | LOGIC | -- | -- |
| agenda.py:444 | `SKELETON_GOALS ; self._root` | LOGIC | -- | -- |
| agenda.py:448 | `getattr(self, "_tq", None)` | LOGIC | -- | -- |
| agenda.py:450 | `self._tech is not None` | LOGIC | -- | -- |
| agenda.py:456 | `flag ; self.ctx is not None ; self.ctx.tier >= k` | LOGIC | -- | -- |
| agenda.py:460 | `self.ctx is not None ; self.ctx.tier >= 3 ; 4  in: return min(CHASE_CLOCK, 4) if self.ctx is not None and self.ctx.ti...` | HEURISTIC | -- | -- |
| agenda.py:467 | `self._pos is None` | LOGIC | -- | -- |
| agenda.py:474 | `self._fen is None` | LOGIC | -- | -- |
| agenda.py:480 | `sorted(... key=lambda q: (-q.stake, q.depth, q.id))` | LOGIC | -- | -- |
| agenda.py:485 | `force` | LOGIC | -- | -- |
| agenda.py:485 | `q.stake + (self._top_other(q) if self._t(NEED_PROPAGATION, 3) else 0) < self.min_stake` | CHESS FACT | [A7] PROPOSED: "goal: the need (what a claim must bring), propagated into the chase" (GOAL; reads material) | no printed signature: 15 rows |
| agenda.py:487 | `self.ctx is not None ; self.ctx.deadline is not None ; self.ctx.out_of_time() ; self.late ; self.ctx.finishing(self.f...` | LOGIC | -- | -- |
| agenda.py:490 | `self.units is not None ; force ; self._unit_root(q)` | LOGIC | -- | -- |
| agenda.py:492 | `self.local is not None ; self.local["full"] ; force` | LOGIC | -- | -- |
| agenda.py:494 | `cls is None` | LOGIC | -- | -- |
| agenda.py:500 | `heapq.heappush(self.heap, (-q.stake, -self._recall_score(q)[0], q.depth, self._seq, q))` | LOGIC | -- | -- |
| agenda.py:505 | `r.get("goal") == q.goal` | LOGIC | -- | -- |
| agenda.py:506 | `hit ; max(... key=lambda t: t[0]) ; 3  in: return (round(sum(s for s, _ in hit), 3), max(hit, key=lambda t: t[0])[1] ...` | LOGIC | -- | -- |
| agenda.py:514 | `any(r.move.uci() == move for r in R.all_routes(g))` | LOGIC | -- | -- |
| agenda.py:524 | `claim.proved ; claim.word ; any(x.id == claim.qid for x in self.all)` | LOGIC | -- | -- |
| agenda.py:526 | `old is None ; (value, second) > old[:2]` | LOGIC | -- | -- |
| agenda.py:528 | `self.best is None` | LOGIC | -- | -- |
| agenda.py:530 | `(value, second) != self.best[:2]` | LOGIC | -- | -- |
| agenda.py:531 | `(value, second) > self.best[:2]` | LOGIC | -- | -- |
| agenda.py:532 | `self._claim_key(claim) > self._claim_key(self.best[2])` | LOGIC | -- | -- |
| agenda.py:536 | `q.id == c.qid` | LOGIC | -- | -- |
| agenda.py:537 | `c.move ; ""` | LOGIC | -- | -- |
| agenda.py:539 | `len(mv) >= 4 ; 2  in: sh = PA.shape_key(PA.shape(self.fen, [(mv[:2], mv[2:4])], self.us, threats=False)) if len(mv) >...` | LOGIC | -- | -- |
| agenda.py:542 | `sh` | LOGIC | -- | -- |
| agenda.py:542 | `c.proved_by == "table"` | CHESS FACT | [D5] PROPOSED: "heuristic: the claim order at equal value (word > table, stake, shape key)" (HEURISTIC; reads material, attack) (beside existing: the shape key) | cases 1/1, DECISION 0 |
| agenda.py:553 | `b.is_check()` | RULE | -- | -- |
| agenda.py:555 | `new_question('save', ...)` | CHESS FACT | [A1s] PROPOSED: "goal: save material (don't lose)" (GOAL; reads material, attack) (beside existing: in check) | cases 78/83, DECISION 0 |
| agenda.py:555 | `R.STAKE_MATE  in: self._push(R.new_question("save", R.STAKE_MATE, dict(checkers=ck), text="save our king (in check)"))` | CHESS FACT | [A2] PROPOSED: "stake: the terminal (mate = 100)" (VALUATION; reads movement, attack) | cases 97/97, DECISION 0 |
| agenda.py:557 | `any(m.to_square == c for m in b.legal_moves)` | RULE | -- | -- |
| agenda.py:559 | `new_question('win', ...)` | CHESS FACT | [A1w] PROPOSED: "goal: win material" (GOAL; reads material, attack) | cases 83/86, DECISION 0 |
| agenda.py:565 | `ksq is None` | LOGIC | -- | -- |
| agenda.py:565 | `b.is_castling(m)` | RULE | -- | -- |
| agenda.py:565 | `pc.piece_type == chess.KING` | CHESS FACT | [A1m] PROPOSED: "goal: mate (the rule's terminal)" (GOAL; reads movement, attack) | cases 100/100, DECISION 0 |
| agenda.py:567 | `m.promotion` | RULE | -- | -- |
| agenda.py:569 | `b.is_castling(m)` | RULE | -- | -- |
| agenda.py:570 | `m.to_square > m.from_square ; chess.H1  in: rf, rt = (chess.H1 if m.to_square > m.from_square else chess.A1), (chess....` | RULE | -- | -- |
| agenda.py:571 | `pc.color == chess.BLACK` | RULE | -- | -- |
| agenda.py:572 | `56  in: rf, rt = rf + 56, rt + 56` | RULE | -- | -- |
| agenda.py:577 | `more` | LOGIC | -- | -- |
| agenda.py:578 | `ksq in cf.attacks(checker)` | RULE | -- | -- |
| agenda.py:582 | `new_question('mate', ...)` | CHESS FACT | [A1m] PROPOSED: "goal: mate (the rule's terminal)" (GOAL; reads movement, attack) | cases 100/100, DECISION 0 |
| agenda.py:582 | `R.STAKE_MATE  in: q = R.new_question("mate", R.STAKE_MATE, dict(sym=placed.symbol(), at=m.to_square, frm=m.from_squar...` | CHESS FACT | [A2] PROPOSED: "stake: the terminal (mate = 100)" (VALUATION; reads movement, attack) | cases 97/97, DECISION 0 |
| agenda.py:584 | `m.promotion ; more` | LOGIC | -- | -- |
| agenda.py:587 | `sorted(... key=lambda t: t[:3])` | CHESS FACT | [D22] PROPOSED: "heuristic: the forcing order (checks, captures, threats; recaptures first)" (HEURISTIC; reads attack, material) | no printed signature: 6 rows |
| agenda.py:591 | `hang` | LOGIC | -- | -- |
| agenda.py:592 | `max(... key=lambda s: (W.base.worth(b.piece_at(s)), -s))` | CHESS FACT | [A4] PROPOSED: "valuation: material (unit worth)" (VALUATION; reads material) | cases 82/85, DECISION 0 |
| agenda.py:593 | `new_question('save', ...)` | CHESS FACT | [A1s] PROPOSED: "goal: save material (don't lose)" (GOAL; reads material, attack) (beside existing: in check) | cases 78/83, DECISION 0 |
| agenda.py:596 | `b.piece_at(s) ; b.piece_at(s).color != me` | RULE | -- | -- |
| agenda.py:596 | `chess.SQUARES  in: theirs = [s for s in chess.SQUARES if b.piece_at(s) and b.piece_at(s).color != me and b.piece_at(s...` | NAME | -- | -- |
| agenda.py:596 | `b.piece_at(s).piece_type != chess.KING` | CHESS FACT | [C1] PROPOSED: "names: king = the royal unit (Game.royal); excluded from the valuation" (VALUATION; reads material) | no printed signature: 8 rows |
| agenda.py:597 | `sorted(... key=lambda s: (-W.base.worth(b.piece_at(s)), s))` | CHESS FACT | [A4] PROPOSED: "valuation: material (unit worth)" (VALUATION; reads material) | cases 82/85, DECISION 0 |
| agenda.py:598 | `new_question('win', ...)` | CHESS FACT | [A1w] PROPOSED: "goal: win material" (GOAL; reads material, attack) | cases 83/86, DECISION 0 |
| agenda.py:608 | `TB.TABLE ; W.table_covered(b)` | LOGIC | -- | -- |
| agenda.py:608 | `b.turn != self.me` | RULE | -- | -- |
| agenda.py:611 | `tc is None` | LOGIC | -- | -- |
| agenda.py:615 | `new_question('win', ...)` | CHESS FACT | [D20] existing: syzygy | cases 1/1, DECISION 0 |
| agenda.py:623 | `self._better(tc["value"], 0, claim)` | LOGIC | -- | -- |
| agenda.py:634 | `TECHNIQUE ; self._root ; self.ctx is None ; self.ctx.tier != 0 ; getattr(self, "_tq", None) is not None` | LOGIC | -- | -- |
| agenda.py:634 | `self.board.turn != self.me` | RULE | -- | -- |
| agenda.py:638 | `ents[:1] (one technique per class: the first by name)` | HEURISTIC | -- | -- |
| agenda.py:641 | `TQ.MN  in: m = TQ.MN.measure_of(e["measure"]["quantity"])` | LOGIC | -- | -- |
| agenda.py:643 | `new_question('technique', ...)` | CHESS FACT | [A1t] existing: mined technique KRvK; no measure-dropping move | cases 0/0, DECISION 0 |
| agenda.py:646 | `got["carried"]` | LOGIC | -- | -- |
| agenda.py:649 | `got["routes"]` | LOGIC | -- | -- |
| agenda.py:657 | `got["kind"] == "floor"` | LOGIC | -- | -- |
| agenda.py:658 | `worst is not None ; worst >= 0 ; j == 1` | LOGIC | -- | -- |
| agenda.py:659 | `len(got['routes']) > 8 ; 8  in: claim = Claim("technique", san, 0, LS.name(name), f"{san} ({why}; routes {', '.join(r...` | LOGIC | -- | -- |
| agenda.py:659 | `Claim('technique', ...)` | CHESS FACT | [A1t] existing: mined technique KRvK; no measure-dropping move | cases 0/0, DECISION 0 |
| agenda.py:660 | `Claim(... goal='technique')` | CHESS FACT | [A1t] existing: mined technique KRvK; no measure-dropping move | cases 0/0, DECISION 0 |
| agenda.py:666 | `len(got["routes"]) > 8 ; 8  in: self.lines.append(f" -> CLAIM technique {san}, {why} (lemma {name}, proved=False; {le...` | LOGIC | -- | -- |
| agenda.py:670 | `self.min_stake` | CHESS FACT | [A7] PROPOSED: "goal: the need (what a claim must bring), propagated into the chase" (GOAL; reads material) | no printed signature: 15 rows |
| agenda.py:673 | `t is None` | LOGIC | -- | -- |
| agenda.py:673 | `b.is_attacked_by(self.me, t)` | RULE | -- | -- |
| agenda.py:675 | `s != t ; [W.base.worth(p) for s, p in b.piece_map().items() if p.color != self.me and p.piece_type != chess.KING and ...` | LOGIC | -- | -- |
| agenda.py:675 | `p.color != self.me` | RULE | -- | -- |
| agenda.py:675 | `p.piece_type != chess.KING` | CHESS FACT | [C1] PROPOSED: "names: king = the royal unit (Game.royal); excluded from the valuation" (VALUATION; reads material) | no printed signature: 8 rows |
| agenda.py:681 | `self.chase_depth ; self.ctx is None` | LOGIC | -- | -- |
| agenda.py:681 | `b.is_check()` | RULE | -- | -- |
| agenda.py:681 | `self.min_stake` | CHESS FACT | [A7] PROPOSED: "goal: the need (what a claim must bring), propagated into the chase" (GOAL; reads material) | no printed signature: 15 rows |
| agenda.py:689 | `new_question('fork', ...)` | CHESS FACT | [A1f] existing: fork; fork: two threats, one move (Ideas.ForkSquare... | cases 5/5, DECISION 0 |
| agenda.py:690 | `f["check"]` | LOGIC | -- | -- |
| agenda.py:699 | `ksq is None` | LOGIC | -- | -- |
| agenda.py:701 | `self.ctx is not None` | LOGIC | -- | -- |
| agenda.py:702 | `b.occupied_co[self.me] & ~b.kings & ~b.pawns (the reach roots' pieces: not the king, not a pawn)` | CHESS FACT | [D1e] PROPOSED: "goal: mate at distance two (the reach defect)" (GOAL; reads movement, attack, tempo) | cases 23/23, DECISION 0 |
| agenda.py:705 | `fs is not None` | LOGIC | -- | -- |
| agenda.py:706 | `f is not None` | LOGIC | -- | -- |
| agenda.py:712 | `fs is not None` | LOGIC | -- | -- |
| agenda.py:715 | `R.STAKE_MATE  in: fs.add("roots", key, self.me, f"reach roots of the {pc.symbol().upper()} on {GAME.loc_name(frm)}, t...` | CHESS FACT | [A2] PROPOSED: "stake: the terminal (mate = 100)" (VALUATION; reads movement, attack) | cases 97/97, DECISION 0 |
| agenda.py:717 | `new_question('mate', ...)` | CHESS FACT | [A1m] PROPOSED: "goal: mate (the rule's terminal)" (GOAL; reads movement, attack) | cases 100/100, DECISION 0 |
| agenda.py:717 | `R.STAKE_MATE  in: q = R.new_question("mate", R.STAKE_MATE, dict(sym=sym, at=s, frm=frm_, king=ksq, more=[], move=None...` | CHESS FACT | [A2] PROPOSED: "stake: the terminal (mate = 100)" (VALUATION; reads movement, attack) | cases 97/97, DECISION 0 |
| agenda.py:719 | `Defect('reach', ...)` | CHESS FACT | [D1e] PROPOSED: "goal: mate at distance two (the reach defect)" (GOAL; reads movement, attack, tempo) | cases 23/23, DECISION 0 |
| agenda.py:722 | `any(q.pattern.get("reach") for q in self.all)` | LOGIC | -- | -- |
| agenda.py:732 | `b.king(me) is not None` | RULE | -- | -- |
| agenda.py:737 | `chess.SQUARES  in: for s in chess.SQUARES: if s == frm or not _aligned(pc.piece_type, s, ksq): continue occ.add(s) oc...` | NAME | -- | -- |
| agenda.py:738 | `s == frm` | LOGIC | -- | -- |
| agenda.py:738 | `_aligned(pc.piece_type, s, ksq)` | RULE | -- | -- |
| agenda.py:743 | `o is not None` | LOGIC | -- | -- |
| agenda.py:743 | `o.color == me` | RULE | -- | -- |
| agenda.py:746 | `ksq not in cf.attacks(s)` | RULE | -- | -- |
| agenda.py:748 | `table is None` | LOGIC | -- | -- |
| agenda.py:752 | `steps` | LOGIC | -- | -- |
| agenda.py:754 | `new_question('mate', ...)` | CHESS FACT | [A1m] PROPOSED: "goal: mate (the rule's terminal)" (GOAL; reads movement, attack) | cases 100/100, DECISION 0 |
| agenda.py:754 | `R.STAKE_MATE  in: q = R.new_question("mate", R.STAKE_MATE, dict(sym=pc.symbol(), at=s, frm=frm, king=ksq, more=[], mo...` | CHESS FACT | [A2] PROPOSED: "stake: the terminal (mate = 100)" (VALUATION; reads movement, attack) | cases 97/97, DECISION 0 |
| agenda.py:759 | `q.defects` | CHESS FACT | [D1e] PROPOSED: "goal: mate at distance two (the reach defect)" (GOAL; reads movement, attack, tempo) | cases 23/23, DECISION 0 |
| agenda.py:762 | `table is not None` | LOGIC | -- | -- |
| agenda.py:772 | `0 < n <= R.PROMOTE_MAX_PUSHES` | CHESS FACT | [A1p] PROPOSED: "goal: promote (the rule)" (GOAL; reads movement, tempo) | cases 48/49, DECISION 0 |
| agenda.py:774 | `PASSER_ROOTS ; R.is_passer(b, s) ; R.promote_defects_cached(b, s, priced=False)` | LOGIC | -- | -- |
| agenda.py:774 | `n > 0` | CHESS FACT | [A1p] PROPOSED: "goal: promote (the rule)" (GOAL; reads movement, tempo) | cases 48/49, DECISION 0 |
| agenda.py:777 | `n > 1` | LOGIC | -- | -- |
| agenda.py:777 | `new_question('promote', ...)` | CHESS FACT | [A1p] PROPOSED: "goal: promote (the rule)" (GOAL; reads movement, tempo) | cases 48/49, DECISION 0 |
| agenda.py:777 | `R.STAKE_PROMOTE  in: self._push(R.new_question("promote", R.STAKE_PROMOTE, dict(pawn=s, pushes=n), text=f"promote our...` | CHESS FACT | [A3] PROPOSED: "stake: promotion (queen less pawn = 8)" (VALUATION; reads material, movement) | cases 35/35, DECISION 0 |
| agenda.py:778 | `near` | LOGIC | -- | -- |
| agenda.py:779 | `R.PROMOTE_MAX_PUSHES  in: self.notes.append(f"promote roots only for pawns with pushes <= {R.PROMOTE_MAX_PUSHES} (the...` | CHESS FACT (text only) | [A1p] PROPOSED: "goal: promote (the rule)" (GOAL; reads movement, tempo) | cases 48/49, DECISION 0 |
| agenda.py:790 | `b.is_check()` | RULE | -- | -- |
| agenda.py:795 | `q.parent is None` | LOGIC | -- | -- |
| agenda.py:798 | `self._arr_matches(q, a)` | LOGIC | -- | -- |
| agenda.py:799 | `match` | LOGIC | -- | -- |
| agenda.py:800 | `a.placement is not None ; q.pattern.get("frm") == a.placement[0]` | LOGIC | -- | -- |
| agenda.py:802 | `a.verdict != "reachable"` | LOGIC | -- | -- |
| agenda.py:804 | `match` | LOGIC | -- | -- |
| agenda.py:806 | `q is None` | LOGIC | -- | -- |
| agenda.py:815 | `"arrangement" in q.pattern ; same_unit ; a.placement is not None ; q.pattern.get("frm") != a.placement[0]` | LOGIC | -- | -- |
| agenda.py:820 | `R.SKELETON_LINES  in: self.lines += [W.arrangement_line(a, texts.get(id(a))) for a in arrs[:R.SKELETON_LINES]]` | LOGIC | -- | -- |
| agenda.py:821 | `a.verdict == v` | LOGIC | -- | -- |
| agenda.py:822 | `1000  in: self.skeleton_stats = dict(targets=rep["targets"], arrangements=len(arrs), ms=(time.perf_counter() - t0) * ...` | LOGIC | -- | -- |
| agenda.py:828 | `s is None` | LOGIC | -- | -- |
| agenda.py:837 | `a.goal == "mate"` | CHESS FACT | [A1m] PROPOSED: "goal: mate (the rule's terminal)" (GOAL; reads movement, attack) | cases 100/100, DECISION 0 |
| agenda.py:838 | `p.get("at") == a.about` | LOGIC | -- | -- |
| agenda.py:838 | `q.goal == "mate"` | CHESS FACT | [A1m] PROPOSED: "goal: mate (the rule's terminal)" (GOAL; reads movement, attack) | cases 100/100, DECISION 0 |
| agenda.py:839 | `a.goal == "win"` | CHESS FACT | [A1w] PROPOSED: "goal: win material" (GOAL; reads material, attack) | cases 83/86, DECISION 0 |
| agenda.py:840 | `p.get("target") == a.about ; p.get("checker")` | LOGIC | -- | -- |
| agenda.py:840 | `q.goal == "win"` | CHESS FACT | [A1w] PROPOSED: "goal: win material" (GOAL; reads material, attack) | cases 83/86, DECISION 0 |
| agenda.py:840 | `"table" not in p` | CHESS FACT | [D20] existing: syzygy | cases 1/1, DECISION 0 |
| agenda.py:841 | `p.get("pawn") == a.about` | LOGIC | -- | -- |
| agenda.py:841 | `q.goal == "promote"` | CHESS FACT | [A1p] PROPOSED: "goal: promote (the rule)" (GOAL; reads movement, tempo) | cases 48/49, DECISION 0 |
| agenda.py:849 | `steps` | LOGIC | -- | -- |
| agenda.py:849 | `a.goal == "mate"` | CHESS FACT | [A1m] PROPOSED: "goal: mate (the rule's terminal)" (GOAL; reads movement, attack) | cases 100/100, DECISION 0 |
| agenda.py:854 | `new_question('mate', ...)` | CHESS FACT | [A1m] PROPOSED: "goal: mate (the rule's terminal)" (GOAL; reads movement, attack) | cases 100/100, DECISION 0 |
| agenda.py:854 | `R.STAKE_MATE  in: q = R.new_question("mate", R.STAKE_MATE, dict(sym=pc.symbol(), at=s, frm=u, king=ksq, more=[], move...` | CHESS FACT | [A2] PROPOSED: "stake: the terminal (mate = 100)" (VALUATION; reads movement, attack) | cases 97/97, DECISION 0 |
| agenda.py:855 | `Defect('reach', ...)` | CHESS FACT | [D1e] PROPOSED: "goal: mate at distance two (the reach defect)" (GOAL; reads movement, attack, tempo) | cases 23/23, DECISION 0 |
| agenda.py:857 | `a.goal == "promote"` | CHESS FACT | [A1p] PROPOSED: "goal: promote (the rule)" (GOAL; reads movement, tempo) | cases 48/49, DECISION 0 |
| agenda.py:860 | `n > 1` | LOGIC | -- | -- |
| agenda.py:860 | `new_question('promote', ...)` | CHESS FACT | [A1p] PROPOSED: "goal: promote (the rule)" (GOAL; reads movement, tempo) | cases 48/49, DECISION 0 |
| agenda.py:860 | `R.STAKE_PROMOTE  in: q = R.new_question("promote", R.STAKE_PROMOTE, dict(pawn=pw, pushes=n), text=f"promote our P on ...` | CHESS FACT | [A3] PROPOSED: "stake: promotion (queen less pawn = 8)" (VALUATION; reads material, movement) | cases 35/35, DECISION 0 |
| agenda.py:863 | `a.goal == "mate"` | CHESS FACT | [A1m] PROPOSED: "goal: mate (the rule's terminal)" (GOAL; reads movement, attack) | cases 100/100, DECISION 0 |
| agenda.py:865 | `q in self.all` | LOGIC | -- | -- |
| agenda.py:871 | `SKELETON_GOALS ; a is None ; self.chase_depth ; self._root ; q.parent is not None` | LOGIC | -- | -- |
| agenda.py:874 | `quiet` | LOGIC | -- | -- |
| agenda.py:876 | `f is not None` | LOGIC | -- | -- |
| agenda.py:886 | `board.turn` | NAME | -- | -- |
| agenda.py:890 | `self.ctx is not None ; units is None` | LOGIC | -- | -- |
| agenda.py:893 | `q.status == "gap"` | LOGIC | -- | -- |
| agenda.py:898 | `route in self._tv` | LOGIC | -- | -- |
| agenda.py:902 | `any(True for _ in c.legal_moves)` | RULE | -- | -- |
| agenda.py:906 | `ag.state.claims ; max(... key=lambda x: x.value)` | LOGIC | -- | -- |
| agenda.py:914 | `b.is_check()` | RULE | -- | -- |
| agenda.py:916 | `self.their is not None` | LOGIC | -- | -- |
| agenda.py:917 | `self.local is not None ; self.local["full"]` | LOGIC | -- | -- |
| agenda.py:919 | `given ; local` | LOGIC | -- | -- |
| agenda.py:920 | `given ; local` | LOGIC | -- | -- |
| agenda.py:921 | `ag.state.claims ; max(... key=lambda x: x.value)` | LOGIC | -- | -- |
| agenda.py:924 | `cl.move` | LOGIC | -- | -- |
| agenda.py:925 | `m is None ; cl.move in seen ; given` | LOGIC | -- | -- |
| agenda.py:925 | `b.piece_at(m.to_square)` | RULE | -- | -- |
| agenda.py:925 | `m.to_square in hang` | CHESS FACT | [D14] existing: threat: ... binds after a pass | cases 9/9, DECISION 0 |
| agenda.py:928 | `cl.kind == "promote"` | CHESS FACT | [D14] existing: threat: ... binds after a pass | cases 9/9, DECISION 0 |
| agenda.py:932 | `ppath` | LOGIC | -- | -- |
| agenda.py:935 | `new_question('save', ...)` | CHESS FACT | [A1s] PROPOSED: "goal: save material (don't lose)" (GOAL; reads material, attack) (beside existing: in check) | cases 78/83, DECISION 0 |
| agenda.py:941 | `cl.value >= BD.MATE_VALUE ; " and ".join(f"our {b.piece_at(t).symbol().upper()} on {GAME.loc_name(t)}" for t in tgt) ...` | CHESS FACT | [A2] PROPOSED: "stake: the terminal (mate = 100)" (VALUATION; reads movement, attack) | cases 97/97, DECISION 0 |
| agenda.py:942 | `cl.qid is not None ; x.id == cl.qid` | LOGIC | -- | -- |
| agenda.py:943 | `their_q is not None ; their_q.claimed is not None ; 2  in: cl_reads = their_q.claimed[2] if their_q is not None and t...` | LOGIC | -- | -- |
| agenda.py:944 | `new_question('save', ...)` | CHESS FACT | [A1s] PROPOSED: "goal: save material (don't lose)" (GOAL; reads material, attack) (beside existing: in check) | cases 78/83, DECISION 0 |
| agenda.py:951 | `cl.value >= BD.MATE_VALUE` | CHESS FACT | [A2] PROPOSED: "stake: the terminal (mate = 100)" (VALUATION; reads movement, attack) | cases 97/97, DECISION 0 |
| agenda.py:954 | `b.is_capture(m)` | CHESS FACT | [D14] existing: threat: ... binds after a pass | cases 9/9, DECISION 0 |
| agenda.py:958 | `b.piece_at(t) ; b.piece_at(t).color == self.me` | RULE | -- | -- |
| agenda.py:958 | `b.piece_at(t).piece_type != chess.KING` | CHESS FACT | [C1] PROPOSED: "names: king = the royal unit (Game.royal); excluded from the valuation" (VALUATION; reads material) | no printed signature: 8 rows |
| agenda.py:958 | `t not in before` | CHESS FACT | [D14] existing: threat: ... binds after a pass | cases 9/9, DECISION 0 |
| agenda.py:959 | `sorted(... key=lambda t: (-W.base.worth(b.piece_at(t)), t))` | CHESS FACT | [A4] PROPOSED: "valuation: material (unit worth)" (VALUATION; reads material) | cases 82/85, DECISION 0 |
| agenda.py:964 | `self._root` | LOGIC | -- | -- |
| agenda.py:965 | `TIERS ; TIERS_UNDER_TIME ; self.ctx is not None ; self.ctx.seconds is not None ; self._root ; self.ctx.tier == 0` | LOGIC | -- | -- |
| agenda.py:966 | `POOL ; GP.load() is not None ; QP.load() is not None` | LOGIC | -- | -- |
| agenda.py:972 | `mark is not None` | LOGIC | -- | -- |
| agenda.py:988 | `self.local is not None` | LOGIC | -- | -- |
| agenda.py:999 | `64  in: bal = self.pos.balance(self.me) - 64` | LOGIC | -- | -- |
| agenda.py:1000 | `b.is_check()` | RULE | -- | -- |
| agenda.py:1003 | `self._pass_done` | LOGIC | -- | -- |
| agenda.py:1005 | `ag.state.claims ; max(... key=lambda x: x.value)` | LOGIC | -- | -- |
| agenda.py:1007 | `cl is not None` | LOGIC | -- | -- |
| agenda.py:1007 | `BD.MATE_VALUE  in: threat = min(cl.value, BD.MATE_VALUE) if cl is not None else 0` | CHESS FACT | [A2] PROPOSED: "stake: the terminal (mate = 100)" (VALUATION; reads movement, attack) | cases 97/97, DECISION 0 |
| agenda.py:1009 | `DEEP_PASS ; cl is None` | LOGIC | -- | -- |
| agenda.py:1009 | `b.is_check()` | RULE | -- | -- |
| agenda.py:1009 | `self._piece_attacked()` | CHESS FACT | [D18] existing: the deep pass | cases 0/0, DECISION 0 |
| agenda.py:1011 | `dv > 0` | CHESS FACT | [D18] existing: the deep pass | cases 0/0, DECISION 0 |
| agenda.py:1013 | `s0 = bal - threat (the status: balance less their pass threat)` | CHESS FACT | [A5] PROPOSED: "goal: the ladder -- don't lose, then win (status, line, keeps)" (GOAL; reads material, tempo) | cases 22/18, DECISION 35 |
| agenda.py:1014 | `s0 >= 1 ; s0 == 0` | CHESS FACT | [A5] PROPOSED: "goal: the ladder -- don't lose, then win (status, line, keeps)" (GOAL; reads material, tempo) | cases 22/18, DECISION 35 |
| agenda.py:1019 | `best is None` | LOGIC | -- | -- |
| agenda.py:1020 | `best is None` | LOGIC | -- | -- |
| agenda.py:1022 | `st["balance"] + best[0] >= st["line"]` | CHESS FACT | [A5] PROPOSED: "goal: the ladder -- don't lose, then win (status, line, keeps)" (GOAL; reads material, tempo) | cases 22/18, DECISION 35 |
| agenda.py:1026 | `best is None` | LOGIC | -- | -- |
| agenda.py:1027 | `best is not None ; st is None` | LOGIC | -- | -- |
| agenda.py:1027 | `best[0] >= BD.MATE_VALUE` | CHESS FACT | [A2] PROPOSED: "stake: the terminal (mate = 100)" (VALUATION; reads movement, attack) | cases 97/97, DECISION 0 |
| agenda.py:1028 | `2  in: return f"STATUS: tier {tier}'s best claim {best[2].what} is a mate: the status is kept"` | LOGIC | -- | -- |
| agenda.py:1029 | `st ; self.status()` | LOGIC | -- | -- |
| agenda.py:1031 | `tier == 0 ; st.get('deep')` | LOGIC | -- | -- |
| agenda.py:1031 | `st['line']` | CHESS FACT (text only) | [A5] PROPOSED: "goal: the ladder -- don't lose, then win (status, line, keeps)" (GOAL; reads material, tempo) | cases 22/18, DECISION 35 |
| agenda.py:1032 | `v is None ; 2  in: claim = "no claim" if v is None else f"best claim {best[2].what} {v:+d} gives {st['balance'] + v:+d}"` | LOGIC | -- | -- |
| agenda.py:1033 | `opened ; kept` | LOGIC | -- | -- |
| agenda.py:1035 | `Agenda.TIER_WHY = ("today's moves", "quiet threats worth the stake followed below the root", "mate-in-two threats", "...` | LOGIC | -- | -- |
| agenda.py:1046 | `self.ctx.need = line - balance` | CHESS FACT | [A5] PROPOSED: "goal: the ladder -- don't lose, then win (status, line, keeps)" (GOAL; reads material, tempo) | cases 22/18, DECISION 35 |
| agenda.py:1047 | `self.status_info["word"] == "losing"` | CHESS FACT | [A5] PROPOSED: "goal: the ladder -- don't lose, then win (status, line, keeps)" (GOAL; reads material, tempo) | cases 22/18, DECISION 35 |
| agenda.py:1050 | `total is not None` | LOGIC | -- | -- |
| agenda.py:1051 | `QP.load() ; {}` | LOGIC | -- | -- |
| agenda.py:1052 | `GP.load() is not None` | LOGIC | -- | -- |
| agenda.py:1057 | `2  in: self.tier_info.append(dict(tier="pool", secs=round(secs, 2), asked=self.asked + self.ctx.asked, kept=kept, hit...` | LOGIC | -- | -- |
| agenda.py:1058 | `self.best is None ; 2  in: self.tier_info.append(dict(tier="pool", secs=round(secs, 2), asked=self.asked + self.ctx.a...` | LOGIC | -- | -- |
| agenda.py:1060 | `total is not None ; kept ; self.ctx.abort` | LOGIC | -- | -- |
| agenda.py:1062 | `d["status"] == "gap"` | LOGIC | -- | -- |
| agenda.py:1063 | `open_gaps` | LOGIC | -- | -- |
| agenda.py:1063 | `left <= 0.05` | HEURISTIC | -- | -- |
| agenda.py:1072 | `len(open_gaps) > 5 ; 5  in: self.lines.append(f"POOL pass {p}: {len(open_gaps)} root route(s) still open ({', '.join(...` | LOGIC | -- | -- |
| agenda.py:1077 | `self._fold_tier(nxt, f"pool {p}", owner)` | LOGIC | -- | -- |
| agenda.py:1080 | `2  in: self.tier_info.append(dict(tier=f"pool {p}", secs=round(psecs, 2), asked=nxt.asked + nxt.ctx.asked, kept=kept,...` | LOGIC | -- | -- |
| agenda.py:1081 | `nxt.best is None ; 2  in: self.tier_info.append(dict(tier=f"pool {p}", secs=round(psecs, 2), asked=nxt.asked + nxt.ct...` | LOGIC | -- | -- |
| agenda.py:1083 | `after == before ; nxt.best is None` | LOGIC | -- | -- |
| agenda.py:1087 | `self.best is not None` | LOGIC | -- | -- |
| agenda.py:1089 | `total is not None` | LOGIC | -- | -- |
| agenda.py:1090 | `self.best is not None ; kept ; 2  in: self.lines.append(f"POOL: ended at {time.monotonic() - t0:.1f} s" + (f" of {tot...` | LOGIC | -- | -- |
| agenda.py:1102 | `DEFER` | LOGIC | -- | -- |
| agenda.py:1104 | `self.ctx.need = line - balance` | CHESS FACT | [A5] PROPOSED: "goal: the ladder -- don't lose, then win (status, line, keeps)" (GOAL; reads material, tempo) | cases 22/18, DECISION 35 |
| agenda.py:1107 | `x is not None` | LOGIC | -- | -- |
| agenda.py:1108 | `cands` | LOGIC | -- | -- |
| agenda.py:1110 | `max(... key=lambda x: (x[0], x[1]))` | LOGIC | -- | -- |
| agenda.py:1111 | `top[0] >= BD.MATE_VALUE` | CHESS FACT | [A2] PROPOSED: "stake: the terminal (mate = 100)" (VALUATION; reads movement, attack) | cases 97/97, DECISION 0 |
| agenda.py:1113 | `self.status_info is not None` | LOGIC | -- | -- |
| agenda.py:1114 | `self.keeps(st, top)[0]` | CHESS FACT | [A5] PROPOSED: "goal: the ladder -- don't lose, then win (status, line, keeps)" (GOAL; reads material, tempo) | cases 22/18, DECISION 35 |
| agenda.py:1116 | `RAISE_TO_WIN` | LOGIC | -- | -- |
| agenda.py:1116 | `st["line"] == 0 ; st["balance"] + top[0] < 1` | CHESS FACT | [A5] PROPOSED: "goal: the ladder -- don't lose, then win (status, line, keeps)" (GOAL; reads material, tempo) | cases 22/18, DECISION 35 |
| agenda.py:1118 | `True` | LOGIC | -- | -- |
| agenda.py:1124 | `q.text not in seen_d` | LOGIC | -- | -- |
| agenda.py:1125 | `ag is not self` | LOGIC | -- | -- |
| agenda.py:1126 | `self._fold_tier(ag, k, owner)` | LOGIC | -- | -- |
| agenda.py:1129 | `overall is not None` | LOGIC | -- | -- |
| agenda.py:1129 | `overall[0] >= BD.MATE_VALUE` | CHESS FACT | [A2] PROPOSED: "stake: the terminal (mate = 100)" (VALUATION; reads movement, attack) | cases 97/97, DECISION 0 |
| agenda.py:1130 | `self.status_info is None ; mate` | LOGIC | -- | -- |
| agenda.py:1132 | `mate` | LOGIC | -- | -- |
| agenda.py:1132 | `self.keeps(self.status_info, overall)[0]` | CHESS FACT | [A5] PROPOSED: "goal: the ladder -- don't lose, then win (status, line, keeps)" (GOAL; reads material, tempo) | cases 22/18, DECISION 35 |
| agenda.py:1133 | `2  in: self.tier_info.append(dict(tier=k, secs=round(secs, 2), asked=ag.asked + ag.ctx.asked, kept=kept, hit=bool(ag....` | LOGIC | -- | -- |
| agenda.py:1134 | `ag.best is None ; 2  in: self.tier_info.append(dict(tier=k, secs=round(secs, 2), asked=ag.asked + ag.ctx.asked, kept=...` | LOGIC | -- | -- |
| agenda.py:1136 | `kept ; k > 0` | LOGIC | -- | -- |
| agenda.py:1137 | `mate ; 2  in: self.lines.append(f"tier {k} keeps the status: {overall[2].what}" + ("" if mate else f" ({st['balance']...` | LOGIC | -- | -- |
| agenda.py:1138 | `total is None` | LOGIC | -- | -- |
| agenda.py:1140 | `kept ; RAISE_TO_WIN ; mate ; overall is not None` | LOGIC | -- | -- |
| agenda.py:1140 | `st["line"] == 0 ; st["balance"] + overall[0] < 1` | CHESS FACT | [A5] PROPOSED: "goal: the ladder -- don't lose, then win (status, line, keeps)" (GOAL; reads material, tempo) | cases 22/18, DECISION 35 |
| agenda.py:1141 | `STOP_AFFORD ; widens ; left is not None ; k < TIER_COUNT - 1 ; self.ctx.abort` | LOGIC | -- | -- |
| agenda.py:1141 | `left > 0.05` | HEURISTIC | -- | -- |
| agenda.py:1143 | `afford is not None ; afford[0]` | LOGIC | -- | -- |
| agenda.py:1146 | `RAISE_TO_WIN ; afford is None ; afford[0] ; kept ; mate ; left is not None ; k < TIER_COUNT - 1 ; self.ctx.abort` | LOGIC | -- | -- |
| agenda.py:1146 | `left > 0.05` | HEURISTIC | -- | -- |
| agenda.py:1147 | `overall is not None` | LOGIC | -- | -- |
| agenda.py:1147 | `st["line"] == 0 ; st["balance"] + overall[0] < 1` | CHESS FACT | [A5] PROPOSED: "goal: the ladder -- don't lose, then win (status, line, keeps)" (GOAL; reads material, tempo) | cases 22/18, DECISION 35 |
| agenda.py:1150 | `2  in: self.lines.append(f"STATUS kept at 'not losing' ({overall[2].what}: {st['balance'] + overall[0]:+d}) with {lef...` | LOGIC | -- | -- |
| agenda.py:1151 | `dict(st, line=1, raised=True) (the line rises to 'stay ahead')` | CHESS FACT | [A5] PROPOSED: "goal: the ladder -- don't lose, then win (status, line, keeps)" (GOAL; reads material, tempo) | cases 22/18, DECISION 35 |
| agenda.py:1153 | `kept ; k >= TIER_COUNT - 1 ; self.ctx.abort` | LOGIC | -- | -- |
| agenda.py:1154 | `stop ; left is not None` | LOGIC | -- | -- |
| agenda.py:1154 | `left <= 0.05` | HEURISTIC | -- | -- |
| agenda.py:1156 | `stop ; afford is not None ; afford[0]` | LOGIC | -- | -- |
| agenda.py:1158 | `k == 0 ; kept` | LOGIC | -- | -- |
| agenda.py:1159 | `stop ; kept ; afford is not None ; afford[0] ; left is not None ; k < TIER_COUNT - 1` | LOGIC | -- | -- |
| agenda.py:1160 | `stop ; kept` | LOGIC | -- | -- |
| agenda.py:1161 | `k == 0` | LOGIC | -- | -- |
| agenda.py:1164 | `why ; ag.best is None ; 2  in: self.lines.append(f"tier {k}'s " + ("no claim" if ag.best is None else f"best claim {a...` | LOGIC | -- | -- |
| agenda.py:1165 | `stop` | LOGIC | -- | -- |
| agenda.py:1171 | `kept ; self.ctx.abort ; afford is None ; afford[0]` | LOGIC | -- | -- |
| agenda.py:1173 | `RESUME ; kept ; self.ctx.abort ; total is not None` | LOGIC | -- | -- |
| agenda.py:1175 | `DEFER ; kept ; self.ctx.abort ; total is not None ; deferred_all` | LOGIC | -- | -- |
| agenda.py:1178 | `len(deferred_all) > 4 ; 4  in: names = ", ".join(q.text for q in deferred_all[:4]) + (", ..." if len(deferred_all) > ...` | LOGIC | -- | -- |
| agenda.py:1179 | `left > 0.05` | HEURISTIC | -- | -- |
| agenda.py:1188 | `self._fold_tier(da, ag.ctx.tier, owner)` | LOGIC | -- | -- |
| agenda.py:1190 | `2  in: self.tier_info.append(dict(tier="deferred", secs=round(secs, 2), asked=da.asked + da.ctx.asked, kept=False, hi...` | LOGIC | -- | -- |
| agenda.py:1191 | `da.best is None ; 2  in: self.tier_info.append(dict(tier="deferred", secs=round(secs, 2), asked=da.asked + da.ctx.ask...` | LOGIC | -- | -- |
| agenda.py:1196 | `self.best is not None ; 2  in: self.lines.append(f"TIERS: ran {len(self.tier_info)} (0..{len(self.tier_info) - 1}); t...` | LOGIC | -- | -- |
| agenda.py:1202 | `old is None ; old["status"] != "killed" ; d["status"] == "killed" ; len(d["gains"]) >= len(old["gains"])` | LOGIC | -- | -- |
| agenda.py:1208 | `ag.best_est is not None ; self.best_est is None ; (ag.best_est[0], ag.best_est[2].confidence) > (self.best_est[0], se...` | LOGIC | -- | -- |
| agenda.py:1209 | `2  in: self.best_est = (ag.best_est[0], 0, dataclasses.replace(ag.best_est[2], qid=-1))` | LOGIC | -- | -- |
| agenda.py:1213 | `ag.best is not None ; id(ag.best[2]) not in mark` | LOGIC | -- | -- |
| agenda.py:1214 | `2  in: mark[id(ag.best[2])] = dataclasses.replace(ag.best[2], qid=-1)` | LOGIC | -- | -- |
| agenda.py:1217 | `ag.best is not None ; owner.best is None ; _wider(owner, ag)` | LOGIC | -- | -- |
| agenda.py:1218 | `2  in: self.best = (ag.best[0], ag.best[1], mark[id(ag.best[2])])` | LOGIC | -- | -- |
| agenda.py:1229 | `DEFER ; st is not None` | LOGIC | -- | -- |
| agenda.py:1229 | `nxt.ctx.need = st['line'] - st['balance']` | CHESS FACT | [A5] PROPOSED: "goal: the ladder -- don't lose, then win (status, line, keeps)" (GOAL; reads material, tempo) | cases 22/18, DECISION 35 |
| agenda.py:1232 | `CARRY` | LOGIC | -- | -- |
| agenda.py:1237 | `CARRY` | LOGIC | -- | -- |
| agenda.py:1241 | `old is None ; (d["net"], d["second"]) > (old["net"], old["second"])` | LOGIC | -- | -- |
| agenda.py:1243 | `old is not None` | LOGIC | -- | -- |
| agenda.py:1244 | `old["dep"] ; d["dep"]` | LOGIC | -- | -- |
| agenda.py:1249 | `kk not in proofs ; d["g"] > proofs[kk]["g"]` | LOGIC | -- | -- |
| agenda.py:1252 | `RESUME` | LOGIC | -- | -- |
| agenda.py:1256 | `kk in seen` | LOGIC | -- | -- |
| agenda.py:1260 | `ok` | LOGIC | -- | -- |
| agenda.py:1261 | `len(it["line"]) ; 1` | LOGIC | -- | -- |
| agenda.py:1264 | `line not in self.lines` | LOGIC | -- | -- |
| agenda.py:1276 | `True` | LOGIC | -- | -- |
| agenda.py:1280 | `(it["key"], it["uci"]) not in seen` | LOGIC | -- | -- |
| agenda.py:1283 | `items` | LOGIC | -- | -- |
| agenda.py:1285 | `left <= 0.05` | HEURISTIC | -- | -- |
| agenda.py:1292 | `ok` | LOGIC | -- | -- |
| agenda.py:1296 | `spn` | LOGIC | -- | -- |
| agenda.py:1297 | `go` | LOGIC | -- | -- |
| agenda.py:1299 | `sort(... key=lambda t: (t[0], t[2]["uci"]))` | HEURISTIC | -- | -- |
| agenda.py:1301 | `p > 1` | LOGIC | -- | -- |
| agenda.py:1307 | `sum(1 for _ in it["line"]) ; 1` | LOGIC | -- | -- |
| agenda.py:1308 | `spn` | LOGIC | -- | -- |
| agenda.py:1314 | `self._fold_tier(ra, f"resume {p}", owner)` | LOGIC | -- | -- |
| agenda.py:1318 | `overall is not None` | LOGIC | -- | -- |
| agenda.py:1318 | `overall[0] >= BD.MATE_VALUE` | CHESS FACT | [A2] PROPOSED: "stake: the terminal (mate = 100)" (VALUATION; reads movement, attack) | cases 97/97, DECISION 0 |
| agenda.py:1318 | `self.keeps(st, overall)[0]` | CHESS FACT | [A5] PROPOSED: "goal: the ladder -- don't lose, then win (status, line, keeps)" (GOAL; reads material, tempo) | cases 22/18, DECISION 35 |
| agenda.py:1319 | `2  in: self.tier_info.append(dict(tier=f"resume {p}", secs=round(secs, 2), asked=ra.asked + ra.ctx.asked, kept=kept, ...` | LOGIC | -- | -- |
| agenda.py:1320 | `ra.best is None ; 2  in: self.tier_info.append(dict(tier=f"resume {p}", secs=round(secs, 2), asked=ra.asked + ra.ctx....` | LOGIC | -- | -- |
| agenda.py:1321 | `kept ; ra.best is None ; 2  in: self.lines.append(f"RESUME pass {p}: " + ("no claim" if ra.best is None else f"best c...` | LOGIC | -- | -- |
| agenda.py:1322 | `kept ; self.ctx.abort` | LOGIC | -- | -- |
| agenda.py:1332 | `bool(m) ; "they mate us" in why ; "stalemate" in why ; "no parry" in why` | CHESS FACT | [D13] PROPOSED: "closing: a kill that proves a loss (refuted)" (CLOSING; reads material, movement) | cases 19/16, DECISION 0 |
| agenda.py:1337 | `self.chase_depth != 0 ; self.ctx is None` | LOGIC | -- | -- |
| agenda.py:1339 | `q is not None ; self._root` | LOGIC | -- | -- |
| agenda.py:1342 | `res is None` | LOGIC | -- | -- |
| agenda.py:1344 | `r.why` | LOGIC | -- | -- |
| agenda.py:1345 | `60  in: self.ctx.item_rows.append(dict(q=q.text, goal=q.goal, stake=q.stake, qid=q.id, route=r.san, uci=r.move.uci(),...` | LOGIC | -- | -- |
| agenda.py:1347 | `self.ctx.route_costs.get((self.pk, r.move.uci(), q.stake)) ; (None,)` | LOGIC | -- | -- |
| agenda.py:1348 | `res.get("killed") is None ; res.get("gap") is None ; res.get("killed")` | LOGIC | -- | -- |
| agenda.py:1353 | `res is None ; self.root_reads.get(u, {}).get("status") == "killed"` | LOGIC | -- | -- |
| agenda.py:1355 | `res.get("gains") ; []` | LOGIC | -- | -- |
| agenda.py:1356 | `res.get("killed")` | LOGIC | -- | -- |
| agenda.py:1356 | `self._refuted(res["killed"])` | CHESS FACT | [D13] PROPOSED: "closing: a kill that proves a loss (refuted)" (CLOSING; reads material, movement) | cases 19/16, DECISION 0 |
| agenda.py:1358 | `res.get("killed") ; res.get("gap")` | LOGIC | -- | -- |
| agenda.py:1359 | `res.get("gap") ; f"not refuted: {res['killed']}"` | LOGIC | -- | -- |
| agenda.py:1361 | `old is None ; len(gains) >= len(old["gains"])` | LOGIC | -- | -- |
| agenda.py:1370 | `d["status"] == "gap"` | LOGIC | -- | -- |
| agenda.py:1371 | `gaps` | LOGIC | -- | -- |
| agenda.py:1379 | `t[4] ; t[2] ; t[3] ; sort(... key=lambda t: (0 if t[4] else 1 if t[2] else 2 if t[3] else 3, t[0], t[1]))` | CHESS FACT | [D22] PROPOSED: "heuristic: the forcing order (checks, captures, threats; recaptures first)" (HEURISTIC; reads attack, material) | no printed signature: 6 rows |
| agenda.py:1381 | `d["gains"]` | LOGIC | -- | -- |
| agenda.py:1383 | `read is not None ; rest is not None ; x is not None` | LOGIC | -- | -- |
| agenda.py:1384 | `floor is not None ; floor is None` | LOGIC | -- | -- |
| agenda.py:1386 | `r[0] == top` | LOGIC | -- | -- |
| agenda.py:1387 | `12  in: best, limit = None, 1 << 12` | HEURISTIC | -- | -- |
| agenda.py:1388 | `sorted(... key=lambda t: t[1])` | HEURISTIC | -- | -- |
| agenda.py:1390 | `best is None` | LOGIC | -- | -- |
| agenda.py:1390 | `cost < best[0]` | HEURISTIC | -- | -- |
| agenda.py:1393 | `x["status"] == "killed" ; 3  in: refuted = [x["san"] for x in self.root_reads.values() if x["status"] == "killed"][:3]` | LOGIC | -- | -- |
| agenda.py:1395 | `getattr(self, "status_info", None) ; self.status()` | LOGIC | -- | -- |
| agenda.py:1396 | `floor is not None` | LOGIC | -- | -- |
| agenda.py:1396 | `floor < -st["threat"]` | CHESS FACT | [D12] existing: unfinished line | cases 22/18, DECISION 35 |
| agenda.py:1398 | `refuted` | LOGIC | -- | -- |
| agenda.py:1400 | `floor is not None` | LOGIC | -- | -- |
| agenda.py:1401 | `rest is not None ; read is not None` | LOGIC | -- | -- |
| agenda.py:1402 | `refuted ; len(refuted) == 1` | LOGIC | -- | -- |
| agenda.py:1403 | `floor is not None` | LOGIC | -- | -- |
| agenda.py:1403 | `Claim('win', ...) ; Claim(... goal='win')` | CHESS FACT | [D12] existing: unfinished line | cases 22/18, DECISION 35 |
| agenda.py:1408 | `pc.color == self.me ; b.is_attacked_by(not self.me, sq)` | RULE | -- | -- |
| agenda.py:1408 | `pc.piece_type != chess.KING` | CHESS FACT | [C1] PROPOSED: "names: king = the royal unit (Game.royal); excluded from the valuation" (VALUATION; reads material) | no printed signature: 8 rows |
| agenda.py:1414 | `self._pass_deep is not None` | LOGIC | -- | -- |
| agenda.py:1419 | `p.turn` | NAME | -- | -- |
| agenda.py:1422 | `self.ctx is not None` | LOGIC | -- | -- |
| agenda.py:1426 | `fin is not None ; time.monotonic() >= fin` | LOGIC | -- | -- |
| agenda.py:1430 | `got is not None ; best is None ; got[2] - got[1] > best[0]` | LOGIC | -- | -- |
| agenda.py:1430 | `got[2] > got[1]` | CHESS FACT | [D18] existing: the deep pass | cases 0/0, DECISION 0 |
| agenda.py:1431 | `2  in: best = (got[2] - got[1], got[0])` | LOGIC | -- | -- |
| agenda.py:1432 | `best` | LOGIC | -- | -- |
| agenda.py:1444 | `spn is None ; spn <= 0` | LOGIC | -- | -- |
| agenda.py:1451 | `len(san) in (4, 5) ; san[0] in "abcdefgh" ; san[1] in "12345678"` | NAME | -- | -- |
| agenda.py:1454 | `ctx.route_costs.get(mkey) ; (0, [])` | LOGIC | -- | -- |
| agenda.py:1458 | `kk != k` | LOGIC | -- | -- |
| agenda.py:1461 | `W.forcing_why(b, m, stake, self.us, mate_threats=True, mate2=k >= 2, delayed=dl and k >= 1, nosafe=ns and k >= 1)` | LOGIC | -- | -- |
| agenda.py:1465 | `cands` | LOGIC | -- | -- |
| agenda.py:1467 | `min(... key=lambda c: (c[0], c[2], c[3], c[4]))` | HEURISTIC | -- | -- |
| agenda.py:1468 | `need <= left` | LOGIC | -- | -- |
| agenda.py:1470 | `words.startswith("priced by")` | LOGIC | -- | -- |
| agenda.py:1476 | `v[1]` | LOGIC | -- | -- |
| agenda.py:1486 | `len(san) in (4, 5) ; san[0] in "abcdefgh" ; san[1] in "12345678"` | NAME | -- | -- |
| agenda.py:1489 | `spn ; ctx.route_costs.get(mkey) ; (0, [])` | LOGIC | -- | -- |
| agenda.py:1491 | `rows` | LOGIC | -- | -- |
| agenda.py:1493 | `r[0] ; 0 ; r[0] is None ; sort(... key=lambda r: (r[0] is None, r[0] or 0, r[1]))` | LOGIC | -- | -- |
| agenda.py:1495 | `p is not None ; w.startswith("priced by")` | LOGIC | -- | -- |
| agenda.py:1499 | `self.ctx is not None ; self.chase_depth == 0 ; self.ctx.landings.asked` | LOGIC | -- | -- |
| agenda.py:1501 | `self.ctx.landing_lines` | LOGIC | -- | -- |
| agenda.py:1503 | `self.ctx is not None ; self.chase_depth == 0 ; self.best is not None ; self.best[2].move` | LOGIC | -- | -- |
| agenda.py:1504 | `2  in: mv = chess.Move.from_uci(self.best[2].move)` | LOGIC | -- | -- |
| agenda.py:1506 | `pc is not None` | LOGIC | -- | -- |
| agenda.py:1507 | `l.frm == mv.from_square ; l.to == mv.to_square ; l.sym == pc.symbol().lower()` | LOGIC | -- | -- |
| agenda.py:1510 | `l.key() not in seen` | LOGIC | -- | -- |
| agenda.py:1512 | `len(seen) <= 5` | LOGIC | -- | -- |
| agenda.py:1514 | `self.ctx is not None ; self.chase_depth == 0 ; self.ctx.finish_lines` | LOGIC | -- | -- |
| agenda.py:1517 | `self.ctx is not None ; self.chase_depth == 0` | LOGIC | -- | -- |
| agenda.py:1519 | `self.ctx is not None ; self.chase_depth == 0 ; self.ctx.final is not None` | LOGIC | -- | -- |
| agenda.py:1521 | `self.ctx.hit` | LOGIC | -- | -- |
| agenda.py:1522 | `self.ctx.hit` | LOGIC | -- | -- |
| agenda.py:1523 | `self.ctx is not None ; self.chase_depth == 0 ; self.ctx.static_lines` | LOGIC | -- | -- |
| agenda.py:1526 | `self.ctx is not None ; self.chase_depth == 0 ; self.ctx.local_lines` | LOGIC | -- | -- |
| agenda.py:1530 | `self.ctx is not None ; self.chase_depth == 0 ; self.ctx.open_lines` | LOGIC | -- | -- |
| agenda.py:1532 | `12  in: self.lines += [" " + x for x in self.ctx.open_lines[:12]]` | LOGIC | -- | -- |
| agenda.py:1533 | `self.ctx is not None ; self.chase_depth == 0 ; self.ctx.rows` | LOGIC | -- | -- |
| agenda.py:1538 | `0o644  in: fd = os.open(paths.experiment("chase_log.jsonl"), os.O_WRONLY \| os.O_APPEND \| os.O_CREAT, 0o644) ; os.O_CR...` | LOGIC | -- | -- |
| agenda.py:1540 | `data` | LOGIC | -- | -- |
| agenda.py:1550 | `GP.load() is None ; self.heap` | LOGIC | -- | -- |
| agenda.py:1556 | `q.status in ("verified", "killed", "gap") ; q.reason` | LOGIC | -- | -- |
| agenda.py:1557 | `q.claimed is not None` | LOGIC | -- | -- |
| agenda.py:1559 | `4  in: q = t[4]` | LOGIC | -- | -- |
| agenda.py:1564 | `self.ctx is not None ; self.ctx.seconds` | LOGIC | -- | -- |
| agenda.py:1565 | `self.ctx.final ; time.monotonic()` | LOGIC | -- | -- |
| agenda.py:1567 | `scores` | LOGIC | -- | -- |
| agenda.py:1570 | `max(... key=lambda i: (open_scores[i], -entries[i][0]))` | LOGIC | -- | -- |
| agenda.py:1574 | `True` | LOGIC | -- | -- |
| agenda.py:1575 | `self._threats_done ; self.best ; self.heap` | LOGIC | -- | -- |
| agenda.py:1575 | `self.best[0] >= R.STAKE_MATE ; self.heap[0][4].stake < R.STAKE_MATE` | CHESS FACT | [A2] PROPOSED: "stake: the terminal (mate = 100)" (VALUATION; reads movement, attack) | cases 97/97, DECISION 0 |
| agenda.py:1578 | `self.heap` | LOGIC | -- | -- |
| agenda.py:1579 | `self.local is not None ; self.local["full"] ; self._local_next()` | LOGIC | -- | -- |
| agenda.py:1582 | `4  in: nstake = self.heap[0][4].stake` | LOGIC | -- | -- |
| agenda.py:1583 | `self.best ; self.best[0] > nstake` | LOGIC | -- | -- |
| agenda.py:1583 | `self.best[0] >= BD.MATE_VALUE` | CHESS FACT | [A2] PROPOSED: "stake: the terminal (mate = 100)" (VALUATION; reads movement, attack) | cases 97/97, DECISION 0 |
| agenda.py:1584 | `2  in: self.lines.append(f"STOP (a): the best claim ({self.best[2].what}, {self.best[0]}) is above the stake of every...` | LOGIC | -- | -- |
| agenda.py:1586 | `self.asked >= self.budget` | LOGIC | -- | -- |
| agenda.py:1587 | `4  in: left = [t[4] for t in self.heap]` | LOGIC | -- | -- |
| agenda.py:1591 | `self.chase_depth ; self.ctx.out_of_time() ; self.late ; self.ctx.finishing(self.frame)` | LOGIC | -- | -- |
| agenda.py:1595 | `min(... key=lambda t: (t[0], t[2], t[3]))` | LOGIC | -- | -- |
| agenda.py:1596 | `PRIOR ; self.ctx is not None ; self.ctx.pool ; self.chase_depth ; self._root` | LOGIC | -- | -- |
| agenda.py:1597 | `graph is not None` | LOGIC | -- | -- |
| agenda.py:1601 | `4  in: q = best[4]` | LOGIC | -- | -- |
| agenda.py:1602 | `plain[4] is not q` | LOGIC | -- | -- |
| agenda.py:1603 | `4  in: self.lines.append(f"policy: Q{q.id} asked first (graph score {gs:+.2f}; the stake order would ask Q{plain[4].i...` | LOGIC | -- | -- |
| agenda.py:1605 | `PRIOR ; self.ctx is not None ; self.ctx.pool ; self.chase_depth ; QP.load() is not None` | LOGIC | -- | -- |
| agenda.py:1606 | `self.ctx is not None ; self.ctx.pool ; 3  in: scored = [(QP.score(QP.features(t[4], self.board, self.us, self.chase_d...` | LOGIC | -- | -- |
| agenda.py:1607 | `3  in: best = max(scored)[3]` | LOGIC | -- | -- |
| agenda.py:1610 | `4  in: q = best[4]` | LOGIC | -- | -- |
| agenda.py:1611 | `plain[4] is not q` | LOGIC | -- | -- |
| agenda.py:1612 | `4  in: self.lines.append(f"prior: Q{q.id} asked first (score {max(scored)[0]:+.2f}; the stake order would ask Q{plain...` | LOGIC | -- | -- |
| agenda.py:1616 | `plain[4] is not q` | LOGIC | -- | -- |
| agenda.py:1619 | `DEFER ; self._root ; self.chase_depth == 0 ; self.ctx is not None ; self.ctx.pool ; self.ctx.need is not None` | LOGIC | -- | -- |
| agenda.py:1619 | `self.ctx.need > 0 ; q.goal != "save" ; q.stake < self.ctx.need` | CHESS FACT | [A5] PROPOSED: "goal: the ladder -- don't lose, then win (status, line, keeps)" (GOAL; reads material, tempo) | cases 22/18, DECISION 35 |
| agenda.py:1624 | `self.ctx is not None ; self.ctx.resume_set is not None ; self.chase_depth == 0 ; self._root ; self._carry_key(q) not ...` | LOGIC | -- | -- |
| agenda.py:1625 | `self.ctx.carry_q.get(self._carry_key(q)) is not None` | LOGIC | -- | -- |
| agenda.py:1628 | `self.ctx is not None ; self.ctx.carry_q ; self.chase_depth == 0 ; self._root ; self._carry_question(q)` | LOGIC | -- | -- |
| agenda.py:1631 | `self._inherit(q)` | LOGIC | -- | -- |
| agenda.py:1634 | `self.chase_depth` | LOGIC | -- | -- |
| agenda.py:1638 | `self.chase_depth == 0` | LOGIC | -- | -- |
| agenda.py:1641 | `self._routes_read > r0` | LOGIC | -- | -- |
| agenda.py:1642 | `self.ctx is not None` | LOGIC | -- | -- |
| agenda.py:1644 | `self.chase_depth` | LOGIC | -- | -- |
| agenda.py:1651 | `self.local is not None ; self.local["cls"].get(id(q)) == "Q1" ; q.status == "verified" ; self._routes_read > r0 ; get...` | LOGIC | -- | -- |
| agenda.py:1665 | `hit is None ; q.status != "verified"` | LOGIC | -- | -- |
| agenda.py:1667 | `2  in: out[self._carry_key(q)] = dict(net=hit[0], second=hit[1], claim=hit[2], dep=bool(getattr(q, "_tier_dep", False)))` | LOGIC | -- | -- |
| agenda.py:1676 | `hit is None` | LOGIC | -- | -- |
| agenda.py:1684 | `self._better(net, hit["second"], cl)` | LOGIC | -- | -- |
| agenda.py:1688 | `st is not None` | LOGIC | -- | -- |
| agenda.py:1688 | `net >= BD.MATE_VALUE` | CHESS FACT | [A2] PROPOSED: "stake: the terminal (mate = 100)" (VALUATION; reads movement, attack) | cases 97/97, DECISION 0 |
| agenda.py:1688 | `st["balance"] + net >= st["line"]` | CHESS FACT | [A5] PROPOSED: "goal: the ladder -- don't lose, then win (status, line, keeps)" (GOAL; reads material, tempo) | cases 22/18, DECISION 35 |
| agenda.py:1690 | `settle` | LOGIC | -- | -- |
| agenda.py:1694 | `keeps ; hit["dep"]` | LOGIC | -- | -- |
| agenda.py:1696 | `keeps` | LOGIC | -- | -- |
| agenda.py:1708 | `self.inherit ; self.ctx is not None ; key is not None` | LOGIC | -- | -- |
| agenda.py:1710 | `key is not None` | LOGIC | -- | -- |
| agenda.py:1711 | `hit is None` | LOGIC | -- | -- |
| agenda.py:1712 | `key is not None ; self.inherit` | LOGIC | -- | -- |
| agenda.py:1715 | `hit["reads"] & hit["diff"]` | LOGIC | -- | -- |
| agenda.py:1719 | `q.goal == "mate"` | NAME | -- | -- |
| agenda.py:1720 | `sq is not None` | LOGIC | -- | -- |
| agenda.py:1721 | `pc ; sq is not None` | LOGIC | -- | -- |
| agenda.py:1722 | `hit["kind"] == "claim"` | LOGIC | -- | -- |
| agenda.py:1725 | `mv not in b.legal_moves` | RULE | -- | -- |
| agenda.py:1733 | `self._better(hit["net"], 0, cl)` | LOGIC | -- | -- |
| agenda.py:1746 | `self.ctx is not None` | LOGIC | -- | -- |
| agenda.py:1746 | `self.min_stake < BD.MATE_VALUE` | CHESS FACT | [A7] PROPOSED: "goal: the need (what a claim must bring), propagated into the chase" (GOAL; reads material) | no printed signature: 15 rows |
| agenda.py:1746 | `q.goal in ("mate", "win", "promote") ; self.chase_depth >= 2 ; q.goal != "mate"` | CHESS FACT | [B6] PROPOSED: "stop: the chase's follow list (forced words below the root)" (STOP; reads attack, movement, tempo) (beside existing: check; capture) | no printed signature: 10 rows |
| agenda.py:1751 | `self._can_chase(q) ; self.chase_depth == 0 ; 2  in: return 0 if not self._can_chase(q) else (2 if self.chase_depth ==...` | LOGIC | -- | -- |
| agenda.py:1755 | `key is None` | LOGIC | -- | -- |
| agenda.py:1762 | `f is not None ; f.kind == "claim"` | LOGIC | -- | -- |
| agenda.py:1764 | `mv not in b.legal_moves` | RULE | -- | -- |
| agenda.py:1766 | `f is None` | LOGIC | -- | -- |
| agenda.py:1767 | `stale_miss` | LOGIC | -- | -- |
| agenda.py:1769 | `self.chase_depth` | LOGIC | -- | -- |
| agenda.py:1770 | `cand` | LOGIC | -- | -- |
| agenda.py:1775 | `q.goal == "mate"` | NAME | -- | -- |
| agenda.py:1776 | `sq is not None` | LOGIC | -- | -- |
| agenda.py:1777 | `pc ; sq is not None` | LOGIC | -- | -- |
| agenda.py:1779 | `f.kind == "claim"` | LOGIC | -- | -- |
| agenda.py:1781 | `f.payload.get("chase")` | LOGIC | -- | -- |
| agenda.py:1788 | `self._better(f.payload["net"], 0, cl)` | LOGIC | -- | -- |
| agenda.py:1805 | `q.status not in ("killed", "verified", "gap")` | LOGIC | -- | -- |
| agenda.py:1809 | `key is not None` | LOGIC | -- | -- |
| agenda.py:1811 | `kind == "claim"` | LOGIC | -- | -- |
| agenda.py:1812 | `q.claimed is not None` | LOGIC | -- | -- |
| agenda.py:1813 | `2  in: value, claim, reads = q.claimed[0], q.claimed[1], q.claimed[2]` | LOGIC | -- | -- |
| agenda.py:1815 | `portable ; claim.lemma.startswith(("chase", "delayed", "mate_in"))` | CHESS FACT | [D26] existing: chase: step_after+step_least; delayed_balance_alive_d1 (+2) | no printed signature: 4 rows |
| agenda.py:1818 | `new_claims` | LOGIC | -- | -- |
| agenda.py:1819 | `claim is not None` | LOGIC | -- | -- |
| agenda.py:1821 | `claim is not None ; key is not None` | LOGIC | -- | -- |
| agenda.py:1821 | `claim.lemma.startswith("chase")` | CHESS FACT | [D26] existing: chase: step_after+step_least; delayed_balance_alive_d1 (+2) | no printed signature: 4 rows |
| agenda.py:1823 | `creads is None` | LOGIC | -- | -- |
| agenda.py:1828 | `kind == "kill"` | LOGIC | -- | -- |
| agenda.py:1829 | `80  in: payload = dict(why=q.reason.removeprefix("inherited: ").split(";")[0][:80], routed=getattr(q, "_routed", False))` | LOGIC | -- | -- |
| agenda.py:1830 | `claim is not None` | LOGIC | -- | -- |
| agenda.py:1831 | `reads is not None` | LOGIC | -- | -- |
| agenda.py:1832 | `key is not None` | LOGIC | -- | -- |
| agenda.py:1833 | `getattr(q, "_reasoned", False) ; kind == "claim" ; getattr(q, "_chased", False) ; kind == "kill" ; getattr(q, "_chain...` | LOGIC | -- | -- |
| agenda.py:1833 | `value >= min(q.stake, BD.MATE_VALUE)` | CHESS FACT | [A2] PROPOSED: "stake: the terminal (mate = 100)" (VALUATION; reads movement, attack) | cases 97/97, DECISION 0 |
| agenda.py:1838 | `claim is not None` | LOGIC | -- | -- |
| agenda.py:1843 | `res.get("carried") ; self.chase_depth != 0 ; res.get("reused")` | LOGIC | -- | -- |
| agenda.py:1846 | `res.get("gains") ; []` | LOGIC | -- | -- |
| agenda.py:1849 | `res.get("gap") ; res.get("killed")` | LOGIC | -- | -- |
| agenda.py:1857 | `self._carried_route` | LOGIC | -- | -- |
| agenda.py:1862 | `f.reads ; f.payload.get("reads")` | LOGIC | -- | -- |
| agenda.py:1863 | `fr` | LOGIC | -- | -- |
| agenda.py:1874 | `chess.WHITE  in: for colour in (chess.WHITE, chess.BLACK): k = c.king(colour) if k is not None: reads.update(W.slide_...` | NAME | -- | -- |
| agenda.py:1876 | `k is not None` | LOGIC | -- | -- |
| agenda.py:1886 | `key is None` | LOGIC | -- | -- |
| agenda.py:1888 | `q.status == "killed" ; q.reads is not None` | LOGIC | -- | -- |
| agenda.py:1889 | `dict(... kind='kill') ; 80  in: out[key] = dict(kind="kill", reads=q.reads, diff=diff, why=q.reason.removeprefix("inh...` | LOGIC | -- | -- |
| agenda.py:1890 | `q.status == "verified" ; q.claimed is not None` | LOGIC | -- | -- |
| agenda.py:1890 | `q.claimed[1].lemma.startswith("chase")` | CHESS FACT | [D26] existing: chase: step_after+step_least; delayed_balance_alive_d1 (+2) | no printed signature: 4 rows |
| agenda.py:1891 | `dict(... kind='claim') ; 2  in: out[key] = dict(kind="claim", reads=q.claimed[2], diff=diff, net=q.claimed[0], claim=...` | LOGIC | -- | -- |
| agenda.py:1901 | `LOCAL_INNER ; q is None ; self.ctx is None` | LOGIC | -- | -- |
| agenda.py:1903 | `q.goal == "mate"` | CHESS FACT | [A1m] PROPOSED: "goal: mate (the rule's terminal)" (GOAL; reads movement, attack) | cases 100/100, DECISION 0 |
| agenda.py:1905 | `q.pattern.get("target") is not None` | LOGIC | -- | -- |
| agenda.py:1905 | `q.goal == "win"` | CHESS FACT | [A1w] PROPOSED: "goal: win material" (GOAL; reads material, attack) | cases 83/86, DECISION 0 |
| agenda.py:1907 | `q.pattern.get("pawn") is not None` | LOGIC | -- | -- |
| agenda.py:1907 | `q.goal == "promote"` | CHESS FACT | [A1p] PROPOSED: "goal: promote (the rule)" (GOAL; reads movement, tempo) | cases 48/49, DECISION 0 |
| agenda.py:1920 | `hit is not None` | LOGIC | -- | -- |
| agenda.py:1923 | `q.pattern.get(k) is not None ; q.pattern[k] != m.from_square` | LOGIC | -- | -- |
| agenda.py:1925 | `d.sq is not None` | LOGIC | -- | -- |
| agenda.py:1926 | `q.pattern.get("target") is not None` | LOGIC | -- | -- |
| agenda.py:1926 | `q.goal == "win"` | CHESS FACT | [A1w] PROPOSED: "goal: win material" (GOAL; reads material, attack) | cases 83/86, DECISION 0 |
| agenda.py:1927 | `chess.WHITE  in: out.update(self.board.attackers(chess.WHITE, q.pattern["target"]) \| self.board.attackers(chess.BLACK...` | NAME | -- | -- |
| agenda.py:1928 | `q.pattern.get("pawn") is not None` | LOGIC | -- | -- |
| agenda.py:1928 | `q.goal == "promote"` | CHESS FACT | [A1p] PROPOSED: "goal: promote (the rule)" (GOAL; reads movement, tempo) | cases 48/49, DECISION 0 |
| agenda.py:1929 | `q.pattern.get("pushes") ; 1` | LOGIC | -- | -- |
| agenda.py:1932 | `th is not None ; ksq is not None` | LOGIC | -- | -- |
| agenda.py:1932 | `th.kind in ("mate", "mate2", "check")` | CHESS FACT | [D24] existing: threat of mate; threat of mate in two (+4) | cases 11/11, DECISION 0 |
| agenda.py:1934 | `chess.BB_KING_ATTACKS  in: out.update(chess.SquareSet(chess.BB_KING_ATTACKS[ksq]))` | CHESS FACT | [C6] PROPOSED: "closing: the king's step (Game.adjacent) in a mate threat's frame" (CLOSING; reads movement, attack) | cases 67/67, DECISION 0 |
| agenda.py:1935 | `th.kind == "check"` | CHESS FACT | [D24] existing: threat of mate; threat of mate in two (+4) | cases 11/11, DECISION 0 |
| agenda.py:1937 | `th.kind == "mate"` | CHESS FACT | [D24] existing: threat of mate; threat of mate in two (+4) | cases 11/11, DECISION 0 |
| agenda.py:1941 | `w.gives_check(mv) ; G.mates(w, mv)` | RULE | -- | -- |
| agenda.py:1946 | `th is not None ; th.target is not None` | LOGIC | -- | -- |
| agenda.py:1955 | `cq is not None` | LOGIC | -- | -- |
| agenda.py:1955 | `stake = cq.stake if cq is not None else 1 (the default stake: one unit)` | CHESS FACT | [A6] PROPOSED: "valuation: a gain is at least one unit of material" (VALUATION; reads material) (beside existing: no claim of ours and no material gain) | cases 80/80, DECISION 0 |
| agenda.py:1957 | `key not in ctx.route_threats` | LOGIC | -- | -- |
| agenda.py:1958 | `ctx.tier >= 2` | HEURISTIC | -- | -- |
| agenda.py:1968 | `gk is None` | LOGIC | -- | -- |
| agenda.py:1974 | `gk[0] == "mate"` | CHESS FACT | [A1m] PROPOSED: "goal: mate (the rule's terminal)" (GOAL; reads movement, attack) | cases 100/100, DECISION 0 |
| agenda.py:1976 | `gk[0] == "win"` | CHESS FACT | [A1w] PROPOSED: "goal: win material" (GOAL; reads material, attack) | cases 83/86, DECISION 0 |
| agenda.py:1978 | `m.to_square == t` | CHESS FACT | [D17] existing: closed by the frame | cases 81/82, DECISION 0 |
| agenda.py:1980 | `r.from_square == t` | CHESS FACT | [D17] existing: closed by the frame | cases 81/82, DECISION 0 |
| agenda.py:1983 | `p0 is None ; p2 is None` | LOGIC | -- | -- |
| agenda.py:1983 | `p2.color == self.me` | RULE | -- | -- |
| agenda.py:1983 | `p2.piece_type != p0.piece_type` | CHESS FACT | [D17] existing: closed by the frame | cases 81/82, DECISION 0 |
| agenda.py:1986 | `gk[0] == "promote"` | CHESS FACT | [A1p] PROPOSED: "goal: promote (the rule)" (GOAL; reads movement, tempo) | cases 48/49, DECISION 0 |
| agenda.py:1988 | `m.from_square == p` | CHESS FACT | [D17] existing: closed by the frame | cases 81/82, DECISION 0 |
| agenda.py:1989 | `m.promotion` | RULE | -- | -- |
| agenda.py:1993 | `pw is None` | LOGIC | -- | -- |
| agenda.py:1993 | `pw.color != self.me` | RULE | -- | -- |
| agenda.py:1993 | `pw.piece_type != chess.PAWN` | CHESS FACT | [C2] PROPOSED: "names: pawn = the unit that promotes (Game.promotes)" (GOAL; reads movement) | no printed signature: 2 rows |
| agenda.py:1998 | `units is not None` | LOGIC | -- | -- |
| agenda.py:1999 | `u != r.to_square` | LOGIC | -- | -- |
| agenda.py:2003 | `th is not None ; check` | LOGIC | -- | -- |
| agenda.py:2003 | `th.kind in ("mate", "threat")` | CHESS FACT | [D24] existing: threat of mate; threat of mate in two (+4) | cases 11/11, DECISION 0 |
| agenda.py:2005 | `check ; capture ; frame & diff` | CHESS FACT | [D17] existing: closed by the frame | cases 81/82, DECISION 0 |
| agenda.py:2007 | `delay` | LOGIC | -- | -- |
| agenda.py:2008 | `check` | LOGIC | -- | -- |
| agenda.py:2009 | `[W.base.see(d, s) for s in sqs if d.piece_at(s) is not None and d.is_attacked_by(self.me, s)] ; [0]` | LOGIC | -- | -- |
| agenda.py:2009 | `d.piece_at(s) is not None ; d.is_attacked_by(self.me, s)` | RULE | -- | -- |
| agenda.py:2010 | `sq is not None ; units is not None` | LOGIC | -- | -- |
| agenda.py:2010 | `gk[0] != "mate"` | CHESS FACT | [A1m] PROPOSED: "goal: mate (the rule's terminal)" (GOAL; reads movement, attack) | cases 100/100, DECISION 0 |
| agenda.py:2011 | `th is not None` | LOGIC | -- | -- |
| agenda.py:2021 | `b.is_check()` | RULE | -- | -- |
| agenda.py:2023 | `q.goal == g["kind"] ; p.get("frm") in g["units"] if g["kind"] == "mate" and g["units"] is not None else g["kind"] == ...` | LOGIC | -- | -- |
| agenda.py:2025 | `q.goal == "save"` | CHESS FACT | [A1s] PROPOSED: "goal: save material (don't lose)" (GOAL; reads material, attack) (beside existing: in check) | cases 78/83, DECISION 0 |
| agenda.py:2026 | `p.get("target") is not None` | LOGIC | -- | -- |
| agenda.py:2027 | `p["target"] in d` | LOGIC | -- | -- |
| agenda.py:2028 | `p.get("threat") ; p.get("push")` | LOGIC | -- | -- |
| agenda.py:2029 | `mv` | LOGIC | -- | -- |
| agenda.py:2031 | `p.get("promoter") is not None ; set(p.get("reads") or ()) ; {mv.from_square, mv.to_square} \| set(p.get("targets") or ...` | LOGIC | -- | -- |
| agenda.py:2032 | `sqs & d` | LOGIC | -- | -- |
| agenda.py:2034 | `p.get("target") == g["to"]` | LOGIC | -- | -- |
| agenda.py:2034 | `q.goal == "win"` | CHESS FACT | [A1w] PROPOSED: "goal: win material" (GOAL; reads material, attack) | cases 83/86, DECISION 0 |
| agenda.py:2047 | `s in d` | LOGIC | -- | -- |
| agenda.py:2047 | `b.attacks_mask(s) & dm` | RULE | -- | -- |
| agenda.py:2055 | `p.get(k) is not None` | LOGIC | -- | -- |
| agenda.py:2056 | `sqs ; any(s in S for s in sqs)` | LOGIC | -- | -- |
| agenda.py:2065 | `self.board.is_check()` | RULE | -- | -- |
| agenda.py:2068 | `L["live"]` | LOGIC | -- | -- |
| agenda.py:2072 | `L["deferred"]` | LOGIC | -- | -- |
| agenda.py:2078 | `heapq.heappush(self.heap, (-q.stake, -self._recall_score(q)[0], q.depth, self._seq, q))` | LOGIC | -- | -- |
| agenda.py:2079 | `self._threats_done ; getattr(self, "_threats_local", False)` | LOGIC | -- | -- |
| agenda.py:2081 | `bool(L["deferred"]) ; self._threats_done` | LOGIC | -- | -- |
| agenda.py:2088 | `L["cls"].get(id(q)) == k` | LOGIC | -- | -- |
| agenda.py:2091 | `q.claimed is not None ; c.qid == q.id` | LOGIC | -- | -- |
| agenda.py:2092 | `q.status == "verified" ; cl is not None` | LOGIC | -- | -- |
| agenda.py:2093 | `q.claimed is not None` | LOGIC | -- | -- |
| agenda.py:2095 | `self.board.is_check()` | RULE | -- | -- |
| agenda.py:2097 | `g["binds"]` | LOGIC | -- | -- |
| agenda.py:2099 | `by["Q1"]` | LOGIC | -- | -- |
| agenda.py:2100 | `q.status == "verified"` | LOGIC | -- | -- |
| agenda.py:2101 | `d.sq is not None` | LOGIC | -- | -- |
| agenda.py:2102 | `dd` | LOGIC | -- | -- |
| agenda.py:2103 | `cl ; L["live"]` | LOGIC | -- | -- |
| agenda.py:2106 | `"; ".join(said(q) for q in by["Q2"]) ; "none"` | LOGIC | -- | -- |
| agenda.py:2107 | `g["delay"] ; by["Q3"]` | LOGIC | -- | -- |
| agenda.py:2108 | `self.best ; 2  in: best = f"{self.best[2].what} ({self.best[0]:+d})" if self.best else "no claim"` | LOGIC | -- | -- |
| agenda.py:2110 | `q3 ; 'none' ; L['why'] ; 'stopped (a claim, the budget or the clock)'` | LOGIC | -- | -- |
| agenda.py:2114 | `L["full"]` | LOGIC | -- | -- |
| agenda.py:2116 | `L["why"].startswith("closed")` | LOGIC | -- | -- |
| agenda.py:2118 | `self.chase_depth == 1 ; line not in ctx.local_lines` | LOGIC | -- | -- |
| agenda.py:2128 | `q.goal == "save"` | CHESS FACT | [A1s] PROPOSED: "goal: save material (don't lose)" (GOAL; reads material, attack) (beside existing: in check) | cases 78/83, DECISION 0 |
| agenda.py:2128 | `"checkers" in q.pattern` | CHESS FACT | [D21] existing: in check | cases 58/64, DECISION 0 |
| agenda.py:2131 | `q.goal == "promote"` | CHESS FACT | [A1p] PROPOSED: "goal: promote (the rule)" (GOAL; reads movement, tempo) | cases 48/49, DECISION 0 |
| agenda.py:2134 | `q.goal == "fork"` | CHESS FACT | [A1f] existing: fork; fork: two threats, one move (Ideas.ForkSquare... | cases 5/5, DECISION 0 |
| agenda.py:2137 | `self._reasoned_kill(q, head)` | LOGIC | -- | -- |
| agenda.py:2137 | `q.goal == "mate"` | CHESS FACT | [A1m] PROPOSED: "goal: mate (the rule's terminal)" (GOAL; reads movement, attack) | cases 100/100, DECISION 0 |
| agenda.py:2139 | `q.goal == "mate"` | CHESS FACT | [A1m] PROPOSED: "goal: mate (the rule's terminal)" (GOAL; reads movement, attack) | cases 100/100, DECISION 0 |
| agenda.py:2139 | `q.pattern.get("reach")` | CHESS FACT | [D1e] PROPOSED: "goal: mate at distance two (the reach defect)" (GOAL; reads movement, attack, tempo) | cases 23/23, DECISION 0 |
| agenda.py:2142 | `q.goal == "save"` | CHESS FACT | [A1s] PROPOSED: "goal: save material (don't lose)" (GOAL; reads material, attack) (beside existing: in check) | cases 78/83, DECISION 0 |
| agenda.py:2142 | `"promoter" in q.pattern` | CHESS FACT | [D14] existing: threat: ... binds after a pass | cases 9/9, DECISION 0 |
| agenda.py:2145 | `q.goal == "save"` | CHESS FACT | [A1s] PROPOSED: "goal: save material (don't lose)" (GOAL; reads material, attack) (beside existing: in check) | cases 78/83, DECISION 0 |
| agenda.py:2145 | `"threat" in q.pattern` | CHESS FACT | [D14] existing: threat: ... binds after a pass | cases 9/9, DECISION 0 |
| agenda.py:2148 | `q.goal == "save"` | CHESS FACT | [A1s] PROPOSED: "goal: save material (don't lose)" (GOAL; reads material, attack) (beside existing: in check) | cases 78/83, DECISION 0 |
| agenda.py:2151 | `q.pattern.get("checker")` | CHESS FACT | [D21] existing: in check | cases 58/64, DECISION 0 |
| agenda.py:2155 | `q.goal == "mate"` | CHESS FACT | [A1m] PROPOSED: "goal: mate (the rule's terminal)" (GOAL; reads movement, attack) | cases 100/100, DECISION 0 |
| agenda.py:2159 | `q.defects` | LOGIC | -- | -- |
| agenda.py:2160 | `q.defects` | LOGIC | -- | -- |
| agenda.py:2169 | `W.tempo_ok(q)` | LOGIC | -- | -- |
| agenda.py:2171 | `self._chase(q)` | LOGIC | -- | -- |
| agenda.py:2177 | `q.goal == "mate"` | CHESS FACT | [A1m] PROPOSED: "goal: mate (the rule's terminal)" (GOAL; reads movement, attack) | cases 100/100, DECISION 0 |
| agenda.py:2182 | `self.chase_depth ; self.ctx is None ; self.side_depth` | LOGIC | -- | -- |
| agenda.py:2197 | `self.ctx is None ; p.get("more") ; frm is None` | LOGIC | -- | -- |
| agenda.py:2197 | `b.piece_at(frm) is None ; b.piece_at(frm).symbol() != p.get("sym")` | RULE | -- | -- |
| agenda.py:2203 | `hit is not None` | LOGIC | -- | -- |
| agenda.py:2208 | `len(self.ctx.landing_lines) < 3` | LOGIC | -- | -- |
| agenda.py:2230 | `P2.legal_count() == 0` | RULE | -- | -- |
| agenda.py:2231 | `P2.is_check()` | RULE | -- | -- |
| agenda.py:2231 | `R.STAKE_MATE  in: net = -R.STAKE_MATE if P2.is_check() else 0` | CHESS FACT | [A9] existing: they mate us; stalemate (+1) | cases 1/1, DECISION 0 |
| agenda.py:2232 | `mate1 ; P2.mate_in_one()` | RULE | -- | -- |
| agenda.py:2233 | `R.STAKE_MATE  in: net = R.STAKE_MATE` | CHESS FACT | [A9] existing: they mate us; stalemate (+1) | cases 1/1, DECISION 0 |
| agenda.py:2235 | `P2.is_check()` | RULE | -- | -- |
| agenda.py:2236 | `self.ctx is not None ; CLOSE_AFTER_CAPTURE` | LOGIC | -- | -- |
| agenda.py:2236 | `self._their_after_capture(P2)[0]` | CHESS FACT | [D23] PROPOSED: "valuation: the exchange law's principal lines (swap list)" (VALUATION; reads attack, material) (beside existing: nothing of theirs) | cases 29/48, DECISION 0 |
| agenda.py:2236 | `ex > 0` | CHESS FACT | [D7] existing: one move answers both threats; fork: two threats, one move (Ideas.ForkSquare... | cases 5/5, DECISION 0 |
| agenda.py:2239 | `worst is None ; net < worst[0]` | LOGIC | -- | -- |
| agenda.py:2245 | `worst is None` | LOGIC | -- | -- |
| agenda.py:2245 | `worst[0] < 1` | CHESS FACT | [D7] existing: one move answers both threats; fork: two threats, one move (Ideas.ForkSquare... | cases 5/5, DECISION 0 |
| agenda.py:2246 | `worst is not None` | LOGIC | -- | -- |
| agenda.py:2248 | `r is not None` | LOGIC | -- | -- |
| agenda.py:2249 | `r.to_square == m.to_square` | CHESS FACT | [D7] existing: one move answers both threats; fork: two threats, one move (Ideas.ForkSquare... | cases 5/5, DECISION 0 |
| agenda.py:2251 | `c.gives_check(r)` | RULE | -- | -- |
| agenda.py:2253 | `r.from_square in [t for t, _w in p["targets"]]` | CHESS FACT | [D7] existing: one move answers both threats; fork: two threats, one move (Ideas.ForkSquare... | cases 5/5, DECISION 0 |
| agenda.py:2257 | `r is not None ; worst` | LOGIC | -- | -- |
| agenda.py:2260 | `Claim('win', ...)` | CHESS FACT | [A1f] existing: fork; fork: two threats, one move (Ideas.ForkSquare... | cases 5/5, DECISION 0 |
| agenda.py:2261 | `p["check"]` | LOGIC | -- | -- |
| agenda.py:2261 | `b.piece_at(t)` | RULE | -- | -- |
| agenda.py:2268 | `self._better(worst[0], 0, claim)` | LOGIC | -- | -- |
| agenda.py:2279 | `n < 1` | CHESS FACT | [D9] existing: reasoned: the wall; reasoned: the measure they hold | cases 12/26, DECISION 0 |
| agenda.py:2282 | `any(d.kind != "stopper" or set(d.by) - {rook} for d in defects)` | CHESS FACT | [D9] existing: reasoned: the wall; reasoned: the measure they hold | cases 12/26, DECISION 0 |
| agenda.py:2284 | `best is None ; n < best[0]` | LOGIC | -- | -- |
| agenda.py:2285 | `"; ".join(d.why for d in defects) ; "no stopper of ours in time"` | LOGIC | -- | -- |
| agenda.py:2293 | `self.ctx is None` | LOGIC | -- | -- |
| agenda.py:2297 | `ksq is None` | LOGIC | -- | -- |
| agenda.py:2299 | `q.defects ; W.mate_defects(b, us, R.placement_of(q))` | LOGIC | -- | -- |
| agenda.py:2300 | `d.kind == "flight"` | CHESS FACT | [D1a] existing: flight; checker takeable (+2) | cases 75/75, DECISION 0 |
| agenda.py:2303 | `flights ; frm is not None` | LOGIC | -- | -- |
| agenda.py:2305 | `k is not None` | LOGIC | -- | -- |
| agenda.py:2307 | `why is None` | LOGIC | -- | -- |
| agenda.py:2308 | `b.piece_at(s) is not None ; b.piece_at(s).color == self.me` | RULE | -- | -- |
| agenda.py:2308 | `chess.SQUARES  in: heavy = [s for s in chess.SQUARES if b.piece_at(s) is not None and b.piece_at(s).color == self.me ...` | NAME | -- | -- |
| agenda.py:2308 | `b.piece_at(s).piece_type not in (chess.KING, chess.PAWN)` | CHESS FACT | [D9] existing: reasoned: the wall; reasoned: the measure they hold | cases 12/26, DECISION 0 |
| agenda.py:2309 | `b.is_check()` | RULE | -- | -- |
| agenda.py:2309 | `len(heavy) == 1 ; b.piece_at(heavy[0]).piece_type == chess.ROOK` | CHESS FACT | [D9] existing: reasoned: the wall; reasoned: the measure they hold | cases 12/26, DECISION 0 |
| agenda.py:2310 | `q.pattern.get("reach") ; q.pattern.get("move")` | LOGIC | -- | -- |
| agenda.py:2311 | `W.forcing_why(b, m, q.stake, us, mate_threats=True)` | LOGIC | -- | -- |
| agenda.py:2312 | `steps` | LOGIC | -- | -- |
| agenda.py:2313 | `passer is not None` | LOGIC | -- | -- |
| agenda.py:2315 | `k is not None` | LOGIC | -- | -- |
| agenda.py:2317 | `2  in: bound = [f"our {b.piece_at(g).symbol().upper()} on {GAME.loc_name(g)} is bound to the {b.piece_at(y).symbol()....` | LOGIC | -- | -- |
| agenda.py:2317 | `b.piece_at(g).piece_type == chess.KING` | CHESS FACT | [C1] PROPOSED: "names: king = the royal unit (Game.royal); excluded from the valuation" (VALUATION; reads material) | no printed signature: 8 rows |
| agenda.py:2318 | `bound ; 2  in: why = k["why"] + f" (their pawn on {GAME.loc_name(pw)} promotes in {passer[0]}: {passer[2]})" + ("; " ...` | LOGIC | -- | -- |
| agenda.py:2320 | `why is None` | LOGIC | -- | -- |
| agenda.py:2335 | `r.move not in by` | LOGIC | -- | -- |
| agenda.py:2339 | `r.trap is not None ; by[r.move].trap is None` | LOGIC | -- | -- |
| agenda.py:2341 | `r.fork is not None ; by[r.move].fork is None` | LOGIC | -- | -- |
| agenda.py:2347 | `f.status == "gap"` | LOGIC | -- | -- |
| agenda.py:2353 | `rs ; f.unforced ; f.text.startswith(('take its squares', 'fork')) ; 4  in: tail = f"{'routes' if f.text.startswith(('...` | LOGIC | -- | -- |
| agenda.py:2354 | `f.fixes ; f.routes` | LOGIC | -- | -- |
| agenda.py:2362 | `q.goal == "mate"` | CHESS FACT | [A1m] PROPOSED: "goal: mate (the rule's terminal)" (GOAL; reads movement, attack) | cases 100/100, DECISION 0 |
| agenda.py:2363 | `q.pattern.get("move")` | LOGIC | -- | -- |
| agenda.py:2366 | `m.to_square == t ; m.promotion` | CHESS FACT | [A1w] PROPOSED: "goal: win material" (GOAL; reads material, attack) | cases 83/86, DECISION 0 |
| agenda.py:2372 | `routes` | LOGIC | -- | -- |
| agenda.py:2373 | `self._chase(q)` | LOGIC | -- | -- |
| agenda.py:2381 | `bind is None` | LOGIC | -- | -- |
| agenda.py:2385 | `len(bind.replies) <= 4 ; 4  in: head = f" verify {r.san}: binds {bind.schema} ({len(bind.replies)} replies{': ' + ' /...` | LOGIC | -- | -- |
| agenda.py:2387 | `self.side_depth` | LOGIC | -- | -- |
| agenda.py:2387 | `bind.value >= BD.MATE_VALUE` | CHESS FACT | [A2] PROPOSED: "stake: the terminal (mate = 100)" (VALUATION; reads movement, attack) | cases 97/97, DECISION 0 |
| agenda.py:2388 | `bind.value >= BD.MATE_VALUE` | CHESS FACT (text only) | [A2] PROPOSED: "stake: the terminal (mate = 100)" (VALUATION; reads movement, attack) | cases 97/97, DECISION 0 |
| agenda.py:2391 | `theirs is not None` | LOGIC | -- | -- |
| agenda.py:2396 | `opp` | LOGIC | -- | -- |
| agenda.py:2397 | `self.side_depth ; self.chase_depth` | LOGIC | -- | -- |
| agenda.py:2397 | `bind.value >= self.min_stake` | CHESS FACT | [A7] PROPOSED: "goal: the need (what a claim must bring), propagated into the chase" (GOAL; reads material) | no printed signature: 15 rows |
| agenda.py:2397 | `bind.schema.startswith("delayed_") ; bind.schema.endswith("_d1")` | CHESS FACT | [D16] existing: refuted at the leaf | cases 4/4, DECISION 0 |
| agenda.py:2399 | `ref is not None` | LOGIC | -- | -- |
| agenda.py:2405 | `theirs is not None` | LOGIC | -- | -- |
| agenda.py:2405 | `theirs.value >= q.stake ; net <= 0` | CHESS FACT | [D15] existing: verify: no route binds | cases 27/26, DECISION 0 |
| agenda.py:2410 | `hits` | LOGIC | -- | -- |
| agenda.py:2411 | `self._chase(q)` | LOGIC | -- | -- |
| agenda.py:2413 | `len(routes) > 6 ; 6  in: self._kill(q, f"verify: no route binds ({', '.join(r.san for r in routes[:6])}{' ...' if len...` | LOGIC | -- | -- |
| agenda.py:2414 | `mates_only ; w ; "no mate in one" in w ; 300  in: self._kill(q, f"verify: no route binds ({', '.join(r.san for r in r...` | LOGIC | -- | -- |
| agenda.py:2415 | `"the future loses it" in w ; 300  in: self._kill(q, f"verify: no route binds ({', '.join(r.san for r in routes[:6])}{...` | LOGIC | -- | -- |
| agenda.py:2417 | `sort(... key=lambda h: (-h[3], len(h[1])))` | CHESS FACT | [D5] PROPOSED: "heuristic: the claim order at equal value (word > table, stake, shape key)" (HEURISTIC; reads material, attack) (beside existing: the shape key) | cases 1/1, DECISION 0 |
| agenda.py:2418 | `self._t(NEED_PROPAGATION, 3) ; self.chase_depth ; self._chase(q)` | LOGIC | -- | -- |
| agenda.py:2418 | `hits[0][3] < self.min_stake` | CHESS FACT | [A7] PROPOSED: "goal: the need (what a claim must bring), propagated into the chase" (GOAL; reads material) | no printed signature: 15 rows |
| agenda.py:2422 | `onet == net` | CHESS FACT | [D5] PROPOSED: "heuristic: the claim order at equal value (word > table, stake, shape key)" (HEURISTIC; reads material, attack) (beside existing: the shape key) | cases 1/1, DECISION 0 |
| agenda.py:2423 | `oopp ; opp` | CHESS FACT | [D5] PROPOSED: "heuristic: the claim order at equal value (word > table, stake, shape key)" (HEURISTIC; reads material, attack) (beside existing: the shape key) | cases 1/1, DECISION 0 |
| agenda.py:2424 | `len(oopp) > len(opp)` | CHESS FACT | [D5] PROPOSED: "heuristic: the claim order at equal value (word > table, stake, shape key)" (HEURISTIC; reads material, attack) (beside existing: the shape key) | cases 1/1, DECISION 0 |
| agenda.py:2426 | `otheirs is not None ; net > onet` | CHESS FACT | [D15] existing: verify: no route binds | cases 27/26, DECISION 0 |
| agenda.py:2428 | `q.goal == "save"` | CHESS FACT | [A1s] PROPOSED: "goal: save material (don't lose)" (GOAL; reads material, attack) (beside existing: in check) | cases 78/83, DECISION 0 |
| agenda.py:2428 | `bind.value >= BD.MATE_VALUE` | CHESS FACT | [A2] PROPOSED: "stake: the terminal (mate = 100)" (VALUATION; reads movement, attack) | cases 97/97, DECISION 0 |
| agenda.py:2434 | `"balance" in str(claim.lemma)` | CHESS FACT | [D26] existing: chase: step_after+step_least; delayed_balance_alive_d1 (+2) | no printed signature: 4 rows |
| agenda.py:2435 | `net == claim.value` | LOGIC | -- | -- |
| agenda.py:2437 | `q.pattern.get("inner_threat")` | CHESS FACT | [D2] existing: ... stands only as a balance, their ... wins ...; ... stands only as a balance and their ... (w... (+3) | cases 56/63, DECISION 0 |
| agenda.py:2439 | `self._better(key[0], key[1], claim)` | LOGIC | -- | -- |
| agenda.py:2445 | `q.reads is not None` | LOGIC | -- | -- |
| agenda.py:2446 | `bind is not None` | LOGIC | -- | -- |
| agenda.py:2448 | `bind.assignment.get(k)` | LOGIC | -- | -- |
| agenda.py:2456 | `self.ctx is None` | LOGIC | -- | -- |
| agenda.py:2456 | `kept < 1` | CHESS FACT | [A6] PROPOSED: "valuation: a gain is at least one unit of material" (VALUATION; reads material) (beside existing: no claim of ours and no material gain) | cases 80/80, DECISION 0 |
| agenda.py:2459 | `lk in self.ctx.theirs` | LOGIC | -- | -- |
| agenda.py:2462 | `THEIR_GAPS` | LOGIC | -- | -- |
| agenda.py:2465 | `g >= kept` | CHESS FACT | [D2] existing: ... stands only as a balance, their ... wins ...; ... stands only as a balance and their ... (w... (+3) | cases 56/63, DECISION 0 |
| agenda.py:2467 | `ag is not None ; q.status == "gap" ; max(... key=lambda q: q.stake)` | LOGIC | -- | -- |
| agenda.py:2468 | `q is not None` | LOGIC | -- | -- |
| agenda.py:2468 | `Claim('win', ...)` | CHESS FACT | [D2] existing: ... stands only as a balance, their ... wins ...; ... stands only as a balance and their ... (w... (+3) | cases 56/63, DECISION 0 |
| agenda.py:2469 | `out is None ; STUCK_LEAF` | LOGIC | -- | -- |
| agenda.py:2469 | `e_rb.is_check()` | RULE | -- | -- |
| agenda.py:2471 | `d["parry"] is None ; d["worth"] - d["price"] >= kept` | CHESS FACT | [D2] existing: ... stands only as a balance, their ... wins ...; ... stands only as a balance and their ... (w... (+3) | cases 56/63, DECISION 0 |
| agenda.py:2472 | `st` | LOGIC | -- | -- |
| agenda.py:2474 | `Claim('win', ...)` | CHESS FACT | [D2] existing: ... stands only as a balance, their ... wins ...; ... stands only as a balance and their ... (w... (+3) | cases 56/63, DECISION 0 |
| agenda.py:2493 | `key in self._leaf` | LOGIC | -- | -- |
| agenda.py:2498 | `fam not in PA.ORDER` | CHESS FACT | [D16] existing: refuted at the leaf | cases 4/4, DECISION 0 |
| agenda.py:2508 | `i1 is None ; i2 is None` | LOGIC | -- | -- |
| agenda.py:2509 | `i1 is None` | LOGIC | -- | -- |
| agenda.py:2512 | `reps` | LOGIC | -- | -- |
| agenda.py:2515 | `lo != t_root` | LOGIC | -- | -- |
| agenda.py:2521 | `self.leaf_all ; v1[i] == lo ; sorted(... key=lambda i: (v1[i], i))` | LOGIC | -- | -- |
| agenda.py:2531 | `us == "w"` | NAME | -- | -- |
| agenda.py:2532 | `top != v` | LOGIC | -- | -- |
| agenda.py:2542 | `e_rb.legal_moves` | RULE | -- | -- |
| agenda.py:2544 | `self.ctx is not None ; lk in self.ctx.theirs` | LOGIC | -- | -- |
| agenda.py:2548 | `ag.state.claims ; max(... key=lambda cl: cl.value)` | LOGIC | -- | -- |
| agenda.py:2549 | `self.ctx is not None` | LOGIC | -- | -- |
| agenda.py:2551 | `theirs is not None ; theirs.value >= gain_here` | CHESS FACT | [D16] existing: refuted at the leaf | cases 4/4, DECISION 0 |
| agenda.py:2552 | `first_ref ; dict(claim=theirs, line=line, w=theirs.value, gain=gain_here)` | LOGIC | -- | -- |
| agenda.py:2554 | `theirs is not None` | LOGIC | -- | -- |
| agenda.py:2555 | `more is not None` | LOGIC | -- | -- |
| agenda.py:2556 | `first_ref ; dict(claim=more, line=line, w=more.value, gain=gain_here)` | LOGIC | -- | -- |
| agenda.py:2560 | `held is None ; first_ref is not None` | LOGIC | -- | -- |
| agenda.py:2563 | `shown is None` | LOGIC | -- | -- |
| agenda.py:2565 | `refuted is not None` | LOGIC | -- | -- |
| agenda.py:2568 | `shown is not None` | LOGIC | -- | -- |
| agenda.py:2570 | `theirs` | LOGIC | -- | -- |
| agenda.py:2571 | `refuted is not None` | LOGIC | -- | -- |
| agenda.py:2580 | `self.side_depth` | LOGIC | -- | -- |
| agenda.py:2580 | `q.goal != "save"` | CHESS FACT | [D4] existing: none; the shape key | cases 17/16, DECISION 9 |
| agenda.py:2582 | `self.board.is_check()` | RULE | -- | -- |
| agenda.py:2589 | `r.move not in seen` | LOGIC | -- | -- |
| agenda.py:2592 | `routes` | LOGIC | -- | -- |
| agenda.py:2597 | `bind is not None` | LOGIC | -- | -- |
| agenda.py:2599 | `theirs is not None` | LOGIC | -- | -- |
| agenda.py:2602 | `sort(... key=lambda t: (-t[0], t[1], t[2].move.uci()))` | CHESS FACT | [D4] existing: none; the shape key | cases 17/16, DECISION 9 |
| agenda.py:2604 | `t[0] == net` | LOGIC | -- | -- |
| agenda.py:2605 | `t[0] < net` | LOGIC | -- | -- |
| agenda.py:2607 | `8  in: for n_, k_, rr, th_, g_ in rows[:8]: self.lines.append(f" {rr.san}: " + (f"their {th_.what} wins {th_.value} (...` | LOGIC | -- | -- |
| agenda.py:2608 | `th_ is not None` | LOGIC | -- | -- |
| agenda.py:2609 | `g_` | LOGIC | -- | -- |
| agenda.py:2610 | `len(rows) > 8` | LOGIC | -- | -- |
| agenda.py:2611 | `8  in: self.lines.append(f" ... and {len(rows) - 8} more")` | LOGIC | -- | -- |
| agenda.py:2612 | `theirs is not None` | LOGIC | -- | -- |
| agenda.py:2613 | `worse` | LOGIC | -- | -- |
| agenda.py:2615 | `wt[3] is not None ; 2  in: why += f" (after {wt[2].san}: " + (f"their {wt[3].what} wins {wt[3].value}" if wt[3] is no...` | LOGIC | -- | -- |
| agenda.py:2616 | `tied` | LOGIC | -- | -- |
| agenda.py:2617 | `2  in: why += f"; tied at net {net} with {', '.join(t[2].san for t in tied[:5])}: the shape key (fewest hanging, most...` | LOGIC | -- | -- |
| agenda.py:2619 | `theirs is not None` | LOGIC | -- | -- |
| agenda.py:2619 | `Claim('save', ...)` | CHESS FACT | [D4] existing: none; the shape key | cases 17/16, DECISION 9 |
| agenda.py:2625 | `self._better(net, 0, claim)` | LOGIC | -- | -- |
| agenda.py:2635 | `note ; ""` | LOGIC | -- | -- |
| agenda.py:2636 | `v is not None ; m ; 2  in: out.append(f"{n} {v if v is not None else (m.group(1) + '/' + m.group(2) if m else '-')}")` | LOGIC | -- | -- |
| agenda.py:2640 | `bind.schema != "mate_in_two"` | LOGIC | -- | -- |
| agenda.py:2641 | `bind.note` | LOGIC | -- | -- |
| agenda.py:2649 | `got` | LOGIC | -- | -- |
| agenda.py:2659 | `note ; v is None` | LOGIC | -- | -- |
| agenda.py:2660 | `self.side_depth` | LOGIC | -- | -- |
| agenda.py:2661 | `notes` | LOGIC | -- | -- |
| agenda.py:2668 | `any(self._mates(c, mv) for mv in c.legal_moves)` | RULE | -- | -- |
| agenda.py:2671 | `esc ; 0 < len(esc) <= 40 ; notes ; 4  in: return f"{r.san} fails" + (f" ({'; '.join(notes)})" if notes else "") + (f"...` | LOGIC | -- | -- |
| agenda.py:2680 | `new_question('cover', ...)` | CHESS FACT | [D1e] PROPOSED: "goal: mate at distance two (the reach defect)" (GOAL; reads movement, attack, tempo) | cases 23/23, DECISION 0 |
| agenda.py:2684 | `why` | LOGIC | -- | -- |
| agenda.py:2687 | `getattr(q, "_arr_root", False)` | LOGIC | -- | -- |
| agenda.py:2688 | `af is not None` | LOGIC | -- | -- |
| agenda.py:2689 | `r.move not in steps` | LOGIC | -- | -- |
| agenda.py:2691 | `forcing ; quiet ; 4  in: self.lines.append(f" -> Q{f.id} {f.text}: " + (", ".join(f"{r.san} ({r.why})" for r in forci...` | LOGIC | -- | -- |
| agenda.py:2693 | `forcing` | LOGIC | -- | -- |
| agenda.py:2694 | `self._chase(q)` | LOGIC | -- | -- |
| agenda.py:2696 | `quiet` | LOGIC | -- | -- |
| agenda.py:2713 | `their is not None` | LOGIC | -- | -- |
| agenda.py:2714 | `wt` | LOGIC | -- | -- |
| agenda.py:2716 | `local is not None` | LOGIC | -- | -- |
| agenda.py:2717 | `self._chase_q is not None` | LOGIC | -- | -- |
| agenda.py:2717 | `self._chase_q.goal == "save" ; "checkers" in self._chase_q.pattern` | CHESS FACT | [B2] existing: a second check of theirs in a row | cases 30/56, DECISION 0 |
| agenda.py:2719 | `hit is None` | LOGIC | -- | -- |
| agenda.py:2720 | `wt ; len(k_) == 3 ; k_[0] == pk ; k_[1] ; ctx.memo.get((pk, False, 0)) ; next((h for k_, h in ctx.memo.items() if len...` | LOGIC | -- | -- |
| agenda.py:2721 | `hit is None ; gk is not None` | LOGIC | -- | -- |
| agenda.py:2723 | `hit is None ; wt` | LOGIC | -- | -- |
| agenda.py:2726 | `h is not None ; k_[3] != gk ; h["need"] <= need ; h["claim"] is not None ; h["value"] >= need` | LOGIC | -- | -- |
| agenda.py:2726 | `need >= BD.MATE_VALUE` | CHESS FACT | [A7] PROPOSED: "goal: the need (what a claim must bring), propagated into the chase" (GOAL; reads material) | no printed signature: 15 rows |
| agenda.py:2729 | `hit is not None ; hit["need"] <= need` | LOGIC | -- | -- |
| agenda.py:2732 | `pk in ctx.active ; CYCLE` | LOGIC | -- | -- |
| agenda.py:2732 | `pk == ctx.root_pk ; pk in ctx.history` | RULE | -- | -- |
| agenda.py:2733 | `CYCLE ; wt` | LOGIC | -- | -- |
| agenda.py:2735 | `cyc is not None` | LOGIC | -- | -- |
| agenda.py:2737 | `wt` | LOGIC | -- | -- |
| agenda.py:2738 | `ctx.left <= 0 ; self.chase_depth >= DEPTH_FUSE ; ctx.out_of_time() ; ctx.finishing(ctx.route_depth)` | LOGIC | -- | -- |
| agenda.py:2739 | `ctx.left > 0` | LOGIC | -- | -- |
| agenda.py:2742 | `COST_JUDGE ; self._judge_node(fen)` | LOGIC | -- | -- |
| agenda.py:2745 | `self._route_move is not None ; self._reply_san` | LOGIC | -- | -- |
| agenda.py:2748 | `self._chase_q is not None` | LOGIC | -- | -- |
| agenda.py:2748 | `self._chase_q.goal == "save" ; "checkers" in self._chase_q.pattern` | CHESS FACT | [B2] existing: a second check of theirs in a row | cases 30/56, DECISION 0 |
| agenda.py:2749 | `evading ; reset` | CHESS FACT | [B2] existing: a second check of theirs in a row | cases 30/56, DECISION 0 |
| agenda.py:2750 | `since is None` | CHESS FACT | [B1] existing: the chase clock | cases 5/9, DECISION 0 |
| agenda.py:2751 | `local is not None` | LOGIC | -- | -- |
| agenda.py:2758 | `ag.best ; 2  in: best = ag.best[2] if ag.best and (ag.best[0] > 0 or ag.best[2].kept) else None` | LOGIC | -- | -- |
| agenda.py:2758 | `ag.best[2].kept` | CHESS FACT | [A10] PROPOSED: "goal: save = keep the balance (target 0)" (GOAL; reads material, tempo) | cases 47/56, DECISION 0 |
| agenda.py:2758 | `ag.best[0] > 0` | CHESS FACT | [A6] PROPOSED: "valuation: a gain is at least one unit of material" (VALUATION; reads material) (beside existing: no claim of ours and no material gain) | cases 80/80, DECISION 0 |
| agenda.py:2759 | `any(q.reads is None for q in ag.all)` | LOGIC | -- | -- |
| agenda.py:2760 | `q.status == "verified" ; q.claimed is not None` | LOGIC | -- | -- |
| agenda.py:2761 | `ag.local is not None ; ag.local["full"]` | LOGIC | -- | -- |
| agenda.py:2762 | `best ; ctx.budget_hits > hits0 ; ctx.chain_hits > chain0` | LOGIC | -- | -- |
| agenda.py:2763 | `c.kind == "promote"` | CHESS FACT | [A1p] PROPOSED: "goal: promote (the rule)" (GOAL; reads movement, tempo) | cases 48/49, DECISION 0 |
| agenda.py:2764 | `best is not None` | LOGIC | -- | -- |
| agenda.py:2765 | `out["cut"] ; best is not None ; stayed is not None` | LOGIC | -- | -- |
| agenda.py:2766 | `cfid is not None ; self._cur is not None` | LOGIC | -- | -- |
| agenda.py:2767 | `self._cur is not None` | LOGIC | -- | -- |
| agenda.py:2770 | `out["cut"] ; out["chain"]` | LOGIC | -- | -- |
| agenda.py:2771 | `stayed is not None ; wt` | LOGIC | -- | -- |
| agenda.py:2772 | `stayed is not None` | LOGIC | -- | -- |
| agenda.py:2778 | `fid is not None ; self._rf is not None ; fid not in self._rf` | LOGIC | -- | -- |
| agenda.py:2786 | `CARRY ; self.ctx.carry_r ; key[1].endswith(":refuter")` | LOGIC | -- | -- |
| agenda.py:2789 | `hit is None` | LOGIC | -- | -- |
| agenda.py:2792 | `self._floor is not None ; self.chase_depth == 0` | LOGIC | -- | -- |
| agenda.py:2794 | `hit["closed"]` | LOGIC | -- | -- |
| agenda.py:2794 | `hit["g"] < need ; gstake ; hit["g"] < gstake` | CHESS FACT | [A7] PROPOSED: "goal: the need (what a claim must bring), propagated into the chase" (GOAL; reads material) | no printed signature: 15 rows |
| agenda.py:2802 | `CARRY ; key[1].endswith(":refuter")` | LOGIC | -- | -- |
| agenda.py:2806 | `old is None ; g > old["g"]` | LOGIC | -- | -- |
| agenda.py:2814 | `cr["lemma"]` | LOGIC | -- | -- |
| agenda.py:2816 | `cr.get("material")` | CHESS FACT | [D8c] existing: the promotion is not claimed | cases 23/23, DECISION 0 |
| agenda.py:2827 | `self.chase_depth == 0 ; self.ctx.deadline is not None` | LOGIC | -- | -- |
| agenda.py:2833 | `COST_JUDGE` | LOGIC | -- | -- |
| agenda.py:2834 | `res is None` | LOGIC | -- | -- |
| agenda.py:2836 | `COST_JUDGE ; CYCLE` | LOGIC | -- | -- |
| agenda.py:2838 | `ESTIMATE ; cheap ; promote ; res.get("gap") is not None` | LOGIC | -- | -- |
| agenda.py:2841 | `cheap` | LOGIC | -- | -- |
| agenda.py:2845 | `ctx.hand is not None ; ctx.abort ; ctx.route_depth == ctx.hand ; cheap` | LOGIC | -- | -- |
| agenda.py:2856 | `promote` | LOGIC | -- | -- |
| agenda.py:2857 | `res.get("gap") is None ; res.get("reused")` | LOGIC | -- | -- |
| agenda.py:2858 | `res.get("gap") is None` | LOGIC | -- | -- |
| agenda.py:2859 | `fid is not None` | LOGIC | -- | -- |
| agenda.py:2863 | `key in ctx.gap_facts` | LOGIC | -- | -- |
| agenda.py:2865 | `q is not None` | LOGIC | -- | -- |
| agenda.py:2866 | `q is not None` | LOGIC | -- | -- |
| agenda.py:2867 | `q is not None` | LOGIC | -- | -- |
| agenda.py:2892 | `promote ; override` | LOGIC | -- | -- |
| agenda.py:2894 | `cheap` | LOGIC | -- | -- |
| agenda.py:2896 | `cq is not None ; cq.pattern.get("target") is not None` | LOGIC | -- | -- |
| agenda.py:2896 | `cq.goal == "win"` | CHESS FACT | [A1w] PROPOSED: "goal: win material" (GOAL; reads material, attack) | cases 83/86, DECISION 0 |
| agenda.py:2898 | `gk is None` | LOGIC | -- | -- |
| agenda.py:2899 | `gk is not None` | LOGIC | -- | -- |
| agenda.py:2902 | `hit is None ; gk is not None` | LOGIC | -- | -- |
| agenda.py:2904 | `hit is None` | LOGIC | -- | -- |
| agenda.py:2907 | `g != gk ; h is not None ; h["gap"] is None ; h["killed"] is None` | LOGIC | -- | -- |
| agenda.py:2910 | `hit is not None` | LOGIC | -- | -- |
| agenda.py:2911 | `hit["gap"] is None ; hit["killed"] is None ; gstake` | LOGIC | -- | -- |
| agenda.py:2911 | `hit.get("closed_min", gstake) < gstake ; hit.get("stake_checked", 0) < gstake` | CHESS FACT | [D3] existing: mate in one; nothing of theirs | cases 70/81, DECISION 0 |
| agenda.py:2912 | `stale ; hit["gap"] is not None ; hit["gap"].startswith("chain stop") ; hit.get("chain_state") != self._chain_state()` | LOGIC | -- | -- |
| agenda.py:2913 | `hit["gap"] is None ; stale` | LOGIC | -- | -- |
| agenda.py:2915 | `ctx.deadline is not None` | LOGIC | -- | -- |
| agenda.py:2916 | `stale ; slice_now <= hit.get("cut_slice", 0) if slice_now is not None else ctx.left <= hit["cut_left"]` | LOGIC | -- | -- |
| agenda.py:2927 | `c_py` | LOGIC | -- | -- |
| agenda.py:2931 | `c_rb.legal_moves` | RULE | -- | -- |
| agenda.py:2932 | `c_rb.is_checkmate()` | RULE | -- | -- |
| agenda.py:2933 | `BD.MATE_VALUE  in: res.update(value=BD.MATE_VALUE, parts=[f"{san} mates"])` | CHESS FACT | [A9] existing: they mate us; stalemate (+1) | cases 1/1, DECISION 0 |
| agenda.py:2942 | `6  in: rule_of = {t[0]: t[6] for t in table}` | LOGIC | -- | -- |
| agenda.py:2943 | `7  in: pos_of = {t[0]: t[7] for t in table}` | LOGIC | -- | -- |
| agenda.py:2954 | `promote ; override ; uci in override` | LOGIC | -- | -- |
| agenda.py:2955 | `sk is not None` | LOGIC | -- | -- |
| agenda.py:2959 | `2  in: self._log(lkey, uci, "standing", dbal, None, sk[2], res["killed"])` | LOGIC | -- | -- |
| agenda.py:2961 | `res["killed"] is not None` | LOGIC | -- | -- |
| agenda.py:2968 | `BD.MATE_VALUE  in: closed_min = BD.MATE_VALUE` | CHESS FACT | [A2] PROPOSED: "stake: the terminal (mate = 100)" (VALUATION; reads movement, attack) | cases 97/97, DECISION 0 |
| agenda.py:2970 | `self.chase_depth == 0 ; cheap ; override ; self._judge_on()` | LOGIC | -- | -- |
| agenda.py:2972 | `hit is not None` | LOGIC | -- | -- |
| agenda.py:2976 | `tally[0] is not None ; done ; tally[0][0] in kid_cost` | LOGIC | -- | -- |
| agenda.py:2983 | `kid_cost is not None` | LOGIC | -- | -- |
| agenda.py:2986 | `override ; uci in override` | LOGIC | -- | -- |
| agenda.py:2990 | `f not in self._rf` | LOGIC | -- | -- |
| agenda.py:2992 | `pre is not None ; uci in pre` | LOGIC | -- | -- |
| agenda.py:2994 | `val < BD.MATE_VALUE` | CHESS FACT | [A2] PROPOSED: "stake: the terminal (mate = 100)" (VALUATION; reads movement, attack) | cases 97/97, DECISION 0 |
| agenda.py:2999 | `override` | LOGIC | -- | -- |
| agenda.py:3000 | `cr is not None` | LOGIC | -- | -- |
| agenda.py:3001 | `cr["closed"]` | LOGIC | -- | -- |
| agenda.py:3001 | `cr["g"] < BD.MATE_VALUE` | CHESS FACT | [A2] PROPOSED: "stake: the terminal (mate = 100)" (VALUATION; reads movement, attack) | cases 97/97, DECISION 0 |
| agenda.py:3005 | `rule_of[uci] == "dead"` | RULE | -- | -- |
| agenda.py:3006 | `chk` | RULE | -- | -- |
| agenda.py:3009 | `promote` | LOGIC | -- | -- |
| agenda.py:3009 | `rule_of[uci] == "mate"` | RULE | -- | -- |
| agenda.py:3009 | `BD.MATE_VALUE  in: closed = (BD.MATE_VALUE, "mate in one") if rule_of[uci] == "mate" else None if promote else self._...` | CHESS FACT | [D3] existing: mate in one; nothing of theirs | cases 70/81, DECISION 0 |
| agenda.py:3010 | `closed is not None ; promote` | LOGIC | -- | -- |
| agenda.py:3012 | `closed is not None` | LOGIC | -- | -- |
| agenda.py:3014 | `val < BD.MATE_VALUE` | CHESS FACT | [A2] PROPOSED: "stake: the terminal (mate = 100)" (VALUATION; reads movement, attack) | cases 97/97, DECISION 0 |
| agenda.py:3023 | `promote` | LOGIC | -- | -- |
| agenda.py:3023 | `dbal < self.promote_stake` | CHESS FACT | [D8d] existing: the ... defect stands (...) and no fix is new... | cases 25/27, DECISION 0 |
| agenda.py:3025 | `why is not None` | LOGIC | -- | -- |
| agenda.py:3030 | `cheap` | LOGIC | -- | -- |
| agenda.py:3034 | `promote` | LOGIC | -- | -- |
| agenda.py:3034 | `R.STAKE_PROMOTE  in: need = min(R.STAKE_PROMOTE, 1 - dbal) if promote and dbal <= 0 else R.STAKE_PROMOTE if promote e...` | CHESS FACT | [A3] PROPOSED: "stake: promotion (queen less pawn = 8)" (VALUATION; reads material, movement) | cases 35/35, DECISION 0 |
| agenda.py:3034 | `dbal <= 0 ; dbal < target ; BD.MATE_VALUE  in: need = min(R.STAKE_PROMOTE, 1 - dbal) if promote and dbal <= 0 else R....` | CHESS FACT | [A7] PROPOSED: "goal: the need (what a claim must bring), propagated into the chase" (GOAL; reads material) | no printed signature: 15 rows |
| agenda.py:3038 | `gk is not None ; override` | LOGIC | -- | -- |
| agenda.py:3039 | `loc is not None` | LOGIC | -- | -- |
| agenda.py:3040 | `P0.is_capture(self._route_move.uci()) ; chess.Move.from_uci(uci).to_square == self._route_move.to_square` | CHESS FACT | [B3] existing: tempo arithmetic; tempo | cases 61/74, DECISION 0 |
| agenda.py:3042 | `inn is None` | LOGIC | -- | -- |
| agenda.py:3048 | `promote` | LOGIC | -- | -- |
| agenda.py:3050 | `inn["value"] >= BD.MATE_VALUE` | CHESS FACT | [A2] PROPOSED: "stake: the terminal (mate = 100)" (VALUATION; reads movement, attack) | cases 97/97, DECISION 0 |
| agenda.py:3051 | `BD.MATE_VALUE  in: g, part = BD.MATE_VALUE, f"{inn['claim'].what} ({inn['claim'].line}; value {inn['claim'].value})"` | CHESS FACT | [A2] PROPOSED: "stake: the terminal (mate = 100)" (VALUATION; reads movement, attack) | cases 97/97, DECISION 0 |
| agenda.py:3052 | `pc is not None` | LOGIC | -- | -- |
| agenda.py:3053 | `R.STAKE_PROMOTE  in: g, part = R.STAKE_PROMOTE, f"{pc.what} ({pc.line}; value {pc.value}, lemma {pc.lemma})"` | CHESS FACT | [A3] PROPOSED: "stake: promotion (queen less pawn = 8)" (VALUATION; reads material, movement) | cases 35/35, DECISION 0 |
| agenda.py:3054 | `inn["claim"] is not None` | LOGIC | -- | -- |
| agenda.py:3054 | `inn["value"] >= self.promote_stake` | CHESS FACT | [D8c] existing: the promotion is not claimed | cases 23/23, DECISION 0 |
| agenda.py:3055 | `R.STAKE_PROMOTE  in: g, part = R.STAKE_PROMOTE, f"{inn['claim'].what} ({inn['claim'].line}; value {inn['claim'].value...` | CHESS FACT | [A3] PROPOSED: "stake: promotion (queen less pawn = 8)" (VALUATION; reads material, movement) | cases 35/35, DECISION 0 |
| agenda.py:3056 | `dbal >= self.promote_stake` | CHESS FACT | [D8c] existing: the promotion is not claimed | cases 23/23, DECISION 0 |
| agenda.py:3058 | `dbal <= 0 ; inn["claim"] is not None ; inn["value"] >= 1 - dbal` | CHESS FACT | [D8c] existing: the promotion is not claimed | cases 23/23, DECISION 0 |
| agenda.py:3061 | `inn.get("cut") ; inn.get("chain")` | LOGIC | -- | -- |
| agenda.py:3065 | `inn.get("cut")` | LOGIC | -- | -- |
| agenda.py:3066 | `inn.get("cut")` | LOGIC | -- | -- |
| agenda.py:3070 | `dbal <= 0` | CHESS FACT | [D8c] existing: the promotion is not claimed | cases 23/23, DECISION 0 |
| agenda.py:3075 | `pc is not None ; g == dbal + inn["value"] ; res.get("material", False) ; pc is None ; inn["value"] < BD.MATE_VALUE` | CHESS FACT | [D8c] existing: the promotion is not claimed | cases 23/23, DECISION 0 |
| agenda.py:3077 | `pc is not None` | LOGIC | -- | -- |
| agenda.py:3080 | `self._chase_q is not None` | LOGIC | -- | -- |
| agenda.py:3080 | `target < 1` | CHESS FACT | [A10] PROPOSED: "goal: save = keep the balance (target 0)" (GOAL; reads material, tempo) | cases 47/56, DECISION 0 |
| agenda.py:3080 | `self._chase_q.pattern.get("value", 0) >= BD.MATE_VALUE` | CHESS FACT | [D2] existing: ... stands only as a balance, their ... wins ...; ... stands only as a balance and their ... (w... (+3) | cases 56/63, DECISION 0 |
| agenda.py:3081 | `mate_save` | LOGIC | -- | -- |
| agenda.py:3081 | `inn["value"] < BD.MATE_VALUE` | CHESS FACT | [A2] PROPOSED: "stake: the terminal (mate = 100)" (VALUATION; reads movement, attack) | cases 97/97, DECISION 0 |
| agenda.py:3081 | `dbal >= 1 ; dbal >= target` | CHESS FACT | [D2] existing: ... stands only as a balance, their ... wins ...; ... stands only as a balance and their ... (w... (+3) | cases 56/63, DECISION 0 |
| agenda.py:3083 | `P2.is_check()` | RULE | -- | -- |
| agenda.py:3083 | `target < 1` | CHESS FACT | [A10] PROPOSED: "goal: save = keep the balance (target 0)" (GOAL; reads material, tempo) | cases 47/56, DECISION 0 |
| agenda.py:3083 | `inn["value"] < BD.MATE_VALUE` | CHESS FACT | [A2] PROPOSED: "stake: the terminal (mate = 100)" (VALUATION; reads movement, attack) | cases 97/97, DECISION 0 |
| agenda.py:3083 | `dbal >= target ; inn["claim"] is None` | CHESS FACT | [D2] existing: ... stands only as a balance, their ... wins ...; ... stands only as a balance and their ... (w... (+3) | cases 56/63, DECISION 0 |
| agenda.py:3089 | `static` | LOGIC | -- | -- |
| agenda.py:3089 | `P2.is_check()` | RULE | -- | -- |
| agenda.py:3091 | `claim_their is not None` | LOGIC | -- | -- |
| agenda.py:3093 | `claim_their is not None` | LOGIC | -- | -- |
| agenda.py:3093 | `dbal - t_their < 1` | CHESS FACT | [D2] existing: ... stands only as a balance, their ... wins ...; ... stands only as a balance and their ... (w... (+3) | cases 56/63, DECISION 0 |
| agenda.py:3099 | `inn2 is None` | LOGIC | -- | -- |
| agenda.py:3105 | `inn.get("cut") ; inn2.get("cut") ; inn.get("chain") ; inn2.get("chain")` | LOGIC | -- | -- |
| agenda.py:3106 | `pc is not None` | LOGIC | -- | -- |
| agenda.py:3107 | `pc is None ; st == "killed"` | LOGIC | -- | -- |
| agenda.py:3108 | `pc is None ; st != "killed"` | LOGIC | -- | -- |
| agenda.py:3111 | `lost ; unsure ; inn2.get("claim") is not None` | LOGIC | -- | -- |
| agenda.py:3111 | `inn2["claim"].kind in ("save", "win") ; inn2["value"] >= 0` | CHESS FACT | [D2] existing: ... stands only as a balance, their ... wins ...; ... stands only as a balance and their ... (w... (+3) | cases 56/63, DECISION 0 |
| agenda.py:3112 | `parry_note` | LOGIC | -- | -- |
| agenda.py:3115 | `PARRY_PASS ; held` | LOGIC | -- | -- |
| agenda.py:3118 | `pc.move` | LOGIC | -- | -- |
| agenda.py:3119 | `mv is not None` | LOGIC | -- | -- |
| agenda.py:3119 | `mv in RB.RBoard(P2).legal_moves` | RULE | -- | -- |
| agenda.py:3120 | `P3 is None` | LOGIC | -- | -- |
| agenda.py:3120 | `P3.is_check() ; P3.legal_count() == 0` | RULE | -- | -- |
| agenda.py:3125 | `P3w is not None` | LOGIC | -- | -- |
| agenda.py:3127 | `THEIR_GAPS` | LOGIC | -- | -- |
| agenda.py:3128 | `cl3 is not None` | LOGIC | -- | -- |
| agenda.py:3128 | `dbal3 - t3 < 1` | CHESS FACT | [D2] existing: ... stands only as a balance, their ... wins ...; ... stands only as a balance and their ... (w... (+3) | cases 56/63, DECISION 0 |
| agenda.py:3132 | `g3` | LOGIC | -- | -- |
| agenda.py:3132 | `dbal3 - max(t3, 0) - g3 < 1` | CHESS FACT | [D2] existing: ... stands only as a balance, their ... wins ...; ... stands only as a balance and their ... (w... (+3) | cases 56/63, DECISION 0 |
| agenda.py:3139 | `lost` | LOGIC | -- | -- |
| agenda.py:3144 | `unsure` | LOGIC | -- | -- |
| agenda.py:3144 | `dbal - unparried - sum(cl.value for cl in unsure) < 1` | CHESS FACT | [D2] existing: ... stands only as a balance, their ... wins ...; ... stands only as a balance and their ... (w... (+3) | cases 56/63, DECISION 0 |
| agenda.py:3145 | `max(... key=lambda cl: cl.value)` | LOGIC | -- | -- |
| agenda.py:3146 | `getattr(worst, "parry_note", None) ; "the save question was cut, not killed"` | LOGIC | -- | -- |
| agenda.py:3151 | `dbal - unparried < 1` | CHESS FACT | [D2] existing: ... stands only as a balance, their ... wins ...; ... stands only as a balance and their ... (w... (+3) | cases 56/63, DECISION 0 |
| agenda.py:3152 | `lost ; max(... key=lambda cl: cl.value)` | LOGIC | -- | -- |
| agenda.py:3159 | `THEIR_GAPS ; res["gap"] is None ; res["killed"] is None` | LOGIC | -- | -- |
| agenda.py:3161 | `g_o` | LOGIC | -- | -- |
| agenda.py:3161 | `dbal - unparried - g_o < 1` | CHESS FACT | [D2] existing: ... stands only as a balance, their ... wins ...; ... stands only as a balance and their ... (w... (+3) | cases 56/63, DECISION 0 |
| agenda.py:3168 | `STUCK_LEAF ; res["gap"] is None ; res["killed"] is None` | LOGIC | -- | -- |
| agenda.py:3168 | `dbal - unparried >= 1` | CHESS FACT | [D2] existing: ... stands only as a balance, their ... wins ...; ... stands only as a balance and their ... (w... (+3) | cases 56/63, DECISION 0 |
| agenda.py:3169 | `d["parry"] is None ; dbal - unparried - (d["worth"] - d["price"]) < 1` | CHESS FACT | [D2] existing: ... stands only as a balance, their ... wins ...; ... stands only as a balance and their ... (w... (+3) | cases 56/63, DECISION 0 |
| agenda.py:3170 | `st` | LOGIC | -- | -- |
| agenda.py:3181 | `inn.get("cut") ; inn.get("chain")` | LOGIC | -- | -- |
| agenda.py:3181 | `inn["value"] < BD.MATE_VALUE` | CHESS FACT | [A2] PROPOSED: "stake: the terminal (mate = 100)" (VALUATION; reads movement, attack) | cases 97/97, DECISION 0 |
| agenda.py:3181 | `dbal + inn["value"] < target` | CHESS FACT | [A7] PROPOSED: "goal: the need (what a claim must bring), propagated into the chase" (GOAL; reads material) | no printed signature: 15 rows |
| agenda.py:3185 | `inn.get("cut")` | LOGIC | -- | -- |
| agenda.py:3186 | `inn.get("cut")` | LOGIC | -- | -- |
| agenda.py:3189 | `inn["value"] < BD.MATE_VALUE` | CHESS FACT | [A2] PROPOSED: "stake: the terminal (mate = 100)" (VALUATION; reads movement, attack) | cases 97/97, DECISION 0 |
| agenda.py:3189 | `dbal + inn["value"] < target` | CHESS FACT | [A7] PROPOSED: "goal: the need (what a claim must bring), propagated into the chase" (GOAL; reads material) | no printed signature: 15 rows |
| agenda.py:3190 | `inn["claim"]` | LOGIC | -- | -- |
| agenda.py:3191 | `inn.get("cut") ; inn.get("chain")` | LOGIC | -- | -- |
| agenda.py:3191 | `inn["claim"] is None` | CHESS FACT | [D25] existing: standing kill | cases 28/26, DECISION 0 |
| agenda.py:3192 | `inn.get("local") is not None` | LOGIC | -- | -- |
| agenda.py:3195 | `inn["value"] >= need` | CHESS FACT | [A7] PROPOSED: "goal: the need (what a claim must bring), propagated into the chase" (GOAL; reads material) | no printed signature: 15 rows |
| agenda.py:3196 | `static ; cl ; self._t(NEED_PROPAGATION, 3)` | LOGIC | -- | -- |
| agenda.py:3196 | `inn["value"] >= BD.MATE_VALUE ; BD.MATE_VALUE  in: gains.append(BD.MATE_VALUE if inn["value"] >= BD.MATE_VALUE else m...` | CHESS FACT | [A2] PROPOSED: "stake: the terminal (mate = 100)" (VALUATION; reads movement, attack) | cases 97/97, DECISION 0 |
| agenda.py:3197 | `cl ; claim_their ; parry_note` | LOGIC | -- | -- |
| agenda.py:3198 | `cl` | LOGIC | -- | -- |
| agenda.py:3199 | `cl` | LOGIC | -- | -- |
| agenda.py:3200 | `cl` | LOGIC | -- | -- |
| agenda.py:3202 | `res["gap"] is None` | LOGIC | -- | -- |
| agenda.py:3203 | `cheap ; res.get("pending") ; res["killed"] is None` | LOGIC | -- | -- |
| agenda.py:3208 | `res["killed"] is not None ; kill_uci is not None` | LOGIC | -- | -- |
| agenda.py:3210 | `res["killed"] is None ; res["gap"] is None` | LOGIC | -- | -- |
| agenda.py:3211 | `gains` | LOGIC | -- | -- |
| agenda.py:3212 | `res["gap"] is not None` | LOGIC | -- | -- |
| agenda.py:3215 | `res["gap"].startswith("out of time")` | LOGIC | -- | -- |
| agenda.py:3216 | `ctx.t0 is not None ; 3  in: res["timed"], res["mark"] = True, round(time.monotonic() - ctx.t0, 3) if ctx.t0 is not No...` | LOGIC | -- | -- |
| agenda.py:3231 | `res_k is None ; res_k["killed"] is None ; res_k.get("refuter") ; res_k.get("gap")` | LOGIC | -- | -- |
| agenda.py:3236 | `j == i ; res2 is None ; res2["killed"] is None ; res2.get("gap") ; res2.get("fixed_by") ; res2.get("refuter")` | LOGIC | -- | -- |
| agenda.py:3236 | `b.is_capture(r2.move)` | CHESS FACT | [D10] existing: refuter fix | cases 0/0, DECISION 0 |
| agenda.py:3237 | `r2.move.to_square != ref.from_square ; r_k.move.to_square == ref.from_square` | CHESS FACT | [D10] existing: refuter fix | cases 0/0, DECISION 0 |
| agenda.py:3240 | `saved < pool` | LOGIC | -- | -- |
| agenda.py:3245 | `got is not None` | LOGIC | -- | -- |
| agenda.py:3247 | `saved > 0` | LOGIC | -- | -- |
| agenda.py:3248 | `new is None` | LOGIC | -- | -- |
| agenda.py:3250 | `new["killed"] is not None ; new["gap"] is not None` | LOGIC | -- | -- |
| agenda.py:3250 | `new["value"] < 1` | CHESS FACT | [A6] PROPOSED: "valuation: a gain is at least one unit of material" (VALUATION; reads material) (beside existing: no claim of ours and no material gain) | cases 80/80, DECISION 0 |
| agenda.py:3266 | `x not in c_rb.legal_moves ; c_rb.is_capture(x)` | RULE | -- | -- |
| agenda.py:3266 | `x.to_square != r2.move.to_square` | CHESS FACT | [D10] existing: refuter fix | cases 0/0, DECISION 0 |
| agenda.py:3271 | `r_k.move not in c2_rb.legal_moves` | RULE | -- | -- |
| agenda.py:3274 | `any(mv.from_square == x.to_square and mv.to_square == ref.to_square for mv in c3_rb.legal_moves)` | CHESS FACT | [D10] existing: refuter fix | cases 0/0, DECISION 0 |
| agenda.py:3276 | `c3_rb.piece_at(s) is not None ; c3_rb.piece_at(s).color != me` | RULE | -- | -- |
| agenda.py:3276 | `c3_rb.piece_at(s).piece_type != chess.KING` | CHESS FACT | [C1] PROPOSED: "names: king = the royal unit (Game.royal); excluded from the valuation" (VALUATION; reads material) | no printed signature: 8 rows |
| agenda.py:3277 | `hits` | LOGIC | -- | -- |
| agenda.py:3277 | `max(... key=lambda s: (W.base.worth(c3_rb.piece_at(s)), -s))` | CHESS FACT | [A4] PROPOSED: "valuation: material (unit worth)" (VALUATION; reads material) | cases 82/85, DECISION 0 |
| agenda.py:3278 | `tgt is not None` | LOGIC | -- | -- |
| agenda.py:3282 | `pc_t is not None ; new_question('win', ...)` | CHESS FACT | [D10] existing: refuter fix | cases 0/0, DECISION 0 |
| agenda.py:3284 | `Agenda(..., min_stake=1) (the re-asked route must gain one unit)` | CHESS FACT | [A6] PROPOSED: "valuation: a gain is at least one unit of material" (VALUATION; reads material) (beside existing: no claim of ours and no material gain) | cases 80/80, DECISION 0 |
| agenda.py:3287 | `res3["gap"] is not None ; res3["killed"] is not None` | LOGIC | -- | -- |
| agenda.py:3287 | `res3["value"] < 1 - dbal` | CHESS FACT | [D10] existing: refuter fix | cases 0/0, DECISION 0 |
| agenda.py:3291 | `2  in: part = f"after {san_x}: {r_k.san} asked again on the board after the exchange ({'; '.join(res3['parts'][:2])})...` | LOGIC | -- | -- |
| agenda.py:3298 | `r is not None` | LOGIC | -- | -- |
| agenda.py:3299 | `r is not None ; self.ctx.t0 is not None ; 3  in: self.ctx.resume.append(dict(question=R.question_key(q), text=q.text,...` | LOGIC | -- | -- |
| agenda.py:3315 | `res["gap"].startswith("chain stop")` | LOGIC | -- | -- |
| agenda.py:3321 | `inn is not None ; inn.get("chain") ; inn.get("cut")` | LOGIC | -- | -- |
| agenda.py:3323 | `self.ctx.out_of_time()` | LOGIC | -- | -- |
| agenda.py:3325 | `self.ctx.judged` | LOGIC | -- | -- |
| agenda.py:3328 | `inner` | LOGIC | -- | -- |
| agenda.py:3335 | `q is not None` | LOGIC | -- | -- |
| agenda.py:3336 | `pawn is None` | LOGIC | -- | -- |
| agenda.py:3336 | `m.from_square == pawn` | CHESS FACT | [D8d] existing: the ... defect stands (...) and no fix is new... | cases 25/27, DECISION 0 |
| agenda.py:3340 | `pc is None` | LOGIC | -- | -- |
| agenda.py:3340 | `pc.color != self.me` | RULE | -- | -- |
| agenda.py:3340 | `pc.piece_type != chess.PAWN` | CHESS FACT | [C2] PROPOSED: "names: pawn = the unit that promotes (Game.promotes)" (GOAL; reads movement) | no printed signature: 2 rows |
| agenda.py:3342 | `RC.RACE ; W.race_claim(b2, pawn) is not None` | LOGIC | -- | -- |
| agenda.py:3342 | `b2.turn == self.me` | RULE | -- | -- |
| agenda.py:3345 | `d.kind in ("stopper", "blocked")` | CHESS FACT | [D1c] existing: stopper; blocked (+2) | cases 32/32, DECISION 0 |
| agenda.py:3346 | `d.key() in old` | LOGIC | -- | -- |
| agenda.py:3347 | `standing` | LOGIC | -- | -- |
| agenda.py:3350 | `new_question('promote', ...)` | CHESS FACT | [D8d] existing: the ... defect stands (...) and no fix is new... | cases 25/27, DECISION 0 |
| agenda.py:3357 | `new` | LOGIC | -- | -- |
| agenda.py:3364 | `x ; lost` | LOGIC | -- | -- |
| agenda.py:3365 | `line not in self.ctx.static_lines` | LOGIC | -- | -- |
| agenda.py:3370 | `CHASE_SAVES ; q is not None` | LOGIC | -- | -- |
| agenda.py:3370 | `q.goal == "save" ; "checkers" not in q.pattern` | CHESS FACT | [A10] PROPOSED: "goal: save = keep the balance (target 0)" (GOAL; reads material, tempo) | cases 47/56, DECISION 0 |
| agenda.py:3375 | `self._keep_goal(self._chase_q)` | CHESS FACT | [A10] PROPOSED: "goal: save = keep the balance (target 0)" (GOAL; reads material, tempo) | cases 47/56, DECISION 0 |
| agenda.py:3377 | `self._t(NEED_PROPAGATION, 3)` | LOGIC | -- | -- |
| agenda.py:3377 | `max(1, self.min_stake) ... else 1 (a route must reach one unit, or the need)` | CHESS FACT | [A7] PROPOSED: "goal: the need (what a claim must bring), propagated into the chase" (GOAL; reads material) | no printed signature: 15 rows |
| agenda.py:3384 | `gk0 is not None ; gk0 != gk` | LOGIC | -- | -- |
| agenda.py:3386 | `reads is None` | LOGIC | -- | -- |
| agenda.py:3386 | `dbal >= 1 ; dbal0 < dbal ; target0 > target` | CHESS FACT | [D25] existing: standing kill | cases 28/26, DECISION 0 |
| agenda.py:3388 | `reads & frozenset(P_old.diff(P2))` | LOGIC | -- | -- |
| agenda.py:3396 | `P2.mate_in_one() if mate1 is None else mate1` | RULE | -- | -- |
| agenda.py:3397 | `BD.MATE_VALUE  in: return BD.MATE_VALUE, "mate in one"` | CHESS FACT | [D3] existing: mate in one; nothing of theirs | cases 70/81, DECISION 0 |
| agenda.py:3398 | `P2.is_check()` | RULE | -- | -- |
| agenda.py:3398 | `dbal >= 1` | CHESS FACT | [D3] existing: mate in one; nothing of theirs | cases 70/81, DECISION 0 |
| agenda.py:3400 | `need = 1 - dbal (what the exchange must still bring)` | CHESS FACT | [A6] PROPOSED: "valuation: a gain is at least one unit of material" (VALUATION; reads material) (beside existing: no claim of ours and no material gain) | cases 80/80, DECISION 0 |
| agenda.py:3402 | `best < need` | CHESS FACT | [D3] existing: mate in one; nothing of theirs | cases 70/81, DECISION 0 |
| agenda.py:3404 | `self._t(NEED_AWARE_CLOSE, 3)` | LOGIC | -- | -- |
| agenda.py:3404 | `dbal + best < self.min_stake` | CHESS FACT | [A7] PROPOSED: "goal: the need (what a claim must bring), propagated into the chase" (GOAL; reads material) | no printed signature: 15 rows |
| agenda.py:3407 | `q is not None ; q.pattern.get("target") is not None` | LOGIC | -- | -- |
| agenda.py:3407 | `q.goal == "win"` | CHESS FACT | [A1w] PROPOSED: "goal: win material" (GOAL; reads material, attack) | cases 83/86, DECISION 0 |
| agenda.py:3407 | `dbal + best < q.stake` | CHESS FACT | [D3] existing: mate in one; nothing of theirs | cases 70/81, DECISION 0 |
| agenda.py:3409 | `fix` | LOGIC | -- | -- |
| agenda.py:3412 | `line not in self.ctx.open_lines` | LOGIC | -- | -- |
| agenda.py:3416 | `t` | CHESS FACT | [D3] existing: mate in one; nothing of theirs | cases 70/81, DECISION 0 |
| agenda.py:3418 | `CLOSE_AFTER_CAPTURE` | LOGIC | -- | -- |
| agenda.py:3420 | `t2` | CHESS FACT | [D23] PROPOSED: "valuation: the exchange law's principal lines (swap list)" (VALUATION; reads attack, material) (beside existing: nothing of theirs) | cases 29/48, DECISION 0 |
| agenda.py:3423 | `line not in self.ctx.open_lines` | LOGIC | -- | -- |
| agenda.py:3432 | `COST_JUDGE ; ctx is not None ; ctx.final is not None ; ctx.deadline is not None` | LOGIC | -- | -- |
| agenda.py:3438 | `n >= JUDGE_MIN_AGENDAS` | LOGIC | -- | -- |
| agenda.py:3442 | `cq is not None` | LOGIC | -- | -- |
| agenda.py:3442 | `cq.stake if cq is not None else 1 (the default stake: one unit)` | CHESS FACT | [A6] PROPOSED: "valuation: a gain is at least one unit of material" (VALUATION; reads material) (beside existing: no claim of ours and no material gain) | cases 80/80, DECISION 0 |
| agenda.py:3448 | `ctx.pool ; ctx.one_route` | LOGIC | -- | -- |
| agenda.py:3451 | `st is not None` | LOGIC | -- | -- |
| agenda.py:3451 | `self.keeps(st)[0]` | CHESS FACT | [A5] PROPOSED: "goal: the ladder -- don't lose, then win (status, line, keeps)" (GOAL; reads material, tempo) | cases 22/18, DECISION 35 |
| agenda.py:3453 | `self.best is not None ; self.best[0] >= q.stake` | LOGIC | -- | -- |
| agenda.py:3464 | `pol is None ; self.chase_depth != 0` | LOGIC | -- | -- |
| agenda.py:3469 | `self.best is not None ; self.best[0] >= q.stake` | LOGIC | -- | -- |
| agenda.py:3470 | `2  in: return "", f"Q{q.id} cannot beat the best claim ({self.best[2].what} {self.best[0]:+d})"` | LOGIC | -- | -- |
| agenda.py:3472 | `p is None` | LOGIC | -- | -- |
| agenda.py:3472 | `p < 1 / n` | HEURISTIC | -- | -- |
| agenda.py:3473 | `p is not None` | LOGIC | -- | -- |
| agenda.py:3474 | `word.startswith(FORCED_WORDS)` | CHESS FACT | [B7] existing: cut by the judge | cases 0/0, DECISION 0 |
| agenda.py:3476 | `self.best is None ; 2  in: best = "none yet" if self.best is None else f"{self.best[2].what} {self.best[0]:+d}"` | LOGIC | -- | -- |
| agenda.py:3485 | `hit is None ; hit[2] ; limit is None ; limit > hit[0]` | LOGIC | -- | -- |
| agenda.py:3487 | `limit is not None ; cost_ > limit` | LOGIC | -- | -- |
| agenda.py:3494 | `priced is None` | LOGIC | -- | -- |
| agenda.py:3495 | `seen is None` | LOGIC | -- | -- |
| agenda.py:3497 | `seen[1]` | LOGIC | -- | -- |
| agenda.py:3502 | `san.endswith(("+", "#")) ; "x" in san` | CHESS FACT | [C3] PROPOSED: "names: check / capture bound to the kernel (Game.gives_check, Game.is_capture), not to the notation" (FORCING; reads attack, movement) | no printed signature: 2 rows |
| agenda.py:3507 | `n >= JUDGE_MIN_AGENDAS` | LOGIC | -- | -- |
| agenda.py:3513 | `n >= JUDGE_MIN_AGENDAS ; t > 0` | LOGIC | -- | -- |
| agenda.py:3526 | `units.get(k)` | LOGIC | -- | -- |
| agenda.py:3528 | `own ; r != spn` | LOGIC | -- | -- |
| agenda.py:3531 | `own` | LOGIC | -- | -- |
| agenda.py:3539 | `seen is not None ; need == seen[0]` | LOGIC | -- | -- |
| agenda.py:3540 | `seen[1]` | LOGIC | -- | -- |
| agenda.py:3558 | `spn <= 0 ; q is None ; ctx.held is None ; self.chase_depth != 0 ; k >= TIER_COUNT - 1` | LOGIC | -- | -- |
| agenda.py:3558 | `q.goal == "save" ; "checkers" in q.pattern` | CHESS FACT | [D21] existing: in check | cases 58/64, DECISION 0 |
| agenda.py:3562 | `cands is None` | LOGIC | -- | -- |
| agenda.py:3567 | `dl ; ns` | LOGIC | -- | -- |
| agenda.py:3567 | `k + 1 >= 1 ; k + 1 >= 2` | HEURISTIC | -- | -- |
| agenda.py:3568 | `why` | LOGIC | -- | -- |
| agenda.py:3575 | `seen is not None ; seen[1]` | LOGIC | -- | -- |
| agenda.py:3577 | `seen is not None ; seen[0] >= top` | LOGIC | -- | -- |
| agenda.py:3581 | `limit < 1` | LOGIC | -- | -- |
| agenda.py:3584 | `lower` | LOGIC | -- | -- |
| agenda.py:3587 | `price < top` | LOGIC | -- | -- |
| agenda.py:3596 | `cheap ; override ; self.chase_depth != 0 ; self._judge_on()` | LOGIC | -- | -- |
| agenda.py:3602 | `spn0 is not None ; spn0 > 0` | LOGIC | -- | -- |
| agenda.py:3603 | `ctx.root_end ; ctx.final` | LOGIC | -- | -- |
| agenda.py:3603 | `2  in: limit = int(max(max((ctx.root_end or ctx.final) - now0, 0.0) / 2, max(ctx.deadline - now0, 0.0)) / self._low_r...` | HEURISTIC | -- | -- |
| agenda.py:3607 | `len(kids) > 6 ; 6  in: shown = ", ".join(f"{k} {c}" for k, c in kids[:6]) + (", ..." if len(kids) > 6 else "")` | LOGIC | -- | -- |
| agenda.py:3611 | `ctx.root_end ; ctx.final` | LOGIC | -- | -- |
| agenda.py:3615 | `seen is None` | LOGIC | -- | -- |
| agenda.py:3616 | `spn is None` | LOGIC | -- | -- |
| agenda.py:3617 | `0.5  in: slice_ = max(equal, min(pool / 4, 0.5)) ; 4  in: slice_ = max(equal, min(pool / 4, 0.5))` | HEURISTIC | -- | -- |
| agenda.py:3619 | `need is not None` | LOGIC | -- | -- |
| agenda.py:3620 | `2  in: slice_ = max(equal, min(2 * need, pool / 2))` | HEURISTIC | -- | -- |
| agenda.py:3623 | `2  in: slice_ = max(equal, min(2 * need, pool / 2))` | HEURISTIC | -- | -- |
| agenda.py:3625 | `ctx.claim_all` | LOGIC | -- | -- |
| agenda.py:3628 | `need is not None` | LOGIC | -- | -- |
| agenda.py:3631 | `COMPETE ; spn is not None ; need is not None ; mkey_c not in ctx.exempt ; mkey_c not in ctx.opened` | LOGIC | -- | -- |
| agenda.py:3632 | `comp is not None ; judge_next(comp[0], need) ; ctx.held()` | LOGIC | -- | -- |
| agenda.py:3636 | `comp[1] is not None` | LOGIC | -- | -- |
| agenda.py:3637 | `2  in: self.lines.append(f" judge: {san} cut ({what}, {need:.2f} s): tier {ctx.tier + 1}'s {comp[2]} ({comp[3]}; {cco...` | LOGIC | -- | -- |
| agenda.py:3638 | `2  in: res = dict(value=0, killed=None, gap=LS.name(f"out of time: {san} cut by the judge (tier {ctx.tier + 1}'s {com...` | LOGIC | -- | -- |
| agenda.py:3643 | `ctx.why_not` | LOGIC | -- | -- |
| agenda.py:3644 | `ctx.pool` | LOGIC | -- | -- |
| agenda.py:3645 | `need is not None ; need > equal` | LOGIC | -- | -- |
| agenda.py:3645 | `need > pool / 2` | HEURISTIC | -- | -- |
| agenda.py:3655 | `need is not None` | LOGIC | -- | -- |
| agenda.py:3662 | `ctx.t0 is not None ; 3  in: res["timed"], res["mark"] = True, round(time.monotonic() - ctx.t0, 3) if ctx.t0 is not No...` | LOGIC | -- | -- |
| agenda.py:3664 | `t[4] ; t[2] ; t[3] ; sorted(... key=lambda t: (0 if t[4] else 1 if t[2] else 2 if t[3] else 3, t[1], t[0]))` | CHESS FACT | [D22] PROPOSED: "heuristic: the forcing order (checks, captures, threats; recaptures first)" (HEURISTIC; reads attack, material) | no printed signature: 6 rows |
| agenda.py:3665 | `rows` | LOGIC | -- | -- |
| agenda.py:3671 | `self.chase_depth == 0 ; ctx.c_lines` | LOGIC | -- | -- |
| agenda.py:3674 | `tok is None` | LOGIC | -- | -- |
| agenda.py:3677 | `res.get("gap") is None` | LOGIC | -- | -- |
| agenda.py:3683 | `res.get("gap") is None ; prev is not None ; prev[1]` | LOGIC | -- | -- |
| agenda.py:3692 | `self._judge_on() ; ctx.hand is not None` | LOGIC | -- | -- |
| agenda.py:3695 | `spn is None` | LOGIC | -- | -- |
| agenda.py:3699 | `self._reply_san` | LOGIC | -- | -- |
| agenda.py:3700 | `cost * rate <= left` | LOGIC | -- | -- |
| agenda.py:3705 | `mv ; self._route_move is not None` | LOGIC | -- | -- |
| agenda.py:3714 | `pos is not None ; 64  in: bal = (pos if pos is not None else AC.Pos(fen)).balance(self.me) - 64` | LOGIC | -- | -- |
| agenda.py:3715 | `bal >= 0` | CHESS FACT | [A8] existing: repetition | cases 5/14, DECISION 0 |
| agenda.py:3717 | `pk == ctx.root_pk ; pk not in ctx.active_label ; pk in ctx.history` | LOGIC | -- | -- |
| agenda.py:3718 | `self._route_move is not None ; self._reply_san` | LOGIC | -- | -- |
| agenda.py:3719 | `label.startswith('the root') ; label.startswith('the game')` | LOGIC | -- | -- |
| agenda.py:3720 | `Claim('draw', ...) ; Claim(... goal='draw')` | CHESS FACT | [A1d] existing: repetition | cases 5/14, DECISION 0 |
| agenda.py:3729 | `PARRIES ; closed is None` | LOGIC | -- | -- |
| agenda.py:3733 | `cq is not None` | LOGIC | -- | -- |
| agenda.py:3733 | `stake = cq.stake if cq is not None else 1 (the default stake: one unit)` | CHESS FACT | [A6] PROPOSED: "valuation: a gain is at least one unit of material" (VALUATION; reads material) (beside existing: no claim of ours and no material gain) | cases 80/80, DECISION 0 |
| agenda.py:3735 | `th is None ; th.kind not in ("mate", "mate2", "threat", "pass-claim")` | CHESS FACT | [D24] existing: threat of mate; threat of mate in two (+4) | cases 11/11, DECISION 0 |
| agenda.py:3741 | `r not in after.legal_moves` | RULE | -- | -- |
| agenda.py:3745 | `d.is_check()` | RULE | -- | -- |
| agenda.py:3746 | `v` | LOGIC | -- | -- |
| agenda.py:3746 | `R._counter(after, r, stake, after.turn)` | RULE | -- | -- |
| agenda.py:3753 | `why.startswith("the threat (")` | LOGIC | -- | -- |
| agenda.py:3763 | `chess.SQUARES  in: for x in chess.SQUARES: p = b_rb.piece_at(x) if p is None or p.color == me or p.piece_type == ches...` | NAME | -- | -- |
| agenda.py:3765 | `p is None` | LOGIC | -- | -- |
| agenda.py:3765 | `p.color == me ; b_rb.is_attacked_by(me, x)` | RULE | -- | -- |
| agenda.py:3765 | `p.piece_type == chess.KING` | CHESS FACT | [C1] PROPOSED: "names: king = the royal unit (Game.royal); excluded from the valuation" (VALUATION; reads material) | no printed signature: 8 rows |
| agenda.py:3768 | `v > best` | CHESS FACT | [D23] PROPOSED: "valuation: the exchange law's principal lines (swap list)" (VALUATION; reads attack, material) (beside existing: nothing of theirs) | cases 29/48, DECISION 0 |
| agenda.py:3770 | `v == best ; v > 0` | CHESS FACT | [D23] PROPOSED: "valuation: the exchange law's principal lines (swap list)" (VALUATION; reads attack, material) (beside existing: nothing of theirs) | cases 29/48, DECISION 0 |
| agenda.py:3775 | `len(sans) < 12` | HEURISTIC | -- | -- |
| agenda.py:3776 | `m.to_square == sq` | RULE | -- | -- |
| agenda.py:3777 | `caps` | LOGIC | -- | -- |
| agenda.py:3777 | `c_rb.piece_at(sq) is None` | RULE | -- | -- |
| agenda.py:3779 | `min(... key=lambda m: (W.base.worth(c_rb.piece_at(m.from_square)), m.uci()))` | CHESS FACT | [D23] PROPOSED: "valuation: the exchange law's principal lines (swap list)" (VALUATION; reads attack, material) (beside existing: nothing of theirs) | cases 29/48, DECISION 0 |
| agenda.py:3789 | `taken < n` | LOGIC | -- | -- |
| agenda.py:3789 | `gains[taken] - f[taken + 1] > 0` | CHESS FACT | [D23] PROPOSED: "valuation: the exchange law's principal lines (swap list)" (VALUATION; reads attack, material) (beside existing: nothing of theirs) | cases 29/48, DECISION 0 |
| agenda.py:3791 | `taken` | LOGIC | -- | -- |
| agenda.py:3792 | `i % 2 == 0 ; i == taken - 1` | CHESS FACT (text only) | [D23] PROPOSED: "valuation: the exchange law's principal lines (swap list)" (VALUATION; reads attack, material) (beside existing: nothing of theirs) | cases 29/48, DECISION 0 |
| agenda.py:3802 | `lines` | LOGIC | -- | -- |
| agenda.py:3808 | `b_rb.legal_moves` | RULE | -- | -- |
| agenda.py:3812 | `hit is None` | LOGIC | -- | -- |
| agenda.py:3814 | `c.move ; ag is not None` | LOGIC | -- | -- |
| agenda.py:3814 | `c.move[2:4] == GAME.loc_name(sq)` | NAME | -- | -- |
| agenda.py:3814 | `c.value <= credit` | CHESS FACT | [D23] PROPOSED: "valuation: the exchange law's principal lines (swap list)" (VALUATION; reads attack, material) (beside existing: nothing of theirs) | cases 29/48, DECISION 0 |
| agenda.py:3815 | `cls ; max(... key=lambda c: c.value)` | LOGIC | -- | -- |
| agenda.py:3816 | `cl is not None` | LOGIC | -- | -- |
| agenda.py:3816 | `cl.value > 0` | CHESS FACT | [D23] PROPOSED: "valuation: the exchange law's principal lines (swap list)" (VALUATION; reads attack, material) (beside existing: nothing of theirs) | cases 29/48, DECISION 0 |
| agenda.py:3817 | `hit[0] > out[0]` | LOGIC | -- | -- |
| agenda.py:3819 | `least is None ; out[0] < least[0]` | LOGIC | -- | -- |
| agenda.py:3821 | `least[0] == 0` | LOGIC | -- | -- |
| agenda.py:3832 | `pc0 is None ; pc2 is None` | LOGIC | -- | -- |
| agenda.py:3832 | `pc2.color == self.me` | RULE | -- | -- |
| agenda.py:3832 | `pc2.piece_type != pc0.piece_type` | CHESS FACT | [D3] existing: mate in one; nothing of theirs | cases 70/81, DECISION 0 |
| agenda.py:3835 | `defects` | LOGIC | -- | -- |
| agenda.py:3838 | `new_question('win', ...)` | CHESS FACT | [D3] existing: mate in one; nothing of theirs | cases 70/81, DECISION 0 |
| agenda.py:3842 | `r.san not in sans` | LOGIC | -- | -- |
| agenda.py:3844 | `sans ; 3  in: return ", ".join(sans[:3]) if sans else None` | LOGIC | -- | -- |
| agenda.py:3848 | `self.ctx.tag is None` | LOGIC | -- | -- |
| agenda.py:3851 | `reads is not None` | LOGIC | -- | -- |
| agenda.py:3856 | `pos is not None` | LOGIC | -- | -- |
| agenda.py:3858 | `w is not None` | LOGIC | -- | -- |
| agenda.py:3872 | `hit is None` | LOGIC | -- | -- |
| agenda.py:3873 | `any(True for _ in c.legal_moves)` | RULE | -- | -- |
| agenda.py:3874 | `ag is not None ; ag.state.claims ; max(... key=lambda x: x.value)` | LOGIC | -- | -- |
| agenda.py:3875 | `cl` | LOGIC | -- | -- |
| agenda.py:3876 | `ag is not None` | LOGIC | -- | -- |
| agenda.py:3878 | `q.status == "gap"` | LOGIC | -- | -- |
| agenda.py:3879 | `cl` | LOGIC | -- | -- |
| agenda.py:3880 | `self._cur is not None` | LOGIC | -- | -- |
| agenda.py:3881 | `ag is None ; getattr(ag, "all", None) ; any(q.reads is None for q in ag.all)` | LOGIC | -- | -- |
| agenda.py:3891 | `g` | LOGIC | -- | -- |
| agenda.py:3894 | `ag is not None ; q.status == "gap" ; max(... key=lambda q: q.stake)` | LOGIC | -- | -- |
| agenda.py:3895 | `q is not None` | LOGIC | -- | -- |
| agenda.py:3905 | `self.chase_depth < RULE2_FROM.get(q.goal, 99)` | CHESS FACT | [B5] PROPOSED: "stop: a check that fixes no defect of its goal (the walk of checks)" (STOP; reads attack, movement, tempo) | cases 0/0, DECISION 0 |
| agenda.py:3907 | `why.startswith("fix:") ; why != "check"` | CHESS FACT | [B5] PROPOSED: "stop: a check that fixes no defect of its goal (the walk of checks)" (STOP; reads attack, movement, tempo) | cases 0/0, DECISION 0 |
| agenda.py:3909 | `q.pattern.get("at") is not None ; q.pattern.get("target") is not None ; q.pattern.get("pawn") is not None ; goal = {'...` | NAME | -- | -- |
| agenda.py:3910 | `q.goal == "mate"` | CHESS FACT | [A1m] PROPOSED: "goal: mate (the rule's terminal)" (GOAL; reads movement, attack) | cases 100/100, DECISION 0 |
| agenda.py:3910 | `q.pattern.get("move") == m.uci() ; m.uci() in (q.pattern.get("reach") or ())` | CHESS FACT | [B5] PROPOSED: "stop: a check that fixes no defect of its goal (the walk of checks)" (STOP; reads attack, movement, tempo) | cases 0/0, DECISION 0 |
| agenda.py:3914 | `any(hit is not None and hit["gap"] is not None for hit in hits)` | LOGIC | -- | -- |
| agenda.py:3918 | `any(ch != m.to_square for ch in c.checkers())` | CHESS FACT | [B5] PROPOSED: "stop: a check that fixes no defect of its goal (the walk of checks)" (STOP; reads attack, movement, tempo) | cases 0/0, DECISION 0 |
| agenda.py:3923 | `d.sq is not None` | LOGIC | -- | -- |
| agenda.py:3923 | `c.is_attacked_by(me, d.sq)` | CHESS FACT | [B5] PROPOSED: "stop: a check that fixes no defect of its goal (the walk of checks)" (STOP; reads attack, movement, tempo) | cases 0/0, DECISION 0 |
| agenda.py:3923 | `d.kind == "flight"` | CHESS FACT | [D1a] existing: flight; checker takeable (+2) | cases 75/75, DECISION 0 |
| agenda.py:3925 | `m.to_square in d.by` | CHESS FACT | [B5] PROPOSED: "stop: a check that fixes no defect of its goal (the walk of checks)" (STOP; reads attack, movement, tempo) | cases 0/0, DECISION 0 |
| agenda.py:3925 | `d.kind == "checker takeable"` | CHESS FACT | [D1a] existing: flight; checker takeable (+2) | cases 75/75, DECISION 0 |
| agenda.py:3927 | `replies ; all(r.from_square in d.by for r in replies)` | CHESS FACT | [B5] PROPOSED: "stop: a check that fixes no defect of its goal (the walk of checks)" (STOP; reads attack, movement, tempo) | cases 0/0, DECISION 0 |
| agenda.py:3927 | `d.kind in ("defended", "checker takeable")` | CHESS FACT | [D1a] existing: flight; checker takeable (+2) | cases 75/75, DECISION 0 |
| agenda.py:3929 | `m.to_square in c.attackers(me, d.sq)` | CHESS FACT | [B5] PROPOSED: "stop: a check that fixes no defect of its goal (the walk of checks)" (STOP; reads attack, movement, tempo) | cases 0/0, DECISION 0 |
| agenda.py:3929 | `d.kind == "not attacked"` | CHESS FACT | [D1b] existing: defended; not attacked | cases 68/68, DECISION 0 |
| agenda.py:3931 | `q.goal == "win" ; q.pattern.get("target") in c.attacks(m.to_square)` | CHESS FACT | [B5] PROPOSED: "stop: a check that fixes no defect of its goal (the walk of checks)" (STOP; reads attack, movement, tempo) | cases 0/0, DECISION 0 |
| agenda.py:3931 | `d.kind in ("not attacked", "defended")` | CHESS FACT | [D1b] existing: defended; not attacked | cases 68/68, DECISION 0 |
| agenda.py:3933 | `q.goal == "win" ; G.skewer_after(c, me, q.pattern.get("target")) == m.to_square` | CHESS FACT | [B5] PROPOSED: "stop: a check that fixes no defect of its goal (the walk of checks)" (STOP; reads attack, movement, tempo) | cases 0/0, DECISION 0 |
| agenda.py:3933 | `d.kind in ("not attacked", "defended")` | CHESS FACT | [D1b] existing: defended; not attacked | cases 68/68, DECISION 0 |
| agenda.py:3935 | `q.goal == "promote"` | CHESS FACT | [B5] PROPOSED: "stop: a check that fixes no defect of its goal (the walk of checks)" (STOP; reads attack, movement, tempo) | cases 0/0, DECISION 0 |
| agenda.py:3937 | `why_fix` | LOGIC | -- | -- |
| agenda.py:3940 | `"; ".join(f"{d.sq and GAME.loc_name(d.sq) + ' ' or ''}{ {'flight': 'stays a flight', 'checker takeable': 'stays takea...` | LOGIC | -- | -- |
| agenda.py:3940 | `{'flight': 'stays a flight', 'checker takeable': ..., 'reach': 'stays out of reach'} (the defect kinds named in the n...` | NAME | -- | -- |
| agenda.py:3948 | `m.to_square in d.by` | CHESS FACT | [D8g] existing: drive off | cases 8/8, DECISION 0 |
| agenda.py:3951 | `ek is None` | LOGIC | -- | -- |
| agenda.py:3951 | `d.kind == "blocked" ; d.kind in ("path attacked", "promo square attacked", "stopper")` | CHESS FACT | [D1c] existing: stopper; blocked (+2) | cases 32/32, DECISION 0 |
| agenda.py:3951 | `d.sq == ek ; ek in d.by` | CHESS FACT | [D8g] existing: drive off | cases 8/8, DECISION 0 |
| agenda.py:3954 | `W.drives_off(b, m, x, ek)` | LOGIC | -- | -- |
| agenda.py:3957 | `reps` | LOGIC | -- | -- |
| agenda.py:3965 | `k2 is not None ; W.drives_off(c2, m2, x, k2)` | LOGIC | -- | -- |
| agenda.py:3966 | `nxt` | LOGIC | -- | -- |
| agenda.py:3969 | `3  in: return f"a step of driving their king off {GAME.loc_name(x)} ({'; '.join(steps[:3])} drives it off)"` | LOGIC | -- | -- |
| agenda.py:3974 | `q.status != "killed" ; str(q.reason).startswith("stopped:")` | LOGIC | -- | -- |
| agenda.py:3977 | `self.kills ; self.kills[-1][0] == q.id` | LOGIC | -- | -- |
| agenda.py:3980 | `self.lines[k].startswith(f" -> KILLED Q{q.id}: stopped:")` | LOGIC | -- | -- |
| agenda.py:3989 | `len(path) == 1` | RULE | -- | -- |
| agenda.py:3989 | `chess.QUEEN  in: push = chess.Move(sq, path[0], promotion=chess.QUEEN if len(path) == 1 else None)` | CHESS FACT | [A3] PROPOSED: "stake: promotion (queen less pawn = 8)" (VALUATION; reads material, movement) | cases 35/35, DECISION 0 |
| agenda.py:3990 | `push not in b.legal_moves` | RULE | -- | -- |
| agenda.py:3998 | `b.is_capture(m)` | RULE | -- | -- |
| agenda.py:3998 | `b.piece_type_at(m.from_square) == chess.PAWN` | CHESS FACT | [B1] existing: the chase clock | cases 5/9, DECISION 0 |
| agenda.py:4008 | `CLOCK_IRREVERSIBLE` | LOGIC | -- | -- |
| agenda.py:4009 | `self._irreversible(uci) ; dbal >= 0 ; self._pawn_move(m)` | CHESS FACT | [B1] existing: the chase clock | cases 5/9, DECISION 0 |
| agenda.py:4010 | `self.board.is_capture(m)` | RULE | -- | -- |
| agenda.py:4010 | `dbal >= 0 ; self._pawn_move(m)` | CHESS FACT | [B1] existing: the chase clock | cases 5/9, DECISION 0 |
| agenda.py:4014 | `EVADED_RESET ; self._irreversible(uci)` | LOGIC | -- | -- |
| agenda.py:4018 | `CLOCK_PAWN ; self.board.piece_type_at(m.from_square) == chess.PAWN` | CHESS FACT | [B1] existing: the chase clock | cases 5/9, DECISION 0 |
| agenda.py:4024 | `hit is not None` | LOGIC | -- | -- |
| agenda.py:4030 | `self.ctx.tier >= 2` | HEURISTIC | -- | -- |
| agenda.py:4032 | `th is None` | CHESS FACT | [B8] PROPOSED: "stop: still forcing -- every reply parried the named threat (Ideas.Forcing.stillForcing)" (STOP; reads attack, tempo) | cases 0/0, DECISION 0 |
| agenda.py:4039 | `d.is_check()` | RULE | -- | -- |
| agenda.py:4040 | `R._counter(after, r, stake, after.turn)` | RULE | -- | -- |
| agenda.py:4040 | `v` | CHESS FACT | [B8] PROPOSED: "stop: still forcing -- every reply parried the named threat (Ideas.Forcing.stillForcing)" (STOP; reads attack, tempo) | cases 0/0, DECISION 0 |
| agenda.py:4048 | `ok` | LOGIC | -- | -- |
| agenda.py:4054 | `self.ctx is None ; self.ctx.extend` | LOGIC | -- | -- |
| agenda.py:4057 | `ok` | LOGIC | -- | -- |
| agenda.py:4067 | `self.chase_depth != 0 ; self.ctx.resume_ext` | LOGIC | -- | -- |
| agenda.py:4070 | `e is None` | LOGIC | -- | -- |
| agenda.py:4074 | `self.ctx.final ; time.monotonic()` | LOGIC | -- | -- |
| agenda.py:4081 | `self.ctx is not None` | LOGIC | -- | -- |
| agenda.py:4085 | `self._t(EVASION_CHASE, 3)` | LOGIC | -- | -- |
| agenda.py:4085 | `q.goal == "save" ; "checkers" in q.pattern` | CHESS FACT | [D21] existing: in check | cases 58/64, DECISION 0 |
| agenda.py:4086 | `self.ctx is None` | LOGIC | -- | -- |
| agenda.py:4086 | `self._keep_goal(q)` | CHESS FACT | [A10] PROPOSED: "goal: save = keep the balance (target 0)" (GOAL; reads material, tempo) | cases 47/56, DECISION 0 |
| agenda.py:4086 | `self.min_stake >= BD.MATE_VALUE` | CHESS FACT | [A7] PROPOSED: "goal: the need (what a claim must bring), propagated into the chase" (GOAL; reads material) | no printed signature: 15 rows |
| agenda.py:4086 | `q.goal not in ("mate", "win", "promote") ; CHASE_SAVES ; q.goal == "save" ; self.chase_depth <= 1 ; self.chase_depth ...` | CHESS FACT | [B6] PROPOSED: "stop: the chase's follow list (forced words below the root)" (STOP; reads attack, movement, tempo) (beside existing: check; capture) | no printed signature: 10 rows |
| agenda.py:4086 | `evading` | CHESS FACT | [D21] existing: in check | cases 58/64, DECISION 0 |
| agenda.py:4088 | `self.chase_depth` | LOGIC | -- | -- |
| agenda.py:4088 | `self.since_capture >= self._clock() ; evading ; self.since_capture >= self._clock() + self._extension()` | CHESS FACT | [B1] existing: the chase clock | cases 5/9, DECISION 0 |
| agenda.py:4092 | `SECOND_CHECK_STOP` | LOGIC | -- | -- |
| agenda.py:4092 | `evading ; self.evaded ; self.evaded > self._extension()` | CHESS FACT | [B2] existing: a second check of theirs in a row | cases 30/56, DECISION 0 |
| agenda.py:4095 | `self.best is not None ; self.best[0] >= q.stake` | LOGIC | -- | -- |
| agenda.py:4096 | `2  in: self.lines.append(f" chase: not asked, the best claim ({self.best[2].what}, {self.best[0]}) is at least the st...` | LOGIC | -- | -- |
| agenda.py:4099 | `self.ctx.out_of_time()` | LOGIC | -- | -- |
| agenda.py:4106 | `self._t(GOAL_JUDGE, 3)` | LOGIC | -- | -- |
| agenda.py:4106 | `q.goal in ("mate", "win")` | CHESS FACT | [B3] existing: tempo arithmetic; tempo | cases 61/74, DECISION 0 |
| agenda.py:4107 | `q.goal == "win" ; self.chase_depth > 0 ; self.since_capture == 0` | CHESS FACT | [B3] existing: tempo arithmetic; tempo | cases 61/74, DECISION 0 |
| agenda.py:4108 | `self.ctx.mate_levels` | LOGIC | -- | -- |
| agenda.py:4109 | `top is None` | LOGIC | -- | -- |
| agenda.py:4109 | `fresh ; q.stake > top[1]` | CHESS FACT | [B3] existing: tempo arithmetic; tempo | cases 61/74, DECISION 0 |
| agenda.py:4110 | `dist = max([1, len(q.defects)] + [f.pattern.get('steps', 1) for f in q.fixes]) (the prep moves a goal may spend)` | CHESS FACT | [B3] existing: tempo arithmetic; tempo | cases 61/74, DECISION 0 |
| agenda.py:4111 | `parent is None` | LOGIC | -- | -- |
| agenda.py:4111 | `self.free_level ; parent - (0 if self.free_level else 1) (one level spent per our move; a recapture spends none)` | CHESS FACT | [B3] existing: tempo arithmetic; tempo | cases 61/74, DECISION 0 |
| agenda.py:4112 | `allow <= 0 ; allow + self._extension() <= 0` | CHESS FACT | [B3] existing: tempo arithmetic; tempo | cases 61/74, DECISION 0 |
| agenda.py:4122 | `evading` | CHESS FACT | [D21] existing: in check | cases 58/64, DECISION 0 |
| agenda.py:4124 | `pc is not None` | LOGIC | -- | -- |
| agenda.py:4126 | `self.ctx.tier >= 2` | HEURISTIC | -- | -- |
| agenda.py:4127 | `DELAYED_THREAT ; NO_SAFE_MOVE` | LOGIC | -- | -- |
| agenda.py:4127 | `self.chase_depth` | CHESS FACT | [B6] PROPOSED: "stop: the chase's follow list (forced words below the root)" (STOP; reads attack, movement, tempo) (beside existing: check; capture) | no printed signature: 10 rows |
| agenda.py:4127 | `self.ctx.tier >= 1` | HEURISTIC | -- | -- |
| agenda.py:4128 | `why` | LOGIC | -- | -- |
| agenda.py:4129 | `why.startswith("delayed threat")` | LOGIC | -- | -- |
| agenda.py:4131 | `why.startswith("no safe move")` | LOGIC | -- | -- |
| agenda.py:4134 | `ok` | LOGIC | -- | -- |
| agenda.py:4136 | `note` | LOGIC | -- | -- |
| agenda.py:4141 | `self._t(DEEP_WIN_CHASE, 3)` | LOGIC | -- | -- |
| agenda.py:4141 | `q.goal == "promote" ; q.goal == "win"` | CHESS FACT | [B6] PROPOSED: "stop: the chase's follow list (forced words below the root)" (STOP; reads attack, movement, tempo) (beside existing: check; capture) | no printed signature: 10 rows |
| agenda.py:4145 | `r.move not in seen` | LOGIC | -- | -- |
| agenda.py:4148 | `getattr(q, "_lift_routes", None)` | LOGIC | -- | -- |
| agenda.py:4148 | `q.goal == "promote"` | CHESS FACT | [D8h] existing: unstoppablePasser2_sound; Ideas.Race.race_sound (+4) | cases 41/47, DECISION 3 |
| agenda.py:4150 | `r.move not in seen` | LOGIC | -- | -- |
| agenda.py:4151 | `getattr(q, "_support_routes", None)` | LOGIC | -- | -- |
| agenda.py:4151 | `q.goal == "promote"` | CHESS FACT | [D8h] existing: unstoppablePasser2_sound; Ideas.Race.race_sound (+4) | cases 41/47, DECISION 3 |
| agenda.py:4153 | `r.move not in seen` | LOGIC | -- | -- |
| agenda.py:4154 | `q.goal == "promote" ; self.chase_depth` | CHESS FACT | [D8f] PROPOSED: "forcing: a threat on the stopper / the pawn offered to it" (FORCING; reads attack, material, movement) | cases 15/15, DECISION 0 |
| agenda.py:4158 | `self.chase_depth == 0 ; self._root ; routes ; self.ctx is not None ; self.ctx.pool ; PRIOR ; GP.load() is not None` | LOGIC | -- | -- |
| agenda.py:4159 | `self.ctx.seconds ; self.ctx.final ; time.monotonic()` | LOGIC | -- | -- |
| agenda.py:4160 | `self.ctx.route_costs.get((self.pk, r.move.uci(), q.stake)) ; (None,)` | LOGIC | -- | -- |
| agenda.py:4163 | `sc` | LOGIC | -- | -- |
| agenda.py:4164 | `sorted(... key=lambda i: -sc[i])` | LOGIC | -- | -- |
| agenda.py:4165 | `order != list(range(len(routes)))` | LOGIC | -- | -- |
| agenda.py:4166 | `len(order) > 6 ; 6  in: self.lines.append(" policy: routes in the policy's order: " + ", ".join(f"{routes[i].san} ({s...` | LOGIC | -- | -- |
| agenda.py:4172 | `self.chase_depth ; evading` | CHESS FACT | [B6] PROPOSED: "stop: the chase's follow list (forced words below the root)" (STOP; reads attack, movement, tempo) (beside existing: check; capture) | no printed signature: 10 rows |
| agenda.py:4173 | `q.goal == "promote"` | CHESS FACT | [B6] PROPOSED: "stop: the chase's follow list (forced words below the root)" (STOP; reads attack, movement, tempo) (beside existing: check; capture) | no printed signature: 10 rows |
| agenda.py:4173 | `self.ctx.tier >= 1` | HEURISTIC | -- | -- |
| agenda.py:4174 | `r.why.removeprefix("fix: ").startswith(follow)` | CHESS FACT | [B6] PROPOSED: "stop: the chase's follow list (forced words below the root)" (STOP; reads attack, movement, tempo) (beside existing: check; capture) | no printed signature: 10 rows |
| agenda.py:4175 | `self.chase_depth >= 2 ; q.goal == "win"` | CHESS FACT | [B6] PROPOSED: "stop: the chase's follow list (forced words below the root)" (STOP; reads attack, movement, tempo) (beside existing: check; capture) | no printed signature: 10 rows |
| agenda.py:4177 | `b.gives_check(r.move)` | RULE | -- | -- |
| agenda.py:4177 | `r.move in fix_moves` | CHESS FACT | [B6] PROPOSED: "stop: the chase's follow list (forced words below the root)" (STOP; reads attack, movement, tempo) (beside existing: check; capture) | no printed signature: 10 rows |
| agenda.py:4178 | `policy_ordered` | LOGIC | -- | -- |
| agenda.py:4179 | `r.why.removeprefix("fix: ") == "check" ; r.why.removeprefix("fix: ").startswith("capture") ; sort(... key=lambda r: (...` | CHESS FACT | [D22] PROPOSED: "heuristic: the forcing order (checks, captures, threats; recaptures first)" (HEURISTIC; reads attack, material) | no printed signature: 6 rows |
| agenda.py:4180 | `self.chase_depth == 0 ; self.ctx.resume_set is not None` | LOGIC | -- | -- |
| agenda.py:4182 | `r.move.uci() == u` | LOGIC | -- | -- |
| agenda.py:4183 | `q.goal == "promote"` | CHESS FACT | [A1p] PROPOSED: "goal: promote (the rule)" (GOAL; reads movement, tempo) | cases 48/49, DECISION 0 |
| agenda.py:4184 | `routes` | LOGIC | -- | -- |
| agenda.py:4188 | `6  in: for n in notes[:6]: self.lines.append(" chase: " + n)` | LOGIC | -- | -- |
| agenda.py:4190 | `len(routes) > 12 ; 12  in: self.lines.append(f" chase: make them move -- {len(routes)} forcing route(s): " + ", ".joi...` | LOGIC | -- | -- |
| agenda.py:4194 | `q.goal == "promote"` | CHESS FACT | [A1p] PROPOSED: "goal: promote (the rule)" (GOAL; reads movement, tempo) | cases 48/49, DECISION 0 |
| agenda.py:4196 | `allow is not None` | LOGIC | -- | -- |
| agenda.py:4201 | `allow is not None` | LOGIC | -- | -- |
| agenda.py:4214 | `q.goal == "promote"` | CHESS FACT | [A1p] PROPOSED: "goal: promote (the rule)" (GOAL; reads movement, tempo) | cases 48/49, DECISION 0 |
| agenda.py:4215 | `got.get("pending")` | LOGIC | -- | -- |
| agenda.py:4219 | `j not in cheap_res` | LOGIC | -- | -- |
| agenda.py:4220 | `saved > 0` | LOGIC | -- | -- |
| agenda.py:4221 | `self.chase_depth == 0 ; saved > 0 ; i not in cheap_res ; self.ctx.final is None` | LOGIC | -- | -- |
| agenda.py:4225 | `i in cheap_res` | LOGIC | -- | -- |
| agenda.py:4227 | `self.ctx.out_of_time()` | LOGIC | -- | -- |
| agenda.py:4230 | `ESTIMATE` | LOGIC | -- | -- |
| agenda.py:4230 | `q.goal != "promote"` | CHESS FACT | [A1p] PROPOSED: "goal: promote (the rule)" (GOAL; reads movement, tempo) | cases 48/49, DECISION 0 |
| agenda.py:4233 | `self.chase_depth == 0 ; self.ctx.final is not None` | LOGIC | -- | -- |
| agenda.py:4235 | `root_final is None` | LOGIC | -- | -- |
| agenda.py:4236 | `self.best is None` | LOGIC | -- | -- |
| agenda.py:4236 | `t[4].goal in ("mate", "win", "promote") ; self.best[0] < t[4].stake` | CHESS FACT | [B4] existing: budget; chase fuse | cases 85/84, DECISION 0 |
| agenda.py:4237 | `cap ; self._whole_share(q)` | LOGIC | -- | -- |
| agenda.py:4237 | `2  in: root_final = now + (self.ctx.final - now) / 2 if cap and not self._whole_share(q) else self.ctx.final` | HEURISTIC | -- | -- |
| agenda.py:4239 | `claim` | LOGIC | -- | -- |
| agenda.py:4244 | `q.goal == "promote"` | CHESS FACT | [A1p] PROPOSED: "goal: promote (the rule)" (GOAL; reads movement, tempo) | cases 48/49, DECISION 0 |
| agenda.py:4246 | `self.chase_depth == 0 ; self.ctx.final is not None` | LOGIC | -- | -- |
| agenda.py:4250 | `self.chase_depth == 0 ; i not in cheap_res` | LOGIC | -- | -- |
| agenda.py:4253 | `saved > 0` | LOGIC | -- | -- |
| agenda.py:4256 | `res["gap"] ; res["gap"].startswith("out of time")` | LOGIC | -- | -- |
| agenda.py:4258 | `res["gap"]` | LOGIC | -- | -- |
| agenda.py:4259 | `res["gap"].startswith("chain stop")` | LOGIC | -- | -- |
| agenda.py:4260 | `self.chase_depth == 0 ; res.get("reused") ; RESUME` | LOGIC | -- | -- |
| agenda.py:4261 | `res.get("stop_line") ; []` | LOGIC | -- | -- |
| agenda.py:4265 | `res.get("reused")` | LOGIC | -- | -- |
| agenda.py:4267 | `res["killed"]` | LOGIC | -- | -- |
| agenda.py:4269 | `res["value"] >= (0 if self._keep_goal(q) else 1)` | CHESS FACT | [A6] PROPOSED: "valuation: a gain is at least one unit of material" (VALUATION; reads material) (beside existing: no claim of ours and no material gain) | cases 80/80, DECISION 0 |
| agenda.py:4270 | `len(res["parts"]) > 6 ; 6  in: self.lines.append(f" {r.san}: " + " / ".join(res["parts"][:6]) + (f" / ... {len(res['p...` | LOGIC | -- | -- |
| agenda.py:4275 | `self.ctx.out_of_time()` | LOGIC | -- | -- |
| agenda.py:4275 | `q.goal == "win"` | CHESS FACT | [D10] existing: refuter fix | cases 0/0, DECISION 0 |
| agenda.py:4283 | `self.chase_depth == 0 ; pool > 1` | LOGIC | -- | -- |
| agenda.py:4283 | `any(t[4].goal in ("mate", "win", "promote") and (self.best is None or self.best[0] < t[4].stake) for t in self.heap)` | CHESS FACT | [B4] existing: budget; chase fuse | cases 85/84, DECISION 0 |
| agenda.py:4286 | `2  in: charged = pool - max(self.ctx.left, pool - pool // 2)` | HEURISTIC | -- | -- |
| agenda.py:4287 | `2  in: self.ctx.left = max(self.ctx.left, pool - pool // 2)` | HEURISTIC | -- | -- |
| agenda.py:4290 | `cut ; stopped` | LOGIC | -- | -- |
| agenda.py:4291 | `ESTIMATE` | LOGIC | -- | -- |
| agenda.py:4293 | `done` | LOGIC | -- | -- |
| agenda.py:4295 | `sort(... key=lambda t: (-t[0], t[1].move.uci()))` | HEURISTIC | -- | -- |
| agenda.py:4301 | `q.goal == "promote"` | CHESS FACT | [A1p] PROPOSED: "goal: promote (the rule)" (GOAL; reads movement, tempo) | cases 48/49, DECISION 0 |
| agenda.py:4301 | `q.goal == "save"` | CHESS FACT | [A1s] PROPOSED: "goal: save material (don't lose)" (GOAL; reads material, attack) (beside existing: in check) | cases 78/83, DECISION 0 |
| agenda.py:4301 | `value >= BD.MATE_VALUE` | CHESS FACT | [A2] PROPOSED: "stake: the terminal (mate = 100)" (VALUATION; reads movement, attack) | cases 97/97, DECISION 0 |
| agenda.py:4301 | `res.get("material")` | CHESS FACT | [D8c] existing: the promotion is not claimed | cases 23/23, DECISION 0 |
| agenda.py:4302 | `kind == "save" ; self._keep_goal(q)` | CHESS FACT | [A10] PROPOSED: "goal: save = keep the balance (target 0)" (GOAL; reads material, tempo) | cases 47/56, DECISION 0 |
| agenda.py:4303 | `kept ; value < q.stake ; self.chase_depth` | CHESS FACT | [A10] PROPOSED: "goal: save = keep the balance (target 0)" (GOAL; reads material, tempo) | cases 47/56, DECISION 0 |
| agenda.py:4307 | `CYCLE` | LOGIC | -- | -- |
| agenda.py:4307 | `kind != "mate" ; any("repetition" in x for x in res["lemmas"]) ; value <= -(self.pos.balance(self.me) - 64)` | CHESS FACT | [A8] existing: repetition | cases 5/14, DECISION 0 |
| agenda.py:4309 | `len(res["parts"]) > 4 ; 4  in: parts = " / ".join(res["parts"][:4]) + (" / ..." if len(res["parts"]) > 4 else "")` | LOGIC | -- | -- |
| agenda.py:4310 | `res["lemmas"] ; 3  in: lemma = "chase: step_after+step_least over " + ", ".join(sorted(set(res["lemmas"]))[:3]) if re...` | LOGIC | -- | -- |
| agenda.py:4317 | `self._better(net, 0, claim)` | LOGIC | -- | -- |
| agenda.py:4330 | `b.is_check()` | RULE | -- | -- |
| agenda.py:4333 | `cq is not None` | LOGIC | -- | -- |
| agenda.py:4333 | `stake = cq.stake if cq is not None else 1 (the default stake: one unit)` | CHESS FACT | [A6] PROPOSED: "valuation: a gain is at least one unit of material" (VALUATION; reads material) (beside existing: no claim of ours and no material gain) | cases 80/80, DECISION 0 |
| agenda.py:4334 | `any(W.forcing_why(b, m, stake, self.us, mate_threats=True) for m in R._moves(b, self.us))` | CHESS FACT | [D11] existing: chase: quiet-status estimate | cases 0/0, DECISION 0 |
| agenda.py:4337 | `t == 0` | CHESS FACT | [D11] existing: chase: quiet-status estimate | cases 0/0, DECISION 0 |
| agenda.py:4347 | `res.get("est_done") ; res.get("gap") is None ; res.get("killed") ; self.chase_depth != 0` | LOGIC | -- | -- |
| agenda.py:4353 | `6  in: rows = [table[i][:6] for i in order]` | LOGIC | -- | -- |
| agenda.py:4354 | `6  in: rule_of = {t[0]: t[6] for t in table}` | LOGIC | -- | -- |
| agenda.py:4355 | `7  in: pos_of = {t[0]: t[7] for t in table}` | LOGIC | -- | -- |
| agenda.py:4358 | `reply is None ; t[0] == reply` | LOGIC | -- | -- |
| agenda.py:4359 | `idx < 0 ; len(before) != idx` | LOGIC | -- | -- |
| agenda.py:4364 | `rule_of[uci] == "dead"` | RULE | -- | -- |
| agenda.py:4366 | `rule_of[uci] == "mate"` | RULE | -- | -- |
| agenda.py:4366 | `BD.MATE_VALUE  in: closed = (BD.MATE_VALUE, "mate in one") if rule_of[uci] == "mate" else self._closed_by_rule(P2, db...` | CHESS FACT | [D3] existing: mate in one; nothing of theirs | cases 70/81, DECISION 0 |
| agenda.py:4367 | `closed is not None` | LOGIC | -- | -- |
| agenda.py:4372 | `self._quiet_leaf(fen)` | CHESS FACT | [D11] existing: chase: quiet-status estimate | cases 0/0, DECISION 0 |
| agenda.py:4374 | `GAME.from_key(fen).turn == me` | CHESS FACT | [D11] existing: chase: quiet-status estimate | cases 0/0, DECISION 0 |
| agenda.py:4378 | `64  in: leaves.append(dict(fen=fen, value=dbal, status=P2.balance(me) - 64, cls=cls, confidence=conf, estimator="quie...` | LOGIC | -- | -- |
| agenda.py:4380 | `leaves` | LOGIC | -- | -- |
| agenda.py:4380 | `value < 1` | CHESS FACT | [D11] existing: chase: quiet-status estimate | cases 0/0, DECISION 0 |
| agenda.py:4381 | `leaves` | LOGIC | -- | -- |
| agenda.py:4392 | `results.get(i) ; results[i].get("est")` | LOGIC | -- | -- |
| agenda.py:4393 | `cands` | LOGIC | -- | -- |
| agenda.py:4395 | `sort(... key=lambda t: (-t[0], t[1].move.uci()))` | HEURISTIC | -- | -- |
| agenda.py:4397 | `done ; max(d[0] for d in done) >= value` | LOGIC | -- | -- |
| agenda.py:4400 | `q.goal == "save"` | CHESS FACT | [A1s] PROPOSED: "goal: save material (don't lose)" (GOAL; reads material, attack) (beside existing: in check) | cases 78/83, DECISION 0 |
| agenda.py:4402 | `len(res["parts"]) > 4 ; 4  in: parts = " / ".join(res["parts"][:4]) + (" / ..." if len(res["parts"]) > 4 else "")` | LOGIC | -- | -- |
| agenda.py:4403 | `parts` | LOGIC | -- | -- |
| agenda.py:4409 | `self.best_est is None ; (value, 0) > self.best_est[:2]` | LOGIC | -- | -- |
| agenda.py:4409 | `(value, 0) == self.best_est[:2] ; claim.confidence > self.best_est[2].confidence` | HEURISTIC | -- | -- |
| agenda.py:4415 | `self._decision is not None` | LOGIC | -- | -- |
| agenda.py:4417 | `self.best ; 2  in: dec, est = (self.best[2] if self.best else None), False` | LOGIC | -- | -- |
| agenda.py:4418 | `TECHNIQUE ; self._tech is not None ; self.chase_depth == 0 ; self.best is not None` | LOGIC | -- | -- |
| agenda.py:4418 | `self.best[0] >= BD.MATE_VALUE` | CHESS FACT | [A2] PROPOSED: "stake: the terminal (mate = 100)" (VALUATION; reads movement, attack) | cases 97/97, DECISION 0 |
| agenda.py:4418 | `self.keeps( getattr(self, "status_info", None) or self.status(), self.best)[0]` | CHESS FACT | [A5] PROPOSED: "goal: the ladder -- don't lose, then win (status, line, keeps)" (GOAL; reads material, tempo) | cases 22/18, DECISION 35 |
| agenda.py:4425 | `ESTIMATE ; self.best_est is not None ; self.chase_depth == 0 ; self.best is not None` | LOGIC | -- | -- |
| agenda.py:4425 | `self.best[0] >= BD.MATE_VALUE` | CHESS FACT | [A2] PROPOSED: "stake: the terminal (mate = 100)" (VALUATION; reads movement, attack) | cases 97/97, DECISION 0 |
| agenda.py:4426 | `getattr(self, "status_info", None) ; self.status()` | LOGIC | -- | -- |
| agenda.py:4427 | `self.keeps(st, self.best)[0] ; self.keeps(st, self.best_est)[0]` | CHESS FACT | [A5] PROPOSED: "goal: the ladder -- don't lose, then win (status, line, keeps)" (GOAL; reads material, tempo) | cases 22/18, DECISION 35 |
| agenda.py:4428 | `2  in: dec, est = self.best_est[2], True` | LOGIC | -- | -- |
| agenda.py:4430 | `dec is None ; UNPROVED ; self._root ; self.chase_depth == 0 ; self.ctx is not None ; self.best_est is None` | LOGIC | -- | -- |
| agenda.py:4433 | `est` | LOGIC | -- | -- |
| agenda.py:4443 | `ESTIMATE ; self.chase_depth != 0 ; self.ctx is None ; self.ctx.est_rows` | LOGIC | -- | -- |
| agenda.py:4449 | `leaned is not None ; r["root_fen"] == root ; r["route"] == leaned.move ; r["depth"] == 0` | LOGIC | -- | -- |
| agenda.py:4453 | `QP.EST_LOG  in: AN._append_once(paths.experiment(QP.EST_LOG), "".join(rows))` | LOGIC | -- | -- |
| agenda.py:4460 | `b.piece_at(s) is not None ; b.piece_at(s).color != me` | RULE | -- | -- |
| agenda.py:4460 | `s != b.king(not me)` | CHESS FACT | [C1] PROPOSED: "names: king = the royal unit (Game.royal); excluded from the valuation" (VALUATION; reads material) | no printed signature: 8 rows |
| agenda.py:4460 | `d.kind in ("stopper", "blocked")` | CHESS FACT | [D1c] existing: stopper; blocked (+2) | cases 32/32, DECISION 0 |
| agenda.py:4463 | `m in have` | LOGIC | -- | -- |
| agenda.py:4463 | `b.is_castling(m)` | CHESS FACT | [D8f] PROPOSED: "forcing: a threat on the stopper / the pawn offered to it" (FORCING; reads attack, material, movement) | cases 15/15, DECISION 0 |
| agenda.py:4467 | `c.piece_at(s) is not None ; c.is_attacked_by(me, s) ; m.to_square in chess.SquareSet(c.attackers(me, s))` | RULE | -- | -- |
| agenda.py:4468 | `hit` | LOGIC | -- | -- |
| agenda.py:4468 | `W.base.see_for(c, m.to_square, not me) <= 0` | CHESS FACT | [D8f] PROPOSED: "forcing: a threat on the stopper / the pawn offered to it" (FORCING; reads attack, material, movement) | cases 15/15, DECISION 0 |
| agenda.py:4472 | `m.from_square == q.pattern["pawn"]` | CHESS FACT | [D8f] PROPOSED: "forcing: a threat on the stopper / the pawn offered to it" (FORCING; reads attack, material, movement) | cases 15/15, DECISION 0 |
| agenda.py:4473 | `c.piece_at(s) is not None ; s in c.attackers(not me, m.to_square)` | RULE | -- | -- |
| agenda.py:4474 | `by` | LOGIC | -- | -- |
| agenda.py:4474 | `W.base.see_for(c, m.to_square, not me) <= 0` | CHESS FACT | [D8f] PROPOSED: "forcing: a threat on the stopper / the pawn offered to it" (FORCING; reads attack, material, movement) | cases 15/15, DECISION 0 |
| agenda.py:4482 | `q.reads is None` | LOGIC | -- | -- |
| agenda.py:4482 | `q.goal in ("mate", "win")` | CHESS FACT | [A1x] PROPOSED: "goal: the kinds' entries carry their read convention" (GOAL; reads movement, attack, material) | no printed signature: 1 rows |
| agenda.py:4484 | `tempo` | LOGIC | -- | -- |
| agenda.py:4491 | `Defect('in check', ...)` | CHESS FACT | [D1d] existing: in check; promotes | cases 6/6, DECISION 0 |
| agenda.py:4493 | `m.to_square in ck` | CHESS FACT | [D21] existing: in check | cases 58/64, DECISION 0 |
| agenda.py:4494 | `m.to_square not in ck` | CHESS FACT | [D21] existing: in check | cases 58/64, DECISION 0 |
| agenda.py:4495 | `new_question('remove', ...)` | CHESS FACT | [D21] existing: in check | cases 58/64, DECISION 0 |
| agenda.py:4498 | `", ".join(f"{r.san} ({r.why})" for r in takes) ; "no capture of the checker"` | LOGIC | -- | -- |
| agenda.py:4501 | `takes` | LOGIC | -- | -- |
| agenda.py:4503 | `len(self.state.claims) == before ; others` | LOGIC | -- | -- |
| agenda.py:4504 | `new_question('cover', ...)` | CHESS FACT | [D21] existing: in check | cases 58/64, DECISION 0 |
| agenda.py:4510 | `len(self.state.claims) == before ; self._decide_by_smallest_loss(q) ; self._t(EVASION_CHASE, 3)` | LOGIC | -- | -- |
| agenda.py:4517 | `Defect('hangs', ...)` | CHESS FACT | [D1h] PROPOSED: "goal: save -- its defects hangs / threatened" (GOAL; reads attack, material) | cases 34/33, DECISION 0 |
| agenda.py:4519 | `new_question('remove', ...)` | CHESS FACT | [D1h] PROPOSED: "goal: save -- its defects hangs / threatened" (GOAL; reads attack, material) | cases 34/33, DECISION 0 |
| agenda.py:4526 | `len(self.state.claims) == before ; self._decide_by_smallest_loss(q)` | LOGIC | -- | -- |
| agenda.py:4535 | `Defect('threatened', ...)` | CHESS FACT | [D1h] PROPOSED: "goal: save -- its defects hangs / threatened" (GOAL; reads attack, material) | cases 34/33, DECISION 0 |
| agenda.py:4540 | `b.gives_check(mv) ; b.is_capture(mv)` | RULE | -- | -- |
| agenda.py:4540 | `p["value"] >= BD.MATE_VALUE ; mv.from_square in tg ; mv.to_square == m.from_square` | CHESS FACT | [D6] PROPOSED: "closing: a mate threat is parried only through its read set (FrameLemma.frame_unchanged)" (CLOSING; reads attack, movement, tempo) (beside existing: no parry keeps the gain; no parry move) | cases 28/39, DECISION 0 |
| agenda.py:4545 | `any(len(c.attackers(self.me, t)) > len(b.attackers(self.me, t)) for t in tg)` | CHESS FACT | [D6] PROPOSED: "closing: a mate threat is parried only through its read set (FrameLemma.frame_unchanged)" (CLOSING; reads attack, movement, tempo) (beside existing: no parry keeps the gain; no parry move) | cases 28/39, DECISION 0 |
| agenda.py:4547 | `new_question('remove', ...)` | CHESS FACT | [D6] PROPOSED: "closing: a mate threat is parried only through its read set (FrameLemma.frame_unchanged)" (CLOSING; reads attack, movement, tempo) (beside existing: no parry keeps the gain; no parry move) | cases 28/39, DECISION 0 |
| agenda.py:4552 | `self.their is not None` | LOGIC | -- | -- |
| agenda.py:4557 | `len(self.state.claims) == before ; self._decide_by_smallest_loss(q)` | LOGIC | -- | -- |
| agenda.py:4568 | `tok` | LOGIC | -- | -- |
| agenda.py:4568 | `tok[0].isdigit()` | CHESS FACT | [C3] PROPOSED: "names: check / capture bound to the kernel (Game.gives_check, Game.is_capture), not to the notation" (FORCING; reads attack, movement) | no printed signature: 2 rows |
| agenda.py:4574 | `c.turn != self.me` | RULE | -- | -- |
| agenda.py:4577 | `v == m.from_square` | LOGIC | -- | -- |
| agenda.py:4582 | `k is not None` | LOGIC | -- | -- |
| agenda.py:4584 | `chess.BB_KING_ATTACKS  in: for s in [k] + list(chess.SquareSet(chess.BB_KING_ATTACKS[k])): for a in c.attackers(not s...` | CHESS FACT | [C6] PROPOSED: "closing: the king's step (Game.adjacent) in a mate threat's frame" (CLOSING; reads movement, attack) | cases 67/67, DECISION 0 |
| agenda.py:4587 | `self.board.piece_at(orig) is not None ; self.board.piece_at(orig).color != self.me` | RULE | -- | -- |
| agenda.py:4599 | `q.pattern.get("value", 0) >= BD.MATE_VALUE ; q.pattern.get("threat")` | CHESS FACT | [D6] PROPOSED: "closing: a mate threat is parried only through its read set (FrameLemma.frame_unchanged)" (CLOSING; reads attack, movement, tempo) (beside existing: no parry keeps the gain; no parry move) | cases 28/39, DECISION 0 |
| agenda.py:4608 | `r.move.to_square == src ; reads is not None ; r.move.from_square in reads ; r.move.to_square in reads` | CHESS FACT | [D6] PROPOSED: "closing: a mate threat is parried only through its read set (FrameLemma.frame_unchanged)" (CLOSING; reads attack, movement, tempo) (beside existing: no parry keeps the gain; no parry move) | cases 28/39, DECISION 0 |
| agenda.py:4609 | `reads is None ; touches` | LOGIC | -- | -- |
| agenda.py:4610 | `q.pattern.get("line") ; ""` | LOGIC | -- | -- |
| agenda.py:4610 | `r.move.to_square in self._mate_line_pieces(q.pattern.get("line") or "")` | CHESS FACT | [D6] PROPOSED: "closing: a mate threat is parried only through its read set (FrameLemma.frame_unchanged)" (CLOSING; reads attack, movement, tempo) (beside existing: no parry keeps the gain; no parry move) | cases 28/39, DECISION 0 |
| agenda.py:4611 | `touches` | LOGIC | -- | -- |
| agenda.py:4611 | `c.is_checkmate()` | RULE | -- | -- |
| agenda.py:4613 | `c.is_check() ; b.is_capture(r.move)` | CHESS FACT | [D6] PROPOSED: "closing: a mate threat is parried only through its read set (FrameLemma.frame_unchanged)" (CLOSING; reads attack, movement, tempo) (beside existing: no parry keeps the gain; no parry move) | cases 28/39, DECISION 0 |
| agenda.py:4618 | `sorted(... key=lambda r: (-(P0.push(r.move.uci()).balance(self.me) - base), r.move.uci()))` | CHESS FACT | [D6] PROPOSED: "closing: a mate threat is parried only through its read set (FrameLemma.frame_unchanged)" (CLOSING; reads attack, movement, tempo) (beside existing: no parry keeps the gain; no parry move) | cases 28/39, DECISION 0 |
| agenda.py:4620 | `self.keep >= 1` | CHESS FACT | [A6] PROPOSED: "valuation: a gain is at least one unit of material" (VALUATION; reads material) (beside existing: no claim of ours and no material gain) | cases 80/80, DECISION 0 |
| agenda.py:4623 | `bind is None` | LOGIC | -- | -- |
| agenda.py:4626 | `r.move.uci() in counters` | LOGIC | -- | -- |
| agenda.py:4633 | `c2.is_check()` | RULE | -- | -- |
| agenda.py:4637 | `cl_r is not None` | LOGIC | -- | -- |
| agenda.py:4637 | `t_r >= BD.MATE_VALUE` | CHESS FACT | [D6] PROPOSED: "closing: a mate threat is parried only through its read set (FrameLemma.frame_unchanged)" (CLOSING; reads attack, movement, tempo) (beside existing: no parry keeps the gain; no parry move) | cases 28/39, DECISION 0 |
| agenda.py:4640 | `still is not None` | LOGIC | -- | -- |
| agenda.py:4643 | `Claim('save', ...)` | CHESS FACT | [D6] PROPOSED: "closing: a mate threat is parried only through its read set (FrameLemma.frame_unchanged)" (CLOSING; reads attack, movement, tempo) (beside existing: no parry keeps the gain; no parry move) | cases 28/39, DECISION 0 |
| agenda.py:4650 | `self._chase(q)` | LOGIC | -- | -- |
| agenda.py:4650 | `self._keep_goal(q)` | CHESS FACT | [A10] PROPOSED: "goal: save = keep the balance (target 0)" (GOAL; reads material, tempo) | cases 47/56, DECISION 0 |
| agenda.py:4652 | `order ; notes ; len(order) == 1 ; 3  in: self._kill(q, f"no parry keeps the gain ({len(order)} parr{'y' if len(order)...` | LOGIC | -- | -- |
| agenda.py:4662 | `lift ; self.chase_depth == 0` | CHESS FACT | [D8h] existing: unstoppablePasser2_sound; Ideas.Race.race_sound (+4) | cases 41/47, DECISION 3 |
| agenda.py:4663 | `facts.get("lifted") is not None ; q.defects` | CHESS FACT | [D8h] existing: unstoppablePasser2_sound; Ideas.Race.race_sound (+4) | cases 41/47, DECISION 3 |
| agenda.py:4668 | `q.defects` | LOGIC | -- | -- |
| agenda.py:4669 | `kd is not None` | LOGIC | -- | -- |
| agenda.py:4671 | `lifted is not None ; lifted` | LOGIC | -- | -- |
| agenda.py:4673 | `lifted is not None` | LOGIC | -- | -- |
| agenda.py:4673 | `q.defects` | CHESS FACT | [D8h] existing: unstoppablePasser2_sound; Ideas.Race.race_sound (+4) | cases 41/47, DECISION 3 |
| agenda.py:4675 | `self._plan_claim(q, sq)` | LOGIC | -- | -- |
| agenda.py:4677 | `q.defects` | LOGIC | -- | -- |
| agenda.py:4678 | `self._race_claim(q, sq)` | LOGIC | -- | -- |
| agenda.py:4683 | `rs` | LOGIC | -- | -- |
| agenda.py:4685 | `f is not None` | LOGIC | -- | -- |
| agenda.py:4688 | `d.kind == "stopper"` | CHESS FACT | [D1c] existing: stopper; blocked (+2) | cases 32/32, DECISION 0 |
| agenda.py:4689 | `stop` | CHESS FACT | [D8h] existing: unstoppablePasser2_sound; Ideas.Race.race_sound (+4) | cases 41/47, DECISION 3 |
| agenda.py:4694 | `self._chase(q)` | LOGIC | -- | -- |
| agenda.py:4699 | `sq in ps` | CHESS FACT | [D8h] existing: unstoppablePasser2_sound; Ideas.Race.race_sound (+4) | cases 41/47, DECISION 3 |
| agenda.py:4700 | `n == 1` | RULE | -- | -- |
| agenda.py:4700 | `chess.QUEEN  in: push = chess.Move(sq, facts["path"][0], promotion=chess.QUEEN if n == 1 else None)` | CHESS FACT | [A3] PROPOSED: "stake: promotion (queen less pawn = 8)" (VALUATION; reads material, movement) | cases 35/35, DECISION 0 |
| agenda.py:4703 | `n > 1 ; rest` | LOGIC | -- | -- |
| agenda.py:4704 | `Claim('promote', ...)` | CHESS FACT | [A1p] PROPOSED: "goal: promote (the rule)" (GOAL; reads movement, tempo) | cases 48/49, DECISION 0 |
| agenda.py:4704 | `R.STAKE_PROMOTE  in: claim = Claim("promote", san, R.STAKE_PROMOTE, LS.name("unstoppablePasser2_sound"), line, move=p...` | CHESS FACT | [A3] PROPOSED: "stake: promotion (queen less pawn = 8)" (VALUATION; reads material, movement) | cases 35/35, DECISION 0 |
| agenda.py:4709 | `self._better(claim.value, 0, claim)` | LOGIC | -- | -- |
| agenda.py:4713 | `self._race_claim(q, sq)` | LOGIC | -- | -- |
| agenda.py:4716 | `ex is not None` | LOGIC | -- | -- |
| agenda.py:4718 | `lifted is not None` | LOGIC | -- | -- |
| agenda.py:4719 | `self.ctx is not None ; self.ctx.deadline is not None ; time.monotonic() >= self.ctx.deadline` | LOGIC | -- | -- |
| agenda.py:4720 | `late ; self._chase(q)` | LOGIC | -- | -- |
| agenda.py:4728 | `h[k] ; PV.HYPS  in: bad = [k for k in PV.HYPS if not h[k]]` | LOGIC | -- | -- |
| agenda.py:4731 | `k == "enemyPieces"` | NAME | -- | -- |
| agenda.py:4733 | `k == "bishopsFar"` | NAME | -- | -- |
| agenda.py:4735 | `k == "noSliders"` | NAME | -- | -- |
| agenda.py:4740 | `why` | LOGIC | -- | -- |
| agenda.py:4751 | `PL.PLAN` | LOGIC | -- | -- |
| agenda.py:4754 | `q.defects ; sq in dict(W.proved_passers(b, bishops=True))` | CHESS FACT | [D8h] existing: unstoppablePasser2_sound; Ideas.Race.race_sound (+4) | cases 41/47, DECISION 3 |
| agenda.py:4757 | `pl.moves` | LOGIC | -- | -- |
| agenda.py:4760 | `pl.closed` | LOGIC | -- | -- |
| agenda.py:4760 | `pl.moves[0].from_square == sq` | CHESS FACT | [D8h] existing: unstoppablePasser2_sound; Ideas.Race.race_sound (+4) | cases 41/47, DECISION 3 |
| agenda.py:4763 | `Claim('promote', ...)` | CHESS FACT | [A1p] PROPOSED: "goal: promote (the rule)" (GOAL; reads movement, tempo) | cases 48/49, DECISION 0 |
| agenda.py:4763 | `R.STAKE_PROMOTE  in: claim = Claim("promote", san, R.STAKE_PROMOTE, LS.name("plan_sound"), f"{pl.line(b)}: {len(pl.we...` | CHESS FACT | [A3] PROPOSED: "stake: promotion (queen less pawn = 8)" (VALUATION; reads material, movement) | cases 35/35, DECISION 0 |
| agenda.py:4768 | `self._better(claim.value, 0, claim)` | LOGIC | -- | -- |
| agenda.py:4772 | `self._root ; self.chase_depth == 0 ; pl.immune` | LOGIC | -- | -- |
| agenda.py:4774 | `fixes` | LOGIC | -- | -- |
| agenda.py:4783 | `RC.RACE` | LOGIC | -- | -- |
| agenda.py:4783 | `b.turn != self.me` | RULE | -- | -- |
| agenda.py:4786 | `rc is None` | LOGIC | -- | -- |
| agenda.py:4790 | `Claim('promote', ...)` | CHESS FACT | [A1p] PROPOSED: "goal: promote (the rule)" (GOAL; reads movement, tempo) | cases 48/49, DECISION 0 |
| agenda.py:4790 | `R.STAKE_PROMOTE  in: claim = Claim("promote", san, R.STAKE_PROMOTE, LS.name("Ideas.Race.race_sound"), line, move=rc.f...` | CHESS FACT | [A3] PROPOSED: "stake: promotion (queen less pawn = 8)" (VALUATION; reads material, movement) | cases 35/35, DECISION 0 |
| agenda.py:4796 | `self._better(claim.value, 0, claim)` | LOGIC | -- | -- |
| agenda.py:4809 | `facts["pushes"] != 1` | CHESS FACT | [D8e] existing: promotion_exchange: the push, then the x-ray ... | cases 2/3, DECISION 0 |
| agenda.py:4811 | `chess.QUEEN  in: push = chess.Move(sq, facts["promo"], promotion=chess.QUEEN)` | CHESS FACT | [A3] PROPOSED: "stake: promotion (queen less pawn = 8)" (VALUATION; reads material, movement) | cases 35/35, DECISION 0 |
| agenda.py:4812 | `push not in b.legal_moves` | RULE | -- | -- |
| agenda.py:4817 | `c.is_checkmate() ; any(m.to_square == facts["promo"] for m in c.legal_moves)` | RULE | -- | -- |
| agenda.py:4820 | `R.STAKE_PROMOTE  in: net = R.STAKE_PROMOTE - lost` | CHESS FACT | [A3] PROPOSED: "stake: promotion (queen less pawn = 8)" (VALUATION; reads material, movement) | cases 35/35, DECISION 0 |
| agenda.py:4821 | `lost <= 0 ; net < 1` | CHESS FACT | [D8e] existing: promotion_exchange: the push, then the x-ray ... | cases 2/3, DECISION 0 |
| agenda.py:4824 | `Claim('win', ...)` | CHESS FACT | [D8e] existing: promotion_exchange: the push, then the x-ray ... | cases 2/3, DECISION 0 |
| agenda.py:4825 | `R.STAKE_PROMOTE  in: claim = Claim("win", san, net, LS.name("promotion_exchange: the push, then the x-ray exchange on...` | CHESS FACT (text only) | [A3] PROPOSED: "stake: promotion (queen less pawn = 8)" (VALUATION; reads material, movement) | cases 35/35, DECISION 0 |
| agenda.py:4829 | `q.reads ; ()` | LOGIC | -- | -- |
| agenda.py:4831 | `self._better(net, 0, claim)` | LOGIC | -- | -- |
| agenda.py:4846 | `Defect('promotes', ...)` | CHESS FACT | [D1d] existing: in check; promotes | cases 6/6, DECISION 0 |
| agenda.py:4852 | `m.to_square == pw` | CHESS FACT | [D8b] existing: no stopper in time | cases 0/0, DECISION 0 |
| agenda.py:4854 | `m.from_square == ksq` | CHESS FACT | [D8b] existing: no stopper in time | cases 0/0, DECISION 0 |
| agenda.py:4857 | `W.base.word(c, ("dist", "L", ("promo", GAME.loc_name(pw))), them) < kd` | CHESS FACT | [D8b] existing: no stopper in time | cases 0/0, DECISION 0 |
| agenda.py:4859 | `new_question('remove', ...)` | CHESS FACT | [D8b] existing: no stopper in time | cases 0/0, DECISION 0 |
| agenda.py:4862 | `", ".join(f"{r.san} ({r.why})" for r in runs) ; "no such move"` | LOGIC | -- | -- |
| agenda.py:4867 | `kd == n + 1` | CHESS FACT | [D8b] existing: no stopper in time | cases 0/0, DECISION 0 |
| agenda.py:4869 | `kd > n + 1` | CHESS FACT | [D8b] existing: no stopper in time | cases 0/0, DECISION 0 |
| agenda.py:4874 | `self.their is not None` | LOGIC | -- | -- |
| agenda.py:4877 | `runs` | LOGIC | -- | -- |
| agenda.py:4879 | `len(self.state.claims) == before` | LOGIC | -- | -- |
| agenda.py:4881 | `kd > n + 1` | CHESS FACT | [D8b] existing: no stopper in time | cases 0/0, DECISION 0 |
| agenda.py:4892 | `q` | LOGIC | -- | -- |
| agenda.py:4900 | `self.best is not None ; 2  in: fid = self.claim_fid.get(id(self.best[2])) if self.best is not None else None` | LOGIC | -- | -- |
| agenda.py:4901 | `fid is not None` | LOGIC | -- | -- |
| agenda.py:4902 | `2  in: out.append(f"justification graph of the decision ({self.best[2].what}):")` | LOGIC | -- | -- |

### ponder.py

| file:line | text | class | store entry | decisions depending |
|---|---|---|---|---|
| ponder.py:27 | `stop is not None ; stop()` | LOGIC | -- | -- |
| ponder.py:28 | `ctx.abort` | LOGIC | -- | -- |
| ponder.py:30 | `ctx.t0 is None` | LOGIC | -- | -- |
| ponder.py:41 | `new_question('win', ...)` | CHESS FACT | [A1w] PROPOSED: "goal: win material" (GOAL; reads material, attack) | cases 83/86, DECISION 0 |
| ponder.py:43 | `r["promote"]` | LOGIC | -- | -- |
| ponder.py:48 | `res.get("gap") is None` | LOGIC | -- | -- |
| ponder.py:56 | `b is None` | LOGIC | -- | -- |
| ponder.py:56 | `b.turn != (us == "w")` | NAME | -- | -- |
| ponder.py:58 | `f.kind != "gap"` | LOGIC | -- | -- |
| ponder.py:59 | `before` | LOGIC | -- | -- |
| ponder.py:63 | `30  in: ctx.left = 1 << 30` | LOGIC | -- | -- |
| ponder.py:65 | `Agenda(..., min_stake=1) (a recomputation asks the roots worth one unit)` | CHESS FACT | [A6] PROPOSED: "valuation: a gain is at least one unit of material" (VALUATION; reads material) (beside existing: no claim of ours and no material gain) | cases 80/80, DECISION 0 |
| ponder.py:69 | `f.kind != "gap"` | LOGIC | -- | -- |
| ponder.py:79 | `x.kind == "gap"` | LOGIC | -- | -- |
| ponder.py:80 | `time.monotonic() >= end ; stop is not None ; stop()` | LOGIC | -- | -- |
| ponder.py:82 | `f.resolved ; f.superseded` | LOGIC | -- | -- |
| ponder.py:85 | `ok` | LOGIC | -- | -- |
| ponder.py:86 | `log` | LOGIC | -- | -- |
| ponder.py:87 | `ok` | LOGIC | -- | -- |
| ponder.py:88 | `time.monotonic() < end ; stop is not None ; stop()` | LOGIC | -- | -- |
| ponder.py:91 | `n ; log` | LOGIC | -- | -- |
| ponder.py:111 | `board.turn` | NAME | -- | -- |
| ponder.py:113 | `seconds` | LOGIC | -- | -- |
| ponder.py:113 | `3600.0  in: end = t0 + (seconds if seconds else 3600.0)` | HEURISTIC | -- | -- |
| ponder.py:114 | `emit ; lambda t: None` | LOGIC | -- | -- |
| ponder.py:115 | `2  in: g = drain(ctx, us, max(end - t0, 0) / 2, log=lambda t: say("background: " + t), stop=lambda: self.stopped)` | HEURISTIC | -- | -- |
| ponder.py:116 | `g["queue"]` | LOGIC | -- | -- |
| ponder.py:123 | `board.is_check() ; c.is_check()` | RULE | -- | -- |
| ponder.py:123 | `order.append((-touched_stake, not board.is_capture(mv), not board.is_check() and not c.is_check(), uci, ...)) (the po...` | CHESS FACT | [D22] PROPOSED: "heuristic: the forcing order (checks, captures, threats; recaptures first)" (HEURISTIC; reads attack, material) | no printed signature: 6 rows |
| ponder.py:124 | `sort(... key=lambda t: t[:4])` | CHESS FACT | [D22] PROPOSED: "heuristic: the forcing order (checks, captures, threats; recaptures first)" (HEURISTIC; reads attack, material) | no printed signature: 6 rows |
| ponder.py:127 | `self.stopped ; time.monotonic() >= end` | LOGIC | -- | -- |
| ponder.py:129 | `_pos_key(c.fen()) in ctx.memo` | LOGIC | -- | -- |
| ponder.py:129 | `c.is_game_over()` | RULE | -- | -- |
| ponder.py:133 | `30  in: ctx.left = 1 << 30` | LOGIC | -- | -- |
| ponder.py:137 | `h._inner(c.fen(), 1) (the reply read for a gain of one unit)` | CHESS FACT | [A6] PROPOSED: "valuation: a gain is at least one unit of material" (VALUATION; reads material) (beside existing: no claim of ours and no material gain) | cases 80/80, DECISION 0 |
| ponder.py:140 | `self.stopped` | LOGIC | -- | -- |
| ponder.py:144 | `ans ; ans.get("claim") is not None` | LOGIC | -- | -- |

## 4. The move order

The CHESS FACT families in the order they should leave agenda.py for the store, grouped into plan 75.2's stages: (a) the goal ladder and the goal kinds, (b) the stops, (c) the names, (d) the rest. Inside a stage: most cases passing through first (tier 0 + tier 2), then DECISION lines citing, then rows; families with no printed signature after, by rows. 'rows' = this audit's table rows of the family (its call sites in agenda.py / ponder.py).

DECISIONS DEPENDING (the method). The trace guard's snapshots of the 73.5b acceptance (`guard_after/` in session b30a4c85's scratchpad: ten + wide = 109 cases, tier 0 and tier 2, the count budget, PYTHONHASHSEED=0; agenda.py unchanged since) were read, not regenerated. A case PASSES THROUGH a family when its trace or its word list (the claim lemmas, kill and GAP reasons of every agenda of the run, `<id>.words.txt`) matches the family's printed signature (the regex in the last column: question heads, stop texts, kill and GAP reasons); a DECISION line CITES it when the signature (or the family's lemma) is on the case's `DECISION` line. A family with no printed signature is ranked by its rows, and says so. Limits: the count-budget snapshots never run the status driver, the pool or the judge (no clock), so the ladder's and the judge's counts are lower bounds; an inner agenda's chase notes are not printed in the root trace (the walk of checks shows 0); 'passes through' is an upper bound of 'depends on'. A DECISION line names only the deciding lemma (the chase, a schema, the unfinished line, 'none'), so most families are cited by none.

### (a) the goal ladder and the goal kinds

| # | family | what is written as code | store entry | cases t0 / t2 (of 109) | DECISION lines (of 218) | rows | signature |
|---|---|---|---|---|---|---|---|
| 1 | A1m | goal kind MATE: the mate roots (every checking placement incl. castling's rook and the promoted piece; the reach roots two moves away) | PROPOSED "goal: mate (the rule's terminal)" (GOAL; reads movement, attack) | 100 / 100 | 0 | 18 | `\bmate on [a-h][1-8] \(` |
| 2 | A2 | the stake of MATE: 100 (MATE_VALUE / STAKE_MATE) as +infinity -- a mate ends the agenda, mate threats stake 100, value >= 100 means 'a mate' | PROPOSED "stake: the terminal (mate = 100)" (VALUATION; reads movement, attack) | 97 / 97 | 0 | 34 | `\(stake 100\)\|value 100\b\|\+100\b\|-> 100\b` |
| 3 | A1w | goal kind WIN: a root per unit of theirs but the king, stake = its worth; the checker as a win target; direct captures (not promotions) as its routes | PROPOSED "goal: win material" (GOAL; reads material, attack) | 83 / 86 | 0 | 12 | `\bwin their [KQRBNP] on [a-h][1-8]` |
| 4 | A4 | MATERIAL as the stake currency: unit worth (P1 N3 B3 R5 Q9, the king none) for win / save stakes and every worth-ordered choice | PROPOSED "valuation: material (unit worth)" (VALUATION; reads material) | 82 / 85 | 0 | 4 | `\(worth \d\)\|capture \d\b` |
| 5 | A1s | goal kind SAVE and its sub-kinds: save the king (in check), save a hanging piece (the biggest), save against their pass threat, stop their pawn | PROPOSED "goal: save material (don't lose)" (GOAL; reads material, attack); beside existing: in check | 78 / 83 | 0 | 12 | `\bsave our \|\bstop their P ` |
| 6 | A6 | THE GAIN UNIT: a route or claim gains when its value is >= 1 (one pawn); need = 1 - dbal; < 1 kills; a static gain is dbal >= 1 | PROPOSED "valuation: a gain is at least one unit of material" (VALUATION; reads material); beside existing: no claim of ours and no material gain | 80 / 80 | 0 | 13 | `=> value \d\|no claim of ours and no material gain\|less than 1\|value \d+ \(no gain` |
| 7 | A10 | SAVE = KEEP: a save is chased with keep semantics (target 0; a kept claim worth the stake at the root, what it kept inside) | PROPOSED "goal: save = keep the balance (target 0)" (GOAL; reads material, tempo) | 47 / 56 | 0 | 9 | `keeps [+-]\d\|=> value 0\b` |
| 8 | A1p | goal kind PROMOTE: a root per pawn with pushes <= PROMOTE_MAX_PUSHES, or a far passer the tempo arithmetic calls unstoppable | PROPOSED "goal: promote (the rule)" (GOAL; reads movement, tempo) | 48 / 49 | 0 | 21 | `\bpromote our P on ` |
| 9 | A3 | the stake of PROMOTION: 8 = queen less pawn (STAKE_PROMOTE); the promotion piece is always the queen | PROPOSED "stake: promotion (queen less pawn = 8)" (VALUATION; reads material, movement) | 35 / 35 | 0 | 14 | `\(stake 8\)\|CLAIM promote` |
| 10 | A5 | THE STATUS LADDER: s0 = balance - their pass threat; winning >= 1 / level 0 / losing < 0; the line 1 or 0; keeps = balance + value >= line; raise to win; the need = line - balance; 'losing' takes one route | PROPOSED "goal: the ladder -- don't lose, then win (status, line, keeps)" (GOAL; reads material, tempo) | 22 / 18 | 35 | 20 | `^STATUS\|no proved claim:\|DECISION ESTIMATED\|DECISION TECHNIQUE\|status (winning\|level\|losing)` |
| 11 | A8 | THE DRAW IS LEVEL: a repetition when behind closes as a draw worth minus the balance; ahead or level no claim | existing: repetition | 5 / 14 | 0 | 2 | `repetition` |
| 12 | A1d | claim kind DRAW: a repetition on the line is a draw claim; a route through one becomes kind 'draw' | existing: repetition | 5 / 14 | 0 | 1 | `repetition` |
| 13 | A1f | goal kind FORK: the fork square as a root (root agenda only, not in check, no need) | existing: fork; fork: two threats, one move (Ideas.ForkSquare.one_move_touches_one); every reply read by the exchange rule (Ideas.ExchangeLaw) | 5 / 5 | 0 | 3 | `\bfork on [a-h][1-8] with our ` |
| 14 | A9 | THE TERMINALS' VALUES: they mate us = -100 / a kill, stalemate = 0 / a kill, our mate = +100 | existing: they mate us; stalemate; mate in one | 1 / 1 | 0 | 3 | `they mate us\|stalemate\|\bmates\b` |
| 15 | A1t | goal kind TECHNIQUE: a store TECHNIQUE entry asked at the root (tier 0, no table claim, one per class); its claim decides only when no proved claim keeps the status | existing: mined technique KRvK; no measure-dropping move | 0 / 0 | 0 | 3 | `^TECHNIQUE ` |
| 16 | A7 | THE NEED PROPAGATED: min_stake / _target / the chain of two (_top_other); a rule closes only at a value that can rescue the route; a mate-only read; the reply's need formula | PROPOSED "goal: the need (what a claim must bring), propagated into the chase" (GOAL; reads material) | -- | -- | 15 | none (ranked by rows) |
| 17 | A1x | the goal kinds as a set: which kinds record a read set when killed (mate, win) | PROPOSED "goal: the kinds' entries carry their read convention" (GOAL; reads movement, attack, material) | -- | -- | 1 | none (ranked by rows) |

### (b) the stops

| # | family | what is written as code | store entry | cases t0 / t2 (of 109) | DECISION lines (of 218) | rows | signature |
|---|---|---|---|---|---|---|---|
| 1 | B4 | THE FUSE'S SHAPE: budget = our pieces x 8 (the squares round their king) + our captures; fuse = FUSE x budget x 2^tier; equal route shares; a root at most half while another can beat the best | existing: budget; chase fuse | 85 / 84 | 0 | 3 | `chase fuse\|STOP \(c\): budget\|fuse: this root is charged` |
| 2 | B3 | TEMPO ARITHMETIC (the goal judge): a mate / win chain may spend max(1, defects, a fix's steps) prep moves; a bigger goal or a fresh win after a capture restarts; a recapture is one event | existing: tempo arithmetic; tempo | 61 / 74 | 0 | 7 | `tempo arithmetic` |
| 3 | B2 | THE SECOND CHECK: one intermezzo check of theirs is followed, a second in a row is not; the count of checks evaded (reset only with EVADED_RESET) | existing: a second check of theirs in a row | 30 / 56 | 0 | 4 | `a second check of theirs in a row` |
| 4 | B1 | THE CHASE CLOCK: our moves since an irreversible event (a capture by either side not bad for us -- dbal >= 0 -- or our pawn move); min(CHASE_CLOCK, 4) from tier 3 | existing: the chase clock | 5 / 9 | 0 | 6 | `the chase clock` |
| 5 | B5 | THE WALK OF CHECKS: below RULE2_FROM's depth a check must fix a defect of its goal (cover the flight, take the checker's taker, deflect, attack the target, skewer, a promote defect) | PROPOSED "stop: a check that fixes no defect of its goal (the walk of checks)" (STOP; reads attack, movement, tempo) | 0 / 0 | 0 | 12 | `not a route of` |
| 6 | B8 | STILL FORCING (the resume rule): a step is forcing when our move named a threat and the reply parried it (in check: parried) | PROPOSED "stop: still forcing -- every reply parried the named threat (Ideas.Forcing.stillForcing)" (STOP; reads attack, tempo) | 0 / 0 | 0 | 2 | `line went quiet\|RESUME` |
| 7 | B7 | THE JUDGE'S FORCEDNESS: a root route decides when its word is forced (FORCED_WORDS) and the policy ranks it at least 1/n; parries priced by kind read off the SAN (+ / # check, x capture, else quiet) | existing: cut by the judge | 0 / 0 | 0 | 1 | `judge:` |
| 8 | B6 | THE FOLLOW LIST: which goals chase at which depth (mate / win / promote; saves to depth 1; win / promote not past depth 2 without DEEP_WIN_CHASE); inner levels follow only FORCED_WORDS (+ threat from tier 1, + push for a promotion) | PROPOSED "stop: the chase's follow list (forced words below the root)" (STOP; reads attack, movement, tempo); beside existing: check; capture; threat of mate | -- | -- | 10 | none (ranked by rows) |

### (c) the names (bind the name to the kernel; the knowledge goes to the valuation or the store)

| # | family | what is written as code | store entry | cases t0 / t2 (of 109) | DECISION lines (of 218) | rows | signature |
|---|---|---|---|---|---|---|---|
| 1 | C6 | THE KING'S ZONE: BB_KING_ATTACKS of their king is a mate threat's frame; of ours, the mating net's guards | PROPOSED "closing: the king's step (Game.adjacent) in a mate threat's frame" (CLOSING; reads movement, attack) | 67 / 67 | 0 | 2 | `local after \S+ \(mate on ` |
| 2 | C1 | THE KING IS NOT MATERIAL: `piece_type != chess.KING` excludes the royal unit from targets, worth, hanging, attacked pieces, stoppers | PROPOSED "names: king = the royal unit (Game.royal); excluded from the valuation" (VALUATION; reads material) | -- | -- | 8 | none (ranked by rows) |
| 3 | C2 | THE PAWN BY NAME: `piece_type == chess.PAWN` / `b.pawns` -- the unit that promotes (Game.promotes), still a pawn after the reply | PROPOSED "names: pawn = the unit that promotes (Game.promotes)" (GOAL; reads movement) | -- | -- | 2 | none (ranked by rows) |
| 4 | C3 | CHECK AND CAPTURE READ OFF THE NOTATION: SAN '+' / '#' / 'x', uci sniffing, SAN replay of a line | PROPOSED "names: check / capture bound to the kernel (Game.gives_check, Game.is_capture), not to the notation" (FORCING; reads attack, movement) | -- | -- | 2 | none (ranked by rows) |

### (d) the rest

| # | family | what is written as code | store entry | cases t0 / t2 (of 109) | DECISION lines (of 218) | rows | signature |
|---|---|---|---|---|---|---|---|
| 1 | D17 | THE LOCAL GOAL TRACKED: the target that ran is followed (same kind, theirs); the pawn still ours and a pawn; a delay = a check or capture touching none of the frame, priced by the exchange | existing: closed by the frame | 81 / 82 | 0 | 5 | `local after ` |
| 2 | D3 | CLOSE BY RULE: a mate in one closes at 100; a capture whose exchange reaches the need with nothing of theirs closes; not in check, not when already ahead; the goal not met by the exchange; the capture's duty | existing: mate in one; nothing of theirs | 70 / 81 | 0 | 10 | `closed by rule\|not closed by rule\|nothing of theirs\|mate in one` |
| 3 | D1a | MATE DEFECT KINDS read by name (flight, checker takeable, interposition) | existing: flight; checker takeable; interposition; no check | 75 / 75 | 0 | 4 | `defect (flight\|checker takeable\|interposition\|no check)` |
| 4 | D1b | MATERIAL DEFECT KINDS read by name (defended, not attacked) | existing: defended; not attacked | 68 / 68 | 0 | 3 | `defect (defended\|not attacked)` |
| 5 | D21 | IN CHECK: the roots are the save-king question and the win of a checker some legal move takes; the evasions (every legal move) are the routes | existing: in check | 58 / 64 | 0 | 10 | `save our king \(in check\)\|\(it gives check\)\|\(evade\)` |
| 6 | D2 | THE STATIC LEAF: a gain standing only as a balance is read by their pass; their threats are save roots; the parry read once more; their open question; the stuck unit; in check no pass (a GAP) | existing: ... stands only as a balance, their ... wins ... (lemma ...), no parry; ... stands only as a balance and their ... (wins ..., lemma ...) is not parried YET: ...; ... stands only as a balance and we are in check (no pass to read their threat, no claim of ours): not parried YET; their open question; stuck_reach | 56 / 63 | 0 | 17 | `stands only as a balance\|static leaves read from their side\|not parried YET\|their open question\|stuck_reach` |
| 7 | D8h | THE PASSER WORDS' ORDER: plan_sound, the race, the proved passer, the lift and the failing hypotheses (enemyPieces, bishopsFar, noSliders) | existing: unstoppablePasser2_sound; Ideas.Race.race_sound; plan_sound; lifted; stopped; the defects say unstoppable but the lemma unstoppablePasser2_sound does not fire | 41 / 47 | 3 | 9 | `unstoppablePasser2_sound\|race_sound\|plan_sound\|lifted\|stopped: ` |
| 8 | D23 | THE EXCHANGE'S PRINCIPAL LINES: cheapest capturer first, the swap list's optimal stop (Ideas.ExchangeLaw in Python); their claims after our capture, recaptures within the credit excluded | PROPOSED "valuation: the exchange law's principal lines (swap list)" (VALUATION; reads attack, material); beside existing: nothing of theirs | 29 / 48 | 0 | 9 | `the capturing piece left a duty` |
| 9 | D6 | THE MATE THREAT'S PARRY: a parry must touch their claim's read set, take the threatener or mate; a counter only if their mate is gone after every reply; the parry set of a threat (move the target, check, take, add a guard) | PROPOSED "closing: a mate threat is parried only through its read set (FrameLemma.frame_unchanged)" (CLOSING; reads attack, movement, tempo); beside existing: no parry keeps the gain; no parry move | 28 / 39 | 0 | 10 | `no parry keeps the gain\|a counter; their mate\|-> PARRY` |
| 10 | D1h | SAVE DEFECT KINDS not in the store (hangs, threatened) | PROPOSED "goal: save -- its defects hangs / threatened" (GOAL; reads attack, material) | 34 / 33 | 0 | 3 | `defect (hangs\|threatened)` |
| 11 | D1c | PROMOTE DEFECT KINDS read by name (stopper, blocked, path attacked, promo square attacked) | existing: stopper; blocked; path attacked; promo square attacked | 32 / 32 | 0 | 4 | `defect (stopper\|blocked\|path attacked\|promo square attacked)` |
| 12 | D25 | THE STANDING KILL: a reply that killed an earlier route with no claim kills this one where its reads are untouched (it took no less, the route needs no less) | existing: standing kill | 28 / 26 | 0 | 2 | `standing kill` |
| 13 | D15 | THEIR VIEW OF OUR FUTURE: a route's net = bound value - their best claim; killed when their claim >= our stake and the net <= 0 | existing: verify: no route binds | 27 / 26 | 0 | 2 | `their view:\|the future loses it\|KILLED as a route` |
| 14 | D8d | THE STOPPER STANDS: a promote route ends where the same stopper / blocker defect stands and no new fix appears | existing: the ... defect stands (...) and no fix is new there | 25 / 27 | 0 | 3 | `defect stands` |
| 15 | D8c | THE PROMOTE CHASE: a reply is answered by the promote claim, a claim of the stake, giving up the stake, or winning the material back; else 'the promotion is not claimed' | existing: the promotion is not claimed | 23 / 23 | 0 | 7 | `the promotion is not claimed\|gave up at least the stake\|wins the material back` |
| 16 | D1e | THE REACH DEFECT: a mate root two moves away (one prep move), its fix routes the first steps; a step is forcing when it checks or threatens mate | PROPOSED "goal: mate at distance two (the reach defect)" (GOAL; reads movement, attack, tempo) | 23 / 23 | 0 | 6 | `defect reach` |
| 17 | D12 | THE UNFINISHED LINE: no claim at all -- the best floor among the GAP routes decides if it loses no more than their threat | existing: unfinished line | 22 / 18 | 35 | 2 | `no proved claim` |
| 18 | D9 | THE REASONED KILL: the wall (flights out of our king's region) and the measure they hold (our only piece a rook, their passer) | existing: reasoned: the wall; reasoned: the measure they hold | 12 / 26 | 0 | 4 | `killed by reasoning\|reasoned: ` |
| 19 | D13 | A KILL THAT PROVES A LOSS, read off its text: '(-n)', 'they mate us', 'stalemate', 'no parry' | PROPOSED "closing: a kill that proves a loss (refuted)" (CLOSING; reads material, movement) | 19 / 16 | 0 | 2 | `is refuted\|are refuted\|nothing is refuted` |
| 20 | D4 | THE SMALLEST LOSS: a save root with no gaining parry decides by the least net, only in check; ties by the shape key, then the uci | existing: none; the shape key | 17 / 16 | 9 | 3 | `deciding by the smallest loss\|lemma none\|by the smallest loss` |
| 21 | D8f | THE STOPPER ROUTES: a quiet move that attacks a stopper (not the king) and is not lost by the exchange; the pawn offered to a stopper | PROPOSED "forcing: a threat on the stopper / the pawn offered to it" (FORCING; reads attack, material, movement) | 15 / 15 | 0 | 5 | `threat on the stopper\|offers the pawn to the stopper` |
| 22 | D24 | THE FORCING KINDS of a route's threat (mate / mate2 / check / threat / pass-claim) read to frame, parry-label and bind | existing: threat of mate; threat of mate in two; check; threat; threat: ... binds after a pass; the threat ... still binds | 11 / 11 | 0 | 5 | `closed by the threat\|still binds` |
| 23 | D14 | THEIR THREATS AS SAVE ROOTS: their pass claims (not the capture of a piece already hanging), their promote claims as 'stop their pawn', the targets newly attacked (not the king) | existing: threat: ... binds after a pass | 9 / 9 | 0 | 6 | `: their threat \|stop their P ` |
| 24 | D8g | DRIVE THEIR KING OFF: a check that fixes a promote defect by driving their king off its square (or a step of it) | existing: drive off | 8 / 8 | 0 | 2 | `drive.* their king off\|drives their king off` |
| 25 | D1d | SAVE DEFECT KINDS in the store made here (in check, promotes) | existing: in check; promotes | 6 / 6 | 0 | 2 | `defect (in check\|promotes)` |
| 26 | D7 | THE FORK'S VERIFICATION: every reply's balance + best exchange >= 1, else the reply that answers both (capture the forker, a check, one piece answers the other) | existing: one move answers both threats; fork: two threats, one move (Ideas.ForkSquare.one_move_touches_one); every reply read by the exchange rule (Ideas.ExchangeLaw) | 5 / 5 | 0 | 4 | `one move answers both threats\|fork: two threats` |
| 27 | D16 | THE LEAF READ: only the d1 balance families ('delayed_' ... '_d1') are followed to their leaves; refuted when their claim >= the gain there | existing: refuted at the leaf | 4 / 4 | 0 | 3 | `leaf read of\|DOWNGRADED to a gap\|refuted at the leaf` |
| 28 | D8e | THE PROMOTION EXCHANGE: one push, an x-ray behind: worth 8 less what they win on the new queen, kept when >= 1 | existing: promotion_exchange: the push, then the x-ray exchange on the promotion square (Ideas.ExchangeLaw seeXB, mirrored by the Rust see) | 2 / 3 | 0 | 3 | `promotion_exchange` |
| 29 | D5 | THE CLAIM ORDER AT EQUAL VALUE: a proved word before a table claim, then the question's stake, then the shape key, then the uci; routes by net then the fallback label | PROPOSED "heuristic: the claim order at equal value (word > table, stake, shape key)" (HEURISTIC; reads material, attack); beside existing: the shape key | 1 / 1 | 0 | 5 | `ties at net\|demoted:` |
| 30 | D20 | THE TABLE ROOT: <= 5 men, our move: the table's claim (a draw when behind); at equal value a proved word beats it | existing: syzygy | 1 / 1 | 0 | 2 | `label: table\|syzygy` |
| 31 | D10 | THE REFUTER FIX: exchange the refuting piece first (a capture whose recapture cannot make the refuting move); the goal asked again is the best unit attacked | existing: refuter fix | 0 / 0 | 0 | 7 | `refuter fix` |
| 32 | D8b | THE STOPPER ARITHMETIC: our king stops their pawn when its distance <= pushes + 1 (forced at = n + 1, no stopper in time above) | existing: no stopper in time | 0 / 0 | 0 | 7 | `no stopper in time\|forced: every move of ours is spent\|slack: distance` |
| 33 | D11 | THE QUIET-STATUS ESTIMATE: a cut route whose open replies are quiet leaves (not in check, no forcing move, their pass 0) closes as an estimate >= 1 | existing: chase: quiet-status estimate | 0 / 0 | 0 | 5 | `quiet-status\|CLAIM ESTIMATED` |
| 34 | D18 | THE DEEP PASS: read when their d1 pass claims nothing and a unit of ours (not the king) is attacked | existing: the deep pass | 0 / 0 | 0 | 3 | `\(deep: ` |
| 35 | D22 | THE ORDER OF THE ROOTS AND ROUTES BY CHESS CONTENT: mate roots by fewest defects then the cheaper piece; routes checks, captures, threats; replies recaptures, checks, captures, the rest; the ponder by captures and checks | PROPOSED "heuristic: the forcing order (checks, captures, threats; recaptures first)" (HEURISTIC; reads attack, material) | -- | -- | 6 | none (ranked by rows) |
| 36 | D26 | THE SCHEMA FAMILIES BY NAME: chase / delayed / mate_in lemma prefixes decide portability, inheritance and the window note | existing: chase: step_after+step_least; delayed_balance_alive_d1; mate_in_one; mate_in_two | -- | -- | 4 | none (ranked by rows) |

THE TOP TEN over all stages (by cases passing through, tier 0 + tier 2; ties by DECISION lines, then rows):

1. A1m -- goal kind MATE: cases 100 / 100, DECISION lines 0, rows 18
2. A2 -- the stake of MATE: cases 97 / 97, DECISION lines 0, rows 34
3. A1w -- goal kind WIN: cases 83 / 86, DECISION lines 0, rows 12
4. B4 -- THE FUSE'S SHAPE: cases 85 / 84, DECISION lines 0, rows 3
5. A4 -- MATERIAL as the stake currency: cases 82 / 85, DECISION lines 0, rows 4
6. D17 -- THE LOCAL GOAL TRACKED: cases 81 / 82, DECISION lines 0, rows 5
7. A1s -- goal kind SAVE and its sub-kinds: cases 78 / 83, DECISION lines 0, rows 12
8. A6 -- THE GAIN UNIT: cases 80 / 80, DECISION lines 0, rows 13
9. D3 -- CLOSE BY RULE: cases 70 / 81, DECISION lines 0, rows 10
10. D1a -- MATE DEFECT KINDS read by name (flight, checker takeable, interposition): cases 75 / 75, DECISION lines 0, rows 4

## 5. Findings

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

## 75.2 recount (appended by plan 75.2; the sections above are the audit of 2026-10-27 and stay as written)

THE RECOUNT (scripts/agenda_audit/recount.py, committed with stage (a)): the audit's own scripts, re-run on the working tree. The classifier's hand readings are keyed by the lines
of agenda.py / ponder.py at 65024fbf; recount.py maps them to the file being counted (difflib, through main 9e4e5673 with the plan 74 agent's three hand carries). A hand reading on a
line 75.2 EDITED is dropped (its fact left the line: the edited line is classified by the content rules alone) unless it is listed in recount.py's KEEP_ON_EDIT -- the readings whose
family's fact is still written on the edited line (the edit moved a goal word, a stake or the unit out of it, the family's own rule stayed); every dropped and every kept reading is
printed by the recount and was read. The baseline recount on 9e4e5673 reproduces the audit: 428 CHESS FACT rows, the plan 74 agent's per-family rows.

### Stage (a): the goal ladder and the goal kinds from the store

```
OBS audit rows total 1635: LOGIC 1211, RULE 105, NAME 23, CHESS FACT 255, HEURISTIC 41
OBS audit rows agenda.py 1601: LOGIC 1185, RULE 103, NAME 21, CHESS FACT 253, HEURISTIC 39
OBS audit rows ponder.py 34: LOGIC 26, RULE 2, NAME 2, CHESS FACT 2, HEURISTIC 2
OBS audit hand readings carried onto an edited line (the family's fact stays there): 29
OBS audit hand readings dropped (their line changed since 65024fbf): 80
OBS audit recount CHESS FACT rows 428 -> 255
OBS audit recount fact A10: rows 9 -> 4
OBS audit recount fact A1f: rows 3 -> 2
OBS audit recount fact A1m: rows 18 -> 1
OBS audit recount fact A1p: rows 21 -> 1
OBS audit recount fact A1s: rows 12 -> 0
OBS audit recount fact A1t: rows 3 -> 0
OBS audit recount fact A1w: rows 12 -> 1
OBS audit recount fact A1x: rows 1 -> 0
OBS audit recount fact A2: rows 34 -> 0
OBS audit recount fact A3: rows 14 -> 0
OBS audit recount fact A5: rows 20 -> 1
OBS audit recount fact A6: rows 13 -> 0
OBS audit recount fact A7: rows 15 -> 10
OBS audit recount fact A8: rows 2 -> 1
OBS audit recount fact A9: rows 3 -> 0
OBS audit recount fact D10: rows 7 -> 6
OBS audit recount fact D12: rows 2 -> 1
OBS audit recount fact D2: rows 17 -> 14
OBS audit recount fact D20: rows 2 -> 1
OBS audit recount fact D21: rows 10 -> 11
OBS audit recount fact D3: rows 10 -> 6
OBS audit recount fact D4: rows 3 -> 2
OBS audit recount fact D6: rows 10 -> 8
OBS audit recount fact D8d: rows 3 -> 2
OBS audit recount fact D8e: rows 3 -> 2
```

What left agenda.py: every goal word a question or a claim is built with (the GOAL entries' `question`: the terminal's, the promotion's, the gain's, the keep's; a technique's
from its entry's kind), every goal-kind test that decides (the entries' roles: ends the game, a keep, the promotion, a gain, chased, reads on a kill), the stake of mate (the terminal
GOAL entry, also read by game.py, binder.py and regress.py: one definition), the stake of promotion and the queen it places (the promotion entry over the material VALUATION), the unit
worths (the VALUATION entry: selfmine's table, Game.measure and BaseWords.worth read it), the gain unit (the VALUATION's `unit`), the promotion's near scope (the entry's `near`), and the
status ladder (worklist.py's outcome ladder: the rung, the line, kept, the raise). The D rows that fell held a goal word or the terminal's stake as their only atom (a claim's or a
question's kind literal; "a mate in one closes at 100"). D21 rose by one: `_keep_goal`'s in-check exclusion is now a D21 row of its own.

What stays, and why: A7 (10) -- the need's comparisons (`value >= min_stake`, `inn["value"] >= need`): machinery comparisons once the unit and the stakes are the store's; the
classifier keys on the need's variable names. A10 (4) -- the calls of `_keep_goal` and the root's kept-claim rule (its body reads the keep entry). A5 (1) -- the deferral test still
reads the run's need (`ctx.need`). A4 (4, unchanged) -- the orders by the valuation's worth (the classifier keys on the word "worth"; the worths are the entry's). A1m (1) -- a
king move is no checking placement (stage (c): the royal unit). A1p (1) -- a far passer can still push (`n > 0`). A1w (1) -- the gain goal's direct routes are captures, not
promotions. A1f (2) -- the fork question kind (no GOAL entry: the store's "fork" is a FORCING word). A8 (1) / A1d (1) -- the draw: a repetition when behind is a draw claim (the
rung now reads the ladder; the draw is no GOAL entry: the kernel has no repetition rule, a finding).

