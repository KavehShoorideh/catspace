"""Plan 75.1 audit, step 2: classify every atom into LOGIC / RULE / NAME / CHESS FACT / HEURISTIC, map CHESS FACTS to a fact family (facts.py), merge
atoms into rows (one row per file, line, class and fact), and write rows.jsonl. Read-only on the engine."""
import json
import re
import sys

sys.path.insert(0, "/tmp/audit751")
from facts import FACTS  # noqa: E402

LOGIC, RULE, NAME, FACT, HEUR = "LOGIC", "RULE", "NAME", "CHESS FACT", "HEURISTIC"

TRACE_FUNCS = {"_local_report", "said", "status_line", "skeleton_obs", "_trace_fix", "_trace_joins", "_static_line", "_asked", "_line_of", "_escape_note",
               "_open_line", "_ask_finish", "facts_lines", "facts_delta", "_log", "_resumed_line", "_lemma_tally", "_closed_tag", "_note_resume"}

items = []
for f in ("items_agenda.jsonl", "items_ponder.jsonl"):
    for line in open("/tmp/audit751/" + f):
        items.append(json.loads(line))

# ---- extra items the AST passes do not see as atoms (literal 1 as the unit of material, formulas outside a test, dict labels): read off the source by hand
EXTRA = [
    ("agenda.py", 702, "LITERAL", "_reach_roots", "b.occupied_co[self.me] & ~b.kings & ~b.pawns (the reach roots' pieces: not the king, not a pawn)", FACT, "D1e"),
    ("agenda.py", 1013, "LITERAL", "status", "s0 = bal - threat (the status: balance less their pass threat)", FACT, "A5"),
    ("agenda.py", 1046, "LITERAL", "_ask_pool", "self.ctx.need = line - balance", FACT, "A5"),
    ("agenda.py", 1104, "LITERAL", "_ask_tiers", "self.ctx.need = line - balance", FACT, "A5"),
    ("agenda.py", 1151, "LITERAL", "_ask_tiers", "dict(st, line=1, raised=True) (the line rises to 'stay ahead')", FACT, "A5"),
    ("agenda.py", 1229, "LITERAL", "_next_agenda", "nxt.ctx.need = st['line'] - st['balance']", FACT, "A5"),
    ("agenda.py", 638, "LITERAL", "_technique_roots", "ents[:1] (one technique per class: the first by name)", HEUR, None),
    ("agenda.py", 1955, "LITERAL", "_route_threat", "stake = cq.stake if cq is not None else 1 (the default stake: one unit)", FACT, "A6"),
    ("agenda.py", 3284, "LITERAL", "_refuter_fix", "Agenda(..., min_stake=1) (the re-asked route must gain one unit)", FACT, "A6"),
    ("agenda.py", 3377, "LITERAL", "_target", "max(1, self.min_stake) ... else 1 (a route must reach one unit, or the need)", FACT, "A7"),
    ("agenda.py", 3400, "LITERAL", "_closed_by_rule", "need = 1 - dbal (what the exchange must still bring)", FACT, "A6"),
    ("agenda.py", 3442, "LITERAL", "_judge_stake", "cq.stake if cq is not None else 1 (the default stake: one unit)", FACT, "A6"),
    ("agenda.py", 3733, "LITERAL", "_parry_label", "stake = cq.stake if cq is not None else 1 (the default stake: one unit)", FACT, "A6"),
    ("agenda.py", 4110, "LITERAL", "_chase", "dist = max([1, len(q.defects)] + [f.pattern.get('steps', 1) for f in q.fixes]) (the prep moves a goal may spend)", FACT, "B3"),
    ("agenda.py", 4111, "LITERAL", "_chase", "parent - (0 if self.free_level else 1) (one level spent per our move; a recapture spends none)", FACT, "B3"),
    ("agenda.py", 4333, "LITERAL", "_quiet_leaf", "stake = cq.stake if cq is not None else 1 (the default stake: one unit)", FACT, "A6"),
    ("agenda.py", 3909, "LITERAL", "_addresses", "goal = {'mate': ..., 'win': ..., 'promote': ...}.get(q.goal, q.goal) (the goal named in the note)", NAME, None),
    ("agenda.py", 3940, "LITERAL", "_addresses", "{'flight': 'stays a flight', 'checker takeable': ..., 'reach': 'stays out of reach'} (the defect kinds named in the note)", NAME, None),
    ("ponder.py", 65, "LITERAL", "recompute_stale", "Agenda(..., min_stake=1) (a recomputation asks the roots worth one unit)", FACT, "A6"),
    ("ponder.py", 123, "LITERAL", "run", "order.append((-touched_stake, not board.is_capture(mv), not board.is_check() and not c.is_check(), uci, ...)) (the ponder's order fields)", FACT, "D22"),
    ("ponder.py", 137, "LITERAL", "run", "h._inner(c.fen(), 1) (the reply read for a gain of one unit)", FACT, "A6"),
]
for fl, ln, kind, func, text, cls, fact in EXTRA:
    items.append(dict(file=fl, line=ln, end=ln, kind=kind, func=func, text=text, text_only=False, _cls=cls, _fact=fact))

# ---- per-line overrides: (file, line, regex on the atom text or None for every atom of the line) -> (class, fact)
O = {}


def ov(line, rx, cls, fact=None, file="agenda.py"):
    O.setdefault((file, line), []).append((re.compile(rx) if rx else None, cls, fact))


# constants
ov(72, None, FACT, "B5")          # RULE2_FROM: the goal kinds and the walk of checks (its numbers: an extra HEURISTIC row below)
ov(105, None, FACT, "B6")         # FORCED_WORDS
# _pos_key, budget, castling
ov(136, None, RULE)
ov(393, r"\* 8|^8", FACT, "B4")
ov(570, None, RULE)
ov(571, None, RULE)
ov(572, None, RULE)
ov(565, r"KING", FACT, "A1m")     # a king move is no checking placement (castling's rook is)
ov(584, None, LOGIC)              # the text of the mate root
ov(587, None, FACT, "D22")        # mate roots: fewest defects, then the cheaper piece
# _roots / their side
ov(925, r"in hang", FACT, "D14")
ov(928, None, FACT, "D14")
ov(941, None, FACT, "A2")
ov(954, None, FACT, "D14")
ov(958, r"not in before", FACT, "D14")
# status / ladder
ov(1009, r"_piece_attacked", FACT, "D18")
ov(1011, None, FACT, "D18")
ov(1014, None, FACT, "A5")
ov(1022, None, FACT, "A5")
ov(1031, r"line", FACT, "A5")
ov(1047, None, FACT, "A5")
ov(1063, r"0\.05", HEUR)
ov(1141, r"0\.05", HEUR)
ov(1146, r"0\.05", HEUR)
ov(1154, r"0\.05", HEUR)
ov(1179, r"0\.05", HEUR)
ov(1285, r"0\.05", HEUR)
ov(1299, None, HEUR)              # resume order: price, then uci
ov(1332, None, FACT, "D13")
ov(1331, None, FACT, "D13")
ov(1356, r"_refuted", FACT, "D13")
ov(1379, None, FACT, "D22")
ov(1387, None, HEUR)
ov(1388, None, HEUR)
ov(1390, r"cost < best", HEUR)
ov(1396, r"threat", FACT, "D12")
ov(1403, r"Claim|goal", FACT, "D12")
ov(1430, r"got\[2\] > got\[1\]", FACT, "D18")
ov(1451, r"abcdefgh|12345678|in \(4, 5\)", NAME)
ov(1486, r"abcdefgh|12345678|in \(4, 5\)", NAME)
ov(1461, r"forcing_why", LOGIC)
ov(1467, None, HEUR)
ov(1575, r"STAKE_MATE", FACT, "A2")
ov(1619, r"need > 0|stake < self.ctx.need|goal != \"save\"", FACT, "A5")
ov(1688, r"line", FACT, "A5")
ov(1719, None, NAME)
ov(1775, None, NAME)
ov(1746, r"goal|chase_depth", FACT, "B6")
ov(1746, r"min_stake", FACT, "A7")
ov(1815, None, FACT, "D26")
ov(1821, r"startswith", FACT, "D26")
ov(1890, r"startswith", FACT, "D26")
ov(1833, r"MATE_VALUE", FACT, "A2")
ov(1932, r"th\.kind", FACT, "D24")
ov(1935, None, FACT, "D24")
ov(1937, None, FACT, "D24")
ov(1958, None, HEUR)
ov(1978, None, FACT, "D17")
ov(1980, None, FACT, "D17")
ov(1983, r"piece_type", FACT, "D17")
ov(1988, None, FACT, "D17")
ov(2003, r"th\.kind", FACT, "D24")
ov(2005, None, FACT, "D17")
ov(2010, r"gk\[0\]", FACT, "A1m")
ov(2023, r"g\[\"kind\"\] == \"mate\"", LOGIC)
# _ask dispatch
ov(2128, r"checkers", FACT, "D21")
ov(2142, r"promoter", FACT, "D14")
ov(2145, r"threat", FACT, "D14")
ov(2139, r"reach", FACT, "D1e")
ov(2151, None, FACT, "D21")
ov(2197, r"symbol", RULE)
ov(2208, None, LOGIC)
# fork
ov(2231, r"STAKE_MATE", FACT, "A9")
ov(2233, r"STAKE_MATE", FACT, "A9")
ov(2236, r"ex > 0", FACT, "D7")
ov(2236, r"_their_after_capture", FACT, "D23")
ov(2245, r"worst\[0\] < 1", FACT, "D7")
ov(2249, None, FACT, "D7")
ov(2253, None, FACT, "D7")
ov(2260, None, FACT, "A1f")
ov(2279, None, FACT, "D9")
ov(2282, None, FACT, "D9")
ov(2308, r"piece_type", FACT, "D9")
ov(2309, r"len\(heavy\)|ROOK", FACT, "D9")
ov(2317, r"KING", FACT, "C1")
ov(2366, r"to_square == t|promotion", FACT, "A1w")
# verify
ov(2397, r"min_stake", FACT, "A7")
ov(2397, r"schema", FACT, "D16")
ov(2405, r"theirs.value >= q.stake|net <= 0", FACT, "D15")
ov(2413, None, LOGIC)
ov(2414, None, LOGIC)
ov(2415, None, LOGIC)
ov(2417, None, FACT, "D5")
ov(2418, r"min_stake", FACT, "A7")
ov(2422, None, FACT, "D5")
ov(2423, None, FACT, "D5")
ov(2424, None, FACT, "D5")
ov(2426, None, FACT, "D15")
ov(2428, r"MATE_VALUE", FACT, "A2")
ov(2428, r"goal", FACT, "A1s")
ov(2434, None, FACT, "D26")
ov(2437, None, FACT, "D2")
# leaf
ov(2456, r"kept < 1", FACT, "A6")
ov(2465, None, FACT, "D2")
ov(2468, r"Claim", FACT, "D2")
ov(2471, None, FACT, "D2")
ov(2474, r"Claim", FACT, "D2")
ov(2498, None, FACT, "D16")
ov(2531, None, NAME)
ov(2551, None, FACT, "D16")
# smallest loss
ov(2580, r"goal", FACT, "D4")
ov(2602, None, FACT, "D4")
ov(2619, r"Claim", FACT, "D4")
ov(2640, None, LOGIC)
ov(2671, r"40", LOGIC)
ov(2680, None, FACT, "D1e")
# inner
ov(2717, r"goal|checkers", FACT, "B2")
ov(2748, r"goal|checkers", FACT, "B2")
ov(2749, None, FACT, "B2")
ov(2750, None, FACT, "B1")
ov(2726, r"need >= BD.MATE_VALUE", FACT, "A7")
ov(2732, r"root_pk|history", RULE)
ov(2758, r"> 0", FACT, "A6")
ov(2758, r"kept", FACT, "A10")
ov(2763, None, FACT, "A1p")
ov(2794, r"need|gstake", FACT, "A7")
ov(2816, None, FACT, "D8c")
# route body
ov(2896, r"goal", FACT, "A1w")
ov(2911, r"closed_min|stake_checked", FACT, "D3")
ov(2933, None, FACT, "A9")
ov(2968, None, FACT, "A2")
ov(3006, None, RULE)
ov(3009, r"MATE_VALUE", FACT, "D3")
ov(3023, r"promote_stake", FACT, "D8d")
ov(3034, r"dbal|MATE_VALUE", FACT, "A7")
ov(3034, r"STAKE_PROMOTE", FACT, "A3")
ov(3040, None, FACT, "B3")
ov(3054, r"promote_stake", FACT, "D8c")
ov(3056, None, FACT, "D8c")
ov(3058, None, FACT, "D8c")
ov(3070, None, FACT, "D8c")
ov(3075, None, FACT, "D8c")
ov(3080, r"target < 1", FACT, "A10")
ov(3080, r"MATE_VALUE", FACT, "D2")
ov(3081, r"dbal", FACT, "D2")
ov(3083, r"target < 1", FACT, "A10")
ov(3083, r"dbal >= target|claim\"\] is None", FACT, "D2")
ov(3093, r"dbal", FACT, "D2")
ov(3111, r"kind|>= 0", FACT, "D2")
ov(3128, r"dbal", FACT, "D2")
ov(3132, r"dbal", FACT, "D2")
ov(3144, r"dbal", FACT, "D2")
ov(3151, None, FACT, "D2")
ov(3161, r"dbal", FACT, "D2")
ov(3168, r"dbal", FACT, "D2")
ov(3169, None, FACT, "D2")
ov(3181, r"target", FACT, "A7")
ov(3189, r"target", FACT, "A7")
ov(3191, r"claim\"\] is None", FACT, "D25")
ov(3195, None, FACT, "A7")
# refuter
ov(3236, r"is_capture|to_square", FACT, "D10")
ov(3237, None, FACT, "D10")
ov(3250, r"< 1", FACT, "A6")
ov(3266, r"to_square", FACT, "D10")
ov(3274, None, FACT, "D10")
ov(3282, None, FACT, "D10")
ov(3287, r"1 - dbal", FACT, "D10")
# promote stands, keep, target, standing kill, close by rule
ov(3336, r"from_square", FACT, "D8d")
ov(3350, None, FACT, "D8d")
ov(3370, r"goal|checkers", FACT, "A10")
ov(3375, None, FACT, "A10")
ov(3386, r"dbal|target", FACT, "D25")
ov(3397, None, FACT, "D3")
ov(3398, r"dbal", FACT, "D3")
ov(3402, None, FACT, "D3")
ov(3404, r"min_stake", FACT, "A7")
ov(3407, r"stake", FACT, "D3")
ov(3416, None, FACT, "D3")
ov(3420, None, FACT, "D23")
# judge
ov(3451, r"keeps", FACT, "A5")
ov(3472, r"1 / n", HEUR)
ov(3474, None, FACT, "B7")
ov(3502, None, FACT, "C3")
ov(3558, r"goal|checkers", FACT, "D21")
ov(3567, r"k \+ 1", HEUR)
ov(3603, r"^2$", HEUR)
ov(3607, None, LOGIC)
ov(3617, None, HEUR)
ov(3620, None, HEUR)
ov(3623, None, HEUR)
ov(3645, r"pool / 2", HEUR)
ov(3664, r"t\[", FACT, "D22")
ov(3715, None, FACT, "A8")
ov(3735, None, FACT, "D24")
# exchange principal lines
ov(3768, None, FACT, "D23")
ov(3770, r"v ", FACT, "D23")
ov(3775, None, HEUR)
ov(3776, None, RULE)
ov(3779, None, FACT, "D23")
ov(3789, r"gains", FACT, "D23")
ov(3792, None, FACT, "D23")
ov(3814, r"\[2:4\]", NAME)
ov(3814, r"credit", FACT, "D23")
ov(3816, r"> 0", FACT, "D23")
ov(3832, r"piece_type", FACT, "D3")
ov(3838, None, FACT, "D3")
# walk of checks
ov(3940, r"join", LOGIC)
ov(3905, None, FACT, "B5")
ov(3907, None, FACT, "B5")
ov(3909, None, NAME)
ov(3910, r"move|reach", FACT, "B5")
ov(3918, None, FACT, "B5")
ov(3923, r"is_attacked_by", FACT, "B5")
ov(3925, r"in d.by", FACT, "B5")
ov(3927, r"replies|all\(", FACT, "B5")
ov(3929, r"attackers", FACT, "B5")
ov(3931, r"attacks\(", FACT, "B5")
ov(3931, r"q.goal", FACT, "B5")
ov(3933, r"skewer|q.goal", FACT, "B5")
ov(3935, None, FACT, "B5")
ov(3948, None, FACT, "D8g")
ov(3951, r"d.sq == ek|ek in d.by", FACT, "D8g")
ov(3974, r"startswith", LOGIC)
ov(3989, r"len\(path\)", RULE)
ov(3998, r"PAWN", FACT, "B1")
ov(4009, r"_irreversible|dbal|_pawn_move", FACT, "B1")
ov(4010, r"is_capture", RULE)
ov(4010, r"dbal|_pawn_move", FACT, "B1")
ov(4018, None, FACT, "B1")
ov(4030, None, HEUR)
ov(4032, None, FACT, "B8")
ov(4040, r"^v$", FACT, "B8")
# the chase
ov(4085, r"goal|checkers", FACT, "D21")
ov(4085, r"^3", LOGIC)
ov(4086, r"^evading$", FACT, "D21")
ov(4086, r"_keep_goal", FACT, "A10")
ov(4086, r"goal|chase_depth|CHASE_SAVES|DEEP_WIN", FACT, "B6")
ov(4086, r"min_stake", FACT, "A7")
ov(4086, r"_keep_goal", FACT, "A10")
ov(4088, r"since_capture|evading", FACT, "B1")
ov(4092, r"evad", FACT, "B2")
ov(4106, r"goal", FACT, "B3")
ov(4107, None, FACT, "B3")
ov(4109, r"fresh|stake", FACT, "B3")
ov(4111, r"free_level", FACT, "B3")
ov(4112, None, FACT, "B3")
ov(4122, None, FACT, "D21")
ov(4124, None, LOGIC)
ov(4126, None, HEUR)
ov(4127, r"tier", HEUR)
ov(4127, r"chase_depth", FACT, "B6")
ov(4129, None, LOGIC)
ov(4131, None, LOGIC)
ov(4141, r"goal", FACT, "B6")
ov(4148, r"goal", FACT, "D8h")
ov(4151, r"goal", FACT, "D8h")
ov(4154, r"goal|chase_depth", FACT, "D8f")
ov(4172, None, FACT, "B6")
ov(4173, r"goal", FACT, "B6")
ov(4173, r"tier", HEUR)
ov(4174, None, FACT, "B6")
ov(4175, None, FACT, "B6")
ov(4177, r"fix_moves", FACT, "B6")
ov(4179, None, FACT, "D22")
ov(4183, None, FACT, "A1p")
ov(4194, None, FACT, "A1p")
ov(4214, None, FACT, "A1p")
ov(4236, r"goal|stake", FACT, "B4")
ov(4237, r"^2", HEUR)
ov(4244, None, FACT, "A1p")
ov(4269, None, FACT, "A6")
ov(4275, r"goal", FACT, "D10")
ov(4283, r"any\(", FACT, "B4")
ov(4286, None, HEUR)
ov(4287, None, HEUR)
ov(4295, None, HEUR)
ov(4301, r"MATE_VALUE", FACT, "A2")
ov(4301, r"goal == \"promote\"", FACT, "A1p")
ov(4301, r"material", FACT, "D8c")
ov(4301, r"goal == \"save\"", FACT, "A1s")
ov(4302, None, FACT, "A10")
ov(4303, None, FACT, "A10")
ov(4307, r"kind|repetition|balance", FACT, "A8")
ov(4334, None, FACT, "D11")
ov(4337, None, FACT, "D11")
ov(4366, r"MATE_VALUE", FACT, "D3")
ov(4372, None, FACT, "D11")
ov(4374, None, FACT, "D11")
ov(4380, r"value < 1", FACT, "D11")
ov(4395, None, HEUR)
ov(4400, None, FACT, "A1s")
ov(4409, r"confidence|== self.best_est", HEUR)
ov(4418, r"keeps", FACT, "A5")
ov(4427, None, FACT, "A5")
# stopper routes, kills, saves
ov(4460, r"king", FACT, "C1")
ov(4463, r"castling", FACT, "D8f")
ov(4468, r"see_for", FACT, "D8f")
ov(4472, None, FACT, "D8f")
ov(4474, r"see_for", FACT, "D8f")
ov(4482, r"goal", FACT, "A1x")
ov(4491, None, FACT, "D1d")
ov(4493, None, FACT, "D21")
ov(4494, None, FACT, "D21")
ov(4495, None, FACT, "D21")
ov(4504, None, FACT, "D21")
ov(4517, None, FACT, "D1h")
ov(4519, None, FACT, "D1h")
ov(4535, None, FACT, "D1h")
ov(4540, r"MATE_VALUE|in tg|m.from_square", FACT, "D6")
ov(4545, None, FACT, "D6")
ov(4547, None, FACT, "D6")
ov(4566, None, FACT, "C3")
ov(4568, r"isdigit", FACT, "C3")
ov(4599, None, FACT, "D6")
ov(4608, None, FACT, "D6")
ov(4610, r"_mate_line_pieces", FACT, "D6")
ov(4613, None, FACT, "D6")
ov(4618, None, FACT, "D6")
ov(4620, None, FACT, "A6")
ov(4637, r"MATE_VALUE", FACT, "D6")
ov(4643, None, FACT, "D6")
ov(4650, r"_keep_goal", FACT, "A10")
ov(4662, None, FACT, "D8h")
ov(4663, None, FACT, "D8h")
ov(4673, r"defects", FACT, "D8h")
ov(4689, None, FACT, "D8h")
ov(4699, None, FACT, "D8h")
ov(4700, r"n == 1", RULE)
ov(4731, None, NAME)
ov(4733, None, NAME)
ov(4735, None, NAME)
ov(4754, r"defects|proved_passers", FACT, "D8h")
ov(4760, r"from_square", FACT, "D8h")
ov(4809, None, FACT, "D8e")
ov(4821, None, FACT, "D8e")
ov(4824, None, FACT, "D8e")
ov(4846, None, FACT, "D1d")
ov(4852, None, FACT, "D8b")
ov(4854, None, FACT, "D8b")
ov(4857, None, FACT, "D8b")
ov(4859, None, FACT, "D8b")
ov(4867, None, FACT, "D8b")
ov(4869, None, FACT, "D8b")
ov(4881, None, FACT, "D8b")
ov(460, None, HEUR)
ov(542, r"table", FACT, "D5")
ov(615, None, FACT, "D20")
ov(840, r"table", FACT, "D20")
ov(774, r"n > 0", FACT, "A1p")
ov(759, None, FACT, "D1e")
ov(886, None, NAME)
ov(1419, None, NAME)
ov(381, None, NAME)
# ponder
ov(41, None, FACT, "A1w", file="ponder.py")
ov(56, r"us == \"w\"", NAME, file="ponder.py")
ov(63, None, LOGIC, file="ponder.py")
ov(133, None, LOGIC, file="ponder.py")
ov(113, r"3600", HEUR, file="ponder.py")
ov(115, None, HEUR, file="ponder.py")
ov(124, None, FACT, "D22", file="ponder.py")
ov(111, None, NAME, file="ponder.py")

GOAL_LIT = {"mate": "A1m", "win": "A1w", "save": "A1s", "promote": "A1p", "fork": "A1f", "technique": "A1t", "draw": "A1d"}
KERNEL = re.compile(r"is_check\(\)|is_checkmate\(\)|is_stalemate\(\)|legal_moves|legal_count\(\)|is_capture\(|gives_check\(|is_castling\(|checkers\(\)|is_game_over\(\)|"
                    r"mate_in_one\(\)|\.promotion\b|is_attacked_by\(|attackers\(|\.attacks\(|attacks_mask\(|G\.mates\(|_mates\(|rule_of\[uci\]|\.color\b|piece_at\(|"
                    r"\.king\(|\bmate1\b|_aligned\(|\bturn\b")


def rule_of(it):
    t, k, fn = it["text"], it["kind"], it["func"]
    if "_cls" in it:
        return it["_cls"], it["_fact"]
    if k == "LITERAL":
        t = t.split("  in: ")[0]                 # a literal is classified by itself, its statement is context only
    for rx, cls, fact in O.get((it["file"], it["line"]), ()):
        if rx is None or rx.search(t):
            return cls, fact
    if t.startswith("[") or t in ('""', "[]", "()", "(None,)", "0"):
        return LOGIC, None                       # an `or` default for an empty list / a missing value
    if k == "CONST":
        name = t.split(" = ")[0]
        if name in ("DEPTH_FUSE", "FUSE", "TIER_COUNT", "JUDGE_PRICE_CAP", "JUDGE_MIN_AGENDAS", "CHASE_CLOCK"):
            return HEUR, None
        return LOGIC, None
    if k == "KIND":
        lit = re.search(r"'([^']+)'", t).group(1)
        if lit in GOAL_LIT:
            return FACT, GOAL_LIT[lit]
        if lit in ("kill", "claim"):
            return LOGIC, None
        return FACT, {"reach": "D1e", "in check": "D1d", "promotes": "D1d", "hangs": "D1h", "threatened": "D1h"}.get(lit)      # a fix kind (cover / remove): the line's override names the family
    # content patterns, most specific first
    if re.search(r"MATE_VALUE|STAKE_MATE", t):
        return FACT, "A2"
    if re.search(r"STAKE_PROMOTE|promote_stake|chess\.QUEEN", t):
        return FACT, "A3"
    if "PROMOTE_MAX_PUSHES" in t:
        return FACT, "A1p"
    if "BB_KING_ATTACKS" in t:
        return FACT, "C6"
    if re.search(r"piece_type.*chess\.KING", t):
        return FACT, "C1"
    if re.search(r"chess\.PAWN", t):
        return FACT, "C2"
    if re.search(r"\bd\.kind\b|d\.kind in", t):
        if re.search(r"flight|checker takeable|interposition", t):
            return FACT, "D1a"
        if re.search(r"defended|not attacked", t):
            return FACT, "D1b"
        if re.search(r"stopper|blocked|path attacked|promo square", t):
            return FACT, "D1c"
    m = re.search(r"(\.goal\b|gk\[0\]|g\[\"kind\"\]|\bkind\b|cl\.kind|c\.kind)\s*(==|!=|in|not in)\s*\(?\"(mate|win|save|promote|fork|technique|draw)\"", t)
    if m:
        if fn in TRACE_FUNCS or it["text_only"]:
            return NAME, None
        return FACT, GOAL_LIT[m.group(3)]
    if "\"checkers\" in" in t or "\"checkers\" not in" in t:
        return FACT, "D21"
    if "th.kind" in t:
        return FACT, "D24"
    if "FORCED_WORDS" in t:
        return FACT, "B6"
    if "RULE2_FROM" in t:
        return FACT, "B5"
    if re.search(r"since_capture|_clock\(\)", t):
        return FACT, "B1"
    if re.search(r"\bevaded\b|\bevading\b", t):
        return FACT, "B2"
    if re.search(r"\bs0\b|st\[\"line\"\]|st\[\"balance\"\]|\.keeps\(|status_info\[\"word\"\]|ctx\.need(?! is)", t):
        return FACT, "A5"
    if re.search(r"min_stake", t) and fn not in ("__init__",):
        return FACT, "A7"
    if re.search(r"W\.base\.worth", t):
        return FACT, "A4"
    if re.search(r"chess\.(WHITE|BLACK)", t):
        return NAME, None
    if "chess.SQUARES" in t:
        return NAME, None
    if re.search(r"chess\.[A-H][1-8]\b", t):
        return RULE, None
    if k == "LITERAL":
        if fn in TRACE_FUNCS or it["text_only"]:
            return LOGIC, None
        return LOGIC, None
    if k == "ORDER":
        if re.search(r"worth", t):
            return FACT, "A4"
        if re.search(r"uci\(\)|\[\"uci\"\]", t):
            return HEUR, None
        return LOGIC, None
    if KERNEL.search(t):
        return RULE, None
    return LOGIC, None


rows = {}
for it in items:
    cls, fact = rule_of(it)
    if cls == FACT and fact is None:
        raise SystemExit(f"no fact family: {it}")
    key = (it["file"], it["line"], cls, fact)
    r = rows.setdefault(key, dict(file=it["file"], line=it["line"], cls=cls, fact=fact, texts=[], kinds=set(), text_only=True, func=it["func"]))
    if it["text"] not in r["texts"]:
        r["texts"].append(it["text"])
    r["kinds"].add(it["kind"])
    r["text_only"] = r["text_only"] and bool(it["text_only"])
# RULE2_FROM's numbers: a HEURISTIC row of their own (the line mixes classes)
rows[("agenda.py", 72, HEUR, None)] = dict(file="agenda.py", line=72, cls=HEUR, fact=None, texts=["RULE2_FROM values 2 / 1 / 1 (the depths)"], kinds={"CONST"}, text_only=False, func="<module>")
out = sorted(rows.values(), key=lambda r: (r["file"] != "agenda.py", r["line"], [LOGIC, RULE, NAME, FACT, HEUR].index(r["cls"]), r["fact"] or ""))
with open("/tmp/audit751/rows.jsonl", "w") as fh:
    for r in out:
        r["kinds"] = sorted(r["kinds"])
        fh.write(json.dumps(r) + "\n")
print("atoms", len(items), "rows", len(out))
import collections
print(collections.Counter(r["cls"] for r in out))
print(collections.Counter(r["fact"] for r in out if r["cls"] == FACT).most_common())
missing = sorted({r["fact"] for r in out if r["cls"] == FACT} - set(FACTS))
print("facts not in the catalogue:", missing)
