"""THE HAND-PROVED IDEAS, registered with everything the loop needs: the Lean theorem, the detector command that
evaluates its hypotheses, its tier, its bridges, its uses in the player and the explainer's sentence. A research piece
that proves new ideas adds entries here (or a translator produces them)."""
from __future__ import annotations

REGISTERED = [
    dict(id="lean:backRankMate", theorem="Ideas.backRankMate_sound", tier="unconditional", detector="bb",
         claim=dict(kind="mate", horizon=1, value=0.0, text="the move is legal and mates"),
         hypotheses=["heavy", "onBackRank", "oneKing", "check", "legal", "rankFlights", "noDefender", "boxed", "noInterpose"],
         protagonist="our rook or queen", footprint="the king's square and flights, the landing square, the rank between them, the squares attacking the landing square (Lean detector `bb`)",
         bridges=["backrank_heldout", "backrank_random"],
         uses=[("closure", "TacticalSpec.proved_mates=1: a proved mate for the side to move closes the zone-search node")],
         sentence="{move} mates: the king is boxed in on its back rank, nothing can take the checker and nothing can step between (proved: Ideas.backRankMate_sound)."),
    dict(id="lean:unstoppablePasser", theorem="Ideas.Passer.unstoppablePasser_sound'", tier="unconditional", detector="passer",
         claim=dict(kind="promotion", horizon=13, value=8.0, text="our pawn promotes to a queen within 2n-1 plies whatever the opponent does"),
         hypotheses=["ownPawn", "pathClear", "bareKing", "noSliders", "notInCheck", "kingOutside", "kingHasMove"],
         protagonist="our pawn", footprint="the pawn, its path, the enemy king and its square (Lean detector `passer`)",
         bridges=["passer1_random"],
         uses=[("closure", "TacticalSpec.proved_passer + frame_player.PASSER: the queen counted at the horizon, the push a candidate (via the passer2 detector, which reports form-1 proofs; it is switched on only with lean:unstoppablePasser2's closure, whose bridges check what that detector reports)")],
         sentence="the pawn on {sq} cannot be stopped: the king is outside its square and has nothing else to stop it with (proved: Ideas.Passer.unstoppablePasser_sound')."),
    dict(id="lean:unstoppablePasser2", theorem="Ideas.Passer2.unstoppablePasser2_sound", tier="unconditional", detector="passer2",
         claim=dict(kind="promotion", horizon=13, value=8.0, text="our pawn promotes to a queen within 2n-1 plies whatever the opponent does"),
         hypotheses=["ownPawn", "pathClear", "enemyKingPawns", "noSliders", "notInCheck", "kingOutside", "pawnsFar", "kingRoom"],
         protagonist="our pawn", footprint="the pawn, its path, the enemy king's reachable squares, the enemy pawns' bad squares (Lean detector `passer2`)",
         bridges=["passer2_random", "passer2_heldout"],
         uses=[("closure", "TacticalSpec.proved_passer + frame_player.PASSER: the queen counted at the horizon, the push a candidate")],
         sentence="the pawn on {sq} cannot be stopped: the king is outside its square and no enemy pawn can get in its way in time (proved: Ideas.Passer2.unstoppablePasser2_sound)."),
]


def by_id(idea: str) -> dict | None:
    return next((r for r in REGISTERED if r["id"] == idea), None)
