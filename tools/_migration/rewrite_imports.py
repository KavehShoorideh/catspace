"""Rewrite catspace.* import paths to their post-restructure locations.

Mechanical, whole-repo, longest-prefix-first so nested prefixes can't be clobbered.
Throwaway tooling for the 2026-08-03 restructure.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SKIP_DIRS = {".venv", ".git", "node_modules", "__pycache__", ".dvc", "mlruns",
             "_migration", "docs", "writing", "reports"}
# archived/historical docs keep their old paths on purpose (banner-noted instead)
SKIP_FILES = {"JOURNAL.md", "MILESTONES.md", "human-written-alt-arch.md"}

C = "catspace.research.components"
E = f"{C}.encoder.approaches"
M = f"{C}.memory.approaches"
S = f"{C}.search.approaches"
P = f"{C}.planner.approaches"
T = "catspace.research.tools"

# module-level moves: old dotted path -> new dotted path
MAP: dict[str, str] = {
    # ---- engine backbone -> wrapper top level
    "catspace.engine.interfaces": "catspace.interfaces",
    "catspace.engine.engine": "catspace.engine_core",
    "catspace.engine.orchestrator": "catspace.orchestrator",
    "catspace.engine.introspection": "catspace.introspection",
    "catspace.engine.watchlist": "catspace.watchlist",
    "catspace.engine.priors": "catspace.priors",
    "catspace.engine.fields": "catspace.fields",
    "catspace.engine.values": "catspace.values",
    "catspace.engine.search": f"{S}.puct_mcts.src.layer",
    # ---- search
    "catspace.search.mcts": f"{S}.puct_mcts.src.mcts",
    "catspace.search.nav": f"{S}.puct_mcts.src.nav",
    "catspace.nn.mcts": f"{S}.puct_mcts.src.mcts",
    "catspace.navigator.mcts": f"{S}.puct_mcts.src.mcts",
    "catspace.navigator.nav": f"{S}.puct_mcts.src.nav",
    "catspace.navigator": f"{S}.puct_mcts.src.nav",
    "catspace.reach.mcts": f"{S}.puct_mcts.src.mcts",
    "catspace.reach.nav": f"{S}.puct_mcts.src.nav",
    "catspace.nn.anytime": f"{S}.anytime_path.src.anytime",
    "catspace.search.memo": f"{T}.chess_specific.memo",
    # ---- encoder
    "catspace.encoder.field": f"{E}.reachability_field.src.field",
    "catspace.encoder.iqe_head": f"{E}.reachability_field.src.iqe_head",
    "catspace.field": f"{E}.reachability_field.src.field",
    "catspace.encoder.jepa": f"{E}.jepa_tokenizer.src.jepa",
    "catspace.nn.encoder": f"{E}.jepa_tokenizer.src.encoder",
    "catspace.nn.features": f"{E}.jepa_tokenizer.src.features",
    "catspace.nn.fb": f"{E}.jepa_tokenizer.src.fb",
    "catspace.nn.iqe": f"{E}.jepa_tokenizer.src.iqe",
    "catspace.nn.monotone_coords": f"{E}.jepa_tokenizer.src.monotone_coords",
    "catspace.nn.hard_negatives": f"{E}.jepa_tokenizer.src.hard_negatives",
    "catspace.nn.unreachable": f"{E}.jepa_tokenizer.src.unreachable",
    "catspace.nn.eval_head": f"{E}.jepa_tokenizer.src.eval_head",
    "catspace.nn.policy_head": f"{E}.jepa_tokenizer.src.policy_head",
    "catspace.nn.policy_fb": f"{E}.jepa_tokenizer.src.policy_fb",
    "catspace.nn.opponent": f"{E}.jepa_tokenizer.src.opponent",
    "catspace.controlfield.control": f"{E}.control_field_wdl.src.control",
    "catspace.controlfield.derivative": f"{E}.control_field_wdl.src.derivative",
    "catspace.controlfield.wdl_decay": f"{E}.control_field_wdl.src.wdl_decay",
    "catspace.cone.embedding": f"{E}.cone_fb_embedding.src.embedding",
    "catspace.cone.neural": f"{E}.cone_fb_embedding.src.neural",
    "catspace.cone.tabular": f"{E}.cone_fb_embedding.src.tabular",
    "catspace.concepts": f"{E}.concept_quantization.src.concepts",
    # ---- memory
    "catspace.memory.store": f"{M}.vector_store_retrieval.src.store",
    "catspace.memory.retrieval": f"{M}.vector_store_retrieval.src.retrieval",
    "catspace.memory.vectordb": f"{M}.vector_store_retrieval.src.vectordb",
    "catspace.vectordb": f"{M}.vector_store_retrieval.src.vectordb",
    "catspace.memory.goal_bank": f"{M}.goal_region_bank.src.goal_bank",
    "catspace.goal_bank": f"{M}.goal_region_bank.src.goal_bank",
    "catspace.memory.plan_store": f"{M}.plan_ledger.src.plan_store",
    "catspace.memory.checkpoint_bank": f"{M}.checkpoint_trap_bank.src.checkpoint_bank",
    "catspace.memory_field": f"{M}.fast_field_knn.src.memory_field",
    "catspace.experience": f"{M}.experience_store.src.experience",
    "catspace.competence": f"{M}.competence_map.src.competence",
    # ---- planner (incl. former predictor/* and style/*)
    "catspace.planner": f"{P}.subgoal_cascade.src",
    "catspace.predictor.atlas.ranker": f"{P}.atlas_region_stats.src.ranker",
    "catspace.predictor.atlas.region_stats": f"{P}.atlas_region_stats.src.region_stats",
    "catspace.predictor.atlas.transition": f"{P}.atlas_region_stats.src.transition",
    "catspace.predictor.atlas": f"{P}.atlas_region_stats.src",
    "catspace.atlas.ranker": f"{P}.atlas_region_stats.src.ranker",
    "catspace.atlas.region_stats": f"{P}.atlas_region_stats.src.region_stats",
    "catspace.atlas.transition": f"{P}.atlas_region_stats.src.transition",
    "catspace.atlas": f"{P}.atlas_region_stats.src",
    "catspace.subgoals": f"{P}.atlas_region_stats.src.ranker",
    "catspace.predictor.endgame.dtm": f"{P}.endgame_groundtruth.src.dtm",
    "catspace.predictor.endgame.material": f"{P}.endgame_groundtruth.src.material",
    "catspace.predictor.endgame.tb": f"{P}.endgame_groundtruth.src.tb",
    "catspace.predictor.endgame": f"{P}.endgame_groundtruth.src",
    "catspace.endgame.dtm": f"{P}.endgame_groundtruth.src.dtm",
    "catspace.endgame.material": f"{P}.endgame_groundtruth.src.material",
    "catspace.endgame.tb": f"{P}.endgame_groundtruth.src.tb",
    "catspace.endgame": f"{P}.endgame_groundtruth.src",
    "catspace.tb": f"{P}.endgame_groundtruth.src.tb",
    "catspace.predictor.reach.head": f"{P}.reach_field.src.head",
    "catspace.predictor.reach.region": f"{P}.reach_field.src.region",
    "catspace.predictor.reach": f"{P}.reach_field.src",
    "catspace.reach.head": f"{P}.reach_field.src.head",
    "catspace.reach.region": f"{P}.reach_field.src.region",
    "catspace.predictor.value.clock_field": f"{P}.committor_value.src.clock_field",
    "catspace.predictor.value.committor": f"{P}.committor_value.src.committor",
    "catspace.predictor.value": f"{P}.committor_value.src",
    "catspace.value.clock_field": f"{P}.committor_value.src.clock_field",
    "catspace.value.committor": f"{P}.committor_value.src.committor",
    "catspace.value": f"{P}.committor_value.src",
    "catspace.predictor.opponent.maia2_policy": f"{P}.opponent_model.src.maia2_policy",
    "catspace.predictor.opponent": f"{P}.opponent_model.src",
    "catspace.opponent.maia2_policy": f"{P}.opponent_model.src.maia2_policy",
    "catspace.opponent": f"{P}.opponent_model.src",
    "catspace.style.dataio": f"{P}.opponent_model.src.style_dataio",
    "catspace.style.estimator": f"{P}.opponent_model.src.style_estimator",
    "catspace.style.live": f"{P}.opponent_model.src.style_live",
    "catspace.style.model": f"{P}.opponent_model.src.style_model",
    "catspace.style.recover": f"{P}.opponent_model.src.style_recover",
    "catspace.two_field": f"{P}.two_perspective_scoring.src.two_field",
    "catspace.armed.detect": f"{P}.armed_tactics.src.detect",
    "catspace.armed": f"{P}.armed_tactics.src",
    # ---- tools
    "catspace.board": f"{T}.chess_specific.board",
    "catspace.chain": f"{T}.chess_specific.chain",
    "catspace.scoring": f"{T}.chess_specific.scoring",
    "catspace.reachgame": f"{T}.chess_specific.reachgame",
    "catspace.transition": f"{T}.chess_specific.transition",
    "catspace.game": f"{T}.chess_specific.game",
    "catspace.arena": f"{T}.chess_specific.arena",
    "catspace.realboard": f"{T}.chess_specific.realboard",
    "catspace.opponents": f"{T}.chess_specific.opponents",
    "catspace.diagnostics": f"{T}.chess_specific.diagnostics",
    "catspace.diagnostic_krrkbp": f"{T}.chess_specific.diagnostic_krrkbp",
    "catspace.uci": f"{T}.chess_specific.uci",
    "catspace.domains": f"{T}.chess_specific.domains",
    "catspace.data": f"{T}.chess_specific.chessdata",
    "catspace.stats": f"{T}.stats_eval.stats",
    "catspace.abtest": f"{T}.stats_eval.abtest",
    "catspace.tracking": f"{T}.stats_eval.tracking",
    "catspace.metrics": f"{T}.stats_eval.metrics",
    "catspace.audit": f"{T}.stats_eval.audit",
    "catspace.util": f"{T}.stats_eval.util",
    "catspace.train": f"{T}.training_infra.train",
    "catspace.viz": f"{T}.viz.viz",
    "catspace.harness.play": "catspace.approaches.gauntlet_harness.play",
    "catspace.harness": "catspace.approaches.gauntlet_harness",
}

# longest first so catspace.nn.encoder is handled before catspace.nn
ORDERED = sorted(MAP.items(), key=lambda kv: -len(kv[0]))


def rewrite(text: str) -> tuple[str, int]:
    n = 0
    for old, new in ORDERED:
        # word-boundary on the right so catspace.value doesn't eat catspace.values
        pat = re.compile(rf"(?<![\w.]){re.escape(old)}(?![\w])")
        text, k = pat.subn(new, text)
        n += k
    return text, n


def main(argv: list[str]) -> int:
    apply = "--apply" in argv
    exts = {".py", ".sh", ".toml", ".cfg", ".json", ".yml", ".yaml", ".md", ".bak"}
    total_files = total_subs = 0
    for p in sorted(ROOT.rglob("*")):
        if not p.is_file() or p.suffix not in exts:
            continue
        rel = p.relative_to(ROOT)
        if any(part in SKIP_DIRS for part in rel.parts) or p.name in SKIP_FILES:
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        new, n = rewrite(text)
        if n:
            total_files += 1
            total_subs += n
            if apply:
                p.write_text(new, encoding="utf-8")
            else:
                print(f"{n:4d}  {rel}")
    verb = "rewrote" if apply else "would rewrite"
    print(f"\n{verb} {total_subs} references across {total_files} files")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
