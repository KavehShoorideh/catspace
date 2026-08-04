"""Assign every experiments/ script to its owning approach, or to a tools/ type.

Rules are ordered and first-match-wins, so specific patterns precede general ones. Import
evidence alone does not work: nn/features.py and nn/fb.py are shared infrastructure, so
counting imports attributes 141 of 290 scripts to encoder:jepa_tokenizer regardless of
what they are actually about. Filename intent is the stronger signal here, with imports
used only to break ties and to flag disagreements for review.

Throwaway tooling for the 2026-08-03 restructure; delete once the migration lands.

    python tools/_migration/classify_experiments.py            # print the plan
    python tools/_migration/classify_experiments.py --apply    # git mv everything
"""
from __future__ import annotations

import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "experiments"

C = "catspace/research/components"
T = "catspace/research/tools"
A = "catspace/approaches"

# (regex over the bare filename, destination directory)
# Destinations ending in /experiments belong to an approach; the rest are tool types.
RULES: list[tuple[str, str]] = [
    # ---------- viz / figures / viewers ----------
    (r"^viz_|_viz$|^article_figures|^build_.*viewer|^build_gallery|^build_.*explorer"
     r"|^build_embedding_atlas|^build_.*dashboard|^decompose_demo", f"{T}/viz"),
    (r"^fig_|^filmstrip", f"{T}/figures"),

    # ---------- generic measurement / reporting infrastructure ----------
    (r"^experiment_report|^experiment_leaderboard|^repro_check|^compare_methods"
     r"|^integration_tests|^eval_audit|^diagnostics$|^stage_checkers"
     r"|^data_distribution_check|^check_precision|^check_conditioning", f"{T}/stats_eval"),
    (r"^ab_convert|^ab_test|^move_ab|^playout_ab|^generalization|^rank_sensitivity"
     r"|^precision_reps|^capacity_forensics", f"{T}/stats_eval"),
    (r"^embed_checkpoints|^visualize_clusters|^goal_clusters|^cluster_(?!finetune)"
     r"|_umap|_tsne", f"{T}/embeddings"),
    (r"^ablate|_ablation", f"{T}/ablations"),
    (r"^improvement_loop|^self_retrain_loop|^expert_iteration|^toy_selfplay_loop"
     r"|^nucleus_pipeline|^launch|^overnight_orch|^gen_rollouts_daemon"
     r"|^losses$|^precompute_trunk_features", f"{T}/training_infra"),

    # ---------- encoder ----------
    (r"jepa|^build_jepa_corpus|^pretrain_jepa|_tok(en)?s?$|^train_dtm_tok"
     r"|^run_jepa|^run_hazard|^train_hazard_head", f"{C}/encoder/approaches/jepa_tokenizer/experiments"),
    (r"^concept|^native_concepts|^conditional_concepts|sae|_cav$|^denoise_cav"
     r"|^steer_concept|^subgoal_codebook|^opening_alphabet|^explain_mate_clusters"
     r"|^validate_concept_arrival", f"{C}/encoder/approaches/concept_quantization/experiments"),
    (r"^controlfield_|^measure_veto_channels|^measure_adversarial_veto"
     r"|^measure_soft_hard_consistency|^audit_channel_balance",
     f"{C}/encoder/approaches/control_field_wdl/experiments"),
    (r"^train_lichess_fb|^train_geometry|^train_iqe|^train_quasimetric|^qm_fitness"
     r"|^eval_geom|^viz_fb|^train_field_geometry|^reach_curvature|^field_manifold"
     r"|^train_rho_head|^gen_pairwise_data|^gen_toy_sets|^adversarial_distance_validation",
     f"{C}/encoder/approaches/cone_fb_embedding/experiments"),
    (r"^train_field|^distill_field|^train_stratified_field|^train_mate_field"
     r"|^train_occupancy_field|^train_lc0_field|^train_clock_field|^diagnose_field"
     r"|^diagnose_fieldenergy|^gen_field_data|^test_field_fullgame|^mate_from_field"
     r"|^train_reach_head|^train_child_rank_field|^gen_child_rank_data|^distill_finetune"
     r"|^distill_validate|^two_channel_distill|^certainty_distill|^certainty_rollouts"
     r"|^train_eval_heads|^train_policy_head|^train_l2_|^probe_readout|^layer_sweep_probe"
     r"|^compute_layer|^arch_bakeoff|^prove_batchnorm|^visualize_batchnorm"
     r"|^structure_probe|^probe_constraint_field|^probe_tri_carry",
     f"{C}/encoder/approaches/reachability_field/experiments"),

    # ---------- planner ----------
    (r"^mine_armed_tactics|armed", f"{C}/planner/approaches/armed_tactics/experiments"),
    (r"^m2b_|^m2c_|opponent|^train_opponent_model|^blunder_model|^build_player_dataset"
     r"|^build_opp_positions|^build_m2a_aug_feats|^build_zopp|^policy_surprise"
     r"|^fallibility_prior|^sf_vs_human_bands|^engine_vs_human_basins"
     r"|^build_home_book|^import_pgn_games", f"{C}/planner/approaches/opponent_model/experiments"),
    (r"^m3_|^m3b_|atlas|region|^basin_|^msm_basins|^metastable_macrostates"
     r"|^transition_|^train_transition_estimator|^gen_transition_data|^strata_diag"
     r"|^propagation_ladder|^rim_staircase|^nucleus", f"{C}/planner/approaches/atlas_region_stats/experiments"),
    (r"^committor|^phead_calibration|^value_fixed_point|^train_dtm_head"
     r"|^decision_flip_probe", f"{C}/planner/approaches/committor_value/experiments"),
    (r"dtm|^bootstrap_dtm|tablebase|^label_stockfish|^sf_label_transitions"
     r"|^forced_mate_set|^gen_forced_mate_data|^gen_stratified_perfect|^ladder_mate"
     r"|^mate_ladder_eval|^mate_bench|^mate_probe|^mate_gradient_probe|^mate_with_search"
     r"|^show_mate|^catalog_mate_directions|^mine_mate_puzzles|^gen_lichess_nearmate"
     r"|^gen_contrast_mate_tuples|^krk|^krrk|^conversion_|^endgame_handover"
     r"|^basin_tb_anchors|^gen_tb_policy_data|^bench_engines|^bench_value_speed"
     r"|^train_krk|^train_board_dtm|^train_board_policy|^conversion_board"
     r"|^gen_wdl_dtm_data|^gen_dtm_data|^sf_wdl_by_material|^gen_escape_data"
     r"|^train_escape_net|^gen_all_captures_labeled|^gen_pawn_capture_pairs"
     r"|^hanging_piece_probe|^defender_circuit_probe|^koopman_dyn_probe"
     r"|^longshort_engine|^basin_mate_engine|^catspace_engine",
     f"{C}/planner/approaches/endgame_groundtruth/experiments"),
    (r"^gen_agentive|^build_agentive_reach_data|^eval_agentive_lift|^build_reach"
     r"|^gen_regime_|^reach_efficiency|^concept_reach_rollout|^diag_region_nav"
     r"|^diag_transfer|^eval_dtm_alignment|^eval_field_dtm|^gen_successor_edges"
     r"|^gen_optimal_occupancy|^gen_traj_lc0|^build_aug_feats",
     f"{C}/planner/approaches/reach_field/experiments"),
    (r"^planner_|^subgoal|^plan_memory|^decompose|^gradient_planner|^train_planner_rl"
     r"|^m4_play_steering|^fair_navigator|^adaptive_vs_uniform|^execution_curves"
     r"|^engine_search_cost|^build_move_selection|^move_rank_check"
     r"|^build_position_memory|^acpl_probe|^tactic_events|^sharpness_"
     r"|^sf_reliability_map|^energy_baseline|^eval_variant|^eval_m1_|^eval_l2_"
     r"|^m1_", f"{C}/planner/approaches/subgoal_cascade/experiments"),

    # ---------- search ----------
    (r"mcts|^m5_mcts_probe|^search_tournament|^search_outcome|^krkn_search_sweep"
     r"|^stratified_mcts|^search_retrieval_combined",
     f"{C}/search/approaches/puct_mcts/experiments"),

    # ---------- memory ----------
    (r"^mine_checkpoints|^embed_checkpoint|trap", f"{C}/memory/approaches/checkpoint_trap_bank/experiments"),
    (r"^build_competence|^competence", f"{C}/memory/approaches/competence_map/experiments"),
    (r"^selfplay_generate|^gen_engine_games|^build_game_records|^balance_game_records"
     r"|^gen_stockfish_continuations|^gen_opening_pool|^build_opening_pool"
     r"|^build_lichess_shards|^gen_confirmatory_starts|^gen_regime_random"
     r"|^mine_only_move_bottlenecks|^opponent_recovery|^table_from_dump"
     r"|^gen_demo_viz_data|^gen_clock_child_data|^gen_rollouts",
     f"{C}/memory/approaches/experience_store/experiments"),
    (r"^m2b_cache|retrieval|vectordb", f"{C}/memory/approaches/vector_store_retrieval/experiments"),

    # ---------- wrapper-level end-to-end ----------
    (r"^register_incumbents|^arena_real|^gauntlet", f"{A}/gauntlet_harness/experiments"),

    # ---------- residue: assigned by reading the file, not by pattern ----------
    (r"^round13_eval$", f"{T}/stats_eval"),                      # full comparison runbook
    (r"^train_krrk$", f"{C}/planner/approaches/endgame_groundtruth/experiments"),
    (r"^run_table_v4$", f"{C}/planner/approaches/atlas_region_stats/experiments"),
    # cluster formation, the decoupled-field acceptance test, and the two overnight field
    # chains are all about the shape of F itself
    (r"^cluster_finetune$|^validate_decoupled$|^run_field_overnight$|^separation_loop$",
     f"{C}/encoder/approaches/reachability_field/experiments"),
]

COMPILED = [(re.compile(p, re.I), d) for p, d in RULES]
COMPONENT_IMPORT = re.compile(r"catspace\.research\.components\.(\w+)\.approaches\.(\w+)")


def destination(name: str) -> str | None:
    for rx, dest in COMPILED:
        if rx.search(name):
            return dest
    return None


def main() -> int:
    apply = "--apply" in sys.argv
    plan: dict[str, list[Path]] = defaultdict(list)
    unmatched: list[Path] = []

    for f in sorted(SRC.glob("*.py")) + sorted(SRC.glob("*.sh")):
        dest = destination(f.stem)
        (plan[dest] if dest else unmatched).append(f)

    for dest in sorted(plan):
        print(f"\n=== {dest}  ({len(plan[dest])})")
        for f in plan[dest]:
            print(f"    {f.name}")

    if unmatched:
        print(f"\n=== UNMATCHED ({len(unmatched)}) -- these need a rule or the incubator")
        for f in unmatched:
            txt = f.read_text(errors="replace")
            hits = Counter(f"{m.group(1)}:{m.group(2)}" for m in COMPONENT_IMPORT.finditer(txt))
            top = ", ".join(f"{k}({v})" for k, v in hits.most_common(2)) or "no component import"
            print(f"    {f.name:45s} {top}")

    print(f"\nmatched {sum(len(v) for v in plan.values())}, unmatched {len(unmatched)}")

    if apply:
        for dest, files in plan.items():
            (ROOT / dest).mkdir(parents=True, exist_ok=True)
            for f in files:
                subprocess.run(["git", "mv", str(f.relative_to(ROOT)), f"{dest}/{f.name}"],
                               cwd=ROOT, check=True)
        print("applied")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
