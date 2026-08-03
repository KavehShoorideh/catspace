#!/usr/bin/env python
"""catspace/approaches/gauntlet_harness/experiments/register_incumbents.py -- port the CURRENT incumbent models + datasets into
the MLflow registry (Kaveh 2026-07-23: "port over our most recent model and data info into
whatever framework you choose"). One registry run per artifact, tagged kind=model|dataset,
with status/verdict/conventions as params. Idempotent: skips names already registered.
Browse: mlflow ui --backend-store-uri sqlite:///mlflow.db  (experiment: "registry").

Source of truth remains DECISIONS.md sec 5 + JOURNAL verdicts; this mirrors them into a
queryable store. Update by re-running after editing the tables below.
"""
from __future__ import annotations

import sys
from pathlib import Path
from catspace.io import paths


MODELS = [
    dict(name="iqe_geom_field", path=paths.sep("iqe_geom_field.pt"),
         role="cooperative field (legal-move reachability geometry)",
         arch="d=512 IQE(32comp) 128ch x 10 blocks GroupNorm", input_planes="20 minus zeroed 18/19 (legacy)",
         status="INCUMBENT (validated 2026-07-22)",
         verdict="ratio 33.0x, eff-rank F 15.1, decoupled from DTM (B ~ -0.1); OPEN: asym inverted 0.61x",
         data="geom_pool + geom_pool_edges + sf_cont_endgame_v1, --w-dtm 0"),
    dict(name="lichess_sharp", path=paths.sep("lichess_sharp.pt"),
         role="human-play field (as-played reachability + mate-pattern recognition)",
         arch="d=512 IQE(32comp) 128ch x 10 blocks GroupNorm spectral-norm omega-free", input_planes="all 20",
         status="TRAINING (30k steps; ladder every 5k)",
         verdict="early @5k: piece-count shortcut halving (0.72->0.32), white-king axis emerging (+0.59), in-stratum backrank 0.80->0.50",
         data="lichess prefix4gb + sf_cont_endgame_v1 (frac 0.3); dtm-hinge OFF (decoupling)"),
    dict(name="dtm_cnn", path=paths.sep("dtm_cnn.pt"),
         role="separate mate-distance head (the don't-overload-d companion)",
         arch="plain CNN 96ch x 4 blocks BatchNorm (deliberately not field-shaped; trunk-head measured worse)",
         input_planes="all 20", status="INCUMBENT head (interim)",
         verdict="spearman(pred,DTM) 3p .89 / 4p .61 / 6p .36 held-out",
         data="dtm_endgame.npz (24k tb-labeled)"),
    dict(name="nucleus_distilled", path=paths.sep("nucleus_distilled.pt"),
         role="DTM value model (field distilled to DTM; NOT a navigator)",
         arch="d=512 field lineage (pre-decoupling, collapsed geometry)", input_planes="20 minus zeroed 18/19",
         status="INTERIM value model -- re-distill from iqe_geom_field after contrast run",
         verdict="spearman 3p .88 / 4p .71 / 6p .53 (best 6p DTM predictor today)",
         data="distill targets from tablebase via MCTS-reachable region"),
]

MODELS.append(
    dict(name="lichess_mc", path=paths.sep("lichess_mc.pt"),
         role="MULTICHANNEL human field (regime-conditioned: 0 human / 1 random / 2 sf-optimal / 3 reserved)",
         arch="d=512 IQE(32comp) 128ch x 10 blocks GN spectral-norm + regime embedding (zero-anchored)",
         input_planes="all 20", status="TRAINING (resumed from lichess_sharp_step20000; 20k->50k)",
         verdict="pending; acceptance = per-channel health + d_r2-d_r1 vs tb deniedness + in-stratum cohesion",
         data="prefix4gb (r0 0.70) + regime_random_v1 (r1 0.15) + sf_cont_endgame_v1 (r2 0.15)"))

DATASETS = [
    dict(name="geom_pool", path=paths.derived("geom_pool.npz"),
         desc="357,223 toy-tree positions + DTM labels", use="L_pos/L_hard positions; DTM labels"),
    dict(name="geom_pool_edges", path=paths.derived("geom_pool_edges.npz"),
         desc="2,172,150 legal 1-ply edges, 9% irreversible", use="L_pos / L_hard"),
    dict(name="sf_cont_endgame_v1", path=paths.shards("sf_cont_endgame_v1/"),
         desc="4,924 Stockfish+tb-completed continuations / 148,714 pos -> 143,790 best-play edges",
         use="best-play L_pos edges; selfplay mix"),
    dict(name="dtm_endgame", path=paths.derived("dtm_endgame.npz"),
         desc="24,000 KRRvKBP-tree positions, DTM 1-193 (tb rollout-counted), 3-6 pieces",
         use="DTM head training, contrast anchors, subgoal bank, eval"),
    dict(name="contrast_mate_tuples", path=paths.derived("contrast_mate_tuples.npz"),
         desc="2,000 matched-anchor tuples / 28,000 states: SF-directed vs filtered-random branches + own mate exemplar (2026-07-23)",
         use="L_con matched-anchor contrast (structure-of-progress)"),
    dict(name="lichess_prefixes", path=paths.shards("lichess_db_standard_rated_2019-01.prefix{256mb,1gb,4gb}/"),
         desc="human games 2019-01; 2.54% <=6-piece; ~3M/12M/48M positions", use="human field; mate harvesting"),
    dict(name="krrkbp_test_n200", path=paths.experiment("krrkbp_test_n200.json"),
         desc="200 FROZEN KRRvKBP starts", use="the conversion eval set (never re-sample)"),
    dict(name="stratified_perfect", path=paths.derived("stratified_perfect.npz"),
         desc="White-mate / winning-simplification region", use="long/short goal region"),
    dict(name="syzygy", path=str(paths.syzygy_dir()),
         desc="Syzygy WDL/DTZ tables: KRRvKBP family + simplifications", use="exact play/defense; label generation"),
    dict(name="regime_random_v1", path=paths.shards("regime_random_v1/"),
         desc="8,000 random walks / 103,820 rows from 1gb-prefix anchors, regime=1 tags (2026-07-23)",
         use="multichannel channel 1 (random-play reachability)"),
    dict(name="regime_rollouts_v1", path=paths.shards("regime_rollouts_v1/"),
         desc="SHARED-ANCHOR rollouts from human lichess positions: regime 2 = SF-vs-SF, regime 3 = "
              "random-vs-SF, 4k anchors x 2 walks x ~13 plies; anchors.json provenance; DVC-tracked (2026-07-23)",
         use="multichannel channels 2/3 (balanced by construction; overlap ~1.0 at anchors)"),
    dict(name="move_selection_v1", path=paths.derived("move_selection_v1.npz"),
         desc="~300k (position, legal-move set, move played, mover Elo bin) rows recovered from shards",
         use="opponent-model training (option A)"),
]


def main():
    import mlflow
    from catspace.io.paths import mlflow_uri
    mlflow.set_tracking_uri(mlflow_uri())
    mlflow.set_experiment("registry")
    exp = mlflow.get_experiment_by_name("registry")
    existing = set()
    for _, row in mlflow.search_runs(experiment_ids=[exp.experiment_id]).iterrows():
        existing.add(row.get("tags.mlflow.runName"))
    n = 0
    for kind, entries in (("model", MODELS), ("dataset", DATASETS)):
        for e in entries:
            if e["name"] in existing:
                print(f"  skip (registered): {e['name']}")
                continue
            with mlflow.start_run(run_name=e["name"]):
                mlflow.set_tag("kind", kind)
                mlflow.log_params({k: str(v)[:450] for k, v in e.items()})
            n += 1
            print(f"  registered {kind}: {e['name']}")
    print(f"VERDICT REGISTRY registered={n} skipped={len(existing)} "
          f"(browse: mlflow ui --backend-store-uri sqlite:///mlflow.db)")


if __name__ == "__main__":
    main()
