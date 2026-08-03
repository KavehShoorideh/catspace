"""gauntlet_harness -- play the engine against external opponents and score it.

Game loops vs UCI opponents (maia, Stockfish, another catspace build) plus the VERDICT
instrumentation that keeps logs comparable across refactors. The drivers that run whole
tournaments live in ./experiments/; ./src/ is the reusable loop.
"""
from catspace.approaches.gauntlet_harness.src import run_games  # noqa: F401
