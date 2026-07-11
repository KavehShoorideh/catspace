"""catspace.nn — torch-based full-board FB stack (optional dependency:
`pip install -e .[nn]`). Everything numpy-facing (feature planes, omega ids,
winprob transform) lives in nn/features.py and imports no torch, so the rest
of the package never needs torch installed.
"""
