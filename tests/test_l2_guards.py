"""Guards on the L2 field's embedding<->objective pairing (train_lichess_fb.py).

IQE is a metric embedding -> it must train with a metric objective (QRL), never InfoNCE (scale-blind,
ignores the triangle inequality, collapses IQE to loss=ln(N) -- the 2026-07-21 footgun). These tests
pin the committed IQE+QRL default and the fail-fast rejection of the footguns. See JOURNAL
'IQE needs QRL not InfoNCE'.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from catspace.research.components.encoder.approaches.cone_fb_embedding.experiments.train_lichess_fb import L2_PRESETS, apply_l2_preset, validate_l2_config


def _args(**kw):
    d = dict(l2_preset="custom", iqe=False, quasimetric=False, qrl_objective=False, iqe_embed_scale=50.0)
    d.update(kw)
    return SimpleNamespace(**d)


def _resolve(args):
    apply_l2_preset(args)
    validate_l2_config(args)
    return args


@pytest.mark.parametrize("preset,exp", [
    ("iqe-qrl", dict(iqe=True, quasimetric=True, qrl_objective=True, iqe_embed_scale=1.0)),
    ("mrn-qm", dict(iqe=False, quasimetric=True, qrl_objective=False)),
    ("cosine", dict(iqe=False, quasimetric=False, qrl_objective=False)),
])
def test_presets_resolve_and_pass(preset, exp):
    a = _resolve(_args(l2_preset=preset))
    for k, v in exp.items():
        assert getattr(a, k) == v


def test_default_preset_is_iqe_qrl():
    """the committed default: bare L2 training is IQE+QRL at scale 1."""
    assert "iqe-qrl" in L2_PRESETS
    a = _resolve(_args(l2_preset="iqe-qrl"))
    assert a.iqe and a.qrl_objective and a.iqe_embed_scale == 1.0


@pytest.mark.parametrize("bad", [
    dict(iqe=True, quasimetric=True),                                    # IQE without QRL -> InfoNCE collapse
    dict(iqe=True, quasimetric=True, qrl_objective=True),                # IQE+QRL but scale=50 footgun
    dict(iqe=True, qrl_objective=True, iqe_embed_scale=1.0),             # IQE without quasimetric
    dict(qrl_objective=True),                                            # QRL without a quasimetric
])
def test_footguns_rejected(bad):
    with pytest.raises(SystemExit):
        _resolve(_args(**bad))


def test_good_custom_iqe_qrl_passes():
    a = _resolve(_args(iqe=True, quasimetric=True, qrl_objective=True, iqe_embed_scale=1.0))
    assert a.iqe and a.qrl_objective and a.iqe_embed_scale == 1.0
