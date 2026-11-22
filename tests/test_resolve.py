"""Tests for randomized rounding and chain resolution (pointer jumping vs scipy)."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from duovision import encoder
from duovision.config import DEFAULT_DEVICE, ViewProfile
from duovision.encoder import PartnerMap, compute_partner_map
from duovision.geometry import FoldGeometry
from duovision.resolve import resolve_components_reference, resolve_roots, round_partners

DEV = DEFAULT_DEVICE.scaled(0.25)
VIEW = ViewProfile()

_spec = importlib.util.spec_from_file_location("_enc_doubles", Path(__file__).with_name("test_encoder.py"))
_doubles = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_doubles)


def same_partition(a, b):
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    na = np.unique(a).size
    nb = np.unique(b).size
    pairs = np.unique(np.stack([a, b], axis=1), axis=0).shape[0]
    return na == nb == pairs


def random_forest(H, W, rng, mode="leftward", p_link=0.7):
    """Random acyclic link field (pi, pj)."""
    pi = np.full((H, W), -1, dtype=np.int32)
    pj = np.full((H, W), -1, dtype=np.int32)
    if mode == "leftward":
        jj, ii = np.nonzero(np.ones((H, W), bool))
        lk = (rng.random(H * W) < p_link) & (ii > 0)
        ti = (rng.random(H * W) * ii).astype(np.int32)  # strictly left
        tj = np.clip(jj + rng.integers(-2, 3, H * W), 0, H - 1)
        pi.ravel()[lk] = ti[lk]
        pj.ravel()[lk] = tj[lk]
    else:  # arbitrary forest: link to any node earlier in a random order
        n = H * W
        order = rng.permutation(n)
        rank = np.empty(n, dtype=np.int64)
        rank[order] = np.arange(n)
        lk = (rng.random(n) < p_link) & (rank > 0)
        tgt_rank = (rng.random(n) * rank).astype(np.int64)
        tgt = order[tgt_rank]
        pi.ravel()[lk] = (tgt % W)[lk]
        pj.ravel()[lk] = (tgt // W)[lk]
    return pi, pj


def check_roots(pi, pj):
    roots = resolve_roots(pi, pj)
    H, W = pi.shape
    idx = np.arange(H * W).reshape(H, W)
    free = pi < 0
    assert np.array_equal(roots[free], idx[free])  # free pixels are their own root
    r = roots.ravel()
    assert np.all(pi.ravel()[r] < 0)  # every root is free
    # a root is a fixed point of the parent map, and each pixel shares its parent's root
    lk = ~free
    assert np.array_equal(roots[lk], roots[pj[lk], pi[lk]])
    ref = resolve_components_reference(pi, pj)
    assert same_partition(roots, ref)
    return roots


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("mode", ["leftward", "arbitrary"])
def test_pointer_jumping_matches_scipy_random(seed, mode):
    rng = np.random.default_rng(seed)
    pi, pj = random_forest(37, 53, rng, mode=mode, p_link=0.5 + 0.1 * seed)
    check_roots(pi, pj)


def test_long_chain():
    W = 1000
    pi = (np.arange(W) - 1).astype(np.int32)[None, :]
    pj = np.zeros((1, W), dtype=np.int32)
    pi[0, 0] = -1
    pj[0, 0] = -1
    roots = check_roots(pi, pj)
    assert np.all(roots == 0)


def test_all_free():
    pi = np.full((5, 7), -1, np.int32)
    roots = check_roots(pi, pi.copy())
    assert np.array_equal(roots.ravel(), np.arange(35))


def test_cycle_detected():
    pi = np.array([[1, 0, -1]], dtype=np.int32)
    pj = np.array([[0, 0, -1]], dtype=np.int32)
    with pytest.raises(ValueError):
        resolve_roots(pi, pj)


def _synthetic_pm(H=40, W=60, rr=5, mode="cross", seed=0):
    rng = np.random.default_rng(seed)
    jj, ii = np.meshgrid(np.arange(H, dtype=float), np.arange(W, dtype=float), indexing="ij")
    sep = rng.uniform(0.5, 20, (H, W))
    if mode == "cross":
        pif = ii - sep
    else:
        pif = ii + sep
    pjf = jj + rng.uniform(-3, 3, (H, W))
    linked = (pif >= 0) & (pif <= W - 1) & (pjf >= rr) & (pjf <= H - 1)
    linked[:rr] = False
    pif = np.where(linked, pif, np.nan)
    pjf = np.where(linked, pjf, np.nan)
    return PartnerMap(pif, pjf, linked, np.full((H, W), 60.0), np.zeros((H, W), bool), rr, mode)


@pytest.mark.parametrize("mode", ["cross", "parallel"])
def test_round_partners_ranges_and_direction(mode):
    pm = _synthetic_pm(mode=mode)
    H, W = pm.linked.shape
    pi, pj = round_partners(pm, np.random.default_rng(1))
    assert pi.shape == (H, W) and np.issubdtype(pi.dtype, np.integer)
    lk = pi >= 0
    assert np.array_equal(lk, pj >= 0)
    assert not (lk & ~pm.linked).any()
    assert not lk[: pm.reserved_rows].any()
    cols = np.broadcast_to(np.arange(W), (H, W))
    if mode == "cross":
        assert np.all(pi[lk] < cols[lk])
    else:
        assert np.all(pi[lk] > cols[lk])
    assert np.all((pi[lk] >= 0) & (pi[lk] < W) & (pj[lk] >= pm.reserved_rows) & (pj[lk] < H))
    # rounding moves by less than one pixel
    assert np.all(np.abs(pi[lk] - pm.partner_i[lk]) < 1)
    assert np.all(np.abs(pj[lk] - pm.partner_j[lk]) < 1)
    # links with separation >= 1 px are never dropped
    assert lk[pm.linked & (np.abs(cols - pm.partner_i) >= 1)].all()
    check_roots(pi, pj)


def test_round_partners_frees_direction_violations():
    # partner 0.5 px to the left: rounding up would point at the pixel itself
    pm = _synthetic_pm()
    H, W = pm.linked.shape
    pm.partner_i[pm.linked] = np.broadcast_to(np.arange(W), (H, W))[pm.linked] - 0.5
    pi, _ = round_partners(pm, np.random.default_rng(0))
    frac = (pi[pm.linked] >= 0).mean()
    assert 0.4 < frac < 0.6
    cols = np.broadcast_to(np.arange(W), (H, W))
    assert np.all(pi[pi >= 0] < cols[pi >= 0])


def test_round_partners_unbiased_and_deterministic():
    H, W = 200, 200
    linked = np.zeros((H, W), bool)
    linked[10:] = True
    pif = np.where(linked, 50.3, np.nan)
    pjf = np.where(linked, 120.75, np.nan)
    pi_cols = np.broadcast_to(np.arange(W), (H, W))
    linked &= pi_cols > 51
    pif = np.where(linked, pif, np.nan)
    pjf = np.where(linked, pjf, np.nan)
    pm = PartnerMap(pif, pjf, linked, np.zeros((H, W)), np.zeros((H, W), bool), 10)
    pi, pj = round_partners(pm, np.random.default_rng(7))
    assert set(np.unique(pi[linked])) == {50, 51}
    assert pi[linked].mean() == pytest.approx(50.3, abs=0.01)
    assert pj[linked].mean() == pytest.approx(120.75, abs=0.01)
    pi2, pj2 = round_partners(pm, np.random.default_rng(7))
    assert np.array_equal(pi, pi2) and np.array_equal(pj, pj2)


@pytest.mark.parametrize("deg", [100, 140])
@pytest.mark.parametrize("seed", [0, 1])
def test_pointer_jumping_matches_scipy_real_partner_map(monkeypatch, deg, seed):
    monkeypatch.setattr(encoder, "_first_hit", _doubles.fake_first_hit)
    geom = FoldGeometry(DEV, deg)
    scene = _doubles.BoxScene(VIEW.z_background, VIEW.z_near, hx=15, hy=12)
    pm = compute_partner_map(geom, VIEW, scene)
    pi, pj = round_partners(pm, np.random.default_rng(seed))
    check_roots(pi, pj)


@pytest.mark.skipif(importlib.util.find_spec("duovision.scene") is None, reason="no scene module yet")
def test_pointer_jumping_matches_scipy_real_scene():
    from duovision.scene import build_scene

    geom = FoldGeometry(DEV, 120)
    pm = compute_partner_map(geom, VIEW, build_scene("torus", geom, VIEW))
    pi, pj = round_partners(pm, np.random.default_rng(0))
    check_roots(pi, pj)
