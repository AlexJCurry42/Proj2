"""Black-box tests for resolve.py on link structures built here."""
import numpy as np
import pytest

from bbphys import own_roots, same_partition

resolve = pytest.importorskip("duovision.resolve")


def leftward_links(H, W, p_link, max_back, seed):
    rng = np.random.default_rng(seed)
    pi = np.full((H, W), -1, dtype=np.int64)
    pj = np.full((H, W), -1, dtype=np.int64)
    jj, ii = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    link = (rng.random((H, W)) < p_link) & (ii >= 1)
    back = rng.integers(1, max_back + 1, (H, W))
    ti = np.clip(ii - back, 0, None)
    tj = np.clip(jj + rng.integers(-1, 2, (H, W)), 0, H - 1)
    pi[link] = ti[link]
    pj[link] = tj[link]
    return pi, pj


def random_order_links(H, W, p_link, seed):
    """Acyclic forest with no spatial direction: each node points to a node earlier in a random order."""
    rng = np.random.default_rng(seed)
    n = H * W
    order = rng.permutation(n)
    pos = np.empty(n, int)
    pos[order] = np.arange(n)
    target = np.full(n, -1)
    for k in range(1, n):
        if rng.random() < p_link:
            target[order[k]] = order[rng.integers(max(0, k - 50), k)]
    pi = np.where(target >= 0, target % W, -1).reshape(H, W)
    pj = np.where(target >= 0, target // W, -1).reshape(H, W)
    return pi, pj


def snake_chain(H, W):
    """One chain through every pixel (boustrophedon), longest possible depth."""
    order = []
    for j in range(H):
        cols = range(W) if j % 2 == 0 else range(W - 1, -1, -1)
        order += [j * W + i for i in cols]
    target = np.full(H * W, -1)
    for a, b in zip(order[1:], order[:-1]):
        target[a] = b
    return np.where(target >= 0, target % W, -1).reshape(H, W), np.where(target >= 0, target // W, -1).reshape(H, W)


STRUCTS = [
    ("left_dense", lambda: leftward_links(37, 53, 0.9, 25, 0)),
    ("left_sparse", lambda: leftward_links(64, 96, 0.4, 5, 1)),
    ("left_big", lambda: leftward_links(120, 200, 0.85, 60, 2)),
    ("random_order", lambda: random_order_links(40, 50, 0.8, 3)),
    ("snake", lambda: snake_chain(30, 40)),
    ("all_free", lambda: (np.full((10, 12), -1), np.full((10, 12), -1))),
]


@pytest.mark.parametrize("label,make", STRUCTS, ids=[s[0] for s in STRUCTS])
def test_roots_match_reference_and_own_walk(label, make):
    pi, pj = make()
    H, W = pi.shape
    roots = np.asarray(resolve.resolve_roots(pi, pj))
    assert roots.shape == (H, W)
    expect = own_roots(pi, pj)
    assert np.array_equal(roots, expect)
    ref = np.asarray(resolve.resolve_components_reference(pi, pj))
    assert ref.shape == (H, W)
    assert same_partition(roots, ref)
    assert same_partition(expect, ref)
    flat = roots.ravel()
    # roots are fixed points and are free pixels
    assert np.array_equal(flat[flat], flat)
    assert (pi.ravel()[flat] == -1).all()
    # every chain member shares its root with its partner
    lk = pi >= 0
    assert np.array_equal(roots[lk], roots[pj[lk], pi[lk]])


def _pm(partner_i, partner_j, linked, reserved=0):
    from duovision.encoder import PartnerMap

    H, W = partner_i.shape
    return PartnerMap(
        partner_i=partner_i, partner_j=partner_j, linked=linked,
        hit_z=np.zeros((H, W)), occluded=np.zeros((H, W), bool), reserved_rows=reserved,
    )


def test_round_partners_randomized_rounding():
    pytest.importorskip("duovision.encoder")
    H, W = 50, 80
    rng0 = np.random.default_rng(5)
    jj, ii = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    linked = (ii >= 20) & (rng0.random((H, W)) < 0.8)
    fi = np.where(linked, ii - 12.3, np.nan)
    fj = np.where(linked, jj.astype(float), np.nan)
    pm = _pm(fi, fj, linked)
    pi, pj = resolve.round_partners(pm, np.random.default_rng(1))
    pi, pj = np.asarray(pi), np.asarray(pj)
    assert pi.shape == (H, W) and np.issubdtype(pi.dtype, np.integer)
    assert (pi[~linked] == -1).all() and (pj[~linked] == -1).all()
    d = pi[linked] - (ii[linked] - 13)
    assert set(np.unique(d)) <= {0, 1}  # floor or ceil
    # CONTRACT-AMBIGUITY: "randomized rounding" — we assume unbiased (P(ceil) = frac).
    assert abs(d.mean() - 0.7) < 0.05
    assert (pj[linked] == jj[linked]).all()
    # deterministic for a given generator seed
    pi2, _ = resolve.round_partners(pm, np.random.default_rng(1))
    assert np.array_equal(pi, np.asarray(pi2))
