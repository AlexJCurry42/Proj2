"""Turn a fractional partner map into integer links and resolve link chains.

Links form a forest: every pixel has at most one outgoing link and, in cross mode,
it always points strictly left (parallel: strictly right), so there are no cycles.
Each tree's root is a free pixel; colouring every pixel by its root's colour
satisfies all links exactly.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

_MAX_JUMPS = 64


def round_partners(pm, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Randomized rounding of fractional partners: ``floor(x + U[0, 1))``.

    Returns int32 arrays ``(pi, pj)`` of shape (H, W); -1 marks free pixels.
    Results are clipped to the display; a link is dropped (pixel freed) if its
    rounded partner lands in the reserved rows or would break the direction
    invariant (cross: ``pi < i``; parallel: ``pi > i``).
    """
    H, W = pm.linked.shape
    rr = int(pm.reserved_rows)
    # Draw for every pixel so the stream does not depend on which pixels are linked.
    ui = rng.random((H, W))
    uj = rng.random((H, W))
    with np.errstate(invalid="ignore"):
        fi = np.floor(pm.partner_i + ui)
        fj = np.floor(pm.partner_j + uj)
    ok = pm.linked & np.isfinite(fi) & np.isfinite(fj)
    fi = np.where(ok, fi, -1.0)
    fj = np.where(ok, fj, -1.0)
    pi = np.clip(fi, 0, W - 1).astype(np.int32)
    pj = np.clip(fj, 0, H - 1).astype(np.int32)

    cols = np.arange(W, dtype=np.int32)[None, :]
    if getattr(pm, "mode", "cross") == "parallel":
        ok &= pi > cols
    else:
        ok &= pi < cols
    ok &= pj >= rr
    ok[:rr] = False
    pi[~ok] = -1
    pj[~ok] = -1
    return pi, pj


def _parents(pi: np.ndarray, pj: np.ndarray) -> np.ndarray:
    pi = np.asarray(pi)
    pj = np.asarray(pj)
    if pi.shape != pj.shape or pi.ndim != 2:
        raise ValueError("pi and pj must be (H, W) arrays of the same shape")
    H, W = pi.shape
    parent = np.arange(H * W, dtype=np.int64)
    lk = (pi.ravel() >= 0) & (pj.ravel() >= 0)
    parent[lk] = pj.ravel()[lk].astype(np.int64) * W + pi.ravel()[lk]
    return parent


def resolve_roots(pi: np.ndarray, pj: np.ndarray) -> np.ndarray:
    """Root (flat index ``j * W + i``) of every pixel's chain, by pointer jumping.

    ``parent = parent[parent]`` until stable; free pixels are their own root.
    Raises ValueError if the links contain a cycle.
    """
    H, W = np.shape(pi)
    parent0 = _parents(pi, pj)
    parent = parent0
    for _ in range(_MAX_JUMPS):
        nxt = parent[parent]
        if np.array_equal(nxt, parent):
            break
        parent = nxt
    else:
        raise ValueError("pointer jumping did not converge (cyclic links?)")
    # Every root must be a genuinely free pixel; a cycle would make a linked pixel its own root.
    if not np.all(parent0[parent] == parent):
        raise ValueError("partner links contain a cycle")
    return parent.reshape(H, W)


def resolve_components_reference(pi: np.ndarray, pj: np.ndarray) -> np.ndarray:
    """Reference partition via scipy connected components (undirected link graph)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    H, W = np.shape(pi)
    n = H * W
    parent = _parents(pi, pj)
    src = np.nonzero(parent != np.arange(n))[0]
    g = coo_matrix((np.ones(src.size, dtype=np.int8), (src, parent[src])), shape=(n, n)).tocsr()
    _, labels = connected_components(g, directed=False)
    return labels.reshape(H, W)
