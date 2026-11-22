"""Independent physics helpers for the black-box tests.

Nothing here imports the modules under test at import time; the fold maths is
re-derived from the physical model (two flat halves meeting at a vertical
hinge, outer edges toward the viewer) rather than calling FoldGeometry, so the
tests do not inherit a bug from shared code paths.
"""
from __future__ import annotations

import functools
import math

import numpy as np

from duovision.config import DEFAULT_DEVICE, ViewProfile, guide_band_rows
from duovision.geometry import FoldGeometry

SCALE = 0.25
DEV = DEFAULT_DEVICE.scaled(SCALE)


# --------------------------------------------------------------------- fold
class Fold:
    """Own re-derivation of the folded display at hinge angle phi."""

    def __init__(self, device, hinge_deg):
        half = math.radians(hinge_deg) / 2.0
        self.s, self.c = math.sin(half), math.cos(half)
        self.W, self.H = device.width_px, device.height_px
        self.mmpp = 25.4 / device.ppi
        self.hw = self.W / 2.0 * self.mmpp  # hinge -> outer edge, along the surface
        self.hh = self.H / 2.0 * self.mmpp
        self.x_extent = self.hw * self.s

    def pix_to_uv(self, i, j):
        u = (np.asarray(i, float) + 0.5 - self.W / 2.0) * self.mmpp
        v = (self.H / 2.0 - (np.asarray(j, float) + 0.5)) * self.mmpp
        return u, v

    def uv_to_pix(self, u, v):
        i = np.asarray(u, float) / self.mmpp + self.W / 2.0 - 0.5
        j = self.H / 2.0 - np.asarray(v, float) / self.mmpp - 0.5
        return i, j

    def uv_to_world(self, u, v):
        u, v = np.broadcast_arrays(np.asarray(u, float), np.asarray(v, float))
        return np.stack([u * self.s, v, np.abs(u) * self.c], axis=-1)

    def pix_to_world(self, i, j):
        return self.uv_to_world(*self.pix_to_uv(i, j))

    def panel_z(self, x):
        # z_panel = |x| * cot(phi/2)
        return np.abs(np.asarray(x, float)) * self.c / self.s

    def hit_display(self, origin, dirs):
        """Nearest t>0 intersection of origin + t*dirs with the two half-planes."""
        o = np.asarray(origin, float)
        d = np.asarray(dirs, float)
        o, d = np.broadcast_arrays(o, d)
        best_t = np.full(o.shape[:-1], np.inf)
        best_u = np.full(o.shape[:-1], np.nan)
        best_v = np.full(o.shape[:-1], np.nan)
        s, c = self.s, self.c
        for sign in (-1.0, 1.0):
            a = np.array([sign * s, 0.0, c])      # hinge -> outer edge
            n = np.array([-sign * c, 0.0, s])     # normal toward viewer
            dn = d @ n
            with np.errstate(divide="ignore", invalid="ignore"):
                t = -(o @ n) / dn
            p = o + t[..., None] * d
            along = p @ a
            ok = (np.isfinite(t) & (t > 1e-9) & (along >= -1e-9) & (along <= self.hw + 1e-9)
                  & (np.abs(p[..., 1]) <= self.hh + 1e-9))
            better = ok & (t < best_t)
            best_t = np.where(better, t, best_t)
            best_u = np.where(better, sign * np.clip(along, 0, None), best_u)
            best_v = np.where(better, p[..., 1], best_v)
        valid = np.isfinite(best_t)
        return best_u, best_v, np.where(valid, best_t, np.nan), valid


# ------------------------------------------------------------------ marching
def march(scene, origin, targets, *, z_hi, z_lo, step=0.05, iters=60):
    """First point along origin->targets (and beyond) where ray.z <= z_of.

    Rays are sampled between the planes z=z_hi and z=z_lo (z_lo may be per ray)
    with a fixed 3D step, then bisected. Returns (points, hit, t) with
    points = origin + t*(targets-origin).
    """
    o = np.asarray(origin, float)
    T = np.atleast_2d(np.asarray(targets, float))
    D = T - o
    n = len(T)
    z_lo = np.broadcast_to(np.asarray(z_lo, float), (n,))
    t0 = (o[2] - z_hi) / (o[2] - T[:, 2])
    t1 = (o[2] - z_lo) / (o[2] - T[:, 2])
    t0 = np.maximum(t0, 0.0)
    length = (t1 - t0) * np.linalg.norm(D, axis=1)
    m = int(np.ceil(length.max() / step)) + 1
    frac = np.linspace(0.0, 1.0, m + 1)
    ts = t0[:, None] + (t1 - t0)[:, None] * frac[None, :]
    P = o + ts[..., None] * D[:, None, :]
    f = P[..., 2] - np.asarray(scene.z_of(P[..., 0], P[..., 1]), float)
    below = f <= 0
    hit = below.any(axis=1) & ~below[:, 0]
    k = np.argmax(below, axis=1)
    rows = np.arange(n)
    lo = ts[rows, np.maximum(k - 1, 0)]
    hi = ts[rows, k]
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        Pm = o + mid[:, None] * D
        fm = Pm[:, 2] - np.asarray(scene.z_of(Pm[:, 0], Pm[:, 1]), float)
        inside = fm <= 0
        hi = np.where(inside, mid, hi)
        lo = np.where(inside, lo, mid)
    pts = o + hi[:, None] * D
    pts[~hit] = np.nan
    return pts, hit, np.where(hit, hi, np.nan)


def physics_partners(scene, fold: Fold, view: ViewProfile, i, j, band):
    """Independent partner computation for pixels (i, j), cross mode.

    Returns dict with arrays: S (N,3), hitL, visible, pi, pj, linked, and
    'ambiguous' (status depends on sub-mm details: silhouettes, display/band edges).
    """
    L, R = view.eyes()
    X = fold.pix_to_world(i, j)
    z_hi = float(scene.z_max) + 1.0
    S, hitL, _ = march(scene, L, X, z_hi=z_hi, z_lo=X[:, 2] - 1.0)
    n = len(X)
    visible = np.zeros(n, bool)
    margin = np.full(n, np.inf)
    pi = np.full(n, np.nan)
    pj = np.full(n, np.nan)
    ok = np.where(hitL)[0]
    if len(ok):
        Sv = S[ok]
        # march from R toward S, stop a little past S
        pts, hitR, tR = march(scene, R, Sv, z_hi=z_hi, z_lo=Sv[:, 2] - 0.3)
        segl = np.linalg.norm(Sv - R, axis=1)
        dist_before = np.where(hitR, (1.0 - tR) * segl, np.inf)
        vis = hitR & (np.abs(dist_before) < 0.05)
        visible[ok] = vis
        margin[ok] = np.abs(dist_before)
        u, v, _, valid = fold.hit_display(R, Sv - R)
        ii, jj = fold.uv_to_pix(u, v)
        pi[ok] = np.where(valid, ii, np.nan)
        pj[ok] = np.where(valid, jj, np.nan)
    on_disp = np.isfinite(pi)
    in_band = on_disp & (pj < band - 0.5)
    linked = hitL & visible & on_disp & ~in_band
    amb = (
        (np.isfinite(margin) & (margin > 0.02) & (margin < 1.0))
        | (on_disp & ((pi < 0.5) | (pi > fold.W - 1.5)))
        | (on_disp & (np.abs(pj - (band - 0.5)) < 1.0))
        | (on_disp & (pj > fold.H - 1.5))
    )
    return dict(S=S, hitL=hitL, visible=visible, pi=pi, pj=pj, linked=linked, ambiguous=amb)


# ------------------------------------------------------------------- roots
def own_roots(pi, pj):
    """Follow pointers one step at a time (slow but obviously correct)."""
    H, W = pi.shape
    idx = np.arange(H * W)
    p = pi.ravel()
    nxt = np.where(p >= 0, pj.ravel() * W + p, idx)
    r = idx.copy()
    for _ in range(H * W + 1):
        r2 = nxt[r]
        if np.array_equal(r2, r):
            break
        r = r2
    return r.reshape(H, W)


def same_partition(a, b):
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    pairs = np.unique(np.stack([a, b], axis=1), axis=0)
    return len(pairs) == len(np.unique(a)) == len(np.unique(b))


# ------------------------------------------------------------------- scenes
def z_bg(view, fold, x, bg_vee):
    return view.z_background + bg_vee * np.abs(np.asarray(x, float)) * fold.c / fold.s


def footprint_grid(fold, nx=241, ny=161):
    x = np.linspace(-fold.x_extent, fold.x_extent, nx)
    y = np.linspace(-fold.hh, fold.hh, ny)
    X, Y = np.meshgrid(x, y)
    return X, Y


def background_rows(scene, fold, view, bg_vee, band, x_half=45.0):
    """Rows (sorted by distance from the centre) whose eye rays only see background.

    For pixel row with surface height v, every point of an eye->pixel ray with
    0 <= z <= 100 has y between 0.66 v and v. Require object_mask False and
    z_of == z_bg on that whole window for |x| <= x_half.
    """
    rows = []
    xs = np.linspace(-x_half, x_half, 181)
    for j in range(band + 2, fold.H - 2):
        _, v = fold.pix_to_uv(0, j)
        ys = np.linspace(0.66 * v, 1.0 * v, 15)
        X, Y = np.meshgrid(xs, ys)
        m = np.asarray(scene.object_mask(X, Y))
        z = np.asarray(scene.z_of(X, Y), float)
        if not m.any() and np.all(np.abs(z - z_bg(view, fold, X, bg_vee)) < 1e-6):
            rows.append(j)
    rows.sort(key=lambda r: abs(r + 0.5 - fold.H / 2.0))
    return rows


# --------------------------------------------------------------- dict utils
def flatten(d, prefix=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            name = f"{prefix}{k}"
            if isinstance(v, dict):
                out.update(flatten(v, name + "."))
            else:
                out[name] = v
    return out


def pick(d, *alternatives, numeric=True):
    """Value of the first key containing all substrings of one alternative.

    alternatives: tuples of lowercase substrings. Raises AssertionError listing keys.
    """
    flat = flatten(d)
    for alt in alternatives:
        for k, v in flat.items():
            kl = k.lower()
            if all(s in kl for s in alt):
                if numeric:
                    try:
                        return k, float(np.asarray(v))
                    except (TypeError, ValueError):
                        continue
                return k, v
    raise AssertionError(f"no key matching {alternatives}; keys: {sorted(flat)}")


# ------------------------------------------------------------------ caching
@functools.lru_cache(maxsize=None)
def geom_view(hinge, scale=SCALE, mode="cross"):
    dev = DEFAULT_DEVICE.scaled(scale)
    return FoldGeometry(dev, hinge), ViewProfile(mode=mode), Fold(dev, hinge)


@functools.lru_cache(maxsize=None)
def scene_for(name, hinge, bg_vee=0.25, scale=SCALE, mode="cross"):
    from duovision.scene import build_scene

    geom, view, _ = geom_view(hinge, scale, mode)
    return build_scene(name, geom, view, bg_vee=bg_vee)


@functools.lru_cache(maxsize=None)
def encoded(name, hinge, style="color", seed=0, bg_vee=0.25, scale=SCALE, mode="cross"):
    from duovision.encoder import encode_stereogram

    geom, view, fold = geom_view(hinge, scale, mode)
    scene = scene_for(name, hinge, bg_vee, scale, mode)
    res = encode_stereogram(geom, view, scene, style, seed=seed)
    return geom, view, fold, scene, res


def band_rows(device=DEV):
    return guide_band_rows(device)


def own_reverse_match_rate(image, scene, fold, view, band, *, n=4000, frac_tol=0.25, seed=0):
    """Own re-implementation of the reverse link check (R -> S -> L), for comparison."""
    L, R = view.eyes()
    rng = np.random.default_rng(seed)
    qj = rng.integers(band, fold.H, n)
    qi = rng.integers(0, fold.W, n)
    X = fold.pix_to_world(qi, qj)
    S, hit, _ = march(scene, R, X, z_hi=float(scene.z_max) + 1, z_lo=X[:, 2] - 1)
    vis = np.zeros(n, bool)
    h = np.where(hit)[0]
    _, hL, tL = march(scene, L, S[h], z_hi=float(scene.z_max) + 1, z_lo=S[h][:, 2] - 0.3)
    vis[h] = hL & ((1 - tL) * np.linalg.norm(S[h] - L, axis=1) < 0.05)
    u, v, _, ok = fold.hit_display(L, np.where(hit[:, None], S - L, 1.0))
    pi, pj = fold.uv_to_pix(u, v)
    pi = np.nan_to_num(pi, nan=-9)
    pj = np.nan_to_num(pj, nan=-9)
    valid = vis & ok & (pj > band - 0.5) & (pi > -0.5) & (pi < fold.W - 0.5) & (pj < fold.H - 0.5)
    conf = valid & (np.abs(pi - np.rint(pi)) < frac_tol) & (np.abs(pj - np.rint(pj)) < frac_tol)
    ri = np.rint(pi[conf]).astype(int)
    rj = np.rint(pj[conf]).astype(int)
    m = (image[rj, ri] == image[qj[conf], qi[conf]]).all(axis=1)
    return m.mean(), int(conf.sum())
