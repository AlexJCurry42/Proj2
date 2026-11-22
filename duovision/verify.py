"""Virtual-viewer verification of DuoVision stereograms.

Nothing here imports the encoder (`encoder.py`, `resolve.py`, `patterns.py`) or
`scene.first_hit`: every check is an independent re-implementation so that a bug
in the encoder cannot hide itself.

Pieces
------
march_heightfield   own ray / height-field intersection: fixed-step march
                    (coarse pass at step*coarse, then dense 0.05 mm steps inside
                    the bracketing interval, linear interpolation of the
                    sign change; no bisection, unlike scene.first_hit).
check_links         traces from the RIGHT eye back to the LEFT eye and checks
                    colour(p) == colour(q) for every (sampled) display pixel.
render_eye_view     pinhole camera (axis -z, image plane parallel to xy)
                    looking at the folded display: what one eye sees.
reconstruct_depth   the virtual viewer: two rectified eye views, ZNCC block
                    matching restricted to the depth band the viewer converges
                    on, parabola sub-pixel refinement, left-right check,
                    triangulation into world (x, y, z).
evaluate            reconstruction vs ground truth (scene.z_of), hinge kink and
                    plaque planarity metrics, pass flag.
tolerance_sweep     evaluate across head offsets and hinge-angle mismatches.
comfort_map         per-pixel vergence-accommodation conflict in dioptres.
"""
from __future__ import annotations

import itertools
import math
import time
import warnings
from typing import Iterable, Sequence

import numpy as np
from scipy import ndimage

from .config import guide_band_rows
from .geometry import FoldGeometry

__all__ = [
    "march_heightfield",
    "check_links",
    "render_eye_view",
    "reconstruct_depth",
    "evaluate",
    "tolerance_sweep",
    "default_tolerance_grid",
    "default_res",
    "design_vertical_disparity",
    "comfort_map",
    "PASS_THRESHOLDS",
]

# Pass thresholds for evaluate() (see docstring of evaluate for the reasoning).
PASS_THRESHOLDS = {
    "median_abs_err_mm": 1.0,
    "hinge_kink_mm": 1.0,
    "plaque_kink_deg": 1.0,
    "valid_fraction_linkable": 0.6,
}


# --------------------------------------------------------------------------
# Own ray / height-field intersection
# --------------------------------------------------------------------------
def march_heightfield(scene, origins, dirs, *, step_mm: float = 0.05, coarse: int = 10,
                      chunk: int = 250_000):
    """First intersection of rays with the scene height field.

    A ray point P is "inside" when ``P.z <= scene.z_of(P.x, P.y)``. The march
    covers the slab ``[z_min - 0.5, z_max + 0.5]`` along each ray: a coarse pass
    with step ``step_mm * coarse`` brackets the first inside sample, then a dense
    pass with ``step_mm`` (default 0.05 mm) inside that bracket, then linear
    interpolation of ``f = P.z - z_of`` between the last outside and first inside
    dense samples. Features thinner than the coarse step along the ray
    (0.5 mm by default) can be missed; scene objects are far larger.

    origins, dirs: broadcastable to (..., 3); dirs need not be unit.
    Returns (t, points, hit): t is the distance along the *unit* direction,
    points (..., 3), hit (...) bool. Missing rays have NaN t / points.
    """
    o = np.asarray(origins, dtype=np.float64)
    d = np.asarray(dirs, dtype=np.float64)
    o, d = np.broadcast_arrays(o, d)
    shape = o.shape[:-1]
    o = o.reshape(-1, 3)
    d = d.reshape(-1, 3)
    d = d / np.linalg.norm(d, axis=-1, keepdims=True)
    n = o.shape[0]
    t_out = np.full(n, np.nan)
    hit_out = np.zeros(n, dtype=bool)
    z_hi = float(scene.z_max) + 0.5
    z_lo = float(scene.z_min) - 0.5
    h = step_mm * coarse
    for s in range(0, n, chunk):
        oo, dd = o[s:s + chunk], d[s:s + chunk]
        dz = dd[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            t_a = np.where(dz < 0, (oo[:, 2] - z_hi) / -dz, 0.0)
            t_b = np.where(dz < 0, (oo[:, 2] - z_lo) / -dz, 0.0)
        t_a = np.maximum(t_a, 0.0)
        live = (dz < 0) & (t_b > t_a)
        idx = np.nonzero(live)[0]
        if idx.size == 0:
            continue
        oo, dd, t_a, t_b = oo[idx], dd[idx], t_a[idx], t_b[idx]
        m = idx.size
        k_hit = np.full(m, -1, dtype=np.int64)
        n_steps = int(np.ceil(np.max(t_b - t_a) / h)) + 1
        active = np.arange(m)
        for k in range(n_steps + 1):
            if active.size == 0:
                break
            t = t_a[active] + k * h
            p = oo[active] + t[:, None] * dd[active]
            f = p[:, 2] - scene.z_of(p[:, 0], p[:, 1])
            inside = f <= 0
            k_hit[active[inside]] = k
            keep = (~inside) & (t <= t_b[active])
            active = active[keep]
        got = k_hit >= 0
        gi = np.nonzero(got)[0]
        t_res = np.full(m, np.nan)
        if gi.size:
            k0 = k_hit[gi]
            # dense samples over the bracketing coarse interval
            base = t_a[gi] + np.maximum(k0 - 1, 0) * h
            mm = np.arange(coarse + 1) * step_mm
            tt = base[:, None] + mm[None, :]
            pp = oo[gi][:, None, :] + tt[..., None] * dd[gi][:, None, :]
            ff = pp[..., 2] - scene.z_of(pp[..., 0], pp[..., 1])
            ins = ff <= 0
            ins[:, -1] |= True  # coarse pass guarantees the end is inside (up to z_of noise)
            first = np.argmax(ins, axis=1)
            rows = np.arange(gi.size)
            prev = np.maximum(first - 1, 0)
            f1 = ff[rows, first]
            f0 = ff[rows, prev]
            t1 = tt[rows, first]
            t0 = tt[rows, prev]
            with np.errstate(divide="ignore", invalid="ignore"):
                frac = np.where((first > 0) & (f0 - f1 > 0), f0 / (f0 - f1), 0.0)
            th = np.where(first > 0, t0 + frac * (t1 - t0), t1)
            th = np.where(k0 == 0, t_a[gi], th)
            t_res[gi] = th
        t_out[s + idx] = t_res
        hit_out[s + idx] = got
    pts = o + t_out[:, None] * d
    return t_out.reshape(shape), pts.reshape(shape + (3,)), hit_out.reshape(shape)


def _visible_from(scene, eye, points, *, tol_mm: float = 0.15, **kw):
    """True where the segment eye -> point does not enter the height field
    before reaching the point (within tol_mm)."""
    eye = np.asarray(eye, dtype=np.float64)
    pts = np.asarray(points, dtype=np.float64)
    dist = np.linalg.norm(pts - eye, axis=-1)
    t, _, hit = march_heightfield(scene, eye, pts - eye, **kw)
    return ~hit | (t >= dist - tol_mm)


def _same_colour(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    if a.ndim == 2:
        return np.all(a == b, axis=-1)
    return a == b


# --------------------------------------------------------------------------
# check_links
# --------------------------------------------------------------------------
def check_links(image, geom: FoldGeometry, view, scene, *, reserved_rows=None, sample=None,
                seed: int = 0, frac_tol: float = 0.25, max_examples: int = 10) -> dict:
    """Independent link check, traced from the right eye back to the left eye.

    For each (sampled) display pixel q outside the reserved rows: the ray
    R -> X(q) meets the scene at S (own march); if S is visible from L, the line
    L -> S lands at fractional pixel p. The encoding rule requires
    image[round(p)] == image[q].

    A link is *valid* when S exists, is visible from L, and p lands on the
    display outside the reserved rows on the correct side of q (cross: right).
    It is *confident* when additionally both |p_i - round(p_i)| and
    |p_j - round(p_j)| < frac_tol.

    Keys:
    match_rate        image[round(p)] == image[q] on confident links (the
                      literal rule). Below 1 even for a correct encode: the
                      encoder links P = round(p) to floor(q' + U) (randomized
                      rounding, per the contract), and q' can be up to
                      ~frac_tol px from q, so P may share its colour with a
                      neighbour of q. ~0.93-0.96 at frac_tol 0.25; -> 1 as
                      frac_tol -> 0.
    match_rate_all    the literal rule over all valid links.
    match_rate_rounding_aware
                      on confident, unambiguous links: colour(P) equals the
                      colour of one of the <= 4 pixels that randomized rounding
                      of q' can produce (q' from an own forward trace of P:
                      left eye -> S' -> visible from R -> q'). ~1.0 for a
                      correct encode; ~0 for noise images.

    sample: None = every pixel outside the reserved rows; an int = that many
    random pixels (seeded); full resolution with sample=None takes minutes.
    """
    img = np.asarray(image)
    dev = geom.device
    H, W = dev.height_px, dev.width_px
    if img.shape[:2] != (H, W):
        raise ValueError(f"image shape {img.shape} does not match device {(H, W)}")
    rr = guide_band_rows(dev) if reserved_rows is None else int(reserved_rows)
    L, R = view.eyes()
    n_all = (H - rr) * W
    rng = np.random.default_rng(seed)
    if sample is None or sample >= n_all:
        flat = np.arange(n_all)
    else:
        flat = np.sort(rng.choice(n_all, size=int(sample), replace=False))
    qj = flat // W + rr
    qi = flat % W
    X = geom.pixel_to_world(qi, qj)
    _, S, hit = march_heightfield(scene, R, X - R)
    vis = np.zeros_like(hit)
    if hit.any():
        vis[hit] = _visible_from(scene, L, S[hit])
    pi = np.full(qi.shape, np.nan)
    pj = np.full(qi.shape, np.nan)
    ok_proj = np.zeros_like(hit)
    if vis.any():
        a, b, okp = geom.project_from_eye(L, S[vis])
        pi[vis], pj[vis], ok_proj[vis] = a, b, okp
    ri = np.rint(pi)
    rj = np.rint(pj)
    cross = getattr(view, "mode", "cross") == "cross"
    with np.errstate(invalid="ignore"):
        on_disp = ok_proj & (ri >= 0) & (ri <= W - 1) & (rj >= rr) & (rj <= H - 1)
        on_disp &= (pi >= qi + 1) if cross else (pi <= qi - 1)
    valid = hit & vis & on_disp
    with np.errstate(invalid="ignore"):
        conf = valid & (np.abs(pi - ri) < frac_tol) & (np.abs(pj - rj) < frac_tol)
    Pi = np.where(valid, ri, 0).astype(np.int64)
    Pj = np.where(valid, rj, rr).astype(np.int64)
    exact = np.zeros_like(valid)
    vi = np.nonzero(valid)[0]
    if vi.size:
        exact[vi] = _same_colour(img[qj[vi], qi[vi]], img[Pj[vi], Pi[vi]])

    # Rounding-aware check on the confident links. The encoder links P = round(p)
    # to a *randomly rounded* partner floor(q' + U) of its fractional partner q'
    # (contract: resolve.round_partners), so P may legitimately share its colour
    # with a neighbour of q instead of q. Trace P forward (own march, left eye ->
    # S' -> visible from R -> q') and accept the link when colour(P) equals the
    # colour of one of the <= 4 pixels the rounding can produce. Links where a
    # candidate would be illegal (reserved rows, off display, wrong direction)
    # are "ambiguous" and left out.
    match = np.zeros_like(valid)
    ambiguous = np.zeros_like(valid)
    ci = np.nonzero(conf)[0]
    q2i = np.full(qi.shape, np.nan)
    q2j = np.full(qi.shape, np.nan)
    if ci.size:
        XP = geom.pixel_to_world(Pi[ci], Pj[ci])
        _, S2, h2 = march_heightfield(scene, L, XP - L)
        ok2 = h2.copy()
        if ok2.any():
            ok2[ok2] = _visible_from(scene, R, S2[ok2])
        a2 = np.full(ci.shape, np.nan)
        b2 = np.full(ci.shape, np.nan)
        if ok2.any():
            a, b, okp = geom.project_from_eye(R, S2[ok2])
            a2[ok2], b2[ok2] = np.where(okp, a, np.nan), np.where(okp, b, np.nan)
        q2i[ci], q2j[ci] = a2, b2
        good = np.isfinite(a2) & np.isfinite(b2)
        cp = img[Pj[ci], Pi[ci]]
        anym = np.zeros(ci.shape, dtype=bool)
        legal_all = good.copy()
        fi0 = np.floor(np.where(good, a2, 0)).astype(np.int64)
        fj0 = np.floor(np.where(good, b2, 0)).astype(np.int64)
        fri = np.where(good, a2, 0) - fi0
        frj = np.where(good, b2, 0) - fj0
        for di in (0, 1):
            for dj in (0, 1):
                possible = good & ((fri > 0) if di else (fri < 1)) & ((frj > 0) if dj else (frj < 1))
                ci_c = fi0 + di
                cj_c = fj0 + dj
                legal = (ci_c >= 0) & (ci_c <= W - 1) & (cj_c >= rr) & (cj_c <= H - 1)
                legal &= (ci_c < Pi[ci]) if cross else (ci_c > Pi[ci])
                legal_all &= ~possible | legal
                use = possible & legal
                if use.any():
                    cc = img[np.where(use, cj_c, rr), np.where(use, ci_c, 0)]
                    anym |= use & _same_colour(cc, cp)
        match[ci] = anym
        ambiguous[ci] = ~legal_all
    scored = conf & ~ambiguous
    n_valid = int(valid.sum())
    n_scored = int(scored.sum())
    examples = []
    bad = np.nonzero(scored & ~match)[0][:max_examples]
    for k in bad:
        examples.append({
            "q": (int(qi[k]), int(qj[k])),
            "p": (float(pi[k]), float(pj[k])),
            "q_forward": (float(q2i[k]), float(q2j[k])),
            "S": tuple(float(v) for v in S[k]),
            "colour_q": img[qj[k], qi[k]].tolist(),
            "colour_p": img[Pj[k], Pi[k]].tolist(),
        })
    sep = np.abs(qi - pi)[valid] * geom.mm_per_px
    return {
        "match_rate": float(exact[conf].mean()) if conf.any() else float("nan"),
        "match_rate_all": float(exact[valid].mean()) if n_valid else float("nan"),
        "match_rate_rounding_aware": float(match[scored].mean()) if n_scored else float("nan"),
        "n_checked": int(qi.size),
        "n_valid": n_valid,
        "n_confident": int(conf.sum()),
        "n_scored": n_scored,
        "n_ambiguous": int((conf & ambiguous).sum()),
        "n_matched": int(match[scored].sum()),
        "valid_fraction": n_valid / max(1, qi.size),
        "n_no_hit": int((~hit).sum()),
        "n_occluded_from_left": int((hit & ~vis).sum()),
        "n_off_display": int((hit & vis & ~on_disp).sum()),
        "pixel_separation_mm_range": (float(np.min(sep)), float(np.max(sep))) if n_valid else None,
        "mismatch_examples": examples,
    }


# --------------------------------------------------------------------------
# render_eye_view
# --------------------------------------------------------------------------
def _display_corners(geom: FoldGeometry) -> np.ndarray:
    w, h2 = geom.half_width_mm, geom.height_mm / 2.0
    pts = []
    for a in (geom.a_left, geom.a_right):
        for y in (-h2, h2):
            pts.append(w * a + np.array([0.0, y, 0.0]))
    pts += [np.array([0.0, -h2, 0.0]), np.array([0.0, h2, 0.0])]
    return np.array(pts)


def _tangents(eye, pts):
    eye = np.asarray(eye, dtype=np.float64)
    dz = eye[2] - pts[..., 2]
    return (pts[..., 0] - eye[0]) / dz, (pts[..., 1] - eye[1]) / dz


def render_eye_view(image, geom: FoldGeometry, eye, *, res=(1600, 1100), fov_deg=None,
                    focal_px=None, principal=None, margin: float = 0.05) -> dict:
    """What one eye sees of the display image on the folded panel.

    Pinhole camera at `eye`, optical axis along -z, image plane parallel to xy,
    square pixels. Retinal pixel (a, b) (column, row; row 0 at top) looks along
    ``((a - cx)/f, -(b - cy)/f, -1)``. By default the focal length and principal
    point make the display's angular extent fill the frame with `margin` on each
    side; `fov_deg` (full horizontal field) fixes f instead; `focal_px` and
    `principal=(cx, cy)` override both (used for rectified stereo pairs).

    Each retinal ray -> geom.intersect -> nearest display pixel. Outside the
    display the image is 0.

    Returns dict: image (h, w[, 3]) same dtype as input, focal_px, cx, cy, eye,
    sees (h, w) bool (ray hits the display), disp_i / disp_j (int display pixel
    sampled, -1 outside), world (h, w, 3) display hit point (NaN outside).
    """
    img = np.asarray(image)
    w, h = int(res[0]), int(res[1])
    eye = np.asarray(eye, dtype=np.float64)
    tx, ty = _tangents(eye, _display_corners(geom))
    if focal_px is None:
        if fov_deg is not None:
            focal_px = (w / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
        else:
            focal_px = min(w / (tx.max() - tx.min()), h / (ty.max() - ty.min())) / (1.0 + 2.0 * margin)
    f = float(focal_px)
    if principal is None:
        cx = (w - 1) / 2.0 - f * (tx.max() + tx.min()) / 2.0
        cy = (h - 1) / 2.0 + f * (ty.max() + ty.min()) / 2.0
    else:
        cx, cy = float(principal[0]), float(principal[1])
    dev = geom.device
    out_shape = (h, w) + img.shape[2:]
    out = np.zeros(out_shape, dtype=img.dtype)
    sees = np.zeros((h, w), dtype=bool)
    di = np.full((h, w), -1, dtype=np.int64)
    dj = np.full((h, w), -1, dtype=np.int64)
    world = np.full((h, w, 3), np.nan)
    a = (np.arange(w) - cx) / f
    rows_per_chunk = max(1, 400_000 // w)
    for b0 in range(0, h, rows_per_chunk):
        b = np.arange(b0, min(h, b0 + rows_per_chunk))
        bb = -(b - cy) / f
        dirs = np.empty((b.size, w, 3))
        dirs[..., 0] = a[None, :]
        dirs[..., 1] = bb[:, None]
        dirs[..., 2] = -1.0
        u, v, t, ok = geom.intersect(eye, dirs)
        i, j = geom.uv_to_pixel(u, v)
        ii = np.rint(np.where(ok, i, 0)).astype(np.int64)
        jj = np.rint(np.where(ok, j, 0)).astype(np.int64)
        ok &= (ii >= 0) & (ii < dev.width_px) & (jj >= 0) & (jj < dev.height_px)
        sl = slice(b[0], b[-1] + 1)
        sees[sl] = ok
        di[sl] = np.where(ok, ii, -1)
        dj[sl] = np.where(ok, jj, -1)
        world[sl] = np.where(ok[..., None], eye + t[..., None] * dirs, np.nan)
        blk = img[np.where(ok, jj, 0), np.where(ok, ii, 0)]
        if img.ndim == 3:
            blk = np.where(ok[..., None], blk, 0)
        else:
            blk = np.where(ok, blk, 0)
        out[sl] = blk
    return {"image": out, "focal_px": f, "cx": cx, "cy": cy, "eye": eye, "sees": sees,
            "disp_i": di, "disp_j": dj, "world": world}


def _default_z_band(view) -> tuple:
    """World-z band the virtual viewer searches. Cross mode (the contract):
    (z_background - 15, z_near + 15). Parallel mode (scene behind the display,
    relief rising toward the viewer from z_background; view.z_near is *farther*
    there): (min - 15, max + 15) of the two, extended up to z_background + 35
    but kept behind the hinge plane."""
    zb, zn = float(view.z_background), float(view.z_near)
    if getattr(view, "mode", "cross") == "cross":
        return (zb - 15.0, zn + 15.0)
    lo, hi = min(zb, zn) - 15.0, max(zb, zn) + 15.0
    return (lo, min(max(hi, zb + 35.0), -5.0))


def default_res(device) -> tuple:
    """Retinal resolution for the virtual viewer: (1600, 1100) at full device
    resolution, scaled with the device's pixel density (so a retinal pixel
    always covers ~1.8 display pixels and the 9 px window spans ~3 dots)."""
    s = device.width_px / 2670.0
    return (max(160, int(round(1600 * s))), max(110, int(round(1100 * s))))


# --------------------------------------------------------------------------
# reconstruct_depth: the virtual viewer
# --------------------------------------------------------------------------
def _gray(img):
    img = np.asarray(img)
    if img.ndim == 3:
        return (img[..., 0] * 0.299 + img[..., 1] * 0.587 + img[..., 2] * 0.114).astype(np.float32)
    return img.astype(np.float32)


def _smooth_v(a, ok, sigma):
    return np.where(ok, ndimage.gaussian_filter1d(a, sigma, axis=0), 0.0).astype(np.float32)


def _shift_cols(a, d, fill):
    """out[:, x] = a[:, x - d] (fill outside)."""
    out = np.full_like(a, fill)
    W = a.shape[1]
    if d >= 0:
        if d < W:
            out[:, d:] = a[:, :W - d]
    else:
        if -d < W:
            out[:, :W + d] = a[:, -d:]
    return out


def _vertical_field(gl, gr, okl, okr, d_lo, d_hi, max_dv, *, patch=None, step=None, min_ncc=0.4):
    """Smooth vertical-disparity field dv(x, y) = y_right - y_left (pixels) on the
    left retinal grid, from patch-wise 2D normalised cross-correlation (FFT),
    searching horizontal disparities [d_lo, d_hi] and |dv| <= max_dv. Pattern
    repeats share the same dv, so the estimate does not depend on which repeat
    wins. Returns (field (h, w) float32, grid of raw estimates)."""
    from scipy.signal import correlate
    h, w = gl.shape
    # patch / step scale with the retinal width (64 / 16 px at 400 px wide):
    # the field is a smooth function of position on the display, so it should be
    # sampled at the same angular spacing whatever the retinal resolution.
    if patch is None:
        patch = 2 * max(16, int(round(32 * w / 400.0)))
    if step is None:
        step = max(8, int(round(16 * w / 400.0)))
    V = int(math.ceil(max_dv))
    half = patch // 2
    ys = np.arange(half + V, h - half - V, step)
    xs = np.arange(half, w - half, step)
    grid = np.full((ys.size, xs.size), np.nan)
    if ys.size == 0 or xs.size == 0 or V == 0:
        return np.zeros((h, w), np.float32), grid
    okl_frac = ndimage.uniform_filter(okl.astype(np.float32), size=patch)
    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            if okl_frac[y, x] < 0.999:
                continue
            c0, c1 = x - half - d_hi, x + half - d_lo
            if c0 < 0 or c1 > w:
                continue
            T = gl[y - half:y + half, x - half:x + half]
            Tz = T - T.mean()
            tn = float(np.sqrt((Tz * Tz).sum()))
            if tn < 1e-3 * patch:
                continue
            Rg = gr[y - half - V:y + half + V, c0:c1]
            if not okr[y - half - V:y + half + V, c0:c1].all():
                continue
            num = correlate(Rg, Tz, mode="valid", method="fft")
            # local energy of the right windows (zero-mean)
            s1 = correlate(Rg, np.ones_like(Tz), mode="valid", method="fft")
            s2 = correlate(Rg * Rg, np.ones_like(Tz), mode="valid", method="fft")
            var = np.maximum(s2 - s1 * s1 / Tz.size, 1e-6)
            ncc = num / (tn * np.sqrt(var))
            r, c = np.unravel_index(np.argmax(ncc), ncc.shape)
            if ncc[r, c] < min_ncc:
                continue
            dv = r - V
            if 0 < r < ncc.shape[0] - 1:
                a, b, cc = ncc[r - 1, c], ncc[r, c], ncc[r + 1, c]
                den = a - 2 * b + cc
                if den < 0:
                    dv += float(np.clip(0.5 * (a - cc) / den, -0.5, 0.5))
            grid[iy, ix] = dv
    good = np.isfinite(grid)
    if not good.any():
        return np.zeros((h, w), np.float32), grid
    # robust smoothing: 3x3 nan-median, then fill holes from the nearest estimate
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN neighbourhoods
        med = ndimage.generic_filter(grid, np.nanmedian, size=3, mode="nearest")
    filled = np.isfinite(med)
    idx = ndimage.distance_transform_edt(~filled, return_distances=False, return_indices=True)
    med = med[tuple(idx)]
    med = np.clip(med, -max_dv, max_dv)
    yy = np.interp(np.arange(h), ys, np.arange(ys.size))
    xx = np.interp(np.arange(w), xs, np.arange(xs.size))
    field = ndimage.map_coordinates(med, np.meshgrid(yy, xx, indexing="ij"), order=1, mode="nearest")
    return field.astype(np.float32), grid


def reconstruct_depth(image, geom: FoldGeometry, view, *, head_offset=(0.0, 0.0, 0.0),
                      geom_viewed: FoldGeometry | None = None, res=None, window: int = 9,
                      min_score: float = 0.5, tie_tol: float = 0.2, lr_tol: float = 1.0, z_band=None,
                      reserved_rows=None, exclude_display_px: float = 2.0,
                      margin: float = 0.06, vertical_compensation: bool = True,
                      max_vertical_arcmin: float = 8.0, field_smooth_px: float = 0.7,
                      match_vsmooth_px: float = 0.5) -> dict:
    """Simulated viewer: stereo-match what the two eyes see, triangulate.

    Eyes = view.eyes(head_offset). res=None uses default_res(device):
    (1600, 1100) at full resolution, scaled down with scaled devices. The image is shown on `geom_viewed`
    (default `geom`; differ to simulate a hinge-angle mismatch). Both pinhole
    cameras share orientation (axis -z) and focal length f, so they are
    rectified; their principal points are shifted so the convergence point
    F = (0, 0, view.z_background) lands at the image centre of both (sensor
    shift, like locked vergence). Image disparity d = xL - xR relates to the
    eye-plane distance Z by ``d + f*IPD/Z_F = f*IPD/Z``.

    Search: integer d over the band of world depths
    ``z_band = (view.z_background - 15, view.z_near + 15)`` in cross mode
    (see _default_z_band for parallel) (the viewer's
    locked convergence; excludes the display surface and second-order pattern
    repeats). Candidates within `exclude_display_px` of the display surface's
    own disparity at that pixel are also rejected (the physical surface is a
    trivial perfect match; it only enters the band at shallow hinges).
    Cost: ZNCC over window x window (box filters per disparity). Pass 1 finds
    the best score per pixel; pass 2 takes, among local maxima scoring at least
    max(min_score, best - tie_tol), the one nearest the convergence disparity
    (pattern repeats give several near-perfect peaks; the viewer's vergence is
    locked near the background by the guide dots). Sub-pixel: parabola through
    the three scores around it. Valid when such a peak exists, the left-right
    check (same rule on the right image) agrees within lr_tol px, and the eye's
    pixel sees the display outside the guide band (the ZNCC window must lie
    entirely on usable display in both eyes).

    Vertical compensation (vertical_compensation=True): a smooth vertical-
    disparity field (|dv| <= max_vertical_arcmin) is estimated by patch-wise 2D
    ZNCC on Gaussian-prefiltered views (field_smooth_px; patch and step scale
    with the retinal width). If it reaches 0.25 px the right view is resampled
    by it and both views get a vertical-only Gaussian (match_vsmooth_px) before
    the horizontal search; otherwise the views are matched unfiltered.

    Returns dict with disparity (image disparity, NaN invalid), valid, score,
    x, y, z (world, on the left-eye retinal grid, NaN invalid), sees_display
    (left retinal pixel sees the display outside the guide band), left/right
    (the render_eye_view dicts without images), focal_px, cx_left, cx_right,
    cy, eyes, d_range, timings.
    """
    t_start = time.perf_counter()
    gv = geom if geom_viewed is None else geom_viewed
    if res is None:
        res = default_res(gv.device)
    w, h = int(res[0]), int(res[1])
    L, R = view.eyes(tuple(head_offset))
    ipd = float(R[0] - L[0])
    rr = guide_band_rows(gv.device) if reserved_rows is None else int(reserved_rows)
    F = np.array([0.0, 0.0, float(view.z_background)])
    # shared focal length: the display (around F) must fit both eyes' frames
    corners = _display_corners(gv)
    need_x, need_y = 0.0, 0.0
    for eye in (L, R):
        tx, ty = _tangents(eye, corners)
        fx, fy = _tangents(eye, F[None])
        need_x = max(need_x, np.max(np.abs(tx - fx[0])))
        need_y = max(need_y, np.max(np.abs(ty - fy[0])))
    f = min((w / 2.0) / need_x, (h / 2.0) / need_y) / (1.0 + margin)
    c0x, c0y = (w - 1) / 2.0, (h - 1) / 2.0
    views = []
    for eye in (L, R):
        fx, fy = _tangents(eye, F[None])
        cx = c0x - f * fx[0]
        cy = c0y + f * fy[0]
        views.append(render_eye_view(image, gv, eye, res=res, focal_px=f, principal=(cx, cy)))
    vl, vr = views
    cy = vl["cy"]
    t_render = time.perf_counter() - t_start
    cxL, cxR = vl["cx"], vr["cx"]
    shift = cxL - cxR  # d_img = f*ipd/Z + shift

    good_l = vl["sees"] & (vl["disp_j"] >= rr)
    good_r = vr["sees"] & (vr["disp_j"] >= rr)
    if z_band is None:
        z_band = _default_z_band(view)
    ez = L[2]
    d_lo_f = f * ipd / (ez - z_band[0]) + shift
    d_hi_f = f * ipd / (ez - z_band[1]) + shift
    d_lo, d_hi = int(math.floor(min(d_lo_f, d_hi_f))), int(math.ceil(max(d_lo_f, d_hi_f)))
    disps = np.arange(d_lo, d_hi + 1)

    # display-surface disparity at each left pixel (excluded: trivial match)
    Xd = vl["world"]
    with np.errstate(invalid="ignore"):
        xr_disp = cxR + f * (Xd[..., 0] - R[0]) / (R[2] - Xd[..., 2])
    a_idx = np.arange(w)[None, :].astype(np.float64)
    d_disp = np.where(vl["sees"], a_idx - xr_disp, np.nan).astype(np.float32)
    xl_disp_r = None
    Xr = vr["world"]
    with np.errstate(invalid="ignore"):
        xl_disp_r = cxL + f * (Xr[..., 0] - L[0]) / (L[2] - Xr[..., 2])
    d_disp_r = np.where(vr["sees"], xl_disp_r - a_idx, np.nan).astype(np.float32)

    gl = _gray(vl["image"])
    gr = _gray(vr["image"])
    gl = np.where(good_l, gl, 0.0).astype(np.float32)
    gr = np.where(good_r, gr, 0.0).astype(np.float32)
    # Vertical disparity: off the design viewpoint (dy != 0, hinge mismatch) the
    # encoded pairs no longer lie on the same retinal row. The visual system
    # tolerates small, smoothly varying vertical disparities (Panum's area is
    # roughly 6-10 arcmin vertically near the fovea); model this by estimating a
    # smooth field dv(x, y) (|dv| <= max_vertical_arcmin) and resampling the
    # right image before the horizontal (epipolar) search.
    max_dv = max_vertical_arcmin / 60.0 * math.pi / 180.0 * f
    dv_field = None
    if vertical_compensation and max_dv >= 0.5:
        # The field is estimated on a Gaussian-prefiltered copy (field_smooth_px,
        # retinal px): the dot pattern is barely resolved (one retinal pixel is
        # ~1.8 display pixels), so the vertical ZNCC peak is too sharp for a
        # parabola and sub-pixel estimates lock to whole pixels without it.
        sl, sr = gl, gr
        if field_smooth_px > 0:
            sl = np.where(good_l, ndimage.gaussian_filter(gl, field_smooth_px), 0.0).astype(np.float32)
            sr = np.where(good_r, ndimage.gaussian_filter(gr, field_smooth_px), 0.0).astype(np.float32)
        dv_field, _ = _vertical_field(sl, sr, good_l, good_r, d_lo, d_hi, max_dv)
        if np.abs(dv_field).max() >= 0.25:
            rows_w = np.arange(h, dtype=np.float32)[:, None] + dv_field
            cols_w = np.broadcast_to(np.arange(w, dtype=np.float32)[None, :], (h, w))
            gr = ndimage.map_coordinates(gr, [rows_w, cols_w], order=1, mode="constant").astype(np.float32)
            good_r = ndimage.map_coordinates(good_r.astype(np.float32), [rows_w, cols_w], order=1,
                                             mode="constant") > 0.999
            gr = np.where(good_r, gr, 0.0).astype(np.float32)
            # Bilinear resampling blurred the right view vertically by a varying
            # sub-pixel amount; blur both views vertically by the same small
            # sigma so ZNCC compares like with like. Vertical only: horizontal
            # sharpness is what tells pattern repeats (disparities) apart.
            if match_vsmooth_px > 0:
                gl = _smooth_v(gl, good_l, match_vsmooth_px)
                gr = _smooth_v(gr, good_r, match_vsmooth_px)
    uf = lambda a: ndimage.uniform_filter(a, size=window, mode="constant")  # noqa: E731
    mL = uf(gl)
    vL = uf(gl * gl) - mL * mL
    mR = uf(gr)
    vR = uf(gr * gr) - mR * mR
    # window must lie entirely on usable display (no black border / guide band inside)
    fullL = uf(good_l.astype(np.float32)) > 0.999
    fullR = uf(good_r.astype(np.float32)) > 0.999
    var_eps = 1.0  # grey levels^2; textureless windows are rejected

    neg = np.float32(-2.0)

    def scores(d):
        """ZNCC for disparity d: left map (indexed by xL) and right map (by xR)."""
        d = int(d)
        grs = _shift_cols(gr, d, 0.0)
        mRs = _shift_cols(mR, d, 0.0)
        vRs = _shift_cols(vR, d, 0.0)
        okR = _shift_cols(fullR, d, False)
        cov = uf(gl * grs) - mL * mRs
        den = np.sqrt(np.maximum(vL, 0) * np.maximum(vRs, 0))
        ok = fullL & okR & (vL > var_eps) & (vRs > var_eps)
        with np.errstate(divide="ignore", invalid="ignore"):
            sc = np.where(ok, cov / den, neg).astype(np.float32)
        scR = _shift_cols(sc, -d, neg)
        if exclude_display_px > 0:
            with np.errstate(invalid="ignore"):
                sc = np.where(np.abs(d - d_disp) < exclude_display_px, neg, sc)
                scR = np.where(np.abs(d - d_disp_r) < exclude_display_px, neg, scR)
        return sc, scR

    # pass 1: best score per pixel over the band
    bestL = np.full((h, w), neg, np.float32)
    bestR = np.full((h, w), neg, np.float32)
    for d in disps:
        sc, scR = scores(d)
        np.maximum(bestL, sc, out=bestL)
        np.maximum(bestR, scR, out=bestR)
    # pass 2: among local maxima scoring within tie_tol of the best (pattern
    # repeats give several near-perfect peaks), take the one nearest the
    # convergence depth (image disparity 0): the viewer's vergence is locked
    # on the guide dots at the background depth.
    accL = np.maximum(bestL - tie_tol, min_score)
    accR = np.maximum(bestR - tie_tol, min_score)
    big = np.int32(10**6)
    dL = np.full((h, w), big, np.int32)
    dR = np.full((h, w), big, np.int32)
    cmL = np.full((h, w), np.nan, np.float32)
    c0L = np.full((h, w), np.nan, np.float32)
    cpL = np.full((h, w), np.nan, np.float32)
    prev = scores(disps[0] - 1)
    cur = scores(disps[0])
    for d in disps:
        nxt = scores(d + 1)
        for side in (0, 1):
            pv, cu, nx = prev[side], cur[side], nxt[side]
            acc = accL if side == 0 else accR
            dd = dL if side == 0 else dR
            cand = (cu >= acc) & (cu >= pv) & (cu >= nx) & (abs(d) < np.abs(dd))
            dd[cand] = d
            if side == 0:
                cmL[cand] = pv[cand]
                c0L[cand] = cu[cand]
                cpL[cand] = nx[cand]
        prev, cur = cur, nxt
    t_match = time.perf_counter() - t_start - t_render

    with np.errstate(divide="ignore", invalid="ignore"):
        den = cmL - 2 * c0L + cpL
        delta = np.where(np.isfinite(den) & (den < 0), 0.5 * (cmL - cpL) / den, 0.0)
    delta = np.clip(delta, -0.5, 0.5)
    found = dL != big
    disp = np.where(found, dL + delta, np.nan)
    # left-right consistency
    xr = np.rint(np.where(found, a_idx - disp, -1)).astype(np.int64)
    inr = found & (xr >= 0) & (xr < w)
    rows = np.arange(h)[:, None]
    dR_at = np.where(inr, dR[rows, np.clip(xr, 0, w - 1)], big)
    lr_ok = inr & (dR_at != big) & (np.abs(dR_at - dL) <= lr_tol)
    valid = good_l & found & lr_ok
    d_abs = disp - shift
    with np.errstate(divide="ignore", invalid="ignore"):
        Z = f * ipd / d_abs
    valid &= np.isfinite(Z) & (Z > 0)
    C = (L + R) / 2.0
    xl_n = (a_idx - cxL) / f
    xr_n = (a_idx - disp - cxR) / f
    yy = -(np.arange(h)[:, None] - cy) / f
    X = C[0] + 0.5 * (xl_n + xr_n) * Z
    Y = C[1] + yy * Z
    Zw = ez - Z
    nan = np.float64(np.nan)
    out = {
        "disparity": np.where(valid, disp, nan),
        "valid": valid,
        "score": np.where(found, c0L, np.nan),
        "x": np.where(valid, X, nan),
        "y": np.where(valid, Y, nan),
        "z": np.where(valid, Zw, nan),
        "sees_display": good_l,
        "focal_px": f,
        "cx_left": cxL,
        "cx_right": cxR,
        "cy": cy,
        "eyes": (L, R),
        "d_range": (d_lo, d_hi),
        "dv_field": dv_field,
        "vertical_disparity_est_arcmin_p95": (
            float(np.percentile(np.abs(dv_field[valid]), 95)) / f * 180.0 / math.pi * 60.0
            if dv_field is not None and valid.any() else 0.0),
        "z_band": tuple(z_band),
        "left": {k: v for k, v in vl.items() if k != "image"},
        "right": {k: v for k, v in vr.items() if k != "image"},
        "left_image": vl["image"],
        "right_image": vr["image"],
        "timings": {"render_s": t_render, "match_s": t_match,
                    "total_s": time.perf_counter() - t_start},
    }
    return out


# --------------------------------------------------------------------------
# evaluate
# --------------------------------------------------------------------------
def _plane_fit(x, y, z, *, robust_mm: float = 2.0):
    """Least-squares plane z = a x + b y + c, one trimming pass (|res| > robust_mm)."""
    A = np.stack([x, y, np.ones_like(x)], axis=1)
    coef, *_ = np.linalg.lstsq(A, z, rcond=None)
    res = z - A @ coef
    keep = np.abs(res) <= max(robust_mm, 3.0 * float(np.median(np.abs(res))) if res.size else 0.0)
    if keep.sum() >= 3 and not keep.all():
        coef, *_ = np.linalg.lstsq(A[keep], z[keep], rcond=None)
        res = z[keep] - A[keep] @ coef
    return coef, res, keep


def _normal(coef):
    n = np.array([-coef[0], -coef[1], 1.0])
    return n / np.linalg.norm(n)


def _linkable_mask(geom_v, view, scene, rec, stride, rr):
    """Ground truth: left retinal pixels (on a stride grid) whose scene point S
    is visible from both eyes and whose eye->S lines land on the display outside
    the guide band. Returns (mask on the stride grid, row idx, col idx)."""
    L, R = rec["eyes"]
    f, cxL, cy = rec["focal_px"], rec["cx_left"], rec["cy"]
    h, w = rec["valid"].shape
    bi = np.arange(stride // 2, h, stride)
    ai = np.arange(stride // 2, w, stride)
    sees = rec["sees_display"][np.ix_(bi, ai)]
    dirs = np.stack(np.broadcast_arrays((ai[None, :] - cxL) / f, -(bi[:, None] - cy) / f, -1.0), axis=-1)
    m = np.zeros(sees.shape, dtype=bool)
    if not sees.any():
        return m, bi, ai
    dsel = dirs[sees]
    _, S, hit = march_heightfield(scene, L, dsel)
    ok = hit.copy()
    if ok.any():
        ok[ok] = _visible_from(scene, R, S[ok])
    for eye in (L, R):
        if ok.any():
            u, v, _, okd = geom_v.intersect(eye, S[ok] - eye)
            _, j = geom_v.uv_to_pixel(u, v)
            with np.errstate(invalid="ignore"):
                ok[ok] = okd & (np.rint(j) >= rr)
    m[sees] = ok
    return m, bi, ai


def design_vertical_disparity(geom_encoded, view, scene, *, geom_viewed=None, head_offset=(0.0, 0.0, 0.0),
                              n: int = 20000, seed: int = 0, reserved_rows=None) -> dict:
    """Vertical disparity (arcmin) of the fold-aware design links seen off-design.

    Samples display pixels q, traces the design links (R0 -> X(q) -> S -> L0 -> p,
    own march, design eyes, encoded fold), then measures the difference between
    the elevation angles of p seen by the displaced left eye and q seen by the
    displaced right eye, with the image shown on geom_viewed. Zero at the design
    viewpoint. Only meaningful for fold-aware encodes (the flat baseline has
    different links)."""
    gv = geom_encoded if geom_viewed is None else geom_viewed
    dev = geom_encoded.device
    rr = guide_band_rows(dev) if reserved_rows is None else int(reserved_rows)
    rng = np.random.default_rng(seed)
    qi = rng.integers(0, dev.width_px, n)
    qj = rng.integers(rr, dev.height_px, n)
    L0, R0 = view.eyes()
    X = geom_encoded.pixel_to_world(qi, qj)
    _, S, hit = march_heightfield(scene, R0, X - R0)
    ok = hit.copy()
    if ok.any():
        ok[ok] = _visible_from(scene, L0, S[ok])
    pi = np.full(n, np.nan)
    pj = np.full(n, np.nan)
    if ok.any():
        a, b, okp = geom_encoded.project_from_eye(L0, S[ok])
        pi[ok], pj[ok] = a, b
        ok[ok] = okp
    with np.errstate(invalid="ignore"):
        ok &= (pj >= rr) & (pi >= 0) & (pi <= dev.width_px - 1) & (pj <= dev.height_px - 1)
    if not ok.any():
        return {"median_arcmin": float("nan"), "p95_arcmin": float("nan"), "max_arcmin": float("nan")}
    L, R = view.eyes(tuple(head_offset))
    Xp = gv.pixel_to_world(pi[ok], pj[ok])
    Xq = gv.pixel_to_world(qi[ok], qj[ok])
    el = np.arctan2(Xp[:, 1] - L[1], L[2] - Xp[:, 2])
    er = np.arctan2(Xq[:, 1] - R[1], R[2] - Xq[:, 2])
    dv = np.abs(np.degrees(el - er) * 60.0)
    return {"median_arcmin": float(np.median(dv)), "p95_arcmin": float(np.percentile(dv, 95)),
            "max_arcmin": float(dv.max())}


def evaluate(image, geom_encoded: FoldGeometry, view, scene, *, geom_viewed=None,
             head_offset=(0.0, 0.0, 0.0), res=None, window: int = 9, gt_stride: int | None = None,
             reserved_rows=None, recon: dict | None = None, return_recon: bool = False,
             **recon_kw) -> dict:
    """Reconstruct what the virtual viewer fuses and compare with the scene.

    Errors: err = z_rec - scene.z_of(x_rec, y_rec) over valid reconstructed
    pixels (world z, mm).

    Metrics
    -------
    valid_fraction            valid / left retinal pixels that see the display
                              outside the guide band (includes the unlinkable
                              outer strips, so < 1 even for a perfect encode).
    valid_fraction_linkable   valid fraction over the pixels whose ground-truth
                              scene point is visible to both eyes and whose two
                              eye lines land on the display outside the guide
                              band (estimated on a gt_stride grid).
    median_abs_err_mm, p95_abs_err_mm, mean_signed_err_mm, inlier_fraction (|err| < 2 mm)
    vertical_disparity_design_p95_arcmin
                              p95 |vertical disparity| of the fold-aware design
                              links at this head offset / fold
                              (design_vertical_disparity); the est_ variant is
                              the field the virtual viewer measured and
                              compensated (capped at max_vertical_arcmin).
    hinge_kink_mm             median signed err in |x| < 5 mm minus the mean of
                              the median signed errs in -25 < x < -10 and
                              10 < x < 25: a bump/dent of the perceived surface
                              along the hinge relative to its neighbourhood.
    hinge_slope_kink_deg      atan(slope_R) - atan(slope_L) of linear fits of
                              signed err vs x on 3 < |x| < 40 mm, per side: a
                              fold-shaped (V) bend of the perceived surface.
    plaque_* (scene.name == 'plaque'): points inside the plaque mask (eroded by
                              1.5 mm), robust plane fits. plaque_flatness_rms_mm
                              = rms residual of one plane over the plaque,
                              plaque_kink_deg = angle between planes fitted to
                              x < -2 and x > 2, plaque_tilt_deg = angle of the
                              whole-plaque plane to the frontal plane.

    return_recon=True adds the reconstruct_depth dict under "recon" (maps);
    otherwise the result holds only scalars and small tuples.

    Pass (PASS_THRESHOLDS): median_abs_err < 1 mm (~3 retinal disparity px at
    res 1600 wide; a viewer cannot tell a 1 mm depth offset at 240 mm),
    |hinge_kink| < 1 mm, plaque_kink < 1 deg (a crease along the hinge is the
    failure this product must avoid; 1 deg is ~0.4 mm over a 25 mm half plaque),
    valid_fraction_linkable > 0.6 (most of the scene actually fuses).
    """
    t0 = time.perf_counter()
    gv = geom_encoded if geom_viewed is None else geom_viewed
    rr = guide_band_rows(gv.device) if reserved_rows is None else int(reserved_rows)
    if gt_stride is None:
        gt_stride = max(1, int(round(4 * gv.device.width_px / 2670.0)))
    if recon is None:
        recon = reconstruct_depth(image, geom_encoded, view, head_offset=head_offset,
                                  geom_viewed=gv, res=res, window=window, reserved_rows=rr, **recon_kw)
    valid = recon["valid"]
    sees = recon["sees_display"]
    x, y, z = recon["x"][valid], recon["y"][valid], recon["z"][valid]
    zt = np.asarray(scene.z_of(x, y), dtype=np.float64)
    err = z - zt
    abs_err = np.abs(err)
    n = int(valid.sum())
    t_gt = time.perf_counter()
    lm, bi, ai = _linkable_mask(gv, view, scene, recon, gt_stride, rr)
    vgrid = valid[np.ix_(bi, ai)]
    t_gt = time.perf_counter() - t_gt

    def med(sel):
        return float(np.median(err[sel])) if np.count_nonzero(sel) >= 20 else float("nan")

    out = {
        "hinge_deg_encoded": geom_encoded.hinge_deg,
        "hinge_deg_viewed": gv.hinge_deg,
        "head_offset": tuple(float(v) for v in head_offset),
        "n_valid": n,
        "valid_fraction": n / max(1, int(sees.sum())),
        "valid_fraction_linkable": float(vgrid[lm].mean()) if lm.any() else float("nan"),
        "median_abs_err_mm": float(np.median(abs_err)) if n else float("nan"),
        "p95_abs_err_mm": float(np.percentile(abs_err, 95)) if n else float("nan"),
        "mean_signed_err_mm": float(np.mean(err)) if n else float("nan"),
        "inlier_fraction": float(np.mean(abs_err < 2.0)) if n else float("nan"),
    }
    out["vertical_disparity_est_p95_arcmin"] = recon.get("vertical_disparity_est_arcmin_p95", 0.0)
    vd = design_vertical_disparity(geom_encoded, view, scene, geom_viewed=gv, head_offset=head_offset,
                                   reserved_rows=rr)
    out["vertical_disparity_design_p95_arcmin"] = vd["p95_arcmin"]
    centre = med(np.abs(x) < 5)
    sides = [med((x > -25) & (x < -10)), med((x > 10) & (x < 25))]
    out["hinge_kink_mm"] = centre - 0.5 * (sides[0] + sides[1])
    slopes = []
    for sel in ((x < -3) & (x > -40), (x > 3) & (x < 40)):
        if np.count_nonzero(sel) >= 50:
            inl = sel & (abs_err < 5.0)
            A = np.stack([x[inl], np.ones(inl.sum())], axis=1)
            slopes.append(np.linalg.lstsq(A, err[inl], rcond=None)[0][0] if inl.sum() >= 50 else np.nan)
        else:
            slopes.append(np.nan)
    out["hinge_slope_kink_deg"] = float(np.degrees(np.arctan(slopes[1]) - np.arctan(slopes[0])))
    if getattr(scene, "name", "") == "plaque" and n:
        e = 1.5
        inside = scene.object_mask(x, y)
        for ddx, ddy in ((e, 0), (-e, 0), (0, e), (0, -e)):
            inside &= scene.object_mask(x + ddx, y + ddy)
        out["plaque_n"] = int(inside.sum())
        if inside.sum() >= 100:
            px, py, pz = x[inside], y[inside], z[inside]
            coef, res_, _ = _plane_fit(px, py, pz)
            out["plaque_flatness_rms_mm"] = float(np.sqrt(np.mean(res_ ** 2)))
            out["plaque_tilt_deg"] = float(np.degrees(np.arccos(np.clip(_normal(coef)[2], -1, 1))))
            out["plaque_mean_err_mm"] = float(np.mean(err[inside]))
            lsel, rsel = px < -2, px > 2
            if lsel.sum() >= 50 and rsel.sum() >= 50:
                cl, _, _ = _plane_fit(px[lsel], py[lsel], pz[lsel])
                cr, _, _ = _plane_fit(px[rsel], py[rsel], pz[rsel])
                cosang = float(np.clip(_normal(cl) @ _normal(cr), -1, 1))
                out["plaque_kink_deg"] = float(np.degrees(np.arccos(cosang)))
            else:
                out["plaque_kink_deg"] = float("nan")
        else:
            out["plaque_flatness_rms_mm"] = float("nan")
            out["plaque_kink_deg"] = float("nan")
    th = PASS_THRESHOLDS
    checks = {
        "median_abs_err": out["median_abs_err_mm"] < th["median_abs_err_mm"],
        "hinge_kink": abs(out["hinge_kink_mm"]) < th["hinge_kink_mm"],
        "valid_fraction_linkable": out["valid_fraction_linkable"] > th["valid_fraction_linkable"],
    }
    if "plaque_kink_deg" in out:
        checks["plaque_kink"] = out["plaque_kink_deg"] < th["plaque_kink_deg"]
    out["checks"] = {k: bool(v) for k, v in checks.items()}
    out["pass"] = bool(all(checks.values()))
    out["timings"] = dict(recon.get("timings", {}), gt_s=t_gt, evaluate_s=time.perf_counter() - t0)
    if return_recon:
        out["recon"] = recon
    return out


# --------------------------------------------------------------------------
# tolerance sweep
# --------------------------------------------------------------------------
def default_tolerance_grid(full: bool = False):
    """Default offsets and hinge errors.

    full=False: one-axis-at-a-time offsets (dx in 0,5,10,20,30; dy in 10,20;
    dz in -50,50) = 9 offsets; hinge deltas (-5,-2,0,2,5). With full=True the
    offsets are the full product (45).
    """
    dxs, dys, dzs = (0, 5, 10, 20, 30), (0, 10, 20), (-50, 0, 50)
    hinge = (-5, -2, 0, 2, 5)
    if full:
        offs = [(float(a), float(b), float(c)) for a, b, c in itertools.product(dxs, dys, dzs)]
    else:
        offs = [(float(a), 0.0, 0.0) for a in dxs]
        offs += [(0.0, float(b), 0.0) for b in dys if b]
        offs += [(0.0, 0.0, float(c)) for c in dzs if c]
    return offs, hinge


_SWEEP_KEYS = ("vertical_disparity_design_p95_arcmin", "vertical_disparity_est_p95_arcmin", "valid_fraction", "valid_fraction_linkable", "median_abs_err_mm", "p95_abs_err_mm",
               "mean_signed_err_mm", "inlier_fraction", "hinge_kink_mm", "hinge_slope_kink_deg",
               "plaque_flatness_rms_mm", "plaque_kink_deg", "plaque_tilt_deg", "pass")


def tolerance_sweep(image, geom: FoldGeometry, view, scene, offsets: Iterable | None = None,
                    hinge_errors: Sequence[float] | None = None, *, pairs: str = "axes",
                    **eval_kw) -> list[dict]:
    """evaluate() across head offsets and hinge mismatches (viewed at hinge + delta).

    offsets/hinge_errors default to default_tolerance_grid(). pairs='axes'
    (default) evaluates every offset at delta 0 and every delta at offset 0;
    pairs='product' evaluates the full product. Extra kwargs go to evaluate
    (e.g. res=(1000, 690) for speed). Each row: dx, dy, dz, hinge_delta,
    hinge_viewed and the scalar metrics.
    """
    d_offs, d_hinge = default_tolerance_grid()
    offsets = d_offs if offsets is None else [tuple(map(float, o)) for o in offsets]
    hinge_errors = d_hinge if hinge_errors is None else list(hinge_errors)
    if pairs == "product":
        combos = list(itertools.product(offsets, hinge_errors))
    elif pairs == "axes":
        combos = [(o, 0.0) for o in offsets]
        combos += [((0.0, 0.0, 0.0), float(dh)) for dh in hinge_errors if dh != 0]
        if not combos or (0.0, 0.0, 0.0) not in offsets:
            combos.insert(0, ((0.0, 0.0, 0.0), 0.0))
    else:
        raise ValueError("pairs must be 'axes' or 'product'")
    rows = []
    for off, dh in combos:
        hv = min(180.0, geom.hinge_deg + dh)
        gv = FoldGeometry(geom.device, hv)
        r = evaluate(image, geom, view, scene, geom_viewed=gv, head_offset=off, **eval_kw)
        row = {"dx": off[0], "dy": off[1], "dz": off[2], "hinge_delta": float(dh), "hinge_viewed": hv}
        for k in _SWEEP_KEYS:
            if k in r:
                row[k] = r[k]
        row["time_s"] = r["timings"]["evaluate_s"]
        rows.append(row)
    return rows


# --------------------------------------------------------------------------
# comfort
# --------------------------------------------------------------------------
def comfort_map(geom: FoldGeometry, view, scene, *, stride: int | None = None,
                reserved_rows=None) -> np.ndarray:
    """Vergence-accommodation conflict per display pixel, dioptres.

    Cyclopean eye E; S = scene point along the ray E -> X(p) (own march);
    conflict = |1000/|S - E| - 1000/|X(p) - E|| (distances in mm). NaN in the
    reserved rows and where the ray misses the scene. With stride > 1 the march
    runs on a stride grid and is bilinearly interpolated (default: stride so that
    at most ~400k rays are marched).
    """
    dev = geom.device
    H, W = dev.height_px, dev.width_px
    rr = guide_band_rows(dev) if reserved_rows is None else int(reserved_rows)
    if stride is None:
        stride = max(1, int(math.ceil(math.sqrt(H * W / 400_000))))
    E = view.cyclopean_eye()
    jj = np.arange(rr, H, stride)
    if jj[-1] != H - 1:
        jj = np.append(jj, H - 1)
    ii = np.arange(0, W, stride)
    if ii[-1] != W - 1:
        ii = np.append(ii, W - 1)
    J, I = np.meshgrid(jj, ii, indexing="ij")
    X = geom.pixel_to_world(I, J)
    _, S, hit = march_heightfield(scene, E, X - E, step_mm=0.1, coarse=10)
    with np.errstate(invalid="ignore"):
        c = np.abs(1000.0 / np.linalg.norm(S - E, axis=-1) - 1000.0 / np.linalg.norm(X - E, axis=-1))
    c = np.where(hit, c, np.nan)
    out = np.full((H, W), np.nan)
    if stride == 1:
        out[rr:] = c
        return out
    from scipy.interpolate import RegularGridInterpolator
    interp = RegularGridInterpolator((jj.astype(float), ii.astype(float)), c, method="linear")
    gj, gi = np.meshgrid(np.arange(rr, H, dtype=float), np.arange(W, dtype=float), indexing="ij")
    out[rr:] = interp(np.stack([gj, gi], axis=-1))
    return out
