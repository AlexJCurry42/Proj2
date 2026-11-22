"""Naive comparison: a classic flat-screen SIRDS shown on the fold.

flat_sirds follows Thimbleby, Inglis & Witten (1994), "Displaying 3D images:
algorithms for single-image random-dot stereograms" (IEEE Computer 27(10)):
per row, each depth sample at cyclopean screen position x links the two screen
pixels x -/+ s/2 where both eyes see that surface point, with hidden-surface
removal and the symmetric "same[]" pointer juggling, then colours are
propagated from the right. Here it is set up for cross-eyed viewing of a FLAT
screen at view.distance_mm: a point at world height z in front of the screen
(eye-plane distance Z = D - z) has separation s = IPD * z / Z (crossed: the left
eye sees it through the right pixel of the pair). The same pairs work for
parallel viewing with z < 0 (s = IPD * |z| / Z); linking is symmetric.

The depth map is sampled exactly: for each pixel of the flat display grid the
cyclopean ray meets the scene at S and the map stores S.z. For a flat screen the
two eye projections of S are then exactly x -/+ s/2 on the same row.
"""
from __future__ import annotations

import numpy as np

from .config import guide_band_rows
from .geometry import FoldGeometry
from .verify import march_heightfield

__all__ = ["baseline_depth_map", "flat_sirds", "flat_separation_mm"]


def flat_separation_mm(z, view) -> np.ndarray:
    """Flat-screen pattern separation (mm) of a point at world height z."""
    z = np.asarray(z, dtype=np.float64)
    return view.ipd_mm * np.abs(z) / (view.distance_mm - z)


def baseline_depth_map(scene, device, view) -> np.ndarray:
    """World z of the scene seen by the cyclopean eye through each pixel of the
    FLAT display (hinge 180, plane z = 0). Shape (H, W); NaN if no hit."""
    g = FoldGeometry(device, 180.0)
    X = g.pixel_grid_world()
    E = view.cyclopean_eye()
    _, S, hit = march_heightfield(scene, E, X - E)
    return np.where(hit, S[..., 2], np.nan)


def _visibility(zmap, view, mm_per_px):
    """Thimbleby hidden-surface removal, vectorised over the whole image.

    For the surface point at cyclopean position x (depth z, eye distance Z), the
    line to the left eye passes over cyclopean positions x - t (t > 0) at depth
    z_line(t) = D - 1 / (1/Z + 2 t / (I D)); the right eye's line over x + t
    symmetrically. The point is hidden if the surface there is nearer than the line.
    """
    D, I = view.distance_mm, view.ipd_mm
    H, W = zmap.shape
    Z = D - zmap
    zmax = np.nanmax(zmap)
    vis = np.ones((H, W), dtype=bool)
    zfill = np.where(np.isfinite(zmap), zmap, -np.inf)
    # largest shift needed: where the line rises above the highest surface
    tmax_mm = (I * D / 2.0) * (1.0 / (D - zmax) - 1.0 / np.nanmax(Z))
    T = int(np.ceil(max(tmax_mm, 0.0) / mm_per_px)) + 1
    for t in range(1, min(T, W) + 1):
        zline = D - 1.0 / (1.0 / Z + 2.0 * t * mm_per_px / (I * D))
        left = np.full((H, W), -np.inf)
        left[:, t:] = zfill[:, :-t]
        right = np.full((H, W), -np.inf)
        right[:, :-t] = zfill[:, t:]
        vis &= ~((left > zline) | (right > zline))
    return vis


def flat_sirds(depth_z_map, device, view, *, seed: int = 0, dot_px: int = 5,
               reserved_rows=None, hidden_surface: bool = True) -> np.ndarray:
    """Classic flat-screen SIRDS (Thimbleby et al. 1994), cross-eyed.

    depth_z_map: (H, W) world z per pixel of the flat display (see
    baseline_depth_map); NaN = no constraint. Returns (H, W, 3) uint8; the
    reserved (guide) rows are left black, like the fold-aware encoder.
    Free pixels take the colour of a random-colour dot grid with dot_px cells.
    """
    zmap = np.asarray(depth_z_map, dtype=np.float64)
    H, W = device.height_px, device.width_px
    if zmap.shape != (H, W):
        raise ValueError(f"depth map shape {zmap.shape} != {(H, W)}")
    rr = guide_band_rows(device) if reserved_rows is None else int(reserved_rows)
    mmpp = device.mm_per_px
    sep = flat_separation_mm(np.where(np.isfinite(zmap), zmap, 0.0), view) / mmpp
    s_px = np.rint(sep).astype(np.int64)
    vis = _visibility(zmap, view, mmpp) if hidden_surface else np.ones((H, W), bool)
    vis &= np.isfinite(zmap) & (s_px > 0)
    rows_all = np.arange(H)
    vis[:rr] = False
    same = np.tile(np.arange(W, dtype=np.int64), (H, 1))
    for x in range(W):
        s = s_px[:, x]
        left = x - (s + (s & rows_all & 1)) // 2
        right = left + s
        ok = vis[:, x] & (left >= 0) & (right < W)
        r = rows_all[ok]
        lft = left[ok]
        rgt = right[ok]
        while r.size:
            k = same[r, lft]
            done = (k == lft) | (k == rgt)
            if done.any():
                same[r[done], lft[done]] = rgt[done]
            nd = ~done
            r, lft, rgt, k = r[nd], lft[nd], rgt[nd], k[nd]
            if not r.size:
                break
            lt = k < rgt
            ge = ~lt
            if ge.any():
                same[r[ge], lft[ge]] = rgt[ge]
            lft, rgt = np.where(lt, k, rgt), np.where(lt, rgt, k)
    rng = np.random.default_rng(seed)
    noise = rng.integers(0, 256, size=(-(-H // dot_px), -(-W // dot_px), 3), dtype=np.uint8)
    pix = np.zeros((H, W, 3), dtype=np.uint8)
    rdot = rows_all // dot_px
    for x in range(W - 1, -1, -1):
        sx = same[:, x]
        free = sx == x
        pix[:, x] = np.where(free[:, None], noise[rdot, x // dot_px], pix[rows_all, sx])
    pix[:rr] = 0
    return pix
