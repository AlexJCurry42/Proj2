"""Fold-aware stereogram encoder.

For every display pixel p the LEFT eye's ray L -> X(p) is traced into the hidden
height-field scene (``duovision.scene.first_hit``) and meets it at S. If S is also
visible from the RIGHT eye R, the line R -> S lands on the folded display at the
partner pixel q, and the stereogram needs ``colour(p) == colour(q)``.

Cross mode (scene in front of the fold): q always lies left of p, so constraint
chains run leftward and can never form cycles. Parallel mode (scene behind the
fold): links point right.

Pixels are *free* (they seed chains with fresh random colour) when
  * they sit in the reserved guide band,
  * the left-eye ray misses the scene, or meets it on the wrong side of the display,
  * S is hidden from the right eye (``occluded``),
  * the partner lands off the display, in the guide band, or not strictly on the
    expected side (cross: ``partner_i <= i - 1``; parallel: ``partner_i >= i + 1``).

Pipeline: ``compute_partner_map`` -> ``resolve.round_partners`` ->
``resolve.resolve_roots`` -> ``patterns.colorize``; see ``encode_stereogram``.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .config import ViewProfile, guide_band_rows
from .geometry import FoldGeometry

VISIBILITY_TOL_MM = 0.05
"""S counts as visible from R if R's first hit toward S lies within this distance of S."""

_CHUNK_PIXELS = 1 << 16  # pixels traced per chunk (bounds peak memory of the tracer)


def _first_hit(scene, eye, targets):
    """Indirection to ``duovision.scene.first_hit`` (patched by unit tests)."""
    from . import scene as _scene_mod  # imported lazily: developed in parallel

    return _scene_mod.first_hit(scene, eye, targets)


@dataclass
class PartnerMap:
    partner_i: np.ndarray  # (H, W) float64, fractional column of the partner, NaN if free
    partner_j: np.ndarray  # (H, W) float64, fractional row of the partner, NaN if free
    linked: np.ndarray  # (H, W) bool
    hit_z: np.ndarray  # (H, W) float64, world z of S seen by the left eye (NaN on miss)
    occluded: np.ndarray  # (H, W) bool, S not visible from the right eye
    reserved_rows: int
    mode: str = "cross"


@dataclass
class EncodeResult:
    image: np.ndarray  # (H, W, 3) uint8, guide band black
    partner_map: PartnerMap
    pi: np.ndarray
    pj: np.ndarray
    roots: np.ndarray
    stats: dict = field(default_factory=dict)


def compute_partner_map(
    geom: FoldGeometry,
    view: ViewProfile,
    scene,
    *,
    reserved_rows: Optional[int] = None,
    chunk_pixels: int = _CHUNK_PIXELS,
) -> PartnerMap:
    """Trace every pixel from the left eye and find its right-eye partner.

    See the module docstring for the link rules. Work is chunked by rows so that
    peak memory stays bounded at full resolution.
    """
    dev = geom.device
    H, W = dev.height_px, dev.width_px
    rr = guide_band_rows(dev) if reserved_rows is None else int(reserved_rows)
    rr = max(0, min(rr, H))
    cross = view.mode == "cross"
    if view.mode not in ("cross", "parallel"):
        raise ValueError(f"unknown view mode {view.mode!r}")
    L, R = view.eyes()

    partner_i = np.full((H, W), np.nan, dtype=np.float64)
    partner_j = np.full((H, W), np.nan, dtype=np.float64)
    hit_z = np.full((H, W), np.nan, dtype=np.float64)
    linked = np.zeros((H, W), dtype=bool)
    occluded = np.zeros((H, W), dtype=bool)

    rows_per_chunk = max(1, int(chunk_pixels) // W)
    cols = np.arange(W, dtype=np.float64)
    for j0 in range(rr, H, rows_per_chunk):
        j1 = min(H, j0 + rows_per_chunk)
        jj, ii = np.meshgrid(np.arange(j0, j1, dtype=np.float64), cols, indexing="ij")
        ii = ii.ravel()
        jj = jj.ravel()
        X = geom.pixel_to_world(ii, jj)  # (n, 3)

        # 1. left eye -> scene
        S, hit = _first_hit(scene, L, X)
        S = np.asarray(S, dtype=np.float64).reshape(-1, 3)
        hit = np.asarray(hit, dtype=bool).ravel() & np.isfinite(S).all(axis=1)
        # S must lie on the correct side of the display along the left-eye ray:
        # cross: between the eye and the pixel; parallel: beyond the pixel.
        d = X - L
        t = np.einsum("ij,ij->i", S - L, d) / np.einsum("ij,ij->i", d, d)
        with np.errstate(invalid="ignore"):
            hit &= (t < 1.0) if cross else (t > 1.0)

        zc = np.where(hit, S[:, 2], np.nan)
        hidx = np.nonzero(hit)[0]
        Sh = S[hidx]

        # 2. visibility of S from the right eye
        S2, hit2 = _first_hit(scene, R, Sh)
        S2 = np.asarray(S2, dtype=np.float64).reshape(-1, 3)
        hit2 = np.asarray(hit2, dtype=bool).ravel()
        with np.errstate(invalid="ignore"):
            vis = hit2 & (np.linalg.norm(S2 - Sh, axis=1) <= VISIBILITY_TOL_MM)
        occ_c = np.zeros(ii.shape, dtype=bool)
        occ_c[hidx[~vis]] = True

        # 3. partner: where R -> S lands on the display
        vidx = hidx[vis]
        qi, qj, qv = geom.project_from_eye(R, S[vidx])
        qi = np.asarray(qi, dtype=np.float64).ravel()
        qj = np.asarray(qj, dtype=np.float64).ravel()
        qv = np.asarray(qv, dtype=bool).ravel()
        with np.errstate(invalid="ignore"):
            ok = qv & (qi >= 0.0) & (qi <= W - 1) & (qj >= rr) & (qj <= H - 1)
            if cross:
                ok &= qi <= ii[vidx] - 1.0
            else:
                ok &= qi >= ii[vidx] + 1.0
        lidx = vidx[ok]

        pic = np.full(ii.shape, np.nan)
        pjc = np.full(ii.shape, np.nan)
        pic[lidx] = qi[ok]
        pjc[lidx] = qj[ok]
        lk = np.zeros(ii.shape, dtype=bool)
        lk[lidx] = True

        shp = (j1 - j0, W)
        partner_i[j0:j1] = pic.reshape(shp)
        partner_j[j0:j1] = pjc.reshape(shp)
        hit_z[j0:j1] = zc.reshape(shp)
        linked[j0:j1] = lk.reshape(shp)
        occluded[j0:j1] = occ_c.reshape(shp)

    return PartnerMap(
        partner_i=partner_i,
        partner_j=partner_j,
        linked=linked,
        hit_z=hit_z,
        occluded=occluded,
        reserved_rows=rr,
        mode=view.mode,
    )


def partner_map_stats(pm: PartnerMap, geom: FoldGeometry) -> dict:
    """Coverage and link-geometry statistics over the encodable (non-reserved) area."""
    H, W = pm.linked.shape
    rr = pm.reserved_rows
    area = max(1, (H - rr) * W)
    lk = pm.linked
    stats = {
        "encodable_pixels": (H - rr) * W,
        "linked_fraction": float(lk[rr:].sum()) / area,
        "occluded_fraction": float(pm.occluded[rr:].sum()) / area,
        "miss_fraction": float(np.isnan(pm.hit_z[rr:]).sum()) / area,
    }
    stats["free_fraction"] = 1.0 - stats["linked_fraction"]
    if lk.any():
        jj, ii = np.nonzero(lk)
        sep = np.abs(ii - pm.partner_i[lk])
        voff = np.abs(pm.partner_j[lk] - jj)
        mm = geom.mm_per_px
        stats.update(
            sep_px_min=float(sep.min()),
            sep_px_median=float(np.median(sep)),
            sep_px_max=float(sep.max()),
            sep_mm_min=float(sep.min() * mm),
            sep_mm_median=float(np.median(sep) * mm),
            sep_mm_max=float(sep.max() * mm),
            vertical_offset_px_max=float(voff.max()),
        )
    else:
        stats.update(
            sep_px_min=float("nan"), sep_px_median=float("nan"), sep_px_max=float("nan"),
            sep_mm_min=float("nan"), sep_mm_median=float("nan"), sep_mm_max=float("nan"),
            vertical_offset_px_max=0.0,
        )
    return stats


def encode_stereogram(
    geom: FoldGeometry,
    view: ViewProfile,
    scene,
    style: str = "color",
    *,
    seed: int = 0,
    dot_px: int = 5,
    texture: str = "marble",
    reserved_rows: Optional[int] = None,
) -> EncodeResult:
    """Full pipeline: partner map -> randomized rounding -> roots -> colours.

    The guide band (reserved rows) is left black for ``guides.draw_guide_band``.
    ``texture`` selects the procedural texture when ``style == "texture"``.
    """
    from .patterns import colorize
    from .resolve import resolve_roots, round_partners

    t0 = time.perf_counter()
    pm = compute_partner_map(geom, view, scene, reserved_rows=reserved_rows)
    t1 = time.perf_counter()
    rng = np.random.default_rng(seed)
    pi, pj = round_partners(pm, rng)
    t2 = time.perf_counter()
    roots = resolve_roots(pi, pj)
    t3 = time.perf_counter()
    image = colorize(roots, geom.device, style, seed=seed, dot_px=dot_px, texture=texture)
    image[: pm.reserved_rows] = 0
    t4 = time.perf_counter()

    stats = partner_map_stats(pm, geom)
    stats.update(
        hinge_deg=geom.hinge_deg,
        mode=view.mode,
        style=style,
        scene=getattr(scene, "name", type(scene).__name__),
        width_px=geom.device.width_px,
        height_px=geom.device.height_px,
        reserved_rows=pm.reserved_rows,
        linked_after_rounding_fraction=float((pi[pm.reserved_rows:] >= 0).sum())
        / max(1, pi[pm.reserved_rows:].size),
        n_chains=int((roots.ravel() == np.arange(roots.size)).sum()),
        time_partner_map_s=t1 - t0,
        time_round_s=t2 - t1,
        time_resolve_s=t3 - t2,
        time_colorize_s=t4 - t3,
        time_total_s=t4 - t0,
    )
    return EncodeResult(image=image, partner_map=pm, pi=pi, pj=pj, roots=roots, stats=stats)
