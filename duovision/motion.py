"""Animated hidden scenes: every named scene with one slow, looping motion.

``scene_frame(name, geom, view, phase)`` returns the HeightScene for a point of
the loop (phase in [0, 1), phase 0 == phase 1), ready for ``encode_stereogram``.
The motions follow the product spec's rules for animated stereograms:

* slow and continuous: sinusoidal glides and constant slow rotations, so depth
  never jumps and every depth change is eased over more than a second;
* the object stays inside the scene layout area (one pattern repeat away from
  the outer edges) for the whole loop;
* size, relief and depth mapping are fixed for the whole loop (``motion_plan``),
  so nothing pulses except where the motion says so;
* glides cross the hinge, which is what the fold-aware encoding is for, but they
  are short and slow: an autostereogram's pattern must shift on the far side of
  a moving object, so the faster an object travels, the more that side shimmers.

The stereogram's dots are seeded by chain-root position (see ``patterns``), so
rows the object never touches, and everything on the chain-root side of it, are
identical in every frame.
"""
from __future__ import annotations

import functools
import math
from typing import Callable, Dict, Tuple

import numpy as np
from scipy import ndimage

from .config import ViewProfile
from .geometry import FoldGeometry
from . import scene as S

MOTION_SECONDS = 6.0          # one loop
MOTION_FPS = 30
MOTION_NAMES = S.SCENE_NAMES
_PLAN_PHASES = 36             # phases sampled to fix size and depth mapping for the loop
_COARSE_MM = 0.5

# What each scene does over one loop (shown in the viewer and stored in the video's manifest entry).
MOTION_DESCRIPTION: Dict[str, str] = {
    "plaque": "glides across the hinge",
    "cube": "turns a quarter turn",
    "sphere": "glides across the hinge",
    "text": "bobs up and down",
    "torus": "circles around the hinge",
    "pyramid": "spins a quarter turn",
    "heart": "beats slowly",
    "pawn": "glides across the hinge",
}


def pose(name: str, phase: float) -> dict:
    """Motion parameters at `phase`: dx, dy (mm, object frame), spin (deg, in the
    picture plane), scale, and for the cube its turn about the vertical axis."""
    a = 2.0 * math.pi * (phase % 1.0)
    p = {"dx": 0.0, "dy": 0.0, "spin": 0.0, "scale": 1.0, "ry": 35.0}
    if name == "plaque":
        p["dx"] = 8.0 * math.sin(a)
    elif name == "cube":
        p["ry"] = 35.0 + 90.0 * (phase % 1.0)           # the cube repeats every quarter turn
    elif name == "sphere":
        p["dx"] = 9.0 * math.sin(a)
    elif name == "text":
        p["dy"] = 4.0 * math.sin(a)
    elif name == "torus":
        p["dx"] = 7.0 * math.cos(a)
        p["dy"] = 4.0 * math.sin(a)
    elif name == "pyramid":
        p["spin"] = 90.0 * (phase % 1.0)                 # square base: repeats every quarter turn
    elif name == "heart":
        p["scale"] = 1.0 + 0.06 * math.sin(2.0 * a)      # two slow beats per loop (3 s each)
    elif name == "pawn":
        p["dx"] = 8.0 * math.sin(a + math.pi / 2.0)
    else:
        raise ValueError(f"unknown scene {name!r}; choose from {MOTION_NAMES}")
    return p


def _moved(fn: Callable, p: dict, X, Y):
    """Evaluate a static shape fn(X, Y) -> (depth, mask, base) under the pose."""
    xs, ys = X - p["dx"], Y - p["dy"]
    if p["spin"]:
        c, s = math.cos(math.radians(p["spin"])), math.sin(math.radians(p["spin"]))
        xs, ys = c * xs + s * ys, -s * xs + c * ys
    k = p["scale"]
    depth, mask, base = fn(xs / k, ys / k)
    return depth * k, mask, (None if base is None else base * k)


@functools.lru_cache(maxsize=8)
def _text_relief(text: str, width: float, height: float, h: float):
    """Bevelled extruded-letter depth image (rows bottom -> top) and its mm origin, as in build_scene."""
    m = S._text_mask(text, width, height, h)[::-1]
    m = np.pad(m, ((10, 10), (10, 10)))
    dist = ndimage.distance_transform_edt(m, sampling=h)
    extrude, bevel_w, bevel_d = 8.0, 0.9, 1.2
    u = np.clip((dist - 0.5 * h) / bevel_w, 0.0, 1.0)
    depth = np.where(m, extrude - bevel_d * (1.0 - np.sqrt(1.0 - (1.0 - u) ** 2)), 0.0)
    ny, nx = m.shape
    return depth, m, -h * (nx - 1) / 2.0, -h * (ny - 1) / 2.0


def _shape_fn(name: str, geom: FoldGeometry, view: ViewProfile, lay: dict, slope: float, text: str) -> Callable:
    """fn(phase, X, Y) -> (depth, mask, base) in nominal object mm (x right, y up from the layout centre)."""
    if name == "cube":
        return lambda ph, X, Y: S._shape_cube(X, Y, ry=pose("cube", ph)["ry"])
    if name == "plaque":
        amp = abs(pose("plaque", 0.25)["dx"])
        hw = min(40.0, lay["x_half"] - amp)
        if slope > 0:   # side edges must clear the V background at the ends of the glide
            hw = min(hw, (S.PLAQUE_OFFSET_MM - S.BG_CLEARANCE_MM) / slope - amp)
        hw = max(hw, 8.0)
        hh = min(27.5, lay["y_half"])

        def plaque(X, Y):
            m = (np.abs(X) <= hw) & (np.abs(Y) <= hh)
            return m.astype(np.float64), m, 0.0
        return lambda ph, X, Y: _moved(plaque, pose("plaque", ph), X, Y)
    if name == "text":
        h = S.GRID_MM
        depth_img, mask_img, x0, y0 = _text_relief(text, min(125.0, 2.0 * lay["x_half"]), 2.0 * lay["y_half"], h)

        def letters(X, Y):
            fi, fj = (X - x0) / h, (Y - y0) / h
            d = ndimage.map_coordinates(depth_img, [fj, fi], order=1, mode="constant", cval=0.0)
            m = ndimage.map_coordinates(mask_img.astype(np.float32), [fj, fi], order=1, mode="constant", cval=0.0) > 0.5
            return np.where(m, d, 0.0), m, 0.0
        return lambda ph, X, Y: _moved(letters, pose("text", ph), X, Y)
    fn = S._SHAPES[name]
    return lambda ph, X, Y: _moved(fn, pose(name, ph), X, Y)


def _plan_key(name, geom, view, bg_vee, text):
    dev = geom.device
    return (name, dev.width_px, dev.height_px, round(dev.ppi, 6), round(geom.hinge_deg, 6), view, round(bg_vee, 6), text)


_PLANS: Dict[tuple, dict] = {}


def motion_plan(name: str, geom: FoldGeometry, view: ViewProfile, *, bg_vee: float = 0.25,
                text: str = "DuoVision") -> dict:
    """Everything that stays fixed over the loop: the fit scale, the depth range and
    the depth -> z mapping (z_lo, z_top), found by sampling the whole loop, plus the
    object grid that covers the motion. Cached per scene / fold / view."""
    if name not in MOTION_NAMES:
        raise ValueError(f"unknown scene {name!r}; choose from {MOTION_NAMES}")
    key = _plan_key(name, geom, view, bg_vee, text)
    if key in _PLANS:
        return _PLANS[key]
    lay = S.scene_layout(geom)
    x_half, yc, y_half = lay["x_half"], lay["y_centre"], lay["y_half"]
    if x_half <= 5 or y_half <= 5:
        raise ValueError("display too small for the scene layout")
    z_bg0 = view.z_background
    slope = bg_vee * geom.c / geom.s
    z_top = view.z_near if view.mode == "cross" else min(z_bg0 + S._relief_range(view), -S.PANEL_CLEARANCE_MM)
    fn = _shape_fn(name, geom, view, lay, slope, text)
    phases = np.arange(_PLAN_PHASES) / _PLAN_PHASES

    cs = _COARSE_MM
    cg = np.arange(-90.0, 90.0 + cs, cs)
    CX, CY = np.meshgrid(cg, cg)
    ex = 0.0
    ylo, yhi = np.inf, -np.inf
    dmin, dmax = np.inf, -np.inf
    samples = []
    for ph in phases:
        depth, mask, base = fn(float(ph), CX, CY)
        if not mask.any():
            raise RuntimeError(f"scene {name} is empty at phase {ph}")
        ex = max(ex, float(np.abs(CX[mask]).max()) + cs)
        ylo, yhi = min(ylo, float(CY[mask].min()) - cs), max(yhi, float(CY[mask].max()) + cs)
        dmin, dmax = min(dmin, float(depth[mask].min())), max(dmax, float(depth[mask].max()))
        samples.append((depth, mask, base))
    ey, y_mid = (yhi - ylo) / 2.0, (yhi + ylo) / 2.0
    sc = min(1.0, x_half / ex, y_half / ey)
    # Depth range of the whole loop: a shape's declared base depth, else the smallest depth seen.
    d_base = min(dmin if b is None else float(b) for (_d, _m, b) in samples)
    span = max(dmax - d_base, 1e-12)

    if name == "plaque":
        z_top = z_bg0 + S.PLAQUE_OFFSET_MM            # exactly planar, as in the still scene
    z_lo = z_bg0 + S.BG_CLEARANCE_MM
    for depth, mask, _b in samples:
        hn = np.clip((depth - d_base) / span, 0.0, 1.0)[mask]
        bm = z_bg0 + slope * np.abs(CX[mask] * sc)
        sel = hn < 0.999
        if sel.any():
            z_lo = max(z_lo, float(np.max((bm[sel] + S.BG_CLEARANCE_MM - hn[sel] * z_top) / (1.0 - hn[sel]))))
    if name == "text":
        z_lo = max(z_lo, z_top - S.TEXT_RELIEF_MM)
    z_lo = min(z_lo, z_top - S.MIN_RELIEF_MM)

    h = S.GRID_MM
    nxh = int(math.ceil((ex * sc + 1.0) / h))
    nyh = int(math.ceil((ey * sc + 1.0) / h))
    plan = {
        "name": name, "fn": fn, "sc": sc, "y_mid": y_mid, "d_base": d_base, "span": span,
        "z_lo": z_lo, "z_top": z_top, "z_bg0": z_bg0, "slope": slope, "layout": lay,
        "xs": h * np.arange(-nxh, nxh + 1), "ys": yc + h * np.arange(-nyh, nyh + 1),
        "extent_mm": (ex * sc, ey * sc), "bg_vee": bg_vee,
    }
    _PLANS[key] = plan
    return plan


def scene_frame(name: str, geom: FoldGeometry, view: ViewProfile, phase: float, *, bg_vee: float = 0.25,
                text: str = "DuoVision") -> S.HeightScene:
    """The hidden scene at `phase` of its loop (phase in [0, 1); 1.0 wraps to 0.0)."""
    plan = motion_plan(name, geom, view, bg_vee=bg_vee, text=text)
    phase = float(phase) % 1.0
    sc, lay = plan["sc"], plan["layout"]
    yc = lay["y_centre"]
    X, Y = np.meshgrid(plan["xs"], plan["ys"])
    depth, mask, _base = plan["fn"](phase, X / sc, (Y - yc) / sc + plan["y_mid"])
    if not mask.any():
        raise RuntimeError(f"scene {name} is empty at phase {phase}")
    hn = np.clip((depth - plan["d_base"]) / plan["span"], 0.0, 1.0)
    z_lo, z_top = plan["z_lo"], plan["z_top"]
    bgg = plan["z_bg0"] + plan["slope"] * np.abs(X)
    grid = np.where(mask, np.maximum(z_lo + hn * (z_top - z_lo), bgg), bgg)
    xin, yin = X[mask], Y[mask]
    meta = {"bg_vee": bg_vee, "layout": lay, "grid_mm": S.GRID_MM, "z_lo": z_lo, "z_top": z_top,
            "fit_scale": sc, "phase": phase, "motion": MOTION_DESCRIPTION[name], "pose": pose(name, phase),
            "object_extent": (float(np.abs(xin).max()), float(yin.min()), float(yin.max())),
            "size_mm": (float(xin.max() - xin.min()), float(yin.max() - yin.min()))}
    if name == "plaque":
        meta["plaque_z"] = z_top
    return S.HeightScene(name, z_bg0=plan["z_bg0"], bg_slope=plan["slope"], grid=grid, grid_x0=plan["xs"][0],
                         grid_y0=plan["ys"][0], grid_step=S.GRID_MM,
                         footprint=(geom.x_extent_mm, geom.height_mm / 2.0), mode=view.mode, meta=meta)
