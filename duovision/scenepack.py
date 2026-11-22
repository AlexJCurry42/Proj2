"""Scene pack: everything the viewer's real-time renderer needs, exported by the reference engine.

The WebGL renderer (``viewer/engine.js``) runs the same pipeline as ``encoder`` on the GPU,
live, at the display's own pixel grid. It does not re-implement the scenes: this module
exports them.

* ``scenes.json``: the view profile, the device, and for every scene the looping motion
  sampled per frame (``motion.pose``) and, per hinge angle, the constants of
  ``motion.motion_plan`` (fit scale, depth range, z_lo / z_top, background V, bounds).
* ``scene_<name>.bin``: the scene's *template*: its nominal depth map and mask on a regular
  grid in nominal object millimetres, before fitting and before the motion. The motion is a
  rigid move / in-plane spin / uniform scale of that template, so one template serves the
  whole loop. The cube (which turns out of the picture plane) and the plaque (an exact
  rectangle) are analytic instead.

``pack_scene`` rebuilds the HeightScene of any frame from the pack alone, with exactly the
arithmetic the shader uses. It is the reference the GPU output is verified against, and
tests compare it with ``motion.scene_frame``.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np

from . import __version__
from . import scene as S
from .config import DEFAULT_DEVICE, GUIDE_BAND_FRACTION, HINGE_SWEEP_DEG, ViewProfile
from .geometry import FoldGeometry
from . import motion

PACK_FORMAT = "duovision-scenes"
PACK_VERSION = 1
TEMPLATE_STEP_MM = 0.2        # template grid for the smooth shapes
TEXT_STEP_MM = 0.1            # letters need the bake resolution
TEXT_NOMINAL_WIDTH_MM = 125.0
POSE_SAMPLES = 360            # motion sampled this many times per loop (the renderer interpolates)
PACK_FILE = "scenes.json"


def template_file(name: str) -> str:
    return f"scene_{name}.bin"


def scene_kind(name: str) -> str:
    return {"cube": "cube", "plaque": "rect"}.get(name, "template")


def _template(name: str, text: str):
    """(depth float32 (ny, nx), mask bool, x0, y0, step) of a scene's nominal shape; rows bottom -> top."""
    if name == "text":
        h = TEXT_STEP_MM
        depth, mask, x0, y0 = motion._text_relief(text, TEXT_NOMINAL_WIDTH_MM, 60.0, h)
        return depth.astype(np.float32), mask.astype(bool), float(x0), float(y0), h
    h = TEMPLATE_STEP_MM
    g = np.arange(-90.0, 90.0 + h / 2, h)
    X, Y = np.meshgrid(g, g)
    depth, mask, _base = S._SHAPES[name](X, Y)
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    r0, r1 = max(rows[0] - 2, 0), min(rows[-1] + 3, mask.shape[0])
    c0, c1 = max(cols[0] - 2, 0), min(cols[-1] + 3, mask.shape[1])
    return (np.where(mask, depth, 0.0)[r0:r1, c0:c1].astype(np.float32), mask[r0:r1, c0:c1],
            float(g[c0]), float(g[r0]), h)


def write_template(path, depth: np.ndarray, mask: np.ndarray) -> float:
    """depth as uint16 (0..65535 = 0..max), then mask as uint8, row-major, rows bottom -> top."""
    dmax = float(depth.max()) or 1.0
    q = np.round(np.clip(depth / dmax, 0.0, 1.0) * 65535.0).astype("<u2")
    Path(path).write_bytes(q.tobytes() + mask.astype(np.uint8).tobytes())
    return dmax


def read_template(path, nx: int, ny: int, dmax: float):
    raw = Path(path).read_bytes()
    n = nx * ny
    depth = np.frombuffer(raw[:2 * n], dtype="<u2").reshape(ny, nx).astype(np.float64) * (dmax / 65535.0)
    mask = np.frombuffer(raw[2 * n:3 * n], dtype=np.uint8).reshape(ny, nx).astype(np.float64)
    return depth, mask


def _view_dict(view: ViewProfile) -> dict:
    return {"distance_mm": view.distance_mm, "ipd_mm": view.ipd_mm, "eye_height_mm": view.eye_height_mm,
            "mode": view.mode, "sep_bg_mm": view.sep_bg_mm, "sep_near_mm": view.sep_near_mm,
            "z_background": view.z_background, "z_near": view.z_near}


def export_scene_pack(out_dir, view: ViewProfile | None = None, *, hinges: Iterable[float] | None = None,
                      scenes: Iterable[str] | None = None, bg_vee: float = 0.25, text: str = "DuoVision") -> dict:
    """Write scenes.json and the template files into out_dir; returns the pack dict."""
    view = view or ViewProfile()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    hinges = [float(h) for h in (hinges or HINGE_SWEEP_DEG)]
    names = list(scenes) if scenes else list(S.SCENE_NAMES)
    dev = DEFAULT_DEVICE
    phases = np.arange(POSE_SAMPLES) / POSE_SAMPLES
    pack = {
        "format": PACK_FORMAT, "version": PACK_VERSION, "generator": f"DuoVision {__version__}",
        "device": {"width_px": dev.width_px, "height_px": dev.height_px, "ppi": dev.ppi,
                   "width_mm": dev.width_px * dev.mm_per_px, "height_mm": dev.height_px * dev.mm_per_px},
        "view": _view_dict(view), "band_fraction": GUIDE_BAND_FRACTION, "bg_vee": bg_vee,
        "visibility_tol_mm": 0.05, "march_step_mm": 0.5,
        "loop_seconds": motion.MOTION_SECONDS, "pose_samples": POSE_SAMPLES, "hinges": hinges, "scenes": {},
    }
    for name in names:
        kind = scene_kind(name)
        poses = [motion.pose(name, float(p)) for p in phases]
        entry = {"kind": kind, "motion": motion.MOTION_DESCRIPTION[name],
                 "pose": {k: [round(float(p[k]), 6) for p in poses] for k in ("dx", "dy", "spin", "scale", "ry")},
                 "plans": {}}
        if kind == "template":
            depth, mask, x0, y0, step = _template(name, text)
            dmax = write_template(out / template_file(name), depth, mask)
            entry["template"] = {"file": template_file(name), "nx": int(depth.shape[1]), "ny": int(depth.shape[0]),
                                 "x0": x0, "y0": y0, "step": step, "depth_max": dmax}
        elif kind == "cube":
            entry["cube"] = {"edge": 45.0, "rx": 25.0}
        for hdeg in hinges:
            geom = FoldGeometry(dev, hdeg)
            plan = motion.motion_plan(name, geom, view, bg_vee=bg_vee, text=text)
            lay = plan["layout"]
            p = {"sc": plan["sc"], "y_mid": plan["y_mid"], "d_base": plan["d_base"], "span": plan["span"],
                 "z_lo": plan["z_lo"], "z_top": plan["z_top"], "z_bg0": plan["z_bg0"], "slope": plan["slope"],
                 "yc": lay["y_centre"], "x_half": lay["x_half"], "y_half": lay["y_half"],
                 "box": [float(plan["xs"][0]), float(plan["xs"][-1]), float(plan["ys"][0]), float(plan["ys"][-1])],
                 "tscale": 1.0}
            if kind == "rect":
                hw, hh = _plaque_half_size(lay, plan["slope"])
                p["rect"] = [hw, hh]
            if name == "text":      # the letters are fitted to the layout width; the template is the nominal width
                p["tscale"] = min(TEXT_NOMINAL_WIDTH_MM, 2.0 * lay["x_half"]) / TEXT_NOMINAL_WIDTH_MM
            entry["plans"][f"{hdeg:g}"] = {k: (v if isinstance(v, list) else float(v)) for k, v in p.items()}
        pack["scenes"][name] = entry
    (out / PACK_FILE).write_text(json.dumps(pack, separators=(",", ":")))
    return pack


def _plaque_half_size(lay: dict, slope: float):
    amp = abs(motion.pose("plaque", 0.25)["dx"])
    hw = min(40.0, lay["x_half"] - amp)
    if slope > 0:
        hw = min(hw, (S.PLAQUE_OFFSET_MM - S.BG_CLEARANCE_MM) / slope - amp)
    return max(hw, 8.0), min(27.5, lay["y_half"])


# ----------------------------------------------------------------- reference
def _lerp_pose(entry: dict, phase: float) -> dict:
    """Pose at `phase` by linear interpolation of the sampled loop (as the renderer does)."""
    n = len(entry["pose"]["dx"])
    f = (phase % 1.0) * n
    k0 = int(math.floor(f)) % n
    k1 = (k0 + 1) % n
    t = f - math.floor(f)
    out = {}
    for key, arr in entry["pose"].items():
        a, b = arr[k0], arr[k1]
        if key in ("spin", "ry") and b < a:           # these wrap at the end of the loop (a quarter turn)
            b += 90.0
        out[key] = a + (b - a) * t
    return out


def _bilinear(img: np.ndarray, fi: np.ndarray, fj: np.ndarray) -> np.ndarray:
    """Bilinear sample of img (ny, nx) at fractional node indices; 0 outside."""
    ny, nx = img.shape
    inside = (fi >= 0) & (fi <= nx - 1) & (fj >= 0) & (fj <= ny - 1)
    fi = np.clip(fi, 0.0, nx - 1.0)
    fj = np.clip(fj, 0.0, ny - 1.0)
    i0 = np.minimum(fi.astype(np.intp), nx - 2)
    j0 = np.minimum(fj.astype(np.intp), ny - 2)
    tx, ty = fi - i0, fj - j0
    a = img[j0, i0] * (1 - tx) + img[j0, i0 + 1] * tx
    b = img[j0 + 1, i0] * (1 - tx) + img[j0 + 1, i0 + 1] * tx
    return np.where(inside, a * (1 - ty) + b * ty, 0.0)


def pack_surface(pack: dict, pack_dir, name: str, hinge: float, phase: float):
    """z(x, y) of a frame, from the pack alone, with the shader's arithmetic.

    Returns a function z_of(x, y) (world mm, numpy arrays) and the plan dict."""
    entry = pack["scenes"][name]
    plan = entry["plans"][f"{float(hinge):g}"]
    ps = _lerp_pose(entry, phase)
    kind = entry["kind"]
    tpl = None
    if kind == "template":
        t = entry["template"]
        depth, mask = read_template(Path(pack_dir) / t["file"], t["nx"], t["ny"], t["depth_max"])
        tpl = (depth, mask, t)

    def z_of(x, y):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        bg = plan["z_bg0"] + plan["slope"] * np.abs(x)
        ox = x / plan["sc"] - ps["dx"]
        oy = (y - plan["yc"]) / plan["sc"] + plan["y_mid"] - ps["dy"]
        if ps["spin"]:
            c, s = math.cos(math.radians(ps["spin"])), math.sin(math.radians(ps["spin"]))
            ox, oy = c * ox + s * oy, -s * ox + c * oy
        k = ps["scale"] * plan["tscale"]
        ox, oy = ox / k, oy / k
        if kind == "cube":
            depth, m, _ = S._shape_cube(ox, oy, edge=entry["cube"]["edge"], ry=ps["ry"], rx=entry["cube"]["rx"])
            m = m.astype(np.float64)
        elif kind == "rect":
            hw, hh = plan["rect"]
            m = ((np.abs(ox) <= hw) & (np.abs(oy) <= hh)).astype(np.float64)
            depth = m
        else:
            dimg, mimg, t = tpl
            fi, fj = (ox - t["x0"]) / t["step"], (oy - t["y0"]) / t["step"]
            depth = _bilinear(dimg, fi, fj)
            m = _bilinear(mimg, fi, fj)
        depth = depth * ps["scale"]             # the motion's own scale; the text fit (tscale) only resizes the outline
        hn = np.clip((depth - plan["d_base"]) / plan["span"], 0.0, 1.0)
        zobj = np.maximum(plan["z_lo"] + hn * (plan["z_top"] - plan["z_lo"]), bg)
        return np.where(m > 0.5, zobj, bg)

    return z_of, plan


def pack_scene(pack: dict, pack_dir, name: str, geom: FoldGeometry, view: ViewProfile, phase: float) -> S.HeightScene:
    """HeightScene of a frame rebuilt from the pack (baked on the usual 0.1 mm grid)."""
    z_of, plan = pack_surface(pack, pack_dir, name, geom.hinge_deg, phase)
    h = S.GRID_MM
    x0, x1, y0, y1 = plan["box"]
    xs = x0 + h * np.arange(int(round((x1 - x0) / h)) + 1)
    ys = y0 + h * np.arange(int(round((y1 - y0) / h)) + 1)
    X, Y = np.meshgrid(xs, ys)
    return S.HeightScene(name, z_bg0=plan["z_bg0"], bg_slope=plan["slope"], grid=z_of(X, Y), grid_x0=xs[0],
                         grid_y0=ys[0], grid_step=h, footprint=(geom.x_extent_mm, geom.height_mm / 2.0),
                         mode=view.mode, meta={"phase": phase % 1.0, "z_lo": plan["z_lo"], "z_top": plan["z_top"]})
