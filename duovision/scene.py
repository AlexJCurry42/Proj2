"""Hidden 3D scenes for DuoVision, as height fields in absolute world z.

A scene is the front-most surface seen along -z (the classic SIRDS depth-map
assumption, in 3D world space): a shallow V-shaped background plus one object
centred on the hinge axis.

Background (both modes, the contract formula):
    z_bg(x) = view.z_background + bg_vee * |x| * c / s
In cross mode view.z_background is in front of the hinge (57.69 mm by default);
in parallel mode it is negative (behind the display), so the same formula
mirrors the scene behind the panels with ample clearance.

Objects are real 3D shapes (analytic surfaces / implicit functions / an extruded
font raster). Their orthographic front-surface depth d(x, y) is baked once in
`build_scene` onto a fine grid (GRID_MM = 0.1 mm), then compressed linearly
(bas-relief) into [z_lo, z_top]:
    z_obj = z_lo + (d - d_base) / (d_max - d_base) * (z_top - z_lo)
Cross mode: z_top = view.z_near. z_lo is the lowest value that keeps every
object point at least BG_CLEARANCE_MM (2 mm) above the local background, so the
whole silhouette stands proud of the V (and never below z_background + 2).
A single affine map per object keeps planar faces planar and avoids a crease
at the hinge axis. The baked grid stores max(background, object) and `z_of`
interpolates it bilinearly (the grid has a node column at x = 0, so the V
background is reproduced exactly); outside the grid the analytic background is
returned.

`first_hit` intersects rays with the height field: the background analytically
(exact), the object grid by marching in z (step_mm) inside the object's 3D
bounding box, then bisection (refine_iters) and a final secant step.
"""
from __future__ import annotations

import functools
import glob
import math
import os
from typing import Callable, Optional, Tuple

import numpy as np
from scipy import ndimage
from scipy.interpolate import PchipInterpolator

from .config import ViewProfile, guide_band_rows
from .geometry import FoldGeometry

SCENE_NAMES = ("plaque", "cube", "sphere", "text", "torus", "pyramid", "heart", "pawn")

GRID_MM = 0.1               # bake resolution of object depth maps
EDGE_MARGIN_MM = 15.0       # objects stay this far in from the outer display edges (x)
BAND_MARGIN_MM = 10.0       # vertical margin inside the area below the guide band
BG_CLEARANCE_MM = 2.0       # every object point is at least this far in front of the local background
PANEL_CLEARANCE_MM = 3.0    # surfaces keep this distance from the display
PLAQUE_OFFSET_MM = 12.0     # plaque z = z_background + 12
TEXT_RELIEF_MM = 12.0       # extruded-letter relief (base of the extrusion to letter face)
MIN_RELIEF_MM = 3.0
_MASK_EPS = 1e-6            # z_of - z_bg above this counts as object
_CHUNK = 1 << 18            # rays per chunk in first_hit


# ---------------------------------------------------------------- the scene
class HeightScene:
    """Height field: world z of the front-most hidden surface at world (x, y).

    Attributes (contract): name, z_min, z_max (min/max of z_of over the display
    footprint), z_of(x, y), object_mask(x, y).
    Extra read-only info: mode, z_bg0, bg_slope, z_lo/z_top (object relief map),
    object_extent (x_half, y_bottom, y_top) in mm, meta (dict).
    """

    def __init__(
        self,
        name: str,
        *,
        z_bg0: float,
        bg_slope: float,
        grid: Optional[np.ndarray],
        grid_x0: float,
        grid_y0: float,
        grid_step: float,
        footprint: Tuple[float, float],
        mode: str = "cross",
        meta: Optional[dict] = None,
    ):
        self.name = name
        self.mode = mode
        self.z_bg0 = float(z_bg0)
        self.bg_slope = float(bg_slope)
        self.meta = dict(meta or {})
        self._grid = None if grid is None else np.ascontiguousarray(grid, dtype=np.float64)
        self._gx0 = float(grid_x0)
        self._gy0 = float(grid_y0)
        self._h = float(grid_step)
        x_ext, h_half = footprint
        bg_lo = min(self.background(0.0), self.background(x_ext))
        bg_hi = max(self.background(0.0), self.background(x_ext))
        if self._grid is not None:
            self._ny, self._nx = self._grid.shape
            self._flat = self._grid.ravel()
            self._gzmin = float(self._grid.min())
            self._gzmax = float(self._grid.max())
            self.z_min = float(min(bg_lo, self._gzmin))
            self.z_max = float(max(bg_hi, self._gzmax))
        else:
            self._ny = self._nx = 0
            self.z_min = float(bg_lo)
            self.z_max = float(bg_hi)

    # -------------------------------------------------------------- queries
    def background(self, x) -> np.ndarray:
        """Analytic V background z_bg(x)."""
        return self.z_bg0 + self.bg_slope * np.abs(np.asarray(x, dtype=np.float64))

    @property
    def grid_bounds(self) -> Optional[Tuple[float, float, float, float]]:
        """(x0, x1, y0, y1) of the baked object grid, None for background only."""
        if self._grid is None:
            return None
        return (self._gx0, self._gx0 + (self._nx - 1) * self._h,
                self._gy0, self._gy0 + (self._ny - 1) * self._h)

    def _interp(self, fi: np.ndarray, fj: np.ndarray) -> np.ndarray:
        """Bilinear interpolation of the grid at fractional node indices (clamped)."""
        nx, ny = self._nx, self._ny
        fi = np.clip(fi, 0.0, nx - 1.0)
        fj = np.clip(fj, 0.0, ny - 1.0)
        i0 = np.minimum(fi.astype(np.intp), nx - 2)
        j0 = np.minimum(fj.astype(np.intp), ny - 2)
        tx = fi - i0
        ty = fj - j0
        k = j0 * nx + i0
        g = self._flat
        z00 = g[k]
        z10 = g[k + 1]
        z01 = g[k + nx]
        z11 = g[k + nx + 1]
        # lerp form a + t*(b - a) is exact when a == b (keeps the plaque planar)
        lo = z00 + tx * (z10 - z00)
        hi = z01 + tx * (z11 - z01)
        return lo + ty * (hi - lo)

    def _tile_bounds(self, tile_nodes: int = 10):
        """Upper bounds of the grid for ray-marching skips (cached).

        tiles[tj, ti] = max of the grid nodes in the 3 x 3 block of
        tile_nodes x tile_nodes tiles centred on tile (ti, tj)."""
        if getattr(self, "_tiles", None) is None:
            ny, nx = self._grid.shape
            ty, tx = -(-ny // tile_nodes), -(-nx // tile_nodes)
            pad = np.full((ty * tile_nodes, tx * tile_nodes), -np.inf)
            pad[:ny, :nx] = self._grid
            tmax = pad.reshape(ty, tile_nodes, tx, tile_nodes).max(axis=(1, 3))
            tiles = ndimage.maximum_filter(tmax, size=3, mode="nearest")
            self._tiles = (np.ascontiguousarray(tiles), tile_nodes)
        return self._tiles

    def _grid_z(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self._interp((x - self._gx0) / self._h, (y - self._gy0) / self._h)

    def z_of(self, x, y) -> np.ndarray:
        """World z of the hidden surface at world (x, y); any (broadcastable) shape."""
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        x, y = np.broadcast_arrays(x, y)
        shape = x.shape
        x = x.reshape(-1)
        y = y.reshape(-1)
        out = self.background(x)
        if self._grid is None:
            return out.reshape(shape)
        fi = (x - self._gx0) / self._h
        fj = (y - self._gy0) / self._h
        inside = (fi >= 0) & (fi <= self._nx - 1) & (fj >= 0) & (fj <= self._ny - 1)
        if inside.any():
            zi = self._interp(fi[inside], fj[inside])
            b = out[inside]
            out[inside] = np.where(zi > b + _MASK_EPS, zi, b)
        return out.reshape(shape)

    def object_mask(self, x, y) -> np.ndarray:
        """True where the surface is an object (stands above the background)."""
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        x, y = np.broadcast_arrays(x, y)
        return self.z_of(x, y) > self.background(x)

    def __repr__(self) -> str:
        return (f"HeightScene({self.name!r}, mode={self.mode!r}, z_min={self.z_min:.3f}, "
                f"z_max={self.z_max:.3f})")


# --------------------------------------------------------------- layout
def scene_layout(geom: FoldGeometry) -> dict:
    """Where objects may go: |x| <= x_half, y in [y_bottom, y_top] (world mm)."""
    dev = geom.device
    band_mm = guide_band_rows(dev) * dev.mm_per_px
    y_top = geom.height_mm / 2.0 - band_mm - BAND_MARGIN_MM
    y_bottom = -geom.height_mm / 2.0 + BAND_MARGIN_MM
    return {
        "x_half": geom.x_extent_mm - EDGE_MARGIN_MM,
        "y_top": y_top,
        "y_bottom": y_bottom,
        "y_centre": (y_top + y_bottom) / 2.0,
        "y_half": (y_top - y_bottom) / 2.0,
        "band_mm": band_mm,
    }


# ------------------------------------------------------------- 3D shapes
# Each shape: fn(X, Y) in its own nominal mm frame (x right, y up, depth toward
# the viewer) -> (depth, mask, base). depth = orthographic front-surface depth,
# base = depth that maps to z_lo (None -> min depth over the mask).

def _rot_x(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def _rot_y(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def _shape_cube(X, Y, edge=45.0, ry=35.0, rx=25.0):
    """Cube turned 35 deg about y, then tipped 25 deg about x: front, side and top faces show."""
    R = _rot_x(rx) @ _rot_y(ry)      # columns = world directions of the body axes
    a = edge / 2.0
    zlo = np.full(X.shape, -np.inf)
    zhi = np.full(X.shape, np.inf)
    for k in range(3):
        n = R[:, k]
        c0 = n[0] * X + n[1] * Y
        t1 = (-a - c0) / n[2]
        t2 = (a - c0) / n[2]
        zlo = np.maximum(zlo, np.minimum(t1, t2))
        zhi = np.minimum(zhi, np.maximum(t1, t2))
    mask = zhi > zlo
    return np.where(mask, zhi, 0.0), mask, None


def _shape_sphere(X, Y, radius=32.0):
    q = radius * radius - X * X - Y * Y
    mask = q > 0
    return np.sqrt(np.maximum(q, 0.0)), mask, 0.0


def _shape_torus(X, Y, major=28.0, minor=10.0):
    rho = np.hypot(X, Y)
    q = minor * minor - (rho - major) ** 2
    mask = q > 0
    return np.sqrt(np.maximum(q, 0.0)), mask, 0.0


def _shape_pyramid(X, Y, base=62.0, height=42.0):
    m = np.maximum(np.abs(X), np.abs(Y)) / (base / 2.0)
    mask = m < 1.0
    return np.where(mask, height * (1.0 - m), 0.0), mask, 0.0


def _shape_heart(X, Y, size=33.0):
    """Taubin's heart surface (x^2 + 9/4 v^2 + w^2 - 1)^3 - x^2 w^3 - 9/80 v^2 w^3 = 0,
    w vertical, v depth; front depth = largest v inside, scaled by `size` mm."""
    u = X / size
    w = Y / size
    A = u * u + w * w - 1.0
    w3 = w ** 3
    u2 = u * u

    def G(S):
        return (A + 2.25 * S) ** 3 - w3 * (u2 + 0.1125 * S)

    mask = G(0.0) < 0
    S = np.zeros(X.shape)
    if mask.any():
        a = A[mask]
        ww = w3[mask]
        uu = u2[mask]

        def g(s):
            return (a + 2.25 * s) ** 3 - ww * (uu + 0.1125 * s)

        # largest root in S = v^2: scan down from S_hi (outside) to the first inside sample
        s_hi_val = 2.0
        n = 64
        hi = np.full(a.shape, s_hi_val)
        lo = np.zeros(a.shape)
        found = np.zeros(a.shape, bool)
        for k in range(n - 1, -1, -1):
            s = s_hi_val * k / n
            inside = (g(s) <= 0) & ~found
            lo = np.where(inside, s, lo)
            hi = np.where(inside, s + s_hi_val / n, hi)
            found |= inside
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            ins = g(mid) <= 0
            lo = np.where(ins, mid, lo)
            hi = np.where(ins, hi, mid)
        S[mask] = 0.5 * (lo + hi)
    return size * np.sqrt(S), mask, 0.0


@functools.lru_cache(maxsize=1)
def _pawn_profile():
    """Radius r(y) of a Staunton-style pawn, y = 0 at the bottom of the base (mm)."""
    pts = np.array([
        # base: foot, rounded bead, cove
        (0.0, 21.0), (2.5, 21.5), (4.5, 21.0), (6.5, 19.2), (8.0, 16.6), (9.2, 14.4),
        (10.5, 13.6), (12.0, 13.9), (13.0, 13.2),
        # stem: concave taper
        (15.0, 11.8), (20.0, 10.0), (27.0, 8.6), (34.0, 7.6), (40.0, 7.0), (42.0, 7.0),
        # collar
        (43.0, 9.0), (44.0, 12.4), (45.5, 13.0), (47.0, 12.4), (48.0, 9.0), (49.0, 6.6),
        # neck
        (51.0, 6.0), (54.0, 6.2), (56.0, 0.0),
    ])
    f = PchipInterpolator(pts[:, 0], pts[:, 1], extrapolate=False)
    head_c, head_r = 64.0, 12.0
    return f, head_c, head_r


def _shape_pawn(X, Y):
    f, hc, hr = _pawn_profile()
    yy = Y + 38.0   # nominal frame: centred roughly on the piece
    r = np.nan_to_num(f(np.clip(yy, 0.0, 56.0)), nan=0.0)
    r = np.where((yy >= 0) & (yy <= 56.0), np.maximum(r, 0.0), 0.0)
    rh = np.sqrt(np.maximum(hr * hr - (yy - hc) ** 2, 0.0))
    r = np.maximum(r, rh)
    q = r * r - X * X
    mask = q > 0
    return np.sqrt(np.maximum(q, 0.0)), mask, 0.0


_SHAPES: dict[str, Callable] = {
    "cube": _shape_cube,
    "sphere": _shape_sphere,
    "torus": _shape_torus,
    "pyramid": _shape_pyramid,
    "heart": _shape_heart,
    "pawn": _shape_pawn,
}


# ------------------------------------------------------------------ text
_FONT_PATHS = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
)
_FONT_GLOBS = ("DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf", "FreeSansBold.ttf",
               "Arial*Bold*.ttf", "*Sans*Bold.ttf")
_FONT_DIRS = ("/usr/share/fonts", "/usr/local/share/fonts", os.path.expanduser("~/.fonts"),
              os.path.expanduser("~/.local/share/fonts"))


@functools.lru_cache(maxsize=1)
def find_bold_font() -> Optional[str]:
    """Path of a bold sans TrueType font, or None (then Pillow's default font is used)."""
    for p in _FONT_PATHS:
        if os.path.isfile(p):
            return p
    for pat in _FONT_GLOBS:
        for d in _FONT_DIRS:
            hits = sorted(glob.glob(os.path.join(d, "**", pat), recursive=True))
            if hits:
                return hits[0]
    return None


def _load_font(size: int):
    from PIL import ImageFont

    path = find_bold_font()
    if path is not None:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


def _text_mask(text: str, width_mm: float, height_mm: float, h: float) -> np.ndarray:
    """Rasterise `text` at h mm per pixel, fitted into width_mm x height_mm.

    Returns a bool array (rows top->bottom) tightly cropped to the ink."""
    from PIL import Image, ImageDraw

    def render(size):
        font = _load_font(size)
        l, t, r, b = font.getbbox(text)
        pad = 4
        img = Image.new("L", (r - l + 2 * pad, b - t + 2 * pad), 0)
        ImageDraw.Draw(img).text((pad - l, pad - t), text, fill=255, font=font)
        m = np.asarray(img) >= 128
        rows = np.flatnonzero(m.any(axis=1))
        cols = np.flatnonzero(m.any(axis=0))
        if rows.size == 0:
            raise ValueError(f"text {text!r} rasterised to nothing")
        return m[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]

    probe = 200
    m = render(probe)       # ink extents (not the font's advance box) set the size
    size = probe
    for _ in range(3):
        scale = min(width_mm / h / m.shape[1], height_mm / h / m.shape[0])
        size = max(4, int(math.floor(size * scale)))
        m = render(size)
        if m.shape[1] * h <= width_mm and m.shape[0] * h <= height_mm:
            break
    while (m.shape[1] * h > width_mm or m.shape[0] * h > height_mm) and size > 4:
        size -= 1
        m = render(size)
    return m


# ----------------------------------------------------------------- build
def _relief_range(view: ViewProfile) -> float:
    cross = view.with_(mode="cross")
    return cross.z_near - cross.z_background


def build_scene(name: str, geom: FoldGeometry, view: ViewProfile, *, bg_vee: float = 0.25,
                text: str = "DuoVision") -> HeightScene:
    """Build the named scene for this fold and viewpoint (deterministic)."""
    if name not in SCENE_NAMES:
        raise ValueError(f"unknown scene {name!r}; choose from {SCENE_NAMES}")
    h = GRID_MM
    lay = scene_layout(geom)
    x_half, yc, y_half = lay["x_half"], lay["y_centre"], lay["y_half"]
    if x_half <= 5 or y_half <= 5:
        raise ValueError("display too small for the scene layout")
    z_bg0 = view.z_background
    slope = bg_vee * geom.c / geom.s
    footprint = (geom.x_extent_mm, geom.height_mm / 2.0)

    def bg(x):
        return z_bg0 + slope * np.abs(x)

    if view.mode == "cross":
        z_top = view.z_near
    else:
        # parallel: same relief as cross mode, rising toward the viewer, kept behind the panels
        z_top = min(z_bg0 + _relief_range(view), -PANEL_CLEARANCE_MM)
    meta = {"bg_vee": bg_vee, "layout": lay, "grid_mm": h}

    # ---------- plaque: exactly planar frontal rectangle
    if name == "plaque":
        zp = z_bg0 + PLAQUE_OFFSET_MM
        hw = min(45.0, x_half)
        if slope > 0:   # its side edges must clear the V background
            hw = min(hw, (PLAQUE_OFFSET_MM - BG_CLEARANCE_MM) / slope)
        hh = min(27.5, y_half)
        nxh = int(math.floor(hw / h + 1e-9))
        hw = nxh * h            # edges on grid nodes
        nx_half = nxh + 10      # 1 mm of background margin
        xs = h * np.arange(-nx_half, nx_half + 1)
        ys = yc + h * np.arange(-int(math.ceil(hh / h)) - 10, int(math.ceil(hh / h)) + 11)
        X, Y = np.meshgrid(xs, ys)
        inside = (np.abs(X) <= hw + 1e-9) & (np.abs(Y - yc) <= hh + 1e-9)
        grid = np.where(inside, np.maximum(zp, bg(X)), bg(X))
        meta.update(z_lo=zp, z_top=zp, plaque_z=zp, object_extent=(hw, yc - hh, yc + hh),
                    size_mm=(2 * hw, 2 * hh))
        return HeightScene(name, z_bg0=z_bg0, bg_slope=slope, grid=grid, grid_x0=xs[0],
                           grid_y0=ys[0], grid_step=h, footprint=footprint, mode=view.mode, meta=meta)

    # ---------- depth map of the object on its own grid
    if name == "text":
        if not text:
            raise ValueError("text must be non-empty")
        width = min(125.0, 2.0 * x_half)
        m = _text_mask(text, width, 2.0 * y_half, h)[::-1]   # rows bottom->top (y increasing)
        my, mx = m.shape
        pad_x = 10 + ((mx + 1) % 2)        # odd width -> a node column at x = 0
        m = np.pad(m, ((10, 10), (pad_x, 10)))
        ny, nx = m.shape
        xs = h * (np.arange(nx) - (nx - 1) // 2)
        ys = yc + h * (np.arange(ny) - (ny - 1) / 2.0)
        X, Y = np.meshgrid(xs, ys)
        mask = m
        dist = ndimage.distance_transform_edt(mask, sampling=h)
        extrude, bevel_w, bevel_d = 8.0, 0.9, 1.2
        u = np.clip((dist - 0.5 * h) / bevel_w, 0.0, 1.0)   # distance from the ink edge
        depth = np.where(mask, extrude - bevel_d * (1.0 - np.sqrt(1.0 - (1.0 - u) ** 2)), 0.0)
        base = 0.0
        relief_cap = TEXT_RELIEF_MM
        ink_x = xs[np.flatnonzero(mask.any(axis=0))]
        ink_y = ys[np.flatnonzero(mask.any(axis=1))]
        meta.update(font=find_bold_font(), text=text, text_width_mm=float(ink_x[-1] - ink_x[0] + h),
                    text_height_mm=float(ink_y[-1] - ink_y[0] + h))
    else:
        fn = _SHAPES[name]
        # coarse pass: nominal extents, then fit scale and vertical centring
        cs = 0.5
        cg = np.arange(-90.0, 90.0 + cs, cs)
        CX, CY = np.meshgrid(cg, cg)
        _, cm, _ = fn(CX, CY)
        if not cm.any():
            raise RuntimeError(f"shape {name} is empty")
        ex = np.abs(CX[cm]).max() + cs
        ylo, yhi = CY[cm].min() - cs, CY[cm].max() + cs
        ey = (yhi - ylo) / 2.0
        y_mid = (yhi + ylo) / 2.0
        sc = min(1.0, x_half / ex, y_half / ey)
        nxh = int(math.ceil((ex * sc + 1.0) / h))
        nyh = int(math.ceil((ey * sc + 1.0) / h))
        xs = h * np.arange(-nxh, nxh + 1)
        ys = yc + h * np.arange(-nyh, nyh + 1)
        X, Y = np.meshgrid(xs, ys)
        depth, mask, base = fn(X / sc, (Y - yc) / sc + y_mid)
        depth = depth * sc
        base = None if base is None else base * sc
        relief_cap = None
        meta.update(fit_scale=sc)

    if not mask.any():
        raise RuntimeError(f"scene {name} has an empty object")
    dm = depth[mask]
    d_base = float(dm.min()) if base is None else float(base)
    d_max = float(dm.max())
    hn = np.clip((depth - d_base) / max(d_max - d_base, 1e-12), 0.0, 1.0)

    # lowest z_lo that keeps every object point >= local background + clearance
    hm = hn[mask]
    bm = bg(X[mask])
    sel = hm < 0.999
    z_lo = z_bg0 + BG_CLEARANCE_MM
    if sel.any():
        z_lo = max(z_lo, float(np.max((bm[sel] + BG_CLEARANCE_MM - hm[sel] * z_top) / (1.0 - hm[sel]))))
    if relief_cap is not None:
        z_lo = max(z_lo, z_top - relief_cap)
    z_lo = min(z_lo, z_top - MIN_RELIEF_MM)
    zobj = z_lo + hn * (z_top - z_lo)
    bgg = bg(X)
    grid = np.where(mask, np.maximum(zobj, bgg), bgg)

    xin = X[mask]
    yin = Y[mask]
    meta.update(z_lo=z_lo, z_top=z_top, depth_range_mm=d_max - d_base,
                compression=(z_top - z_lo) / max(d_max - d_base, 1e-12),
                object_extent=(float(np.abs(xin).max()), float(yin.min()), float(yin.max())),
                size_mm=(float(2 * np.abs(xin).max()), float(yin.max() - yin.min())))
    return HeightScene(name, z_bg0=z_bg0, bg_slope=slope, grid=grid, grid_x0=xs[0], grid_y0=ys[0],
                       grid_step=h, footprint=footprint, mode=view.mode, meta=meta)


# -------------------------------------------------------------- ray casting
def _bg_first_t(scene: HeightScene, o: np.ndarray, d: np.ndarray) -> np.ndarray:
    """First t >= 0 with o.z + t d.z <= z_bg(o.x + t d.x); inf if never.

    f(t) = oz + t dz - z0 - k |ox + t dx| is concave, so when f(0) > 0 there is
    at most one positive root and it is the answer."""
    z0, k = scene.z_bg0, scene.bg_slope
    ox, oz = o[..., 0], o[..., 2]
    dx, dz = d[..., 0], d[..., 2]
    f0 = oz - z0 - k * np.abs(ox)
    with np.errstate(divide="ignore", invalid="ignore"):
        tp = (z0 + k * ox - oz) / (dz - k * dx)     # branch x >= 0
        tm = (z0 - k * ox - oz) / (dz + k * dx)     # branch x <= 0
    tol = 1e-9
    xp = ox + tp * dx
    xm = ox + tm * dx
    okp = np.isfinite(tp) & (tp > 0) & (xp >= -tol * (1 + np.abs(ox)))
    okm = np.isfinite(tm) & (tm > 0) & (xm <= tol * (1 + np.abs(ox)))
    t = np.maximum(np.where(okp, tp, -np.inf), np.where(okm, tm, -np.inf))
    t = np.where(np.isfinite(t), t, np.inf)
    return np.where(f0 <= 0, 0.0, t)


def _box_interval(o, d, lo, hi):
    """Slab test: t-interval of the ray inside the axis-aligned box [lo, hi]."""
    t0 = np.zeros(o.shape[:-1])
    t1 = np.full(o.shape[:-1], np.inf)
    for a in range(3):
        oa, da = o[..., a], d[..., a]
        with np.errstate(divide="ignore", invalid="ignore"):
            ta = (lo[a] - oa) / da
            tb = (hi[a] - oa) / da
        par = da == 0
        inside_par = (oa >= lo[a]) & (oa <= hi[a])
        tmin = np.where(par, np.where(inside_par, -np.inf, np.inf), np.minimum(ta, tb))
        tmax = np.where(par, np.where(inside_par, np.inf, -np.inf), np.maximum(ta, tb))
        t0 = np.maximum(t0, tmin)
        t1 = np.minimum(t1, tmax)
    return t0, t1


def _march_grid(scene: HeightScene, o, d, ta, tb, step_mm, refine_iters):
    """March rays (n,3) over [ta, tb] against the baked grid; returns t of first hit (inf if none).

    o is (1, 3) (one eye for all rays) or (n, 3). Samples are step_mm apart in z.
    Empty-space skipping: where the ray is above the maximum of the grid over the
    3 x 3 tiles around it, it jumps ahead as far as provably stays above the
    surface (never past a possible hit, so the result equals plain marching)."""
    n = ta.shape[0]
    t_hit = np.full(n, np.inf)
    if n == 0:
        return t_hit
    gx0, gy0, hh = scene._gx0, scene._gy0, scene._h
    tiles, tn = scene._tile_bounds()
    nty, ntx = tiles.shape
    tflat = tiles.ravel()

    if o.shape[0] > 1:
        ox, oy, oz = o[:, 0].copy(), o[:, 1].copy(), o[:, 2].copy()
    else:
        ox, oy, oz = (np.full(n, o[0, k]) for k in range(3))
    dx, dy, dz = d[:, 0].copy(), d[:, 1].copy(), d[:, 2].copy()
    # work in grid-index units horizontally: fi = ax + t * bx
    ax, bx = (ox - gx0) / hh, dx / hh
    ay, by = (oy - gy0) / hh, dy / hh

    def f(ax, bx, ay, by, oz, dz, t):
        return oz + t * dz - scene._interp(ax + t * bx, ay + t * by)

    dz_abs = np.abs(dz)
    dt = step_mm / np.maximum(dz_abs, 0.1 * np.sqrt(dx * dx + dy * dy + dz * dz))
    inv_down = np.where(dz < 0, 1.0 / np.maximum(dz_abs, 1e-300), 0.0)
    with np.errstate(divide="ignore"):
        inv_bx = np.where(bx != 0, 1.0 / np.abs(bx), np.inf)
        inv_by = np.where(by != 0, 1.0 / np.abs(by), np.inf)
    sx = bx > 0
    sy = by > 0

    f_prev = f(ax, bx, ay, by, oz, dz, ta)
    start_in = f_prev <= 0
    t_hit[start_in] = ta[start_in]
    keep = ~start_in
    idx = np.flatnonzero(keep)
    arrs = [a_[keep] for a_ in (ax, bx, ay, by, oz, dz, dt, inv_down, inv_bx, inv_by, sx, sy, tb)]
    t_prev, f_prev = ta[keep], f_prev[keep]
    out_idx, out_lo, out_hi, out_flo, out_fhi, out_dt = [], [], [], [], [], []
    while idx.size:
        ax_, bx_, ay_, by_, oz_, dz_, dt_, invd, ibx, iby, sx_, sy_, tb_ = arrs
        fi = ax_ + t_prev * bx_
        fj = ay_ + t_prev * by_
        ti = np.clip(np.floor(fi / tn).astype(np.intp), 0, ntx - 1)
        tj = np.clip(np.floor(fj / tn).astype(np.intp), 0, nty - 1)
        gap = (oz_ + t_prev * dz_) - tflat[tj * ntx + ti] - 1e-6
        # nodes up to one tile away are covered; keep one node of slack for bilinear support
        room_x = np.where(sx_, (ti + 2) * tn - 2 - fi, fi - (ti - 1) * tn - 1)
        room_y = np.where(sy_, (tj + 2) * tn - 2 - fj, fj - (tj - 1) * tn - 1)
        skip = np.minimum(np.minimum(room_x * ibx, room_y * iby),
                          np.where(invd > 0, gap * invd, np.inf))
        skip = np.where(gap > 0, skip, 0.0)
        t = np.minimum(t_prev + np.maximum(skip, dt_), tb_)
        fv = f(ax_, bx_, ay_, by_, oz_, dz_, t)
        hit = fv <= 0
        if hit.any():
            out_idx.append(idx[hit]); out_lo.append(t_prev[hit]); out_hi.append(t[hit])
            out_flo.append(f_prev[hit]); out_fhi.append(fv[hit]); out_dt.append(dt_[hit])
        keep = ~hit & (t < tb_)
        if not keep.all():
            idx = idx[keep]
            arrs = [a_[keep] for a_ in arrs]
            t_prev, f_prev = t[keep], fv[keep]
        else:
            t_prev, f_prev = t, fv
    if not out_idx:
        return t_hit
    hidx = np.concatenate(out_idx)
    lo, hi = np.concatenate(out_lo), np.concatenate(out_hi)
    flo, fhi = np.concatenate(out_flo), np.concatenate(out_fhi)
    hdt = np.concatenate(out_dt)
    a4 = [a_[hidx] for a_ in (ax, bx, ay, by, oz, dz)]
    # brackets are at most one step long (skips never jump over a possible hit)
    extra = int(np.ceil(np.log2(max(1.0, float(np.max((hi - lo) / hdt))))))
    for _ in range(refine_iters + extra):
        mid = 0.5 * (lo + hi)
        fm = f(*a4, mid)
        ins = fm <= 0
        hi = np.where(ins, mid, hi)
        fhi = np.where(ins, fm, fhi)
        lo = np.where(ins, lo, mid)
        flo = np.where(ins, flo, fm)
    # secant step inside the final bracket
    w = flo / np.where(flo - fhi > 0, flo - fhi, 1.0)
    t_hit[hidx] = lo + np.clip(w, 0.0, 1.0) * (hi - lo)
    return t_hit


def first_hit(scene: HeightScene, eye, targets, *, step_mm: float = 0.5,
              refine_iters: int = 12) -> Tuple[np.ndarray, np.ndarray]:
    """First intersection of rays eye -> targets (and beyond, t >= 0) with the height field.

    eye: (3,) or broadcastable to targets (..., 3). Returns (points (..., 3), hit (...)).
    The background is intersected analytically; the object is found by marching in
    z steps of step_mm inside its bounding box between its z range (within
    [scene.z_min, scene.z_max]), followed by refine_iters bisection steps
    (bracket < 0.5 / 2**12 mm for the defaults) and a secant step.
    hit is False (points NaN) where the ray never meets the surface.
    """
    eye = np.asarray(eye, dtype=np.float64)
    tg = np.asarray(targets, dtype=np.float64)
    if eye.shape[-1] != 3 or tg.shape[-1] != 3:
        raise ValueError("eye and targets must have a last axis of length 3")
    shape = np.broadcast_shapes(eye.shape, tg.shape)[:-1]
    single_eye = eye.ndim == 1
    tg_f = np.broadcast_to(tg, shape + (3,)).reshape(-1, 3)
    eye_f = eye.reshape(1, 3) if single_eye else np.broadcast_to(eye, shape + (3,)).reshape(-1, 3)
    n = tg_f.shape[0]
    pts = np.full((n, 3), np.nan)
    hit = np.zeros(n, dtype=bool)
    eps = 1e-3
    box = scene.grid_bounds
    if box is not None:
        blo = np.array([box[0], box[2], scene._gzmin - eps])
        bhi = np.array([box[1], box[3], scene._gzmax + eps])
    for s in range(0, n, _CHUNK):
        e = min(n, s + _CHUNK)
        tgt = tg_f[s:e]
        o = eye_f if single_eye else eye_f[s:e]
        d = tgt - o
        t = _bg_first_t(scene, o, d) if not single_eye else _bg_first_t(scene, o[0], d)
        if box is not None:
            ob = o if not single_eye else o[0]
            t0, t1 = _box_interval(ob, d, blo, bhi)
            t1 = np.minimum(t1, t)
            cand = np.flatnonzero(t0 < t1)
            if cand.size:
                oc = o if single_eye else o[cand]
                tc = _march_grid(scene, oc, d[cand], t0[cand], t1[cand], step_mm, refine_iters)
                t[cand] = np.minimum(t[cand], tc)
        ok = np.isfinite(t)
        hit[s:e] = ok
        with np.errstate(invalid="ignore"):
            p = o + np.where(ok, t, 0.0)[:, None] * d
        pts[s:e] = np.where(ok[:, None], p, np.nan)
    return pts.reshape(shape + (3,)), hit.reshape(shape)
