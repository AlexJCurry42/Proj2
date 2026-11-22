"""Fold geometry: where every pixel of the folded inner display sits in 3D,
and where any ray lands on it.

Model: two flat halves meeting at a sharp vertical hinge, opened to
`hinge_deg` (180 = flat). Book posture, inner display facing the viewer,
outer edges swing toward the viewer (+z). The fold strip is treated as
part of the two flat halves (its measured depth is ~0.05 mm; see spec).

Unfolded surface coordinates (u, v) in mm:
    u = signed distance along the surface from the hinge (negative = left half)
    v = height above the hinge midpoint
World position of (u, v):
    (u * sin(phi/2),  v,  |u| * cos(phi/2))
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from .config import DEFAULT_DEVICE, DEFAULT_HINGE_DEG, DeviceProfile

_EPS = 1e-9


class FoldGeometry:
    """Geometry of the inner display folded at `hinge_deg`."""

    def __init__(self, device: DeviceProfile = DEFAULT_DEVICE, hinge_deg: float = DEFAULT_HINGE_DEG):
        if not (0.0 < hinge_deg <= 180.0):
            raise ValueError(f"hinge_deg must be in (0, 180], got {hinge_deg}")
        self.device = device
        self.hinge_deg = float(hinge_deg)
        half = math.radians(hinge_deg) / 2.0
        self.s = math.sin(half)
        self.c = math.cos(half)
        # Unit vectors along each half, pointing from the hinge to its outer edge.
        self.a_left = np.array([-self.s, 0.0, self.c])
        self.a_right = np.array([self.s, 0.0, self.c])
        # Unit normals of each half, pointing toward the viewer.
        self.n_left = np.array([self.c, 0.0, self.s])
        self.n_right = np.array([-self.c, 0.0, self.s])
        self.mm_per_px = device.mm_per_px
        self.half_width_mm = device.half_width_mm
        self.height_mm = device.height_mm

    # ------------------------------------------------------------------ pixels
    def pixel_to_uv(self, i, j) -> Tuple[np.ndarray, np.ndarray]:
        """Pixel centre (or fractional pixel coordinate) -> unfolded (u, v) mm."""
        d = self.device
        u = (np.asarray(i, dtype=np.float64) + 0.5 - d.width_px / 2.0) * self.mm_per_px
        v = (d.height_px / 2.0 - (np.asarray(j, dtype=np.float64) + 0.5)) * self.mm_per_px
        return u, v

    def uv_to_pixel(self, u, v) -> Tuple[np.ndarray, np.ndarray]:
        """Unfolded (u, v) mm -> fractional pixel coordinates (i, j).

        Integer values of the result are pixel centres.
        """
        d = self.device
        i = np.asarray(u, dtype=np.float64) / self.mm_per_px + d.width_px / 2.0 - 0.5
        j = d.height_px / 2.0 - np.asarray(v, dtype=np.float64) / self.mm_per_px - 0.5
        return i, j

    def uv_to_world(self, u, v) -> np.ndarray:
        u = np.asarray(u, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        u, v = np.broadcast_arrays(u, v)
        return np.stack([u * self.s, v, np.abs(u) * self.c], axis=-1)

    def pixel_to_world(self, i, j) -> np.ndarray:
        return self.uv_to_world(*self.pixel_to_uv(i, j))

    def pixel_grid_world(self) -> np.ndarray:
        """World positions of every pixel centre, shape (H, W, 3)."""
        d = self.device
        jj, ii = np.meshgrid(np.arange(d.height_px), np.arange(d.width_px), indexing="ij")
        return self.pixel_to_world(ii, jj)

    # --------------------------------------------------------------- surfaces
    @property
    def x_extent_mm(self) -> float:
        """Half the horizontal extent of the folded display (world x)."""
        return self.half_width_mm * self.s

    def panel_z(self, x) -> np.ndarray:
        """World z of the display surface at world x (valid for |x| <= x_extent_mm)."""
        x = np.asarray(x, dtype=np.float64)
        if self.s < _EPS:
            raise ValueError("fully closed fold has no frontal extent")
        return np.abs(x) * self.c / self.s

    def max_panel_z(self) -> float:
        """z of the outer edges (nearest part of the display to the viewer)."""
        return self.half_width_mm * self.c

    # ------------------------------------------------------------ intersection
    def intersect(self, origins, dirs) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Intersect rays origin + t*dir (t > 0) with the folded display.

        origins, dirs: arrays broadcastable to (..., 3). dirs need not be unit.
        Returns (u, v, t, valid). u, v are NaN where valid is False.
        When a ray meets both halves inside their extents, the nearer hit wins.
        """
        o = np.asarray(origins, dtype=np.float64)
        d = np.asarray(dirs, dtype=np.float64)
        o, d = np.broadcast_arrays(o, d)
        best_t = np.full(o.shape[:-1], np.inf)
        best_u = np.full(o.shape[:-1], np.nan)
        best_v = np.full(o.shape[:-1], np.nan)
        tol = 1e-7
        for n, a, sign in ((self.n_left, self.a_left, -1.0), (self.n_right, self.a_right, 1.0)):
            nd = d @ n
            no = o @ n
            with np.errstate(divide="ignore", invalid="ignore"):
                t = np.where(np.abs(nd) > _EPS, -no / nd, np.inf)
            p = o + t[..., None] * d
            along = p @ a
            v = p[..., 1]
            ok = (
                np.isfinite(t)
                & (t > _EPS)
                & (along >= -tol)
                & (along <= self.half_width_mm + tol)
                & (np.abs(v) <= self.height_mm / 2.0 + tol)
            )
            better = ok & (t < best_t)
            best_t = np.where(better, t, best_t)
            best_u = np.where(better, sign * np.clip(along, 0.0, None), best_u)
            best_v = np.where(better, v, best_v)
        valid = np.isfinite(best_t)
        best_t = np.where(valid, best_t, np.nan)
        return best_u, best_v, best_t, valid

    def project_from_eye(self, eye, points) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Where the line from `eye` through `points` lands on the display.

        Works whether the points lie in front of the display (cross mode) or
        behind it (parallel mode). Returns fractional pixel coords (i, j) and valid.
        """
        eye = np.asarray(eye, dtype=np.float64)
        pts = np.asarray(points, dtype=np.float64)
        u, v, _, valid = self.intersect(eye, pts - eye)
        i, j = self.uv_to_pixel(u, v)
        return i, j, valid

    def pixel_side(self, i) -> np.ndarray:
        """-1 for the left half, +1 for the right half."""
        return np.where(np.asarray(i, dtype=np.float64) + 0.5 < self.device.width_px / 2.0, -1, 1)

    def panel_corners(self, side: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(pa, pb, pc) = lower-left, lower-right, upper-left corners of one half,
        in the convention of Kooima's generalized perspective projection."""
        w, h2 = self.half_width_mm, self.height_mm / 2.0
        edge_l = w * self.a_left
        edge_r = w * self.a_right
        down = np.array([0.0, -h2, 0.0])
        up = np.array([0.0, h2, 0.0])
        if side == "left":
            return edge_l + down, down.copy(), edge_l + up
        if side == "right":
            return down.copy(), edge_r + down, up.copy()
        raise ValueError("side must be 'left' or 'right'")
