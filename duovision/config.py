"""Device and viewing configuration for DuoVision.

All lengths are millimetres, all angles degrees unless a name says otherwise.

World frame (shared by every module):
    origin  = centre of the hinge line
    +x      = to the viewer's right
    +y      = up, along the hinge
    +z      = toward the viewer
The eyes sit at z = distance_mm (in front of the hinge), level with
y = eye_height_mm, centred on x = 0.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal, Tuple

import numpy as np

ViewMode = Literal["cross", "parallel"]


@dataclass(frozen=True)
class DeviceProfile:
    """iPhone Duo inner display, unfolded landscape: both halves side by side.

    Pixel (i, j): i = column 0..width_px-1 left->right, j = row 0..height_px-1
    top->bottom. The hinge runs vertically through the middle column boundary.
    Source: Apple iPhone Duo specs (7.6 in, 2670 x 1878 px, 430 ppi).
    """

    width_px: int = 2670
    height_px: int = 1878
    ppi: float = 430.0

    @property
    def mm_per_px(self) -> float:
        return 25.4 / self.ppi

    @property
    def half_width_px(self) -> float:
        return self.width_px / 2.0

    @property
    def half_width_mm(self) -> float:
        """Width of one half, hinge to outer edge."""
        return self.half_width_px * self.mm_per_px

    @property
    def height_mm(self) -> float:
        return self.height_px * self.mm_per_px

    def scaled(self, factor: float) -> "DeviceProfile":
        """Same physical panel at a lower pixel density (for fast tests).

        factor=0.25 gives a quarter-resolution panel with identical size in mm.
        """
        w = max(2, int(round(self.width_px * factor)))
        if w % 2:
            w += 1
        h = max(2, int(round(self.height_px * factor)))
        return DeviceProfile(width_px=w, height_px=h, ppi=self.ppi * w / self.width_px)


@dataclass(frozen=True)
class ViewProfile:
    """Fixed design viewpoint (no head tracking).

    distance_mm   eye plane to hinge line, along z
    ipd_mm        interpupillary distance
    eye_height_mm eye level relative to the hinge midpoint (y)
    mode          'cross' (eyes converge in front of the fold, scene floats
                  in front) or 'parallel' (scene behind the fold)
    sep_bg_mm     pattern separation of the background at the hinge axis
    sep_near_mm   pattern separation of the nearest allowed scene point
    """

    distance_mm: float = 300.0
    ipd_mm: float = 63.0
    eye_height_mm: float = 0.0
    mode: ViewMode = "cross"
    sep_bg_mm: float = 15.0
    sep_near_mm: float = 22.0

    def eyes(self, offset: Tuple[float, float, float] = (0.0, 0.0, 0.0)) -> Tuple[np.ndarray, np.ndarray]:
        """Left and right eye positions in the world frame.

        offset shifts both eyes together (head displacement), used by the
        verifier for off-centre tolerance tests. Encoding always uses (0,0,0).
        """
        dx, dy, dz = offset
        y = self.eye_height_mm + dy
        z = self.distance_mm + dz
        half = self.ipd_mm / 2.0
        left = np.array([-half + dx, y, z], dtype=np.float64)
        right = np.array([half + dx, y, z], dtype=np.float64)
        return left, right

    def cyclopean_eye(self, offset: Tuple[float, float, float] = (0.0, 0.0, 0.0)) -> np.ndarray:
        left, right = self.eyes(offset)
        return (left + right) / 2.0

    def fused_distance(self, sep_mm: float) -> float:
        """Distance from the eye plane at which a flat-screen separation fuses.

        cross:    z = I*D/(I+s)  (in front of the screen)
        parallel: z = I*D/(I-s)  (behind the screen), requires s < I
        """
        i, d = self.ipd_mm, self.distance_mm
        if self.mode == "cross":
            return i * d / (i + sep_mm)
        if sep_mm >= i:
            raise ValueError("parallel separation must be smaller than the IPD")
        return i * d / (i - sep_mm)

    def world_z_for_separation(self, sep_mm: float) -> float:
        """World z (relative to the hinge) of the point fused at the hinge axis.

        Positive = in front of the hinge (cross mode), negative = behind (parallel).
        """
        return self.distance_mm - self.fused_distance(sep_mm)

    @property
    def z_background(self) -> float:
        return self.world_z_for_separation(self.sep_bg_mm)

    @property
    def z_near(self) -> float:
        return self.world_z_for_separation(self.sep_near_mm)

    def conflict_dioptres(self, sep_mm: float) -> float:
        """Vergence-accommodation conflict for a separation at the hinge: s/(I*D)."""
        return sep_mm / (self.ipd_mm * self.distance_mm / 1000.0)

    def with_(self, **kw) -> "ViewProfile":
        return replace(self, **kw)


GUIDE_BAND_FRACTION = 0.07  # top band reserved for guide dots + label


def guide_band_rows(device: DeviceProfile) -> int:
    """Rows at the top of the image reserved for guides (never encoded)."""
    return int(round(device.height_px * GUIDE_BAND_FRACTION))


DEFAULT_DEVICE = DeviceProfile()
DEFAULT_VIEW = ViewProfile()
DEFAULT_HINGE_DEG = 120.0
HINGE_SWEEP_DEG = tuple(range(100, 141, 5))
