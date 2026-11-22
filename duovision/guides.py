"""Viewing guides: fold-aware guide dots, the guide band, and the alignment card.

The user views pre-rendered images in Photos/Safari, so every image is encoded
for one hinge angle and one fixed viewpoint.  These guides help them match it:

* Guide dots (top band of every stereogram).  A single world point ``G`` at
  the background depth on the hinge axis is projected onto the folded display
  from each eye.  Cross-eyed, the right eye's dot lands on the left half and
  the left eye's dot on the right half; converging at the background depth
  fuses the two dots into three (the middle one is ``G``).
* ``alignment_image``: a frontal reference figure (square, inscribed circle,
  crosshair, grid) in the plane ``z = z_background`` drawn as seen from the
  cyclopean eye.  From the design viewpoint at the encoded hinge angle every
  line runs straight across the hinge; a wrong fold angle kinks the
  horizontal lines at the hinge, an off-centre head makes the halves look
  unequally wide.
* ``fold_guide_image``: an anamorphic picture (frame, checkerboard bands,
  round face, "FOLD / 120°") for finding the fold angle by eye.  The flat
  picture is pre-distorted so that it only looks right (straight frame, square
  checks, round face) from the design viewpoint when the phone is folded to
  the encoded hinge angle: flatter than that it looks tall in the middle,
  folded further it looks pinched in the middle.

Everything is vectorised numpy; text is rendered with Pillow.
"""
from __future__ import annotations

import functools
import math
from typing import Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .config import ViewProfile, guide_band_rows
from .geometry import FoldGeometry

# Appearance (grey levels / RGB, sizes in mm on the display).
BAND_BG = 34
DOT_LEVEL = 255
TEXT_LEVEL = 200
TICK_LEVEL = 150
DOT_RADIUS_MM = 1.2
TEXT_HEIGHT_FRAC = 0.30          # font size as a fraction of the band height (2-line blocks)
LABEL_HEIGHT_FRAC = 0.32         # font size for a single-line label
EDGE_MARGIN_MM = 4.0             # keep text clear of the rounded outer corners
AA_LEVELS = 16                   # coverage quantisation -> few colours -> compact palette PNGs

_FONT_CANDIDATES = (
    "DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    "Arial.ttf",
    "Helvetica.ttc",
)

DEFAULT_HINT = "Cross your eyes until the two dots become three"
ALIGN_INSTRUCTIONS = (
    "Fold until the lines run straight across the hinge;",
    "centre until both halves look equally wide.",
)
FOLD_INSTRUCTIONS = (
    "Fold until the picture looks flat: straight frame, round face.",
    "Tall in the middle: fold more · pinched in the middle: open up.",
)
FOLD_SAFE_WIDTH_FRAC = 0.78      # iOS hides ~9% of the Duo's inner display behind its status strip on
                                 # one side (the viewer mirrors it on the other): keep the frame inside
FOLD_MARGIN_MM = 6.0             # background kept around the picture frame in the target bitmap
FOLD_TARGET_OVERSAMPLE = 1.5     # target bitmap pixels per display pixel
FOLD_BG = (6, 6, 10)
FOLD_FRAME = (255, 255, 255)
FOLD_PAPER = (14, 22, 42)
FOLD_CHECK = (36, 74, 150)
FOLD_FACE = (255, 204, 64)
FOLD_INK = (40, 30, 10)


# --------------------------------------------------------------------- fonts
@functools.lru_cache(maxsize=64)
def _font(size_px: int):
    size_px = max(6, int(size_px))
    for name in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(name, size_px)
        except OSError:
            continue
    try:  # Pillow >= 10.1 ships a scalable default font
        return ImageFont.load_default(size=size_px)
    except TypeError:  # pragma: no cover - very old Pillow
        return ImageFont.load_default()


def _text_width(text: str, font) -> int:
    l, _t, r, _b = font.getbbox(text)
    return int(math.ceil(r - l))


def _fit_font(lines: Sequence[str], size_px: int, max_width: float):
    """Largest font <= size_px (>= 6 px) whose widest line fits in max_width."""
    size = max(6, int(size_px))
    while size > 6:
        f = _font(size)
        if max(_text_width(t, f) for t in lines) <= max_width:
            return f
        size -= 1
    return _font(6)


def _quantize(cov: np.ndarray) -> np.ndarray:
    """Quantise coverage [0,1] to AA_LEVELS steps (keeps PNG palettes small)."""
    return np.round(np.clip(cov, 0.0, 1.0) * (AA_LEVELS - 1)) / (AA_LEVELS - 1)


# ----------------------------------------------------------------- guide dots
def _band_mid_v(geom: FoldGeometry) -> float:
    nb = guide_band_rows(geom.device)
    _, v = geom.pixel_to_uv(0.0, (nb - 1) / 2.0)
    return float(v)


def _guide_z(view: ViewProfile) -> float:
    if view.mode == "cross":
        return float(view.z_background)
    return float(view.world_z_for_separation(view.sep_bg_mm))


def guide_world_point(geom: FoldGeometry, view: ViewProfile) -> np.ndarray:
    """World point G on the hinge axis (x=0) at the background depth.

    Its height y is chosen so that both eyes' projections of G land exactly on
    the middle row of the guide band (the projection scales heights about the
    eye level by (z_eye - z_display)/(z_eye - z_G), the same for both eyes by
    symmetry).
    """
    left, right = view.eyes()
    z_g = _guide_z(view)
    y_e = float(left[1])
    z_e = float(left[2])
    # Where the eye->G line lands in the x-z plane does not depend on y.
    g0 = np.array([0.0, y_e, z_g])
    zs = []
    for eye in (left, right):
        u, _v, _t, ok = geom.intersect(eye, g0 - eye)
        if not bool(ok):
            raise ValueError("guide point does not project onto the display")
        zs.append(abs(float(u)) * geom.c)
    z_d = float(np.mean(zs))
    v_mid = _band_mid_v(geom)
    y_g = y_e + (v_mid - y_e) * (z_e - z_g) / (z_e - z_d)
    return np.array([0.0, y_g, z_g])


def guide_dot_pixels(geom: FoldGeometry, view: ViewProfile) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Fractional pixel centres ((i_l, j_l), (i_r, j_r)) of the two guide dots.

    Cross mode: the left dot is where the RIGHT eye's line through G lands, the
    right dot where the LEFT eye's line lands.  Parallel mode: the reverse
    (each eye's dot on its own side).  Returned ordered left, right.
    """
    g = guide_world_point(geom, view)
    left, right = view.eyes()
    dots = []
    for eye in (left, right):
        i, j, ok = geom.project_from_eye(eye, g)
        if not bool(ok):
            raise ValueError("guide dot falls off the display")
        dots.append((float(i), float(j)))
    dots.sort(key=lambda p: p[0])
    return dots[0], dots[1]


# ------------------------------------------------------------------ the band
def _disc_coverage(shape, centre, radius_px):
    h, w = shape
    jj, ii = np.mgrid[0:h, 0:w]
    d = np.hypot(ii - centre[0], jj - centre[1])
    return np.clip(radius_px + 0.5 - d, 0.0, 1.0)


def _text_mask(shape, blocks) -> Tuple[np.ndarray, list]:
    """Render text blocks into an L mask. blocks: [(lines, font, x0, align)]
    align 'left' -> x0 is the left edge, 'right' -> x0 is the right edge.
    Lines are stacked and vertically centred in the band."""
    h, w = shape
    img = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(img)
    boxes = []
    for lines, font, x0, align in blocks:
        if not lines:
            continue
        asc, desc = font.getmetrics()
        lh = asc + desc
        gap = max(1, int(round(lh * 0.15)))
        total = len(lines) * lh + (len(lines) - 1) * gap
        y = (h - total) / 2.0
        for t in lines:
            tw = _text_width(t, font)
            x = x0 if align == "left" else x0 - tw
            draw.text((x, y), t, fill=255, font=font)
            boxes.append((x, y, x + tw, y + lh))
            y += lh + gap
    return np.asarray(img, dtype=np.float64) / 255.0, boxes


def _render_band(geom: FoldGeometry, view: ViewProfile, left_lines: Sequence[str],
                 right_lines: Sequence[str], *, dots: bool = True, ticks: bool = True,
                 single_line: bool = False) -> np.ndarray:
    """Greyscale band (nb, W) uint8."""
    dev = geom.device
    nb = guide_band_rows(dev)
    w = dev.width_px
    if nb <= 0:
        return np.zeros((0, w), np.uint8)
    mmpx = dev.mm_per_px
    r_px = DOT_RADIUS_MM / mmpx
    margin = EDGE_MARGIN_MM / mmpx
    hinge = w / 2.0  # boundary between the halves, in pixel-edge coordinates

    val = np.full((nb, w), float(BAND_BG))

    if dots:
        (il, jl), (ir, jr) = guide_dot_pixels(geom, view)
        cov = np.maximum(_disc_coverage((nb, w), (il, jl), r_px),
                         _disc_coverage((nb, w), (ir, jr), r_px))
        val = val + (DOT_LEVEL - val) * _quantize(cov)
        left_limit = il - r_px - 2.0 * r_px
        right_limit = ir + r_px + 2.0 * r_px
    else:
        left_limit = hinge - 2.0 * r_px
        right_limit = hinge + 2.0 * r_px

    if ticks:
        # Two thin ticks flanking the hinge column at the band's bottom edge.
        tick_h = max(2.0, 0.35 * nb)
        tick_w = max(1.0, 0.25 / mmpx)
        off = max(1.5, 0.6 / mmpx)
        jj, ii = np.mgrid[0:nb, 0:w]
        xc = ii + 0.5
        yc = jj + 0.5
        vert = np.clip(yc - (nb - tick_h) + 0.5, 0.0, 1.0)
        cov = np.zeros((nb, w))
        for cx in (hinge - off, hinge + off):
            cov = np.maximum(cov, np.clip(tick_w / 2.0 + 0.5 - np.abs(xc - cx), 0.0, 1.0) * vert)
        val = val + (TICK_LEVEL - val) * _quantize(cov)

    frac = LABEL_HEIGHT_FRAC if single_line else TEXT_HEIGHT_FRAC
    size = int(round(nb * frac))
    blocks = []
    if left_lines:
        f = _fit_font(tuple(left_lines), size, left_limit - margin)
        blocks.append((list(left_lines), f, margin, "left"))
    if right_lines:
        f = _fit_font(tuple(right_lines), size, (w - margin) - right_limit)
        blocks.append((list(right_lines), f, w - margin, "right"))
    if blocks:
        tmask, _ = _text_mask((nb, w), blocks)
        val = val + np.clip(TEXT_LEVEL - val, 0.0, None) * _quantize(tmask)

    return np.clip(np.round(val), 0, 255).astype(np.uint8)


def draw_guide_band(image: np.ndarray, geom: FoldGeometry, view: ViewProfile, label: str,
                    *, hint: str | None = DEFAULT_HINT) -> np.ndarray:
    """Fill the reserved top rows with the guide band and return the image.

    Dark grey background, two white anti-aliased guide dots (fold-aware,
    radius 1.2 mm), the label at the left edge, an optional hint on the right,
    and two thin ticks flanking the hinge column at the band's bottom edge.
    A (H, W, 3) uint8 image is modified in place; a (H, W) image is promoted to
    a new (H, W, 3) array.  Rows below the band are never touched.
    """
    dev = geom.device
    if image.shape[:2] != (dev.height_px, dev.width_px):
        raise ValueError(f"image shape {image.shape[:2]} does not match device "
                         f"{(dev.height_px, dev.width_px)}")
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    if image.dtype != np.uint8:
        raise ValueError("image must be uint8")
    nb = guide_band_rows(dev)
    band = _render_band(geom, view, [label] if label else [], [hint] if hint else [],
                        single_line=True)
    image[:nb] = band[..., None]
    return image


# ------------------------------------------------------------ alignment card
def _plane_coords(geom: FoldGeometry, view: ViewProfile):
    """Where the cyclopean ray through each pixel centre crosses z = z_plane.

    Returns plane coordinates (qx, qy) relative to the cyclopean eye's (x, y),
    shape (H, W) each.
    """
    e = view.cyclopean_eye()
    z_p = _guide_z(view)
    p = geom.pixel_grid_world()
    dz = p[..., 2] - e[2]
    t = (z_p - e[2]) / dz
    qx = t * (p[..., 0] - e[0])
    qy = t * (p[..., 1] - e[1])
    return qx, qy


def _line_cov(f, gnorm, half_w_px):
    """AA coverage of the zero set of f, with |grad f| (per pixel) = gnorm."""
    d = np.abs(f) / np.maximum(gnorm, 1e-12)
    return np.clip(half_w_px + 0.5 - d, 0.0, 1.0)


def _plane_limits(geom: FoldGeometry, view: ViewProfile) -> Tuple[float, float, float]:
    """How far (mm, in the reference plane, from the cyclopean axis) a figure can
    extend up, down and sideways and still be drawn on the display below the band."""
    dev = geom.device
    e = view.cyclopean_eye()
    z_p = _guide_z(view)
    nb = guide_band_rows(dev)
    # Top limit: band's bottom edge at the hinge column (the most restrictive,
    # since the hinge is the farthest part of the display from the eye).
    _, v_top = geom.pixel_to_uv(0.0, nb + 2.0)
    _, v_bot = geom.pixel_to_uv(0.0, dev.height_px - 3.0)
    k_hinge = (e[2] - z_p) / e[2]
    up = (float(v_top) - e[1]) * k_hinge
    down = (e[1] - float(v_bot)) * k_hinge
    # Side limit: outer edges.
    xe = geom.half_width_mm * geom.s
    ze = geom.half_width_mm * geom.c
    side = (xe - 3.0 * geom.mm_per_px) * (e[2] - z_p) / (e[2] - ze) - abs(e[0])
    return float(up), float(down), float(side)


def alignment_figure_half_size(geom: FoldGeometry, view: ViewProfile) -> float:
    """Half side (mm, in the reference plane) of the alignment square.

    Nominally 50 mm (a 100 mm square); shrunk in 5 mm steps so the whole square
    is visible below the guide band and inside the side edges.
    """
    up, down, side = _plane_limits(geom, view)
    lim = min(up, down, side) - 2.0
    half = min(50.0, math.floor(lim / 5.0) * 5.0)
    return max(half, 5.0)


def alignment_image(geom: FoldGeometry, view: ViewProfile, label: str = "") -> np.ndarray:
    """Alignment card (H, W, 3) uint8 for matching the fold angle and centring.

    A frontal figure in the plane z = z_background (square, inscribed circle,
    crosshair through the centre, 10 mm grid) is drawn where the cyclopean
    eye sees it on the folded display.  For every pixel the cyclopean ray
    through it is intersected with the plane; lines are drawn by evaluating
    each line's implicit function there and dividing by its pixel-space
    gradient, giving ~0.5 mm (0.25 mm for the grid) anti-aliased strokes.
    """
    dev = geom.device
    mmpx = dev.mm_per_px
    h, w = dev.height_px, dev.width_px
    qx, qy = _plane_coords(geom, view)

    # Pixel-space gradients of the plane coordinates (Jacobian columns).
    dqx_di = np.gradient(qx, axis=1)
    dqx_dj = np.gradient(qx, axis=0)
    dqy_di = np.gradient(qy, axis=1)
    dqy_dj = np.gradient(qy, axis=0)
    g_x = np.hypot(dqx_di, dqx_dj)          # |grad qx|
    g_y = np.hypot(dqy_di, dqy_dj)          # |grad qy|

    half = alignment_figure_half_size(geom, view)
    hw_main = 0.5 / mmpx / 2.0               # 0.5 mm stroke
    hw_grid = 0.25 / mmpx / 2.0              # 0.25 mm stroke
    grid = 10.0

    img = np.zeros((h, w, 3), dtype=np.float64)
    img[:] = (6.0, 6.0, 10.0)

    def over(cov, colour):
        a = _quantize(cov)[..., None]
        img[:] = img * (1.0 - a) + np.asarray(colour, dtype=np.float64) * a

    # Grid (10 mm in the reference plane).
    fx = qx - np.round(qx / grid) * grid
    fy = qy - np.round(qy / grid) * grid
    cov = np.maximum(_line_cov(fx, g_x, hw_grid), _line_cov(fy, g_y, hw_grid))
    over(cov, (70, 78, 100))

    # Crosshair through the centre, across the whole display.
    cov = np.maximum(_line_cov(qx, g_x, hw_main), _line_cov(qy, g_y, hw_main))
    over(cov, (90, 220, 255))

    # Inscribed circle.
    rho = np.hypot(qx, qy)
    with np.errstate(invalid="ignore", divide="ignore"):
        nx = np.where(rho > 0, qx / rho, 0.0)
        ny = np.where(rho > 0, qy / rho, 0.0)
    g_r = np.hypot(nx * dqx_di + ny * dqy_di, nx * dqx_dj + ny * dqy_dj)
    over(_line_cov(rho - half, g_r, hw_main), (255, 170, 60))

    # Square: four segments with square-capped ends.
    def extent(f_along, g_along):
        # 1 inside |along| <= half (+ half stroke), AA at the ends.
        d = (np.abs(f_along) - half) / np.maximum(g_along, 1e-12)
        return np.clip(hw_main + 0.5 - d, 0.0, 1.0)

    ext_x = extent(qx, g_x)
    ext_y = extent(qy, g_y)
    cov = np.zeros((h, w))
    for s in (-1.0, 1.0):
        cov = np.maximum(cov, _line_cov(qy - s * half, g_y, hw_main) * ext_x)   # horizontal edges
        cov = np.maximum(cov, _line_cov(qx - s * half, g_x, hw_main) * ext_y)   # vertical edges
    over(cov, (255, 255, 255))

    out = np.clip(np.round(img), 0, 255).astype(np.uint8)

    # Instruction band.
    nb = guide_band_rows(dev)
    dist_cm = view.distance_mm / 10.0
    mode = "cross-eyed" if view.mode == "cross" else "parallel"
    head = label or "DuoVision align"
    left_lines = [f"{head} · fold {geom.hinge_deg:g}°",
                  f"eyes {dist_cm:g} cm, centred, level with hinge middle · {mode}"]
    band = _render_band(geom, view, left_lines, list(ALIGN_INSTRUCTIONS))
    out[:nb] = band[..., None]
    return out


# ----------------------------------------------------------------- fold guide
def fold_guide_extent(geom: FoldGeometry, view: ViewProfile) -> Tuple[float, float]:
    """(half_w, half_h) of the fold-guide picture frame, mm in the reference plane,
    centred on the cyclopean axis; the largest frame that fits below the band."""
    up, down, _side = _plane_limits(geom, view)
    half_h = max(8.0, min(up, down) - 2.0)
    # Side limit: the frame stays inside the central FOLD_SAFE_WIDTH_FRAC of the display width.
    e = view.cyclopean_eye()
    z_p = _guide_z(view)
    u = FOLD_SAFE_WIDTH_FRAC * geom.half_width_mm
    side = u * geom.s * (e[2] - z_p) / (e[2] - u * geom.c) - abs(e[0])
    half_w = max(8.0, min(side, 1.7 * half_h))
    return float(half_w), float(half_h)


def fold_guide_target(half_w: float, half_h: float, px_per_mm: float, hinge_deg: float) -> np.ndarray:
    """The undistorted fold-guide picture, (h, w, 3) uint8.

    Covers x in [-half_w - m, half_w + m], y in [-half_h - m, half_h + m] mm
    (m = FOLD_MARGIN_MM; row 0 is the top, +y up) at `px_per_mm`.  Bitmap pixel
    (col, row) is centred at x = (col + 0.5) / px_per_mm - half_w - m,
    y = half_h + m - (row + 0.5) / px_per_mm.

    White frame, two-row checkerboard bands along the top and bottom (a square
    boundary on the hinge), a round face in a white ring on the hinge, "FOLD"
    on the left and the angle on the right.  Straight edges crossing the hinge
    far from eye level and a circle on the hinge are the shapes whose
    distortion shows a wrong fold angle most clearly.
    """
    m = FOLD_MARGIN_MM
    ss = 2                                       # supersampling for anti-aliasing
    w = max(8, int(math.ceil((2 * half_w + 2 * m) * px_per_mm)))
    h = max(8, int(math.ceil((2 * half_h + 2 * m) * px_per_mm)))
    k = px_per_mm * ss
    img = Image.new("RGB", (w * ss, h * ss), FOLD_BG)
    d = ImageDraw.Draw(img)

    def X(x):
        return (x + half_w + m) * k

    def Y(y):
        return (half_h + m - y) * k

    def rect(x0, y0, x1, y1, fill):              # mm, any corner order
        d.rectangle([X(min(x0, x1)), Y(max(y0, y1)), X(max(x0, x1)) - 1, Y(min(y0, y1)) - 1], fill=fill)

    def disc(cx, cy, rx, ry, fill):
        d.ellipse([X(cx - rx), Y(cy + ry), X(cx + rx), Y(cy - ry)], fill=fill)

    t = min(1.8, 0.08 * half_h)                  # frame thickness
    rect(-half_w, -half_h, half_w, half_h, FOLD_FRAME)
    rect(-half_w + t, -half_h + t, half_w - t, half_h - t, FOLD_PAPER)

    # Checkerboard bands: two rows of squares, a boundary on the hinge (x = 0).
    xi = half_w - t
    n = max(1, int(round(xi / min(6.0, max(1.5, 0.14 * half_h)))))
    q = xi / n                                   # a whole number of squares on each side of the hinge
    for band_top in (half_h - t, -half_h + t + 2 * q):
        for r in range(2):
            y1 = band_top - r * q
            for c in range(-n, n):
                x0, x1 = max(c * q, -xi), min((c + 1) * q, xi)
                if x1 <= x0:
                    continue
                rect(x0, y1 - q, x1, y1, FOLD_FRAME if (c + r) % 2 == 0 else FOLD_CHECK)

    # Round face in a white ring, centred on the hinge at eye level.
    free = half_h - t - 2 * q                    # half height left between the bands
    ring = 0.88 * free
    face = ring - min(2.4, 0.12 * ring)
    disc(0, 0, ring, ring, FOLD_FRAME)
    disc(0, 0, ring - min(1.0, 0.05 * ring), ring - min(1.0, 0.05 * ring), FOLD_PAPER)
    disc(0, 0, face, face, FOLD_FACE)
    for sx in (-1, 1):
        disc(sx * 0.36 * face, 0.26 * face, 0.10 * face, 0.16 * face, FOLD_INK)
    sm = 0.60 * face
    d.arc([X(-sm), Y(sm), X(sm), Y(-sm)], 25, 155, fill=FOLD_INK, width=max(1, int(round(0.09 * face * k))))

    # "FOLD" on the left half, the angle on the right half.
    room = xi - ring - 0.12 * xi                 # width available on each side of the face
    if room > 4.0:
        words = ("FOLD", f"{hinge_deg:g}°")
        font = _fit_font(words, int(round(0.62 * free * k)), room * k)
        asc, desc = font.getmetrics()
        for word, sx in zip(words, (-1, 1)):
            tw = _text_width(word, font)
            cx = sx * (ring + (xi - ring) / 2.0)
            d.text((X(cx) - tw / 2.0, Y(0) - (asc + desc) / 2.0), word, fill=FOLD_FRAME, font=font)

    img = img.resize((w, h), Image.LANCZOS)
    return np.asarray(img, dtype=np.uint8)


def fold_guide_image(geom: FoldGeometry, view: ViewProfile, label: str = "") -> np.ndarray:
    """Fold-angle guide (H, W, 3) uint8: a picture that only looks right when folded.

    The picture of ``fold_guide_target`` lies in the frontal plane
    z = z_background.  Every display pixel shows the colour of the plane point
    its cyclopean ray passes through (bilinear), so from the design viewpoint at
    the encoded hinge angle the picture looks flat and undistorted.  Shown on a
    flatter display it looks tall in the middle (fold more); folded too far it
    looks pinched in the middle (open up).  The band carries the instructions.
    """
    from scipy import ndimage

    dev = geom.device
    h, w = dev.height_px, dev.width_px
    qx, qy = _plane_coords(geom, view)
    half_w, half_h = fold_guide_extent(geom, view)
    ppm = FOLD_TARGET_OVERSAMPLE / dev.mm_per_px
    target = fold_guide_target(half_w, half_h, ppm, geom.hinge_deg)
    cols = (qx + half_w + FOLD_MARGIN_MM) * ppm - 0.5
    rows = (half_h + FOLD_MARGIN_MM - qy) * ppm - 0.5
    out = np.empty((h, w, 3), dtype=np.uint8)
    for c in range(3):
        ch = ndimage.map_coordinates(target[..., c].astype(np.float32), [rows, cols], order=1, mode="nearest")
        out[..., c] = np.clip(np.rint(ch), 0, 255).astype(np.uint8)

    nb = guide_band_rows(dev)
    dist_cm = view.distance_mm / 10.0
    head = label or "DuoVision fold guide"
    left_lines = [f"{head} · fold {geom.hinge_deg:g}°",
                  f"eyes {dist_cm:g} cm, centred, level with hinge middle"]
    band = _render_band(geom, view, left_lines, list(FOLD_INSTRUCTIONS), dots=False)
    out[:nb] = band[..., None]
    return out


def default_label(scene: str, hinge_deg: float, view: ViewProfile, style: str | None = None) -> str:
    mode = "cross-eyed" if view.mode == "cross" else "parallel"
    parts = ["DuoVision", scene, f"{hinge_deg:g}°", f"{view.distance_mm / 10.0:g} cm", mode]
    if style:
        parts.insert(2, style)
    return " · ".join(parts)
