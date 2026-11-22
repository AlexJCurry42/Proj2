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
* ``fold_guide_image``: an anamorphic picture (fine double frame, parallel
  rules, a graduated dial with the angle) for finding the fold angle by eye.
  The flat picture is pre-distorted so that it only looks right (straight
  rules, a perfect circle) from the design viewpoint when the phone is folded to
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
FOLD_SAFE_WIDTH_FRAC = 0.78      # iOS hides ~9% of the Duo's inner display behind its status strip on
                                 # one side (the viewer mirrors it on the other): keep the frame inside
FOLD_MARGIN_MM = 6.0             # background kept around the picture frame in the target bitmap
FOLD_TARGET_OVERSAMPLE = 1.5     # target bitmap pixels per display pixel
FOLD_BG = (0, 0, 0)              # black (the viewer inverts the cards to white in light mode)
FOLD_INK = (240, 243, 248)       # white
FOLD_INK_DIM = (124, 131, 142)
FOLD_ACCENT = (76, 141, 255)     # blue, the viewer's accent


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
    img[:] = FOLD_BG                             # same palette as the fold guide: white and blue on black

    def over(cov, colour):
        a = _quantize(cov)[..., None]
        img[:] = img * (1.0 - a) + np.asarray(colour, dtype=np.float64) * a

    # Grid (10 mm in the reference plane).
    fx = qx - np.round(qx / grid) * grid
    fy = qy - np.round(qy / grid) * grid
    cov = np.maximum(_line_cov(fx, g_x, hw_grid), _line_cov(fy, g_y, hw_grid))
    over(cov, (52, 56, 62))

    # Crosshair through the centre, across the whole display.
    cov = np.maximum(_line_cov(qx, g_x, hw_main), _line_cov(qy, g_y, hw_main))
    over(cov, FOLD_ACCENT)

    # Inscribed circle.
    rho = np.hypot(qx, qy)
    with np.errstate(invalid="ignore", divide="ignore"):
        nx = np.where(rho > 0, qx / rho, 0.0)
        ny = np.where(rho > 0, qy / rho, 0.0)
    g_r = np.hypot(nx * dqx_di + ny * dqy_di, nx * dqx_dj + ny * dqy_dj)
    over(_line_cov(rho - half, g_r, hw_main), (168, 174, 184))

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
    over(cov, FOLD_INK)

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


def fold_guide_layout(half_w: float, half_h: float) -> dict:
    """Geometry (mm in the reference plane) of the fold-guide picture's elements.

    frame: outer rule at (+-half_w, +-half_h), stroke `frame_w`; inner rule inset by
    `inner_inset`.  rules_y: heights of the parallel rules that run the full width
    near the top and bottom.  ring_r: radius of the dial's ring, centred on the hinge
    at eye level, with `ticks` graduations between tick_r0 (long: tick_r0_long) and
    tick_r1, and a fine inner circle at inner_r.
    """
    frame_w = min(0.45, 0.02 * half_h)
    inner_inset = min(1.6, 0.06 * half_h)
    rule_gap = min(1.4, 0.05 * half_h)
    rule_top = half_h - min(4.6, 0.14 * half_h)
    rules = [rule_top - k * rule_gap for k in range(3)]
    ring_r = min(0.80 * (rules[-1] - 0.05 * half_h), 0.62 * half_w)
    return {
        "frame_w": frame_w, "inner_inset": inner_inset, "inner_w": 0.45 * frame_w,
        "rules_y": [y for y in rules] + [-y for y in rules], "rule_w": 0.45 * frame_w,
        "ring_r": ring_r, "ring_w": 1.25 * frame_w, "inner_r": 0.66 * ring_r,
        "ticks": 72, "tick_r1": ring_r - 0.045 * ring_r, "tick_r0": ring_r - 0.10 * ring_r,
        "tick_r0_long": ring_r - 0.17 * ring_r, "tick_w": 0.45 * frame_w,
    }


def _card_font(size_px: int):
    """Font for the fold-guide lettering: Pillow's bundled face, so the card looks the
    same wherever it is rendered; falls back to the installed fonts on old Pillow."""
    size_px = max(6, int(size_px))
    try:
        return ImageFont.load_default(size=size_px)
    except TypeError:  # pragma: no cover - Pillow < 10.1
        return _font(size_px)


def _spaced_text(draw, cx: float, cy: float, text: str, font, tracking: float, fill) -> None:
    """Draw `text` centred on (cx, cy) with `tracking` extra pixels between letters."""
    widths = [font.getlength(ch) for ch in text]
    total = sum(widths) + tracking * (len(text) - 1)
    l, t, r, b = font.getbbox(text)
    x = cx - total / 2.0
    y = cy - (t + b) / 2.0
    for ch, w in zip(text, widths):
        draw.text((x, y), ch, fill=fill, font=font)
        x += w + tracking


def fold_guide_target(half_w: float, half_h: float, px_per_mm: float, hinge_deg: float) -> np.ndarray:
    """The undistorted fold-guide picture, (h, w, 3) uint8.

    Covers x in [-half_w - m, half_w + m], y in [-half_h - m, half_h + m] mm
    (m = FOLD_MARGIN_MM; row 0 is the top, +y up) at `px_per_mm`.  Bitmap pixel
    (col, row) is centred at x = (col + 0.5) / px_per_mm - half_w - m,
    y = half_h + m - (row + 0.5) / px_per_mm.

    A fine double frame, three parallel rules across the full width at the top
    and bottom, and a graduated dial on the hinge with the angle at its centre
    (see fold_guide_layout).  Long straight rules crossing the hinge far from eye
    level and a circle on the hinge are the shapes whose distortion shows a wrong
    fold angle most clearly; everything is drawn as light hairlines on black.
    """
    m = FOLD_MARGIN_MM
    ss = 3                                       # supersampling for anti-aliasing
    w = max(8, int(math.ceil((2 * half_w + 2 * m) * px_per_mm)))
    h = max(8, int(math.ceil((2 * half_h + 2 * m) * px_per_mm)))
    k = px_per_mm * ss
    lay = fold_guide_layout(half_w, half_h)
    img = Image.new("RGB", (w * ss, h * ss), FOLD_BG)
    d = ImageDraw.Draw(img)

    def X(x):
        return (x + half_w + m) * k

    def Y(y):
        return (half_h + m - y) * k

    def px(mm):                                   # stroke width in bitmap pixels (>= 1)
        return max(1, int(round(mm * k)))

    def hline(y, x0, x1, width_mm, fill):
        t = px(width_mm)
        d.rectangle([X(x0), Y(y) - t / 2.0, X(x1), Y(y) + t / 2.0 - 1], fill=fill)

    def vline(x, y0, y1, width_mm, fill):
        t = px(width_mm)
        d.rectangle([X(x) - t / 2.0, Y(y1), X(x) + t / 2.0 - 1, Y(y0)], fill=fill)

    def box(hw, hh, width_mm, fill):
        e = width_mm / 2.0                        # square corners
        for sgn in (-1, 1):
            hline(sgn * hh, -hw - e, hw + e, width_mm, fill)
            vline(sgn * hw, -hh - e, hh + e, width_mm, fill)

    def circle(r, width_mm, fill):
        d.ellipse([X(-r), Y(r), X(r), Y(-r)], outline=fill, width=px(width_mm))

    # Double frame.
    box(half_w, half_h, lay["frame_w"], FOLD_INK)
    box(half_w - lay["inner_inset"], half_h - lay["inner_inset"], lay["inner_w"], FOLD_INK_DIM)

    # Parallel rules across the full width, top and bottom; the middle one in the accent blue.
    xr = half_w - 2.2 * lay["inner_inset"]
    for n, y in enumerate(lay["rules_y"]):
        hline(y, -xr, xr, lay["rule_w"], FOLD_ACCENT if n % 3 == 1 else FOLD_INK_DIM)

    # Dial on the hinge: accent ring, graduations, fine inner circle.
    circle(lay["ring_r"], lay["ring_w"], FOLD_ACCENT)
    circle(lay["inner_r"], lay["inner_w"], FOLD_INK_DIM)
    for n in range(lay["ticks"]):
        ang = 2.0 * math.pi * n / lay["ticks"]
        major = n % (lay["ticks"] // 12) == 0
        r0 = lay["tick_r0_long"] if major else lay["tick_r0"]
        ca, sa = math.cos(ang), math.sin(ang)
        d.line([X(r0 * ca), Y(r0 * sa), X(lay["tick_r1"] * ca), Y(lay["tick_r1"] * sa)],
               fill=FOLD_INK if major else FOLD_INK_DIM, width=px(lay["tick_w"] * (1.6 if major else 1.0)))

    # The angle at the centre, with fine captions above and below.
    r_in = lay["inner_r"]
    num = f"{hinge_deg:g}"
    font = _card_font(int(round(0.62 * r_in * k)))
    while font.size > 8 and font.getlength(num) > 1.25 * r_in * k:
        font = _card_font(font.size - 2)
    l, t, r, b = font.getbbox(num)
    nx = X(0) - (l + r) / 2.0
    ny = Y(0) - (t + b) / 2.0
    d.text((nx, ny), num, fill=FOLD_INK, font=font)
    deg_r = 0.11 * (b - t)                        # the degree sign: a small ring, top right of the digits
    dx, dy = nx + r + 1.9 * deg_r, ny + t + 1.1 * deg_r
    d.ellipse([dx - deg_r, dy - deg_r, dx + deg_r, dy + deg_r], outline=FOLD_INK,
              width=max(1, int(round(0.34 * deg_r))))
    cap = _card_font(int(round(0.105 * r_in * k)))
    if cap.size >= 9:
        _spaced_text(d, X(0), Y(0.60 * r_in), "FOLD", cap, 0.42 * cap.size, FOLD_ACCENT)
        _spaced_text(d, X(0), Y(-0.60 * r_in), "DEGREES", cap, 0.42 * cap.size, FOLD_INK_DIM)

    # Small captions beside the dial.
    side = half_w - lay["inner_inset"] - lay["ring_r"]
    side_font = _card_font(int(round(min(0.045 * half_h, 0.11 * side) * k)))
    if side_font.size >= 9 and side > 6.0:
        cx = lay["ring_r"] + side / 2.0
        _spaced_text(d, X(-cx), Y(0), "DUOVISION", side_font, 0.40 * side_font.size, FOLD_INK_DIM)
        _spaced_text(d, X(cx), Y(0), "FOLD GUIDE", side_font, 0.40 * side_font.size, FOLD_INK_DIM)

    img = img.resize((w, h), Image.LANCZOS)
    return np.asarray(img, dtype=np.uint8)


def fold_guide_image(geom: FoldGeometry, view: ViewProfile, label: str = "") -> np.ndarray:
    """Fold-angle guide (H, W, 3) uint8: a picture that only looks right when folded.

    The picture of ``fold_guide_target`` lies in the frontal plane
    z = z_background.  Every display pixel shows the colour of the plane point
    its cyclopean ray passes through (bilinear), so from the design viewpoint at
    the encoded hinge angle the picture looks flat and undistorted.  Shown on a
    flatter display it looks tall in the middle (fold more); folded too far it
    looks pinched in the middle (open up).  The card carries no instructions (the
    viewer's setup guide does); the reserved band rows are left black.  `label`
    is accepted for symmetry with alignment_image and is not drawn.
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
    out[:guide_band_rows(dev)] = np.array(FOLD_BG, dtype=np.uint8)
    return out


def default_label(scene: str, hinge_deg: float, view: ViewProfile, style: str | None = None) -> str:
    mode = "cross-eyed" if view.mode == "cross" else "parallel"
    parts = ["DuoVision", scene, f"{hinge_deg:g}°", f"{view.distance_mm / 10.0:g} cm", mode]
    if style:
        parts.insert(2, style)
    return " · ".join(parts)
