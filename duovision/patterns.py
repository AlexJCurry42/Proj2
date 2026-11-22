"""Colour patterns for stereogram chains.

``colorize`` gives every pixel the colour of its chain's root. The colour is a pure
function of the root's pixel coordinates ``(ri, rj)`` (and the seed), so every member
of a chain matches exactly. Colours are evaluated only at root pixels and gathered.

Styles
  bw       black/white random dots on ``dot_px`` x ``dot_px`` cells (~50 % white)
  color    random colours from a bright palette, one per cell
  texture  smooth procedural textures with fine detail: ``marble``, ``waves``, ``leaves``
"""
from __future__ import annotations

import numpy as np

STYLES = ("bw", "color", "texture")
TEXTURES = ("marble", "waves", "leaves")

# Bright, mutually distinct colours (sRGB).
PALETTE = np.array(
    [
        (230, 25, 75),    # red
        (60, 180, 75),    # green
        (255, 225, 25),   # yellow
        (0, 130, 200),    # blue
        (245, 130, 48),   # orange
        (145, 30, 180),   # purple
        (70, 240, 240),   # cyan
        (240, 50, 230),   # magenta
        (210, 245, 60),   # lime
        (250, 190, 212),  # pink
        (0, 128, 128),    # teal
        (220, 190, 255),  # lavender
        (170, 110, 40),   # brown
        (255, 250, 200),  # cream
        (128, 0, 0),      # maroon
        (255, 255, 255),  # white
    ],
    dtype=np.uint8,
)

_U = np.uint64
_K1 = _U(0x9E3779B97F4A7C15)
_K2 = _U(0xC2B2AE3D27D4EB4F)
_K3 = _U(0x165667B19E3779F9)
_M1 = _U(0xBF58476D1CE4E5B9)
_M2 = _U(0x94D049BB133111EB)


# ----------------------------------------------------------------- hashing
def _mix(h: np.ndarray) -> np.ndarray:
    """splitmix64 finaliser (uint64 arrays, wrapping arithmetic)."""
    h = h ^ (h >> _U(30))
    h = h * _M1
    h = h ^ (h >> _U(27))
    h = h * _M2
    return h ^ (h >> _U(31))


def _hash2(a, b, seed: int) -> np.ndarray:
    """uint64 hash of integer arrays a, b (may be negative) and a seed."""
    a = np.asarray(a, dtype=np.int64).astype(np.uint64)
    b = np.asarray(b, dtype=np.int64).astype(np.uint64)
    s = _mix(np.array([seed & 0xFFFFFFFFFFFFFFFF], dtype=np.uint64) + _K3)
    return _mix(_mix(a * _K1 + s) ^ (b * _K2))


def _unit(h: np.ndarray) -> np.ndarray:
    """Hash -> float32 in [0, 1)."""
    return (h >> _U(40)).astype(np.float32) * np.float32(1.0 / (1 << 24))


# ------------------------------------------------------------------- noise
def value_noise(x, y, seed: int) -> np.ndarray:
    """Smooth 2D value noise in [0, 1) with unit lattice spacing (float32)."""
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    x0 = np.floor(x)
    y0 = np.floor(y)
    fx = x - x0
    fy = y - y0
    fx = fx * fx * (3.0 - 2.0 * fx)
    fy = fy * fy * (3.0 - 2.0 * fy)
    xi = x0.astype(np.int64)
    yi = y0.astype(np.int64)
    v00 = _unit(_hash2(xi, yi, seed))
    v10 = _unit(_hash2(xi + 1, yi, seed))
    v01 = _unit(_hash2(xi, yi + 1, seed))
    v11 = _unit(_hash2(xi + 1, yi + 1, seed))
    a = v00 + (v10 - v00) * fx
    b = v01 + (v11 - v01) * fx
    return a + (b - a) * fy


def fbm(x, y, seed: int, octaves: int = 5, lacunarity: float = 2.0, gain: float = 0.5) -> np.ndarray:
    """Fractal Brownian motion of value noise, normalised to [0, 1)."""
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    total = np.zeros(np.broadcast(x, y).shape, dtype=np.float32)
    amp, freq, norm = 1.0, 1.0, 0.0
    for o in range(octaves):
        # rotate/offset each octave a little to hide lattice alignment
        total += np.float32(amp) * value_noise(x * freq + 17.3 * o, y * freq - 9.1 * o, seed + 101 * o)
        norm += amp
        amp *= gain
        freq *= lacunarity
    return total / np.float32(norm)


def _cosine_palette(t, phase=(0.0, 0.33, 0.67)) -> np.ndarray:
    """Rainbow-like cosine palette (Quilez): 0.5 + 0.5 cos(2 pi (t + d)), (n, 3) in [0, 1]."""
    t = np.asarray(t, dtype=np.float32)[..., None]
    d = np.asarray(phase, dtype=np.float32)
    return 0.5 + 0.5 * np.cos(np.float32(2 * np.pi) * (t + d))


def _to_u8(rgb: np.ndarray) -> np.ndarray:
    return np.clip(rgb * 255.0 + 0.5, 0, 255).astype(np.uint8)


# ----------------------------------------------------------------- styles
def _bw(ri, rj, seed, dot_px):
    h = _hash2(ri // dot_px, rj // dot_px, seed)
    white = ((h >> _U(33)) & _U(1)).astype(bool)
    out = np.zeros(ri.shape + (3,), dtype=np.uint8)
    out[white] = 255
    return out


def _color(ri, rj, seed, dot_px):
    h = _hash2(ri // dot_px, rj // dot_px, seed + 0x5151)
    k = ((h >> _U(32)) % _U(len(PALETTE))).astype(np.intp)
    return PALETTE[k]


def _detail(u, v, seed):
    """Fine-scale brightness modulation at about one dot cell (helps fusion)."""
    return 0.72 + 0.5 * fbm(u * 1.3, v * 1.3, seed, octaves=2)


def _marble(u, v, seed):
    turb = fbm(u / 12.0, v / 12.0, seed + 1, octaves=5)
    t = (0.9 * u + 0.35 * v) / 6.0 + 3.5 * turb
    vein = 0.5 + 0.5 * np.sin(np.float32(2 * np.pi) * t)
    vein = np.sqrt(vein)
    hue = 1.6 * fbm(u / 25.0, v / 25.0, seed + 2, octaves=3) + 0.25 * vein
    rgb = _cosine_palette(hue) * (0.35 + 0.65 * vein)[..., None]
    return rgb * _detail(u, v, seed + 3)[..., None]


def _waves(u, v, seed):
    w1 = fbm(u / 10.0, v / 10.0, seed + 11, octaves=4)
    w2 = fbm(u / 7.0 + 5.2, v / 7.0 - 1.3, seed + 12, octaves=4)
    tau = np.float32(2 * np.pi)
    a = np.sin(tau * (u / 6.0 + v / 17.0 + 2.2 * w1))
    b = np.sin(tau * (v / 4.5 - u / 13.0 + 1.8 * w2))
    t = 0.5 + 0.25 * (a + b)
    hue = 0.8 * t + 1.2 * fbm(u / 30.0, v / 30.0, seed + 13, octaves=2)
    shade = 0.3 + 0.7 * (0.5 + 0.5 * a * b)
    rgb = _cosine_palette(hue, phase=(0.1, 0.4, 0.75)) * shade[..., None]
    return rgb * _detail(u, v, seed + 14)[..., None]


_LEAF_COLOURS = np.array(
    [
        (0.20, 0.55, 0.15), (0.45, 0.70, 0.10), (0.85, 0.75, 0.10), (0.95, 0.50, 0.08),
        (0.80, 0.20, 0.10), (0.55, 0.10, 0.25), (0.10, 0.45, 0.35), (0.98, 0.85, 0.45),
    ],
    dtype=np.float32,
)


def _leaves(u, v, seed, cell=4.0):
    """Worley-cell 'leaves': one colour per cell, darker veins along cell borders."""
    x = u / cell + 0.35 * fbm(u / 9.0, v / 9.0, seed + 21, octaves=3)
    y = v / cell + 0.35 * fbm(u / 9.0 + 3.7, v / 9.0, seed + 22, octaves=3)
    cx = np.floor(x).astype(np.int64)
    cy = np.floor(y).astype(np.int64)
    f1 = np.full(x.shape, np.inf, dtype=np.float32)
    f2 = np.full(x.shape, np.inf, dtype=np.float32)
    best = np.zeros(x.shape, dtype=np.uint64)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            nx, ny = cx + dx, cy + dy
            h = _hash2(nx, ny, seed + 23)
            px = nx + 0.1 + 0.8 * _unit(h)
            py = ny + 0.1 + 0.8 * _unit(_mix(h))
            d = (x - px) ** 2 + (y - py) ** 2
            closer = d < f1
            f2 = np.where(closer, f1, np.minimum(f2, d))
            f1 = np.where(closer, d, f1)
            best = np.where(closer, h, best)
    edge = np.sqrt(f2) - np.sqrt(f1)  # 0 on cell borders
    k = ((best >> _U(20)) % _U(len(_LEAF_COLOURS))).astype(np.intp)
    base = _LEAF_COLOURS[k]
    vein = np.clip(edge / 0.18, 0.0, 1.0) ** 0.6
    tint = 0.8 + 0.4 * _unit(best >> _U(7))
    rgb = base * (0.25 + 0.75 * vein * tint)[..., None]
    return rgb * _detail(u, v, seed + 24)[..., None]


_TEXTURE_FNS = {"marble": _marble, "waves": _waves, "leaves": _leaves}


def texture_rgb(ri, rj, *, seed=0, dot_px=5, texture="marble") -> np.ndarray:
    """Evaluate a procedural texture at pixel coordinates -> uint8 (n, 3)."""
    try:
        fn = _TEXTURE_FNS[texture]
    except KeyError:
        raise ValueError(f"unknown texture {texture!r}; choose from {TEXTURES}") from None
    s = np.float32(max(1, int(dot_px)))
    u = np.asarray(ri, dtype=np.float32) / s
    v = np.asarray(rj, dtype=np.float32) / s
    return _to_u8(fn(u, v, int(seed)))


# ---------------------------------------------------------------- public
def colorize(roots, device, style, *, seed=0, dot_px=5, texture="marble") -> np.ndarray:
    """Colour every pixel by its chain root -> uint8 (H, W, 3).

    ``roots`` holds flat root indices ``rj * W + ri`` (from ``resolve_roots``).
    """
    if style not in STYLES:
        raise ValueError(f"unknown style {style!r}; choose from {STYLES}")
    dot_px = max(1, int(dot_px))
    H, W = device.height_px, device.width_px
    roots = np.asarray(roots)
    if roots.shape != (H, W):
        raise ValueError(f"roots shape {roots.shape} != {(H, W)}")
    flat = roots.ravel().astype(np.int64, copy=False)
    n = flat.size
    if flat.min() < 0 or flat.max() >= n:
        raise ValueError("root indices out of range")
    is_root = flat == np.arange(n)
    if is_root[flat].all():
        idx = np.nonzero(is_root)[0]
    else:  # labels that are not self-rooted: colour each distinct value
        idx = np.unique(flat)
    rj, ri = np.divmod(idx, W)

    if style == "bw":
        cols = _bw(ri, rj, int(seed), dot_px)
    elif style == "color":
        cols = _color(ri, rj, int(seed), dot_px)
    else:
        cols = texture_rgb(ri, rj, seed=seed, dot_px=dot_px, texture=texture)

    table = np.zeros((n, 3), dtype=np.uint8)
    table[idx] = cols
    return table[flat].reshape(H, W, 3)
