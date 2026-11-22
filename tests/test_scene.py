import time

import numpy as np
import pytest

from duovision.config import DEFAULT_DEVICE, DEFAULT_VIEW, guide_band_rows
from duovision.geometry import FoldGeometry
from duovision.scene import (
    GRID_MM,
    SCENE_NAMES,
    HeightScene,
    build_scene,
    first_hit,
    scene_layout,
)

DEV = DEFAULT_DEVICE.scaled(0.25)
HINGES = (100, 120, 140)
CROSS = DEFAULT_VIEW
PARALLEL = DEFAULT_VIEW.with_(mode="parallel")

_cache = {}


def scene(name, hinge=120, view=CROSS, **kw):
    key = (name, hinge, view, tuple(sorted(kw.items())))
    if key not in _cache:
        _cache[key] = (build_scene(name, FoldGeometry(DEV, hinge), view, **kw), FoldGeometry(DEV, hinge))
    return _cache[key]


def footprint_grid(geom, step=0.25):
    xs = np.arange(-geom.x_extent_mm, geom.x_extent_mm + 1e-9, step)
    ys = np.arange(-geom.height_mm / 2, geom.height_mm / 2 + 1e-9, step)
    return np.meshgrid(xs, ys)


def node_grid(sc):
    """World coordinates of the baked grid nodes (z_of is exact there)."""
    x0, x1, y0, y1 = sc.grid_bounds
    nx = int(round((x1 - x0) / GRID_MM)) + 1
    ny = int(round((y1 - y0) / GRID_MM)) + 1
    return np.meshgrid(x0 + GRID_MM * np.arange(nx), y0 + GRID_MM * np.arange(ny))


# ------------------------------------------------------------ basic contract
def test_scene_names_and_unknown():
    assert SCENE_NAMES == ("plaque", "cube", "sphere", "text", "torus", "pyramid", "heart", "pawn")
    with pytest.raises(ValueError):
        build_scene("teapot", FoldGeometry(DEV, 120), CROSS)


@pytest.mark.parametrize("name", SCENE_NAMES)
def test_z_of_shapes_and_types(name):
    sc, _ = scene(name)
    assert isinstance(sc, HeightScene) and sc.name == name
    assert sc.z_of(0.0, 0.0).shape == ()
    x = np.linspace(-30, 30, 24).reshape(2, 3, 4)
    assert sc.z_of(x, 1.0).shape == (2, 3, 4)
    assert sc.object_mask(x, x).shape == (2, 3, 4)
    assert sc.object_mask(x, x).dtype == bool


# ------------------------------------------------- requirement 1 (cross mode)
@pytest.mark.parametrize("hinge", HINGES)
@pytest.mark.parametrize("name", SCENE_NAMES)
def test_cross_bounds_and_background(name, hinge):
    sc, g = scene(name, hinge)
    X, Y = footprint_grid(g)
    z = sc.z_of(X, Y)
    m = sc.object_mask(X, Y)
    bg = CROSS.z_background + 0.25 * np.abs(X) * g.c / g.s
    assert np.all(z >= g.panel_z(X) + 3.0)
    assert np.all(z <= CROSS.z_near + 0.5)
    assert np.all(z >= bg - 1e-9)
    assert m.any() and not m.all()
    assert np.allclose(z[~m], bg[~m], rtol=0, atol=1e-9)  # background is the formula
    # march bounds are the footprint min / max
    assert sc.z_min == pytest.approx(z.min(), abs=0.1)
    assert sc.z_max == pytest.approx(z.max(), abs=0.2)  # sampling of sharp apices
    assert sc.z_min <= z.min() + 1e-9 and sc.z_max >= z.max() - 1e-9
    # objects: front surfaces between z_bg + 2 and z_near (checked at the exact grid nodes)
    NX, NY = node_grid(sc)
    zn = sc.z_of(NX, NY)
    mn = sc.object_mask(NX, NY)
    bgn = sc.background(NX)
    assert np.all(zn[mn] >= bgn[mn] + 2.0 - 1e-9)
    if name != "plaque":
        assert zn.max() == pytest.approx(CROSS.z_near, abs=1e-6)
    # placement: inside the edge margins, below the guide band with 10 mm margin
    lay = scene_layout(g)
    xo, yo = NX[mn], NY[mn]
    assert np.abs(xo).max() <= g.x_extent_mm - 15.0 + GRID_MM
    band_y = g.height_mm / 2 - guide_band_rows(DEV) * DEV.mm_per_px
    assert yo.max() <= band_y - 10.0 + GRID_MM
    assert yo.min() >= -g.height_mm / 2 + 10.0 - GRID_MM
    assert abs((yo.max() + yo.min()) / 2 - lay["y_centre"]) < 1.0      # vertically centred
    assert abs(xo.max() + xo.min()) < 1.0                              # centred on the hinge axis


def test_flat_background_option():
    sc, g = scene("sphere", 120, bg_vee=0.0)
    X, Y = footprint_grid(g, 1.0)
    m = sc.object_mask(X, Y)
    assert np.all(sc.z_of(X, Y)[~m] == CROSS.z_background)


# ---------------------------------------------------------- requirement 2
@pytest.mark.parametrize("hinge", HINGES)
def test_plaque_planar_and_clears_background(hinge):
    sc, g = scene("plaque", hinge)
    zp = CROSS.z_background + 12.0
    x_half, y0, y1 = sc.meta["object_extent"]
    assert 2 * x_half == pytest.approx(min(90.0, 2 * (g.x_extent_mm - 15)), abs=0.11)
    assert y1 - y0 == pytest.approx(55.0, abs=0.11)
    rng = np.random.default_rng(1)
    xs = rng.uniform(-x_half + 0.11, x_half - 0.11, 20000)
    ys = rng.uniform(y0 + 0.11, y1 - 0.11, 20000)
    z = sc.z_of(xs, ys)
    assert np.all(z == zp)                                   # exactly planar
    assert np.all(sc.object_mask(xs, ys))
    # edges and corners sit above the V background
    t = np.linspace(-1, 1, 101)
    ex = np.concatenate([x_half * t, x_half * t, np.full(101, -x_half), np.full(101, x_half)])
    ey = np.concatenate([np.full(101, y0), np.full(101, y1), y0 + (y1 - y0) * (t + 1) / 2,
                         y0 + (y1 - y0) * (t + 1) / 2])
    assert np.all(sc.z_of(ex, ey) == zp)
    assert np.all(sc.background(ex) <= zp - 2.0)
    # just outside: background
    assert not sc.object_mask(x_half + 0.2, 0.5 * (y0 + y1))


# ---------------------------------------------------------- requirement 3
def _brute_force(sc, eye, tg, dz_step=0.02):
    d = tg - eye
    z_hi, z_lo = sc.z_max + 1e-3, sc.z_min - 1e-3
    zs = np.arange(z_hi, z_lo - dz_step, -dz_step)
    t = (zs[:, None] - eye[2]) / d[None, :, 2]                       # (K, N)
    p = eye + t[..., None] * d
    inside = p[..., 2] <= sc.z_of(p[..., 0], p[..., 1])
    first = np.argmax(inside, axis=0)
    found = inside.any(axis=0)
    return p[first, np.arange(len(tg))], found


def _random_pixel_targets(g, n, seed):
    rng = np.random.default_rng(seed)
    band = guide_band_rows(g.device)
    i = rng.uniform(0, g.device.width_px - 1, n)
    j = rng.uniform(band, g.device.height_px - 1, n)
    return g.pixel_to_world(i, j)


@pytest.mark.parametrize("name", SCENE_NAMES)
@pytest.mark.parametrize("hinge", (100, 140))
def test_first_hit_accuracy(name, hinge):
    sc, g = scene(name, hinge)
    eye, _ = CROSS.eyes()
    tg = _random_pixel_targets(g, 1500, seed=hinge)
    pts, hit = first_hit(sc, eye, tg)
    assert hit.all()
    d = tg - eye
    # on the ray
    v = pts - eye
    cross = np.linalg.norm(np.cross(v, d), axis=1) / np.linalg.norm(d, axis=1)
    assert cross.max() < 1e-9
    assert np.all(np.einsum("ij,ij->i", v, d) > 0)
    # on the surface: bracket of 0.01 mm along the ray (outside before, inside after)
    u = d / np.linalg.norm(d, axis=1, keepdims=True)
    before = pts - 0.005 * u
    after = pts + 0.005 * u
    assert np.all(before[:, 2] > sc.z_of(before[:, 0], before[:, 1]))
    assert np.mean(after[:, 2] <= sc.z_of(after[:, 0], after[:, 1])) > 0.995
    resid = np.abs(pts[:, 2] - sc.z_of(pts[:, 0], pts[:, 1]))
    assert np.mean(resid < 0.01) > 0.99           # only near-vertical silhouette walls exceed
    # agrees with a brute-force 0.02 mm march
    ref, found = _brute_force(sc, eye, tg)
    assert found.all()
    dz = pts[:, 2] - ref[:, 2]
    agree = np.abs(dz) <= 0.02 + 0.01
    assert agree.mean() > 0.99
    # any disagreement is a sliver the 0.5 mm march stepped over (hit further along the ray)
    assert np.all(dz[~agree] < 0)


def test_first_hit_right_eye_parallel_and_shapes():
    for view in (CROSS, PARALLEL):
        sc, g = scene("torus", 120, view)
        _, eye = view.eyes()
        tg = _random_pixel_targets(g, 800, seed=7).reshape(20, 40, 3)
        pts, hit = first_hit(sc, eye, tg)
        assert pts.shape == (20, 40, 3) and hit.shape == (20, 40) and hit.all()
        ref, found = _brute_force(sc, eye, tg.reshape(-1, 3))
        assert np.mean(np.abs(pts.reshape(-1, 3)[:, 2] - ref[:, 2]) <= 0.03) > 0.99
        # per-ray eyes give the same answer
        eyes = np.broadcast_to(eye, tg.shape)
        pts2, hit2 = first_hit(sc, eyes, tg)
        assert np.array_equal(hit, hit2) and np.allclose(pts, pts2, atol=1e-9, equal_nan=True)


def test_first_hit_miss_is_nan():
    sc, _ = scene("sphere", 120)
    eye, _ = CROSS.eyes()
    tg = np.array([[0.0, 0.0, 400.0], [0.0, 0.0, 0.0]])    # away from the scene / toward it
    pts, hit = first_hit(sc, eye, tg)
    assert hit.tolist() == [False, True]
    assert np.all(np.isnan(pts[0])) and np.all(np.isfinite(pts[1]))


def test_first_hit_hits_background_exactly():
    sc, g = scene("pawn", 120)
    eye, _ = CROSS.eyes()
    tg = g.pixel_to_world(np.array([5.0, 600.0, 660.0]), np.array([60.0, 300.0, 400.0]))
    pts, hit = first_hit(sc, eye, tg)
    assert hit.all()
    assert not sc.object_mask(pts[:, 0], pts[:, 1]).any()
    assert np.allclose(pts[:, 2], sc.background(pts[:, 0]), atol=1e-9)


# ---------------------------------------------------------- requirement 4
@pytest.mark.parametrize("hinge", HINGES)
def test_torus_hole_shows_background(hinge):
    sc, g = scene("torus", hinge)
    yc = scene_layout(g)["y_centre"]
    r = np.linspace(0, 14, 50)
    a = np.linspace(0, 2 * np.pi, 50)
    R, A = np.meshgrid(r, a)
    x, y = R * np.cos(A), yc + R * np.sin(A)
    assert not sc.object_mask(x, y).any()
    assert np.array_equal(sc.z_of(x, y), sc.background(x))
    ring = sc.z_of(28.0 * np.cos(a), yc + 28.0 * np.sin(a))
    assert np.allclose(ring, CROSS.z_near, atol=0.05)


@pytest.mark.parametrize("hinge", HINGES)
def test_pyramid_apex_is_max(hinge):
    sc, g = scene("pyramid", hinge)
    NX, NY = node_grid(sc)
    z = sc.z_of(NX, NY)
    k = np.unravel_index(np.argmax(z), z.shape)
    yc = scene_layout(g)["y_centre"]
    assert abs(NX[k]) < 0.11 and abs(NY[k] - yc) < 0.11
    assert z[k] == pytest.approx(CROSS.z_near, abs=1e-6)
    assert z[k] == pytest.approx(sc.z_max)


@pytest.mark.parametrize("hinge", HINGES)
def test_text_mask_width(hinge):
    sc, g = scene("text", hinge)
    NX, NY = node_grid(sc)
    m = sc.object_mask(NX, NY)
    assert m.mean() > 0.05
    width = NX[m].max() - NX[m].min()
    expected = min(125.0, 2 * (g.x_extent_mm - 15.0))
    assert 0.95 * expected <= width <= expected + 0.2
    assert 8.0 < NY[m].max() - NY[m].min() < 25.0
    # letters have a flat face at z_near and a bevel toward their edges
    z = sc.z_of(NX, NY)[m]
    assert z.max() == pytest.approx(CROSS.z_near, abs=1e-6)
    assert np.mean(np.isclose(z, CROSS.z_near)) > 0.2
    assert z.min() < CROSS.z_near - 0.8


def test_custom_text():
    sc, g = scene("text", 120, text="Hi")
    NX, NY = node_grid(sc)
    w = np.ptp(NX[sc.object_mask(NX, NY)])
    assert 10 < w < 2 * (g.x_extent_mm - 15) + 0.2


@pytest.mark.parametrize("name", ["sphere", "cube", "heart", "pawn"])
def test_shapes_have_relief(name):
    sc, _ = scene(name)
    NX, NY = node_grid(sc)
    m = sc.object_mask(NX, NY)
    z = sc.z_of(NX, NY)[m]
    assert z.max() - z.min() > 8.0          # genuinely 3D, not a flat card


def test_pawn_is_tall_and_narrow_and_heart_is_wide():
    sc, _ = scene("pawn")
    w, h = sc.meta["size_mm"]
    assert h > 1.5 * w
    sc, _ = scene("heart")
    w, h = sc.meta["size_mm"]
    assert 0.8 < w / h < 1.3


# ---------------------------------------------------------- requirement 5
@pytest.mark.parametrize("hinge", HINGES)
@pytest.mark.parametrize("name", SCENE_NAMES)
def test_parallel_behind_display(name, hinge):
    sc, g = scene(name, hinge, PARALLEL)
    X, Y = footprint_grid(g, 0.5)
    z = sc.z_of(X, Y)
    m = sc.object_mask(X, Y)
    assert np.all(z <= g.panel_z(X) - 3.0)
    assert np.all(z >= sc.background(X) - 1e-9)
    assert m.any()
    assert np.allclose(z[~m], PARALLEL.z_background + 0.25 * np.abs(X[~m]) * g.c / g.s, rtol=0, atol=1e-9)
    assert PARALLEL.z_background < 0
    assert sc.z_max <= -3.0
    if name != "plaque":
        assert z.max() - PARALLEL.z_background > 15.0      # objects rise toward the viewer


# ---------------------------------------------------------- requirement 6
@pytest.mark.parametrize("name", SCENE_NAMES)
def test_deterministic(name):
    g = FoldGeometry(DEV, 120)
    a = build_scene(name, g, CROSS)
    b = build_scene(name, g, CROSS)
    X, Y = footprint_grid(g, 0.5)
    assert np.array_equal(a.z_of(X, Y), b.z_of(X, Y))
    assert (a.z_min, a.z_max) == (b.z_min, b.z_max)
    eye, _ = CROSS.eyes()
    tg = _random_pixel_targets(g, 500, seed=3)
    pa, ha = first_hit(a, eye, tg)
    pb, hb = first_hit(b, eye, tg)
    assert np.array_equal(ha, hb) and np.array_equal(pa, pb, equal_nan=True)


# --------------------------------------------------- full resolution smoke
def test_full_resolution_timing_smoke():
    """Full 2670 x 1878 panel: build + one first_hit pass per eye must stay well under 30 s."""
    g = FoldGeometry(DEFAULT_DEVICE, 120)
    t0 = time.perf_counter()
    sc = build_scene("sphere", g, CROSS)
    t_build = time.perf_counter() - t0
    tg = g.pixel_grid_world()
    eye, _ = CROSS.eyes()
    t1 = time.perf_counter()
    pts, hit = first_hit(sc, eye, tg)
    t_hit = time.perf_counter() - t1
    assert pts.shape == tg.shape and hit.all()
    assert t_build < 5.0
    assert t_build + t_hit < 30.0
