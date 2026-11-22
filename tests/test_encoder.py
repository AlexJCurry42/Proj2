"""Encoder tests: partner-map geometry, occlusion, link satisfaction, pipeline.

Scenes here are small test doubles (frontal plane, raised box) traced by a simple
reference marcher monkeypatched over ``duovision.encoder._first_hit``, so the
encoder is tested independently of ``duovision.scene``. Integration tests with the
real scenes run when ``duovision.scene`` is importable.
"""
import importlib.util
import time

import numpy as np
import pytest

from duovision import encoder
from duovision.config import DEFAULT_DEVICE, ViewProfile, guide_band_rows
from duovision.encoder import compute_partner_map, encode_stereogram
from duovision.geometry import FoldGeometry

DEV = DEFAULT_DEVICE.scaled(0.25)
VIEW = ViewProfile()
HAVE_SCENE = importlib.util.find_spec("duovision.scene") is not None


# --------------------------------------------------------------- test doubles
class PlaneScene:
    """Frontal plane z = z0 everywhere."""

    def __init__(self, z0, name="plane"):
        self.name = name
        self.z0 = float(z0)
        self.z_min = self.z_max = self.z0

    def z_of(self, x, y):
        return np.full(np.broadcast(np.asarray(x), np.asarray(y)).shape, self.z0)

    def object_mask(self, x, y):
        return np.zeros(np.broadcast(np.asarray(x), np.asarray(y)).shape, dtype=bool)


class BoxScene:
    """Background plane z_bg with a raised box (|x| < hx, |y| < hy) at z_top."""

    def __init__(self, z_bg, z_top, hx=10.0, hy=10.0, cx=0.0, name="box"):
        self.name = name
        self.z_bg, self.z_top, self.hx, self.hy, self.cx = z_bg, z_top, hx, hy, cx
        self.z_min, self.z_max = min(z_bg, z_top), max(z_bg, z_top)

    def object_mask(self, x, y):
        return (np.abs(np.asarray(x) - self.cx) < self.hx) & (np.abs(np.asarray(y)) < self.hy)

    def z_of(self, x, y):
        return np.where(self.object_mask(x, y), self.z_top, self.z_bg)


def fake_first_hit(scene, eye, targets, *, step_mm=0.2, refine_iters=30):
    """Reference tracer: march the ray eye -> targets (and beyond) through the
    slab [z_min, z_max], then bisect. Exact for PlaneScene."""
    eye = np.asarray(eye, dtype=np.float64)
    tg = np.asarray(targets, dtype=np.float64)
    shape = tg.shape[:-1]
    tg = tg.reshape(-1, 3)
    d = tg - eye
    dz = d[:, 2]
    pts = np.full(tg.shape, np.nan)
    hit = np.zeros(len(tg), dtype=bool)
    if len(tg) == 0:
        return pts.reshape(shape + (3,)), hit.reshape(shape)
    with np.errstate(divide="ignore", invalid="ignore"):
        if isinstance(scene, PlaneScene):
            t = (scene.z0 - eye[2]) / dz
            ok = np.isfinite(t) & (t > 0)
            pts[ok] = eye + t[ok, None] * d[ok]
            return pts.reshape(shape + (3,)), ok.reshape(shape)
        t0 = (scene.z_max + 1e-6 - eye[2]) / dz
        t1 = (scene.z_min - 1e-6 - eye[2]) / dz
    ok = (dz < 0) & np.isfinite(t0) & np.isfinite(t1)
    t0 = np.where(ok, np.maximum(t0, 0), 0.0)
    t1 = np.where(ok, t1, 0.0)
    norm = np.linalg.norm(d, axis=1)
    dt = step_mm / np.where(norm > 0, norm, 1)
    nsteps = int(np.ceil(np.max(np.where(ok, (t1 - t0) / dt, 0)))) + 2
    lo = t0.copy()
    found = np.zeros(len(tg), dtype=bool)
    hi = np.zeros(len(tg))
    for k in range(1, nsteps + 1):
        t = np.minimum(t0 + k * dt, t1)
        p = eye + t[:, None] * d
        inside = ok & ~found & (p[:, 2] <= scene.z_of(p[:, 0], p[:, 1]))
        hi[inside] = t[inside]
        found |= inside
        lo = np.where(~found, t, lo)
    for _ in range(refine_iters):
        mid = 0.5 * (lo + hi)
        p = eye + mid[:, None] * d
        inside = p[:, 2] <= scene.z_of(p[:, 0], p[:, 1])
        hi = np.where(found & inside, mid, hi)
        lo = np.where(found & ~inside, mid, lo)
    pts[found] = eye + hi[found, None] * d[found]
    return pts.reshape(shape + (3,)), found.reshape(shape)


@pytest.fixture
def fake_tracer(monkeypatch):
    monkeypatch.setattr(encoder, "_first_hit", fake_first_hit)


# ------------------------------------------------------------------- helpers
def assert_links_satisfied(res):
    img = res.image
    pi, pj = res.pi, res.pj
    jj, ii = np.nonzero(pi >= 0)
    assert len(jj) > 0
    a = img[jj, ii]
    b = img[pj[jj, ii], pi[jj, ii]]
    assert np.array_equal(a, b), "a linked pair has different colours"


def assert_direction_and_band(res, mode="cross"):
    pi, pj = res.pi, res.pj
    rr = res.partner_map.reserved_rows
    H, W = pi.shape
    cols = np.arange(W)[None, :].repeat(H, 0)
    lk = pi >= 0
    if mode == "cross":
        assert np.all(pi[lk] < cols[lk])
    else:
        assert np.all(pi[lk] > cols[lk])
    assert np.all(pj[lk] >= rr)
    assert not lk[:rr].any()
    assert not res.partner_map.linked[:rr].any()
    assert np.all(res.image[:rr] == 0)
    pm = res.partner_map
    fl = pm.linked
    if mode == "cross":
        assert np.all(pm.partner_i[fl] <= cols[fl] - 1 + 1e-9)
    else:
        assert np.all(pm.partner_i[fl] >= cols[fl] + 1 - 1e-9)
    assert np.all(pm.partner_j[fl] >= rr)
    assert np.all(np.isnan(pm.partner_i[~fl])) and np.all(np.isnan(pm.partner_j[~fl]))


def left_eye_hits(geom, view, scene, mask):
    """World S for the pixels in `mask`, traced independently with the fake tracer."""
    jj, ii = np.nonzero(mask)
    L, _ = view.eyes()
    S, hit = fake_first_hit(scene, L, geom.pixel_to_world(ii, jj))
    return ii, jj, S, hit


# --------------------------------------------------------------------- tests
@pytest.mark.parametrize("style", ["bw", "color", "texture"])
def test_every_link_satisfied_plane(fake_tracer, style):
    geom = FoldGeometry(DEV, 120)
    res = encode_stereogram(geom, VIEW, PlaneScene(VIEW.z_background), style, seed=3)
    assert res.image.shape == (DEV.height_px, DEV.width_px, 3) and res.image.dtype == np.uint8
    assert_links_satisfied(res)
    assert_direction_and_band(res)
    assert res.stats["linked_fraction"] > 0.6


@pytest.mark.parametrize("deg", [100, 120, 140])
def test_every_link_satisfied_box(fake_tracer, deg):
    geom = FoldGeometry(DEV, deg)
    scene = BoxScene(VIEW.z_background, VIEW.z_near, hx=15, hy=15)
    res = encode_stereogram(geom, VIEW, scene, "color", seed=1)
    assert_links_satisfied(res)
    assert_direction_and_band(res)
    st = res.stats
    assert st["occluded_fraction"] > 0
    assert abs(st["linked_fraction"] + st["free_fraction"] - 1) < 1e-12
    for k in ("time_partner_map_s", "time_resolve_s", "time_colorize_s", "sep_px_min",
              "sep_px_median", "sep_px_max", "vertical_offset_px_max"):
        assert k in st


def test_reserved_rows_override(fake_tracer):
    geom = FoldGeometry(DEV, 120)
    res = encode_stereogram(geom, VIEW, PlaneScene(VIEW.z_background), "bw", reserved_rows=60)
    assert res.partner_map.reserved_rows == 60
    assert_direction_and_band(res)
    pm = compute_partner_map(geom, VIEW, PlaneScene(VIEW.z_background))
    assert pm.reserved_rows == guide_band_rows(DEV)


def _axis_separation(geom, view, scene, z):
    """(separation from the partner map, analytic separation) in px at the hinge axis."""
    pm = compute_partner_map(geom, view, scene)
    L, R = view.eyes()
    S0 = np.array([0.0, 0.0, z])
    il, jl, okl = geom.project_from_eye(L, S0)
    ir, jr, okr = geom.project_from_eye(R, S0)
    assert okl and okr
    j = int(round(float(jl)))
    i = int(round(float(il)))
    assert pm.linked[j, i]
    sep_map = i - pm.partner_i[j, i]
    return float(sep_map), float(il - ir), pm


@pytest.mark.parametrize("which,sep_mm", [("bg", 15.0), ("near", 22.0)])
def test_flat_display_separation_matches_view_profile(fake_tracer, which, sep_mm):
    # At 180 deg the display is the plane z = 0: separation is exactly sep_bg / sep_near.
    geom = FoldGeometry(DEV, 180)
    z = VIEW.z_background if which == "bg" else VIEW.z_near
    sep_map, sep_ana, pm = _axis_separation(geom, VIEW, PlaneScene(z), z)
    assert sep_ana * geom.mm_per_px == pytest.approx(sep_mm, abs=1e-6)
    assert abs(sep_map - sep_mm / geom.mm_per_px) <= 1.0
    # a frontal plane on a flat screen has constant separation everywhere
    jj, ii = np.nonzero(pm.linked)
    sep = ii - pm.partner_i[pm.linked]
    assert np.allclose(sep, sep_mm / geom.mm_per_px, atol=1e-6)
    assert np.allclose(pm.partner_j[pm.linked], jj, atol=1e-6)


@pytest.mark.parametrize("deg", [100, 120, 140])
@pytest.mark.parametrize("which", ["bg", "near"])
def test_folded_axis_separation_matches_analytic(fake_tracer, deg, which):
    # On the fold the surface-measured separation of a point on the hinge axis is
    # set by where the two eye rays meet the tilted halves (larger than on a flat screen).
    geom = FoldGeometry(DEV, deg)
    z = VIEW.z_background if which == "bg" else VIEW.z_near
    sep_map, sep_ana, _ = _axis_separation(geom, VIEW, PlaneScene(z), z)
    assert abs(sep_map - sep_ana) <= 1.0
    # independent closed form: L ray to (0, 0, z) meets the right half at u = x / s
    L, _ = VIEW.eyes()
    k = geom.c / geom.s
    t = (L[2] - k * L[0]) / ((L[2] - z) - k * L[0])
    x = L[0] - t * L[0]
    u = x / geom.s
    assert sep_ana * geom.mm_per_px == pytest.approx(2 * u, abs=1e-6)
    flat = VIEW.sep_bg_mm if which == "bg" else VIEW.sep_near_mm
    assert sep_ana * geom.mm_per_px > flat


def test_occlusion_left_of_raised_box(fake_tracer):
    geom = FoldGeometry(DEV, 120)
    zb, zt, hx, hy = VIEW.z_background, VIEW.z_near, 10.0, 10.0
    scene = BoxScene(zb, zt, hx=hx, hy=hy)
    pm = compute_partner_map(geom, VIEW, scene)
    occ = pm.occluded
    assert occ.sum() > 20
    assert not (occ & pm.linked).any()
    ii, jj, S, hit = left_eye_hits(geom, VIEW, scene, occ)
    assert hit.all()
    # Hidden from R = just left of the box: on the background within the shadow
    # width (z_top - z_bg) * (R.x - x) / (R.z - z_top) ~ 3.7 mm, or on the left wall.
    _, R = VIEW.eyes()
    shadow = (zt - zb) * (R[0] + hx + 6) / (R[2] - zt)
    # (A few background points just past the box's top/bottom edge, whose left-eye
    # ray grazes the box corner between march steps, are also hidden from R.)
    grazing = (np.abs(S[:, 1]) >= hy) & (np.abs(S[:, 1]) <= hy + 1.5) & (S[:, 0] > -hx + 0.05)
    assert grazing.mean() < 0.05
    main = S[~grazing]
    assert np.all(main[:, 0] <= -hx + 0.05)
    assert np.all(main[:, 0] >= -hx - shadow - 0.05)
    assert np.all(np.abs(main[:, 1]) <= hy + 1.5)
    assert np.all(S[:, 0] < 0)

    # Analytic shadow test on all background hits: segment R -> S enters the box?
    bg = ~np.isnan(pm.hit_z) & np.isclose(pm.hit_z, zb, atol=1e-6)
    ii, jj, S, hit = left_eye_hits(geom, VIEW, scene, bg)
    # slab test: segment R + t (S - R), t in (0, 1), against the box column
    d = S - R
    lo = np.array([-hx, -hy, zb - 1.0])
    hi = np.array([hx, hy, zt])
    with np.errstate(divide="ignore", invalid="ignore"):
        ta = (lo - R) / d
        tb = (hi - R) / d
    tmin = np.nanmax(np.minimum(ta, tb), axis=1)
    tmax = np.nanmin(np.maximum(ta, tb), axis=1)
    blocked = (tmin < tmax) & (tmax > 0) & (tmin < 1 - 1e-9)
    got = occ[jj, ii]
    mismatch = (blocked != got).sum()
    assert blocked.sum() > 10
    assert mismatch <= max(2, 0.03 * blocked.sum())
    # nothing on the right side of the box is occluded
    assert not got[S[:, 0] > 0].any()


def test_left_wall_hits_are_occluded(fake_tracer):
    geom = FoldGeometry(DEV, 120)
    scene = BoxScene(VIEW.z_background, VIEW.z_near, hx=10, hy=10)
    pm = compute_partner_map(geom, VIEW, scene)
    z = pm.hit_z
    wall = (z > VIEW.z_background + 1) & (z < VIEW.z_near - 1)
    assert wall.any()
    ii, jj, S, _ = left_eye_hits(geom, VIEW, scene, wall)
    assert np.allclose(S[:, 0], -10, atol=0.05)  # L sees the left wall only
    assert pm.occluded[wall].all() and not pm.linked[wall].any()


def test_parallel_mode_links_point_right(fake_tracer):
    view = VIEW.with_(mode="parallel")
    geom = FoldGeometry(DEV, 120)
    z = view.z_background
    assert z < 0
    res = encode_stereogram(geom, view, PlaneScene(z), "bw", seed=2)
    assert res.stats["linked_fraction"] > 0.5
    assert_links_satisfied(res)
    assert_direction_and_band(res, mode="parallel")
    # flat-screen separation check in parallel mode
    g180 = FoldGeometry(DEV, 180)
    pm = compute_partner_map(g180, view, PlaneScene(z))
    jj, ii = np.nonzero(pm.linked)
    sep = pm.partner_i[pm.linked] - ii
    assert np.allclose(sep * g180.mm_per_px, view.sep_bg_mm, atol=1e-6)


def test_missed_rays_are_free(fake_tracer):
    geom = FoldGeometry(DEV, 120)

    class Empty(PlaneScene):
        pass

    def miss(scene, eye, targets):
        t = np.asarray(targets, dtype=float)
        return np.full(t.shape, np.nan), np.zeros(t.shape[:-1], dtype=bool)

    encoder_first = encoder._first_hit
    try:
        encoder._first_hit = miss
        pm = compute_partner_map(geom, VIEW, Empty(60))
    finally:
        encoder._first_hit = encoder_first
    assert not pm.linked.any() and not pm.occluded.any() and np.isnan(pm.hit_z).all()


def test_deterministic(fake_tracer):
    geom = FoldGeometry(DEV, 120)
    scene = PlaneScene(VIEW.z_background)
    a = encode_stereogram(geom, VIEW, scene, "color", seed=5).image
    b = encode_stereogram(geom, VIEW, scene, "color", seed=5).image
    c = encode_stereogram(geom, VIEW, scene, "color", seed=6).image
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


# ------------------------------------------------------- real-scene integration
needs_scene = pytest.mark.skipif(not HAVE_SCENE, reason="duovision.scene not available yet")


@needs_scene
@pytest.mark.parametrize("name", ["plaque", "cube", "torus"])
@pytest.mark.parametrize("deg", [100, 120, 140])
def test_real_scene_links(name, deg):
    from duovision.scene import build_scene

    geom = FoldGeometry(DEV, deg)
    scene = build_scene(name, geom, VIEW)
    res = encode_stereogram(geom, VIEW, scene, "color", seed=0)
    assert_links_satisfied(res)
    assert_direction_and_band(res)
    st = res.stats
    assert st["linked_fraction"] > 0.5
    assert st["miss_fraction"] < 0.05
    if name != "plaque":
        assert st["occluded_fraction"] > 0
    # separations bounded by the background/near design range (+ fold stretch)
    assert st["sep_mm_max"] < 40
    assert st["vertical_offset_px_max"] < 0.1 * DEV.height_px


@needs_scene
def test_real_scene_parallel():
    from duovision.scene import build_scene

    view = VIEW.with_(mode="parallel")
    geom = FoldGeometry(DEV, 120)
    scene = build_scene("cube", geom, view)
    res = encode_stereogram(geom, view, scene, "bw", seed=0)
    assert res.stats["linked_fraction"] > 0.5
    assert_links_satisfied(res)
    assert_direction_and_band(res, mode="parallel")


@needs_scene
def test_full_resolution_timing_smoke():
    from duovision.scene import build_scene

    geom = FoldGeometry(DEFAULT_DEVICE, 120)
    scene = build_scene("torus", geom, VIEW)
    t0 = time.perf_counter()
    res = encode_stereogram(geom, VIEW, scene, "color", seed=0)
    dt = time.perf_counter() - t0
    assert res.image.shape == (1878, 2670, 3)
    assert_links_satisfied(res)
    print("full-res encode", round(dt, 1), "s", {k: v for k, v in res.stats.items()})
    assert dt < 180  # target <= 60 s on the reference machine; generous for CI


@needs_scene
def test_real_scene_occluders_are_objects():
    """Occluded pixels: the right eye's view of S is blocked by a scene object."""
    from duovision.scene import build_scene, first_hit

    geom = FoldGeometry(DEV, 120)
    scene = build_scene("cube", geom, VIEW)
    pm = compute_partner_map(geom, VIEW, scene)
    jj, ii = np.nonzero(pm.occluded)
    assert len(jj) > 0 and not pm.linked[pm.occluded].any()
    L, R = VIEW.eyes()
    S, hit = first_hit(scene, L, geom.pixel_to_world(ii, jj))
    assert hit.all()
    S2, hit2 = first_hit(scene, R, S)
    # Either a genuine blocker in front of S, or a grazing silhouette point whose
    # right-eye ray slips past S between march steps (hit lands beyond S): both free.
    t2 = np.einsum("ij,ij->i", S2 - R, S - R) / np.einsum("ij,ij->i", S - R, S - R)
    grazing = t2 > 1.0
    assert grazing.mean() < 0.1
    blk, Sb = S2[~grazing], S[~grazing]
    assert scene.object_mask(blk[:, 0], blk[:, 1]).mean() > 0.95
    assert np.all(blk[:, 2] >= Sb[:, 2] - 0.05)  # blocker is nearer the eye than S
    assert np.all(blk[:, 0] > Sb[:, 0] - 0.05)  # cross-eyed occlusion: blocker lies right of S
