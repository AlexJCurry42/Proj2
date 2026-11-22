"""Black-box tests for duovision.scene against ARCHITECTURE.md and the physics."""
import numpy as np
import pytest

from bbphys import DEV, footprint_grid, geom_view, scene_for, z_bg

scene_mod = pytest.importorskip("duovision.scene")
SCENE_NAMES = scene_mod.SCENE_NAMES
HINGES = (100, 120, 140)


def test_scene_names_match_contract():
    assert tuple(SCENE_NAMES) == ("plaque", "cube", "sphere", "text", "torus", "pyramid", "heart", "pawn")


@pytest.mark.parametrize("hinge", HINGES)
@pytest.mark.parametrize("name", SCENE_NAMES)
def test_scene_builds_and_has_attributes(name, hinge):
    sc = scene_for(name, hinge)
    assert isinstance(sc.name, str)
    assert np.isfinite(sc.z_min) and np.isfinite(sc.z_max) and sc.z_min <= sc.z_max


@pytest.mark.parametrize("name", SCENE_NAMES)
def test_z_of_is_vectorised(name):
    sc = scene_for(name, 120)
    rng = np.random.default_rng(1)
    for shape in [(7,), (3, 4), (2, 3, 4), (1, 1), (0,)]:
        x = rng.uniform(-50, 50, shape)
        y = rng.uniform(-40, 40, shape)
        z = np.asarray(sc.z_of(x, y))
        m = np.asarray(sc.object_mask(x, y))
        assert z.shape == shape and m.shape == shape
        assert m.dtype == bool
        if z.size:
            # elementwise: same answer as evaluating one by one
            flat = [float(np.asarray(sc.z_of(np.array([a]), np.array([b])))[0]) for a, b in zip(x.ravel(), y.ravel())]
            assert np.allclose(z.ravel(), flat, atol=1e-9)
    # scalar inputs give a 0-d result
    assert np.asarray(sc.z_of(0.0, 0.0)).shape == ()


@pytest.mark.parametrize("hinge", HINGES)
@pytest.mark.parametrize("name", SCENE_NAMES)
def test_scene_clears_display_by_3mm(name, hinge):
    _, view, fold = geom_view(hinge)
    sc = scene_for(name, hinge)
    X, Y = footprint_grid(fold)
    z = np.asarray(sc.z_of(X, Y), float)
    clearance = z - fold.panel_z(X)  # own panel z = |x| cot(phi/2)
    assert clearance.min() >= 3.0 - 1e-9, f"min clearance {clearance.min():.3f} mm"


@pytest.mark.parametrize("hinge", HINGES)
@pytest.mark.parametrize("name", SCENE_NAMES)
def test_z_bounds_and_object_depth_limits(name, hinge):
    _, view, fold = geom_view(hinge)
    sc = scene_for(name, hinge)
    X, Y = footprint_grid(fold)
    z = np.asarray(sc.z_of(X, Y), float)
    m = np.asarray(sc.object_mask(X, Y))
    assert z.min() >= sc.z_min - 1e-6 and z.max() <= sc.z_max + 1e-6
    assert m.any(), "scene has no object"
    assert z[m].max() <= view.z_near + 0.5, f"object max z {z[m].max():.2f} > z_near+0.5"
    # height field never dips below the background
    assert (z - z_bg(view, fold, X, 0.25)).min() >= -0.01
    # objects rise at least ~2 mm above the background somewhere
    assert (z - z_bg(view, fold, X, 0.25))[m].max() >= 2.0 - 0.01


@pytest.mark.parametrize("name", SCENE_NAMES)
def test_objects_keep_clear_of_edges_and_band(name):
    # CONTRACT-AMBIGUITY: "15 mm in from the outer edges" and "out of the guide band"
    # do not say how measured. We test the weakest reading: object world x within
    # (half_width - 15 mm along the surface) * sin(phi/2), and object world y below
    # the band's lower edge on the display surface.
    from duovision.config import guide_band_rows

    _, view, fold = geom_view(120)
    sc = scene_for(name, 120)
    X, Y = footprint_grid(fold, 401, 301)
    m = np.asarray(sc.object_mask(X, Y))
    x_lim = (fold.hw - 15.0) * fold.s
    assert not (m & (np.abs(X) > x_lim)).any(), f"object beyond |x|={x_lim:.1f}"
    band_y = fold.hh - guide_band_rows(DEV) * fold.mmpp
    assert not (m & (Y > band_y)).any()


@pytest.mark.parametrize("hinge", HINGES)
def test_plaque_is_planar_frontal(hinge):
    from scipy.ndimage import binary_erosion

    _, _, fold = geom_view(hinge)
    sc = scene_for("plaque", hinge)
    X, Y = footprint_grid(fold, 401, 301)
    m = binary_erosion(np.asarray(sc.object_mask(X, Y)), iterations=3)
    assert m.sum() > 100
    z = np.asarray(sc.z_of(X, Y), float)[m]
    assert np.ptp(z) < 1e-3, f"plaque z range {np.ptp(z):.4f} mm"


@pytest.mark.parametrize("bg_vee", [0.0, 0.25])
@pytest.mark.parametrize("hinge", HINGES)
@pytest.mark.parametrize("name", SCENE_NAMES)
def test_background_formula_away_from_objects(name, hinge, bg_vee):
    from scipy.ndimage import binary_dilation

    _, view, fold = geom_view(hinge)
    sc = scene_for(name, hinge, bg_vee)
    X, Y = footprint_grid(fold, 401, 301)
    m = np.asarray(sc.object_mask(X, Y))
    dx = X[0, 1] - X[0, 0]
    far = ~binary_dilation(m, iterations=int(np.ceil(5.0 / dx)))
    assert far.sum() > 0.2 * far.size
    z = np.asarray(sc.z_of(X, Y), float)
    expect = view.z_background + bg_vee * np.abs(X) * fold.c / fold.s
    err = np.abs(z - expect)[far]
    assert err.max() < 1e-3, f"background deviates by {err.max():.4f} mm"


@pytest.mark.parametrize("hinge", HINGES)
@pytest.mark.parametrize("name", SCENE_NAMES)
def test_first_hit_on_ray_and_on_surface(name, hinge):
    _, view, fold = geom_view(hinge)
    sc = scene_for(name, hinge)
    rng = np.random.default_rng(hinge)
    i = rng.uniform(0, fold.W - 1, 400)
    j = rng.uniform(0, fold.H - 1, 400)
    T = fold.pix_to_world(i, j)
    for eye in view.eyes():
        pts, hit = scene_mod.first_hit(sc, eye, T)
        pts = np.asarray(pts, float)
        hit = np.asarray(hit, bool)
        assert pts.shape == (400, 3) and hit.shape == (400,)
        assert hit.mean() > 0.95
        p = pts[hit]
        a = p - eye
        b = T[hit] - eye
        cross = np.linalg.norm(np.cross(a, b), axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
        assert cross.max() < 1e-6
        assert ((a * b).sum(axis=1) > 0).all()
        # on the surface; at a cliff the hit sits on the wall, so allow the
        # bracket [z_of just before, z_of just after] along the ray
        d = b / np.linalg.norm(b, axis=1, keepdims=True)
        zc = np.asarray(sc.z_of(p[:, 0], p[:, 1]), float)
        pb = p - 0.05 * d
        pa = p + 0.05 * d
        z1 = np.asarray(sc.z_of(pb[:, 0], pb[:, 1]), float)
        z2 = np.asarray(sc.z_of(pa[:, 0], pa[:, 1]), float)
        cliff = np.abs(z1 - z2) > 0.5
        smooth_ok = np.abs(p[:, 2] - zc) < 0.02
        cliff_ok = (p[:, 2] >= np.minimum(z1, z2) - 0.02) & (p[:, 2] <= np.maximum(z1, z2) + 0.02)
        ok = np.where(cliff, cliff_ok, smooth_ok)
        assert ok.all(), f"{(~ok).sum()} hits off-surface, worst {np.abs(p[:,2]-zc)[~ok].max():.4f}"
        # it is the FIRST hit: the ray is above the surface just before it
        zprev = np.asarray(sc.z_of(pb[:, 0], pb[:, 1]), float)
        assert (pb[:, 2] - zprev > -0.02).all()


@pytest.mark.parametrize("name", SCENE_NAMES)
def test_first_hit_shapes(name):
    _, view, fold = geom_view(120)
    sc = scene_for(name, 120)
    T = fold.pix_to_world(*np.meshgrid(np.linspace(0, fold.W - 1, 5), np.linspace(100, fold.H - 1, 4)))
    pts, hit = scene_mod.first_hit(sc, view.eyes()[0], T)
    assert np.shape(pts) == (4, 5, 3) and np.shape(hit) == (4, 5)


@pytest.mark.parametrize("name", SCENE_NAMES)
def test_first_hit_cyclopean_to_hinge_centre(name):
    _, view, _ = geom_view(120)
    sc = scene_for(name, 120)
    eye = view.cyclopean_eye()
    pts, hit = scene_mod.first_hit(sc, eye, np.array([[0.0, 0.0, 0.0]]))
    pts = np.asarray(pts, float).reshape(-1, 3)
    assert bool(np.asarray(hit).ravel()[0])
    zc = float(np.asarray(sc.z_of(0.0, 0.0)))
    assert abs(pts[0, 0]) < 1e-6 and abs(pts[0, 1]) < 1e-6
    assert abs(pts[0, 2] - zc) < 0.01


def test_text_parameter_changes_text_scene():
    geom, view, fold = geom_view(120)
    a = scene_mod.build_scene("text", geom, view, text="DuoVision")
    b = scene_mod.build_scene("text", geom, view, text="II")
    X, Y = footprint_grid(fold)
    assert not np.array_equal(np.asarray(a.object_mask(X, Y)), np.asarray(b.object_mask(X, Y)))


@pytest.mark.parametrize("name", ["plaque", "sphere"])
def test_parallel_mode_scene_behind_display(name):
    geom, view, fold = geom_view(120, mode="parallel")
    sc = scene_mod.build_scene(name, geom, view)
    X, Y = footprint_grid(fold)
    z = np.asarray(sc.z_of(X, Y), float)
    assert (z < fold.panel_z(X)).all()
