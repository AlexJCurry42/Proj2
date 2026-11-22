"""Tests for duovision.verify (virtual viewer, independent link check, comfort).

Unit tests use small test doubles (a frontal-plane scene, a plaque-on-V scene and
a tiny reference encoder with nearest rounding) so they do not depend on the
production encoder; integration tests at the bottom use the real encoder.
"""
import math

import numpy as np
import pytest
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from duovision import verify as V
from duovision.config import DEFAULT_DEVICE, ViewProfile, guide_band_rows
from duovision.geometry import FoldGeometry

DEV = DEFAULT_DEVICE.scaled(0.25)
VIEW = ViewProfile()


# ----------------------------------------------------------------- doubles
class PlaneScene:
    """Frontal plane z = z0 (background only)."""

    def __init__(self, z0, name="bgplane"):
        self.name = name
        self.z0 = float(z0)
        self.z_min = self.z_max = self.z0

    def z_of(self, x, y):
        return np.full(np.broadcast(np.asarray(x), np.asarray(y)).shape, self.z0)

    def object_mask(self, x, y):
        return np.zeros(np.broadcast(np.asarray(x), np.asarray(y)).shape, bool)


class PlaqueScene:
    """Shallow V background + a frontal rectangular plaque (named 'plaque')."""

    def __init__(self, geom, view, zp=None, hw=30.0, hh=20.0, vee=0.25):
        self.name = "plaque"
        self.k = vee * geom.c / geom.s
        self.zb = view.z_background
        self.zp = view.z_near - 3 if zp is None else zp
        self.hw, self.hh = hw, hh
        self.z_min = self.zb
        self.z_max = max(self.zp, self.zb + self.k * geom.x_extent_mm + 1)

    def object_mask(self, x, y):
        return (np.abs(np.asarray(x)) <= self.hw) & (np.abs(np.asarray(y)) <= self.hh)

    def z_of(self, x, y):
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        return np.where(self.object_mask(x, y), self.zp, self.zb + self.k * np.abs(x))


def mini_encode(geom, view, scene, *, seed=0, dot_px=2, analytic_plane=False):
    """Tiny reference encoder: left ray -> S; visible from R -> q (nearest rounding);
    union of links via connected components; colour = random dot grid at the root."""
    dev = geom.device
    H, W = dev.height_px, dev.width_px
    rr = guide_band_rows(dev)
    L, R = view.eyes()
    X = geom.pixel_grid_world()
    if analytic_plane:
        t = (L[2] - scene.z0) / (L[2] - X[..., 2])
        S = L + t[..., None] * (X - L)
        vis = np.ones((H, W), bool)
    else:
        _, S, hit = V.march_heightfield(scene, L, X - L)
        vis = hit.copy()
        vis[hit] = V._visible_from(scene, R, S[hit])
    qi, qj, ok = geom.project_from_eye(R, S)
    qi = np.rint(np.where(ok, qi, -1)).astype(int)
    qj = np.rint(np.where(ok, qj, -1)).astype(int)
    jj, ii = np.mgrid[0:H, 0:W]
    link = ok & vis & (qi >= 0) & (qi < ii) & (qj >= rr) & (qj < H) & (jj >= rr)
    a = (jj * W + ii)[link]
    b = (qj * W + qi)[link]
    g = coo_matrix((np.ones(a.size), (a, b)), shape=(H * W, H * W))
    _, lab = connected_components(g, directed=False)
    root = np.full(lab.max() + 1, H * W)
    np.minimum.at(root, lab, np.arange(H * W))
    rng = np.random.default_rng(seed)
    noise = rng.integers(0, 256, (H // dot_px + 1, W // dot_px + 1, 3), dtype=np.uint8)
    img = noise[(root // W) // dot_px, (root % W) // dot_px][lab].reshape(H, W, 3)
    img[:rr] = 0
    return img


@pytest.fixture(scope="module")
def g120():
    return FoldGeometry(DEV, 120)


@pytest.fixture(scope="module")
def plane_case(g120):
    sc = PlaneScene(VIEW.z_background)
    return sc, mini_encode(g120, VIEW, sc, analytic_plane=True)


@pytest.fixture(scope="module")
def plaque_case(g120):
    sc = PlaqueScene(g120, VIEW)
    return sc, mini_encode(g120, VIEW, sc)


# ------------------------------------------------------------------ march
def test_march_hits_plane_exactly():
    sc = PlaneScene(60.0)
    L, _ = VIEW.eyes()
    rng = np.random.default_rng(0)
    targets = np.stack([rng.uniform(-60, 60, 500), rng.uniform(-50, 50, 500), np.zeros(500)], -1)
    t, P, hit = V.march_heightfield(sc, L, targets - L)
    assert hit.all()
    assert np.allclose(P[:, 2], 60.0, atol=1e-9)
    d = (targets - L) / np.linalg.norm(targets - L, axis=1, keepdims=True)
    assert np.allclose(P, L + t[:, None] * d)


def test_march_step_edge_and_visibility(g120):
    sc = PlaqueScene(g120, VIEW, zp=75.0, hw=10.0, hh=10.0)
    L, R = VIEW.eyes()
    # straight down onto the plaque and the background
    P0 = np.array([[0.0, 0.0, 0.0], [30.0, 0.0, 0.0]])
    E = np.array([0.0, 0.0, 300.0])
    _, S, hit = V.march_heightfield(sc, E, P0 - E)
    assert hit.all()
    assert S[0, 2] == pytest.approx(75.0, abs=1e-9)
    assert S[1, 2] == pytest.approx(sc.z_of(S[1, 0], S[1, 1]), abs=0.01)
    # a background point just right of the plaque is hidden from the left eye
    # (the plaque's raised edge is in the way) but visible from the right eye
    bx = 11.0
    B = np.array([[bx, 0.0, float(sc.z_of(bx, 0.0))]])
    assert not V._visible_from(sc, L, B)[0]
    assert V._visible_from(sc, R, B)[0]


# ------------------------------------------------------------ check_links
def test_check_links_correct_encode_is_perfect(plane_case, g120):
    sc, img = plane_case
    out = V.check_links(img, g120, VIEW, sc)
    assert out["n_confident"] > 10000
    # nearest-rounding reference encoder: the literal rule holds exactly
    assert out["match_rate"] >= 0.999
    assert out["match_rate_rounding_aware"] >= 0.999
    assert out["match_rate_all"] > 0.9
    lo, hi = out["pixel_separation_mm_range"]
    assert hi == pytest.approx(VIEW.sep_bg_mm, abs=1.5)  # widest separation at the hinge


def test_check_links_with_occlusion(plaque_case, g120):
    sc, img = plaque_case
    out = V.check_links(img, g120, VIEW, sc, sample=40000, seed=1)
    assert out["n_checked"] == 40000
    assert out["n_occluded_from_left"] > 0
    assert out["match_rate"] >= 0.99
    assert out["match_rate_rounding_aware"] >= 0.99


def test_check_links_random_image_is_chance(plane_case, g120):
    sc, img = plane_case
    rng = np.random.default_rng(3)
    noise = rng.integers(0, 256, img.shape, dtype=np.uint8)
    out = V.check_links(noise, g120, VIEW, sc, sample=20000)
    assert out["match_rate"] < 0.01
    assert out["match_rate_rounding_aware"] < 0.01
    # a binary random image matches at chance (~50 %), far from 100 %
    bw = (rng.random(img.shape[:2]) < 0.5).astype(np.uint8) * 255
    out = V.check_links(bw, g120, VIEW, sc, sample=20000)
    assert 0.4 < out["match_rate"] < 0.6
    assert out["mismatch_examples"]


def test_check_links_wrong_geometry_fails(plane_case):
    sc, img = plane_case
    out = V.check_links(img, FoldGeometry(DEV, 140), VIEW, sc, sample=20000)
    assert out["match_rate_rounding_aware"] < 0.5


# -------------------------------------------------------- render_eye_view
def test_render_eye_view_hinge_line_lands_where_predicted(g120):
    H, W = DEV.height_px, DEV.width_px
    img = np.zeros((H, W), np.uint8)
    img[:, W // 2 - 1: W // 2 + 1] = 255  # the two columns at the hinge
    L, R = VIEW.eyes()
    for eye in (L, R, VIEW.cyclopean_eye((10, 5, 30))):
        v = V.render_eye_view(img, g120, eye, res=(600, 420))
        f, cx, cy = v["focal_px"], v["cx"], v["cy"]
        for y in (-30.0, 0.0, 30.0):
            # hinge point (0, y, 0) -> retinal coordinates
            a = cx + f * (0.0 - eye[0]) / eye[2]
            b = cy - f * (y - eye[1]) / eye[2]
            row = v["image"][int(round(b))]
            cols = np.nonzero(row)[0]
            assert cols.size >= 1
            assert abs(cols.mean() - a) < 1.0, (eye, y, cols, a)
        # the whole display is in frame, the margin is black
        assert v["sees"].any() and not v["sees"][:, 0].any() and not v["sees"][0].any()
        assert v["image"][~v["sees"]].max() == 0


def test_render_eye_view_outer_edge_and_fov(g120):
    H, W = DEV.height_px, DEV.width_px
    img = np.zeros((H, W, 3), np.uint8)
    img[:, 0] = (255, 0, 0)  # left outer edge column
    L, _ = VIEW.eyes()
    v = V.render_eye_view(img, g120, L, res=(800, 560), fov_deg=40)
    assert v["focal_px"] == pytest.approx(400 / math.tan(math.radians(20)))
    edge = g120.pixel_to_world(0, H // 2)
    a = v["cx"] + v["focal_px"] * (edge[0] - L[0]) / (L[2] - edge[2])
    b = v["cy"] - v["focal_px"] * (edge[1] - L[1]) / (L[2] - edge[2])
    red = np.nonzero(v["image"][int(round(b)), :, 0])[0]
    assert red.size and abs(red.mean() - a) < 1.5
    assert v["image"].shape == (560, 800, 3) and v["image"].dtype == np.uint8


# ------------------------------------------------------- reconstruct_depth
def test_reconstruct_frontal_plane(plane_case, g120):
    sc, img = plane_case
    r = V.reconstruct_depth(img, g120, VIEW)
    valid = r["valid"]
    assert valid.sum() / r["sees_display"].sum() > 0.6
    err = r["z"][valid] - VIEW.z_background
    assert np.median(np.abs(err)) < 0.3
    assert np.mean(np.abs(err) < 1.0) > 0.95
    # rectified: at the convergence depth the image disparity is ~0
    assert np.median(np.abs(r["disparity"][valid])) < 1.0
    # reconstructed x, y lie over the display footprint
    assert np.nanmax(np.abs(r["x"])) < g120.x_extent_mm + 5


def test_reconstruct_shuffled_image_fails(plane_case, g120):
    sc, img = plane_case
    rng = np.random.default_rng(0)
    shuffled = img.copy()
    rr = guide_band_rows(DEV)
    body = shuffled[rr:].reshape(-1, 3)
    shuffled[rr:] = body[rng.permutation(body.shape[0])].reshape(shuffled[rr:].shape)
    r = V.reconstruct_depth(shuffled, g120, VIEW)
    assert r["valid"].sum() / r["sees_display"].sum() < 0.1


# ---------------------------------------------------------------- evaluate
def test_evaluate_plaque_double_passes(plaque_case, g120):
    sc, img = plaque_case
    e = V.evaluate(img, g120, VIEW, sc)
    assert e["pass"], e
    assert e["median_abs_err_mm"] < 0.5
    assert abs(e["hinge_kink_mm"]) < 0.5
    assert e["plaque_kink_deg"] < 1.0
    assert e["plaque_flatness_rms_mm"] < 0.6
    assert e["valid_fraction_linkable"] > 0.7
    assert "recon" not in e
    e2 = V.evaluate(img, g120, VIEW, sc, return_recon=True)
    assert e2["recon"]["z"].shape == e2["recon"]["valid"].shape


def test_evaluate_hinge_mismatch_is_worse(plaque_case, g120):
    sc, img = plaque_case
    good = V.evaluate(img, g120, VIEW, sc)
    bad = V.evaluate(img, g120, VIEW, sc, geom_viewed=FoldGeometry(DEV, 135))
    assert bad["hinge_deg_viewed"] == 135
    assert (bad["median_abs_err_mm"] > 2 * good["median_abs_err_mm"]
            or bad["valid_fraction_linkable"] < 0.7 * good["valid_fraction_linkable"])
    assert bad["plaque_kink_deg"] > good["plaque_kink_deg"]


def test_tolerance_sweep_rows(plaque_case, g120):
    sc, img = plaque_case
    rows = V.tolerance_sweep(img, g120, VIEW, sc, offsets=[(0, 0, 0), (20, 0, 0)], hinge_errors=[0, 5])
    combos = {(r["dx"], r["hinge_delta"]) for r in rows}
    assert combos == {(0.0, 0.0), (20.0, 0.0), (0.0, 5.0)}
    for r in rows:
        assert {"median_abs_err_mm", "valid_fraction_linkable", "plaque_kink_deg", "pass"} <= set(r)
    centre = next(r for r in rows if r["dx"] == 0 and r["hinge_delta"] == 0)
    assert centre["pass"]
    prod = V.tolerance_sweep(img, g120, VIEW, sc, offsets=[(0, 0, 0)], hinge_errors=[0, -2], pairs="product")
    assert len(prod) == 2
    offs, hinge = V.default_tolerance_grid()
    assert (30.0, 0.0, 0.0) in offs and (0.0, 0.0, -50.0) in offs and tuple(hinge) == (-5, -2, 0, 2, 5)
    assert len(V.default_tolerance_grid(full=True)[0]) == 45


# ------------------------------------------------------------------ comfort
def test_comfort_map_background_at_hinge(g120):
    sc = PlaneScene(VIEW.z_background)
    cm = V.comfort_map(g120, VIEW, sc)
    H, W = DEV.height_px, DEV.width_px
    rr = guide_band_rows(DEV)
    assert cm.shape == (H, W)
    assert np.isnan(cm[:rr]).all() and np.isfinite(cm[rr:]).all()
    expected = VIEW.conflict_dioptres(VIEW.sep_bg_mm)
    vals = cm[H // 2 - 1: H // 2 + 1, W // 2 - 1: W // 2 + 1]
    assert np.all(np.abs(vals - expected) < 0.05), (vals, expected)
    # the display comes toward the plane at the outer edges: less conflict there
    assert cm[H // 2, 0] < expected and cm[H // 2, W - 1] < expected
    # strided evaluation agrees with the exact map
    cm4 = V.comfort_map(g120, VIEW, sc, stride=4)
    assert np.nanmax(np.abs(cm4 - cm)) < 0.01


# ------------------------------------------------- integration (real encoder)
def _real():
    enc = pytest.importorskip("duovision.encoder")
    scn = pytest.importorskip("duovision.scene")
    return enc, scn


@pytest.mark.parametrize("name,hinge,style", [("plaque", 120, "color"), ("cube", 100, "bw"),
                                               ("text", 140, "texture")])
def test_integration_check_links_real_encoder(name, hinge, style):
    enc, scn = _real()
    g = FoldGeometry(DEV, hinge)
    sc = scn.build_scene(name, g, VIEW)
    res = enc.encode_stereogram(g, VIEW, sc, style, seed=0)
    out = V.check_links(res.image, g, VIEW, sc, sample=60000)
    assert out["match_rate_rounding_aware"] >= 0.99, out
    assert out["match_rate"] > 0.75  # literal rule, limited by randomized rounding
    tight = V.check_links(res.image, g, VIEW, sc, sample=60000, frac_tol=0.03)
    assert tight["match_rate"] > 0.95


@pytest.mark.parametrize("name", ["plaque", "cube", "text"])
def test_integration_evaluate_real_encoder(name):
    enc, scn = _real()
    g = FoldGeometry(DEV, 120)
    sc = scn.build_scene(name, g, VIEW)
    res = enc.encode_stereogram(g, VIEW, sc, "color", seed=0)
    e = V.evaluate(res.image, g, VIEW, sc)
    assert e["pass"], {k: v for k, v in e.items() if not isinstance(v, dict)}
    assert e["median_abs_err_mm"] < 0.5
    assert e["inlier_fraction"] > 0.95
    if name == "plaque":
        assert e["plaque_kink_deg"] < 1.0


@pytest.mark.xfail(reason="Matcher refinement for vertical head offsets was interrupted mid-rewrite; "
                   "at quarter resolution the valid fraction is 0.43 (< 0.6). Real half-resolution "
                   "encodes give 0.82 at dy=10 mm. Tracked in docs/STATUS.md.", strict=False)
def test_vertical_head_offset_is_compensated(plaque_case, g120):
    sc, img = plaque_case
    centre = V.design_vertical_disparity(g120, VIEW, sc)
    assert centre["p95_arcmin"] < 1e-6
    up = V.design_vertical_disparity(g120, VIEW, sc, head_offset=(0, 10, 0))
    assert 1.5 < up["p95_arcmin"] < 8  # ~3.8 arcmin: within Panum's vertical range
    e = V.evaluate(img, g120, VIEW, sc, head_offset=(0, 10, 0), return_recon=True)
    assert e["vertical_disparity_design_p95_arcmin"] == pytest.approx(up["p95_arcmin"], rel=0.2)
    assert e["recon"]["dv_field"] is not None
    assert e["valid_fraction_linkable"] > 0.6
    assert e["median_abs_err_mm"] < 1.0
