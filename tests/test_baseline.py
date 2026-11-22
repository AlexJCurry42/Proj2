"""Tests for duovision.baseline (classic flat-screen SIRDS for comparison)."""
import numpy as np
import pytest

from duovision import baseline as B
from duovision import verify as V
from duovision.config import DEFAULT_DEVICE, ViewProfile, guide_band_rows
from duovision.geometry import FoldGeometry
from test_verify import PlaneScene, PlaqueScene, mini_encode

DEV = DEFAULT_DEVICE.scaled(0.25)
VIEW = ViewProfile()


def test_flat_separation_matches_view_profile():
    assert B.flat_separation_mm(VIEW.z_background, VIEW) == pytest.approx(VIEW.sep_bg_mm)
    assert B.flat_separation_mm(VIEW.z_near, VIEW) == pytest.approx(VIEW.sep_near_mm)


def test_constant_depth_has_sep_bg_period():
    H, W = DEV.height_px, DEV.width_px
    img = B.flat_sirds(np.full((H, W), VIEW.z_background), DEV, VIEW, seed=1)
    assert img.shape == (H, W, 3) and img.dtype == np.uint8
    rr = guide_band_rows(DEV)
    assert img[:rr].max() == 0
    k = int(round(VIEW.sep_bg_mm / DEV.mm_per_px))
    rows = img[rr:]
    same = np.all(rows[:, k:] == rows[:, :-k], axis=-1).mean()
    assert same > 0.99
    other = np.all(rows[:, k + 7:] == rows[:, :-(k + 7)], axis=-1).mean()
    assert other < 0.2
    # deterministic for a seed, different across seeds
    assert np.array_equal(img, B.flat_sirds(np.full((H, W), VIEW.z_background), DEV, VIEW, seed=1))
    assert not np.array_equal(img, B.flat_sirds(np.full((H, W), VIEW.z_background), DEV, VIEW, seed=2))


def test_hidden_surface_removal_drops_occluded_links():
    H, W = DEV.height_px, DEV.width_px
    g = FoldGeometry(DEV, 180)
    sc = PlaqueScene(g, VIEW, zp=VIEW.z_near, hw=15, hh=15, vee=0.0)
    dm = B.baseline_depth_map(sc, DEV, VIEW)
    vis = B._visibility(dm, VIEW, DEV.mm_per_px)
    # background just outside the plaque's left/right edges is hidden from one eye
    assert (~vis[H // 2]).sum() > 4
    assert vis[H // 2, 5] and vis[H // 2, W // 2]


def test_baseline_depth_map_samples_cyclopean_ray():
    sc = PlaneScene(60.0)
    dm = B.baseline_depth_map(sc, DEV, VIEW)
    assert dm.shape == (DEV.height_px, DEV.width_px)
    assert np.allclose(dm, 60.0)


@pytest.fixture(scope="module")
def plaque_flat_sirds():
    g120 = FoldGeometry(DEV, 120)
    sc = PlaqueScene(g120, VIEW)
    dm = B.baseline_depth_map(sc, DEV, VIEW)
    return sc, B.flat_sirds(dm, DEV, VIEW, dot_px=2)


def test_flat_sirds_on_flat_display_is_correct(plaque_flat_sirds):
    sc, img = plaque_flat_sirds
    flat = FoldGeometry(DEV, 180)
    links = V.check_links(img, flat, VIEW, sc, sample=40000)
    assert links["match_rate_rounding_aware"] > 0.98
    assert links["match_rate"] > 0.9  # integer separations: off-by-one at half pixels
    e = V.evaluate(img, flat, VIEW, sc)
    assert e["median_abs_err_mm"] < 0.6
    assert e["plaque_kink_deg"] < 1.0
    assert e["valid_fraction_linkable"] > 0.7


def test_flat_sirds_on_fold_is_much_worse_than_fold_aware(plaque_flat_sirds):
    sc, img = plaque_flat_sirds
    g120 = FoldGeometry(DEV, 120)
    aware = V.evaluate(mini_encode(g120, VIEW, sc), g120, VIEW, sc)
    naive = V.evaluate(img, g120, VIEW, sc)
    assert aware["pass"] and not naive["pass"]
    assert naive["median_abs_err_mm"] > 5 * aware["median_abs_err_mm"]
    assert naive["valid_fraction_linkable"] < 0.5 * aware["valid_fraction_linkable"]


def test_integration_flat_baseline_vs_real_encoder():
    enc = pytest.importorskip("duovision.encoder")
    scn = pytest.importorskip("duovision.scene")
    g = FoldGeometry(DEV, 120)
    sc = scn.build_scene("plaque", g, VIEW)
    aware = V.evaluate(enc.encode_stereogram(g, VIEW, sc, "color", seed=0).image, g, VIEW, sc)
    naive_img = B.flat_sirds(B.baseline_depth_map(sc, DEV, VIEW), DEV, VIEW)
    naive = V.evaluate(naive_img, g, VIEW, sc)
    assert aware["pass"] and not naive["pass"]
    assert naive["median_abs_err_mm"] > 5 * aware["median_abs_err_mm"]
    assert naive["valid_fraction_linkable"] < 0.5 * aware["valid_fraction_linkable"]
    # shown flat, the same naive image is fine
    flat = FoldGeometry(DEV, 180)
    ok = V.evaluate(B.flat_sirds(B.baseline_depth_map(sc, DEV, VIEW), DEV, VIEW), flat, VIEW, sc)
    assert ok["median_abs_err_mm"] < 1.0
