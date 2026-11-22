import math

import numpy as np
import pytest

from duovision.config import DEFAULT_DEVICE, DeviceProfile, ViewProfile
from duovision.geometry import FoldGeometry


def test_device_dimensions_match_spec():
    d = DEFAULT_DEVICE
    assert d.width_px == 2670 and d.height_px == 1878
    assert d.half_width_mm == pytest.approx(78.87, abs=0.05)
    assert d.height_mm == pytest.approx(110.93, abs=0.05)


def test_scaled_device_keeps_physical_size():
    small = DEFAULT_DEVICE.scaled(0.25)
    assert small.width_px % 2 == 0
    assert small.half_width_mm == pytest.approx(DEFAULT_DEVICE.half_width_mm, rel=1e-9)
    assert small.height_mm == pytest.approx(DEFAULT_DEVICE.height_mm, rel=2e-3)


def test_flat_fold_is_plane_z0():
    g = FoldGeometry(hinge_deg=180)
    w = g.pixel_grid_world()
    assert np.allclose(w[..., 2], 0.0, atol=1e-9)
    u, _ = g.pixel_to_uv(np.arange(g.device.width_px), 0)
    assert np.allclose(w[0, :, 0], u)


@pytest.mark.parametrize("deg", [100, 120, 140])
def test_outer_edges_toward_viewer(deg):
    g = FoldGeometry(hinge_deg=deg)
    pa, pb, pc = g.panel_corners("left")
    assert pa[2] == pytest.approx(g.half_width_mm * math.cos(math.radians(deg) / 2))
    assert pb[0] == 0 and pb[2] == 0
    assert g.max_panel_z() == pytest.approx(pa[2])
    assert np.allclose(np.linalg.norm(g.n_left), 1) and np.allclose(g.n_left @ g.a_left, 0)
    assert np.allclose(g.n_right @ g.a_right, 0)


def test_pixel_uv_roundtrip():
    g = FoldGeometry(hinge_deg=120)
    i = np.array([0.0, 10.3, 1334.0, 1335.0, 2669.0])
    j = np.array([0.0, 5.5, 900.0, 1877.0, 1000.2])
    u, v = g.pixel_to_uv(i, j)
    i2, j2 = g.uv_to_pixel(u, v)
    assert np.allclose(i, i2) and np.allclose(j, j2)
    # hinge sits on the boundary between the two middle columns
    u_mid, _ = g.pixel_to_uv(np.array([1334, 1335]), 0)
    assert u_mid[0] < 0 < u_mid[1]
    assert abs(u_mid[0]) == pytest.approx(u_mid[1])


@pytest.mark.parametrize("deg", [100, 120, 140, 180])
def test_rays_from_eye_land_on_their_pixels(deg):
    g = FoldGeometry(hinge_deg=deg)
    view = ViewProfile()
    left, right = view.eyes()
    rng = np.random.default_rng(0)
    i = rng.uniform(0, g.device.width_px - 1, 500)
    j = rng.uniform(0, g.device.height_px - 1, 500)
    pts = g.pixel_to_world(i, j)
    for eye in (left, right, view.cyclopean_eye((7, -5, 20))):
        i2, j2, ok = g.project_from_eye(eye, pts)
        assert ok.all()
        assert np.allclose(i, i2, atol=1e-6) and np.allclose(j, j2, atol=1e-6)


def test_points_in_front_project_onto_display():
    g = FoldGeometry(hinge_deg=120)
    left, right = ViewProfile().eyes()
    p = np.array([0.0, 0.0, 60.0])  # floating in front of the hinge
    il, _, okl = g.project_from_eye(left, p)
    ir, _, okr = g.project_from_eye(right, p)
    assert okl and okr
    assert ir < il  # crossed: right eye's landing point is left of the left eye's


def test_missing_rays_are_invalid():
    g = FoldGeometry(hinge_deg=120)
    u, v, t, ok = g.intersect(np.array([0, 0, 300.0]), np.array([[0, 0, 1.0], [1, 0, -0.01], [0, 10, -1.0]]))
    assert not ok.any()
    assert np.isnan(u).all() and np.isnan(t).all()


def test_view_profile_numbers_match_spec():
    v = ViewProfile()
    assert v.fused_distance(15) == pytest.approx(242.31, abs=0.01)
    assert v.z_background == pytest.approx(57.69, abs=0.01)
    assert v.z_near == pytest.approx(77.65, abs=0.01)
    assert v.conflict_dioptres(15) == pytest.approx(0.794, abs=0.001)
    p = v.with_(mode="parallel")
    assert p.world_z_for_separation(15) < 0
    with pytest.raises(ValueError):
        p.fused_distance(70)


def test_invalid_hinge():
    with pytest.raises(ValueError):
        FoldGeometry(hinge_deg=0)
    with pytest.raises(ValueError):
        FoldGeometry(hinge_deg=190)
