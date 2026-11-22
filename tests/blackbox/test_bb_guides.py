"""Black-box tests for guides.py."""
import numpy as np
import pytest

from bbphys import geom_view
from duovision.config import guide_band_rows

guides = pytest.importorskip("duovision.guides")
HINGES = (100, 120, 140, 180)


def closest_approach(p1, d1, p2, d2):
    w0 = p1 - p2
    a, b, c = d1 @ d1, d1 @ d2, d2 @ d2
    d, e = d1 @ w0, d2 @ w0
    den = a * c - b * b
    s = (b * e - c * d) / den
    t = (a * e - b * d) / den
    q1, q2 = p1 + s * d1, p2 + t * d2
    return np.linalg.norm(q1 - q2), 0.5 * (q1 + q2)


def fused_point(fold, view, il, jl, ir, jr):
    L, R = view.eyes()
    Pl = fold.pix_to_world(il, jl)
    Pr = fold.pix_to_world(ir, jr)
    # cross-eyed: right eye looks at the left dot, left eye at the right dot
    return closest_approach(R, Pl - R, L, Pr - L)


@pytest.mark.parametrize("hinge", HINGES)
def test_guide_dots_fuse_at_background_depth_on_axis(hinge):
    geom, view, fold = geom_view(hinge)
    (il, jl), (ir, jr) = guides.guide_dot_pixels(geom, view)
    il, jl, ir, jr = map(float, (il, jl, ir, jr))
    band = guide_band_rows(geom.device)
    assert il < fold.W / 2 - 0.5 < ir
    for jv in (jl, jr):
        assert 0 <= jv <= band - 1
    gap, M = fused_point(fold, view, il, jl, ir, jr)
    assert gap < 0.1, f"lines miss by {gap:.3f} mm"
    assert abs(M[0]) < 0.1
    # tolerance from pixel quantisation: perturb dots by half a pixel
    zs = [fused_point(fold, view, il + a, jl, ir + b, jr)[1][2] for a in (-0.5, 0.5) for b in (-0.5, 0.5)]
    tol = max(abs(z - M[2]) for z in zs) + 0.05
    assert abs(M[2] - view.z_background) < tol, f"fused z {M[2]:.2f} vs {view.z_background:.2f} (tol {tol:.2f})"


@pytest.mark.parametrize("hinge", [100, 140])
def test_draw_guide_band_only_touches_band(hinge):
    geom, view, fold = geom_view(hinge)
    band = guide_band_rows(geom.device)
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (fold.H, fold.W, 3), dtype=np.uint8)
    img[:band] = 0
    orig = img.copy()
    out = np.asarray(guides.draw_guide_band(img.copy(), geom, view, "test 120"))
    assert out.shape == orig.shape and out.dtype == np.uint8
    assert np.array_equal(out[band:], orig[band:])
    assert (out[:band] != 0).any()
    (il, jl), (ir, jr) = guides.guide_dot_pixels(geom, view)
    bg = np.median(out[:band].reshape(-1, 3), axis=0)
    for i, j in ((il, jl), (ir, jr)):
        px = out[int(round(float(j))), int(round(float(i)))]
        assert np.abs(px.astype(int) - bg).max() > 30, "no dot drawn at guide_dot_pixels"


@pytest.mark.parametrize("hinge", [100, 120, 140])
def test_alignment_image(hinge):
    geom, view, fold = geom_view(hinge)
    img = np.asarray(guides.alignment_image(geom, view, "align"))
    assert img.shape == (fold.H, fold.W, 3) and img.dtype == np.uint8
    lum = img.mean(axis=2)
    assert lum.mean() < 100 and (lum > 128).mean() < 0.3, "alignment image should be mostly dark"
    assert (lum > 128).any()


def test_alignment_image_is_fold_aware():
    band = guide_band_rows(geom_view(100)[0].device)
    a = np.asarray(guides.alignment_image(geom_view(100)[0], geom_view(100)[1], "x"))
    b = np.asarray(guides.alignment_image(geom_view(140)[0], geom_view(140)[1], "x"))
    assert not np.array_equal(a[band:], b[band:])
