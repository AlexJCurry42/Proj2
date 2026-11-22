import numpy as np
import pytest

from duovision.config import DEFAULT_DEVICE, DEFAULT_VIEW, guide_band_rows
from duovision.geometry import FoldGeometry
from duovision import guides

DEV = DEFAULT_DEVICE.scaled(0.25)


def _closest_points(p1, d1, p2, d2):
    """Closest points between lines p1 + s d1 and p2 + t d2."""
    w0 = p1 - p2
    a, b, c = d1 @ d1, d1 @ d2, d2 @ d2
    d, e = d1 @ w0, d2 @ w0
    den = a * c - b * b
    s = (b * e - c * d) / den
    t = (a * e - b * d) / den
    return p1 + s * d1, p2 + t * d2


@pytest.mark.parametrize("hinge", [100, 120, 140, 180])
@pytest.mark.parametrize("view", [DEFAULT_VIEW, DEFAULT_VIEW.with_(eye_height_mm=10.0, ipd_mm=58.0)])
def test_guide_dot_rays_meet_at_G(hinge, view):
    geom = FoldGeometry(DEV, hinge)
    (il, jl), (ir, jr) = guides.guide_dot_pixels(geom, view)
    g = guides.guide_world_point(geom, view)
    assert g[0] == 0.0 and g[2] == pytest.approx(view.z_background)
    left, right = view.eyes()
    pl = geom.pixel_to_world(il, jl)
    pr = geom.pixel_to_world(ir, jr)
    # Cross-eyed: right eye -> left dot, left eye -> right dot.
    a, b = _closest_points(right, pl - right, left, pr - left)
    assert np.linalg.norm(a - b) < 0.05
    assert np.linalg.norm((a + b) / 2 - g) < 0.05
    assert il < DEV.width_px / 2 < ir


@pytest.mark.parametrize("hinge", [100, 120, 140])
def test_guide_dots_inside_band(hinge):
    geom = FoldGeometry(DEV, hinge)
    nb = guide_band_rows(DEV)
    r = guides.DOT_RADIUS_MM / DEV.mm_per_px
    for i, j in guides.guide_dot_pixels(geom, DEFAULT_VIEW):
        assert r <= j + 0.5 and j + 0.5 + r <= nb
        assert 0 <= i < DEV.width_px
        assert j == pytest.approx((nb - 1) / 2, abs=1e-6)


def test_parallel_mode_dots_each_on_own_side():
    view = DEFAULT_VIEW.with_(mode="parallel")
    geom = FoldGeometry(DEV, 120)
    (il, jl), (ir, jr) = guides.guide_dot_pixels(geom, view)
    g = guides.guide_world_point(geom, view)
    assert g[2] < 0
    left, right = view.eyes()
    a, b = _closest_points(left, geom.pixel_to_world(il, jl) - left,
                           right, geom.pixel_to_world(ir, jr) - right)
    assert np.linalg.norm(a - b) < 0.05 and np.linalg.norm(a - g) < 0.05


def test_draw_guide_band_only_band_rows():
    geom = FoldGeometry(DEV, 120)
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (DEV.height_px, DEV.width_px, 3), dtype=np.uint8)
    orig = img.copy()
    out = guides.draw_guide_band(img, geom, DEFAULT_VIEW, guides.default_label("cube", 120, DEFAULT_VIEW))
    nb = guide_band_rows(DEV)
    assert out.shape == orig.shape and out.dtype == np.uint8
    assert np.array_equal(out[nb:], orig[nb:])
    band = out[:nb]
    assert np.all(band[..., 0] == band[..., 1])          # grey
    assert np.median(band) == guides.BAND_BG              # mostly dark grey
    # Dot centres are white.
    for i, j in guides.guide_dot_pixels(geom, DEFAULT_VIEW):
        assert out[int(round(j)), int(round(i))].min() >= 240
    # Label pixels exist in the left part.
    assert (band[:, : DEV.width_px // 4, 0] > 120).any()
    # Hinge ticks at the bottom edge flank the hinge.
    bottom = band[-1, :, 0]
    hw = DEV.width_px // 2
    assert bottom[hw - 4: hw + 4].max() > guides.BAND_BG + 30


def test_draw_guide_band_promotes_grey():
    geom = FoldGeometry(DEV, 120)
    img = np.full((DEV.height_px, DEV.width_px), 77, np.uint8)
    out = guides.draw_guide_band(img, geom, DEFAULT_VIEW, "x")
    assert out.shape == (DEV.height_px, DEV.width_px, 3)
    assert (out[guide_band_rows(DEV):] == 77).all()


def _project_cyc(geom, view, pts):
    e = view.cyclopean_eye()
    i, j, ok = geom.project_from_eye(e, pts)
    return i, j, ok


@pytest.mark.parametrize("hinge", [100, 120, 140])
def test_alignment_square_drawn_where_cyclopean_eye_sees_it(hinge):
    geom = FoldGeometry(DEV, hinge)
    view = DEFAULT_VIEW
    img = guides.alignment_image(geom, view, "test")
    assert img.shape == (DEV.height_px, DEV.width_px, 3) and img.dtype == np.uint8
    nb = guide_band_rows(DEV)
    half = guides.alignment_figure_half_size(geom, view)
    e = view.cyclopean_eye()
    t = np.linspace(-half, half, 41)
    pts = []
    for s in (-1, 1):
        pts += [(x, s * half) for x in t] + [(s * half, y) for y in t]
    th = np.linspace(0, 2 * np.pi, 60, endpoint=False)
    pts += list(zip(half * np.cos(th), half * np.sin(th)))
    pts += [(x, 0.0) for x in np.linspace(-45, 45, 31)] + [(0.0, y) for y in t]
    P = np.array([[e[0] + x, e[1] + y, view.z_background] for x, y in pts])
    i, j, ok = _project_cyc(geom, view, P)
    assert ok.all()
    ii = np.round(i).astype(int)
    jj = np.round(j).astype(int)
    assert (jj >= nb).all()
    lum = img[nb:].max(axis=2)
    bright = img.max(axis=2)[jj, ii]
    assert (bright >= 150).mean() > 0.97, np.sort(bright)[:10]
    # Mostly dark elsewhere.
    assert (lum < 40).mean() > 0.8


def test_alignment_square_is_square_at_180():
    geom = FoldGeometry(DEV, 180)
    img = guides.alignment_image(geom, DEFAULT_VIEW)
    nb = guide_band_rows(DEV)
    half = guides.alignment_figure_half_size(geom, DEFAULT_VIEW)
    white = (img[..., 0] > 200) & (img[..., 1] > 200) & (img[..., 2] > 200)
    white[:nb] = False
    jj, ii = np.nonzero(white)
    w = ii.max() - ii.min()
    h = jj.max() - jj.min()
    assert abs(w - h) <= 1
    # Size: plane at z_bg seen from 300 mm on a flat display at z=0 is magnified 300/(300-z_bg).
    side_px = 2 * half * 300.0 / (300.0 - DEFAULT_VIEW.z_background) / DEV.mm_per_px
    assert w == pytest.approx(side_px, abs=3)
    # Edges are axis-aligned: each white row inside the square has pixels at the two columns.
    cx = (ii.max() + ii.min()) / 2
    cy = (jj.max() + jj.min()) / 2
    assert abs(cx - (DEV.width_px - 1) / 2) <= 1.0
    col_l = white[jj.min() + 5: jj.max() - 5, ii.min(): ii.min() + 3].any(axis=1)
    assert col_l.all()
    row_t = white[jj.min(): jj.min() + 3, ii.min() + 5: ii.max() - 5].any(axis=0)
    assert row_t.all()


def test_alignment_lines_kink_unless_folded():
    # At 120 deg the top edge of the square is a V in pixel space (peaks at the hinge);
    # at 180 deg it is flat.
    for hinge, expect_kink in ((120, True), (180, False)):
        geom = FoldGeometry(DEV, hinge)
        img = guides.alignment_image(geom, DEFAULT_VIEW)
        nb = guide_band_rows(DEV)
        white = (img.min(axis=2) > 200)
        white[:nb] = False
        cols = np.nonzero(white.any(axis=0))[0]
        top_at = lambda c: np.nonzero(white[:, c])[0].min()
        hinge_c = DEV.width_px // 2
        diff = top_at(cols.min() + 4) - top_at(hinge_c)
        assert (diff > 3) == expect_kink


# ----------------------------------------------------------------- fold guide
def _fold_edge_rows(image, geom_viewed, eye):
    """Rows (retinal px) of the picture frame's bottom edge at columns left and right of the hinge,
    as seen from `eye` with the phone folded to geom_viewed."""
    from duovision import verify

    r = verify.render_eye_view(image, geom_viewed, eye, res=(1500, 1060))
    white = (r["image"].min(axis=2) > 150)             # the bright outer frame (the inner rules are dimmer)
    f = r["focal_px"]
    hx = r["cx"] + f * (-eye[0] / eye[2])                # retinal column of the hinge (x = 0, z = 0)
    out = {}
    for name, (x0, x1) in (("left", (-0.16, -0.03)), ("right", (0.03, 0.16))):
        cols = np.arange(int(round(hx + x0 * f)), int(round(hx + x1 * f)) + 1)
        cols = cols[white[:, cols].any(axis=0)]          # an off-centre eye sees the picture shifted
        rows = np.array([np.nonzero(white[:, c])[0].max() for c in cols], dtype=float)
        out[name] = (cols.astype(float), rows)
    return out


def _kink_deg(edges):
    import math
    sl = [np.polyfit(*edges[k], 1)[0] for k in ("left", "right")]
    return math.degrees(math.atan(sl[1]) - math.atan(sl[0]))


@pytest.mark.parametrize("hinge", [105, 120, 140])
def test_fold_guide_shows_the_target_through_the_cyclopean_eye(hinge):
    """Points of the picture (dial ring, frame, empty zones) appear at the display pixels
    their cyclopean rays pass through."""
    geom = FoldGeometry(DEV, hinge)
    view = DEFAULT_VIEW
    img = guides.fold_guide_image(geom, view).astype(int)
    assert img.shape == (DEV.height_px, DEV.width_px, 3)
    half_w, half_h = guides.fold_guide_extent(geom, view)
    lay = guides.fold_guide_layout(half_w, half_h)
    e = view.cyclopean_eye()

    def patch(x, y):                      # 3x3 display neighbourhood of plane point (x, y)
        i, j, ok = geom.project_from_eye(e, np.array([e[0] + x, e[1] + y, view.z_background]))
        assert bool(ok)
        i, j = int(round(float(i))), int(round(float(j)))
        return img[j - 1:j + 2, i - 1:i + 2].reshape(-1, 3)

    gold = np.array(guides.FOLD_GOLD)
    r = lay["ring_r"]
    for ang in np.radians(np.arange(7.5, 360, 15)):
        on = patch(r * np.cos(ang), r * np.sin(ang))
        best = on[np.argmax(on.sum(axis=1))]
        assert best.sum() > 0.45 * gold.sum() and best[0] > best[2] + 15, (ang, best)      # the gold ring
        gap = patch(0.5 * (lay["inner_r"] + lay["tick_r0_long"]) * np.cos(ang), 0.5 * (lay["inner_r"] + lay["tick_r0_long"]) * np.sin(ang))
        assert gap.max() < 40, (ang, gap.max())                                              # nothing between dial and ticks
    for x in np.linspace(-0.9, 0.9, 13) * half_w:
        for sgn in (-1, 1):
            assert patch(x, sgn * half_h).max() > 110                                        # the frame, top and bottom
            assert patch(x, sgn * (half_h + 3.0)).max() < 40                                 # black outside it
    for y in np.linspace(-0.9, 0.9, 9) * half_h:
        for sgn in (-1, 1):
            assert patch(sgn * half_w, y).max() > 110                                        # the frame's sides


def test_fold_guide_is_straight_only_at_the_encoded_fold():
    geom = FoldGeometry(DEV, 120)
    view = DEFAULT_VIEW
    img = guides.fold_guide_image(geom, view)
    e = view.cyclopean_eye()
    ok = _kink_deg(_fold_edge_rows(img, geom, e))
    flatter = _kink_deg(_fold_edge_rows(img, FoldGeometry(DEV, 150), e))
    deeper = _kink_deg(_fold_edge_rows(img, FoldGeometry(DEV, 95), e))
    assert abs(ok) < 0.35                       # straight across the hinge
    # Bottom edge: flatter -> the middle hangs lower (tall in the middle), deeper -> pinched.
    assert flatter < -2.0 and deeper > 2.0
    # Each eye sees the same bend as the cyclopean eye: the cue does not depend on which eye looks.
    left, _right = view.eyes()
    assert abs(_kink_deg(_fold_edge_rows(img, FoldGeometry(DEV, 150), left)) - flatter) < 0.5


def test_fold_guide_band_is_left_black():
    geom = FoldGeometry(DEV, 120)
    img = guides.fold_guide_image(geom, DEFAULT_VIEW)
    nb = guide_band_rows(DEV)
    assert (img[:nb] == np.array(guides.FOLD_BG)).all()          # no text, no dots: the viewer's guide explains it
    below = img[nb:].astype(int)
    assert ((below[..., 0] > 150) & (below[..., 0] > below[..., 2] + 40)).sum() > 100    # the gold ring is there
