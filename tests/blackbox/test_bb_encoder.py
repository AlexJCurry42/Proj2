"""Black-box end-to-end tests of encode_stereogram / compute_partner_map."""
import numpy as np
import pytest

from bbphys import background_rows, encoded, geom_view, own_roots, physics_partners, pick, scene_for
from duovision.config import guide_band_rows

pytest.importorskip("duovision.encoder")
pytest.importorskip("duovision.scene")

CASES = [
    ("plaque", 120, "bw"),
    ("cube", 100, "color"),
    ("sphere", 140, "texture"),
    ("text", 120, "color"),
    ("torus", 130, "bw"),
    ("pyramid", 100, "texture"),
    ("heart", 140, "color"),
    ("pawn", 110, "bw"),
]
IDS = [f"{a}-{b}-{c}" for a, b, c in CASES]


@pytest.mark.parametrize("name,hinge,style", CASES, ids=IDS)
def test_image_shape_band_and_link_consistency(name, hinge, style):
    geom, view, fold, scene, res = encoded(name, hinge, style)
    H, W = fold.H, fold.W
    band = guide_band_rows(geom.device)
    img = res.image
    assert img.shape == (H, W, 3) and img.dtype == np.uint8
    assert (img[:band] == 0).all(), "guide band must be left black"
    pm = res.partner_map
    assert pm.reserved_rows == band
    for a in (pm.partner_i, pm.partner_j, pm.linked, pm.hit_z, pm.occluded):
        assert a.shape == (H, W)
    assert not pm.linked[:band].any()
    pi, pj = np.asarray(res.pi), np.asarray(res.pj)
    assert pi.shape == (H, W) and pj.shape == (H, W)
    assert np.issubdtype(pi.dtype, np.integer) and np.issubdtype(pj.dtype, np.integer)
    lk = pi >= 0
    # CONTRACT-AMBIGUITY: rounding could in principle push a partner into the band;
    # we require rounded links to be a subset of pm.linked and to cover >= 99.9%.
    assert not (lk & ~pm.linked).any()
    assert (lk == pm.linked).mean() > 0.999
    assert (pi[~lk] == -1).all()
    jj, ii = np.nonzero(lk)
    qi, qj = pi[lk], pj[lk]
    assert (qi >= 0).all() and (qi < W).all() and (qj >= band).all() and (qj < H).all()
    # cross mode: every link points left
    assert (qi <= ii - 1).all()
    assert (pm.partner_i[pm.linked] <= np.nonzero(pm.linked)[1] - 1 + 1e-9).all()
    assert np.isnan(pm.partner_i[~pm.linked]).all()
    # 100% of links have identical colour
    same = (img[jj, ii] == img[qj, qi]).all(axis=1)
    assert same.all(), f"{(~same).sum()} of {same.size} linked pixels mismatch"
    # rounded partners are floor/ceil of the fractional ones
    fi = pm.partner_i[lk]
    fj = pm.partner_j[lk]
    assert (np.abs(qi - fi) < 1.0 + 1e-9).all() and (np.abs(qj - fj) < 1.0 + 1e-9).all()
    # sanity: most encodable pixels are linked
    assert pm.linked[band:].mean() > 0.6


@pytest.mark.parametrize("name,hinge,style", CASES[:4], ids=IDS[:4])
def test_roots_are_pointer_chain_roots(name, hinge, style):
    _, _, fold, _, res = encoded(name, hinge, style)
    roots = np.asarray(res.roots)
    assert roots.shape == (fold.H, fold.W)
    assert np.array_equal(roots, own_roots(np.asarray(res.pi), np.asarray(res.pj)))


@pytest.mark.parametrize("name,hinge,style", CASES, ids=IDS)
def test_partners_match_independent_physics(name, hinge, style):
    geom, view, fold, scene, res = encoded(name, hinge, style)
    pm = res.partner_map
    band = guide_band_rows(geom.device)
    rng = np.random.default_rng(42)
    n = 400
    j = rng.integers(band, fold.H, n)
    i = rng.integers(0, fold.W, n)
    ph = physics_partners(scene, fold, view, i, j, band)
    use = ~ph["ambiguous"]
    imp_linked = pm.linked[j, i]
    agree = (imp_linked == ph["linked"])[use]
    assert agree.mean() >= 0.97, f"link status agreement {agree.mean():.3f}"
    both = use & imp_linked & ph["linked"]
    assert both.sum() > 0.5 * n
    di = np.abs(pm.partner_i[j, i] - ph["pi"])[both]
    dj = np.abs(pm.partner_j[j, i] - ph["pj"])[both]
    d = np.maximum(di, dj)
    bad = d > 0.05
    assert bad.mean() <= 0.01, (
        f"{bad.sum()}/{both.sum()} partners off by >0.05 px; median {np.median(d):.4f}, max {d.max():.3f}"
    )
    assert np.median(d) < 0.01
    # hit_z is the z of S seen by the LEFT eye
    hz = np.abs(pm.hit_z[j, i] - ph["S"][:, 2])[both]
    assert (hz < 0.02).mean() >= 0.99, f"hit_z max err {hz.max():.3f}"
    # occluded agrees with own visibility test where the left ray hits
    occ_use = use & ph["hitL"]
    occ_agree = (pm.occluded[j, i] == ~ph["visible"])[occ_use]
    assert occ_agree.mean() >= 0.97


def test_flat_background_separation_is_15mm_at_hinge_180():
    """On a flat screen a flat background at z_bg must have separation exactly sep_bg = 15 mm."""
    geom, view, fold, scene, res = encoded("plaque", 180, "bw", bg_vee=0.0)
    band = guide_band_rows(geom.device)
    rows = [r for name in ["plaque"] for r in background_rows(scene, fold, view, 0.0, band)]
    if not rows:
        pytest.skip("plaque has no background-only row")
    jrow = rows[0]
    pm = res.partner_map
    cols = np.arange(fold.W)
    u, _ = fold.pix_to_uv(cols, jrow)
    sel = (u > -10) & (u < 30) & pm.linked[jrow]
    assert sel.sum() > 20
    up, _ = fold.pix_to_uv(pm.partner_i[jrow, sel], jrow)
    sep = u[sel] - up
    assert np.abs(sep - 15.0).max() < fold.mmpp, f"separation {sep.min():.3f}..{sep.max():.3f} mm"
    assert np.abs(pm.partner_j[jrow, sel] - jrow).max() < 0.05


def test_background_separation_at_hinge_axis_folded():
    """At hinge 120 with a flat background, the pair straddling the hinge for the scene point
    (0, y, z_bg) must be where the eyes' rays through that point land (own maths)."""
    geom, view, fold, scene, res = encoded("plaque", 120, "bw", bg_vee=0.0)
    band = guide_band_rows(geom.device)
    rows = background_rows(scene, fold, view, 0.0, band)
    if not rows:
        pytest.skip("no background-only row")
    jrow = rows[0]
    _, v = fold.pix_to_uv(0, jrow)
    L, R = view.eyes()
    # find y_S such that the left-eye ray through S=(0,y_S,z_bg) lands on row jrow: solve by iteration
    S = np.array([0.0, v * (view.distance_mm - view.z_background) / view.distance_mm, view.z_background])
    for _ in range(50):
        uL, vL, _, _ = fold.hit_display(L, S - L)
        S[1] *= v / vL
    uL, vL, _, _ = fold.hit_display(L, S - L)
    uR, vR, _, _ = fold.hit_display(R, S - R)
    iL, jL = fold.uv_to_pix(uL, vL)
    iR, jR = fold.uv_to_pix(uR, vR)
    assert abs(jL - jrow) < 1e-6
    # interpolate the implementation's partner_i at fractional column iL
    k = int(np.floor(iL))
    w = iL - k
    pm = res.partner_map
    assert pm.linked[jrow, k] and pm.linked[jrow, k + 1]
    p_imp = (1 - w) * pm.partner_i[jrow, k] + w * pm.partner_i[jrow, k + 1]
    assert abs(p_imp - iR) < 0.05
    # and in mm along the surface this is ~15 mm (exactly 15 only on a flat screen)
    sep_mm = (iL - iR) * fold.mmpp
    assert 10 < sep_mm < 20


def test_determinism_and_seed_dependence():
    from duovision.encoder import encode_stereogram

    geom, view, _ = geom_view(120)
    scene = scene_for("sphere", 120)
    a = encode_stereogram(geom, view, scene, "color", seed=7)
    b = encode_stereogram(geom, view, scene, "color", seed=7)
    c = encode_stereogram(geom, view, scene, "color", seed=8)
    assert np.array_equal(a.image, b.image)
    assert np.array_equal(a.pi, b.pi) and np.array_equal(a.roots, b.roots)
    assert not np.array_equal(a.image, c.image)


@pytest.mark.parametrize("name,hinge,style", CASES[:3], ids=IDS[:3])
def test_stats_are_sane(name, hinge, style):
    geom, view, fold, scene, res = encoded(name, hinge, style)
    st = res.stats
    assert isinstance(st, dict)
    from bbphys import flatten

    flat = flatten(st)
    band = guide_band_rows(geom.device)
    encodable = (fold.H - band) * fold.W
    pm = res.partner_map
    # CONTRACT-AMBIGUITY: key names are not specified ("timings, linked fraction,
    # occluded fraction, max separation px"); we look them up by substring.
    assert any("tim" in k.lower() for k in flat), f"no timing key in {sorted(flat)}"
    for k, v in flat.items():
        if "frac" in k.lower():
            assert 0.0 <= float(v) <= 1.0, (k, v)
    k_l, linked = pick(st, ("linked", "frac"), ("linked",))
    n_linked = pm.linked.sum()
    assert any(abs(linked - n_linked / d) < 0.01 for d in (encodable, fold.H * fold.W)) or abs(
        linked - n_linked
    ) < 1, (k_l, linked, n_linked / encodable)
    k_o, occ = pick(st, ("occlu", "frac"), ("occlu",))
    assert 0.0 <= occ <= 1.0 or occ <= fold.H * fold.W
    try:
        k_f, free = pick(st, ("free", "frac"), ("free",))
    except AssertionError:
        free = None
    if free is not None:
        if linked <= 1.0 and free <= 1.0:
            assert abs(linked + free - 1.0) < 0.02 or abs((linked + free) - encodable / (fold.H * fold.W)) < 0.02
        else:
            assert abs(linked + free - encodable) <= 0.01 * encodable
    k_s, sep = pick(st, ("max", "sep"), ("sep",))
    jj, ii = np.nonzero(pm.linked)
    own_max = (ii - pm.partner_i[jj, ii]).max()
    assert 0.9 * own_max <= sep <= 1.1 * own_max + 1, (k_s, sep, own_max)


def test_half_scale_plaque_consistency():
    geom, view, fold, scene, res = encoded("plaque", 120, "color", scale=0.5)
    band = guide_band_rows(geom.device)
    assert res.image.shape == (fold.H, fold.W, 3)
    assert (res.image[:band] == 0).all()
    lk = res.pi >= 0
    jj, ii = np.nonzero(lk)
    assert (res.image[jj, ii] == res.image[res.pj[lk], res.pi[lk]]).all()
    assert (res.pi[lk] <= ii - 1).all()


def test_parallel_mode_links_point_right():
    geom, view, fold, scene, res = encoded("plaque", 120, "bw", mode="parallel")
    lk = res.pi >= 0
    assert lk.mean() > 0.5
    jj, ii = np.nonzero(lk)
    assert (res.pi[lk] >= ii + 1).all()
    assert (res.image[jj, ii] == res.image[res.pj[lk], res.pi[lk]]).all()
