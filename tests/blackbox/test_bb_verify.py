"""Black-box tests for verify.py and baseline.py."""
import inspect

import numpy as np
import pytest

from bbphys import background_rows, encoded, geom_view, pick, scene_for
from duovision.config import guide_band_rows

verify = pytest.importorskip("duovision.verify")

MATCH_KEYS = (("match", "rate"), ("match", "frac"), ("match",))
ERR_KEYS = (("median", "abs", "err"), ("median", "err"), ("mae",))
KINK_KEYS = (("plaque", "kink"), ("kink",))


def link_rates(out):
    """(rounding_aware_rate or None, literal_rate) from a check_links dict.

    CONTRACT-AMBIGUITY: key names are not in the contract and changed during
    development ('match_rate' / 'match_rate_exact' / 'match_rate_rounding_aware').
    """
    keys = {k.lower(): k for k in out}
    aware_k = next((keys[k] for k in keys if "match" in k and "rounding" in k), None)
    exact_k = next((keys[k] for k in keys if "match" in k and "exact" in k), None)
    if aware_k is None and exact_k is not None:
        aware_k = "match_rate"
    literal_k = exact_k or "match_rate"
    aware = float(out[aware_k]) if aware_k else None
    return aware, float(out[literal_k])


CL_CASES = [("plaque", 120, "bw"), ("sphere", 100, "color"), ("text", 140, "texture")]


@pytest.mark.parametrize("name,hinge,style", CL_CASES)
def test_check_links_on_correct_encode(name, hinge, style):
    """A correct encode scores high, and check_links agrees with our own reverse trace.

    PHYSICS NOTE: under the contract's randomized rounding the literal rule
    image[round(p)] == image[q] cannot reach 0.99 (a 'confident' reverse link with
    |frac| < 0.25 has a forward partner up to ~0.25 px from q, and rounding then picks
    q's neighbour with that probability; measured unbiased P(ceil) == frac). The
    implementation reports that literal rate as 'match_rate_exact' and a rounding-aware
    'match_rate'. We require match_rate >= 0.99 and check match_rate_exact against our
    own independent re-implementation of the literal rule.
    """
    from bbphys import own_reverse_match_rate

    geom, view, fold, scene, res = encoded(name, hinge, style)
    band = guide_band_rows(geom.device)
    out = verify.check_links(res.image, geom, view, scene)
    assert isinstance(out, dict)
    aware, exact = link_rates(out)
    own, n_own = own_reverse_match_rate(res.image, scene, fold, view, band)
    assert n_own > 300
    assert abs(exact - own) < 0.03, (exact, own)
    assert aware is not None, f"no rounding-aware match rate reported: {sorted(out)}"
    assert aware >= 0.99, aware
    tight = verify.check_links(res.image, geom, view, scene, frac_tol=0.02)
    _, exact_t = link_rates(tight)
    assert exact_t >= 0.95, exact_t


def test_check_links_on_noise():
    geom, view, fold, scene, res = encoded("plaque", 120, "bw")
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 256, res.image.shape, dtype=np.uint8)
    out = verify.check_links(noise, geom, view, scene)
    aware, exact = link_rates(out)
    assert exact <= 0.2, exact
    if aware is not None:
        assert aware <= 0.2, aware


def test_check_links_detects_wrong_hinge():
    """An image encoded for 100 deg checked against a 140 deg fold should match much worse."""
    geom, view, fold, scene, res = encoded("plaque", 100, "color")
    g140, _, _ = geom_view(140)
    out = verify.check_links(res.image, g140, view, scene_for("plaque", 140))
    aware, exact = link_rates(out)
    assert exact < 0.9
    if aware is not None:
        assert aware < 0.9, aware


@pytest.fixture(scope="module")
def plaque_eval():
    geom, view, fold, scene, res = encoded("plaque", 120, "color")
    return verify.evaluate(res.image, geom, view, scene)


def test_evaluate_fold_aware_plaque(plaque_eval):
    assert isinstance(plaque_eval, dict)
    k, err = pick(plaque_eval, *ERR_KEYS)
    assert err < 1.5, (k, err)
    # CONTRACT-AMBIGUITY: the kink metric's units/definition are not in the contract;
    # "small" taken as < 2 (mm or degrees).
    k2, kink = pick(plaque_eval, *KINK_KEYS)
    assert abs(kink) < 2.0, (k2, kink)


def _flat_sirds_image(depth_z, geom, view, seed=0):
    """Call baseline.flat_sirds with whatever keyword arguments its signature asks for.

    CONTRACT-AMBIGUITY: the contract only gives `flat_sirds(depth_map, ...)`. The public
    docstring (not the contract) says depth_z_map is world z in mm per flat-display pixel,
    NaN = unconstrained; we pass world z. (An earlier version of this test assumed
    Thimbleby's [0,1] convention and was wrong.)
    """
    baseline = pytest.importorskip("duovision.baseline")
    sig = inspect.signature(baseline.flat_sirds)
    cand = {
        "device": geom.device, "view": view, "seed": seed, "geom": geom, "geometry": geom,
        "rng": np.random.default_rng(seed), "style": "bw",
    }
    kwargs = {}
    params = list(sig.parameters.values())[1:]
    for p in params:
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        if p.name in cand:
            kwargs[p.name] = cand[p.name]
        elif p.default is p.empty:
            pytest.fail(f"flat_sirds requires unknown parameter {p.name!r}: {sig}")
    out = baseline.flat_sirds(depth_z, **kwargs)
    if isinstance(out, (tuple, list)):
        out = next(o for o in out if isinstance(o, np.ndarray))
    elif hasattr(out, "image"):
        out = out.image
    out = np.asarray(out)
    if out.dtype == bool:
        out = out.astype(np.uint8) * 255
    if out.ndim == 2:
        out = np.repeat(out[..., None], 3, axis=2)
    assert out.shape == (geom.device.height_px, geom.device.width_px, 3), out.shape
    return out.astype(np.uint8)


def test_flat_sirds_background_period_is_sep_bg():
    geom, view, fold = geom_view(120)
    img = _flat_sirds_image(np.full((fold.H, fold.W), view.z_background), geom, view)
    rows = img[fold.H // 2: fold.H // 2 + 40, :, 0].astype(int)
    best, best_k = -1, None
    for k in range(10, 200):
        m = (rows[:, k:] == rows[:, :-k]).mean()
        if m > best:
            best, best_k = m, k
    assert best > 0.95
    assert abs(best_k * fold.mmpp - view.sep_bg_mm) <= fold.mmpp + 1e-9, (best_k, best_k * fold.mmpp)


def test_flat_baseline_on_fold_is_worse(plaque_eval):
    from duovision.scene import first_hit

    geom, view, fold, scene, res = encoded("plaque", 120, "color")
    # Depth map as a flat-screen SIRDS designer would see it: scene depth along
    # cyclopean rays through a flat screen at the hinge plane.
    g180, _, f180 = geom_view(180)
    T = f180.pix_to_world(*np.meshgrid(np.arange(fold.W), np.arange(fold.H)))
    pts, hit = first_hit(scene, view.cyclopean_eye(), T)
    z = np.where(np.asarray(hit), np.asarray(pts)[..., 2], view.z_background)
    img = _flat_sirds_image(z, geom, view)
    base = verify.evaluate(img, geom, view, scene)
    _, e_fold = pick(plaque_eval, *ERR_KEYS)
    _, e_base = pick(base, *ERR_KEYS)
    _, k_fold = pick(plaque_eval, *KINK_KEYS)
    _, k_base = pick(base, *KINK_KEYS)
    assert e_base > e_fold, (e_base, e_fold)
    assert abs(k_base) > abs(k_fold), (k_base, k_fold)
    print("fold-aware err/kink", e_fold, k_fold, "| flat baseline err/kink", e_base, k_base)


def test_comfort_map_at_hinge_axis_flat_background():
    geom, view, fold = geom_view(120)
    scene = scene_for("plaque", 120, 0.0)
    cm = np.asarray(verify.comfort_map(geom, view, scene))
    assert cm.shape == (fold.H, fold.W)
    band = guide_band_rows(geom.device)
    rows = background_rows(scene, fold, view, 0.0, band)
    assert rows, "no background-only row"
    expected = view.sep_bg_mm / (view.ipd_mm * view.distance_mm / 1000.0)  # 0.794 D
    assert expected == pytest.approx(0.794, abs=1e-3)
    j = rows[0]
    vals = np.abs(cm[j, fold.W // 2 - 1: fold.W // 2 + 1])
    # off-centre rows lengthen both distances slightly; allow 0.04 D
    assert np.all(np.abs(vals - expected) < 0.04), (j, vals)
