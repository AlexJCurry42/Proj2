"""Black-box tests for patterns.colorize."""
import numpy as np
import pytest

from bbphys import DEV

patterns = pytest.importorskip("duovision.patterns")
H, W = DEV.height_px, DEV.width_px


def shared_roots(seed=0, n_roots=400):
    rng = np.random.default_rng(seed)
    root_set = rng.choice(H * W, n_roots, replace=False)
    roots = rng.choice(root_set, H * W)
    roots[root_set] = root_set  # roots are fixed points
    return roots.reshape(H, W)


def identity_roots():
    return np.arange(H * W).reshape(H, W)


def test_styles_contract():
    assert tuple(patterns.STYLES) == ("bw", "color", "texture")


@pytest.mark.parametrize("style", ["bw", "color", "texture"])
def test_colour_is_pure_function_of_root(style):
    roots = shared_roots()
    img = np.asarray(patterns.colorize(roots, DEV, style, seed=3))
    assert img.shape == (H, W, 3) and img.dtype == np.uint8
    flat = roots.ravel()
    cols = img.reshape(-1, 3)
    order = np.argsort(flat, kind="stable")
    r = flat[order]
    c = cols[order]
    same_root = r[1:] == r[:-1]
    assert (c[1:][same_root] == c[:-1][same_root]).all()
    # each pixel has its root pixel's colour
    assert (cols == cols[flat]).all()
    # deterministic
    img2 = np.asarray(patterns.colorize(roots, DEV, style, seed=3))
    assert np.array_equal(img, img2)


def test_bw_is_black_and_white_only():
    img = np.asarray(patterns.colorize(identity_roots(), DEV, "bw", seed=0))
    assert set(np.unique(img)) <= {0, 255}
    assert (img[..., 0] == img[..., 1]).all() and (img[..., 1] == img[..., 2]).all()
    frac_white = (img[..., 0] == 255).mean()
    assert 0.1 < frac_white < 0.9


def test_styles_differ_and_color_is_colourful():
    roots = identity_roots()
    imgs = {s: np.asarray(patterns.colorize(roots, DEV, s, seed=0)) for s in ("bw", "color", "texture")}
    assert not np.array_equal(imgs["bw"], imgs["color"])
    assert not np.array_equal(imgs["bw"], imgs["texture"])
    assert not np.array_equal(imgs["color"], imgs["texture"])
    c = imgs["color"].reshape(-1, 3)
    assert len(np.unique(c, axis=0)) > 8
    assert (c.max(axis=1).astype(int) - c.min(axis=1)).mean() > 10  # not grey


@pytest.mark.parametrize("style", ["bw", "color", "texture"])
def test_seed_changes_pattern(style):
    roots = identity_roots()
    a = np.asarray(patterns.colorize(roots, DEV, style, seed=0))
    b = np.asarray(patterns.colorize(roots, DEV, style, seed=1))
    assert not np.array_equal(a, b)
