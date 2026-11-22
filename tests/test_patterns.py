"""Tests for chain colouring patterns."""
import numpy as np
import pytest

from duovision.config import DEFAULT_DEVICE
from duovision.patterns import PALETTE, STYLES, TEXTURES, colorize, fbm, value_noise
from duovision.resolve import resolve_roots

DEV = DEFAULT_DEVICE.scaled(0.25)
H, W = DEV.height_px, DEV.width_px


def leftward_roots(seed=0, sep=60):
    """Roots of a simple periodic chain field (every pixel links sep px left)."""
    pi = np.arange(W)[None, :] - sep + np.random.default_rng(seed).integers(-1, 2, (H, W))
    pi = np.where(pi >= 0, pi, -1).astype(np.int32)
    pj = np.where(pi >= 0, np.arange(H)[:, None], -1).astype(np.int32)
    return resolve_roots(pi, pj)


def identity_roots():
    return np.arange(H * W).reshape(H, W)


def test_styles_constant():
    assert STYLES == ("bw", "color", "texture")
    assert {"marble", "waves", "leaves"} <= set(TEXTURES)


@pytest.mark.parametrize("style,tex", [("bw", None), ("color", None), ("texture", "marble"),
                                       ("texture", "waves"), ("texture", "leaves")])
def test_purity_same_root_same_colour(style, tex):
    roots = leftward_roots()
    kw = {"texture": tex} if tex else {}
    img = colorize(roots, DEV, style, seed=3, **kw)
    assert img.shape == (H, W, 3) and img.dtype == np.uint8
    flat = img.reshape(-1, 3)
    r = roots.ravel()
    assert np.array_equal(flat, flat[r])  # every pixel has its root's colour
    # colour depends only on root coordinates: recolouring the roots alone gives the same values
    ident = colorize(identity_roots(), DEV, style, seed=3, **kw).reshape(-1, 3)
    assert np.array_equal(flat, ident[r])


@pytest.mark.parametrize("style", ["bw", "color", "texture"])
def test_deterministic_and_seed_sensitive(style):
    roots = leftward_roots()
    a = colorize(roots, DEV, style, seed=11)
    b = colorize(roots, DEV, style, seed=11)
    c = colorize(roots, DEV, style, seed=12)
    assert np.array_equal(a, b)
    assert (a != c).any(axis=-1).mean() > 0.3


def test_styles_differ():
    roots = identity_roots()
    imgs = [colorize(roots, DEV, s, seed=0) for s in STYLES]
    imgs += [colorize(roots, DEV, "texture", texture=t) for t in TEXTURES if t != "marble"]
    for a in range(len(imgs)):
        for b in range(a + 1, len(imgs)):
            assert (imgs[a] != imgs[b]).any(axis=-1).mean() > 0.5


def test_bw_dots():
    img = colorize(identity_roots(), DEV, "bw", seed=4, dot_px=5)
    vals = np.unique(img)
    assert set(vals.tolist()) <= {0, 255}
    assert np.all(img[..., 0] == img[..., 1]) and np.all(img[..., 1] == img[..., 2])
    white = (img[..., 0] == 255).mean()
    assert 0.47 < white < 0.53
    # constant on dot cells
    cell = img[:5 * (H // 5), :5 * (W // 5), 0].reshape(H // 5, 5, W // 5, 5)
    assert np.all(cell == cell[:, :1, :, :1])
    img2 = colorize(identity_roots(), DEV, "bw", seed=4, dot_px=2)
    assert not np.array_equal(img, img2)


def test_color_palette():
    img = colorize(identity_roots(), DEV, "color", seed=1, dot_px=4)
    cols = np.unique(img.reshape(-1, 3), axis=0)
    pal = {tuple(c) for c in PALETTE.tolist()}
    assert {tuple(c) for c in cols.tolist()} <= pal
    assert 12 <= len(PALETTE) <= 16
    assert len(cols) == len(PALETTE)
    counts = np.array([(img.reshape(-1, 3) == c).all(-1).sum() for c in cols])
    assert counts.min() > 0.5 * counts.mean()  # roughly uniform use of the palette


@pytest.mark.parametrize("tex", ["marble", "waves", "leaves"])
def test_texture_is_colourful_with_fine_detail(tex):
    img = colorize(identity_roots(), DEV, "texture", texture=tex, seed=0).astype(np.float32)
    # colourful: substantial chroma and hue variety
    chroma = img.max(-1) - img.min(-1)
    assert chroma.mean() > 40
    hue_bins = np.unique((np.argmax(img, -1) * 4 + (img.min(-1) > 100)).ravel())
    assert len(hue_bins) >= 3
    # fine detail: neighbouring pixels differ noticeably on average (texture for fusion)
    lum = img.mean(-1)
    dx = np.abs(np.diff(lum, axis=1)).mean()
    dx5 = np.abs(lum[:, 5:] - lum[:, :-5]).mean()
    assert dx > 1.0 and dx5 > 5.0
    assert lum.std() > 18


def test_noise_ranges():
    x = np.linspace(-50, 50, 2001, dtype=np.float32)
    n = value_noise(x, x * 0.3, 1)
    assert n.min() >= 0 and n.max() < 1 and n.std() > 0.1
    f = fbm(x, -x, 2)
    assert f.min() >= 0 and f.max() < 1
    # continuity
    assert np.abs(np.diff(value_noise(x, 0 * x, 3))).max() < 0.2


def test_invalid_inputs():
    roots = identity_roots()
    with pytest.raises(ValueError):
        colorize(roots, DEV, "plaid")
    with pytest.raises(ValueError):
        colorize(roots, DEV, "texture", texture="nope")
    with pytest.raises(ValueError):
        colorize(roots[:-1], DEV, "bw")


def test_non_selfrooted_labels_still_pure():
    # labels such as scipy's (not self-rooted) are coloured consistently too
    lab = (np.arange(H * W) // 7).reshape(H, W)
    img = colorize(lab, DEV, "color", seed=0).reshape(-1, 3)
    n = (H * W // 7) * 7
    groups = img[:n].reshape(-1, 7, 3)
    assert np.all(groups == groups[:, :1])
