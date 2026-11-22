import json
import os
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from duovision import batch as B
from duovision.cli import main
from duovision.config import DEFAULT_DEVICE, HINGE_SWEEP_DEG, guide_band_rows

SCALE = 0.15
DEV = DEFAULT_DEVICE.scaled(SCALE)
VIEWER = Path(__file__).resolve().parent.parent / "viewer" / "index.html"


def _need_pipeline():
    pytest.importorskip("duovision.scene")
    pytest.importorskip("duovision.patterns")
    pytest.importorskip("duovision.encoder")


def _meta(path):
    with Image.open(path) as im:
        return im.size, im.mode, dict(im.text)


def test_align_cli(tmp_path, capsys):
    out = tmp_path / "a.png"
    assert main(["align", "--hinge", "110", "--scale", str(SCALE), "--out", str(out)]) == 0
    size, mode, text = _meta(out)
    assert size == (DEV.width_px, DEV.height_px)
    assert mode == "P"
    assert text["duovision.kind"] == "align"
    assert float(text["duovision.hinge"]) == 110.0
    params = json.loads(text["duovision.params"])
    assert params["distance"] == 300.0 and params["scale"] == SCALE
    assert "wrote" in capsys.readouterr().out


def test_palette_roundtrip_exact(tmp_path):
    rng = np.random.default_rng(1)
    pal = rng.integers(0, 256, (200, 3), dtype=np.uint8)
    img = pal[rng.integers(0, 200, (40, 60))]
    p = tmp_path / "x.png"
    info = B.save_png(p, img, B.RenderParams(scale=SCALE), {"a": float("nan"), "b": np.float32(2)})
    assert info["png_mode"] == "P"
    back = np.asarray(Image.open(p).convert("RGB"))
    assert np.array_equal(back, img)
    img2 = rng.integers(0, 256, (40, 60, 3), dtype=np.uint8)
    assert B.save_png(p, img2, B.RenderParams(), {})["png_mode"] == "RGB"
    assert np.array_equal(np.asarray(Image.open(p).convert("RGB")), img2)
    _, stats, _ = B.read_png_meta(p)
    assert stats == {}


def test_batch_job_list_default():
    jobs = B.batch_jobs()
    files = [j.file for j in jobs]
    n_h = len(HINGE_SWEEP_DEG)
    assert len(files) == len(set(files)) == n_h + 8 * n_h + 16
    assert "align_100.png" in files and "cube_135_color.png" in files
    assert "heart_120_texture.png" in files and "pawn_120_bw.png" in files
    assert "cube_125_bw.png" not in files


@pytest.mark.parametrize("style", ["color", "bw", "texture"])
def test_generate_cli(tmp_path, style, capsys):
    _need_pipeline()
    out = tmp_path / f"g_{style}.png"
    rc = main(["generate", "--scene", "cube", "--hinge", "125", "--style", style, "--scale", str(SCALE),
               "--seed", "3", "--out", str(out)])
    assert rc == 0
    size, mode, text = _meta(out)
    assert size == (DEV.width_px, DEV.height_px)
    assert text["duovision.scene"] == "cube"
    assert text["duovision.style"] == style
    assert float(text["duovision.hinge"]) == 125.0
    assert text["duovision.seed"] == "3"
    stats = json.loads(text["duovision.stats"])
    assert stats and "time_render_s" in stats
    if style == "texture":
        assert mode == "RGB" and text["duovision.texture"] == "marble"
    arr = np.asarray(Image.open(out).convert("RGB"))
    nb = guide_band_rows(DEV)
    band = arr[:nb]
    assert (band[..., 0] == band[..., 2]).all() and band.max() == 255   # grey band with white dots
    assert arr[nb:].std() > 20                                            # pattern below
    assert "wrote" in capsys.readouterr().out


def test_generate_unknown_scene(tmp_path):
    _need_pipeline()
    assert main(["generate", "--scene", "nope", "--scale", str(SCALE), "--out", str(tmp_path / "x.png")]) == 2


def test_batch_cli_subset(tmp_path):
    _need_pipeline()
    out = tmp_path / "batch"
    args = ["batch", "--out", str(out), "--scale", str(SCALE), "--jobs", "2",
            "--scenes", "plaque", "cube", "--hinges", "115", "120"]
    assert main(args) == 0
    m = json.loads((out / "manifest.json").read_text())
    files = sorted(p.name for p in out.glob("*.png") if p.name not in B.VIEWER_FILES)
    assert sorted(e["file"] for e in m["entries"]) == files
    expected = {"align_115.png", "align_120.png", "plaque_115_color.png", "plaque_120_color.png",
                "cube_115_color.png", "cube_120_color.png", "plaque_120_bw.png", "cube_120_bw.png",
                "plaque_120_texture.png", "cube_120_texture.png"}
    assert set(files) == expected
    for e in m["entries"]:
        size, mode, text = _meta(out / e["file"])
        assert (e["width"], e["height"]) == size == (DEV.width_px, DEV.height_px)
        assert e["hinge"] == int(float(text["duovision.hinge"]))
        assert e["params"]["scale"] == SCALE
        if e["kind"] == "align":
            assert e["scene"] == "align" and e["style"] == "align"
        else:
            assert text["duovision.scene"] == e["scene"] and text["duovision.style"] == e["style"]
            if e["style"] == "texture":
                assert mode == "RGB"
            assert "linked_fraction" in e["stats"]
    gids = [g["id"] for g in m["groups"]]
    assert gids[0] == "align" and set(gids) == {"align", "plaque", "cube"}
    cube = next(g for g in m["groups"] if g["id"] == "cube")
    assert cube["hinges"] == [115, 120] and set(cube["styles"]) == {"color", "bw", "texture"}
    assert (out / "index.html").exists()
    for name in ("app.webmanifest", "icon-192.png", "icon-512.png"):
        assert (out / name).exists(), name

    # Re-run: everything is skipped, files untouched, manifest still complete.
    mtimes = {f: (out / f).stat().st_mtime_ns for f in files}
    assert main(args + ["--jobs", "1"]) == 0
    assert {f: (out / f).stat().st_mtime_ns for f in files} == mtimes
    m2 = json.loads((out / "manifest.json").read_text())
    assert sorted(e["file"] for e in m2["entries"]) == files
    assert m2["entries"][0]["stats"] == m["entries"][0]["stats"]

    # A second subset merges into the manifest.
    assert main(["batch", "--out", str(out), "--scale", str(SCALE), "--jobs", "1",
                 "--scenes", "sphere", "--hinges", "130", "--styles", "color"]) == 0
    m3 = json.loads((out / "manifest.json").read_text())
    assert {e["file"] for e in m3["entries"]} == set(files) | {"sphere_130_color.png"}


def test_batch_dry_run(capsys):
    assert main(["batch", "--out", "/nonexistent/dir", "--dry-run"]) == 0
    assert capsys.readouterr().out.strip().endswith(f"{len(B.batch_jobs())} jobs")


def test_verify_cli(tmp_path, capsys):
    _need_pipeline()
    try:
        import duovision.verify as V  # noqa: F401
        has_verify = hasattr(V, "evaluate")
    except ImportError:
        has_verify = False
    out = tmp_path / "v.png"
    assert main(["generate", "--scene", "plaque", "--hinge", "120", "--scale", "0.25", "--out", str(out)]) == 0
    capsys.readouterr()
    rc = main(["verify", "--image", str(out), "--json", "--res", "400", "280"])
    if has_verify:
        assert rc == 0
        metrics = json.loads(capsys.readouterr().out)
        assert isinstance(metrics, dict) and metrics
    else:
        assert rc == 1


class _Collector(HTMLParser):
    VOID = {"meta", "link", "img", "br", "input", "hr", "source", "area", "base", "col", "embed", "wbr", "track", "param"}

    def __init__(self):
        super().__init__()
        self.stack, self.errors, self.tags, self.urls, self.script = [], [], [], [], ""

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)
        for k, v in attrs:
            if k in ("src", "href") and v:
                self.urls.append(v)
        if tag not in self.VOID:
            self.stack.append(tag)

    def handle_endtag(self, tag):
        if not self.stack or self.stack[-1] != tag:
            self.errors.append(f"unexpected </{tag}>, open: {self.stack[-3:]}")
        else:
            self.stack.pop()

    def handle_data(self, data):
        if self.stack and self.stack[-1] == "script":
            self.script += data


def test_viewer_html_well_formed():
    html = VIEWER.read_text(encoding="utf-8")
    assert html.lstrip().lower().startswith("<!doctype html>")
    p = _Collector()
    p.feed(html)
    p.close()
    assert not p.errors, p.errors
    assert not p.stack, p.stack
    assert "manifest.json" in p.script
    assert "devicePixelRatio" in p.script and "naturalWidth" in p.script
    assert "pixelated" in html
    # True full screen (prefixed for WebKit) and a hinge-centred 1:1 layout, not a scaled fit.
    assert "requestFullscreen" in p.script and "webkitRequestFullscreen" in p.script
    assert "fullscreenchange" in p.script and "Add to Home Screen" in html
    assert "coversScreen" in p.script and "(v[0] - nw) / 2" in p.script
    # Installable as a full-screen landscape app (Chrome reads the web-app manifest).
    app = json.loads((VIEWER.parent / "app.webmanifest").read_text(encoding="utf-8"))
    assert app["display"] == "fullscreen" and app["orientation"] == "landscape"
    assert {i["sizes"] for i in app["icons"]} >= {"192x192", "512x512"}
    for i in app["icons"]:
        assert (VIEWER.parent / i["src"]).exists(), i["src"]
    assert 'rel="manifest" href="app.webmanifest"' in html
    assert "displayMismatch" in p.script and "navigator.standalone === true" in p.script
    # iPhone Duo: works alongside the iOS status strip (hinge on the fold, panel scale, nudge).
    assert "panelLayout" in p.script and "nudgeMm" in p.script and 'id="mask"' in html
    assert "DUO_PANEL_PT = [951, 669]" in p.script and "flipBand" in p.script
    # Self-contained: no external scripts/styles.
    assert not any(u.startswith(("http:", "https:", "//")) for u in p.urls), p.urls
    assert "<script src" not in html
    assert all(u.startswith(("data:", "#")) or "/" not in u for u in p.urls), p.urls
    assert (VIEWER.parent / "README.md").exists()
