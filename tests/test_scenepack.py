"""Scene pack for the viewer's real-time renderer (duovision.scenepack)."""
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from duovision import motion, scenepack as P
from duovision import scene as S
from duovision.batch import RenderParams, VIEWER_FILES, batch_jobs, run_batch
from duovision.config import DEFAULT_DEVICE, DEFAULT_VIEW
from duovision.geometry import FoldGeometry

VIEW = DEFAULT_VIEW
ENGINE = Path(__file__).resolve().parent.parent / "viewer" / "engine.js"


@pytest.fixture(scope="module")
def pack_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("pack")
    pack = P.export_scene_pack(d, VIEW, hinges=[110, 120], scenes=["plaque", "cube", "sphere", "pyramid", "text"])
    return d, pack


def test_pack_contents(pack_dir):
    d, pack = pack_dir
    on_disk = json.loads((d / P.PACK_FILE).read_text())
    assert on_disk["format"] == P.PACK_FORMAT and on_disk["version"] == P.PACK_VERSION
    assert on_disk["hinges"] == [110.0, 120.0] and set(on_disk["scenes"]) == {"plaque", "cube", "sphere", "pyramid", "text"}
    assert on_disk["view"]["z_background"] == pytest.approx(VIEW.z_background)
    assert on_disk["device"]["width_mm"] == pytest.approx(DEFAULT_DEVICE.width_px * DEFAULT_DEVICE.mm_per_px)
    kinds = {n: s["kind"] for n, s in on_disk["scenes"].items()}
    assert kinds == {"plaque": "rect", "cube": "cube", "sphere": "template", "pyramid": "template", "text": "template"}
    for name, s in on_disk["scenes"].items():
        assert set(s["plans"]) == {"110", "120"}
        assert all(len(v) == P.POSE_SAMPLES for v in s["pose"].values())
        if s["kind"] == "template":
            t = s["template"]
            assert (d / t["file"]).stat().st_size == 3 * t["nx"] * t["ny"]          # uint16 depth + uint8 mask
            depth, mask = P.read_template(d / t["file"], t["nx"], t["ny"], t["depth_max"])
            assert depth.max() == pytest.approx(t["depth_max"], rel=1e-4) and set(np.unique(mask)) <= {0.0, 1.0}
    assert "rect" in on_disk["scenes"]["plaque"]["plans"]["120"] and "cube" in on_disk["scenes"]["cube"]


@pytest.mark.parametrize("name", ["plaque", "cube", "sphere", "pyramid"])
@pytest.mark.parametrize("hinge", [110, 120])
def test_pack_scene_matches_the_reference_scene(pack_dir, name, hinge):
    """The surface rebuilt from the pack (the shader's arithmetic) is the reference scene, away from
    the one-cell rim where a 0.2 mm template and the 0.1 mm bake place the silhouette differently."""
    d, pack = pack_dir
    g = FoldGeometry(DEFAULT_DEVICE, hinge)
    plan = motion.motion_plan(name, g, VIEW)
    X, Y = np.meshgrid(plan["xs"][::3], plan["ys"][::3])
    for ph in (0.0, 0.2, 0.61):
        a = motion.scene_frame(name, g, VIEW, ph).z_of(X, Y)
        b = P.pack_scene(pack, d, name, g, VIEW, ph).z_of(X, Y)
        diff = np.abs(a - b)
        assert np.percentile(diff, 98) < 0.08, (name, hinge, ph, float(np.percentile(diff, 98)))
        assert (diff > 0.3).mean() < 0.012


def test_pack_text_is_the_reference_text_resized(pack_dir):
    """The letters come from one nominal template, resized to the layout width: same relief, same
    footprint to within a stroke edge."""
    d, pack = pack_dir
    g = FoldGeometry(DEFAULT_DEVICE, 120)
    plan = motion.motion_plan("text", g, VIEW)
    X, Y = np.meshgrid(plan["xs"][::2], plan["ys"][::2])
    a = motion.scene_frame("text", g, VIEW, 0.3)
    b = P.pack_scene(pack, d, "text", g, VIEW, 0.3)
    ma, mb = a.object_mask(X, Y), b.object_mask(X, Y)
    assert abs(ma.mean() - mb.mean()) < 0.02 and (ma & mb).sum() / (ma | mb).sum() > 0.8
    assert b.z_of(X, Y).max() == pytest.approx(a.z_of(X, Y).max(), abs=0.05)
    assert pack["scenes"]["text"]["plans"]["120"]["tscale"] < 1.0


def test_pose_interpolation_wraps(pack_dir):
    _d, pack = pack_dir
    cube = pack["scenes"]["cube"]
    last = P._lerp_pose(cube, 1.0 - 0.5 / P.POSE_SAMPLES)       # between the last sample and the first
    assert 35.0 + 89.5 < last["ry"] < 35.0 + 90.0                # keeps turning, does not swing back
    for ph in (0.0, 0.123, 0.77):
        assert P._lerp_pose(pack["scenes"]["sphere"], ph)["dx"] == pytest.approx(motion.pose("sphere", ph)["dx"], abs=2e-3)


def test_batch_writes_the_pack_and_the_renderer(tmp_path):
    jobs = batch_jobs(["cube", "plaque"], [115, 120], base=RenderParams(scale=0.25), styles=["color"], align=False, fold=False)
    m = run_batch(tmp_path, jobs, n_jobs=1, log=lambda *_: None)
    assert m["live"] == {"pack": P.PACK_FILE, "scenes": ["cube", "plaque"], "hinges": [115.0, 120.0]}
    assert json.loads((tmp_path / "manifest.json").read_text())["live"]["pack"] == P.PACK_FILE
    pack = json.loads((tmp_path / P.PACK_FILE).read_text())
    assert set(pack["scenes"]) == {"cube", "plaque"} and pack["device"]["width_px"] == DEFAULT_DEVICE.width_px
    assert "engine.js" in VIEWER_FILES and (tmp_path / "engine.js").read_text() == ENGINE.read_text()
    m2 = run_batch(tmp_path, [], n_jobs=1, log=lambda *_: None)            # a later run keeps the pack entry
    assert m2["live"]["pack"] == P.PACK_FILE


@pytest.mark.skipif(shutil.which("node") is None, reason="node not available")
def test_engine_js_loads_and_makes_a_seamless_tile():
    js = (
        "const E = require(process.argv[1]); const t = E.makeTile('azure', 7); const n = E.TILE_SIZE;"
        "let lum = 0; for (let i = 0; i < n * n; i++) lum += 0.299 * t[4 * i] + 0.587 * t[4 * i + 1] + 0.114 * t[4 * i + 2];"
        "const d = E.guideDots({ W: 2670, H: 1878, mm: 25.4 / 430, s: Math.sin(Math.PI / 3), c: Math.cos(Math.PI / 3),"
        " halfW: 1335 * 25.4 / 430, halfH: 939 * 25.4 / 430 }, { ipd_mm: 63, eye_height_mm: 0, distance_mm: 300, z_background: 57.6923 }, 131);"
        "console.log(JSON.stringify({ keys: Object.keys(E), len: t.length, mean: lum / (n * n), dots: d }));"
    )
    out = json.loads(subprocess.run(["node", "-e", js, str(ENGINE)], capture_output=True, text=True, check=True).stdout)
    assert {"Engine", "makeTile", "guideDots"} <= set(out["keys"]) and out["len"] == 512 * 512 * 4
    assert 90 < out["mean"] < 165                                # about half light, half dark
    # the renderer's guide dots are where the reference engine puts them
    from duovision import guides
    (il, jl), (ir, jr) = guides.guide_dot_pixels(FoldGeometry(DEFAULT_DEVICE, 120), VIEW)
    assert np.allclose(out["dots"], [[il, jl], [ir, jr]], atol=0.02)
