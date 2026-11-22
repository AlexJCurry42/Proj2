"""Animated scenes (duovision.motion) and looping videos (duovision.video)."""
import numpy as np
import pytest

from duovision import motion, video
from duovision import scene as S
from duovision.batch import RenderParams, batch_jobs, video_filename
from duovision.cli import main
from duovision.config import DEFAULT_DEVICE, DEFAULT_VIEW, guide_band_rows
from duovision.encoder import encode_stereogram
from duovision.geometry import FoldGeometry

DEV = DEFAULT_DEVICE.scaled(0.25)
VIEW = DEFAULT_VIEW
N = motion.MOTION_FPS * int(motion.MOTION_SECONDS)      # frames in one loop
needs_ffmpeg = pytest.mark.skipif(video.find_ffmpeg() is None, reason="ffmpeg not available")


@pytest.fixture(scope="module")
def g120():
    return FoldGeometry(DEV, 120)


def test_every_scene_has_a_motion():
    assert motion.MOTION_NAMES == S.SCENE_NAMES
    assert set(motion.MOTION_DESCRIPTION) == set(S.SCENE_NAMES)
    with pytest.raises(ValueError):
        motion.pose("nope", 0.0)


@pytest.mark.parametrize("name", S.SCENE_NAMES)
def test_loop_closes_and_stays_in_the_layout(name, g120):
    plan = motion.motion_plan(name, g120, VIEW)
    lay = plan["layout"]
    a = motion.scene_frame(name, g120, VIEW, 0.0)
    b = motion.scene_frame(name, g120, VIEW, 1.0)                 # phase wraps
    assert np.array_equal(a._grid, b._grid)
    for ph in np.linspace(0.0, 1.0, 13)[:-1]:
        sc = motion.scene_frame(name, g120, VIEW, ph)
        x_half, y0, y1 = sc.meta["object_extent"]
        # inside the layout area for the whole loop: at least one pattern repeat from the outer edges
        assert x_half <= lay["x_half"] + 0.2
        assert y0 >= lay["y_bottom"] - 0.2 and y1 <= lay["y_top"] + 0.2
        # size and depth mapping are fixed for the loop
        assert sc.meta["fit_scale"] == plan["sc"] and sc.meta["z_lo"] == plan["z_lo"] and sc.meta["z_top"] == plan["z_top"]
        X, Y = np.meshgrid(plan["xs"], plan["ys"])
        m = sc.object_mask(X, Y)
        z = sc.z_of(X, Y)
        assert m.any()
        assert z[m].max() <= plan["z_top"] + 1e-6
        assert (z[m] >= sc.background(X[m]) - 1e-9).all()
        assert sc.z_max <= VIEW.z_near + 1e-6                      # never nearer than the comfort limit


@pytest.mark.parametrize("name", S.SCENE_NAMES)
def test_motion_is_slow_and_continuous(name, g120):
    """Per frame the object moves well under a millimetre and its depth map changes smoothly;
    the step across the loop's wrap is no bigger than any other step."""
    plan = motion.motion_plan(name, g120, VIEW)
    X, Y = np.meshgrid(plan["xs"][::4], plan["ys"][::4])

    def centroid(ph):
        sc = motion.scene_frame(name, g120, VIEW, ph)
        m = sc.object_mask(X, Y)
        return np.array([X[m].mean(), Y[m].mean()])

    steps = []
    for k in (0, 30, 45, 77, 100, 135, N - 1):                    # includes the wrap N-1 -> N (= 0)
        steps.append(float(np.linalg.norm(centroid((k + 1) / N) - centroid(k / N))))
    assert max(steps) < 0.5                                       # mm per frame at 30 fps
    assert steps[-1] <= max(steps[:-1]) * 1.5 + 1e-3              # no jump where the loop restarts
    p0, p1 = motion.pose(name, 0.25), motion.pose(name, 0.25 + 1.0 / N)
    assert abs(p1["ry"] - p0["ry"]) <= 0.6 and abs(p1["spin"] - p0["spin"]) <= 0.6      # deg per frame
    assert abs(p1["scale"] - p0["scale"]) <= 0.005


def test_plaque_stays_planar(g120):
    for ph in (0.0, 0.2, 0.5, 0.8):
        sc = motion.scene_frame("plaque", g120, VIEW, ph)
        plan = motion.motion_plan("plaque", g120, VIEW)
        X, Y = np.meshgrid(plan["xs"], plan["ys"])
        m = sc.object_mask(X, Y)
        assert np.allclose(sc.z_of(X, Y)[m], VIEW.z_background + S.PLAQUE_OFFSET_MM)
    # it really moves across the hinge
    xs = [motion.pose("plaque", ph)["dx"] for ph in (0.25, 0.75)]
    assert xs[0] > 5 and xs[1] < -5


def test_rows_the_object_never_touches_do_not_change(g120):
    """Dots are seeded by chain-root position, so consecutive frames differ only in rows that
    cross the moving object (and the band is identical)."""
    nb = guide_band_rows(DEV)
    frames, rows = [], np.zeros(DEV.height_px, dtype=bool)
    for k in (40, 41):
        sc = motion.scene_frame("text", g120, VIEW, k / N)
        frames.append(encode_stereogram(g120, VIEW, sc, "color", seed=0, dot_px=5).image)
        _xh, y0, y1 = sc.meta["object_extent"]
        # display rows whose lines of sight can reach the object (generous margin)
        _i, j_top = g120.uv_to_pixel(0.0, y1 + 8.0)
        _i, j_bot = g120.uv_to_pixel(0.0, y0 - 8.0)
        rows[max(0, int(j_top)):int(j_bot) + 1] = True
    changed = (frames[0] != frames[1]).any(axis=2).any(axis=1)
    assert changed.any()                                          # the letters did move
    assert not changed[~rows].any()                               # everything away from them is still
    assert rows.mean() < 0.6
    assert np.array_equal(frames[0][:nb], frames[1][:nb])


def test_even_scale_and_job_list():
    assert video.even_scale(2 / 3) == pytest.approx(2 / 3)
    dev = DEFAULT_DEVICE.scaled(video.VIDEO_SCALE)
    assert (dev.width_px, dev.height_px) == (1780, 1252)
    for s in (0.25, 1 / 6, 0.5):
        d = DEFAULT_DEVICE.scaled(video.even_scale(s))
        assert d.width_px % 2 == 0 and d.height_px % 2 == 0
    stills = {j.file for j in batch_jobs()}
    assert not any(f.endswith(".mp4") for f in stills)            # videos are opt-in
    jobs = batch_jobs(videos=True)
    vids = [j for j in jobs if j.params.kind == "video"]
    assert len(vids) == len(S.SCENE_NAMES) * len(video.VIDEO_HINGES)
    assert video.VIDEO_HINGES == (110.0, 120.0, 130.0)            # the recommended pose and the ends of its band
    assert {j.file for j in vids} >= {video_filename("cube", 120), "sphere_110_motion.mp4", "pawn_130_motion.mp4"}
    assert all(j.params.style == "color" and j.params.scale == pytest.approx(video.VIDEO_SCALE) for j in vids)
    assert video.frame_count() == N


@needs_ffmpeg
def test_render_video_and_read_back(tmp_path):
    p = RenderParams(kind="video", scene="cube", hinge=120.0, style="color", scale=video.even_scale(0.25))
    out = tmp_path / "cube.mp4"
    st = video.render_video(p, out, seconds=0.3, fps=10, crf=video.VIDEO_CRF, n_jobs=1)
    dev = p.device()
    assert out.exists() and st["frames"] == 3 and (st["width"], st["height"]) == (dev.width_px, dev.height_px)
    dec = video.read_frame(out, 1)
    orig = video.render_frame(p, 1 / 3)
    assert dec.shape == orig.shape == (dev.height_px, dev.width_px, 3)
    # lossy, but the same picture: the dot pattern survives compression
    a, b = dec.astype(float).mean(axis=2), orig.astype(float).mean(axis=2)
    assert np.corrcoef(a.ravel(), b.ravel())[0, 1] > 0.8
    with pytest.raises(ValueError):                               # odd frame sizes are refused
        video.render_video(RenderParams(kind="video", scene="cube", scale=0.5), tmp_path / "odd.mp4", seconds=0.2, fps=10)


@needs_ffmpeg
def test_video_cli_and_batch(tmp_path, capsys):
    out = tmp_path / "v.mp4"
    assert main(["video", "--scene", "sphere", "--hinge", "120", "--scale", "0.25", "--seconds", "0.2", "--fps", "10",
                 "--jobs", "1", "--out", str(out)]) == 0
    assert out.exists() and "glides across the hinge" in capsys.readouterr().out
    assert main(["video", "--scene", "nope", "--out", str(tmp_path / "x.mp4")]) == 2


def test_batch_manifest_lists_videos(tmp_path, monkeypatch):
    """The manifest entry of a video: kind video, style motion, the still as its poster."""
    import json
    from duovision import batch as B

    def fake_render(p, out, **kw):                                # no need to encode 180 frames here
        out.write_bytes(b"mp4")
        return {"frames": N, "fps": 30, "seconds": 6.0, "bytes": 3, "width": p.device().width_px, "height": p.device().height_px}
    monkeypatch.setattr(video, "render_video", fake_render)
    jobs = batch_jobs(["cube"], [115, 120], base=RenderParams(scale=0.25), videos=True, video_hinges=[120])
    m = B.run_batch(tmp_path, jobs, n_jobs=1, log=lambda *_: None, copy_viewer=False)
    vids = [e for e in m["entries"] if e["kind"] == "video"]
    assert [e["file"] for e in vids] == ["cube_120_motion.mp4"]
    e = vids[0]
    assert e["style"] == "motion" and e["scene"] == "cube" and e["hinge"] == 120 and e["poster"] == "cube_120_color.png"
    assert e["stats"]["motion"] == "turns a quarter turn" and (tmp_path / e["poster"]).exists()
    cube = next(g for g in m["groups"] if g["id"] == "cube")
    assert cube["styles"][0] == "motion" and {"file": "cube_120_motion.mp4", "hinge": 120, "style": "motion"} in cube["items"]
    assert json.loads((tmp_path / "manifest.json").read_text())["device"]["width_px"] == 668     # the stills' size
