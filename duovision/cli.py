"""Command line: ``python -m duovision {generate,align,fold,video,batch,verify} ...``"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from .batch import STYLE_CHOICES, RenderParams
from .config import DEFAULT_DEVICE, DEFAULT_HINGE_DEG, DeviceProfile


def _add_view_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("viewing (design viewpoint)")
    g.add_argument("--distance", type=float, default=300.0, help="eye plane to hinge, mm (default 300)")
    g.add_argument("--ipd", type=float, default=63.0, help="interpupillary distance, mm (default 63)")
    g.add_argument("--eye-height", type=float, default=0.0, help="eye level above the hinge midpoint, mm")
    g.add_argument("--mode", choices=("cross", "parallel"), default="cross")
    g.add_argument("--sep-bg", type=float, default=15.0, help="background separation at the hinge, mm")
    g.add_argument("--sep-near", type=float, default=22.0, help="nearest-point separation at the hinge, mm")
    p.add_argument("--scale", type=float, default=1.0,
                   help="resolution factor (1.0 = native 2670x1878; smaller for previews)")


def _add_encode_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("encoding")
    g.add_argument("--style", choices=STYLE_CHOICES, default="color")
    g.add_argument("--texture", default="marble", help="procedural texture for --style texture")
    g.add_argument("--bg-vee", type=float, default=0.25, help="background V depth (0 flat .. 1 parallel to panels)")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--dot-px", type=int, default=5, help="random-dot size in pixels")
    g.add_argument("--text", default="DuoVision", help="word for the 'text' scene")


def _params(a, kind: str, **over) -> RenderParams:
    d = dict(kind=kind, hinge=float(a.hinge), distance=a.distance, ipd=a.ipd, eye_height=a.eye_height,
             mode=a.mode, sep_bg=a.sep_bg, sep_near=a.sep_near, scale=a.scale)
    for k in ("scene", "style", "texture", "bg_vee", "seed", "dot_px", "text"):
        if hasattr(a, k) and getattr(a, k) is not None:
            d[k] = getattr(a, k)
    d.update(over)
    return RenderParams(**d)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m duovision",
        description="Fold-aware cross-eyed stereograms for the iPhone Duo inner display.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="encode one stereogram (with guide band) to PNG")
    g.add_argument("--scene", required=True)
    g.add_argument("--hinge", type=float, default=DEFAULT_HINGE_DEG, help="hinge angle, degrees (180 = flat)")
    _add_encode_args(g)
    _add_view_args(g)
    g.add_argument("--out", required=True, help="output PNG path")
    g.add_argument("--rgb", action="store_true", help="always write RGB PNG (no palette)")

    a = sub.add_parser("align", help="render the fold-angle / centring alignment card")
    a.add_argument("--hinge", type=float, default=DEFAULT_HINGE_DEG)
    _add_view_args(a)
    a.add_argument("--out", required=True)

    f = sub.add_parser("fold", help="render the fold-angle guide (a picture that only looks right when folded)")
    f.add_argument("--hinge", type=float, default=DEFAULT_HINGE_DEG)
    _add_view_args(f)
    f.add_argument("--out", required=True)

    m = sub.add_parser("video", help="render a looping animated stereogram of a scene to MP4 (needs ffmpeg)")
    m.add_argument("--scene", required=True)
    m.add_argument("--hinge", type=float, default=DEFAULT_HINGE_DEG, help="hinge angle, degrees (180 = flat)")
    _add_encode_args(m)
    _add_view_args(m)
    m.add_argument("--seconds", type=float, default=None, help="loop length (default 6)")
    m.add_argument("--fps", type=int, default=None, help="frames per second (default 30)")
    m.add_argument("--crf", type=int, default=None, help="x264 quality, lower is better and larger (default 32)")
    m.add_argument("--jobs", type=int, default=None, help="worker processes (default: CPU count)")
    m.add_argument("--out", required=True, help="output .mp4 path")

    b = sub.add_parser("batch", help="render the agreed batch + manifest.json for the viewer")
    b.add_argument("--out", required=True, help="output directory")
    b.add_argument("--jobs", type=int, default=None, help="worker processes (default: CPU count)")
    b.add_argument("--force", action="store_true", help="re-render files that already exist")
    b.add_argument("--no-viewer", action="store_true", help="do not copy viewer/index.html into --out")
    _add_view_args(b)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--dot-px", type=int, default=5)
    b.add_argument("--bg-vee", type=float, default=0.25)
    # Hidden subset flags (testing / partial runs).
    b.add_argument("--scenes", nargs="+", default=None, help=argparse.SUPPRESS)
    b.add_argument("--hinges", nargs="+", type=float, default=None, help=argparse.SUPPRESS)
    b.add_argument("--styles", nargs="+", default=None, help=argparse.SUPPRESS)
    b.add_argument("--extra-hinge", type=float, default=DEFAULT_HINGE_DEG, help=argparse.SUPPRESS)
    b.add_argument("--videos", action="store_true",
                   help="also render a looping video of every scene at 110, 120 and 130 deg (needs ffmpeg; slow)")
    b.add_argument("--video-hinges", nargs="+", type=float, default=None, help=argparse.SUPPRESS)
    b.add_argument("--video-scale", type=float, default=None, help=argparse.SUPPRESS)
    b.add_argument("--dry-run", action="store_true", help="list the jobs and exit")

    v = sub.add_parser("verify", help="evaluate an image with the virtual viewer (duovision.verify)")
    v.add_argument("--image", required=True)
    v.add_argument("--scene", default=None, help="default: from the PNG metadata")
    v.add_argument("--hinge", type=float, default=None, help="encoded hinge; default: from the PNG metadata")
    v.add_argument("--view-hinge", type=float, default=None, help="hinge the phone is actually folded to")
    v.add_argument("--head-offset", type=float, nargs=3, default=(0.0, 0.0, 0.0), metavar=("DX", "DY", "DZ"))
    v.add_argument("--res", type=int, nargs=2, default=None, metavar=("W", "H"),
                   help="virtual-eye render resolution (default: verify's own default)")
    v.add_argument("--json", action="store_true", help="print metrics as JSON")
    for k, d in (("--distance", None), ("--ipd", None), ("--eye-height", None), ("--sep-bg", None),
                 ("--sep-near", None), ("--bg-vee", None)):
        v.add_argument(k, type=float, default=d)
    v.add_argument("--mode", choices=("cross", "parallel"), default=None)
    v.add_argument("--text", default=None)
    return ap


# ------------------------------------------------------------------ commands
def _print_stats(stats: dict, keys=None) -> None:
    for k, val in stats.items():
        if keys and k not in keys:
            continue
        if isinstance(val, float):
            print(f"  {k:32s} {val:.4g}")
        elif isinstance(val, (int, str, bool)) or val is None:
            print(f"  {k:32s} {val}")


def cmd_generate(a) -> int:
    from .batch import render_stereogram, save_png

    try:
        from .scene import SCENE_NAMES
        if a.scene not in SCENE_NAMES:
            print(f"unknown scene {a.scene!r}; choose from {', '.join(SCENE_NAMES)}", file=sys.stderr)
            return 2
    except ImportError as e:
        print(f"scene module unavailable: {e}", file=sys.stderr)
        return 1
    p = _params(a, "stereogram")
    t0 = time.perf_counter()
    image, stats = render_stereogram(p)
    info = save_png(a.out, image, p, stats, palette=(not a.rgb and p.style in ("bw", "color")))
    print(f"wrote {a.out}  {image.shape[1]}x{image.shape[0]}  {info['png_mode']}  "
          f"{info['bytes'] / 1e6:.2f} MB  ({time.perf_counter() - t0:.1f}s)")
    _print_stats(stats)
    return 0


def cmd_align(a) -> int:
    from .batch import render_align, save_png

    p = _params(a, "align")
    image, stats = render_align(p)
    info = save_png(a.out, image, p, stats, palette=True)
    print(f"wrote {a.out}  {image.shape[1]}x{image.shape[0]}  {info['png_mode']}  "
          f"{info['bytes'] / 1e6:.2f} MB  square {stats['square_side_mm']:g} mm at z={stats['plane_z_mm']:.1f} mm")
    return 0


def cmd_fold(a) -> int:
    from .batch import render_fold, save_png

    p = _params(a, "fold")
    image, stats = render_fold(p)
    info = save_png(a.out, image, p, stats, palette=True)
    fw, fh = stats["frame_mm"]
    print(f"wrote {a.out}  {image.shape[1]}x{image.shape[0]}  {info['png_mode']}  "
          f"{info['bytes'] / 1e6:.2f} MB  frame {fw:.0f} x {fh:.0f} mm at z={stats['plane_z_mm']:.1f} mm")
    return 0


def cmd_video(a) -> int:
    from .motion import MOTION_DESCRIPTION, MOTION_FPS, MOTION_NAMES, MOTION_SECONDS
    from .video import VIDEO_CRF, VIDEO_SCALE, even_scale, find_ffmpeg, render_video

    if a.scene not in MOTION_NAMES:
        print(f"unknown scene {a.scene!r}; choose from {', '.join(MOTION_NAMES)}", file=sys.stderr)
        return 2
    if not find_ffmpeg():
        print("ffmpeg not found: install ffmpeg, or `pip install imageio-ffmpeg`", file=sys.stderr)
        return 1
    scale = even_scale(a.scale if a.scale != 1.0 else VIDEO_SCALE)   # default 2/3 of the device resolution
    p = _params(a, "video", scale=scale)
    try:
        st = render_video(p, a.out, seconds=a.seconds or MOTION_SECONDS, fps=a.fps or MOTION_FPS,
                          crf=a.crf if a.crf is not None else VIDEO_CRF, n_jobs=a.jobs)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 2
    print(f"wrote {a.out}  {st['width']}x{st['height']}  {st['frames']} frames at {st['fps']} fps  "
          f"{st['bytes'] / 1e6:.1f} MB ({st['mbit_per_s']:.1f} Mbit/s)  {a.scene} {MOTION_DESCRIPTION[a.scene]}  "
          f"({st['time_render_s']:.0f}s)")
    return 0


def cmd_batch(a) -> int:
    from .batch import batch_jobs, run_batch

    base = RenderParams(distance=a.distance, ipd=a.ipd, eye_height=a.eye_height, mode=a.mode,
                        sep_bg=a.sep_bg, sep_near=a.sep_near, bg_vee=a.bg_vee, seed=a.seed,
                        dot_px=a.dot_px, scale=a.scale)
    jobs = batch_jobs(a.scenes, a.hinges, styles=a.styles, extra_hinge=a.extra_hinge, base=base,
                      align=(a.styles is None or "align" in a.styles),
                      fold=(a.styles is None or "fold" in a.styles),
                      videos=a.videos, video_hinges=a.video_hinges, video_scale=a.video_scale)
    if a.videos and not a.dry_run:
        from .video import find_ffmpeg
        if not find_ffmpeg():
            print("ffmpeg not found: install ffmpeg, or `pip install imageio-ffmpeg`", file=sys.stderr)
            return 1
    if a.dry_run:
        for j in jobs:
            print(j.file)
        print(f"{len(jobs)} jobs")
        return 0
    run_batch(a.out, jobs, n_jobs=a.jobs, force=a.force, copy_viewer=not a.no_viewer)
    return 0


def _device_for(width: int, height: int, scale: float | None) -> DeviceProfile:
    if scale:
        d = DEFAULT_DEVICE if scale == 1.0 else DEFAULT_DEVICE.scaled(scale)
        if (d.width_px, d.height_px) == (width, height):
            return d
    if (width, height) == (DEFAULT_DEVICE.width_px, DEFAULT_DEVICE.height_px):
        return DEFAULT_DEVICE
    return DeviceProfile(width_px=width, height_px=height, ppi=DEFAULT_DEVICE.ppi * width / DEFAULT_DEVICE.width_px)


def cmd_verify(a) -> int:
    import numpy as np
    from PIL import Image

    from .batch import read_png_meta
    from .geometry import FoldGeometry

    try:
        from . import verify as V
        from .scene import build_scene
    except ImportError as e:
        print(f"verify is not available yet ({e}); nothing to do.", file=sys.stderr)
        return 1
    if not hasattr(V, "evaluate"):
        print("duovision.verify has no evaluate(); nothing to do.", file=sys.stderr)
        return 1
    meta, _stats, _size = read_png_meta(a.image)
    scene_name = a.scene or meta.get("scene")
    hinge = a.hinge if a.hinge is not None else meta.get("hinge")
    if not scene_name or hinge is None:
        print("need --scene and --hinge (no DuoVision metadata in the PNG)", file=sys.stderr)
        return 2

    def pick(name, key, default):
        val = getattr(a, name)
        return val if val is not None else meta.get(key, default)

    p = RenderParams(scene=scene_name, hinge=float(hinge),
                     distance=pick("distance", "distance", 300.0), ipd=pick("ipd", "ipd", 63.0),
                     eye_height=pick("eye_height", "eye_height", 0.0), mode=pick("mode", "mode", "cross"),
                     sep_bg=pick("sep_bg", "sep_bg", 15.0), sep_near=pick("sep_near", "sep_near", 22.0),
                     bg_vee=pick("bg_vee", "bg_vee", 0.25), text=pick("text", "text", "DuoVision"),
                     scale=float(meta.get("scale", 1.0)))
    with Image.open(a.image) as im:
        image = np.asarray(im.convert("RGB"))
    dev = _device_for(image.shape[1], image.shape[0], p.scale)
    geom = FoldGeometry(dev, p.hinge)
    view = p.view()
    kw = {"bg_vee": p.bg_vee}
    if p.scene == "text":
        kw["text"] = p.text
    scene = build_scene(p.scene, geom, view, **kw)
    geom_viewed = FoldGeometry(dev, a.view_hinge) if a.view_hinge is not None else None
    t0 = time.perf_counter()
    kw = {"res": tuple(a.res)} if a.res else {}
    metrics = V.evaluate(image, geom, view, scene, geom_viewed=geom_viewed, head_offset=tuple(a.head_offset), **kw)
    dt = time.perf_counter() - t0
    if a.json:
        from .batch import _jsonable
        print(json.dumps(_jsonable(_scalars(metrics)), indent=1))
    else:
        print(f"verify {a.image}: scene={p.scene} hinge={p.hinge:g} viewed at "
              f"{(a.view_hinge if a.view_hinge is not None else p.hinge):g} head_offset={tuple(a.head_offset)} "
              f"({dt:.1f}s)")
        for k, val in _scalars(metrics).items():
            print(f"  {k:40s} {val:.4g}" if isinstance(val, float) else f"  {k:40s} {val}")
    return 0


def _scalars(d, prefix=""):
    """Flatten nested dicts, keep scalars and short lists (skip arrays/images)."""
    import numpy as np
    out = {}
    for k, v in (d or {}).items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_scalars(v, key + "."))
        elif isinstance(v, np.ndarray):
            if v.ndim == 0:
                out[key] = v.item()
            elif v.size <= 8:
                out[key] = v.tolist()
        elif isinstance(v, (np.floating, np.integer, np.bool_)):
            out[key] = v.item()
        elif isinstance(v, (int, float, str, bool)) or v is None:
            out[key] = v
        elif isinstance(v, (list, tuple)) and len(v) <= 8 and all(isinstance(x, (int, float, str)) for x in v):
            out[key] = list(v)
    return out


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "scale", 1.0) <= 0:
        print("--scale must be positive", file=sys.stderr)
        return 2
    return {"generate": cmd_generate, "align": cmd_align, "fold": cmd_fold, "video": cmd_video, "batch": cmd_batch,
            "verify": cmd_verify}[args.cmd](args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
