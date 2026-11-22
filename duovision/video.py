"""Looping stereogram videos: render the frames of an animated scene and encode them as H.264.

Each frame is a complete fold-aware stereogram (``motion.scene_frame`` ->
``encode_stereogram`` -> guide band), so every frame can be checked by the
virtual viewer exactly like a still. Frames are rendered in worker processes
and piped to ffmpeg as raw RGB; the MP4 is all-intra-friendly (a keyframe every
second), tagged BT.709, yuv420p, with the moov atom first so it starts playing
while it downloads.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .motion import MOTION_FPS, MOTION_SECONDS

VIDEO_SCALE = 2.0 / 3.0       # 1780 x 1252: even dimensions (4:2:0 needs them), 4/9 of the pixels to encode
VIDEO_CRF = 32                # x264 quality. Random dots barely compress: 20 gives ~60 MB per loop, 32 about 13 MB,
                              # and the virtual viewer reads the decoded frames equally well (see docs/STATUS.md)
VIDEO_HINGES = (110.0, 120.0, 130.0)   # the recommended pose: 120 deg, and the ends of the 110-130 deg band


def find_ffmpeg() -> str | None:
    """Path of an ffmpeg binary: the system one, else the one bundled with imageio-ffmpeg."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def even_scale(scale: float) -> float:
    """The resolution factor nearest `scale` whose frame has an even width and height
    (H.264 4:2:0 needs both). 2/3 and 1 are already even: 1780 x 1252 and 2670 x 1878."""
    from .config import DEFAULT_DEVICE

    for k in range(400):
        for s in (scale + k * 0.0005, scale - k * 0.0005):
            if s <= 0:
                continue
            dev = DEFAULT_DEVICE.scaled(s)
            if dev.width_px % 2 == 0 and dev.height_px % 2 == 0:
                return s
    raise ValueError(f"no even-sized frame near scale {scale}")


def frame_count(seconds: float = MOTION_SECONDS, fps: int = MOTION_FPS) -> int:
    return max(2, int(round(seconds * fps)))


def render_frame(p, phase: float) -> np.ndarray:
    """One frame (H, W, 3) uint8 of the animated scene described by RenderParams `p`."""
    from .encoder import encode_stereogram
    from .guides import default_label, draw_guide_band
    from .motion import scene_frame

    geom = p.geom()
    view = p.view()
    kw = {"bg_vee": p.bg_vee}
    if p.scene == "text":
        kw["text"] = p.text
    scene = scene_frame(p.scene, geom, view, phase, **kw)
    res = encode_stereogram(geom, view, scene, p.style, seed=p.seed, dot_px=p.dot_px, texture=p.texture)
    image = np.ascontiguousarray(res.image)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    return draw_guide_band(image, geom, view, default_label(p.scene, p.hinge, view))


def _frame_bytes(args) -> bytes:
    p, phase = args
    return render_frame(p, phase).tobytes()


def render_video(p, out, *, seconds: float = MOTION_SECONDS, fps: int = MOTION_FPS, crf: int = VIDEO_CRF,
                 n_jobs: int | None = None, log=None) -> dict:
    """Render the loop of `p.scene` and write it to `out` (.mp4). Returns stats.

    The frame size is the device size of `p` and must be even in both directions.
    """
    exe = find_ffmpeg()
    if not exe:
        raise RuntimeError("ffmpeg not found: install ffmpeg, or `pip install imageio-ffmpeg`")
    dev = p.device()
    w, h = dev.width_px, dev.height_px
    if w % 2 or h % 2:
        raise ValueError(f"video frames must have even dimensions, got {w} x {h}; use a scale such as 2/3 or 1")
    n = frame_count(seconds, fps)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + f".tmp{os.getpid()}.mp4")
    cmd = [exe, "-hide_banner", "-loglevel", "error", "-y",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
           "-an", "-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-profile:v", "high",
           "-vf", "scale=out_color_matrix=bt709:out_range=tv", "-pix_fmt", "yuv420p",
           "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
           "-g", str(fps), "-movflags", "+faststart", str(tmp)]
    t0 = time.perf_counter()
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    jobs = [(p, k / n) for k in range(n)]
    n_jobs = max(1, int(n_jobs or os.cpu_count() or 1))
    try:
        if n_jobs == 1:
            frames = map(_frame_bytes, jobs)
            for k, buf in enumerate(frames):
                proc.stdin.write(buf)
        else:
            with ProcessPoolExecutor(max_workers=n_jobs) as ex:
                for k, buf in enumerate(ex.map(_frame_bytes, jobs, chunksize=1)):   # in order
                    proc.stdin.write(buf)
                    if log and (k + 1) % fps == 0:
                        log(f"    {out.name}: frame {k + 1}/{n}")
        proc.stdin.close()
        err = proc.stderr.read().decode("utf-8", "replace")
        if proc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed: {err.strip()[-400:]}")
    except BaseException:
        proc.kill()
        if tmp.exists():
            tmp.unlink()
        raise
    os.replace(tmp, out)
    size = out.stat().st_size
    return {"frames": n, "fps": fps, "seconds": n / fps, "bytes": size, "width": w, "height": h, "crf": crf,
            "mbit_per_s": size * 8 / (n / fps) / 1e6, "time_render_s": time.perf_counter() - t0}


def read_frame(path, index: int = 0) -> np.ndarray:
    """Decode one frame of a video as (H, W, 3) uint8 (for verifying what was actually encoded)."""
    exe = find_ffmpeg()
    if not exe:
        raise RuntimeError("ffmpeg not found")
    probe = subprocess.run([exe, "-hide_banner", "-i", str(path)], capture_output=True, text=True).stderr
    import re
    m = re.search(r", (\d{2,5})x(\d{2,5})[ ,]", probe)
    if not m:
        raise RuntimeError(f"cannot read the size of {path}")
    w, h = int(m.group(1)), int(m.group(2))
    raw = subprocess.run([exe, "-hide_banner", "-loglevel", "error", "-i", str(path), "-vf", f"select=eq(n\\,{int(index)})",
                          "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True).stdout
    if len(raw) != w * h * 3:
        raise RuntimeError(f"could not decode frame {index} of {path}")
    return np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3).copy()
