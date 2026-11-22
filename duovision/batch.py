"""Rendering jobs, PNG writing (with metadata) and the batch generator.

Shared by ``cli.py``.  Heavy dependencies (scene, encoder, patterns) are
imported lazily inside functions so that ``guides``/``align`` work without
them and worker processes import only what they need.
"""
from __future__ import annotations

import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from . import __version__
from .config import DEFAULT_DEVICE, DEFAULT_HINGE_DEG, HINGE_SWEEP_DEG, DeviceProfile, ViewProfile

STYLE_CHOICES = ("bw", "color", "texture")
DEFAULT_SCENES = ("plaque", "cube", "sphere", "text", "torus", "pyramid", "heart", "pawn")
BATCH_SWEEP_STYLE = "color"
BATCH_EXTRA_STYLES = ("bw", "texture")
META_PREFIX = "duovision."


# ------------------------------------------------------------------ params
@dataclass
class RenderParams:
    """Everything needed to reproduce one image (stored in the PNG)."""

    kind: str = "stereogram"          # "stereogram" | "align"
    scene: str = "cube"
    hinge: float = DEFAULT_HINGE_DEG
    style: str = "color"
    texture: str = "marble"
    distance: float = 300.0
    ipd: float = 63.0
    eye_height: float = 0.0
    mode: str = "cross"
    sep_bg: float = 15.0
    sep_near: float = 22.0
    bg_vee: float = 0.25
    seed: int = 0
    dot_px: int = 5
    scale: float = 1.0
    text: str = "DuoVision"

    def device(self) -> DeviceProfile:
        return DEFAULT_DEVICE if self.scale == 1.0 else DEFAULT_DEVICE.scaled(self.scale)

    def view(self) -> ViewProfile:
        return ViewProfile(distance_mm=self.distance, ipd_mm=self.ipd, eye_height_mm=self.eye_height,
                           mode=self.mode, sep_bg_mm=self.sep_bg, sep_near_mm=self.sep_near)

    def geom(self):
        from .geometry import FoldGeometry
        return FoldGeometry(self.device(), self.hinge)

    def to_meta(self) -> dict:
        d = asdict(self)
        if self.kind == "align":
            for k in ("scene", "style", "texture", "bg_vee", "seed", "dot_px", "text"):
                d.pop(k, None)
        elif self.style != "texture":
            d.pop("texture", None)
        if self.kind == "stereogram" and self.scene != "text":
            d.pop("text", None)
        return d

    @classmethod
    def from_meta(cls, d: dict) -> "RenderParams":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


def stereogram_filename(scene: str, hinge: float, style: str) -> str:
    return f"{scene}_{int(round(hinge)):03d}_{style}.png"


def align_filename(hinge: float) -> str:
    return f"align_{int(round(hinge)):03d}.png"


# ---------------------------------------------------------------- rendering
def render_stereogram(p: RenderParams) -> tuple[np.ndarray, dict]:
    from .encoder import encode_stereogram
    from .guides import default_label, draw_guide_band
    from .scene import build_scene

    geom = p.geom()
    view = p.view()
    t0 = time.perf_counter()
    kw = {"bg_vee": p.bg_vee}
    if p.scene == "text":
        kw["text"] = p.text
    scene = build_scene(p.scene, geom, view, **kw)
    t1 = time.perf_counter()
    res = encode_stereogram(geom, view, scene, p.style, seed=p.seed, dot_px=p.dot_px, texture=p.texture)
    t2 = time.perf_counter()
    image = np.ascontiguousarray(res.image)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    image = draw_guide_band(image, geom, view, default_label(p.scene, p.hinge, view))
    t3 = time.perf_counter()
    stats = dict(res.stats)
    stats.update(time_scene_s=t1 - t0, time_guides_s=t3 - t2, time_render_s=t3 - t0)
    return image, stats


def render_align(p: RenderParams) -> tuple[np.ndarray, dict]:
    from .guides import alignment_figure_half_size, alignment_image, guide_dot_pixels

    geom = p.geom()
    view = p.view()
    t0 = time.perf_counter()
    image = alignment_image(geom, view, "DuoVision align")
    (il, jl), (ir, jr) = guide_dot_pixels(geom, view)
    stats = {
        "square_side_mm": 2 * alignment_figure_half_size(geom, view),
        "plane_z_mm": float(view.z_background),
        "guide_dots_px": [[il, jl], [ir, jr]],
        "time_render_s": time.perf_counter() - t0,
    }
    return image, stats


def render(p: RenderParams) -> tuple[np.ndarray, dict]:
    return render_align(p) if p.kind == "align" else render_stereogram(p)


# ----------------------------------------------------------------- PNG I/O
def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist()) if x.size <= 64 else f"<array {x.shape}>"
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        f = float(x)
        return f if math.isfinite(f) else None
    if isinstance(x, (int, str)) or x is None:
        return x
    return str(x)


def to_pil(image: np.ndarray, *, allow_palette: bool = True) -> Image.Image:
    """RGB uint8 -> exact palette ('P') image when <= 256 colours, else RGB."""
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    image = np.ascontiguousarray(image, dtype=np.uint8)
    if allow_palette:
        packed = (image[..., 0].astype(np.uint32) << 16) | (image[..., 1].astype(np.uint32) << 8) | image[..., 2]
        flat = packed.ravel()
        # Cheap early-out on a sample before the full unique.
        sample = np.unique(flat[:: max(1, flat.size // 200_000)])
        if sample.size <= 256:
            colours, inv = np.unique(flat, return_inverse=True)
            if colours.size <= 256:
                pal = np.stack([(colours >> 16) & 255, (colours >> 8) & 255, colours & 255], axis=1)
                im = Image.fromarray(inv.reshape(image.shape[:2]).astype(np.uint8), "P")
                im.putpalette(pal.astype(np.uint8).ravel().tolist())
                return im
    return Image.fromarray(image, "RGB")


def save_png(path, image: np.ndarray, params: RenderParams, stats: dict, *, palette: bool = True) -> dict:
    """Write a PNG with tEXt metadata (each parameter + JSON blobs). Atomic."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = params.to_meta()
    info = PngInfo()
    info.add_text("Software", f"DuoVision {__version__}")
    for k, v in meta.items():
        info.add_text(META_PREFIX + k, str(v))
    info.add_text(META_PREFIX + "params", json.dumps(_jsonable(meta), ensure_ascii=True))
    info.add_text(META_PREFIX + "stats", json.dumps(_jsonable(stats), ensure_ascii=True))
    im = to_pil(image, allow_palette=palette)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    im.save(tmp, format="PNG", pnginfo=info, optimize=True)
    os.replace(tmp, path)
    return {"png_mode": im.mode, "bytes": path.stat().st_size}


def read_png_meta(path) -> tuple[dict, dict, tuple[int, int]]:
    with Image.open(path) as im:
        text = dict(getattr(im, "text", {}) or {})
        size = im.size
    params = json.loads(text.get(META_PREFIX + "params", "{}"))
    stats = json.loads(text.get(META_PREFIX + "stats", "{}"))
    return params, stats, size


# -------------------------------------------------------------------- jobs
@dataclass
class Job:
    file: str
    params: RenderParams


def batch_jobs(scenes: Iterable[str] | None = None, hinges: Iterable[float] | None = None, *,
               styles: Iterable[str] | None = None, extra_hinge: float = DEFAULT_HINGE_DEG,
               base: RenderParams | None = None, align: bool = True) -> list[Job]:
    """The agreed first batch (optionally restricted).

    - every scene x every hinge in style 'color'
    - every scene at `extra_hinge` in styles 'bw' and 'texture' (marble)
    - align_XXX.png for every hinge
    """
    base = base or RenderParams()
    scenes = list(scenes) if scenes else list(_scene_names())
    hinges = [float(h) for h in (hinges or HINGE_SWEEP_DEG)]
    styles = set(styles) if styles else set(STYLE_CHOICES)
    jobs: list[Job] = []
    seen = set()

    def add(job):
        if job.file not in seen:
            seen.add(job.file)
            jobs.append(job)

    if align:
        for h in hinges:
            p = RenderParams(**{**asdict(base), "kind": "align", "hinge": h})
            add(Job(align_filename(h), p))
    if BATCH_SWEEP_STYLE in styles:
        for sc in scenes:
            for h in hinges:
                p = RenderParams(**{**asdict(base), "kind": "stereogram", "scene": sc, "hinge": h,
                                    "style": BATCH_SWEEP_STYLE})
                add(Job(stereogram_filename(sc, h, BATCH_SWEEP_STYLE), p))
    for st in BATCH_EXTRA_STYLES:
        if st not in styles:
            continue
        for sc in scenes:
            p = RenderParams(**{**asdict(base), "kind": "stereogram", "scene": sc, "hinge": float(extra_hinge),
                                "style": st, "texture": "marble"})
            add(Job(stereogram_filename(sc, extra_hinge, st), p))
    return jobs


def _scene_names():
    try:
        from .scene import SCENE_NAMES
        return SCENE_NAMES
    except ImportError:
        return DEFAULT_SCENES


def run_job(out_dir: str, job: Job, force: bool = False) -> dict:
    """Render one job (in a worker) unless the file exists. Returns a manifest entry."""
    path = Path(out_dir) / job.file
    t0 = time.perf_counter()
    if path.exists() and not force:
        params, stats, (w, h) = read_png_meta(path)
        with Image.open(path) as im:
            stats = {**stats, "png_mode": im.mode, "bytes": path.stat().st_size}
        return _entry(job.file, params or job.params.to_meta(), stats, w, h, skipped=True)
    image, stats = render(job.params)
    palette = job.params.kind == "align" or job.params.style in ("bw", "color")
    info = save_png(path, image, job.params, stats, palette=palette)
    stats = {**stats, **info, "time_job_s": time.perf_counter() - t0}
    return _entry(job.file, job.params.to_meta(), _jsonable(stats), image.shape[1], image.shape[0], skipped=False)


def _entry(file, params, stats, w, h, skipped):
    kind = params.get("kind", "stereogram")
    return {
        "file": file,
        "kind": kind,
        "scene": "align" if kind == "align" else params.get("scene"),
        "hinge": int(round(float(params.get("hinge", DEFAULT_HINGE_DEG)))),
        "style": "align" if kind == "align" else params.get("style"),
        "width": int(w),
        "height": int(h),
        "params": _jsonable(params),
        "stats": _jsonable(stats),
        "_skipped": skipped,
    }


def build_groups(entries: list[dict]) -> list[dict]:
    """Viewer menu: Align first, then one group per scene (SCENE_NAMES order)."""
    order = list(_scene_names())
    groups: dict[str, dict] = {}
    for e in entries:
        gid = e["scene"]
        g = groups.setdefault(gid, {"id": gid, "title": "Align" if gid == "align" else gid.capitalize(),
                                    "kind": "align" if gid == "align" else "scene",
                                    "hinges": set(), "styles": set(), "items": []})
        g["hinges"].add(e["hinge"])
        g["styles"].add(e["style"])
        g["items"].append({"file": e["file"], "hinge": e["hinge"], "style": e["style"]})
    style_order = {"color": 0, "bw": 1, "texture": 2, "align": 3}

    def gkey(gid):
        if gid == "align":
            return (-1, "")
        return (order.index(gid) if gid in order else len(order), gid)

    out = []
    for gid in sorted(groups, key=gkey):
        g = groups[gid]
        g["hinges"] = sorted(g["hinges"])
        g["styles"] = sorted(g["styles"], key=lambda s: (style_order.get(s, 9), s))
        g["items"].sort(key=lambda it: (it["hinge"], style_order.get(it["style"], 9)))
        out.append(g)
    return out


def write_manifest(out_dir, entries: list[dict], *, merge: bool = True) -> dict:
    out_dir = Path(out_dir)
    mpath = out_dir / "manifest.json"
    by_file = {}
    if merge and mpath.exists():
        try:
            old = json.loads(mpath.read_text())
            for e in old.get("entries", []):
                if (out_dir / e["file"]).exists():
                    by_file[e["file"]] = e
        except (ValueError, KeyError, TypeError):
            pass
    for e in entries:
        if e.get("_skipped") and e["file"] in by_file:
            continue  # keep the richer entry written when the file was rendered
        e = {k: v for k, v in e.items() if not k.startswith("_")}
        by_file[e["file"]] = e
    ents = sorted(by_file.values(), key=lambda e: (e["kind"] != "align", e["scene"], e["hinge"], e["style"]))
    first = ents[0] if ents else {}
    manifest = {
        "format": "duovision-manifest",
        "version": 1,
        "generator": f"DuoVision {__version__}",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "device": {"width_px": first.get("width"), "height_px": first.get("height"),
                   "full_width_px": DEFAULT_DEVICE.width_px, "full_height_px": DEFAULT_DEVICE.height_px,
                   "ppi": DEFAULT_DEVICE.ppi},
        "entries": ents,
        "groups": build_groups(ents),
    }
    tmp = mpath.with_name("manifest.json.tmp")
    tmp.write_text(json.dumps(manifest, indent=1))
    os.replace(tmp, mpath)
    return manifest


def run_batch(out_dir, jobs: list[Job], *, n_jobs: int | None = None, force: bool = False,
              log=print, copy_viewer: bool = True) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_jobs = max(1, int(n_jobs or os.cpu_count() or 1))
    t0 = time.perf_counter()
    entries = []
    todo = [j for j in jobs if force or not (out_dir / j.file).exists()]
    log(f"batch: {len(jobs)} images, {len(jobs) - len(todo)} already present, "
        f"{len(todo)} to render with {min(n_jobs, max(1, len(todo)))} worker(s) -> {out_dir}")
    done = 0

    def report(e):
        nonlocal done
        done += 1
        if not e["_skipped"]:
            s = e["stats"]
            log(f"  [{done}/{len(jobs)}] {e['file']}  {s.get('time_job_s', 0):.1f}s  "
                f"{s.get('png_mode', '?')} {s.get('bytes', 0) / 1e6:.2f} MB")

    if n_jobs == 1 or len(todo) <= 1:
        for j in jobs:
            e = run_job(str(out_dir), j, force)
            entries.append(e)
            report(e)
    else:
        # Align cards first (fast), heavy jobs spread across workers.
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            futs = {ex.submit(run_job, str(out_dir), j, force): j for j in jobs}
            for f in as_completed(futs):
                e = f.result()
                entries.append(e)
                report(e)
    manifest = write_manifest(out_dir, entries)
    if copy_viewer:
        copy_viewer_files(out_dir)
    dt = time.perf_counter() - t0
    log(f"batch done: {len(entries)} entries, {dt:.1f}s, manifest {out_dir / 'manifest.json'}")
    manifest["_elapsed_s"] = dt
    return manifest


VIEWER_FILES = ("index.html", "app.webmanifest", "icon-192.png", "icon-512.png")


def copy_viewer_files(out_dir) -> list[str]:
    """Copy the viewer (index.html, web-app manifest, icons) next to the images so the
    folder is self-contained and installable as a full-screen home-screen app."""
    import shutil
    src = Path(__file__).resolve().parent.parent / "viewer"
    copied = []
    for name in VIEWER_FILES:
        s = src / name
        if s.exists():
            shutil.copyfile(s, Path(out_dir) / name)
            copied.append(name)
    return copied
