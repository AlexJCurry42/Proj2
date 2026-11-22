"""Black-box CLI tests via subprocess."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from duovision.config import DEFAULT_DEVICE, guide_band_rows

ROOT = Path(__file__).resolve().parents[2]
pytest.importorskip("duovision.cli")


def run(args, tmp_path):
    r = subprocess.run([sys.executable, "-m", "duovision", *args], cwd=ROOT, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, f"exit {r.returncode}\nstdout:\n{r.stdout}\nstderr:\n{r.stderr}"
    return r


def test_generate_png(tmp_path):
    out = tmp_path / "tmp.png"
    run(["generate", "--scene", "plaque", "--hinge", "120", "--style", "bw", "--scale", "0.15", "--out", str(out)], tmp_path)
    dev = DEFAULT_DEVICE.scaled(0.15)
    im = Image.open(out)
    assert im.size == (dev.width_px, dev.height_px)
    a = np.asarray(im.convert("RGB"))
    band = guide_band_rows(dev)
    body = a[band:]
    # bw style: body is black and white only
    assert set(np.unique(body)) <= {0, 255}
    # guides were drawn into the band
    assert (a[:band] != 0).any()


def test_align_png(tmp_path):
    out = tmp_path / "align.png"
    run(["align", "--hinge", "120", "--scale", "0.15", "--out", str(out)], tmp_path)
    dev = DEFAULT_DEVICE.scaled(0.15)
    im = Image.open(out)
    assert im.size == (dev.width_px, dev.height_px)
