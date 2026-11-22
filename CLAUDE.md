# DuoVision (proj2)

Fold-aware cross-eyed stereograms for the iPhone Duo's folded inner display, viewed
from a fixed centred viewpoint (no head tracking). Python reference engine plus a
virtual-viewer verifier; a Swift/Metal port comes later.

- Contracts: `docs/ARCHITECTURE.md`. Follow them exactly.
- Units mm/degrees; world frame origin at the hinge centre, +x right, +y up, +z toward viewer.
- `duovision/config.py` and `duovision/geometry.py` are the shared core: do not change
  their behaviour without updating the contracts and tests.
- Vectorised numpy only on full images; explicit seeds for all randomness.
- Run tests: `python -m pytest -q` (tests use `DEFAULT_DEVICE.scaled(0.25)` for speed).
- Generated images go to `out/` (git-ignored).
