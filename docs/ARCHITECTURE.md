# DuoVision architecture and module contracts

DuoVision encodes cross-eyed stereograms (autostereograms) for the iPhone Duo's
inner display **folded at an angle**, viewed from a **fixed, centred viewpoint**
(no head tracking). Every module below follows these contracts exactly; change a
contract only through the project manager.

## Conventions (all modules)

- Units: millimetres and degrees. Pixel coordinates are `(i, j)` = (column, row),
  row 0 at the top. Integer values are pixel centres.
- World frame: origin at the hinge centre, `+x` right, `+y` up the hinge,
  `+z` toward the viewer. Eyes at `z = view.distance_mm`.
- Image arrays are `numpy` `(H, W)` or `(H, W, 3)` `uint8`, H = `device.height_px`,
  W = `device.width_px` (2670 x 1878 full size, landscape, hinge vertical in the middle).
- Everything is vectorised `numpy`; no per-pixel Python loops on full images.
- Determinism: any randomness takes an explicit `seed` / `numpy.random.Generator`.
- Tests use `DEFAULT_DEVICE.scaled(0.25)` (same physical size, quarter resolution)
  so the suite stays fast. Full resolution must also work.
- Python >= 3.10, dependencies: numpy, scipy, Pillow, pytest.

## Shared core (manager-owned, do not edit)

`duovision/config.py`
- `DeviceProfile(width_px=2670, height_px=1878, ppi=430)`: `.mm_per_px`, `.half_width_mm`,
  `.height_mm`, `.scaled(f)`.
- `ViewProfile(distance_mm=300, ipd_mm=63, eye_height_mm=0, mode='cross', sep_bg_mm=15, sep_near_mm=22)`:
  `.eyes(offset=(dx,dy,dz)) -> (L, R)`, `.cyclopean_eye(offset)`, `.fused_distance(s)`,
  `.world_z_for_separation(s)`, `.z_background` (57.69 mm), `.z_near` (77.65 mm),
  `.conflict_dioptres(s)`, `.with_(**kw)`.
- `guide_band_rows(device)`: rows at the top reserved for guides (7% of H). The encoder
  never links into them; guides draw there.
- `DEFAULT_HINGE_DEG = 120`, `HINGE_SWEEP_DEG = (100, 105, ..., 140)`.

`duovision/geometry.py` — `FoldGeometry(device, hinge_deg)`
- `.s, .c` = sin/cos(hinge/2); `.a_left/.a_right` unit vectors hinge->outer edge;
  `.n_left/.n_right` unit normals toward the viewer.
- `.pixel_to_uv(i, j)`, `.uv_to_pixel(u, v)`, `.uv_to_world(u, v)`, `.pixel_to_world(i, j)`,
  `.pixel_grid_world() -> (H, W, 3)`.
- `.intersect(origins, dirs) -> (u, v, t, valid)`: rays vs the folded display, nearest hit, t > 0.
- `.project_from_eye(eye, points) -> (i, j, valid)`: where the eye->point line lands
  (works for points in front of or behind the display).
- `.panel_z(x)`, `.max_panel_z()`, `.x_extent_mm`, `.panel_corners(side)`.

## Hidden scenes — `duovision/scene.py`

The hidden 3D content is a **height field**: the front-most surface as seen along
`-z`, in absolute world z (the classic SIRDS depth-map assumption, here in 3D world space).

```python
class HeightScene:
    name: str
    z_min: float            # min/max of z_of over the display footprint (march bounds)
    z_max: float
    def z_of(self, x, y) -> np.ndarray   # world z of the hidden surface at world (x, y)
    def object_mask(self, x, y) -> np.ndarray  # True where an object (not background) is

SCENE_NAMES = ("plaque", "cube", "sphere", "text", "torus", "pyramid", "heart", "pawn")
def build_scene(name, geom, view, *, bg_vee=0.25, text="DuoVision") -> HeightScene
def first_hit(scene, eye, targets, *, step_mm=0.5, refine_iters=12) -> (points (...,3), hit (...,))
```
- Background: a shallow V, `z_bg(x) = view.z_background + bg_vee * |x| * c/s`
  (`bg_vee=0` flat frontal plane, `1` parallel to the panels).
- Objects: centred on the hinge axis, front surfaces between `z_bg + 2` and `view.z_near`
  (bas-relief compression of real 3D shapes). Keep objects at least 15 mm in from the
  outer edges and out of the guide band. `plaque` is a flat frontal rectangle
  (the hinge-continuity test card). `text` is the word "DuoVision".
- Cross mode: the surface must stay at least 3 mm in front of the display everywhere
  (`z_of(x, y) >= geom.panel_z(x) + 3`) for hinge 100..140 deg.
- Parallel mode: mirror the same shapes behind the display (z < panel), relief toward viewer.
- `first_hit`: ray from `eye` toward `targets` (and beyond), first point where the ray
  meets the height field (`ray.z <= z_of(ray.x, ray.y)`), refined by bisection to < 0.01 mm.

## Encoder — `duovision/encoder.py`, `resolve.py`, `patterns.py`

```python
@dataclass
class PartnerMap:
    partner_i: np.ndarray   # (H, W) float, fractional column of the partner pixel, NaN if free
    partner_j: np.ndarray   # (H, W) float
    linked: np.ndarray      # (H, W) bool
    hit_z: np.ndarray       # (H, W) world z of the scene point S seen by the LEFT eye through the pixel
    occluded: np.ndarray    # (H, W) bool, S not visible to the right eye
    reserved_rows: int

def compute_partner_map(geom, view, scene, *, reserved_rows=None) -> PartnerMap
```
For each pixel p: left-eye ray L->X(p) meets the scene at S; if S is visible from R,
the line R->S lands on the display at partner q; constraint `colour(p) == colour(q)`.
Cross mode: every link points left (`partner_i <= i - 1`); parallel: right. Pixels in
reserved rows are never linked; partners landing in reserved rows or off the display
make p free.

`resolve.py`
```python
def round_partners(pm, rng) -> (pi, pj)       # int arrays, -1 = free, randomized rounding
def resolve_roots(pi, pj) -> roots            # (H, W) int flat index of each chain's root (pointer jumping)
def resolve_components_reference(pi, pj) -> labels  # scipy connected_components, test reference
```

`patterns.py`
```python
STYLES = ("bw", "color", "texture")
def colorize(roots, device, style, *, seed=0, dot_px=5, texture="marble") -> uint8 (H, W, 3)
```
Colour must be a pure function of the root index so every chain member matches exactly.

`encoder.py` also provides the pipeline:
```python
@dataclass
class EncodeResult:
    image: np.ndarray          # (H, W, 3) uint8, guide band left black (guides draw it)
    partner_map: PartnerMap
    pi: np.ndarray; pj: np.ndarray; roots: np.ndarray
    stats: dict                # timings, linked fraction, occluded fraction, max separation px

def encode_stereogram(geom, view, scene, style="color", *, seed=0, dot_px=5) -> EncodeResult
```

## Guides, CLI, viewer — `duovision/guides.py`, `cli.py`, `viewer/`

```python
def guide_dot_pixels(geom, view) -> ((i_l, j_l), (i_r, j_r))   # fold-aware, fuse to 3 dots at the background depth
def draw_guide_band(image, geom, view, label: str) -> image      # fills reserved rows: dots + small label
def alignment_image(geom, view, label: str) -> uint8 (H, W, 3)   # frontal square+circle+cross projected from the cyclopean eye
def fold_guide_extent(geom, view) -> (half_w, half_h)            # mm in the plane z = z_background, frame of the fold-guide picture
def fold_guide_target(half_w, half_h, px_per_mm, hinge_deg) -> uint8 (h, w, 3)   # the undistorted picture (frame, checks, face, "FOLD 120°")
def fold_guide_image(geom, view, label: str = "") -> uint8 (H, W, 3)             # that picture projected from the cyclopean eye
```
`fold_guide_image` is the fold-angle guide: each display pixel shows the colour of the point of
the frontal plane `z = z_background` that its cyclopean ray passes through, so the picture looks
flat and undistorted only from the design viewpoint at the encoded hinge angle (flatter: tall in
the middle; folded further: pinched in the middle). The frame stays inside the central 78 % of the
display width so the iPhone Duo's status strip never hides it. A 5 deg fold error bends the frame
edges by about 0.75 deg at the hinge, the same for either eye.

CLI (`python -m duovision ...`): `generate`, `align`, `fold`, `batch`, `verify`.
Batch output: `{scene}_{hinge:03d}_{style}.png`, `fold_{hinge:03d}.png`, `align_{hinge:03d}.png`,
`manifest.json` (entry/group kinds `fold`, `align`, `scene`; groups ordered fold, align, scenes).
Viewer: static `viewer/index.html` that loads `manifest.json` and shows each image at
exact device pixels, with a diagnostics readout (viewport, devicePixelRatio) and a four-step
setup guide (unfold, fold on the fold-guide card, centre on the align card, fuse a scene).

## Verifier — `duovision/verify.py`, `baseline.py`

```python
def check_links(image, geom, view, scene, *, reserved_rows=None, sample=None, seed=0) -> dict
def render_eye_view(image, geom, eye, *, res=(W, H), fov_deg=None) -> dict   # what one eye sees
def reconstruct_depth(image, geom, view, *, head_offset=(0,0,0), ...) -> dict  # stereo-matching virtual viewer
def evaluate(image, geom_encoded, view, scene, *, geom_viewed=None, head_offset=(0,0,0)) -> dict
def tolerance_sweep(image, geom, view, scene, offsets, hinge_errors) -> list[dict]
def comfort_map(geom, view, scene) -> np.ndarray   # per-pixel vergence-accommodation conflict, dioptres
```
`baseline.py`: `flat_sirds(depth_map, ...)` classic flat-screen SIRDS (Thimbleby et al. 1994)
for comparison when shown on the fold.

`check_links` must be an independent re-implementation (trace from the right eye,
back to the left), not a call into the encoder.
