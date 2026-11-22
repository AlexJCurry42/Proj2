# Status (milestone 1)

Working and tested: fold geometry, 8 hidden scenes, fold-aware encoder (partner map,
pointer jumping, bw/colour/texture patterns), guides and alignment card, CLI, batch,
Safari viewer, virtual-viewer verifier, flat-SIRDS baseline, 248 black-box tests.

Virtual-viewer results, plaque at 120 deg, half resolution, colour dots:

| Viewing condition | Median depth error (mm) | Plaque kink at hinge (deg) |
| --- | --- | --- |
| Design position | 0.13 | 0.15 |
| Head 10 mm sideways | 0.86 | 0.64 |
| Head 20 mm sideways | 1.67 | 2.28 |
| Head 30 mm sideways | 2.43 | 4.17 |
| Head 10 mm up | 0.12 | 0.05 |
| Head 20 mm up | 0.09 | 0.03 |
| 50 mm nearer / farther | 8.5 / 8.0 | 12.2 / 12.1 |
| Fold 2 deg off | 0.34 | 2.6-2.7 |
| Fold 5 deg off | 0.85-0.89 | 6.7-6.9 |
| Flat SIRDS shown on the fold (quarter res) | 7.1 | 64 |

Fold-angle guide (`fold_XXX.png`, viewer setup guide): an anamorphic picture that looks flat
only at the encoded fold. Cue strength: 0.15 deg of bend at the hinge per degree of fold error
(0.75 deg at 5 deg off), identical for either eye, so expect people to land within about 5 deg.
Not yet tried by a person on the device.

Open items:
- Resolved: `tests/test_verify.py::test_vertical_head_offset_is_compensated` now passes
  (xfail removed). The two-pass, fixation-nearest peak selection was already complete;
  the failure was in vertical compensation. The dv estimate locked to whole pixels
  (median error 0.4 px vs a design field of about ±0.7 px that changes sign at the
  hinge), and one-sided bilinear resampling blurred only the right view. Fixes in
  `reconstruct_depth`: the dv field is estimated on Gaussian-prefiltered views
  (`field_smooth_px=0.7`), `_vertical_field` patch/step scale with retinal width
  (64/16 px at 400 px wide), and when the right view is resampled both views get a
  vertical-only blur (`match_vsmooth_px=0.5`). At the design position nothing changes.
  Quarter-res test double at dy=10 mm: linkable valid 0.43 -> 0.64, est/design
  vertical p95 4.3/4.2 arcmin. Half-res real encodes: depth error and plaque kink
  within 0.01 mm / 0.02 deg of the table above; valid fraction at dy=20 mm 0.83 -> 0.88.
  An isotropic prefilter for matching was rejected: it pushed the flat-SIRDS
  baseline's "valid" fraction from 0.27 to 0.51 at 9 % inliers, so it made the
  verifier too permissive.
- Depth error for distance offsets is measured against absolute ground truth, so it
  includes the expected rescaling of depth with distance, not only distortion.
- Nothing has been viewed on a real iPhone Duo yet. Human fusion is unverified.
- ~0.1-0.2% of pixels at grazing silhouettes are freed instead of linked.
- Vertical partner offsets reach 25-35 px near the outer corners at full resolution.
