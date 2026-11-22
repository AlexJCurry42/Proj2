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

Open items:
- `tests/test_verify.py::test_vertical_head_offset_is_compensated` is xfail: the
  stereo matcher's two-pass, fixation-nearest peak selection was left unfinished.
- Depth error for distance offsets is measured against absolute ground truth, so it
  includes the expected rescaling of depth with distance, not only distortion.
- Nothing has been viewed on a real iPhone Duo yet. Human fusion is unverified.
- ~0.1-0.2% of pixels at grazing silhouettes are freed instead of linked.
- Vertical partner offsets reach 25-35 px near the outer corners at full resolution.
