# DuoVision web viewer

`index.html` is a single self-contained page (no external libraries, works
offline). It loads `manifest.json` from its own folder and shows each PNG at
**exact device pixels**: CSS size = natural size / `devicePixelRatio`, placed at
the top-left of a black full-screen stage, `image-rendering: pixelated`, no page
scroll or zoom.

## Make the folder

```sh
python -m duovision batch --out out/batch      # PNGs + manifest.json + a copy of index.html
```

The batch command copies `viewer/index.html` into the output folder, so
`out/batch/` is all you need to host (`--no-viewer` skips the copy; you can
also copy it by hand). A different manifest can be chosen with
`index.html?manifest=path/to/manifest.json` (images are resolved relative to it).

## Host it

Any static host works; the page only needs HTTP(S) because browsers block
`fetch()` of `manifest.json` from `file://`.

- Local network: `cd out/batch && python -m http.server 8000`, then open
  `http://<computer-ip>:8000/` in Safari on the Duo (same Wi-Fi).
- Public: upload the folder to GitHub Pages, Netlify Drop, Cloudflare Pages, S3,
  etc. Keep `index.html`, `manifest.json` and the PNGs side by side.

For the cleanest full screen, open the page in Safari, Share -> **Add to Home
Screen**, and launch it from the home-screen icon (no Safari toolbars). Then
open **Menu -> Diagnostics**: "fits exactly: YES" means one image pixel equals
one display pixel. **Pixel test** fills the screen with 1-pixel stripes and a
2-pixel checkerboard; any moire or blur means Safari is scaling.

## Use it

- Start with **Align** for the angle you want: fold until the grid lines run
  straight across the hinge, and move sideways until both halves look equally
  wide. Then open a scene at the same angle.
- Tap: show/hide the bar (angle pills, style pills, prev/next, menu, info).
- Swipe left/right: next/previous hinge angle in the scene; up/down: next scene.
- Desktop keys: arrows (angle / scene), `s` style, `m` menu, `i` diagnostics,
  `f` toggle 1:1 / fit-to-window (fit is scaled and only for previewing).
- Saving to Photos: long-press the picture -> "Save to Photos" (or "open the
  image alone" in the bar, then long-press). Photos shows it full screen too,
  but may zoom it to fill; Safari's 1:1 mode is the reference.
- The URL hash (`#cube/120/color`) remembers the current image.
