# DuoVision web viewer

`index.html` is a single self-contained page (no external libraries, works
offline). It loads `manifest.json` from its own folder and shows each PNG at
**exact device pixels**: CSS size = natural size / `devicePixelRatio`, centred on
the viewport (snapped to whole device pixels) on a black stage,
`image-rendering: pixelated`, no page scroll or zoom. The middle column of every
image is the hinge, so it lines up with the physical hinge only when the page
covers the **whole** display; see *Full screen* below.

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

## Full screen

The projection only works when the page fills the entire display at 1:1. If
browser bars take part of the screen, the picture is cropped and its hinge
column cannot sit on the physical hinge. (The old fit toggle scaled the picture
to the window, which breaks the projection; it is now only a labelled preview.)

- **Full screen** button (or `f`): uses the browser's real Fullscreen API
  (desktop browsers, Android, iPad) and stays at 1:1 pixels.
- **iPhone Safari** has no full-screen mode for web pages, so the button opens
  instructions instead: Share -> **Add to Home Screen**, then launch DuoVision
  from the home-screen icon. It then runs with no Safari bars at all.
- While the page does not cover the whole display, the bottom bar shows a
  yellow "Not full screen" warning with the page and screen sizes.

Check it in **Info** (Diagnostics): "viewport covers: YES" and "fits exactly:
YES" mean one image pixel equals one display pixel across the whole screen.
**Pixel test** fills the screen with 1-pixel stripes and a 2-pixel
checkerboard; any moire or blur means the browser is scaling.

## Use it

- Start with **Align** for the angle you want: fold until the grid lines run
  straight across the hinge, and move sideways until both halves look equally
  wide. Then open a scene at the same angle.
- Tap: show/hide the bar (angle pills, style pills, prev/next, menu, info).
- Swipe left/right: next/previous hinge angle in the scene; up/down: next scene.
- Desktop keys: arrows (angle / scene), `s` style, `m` menu, `i` diagnostics,
  `f` full screen on/off, `p` toggle 1:1 / fit preview (fit is scaled, so the
  3D does not work; for previewing only).
- Saving to Photos: long-press the picture -> "Save to Photos" (or "open the
  image alone" in the bar, then long-press). Photos shows it full screen too,
  but may zoom it to fill; Safari's 1:1 mode is the reference.
- The URL hash (`#cube/120/color`) remembers the current image.
