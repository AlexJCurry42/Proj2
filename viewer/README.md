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
python -m duovision batch --out out/batch      # PNGs + manifest.json + the viewer files
```

The batch command copies the viewer (`index.html`, `app.webmanifest`, icons) into the output folder, so
`out/batch/` is all you need to host (`--no-viewer` skips the copy; you can
also copy it by hand). A different manifest can be chosen with
`index.html?manifest=path/to/manifest.json` (images are resolved relative to it).

## Host it

**Easiest: the hosted site.** `.github/workflows/pages.yml` renders the full
batch and publishes it with this viewer on GitHub Pages on every push to `main`
(or on demand: Actions -> Pages -> Run workflow). Open
`https://alexjcurry42.github.io/Proj2/` in Chrome (or Safari) on the Duo.
One-time setup: repo Settings -> Pages -> Source: **GitHub Actions**.

Any other static host works too; the page only needs HTTP(S) because browsers
block `fetch()` of `manifest.json` from `file://`.

- Local network: `cd out/batch && python -m http.server 8000`, then open
  `http://<computer-ip>:8000/` in Safari on the Duo (same Wi-Fi).
- Public: upload the folder to GitHub Pages, Netlify Drop, Cloudflare Pages, S3,
  etc. Keep `index.html`, `manifest.json` and the PNGs side by side.

## Full screen

The projection only works when the page fills the entire display at 1:1. If
browser bars take part of the screen, the picture is cropped and its hinge
column cannot sit on the physical hinge. (The old fit toggle scaled the picture
to the window, which breaks the projection; it is now only a labelled preview.)

- **Install it as an app (best).** `app.webmanifest` asks for a full-screen,
  landscape launch with no browser or system bars.
  - **Chrome:** open the site, tap the menu -> **Install app** (or **Add to Home
    screen** -> **Install**), and launch from the new icon. An icon added before
    the manifest existed is a plain shortcut that opens inside the browser:
    delete it and install again.
  - **iPhone:** Share -> **Add to Home Screen**, then launch from the icon.
- **Tap to go full screen:** while the page does not cover the display, the
  first tap on the picture requests real full screen (Fullscreen API, system
  bars hidden), as does the **Full screen** button or `f`. It always stays at
  1:1 pixels. After you leave full screen on purpose, taps show the bar again.
- Where the browser has no Fullscreen API (iPhone), the button explains how to
  install the app instead.
- While the page does not cover the whole display, the bottom bar shows a
  yellow "Not full screen" warning with the page and screen sizes.

### iPhone Duo inner display (status strip)

On the unfolded inner display iOS keeps a vertical status strip (clock, Wi-Fi)
on one side, and web apps get a window that stops at it: 867 x 669 of the
951 x 669 pt panel in landscape. iOS also renders pages at 3x (2853 x 2007)
and scales them to the 2670 x 1878 panel, so 1:1 physical pixels are not
possible for any web page there. The viewer works alongside this:

- iOS does not always give the web app the same window: 867 x 669 pt, or
  867 x 587 pt with an extra 82 pt bar reserved at the bottom (or top). So the
  viewer does not infer the panel from the window. In the home-screen app, in
  landscape, it scales the picture to the fixed 951 x 669 pt panel (true size
  in mm) and places it from where the window sits on the panel, so the hinge
  column is on the physical fold (CSS x 475.5) whatever window iOS gives.
- **Settings -> Hinge alignment**: Adjust the hinge position (0.5 and 0.1 mm
  steps, reset), choose which side the strip is on, whether the extra bar (when
  present) is at the bottom or top, and mirror the strip (hide the
  same width on the other side so both eyes get a symmetric frame). Settings
  are remembered per orientation.
- Check on the **Align** card: the gold vertical line should sit exactly on
  the fold. Diagnostics shows the layout, scale and hinge position.
- The browser resamples the picture by about 7%, so it is slightly soft. A
  native app is the only route to sharp 1:1 pixels across the full width.

Check it in **Settings -> Diagnostics**: "viewport covers: YES" and "fits exactly:
YES" mean one image pixel equals one display pixel across the whole screen. On
the iPhone Duo inner display "fits exactly" reads "n/a" (iOS scales web pages
there) and the "layout" and "hinge" lines describe the panel layout instead.
**Pixel test** fills the screen with 1-pixel stripes and a 2-pixel
checkerboard; any moire or blur means the browser is scaling.

## Use it

- **Viewing.** Tap the picture to show or hide the bars. The top bar has the
  Library (left) and Settings (right). The bottom dock steps through pictures
  and angles and switches style (Colour, Mono, Texture). Swipe left or right
  for the angle, up or down for the next picture.
- **Library.** Pick a viewing angle, then a tile: *Fold guide* and *Align*
  under Prepare, the 3D pictures below. The app reopens on the last picture.
- **Settings** (a drawer on the right, so the hinge stays visible):
  - *Picture*: viewing angle and style.
  - *Display*: full screen, and True size or Fit (a scaled preview; the 3D
    does not work in it).
  - *Hinge alignment* (iPhone Duo): **Adjust** opens the alignment card with
    large -0.5 / -0.1 / +0.1 / +0.5 mm buttons to move the gold line onto the
    fold; plus the status strip side, the system bar position when iOS
    reserves one, and the mirror switch.
  - *Tools*: setup guide, save or open the picture, pixel test, diagnostics,
    reset.
- **Setup guide** (starts on the first visit; later from the Library or Settings). A
  glass card on the left half shows each step with an animated drawing of the
  gesture:
  1. *Open*: phone flat like a book; choose a viewing angle.
  2. *Fold*: bring both sides toward you until the fold-guide picture runs true
     (straight lines, a perfect circle). Taller in the middle: fold further.
     Pinched: open slightly.
  3. *Centre*: on the Align card, nose in line with the hinge, eyes level with
     the gold centre line, both halves equally wide.
  4. *Look*: cross your eyes until the two dots become three, then look down.
  Tap the picture to hide the card (a small chip brings it back), swipe the
  card left or right to change step, or use Continue / Back.
- **Live fold angle (native wrapper only).** Web pages cannot read the hinge.
  A native iOS 27.1 app hosting this page can pass the angle from
  `UIHinge` / `onHingeChange` by calling `window.duovisionSetHingeAngle(deg)`
  or dispatching a `duovision-hinge` event with `detail.degrees` (180 = flat).
  The Fold step then shows the live angle and says how far to fold or open.
- Desktop keys: arrows (angle / picture), `s` style, `m` library, `,` settings,
  `i` diagnostics, `f` full screen, `p` True size / Fit, `[` `]` hinge position
  by 0.1 mm, Esc closes whatever is open.
- Saving to Photos: long-press the picture -> "Save to Photos", or Settings ->
  Save this picture -> Open, then long-press.
- The URL hash (`#cube/120/color`) remembers the current image.
