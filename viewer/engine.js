/* DuoVision real-time renderer (WebGL2).
 *
 * Runs the reference encoder's pipeline on the GPU, live, on the display's own pixel grid:
 *   1. partner pass   every pixel p: left-eye ray -> hidden scene (ray-marched height field) -> point S;
 *                     if the right eye sees S too, its line through S lands on the folded display at the
 *                     partner Q(p) (fractional). Same rules as duovision/encoder.py.
 *   2. jump passes    chains of links are followed by pointer doubling with sub-pixel interpolation.
 *   3. colour pass    colour = tile(C_bg(p) + r(p)).
 *
 * Colour is not a hash of the chain root (random dots). C_bg(p) is an exact invariant coordinate system
 * of the background-only stereogram on this fold: its first component advances by exactly 1 per background
 * link, so a tile with period 1 in that component satisfies every background link, everywhere, whatever
 * the object does. Each link through the object adds its phase error e to r, so the pattern beside a
 * moving object slides smoothly by sub-pixel amounts instead of re-randomising, and depth is continuous
 * (no whole-pixel disparity steps).
 *
 * The scenes come from the reference engine's scene pack (duovision/scenepack.py): this file has no
 * shape code except the analytic cube and rectangle.
 */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.DuoEngine = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  var JUMP_PASSES = 5;          // 2^5 = 32 links; chains are at most about 12 links across the display
  var TILE_SIZE = 512;

  // ------------------------------------------------------------------ GLSL
  var VS = "#version 300 es\nin vec2 aPos; void main(){ gl_Position = vec4(aPos, 0.0, 1.0); }";

  var COMMON = [
    "#version 300 es",
    "precision highp float; precision highp int; precision highp usampler2D; precision highp sampler2D;",
    "uniform vec2 uGrid;        // panel grid W, H (px)",
    "uniform float uMm;         // mm per grid px",
    "uniform vec2 uSC;          // sin, cos of half the hinge angle",
    "uniform vec2 uHalf;        // half width of one panel (mm), half height (mm)",
    "uniform vec3 uL, uR;       // eyes",
    "uniform vec2 uBg;          // z_bg0, slope of the V background",
    "uniform float uBand;       // reserved rows at the top",
    "const float BIG = 1e30;",
    "vec2 pixUV(vec2 ij){ return vec2((ij.x + 0.5 - uGrid.x * 0.5) * uMm, (uGrid.y * 0.5 - (ij.y + 0.5)) * uMm); }",
    "vec2 uvPix(vec2 uv){ return vec2(uv.x / uMm + uGrid.x * 0.5 - 0.5, uGrid.y * 0.5 - uv.y / uMm - 0.5); }",
    "vec3 uvWorld(vec2 uv){ return vec3(uv.x * uSC.x, uv.y, abs(uv.x) * uSC.y); }",
    "float bgZ(float x){ return uBg.x + uBg.y * abs(x); }",
    "// first t > 0 with the ray at or below the V background (port of scene._bg_first_t)",
    "float bgT(vec3 o, vec3 d){",
    "  float f0 = o.z - uBg.x - uBg.y * abs(o.x);",
    "  if (f0 <= 0.0) return 0.0;",
    "  float t = -BIG;",
    "  float den = d.z - uBg.y * d.x;",
    "  if (abs(den) > 1e-12) { float tp = (uBg.x + uBg.y * o.x - o.z) / den; if (tp > 0.0 && o.x + tp * d.x >= -1e-6) t = max(t, tp); }",
    "  den = d.z + uBg.y * d.x;",
    "  if (abs(den) > 1e-12) { float tm = (uBg.x - uBg.y * o.x - o.z) / den; if (tm > 0.0 && o.x + tm * d.x <= 1e-6) t = max(t, tm); }",
    "  return t > -BIG * 0.5 ? t : BIG;",
    "}",
    "// where the ray meets the folded display; ext = true ignores the outer and top/bottom edges",
    "bool panelHit(vec3 o, vec3 d, bool ext, out vec2 uv){",
    "  float best = BIG; bool ok = false; uv = vec2(0.0);",
    "  for (int k = 0; k < 2; k++) {",
    "    float sg = k == 0 ? -1.0 : 1.0;",
    "    vec3 n = vec3(-sg * uSC.y, 0.0, uSC.x), a = vec3(sg * uSC.x, 0.0, uSC.y);",
    "    float nd = dot(d, n);",
    "    if (abs(nd) < 1e-9) continue;",
    "    float t = -dot(o, n) / nd;",
    "    if (t <= 1e-9) continue;",
    "    vec3 p = o + t * d; float al = dot(p, a);",
    "    bool in_ = al >= -1e-7 && (ext || (al <= uHalf.x + 1e-7 && abs(p.y) <= uHalf.y + 1e-7));",
    "    if (in_ && t < best) { best = t; ok = true; uv = vec2(sg * max(al, 0.0), p.y); }",
    "  }",
    "  return ok;",
    "}",
    "// ---- canonical background coordinates: (phase in background periods from the hinge, height where the chain crosses it)",
    "vec2 bgStep(vec2 uv, bool left){",
    "  vec3 X = uvWorld(uv); vec3 e0 = left ? uL : uR, e1 = left ? uR : uL;",
    "  vec3 d = X - e0; float t = bgT(e0, d);",
    "  vec3 S = e0 + t * d; vec2 q; panelHit(e1, S - e1, true, q); return q;",
    "}",
    "vec2 canon(vec2 uv){",
    "  bool right = uv.x >= 0.0; vec2 cur = uv; float k = 0.0;",
    "  for (int n = 0; n < 16; n++) {",
    "    vec2 nx = bgStep(cur, right);",
    "    if (right ? nx.x < 0.0 : nx.x >= 0.0) {",
    "      float f = abs(cur.x) / max(abs(cur.x - nx.x), 1e-9);",
    "      return vec2((right ? 1.0 : -1.0) * (k + f), mix(cur.y, nx.y, f));",
    "    }",
    "    k += 1.0; cur = nx;",
    "  }",
    "  return vec2((right ? 1.0 : -1.0) * k, cur.y);",
    "}",
    "// ---- state texture: (Pi, Pj) as unsigned fixed point /4096, (ra, rb) as signed fixed point /65536",
    "uvec4 enc(vec2 P, vec2 r){ return uvec4(uint(P.x * 4096.0 + 0.5), uint(P.y * 4096.0 + 0.5), uint(int(floor(r.x * 65536.0 + 0.5))), uint(int(floor(r.y * 65536.0 + 0.5)))); }",
    "vec2 decP(uvec4 s){ return vec2(float(s.x), float(s.y)) / 4096.0; }",
    "vec2 decR(uvec4 s){ return vec2(float(int(s.z)), float(int(s.w))) / 65536.0; }",
    ""
  ].join("\n");

  var SCENE = [
    "uniform int uKind;         // 0 template, 1 cube, 2 rectangle",
    "uniform sampler2D uTpl;    // RG16F: depth / depth_max, mask; LINEAR",
    "uniform vec4 uTplGrid;     // x0, y0 (nominal mm of texel 0 centre), step, depth_max",
    "uniform vec2 uTplSize;",
    "uniform vec4 uPlanA;       // sc, yc, y_mid, tscale",
    "uniform vec4 uPlanB;       // d_base, span, z_lo, z_top",
    "uniform vec4 uBox;         // x0, x1, y0, y1 of the object's travel (world mm)",
    "uniform vec4 uPose;        // dx, dy, spin (rad), scale",
    "uniform vec3 uCube;        // half edge, rx, ry (rad)",
    "uniform vec2 uRect;        // half width, half height",
    "uniform float uStep;       // march step in z (mm)",
    "void cubeDepth(vec2 o, out float depth, out float m){",
    "  float cy = cos(uCube.z), sy = sin(uCube.z), cx = cos(uCube.y), sx = sin(uCube.y);",
    "  mat3 Ry = mat3(cy, 0.0, -sy, 0.0, 1.0, 0.0, sy, 0.0, cy);      // columns",
    "  mat3 Rx = mat3(1.0, 0.0, 0.0, 0.0, cx, sx, 0.0, -sx, cx);",
    "  mat3 R = Rx * Ry; float a = uCube.x, lo = -BIG, hi = BIG; m = 1.0;",
    "  for (int k = 0; k < 3; k++) {",
    "    vec3 n = R[k]; float c0 = n.x * o.x + n.y * o.y;",
    "    if (abs(n.z) < 1e-6) { if (abs(c0) > a) m = 0.0; continue; }",
    "    float t1 = (-a - c0) / n.z, t2 = (a - c0) / n.z;",
    "    lo = max(lo, min(t1, t2)); hi = min(hi, max(t1, t2));",
    "  }",
    "  if (hi <= lo) m = 0.0;",
    "  depth = m > 0.5 ? hi : 0.0;",
    "}",
    "// world z of the hidden surface at world (x, y): the arithmetic of scenepack.pack_surface",
    "float sceneZ(vec2 xy){",
    "  float bg = bgZ(xy.x);",
    "  if (xy.x < uBox.x || xy.x > uBox.y || xy.y < uBox.z || xy.y > uBox.w) return bg;",
    "  vec2 o = vec2(xy.x / uPlanA.x, (xy.y - uPlanA.y) / uPlanA.x + uPlanA.z) - uPose.xy;",
    "  float c = cos(uPose.z), s = sin(uPose.z);",
    "  o = vec2(c * o.x + s * o.y, -s * o.x + c * o.y) / (uPose.w * uPlanA.w);",
    "  float depth, m;",
    "  if (uKind == 1) cubeDepth(o, depth, m);",
    "  else if (uKind == 2) { m = (abs(o.x) <= uRect.x && abs(o.y) <= uRect.y) ? 1.0 : 0.0; depth = m; }",
    "  else {",
    "    vec2 f = (o - uTplGrid.xy) / uTplGrid.z;",
    "    if (f.x < 0.0 || f.y < 0.0 || f.x > uTplSize.x - 1.0 || f.y > uTplSize.y - 1.0) return bg;",
    "    vec2 t = texture(uTpl, (f + 0.5) / uTplSize).rg; depth = t.r * uTplGrid.w; m = t.g;",
    "  }",
    "  if (m <= 0.5) return bg;",
    "  float hn = clamp((depth * uPose.w - uPlanB.x) / uPlanB.y, 0.0, 1.0);",
    "  return max(uPlanB.z + hn * (uPlanB.w - uPlanB.z), bg);",
    "}",
    "// first t in (0, tmax) where the ray dips to or below the object, BIG if none",
    "float objT(vec3 o, vec3 d, float tmax){",
    "  vec3 lo = vec3(uBox.x, uBox.z, uBg.x - 1e-3), hi = vec3(uBox.y, uBox.w, uPlanB.w + 1e-3);",
    "  float t0 = 0.0, t1 = tmax;",
    "  for (int a = 0; a < 3; a++) {",
    "    if (abs(d[a]) < 1e-12) { if (o[a] < lo[a] || o[a] > hi[a]) return BIG; continue; }",
    "    float ta = (lo[a] - o[a]) / d[a], tb = (hi[a] - o[a]) / d[a];",
    "    t0 = max(t0, min(ta, tb)); t1 = min(t1, max(ta, tb));",
    "  }",
    "  if (t0 >= t1) return BIG;",
    "  float dt = uStep / max(abs(d.z), 0.1 * length(d));",
    "  float tp = t0; vec3 p = o + tp * d; float fp = p.z - sceneZ(p.xy);",
    "  if (fp <= 0.0) return tp;",
    "  for (int n = 0; n < 96; n++) {",
    "    float t = min(tp + dt, t1); p = o + t * d; float f = p.z - sceneZ(p.xy);",
    "    if (f <= 0.0) {",
    "      float a = tp, b = t, fa = fp, fb = f;",
    "      for (int k = 0; k < 10; k++) { float mid = 0.5 * (a + b); vec3 q = o + mid * d; float fm = q.z - sceneZ(q.xy); if (fm <= 0.0) { b = mid; fb = fm; } else { a = mid; fa = fm; } }",
    "      float w = fa - fb > 0.0 ? clamp(fa / (fa - fb), 0.0, 1.0) : 0.0;",
    "      return a + w * (b - a);",
    "    }",
    "    if (t >= t1) break;",
    "    tp = t; fp = f;",
    "  }",
    "  return BIG;",
    "}",
    "float firstT(vec3 o, vec3 d){ float tb = bgT(o, d); float to = objT(o, d, min(tb, BIG)); return min(tb, to); }",
    ""
  ].join("\n");

  // 1. partner pass: FBO row = image row j
  var FS_PARTNER = COMMON + SCENE + [
    "out uvec4 outState;",
    "void main(){",
    "  vec2 ij = floor(gl_FragCoord.xy);",
    "  uvec4 self = enc(ij, vec2(0.0));",
    "  outState = self;",
    "  if (ij.y < uBand) return;",
    "  vec2 uv = pixUV(ij); vec3 X = uvWorld(uv);",
    "  vec3 d = X - uL; float tb = bgT(uL, d); float to = objT(uL, d, tb); float t = min(tb, to);",
    "  if (t >= 1.0) return;                                   // the scene must lie between the eye and the pixel",
    "  vec3 S = uL + t * d;",
    "  vec3 d2 = S - uR; float t2 = firstT(uR, d2);            // is S visible from the right eye?",
    "  if (length((uR + t2 * d2) - S) > 0.05) return;",
    "  vec2 quv; if (!panelHit(uR, d2, false, quv)) return;",
    "  vec2 q = uvPix(quv);",
    "  if (q.x < 0.0 || q.x > uGrid.x - 1.0 || q.y < uBand || q.y > uGrid.y - 1.0 || q.x > ij.x - 1.0) return;",
    "  vec2 e = vec2(0.0);",
    "  if (to < tb) e = canon(quv) + vec2(1.0, 0.0) - canon(uv);   // background links have zero phase error by construction",
    "  outState = enc(q, e);",
    "}"
  ].join("\n");

  // 2. pointer doubling with sub-pixel interpolation of the displacement field
  var FS_JUMP = COMMON + [
    "uniform usampler2D uState;",
    "out uvec4 outState;",
    "void main(){",
    "  ivec2 ip = ivec2(gl_FragCoord.xy); vec2 ij = vec2(ip);",
    "  uvec4 s = texelFetch(uState, ip, 0);",
    "  outState = s;",
    "  if (s.xy == uvec2(uint(ij.x * 4096.0 + 0.5), uint(ij.y * 4096.0 + 0.5))) return;      // free: a root",
    "  vec2 P = decP(s); vec2 fl = floor(P), f = P - fl;",
    "  ivec2 b = ivec2(fl), mx = ivec2(uGrid) - 1;",
    "  vec2 D[4]; vec2 Rr[4]; float w[4];",
    "  w[0] = (1.0 - f.x) * (1.0 - f.y); w[1] = f.x * (1.0 - f.y); w[2] = (1.0 - f.x) * f.y; w[3] = f.x * f.y;",
    "  int best = 0;",
    "  for (int k = 0; k < 4; k++) {",
    "    ivec2 c = clamp(b + ivec2(k & 1, k >> 1), ivec2(0), mx);",
    "    uvec4 n = texelFetch(uState, c, 0);",
    "    D[k] = decP(n) - vec2(c); Rr[k] = decR(n);",
    "    if (w[k] > w[best]) best = k;",
    "  }",
    "  vec2 dd = vec2(0.0), rr = vec2(0.0); bool smooth_ = true;",
    "  for (int k = 0; k < 4; k++) { if (length(D[k] - D[best]) > 1.0) smooth_ = false; dd += w[k] * D[k]; rr += w[k] * Rr[k]; }",
    "  if (!smooth_) { dd = D[best]; rr = Rr[best]; }             // across a depth edge: take the nearest pixel's chain",
    "  vec2 P2 = clamp(P + dd, vec2(0.0), uGrid - 1.0);",
    "  outState = enc(P2, decR(s) + rr);",
    "}"
  ].join("\n");

  // 3. colour pass to the canvas (GL y is up; uOff is the canvas origin on the panel grid)
  var FS_COLOUR = COMMON + [
    "uniform usampler2D uState;",
    "uniform sampler2D uTile;",
    "uniform vec2 uOff; uniform float uCanvasH;",
    "uniform float uTileV;      // tile height in mm of psi",
    "uniform vec4 uDots;        // guide dots: left (i, j), right (i, j)",
    "uniform float uDotR;       // radius px",
    "uniform int uDebug;",
    "out vec4 outColor;",
    "void main(){",
    "  vec2 ij = vec2(floor(gl_FragCoord.x) + uOff.x, uCanvasH - 1.0 - floor(gl_FragCoord.y) + uOff.y);",
    "  if (ij.x < 0.0 || ij.y < 0.0 || ij.x >= uGrid.x || ij.y >= uGrid.y) { outColor = vec4(0.0, 0.0, 0.0, 1.0); return; }",
    "  if (ij.y < uBand) {",
    "    float v = 34.0 / 255.0;",
    "    float dl = length(ij - uDots.xy), dr = length(ij - uDots.zw);",
    "    v = mix(v, 1.0, clamp(uDotR + 0.5 - min(dl, dr), 0.0, 1.0));",
    "    float hx = uGrid.x * 0.5, off = max(1.5, 0.6 / uMm), tw = max(1.0, 0.25 / uMm), th = max(2.0, 0.35 * uBand);",
    "    float xc = ij.x + 0.5, tick = max(clamp(tw * 0.5 + 0.5 - abs(xc - (hx - off)), 0.0, 1.0), clamp(tw * 0.5 + 0.5 - abs(xc - (hx + off)), 0.0, 1.0));",
    "    tick *= clamp(ij.y + 0.5 - (uBand - th) + 0.5, 0.0, 1.0);",
    "    v = mix(v, 150.0 / 255.0, tick * step(min(dl, dr), 1e9) * (1.0 - clamp(uDotR + 0.5 - min(dl, dr), 0.0, 1.0)));",
    "    outColor = vec4(v, v, v, 1.0); return;",
    "  }",
    "  uvec4 s = texelFetch(uState, ivec2(ij), 0);",
    "  vec2 c = canon(pixUV(ij)) + decR(s);",
    "  if (uDebug == 1) { outColor = vec4(fract(c.x), fract(c.y / uTileV), 0.5 + 0.5 * decR(s).x, 1.0); return; }",
    "  outColor = vec4(texture(uTile, vec2(c.x, c.y / uTileV)).rgb, 1.0);",
    "}"
  ].join("\n");

  // ------------------------------------------------------------- patterns
  function mulberry32(a) {
    return function () { a |= 0; a = a + 0x6D2B79F5 | 0; var t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; };
  }
  // Periodic value noise on an n x n lattice, smoothstep-interpolated, sampled on a size x size tile.
  function periodicNoise(size, n, rnd) {
    var lat = new Float32Array(n * n), out = new Float32Array(size * size), i, j;
    for (i = 0; i < n * n; i++) lat[i] = rnd();
    for (j = 0; j < size; j++) {
      var y = j * n / size, y0 = Math.floor(y), fy = y - y0; fy = fy * fy * (3 - 2 * fy); var y1 = (y0 + 1) % n;
      for (i = 0; i < size; i++) {
        var x = i * n / size, x0 = Math.floor(x), fx = x - x0; fx = fx * fx * (3 - 2 * fx); var x1 = (x0 + 1) % n;
        var a = lat[y0 * n + x0] + (lat[y0 * n + x1] - lat[y0 * n + x0]) * fx, b = lat[y1 * n + x0] + (lat[y1 * n + x1] - lat[y1 * n + x0]) * fx;
        out[j * size + i] = a + (b - a) * fy;
      }
    }
    return out;
  }
  var PALETTES = {
    // deep navy -> azure -> ice -> white, with a warm fleck: reads as water / marble, strong luminance range
    azure: [[6, 14, 38], [16, 52, 120], [40, 118, 214], [150, 205, 250], [246, 250, 255]],
    mono: [[8, 8, 10], [70, 72, 78], [150, 153, 160], [225, 228, 233], [255, 255, 255]]
  };
  function ramp(pal, t) {
    t = Math.min(0.99999, Math.max(0, t)) * (pal.length - 1);
    var k = Math.floor(t), f = t - k, a = pal[k], b = pal[k + 1];
    return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
  }
  /* A seamless tile, periodic in both directions: large soft forms (the look), mid-scale texture, and
   * fine high-entropy detail down to about two display pixels (what stereo matching locks on to). The
   * signal is carried in luminance; hue is decoration. */
  // Octave weights for lattices of 3, 6, 12, 24, 48, 96, 192 cells per period (features from about 90 px down
  // to 1.4 px on the panel). Most of the energy sits at 3-11 px: with the weight on the large forms instead,
  // the reference viewer mis-matched 7 % of a full-resolution frame; with these it reads 98 % within 2 mm.
  var TILE_WEIGHTS = [0.55, 0.5, 0.5, 0.7, 0.95, 0.85, 0.45];
  function makeTile(name, seed, weights) {
    var size = TILE_SIZE, rnd = mulberry32(seed || 7), pal = PALETTES[name] || PALETTES.azure, wt = weights || TILE_WEIGHTS;
    var oct = [[3, wt[0]], [6, wt[1]], [12, wt[2]], [24, wt[3]], [48, wt[4]], [96, wt[5]], [192, wt[6]]], layers = [], k, i;
    for (k = 0; k < oct.length; k++) layers.push(periodicNoise(size, oct[k][0], rnd));
    var speck = new Float32Array(size * size); for (i = 0; i < speck.length; i++) speck[i] = rnd();
    var px = new Uint8Array(size * size * 4), norm = 0; for (k = 0; k < oct.length; k++) norm += oct[k][1];
    var vals = new Float32Array(size * size), lo = 1e9, hi = -1e9;
    for (i = 0; i < size * size; i++) {
      var v = 0; for (k = 0; k < oct.length; k++) v += oct[k][1] * layers[k][i];
      v = v / norm + (speck[i] - 0.5) * 0.10; vals[i] = v; if (v < lo) lo = v; if (v > hi) hi = v;
    }
    // equalise: map through the sorted rank so every luminance band gets equal area (about 50 % light / dark)
    var idx = new Uint32Array(size * size); for (i = 0; i < idx.length; i++) idx[i] = i;
    idx.sort(function (a, b) { return vals[a] - vals[b]; });
    for (i = 0; i < idx.length; i++) {
      var c = ramp(pal, i / idx.length), o = idx[i] * 4;
      px[o] = c[0]; px[o + 1] = c[1]; px[o + 2] = c[2]; px[o + 3] = 255;
    }
    return px;
  }

  // ------------------------------------------------------- CPU geometry (guide dots)
  function intersectJS(g, o, d) {     // nearest hit of a ray with the folded display, inside the panels
    var best = Infinity, res = null;
    for (var k = 0; k < 2; k++) {
      var sg = k === 0 ? -1 : 1, n = [-sg * g.c, 0, g.s], a = [sg * g.s, 0, g.c];
      var nd = d[0] * n[0] + d[2] * n[2]; if (Math.abs(nd) < 1e-9) continue;
      var t = -(o[0] * n[0] + o[2] * n[2]) / nd; if (t <= 1e-9) continue;
      var p = [o[0] + t * d[0], o[1] + t * d[1], o[2] + t * d[2]], al = p[0] * a[0] + p[2] * a[2];
      if (al >= -1e-7 && al <= g.halfW + 1e-7 && Math.abs(p[1]) <= g.halfH + 1e-7 && t < best) { best = t; res = [sg * Math.max(al, 0), p[1]]; }
    }
    return res;
  }
  function guideDots(g, view, bandRows) {
    var half = view.ipd_mm / 2, ye = view.eye_height_mm, ze = view.distance_mm, zg = view.z_background;
    var eyes = [[-half, ye, ze], [half, ye, ze]], zs = 0, k;
    for (k = 0; k < 2; k++) { var h = intersectJS(g, eyes[k], [0 - eyes[k][0], 0, zg - ze]); if (!h) return null; zs += Math.abs(h[0]) * g.c / 2; }
    var vMid = (g.H / 2 - ((bandRows - 1) / 2 + 0.5)) * g.mm;
    var yg = ye + (vMid - ye) * (ze - zg) / (ze - zs), dots = [];
    for (k = 0; k < 2; k++) {
      var q = intersectJS(g, eyes[k], [0 - eyes[k][0], yg - ye, zg - ze]); if (!q) return null;
      dots.push([q[0] / g.mm + g.W / 2 - 0.5, g.H / 2 - q[1] / g.mm - 0.5]);
    }
    dots.sort(function (a, b) { return a[0] - b[0]; });
    return dots;
  }

  // --------------------------------------------------------------- engine
  function compile(gl, type, src) {
    var sh = gl.createShader(type); gl.shaderSource(sh, src); gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) { var log = gl.getShaderInfoLog(sh); gl.deleteShader(sh); throw new Error("shader: " + log); }
    return sh;
  }
  function program(gl, fs) {
    var p = gl.createProgram(); gl.attachShader(p, compile(gl, gl.VERTEX_SHADER, VS)); gl.attachShader(p, compile(gl, gl.FRAGMENT_SHADER, fs));
    gl.bindAttribLocation(p, 0, "aPos"); gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error("link: " + gl.getProgramInfoLog(p));
    var u = {}, n = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS);
    for (var i = 0; i < n; i++) { var info = gl.getActiveUniform(p, i); u[info.name.replace(/\[0\]$/, "")] = gl.getUniformLocation(p, info.name); }
    return { p: p, u: u };
  }

  function Engine(canvas, pack, opts) {
    opts = opts || {};
    this.canvas = canvas; this.pack = pack; this.pattern = opts.pattern || "azure"; this.debug = 0;
    this.templates = {};          // name -> { data: Float32Array RG, nx, ny }
    this.grid = null; this.sceneName = null; this.hinge = null; this.lost = false;
    var self = this;
    canvas.addEventListener("webglcontextlost", function (ev) { ev.preventDefault(); self.lost = true; }, false);
    canvas.addEventListener("webglcontextrestored", function () { try { self._init(); self.lost = false; } catch (e) { self.lost = true; } }, false);
    this._init();
  }
  Engine.supported = function () {
    try {
      var c = document.createElement("canvas"), gl = c.getContext("webgl2", { failIfMajorPerformanceCaveat: false });
      if (!gl) return false;
      if (!gl.getExtension("EXT_color_buffer_float")) { /* not needed: state is RGBA32UI */ }
      return gl.getParameter(gl.MAX_TEXTURE_SIZE) >= 4096;
    } catch (e) { return false; }
  };
  Engine.prototype._init = function () {
    var gl = this.canvas.getContext("webgl2", { antialias: false, alpha: false, depth: false, stencil: false, preserveDrawingBuffer: false, powerPreference: "high-performance" });
    if (!gl) throw new Error("WebGL2 is not available");
    this.gl = gl;
    this.progPartner = program(gl, FS_PARTNER); this.progJump = program(gl, FS_JUMP); this.progColour = program(gl, FS_COLOUR);
    var vb = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, vb);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    gl.disable(gl.DEPTH_TEST); gl.disable(gl.BLEND); gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    this.fbo = gl.createFramebuffer(); this.state = [null, null]; this.stateSize = [0, 0];
    this.tileTex = null; this.tileName = null; this.tplTex = {}; 
    if (this.grid) { var g = this.grid; this.grid = null; this.setGrid(g); }
  };
  Engine.prototype._tile = function () {
    var gl = this.gl;
    var key = this.pattern + "|" + (this.tileWeights ? this.tileWeights.join(",") : "");
    if (this.tileTex && this.tileName === key) return;
    if (!this.tileTex) this.tileTex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.tileTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, TILE_SIZE, TILE_SIZE, 0, gl.RGBA, gl.UNSIGNED_BYTE, makeTile(this.pattern, 7, this.tileWeights));
    gl.generateMipmap(gl.TEXTURE_2D);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    this.tileName = key;
  };
  /* grid: { W, H: the whole picture in pixels; mm: millimetres per pixel; offX, offY: where the canvas's
   * top-left pixel sits on that grid; cw, ch: canvas backing size }. */
  Engine.prototype.setGrid = function (g) {
    var gl = this.gl, cur = this.grid;
    this.grid = { W: g.W, H: g.H, mm: g.mm, offX: g.offX || 0, offY: g.offY || 0, cw: g.cw || g.W, ch: g.ch || g.H };
    if (this.canvas.width !== this.grid.cw) this.canvas.width = this.grid.cw;
    if (this.canvas.height !== this.grid.ch) this.canvas.height = this.grid.ch;
    if (this.stateSize[0] !== g.W || this.stateSize[1] !== g.H) {
      for (var k = 0; k < 2; k++) {
        if (this.state[k]) gl.deleteTexture(this.state[k]);
        var t = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, t);
        gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA32UI, g.W, g.H);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        this.state[k] = t;
      }
      this.stateSize = [g.W, g.H];
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.state[0], 0);
      if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) throw new Error("integer render target not supported");
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    }
    return cur;
  };
  /* Template data: ArrayBuffer of scene_<name>.bin (uint16 depth then uint8 mask, rows bottom -> top). */
  Engine.prototype.addTemplate = function (name, buf) {
    var t = this.pack.scenes[name].template, n = t.nx * t.ny, d = new Uint16Array(buf, 0, n), m = new Uint8Array(buf, 2 * n, n);
    var rg = new Float32Array(2 * n);
    for (var i = 0; i < n; i++) { rg[2 * i] = d[i] / 65535; rg[2 * i + 1] = m[i]; }
    this.templates[name] = { rg: rg, nx: t.nx, ny: t.ny };
  };
  Engine.prototype.needsTemplate = function (name) {
    var s = this.pack.scenes[name]; return !!(s && s.kind === "template" && !this.templates[name]);
  };
  Engine.prototype._tplTexture = function (name) {
    var gl = this.gl;
    if (this.tplTex[name]) return this.tplTex[name];
    var t = this.templates[name], tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RG16F, t.nx, t.ny, 0, gl.RG, gl.FLOAT, t.rg);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return (this.tplTex[name] = tex);
  };
  Engine.prototype.hasScene = function (name, hinge) {
    var s = this.pack.scenes[name]; return !!(s && s.plans[String(hinge)]);
  };
  Engine.prototype.setScene = function (name, hinge) { this.sceneName = name; this.hinge = hinge; };
  Engine.prototype.poseAt = function (name, phase) {       // linear interpolation of the sampled loop
    var ps = this.pack.scenes[name].pose, n = ps.dx.length, f = ((phase % 1) + 1) % 1 * n, k0 = Math.floor(f) % n, k1 = (k0 + 1) % n, t = f - Math.floor(f), out = {};
    ["dx", "dy", "spin", "scale", "ry"].forEach(function (key) {
      var a = ps[key][k0], b = ps[key][k1]; if ((key === "spin" || key === "ry") && b < a) b += 90;
      out[key] = a + (b - a) * t;
    });
    return out;
  };
  Engine.prototype._common = function (prog) {
    var gl = this.gl, u = prog.u, g = this.grid, v = this.pack.view, plan = this.pack.scenes[this.sceneName].plans[String(this.hinge)];
    var half = this.hinge * Math.PI / 360, dev = this.pack.device;
    gl.uniform2f(u.uGrid, g.W, g.H); gl.uniform1f(u.uMm, g.mm);
    gl.uniform2f(u.uSC, Math.sin(half), Math.cos(half));
    gl.uniform2f(u.uHalf, g.W * g.mm / 2, g.H * g.mm / 2);
    gl.uniform3f(u.uL, -v.ipd_mm / 2, v.eye_height_mm, v.distance_mm); gl.uniform3f(u.uR, v.ipd_mm / 2, v.eye_height_mm, v.distance_mm);
    gl.uniform2f(u.uBg, plan.z_bg0, plan.slope);
    gl.uniform1f(u.uBand, Math.round(g.H * this.pack.band_fraction));
    return plan;
  };
  Engine.prototype.render = function (phase) {
    if (this.lost || !this.grid || !this.sceneName) return false;
    var gl = this.gl, g = this.grid, sc = this.pack.scenes[this.sceneName], pr, u, plan, k;
    if (sc.kind === "template" && !this.templates[this.sceneName]) return false;
    var pose = this.poseAt(this.sceneName, phase);
    // 1. partner pass
    pr = this.progPartner; u = pr.u; gl.useProgram(pr.p); plan = this._common(pr);
    gl.uniform1i(u.uKind, sc.kind === "cube" ? 1 : sc.kind === "rect" ? 2 : 0);
    gl.activeTexture(gl.TEXTURE0);
    if (sc.kind === "template") {
      var t = sc.template; gl.bindTexture(gl.TEXTURE_2D, this._tplTexture(this.sceneName));
      gl.uniform4f(u.uTplGrid, t.x0, t.y0, t.step, t.depth_max); gl.uniform2f(u.uTplSize, t.nx, t.ny);
    } else { gl.bindTexture(gl.TEXTURE_2D, null); }
    gl.uniform1i(u.uTpl, 0);
    gl.uniform4f(u.uPlanA, plan.sc, plan.yc, plan.y_mid, plan.tscale); gl.uniform4f(u.uPlanB, plan.d_base, plan.span, plan.z_lo, plan.z_top);
    gl.uniform4f(u.uBox, plan.box[0], plan.box[1], plan.box[2], plan.box[3]);
    gl.uniform4f(u.uPose, pose.dx, pose.dy, pose.spin * Math.PI / 180, pose.scale);
    if (sc.kind === "cube") gl.uniform3f(u.uCube, sc.cube.edge / 2, sc.cube.rx * Math.PI / 180, pose.ry * Math.PI / 180);
    if (sc.kind === "rect") gl.uniform2f(u.uRect, plan.rect[0], plan.rect[1]);
    gl.uniform1f(u.uStep, this.pack.march_step_mm * 2);       // 1 mm in z; rays are within about 16 deg of the z axis
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.state[0], 0);
    gl.viewport(0, 0, g.W, g.H); gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    // 2. pointer doubling
    pr = this.progJump; u = pr.u; gl.useProgram(pr.p); this._common(pr); gl.uniform1i(u.uState, 0);
    var src = 0;
    for (k = 0; k < JUMP_PASSES; k++) {
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.state[1 - src], 0);
      gl.bindTexture(gl.TEXTURE_2D, this.state[src]); gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4); src = 1 - src;
    }
    // 3. colour
    this._tile();
    gl.bindFramebuffer(gl.FRAMEBUFFER, null); gl.viewport(0, 0, g.cw, g.ch);
    pr = this.progColour; u = pr.u; gl.useProgram(pr.p); this._common(pr);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.state[src]); gl.uniform1i(u.uState, 0);
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, this.tileTex); gl.uniform1i(u.uTile, 1);
    gl.uniform2f(u.uOff, g.offX, g.offY); gl.uniform1f(u.uCanvasH, g.ch);
    gl.uniform1f(u.uTileV, this.pack.view.sep_bg_mm);          // square texels: one period wide, one period tall
    var band = Math.round(g.H * this.pack.band_fraction), half = this.hinge * Math.PI / 360;
    var dots = guideDots({ W: g.W, H: g.H, mm: g.mm, s: Math.sin(half), c: Math.cos(half), halfW: g.W * g.mm / 2, halfH: g.H * g.mm / 2 }, this.pack.view, band);
    if (dots) gl.uniform4f(u.uDots, dots[0][0], dots[0][1], dots[1][0], dots[1][1]); else gl.uniform4f(u.uDots, -99, -99, -99, -99);
    gl.uniform1f(u.uDotR, 1.2 / g.mm); gl.uniform1i(u.uDebug, this.debug);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    gl.activeTexture(gl.TEXTURE0);
    return true;
  };
  Engine.prototype.dispose = function () {
    var ext = this.gl && this.gl.getExtension("WEBGL_lose_context"); if (ext) ext.loseContext();
  };

  return { Engine: Engine, makeTile: makeTile, guideDots: guideDots, PALETTES: PALETTES, TILE_SIZE: TILE_SIZE };
});
