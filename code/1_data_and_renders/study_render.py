"""Main study: defects with randomized magnitudes, two render styles, prompts.
aligned : both series on a shared physical z-axis, black = no data (explicit cue).
native  : each series stacked by slice index, top-aligned, own z tick labels, no padding
          (what a viewer shows; defects must be inferred from anatomy and tick values)."""
import io, base64, numpy as np
from PIL import Image, ImageDraw, ImageFont

WL, WW = -300, 1400
MM_PER_PX_X = 1.4          # horizontal resample (both series to same physical width scale)
PX_PER_MM_Z = 1.0
try: FONT = ImageFont.load_default(size=15)
except TypeError: FONT = ImageFont.load_default()

def sample_defects(rng):
    return {"truncation": {"frac": float(rng.uniform(0.10, 0.35)), "side": str(rng.choice(["inferior", "superior"]))},
            "gap": {"frac": float(rng.uniform(0.05, 0.20)), "center": float(rng.uniform(0.3, 0.7))},
            "zshift": {"frac": float(rng.uniform(0.05, 0.20)), "dir": str(rng.choice(["B_superior", "B_inferior"]))}}

def apply(ca, za, cb, zb, variant, p):
    """Return (ca, za, cb, zb) after the defect. Defects act on series B (z-shift trims both)."""
    n = len(zb)
    if variant == "clean": return ca, za, cb, zb
    if variant == "truncation":
        k = max(1, int(round(n * p["frac"])))
        return (ca, za, cb[k:], zb[k:]) if p["side"] == "inferior" else (ca, za, cb[:-k], zb[:-k])
    if variant == "gap":
        k = max(1, int(round(n * p["frac"]))); s = int(round(n * p["center"] - k / 2)); s = min(max(s, 1), n - k - 1)
        keep = np.r_[0:s, s + k:n]; return ca, za, cb[keep], zb[keep]
    if variant == "zshift":                     # equal length; B moved toward one end
        k = max(1, int(round(len(za) * p["frac"])))
        if p["dir"] == "B_superior": return ca[:-k], za[:-k], cb[k:], zb[k:]
        return ca[k:], za[k:], cb[:-k], zb[:-k]
    raise ValueError(variant)

def gray(a):
    return (np.clip((a.astype(np.float32) - (WL - WW / 2)) / WW, 0, 1) * 255).astype(np.uint8)

def resample_x(c, x0, dx, xmin, xmax):
    xs = np.arange(xmin, xmax, MM_PER_PX_X); idx = np.round((xs - x0) / dx).astype(int)
    ok = (idx >= 0) & (idx < c.shape[1]); out = np.full((c.shape[0], len(xs)), -1024, np.int16)
    out[:, ok] = c[:, idx[ok]]; return out

def _label(d, x, y, s): d.text((x, y), s, fill=255, font=FONT)

def render_aligned(A, B):
    (ca, za), (cb, zb) = A, B
    zmin = min(za.min(), zb.min()) - 10; zmax = max(za.max(), zb.max()) + 10
    H = int(round((zmax - zmin) * PX_PER_MM_Z)) + 1; W = ca.shape[1]
    def place(c, z):
        img = np.zeros((H, W), np.uint8); th = np.median(np.diff(z)) if len(z) > 1 else 1
        g = gray(c)
        for row, zz in zip(g, z):
            t = int(round((zmax - (zz + th / 2)) * PX_PER_MM_Z)); b = int(round((zmax - (zz - th / 2)) * PX_PER_MM_Z))
            img[max(t, 0):min(b + 1, H)] = row
        return Image.fromarray(img)
    ml, gap, top = 70, 20, 30
    cv = Image.new("L", (ml + 2 * W + gap + 10, H + top + 10), 90)
    cv.paste(place(ca, za), (ml, top)); cv.paste(place(cb, zb), (ml + W + gap, top)); d = ImageDraw.Draw(cv)
    _label(d, ml + W // 2 - 30, 6, "Series A"); _label(d, ml + W + gap + W // 2 - 30, 6, "Series B")
    for zt in np.arange(np.ceil(zmin / 50) * 50, zmax, 50):
        y = top + int(round((zmax - zt) * PX_PER_MM_Z)); d.line((ml - 8, y, ml - 1, y), fill=255); _label(d, 4, y - 8, f"{int(zt)}")
    _label(d, 4, top + H // 2 + 14, "z (mm)")
    return cv.convert("RGB")

def render_native(A, B):
    """Slices stacked by index (superior at top), each panel with its own z labels every ~50 mm of slices."""
    panels = []
    for c, z in (A, B):
        order = np.argsort(-z); g = gray(c[order]); zz = z[order]
        th = np.median(np.abs(np.diff(zz))) if len(zz) > 1 else 1
        h = int(round(len(zz) * th * PX_PER_MM_Z))
        panels.append((Image.fromarray(g).resize((g.shape[1], h)), zz, h))
    W = panels[0][0].size[0]; H = max(p[2] for p in panels); lw, gap, top = 62, 16, 30
    cv = Image.new("L", (2 * (lw + W) + gap + 10, H + top + 10), 90); d = ImageDraw.Draw(cv)
    for j, (im, zz, h) in enumerate(panels):
        x = j * (lw + W + gap) + lw; cv.paste(im, (x, top))
        _label(d, x + W // 2 - 30, 6, "Series " + "AB"[j])
        step = max(1, int(round(len(zz) / max(h / 50, 1))))
        for k in range(0, len(zz), step):
            y = top + int(round(k * h / len(zz))); d.line((x - 7, y, x - 1, y), fill=255); _label(d, x - lw + 2, y - 8, f"{zz[k]:.0f}")
    _label(d, 4, top + H + 0 - 4 if False else top - 22, "")
    return cv.convert("RGB")

def jpeg_b64(img, max_side=640, q=85):
    s = max_side / max(img.size)
    if s < 1: img = img.resize((int(img.size[0] * s), int(img.size[1] * s)))
    b = io.BytesIO(); img.save(b, format="JPEG", quality=q); return base64.b64encode(b.getvalue()).decode()

def header_text(za, zb, ta, tb):
    def seg(z, t):
        dz = np.diff(np.sort(z)); g = int((dz > 1.5 * np.median(dz)).sum()) if len(dz) else 0
        return f"z from {z.min():.1f} to {z.max():.1f} mm, {len(z)} slices, slice thickness {t} mm, {g} interior gaps"
    return f"Series A: {seg(za, ta)}. Series B: {seg(zb, tb)}."

TASK_IMG = {
 "aligned": ("You are checking inputs for a paired CT volumetry analysis. The image shows mid-coronal reformats of two "
             "reconstructions (Series A and Series B) from the same chest CT session, placed on a common patient z-axis "
             "(vertical, in mm). Black regions contain no image data."),
 "native": ("You are checking inputs for a paired CT volumetry analysis. The image shows mid-coronal reformats of two "
            "reconstructions (Series A and Series B) from the same chest CT session. Each series is displayed by slice "
            "order from superior (top) to inferior (bottom), and the numbers beside each series give the z-position "
            "(mm) of the labelled slices."),
 "none": "You are checking inputs for a paired CT volumetry analysis. You are given header metadata for two reconstructions "
         "(Series A and Series B) from the same chest CT session."}
RULE = "The pair is admissible only if both series cover the same z-range and neither has missing slices."
FORMAT = ("Reply with exactly one line starting 'DECISION: ADMIT' or 'DECISION: REFUSE', followed by one short sentence "
          "giving the reason.")

def prompt(style, hdr=None):
    s = f"{TASK_IMG[style]} {RULE}"
    if hdr: s += f"\n\nDICOM header summary: {hdr}"
    return s + f"\n\n{FORMAT}"
