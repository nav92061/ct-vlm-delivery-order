"""Addendum A3: pixel-only reference baseline on the exact renders shown to the VLMs. No header, no OCR."""
import numpy as np
from PIL import Image

BG, DATA_T, THR = 90, 40, 0.03

def panels(path):
    im = np.asarray(Image.open(path).convert("L")).astype(float)
    colfrac = (np.abs(im - BG) > 12).mean(0)
    on = np.r_[False, colfrac > 0.15, False]          # panel columns (tick labels cover far fewer rows)
    edges = np.flatnonzero(np.diff(on.astype(int)))
    runs = [(edges[i], edges[i + 1]) for i in range(0, len(edges) - 1, 2)]
    runs = [r for r in runs if r[1] - r[0] > 0.2 * im.shape[1]]
    runs = sorted(runs, key=lambda r: r[1] - r[0], reverse=True)[:2]
    runs = sorted(runs)
    assert len(runs) == 2, (path, edges)
    out = []
    for a, b in runs:
        p = im[:, a + 3:b - 3]
        rows = np.flatnonzero((np.abs(p - BG) > 12).mean(1) > 0.5)   # rows belonging to the panel (not canvas)
        out.append(p[rows.min():rows.max() + 1] if len(rows) else p)
        out[-1] = (out[-1], rows.min() if len(rows) else 0)
    return out

def score_aligned(path):
    (pa, ya), (pb, yb) = panels(path)
    # aligned view: both panels share one z-axis, so compare row-by-row on the full canvas extent
    top, bot = min(ya, yb), max(ya + len(pa), yb + len(pb))
    def mask(p, y0):
        m = np.zeros(bot - top, bool); m[y0 - top:y0 - top + len(p)] = p.max(1) > DATA_T; return m
    ma, mb = mask(pa, ya), mask(pb, yb)
    return float((ma != mb).sum() / max(ma.sum(), 1))

def score_native(path):
    (pa, _), (pb, _) = panels(path)
    fa, fb = pa.mean(1), pb.mean(1)
    ha, hb = len(fa), len(fb)
    za, zb = (fa - fa.mean()) / (fa.std() + 1e-6), (fb - fb.mean()) / (fb.std() + 1e-6)
    best, bd = -9, 0
    R = int(0.4 * ha)
    for d in range(-R, R + 1):                 # B row k aligns with A row k + d
        a0, a1 = max(0, d), min(ha, hb + d)
        if a1 - a0 < 0.5 * min(ha, hb): continue
        x, y = za[a0:a1], zb[a0 - d:a1 - d]
        c = float(np.mean((x - x.mean()) * (y - y.mean())) / (x.std() * y.std() + 1e-6))
        if c > best: best, bd = c, d
    return float(max(abs(ha - hb) / ha, abs(bd) / ha)), int(bd), ha, hb

def decide(score): return "REFUSE" if score > THR else "ADMIT"
