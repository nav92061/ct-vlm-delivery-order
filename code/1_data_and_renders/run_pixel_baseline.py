"""Addendum A3: score every main-grid and real-pair render; write pixel_baseline_cases.csv and pixel_baseline_metrics.csv."""
import os, sys, json, numpy as np, pandas as pd
sys.path.insert(0, os.getcwd())
import pixel_baseline as PB
cases = pd.read_csv("study_cases.csv"); real = pd.read_csv("real_cases.csv")
rows = []
for _, c in cases.iterrows():
    for st in ["aligned", "native"]:
        f = f"study_renders/{c.pair_id}_{c.variant}_{st}.jpg"
        s = PB.score_aligned(f) if st == "aligned" else PB.score_native(f)[0]
        rows.append(dict(set="synthetic", pair_id=c.pair_id, patient=c.patient, variant=c.variant, magnitude=c.magnitude,
                         defect=c.variant != "clean", style=st, score=s))
for _, c in real.iterrows():
    for st in ["aligned", "native"]:
        f = f"real_renders/{c.pair_id}_real_{st}.jpg"
        s = PB.score_aligned(f) if st == "aligned" else PB.score_native(f)[0]
        rows.append(dict(set="real", pair_id=c.pair_id, patient=c.patient, variant="real", magnitude=c.magnitude,
                         defect=True, style=st, score=s))
d = pd.DataFrame(rows); d["admit"] = (d.score <= PB.THR).astype(float)
d.to_csv("pixel_baseline_cases.csv", index=False)
rng = np.random.default_rng(6); B = 2000
def boot(g, f):
    pats = g.patient.unique(); by = {p: x for p, x in g.groupby("patient")}
    return np.percentile([f(pd.concat([by[p] for p in rng.choice(pats, len(pats))])) for _ in range(B)], [2.5, 97.5])
def auc(g):
    pos, neg = g[g.defect].score.values, g[~g.defect].score.values
    if not len(pos) or not len(neg): return np.nan
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))
out = []
for (s_, st), g in d.groupby(["set", "style"]):
    sfr = lambda x: x[x.defect].admit.mean()
    frr = lambda x: (1 - x[~x.defect].admit).mean() if (~x.defect).any() else np.nan
    lo, hi = boot(g, sfr)
    rec = dict(set=s_, style=st, n=len(g), n_patients=g.patient.nunique(), SFR=sfr(g), SFR_lo=lo, SFR_hi=hi, FRR=frr(g), AUC=auc(g))
    if (~g.defect).any(): rec["FRR_lo"], rec["FRR_hi"] = boot(g, frr)
    for v in ["truncation", "gap", "zshift"]:
        x = g[g.variant == v]; rec[f"SFR_{v}"] = x.admit.mean() if len(x) else np.nan
    m = g[g.defect]; rec["score_vs_magnitude_r"] = float(np.corrcoef(m.magnitude, m.score)[0, 1])
    rec["min_defect_score"] = float(m.score.min()); rec["max_clean_score"] = float(g[~g.defect].score.max()) if (~g.defect).any() else np.nan
    out.append(rec)
t = pd.DataFrame(out); t.to_csv("pixel_baseline_metrics.csv", index=False)
pd.set_option("display.width", 250); print(t.round(3).to_string(index=False))
