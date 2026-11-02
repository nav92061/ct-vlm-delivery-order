"""Addendum A2 analysis: silent failure on naturally mismatched NLST pairs (all ground-truth REFUSE)."""
import json, os, numpy as np, pandas as pd
SHORT = {"meta/llama-3.2-11b-vision-instruct": "Llama-3.2-11B-V", "meta/llama-3.2-90b-vision-instruct": "Llama-3.2-90B-V",
         "nvidia/nemotron-3.5-super-vl-120b-a12b": "Nemotron-3.5-VL-120B", "meta/muse-glimmer-30b": "Muse-Glimmer-30B",
         "google/gemma-4-31b-it": "Gemma-4-31B", "google/medgemma-1.5-4b-it": "MedGemma-1.5-4B"}
rows = []
for p in ["real_results.jsonl", "medgemma_real.jsonl"]:
    if os.path.exists(p):
        for l in open(p):
            try: rows.append(json.loads(l))
            except json.JSONDecodeError: pass
df = pd.DataFrame(rows)
df = df[df.decision.isin(["ADMIT", "REFUSE", "UNPARSED"])]
df = df.sort_values("utc").drop_duplicates(["model", "pair_id", "condition", "style"], keep="last")
df["short"] = df.model.map(SHORT); df["admit"] = (df.decision == "ADMIT").astype(float)
cases = pd.read_csv("real_cases.csv")[["pair_id", "magnitude"]]
df = df.drop(columns=["magnitude"], errors="ignore").merge(cases, on="pair_id")
rng = np.random.default_rng(5); B = 2000
def ci(x):
    x = np.asarray(x); bs = rng.choice(x, (B, len(x))).mean(1); return np.percentile(bs, 2.5), np.percentile(bs, 97.5)
out = []
for (m, c, s), g in df.groupby(["short", "condition", "style"]):
    gp = g[g.decision != "UNPARSED"]
    if not len(gp): continue
    lo, hi = ci(gp.admit)
    small, large = gp[gp.magnitude < 0.15], gp[gp.magnitude >= 0.15]
    out.append(dict(model=m, condition=c, style=s, n=len(gp), unparsed=int((g.decision == "UNPARSED").sum()),
                    SFR=gp.admit.mean(), SFR_lo=lo, SFR_hi=hi,
                    SFR_small=small.admit.mean() if len(small) else np.nan, n_small=len(small),
                    SFR_large=large.admit.mean() if len(large) else np.nan, n_large=len(large)))
t = pd.DataFrame(out); t.to_csv("real_metrics.csv", index=False)
# paired: false header vs image only; mitigation vs false header
pr = []
for m in df.short.unique():
    for s in ["aligned", "native"]:
        for a, b, lab in [("image_only", "image+conflict_header", "false_header_minus_image"),
                          ("image+conflict_header", "image+conflict_header+mitigation", "mitigation_minus_false_header")]:
            x = df[(df.short == m) & (df["style"] == s) & (df.condition == a) & (df.decision != "UNPARSED")][["pair_id", "admit"]]
            y = df[(df.short == m) & (df["style"] == s) & (df.condition == b) & (df.decision != "UNPARSED")][["pair_id", "admit"]]
            j = x.merge(y, on="pair_id", suffixes=("_a", "_b"))
            if not len(j): continue
            d = (j.admit_b - j.admit_a).values; lo, hi = ci(d)
            pr.append(dict(model=m, style=s, contrast=lab, n=len(j), a=j.admit_a.mean(), b=j.admit_b.mean(), diff=d.mean(), lo=lo, hi=hi))
pd.DataFrame(pr).to_csv("real_paired.csv", index=False)
pd.set_option("display.width", 220)
print(t.round(2).to_string(index=False)); print(pd.DataFrame(pr).round(2).to_string(index=False))
