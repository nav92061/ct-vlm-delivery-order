"""Claude Haiku 5.5 / Sonnet 5.5: same endpoints as the main study (patient-clustered bootstrap)."""
import json, numpy as np, pandas as pd
rng = np.random.default_rng(11)
rows = []
import glob, os
FILES = [f"frontier_package/results_anthropic_{m}.jsonl" for m in ["claude-haiku-5-5", "claude-sonnet-5-5"]] + \
        sorted(glob.glob("hpc/*/out/qwen_results.jsonl"))
for fp in FILES:
    for l in open(fp):
        d = json.loads(l)
        if d.get("decision") in ("ADMIT", "REFUSE", "UNPARSED"): rows.append(d)
df = pd.DataFrame(rows)
df["defect"] = df.defect.astype(str).str.lower() == "true"
df["parsed"] = df.decision != "UNPARSED"; df["admit"] = (df.decision == "ADMIT").astype(float)
def boot(g, col="admit"):
    pats = g.patient.unique(); by = {p: x[col].values for p, x in g.groupby("patient")}
    bs = [np.concatenate([by[p] for p in rng.choice(pats, len(pats))]).mean() for _ in range(2000)]
    return np.percentile(bs, 2.5), np.percentile(bs, 97.5)
out = []
for (m, c, s, st), g in df.groupby(["model", "condition", "style", "set"]):
    gp = g[g.parsed]
    for ep, sub, flip in [("SFR", gp[gp.defect], False), ("FRR", gp[~gp.defect], True)]:
        if not len(sub): continue
        x = sub.copy(); x["v"] = 1 - x.admit if flip else x.admit
        lo, hi = boot(x, "v")
        out.append(dict(model=m, condition=c, style=s, set=st, endpoint=ep, n=len(x), n_patients=x.patient.nunique(),
                        unparsed=int((~g.parsed).sum()), rate=x.v.mean(), lo=lo, hi=hi))
t = pd.DataFrame(out); t.to_csv("frontier_metrics.csv", index=False)
# paired override and two-turn reversals, defective synthetic cases
k = ["pair_id", "variant", "style"]
res = []
for m in df.model.unique():
    d = df[(df.model == m) & (df.set == "synthetic") & df.defect & df.parsed]
    img = d[d.condition == "image_only"][k + ["patient", "admit"]].rename(columns={"admit": "img"})
    for cond in ["image+conflict_header", "image+conflict_header+two_stage", "C3_two_stage_true_header", "C4_two_stage_neutral",
                 "image+conflict_header+mitigation"]:
        x = d[d.condition == cond][k + ["admit"]].merge(img, on=k)
        for st in ["aligned", "native"]:
            y = x[x["style"] == st]
            if not len(y): continue
            caught = y[y.img == 0]
            res.append(dict(model=m, condition=cond, style=st, n=len(y), n_patients=y.patient.nunique(),
                            SFR_image_only=y.img.mean(), SFR_condition=y.admit.mean(),
                            refused_from_image=len(caught), reversed_to_admit=int(caught.admit.sum())))
r = pd.DataFrame(res); r.to_csv("frontier_paired.csv", index=False)
pd.set_option("display.width", 250)
print(r.round(3).to_string(index=False))
print(t[(t.set == "synthetic") & t.condition.isin(["image_only", "image+conflict_header", "image+correct_header", "header_only"])].round(3).to_string(index=False))
print(t[t.set == "real"].round(3).to_string(index=False))
