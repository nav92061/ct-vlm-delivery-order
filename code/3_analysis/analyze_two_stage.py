"""Addendum A4 analysis. Paired per case: one-stage false header vs two-stage, and vs A1 instruction."""
import json, os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.getcwd())

SH = {"nvidia/nemotron-3.5-super-vl-120b-a12b": "Nemotron-3.5-VL-120B", "google/gemma-4-31b-it": "Gemma-4-31B"}
def rows(p):
    out = []
    for l in open(p):
        try: d = json.loads(l)
        except json.JSONDecodeError: continue
        if d.get("decision") in ("ADMIT", "REFUSE", "UNPARSED") and d.get("temperature", 0) == 0 and d.get("sample", 0) == 0:
            out.append(d)
    d = pd.DataFrame(out).sort_values("utc").drop_duplicates(["model", "pair_id", "variant", "condition", "style"], keep="last")
    return d[d.model.isin(SH)]
two = rows("two_stage_results.jsonl"); one = rows("study_results.jsonl"); mit = rows("mitigation_results.jsonl")
one = one[one.condition == "image+conflict_header"]; mit = mit[mit.condition == "image+conflict_header+mitigation"]
k = ["model", "pair_id", "variant", "style"]
j = two[k + ["patient", "defect", "decision", "stage1_decision"]].merge(one[k + ["decision"]], on=k, suffixes=("_two", "_one"))
j = j.merge(mit[k + ["decision"]].rename(columns={"decision": "decision_mit"}), on=k, how="left")
j["defect"] = j.defect.astype(str).str.lower().eq("true")
rng = np.random.default_rng(7); B = 2000
def pboot(x, a, b):
    pats = x.patient.unique(); g = {p_: v for p_, v in x.groupby("patient")}
    d = []
    for _ in range(B):
        s = pd.concat([g[p_] for p_ in rng.choice(pats, len(pats))]); d.append(s[b].mean() - s[a].mean())
    return np.percentile(d, [2.5, 97.5])
res = []
for (m, st), g in j.groupby(["model", "style"]):
    for defect, lab, tgt in [(True, "SFR_defective", "ADMIT"), (False, "refusal_clean_alarming_header", "REFUSE")]:
        x = g[(g.defect == defect) & g.decision_two.isin(["ADMIT", "REFUSE"]) & g.decision_one.isin(["ADMIT", "REFUSE"])].copy()
        if not len(x): continue
        x["one"] = (x.decision_one == tgt).astype(float); x["two"] = (x.decision_two == tgt).astype(float)
        lo, hi = pboot(x, "one", "two")
        rec = dict(model=SH[m], style=st, endpoint=lab, n=len(x), n_patients=x.patient.nunique(),
                   one_stage=x.one.mean(), two_stage=x.two.mean(), diff=x.two.mean() - x.one.mean(), lo=lo, hi=hi,
                   unparsed_two=int((g[g.defect == defect].decision_two == "UNPARSED").sum()))
        if defect:
            s1r = x[x.stage1_decision == "REFUSE"]
            rec.update(stage1_refused=len(s1r), reversed_to_admit=int((s1r.decision_two == "ADMIT").sum()))
            y = x[x.decision_mit.isin(["ADMIT", "REFUSE"])].copy()
            if len(y):
                y["mit"] = (y.decision_mit == "ADMIT").astype(float)
                rec.update(n_vs_mit=len(y), two_stage_on_mit_cases=y.two.mean(), mitigation=y.mit.mean())
        res.append(rec)
t = pd.DataFrame(res); t.to_csv("two_stage_metrics.csv", index=False)
pd.set_option("display.width", 250); print(t.round(3).to_string(index=False))
