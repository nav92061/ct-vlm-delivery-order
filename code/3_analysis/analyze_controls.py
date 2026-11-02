"""A5 analysis: each control paired with its reference condition (same model, case, view)."""
import json, numpy as np, pandas as pd
M = "nvidia/nemotron-3.5-super-vl-120b-a12b"
def rows(p):
    o = []
    for l in open(p):
        try: d = json.loads(l)
        except json.JSONDecodeError: continue
        if d.get("model") == M and d.get("decision") in ("ADMIT", "REFUSE", "UNPARSED") and d.get("temperature", 0) == 0 and d.get("sample", 0) == 0:
            o.append(d)
    d = pd.DataFrame(o).sort_values("utc").drop_duplicates(["pair_id", "variant", "condition", "style"], keep="last")
    d["defect"] = d.defect.astype(str).str.lower().eq("true"); return d
R = pd.concat([rows("study_results.jsonl"), rows("two_stage_results.jsonl"), rows("controls_results.jsonl")])
PAIRS = [("C1_raw_false_header", "image+conflict_header", ["aligned", "native"], "raw fields vs summary false header"),
         ("C1_raw_false_header", "image_only", ["aligned", "native"], "raw false header vs image only"),
         ("C2_neutral_metadata", "image_only", ["aligned", "native"], "neutral text vs image only"),
         ("C3_two_stage_true_header", "image+conflict_header+two_stage", ["aligned", "native"], "two-turn true vs false header"),
         ("C4_two_stage_neutral", "image+conflict_header+two_stage", ["aligned", "native"], "two-turn neutral vs false header"),
         ("C5_native_no_zlabels", "image_only", ["native"], "no z labels vs with labels")]
rng = np.random.default_rng(11); B = 2000; out = []
for c, ref, styles, lab in PAIRS:
    for st in styles:
        a = R[(R.condition == ref) & (R["style"] == st)]; b = R[(R.condition == c) & (R["style"] == st)]
        j = a.merge(b, on=["pair_id", "variant"], suffixes=("_r", "_c"))
        for defect, ep in [(True, "SFR"), (False, "FRR")]:
            x = j[(j.defect_r == defect)]
            unp = int((x.decision_c == "UNPARSED").sum())
            x = x[x.decision_r.isin(["ADMIT", "REFUSE"]) & x.decision_c.isin(["ADMIT", "REFUSE"])].copy()
            if not len(x): continue
            tgt = "ADMIT" if defect else "REFUSE"
            x["r"] = (x.decision_r == tgt).astype(float); x["c"] = (x.decision_c == tgt).astype(float)
            pats = x.patient_r.unique(); g = {p: v for p, v in x.groupby("patient_r")}
            bs = [(lambda s: s.c.mean() - s.r.mean())(pd.concat([g[p] for p in rng.choice(pats, len(pats))])) for _ in range(B)]
            rec = dict(control=c, reference=ref, comparison=lab, style=st, endpoint=ep, n=len(x), unparsed=unp,
                       reference_rate=x.r.mean(), control_rate=x.c.mean(), diff=x.c.mean() - x.r.mean(),
                       lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5))
            if defect and "two_stage" in ref:
                s1 = x[x.get("stage1_decision_c", pd.Series(index=x.index, dtype=object)) == "REFUSE"] if "stage1_decision_c" in x else x.iloc[:0]
                rec.update(stage1_refused=len(s1), reversed=int((s1.decision_c == "ADMIT").sum()))
            if c == "C5_native_no_zlabels" and defect:
                for v in ["truncation", "gap", "zshift"]:
                    y = x[x.variant == v]; rec[f"{v}_with_labels"] = y.r.mean(); rec[f"{v}_no_labels"] = y.c.mean()
            out.append(rec)
t = pd.DataFrame(out); t.to_csv("controls_metrics.csv", index=False)
pd.set_option("display.width", 260); print(t.round(3).to_string(index=False))
