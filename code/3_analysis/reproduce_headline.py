#!/usr/bin/env python3
"""Reproduce the paper's headline order effect (Table 1, "Tog." vs "After") from the raw call logs.

For each model and view: among defects the model REFUSED from the image alone, the share it ADMITTED when a false
"complete" header was given (a) together with the image in one prompt and (b) in a second turn after its own answer.
Needs only pandas. Run from the repository root:  python code/3_analysis/reproduce_headline.py
"""
import json, os
import pandas as pd

LOGS = "logs"
K = ["pair_id", "variant", "style"]
NEM = "nvidia/nemotron-3.5-super-vl-120b-a12b"

def load(path):
    rows = []
    for line in open(os.path.join(LOGS, path)):
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d.get("decision") not in ("ADMIT", "REFUSE"):          # unparsed answers are excluded
            continue
        if d.get("set", "synthetic") != "synthetic":                # main grid only (not the 24 real pairs)
            continue
        if d.get("sample", 0) not in (0, None) or d.get("temperature") not in (0, 0.0, None):
            continue                                                # first, deterministic draw only
        rows.append(d)
    df = pd.DataFrame(rows).sort_values("utc").drop_duplicates(["model"] + K + ["condition"], keep="last")
    df["defect"] = df.variant != "clean"
    return df

study, two = load("study_results.jsonl"), load("two_stage_results.jsonl")
haiku, sonnet = load("results_anthropic_claude-haiku-5-5.jsonl"), load("results_anthropic_claude-sonnet-5-5.jsonl")
sources = {"Nemotron-3.5-VL": (study[study.model == NEM], two[two.model == NEM]),
           "Claude Haiku 5.5": (haiku, haiku), "Claude Sonnet 5.5": (sonnet, sonnet)}

out = []
for name, (one_src, two_src) in sources.items():
    detected = one_src[(one_src.condition == "image_only") & one_src.defect & (one_src.decision == "REFUSE")][K]
    together = one_src[(one_src.condition == "image+conflict_header") & one_src.defect][K + ["decision"]]
    after = two_src[(two_src.condition == "image+conflict_header+two_stage") & two_src.defect][K + ["decision"]]
    x = detected.merge(together.rename(columns={"decision": "together"}), on=K).merge(after.rename(columns={"decision": "after"}), on=K)
    for view, g in x.groupby("style"):
        out.append(dict(model=name, view=view, n=len(g),
                        together_pct=round(100 * (g.together == "ADMIT").mean(), 1),
                        after_pct=round(100 * (g.after == "ADMIT").mean(), 1)))
print(pd.DataFrame(out).to_string(index=False))
