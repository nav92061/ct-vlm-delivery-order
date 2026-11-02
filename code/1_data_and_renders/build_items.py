"""Build renders + prompt items for every extracted pair. Deterministic: defect magnitudes
seeded by pair index (seed 2026 + i). Idempotent."""
import os, sys, json, numpy as np, pandas as pd
sys.path.insert(0, os.getcwd())
import study_render as R

main = pd.read_csv("main_pairs.csv")
os.makedirs("study_renders", exist_ok=True)
VARIANTS = ["clean", "truncation", "gap", "zshift"]
items, cases = [], []
for i, r in main.iterrows():
    f = f"planes/{r.pair_id}.npz"
    if not os.path.exists(f): continue
    d = np.load(f)
    xmin = max(float(d["xa0"]), float(d["xb0"]))
    xmax = min(float(d["xa0"]) + float(d["dxa"]) * d["ca"].shape[1], float(d["xb0"]) + float(d["dxb"]) * d["cb"].shape[1])
    ca = R.resample_x(d["ca"], float(d["xa0"]), float(d["dxa"]), xmin, xmax)
    cb = R.resample_x(d["cb"], float(d["xb0"]), float(d["dxb"]), xmin, xmax)
    ta, tb = float(d["ta"]), float(d["tb"])
    P = R.sample_defects(np.random.default_rng(2026 + i))
    V = {v: R.apply(ca, d["za"], cb, d["zb"], v, P.get(v)) for v in VARIANTS}
    H = {v: R.header_text(V[v][1], V[v][3], ta, tb) for v in VARIANTS}
    for v in VARIANTS:
        a, za, b, zb = V[v]
        imgs = {}
        for st, fn in (("aligned", R.render_aligned), ("native", R.render_native)):
            path = f"study_renders/{r.pair_id}_{v}_{st}.jpg"
            if not os.path.exists(path): fn((a, za), (b, zb)).save(path, quality=90)
            imgs[st] = path
        defect = v != "clean"
        mag = P[v]["frac"] if defect else 0.0
        cases.append(dict(pair_id=r.pair_id, patient=int(r.patient), manu=r.manu, variant=v, defect=defect,
                          magnitude=mag, params=json.dumps(P.get(v)), header_true=H[v],
                          coverage_diff_mm=float(abs(za.min() - zb.min()) + abs(za.max() - zb.max()))))
        h_conf = H["clean"] if defect else H["truncation"]
        grid = [("header_only", "none", None, H[v]),
                ("image_only", "aligned", imgs["aligned"], None),
                ("image_only", "native", imgs["native"], None),
                ("image+correct_header", "aligned", imgs["aligned"], H[v]),
                ("image+conflict_header", "aligned", imgs["aligned"], h_conf),
                ("image+conflict_header", "native", imgs["native"], h_conf)]
        for cond, st, img, hdr in grid:
            items.append(dict(pair_id=r.pair_id, patient=int(r.patient), manu=r.manu, variant=v, defect=defect,
                              magnitude=mag, condition=cond, style=st, image=img,
                              header_kind=("none" if hdr is None else ("conflict" if "conflict" in cond else "true")),
                              prompt=R.prompt(st, hdr)))
json.dump(items, open("study_items.json", "w"))
pd.DataFrame(cases).to_csv("study_cases.csv", index=False)
print(len(cases) // 4, "pairs", len(items), "items")
