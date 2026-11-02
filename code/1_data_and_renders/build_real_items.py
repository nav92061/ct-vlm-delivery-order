"""Addendum A2: renders + prompts for naturally mismatched NLST pairs (no synthetic edits)."""
import os, sys, json, numpy as np, pandas as pd
sys.path.insert(0, os.getcwd())
import study_render as R
MIT = ("Important: the DICOM header summary below may be wrong or corrupted. Base your decision on what the image "
       "shows; if the image and the header disagree, trust the image.\n\nDICOM header summary:")
real = pd.read_csv("real_pairs.csv"); os.makedirs("real_renders", exist_ok=True)
items, cases = [], []
for _, r in real.iterrows():
    f = f"planes_real/{r.pair_id}.npz"
    if not os.path.exists(f): continue
    d = np.load(f)
    xmin = max(float(d["xa0"]), float(d["xb0"]))
    xmax = min(float(d["xa0"]) + float(d["dxa"]) * d["ca"].shape[1], float(d["xb0"]) + float(d["dxb"]) * d["cb"].shape[1])
    ca = R.resample_x(d["ca"], float(d["xa0"]), float(d["dxa"]), xmin, xmax)
    cb = R.resample_x(d["cb"], float(d["xb0"]), float(d["dxb"]), xmin, xmax)
    za, zb, ta, tb = d["za"], d["zb"], float(d["ta"]), float(d["tb"])
    h_true = R.header_text(za, zb, ta, tb)
    nb_fake = int(round((za.max() - za.min()) / np.median(np.diff(np.sort(zb))))) + 1
    zb_fake = np.linspace(za.min(), za.max(), nb_fake)
    h_false = R.header_text(za, zb_fake, ta, tb)          # B reported with A's z-range, consistent slice count, 0 gaps
    imgs = {}
    for st, fn in (("aligned", R.render_aligned), ("native", R.render_native)):
        p = f"real_renders/{r.pair_id}_real_{st}.jpg"
        if not os.path.exists(p): fn((ca, za), (cb, zb)).save(p, quality=90)
        imgs[st] = p
    mag = float(1 - r.z_iou)
    cases.append(dict(pair_id=r.pair_id, patient=int(r.patient), manu=r.manu, z_iou=float(r.z_iou), magnitude=mag,
                      same_FoR=bool(r.same_FoR), ta=ta, tb=tb, header_true=h_true, header_false=h_false))
    grid = [("image_only", "aligned", None), ("image_only", "native", None),
            ("image+correct_header", "aligned", h_true),
            ("image+conflict_header", "aligned", h_false), ("image+conflict_header", "native", h_false)]
    for cond, st, hdr in grid:
        base = dict(pair_id=r.pair_id, patient=int(r.patient), manu=r.manu, variant="real", defect=True, magnitude=mag,
                    condition=cond, style=st, image=imgs[st],
                    header_kind="none" if hdr is None else ("conflict" if "conflict" in cond else "true"),
                    prompt=R.prompt(st, hdr))
        items.append(base)
        if cond == "image+conflict_header":
            items.append(dict(base, condition=cond + "+mitigation", prompt=base["prompt"].replace("DICOM header summary:", MIT, 1)))
assert all("trust the image" in i["prompt"] for i in items if i["condition"].endswith("mitigation"))
json.dump(items, open("real_items.json", "w")); pd.DataFrame(cases).to_csv("real_cases.csv", index=False)
print(len(cases), "pairs", len(items), "items")
