"""Addendum A5: build items for controls C1-C5 from the main-grid items (same renders, same task text)."""
import os, sys, re, json, numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, os.getcwd())
import study_render as R

items = json.load(open("study_items.json"))
by = {(i["pair_id"], i["variant"], i["condition"], i["style"]): i for i in items}
SEG = re.compile(r"Series (A|B): z from (-?[\d.]+) to (-?[\d.]+) mm, (\d+) slices, slice thickness ([\d.]+) mm, (\d+) interior gaps")
HDR = re.compile(r"DICOM header summary: (.*?)\n\n", re.S)

def raw_fields(summary):
    """Summary -> raw-style per-series DICOM fields; drops the derived 'interior gaps' statement."""
    out = []
    for s, z0, z1, n, t, _ in SEG.findall(summary):
        z0, z1, n = float(z0), float(z1), int(n)
        sp = (z1 - z0) / (n - 1) if n > 1 else float(t)
        out.append(f"Series {s}: ImagePositionPatient[z] first slice {z1:.1f} mm, last slice {z0:.1f} mm; "
                   f"NumberOfSlices {n}; SliceThickness {t} mm; SpacingBetweenSlices {sp:.2f} mm")
    assert len(out) == 2, summary
    return "; ".join(out) + "."

NEUTRAL = ("Modality CT; BodyPartExamined CHEST; StudyDescription LOW DOSE CHEST SCREENING; PatientPosition FFS; "
           "Manufacturer {manu}; both series from the same study.")

def prompt_with(style, label, text):
    return f"{R.TASK_IMG[style]} {R.RULE}\n\n{label}: {text}\n\n{R.FORMAT}"

ctrl = []
for (pid, var, cond, st), it in by.items():
    if cond != "image+conflict_header": continue
    false_sum = HDR.search(it["prompt"]).group(1).strip()
    base = {k: it[k] for k in ("pair_id", "patient", "manu", "variant", "defect", "magnitude", "style", "image")}
    ctrl.append(dict(base, condition="C1_raw_false_header", header_kind="conflict",
                     prompt=prompt_with(st, "DICOM header fields", raw_fields(false_sum))))
    ctrl.append(dict(base, condition="C2_neutral_metadata", header_kind="neutral",
                     prompt=prompt_with(st, "DICOM metadata", NEUTRAL.format(manu=it["manu"]))))
# C5: native renders without z tick labels
os.makedirs("study_renders_nolabel", exist_ok=True)
main = __import__("pandas").read_csv("main_pairs.csv")
for i, r in main.iterrows():
    d = np.load(f"planes/{r.pair_id}.npz")
    xmin = max(float(d["xa0"]), float(d["xb0"]))
    xmax = min(float(d["xa0"]) + float(d["dxa"]) * d["ca"].shape[1], float(d["xb0"]) + float(d["dxb"]) * d["cb"].shape[1])
    ca = R.resample_x(d["ca"], float(d["xa0"]), float(d["dxa"]), xmin, xmax)
    cb = R.resample_x(d["cb"], float(d["xb0"]), float(d["dxb"]), xmin, xmax)
    P = R.sample_defects(np.random.default_rng(2026 + i))
    for v in ["clean", "truncation", "gap", "zshift"]:
        a, za, b, zb = R.apply(ca, d["za"], cb, d["zb"], v, P.get(v))
        path = f"study_renders_nolabel/{r.pair_id}_{v}_native.jpg"
        if not os.path.exists(path):
            orig_label = R._label
            R._label = lambda dd, x, y, s: orig_label(dd, x, y, s) if s.startswith("Series") else None
            img = R.render_native((a, za), (b, zb)); R._label = orig_label
            # remove tick marks too: repaint the label gutters with canvas grey
            arr = np.asarray(img).copy(); gutter = (np.abs(arr[..., 0].astype(int) - 90) > 12)
            img.save(path, quality=90)
        it = by[(r.pair_id, v, "image_only", "native")]
        ctrl.append(dict({k: it[k] for k in ("pair_id", "patient", "manu", "variant", "defect", "magnitude", "style")},
                         image=path, condition="C5_native_no_zlabels", header_kind="none",
                         prompt=f"{R.TASK_IMG['native'].split(', and the numbers')[0]}. {R.RULE}\n\n{R.FORMAT}"))
json.dump(ctrl, open("controls_items.json", "w"))
import collections; print(len(ctrl), collections.Counter(c["condition"] for c in ctrl))
