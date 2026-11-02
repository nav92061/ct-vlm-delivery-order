"""Run MedGemma on study items. Usage: python medgemma_infer.py items.json renders.tar out/results.jsonl [limit]
Greedy decoding (do_sample=False), as on the model card. Resumable; one JSONL line per item."""
import os, sys, json, time, re, tarfile, datetime, glob
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

items_p, tar_p, out_p = sys.argv[1:4]
limit = int(sys.argv[4]) if len(sys.argv) > 4 else None
os.makedirs(os.path.dirname(out_p), exist_ok=True)
with tarfile.open(tar_p) as t: t.extractall(".")
items = json.load(open(items_p))[:limit]
REPO = "google/medgemma-1.5-4b-it"
snap = glob.glob(f"/opt/hf/hub/models--google--medgemma-1.5-4b-it/snapshots/*")[0]
proc = AutoProcessor.from_pretrained(snap)
model = AutoModelForImageTextToText.from_pretrained(snap, torch_dtype=torch.bfloat16, device_map="cuda")
model.eval()
DEC = re.compile(r"DECISION:?\s*\**\s*:?\s*(ADMIT|REFUSE)", re.I)
done = set()
if os.path.exists(out_p):
    for l in open(out_p):
        try: d = json.loads(l); done.add((d["pair_id"], d["variant"], d["condition"], d["style"]))
        except Exception: pass
print("device", torch.cuda.get_device_name(), "items", len(items), "done", len(done), flush=True)
t0 = time.time(); n = 0
with open(out_p, "a") as f:
    for it in items:
        k = (it["pair_id"], it["variant"], it["condition"], it["style"])
        if k in done: continue
        content = [{"type": "text", "text": it["prompt"]}]
        if it["image"]:
            content.append({"type": "image", "image": Image.open(it["image"]).convert("RGB")})
        msgs = [{"role": "user", "content": content}]
        inp = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                       return_dict=True, return_tensors="pt").to(model.device, dtype=torch.bfloat16)
        L = inp["input_ids"].shape[-1]; t = time.time()
        with torch.inference_mode():
            g = model.generate(**inp, max_new_tokens=400, do_sample=False)
        txt = proc.decode(g[0][L:], skip_special_tokens=True)
        hits = DEC.findall(txt)
        rec = {kk: it[kk] for kk in ("pair_id", "patient", "manu", "variant", "defect", "magnitude", "condition", "style", "header_kind")}
        rec.update(model=REPO, snapshot=os.path.basename(snap), max_tokens=400, temperature=0.0, sample=0,
                   latency_s=round(time.time() - t, 2), http=200, text=txt[:2000], prompt_tokens=int(L),
                   completion_tokens=int(g.shape[-1] - L), finish="length" if g.shape[-1] - L >= 400 else "stop",
                   decision=hits[-1].upper() if hits else "UNPARSED", utc=datetime.datetime.utcnow().isoformat())
        f.write(json.dumps(rec) + "\n"); f.flush(); n += 1
        if n % 50 == 0: print(n, round(time.time() - t0), flush=True)
print("DONE", n, round(time.time() - t0), flush=True)
