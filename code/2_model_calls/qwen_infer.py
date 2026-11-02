"""Run Qwen3-VL-8B on frontier jobs.json. Usage: python qwen_infer.py jobs.json renders.tar out/results.jsonl [limit] [tier]
Greedy decoding. Phase 1 single-turn jobs; phase 2 two-turn jobs whose turn 1 is this model's own image-only answer.
Records match run_frontier.py so the same analysis applies. Resumable."""
import os, sys, json, time, re, tarfile, datetime, glob
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

jobs_p, tar_p, out_p = sys.argv[1:4]
limit = int(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] != "0" else None
tier = int(sys.argv[5]) if len(sys.argv) > 5 else 2
os.makedirs(os.path.dirname(out_p), exist_ok=True)
with tarfile.open(tar_p) as t: t.extractall(".")
jobs = [j for j in json.load(open(jobs_p)) if j["tier"] <= tier]
REPO = "Qwen/Qwen3-VL-8B-Instruct"
snap = glob.glob("/opt/hf/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/*")[0]
proc = AutoProcessor.from_pretrained(snap)
model = AutoModelForImageTextToText.from_pretrained(snap, torch_dtype=torch.bfloat16, device_map="cuda").eval()
DEC = re.compile(r"DECISION:?\s*\**\s*:?\s*(ADMIT|REFUSE)", re.I)
MAXTOK = 300

def load():
    res = {}
    if os.path.exists(out_p):
        for l in open(out_p):
            try: d = json.loads(l)
            except json.JSONDecodeError: continue
            res[d["job_id"]] = d
    return res

def gen(turns, image):
    first = [{"type": "image", "image": Image.open(image).convert("RGB")}] if image else []
    msgs = [{"role": "user", "content": first + [{"type": "text", "text": turns[0]}]}]
    if len(turns) == 3:
        msgs += [{"role": "assistant", "content": [{"type": "text", "text": turns[1]}]},
                 {"role": "user", "content": [{"type": "text", "text": turns[2]}]}]
    inp = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True,
                                   return_tensors="pt").to(model.device)
    L = inp["input_ids"].shape[-1]
    with torch.inference_mode():
        g = model.generate(**inp, max_new_tokens=MAXTOK, do_sample=False)
    n = int(g.shape[-1] - L)
    return proc.decode(g[0][L:], skip_special_tokens=True), int(L), n

def run(job, turns, f):
    t = time.time()
    txt, L, n = gen(turns, job["image"])
    hits = DEC.findall(txt)
    rec = {k: job[k] for k in ("job_id", "pair_id", "patient", "manu", "variant", "defect", "magnitude", "condition",
                               "style", "header_kind", "set", "tier", "kind")}
    rec.update(provider="modal", model=REPO, snapshot=os.path.basename(snap), max_tokens=MAXTOK, temperature=0.0,
               http=200, latency_s=round(time.time() - t, 2), text=txt[:3000], finish="length" if n >= MAXTOK else "stop",
               usage={"input": L, "output": n}, decision=hits[-1].upper() if hits else "UNPARSED",
               utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    if len(turns) == 3: rec["stage1_text"] = turns[1]
    f.write(json.dumps(rec) + "\n"); f.flush()

done = load()
single = [j for j in jobs if j["kind"] == "single" and j["job_id"] not in done][:limit]
print("device", torch.cuda.get_device_name(), "jobs", len(jobs), "done", len(done), "single todo", len(single), flush=True)
t0 = time.time()
with open(out_p, "a") as f:
    for i, j in enumerate(single):
        run(j, [j["prompt"]], f)
        if (i + 1) % 100 == 0: print("single", i + 1, round(time.time() - t0), flush=True)
    if limit is None:
        done = load()
        own = {(d["pair_id"], d["variant"], d["condition"], d["style"]): d for d in done.values() if d["condition"] == "image_only"}
        two = []
        for j in jobs:
            if j["kind"] != "two_turn" or j["job_id"] in done: continue
            s1 = own.get(tuple(j["stage1_ref"]))
            if s1 is None or s1["decision"] == "UNPARSED": continue
            two.append((j, [j["prompt"], s1["text"], j["turn2"]]))
        print("two-turn todo", len(two), flush=True)
        for i, (j, turns) in enumerate(two):
            run(j, turns, f)
            if (i + 1) % 100 == 0: print("two", i + 1, round(time.time() - t0), flush=True)
print("DONE", round(time.time() - t0), flush=True)
