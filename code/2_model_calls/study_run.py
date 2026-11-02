"""Main-grid runner (protocol: study_protocol.md). Resumable; one JSONL line per call.
Usage inside a kernel:  import study_run as SR; SR.run(key, items, models, out, workers)"""
import json, os, re, time, base64, datetime, threading, requests
from concurrent.futures import ThreadPoolExecutor

URL = "https://integrate.api.nvidia.com/v1/chat/completions"
MODELS = {  # model -> max_tokens (temperature fixed by caller)
    "meta/llama-3.2-11b-vision-instruct": 400,
    "meta/llama-3.2-90b-vision-instruct": 400,
    "nvidia/nemotron-3.5-super-vl-120b-a12b": 8192,
    "meta/muse-glimmer-30b": 6144,
    "google/gemma-4-31b-it": 4096,
}
DEC = re.compile(r"DECISION:?\s*\**\s*:?\s*(ADMIT|REFUSE)", re.I)
_lock = threading.Lock()
_b64 = {}

def b64(p):
    if p not in _b64: _b64[p] = base64.b64encode(open(p, "rb").read()).decode()
    return _b64[p]

def key_of(d):
    return (d["model"], d["pair_id"], d["variant"], d["condition"], d["style"], d.get("sample", 0))

def call(key, model, it, out, temperature=0.0, sample=0, timeout=300):
    content = [{"type": "text", "text": it["prompt"]}]
    if it["image"]:
        content.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64(it["image"])}})
    body = {"model": model, "messages": [{"role": "user", "content": content}],
            "max_tokens": MODELS[model], "temperature": temperature}
    rec = {k: it[k] for k in ("pair_id", "patient", "manu", "variant", "defect", "magnitude", "condition", "style", "header_kind")}
    rec.update(model=model, max_tokens=MODELS[model], temperature=temperature, sample=sample)
    for attempt in range(8):
        t = time.time()
        try:
            r = requests.post(URL, headers={"Authorization": f"Bearer {key}", "Accept": "application/json"}, json=body, timeout=timeout)
            rec.update(latency_s=round(time.time() - t, 2), http=r.status_code, attempt=attempt)
            if r.status_code == 200:
                j = r.json(); m = j["choices"][0]["message"]; txt = m.get("content") or ""
                hits = DEC.findall(txt)
                rec.update(text=txt[:2000], reasoning_chars=len(m.get("reasoning_content") or ""),
                           finish=j["choices"][0].get("finish_reason"), usage=j.get("usage"),
                           decision=hits[-1].upper() if hits else "UNPARSED")
                break
            if r.status_code in (408, 429, 500, 502, 503, 504):
                time.sleep(min(60, 4 * (attempt + 1))); continue
            rec.update(text=r.text[:300], decision="HTTP_ERROR"); break
        except Exception as e:
            rec.update(text=type(e).__name__, decision="EXCEPTION", latency_s=round(time.time() - t, 2)); time.sleep(5)
    rec["utc"] = datetime.datetime.utcnow().isoformat()
    with _lock:
        with open(out, "a") as f: f.write(json.dumps(rec) + "\n")
    return rec

def done_keys(out):
    s = set()
    if os.path.exists(out):
        for l in open(out):
            try: d = json.loads(l)
            except json.JSONDecodeError: continue
            if d.get("decision") in ("ADMIT", "REFUSE", "UNPARSED"): s.add(key_of(d))
    return s

def run_parallel(key, items, workers_per_model, out, temperature=0.0, samples=(0,)):
    """One thread pool per model, all models concurrently (slow models don't block fast ones)."""
    done = done_keys(out)
    PRI = {("image_only", "native"): 0, ("image+conflict_header", "aligned"): 1, ("image_only", "aligned"): 2,
           ("header_only", "none"): 3, ("image+correct_header", "aligned"): 4, ("image+conflict_header", "native"): 5}
    items = sorted(items, key=lambda it: (PRI.get((it["condition"], it["style"]), 9), it["pair_id"], it["variant"]))
    def one_model(m):
        jobs = [(it, s) for s in samples for it in items
                if (m, it["pair_id"], it["variant"], it["condition"], it["style"], s) not in done]
        with ThreadPoolExecutor(workers_per_model[m]) as ex:
            list(ex.map(lambda j: call(key, m, j[0], out, temperature, j[1]), jobs))
        return m, len(jobs)
    with ThreadPoolExecutor(len(workers_per_model)) as ex:
        return list(ex.map(one_model, list(workers_per_model)))

def run(key, items, models, out, workers=6, temperature=0.0, samples=(0,)):
    done = done_keys(out)
    jobs = [(m, it, s) for m in models for s in samples for it in items
            if (m, it["pair_id"], it["variant"], it["condition"], it["style"], s) not in done]
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(lambda j: call(key, j[0], j[1], out, temperature, j[2]), jobs))
    return len(jobs)
