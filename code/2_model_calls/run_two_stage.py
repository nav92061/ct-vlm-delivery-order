"""Addendum A4 runner: two-turn conversation; turn 1 = logged image-only answer reused verbatim. Key in env K. Resumable."""
import os, sys, json, re, time, datetime, threading, requests
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study_run as SR
from concurrent.futures import ThreadPoolExecutor

OUT = "two_stage_results.jsonl"; K = os.environ.get("K", "")
FORMAT = ("Reply with exactly one line starting 'DECISION: ADMIT' or 'DECISION: REFUSE', followed by one short sentence "
          "giving the reason.")
HDR_RE = re.compile(r"DICOM header summary: (.*?)\n\n", re.S)

def build_jobs():
    items = json.load(open("study_items.json"))
    by = {(i["pair_id"], i["variant"], i["condition"], i["style"]): i for i in items}
    logged = {}
    for l in open("study_results.jsonl"):
        try: r = json.loads(l)
        except json.JSONDecodeError: continue
        if r.get("temperature", 0) != 0 or r.get("sample", 0) != 0: continue
        if r.get("condition") == "image_only" and r.get("decision") in ("ADMIT", "REFUSE", "UNPARSED"):
            k = (r["model"], r["pair_id"], r["variant"], r["style"])
            if k not in logged or r["utc"] > logged[k]["utc"]: logged[k] = r
    plan = {"nvidia/nemotron-3.5-super-vl-120b-a12b": None, "google/gemma-4-31b-it": 15}
    jobs, skipped = [], 0
    for m, lim in plan.items():
        for (pid, var, cond, st), it in by.items():
            if cond != "image+conflict_header": continue
            if lim is not None and int(pid[1:]) >= lim: continue
            s1 = logged.get((m, pid, var, st)); io = by[(pid, var, "image_only", st)]
            if s1 is None or s1["decision"] == "UNPARSED": skipped += 1; continue
            hdr = HDR_RE.search(it["prompt"]).group(1).strip()
            jobs.append(dict(model=m, item=it, img_item=io, stage1_text=s1["text"], stage1_decision=s1["decision"],
                             turn2=f"Here is the DICOM header summary for the same pair: {hdr}\n\nGive your final decision. {FORMAT}"))
    return jobs, skipped

def call2(m, job, timeout=300):
    io, it = job["img_item"], job["item"]
    msgs = [{"role": "user", "content": [{"type": "text", "text": io["prompt"]},
                                         {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + SR.b64(io["image"])}}]},
            {"role": "assistant", "content": job["stage1_text"]},
            {"role": "user", "content": job["turn2"]}]
    body = {"model": m, "messages": msgs, "max_tokens": SR.MODELS[m], "temperature": 0.0}
    rec = {k: it[k] for k in ("pair_id", "patient", "manu", "variant", "defect", "magnitude", "style", "header_kind")}
    rec.update(condition="image+conflict_header+two_stage", model=m, max_tokens=SR.MODELS[m], temperature=0.0, sample=0,
               stage1_decision=job["stage1_decision"])
    for attempt in range(8):
        t = time.time()
        try:
            r = requests.post(SR.URL, headers={"Authorization": f"Bearer {K}", "Accept": "application/json"}, json=body, timeout=timeout)
            rec.update(latency_s=round(time.time() - t, 2), http=r.status_code, attempt=attempt)
            if r.status_code == 200:
                j = r.json(); msg = j["choices"][0]["message"]; txt = msg.get("content") or ""
                hits = SR.DEC.findall(txt)
                rec.update(text=txt[:2000], finish=j["choices"][0].get("finish_reason"), usage=j.get("usage"),
                           decision=hits[-1].upper() if hits else "UNPARSED"); break
            if r.status_code in (408, 429, 500, 502, 503, 504): time.sleep(min(60, 4 * (attempt + 1))); continue
            rec.update(text=r.text[:300], decision="HTTP_ERROR"); break
        except Exception as e:
            rec.update(text=type(e).__name__, decision="EXCEPTION", latency_s=round(time.time() - t, 2)); time.sleep(5)
    rec["utc"] = datetime.datetime.utcnow().isoformat()
    with SR._lock:
        with open(OUT, "a") as f: f.write(json.dumps(rec) + "\n")
    return rec

if __name__ == "__main__":
    jobs, skipped = build_jobs()
    if "--plan" in sys.argv:
        import collections
        print(len(jobs), "jobs; skipped (stage-1 unparsed or missing):", skipped,
              collections.Counter((j["model"].split("/")[1][:10], j["item"]["style"]) for j in jobs)); sys.exit()
    done = SR.done_keys(OUT)
    todo = [j for j in jobs if (j["model"], j["item"]["pair_id"], j["item"]["variant"], "image+conflict_header+two_stage",
                                j["item"]["style"], 0) not in done]
    if "--smoke" in sys.argv: todo = [j for j in todo if j["model"].startswith("nvidia")][:4]
    W = {"nvidia/nemotron-3.5-super-vl-120b-a12b": 8, "google/gemma-4-31b-it": 12}
    def go(m):
        js = [j for j in todo if j["model"] == m]
        with ThreadPoolExecutor(W[m]) as ex: list(ex.map(lambda j: call2(m, j), js))
        print("done", m, len(js), flush=True)
    with ThreadPoolExecutor(2) as ex: list(ex.map(go, list(W)))
    print("DONE", flush=True)
