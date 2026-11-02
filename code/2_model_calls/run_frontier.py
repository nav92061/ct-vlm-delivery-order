#!/usr/bin/env python3
"""Run the CT image-metadata conflict study on a commercial VLM (OpenAI, Anthropic or Google).

Only dependency: requests  (pip install requests)
Key: environment variable OPENAI_API_KEY, ANTHROPIC_API_KEY or GEMINI_API_KEY.

  python run_frontier.py --provider openai    --model <model-id> --tier 1 --limit 20      # test first
  python run_frontier.py --provider openai    --model <model-id> --tier 1                 # full tier 1
  python run_frontier.py --provider anthropic --model <model-id> --tier 1
  python run_frontier.py --provider google    --model <model-id> --tier 1
  python run_frontier.py --provider mock      --model mock --limit 30                     # offline self-test, no key

Resumable: re-run the same command after any interruption; finished jobs are skipped.
Output: results_<provider>_<model>.jsonl (send this file back).
"""
import argparse, base64, json, os, random, re, sys, threading, time, datetime
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
DEC = re.compile(r"DECISION:?\s*\**\s*:?\s*(ADMIT|REFUSE)", re.I)
LOCK = threading.Lock()

def b64(path):
    with open(os.path.join(HERE, path), "rb") as f: return base64.b64encode(f.read()).decode()

# ---------------------------------------------------------------- providers
def call_openai(model, turns, img, a):
    import requests
    content = [{"type": "text", "text": turns[0]}]
    if img: content.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64(img)}})
    msgs = [{"role": "user", "content": content}]
    if len(turns) == 3: msgs += [{"role": "assistant", "content": turns[1]}, {"role": "user", "content": turns[2]}]
    body = {"model": model, "messages": msgs, "max_completion_tokens": a.max_tokens}
    if a.temperature is not None: body["temperature"] = a.temperature
    if a.reasoning_effort: body["reasoning_effort"] = a.reasoning_effort
    r = requests.post("https://api.openai.com/v1/chat/completions", timeout=a.timeout, json=body,
                      headers={"Authorization": "Bearer " + os.environ["OPENAI_API_KEY"]})
    if r.status_code != 200: return r.status_code, r.text[:400], None, None
    j = r.json(); u = j.get("usage", {})
    return 200, j["choices"][0]["message"].get("content") or "", j["choices"][0].get("finish_reason"), \
        dict(input=u.get("prompt_tokens"), output=u.get("completion_tokens"), model_returned=j.get("model"))

def call_anthropic(model, turns, img, a):
    import requests
    t2img = isinstance(img, tuple)
    if t2img: img = img[1]
    content = [{"type": "text", "text": turns[0]}]
    if img and not t2img: content.insert(0, {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64(img)}})
    msgs = [{"role": "user", "content": content}]
    if len(turns) == 3:
        last = [{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64(img)}}, {"type": "text", "text": turns[2]}] if t2img else turns[2]
        msgs += [{"role": "assistant", "content": turns[1]}, {"role": "user", "content": last}]
    body = {"model": model, "messages": msgs, "max_tokens": a.max_tokens}
    if len(turns) == 4:
        msgs += [{"role": "assistant", "content": [{"type": "text", "text": turns[1]},
                  {"type": "tool_use", "id": "toolu_hdr1", "name": "read_dicom_header", "input": {"series": "A,B"}}]},
                 {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_hdr1", "content": turns[2]},
                                              {"type": "text", "text": turns[3]}]}]
        body["tools"] = [TOOL_ANTH]; body["tool_choice"] = {"type": "none"}
    if a.temperature is not None: body["temperature"] = a.temperature
    if a.no_thinking: body["thinking"] = {"type": a.thinking_off_type}
    if a.effort: body["output_config"] = {"effort": a.effort}
    r = requests.post("https://api.anthropic.com/v1/messages", timeout=a.timeout, json=body,
                      headers={"x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01"})
    if r.status_code != 200: return r.status_code, r.text[:400], None, None
    j = r.json(); u = j.get("usage", {})
    txt = "".join(b.get("text", "") for b in j.get("content", []) if b.get("type") == "text")
    return 200, txt, j.get("stop_reason"), dict(input=u.get("input_tokens"), output=u.get("output_tokens"), model_returned=j.get("model"))

def call_google(model, turns, img, a):
    import requests
    parts = [{"text": turns[0]}]
    if img: parts.append({"inline_data": {"mime_type": "image/jpeg", "data": b64(img)}})
    contents = [{"role": "user", "parts": parts}]
    if len(turns) == 3: contents += [{"role": "model", "parts": [{"text": turns[1]}]}, {"role": "user", "parts": [{"text": turns[2]}]}]
    cfg = {"maxOutputTokens": a.max_tokens}
    if a.temperature is not None: cfg["temperature"] = a.temperature
    r = requests.post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent", timeout=a.timeout,
                      json={"contents": contents, "generationConfig": cfg},
                      headers={"x-goog-api-key": os.environ["GEMINI_API_KEY"]})
    if r.status_code != 200: return r.status_code, r.text[:400], None, None
    j = r.json(); u = j.get("usageMetadata", {}); c = (j.get("candidates") or [{}])[0]
    txt = "".join(p.get("text", "") for p in c.get("content", {}).get("parts", []) if not p.get("thought"))
    return 200, txt, c.get("finishReason"), dict(input=u.get("promptTokenCount"),
                                                  output=(u.get("candidatesTokenCount") or 0) + (u.get("thoughtsTokenCount") or 0),
                                                  model_returned=j.get("modelVersion"))

def call_mock(model, turns, img, a):
    time.sleep(0.01); d = random.choice(["ADMIT", "REFUSE"])
    return 200, f"DECISION: {d}\nMock reason.", "stop", dict(input=500, output=20, model_returned="mock")


TOOL_DESC = "Return the DICOM header summary (z-range, slice count, thickness, interior gaps) of Series A and B."
TOOL_ANTH = {"name": "read_dicom_header", "description": TOOL_DESC,
             "input_schema": {"type": "object", "properties": {"series": {"type": "string"}}, "required": ["series"]}}
TOOL_OAI = {"type": "function", "function": {"name": "read_dicom_header", "description": TOOL_DESC,
            "parameters": {"type": "object", "properties": {"series": {"type": "string"}}, "required": ["series"]}}}

def call_nvidia(model, turns, img, a):
    import requests
    t2img = isinstance(img, tuple)
    if t2img: img = img[1]
    content = [{"type": "text", "text": turns[0]}]
    if img and not t2img: content.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64(img)}})
    msgs = [{"role": "user", "content": content if (img and not t2img) else turns[0]}]
    body = {"model": model, "max_tokens": a.max_tokens}
    if len(turns) == 3:
        last = [{"type": "text", "text": turns[2]}, {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64(img)}}] if t2img else turns[2]
        msgs += [{"role": "assistant", "content": turns[1]}, {"role": "user", "content": last}]
    if len(turns) == 4:
        msgs += [{"role": "assistant", "content": turns[1], "tool_calls": [{"id": "call_hdr1", "type": "function",
                  "function": {"name": "read_dicom_header", "arguments": "{\"series\": \"A,B\"}"}}]},
                 {"role": "tool", "tool_call_id": "call_hdr1", "content": turns[2]},
                 {"role": "user", "content": turns[3]}]
        body["tools"] = [TOOL_OAI]; body["tool_choice"] = "none"
    body["messages"] = msgs
    if a.temperature is not None: body["temperature"] = a.temperature
    r = requests.post("https://integrate.api.nvidia.com/v1/chat/completions", timeout=a.timeout, json=body,
                      headers={"Authorization": "Bearer " + os.environ["NVIDIA_API_KEY"], "Accept": "application/json"})
    if r.status_code != 200: return r.status_code, r.text[:400], None, None
    j = r.json(); u = j.get("usage") or {}; m = j["choices"][0]["message"]
    return 200, m.get("content") or "", j["choices"][0].get("finish_reason"), \
        dict(input=u.get("prompt_tokens"), output=u.get("completion_tokens"), model_returned=j.get("model"))

CALL = {"openai": call_openai, "anthropic": call_anthropic, "google": call_google, "mock": call_mock, "nvidia": call_nvidia}
KEYVAR = {"nvidia": "NVIDIA_API_KEY", "openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "google": "GEMINI_API_KEY"}

# ---------------------------------------------------------------- run
def run_one(job, turns, a, out, stats):
    rec = {k: job[k] for k in ("job_id", "pair_id", "patient", "manu", "variant", "defect", "magnitude", "condition",
                               "style", "header_kind", "set", "tier", "kind")}
    rec.update(provider=a.provider, model=a.model, max_tokens=a.max_tokens, temperature=a.temperature,
               reasoning_effort=a.reasoning_effort, no_thinking=a.no_thinking, thinking_off_type=a.thinking_off_type, effort=a.effort)
    if job["kind"] == "two_turn": rec["stage1_text"] = turns[1]
    for attempt in range(10):
        t = time.time()
        try:
            code, txt, fin, use = CALL[a.provider](a.model, turns, ("T2", job["image"]) if job.get("image_turn") == 2 else job["image"], a)
        except Exception as e:
            code, txt, fin, use = -1, f"{type(e).__name__}: {e}"[:300], None, None
        rec.update(http=code, latency_s=round(time.time() - t, 2), attempt=attempt)
        if code == 200:
            hits = DEC.findall(txt or "")
            rec.update(text=(txt or "")[:3000], finish=fin, usage=use, decision=hits[-1].upper() if hits else "UNPARSED")
            break
        if code in (-1, 408, 409, 429, 500, 502, 503, 504, 529):
            time.sleep(min(90, 5 * 2 ** min(attempt, 4)) * (0.5 + random.random())); continue
        rec.update(text=txt, decision="HTTP_ERROR"); break
    else:
        rec.update(decision="GAVE_UP")
    rec["utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with LOCK:
        with open(out, "a") as f: f.write(json.dumps(rec) + "\n")
        u = rec.get("usage") or {}
        stats["n"] += 1; stats["in"] += u.get("input") or 0; stats["out"] += u.get("output") or 0
        stats[rec["decision"]] = stats.get(rec["decision"], 0) + 1
        cost = (stats["in"] * a.price_in + stats["out"] * a.price_out) / 1e6
        if stats["n"] % 25 == 0 or rec["decision"] in ("HTTP_ERROR", "GAVE_UP"):
            print(f"{stats['n']} done | tokens in {stats['in']} out {stats['out']} | est. ${cost:.2f} | "
                  + " ".join(f"{k}={v}" for k, v in stats.items() if k.isupper()), flush=True)
            if rec["decision"] == "HTTP_ERROR": print("  error:", rec["text"][:200], flush=True)
        if a.max_cost and cost > a.max_cost: stats["stop"] = True
    return rec

def load(out):
    res = {}
    if os.path.exists(out):
        for l in open(out):
            try: d = json.loads(l)
            except json.JSONDecodeError: continue
            if d.get("decision") in ("ADMIT", "REFUSE", "UNPARSED"): res[d["job_id"]] = d
    return res

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--provider", required=True, choices=list(CALL))
    p.add_argument("--model", required=True)
    p.add_argument("--tier", type=int, default=1, choices=[1, 2], help="1 = core (~2,100 calls); 2 = core + controls (~4,250)")
    p.add_argument("--limit", type=int, default=0, help="run at most N single-turn jobs (test runs)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max-tokens", dest="max_tokens", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=0.0, help="use --no-temperature for models that reject it")
    p.add_argument("--no-temperature", dest="temperature", action="store_const", const=None)
    p.add_argument("--reasoning-effort", dest="reasoning_effort", default=None, help="OpenAI reasoning models only, e.g. low")
    p.add_argument("--timeout", type=int, default=300)
    p.add_argument("--no-thinking", dest="no_thinking", action="store_true", help="Anthropic: disable extended thinking (one-line answers, lowest cost)")
    p.add_argument("--stage1-file", dest="stage1_file", default="", help="JSONL of turn-1 answers to reuse")
    p.add_argument("--conditions", default="", help="comma-separated subset of conditions (image_only kept for two-turn stage 1)")
    p.add_argument("--effort", default=None, help="Anthropic output effort, e.g. low")
    p.add_argument("--thinking-off-type", dest="thinking_off_type", default="disabled", help="value that turns thinking off; some models require between_tools")
    p.add_argument("--price-in", dest="price_in", type=float, default=0.0, help="$ per million input tokens (for cost display)")
    p.add_argument("--price-out", dest="price_out", type=float, default=0.0, help="$ per million output tokens")
    p.add_argument("--max-cost", dest="max_cost", type=float, default=0.0, help="stop when estimated $ exceeds this")
    a = p.parse_args()
    if a.provider != "mock" and not os.environ.get(KEYVAR[a.provider]):
        sys.exit(f"Set the {KEYVAR[a.provider]} environment variable first.")
    if a.max_cost and not (a.price_in or a.price_out): sys.exit("--max-cost needs --price-in and --price-out.")
    jobs = [j for j in json.load(open(os.path.join(HERE, "jobs.json"))) if j["tier"] <= a.tier]
    if a.conditions:
        keep = set(a.conditions.split(","))
        jobs = [j for j in jobs if j["condition"] in keep or (j["kind"] == "single" and j["condition"] == "image_only")]
    out = os.path.join(HERE, f"results_{a.provider}_{re.sub(r'[^A-Za-z0-9._-]', '_', a.model)}.jsonl")
    stats = {"n": 0, "in": 0, "out": 0}
    # count tokens already billed in earlier (interrupted) runs of this same output file, so --max-cost is cumulative
    if os.path.exists(out):
        for l in open(out):
            try: d = json.loads(l)
            except json.JSONDecodeError: continue
            u = d.get("usage") or {}
            stats["in"] += u.get("input") or 0; stats["out"] += u.get("output") or 0
    prior = (stats["in"] * a.price_in + stats["out"] * a.price_out) / 1e6
    print(f"already spent on this model (from {os.path.basename(out)}): ${prior:.2f}", flush=True)
    if a.max_cost and prior >= a.max_cost: sys.exit("cost cap already reached; raise --max-cost to continue.")
    # phase 1: single-turn jobs
    done = load(out)
    single = [j for j in jobs if j["kind"] == "single" and j["job_id"] not in done]
    if a.stage1_file: single = [j for j in single if j["condition"] == "A9_false_header_only"]   # other turn-1 answers come from the file
    if a.limit: single = single[:a.limit]
    print(f"phase 1: {len(single)} single-turn jobs to run ({len(done)} already done) -> {out}", flush=True)
    def go1(j):
        if not stats.get("stop"): run_one(j, [j["prompt"]], a, out, stats)

    with ThreadPoolExecutor(a.workers) as ex: list(ex.map(go1, single))
    if (a.limit and single) or stats.get("stop"):
        print("stopped after test/limit or cost cap; re-run without --limit to continue." if not stats.get("stop") else "cost cap reached.")
        return
    # phase 2: two-turn jobs; turn 1 = this model's own image-only answer for the same case
    done = load(out)
    own = {(d["pair_id"], d["variant"], d["condition"], d["style"]): d for d in done.values() if d["condition"] in ("image_only", "A9_false_header_only")}
    if a.stage1_file:
        for l in open(a.stage1_file):
            d = json.loads(l); own[(d["pair_id"], d["variant"], d["condition"], d["style"])] = d
    two, skipped = [], 0
    for j in jobs:
        if j["kind"] != "two_turn" or j["job_id"] in done: continue
        s1 = own.get(tuple(j["stage1_ref"]))
        if s1 is None or s1["decision"] == "UNPARSED": skipped += 1; continue
        two.append((j, [j["prompt"], s1["text"], j["tool_result"], j["turn2"]] if j.get("tool_result") else [j["prompt"], s1["text"], j["turn2"]]))
    print(f"phase 2: {len(two)} two-turn jobs ({skipped} skipped: own image-only answer missing or unparsed)", flush=True)
    def go2(x):
        if not stats.get("stop"): run_one(x[0], x[1], a, out, stats)
    if a.limit: two = two[:a.limit]
    with ThreadPoolExecutor(a.workers) as ex: list(ex.map(go2, two))
    final = load(out); left = sum(1 for j in jobs if j["job_id"] not in final)
    print(f"DONE. {len(final)} parsed or unparsed answers saved; {left} jobs without a usable answer "
          f"(re-run to retry, ~{skipped} are expected skips). Send back: {os.path.basename(out)}")

if __name__ == "__main__":
    main()
