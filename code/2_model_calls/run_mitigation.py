"""Mitigation addendum A1 runner. Key in env K. Resumable; stops if file STOP_MIT exists."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study_run as SR
from concurrent.futures import ThreadPoolExecutor
K = os.environ["K"]; OUT = "mitigation_results.jsonl"
items = json.load(open("mitigation_items.json"))
SUB = [i for i in items if int(i["pair_id"][1:]) < 15]
PLAN = {"nvidia/nemotron-3.5-super-vl-120b-a12b": (items, 8), "meta/llama-3.2-90b-vision-instruct": (items, 8),
        "meta/llama-3.2-11b-vision-instruct": (items, 3), "google/gemma-4-31b-it": (SUB, 10)}
def go(m):
    its, w = PLAN[m]; d = SR.done_keys(OUT)
    jobs = [i for i in its if (m, i["pair_id"], i["variant"], i["condition"], i["style"], 0) not in d]
    def one(i):
        if os.path.exists("STOP_MIT"): return
        SR.call(K, m, i, OUT, 0.0, 0)
    with ThreadPoolExecutor(w) as ex: list(ex.map(one, jobs))
    print("done", m, len(jobs), flush=True)
with ThreadPoolExecutor(len(PLAN)) as ex: list(ex.map(go, list(PLAN)))
print("DONE", flush=True)
