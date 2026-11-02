"""Addendum A2 runner (hosted models). Key in env K. Resumable."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study_run as SR
from concurrent.futures import ThreadPoolExecutor
K = os.environ["K"]; OUT = "real_results.jsonl"
items = json.load(open("real_items.json"))
W = {"nvidia/nemotron-3.5-super-vl-120b-a12b": 8, "meta/llama-3.2-11b-vision-instruct": 3,
     "meta/llama-3.2-90b-vision-instruct": 8, "meta/muse-glimmer-30b": 8, "google/gemma-4-31b-it": 10}
def go(m):
    d = SR.done_keys(OUT)
    jobs = [i for i in items if (m, i["pair_id"], i["variant"], i["condition"], i["style"], 0) not in d]
    with ThreadPoolExecutor(W[m]) as ex: list(ex.map(lambda i: SR.call(K, m, i, OUT, 0.0, 0), jobs))
    print("done", m, len(jobs), flush=True)
with ThreadPoolExecutor(len(W)) as ex: list(ex.map(go, list(W)))
print("DONE", flush=True)
