"""A5: C1, C2, C5 one-stage (study_run.call); C3, C4 two-turn (run_two_stage_ctrl). Nemotron only. Key in env K."""
import os, sys, json, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study_run as SR
from concurrent.futures import ThreadPoolExecutor
M = "nvidia/nemotron-3.5-super-vl-120b-a12b"; OUT = "controls_results.jsonl"; K = os.environ["K"]
items = json.load(open("controls_items.json"))
done = SR.done_keys(OUT)
todo = [i for i in items if (M, i["pair_id"], i["variant"], i["condition"], i["style"], 0) not in done
        and (i["condition"] != "C2_neutral_metadata" or int(i["pair_id"][1:]) < 15)]
PRI = {"C1_raw_false_header": 0, "C5_native_no_zlabels": 1, "C2_neutral_metadata": 2}
todo.sort(key=lambda i: (PRI[i["condition"]], i["pair_id"]))
def two(mode):
    subprocess.run([sys.executable, "run_two_stage_ctrl.py"], env={**os.environ, "TS_MODE": mode, "TS_OUT": OUT, "TS_NEM_ONLY": "1", "TS_W": "2"})
with ThreadPoolExecutor(3) as ex:
    f1 = ex.submit(lambda: list(ThreadPoolExecutor(5).map(lambda i: SR.call(K, M, i, OUT, 0.0, 0), todo)))
    f1.result()
print("DONE", flush=True)
