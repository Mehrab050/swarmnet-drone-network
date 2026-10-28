"""Run one mission with the central planner and the nearest-task rule, and save HTML replays.

No PyTorch needed. Usage (from the repository folder):
    python examples/run_one_mission.py
    python examples/run_one_mission.py --drones 30 --radio 120 --fail 1
Then open replays/central.html or replays/nearest.html in a browser.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from swarmnet.config import Config
from swarmnet.experiments import ideal_makespan, run_one
from swarmnet.replay import save_html

ap = argparse.ArgumentParser()
ap.add_argument("--drones", type=int, default=10, help="swarm size (10 survey points per drone)")
ap.add_argument("--radio", type=float, default=120.0, help="radio range in metres")
ap.add_argument("--fail", type=int, default=1, help="drones that fail at t = 150 s")
ap.add_argument("--seed", type=int, default=0, help="which random farm")
a = ap.parse_args()

cfg = Config(n_drones=a.drones, n_tasks=10 * a.drones, radio_range=a.radio, fail_count=a.fail, seed=a.seed)
ref = ideal_makespan(cfg, central_time=1)
print(f"our central plan (unlimited radio, no failures): {ref:.0f} s")
for method in ("central", "nearest"):
    m, w = run_one(method, cfg, central_time=1, keep_world=True)
    path = save_html(w.trace(), f"replays/{method}.html")
    gap = 100 * (m["makespan"] / ref - 1) if m["makespan"] else float("nan")
    print(f"{method:8s} finished in {m['makespan']} s  (gap {gap:+.0f}%)  "
          f"readings lost {m['readings_lost']}  repeat surveys {m['repeat_surveys']}  -> {path}")
