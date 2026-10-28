"""Train the learned swarm (small budget) and compare it with the nearest-task rule on the same farms.

Needs PyTorch. Takes about 10 minutes on a laptop CPU. Usage:
    python examples/train_and_compare.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from swarmnet.experiments import paired_compare, sweep, validate
from swarmnet.marl import load_policy, train

train(iterations=40, episodes_per_iter=6, save_path="runs/learned_policy.pt", eval_every=5, device="cpu")
policy = load_policy("runs/learned_policy.pt", seed=0)
df = sweep(methods=("central", "nearest", "learned"), policy=policy, radio_ranges=(120, 1e6),
           fail_counts=(0, 1), seeds=range(2), central_time=1, out_csv="results/sweep_example.csv")
print(validate(df))
print(paired_compare(df, "learned", "nearest").to_string())
