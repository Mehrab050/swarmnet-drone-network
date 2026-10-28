"""Run the study: every method on the same farms, across radio ranges, failures and swarm sizes.

Main measure: GAP TO OUR CENTRAL PLAN
    reference ("ideal" in the code) = our central planner with unlimited radio and no failures
    (perfect information, readings reach the base instantly). It is near-optimal on small farms
    (checked against an exact solver) but not proven optimal on 100 points.
    gap = (method's finish time / reference finish time - 1) * 100 %, on the SAME farm (same seed).

Methods
    central   our central planner, run with the real radio range and failures
    nearest   the nearest-task rule, run inside the learned-swarm controller (same plumbing)
    learned   the trained policy, in the same controller
    auction   the auction swarm (CBBA); kept in the code for reference, not used in the notebook
"""
import os
import time

import numpy as np
import pandas as pd

from .central import CentralPlanner
from .config import Config
from .learned import LearnedSwarm, nearest_policy
from .swarm import AuctionSwarm, NearestTask
from .world import World


def make_controller(method, policy=None, central_time=3):
    if method == "central":
        return CentralPlanner(time_limit=central_time)
    if method == "nearest":
        return LearnedSwarm(nearest_policy, name="nearest")
    if method == "learned":
        if policy is None:
            raise ValueError("method 'learned' needs a trained policy (see marl.load_policy)")
        return LearnedSwarm(policy, name="learned")
    if method == "auction":
        return AuctionSwarm()
    if method == "nearest_v13":
        return NearestTask()   # the older stand-alone nearest-task swarm (v1.3 and before)
    raise ValueError(method)


def run_one(method, cfg, policy=None, central_time=3, keep_world=False):
    w = World(cfg, make_controller(method, policy, central_time))
    m = w.run()
    return (m, w) if keep_world else m


def ideal_makespan(cfg, central_time=3):
    """Central planner, unlimited radio, no failures: the best case for this farm."""
    m = run_one("central", cfg.with_(radio_range=1e6, fail_count=0, trace_every=0), central_time=central_time)
    return m["makespan"]


def sweep(methods=("central", "nearest"), radio_ranges=(60, 90, 120, 150, 200, 1e6),
          fail_counts=(0, 1, 3), seeds=range(5), n_drones=10, tasks_per_drone=10, policy=None,
          base_cfg=None, central_time=3, out_csv="results/sweep.csv", log=print):
    base_cfg = base_cfg or Config()
    rows, t0 = [], time.time()
    for s in seeds:
        cfg0 = base_cfg.with_(n_drones=n_drones, n_tasks=tasks_per_drone * n_drones, seed=s, trace_every=0)
        ideal = ideal_makespan(cfg0, central_time)
        for f in fail_counts:
            for r in radio_ranges:
                for meth in methods:
                    m = run_one(meth, cfg0.with_(radio_range=float(r), fail_count=f), policy, central_time)
                    m["ideal"] = ideal
                    m["gap_pct"] = (100.0 * (m["makespan"] / ideal - 1.0)) if m["makespan"] else np.nan
                    rows.append(m)
        log(f"seed {s} done  ({(time.time() - t0) / 60:.1f} min)")
    df = pd.DataFrame(rows)
    if out_csv:
        os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
        df.to_csv(out_csv, index=False)
    return df


def scaling(methods=("nearest",), sizes=(5, 10, 20, 30), seeds=range(3), radio_range=150,
            fail_count=0, policy=None, base_cfg=None, central_time=3, out_csv="results/scaling.csv", log=print):
    """Same work per drone (10 points), growing swarm. For 'learned', the SAME trained policy is used
    at every size: no retraining."""
    base_cfg = base_cfg or Config()
    rows = []
    for n in sizes:
        for s in seeds:
            cfg = base_cfg.with_(n_drones=n, n_tasks=10 * n, seed=s, radio_range=radio_range,
                                 fail_count=fail_count, trace_every=0)
            ideal = ideal_makespan(cfg, central_time)
            for meth in methods:
                t0 = time.time()
                m = run_one(meth, cfg, policy, central_time)
                m["ideal"] = ideal
                m["gap_pct"] = (100.0 * (m["makespan"] / ideal - 1.0)) if m["makespan"] else np.nan
                m["wall_seconds"] = time.time() - t0
                rows.append(m)
        log(f"{n} drones done")
    df = pd.DataFrame(rows)
    if out_csv:
        os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
        df.to_csv(out_csv, index=False)
    return df


def summary(df, by=("method", "radio_range", "fail_count")):
    cols = ["gap_pct", "makespan", "delivered_frac", "mean_lambda2", "repeat_surveys", "messages",
            "recovery_time", "relay_seconds", "readings_lost"]
    cols = [c for c in cols if c in df.columns]
    g = df.groupby(list(by))[cols].agg(["mean", "std"]).round(2)
    return g


def paired_compare(df, a="learned", b="nearest", by=("radio_range", "fail_count"), level=0.95):
    """Method a vs method b on the SAME farms: mean of (gap_a - gap_b), its confidence interval, and
    how often a finished faster. Negative = a is faster. Pairing removes farm-to-farm noise, so this
    is the fair way to ask "did learning beat the rule?"."""
    from scipy import stats
    keys = ["seed", "n_drones"] + list(by)
    keys = [k for k in dict.fromkeys(keys) if k in df.columns]
    A = df[df["method"] == a].set_index(keys)["makespan"]
    B = df[df["method"] == b].set_index(keys)["makespan"]
    j = pd.concat([A.rename("a"), B.rename("b")], axis=1).dropna().reset_index()
    j["diff_pct"] = 100.0 * (j["a"] / j["b"] - 1.0)
    rows = []
    for key, g in j.groupby(list(by)):
        v = g["diff_pct"].to_numpy()
        n = len(v)
        half = float(stats.t.ppf(0.5 + level / 2, n - 1) * v.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan
        key = key if isinstance(key, tuple) else (key,)
        rows.append({**dict(zip(by, key)), "n_farms": n,
                     f"{a}_vs_{b}_pct": round(float(v.mean()), 1),
                     "ci_low": round(float(v.mean()) - half, 1) if n > 1 else np.nan,
                     "ci_high": round(float(v.mean()) + half, 1) if n > 1 else np.nan,
                     f"{a}_faster_on": f"{int((v < 0).sum())}/{n}"})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------- validity checks
def makespan_lower_bound(cfg):
    """A finish time no plan can beat (open routes from the base, no failures).

    Two bounds, take the larger:
      * the farthest point: fly there and survey it;
      * total work shared perfectly: every point is surveyed once and entered by one flight at
        least as long as its nearest neighbour (base or another point), split over all drones.
    """
    w = World(cfg.with_(trace_every=0), NearestTask())
    tasks, base = w.task_xy, w.base
    d_base = np.hypot(*(tasks - base).T) / cfg.speed
    pts = np.vstack([tasks, base[None, :]])
    D = np.hypot(*(tasks[:, None, :] - pts[None, :, :]).transpose(2, 0, 1)) / cfg.speed
    D[np.arange(len(tasks)), np.arange(len(tasks))] = np.inf
    far = float(np.max(d_base + cfg.survey_time))
    share = float((len(tasks) * cfg.survey_time + D.min(axis=1).sum()) / cfg.n_drones)
    return max(far, share)


def check_ideal(cfg, central_time=3, repeats=2):
    """The reference plan must be reproducible and must not beat the lower bound. Returns a dict."""
    vals = [ideal_makespan(cfg, central_time) for _ in range(repeats)]
    lb = makespan_lower_bound(cfg)
    return {"ideal_runs": vals, "reproducible": len(set(vals)) == 1, "lower_bound": round(lb, 1),
            "ideal_over_bound": round(vals[0] / lb, 3), "above_bound": vals[0] >= lb - 1e-6}


def validate(df, tol_pct=0.5):
    """Sanity checks on a sweep or scaling table. Raises AssertionError with a clear message.

      1. no mission may finish faster than our central plan for its farm (beyond tol_pct);
      2. the central planner with unlimited radio and no failures IS the reference, so its gap is 0.
    """
    d = df[df["makespan"].notna()]
    beat = d[d["gap_pct"] < -tol_pct]
    msgs = []
    if len(beat):
        worst = beat.sort_values("gap_pct").iloc[0]
        msgs.append(f"{len(beat)} missions beat our central plan (worst: {worst['method']} at "
                    f"{worst['gap_pct']:.1f}%). The reference plan is not near-optimal: do not use these results.")
    if "radio_range" in d and "fail_count" in d:
        c = d[(d["method"] == "central") & (d["radio_range"] >= 1e5) & (d["fail_count"] == 0)]
        bad = c[c["gap_pct"].abs() > 1e-6]
        if len(bad):
            msgs.append(f"central planner at unlimited radio differs from the reference on {len(bad)} farms "
                        f"(gaps {sorted(bad['gap_pct'].round(1).tolist())}): the planner is not deterministic.")
    if msgs:
        raise AssertionError("RESULTS NOT VALID:\n - " + "\n - ".join(msgs))
    return f"checks passed: {len(d)} missions, none beat our central plan"


def mean_ci(df, by, col="gap_pct", level=0.95):
    """Mean of `col` per group with a t-based confidence interval. With few farms the interval is
    wide: groups whose intervals overlap are not distinguishable."""
    from scipy import stats
    rows = []
    for key, g in df.groupby(list(by)):
        v = g[col].dropna().to_numpy()
        n = len(v)
        half = float(stats.t.ppf(0.5 + level / 2, n - 1) * v.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan
        key = key if isinstance(key, tuple) else (key,)
        rows.append({**dict(zip(by, key)), "n_farms": n, "mean": round(float(v.mean()), 1) if n else np.nan,
                     "ci_low": round(float(v.mean()) - half, 1) if n > 1 else np.nan,
                     "ci_high": round(float(v.mean()) + half, 1) if n > 1 else np.nan})
    return pd.DataFrame(rows)
