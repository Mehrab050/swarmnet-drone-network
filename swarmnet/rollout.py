"""Training bookkeeping that does not need PyTorch: rewards, decision records and advantages.

Team reward, given every second:
    +10 * (new readings delivered to base) / (number of points)   reward for getting data home
    +5  * (change in the share of points surveyed)                 credit the moment a survey is done
    -1/300 per second                                              pressure to finish sooner
    +CONNECT_BONUS/300 * (share of live drones linked to the base) small reward for a connected team
    -10 * (share of readings still missing) if time runs out       penalty for an unfinished job

The survey term is "potential-based shaping": it is the change of a score (share surveyed), so it adds
up to the same total whatever the drones do and cannot change which plan is best. It only gives the
credit earlier: a choice that gets a survey done sooner is rewarded at once instead of at the very end.
A lost reading takes its survey credit back (the share surveyed goes down).
"""
import numpy as np

from .learned import build_obs, global_features

TIME_SCALE = 300.0
SURVEY_SHAPING = 5.0
CONNECT_BONUS = 0.5   # at most half the size of the time penalty, so finishing sooner still matters more


def critic_state(world, comm):
    """Per-drone features for the training critic (all live drones) plus global features."""
    feats = []
    for i in np.flatnonzero(world.alive):
        o = build_obs(world, comm, i)
        feats.append(np.concatenate([o["self"], o["base"]]))
    if not feats:
        feats = [np.zeros(12, dtype=np.float32)]
    return np.stack(feats).astype(np.float32), global_features(world)


class Recorder:
    def __init__(self):
        self.episodes = []

    def start(self, world):
        self.cur = {"steps": [], "rewards": [], "n_tasks": world.M}
        self.prev_delivered = 0
        self.prev_surveyed = 0.0

    def tick(self, world):
        d = int(np.sum(world.task_state == 2))
        sv = float(np.mean(world.task_state >= 1))
        alive = world.alive.sum()
        linked = float(world.base_comp[: world.N].sum() / alive) if alive else 0.0
        r = (10.0 * (d - self.prev_delivered) / world.M
             + SURVEY_SHAPING * (sv - self.prev_surveyed)
             - world.cfg.dt / TIME_SCALE
             + CONNECT_BONUS * linked * world.cfg.dt / TIME_SCALE)
        self.prev_delivered, self.prev_surveyed = d, sv
        self.cur["rewards"].append((world.t, r))

    def add_decision(self, t, batch, actions, logp, cstate):
        self.cur["steps"].append({"t": t, "obs": batch, "act": np.asarray(actions),
                                  "logp": np.asarray(logp, dtype=np.float32), "cstate": cstate})

    def finish(self, world):
        self.tick(world)
        if not world.makespan:
            missing = 1.0 - float(np.mean(world.task_state == 2))
            self.cur["rewards"].append((world.t, -10.0 * missing))
        self.cur["metrics"] = world.metrics()
        self.cur["return"] = float(sum(r for _, r in self.cur["rewards"]))
        self.episodes.append(self.cur)


def segment_rewards(ep, gamma):
    """Discounted reward collected between consecutive decision steps, and the discount to the next."""
    steps, rew = ep["steps"], ep["rewards"]
    times = [s["t"] for s in steps]
    R = np.zeros(len(steps))
    disc = np.zeros(len(steps))
    k = 0
    for (t, r) in rew:
        while k + 1 < len(times) and t >= times[k + 1]:
            k += 1
        if t >= times[0]:
            R[k] += (gamma ** (t - times[k])) * r
    for k in range(len(steps) - 1):
        disc[k] = gamma ** (times[k + 1] - times[k])
    return R, disc


def gae(R, disc, V, lam):
    """Generalised advantage estimation over decision steps with uneven time gaps."""
    A = np.zeros(len(R))
    last = 0.0
    for k in range(len(R) - 1, -1, -1):
        v_next = V[k + 1] if k + 1 < len(R) else 0.0
        delta = R[k] + disc[k] * v_next - V[k]
        last = delta + disc[k] * lam * last
        A[k] = last
    return A, A + V


def discounted_returns(R, disc):
    """Return-to-go at every decision step (used to pre-train the critic)."""
    G = np.zeros(len(R))
    run = 0.0
    for k in range(len(R) - 1, -1, -1):
        run = R[k] + disc[k] * run
        G[k] = run
    return G
