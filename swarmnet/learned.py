"""Learned swarm: each drone's choice comes from a policy (a neural network after training).

A drone asks its policy for a decision whenever it has nothing to do. It sees ONLY local information:
  * itself:      position, battery, readings carried, whether it can reach the base, whether it is relaying
  * nearby points: the K nearest points its own table says are still open
  * neighbours:  drones within radio range (what they broadcast: position, battery, busy or not)
  * the base:    where it is and whether it is in range
It then picks one of three kinds of action:
  * fly to one of the K points and survey it
  * head home (deliver the readings it carries, recharge, park)
  * RELAY: hover where it is and pass messages along, so drones further out stay connected to the
    base and their readings reach home without flying back. The nearest-task rule never does this.

The observation has the same shape whatever the swarm size: K points and up to M neighbours, with
masks for missing ones. That is what lets a policy trained on 10 drones run on 20 or 30.

Fairness: the nearest-task rule (`nearest_policy`) runs inside this SAME controller, so a difference
between "learned" and "nearest" comes from the decisions alone, not from different plumbing.
"""
import numpy as np

from .comm import LOCK, OPEN_S
from .swarm import SwarmBase
from .world import IDLE, HOME, CHARGE, SURVEY

K_TASKS = 6        # points offered to the policy
M_NEIGH = 8        # neighbours the policy can see
SELF_DIM, TASK_DIM, NB_DIM, BASE_DIM = 9, 5, 6, 4
HOME_ACTION = K_TASKS            # head home
RELAY_ACTION = K_TASKS + 1       # hover here as a radio relay
N_ACTIONS = K_TASKS + 2
RETHINK = 15.0                   # seconds before a drone that chose "home" or "relay" decides again


def build_obs(world, comm, i):
    """Local observation for drone i. Returns a dict of fixed-shape arrays."""
    c, L, R, N = world.cfg, world.cfg.field_size, world.cfg.radio_range, world.N
    p = world.pos[i]
    db = np.linalg.norm(p - world.base)
    contact = float(world.t - comm.hb[i, N] <= 3.0)
    nb = np.flatnonzero(world.adj[i, :N])
    progress = float((comm.dlv[i] | (comm.st[i] == 2)).mean())
    self_f = np.array([p[0] / L, p[1] / L, world.battery[i] / c.battery_range, len(world.held[i]) / 5.0,
                       db / L, contact, len(nb) / 5.0, progress, float(world.hold[i])], dtype=np.float32)

    open_ = np.flatnonzero((comm.st[i] == OPEN_S) & ~comm.dlv[i])
    task_f = np.zeros((K_TASKS, TASK_DIM), dtype=np.float32)
    task_m = np.zeros(K_TASKS, dtype=bool)
    task_ids = np.full(K_TASKS, -1)
    if len(open_):
        d = np.hypot(*(world.task_xy[open_] - p).T)
        order = np.argsort(d)[:K_TASKS]
        for s, o in enumerate(order):
            j = open_[o]
            q = world.task_xy[j]
            dtb = np.linalg.norm(q - world.base)
            task_f[s] = [(q[0] - p[0]) / L, (q[1] - p[1]) / L, d[o] / L, dtb / L, float(dtb <= R)]
            task_m[s] = True
            task_ids[s] = j

    nb_f = np.zeros((M_NEIGH, NB_DIM), dtype=np.float32)
    nb_m = np.zeros(M_NEIGH, dtype=bool)
    if len(nb):
        dn = np.hypot(*(world.pos[nb] - p).T)
        for s, o in enumerate(np.argsort(dn)[:M_NEIGH]):
            k = nb[o]
            q = world.pos[k]
            nb_f[s] = [(q[0] - p[0]) / R, (q[1] - p[1]) / R, world.battery[k] / c.battery_range,
                       len(world.held[k]) / 5.0, float(not world.queue[k]), float(world.hold[k])]
            nb_m[s] = True

    base_f = np.array([(world.base[0] - p[0]) / L, (world.base[1] - p[1]) / L, db / L, float(db <= R)],
                      dtype=np.float32)
    return {"self": self_f, "tasks": task_f, "task_mask": task_m, "nb": nb_f, "nb_mask": nb_m,
            "base": base_f, "task_ids": task_ids}


def global_features(world):
    """Extra information for the training critic only (never used by drones in the field)."""
    return np.array([world.t / 600.0, float(np.mean(world.task_state == 2)),
                     float(np.mean(world.task_state >= 1)), world.alive.mean()], dtype=np.float32)


def stack(obs_list):
    keys = ["self", "tasks", "task_mask", "nb", "nb_mask", "base"]
    return {k: np.stack([o[k] for o in obs_list]) for k in keys}


class LearnedSwarm(SwarmBase):
    """policy(batch_obs, world, ids, obs) -> array of action indices, one per deciding drone."""

    name = "learned"

    def __init__(self, policy, name=None, recorder=None):
        self.policy = policy
        self.recorder = recorder      # used during training to store decisions
        if name:
            self.name = name

    def reset(self, world):
        super().reset(world)
        self.wait_until = np.zeros(world.N)
        if self.recorder is not None:
            self.recorder.start(world)

    def think(self, world, i):
        pass  # decisions are made together in decide(), not one by one

    def update(self, world):
        if self.recorder is not None:
            self.recorder.tick(world)
        super().update(world)
        self.decide(world)

    def decide(self, world):
        need = [i for i in np.flatnonzero(world.alive)
                if not world.queue[i] and world.mode[i] in (IDLE, HOME)
                and not (world.mode[i] == HOME and world.home_reason[i] == "charge")
                and world.t >= self.wait_until[i]]
        if not need:
            return
        obs = [build_obs(world, self.comm, i) for i in need]
        acts = np.asarray(self.policy(stack(obs), world, need, obs), dtype=int)
        for i, o, a in zip(need, obs, acts):
            if a < K_TASKS and o["task_mask"][a]:
                j = int(o["task_ids"][a])
                d = float(np.linalg.norm(world.task_xy[j] - world.pos[i]))
                self.comm.claim(i, j, LOCK + 1000.0 / (1.0 + d))   # closer drone wins a clash
                world.queue[i] = [j]
                world.hold[i] = False
            elif a == RELAY_ACTION:
                world.queue[i] = []
                world.hold[i] = True                                # hover here (see World._move)
                self.wait_until[i] = world.t + RETHINK
            else:
                world.queue[i] = []
                world.hold[i] = False                               # head home
                self.wait_until[i] = world.t + RETHINK


# ------------------------------------------------------------------ simple fixed policies
def nearest_policy(batch, world, ids, obs):
    """The nearest-task rule: always take the nearest open point; go home if none. Never relays."""
    return np.where(batch["task_mask"][:, 0], 0, HOME_ACTION)


def make_random_policy(seed=0, relay_prob=0.1):
    rng = np.random.default_rng(seed)

    def pol(batch, world, ids, obs):
        out = []
        for m in batch["task_mask"]:
            if rng.random() < relay_prob:
                out.append(RELAY_ACTION)
                continue
            valid = np.append(np.flatnonzero(m), HOME_ACTION)
            out.append(rng.choice(valid))
        return np.array(out)
    return pol
