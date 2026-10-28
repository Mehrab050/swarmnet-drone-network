"""The simulated farm: drones, survey points, base station, radio links, battery and failures.

The world only knows physics and the ground truth. Decisions (which drone goes where) come from a
controller: the central planner, the auction swarm, or the learned swarm. Every controller runs in
exactly the same world, so their results can be compared fairly.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from .config import Config

# drone modes
IDLE, FLY, SURVEY, HOME, CHARGE, DEAD = range(6)
MODE_NAMES = ["idle", "fly", "survey", "home", "charge", "dead"]

# ground-truth task states
OPEN, HELD, DELIVERED = 0, 1, 2


class World:
    def __init__(self, cfg: Config, controller):
        self.cfg = cfg
        self.controller = controller
        self.reset()

    # ------------------------------------------------------------------ setup
    def reset(self):
        c = self.cfg
        self.rng = np.random.default_rng(c.seed)
        self.N, self.M = c.n_drones, c.n_tasks
        self.base = np.array(c.base, dtype=float)
        self.task_xy = self.rng.uniform(0, c.field_size, size=(self.M, 2))

        # drones start parked at the base
        self.pos = np.repeat(self.base[None, :], self.N, axis=0).astype(float)
        self.battery = np.full(self.N, c.battery_range)
        self.alive = np.ones(self.N, dtype=bool)
        self.mode = np.full(self.N, IDLE)
        self.home_reason = ["park"] * self.N
        self.queue = [[] for _ in range(self.N)]
        self.hold = np.zeros(self.N, dtype=bool)       # learned swarm can choose to hold position
        self.survey_left = np.zeros(self.N)
        self.survey_task = np.full(self.N, -1)
        self.charge_left = np.zeros(self.N)
        self.held = [set() for _ in range(self.N)]      # readings carried, not yet delivered
        self.flown = np.zeros(self.N)
        self.hover_time = 0.0                           # drone-seconds spent hovering as a relay

        self.task_state = np.full(self.M, OPEN)
        self.holders = [set() for _ in range(self.M)]
        self.delivered_time = np.full(self.M, np.nan)
        self.complete_count = np.zeros(self.M, dtype=int)

        self.t = 0.0
        self.steps = 0
        self.done = False
        self.makespan = None
        self.failed_yet = False
        self.events = []
        self.lost_tasks = set()
        self.fail_t = None
        self.first_redo = {}

        self.lam2_log, self.conn_log, self.basefrac_log = [], [], []
        self.frames = []
        self._update_comm()
        self.controller.reset(self)
        self._record(force=True)

    # ------------------------------------------------------------------ geometry helpers
    def node_positions(self):
        """Drone positions followed by the base (index N)."""
        return np.vstack([self.pos, self.base[None, :]])

    def node_alive(self):
        return np.append(self.alive, True)

    def dist_to_base(self, i):
        return float(np.linalg.norm(self.pos[i] - self.base))

    # ------------------------------------------------------------------ communication
    def _update_comm(self):
        P = self.node_positions()
        alive = self.node_alive()
        D = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
        adj = (D <= self.cfg.radio_range) & alive[:, None] & alive[None, :]
        np.fill_diagonal(adj, False)
        self.adj = adj
        n_comp, labels = connected_components(csr_matrix(adj), directed=False)
        self.comp_labels = labels
        self.base_comp = (labels == labels[self.N]) & alive  # nodes that can reach the base

        # algebraic connectivity (Fiedler value) of the live network, base included
        idx = np.flatnonzero(alive)
        A = adj[np.ix_(idx, idx)].astype(float)
        L = np.diag(A.sum(1)) - A
        ev = np.linalg.eigvalsh(L) if len(idx) > 1 else np.array([0.0])
        self.lam2 = float(ev[1]) if len(ev) > 1 else 0.0
        self.connected = bool(np.all(labels[idx] == labels[idx[0]]))

    # ------------------------------------------------------------------ main loop
    def run(self):
        while not self.done:
            self.step()
        return self.metrics()

    def step(self):
        c = self.cfg
        if c.fail_count > 0 and not self.failed_yet and self.t >= c.fail_time:
            self._fail_random(c.fail_count)

        self.controller.update(self)
        for i in range(self.N):
            if self.alive[i]:
                if self.hold[i] and not self.queue[i] and self.mode[i] == IDLE:
                    self.hover_time += c.dt
                self._move(i)

        self._update_comm()
        self._deliver()
        self.lam2_log.append(self.lam2)
        self.conn_log.append(self.connected)
        a = self.alive.sum()
        self.basefrac_log.append(float(self.base_comp[: self.N].sum() / a) if a else 0.0)

        self.t += c.dt
        self.steps += 1
        if np.all(self.task_state == DELIVERED):
            self.done, self.makespan = True, self.t
        elif self.t >= c.max_time or not self.alive.any():
            self.done = True
        self._record(force=self.done)

    # ------------------------------------------------------------------ drone physics
    def _fly_toward(self, i, target):
        d = target - self.pos[i]
        dist = float(np.hypot(*d))
        step = self.cfg.speed * self.cfg.dt
        if dist <= step:
            self.pos[i] = target
            moved, arrived = dist, True
        else:
            self.pos[i] = self.pos[i] + d / dist * step
            moved, arrived = step, False
        self.battery[i] -= moved
        self.flown[i] += moved
        if self.battery[i] <= 0 and not arrived:
            self._kill(i, "battery ran out")
        return arrived

    def _move(self, i):
        c = self.cfg
        m = self.mode[i]
        if m == CHARGE:
            self.charge_left[i] -= c.dt
            if self.charge_left[i] <= 0:
                self.battery[i] = c.battery_range
                self.mode[i] = IDLE
            return
        if m == SURVEY:
            self.survey_left[i] -= c.dt
            if self.survey_left[i] <= 0:
                task = int(self.survey_task[i])
                if task in self.queue[i]:
                    self.queue[i].remove(task)
                self.survey_task[i] = -1
                self.mode[i] = IDLE
                self._complete(i, task)
            return
        if m == HOME and self.home_reason[i] == "charge":
            if self._fly_toward(i, self.base):
                self.mode[i] = CHARGE
                self.charge_left[i] = c.recharge_time
            return

        if self.queue[i]:
            task = self.queue[i][0]
            tgt = self.task_xy[task]
            need = (np.linalg.norm(self.pos[i] - tgt) + np.linalg.norm(tgt - self.base)
                    + c.battery_margin)
            if self.battery[i] < need:
                self.mode[i], self.home_reason[i] = HOME, "charge"
                if self._fly_toward(i, self.base):
                    self.mode[i] = CHARGE
                    self.charge_left[i] = c.recharge_time
                return
            self.mode[i] = FLY
            if self._fly_toward(i, tgt):
                self.mode[i] = SURVEY
                self.survey_left[i] = c.survey_time
                self.survey_task[i] = task
            return

        if self.hold[i]:
            # hovering in place as a radio relay: costs battery, so come home before it runs low
            if self.battery[i] < self.dist_to_base(i) + c.battery_margin + c.hover_drain * c.dt:
                self.hold[i] = False
                self.mode[i], self.home_reason[i] = HOME, "charge"
                if self._fly_toward(i, self.base):
                    self.mode[i] = CHARGE
                    self.charge_left[i] = c.recharge_time
                return
            self.battery[i] -= c.hover_drain * c.dt
            self.mode[i] = IDLE
            return
        if np.allclose(self.pos[i], self.base):
            self.mode[i] = IDLE
        else:
            self.mode[i], self.home_reason[i] = HOME, "park"
            if self._fly_toward(i, self.base):
                self.mode[i] = IDLE

    # ------------------------------------------------------------------ tasks and data
    def _complete(self, i, task):
        self.complete_count[task] += 1
        if self.fail_t is not None and task in self.lost_tasks and task not in self.first_redo:
            self.first_redo[task] = self.t
        if self.task_state[task] != DELIVERED:
            self.task_state[task] = HELD
            self.holders[task].add(i)
            self.held[i].add(task)
        self.controller.on_complete(self, i, task)

    def _deliver(self):
        new = []
        for i in np.flatnonzero(self.base_comp[: self.N]):
            if self.held[i]:
                for task in list(self.held[i]):
                    if self.task_state[task] != DELIVERED:
                        self.task_state[task] = DELIVERED
                        self.delivered_time[task] = self.t
                        new.append(task)
                    for h in self.holders[task]:
                        if h != i:
                            self.held[h].discard(task)
                    self.holders[task].clear()
                self.held[i].clear()
        if new:
            self.controller.on_delivered(self, new)

    # ------------------------------------------------------------------ failures
    def _kill(self, i, why):
        self.alive[i] = False
        self.mode[i] = DEAD
        self.queue[i] = []
        self.hold[i] = False
        self.survey_task[i] = -1
        lost = []
        for task in self.held[i]:
            self.holders[task].discard(i)
            if not self.holders[task] and self.task_state[task] == HELD:
                self.task_state[task] = OPEN
                lost.append(task)
        self.held[i].clear()
        self.events.append({"t": self.t, "drone": int(i), "what": why, "lost": len(lost)})
        return lost

    def _fail_random(self, k):
        self.failed_yet = True
        cand = np.flatnonzero(self.alive)
        k = min(k, len(cand))
        ids = self.rng.choice(cand, size=k, replace=False)
        self.fail_t = self.t
        for i in ids:
            plan = [x for x in self.queue[i] if self.task_state[x] == OPEN]
            lost = self._kill(i, "failure")
            self.lost_tasks.update(lost)
            self.lost_tasks.update(plan)
        self.controller.on_failure(self, [int(i) for i in ids])

    # ------------------------------------------------------------------ results
    def metrics(self):
        rec = None
        if self.fail_t is not None and self.lost_tasks:
            if all(t in self.first_redo for t in self.lost_tasks):
                rec = max(self.first_redo[t] for t in self.lost_tasks) - self.fail_t
        return {
            "method": getattr(self.controller, "name", "?"),
            "n_drones": self.N,
            "radio_range": self.cfg.radio_range,
            "fail_count": self.cfg.fail_count,
            "seed": self.cfg.seed,
            "finished": self.makespan is not None,
            "makespan": self.makespan,
            "time_used": self.t,
            "delivered_frac": float(np.mean(self.task_state == DELIVERED)),
            "total_distance": float(self.flown.sum()),
            "mean_lambda2": float(np.mean(self.lam2_log)) if self.lam2_log else 0.0,
            "connected_frac": float(np.mean(self.conn_log)) if self.conn_log else 0.0,
            "base_contact_frac": float(np.mean(self.basefrac_log)) if self.basefrac_log else 0.0,
            "repeat_surveys": int(np.maximum(self.complete_count - 1, 0).sum()),  # redone (lost or overlap)
            "messages": int(getattr(self.controller, "messages", 0)),
            "recovery_time": rec,
            "relay_seconds": float(self.hover_time),
            "readings_lost": int(sum(e["lost"] for e in self.events)),
        }

    # ------------------------------------------------------------------ replay
    def _record(self, force=False):
        k = self.cfg.trace_every
        if not k or (self.steps % k and not force):
            return
        drones = []
        for i in range(self.N):
            tgt = None
            if self.alive[i]:
                if self.mode[i] == HOME or self.mode[i] == CHARGE:
                    tgt = self.base.tolist()
                elif self.queue[i]:
                    tgt = self.task_xy[self.queue[i][0]].tolist()
            drones.append([round(float(self.pos[i, 0]), 1), round(float(self.pos[i, 1]), 1),
                           int(self.alive[i]), int(self.mode[i]), len(self.held[i]),
                           None if tgt is None else [round(tgt[0], 1), round(tgt[1], 1)],
                           round(float(self.battery[i] / self.cfg.battery_range), 3)])
        iu = np.argwhere(np.triu(self.adj, 1))
        self.frames.append({
            "t": round(self.t, 1),
            "d": drones,
            "l": iu.tolist(),
            "bc": np.flatnonzero(self.base_comp).tolist(),
            "ts": "".join(map(str, self.task_state.tolist())),
            "lam2": round(self.lam2, 3),
        })

    def trace(self):
        return {
            "meta": {
                "method": getattr(self.controller, "name", "?"),
                "n_drones": self.N, "n_tasks": self.M,
                "field": self.cfg.field_size, "base": self.base.tolist(),
                "radio_range": self.cfg.radio_range,
                "tasks": np.round(self.task_xy, 1).tolist(),
                "events": self.events,
                "metrics": self.metrics(),
            },
            "frames": self.frames,
        }
