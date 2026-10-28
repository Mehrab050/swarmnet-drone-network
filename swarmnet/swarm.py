"""Decentralized swarm controllers: no boss, each drone decides from its own table.

AuctionSwarm  (CBBA, the Consensus-Based Bundle Algorithm)
    Every few seconds each drone looks at open points near it and bids for the ones it can do best.
    A bid is how much the point adds to the drone's own plan, scored so that finishing sooner is
    worth more. Bids spread one hop per second. If someone else holds a higher bid, the drone gives
    the point up, plus every point it added after it (those bids assumed it had the first one).

NearestTask   (simple baseline)
    Each drone claims the single nearest open point it knows about. Closer drones win ties.
"""
import numpy as np

from .comm import Comm, LOCK, OPEN_S, CLAIMED_S
from .controller import Controller
from .world import FLY, SURVEY, CHARGE


class SwarmBase(Controller):
    """Shared plumbing: messaging, failure detection, giving up lost claims."""

    name = "swarm"

    def reset(self, world):
        self.comm = Comm(world.N, world.M)
        self.next_think = np.arange(world.N) * 0.3  # stagger thinking so drones do not move in lockstep

    @property
    def messages(self):
        return self.comm.messages if hasattr(self, "comm") else 0

    def on_complete(self, world, i, task):
        self.comm.mark_done(i, task)

    def on_delivered(self, world, tasks):
        self.comm.mark_delivered(world.N, tasks)

    # ------------------------------------------------------------------
    def update(self, world):
        c, N, t = world.cfg, world.N, world.t
        comm = self.comm
        comm.exchange(world.adj, world.node_alive(), t)

        for i in np.flatnonzero(world.alive):
            # 1. presume silent drones lost, but only if I can still reach the base myself
            if t - comm.hb[i, N] <= 10.0:
                gap = t - comm.hb[i, :N]
                for k in np.flatnonzero(gap > c.fail_detect_time):
                    if k != i:
                        comm.reopen_lost(i, k, include_done=gap[k] > 3 * c.fail_detect_time)

            # 2. drop claims I have lost, and everything I planned after them
            q = world.queue[i]
            protected = int(world.survey_task[i]) if world.mode[i] == SURVEY else -1
            for idx, j in enumerate(q):
                if j == protected:
                    continue
                if not (comm.win[i, j] == i and comm.st[i, j] == CLAIMED_S and not comm.dlv[i, j]):
                    for later in q[idx + 1:]:
                        if later != protected and comm.win[i, later] == i and comm.st[i, later] == CLAIMED_S:
                            comm.release(i, later)
                    world.queue[i] = [x for x in q[:idx] ] + ([protected] if protected in q[idx:] else [])
                    break

            # 3. think: add new claims
            if t >= self.next_think[i]:
                self.next_think[i] = t + c.rebuild_every
                self.think(world, i)

            # 4. lock the point I am flying to, so nobody steals it mid-flight
            q = world.queue[i]
            if q and world.mode[i] in (FLY, SURVEY):
                j = q[0]
                if comm.win[i, j] == i and comm.bid[i, j] < LOCK:
                    comm.bid[i, j] += LOCK

    def think(self, world, i):
        raise NotImplementedError

    # helpers -------------------------------------------------------------
    def start_state(self, world, i):
        """Where and when drone i can start new work."""
        off = 0.0
        if world.mode[i] == SURVEY:
            off = float(world.survey_left[i])
        elif world.mode[i] == CHARGE:
            off = float(world.charge_left[i])
        return world.pos[i].copy(), off

    def candidates(self, world, i, allow_outbid):
        comm = self.comm
        free = (comm.st[i] == OPEN_S) & ~comm.dlv[i]
        if allow_outbid:
            free |= ((comm.st[i] == CLAIMED_S) & (comm.win[i] != i) & (comm.bid[i] < LOCK)
                     & ~comm.dlv[i])
        free[world.queue[i]] = False
        idx = np.flatnonzero(free)
        if len(idx) > world.cfg.candidate_limit:
            d = np.hypot(*(world.task_xy[idx] - world.pos[i]).T)
            idx = idx[np.argsort(d)[: world.cfg.candidate_limit]]
        return idx


class AuctionSwarm(SwarmBase):
    name = "auction"

    def think(self, world, i):
        c, comm = world.cfg, self.comm
        tau, v, s = c.score_tau, c.speed, c.survey_time
        while len(world.queue[i]) < c.bundle_size:
            cand = self.candidates(world, i, allow_outbid=True)
            if len(cand) == 0:
                return
            path = world.queue[i]
            p0, off = self.start_state(world, i)
            pts = [p0] + [world.task_xy[j] for j in path]
            done_t, t_cur = [], off           # finish time of each planned point
            for a in range(len(path)):
                t_cur += np.hypot(*(pts[a + 1] - pts[a])) / v + s
                done_t.append(t_cur)
            done_t = np.array(done_t)
            C = world.task_xy[cand]
            best_gain = np.full(len(cand), -np.inf)
            best_pos = np.zeros(len(cand), dtype=int)
            for q in range(len(path) + 1):
                P = pts[q]
                t_prev = off if q == 0 else done_t[q - 1]
                d_in = np.hypot(*(C - P).T)
                gain = np.exp(-(t_prev + d_in / v + s) / tau)
                if q < len(path):
                    nxt = pts[q + 1]
                    delta = (d_in + np.hypot(*(C - nxt).T) - np.hypot(*(nxt - P))) / v + s
                    after = done_t[q:]
                    gain = gain + (np.exp(-(after[None, :] + delta[:, None]) / tau)
                                   - np.exp(-after[None, :] / tau)).sum(1)
                better = gain > best_gain
                best_gain[better], best_pos[better] = gain[better], q
            current = comm.bid[i, cand] * np.where(comm.st[i, cand] == CLAIMED_S, 1 + c.outbid_margin, 1.0)
            ok = best_gain > current + 1e-9
            if not ok.any():
                return
            k = int(np.argmax(np.where(ok, best_gain, -np.inf)))
            j = int(cand[k])
            comm.claim(i, j, float(best_gain[k]))
            path.insert(int(best_pos[k]), j)


class NearestTask(SwarmBase):
    name = "nearest"

    def think(self, world, i):
        if world.queue[i]:
            return
        cand = self.candidates(world, i, allow_outbid=False)
        if len(cand) == 0:
            return
        d = np.hypot(*(world.task_xy[cand] - world.pos[i]).T)
        k = int(np.argmin(d))
        self.comm.claim(i, int(cand[k]), 1000.0 / (1.0 + d[k]))
        world.queue[i] = [int(cand[k])]
