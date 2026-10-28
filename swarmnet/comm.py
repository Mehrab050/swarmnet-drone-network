"""Neighbour-only messaging shared by every swarm controller.

Each drone (and the base) keeps its own copy of a task table: for every survey point, who it thinks
has claimed it, with what bid, and whether it is done or delivered. Every second, each node swaps
tables ONLY with nodes in radio range, and keeps the better entry for each task. News therefore
travels one hop per second, like a message passed along a line of people.

Better = compared in this order:
  1. delivered beats everything (final)
  2. higher version (a release or a re-open bumps the version)
  3. done beats claimed beats open
  4. higher bid
  5. lower drone id (tie-break)
Drones also pass along heartbeats ("drone k was alive at time t"). If a drone that can reach the base
has not heard of drone k for a while, it presumes k lost and re-opens k's claimed tasks. Points k
surveyed but never delivered are re-opened only after a longer silence (three times as long), because
a drone that is simply out of range will usually come back and deliver them.
"""
import numpy as np

OPEN_S, CLAIMED_S, DONE_S = 0, 1, 2
LOCK = 1e6          # bids at or above this are "locked": the drone is already on its way


class Comm:
    def __init__(self, n_drones, n_tasks):
        n = n_drones + 1  # last node is the base
        self.N, self.M, self.n = n_drones, n_tasks, n
        self.ver = np.zeros((n, n_tasks), dtype=np.int64)
        self.st = np.zeros((n, n_tasks), dtype=np.int64)
        self.bid = np.zeros((n, n_tasks))
        self.win = np.full((n, n_tasks), -1, dtype=np.int64)
        self.dlv = np.zeros((n, n_tasks), dtype=bool)
        self.hb = np.zeros((n, n))
        self.messages = 0
        self._snap = None

    # ------------------------------------------------------------------ exchange
    def exchange(self, adj, alive_nodes, t, hb_period=5.0):
        for i in np.flatnonzero(alive_nodes):
            self.hb[i, i] = t
        old = (self.ver.copy(), self.st.copy(), self.bid.copy(), self.win.copy(), self.dlv.copy(),
               self.hb.copy())
        changed_before = self._changed_mask()
        has_nb = adj.any(1)
        beat = (int(round(t)) % int(hb_period) == 0)
        self.messages += int(np.sum(has_nb & alive_nodes & (changed_before | beat)))

        for i, j in np.argwhere(adj):
            self._merge_into(i, old, j)
            self.hb[i] = np.maximum(self.hb[i], old[5][j])
        self._snap = (self.ver.copy(), self.st.copy(), self.bid.copy(), self.win.copy(), self.dlv.copy())

    def _changed_mask(self):
        if self._snap is None:
            return np.ones(self.n, dtype=bool)
        v, s, b, w, d = self._snap
        return ((self.ver != v) | (self.st != s) | (self.bid != b) | (self.win != w)
                | (self.dlv != d)).any(1)

    def _merge_into(self, i, old, j):
        v, s, b, w, d = old[0][j], old[1][j], old[2][j], old[3][j], old[4][j]
        cv, cs, cb, cw, cd = self.ver[i], self.st[i], self.bid[i], self.win[i], self.dlv[i]
        better = (d & ~cd) | ((d == cd) & (
            (v > cv) | ((v == cv) & (
                (s > cs) | ((s == cs) & (
                    (b > cb) | ((b == cb) & (w < cw) & (w >= 0))))))))
        if better.any():
            self.ver[i, better] = v[better]
            self.st[i, better] = s[better]
            self.bid[i, better] = b[better]
            self.win[i, better] = w[better]
            self.dlv[i, better] = d[better]

    # ------------------------------------------------------------------ local edits
    def claim(self, i, j, bid):
        if not self.dlv[i, j]:
            self.st[i, j], self.bid[i, j], self.win[i, j] = CLAIMED_S, bid, i

    def release(self, i, j):
        if not self.dlv[i, j]:
            self.ver[i, j] += 1
            self.st[i, j], self.bid[i, j], self.win[i, j] = OPEN_S, 0.0, -1

    def mark_done(self, i, j):
        if not self.dlv[i, j]:
            self.st[i, j], self.win[i, j], self.bid[i, j] = DONE_S, i, LOCK * 10

    def mark_delivered(self, node, tasks):
        self.dlv[node, tasks] = True
        self.st[node, tasks] = DONE_S

    def reopen_lost(self, i, k, include_done=False):
        """Node i presumes drone k lost: re-open k's claimed tasks in i's table. With include_done,
        also re-open points k surveyed but never delivered (their readings died with k)."""
        m = (self.win[i] == k) & ((self.st[i] == CLAIMED_S) | (include_done & (self.st[i] == DONE_S))) & ~self.dlv[i]
        if m.any():
            self.ver[i, m] += 1
            self.st[i, m], self.bid[i, m], self.win[i, m] = OPEN_S, 0.0, -1

    def mine(self, i):
        return (self.win[i] == i) & (self.st[i] == CLAIMED_S) & ~self.dlv[i]
