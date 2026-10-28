"""Central planner: one computer sees every drone and every point and plans all routes.

Real swarms cannot do this (it needs perfect information and a link to every drone), so it is used
only as the MEASURING STICK: the best result the swarm could hope for. It even replans instantly
when a drone fails, because it is told about failures directly.
"""
import numpy as np

from .controller import Controller
from .routing import solve
from .world import OPEN, SURVEY, CHARGE


class CentralPlanner(Controller):
    name = "central"

    def __init__(self, prefer=None, time_limit=3):
        """prefer=None follows the config (use_ortools); "ortools" or "heuristic" forces one."""
        self.prefer, self.time_limit = prefer, time_limit
        self.solver_used = None

    def reset(self, world):
        self.messages = 0
        self._plan(world)

    def on_failure(self, world, ids):
        self._plan(world)

    def _plan(self, world):
        c = world.cfg
        alive = np.flatnonzero(world.alive)
        busy = {int(world.survey_task[i]) for i in alive if world.mode[i] == SURVEY}
        todo = [j for j in range(world.M) if world.task_state[j] == OPEN and j not in busy]
        starts, offsets = [], []
        for i in alive:
            starts.append(world.pos[i].copy())
            if world.mode[i] == SURVEY:
                offsets.append(float(world.survey_left[i]))
            elif world.mode[i] == CHARGE:
                offsets.append(float(world.charge_left[i]))
            else:
                offsets.append(0.0)
        prefer = self.prefer or ("ortools" if getattr(c, "use_ortools", False) else "heuristic")
        routes, self.solver_used = solve(np.array(starts), offsets, world.task_xy[todo], world.base,
                                         c.speed, c.survey_time, prefer, self.time_limit)
        for k, i in enumerate(alive):
            plan = [todo[j] for j in routes[k]]
            if world.mode[i] == SURVEY:
                plan = [int(world.survey_task[i])] + plan
            world.queue[i] = plan
