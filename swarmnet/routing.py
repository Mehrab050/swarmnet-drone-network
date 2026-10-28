"""Vehicle routing solvers used by the central planner.

The problem: K drones start at given points (possibly busy for a few seconds first), must visit every
task once, and end at the base. Objective: minimise the time the LAST drone finishes (makespan),
which spreads the work evenly instead of just shortening total distance.

Two solvers:
  * solve_ortools   Google OR-Tools (best quality; installed with `pip install ortools`)
  * solve_heuristic pure Python: cheapest balanced insertion + local search (always available)
plus solve_exact, an exact MILP (scipy) for small cases, used to measure how close the others get.
"""
from __future__ import annotations

import numpy as np


def route_time(route, start_xy, offset, tasks_xy, base, speed, survey, open_end=True):
    """Seconds for one drone: start offset + travel + surveys (+ flight back to base if not open_end).

    open_end=True matches the mission score when every reading can be relayed home at once, so the
    plan finishes at the last survey. This makes the central planner an optimistic best case."""
    t, p = offset, start_xy
    for j in route:
        q = tasks_xy[j]
        t += np.hypot(*(q - p)) / speed + survey
        p = q
    return t if open_end else t + np.hypot(*(base - p)) / speed


# ---------------------------------------------------------------------------- heuristic
def solve_heuristic(starts, offsets, tasks_xy, base, speed, survey, time_budget=2.0, open_end=True,
                    seed=0):
    """Large neighbourhood search: build routes by balanced insertion, then repeatedly remove a few
    points and re-insert them where they hurt the slowest drone least. Keeps the best plan found."""
    import time as _time
    rng = np.random.default_rng(seed)
    starts, tasks_xy, base = np.asarray(starts, float), np.asarray(tasks_xy, float), np.asarray(base, float)
    K, T = len(starts), len(tasks_xy)
    if T == 0 or K == 0:
        return [[] for _ in range(K)]
    pts = np.vstack([starts, tasks_xy, base[None, :]])
    D = np.hypot(*(pts[:, None, :] - pts[None, :, :]).transpose(2, 0, 1)) / speed
    BASE = K + T
    off = np.asarray(offsets, float)

    def r_time(k, r):
        if not r:
            return off[k]
        nodes = [k] + [K + j for j in r]
        t = off[k] + D[nodes[:-1], nodes[1:]].sum() + survey * len(r)
        return t if open_end else t + D[nodes[-1], BASE]

    def insert_all(routes, times, todo):
        for c in todo:
            cn = K + c
            best = None
            for k in range(K):
                r = routes[k]
                prev = np.array([k] + [K + j for j in r])
                if open_end:
                    nxt = np.array([K + j for j in r] + [-1])
                    delta = D[prev, cn] + survey
                    has = nxt >= 0
                    delta[has] += D[cn, nxt[has]] - D[prev[has], nxt[has]]
                else:
                    nxt = np.array([K + j for j in r] + [BASE])
                    delta = D[prev, cn] + D[cn, nxt] - D[prev, nxt] + survey
                q = int(np.argmin(delta))
                new_t = times[k] + delta[q]
                others = max([times[m] for m in range(K) if m != k], default=0.0)
                key = (max(new_t, others), delta[q])
                if best is None or key < best[0]:
                    best = (key, k, q, new_t)
            _, k, q, new_t = best
            routes[k].insert(q, c)
            times[k] = new_t

    def two_opt(routes, times, k):
        r = routes[k]
        improved = True
        while improved and len(r) > 2:
            improved = False
            for a_ in range(len(r) - 1):
                for b_ in range(a_ + 1, len(r)):
                    cand = r[:a_] + r[a_:b_ + 1][::-1] + r[b_ + 1:]
                    if r_time(k, cand) < r_time(k, r) - 1e-9:
                        r, improved = cand, True
        routes[k] = r
        times[k] = r_time(k, r)

    def key_of(times):
        return (max(times), sum(times))

    routes = [[] for _ in range(K)]
    times = [off[k] for k in range(K)]
    order = list(np.argsort(-np.hypot(*(tasks_xy - base).T)))
    insert_all(routes, times, order)
    for k in range(K):
        two_opt(routes, times, k)
    best_routes, best_times = [list(r) for r in routes], list(times)

    # a fixed number of attempts, never a clock: the same farm always gives the same plan,
    # on any computer (a wall-clock cut-off would give different plans on slower machines)
    max_iters = int(300 * time_budget)
    for it in range(max_iters):
        cur = [list(r) for r in best_routes]
        n_rm = int(rng.integers(2, max(3, min(15, T // 4 + 2))))
        mode = rng.integers(3)
        if mode == 0:     # random points
            removed = list(rng.choice(T, size=min(n_rm, T), replace=False))
        elif mode == 1:   # points near a random point
            c0 = int(rng.integers(T))
            removed = list(np.argsort(D[K + c0, K:K + T])[:n_rm])
        else:             # points from the slowest drone
            k_max = int(np.argmax(best_times))
            pool = cur[k_max]
            removed = list(rng.choice(pool, size=min(n_rm, len(pool)), replace=False)) if pool else []
        if not removed:
            continue
        rm = set(int(x) for x in removed)
        cur = [[j for j in r if j not in rm] for r in cur]
        cur_t = [r_time(k, cur[k]) for k in range(K)]
        rng.shuffle(removed)
        insert_all(cur, cur_t, [int(x) for x in removed])
        if max(cur_t) <= max(best_times) * 1.02:   # only polish promising plans
            for k in range(K):
                two_opt(cur, cur_t, k)
            if key_of(cur_t) < key_of(best_times):
                best_routes, best_times = cur, cur_t
    return [[int(j) for j in r] for r in best_routes]


# ---------------------------------------------------------------------------- OR-Tools
_ORTOOLS_WARNED = []


def plan_makespan(routes, starts, offsets, tasks_xy, base, speed, survey, open_end=True):
    """Finish time of the slowest drone for a set of routes."""
    return max((route_time(r, starts[k], offsets[k], tasks_xy, base, speed, survey, open_end)
                for k, r in enumerate(routes)), default=0.0)


def solve_ortools(starts, offsets, tasks_xy, base, speed, survey, solution_limit=300, safety_seconds=60,
                  open_end=True, initial_routes=None):
    """Min-makespan routing with OR-Tools. Returns routes, or None if OR-Tools is missing or fails.

    Objective: the time dimension's global span (latest finish - earliest start) = the makespan,
    with a tiny arc cost as tie-break. Two things make it reliable:
      * it is WARM-STARTED from `initial_routes` (the built-in search's plan), so it can only
        improve on a plan that already spreads the work over every drone;
      * it stops after a fixed number of solutions (`solution_limit`), not a wall-clock limit,
        so the same farm always gives the same plan. `safety_seconds` is only a backstop.
    """
    try:
        from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    except Exception:
        return None
    try:
        starts, tasks_xy, base = np.asarray(starts, float), np.asarray(tasks_xy, float), np.asarray(base, float)
        K, T = len(starts), len(tasks_xy)
        if T == 0:
            return [[] for _ in range(K)]
        pts = np.vstack([starts, tasks_xy, base[None, :]])
        n = len(pts)
        end = n - 1
        SCALE = 10  # deciseconds: OR-Tools needs integers
        D = np.hypot(*(pts[:, None, :] - pts[None, :, :]).transpose(2, 0, 1))
        travel = np.rint(D / speed * SCALE).astype(np.int64)
        if open_end:
            travel[:, end] = 0   # the route ends at its last survey
        service = np.zeros(n, dtype=np.int64)
        service[K:K + T] = int(round(survey * SCALE))
        transit = (travel + service[:, None]).tolist()    # time from leaving i to arriving at j

        mgr = pywrapcp.RoutingIndexManager(n, K, list(range(K)), [end] * K)
        model = pywrapcp.RoutingModel(mgr)

        def time_cb(i, j):
            return transit[mgr.IndexToNode(i)][mgr.IndexToNode(j)]

        t_idx = model.RegisterTransitCallback(time_cb)
        model.SetArcCostEvaluatorOfAllVehicles(t_idx)            # tie-break only (coefficient 1)
        horizon = int(travel.max() * (T + 2) + service.sum() + max(offsets) * SCALE + 10)
        model.AddDimension(t_idx, 0, horizon, False, "Time")
        dim = model.GetDimensionOrDie("Time")
        for k in range(K):
            off = int(round(offsets[k] * SCALE))
            dim.CumulVar(model.Start(k)).SetRange(off, off)
        # span weight >> total arc cost, so the makespan always dominates
        dim.SetGlobalSpanCostCoefficient(int(10 * (T + K)))

        params = pywrapcp.DefaultRoutingSearchParameters()
        params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
        params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        params.solution_limit = int(solution_limit)
        params.time_limit.seconds = int(safety_seconds)

        if initial_routes is not None:
            model.CloseModelWithParameters(params)
            init = model.ReadAssignmentFromRoutes([[K + int(j) for j in r] for r in initial_routes], True)
            sol = model.SolveFromAssignmentWithParameters(init, params) if init is not None else None
        else:
            sol = model.SolveWithParameters(params)
        if sol is None:
            return None
        routes = []
        for k in range(K):
            i, r = sol.Value(model.NextVar(model.Start(k))), []
            while not model.IsEnd(i):
                r.append(mgr.IndexToNode(i) - K)
                i = sol.Value(model.NextVar(i))
            routes.append(r)
        return routes
    except Exception as e:  # an OR-Tools API difference must never break the study
        if not _ORTOOLS_WARNED:
            _ORTOOLS_WARNED.append(1)
            print(f"[routing] OR-Tools failed ({type(e).__name__}: {e}); using the built-in search only.")
        return None


def solve(starts, offsets, tasks_xy, base, speed, survey, prefer="ortools", time_limit=3, open_end=True):
    """Best available plan. Returns (routes, solver_name). Deterministic for a given farm.

    Always runs the built-in search first (fixed iteration count). If OR-Tools is installed and
    prefer == "ortools", it is warm-started from that plan and its answer is kept only if it is
    strictly faster. So adding OR-Tools can never make the plan worse.
    """
    starts = np.asarray(starts, float)
    base_routes = solve_heuristic(starts, offsets, tasks_xy, base, speed, survey, time_budget=time_limit,
                                  open_end=open_end)
    if prefer != "ortools" or len(tasks_xy) == 0:
        return base_routes, "heuristic"
    r = solve_ortools(starts, offsets, tasks_xy, base, speed, survey, solution_limit=int(100 * time_limit),
                      open_end=open_end, initial_routes=base_routes)
    if r is None:
        return base_routes, "heuristic"
    mk = lambda rr: plan_makespan(rr, starts, offsets, tasks_xy, base, speed, survey, open_end)
    if mk(r) < mk(base_routes) - 1e-9:
        return r, "ortools"
    return base_routes, "heuristic"


# ---------------------------------------------------------------------------- exact (small)
def solve_exact(tasks_xy, base, n_vehicles, speed, survey, time_limit=120, open_end=True):
    """Exact min-makespan routing from the base, as a MILP (scipy/HiGHS). Use for <= ~9 tasks.

    Returns (routes, makespan_seconds, proven_optimal)."""
    from scipy.optimize import LinearConstraint, Bounds, milp

    tasks_xy, base = np.asarray(tasks_xy, float), np.asarray(base, float)
    T, K = len(tasks_xy), n_vehicles
    pts = np.vstack([base[None, :], tasks_xy])
    n = T + 1
    D = np.hypot(*(pts[:, None, :] - pts[None, :, :]).transpose(2, 0, 1)) / speed
    if open_end:
        D[:, 0] = 0.0   # returning to the base costs nothing: the route ends at its last survey
    arcs = [(i, j) for i in range(n) for j in range(n) if i != j]
    A = len(arcs)
    nx = A * K
    iu = nx            # u_1..u_T  (MTZ order variables)
    iz = nx + T        # makespan
    nv = nx + T + 1

    def x(a, k):
        return k * A + a

    rows, lo, hi = [], [], []

    def add(coefs, l, h):
        row = np.zeros(nv)
        for idx, v in coefs:
            row[idx] += v
        rows.append(row); lo.append(l); hi.append(h)

    for j in range(1, n):   # each task entered exactly once
        add([(x(a, k), 1) for k in range(K) for a, (i, jj) in enumerate(arcs) if jj == j], 1, 1)
    for k in range(K):
        for v in range(n):  # flow conservation
            co = [(x(a, k), 1) for a, (i, j) in enumerate(arcs) if j == v]
            co += [(x(a, k), -1) for a, (i, j) in enumerate(arcs) if i == v]
            add(co, 0, 0)
        add([(x(a, k), 1) for a, (i, j) in enumerate(arcs) if i == 0], 0, 1)  # leave base once
        co = [(x(a, k), D[i, j] + (survey if j > 0 else 0)) for a, (i, j) in enumerate(arcs)]
        co.append((iz, -1))
        add(co, -np.inf, 0)                                                     # route time <= Z
    for k in range(K - 1):  # symmetry breaking: drone k's route is at least as long as drone k+1's
        co = [(x(a, k), D[i, j] + (survey if j > 0 else 0)) for a, (i, j) in enumerate(arcs)]
        co += [(x(a, k + 1), -(D[i, j] + (survey if j > 0 else 0))) for a, (i, j) in enumerate(arcs)]
        add(co, 0, np.inf)
    for a, (i, j) in enumerate(arcs):  # MTZ subtour elimination
        if i > 0 and j > 0:
            co = [(iu + i - 1, 1), (iu + j - 1, -1)] + [(x(a, k), T) for k in range(K)]
            add(co, -np.inf, T - 1)

    c = np.zeros(nv); c[iz] = 1
    integrality = np.zeros(nv); integrality[:nx] = 1
    lb = np.zeros(nv); ub = np.ones(nv)
    lb[iu:iu + T], ub[iu:iu + T] = 1, T
    ub[iz] = np.inf
    res = milp(c, constraints=LinearConstraint(np.array(rows), lo, hi), integrality=integrality,
               bounds=Bounds(lb, ub), options={"time_limit": time_limit})
    if res.x is None:
        return None, None, False
    xs = res.x
    routes = []
    for k in range(K):
        nxt = {i: j for a, (i, j) in enumerate(arcs) if xs[x(a, k)] > 0.5}
        r, cur = [], nxt.get(0)
        while cur is not None and cur != 0:
            r.append(cur - 1)
            cur = nxt.get(cur)
        routes.append(r)
    return routes, float(res.fun), res.status == 0   # status 0 = proven optimal
