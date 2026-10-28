"""All settings for one mission. Change a number here (or pass it in) to change the world.

The number of drones is a single setting: Config(n_drones=40) is all it takes to scale up.
"""
from dataclasses import dataclass, field, replace
from typing import List, Tuple


@dataclass
class Config:
    # --- the world -------------------------------------------------------------------
    n_drones: int = 10            # swarm size: change this to scale up or down
    n_tasks: int = 100            # survey points on the farm
    field_size: float = 600.0     # square field, metres per side
    base: Tuple[float, float] = (300.0, 0.0)  # base station: middle of the bottom edge

    # --- drones ----------------------------------------------------------------------
    speed: float = 8.0            # metres per second
    survey_time: float = 10.0     # seconds spent at each point taking the reading
    battery_range: float = 4000.0 # metres a full battery can fly
    battery_margin: float = 50.0  # metres kept in reserve when deciding to go home
    recharge_time: float = 30.0   # seconds on the pad to recharge

    # --- communication ---------------------------------------------------------------
    radio_range: float = 150.0    # metres: two nodes can talk if they are this close
    fail_detect_time: float = 90.0  # seconds of silence before a drone is presumed lost (trade-off: faster recovery vs false alarms)

    # --- failures --------------------------------------------------------------------
    fail_count: int = 0           # how many drones fail during the mission
    fail_time: float = 150.0      # when they fail (seconds after start)

    # --- simulation ------------------------------------------------------------------
    dt: float = 1.0               # seconds per simulation step
    max_time: float = 3000.0      # stop here even if the job is not finished
    seed: int = 0
    use_ortools: bool = False     # central planner: polish its plan with OR-Tools (slower; optional)
    trace_every: int = 2          # record a replay frame every N steps (0 = no replay)

    # --- relay (learned swarm) -------------------------------------------------------
    hover_drain: float = 4.0      # metres of battery used per second while hovering as a relay

    # --- auction swarm (CBBA) --------------------------------------------------------
    bundle_size: int = 5          # most tasks a drone can claim at once
    score_tau: float = 300.0      # time scale of the bid score (seconds)
    outbid_margin: float = 0.10   # a new bid must beat the old one by 10% to steal a task
    candidate_limit: int = 25     # only the nearest open tasks are considered when bidding
    rebuild_every: float = 3.0    # seconds between bundle rebuilds

    def with_(self, **kw) -> "Config":
        """Copy with some fields changed, e.g. cfg.with_(radio_range=60)."""
        return replace(self, **kw)
