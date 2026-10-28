"""SwarmNet: a drone survey network with limited radio, where a learned policy decides for each drone.

Modules
  config       all settings (the number of drones is one setting)
  world        the simulated farm: drones, points, base, radio links, battery, failures, relay hovering
  comm         neighbour-only messaging and lost-drone detection
  central      central planner (our optimization reference)
  routing      routing solvers: a built-in search, OR-Tools (optional), and an exact MILP for small cases
  swarm        swarm plumbing (and the auction swarm, kept for reference but not used in the notebook)
  learned      learned-swarm controller, the local observation, and the nearest-task rule
  rollout      rewards and advantages for training (no PyTorch needed)
  marl         graph-attention policy + MAPPO training (needs PyTorch)
  experiments  sweeps over radio range, failures and swarm size, plus validity checks
  plots        charts
  replay       HTML replay dashboard
"""
__version__ = "1.4.1"
