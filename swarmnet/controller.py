"""The common shape every controller follows. The world calls these hooks."""


class Controller:
    name = "base"
    messages = 0

    def reset(self, world):
        """Called once when the mission starts."""

    def update(self, world):
        """Called every step before the drones move. Change world.queue[i] to give orders."""

    def on_complete(self, world, i, task):
        """Drone i has just surveyed `task` and now carries its reading."""

    def on_delivered(self, world, tasks):
        """These readings just reached the base."""

    def on_failure(self, world, ids):
        """Drones in `ids` have just failed. Only the central planner may use this directly;
        swarm controllers must notice failures on their own, through missing messages."""
