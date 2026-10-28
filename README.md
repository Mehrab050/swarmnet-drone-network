# SwarmNet: drone survey networks with limited radio and drone failures

A simulation study of how a team of drones should split up a survey job when each drone can only talk to drones close to it, and some drones break down mid-mission. It compares an **optimization model** (central routing plan) with **decentralized control** (each drone decides for itself), including a **learned policy** trained with multi-agent reinforcement learning. The swarm size is one setting: the same code runs 5, 10, 20 or 30 drones.

---

## The idea in plain words

Picture a 600 m × 600 m farm with 100 spots that need a sensor reading. Ten drones start at a base station at the edge of the field. Every reading must end up back at the base.

Two things make this hard:

1. **Short radio range.** A drone can only talk to drones (or the base) within, say, 120 m. A reading reaches the base by hopping from drone to drone, or by a drone flying back.
2. **Breakdowns.** A drone can fail mid-flight. Readings it was carrying are lost, and the team only notices when it goes silent.

We compare three ways to run the team:

| Method | How it decides | Analogy |
|---|---|---|
| **Central planner** | One computer sees everything and plans every drone's route. It solves a routing optimization problem: finish all points as early as possible. | A dispatcher with a perfect map and a phone line to every driver. |
| **Nearest-task rule** | Each drone goes to the closest open point it knows about. | Every driver just takes the closest job. |
| **Learned swarm** | Each drone runs the same small neural network, trained by trial and error (reinforcement learning). It sees only nearby points and nearby drones. It can also choose to **relay**: hover in place and pass messages along. | Drivers who learned from experience, and can also act as a radio tower. |

The central planner with unlimited radio and no failures is **our central plan**: the reference everything is measured against. Every result is reported as **"gap" = how much slower than our central plan (%)**. Lower is better.

---

## Key findings (reported run: 3 farms per setting, 231 missions, all validity checks passed)

1. **The reference plan is trustworthy.** On 6 small farms an exact solver (MILP) found the true optimum, and our plan matched it on all 6. The plan gives the same answer every run, never beats a mathematical lower bound, and does not get slower as the swarm grows (216, 201, 197, 200 s for 5, 10, 20, 30 drones).

2. **Decentralized control loses most when drones fail, not when choosing tasks.**

   | Setting (10 drones) | Central planner (real radio) | Nearest-task rule | Learned swarm |
   |---|---|---|---|
   | 150 m radio, no failures | +14% | +33% | +28% |
   | 150 m radio, 1 failure | +31% | +161% | +153% |
   | 150 m radio, 2 failures | +39% | +163% | +157% |

   A swarm drone cannot see a failure. It waits for silence (90 s by default, and 3× longer before redoing readings that were surveyed but not yet delivered). That waiting is the main cost.

3. **The failure-detection timeout is the biggest lever.** For the nearest-task rule with one failure, shortening the timeout from 90 s to 30 s cuts the gap from 161% to 81% at 150 m radio (and from 155% to 109% at 100 m), with no extra cost when nothing fails.

4. **Reinforcement learning converges to the simple rule.** The learned policy (graph attention + MAPPO) quickly stopped using the relay option (hovering costs more time than it saves here) and ends up statistically tied with the nearest-task rule on the same farms:

   | Failures | Learned vs rule (same farms) | 95% interval | Learned faster on |
   |---|---|---|---|
   | 0 | +4.9% | −5.8% to +15.7% | 11 of 15 |
   | 1 | +1.6% | −5.7% to +9.0% | 10 of 15 |
   | 2 | −0.1% | −4.8% to +4.6% | 6 of 15 |

   It is worse at the shortest radio (80 m), where it surveys some points twice. This is reported as a negative result: in this setting, learning did not find anything better than "go to the nearest point".

5. **The learned policy scales without retraining.** Trained on at most 10 drones, it runs on 20 and 30 drones and stays within about 3% of the rule.

### Limitations (read before citing numbers)

* 3 farms per setting: intervals are wide; differences under about 10% are not established.
* The reference plan is near-optimal on small farms, not proven optimal on 100 points, so gaps are "to our central plan", not to the true optimum.
* The central planner with "real radio" is still optimistic: it learns about failures instantly and its orders reach every drone.
* Simple physics: constant speed, no wind, hovering costs half of flying.

---

## Interactive dashboard

Open `docs/index.html` in a browser (or turn on GitHub Pages for the `docs` folder) to watch the central planner, the nearest-task rule and the trained learned swarm fly the same farm side by side, and to explore the results.

## How to run

**Easiest: Google Colab (free tier is enough).** Open `notebooks/SwarmNet.ipynb` in Colab and choose Runtime → Run all.
* `QUICK = True`: a 25-minute preview.
* `QUICK = False`: the reported study (35–75 minutes).
* `SAVE_TO_DRIVE = True`: asks for Google Drive access once at the start, then saves results after every part.

The notebook writes the `swarmnet` package itself, so you only need the notebook.

**On your own computer:**
```bash
pip install -r requirements.txt
python examples/run_one_mission.py                       # no PyTorch needed; writes HTML replays
python examples/run_one_mission.py --drones 30 --radio 120 --fail 1
python examples/train_and_compare.py                     # trains the learned swarm (~10 min on CPU)
```

**Change the swarm size** with one setting:
```python
from swarmnet.config import Config
from swarmnet.experiments import run_one
m = run_one("nearest", Config(n_drones=30, n_tasks=300, radio_range=120, fail_count=1))
print(m["makespan"], "seconds")
```

---

## What is in the repository

```
swarmnet/
  config.py       every setting in one place (drones, radio range, battery, failures, ...)
  world.py        the simulator: drones, survey points, base, radio links, battery, failures, relay hovering
  comm.py         neighbour-only messaging and lost-drone detection (the silence timeout)
  routing.py      routing solvers: built-in search, optional OR-Tools, exact MILP for small farms
  central.py      the central planner (uses routing.py)
  swarm.py        shared swarm plumbing (plus an auction swarm, CBBA, kept for reference)
  learned.py      learned-swarm controller, what each drone sees, and the nearest-task rule
  rollout.py      rewards and advantage estimates for training (no PyTorch)
  marl.py         graph-attention actor, pooled critic, warm start, MAPPO training (PyTorch)
  experiments.py  sweeps, scaling tests, paired comparisons, and validity checks
  plots.py        charts
  replay.py       self-contained HTML replay of a mission
notebooks/SwarmNet.ipynb   the full study, step by step, with checks that stop the run if something is wrong
examples/                  small scripts to run one mission or a short training
docs/index.html            interactive dashboard (mission replays + results)
```

---

## Method details (for readers who want them)

* **Central plan:** min-makespan vehicle routing with open routes (a route ends at its last survey). Solved by balanced cheapest insertion plus large neighbourhood search with a fixed iteration budget (deterministic). OR-Tools can polish the plan (`use_ortools=True`) and is kept only if strictly faster. The exact check is a MILP with MTZ subtour constraints (SciPy/HiGHS).
* **Communication:** every second each node merges its task table with neighbours in range (versioned entries: delivered > newer version > done > claimed > open). A drone that can reach the base presumes a silent teammate lost after `fail_detect_time` seconds.
* **Learned policy:** each drone sees itself, its 6 nearest open points and up to 8 radio neighbours (fixed-size, masked), so one network runs on any swarm size. The actor uses attention over neighbours; the critic pools all drones (centralized training, decentralized execution).
* **Training:** behaviour cloning of the nearest-task rule (98% match), relay given a 5% starting probability, then MAPPO with separate actor and critic optimisers on mostly hard missions (80–150 m radio, 0–2 failures). Curriculum of 3 → 5 → 10 drones. The checkpoint is chosen on 10 validation farms that are never used for the reported results.
* **Fair comparison:** the nearest-task rule runs inside the same controller as the learned policy, so differences come from decisions alone. Results are paired by farm.

## Author

Md Mehrab Hossain. Questions and suggestions are welcome via GitHub issues.
