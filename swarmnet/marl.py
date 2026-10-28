"""Learned swarm: graph-attention policy trained with MAPPO (needs PyTorch; Colab has it).

Networks
  Actor  (runs on every drone, shared weights)
         encodes itself, its neighbours and the base; ATTENTION decides how much to listen to each
         neighbour; then it scores each of the K nearby open points, "head home", and "relay"
         (hover here and pass messages to the base). Works for any number of neighbours and any
         swarm size.
  Critic (training only)
         encodes every live drone, averages and max-pools them, adds global progress, and predicts
         how well the rest of the mission will go. Pooling makes it work for any swarm size too.

Training (MAPPO = multi-agent PPO, centralised training, decentralised execution)
  0. WARM START (behaviour cloning): the actor first copies the nearest-task rule on recorded
     missions. The rule never relays, so the relay choice is left out of the copying and then
     given a small starting probability (RELAY_START), so reinforcement learning can try it.
  1. collect missions on HARD settings (short radio, drone failures: where the rule is weak) ->
     score each decision with the team reward (advantage) -> nudge the policy towards
     better-than-expected decisions, with PPO's clipping to keep each update small.
  A curriculum starts with 3 drones, then 5, then 10. The best checkpoint is kept, judged on
  validation farms that are never used for the results; the warm start is the first candidate.
"""
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn

from .config import Config
from .learned import (BASE_DIM, K_TASKS, NB_DIM, SELF_DIM, TASK_DIM, RELAY_ACTION, LearnedSwarm,
                      nearest_policy)
from .rollout import Recorder, critic_state, discounted_returns, gae, segment_rewards
from .world import World

VALUE_SCALE = 10.0                       # the critic predicts return / VALUE_SCALE
VALIDATION_SEEDS = range(2000, 2010)     # 10 farms used to PICK the best checkpoint (never in results)
VALIDATION_SETTINGS = ((100.0, 1), (150.0, 0))   # (radio range m, drones that fail): one hard, one easy
RELAY_START = 0.05                       # starting probability of "relay" after the warm start


def step(opt, loss, params, max_norm):
    """One optimiser step with its OWN gradient clip, so one network cannot shrink the other's update."""
    opt.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(params, max_norm)
    opt.step()


def mlp(i, h):
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU())


class Actor(nn.Module):
    def __init__(self, h=64):
        super().__init__()
        self.h = h
        self.self_enc, self.nb_enc, self.base_enc = mlp(SELF_DIM, h), mlp(NB_DIM, h), mlp(BASE_DIM, h)
        self.q, self.k, self.v = nn.Linear(h, h), nn.Linear(h, h), nn.Linear(h, h)
        self.ctx = mlp(3 * h, h)
        self.task_enc = mlp(TASK_DIM, h)
        self.task_head = nn.Sequential(nn.Linear(2 * h, h), nn.ReLU(), nn.Linear(h, 1))
        self.home_head = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))
        self.relay_head = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))

    def forward(self, o):
        hs = self.self_enc(o["self"])                                  # (B, h)
        hn = self.nb_enc(o["nb"])                                      # (B, M, h)
        nb_mask = o["nb_mask"]                                         # (B, M) bool
        score = (self.q(hs).unsqueeze(1) * self.k(hn)).sum(-1) / math.sqrt(self.h)
        score = score.masked_fill(~nb_mask, -1e9)
        alpha = torch.softmax(score, dim=-1) * nb_mask.float()         # no neighbours -> all zero
        msg = (alpha.unsqueeze(-1) * self.v(hn)).sum(1)                # (B, h)
        c = self.ctx(torch.cat([hs, msg, self.base_enc(o["base"])], -1))
        ht = self.task_enc(o["tasks"])                                 # (B, K, h)
        tl = self.task_head(torch.cat([c.unsqueeze(1).expand(-1, ht.shape[1], -1), ht], -1)).squeeze(-1)
        tl = tl.masked_fill(~o["task_mask"], -1e9)
        logits = torch.cat([tl, self.home_head(c), self.relay_head(c)], -1)   # (B, K+2)
        return logits, alpha


class Critic(nn.Module):
    def __init__(self, h=64, agent_dim=SELF_DIM + BASE_DIM, glob_dim=4):
        super().__init__()
        self.enc = mlp(agent_dim, h)
        self.head = nn.Sequential(nn.Linear(2 * h + glob_dim, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(),
                                  nn.Linear(h, 1))

    def forward(self, feats, mask, glob):
        e = self.enc(feats)                                            # (B, N, h)
        m = mask.unsqueeze(-1).float()
        mean = (e * m).sum(1) / m.sum(1).clamp(min=1.0)
        mx = e.masked_fill(~mask.unsqueeze(-1), -1e9).max(1).values
        mx = torch.where(mask.any(1, keepdim=True), mx, torch.zeros_like(mx))
        return self.head(torch.cat([mean, mx, glob], -1)).squeeze(-1)


# ------------------------------------------------------------------ helpers
def to_t(batch, device):
    out = {}
    for k in ["self", "tasks", "nb", "base"]:
        out[k] = torch.as_tensor(batch[k], dtype=torch.float32, device=device)
    for k in ["task_mask", "nb_mask"]:
        out[k] = torch.as_tensor(batch[k], dtype=torch.bool, device=device)
    return out


def pad_states(cstates, device):
    n = max(f.shape[0] for f, _ in cstates)
    F = np.zeros((len(cstates), n, cstates[0][0].shape[1]), dtype=np.float32)
    Mk = np.zeros((len(cstates), n), dtype=bool)
    G = np.stack([g for _, g in cstates]).astype(np.float32)
    for b, (f, _) in enumerate(cstates):
        F[b, : len(f)] = f
        Mk[b, : len(f)] = True
    return (torch.as_tensor(F, device=device), torch.as_tensor(Mk, device=device),
            torch.as_tensor(G, device=device))


def make_policy(actor, device="cpu", greedy=False, recorder=None, seed=None):
    """Wrap the actor as a policy the LearnedSwarm controller can call.

    greedy=False (default) samples an action from the policy. A purely greedy policy can get stuck
    (a drone parked at base picks "stay home" forever). Sampling with a fixed seed is reproducible.
    """
    gen = None
    if seed is not None:
        gen = torch.Generator(device=device)
        gen.manual_seed(int(seed))

    def pol(batch, world, ids, obs):
        with torch.no_grad():
            logits, _ = actor(to_t(batch, device))
            dist = torch.distributions.Categorical(logits=logits)
            if greedy:
                a = logits.argmax(-1)
            else:
                a = torch.multinomial(torch.softmax(logits, -1), 1, generator=gen).squeeze(-1)
            lp = dist.log_prob(a)
        a_np = a.cpu().numpy()
        if recorder is not None:
            recorder.add_decision(world.t, batch, a_np, lp.cpu().numpy(),
                                  critic_state(world, world.controller.comm))
        return a_np
    return pol


def load_policy(path, device="cpu", greedy=False, seed=0):
    ck = torch.load(path, map_location=device, weights_only=False)
    actor = Actor(ck.get("hidden", 64)).to(device)
    actor.load_state_dict(ck["actor"])
    actor.eval()
    return make_policy(actor, device, greedy=greedy, seed=seed)


# ------------------------------------------------------------------ warm start
def collect_teacher(rng, base_cfg, sizes=(3, 5, 10), episodes=30, teacher=nearest_policy):
    """Run missions with a fixed rule and record every decision (observation + the rule's action)."""
    rec = Recorder()

    def pol(batch, world, ids, obs):
        a = np.asarray(teacher(batch, world, ids, obs))
        rec.add_decision(world.t, batch, a, np.zeros(len(a), np.float32), critic_state(world, world.controller.comm))
        return a

    for e in range(episodes):
        w = World(sample_cfg(rng, sizes[e % len(sizes)], base_cfg), LearnedSwarm(pol, recorder=rec))
        w.run()
        rec.finish(w)
    return rec.episodes


def _no_relay(logits):
    """The rule never relays, so the relay choice is left out while copying it."""
    out = logits.clone()
    out[:, RELAY_ACTION] = -1e9
    return out


def calibrate_relay(actor, batch, device, p=RELAY_START):
    """Shift the relay head's bias so that, on the recorded decisions, "relay" starts with average
    probability about p. Without this the copied policy would never try relaying."""
    with torch.no_grad():
        lg, _ = actor(to_t(batch, device))
        others = torch.logsumexp(_no_relay(lg), -1)
        base = lg[:, RELAY_ACTION]
        mean_p = lambda s: float(torch.sigmoid(base + s - others).mean())
        lo, hi = -50.0, 50.0                      # find the bias shift that gives mean probability p
        for _ in range(60):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if mean_p(mid) < p else (lo, mid)
        actor.relay_head[-1].bias += (lo + hi) / 2
        lg2, _ = actor(to_t(batch, device))
        return float(torch.softmax(lg2, -1)[:, RELAY_ACTION].mean())


def pretrain_bc(actor, critic, opt_a, opt_c, episodes, gamma=0.995, max_epochs=60, target=0.98, minibatch=256,
                device="cpu", rng=None, log=print):
    """Behaviour cloning. The actor learns to copy the rule (cross-entropy, relay left out); the critic
    learns how those missions went. Runs until the actor matches the rule on `target` of the
    decisions or `max_epochs` is reached. Returns (match rate, relay start probability)."""
    rng = rng or np.random.default_rng(0)
    obs_rows, acts, cst, rets = [], [], [], []
    for ep in episodes:
        if not ep["steps"]:
            continue
        R, disc = segment_rewards(ep, gamma)
        G = discounted_returns(R, disc)
        for k, st in enumerate(ep["steps"]):
            obs_rows.append(st["obs"]); acts.append(st["act"]); cst.append(st["cstate"]); rets.append(G[k])
    batch = {k: np.concatenate([o[k] for o in obs_rows]) for k in obs_rows[0]}
    acts_np = np.concatenate(acts).astype(np.int64)
    rets_np = np.asarray(rets, dtype=np.float32) / VALUE_SCALE
    n_dec, n_st = len(acts_np), len(cst)
    forced = float(np.mean(~batch["task_mask"][:, 0]))

    def match_all():
        actor.eval()
        hit = 0
        with torch.no_grad():
            for b0 in range(0, n_dec, 1024):
                sl = slice(b0, b0 + 1024)
                lg, _ = actor(to_t({k: v[sl] for k, v in batch.items()}, device))
                hit += int((_no_relay(lg).argmax(-1).cpu().numpy() == acts_np[sl]).sum())
        actor.train()
        return hit / max(1, n_dec)

    log(f"warm start: {n_dec} recorded decisions ({forced:.0%} have only one choice, 'head home')")
    actor.train()
    m = match_all()
    for ep_i in range(max_epochs):
        perm = rng.permutation(n_dec)
        tot = 0.0
        for b0 in range(0, n_dec, minibatch):
            idx = perm[b0:b0 + minibatch]
            logits, _ = actor(to_t({k: v[idx] for k, v in batch.items()}, device))
            a = torch.as_tensor(acts_np[idx], device=device)
            ce = -torch.distributions.Categorical(logits=_no_relay(logits)).log_prob(a).mean()
            step(opt_a, ce, actor.parameters(), 1.0)
            sidx = rng.choice(n_st, size=min(minibatch, n_st), replace=False)
            F, Mk, G = pad_states([cst[j] for j in sidx], device)
            vl = ((critic(F, Mk, G) - torch.as_tensor(rets_np[sidx], device=device)) ** 2).mean()
            step(opt_c, vl, critic.parameters(), 1.0)
            tot += float(ce.item()) * len(idx)
        m = match_all()
        if ep_i % 5 == 4 or m >= target:
            log(f"warm start epoch {ep_i + 1}: imitation loss {tot / n_dec:.3f}, matches the rule {m:.1%}")
        if m >= target:
            break
    if m < 0.95:
        log(f"WARNING: the warm start matches the rule only {m:.1%} (< 95%). Learning results are not meaningful.")
    p_relay = calibrate_relay(actor, batch, device)
    log(f"relay starts at {p_relay:.1%} average probability, so reinforcement learning can try it")
    return m, p_relay


# ------------------------------------------------------------------ training
def default_curriculum(it, total):
    """Which swarm size to train on: 3 drones for the first 20%, then 5, then 10 from halfway."""
    if it < 0.2 * total:
        return 3
    if it < 0.5 * total:
        return 5
    return 10


def sample_cfg(rng, n_drones, base_cfg, hard=True):
    """A random training farm. hard=True favours short radio and failures, where the rule is weak."""
    radios = [80, 100, 120, 150] if hard else [80, 120, 150, 200]
    fails = ([0, 1, 1, 2] if hard else [0, 0, 1, 2]) if n_drones >= 5 else [0, 0, 1]
    return base_cfg.with_(
        n_drones=n_drones,
        n_tasks=10 * n_drones,
        radio_range=float(rng.choice(radios)),
        fail_count=int(rng.choice(fails)),
        seed=int(rng.integers(1_000_000)),
        trace_every=0,
    )


def evaluate_policy(policy, n_drones=10, radio_range=150, fail_count=0, seeds=VALIDATION_SEEDS, base_cfg=None,
                    name="learned"):
    """Runs the policy on the given farm seeds (default: the validation farms, separate from the
    results' farms 0-4, so picking a checkpoint never peeks at the reported results)."""
    base_cfg = base_cfg or Config()
    out = []
    for s in seeds:
        cfg = base_cfg.with_(n_drones=n_drones, n_tasks=10 * n_drones, radio_range=radio_range,
                             fail_count=fail_count, seed=int(s), trace_every=0)
        out.append(World(cfg, LearnedSwarm(policy, name=name)).run())
    return out


def score_of(ev):
    """Higher is better: 10 per mission if every reading arrives, minus 1 per 300 s taken."""
    return float(np.mean([10 * e["delivered_frac"] - e["time_used"] / 300 for e in ev]))


def validation_score(policy, n_drones, base_cfg):
    return float(np.mean([score_of(evaluate_policy(policy, n_drones=n_drones, radio_range=r, fail_count=f,
                                                   base_cfg=base_cfg))
                          for r, f in VALIDATION_SETTINGS]))


def train(iterations=150, episodes_per_iter=6, hidden=64, lr_actor=1e-4, lr_critic=3e-4, gamma=0.995, lam=0.95,
          clip=0.2, epochs=4, minibatch=256, ent_coef=0.01, curriculum=default_curriculum, hard=True,
          save_path="runs/learned_policy.pt", device=None, seed=0, base_cfg=None, eval_every=10,
          bc_episodes=40, bc_max_epochs=60, bc_lr=1e-3, log=print):
    """Returns (actor, history). The saved checkpoint holds the best validation score; its "iter" is -1
    if the warm start (the copied rule) was never beaten.

    bc_episodes=0 turns the warm start off (training then starts from a random policy).
    The actor and the critic have separate optimisers and separate gradient clips."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    base_cfg = base_cfg or Config()
    actor, critic = Actor(hidden).to(device), Critic(hidden).to(device)
    opt_c = torch.optim.Adam(critic.parameters(), lr=lr_critic)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    history, best = [], -1e9
    t_start = time.time()
    final_n = curriculum(iterations - 1, iterations)
    val = lambda pol: validation_score(pol, final_n, base_cfg)
    ref = val(nearest_policy)
    log(f"reference: nearest-task rule scores {ref:.3f} on {len(VALIDATION_SEEDS)} validation farms x "
        f"{len(VALIDATION_SETTINGS)} settings {VALIDATION_SETTINGS} ({final_n} drones; never used for results)")

    meta = {"hidden": hidden, "rule_score": ref}
    if bc_episodes > 0:
        teach = collect_teacher(rng, base_cfg, episodes=bc_episodes)
        opt_bc = torch.optim.Adam(actor.parameters(), lr=bc_lr)
        bc_match, p_relay = pretrain_bc(actor, critic, opt_bc, opt_c, teach, gamma=gamma,
                                        max_epochs=bc_max_epochs, minibatch=minibatch, device=device, rng=rng,
                                        log=log)
        meta.update(bc_match=bc_match, relay_start=p_relay)
        actor.eval()
        best = val(make_policy(actor, device, seed=123))
        torch.save({"actor": actor.state_dict(), "critic": critic.state_dict(), "iter": -1,
                    "val_score": best, **meta}, save_path)
        log(f"   warm-start policy scores {best:.3f} (rule {ref:.3f}) -> saved as the first best: {save_path}")
    opt_a = torch.optim.Adam(actor.parameters(), lr=lr_actor)

    for it in range(iterations):
        n = curriculum(it, iterations)
        rec = Recorder()
        actor.eval()
        sampler = make_policy(actor, device, greedy=False, recorder=rec)
        for _ in range(episodes_per_iter):
            w = World(sample_cfg(rng, n, base_cfg, hard=hard), LearnedSwarm(sampler, recorder=rec))
            w.run()
            rec.finish(w)

        # ---- advantages per decision step
        obs_rows, acts, old_lp, advs, cst, rets = [], [], [], [], [], []
        for ep in rec.episodes:
            if not ep["steps"]:
                continue
            R, disc = segment_rewards(ep, gamma)
            with torch.no_grad():
                F, Mk, G = pad_states([s["cstate"] for s in ep["steps"]], device)
                V = critic(F, Mk, G).cpu().numpy() * VALUE_SCALE
            A, ret = gae(R, disc, V, lam)
            for k, s in enumerate(ep["steps"]):
                n_k = len(s["act"])
                obs_rows.append(s["obs"])
                acts.append(s["act"])
                old_lp.append(s["logp"])
                advs.append(np.full(n_k, A[k], dtype=np.float32))
                cst.append(s["cstate"])
                rets.append(ret[k])
        if not obs_rows:
            continue
        batch = {k: np.concatenate([o[k] for o in obs_rows]) for k in obs_rows[0]}
        acts_np, lp_np, adv_np = np.concatenate(acts), np.concatenate(old_lp), np.concatenate(advs)
        adv_np = (adv_np - adv_np.mean()) / (adv_np.std() + 1e-8)
        rets_np = np.asarray(rets, dtype=np.float32) / VALUE_SCALE

        # ---- PPO update
        actor.train()
        n_dec, n_st = len(acts_np), len(cst)
        for _ in range(epochs):
            perm = rng.permutation(n_dec)
            for b0 in range(0, n_dec, minibatch):
                idx = perm[b0:b0 + minibatch]
                ob = to_t({k: v[idx] for k, v in batch.items()}, device)
                logits, _ = actor(ob)
                dist = torch.distributions.Categorical(logits=logits)
                a = torch.as_tensor(acts_np[idx], device=device)
                lp = dist.log_prob(a)
                ratio = torch.exp(lp - torch.as_tensor(lp_np[idx], device=device))
                ad = torch.as_tensor(adv_np[idx], device=device)
                pg = -torch.min(ratio * ad, torch.clamp(ratio, 1 - clip, 1 + clip) * ad).mean()
                ent = dist.entropy().mean()
                step(opt_a, pg - ent_coef * ent, actor.parameters(), 0.5)
                sidx = rng.choice(n_st, size=min(minibatch, n_st), replace=False)
                F, Mk, G = pad_states([cst[j] for j in sidx], device)
                vl = ((critic(F, Mk, G) - torch.as_tensor(rets_np[sidx], device=device)) ** 2).mean()
                step(opt_c, vl, critic.parameters(), 1.0)

        mk = [ep["metrics"]["makespan"] for ep in rec.episodes]
        fin = np.mean([m is not None for m in mk])
        relay_share = float(np.mean(acts_np == RELAY_ACTION))
        row = {"iter": it, "n_drones": n, "return": float(np.mean([e["return"] for e in rec.episodes])),
               "finished": float(fin), "decisions": n_dec, "relay_share": relay_share,
               "makespan": float(np.mean([m for m in mk if m is not None])) if fin > 0 else None,
               "minutes": (time.time() - t_start) / 60}
        history.append(row)
        log(f"iter {it:3d}  drones {n:2d}  return {row['return']:6.2f}  finished {fin:4.0%}  "
            f"makespan {row['makespan'] if row['makespan'] else float('nan'):6.1f}  relay {relay_share:5.1%}  "
            f"[{row['minutes']:.1f} min]")

        if n == final_n and ((it + 1) % eval_every == 0 or it == iterations - 1):
            actor.eval()
            score = val(make_policy(actor, device, seed=123))
            row["eval_score"] = score
            log(f"   validation: {score:.3f}   (nearest-task rule {ref:.3f}, best so far {best:.3f})")
            if score > best:
                best = score
                torch.save({"actor": actor.state_dict(), "critic": critic.state_dict(), "iter": it,
                            "val_score": score, **meta}, save_path)
                log(f"   saved best policy -> {save_path}")
    if best == -1e9:  # no warm start and never evaluated: save what we have
        torch.save({"actor": actor.state_dict(), "critic": critic.state_dict(), "iter": iterations - 1,
                    "val_score": None, **meta}, save_path)
    return actor, history
