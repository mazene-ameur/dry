"""
SA-QL-GSO v5: Semantic-Aware, Agent-Assisted Q-Learning + PSO for Perching UAV Relays
=====================================================================================

Changes relative to v4:
  * Bug fix: agent PSO trigger and beta_D were unreachable (episode ended before them).
  * Scenario recalibrated so the relay mesh is actually stressed (urban A2G, 20 dBm,
    h <= 120 m). In v4 a single UAV covered every user, so N and faults had no effect.
  * No teleporting: UAVs fly to targets and perch sites at finite speed; propulsion
    energy follows the Zeng et al. rotary-wing model P(V).
  * Lifetime = time until sustained QoS outage (> T_out s). Depleted UAVs leave the mesh
    instead of ending the episode, so perching and rotation strategies matter.
  * Common initial deployment and a common safety shield for every method (fair).
  * Proper Q-learning: true next-state bootstrapping, terminal penalty, parameter sharing.
  * Agent: event-triggered PSO, energy-aware perch handover, reward reweighting, and its
    inference energy is deducted from a real UAV battery.
  * Ablations (+Sem, +Agent) so every claimed contribution is isolated.
  * Paired seeds, 95% t-intervals, Wilcoxon tests, LaTeX table, IEEE-ready vector figures.
  * Agent reward re-weighting (v4 behaviour) is kept only as the ablation "SA+Shaping":
    in a controlled test it slightly reduced lifetime and increased variance.

Debugging: set SAQL_TRACE=1 to log perch, handover, departure and failure events.

Usage:
    python sa_qlgso_v5.py            # full run
    python sa_qlgso_v5.py --quick    # fast smoke test
    python sa_qlgso_v5.py --plot-only  # re-plot from results/results.pkl
"""
import os
import sys
import time
import json
import pickle
import argparse
import warnings
from types import SimpleNamespace

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

warnings.filterwarnings("ignore")
TRACE = []


def _tr(*a):
    if TRACE is not None and len(TRACE) < 400 and os.environ.get("SAQL_TRACE"):
        TRACE.append(" ".join(str(x) for x in a))


# =====================================================================================
# 1. CONFIGURATION
# =====================================================================================
def make_cfg(**ov):
    C = SimpleNamespace(
        # geography
        area=1500.0, h_min=30.0, h_max=120.0, h_takeoff=80.0, d_sep=20.0, margin=100.0,
        M=6, n_sites=40, site_h=(10.0, 40.0), R_perch=300.0, site_near=0.5, site_sigma=120.0,
        # radio: urban A2G (Al-Hourani), 2.4 GHz, 20 dBm
        fc=2.4e9, B=10e6, Pt_dBm=20.0, NF_dB=9.0, N0_dBm_Hz=-174.0,
        a_los=9.61, b_los=0.16, eta_los=1.0, eta_nlos=20.0,
        gamma_th_dB=10.0, R_conv_peak=35.0, R_sem_peak=30.0,
        sem_snr0=1.0, sem_k=2.0,
        conv_model="fixed",           # "fixed" (single MCS, as in the paper) or "amc"
        # semantic task quality
        Q_R0=12.0, Q_k=1.2, Q_min=0.65, Q_warn=0.85, Q_shield=0.80,
        # energy
        E_batt=120e3, E_het=0.25, E_res=4e3,
        P_perch=15.0, P_comm=5.0, P_sem=2.0, P_llm=8.0, t_llm=0.5,
        v_fly=15.0,
        # timing
        dt=1.0, T_mission=7200, T_PSO=120, T_LLM=30, T_out=30, T_dwell=20,
        T_retry=5, agent_cooldown=5,
        # PSO (online / initial deployment)
        K=8, I_pso=8, K0=16, I0=25, w_inert=0.729, c1=1.494, c2=1.494,
        v_max_it=250.0, lam_c=10.0, lam_m=2e-4, lam_s=5.0, lam_r=0.05, lam_e=0.5,
        # Q-learning (shared table, decentralised execution)
        alpha=0.1, gamma_d=0.98, eps0=1.0, eps_min=0.05, kappa_eps=0.8,
        NB=5, NR=5, NS=5,
        # reward
        beta_E=1.0, beta_Q=1.0, beta_B=2.0, beta_D=5.0, beta_T=50.0, beta_sw=3.0, q_hyst=0.2,
        # agent handover thresholds
        ho_low=0.35, ho_gap=0.20, ho_backoff=120, agent_shaping=False,
        Q_opt=0.90, P_sleep=0.5, rt_low=0.30, rt_backoff=200,
    )
    for k, v in ov.items():
        if not hasattr(C, k):
            raise KeyError(f"unknown config key {k}")
        setattr(C, k, v)
    C.n_states = C.NB * C.NR * C.NS * 2 * 2 * 2
    return C


# Method specification: every method shares channel, flow solver, initial deployment,
# safety shield and batteries. Only policy / codec / PSO / agent differ.
METHODS = {
    "Random":       dict(policy="random", codec="conv", pso=True,  agent=False),
    "Greedy":       dict(policy="greedy", codec="conv", pso=True,  agent=False),
    "Static-Opt":   dict(policy="static", codec="conv", pso=True,  agent=False),
    "PSO-Only":     dict(policy="fly",    codec="conv", pso=True,  agent=False),
    "MARL":         dict(policy="ql",     codec="conv", pso=False, agent=False),
    "QL-GSO":       dict(policy="ql",     codec="conv", pso=True,  agent=False),
    "QL-GSO+Sem":   dict(policy="ql",     codec="sem",  pso=True,  agent=False),
    "QL-GSO+Agent": dict(policy="ql",     codec="conv", pso=True,  agent=True),
    "SA-base":      dict(policy="ql",     codec="sem",  pso=True,  agent=True),
    "SA+Gate":      dict(policy="ql",     codec="sem",  pso=True,  agent=True, gate=True),
    "SA+Gate+Sleep": dict(policy="ql",    codec="sem",  pso=True,  agent=True, gate=True,
                          sleep=True),
    # proposed: agent also gates encoders, sleeps redundant radios, transfers relay roles
    "SA-QL-GSO":    dict(policy="ql",     codec="sem",  pso=True,  agent=True, gate=True,
                         sleep=True, relay_ho=True),
    # no-learning ablation: identical agent, Greedy reflex instead of Q-learning
    "SA-Greedy":    dict(policy="greedy", codec="sem",  pso=True,  agent=True, gate=True,
                         sleep=True, relay_ho=True),
    # ablation: agent additionally re-weights the reward by mission phase (v4 behaviour)
    "SA+Shaping":   dict(policy="ql",     codec="sem",  pso=True,  agent=True, shaping=True),
}
MAIN_METHODS = ["Random", "Greedy", "Static-Opt", "PSO-Only", "MARL", "QL-GSO", "SA-QL-GSO"]
ABL_METHODS = ["QL-GSO", "QL-GSO+Sem", "QL-GSO+Agent", "SA-QL-GSO"]
EXTRA_METHODS = ["QL-GSO+Sem", "QL-GSO+Agent", "SA-base", "SA+Gate", "SA+Gate+Sleep", "SA-Greedy"]
LEARNED = ["MARL", "QL-GSO", "QL-GSO+Sem", "QL-GSO+Agent", "SA-QL-GSO"]
# agent-tool ablations reuse the proposed method's trained reflex (isolates the tools)
Q_ALIAS = {"SA-base": "SA-QL-GSO", "SA+Gate": "SA-QL-GSO", "SA+Gate+Sleep": "SA-QL-GSO",
           "SA+Shaping": "SA-QL-GSO"}
CHAIN = ["QL-GSO", "QL-GSO+Agent", "QL-GSO+Sem", "SA-base", "SA+Gate", "SA+Gate+Sleep",
         "SA-QL-GSO", "SA-Greedy"]


# =====================================================================================
# 2. PROPULSION, CHANNEL, SEMANTIC MODELS
# =====================================================================================
def rotary_power(V):
    """Zeng, Xu, Zhang, IEEE TWC 2019, rotary-wing propulsion power (W) at speed V (m/s)."""
    P0, Pi, Utip, v0, d0, rho, s, A = 79.86, 88.63, 120.0, 4.03, 0.6, 1.225, 0.05, 0.503
    V = np.asarray(V, dtype=float)
    blade = P0 * (1.0 + 3.0 * V ** 2 / Utip ** 2)
    induced = Pi * np.sqrt(np.sqrt(1.0 + V ** 4 / (4.0 * v0 ** 4)) - V ** 2 / (2.0 * v0 ** 2))
    parasite = 0.5 * d0 * rho * s * A * V ** 3
    return blade + induced + parasite


P_HOV = float(rotary_power(0.0))   # about 168.5 W

# LTE CQI-like ladder (SNR threshold dB, spectral efficiency bit/s/Hz) for the AMC option
MCS = np.array([(-6.7, 0.15), (-4.7, 0.23), (-2.3, 0.38), (0.2, 0.60), (2.4, 0.88),
                (4.3, 1.18), (5.9, 1.48), (8.1, 1.91), (10.3, 2.41), (11.7, 2.73),
                (14.1, 3.32), (16.3, 3.90), (18.7, 4.52), (21.0, 5.12), (22.7, 5.55)])


def _fspl(d, C):
    return 20 * np.log10(d) + 20 * np.log10(C.fc) + 20 * np.log10(4 * np.pi / 3e8)


def _noise(C):
    return C.N0_dBm_Hz + 10 * np.log10(C.B) + C.NF_dB


def snr_a2g(C, uav, g):
    """uav (...,N,3), g (M,2) -> SNR dB (...,N,M). Al-Hourani probabilistic LoS."""
    dx = uav[..., :, None, 0] - g[:, 0]
    dy = uav[..., :, None, 1] - g[:, 1]
    r = np.sqrt(dx ** 2 + dy ** 2) + 1e-6
    h = np.broadcast_to(uav[..., :, None, 2], r.shape)
    d = np.sqrt(r ** 2 + h ** 2)
    th = np.degrees(np.arctan2(h, r))
    p_los = 1.0 / (1.0 + C.a_los * np.exp(-C.b_los * (th - C.a_los)))
    PL = _fspl(d, C) + C.eta_los + (1.0 - p_los) * (C.eta_nlos - C.eta_los)
    return C.Pt_dBm - PL - _noise(C)


def snr_u2u(C, uav):
    """uav (...,N,3) -> SNR dB (...,N,N), free-space U2U."""
    diff = uav[..., :, None, :] - uav[..., None, :, :]
    d = np.sqrt(np.sum(diff ** 2, axis=-1)) + 1e-6
    return C.Pt_dBm - _fspl(d, C) - _noise(C)


def link_rate(C, snr, codec):
    """Effective link rate (Mbps). 'sem' is a DeepJSCC-style task-equivalent rate."""
    snr = np.asarray(snr, dtype=float)
    if codec == "sem":
        return C.R_sem_peak / (1.0 + np.exp(-(snr - C.sem_snr0) / C.sem_k))
    if C.conv_model == "amc":
        idx = np.searchsorted(MCS[:, 0], snr, side="right") - 1
        se = np.where(idx >= 0, MCS[np.clip(idx, 0, None), 1], 0.0)
        return np.minimum(C.B * se / 1e6, C.R_conv_peak)
    R = C.B * np.log2(1.0 + 10 ** (np.minimum(snr, 60.0) / 10)) / 1e6
    return np.where(snr >= C.gamma_th_dB, np.minimum(R, C.R_conv_peak), 0.0)


def hybrid_rate(C, snr, codec, kind):
    """codec: 'conv', 'sem' (all encoders on) or a bool mask (N,) of UAVs whose semantic
    encoder is on. A semantic-capable hop uses the better of the two transceivers."""
    rc = link_rate(C, snr, "conv")
    if isinstance(codec, str):
        if codec == "conv":
            return rc
        return np.maximum(rc, link_rate(C, snr, "sem"))
    m = np.asarray(codec, bool)
    cap = m[..., :, None] if kind == "a2g" else (m[..., :, None] & m[..., None, :])
    return np.where(cap, np.maximum(rc, link_rate(C, snr, "sem")), rc)


def q_from_rate(C, R):
    return 1.0 / (1.0 + np.exp(-(np.maximum(R, 0.0) - C.Q_R0) / C.Q_k))


# =====================================================================================
# 3. FLOW SOLVER: all-pairs widest paths, only UAVs may relay (vectorised, batched)
# =====================================================================================
BIG = 1e9


def flow_rates(C, uav, g, flows, alive, codec):
    """uav (...,N,3); alive (N,) bool. Returns bottleneck rates (...,F) in Mbps.
    Bottleneck Floyd-Warshall restricted to UAV intermediates, so ground users never relay."""
    N = uav.shape[-2]
    a = alive.astype(float)
    Ra = hybrid_rate(C, snr_a2g(C, uav, g), codec, "a2g") * a[..., :, None]
    W = hybrid_rate(C, snr_u2u(C, uav), codec, "u2u") * a[..., :, None] * a[..., None, :]
    idx = np.arange(N)
    W[..., idx, idx] = BIG
    for k in range(N):
        W = np.maximum(W, np.minimum(W[..., :, k:k + 1], W[..., k:k + 1, :]))
    out = []
    for (s, d) in flows:
        T = np.minimum(Ra[..., :, s][..., :, None], W)
        T = np.minimum(T, Ra[..., :, d][..., None, :])
        out.append(T.max(axis=(-1, -2)))
    return np.stack(out, axis=-1)


def net_quality(C, uav, g, flows, alive, codec):
    rates = flow_rates(C, uav, g, flows, alive, codec)
    return float(np.min(q_from_rate(C, rates))), rates


# =====================================================================================
# 4. PSO (batched fitness; incumbent is always a particle, so PSO never degrades)
# =====================================================================================
def pso_fitness(C, X, cfg_pos, cur_pos, fly, g, flows, alive, codec, weak=None):
    K, Nf = X.shape[0], len(fly)
    P = np.broadcast_to(cfg_pos, (K,) + cfg_pos.shape).copy()
    P[:, fly] = X.reshape(K, Nf, 3)
    rates = flow_rates(C, P, g, flows, alive, codec)                  # (K,F)
    q = q_from_rate(C, rates)
    score = q.min(axis=1) + C.lam_r * np.mean(np.minimum(rates, 30.0) / 30.0, axis=1)
    if weak is not None and weak.any():
        # energy-aware term: keep QoS if the low-battery relays leave (pre-positions spares)
        rw = flow_rates(C, P, g, flows, alive & ~weak, codec)
        score = score + C.lam_e * q_from_rate(C, rw).min(axis=1)
    c_disc = np.sum(rates < 1.0, axis=1)
    c_mot = np.sum(np.linalg.norm(P[:, fly] - cur_pos[fly], axis=-1), axis=1)
    if Nf > 1:
        dd = np.linalg.norm(P[:, fly, None, :] - P[:, None, fly, :], axis=-1)
        iu = np.triu_indices(Nf, 1)
        c_sep = np.sum(np.maximum(C.d_sep - dd[:, iu[0], iu[1]], 0.0), axis=1)
    else:
        c_sep = 0.0
    return score - C.lam_c * c_disc - C.lam_m * c_mot - C.lam_s * c_sep


def swap_seeds(C, cfg_pos, fly, weak, n_max=4):
    """Role-swap particles: a healthy flyer takes over the spot of a low-battery relay.
    PSO is a local search; without these it cannot discover long-range handovers."""
    out = []
    fly = np.asarray(fly, dtype=int)
    healthy = [j for j in fly if not weak[j]]
    for w in np.where(weak)[0]:
        for j in sorted(healthy, key=lambda j: np.linalg.norm(cfg_pos[j, :2] - cfg_pos[w, :2]))[:2]:
            P = cfg_pos.copy()
            P[j] = cfg_pos[w] + np.array([C.d_sep, 0.0, 0.0])
            out.append(P)
    return out[:n_max]


def pso_solve(C, cfg_pos, cur_pos, fly, g, flows, alive, codec, rng,
              K=None, I=None, spread=120.0, seeds=None, weak=None):
    """Optimise 3D targets of UAVs in `fly`; others fixed at cfg_pos. Returns targets."""
    K = K or C.K
    I = I or C.I_pso
    fly = np.asarray(fly, dtype=int)
    Nf = len(fly)
    if Nf == 0:
        return cfg_pos.copy()
    lo = np.tile([C.margin, C.margin, C.h_min], Nf)
    hi = np.tile([C.area - C.margin, C.area - C.margin, C.h_max], Nf)
    base = np.clip(cfg_pos[fly].flatten(), lo, hi)
    X = np.clip(base + rng.uniform(-spread, spread, (K, 3 * Nf)), lo, hi)
    X[0] = base                                   # incumbent
    if weak is not None and weak.any():
        seeds = list(seeds or []) + swap_seeds(C, cfg_pos, fly, weak)
    if seeds:
        for j, sd in enumerate(seeds[:K - 1]):
            X[j + 1] = np.clip(sd[fly].flatten(), lo, hi)
    V = rng.uniform(-5, 5, X.shape)
    f = pso_fitness(C, X, cfg_pos, cur_pos, fly, g, flows, alive, codec, weak)
    pb, pbv = X.copy(), f.copy()
    gi = int(np.argmax(pbv))
    gb, gv = pb[gi].copy(), pbv[gi]
    for _ in range(I):
        r1, r2 = rng.rand(*X.shape), rng.rand(*X.shape)
        V = C.w_inert * V + C.c1 * r1 * (pb - X) + C.c2 * r2 * (gb - X)
        V = np.clip(V, -C.v_max_it, C.v_max_it)
        X = np.clip(X + V, lo, hi)
        f = pso_fitness(C, X, cfg_pos, cur_pos, fly, g, flows, alive, codec, weak)
        imp = f > pbv
        pb[imp], pbv[imp] = X[imp], f[imp]
        gi = int(np.argmax(pbv))
        if pbv[gi] > gv:
            gv, gb = pbv[gi], pb[gi].copy()
    out = cfg_pos.copy()
    out[fly] = gb.reshape(Nf, 3)
    return out


# =====================================================================================
# 5. LAYOUT AND INITIAL DEPLOYMENT (common to all methods)
# =====================================================================================
ANCHORS = np.array([[0.05, 0.05], [0.95, 0.05], [0.05, 0.95], [0.95, 0.95],
                    [0.50, 0.05], [0.50, 0.95], [0.05, 0.50], [0.95, 0.50]])
FLOWS = [(0, 3), (1, 2), (4, 5)]


def make_layout(C, N, seed):
    """Identical for every method given the seed (paired comparisons)."""
    rng = np.random.RandomState(seed)
    g = ANCHORS[:C.M] * C.area + rng.uniform(-40, 40, (C.M, 2))
    # perch structures: a fraction clustered around user hotspots (buildings are where
    # users are), the rest uniform over the area
    n_near = int(round(C.site_near * C.n_sites))
    near = g[rng.randint(C.M, size=n_near)] + rng.normal(0, C.site_sigma, (n_near, 2))
    unif = rng.uniform(C.margin, C.area - C.margin, (C.n_sites - n_near, 2))
    xy = np.clip(np.vstack([near, unif]), 20.0, C.area - 20.0)
    sites = np.column_stack([xy, rng.uniform(*C.site_h, C.n_sites)])
    E0 = C.E_batt * rng.uniform(1 - C.E_het, 1 + C.E_het, N)
    return g, sites, list(FLOWS), E0


def heuristic_seed(C, N, g):
    """Access UAV above each user (pulled toward centre), extra UAVs as backbone relays."""
    ctr = np.array([C.area / 2, C.area / 2])
    pts = [g[m] + 0.12 * (ctr - g[m]) for m in range(C.M)]
    extra = [ctr, [ctr[0], 0.27 * C.area], [ctr[0], 0.73 * C.area],
             [0.27 * C.area, ctr[1]], [0.73 * C.area, ctr[1]],
             [0.3 * C.area, 0.3 * C.area], [0.7 * C.area, 0.7 * C.area],
             [0.3 * C.area, 0.7 * C.area], [0.7 * C.area, 0.3 * C.area]]
    pts = (pts + [np.asarray(e) for e in extra] * 3)[:N]
    xy = np.clip(np.array(pts), C.margin, C.area - C.margin)
    return np.column_stack([xy, np.full(N, C.h_max)])


def initial_deployment(C, N, g, flows, codec, seed):
    rng = np.random.RandomState(seed + 7777)
    alive = np.ones(N, bool)
    seed_pos = heuristic_seed(C, N, g)
    rnd = np.column_stack([rng.uniform(C.margin, C.area - C.margin, (N, 2)),
                           rng.uniform(C.h_min, C.h_max, N)])
    pos = pso_solve(C, seed_pos, seed_pos, np.arange(N), g, flows, alive, codec, rng,
                    K=C.K0, I=C.I0, spread=150.0, seeds=[rnd])
    return pos


# =====================================================================================
# 6. AGENTIC META-CONTROLLER (rule-based stand-in for an LLM planner with tools)
# =====================================================================================
class Agent:
    """Slow-timescale strategist. Invoked every T_LLM s, on topology events (fault or
    battery departure) or on QoS early warning. Each invocation costs P_llm * t_llm J,
    deducted from the battery of the UAV hosting the model (highest-energy alive UAV)."""

    def __init__(self, C):
        self.C = C
        self.calls = 0
        self.energy = 0.0
        self.beta_E, self.beta_Q = C.beta_E, C.beta_Q

    def invoke(self, E, alive):
        C = self.C
        self.calls += 1
        cost = C.P_llm * C.t_llm
        self.energy += cost
        host = int(np.argmax(np.where(alive, E, -np.inf)))
        E[host] -= cost
        f = float(np.mean(E[alive])) / C.E_batt
        if f > 0.7:
            self.beta_E, self.beta_Q = 1.5, 0.5      # early: harvest energy savings
        elif f > 0.3:
            self.beta_E, self.beta_Q = 1.0, 1.0
        else:
            self.beta_E, self.beta_Q = 0.5, 2.0      # late: protect connectivity
        return cost


# =====================================================================================
# 7. EPISODE SIMULATOR
# =====================================================================================
def encode_states(C, E, alive, mode, snr_uu_mat, q_now, stress):
    N = len(E)
    Ebar = max(float(np.mean(E[alive])), 1.0) if alive.any() else 1.0
    b = np.clip((E / C.E_batt * C.NB).astype(int), 0, C.NB - 1)
    rel = np.clip(((E / Ebar - 0.6) / 0.8 * C.NR).astype(int), 0, C.NR - 1)
    S = snr_uu_mat.copy()
    S[:, ~alive] = -np.inf
    np.fill_diagonal(S, -np.inf)
    s2 = np.sort(S, axis=1)[:, -2] if N >= 3 else S.max(axis=1)
    s2 = np.where(np.isfinite(s2), s2, 0.0)
    loc = np.clip((s2 / 30.0 * C.NS).astype(int), 0, C.NS - 1)
    c = int(q_now >= C.Q_warn)
    m = (mode > 0).astype(int)
    return ((((b * C.NR + rel) * C.NS + loc) * 2 + c) * 2 + int(stress)) * 2 + m


def run_episode(C, N, seed, method, Q=None, train=False, eps=0.0, faults=(),
                record=True):
    spec = METHODS[method]
    codec, policy = spec["codec"], spec["policy"]
    use_pso, use_agent = spec["pso"], spec["agent"]
    use_shaping = spec.get("shaping", C.agent_shaping)
    g, sites, flows, E0 = make_layout(C, N, seed)
    rng = np.random.RandomState(seed + 10 ** 6)

    pos = initial_deployment(C, N, g, flows, codec, seed)
    target = pos.copy()
    E = E0.copy()
    alive = np.ones(N, bool)
    mode = np.zeros(N, int)            # 0 fly, 1 perched, 2 transit-to-perch
    site_of = -np.ones(N, int)
    lock = np.zeros(N)
    agent = Agent(C) if use_agent else None
    if Q is None and policy == "ql":
        Q = np.zeros((C.n_states, 2))
    learn = train and policy == "ql"

    T = int(C.T_mission)
    hist = {k: np.full(T, np.nan) for k in ("q", "r", "pr", "alive", "sleep", "enc")} if record else None
    E_prop = E_comm = E_sem = E_llm = 0.0
    out_cnt, last_ok, life, failed = 0, 0, T, False
    last_pso = last_agent = last_trig = -10 ** 9
    last_pso = 0
    pso_calls = 0
    prev_s = -np.ones(N, int)
    prev_a = np.zeros(N, int)
    prev_r = np.zeros(N)
    fault_log = []                      # [t_fault, t_recovered or None, dipped]
    pending = []                        # (relief p, relieved f, site k, t_start)
    backoff = np.zeros(N)
    awake = np.ones(N, bool)            # radio on (sleeping UAVs are perched, radio off)
    sem_on = np.ones(N, bool)           # semantic encoder on (only used if codec == 'sem')
    gate, sleep_on, relay_ho = spec.get("gate", False), spec.get("sleep", False), spec.get("relay_ho", False)
    E_sleep_saved = 0.0

    def cm(mask=None):
        if codec == "conv":
            return "conv"
        return (sem_on if mask is None else mask) if gate else "sem"

    def cmp():
        # planning assumes full capability: the agent can switch encoders back on
        return "conv" if codec == "conv" else "sem"

    def netq(cfg, net=None, mask=None):
        return net_quality(C, cfg, g, flows, (alive & awake) if net is None else net, cm(mask))[0]

    def wake_all(t):
        sl = alive & ~awake
        awake[sl] = True
        lock[sl] = t

    def optimise_radio(t, busy):
        """Greedy agent tool: sleep redundant perched radios, then switch off encoders,
        keeping QoS >= Q_opt for both the current and the intended configuration."""
        if sleep_on:
            for i in sorted(np.where(alive & awake & (mode == 1))[0], key=lambda i: -E[i]):
                if i in busy:
                    continue
                net = alive & awake
                net[i] = False
                if min(netq(pos, net), netq(target, net)) >= C.Q_opt:
                    awake[i] = False
                    lock[i] = np.inf
        if gate and codec == "sem":
            for i in sorted(np.where(alive & awake & sem_on)[0], key=lambda i: E[i]):
                m = sem_on.copy()
                m[i] = False
                if min(netq(pos, mask=m), netq(target, mask=m)) >= C.Q_opt:
                    sem_on[i] = False
    q_sum, q_n = 0.0, 0

    def free_sites(exclude=None):
        taken = set(site_of[(site_of >= 0) & alive].tolist())
        if exclude is not None:
            taken.discard(exclude)
        return np.array([k for k in range(C.n_sites) if k not in taken], dtype=int)

    def try_perch(i, radius, cfg):
        fs = free_sites()
        if len(fs) == 0 or E[i] <= 2 * C.E_res:
            return -1
        d = np.linalg.norm(sites[fs, :2] - pos[i, :2], axis=1)
        ok = fs[d <= radius]
        if len(ok) == 0:
            return -1
        k = int(ok[np.argmin(np.linalg.norm(sites[ok, :2] - pos[i, :2], axis=1))])
        trial = cfg.copy()
        trial[i] = sites[k]
        qa, _ = net_quality(C, trial, g, flows, alive & awake, cmp())
        return k if qa >= C.Q_shield else -1

    def start_perch(i, k, t):
        site_of[i] = k
        target[i] = sites[k]
        mode[i] = 2
        lock[i] = t + C.T_dwell

    def unperch(i, t):
        awake[i] = True
        site_of[i] = -1
        target[i] = np.array([pos[i, 0], pos[i, 1], C.h_takeoff])
        mode[i] = 0
        lock[i] = t + C.T_dwell

    for t in range(T):
        event = False
        lost_now = []
        # ---- faults: kill k relays, flying ones first, then perched (lowest index) ----
        for (tf, k) in faults:
            if t == tf:
                cand = ([i for i in range(N) if alive[i] and mode[i] != 1]
                        + [i for i in range(N) if alive[i] and mode[i] == 1])[:k]
                for i in cand:
                    lost_now.append(pos[i].copy())
                    alive[i] = False
                    site_of[i] = -1
                event = True
                fault_log.append([t, None, False])
        # ---- battery departures (return-to-base at reserve) ----
        dep = alive & (E <= C.E_res)
        if dep.any():
            for i in np.where(dep)[0]:
                if learn and prev_s[i] >= 0:
                    Q[prev_s[i], prev_a[i]] += C.alpha * (prev_r[i] - Q[prev_s[i], prev_a[i]])
                    prev_s[i] = -1
            _tr(t, "DEPART", np.where(dep)[0].tolist(), "mode", mode[dep].tolist())
            lost_now += [pos[i].copy() for i in np.where(dep)[0]]
            alive[dep] = False
            site_of[dep] = -1
            event = True
        if not alive.any():
            life, failed = last_ok, True
            break

        # ---- kinematics ----
        mv = alive & (mode != 1)
        delta = target - pos
        dist = np.linalg.norm(delta, axis=1)
        step = np.minimum(dist, C.v_fly * C.dt)
        with np.errstate(invalid="ignore", divide="ignore"):
            pos[mv] += (delta[mv] / np.maximum(dist[mv], 1e-9)[:, None]) * step[mv][:, None]
        speed = np.where(mv, step / C.dt, 0.0)
        arrived = (mode == 2) & alive & (dist - step < 1.0)
        mode[arrived] = 1
        pos[arrived] = target[arrived]

        # ---- network state ----
        q_now, rates = net_quality(C, pos, g, flows, alive & awake, cm())
        rmin = float(np.min(rates))
        if record:
            hist["q"][t], hist["r"][t] = q_now, rmin
            hist["pr"][t] = np.sum(alive & (mode == 1)) / N
            hist["alive"][t] = alive.mean()
            hist["sleep"][t] = np.sum(alive & ~awake) / N
            hist["enc"][t] = (np.sum(alive & awake & sem_on) / max(np.sum(alive & awake), 1)
                              if codec == "sem" else 0.0)
        q_sum += q_now
        q_n += 1
        for ev in fault_log:
            if q_now < C.Q_warn:
                ev[2] = True
            if ev[1] is None and ev[2] and q_now >= C.Q_warn:
                ev[1] = t - ev[0]
        if q_now < C.Q_min:
            out_cnt += 1
        else:
            out_cnt, last_ok = 0, t
        if out_cnt > C.T_out:
            _tr(t, "FAIL rates", np.round(rates, 1).tolist(), "alive", int(alive.sum()),
                "modes", mode.tolist(), "E", (E / 1e3).astype(int).tolist())
            life, failed = last_ok, True
            if learn:
                for i in range(N):
                    if prev_s[i] >= 0:
                        Q[prev_s[i], prev_a[i]] += C.alpha * (
                            prev_r[i] - C.beta_T - Q[prev_s[i], prev_a[i]])
            break

        # ---- agent (slow timescale, event driven) ----
        force_pso = False
        weak = alive & (mode == 0) & (E < C.ho_low * C.E_batt)   # flying only (not in transit)
        if use_agent:
            # complete pending make-before-break handovers once the relief UAV arrives
            for ho in list(pending):
                p_, f_, k_, t0_ = ho
                if not (alive[p_] and alive[f_]) or (site_of == k_).any():
                    pending.remove(ho)
                    lock[f_] = t
                    continue
                if np.linalg.norm(target[p_] - pos[p_]) < 5.0 or t - t0_ > 120:
                    trial = target.copy()
                    trial[f_] = sites[k_]
                    if net_quality(C, trial, g, flows, alive & awake, cmp())[0] >= C.Q_shield:
                        start_perch(f_, k_, t)
                        _tr(t, "HANDOVER done", p_, f_)
                    else:
                        # failed: release f, send the relief UAV back to a perch, back off
                        lock[f_] = t
                        backoff[f_] = t + C.ho_backoff
                        kk = try_perch(p_, C.R_perch, target)
                        if kk >= 0:
                            start_perch(p_, kk, t)
                        _tr(t, "HANDOVER failed", p_, f_)
                    pending.remove(ho)
            busy = {x for ho in pending for x in ho[:2]}
            low_evt = bool(np.any(weak & ~np.isin(np.arange(N), list(busy)) & (backoff <= t)))
            warn = (q_now < C.Q_warn) and (t - last_trig >= C.agent_cooldown)
            if (t - last_agent >= C.T_LLM) or event or warn or (low_evt and t - last_agent >= 5):
                E_llm += agent.invoke(E, alive)
                last_agent = t
                if event or warn:
                    force_pso, last_trig = True, t
                    wake_all(t)
                    sem_on[:] = True
                # failure recovery: rested perched UAVs nearest to the lost relays take off
                # and PSO is seeded with them at the lost positions
                if lost_now and q_now < C.Q_warn:
                    busy0 = {x for ho in pending for x in ho[:2]}
                    pe = [i for i in np.where(alive & (mode == 1))[0]
                          if i not in busy0 and E[i] > 0.25 * C.E_batt]
                    chosen = []
                    for lp in lost_now:
                        if not pe:
                            break
                        j = min(pe, key=lambda i: np.linalg.norm(pos[i, :2] - lp[:2]))
                        chosen.append((j, lp))
                        pe.remove(j)
                    if chosen:
                        for j, _ in chosen:
                            unperch(j, t)
                        fly = np.where(alive & (mode == 0))[0]
                        sw = target.copy()
                        for j, lp in chosen:
                            sw[j] = np.array([lp[0], lp[1], max(lp[2], C.h_takeoff)])
                        target = pso_solve(C, target, pos, fly, g, flows, alive & awake, cmp(), rng,
                                           seeds=[sw], weak=weak)
                        pso_calls += 1
                        force_pso, last_pso = False, t
                        _tr(t, "RECOVERY", [j for j, _ in chosen])
                # energy-aware handover (make-before-break): nearest rested perched UAV
                # flies to relieve the weakest flyer; the flyer perches only after arrival
                fl = [i for i in np.where(weak)[0] if i not in busy and backoff[i] <= t]
                pe = [i for i in np.where(alive & (mode == 1) & ((lock <= t) | ~awake))[0]
                      if i not in busy]
                for f in sorted(fl, key=lambda i: E[i])[:2]:
                    cand = [p for p in pe if E[p] - E[f] > C.ho_gap * C.E_batt]
                    if not cand:
                        break
                    p = min(cand, key=lambda i: np.linalg.norm(pos[i, :2] - pos[f, :2]))
                    fs = free_sites()
                    fs = fs[fs != site_of[p]]
                    if not len(fs):
                        break
                    d = np.linalg.norm(sites[fs, :2] - pos[f, :2], axis=1)
                    if d.min() > 2 * C.R_perch:
                        continue
                    k = int(fs[np.argmin(d)])
                    cfg = target.copy()
                    cfg[f] = sites[k]
                    cfg[p] = np.array([pos[p, 0], pos[p, 1], C.h_takeoff])
                    was_awake = awake[p]
                    awake[p] = True                      # a sleeping relief must count in the plan
                    fly = np.where(alive & (mode == 0))[0]
                    fly = np.append(fly[fly != f], p)
                    sw = cfg.copy()
                    sw[p] = pos[f]
                    plan = pso_solve(C, cfg, pos, fly, g, flows, alive & awake, cmp(), rng,
                                     seeds=[sw], weak=weak & (np.arange(N) != f))
                    pso_calls += 1
                    ok_plan = net_quality(C, plan, g, flows, alive & awake, cmp())[0] >= C.Q_shield
                    if not ok_plan:
                        awake[p] = was_awake
                    if ok_plan:
                        unperch(p, t)
                        target[fly] = plan[fly]
                        lock[f] = t + 10 ** 6            # f waits for its relief
                        pending.append((p, f, k, t))
                        _tr(t, "HANDOVER p", p, "f", f, "Ef", int(E[f]), "Ep", int(E[p]))
                        pe.remove(p)
                        busy |= {p, f}
                        force_pso = False
                        last_pso = t
                # perch-to-perch role transfer: a rested UAV perches near a critical relay
                # whose battery is low; the old relay becomes redundant (then sleeps)
                if relay_ho and not (event or warn):
                    full = np.ones(N, bool)
                    net0 = alive & awake
                    wl = [w for w in np.where(net0 & (mode == 1) & (E < C.rt_low * C.E_batt)
                                              & (backoff <= t))[0] if w not in busy]
                    for w in sorted(wl, key=lambda i: E[i])[:2]:
                        nw = net0.copy()
                        nw[w] = False
                        if netq(target, nw, full) >= C.Q_shield:
                            continue                      # not critical
                        backoff[w] = t + C.rt_backoff
                        cand = [p for p in np.where(alive & (mode == 1))[0]
                                if p != w and p not in busy and E[p] - E[w] > C.ho_gap * C.E_batt]
                        fs = free_sites()
                        if not cand or not len(fs):
                            continue
                        fs = fs[np.argsort(np.linalg.norm(sites[fs, :2] - pos[w, :2], axis=1))[:6]]
                        done = False
                        for k in fs:
                            for p in sorted(cand, key=lambda i: np.linalg.norm(pos[i, :2] - sites[k, :2]))[:3]:
                                cfg = target.copy()
                                cfg[p] = sites[k]
                                net = nw.copy()
                                net[p] = True
                                if netq(cfg, net, full) >= C.Q_shield:
                                    unperch(p, t)
                                    start_perch(p, int(k), t)
                                    busy |= {p, w}
                                    done = True
                                    _tr(t, "ROLE-TRANSFER", p, "->", w)
                                    break
                            if done:
                                break
                if (gate or sleep_on) and not (event or warn):
                    optimise_radio(t, busy)

        # ---- fast timescale: per-UAV perch decisions + common safety shield ----
        mode_before = (mode > 0).astype(int)
        ql = policy == "ql"
        if ql or learn:
            S_uu = snr_u2u(C, pos)
            s_now = encode_states(C, E, alive, mode, S_uu, q_now, weak.any())
        if learn:
            for i in range(N):
                if alive[i] and prev_s[i] >= 0:
                    Q[prev_s[i], prev_a[i]] += C.alpha * (
                        prev_r[i] + C.gamma_d * Q[s_now[i]].max() - Q[prev_s[i], prev_a[i]])
        for i in range(N):
            if not alive[i]:
                continue
            if t < lock[i] or mode[i] == 2 or not awake[i]:
                continue
            if policy == "random":
                a = rng.randint(2)
            elif policy == "fly":
                a = 0
            elif policy == "static":
                a = 1 if i < N // 2 else 0
            elif policy == "greedy":
                a = 0 if (mode[i] == 1 and q_now < C.Q_warn) else 1
            else:
                if learn and rng.rand() < eps:
                    a = rng.randint(2)
                else:
                    qv = Q[s_now[i]]
                    cur = int(mode[i] > 0)
                    a = cur if abs(qv[1] - qv[0]) < C.q_hyst else int(qv[1] > qv[0])
            if a == 1 and mode[i] == 0:
                k = try_perch(i, C.R_perch, target)
                if k >= 0:
                    start_perch(i, k, t)
                    _tr(t, "PERCH", i)
                else:
                    lock[i] = t + C.T_retry
            elif a == 0 and mode[i] == 1:
                unperch(i, t)

        # ---- energy ----
        prev_mode = mode_before
        P_prop = np.where(mode == 1, C.P_perch, rotary_power(speed)) * alive
        on = alive & awake
        enc = on & (sem_on if gate else True) if codec == "sem" else np.zeros(N, bool)
        P_rad = C.P_comm * on + C.P_sleep * (alive & ~awake) + C.P_sem * enc
        E -= (P_prop + P_rad) * C.dt
        E_prop += float(P_prop.sum()) * C.dt
        E_comm += float((C.P_comm * on + C.P_sleep * (alive & ~awake)).sum()) * C.dt
        E_sem += C.P_sem * float(enc.sum()) * C.dt

        # ---- rewards (stored, bootstrapped next step) ----
        if learn:
            bE = agent.beta_E if (agent and use_shaping) else C.beta_E
            bQ = agent.beta_Q if (agent and use_shaping) else C.beta_Q
            Ebar = float(np.mean(E[alive]))
            a_exec = (mode > 0).astype(int)
            r = (bE * (P_HOV - P_prop) / P_HOV + bQ * (q_now - C.Q_warn)
                 - C.beta_D * (q_now < C.Q_min)
                 - C.beta_B * np.maximum(0.0, Ebar - E) / C.E_batt * (a_exec == 0)
                 - C.beta_B * np.maximum(0.0, E - Ebar) / C.E_batt * (a_exec == 1) * weak.any())
            r = r - C.beta_sw * (a_exec != prev_mode)
            prev_s = np.where(alive & awake, s_now, -1)
            prev_a, prev_r = a_exec, r

        # ---- slow timescale: PSO repositioning of flying relays ----
        if use_pso and ((t - last_pso) >= C.T_PSO or force_pso):
            fly = np.where(alive & (mode == 0))[0]
            if len(fly):
                target = pso_solve(C, target, pos, fly, g, flows, alive & awake, cmp(), rng, weak=weak)
                pso_calls += 1
            last_pso = t

        # ---- encoder guard: intended configuration must meet Q_opt under current gating ----
        if gate and codec == "sem" and not sem_on[alive & awake].all():
            if min(netq(target), netq(pos)) < C.Q_opt:
                sem_on[:] = True

    recov = [ev[1] if ev[1] is not None else (np.nan if ev[2] else 0.0) for ev in fault_log]
    res = dict(life=float(life), failed=failed, mean_q=q_sum / max(q_n, 1),
               E_prop=E_prop, E_comm=E_comm, E_sem=E_sem, E_llm=E_llm,
               llm_calls=agent.calls if agent else 0, pso_calls=pso_calls,
               recovery=recov, method=method, N=N, seed=seed)
    if record:
        n = int(np.sum(~np.isnan(hist["q"])))
        res.update({k: v[:n] for k, v in hist.items()})
        res["t"] = np.arange(n)
    return res, Q


# =====================================================================================
# 8. TRAINING (shared Q-table, per-episode epsilon decay, cycled training layouts)
# =====================================================================================
def train(C, method, N, n_ep, seed, eval_seeds=()):
    """Returns Q, training-episode lifetimes, and greedy-policy lifetimes on held-out
    layouts after each episode (the latter is what a learning curve should report)."""
    Q = np.zeros((C.n_states, 2))
    curve, ev = [], []
    for ep in range(n_ep):
        eps = max(C.eps_min, C.kappa_eps ** ep)
        res, Q = run_episode(C, N, 90_000 + 100 * seed + ep % 5, method, Q=Q, train=True,
                             eps=eps, record=False)
        curve.append(res["life"])
        if eval_seeds:
            ev.append(np.mean([run_episode(C, N, es, method, Q=Q.copy(), record=False)[0]["life"]
                               for es in eval_seeds]))
    return Q, np.array(curve), np.array(ev)


# =====================================================================================
# 9. EXPERIMENTS
# =====================================================================================
def exp_sem_vs_snr(C):
    snr = np.linspace(-15, 30, 400)
    out = {"snr": snr}
    out["conv_fixed"] = q_from_rate(C, link_rate(make_cfg(conv_model="fixed"), snr, "conv"))
    out["conv_amc"] = q_from_rate(C, link_rate(make_cfg(conv_model="amc"), snr, "conv"))
    out["sem"] = q_from_rate(C, link_rate(C, snr, "sem"))
    return out


def exp_utility_per_joule(C, N, seeds):
    rhos = np.linspace(0, 0.9, 10)
    U = {"conv": np.zeros((len(seeds), len(rhos))), "sem": np.zeros((len(seeds), len(rhos)))}
    for si, sd in enumerate(seeds):
        g, sites, flows, _ = make_layout(C, N, sd)
        for codec in ("conv", "sem"):
            base = initial_deployment(C, N, g, flows, codec, sd)
            alive = np.ones(N, bool)
            for ri, rho in enumerate(rhos):
                k = int(round(N * rho))
                pos = base.copy()
                # perch the k UAVs whose removal hurts least (greedy), at nearest free site
                used = set()
                order = np.argsort([-net_quality(C, np.where(np.arange(N)[:, None] == i,
                                    sites[np.argmin(np.linalg.norm(sites[:, :2] - base[i, :2], axis=1))], base),
                                    g, flows, alive, codec)[0] for i in range(N)])
                for i in order[:k]:
                    d = np.linalg.norm(sites[:, :2] - pos[i, :2], axis=1)
                    for j in np.argsort(d):
                        if j not in used:
                            used.add(j)
                            pos[i] = sites[j]
                            break
                qv, _ = net_quality(C, pos, g, flows, alive, codec)
                P = (k * C.P_perch + (N - k) * P_HOV + N * C.P_comm
                     + (N * C.P_sem + C.P_llm * C.t_llm / C.T_LLM if codec == "sem" else 0.0))
                U[codec][si, ri] = qv / P
    return rhos, U


def run_all(quick=False):
    C = make_cfg()
    if quick:
        C.T_mission = 3000
    N_list = [6, 8, 10, 12]
    seeds = list(range(3)) if quick else list(range(8))
    n_ep = 4 if quick else 12
    tr_seeds = [0, 1] if quick else [0, 1, 2]
    R = {"cfg": vars(C).copy(), "N_list": N_list, "seeds": seeds}
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)

    # ---- training (N=10), learning curves = greedy policy on held-out layouts ----
    Qtab, curves, tcurves = {}, {}, {}
    for m in LEARNED:
        ss = tr_seeds if m in ABL_METHODS else tr_seeds[:1]
        cs, ts = [], []
        for s in ss:
            Q, c, e = train(C, m, 10, n_ep, s, eval_seeds=(80_001 + s,))
            cs.append(e)
            ts.append(c)
            if s == ss[0]:
                Qtab[m] = Q
        curves[m], tcurves[m] = np.array(cs), np.array(ts)
        log(f"trained {m}: held-out lifetime after last episode {cs[0][-1]:.0f} s")
    R["curves"], R["train_curves"], R["Q"] = curves, tcurves, Qtab

    # ---- main grid: lifetime vs N (all methods incl. ablations), paired seeds ----
    grid = {}
    for N in N_list:
        for m in MAIN_METHODS + EXTRA_METHODS:
            for s in seeds:
                res, _ = run_episode(C, N, s, m, Q=Qtab.get(Q_ALIAS.get(m, m)),
                                     record=(N == 10 and s == 0))
                grid[(m, N, s)] = res
        log(f"grid N={N} done")
    R["grid"] = grid

    # ---- faults ----
    Cf = make_cfg(**{**{k: getattr(C, k) for k in ("T_mission",)}})
    faults = [(300, 2), (900, 1)]
    fres = {}
    for m in ["PSO-Only", "QL-GSO", "QL-GSO+Sem", "SA-QL-GSO"]:
        for s in seeds:
            fres[(m, s)], _ = run_episode(Cf, 8, 100 + s, m, Q=Qtab.get(m), faults=faults,
                                          record=(s == 0))
    R["faults"], R["fault_sched"] = fres, faults
    log("faults done")

    # ---- site density ----
    dens = [10, 20, 40, 80]
    sres = {}
    for nd in dens:
        Cs = make_cfg(n_sites=nd, T_mission=C.T_mission)
        for m in ["QL-GSO", "QL-GSO+Sem", "SA-QL-GSO"]:
            for s in seeds[:6]:
                sres[(m, nd, s)] = run_episode(Cs, 10, 200 + s, m, Q=Qtab[m], record=False)[0]
    R["sites"], R["dens"] = sres, dens
    log("site density done")

    # ---- T_PSO: periodic vs agent-triggered, same codec (fair) ----
    Tvals = [30, 60, 120, 240, 480]
    tres = {}
    for Tv in Tvals:
        Ct = make_cfg(T_PSO=Tv, T_mission=C.T_mission)
        for m in ABL_METHODS:
            for s in seeds[:6]:
                tres[(m, Tv, s)] = run_episode(Ct, 10, 300 + s, m, Q=Qtab[m],
                                               faults=[(600, 1)], record=False)[0]
    R["tpso"], R["Tvals"] = tres, Tvals
    log("T_PSO done")

    R["sem_snr"] = exp_sem_vs_snr(C)
    R["upj"] = exp_utility_per_joule(C, 10, seeds[:6])
    log("static experiments done")
    return R


# =====================================================================================
# 10. STATISTICS
# =====================================================================================
def mci(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, 0.0
    if len(x) < 2:
        return float(x.mean()), 0.0
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean()), float(h)


def summarise(R, outdir):
    grid, N_list, seeds = R["grid"], R["N_list"], R["seeds"]
    allm = MAIN_METHODS + EXTRA_METHODS
    summ = {}
    for m in allm:
        summ[m] = {}
        for N in N_list:
            L = [grid[(m, N, s)]["life"] for s in seeds]
            q = [grid[(m, N, s)]["mean_q"] for s in seeds]
            mu, h = mci(L)
            summ[m][N] = dict(life_mean=mu, life_ci=h, q_mean=float(np.mean(q)))
    tests = {}
    for N in N_list:
        a = np.array([grid[("SA-QL-GSO", N, s)]["life"] for s in seeds])
        for m in allm:
            if m == "SA-QL-GSO":
                continue
            b = np.array([grid[(m, N, s)]["life"] for s in seeds])
            d = a - b
            p = 1.0 if np.allclose(d, 0) else float(stats.wilcoxon(a, b, zero_method="zsplit").pvalue)
            tests[f"{m}@N={N}"] = dict(mean_gain_s=float(d.mean()), p_wilcoxon=p)
    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump({"lifetime": summ, "paired_tests_vs_SA": tests}, f, indent=2)

    # LaTeX table (booktabs), best per column in bold
    lines = [r"\begin{tabular}{l" + "c" * len(N_list) + "}", r"\toprule",
             "Method & " + " & ".join(f"$N={N}$" for N in N_list) + r" \\", r"\midrule"]
    best = {N: max(summ[m][N]["life_mean"] for m in allm) for N in N_list}
    for m in allm:
        cells = []
        for N in N_list:
            mu, h = summ[m][N]["life_mean"], summ[m][N]["life_ci"]
            s = f"{mu:.0f} $\\pm$ {h:.0f}"
            cells.append(r"\textbf{" + s + "}" if np.isclose(mu, best[N]) else s)
        lines.append(f"{m} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(outdir, "table_lifetime.tex"), "w") as f:
        f.write("\n".join(lines))
    return summ, tests


# =====================================================================================
# 11. PLOTTING (IEEE-ready: 3.5 in single column, Type-42 fonts, vector PDF)
# =====================================================================================
STY = {
    "Random":       dict(c="#8C8C8C", m="x", ls=":"),
    "Greedy":       dict(c="#E69F00", m="v", ls="--"),
    "Static-Opt":   dict(c="#009E73", m="^", ls="--"),
    "PSO-Only":     dict(c="#D55E00", m="D", ls="--"),
    "MARL":         dict(c="#CC79A7", m="P", ls="-."),
    "QL-GSO":       dict(c="#0072B2", m="o", ls="-"),
    "QL-GSO+Sem":   dict(c="#56B4E9", m="s", ls="-"),
    "QL-GSO+Agent": dict(c="#7B4F2C", m="<", ls="-"),
    "SA-QL-GSO":    dict(c="#C00000", m="*", ls="-"),
    "SA+Shaping":   dict(c="#F4A6A6", m="h", ls="-"),
    "SA-base":      dict(c="#E8A0A0", m="h", ls="-"),
    "SA+Gate":      dict(c="#D96C6C", m="p", ls="-"),
    "SA+Gate+Sleep": dict(c="#B84040", m="d", ls="-"),
    "SA-Greedy":    dict(c="#6B6B6B", m="*", ls="--"),
}
W1, H1 = 3.5, 2.5


def set_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8,
        "legend.fontsize": 6.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.linewidth": 0.6, "lines.linewidth": 1.2, "lines.markersize": 4,
        "axes.grid": True, "grid.linewidth": 0.4, "grid.alpha": 0.35,
        "legend.frameon": True, "legend.framealpha": 0.9, "legend.edgecolor": "0.8",
        "legend.handlelength": 2.0, "savefig.dpi": 300, "figure.dpi": 150,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "xtick.direction": "in", "ytick.direction": "in",
    })


def lw(m):
    return 1.8 if m == "SA-QL-GSO" else 1.1


def ms(m):
    return 7 if m == "SA-QL-GSO" else 4


def smooth(x, w=15):
    if len(x) < w:
        return x
    k = np.ones(w) / w
    y = np.convolve(np.pad(x, (w // 2, w - 1 - w // 2), mode="edge"), k, mode="valid")
    return y


def _save(fig, outdir, name):
    fig.savefig(os.path.join(outdir, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(outdir, name + ".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)


# ---- individual panels (each draws into a given axis) ----
def p_sem_snr(ax, R, C):
    d = R["sem_snr"]
    ax.axvspan(-15, C.gamma_th_dB, color="#D55E00", alpha=0.06, lw=0)
    ax.plot(d["snr"], d["conv_fixed"], color="#D55E00", ls="--", label="Conventional, fixed MCS")
    ax.plot(d["snr"], d["conv_amc"], color="#0072B2", ls="-.", label="Conventional, AMC ladder")
    ax.plot(d["snr"], d["sem"], color="#C00000", lw=1.8, label="SemCom (DeepJSCC-style)")
    ax.axhline(C.Q_min, color="0.4", ls=":", lw=0.8)
    ax.text(-14.3, C.Q_min + 0.02, r"$Q_{\min}$", fontsize=7, color="0.3")
    ax.axvline(C.gamma_th_dB, color="0.3", ls=":", lw=0.8)
    ax.text(C.gamma_th_dB + 0.5, 0.08, r"$\gamma_{\mathrm{th}}$", fontsize=7)
    ax.set_xlabel("Link SNR (dB)")
    ax.set_ylabel("Semantic task quality $Q$")
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlim(-15, 30)
    ax.legend(loc="lower right")


def p_life_N(ax, R):
    N_list, seeds, grid = R["N_list"], R["seeds"], R["grid"]
    off = np.linspace(-0.35, 0.35, len(MAIN_METHODS))
    for j, m in enumerate(MAIN_METHODS):
        mu, h = zip(*[mci([grid[(m, N, s)]["life"] for s in seeds]) for N in N_list])
        st = STY[m]
        eb = ax.errorbar(np.array(N_list) + off[j], mu, yerr=h, color=st["c"], marker=st["m"],
                         ls=st["ls"], lw=lw(m), ms=ms(m), capsize=1.5, elinewidth=0.5, label=m)
        for bar in eb[2]:
            bar.set_alpha(0.45)
    ax.set_xticks(N_list)
    ax.set_xlabel("Number of UAVs $N$")
    ax.set_ylabel("Network lifetime (s)")
    ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.01), fontsize=6,
              columnspacing=0.8, handlelength=1.8, frameon=False)


def p_upj(ax, R):
    rhos, U = R["upj"]
    for codec, lab, st in [("conv", "Conventional", STY["QL-GSO"]),
                           ("sem", "SemCom (incl. encoder + agent power)", STY["SA-QL-GSO"])]:
        mu = U[codec].mean(0) * 1e3
        sd = U[codec].std(0) * 1e3
        ax.plot(rhos, mu, color=st["c"], marker=st["m"], ls=st["ls"], label=lab,
                ms=5 if codec == "sem" else 4)
        ax.fill_between(rhos, mu - sd, mu + sd, color=st["c"], alpha=0.15, lw=0)
    ax.set_xlabel(r"Perching ratio $\rho$")
    ax.set_ylabel(r"Task utility per watt ($\times 10^{-3}$ W$^{-1}$)")
    ax.legend(loc="upper left")


def p_faults(ax, R, C):
    for m in ["PSO-Only", "QL-GSO", "QL-GSO+Sem", "SA-QL-GSO"]:
        res = R["faults"][(m, 0)]
        st = STY[m]
        ax.plot(res["t"], res["q"], color=st["c"], ls=st["ls"], lw=lw(m), label=m)
        if res["failed"]:
            ax.plot(res["t"][-1], res["q"][-1], marker="X", color=st["c"], ms=6, mec="k", mew=0.4)
    dead = [m for m in ["PSO-Only", "QL-GSO", "QL-GSO+Sem"] if R["faults"][(m, 0)]["failed"]
            and R["faults"][(m, 0)]["t"][-1] < R["fault_sched"][0][0] + 60]
    if dead:
        t_d = R["faults"][(dead[0], 0)]["t"][-1]
        ax.annotate("outage > $T_{\\mathrm{out}}$:\n" + ", ".join(dead), xy=(t_d, 0.02),
                    xytext=(t_d + 60, 0.25), fontsize=6, arrowprops=dict(arrowstyle="->", lw=0.6))
    for tf, k in R["fault_sched"]:
        ax.axvline(tf, color="0.2", ls=":", lw=0.8)
        ax.text(tf, 1.07, f"{k} relay{'s' if k > 1 else ''} lost", fontsize=6, ha="center")
    ax.axhline(C.Q_min, color="0.5", ls="--", lw=0.7)
    ax.set_xlim(0, max(R["fault_sched"][-1][0] + 300,
                       max(R["faults"][(m, 0)]["t"][-1] for m in ["QL-GSO", "SA-QL-GSO"])))
    ax.set_xlabel("Mission time (s)")
    ax.set_ylabel("Min. semantic quality")
    ax.set_ylim(-0.02, 1.13)
    ax.legend(loc="center right", fontsize=6)


def p_recovery(ax, R):
    ms_ = ["PSO-Only", "QL-GSO", "QL-GSO+Sem", "SA-QL-GSO"]
    data, labels = [], []
    for m in ms_:
        v = []
        for s in R["seeds"]:
            for x in R["faults"][(m, s)]["recovery"]:
                v.append(x if np.isfinite(x) else np.nan)
        data.append(np.array(v))
        labels.append(m)
    # unrecovered faults are plotted as censored at the top
    top = np.nanmax([np.nanmax(d) if np.isfinite(d).any() else 0 for d in data] + [30]) * 1.15
    for j, (d, m) in enumerate(zip(data, labels)):
        fin = np.maximum(d[np.isfinite(d)], 1.0)
        cen = np.sum(~np.isfinite(d))
        x = np.full(len(fin), j) + np.random.RandomState(j).uniform(-0.12, 0.12, len(fin))
        ax.scatter(x, fin, s=10, color=STY[m]["c"], alpha=0.8, lw=0)
        if len(fin):
            ax.hlines(np.median(fin), j - 0.25, j + 0.25, color="k", lw=1.2)
        if cen:
            ax.text(j, top * 3.5, f"{cen}/{len(d)}\nunrecovered", ha="center", va="top",
                    fontsize=6, color=STY[m]["c"])
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=15, fontsize=6.5)
    ax.set_ylabel("Recovery time to $Q_{\\mathrm{warn}}$ (s)")
    ax.set_yscale("log")
    ax.set_ylim(0.8, top * 4)


def p_learning(ax, R):
    for m in ABL_METHODS:
        c = R["curves"][m]
        ep = np.arange(1, c.shape[1] + 1)
        mu, sd = c.mean(0), c.std(0)
        st = STY[m]
        ax.plot(ep, mu, color=st["c"], marker=st["m"], ls=st["ls"], lw=lw(m), ms=ms(m) - 1, label=m)
        ax.fill_between(ep, mu - sd, mu + sd, color=st["c"], alpha=0.15, lw=0)
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Held-out lifetime (s)")
    ax.legend(loc="lower right", fontsize=6, ncol=2)


def p_pareto(ax, R, N=10):
    grid, seeds = R["grid"], R["seeds"]
    pts = []
    for m in MAIN_METHODS + EXTRA_METHODS:
        L = mci([grid[(m, N, s)]["life"] for s in seeds])
        Qm = mci([grid[(m, N, s)]["mean_q"] for s in seeds])
        st = STY[m]
        ax.errorbar(Qm[0], L[0], xerr=Qm[1], yerr=L[1], color=st["c"], marker=st["m"],
                    ms=ms(m) + 1, capsize=2, elinewidth=0.7, ls="none", label=m)
        pts.append((Qm[0], L[0]))
    P = np.array(pts)
    front = [i for i in range(len(P))
             if not any((P[j, 0] >= P[i, 0]) and (P[j, 1] >= P[i, 1]) and j != i
                        and (P[j] != P[i]).any() for j in range(len(P)))]
    F = P[front][np.argsort(P[front, 0])]
    ax.step(F[:, 0], F[:, 1], where="post", color="0.3", lw=0.8, ls="--", zorder=0)
    ax.set_xlabel("Mean semantic quality")
    ax.set_ylabel("Network lifetime (s)")
    ax.legend(loc="lower left", ncol=2, fontsize=6)


def p_power(ax, R, N=10):
    grid, seeds = R["grid"], R["seeds"]
    ms_ = ["PSO-Only", "Greedy", "QL-GSO", "QL-GSO+Sem", "SA-base", "SA-QL-GSO"]
    comp = {k: [] for k in ("E_prop", "E_comm", "E_sem", "E_llm")}
    for m in ms_:
        for k in comp:
            comp[k].append(np.mean([grid[(m, N, s)][k] / max(grid[(m, N, s)]["life"], 1)
                                    for s in seeds]))
    x = np.arange(len(ms_))
    bottom = np.zeros(len(ms_))
    cols = {"E_prop": "#0072B2", "E_comm": "#E69F00", "E_sem": "#009E73", "E_llm": "#C00000"}
    labs = {"E_prop": "Propulsion", "E_comm": "Communication", "E_sem": "Semantic encoder",
            "E_llm": "Agent inference"}
    for k in comp:
        v = np.array(comp[k])
        ax.bar(x, v, bottom=bottom, color=cols[k], label=labs[k], width=0.65, lw=0)
        bottom += v
    for xi, b in zip(x, bottom):
        ax.text(xi, b + 15, f"{b:.0f}", ha="center", fontsize=6)
    ax.set_xticks(x)
    ax.set_xticklabels(ms_, rotation=20, fontsize=6.5)
    ax.set_ylabel("Mean fleet power (W)")
    ax.set_ylim(0, bottom.max() * 1.15)
    ax.legend(loc="upper right", fontsize=6)


def p_perch(ax, R, N=10):
    for m in ["Greedy", "MARL", "QL-GSO", "SA-QL-GSO"]:
        res = R["grid"][(m, N, 0)]
        st = STY[m]
        ax.plot(res["t"], smooth(res["pr"], 31), color=st["c"], ls=st["ls"], lw=lw(m), label=m)
    res = R["grid"][("SA-QL-GSO", N, 0)]
    if "sleep" in res:
        ax.plot(res["t"], smooth(res["sleep"], 31), color=STY["SA-QL-GSO"]["c"], ls=":", lw=1.2,
                label="SA-QL-GSO, radio asleep")
        ax.plot(res["t"], smooth(res["enc"], 31), color="#009E73", ls="-.", lw=1.0,
                label="SA-QL-GSO, encoders on")
    ax.set_xlabel("Mission time (s)")
    ax.set_ylabel("Fraction of UAVs")
    ax.set_ylim(-0.02, 1.0)
    ax.legend(loc="upper right")


def p_sites(ax, R):
    dens = R["dens"]
    for m in ["QL-GSO", "QL-GSO+Sem", "SA-QL-GSO"]:
        mu, h = zip(*[mci([R["sites"][(m, nd, s)]["life"] for s in R["seeds"][:6]]) for nd in dens])
        st = STY[m]
        ax.errorbar(dens, mu, yerr=h, color=st["c"], marker=st["m"], ls=st["ls"], lw=lw(m),
                    ms=ms(m), capsize=2, elinewidth=0.7, label=m)
    ax.set_xscale("log", base=2)
    ax.set_xticks(dens)
    ax.set_xticklabels([str(d) for d in dens])
    ax.set_xlabel("Number of perching sites")
    ax.set_ylabel("Network lifetime (s)")
    ax.legend(loc="lower right")


def p_tpso(ax, R):
    Tv = R["Tvals"]
    for m, lab in [("QL-GSO", "Periodic, conventional"), ("QL-GSO+Agent", "Agent-triggered, conventional"),
                   ("QL-GSO+Sem", "Periodic, SemCom"), ("SA-QL-GSO", "Agent-triggered, SemCom")]:
        mu, h = map(np.array, zip(*[mci([R["tpso"][(m, T, s)]["life"] for s in R["seeds"][:6]])
                                    for T in Tv]))
        st = STY[m]
        ax.plot(Tv, mu, color=st["c"], marker=st["m"], ls=st["ls"], lw=lw(m), ms=ms(m), label=lab)
        if m in ("SA-QL-GSO", "QL-GSO+Sem"):
            ax.fill_between(Tv, mu - h, mu + h, color=st["c"], alpha=0.12, lw=0)
    ax.set_xscale("log")
    ax.set_xticks(Tv)
    ax.set_xticklabels([str(v) for v in Tv])
    ax.set_xlabel(r"PSO period $T_{\mathrm{PSO}}$ (s)")
    ax.set_ylabel("Network lifetime (s)")
    ax.set_ylim(0, None)
    ax.legend(loc="lower center", ncol=2, fontsize=5.8, bbox_to_anchor=(0.5, 1.01), frameon=False)


def p_q_ts(ax, R, C, N=10):
    for m in ["PSO-Only", "MARL", "QL-GSO", "SA-QL-GSO"]:
        res = R["grid"][(m, N, 0)]
        st = STY[m]
        ax.plot(res["t"], res["q"], color=st["c"], ls=st["ls"], lw=lw(m), label=m)
    ax.axhline(C.Q_min, color="0.5", ls="--", lw=0.7)
    ax.set_xlabel("Mission time (s)")
    ax.set_ylabel("Min. semantic quality")
    ax.set_ylim(-0.02, 1.05)
    ax.legend(loc="lower left")


def p_r_ts(ax, R, N=10):
    for m in ["PSO-Only", "MARL", "QL-GSO", "SA-QL-GSO"]:
        res = R["grid"][(m, N, 0)]
        st = STY[m]
        ax.plot(res["t"], smooth(res["r"], 11), color=st["c"], ls=st["ls"], lw=lw(m), label=m)
    ax.set_xlabel("Mission time (s)")
    ax.set_ylabel("Bottleneck rate (Mbps)")
    ax.legend(loc="lower left")


def p_ablation(ax, R, N=10):
    grid, seeds = R["grid"], R["seeds"]
    labs = {"QL-GSO": "QL-GSO (conv.)", "QL-GSO+Agent": "+ agent (conv.)",
            "QL-GSO+Sem": "QL-GSO + SemCom", "SA-base": "+ agent: triggers, handover",
            "SA+Gate": "+ encoder gating", "SA+Gate+Sleep": "+ radio sleep",
            "SA-QL-GSO": "+ role transfer (SA-QL-GSO)", "SA-Greedy": "SA-QL-GSO, Greedy reflex"}
    y = np.arange(len(CHAIN))[::-1]
    for yi, m in zip(y, CHAIN):
        mu, h = mci([grid[(m, N, s)]["life"] for s in seeds])
        ax.barh(yi, mu, xerr=h, color=STY[m]["c"], height=0.62, capsize=1.5,
                error_kw=dict(elinewidth=0.6), lw=0)
        ax.text(mu + h + 60, yi, f"{mu:.0f}", va="center", fontsize=6)
    ax.set_yticks(y)
    ax.set_yticklabels([labs[m] for m in CHAIN], fontsize=6.5)
    ax.set_xlabel(f"Network lifetime (s), $N={N}$")
    ax.grid(axis="y", visible=False)
    xmax = max(np.mean([grid[(m, N, s)]["life"] for s in seeds]) for m in CHAIN)
    ax.set_xlim(0, xmax * 1.22)


def p_strip(ax, R, N=10):
    grid, seeds = R["grid"], R["seeds"]
    ms_ = ["PSO-Only", "Greedy", "QL-GSO", "QL-GSO+Agent", "QL-GSO+Sem", "SA-base", "SA-QL-GSO"]
    for s in seeds:
        ys = [grid[(m, N, s)]["life"] for m in ms_]
        ax.plot(range(len(ms_)), ys, color="0.75", lw=0.4, zorder=0)
    for j, m in enumerate(ms_):
        y = np.array([grid[(m, N, s)]["life"] for s in seeds])
        ax.scatter(np.full(len(y), j), y, s=14 if m != "SA-QL-GSO" else 26, marker=STY[m]["m"],
                   color=STY[m]["c"], zorder=2, lw=0.3, edgecolors="white")
        ax.hlines(np.median(y), j - 0.28, j + 0.28, color="k", lw=1.0, zorder=3)
    ax.set_xticks(range(len(ms_)))
    ax.set_xticklabels(ms_, rotation=20, fontsize=6.5)
    ax.set_ylabel("Network lifetime (s)")


PANELS = [
    ("fig01_semantic_vs_snr", "Semantic quality vs link SNR", lambda ax, R, C: p_sem_snr(ax, R, C)),
    ("fig02_lifetime_vs_N", "Network lifetime vs N", lambda ax, R, C: p_life_N(ax, R)),
    ("fig03_utility_per_watt", "Utility per watt vs perching ratio", lambda ax, R, C: p_upj(ax, R)),
    ("fig04_fault_trajectory", "QoS under relay failures (N=8)", lambda ax, R, C: p_faults(ax, R, C)),
    ("fig05_learning_curves", "Convergence on 4 held-out layouts (N=10)", lambda ax, R, C: p_learning(ax, R)),
    ("fig06_pareto", "Lifetime vs quality trade-off (N=10)", lambda ax, R, C: p_pareto(ax, R)),
    ("fig07_power_breakdown", "Mean fleet power breakdown (N=10)", lambda ax, R, C: p_power(ax, R)),
    ("fig08_perching_ratio", "Perching, sleep and encoder duty (N=10)", lambda ax, R, C: p_perch(ax, R)),
    ("fig09_site_density", "Lifetime vs perching-site density", lambda ax, R, C: p_sites(ax, R)),
    ("fig10_tpso", "Periodic vs agent-triggered PSO", lambda ax, R, C: p_tpso(ax, R)),
    ("fig11_quality_ts", "Semantic quality over mission (N=10)", lambda ax, R, C: p_q_ts(ax, R, C)),
    ("fig12_rate_ts", "Bottleneck rate over mission (N=10)", lambda ax, R, C: p_r_ts(ax, R)),
    ("fig13_ablation", "Incremental ablation (N=10)", lambda ax, R, C: p_ablation(ax, R)),
    ("fig14_recovery", "Fault recovery time (N=8)", lambda ax, R, C: p_recovery(ax, R)),
    ("fig15_per_layout", "Per-layout lifetime, paired seeds (N=10)", lambda ax, R, C: p_strip(ax, R)),
]


def plot_all(R, outdir):
    set_style()
    C = make_cfg(**{k: v for k, v in R["cfg"].items() if k != "n_states"})
    os.makedirs(outdir, exist_ok=True)
    for name, title, fn in PANELS:
        fig, ax = plt.subplots(figsize=(W1, H1))
        fn(ax, R, C)
        _save(fig, outdir, name)
    # composite overview (for slides / quick inspection)
    for part, rng_ in (("overview_A", range(0, 8)), ("overview_B", range(8, 15))):
        idx = list(rng_)
        rows = int(np.ceil(len(idx) / 2))
        fig, axs = plt.subplots(rows, 2, figsize=(2 * W1 + 0.6, rows * (H1 + 0.45)))
        for a, i in zip(axs.flat, idx):
            name, title, fn = PANELS[i]
            fn(a, R, C)
            pad = 24 if name in ("fig02_lifetime_vs_N", "fig10_tpso") else 4
            a.set_title(f"({chr(97 + i)}) {title}", fontsize=8, pad=pad)
        for a in list(axs.flat)[len(idx):]:
            a.axis("off")
        fig.tight_layout()
        _save(fig, outdir, part)


# =====================================================================================
# 12. MAIN
# =====================================================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    pkl = os.path.join(args.out, "results.pkl")
    if args.plot_only:
        with open(pkl, "rb") as f:
            R = pickle.load(f)
    else:
        R = run_all(quick=args.quick)
        with open(pkl, "wb") as f:
            pickle.dump(R, f)
    summ, tests = summarise(R, args.out)
    plot_all(R, os.path.join(args.out, "figs"))
    print("\nLifetime (s), mean +/- 95% CI")
    for m in summ:
        print(f"  {m:13s}", "  ".join(f"N={N}: {summ[m][N]['life_mean']:6.0f}+/-{summ[m][N]['life_ci']:4.0f}"
                                      for N in R["N_list"]))
    print(f"\nFigures in {args.out}/figs, table in {args.out}/table_lifetime.tex")
