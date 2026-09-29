"""Resumable, time-budgeted driver. `python run_stages.py [budget_s]` runs pending stages
until the budget is used; call repeatedly. `python run_stages.py plot` assembles + plots."""
import os, sys, time, pickle
import numpy as np
import sa_qlgso_v5 as S

OUT, ST = "results", os.path.join("results", "stages")
os.makedirs(ST, exist_ok=True)
C = S.make_cfg()
N_LIST, SEEDS, N_EP, TR = [6, 8, 10, 12], list(range(8)), 12, [0, 1, 2]
EVAL = (80_001, 80_002, 80_003, 80_004)
PARTS = {"a": ["Random", "Greedy", "Static-Opt", "PSO-Only", "MARL", "QL-GSO", "QL-GSO+Agent"],
         "b": ["QL-GSO+Sem", "SA-base", "SA+Gate"],
         "c": ["SA+Gate+Sleep", "SA-QL-GSO", "SA-Greedy"]}
DENS, TVALS, FAULTS = [10, 20, 40, 80], [30, 60, 120, 240, 480], [(300, 2), (900, 1)]


def key(m): return m.replace("+", "p")
def path(n): return os.path.join(ST, n + ".pkl")
def save(n, o): pickle.dump(o, open(path(n), "wb"))
def load(n): return pickle.load(open(path(n), "rb"))


def stages():
    L = []
    for m in S.LEARNED:
        for sd in (TR if m in S.ABL_METHODS else TR[:1]):
            L.append(f"train_{key(m)}_{sd}")
    L += [f"grid_{N}_{p}" for N in N_LIST for p in PARTS]
    L += ["faults"] + [f"sites_{d}" for d in DENS] + [f"tpso_{T}" for T in TVALS] + ["static"]
    return L


def qtab():
    Q = {m: load(f"train_{key(m)}_0")["Q"] for m in S.LEARNED}
    return {m: Q[S.Q_ALIAS.get(m, m)] for m in list(S.METHODS) if S.Q_ALIAS.get(m, m) in Q}


def run(n):
    if n.startswith("train_"):
        mk, sd = n[6:].rsplit("_", 1)
        m = [x for x in S.LEARNED if key(x) == mk][0]
        Q, c, e = S.train(C, m, 10, N_EP, int(sd), eval_seeds=EVAL if m in S.ABL_METHODS else ())
        save(n, dict(Q=Q, tcurve=c, curve=e))
    elif n.startswith("grid_"):
        _, N, p = n.split("_"); N = int(N); Qt = qtab()
        save(n, {(m, N, s): S.run_episode(C, N, s, m, Q=Qt.get(m), record=(N == 10 and s == 0))[0]
                 for m in PARTS[p] for s in SEEDS})
    elif n == "faults":
        Qt = qtab()
        save(n, {(m, s): S.run_episode(C, 8, 100 + s, m, Q=Qt.get(m), faults=FAULTS, record=(s == 0))[0]
                 for m in ["PSO-Only", "QL-GSO", "QL-GSO+Sem", "SA-QL-GSO"] for s in SEEDS})
    elif n.startswith("sites_"):
        d = int(n[6:]); Qt = qtab(); Cs = S.make_cfg(n_sites=d)
        save(n, {(m, d, s): S.run_episode(Cs, 10, 200 + s, m, Q=Qt[m], record=False)[0]
                 for m in ["QL-GSO", "QL-GSO+Sem", "SA-QL-GSO"] for s in SEEDS[:6]})
    elif n.startswith("tpso_"):
        T = int(n[5:]); Qt = qtab(); Ct = S.make_cfg(T_PSO=T)
        save(n, {(m, T, s): S.run_episode(Ct, 10, 300 + s, m, Q=Qt[m], faults=[(600, 1)], record=False)[0]
                 for m in S.ABL_METHODS for s in SEEDS[:6]})
    elif n == "static":
        save(n, dict(sem_snr=S.exp_sem_vs_snr(C), upj=S.exp_utility_per_joule(C, 10, SEEDS[:6])))


def assemble():
    R = {"cfg": vars(C).copy(), "N_list": N_LIST, "seeds": SEEDS, "curves": {}, "train_curves": {},
         "grid": {}, "sites": {}, "tpso": {}, "dens": DENS, "Tvals": TVALS, "fault_sched": FAULTS}
    for m in S.LEARNED:
        ss = TR if m in S.ABL_METHODS else TR[:1]
        d = [load(f"train_{key(m)}_{s}") for s in ss]
        R["curves"][m] = np.array([x["curve"] for x in d])
        R["train_curves"][m] = np.array([x["tcurve"] for x in d])
    for n in stages():
        if n.startswith("grid_"): R["grid"].update(load(n))
        if n.startswith("sites_"): R["sites"].update(load(n))
        if n.startswith("tpso_"): R["tpso"].update(load(n))
    R["faults"] = load("faults"); R.update(load("static"))
    pickle.dump(R, open(os.path.join(OUT, "results.pkl"), "wb"))


if __name__ == "__main__":
    if sys.argv[1:] == ["plot"]:
        assemble(); os.system(f"{sys.executable} sa_qlgso_v5.py --plot-only --out {OUT}")
        sys.exit()
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else 1e9
    t0 = time.time()
    todo = [n for n in stages() if not os.path.exists(path(n))]
    for n in todo:
        if time.time() - t0 > budget:
            break
        t1 = time.time(); run(n)
        print(f"{n}: {time.time() - t1:.0f} s", flush=True)
    left = [n for n in stages() if not os.path.exists(path(n))]
    print(f"pending: {len(left)}", flush=True)
