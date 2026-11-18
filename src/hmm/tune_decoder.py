"""Bayesian (Optuna TPE) tuning of the HMM decoder parameters (owner: Sanyam).

Faculty rule: hyperparameters are tuned with Bayesian optimisation only - no grid / random search.
Objective = mean HMM Top-1 over 20 seeded TRAIN trips + the 10 STUDENT_TEST trips (allowed for the
decoder by the guidelines). Evaluator trips are never touched.

    python -m src.hmm.tune_decoder --trials 25
"""
import argparse
import json
import time

import matplotlib
matplotlib.use("Agg")
import numpy as np
import optuna

from src.common.io import ROOT, list_trips, load_route, trip_dir
from src.common.metrics import top1_accuracy
from src.features.labels import label_points
from src.hmm.transitions import DEFAULT_CFG
from src.pipeline.run_trip import decode, load_artifacts, prepare_trip

SEED = 42
RESULTS = ROOT / "results"


def prepare_all(art):
    rng = np.random.default_rng(SEED)
    train = sorted(rng.choice(list_trips("train"), size=20, replace=False).tolist())
    trips = [("train", t) for t in train] + [("student_test", t) for t in list_trips("student_test")]
    preps = []
    for part, tid in trips:
        d = trip_dir(part, tid)
        prep = prepare_trip(d, art)
        px, py, nx_, ny_ = prep["xy"]
        prep["true"], _ = label_points(px, py, load_route(d), prep["arcs"], nx_, ny_)  # scoring only
        prep["name"] = f"{part}/{tid}"
        preps.append(prep)
    return preps


def objective(trial, preps):
    cfg = dict(DEFAULT_CFG,
               beta=trial.suggest_float("beta", 2.0, 60.0, log=True),
               penalty=trial.suggest_float("penalty", 5.0, 60.0),
               lam=trial.suggest_float("lam", 0.05, 3.0, log=True),
               uturn_penalty=trial.suggest_float("uturn_penalty", 0.0, 20.0),
               gap_restart_s=trial.suggest_float("gap_restart_s", 20.0, 600.0, log=True))
    return float(np.mean([top1_accuracy(decode(p, cfg), p["true"]) for p in preps]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=25)
    a = ap.parse_args()

    art = load_artifacts()
    t0 = time.time()
    preps = prepare_all(art)
    base = float(np.mean([top1_accuracy(decode(p, DEFAULT_CFG), p["true"]) for p in preps]))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED),
                                study_name="hmm_decoder")
    study.enqueue_trial({k: DEFAULT_CFG[k] for k in ("beta", "penalty", "lam", "uturn_penalty", "gap_restart_s")})
    study.optimize(lambda t: objective(t, preps), n_trials=a.trials)
    wall = time.time() - t0

    study.trials_dataframe().to_csv(RESULTS / "decoder_trials.csv", index=False)
    best = dict(DEFAULT_CFG, **study.best_params)
    (ROOT / "models" / "hmm_config.json").write_text(json.dumps({
        "params": best, "objective": "mean HMM Top-1 (20 seeded train trips + 10 student_test trips)",
        "sampler": f"optuna TPESampler(seed={SEED})", "trials": len(study.trials),
        "default_score": round(base, 4), "best_score": round(study.best_value, 4), "wall_time_s": round(wall, 1),
        "trips": [p["name"] for p in preps]}, indent=2))
    (RESULTS / "logs" / "decoder_tuning.log").write_text(
        f"TPE seed {SEED}, {len(study.trials)} trials, wall {wall:.1f} s\n"
        f"default cfg Top-1 {base:.4f} -> best {study.best_value:.4f} (trial {study.best_trial.number})\n"
        f"best params {json.dumps(study.best_params, indent=2)}\n")

    ax = optuna.visualization.matplotlib.plot_optimization_history(study)
    ax.figure.set_size_inches(8, 4.5); ax.set_title("HMM decoder - Optuna TPE optimisation history")
    ax.figure.tight_layout(); ax.figure.savefig(RESULTS / "figures" / "hmm_optuna_history.png", dpi=130)
    print(f"default {base:.4f} -> best {study.best_value:.4f}", study.best_params, f"{wall:.0f}s")


if __name__ == "__main__":
    main()
