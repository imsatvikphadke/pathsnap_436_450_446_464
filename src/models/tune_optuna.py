"""Bayesian hyperparameter optimisation of the selected model (LightGBM) with Optuna TPE (owner: Ishan).

TPE (Tree-structured Parzen Estimator) is Bayesian: it models which hyperparameter regions gave good
scores so far and samples the next trial where improvement is most likely - no grid, no random search.
Objective = mean Top-1 over a 3-fold GroupKFold on the 80 TRAIN trips only.

Run:  python -m src.models.tune_optuna [--trials 25] [--timeout 900]
"""
import argparse
import json
import time
from datetime import datetime

import joblib
import lightgbm as lgb
import matplotlib
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import GroupKFold

from src.models.evaluate import (FIG, RESULTS, ROOT, SEED, classification_metrics, load_partition, selected_features,
                                 top1_accuracy)
from src.models.train_compare import LGBM_DEFAULT

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FOLDS = 3


def cv_scores(params, tr, tr_pts, feats, trial=None):
    """Mean Top-1 / macro-F1 / log loss over GroupKFold folds for one LightGBM configuration.
    With a trial, the running mean is reported after each fold so the MedianPruner can stop bad trials early."""
    top1, f1, ll = [], [], []
    for i_tr, i_va in GroupKFold(n_splits=FOLDS).split(tr, tr.label, tr.trip_id):
        a, b = tr.iloc[i_tr], tr.iloc[i_va]
        m = lgb.LGBMClassifier(**params).fit(a[feats], a.label)
        prob = m.predict_proba(b[feats])[:, 1]
        top1.append(top1_accuracy(b, prob, tr_pts))
        c = classification_metrics(b.label, prob)
        f1.append(c["macro_f1"]); ll.append(c["logloss"])
        if trial is not None:
            trial.report(float(np.mean(top1)), len(top1))
            if trial.should_prune():
                raise optuna.TrialPruned()
    return float(np.mean(top1)), float(np.mean(f1)), float(np.mean(ll))


def objective(trial, tr, tr_pts, feats):
    params = dict(
        num_leaves=trial.suggest_int("num_leaves", 15, 255, log=True),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        n_estimators=trial.suggest_int("n_estimators", 100, 500),
        min_child_samples=trial.suggest_int("min_child_samples", 5, 200),
        subsample=trial.suggest_float("subsample", 0.5, 1.0), subsample_freq=1,
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
        scale_pos_weight=trial.suggest_float("scale_pos_weight", 1, 8),
        random_state=SEED, n_jobs=-1, verbose=-1,
    )
    top1, f1, ll = cv_scores(params, tr, tr_pts, feats, trial)
    trial.set_user_attr("cv_macro_f1", f1)
    trial.set_user_attr("cv_logloss", ll)
    return top1


def test_scores(model, te, te_pts, feats):
    t0 = time.time()
    prob = model.predict_proba(te[feats])[:, 1]
    ms = (time.time() - t0) * 1000 / te.trip_id.nunique()
    c = classification_metrics(te.label, prob)
    return top1_accuracy(te, prob, te_pts), c["macro_f1"], c["logloss"], ms


def run(n_trials, timeout):
    feats = selected_features()
    tr, tr_pts = load_partition("train")
    te, te_pts = load_partition("student_test")

    t0 = time.time()
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=5), study_name="lightgbm_top1")
    study.enqueue_trial({k: v for k, v in LGBM_DEFAULT.items() if k in ("num_leaves", "learning_rate", "n_estimators")})
    study.optimize(lambda t: objective(t, tr, tr_pts, feats), n_trials=n_trials, timeout=timeout,
                   show_progress_bar=False)
    wall = time.time() - t0

    trials = study.trials_dataframe()
    trials.to_csv(RESULTS / "optuna_trials.csv", index=False)
    best = dict(study.best_params, subsample_freq=1, random_state=SEED, n_jobs=-1, verbose=-1)
    (RESULTS / "logs" / "optuna.log").write_text(
        f"sampler: TPESampler(seed={SEED})  objective: mean {FOLDS}-fold GroupKFold Top-1 on train\n"
        f"trials: {len(study.trials)} ({sum(t.state.name == 'PRUNED' for t in study.trials)} pruned)  wall time: {wall:.1f} s\n"
        f"best Top-1: {study.best_value:.4f}  (trial {study.best_trial.number})\nbest params: {json.dumps(study.best_params, indent=2)}\n")

    for fn, name in [(optuna.visualization.matplotlib.plot_optimization_history, "optuna_history.png"),
                     (optuna.visualization.matplotlib.plot_param_importances, "optuna_param_importance.png")]:
        ax = fn(study)
        ax.figure.set_size_inches(8, 4.5); ax.figure.tight_layout()
        ax.figure.savefig(FIG / name, dpi=150); plt.close(ax.figure)

    # untuned vs tuned: same features, same folds, same test trips
    rows = []
    for label, params in [("Untuned LightGBM (defaults from comparison)", LGBM_DEFAULT), ("Tuned LightGBM (Optuna TPE)", best)]:
        cv_top1, cv_f1, cv_ll = cv_scores(params, tr, tr_pts, feats)
        model = lgb.LGBMClassifier(**params).fit(tr[feats], tr.label)
        t_top1, t_f1, t_ll, ms = test_scores(model, te, te_pts, feats)
        rows.append({"Model": label, "CV macro-F1": cv_f1, "CV Top-1": cv_top1, "CV log loss": cv_ll,
                     "Test Top-1": t_top1, "Test macro-F1": t_f1, "Test log loss": t_ll, "Inference ms/trip": ms})
    pd.DataFrame(rows).round(4).to_csv(RESULTS / "tuned_vs_untuned.csv", index=False)

    joblib.dump(model, ROOT / "models" / "final_model.joblib", compress=3)
    meta = {"model": "LightGBM (tuned with Optuna TPE)", "features": feats, "params": best, "seed": SEED,
            "cv_top1": rows[1]["CV Top-1"], "test_top1": rows[1]["Test Top-1"], "n_trials": len(study.trials),
            "trained_on": "all 80 train trips", "trained_at": datetime.now().isoformat(timespec="seconds")}
    (ROOT / "models" / "final_model_meta.json").write_text(json.dumps(meta, indent=2))
    print(pd.DataFrame(rows).round(4).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=25)
    ap.add_argument("--timeout", type=int, default=None, help="stop starting new trials after this many seconds")
    a = ap.parse_args()
    run(a.trials, a.timeout)
