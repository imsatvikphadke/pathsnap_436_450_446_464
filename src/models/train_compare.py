"""Train + compare 6 classifiers that score 'is this candidate arc the true arc?' (owner: Ishan).

Protocol
  * GroupKFold by trip_id on the 80 train trips (points of one trip never sit in both train and validation)
  * per fold: macro-F1, precision, recall, log loss, observation-level Top-1, fit time, + train-side scores
  * refit on all train -> evaluate on the 10 student_test trips (Top-1, F1, log loss, route continuity, inference ms)
  * selection rule: highest CV Top-1; ties broken by log loss, fold std, inference time

Run:  python -m src.models.train_compare [--folds 3]
"""
import argparse
import time

import joblib
import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.evaluate import (FIG, RESULTS, SEED, classification_metrics, load_partition, mean_route_continuity,
                                 selected_features, top1_accuracy)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# rows used to FIT each model (None = all ~1.1M rows). Slow models are trained on a random subsample.
TRAIN_ROWS = {"LogReg": None, "GaussianNB": None, "KNN": 60_000, "RandomForest": 300_000, "LightGBM": None, "MLP": 300_000}

LGBM_DEFAULT = dict(n_estimators=300, learning_rate=0.05, num_leaves=63, random_state=SEED, n_jobs=-1, verbose=-1)


def make_models():
    scaled = lambda m: Pipeline([("scale", StandardScaler()), ("model", m)])  # noqa: E731
    return {
        "LogReg": (scaled(LogisticRegression(class_weight="balanced", max_iter=500)), "Linear",
                   "L2, C=1, class_weight=balanced, scaled"),
        "GaussianNB": (Pipeline([("model", GaussianNB())]), "Probabilistic", "default var_smoothing=1e-9"),
        "KNN": (scaled(KNeighborsClassifier(n_neighbors=25, weights="distance", n_jobs=-1)), "Instance-based",
                "k=25, distance-weighted, scaled, fit on 60k rows"),
        "RandomForest": (Pipeline([("model", RandomForestClassifier(n_estimators=100, max_depth=16, min_samples_leaf=5,
                                                                     n_jobs=-1, random_state=SEED))]),
                         "Bagging ensemble", "100 trees, max_depth=16, min_leaf=5, fit on 300k rows"),
        "LightGBM": (Pipeline([("model", lgb.LGBMClassifier(**LGBM_DEFAULT))]), "Boosting",
                     "300 trees, lr=0.05, 63 leaves (untuned)"),
        "MLP": (scaled(MLPClassifier(hidden_layer_sizes=(64, 32), early_stopping=True, max_iter=60,
                                     random_state=SEED)), "Neural network", "64-32 ReLU, adam, early stopping, fit on 300k rows"),
    }


def subsample(X, y, n):
    if n is None or len(X) <= n:
        return X, y
    idx = np.random.default_rng(SEED).choice(len(X), n, replace=False)
    return X.iloc[idx], y.iloc[idx]


def train_side_scores(model, cands, points, feats, trips):
    """Scores on 8 training-fold trips -> compared with validation scores to spot overfitting."""
    part = cands[cands.trip_id.isin(trips)]
    prob = model.predict_proba(part[feats])[:, 1]
    return top1_accuracy(part, prob, points), classification_metrics(part.label, prob)["macro_f1"]


TIE_TOL = 0.005  # CV Top-1 differences below half a percentage point are treated as a tie


def select_model(comp):
    """Primary: CV Top-1. Models within TIE_TOL of the best are tied -> lower CV log loss (better probabilities
    for the HMM), then lower fold std, then faster inference."""
    tied = comp[comp["CV Top-1"] >= comp["CV Top-1"].max() - TIE_TOL]
    best = tied.sort_values(["CV log loss", "CV Top-1 std", "Inference ms/trip"]).iloc[0]["Model"]
    ranked = comp.sort_values("CV Top-1", ascending=False)
    (RESULTS / "logs" / "model_selection.txt").write_text(
        f"Selection rule: highest CV Top-1 (GroupKFold by trip, train only). Models within {TIE_TOL} of the best "
        "are a tie -> lower CV log loss, then lower fold std, then faster inference.\n\n"
        + ranked[["Model", "CV Top-1", "CV Top-1 std", "CV macro-F1", "CV log loss", "Inference ms/trip"]].round(4)
        .to_string(index=False)
        + f"\n\nTied on CV Top-1: {', '.join(tied.Model)}\nSelected model: {best}\n")
    return best


def run(folds):
    feats = selected_features()
    tr, tr_pts = load_partition("train")
    te, te_pts = load_partition("student_test")
    X, y, groups = tr[feats], tr.label, tr.trip_id
    print(f"{len(feats)} features, train rows {len(tr):,}, student_test rows {len(te):,}, {folds}-fold GroupKFold")

    fold_rows, comp_rows, fitted = [], [], {}
    for name, (model, family, config) in make_models().items():
        for k, (i_tr, i_va) in enumerate(GroupKFold(n_splits=folds).split(X, y, groups)):
            Xf, yf = subsample(X.iloc[i_tr], y.iloc[i_tr], TRAIN_ROWS[name])
            t0 = time.time()
            model.fit(Xf, yf)
            fit_s = time.time() - t0
            va = tr.iloc[i_va]
            prob = model.predict_proba(va[feats])[:, 1]
            m = classification_metrics(va.label, prob)
            tr_trips = np.random.default_rng(SEED + k).choice(groups.iloc[i_tr].unique(), 8, replace=False)
            tr_top1, tr_f1 = train_side_scores(model, tr, tr_pts, feats, tr_trips)
            fold_rows.append(dict(model=name, fold=k, top1=top1_accuracy(va, prob, tr_pts), **m,
                                  train_top1=tr_top1, train_macro_f1=tr_f1, fit_s=fit_s))
            print(f"  {name:12s} fold {k}: top1={fold_rows[-1]['top1']:.4f} f1={m['macro_f1']:.4f} fit={fit_s:.1f}s", flush=True)

        # refit on all train -> student_test
        Xf, yf = subsample(X, y, TRAIN_ROWS[name])
        t0 = time.time()
        model.fit(Xf, yf)
        fit_s = time.time() - t0
        t0 = time.time()
        prob = model.predict_proba(te[feats])[:, 1]
        infer_ms = (time.time() - t0) * 1000 / te.trip_id.nunique()
        m = classification_metrics(te.label, prob)
        cv = pd.DataFrame([r for r in fold_rows if r["model"] == name])
        comp_rows.append({
            "Model": name, "Algorithm family": family, "Key configuration": config,
            "CV macro-F1": cv.macro_f1.mean(), "CV macro-F1 std": cv.macro_f1.std(),
            "CV Top-1": cv.top1.mean(), "CV Top-1 std": cv.top1.std(), "CV log loss": cv.logloss.mean(),
            "Train-fold Top-1": cv.train_top1.mean(),
            "Test Top-1": top1_accuracy(te, prob, te_pts), "Test macro-F1": m["macro_f1"],
            "Test precision": m["precision"], "Test recall": m["recall"], "Test log loss": m["logloss"],
            "Route continuity": mean_route_continuity(te, prob, "student_test"),
            "Fit time (s)": fit_s, "Inference ms/trip": infer_ms, "Train rows used": len(Xf),
        })
        fitted[name] = (model, prob)
        print(f"{name}: test top1={comp_rows[-1]['Test Top-1']:.4f}", flush=True)

    cv_df, comp = pd.DataFrame(fold_rows), pd.DataFrame(comp_rows)
    cv_df.to_csv(RESULTS / "cv_folds.csv", index=False)
    comp.round(4).to_csv(RESULTS / "model_comparison.csv", index=False)

    best = select_model(comp)
    print("selected:", best)
    joblib.dump({n: p for n, (_, p) in fitted.items()}, RESULTS / "logs" / "student_test_probs.joblib")
    plots(cv_df, comp, fitted, te, feats, tr)


def plots(cv_df, comp, fitted, te, feats, tr):
    names = comp.Model.tolist()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(names, comp["CV macro-F1"], yerr=comp["CV macro-F1 std"], capsize=4, color="#2f5d9e")
    ax.set(title="Cross-validation macro-F1 (GroupKFold by trip, train)", ylabel="macro-F1", ylim=(0, 1))
    fig.tight_layout(); fig.savefig(FIG / "model_cv_macro_f1.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    w = 0.38; x = np.arange(len(names))
    ax.bar(x - w / 2, comp["Test Top-1"], w, label="Top-1 accuracy", color="#1b2a4a")
    ax.bar(x + w / 2, comp["Test macro-F1"], w, label="macro-F1", color="#7fa6d9")
    ax.set(xticks=x, xticklabels=names, ylim=(0, 1), title="Student-test Top-1 and macro-F1")
    ax.legend(); fig.tight_layout(); fig.savefig(FIG / "model_test_top1_f1.png", dpi=150); plt.close(fig)

    for col, fname, title in [("Test log loss", "model_logloss.png", "Student-test log loss (lower is better)"),
                              ("Inference ms/trip", "model_inference_time.png", "Inference time per trip (ms)")]:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(names, comp[col], color="#2f5d9e")
        ax.set(title=title); fig.tight_layout(); fig.savefig(FIG / fname, dpi=150); plt.close(fig)

    for name, (_, prob) in fitted.items():
        fig, ax = plt.subplots(figsize=(4, 4))
        ConfusionMatrixDisplay.from_predictions(te.label, (prob >= 0.5).astype(int), ax=ax, cmap="Blues",
                                                display_labels=["wrong arc", "true arc"], colorbar=False)
        ax.set_title(f"{name} - student_test")
        fig.tight_layout(); fig.savefig(FIG / f"model_confusion_{name}.png", dpi=150); plt.close(fig)

    # real training histories only: LightGBM (per boosting round) and MLP (per epoch)
    gkf = next(GroupKFold(n_splits=3).split(tr, tr.label, tr.trip_id))
    a, b = tr.iloc[gkf[0]], tr.iloc[gkf[1]]
    m = lgb.LGBMClassifier(**LGBM_DEFAULT)
    m.fit(a[feats], a.label, eval_set=[(a[feats], a.label), (b[feats], b.label)],
          eval_names=["train", "validation"], eval_metric="binary_logloss")
    fig, ax = plt.subplots(figsize=(6, 4))
    for k, v in m.evals_result_.items():
        ax.plot(v["binary_logloss"], label=k)
    ax.set(title="LightGBM log loss per boosting round (train fold vs held-out trips)", xlabel="round", ylabel="log loss")
    ax.legend(); fig.tight_layout(); fig.savefig(FIG / "model_lgbm_curve.png", dpi=150); plt.close(fig)

    mlp = fitted["MLP"][0].named_steps["model"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(mlp.loss_curve_, label="training loss (log loss)")
    ax2 = ax.twinx()
    ax2.plot(mlp.validation_scores_, color="#d62728", label="validation accuracy (10% early-stop split)")
    ax.set(title="MLP training history", xlabel="epoch", ylabel="loss"); ax2.set_ylabel("accuracy")
    fig.legend(loc="center right"); fig.tight_layout(); fig.savefig(FIG / "model_mlp_curve.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=3)
    run(ap.parse_args().folds)
