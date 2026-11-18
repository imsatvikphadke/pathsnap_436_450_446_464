"""Data loading + metrics shared by train_compare.py and tune_optuna.py (owner: Ishan)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, log_loss, precision_score, recall_score

from src.common.io import load_arcs, trip_dir
from src.common.metrics import route_continuity

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIG = RESULTS / "figures"
SEED = 42


def selected_features():
    f = ROOT / "models" / "selected_features.json"
    if f.exists():
        return json.loads(f.read_text())["features"]
    return json.loads((PROC / "feature_cols.json").read_text())


def load_partition(part):
    """Candidate rows + per-point table for 'train' or 'student_test'."""
    cands = pd.read_parquet(PROC / f"{part}_candidates.parquet")
    points = pd.read_parquet(PROC / f"{part}_points.parquet")
    return cands, points


def top1_predictions(cands, prob):
    """Arg-max candidate per GPS point -> DataFrame(trip_id, point_idx, pred_arc)."""
    df = cands[["trip_id", "point_idx", "cand_arc"]].assign(prob=prob)
    best = df.loc[df.groupby(["trip_id", "point_idx"])["prob"].idxmax()]
    return best.rename(columns={"cand_arc": "pred_arc"})[["trip_id", "point_idx", "pred_arc"]]


def top1_accuracy(cands, prob, points):
    """Observation-level Top-1 over labelled points (true_arc >= 0).
    A point whose true arc is not among its candidates can never be right -> counts as wrong."""
    pred = top1_predictions(cands, prob)
    pts = points[(points.true_arc >= 0) & points.trip_id.isin(cands.trip_id.unique())]
    lab = pts.merge(pred, on=["trip_id", "point_idx"], how="left")
    return float((lab.pred_arc == lab.true_arc).mean())


def classification_metrics(y, prob):
    yhat = (prob >= 0.5).astype(int)
    return {
        "macro_f1": f1_score(y, yhat, average="macro"),
        "precision": precision_score(y, yhat, average="macro", zero_division=0),
        "recall": recall_score(y, yhat, average="macro", zero_division=0),
        "logloss": log_loss(y, np.clip(prob, 1e-7, 1 - 1e-7), labels=[0, 1]),
    }


def mean_route_continuity(cands, prob, part):
    """Route continuity of the classifier-only (arg-max) arc sequence, averaged over trips."""
    pred = top1_predictions(cands, prob).sort_values(["trip_id", "point_idx"])
    vals = [route_continuity(g.pred_arc.tolist(), load_arcs(trip_dir(part, tid))) for tid, g in pred.groupby("trip_id")]
    return float(np.mean(vals))
