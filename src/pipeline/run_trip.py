"""End-to-end prediction for one trip (owner: Sanyam). Used by the API and by evaluate_hmm.py.

run_trip      : raw trip -> clean -> candidates -> features -> classifier -> Viterbi.   NEVER opens .route.
evaluate_trip : only AFTER prediction, reads .route to score the result (server-side reference data).
"""
import json
import time
from pathlib import Path

import numpy as np

from src.common.geo import to_xy
from src.common.io import ROOT, load_arcs, load_nodes, load_route, load_track
from src.common.metrics import candidate_miss_rate, route_continuity, top1_accuracy
from src.features.candidates import generate_candidates
from src.features.features import make_features
from src.features.labels import label_points
from src.hmm.transitions import DEFAULT_CFG, RoadGraph
from src.hmm.viterbi import viterbi_decode
from src.models.predictor import load_model, predict_proba
from src.preprocess.clean import clean_track

HMM_CONFIG = ROOT / "models" / "hmm_config.json"


def load_artifacts():
    """Load the classifier and decoder config once (API startup)."""
    model, meta = load_model()
    cfg = dict(DEFAULT_CFG)
    if HMM_CONFIG.exists():
        cfg.update(json.loads(HMM_CONFIG.read_text()).get("params", {}))
    return {"model": model, "meta": meta, "cfg": cfg}


def load_trip_arcs_nodes(trip_dir):
    """(track, nodes, arcs) for map drawing - no reference route."""
    return load_track(trip_dir), load_nodes(trip_dir), load_arcs(trip_dir)


def _per_point_candidates(cands, prob, n):
    """Group candidate rows by point -> list of {'arcs', 'u', 'prob'} in point order."""
    out = [{"arcs": np.array([], int), "u": np.array([]), "prob": np.array([])} for _ in range(n)]
    order = np.argsort(cands.pi.to_numpy(), kind="stable")
    pi = cands.pi.to_numpy()[order]
    arcs, u, p = cands.cand_arc.to_numpy()[order], cands.proj_u.to_numpy()[order], prob[order]
    starts = np.flatnonzero(np.r_[True, pi[1:] != pi[:-1]])
    ends = np.r_[starts[1:], len(pi)]
    for s, e in zip(starts, ends):
        out[pi[s]] = {"arcs": arcs[s:e], "u": u[s:e], "prob": p[s:e]}
    return out


def prepare_trip(trip_dir, art):
    """Everything that does not depend on the decoder config: clean, candidates, features, classifier."""
    trip_dir = Path(trip_dir)
    tm = {}
    t0 = time.perf_counter()
    track, nodes, arcs = load_trip_arcs_nodes(trip_dir)
    pts = clean_track(track)
    nx_, ny_ = to_xy(nodes.lon.to_numpy(), nodes.lat.to_numpy(), *pts.attrs["origin"])
    px, py = pts.x.to_numpy(), pts.y.to_numpy()
    cands = make_features(pts, generate_candidates(px, py, arcs, nx_, ny_), arcs, nx_, ny_)
    tm["features_ms"] = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    prob = predict_proba(art["model"], cands, art["meta"])
    tm["classifier_ms"] = (time.perf_counter() - t1) * 1000
    per_point = _per_point_candidates(cands, prob, len(pts))
    return {"trip_dir": trip_dir, "pts": pts, "arcs": arcs, "xy": (px, py, nx_, ny_),
            "per_point": per_point, "graph": None, "tm": tm, "t0": t0}


def decode(prep, cfg):
    """Viterbi on a prepared trip; the road graph is built once and cached in prep."""
    px, py, nx_, ny_ = prep["xy"]
    if prep["graph"] is None:
        prep["graph"] = RoadGraph(prep["arcs"], nx_, ny_, px, py, cfg.get("crop_m", 1000.0))
    return viterbi_decode(px, py, prep["pts"].t.to_numpy(), prep["per_point"], prep["graph"], cfg).tolist()


def run_trip(trip_dir, art, cfg=None):
    """Predict one arc per GPS point, classifier-only and with HMM. Reads .track/.nodes/.arcs only."""
    cfg = cfg or art["cfg"]
    prep = prepare_trip(trip_dir, art)
    per_point = prep["per_point"]
    pred_clf = [int(c["arcs"][c["prob"].argmax()]) if len(c["arcs"]) else -1 for c in per_point]

    t2 = time.perf_counter()
    pred_hmm = decode(prep, cfg)
    prep["tm"]["hmm_ms"] = (time.perf_counter() - t2) * 1000

    return {
        "trip_id": prep["trip_dir"].name,
        "points": prep["pts"][["point_idx", "lon", "lat", "t"]].reset_index(drop=True),
        "pred_classifier": pred_clf,
        "pred_hmm": pred_hmm,
        "cand_sets": [c["arcs"].tolist() for c in per_point],
        "timings_ms": {k: round(v, 1) for k, v in prep["tm"].items()},
        "inference_ms": round((time.perf_counter() - prep["t0"]) * 1000, 1),
        "_xy": prep["xy"],  # reused by evaluate_trip, never sent to the UI
    }


def _route_prf(pred, route):
    pred_set, true_set = set(a for a in pred if a >= 0), set(route)
    tp = len(pred_set & true_set)
    p = tp / len(pred_set) if pred_set else 0.0
    r = tp / len(true_set) if true_set else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def evaluate_trip(trip_dir, result, arcs=None):
    """Score a finished prediction against the reference route. Returns None if the trip has no .route."""
    trip_dir = Path(trip_dir)
    route = load_route(trip_dir)
    if route is None:
        return None
    arcs = arcs if arcs is not None else load_arcs(trip_dir)
    px, py, nx_, ny_ = result["_xy"]
    true_arc, _ = label_points(px, py, route, arcs, nx_, ny_)
    true_arc = true_arc.tolist()
    p, r, f = _route_prf(result["pred_hmm"], route)
    return {
        "top1_classifier": round(top1_accuracy(result["pred_classifier"], true_arc), 4),
        "top1_hmm": round(top1_accuracy(result["pred_hmm"], true_arc), 4),
        "route_continuity_classifier": round(route_continuity(result["pred_classifier"], arcs), 4),
        "route_continuity_hmm": round(route_continuity(result["pred_hmm"], arcs), 4),
        "candidate_miss_rate": round(candidate_miss_rate(true_arc, result["cand_sets"]), 4),
        "n_labelled_points": int(sum(a >= 0 for a in true_arc)),
        "route_precision": round(p, 4), "route_recall": round(r, 4), "route_f1": round(f, 4),
        "_true_arc": true_arc,
    }
