"""Raw trip -> ML dataset (owner: Samyak).

    python -m src.features.build_dataset --partition train
writes data/processed/{partition}_candidates.parquet, {partition}_points.parquet, feature_cols.json
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.common.geo import to_xy
from src.common.io import ROOT, list_trips, load_trip, trip_dir
from src.features.candidates import generate_candidates
from src.features.features import FEATURE_COLS, make_features
from src.features.labels import label_points
from src.preprocess.clean import clean_track

OUT = ROOT / "data" / "processed"


def build_trip(d, with_labels=True):
    """Returns (points_df, cands_df) for one trip folder. with_labels=False never reads .route."""
    trip = load_trip(d)
    pts = clean_track(trip["track"])
    nx, ny = to_xy(trip["nodes"].lon.to_numpy(), trip["nodes"].lat.to_numpy(), *pts.attrs["origin"])
    px, py = pts.x.to_numpy(), pts.y.to_numpy()

    cand = generate_candidates(px, py, trip["arcs"], nx, ny)
    cand = make_features(pts, cand, trip["arcs"], nx, ny)

    points = pts[["point_idx", "lon", "lat", "t"]].copy()
    points.insert(0, "trip_id", trip["trip_id"])
    points["n_cands"] = cand.groupby("pi").size().reindex(range(len(pts)), fill_value=0).to_numpy()

    cands = cand.assign(trip_id=trip["trip_id"], point_idx=pts.point_idx.to_numpy()[cand.pi.to_numpy()])
    cols = ["trip_id", "point_idx", "cand_arc"]

    if with_labels and trip["route"] is not None:
        true_arc, label_dist = label_points(px, py, trip["route"], trip["arcs"], nx, ny)
        points["true_arc"] = true_arc
        points["label_dist"] = label_dist
        cands["label"] = (cands.cand_arc.to_numpy() == true_arc[cand.pi.to_numpy()]).astype(int)
        points["true_in_cands"] = cands.groupby("pi").label.max().reindex(range(len(pts)), fill_value=0).to_numpy().astype(bool)
        cols.append("label")
    else:
        points["true_arc"] = -1
        points["true_in_cands"] = False
    return points.reset_index(drop=True), cands[cols + FEATURE_COLS].reset_index(drop=True)


def _one(part, tid):
    t0 = time.time()
    pts, cands = build_trip(trip_dir(part, tid), with_labels=part != "eval")
    print(f"{part}/{tid}: {len(pts)} points, {len(cands)} rows, {time.time() - t0:.1f}s", flush=True)
    return pts, cands


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", required=True, choices=["train", "student_test", "eval"])
    ap.add_argument("--n-jobs", type=int, default=-1)
    a = ap.parse_args()

    t0 = time.time()
    res = Parallel(n_jobs=a.n_jobs)(delayed(_one)(a.partition, t) for t in list_trips(a.partition))
    points = pd.concat([r[0] for r in res], ignore_index=True)
    cands = pd.concat([r[1] for r in res], ignore_index=True)

    OUT.mkdir(parents=True, exist_ok=True)
    points.to_parquet(OUT / f"{a.partition}_points.parquet", index=False)
    cands.to_parquet(OUT / f"{a.partition}_candidates.parquet", index=False)
    (OUT / "feature_cols.json").write_text(json.dumps(FEATURE_COLS, indent=2))

    lab = points[points.true_arc >= 0]
    summary = {
        "partition": a.partition, "trips": len(res), "points": len(points), "candidate_rows": len(cands),
        "avg_cands_per_point": round(len(cands) / max(len(points), 1), 2),
        "labelled_share": round(len(lab) / max(len(points), 1), 4),
        "median_label_dist_m": round(float(lab.label_dist.median()), 2) if "label_dist" in lab and len(lab) else None,
        "positive_rate": round(float(cands.label.mean()), 4) if "label" in cands else None,
        "candidate_miss_rate": round(float(1 - lab.true_in_cands.mean()), 4) if len(lab) else None,
        "seconds": round(time.time() - t0, 1),
    }
    log = ROOT / "results" / "logs" / f"build_dataset_{a.partition}.log"
    log.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
