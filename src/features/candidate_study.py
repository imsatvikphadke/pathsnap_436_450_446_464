"""Sensitivity analysis of the candidate generator (owner: Satvik).

NOT hyperparameter tuning: we only measure how the search radius R and the max set size K change
(a) the candidate miss rate - share of labelled points whose true arc is not in their candidate set,
which is a hard ceiling on Top-1 accuracy - and (b) the mean set size, which drives how hard the
classifier's job is and how slow the HMM gets. TRAIN trips only.

    python -m src.features.candidate_study
writes results/candidate_stats.csv, results/figures/feat_miss_rate_vs_radius.png, results/logs/candidate_study.log
"""
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.common.geo import to_xy
from src.common.io import ROOT, list_trips, load_trip, trip_dir
from src.features.candidates import generate_candidates
from src.features.labels import label_points
from src.preprocess.clean import clean_track

RADII = [20, 30, 50, 75, 100]
KS = [4, 8, 12]
CHOSEN = (50, 8)


def study_trip(tid):
    """For one train trip: labels once, then candidates for every (R, K)."""
    trip = load_trip(trip_dir("train", tid))
    pts = clean_track(trip["track"])
    nx, ny = to_xy(trip["nodes"].lon.to_numpy(), trip["nodes"].lat.to_numpy(), *pts.attrs["origin"])
    px, py = pts.x.to_numpy(), pts.y.to_numpy()
    true_arc, _ = label_points(px, py, trip["route"], trip["arcs"], nx, ny)
    labelled = true_arc >= 0

    rows = []
    for r in RADII:
        for k in KS:
            t0 = time.time()
            cand = generate_candidates(px, py, trip["arcs"], nx, ny, radius=r, k=k)
            secs = time.time() - t0
            hit = np.zeros(len(pts), dtype=bool)
            hit[cand.pi.to_numpy()[cand.cand_arc.to_numpy() == true_arc[cand.pi.to_numpy()]]] = True
            rows.append(dict(trip_id=tid, R=r, K=k, labelled=int(labelled.sum()),
                             misses=int((labelled & ~hit).sum()), cand_rows=len(cand), points=len(pts), seconds=secs))
    return rows


def main():
    trips = list_trips("train")
    per_trip = pd.DataFrame([row for rows in Parallel(n_jobs=-1)(delayed(study_trip)(t) for t in trips) for row in rows])

    stats = per_trip.groupby(["R", "K"]).agg(labelled=("labelled", "sum"), misses=("misses", "sum"),
                                             cand_rows=("cand_rows", "sum"), points=("points", "sum"),
                                             seconds=("seconds", "sum")).reset_index()
    stats["miss_rate"] = stats.misses / stats.labelled
    stats["mean_set_size"] = stats.cand_rows / stats.points
    stats = stats[["R", "K", "miss_rate", "mean_set_size", "seconds", "misses", "labelled", "cand_rows"]]
    stats.to_csv(ROOT / "results" / "candidate_stats.csv", index=False)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for k, g in stats.groupby("K"):
        ax[0].plot(g.R, g.miss_rate * 100, marker="o", label=f"K = {k}")
        ax[1].plot(g.R, g.mean_set_size, marker="o", label=f"K = {k}")
    ax[0].set(xlabel="search radius R (m)", ylabel="candidate miss rate (%)", title="Miss rate vs radius (train, 80 trips)")
    ax[1].set(xlabel="search radius R (m)", ylabel="mean candidates per point", title="Candidate set size vs radius (train)")
    for a in ax:
        a.axvline(CHOSEN[0], color="grey", ls="--", lw=0.8)
        a.legend()
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "figures" / "feat_miss_rate_vs_radius.png", dpi=150)

    chosen = stats[(stats.R == CHOSEN[0]) & (stats.K == CHOSEN[1])].iloc[0]
    bigger = stats[(stats.R == 100) & (stats.K == 12)].iloc[0]
    small = stats[(stats.R == 20) & (stats.K == 8)].iloc[0]
    lines = [
        f"Candidate generator sensitivity analysis - train partition, {len(trips)} trips, "
        f"{int(chosen.labelled)} labelled points",
        stats.to_string(index=False, float_format=lambda v: f"{v:.4f}"),
        "",
        f"Chosen R={CHOSEN[0]} m, K={CHOSEN[1]}: miss rate {chosen.miss_rate:.2%}, "
        f"{chosen.mean_set_size:.2f} candidates per point.",
        f"R=20, K=8 is smaller ({small.mean_set_size:.2f} per point) but misses {small.miss_rate:.2%}.",
        f"R=100, K=12 misses {bigger.miss_rate:.2%} but needs {bigger.mean_set_size:.2f} per point "
        f"({bigger.mean_set_size / chosen.mean_set_size:.1f}x more rows for the classifier and HMM).",
    ]
    k12 = stats[(stats.R == CHOSEN[0]) & (stats.K == 12)].iloc[0]
    lines += [
        f"Above R=50 the miss rate does not change: a point only gets a label if it is within 50 m of its arc "
        f"(labels.MAX_LABEL_DIST), so beyond 50 m only K matters. Remaining misses at R=50 come from the K cap.",
        f"K=12 at R=50 cuts misses to {k12.miss_rate:.2%} for {k12.mean_set_size / chosen.mean_set_size:.2f}x the rows - "
        f"a possible v2 change if the classifier handles larger sets well.",
    ]
    text = "\n".join(lines)
    print(text)
    (ROOT / "results" / "logs" / "candidate_study.log").write_text(text + "\n")


if __name__ == "__main__":
    main()
