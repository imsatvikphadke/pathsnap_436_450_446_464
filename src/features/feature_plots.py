"""Feature analysis plots (owner: Satvik). TRAIN partition only.

    python -m src.features.feature_plots
reads data/processed/train_{candidates,points}.parquet, writes results/figures/feat_*.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.common.io import ROOT

FIG = ROOT / "results" / "figures"


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=150)
    plt.close(fig)


def main():
    cands = pd.read_parquet(ROOT / "data" / "processed" / "train_candidates.parquet")
    points = pd.read_parquet(ROOT / "data" / "processed" / "train_points.parquet")
    lab = points[points.true_arc >= 0]

    # 1. GPS noise = distance from each labelled point to its reference arc
    d = lab.label_dist
    med, p95 = d.median(), d.quantile(0.95)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(d, bins=100, range=(0, 50), color="tab:blue")
    ax.axvline(med, color="k", ls="--", label=f"median {med:.1f} m")
    ax.axvline(p95, color="tab:red", ls="--", label=f"95th pct {p95:.1f} m")
    ax.set(xlabel="distance GPS point -> reference arc (m)", ylabel="points",
           title=f"GPS noise (train): median {med:.1f} m, p95 {p95:.1f} m")
    ax.legend()
    save(fig, "feat_true_dist_hist.png")

    # 2. class balance of candidate rows
    counts = cands.label.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["wrong arc (0)", "true arc (1)"], counts.values, color=["tab:grey", "tab:green"])
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v:,}\n({v / counts.sum():.1%})", ha="center", va="bottom")
    ax.set(ylabel="candidate rows", title="Class balance of candidate rows (train)")
    ax.set_ylim(0, counts.max() * 1.2)
    save(fig, "feat_class_balance.png")

    # 3. heading difference: true vs wrong candidates (moving points only, heading is noise when stopped)
    moving = cands[cands.speed_mps > 2]
    fig, ax = plt.subplots(figsize=(7, 4))
    for lbl, name, c in [(0, "wrong arc", "tab:grey"), (1, "true arc", "tab:green")]:
        ax.hist(moving.heading_diff[moving.label == lbl], bins=36, range=(0, 180), density=True,
                histtype="step", lw=2, color=c, label=name)
    ax.set(xlabel="heading difference GPS vs arc (degrees)", ylabel="density",
           title="Heading difference by label (train, speed > 2 m/s)")
    ax.legend()
    save(fig, "feat_heading_diff_by_label.png")

    # 4. where in the distance ranking does the true arc sit?
    rank_share = cands[cands.label == 1].dist_rank.value_counts(normalize=True).sort_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(rank_share.index.astype(int), rank_share.values * 100, color="tab:blue")
    for r, v in rank_share.items():
        ax.text(r, v * 100, f"{v:.1%}", ha="center", va="bottom", fontsize=8)
    ax.set(xlabel="distance rank of the true arc (0 = closest candidate)", ylabel="% of labelled points",
           title=f"True arc is the closest candidate for only {rank_share.iloc[0]:.1%} of points (train)")
    save(fig, "feat_dist_rank_by_label.png")

    near_0 = (moving[moving.label == 1].heading_diff < 30).mean()
    wrong_far = (moving[moving.label == 0].heading_diff > 150).mean()
    print(f"positive rate {cands.label.mean():.3f} | GPS noise median {med:.2f} m p95 {p95:.2f} m | "
          f"true arc at rank 0: {rank_share.iloc[0]:.3f} | moving true arcs with heading_diff < 30: {near_0:.3f} | "
          f"moving wrong arcs with heading_diff > 150: {wrong_far:.3f}")


if __name__ == "__main__":
    main()
