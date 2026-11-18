"""Classifier-only vs classifier + HMM/Viterbi on the 10 student_test trips (owner: Sanyam).

    python -m src.hmm.evaluate_hmm
writes results/hmm_vs_classifier.csv, results/predictions/<trip>.csv, hmm_top1_per_trip.png, hmm_continuity.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.common.io import ROOT, list_trips, trip_dir
from src.pipeline.run_trip import evaluate_trip, load_artifacts, run_trip

RESULTS = ROOT / "results"
FIG = RESULTS / "figures"


def main():
    art = load_artifacts()
    (RESULTS / "predictions").mkdir(parents=True, exist_ok=True)
    rows = []
    for tid in list_trips("student_test"):
        d = trip_dir("student_test", tid)
        res = run_trip(d, art)
        ev = evaluate_trip(d, res)
        pred = res["points"].assign(true_arc=ev["_true_arc"], pred_classifier=res["pred_classifier"],
                                    pred_hmm=res["pred_hmm"])
        pred.to_csv(RESULTS / "predictions" / f"{tid}.csv", index=False)
        rows.append({"trip_id": tid, "n_points": len(pred), **{k: v for k, v in ev.items() if not k.startswith("_")},
                     "inference_ms": res["inference_ms"], "hmm_ms": res["timings_ms"]["hmm_ms"]})
        print(rows[-1])

    df = pd.DataFrame(rows)
    df["top1_gain"] = df.top1_hmm - df.top1_classifier
    mean = df.drop(columns="trip_id").mean(numeric_only=True).round(4)
    pooled_w = df.n_labelled_points / df.n_labelled_points.sum()
    out = pd.concat([df, pd.DataFrame([{"trip_id": "MEAN", **mean.to_dict()}])], ignore_index=True)
    out.to_csv(RESULTS / "hmm_vs_classifier.csv", index=False)

    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x - 0.2, df.top1_classifier, 0.4, label="classifier only (argmax)")
    ax.bar(x + 0.2, df.top1_hmm, 0.4, label="classifier + HMM/Viterbi")
    ax.set(xticks=x, xticklabels=df.trip_id, ylim=(0.5, 1.02), ylabel="Top-1 accuracy",
           title="student_test: observation-level Top-1 per trip")
    ax.tick_params(axis="x", rotation=45); ax.legend(loc="lower right"); fig.tight_layout()
    fig.savefig(FIG / "hmm_top1_per_trip.png", dpi=130)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x - 0.2, df.route_continuity_classifier, 0.4, label="classifier only")
    ax.bar(x + 0.2, df.route_continuity_hmm, 0.4, label="classifier + HMM/Viterbi")
    ax.set(xticks=x, xticklabels=df.trip_id, ylim=(0, 1.02), ylabel="route continuity",
           title="student_test: share of predicted arc changes that are physically connected")
    ax.tick_params(axis="x", rotation=45); ax.legend(loc="lower right"); fig.tight_layout()
    fig.savefig(FIG / "hmm_continuity.png", dpi=130)

    worst = df.sort_values("top1_gain").head(3)
    lines = [f"MEAN over 10 student_test trips: Top-1 {mean.top1_classifier} -> {mean.top1_hmm}, "
             f"continuity {mean.route_continuity_classifier} -> {mean.route_continuity_hmm}",
             f"pooled (point-weighted) Top-1: {np.sum(df.top1_classifier * pooled_w):.4f} -> {np.sum(df.top1_hmm * pooled_w):.4f}",
             "Trips where HMM helped least:"]
    lines += [f"  {r.trip_id}: {r.top1_classifier} -> {r.top1_hmm} (gain {r.top1_gain:+.4f}), miss rate {r.candidate_miss_rate}, "
              f"continuity {r.route_continuity_hmm}" for r in worst.itertuples()]
    (RESULTS / "logs" / "hmm_analysis.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
