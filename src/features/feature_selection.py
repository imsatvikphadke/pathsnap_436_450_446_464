"""Feature selection on candidate dataset (owner: Sanyam Deven Hiran - PES2UG24CS446).

Uses train data ONLY:
1. Pearson correlation matrix: drop one feature from each pair with |r| > 0.95
2. Mutual information with binary label (sklearn mutual_info_classif on sample)
3. LightGBM gain importance (100 trees)
Produces:
- results/feature_selection.csv
- results/figures/featsel_corr_heatmap.png
- results/figures/featsel_mi.png
- results/figures/featsel_importance.png
- models/selected_features.json
"""
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif
import lightgbm as lgb

from src.common.io import ROOT

PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
MODELS_DIR = ROOT / "models"


def run_feature_selection(sample_size=200_000, corr_threshold=0.95, random_state=42):
    train_parquet = PROCESSED_DIR / "train_candidates.parquet"
    feature_cols_path = PROCESSED_DIR / "feature_cols.json"

    if not train_parquet.exists() or not feature_cols_path.exists():
        raise FileNotFoundError(f"Missing {train_parquet} or {feature_cols_path}. Run build_dataset first!")

    with open(feature_cols_path, "r") as f:
        feature_cols = json.load(f)

    print(f"Loading candidate dataset from {train_parquet}...")
    df = pd.read_parquet(train_parquet)
    print(f"Total candidate rows: {len(df)}, Features: {feature_cols}")

    # Ensure label column exists
    if "label" not in df.columns:
        raise ValueError("Missing 'label' column in candidates dataset!")

    # Subsample for Mutual Information computation efficiency
    if len(df) > sample_size:
        print(f"Sampling {sample_size} rows for Mutual Information calculation...")
        df_sample = df.sample(n=sample_size, random_state=random_state)
    else:
        df_sample = df

    X_sample = df_sample[feature_cols].fillna(0)
    y_sample = df_sample["label"]

    X_all = df[feature_cols].fillna(0)
    y_all = df["label"]

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Pearson Correlation Analysis
    print("Computing Pearson correlation matrix...")
    corr_matrix = X_all.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True)
    plt.title("Candidate Features - Pearson Correlation Heatmap")
    plt.tight_layout()
    corr_fig_path = FIG_DIR / "featsel_corr_heatmap.png"
    plt.savefig(corr_fig_path, dpi=200)
    plt.close()
    print(f"Saved {corr_fig_path}")

    # Identify pairs with |r| > corr_threshold
    dropped_corr = set()
    max_corr_vals = {}
    for i, col1 in enumerate(feature_cols):
        max_r = 0.0
        for j, col2 in enumerate(feature_cols):
            if i != j:
                r = abs(corr_matrix.loc[col1, col2])
                if r > max_r:
                    max_r = r
                if r > corr_threshold and col2 not in dropped_corr and col1 not in dropped_corr:
                    # Mark col2 to drop
                    print(f"High correlation (|r| = {r:.3f}) between '{col1}' and '{col2}'. Dropping '{col2}'")
                    dropped_corr.add(col2)
        max_corr_vals[col1] = max_r

    # 2. Mutual Information
    print("Computing Mutual Information with target label...")
    mi_scores = mutual_info_classif(X_sample, y_sample, random_state=random_state)
    mi_dict = dict(zip(feature_cols, mi_scores))

    plt.figure(figsize=(8, 5))
    s_mi = pd.Series(mi_dict).sort_values(ascending=True)
    s_mi.plot(kind="barh", color="steelblue")
    plt.title("Mutual Information with True Arc Label")
    plt.xlabel("Mutual Information")
    plt.tight_layout()
    mi_fig_path = FIG_DIR / "featsel_mi.png"
    plt.savefig(mi_fig_path, dpi=200)
    plt.close()
    print(f"Saved {mi_fig_path}")

    # 3. LightGBM Gain Importance
    print("Fitting LightGBM (100 trees) for Gain Feature Importance...")
    lgb_train = lgb.Dataset(X_all, label=y_all)
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "n_estimators": 100,
        "learning_rate": 0.1,
        "num_leaves": 31,
        "random_state": random_state,
        "verbose": -1,
    }
    gbm = lgb.train(
        params,
        lgb_train,
        num_boost_round=100,
    )
    gain_importance = gbm.feature_importance(importance_type="gain")
    importance_dict = dict(zip(feature_cols, gain_importance))

    plt.figure(figsize=(8, 5))
    s_imp = pd.Series(importance_dict).sort_values(ascending=True)
    s_imp.plot(kind="barh", color="forestgreen")
    plt.title("LightGBM Feature Importance (Gain)")
    plt.xlabel("Total Gain")
    plt.tight_layout()
    imp_fig_path = FIG_DIR / "featsel_importance.png"
    plt.savefig(imp_fig_path, dpi=200)
    plt.close()
    print(f"Saved {imp_fig_path}")

    # Decision logic:
    # Drop if dropped_corr OR (low MI and low importance)
    # Define thresholds
    mi_threshold = 0.001
    importance_threshold = np.percentile(gain_importance, 10)

    rows = []
    kept_features = []
    dropped_features = []

    for f in feature_cols:
        mi = mi_dict[f]
        gain = importance_dict[f]
        max_r = max_corr_vals[f]

        drop_reason = []
        if f in dropped_corr:
            drop_reason.append(f"collinear (|r| > {corr_threshold})")
        if mi < mi_threshold and gain <= importance_threshold:
            drop_reason.append("low MI and low importance")

        is_kept = len(drop_reason) == 0
        if is_kept:
            kept_features.append(f)
        else:
            dropped_features.append(f)

        rows.append({
            "feature": f,
            "mutual_info": round(mi, 5),
            "lgb_gain_importance": round(gain, 2),
            "max_abs_corr": round(max_r, 4),
            "kept": is_kept,
            "drop_reason": "; ".join(drop_reason) if drop_reason else "None"
        })

    summary_df = pd.DataFrame(rows)
    csv_path = RESULTS_DIR / "feature_selection.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")

    selected_json = {
        "features": kept_features,
        "dropped": dropped_features,
        "method": "correlation (|r| > 0.95) + mutual information + LightGBM gain importance (train only)",
        "train_candidate_rows": len(df),
    }
    json_path = MODELS_DIR / "selected_features.json"
    with open(json_path, "w") as f:
        json.dump(selected_json, f, indent=2)
    print(f"Saved {json_path}")
    print(f"Selected {len(kept_features)}/{len(feature_cols)} features: {kept_features}")
    if dropped_features:
        print(f"Dropped features: {dropped_features}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=200_000)
    parser.add_argument("--corr-threshold", type=float, default=0.95)
    args = parser.parse_args()
    run_feature_selection(args.sample_size, args.corr_threshold)
