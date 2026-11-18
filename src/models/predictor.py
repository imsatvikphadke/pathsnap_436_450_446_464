"""Load the final tuned model and score candidate rows (owner: Ishan). Used by the HMM and the API."""
import json
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[2]


def load_model(path=ROOT / "models" / "final_model.joblib"):
    """Returns (model, meta). meta['features'] = the exact column order the model was trained on."""
    path = Path(path)
    meta = json.loads(path.with_name("final_model_meta.json").read_text())
    return joblib.load(path), meta


def predict_proba(model, cands_df, meta):
    """1-D array: P(candidate is the true arc), one value per row of cands_df, same order."""
    return model.predict_proba(cands_df[meta["features"]])[:, 1]
