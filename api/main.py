"""PathSnap FastAPI backend (owner: Ishan). Runs on Laptop 1.

    uvicorn api.main:app --host 0.0.0.0 --port 8000

Env:
  PATHSNAP_PARTITION  eval (default) | student_test | train   -> which trip folders are served
  PATHSNAP_EVAL_DIR   optional folder override (any dir of <trip_id>/<trip_id>.* folders)

Ground truth (.route) is read ONLY server-side, after prediction, to compute metrics.
It is never returned to the client.
"""
import logging
import os
import re
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.common.io import PARTITIONS, ROOT, load_trip
from src.pipeline.run_trip import evaluate_trip, load_artifacts, run_trip
from src.viz.maps import make_route_map

PARTITION = os.environ.get("PATHSNAP_PARTITION", "eval")
TRIP_ROOT = Path(os.environ.get("PATHSNAP_EVAL_DIR") or PARTITIONS[PARTITION])
TRIP_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

LOG_FILE = ROOT / "results" / "logs" / "api_requests.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
log = logging.getLogger("pathsnap")
log.setLevel(logging.INFO)
_fmt = logging.Formatter("%(asctime)s | %(message)s", "%Y-%m-%d %H:%M:%S")
for h in (logging.StreamHandler(), logging.FileHandler(LOG_FILE)):
    h.setFormatter(_fmt)
    log.addHandler(h)

app = FastAPI(title="PathSnap API - Team 5")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

ART = load_artifacts()  # model + decoder config, loaded once at startup
log.info(f"startup | partition={PARTITION} | trips dir={TRIP_ROOT} | model={ART['meta'].get('model', 'final_model')}")


def list_trip_ids():
    if not TRIP_ROOT.exists():
        return []
    return sorted(p.name for p in TRIP_ROOT.iterdir() if p.is_dir() and (p / f"{p.name}.track").exists())


class PredictRequest(BaseModel):
    trip_id: str


def _dedupe(seq):
    return [int(a) for i, a in enumerate(seq) if a >= 0 and (i == 0 or a != seq[i - 1])]


def _num(x):
    return None if x is None else round(float(x), 4)


@app.get("/health")
def health():
    return {"status": "ok", "model": ART["meta"].get("model", "final_model"), "partition": PARTITION,
            "n_trips": len(list_trip_ids())}


@app.get("/trips")
def trips():
    return {"trips": list_trip_ids()}


@app.post("/predict")
def predict(req: PredictRequest, request: Request):
    tid = req.trip_id.strip()
    if not TRIP_ID_RE.match(tid):
        raise HTTPException(422, "invalid trip_id")
    if tid not in list_trip_ids():  # whitelist -> no path traversal possible
        raise HTTPException(404, f"unknown trip_id {tid}")
    d = TRIP_ROOT / tid

    t0 = time.time()
    res = run_trip(d, ART)  # prediction never reads .route
    inference_ms = res.get("inference_ms", (time.time() - t0) * 1000)

    trip = load_trip(d)
    ev = evaluate_trip(d, res, trip["arcs"]) or {}  # server-side only; None when the trip has no .route
    # private keys (e.g. "_xy", "_true_arc" = reference labels) must never leave Laptop 1
    ev = {k: v for k, v in ev.items() if not k.startswith("_")}
    pred_hmm = list(res["pred_hmm"])

    m = make_route_map(trip["track"], trip["nodes"], trip["arcs"], routes={"Predicted route": pred_hmm})

    metrics = {
        "top1_classifier": _num(ev.get("top1_classifier")),
        "top1_hmm": _num(ev.get("top1_hmm")),
        "route_continuity_classifier": _num(ev.get("route_continuity_classifier")),
        "route_continuity_hmm": _num(ev.get("route_continuity_hmm")),
        "candidate_miss_rate": _num(ev.get("candidate_miss_rate")),
        "inference_ms": round(inference_ms, 1),
    }
    out = {
        "trip_id": tid,
        "n_points": int(len(pred_hmm)),
        "n_labelled_points": int(ev.get("n_labelled_points") or 0),
        "metrics": metrics,
        "predicted_route": _dedupe(pred_hmm),
        "map_html": m.get_root().render(),
    }
    client = request.client.host if request.client else "?"
    log.info(f"POST /predict | client={client} | trip={tid} | n_points={out['n_points']} | "
             f"top1_clf={metrics['top1_classifier']} | top1_hmm={metrics['top1_hmm']} | "
             f"continuity_hmm={metrics['route_continuity_hmm']} | inference_ms={metrics['inference_ms']}")
    return out
