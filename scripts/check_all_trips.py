"""Integration check: run EVERY trip served by a live API and validate the response.

    python scripts/check_all_trips.py http://127.0.0.1:8000
"""
import csv
import json
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
METRIC_KEYS = {"top1_classifier", "top1_hmm", "route_continuity_classifier", "route_continuity_hmm",
               "candidate_miss_rate", "inference_ms"}
TOP_KEYS = {"trip_id", "n_points", "n_labelled_points", "metrics", "predicted_route", "map_html"}
LEAK_WORDS = ("true_arc", "_xy", "reference", "route_gt", "ground_truth")


def check(url):
    health = requests.get(f"{url}/health", timeout=10).json()
    trips = requests.get(f"{url}/trips", timeout=10).json()["trips"]
    print(f"health: {health}\n{len(trips)} trips: {trips}\n")
    rows, failed = [], 0
    for tid in trips:
        t0 = time.time()
        r = requests.post(f"{url}/predict", json={"trip_id": tid}, timeout=600)
        rtt = (time.time() - t0) * 1000
        problems = []
        if r.status_code != 200:
            problems.append(f"HTTP {r.status_code}")
            res = {}
        else:
            res = r.json()
            if set(res) != TOP_KEYS:
                problems.append(f"keys {sorted(set(res) ^ TOP_KEYS)}")
            if set(res.get("metrics", {})) != METRIC_KEYS:
                problems.append("metric keys")
            if not res.get("predicted_route"):
                problems.append("empty route")
            html = res.get("map_html", "")
            if "leaflet" not in html.lower() or len(html) < 1000:
                problems.append("map_html not a folium map")
            if any(k.startswith("_") or k == "route" for k in list(res) + list(res.get("metrics", {}))):
                problems.append("private/route key leaked")
            body = json.dumps({k: v for k, v in res.items() if k != "map_html"}).lower()
            if any(w in body for w in LEAK_WORDS) or "Reference route" in html:
                problems.append("ground-truth leak")
        m = res.get("metrics", {})
        rows.append({"trip_id": tid, "status": "OK" if not problems else "FAIL: " + "; ".join(problems),
                     "n_points": res.get("n_points"), "route_arcs": len(res.get("predicted_route", [])),
                     "top1_classifier": m.get("top1_classifier"), "top1_hmm": m.get("top1_hmm"),
                     "continuity_classifier": m.get("route_continuity_classifier"),
                     "continuity_hmm": m.get("route_continuity_hmm"),
                     "miss_rate": m.get("candidate_miss_rate"), "server_ms": m.get("inference_ms"),
                     "round_trip_ms": round(rtt, 1)})
        failed += bool(problems)
        print(" | ".join(f"{k}={v}" for k, v in rows[-1].items()))
    out = ROOT / "results" / f"api_check_{health.get('partition', 'unknown')}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\n{len(trips) - failed}/{len(trips)} trips OK -> {out}")
    return failed


if __name__ == "__main__":
    sys.exit(1 if check(sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000") else 0)
