"""Report maps (owner: Samyak): classifier-only vs HMM routes for 3 student_test trips.

    python -m src.viz.make_report_maps
reads results/predictions/<trip>.csv (point_idx, lon, lat, t, true_arc, pred_classifier, pred_hmm),
writes results/maps/trip_<id>.html and a PNG screenshot results/figures/map_<id>.png.
"""
import subprocess
from pathlib import Path

import pandas as pd

from src.common.io import ROOT, load_trip, trip_dir
from src.viz.maps import make_route_map

PRED = ROOT / "results" / "predictions"
MAPS = ROOT / "results" / "maps"
FIG = ROOT / "results" / "figures"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def trip_scores():
    rows = []
    for f in sorted(PRED.glob("*.csv")):
        p = pd.read_csv(f)
        lab = p[p.true_arc >= 0]
        rows.append({"trip_id": f.stem, "top1_clf": (lab.pred_classifier == lab.true_arc).mean(),
                     "top1_hmm": (lab.pred_hmm == lab.true_arc).mean()})
    s = pd.DataFrame(rows)
    s["gain"] = s.top1_hmm - s.top1_clf
    return s


def pick_trips(s):
    """best HMM trip, a typical (median) one, and the one where the HMM helped most."""
    best = s.sort_values("top1_hmm").trip_id.iloc[-1]
    typical = s.iloc[(s.top1_hmm - s.top1_hmm.median()).abs().argsort()].trip_id.iloc[0]
    gain = s.sort_values("gain").trip_id.iloc[-1]
    return list(dict.fromkeys([best, typical, gain] + s.trip_id.tolist()))[:3]


def screenshot(html, png):
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--screenshot={png}",
                    "--window-size=1400,900", "--virtual-time-budget=8000", html.resolve().as_uri()],
                   check=False, capture_output=True, timeout=120)


def main():
    s = trip_scores()
    print(s.round(3).to_string(index=False))
    for tid in pick_trips(s):
        trip = load_trip(trip_dir("student_test", tid))
        p = pd.read_csv(PRED / f"{tid}.csv")
        routes = {"Reference route": trip["route"],
                  "Classifier only": p.pred_classifier.tolist(),
                  "HMM / Viterbi": p.pred_hmm.tolist()}
        html = MAPS / f"trip_{tid}.html"
        make_route_map(trip["track"], trip["nodes"], trip["arcs"], routes).save(str(html))
        screenshot(html, FIG / f"map_{tid}.png")
        print("wrote", html.name, f"map_{tid}.png")


if __name__ == "__main__":
    main()
