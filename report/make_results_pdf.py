"""Results book (owner: Samyak). Builds report/PathSnap_Results.pdf purely from files in the repo.

    python report/make_results_pdf.py

Every number below is read from results/*.csv, results/logs/*, models/*.json - nothing is typed in by hand.
"""
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, Preformatted, SimpleDocTemplate,
                                Spacer, Table, TableStyle)
from reportlab.platypus.tableofcontents import TableOfContents

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RES, FIG, LOGS, MODELS = ROOT / "results", ROOT / "results" / "figures", ROOT / "results" / "logs", ROOT / "models"
OUT = ROOT / "report" / "PathSnap_Results.pdf"
REPO = "https://github.com/imsatvikphadke/pathsnap_436_450_446_464"
TEAM = [("Samyak Sanklecha", "PES2UG24CS436"), ("Sanyam Deven Hiran", "PES2UG24CS446"),
        ("Satvik Phadke", "PES2UG24CS450"), ("Shaurya Ishan", "PES2UG24CS464")]

NAVY, BLUE = colors.HexColor("#1b2a4a"), colors.HexColor("#2f5d9e")
ss = getSampleStyleSheet()
TITLE = ParagraphStyle("t", parent=ss["Title"], textColor=NAVY, fontSize=26, leading=30)
H1 = ParagraphStyle("h1", parent=ss["Heading1"], textColor=NAVY, fontSize=16, leading=20, spaceBefore=6, spaceAfter=6)
H2 = ParagraphStyle("h2", parent=ss["Heading2"], textColor=BLUE, fontSize=12, leading=15, spaceBefore=8, spaceAfter=4)
BODY = ParagraphStyle("b", parent=ss["Normal"], fontSize=9.4, leading=12.5, spaceAfter=4)
SRC = ParagraphStyle("src", parent=BODY, fontSize=7.6, textColor=colors.HexColor("#5a6a85"), spaceAfter=8)
CELL = ParagraphStyle("cell", parent=BODY, fontSize=7.4, leading=9, spaceAfter=0)
CODE = ParagraphStyle("c", parent=ss["Code"], fontSize=7.2, leading=8.8, backColor=colors.HexColor("#f2f4f8"),
                      borderPadding=4, spaceAfter=8)
WIDTH = 180 * mm


class Doc(SimpleDocTemplate):
    def afterFlowable(self, f):
        if isinstance(f, Paragraph) and f.style.name in ("h1", "h2"):
            self.notify("TOCEntry", (0 if f.style.name == "h1" else 1, f.getPlainText(), self.page))


def h1(t): return Paragraph(t, H1)
def h2(t): return Paragraph(t, H2)
def p(t): return Paragraph(t, BODY)
def src(t): return Paragraph(f"Source: {t}", SRC)
def code(t, width=118):
    """Monospace block; long lines are wrapped so nothing runs off the page."""
    import textwrap
    lines = [w for line in t.rstrip().splitlines() for w in (textwrap.wrap(line, width, subsequent_indent="    ") or [""])]
    return Preformatted("\n".join(lines), CODE)


def fmt(v):
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        if pd.isna(v):
            return "-"
        return f"{v:.4f}" if abs(v) < 10 else f"{v:,.1f}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


def df_table(df, widths=None, max_rows=None):
    if max_rows:
        df = df.head(max_rows)
    rows = [[Paragraph(f"<b>{c}</b>", CELL) for c in df.columns]]
    rows += [[Paragraph(fmt(v), CELL) for v in r] for r in df.itertuples(index=False)]
    n = len(df.columns)
    widths = widths or [WIDTH / n] * n
    t = Table(rows, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#b8c2d3")),
                           ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dfe6f2")),
                           ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7f9fc")]),
                           ("VALIGN", (0, 0), (-1, -1), "TOP"),
                           ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3)]))
    return t


def kv_table(pairs, w=(70, 110)):
    return df_table(pd.DataFrame(pairs, columns=["Item", "Value"]), [w[0] * mm, w[1] * mm])


def fig(name, width=WIDTH, max_h=110 * mm):
    path = FIG / name
    if not path.exists():
        return [p(f"<i>missing figure: {name}</i>")]
    iw, ih = ImageReader(str(path)).getSize()
    w = width
    h = w * ih / iw
    if h > max_h:
        h, w = max_h, max_h * iw / ih
    return [Image(str(path), width=w, height=h), src(f"results/figures/{name}")]


def fig_grid(names, cols=2, max_h=70 * mm):
    cells, cw = [], WIDTH / cols
    for n in names:
        path = FIG / n
        if not path.exists():
            cells.append(Paragraph(f"missing: {n}", CELL)); continue
        iw, ih = ImageReader(str(path)).getSize()
        w = cw - 4 * mm
        h = w * ih / iw
        if h > max_h:
            h, w = max_h, max_h * iw / ih
        cells.append([Image(str(path), width=w, height=h), Paragraph(n, SRC)])
    rows = [cells[i:i + cols] for i in range(0, len(cells), cols)]
    rows[-1] += [""] * (cols - len(rows[-1]))
    t = Table(rows, colWidths=[cw] * cols)
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("ALIGN", (0, 0), (-1, -1), "CENTER")]))
    return t


def read_json(path):
    return json.loads(Path(path).read_text())


def read(path):
    return Path(path).read_text()


def footer(c, doc):
    c.saveState()
    c.setFont("Helvetica", 7.5)
    c.setFillColor(colors.grey)
    c.drawString(15 * mm, 9 * mm, "PathSnap - Team 5 - Results Book (auto-generated from repository files)")
    c.drawRightString(195 * mm, 9 * mm, f"page {doc.page}")
    c.restoreState()


def build():
    story = []
    try:
        commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        commit = "n/a"

    # ---------------- title + TOC ----------------
    story += [Spacer(1, 30 * mm), Paragraph("PathSnap - Results Book", TITLE),
              p("<font size=12 color='#2f5d9e'>GPS map-matching &amp; route reconstruction - UE24CS352A Machine Learning Hackathon</font>"),
              Spacer(1, 8 * mm), p("<b>Team 5</b>")]
    story.append(kv_table(TEAM, (80, 60)))
    story += [Spacer(1, 4 * mm), p(f"Repository: <b>{REPO}</b>"),
              p(f"Generated: {datetime.now():%Y-%m-%d %H:%M} from commit <b>{commit}</b> by report/make_results_pdf.py"),
              p("Every number in this book is read from a file in the repository (results/, models/). The source file and "
                "partition are printed under each table and figure. Evaluator-test data is never used in any result here."),
              PageBreak(), h1("Contents")]
    toc = TableOfContents()
    toc.levelStyles = [ParagraphStyle("toc1", parent=BODY, fontSize=10, leading=14),
                       ParagraphStyle("toc2", parent=BODY, fontSize=8.5, leading=11, leftIndent=12)]
    story += [toc, PageBreak()]

    # ---------------- 1. dataset & EDA ----------------
    inv = pd.read_csv(RES / "eda_inventory.csv", dtype={"trip_id": str})
    qual = pd.read_csv(RES / "eda_quality.csv", dtype={"trip_id": str})
    net = pd.read_csv(RES / "eda_network.csv", dtype={"trip_id": str})
    story += [h1("1. Dataset and exploratory data analysis"), h2("1.1 Inventory per partition")]
    cols = ["n_points", "n_nodes", "n_arcs", "n_route_arcs", "length_km", "duration_min"]
    s = inv.groupby("partition")[cols].agg(["min", "median", "max"]).T.reset_index()
    s.columns = ["metric", "stat"] + list(s.columns[2:])
    tot = inv.groupby("partition").agg(trips=("trip_id", "count"), points=("n_points", "sum")).reset_index()
    story += [df_table(tot, [60 * mm, 60 * mm, 60 * mm]), Spacer(1, 3 * mm), df_table(s),
              src("results/eda_inventory.csv - train + student_test")]
    story += [h2("1.2 Missing / invalid values")]
    q = qual.groupby("partition")[["nan_values", "out_of_range", "duplicate_t", "non_increasing_t"]].sum().reset_index()
    story += [df_table(q), src("results/eda_quality.csv - train + student_test")]
    story += [h2("1.3 Road-network size and connectivity")]
    ncols = ["n_nodes", "n_arcs", "weak_components", "largest_wcc_share", "strong_components", "largest_scc_share",
             "isolated_nodes", "two_way_share"]
    ns = net.groupby("partition")[ncols].agg(["min", "median", "max"]).T.reset_index()
    ns.columns = ["metric", "stat"] + list(ns.columns[2:])
    story += [df_table(ns), src("results/eda_network.csv - train + student_test")]
    story += [h2("1.4 Sampling gaps, speed jumps, GPS noise, labels (from the EDA notebook figures)")]
    story += [p("Reference-label coverage and GPS noise are read from the build_dataset logs (section 2.1). "
                "Gap and speed distributions are shown in the figures below.")]
    story += [fig_grid(["eda_dt_hist.png", "eda_speed_hist.png", "eda_trip_stats.png", "eda_label_dist_hist.png",
                        "eda_degree_hist.png", "eda_trip_locations.png"], cols=2, max_h=62 * mm),
              src("eda/eda.ipynb - train + student_test"), PageBreak()]
    story += [h2("1.5 Trips over their road network")]
    story += [fig_grid(["eda_trip_00000003_network.png", "eda_trip_00000005_network.png"], cols=2, max_h=95 * mm),
              src("eda/eda.ipynb section 8 - train")]
    story += fig("map_00000007.png", max_h=100 * mm)
    story += [p("Folium map screenshot (interactive version: results/maps/trip_00000007.html; EDA map: "
                "results/maps/eda_trip_00000007.html) - student_test."), PageBreak()]

    # ---------------- 2. preprocessing & features ----------------
    story += [h1("2. Preprocessing, labels, candidates and features"), h2("2.1 Built datasets")]
    logs = {part: read_json(LOGS / f"build_dataset_{part}.log") for part in ("train", "student_test")}
    keys = list(logs["train"].keys())
    bd = pd.DataFrame({"field": keys, "train": [logs["train"][k] for k in keys],
                       "student_test": [logs["student_test"].get(k) for k in keys]})
    story += [df_table(bd, [70 * mm, 55 * mm, 55 * mm]),
              src("results/logs/build_dataset_train.log, build_dataset_student_test.log")]
    story += [h2("2.2 Candidate generator sensitivity (radius R, top-K)")]
    cs = pd.read_csv(RES / "candidate_stats.csv")
    story += [df_table(cs), src("results/candidate_stats.csv - train (80 trips)")]
    story += fig("feat_miss_rate_vs_radius.png", max_h=75 * mm)
    story += [h2("2.3 Feature set")]
    from src.features.features import FEATURE_COLS
    meaning = {
        "dist_m": "perpendicular distance GPS point -> candidate arc (m)",
        "dist_rank": "rank of that distance among the point's candidates (0 = closest)",
        "dist_gap": "distance minus the closest candidate's distance (m)",
        "n_cands": "number of candidates of the point",
        "proj_u": "position of the projection along the arc (0 = start, 1 = end)",
        "heading_diff": "angle between GPS heading and arc bearing (0-180 deg)",
        "speed_mps": "GPS speed (m/s)",
        "is_gap": "1 if the point follows a time gap > 10 s",
        "arc_len_m": "length of the candidate arc (m)",
        "dist_prev": "distance of the previous GPS point to the same arc (m)",
        "dist_next": "distance of the next GPS point to the same arc (m)",
        "in_deg": "number of arcs entering the arc's start node",
        "out_deg": "number of arcs leaving the arc's end node",
        "two_way": "1 if the reverse arc exists",
    }
    story += [df_table(pd.DataFrame({"feature": FEATURE_COLS, "meaning": [meaning.get(f, "") for f in FEATURE_COLS]}),
                       [40 * mm, 140 * mm]), src("src/features/features.py FEATURE_COLS")]
    story += [fig_grid(["feat_true_dist_hist.png", "feat_class_balance.png", "feat_heading_diff_by_label.png",
                        "feat_dist_rank_by_label.png"], cols=2, max_h=65 * mm), src("src/features/feature_plots.py - train"),
              PageBreak()]
    story += [h2("2.4 Feature selection")]
    fs = pd.read_csv(RES / "feature_selection.csv")
    sel = read_json(MODELS / "selected_features.json")
    story += [df_table(fs), src("results/feature_selection.csv - train"),
              p(f"<b>Method:</b> {sel['method']}. <b>Kept ({len(sel['features'])}):</b> {', '.join(sel['features'])}. "
                f"<b>Dropped:</b> {', '.join(sel['dropped']) or 'none'}."), src("models/selected_features.json")]
    story += [fig_grid(["featsel_corr_heatmap.png", "featsel_mi.png", "featsel_importance.png"], cols=2, max_h=70 * mm),
              PageBreak()]

    # ---------------- 3. models ----------------
    mc = pd.read_csv(RES / "model_comparison.csv")
    story += [h1("3. Model experimentation and comparison"), h2("3.1 Overall comparison (6 models)")]
    a = ["Model", "Algorithm family", "Key configuration", "CV macro-F1", "CV macro-F1 std", "CV Top-1", "CV Top-1 std",
         "CV log loss", "Train-fold Top-1"]
    b = ["Model", "Test Top-1", "Test macro-F1", "Test precision", "Test recall", "Test log loss", "Route continuity",
         "Fit time (s)", "Inference ms/trip", "Train rows used"]
    story += [p("<b>Cross-validation (GroupKFold by trip, train only)</b>"),
              df_table(mc[a], [20 * mm, 20 * mm, 40 * mm] + [16.7 * mm] * 6),
              src("results/model_comparison.csv - train (CV)"), Spacer(1, 3 * mm),
              p("<b>Student-test evaluation (refit on all train)</b>"), df_table(mc[b]),
              src("results/model_comparison.csv - student_test")]
    story += [h2("3.2 Per-fold results")]
    cv = pd.read_csv(RES / "cv_folds.csv")
    story += [df_table(cv), src("results/cv_folds.csv - train (3-fold GroupKFold)")]
    story += [h2("3.3 Model selection"), code(read(LOGS / "model_selection.txt")), src("results/logs/model_selection.txt"),
              PageBreak()]
    story += [h2("3.4 Comparison plots")]
    story += [fig_grid(["model_cv_macro_f1.png", "model_test_top1_f1.png", "model_logloss.png", "model_inference_time.png"],
                       cols=2, max_h=65 * mm), src("src/models/train_compare.py")]
    story += [h2("3.5 Confusion matrices (student_test, threshold 0.5)")]
    story += [fig_grid([f"model_confusion_{m}.png" for m in mc.Model], cols=3, max_h=55 * mm)]
    story += [h2("3.6 Training curves (only models with epoch-wise histories)")]
    story += [fig_grid(["model_lgbm_curve.png", "model_mlp_curve.png"], cols=2, max_h=65 * mm), PageBreak()]

    # ---------------- 4. Bayesian optimisation ----------------
    story += [h1("4. Bayesian hyperparameter optimisation (Optuna TPE)"), h2("4.1 Search summary")]
    story += [code(read(LOGS / "optuna.log")), src("results/logs/optuna.log - train only")]
    meta = read_json(MODELS / "final_model_meta.json")
    story += [h2("4.2 Final model"), kv_table([(k, fmt(v)) for k, v in meta.get("params", {}).items()]),
              p(f"<b>Model:</b> {meta.get('model', '')}"), src("models/final_model_meta.json")]
    ot = pd.read_csv(RES / "optuna_trials.csv")
    pc = [c for c in ot.columns if c.startswith("params_")]
    top = ot[ot.state == "COMPLETE"].sort_values("value", ascending=False)[["number", "value"] + pc].head(10)
    top.columns = [c.replace("params_", "") for c in top.columns]
    story += [h2("4.3 Top-10 trials (objective = CV Top-1)"), df_table(top),
              src(f"results/optuna_trials.csv ({len(ot)} trials, {int((ot.state == 'PRUNED').sum())} pruned)")]
    story += [h2("4.4 Untuned vs tuned"), df_table(pd.read_csv(RES / "tuned_vs_untuned.csv")),
              src("results/tuned_vs_untuned.csv - CV on train, test on student_test")]
    story += [fig_grid(["optuna_history.png", "optuna_param_importance.png"], cols=2, max_h=65 * mm), PageBreak()]

    # ---------------- 5. HMM ----------------
    hc = read_json(MODELS / "hmm_config.json")
    story += [h1("5. HMM and Viterbi route reconstruction"), h2("5.1 Decoder configuration")]
    story += [kv_table([(k, fmt(v)) for k, v in hc["params"].items()] +
                       [("objective", hc.get("objective", "")), ("sampler", hc.get("sampler", "")),
                        ("trials", hc.get("trials", "")), ("default score", hc.get("default_score", "")),
                        ("best score", hc.get("best_score", "")), ("wall time (s)", hc.get("wall_time_s", ""))]),
              src("models/hmm_config.json")]
    dt = pd.read_csv(RES / "decoder_trials.csv")
    dpc = [c for c in dt.columns if c.startswith("params_")]
    dtop = dt[dt.state == "COMPLETE"].sort_values("value", ascending=False)[["number", "value"] + dpc].head(5)
    dtop.columns = [c.replace("params_", "") for c in dtop.columns]
    story += [h2("5.2 Bayesian decoder tuning - top-5 trials"), df_table(dtop),
              src("results/decoder_trials.csv; results/logs/decoder_tuning.log"), code(read(LOGS / "decoder_tuning.log"))]
    hv = pd.read_csv(RES / "hmm_vs_classifier.csv", dtype={"trip_id": str})
    hcols = ["trip_id", "n_points", "top1_classifier", "top1_hmm", "top1_gain", "route_continuity_classifier",
             "route_continuity_hmm", "candidate_miss_rate", "route_precision", "route_recall", "route_f1", "inference_ms",
             "hmm_ms"]
    hcols = [c for c in hcols if c in hv.columns]
    if (hv.trip_id == "MEAN").any():  # evaluate_hmm.py already appends the mean row
        hv2 = hv[hcols]
    else:
        hv2 = pd.concat([hv[hcols], pd.DataFrame([{"trip_id": "MEAN", **hv[hcols[1:]].mean().to_dict()}])], ignore_index=True)
    hv = hv[hv.trip_id != "MEAN"]  # per-trip rows only for the maps and summary below
    story += [PageBreak(), h2("5.3 Classifier-only vs classifier + HMM (per trip)"), df_table(hv2),
              src("results/hmm_vs_classifier.csv - student_test (10 trips)")]
    story += [fig_grid(["hmm_top1_per_trip.png", "hmm_continuity.png", "hmm_optuna_history.png"], cols=2, max_h=65 * mm)]
    story += [h2("5.4 Analysis"), code(read(LOGS / "hmm_analysis.txt")), src("results/logs/hmm_analysis.txt"), PageBreak()]

    # ---------------- 6. maps ----------------
    story += [h1("6. Geographical visualisation (Folium)")]
    for m in sorted(FIG.glob("map_*.png")):
        tid = re.findall(r"map_(\d+)", m.name)[0]
        row = hv[hv.trip_id == tid]
        cap = ""
        if len(row):
            r = row.iloc[0]
            cap = (f"Trip {tid}: Top-1 classifier-only {r.top1_classifier:.4f} -> HMM {r.top1_hmm:.4f}; "
                   f"continuity {r.route_continuity_classifier:.4f} -> {r.route_continuity_hmm:.4f}")
        story += [KeepTogether([h2(f"Trip {tid}"), p(cap)] + fig(m.name, max_h=95 * mm) +
                               [src(f"results/maps/trip_{tid}.html (interactive) - student_test; layers: GPS, network, "
                                    "reference (green), classifier-only (orange dashed), HMM (red)")])]
    story += [PageBreak()]

    # ---------------- 7. API & UI ----------------
    story += [h1("7. API, UI and deployment"), h2("7.1 Endpoints (api/main.py)")]
    ep = re.findall(r'@app\.(get|post)\("([^"]+)"\)', read(ROOT / "api" / "main.py"))
    story += [df_table(pd.DataFrame([(m.upper(), path) for m, path in ep], columns=["method", "path"]),
                       [30 * mm, 150 * mm]), src("api/main.py")]
    story += [h2("7.2 All-trips integration check through the live API")]
    for name, label in [("api_check_student_test.csv", "student_test"),
                        ("api_check_eval_sim_withroute.csv", "simulated eval (student_test copies, .route kept)"),
                        ("api_check_eval_sim_noroute.csv", "simulated eval (student_test copies, .route REMOVED - metrics null by design)")]:
        if (RES / name).exists():
            story += [p(f"<b>{label}</b>"), df_table(pd.read_csv(RES / name, dtype={"trip_id": str})), src(f"results/{name}")]
    req = read(LOGS / "api_requests.log").splitlines()
    story += [h2("7.3 Request log excerpt (Laptop 1 terminal output)"), code("\n".join(req[:14])),
              src(f"results/logs/api_requests.log ({len(req)} lines)"), PageBreak()]
    story += [h2("7.4 Evaluator UI (Streamlit, Laptop 2)")]
    story += fig("ui_home.png", max_h=100 * mm) + fig("ui_result.png", max_h=110 * mm) + [PageBreak()]

    # ---------------- 8. summary ----------------
    best = mc.set_index("Model").loc[meta.get("model", "LightGBM").split(" ")[0]] if meta.get("model", "").split(" ")[0] in set(mc.Model) else mc.iloc[0]
    tvu = pd.read_csv(RES / "tuned_vs_untuned.csv")
    api = pd.read_csv(RES / "api_check_student_test.csv")
    story += [h1("8. Key numbers")]
    story.append(kv_table([
        ("Train trips / GPS points", f"{logs['train']['trips']} / {logs['train']['points']:,}"),
        ("Student-test trips / GPS points", f"{logs['student_test']['trips']} / {logs['student_test']['points']:,}"),
        ("Training rows (point x candidate)", f"{logs['train']['candidate_rows']:,} (positive rate {logs['train']['positive_rate']})"),
        ("Labelled share train / student_test", f"{logs['train']['labelled_share']} / {logs['student_test']['labelled_share']}"),
        ("Median GPS-to-true-arc distance (m) train / student_test", f"{logs['train']['median_label_dist_m']} / {logs['student_test']['median_label_dist_m']}"),
        ("Candidate miss rate train / student_test", f"{logs['train']['candidate_miss_rate']} / {logs['student_test']['candidate_miss_rate']}"),
        ("Models compared", f"{len(mc)} ({', '.join(mc.Model)})"),
        ("Selected model", meta.get("model", "")),
        ("Best untuned CV Top-1 / test Top-1", f"{tvu.iloc[0]['CV Top-1']} / {tvu.iloc[0]['Test Top-1']}"),
        ("Tuned CV Top-1 / test Top-1", f"{tvu.iloc[1]['CV Top-1']} / {tvu.iloc[1]['Test Top-1']}"),
        ("Classifier-only -> HMM Top-1 (mean of trips)", f"{hv.top1_classifier.mean():.4f} -> {hv.top1_hmm.mean():.4f}"),
        ("Classifier-only -> HMM continuity (mean)", f"{hv.route_continuity_classifier.mean():.4f} -> {hv.route_continuity_hmm.mean():.4f}"),
        ("HMM route F1 (mean)", f"{hv.route_f1.mean():.4f}" if "route_f1" in hv else "-"),
        ("API server time per trip (ms, min-max)", f"{api.server_ms.min():.0f} - {api.server_ms.max():.0f}"),
        ("API round-trip per trip (ms, min-max)", f"{api.round_trip_ms.min():.0f} - {api.round_trip_ms.max():.0f}"),
        ("API trips passing integration check", f"{(api.status == 'OK').sum()} / {len(api)}"),
    ], (95, 85)))
    story += [src("build_dataset logs, model_comparison.csv, tuned_vs_untuned.csv, hmm_vs_classifier.csv, api_check_student_test.csv")]

    doc = Doc(str(OUT), pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm, topMargin=14 * mm, bottomMargin=15 * mm,
              title="PathSnap - Results Book", author="Team 5")
    doc.multiBuild(story, onFirstPage=footer, onLaterPages=footer)
    print("wrote", OUT)


if __name__ == "__main__":
    build()
