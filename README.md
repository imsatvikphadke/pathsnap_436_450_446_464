# PathSnap: GPS Map-Matching & Route Reconstruction (Team 5)

UE24CS352A Machine Learning hackathon, PES University RR campus.

PathSnap takes the noisy GPS track of a trip and the road network around it, and returns the road segment (arc) the vehicle was on at every GPS observation, as a connected, drivable route.

**Pipeline**
1. Candidate arcs are generated for each GPS point.
2. A classifier scores each candidate.
3. An HMM + Viterbi decoder uses road connectivity to pick one coherent route.
4. A FastAPI backend serves the route on Laptop 1, and a Streamlit UI on Laptop 2 shows it.

| Member | SRN | Owns |
|---|---|---|
| Samyak Sanklecha | PES2UG24CS436 | Data loading, GPS cleaning, labels, candidates, v1 features, EDA, Folium maps |
| Sanyam Deven Hiran | PES2UG24CS446 | Feature selection, HMM + Viterbi decoder, decoder tuning |
| Satvik Phadke | PES2UG24CS450 | Candidate sensitivity study, feature analysis plots, Streamlit UI, README |
| Shaurya Ishan | PES2UG24CS464 | Model comparison (6 models), Bayesian optimisation (Optuna), final model, FastAPI backend |

## Architecture

```
 Laptop 1 (Model + API)                                    Laptop 2 (UI)
 ┌─────────────────────────────────────────────┐          ┌──────────────────────────┐
 │ data/eval/<trip>/  (.track .nodes .arcs     │          │ ui/app.py (Streamlit)    │
 │                     + .route kept server-side)│  HTTP   │  - lists the 10 trips    │
 │ clean_track -> candidates -> features        │ <──────> │  - sends trip_id         │
 │ -> LightGBM probabilities -> HMM/Viterbi     │   JSON   │  - shows map + metrics   │
 │ -> metrics vs reference (server-side only)   │          │  (never sees the route)  │
 │ api/main.py (FastAPI, uvicorn :8000)         │          └──────────────────────────┘
 └─────────────────────────────────────────────┘
```

## Repository layout

```
├── README.md, requirements.txt, .gitignore
├── data/                     NOT committed (see "Data")
│   ├── raw/train/<id>/<id>.{track,nodes,arcs,route}     80 trips
│   ├── raw/student_test/<id>/...                        10 trips
│   ├── eval/<id>/...         evaluator trips, copied in on demo day (Laptop 1)
│   └── processed/            parquet files written by build_dataset
├── src/
│   ├── common/      io.py (loaders), geo.py (projection, distances), metrics.py
│   ├── preprocess/  clean.py (GPS cleaning)
│   ├── features/    labels.py, candidates.py, features.py, build_dataset.py,
│   │                candidate_study.py, feature_plots.py, feature_selection.py
│   ├── models/      train_compare.py, tune_optuna.py, evaluate.py, predictor.py
│   ├── hmm/         transitions.py, viterbi.py, tune_decoder.py, evaluate_hmm.py
│   ├── pipeline/    run_trip.py (end-to-end prediction + server-side evaluation for one trip)
│   └── viz/         maps.py (Folium route maps), make_report_maps.py
├── eda/eda.ipynb             exploratory data analysis (executed, with outputs)
├── api/main.py               FastAPI backend (Laptop 1)
├── ui/                       app.py (Streamlit UI, Laptop 2), style.css, mock_response.json
├── .streamlit/config.toml    UI theme
├── scripts/                  run_api.sh (start API on Laptop 1), check_all_trips.py (integration check)
├── models/                   final_model.joblib, final_model_meta.json, selected_features.json, hmm config
├── results/                  CSV results, figures/, maps/, logs/, predictions/
└── report/                   final report PDF
```

## Setup

Python 3.10+.

```bash
git clone https://github.com/imsatvikphadke/pathsnap_436_450_446_464.git
cd pathsnap_436_450_446_464
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Data
The raw data is not committed. Unzip `Student folder.zip` and copy its two folders so these paths exist:

```
data/raw/train/00000001/00000001.track
data/raw/student_test/00000000/00000000.track
```

## Reproducing the results (run from the repo root, in this order)

```bash
# 1. raw trips -> candidate / feature tables (about 10 s each)
python -m src.features.build_dataset --partition train
python -m src.features.build_dataset --partition student_test

# 2. EDA: open and run eda/eda.ipynb
jupyter notebook eda/eda.ipynb

# 3. feature analysis + selection (train only)
python -m src.features.candidate_study       # candidate radius / K sensitivity -> results/candidate_stats.csv
python -m src.features.feature_plots         # results/figures/feat_*.png
python -m src.features.feature_selection     # models/selected_features.json, results/feature_selection.csv

# 4. model comparison (6 models, GroupKFold by trip) + Bayesian optimisation (Optuna TPE)
python -m src.models.train_compare           # results/model_comparison.csv, cv_folds.csv, model_*.png
python -m src.models.tune_optuna --trials 30 # results/optuna_trials.csv, tuned_vs_untuned.csv, models/final_model.joblib
```

```bash
# 5. HMM / Viterbi decoder: Bayesian tuning of decoder parameters (Optuna TPE) + evaluation
python -m src.hmm.tune_decoder --trials 30   # models/hmm_config.json, results/decoder_trials.csv
python -m src.hmm.evaluate_hmm               # results/hmm_vs_classifier.csv, results/predictions/<trip>.csv, hmm_*.png

# 6. Folium maps for the report (GPS, network, reference, classifier-only, HMM)
python -m src.viz.make_report_maps           # results/maps/trip_<id>.html, results/figures/map_<id>.png

# 7. integration check: every trip through a running API (start the API first, see below)
python scripts/check_all_trips.py http://127.0.0.1:8000
```

## Two-laptop demo (mandatory deployment)

Both laptops must be on the **same network**. Use a **phone hotspot**, because college Wi-Fi usually blocks traffic between laptops.

### Laptop 1: model + API

```bash
cd pathsnap_436_450_446_464 && git pull && source .venv/bin/activate
# copy the 10 evaluator trip folders into data/eval/  ->  data/eval/<id>/<id>.track ...
bash scripts/run_api.sh            # = PATHSNAP_PARTITION=eval uvicorn api.main:app --host 0.0.0.0 --port 8000
                                   #   and prints the URL Laptop 2 must use
# Windows: set PATHSNAP_PARTITION=eval  then  uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Find Laptop 1's IP address:
- macOS: `ipconfig getifaddr en0`
- Windows: run `ipconfig` and read the "IPv4 Address".

Allow Python / uvicorn through the firewall if the OS asks. Keep this terminal visible: it prints one log line per request (client IP, trip, Top-1, continuity, inference ms), and those logs are shown to the evaluator. They are also appended to `results/logs/api_requests.log`.

Before the evaluation, run `python scripts/check_all_trips.py http://127.0.0.1:8000` on Laptop 1. It sends every served trip through the API and validates each response.

### Laptop 2: UI

First open `http://<laptop1-ip>:8000/health` in a browser. It must return `"status": "ok"`. Then:

```bash
cd pathsnap_436_450_446_464 && source .venv/bin/activate
PATHSNAP_API=http://<laptop1-ip>:8000 streamlit run ui/app.py
# Windows (PowerShell):  $env:PATHSNAP_API="http://<laptop1-ip>:8000"; streamlit run ui/app.py
```

The API URL can also be pasted into the sidebar, where **Check connection** tests it. The UI shows the trips returned by `GET /trips`. The evaluator picks one and clicks **Reconstruct route**; the UI calls `POST /predict` and shows the map and the metrics.

**Rehearsal without the evaluator data:** on Laptop 1, run `bash scripts/run_api.sh student_test` to serve the 10 student-test trips. The UI also has a **mock mode** (sidebar checkbox) that renders `ui/mock_response.json` without any API.

### API contract

```
GET  /health   -> {"status": "ok", ...}
GET  /trips    -> {"trips": ["<trip_id>", ...]}
POST /predict  {"trip_id": "<id>"}
     -> {"trip_id", "n_points", "n_labelled_points",
         "metrics": {"top1_classifier", "top1_hmm", "route_continuity_classifier",
                     "route_continuity_hmm", "candidate_miss_rate", "inference_ms"},
         "predicted_route": [arc ids], "map_html": "<folium map: GPS + predicted route>"}
```

The reference route never leaves Laptop 1: accuracy is computed server-side and only the numbers are returned.

## Results (student_test unless stated)

### Data
- Train: 80 trips, 208,417 GPS points, 1,139,479 candidate rows.
- Candidate miss rate: 0.85% (train), 0.26% (student_test), with R = 50 m and K = 8.

### Model comparison
The comparison uses 3-fold GroupKFold by trip on train, with the 13 selected features.

| Model | CV Top-1 | Test Top-1 | Test macro-F1 | Test log loss | Route continuity (classifier only) |
|---|---|---|---|---|---|
| Logistic Regression | 0.839 | 0.850 | 0.833 | 0.255 | 0.758 |
| Gaussian NB | 0.822 | 0.827 | 0.755 | 0.741 | 0.723 |
| KNN | 0.832 | 0.848 | 0.903 | 0.187 | 0.734 |
| Random Forest | 0.846 | 0.864 | 0.914 | 0.121 | 0.765 |
| **LightGBM (selected)** | 0.846 | 0.873 | 0.918 | 0.116 | 0.775 |
| MLP | 0.840 | 0.861 | 0.912 | 0.127 | 0.760 |

### Bayesian optimisation
- Optuna TPE sampler, 30 trials, objective = CV Top-1 on train.
- The tuned LightGBM reaches CV Top-1 0.848 (untuned: 0.846).

See `results/tuned_vs_untuned.csv` for the full comparison.

### HMM / Viterbi vs classifier only
The table is the mean over the 10 student-test trips, from `results/hmm_vs_classifier.csv`.

| | Top-1 accuracy | Route continuity |
|---|---|---|
| Classifier only (argmax per point) | 0.884 | 0.769 |
| Classifier + HMM / Viterbi | **0.957** | **0.893** |

Decoder parameters were tuned with Optuna TPE (30 trials) on 20 train trips + the student-test trips, never on evaluator trips. See `models/hmm_config.json`.

## Rules we follow (no leakage)
- Evaluator trips (`data/eval`) are used only at the demo, never for training, tuning, feature selection or thresholds.
- Model choice, feature selection and hyperparameter tuning use only the 80 train trips (GroupKFold by trip). The 10 student-test trips are used only to compare models and decoder settings.
- Hyperparameter tuning is Bayesian (Optuna TPE) only, with no grid or random search.
- Candidates and features are built without reading `.route`. The route is used only to create training labels and to score predictions server-side.
- No hardcoded predictions: every API request runs the full pipeline.

## AI assistance
AI coding assistants were used while writing parts of the code and documentation. All results in this repository come from running the code in this repository.
