"""PathSnap evaluator UI (owner: Satvik). Runs on Laptop 2 and only talks to the FastAPI on Laptop 1.

    PATHSNAP_API=http://<laptop1-ip>:8000 streamlit run ui/app.py

The UI never sees reference routes: it sends a trip id and shows what the API returns.
"""
import json
import os
import time
from pathlib import Path

import requests
import streamlit as st
import streamlit.components.v1 as components

MOCK_FILE = Path(__file__).with_name("mock_response.json")
CSS_FILE = Path(__file__).with_name("style.css")
TEAM = [("Samyak Sanklecha", "PES2UG24CS436"), ("Sanyam Deven Hiran", "PES2UG24CS446"),
        ("Satvik Phadke", "PES2UG24CS450"), ("Shaurya Ishan", "PES2UG24CS464")]
STEPS = ["Candidates", "Classifier", "HMM / Viterbi", "Route"]

st.set_page_config(page_title="PathSnap - Team 5", page_icon="🗺️", layout="wide", initial_sidebar_state="collapsed")
st.markdown(f"<style>{CSS_FILE.read_text()}</style>", unsafe_allow_html=True)


# ---------- small HTML helpers ----------
def html(s):
    st.markdown(s, unsafe_allow_html=True)


def pct(v):
    return "n/a" if v is None else f"{v * 100:.1f}%"


def card(label, value, sub="", highlight=False):
    cls = "metric hl" if highlight else "metric"
    return f'<div class="{cls}"><div class="m-label">{label}</div><div class="m-value">{value}</div><div class="m-sub">{sub}</div></div>'


def pipeline_strip(active=None, done=False):
    items = []
    for i, s in enumerate(STEPS):
        state = "done" if done or (active is not None and i < active) else "active" if i == active else ""
        items.append(f'<div class="step {state}"><span class="n">{i + 1}</span>{s}</div>')
    return '<div class="pipeline">' + '<div class="arrow">→</div>'.join(items) + "</div>"


@st.cache_data(ttl=30, show_spinner=False)
def get_trips(url):
    r = requests.get(f"{url}/trips", timeout=10)
    r.raise_for_status()
    return r.json()["trips"]


def check_health(url):
    try:
        r = requests.get(f"{url}/health", timeout=3)
        r.raise_for_status()
        return True, r.json()
    except Exception as e:
        return False, str(e)


# ---------- header ----------
team = " · ".join(f"{n} <span>{s}</span>" for n, s in TEAM)
html(f"""<div class="hero"><div class="hero-title">PathSnap</div>
<div class="hero-sub">GPS map-matching &amp; route reconstruction</div>
<div class="hero-team">Team 5 · {team}</div></div>""")

# ---------- settings (sidebar, collapsed by default) ----------
st.sidebar.header("Settings")
api_url = st.sidebar.text_input("API base URL (Laptop 1)", os.environ.get("PATHSNAP_API", "http://127.0.0.1:8000")).strip().rstrip("/")
mock = st.sidebar.checkbox("Mock mode (offline UI test)", value=False)

left, right = st.columns([1, 2.4], gap="large")

with left:
    ok, info = (True, "mock") if mock else check_health(api_url)
    dot = "ok" if ok else "bad"
    html(f'<div class="pill {dot}"><span class="dot"></span>{"API connected" if ok else "API not reachable"}'
         f'<code>{"mock mode" if mock else api_url}</code></div>')

    trips, err = [], None
    if mock:
        trips = [json.loads(MOCK_FILE.read_text())["trip_id"]]
    elif ok:
        try:
            trips = get_trips(api_url)
        except Exception as e:
            err = f"Could not load the trip list: {e}"
    else:
        err = (f"Cannot reach the API at {api_url}. Start uvicorn on Laptop 1, keep both laptops on the same network, "
               f"and set the URL in the sidebar (›).")

    if err:
        html(f'<div class="errcard"><b>Connection problem</b><br>{err}</div>')
    elif not trips:
        html('<div class="errcard"><b>No trips</b><br>Copy the evaluation trip folders into data/eval/ on Laptop 1.</div>')

    trip_id = st.selectbox(f"Evaluation trip ({len(trips)} available)", trips, disabled=not trips)
    run = st.button("Reconstruct route", type="primary", use_container_width=True, disabled=not trips)
    strip = st.empty()
    strip.markdown(pipeline_strip(done="res" in st.session_state), unsafe_allow_html=True)
    html('<div class="note">Accuracy is computed on Laptop 1. Reference routes never leave the server; '
         'the map shows the GPS observations and the predicted route only.</div>')

if run and trip_id:
    t0 = time.time()
    try:
        # one API call runs the whole pipeline on Laptop 1; the strip shows it is in progress
        strip.markdown(pipeline_strip(active=0), unsafe_allow_html=True)
        with left, st.spinner(f"Laptop 1 is running candidates → classifier → HMM for trip {trip_id}..."):
            if mock:
                res = json.loads(MOCK_FILE.read_text())
            else:
                r = requests.post(f"{api_url}/predict", json={"trip_id": trip_id}, timeout=300)
                r.raise_for_status()
                res = r.json()
        st.session_state.update(res=res, rtt=(time.time() - t0) * 1000, error=None)
        strip.markdown(pipeline_strip(done=True), unsafe_allow_html=True)
    except requests.Timeout:
        st.session_state.update(res=None, error="The API did not answer within 300 s.")
    except Exception as e:
        st.session_state.update(res=None, error=f"API call failed: {e}")

res = st.session_state.get("res")
with right:
    if st.session_state.get("error"):
        html(f'<div class="errcard"><b>Prediction failed</b><br>{st.session_state["error"]}</div>')
    if not res:
        html('<div class="empty"><div class="empty-title">No route yet</div>'
             '<div>Pick an evaluation trip on the left and press <b>Reconstruct route</b>. The API will match every GPS '
             'point to a road, decode a connected route with the HMM, and report its accuracy.</div></div>')
    else:
        m = res.get("metrics", {})
        lab = res.get("n_labelled_points")
        html(f'<div class="trip-head">Trip <b>{res["trip_id"]}</b> · {res["n_points"]:,} GPS points'
             f'{f" · {lab:,} with reference labels" if lab else " · no reference labels on server"}</div>')
        chip = ""
        if m.get("top1_hmm") is not None and m.get("top1_classifier") is not None:
            d = (m["top1_hmm"] - m["top1_classifier"]) * 100
            chip = f'<span class="chip {"up" if d >= 0 else "down"}">{"↑" if d >= 0 else "↓"} {d:+.1f} pts vs classifier</span>'
        cont = f'{pct(m.get("route_continuity_classifier"))} → <b>{pct(m.get("route_continuity_hmm"))}</b>'
        inf = m.get("inference_ms")
        html('<div class="grid">'
             + card("Top-1 accuracy · HMM / Viterbi", pct(m.get("top1_hmm")), chip, highlight=True)
             + card("Top-1 accuracy · classifier only", pct(m.get("top1_classifier")), "argmax per GPS point")
             + card("Route continuity", pct(m.get("route_continuity_hmm")), f"classifier → HMM: {cont}")
             + card("Candidate miss rate", pct(m.get("candidate_miss_rate")), "true road not among candidates")
             + card("Server inference", "n/a" if inf is None else f"{inf:,.0f} ms",
                    f'round trip {st.session_state["rtt"]:,.0f} ms')
             + "</div>")
        html('<div class="mapcard">')
        components.html(res["map_html"], height=560)
        html("</div>")

if res:
    t1, t2 = st.tabs([f"Predicted route ({len(res.get('predicted_route', []))} arcs)", "Raw API response"])
    with t1:
        st.code(", ".join(map(str, res.get("predicted_route", []))) or "(empty)", language=None)
    with t2:
        st.json({k: v for k, v in res.items() if k != "map_html"})
