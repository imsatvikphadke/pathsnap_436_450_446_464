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
TEAM = [("Samyak Sanklecha", "PES2UG24CS436"), ("Sanyam Deven Hiran", "PES2UG24CS446"),
        ("Satvik Phadke", "PES2UG24CS450"), ("Shaurya Ishan", "PES2UG24CS464")]

st.set_page_config(page_title="PathSnap - Team 5", page_icon="🗺️", layout="wide")
st.markdown("""<style>
h1, h2, h3 {color: #1b2a4a;}
div[data-testid="stMetricValue"] {color: #2f5d9e;}
</style>""", unsafe_allow_html=True)

st.title("PathSnap: GPS Map-Matching & Route Reconstruction")
st.caption("Team 5 | " + " | ".join(f"{n} ({s})" for n, s in TEAM))

# ---------- sidebar: where is the API? ----------
st.sidebar.header("API connection (Laptop 1)")
api_url = st.sidebar.text_input("API base URL", os.environ.get("PATHSNAP_API", "http://127.0.0.1:8000")).strip().rstrip("/")
mock = st.sidebar.checkbox("Use mock response (offline UI test)", value=False)

if st.sidebar.button("Check connection"):
    try:
        r = requests.get(f"{api_url}/health", timeout=5)
        r.raise_for_status()
        st.sidebar.success(f"Connected: {r.json()}")
    except Exception as e:
        st.sidebar.error(f"Cannot reach {api_url}/health\n\n{e}")


@st.cache_data(ttl=30, show_spinner=False)
def get_trips(url):
    r = requests.get(f"{url}/trips", timeout=10)
    r.raise_for_status()
    return r.json()["trips"]


def pct(v):
    return "n/a" if v is None else f"{v * 100:.1f}%"


# ---------- trip list ----------
if mock:
    trips = [json.loads(MOCK_FILE.read_text())["trip_id"]]
else:
    try:
        trips = get_trips(api_url)
    except Exception as e:
        st.error(f"Could not load the trip list from {api_url}/trips. Is the API running on Laptop 1 and is the URL "
                 f"correct? Both laptops must be on the same network.\n\nDetails: {e}")
        st.stop()
    if not trips:
        st.warning("The API returned no trips. Copy the evaluation trip folders into data/eval/ on Laptop 1.")
        st.stop()

trip_id = st.selectbox(f"Select one of the {len(trips)} evaluation trips", trips)

# ---------- run prediction ----------
if st.button("Reconstruct route", type="primary"):
    t0 = time.time()
    if mock:
        res = json.loads(MOCK_FILE.read_text())
    else:
        with st.spinner(f"Sending trip {trip_id} to the API..."):
            try:
                r = requests.post(f"{api_url}/predict", json={"trip_id": trip_id}, timeout=300)
                r.raise_for_status()
                res = r.json()
            except requests.Timeout:
                st.error("The API did not answer within 300 s.")
                st.stop()
            except Exception as e:
                st.error(f"API call failed: {e}")
                st.stop()
    st.session_state["res"], st.session_state["rtt"] = res, (time.time() - t0) * 1000

res = st.session_state.get("res")
if res:
    m = res.get("metrics", {})
    st.subheader(f"Trip {res['trip_id']}  ({res['n_points']} GPS points, "
                 f"{res.get('n_labelled_points') if res.get('n_labelled_points') is not None else 'n/a'} with reference labels)")
    c = st.columns(3)
    c[0].metric("Top-1 accuracy: classifier only", pct(m.get("top1_classifier")))
    delta = None
    if m.get("top1_hmm") is not None and m.get("top1_classifier") is not None:
        delta = f"{(m['top1_hmm'] - m['top1_classifier']) * 100:+.1f} pts vs classifier"
    c[1].metric("Top-1 accuracy: HMM / Viterbi", pct(m.get("top1_hmm")), delta=delta)
    c[2].metric("Candidate miss rate", pct(m.get("candidate_miss_rate")))
    c = st.columns(3)
    c[0].metric("Route continuity: classifier only", pct(m.get("route_continuity_classifier")))
    c[1].metric("Route continuity: HMM / Viterbi", pct(m.get("route_continuity_hmm")))
    inf = m.get("inference_ms")
    c[2].metric("Server inference time", "n/a" if inf is None else f"{inf:.0f} ms",
                delta=f"round trip {st.session_state['rtt']:.0f} ms", delta_color="off")
    st.caption("Accuracy is computed on the API server (Laptop 1). Reference routes never leave Laptop 1; "
               "the map shows the GPS observations and the predicted route only.")

    components.html(res["map_html"], height=620)

    with st.expander(f"Predicted route ({len(res.get('predicted_route', []))} arcs, in driving order)"):
        st.write(res.get("predicted_route", []))
    with st.expander("Raw API response (JSON)"):
        st.json({k: v for k, v in res.items() if k != "map_html"})
