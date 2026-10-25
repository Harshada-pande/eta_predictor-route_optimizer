"""Streamlit UI for Delivery ETA Predictor and Route Optimizer."""

from datetime import date, time
import os
from pathlib import Path
import random

import folium
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st
from streamlit_folium import st_folium

from optimizer.local_optima_demo import run_local_optima_demo

API_URL = os.getenv("API_URL", "http://localhost:8000")
DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"

LANDMARKS = [
    {"id": "Empire State", "lat": 40.7484, "lon": -73.9857},
    {"id": "Central Park Zoo", "lat": 40.7678, "lon": -73.9718},
    {"id": "Grand Central", "lat": 40.7527, "lon": -73.9772},
    {"id": "Rockefeller Center", "lat": 40.7587, "lon": -73.9787},
    {"id": "Flatiron Building", "lat": 40.7411, "lon": -73.9897},
    {"id": "One World Trade", "lat": 40.7127, "lon": -74.0134},
    {"id": "Wall Street", "lat": 40.7069, "lon": -74.0090},
    {"id": "Brooklyn Bridge", "lat": 40.7061, "lon": -73.9969},
    {"id": "Met Museum", "lat": 40.7794, "lon": -73.9632},
    {"id": "Guggenheim", "lat": 40.7829, "lon": -73.9589},
    {"id": "The High Line", "lat": 40.7480, "lon": -74.0048},
    {"id": "Union Square", "lat": 40.7359, "lon": -73.9911},
    {"id": "Washington Square", "lat": 40.7308, "lon": -73.9973},
    {"id": "Columbus Circle", "lat": 40.7681, "lon": -73.9819},
    {"id": "Madison Sq Garden", "lat": 40.7505, "lon": -73.9934},
]

st.set_page_config(page_title="Delivery ETA & Route Optimizer", layout="wide")
st.title("Delivery ETA Predictor & Route Optimizer")

st.sidebar.header("Depot & Departure")
depot_lat = st.sidebar.number_input("Depot Latitude", value=40.7580, format="%.4f")
depot_lon = st.sidebar.number_input("Depot Longitude", value=-73.9855, format="%.4f")
dep_date = st.sidebar.date_input("Departure Date", value=date(2016, 6, 15))
dep_time = st.sidebar.time_input("Departure Time", value=time(8, 30))
departure_iso = f"{dep_date.isoformat()}T{dep_time.strftime('%H:%M:%S')}"
algorithm = st.sidebar.selectbox("Optimizer Algorithm", ["hill_climbing"])

st.sidebar.header("Stops Configuration (Max 12)")
num_stops = st.sidebar.slider("Number of stops (N)", min_value=1, max_value=12, value=5)
seed = st.sidebar.number_input("Random Seed", value=42, step=1)


def _sample_stops(n: int, s: int | None = None) -> list[dict]:
    count = min(max(1, int(n)), len(LANDMARKS))
    if s is not None:
        return random.Random(int(s)).sample(LANDMARKS, count)
    return random.sample(LANDMARKS, count)


params = (int(num_stops), int(seed))

# Auto-sample when first loading or whenever N / Seed changes
if "sample_ver" not in st.session_state:
    st.session_state["sample_ver"] = 0
    st.session_state["stops"] = _sample_stops(num_stops, seed)
    st.session_state["last_params"] = params
elif st.session_state.get("last_params") != params:
    st.session_state["stops"] = _sample_stops(num_stops, seed)
    st.session_state["sample_ver"] += 1
    st.session_state["last_params"] = params
    st.session_state.pop("optimize_result", None)
    st.session_state.pop("last_optimized_payload", None)

if st.sidebar.button("Sample Random Stops"):
    # Generate fresh random sample when user clicks the button
    st.session_state["stops"] = _sample_stops(num_stops, None)
    st.session_state["sample_ver"] += 1
    st.session_state.pop("optimize_result", None)
    st.session_state.pop("last_optimized_payload", None)

editor_data = st.sidebar.data_editor(
    st.session_state["stops"],
    num_rows="dynamic",
    key=f"stops_editor_{st.session_state['sample_ver']}",
)


def build_map(depot, stops_coords, route_order, color="#1976D2"):
    coords = {"depot": (depot["lat"], depot["lon"]), **stops_coords}
    pts = [coords[n] for n in route_order if n in coords]
    center = (
        [sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)]
        if pts
        else [depot["lat"], depot["lon"]]
    )
    m = folium.Map(location=center, zoom_start=12)
    if len(pts) > 1:
        folium.PolyLine(pts, color=color, weight=4, opacity=0.8).add_to(m)
    depot_icon = folium.DivIcon(
        html='<div style="font-size:10px;font-weight:bold;color:white;background:#D32F2F;border-radius:4px;padding:2px 6px;border:2px solid white;box-shadow:1px 1px 3px rgba(0,0,0,0.5);">DEPOT</div>'
    )
    folium.Marker(
        [depot["lat"], depot["lon"]], tooltip="Depot (Start)", icon=depot_icon
    ).add_to(m)
    for i, node in enumerate(route_order):
        if node == "depot" or node not in coords:
            continue
        badge_html = f'<div style="font-size:11px;font-weight:bold;color:white;background:#1976D2;border-radius:50%;width:24px;height:24px;display:flex;align-items:center;justify-content:center;border:2px solid white;box-shadow:1px 1px 3px rgba(0,0,0,0.5);">{i}</div>'
        folium.Marker(
            coords[node], tooltip=f"{i}. {node}", icon=folium.DivIcon(html=badge_html)
        ).add_to(m)
    return m


tab_route, tab_demo, tab_model = st.tabs(
    ["Route Optimizer", "Local Optima Demo", "Model Comparison"]
)

with tab_route:
    raw_stops = (
        editor_data.to_dict(orient="records")
        if isinstance(editor_data, pd.DataFrame)
        else list(editor_data)
    )
    stops_list = [
        {"id": str(r["id"]).strip(), "lat": float(r["lat"]), "lon": float(r["lon"])}
        for r in raw_stops
        if str(r.get("id", "")).strip()
    ]

    if len(stops_list) > 12:
        st.error(
            f"Stops are capped at 12 in the UI (currently {len(stops_list)}). Please remove extra stops."
        )

    can_optimize = 1 <= len(stops_list) <= 12
    current_payload = {
        "depot": {"lat": float(depot_lat), "lon": float(depot_lon)},
        "stops": stops_list,
        "departure_time": departure_iso,
        "algorithm": algorithm,
    }

    if st.button("Optimize Route", type="primary", disabled=not can_optimize):
        with st.spinner("Optimizing route with ETA model..."):
            try:
                resp = requests.post(f"{API_URL}/optimize", json=current_payload, timeout=30)
                if resp.status_code == 200:
                    st.session_state["optimize_result"] = resp.json()
                    st.session_state["last_optimized_payload"] = current_payload
                elif resp.status_code == 422:
                    detail = resp.json().get("detail", resp.text)
                    errs = (
                        [
                            f"• **{' -> '.join(str(x) for x in d.get('loc', []) if x != 'body')}**: {d.get('msg', 'Invalid')}"
                            for d in detail
                        ]
                        if isinstance(detail, list)
                        else [str(detail)]
                    )
                    st.error("Validation Error (422):\n\n" + "\n".join(errs))
                else:
                    st.error(f"API Error ({resp.status_code}): {resp.text}")
            except requests.exceptions.ConnectionError:
                st.error(
                    f"API is unreachable at {API_URL}. Please ensure the FastAPI backend is running."
                )
            except requests.exceptions.RequestException as exc:
                st.error(f"API request failed: {exc}")

    result = st.session_state.get("optimize_result")
    last_payload = st.session_state.get("last_optimized_payload")

    # If configuration has changed since last run, warn the user
    if result and last_payload and last_payload != current_payload:
        st.warning(
            "⚠️ **Configuration changed!** The results below reflect your previous run. "
            "Click **'Optimize Route'** to calculate the route for your new settings."
        )

    if result:
        st.markdown("---")
        m1, m2, m3 = st.columns(3)
        m1.metric("Baseline Total", f"{result['baseline_total_minutes']} min")
        m2.metric("Optimized Total", f"{result['total_minutes']} min")
        m3.metric("Improvement", f"{result['improvement_pct']}%")

        col_before, col_after = st.columns(2)
        depot_point = {"lat": float(depot_lat), "lon": float(depot_lon)}
        stops_coords = {s["id"]: (s["lat"], s["lon"]) for s in result.get("stops", [])}

        with col_before:
            st.subheader("Before: Baseline Route")
            m_before = build_map(
                depot_point, stops_coords, result["baseline_route"], color="#757575"
            )
            st_folium(
                m_before,
                width=None,
                height=450,
                returned_objects=[],
                use_container_width=True,
                key="map_before",
            )

        with col_after:
            st.subheader("After: Optimized Route")
            m_after = build_map(
                depot_point, stops_coords, result["route"], color="#2E7D32"
            )
            st_folium(
                m_after,
                width=None,
                height=450,
                returned_objects=[],
                use_container_width=True,
                key="map_after",
            )

        st.subheader("Per-Stop Schedule")
        st.dataframe(result["stops"], use_container_width=True)

with tab_demo:
    st.caption(
        "Hill climbing from random starting routes. Costs come from a synthetic time matrix "
        "(seeded), sized to the current number of stops + depot."
    )
    starts = st.slider("Number of random starts", 10, 100, 50, key="demo_starts")
    demo_n = max(num_stops + 1, 4)
    demo = run_local_optima_demo(num_starts=starts, n_stops=demo_n, seed=int(seed))
    costs = [r["final_cost"] for r in demo]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(
        range(1, len(costs) + 1),
        costs,
        marker="o",
        linewidth=1,
        label="Final hill-climbing cost",
    )
    ax.axhline(min(costs), linestyle="--", color="green", label=f"Best = {min(costs):.2f} min")
    ax.axhline(max(costs), linestyle="--", color="red", label=f"Worst = {max(costs):.2f} min")
    ax.set_xlabel("Random starting route")
    ax.set_ylabel("Final route cost (minutes)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)
    st.write(
        f"Unique final costs: **{len(set(round(c, 6) for c in costs))}** out of {len(costs)} runs"
    )

with tab_model:
    md_path = DOCS_DIR / "model_comparison.md"
    if md_path.exists():
        content = md_path.read_text(encoding="utf-8")
        if "## Key Findings & Viva Talking Points" in content:
            content = content.split("## Key Findings & Viva Talking Points")[0].strip()
        st.markdown(content)
    else:
        st.warning("File docs/model_comparison.md not found.")
