"""Tests for frontend helpers and invariants."""

from frontend.app import LANDMARKS, build_map

NYC_LAT_MIN = 40.49
NYC_LAT_MAX = 40.92
NYC_LON_MIN = -74.27
NYC_LON_MAX = -73.68


def test_landmarks_count_and_uniqueness():
    assert len(LANDMARKS) == 15
    ids = [lm["id"] for lm in LANDMARKS]
    assert len(ids) == len(set(ids)), "Landmark IDs must be unique"


def test_landmarks_within_nyc_box():
    for lm in LANDMARKS:
        assert NYC_LAT_MIN <= lm["lat"] <= NYC_LAT_MAX, f"{lm['id']} lat out of bounds"
        assert NYC_LON_MIN <= lm["lon"] <= NYC_LON_MAX, f"{lm['id']} lon out of bounds"


def test_build_map_creates_folium_map():
    depot = {"lat": 40.7580, "lon": -73.9855}
    stops_coords = {
        "s1": (40.7484, -73.9857),
        "s2": (40.7061, -73.9969),
    }
    route = ["depot", "s1", "s2"]
    m = build_map(depot, stops_coords, route)
    assert m is not None
    html = m.get_root().render()
    assert "DEPOT" in html
    assert "s1" in html
    assert "s2" in html


def test_app_test_initial_load():
    from pathlib import Path
    from streamlit.testing.v1 import AppTest

    app_path = Path(__file__).resolve().parent.parent / "frontend" / "app.py"
    at = AppTest.from_file(str(app_path)).run()
    assert not at.exception
    assert at.title[0].value == "Delivery ETA Predictor & Route Optimizer"
    assert [t.label for t in at.tabs] == [
        "Route Optimizer",
        "Local Optima Demo",
        "Model Comparison",
    ]


