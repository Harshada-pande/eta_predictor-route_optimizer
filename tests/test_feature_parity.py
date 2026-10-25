"""
Feature parity test
===================
Question answered: does the API-side feature construction give the model the SAME
27 numbers that the training/test CSVs contain?

Flow under test (the same flow the API uses):

    raw row  ->  clean_raw_delivery_data()  ->  cleaned values
             ->  preprocessing.feature_builder.build_features()  ->  27 features
             ->  compared with data/processed/test_features.csv

The sample is NOT simply the first rows. It deliberately includes rows that exercise
the cleaning rules (invalid age, invalid rating, missing order time, sign-flipped
coordinates, unknown weather/traffic/festival/city) plus random rows, so a problem in
any of those rules cannot hide.
"""

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.cleaning import clean_raw_delivery_data  # noqa: E402
from preprocessing.feature_builder import FEATURE_COLUMNS, build_features  # noqa: E402

RAW_TEST = PROJECT_ROOT / "data" / "raw" / "testing_dataset.csv"
REFERENCE_FEATURES = PROJECT_ROOT / "data" / "processed" / "test_features.csv"
TARGET_COLUMN = "Time_taken_min"   # target column name used in test_features.csv

SAMPLE_SIZE = 200      # at least this many rows are compared
PER_CASE = 40          # up to this many rows per edge case
SEED = 42


# --------------------------------------------------------------------------- helpers
def _haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def _is_missing_text(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip().str.lower()
    return series.isna() | text.isin(["nan", "", "none"])


def _edge_case_masks(raw: pd.DataFrame) -> dict:
    """Boolean masks (on the RAW data) for rows that exercise the cleaning rules."""
    age = pd.to_numeric(raw["Delivery_person_Age"], errors="coerce")
    rating = pd.to_numeric(raw["Delivery_person_Ratings"], errors="coerce")
    raw_dist = _haversine_km(
        raw["Restaurant_latitude"], raw["Restaurant_longitude"],
        raw["Delivery_location_latitude"], raw["Delivery_location_longitude"],
    )
    return {
        "invalid_or_missing_age": ~age.between(20, 39),
        "invalid_or_missing_rating": ~rating.between(2.5, 5.0),
        "missing_order_time": _is_missing_text(raw["Time_Orderd"]),
        "sign_flipped_coordinates": raw_dist > 25,
        "unknown_weather": raw["Weatherconditions"].astype(str).str.contains("nan", case=False),
        "unknown_traffic": _is_missing_text(raw["Road_traffic_density"]),
        "unknown_festival": ~raw["Festival"].astype(str).str.strip().isin(["Yes", "No"]),
        "unknown_city": _is_missing_text(raw["City"]),
    }


def _select_rows(raw: pd.DataFrame):
    """Deterministic sample: edge cases first, then random rows up to SAMPLE_SIZE."""
    rng = np.random.default_rng(SEED)
    chosen, coverage = set(), {}
    for name, mask in _edge_case_masks(raw).items():
        candidates = list(raw.index[mask.to_numpy()])
        take = rng.choice(candidates, size=min(PER_CASE, len(candidates)), replace=False) if candidates else []
        coverage[name] = len(take)
        chosen.update(int(i) for i in take)
    remaining = [i for i in raw.index if int(i) not in chosen]
    need = max(SAMPLE_SIZE - len(chosen), 0)
    chosen.update(int(i) for i in rng.choice(remaining, size=need, replace=False))
    return sorted(chosen), coverage


def _order_date(value) -> pd.Timestamp:
    if isinstance(value, (pd.Timestamp, datetime)):
        return pd.Timestamp(value)
    return pd.to_datetime(str(value).strip(), format="%d-%m-%Y")


def _api_inputs(row: pd.Series, raw_row: pd.Series) -> dict:
    """Turn one cleaned row into the arguments the API passes to build_features().

    If your cleaning keeps raw labels, normalise them HERE (one place), e.g. the
    'conditions ' prefix on weather or the 'Unknown' city.
    """
    clean_time = row["_clean_order_datetime"]            # only the time-of-day part is used
    day = _order_date(raw_row["Order_Date"])             # weekday comes from Order_Date
    order_datetime = pd.Timestamp(
        year=day.year, month=day.month, day=day.day,
        hour=clean_time.hour, minute=clean_time.minute, second=clean_time.second,
    )
    city = str(row["City"]).strip()
    if city in {"Unknown", "NaN", "nan", "Metropolitian"}:
        city = "Metropolitan"                            # same rule as the training data
    weather = str(row["Weatherconditions"]).replace("conditions ", "").strip()
    return dict(
        dist_km=float(row["_clean_dist_km"]),
        order_datetime=order_datetime,
        weather=weather,
        traffic=str(row["Road_traffic_density"]).strip(),
        vehicle_type=str(row["Type_of_vehicle"]).strip(),
        vehicle_condition=int(row["Vehicle_condition"]),
        delivery_person_age=float(row["Delivery_person_Age"]),
        delivery_person_rating=float(row["Delivery_person_Ratings"]),
        multiple_deliveries=int(row["multiple_deliveries"]),
        festival=str(row["Festival"]).strip(),
        city_type=city,
    )


# --------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def data():
    raw = pd.read_csv(RAW_TEST)
    reference = pd.read_csv(REFERENCE_FEATURES)
    cleaned = clean_raw_delivery_data(raw)
    return raw, reference, cleaned


# --------------------------------------------------------------------------- tests
def test_reference_file_has_exactly_the_model_features(data):
    _, reference, _ = data
    assert len(FEATURE_COLUMNS) == 27
    assert set(reference.columns) == set(FEATURE_COLUMNS) | {TARGET_COLUMN}


def test_raw_and_reference_rows_line_up(data):
    raw, reference, cleaned = data
    assert len(raw) == len(reference) == len(cleaned), "row counts differ"
    assert cleaned.index.equals(raw.index), "cleaning changed the row index"

    # Row order check: the delivery time in the raw file must equal the reference target.
    raw_target = raw["Time_taken(min)"].astype(str).str.extract(r"(\d+)")[0].astype(int).to_numpy()
    assert np.array_equal(raw_target, reference[TARGET_COLUMN].to_numpy()), (
        "Rows are in a different order in the raw file and in test_features.csv - "
        "positional comparison would be meaningless."
    )


def test_sample_exercises_the_cleaning_rules(data):
    raw, _, _ = data
    _, coverage = _select_rows(raw)
    must_cover = ["invalid_or_missing_age", "invalid_or_missing_rating", "missing_order_time"]
    for name in must_cover:
        assert coverage[name] > 0, f"sample contains no '{name}' rows"
    print("\nEdge-case rows in the sample:", coverage)


def test_feature_parity(data):
    raw, reference, cleaned = data
    indices, _ = _select_rows(raw)
    assert len(indices) >= SAMPLE_SIZE

    generated = []
    for idx in indices:
        features = build_features(**_api_inputs(cleaned.loc[idx], raw.loc[idx]))
        generated.append(features.iloc[0])
    generated = pd.DataFrame(generated).reset_index(drop=True)[FEATURE_COLUMNS]
    expected = reference.loc[indices, FEATURE_COLUMNS].reset_index(drop=True)

    assert list(generated.columns) == list(expected.columns)
    assert not generated.isna().any().any(), "builder produced missing values"

    problems = []
    for column in FEATURE_COLUMNS:
        got = generated[column].to_numpy(dtype=float)
        want = expected[column].to_numpy(dtype=float)
        bad = np.where(~np.isclose(got, want, rtol=1e-9, atol=1e-9))[0]
        if len(bad):
            first = bad[:3]
            problems.append(
                f"{column}: {len(bad)} mismatches; e.g. raw rows {[indices[i] for i in first]} "
                f"built {got[first].tolist()} expected {want[first].tolist()}"
            )

    assert not problems, "Feature parity FAILED:\n  " + "\n  ".join(problems)
    print(f"\nFeature parity PASSED: {len(indices)} rows x {len(FEATURE_COLUMNS)} features")