"""
Prepare the reproducible train/test feature datasets.

Train:
    synthetic_train.csv is already engineered.
    We select the exact 27 model features and add hour_sin/hour_cos.

Test:
    testing_dataset.csv is raw food-delivery data.
    It is processed through the shared cleaning and feature-building
    pipeline.

Outputs:
    data/processed/train_features.csv
    data/processed/test_features.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from preprocessing.cleaning import clean_raw_delivery_data
from preprocessing.feature_builder import FEATURE_COLUMNS, build_features


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_RAW_PATH = PROJECT_ROOT / "data" / "raw" / "synthetic_train.csv"
TEST_RAW_PATH = PROJECT_ROOT / "data" / "raw" / "testing_dataset.csv"

TRAIN_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "train_features.csv"
TEST_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "test_features.csv"


def add_cyclic_hour_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add hour_sin and hour_cos using the existing order_hour column."""
    df = df.copy()

    df["hour_sin"] = np.sin(2 * np.pi * df["order_hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["order_hour"] / 24)

    return df


def prepare_train_dataset() -> pd.DataFrame:
    """
    Prepare the already-engineered training dataset.

    synthetic_train.csv already contains the categorical encodings and
    engineered variables. We only select the model's exact features,
    generate the two cyclic hour features, and append the target.
    """

    raw = pd.read_csv(TRAIN_RAW_PATH)

    required_source_columns = [
        "dist_km",
        "order_hour",
        "weekday",
        "Weatherconditions_Cloudy",
        "Weatherconditions_Fog",
        "Weatherconditions_NaN",
        "Weatherconditions_Sandstorms",
        "Weatherconditions_Stormy",
        "Weatherconditions_Sunny",
        "Weatherconditions_Windy",
        "traffic_level",
        "Type_of_vehicle_bicycle",
        "Type_of_vehicle_electric_scooter",
        "Type_of_vehicle_motorcycle",
        "Type_of_vehicle_scooter",
        "Vehicle_condition",
        "Delivery_person_Age",
        "Delivery_person_Ratings",
        "multiple_deliveries",
        "Festival_No",
        "Festival_Unknown",
        "Festival_Yes",
        "city_type_Metropolitan",
        "city_type_Semi-Urban",
        "city_type_Urban",
        "Time_taken(min)",
    ]

    missing = [
        column
        for column in required_source_columns
        if column not in raw.columns
    ]

    if missing:
        raise ValueError(
            f"Training dataset is missing required columns: {missing}"
        )

    features = raw[
        [
            column
            for column in required_source_columns
            if column != "Time_taken(min)"
        ]
    ].copy()

    features = add_cyclic_hour_features(features)

    features["Time_taken_min"] = pd.to_numeric(
        raw["Time_taken(min)"],
        errors="coerce",
    )

    return features[list(FEATURE_COLUMNS) + ["Time_taken_min"]]


def prepare_test_dataset() -> pd.DataFrame:
    """
    Prepare the raw held-out test dataset using the shared pipeline.
    """

    raw = pd.read_csv(TEST_RAW_PATH)

    cleaned = clean_raw_delivery_data(raw)

    rows = []

    for index in cleaned.index:
        order_date = pd.to_datetime(
            cleaned.loc[index, "Order_Date"],
            dayfirst=True,
            errors="coerce",
        )

        clean_time = cleaned.loc[index, "_clean_order_datetime"]

        if pd.isna(order_date) or pd.isna(clean_time):
            raise ValueError(
                f"Could not construct order datetime for test row {index}."
            )

        # Keep the actual calendar date from Order_Date and take
        # only the time component from the cleaned order time.
        order_datetime = order_date.normalize() + (
            clean_time - clean_time.normalize()
        )

        row_features = build_features(
            dist_km=float(cleaned.loc[index, "_clean_dist_km"]),
            order_datetime=order_datetime,
            weather=str(cleaned.loc[index, "Weatherconditions"]),
            traffic=str(cleaned.loc[index, "Road_traffic_density"]),
            vehicle_type=str(cleaned.loc[index, "Type_of_vehicle"]),
            vehicle_condition=int(cleaned.loc[index, "Vehicle_condition"]),
            delivery_person_age=float(
                cleaned.loc[index, "Delivery_person_Age"]
            ),
            delivery_person_rating=float(
                cleaned.loc[index, "Delivery_person_Ratings"]
            ),
            multiple_deliveries=int(
                cleaned.loc[index, "multiple_deliveries"]
            ),
            festival=str(cleaned.loc[index, "Festival"]),
            city_type=(
                "Metropolitan"
                if str(cleaned.loc[index, "City"]).strip().lower() == "unknown"
                else str(cleaned.loc[index, "City"])
            ),
        )

        rows.append(row_features)

    features = pd.concat(rows, ignore_index=True)

    features["Time_taken_min"] = pd.to_numeric(
        cleaned["Time_taken_min"].reset_index(drop=True),
        errors="coerce",
    )

    return features[list(FEATURE_COLUMNS) + ["Time_taken_min"]]


def main() -> None:
    """Generate reproducible processed train/test datasets."""

    TRAIN_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    train_features = prepare_train_dataset()
    test_features = prepare_test_dataset()

    train_features.to_csv(TRAIN_OUTPUT_PATH, index=False)
    test_features.to_csv(TEST_OUTPUT_PATH, index=False)

    print("Data preparation completed.")
    print()
    print(f"Train shape : {train_features.shape}")
    print(f"Test shape  : {test_features.shape}")
    print(f"Feature count: {len(FEATURE_COLUMNS)}")
    print("Target      : Time_taken_min")
    print()
    print(f"Train output: {TRAIN_OUTPUT_PATH}")
    print(f"Test output : {TEST_OUTPUT_PATH}")


if __name__ == "__main__":
    main()