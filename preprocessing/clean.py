"""
Shared cleaning rules for the raw food-delivery dataset.

This module contains the dataset-level cleaning rules that were used
to construct the train/test feature files.

It intentionally does NOT build model features.
Feature encoding remains in preprocessing.feature_builder.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def haversine_km(lat1, lon1, lat2, lon2):
    """Calculate Haversine distance in kilometres."""
    r = 6371.0

    p1 = np.radians(lat1)
    p2 = np.radians(lat2)

    dp = p2 - p1
    dl = np.radians(lon2 - lon1)

    a = (
        np.sin(dp / 2) ** 2
        + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    )

    return 2 * r * np.arcsin(np.sqrt(a))


def clean_raw_delivery_data(d: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the shared raw-data cleaning rules.

    The returned DataFrame keeps the original columns but replaces
    values that require cleaning.
    """
    d = d.copy()

    # ---------------------------------------------------------
    # String columns
    # ---------------------------------------------------------
    def s(column):
        return d[column].astype(str).str.strip()

    # ---------------------------------------------------------
    # Distance-related coordinate correction
    #
    # Preserve the existing dataset-generation rule:
    # if the raw Haversine distance is > 25 km, recompute using
    # absolute coordinate values.
    # ---------------------------------------------------------
    raw_distance = haversine_km(
        d["Restaurant_latitude"],
        d["Restaurant_longitude"],
        d["Delivery_location_latitude"],
        d["Delivery_location_longitude"],
    )

    fixed_distance = haversine_km(
        d["Restaurant_latitude"].abs(),
        d["Restaurant_longitude"].abs(),
        d["Delivery_location_latitude"].abs(),
        d["Delivery_location_longitude"].abs(),
    )

    d["_clean_dist_km"] = np.where(
        raw_distance > 25,
        fixed_distance,
        raw_distance,
    )

    # ---------------------------------------------------------
    # Order time
    #
    # Missing order time:
    # pickup time - 10 minutes.
    # ---------------------------------------------------------
    ordered = pd.to_datetime(
        s("Time_Orderd"),
        format="%H:%M:%S",
        errors="coerce",
    )

    picked = pd.to_datetime(
        s("Time_Order_picked"),
        format="%H:%M:%S",
        errors="coerce",
    )

    d["_clean_order_datetime"] = ordered.fillna(
        picked - pd.Timedelta(minutes=10)
    )

    # ---------------------------------------------------------
    # Age
    #
    # Invalid/missing age -> training median replacement: 30.
    # ---------------------------------------------------------
    age = pd.to_numeric(
        d["Delivery_person_Age"],
        errors="coerce",
    )

    d["Delivery_person_Age"] = (
        age.where(age.between(20, 39))
        .fillna(30)
    )

    # ---------------------------------------------------------
    # Rating
    #
    # Invalid/missing rating -> training median replacement: 4.7.
    # ---------------------------------------------------------
    rating = pd.to_numeric(
        d["Delivery_person_Ratings"],
        errors="coerce",
    )

    d["Delivery_person_Ratings"] = (
        rating.where(rating.between(2.5, 5.0))
        .fillna(4.7)
    )

    # ---------------------------------------------------------
    # Multiple deliveries
    #
    # Missing value -> 1.
    # ---------------------------------------------------------
    d["multiple_deliveries"] = (
        pd.to_numeric(
            d["multiple_deliveries"],
            errors="coerce",
        )
        .fillna(1)
        .astype(int)
    )

    # ---------------------------------------------------------
    # Weather
    #
    # Normalize:
    # "conditions Sunny" -> "Sunny"
    # "conditions NaN"   -> "Unknown"
    # ---------------------------------------------------------
    weather = (
        s("Weatherconditions")
        .str.replace("conditions ", "", regex=False)
        .replace("NaN", "Unknown")
    )

    d["Weatherconditions"] = weather

    # ---------------------------------------------------------
    # Traffic
    #
    # Keep the human-readable category here.
    # Feature encoding maps Unknown -> -1.
    # ---------------------------------------------------------
    traffic = s("Road_traffic_density").replace(
        "NaN",
        "Unknown",
    )

    d["Road_traffic_density"] = traffic

    # ---------------------------------------------------------
    # Vehicle type
    # ---------------------------------------------------------
    d["Type_of_vehicle"] = s("Type_of_vehicle")

    # ---------------------------------------------------------
    # Festival
    #
    # Missing/NaN -> Unknown.
    # ---------------------------------------------------------
    d["Festival"] = s("Festival").replace(
        "NaN",
        "Unknown",
    )

    # ---------------------------------------------------------
    # City
    #
    # Dataset spelling "Metropolitian" is normalized to
    # "Metropolitan".
    #
    # NaN is retained as Unknown here. The feature builder can
    # decide how that category is represented.
    # ---------------------------------------------------------
    city = s("City").replace(
        "Metropolitian",
        "Metropolitan",
    )

    city = city.replace(
        "NaN",
        "Unknown",
    )

    d["City"] = city

    return d