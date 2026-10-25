"""
Build the exact 27 features required by the trained ETA model.

The application can provide human-friendly values such as:
- distance
- date/time
- weather
- traffic
- vehicle type
- vehicle condition
- delivery person information
- festival
- city type

This module converts them into the exact feature columns expected
by the trained XGBoost model.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd


# Exact feature order used by the trained model.
FEATURE_COLUMNS = [
    "dist_km",
    "order_hour",
    "hour_sin",
    "hour_cos",
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
]


TRAFFIC_MAPPING = {
    "unknown": -1,
    "low": 0,
    "medium": 1,
    "high": 2,
    "jam": 3,
}


WEATHER_COLUMNS = {
    "cloudy": "Weatherconditions_Cloudy",
    "fog": "Weatherconditions_Fog",
    "unknown": "Weatherconditions_NaN",
    "sandstorms": "Weatherconditions_Sandstorms",
    "stormy": "Weatherconditions_Stormy",
    "sunny": "Weatherconditions_Sunny",
    "windy": "Weatherconditions_Windy",
}


VEHICLE_COLUMNS = {
    "bicycle": "Type_of_vehicle_bicycle",
    "electric scooter": "Type_of_vehicle_electric_scooter",
    "electric_scooter": "Type_of_vehicle_electric_scooter",
    "motorcycle": "Type_of_vehicle_motorcycle",
    "scooter": "Type_of_vehicle_scooter",
}


FESTIVAL_COLUMNS = {
    "no": "Festival_No",
    "yes": "Festival_Yes",
    "unknown": "Festival_Unknown",
}


CITY_COLUMNS = {
    "metropolitan": "city_type_Metropolitan",
    "semi-urban": "city_type_Semi-Urban",
    "urban": "city_type_Urban",
}


def _normalise(value: Any) -> str:
    """Convert a user-provided categorical value to lowercase text."""
    return str(value).strip().lower()


def build_features(
    *,
    dist_km: float,
    order_datetime: datetime | str,
    weather: str = "Sunny",
    traffic: str = "Medium",
    vehicle_type: str = "Motorcycle",
    vehicle_condition: int = 2,
    delivery_person_age: float = 25,
    delivery_person_rating: float = 4.5,
    multiple_deliveries: int = 0,
    festival: str = "No",
    city_type: str = "Metropolitan",
) -> pd.DataFrame:
    """
    Convert application-level inputs into the exact 27 model features.

    Parameters
    ----------
    dist_km:
        Delivery distance in kilometres.

    order_datetime:
        Delivery/order date and time. Accepts either a datetime object
        or a string understood by pandas.to_datetime().

    weather:
        One of:
        Sunny, Cloudy, Fog, Sandstorms, Stormy, Windy, Unknown

    traffic:
        One of:
        Unknown, Low, Medium, High, Jam

    vehicle_type:
        One of:
        Bicycle, Electric Scooter, Motorcycle, Scooter

    vehicle_condition:
        Dataset encoding: 0, 1, 2, or 3.

    delivery_person_age:
        Delivery person's age.

    delivery_person_rating:
        Delivery person's rating.

    multiple_deliveries:
        Number of simultaneous deliveries: 0, 1, 2, or 3.

    festival:
        No, Yes, or Unknown.

    city_type:
        Metropolitan, Semi-Urban, or Urban.
    """

    # ---------------------------------------------------------
    # Validate and parse datetime
    # ---------------------------------------------------------
    dt = pd.to_datetime(order_datetime)

    if pd.isna(dt):
        raise ValueError("Invalid order_datetime.")

    order_hour = int(dt.hour)
    weekday = int(dt.weekday())

    # Exact cyclic encoding verified from train_features.csv.
    hour_sin = float(np.sin(2 * np.pi * order_hour / 24))
    hour_cos = float(np.cos(2 * np.pi * order_hour / 24))

    # ---------------------------------------------------------
    # Validate numeric inputs
    # ---------------------------------------------------------
    if dist_km < 0:
        raise ValueError("dist_km cannot be negative.")

    if vehicle_condition not in {0, 1, 2, 3}:
        raise ValueError("vehicle_condition must be 0, 1, 2, or 3.")

    if multiple_deliveries not in {0, 1, 2, 3}:
        raise ValueError("multiple_deliveries must be 0, 1, 2, or 3.")

    if not 2.5 <= float(delivery_person_rating) <= 5.0:
        raise ValueError(
            "delivery_person_rating must be between 2.5 and 5.0."
        )

    if not 20 <= float(delivery_person_age) <= 39:
        raise ValueError(
            "delivery_person_age must be between 20 and 39."
        )

    # ---------------------------------------------------------
    # Traffic
    # ---------------------------------------------------------
    traffic_key = _normalise(traffic)

    if traffic_key not in TRAFFIC_MAPPING:
        raise ValueError(
            "traffic must be one of: "
            "Unknown, Low, Medium, High, Jam."
        )

    traffic_level = TRAFFIC_MAPPING[traffic_key]

    # ---------------------------------------------------------
    # Weather one-hot encoding
    # ---------------------------------------------------------
    weather_key = _normalise(weather)

    if weather_key not in WEATHER_COLUMNS:
        raise ValueError(
            "weather must be one of: "
            "Sunny, Cloudy, Fog, Sandstorms, Stormy, Windy, Unknown."
        )

    weather_features = {
        column: 0
        for column in WEATHER_COLUMNS.values()
    }
    weather_features[WEATHER_COLUMNS[weather_key]] = 1

    # ---------------------------------------------------------
    # Vehicle one-hot encoding
    # ---------------------------------------------------------
    vehicle_key = _normalise(vehicle_type)

    if vehicle_key not in VEHICLE_COLUMNS:
        raise ValueError(
            "vehicle_type must be one of: "
            "Bicycle, Electric Scooter, Motorcycle, Scooter."
        )

    vehicle_features = {
        column: 0
        for column in VEHICLE_COLUMNS.values()
    }
    vehicle_features[VEHICLE_COLUMNS[vehicle_key]] = 1

    # ---------------------------------------------------------
    # Festival one-hot encoding
    # ---------------------------------------------------------
    festival_key = _normalise(festival)

    if festival_key not in FESTIVAL_COLUMNS:
        raise ValueError(
            "festival must be one of: No, Yes, Unknown."
        )

    festival_features = {
        column: 0
        for column in FESTIVAL_COLUMNS.values()
    }
    festival_features[FESTIVAL_COLUMNS[festival_key]] = 1

    # ---------------------------------------------------------
    # City type one-hot encoding
    # ---------------------------------------------------------
    city_key = _normalise(city_type)

    if city_key not in CITY_COLUMNS:
        raise ValueError(
            "city_type must be one of: "
            "Metropolitan, Semi-Urban, Urban."
        )

    city_features = {
        column: 0
        for column in CITY_COLUMNS.values()
    }
    city_features[CITY_COLUMNS[city_key]] = 1

    # ---------------------------------------------------------
    # Construct exactly one row
    # ---------------------------------------------------------
    features = {
        "dist_km": float(dist_km),
        "order_hour": order_hour,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "weekday": weekday,
        **weather_features,
        "traffic_level": traffic_level,
        **vehicle_features,
        "Vehicle_condition": int(vehicle_condition),
        "Delivery_person_Age": float(delivery_person_age),
        "Delivery_person_Ratings": float(delivery_person_rating),
        "multiple_deliveries": int(multiple_deliveries),
        **festival_features,
        **city_features,
    }

    # Return exactly the same order as the trained model.
    return pd.DataFrame(
        [[features[column] for column in FEATURE_COLUMNS]],
        columns=FEATURE_COLUMNS,
    )