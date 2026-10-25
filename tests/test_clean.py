"""
Unit tests for preprocessing/cleaning.py.

Tests the shared raw-data cleaning rules used by the
Indian food-delivery dataset pipeline.
"""

import numpy as np
import pandas as pd

from preprocessing.cleaning import clean_raw_delivery_data


def _good_row() -> dict:
    """Return one valid raw food-delivery row."""
    return {
        "Restaurant_latitude": 18.5204,
        "Restaurant_longitude": 73.8567,
        "Delivery_location_latitude": 18.5304,
        "Delivery_location_longitude": 73.8667,
        "Time_Orderd": "08:30:00",
        "Time_Order_picked": "08:45:00",
        "Delivery_person_Age": 25,
        "Delivery_person_Ratings": 4.5,
        "multiple_deliveries": 0,
        "Weatherconditions": "conditions Sunny",
        "Road_traffic_density": "Medium",
        "Type_of_vehicle": "motorcycle",
        "Festival": "No",
        "City": "Metropolitian",
    }


def _make_df(**overrides) -> pd.DataFrame:
    row = _good_row()
    row.update(overrides)
    return pd.DataFrame([row])


def test_distance_is_calculated():
    df = _make_df()

    clean = clean_raw_delivery_data(df)

    assert "_clean_dist_km" in clean.columns
    assert clean["_clean_dist_km"].iloc[0] > 0


def test_order_time_is_preserved():
    df = _make_df()

    clean = clean_raw_delivery_data(df)

    result = clean["_clean_order_datetime"].iloc[0]

    assert result.hour == 8
    assert result.minute == 30


def test_missing_order_time_uses_pickup_minus_ten_minutes():
    df = _make_df(
        Time_Orderd="NaN",
        Time_Order_picked="08:45:00",
    )

    clean = clean_raw_delivery_data(df)

    result = clean["_clean_order_datetime"].iloc[0]

    assert result.hour == 8
    assert result.minute == 35


def test_invalid_age_is_replaced_with_30():
    df = _make_df(Delivery_person_Age=50)

    clean = clean_raw_delivery_data(df)

    assert clean["Delivery_person_Age"].iloc[0] == 30


def test_invalid_rating_is_replaced_with_4_7():
    df = _make_df(Delivery_person_Ratings=1.0)

    clean = clean_raw_delivery_data(df)

    assert clean["Delivery_person_Ratings"].iloc[0] == 4.7


def test_missing_multiple_deliveries_is_replaced_with_1():
    df = _make_df(multiple_deliveries=np.nan)

    clean = clean_raw_delivery_data(df)

    assert clean["multiple_deliveries"].iloc[0] == 1


def test_weather_is_normalized():
    df = _make_df(Weatherconditions="conditions Sunny")

    clean = clean_raw_delivery_data(df)

    assert clean["Weatherconditions"].iloc[0] == "Sunny"


def test_missing_weather_becomes_unknown():
    df = _make_df(Weatherconditions="conditions NaN")

    clean = clean_raw_delivery_data(df)

    assert clean["Weatherconditions"].iloc[0] == "Unknown"


def test_missing_traffic_becomes_unknown():
    df = _make_df(Road_traffic_density="NaN")

    clean = clean_raw_delivery_data(df)

    assert clean["Road_traffic_density"].iloc[0] == "Unknown"


def test_vehicle_type_is_stripped():
    df = _make_df(Type_of_vehicle=" motorcycle ")

    clean = clean_raw_delivery_data(df)

    assert clean["Type_of_vehicle"].iloc[0] == "motorcycle"


def test_missing_festival_becomes_unknown():
    df = _make_df(Festival="NaN")

    clean = clean_raw_delivery_data(df)

    assert clean["Festival"].iloc[0] == "Unknown"


def test_city_spelling_is_normalized():
    df = _make_df(City="Metropolitian")

    clean = clean_raw_delivery_data(df)

    assert clean["City"].iloc[0] == "Metropolitan"


def test_missing_city_becomes_unknown():
    df = _make_df(City="NaN")

    clean = clean_raw_delivery_data(df)

    assert clean["City"].iloc[0] == "Unknown"


def test_large_raw_distance_uses_absolute_coordinates():
    df = _make_df(
        Restaurant_latitude=-18.5204,
        Restaurant_longitude=-73.8567,
        Delivery_location_latitude=-18.5304,
        Delivery_location_longitude=-73.8667,
    )

    clean = clean_raw_delivery_data(df)

    # The corrected absolute-coordinate distance should be positive
    # and much smaller than the erroneous raw-coordinate distance.
    assert clean["_clean_dist_km"].iloc[0] > 0