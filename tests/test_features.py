import numpy as np
import pandas as pd
import pytest

from preprocessing.feature_builder import (
    FEATURE_COLUMNS,
    build_features,
)


def test_feature_columns_count():
    assert len(FEATURE_COLUMNS) == 27


def test_feature_columns_are_unique():
    assert len(FEATURE_COLUMNS) == len(set(FEATURE_COLUMNS))


def test_build_features_returns_27_features():
    features = build_features(
        dist_km=5.0,
        order_datetime="2026-10-24 11:00:00",
    )

    assert isinstance(features, pd.DataFrame)
    assert features.shape == (1, 27)
    assert list(features.columns) == FEATURE_COLUMNS


def test_distance_is_preserved():
    features = build_features(
        dist_km=7.5,
        order_datetime="2026-10-24 11:00:00",
    )

    assert features.loc[0, "dist_km"] == pytest.approx(7.5)


def test_cyclic_hour_encoding():
    features = build_features(
        dist_km=5.0,
        order_datetime="2026-10-24 00:00:00",
    )

    assert features.loc[0, "order_hour"] == 0
    assert features.loc[0, "hour_sin"] == pytest.approx(0.0)
    assert features.loc[0, "hour_cos"] == pytest.approx(1.0)


def test_noon_cyclic_encoding():
    features = build_features(
        dist_km=5.0,
        order_datetime="2026-10-24 12:00:00",
    )

    assert features.loc[0, "order_hour"] == 12
    assert features.loc[0, "hour_sin"] == pytest.approx(0.0, abs=1e-6)
    assert features.loc[0, "hour_cos"] == pytest.approx(-1.0)


def test_weekday_encoding():
    # 2026-10-19 is Monday.
    features = build_features(
        dist_km=5.0,
        order_datetime="2026-10-19 11:00:00",
    )

    assert features.loc[0, "weekday"] == 0


def test_weather_one_hot_encoding():
    features = build_features(
        dist_km=5.0,
        order_datetime="2026-10-24 11:00:00",
        weather="Sunny",
    )

    assert features.loc[0, "Weatherconditions_Sunny"] == 1

    weather_columns = [
        column
        for column in FEATURE_COLUMNS
        if column.startswith("Weatherconditions_")
    ]

    assert features.loc[0, weather_columns].sum() == 1


def test_vehicle_one_hot_encoding():
    features = build_features(
        dist_km=5.0,
        order_datetime="2026-10-24 11:00:00",
        vehicle_type="Motorcycle",
    )

    assert features.loc[0, "Type_of_vehicle_motorcycle"] == 1

    vehicle_columns = [
        column
        for column in FEATURE_COLUMNS
        if column.startswith("Type_of_vehicle_")
    ]

    assert features.loc[0, vehicle_columns].sum() == 1


def test_festival_one_hot_encoding():
    features = build_features(
        dist_km=5.0,
        order_datetime="2026-10-24 11:00:00",
        festival="Yes",
    )

    assert features.loc[0, "Festival_Yes"] == 1

    festival_columns = [
        column
        for column in FEATURE_COLUMNS
        if column.startswith("Festival_")
    ]

    assert features.loc[0, festival_columns].sum() == 1


def test_city_one_hot_encoding():
    features = build_features(
        dist_km=5.0,
        order_datetime="2026-10-24 11:00:00",
        city_type="Urban",
    )

    assert features.loc[0, "city_type_Urban"] == 1

    city_columns = [
        column
        for column in FEATURE_COLUMNS
        if column.startswith("city_type_")
    ]

    assert features.loc[0, city_columns].sum() == 1


def test_traffic_mapping():
    traffic_values = {
        "Unknown": -1,
        "Low": 0,
        "Medium": 1,
        "High": 2,
        "Jam": 3,
    }

    for traffic, expected in traffic_values.items():
        features = build_features(
            dist_km=5.0,
            order_datetime="2026-10-24 11:00:00",
            traffic=traffic,
        )

        assert features.loc[0, "traffic_level"] == expected


def test_numeric_context_features():
    features = build_features(
        dist_km=8.25,
        order_datetime="2026-10-24 11:00:00",
        vehicle_condition=3,
        delivery_person_age=32,
        delivery_person_rating=4.8,
        multiple_deliveries=2,
    )

    assert features.loc[0, "dist_km"] == pytest.approx(8.25)
    assert features.loc[0, "Vehicle_condition"] == 3
    assert features.loc[0, "Delivery_person_Age"] == 32
    assert features.loc[0, "Delivery_person_Ratings"] == pytest.approx(4.8)
    assert features.loc[0, "multiple_deliveries"] == 2


def test_invalid_distance_is_rejected():
    with pytest.raises(ValueError, match="dist_km"):
        build_features(
            dist_km=-1,
            order_datetime="2026-10-24 11:00:00",
        )


def test_invalid_vehicle_condition_is_rejected():
    with pytest.raises(ValueError, match="vehicle_condition"):
        build_features(
            dist_km=5,
            order_datetime="2026-10-24 11:00:00",
            vehicle_condition=4,
        )


def test_invalid_age_is_rejected():
    with pytest.raises(ValueError, match="delivery_person_age"):
        build_features(
            dist_km=5,
            order_datetime="2026-10-24 11:00:00",
            delivery_person_age=50,
        )


def test_invalid_rating_is_rejected():
    with pytest.raises(ValueError, match="delivery_person_rating"):
        build_features(
            dist_km=5,
            order_datetime="2026-10-24 11:00:00",
            delivery_person_rating=2.0,
        )


def test_invalid_multiple_deliveries_is_rejected():
    with pytest.raises(ValueError, match="multiple_deliveries"):
        build_features(
            dist_km=5,
            order_datetime="2026-10-24 11:00:00",
            multiple_deliveries=4,
        )


def test_invalid_weather_is_rejected():
    with pytest.raises(ValueError, match="weather"):
        build_features(
            dist_km=5,
            order_datetime="2026-10-24 11:00:00",
            weather="Rain",
        )


def test_invalid_vehicle_type_is_rejected():
    with pytest.raises(ValueError, match="vehicle_type"):
        build_features(
            dist_km=5,
            order_datetime="2026-10-24 11:00:00",
            vehicle_type="Car",
        )


def test_invalid_festival_is_rejected():
    with pytest.raises(ValueError, match="festival"):
        build_features(
            dist_km=5,
            order_datetime="2026-10-24 11:00:00",
            festival="Maybe",
        )


def test_invalid_city_type_is_rejected():
    with pytest.raises(ValueError, match="city_type"):
        build_features(
            dist_km=5,
            order_datetime="2026-10-24 11:00:00",
            city_type="Rural",
        )


def test_all_features_are_numeric_and_finite():
    features = build_features(
        dist_km=5.0,
        order_datetime="2026-10-24 11:00:00",
    )

    assert all(
        np.issubdtype(dtype, np.number)
        for dtype in features.dtypes
    )

    assert np.isfinite(features.to_numpy()).all()