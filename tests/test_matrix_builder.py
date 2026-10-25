import numpy as np
import pytest
from unittest.mock import patch

from api.matrix_builder import build_eta_matrix


def make_context(**overrides):
    context = {
        "weather": "Sunny",
        "traffic": "Medium",
        "vehicle_type": "Motorcycle",
        "vehicle_condition": 2,
        "delivery_person_age": 25,
        "delivery_person_rating": 4.5,
        "multiple_deliveries": 0,
        "festival": "No",
        "city_type": "Metropolitan",
    }
    context.update(overrides)
    return context


def test_matrix_shape_diagonal_and_finite_values():
    coordinates = [
        (18.5204, 73.8567),  # Depot
        (18.5314, 73.8446),  # A
        (18.5074, 73.8077),  # B
        (18.5642, 73.7769),  # C
    ]

    contexts = [
        make_context(),
        make_context(),
        make_context(),
        make_context(),
    ]

    with patch(
        "api.matrix_builder.predict_one",
        return_value=15.0,
    ):
        matrix = build_eta_matrix(
            coordinates,
            "2026-10-24T11:00:00",
            contexts,
        )

    assert matrix.shape == (4, 4)
    assert np.all(np.diag(matrix) == 0)
    assert np.isfinite(matrix).all()
    assert np.all(matrix >= 0)

    # 4 nodes -> 4 * 3 directed edges.
    assert np.count_nonzero(matrix) == 12


def test_matrix_calls_model_for_every_directed_pair():
    coordinates = [
        (18.5204, 73.8567),
        (18.5314, 73.8446),
        (18.5074, 73.8077),
    ]

    contexts = [
        make_context(),
        make_context(),
        make_context(),
    ]

    with patch(
        "api.matrix_builder.predict_one",
        return_value=15.0,
    ) as mock_predict:

        matrix = build_eta_matrix(
            coordinates,
            "2026-10-24T11:00:00",
            contexts,
        )

    # 3 nodes -> 3 * 2 = 6 directed edges.
    assert mock_predict.call_count == 6

    assert matrix.shape == (3, 3)
    assert np.all(np.diag(matrix) == 0)


def test_single_coordinate_needs_no_model_call():
    coordinates = [
        (18.5204, 73.8567),
    ]

    contexts = [
        make_context(),
    ]

    with patch(
        "api.matrix_builder.predict_one"
    ) as mock_predict:

        matrix = build_eta_matrix(
            coordinates,
            "2026-10-24T11:00:00",
            contexts,
        )

    mock_predict.assert_not_called()

    assert matrix.shape == (1, 1)
    assert matrix[0, 0] == 0


def test_context_count_must_match_coordinates():
    coordinates = [
        (18.5204, 73.8567),
        (18.5314, 73.8446),
    ]

    contexts = [
        make_context(),
    ]

    with pytest.raises(
        ValueError,
        match="Expected 2 contexts, received 1",
    ):
        build_eta_matrix(
            coordinates,
            "2026-10-24T11:00:00",
            contexts,
        )


def test_invalid_latitude_is_rejected():
    coordinates = [
        (91.0, 73.8567),
    ]

    contexts = [
        make_context(),
    ]

    with pytest.raises(
        ValueError,
        match="Latitude must be between -90 and 90",
    ):
        build_eta_matrix(
            coordinates,
            "2026-10-24T11:00:00",
            contexts,
        )


def test_invalid_longitude_is_rejected():
    coordinates = [
        (18.5204, 181.0),
    ]

    contexts = [
        make_context(),
    ]

    with pytest.raises(
        ValueError,
        match="Longitude must be between -180 and 180",
    ):
        build_eta_matrix(
            coordinates,
            "2026-10-24T11:00:00",
            contexts,
        )

def test_distance_below_training_range_is_clipped():
    from api.matrix_builder import clip_distance_km

    clipped_distance, was_clipped = clip_distance_km(0.5)

    assert clipped_distance == 1.47
    assert was_clipped is True


def test_distance_above_training_range_is_clipped():
    from api.matrix_builder import clip_distance_km

    clipped_distance, was_clipped = clip_distance_km(25.0)

    assert clipped_distance == 20.97
    assert was_clipped is True


def test_distance_inside_training_range_is_unchanged():
    from api.matrix_builder import clip_distance_km

    clipped_distance, was_clipped = clip_distance_km(10.0)

    assert clipped_distance == 10.0
    assert was_clipped is False

def test_find_clipped_legs_detects_out_of_range_distance():
    from api.matrix_builder import find_clipped_legs

    coordinates = [
        (18.5204, 73.8567),
        (18.5204, 73.8568),
        (18.5204, 74.2000),
    ]

    clipped_legs = find_clipped_legs(coordinates)

    assert len(clipped_legs) > 0
    assert all("from_index" in leg for leg in clipped_legs)
    assert all("to_index" in leg for leg in clipped_legs)
    assert all("original_distance_km" in leg for leg in clipped_legs)
    assert all("clipped_distance_km" in leg for leg in clipped_legs)