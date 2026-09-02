import pytest

from edit_distance import (
    build_weighted_edit_distance_matrix,
    edit_distance,
    weighted_edit_distance,
)


def test_identical_strings_have_zero_weighted_distance():
    assert weighted_edit_distance("computer", "computer") == 0.0


def test_default_weights_match_standard_edit_distance():
    assert weighted_edit_distance("cat", "cut") == float(
        edit_distance("cat", "cut")
    )
    assert weighted_edit_distance("kitten", "sitting") == float(
        edit_distance("kitten", "sitting")
    )


def test_custom_m_to_n_replacement_cost():
    weights = {("m", "n"): 0.5}

    assert weighted_edit_distance("m", "n", weights) == pytest.approx(0.5)


def test_unlisted_m_to_q_replacement_uses_default_cost():
    weights = {("m", "n"): 0.5}

    assert weighted_edit_distance("m", "q", weights) == pytest.approx(1.0)


def test_replacement_weights_are_directional():
    weights = {("m", "n"): 0.5}

    assert weighted_edit_distance("m", "n", weights) == pytest.approx(0.5)
    assert weighted_edit_distance("n", "m", weights) == pytest.approx(1.0)


def test_symmetric_weights_work_when_both_directions_are_supplied():
    weights = {
        ("m", "n"): 0.5,
        ("n", "m"): 0.5,
    }

    assert weighted_edit_distance("m", "n", weights) == pytest.approx(0.5)
    assert weighted_edit_distance("n", "m", weights) == pytest.approx(0.5)


def test_custom_insertion_cost():
    distance = weighted_edit_distance("", "abc", insertion_cost=0.25)

    assert distance == pytest.approx(0.75)


def test_custom_deletion_cost():
    distance = weighted_edit_distance("abc", "", deletion_cost=0.25)

    assert distance == pytest.approx(0.75)


def test_weighted_matrix_dimensions_are_correct():
    matrix = build_weighted_edit_distance_matrix("cat", "cut")

    assert len(matrix) == 4
    assert all(len(row) == 4 for row in matrix)


def test_first_row_uses_insertion_cost():
    matrix = build_weighted_edit_distance_matrix(
        "cat", "cut", insertion_cost=0.5
    )

    assert matrix[0] == [0.0, 0.5, 1.0, 1.5]


def test_first_column_uses_deletion_cost():
    matrix = build_weighted_edit_distance_matrix(
        "cat", "cut", deletion_cost=0.5
    )

    assert [row[0] for row in matrix] == [0.0, 0.5, 1.0, 1.5]


def test_final_matrix_cell_equals_weighted_distance():
    weights = {("m", "n"): 0.5}
    matrix = build_weighted_edit_distance_matrix("m", "n", weights)

    assert matrix[-1][-1] == weighted_edit_distance("m", "n", weights)


def test_standard_edit_distance_remains_unchanged():
    assert edit_distance("m", "n") == 1
    assert edit_distance("cat", "cut") == 1
    assert edit_distance("ab", "ba") == 2


def test_weighted_distance_has_no_transposition_operation():
    assert weighted_edit_distance("ab", "ba") == pytest.approx(2.0)
