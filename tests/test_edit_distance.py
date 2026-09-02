from edit_distance import (
    build_edit_distance_matrix,
    edit_distance,
    normalize_comparison_term,
)


def test_identical_strings_have_zero_distance():
    assert edit_distance("same", "same") == 0


def test_one_insertion():
    assert edit_distance("cat", "cart") == 1


def test_one_deletion():
    assert edit_distance("cart", "cat") == 1


def test_one_replacement():
    assert edit_distance("cat", "cut") == 1


def test_empty_source():
    assert edit_distance("", "abc") == 3


def test_empty_target():
    assert edit_distance("abc", "") == 3


def test_both_strings_empty():
    assert edit_distance("", "") == 0


def test_multiple_edits():
    assert edit_distance("kitten", "sitting") == 3


def test_cat_to_dog_requires_three_edits():
    assert edit_distance("cat", "dog") == 3


def test_computer_to_computer_is_zero():
    assert edit_distance("computer", "computer") == 0


def test_distance_is_symmetric():
    assert edit_distance("computer", "commuter") == edit_distance(
        "commuter", "computer"
    )


def test_matrix_dimensions_are_correct():
    matrix = build_edit_distance_matrix("cat", "cut")

    assert len(matrix) == 4
    assert all(len(row) == 4 for row in matrix)


def test_first_matrix_row_is_initialized():
    matrix = build_edit_distance_matrix("cat", "cut")

    assert matrix[0] == [0, 1, 2, 3]


def test_first_matrix_column_is_initialized():
    matrix = build_edit_distance_matrix("cat", "cut")

    assert [row[0] for row in matrix] == [0, 1, 2, 3]


def test_final_matrix_cell_equals_edit_distance():
    source = "cat"
    target = "cut"
    matrix = build_edit_distance_matrix(source, target)

    assert matrix[-1][-1] == edit_distance(source, target)


def test_transposition_is_not_a_special_operation():
    assert edit_distance("ab", "ba") == 2


def test_ir_normalization_is_separate_from_core_algorithm():
    normalized_source = normalize_comparison_term("COMPUTER!")
    normalized_target = normalize_comparison_term("computer")

    assert normalized_source == "computer"
    assert edit_distance(normalized_source, normalized_target) == 0
    assert edit_distance("COMPUTER", "computer") == 8
