from pathlib import Path

import pytest

from engine import (
    build_positional_index,
    load_documents,
    positions_within_k,
    search_proximity,
)


DOCUMENTS_DIRECTORY = Path(__file__).parent.parent / "documents"


def make_documents():
    return {
        1: {
            "filename": "synthetic1.txt",
            "text": "machine learning uses advanced computer systems",
        },
        2: {
            "filename": "synthetic2.txt",
            "text": "learning extra extra machine computer",
        },
    }


def matching_ids(term1, term2, k, positional_index, documents):
    matches = search_proximity(
        term1, term2, k, positional_index, documents
    )
    return [document["doc_id"] for document in matches]


def test_positions_within_k_match():
    assert positions_within_k([0], [1], 1) is True


def test_positions_farther_than_k_do_not_match():
    assert positions_within_k([0], [4], 3) is False


def test_exact_boundary_distance_matches():
    assert positions_within_k([0], [4], 4) is True


def test_reverse_positional_order_matches():
    assert positions_within_k([4], [0], 4) is True


def test_multiple_occurrences_are_checked():
    assert positions_within_k([0, 10], [6, 12], 2) is True


def test_one_valid_occurrence_pair_is_enough():
    assert positions_within_k([0, 20, 40], [10, 22, 60], 2) is True


def test_unknown_proximity_term_has_no_result():
    documents = make_documents()
    positional_index = build_positional_index(documents)

    assert matching_ids("machine", "unknown", 3, positional_index, documents) == []


def test_proximity_results_remain_sorted():
    documents = {2: make_documents()[2], 1: make_documents()[1]}
    positional_index = build_positional_index(documents)

    result = matching_ids("machine", "computer", 4, positional_index, documents)

    assert result == [1, 2]
    assert result == sorted(result)


def test_proximity_terms_normalize_case_and_punctuation():
    documents = make_documents()
    positional_index = build_positional_index(documents)

    assert matching_ids(
        "MACHINE!", "COMPUTER?", 4, positional_index, documents
    ) == [1, 2]


def test_distance_one_matches_adjacency_in_either_direction():
    documents = make_documents()
    positional_index = build_positional_index(documents)

    assert matching_ids("machine", "learning", 1, positional_index, documents) == [1]
    assert matching_ids("computer", "machine", 1, positional_index, documents) == [2]


@pytest.mark.parametrize("invalid_k", [0, -1, 1.5, "3", True])
def test_invalid_proximity_distances_are_rejected(invalid_k):
    documents = make_documents()
    positional_index = build_positional_index(documents)

    with pytest.raises(ValueError, match="positive integer"):
        search_proximity(
            "machine", "learning", invalid_k, positional_index, documents
        )


def test_same_term_requires_two_distinct_occurrences():
    documents = {
        1: {"filename": "once.txt", "text": "computer systems"},
        2: {
            "filename": "twice.txt",
            "text": "computer useful computer",
        },
    }
    positional_index = build_positional_index(documents)

    assert matching_ids("computer", "computer", 2, positional_index, documents) == [2]


def test_candidate_documents_come_from_positional_index():
    documents = make_documents()
    positional_index = build_positional_index(documents)

    documents[3] = {
        "filename": "changed.txt",
        "text": "machine computer",
    }

    assert matching_ids("machine", "computer", 4, positional_index, documents) == [1, 2]


def test_proximity_search_does_not_rescan_raw_text():
    documents = make_documents()
    positional_index = build_positional_index(documents)

    for document in documents.values():
        document["text"] = ""

    assert matching_ids("machine", "computer", 4, positional_index, documents) == [1, 2]
