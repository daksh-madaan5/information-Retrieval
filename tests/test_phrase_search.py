from pathlib import Path

import pytest

from engine import (
    build_positional_index,
    load_documents,
    positional_intersect,
    search_phrase,
)


DOCUMENTS_DIRECTORY = Path(__file__).parent.parent / "documents"


def make_synthetic_documents():
    return {
        1: {
            "filename": "synthetic1.txt",
            "text": "machine learning uses machine vision and machine learning",
        },
        2: {
            "filename": "synthetic2.txt",
            "text": "a machine learning system",
        },
    }


def matching_ids(phrase, positional_index, documents):
    matches = search_phrase(phrase, positional_index, documents)
    return [document["doc_id"] for document in matches]


def test_two_term_positional_intersection():
    positions1 = [0, 5, 10]
    positions2 = [1, 6, 14]

    assert positional_intersect(positions1, positions2) == [0, 5]


def test_exact_two_word_phrase_matches():
    documents = make_synthetic_documents()
    positional_index = build_positional_index(documents)

    assert matching_ids("machine learning", positional_index, documents) == [1, 2]


def test_reverse_word_order_does_not_match():
    documents = make_synthetic_documents()
    positional_index = build_positional_index(documents)

    assert matching_ids("learning machine", positional_index, documents) == []


def test_non_adjacent_terms_do_not_match():
    documents = make_synthetic_documents()
    positional_index = build_positional_index(documents)

    assert matching_ids("machine uses", positional_index, documents) == []


def test_repeated_phrase_start_positions_are_found():
    positions1 = [0, 3, 6]
    positions2 = [1, 7]

    assert positional_intersect(positions1, positions2) == [0, 6]


def test_phrase_occurring_multiple_times_returns_document_once():
    documents = make_synthetic_documents()
    positional_index = build_positional_index(documents)

    assert matching_ids("machine learning", positional_index, documents).count(1) == 1


def test_three_word_phrase_matches():
    documents = make_synthetic_documents()
    positional_index = build_positional_index(documents)

    assert matching_ids("machine learning uses", positional_index, documents) == [1]


def test_phrase_punctuation_is_normalized():
    documents = make_synthetic_documents()
    positional_index = build_positional_index(documents)

    assert matching_ids('"machine, learning!"', positional_index, documents) == [
        1,
        2,
    ]


def test_phrase_case_is_normalized():
    documents = make_synthetic_documents()
    positional_index = build_positional_index(documents)

    assert matching_ids("MACHINE LEARNING", positional_index, documents) == [1, 2]


def test_phrase_containing_stop_word_matches():
    documents = load_documents(DOCUMENTS_DIRECTORY)
    positional_index = build_positional_index(documents)

    assert matching_ids("a computer", positional_index, documents) == [1, 2]


def test_phrase_with_unknown_term_has_no_result():
    documents = make_synthetic_documents()
    positional_index = build_positional_index(documents)

    assert matching_ids("machine unknown", positional_index, documents) == []


def test_phrase_document_results_remain_sorted():
    documents = make_synthetic_documents()
    positional_index = build_positional_index(documents)

    result = matching_ids("machine learning", positional_index, documents)

    assert result == [1, 2]
    assert result == sorted(result)


def test_phrase_search_does_not_rescan_raw_document_text():
    documents = make_synthetic_documents()
    positional_index = build_positional_index(documents)

    for document in documents.values():
        document["text"] = ""

    assert matching_ids("machine learning", positional_index, documents) == [1, 2]


def test_phrase_requires_at_least_two_tokens():
    documents = make_synthetic_documents()
    positional_index = build_positional_index(documents)

    with pytest.raises(ValueError, match="at least two terms"):
        search_phrase("machine", positional_index, documents)

    with pytest.raises(ValueError, match="at least two terms"):
        search_phrase("", positional_index, documents)
