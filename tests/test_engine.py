from pathlib import Path

import pytest

from engine import (
    build_inverted_index,
    complement_postings,
    intersect_postings,
    load_documents,
    search,
    search_and,
    search_not,
    search_or,
    union_postings,
)
from preprocessing import tokenize


DOCUMENTS_DIRECTORY = Path(__file__).parent.parent / "documents"


@pytest.fixture
def indexed_collection():
    documents = load_documents(DOCUMENTS_DIRECTORY)
    inverted_index, document_frequency = build_inverted_index(documents)
    return documents, inverted_index, document_frequency


def test_all_sample_documents_are_loaded():
    documents = load_documents(DOCUMENTS_DIRECTORY)

    assert len(documents) == 6
    assert documents[1]["filename"] == "doc1.txt"
    assert documents[6]["filename"] == "doc6.txt"


def test_tokenization_lowercases_text():
    assert tokenize("Computer SYSTEMS") == ["computer", "systems"]


def test_tokenization_removes_punctuation():
    assert tokenize("Search, engine! (useful).") == ["search", "engine", "useful"]


def test_inverted_index_has_correct_postings(indexed_collection):
    _, inverted_index, _ = indexed_collection

    assert inverted_index["computer"] == [1, 2]
    assert inverted_index["search"] == [1, 3]


def test_postings_contain_each_document_id_once(indexed_collection):
    _, inverted_index, _ = indexed_collection

    # "computer" occurs twice in doc1.txt, but docID 1 appears once.
    assert inverted_index["computer"].count(1) == 1
    assert all(len(postings) == len(set(postings)) for postings in inverted_index.values())


def test_postings_are_sorted(indexed_collection):
    _, inverted_index, _ = indexed_collection

    assert all(postings == sorted(postings) for postings in inverted_index.values())


def test_document_frequency_is_correct(indexed_collection):
    _, _, document_frequency = indexed_collection

    assert document_frequency["computer"] == 2
    assert document_frequency["search"] == 2


def test_search_returns_expected_document_ids(indexed_collection):
    _, inverted_index, _ = indexed_collection

    assert search("COMPUTER!", inverted_index) == [1, 2]
    assert search("missing", inverted_index) == []


def test_search_uses_the_existing_index(indexed_collection):
    documents, inverted_index, _ = indexed_collection

    # Changing raw text after indexing does not affect index-based search.
    for document in documents.values():
        document["text"] = ""

    assert search("computer", inverted_index) == [1, 2]


def test_search_rejects_more_than_one_term(indexed_collection):
    _, inverted_index, _ = indexed_collection

    with pytest.raises(ValueError):
        search("machine learning", inverted_index)


def test_intersection_with_multiple_shared_document_ids():
    result = intersect_postings([1, 2, 4, 7], [2, 3, 4, 8])

    assert result == [2, 4]


def test_intersection_with_no_common_documents():
    assert intersect_postings([1, 3, 5], [2, 4, 6]) == []


def test_intersection_when_one_list_is_empty():
    assert intersect_postings([1, 2], []) == []
    assert intersect_postings([], [1, 2]) == []


def test_intersection_when_both_lists_are_empty():
    assert intersect_postings([], []) == []


def test_and_search_returns_expected_documents(indexed_collection):
    documents, inverted_index, _ = indexed_collection

    matches = search_and("computer", "search", inverted_index, documents)

    assert [document["doc_id"] for document in matches] == [1]
    assert [document["filename"] for document in matches] == ["doc1.txt"]


def test_and_search_normalizes_case_and_punctuation(indexed_collection):
    documents, inverted_index, _ = indexed_collection

    matches = search_and("COMPUTER!", "SEARCH?", inverted_index, documents)

    assert [document["doc_id"] for document in matches] == [1]


def test_and_search_result_is_sorted(indexed_collection):
    documents, inverted_index, _ = indexed_collection

    matches = search_and("search", "engine", inverted_index, documents)

    assert [document["doc_id"] for document in matches] == [1, 3]


def test_and_search_uses_postings_without_rescanning_text(indexed_collection):
    documents, inverted_index, _ = indexed_collection

    # The index has already been built, so changing raw text cannot change results.
    for document in documents.values():
        document["text"] = ""

    matches = search_and("computer", "search", inverted_index, documents)

    assert [document["doc_id"] for document in matches] == [1]


def test_union_with_overlapping_postings():
    result = union_postings([1, 2, 4, 7], [2, 3, 4, 8])

    assert result == [1, 2, 3, 4, 7, 8]


def test_union_with_no_overlap():
    assert union_postings([1, 3, 5], [2, 4, 6]) == [1, 2, 3, 4, 5, 6]


def test_union_when_one_list_is_empty():
    assert union_postings([1, 2], []) == [1, 2]
    assert union_postings([], [3, 4]) == [3, 4]


def test_union_when_both_lists_are_empty():
    assert union_postings([], []) == []


def test_union_does_not_repeat_duplicate_ids():
    assert union_postings([1, 1, 2, 4], [2, 2, 3, 4]) == [1, 2, 3, 4]


def test_union_output_is_sorted():
    result = union_postings([1, 4, 7], [2, 3, 8])

    assert result == sorted(result)


def test_or_search_returns_expected_documents(indexed_collection):
    documents, inverted_index, _ = indexed_collection

    matches = search_or("computer", "search", inverted_index, documents)

    assert [document["doc_id"] for document in matches] == [1, 2, 3]
    assert [document["filename"] for document in matches] == [
        "doc1.txt",
        "doc2.txt",
        "doc3.txt",
    ]


def test_or_search_normalizes_case_and_punctuation(indexed_collection):
    documents, inverted_index, _ = indexed_collection

    matches = search_or("COMPUTER!", "SEARCH?", inverted_index, documents)

    assert [document["doc_id"] for document in matches] == [1, 2, 3]


def test_or_search_uses_postings_without_rescanning_text(indexed_collection):
    documents, inverted_index, _ = indexed_collection

    # The index has already been built, so changing raw text cannot change results.
    for document in documents.values():
        document["text"] = ""

    matches = search_or("computer", "search", inverted_index, documents)

    assert [document["doc_id"] for document in matches] == [1, 2, 3]


def test_normal_complement():
    result = complement_postings([1, 2, 3, 4, 5, 6], [1, 2])

    assert result == [3, 4, 5, 6]


def test_complement_with_empty_postings_returns_all_documents():
    assert complement_postings([1, 2, 3], []) == [1, 2, 3]


def test_complement_with_all_documents_returns_empty_result():
    assert complement_postings([1, 2, 3], [1, 2, 3]) == []


def test_complement_excludes_first_document():
    assert complement_postings([1, 2, 3, 4], [1]) == [2, 3, 4]


def test_complement_excludes_last_document():
    assert complement_postings([1, 2, 3, 4], [4]) == [1, 2, 3]


def test_complement_excludes_separated_documents():
    assert complement_postings([1, 2, 3, 4, 5, 6], [2, 4, 6]) == [1, 3, 5]


def test_complement_output_is_sorted():
    result = complement_postings([1, 2, 3, 4, 5], [2, 4])

    assert result == sorted(result)


def test_not_search_returns_expected_documents(indexed_collection):
    documents, inverted_index, _ = indexed_collection

    matches = search_not("computer", inverted_index, documents)

    assert [document["doc_id"] for document in matches] == [3, 4, 5, 6]
    assert [document["filename"] for document in matches] == [
        "doc3.txt",
        "doc4.txt",
        "doc5.txt",
        "doc6.txt",
    ]


def test_not_search_normalizes_case_and_punctuation(indexed_collection):
    documents, inverted_index, _ = indexed_collection

    matches = search_not("COMPUTER!", inverted_index, documents)

    assert [document["doc_id"] for document in matches] == [3, 4, 5, 6]


def test_not_search_for_unknown_term_returns_all_documents(indexed_collection):
    documents, inverted_index, _ = indexed_collection

    matches = search_not("nonexistentterm", inverted_index, documents)

    assert [document["doc_id"] for document in matches] == [1, 2, 3, 4, 5, 6]


def test_not_search_uses_index_without_rescanning_text(indexed_collection):
    documents, inverted_index, _ = indexed_collection

    # The index has already been built, so changing raw text cannot change results.
    for document in documents.values():
        document["text"] = "computer"

    matches = search_not("computer", inverted_index, documents)

    assert [document["doc_id"] for document in matches] == [3, 4, 5, 6]
