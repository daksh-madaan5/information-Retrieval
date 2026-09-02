from pathlib import Path

import pytest

from engine import (
    build_bigram_index,
    build_inverted_index,
    get_wildcard_candidates,
    load_documents,
    matches_wildcard,
    search_wildcard,
    term_bigrams,
    wildcard_query_bigrams,
)


DOCUMENTS_DIRECTORY = Path(__file__).parent.parent / "documents"


def make_synthetic_collection():
    documents = {
        1: {"filename": "one.txt", "text": "unused"},
        2: {"filename": "two.txt", "text": "unused"},
        3: {"filename": "three.txt", "text": "unused"},
    }
    inverted_index = {
        "computer": [1, 2],
        "computing": [2],
        "computational": [3],
        "automation": [3],
        "information": [1],
        "procent": [1],
        "proxxcent": [2],
    }
    return documents, inverted_index, build_bigram_index(inverted_index)


def test_computer_produces_expected_boundary_bigrams():
    assert term_bigrams("computer") == [
        "$c",
        "co",
        "om",
        "mp",
        "pu",
        "ut",
        "te",
        "er",
        "r$",
    ]


def test_bigram_postings_contain_unique_terms():
    bigram_index = build_bigram_index({"aaaa": [1], "aa": [2]})

    assert bigram_index["aa"] == ["aa", "aaaa"]


def test_bigram_term_lists_are_sorted():
    bigram_index = build_bigram_index(
        {"computer": [1], "color": [2], "car": [3]}
    )

    assert all(terms == sorted(terms) for terms in bigram_index.values())


def test_bigram_index_uses_dictionary_terms_only():
    inverted_index = {"computer": [1]}

    bigram_index = build_bigram_index(inverted_index)

    indexed_terms = []
    for terms in bigram_index.values():
        for term in terms:
            if term not in indexed_terms:
                indexed_terms.append(term)
    assert indexed_terms == ["computer"]


def test_prefix_wildcard_matches_expected_terms():
    _, _, bigram_index = make_synthetic_collection()

    candidates = get_wildcard_candidates("comput*", bigram_index)
    matches = [term for term in candidates if matches_wildcard(term, "comput*")]

    assert matches == ["computational", "computer", "computing"]


def test_prefix_bigrams_use_left_but_not_right_boundary():
    bigrams = wildcard_query_bigrams("comput*")

    assert bigrams == ["$c", "co", "om", "mp", "pu", "ut"]
    assert "t$" not in bigrams


def test_suffix_wildcard_matches_terms_and_uses_right_boundary():
    _, _, bigram_index = make_synthetic_collection()

    candidates = get_wildcard_candidates("*tion", bigram_index)
    matches = [term for term in candidates if matches_wildcard(term, "*tion")]

    assert matches == ["automation", "information"]
    assert wildcard_query_bigrams("*tion") == ["ti", "io", "on", "n$"]


def test_middle_wildcard_matches_and_does_not_cross_star():
    _, _, bigram_index = make_synthetic_collection()

    candidates = get_wildcard_candidates("pro*cent", bigram_index)
    matches = [
        term for term in candidates if matches_wildcard(term, "pro*cent")
    ]

    assert matches == ["procent", "proxxcent"]
    assert wildcard_query_bigrams("pro*cent") == [
        "$p",
        "pr",
        "ro",
        "ce",
        "en",
        "nt",
        "t$",
    ]
    assert "oc" not in wildcard_query_bigrams("pro*cent")


def test_post_filter_removes_false_bigram_candidate():
    inverted_index = {"monkey": [1], "moon": [2]}
    bigram_index = build_bigram_index(inverted_index)

    candidates = get_wildcard_candidates("mon*", bigram_index)
    matches = [term for term in candidates if matches_wildcard(term, "mon*")]

    assert candidates == ["monkey", "moon"]
    assert matches == ["monkey"]


def test_wildcard_search_uses_normal_index_postings():
    documents, inverted_index, bigram_index = make_synthetic_collection()
    details = {}

    matches = search_wildcard(
        "comput*", bigram_index, inverted_index, documents, details
    )

    assert details["matching_terms"] == [
        "computational",
        "computer",
        "computing",
    ]
    assert [document["doc_id"] for document in matches] == [1, 2, 3]


def test_wildcard_search_removes_duplicate_document_ids():
    documents, inverted_index, bigram_index = make_synthetic_collection()

    matches = search_wildcard(
        "comput*", bigram_index, inverted_index, documents
    )

    matching_ids = [document["doc_id"] for document in matches]
    assert matching_ids.count(2) == 1


def test_wildcard_search_results_are_sorted():
    documents, inverted_index, bigram_index = make_synthetic_collection()

    matches = search_wildcard(
        "comput*", bigram_index, inverted_index, documents
    )
    matching_ids = [document["doc_id"] for document in matches]

    assert matching_ids == [1, 2, 3]
    assert matching_ids == sorted(matching_ids)


def test_wildcard_search_does_not_rescan_raw_text():
    documents, inverted_index, bigram_index = make_synthetic_collection()

    for document in documents.values():
        document["text"] = "completely changed"

    matches = search_wildcard(
        "comput*", bigram_index, inverted_index, documents
    )

    assert [document["doc_id"] for document in matches] == [1, 2, 3]


@pytest.mark.parametrize("pattern", ["computer", "*", "co**ter", "pro*cess*ing"])
def test_invalid_wildcard_patterns_are_rejected(pattern):
    documents, inverted_index, bigram_index = make_synthetic_collection()

    with pytest.raises(ValueError):
        search_wildcard(pattern, bigram_index, inverted_index, documents)


def test_wildcard_query_is_case_insensitive():
    documents, inverted_index, bigram_index = make_synthetic_collection()

    matches = search_wildcard(
        "COMPUT*", bigram_index, inverted_index, documents
    )

    assert [document["doc_id"] for document in matches] == [1, 2, 3]


def test_surrounding_punctuation_is_normalized_predictably():
    documents, inverted_index, bigram_index = make_synthetic_collection()

    matches = search_wildcard(
        '"COMPUT*!"', bigram_index, inverted_index, documents
    )

    assert [document["doc_id"] for document in matches] == [1, 2, 3]


def test_real_corpus_prefix_wildcard():
    documents = load_documents(DOCUMENTS_DIRECTORY)
    inverted_index, _ = build_inverted_index(documents)
    bigram_index = build_bigram_index(inverted_index)

    matches = search_wildcard(
        "comput*", bigram_index, inverted_index, documents
    )

    assert [document["doc_id"] for document in matches] == [1, 2]
