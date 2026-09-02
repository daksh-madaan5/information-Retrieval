from pathlib import Path

import pytest

from engine import (
    build_bigram_index,
    build_inverted_index,
    get_wildcard_candidates,
    load_documents,
    matches_wildcard,
    normalize_wildcard_pattern,
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


@pytest.mark.parametrize("pattern", ["computer", "*", "**", "***"])
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


def test_two_wildcards_match_expected_term():
    inverted_index = {
        "computational": [1],
        "comptixional": [2],
        "computer": [3],
    }
    bigram_index = build_bigram_index(inverted_index)

    candidates = get_wildcard_candidates("comp*tion*al", bigram_index)
    matching_terms = [
        term
        for term in candidates
        if matches_wildcard(term, "comp*tion*al")
    ]

    assert matching_terms == ["computational"]


def test_three_wildcards_are_supported():
    assert matches_wildcard("computational", "c*m*put*al") is True
    assert matches_wildcard("computer", "c*m*put*al") is False


def test_wildcards_at_both_boundaries():
    _, inverted_index, bigram_index = make_synthetic_collection()

    candidates = get_wildcard_candidates("*put*", bigram_index)
    matching_terms = [
        term for term in candidates if matches_wildcard(term, "*put*")
    ]

    assert matching_terms == ["computational", "computer", "computing"]


def test_multiple_wildcard_boundary_bigrams_are_correct():
    bigrams = wildcard_query_bigrams("comp*tion*al")

    assert bigrams == [
        "$c",
        "co",
        "om",
        "mp",
        "ti",
        "io",
        "on",
        "al",
        "l$",
    ]


def test_both_boundary_wildcards_omit_word_boundaries():
    bigrams = wildcard_query_bigrams("*put*")

    assert bigrams == ["pu", "ut"]
    assert "$p" not in bigrams
    assert "t$" not in bigrams


def test_no_bigram_crosses_multiple_wildcard_boundaries():
    bigrams = wildcard_query_bigrams("comp*tion*al")

    assert "pt" not in bigrams
    assert "na" not in bigrams


def test_middle_fixed_piece_generates_internal_bigrams():
    bigrams = wildcard_query_bigrams("comp*tion*al")

    assert "ti" in bigrams
    assert "io" in bigrams
    assert "on" in bigrams


def test_multiple_wildcard_post_filter_rejects_false_candidate():
    inverted_index = {"computational": [1], "comptixional": [2]}
    bigram_index = build_bigram_index(inverted_index)

    candidates = get_wildcard_candidates("comp*tion*al", bigram_index)
    matching_terms = [
        term
        for term in candidates
        if matches_wildcard(term, "comp*tion*al")
    ]

    assert candidates == ["comptixional", "computational"]
    assert matching_terms == ["computational"]


def test_consecutive_wildcards_are_normalized_deterministically():
    assert normalize_wildcard_pattern("comp**ing") == "comp*ing"
    assert matches_wildcard("computing", "comp**ing") is True


def test_one_character_middle_piece_uses_dictionary_universe():
    inverted_index = {"abc": [1], "axbyc": [2], "ac": [3]}
    bigram_index = build_bigram_index(inverted_index)

    candidates = get_wildcard_candidates("*b*", bigram_index)
    matching_terms = [
        term for term in candidates if matches_wildcard(term, "*b*")
    ]

    assert candidates == ["abc", "ac", "axbyc"]
    assert matching_terms == ["abc", "axbyc"]


def test_multiple_wildcard_search_returns_each_document_once():
    documents, inverted_index, bigram_index = make_synthetic_collection()

    matches = search_wildcard(
        "*put*", bigram_index, inverted_index, documents
    )

    assert [document["doc_id"] for document in matches] == [1, 2, 3]


def test_multiple_wildcard_search_does_not_scan_raw_text():
    documents, inverted_index, bigram_index = make_synthetic_collection()

    for document in documents.values():
        document["text"] = ""

    matches = search_wildcard(
        "comp*tion*al", bigram_index, inverted_index, documents
    )

    assert [document["doc_id"] for document in matches] == [3]
