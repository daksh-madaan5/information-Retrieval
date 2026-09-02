from pathlib import Path

import pytest

from engine import build_inverted_index, load_documents
from soundex import (
    build_soundex_index,
    get_soundex_terms,
    search_soundex,
    soundex,
)


def test_herman_uses_lecture_code():
    assert soundex("Herman") == "H655"


def test_hermann_matches_herman():
    assert soundex("Hermann") == "H655"
    assert soundex("Hermann") == soundex("Herman")


@pytest.mark.parametrize("term", ["a", "Robert", "Washington"])
def test_valid_soundex_output_always_has_four_characters(term):
    assert len(soundex(term)) == 4


def test_first_character_is_original_first_letter_in_uppercase():
    assert soundex("robert")[0] == "R"


def test_remaining_characters_are_digits():
    assert soundex("Robert")[1:].isdigit()


def test_vowels_h_w_and_y_encode_as_removed_zeroes():
    assert soundex("Aeiouhwy") == "A000"


@pytest.mark.parametrize(
    ("letter", "expected_digit"),
    [
        ("b", "1"),
        ("f", "1"),
        ("p", "1"),
        ("v", "1"),
        ("c", "2"),
        ("g", "2"),
        ("j", "2"),
        ("k", "2"),
        ("q", "2"),
        ("s", "2"),
        ("x", "2"),
        ("z", "2"),
        ("d", "3"),
        ("t", "3"),
        ("l", "4"),
        ("m", "5"),
        ("n", "5"),
        ("r", "6"),
    ],
)
def test_letter_mapping_groups(letter, expected_digit):
    assert soundex("a" + letter) == "A" + expected_digit + "00"


def test_consecutive_duplicate_codes_are_collapsed():
    assert soundex("Abfpv") == "A100"


def test_zero_between_codes_prevents_duplicate_collapse():
    assert soundex("Abab") == "A110"


def test_short_output_is_padded():
    assert soundex("A") == "A000"


def test_long_output_is_truncated():
    assert soundex("Adlmrb") == "A345"


def test_case_does_not_change_code():
    assert soundex("Herman") == soundex("herman") == soundex("HERMAN")


def test_surrounding_punctuation_is_normalized():
    assert soundex("(Herman!)") == "H655"


@pytest.mark.parametrize("term", ["", "123", "Herman Smith", "abc123"])
def test_invalid_input_is_rejected(term):
    with pytest.raises(ValueError, match="exactly one alphabetic term"):
        soundex(term)


def test_soundex_index_is_built_from_normal_dictionary_terms():
    inverted_index = {"steven": [1], "stephen": [2], "other": [3]}

    soundex_index = build_soundex_index(inverted_index)

    assert soundex_index["S315"] == ["stephen", "steven"]


def test_soundex_index_terms_are_unique_and_sorted():
    inverted_index = {"steven": [1, 2], "stephen": [2, 3]}

    soundex_index = build_soundex_index(inverted_index)

    assert soundex_index["S315"] == ["stephen", "steven"]


def test_soundex_index_does_not_need_documents():
    inverted_index = {"herman": [1], "hermann": [2]}

    assert build_soundex_index(inverted_index)["H655"] == [
        "herman",
        "hermann",
    ]


def test_soundex_index_ignores_non_alphabetic_dictionary_terms():
    inverted_index = {"herman": [1], "2026": [2]}

    soundex_index = build_soundex_index(inverted_index)

    assert soundex_index == {"H655": ["herman"]}


def test_get_soundex_terms_returns_code_and_terms():
    soundex_index = {"S315": ["stephen", "steven"]}

    assert get_soundex_terms("Steven", soundex_index) == (
        "S315",
        ["stephen", "steven"],
    )


def make_search_collection():
    documents = {
        1: {"filename": "one.txt", "text": "Steven met Stephen."},
        2: {"filename": "two.txt", "text": "Stephen arrived."},
        3: {"filename": "three.txt", "text": "Steven left."},
    }
    inverted_index, _ = build_inverted_index(documents)
    soundex_index = build_soundex_index(inverted_index)
    return documents, inverted_index, soundex_index


def test_soundex_search_uses_normal_inverted_postings():
    documents, inverted_index, soundex_index = make_search_collection()

    matches = search_soundex(
        "Steven", soundex_index, inverted_index, documents
    )

    assert [document["doc_id"] for document in matches] == [1, 2, 3]


def test_duplicate_document_ids_are_removed():
    documents, inverted_index, soundex_index = make_search_collection()

    matches = search_soundex(
        "Stephen", soundex_index, inverted_index, documents
    )

    assert [document["doc_id"] for document in matches].count(1) == 1


def test_soundex_result_order_is_sorted():
    documents, inverted_index, soundex_index = make_search_collection()

    matches = search_soundex(
        "Steven", soundex_index, inverted_index, documents
    )

    doc_ids = [document["doc_id"] for document in matches]
    assert doc_ids == sorted(doc_ids)


def test_unknown_soundex_code_returns_no_matches():
    documents = {1: {"filename": "one.txt", "text": "Herman"}}
    inverted_index, _ = build_inverted_index(documents)
    soundex_index = build_soundex_index(inverted_index)
    details = {}

    matches = search_soundex(
        "Xavier", soundex_index, inverted_index, documents, details
    )

    assert matches == []
    assert details["matching_terms"] == []
    assert details["matching_doc_ids"] == []


def test_search_does_not_rescan_raw_document_text():
    documents, inverted_index, soundex_index = make_search_collection()
    for document in documents.values():
        document["text"] = ""

    matches = search_soundex(
        "Steven", soundex_index, inverted_index, documents
    )

    assert [document["doc_id"] for document in matches] == [1, 2, 3]


def test_search_details_are_exposed():
    documents, inverted_index, soundex_index = make_search_collection()
    details = {}

    search_soundex(
        "STEVEN!", soundex_index, inverted_index, documents, details
    )

    assert details == {
        "query": "STEVEN!",
        "normalized_query": "steven",
        "soundex_code": "S315",
        "matching_terms": ["stephen", "steven"],
        "matching_doc_ids": [1, 2, 3],
    }


def test_real_corpus_groups_steven_and_stephen():
    documents_path = Path(__file__).parents[1] / "documents"
    documents = load_documents(documents_path)
    inverted_index, _ = build_inverted_index(documents)
    soundex_index = build_soundex_index(inverted_index)

    assert soundex("Steven") == "S315"
    assert soundex("Stephen") == "S315"
    assert "steven" in soundex_index["S315"]
    assert "stephen" in soundex_index["S315"]
