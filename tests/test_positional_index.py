from pathlib import Path

from engine import (
    build_inverted_index,
    build_positional_index,
    get_term_positions,
    load_documents,
)


DOCUMENTS_DIRECTORY = Path(__file__).parent.parent / "documents"


def make_synthetic_documents(text):
    return {
        1: {
            "filename": "synthetic.txt",
            "text": text,
        }
    }


def test_positional_index_structure_is_correct():
    documents = make_synthetic_documents(
        "machine learning uses machine vision"
    )

    positional_index = build_positional_index(documents)

    assert positional_index["machine"] == {1: [0, 3]}
    assert positional_index["learning"] == {1: [1]}
    assert positional_index["uses"] == {1: [2]}
    assert positional_index["vision"] == {1: [4]}


def test_first_token_has_position_zero():
    documents = make_synthetic_documents("machine learning")

    positional_index = build_positional_index(documents)

    assert positional_index["machine"] == {1: [0]}


def test_repeated_terms_store_multiple_positions():
    documents = make_synthetic_documents("computer computer systems")

    positional_index = build_positional_index(documents)

    assert positional_index["computer"] == {1: [0, 1]}


def test_stored_positions_remain_sorted():
    documents = make_synthetic_documents("term other term other term")

    positional_index = build_positional_index(documents)

    assert positional_index["term"][1] == [0, 2, 4]
    assert positional_index["term"][1] == sorted(positional_index["term"][1])


def test_positions_are_token_positions_not_character_offsets():
    documents = make_synthetic_documents(
        "machine learning uses machine vision"
    )

    positional_index = build_positional_index(documents)

    assert positional_index["machine"][1] == [0, 3]


def test_punctuation_does_not_change_token_position_rules():
    documents = make_synthetic_documents("Machine, learning! MACHINE.")

    positional_index = build_positional_index(documents)

    assert positional_index["machine"] == {1: [0, 2]}
    assert positional_index["learning"] == {1: [1]}


def test_case_normalization_is_consistent():
    documents = make_synthetic_documents("Machine machine MACHINE")

    positional_index = build_positional_index(documents)

    assert positional_index["machine"] == {1: [0, 1, 2]}


def test_stop_words_are_retained_with_positions():
    documents = make_synthetic_documents("the computer is useful")

    positional_index = build_positional_index(documents)

    assert positional_index["the"] == {1: [0]}
    assert positional_index["is"] == {1: [2]}


def test_unknown_term_position_lookup_is_empty():
    documents = make_synthetic_documents("machine learning")
    positional_index = build_positional_index(documents)

    assert get_term_positions("unknown", positional_index) == {}


def test_position_lookup_uses_index_without_rescanning_text():
    documents = make_synthetic_documents("machine learning uses machine")
    positional_index = build_positional_index(documents)

    documents[1]["text"] = ""

    assert get_term_positions("MACHINE!", positional_index) == {1: [0, 3]}


def test_positional_doc_ids_agree_with_normal_postings():
    documents = load_documents(DOCUMENTS_DIRECTORY)
    inverted_index, _ = build_inverted_index(documents)
    positional_index = build_positional_index(documents)

    for term, postings in inverted_index.items():
        positional_doc_ids = sorted(positional_index[term])
        assert positional_doc_ids == postings


def test_building_positional_index_does_not_change_normal_index():
    documents = load_documents(DOCUMENTS_DIRECTORY)
    inverted_index, _ = build_inverted_index(documents)
    original_normal_index = {
        term: list(postings) for term, postings in inverted_index.items()
    }

    build_positional_index(documents)

    assert inverted_index == original_normal_index
