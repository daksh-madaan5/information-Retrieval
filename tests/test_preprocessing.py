from pathlib import Path

from engine import build_inverted_index, load_documents, search, search_boolean
from preprocessing import (
    normalize_token,
    remove_stop_words,
    stem_tokens,
    tokenize,
)


DOCUMENTS_DIRECTORY = Path(__file__).parent.parent / "documents"


def test_normalize_token_lowercases_text():
    assert normalize_token("COMPUTER") == "computer"


def test_tokenize_still_handles_punctuation():
    assert tokenize("COMPUTER! Search, engine.") == [
        "computer",
        "search",
        "engine",
    ]


def test_stop_word_helper_removes_listed_words():
    tokens = ["the", "computer", "is", "useful"]

    assert remove_stop_words(tokens) == ["computer", "useful"]


def test_stop_word_helper_retains_non_stop_words():
    tokens = ["automated", "computer", "systems", "results"]

    assert remove_stop_words(tokens) == tokens


def test_main_index_still_keeps_stop_words():
    documents = load_documents(DOCUMENTS_DIRECTORY)
    inverted_index, _ = build_inverted_index(documents)

    assert inverted_index["the"] == [5]
    assert "and" in inverted_index


def test_stemming_is_deterministic():
    tokens = ["automated", "computing", "systems"]

    assert stem_tokens(tokens) == ["autom", "comput", "system"]
    assert stem_tokens(tokens) == stem_tokens(tokens)


def test_related_word_forms_produce_common_stems():
    assert stem_tokens(["automate", "automated", "automation"]) == [
        "autom",
        "autom",
        "autom",
    ]
    assert stem_tokens(["computer", "computing", "computational"]) == [
        "comput",
        "comput",
        "comput",
    ]


def test_stemming_does_not_change_the_main_index():
    documents = load_documents(DOCUMENTS_DIRECTORY)
    inverted_index, _ = build_inverted_index(documents)

    assert "computer" in inverted_index
    assert "computing" in inverted_index
    assert "computational" in inverted_index
    assert "comput" not in inverted_index
    assert search("computing", inverted_index) == [2]


def test_previous_boolean_query_behavior_is_unchanged():
    documents = load_documents(DOCUMENTS_DIRECTORY)
    inverted_index, _ = build_inverted_index(documents)

    matches = search_boolean("COMPUTER! AND search", inverted_index, documents)

    assert [document["doc_id"] for document in matches] == [1]
