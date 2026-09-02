from copy import deepcopy
from pathlib import Path

import pytest

import thesaurus
from engine import build_inverted_index, load_documents
from thesaurus import (
    WORDNET_SETUP_MESSAGE,
    expand_query_term,
    get_wordnet_synonyms,
    search_with_thesaurus,
)


class FakeSynset:
    def __init__(self, lemma_names):
        self._lemma_names = lemma_names

    def lemma_names(self):
        return list(self._lemma_names)


def set_fake_synsets(monkeypatch, lemma_groups):
    fake_synsets = [FakeSynset(group) for group in lemma_groups]
    monkeypatch.setattr(
        thesaurus.wordnet,
        "synsets",
        lambda term: fake_synsets,
    )


def test_original_term_is_excluded_from_synonyms(monkeypatch):
    set_fake_synsets(monkeypatch, [["car", "automobile"]])

    assert get_wordnet_synonyms("car") == ["automobile"]


def test_duplicate_lemma_names_are_removed(monkeypatch):
    set_fake_synsets(
        monkeypatch,
        [["automobile", "auto"], ["automobile", "AUTO"]],
    )

    assert get_wordnet_synonyms("car") == ["auto", "automobile"]


def test_lemma_names_are_lowercased(monkeypatch):
    set_fake_synsets(monkeypatch, [["AUTOMOBILE"]])

    assert get_wordnet_synonyms("CAR") == ["automobile"]


def test_multiword_underscore_lemmas_are_excluded(monkeypatch):
    set_fake_synsets(
        monkeypatch,
        [["automobile", "motor_vehicle", "railroad_car"]],
    )

    assert get_wordnet_synonyms("car") == ["automobile"]


def test_synonym_order_is_deterministic(monkeypatch):
    set_fake_synsets(monkeypatch, [["zeta", "alpha", "beta"]])

    first = get_wordnet_synonyms("term")
    second = get_wordnet_synonyms("term")

    assert first == second == ["alpha", "beta", "zeta"]


def test_max_synonyms_is_applied_after_sorting(monkeypatch):
    set_fake_synsets(monkeypatch, [["zeta", "alpha", "beta", "alpha"]])

    assert get_wordnet_synonyms("term", max_synonyms=2) == [
        "alpha",
        "beta",
    ]


@pytest.mark.parametrize("maximum", [-1, 1.5, True])
def test_invalid_max_synonyms_is_rejected(maximum):
    with pytest.raises(ValueError, match="non-negative integer"):
        get_wordnet_synonyms("car", max_synonyms=maximum)


def test_zero_max_synonyms_returns_empty_list(monkeypatch):
    set_fake_synsets(monkeypatch, [["automobile"]])

    assert get_wordnet_synonyms("car", max_synonyms=0) == []


def test_original_query_is_always_in_expansion(monkeypatch):
    monkeypatch.setattr(thesaurus, "get_wordnet_synonyms", lambda *args, **kwargs: [])

    assert expand_query_term("CAR!", {}) == ["car"]


def test_only_indexed_synonyms_enter_expansion(monkeypatch):
    monkeypatch.setattr(
        thesaurus,
        "get_wordnet_synonyms",
        lambda *args, **kwargs: ["auto", "automobile", "motorcar"],
    )
    inverted_index = {"car": [1], "automobile": [2]}

    details = {}
    expansion = expand_query_term("car", inverted_index, details=details)

    assert expansion == ["car", "automobile"]
    assert details["wordnet_synonyms"] == [
        "auto",
        "automobile",
        "motorcar",
    ]


def test_expansion_removes_duplicate_terms(monkeypatch):
    monkeypatch.setattr(
        thesaurus,
        "get_wordnet_synonyms",
        lambda *args, **kwargs: ["automobile", "automobile"],
    )
    inverted_index = {"car": [1], "automobile": [2]}

    assert expand_query_term("car", inverted_index) == [
        "car",
        "automobile",
    ]


def test_expansion_is_not_recursive(monkeypatch):
    calls = []

    def recording_synonyms(term, max_synonyms=5):
        calls.append(term)
        return ["automobile"]

    monkeypatch.setattr(thesaurus, "get_wordnet_synonyms", recording_synonyms)

    expand_query_term("car", {"car": [1], "automobile": [2]})

    assert calls == ["car"]


def make_search_collection():
    documents = {
        1: {"filename": "one.txt", "text": "car automobile"},
        2: {"filename": "two.txt", "text": "automobile"},
        3: {"filename": "three.txt", "text": "machine"},
        4: {"filename": "four.txt", "text": "unrelated"},
    }
    inverted_index, _ = build_inverted_index(documents)
    return documents, inverted_index


def use_car_synonyms(monkeypatch):
    monkeypatch.setattr(
        thesaurus,
        "get_wordnet_synonyms",
        lambda *args, **kwargs: ["auto", "automobile", "machine"],
    )


def test_search_unions_all_expansion_postings(monkeypatch):
    use_car_synonyms(monkeypatch)
    documents, inverted_index = make_search_collection()

    matches = search_with_thesaurus("car", inverted_index, documents)

    assert [document["doc_id"] for document in matches] == [1, 2, 3]


def test_duplicate_document_ids_appear_once(monkeypatch):
    use_car_synonyms(monkeypatch)
    documents, inverted_index = make_search_collection()

    matches = search_with_thesaurus("car", inverted_index, documents)

    assert [document["doc_id"] for document in matches].count(1) == 1


def test_final_document_ids_are_sorted(monkeypatch):
    use_car_synonyms(monkeypatch)
    documents, inverted_index = make_search_collection()

    matches = search_with_thesaurus("car", inverted_index, documents)
    doc_ids = [document["doc_id"] for document in matches]

    assert doc_ids == sorted(doc_ids)


def test_raw_documents_are_not_rescanned(monkeypatch):
    use_car_synonyms(monkeypatch)
    documents, inverted_index = make_search_collection()
    for document in documents.values():
        document["text"] = ""

    matches = search_with_thesaurus("car", inverted_index, documents)

    assert [document["doc_id"] for document in matches] == [1, 2, 3]


def test_normal_inverted_index_is_not_modified(monkeypatch):
    use_car_synonyms(monkeypatch)
    documents, inverted_index = make_search_collection()
    original_index = deepcopy(inverted_index)

    search_with_thesaurus("car", inverted_index, documents)

    assert inverted_index == original_index


def test_no_synonyms_behaves_like_normal_single_term_search(monkeypatch):
    monkeypatch.setattr(thesaurus, "get_wordnet_synonyms", lambda *args, **kwargs: [])
    documents, inverted_index = make_search_collection()

    matches = search_with_thesaurus("unrelated", inverted_index, documents)

    assert [document["doc_id"] for document in matches] == [4]


def test_search_exposes_details(monkeypatch):
    use_car_synonyms(monkeypatch)
    documents, inverted_index = make_search_collection()
    details = {}

    search_with_thesaurus("CAR!", inverted_index, documents, details=details)

    assert details == {
        "query": "CAR!",
        "normalized_query": "car",
        "wordnet_synonyms": ["auto", "automobile", "machine"],
        "indexed_expansion_terms": ["car", "automobile", "machine"],
        "matching_doc_ids": [1, 2, 3],
    }


def test_missing_wordnet_has_clear_setup_message(monkeypatch):
    def missing_wordnet(term):
        raise LookupError("long NLTK resource message")

    monkeypatch.setattr(thesaurus.wordnet, "synsets", missing_wordnet)

    with pytest.raises(LookupError) as error:
        get_wordnet_synonyms("car")

    assert str(error.value) == WORDNET_SETUP_MESSAGE
    assert "python -m nltk.downloader wordnet" in str(error.value)


def test_real_corpus_car_expansion_when_wordnet_is_available():
    documents_path = Path(__file__).parents[1] / "documents"
    documents = load_documents(documents_path)
    inverted_index, _ = build_inverted_index(documents)
    details = {}

    try:
        matches = search_with_thesaurus(
            "car", inverted_index, documents, details=details
        )
    except LookupError:
        pytest.skip("Local WordNet data is not installed")

    assert details["normalized_query"] == "car"
    assert details["indexed_expansion_terms"][0] == "car"
    assert [document["doc_id"] for document in matches] == [2, 6]
