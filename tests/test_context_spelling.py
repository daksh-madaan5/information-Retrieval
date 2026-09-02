import context_spelling
import pytest

from context_spelling import (
    get_context_term_candidates,
    rank_context_alternatives,
    suggest_context_correction,
)
from edit_distance import weighted_edit_distance
from engine import (
    build_inverted_index,
    build_positional_index,
)
from spelling import build_trigram_index, suggest_spelling, term_trigrams


def make_context_collection():
    documents = {
        1: {
            "filename": "flight1.txt",
            "text": "I flew from Heathrow to Narita.",
        },
        2: {
            "filename": "grammar.txt",
            "text": "We study the form and farm carefully.",
        },
        3: {
            "filename": "flight2.txt",
            "text": "They flew from Heathrow yesterday.",
        },
    }
    inverted_index, _ = build_inverted_index(documents)
    positional_index = build_positional_index(documents)
    return documents, inverted_index, positional_index


def test_form_is_a_valid_dictionary_term():
    _, inverted_index, _ = make_context_collection()

    assert "form" in inverted_index


def test_valid_form_still_gets_from_as_candidate():
    _, inverted_index, _ = make_context_collection()

    candidates = get_context_term_candidates("form", inverted_index)

    assert ("from", 2.0) in candidates


def test_real_word_error_receives_context_suggestion():
    documents, inverted_index, positional_index = make_context_collection()

    details = suggest_context_correction(
        "flew form Heathrow",
        positional_index,
        inverted_index,
        documents,
    )

    assert details["original_matching_doc_ids"] == []
    assert details["suggestion"] == "flew from heathrow"


def test_matching_original_phrase_skips_expensive_generation(monkeypatch):
    documents, inverted_index, positional_index = make_context_collection()

    def fail_if_called(*args, **kwargs):
        raise AssertionError("Candidate generation should not run")

    monkeypatch.setattr(
        context_spelling, "get_context_term_candidates", fail_if_called
    )

    details = suggest_context_correction(
        "flew from heathrow",
        positional_index,
        inverted_index,
        documents,
    )

    assert details["original_matching_doc_ids"] == [1, 3]
    assert details["suggestion"] is None
    assert details["reason"] == "original phrase already matches"


def test_each_generated_alternative_changes_one_token():
    documents, inverted_index, positional_index = make_context_collection()

    details = suggest_context_correction(
        "flew form Heathrow",
        positional_index,
        inverted_index,
        documents,
    )
    original_tokens = details["normalized_tokens"]

    for alternative in details["alternatives_tested"]:
        alternative_tokens = alternative["phrase"].split()
        changed_count = sum(
            original != changed
            for original, changed in zip(original_tokens, alternative_tokens)
        )
        assert changed_count == 1


def test_no_two_word_simultaneous_corrections_are_generated():
    documents, inverted_index, positional_index = make_context_collection()

    details = suggest_context_correction(
        "flew form Heathrow",
        positional_index,
        inverted_index,
        documents,
    )

    assert all(
        alternative["phrase"] != "flow from heathrow"
        for alternative in details["alternatives_tested"]
    )


def test_context_candidates_obey_maximum_distance():
    inverted_index = {"cat": [1], "cut": [2], "dog": [3]}

    candidates = get_context_term_candidates(
        "cat", inverted_index, max_edit_distance=1
    )

    assert ("cut", 1.0) in candidates
    assert all(distance <= 1 for _, distance in candidates)
    assert "dog" not in [term for term, _ in candidates]


def test_context_candidates_exclude_original_term():
    inverted_index = {"cat": [1], "cut": [2]}

    candidates = get_context_term_candidates("cat", inverted_index)

    assert "cat" not in [term for term, _ in candidates]


def test_context_candidate_limit_is_respected():
    inverted_index = {
        "cat": [1],
        "bat": [2],
        "cap": [3],
        "car": [4],
        "cut": [5],
    }

    candidates = get_context_term_candidates(
        "cat", inverted_index, maximum_candidates=2
    )

    assert len(candidates) == 2


def test_context_candidate_sorting_is_deterministic():
    inverted_index = {"cat": [1], "cut": [2], "bat": [3]}

    candidates = get_context_term_candidates("cat", inverted_index)

    assert candidates == [("bat", 1.0), ("cut", 1.0)]


def test_zero_hit_alternatives_are_discarded_from_successes():
    documents, inverted_index, positional_index = make_context_collection()

    details = suggest_context_correction(
        "flew form Heathrow",
        positional_index,
        inverted_index,
        documents,
    )

    assert any(
        alternative["hit_count"] == 0
        for alternative in details["alternatives_tested"]
    )
    assert all(
        alternative["hit_count"] > 0
        for alternative in details["successful_alternatives"]
    )


def test_candidate_phrase_with_hits_survives():
    documents, inverted_index, positional_index = make_context_collection()

    details = suggest_context_correction(
        "flew form Heathrow",
        positional_index,
        inverted_index,
        documents,
    )

    successful_phrases = [
        alternative["phrase"]
        for alternative in details["successful_alternatives"]
    ]
    assert "flew from heathrow" in successful_phrases


def test_more_document_hits_rank_first():
    alternatives = [
        {"phrase": "alpha phrase", "hit_count": 1, "edit_distance": 1},
        {"phrase": "beta phrase", "hit_count": 2, "edit_distance": 2},
    ]

    ranked = rank_context_alternatives(alternatives)

    assert ranked[0]["phrase"] == "beta phrase"


def test_smaller_distance_breaks_equal_hit_tie():
    alternatives = [
        {"phrase": "alpha phrase", "hit_count": 1, "edit_distance": 2},
        {"phrase": "beta phrase", "hit_count": 1, "edit_distance": 1},
    ]

    ranked = rank_context_alternatives(alternatives)

    assert ranked[0]["phrase"] == "beta phrase"


def test_alphabetical_phrase_breaks_remaining_tie():
    alternatives = [
        {"phrase": "beta phrase", "hit_count": 1, "edit_distance": 1},
        {"phrase": "alpha phrase", "hit_count": 1, "edit_distance": 1},
    ]

    ranked = rank_context_alternatives(alternatives)

    assert ranked[0]["phrase"] == "alpha phrase"


@pytest.mark.parametrize("query", ["", "computer"])
def test_context_correction_requires_at_least_two_terms(query):
    documents, inverted_index, positional_index = make_context_collection()

    with pytest.raises(ValueError, match="at least two terms"):
        suggest_context_correction(
            query,
            positional_index,
            inverted_index,
            documents,
        )


def test_phrase_evaluation_reuses_search_phrase(monkeypatch):
    documents, inverted_index, positional_index = make_context_collection()
    original_search_phrase = context_spelling.search_phrase
    calls = []

    def recording_search_phrase(phrase, index, stored_documents):
        calls.append(phrase)
        return original_search_phrase(phrase, index, stored_documents)

    monkeypatch.setattr(
        context_spelling, "search_phrase", recording_search_phrase
    )

    suggest_context_correction(
        "flew form Heathrow",
        positional_index,
        inverted_index,
        documents,
    )

    assert calls[0] == "flew form heathrow"
    assert "flew from heathrow" in calls


def test_raw_document_text_is_not_rescanned_for_phrase_hits():
    documents, inverted_index, positional_index = make_context_collection()

    for document in documents.values():
        document["text"] = ""

    details = suggest_context_correction(
        "flew form Heathrow",
        positional_index,
        inverted_index,
        documents,
    )

    assert details["suggestion"] == "flew from heathrow"


def test_isolated_spelling_behavior_is_unchanged():
    inverted_index = {"computer": [1], "computing": [2]}
    trigram_index = build_trigram_index(inverted_index)

    details = suggest_spelling("computar", trigram_index, inverted_index)

    assert details["suggestion"] == "computer"


def test_weighted_edit_distance_behavior_is_unchanged():
    weights = {("m", "n"): 0.5}

    assert weighted_edit_distance("m", "n", weights) == 0.5


def test_form_and_from_share_no_trigrams():
    assert set(term_trigrams("form")).isdisjoint(term_trigrams("from"))
