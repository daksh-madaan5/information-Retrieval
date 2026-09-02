from pathlib import Path

import spelling
from engine import build_inverted_index, load_documents
from spelling import (
    build_trigram_index,
    rank_spelling_suggestions,
    suggest_spelling,
)


DOCUMENTS_DIRECTORY = Path(__file__).parent.parent / "documents"


def real_dictionary():
    documents = load_documents(DOCUMENTS_DIRECTORY)
    inverted_index, _ = build_inverted_index(documents)
    trigram_index = build_trigram_index(inverted_index)
    return inverted_index, trigram_index


def test_suggestion_candidates_come_from_trigram_retrieval():
    inverted_index, trigram_index = real_dictionary()

    ranking = rank_spelling_suggestions(
        "computar", trigram_index, inverted_index
    )

    assert [term for term, _, _ in ranking] == [
        "computer",
        "computing",
        "computational",
    ]


def test_edit_distance_runs_only_for_retrieved_candidates(monkeypatch):
    inverted_index = {
        "computer": [1],
        "computing": [2],
        "zebra": [3],
    }
    trigram_index = build_trigram_index(inverted_index)
    calls = []
    original_edit_distance = spelling.edit_distance

    def recording_edit_distance(source, target):
        calls.append(target)
        return original_edit_distance(source, target)

    monkeypatch.setattr(spelling, "edit_distance", recording_edit_distance)

    rank_spelling_suggestions("computar", trigram_index, inverted_index)

    assert sorted(calls) == ["computer", "computing"]
    assert "zebra" not in calls


def test_smaller_edit_distance_ranks_first():
    inverted_index, trigram_index = real_dictionary()

    ranking = rank_spelling_suggestions(
        "computar", trigram_index, inverted_index
    )

    assert ranking[0][0] == "computer"
    assert ranking[0][1] == 1


def test_higher_jaccard_breaks_equal_distance_tie(monkeypatch):
    inverted_index = {"november": [1], "december": [2]}
    trigram_index = build_trigram_index(inverted_index)

    monkeypatch.setattr(spelling, "edit_distance", lambda source, target: 2)

    ranking = rank_spelling_suggestions(
        "novembar", trigram_index, inverted_index
    )

    assert [term for term, _, _ in ranking] == ["november", "december"]
    assert ranking[0][2] > ranking[1][2]


def test_alphabetical_order_breaks_remaining_tie(monkeypatch):
    inverted_index = {"abcxxx": [1], "defxxx": [2]}
    trigram_index = build_trigram_index(inverted_index)

    monkeypatch.setattr(spelling, "edit_distance", lambda source, target: 2)

    ranking = rank_spelling_suggestions(
        "abcdef", trigram_index, inverted_index
    )

    assert ranking[0][2] == ranking[1][2]
    assert [term for term, _, _ in ranking] == ["abcxxx", "defxxx"]


def test_spelling_ranking_is_deterministic():
    inverted_index, trigram_index = real_dictionary()

    first = rank_spelling_suggestions(
        "computar", trigram_index, inverted_index
    )
    second = rank_spelling_suggestions(
        "computar", trigram_index, inverted_index
    )

    assert first == second


def test_computar_suggests_computer():
    inverted_index, trigram_index = real_dictionary()

    details = suggest_spelling(
        "computar", trigram_index, inverted_index
    )

    assert details["suggestion"] == "computer"
    assert details["already_correct"] is False


def test_exact_dictionary_term_is_already_correct(monkeypatch):
    inverted_index, trigram_index = real_dictionary()

    def fail_if_called(source, target):
        raise AssertionError("Correction work should not run")

    monkeypatch.setattr(spelling, "edit_distance", fail_if_called)

    details = suggest_spelling(
        "computer", trigram_index, inverted_index
    )

    assert details["already_correct"] is True
    assert details["suggestion"] is None
    assert details["ranked_suggestions"] == []


def test_no_trigram_candidates_gives_no_suggestion(monkeypatch):
    inverted_index = {"computer": [1]}
    trigram_index = build_trigram_index(inverted_index)

    def fail_if_called(source, target):
        raise AssertionError("No candidates means no edit distance")

    monkeypatch.setattr(spelling, "edit_distance", fail_if_called)

    details = suggest_spelling("zzzxq", trigram_index, inverted_index)

    assert details["candidate_terms"] == []
    assert details["suggestion"] is None


def test_candidate_beyond_maximum_distance_is_rejected():
    inverted_index = {"computer": [1]}
    trigram_index = build_trigram_index(inverted_index)

    details = suggest_spelling(
        "computxx",
        trigram_index,
        inverted_index,
        max_edit_distance=1,
    )

    assert details["ranked_suggestions"][0]["edit_distance"] == 2
    assert details["suggestion"] is None


def test_candidate_at_maximum_distance_is_accepted():
    inverted_index = {"computer": [1]}
    trigram_index = build_trigram_index(inverted_index)

    details = suggest_spelling(
        "computxx",
        trigram_index,
        inverted_index,
        max_edit_distance=2,
    )

    assert details["suggestion"] == "computer"


def test_configurable_minimum_jaccard_filters_before_edit_distance(monkeypatch):
    inverted_index = {"november": [1], "december": [2]}
    trigram_index = build_trigram_index(inverted_index)
    calls = []
    original_edit_distance = spelling.edit_distance

    def recording_edit_distance(source, target):
        calls.append(target)
        return original_edit_distance(source, target)

    monkeypatch.setattr(spelling, "edit_distance", recording_edit_distance)

    ranking = rank_spelling_suggestions(
        "novembar",
        trigram_index,
        inverted_index,
        minimum_jaccard=0.2,
    )

    assert [term for term, _, _ in ranking] == ["november"]
    assert calls == ["november"]


def test_uppercase_query_is_normalized():
    inverted_index, trigram_index = real_dictionary()

    details = suggest_spelling(
        "COMPUTAR", trigram_index, inverted_index
    )

    assert details["normalized_query"] == "computar"
    assert details["suggestion"] == "computer"


def test_query_punctuation_is_normalized():
    inverted_index, trigram_index = real_dictionary()

    details = suggest_spelling(
        "COMPUTAR!", trigram_index, inverted_index
    )

    assert details["normalized_query"] == "computar"
    assert details["suggestion"] == "computer"


def test_suggestion_details_are_complete():
    inverted_index, trigram_index = real_dictionary()

    details = suggest_spelling(
        "computar", trigram_index, inverted_index
    )

    assert details["normalized_query"] == "computar"
    assert details["candidate_terms"] == [
        "computational",
        "computer",
        "computing",
    ]
    assert details["suggestion"] == "computer"
    assert details["ranked_suggestions"]

    first = details["ranked_suggestions"][0]
    assert first["term"] == "computer"
    assert first["edit_distance"] == 1
    assert first["jaccard"] == 0.5


def test_no_full_dictionary_edit_distance_scan(monkeypatch):
    inverted_index = {
        "computer": [1],
        "computing": [2],
        "zebra": [3],
        "colour": [4],
        "machine": [5],
    }
    trigram_index = build_trigram_index(inverted_index)
    calls = []

    def recording_edit_distance(source, target):
        calls.append(target)
        return 1

    monkeypatch.setattr(spelling, "edit_distance", recording_edit_distance)

    rank_spelling_suggestions("computar", trigram_index, inverted_index)

    assert set(calls) == {"computer", "computing"}
    assert len(calls) < len(inverted_index)
