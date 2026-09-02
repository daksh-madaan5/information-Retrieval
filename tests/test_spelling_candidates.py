import edit_distance as edit_distance_module
import pytest

from spelling import (
    build_trigram_index,
    get_ngram_spelling_candidates,
    jaccard_coefficient,
    rank_candidates_by_jaccard,
    term_jaccard_similarity,
    term_trigrams,
)


def make_dictionary():
    return {
        "november": [1],
        "december": [2],
        "barrel": [3],
        "machine": [4],
    }


def test_november_trigrams():
    assert term_trigrams("november") == [
        "nov",
        "ove",
        "vem",
        "emb",
        "mbe",
        "ber",
    ]


def test_december_trigrams():
    assert term_trigrams("december") == [
        "dec",
        "ece",
        "cem",
        "emb",
        "mbe",
        "ber",
    ]


def test_trigrams_overlap_by_two_characters():
    assert term_trigrams("abcd") == ["abc", "bcd"]


def test_trigrams_have_no_boundary_markers():
    assert all("$" not in trigram for trigram in term_trigrams("november"))


def test_short_term_fallback_is_deterministic():
    assert term_trigrams("a") == ["a"]
    assert term_trigrams("at") == ["at"]


def test_trigram_index_is_built_from_dictionary_terms():
    trigram_index = build_trigram_index({"november": [1]})

    assert trigram_index["nov"] == ["november"]
    assert "december" not in trigram_index["nov"]


def test_trigram_index_does_not_need_raw_documents():
    inverted_index = {"computer": [1], "computing": [2]}

    trigram_index = build_trigram_index(inverted_index)

    assert trigram_index["com"] == ["computer", "computing"]


def test_trigram_postings_are_unique():
    trigram_index = build_trigram_index({"aaaa": [1]})

    assert trigram_index["aaa"] == ["aaaa"]


def test_trigram_postings_are_sorted():
    trigram_index = build_trigram_index(
        {"computing": [2], "computer": [1]}
    )

    assert trigram_index["com"] == ["computer", "computing"]


def test_candidate_retrieval_uses_union_behavior():
    trigram_index = build_trigram_index(make_dictionary())

    candidates = get_ngram_spelling_candidates("novembar", trigram_index)

    assert candidates == ["barrel", "december", "november"]


def test_unrelated_candidate_is_absent():
    trigram_index = build_trigram_index(make_dictionary())

    candidates = get_ngram_spelling_candidates("novembar", trigram_index)

    assert "machine" not in candidates


def test_duplicate_candidates_are_removed():
    trigram_index = build_trigram_index(make_dictionary())

    candidates = get_ngram_spelling_candidates("novembar", trigram_index)

    assert candidates.count("november") == 1


def test_identical_sets_have_jaccard_one():
    assert jaccard_coefficient({"a", "b"}, {"a", "b"}) == 1.0
    assert jaccard_coefficient(set(), set()) == 1.0


def test_disjoint_sets_have_jaccard_zero():
    assert jaccard_coefficient({"a"}, {"b"}) == 0.0


def test_november_december_jaccard_is_one_third():
    score = term_jaccard_similarity("november", "december")

    assert score == pytest.approx(1 / 3)


@pytest.mark.parametrize(
    "term1, term2",
    [("november", "december"), ("same", "same"), ("cat", "dog")],
)
def test_jaccard_score_is_between_zero_and_one(term1, term2):
    score = term_jaccard_similarity(term1, term2)

    assert 0.0 <= score <= 1.0


def test_term_jaccard_similarity_for_identical_terms():
    assert term_jaccard_similarity("computer", "computer") == 1.0


def test_higher_jaccard_score_sorts_first():
    ranking = rank_candidates_by_jaccard(
        "november", ["december", "november"]
    )

    assert ranking[0] == ("november", 1.0)
    assert ranking[1][0] == "december"


def test_jaccard_ties_sort_alphabetically():
    ranking = rank_candidates_by_jaccard(
        "abcdef", ["defxxx", "abcxxx"]
    )

    assert ranking[0][1] == ranking[1][1]
    assert [term for term, _ in ranking] == ["abcxxx", "defxxx"]


def test_configurable_jaccard_threshold():
    ranking = rank_candidates_by_jaccard(
        "november", ["november", "december"], minimum_score=0.5
    )

    assert ranking == [("november", 1.0)]


def test_spelling_normalization_handles_case_and_punctuation():
    trigram_index = build_trigram_index(make_dictionary())

    candidates = get_ngram_spelling_candidates("NOVEMBAR!", trigram_index)

    assert candidates == ["barrel", "december", "november"]


def test_candidate_ranking_does_not_call_edit_distance(monkeypatch):
    def fail_if_called(source, target):
        raise AssertionError("edit_distance must not be called")

    monkeypatch.setattr(
        edit_distance_module, "edit_distance", fail_if_called
    )

    ranking = rank_candidates_by_jaccard(
        "november", ["november", "december"]
    )

    assert [term for term, _ in ranking] == ["november", "december"]
