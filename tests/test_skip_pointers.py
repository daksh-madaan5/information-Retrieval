from pathlib import Path

from engine import (
    build_inverted_index,
    build_skip_pointers,
    intersect_postings,
    intersect_postings_with_skips,
    load_documents,
    search_and,
    search_and_with_skips,
)


DOCUMENTS_DIRECTORY = Path(__file__).parent.parent / "documents"


def test_skip_pointer_construction_is_deterministic():
    postings = [2, 4, 8, 16, 32, 64, 128]

    assert build_skip_pointers(postings) == {0: 2, 2: 4, 4: 6}
    assert build_skip_pointers(postings) == build_skip_pointers(postings)


def test_short_postings_have_no_unnecessary_skips():
    assert build_skip_pointers([]) == {}
    assert build_skip_pointers([1]) == {}
    assert build_skip_pointers([1, 2, 3]) == {}


def test_skip_targets_always_move_forward():
    skips = build_skip_pointers(list(range(20)))

    assert all(destination > source for source, destination in skips.items())


def test_skip_destinations_remain_within_postings_bounds():
    postings = list(range(20))
    skips = build_skip_pointers(postings)

    assert all(destination < len(postings) for destination in skips.values())


def test_optimized_intersection_equals_normal_intersection():
    postings1 = [2, 4, 8, 16, 32, 64, 128]
    postings2 = [1, 2, 3, 5, 8, 17, 21, 31]
    skips1 = build_skip_pointers(postings1)
    skips2 = build_skip_pointers(postings2)

    normal_result = intersect_postings(postings1, postings2)
    skip_result = intersect_postings_with_skips(
        postings1, postings2, skips1, skips2
    )

    assert skip_result == normal_result


def test_skip_intersection_with_overlapping_lists():
    postings1 = [1, 2, 4, 7, 10, 13]
    postings2 = [2, 3, 4, 8, 10, 14]

    result = intersect_postings_with_skips(
        postings1,
        postings2,
        build_skip_pointers(postings1),
        build_skip_pointers(postings2),
    )

    assert result == [2, 4, 10]


def test_skip_intersection_with_no_overlap():
    postings1 = [1, 3, 5, 7, 9]
    postings2 = [2, 4, 6, 8, 10]

    result = intersect_postings_with_skips(
        postings1,
        postings2,
        build_skip_pointers(postings1),
        build_skip_pointers(postings2),
    )

    assert result == []


def test_skip_intersection_with_empty_lists():
    assert intersect_postings_with_skips([], [], {}, {}) == []
    assert intersect_postings_with_skips([1, 2], [], {}, {}) == []
    assert intersect_postings_with_skips([], [1, 2], {}, {}) == []


def test_synthetic_intersection_uses_at_least_one_skip():
    postings1 = [2, 4, 8, 16, 32, 64, 128]
    postings2 = [31, 32, 100]
    statistics = {}

    result = intersect_postings_with_skips(
        postings1,
        postings2,
        build_skip_pointers(postings1),
        build_skip_pointers(postings2),
        statistics,
    )

    assert result == [32]
    assert statistics["skip_jumps"] >= 1


def test_skip_intersection_result_remains_sorted():
    postings1 = [1, 2, 4, 7, 10, 13]
    postings2 = [2, 3, 4, 8, 10, 14]

    result = intersect_postings_with_skips(
        postings1,
        postings2,
        build_skip_pointers(postings1),
        build_skip_pointers(postings2),
    )

    assert result == sorted(result)


def test_skip_search_returns_same_documents_as_normal_and():
    documents = load_documents(DOCUMENTS_DIRECTORY)
    inverted_index, _ = build_inverted_index(documents)

    normal_matches = search_and("computer", "search", inverted_index, documents)
    skip_matches = search_and_with_skips(
        "computer", "search", inverted_index, documents
    )

    assert [document["doc_id"] for document in skip_matches] == [
        document["doc_id"] for document in normal_matches
    ]


def test_skip_search_uses_index_without_rescanning_text():
    documents = load_documents(DOCUMENTS_DIRECTORY)
    inverted_index, _ = build_inverted_index(documents)

    for document in documents.values():
        document["text"] = ""

    matches = search_and_with_skips(
        "computer", "search", inverted_index, documents
    )

    assert [document["doc_id"] for document in matches] == [1]
