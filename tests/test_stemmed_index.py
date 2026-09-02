from pathlib import Path

from engine import (
    build_inverted_index,
    build_stemmed_index,
    load_documents,
    search,
    search_stemmed,
    stem_query_term,
)


DOCUMENTS_DIRECTORY = Path(__file__).parent.parent / "documents"


def load_both_indexes():
    documents = load_documents(DOCUMENTS_DIRECTORY)
    inverted_index, document_frequency = build_inverted_index(documents)
    stemmed_index = build_stemmed_index(documents)
    return documents, inverted_index, document_frequency, stemmed_index


def test_stemmed_index_is_created_correctly():
    _, _, _, stemmed_index = load_both_indexes()

    assert stemmed_index["comput"] == [1, 2]
    assert stemmed_index["autom"] == [4]


def test_related_forms_map_to_the_same_stem():
    assert stem_query_term("computer") == "comput"
    assert stem_query_term("computing") == "comput"
    assert stem_query_term("computational") == "comput"


def test_stemmed_postings_contain_unique_document_ids():
    _, _, _, stemmed_index = load_both_indexes()

    assert stemmed_index["comput"].count(2) == 1
    assert all(
        len(postings) == len(set(postings))
        for postings in stemmed_index.values()
    )


def test_stemmed_postings_are_sorted():
    _, _, _, stemmed_index = load_both_indexes()

    assert all(
        postings == sorted(postings) for postings in stemmed_index.values()
    )


def test_query_term_is_stemmed_like_indexed_terms():
    documents, _, _, stemmed_index = load_both_indexes()

    matches = search_stemmed("COMPUTING!", stemmed_index, documents)

    assert stem_query_term("COMPUTING!") == "comput"
    assert [document["doc_id"] for document in matches] == [1, 2]


def test_stemmed_search_finds_related_word_forms():
    documents, _, _, stemmed_index = load_both_indexes()

    matches = search_stemmed("computational", stemmed_index, documents)

    assert [document["doc_id"] for document in matches] == [1, 2]
    assert [document["filename"] for document in matches] == [
        "doc1.txt",
        "doc2.txt",
    ]


def test_normal_search_behavior_is_unchanged():
    _, inverted_index, _, _ = load_both_indexes()

    assert search("computing", inverted_index) == [2]
    assert search("computer", inverted_index) == [1, 2]
    assert search("computational", inverted_index) == [2]


def test_stemmed_index_does_not_alter_normal_index():
    documents = load_documents(DOCUMENTS_DIRECTORY)
    inverted_index, _ = build_inverted_index(documents)
    original_normal_index = {
        term: list(postings) for term, postings in inverted_index.items()
    }

    build_stemmed_index(documents)

    assert inverted_index == original_normal_index
    assert "comput" not in inverted_index


def test_unknown_stem_returns_no_documents():
    documents, _, _, stemmed_index = load_both_indexes()

    assert search_stemmed("nonexistentterm", stemmed_index, documents) == []


def test_stemmed_search_does_not_rescan_document_text():
    documents, _, _, stemmed_index = load_both_indexes()

    for document in documents.values():
        document["text"] = ""

    matches = search_stemmed("computing", stemmed_index, documents)

    assert [document["doc_id"] for document in matches] == [1, 2]


def test_stop_words_remain_in_both_indexes():
    _, inverted_index, _, stemmed_index = load_both_indexes()

    assert inverted_index["the"] == [5]
    assert stemmed_index["the"] == [5]
    assert "and" in inverted_index
    assert "and" in stemmed_index
