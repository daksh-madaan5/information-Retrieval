from pathlib import Path

import pytest

import app as web_app
from app import (
    MIXED_SYNTAX_MESSAGE,
    build_search_data,
    detect_query_type,
    execute_auto_search,
)


DOCUMENTS_DIRECTORY = Path(__file__).parents[1] / "documents"


@pytest.fixture(scope="module")
def search_data():
    return build_search_data(DOCUMENTS_DIRECTORY)


@pytest.mark.parametrize(
    ("query", "expected_type"),
    [
        ("computer", "exact"),
        ("machine learning", "multi_term"),
        ("comput*", "wildcard"),
        ('"machine learning"', "phrase"),
        ("machine /4 computer", "proximity"),
        ("computer AND machine", "boolean"),
        ("computer or search", "boolean"),
        ("(computer)", "boolean"),
    ],
)
def test_automatic_query_detection(query, expected_type):
    assert detect_query_type(query) == expected_type


def test_detection_precedence_prefers_complete_proximity():
    assert detect_query_type("information /3 retrieval") == "proximity"


def test_fully_quoted_boolean_word_remains_a_phrase():
    assert detect_query_type('"machine AND learning"') == "phrase"


def test_boolean_word_prefix_with_wildcard_remains_a_wildcard():
    assert detect_query_type("and*") == "wildcard"


@pytest.mark.parametrize(
    "query",
    [
        "comput* AND machine",
        '"machine learning" AND computer',
        "comput* /4 computer",
    ],
)
def test_mixed_special_syntax_is_rejected(query):
    with pytest.raises(ValueError, match="Mixed special query syntax"):
        detect_query_type(query)


def test_malformed_proximity_has_useful_error():
    with pytest.raises(ValueError, match="machine /4 computer"):
        detect_query_type("machine / computer")


def test_malformed_phrase_has_useful_error():
    with pytest.raises(ValueError, match="enclosed"):
        detect_query_type('"machine learning')


def test_advanced_mode_overrides_automatic_detection(search_data):
    result = execute_auto_search(
        "computing", search_data, advanced_mode="stemmed"
    )

    assert result["detected_type"] == "stemmed"
    assert result["detected_label"] == "Stemmed search"


def test_default_multi_term_search_uses_and(search_data):
    result = execute_auto_search("machine learning", search_data)

    assert result["detected_type"] == "multi_term"
    assert [document["doc_id"] for document in result["documents"]] == [2]


def test_phrase_quotes_are_removed_before_backend_search(search_data):
    result = execute_auto_search('"machine learning"', search_data)

    assert result["detected_type"] == "phrase"
    assert [document["doc_id"] for document in result["documents"]] == [2]


def test_spelling_override_does_not_search_suggestion(search_data):
    result = execute_auto_search(
        "computar", search_data, advanced_mode="spelling"
    )

    assert result["suggestion"] == "computer"
    assert result["documents"] == []


def test_get_route_renders_single_search_page():
    client = web_app.app.test_client()

    response = client.get("/")

    assert response.status_code == 200
    assert b"Information Retrieval" in response.data
    assert b"Search across 10 indexed documents" in response.data
    assert response.data.count(b'name="query"') == 1


def test_post_route_renders_detected_type_and_results():
    client = web_app.app.test_client()

    response = client.post(
        "/", data={"query": "comput*", "advanced_mode": "auto"}
    )

    assert response.status_code == 200
    assert b"Detected: Wildcard search" in response.data
    assert b"doc1.txt" in response.data
    assert b"doc2.txt" in response.data


def test_post_route_renders_inline_validation_error():
    client = web_app.app.test_client()

    response = client.post(
        "/",
        data={
            "query": "comput* AND machine",
            "advanced_mode": "auto",
        },
    )

    assert response.status_code == 200
    assert b"Unable to search" in response.data
    assert MIXED_SYNTAX_MESSAGE.encode() in response.data


def test_wordnet_is_not_used_during_search_data_build(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("WordNet must only run in thesaurus mode")

    monkeypatch.setattr(web_app, "search_with_thesaurus", fail_if_called)

    data = build_search_data(DOCUMENTS_DIRECTORY)

    assert len(data["documents"]) == 10
