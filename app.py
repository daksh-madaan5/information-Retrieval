import re
from pathlib import Path

from flask import Flask, render_template, request

from context_spelling import suggest_context_correction
from engine import (
    build_bigram_index,
    build_inverted_index,
    build_positional_index,
    build_stemmed_index,
    load_documents,
    search,
    search_and_multiple,
    search_boolean,
    search_phrase,
    search_proximity,
    search_stemmed,
    search_wildcard,
    stem_query_term,
)
from preprocessing import tokenize, tokenize_boolean_query
from soundex import build_soundex_index, search_soundex
from spelling import build_trigram_index, suggest_spelling
from thesaurus import search_with_thesaurus


app = Flask(__name__)

ADVANCED_MODES = (
    "auto",
    "stemmed",
    "spelling",
    "context",
    "soundex",
    "thesaurus",
)

PROXIMITY_MESSAGE = (
    "Use proximity format: term1 /k term2. "
    "Example: machine /4 computer"
)

MIXED_SYNTAX_MESSAGE = (
    "Mixed special query syntax is not supported. "
    "Use one wildcard, proximity, phrase, or Boolean query at a time."
)


def build_search_data(documents_directory=None):
    """Load documents and build the application's indexes once."""
    if documents_directory is None:
        documents_directory = Path(__file__).parent / "documents"

    documents = load_documents(documents_directory)
    inverted_index, document_frequency = build_inverted_index(documents)

    return {
        "documents": documents,
        "inverted_index": inverted_index,
        "document_frequency": document_frequency,
        "stemmed_index": build_stemmed_index(documents),
        "positional_index": build_positional_index(documents),
        "bigram_index": build_bigram_index(inverted_index),
        "trigram_index": build_trigram_index(inverted_index),
        "soundex_index": build_soundex_index(inverted_index),
    }


def parse_proximity_query(query):
    """Parse term1 /k term2 and return both terms and k."""
    match = re.fullmatch(r"\s*(\S+)\s+/(\d+)\s+(\S+)\s*", query)
    if match is None:
        raise ValueError(PROXIMITY_MESSAGE)

    distance = int(match.group(2))
    if distance <= 0:
        raise ValueError(PROXIMITY_MESSAGE)

    return match.group(1), distance, match.group(3)


def detect_query_type(query):
    """Classify one query using deterministic syntax precedence."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Enter a search query.")

    stripped_query = query.strip()
    proximity_match = re.fullmatch(
        r"\s*(\S+)\s+/(\d+)\s+(\S+)\s*", stripped_query
    )
    has_proximity_marker = re.search(r"/\s*\d*", stripped_query) is not None
    has_wildcard = "*" in stripped_query
    has_quote = '"' in stripped_query
    fully_quoted = (
        len(stripped_query) >= 2
        and stripped_query.startswith('"')
        and stripped_query.endswith('"')
        and stripped_query.count('"') == 2
    )
    has_boolean_operator = (
        re.search(
            r"(?<![A-Za-z0-9*])(?:AND|OR|NOT)(?![A-Za-z0-9*])",
            stripped_query,
            re.IGNORECASE,
        )
        is not None
    )
    has_parenthesis = "(" in stripped_query or ")" in stripped_query

    if proximity_match is not None:
        if has_wildcard or has_quote or has_boolean_operator or has_parenthesis:
            raise ValueError(MIXED_SYNTAX_MESSAGE)
        parse_proximity_query(stripped_query)
        return "proximity"

    if has_proximity_marker:
        raise ValueError(PROXIMITY_MESSAGE)

    if has_wildcard:
        if has_quote or has_boolean_operator or has_parenthesis:
            raise ValueError(MIXED_SYNTAX_MESSAGE)
        return "wildcard"

    if has_quote:
        if fully_quoted:
            return "phrase"
        if has_boolean_operator or has_parenthesis:
            raise ValueError(MIXED_SYNTAX_MESSAGE)
        raise ValueError(
            "Phrase queries should be enclosed in one pair of "
            'quotation marks, for example: "machine learning"'
        )

    if has_boolean_operator or has_parenthesis:
        return "boolean"

    query_tokens = tokenize(stripped_query)
    if len(query_tokens) == 1:
        return "exact"
    if len(query_tokens) > 1:
        return "multi_term"

    raise ValueError("Enter a search query containing normal terms.")


def documents_from_ids(doc_ids, documents):
    """Map sorted IDs to document metadata for template rendering."""
    matches = []
    for doc_id in doc_ids:
        document = documents[doc_id].copy()
        document["doc_id"] = doc_id
        matches.append(document)
    return matches


def detail(label, value):
    """Create one simple query-detail row."""
    return {"label": label, "value": str(value)}


def basic_result(detected_type, detected_label, documents, details):
    """Create the common search result dictionary."""
    return {
        "detected_type": detected_type,
        "detected_label": detected_label,
        "documents": documents,
        "result_count": len(documents),
        "details": details,
        "special_type": None,
    }


def execute_auto_search(query, search_data, advanced_mode="auto"):
    """Dispatch an automatic or explicitly selected advanced search."""
    if advanced_mode not in ADVANCED_MODES:
        raise ValueError("Choose a valid advanced search mode.")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Enter a search query.")

    documents = search_data["documents"]
    inverted_index = search_data["inverted_index"]

    if advanced_mode == "stemmed":
        stem = stem_query_term(query)
        matches = search_stemmed(
            query, search_data["stemmed_index"], documents
        )
        return basic_result(
            "stemmed",
            "Stemmed search",
            matches,
            [
                detail("Query stem", stem),
                detail(
                    "Stemmed postings",
                    search_data["stemmed_index"].get(stem, []),
                ),
            ],
        )

    if advanced_mode == "spelling":
        spelling_details = suggest_spelling(
            query, search_data["trigram_index"], inverted_index
        )
        result = basic_result(
            "spelling",
            "Spelling correction",
            [],
            [
                detail("Query trigrams", spelling_details["query_trigrams"]),
                detail("Candidate terms", spelling_details["candidate_terms"]),
                detail(
                    "Ranked candidates",
                    spelling_details["ranked_suggestions"],
                ),
            ],
        )
        result["special_type"] = "spelling"
        result["normalized_query"] = spelling_details["normalized_query"]
        result["already_correct"] = spelling_details["already_correct"]
        result["suggestion"] = spelling_details["suggestion"]
        result["edit_distance"] = None
        if spelling_details["ranked_suggestions"]:
            result["edit_distance"] = spelling_details[
                "ranked_suggestions"
            ][0]["edit_distance"]
        return result

    if advanced_mode == "context":
        context_details = suggest_context_correction(
            query,
            search_data["positional_index"],
            inverted_index,
            documents,
        )
        result = basic_result(
            "context",
            "Context spelling",
            [],
            [
                detail(
                    "Alternatives tested",
                    context_details["alternatives_tested"],
                ),
                detail(
                    "Successful alternatives",
                    context_details["successful_alternatives"],
                ),
            ],
        )
        result["special_type"] = "context"
        result["original_phrase"] = context_details["original_phrase"]
        result["original_matching_doc_ids"] = context_details[
            "original_matching_doc_ids"
        ]
        result["suggestion"] = context_details["suggestion"]
        return result

    if advanced_mode == "soundex":
        soundex_details = {}
        matches = search_soundex(
            query,
            search_data["soundex_index"],
            inverted_index,
            documents,
            soundex_details,
        )
        result = basic_result(
            "soundex",
            "Soundex search",
            matches,
            [
                detail("Soundex code", soundex_details["soundex_code"]),
                detail("Matching terms", soundex_details["matching_terms"]),
                detail(
                    "Matching document IDs",
                    soundex_details["matching_doc_ids"],
                ),
            ],
        )
        result["special_type"] = "soundex"
        result["soundex_code"] = soundex_details["soundex_code"]
        result["matching_terms"] = soundex_details["matching_terms"]
        return result

    if advanced_mode == "thesaurus":
        thesaurus_details = {}
        matches = search_with_thesaurus(
            query,
            inverted_index,
            documents,
            details=thesaurus_details,
        )
        result = basic_result(
            "thesaurus",
            "Thesaurus expansion",
            matches,
            [
                detail(
                    "WordNet synonyms",
                    thesaurus_details["wordnet_synonyms"],
                ),
                detail(
                    "Indexed expansion terms",
                    thesaurus_details["indexed_expansion_terms"],
                ),
                detail(
                    "Matching document IDs",
                    thesaurus_details["matching_doc_ids"],
                ),
            ],
        )
        result["special_type"] = "thesaurus"
        result["expanded_terms"] = thesaurus_details[
            "indexed_expansion_terms"
        ]
        return result

    query_type = detect_query_type(query)

    if query_type == "proximity":
        term1, distance, term2 = parse_proximity_query(query)
        matches = search_proximity(
            term1,
            term2,
            distance,
            search_data["positional_index"],
            documents,
        )
        return basic_result(
            query_type,
            "Proximity search",
            matches,
            [
                detail("Term 1", tokenize(term1)[0]),
                detail("Term 2", tokenize(term2)[0]),
                detail("Maximum distance", distance),
            ],
        )

    if query_type == "wildcard":
        wildcard_details = {}
        matches = search_wildcard(
            query,
            search_data["bigram_index"],
            inverted_index,
            documents,
            wildcard_details,
        )
        return basic_result(
            query_type,
            "Wildcard search",
            matches,
            [
                detail("Required bigrams", wildcard_details["required_bigrams"]),
                detail("Candidate terms", wildcard_details["candidate_terms"]),
                detail("Matching terms", wildcard_details["matching_terms"]),
            ],
        )

    if query_type == "phrase":
        phrase = query.strip()[1:-1]
        matches = search_phrase(
            phrase, search_data["positional_index"], documents
        )
        return basic_result(
            query_type,
            "Phrase search",
            matches,
            [detail("Normalized phrase", " ".join(tokenize(phrase)))],
        )

    if query_type == "boolean":
        matches = search_boolean(query, inverted_index, documents)
        return basic_result(
            query_type,
            "Boolean search",
            matches,
            [
                detail("Parsed tokens", tokenize_boolean_query(query)),
                detail(
                    "Matching document IDs",
                    [document["doc_id"] for document in matches],
                ),
            ],
        )

    if query_type == "multi_term":
        terms = tokenize(query)
        matches = search_and_multiple(
            terms,
            inverted_index,
            search_data["document_frequency"],
            documents,
        )
        return basic_result(
            query_type,
            "Multi-term AND search",
            matches,
            [
                detail("Normalized terms", terms),
                detail(
                    "Matching document IDs",
                    [document["doc_id"] for document in matches],
                ),
            ],
        )

    doc_ids = search(query, inverted_index)
    matches = documents_from_ids(doc_ids, documents)
    normalized_term = tokenize(query)[0]
    return basic_result(
        "exact",
        "Exact search",
        matches,
        [
            detail("Normalized term", normalized_term),
            detail("Postings", doc_ids),
        ],
    )


SEARCH_DATA = build_search_data()


@app.route("/", methods=["GET", "POST"])
def index():
    """Render the initial page or process one submitted search."""
    query = ""
    advanced_mode = "auto"
    result = None
    error = None

    if request.method == "POST":
        query = request.form.get("query", "")
        advanced_mode = request.form.get("advanced_mode", "auto")

        try:
            result = execute_auto_search(
                query, SEARCH_DATA, advanced_mode=advanced_mode
            )
        except (ValueError, LookupError) as exception:
            error = str(exception)

    return render_template(
        "index.html",
        document_count=len(SEARCH_DATA["documents"]),
        query=query,
        advanced_mode=advanced_mode,
        advanced_modes=ADVANCED_MODES,
        result=result,
        error=error,
        searched=request.method == "POST",
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
