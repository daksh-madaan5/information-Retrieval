from edit_distance import weighted_edit_distance
from engine import search_phrase
from preprocessing import tokenize
from spelling import normalize_spelling_term


def get_context_term_candidates(
    term,
    inverted_index,
    max_edit_distance=2,
    replacement_costs=None,
    maximum_candidates=5,
):
    """Return nearby dictionary terms for an explicit context check."""
    if max_edit_distance < 0:
        raise ValueError("Maximum edit distance cannot be negative.")
    if (
        not isinstance(maximum_candidates, int)
        or isinstance(maximum_candidates, bool)
        or maximum_candidates <= 0
    ):
        raise ValueError("Maximum candidates must be a positive integer.")

    normalized_term = normalize_spelling_term(term)
    candidates = []

    for dictionary_term in sorted(inverted_index):
        if dictionary_term == normalized_term:
            continue

        distance = weighted_edit_distance(
            normalized_term,
            dictionary_term,
            replacement_costs=replacement_costs,
        )

        if distance <= max_edit_distance:
            candidates.append((dictionary_term, distance))

    candidates.sort(key=lambda item: (item[1], item[0]))
    return candidates[:maximum_candidates]


def rank_context_alternatives(alternatives):
    """Rank successful phrases by hits, distance, and phrase text."""
    ranked = list(alternatives)
    ranked.sort(
        key=lambda item: (
            -item["hit_count"],
            item["edit_distance"],
            item["phrase"],
        )
    )
    return ranked


def suggest_context_correction(
    query_phrase,
    positional_index,
    inverted_index,
    documents,
    max_edit_distance=2,
    maximum_candidates_per_term=5,
    replacement_costs=None,
):
    """Suggest one-word phrase corrections supported by indexed hits."""
    normalized_tokens = tokenize(query_phrase)
    if len(normalized_tokens) < 2:
        raise ValueError("Context correction requires at least two terms.")

    normalized_phrase = " ".join(normalized_tokens)
    original_matches = search_phrase(
        normalized_phrase, positional_index, documents
    )
    original_matching_doc_ids = [
        document["doc_id"] for document in original_matches
    ]
    details = {
        "original_phrase": normalized_phrase,
        "normalized_tokens": list(normalized_tokens),
        "original_matching_doc_ids": original_matching_doc_ids,
        "alternatives_tested": [],
        "successful_alternatives": [],
        "suggestion": None,
        "reason": None,
    }

    if original_matches:
        details["reason"] = "original phrase already matches"
        return details

    for changed_position, original_term in enumerate(normalized_tokens):
        replacements = get_context_term_candidates(
            original_term,
            inverted_index,
            max_edit_distance=max_edit_distance,
            replacement_costs=replacement_costs,
            maximum_candidates=maximum_candidates_per_term,
        )

        for replacement_term, distance in replacements:
            corrected_tokens = list(normalized_tokens)
            corrected_tokens[changed_position] = replacement_term
            corrected_phrase = " ".join(corrected_tokens)
            matches = search_phrase(
                corrected_phrase, positional_index, documents
            )
            matching_doc_ids = [
                document["doc_id"] for document in matches
            ]
            alternative = {
                "phrase": corrected_phrase,
                "changed_position": changed_position,
                "original_term": original_term,
                "replacement_term": replacement_term,
                "edit_distance": distance,
                "matching_doc_ids": matching_doc_ids,
                "hit_count": len(matching_doc_ids),
            }
            details["alternatives_tested"].append(alternative)

            if alternative["hit_count"] > 0:
                details["successful_alternatives"].append(alternative)

    details["successful_alternatives"] = rank_context_alternatives(
        details["successful_alternatives"]
    )

    if details["successful_alternatives"]:
        details["suggestion"] = details["successful_alternatives"][0][
            "phrase"
        ]
        details["reason"] = "corrected phrase has indexed matches"
    else:
        details["reason"] = "no corrected phrase has indexed matches"

    return details
