from edit_distance import edit_distance
from preprocessing import tokenize


def normalize_spelling_term(term):
    """Apply project normalization to exactly one spelling term."""
    tokens = tokenize(term)
    if len(tokens) != 1:
        raise ValueError("Spelling comparison requires exactly one term.")

    return tokens[0]


def term_trigrams(term):
    """Return overlapping trigrams without word-boundary markers."""
    normalized_term = normalize_spelling_term(term)

    if len(normalized_term) < 3:
        return [normalized_term]

    trigrams = []
    for index in range(len(normalized_term) - 2):
        trigrams.append(normalized_term[index:index + 3])

    return trigrams


def build_trigram_index(inverted_index):
    """Build trigram-to-term postings from normal dictionary terms."""
    trigram_index = {}

    for term in sorted(inverted_index):
        trigrams = term_trigrams(term)

        for trigram in trigrams:
            if trigram not in trigram_index:
                trigram_index[trigram] = []
            if term not in trigram_index[trigram]:
                trigram_index[trigram].append(term)

    for terms in trigram_index.values():
        terms.sort()

    return trigram_index


def get_ngram_spelling_candidates(query_term, trigram_index):
    """Return the union of terms sharing at least one query trigram."""
    query_trigrams = term_trigrams(query_term)
    candidate_terms = set()

    for trigram in query_trigrams:
        if trigram in trigram_index:
            for term in trigram_index[trigram]:
                candidate_terms.add(term)

    return sorted(candidate_terms)


def jaccard_coefficient(set1, set2):
    """Return intersection size divided by union size for two sets."""
    if not set1 and not set2:
        return 1.0

    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union)


def term_jaccard_similarity(term1, term2):
    """Compare two normalized terms using their trigram sets."""
    trigrams1 = set(term_trigrams(term1))
    trigrams2 = set(term_trigrams(term2))
    return jaccard_coefficient(trigrams1, trigrams2)


def rank_candidates_by_jaccard(
    query_term, candidate_terms, minimum_score=0.0
):
    """Rank candidates by descending Jaccard score and then alphabetically."""
    if (
        not isinstance(minimum_score, (int, float))
        or isinstance(minimum_score, bool)
        or minimum_score < 0
        or minimum_score > 1
    ):
        raise ValueError("Jaccard threshold must be between 0 and 1.")

    normalized_query = normalize_spelling_term(query_term)
    scores_by_term = {}

    for candidate in candidate_terms:
        normalized_candidate = normalize_spelling_term(candidate)
        score = term_jaccard_similarity(
            normalized_query, normalized_candidate
        )

        if score >= minimum_score:
            scores_by_term[normalized_candidate] = score

    ranked_candidates = list(scores_by_term.items())
    ranked_candidates.sort(key=lambda item: (-item[1], item[0]))
    return ranked_candidates


def rank_spelling_suggestions(
    query_term,
    trigram_index,
    inverted_index,
    minimum_jaccard=0.0,
):
    """Rank trigram candidates by edit distance, Jaccard, and term."""
    normalized_query = normalize_spelling_term(query_term)
    retrieved_candidates = get_ngram_spelling_candidates(
        normalized_query, trigram_index
    )
    dictionary_candidates = []

    for term in retrieved_candidates:
        if term in inverted_index:
            dictionary_candidates.append(term)

    jaccard_ranking = rank_candidates_by_jaccard(
        normalized_query,
        dictionary_candidates,
        minimum_score=minimum_jaccard,
    )
    ranked_suggestions = []

    for term, jaccard_score in jaccard_ranking:
        distance = edit_distance(normalized_query, term)
        ranked_suggestions.append((term, distance, jaccard_score))

    ranked_suggestions.sort(
        key=lambda item: (item[1], -item[2], item[0])
    )
    return ranked_suggestions


def suggest_spelling(
    query_term,
    trigram_index,
    inverted_index,
    max_edit_distance=2,
    minimum_jaccard=0.0,
):
    """Return isolated-word spelling details and an optional suggestion."""
    if (
        not isinstance(max_edit_distance, int)
        or isinstance(max_edit_distance, bool)
        or max_edit_distance < 0
    ):
        raise ValueError("Maximum edit distance must be a non-negative integer.")

    normalized_query = normalize_spelling_term(query_term)
    details = {
        "query": query_term,
        "normalized_query": normalized_query,
        "query_trigrams": term_trigrams(normalized_query),
        "candidate_terms": [],
        "ranked_suggestions": [],
        "already_correct": normalized_query in inverted_index,
        "suggestion": None,
    }

    if details["already_correct"]:
        return details

    candidate_terms = get_ngram_spelling_candidates(
        normalized_query, trigram_index
    )
    details["candidate_terms"] = [
        term for term in candidate_terms if term in inverted_index
    ]

    ranked_suggestions = rank_spelling_suggestions(
        normalized_query,
        trigram_index,
        inverted_index,
        minimum_jaccard=minimum_jaccard,
    )

    for term, distance, jaccard_score in ranked_suggestions:
        details["ranked_suggestions"].append(
            {
                "term": term,
                "edit_distance": distance,
                "jaccard": jaccard_score,
            }
        )

    if (
        ranked_suggestions
        and ranked_suggestions[0][1] <= max_edit_distance
    ):
        details["suggestion"] = ranked_suggestions[0][0]

    return details
