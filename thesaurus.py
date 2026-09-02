from nltk.corpus import wordnet

from engine import union_postings
from preprocessing import tokenize
from spelling import normalize_spelling_term


WORDNET_SETUP_MESSAGE = (
    "NLTK WordNet data is not installed.\n"
    "Run: python -m nltk.downloader wordnet"
)


def get_wordnet_synonyms(term, max_synonyms=5):
    """Return sorted, single-token WordNet synonyms for one term."""
    if (
        not isinstance(max_synonyms, int)
        or isinstance(max_synonyms, bool)
        or max_synonyms < 0
    ):
        raise ValueError("Maximum synonyms must be a non-negative integer.")

    normalized_term = normalize_spelling_term(term)

    try:
        synsets = wordnet.synsets(normalized_term)
    except LookupError:
        raise LookupError(WORDNET_SETUP_MESSAGE) from None

    synonyms = set()

    for synset in synsets:
        for lemma_name in synset.lemma_names():
            lowercase_name = lemma_name.casefold()
            lemma_tokens = tokenize(lowercase_name)

            if len(lemma_tokens) != 1:
                continue

            synonym = lemma_tokens[0]
            if synonym != normalized_term:
                synonyms.add(synonym)

    sorted_synonyms = sorted(synonyms)
    return sorted_synonyms[:max_synonyms]


def expand_query_term(
    term,
    inverted_index,
    max_synonyms=5,
    details=None,
):
    """Expand one query with indexed, single-token WordNet synonyms."""
    normalized_term = normalize_spelling_term(term)
    wordnet_synonyms = get_wordnet_synonyms(
        normalized_term, max_synonyms=max_synonyms
    )

    indexed_expansion_terms = [normalized_term]
    for synonym in wordnet_synonyms:
        if (
            synonym in inverted_index
            and synonym not in indexed_expansion_terms
        ):
            indexed_expansion_terms.append(synonym)

    if details is not None:
        details["query"] = term
        details["normalized_query"] = normalized_term
        details["wordnet_synonyms"] = list(wordnet_synonyms)
        details["indexed_expansion_terms"] = list(
            indexed_expansion_terms
        )

    return indexed_expansion_terms


def search_with_thesaurus(
    query_term,
    inverted_index,
    documents,
    max_synonyms=5,
    details=None,
):
    """Search the union of normal postings for an expanded query."""
    expansion_details = details if details is not None else {}
    expansion_terms = expand_query_term(
        query_term,
        inverted_index,
        max_synonyms=max_synonyms,
        details=expansion_details,
    )

    matching_doc_ids = []
    for term in expansion_terms:
        postings = inverted_index.get(term, [])
        matching_doc_ids = union_postings(matching_doc_ids, postings)

    if details is not None:
        details["matching_doc_ids"] = list(matching_doc_ids)

    matching_documents = []
    for doc_id in matching_doc_ids:
        document = documents[doc_id].copy()
        document["doc_id"] = doc_id
        matching_documents.append(document)

    return matching_documents
