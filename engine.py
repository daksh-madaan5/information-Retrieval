import math
from pathlib import Path

from preprocessing import stem_tokens, tokenize, tokenize_boolean_query


def load_documents(documents_directory):
    """Load sorted .txt files and assign document IDs starting at 1."""
    documents_path = Path(documents_directory)
    text_files = sorted(documents_path.glob("*.txt"), key=lambda path: path.name)

    documents = {}
    for doc_id, file_path in enumerate(text_files, start=1):
        documents[doc_id] = {
            "filename": file_path.name,
            "text": file_path.read_text(encoding="utf-8"),
        }

    return documents


def build_inverted_index(documents):
    """Build term postings and document-frequency dictionaries."""
    inverted_index = {}

    for doc_id in sorted(documents):
        tokens = tokenize(documents[doc_id]["text"])

        # A set prevents a repeated word from adding the same docID twice.
        unique_tokens = set(tokens)
        for term in unique_tokens:
            if term not in inverted_index:
                inverted_index[term] = []
            inverted_index[term].append(doc_id)

    document_frequency = {}
    for term, postings in inverted_index.items():
        postings.sort()
        document_frequency[term] = len(postings)

    return inverted_index, document_frequency


def build_stemmed_index(documents):
    """Build a separate inverted index containing Porter stems."""
    stemmed_index = {}

    for doc_id in sorted(documents):
        tokens = tokenize(documents[doc_id]["text"])
        stems = stem_tokens(tokens)
        unique_stems = set(stems)

        for stem in unique_stems:
            if stem not in stemmed_index:
                stemmed_index[stem] = []
            stemmed_index[stem].append(doc_id)

    for postings in stemmed_index.values():
        postings.sort()

    return stemmed_index


def build_positional_index(documents):
    """Build an unstemmed index of zero-based token positions."""
    positional_index = {}

    for doc_id in sorted(documents):
        tokens = tokenize(documents[doc_id]["text"])

        for position, term in enumerate(tokens):
            if term not in positional_index:
                positional_index[term] = {}
            if doc_id not in positional_index[term]:
                positional_index[term][doc_id] = []

            positional_index[term][doc_id].append(position)

    return positional_index


def search(term, inverted_index):
    """Return document IDs for one normalized term using the index."""
    query_tokens = tokenize(term)
    if len(query_tokens) != 1:
        raise ValueError("Search supports exactly one alphanumeric term.")

    normalized_term = query_tokens[0]
    return list(inverted_index.get(normalized_term, []))


def get_term_positions(term, positional_index):
    """Return docID-to-position mappings for exactly one normalized term."""
    query_tokens = tokenize(term)
    if len(query_tokens) != 1:
        raise ValueError("Position lookup supports exactly one alphanumeric term.")

    normalized_term = query_tokens[0]
    stored_positions = positional_index.get(normalized_term, {})
    result = {}

    for doc_id, positions in stored_positions.items():
        result[doc_id] = list(positions)

    return result


def stem_query_term(term):
    """Normalize and stem exactly one query term."""
    query_tokens = tokenize(term)
    if len(query_tokens) != 1:
        raise ValueError("Stemmed search supports exactly one alphanumeric term.")

    return stem_tokens(query_tokens)[0]


def search_stemmed(term, stemmed_index, documents):
    """Search the separate stemmed index and return document metadata."""
    query_stem = stem_query_term(term)
    matching_ids = stemmed_index.get(query_stem, [])

    matching_documents = []
    for doc_id in matching_ids:
        document = documents[doc_id].copy()
        document["doc_id"] = doc_id
        matching_documents.append(document)

    return matching_documents


def intersect_postings(postings1, postings2):
    """Return docIDs that occur in both sorted postings lists."""
    result = []
    pointer1 = 0
    pointer2 = 0

    while pointer1 < len(postings1) and pointer2 < len(postings2):
        doc_id1 = postings1[pointer1]
        doc_id2 = postings2[pointer2]

        if doc_id1 == doc_id2:
            result.append(doc_id1)
            pointer1 += 1
            pointer2 += 1
        elif doc_id1 < doc_id2:
            pointer1 += 1
        else:
            pointer2 += 1

    return result


def build_skip_pointers(postings):
    """Build deterministic index-to-index skips using sqrt spacing."""
    skip_pointers = {}
    skip_distance = int(math.sqrt(len(postings)))

    if skip_distance < 2:
        return skip_pointers

    source_index = 0
    destination_index = source_index + skip_distance

    while destination_index < len(postings):
        skip_pointers[source_index] = destination_index
        source_index = destination_index
        destination_index = source_index + skip_distance

    return skip_pointers


def intersect_postings_with_skips(
    postings1, postings2, skips1, skips2, statistics=None
):
    """Intersect sorted postings, using safe skip jumps when possible."""
    result = []
    pointer1 = 0
    pointer2 = 0
    comparisons = 0
    skip_jumps = 0

    while pointer1 < len(postings1) and pointer2 < len(postings2):
        comparisons += 1
        doc_id1 = postings1[pointer1]
        doc_id2 = postings2[pointer2]

        if doc_id1 == doc_id2:
            result.append(doc_id1)
            pointer1 += 1
            pointer2 += 1
        elif doc_id1 < doc_id2:
            if (
                pointer1 in skips1
                and postings1[skips1[pointer1]] <= doc_id2
            ):
                pointer1 = skips1[pointer1]
                skip_jumps += 1
            else:
                pointer1 += 1
        else:
            if (
                pointer2 in skips2
                and postings2[skips2[pointer2]] <= doc_id1
            ):
                pointer2 = skips2[pointer2]
                skip_jumps += 1
            else:
                pointer2 += 1

    if statistics is not None:
        statistics["comparisons"] = comparisons
        statistics["skip_jumps"] = skip_jumps

    return result


def search_and(term1, term2, inverted_index, documents):
    """Return metadata for documents containing both query terms."""
    postings1 = search(term1, inverted_index)
    postings2 = search(term2, inverted_index)
    matching_ids = intersect_postings(postings1, postings2)

    matching_documents = []
    for doc_id in matching_ids:
        document = documents[doc_id].copy()
        document["doc_id"] = doc_id
        matching_documents.append(document)

    return matching_documents


def search_and_with_skips(term1, term2, inverted_index, documents):
    """Return AND matches using separately generated skip pointers."""
    postings1 = search(term1, inverted_index)
    postings2 = search(term2, inverted_index)
    skips1 = build_skip_pointers(postings1)
    skips2 = build_skip_pointers(postings2)
    matching_ids = intersect_postings_with_skips(
        postings1, postings2, skips1, skips2
    )

    matching_documents = []
    for doc_id in matching_ids:
        document = documents[doc_id].copy()
        document["doc_id"] = doc_id
        matching_documents.append(document)

    return matching_documents


def union_postings(postings1, postings2):
    """Return sorted docIDs that occur in either postings list."""
    result = []
    pointer1 = 0
    pointer2 = 0

    while pointer1 < len(postings1) and pointer2 < len(postings2):
        doc_id1 = postings1[pointer1]
        doc_id2 = postings2[pointer2]

        if doc_id1 == doc_id2:
            if not result or result[-1] != doc_id1:
                result.append(doc_id1)
            pointer1 += 1
            pointer2 += 1
        elif doc_id1 < doc_id2:
            if not result or result[-1] != doc_id1:
                result.append(doc_id1)
            pointer1 += 1
        else:
            if not result or result[-1] != doc_id2:
                result.append(doc_id2)
            pointer2 += 1

    while pointer1 < len(postings1):
        doc_id1 = postings1[pointer1]
        if not result or result[-1] != doc_id1:
            result.append(doc_id1)
        pointer1 += 1

    while pointer2 < len(postings2):
        doc_id2 = postings2[pointer2]
        if not result or result[-1] != doc_id2:
            result.append(doc_id2)
        pointer2 += 1

    return result


def search_or(term1, term2, inverted_index, documents):
    """Return metadata for documents containing either query term."""
    postings1 = search(term1, inverted_index)
    postings2 = search(term2, inverted_index)
    matching_ids = union_postings(postings1, postings2)

    matching_documents = []
    for doc_id in matching_ids:
        document = documents[doc_id].copy()
        document["doc_id"] = doc_id
        matching_documents.append(document)

    return matching_documents


def complement_postings(all_doc_ids, postings):
    """Return sorted docIDs that do not occur in the postings list."""
    result = []
    postings_pointer = 0

    for doc_id in all_doc_ids:
        if (
            postings_pointer < len(postings)
            and doc_id == postings[postings_pointer]
        ):
            postings_pointer += 1
        else:
            result.append(doc_id)

    return result


def search_not(term, inverted_index, documents):
    """Return metadata for documents that do not contain the query term."""
    postings = search(term, inverted_index)
    all_doc_ids = sorted(documents)
    matching_ids = complement_postings(all_doc_ids, postings)

    matching_documents = []
    for doc_id in matching_ids:
        document = documents[doc_id].copy()
        document["doc_id"] = doc_id
        matching_documents.append(document)

    return matching_documents


def order_terms_by_frequency(terms, document_frequency):
    """Normalize terms and order them from lowest to highest frequency."""
    normalized_terms = []

    for term in terms:
        query_tokens = tokenize(term)
        if len(query_tokens) != 1:
            raise ValueError("Each query item must be one alphanumeric term.")
        normalized_terms.append(query_tokens[0])

    normalized_terms.sort(
        key=lambda normalized_term: document_frequency.get(normalized_term, 0)
    )
    return normalized_terms


def search_and_multiple(
    terms, inverted_index, document_frequency, documents
):
    """Return metadata for documents containing every supplied term."""
    ordered_terms = order_terms_by_frequency(terms, document_frequency)

    if not ordered_terms:
        return []

    for term in ordered_terms:
        if term not in inverted_index:
            return []

    matching_ids = list(inverted_index[ordered_terms[0]])

    for term in ordered_terms[1:]:
        postings = inverted_index[term]
        matching_ids = intersect_postings(matching_ids, postings)

        if not matching_ids:
            return []

    matching_documents = []
    for doc_id in matching_ids:
        document = documents[doc_id].copy()
        document["doc_id"] = doc_id
        matching_documents.append(document)

    return matching_documents


def search_boolean(query, inverted_index, documents):
    """Evaluate a Boolean query with precedence NOT, then AND, then OR."""
    query_tokens = tokenize_boolean_query(query)
    all_doc_ids = sorted(documents)
    position = 0

    if not query_tokens:
        raise ValueError("Boolean query cannot be empty.")

    def parse_primary():
        nonlocal position

        if position >= len(query_tokens):
            raise ValueError("Expected a term or opening parenthesis.")

        token = query_tokens[position]

        if token == "(":
            position += 1
            result = parse_or()

            if position >= len(query_tokens) or query_tokens[position] != ")":
                raise ValueError("Missing closing parenthesis.")

            position += 1
            return result

        if token in ("AND", "OR", "NOT", ")"):
            raise ValueError("Expected a term or opening parenthesis.")

        position += 1
        return search(token, inverted_index)

    def parse_not():
        nonlocal position

        if position < len(query_tokens) and query_tokens[position] == "NOT":
            position += 1
            postings = parse_not()
            return complement_postings(all_doc_ids, postings)

        return parse_primary()

    def parse_and():
        nonlocal position
        result = parse_not()

        while position < len(query_tokens) and query_tokens[position] == "AND":
            position += 1
            right_postings = parse_not()
            result = intersect_postings(result, right_postings)

        return result

    def parse_or():
        nonlocal position
        result = parse_and()

        while position < len(query_tokens) and query_tokens[position] == "OR":
            position += 1
            right_postings = parse_and()
            result = union_postings(result, right_postings)

        return result

    matching_ids = parse_or()

    if position != len(query_tokens):
        raise ValueError(f"Unexpected token: {query_tokens[position]}")

    matching_documents = []
    for doc_id in matching_ids:
        document = documents[doc_id].copy()
        document["doc_id"] = doc_id
        matching_documents.append(document)

    return matching_documents
