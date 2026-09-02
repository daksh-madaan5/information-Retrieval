import math
import re
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


def term_bigrams(term):
    """Return character bigrams after adding both word boundaries."""
    boundary_term = "$" + term + "$"
    bigrams = []

    for index in range(len(boundary_term) - 1):
        bigrams.append(boundary_term[index:index + 2])

    return bigrams


def build_bigram_index(inverted_index):
    """Build bigram-to-dictionary-term postings from the normal index."""
    bigram_index = {}

    for term in sorted(inverted_index):
        bigrams = term_bigrams(term)

        for bigram in bigrams:
            if bigram not in bigram_index:
                bigram_index[bigram] = []
            if term not in bigram_index[bigram]:
                bigram_index[bigram].append(term)

    for terms in bigram_index.values():
        terms.sort()

    return bigram_index


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


def normalize_wildcard_pattern(pattern):
    """Normalize a pattern containing one or more star wildcards."""
    if not isinstance(pattern, str) or not pattern.strip():
        raise ValueError("Wildcard pattern cannot be empty.")
    if "*" not in pattern:
        raise ValueError("Wildcard pattern must contain at least one '*'.")

    raw_parts = pattern.split("*")
    normalized_parts = []

    for raw_part in raw_parts:
        part_tokens = tokenize(raw_part)
        if len(part_tokens) > 1:
            raise ValueError(
                "Each fixed wildcard part must be one term fragment."
            )
        normalized_parts.append(part_tokens[0] if part_tokens else "")

    normalized_pattern = "*".join(normalized_parts)
    while "**" in normalized_pattern:
        normalized_pattern = normalized_pattern.replace("**", "*")

    if not normalized_pattern.replace("*", ""):
        raise ValueError("A wildcard pattern needs a fixed term fragment.")

    return normalized_pattern


def wildcard_query_bigrams(pattern):
    """Return required bigrams without crossing wildcard boundaries."""
    normalized_pattern = normalize_wildcard_pattern(pattern)
    fixed_parts = normalized_pattern.split("*")
    required_bigrams = []

    for part_index, fixed_part in enumerate(fixed_parts):
        if not fixed_part:
            continue

        text_for_bigrams = fixed_part
        if part_index == 0 and not normalized_pattern.startswith("*"):
            text_for_bigrams = "$" + text_for_bigrams
        if (
            part_index == len(fixed_parts) - 1
            and not normalized_pattern.endswith("*")
        ):
            text_for_bigrams = text_for_bigrams + "$"

        for character_index in range(len(text_for_bigrams) - 1):
            bigram = text_for_bigrams[character_index:character_index + 2]
            if bigram not in required_bigrams:
                required_bigrams.append(bigram)

    return required_bigrams


def get_wildcard_candidates(pattern, bigram_index):
    """Intersect bigram term lists to obtain wildcard candidates."""
    required_bigrams = wildcard_query_bigrams(pattern)

    if not required_bigrams:
        all_dictionary_terms = set()
        for terms in bigram_index.values():
            for term in terms:
                all_dictionary_terms.add(term)
        return sorted(all_dictionary_terms)

    candidate_terms = None

    for bigram in required_bigrams:
        if bigram not in bigram_index:
            return []

        bigram_terms = set(bigram_index[bigram])
        if candidate_terms is None:
            candidate_terms = bigram_terms
        else:
            candidate_terms = candidate_terms.intersection(bigram_terms)

        if not candidate_terms:
            return []

    return sorted(candidate_terms)


def matches_wildcard(term, pattern):
    """Check one candidate term against the complete wildcard pattern."""
    normalized_pattern = normalize_wildcard_pattern(pattern)
    normalized_terms = tokenize(term)

    if len(normalized_terms) != 1:
        return False

    escaped_pattern = re.escape(normalized_pattern)
    regular_expression = escaped_pattern.replace(r"\*", ".*")

    return re.fullmatch(regular_expression, normalized_terms[0]) is not None


def search_wildcard(
    pattern, bigram_index, inverted_index, documents, details=None
):
    """Search normal postings after bigram generation and post-filtering."""
    normalized_pattern = normalize_wildcard_pattern(pattern)
    required_bigrams = wildcard_query_bigrams(normalized_pattern)
    candidate_terms = get_wildcard_candidates(
        normalized_pattern, bigram_index
    )
    matching_terms = []

    for term in candidate_terms:
        if matches_wildcard(term, normalized_pattern):
            matching_terms.append(term)

    matching_doc_ids = []
    for term in matching_terms:
        postings = inverted_index.get(term, [])
        matching_doc_ids = union_postings(matching_doc_ids, postings)

    if details is not None:
        details["pattern"] = normalized_pattern
        details["fixed_parts"] = [
            part for part in normalized_pattern.split("*") if part
        ]
        details["required_bigrams"] = list(required_bigrams)
        details["candidate_terms"] = list(candidate_terms)
        details["matching_terms"] = list(matching_terms)
        details["matching_doc_ids"] = list(matching_doc_ids)

    matching_documents = []
    for doc_id in matching_doc_ids:
        document = documents[doc_id].copy()
        document["doc_id"] = doc_id
        matching_documents.append(document)

    return matching_documents


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


def positional_intersect(positions1, positions2, distance=1):
    """Return positions from the first list followed at the given distance."""
    matching_positions = []
    pointer1 = 0
    pointer2 = 0

    while pointer1 < len(positions1) and pointer2 < len(positions2):
        expected_position = positions1[pointer1] + distance
        position2 = positions2[pointer2]

        if position2 == expected_position:
            matching_positions.append(positions1[pointer1])
            pointer1 += 1
            pointer2 += 1
        elif position2 < expected_position:
            pointer2 += 1
        else:
            pointer1 += 1

    return matching_positions


def search_phrase(phrase, positional_index, documents):
    """Return documents containing an exact phrase in consecutive positions."""
    phrase_terms = tokenize(phrase)
    if len(phrase_terms) < 2:
        raise ValueError("Phrase search requires at least two terms.")

    for term in phrase_terms:
        if term not in positional_index:
            return []

    candidate_doc_ids = sorted(positional_index[phrase_terms[0]])

    for term in phrase_terms[1:]:
        term_doc_ids = sorted(positional_index[term])
        candidate_doc_ids = intersect_postings(
            candidate_doc_ids, term_doc_ids
        )

        if not candidate_doc_ids:
            return []

    matching_doc_ids = []

    for doc_id in candidate_doc_ids:
        phrase_start_positions = list(
            positional_index[phrase_terms[0]][doc_id]
        )

        for distance, term in enumerate(phrase_terms[1:], start=1):
            term_positions = positional_index[term][doc_id]
            phrase_start_positions = positional_intersect(
                phrase_start_positions, term_positions, distance
            )

            if not phrase_start_positions:
                break

        if phrase_start_positions:
            matching_doc_ids.append(doc_id)

    matching_documents = []
    for doc_id in matching_doc_ids:
        document = documents[doc_id].copy()
        document["doc_id"] = doc_id
        matching_documents.append(document)

    return matching_documents


def positions_within_k(positions1, positions2, k):
    """Return True when two distinct sorted positions are at most k apart."""
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise ValueError("Proximity distance k must be a positive integer.")

    pointer1 = 0
    pointer2 = 0

    while pointer1 < len(positions1) and pointer2 < len(positions2):
        position1 = positions1[pointer1]
        position2 = positions2[pointer2]
        distance = abs(position1 - position2)

        if distance != 0 and distance <= k:
            return True

        if position1 <= position2:
            pointer1 += 1
        else:
            pointer2 += 1

    return False


def search_proximity(term1, term2, k, positional_index, documents):
    """Return documents where two terms occur within k token positions."""
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise ValueError("Proximity distance k must be a positive integer.")

    query_tokens1 = tokenize(term1)
    query_tokens2 = tokenize(term2)

    if len(query_tokens1) != 1 or len(query_tokens2) != 1:
        raise ValueError("Proximity search requires exactly two single terms.")

    normalized_term1 = query_tokens1[0]
    normalized_term2 = query_tokens2[0]

    if (
        normalized_term1 not in positional_index
        or normalized_term2 not in positional_index
    ):
        return []

    doc_ids1 = sorted(positional_index[normalized_term1])
    doc_ids2 = sorted(positional_index[normalized_term2])
    candidate_doc_ids = intersect_postings(doc_ids1, doc_ids2)
    matching_doc_ids = []

    for doc_id in candidate_doc_ids:
        positions1 = positional_index[normalized_term1][doc_id]
        positions2 = positional_index[normalized_term2][doc_id]

        if positions_within_k(positions1, positions2, k):
            matching_doc_ids.append(doc_id)

    matching_documents = []
    for doc_id in matching_doc_ids:
        document = documents[doc_id].copy()
        document["doc_id"] = doc_id
        matching_documents.append(document)

    return matching_documents


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
