from pathlib import Path

from preprocessing import tokenize


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


def search(term, inverted_index):
    """Return document IDs for one normalized term using the index."""
    query_tokens = tokenize(term)
    if len(query_tokens) != 1:
        raise ValueError("Search supports exactly one alphanumeric term.")

    normalized_term = query_tokens[0]
    return list(inverted_index.get(normalized_term, []))


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
