from engine import union_postings
from preprocessing import tokenize


SOUNDEX_DIGITS = {
    "a": "0",
    "e": "0",
    "i": "0",
    "o": "0",
    "u": "0",
    "h": "0",
    "w": "0",
    "y": "0",
    "b": "1",
    "f": "1",
    "p": "1",
    "v": "1",
    "c": "2",
    "g": "2",
    "j": "2",
    "k": "2",
    "q": "2",
    "s": "2",
    "x": "2",
    "z": "2",
    "d": "3",
    "t": "3",
    "l": "4",
    "m": "5",
    "n": "5",
    "r": "6",
}


def normalize_soundex_term(term):
    """Normalize exactly one alphabetic term for Soundex."""
    tokens = tokenize(term)
    if len(tokens) != 1 or not tokens[0].isalpha():
        raise ValueError("Soundex requires exactly one alphabetic term.")

    return tokens[0]


def soundex(term):
    """Return the four-character Soundex code from the lecture steps."""
    normalized_term = normalize_soundex_term(term)
    first_letter = normalized_term[0].upper()
    digit_codes = []

    for letter in normalized_term[1:]:
        digit_codes.append(SOUNDEX_DIGITS[letter])

    collapsed_codes = []
    for digit in digit_codes:
        if not collapsed_codes or digit != collapsed_codes[-1]:
            collapsed_codes.append(digit)

    nonzero_codes = []
    for digit in collapsed_codes:
        if digit != "0":
            nonzero_codes.append(digit)

    code = first_letter + "".join(nonzero_codes)
    while len(code) < 4:
        code += "0"

    return code[:4]


def build_soundex_index(inverted_index):
    """Build Soundex-code postings from normal dictionary terms."""
    soundex_index = {}

    for term in sorted(inverted_index):
        if not term.isalpha():
            continue

        code = soundex(term)
        if code not in soundex_index:
            soundex_index[code] = []
        if term not in soundex_index[code]:
            soundex_index[code].append(term)

    for terms in soundex_index.values():
        terms.sort()

    return soundex_index


def get_soundex_terms(query_term, soundex_index):
    """Return a query's Soundex code and matching dictionary terms."""
    code = soundex(query_term)
    matching_terms = list(soundex_index.get(code, []))
    return code, matching_terms


def search_soundex(
    query_term,
    soundex_index,
    inverted_index,
    documents,
    details=None,
):
    """Search normal postings for terms sharing the query's Soundex code."""
    normalized_query = normalize_soundex_term(query_term)
    code, matching_terms = get_soundex_terms(
        normalized_query, soundex_index
    )

    matching_doc_ids = []
    for term in matching_terms:
        postings = inverted_index.get(term, [])
        matching_doc_ids = union_postings(matching_doc_ids, postings)

    if details is not None:
        details["query"] = query_term
        details["normalized_query"] = normalized_query
        details["soundex_code"] = code
        details["matching_terms"] = list(matching_terms)
        details["matching_doc_ids"] = list(matching_doc_ids)

    matching_documents = []
    for doc_id in matching_doc_ids:
        document = documents[doc_id].copy()
        document["doc_id"] = doc_id
        matching_documents.append(document)

    return matching_documents
