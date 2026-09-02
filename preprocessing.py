import re

from nltk.stem import PorterStemmer


STOP_WORDS = {"a", "an", "and", "are", "is", "of", "the", "to"}
porter_stemmer = PorterStemmer()


def normalize_token(token):
    """Normalize one token to a consistent lowercase form."""
    return token.casefold()


def tokenize(text):
    """Convert text to lowercase alphanumeric word tokens."""
    raw_tokens = re.findall(r"[a-zA-Z0-9]+", text)
    normalized_tokens = []

    for token in raw_tokens:
        normalized_tokens.append(normalize_token(token))

    return normalized_tokens


def remove_stop_words(tokens):
    """Remove words in the small demonstration stop-word list."""
    filtered_tokens = []

    for token in tokens:
        if token not in STOP_WORDS:
            filtered_tokens.append(token)

    return filtered_tokens


def stem_tokens(tokens):
    """Apply English Porter stemming to a list of tokens."""
    stemmed_tokens = []

    for token in tokens:
        stemmed_tokens.append(porter_stemmer.stem(token))

    return stemmed_tokens


def tokenize_boolean_query(query):
    """Extract Boolean operators, parentheses, and normalized terms."""
    raw_tokens = re.findall(r"[a-zA-Z0-9]+|[()]", query)
    query_tokens = []

    for token in raw_tokens:
        normalized_token = normalize_token(token)
        if normalized_token in ("and", "or", "not"):
            query_tokens.append(normalized_token.upper())
        else:
            query_tokens.append(normalized_token)

    return query_tokens
