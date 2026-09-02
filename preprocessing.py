import re


def tokenize(text):
    """Convert text to lowercase alphanumeric word tokens."""
    return re.findall(r"[a-z0-9]+", text.lower())


def tokenize_boolean_query(query):
    """Extract Boolean operators, parentheses, and normalized terms."""
    raw_tokens = re.findall(r"[a-z0-9]+|[()]", query.lower())
    query_tokens = []

    for token in raw_tokens:
        if token in ("and", "or", "not"):
            query_tokens.append(token.upper())
        else:
            query_tokens.append(token)

    return query_tokens
