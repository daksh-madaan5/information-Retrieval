import re


def tokenize(text):
    """Convert text to lowercase alphanumeric word tokens."""
    return re.findall(r"[a-z0-9]+", text.lower())
