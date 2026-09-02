# Information Retrieval Search

A Python information-retrieval system implementing Boolean retrieval, positional queries, tolerant retrieval, and query expansion over a 10-document text corpus.

## Features

- Inverted index, Boolean AND/OR/NOT, and query optimization
- Stemming and skip pointers
- Positional indexing, phrase search, and proximity search
- Wildcard retrieval using bigram indexing
- Spelling correction using trigrams, Jaccard similarity, and edit distance
- Weighted edit distance demonstration and context-sensitive spelling correction
- Soundex search and WordNet query expansion
- Automatic query syntax detection and a local Flask search interface

## Corpus

- 10 custom educational text documents
- Approximately 1,000 lines per document
- Approximately 10,000 lines in total

## Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m nltk.downloader wordnet
```

WordNet data is required only for thesaurus search.

## Run

```powershell
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000).

Optional development CLI:

```powershell
python cli.py
```

## Query examples

```text
computer
machine learning              -> multi-term AND
computer AND machine
"machine learning"
machine /4 computer
comput*
```

Advanced examples:

```text
Spelling  -> computar
Soundex   -> Steven
Thesaurus -> car
```

## Tests

```powershell
python -m pytest
```
