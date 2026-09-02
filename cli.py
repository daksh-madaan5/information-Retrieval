"""Development command-line demonstrations for the IR project."""

from pathlib import Path

from context_spelling import suggest_context_correction
from edit_distance import (
    build_edit_distance_matrix,
    edit_distance,
    normalize_comparison_term,
    weighted_edit_distance,
)
from engine import (
    build_inverted_index,
    build_bigram_index,
    build_positional_index,
    build_skip_pointers,
    build_stemmed_index,
    intersect_postings,
    intersect_postings_with_skips,
    get_term_positions,
    load_documents,
    order_terms_by_frequency,
    search,
    search_and,
    search_and_with_skips,
    search_and_multiple,
    search_boolean,
    search_not,
    search_or,
    search_phrase,
    search_proximity,
    search_stemmed,
    search_wildcard,
    stem_query_term,
)
from preprocessing import remove_stop_words, stem_tokens, tokenize
from spelling import (
    build_trigram_index,
    get_ngram_spelling_candidates,
    normalize_spelling_term,
    rank_candidates_by_jaccard,
    suggest_spelling,
    term_trigrams,
)
from soundex import build_soundex_index, search_soundex
from thesaurus import search_with_thesaurus


def main():
    documents_directory = Path(__file__).parent / "documents"
    documents = load_documents(documents_directory)
    inverted_index, document_frequency = build_inverted_index(documents)
    stemmed_index = build_stemmed_index(documents)
    positional_index = build_positional_index(documents)
    bigram_index = build_bigram_index(inverted_index)
    trigram_index = build_trigram_index(inverted_index)
    soundex_index = build_soundex_index(inverted_index)

    print(f"Indexed documents: {len(documents)}")
    print(f"Vocabulary size: {len(inverted_index)}")

    print("1. Single-term search")
    print("2. AND search")
    print("3. OR search")
    print("4. NOT search")
    print("5. Multi-term AND search")
    print("6. Boolean expression search")
    print("7. Preprocessing demonstration")
    print("8. Stemmed search")
    print("9. Skip-pointer demonstration")
    print("10. Positional index lookup")
    print("11. Phrase search")
    print("12. Proximity search")
    print("13. Wildcard search")
    print("14. Edit distance demonstration")
    print("15. Spelling candidate demonstration")
    print("16. Spelling correction")
    print("17. Weighted edit distance demonstration")
    print("18. Context-sensitive spelling demonstration")
    print("19. Soundex search")
    print("20. WordNet thesaurus/query expansion")
    choice = input("Choose an option: ")

    try:
        if choice == "1":
            term = input("Enter one search term: ")
            matching_ids = search(term, inverted_index)
            matches = []
            for doc_id in matching_ids:
                document = documents[doc_id].copy()
                document["doc_id"] = doc_id
                matches.append(document)
        elif choice == "2":
            term1 = input("Enter the first term: ")
            term2 = input("Enter the second term: ")
            matches = search_and(term1, term2, inverted_index, documents)
        elif choice == "3":
            term1 = input("Enter the first term: ")
            term2 = input("Enter the second term: ")
            matches = search_or(term1, term2, inverted_index, documents)
        elif choice == "4":
            term = input("Enter one term to exclude: ")
            matches = search_not(term, inverted_index, documents)
        elif choice == "5":
            query = input("Enter terms separated by spaces: ")
            terms = query.split()
            processing_order = order_terms_by_frequency(
                terms, document_frequency
            )
            print("Processing order:", " -> ".join(processing_order))
            matches = search_and_multiple(
                terms, inverted_index, document_frequency, documents
            )
        elif choice == "6":
            query = input("Enter a Boolean expression: ")
            matches = search_boolean(query, inverted_index, documents)
        elif choice == "7":
            sentence = input("Enter a sentence: ")
            tokens = tokenize(sentence)
            tokens_without_stop_words = remove_stop_words(tokens)
            stemmed_tokens = stem_tokens(tokens)

            print("Original:")
            print(sentence)
            print("Tokens:")
            print(tokens)
            print("Without stop words:")
            print(tokens_without_stop_words)
            print("Stemmed:")
            print(stemmed_tokens)
            return
        elif choice == "8":
            term = input("Enter one search term: ")
            query_stem = stem_query_term(term)
            matches = search_stemmed(term, stemmed_index, documents)
            print(f"Query: {term}")
            print(f"Stem used: {query_stem}")
        elif choice == "9":
            postings1 = [2, 4, 8, 16, 32, 64, 128]
            postings2 = [31, 32, 100]
            skips1 = build_skip_pointers(postings1)
            skips2 = build_skip_pointers(postings2)
            statistics = {}

            normal_result = intersect_postings(postings1, postings2)
            skip_result = intersect_postings_with_skips(
                postings1, postings2, skips1, skips2, statistics
            )

            print("Postings 1:", postings1)
            print("Skip pointers 1:", skips1)
            print("Postings 2:", postings2)
            print("Skip pointers 2:", skips2)
            print("Normal AND result:", normal_result)
            print("Skip-pointer AND result:", skip_result)
            print("Comparisons:", statistics["comparisons"])
            print("Skip jumps:", statistics["skip_jumps"])
            return
        elif choice == "10":
            term = input("Enter one term: ")
            positions_by_document = get_term_positions(
                term, positional_index
            )

            if not positions_by_document:
                print("No occurrences found.")
                return

            print(f"Term: {tokenize(term)[0]}")
            for doc_id, positions in positions_by_document.items():
                print(documents[doc_id]["filename"])
                position_text = ", ".join(
                    str(position) for position in positions
                )
                print(f"Positions: {position_text}")
                print("Token sequence:")
                tokens = tokenize(documents[doc_id]["text"])
                for position, token in enumerate(tokens):
                    print(f"{position}: {token}")
            return
        elif choice == "11":
            phrase = input("Enter a phrase: ")
            matches = search_phrase(phrase, positional_index, documents)
            normalized_phrase = " ".join(tokenize(phrase))
            print(f'Phrase: "{normalized_phrase}"')
        elif choice == "12":
            term1 = input("First term: ")
            term2 = input("Second term: ")
            k = int(input("Maximum distance k: "))
            matches = search_proximity(
                term1, term2, k, positional_index, documents
            )
            normalized_term1 = tokenize(term1)[0]
            normalized_term2 = tokenize(term2)[0]
            print(f"Query: {normalized_term1} /{k} {normalized_term2}")
        elif choice == "13":
            print("Use one or more '*' characters as wildcards.")
            pattern = input("Enter a wildcard pattern: ")
            details = {}
            matches = search_wildcard(
                pattern,
                bigram_index,
                inverted_index,
                documents,
                details,
            )
            print(f"Wildcard query: {details['pattern']}")
            print("Fixed parts:", " | ".join(details["fixed_parts"]))
            print("Required bigrams:", ", ".join(details["required_bigrams"]))
            print(
                "Candidate dictionary terms:",
                ", ".join(details["candidate_terms"]) or "None",
            )
            print(
                "Matching dictionary terms:",
                ", ".join(details["matching_terms"]) or "None",
            )
            print("Wildcard search uses additional index processing.")
        elif choice == "14":
            word1 = input("First word: ")
            word2 = input("Second word: ")
            normalized_word1 = normalize_comparison_term(word1)
            normalized_word2 = normalize_comparison_term(word2)
            distance = edit_distance(normalized_word1, normalized_word2)
            matrix = build_edit_distance_matrix(
                normalized_word1, normalized_word2
            )

            print(f"Word 1: {normalized_word1}")
            print(f"Word 2: {normalized_word2}")
            print(f"Edit distance: {distance}")
            print("DP matrix:")

            column_labels = ['""'] + list(normalized_word2)
            print("    " + " ".join(f"{label:>2}" for label in column_labels))

            row_labels = ['""'] + list(normalized_word1)
            for row_label, row_values in zip(row_labels, matrix):
                values = " ".join(f"{value:>2}" for value in row_values)
                print(f"{row_label:>2}  {values}")
            return
        elif choice == "15":
            query_term = input("Enter one term: ")
            normalized_query = normalize_spelling_term(query_term)
            query_trigrams = term_trigrams(normalized_query)
            candidates = get_ngram_spelling_candidates(
                normalized_query, trigram_index
            )
            ranking = rank_candidates_by_jaccard(
                normalized_query, candidates
            )

            print("Normalized query:", normalized_query)
            print("Trigrams:", ", ".join(query_trigrams))
            print(
                "Candidate dictionary terms:",
                ", ".join(candidates) or "None",
            )

            if not ranking:
                print("No candidates found.")
                return

            print("Jaccard ranking:")
            for term, score in ranking:
                print(f"{term} -> {score:.4f}")
            return
        elif choice == "16":
            query_term = input("Word: ")
            details = suggest_spelling(
                query_term, trigram_index, inverted_index
            )

            print("Query:", details["normalized_query"])

            if details["already_correct"]:
                print(
                    f"{details['normalized_query']} is already in the dictionary."
                )
                return

            print("Candidate terms:")
            if details["candidate_terms"]:
                for term in details["candidate_terms"]:
                    print(term)
            else:
                print("None")

            print("Ranking:")
            for item in details["ranked_suggestions"]:
                print(
                    f"{item['term']}  edit distance = "
                    f"{item['edit_distance']}, Jaccard = "
                    f"{item['jaccard']:.4f}"
                )

            if details["suggestion"]:
                print("Suggested correction:")
                print(details["suggestion"])
            else:
                print("No spelling suggestion found.")
            return
        elif choice == "17":
            word1 = input("First word: ")
            word2 = input("Second word: ")
            normalized_word1 = normalize_comparison_term(word1)
            normalized_word2 = normalize_comparison_term(word2)
            demonstration_weights = {
                ("m", "n"): 0.5,
                ("n", "m"): 0.5,
            }

            standard_distance = edit_distance(
                normalized_word1, normalized_word2
            )
            weighted_distance = weighted_edit_distance(
                normalized_word1,
                normalized_word2,
                replacement_costs=demonstration_weights,
            )

            print("Demonstration weights:")
            print("m <-> n replacement cost = 0.5")
            print("other replacements = 1.0")
            print("Word 1:", normalized_word1)
            print("Word 2:", normalized_word2)
            print("Standard edit distance:", standard_distance)
            print("Weighted edit distance:", weighted_distance)
            return
        elif choice == "18":
            query_phrase = input("Phrase: ")
            details = suggest_context_correction(
                query_phrase,
                positional_index,
                inverted_index,
                documents,
            )

            print("Original query:", details["original_phrase"])
            print(
                "Original phrase matches:",
                len(details["original_matching_doc_ids"]),
            )

            if details["original_matching_doc_ids"]:
                print("Original phrase already matches.")
                print("No correction needed.")
                return

            print("Context alternatives with hits:")
            if details["successful_alternatives"]:
                for alternative in details["successful_alternatives"]:
                    print(
                        f"{alternative['phrase']} -> "
                        f"{alternative['hit_count']} document(s)"
                    )
            else:
                print("None")

            if details["suggestion"]:
                print("Suggested query:", details["suggestion"])
            else:
                print("No context-sensitive correction found.")
            return
        elif choice == "19":
            query_term = input("Enter one name or term: ")
            details = {}
            matches = search_soundex(
                query_term,
                soundex_index,
                inverted_index,
                documents,
                details,
            )

            print("Query:", details["normalized_query"])
            print("Soundex code:", details["soundex_code"])
            print("Matching dictionary terms:")
            if details["matching_terms"]:
                for term in details["matching_terms"]:
                    print(term)
            else:
                print("None")

            if not matches:
                print("Matching files:")
                print("None")
                return
        elif choice == "20":
            query_term = input("Enter one term: ")
            details = {}
            matches = search_with_thesaurus(
                query_term,
                inverted_index,
                documents,
                details=details,
            )
            ordinary_doc_ids = search(query_term, inverted_index)

            print("Original query:", details["normalized_query"])
            print("WordNet synonyms:")
            if details["wordnet_synonyms"]:
                print(", ".join(details["wordnet_synonyms"]))
            else:
                print("None")

            print(
                "Indexed expansion:",
                " OR ".join(details["indexed_expansion_terms"]),
            )
            print("Ordinary single-term matching files:")
            if ordinary_doc_ids:
                for doc_id in ordinary_doc_ids:
                    print(f"{doc_id}: {documents[doc_id]['filename']}")
            else:
                print("None")

            print("Matching files (expanded query):")
            if matches:
                for document in matches:
                    print(
                        f"{document['doc_id']}: {document['filename']}"
                    )
            else:
                print("None")
            return
        else:
            print("Invalid option.")
            return
    except (ValueError, LookupError) as error:
        print(error)
        return

    if not matches:
        print("No matching documents.")
        return

    print("Matching files:")
    for document in matches:
        print(f"{document['doc_id']}: {document['filename']}")


if __name__ == "__main__":
    main()
