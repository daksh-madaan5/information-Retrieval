from pathlib import Path

from engine import (
    build_inverted_index,
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
    stem_query_term,
)
from preprocessing import remove_stop_words, stem_tokens, tokenize


def main():
    documents_directory = Path(__file__).parent / "documents"
    documents = load_documents(documents_directory)
    inverted_index, document_frequency = build_inverted_index(documents)
    stemmed_index = build_stemmed_index(documents)
    positional_index = build_positional_index(documents)

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
        else:
            print("Invalid option.")
            return
    except ValueError as error:
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
