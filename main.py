from pathlib import Path

from engine import (
    build_inverted_index,
    build_stemmed_index,
    load_documents,
    order_terms_by_frequency,
    search,
    search_and,
    search_and_multiple,
    search_boolean,
    search_not,
    search_or,
    search_stemmed,
    stem_query_term,
)
from preprocessing import remove_stop_words, stem_tokens, tokenize


def main():
    documents_directory = Path(__file__).parent / "documents"
    documents = load_documents(documents_directory)
    inverted_index, document_frequency = build_inverted_index(documents)
    stemmed_index = build_stemmed_index(documents)

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
