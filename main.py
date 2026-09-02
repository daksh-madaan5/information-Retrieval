from pathlib import Path

from engine import (
    build_inverted_index,
    load_documents,
    order_terms_by_frequency,
    search,
    search_and,
    search_and_multiple,
    search_boolean,
    search_not,
    search_or,
)


def main():
    documents_directory = Path(__file__).parent / "documents"
    documents = load_documents(documents_directory)
    inverted_index, document_frequency = build_inverted_index(documents)

    print(f"Indexed documents: {len(documents)}")
    print(f"Vocabulary size: {len(inverted_index)}")

    print("1. Single-term search")
    print("2. AND search")
    print("3. OR search")
    print("4. NOT search")
    print("5. Multi-term AND search")
    print("6. Boolean expression search")
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
