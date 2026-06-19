from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_documents(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip()]


def search_documents(query, documents, top_k=3):
    vectorizer = TfidfVectorizer(stop_words="english")

    document_vectors = vectorizer.fit_transform(documents)
    query_vector = vectorizer.transform([query])

    similarities = cosine_similarity(query_vector, document_vectors)[0]

    ranked_results = sorted(
        enumerate(similarities),
        key=lambda item: item[1],
        reverse=True
    )

    return ranked_results[:top_k]


def main():
    documents = load_documents("data/documents.txt")

    query = input("Enter your search query: ")

    results = search_documents(query, documents, top_k=3)

    print("\nTop matching documents:\n")

    for rank, (index, score) in enumerate(results, start=1):
        print(f"{rank}. {documents[index]}")
        print(f"Similarity Score: {score:.4f}\n")


if __name__ == "__main__":
    main()