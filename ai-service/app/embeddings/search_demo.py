from app.embeddings.embedding_engine import EmbeddingEngine
from app.embeddings.semantic_search import SemanticSearch


def main():
    engine = EmbeddingEngine()
    search_engine = SemanticSearch(engine)

    documents = [
        "Our company develops software applications.",
        "We build software products.",
        "Our company sells fresh vegetables."
    ]

    query = "I want to know about software development."

    results = search_engine.search(
        query=query,
        documents=documents,
        top_k=2
    )

    print("Search Query:", query)
    print("\nSearch Results:")

    for rank, result in enumerate(results, start=1):
        print(f"\nRank: {rank}")
        print(f"Document: {result['document']}")
        print(f"Similarity: {result['score']:.4f}")


if __name__ == "__main__":
    main()
