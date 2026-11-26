from app.embeddings.embedding_engine import EmbeddingEngine
from sentence_transformers import util


def main():
    engine = EmbeddingEngine()

    texts = [
        "I love programming.",
        "I enjoy software development.",
        "من برنامه نویسی را دوست دارم.",
        "امروز هوا آفتابی است."
    ]

    embeddings = engine.embed_batch(texts)

    print("Embedding Dimension:", engine.get_dimension())
    print("Embedding Shape:", embeddings.shape)

    print("\nFirst Vector (first 8 dimensions):")
    print(embeddings[0][:8])

    print("\nSemantic Similarities:")

    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            score = util.cos_sim(
                embeddings[i],
                embeddings[j]
            ).item()

            print(f"\n{texts[i]}")
            print(f"{texts[j]}")
            print(f"Similarity: {score:.4f}")


if __name__ == "__main__":
    main()
