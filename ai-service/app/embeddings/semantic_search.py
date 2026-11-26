from sentence_transformers import util

from app.embeddings.embedding_engine import EmbeddingEngine


class SemanticSearch:

    def __init__(self, embedding_engine: EmbeddingEngine):
        self.embedding_engine = embedding_engine

    def search(
        self,
        query: str,
        documents: list[str],
        top_k: int = 3
    ) -> list[dict[str, str | float]]:

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        if top_k < 1:
            raise ValueError("top_k must be greater than zero.")

        if not documents:
            return []

        query_embedding = self.embedding_engine.embed(query)

        document_embeddings = self.embedding_engine.embed_batch(documents)

        similarities = util.cos_sim(
            query_embedding,
            document_embeddings
        )[0]

        ranked_indices = sorted(
            range(len(documents)),
            key=lambda index: similarities[index].item(),
            reverse=True
        )

        results = []

        for index in ranked_indices[:top_k]:
            results.append({
                "document": documents[index],
                "score": float(similarities[index].item())
            })

        return results
