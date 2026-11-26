from sentence_transformers import SentenceTransformer


class EmbeddingEngine:

    def __init__(
        self,
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"
    ):
        self.model = SentenceTransformer(model_name)

    def embed(self, text: str):
        return self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

    def embed_batch(self, texts: list[str]):
        return self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

    def get_dimension(self) -> int:
        return self.model.get_embedding_dimension()
