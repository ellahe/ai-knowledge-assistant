import tiktoken


class TokenizationEngine:

    def __init__(self, encoding_name: str = "cl100k_base"):
        self.encoding = tiktoken.get_encoding(encoding_name)

    def encode(self, text: str) -> list[int]:
        return self.encoding.encode(text)

    def count_tokens(self, text: str) -> int:
        return len(self.encode(text))

    def decode(self, token_ids: list[int]) -> str:
        return self.encoding.decode(token_ids)

    def analyze(self, text: str) -> dict:
        token_ids = self.encode(text)

        return {
            "text": text,
            "token_count": len(token_ids),
            "token_ids": token_ids,
            "tokens": [
                self.encoding.decode([token_id])
                for token_id in token_ids
            ]
        }