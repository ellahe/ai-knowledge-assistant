
import math


def cosine_similarity(vector_a: list[float], vector_b: list[float]) -> float:
    if len(vector_a) != len(vector_b):
        raise ValueError("Vectors must have the same dimensions.")

    dot_product = sum(
        a * b for a, b in zip(vector_a, vector_b)
    )

    magnitude_a = math.sqrt(sum(a * a for a in vector_a))
    magnitude_b = math.sqrt(sum(b * b for b in vector_b))

    if magnitude_a == 0 or magnitude_b == 0:
        raise ValueError("Zero vectors are not supported.")

    return dot_product / (magnitude_a * magnitude_b)


def main():
    vector_a = [0.82, 0.31, 0.15, 0.91]
    vector_b = [0.79, 0.35, 0.18, 0.88]
    vector_c = [0.12, 0.91, 0.76, 0.22]

    print("A vs B:", cosine_similarity(vector_a, vector_b))
    print("A vs C:", cosine_similarity(vector_a, vector_c))


if __name__ == "__main__":
    main()
