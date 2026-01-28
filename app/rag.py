from app.embeddings import generate_embeddings
import numpy as np


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def retrieve_context(question: str, vector_store, top_k: int = 5, threshold: float = 0.65):
    """
    Returns only relevant chunks.
    If nothing is relevant, returns empty list.
    """
    # Embed the question
    query_embedding = generate_embeddings([question])[0]

    # Search FAISS
    distances, indices = vector_store.index.search(
        np.array([query_embedding]).astype("float32"), top_k
    )

    results = []
    for i in indices[0]:
        if i < len(vector_store.texts):
            chunk = vector_store.texts[i]
            chunk_embedding = generate_embeddings([chunk])[0]

            sim = cosine_similarity(query_embedding, chunk_embedding)

            # 🔥 Only keep chunk if similarity is high
            if sim >= threshold:
                results.append(chunk)

    return results
