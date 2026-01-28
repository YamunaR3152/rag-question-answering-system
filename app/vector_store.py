import faiss
import numpy as np
from app.embeddings import generate_embeddings


class VectorStore:
    def __init__(self, dimension: int):
        self.index = faiss.IndexFlatL2(dimension)
        self.texts = []

    def add(self, embeddings, texts):
        print(f"\n📦 Adding {len(texts)} chunks to FAISS")

        if len(texts) > 0:
            print("📝 First chunk preview:", texts[0][:200], "\n")

        vectors = np.array(embeddings).astype("float32")

        if vectors.shape[1] != self.index.d:
            raise ValueError(
                f"Embedding dimension mismatch! Expected {self.index.d}, got {vectors.shape[1]}"
            )

        self.index.add(vectors)
        self.texts.extend(texts)

        print("✅ Stored successfully. Total chunks in DB:", len(self.texts))

    def search(self, query: str, top_k: int = 5):
        if len(self.texts) == 0:
            print("⚠️ No data in vector store yet")
            return []

        print(f"\n🔎 Searching for: {query}")

        query_embedding = generate_embeddings([query])[0]
        query_vector = np.array([query_embedding]).astype("float32")

        distances, indices = self.index.search(query_vector, top_k)

        print("📏 Distances:", distances)
        print("📍 Indices:", indices)

        results = []
        for i in indices[0]:
            if 0 <= i < len(self.texts):
                chunk = self.texts[i]
                print("🧩 Match preview:", chunk[:150], "\n")
                results.append(chunk)

        if not results:
            print("❌ No relevant chunks found")

        return results


# Helper function used by main.py
def retrieve_context(question: str, vector_store: VectorStore, top_k: int = 5):
    return vector_store.search(question, top_k)
