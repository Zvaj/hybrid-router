import os
import numpy as np
from sentence_transformers import SentenceTransformer


class RAGBackend:
    def __init__(self, chapters_dir, chunk_size=512, overlap=50):
        self.available = False
        self.chunks = []
        self._embeddings = None

        model_name = "all-MiniLM-L6-v2"
        self.model = SentenceTransformer(model_name)

        md_files = [
            os.path.join(chapters_dir, f)
            for f in os.listdir(chapters_dir)
            if f.endswith(".md")
        ] if os.path.isdir(chapters_dir) else []

        if not md_files:
            print(f"RAG warning: no .md files found in {chapters_dir}")
            return

        all_chunks = []
        for path in md_files:
            with open(path, encoding="utf-8") as f:
                text = f.read()
            all_chunks.extend(self._chunk_text(text, chunk_size, overlap))

        self.chunks = all_chunks
        self._build_index(all_chunks)
        self.available = True
        print(f"RAG index built: {len(all_chunks)} chunks")

    def _chunk_text(self, text, size, overlap):
        words = text.split()
        chunks = []
        step = size - overlap
        for start in range(0, len(words), step):
            chunk_words = words[start: start + size]
            if len(chunk_words) < 20:
                continue
            chunks.append(" ".join(chunk_words))
        return chunks

    @staticmethod
    def _l2_normalize(vecs):
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.maximum(norms, 1e-10)

    def _build_index(self, chunks):
        embeddings = self.model.encode(chunks, convert_to_numpy=True).astype("float32")
        self._embeddings = self._l2_normalize(embeddings)

    def retrieve(self, query, k=3):
        if not self.available:
            return ["No prose content available"]

        q_vec = self.model.encode([query], convert_to_numpy=True).astype("float32")
        q_vec = self._l2_normalize(q_vec)
        scores = (self._embeddings @ q_vec.T).flatten()
        top_indices = np.argsort(scores)[::-1][:k]

        results = []
        total_words = 0
        for idx in top_indices:
            if idx < 0 or idx >= len(self.chunks):
                continue
            chunk = self.chunks[int(idx)]
            chunk_words = len(chunk.split())
            if total_words + chunk_words > 1500:
                break
            results.append(chunk)
            total_words += chunk_words

        return results
