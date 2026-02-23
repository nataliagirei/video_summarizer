import faiss
import numpy as np
import pickle
import os
from pathlib import Path
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer


class LocalVectorStore:
    def __init__(self, persist_dir: str = None, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the vector store using SentenceTransformers and FAISS.
        Optimized for cosine similarity using Inner Product (IP) index and L2 normalization.
        """
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()

        # Dynamic path configuration for project data structure
        if persist_dir is None:
            base_dir = Path(__file__).resolve().parent.parent
            self.persist_dir = base_dir / "data" / "audio" / "processed" / "faiss"
        else:
            self.persist_dir = Path(persist_dir)

        self.index_path = str(self.persist_dir / "index.faiss")
        self.metadata_path = str(self.persist_dir / "metadata.pkl")
        self.vectors_path = str(self.persist_dir / "vectors.npy")
        self.id_map_path = str(self.persist_dir / "id_mapping.pkl")

        # Ensure database directory exists
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # Initialize FAISS index for Inner Product (used for cosine similarity after normalization)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata_store = {}
        self.id_mapping = {}
        self.vectors = []

        if os.path.exists(self.index_path):
            self.load()

    def add_chunks(self, chunks: List[Dict[str, Any]]):
        """
        Embed text chunks, normalize them, and add to the FAISS index.
        """
        if not chunks:
            return

        texts_to_embed = [c["vector_text"] for c in chunks]
        embeddings = self.model.encode(texts_to_embed).astype("float32")

        # Normalize vectors for cosine similarity
        faiss.normalize_L2(embeddings)

        start_id = self.index.ntotal
        self.index.add(embeddings)

        for i, chunk in enumerate(chunks):
            idx = start_id + i
            self.metadata_store[idx] = chunk
            self.id_mapping[chunk["id"]] = idx

        self.vectors.append(embeddings)

    def search(self, query: str, k: int = 5):
        """
        Search for the most semantically similar chunks based on a query.
        """
        query_vector = self.model.encode([query]).astype("float32")

        # Normalize query vector to maintain cosine similarity calculation
        faiss.normalize_L2(query_vector)

        distances, indices = self.index.search(query_vector, k)

        results = []
        for i, idx in enumerate(indices[0]):
            idx_int = int(idx)
            if idx_int != -1 and idx_int in self.metadata_store:
                results.append({
                    "score": float(distances[0][i]),  # Now represents cosine similarity score
                    "content": self.metadata_store[idx_int]
                })
        return results

    def persist(self):
        """
        Save the FAISS index and associated metadata to disk.
        """
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, "wb") as f:
            pickle.dump(self.metadata_store, f)
        with open(self.id_map_path, "wb") as f:
            pickle.dump(self.id_mapping, f)

        if self.vectors:
            combined_vectors = np.vstack(self.vectors)
            np.save(self.vectors_path, combined_vectors)

        print(f"Vector database persisted at: {self.persist_dir}")

    def load(self):
        """
        Load the FAISS index and metadata from disk.
        """
        print(f"Loading existing vector database from: {self.persist_dir}")
        self.index = faiss.read_index(self.index_path)
        with open(self.metadata_path, "rb") as f:
            self.metadata_store = pickle.load(f)
        with open(self.id_map_path, "rb") as f:
            self.id_mapping = pickle.load(f)

        if os.path.exists(self.vectors_path):
            loaded_vectors = np.load(self.vectors_path)
            self.vectors = [loaded_vectors]
        else:
            self.vectors = []