import faiss
import numpy as np
import pickle
import os
import torch
from pathlib import Path
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer

# --- CRITICAL STABILITY FIX FOR MAC ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class LocalVectorStore:
    def __init__(self, persist_dir: Path = None, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the vector store using SentenceTransformers and FAISS.
        Optimized for macOS stability and Audit data persistence.
        """
        # Limit FAISS to a single thread to prevent 'Silent Crashing' on Apple Silicon
        faiss.omp_set_num_threads(1)

        try:
            self.model = SentenceTransformer(model_name, device='cpu')
            self.model.eval()
        except Exception as e:
            print(f"Embedding Model Error: {e}")
            from sentence_transformers import SentenceTransformer as ST
            self.model = ST(model_name).to('cpu')

        self.dimension = self.model.get_sentence_embedding_dimension()

        # FIX: Aligning directory with the actual Pipeline structure
        if persist_dir is None:
            base_dir = Path(__file__).resolve().parent.parent
            # We use the standard path where the Pipeline expects to find it
            self.persist_dir = base_dir / "data" / "faiss"
        else:
            self.persist_dir = Path(persist_dir)

        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # Persistence paths
        self.index_path = str(self.persist_dir / "index.faiss")
        self.metadata_path = str(self.persist_dir / "metadata.pkl")
        self.id_map_path = str(self.persist_dir / "id_mapping.pkl")

        # Initialize FAISS index
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata_store = {}
        self.id_mapping = {}

        # Load existing data if available
        if os.path.exists(self.index_path):
            try:
                self.load()
            except Exception as e:
                print(f"Warning: Failed to load existing vector store: {e}")
                self._reset_store()

    def _reset_store(self):
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata_store = {}
        self.id_mapping = {}

    def add_chunks(self, chunks: List[Dict[str, Any]]):
        """Generates embeddings and adds them to the semantic index."""
        if not chunks:
            return

        texts_to_embed = [c["vector_text"] for c in chunks]

        with torch.no_grad():
            embeddings = self.model.encode(
                texts_to_embed,
                convert_to_numpy=True,
                show_progress_bar=False
            ).astype("float32")

        faiss.normalize_L2(embeddings)
        start_id = self.index.ntotal
        self.index.add(embeddings)

        for i, chunk in enumerate(chunks):
            idx = start_id + i
            self.metadata_store[idx] = chunk
            self.id_mapping[chunk["id"]] = idx

    def search(self, query: str, k: int = 5):
        """Semantic search for RAG."""
        if self.index.ntotal == 0:
            return []

        with torch.no_grad():
            query_vector = self.model.encode(
                [query],
                convert_to_numpy=True,
                show_progress_bar=False
            ).astype("float32")

        faiss.normalize_L2(query_vector)
        distances, indices = self.index.search(query_vector, k)

        results = []
        for i, idx in enumerate(indices[0]):
            idx_int = int(idx)
            if idx_int != -1 and idx_int in self.metadata_store:
                results.append({
                    "score": float(distances[0][i]),
                    "content": self.metadata_store[idx_int]
                })
        return results

    def persist(self):
        """Saves the database to disk. This is the heart of your knowledge base."""
        try:
            # 1. Save FAISS index
            faiss.write_index(self.index, self.index_path)

            # 2. Save Metadata
            with open(self.metadata_path, "wb") as f:
                pickle.dump(self.metadata_store, f)

            # 3. Save ID mapping
            with open(self.id_map_path, "wb") as f:
                pickle.dump(self.id_mapping, f)

            print(f"✅ Vector database successfully persisted at: {self.persist_dir}")
        except Exception as e:
            print(f"❌ Critical Persistence Error: {e}")

    def load(self):
        """Loads index and metadata into memory."""
        if not os.path.exists(self.index_path):
            return

        print(f"🔄 Loading Audit Index from: {self.persist_dir}")
        self.index = faiss.read_index(self.index_path)

        with open(self.metadata_path, "rb") as f:
            self.metadata_store = pickle.load(f)

        with open(self.id_map_path, "rb") as f:
            self.id_mapping = pickle.load(f)