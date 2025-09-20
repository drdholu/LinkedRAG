import numpy as np
try:
    import faiss  # type: ignore
    FAISS_AVAILABLE = True
except Exception:
    FAISS_AVAILABLE = False
import os
from openai import OpenAI
from typing import List, Dict, Any, Tuple
import streamlit as st
import pickle
import hashlib
import random
import time
from openai import RateLimitError, APIStatusError
from utils.helpers import setup_logger

class VectorStore:
    """Handles vector embeddings and similarity search using FAISS"""
    
    def __init__(self):
        # Initialize logger
        self.logger = setup_logger("LinkedRAG.VectorStore", level=os.getenv("LOG_LEVEL", "INFO"))

        # the newest OpenAI model is "gpt-5" which was released August 7, 2025.
        # do not change this unless explicitly requested by the user
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.embedding_model = "text-embedding-3-large"
        self.dimension = None  # Will be set dynamically based on first embedding
        self.embeddings_mode = os.getenv("EMBEDDINGS_MODE", "openai").lower()  # openai, mock, ollama
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip('/')
        self.ollama_embedding_model = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
        self.index_embeddings_mode = None  # Embeddings mode used to build current index
        
        # FAISS index
        self.index = None  # FAISS index or numpy ndarray when FAISS is unavailable
        self.connections_data = []
        self.connection_ids = []
        
        # Cache for embeddings
        self.cache_file = ".cache/embeddings.pkl"
        self.embeddings_cache = self._load_cache()

        # Persistence paths
        self.data_dir = "data"
        self.index_path = os.path.join(self.data_dir, "index.faiss" if FAISS_AVAILABLE else "index.npy")
        self.meta_path = os.path.join(self.data_dir, "meta.pkl")
        self.connections_path = os.path.join(self.data_dir, "connections.pkl")


    def load_from_disk(self) -> bool:
        """Load index, metadata, and connections from disk if available. Returns True on success."""
        try:
            if not (os.path.exists(self.meta_path) and os.path.exists(self.connections_path) and os.path.exists(self.index_path)):
                return False
            # Load metadata
            with open(self.meta_path, 'rb') as f:
                meta = pickle.load(f)
            self.embedding_model = meta.get('embedding_model', self.embedding_model)
            self.dimension = meta.get('dimension', self.dimension)
            self.index_embeddings_mode = meta.get('embeddings_mode', self.embeddings_mode)
            # Load index
            if FAISS_AVAILABLE:
                self.index = faiss.read_index(self.index_path)
            else:
                self.index = np.load(self.index_path)
            # Load connections
            with open(self.connections_path, 'rb') as f:
                payload = pickle.load(f)
                self.connections_data = payload.get('connections', [])
                self.connection_ids = payload.get('ids', [])
            return self.index is not None and len(self.connections_data) > 0
        except Exception:
            return False


    def _load_cache(self) -> Dict[str, List[float]]:
        """Load embeddings cache from file"""
        try:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'rb') as f:
                    return pickle.load(f)
        except Exception:
            pass
        return {}


    def create_embeddings(self, connections: List[Dict[str, Any]]):
        """Create embeddings for all connections and build FAISS index with consistent vector dimensions."""
        try:
            # Check for empty data
            if not connections:
                st.warning("⚠️ No connections data to process. Please check your CSV file.")
                self.logger.warning("No connections data to process")
                return

            # Extract searchable texts
            texts = [conn['searchable_text'] for conn in connections]

            if not texts:
                st.warning("⚠️ No valid text data found in connections.")
                self.logger.warning("No valid text data found in connections")
                return

            progress_bar = st.progress(0)

            # Always compute embeddings using a single mode for the entire dataset
            original_mode = self.embeddings_mode
            used_mode = original_mode

            def compute_all(texts_list, target_mode):
                return self._compute_embeddings_all(texts_list, target_mode, progress_bar)

            try:
                embeddings = compute_all(texts, original_mode)
            except Exception as e:
                error_msg = str(e).lower()
                if ("insufficient_quota" in error_msg or "invalid" in error_msg or "rate limit" in error_msg):
                    st.warning("🔄 API issue detected. Recomputing all embeddings in mock mode to continue processing...")
                    used_mode = "mock"
                    embeddings = compute_all(texts, used_mode)
                else:
                    raise

            progress_bar.empty()

            # Convert to numpy array and set dimension dynamically
            if not embeddings or not embeddings[0] or not isinstance(embeddings[0], (list, tuple)):
                raise Exception("Received invalid embeddings format")

            expected_dim = len(embeddings[0])
            # Validate all dimensions match before creating the array
            for vec in embeddings:
                if len(vec) != expected_dim:
                    raise Exception("Embedding dimension mismatch detected after computation")

            embeddings_array = np.asarray(embeddings, dtype=np.float32)

            # Set dimension from first embedding if not set
            if self.dimension is None:
                self.dimension = expected_dim

            # Build index (FAISS if available, otherwise numpy fallback)
            if FAISS_AVAILABLE:
                self.index = faiss.IndexFlatIP(self.dimension)  # Inner Product (cosine similarity)
                # Normalize embeddings for cosine similarity
                faiss.normalize_L2(embeddings_array)
                # Add embeddings to index
                self.index.add(embeddings_array)
            else:
                # Normalize and store as numpy matrix
                norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                self.index = embeddings_array / norms

            # Store connection data
            self.connections_data = connections
            self.connection_ids = [conn['id'] for conn in connections]
            self.index_embeddings_mode = used_mode

            st.success(f"✅ Created embeddings for {len(connections)} connections")

            # Persist to disk
            self._save_to_disk()

        except Exception as e:
            raise Exception(f"Error creating embeddings: {str(e)}")
    
    def _save_cache(self):
        """Save embeddings cache to file"""
        try:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.embeddings_cache, f)
        except Exception:
            pass
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text that is specific to mode, model, and dimension."""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        mode = self.embeddings_mode
        if mode == "ollama":
            model_id = self.ollama_embedding_model
        elif mode == "mock":
            model_id = f"mock-{self.dimension or 0}"
        else:
            model_id = self.embedding_model
        dim_part = str(self.dimension or "unk")
        return f"{mode}:{model_id}:{dim_part}:{text_hash}"
    
    def _get_mock_embedding(self, text: str, dimension: int = 384) -> List[float]:
        """Generate deterministic mock embedding for text"""
        # Use text hash as seed for reproducible embeddings
        text_hash = hashlib.md5(text.encode()).hexdigest()
        seed = int(text_hash[:8], 16)
        random.seed(seed)
        
        # Generate normalized random vector
        embedding = [random.gauss(0, 1) for _ in range(dimension)]
        norm = sum(x**2 for x in embedding) ** 0.5
        return [x / norm for x in embedding]
    
    def _get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a batch of texts"""
        # Prepare list with placeholders and check cache for all modes
        embeddings: List[List[float] | None] = []
        to_compute: List[Tuple[int, str]] = []
        for text in texts:
            cache_key = self._get_cache_key(text)
            if cache_key in self.embeddings_cache:
                embeddings.append(self.embeddings_cache[cache_key])
            else:
                to_compute.append((len(embeddings), text))
                embeddings.append(None)

        if not to_compute:
            # All hits from cache
            return [e for e in embeddings if e is not None]  # type: ignore

        # Compute missing based on mode
        if self.embeddings_mode == "mock":
            mock_dim = self.dimension if self.dimension else 384
            computed = [self._get_mock_embedding(text, dimension=mock_dim) for _, text in to_compute]
        elif self.embeddings_mode == "ollama":
            texts_only = [text for _, text in to_compute]
            computed = self._get_ollama_embeddings_parallel(texts_only)
        else:
            # OpenAI embeddings with retries
            api_texts = [text for _, text in to_compute]
            max_retries = 3
            retry_delay = 1
            computed = []
            for attempt in range(max_retries):
                try:
                    response = self.openai_client.embeddings.create(
                        model=self.embedding_model,
                        input=api_texts
                    )
                    computed = [d.embedding for d in response.data]
                    break
                except RateLimitError as e:
                    if attempt < max_retries - 1:
                        jitter = random.uniform(0.1, 0.5)
                        sleep_time = retry_delay * (2 ** attempt) + jitter
                        time.sleep(min(sleep_time, 8))
                        continue
                    else:
                        raise Exception(f"Rate limit exceeded after {max_retries} attempts: {str(e)}")
                except APIStatusError as e:
                    if "insufficient_quota" in str(e).lower():
                        raise Exception("insufficient_quota: Your OpenAI API key has exceeded its quota. Please check your billing plan or try mock embeddings mode.")
                    elif e.status_code in [429, 500, 502, 503, 504] and attempt < max_retries - 1:
                        jitter = random.uniform(0.1, 0.5)
                        sleep_time = retry_delay * (2 ** attempt) + jitter
                        time.sleep(min(sleep_time, 8))
                        continue
                    else:
                        raise Exception(f"OpenAI API error: {str(e)}")
                except Exception as e:
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        retry_delay *= 2
                        continue
                    else:
                        raise Exception(f"Error getting embeddings from OpenAI after {max_retries} attempts: {str(e)}")

        # Fill results and cache them
        for (orig_idx, text), emb in zip(to_compute, computed):
            embeddings[orig_idx] = emb
            self.embeddings_cache[self._get_cache_key(text)] = emb
        self._save_cache()

        # All should be filled now
        return [e for e in embeddings if e is not None]  # type: ignore

    def _compute_embeddings_all(self, texts: List[str], target_mode: str, progress_bar) -> List[List[float]]:
        """Compute embeddings for all texts using a single target mode, ensuring consistent vector dimensions."""
        original_mode = self.embeddings_mode
        try:
            self.embeddings_mode = target_mode
            embeddings: List[List[float]] = []
            batch_size = 100
            expected_dim: int | None = None

            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_embeddings = self._get_embeddings_batch(batch_texts)
                if not batch_embeddings:
                    continue
                if expected_dim is None:
                    expected_dim = len(batch_embeddings[0])
                # Validate dimension consistency within batch
                for vec in batch_embeddings:
                    if len(vec) != expected_dim:
                        raise Exception("Embedding dimension mismatch within batch")
                embeddings.extend(batch_embeddings)
                # Update progress
                progress = min((i + batch_size) / len(texts), 1.0)
                try:
                    progress_bar.progress(progress)
                except Exception:
                    pass

            if expected_dim is None:
                raise Exception("No embeddings were generated")
            return embeddings
        finally:
            self.embeddings_mode = original_mode

    def _get_ollama_embeddings_parallel(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings from local Ollama server in parallel to speed up large datasets."""
        import requests
        from concurrent.futures import ThreadPoolExecutor, as_completed
        url = f"{self.ollama_base_url}/api/embeddings"
        max_workers = int(os.getenv("OLLAMA_EMBED_CONCURRENCY", "4"))

        def fetch_one(text: str) -> List[float]:
            resp = requests.post(url, json={"model": self.ollama_embedding_model, "prompt": text}, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            emb = data.get('embedding') or data.get('vector') or data.get('data')
            if isinstance(emb, list):
                return [float(x) for x in emb]
            raise Exception("Unexpected Ollama embedding response format")

        results: List[List[float]] = [None] * len(texts)  # type: ignore
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {executor.submit(fetch_one, t): i for i, t in enumerate(texts)}
            for future in as_completed(future_to_idx):
                i = future_to_idx[future]
                results[i] = future.result()
        # Set dimension from first vector if needed
        if self.dimension is None and results and results[0] is not None:
            self.dimension = len(results[0])
        return results  # type: ignore
    
    def similarity_search(self, query: str, k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """Perform similarity search for a query"""
        if self.index is None:
            raise Exception("Vector store not initialized. Please process data first.")
        
        try:
            # Ensure query uses the same mode and dimension as the index
            original_mode = self.embeddings_mode
            if self.index_embeddings_mode:
                self.embeddings_mode = self.index_embeddings_mode
            # Get query embedding
            query_embedding = self._get_embeddings_batch([query])[0]
            # Restore mode
            self.embeddings_mode = original_mode
            query_vector = np.array([query_embedding], dtype=np.float32)
            
            # Normalize for cosine similarity and search
            k_search = min(k, len(self.connections_data))
            results: List[Tuple[Dict[str, Any], float]] = []
            if FAISS_AVAILABLE:
                faiss.normalize_L2(query_vector)
                scores, indices = self.index.search(query_vector, k_search)
                for score, idx in zip(scores[0], indices[0]):
                    if idx < len(self.connections_data):
                        connection = self.connections_data[idx]
                        results.append((connection, float(score)))
            else:
                # numpy cosine similarity with normalized matrix in self.index
                q_norm = np.linalg.norm(query_vector, axis=1, keepdims=True)
                q_norm[q_norm == 0] = 1.0
                q = query_vector / q_norm
                sims = (self.index @ q.T).ravel()
                top_idx = np.argsort(-sims)[:k_search]
                for idx in top_idx:
                    if idx < len(self.connections_data):
                        results.append((self.connections_data[int(idx)], float(sims[int(idx)])))
            return results
            
        except Exception as e:
            raise Exception(f"Error performing similarity search: {str(e)}")
    
    def search_by_company(self, company_name: str, k: int = 10) -> List[Dict[str, Any]]:
        """Search connections by company name"""
        if not self.connections_data:
            return []
        
        company_lower = company_name.lower()
        results = []
        
        for conn in self.connections_data:
            if company_lower in conn.get('company', '').lower():
                results.append(conn)
        
        return results[:k]
    
    def search_by_keywords(self, keywords: List[str], k: int = 10) -> List[Tuple[Dict[str, Any], float]]:
        """Search connections by multiple keywords"""
        query = ' '.join(keywords)
        return self.similarity_search(query, k)
    
    def get_all_companies(self) -> List[str]:
        """Get all unique companies from connections"""
        if not self.connections_data:
            return []
        
        companies = set()
        for conn in self.connections_data:
            company = conn.get('company', '').strip()
            if company and company.lower() != 'unknown':
                companies.add(company)
        
        return sorted(list(companies))
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get statistics about the connections"""
        if not self.connections_data:
            return {}
        
        total_connections = len(self.connections_data)
        companies = self.get_all_companies()
        
        # Count connections per company
        company_counts = {}
        for conn in self.connections_data:
            company = conn.get('company', '').strip()
            if company:
                company_counts[company] = company_counts.get(company, 0) + 1
        
        # Get top companies
        top_companies = sorted(company_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'total_connections': total_connections,
            'total_companies': len(companies),
            'top_companies': top_companies,
            'connections_with_email': len([c for c in self.connections_data if c.get('email')]),
            'connections_with_position': len([c for c in self.connections_data if c.get('position')])
        }

    def _save_to_disk(self):
        """Persist index, metadata, and connections to disk."""
        try:
            os.makedirs(self.data_dir, exist_ok=True)
            # Save index
            if FAISS_AVAILABLE and self.index is not None:
                faiss.write_index(self.index, self.index_path)
            elif not FAISS_AVAILABLE and isinstance(self.index, np.ndarray):
                np.save(self.index_path, self.index)
            # Save metadata
            meta = {
                'embedding_model': self.embedding_model,
                'embeddings_mode': self.index_embeddings_mode or self.embeddings_mode,
                'dimension': self.dimension,
            }
            with open(self.meta_path, 'wb') as f:
                pickle.dump(meta, f)
            # Save connections
            with open(self.connections_path, 'wb') as f:
                pickle.dump({'connections': self.connections_data, 'ids': self.connection_ids}, f)
        except Exception:
            # Non-fatal: do not crash app on save failure
            pass

    