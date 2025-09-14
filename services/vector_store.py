import numpy as np
import faiss
import os
from openai import OpenAI
from typing import List, Dict, Any, Tuple
import streamlit as st
import pickle
import hashlib
import random
import time
from openai import RateLimitError, APIStatusError

class VectorStore:
    """Handles vector embeddings and similarity search using FAISS"""
    
    def __init__(self):
        # the newest OpenAI model is "gpt-5" which was released August 7, 2025.
        # do not change this unless explicitly requested by the user
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.embedding_model = "text-embedding-3-large"
        self.dimension = None  # Will be set dynamically based on first embedding
        self.embeddings_mode = os.getenv("EMBEDDINGS_MODE", "openai")  # openai, mock, local
        
        # FAISS index
        self.index = None
        self.connections_data = []
        self.connection_ids = []
        
        # Cache for embeddings
        self.cache_file = ".cache/embeddings.pkl"
        self.embeddings_cache = self._load_cache()
    
    def create_embeddings(self, connections: List[Dict[str, Any]]):
        """Create embeddings for all connections and build FAISS index"""
        try:
            # Check for empty data
            if not connections:
                st.warning("⚠️ No connections data to process. Please check your CSV file.")
                return
            
            # Extract searchable texts
            texts = [conn['searchable_text'] for conn in connections]
            
            if not texts:
                st.warning("⚠️ No valid text data found in connections.")
                return
            
            # Create embeddings in batches to handle rate limits
            embeddings = []
            batch_size = 100
            auto_fallback_triggered = False
            
            progress_bar = st.progress(0)
            total_batches = (len(texts) + batch_size - 1) // batch_size
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                try:
                    batch_embeddings = self._get_embeddings_batch(batch_texts)
                    embeddings.extend(batch_embeddings)
                except Exception as e:
                    error_msg = str(e).lower()
                    if ("insufficient_quota" in error_msg or "invalid" in error_msg or 
                        "rate limit" in error_msg) and not auto_fallback_triggered:
                        # Auto-fallback to mock embeddings
                        st.warning("🔄 API issue detected. Automatically switching to mock embeddings to continue processing...")
                        self.embeddings_mode = "mock"
                        auto_fallback_triggered = True
                        batch_embeddings = self._get_embeddings_batch(batch_texts)
                        embeddings.extend(batch_embeddings)
                    else:
                        raise e
                
                # Update progress
                progress = min((i + batch_size) / len(texts), 1.0)
                progress_bar.progress(progress)
            
            progress_bar.empty()
            
            # Convert to numpy array and set dimension dynamically
            embeddings_array = np.array(embeddings, dtype=np.float32)
            
            # Set dimension from first embedding if not set
            if self.dimension is None:
                self.dimension = len(embeddings[0])
            
            # Create FAISS index
            self.index = faiss.IndexFlatIP(self.dimension)  # Inner Product (cosine similarity)
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings_array)
            
            # Add embeddings to index
            self.index.add(embeddings_array)
            
            # Store connection data
            self.connections_data = connections
            self.connection_ids = [conn['id'] for conn in connections]
            
            st.success(f"✅ Created embeddings for {len(connections)} connections")
            
        except Exception as e:
            raise Exception(f"Error creating embeddings: {str(e)}")
    
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
    
    def _save_cache(self):
        """Save embeddings cache to file"""
        try:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.embeddings_cache, f)
        except Exception:
            pass
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        return f"{self.embedding_model}_{text_hash}"
    
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
        if self.embeddings_mode == "mock":
            return [self._get_mock_embedding(text) for text in texts]
        
        # OpenAI embeddings with caching and error handling
        embeddings = []
        texts_to_process = []
        cached_embeddings = []
        
        # Check cache first
        for text in texts:
            cache_key = self._get_cache_key(text)
            if cache_key in self.embeddings_cache:
                cached_embeddings.append((len(embeddings), self.embeddings_cache[cache_key]))
            else:
                texts_to_process.append((len(embeddings), text))
            embeddings.append(None)  # Placeholder
        
        # Fill cached embeddings
        for idx, embedding in cached_embeddings:
            embeddings[idx] = embedding
        
        # Process remaining texts with API
        if texts_to_process:
            api_texts = [text for _, text in texts_to_process]
            max_retries = 3
            retry_delay = 1  # Initial delay in seconds
            
            for attempt in range(max_retries):
                try:
                    response = self.openai_client.embeddings.create(
                        model=self.embedding_model,
                        input=api_texts
                    )
                    
                    # Fill API embeddings and cache them
                    for i, (orig_idx, text) in enumerate(texts_to_process):
                        embedding = response.data[i].embedding
                        embeddings[orig_idx] = embedding
                        
                        # Cache the embedding
                        cache_key = self._get_cache_key(text)
                        self.embeddings_cache[cache_key] = embedding
                    
                    # Save cache
                    self._save_cache()
                    break  # Success, exit retry loop
                    
                except RateLimitError as e:
                    if attempt < max_retries - 1:
                        # Add jitter to avoid thundering herd
                        jitter = random.uniform(0.1, 0.5)
                        sleep_time = retry_delay * (2 ** attempt) + jitter
                        time.sleep(min(sleep_time, 8))  # Cap at 8 seconds
                        continue
                    else:
                        raise Exception(f"Rate limit exceeded after {max_retries} attempts: {str(e)}")
                        
                except APIStatusError as e:
                    if "insufficient_quota" in str(e).lower():
                        raise Exception("insufficient_quota: Your OpenAI API key has exceeded its quota. Please check your billing plan or try mock embeddings mode.")
                    elif e.status_code in [429, 500, 502, 503, 504] and attempt < max_retries - 1:
                        # Retry on server errors and rate limits
                        jitter = random.uniform(0.1, 0.5)
                        sleep_time = retry_delay * (2 ** attempt) + jitter
                        time.sleep(min(sleep_time, 8))
                        continue
                    else:
                        raise Exception(f"OpenAI API error: {str(e)}")
                        
                except Exception as e:
                    if attempt < max_retries - 1:
                        # Generic retry for other errors
                        time.sleep(retry_delay)
                        retry_delay *= 2
                        continue
                    else:
                        raise Exception(f"Error getting embeddings from OpenAI after {max_retries} attempts: {str(e)}")
        
        return embeddings
    
    def similarity_search(self, query: str, k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """Perform similarity search for a query"""
        if self.index is None:
            raise Exception("Vector store not initialized. Please process data first.")
        
        try:
            # Get query embedding
            query_embedding = self._get_embeddings_batch([query])[0]
            query_vector = np.array([query_embedding], dtype=np.float32)
            
            # Normalize for cosine similarity
            faiss.normalize_L2(query_vector)
            
            # Search
            k_search = min(k, len(self.connections_data))
            scores, indices = self.index.search(query_vector, k_search)
            
            # Prepare results
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.connections_data):  # Valid index
                    connection = self.connections_data[idx]
                    results.append((connection, float(score)))
            
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
