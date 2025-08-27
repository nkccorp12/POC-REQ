"""
FIXED Core POM search functionality
Fixes the 1.0000 similarity score bug and improves search reliability
"""

import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Optional
import os
import warnings

from config import (
    PANEL_WORDS, EMBEDDINGS_FILE, MOLECULES_FILE, FAISS_INDEX_FILE,
    SENTENCE_TRANSFORMER_MODEL, POM_DIMENSIONS, RATA_DIMENSIONS,
    DEFAULT_K
)


class FixedPOMSearchEngine:
    """Fixed POM Search Engine with proper similarity calculations"""
    
    def __init__(self, debug_mode=False):
        self.sentence_model = None
        self.panel_embeddings = None
        self.faiss_index = None
        self.molecules_df = None
        self.embeddings = None
        self.debug_mode = debug_mode
        
    def initialize(self):
        """Initialize all components of the search engine"""
        if self.debug_mode:
            print("Initializing FIXED POM Search Engine...")
        
        # Load sentence transformer
        if self.debug_mode:
            print("Loading sentence transformer model...")
        self.sentence_model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)
        
        # Create panel word embeddings matrix
        if self.debug_mode:
            print("Creating panel word embeddings...")
        self._create_panel_embeddings()
        
        # Load POM embeddings and molecule data
        if self.debug_mode:
            print("Loading POM embeddings and molecule data...")
        self._load_data()
        
        # Load or build FAISS index
        if self.debug_mode:
            print("Loading FAISS index...")
        self._load_or_build_faiss_index()
        
        if self.debug_mode:
            print("Initialization complete!")
    
    def _create_panel_embeddings(self):
        """Create embeddings for the 55 panel words"""
        panel_vecs = self.sentence_model.encode(PANEL_WORDS)
        self.panel_embeddings = panel_vecs.T
        
    def _load_data(self):
        """Load embeddings and molecule metadata"""
        if not os.path.exists(EMBEDDINGS_FILE):
            raise FileNotFoundError(f"Embeddings file not found: {EMBEDDINGS_FILE}")
        if not os.path.exists(MOLECULES_FILE):
            raise FileNotFoundError(f"Molecules file not found: {MOLECULES_FILE}")
            
        self.embeddings = np.load(EMBEDDINGS_FILE)
        self.molecules_df = pd.read_csv(MOLECULES_FILE)
        
        if self.debug_mode:
            print(f"Loaded {len(self.embeddings)} embeddings with {self.embeddings.shape[1]} dimensions")
            print(f"Loaded {len(self.molecules_df)} molecules")
        
        # Validate data consistency
        if len(self.embeddings) != len(self.molecules_df):
            raise ValueError(f"Data mismatch: {len(self.embeddings)} embeddings vs {len(self.molecules_df)} molecules")
        
    def _load_faiss_index(self):
        """Load pre-built FAISS index from disk"""
        if not os.path.exists(FAISS_INDEX_FILE):
            return False
            
        try:
            if self.debug_mode:
                print(f"Loading FAISS index from: {FAISS_INDEX_FILE}")
            self.faiss_index = faiss.read_index(FAISS_INDEX_FILE)
            
            # Validate index
            if self.faiss_index.ntotal != len(self.embeddings):
                if self.debug_mode:
                    print(f"WARNING: Index size mismatch - index: {self.faiss_index.ntotal}, embeddings: {len(self.embeddings)}")
                return False
            
            if self.debug_mode:
                print(f"SUCCESS: Loaded FAISS index with {self.faiss_index.ntotal} vectors")
            return True
        except Exception as e:
            if self.debug_mode:
                print(f"WARNING: Failed to load FAISS index: {e}")
            return False
    
    def _load_or_build_faiss_index(self):
        """Load existing FAISS index or build new one if not found"""
        if self._load_faiss_index():
            return
            
        if self.debug_mode:
            print("BUILDING: FAISS index not found, building new one...")
            print("TIP: Run 'python build_index_fixed.py' once to speed up future startups")
        self._build_faiss_index()
        
    def _build_faiss_index(self):
        """Build FAISS index for similarity search with proper validation"""
        # Prepare embeddings
        embeddings_norm = self.embeddings.astype('float32')
        
        if self.debug_mode:
            original_norms = np.linalg.norm(embeddings_norm, axis=1)
            print(f"   Original embedding norms: min={original_norms.min():.4f}, max={original_norms.max():.4f}")
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings_norm)
        
        if self.debug_mode:
            normalized_norms = np.linalg.norm(embeddings_norm, axis=1)
            print(f"   Normalized embedding norms: min={normalized_norms.min():.4f}, max={normalized_norms.max():.4f}")
        
        # Create and populate index
        self.faiss_index = faiss.IndexFlatIP(POM_DIMENSIONS)
        self.faiss_index.add(embeddings_norm)
        
        if self.debug_mode:
            print(f"SUCCESS: Built FAISS index with {self.faiss_index.ntotal} vectors")
    
    def prompt_to_rata(self, prompt: str) -> np.ndarray:
        """Convert text prompt to 55-dimensional RATA vector with enhanced vanilla detection"""
        if self.sentence_model is None or self.panel_embeddings is None:
            raise RuntimeError("Search engine not initialized. Call initialize() first.")
        
        # Enhanced RATA generation with focus on vanilla queries
        rata_vec = np.zeros(55)
        
        # Clean and tokenize prompt
        prompt_lower = prompt.lower()
        prompt_tokens = prompt_lower.replace(',', ' ').replace(';', ' ').split()
        
        if self.debug_mode:
            print(f"   Tokenized prompt: {prompt_tokens}")
        
        # Direct matching with boosted vanilla terms
        key_vanilla_words = {'vanilla', 'sweet', 'creamy', 'fruity', 'buttery', 'dairy'}
        
        for i, panel_word in enumerate(PANEL_WORDS):
            if panel_word in prompt_tokens:
                if panel_word in key_vanilla_words:
                    rata_vec[i] = 100.0  # Strong signal for vanilla-related words
                    if self.debug_mode:
                        print(f"   KEY Direct match: '{panel_word}' = {rata_vec[i]}")
                else:
                    rata_vec[i] = 50.0
                    if self.debug_mode:
                        print(f"   Direct match: '{panel_word}' = {rata_vec[i]}")
            elif any(panel_word in token or token in panel_word for token in prompt_tokens):
                if panel_word in key_vanilla_words:
                    rata_vec[i] = 60.0  # Partial match for vanilla words
                else:
                    rata_vec[i] = 30.0
        
        # Semantic similarity enhancement
        if self.debug_mode:
            print(f"   After Direct Matching: {np.sum(rata_vec > 0)} non-zero values, sum={np.sum(rata_vec):.2f}")
        
        clean_prompt = ' '.join(prompt_tokens)
        prompt_vec = self.sentence_model.encode([clean_prompt])
        panel_vecs = self.sentence_model.encode(PANEL_WORDS)
        
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(prompt_vec, panel_vecs)[0]
        
        # Apply semantic enhancement
        threshold = 0.02
        boost_factor = 15.0 if np.sum(rata_vec) > 0 else 20.0
        
        semantic_boost = np.where(similarities > threshold, similarities * boost_factor, 0)
        rata_vec = np.maximum(rata_vec, semantic_boost)
        
        # Special boost for vanilla queries
        if any(term in prompt_lower for term in ['vanilla', 'vanillin']):
            if self.debug_mode:
                print(f"   VANILLA QUERY: Applying special boosts")
            
            vanilla_dims = {
                'vanilla': 1.5,
                'buttery': 1.4, 
                'fruity': 1.3,
                'sweet': 1.2,
                'green': 1.1,
                'sulfurous': 1.1
            }
            
            for word, boost in vanilla_dims.items():
                if word in PANEL_WORDS:
                    idx = PANEL_WORDS.index(word)
                    if rata_vec[idx] > 0.01:  # Only boost if already activated
                        rata_vec[idx] *= boost
        
        # Normalize
        if np.sum(rata_vec) == 0:
            rata_vec = np.ones(55) * 0.1  # Fallback
        
        if np.linalg.norm(rata_vec) > 0:
            rata_vec = rata_vec / np.linalg.norm(rata_vec)
        
        if self.debug_mode:
            print(f"   Final RATA stats: min={rata_vec.min():.4f}, max={rata_vec.max():.4f}")
        
        return rata_vec
    
    def search_molecules(self, rata_vector: np.ndarray, k: int = DEFAULT_K) -> pd.DataFrame:
        """Search for similar molecules using RATA vector with FIXED similarity calculation"""
        if self.faiss_index is None:
            raise RuntimeError("FAISS index not built. Call initialize() first.")
        
        if self.debug_mode:
            print(f"FAISS SEARCH (FIXED VERSION):")
            print(f"   Input RATA vector shape: {rata_vector.shape}")
            print(f"   Input RATA vector norm: {np.linalg.norm(rata_vector):.4f}")
        
        # Prepare query vector
        query_vector = rata_vector.astype('float32').reshape(1, -1)
        faiss.normalize_L2(query_vector)
        
        if self.debug_mode:
            print(f"   Query vector after normalization: {np.linalg.norm(query_vector):.4f}")
        
        # FAISS search
        distances, indices = self.faiss_index.search(query_vector, k)
        
        if self.debug_mode:
            print(f"   FAISS raw scores: {distances[0]}")
            print(f"   FAISS indices: {indices[0]}")
        
        # CRITICAL FIX: Handle the similarity scores properly
        similarities = distances[0].copy()  # Direct copy, no clipping!
        
        # Validate similarity range
        if np.all(similarities == 1.0):
            if self.debug_mode:
                print("   WARNING: All similarities are 1.0 - possible FAISS issue")
            
            # FALLBACK: Manual cosine similarity calculation
            if self.debug_mode:
                print("   FALLBACK: Using manual cosine similarity")
            
            similarities = self._manual_cosine_search(rata_vector, indices[0])
        
        # Ensure reasonable range (but don't clip to [0,1] - that was the bug!)
        if similarities.max() > 1.5 or similarities.min() < -1.5:
            if self.debug_mode:
                print(f"   WARNING: Extreme similarity values: [{similarities.min():.4f}, {similarities.max():.4f}]")
        
        if self.debug_mode:
            print(f"   Final similarities: {similarities}")
            print(f"   Similarity range: [{similarities.min():.4f}, {similarities.max():.4f}]")
        
        # Get results
        results = self.molecules_df.iloc[indices[0]].copy()
        results['similarity_score'] = similarities
        results = results.sort_values('similarity_score', ascending=False)
        
        return results
    
    def _manual_cosine_search(self, query_vector: np.ndarray, indices: np.ndarray) -> np.ndarray:
        """Manual cosine similarity calculation as fallback"""
        similarities = []
        query_norm = query_vector / np.linalg.norm(query_vector)
        
        for idx in indices:
            if idx < len(self.embeddings):
                db_vector = self.embeddings[idx]
                db_norm = db_vector / np.linalg.norm(db_vector) 
                cosine_sim = np.dot(query_norm, db_norm)
                similarities.append(cosine_sim)
            else:
                similarities.append(0.0)
        
        return np.array(similarities)
        
    def search_by_prompt(self, prompt: str, k: int = DEFAULT_K) -> Tuple[np.ndarray, pd.DataFrame]:
        """End-to-end search: prompt -> RATA -> molecule matches"""
        rata_vector = self.prompt_to_rata(prompt)
        results = self.search_molecules(rata_vector, k)
        return rata_vector, results


# Global fixed search engine instance
_fixed_search_engine = None

def get_fixed_search_engine(debug_mode=False) -> FixedPOMSearchEngine:
    """Get the global fixed search engine instance"""
    global _fixed_search_engine
    if _fixed_search_engine is None:
        _fixed_search_engine = FixedPOMSearchEngine(debug_mode=debug_mode)
        _fixed_search_engine.initialize()
    return _fixed_search_engine

def search_odor_fixed(prompt: str, k: int = DEFAULT_K, debug_mode=False) -> Tuple[np.ndarray, pd.DataFrame]:
    """Fixed version of odor search with proper similarity calculation"""
    engine = get_fixed_search_engine(debug_mode=debug_mode)
    return engine.search_by_prompt(prompt, k)


# Test function
def test_fixed_search():
    """Test the fixed search engine"""
    print("TESTING FIXED SEARCH ENGINE")
    print("=" * 40)
    
    test_queries = [
        "sweet, vanilla, creamy",
        "vanilla",
        "buttery vanilla",
        "fruity lemon"
    ]
    
    for query in test_queries:
        print(f"\\n🧪 Testing: '{query}'")
        try:
            rata_vector, results = search_odor_fixed(query, k=5, debug_mode=True)
            
            print(f"✅ Success: {len(results)} results")
            print(f"   Similarity scores: {results['similarity_score'].values}")
            
            # Check for the vanillin bug
            scores = results['similarity_score'].values
            if np.all(scores == 1.0):
                print("❌ BUG PERSISTS: All scores are 1.0")
            elif len(set(scores)) == 1:
                print(f"⚠️  WARNING: All scores identical: {scores[0]}")
            else:
                print("✅ FIXED: Scores are diverse")
                
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    test_fixed_search()