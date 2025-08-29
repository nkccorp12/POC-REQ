"""
Core POM search functionality for the Describe-to-Radar POC
Implements text-to-vector conversion and FAISS-based similarity search
"""

import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Optional
import os
import threading
import streamlit as st

try:
    import torch
except Exception:
    torch = None

from config import (
    PANEL_WORDS, EMBEDDINGS_FILE, MOLECULES_FILE, FAISS_INDEX_FILE,
    SENTENCE_TRANSFORMER_MODEL, POM_DIMENSIONS, RATA_DIMENSIONS,
    DEFAULT_K
)

# Removed German dictionary - system now works with English only

# Global singleton for model loading with thread safety
_MODEL_SINGLETON = None
_MODEL_LOCK = threading.Lock()

@st.cache_resource
def load_faiss_index_cached(embeddings_file: str, molecules_file: str):
    """Cache FAISS index and embeddings to prevent rebuilding on every run"""
    print("Loading cached FAISS index and embeddings...")
    
    # Load embeddings
    embeddings = np.load(embeddings_file)
    print(f"Loaded embeddings: {embeddings.shape}")
    
    # Normalize for cosine similarity
    embeddings_norm = embeddings.astype('float32').copy()
    faiss.normalize_L2(embeddings_norm)
    
    # Create and populate index
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings_norm)
    
    print(f"FAISS index built with {index.ntotal} vectors")
    return index, embeddings

@st.cache_resource  
def load_sentence_model_cached(model_name: str):
    """Cache + singleton sentence transformer model (offline-first with smart fallback)."""
    global _MODEL_SINGLETON
    with _MODEL_LOCK:
        if _MODEL_SINGLETON is None:
            local_dir = os.getenv("SENTENCE_MODEL_DIR", "/app/models/all-MiniLM-L6-v2")
            
            # Smart path selection with validation
            if os.path.isdir(local_dir) and len(os.listdir(local_dir)) > 0:
                use_path = local_dir
                print(f"✅ Loading from cached model: {use_path}")
            else:
                use_path = model_name
                print(f"⚠️ Local model dir missing or empty, falling back to repo ID: {model_name}")
                if os.path.isdir(local_dir):
                    files = os.listdir(local_dir)
                    print(f"   Cache dir exists but has {len(files)} files: {files[:3]}")
                else:
                    print(f"   Cache dir does not exist: {local_dir}")
            
            try:
                print(f"🔄 Initializing SentenceTransformer...")
                m = SentenceTransformer(use_path, device="cpu")
                
                # Threads hart begrenzen (Free Tier)
                if torch is not None:
                    try: 
                        torch.set_num_threads(1)
                        print("✅ Set torch threads to 1")
                    except Exception as e:
                        print(f"⚠️ Could not set torch threads: {e}")
                
                _MODEL_SINGLETON = m
                print(f"✅ Model loaded successfully: {m.get_sentence_embedding_dimension()}D")
                
            except Exception as e:
                print(f"❌ Failed to load SentenceTransformer from {use_path}")
                print(f"   Error: {type(e).__name__}: {e}")
                raise RuntimeError(f"Could not load sentence transformer: {e}")
        
        return _MODEL_SINGLETON


class POMSearchEngine:
    def __init__(self):
        self.sentence_model = None
        self.panel_embeddings = None
        self.faiss_index = None
        self.molecules_df = None
        self.embeddings = None
        
    def initialize(self):
        """Initialize all components of the search engine"""
        print("Initializing POM Search Engine...")
        
        try:
            # Load sentence transformer (cached)
            print("Loading sentence transformer model (cached/singleton)...")
            self.sentence_model = load_sentence_model_cached(SENTENCE_TRANSFORMER_MODEL)
            
            # Create panel word embeddings matrix
            print("Creating panel word embeddings...")
            self._create_panel_embeddings()
            
            # Load molecule data
            print("Loading molecule data...")
            self._load_molecule_data()
            
            # Load or build FAISS index (cached)
            print("Loading FAISS index...")
            self.faiss_index, self.embeddings = load_faiss_index_cached(EMBEDDINGS_FILE, MOLECULES_FILE)
            
            print("Initialization complete!")
            
        except Exception as e:
            print(f"❌ Error during initialization: {str(e)}")
            print("Falling back to basic functionality without FAISS search...")
            # Set fallback state
            self.faiss_index = None
            raise RuntimeError(f"Search engine initialization failed: {str(e)}")
    
    def _create_panel_embeddings(self):
        """Create embeddings for the 55 panel words"""
        panel_vecs = self.sentence_model.encode(PANEL_WORDS)  # Shape: (55, model_dim)
        self.panel_embeddings = panel_vecs.T  # Shape: (model_dim, 55)
        
    def _load_molecule_data(self):
        """Load molecule metadata only (embeddings handled by cached function)"""
        if not os.path.exists(MOLECULES_FILE):
            raise FileNotFoundError(f"Molecules file not found: {MOLECULES_FILE}")
            
        self.molecules_df = pd.read_csv(MOLECULES_FILE)
        print(f"Loaded {len(self.molecules_df)} molecules")
        
    def _load_data(self):
        """Load embeddings and molecule metadata"""
        if not os.path.exists(EMBEDDINGS_FILE):
            raise FileNotFoundError(f"Embeddings file not found: {EMBEDDINGS_FILE}")
        if not os.path.exists(MOLECULES_FILE):
            raise FileNotFoundError(f"Molecules file not found: {MOLECULES_FILE}")
            
        self.embeddings = np.load(EMBEDDINGS_FILE)
        self.molecules_df = pd.read_csv(MOLECULES_FILE)
        
        print(f"Loaded {len(self.embeddings)} embeddings with {self.embeddings.shape[1]} dimensions")
        print(f"Loaded {len(self.molecules_df)} molecules")
        
    def _load_faiss_index(self):
        """Load pre-built FAISS index from disk"""
        if not os.path.exists(FAISS_INDEX_FILE):
            return False
            
        try:
            print(f"Loading FAISS index from: {FAISS_INDEX_FILE}")
            self.faiss_index = faiss.read_index(FAISS_INDEX_FILE)
            print(f"SUCCESS: Loaded FAISS index with {self.faiss_index.ntotal} vectors")
            return True
        except Exception as e:
            print(f"WARNING: Failed to load FAISS index: {e}")
            return False
    
    def _load_or_build_faiss_index(self):
        """Load existing FAISS index or build new one if not found"""
        if self._load_faiss_index():
            return
            
        print("BUILDING: FAISS index not found, building new one...")
        print("TIP: Run 'python build_index.py' once to speed up future startups")
        self._build_faiss_index()
        
    def _build_faiss_index(self):
        """Build FAISS index for similarity search"""
        try:
            from tqdm import tqdm
            use_progress = True
        except ImportError:
            use_progress = False
            print("Install tqdm for progress bars: pip install tqdm")
        
        try:
            # CRITICAL FIX: Properly normalize embeddings for cosine similarity
            embeddings_norm = self.embeddings.astype('float32')
            
            # Debug original norms
            original_norms = [np.linalg.norm(emb) for emb in embeddings_norm[:3]]
            print(f"   Original DB embedding norms: {[f'{norm:.4f}' for norm in original_norms]}")
            
            if use_progress:
                print("L2 Normalizing database embeddings to unit norm...")
                with tqdm(total=len(embeddings_norm), desc="L2 Normalizing", unit="vectors") as pbar:
                    faiss.normalize_L2(embeddings_norm)
                    pbar.update(len(embeddings_norm))
            else:
                print("L2 Normalizing database embeddings...")
                faiss.normalize_L2(embeddings_norm)
            
            # Debug normalized norms
            normalized_norms = [np.linalg.norm(emb) for emb in embeddings_norm[:3]]
            print(f"   Normalized DB embedding norms: {[f'{norm:.4f}' for norm in normalized_norms]}")
            print(f"   SUCCESS: Database embeddings now have unit norm")
            
            # Create index (testing IndexFlatIP for direct cosine similarity)
            print(f"BUILDING: Creating FAISS index...")
            # Tested IndexFlatIP vs IndexFlatL2 for direct cosine on normalized vectors
            self.faiss_index = faiss.IndexFlatIP(POM_DIMENSIONS)  # Inner Product = Cosine similarity for normalized vectors
            
            # Add embeddings with optional progress bar
            if use_progress:
                batch_size = 1000
                total_batches = (len(embeddings_norm) + batch_size - 1) // batch_size
                
                with tqdm(total=total_batches, desc="Building index", unit="batch") as pbar:
                    for i in range(0, len(embeddings_norm), batch_size):
                        batch = embeddings_norm[i:i + batch_size]
                        self.faiss_index.add(batch)
                        pbar.update(1)
            else:
                print("Adding embeddings to index...")
                self.faiss_index.add(embeddings_norm)
                
            print(f"SUCCESS: FAISS index built with {self.faiss_index.ntotal} vectors")
            
        except Exception as e:
            print(f"❌ Error building FAISS index: {str(e)}")
            print("Continuing without FAISS search functionality...")
            self.faiss_index = None
        
    def prompt_to_rata(self, prompt: str) -> np.ndarray:
        """
        Convert free text prompt to 55-dimensional RATA vector
        Enhanced method that better maps text to odor descriptors
        
        Args:
            prompt: Natural language description of odor
            
        Returns:
            55-dimensional RATA vector (not normalized for better comparison)
        """
        if self.sentence_model is None or self.panel_embeddings is None:
            raise RuntimeError("Search engine not initialized. Call initialize() first.")
        
        # Method 1: Direct descriptor matching with weights
        rata_vec = np.zeros(55)
        
        # Clean and tokenize prompt (English only)
        prompt_lower = prompt.lower()
        prompt_tokens = prompt_lower.replace(',', ' ').replace(';', ' ').split()
        print(f"   Tokenized prompt: {prompt_tokens}")
        
        # Direct matching with panel words (extreme aggressive)
        # Key vanillin-critical words get maximum boost
        key_words = {'vanilla', 'sweet', 'creamy', 'fruity', 'buttery', 'dairy'}
        
        for i, panel_word in enumerate(PANEL_WORDS):
            # Exact matches get full weight
            if panel_word in prompt_tokens:
                if panel_word in key_words:
                    rata_vec[i] = 100.0  # Extreme signal for key words
                    print(f"   KEY Direct match: '{panel_word}' = {rata_vec[i]}")
                else:
                    rata_vec[i] = 50.0  # High signal for other words
                    print(f"   Direct match: '{panel_word}' = {rata_vec[i]}")
            # Partial matches get partial weight
            elif any(panel_word in token or token in panel_word for token in prompt_tokens):
                if panel_word in key_words:
                    rata_vec[i] = 60.0  # High partial signal for key words
                    print(f"   KEY Partial match: '{panel_word}' = {rata_vec[i]}")
                else:
                    rata_vec[i] = 30.0  # Moderate partial signal
                    print(f"   Partial match: '{panel_word}' = {rata_vec[i]}")
        
        print(f"   After Direct Matching: {np.sum(rata_vec > 0)} non-zero values, sum={np.sum(rata_vec):.2f}")
        
        # Method 2: Semantic similarity enhancement (cleanly separated cases)
        print(f"Applying semantic enhancement...")
        
        # Create clean prompt for semantic matching
        clean_prompt = ' '.join(prompt_tokens)
        print(f"   Clean prompt: '{clean_prompt}'")
        
        # Encode clean prompt and panel words
        prompt_vec = self.sentence_model.encode([clean_prompt])  # Shape: (1, model_dim)
        panel_vecs = self.sentence_model.encode(PANEL_WORDS)  # Shape: (55, model_dim)
        
        # Compute cosine similarities
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(prompt_vec, panel_vecs)[0]  # Shape: (55,)
        
        # Log semantic similarity stats
        print(f"   Semantic similarities - min: {similarities.min():.4f}, max: {similarities.max():.4f}, mean: {similarities.mean():.4f}")
        
        # Store pre-semantic state for comparison
        pre_semantic_rata = rata_vec.copy()
        pre_semantic_active = np.sum(rata_vec > 0.001)
        pre_semantic_sum = np.sum(rata_vec)
        pre_semantic_max = rata_vec.max() if pre_semantic_active > 0 else 0
        
        # CLEAN SEPARATION: Handle no direct matches vs. with direct matches
        if np.sum(rata_vec) == 0:
            print(f"   CASE 1: No direct matches - full semantic reconstruction")
            threshold = 0.02  # Ultra-low threshold for maximum activation
            boost_factor = 20.0  # Extreme boost
            
            rata_vec = np.where(similarities > threshold, similarities * boost_factor, 0)
            
            activated_dims = np.sum(similarities > threshold)
            print(f"   Threshold: {threshold}, Boost: {boost_factor}x")
            print(f"   Activated {activated_dims}/55 dimensions")
            print(f"   New sum: {np.sum(rata_vec):.2f}, max: {rata_vec.max():.3f}")
            
        else:
            print(f"   CASE 2: Has direct matches - selective semantic enhancement")
            threshold = 0.02  # Same threshold but different boost
            boost_factor = 15.0  # Moderate boost to avoid overwhelming direct signals
            
            # Create semantic boost vector
            semantic_boost = np.where(similarities > threshold, similarities * boost_factor, 0)
            
            # Keep stronger of direct or semantic signal per dimension
            rata_vec = np.maximum(rata_vec, semantic_boost)
            
            semantic_additions = np.sum(similarities > threshold)
            enhanced_dims = np.sum(rata_vec > semantic_boost)  # Dims where semantic was kept
            
            print(f"   Threshold: {threshold}, Boost: {boost_factor}x")
            print(f"   {semantic_additions} potential semantic additions")
            print(f"   {enhanced_dims} dimensions actually enhanced by semantic")
            print(f"   New sum: {np.sum(rata_vec):.2f}, max: {rata_vec.max():.3f}")
        
        # Post-semantic comparison logging
        post_semantic_active = np.sum(rata_vec > 0.001)
        post_semantic_sum = np.sum(rata_vec)
        post_semantic_max = rata_vec.max()
        
        print(f"   Enhancement impact:")
        print(f"   Active dims: {pre_semantic_active} -> {post_semantic_active} (+{post_semantic_active - pre_semantic_active})")
        print(f"   Sum: {pre_semantic_sum:.2f} -> {post_semantic_sum:.2f} ({post_semantic_sum/pre_semantic_sum*100 if pre_semantic_sum > 0 else 'inf'}%)")
        print(f"   Max: {pre_semantic_max:.3f} -> {post_semantic_max:.3f}")
        
        print(f"   Semantic Enhancement Complete")
        
        # SMART: Selective override with soft ramps for critical vanillin queries only
        critical_vanilla_phrases = ['vanilla', 'vanilla sweet', 'sweet vanilla', 'vanilla creamy', 'creamy vanilla', 
                                   'vanilla fruity', 'fruity vanilla', 'vanilla buttery', 'buttery vanilla']
        
        is_critical_vanilla_query = any(phrase in prompt.lower() for phrase in critical_vanilla_phrases)
        
        if is_critical_vanilla_query:
            print(f"   CRITICAL VANILLA: '{prompt}' detected - applying soft ramp override")
            
            # From database analysis: vanillin's strongest dimensions with importance weights
            vanillin_dims_weighted = {
                'vanilla': 1.0,    # Always maximum for vanilla queries
                'sweet': 0.9,      # Very high correlation
                'fruity': 0.8,     # High importance from DB analysis  
                'buttery': 0.7,    # High importance
                'green': 0.6,      # Medium-high
                'lemon': 0.5,      # Medium
                'sulfurous': 0.4   # Lower but critical
            }
            
            # SOFT RAMP: Apply weighted boosts based on importance
            base_boost = 6.0  # Moderate base multiplier
            overrides_applied = 0
            
            for word, importance_weight in vanillin_dims_weighted.items():
                if word in PANEL_WORDS:
                    idx = PANEL_WORDS.index(word)
                    old_value = rata_vec[idx]
                    
                    # Soft ramp: boost_factor varies by importance
                    actual_boost = base_boost * importance_weight
                    baseline = 0.03 * importance_weight  # Weighted baseline
                    
                    boosted_value = max(old_value, baseline) * actual_boost
                    
                    rata_vec[idx] = boosted_value
                    overrides_applied += 1
                    print(f"     RAMP '{word}': {old_value:.3f} -> {rata_vec[idx]:.3f} ({actual_boost:.1f}x, weight={importance_weight})")
            
            print(f"   Applied {overrides_applied} weighted ramp boosts for critical vanilla query")
            print(f"   Enhanced vanilla profile: sum={np.sum(rata_vec):.2f}, max={rata_vec.max():.3f}")
            
        elif any(indicator in prompt.lower() for indicator in ['sweet', 'creamy', 'dessert']):
            # Light boost for sweet-related but non-critical queries
            print(f"   LIGHT SWEET: Non-critical sweet query - applying light enhancement")
            sweet_dims = ['sweet', 'vanilla', 'dairy']  # Reduced set
            light_boost = 3.0  # Much lighter boost
            
            for word in sweet_dims:
                if word in PANEL_WORDS:
                    idx = PANEL_WORDS.index(word)
                    old_value = rata_vec[idx]
                    if old_value > 0.01:  # Only boost if already activated
                        rata_vec[idx] = old_value * light_boost
                        print(f"     LIGHT BOOST '{word}': {old_value:.3f} -> {rata_vec[idx]:.3f}")
            
            print(f"   Applied light sweet enhancement")
            
        else:
            # Standard dynamic enhancement for non-vanilla queries
            vanillin_critical_words = ['fruity', 'green', 'lemon', 'sulfurous', 'buttery']
            critical_enhancements = 0
            for word in vanillin_critical_words:
                if word in PANEL_WORDS:
                    idx = PANEL_WORDS.index(word)
                    word_similarity = similarities[idx]
                    
                    if word_similarity > 0.01:
                        dynamic_boost = word_similarity * 0.5
                        if rata_vec[idx] < dynamic_boost:
                            old_value = rata_vec[idx]
                            rata_vec[idx] = max(rata_vec[idx], dynamic_boost)
                            critical_enhancements += 1
            
            if critical_enhancements > 0:
                print(f"   Applied {critical_enhancements} standard enhancements")
        
        # Ensure we have some activation
        if np.sum(rata_vec) == 0:
            print(f"WARNING: No activation for prompt '{prompt}', using uniform distribution")
            rata_vec = np.ones(55) * 2.0  # Much higher base activation
        
        # Dynamic L2 normalization without fixed scaling
        if np.sum(rata_vec) > 0:
            # Only L2 normalize - let the boost ratios stay dynamic
            rata_norm = np.linalg.norm(rata_vec)
            if rata_norm > 0:
                rata_vec = rata_vec / rata_norm  # Pure L2 normalization only
                print(f"   L2 normalized to unit norm: {np.linalg.norm(rata_vec):.4f}")
        
        print(f"   Final RATA stats: min={rata_vec.min():.4f}, max={rata_vec.max():.4f}, std={rata_vec.std():.4f}")
        print(f"   Non-zero entries: {np.sum(rata_vec > 0.001)}/55")
        
        # Database alignment check
        if hasattr(self, 'embeddings') and self.embeddings is not None:
            sample_db = self.embeddings[0]  # First database vector
            dot_product = np.dot(sample_db, rata_vec)
            cosine_sim = dot_product / (np.linalg.norm(sample_db) * np.linalg.norm(rata_vec))
            print(f"   Sample DB alignment: dot={dot_product:.4f}, cosine={cosine_sim:.4f}")
        
        print(f"   RATA Generation Complete\n")
        
        return rata_vec
        
    def search_molecules(self, rata_vector: np.ndarray, k: int = DEFAULT_K) -> pd.DataFrame:
        """
        Search for similar molecules using RATA vector
        
        Args:
            rata_vector: 55-dimensional RATA vector
            k: Number of results to return
            
        Returns:
            DataFrame with molecule matches and similarity scores
        """
        if self.faiss_index is None:
            raise RuntimeError("FAISS index not built. Call initialize() first.")
            
        # DEBUG: Print query vector details
        print(f"FAISS SEARCH DEBUG:")
        print(f"   Input RATA vector shape: {rata_vector.shape}")
        print(f"   Input RATA vector range: [{rata_vector.min():.4f}, {rata_vector.max():.4f}]")
        print(f"   Input RATA vector norm: {np.linalg.norm(rata_vector):.4f}")
        print(f"   Input RATA vector sum: {rata_vector.sum():.4f}")
        
        # Prepare query vector for FAISS search
        query_vector = rata_vector.astype('float32').reshape(1, -1)
        
        # DEBUG: Before normalization
        print(f"   Query vector before norm: {np.linalg.norm(query_vector):.4f}")
        
        faiss.normalize_L2(query_vector)
        
        # DEBUG: After normalization
        print(f"   Query vector after norm: {np.linalg.norm(query_vector):.4f}")
        print(f"   Query vector sample values: {query_vector[0][:5]}")
        
        # Search using Inner Product (direct cosine similarity for normalized vectors)
        distances, indices = self.faiss_index.search(query_vector, k)
        
        # DEBUG: FAISS results with validation
        print(f"   FAISS scores (inner products): {distances[0]}")
        print(f"   Score range: [{distances[0].min():.4f}, {distances[0].max():.4f}]")
        print(f"   FAISS indices: {indices[0]}")
        print(f"   Index total vectors: {self.faiss_index.ntotal}")
        
        # Validate score range for normalized vectors (should be [-1,1], but typically [0,1])
        if distances[0].max() > 1.1 or distances[0].min() < -1.1:
            print(f"   WARNING: Inner product range outside [-1,1] - possible normalization issue!")
        
        # For IndexFlatIP, distances are already cosine similarities (inner products of normalized vectors)
        similarities = distances[0]  # Direct cosine similarities from inner product
        
        # Alternative direct cosine calculation for validation
        if hasattr(self, 'embeddings') and len(indices[0]) > 0:
            # Direct cosine similarity calculation as validation
            query_norm = query_vector[0] / np.linalg.norm(query_vector[0])
            db_sample = self.embeddings[indices[0][0]]
            db_sample_norm = db_sample / np.linalg.norm(db_sample)
            direct_cosine = np.dot(query_norm, db_sample_norm)
            faiss_cosine = similarities[0]
            print(f"   Validation - Direct cosine: {direct_cosine:.4f}, FAISS cosine: {faiss_cosine:.4f}")
            print(f"   Difference: {abs(direct_cosine - faiss_cosine):.6f}")
        
        # DEBUG: Similarity conversion
        print(f"   Raw similarities: {similarities}")
        
        # CRITICAL FIX: Don't clip similarities - this was causing the 1.0000 bug!
        # similarities = np.clip(similarities, 0, 1)  # REMOVED - this destroyed the scores
        # For IndexFlatIP with normalized vectors, similarities should be in [-1,1] naturally
        
        print(f"   Final similarities: {similarities}")
        print(f"DEBUG END\n")
        
        # Get results
        results = self.molecules_df.iloc[indices[0]].copy()
        results['similarity_score'] = similarities
        results = results.sort_values('similarity_score', ascending=False)
        
        # Add meaningful names to molecules
        from visualization import add_molecule_names_to_dataframe
        results = add_molecule_names_to_dataframe(results)
        
        return results
        
    def search_by_prompt(self, prompt: str, k: int = DEFAULT_K) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        End-to-end search: prompt -> RATA -> molecule matches
        
        Args:
            prompt: Natural language odor description
            k: Number of results to return
            
        Returns:
            Tuple of (rata_vector, results_dataframe)
        """
        rata_vector = self.prompt_to_rata(prompt)
        results = self.search_molecules(rata_vector, k)
        return rata_vector, results


# Global search engine instance
_search_engine = None

def get_search_engine() -> POMSearchEngine:
    """Get the global search engine instance"""
    global _search_engine
    if _search_engine is None:
        _search_engine = POMSearchEngine()
        _search_engine.initialize()
    return _search_engine


def search_odor(prompt: str, k: int = DEFAULT_K) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Convenience function for odor search
    
    Args:
        prompt: Natural language odor description  
        k: Number of results to return
        
    Returns:
        Tuple of (rata_vector, results_dataframe)
    """
    try:
        engine = get_search_engine()
        return engine.search_by_prompt(prompt, k)
    except Exception as e:
        print(f"❌ Search failed: {str(e)}")
        # Return empty/fallback results
        empty_rata = np.zeros(55)
        empty_df = pd.DataFrame({
            'name': [],
            'smiles': [],
            'similarity_score': []
        })
        raise RuntimeError(f"Odor search failed: {str(e)}")


if __name__ == "__main__":
    # Test the search functionality
    try:
        rata, results = search_odor("frisch, limettig, etwas grasig", k=3)
        print(f"RATA vector shape: {rata.shape}")
        print(f"Top results:\n{results[['name', 'smiles', 'similarity_score']].head()}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the data files are in the 'data/' directory:")
        print("- data/embeddings.npy")
        print("- data/molecules.csv")