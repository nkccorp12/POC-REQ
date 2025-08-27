"""
Build and save FAISS index for faster app startup
Run this script once after generating embeddings to create a persistent FAISS index
"""

import numpy as np
import faiss
import os
from tqdm import tqdm
from config import EMBEDDINGS_FILE, FAISS_INDEX_FILE, POM_DIMENSIONS


def build_and_save_faiss_index():
    """Build FAISS index from embeddings and save to disk"""
    
    print("🔧 Building FAISS Index...")
    print("=" * 50)
    
    # Check if embeddings exist
    if not os.path.exists(EMBEDDINGS_FILE):
        raise FileNotFoundError(f"Embeddings file not found: {EMBEDDINGS_FILE}")
    
    print(f"📂 Loading embeddings from: {EMBEDDINGS_FILE}")
    embeddings = np.load(EMBEDDINGS_FILE)
    print(f"✅ Loaded {len(embeddings)} embeddings with {embeddings.shape[1]} dimensions")
    
    # CRITICAL FIX: Properly normalize embeddings for cosine similarity
    print("🔄 Normalizing embeddings to unit norm...")
    embeddings_norm = embeddings.astype('float32')
    
    # Debug: Check original norms
    original_norms = [np.linalg.norm(emb) for emb in embeddings_norm[:5]]
    print(f"   Original embedding norms (first 5): {[f'{norm:.4f}' for norm in original_norms]}")
    
    # Progress bar for normalization
    with tqdm(total=len(embeddings_norm), desc="L2 Normalizing", unit="vectors") as pbar:
        faiss.normalize_L2(embeddings_norm)
        pbar.update(len(embeddings_norm))
    
    # Debug: Check normalized norms
    normalized_norms = [np.linalg.norm(emb) for emb in embeddings_norm[:5]]
    print(f"   Normalized embedding norms (first 5): {[f'{norm:.4f}' for norm in normalized_norms]}")
    print(f"   ✅ All embeddings now have unit norm (should be ~1.0000)")
    
    # Create FAISS index
    print(f"🏗️  Creating FAISS index (dimensions: {POM_DIMENSIONS})...")
    index = faiss.IndexFlatL2(POM_DIMENSIONS)  # L2 distance for normalized vectors = cosine similarity
    
    # Add embeddings with progress bar
    batch_size = 1000
    total_batches = (len(embeddings_norm) + batch_size - 1) // batch_size
    
    with tqdm(total=total_batches, desc="Building index", unit="batch") as pbar:
        for i in range(0, len(embeddings_norm), batch_size):
            batch = embeddings_norm[i:i + batch_size]
            index.add(batch)
            pbar.update(1)
    
    print(f"✅ Index built successfully! Total vectors: {index.ntotal}")
    
    # Save index to disk
    print(f"💾 Saving index to: {FAISS_INDEX_FILE}")
    
    # Create data directory if it doesn't exist
    os.makedirs(os.path.dirname(FAISS_INDEX_FILE), exist_ok=True)
    
    # Save with progress indication
    with tqdm(total=1, desc="Saving index", unit="file") as pbar:
        faiss.write_index(index, FAISS_INDEX_FILE)
        pbar.update(1)
    
    # Verify saved index
    print("🔍 Verifying saved index...")
    test_index = faiss.read_index(FAISS_INDEX_FILE)
    print(f"✅ Verification successful! Saved index has {test_index.ntotal} vectors")
    
    print("\n" + "=" * 50)
    print("🎉 FAISS Index build complete!")
    print(f"📁 Index saved to: {FAISS_INDEX_FILE}")
    print("🚀 Your app will now start much faster!")
    

if __name__ == "__main__":
    try:
        build_and_save_faiss_index()
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure to run 'python data/OpenPOM.py' first to generate embeddings")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()