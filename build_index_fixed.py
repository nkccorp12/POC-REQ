"""
FIXED FAISS Index Builder
Rebuilds the FAISS index with proper normalization and debugging
Fixes the 1.0000 similarity score bug
"""

import numpy as np
import pandas as pd
import faiss
import os
from datetime import datetime

from config import EMBEDDINGS_FILE, MOLECULES_FILE, FAISS_INDEX_FILE, POM_DIMENSIONS


def build_faiss_index_fixed():
    """Build FAISS index with comprehensive debugging and validation"""
    print("BUILDING FIXED FAISS INDEX")
    print("=" * 50)
    print("This will fix the 1.0000 similarity score bug")
    print()
    
    # Step 1: Load and validate data
    print("📊 STEP 1: Loading and validating data...")
    
    if not os.path.exists(EMBEDDINGS_FILE):
        print(f"❌ ERROR: Embeddings file not found: {EMBEDDINGS_FILE}")
        return False
    
    if not os.path.exists(MOLECULES_FILE):
        print(f"❌ ERROR: Molecules file not found: {MOLECULES_FILE}")
        return False
    
    # Load embeddings
    embeddings = np.load(EMBEDDINGS_FILE)
    print(f"✅ Loaded embeddings: {embeddings.shape}")
    print(f"   Data type: {embeddings.dtype}")
    print(f"   Memory usage: {embeddings.nbytes / 1024 / 1024:.1f} MB")
    
    # Load molecules
    molecules_df = pd.read_csv(MOLECULES_FILE)
    print(f"✅ Loaded molecules: {len(molecules_df)} entries")
    
    # Validate alignment
    if len(embeddings) != len(molecules_df):
        print(f"❌ ERROR: Size mismatch - embeddings: {len(embeddings)}, molecules: {len(molecules_df)}")
        return False
    
    # Validate dimensions
    if embeddings.shape[1] != POM_DIMENSIONS:
        print(f"❌ ERROR: Wrong dimensions - got {embeddings.shape[1]}, expected {POM_DIMENSIONS}")
        return False
    
    print("✅ Data validation passed")
    
    # Step 2: Analyze embedding statistics
    print("\\n📈 STEP 2: Analyzing embedding statistics...")
    
    # Check for problematic values
    nan_count = np.sum(np.isnan(embeddings))
    inf_count = np.sum(np.isinf(embeddings))
    zero_count = np.sum(embeddings == 0)
    
    print(f"   NaN values: {nan_count}")
    print(f"   Infinite values: {inf_count}")
    print(f"   Zero values: {zero_count}")
    
    if nan_count > 0 or inf_count > 0:
        print("❌ ERROR: Embeddings contain NaN or infinite values!")
        return False
    
    # Analyze value ranges
    print(f"   Value range: [{embeddings.min():.4f}, {embeddings.max():.4f}]")
    print(f"   Mean: {embeddings.mean():.4f}, Std: {embeddings.std():.4f}")
    
    # Analyze norms BEFORE normalization
    original_norms = np.linalg.norm(embeddings, axis=1)
    print(f"   Original norms - min: {original_norms.min():.4f}, max: {original_norms.max():.4f}, mean: {original_norms.mean():.4f}")
    
    # Check if already normalized
    is_normalized = np.allclose(original_norms, 1.0, atol=1e-3)
    print(f"   Already normalized: {is_normalized}")
    
    # Step 3: Prepare embeddings for FAISS
    print("\\n🔧 STEP 3: Preparing embeddings for FAISS...")
    
    # Convert to float32 (required by FAISS)
    embeddings_float32 = embeddings.astype('float32')
    print(f"✅ Converted to float32")
    
    # Normalize embeddings for cosine similarity
    print("🔄 Normalizing embeddings...")
    faiss.normalize_L2(embeddings_float32)
    
    # Verify normalization
    normalized_norms = np.linalg.norm(embeddings_float32, axis=1)
    print(f"   Post-normalization norms - min: {normalized_norms.min():.4f}, max: {normalized_norms.max():.4f}")
    
    if not np.allclose(normalized_norms, 1.0, atol=1e-3):
        print("❌ ERROR: Normalization failed!")
        return False
    
    print("✅ Normalization successful - all vectors have unit norm")
    
    # Step 4: Create and populate FAISS index
    print("\\n🏗️  STEP 4: Creating FAISS index...")
    
    # Create IndexFlatIP for cosine similarity (inner product of normalized vectors)
    index = faiss.IndexFlatIP(POM_DIMENSIONS)
    print(f"✅ Created IndexFlatIP with {POM_DIMENSIONS} dimensions")
    
    # Add embeddings to index
    print("📥 Adding embeddings to index...")
    index.add(embeddings_float32)
    print(f"✅ Added {index.ntotal} vectors to index")
    
    # Step 5: Validate the index with test searches
    print("\\n🧪 STEP 5: Validating index with test searches...")
    
    # Test 1: Random query
    test_query = np.random.random((1, POM_DIMENSIONS)).astype('float32')
    faiss.normalize_L2(test_query)
    
    distances, indices = index.search(test_query, 5)
    print(f"✅ Random test - distances: {distances[0]}, indices: {indices[0]}")
    
    # Validate distance range
    if np.all(distances[0] == 1.0):
        print("❌ CRITICAL: All distances are 1.0 - the bug persists!")
        return False
    
    if distances[0].max() > 1.1 or distances[0].min() < -1.1:
        print(f"⚠️  WARNING: Distances outside [-1,1] range: {distances[0]}")
    
    # Test 2: Vanillin-like query
    vanillin_query = np.zeros((1, POM_DIMENSIONS), dtype='float32')
    # Assuming vanilla is at index 134 (need to verify with PANEL_WORDS)
    from config import PANEL_WORDS
    if 'vanilla' in PANEL_WORDS:
        vanilla_idx = PANEL_WORDS.index('vanilla')
        vanillin_query[0, vanilla_idx] = 1.0
        faiss.normalize_L2(vanillin_query)
        
        distances, indices = index.search(vanillin_query, 10)
        print(f"✅ Vanilla test - top distances: {distances[0][:3]}")
        
        # Check if vanillin appears in results
        vanillin_smiles = "COc1cc(C=O)ccc1O"
        smiles_col = 'smiles' if 'smiles' in molecules_df.columns else 'nonStereoSMILES'
        
        vanillin_found = False
        for i, idx in enumerate(indices[0][:10]):
            molecule_smiles = molecules_df.iloc[idx][smiles_col]
            if molecule_smiles == vanillin_smiles:
                vanillin_found = True
                print(f"🎯 Vanillin found at position {i+1} with distance {distances[0][i]:.4f}")
                break
        
        if not vanillin_found:
            print("⚠️  Vanillin not found in top 10 for vanilla query")
    
    # Test 3: Self-similarity test
    # Take first embedding and search for itself
    self_query = embeddings_float32[0:1]  # First embedding as query
    distances, indices = index.search(self_query, 3)
    
    print(f"✅ Self-similarity test:")
    print(f"   Query is embedding #{0}")
    print(f"   Top match: embedding #{indices[0][0]} with distance {distances[0][0]:.6f}")
    print(f"   Expected: embedding #0 with distance ~1.0")
    
    if indices[0][0] != 0:
        print("⚠️  WARNING: Self-similarity test failed - query didn't match itself!")
    
    if abs(distances[0][0] - 1.0) > 0.001:
        print(f"⚠️  WARNING: Self-similarity distance should be ~1.0, got {distances[0][0]:.6f}")
    
    # Step 6: Save the index
    print("\\n💾 STEP 6: Saving FAISS index...")
    
    # Backup old index if it exists
    if os.path.exists(FAISS_INDEX_FILE):
        backup_file = f"{FAISS_INDEX_FILE}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.rename(FAISS_INDEX_FILE, backup_file)
        print(f"📦 Backed up old index to: {backup_file}")
    
    # Save new index
    faiss.write_index(index, FAISS_INDEX_FILE)
    print(f"✅ Saved new index to: {FAISS_INDEX_FILE}")
    
    # Verify saved index
    try:
        test_index = faiss.read_index(FAISS_INDEX_FILE)
        if test_index.ntotal == index.ntotal:
            print("✅ Index verification successful")
        else:
            print(f"❌ Index verification failed - expected {index.ntotal}, got {test_index.ntotal}")
            return False
    except Exception as e:
        print(f"❌ Index verification failed: {e}")
        return False
    
    # Final summary
    print("\\n" + "="*60)
    print("✅ FAISS INDEX BUILD COMPLETE")
    print("="*60)
    print(f"📊 Index Statistics:")
    print(f"   Total vectors: {index.ntotal}")
    print(f"   Dimensions: {POM_DIMENSIONS}")
    print(f"   Index type: IndexFlatIP (cosine similarity)")
    print(f"   File size: {os.path.getsize(FAISS_INDEX_FILE) / 1024 / 1024:.1f} MB")
    print()
    print("🎯 The 1.0000 similarity bug should now be fixed!")
    print("   Test with: python -c \\"from pom_search import search_odor; print(search_odor('vanilla', 3))\\"")
    
    return True


def main():
    """Build the fixed FAISS index"""
    success = build_faiss_index_fixed()
    
    if success:
        print("\\n🎉 SUCCESS: FAISS index built successfully!")
        print("   The search engine should now return realistic similarity scores")
    else:
        print("\\n💥 FAILED: Could not build FAISS index")
        print("   Please check the error messages above and fix the issues")


if __name__ == "__main__":
    main()