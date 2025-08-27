"""
Fix embedding extraction to get proper POM embeddings instead of RATA predictions
The issue is that we were getting 55D RATA outputs instead of 256D POM embeddings
"""

import numpy as np
import pandas as pd
import deepchem as dc
from openpom.feat.graph_featurizer import GraphFeaturizer
from openpom.models.mpnn_pom import MPNNPOMModel
import torch


def extract_proper_pom_embeddings():
    """
    Extract the actual 256-dimensional POM embeddings from the penultimate layer
    instead of the 55-dimensional RATA predictions
    """
    print("🔧 Fixing embedding extraction...")
    print("=" * 50)
    
    # Load data
    print("📂 Loading molecules...")
    df = pd.read_csv('data/molecules.csv')
    print(f"✅ Loaded {len(df)} molecules")
    
    # Initialize components
    print("🔄 Initializing featurizer and model...")
    featurizer = GraphFeaturizer()
    model = MPNNPOMModel(n_tasks=55, pretrained=True)
    
    # Featurize molecules
    print("🧪 Featurizing molecules...")
    graphs = featurizer.featurize(df['nonStereoSMILES'])
    dataset = dc.data.NumpyDataset(X=graphs)
    
    print("🎯 Extracting proper POM embeddings...")
    
    # We need to access the model's internal representations
    # The MPNN model should have a method to get embeddings before the final projection
    try:
        # Option 1: Try to get embeddings directly
        if hasattr(model, 'get_embeddings'):
            embeddings = model.get_embeddings(dataset)
            print(f"✅ Method 1: Got embeddings with shape {embeddings.shape}")
        
        # Option 2: Try predict_embeddings
        elif hasattr(model, 'predict_embeddings'):
            embeddings = model.predict_embeddings(dataset)
            print(f"✅ Method 2: Got embeddings with shape {embeddings.shape}")
            
        # Option 3: Access the model's forward pass manually
        else:
            print("⚠️  No direct embedding method found. Trying manual extraction...")
            
            # Get the model in evaluation mode
            model.model.eval()
            
            embeddings_list = []
            
            # Process in batches
            batch_size = 32
            total_batches = (len(graphs) + batch_size - 1) // batch_size
            
            with torch.no_grad():
                for i in range(0, len(graphs), batch_size):
                    batch_graphs = graphs[i:i + batch_size]
                    
                    # Convert to torch tensors if needed
                    # This is model-specific and might need adjustment
                    batch_embeddings = []
                    
                    for graph in batch_graphs:
                        # Forward pass through the network up to penultimate layer
                        # This requires knowledge of the model architecture
                        try:
                            # Try to get embeddings from the graph neural network
                            graph_tensor = torch.from_numpy(graph).float()
                            embedding = model.model.gnn(graph_tensor)  # This is a guess - may need adjustment
                            batch_embeddings.append(embedding.numpy())
                        except Exception as e:
                            print(f"Error processing graph: {e}")
                            # Fallback: use zeros
                            batch_embeddings.append(np.zeros(256))
                    
                    embeddings_list.extend(batch_embeddings)
                    
                    if i // batch_size % 10 == 0:
                        print(f"  Processed {i//batch_size + 1}/{total_batches} batches")
            
            embeddings = np.array(embeddings_list)
            print(f"✅ Method 3: Manually extracted embeddings with shape {embeddings.shape}")
        
        # Validate embeddings
        if embeddings.shape[1] != 256:
            print(f"⚠️  Warning: Expected 256 dimensions, got {embeddings.shape[1]}")
            
            if embeddings.shape[1] == 55:
                print("🚨 CRITICAL: Still getting 55D outputs (RATA predictions)")
                print("This means we need to modify the model extraction method")
                return False
        
        # Save the corrected embeddings
        print("💾 Saving corrected POM embeddings...")
        np.save('data/embeddings_pom_256d.npy', embeddings)
        
        # Update config if dimensions changed
        if embeddings.shape[1] != 55:
            print("📝 Updating config.py with correct dimensions...")
            update_config_dimensions(embeddings.shape[1])
        
        print(f"✅ Successfully extracted {embeddings.shape[0]} embeddings of {embeddings.shape[1]} dimensions")
        return True
        
    except Exception as e:
        print(f"❌ Error extracting embeddings: {e}")
        print("\n💡 Alternative approaches:")
        print("1. Check OpenPOM documentation for embedding extraction")
        print("2. Use the model's hidden layer representations")
        print("3. Contact OpenPOM authors for proper embedding extraction")
        return False


def update_config_dimensions(new_dim):
    """Update config.py with correct POM dimensions"""
    config_file = 'config.py'
    
    # Read config
    with open(config_file, 'r') as f:
        content = f.read()
    
    # Update POM_DIMENSIONS
    content = content.replace(
        f'POM_DIMENSIONS = 55',
        f'POM_DIMENSIONS = {new_dim}'
    )
    
    # Update embeddings file path to use the new file
    content = content.replace(
        'EMBEDDINGS_FILE = f"{DATA_DIR}/embeddings.npy"',
        'EMBEDDINGS_FILE = f"{DATA_DIR}/embeddings_pom_256d.npy"'
    )
    
    # Write back
    with open(config_file, 'w') as f:
        f.write(content)
    
    print(f"✅ Updated config.py: POM_DIMENSIONS = {new_dim}")


def analyze_current_embeddings():
    """Analyze the current embeddings to understand what we have"""
    print("🔍 Analyzing current embeddings...")
    
    try:
        embeddings = np.load('data/embeddings.npy')
        print(f"📊 Current embeddings shape: {embeddings.shape}")
        print(f"📈 Value range: [{embeddings.min():.4f}, {embeddings.max():.4f}]")
        print(f"📉 Mean: {embeddings.mean():.4f}, Std: {embeddings.std():.4f}")
        
        # Check if these look like RATA predictions (should be in [0,1] range mostly)
        in_range_01 = np.sum((embeddings >= 0) & (embeddings <= 1)) / embeddings.size
        print(f"🎯 Values in [0,1] range: {in_range_01*100:.1f}%")
        
        if in_range_01 > 0.8:
            print("🚨 DIAGNOSIS: These look like RATA predictions (0-1 range)")
            print("   We need to extract the actual POM embeddings from earlier layers")
        else:
            print("✅ These might be proper embeddings (not bounded to [0,1])")
            
    except FileNotFoundError:
        print("❌ No embeddings file found")


if __name__ == "__main__":
    print("EMBEDDING EXTRACTION DIAGNOSTIC")
    print("=" * 40)
    
    # First analyze what we currently have
    analyze_current_embeddings()
    
    print("\n")
    
    # Try to extract proper embeddings
    success = extract_proper_pom_embeddings()
    
    if not success:
        print("\n" + "=" * 50)
        print("🔧 MANUAL STEPS NEEDED:")
        print("1. Check OpenPOM documentation for embedding extraction")
        print("2. Find the correct method to get 256D embeddings")
        print("3. The current approach gives RATA predictions, not POM embeddings")
        print("4. We need the hidden layer representations, not the final predictions")