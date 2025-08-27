"""
Visualization components for the Describe-to-Radar POC
Includes radar charts and molecule structure rendering
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from rdkit import Chem
from rdkit.Chem import Draw
from PIL import Image
import io
from typing import List, Optional, Tuple

from config import PANEL_WORDS, RADAR_FIGSIZE, MOLECULE_IMAGE_SIZE


def create_radar_chart_matplotlib(rata_vector: np.ndarray, 
                                title: str = "Odor Profile",
                                figsize: Tuple[int, int] = RADAR_FIGSIZE) -> plt.Figure:
    """
    Create radar chart using matplotlib
    
    Args:
        rata_vector: 55-dimensional RATA vector
        title: Chart title
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    # Prepare data
    N = len(PANEL_WORDS)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    
    # Close the plot
    values = rata_vector.tolist()
    values += values[:1]  # Complete the circle
    angles += angles[:1]
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    
    # Plot data
    ax.plot(angles, values, 'o-', linewidth=2, label='Odor Profile')
    ax.fill(angles, values, alpha=0.25)
    
    # Customize
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(PANEL_WORDS, fontsize=8)
    ax.set_ylim(0, max(rata_vector) * 1.1 if max(rata_vector) > 0 else 1)
    ax.set_title(title, size=16, fontweight='bold', pad=20)
    ax.grid(True)
    
    plt.tight_layout()
    return fig


def create_radar_chart_plotly(rata_vector: np.ndarray, 
                            title: str = "Odor Profile") -> go.Figure:
    """
    Create interactive radar chart using plotly with modern styling
    
    Args:
        rata_vector: 55-dimensional RATA vector
        title: Chart title
        
    Returns:
        plotly Figure object
    """
    # Create DataFrame for px.line_polar
    df = pd.DataFrame({
        'theta': PANEL_WORDS,
        'r': rata_vector,
        'hover_text': [f"{word}: {val:.3f}" for word, val in zip(PANEL_WORDS, rata_vector)]
    })
    
    # Create polar line chart with fill
    fig = px.line_polar(
        df, 
        r='r', 
        theta='theta',
        line_close=True,
        hover_data={'hover_text': True, 'r': ':.3f'},
        title=title
    )
    
    # Update traces for smooth green styling
    fig.update_traces(
        mode='lines',  # Ensures continuous line instead of bars/markers
        fill='toself',
        line=dict(
            color='green',        # Green contour
            width=2,
            shape='spline',       # Spline instead of straight segments
            smoothing=1.3         # Degree of smoothing (0-1.3)
        ),
        fillcolor='rgba(0, 128, 0, 0.4)',  # Green semi-transparent fill
        hovertemplate='<b>%{theta}</b><br>Intensity: %{r:.3f}<extra></extra>'
    )
    
    # Update layout for better appearance
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(rata_vector) * 1.2 if max(rata_vector) > 0 else 1],
                gridcolor='rgba(0,0,0,0.1)',
                tickfont=dict(size=10, color='rgba(0,0,0,0.6)')
            ),
            angularaxis=dict(
                tickfont=dict(size=9, color='rgba(0,0,0,0.8)'),
                rotation=90,
                direction="counterclockwise",
                gridcolor='rgba(0,0,0,0.1)'
            ),
            bgcolor='rgba(255,255,255,0.8)'
        ),
        showlegend=False,
        title=dict(
            text=title,
            x=0.5,
            font=dict(size=18, color='rgba(0,0,0,0.8)'),
            y=0.95
        ),
        width=700,
        height=700,
        margin=dict(l=80, r=80, t=80, b=80),
        paper_bgcolor='rgba(255,255,255,0.95)',
        plot_bgcolor='rgba(255,255,255,0.95)'
    )
    
    return fig


def render_molecule_structure(smiles: str, 
                            size: Tuple[int, int] = MOLECULE_IMAGE_SIZE) -> Optional[Image.Image]:
    """
    Render molecule structure from SMILES string
    
    Args:
        smiles: SMILES representation of molecule
        size: Image size (width, height)
        
    Returns:
        PIL Image object or None if invalid SMILES
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
            
        # Generate image
        img = Draw.MolToImage(mol, size=size)
        return img
        
    except Exception as e:
        print(f"Error rendering molecule {smiles}: {e}")
        return None


def create_molecule_grid(molecules_df: pd.DataFrame, 
                        max_molecules: int = 9,
                        mols_per_row: int = 3) -> Optional[Image.Image]:
    """
    Create a grid of molecule structures
    
    Args:
        molecules_df: DataFrame with 'smiles' column
        max_molecules: Maximum number of molecules to display
        mols_per_row: Molecules per row in grid
        
    Returns:
        PIL Image with molecule grid or None if error
    """
    try:
        # Limit molecules
        df = molecules_df.head(max_molecules).copy()
        
        # Convert SMILES to molecule objects
        mols = []
        labels = []
        
        for idx, row in df.iterrows():
            # Get SMILES from either column name
            smiles = row.get('smiles', row.get('nonStereoSMILES', ''))
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                mols.append(mol)
                # Create label with molecule name and score
                name = row.get('name', f'Molecule {idx}')
                score = row.get('similarity_score', 0.0)
                labels.append(f"{name}\nSimilarity: {score:.3f}")
        
        if not mols:
            return None
            
        # Create grid image
        img = Draw.MolsToGridImage(
            mols, 
            molsPerRow=mols_per_row,
            subImgSize=MOLECULE_IMAGE_SIZE,
            legends=labels
        )
        
        return img
        
    except Exception as e:
        print(f"Error creating molecule grid: {e}")
        return None


def display_search_results(rata_vector: np.ndarray, 
                         results_df: pd.DataFrame,
                         prompt: str = "") -> None:
    """
    Display complete search results with radar chart and molecules
    
    Args:
        rata_vector: 55-dimensional RATA vector
        results_df: Search results DataFrame
        prompt: Original search prompt
    """
    print(f"\n{'='*60}")
    print(f"SEARCH RESULTS FOR: '{prompt}'")
    print(f"{'='*60}")
    
    # Display radar chart
    fig = create_radar_chart_matplotlib(rata_vector, f"Odor Profile: {prompt}")
    plt.show()
    
    # Display top matches
    print(f"\nTop {len(results_df)} molecule matches:")
    print("-" * 40)
    
    for idx, row in results_df.iterrows():
        print(f"{idx + 1}. {row.get('name', 'Unknown')}")
        print(f"   SMILES: {row['smiles']}")
        print(f"   Similarity: {row['similarity_score']:.4f}")
        print()
    
    # Create and display molecule grid
    mol_grid = create_molecule_grid(results_df)
    if mol_grid:
        mol_grid.show()
    else:
        print("Could not generate molecule structures")


def get_top_panel_words(rata_vector: np.ndarray, top_n: int = 10) -> List[Tuple[str, float]]:
    """
    Get the top N panel words by RATA score
    
    Args:
        rata_vector: 55-dimensional RATA vector
        top_n: Number of top words to return
        
    Returns:
        List of (word, score) tuples sorted by score
    """
    word_scores = list(zip(PANEL_WORDS, rata_vector))
    word_scores.sort(key=lambda x: x[1], reverse=True)
    return word_scores[:top_n]


def generate_molecule_name(smiles: str, descriptors: str = "", similarity_score: float = 0.0) -> str:
    """
    Generate a meaningful name for a molecule based on SMILES and descriptors
    
    Args:
        smiles: SMILES representation of the molecule
        descriptors: Semicolon-separated odor descriptors
        similarity_score: Similarity score for fallback naming
        
    Returns:
        Generated molecule name
    """
    try:
        # First try to get molecular formula from SMILES
        from rdkit import Chem
        from rdkit.Chem import Descriptors
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            # Get molecular formula
            formula = Chem.rdMolDescriptors.CalcMolFormula(mol)
            
            # Get molecular weight
            mw = Descriptors.MolWt(mol)
            
            # Try to get a common name from descriptors
            if descriptors:
                desc_list = [d.strip().title() for d in descriptors.split(';') if d.strip()]
                if desc_list:
                    # Use top 2 descriptors for name
                    main_descriptors = desc_list[:2]
                    descriptor_name = "-".join(main_descriptors)
                    return f"{descriptor_name} ({formula})"
            
            # Fallback to formula with MW
            return f"Compound {formula} (MW: {mw:.0f})"
            
    except Exception as e:
        print(f"Warning: Could not generate name for {smiles}: {e}")
    
    # Ultimate fallback - use descriptors only
    if descriptors:
        desc_list = [d.strip().title() for d in descriptors.split(';') if d.strip()]
        if desc_list:
            return f"{desc_list[0]} Compound"
    
    # Last resort
    return f"Unknown Compound (Score: {similarity_score:.3f})"


def add_molecule_names_to_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add meaningful names to a molecule dataframe
    
    Args:
        df: DataFrame with SMILES and optionally descriptors
        
    Returns:
        DataFrame with added 'name' column
    """
    df = df.copy()
    
    # Get SMILES column name
    smiles_col = 'smiles' if 'smiles' in df.columns else 'nonStereoSMILES'
    
    names = []
    for idx, row in df.iterrows():
        smiles = row.get(smiles_col, '')
        descriptors = row.get('descriptors', '')
        similarity_score = row.get('similarity_score', 0.0)
        
        name = generate_molecule_name(smiles, descriptors, similarity_score)
        names.append(name)
    
    df['name'] = names
    return df


if __name__ == "__main__":
    # Test visualization functions
    import numpy as np
    
    # Create test RATA vector
    test_rata = np.random.rand(55)
    test_rata = test_rata / np.linalg.norm(test_rata)
    
    # Test radar chart
    fig_mpl = create_radar_chart_matplotlib(test_rata, "Test Profile")
    plt.show()
    
    fig_plotly = create_radar_chart_plotly(test_rata, "Test Profile (Interactive)")
    fig_plotly.show()
    
    # Test top words
    top_words = get_top_panel_words(test_rata, 5)
    print("Top 5 odor descriptors:")
    for word, score in top_words:
        print(f"  {word}: {score:.3f}")
    
    # Test molecule rendering
    test_smiles = "CCO"  # Ethanol
    mol_img = render_molecule_structure(test_smiles)
    if mol_img:
        mol_img.show()
        print("Molecule structure rendered successfully")
    else:
        print("Failed to render molecule structure")