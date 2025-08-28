"""
Streamlit web interface for the Describe-to-Radar POC
Interactive odor search with radar charts and molecule visualization
PROTECTED VERSION with Authentication and Code Protection
"""

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import io
import base64
import os
import sys

# Add parent directory to Python path for module imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our modules
from pom_search import search_odor, get_search_engine
from visualization import (
    create_radar_chart_plotly, 
    create_molecule_grid, 
    render_molecule_structure,
    get_top_panel_words
)
from pdf_export import create_pdf_report
from config import PANEL_WORDS, DEFAULT_K, MAX_K, AUTH_ENABLED

# Import security modules
from auth import require_auth, get_current_user, logout, is_authenticated
from protection import full_protection


def get_base64_image(image_path):
    """Convert image to base64 string"""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except:
        return ""

def main():
    st.set_page_config(
        page_title="OSME - Odor Search Molecule Engine",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # TEMPORARILY DISABLE PROTECTION TO FIX CSP CONFLICTS
    # full_protection()
    
    # REQUIRE AUTHENTICATION IF ENABLED
    if AUTH_ENABLED:
        require_auth()
    
    # Header with SVG icon - Odor Search Molecule Engine text in white
    st.markdown("""
    <div style="display: flex; align-items: center; margin-bottom: 20px;">
        <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24" style="margin-right: 15px;">
            <path fill="currentColor" fill-rule="evenodd" d="M11 3.055A9 9 0 0 0 3.055 11H6v2H3.055A9 9 0 0 0 11 20.945V18h2v2.945A9 9 0 0 0 20.945 13H18v-2h2.945A9 9 0 0 0 13 3.055V6h-2zM1 12C1 5.925 5.925 1 12 1s11 4.925 11 11s-4.925 11-11 11S1 18.075 1 12m10-1V9h2v2h2v2h-2v2h-2v-2H9v-2z" clip-rule="evenodd"/>
        </svg>
        <h1 style="margin: 0; color: white;">Odor Search Molecule Engine</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # White bar with logo
    logo_path = "public/logo.png"
    logo_base64 = get_base64_image(logo_path)
    
    st.markdown("""
    <div style="background-color: white; padding: 10px; margin: 0 0 20px 0; text-align: center;">
        <img src="data:image/png;base64,{}" style="max-height: 60px; width: auto;">
    </div>
    """.format(logo_base64), unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Sidebar for configuration
    st.sidebar.header("Search Configuration")
    
    # Add logout button if authenticated
    if is_authenticated():
        current_user = get_current_user()
        st.sidebar.markdown(f"**Logged in as:** {current_user}")
        if st.sidebar.button("Logout", type="secondary"):
            logout()
    
    # Search parameters
    k_results = st.sidebar.slider(
        "Number of results",
        min_value=1,
        max_value=MAX_K,
        value=DEFAULT_K,
        help="How many similar molecules to find"
    )
    
    # Example prompts (English only)
    st.sidebar.subheader("Example Searches")
    example_prompts = [
        "sweet, vanilla, creamy",
        "floral, jasmine, fresh", 
        "woody, pine, earthy",
        "fruity, citrus, lemon",
        "spicy, pepper, warm",
        "green, grassy, herbal"
    ]
    
    selected_example = st.sidebar.selectbox(
        "Try an example:",
        [""] + example_prompts,
        help="Select an example search to get started"
    )
    
    # Main search interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Describe the Odor")
        
        # Use selected example or allow custom input
        default_prompt = selected_example if selected_example else ""
        
        search_prompt = st.text_area(
            "Enter your odor description:",
            value=default_prompt,
            height=100,
            placeholder="e.g., fresh, citrusy, with grassy notes...",
            help="Describe the odor using natural language. You can use words like 'fresh', 'sweet', 'woody', etc."
        )
        
        # Custom search button with SVG icon
        search_button_html = """
        <style>
        .search-button {
            display: inline-flex;
            align-items: center;
            background-color: #0066cc;
            color: white;
            border: none;
            border-radius: 8px;
            padding: 12px 24px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: background-color 0.3s;
            text-decoration: none;
            margin-top: 10px;
        }
        .search-button:hover {
            background-color: #0052a3;
        }
        .search-button svg {
            margin-right: 8px;
            fill: white;
        }
        </style>
        """
        st.markdown(search_button_html, unsafe_allow_html=True)
        
        search_button = st.button(
            label="Search Molecules",
            type="primary",
            help="Search for similar molecules",
            use_container_width=False
        )
        
        # Add search icon with CSS
        if search_button:
            st.markdown("""
            <script>
            document.querySelector('button[kind="primary"]').innerHTML = 
                '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 16 16" style="margin-right: 8px;"><path fill="white" fill-rule="evenodd" d="M11.5 7a4.5 4.5 0 1 1-9 0a4.5 4.5 0 0 1 9 0m-.82 4.74a6 6 0 1 1 1.06-1.06l2.79 2.79a.75.75 0 1 1-1.06 1.06z" clip-rule="evenodd"/></svg> Search Molecules';
            </script>
            """, unsafe_allow_html=True)
    
    with col2:
        st.header("Quick Examples")
        st.markdown("**Try these example searches:**")
        st.markdown("• fresh, citrus, sweet")
        st.markdown("• vanilla, woody, floral") 
        st.markdown("• spicy, herbal, green")
        st.markdown("• fruity, creamy, dairy")
    
    # Process search
    if search_button and search_prompt.strip():
        
        # Create progress container
        progress_container = st.empty()
        
        try:
            # Initialize search engine with detailed progress updates
            with progress_container.container():
                st.info("Initializing OSME Molecular Intelligence System...")
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                import time
                
                # Step 1: Neural network initialization
                status_text.text("Loading neural sentence transformer (384-dimensional embeddings)...")
                progress_bar.progress(10)
                time.sleep(1.2)
                
                # Step 2: Panel words setup
                status_text.text("Initializing 55-dimensional odor lexicon from Science paper...")
                progress_bar.progress(25)
                time.sleep(1.0)
                
                # Step 3: Database loading
                status_text.text("Loading Principal Odor Map (POM) database...")
                progress_bar.progress(40)
                time.sleep(0.8)
                
                # Step 4: FAISS index
                status_text.text("Building FAISS similarity index for vector search...")
                progress_bar.progress(60)
                time.sleep(1.0)
                
                # Actual initialization
                search_engine = get_search_engine()
                
                # Step 5: Calibration
                status_text.text("Calibrating molecular similarity algorithms...")
                progress_bar.progress(80)
                time.sleep(0.8)
                
                # Step 6: Ready
                status_text.text("OSME System ready for odor analysis!")
                progress_bar.progress(100)
                time.sleep(0.6)
            
            # Clear progress and show detailed search status
            progress_container.empty()
            
            # Enhanced search progress
            search_container = st.empty()
            with search_container.container():
                st.info("Analyzing your odor description...")
                search_progress = st.progress(0)
                search_status = st.empty()
                
                # Search step 1: Text processing
                search_status.text("Converting text to semantic vectors...")
                search_progress.progress(20)
                time.sleep(0.8)
                
                # Search step 2: RATA generation
                search_status.text("Generating 55D RATA odor fingerprint...")
                search_progress.progress(50)
                time.sleep(1.0)
                
                # Search step 3: Database search
                search_status.text("Searching molecular database for similar compounds...")
                search_progress.progress(80)
                time.sleep(0.8)
                
                # Actual search
                rata_vector, results_df = search_odor(search_prompt.strip(), k_results)
                
                # Search complete
                search_status.text("Found molecular matches! Preparing results...")
                search_progress.progress(100)
                time.sleep(0.6)
            
            # Clear search progress
            search_container.empty()
                
            # Store results in session state
            st.session_state.search_results = {
                'prompt': search_prompt,
                'rata_vector': rata_vector,
                'results_df': results_df
            }
            
            st.success(f"✅ Found {len(results_df)} similar molecules!")
            
        except FileNotFoundError as e:
            progress_container.empty()
            st.error("❌ Data files not found! Please ensure you have:")
            st.code("""
data/
├── embeddings.npy
└── molecules.csv
            """)
            st.info("💡 Run `python data/OpenPOM.py` to generate embeddings")
            st.info("💡 Run `python build_index.py` to build FAISS index for faster startup")
            return
            
        except Exception as e:
            progress_container.empty()
            st.error(f"❌ Search failed: {str(e)}")
            st.info("💡 Try running `python build_index.py` to rebuild the search index")
            return
    
    # Display results if available
    if 'search_results' in st.session_state:
        results = st.session_state.search_results
        
        st.markdown("---")
        st.header(f"Results for: '{results['prompt']}'")
        
        # Create two columns for radar chart and top descriptors
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Odor Profile (Radar Chart)")
            
            # Create interactive radar chart
            radar_fig = create_radar_chart_plotly(
                results['rata_vector'], 
                f"Odor Profile"
            )
            st.plotly_chart(radar_fig, use_container_width=True)
        
        with col2:
            st.subheader("Top Descriptors")
            
            # Show top panel words
            top_words = get_top_panel_words(results['rata_vector'], 10)
            
            for i, (word, score) in enumerate(top_words, 1):
                if score > 0.01:  # Only show significant scores
                    st.metric(
                        label=f"{i}. {word.title()}",
                        value=f"{score:.3f}"
                    )
        
        # Add PDF export button
        col_export, col_spacer = st.columns([1, 3])
        with col_export:
            if st.button("📄 Export to PDF", type="secondary", help="Download search results as PDF report"):
                with st.spinner("Generating PDF report..."):
                    try:
                        pdf_bytes = create_pdf_report(
                            results['prompt'], 
                            results['rata_vector'], 
                            results['results_df']
                        )
                        
                        # Create download
                        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"odor_search_report_{timestamp}.pdf"
                        
                        st.download_button(
                            label="⬇️ Download PDF Report",
                            data=pdf_bytes,
                            file_name=filename,
                            mime="application/pdf",
                            type="primary"
                        )
                        st.success("✅ PDF report generated successfully!")
                        
                    except Exception as e:
                        st.error(f"❌ PDF generation failed: {str(e)}")
        
        # Molecule results
        st.subheader("Similar Molecules")
        
        # Display results table
        display_df = results['results_df'].copy()
        
        # Format the dataframe for display
        smiles_col = 'smiles' if 'smiles' in display_df.columns else 'nonStereoSMILES'
        
        if 'name' in display_df.columns:
            display_columns = ['name', smiles_col, 'similarity_score']
        else:
            display_columns = [smiles_col, 'similarity_score']
            
        # Rename columns for display
        column_names = {
            'name': 'Molecule Name',
            'smiles': 'SMILES',
            'nonStereoSMILES': 'SMILES',
            'similarity_score': 'Similarity Score'
        }
        
        display_df = display_df[display_columns].rename(columns=column_names)
        
        # Format similarity scores
        if 'Similarity Score' in display_df.columns:
            display_df['Similarity Score'] = display_df['Similarity Score'].apply(lambda x: f"{x:.4f}")
        
        st.dataframe(display_df, use_container_width=True)
        
        # Molecule structures
        st.subheader("Molecule Structures")
        
        # Create molecule grid
        mol_grid = create_molecule_grid(results['results_df'], max_molecules=k_results)
        
        if mol_grid:
            # Left-aligned with fixed width
            st.image(mol_grid, caption="Top similar molecules", width=650)
        else:
            st.warning("Could not render molecule structures")
            
            # Fallback: show individual molecules
            cols = st.columns(min(3, len(results['results_df'])))
            
            for idx, (_, row) in enumerate(results['results_df'].iterrows()):
                if idx >= len(cols):
                    break
                    
                with cols[idx]:
                    smiles = row.get('smiles', row.get('nonStereoSMILES', ''))
                    mol_img = render_molecule_structure(smiles)
                    if mol_img:
                        st.image(mol_img, caption=f"Similarity: {row['similarity_score']:.3f}")
                    else:
                        st.text(f"SMILES: {smiles}")
                        st.text(f"Score: {row['similarity_score']:.3f}")
    
    # Footer with information in two columns
    st.markdown("---")
    
    col_about, col_descriptors = st.columns([1, 1])
    
    with col_about:
        st.markdown("""
        ### About OSME
        
        This is a Proof of Concept implementation of the Odor Search Molecule Engine based on the research paper 
        *"A principal odor map unifies diverse tasks in olfactory perception"* (Lee et al., Science 2023).
        
        **How it works:**
        1. **Text Processing**: Your description is converted to a 55-dimensional odor vector using the same descriptors from the paper
        2. **Similarity Search**: The system searches through a database of molecules using their Principal Odor Map (POM) embeddings
        3. **Results**: You get the most similar molecules with their chemical structures and similarity scores
        
        **Technical Details:**
        - Based on Graph Neural Networks trained on ~5,000 molecules
        - Uses 55-dimensional RATA embeddings from the POM
        - Employs FAISS for fast similarity search
        - Interactive radar charts show the odor profile breakdown
        - Uses English odor descriptions based on the 55-panel lexicon
        """)
    
    with col_descriptors:
        st.markdown(f"""
        ### Available Descriptors ({len(PANEL_WORDS)})
        
        The system understands these odor descriptors:
        """)
        
        # Display panel words in a compact format - 7 per row
        words_per_row = 7
        for i in range(0, len(PANEL_WORDS), words_per_row):
            row_words = PANEL_WORDS[i:i+words_per_row]
            st.markdown("• " + " • ".join(row_words))


if __name__ == "__main__":
    main()