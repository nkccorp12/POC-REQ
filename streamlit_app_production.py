"""
PRODUCTION Streamlit web interface for OSME - Odor Search Molecule Engine
Enhanced with 0.25 cosine threshold and intelligent user guidance
"""

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import io
from datetime import datetime

# Import our modules - PRODUCTION VERSION
from pom_search import search_odor, get_search_engine
from visualization import (
    create_radar_chart_plotly, 
    create_molecule_grid, 
    render_molecule_structure,
    get_top_panel_words
)
from pdf_export import create_pdf_report
from config import PANEL_WORDS, DEFAULT_K, MAX_K
from production_config import (
    VANILLIN_COSINE_THRESHOLD, BEST_VANILLA_QUERIES, QUERY_SUGGESTIONS,
    is_successful_vanillin_search, get_user_feedback
)
from vanillin_analysis import find_vanillin_in_database


def initialize_vanillin_detection():
    """Initialize vanillin detection for quality assessment"""
    if 'vanillin_embedding' not in st.session_state:
        try:
            vanillin_idx, _, vanillin_embedding = find_vanillin_in_database()
            st.session_state.vanillin_embedding = vanillin_embedding
            st.session_state.vanillin_idx = vanillin_idx
        except:
            st.session_state.vanillin_embedding = None
            st.session_state.vanillin_idx = None


def assess_search_quality(prompt, rata_vector, results_df):
    """Assess search quality and provide user feedback"""
    quality_assessment = {
        'vanillin_cosine': 0.0,
        'vanillin_rank': -1,
        'is_vanilla_query': False,
        'feedback_message': "",
        'suggestions': [],
        'quality_level': "unknown"
    }
    
    # Check if this is a vanilla-related query
    is_vanilla_query = any(term in prompt.lower() for term in ['vanilla', 'vanillin'])
    quality_assessment['is_vanilla_query'] = is_vanilla_query
    
    # If we have vanillin embedding, calculate cosine similarity
    if st.session_state.vanillin_embedding is not None and is_vanilla_query:
        vanillin_cosine = np.dot(rata_vector, st.session_state.vanillin_embedding) / (
            np.linalg.norm(rata_vector) * np.linalg.norm(st.session_state.vanillin_embedding)
        )
        quality_assessment['vanillin_cosine'] = vanillin_cosine
        
        # Find vanillin rank in results
        vanillin_smiles = "COc1cc(C=O)ccc1O"
        smiles_col = 'smiles' if 'smiles' in results_df.columns else 'nonStereoSMILES'
        
        for idx, (_, row) in enumerate(results_df.iterrows()):
            if row[smiles_col] == vanillin_smiles:
                quality_assessment['vanillin_rank'] = idx + 1
                break
        
        # Get feedback and suggestions
        quality_assessment['feedback_message'] = get_user_feedback(vanillin_cosine)
        
        # Quality level assessment
        if vanillin_cosine >= 0.27:
            quality_assessment['quality_level'] = "excellent"
        elif vanillin_cosine >= VANILLIN_COSINE_THRESHOLD:
            quality_assessment['quality_level'] = "good"
        elif vanillin_cosine >= 0.15:
            quality_assessment['quality_level'] = "fair"
        else:
            quality_assessment['quality_level'] = "poor"
        
        # Add suggestions based on current query
        simple_query = prompt.lower().strip()
        if simple_query in QUERY_SUGGESTIONS:
            quality_assessment['suggestions'] = QUERY_SUGGESTIONS[simple_query]
        elif vanillin_cosine < VANILLIN_COSINE_THRESHOLD:
            quality_assessment['suggestions'] = [
                "Try 'buttery vanilla' for better results",
                "Consider 'creamy vanilla' for dairy notes",
                "Use 'vanilla buttery' as alternative"
            ]
    
    return quality_assessment


def display_quality_feedback(quality_assessment):
    """Display quality feedback to user"""
    if not quality_assessment['is_vanilla_query']:
        return
    
    # Create quality indicator
    quality_level = quality_assessment['quality_level']
    vanillin_cosine = quality_assessment['vanillin_cosine']
    vanillin_rank = quality_assessment['vanillin_rank']
    
    # Color coding for quality levels
    color_map = {
        "excellent": "🟢",
        "good": "🟡", 
        "fair": "🟠",
        "poor": "🔴",
        "unknown": "⚪"
    }
    
    color = color_map.get(quality_level, "⚪")
    
    # Display quality assessment
    st.markdown("### 🎯 Vanilla Search Quality Assessment")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Vanillin Similarity",
            value=f"{vanillin_cosine:.3f}",
            delta=f"Target: {VANILLIN_COSINE_THRESHOLD}",
            delta_color="normal" if vanillin_cosine >= VANILLIN_COSINE_THRESHOLD else "inverse"
        )
    
    with col2:
        rank_display = str(vanillin_rank) if vanillin_rank > 0 else "Not in Top-20"
        st.metric(
            label="Vanillin Rank",
            value=rank_display,
            delta="Lower is better" if vanillin_rank > 0 else "Not found"
        )
    
    with col3:
        st.metric(
            label="Quality Level",
            value=f"{color} {quality_level.title()}"
        )
    
    # Display feedback message
    st.info(quality_assessment['feedback_message'])
    
    # Show suggestions if quality is poor
    if quality_assessment['suggestions'] and quality_level in ['fair', 'poor']:
        st.markdown("**💡 Suggestions for better results:**")
        for suggestion in quality_assessment['suggestions']:
            st.markdown(f"• {suggestion}")


def render_smart_query_suggestions():
    """Render intelligent query suggestions"""
    st.sidebar.markdown("### 🎯 Smart Query Suggestions")
    
    # Best vanilla queries (proven to work)
    st.sidebar.markdown("**🏆 Best Vanilla Queries:**")
    for query in BEST_VANILLA_QUERIES:
        if st.sidebar.button(f"✨ {query}", key=f"best_{query}"):
            st.session_state.suggested_query = query
    
    # Category-based suggestions
    st.sidebar.markdown("**📂 By Category:**")
    
    categories = {
        "🍰 Sweet & Dessert": [
            "sweet vanilla cream",
            "vanilla ice cream", 
            "vanilla cake",
            "custard vanilla"
        ],
        "🌿 Fresh & Green": [
            "fresh citrus",
            "grassy green",
            "herbal mint",
            "lemon fresh"
        ],
        "🌸 Floral": [
            "rose floral",
            "jasmine flower",
            "lavender fresh",
            "floral sweet"
        ],
        "🌲 Woody & Earthy": [
            "woody pine",
            "earthy mushroom",
            "cedar wood",
            "smoky woody"
        ]
    }
    
    for category, queries in categories.items():
        with st.sidebar.expander(category):
            for query in queries:
                if st.button(f"🔍 {query}", key=f"cat_{query}"):
                    st.session_state.suggested_query = query


def main():
    st.set_page_config(
        page_title="OSME - Odor Search Molecule Engine",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize vanillin detection
    initialize_vanillin_detection()
    
    # Header with production badge
    st.markdown("""
    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 20px;">
        <div style="display: flex; align-items: center;">
            <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24" style="margin-right: 15px;">
                <path fill="currentColor" fill-rule="evenodd" d="M11 3.055A9 9 0 0 0 3.055 11H6v2H3.055A9 9 0 0 0 11 20.945V18h2v2.945A9 9 0 0 0 20.945 13H18v-2h2.945A9 9 0 0 0 13 3.055V6h-2zM1 12C1 5.925 5.925 1 12 1s11 4.925 11 11s-4.925 11-11 11S1 18.075 1 12m10-1V9h2v2h2v2h-2v2h-2v-2H9v-2z" clip-rule="evenodd"/>
            </svg>
            <h1 style="margin: 0; color: #1f1f1f;">OSME</h1>
        </div>
        <div style="background: #28a745; color: white; padding: 5px 12px; border-radius: 15px; font-size: 12px; font-weight: bold;">
            PRODUCTION v2.0
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("Odor Search Molecule Engine")
    st.markdown("*Enhanced with intelligent vanilla search guidance*")
    st.markdown("---")
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Search Configuration")
    
    # Search parameters
    k_results = st.sidebar.slider(
        "Number of results",
        min_value=1,
        max_value=MAX_K,
        value=DEFAULT_K,
        help="How many similar molecules to find"
    )
    
    # Render smart query suggestions
    render_smart_query_suggestions()
    
    # Performance info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 System Performance")
    st.sidebar.info(f"""
    **Vanilla Search Quality:**
    • Target Similarity: ≥{VANILLIN_COSINE_THRESHOLD}
    • Best Query: "buttery vanilla"
    • Expected Performance: ~0.277
    """)
    
    # Main search interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🔍 Describe the Odor")
        
        # Use suggested query if available
        default_prompt = ""
        if 'suggested_query' in st.session_state:
            default_prompt = st.session_state.suggested_query
            st.session_state.pop('suggested_query')  # Use once
        
        search_prompt = st.text_area(
            "Enter your odor description:",
            value=default_prompt,
            height=100,
            placeholder="e.g., buttery vanilla, fresh citrus, woody pine...",
            help="💡 For vanilla searches, try 'buttery vanilla' for best results!"
        )
        
        # Smart search assistance
        if search_prompt.strip():
            prompt_lower = search_prompt.lower()
            if any(term in prompt_lower for term in ['vanilla', 'vanillin']):
                if 'buttery' not in prompt_lower:
                    st.warning("💡 **Tip:** Try 'buttery vanilla' for optimal vanillin detection!")
        
        search_button = st.button(
            "🔍 Search Molecules",
            type="primary",
            help="Search for similar molecules with quality assessment",
            use_container_width=False
        )
    
    with col2:
        st.header("📝 Available Descriptors")
        
        # Highlight vanilla-related terms
        vanilla_terms = ['vanilla', 'sweet', 'dairy', 'buttery', 'fruity', 'creamy']
        
        with st.expander(f"View all {len(PANEL_WORDS)} odor descriptors", expanded=False):
            st.markdown("**🍰 Vanilla-related terms (recommended):**")
            vanilla_in_panel = [word for word in PANEL_WORDS if word in vanilla_terms]
            st.markdown(" • ".join(vanilla_in_panel))
            
            st.markdown("**📋 All available descriptors:**")
            words_per_row = 4
            for i in range(0, len(PANEL_WORDS), words_per_row):
                row_words = PANEL_WORDS[i:i+words_per_row]
                # Highlight vanilla terms
                formatted_words = []
                for word in row_words:
                    if word in vanilla_terms:
                        formatted_words.append(f"**{word}**")
                    else:
                        formatted_words.append(word)
                st.markdown(" • ".join(formatted_words))
        
        # Quick vanilla search
        st.markdown("---")
        st.markdown("**🎯 Quick Vanilla Search:**")
        if st.button("🍰 Buttery Vanilla", help="Best vanilla search query"):
            st.session_state.quick_search = "buttery vanilla"
        if st.button("🥛 Creamy Vanilla", help="Dairy-focused vanilla"):
            st.session_state.quick_search = "creamy vanilla"
    
    # Handle quick search
    if 'quick_search' in st.session_state:
        search_prompt = st.session_state.quick_search
        st.session_state.pop('quick_search')
        search_button = True
        st.rerun()
    
    # Process search
    if search_button and search_prompt.strip():
        
        # Create progress container
        progress_container = st.empty()
        
        try:
            # Initialize search engine with progress updates
            with progress_container.container():
                st.info("🚀 Initializing search engine...")
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("Loading sentence transformer model...")
                progress_bar.progress(20)
                
                search_engine = get_search_engine()
                
                status_text.text("Search engine ready!")
                progress_bar.progress(100)
            
            # Clear progress and show search status
            progress_container.empty()
            
            with st.spinner("🔍 Searching for similar molecules..."):
                # Perform search
                rata_vector, results_df = search_odor(search_prompt.strip(), k_results)
                
            # Assess search quality
            quality_assessment = assess_search_quality(search_prompt, rata_vector, results_df)
            
            # Store results in session state
            st.session_state.search_results = {
                'prompt': search_prompt,
                'rata_vector': rata_vector,
                'results_df': results_df,
                'quality_assessment': quality_assessment,
                'timestamp': datetime.now()
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
        
        # Display quality feedback for vanilla queries
        if results['quality_assessment']['is_vanilla_query']:
            display_quality_feedback(results['quality_assessment'])
            st.markdown("---")
        
        # Create two columns for radar chart and top descriptors
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📡 Odor Profile (Radar Chart)")
            
            # Create interactive radar chart
            radar_fig = create_radar_chart_plotly(
                results['rata_vector'], 
                f"Odor Profile"
            )
            st.plotly_chart(radar_fig, use_container_width=True)
        
        with col2:
            st.subheader("🏆 Top Descriptors")
            
            # Show top panel words
            top_words = get_top_panel_words(results['rata_vector'], 10)
            
            for i, (word, score) in enumerate(top_words, 1):
                if score > 0.01:  # Only show significant scores
                    # Highlight vanilla-related terms
                    label = f"{i}. {word.title()}"
                    if word in ['vanilla', 'sweet', 'dairy', 'buttery', 'fruity']:
                        label += " 🍰"
                    
                    st.metric(
                        label=label,
                        value=f"{score:.3f}"
                    )
        
        # Add export and sharing options
        col_export, col_share, col_spacer = st.columns([1, 1, 2])
        
        with col_export:
            if st.button("📄 Export PDF", type="secondary", help="Download detailed PDF report"):
                with st.spinner("Generating PDF report..."):
                    try:
                        pdf_bytes = create_pdf_report(
                            results['prompt'], 
                            results['rata_vector'], 
                            results['results_df']
                        )
                        
                        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"osme_report_{timestamp}.pdf"
                        
                        st.download_button(
                            label="⬇️ Download PDF",
                            data=pdf_bytes,
                            file_name=filename,
                            mime="application/pdf",
                            type="primary"
                        )
                        st.success("✅ PDF generated!")
                        
                    except Exception as e:
                        st.error(f"❌ PDF generation failed: {str(e)}")
        
        with col_share:
            if st.button("🔗 Share Results", type="secondary", help="Get shareable link"):
                # Create shareable parameter string
                share_params = f"?query={search_prompt.replace(' ', '+')}&k={k_results}"
                st.code(f"Shareable URL: {st.get_option('browser.serverAddress')}{share_params}")
                st.info("💡 Copy this URL to share your search results")
        
        # Molecule results
        st.subheader("🧬 Similar Molecules")
        
        # Enhanced results table with quality indicators
        display_df = results['results_df'].copy()
        smiles_col = 'smiles' if 'smiles' in display_df.columns else 'nonStereoSMILES'
        
        # Add quality indicators for vanillin
        if results['quality_assessment']['is_vanilla_query']:
            vanillin_smiles = "COc1cc(C=O)ccc1O"
            display_df['Is_Vanillin'] = display_df[smiles_col] == vanillin_smiles
            
            # Highlight vanillin row
            def highlight_vanillin(row):
                if row['Is_Vanillin']:
                    return ['background-color: #90EE90'] * len(row)
                return [''] * len(row)
            
            # Show vanillin detection
            vanillin_rows = display_df[display_df['Is_Vanillin']]
            if not vanillin_rows.empty:
                vanillin_rank = vanillin_rows.index[0] + 1
                vanillin_score = vanillin_rows.iloc[0]['similarity_score']
                st.success(f"🎯 **Vanillin found at rank {vanillin_rank}** with similarity score {vanillin_score:.4f}")
            else:
                st.warning("⚠️ Vanillin not found in top results. Try 'buttery vanilla' for better detection.")
        
        # Format display columns
        if 'name' in display_df.columns:
            display_columns = ['name', smiles_col, 'similarity_score']
        else:
            display_columns = [smiles_col, 'similarity_score']
            
        column_names = {
            'name': 'Molecule Name',
            'smiles': 'SMILES',
            'nonStereoSMILES': 'SMILES', 
            'similarity_score': 'Similarity Score'
        }
        
        final_display_df = display_df[display_columns].rename(columns=column_names)
        
        # Format similarity scores
        if 'Similarity Score' in final_display_df.columns:
            final_display_df['Similarity Score'] = final_display_df['Similarity Score'].apply(lambda x: f"{x:.4f}")
        
        # Apply styling if vanillin is present
        if results['quality_assessment']['is_vanilla_query'] and 'Is_Vanillin' in display_df.columns:
            styled_df = final_display_df.style.apply(
                lambda row: ['background-color: #90EE90' if display_df.iloc[row.name]['Is_Vanillin'] else '' 
                           for _ in row], axis=1
            )
            st.dataframe(styled_df, use_container_width=True)
        else:
            st.dataframe(final_display_df, use_container_width=True)
        
        # Molecule structures with enhanced display
        st.subheader("🔬 Molecule Structures")
        
        mol_grid = create_molecule_grid(results['results_df'], max_molecules=k_results)
        
        if mol_grid:
            st.image(mol_grid, caption="Top similar molecules (green highlight = vanillin if found)", use_column_width=True)
        else:
            st.warning("Could not render molecule structures")
            
            # Fallback: show individual molecules with vanillin highlighting
            cols = st.columns(min(3, len(results['results_df'])))
            
            for idx, (_, row) in enumerate(results['results_df'].iterrows()):
                if idx >= len(cols):
                    break
                    
                with cols[idx]:
                    smiles = row.get('smiles', row.get('nonStereoSMILES', ''))
                    mol_img = render_molecule_structure(smiles)
                    
                    # Check if this is vanillin
                    is_vanillin = smiles == "COc1cc(C=O)ccc1O"
                    caption_text = f"Similarity: {row['similarity_score']:.3f}"
                    if is_vanillin:
                        caption_text += " 🎯 VANILLIN"
                    
                    if mol_img:
                        st.image(mol_img, caption=caption_text)
                    else:
                        st.text(f"SMILES: {smiles}")
                        st.text(caption_text)
    
    # Enhanced footer with production info
    st.markdown("---")
    st.markdown("""
    ### 🏭 About OSME Production v2.0
    
    This production-ready system is optimized for **vanilla/vanillin detection** with intelligent user guidance.
    
    **🎯 Key Features:**
    - **Smart Quality Assessment**: Automatic evaluation of vanilla search quality
    - **Intelligent Suggestions**: Recommends optimal queries for better results  
    - **Production Threshold**: Uses empirically validated 0.25 cosine similarity threshold
    - **Best Performance**: "Buttery vanilla" achieves ~0.277 similarity with vanillin
    
    **📊 System Specifications:**
    - **Target Similarity**: ≥0.25 (achievable and practical)
    - **Best Queries**: "buttery vanilla", "vanilla buttery", "creamy vanilla"
    - **Database**: ~5,000 molecules with POM embeddings
    - **Search Engine**: FAISS-powered similarity search
    - **Quality Control**: Real-time vanillin detection and ranking
    
    **🔬 Based on Research:**
    *"A principal odor map unifies diverse tasks in olfactory perception"* (Lee et al., Science 2023)
    """)
    
    # Production metrics
    if st.session_state.vanillin_embedding is not None:
        st.sidebar.markdown("---")
        st.sidebar.success("✅ Vanillin Detection: Active")
        st.sidebar.info(f"Quality Threshold: {VANILLIN_COSINE_THRESHOLD}")
    else:
        st.sidebar.warning("⚠️ Vanillin Detection: Unavailable")


if __name__ == "__main__":
    main()