# Describe-to-Radar POC

A Proof of Concept implementation of the "Describe-to-Radar" odor molecule search engine, based on the research paper *"A principal odor map unifies diverse tasks in olfactory perception"* (Lee et al., Science 2023).

## Overview

This system allows users to search for odor molecules using natural language descriptions. It converts text descriptions into molecular similarity searches using the Principal Odor Map (POM) from the research paper.

**Example**: Input "fresh, citrus, grassy" → Get the 3 most similar molecules with their structures and similarity scores.

## Features

- 🔍 **Natural Language Search**: Describe odors in everyday language
- 📊 **Interactive Radar Charts**: Visualize odor profiles across 55 descriptors
- 🧪 **Molecule Visualization**: See chemical structures of similar molecules
- 📈 **Similarity Scoring**: Quantified similarity based on POM embeddings
- 🌐 **Web Interface**: Easy-to-use Streamlit dashboard

## Requirements

- Python 3.8+
- Virtual environment (recommended)

## Installation

1. **Clone or download this repository**

2. **Create and activate virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download data files** (see Data Setup section below)

## Data Setup

You need two data files to run this POC:

```
data/
├── embeddings.npy    # 256-dimensional POM embeddings for molecules
└── molecules.csv     # Molecule metadata (names, SMILES, etc.)
```

### Data File Format

**embeddings.npy**: NumPy array with shape `(N, 256)` where N is the number of molecules

**molecules.csv**: CSV file with at least these columns:
- `smiles`: SMILES representation of molecules
- `name` (optional): Human-readable molecule names
- Any additional metadata columns

### Where to Get the Data

The original POM embeddings and molecule data should be obtained from:
- OpenPOM release/repository
- Science paper supplementary materials
- Contact the authors for the specific embeddings used

## Usage

### Command Line Testing

Test the core functionality:

```bash
python pom_search.py
```

Test visualizations:

```bash
python visualization.py
```

### Web Interface

Launch the Streamlit app:

```bash
streamlit run streamlit_app.py
```

Then open your browser to `http://localhost:8501`

### Example Searches

Try these example descriptions:
- "sweet, vanilla, creamy"
- "floral, jasmine, fresh"
- "woody, pine, earthy"
- "fruity, citrus, lemon"
- "spicy, pepper, warm"
- "green, grassy, herbal"

## Architecture

### Core Components

1. **config.py**: Configuration and the 55 panel words from the paper
2. **pom_search.py**: Main search engine with text-to-vector conversion
3. **visualization.py**: Radar charts and molecule structure rendering
4. **streamlit_app.py**: Web interface

### Technical Flow

1. **Text Input**: User provides natural language odor description
2. **Sentence Embedding**: Convert text to vector using Sentence-Transformers
3. **RATA Conversion**: Project to 55-dimensional odor space
4. **Similarity Search**: Find similar molecules using FAISS k-NN search
5. **Visualization**: Display radar chart and molecule structures

### Key Technologies

- **Sentence-Transformers**: Text-to-vector conversion
- **FAISS**: Fast similarity search
- **RDKit**: Molecule structure visualization
- **Streamlit**: Web interface
- **Plotly**: Interactive charts

## Limitations & Notes

- This is a simplified POC implementation
- The RATA→POM mapping is approximated (in a full system, you'd learn this mapping)
- Requires the original POM embeddings and molecule database
- Performance depends on the quality of input data

## Science Paper Reference

This implementation is based on:

**Lee, B.K., Mayhew, E.J., Sanchez-Lengeling, B. et al.** *A principal odor map unifies diverse tasks in olfactory perception.* Science 381, 999–1006 (2023). DOI: 10.1126/science.ade4401

## File Structure

```
describe-to-radar/
├── README.md
├── requirements.txt
├── config.py                 # Configuration and panel words
├── pom_search.py             # Core search functionality
├── visualization.py          # Charts and molecule rendering
├── streamlit_app.py          # Web interface
├── data/                     # Data files (not included)
│   ├── embeddings.npy
│   └── molecules.csv
└── venv/                     # Virtual environment
```

## Troubleshooting

**Error: Data files not found**
- Ensure `data/embeddings.npy` and `data/molecules.csv` exist
- Check file paths in `config.py`

**RDKIT issues**
- Make sure `rdkit-pypi` is installed correctly
- Some SMILES strings may be invalid

**Performance issues**
- FAISS search should be fast for reasonable database sizes
- Initial model loading takes a few seconds

## Future Improvements

- Learn proper RATA→POM mapping instead of approximation
- Add more visualization options
- Implement caching for better performance
- Add molecule property predictions
- Support for batch processing