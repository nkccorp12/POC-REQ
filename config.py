"""
Configuration file for the Describe-to-Radar POC
Contains the 55 panel words from the Science paper and other constants
"""

# 55-dimensional odor lexicon from the Science paper (Table S1)
PANEL_WORDS = [
    'green', 'grassy', 'cucumber', 'tomato', 'hay', 'herbal', 'mint', 'woody', 
    'pine', 'floral', 'jasmine', 'rose', 'honey', 'fruity', 'citrus', 'lemon', 
    'orange', 'tropical', 'berry', 'peach', 'apple', 'sour', 'fermented', 
    'alcoholic', 'winey', 'rummy', 'caramellic', 'vanilla', 'spicy', 'coffee', 
    'smoky', 'roasted', 'meaty', 'nutty', 'fatty', 'coconut', 'waxy', 'dairy', 
    'buttery', 'cheesy', 'sulfurous', 'garlic', 'earthy', 'musty', 'animal', 
    'musk', 'powdery', 'sweet', 'cooling', 'sharp', 'medicinal', 'camphoreous', 
    'metallic', 'ozone', 'fishy'
]

# File paths
DATA_DIR = "data"
EMBEDDINGS_FILE = f"{DATA_DIR}/embeddings.npy"
MOLECULES_FILE = f"{DATA_DIR}/molecules.csv"
FAISS_INDEX_FILE = f"{DATA_DIR}/faiss_index.bin"

# Model configuration
import os
# Repo-ID kann via ENV überschrieben werden
SENTENCE_TRANSFORMER_MODEL = os.getenv(
    "SENTENCE_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"
)
# Lokaler Modellpfad (vom Dockerfile befüllt)
SENTENCE_MODEL_DIR = os.getenv("SENTENCE_MODEL_DIR", "/app/models/all-MiniLM-L6-v2")
POM_DIMENSIONS = 55  # 55D RATA dimensions from Lee et al. Science paper
RATA_DIMENSIONS = 55

# Search parameters
DEFAULT_K = 3
MAX_K = 10

# Visualization parameters
RADAR_FIGSIZE = (8, 8)
MOLECULE_IMAGE_SIZE = (400, 400)

# Authentication settings
AUTH_ENABLED = True
LOGIN_TITLE = "OSME Access Portal"
LOGIN_SUBTITLE = "Odor Search Molecule Engine - Authorized Access Only"