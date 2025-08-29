FROM python:3.11-slim

# System libs für ML/SCI und RDKit drawing dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgomp1 git \
    libcairo2-dev pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# ---- Offline/Thread ENV + HF Caches ----
ENV SENTENCE_MODEL="sentence-transformers/all-MiniLM-L6-v2" \
    SENTENCE_MODEL_DIR="/app/models/all-MiniLM-L6-v2" \
    HF_HOME="/app/hf_cache" \
    TRANSFORMERS_CACHE="/app/hf_cache" \
    TRANSFORMERS_OFFLINE=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    FAISS_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false \
    STREAMLIT_SERVER_FILE_WATCHER_TYPE=none

# ---- Modell in der Build-Phase cachen (kein Download zur Laufzeit) ----
RUN python - <<'PY'
from huggingface_hub import snapshot_download
import os
repo = os.environ.get("SENTENCE_MODEL")
dst  = os.environ.get("SENTENCE_MODEL_DIR")
os.makedirs(dst, exist_ok=True)
snapshot_download(
    repo_id=repo,
    local_dir=dst,
    local_dir_use_symlinks=False,
    ignore_patterns=["*.onnx","*tflite*"]
)
print("✔ Cached model at:", dst)
PY

# Expose (Render ignoriert Port, nutzt $PORT)
EXPOSE 8080

# Start Streamlit app using Render's PORT environment variable
CMD ["sh", "-c", "streamlit run api/index.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true"]