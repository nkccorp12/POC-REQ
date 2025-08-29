FROM python:3.11-slim

# System libs für ML/SCI und RDKit drawing dependencies (+ CA certificates for HTTPS)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgomp1 git ca-certificates \
    libcairo2-dev pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .

# 1) Torch CPU-Wheel vorab installieren, damit keine CUDA-Pakete kommen
#    (danach normale Deps; pip lässt das CPU-Torch in Ruhe)
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    torch==2.4.1+cpu

# 2) Restliche Deps
RUN pip install --no-cache-dir -r requirements.txt

# 3) huggingface_hub sicher vorhanden (unabhängig vom req-Resolver)
RUN python3 -m pip install --no-cache-dir "huggingface_hub==0.25.2"

COPY . .

# ---- HF Pfade (OHNE offline), Threads etc. ----
ENV SENTENCE_MODEL="sentence-transformers/all-MiniLM-L6-v2" \
    SENTENCE_MODEL_DIR="/app/models/all-MiniLM-L6-v2" \
    HF_HOME="/app/hf_cache" \
    TRANSFORMERS_CACHE="/app/hf_cache" \
    HF_HUB_DISABLE_TELEMETRY=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    FAISS_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false \
    STREAMLIT_SERVER_FILE_WATCHER_TYPE=none

# 4) Modell WÄHREND DES BUILDS cachen – OFFLINE AUS!
RUN HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python3 - <<'PY' || { echo "=== HF snapshot failed ==="; exit 1; }
from huggingface_hub import snapshot_download
import os, sys, traceback

repo = os.environ["SENTENCE_MODEL"]
dst  = os.environ["SENTENCE_MODEL_DIR"]

print(f"🔄 Caching model: {repo} -> {dst}")

try:
    os.makedirs(dst, exist_ok=True)
    
    result_path = snapshot_download(
        repo_id=repo,
        local_dir=dst,
        local_dir_use_symlinks=False,
        ignore_patterns=["*.onnx","*tflite*"]
    )
    
    # Verify files were actually downloaded
    files = os.listdir(dst)
    print(f"✔ Cached model successfully at: {result_path}")
    print(f"  Files: {len(files)} items")
    print(f"  Sample: {files[:3] if files else 'EMPTY!'}")
    
except Exception as e:
    print(f"❌ Failed to cache model: {repo} -> {dst}", file=sys.stderr)
    print(f"   Error type: {type(e).__name__}", file=sys.stderr)
    print(f"   Error message: {str(e)}", file=sys.stderr)
    traceback.print_exc()
    raise

PY

# 5) JETZT offline für die Laufzeit aktivieren
ENV TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1

# Expose port (Render nutzt $PORT)
EXPOSE 8080

# Start Streamlit app using Render's PORT environment variable
CMD ["sh", "-c", "streamlit run api/index.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true"]