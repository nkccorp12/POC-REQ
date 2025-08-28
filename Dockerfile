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

# Expose port
EXPOSE 8080

# Start Streamlit app using Render's PORT environment variable
CMD ["sh", "-c", "streamlit run api/index.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true"]