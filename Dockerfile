FROM python:3.11-slim

# System libs für ML/SCI (bei Bedarf anpassen/erweitern)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgomp1 git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose port
EXPOSE 8080

# Start Streamlit app
CMD ["streamlit", "run", "api/index.py", "--server.port", "8080", "--server.address", "0.0.0.0", "--server.headless", "true"]