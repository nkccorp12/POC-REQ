# Vercel Environment Variables Configuration

## Required Environment Variables for Vercel Dashboard

Configure these variables in your Vercel project dashboard under **Settings → Environment Variables**:

### Core Streamlit Configuration
```
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_BROWSER_GATHER_USAGE_STATS=false  
STREAMLIT_SERVER_ENABLE_CORS=false
```

### ML Model Cache Optimization (Critical)
```
TRANSFORMERS_CACHE=/tmp/.cache/transformers
HF_HOME=/tmp/.cache/huggingface
SENTENCE_TRANSFORMERS_HOME=/tmp/.cache/sentence_transformers
```

## Why These Variables Are Important

1. **Streamlit Variables**: Required for serverless deployment compatibility
2. **Cache Variables**: Redirect ML model downloads to `/tmp` instead of project root to avoid deployment size issues

## Deployment Size Fix Applied

- `.vercelignore`: Excludes ML models, caches, and data files
- `vercel.json`: Added `excludeFiles` configuration
- Environment variables redirect caches to temporary directories

This prevents the 4.3GB deployment error by keeping ML model files out of the deployment bundle.