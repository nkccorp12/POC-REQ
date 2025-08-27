# Vercel Deployment Guide

## Quick Deploy

1. **Connect to Vercel:**
   ```bash
   npm i -g vercel
   vercel login
   ```

2. **Deploy:**
   ```bash
   vercel --prod
   ```

## Configuration

- **vercel.json**: Configured for Python 3.9 with Streamlit
- **main.py**: Entry point for Vercel deployment
- **.vercelignore**: Excludes unnecessary files
- **Data files**: Included (14MB total - within limits)

## Environment Variables

No environment variables needed - everything runs locally.

## Troubleshooting

**Build fails?**
- Check Python version (3.9 specified)
- Verify all dependencies in requirements.txt

**App doesn't start?**
- Check main.py entry point
- Verify streamlit_app.py exists

**Large files issue?**
- Data files are included but optimized
- Use .vercelignore to exclude unnecessary files

## Performance Notes

- First load may be slow (model downloads)
- Subsequent loads are faster due to caching
- FAISS index speeds up similarity search

## Direct Deploy Command

```bash
vercel --prod --yes
```

That's it! Your OSME app will be live on Vercel.