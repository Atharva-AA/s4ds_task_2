# Deployment Guide

## Backend (Hugging Face Spaces)

### Step 1: Create HF Space
- Go to [huggingface.co/spaces](https://huggingface.co/spaces)
- Click "Create new Space"
- Name: `rag-chatbot` (or your choice)
- License: OpenRAIL
- SDK: **Docker**
- Visibility: Public

### Step 2: Add Secret
- In Space settings → Secrets
- Name: `GROQ_API_KEY`
- Value: Your Groq API key

### Step 3: Push Code
```bash
cd /path/to/RAG-ChatBot-main
git init
git add .
git commit -m "Initial commit"
git remote add origin https://huggingface.co/spaces/YOUR-USERNAME/rag-chatbot
git push -u origin main
```

**Your backend URL:** `https://YOUR-USERNAME-rag-chatbot.hf.space`

---

## Frontend (Vercel)

### Step 1: Push to GitHub
```bash
git push origin main
```

### Step 2: Deploy to Vercel
- Go to [vercel.com](https://vercel.com)
- Click "New Project"
- Import your GitHub repo
- **Root Directory:** `frontend`
- Click "Deploy"

### Step 3: Set Environment Variable
- In Vercel Project Settings → Environment Variables
- Add: `NEXT_PUBLIC_API_URL=https://YOUR-USERNAME-rag-chatbot.hf.space`

### Step 4: Update Frontend
Edit `frontend/script.js` line 3:
```javascript
const API_BASE_URL = 'https://YOUR-USERNAME-rag-chatbot.hf.space';
```

Redeploy on Vercel after updating.

---

## Test

1. Open your Vercel frontend URL
2. Upload a PDF via the interface
3. Ask a question about it

Done! 🎉
