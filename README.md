---
title: RAG ChatBot Backend
emoji: 📚
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# RAG ChatBot

A Retrieval-Augmented Generation chatbot using Groq `llama-3.3-70b-versatile`, LangChain, and ChromaDB.

## Deployment

### Backend (HF Spaces)
1. Create a new Space on [huggingface.co/spaces](https://huggingface.co/spaces)
2. Select **Docker** as the SDK
3. Add `GROQ_API_KEY` as a secret in Space settings
4. Push this repo (the Dockerfile will deploy automatically)
5. Note your Space URL: `https://username-rag-chatbot.hf.space`

### Frontend (Vercel)
1. Push the `frontend/` folder to GitHub
2. Import the repo in [Vercel Dashboard](https://vercel.com)
3. Set the root directory to `frontend/`
4. Create environment variable: `NEXT_PUBLIC_API_URL=https://username-rag-chatbot.hf.space`
5. Update `frontend/script.js` line 2 with your HF Space URL
6. Deploy

## Local Development

```bash
pip install -r requirements.txt
cd backend
uvicorn main:app --reload --port 5000
```

Then open `http://localhost:5000` in your browser (which serves the frontend from `/frontend`).
