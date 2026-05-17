# 🧠 Production-Grade RAG System

A fully production-hardened Retrieval-Augmented Generation system using **LangChain**, **Qdrant Cloud**, **Cohere**, and **Gemini** — with a CI/CD quality gate powered by **Ragas** and **GitHub Actions**. Zero hosting cost.

[![CI](https://github.com/YOUR_USERNAME/YOUR_REPO/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/YOUR_REPO/actions)

---

## Architecture

```
User Query
    │
    ▼
[Next.js — Vercel]
    │ POST /api/v1/query
    ▼
[FastAPI Backend — Docker → Koyeb]
    ├── EnsembleRetriever (Dense 50% + BM25 50%) ← RRF
    ├── CohereRerank (Cross-Encoder, top-5)
    ├── Gemini 1.5 Flash (with_structured_output)
    ├── Pydantic Citation Enforcer + Refusal Protocol
    └── Langfuse Tracing
         │
         ▼
    [Qdrant Cloud]

[GitHub Actions CI]
    ├── pytest (unit tests)
    └── Ragas Evaluation Gate (faithfulness > 90%)
```

---

## Tech Stack (All Free Tiers)

| Component | Tool | Cost |
|---|---|---|
| LLM | Google Gemini 1.5 Flash | Free (15 RPM) |
| Embeddings | Cohere `embed-english-v3.0` | Free dev tier |
| Re-ranking | Cohere `rerank-english-v3.0` | Free dev tier |
| Vector DB | Qdrant Cloud | Free (1GB) |
| Backend | FastAPI + LangChain | Open-source |
| Container Hosting | Koyeb / Render | Free tier |
| Frontend | Next.js | Open-source |
| CDN | Vercel | Free |
| CI/CD | GitHub Actions | Free (public repo) |
| Observability | Langfuse Cloud | Free (50k obs/mo) |
| Evaluation | Ragas + Pytest | Open-source |

---

## Project Structure

```
├── backend/
│   ├── app/
│   │   ├── api/            # FastAPI route handlers
│   │   ├── ingestion/      # load → chunk → upsert
│   │   ├── retrieval/      # embedder, vector_store, hybrid, reranker
│   │   ├── rag/            # LCEL chain, prompt registry
│   │   └── observability/  # Langfuse callback handler
│   ├── prompts/            # Version-controlled YAML prompts
│   ├── scripts/seed.py     # Quick seeder for recruiters
│   ├── tests/              # pytest unit tests
│   ├── Dockerfile
│   └── requirements.txt
├── evaluation/
│   ├── golden_dataset.json # 20 ground-truth Q&A pairs
│   └── run_evaluation.py   # Ragas gate (exit 1 if faith < 0.90)
├── frontend/               # Next.js glassmorphic UI
├── .github/workflows/ci.yml
└── docker-compose.yml
```

---

## Quick Start

### 1. Prerequisites

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd "Production-Grade RAG system"

# Copy env file and fill in your API keys
cp .env.example .env
```

You'll need (all free):
- [Google Gemini API Key](https://aistudio.google.com)
- [Cohere API Key](https://dashboard.cohere.com)
- [Qdrant Cloud](https://cloud.qdrant.io) — free 1GB cluster
- [Langfuse Cloud](https://cloud.langfuse.com) — optional

### 2. Run Locally (Docker Compose)

```bash
# Spins up FastAPI backend + local Qdrant
docker-compose up --build
```

API will be live at `http://localhost:8000`  
Docs at `http://localhost:8000/docs`

### 3. Seed the Vector Database

```bash
cd backend
pip install -r requirements.txt
python scripts/seed.py
```

This fetches LangChain documentation, chunks it, embeds it, and upserts to Qdrant.

### 4. Run Unit Tests

```bash
cd backend
pytest tests/ -v
```

### 5. Run Ragas Evaluation

```bash
cd evaluation
python run_evaluation.py
```

> **Note:** Rate limiting is built in (4-second sleep between calls to respect Gemini's 15 RPM free tier).

---

## API Reference

### `POST /api/v1/query`

```json
{
  "question": "What is Reciprocal Rank Fusion?",
  "session_id": "optional-session-id"
}
```

**Response:**
```json
{
  "answer": "RRF combines ranked lists...",
  "citations": ["chunk_a1b2c3d4"],
  "sources": ["langchain_docs.html"],
  "refusal": null,
  "confidence": 0.92,
  "retrieved_chunks": [...],
  "latency_ms": 1247.3,
  "prompt_version": "v1"
}
```

### `POST /api/v1/ingest`

```json
{
  "urls": ["https://python.langchain.com/docs/introduction/"]
}
```

---

## CI/CD Pipeline

Every push to `main` triggers:
1. **Unit tests** via pytest
2. **Docker build** verification  
3. **Ragas evaluation** against the golden dataset  
4. ❌ Build **fails** if `faithfulness < 0.90`  
5. ✅ On pass → Koyeb auto-redeploys from `main`

To add GitHub secrets, go to **Settings → Secrets → Actions**:
```
GEMINI_API_KEY
COHERE_API_KEY
QDRANT_URL
QDRANT_API_KEY
```

---

## Frontend Deployment (Vercel)

```bash
cd frontend
npm install
npm run dev     # localhost:3000
```

For Vercel production:
```bash
# Set NEXT_PUBLIC_API_URL=https://your-koyeb-backend.koyeb.app
vercel deploy
```

---

## ⚠️ Cold Start Warning

The free tiers of **Koyeb** and **Render** spin down after 15 minutes of inactivity.  
**The first request after inactivity may take 30–60 seconds to respond.** This is expected — subsequent requests will be fast.

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| `with_structured_output` | More robust than PydanticOutputParser for Gemini JSON enforcement |
| EnsembleRetriever (0.5/0.5) | Balanced RRF between semantic and keyword matching |
| Tenacity retry in evaluation | Prevents CI failure on Gemini 429 rate-limit errors |
| YAML prompt versioning | Prompts treated as infrastructure, enabling A/B testing |
| Refusal protocol | Prevents hallucination when context is insufficient |
