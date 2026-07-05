<div align="center">


# GroundedRAG

### *Answers you can trust, backed by the sources that prove them.*

**Hybrid Search&nbsp;&nbsp;·&nbsp;&nbsp;Cross-Encoder Re-Ranking&nbsp;&nbsp;·&nbsp;&nbsp;Citation-Enforced Generation&nbsp;&nbsp;·&nbsp;&nbsp;Cerebras Llama-3.1-8b**

[Live Demo](#live-demo) &nbsp;|&nbsp; [Pipeline](#pipeline) &nbsp;|&nbsp; [Stack](#stack) &nbsp;|&nbsp; [Structure](#structure) &nbsp;|&nbsp; [Deployment](#deployment) &nbsp;|&nbsp; [Run Locally](#run-locally)

![Next.js](https://img.shields.io/badge/Next.js-black?style=flat-square&logo=next.js&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Qdrant](https://img.shields.io/badge/Qdrant-FE4256?style=flat-square&logo=qdrant&logoColor=white)
![HuggingFace](https://img.shields.io/badge/Hugging%20Face-FFD21E?style=flat-square&logo=huggingface&logoColor=000)
![Vercel](https://img.shields.io/badge/Vercel-000000?style=flat-square&logo=vercel&logoColor=white)

</div>

<br>

## Live Demo

GroundedRAG is fully deployed and running right now.
<img src="Screenshot 1.png" alt="GroundedRAG Architecture" width="100%" />


| Layer | Platform | URL |
|---|---|---|
| Frontend | Vercel | [production-grade-rag-system.vercel.app](https://production-grade-rag-system.vercel.app/) |
| Backend API | Hugging Face Spaces | [rag-backend.hf.space](https://ayush707-rag-backend.hf.space) |

<br>

## Showcase

<div align="center">
<img src="Screenshot 1.png" alt="Frontend Dashboard" width="85%" />
<br><br>
<img src="Screenshot 2.png" alt="Observability Tooltips" width="85%" />
</div>

<br>

## Pipeline

GroundedRAG isn't a toy semantic-search demo — it's a fully observable, enterprise-grade retrieval pipeline engineered to kill hallucinations and squeeze out maximum retrieval precision.

```
  Query
    │
    ▼
┌───────────────────────┐
│ 1. Hybrid Retrieval    │  BM25 (sparse) + Dense Embeddings, searched together in Qdrant
└───────────┬───────────┘
            ▼
┌───────────────────────┐
│ 2. Cross-Encoder Rerank│  ms-marco-MiniLM-L-6-v2 rescoring top 10 → true top 5
└───────────┬───────────┘
            ▼
┌───────────────────────┐
│ 3. Semantic Dedup      │  Content hashing strips repeated chunks (nav bars, boilerplate)
└───────────┬───────────┘
            ▼
┌───────────────────────┐
│ 4. Sub-second Generate │  Llama-3.1-8b on Cerebras hardware, citation-locked prompt
└───────────┬───────────┘
            ▼
┌───────────────────────┐
│ 5. Dynamic Rendering   │  Next.js streams the answer with live latency/confidence UI
└───────────────────────┘
```

**1 · Hybrid Retrieval** — Every query is searched two ways at once: **BM25** sparse keyword matching and **dense vector embeddings** for semantic meaning, both running against a Qdrant index.

**2 · Cross-Encoder Re-Ranking** — The top 10 hybrid results get run through an `ms-marco-MiniLM-L-6-v2` cross-encoder, which scores query–chunk relevance directly and re-ranks down to the true top 5.

**3 · Semantic Deduplication** — Custom middleware hashes each re-ranked chunk's content. Duplicate text — like nav bars scraped across multiple pages — gets stripped before it ever reaches the LLM's context window.

**4 · Sub-second Generation** — The cleaned, ranked context flows into a strict citation-enforcement prompt, answered by `Llama-3.1-8b` on **Cerebras Inference** hardware for millisecond-scale token generation.

**5 · Dynamic UI Rendering** — The Next.js frontend cascades the response in with physics-based CSS animation, complete with hover tooltips showing per-chunk latency and neural confidence scores.

<br>

## Stack

<table>
<tr><td width="50%" valign="top">

### Frontend

![Next.js](https://img.shields.io/badge/Next.js-black?style=flat-square&logo=next.js&logoColor=white)
**Next.js 14 (App Router)** — server-rendered React for SEO and speed.

![CSS](https://img.shields.io/badge/CSS3-1572B6?style=flat-square&logo=css3&logoColor=white)
**CSS Modules** — vanilla CSS, glassmorphism, `cubic-bezier` transitions, staggered cinematic animation.

</td><td width="50%" valign="top">

### Backend & AI Orchestration

![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)
**FastAPI (Python)** — async API layer handling full orchestration.

**LangChain (LCEL)** — modular RAG routing and retrieval chains.

**Cerebras Inference** — ultra-low-latency `llama3.1-8b` generation.

**HuggingFace Embeddings** — local dense vectors via `all-MiniLM-L6-v2`.

**MS-MARCO Reranker** — local cross-encoder reranking.

![Qdrant](https://img.shields.io/badge/Qdrant-FE4256?style=flat-square&logo=qdrant&logoColor=white)
**Qdrant Cloud** — sparse + dense vector indexing.

</td></tr>
</table>

<br>

## Structure

GroundedRAG is a scalable monorepo, cleanly separating the UI layer from the AI logic layer.

```
GroundedRAG/
│
├── backend/                    FastAPI Python backend — the AI engine
│   ├── app/
│   │   ├── api/                REST endpoints
│   │   ├── ingestion/          Data loaders, scraping, text chunking
│   │   ├── observability/      Langfuse tracing callbacks
│   │   ├── rag/                LCEL chains, prompts, dedup logic
│   │   └── retrieval/          Hybrid retrievers, MS-MARCO reranker
│   ├── prompts/                YAML system prompts (citation-enforced)
│   ├── Dockerfile              Containerization (Python venv, HF-ready)
│   └── requirements.txt        Python dependencies
│
└── frontend/                   Next.js React frontend — the UI
    ├── app/                    App Router (page.tsx, layout.tsx)
    ├── components/             AnswerCard, QueryInput, Tooltips
    ├── public/                 Static assets, architecture images
    └── package.json            Node dependencies
```

<br>

## Deployment

A 100% free, microservice-based deployment architecture.

**Frontend — Vercel Edge Network**
The React UI runs on Vercel, using global Edge CDNs for near-zero-latency static delivery. It talks to the backend through `NEXT_PUBLIC_API_URL`.

**Backend — Hugging Face Spaces (Docker)**
The FastAPI backend runs as a containerized service on Hugging Face Spaces.
- A multi-stage Dockerfile uses a native Python `venv` to keep heavy dependencies (`torch`, `typing-extensions`, etc.) intact in the runtime image.
- Hugging Face's Docker SDK avoids the ~50s cold-start sleep common on other free tiers, keeping the pipeline responsive around the clock.

<br>

## Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/CH4RUU/groundedRag.git
cd groundedRag
```

**2. Backend**
```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env   # add your API keys

uvicorn app.main:app --reload --port 8000
```

**3. Frontend**
```bash
cd ../frontend
npm install
npm run dev
```

<br>

<div align="center">

Built with ❤️ by Charu Jagguka.

</div>
