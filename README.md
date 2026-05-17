<div align="center">
  <img src="./frontend/public/hero.png" alt="RAG Intelligence Architecture" width="100%" />
  
  <h1>Production-Grade RAG System</h1>
  <p><strong>Hybrid Search • Cross-Encoder Re-Ranking • Citation-Enforced Answers • Cerebras Llama-3.1-8b</strong></p>

  <p>
    <a href="#live-demo">Live Demo</a> •
    <a href="#how-it-works">How It Works</a> •
    <a href="#tech-stack">Tech Stack</a> •
    <a href="#deployment-architecture">Deployment</a> •
    <a href="#showcase">Screenshots</a>
  </p>
</div>

---

## ✦ Live Demo

The entire application is currently live and deployed!
- **Frontend (Vercel):** [https://production-grade-rag-system.vercel.app/](https://production-grade-rag-system.vercel.app/)
- **Backend API (Hugging Face):** `https://ayush707-rag-backend.hf.space`

---

## ✦ Showcase (Frontend UI)

<div align="center">
  <!-- NOTE: Add your real screenshots to the /docs folder and update these paths! -->
  <img src="https://via.placeholder.com/800x450/111111/8b5cf6?text=Frontend+Dashboard+-+Add+Screenshot+1+Here" alt="Frontend Dashboard" width="48%" />
  <img src="https://via.placeholder.com/800x450/111111/10b981?text=Latency+Tooltips+-+Add+Screenshot+2+Here" alt="Observability Tooltips" width="48%" />
</div>

---

## ✦ How It Works (The Pipeline)

This system goes far beyond a basic "Semantic Search" RAG tutorial. It implements a fully observable, enterprise-ready pipeline designed to eliminate hallucinations and maximize retrieval accuracy:

1. **Hybrid Retrieval:** When a user asks a question, the query is simultaneously searched using **BM25** (Sparse keyword matching) and **Dense Vector Embeddings** (semantic matching) inside a Qdrant database.
2. **Cross-Encoder Re-Ranking:** The top 10 results from the Hybrid Search are passed through an `ms-marco-MiniLM-L-6-v2` neural network. This Cross-Encoder surgically scores the exact relationship between the query and each chunk, re-ranking them to find the true top 5 results.
3. **Semantic Deduplication:** Custom middleware intercepts the re-ranked chunks and hashes their content. If two chunks contain identical text (e.g., website navigation bars scraped from multiple pages), the duplicates are stripped out to save LLM context window space.
4. **Sub-second Generation:** The highly refined context is passed into a strict citation-enforcement prompt. It is generated using `Llama-3.1-8b` running on **Cerebras Inference hardware**, guaranteeing lightning-fast, millisecond token generation.
5. **Dynamic UI Rendering:** The Next.js frontend uses physics-based CSS animations to cascade the response onto the screen, rendering hover-glow tooltips that display the exact latency and neural confidence score of every retrieved chunk.

---

## ✦ Tech Stack

### Frontend
- **Next.js 14 (App Router):** Server-side rendered React framework.
- **CSS Modules:** Pure vanilla CSS for extreme performance, featuring glassmorphism, `cubic-bezier` physics transitions, and cinematic stagger animations.

### Backend & AI Orcehstration
- **FastAPI (Python):** Blazing fast async API framework.
- **LangChain (LCEL):** Modular pipeline construction for complex RAG flows.
- **Cerebras Inference:** Ultra-low latency LLM generation (`llama3.1-8b`).
- **HuggingFace Embeddings:** Local dense embeddings via `all-MiniLM-L6-v2`.
- **MS-MARCO Reranker:** Local cross-encoder reranking for maximum precision.
- **Qdrant Cloud:** Vector Database handling both sparse (BM25) and dense indexes.

---

## ✦ Deployment Architecture

This project is deployed using a 100% free, highly scalable microservice architecture:

### 1. Frontend: Vercel Edge Network
The React UI is deployed on **Vercel**, taking advantage of global Edge CDNs for zero-latency static asset delivery and instant page loads. It connects to the backend securely via the `NEXT_PUBLIC_API_URL` environment variable.

### 2. Backend: Hugging Face Spaces (Docker)
The FastAPI Python backend is deployed as a containerized microservice on **Hugging Face Spaces**.
- **Containerization:** A custom multi-stage `Dockerfile` utilizes a native Python **Virtual Environment (`venv`)** to ensure complex dependencies (like `typing-extensions` and `torch`) are perfectly preserved in the runtime image.
- **Reliability:** By using Hugging Face's Docker SDK, the backend avoids the 50-second "cold start" sleep cycles common on other free tiers (like Render), keeping the AI pipeline incredibly responsive.

---

## ✦ Local Development

### 1. Clone the repository
```bash
git clone https://github.com/Ayush1Deshmukh/Production-Grade-RAG-system.git
cd Production-Grade-RAG-system
```

### 2. Backend Setup
```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Start the FastAPI server
uvicorn app.main:app --reload --port 8000
```

### 3. Frontend Setup
```bash
cd ../frontend
npm install

# Start the Next.js development server
npm run dev
```

---

<div align="center">
  <p>Built with ❤️ by Ayush Deshmukh.</p>
</div>
