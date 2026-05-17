<div align="center">
  <img src="Screenshot 1.png" alt="RAG Intelligence Architecture" width="100%" />
  
  <h1>🚀 Production-Grade RAG System</h1>
  <p><strong>Hybrid Search • Cross-Encoder Re-Ranking • Citation-Enforced Answers • Cerebras Llama-3.1-8b</strong></p>

  <p>
    <a href="#-live-demo">Live Demo</a> •
    <a href="#-how-it-works-the-pipeline">How It Works</a> •
    <a href="#-tech-stack">Tech Stack</a> •
    <a href="#-project-architecture--file-structure">Architecture</a> •
    <a href="#-deployment-architecture">Deployment</a>
  </p>
  
  <p>
    <img src="https://img.shields.io/badge/Next.js-black?style=for-the-badge&logo=next.js&logoColor=white" alt="Next.js" />
    <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI" />
    <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
    <img src="https://img.shields.io/badge/Qdrant-FE4256?style=for-the-badge&logo=qdrant&logoColor=white" alt="Qdrant" />
    <img src="https://img.shields.io/badge/Hugging%20Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=000" alt="Hugging Face" />
    <img src="https://img.shields.io/badge/Vercel-000000?style=for-the-badge&logo=vercel&logoColor=white" alt="Vercel" />
  </p>
</div>

---

## ✦ 🌐 Live Demo

The entire application is currently live and deployed!
- 🖥️ **Frontend (Vercel):** [https://production-grade-rag-system.vercel.app/](https://production-grade-rag-system.vercel.app/)
- ⚙️ **Backend API (Hugging Face):** [https://ayush707-rag-backend.hf.space](https://ayush707-rag-backend.hf.space)

---

## ✦ 📸 Showcase (Frontend UI)

<div align="center">
  <!-- NOTE: Add your real screenshots to the /docs folder and update these paths! -->
  <img src="Screenshot 1.png" alt="Frontend Dashboard" width="89%" />
  <img src="Screenshot 2.png" alt="Observability Tooltips" width="89%" />
</div>

---

## ✦ 🧠 How It Works (The Pipeline)

This system goes far beyond a basic "Semantic Search" RAG tutorial. It implements a fully observable, enterprise-ready pipeline designed to eliminate hallucinations and maximize retrieval accuracy:

1. 🔍 **Hybrid Retrieval:** When a user asks a question, the query is simultaneously searched using **BM25** (Sparse keyword matching) and **Dense Vector Embeddings** (semantic matching) inside a Qdrant database.
2. 🎯 **Cross-Encoder Re-Ranking:** The top 10 results from the Hybrid Search are passed through an `ms-marco-MiniLM-L-6-v2` neural network. This Cross-Encoder surgically scores the exact relationship between the query and each chunk, re-ranking them to find the true top 5 results.
3. 🛡️ **Semantic Deduplication:** Custom middleware intercepts the re-ranked chunks and hashes their content. If two chunks contain identical text (e.g., website navigation bars scraped from multiple pages), the duplicates are stripped out to save LLM context window space.
4. ⚡ **Sub-second Generation:** The highly refined context is passed into a strict citation-enforcement prompt. It is generated using `Llama-3.1-8b` running on **Cerebras Inference hardware**, guaranteeing lightning-fast, millisecond token generation.
5. 🎨 **Dynamic UI Rendering:** The Next.js frontend uses physics-based CSS animations to cascade the response onto the screen, rendering hover-glow tooltips that display the exact latency and neural confidence score of every retrieved chunk.

---

## ✦ 💻 Tech Stack

### 🖥️ Frontend Frameworks
- <img src="https://img.shields.io/badge/Next.js-black?style=flat&logo=next.js&logoColor=white" alt="Next.js" /> **Next.js 14 (App Router):** Server-side rendered React framework for maximum SEO and speed.
- <img src="https://img.shields.io/badge/CSS3-1572B6?style=flat&logo=css3&logoColor=white" alt="CSS" /> **CSS Modules:** Pure vanilla CSS for extreme performance, featuring glassmorphism, `cubic-bezier` physics transitions, and cinematic stagger animations.

### ⚙️ Backend & AI Orchestration
- <img src="https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white" alt="FastAPI" /> **FastAPI (Python):** Blazing fast async API framework handling the entire orchestration.
- 🦜 **LangChain (LCEL):** Modular pipeline construction for complex RAG routing and retrieval logic.
- 🧠 **Cerebras Inference:** Ultra-low latency LLM generation (`llama3.1-8b`).
- 🤗 **HuggingFace Embeddings:** Local dense embeddings via `all-MiniLM-L6-v2`.
- 🎯 **MS-MARCO Reranker:** Local cross-encoder reranking for maximum precision.
- <img src="https://img.shields.io/badge/Qdrant-FE4256?style=flat&logo=qdrant&logoColor=white" alt="Qdrant" /> **Qdrant Cloud:** Vector Database handling both sparse (BM25) and dense indexes.

---

## ✦ 📂 Project Architecture & File Structure

The project is structured as a scalable monorepo separating the UI layer from the AI logic layer.

```text
📦 Production-Grade-RAG-system
 ┣ 📂 backend/                 # ⚙️ FastAPI Python Backend (AI Engine)
 ┃ ┣ 📂 app/
 ┃ ┃ ┣ 📂 api/                 # 🌐 FastAPI Router Endpoints (REST API)
 ┃ ┃ ┣ 📂 ingestion/           # 📥 Data Loaders & Text Chunking Logic (Web Scraping)
 ┃ ┃ ┣ 📂 observability/       # 📊 Langfuse Tracing Callbacks for monitoring
 ┃ ┃ ┣ 📂 rag/                 # 🦜 LCEL Chains, Prompts, and Deduplication logic
 ┃ ┃ ┗ 📂 retrieval/           # 🔍 Hybrid Retrievers & MS-MARCO Reranker nodes
 ┃ ┣ 📂 prompts/               # 📝 YAML System Prompts (Citation-enforced)
 ┃ ┣ 📜 Dockerfile             # 🐳 Containerization config (Python venv for HF)
 ┃ ┗ 📜 requirements.txt       # 📦 Python Dependencies
 ┃
 ┗ 📂 frontend/                # 🖥️ Next.js React Frontend (User Interface)
   ┣ 📂 app/                   # 📄 Next.js App Router (page.tsx, layout.tsx)
   ┣ 📂 components/            # 🧩 UI Components (AnswerCard, QueryInput, Tooltips)
   ┣ 📂 public/                # 🖼️ Static Assets (hero.png, RAG Architecture images)
   ┗ 📜 package.json           # 📦 Node Dependencies
```

---

## ✦ 🚀 Deployment Architecture

This project is deployed using a 100% free, highly scalable microservice architecture:

### 1. Frontend: Vercel Edge Network <img src="https://img.shields.io/badge/Vercel-000000?style=flat&logo=vercel&logoColor=white" alt="Vercel" align="right" />
The React UI is deployed on **Vercel**, taking advantage of global Edge CDNs for zero-latency static asset delivery and instant page loads. It connects to the backend securely via the `NEXT_PUBLIC_API_URL` environment variable.

### 2. Backend: Hugging Face Spaces (Docker) <img src="https://img.shields.io/badge/Hugging%20Face-FFD21E?style=flat&logo=huggingface&logoColor=000" alt="Hugging Face" align="right"/>
The FastAPI Python backend is deployed as a containerized microservice on **Hugging Face Spaces**.
- **Containerization:** A custom multi-stage `Dockerfile` utilizes a native Python **Virtual Environment (`venv`)** to ensure complex dependencies (like `typing-extensions` and `torch`) are perfectly preserved in the runtime image.
- **Reliability:** By using Hugging Face's Docker SDK, the backend avoids the 50-second "cold start" sleep cycles common on other free tiers (like Render), keeping the AI pipeline incredibly responsive 24/7.

---

## ✦ 🛠️ Local Development

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

# Duplicate .env.example to .env and fill in your API keys
cp .env.example .env

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
