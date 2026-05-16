# GroundedRAG

Production-grade Retrieval-Augmented Generation (RAG) application for domain-specific document question answering with **hybrid retrieval, reranking, grounded citations, and automated evaluation**.

## Overview

GroundedRAG is an enterprise-style **Ask My Docs** system built to answer questions from uploaded documents with reliable, evidence-backed responses.

Instead of hallucinating, the system retrieves relevant document chunks, reranks them, and generates answers strictly grounded in source evidence.

---

## Features

* Hybrid Retrieval (**BM25 + Vector Search**)
* Cross-Encoder / Cohere Re-ranking
* Citation-Enforced Answer Generation
* Domain-Specific Document QA
* Automated Evaluation with Ragas
* Workflow Orchestration using LangGraph
* Scalable Vector Storage

---

## Tech Stack

### Core Framework

* LangChain
* LangGraph

### Vector Database

* ChromaDB / Weaviate

### Retrieval & Ranking

* BM25 Retriever
* Vector Similarity Search
* Cohere Reranker / Cross Encoder

### Evaluation

* Ragas

### LLM Layer

* OpenAI / Local LLM support

---

## Architecture

```text
Document Upload
   ↓
Parsing + Chunking
   ↓
Embedding Generation
   ↓
Vector Store Indexing
   ↓
Hybrid Retrieval
(BM25 + Semantic Search)
   ↓
Re-ranking
   ↓
Context Selection
   ↓
LLM Generation
   ↓
Answer with Citations
   ↓
Evaluation Pipeline
```

---

## Project Goals

This project focuses on building a **production-grade RAG system** that prioritizes:

* Retrieval accuracy
* Faithful grounded generation
* Explainability through citations
* Measurable evaluation metrics
* Real-world deployment readiness

---

## Example Query

**Question:**

> What is the approved treatment process for hazardous chromium waste?

**Response:**

> Chromium waste must undergo chemical reduction before precipitation treatment.
>
> **Source:** Environmental Compliance Manual, Section 4.2, Page 18

---

## Repository Structure

```text
src/
 ├── ingestion/
 ├── retrieval/
 ├── reranking/
 ├── generation/
 ├── evaluation/
 ├── workflows/
 └── api/

docs/
tests/
configs/
```

---

## Setup

```bash
git clone <repo-url>
cd groundedrag
pip install -r requirements.txt
```

Run:

```bash
python app.py
```

---

## Evaluation Metrics

Measured using Ragas:

* Faithfulness
* Answer Relevance
* Context Precision
* Context Recall

---

## Future Improvements

* Multi-document comparison
* Streaming responses
* PDF annotation highlighting
* Feedback-driven retrieval optimization
* Deployment on cloud infrastructure

---

## Why This Project?

GroundedRAG demonstrates advanced RAG engineering practices used in modern AI systems for enterprise knowledge retrieval.

It showcases:

* Retrieval engineering
* Evaluation pipelines
* Workflow orchestration
* Grounded generation

---

## Author

**Charu Jagguka**

Building AI systems for sustainability, intelligent knowledge retrieval, and environmental innovation.
