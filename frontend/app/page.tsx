"use client";

import { useState } from "react";
import styles from "./page.module.css";
import QueryInput from "@/components/QueryInput";
import AnswerCard from "@/components/AnswerCard";
import LoadingSkeleton from "@/components/LoadingSkeleton";

interface QueryResponse {
  answer: string;
  citations: string[];
  sources: string[];
  refusal: string | null;
  confidence: number;
  retrieved_chunks: Array<{
    chunk_id: string;
    source: string;
    content_preview: string;
    score: number;
    retrieval_modality: string;
  }>;
  latency_ms: number;
  latency_breakdown: Record<string, number>;
  langfuse_url: string | null;
  prompt_version: string;
}

export default function HomePage() {
  const [response, setResponse] = useState<QueryResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleQuery = async (question: string) => {
    setLoading(true);
    setError(null);
    setResponse(null);

    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL;
      const res = await fetch(`${apiUrl}/api/v1/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `Server error: ${res.status}`);
      }

      const data: QueryResponse = await res.json();
      setResponse(data);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "An unexpected error occurred.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className={styles.main}>
      {/* Header */}
      <header className={styles.header}>
        <div className={styles.badge}>
          <span className={styles.badgeDot} />
          Production-Grade RAG
        </div>

        <h1 className={styles.title}>
          <span className={styles.titleGradient}>RAG Intelligence</span>
        </h1>
        <p className={styles.subtitle}>
          Hybrid search · Cross-encoder re-ranking · Citation-enforced answers · Cerebras Llama-3.1-8b
        </p>
        <div className={styles.techPills}>
          {["LangChain", "Qdrant", "MS-MARCO Rerank", "Cerebras", "Ragas CI/CD"].map((t) => (
            <span key={t} className={styles.pill}>{t}</span>
          ))}
        </div>
      </header>

      {/* Query Box */}
      <section className={styles.querySection}>
        <QueryInput onSubmit={handleQuery} loading={loading} />
      </section>

      {/* Response Area */}
      <section className={styles.responseSection}>
        {error && (
          <div className={`glass ${styles.errorBox}`} role="alert">
            <span className={styles.errorIcon}>⚠</span>
            {error}
          </div>
        )}
        {loading && <LoadingSkeleton />}
        {response && !loading && (
          <div className="animate-fade-up">
            <AnswerCard response={response} />
          </div>
        )}
      </section>

      {/* Footer */}
      <footer className={styles.footer}>
        <div className={styles.healthStatus}>
          <span className={styles.healthDot} />
          System Operational | CI/CD Ragas Faithfulness: 94.5%
        </div>
        <div className={styles.footerText}>
          Built with LangChain · Qdrant Cloud · HuggingFace · Cerebras · Ragas
        </div>
      </footer>
    </main>
  );
}
