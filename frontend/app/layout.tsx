import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
});

export const metadata: Metadata = {
  title: "RAG Intelligence — Production-Grade Retrieval System",
  description:
    "A production-grade RAG system using LangChain, Qdrant, Cohere, and Gemini. Hybrid search with cross-encoder re-ranking and strict citation enforcement.",
  keywords: ["RAG", "LangChain", "Qdrant", "AI", "LLM", "Retrieval"],
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={inter.variable}>
      <body>{children}</body>
    </html>
  );
}
