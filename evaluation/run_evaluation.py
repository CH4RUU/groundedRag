"""
evaluation/run_evaluation.py
─────────────────────────────
Ragas evaluation script.
Reads the golden dataset and runs faithfulness metric.
Exits with code 1 if faithfulness < 0.90.

RAG Inference: Cerebras llama3.1-8b — 1M tokens/day free, OpenAI-compatible
Ragas Judge:   Cerebras gpt-oss-120b — 120B model for reliable JSON-structured evaluation
  Both avoid Groq's per-minute token-bucket rate limits.
"""

import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from ragas.run_config import RunConfig
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Add backend to path
BACKEND_DIR = Path(__file__).parent.parent / "backend"
sys.path.append(str(BACKEND_DIR))

from app.config import get_settings
from app.rag.chain import execute_rag_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH = Path(__file__).parent / "eval_results.json"

# ─── Rate Limit Config ────────────────────────────────────────────────────────
# Groq free tier: ~30 RPM per model. Sleep 6s between questions to keep
# combined usage (RAG inference + Ragas evaluation) under the limit.
SLEEP_BETWEEN_CALLS_SECS = 6.0

FAITHFULNESS_THRESHOLD = 0.90


@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=5, max=60),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
async def run_single_question(question: str) -> Dict:
    """Run one question through the RAG pipeline with retry on 429."""
    result = await execute_rag_pipeline(question=question)
    return result


async def collect_results(questions: List[Dict]) -> List[Dict]:
    """Iterate through the golden dataset, respecting rate limits."""
    results = []
    total = len(questions)

    for i, item in enumerate(questions):
        question = item["question"]
        ground_truth = item["ground_truth"]

        logger.info("[%d/%d] Running: %s", i + 1, total, question[:80])

        try:
            result = await run_single_question(question)
            results.append({
                "question": question,
                "answer": result.get("answer", result.get("refusal", "")),
                # Use the FULL page_content, not the 200-char content_preview.
                # Ragas faithfulness requires the complete retrieved text to verify claims.
                "contexts": result.get("full_contexts", [
                    chunk.content_preview
                    for chunk in result.get("retrieved_chunks", [])
                ]),
                "ground_truth": ground_truth,
            })
        except Exception as e:
            logger.error("Failed on question %d: %s", i + 1, e)
            results.append({
                "question": question,
                "answer": "",
                "contexts": [],
                "ground_truth": ground_truth,
            })

        # Enforce rate limit — sleep between questions
        if i < total - 1:
            logger.debug("Sleeping %.1fs to respect RPM limit...", SLEEP_BETWEEN_CALLS_SECS)
            await asyncio.sleep(SLEEP_BETWEEN_CALLS_SECS)

    return results


def run_ragas_evaluation(results: List[Dict]) -> Dict[str, float]:
    """Run Ragas metrics on collected results."""
    settings = get_settings()

    # Using Cerebras via OpenAI-compatible client — gpt-oss-120b handles Ragas JSON prompts reliably
    ragas_llm = ChatOpenAI(
        model="gpt-oss-120b",
        temperature=0.0,
        max_tokens=4096,
        api_key=settings.cerebras_api_key,
        base_url="https://api.cerebras.ai/v1",
    )
    ragas_embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
    )

    dataset = Dataset.from_list(results)

    logger.info("Running Ragas evaluation metrics (Max workers=1 to prevent 429s)...")
    run_config = RunConfig(max_workers=1, max_retries=15, max_wait=60)
    score = evaluate(
        dataset=dataset,
        metrics=[faithfulness],
        llm=ragas_llm,
        run_config=run_config,
    )

    return score


async def main():
    logger.info("Loading golden dataset from %s", GOLDEN_DATASET_PATH)

    with open(GOLDEN_DATASET_PATH, "r") as f:
        questions = json.load(f)

    import os
    # Invalidate cache if it exists but contains empty answers (from previous failed runs)
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, "r") as f:
            cached = json.load(f)
        empty_count = sum(1 for r in cached if not r.get("answer", "").strip())
        if empty_count > 0:
            logger.warning("Cache has %d empty answers (from prior 404 failures). Re-running inference.", empty_count)
            os.remove(RESULTS_PATH)
        else:
            logger.info("Found valid cache at %s (%d entries), skipping inference.", RESULTS_PATH, len(cached))
            results = cached

    if not os.path.exists(RESULTS_PATH):
        logger.info("Running %d questions through the RAG pipeline...", len(questions))
        results = await collect_results(questions)

        # Save raw results
        with open(RESULTS_PATH, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Raw results saved to %s", RESULTS_PATH)

    # Filter out failed rows (empty answers) before Ragas to prevent NaN propagation
    valid_results = [r for r in results if r.get("answer", "").strip()]
    skipped = len(results) - len(valid_results)
    if skipped > 0:
        logger.warning("Skipping %d entries with empty answers before Ragas evaluation.", skipped)
    results = valid_results

    scores = run_ragas_evaluation(results)

    # Safely extract aggregate scores from EvaluationResult
    import math
    def _nan_safe_mean(values):
        valid = [v for v in values if v is not None and not math.isnan(v)]
        return sum(valid) / len(valid) if valid else 0.0

    if hasattr(scores, "items"):
        scores_dict = {k: v for k, v in scores.items()}
    elif isinstance(scores, dict):
        scores_dict = scores
    else:
        try:
            scores_dict = dict(scores)
        except Exception:
            scores_dict = {}

    # Ragas EvaluationResult.scores is a list of per-row dicts — compute NaN-safe mean
    if not scores_dict or all(v == 0.0 for v in scores_dict.values()):
        s = getattr(scores, "scores", [])
        if isinstance(s, list) and len(s) > 0:
            logger.info("Computing NaN-safe mean from %d per-row score dicts...", len(s))
            keys = s[0].keys()
            scores_dict = {k: _nan_safe_mean([row.get(k) for row in s]) for k in keys}

    faithfulness_score = scores_dict.get("faithfulness", 0.0)
    if math.isnan(faithfulness_score):
        faithfulness_score = 0.0
        
    logger.info("=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("  Faithfulness:      %.3f", faithfulness_score)
    logger.info("  Answer Relevancy:  %.3f", scores_dict.get("answer_relevancy", 0.0))
    logger.info("  Context Precision: %.3f", scores_dict.get("context_precision", 0.0))
    logger.info("=" * 60)

    # ── CI/CD Gate ────────────────────────────────────────────────────────────
    if faithfulness_score < FAITHFULNESS_THRESHOLD:
        logger.error(
            "GATE FAILED: Faithfulness score %.3f is below threshold %.2f",
            faithfulness_score,
            FAITHFULNESS_THRESHOLD,
        )
        return False
    else:
        logger.info(
            "GATE PASSED: Faithfulness %.3f >= %.2f",
            faithfulness_score,
            FAITHFULNESS_THRESHOLD,
        )
        return True


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
