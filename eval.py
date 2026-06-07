import os
import json
import sys
import time
from dotenv import load_dotenv

# Ragas and Datasets
from datasets import Dataset
from ragas import evaluate
from ragas.metrics.collections import faithfulness, answer_relevancy

# LangChain models
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from qdrant_client import QdrantClient
from langchain_core.prompts import PromptTemplate
from langchain_cohere import CohereRerank
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type, retry_if_exception
from google.genai.errors import ClientError

# Fix for protobuf
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
load_dotenv()

def format_docs(docs):
    return "\n\n".join(
        f"Source: {doc.metadata.get('source', 'Unknown')}\n"
        f"Page: {doc.metadata.get('page', 'Unknown')}\n"
        f"Content: {doc.page_content}"
        for doc in docs
    )

def is_429_error(exception):
    if hasattr(exception, 'code') and exception.code == 429:
        return True
    if "429" in str(exception) or "RESOURCE_EXHAUSTED" in str(exception) or "Quota exceeded" in str(exception):
        return True
    return False

@retry(
    wait=wait_exponential(multiplier=5, min=10, max=120),
    stop=stop_after_attempt(10),
    retry=retry_if_exception(is_429_error)
)
def run_chain_with_retry(chain, inputs):
    return chain.invoke(inputs)

def main():
    print("Initializing Evaluation Pipeline...", flush=True)
    qdrant_url = os.environ.get("QDRANT_URL")
    qdrant_api_key = os.environ.get("QDRANT_API_KEY")
    cohere_api_key = os.environ.get("COHERE_API_KEY")
    google_api_key = os.environ.get("GOOGLE_API_KEY")
    
    if not all([qdrant_url, qdrant_api_key, cohere_api_key, google_api_key]):
        raise ValueError("Missing required environment variables.")

    # 1. Setup Retriever Pipeline
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
    
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="domain_context",
        embedding=embeddings,
        sparse_embedding=sparse_embeddings,
        retrieval_mode=RetrievalMode.HYBRID,
    )
    
    base_retriever = vector_store.as_retriever(search_kwargs={"k": 15})
    compressor = CohereRerank(cohere_api_key=cohere_api_key, model="rerank-english-v3.0", top_n=4)
    retriever = ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=base_retriever
    )
    
    # 2. Setup Generator Pipeline with High Retry to bypass free limits
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, max_retries=10)
    with open("prompts_v1.json", "r") as f:
        prompt_config = json.load(f)
    prompt = PromptTemplate.from_template(prompt_config["system_prompt"])
    chain = prompt | llm

    # 3. Load Golden Dataset
    if not os.path.exists("golden_dataset.json"):
        raise FileNotFoundError("golden_dataset.json not found. Please provide it.")
        
    with open("golden_dataset.json", "r") as f:
        golden_data = json.load(f)

    print(f"Loaded {len(golden_data)} golden examples. Running RAG pipeline...")

    # Data collection for Ragas
    questions = []
    answers = []
    contexts_list = []
    ground_truths = []

    for idx, item in enumerate(golden_data):
        question = item["question"]
        ground_truth = item["ground_truth"]
        
        # Run retrieval
        retrieved_docs = retriever.invoke(question)
        contexts = [doc.page_content for doc in retrieved_docs]
        
        # Generate answer
        response = run_chain_with_retry(chain, {
            "context": format_docs(retrieved_docs),
            "question": question
        })
        answer = response.content
        
        questions.append(question)
        answers.append(answer)
        contexts_list.append(contexts)
        ground_truths.append([ground_truth]) # Ragas expects ground truth as list of strings
        
        print(f"Processed {idx+1}/{len(golden_data)}. Sleeping 15s to respect free tier rate limits...")
        time.sleep(15)

    # 4. Construct HuggingFace Dataset
    # Supporting both old and new Ragas schema implicitly
    eval_dataset = Dataset.from_dict({
        "question": questions,
        "user_input": questions,
        "answer": answers,
        "response": answers,
        "contexts": contexts_list,
        "retrieved_contexts": contexts_list,
        "ground_truth": ground_truths,
        "reference": [g[0] for g in ground_truths]
    })

    # 5. Evaluate using Ragas with Gemini Judge
    print("Running Ragas evaluation with Gemini 2.5 Flash Judge...")
    
    # Ragas needs LLM and Embeddings explicitly configured for its metrics
    judge_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, max_retries=10)
    judge_embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    
    # Set the models in the metrics
    for metric in [faithfulness, answer_relevancy]:
        metric.llm = judge_llm
        if hasattr(metric, "embeddings"):
            metric.embeddings = judge_embeddings

    result = evaluate(
        dataset=eval_dataset,
        metrics=[faithfulness, answer_relevancy],
    )
    
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(result)

    # 6. CI/CD Gating Assertion
    avg_faithfulness = result.get("faithfulness", 0)
    print(f"\nAverage Faithfulness Score: {avg_faithfulness:.2f}")
    
    if avg_faithfulness < 0.85:
        print("❌ FAILED: Faithfulness score is below the threshold of 0.85.")
        sys.exit(1)
    else:
        print("✅ PASSED: Faithfulness score meets the threshold.")
        sys.exit(0)

if __name__ == "__main__":
    main()
