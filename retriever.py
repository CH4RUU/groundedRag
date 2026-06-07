import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
from dotenv import load_dotenv
import json
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from qdrant_client import QdrantClient
from langchain_core.prompts import PromptTemplate
from langchain_cohere import CohereRerank
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever

load_dotenv()

def format_docs(docs):
    """Format documents into a single context string."""
    return "\n\n".join(
        f"Source: {doc.metadata.get('source', 'Unknown')}\n"
        f"Page: {doc.metadata.get('page', 'Unknown')}\n"
        f"Content: {doc.page_content}"
        for doc in docs
    )

def get_sources(docs):
    """Extract source metadata and a text snippet from each document."""
    sources = []
    for doc in docs:
        sources.append({
            "source": doc.metadata.get("source", "Unknown"),
            "page": doc.metadata.get("page", "Unknown"),
            "snippet": doc.page_content.replace("\n", " ")[:200] + "..."
        })
    return sources

def main():
    qdrant_url = os.environ.get("QDRANT_URL")
    qdrant_api_key = os.environ.get("QDRANT_API_KEY")
    cohere_api_key = os.environ.get("COHERE_API_KEY")
    
    if not qdrant_url or not qdrant_api_key:
        raise ValueError("Please set QDRANT_URL and QDRANT_API_KEY in your .env file.")
    if not cohere_api_key:
        raise ValueError("Please set COHERE_API_KEY in your .env file.")

    # Initialize Qdrant Client
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    
    # Initialize embeddings using text-embedding-004
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
    
    # Connect to the existing vector store collection with Hybrid Search
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="domain_context",
        embedding=embeddings,
        sparse_embedding=sparse_embeddings,
        retrieval_mode=RetrievalMode.HYBRID,
    )
    
    # Create a base retriever that fetches the top-15 most relevant chunks
    base_retriever = vector_store.as_retriever(search_kwargs={"k": 15})
    
    # Cohere Re-ranker to filter down to the top-4 best chunks
    compressor = CohereRerank(cohere_api_key=cohere_api_key, model="rerank-english-v3.0", top_n=4)
    retriever = ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=base_retriever
    )
    
    # Initialize Gemini LLM using an available model from the API key
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    
    # Load custom prompt template from configuration
    with open("prompts_v1.json", "r") as f:
        prompt_config = json.load(f)
        
    prompt = PromptTemplate.from_template(prompt_config["system_prompt"])
    
    # Chain composition
    chain = prompt | llm
    
    print("RAG System Ready. Type 'quit' to exit.\n")
    while True:
        question = input("Enter your question: ")
        if question.lower() in ['quit', 'exit', 'q']:
            break
            
        if not question.strip():
            continue
            
        print("\nRetrieving context and generating answer...")
        
        # Retrieve top-4 chunks
        retrieved_docs = retriever.invoke(question)
        
        if not retrieved_docs:
            print("No relevant documents found in the database.")
            continue
            
        # Generate response
        response = chain.invoke({
            "context": format_docs(retrieved_docs),
            "question": question
        })
        
        # Output the answer and strict citations
        print("\n" + "="*50)
        print("GENERATED ANSWER:")
        print("="*50)
        print(response.content)
        
        print("\n" + "="*50)
        print("SOURCES:")
        print("="*50)
        for i, source in enumerate(get_sources(retrieved_docs), 1):
            print(f"{i}. File: {source['source']} (Page/Section: {source['page']})")
            print(f"   Snippet: {source['snippet']}\n")

if __name__ == "__main__":
    main()
