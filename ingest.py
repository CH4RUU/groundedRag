import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode

load_dotenv()

def load_documents(data_dir: str):
    """Load PDFs and Markdown files from the specified directory."""
    print(f"Loading documents from {data_dir}...")
    
    # Load PDFs
    pdf_loader = DirectoryLoader(data_dir, glob="**/*.pdf", loader_cls=PyPDFLoader)
    pdf_docs = pdf_loader.load()
    print(f"Loaded {len(pdf_docs)} PDF documents.")
    
    # Load Markdown files
    md_loader = DirectoryLoader(data_dir, glob="**/*.md", loader_cls=UnstructuredMarkdownLoader)
    md_docs = md_loader.load()
    print(f"Loaded {len(md_docs)} Markdown documents.")
    
    docs = pdf_docs + md_docs
    return docs

def main():
    # Load environment variables
    qdrant_url = os.environ.get("QDRANT_URL")
    qdrant_api_key = os.environ.get("QDRANT_API_KEY")
    
    if not qdrant_url or not qdrant_api_key:
        raise ValueError("Please set QDRANT_URL and QDRANT_API_KEY in your .env file.")
        
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    
    docs = load_documents(data_dir)
    if not docs:
        print(f"No documents found in '{data_dir}'. Please add some PDFs or Markdown files and try again.")
        return

    # Initialize text splitter for chunks of 700 tokens with 100 token overlap
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=700,
        chunk_overlap=100,
    )
    
    print("Splitting documents into chunks...")
    chunks = text_splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks.")
    
    # Use text-embedding-004 (latest embedding model)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    
    # Initialize sparse embeddings for BM25 hybrid search
    sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
    
    print("Connecting to Qdrant Cloud and upserting chunks with Hybrid Search...")
    
    # Create or update the vector store in the 'domain_context' collection
    QdrantVectorStore.from_documents(
        chunks,
        embeddings,
        sparse_embedding=sparse_embeddings,
        retrieval_mode=RetrievalMode.HYBRID,
        url=qdrant_url,
        api_key=qdrant_api_key,
        collection_name="domain_context",
        force_recreate=True, # Must recreate the collection to add sparse vectors
    )
    
    print("Successfully ingested documents into Qdrant!")

if __name__ == "__main__":
    main()
