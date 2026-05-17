import asyncio
from app.retrieval.hybrid_retriever import get_hybrid_retriever
from app.retrieval.reranker import get_reranking_compressor

async def main():
    try:
        base_retriever = await get_hybrid_retriever(None)
        docs = await base_retriever.ainvoke("How does hybrid search combine BM25 and vector search?")
        print("Base docs top 1 metadata:", docs[0].metadata if docs else "No docs")
        
        compressor = get_reranking_compressor()
        compressed_docs = compressor.compress_documents(docs, "How does hybrid search combine BM25 and vector search?")
        print("Compressed docs top 1 metadata keys:", compressed_docs[0].metadata.keys() if compressed_docs else "No docs")
        if compressed_docs:
            print("Compressed doc top 1 metadata dict:", compressed_docs[0].metadata)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    asyncio.run(main())
