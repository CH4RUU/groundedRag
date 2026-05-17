"""
app/ingestion/loader.py
───────────────────────
Unified document loader factory.
Supports PDF, Markdown, HTML, and plain text.
Each loaded document gets `source` and `file_type` metadata attached.
"""

import logging
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredMarkdownLoader,
    BSHTMLLoader,
    TextLoader,
    WebBaseLoader,
)

logger = logging.getLogger(__name__)

LOADER_MAP = {
    ".pdf": PyPDFLoader,
    ".md": UnstructuredMarkdownLoader,
    ".markdown": UnstructuredMarkdownLoader,
    ".html": BSHTMLLoader,
    ".htm": BSHTMLLoader,
    ".txt": TextLoader,
}


def load_file(path: Path) -> List[Document]:
    """Load a single file and attach source metadata."""
    suffix = path.suffix.lower()
    loader_cls = LOADER_MAP.get(suffix)
    if loader_cls is None:
        logger.warning("Unsupported file type '%s', skipping: %s", suffix, path)
        return []

    logger.info("Loading %s (%s)", path.name, suffix)
    loader = loader_cls(str(path))
    docs = loader.load()

    for doc in docs:
        doc.metadata["source"] = path.name
        doc.metadata["file_type"] = suffix
        doc.metadata["file_path"] = str(path)

    return docs


def load_directory(directory: Path) -> List[Document]:
    """Recursively load all supported files from a directory."""
    all_docs: List[Document] = []
    supported = set(LOADER_MAP.keys())

    for file_path in sorted(directory.rglob("*")):
        if file_path.is_file() and file_path.suffix.lower() in supported:
            docs = load_file(file_path)
            all_docs.extend(docs)

    logger.info("Loaded %d raw documents from %s", len(all_docs), directory)
    return all_docs


def load_urls(urls: List[str]) -> List[Document]:
    """Load documents from web URLs using WebBaseLoader."""
    logger.info("Fetching %d URL(s)...", len(urls))
    loader = WebBaseLoader(urls)
    docs = loader.load()
    for doc in docs:
        doc.metadata.setdefault("source", doc.metadata.get("source", "web"))
        doc.metadata["file_type"] = "html"
    logger.info("Fetched %d documents from web", len(docs))
    return docs
