"""
Phase 2 - Task 2.2: Embeddings + Vector Database
----------------------------------------------------
Turns document chunks into numeric "embeddings" (vectors that represent
meaning) and stores them in ChromaDB so we can later search "what's
semantically similar to this question" instead of just keyword matching.

Beginner notes:
- We use a HuggingFace embedding model that runs locally on your machine
  (no API key, no cost) - NOT Groq, because Groq only serves chat models,
  not embedding models.
- "Persisting" the vector store means saving it to disk in
  `chroma_db/` so you don't have to re-embed all your PDFs every time you
  restart the app - it loads instantly after the first build.
"""

import os
from typing import List, Optional
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from rag.document_loader import load_and_split

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
PERSIST_DIRECTORY = "chroma_db"
COLLECTION_NAME = "compliance_policies"


def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Loads a small, fast, free local embedding model.
    First run downloads it (~90MB); after that it's cached locally.
    """
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def build_vector_store(
    documents_dir: str = "documents",
    persist_directory: str = PERSIST_DIRECTORY,
) -> Chroma:
    """
    Builds a fresh vector store from all PDFs in `documents_dir` and saves
    it to disk. Call this whenever you add new policy documents.
    """
    chunks = load_and_split(documents_dir)
    embeddings = get_embedding_model()

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=persist_directory,
    )
    return vector_store


def load_vector_store(persist_directory: str = PERSIST_DIRECTORY) -> Optional[Chroma]:
    """
    Loads an already-built vector store from disk, if it exists.
    Returns None if it hasn't been built yet.
    """
    if not os.path.isdir(persist_directory):
        return None

    embeddings = get_embedding_model()
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )


def get_or_build_vector_store(
    documents_dir: str = "documents",
    persist_directory: str = PERSIST_DIRECTORY,
) -> Chroma:
    """
    Loads the vector store if it already exists on disk; otherwise builds
    it fresh from the PDFs in documents_dir. This is what the agents and
    dashboard should call.
    """
    store = load_vector_store(persist_directory)
    if store is not None:
        return store
    return build_vector_store(documents_dir, persist_directory)


def similarity_search(query: str, k: int = 4) -> List[Document]:
    """Returns the top-k most relevant chunks for a given query."""
    store = get_or_build_vector_store()
    return store.similarity_search(query, k=k)


if __name__ == "__main__":
    # Quick manual check: python rag/vector_store.py
    print("Building vector store from documents/ ...")
    build_vector_store()
    print("Done. Running a test search...")
    results = similarity_search("What are the data protection obligations?")
    for i, doc in enumerate(results, start=1):
        print(f"\n--- Result {i} (source: {doc.metadata.get('source')}, page {doc.metadata.get('page')}) ---")
        print(doc.page_content[:200])
