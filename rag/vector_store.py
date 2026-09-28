"""Phase 2 - Embeddings + ChromaDB Vector Store"""
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
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def build_vector_store(documents_dir: str = "documents", persist_directory: str = PERSIST_DIRECTORY) -> Chroma:
    chunks = load_and_split(documents_dir)
    embeddings = get_embedding_model()
    return Chroma.from_documents(
        documents=chunks, embedding=embeddings,
        collection_name=COLLECTION_NAME, persist_directory=persist_directory,
    )


def load_vector_store(persist_directory: str = PERSIST_DIRECTORY) -> Optional[Chroma]:
    if not os.path.isdir(persist_directory):
        return None
    embeddings = get_embedding_model()
    return Chroma(collection_name=COLLECTION_NAME, embedding_function=embeddings, persist_directory=persist_directory)


def get_or_build_vector_store(documents_dir: str = "documents", persist_directory: str = PERSIST_DIRECTORY) -> Chroma:
    store = load_vector_store(persist_directory)
    if store is not None:
        return store
    return build_vector_store(documents_dir, persist_directory)


def similarity_search(query: str, k: int = 4) -> List[Document]:
    store = get_or_build_vector_store()
    return store.similarity_search(query, k=k)
