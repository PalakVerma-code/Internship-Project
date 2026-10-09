"""Supabase Storage + pgvector backend for persistent production RAG."""

import os
import tempfile
from pathlib import Path
from typing import List

from langchain_core.documents import Document

from rag.document_loader import load_and_split
from supabase_client import supabase

SUPABASE_DOCUMENT_BUCKET = os.getenv("SUPABASE_DOCUMENT_BUCKET", "compliance-documents")
SUPABASE_CHUNK_TABLE = os.getenv("SUPABASE_CHUNK_TABLE", "document_chunks")


class SupabaseVectorStore:
    """LangChain-compatible similarity search over Supabase pgvector rows."""

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        if supabase is None:
            raise RuntimeError("Supabase is not configured for the cloud vector backend.")

        from rag.vector_store import get_embedding_model

        query_embedding = get_embedding_model().embed_query(query)
        response = supabase.rpc(
            "match_document_chunks",
            {"query_embedding": query_embedding, "match_count": k},
        ).execute()

        return [
            Document(
                page_content=row.get("content", ""),
                metadata={
                    "source": row.get("source", "unknown"),
                    "page": row.get("page", "?"),
                    "similarity": row.get("similarity"),
                },
            )
            for row in (response.data or [])
        ]


def _require_supabase() -> None:
    if supabase is None:
        raise RuntimeError("SUPABASE_URL and SUPABASE_KEY are required for VECTOR_BACKEND=supabase.")


def upload_document(filename: str, content: bytes) -> str:
    """Upload one PDF to durable Supabase Storage."""
    _require_supabase()
    filename = os.path.basename(filename)
    if not filename.lower().endswith(".pdf"):
        raise ValueError("Only PDF files are accepted.")
    supabase.storage.from_(SUPABASE_DOCUMENT_BUCKET).upload(
        filename,
        content,
        {"upsert": "true", "content-type": "application/pdf"},
    )
    return f"supabase://{SUPABASE_DOCUMENT_BUCKET}/{filename}"


def list_documents() -> List[str]:
    """List PDF names in the configured Supabase Storage bucket."""
    _require_supabase()
    entries = supabase.storage.from_(SUPABASE_DOCUMENT_BUCKET).list()
    names = []
    for entry in entries:
        name = entry.get("name", "") if isinstance(entry, dict) else getattr(entry, "name", "")
        if name.lower().endswith(".pdf"):
            names.append(name)
    return sorted(names)


def _download_documents(directory: str) -> None:
    Path(directory).mkdir(parents=True, exist_ok=True)
    for filename in list_documents():
        content = supabase.storage.from_(SUPABASE_DOCUMENT_BUCKET).download(filename)
        Path(directory, filename).write_bytes(content)


def build_vector_store() -> SupabaseVectorStore:
    """Rebuild all pgvector rows from PDFs currently in Supabase Storage."""
    _require_supabase()

    with tempfile.TemporaryDirectory(prefix="compliance-documents-") as directory:
        _download_documents(directory)
        chunks = load_and_split(directory)
        if not chunks:
            raise FileNotFoundError("No PDF files found in Supabase Storage.")

        from rag.vector_store import get_embedding_model

        embeddings = get_embedding_model()
        rows = []
        for chunk in chunks:
            source = Path(chunk.metadata.get("source", "unknown")).name
            rows.append({
                "source": source,
                "page": chunk.metadata.get("page"),
                "content": chunk.page_content,
                "embedding": embeddings.embed_query(chunk.page_content),
            })

        supabase.table(SUPABASE_CHUNK_TABLE).delete().neq("id", 0).execute()
        for start in range(0, len(rows), 100):
            supabase.table(SUPABASE_CHUNK_TABLE).insert(rows[start:start + 100]).execute()

    return SupabaseVectorStore()


def get_vector_store() -> SupabaseVectorStore:
    _require_supabase()
    return SupabaseVectorStore()
