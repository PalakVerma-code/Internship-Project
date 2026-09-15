"""
Test script for Phase 2: RAG pipeline (document loading + chunking).

Run with:
    pytest tests/test_rag_pipeline.py -v

Note: This test only checks loading/chunking logic, not the embedding
model itself (that requires a real internet connection to download the
model on first run, which pytest can't mock away easily). To confirm the
full pipeline including embeddings, run:
    python rag/vector_store.py
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag.document_loader import load_pdfs_from_directory, split_documents, load_and_split


def test_pdf_loading_finds_sample_document():
    """Confirms the sample PDF in documents/ loads without errors."""
    documents = load_pdfs_from_directory("documents")
    assert len(documents) > 0, "Expected at least one page loaded from documents/"


def test_pdf_loading_missing_directory_raises_error():
    """Confirms a clear error is raised if the documents folder doesn't exist."""
    try:
        load_pdfs_from_directory("this_folder_does_not_exist")
        assert False, "Expected FileNotFoundError to be raised"
    except FileNotFoundError:
        pass  # expected


def test_document_splitting_produces_chunks():
    """Confirms documents are actually split into multiple smaller chunks."""
    documents = load_pdfs_from_directory("documents")
    chunks = split_documents(documents, chunk_size=300, chunk_overlap=50)
    assert len(chunks) >= len(documents), "Splitting should produce >= as many chunks as pages"
    for chunk in chunks:
        assert len(chunk.page_content) <= 350  # allow small buffer over chunk_size


def test_chunk_metadata_includes_source_and_page():
    """Confirms each chunk retains traceability - which PDF and page it came from."""
    chunks = load_and_split("documents")
    for chunk in chunks:
        assert "source" in chunk.metadata
        assert "page" in chunk.metadata


if __name__ == "__main__":
    test_pdf_loading_finds_sample_document()
    test_pdf_loading_missing_directory_raises_error()
    test_document_splitting_produces_chunks()
    test_chunk_metadata_includes_source_and_page()
    print("All RAG pipeline tests passed.")
