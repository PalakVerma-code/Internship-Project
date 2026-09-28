"""
Test script for Phase 2: RAG pipeline (document loading + chunking).

Run with:
    pytest tests/test_rag_pipeline.py -v
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag.document_loader import load_pdfs_from_directory, split_documents, load_and_split


def test_pdf_loading_finds_sample_document():
    documents = load_pdfs_from_directory("documents")
    assert len(documents) > 0


def test_pdf_loading_missing_directory_raises_error():
    try:
        load_pdfs_from_directory("this_folder_does_not_exist")
        assert False, "Expected FileNotFoundError"
    except FileNotFoundError:
        pass


def test_document_splitting_produces_chunks():
    documents = load_pdfs_from_directory("documents")
    chunks = split_documents(documents, chunk_size=300, chunk_overlap=50)
    assert len(chunks) >= len(documents)


def test_chunk_metadata_includes_source_and_page():
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
