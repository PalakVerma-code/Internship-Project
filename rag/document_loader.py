"""
Phase 2 - Task 2.1: Document Ingestion
-----------------------------------------
Loads PDF files (corporate policies, Indian legal framework texts) and
splits them into small overlapping chunks so they can be embedded and
searched later.

Beginner notes:
- "Chunking" matters because an LLM can't search a 200-page PDF directly.
  We break it into small pieces (~1000 characters), so later we can pull
  out just the 3-4 relevant pieces for a given question instead of the
  whole document.
- "chunk_overlap" repeats a little text between chunks so we don't cut a
  sentence in half and lose meaning at the boundary.
"""

import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def load_pdfs_from_directory(directory: str) -> List[Document]:
    """
    Loads every PDF in a directory. Each page becomes one Document object,
    tagged with metadata (source filename + page number) so we can later
    tell the user exactly where an answer came from.
    """
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Documents directory not found: {directory}")

    all_documents: List[Document] = []
    pdf_files = [f for f in os.listdir(directory) if f.lower().endswith(".pdf")]

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in '{directory}'. Add at least one policy "
            f"or legal framework PDF before building the vector store."
        )

    for filename in pdf_files:
        filepath = os.path.join(directory, filename)
        loader = PyPDFLoader(filepath)
        pages = loader.load()  # one Document per page, metadata includes source + page
        all_documents.extend(pages)

    return all_documents


def split_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> List[Document]:
    """
    Splits loaded documents into smaller overlapping chunks suitable for
    embedding and semantic search.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def load_and_split(directory: str) -> List[Document]:
    """Convenience function: load all PDFs in a directory, then chunk them."""
    documents = load_pdfs_from_directory(directory)
    chunks = split_documents(documents)
    return chunks


if __name__ == "__main__":
    # Quick manual check: python rag/document_loader.py
    chunks = load_and_split("documents")
    print(f"Loaded and split into {len(chunks)} chunks.")
    if chunks:
        print("\nSample chunk metadata:", chunks[0].metadata)
        print("Sample chunk text:\n", chunks[0].page_content[:300])
