"""Phase 2 - Document Ingestion"""
import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def load_pdfs_from_directory(directory: str) -> List[Document]:
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Documents directory not found: {directory}")
    all_documents: List[Document] = []
    pdf_files = [f for f in os.listdir(directory) if f.lower().endswith(".pdf")]
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in '{directory}'.")
    for filename in pdf_files:
        filepath = os.path.join(directory, filename)
        loader = PyPDFLoader(filepath)
        all_documents.extend(loader.load())
    return all_documents


def split_documents(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 150) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def load_and_split(directory: str) -> List[Document]:
    documents = load_pdfs_from_directory(directory)
    return split_documents(documents)
