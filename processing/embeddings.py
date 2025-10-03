from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from typing import List
def build_index(chunks: List[str]):
    """
    Build a FAISS index from a list of text chunks using HuggingFace embeddings.
    Returns the FAISS index object (LangChain wrapper).
    """
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    index = FAISS.from_texts(chunks, embedder)
    return index
