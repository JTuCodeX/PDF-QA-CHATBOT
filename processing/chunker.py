import math
from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_text(text: str, target_chunks: int = 50, overlap: int = 100):
    """
    Adaptive chunking - returns list of text chunks.
    """
    total_chars = len(text)
    chunk_size = max(500, math.ceil(total_chars / max(1, target_chunks)))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )
    return splitter.split_text(text)
