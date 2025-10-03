from typing import List
from langchain_community.vectorstores import FAISS

def retrieve_passages(query: str, index: FAISS, k: int = 3):
    """
    Given a query and a FAISS index, return top-k passages as strings.
    """
    docs = index.similarity_search(query, k=k)
    return [d.page_content for d in docs]
