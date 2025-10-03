"""
Entry point example script that demonstrates:
 - processing an uploaded file
 - building index
 - running RAG Q&A
 - running MCQ pipeline

You can adapt this file to Streamlit/Gradio/FastAPI quickly.
"""

import os
from config import  DEFAULT_TARGET_CHUNKS, DEFAULT_CHUNK_OVERLAP, DEFAULT_K
from loaders.document_loader import load_document
from processing.chunker import chunk_text
from processing.embeddings import build_index
from rag.qa_pipeline import answer_question
from utils.file_utils import save_temp_file, save_results


def process_file_and_build_index(path_or_upload):
    """
    Accepts either a path string or a Streamlit UploadedFile object.
    Saves UploadedFile to a temp file before loading.
    """
    # Convert UploadedFile -> temp file path
    print("Path: ", path_or_upload)
    if isinstance(path_or_upload, str):
        path = path_or_upload
        cleanup = False
    else:
        path = save_temp_file(path_or_upload)
        print(path)
        cleanup = True

    # Load + chunk
    text = load_document(path).strip()
    if not text:
        raise ValueError(f"No text could be extracted from {path}")

    chunks = chunk_text(text, target_chunks=DEFAULT_TARGET_CHUNKS, overlap=DEFAULT_CHUNK_OVERLAP)
    if not chunks:
        raise ValueError(f"Document produced no chunks: {path}")

    index = build_index(chunks)

    # Remove temp file if created
    if cleanup:
        os.remove(path)

    return text, chunks, index


def run_qa_mode(index, question: str, history=None):
    return answer_question(question, index, k=DEFAULT_K, history=history)

if __name__ == "__main__":
    # Example usage from CLI:
    SAMPLE_FILE = os.environ.get("SAMPLE_FILE", "sample.pdf")
    # Build index
    print(f"Processing file {SAMPLE_FILE}...")
    text, chunks, index = process_file_and_build_index(SAMPLE_FILE)
    print(f"Document processed: {len(chunks)} chunks indexed.")

    # Example Q&A
    q = "What is the main conclusion of the document?"
    print("Running RAG Q&A for:", q)
    ans = run_qa_mode(index, q)
    print("Answer:", ans["answer"])

    # Example MCQ mode
    print("Generating MCQs (this will call the LLM)...")
    mcqs = run_mcq_mode(chunks, index, questions_per_idea=2, max_api_calls=10)
    out_path = save_results(mcqs, "questions.json")
    print("MCQs saved to", out_path)
