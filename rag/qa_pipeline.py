"""
Conversational Q&A pipeline using RAG (retrieval-augmented generation).
Now includes conversation history so the assistant can handle follow-up questions.
"""

from google import genai
from google.genai import types

# Gemini client (make sure GEMINI_API_KEY is set in env)
import os
MODEL_NAME = "gemini-2.0-flash-lite"
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))


def answer_question(question, index, k=3, history=None):
    """
    Run conversational RAG with document retrieval + history.

    Args:
        question (str): User's question.
        index (FAISS): Vector index for semantic retrieval.
        k (int): Number of top passages to retrieve.
        history (list[tuple]): Optional conversation history as [(q, a, ctx), ...].

    Returns:
        dict: {
            "answer": str,
            "context": list[str]
        }
    """
    # Retrieve top-k docs
    docs = index.similarity_search(question, k=k)
    context = "\n\n".join([d.page_content for d in docs])

    # Build conversation history text
    history_text = ""
    if history:
        for q, a, _ in history[-5:]:  # include last 5 exchanges
            history_text += f"User: {q}\nAssistant: {a}\n"

    # Prompt with conversation + context
    prompt = f"""
    You are a helpful assistant. Use the ongoing conversation and
    the provided document context to answer the user’s latest question.

    Conversation so far:
    {history_text}

    Context from the document:
    {context}

    Latest Question:
    {question}
    """

    try:
        res = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.2)
        )
        return {
            "answer": res.text.strip(),
            "context": [d.page_content for d in docs]
        }
    except Exception as e:
        return {
            "answer": f"⚠️ Error generating answer: {e}",
            "context": []
        }
