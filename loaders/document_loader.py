from pypdf import PdfReader
import os
from typing import Union

def load_document(path: Union[str, bytes]) -> str:
    """
    Loads text from a given document path or bytes-like object.

    Supported inputs:
      - file path string pointing to .pdf or .txt
      - bytes-like object representing the file content (not implemented in this loader)
    Returns extracted text as a single string.
    """
    if isinstance(path, str):
        ext = os.path.splitext(path)[1].lower()
        if ext == ".pdf":
            reader = PdfReader(path)
            text = " ".join([page.extract_text() or "" for page in reader.pages])
        elif ext == ".txt":
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
        else:
            raise ValueError(f"Unsupported file extension: {ext}")
    else:
        # If callers want to pass bytes they should write a wrapper that writes bytes to temp file and calls this loader
        raise ValueError("Unsupported input type. Provide a filesystem path to a PDF or TXT file.")

    return text
