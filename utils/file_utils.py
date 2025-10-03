import tempfile
import os
import json
from typing import Union

def save_temp_file(uploaded_file):
    suffix = os.path.splitext(getattr(uploaded_file, "name", "tmp"))[1] or ""
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        uploaded_file.seek(0)   # 🔑 rewind before reading
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    return tmp_path

def save_results(results: dict, filename: str = "results.json") -> str:
    path = os.path.abspath(filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    return path
