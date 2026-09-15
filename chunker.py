"""
chunker.py
----------
Walks a cloned repo, extracts function/class-level chunks from Python files
using the ast module, and pulls in README / markdown docs as separate
"documentation" chunks.

Each chunk is a dict:
{
    "id": str,
    "type": "code" | "doc",
    "file_path": str,
    "name": str,              # function/class name, or doc filename
    "start_line": int,
    "end_line": int,
    "content": str,           # the actual code or text
}
"""

import ast
import os
import json
import hashlib

CODE_EXTENSIONS = {".py"}
DOC_EXTENSIONS = {".md", ".rst", ".txt"}
SKIP_DIRS = {".git", "__pycache__", "node_modules", "venv", ".venv", "dist", "build"}


def make_chunk_id(file_path: str, name: str, start_line: int) -> str:
    raw = f"{file_path}:{name}:{start_line}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def chunk_python_file(file_path: str) -> list:
    """Parse a single .py file into function/class-level chunks."""
    chunks = []
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            source = f.read()
        tree = ast.parse(source)
    except (SyntaxError, UnicodeDecodeError):
        return chunks

    source_lines = source.splitlines()

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start = node.lineno - 1
            end = getattr(node, "end_lineno", start + 1)
            snippet = "\n".join(source_lines[start:end])

            docstring = ast.get_docstring(node) or ""
            kind = "class" if isinstance(node, ast.ClassDef) else "function"

            chunks.append({
                "id": make_chunk_id(file_path, node.name, node.lineno),
                "type": "code",
                "file_path": file_path,
                "name": f"{kind}:{node.name}",
                "start_line": node.lineno,
                "end_line": end,
                "content": f"# {kind} {node.name}\n# docstring: {docstring}\n{snippet}",
            })

    # Fallback: if no functions/classes found (e.g. script-style file),
    # chunk the whole file so it isn't lost.
    if not chunks and source.strip():
        chunks.append({
            "id": make_chunk_id(file_path, "module", 1),
            "type": "code",
            "file_path": file_path,
            "name": "module",
            "start_line": 1,
            "end_line": len(source_lines),
            "content": source,
        })

    return chunks


def chunk_doc_file(file_path: str, max_chars: int = 1500) -> list:
    """Split a doc file into ~max_chars chunks on paragraph boundaries."""
    chunks = []
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    except UnicodeDecodeError:
        return chunks

    paragraphs = text.split("\n\n")
    buffer = ""
    part = 0
    start_line_estimate = 1

    for para in paragraphs:
        if len(buffer) + len(para) > max_chars and buffer:
            chunks.append({
                "id": make_chunk_id(file_path, f"part{part}", start_line_estimate),
                "type": "doc",
                "file_path": file_path,
                "name": f"{os.path.basename(file_path)} (part {part})",
                "start_line": start_line_estimate,
                "end_line": start_line_estimate,
                "content": buffer.strip(),
            })
            part += 1
            buffer = ""
        buffer += para + "\n\n"

    if buffer.strip():
        chunks.append({
            "id": make_chunk_id(file_path, f"part{part}", start_line_estimate),
            "type": "doc",
            "file_path": file_path,
            "name": f"{os.path.basename(file_path)} (part {part})",
            "start_line": start_line_estimate,
            "end_line": start_line_estimate,
            "content": buffer.strip(),
        })

    return chunks


def chunk_repo(repo_path: str) -> list:
    """Walk the repo and return all code + doc chunks."""
    all_chunks = []

    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]

        for fname in files:
            fpath = os.path.join(root, fname)
            ext = os.path.splitext(fname)[1].lower()

            if ext in CODE_EXTENSIONS:
                all_chunks.extend(chunk_python_file(fpath))
            elif ext in DOC_EXTENSIONS:
                all_chunks.extend(chunk_doc_file(fpath))

    return all_chunks


if __name__ == "__main__":
    import sys
    repo_path = sys.argv[1] if len(sys.argv) > 1 else "./target_repo"
    chunks = chunk_repo(repo_path)
    print(f"Extracted {len(chunks)} chunks from {repo_path}")

    os.makedirs("data", exist_ok=True)
    with open("data/chunks.json", "w") as f:
        json.dump(chunks, f, indent=2)
    print("Saved to data/chunks.json")
