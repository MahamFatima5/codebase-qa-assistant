# Codebase Q&A Assistant

A RAG-based assistant that answers questions about a codebase using
AST-aware chunking, FAISS retrieval, and Claude for generation.

## Setup

```bash
# 1. Create a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your Anthropic API key
echo "ANTHROPIC_API_KEY=your_key_here" > .env
```

## Usage

### Step 1 — Clone the repo you want to index
```bash
git clone https://github.com/pallets/flask.git target_repo
```
Pick something small (5k–20k lines) for your first run — a Flask
extension, a small CLI tool, etc. Avoid huge repos at first.

### Step 2 — Chunk the repo
```bash
python chunker.py target_repo
```
This walks the repo, extracts function/class-level chunks from `.py`
files via `ast`, and treats `.md`/`.rst`/`.txt` files as doc chunks.
Output: `data/chunks.json`.

### Step 3 — Build the FAISS index (run once)
```bash
python build_index.py
```
Embeds every chunk and saves `data/faiss_index.bin` +
`data/metadata.pkl`. You only re-run this when the repo changes.

### Step 4 — Launch the app
```bash
streamlit run app.py
```
The app loads the **saved** index on startup — it does not
re-embed anything. Ask questions in the chat box; expand "Sources"
under any answer to see the exact file/line the answer came from.

## Project structure
```
codebase-qa-assistant/
├── chunker.py        # Step 2: AST-based chunking
├── build_index.py     # Step 3: embeddings + FAISS index
├── retriever.py        # Loads index, does similarity search
├── app.py               # Streamlit chat UI
├── requirements.txt
├── .env                  # your ANTHROPIC_API_KEY (not committed)
└── data/
    ├── chunks.json
    ├── faiss_index.bin
    └── metadata.pkl
```

## Testing retrieval quality (before demoing)
Write ~15 questions covering:
- "How does X work?" (architecture/behavior questions)
- "Where is X defined?" (exact lookups)
- "Show me an example of X" (usage questions)
- Setup/config questions

Run them through `retriever.py` directly and manually check if the
top results are actually relevant — this is where most codebase RAG
projects quietly fail.

```python
from retriever import Retriever
r = Retriever()
for chunk in r.search("how does authentication work", top_k=5):
    print(chunk["file_path"], chunk["name"], chunk["score"])
```

## Next upgrades (once this works end-to-end)
- Swap `all-MiniLM-L6-v2` for a code-aware embedding model
  (`jinaai/jina-embeddings-v2-base-code`) for better retrieval on code.
- Use `tree-sitter` instead of `ast` to support non-Python repos.
- Add exact-name-match lookup (metadata filter) before falling back
  to semantic search for "where is X defined" queries.
- Add incremental re-indexing (only re-chunk changed files via
  `git diff`) instead of rebuilding the whole index.
