"""
app.py
------
Streamlit UI: ask questions about the indexed codebase, get grounded
answers with file/line citations.

Run:
    streamlit run app.py
"""

import os

import streamlit as st
from dotenv import load_dotenv
from anthropic import Anthropic

from retriever import Retriever

load_dotenv()

st.set_page_config(page_title="Codebase Q&A Assistant", layout="wide")

# ---------- Cache the retriever + FAISS index so it loads ONCE ----------
@st.cache_resource
def load_retriever():
    return Retriever()


retriever = load_retriever()
client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

SYSTEM_PROMPT = """You are a codebase assistant. Answer ONLY using the
provided code/doc snippets below. Always mention the file path(s) you
used in your answer. If the snippets don't contain the answer, say
"I couldn't find that in the indexed codebase" — do not guess or use
outside knowledge."""


def build_context(chunks):
    blocks = []
    for c in chunks:
        blocks.append(
            f"[{c['type'].upper()}] {c['file_path']} "
            f"(lines {c['start_line']}-{c['end_line']}, {c['name']})\n"
            f"{c['content']}\n"
        )
    return "\n---\n".join(blocks)


def ask_claude(question: str, chunks: list) -> str:
    context = build_context(chunks)
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"CONTEXT:\n{context}\n\nQUESTION: {question}",
            }
        ],
    )
    return "".join(block.text for block in response.content if block.type == "text")


# ---------------------------- Sidebar ----------------------------
with st.sidebar:
    st.header("Index Info")
    st.write(f"**Chunks indexed:** {len(retriever.chunks)}")
    code_count = sum(1 for c in retriever.chunks if c["type"] == "code")
    doc_count = sum(1 for c in retriever.chunks if c["type"] == "doc")
    st.write(f"Code chunks: {code_count}")
    st.write(f"Doc chunks: {doc_count}")
    top_k = st.slider("Chunks to retrieve", 3, 10, 5)

# ---------------------------- Main chat ----------------------------
st.title("🔍 Codebase Q&A Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg:
            with st.expander("📁 Sources"):
                for c in msg["sources"]:
                    st.markdown(f"**{c['file_path']}** — {c['name']} "
                                f"(lines {c['start_line']}-{c['end_line']}, "
                                f"score {c['score']:.2f})")
                    st.code(c["content"][:500], language="python")

if question := st.chat_input("Ask about the codebase..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching index and asking Claude..."):
            retrieved = retriever.search(question, top_k=top_k)
            answer = ask_claude(question, retrieved)
        st.markdown(answer)
        with st.expander("📁 Sources"):
            for c in retrieved:
                st.markdown(f"**{c['file_path']}** — {c['name']} "
                            f"(lines {c['start_line']}-{c['end_line']}, "
                            f"score {c['score']:.2f})")
                st.code(c["content"][:500], language="python")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": retrieved,
    })
