"""DocIntel RAG - Streamlit Interface."""

import sys
import time
import logging
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import settings
from pipeline import RAGPipeline

logging.basicConfig(level=settings.LOG_LEVEL,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

st.set_page_config(page_title="DocIntel RAG", layout="wide",
                    initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

:root {
    --bg: #06080d;
    --bg2: #0c1018;
    --card: rgba(15, 20, 35, 0.85);
    --card-border: rgba(56, 120, 255, 0.08);
    --card-hover: rgba(56, 120, 255, 0.12);
    --glass: rgba(20, 28, 50, 0.65);
    --accent1: #3b82f6;
    --accent2: #8b5cf6;
    --accent3: #06b6d4;
    --grad: linear-gradient(135deg, #3b82f6, #8b5cf6, #06b6d4);
    --grad-subtle: linear-gradient(135deg, rgba(59,130,246,0.15), rgba(139,92,246,0.10), rgba(6,182,212,0.08));
    --text: #e2e8f0;
    --text2: #94a3b8;
    --muted: #475569;
    --border: rgba(148,163,184,0.08);
    --green: #22c55e;
    --red: #ef4444;
}

html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', -apple-system, sans-serif !important;
    background: var(--bg) !important;
    color: var(--text) !important;
}

[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background:
        radial-gradient(ellipse 600px 400px at 15% 10%, rgba(59,130,246,0.06), transparent),
        radial-gradient(ellipse 500px 350px at 85% 20%, rgba(139,92,246,0.05), transparent),
        radial-gradient(ellipse 400px 300px at 50% 80%, rgba(6,182,212,0.04), transparent);
    pointer-events: none;
    z-index: 0;
}

[data-testid="stSidebar"] {
    background: var(--bg2) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: var(--grad);
}
[data-testid="stHeader"] { background: transparent !important; }
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; }

/* === HEADER === */
.hero {
    text-align: center;
    padding: 2rem 0 1.5rem;
    position: relative;
}
.hero h1 {
    font-size: 2.4rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    margin: 0;
    background: var(--grad);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-line {
    width: 80px;
    height: 3px;
    background: var(--grad);
    margin: 0.6rem auto 0.4rem;
    border-radius: 2px;
}
.hero p {
    color: var(--text2);
    font-size: 0.92rem;
    font-weight: 400;
    letter-spacing: 0.02em;
}

/* === GLASS CARDS === */
.gcard {
    background: var(--glass);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid var(--card-border);
    border-radius: 14px;
    padding: 1.3rem 1.5rem;
    transition: all 0.25s ease;
    position: relative;
    overflow: hidden;
}
.gcard::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: var(--grad);
    opacity: 0;
    transition: opacity 0.25s;
}
.gcard:hover {
    border-color: var(--card-hover);
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(59,130,246,0.08);
}
.gcard:hover::before { opacity: 1; }

.stat-num {
    font-size: 2rem;
    font-weight: 800;
    background: var(--grad);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.1;
}
.stat-label {
    font-size: 0.72rem;
    color: var(--text2);
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-weight: 600;
    margin-top: 6px;
}

/* === STATUS DOT === */
.dot {
    display: inline-block;
    width: 9px;
    height: 9px;
    border-radius: 50%;
    margin-right: 7px;
    position: relative;
    top: 0px;
}
.dot-on {
    background: var(--green);
    box-shadow: 0 0 8px var(--green), 0 0 20px rgba(34,197,94,0.2);
    animation: pulse-green 2s infinite;
}
.dot-off {
    background: var(--red);
    box-shadow: 0 0 8px var(--red);
}
@keyframes pulse-green {
    0%, 100% { box-shadow: 0 0 8px var(--green), 0 0 20px rgba(34,197,94,0.2); }
    50% { box-shadow: 0 0 12px var(--green), 0 0 30px rgba(34,197,94,0.3); }
}

/* === CHAT === */
.q-msg {
    background: linear-gradient(135deg, var(--accent1), var(--accent2));
    border-radius: 18px 18px 4px 18px;
    padding: 0.9rem 1.3rem;
    margin: 0.5rem 0;
    margin-left: 22%;
    color: white;
    font-size: 0.92rem;
    font-weight: 500;
    box-shadow: 0 4px 20px rgba(59,130,246,0.2);
}
.a-msg {
    background: var(--glass);
    backdrop-filter: blur(10px);
    border: 1px solid var(--card-border);
    border-radius: 18px 18px 18px 4px;
    padding: 1.1rem 1.4rem;
    margin: 0.5rem 0;
    margin-right: 8%;
    font-size: 0.92rem;
    line-height: 1.75;
}

/* === TAGS === */
.tag {
    display: inline-block;
    background: linear-gradient(135deg, rgba(59,130,246,0.12), rgba(139,92,246,0.08));
    border: 1px solid rgba(59,130,246,0.18);
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.72rem;
    font-weight: 500;
    color: var(--accent3);
    margin: 2px 4px 2px 0;
    letter-spacing: 0.02em;
}

/* === CHUNK CARDS === */
.chunk {
    background: rgba(15, 20, 35, 0.5);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent1);
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
    margin: 0.4rem 0;
    font-size: 0.82rem;
    transition: border-color 0.2s;
}
.chunk:hover { border-left-color: var(--accent2); }
.chunk-head {
    display: flex;
    justify-content: space-between;
    margin-bottom: 0.35rem;
    font-size: 0.72rem;
    color: var(--text2);
    font-weight: 500;
}
.chunk-sc {
    color: var(--green);
    font-weight: 700;
    font-size: 0.7rem;
}

/* === UPLOAD === */
.upload-box {
    border: 2px dashed rgba(59,130,246,0.25);
    border-radius: 16px;
    padding: 2.5rem 2rem;
    text-align: center;
    background: var(--grad-subtle);
    margin-bottom: 1rem;
    transition: all 0.3s;
}
.upload-box:hover {
    border-color: rgba(59,130,246,0.5);
    background: linear-gradient(135deg, rgba(59,130,246,0.12), rgba(139,92,246,0.08));
}
.upload-icon { font-size: 2.5rem; color: var(--accent1); margin-bottom: 0.5rem; }
.upload-title { font-size: 1.05rem; color: var(--text); font-weight: 600; }
.upload-sub { font-size: 0.8rem; color: var(--text2); margin-top: 0.3rem; }

/* === BUTTONS === */
.stButton > button {
    background: var(--grad) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 0.6rem 1.4rem !important;
    letter-spacing: 0.01em !important;
    box-shadow: 0 4px 15px rgba(59,130,246,0.15) !important;
    transition: all 0.25s ease !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 25px rgba(59,130,246,0.25) !important;
}
.stProgress > div > div {
    background: var(--grad) !important;
}

/* === TABS === */
.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important;
    padding: 8px 20px !important;
    background: var(--glass) !important;
    border: 1px solid var(--border) !important;
    color: var(--text2) !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
    backdrop-filter: blur(8px) !important;
}
.stTabs [aria-selected="true"] {
    background: var(--grad) !important;
    color: white !important;
    border-color: transparent !important;
    box-shadow: 0 2px 12px rgba(59,130,246,0.2) !important;
}

/* === CONFIG TABLE === */
.cfg {
    display: flex;
    justify-content: space-between;
    padding: 0.45rem 0;
    border-bottom: 1px solid var(--border);
    font-size: 0.85rem;
}
.cfg-k { color: var(--text2); font-weight: 400; }
.cfg-v { color: var(--text); font-weight: 600; }

::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: rgba(148,163,184,0.15); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(148,163,184,0.25); }
</style>
""", unsafe_allow_html=True)

# Session state
for k, v in {"messages": [], "pipeline": None, "last_error": None}.items():
    if k not in st.session_state:
        st.session_state[k] = v


def get_pipeline() -> RAGPipeline:
    if st.session_state.pipeline is None:
        st.session_state.pipeline = RAGPipeline(lazy_init=True)
    return st.session_state.pipeline


def chunk_card(chunk: dict, idx: int):
    sc = chunk.get("rerank_score", chunk.get("score", 0))
    mod = {"text": "Text", "table": "Table", "form": "Form"}.get(chunk.get("modality","text"), "Text")
    txt = chunk["text"][:400] + ("..." if len(chunk["text"]) > 400 else "")
    st.markdown(f"""<div class="chunk">
        <div class="chunk-head">
            <span>[{mod}] {chunk.get('source','')}, Chunk {chunk.get('chunk_index',idx)}</span>
            <span class="chunk-sc">{sc:.4f}</span>
        </div>
        <div style="color:var(--text2);line-height:1.5;">{txt}</div>
    </div>""", unsafe_allow_html=True)


def sidebar():
    with st.sidebar:
        st.markdown("""<div style="padding:0.8rem 0 0.3rem;">
            <h3 style="margin:0;font-size:1.15rem;font-weight:700;
                background:var(--grad);-webkit-background-clip:text;
                -webkit-text-fill-color:transparent;">DocIntel RAG</h3>
            <p style="color:var(--muted);font-size:0.72rem;margin:0.15rem 0 0;
                letter-spacing:0.05em;text-transform:uppercase;">Document Intelligence</p>
        </div>""", unsafe_allow_html=True)
        st.markdown('<div style="height:2px;background:var(--grad);border-radius:1px;margin:0.5rem 0 1rem;opacity:0.4;"></div>', unsafe_allow_html=True)

        try:
            pipe = get_pipeline()
            health = pipe.os_client.health_check()
            ok = health.get("status") in ("green", "yellow")
            dot = "dot-on" if ok else "dot-off"
            lbl = "Connected" if ok else "Offline"
            st.markdown(f'<span class="dot {dot}"></span> <span style="font-size:0.88rem;">OpenSearch: {lbl}</span>', unsafe_allow_html=True)
            total = pipe.os_client.get_doc_count()
            sources = pipe.os_client.get_sources()
            st.caption(f"{total} chunks | {len(sources)} documents")
        except Exception:
            st.markdown('<span class="dot dot-off"></span> <span style="font-size:0.88rem;">OpenSearch: Offline</span>', unsafe_allow_html=True)
            sources = []

        st.markdown('<div style="height:1px;background:var(--border);margin:1rem 0;"></div>', unsafe_allow_html=True)
        st.markdown('<p style="font-size:0.78rem;font-weight:600;color:var(--text2);letter-spacing:0.05em;text-transform:uppercase;">Indexed Documents</p>', unsafe_allow_html=True)
        if sources:
            for src in sources:
                c1, c2 = st.columns([4, 1])
                with c1:
                    st.markdown(f'<span class="tag">{src}</span>', unsafe_allow_html=True)
                with c2:
                    if st.button("X", key=f"d_{src}"):
                        pipe.delete_source(src)
                        st.rerun()
        else:
            st.caption("No documents indexed")

        st.markdown('<div style="height:1px;background:var(--border);margin:1rem 0;"></div>', unsafe_allow_html=True)
        st.session_state["top_k"] = st.slider("Retrieval depth", 3, 20, 10)
        st.session_state["rerank_k"] = st.slider("Results for answer", 2, 10, 5)


def upload_tab():
    st.markdown("""<div class="upload-box">
        <div class="upload-icon">&#9650;</div>
        <div class="upload-title">Drop files here to upload</div>
        <div class="upload-sub">PDF, PNG, JPG, TIFF supported</div>
    </div>""", unsafe_allow_html=True)

    files = st.file_uploader("Browse files", type=["pdf","png","jpg","jpeg","tiff","tif"],
                              accept_multiple_files=True, label_visibility="collapsed")

    if files and st.button("Process and Index", use_container_width=True):
        pipe = get_pipeline()
        prog = st.progress(0)
        out = st.container()
        for i, f in enumerate(files):
            prog.progress(i / len(files), text=f"Processing {f.name}...")
            try:
                r = pipe.ingest(f.name, f.read())
                with out:
                    st.success(f"**{f.name}**: {r['num_chunks']} chunks, "
                               f"{r['num_text_blocks']} text blocks, {r['num_tables']} tables")
            except Exception as e:
                with out:
                    st.error(f"**{f.name}**: {e}")
                st.session_state.last_error = str(e)
            prog.progress((i + 1) / len(files))
        prog.progress(1.0, text="Complete")

    if st.session_state.last_error:
        with st.expander("Error details"):
            st.code(st.session_state.last_error)
            if st.button("Clear"):
                st.session_state.last_error = None
                st.rerun()


def chat_tab():
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f'<div class="q-msg">{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="a-msg">{msg["content"]}</div>', unsafe_allow_html=True)
            if msg.get("sources"):
                st.markdown(" ".join(f'<span class="tag">{s}</span>' for s in msg["sources"]),
                            unsafe_allow_html=True)
            if msg.get("chunks"):
                st.markdown(f'<p style="font-size:0.82rem;font-weight:600;color:var(--text2);margin-top:0.8rem;">Retrieved Chunks ({len(msg["chunks"])})</p>', unsafe_allow_html=True)
                for ci, c in enumerate(msg["chunks"]):
                    chunk_card(c, ci)

    question = st.chat_input("Ask a question about your documents...")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        st.markdown(f'<div class="q-msg">{question}</div>', unsafe_allow_html=True)

        with st.spinner("Searching and generating answer..."):
            try:
                result = get_pipeline().query(
                    question=question,
                    top_k=st.session_state.get("top_k", 10),
                    rerank_top_k=st.session_state.get("rerank_k", 5),
                    search_mode="hybrid")

                answer = result.get("answer", "")
                sources = result.get("sources", [])
                chunks = result.get("retrieved_chunks", [])
                for c in chunks:
                    c.pop("embedding", None)

                st.markdown(f'<div class="a-msg">{answer}</div>', unsafe_allow_html=True)
                if sources:
                    st.markdown(" ".join(f'<span class="tag">{s}</span>' for s in sources),
                                unsafe_allow_html=True)
                usage = result.get("usage", {})
                if usage:
                    st.caption(f"{usage.get('total_tokens',0)} tokens | Model: {result.get('model','')}")

                st.markdown(f'<p style="font-size:0.82rem;font-weight:600;color:var(--text2);margin-top:0.8rem;">Retrieved Chunks ({len(chunks)})</p>', unsafe_allow_html=True)
                for ci, c in enumerate(chunks):
                    chunk_card(c, ci)

                st.session_state.messages.append({
                    "role": "assistant", "content": answer,
                    "sources": sources, "chunks": chunks})
            except Exception as e:
                st.error(str(e))
                st.session_state.messages.append({
                    "role": "assistant", "content": f"Error: {e}",
                    "sources": [], "chunks": []})


def dashboard_tab():
    pipe = get_pipeline()
    try:
        status = pipe.get_status()
    except Exception as e:
        st.error(str(e))
        return

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="gcard"><div class="stat-num">{status.get("total_chunks",0)}</div>'
                     '<div class="stat-label">Total Chunks</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="gcard"><div class="stat-num">{len(status.get("indexed_sources",[]))}</div>'
                     '<div class="stat-label">Documents</div></div>', unsafe_allow_html=True)
    with c3:
        ok = status.get("opensearch", {}).get("status") in ("green", "yellow")
        d = "dot-on" if ok else "dot-off"
        l = "Connected" if ok else "Offline"
        st.markdown(f'<div class="gcard"><div class="stat-num" style="font-size:1.3rem;"><span class="dot {d}"></span>{l}</div>'
                     '<div class="stat-label">OpenSearch</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="gcard"><div class="stat-num" style="font-size:1.3rem;">Llama 3.3</div>'
                     '<div class="stat-label">LLM via Groq</div></div>', unsafe_allow_html=True)

    st.markdown("")
    left, right = st.columns(2)
    with left:
        st.markdown('<p style="font-size:0.78rem;font-weight:600;color:var(--text2);letter-spacing:0.05em;text-transform:uppercase;margin-bottom:0.5rem;">Configuration</p>', unsafe_allow_html=True)
        for k, v in [("Embedding Model", status.get("embedding_model","")),
                      ("Vector Dimension", status.get("embedding_dimension","")),
                      ("Chunk Size", settings.CHUNK_SIZE),
                      ("Chunk Overlap", settings.CHUNK_OVERLAP),
                      ("LLM Model", status.get("llm_model",""))]:
            st.markdown(f'<div class="cfg"><span class="cfg-k">{k}</span><span class="cfg-v">{v}</span></div>', unsafe_allow_html=True)
    with right:
        st.markdown('<p style="font-size:0.78rem;font-weight:600;color:var(--text2);letter-spacing:0.05em;text-transform:uppercase;margin-bottom:0.5rem;">Indexed Files</p>', unsafe_allow_html=True)
        for s in status.get("indexed_sources", []):
            st.markdown(f'<span class="tag" style="display:block;margin:4px 0;">{s}</span>', unsafe_allow_html=True)
        if not status.get("indexed_sources"):
            st.caption("No documents indexed")

    st.markdown("")
    a, b, c = st.columns(3)
    with a:
        if st.button("Refresh", use_container_width=True): st.rerun()
    with b:
        if st.button("Reset Index", use_container_width=True):
            pipe.reset_index(); st.session_state.messages = []; st.rerun()
    with c:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []; st.rerun()


def main():
    sidebar()
    st.markdown("""<div class="hero">
        <h1>DocIntel RAG</h1>
        <div class="hero-line"></div>
        <p>Multimodal Document Intelligence System</p>
    </div>""", unsafe_allow_html=True)

    t1, t2, t3 = st.tabs(["Upload", "Chat", "Dashboard"])
    with t1: upload_tab()
    with t2: chat_tab()
    with t3: dashboard_tab()


if __name__ == "__main__":
    main()
