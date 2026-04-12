"""
Streamlit frontend for the Quantum RAG Assistant.
A clean, interactive UI for students to ask questions.
"""

import time

import requests
import streamlit as st

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="⚛️ Quantum RAG Assistant",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Service Startup (Streamlit Cloud Hack) ──────────────────────────────────
import socket
import subprocess
import sys

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

if not is_port_in_use(8000):
    print("starting uvicorn backend...")
    
    # Safely inject Streamlit Secrets into the background FastApi process!
    env = os.environ.copy()
    try:
        import streamlit as st
        
        print("====== SECRET DIAGNOSTICS ======")
        print("Available Streamlit secrets:", list(st.secrets.keys()))
        print("================================")
        
        # Check explicit formats
        if "GROQ_API_KEY" in st.secrets:
            env["GROQ_API_KEY"] = str(st.secrets["GROQ_API_KEY"])
            print("Successfully injected GROQ_API_KEY from st.secrets!")
        elif "groq_api_key" in st.secrets:
            env["GROQ_API_KEY"] = str(st.secrets["groq_api_key"])
            print("Successfully injected groq_api_key from st.secrets!")
            
    except Exception as e:
        print(f"Secret injection warning: {e}")

    subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--port", "8000"], 
        env=env
    )
    time.sleep(2)  # Give the server a moment to spin up

API_BASE = "http://localhost:8000/api/v1"

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .source-badge {
        background: #1e3a5f;
        color: #7ec8e3;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.78rem;
        margin-right: 6px;
        display: inline-block;
    }
    .metric-card {
        background: #1e1e2e;
        border: 1px solid #3d3d5c;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
    }
    .chat-bubble-user {
        background: #2d3748;
        padding: 12px 16px;
        border-radius: 12px 12px 4px 12px;
        margin: 8px 0;
    }
    .chat-bubble-ai {
        background: #1a202c;
        border-left: 3px solid #667eea;
        padding: 12px 16px;
        border-radius: 4px 12px 12px 12px;
        margin: 8px 0;
    }
</style>
""", unsafe_allow_html=True)


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚛️ Configuration")

    top_k = st.slider(
        "Retrieved Chunks (top_k)",
        min_value=1, max_value=10, value=5,
        help="How many document chunks to retrieve per query"
    )
    threshold = st.slider(
        "Similarity Threshold",
        min_value=0.0, max_value=1.0, value=0.3, step=0.05,
        help="Minimum relevance score for retrieved chunks"
    )

    st.divider()
    st.markdown("### 💡 Example Questions")
    example_questions = [
        "What is quantum superposition?",
        "Explain quantum entanglement",
        "How does a quantum gate work?",
        "What is Shor's algorithm?",
        "What is decoherence in quantum computing?",
        "Explain the Bloch sphere",
        "What are qubits?",
    ]
    for q in example_questions:
        if st.button(q, use_container_width=True, key=f"btn_{q[:20]}"):
            st.session_state["prefill_question"] = q

    st.divider()
    # API Health Check
    try:
        r = requests.get(f"{API_BASE}/health", timeout=2)
        if r.status_code == 200:
            data = r.json()
            st.success(f"✅ API Online")
            st.caption(f"Model: `{data.get('model', 'N/A')}`")
        else:
            st.error("❌ API Offline")
    except Exception:
        st.error("❌ API Offline — Start with:\n`uvicorn app.main:app --reload`")

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()


# ─── Main UI ─────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">⚛️ Quantum RAG Assistant</div>', unsafe_allow_html=True)
st.caption("AI-powered tutor grounded in trusted Quantum Computing documents. No hallucinations — only document-backed answers.")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display chat history
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            st.markdown(
                " ".join(f'<span class="source-badge">📄 {s}</span>' for s in msg["sources"]),
                unsafe_allow_html=True,
            )
        if msg.get("meta"):
            m = msg["meta"]
            cols = st.columns(3)
            cols[0].metric("⏱️ Time", f"{m.get('processing_time_ms', 0):.0f}ms")
            cols[1].metric("🎯 Chunks", m.get("chunks_retrieved", 0))
            cols[2].metric("🤖 Model", m.get("model", "N/A").split("-")[0])

# Handle prefilled question from sidebar
prefill = st.session_state.pop("prefill_question", None)

# Chat input
user_input = st.chat_input("Ask about Quantum Computing...") or prefill

if user_input:
    # Display user message
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Query API
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching knowledge base..."):
            try:
                response = requests.post(
                    f"{API_BASE}/chat",
                    json={
                        "question": user_input,
                        "top_k": top_k,
                        "threshold": threshold,
                    },
                    timeout=30,
                )

                if response.status_code == 200:
                    data = response.json()
                    answer = data["answer"]
                    sources = data.get("sources", [])
                    meta = {
                        "processing_time_ms": data.get("processing_time_ms", 0),
                        "chunks_retrieved": len(data.get("retrieval_info", [])),
                        "model": data.get("model_used", "N/A"),
                    }

                    st.markdown(answer)

                    if sources:
                        st.markdown(
                            " ".join(f'<span class="source-badge">📄 {s}</span>' for s in sources),
                            unsafe_allow_html=True,
                        )

                    cols = st.columns(3)
                    cols[0].metric("⏱️ Time", f"{meta['processing_time_ms']:.0f}ms")
                    cols[1].metric("🎯 Chunks", meta["chunks_retrieved"])
                    cols[2].metric("🤖 Model", meta["model"].split("-")[0])

                    # Show retrieved chunks in expander
                    retrieval_info = data.get("retrieval_info", [])
                    if retrieval_info:
                        with st.expander("🔍 View Retrieved Context"):
                            for i, info in enumerate(retrieval_info, 1):
                                st.markdown(
                                    f"**Chunk {i}** — `{info['source']}` "
                                    f"(score: `{info['score']:.3f}`)"
                                )

                    st.session_state["messages"].append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "meta": meta,
                    })

                elif response.status_code == 503:
                    error_msg = "⚠️ Knowledge base not ready. Please build the index first:\n```\npython scripts/build_index.py\n```"
                    st.warning(error_msg)
                    st.session_state["messages"].append({"role": "assistant", "content": error_msg})
                else:
                    error_msg = f"❌ API Error {response.status_code}: {response.json().get('detail', 'Unknown error')}"
                    st.error(error_msg)
                    st.session_state["messages"].append({"role": "assistant", "content": error_msg})

            except requests.exceptions.ConnectionError:
                error_msg = "❌ Cannot connect to API. Start it with:\n```bash\nuvicorn app.main:app --reload\n```"
                st.error(error_msg)
            except Exception as e:
                st.error(f"❌ Unexpected error: {e}")