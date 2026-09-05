import os
import tempfile
import time
from pathlib import Path

import streamlit as st

from app.rag import RAG
from app.stt import Transcriber


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="VoiceRAG AI | Intelligent Knowledge Assistant",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ---------- GLOBAL ---------- */

    .stApp {
        background:
            radial-gradient(
                circle at 10% 10%,
                rgba(124, 58, 237, 0.10),
                transparent 28%
            ),
            radial-gradient(
                circle at 90% 20%,
                rgba(59, 130, 246, 0.08),
                transparent 25%
            ),
            #f8fafc;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1450px;
    }

    /* ---------- HERO ---------- */

    .hero {
        padding: 30px 35px;
        border-radius: 24px;
        background:
            linear-gradient(
                135deg,
                #111827 0%,
                #312e81 48%,
                #4c1d95 100%
            );
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 18px 45px rgba(49, 46, 129, 0.22);
    }

    .hero-title {
        font-size: 44px;
        font-weight: 800;
        margin-bottom: 5px;
        letter-spacing: -1px;
    }

    .hero-subtitle {
        font-size: 17px;
        opacity: 0.90;
        margin-bottom: 18px;
    }

    .badge {
        display: inline-block;
        padding: 7px 14px;
        border-radius: 999px;
        background: rgba(255,255,255,0.14);
        border: 1px solid rgba(255,255,255,0.20);
        margin-right: 8px;
        font-size: 13px;
    }

    /* ---------- CARDS ---------- */

    .card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 18px;
        padding: 20px;
        margin-bottom: 18px;
        box-shadow: 0 8px 25px rgba(15, 23, 42, 0.05);
    }

    .card-title {
        font-size: 19px;
        font-weight: 750;
        margin-bottom: 7px;
        color: #111827;
    }

    .card-text {
        color: #64748b;
        font-size: 14px;
    }

    /* ---------- METRICS ---------- */

    .metric-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 18px;
        text-align: center;
        box-shadow: 0 5px 20px rgba(15,23,42,0.04);
    }

    .metric-number {
        font-size: 28px;
        font-weight: 800;
        color: #312e81;
    }

    .metric-label {
        color: #64748b;
        font-size: 13px;
    }

    /* ---------- PIPELINE ---------- */

    .pipeline {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 8px;
        overflow-x: auto;
        padding: 15px 5px;
    }

    .pipeline-step {
        min-width: 105px;
        text-align: center;
        padding: 12px 8px;
        border-radius: 14px;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
    }

    .pipeline-icon {
        font-size: 25px;
    }

    .pipeline-label {
        font-size: 11px;
        font-weight: 650;
        color: #475569;
    }

    .arrow {
        font-size: 18px;
        color: #94a3b8;
    }

    /* ---------- ANSWER ---------- */

    .answer-box {
        background: linear-gradient(
            135deg,
            #eef2ff,
            #f5f3ff
        );
        border-left: 5px solid #6366f1;
        border-radius: 15px;
        padding: 22px;
        color: #1e1b4b;
        line-height: 1.7;
        font-size: 16px;
    }

    .safe-box {
        background: #fff7ed;
        border-left: 5px solid #f97316;
        border-radius: 15px;
        padding: 20px;
        color: #7c2d12;
    }

    /* ---------- SOURCE ---------- */

    .source-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 13px;
        margin-bottom: 10px;
    }

    /* ---------- STATUS ---------- */

    .status-online {
        color: #15803d;
        font-weight: 700;
    }

    .status-waiting {
        color: #ca8a04;
        font-weight: 700;
    }

    /* ---------- FOOTER ---------- */

    .footer {
        text-align: center;
        color: #64748b;
        padding: 25px;
        font-size: 13px;
    }

    /* ---------- BUTTONS ---------- */

    .stButton > button {
        border-radius: 12px;
        font-weight: 650;
        min-height: 45px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

defaults = {
    "rag": None,
    "indexed_file": None,
    "indexed_documents": 0,
    "indexed_chunks": 0,
    "answer": None,
    "question": "",
    "transcript": "",
    "transcript_confidence": 0.0,
    "history": [],
    "last_audio": None,
    "processing_time": 0.0,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def reset_answer():
    st.session_state.answer = None


def clear_all():
    st.session_state.answer = None
    st.session_state.question = ""
    st.session_state.transcript = ""
    st.session_state.transcript_confidence = 0.0
    st.session_state.history = []


def save_uploaded_file(uploaded_file):
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    file_path = data_dir / uploaded_file.name

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return str(file_path)


def transcribe_audio(audio_bytes):
    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".wav"
        ) as tmp:

            tmp.write(audio_bytes)
            temp_path = tmp.name

        transcriber = Transcriber()

        transcript, confidence = (
            transcriber.transcribe_file(temp_path)
        )

        return transcript.strip(), float(confidence)

    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


def render_status(label, active=True):
    if active:
        st.markdown(
            f'<span class="status-online">🟢 {label}</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<span class="status-waiting">🟡 {label}</span>',
            unsafe_allow_html=True,
        )


# ============================================================
# HERO HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">

        <div class="hero-title">
            🎙️ VoiceRAG AI
        </div>

        <div class="hero-subtitle">
            Talk to your documents — powered by Voice,
            Retrieval-Augmented Generation & Grounded AI.
        </div>

        <span class="badge">🎤 Voice AI</span>
        <span class="badge">🧠 RAG</span>
        <span class="badge">🔎 Semantic Search</span>
        <span class="badge">🛡️ Anti-Hallucination</span>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## ⚙️ Control Center")

    st.divider()

    # Language
    language = st.selectbox(
        "🌐 Response Language",
        [
            "English",
            "Hindi",
            "Hinglish",
        ],
    )

    st.caption(
        "Language preference is used as a UI preference. "
        "The current Whisper backend is configured for English."
    )

    st.divider()

    # Upload
    st.markdown("### 📚 Knowledge Base")

    uploaded_file = st.file_uploader(
        "Upload a document",
        type=["pdf", "txt", "docx"],
        help="Upload PDF, TXT or DOCX files.",
    )

    if uploaded_file:

        st.success(
            f"📄 {uploaded_file.name}"
        )

        size_kb = uploaded_file.size / 1024

        st.caption(
            f"File size: {size_kb:.1f} KB"
        )

        if st.button(
            "🧠 Index Knowledge Base",
            use_container_width=True,
            type="primary",
        ):

            start = time.perf_counter()

            with st.spinner(
                "🔄 Processing document → chunking → embeddings → indexing..."
            ):

                try:

                    save_uploaded_file(uploaded_file)

                    rag = RAG()

                    result = rag.ingest(
                        directory="data",
                        strategy="all",
                    )

                    st.session_state.rag = rag
                    st.session_state.indexed_file = uploaded_file.name

                    st.session_state.indexed_documents = result.get(
                        "documents",
                        0,
                    )

                    st.session_state.indexed_chunks = result.get(
                        "chunks",
                        0,
                    )

                    st.session_state.answer = None

                    elapsed = time.perf_counter() - start

                    st.success(
                        "✅ Knowledge base indexed!"
                    )

                    st.toast(
                        "Document indexed successfully!",
                        icon="🚀",
                    )

                    st.info(
                        f"""
                        **Documents:** {result.get("documents", 0)}

                        **Chunks:** {result.get("chunks", 0)}

                        **Strategy:** {result.get("strategy", "all")}

                        **Backend:** {result.get("backend", "FAISS")}

                        **Time:** {elapsed:.2f}s
                        """
                    )

                except Exception as e:

                    st.error(
                        "❌ Indexing failed"
                    )

                    st.code(
                        str(e),
                        language="text",
                    )

    st.divider()

    # Demo questions
    st.markdown("### ⚡ Demo Questions")

    demo_questions = [
        "What is VoiceRAG AI?",
        "What problem does VoiceRAG solve?",
        "What are the main features?",
        "How does speech become an answer?",
        "Why is chunking important?",
        "How does VoiceRAG reduce hallucination?",
    ]

    for demo in demo_questions:

        if st.button(
            demo,
            use_container_width=True,
            key=f"demo_{demo}",
        ):

            st.session_state.question = demo
            st.session_state.answer = None
            st.rerun()

    st.divider()

    if st.button(
        "🗑️ Clear Conversation",
        use_container_width=True,
    ):
        clear_all()
        st.rerun()


# ============================================================
# TOP METRICS
# ============================================================

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-number">🎤</div>
            <div class="metric-label">Voice Interface</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-number">
                {st.session_state.indexed_documents}
            </div>
            <div class="metric-label">Documents Indexed</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-number">
                {st.session_state.indexed_chunks}
            </div>
            <div class="metric-label">Knowledge Chunks</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m4:
    status = (
        "READY"
        if st.session_state.rag
        else "WAITING"
    )

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-number">
                {"🟢" if status == "READY" else "🟡"}
            </div>
            <div class="metric-label">
                RAG {status}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.write("")


# ============================================================
# MAIN COLUMNS
# ============================================================

left, right = st.columns(
    [2.2, 1],
    gap="large",
)


# ============================================================
# LEFT: VOICE + QUESTION
# ============================================================

with left:

    st.markdown(
        """
        <div class="card">
            <div class="card-title">
                🎤 Ask Your Knowledge Base
            </div>
            <div class="card-text">
                Speak naturally or type your question.
                VoiceRAG retrieves relevant evidence before generating an answer.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # VOICE INPUT
    # --------------------------------------------------------

    st.markdown("### 🎙️ Voice Input")

    audio = st.audio_input(
        "Click the microphone and speak",
        key="voice_recording",
    )

    if audio is not None:

        st.session_state.last_audio = audio

        st.audio(
            audio,
            format="audio/wav",
        )

        # Avoid processing identical recording repeatedly
        audio_hash = hash(audio.getvalue())

        if st.session_state.get("processed_audio_hash") != audio_hash:

            st.session_state.processed_audio_hash = audio_hash

            with st.spinner(
                "🎧 Whisper is converting speech into text..."
            ):

                try:

                    transcript, confidence = transcribe_audio(
                        audio.getvalue()
                    )

                    if transcript:

                        st.session_state.transcript = transcript
                        st.session_state.question = transcript
                        st.session_state.transcript_confidence = confidence

                        st.success(
                            "✅ Speech converted to text!"
                        )

                    else:

                        st.warning(
                            "⚠️ No speech detected. "
                            "Please try speaking again."
                        )

                except Exception as e:

                    st.error(
                        "❌ Speech-to-text failed."
                    )

                    st.code(
                        str(e),
                        language="text",
                    )

    # --------------------------------------------------------
    # TRANSCRIPT
    # --------------------------------------------------------

    if st.session_state.transcript:

        st.markdown("### 📝 Transcribed Question")

        st.info(
            st.session_state.transcript
        )

        confidence = (
            st.session_state.transcript_confidence * 100
        )

        st.progress(
            min(max(
                st.session_state.transcript_confidence,
                0.0
            ), 1.0)
        )

        st.caption(
            f"🎯 Speech confidence: {confidence:.1f}%"
        )

    # --------------------------------------------------------
    # TEXT QUESTION
    # --------------------------------------------------------

    st.markdown("### 💬 Question")

    question = st.text_area(
        "Ask something about your document",
        value=st.session_state.question,
        height=110,
        placeholder=(
            "Example: What are the main features of VoiceRAG AI?"
        ),
        label_visibility="collapsed",
    )

    st.session_state.question = question

    # --------------------------------------------------------
    # ANSWER BUTTON
    # --------------------------------------------------------

    b1, b2 = st.columns(
        [3, 1]
    )

    with b1:

        ask_clicked = st.button(
            "🤖 Get Grounded AI Answer",
            use_container_width=True,
            type="primary",
        )

    with b2:

        clear_clicked = st.button(
            "↺ Clear",
            use_container_width=True,
        )

    if clear_clicked:

        st.session_state.question = ""
        st.session_state.transcript = ""
        st.session_state.answer = None

        st.rerun()

    # --------------------------------------------------------
    # RAG QUERY
    # --------------------------------------------------------

    if ask_clicked:

        if not question.strip():

            st.warning(
                "⚠️ Please type or speak a question."
            )

        elif st.session_state.rag is None:

            st.warning(
                "⚠️ Please upload and index a document first."
            )

        else:

            start = time.perf_counter()

            with st.spinner(
                "🔎 Searching knowledge base and generating grounded answer..."
            ):

                try:

                    result = st.session_state.rag.ask(
                        question.strip(),
                        transcript=st.session_state.transcript,
                    )

                    elapsed = (
                        time.perf_counter()
                        - start
                    )

                    st.session_state.answer = result
                    st.session_state.processing_time = elapsed

                    # History
                    st.session_state.history.append(
                        {
                            "question": question.strip(),
                            "answer": result.answer,
                            "confidence": result.confidence,
                        }
                    )

                except Exception as e:

                    st.error(
                        "❌ RAG processing failed."
                    )

                    st.code(
                        str(e),
                        language="text",
                    )

    # ========================================================
    # ANSWER DISPLAY
    # ========================================================

    if st.session_state.answer:

        result = st.session_state.answer

        st.divider()

        st.markdown(
            "## 🤖 AI Answer"
        )

        if getattr(result, "should_answer", False) and result.answer:

            st.markdown(
                f"""
                <div class="answer-box">
                    {result.answer}
                </div>
                """,
                unsafe_allow_html=True,
            )

        else:

            st.markdown(
                """
                <div class="safe-box">
                    <strong>🛡️ Grounding Guardrail Activated</strong>
                    <br><br>
                    I don't have enough information in the
                    provided documents to answer this reliably.
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.write("")

        # ----------------------------------------------------
        # ANSWER METRICS
        # ----------------------------------------------------

        a1, a2, a3 = st.columns(3)

        confidence = safe_float(
            getattr(result, "confidence", 0)
        )

        with a1:

            st.metric(
                "🎯 Retrieval Confidence",
                f"{confidence * 100:.1f}%",
            )

        with a2:

            total_ms = safe_float(
                getattr(
                    result,
                    "latency_ms",
                    {}
                ).get(
                    "total_ms",
                    st.session_state.processing_time * 1000,
                )
            )

            st.metric(
                "⚡ Response Time",
                f"{total_ms:.0f} ms",
            )

        with a3:

            source_count = len(
                getattr(
                    result,
                    "sources",
                    []
                )
            )

            st.metric(
                "📚 Sources",
                source_count,
            )

        # ----------------------------------------------------
        # LATENCY BREAKDOWN
        # ----------------------------------------------------

        latency = getattr(
            result,
            "latency_ms",
            {}
        )

        if latency:

            with st.expander(
                "⚡ Performance Breakdown"
            ):

                l1, l2, l3 = st.columns(3)

                with l1:
                    st.metric(
                        "Retrieval",
                        f"{safe_float(latency.get('retrieval_ms')):.0f} ms",
                    )

                with l2:
                    st.metric(
                        "Generation",
                        f"{safe_float(latency.get('generation_ms')):.0f} ms",
                    )

                with l3:
                    st.metric(
                        "Total",
                        f"{safe_float(latency.get('total_ms')):.0f} ms",
                    )

        # ----------------------------------------------------
        # SOURCES
        # ----------------------------------------------------

        sources = getattr(
            result,
            "sources",
            []
        )

        if sources:

            st.markdown(
                "### 📚 Retrieved Sources"
            )

            for i, source in enumerate(
                sources,
                start=1,
            ):

                source_name = source.get(
                    "source",
                    "Unknown document",
                )

                score = safe_float(
                    source.get(
                        "score",
                        0,
                    )
                )

                strategy = source.get(
                    "strategy",
                    "unknown",
                )

                chunk_id = source.get(
                    "chunk_id",
                    "N/A",
                )

                st.markdown(
                    f"""
                    <div class="source-box">

                    <strong>#{i} — 📄 {source_name}</strong>

                    <br>

                    <small>
                    Similarity: {score:.3f}
                    &nbsp; | &nbsp;
                    Strategy: {strategy}
                    &nbsp; | &nbsp;
                    Chunk: {chunk_id}
                    </small>

                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # ----------------------------------------------------
        # REFUSAL REASON
        # ----------------------------------------------------

        refusal = getattr(
            result,
            "refusal_reason",
            None,
        )

        if refusal:

            with st.expander(
                "🛡️ Guardrail Information"
            ):

                st.write(
                    refusal
                )


# ============================================================
# RIGHT: SYSTEM DASHBOARD
# ============================================================

with right:

    st.markdown(
        """
        <div class="card">
            <div class="card-title">
                📊 System Dashboard
            </div>
            <div class="card-text">
                Live status of the VoiceRAG pipeline.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Status
    st.markdown("### 🔌 System Status")

    render_status(
        "Streamlit Interface",
        True,
    )

    render_status(
        "Voice Input",
        audio is not None or bool(
            st.session_state.transcript
        ),
    )

    render_status(
        "Speech-to-Text",
        bool(
            st.session_state.transcript
        ),
    )

    render_status(
        "Knowledge Base",
        st.session_state.rag is not None,
    )

    render_status(
        "RAG Retrieval",
        st.session_state.rag is not None,
    )

    render_status(
        "AI Answer",
        st.session_state.answer is not None,
    )

    st.divider()

    # Current document
    st.markdown("### 📄 Current Knowledge Base")

    if st.session_state.indexed_file:

        st.success(
            st.session_state.indexed_file
        )

        st.caption(
            f"{st.session_state.indexed_chunks} chunks indexed"
        )

    else:

        st.info(
            "No document indexed."
        )

    st.divider()

    # Pipeline
    st.markdown("### 🚀 AI Pipeline")

    st.markdown(
        """
        <div class="pipeline">

            <div class="pipeline-step">
                <div class="pipeline-icon">🎤</div>
                <div class="pipeline-label">Voice</div>
            </div>

            <div class="arrow">→</div>

            <div class="pipeline-step">
                <div class="pipeline-icon">📝</div>
                <div class="pipeline-label">Whisper</div>
            </div>

            <div class="arrow">→</div>

            <div class="pipeline-step">
                <div class="pipeline-icon">🧩</div>
                <div class="pipeline-label">Chunks</div>
            </div>

            <div class="arrow">→</div>

            <div class="pipeline-step">
                <div class="pipeline-icon">🧠</div>
                <div class="pipeline-label">Embedding</div>
            </div>

            <div class="arrow">→</div>

            <div class="pipeline-step">
                <div class="pipeline-icon">🔎</div>
                <div class="pipeline-label">Retrieval</div>
            </div>

            <div class="arrow">→</div>

            <div class="pipeline-step">
                <div class="pipeline-icon">🤖</div>
                <div class="pipeline-label">LLM</div>
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # Features
    st.markdown("### 🛡️ Reliability Features")

    features = [
        ("🔎", "Semantic Retrieval"),
        ("🛡️", "Query Guardrails"),
        ("🎯", "Confidence Scoring"),
        ("📚", "Source Attribution"),
        ("🚫", "Hallucination Protection"),
    ]

    for icon, feature in features:

        st.write(
            f"{icon} **{feature}**"
        )


# ============================================================
# CONVERSATION HISTORY
# ============================================================

if st.session_state.history:

    st.divider()

    st.markdown(
        "## 🕘 Conversation History"
    )

    for i, item in enumerate(
        reversed(st.session_state.history[-5:]),
        start=1,
    ):

        with st.expander(
            f"Q{i}: {item['question']}"
        ):

            st.write(
                item["answer"]
            )

            st.caption(
                f"Confidence: "
                f"{item['confidence'] * 100:.1f}%"
            )


# ============================================================
# DEMO MODE
# ============================================================

st.divider()

st.markdown(
    """
    <div class="card">

        <div class="card-title">
            🏆 Hackathon Demo Flow
        </div>

        <div class="card-text">

        <b>1.</b> Upload a trusted PDF/TXT/DOCX document
        <br><br>

        <b>2.</b> Click <b>Index Knowledge Base</b>
        <br><br>

        <b>3.</b> Click the microphone 🎤
        <br><br>

        <b>4.</b> Ask a natural-language question
        <br><br>

        <b>5.</b> Watch Speech → Text → Retrieval → AI Answer
        <br><br>

        <b>6.</b> Show confidence score and source attribution
        <br><br>

        <b>7.</b> Ask something unrelated to the document
        <br><br>

        <b>8.</b> Demonstrate the anti-hallucination guardrail

        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

        🎙️ <b>VoiceRAG AI</b>

        &nbsp;•&nbsp;

        Voice + RAG + Generative AI

        &nbsp;•&nbsp;

        Built for Hackathon Demo

        <br><br>

        <small>
        Trusted documents → Semantic retrieval → Grounded answers
        </small>

    </div>
    """,
    unsafe_allow_html=True,
)